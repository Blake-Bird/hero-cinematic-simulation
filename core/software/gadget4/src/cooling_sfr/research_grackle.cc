/* Native C Local API binding to pinned Grackle 3.4.1. Grackle source/data
 * licenses are retained in configs; this adapter is original Research2 code.
 * Supported: equilibrium (0), atomic (1), H2 (2), and H2/HD (3).
 * SH03 effective-EOS gas does not call this single-phase cooling adapter. */
#include "gadgetconfig.h"
#ifdef RESEARCH_GRACKLE
#ifndef STELLAR_EVOLUTION
#error "RESEARCH_GRACKLE requires the eleven-element STELLAR_EVOLUTION state"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <grackle.h>
#include "../main/simulation.h"
#include "../data/research_stellar_birth.h"
#include "research_stellar_returns.h"
#include "research_primordial_network.h"
#if defined(PRESSURE_ENTROPY_SPH)
#error "Grackle variable molecular EOS requires density-entropy SPH with explicit compression work"
#endif
static_assert(sizeof(gr_float)==8,"Research2 Grackle bindings require a precision-64 library");

namespace {
struct chemistry_kernel {
  chemistry_data chemistry{};chemistry_data_storage rates{};code_units units{};
  uint64_t data_crc=0;
  chemistry_kernel() {
    const grackle_version version=get_grackle_version();
    if(!version.version || strcmp(version.version,"3.4.1"))Terminate("Grackle runtime version differs from pinned 3.4.1");
    uint64_t bytes=0,crc=0,expected=0;
    if(!research_stellar_seed(All.GrackleDataCRC32,expected) || expected>UINT32_MAX ||
        !research_stellar_file_census(All.GrackleDataFile,bytes,crc) || crc!=expected || bytes>512*1024*1024)
      Terminate("Grackle cooling/UV data CRC mismatch or excessive data file");
    data_crc=crc;
    if((All.GracklePrimordialChemistry<0 || All.GracklePrimordialChemistry>3) ||
       (All.GrackleMetalCooling!=0 && All.GrackleMetalCooling!=1) || (All.GrackleUVBackground!=0 && All.GrackleUVBackground!=1))
      Terminate("Unsupported Grackle network or cooling switch");
    if(!std::isfinite(All.GrackleDeuteriumToHydrogenRatio) || All.GrackleDeuteriumToHydrogenRatio<0 ||
       All.GrackleDeuteriumToHydrogenRatio>1e-3 || (All.GracklePrimordialChemistry==3 && All.GrackleDeuteriumToHydrogenRatio==0))
      Terminate("Explicit supported primordial D/H number ratio required");
    if(!local_initialize_chemistry_parameters(&chemistry))Terminate("Grackle parameter initialization failed");
    chemistry.use_grackle=1;chemistry.with_radiative_cooling=1;
    chemistry.primordial_chemistry=All.GracklePrimordialChemistry;
    chemistry.DeuteriumToHydrogenRatio=2*All.GrackleDeuteriumToHydrogenRatio;
    chemistry.metal_cooling=All.GrackleMetalCooling;chemistry.UVbackground=All.GrackleUVBackground;
    chemistry.grackle_data_file=All.GrackleDataFile;
    chemistry.Gamma=GAMMA;
    const double h=All.ComovingIntegrationOn ? All.HubbleParam : 1.;
    units.comoving_coordinates=0; // Every incoming density is proper already.
    units.density_units=All.UnitDensity_in_cgs*h*h;
    units.length_units=All.UnitLength_in_cm/h;units.time_units=All.UnitTime_in_s/h;
    units.a_units=1.;units.a_value=All.ComovingIntegrationOn ? All.Time : 1.;
    set_velocity_units(&units);
    if(fabs(units.velocity_units/All.UnitVelocity_in_cm_per_s-1)>1.e-12)
      Terminate("Grackle and native velocity/specific-energy units disagree");
    if(!local_initialize_chemistry_data(&chemistry,&rates,&units))Terminate("Grackle rate initialization failed");
  }
  ~chemistry_kernel(){local_free_chemistry_data(&chemistry,&rates);}
};
chemistry_kernel& kernel(){static chemistry_kernel value;return value;}
double inverse_cooling_time(chemistry_data &chemistry,chemistry_data_storage &rates,code_units &units,grackle_field_data &fields) {
  gr_float cooling_time=0;
  if(!local_calculate_cooling_time(&chemistry,&rates,&units,&fields,&cooling_time) || std::isnan(double(cooling_time)) || cooling_time==0)
    Terminate("Grackle instantaneous cooling-time diagnostic failed");
  // Grackle's energy/edot convention is negative for net radiative cooling.
  // Store its inverse in physical seconds: exact equilibrium has inverse zero.
  const double value=-1/(double(cooling_time)*units.time_units);
  if(!std::isfinite(value))Terminate("Grackle inverse cooling time is not finite");
  return value;
}
}

void research_grackle_initialize(void){(void)kernel();}

void research_grackle_initialize_particle(simparticles *particles,int index) {
  auto &state=particles->P[index].ResearchSourceState;
  if(state.grackle_state_version)return; // retained restart state is never reseeded
  auto &solver=kernel();
  if(state.chemical_state_version!=1)Terminate("Primordial initialization requires measured elemental composition");
  state.grackle_state_version=research_primordial::version(All.GracklePrimordialChemistry);
  state.chemical_last_update_time=All.Time;state.grackle_data_crc32=solver.data_crc;
  research_grackle_refresh_thermodynamics(particles,index);
  const double proton=state.element_mass_fraction[0]-state.grackle_molecular_mass_fraction[3]-state.grackle_molecular_mass_fraction[4]-2*state.grackle_molecular_mass_fraction[5]/3;
  particles->SphP[index].Ne=proton>0?state.grackle_species_mass_fraction[5]/proton:0;
  particles->SphP[index].set_thermodynamic_variables();
}

double research_grackle_cool_particle(simparticles *particles,int index) {
  auto &p=particles->P[index];auto &state=p.ResearchSourceState;auto &sph=particles->SphP[index];
  auto &solver=kernel();
  if(p.getType()!=0 || state.chemical_state_version!=1)Terminate("Grackle requires live gas with an eleven-element composition");
  double old=std::max<double>(All.MinEgySpec,particles->get_utherm_from_entropy(index));
  const double beginning=state.grackle_state_version ? state.chemical_last_update_time : All.TimeOld;
  if(state.grackle_state_version>4 || beginning>All.Time || beginning<0)Terminate("Grackle state clock is invalid");
  if(state.grackle_state_version && state.grackle_state_version!=research_primordial::version(All.GracklePrimordialChemistry))
    Terminate("Grackle chemistry network changed across stored state");
  if(state.grackle_state_version && state.grackle_data_crc32!=solver.data_crc)Terminate("Grackle source data changed across restart");
  if(beginning==All.Time || All.ComovingIntegrationOn && beginning==0)return old;
  const double years=research_stellar_elapsed_years(beginning,All.Time);
  code_units units=solver.units;units.a_value=All.ComovingIntegrationOn ? All.Time : 1.;
  gr_float density=sph.Density*(All.ComovingIntegrationOn ? All.cf_a3inv : 1.),u=old;
  if(!std::isfinite(density) || density<=0)Terminate("Grackle proper gas density is invalid");
  gr_float chemical[12]={};
  const double isotope=All.GracklePrimordialChemistry==3?All.GrackleDeuteriumToHydrogenRatio:0.;
  try {research_primordial::restore(state,density,isotope,All.GracklePrimordialChemistry,chemical);}
  catch(const std::exception &error){Terminate("Grackle source restoration: %s",error.what());}
  gr_float metal=density*(1-state.element_mass_fraction[0]-state.element_mass_fraction[1]);
  gr_float vx=0,vy=0,vz=0;int dimension[3]={1,1,1},start[3]={0,0,0},end[3]={0,0,0};
  grackle_field_data fields{};
  if(!gr_initialize_field_data(&fields))Terminate("Grackle field initialization failed");
  fields.grid_rank=3;fields.grid_dimension=dimension;fields.grid_start=start;fields.grid_end=end;fields.grid_dx=1.;
  fields.density=&density;fields.internal_energy=&u;fields.metal_density=&metal;
  fields.x_velocity=&vx;fields.y_velocity=&vy;fields.z_velocity=&vz;
  fields.HI_density=&chemical[0];fields.HII_density=&chemical[1];fields.HeI_density=&chemical[2];
  fields.HeII_density=&chemical[3];fields.HeIII_density=&chemical[4];fields.e_density=&chemical[5];
  fields.HM_density=&chemical[6];fields.H2I_density=&chemical[7];fields.H2II_density=&chemical[8];
  fields.DI_density=&chemical[9];fields.DII_density=&chemical[10];fields.HDI_density=&chemical[11];
  const double dt=years*31557600./units.time_units;
  chemistry_data chemistry=solver.chemistry;
  const double transported_d=research_primordial::deuterium(chemical);
  const double desired_h=density*state.element_mass_fraction[0]-transported_d;
  const double desired_he=density*state.element_mass_fraction[1];
  const bool molecular_deuterium=All.GracklePrimordialChemistry==3;
  const double solver_baryons=chemistry.metal_cooling ? density-metal : density;
  const double protium_and_helium=desired_h+desired_he;
  chemistry.HydrogenFractionByMass=molecular_deuterium ? desired_h/protium_and_helium : desired_h/density;
  // Grackle's parameter is a mass ratio. Its final consistency pass sets
  // D to this ratio times its normalized H mass (excluding HD's H).
  chemistry.DeuteriumToHydrogenRatio=molecular_deuterium ?
      transported_d/(solver_baryons*chemistry.HydrogenFractionByMass) : 0.;
  if(!std::isfinite(dt) || dt<=0 || !local_solve_chemistry(&chemistry,&solver.rates,&units,&fields,dt))
    Terminate("Native Grackle chemistry/cooling solve failed");
  if(!std::isfinite(u) || u<0)Terminate("Grackle returned invalid specific internal energy");
  u=std::max(double(u),double(All.MinEgySpec));
  if(molecular_deuterium)try {
    research_primordial::undo_grackle_hd_normalization(chemical,desired_h,desired_he,transported_d,solver_baryons);
  }catch(const std::exception &error){Terminate("Grackle normalization: %s",error.what());}
  if(All.GracklePrimordialChemistry)try {research_primordial::conserve(chemical,desired_h,desired_he,transported_d,All.GracklePrimordialChemistry);}
  catch(const std::exception &error){Terminate("Grackle source conservation: %s",error.what());}
  for(int e=0;e<6;e++){state.grackle_species_mass_fraction[e]=chemical[e]/density;state.grackle_molecular_mass_fraction[e]=chemical[e+6]/density;}
  gr_float pressure=0,gamma=GAMMA,temperature=0;
  if(!local_calculate_pressure(&chemistry,&solver.rates,&units,&fields,&pressure) ||
     !local_calculate_gamma(&chemistry,&solver.rates,&units,&fields,&gamma) ||
     !local_calculate_temperature(&chemistry,&solver.rates,&units,&fields,&temperature))
    Terminate("Grackle molecular thermodynamic calculation failed");
  const double gamma_pressure=u>0?1+pressure/(density*u):GAMMA;
  if(!std::isfinite(gamma_pressure) || gamma_pressure<=1 || gamma_pressure>GAMMA+1e-6 ||
     !std::isfinite(gamma) || gamma<=1 || gamma>GAMMA+1e-6 || !std::isfinite(temperature) || temperature<0)
    Terminate("Grackle molecular EOS left its physical domain");
  state.grackle_gamma_pressure=std::min(double(GAMMA),gamma_pressure);
  state.grackle_gamma_sound=std::min(double(GAMMA),double(gamma));state.grackle_temperature_kelvin=temperature;
  state.grackle_mean_molecular_weight=temperature*1.380649e-16/
      (1.67262192595e-24*(gamma_pressure-1)*std::max(double(u),double(All.MinEgySpec))*units.velocity_units*units.velocity_units);
  if(!std::isfinite(state.grackle_mean_molecular_weight) || state.grackle_mean_molecular_weight<=0)
    Terminate("Grackle EOS returned invalid mean molecular weight");
  state.grackle_inverse_cooling_time_per_s=inverse_cooling_time(chemistry,solver.rates,units,fields);
  sph.ResearchGammaPressure=state.grackle_gamma_pressure;sph.ResearchGammaSound=state.grackle_gamma_sound;
  sph.Ne=desired_h>0 ? chemical[5]/desired_h : 0.;
  const double result=std::max<double>(u,All.MinEgySpec);
  state.chemical_radiated_energy+=p.getMass()*(old-result); // Signed: photoheating is negative radiation loss.
  state.chemical_last_update_time=All.Time;state.grackle_state_version=research_primordial::version(All.GracklePrimordialChemistry);state.grackle_data_crc32=solver.data_crc;
  return result;
}

void research_grackle_refresh_thermodynamics(simparticles *particles,int index) {
  research_grackle_refresh_thermodynamics(&particles->P[index],&particles->SphP[index]);
}
void research_grackle_refresh_thermodynamics(particle_data *particle,sph_particle_data *gas) {
  auto &state=particle->ResearchSourceState;auto &sph=*gas;
  sph.ResearchGammaPressure=GAMMA;sph.ResearchGammaSound=GAMMA;
  state.grackle_inverse_cooling_time_per_s=0;
  if(state.chemical_state_version!=1 || !state.grackle_state_version)return;
  if(state.grackle_state_version!=research_primordial::version(All.GracklePrimordialChemistry))
    Terminate("Grackle EOS network differs from stored chemical state");
#ifdef STARFORMATION
  if(sph.Sfr>0)return; // effective EOS is not a molecular single-phase gas
#endif
  auto &solver=kernel();
  code_units units=solver.units;units.a_value=All.ComovingIntegrationOn?All.Time:1;
  gr_float density=sph.Density*All.cf_a3inv;
  gr_float u=std::max(double(All.MinEgySpec),double(sph.EntropyPred)*std::pow(double(sph.Density)*All.cf_a3inv,GAMMA_MINUS1)/GAMMA_MINUS1);
  if(!std::isfinite(density) || density<=0 || !std::isfinite(u) || u<=0)
    Terminate("Grackle EOS refresh requires positive finite predicted density and internal energy");
  gr_float chemical[12]={};const double isotope=All.GracklePrimordialChemistry==3?All.GrackleDeuteriumToHydrogenRatio:0;
  try {research_primordial::restore(state,density,isotope,All.GracklePrimordialChemistry,chemical);}
  catch(const std::exception &error){Terminate("Grackle EOS species restoration: %s",error.what());}
  gr_float metal=density*(1-state.element_mass_fraction[0]-state.element_mass_fraction[1]);
  int dimension[3]={1,1,1},start[3]={0,0,0},end[3]={0,0,0};grackle_field_data fields{};
  if(!gr_initialize_field_data(&fields))Terminate("Grackle EOS field initialization failed");
  fields.grid_rank=3;fields.grid_dimension=dimension;fields.grid_start=start;fields.grid_end=end;fields.grid_dx=1;
  fields.density=&density;fields.internal_energy=&u;fields.metal_density=&metal;
  fields.HI_density=&chemical[0];fields.HII_density=&chemical[1];fields.HeI_density=&chemical[2];
  fields.HeII_density=&chemical[3];fields.HeIII_density=&chemical[4];fields.e_density=&chemical[5];
  fields.HM_density=&chemical[6];fields.H2I_density=&chemical[7];fields.H2II_density=&chemical[8];
  fields.DI_density=&chemical[9];fields.DII_density=&chemical[10];fields.HDI_density=&chemical[11];
  chemistry_data chemistry=solver.chemistry;
  const double transported_d=research_primordial::deuterium(chemical);
  chemistry.HydrogenFractionByMass=state.element_mass_fraction[0]-transported_d/density;
  chemistry.DeuteriumToHydrogenRatio=chemistry.HydrogenFractionByMass>0?transported_d/(density*chemistry.HydrogenFractionByMass):0;
  gr_float pressure=0,gamma=GAMMA,temperature=0;
  if(!local_calculate_pressure(&chemistry,&solver.rates,&units,&fields,&pressure) ||
     !local_calculate_gamma(&chemistry,&solver.rates,&units,&fields,&gamma) ||
     !local_calculate_temperature(&chemistry,&solver.rates,&units,&fields,&temperature))
    Terminate("Grackle EOS refresh failed");
  const double gp=1+pressure/(density*u);
  if(!std::isfinite(gp) || gp<=1 || gp>GAMMA+1e-6 || !std::isfinite(gamma) || gamma<=1 || gamma>GAMMA+1e-6 || !std::isfinite(temperature) || temperature<0)
    Terminate("Refreshed Grackle EOS is outside its physical domain");
  state.grackle_gamma_pressure=std::min(double(GAMMA),gp);state.grackle_gamma_sound=std::min(double(GAMMA),double(gamma));
  state.grackle_temperature_kelvin=temperature;state.grackle_mean_molecular_weight=temperature*1.380649e-16/(1.67262192595e-24*(gp-1)*u*units.velocity_units*units.velocity_units);
  if(!std::isfinite(state.grackle_mean_molecular_weight) || state.grackle_mean_molecular_weight<=0)
    Terminate("Refreshed Grackle mean molecular weight is invalid");
  state.grackle_inverse_cooling_time_per_s=inverse_cooling_time(chemistry,solver.rates,units,fields);
  sph.ResearchGammaPressure=state.grackle_gamma_pressure;sph.ResearchGammaSound=state.grackle_gamma_sound;
}
#endif
