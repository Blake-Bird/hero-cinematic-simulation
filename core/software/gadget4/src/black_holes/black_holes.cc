/* Project extension: synchronized thermal Bondi/Eddington source terms.
 * Fixed proper-radius cubic-spline gas kernel; all MPI ranks participate.
 * BH state migrates and restarts with particle_data. Seed placement is supplied
 * by explicit type-5 IC particles or a declared density/exclusion rule.
 * No implicit halo mass threshold or BH merging is assumed.
 */
#include "gadgetconfig.h"

#ifdef BLACK_HOLES
#if defined(IDS_32BIT) || defined(IDS_48BIT)
#error "Native BH lifecycle requires exact IDS_64BIT identities"
#endif
#ifdef GADGET2_HEADER
#error "Native BH lifecycle snapshots require the extended GADGET-4 header"
#endif
#if !defined(FORCE_EQUAL_TIMESTEPS) || defined(HIERARCHICAL_GRAVITY) || defined(LEAN) || defined(ISOTHERM_EQS) || defined(PRESSURE_ENTROPY_SPH)
#error "BLACK_HOLES requires global timesteps, density-entropy SPH, and ordinary particle storage"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include <set>
#include <map>
#include "research_bh_seed_collective.h"
#include "research_bh_seed_storage.h"
#include "research_bh_seed_journal.h"
#include "research_bh_accretion_roundoff.h"
#include "../main/simulation.h"
#include "research_bh_dynamics.h"

namespace {
struct bh_record {
  MyIDType id;
  MyIntPosType pos[3];
  double vel[3], mass, submass, last_time, spin[3];
  double pending_mass, pending_correction, gravity_correction, physical_correction;
  uint64_t counter;
};

double kernel(double r, double h) {
  double q = r/h;
  if(q >= 1) return 0;
  double shape = q < .5 ? 1 - 6*q*q + 6*q*q*q : 2*std::pow(1-q, 3);
  return 8/M_PI/(h*h*h)*shape;
}
}

void sim::black_hole_source_step(void) {
  if(!All.ComovingIntegrationOn)
    Terminate("BLACK_HOLES currently requires comoving integration and declared h-scaled units");
  if(!(All.BHRadiativeEfficiency > 0 && All.BHRadiativeEfficiency < 1 &&
       All.BHFeedbackCoupling >= 0 && All.BHFeedbackCoupling <= 1 &&
       All.BHBondiBoost > 0 && All.BHEddingtonFactor > 0 &&
       All.BHKernelRadiusProperKpc > 0 && All.BHSeedMassMsun > 0 &&
       All.BHSourceFraction > 0 && All.BHSourceFraction <= .1 && All.BHMaximumCount > 0))
    Terminate("Invalid BLACK_HOLES model parameters");
  if((All.BHAngularMomentumModel!=0 && All.BHAngularMomentumModel!=1) ||
      (All.BHFeedbackMode!=0 && All.BHFeedbackMode!=1) || !(All.BHViscosityParameter>0))Terminate("Invalid BH angular/feedback model");
  All.set_cosmo_factors_for_current_time();
  std::vector<bh_record> local;
  std::vector<std::string> journal;
  auto event_prefix=[&](const char*kind,uint64_t id,uint64_t counter){std::ostringstream out;out<<std::setprecision(17)<<"{\"kind\":\""<<kind<<"\",\"native_time\":"<<All.Time<<",\"particle_id\":"<<id<<",\"counter\":"<<counter;return out.str();};
  double unit_mass = All.UnitMass_in_g / All.HubbleParam;
  double unit_time = All.UnitTime_in_s / All.HubbleParam;
  const double solar = 1.98847e33, c = 2.99792458e10/All.UnitVelocity_in_cm_per_s;
  const double grav_cgs = 6.67430e-8, proton = 1.67262192595e-24, thomson = 6.6524587051e-25;
  for(int i=0; i<Sp.NumPart; i++) if(Sp.P[i].getType() == 5 && Sp.P[i].getMass()>0 && Sp.P[i].ID.get()!=0) {
    if(Sp.P[i].BHMass == 0 && All.RestartFlag==RST_STARTFROMSNAP)Terminate("Snapshot restart cannot silently reseed a BH with missing physical mass");
    if(Sp.P[i].BHMass == 0) {
      Sp.P[i].BHMass = All.BHSeedMassMsun * solar / unit_mass;
      auto &state=Sp.P[i].ResearchSourceState;
      if(All.BHDynamicsEnabled){state.bh_state_version=1;state.bh_spin[0]=All.BHSeedSpinX;state.bh_spin[1]=All.BHSeedSpinY;state.bh_spin[2]=All.BHSeedSpinZ;state.bh_last_dynamics_time=All.TimeOld;}
      if(state.bh_event_counter==UINT64_MAX)Terminate("BH seed event counter overflow");++state.bh_event_counter;
      std::ostringstream row;row<<std::setprecision(17)<<event_prefix("seed",Sp.P[i].ID.get(),state.bh_event_counter)
        <<",\"previous_species\":5,\"gravity_mass_native\":"<<Sp.P[i].getMass()<<",\"physical_mass_native\":"<<Sp.P[i].BHMass<<",\"rule\":\"ic_particles\"}";
      research_bh::append(journal,row.str());state.bh_last_source_time=All.TimeOld;
    }
    auto &initial=Sp.P[i].ResearchSourceState;
    if(initial.bh_accretion_roundoff_version==0) {
      if(All.RestartFlag==RST_STARTFROMSNAP)
        Terminate("BH snapshot restart lacks the versioned accretion integral and compensation state");
      if(initial.bh_accretion_pending_mass!=0 || initial.bh_accretion_pending_correction!=0 ||
         initial.bh_gravity_mass_correction!=0 || initial.bh_physical_mass_correction!=0)
        Terminate("Inactive BH accretion roundoff state is nonzero");
      initial.bh_accretion_roundoff_version=1;
    }
    if(initial.bh_accretion_roundoff_version!=1 || !std::isfinite(initial.bh_accretion_pending_mass) ||
       !std::isfinite(initial.bh_accretion_pending_correction) ||
       initial.bh_accretion_pending_mass-initial.bh_accretion_pending_correction<0 ||
       !std::isfinite(initial.bh_gravity_mass_correction) || !std::isfinite(initial.bh_physical_mass_correction))
      Terminate("Invalid versioned BH accretion roundoff state");
    if(initial.bh_event_counter==0 && All.RestartFlag!=RST_STARTFROMSNAP){
      initial.bh_event_counter=1;initial.bh_last_source_time=All.TimeOld;
      std::ostringstream row;row<<std::setprecision(17)<<event_prefix("seed",Sp.P[i].ID.get(),1)
        <<",\"previous_species\":5,\"gravity_mass_native\":"<<Sp.P[i].getMass()<<",\"physical_mass_native\":"<<Sp.P[i].BHMass<<",\"rule\":\"ic_particles\"}";
      research_bh::append(journal,row.str());
    }
    if(All.BHDynamicsEnabled && !initial.bh_state_version){
      if(All.RestartFlag==RST_STARTFROMSNAP)Terminate("Snapshot restart lacks BH spin/dynamical state");
      initial.bh_state_version=1;initial.bh_spin[0]=All.BHSeedSpinX;initial.bh_spin[1]=All.BHSeedSpinY;initial.bh_spin[2]=All.BHSeedSpinZ;initial.bh_last_dynamics_time=All.TimeOld;
    }
    if(!(Sp.P[i].BHMass > 0 && Sp.P[i].getMass() >= Sp.P[i].BHMass))
      Terminate("BH subgrid mass must be positive and no larger than gravity mass");
    bh_record b{};
    b.id = Sp.P[i].ID.get(); b.mass = Sp.P[i].getMass(); b.submass = Sp.P[i].BHMass;
    b.last_time=Sp.P[i].ResearchSourceState.bh_last_source_time;b.counter=Sp.P[i].ResearchSourceState.bh_event_counter;
    b.pending_mass=initial.bh_accretion_pending_mass;b.pending_correction=initial.bh_accretion_pending_correction;
    b.gravity_correction=initial.bh_gravity_mass_correction;b.physical_correction=initial.bh_physical_mass_correction;
    for(int k=0;k<3;k++) { b.pos[k]=Sp.P[i].IntPos[k]; b.vel[k]=Sp.P[i].Vel[k]/All.cf_atime;b.spin[k]=Sp.P[i].ResearchSourceState.bh_spin[k]; }
    local.push_back(b);
    Sp.P[i].BHInflowRate = 0;
  }
  std::vector<int> sizes(NTask), offsets(NTask);
  if(local.size()>size_t(INT_MAX)/sizeof(bh_record))Terminate("BH local record byte overflow");
  int bytes = local.size()*sizeof(bh_record);
  MPI_Allgather(&bytes,1,MPI_INT,sizes.data(),1,MPI_INT,Communicator);
  long long totalbytes=0;
  for(int rank=0;rank<NTask;rank++) { if(totalbytes>INT_MAX)Terminate("BH global record byte overflow");offsets[rank]=totalbytes; totalbytes+=sizes[rank]; }
  if(totalbytes > INT_MAX || totalbytes/sizeof(bh_record) > (size_t)All.BHMaximumCount || totalbytes>All.MaxMemSize*1024.*1024.*.25)
    Terminate("BH replicated-record allocation exceeds BHMaximumCount/MPI byte limit");
  std::vector<bh_record> holes(totalbytes/sizeof(bh_record));
  MPI_Allgatherv(local.data(),bytes,MPI_BYTE,holes.data(),sizes.data(),offsets.data(),MPI_BYTE,Communicator);
  std::sort(holes.begin(),holes.end(),[](const bh_record&a,const bh_record&b){return a.id<b.id;});
  for(size_t j=1;j<holes.size();j++) if(holes[j].id==holes[j-1].id) Terminate("Duplicated BH particle ID");
  int seeds=0;
  if(All.BHSeedingMode == 1 || All.BHSeedingMode == 2) {
    if(!(All.BHSeedDensityHydrogenCm3>0 && All.BHSeedExclusionProperKpc>0))
      Terminate("Density seeding requires positive threshold and exclusion radius");
    double seedmass=All.BHSeedMassMsun*solar/unit_mass;
    double exclusion=All.BHSeedExclusionProperKpc*3.0856775814913673e21*All.HubbleParam/All.UnitLength_in_cm/All.cf_atime;
    if(exclusion>=All.BoxSize/2) Terminate("BH seed exclusion exceeds unique periodic region");
    std::set<uint64_t> unavailable_seed_carriers;
    while(true) {
      bh_record best{};
      for(int i=0;i<Sp.NumGas;i++) if(Sp.P[i].getType()==0 && Sp.P[i].getMass()>0 && (All.BHSeedingMode==2 || Sp.P[i].getMass()>=seedmass)) {
        double rho=Sp.SphP[i].Density*All.cf_a3inv*All.UnitDensity_in_cgs*All.HubbleParam*All.HubbleParam;
        double nh=HYDROGEN_MASSFRAC*rho/proton;
        if(nh<All.BHSeedDensityHydrogenCm3) continue;
        bool excluded=false;
        for(auto &hole: holes) {
          double d[3]; Sp.nearest_image_intpos_to_pos(Sp.P[i].IntPos,hole.pos,d);
          if(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]<exclusion*exclusion) {excluded=true;break;}
        }
        if(excluded || unavailable_seed_carriers.count(Sp.P[i].ID.get())) continue;
        auto id=Sp.P[i].ID.get();
        if(best.mass==0 || nh>best.submass || (nh==best.submass && id<best.id)) {
          best.id=id; best.mass=Sp.P[i].getMass(); best.submass=nh;
          for(int k=0;k<3;k++) {best.pos[k]=Sp.P[i].IntPos[k];best.vel[k]=Sp.P[i].Vel[k]/All.cf_atime;}
        }
      }
      std::vector<bh_record> candidates(NTask);
      MPI_Allgather(&best,sizeof(best),MPI_BYTE,candidates.data(),sizeof(best),MPI_BYTE,Communicator);
      bh_record selected{};
      for(auto &candidate:candidates) if(candidate.mass>0 &&
          (selected.mass==0 || candidate.submass>selected.submass ||
           (candidate.submass==selected.submass && candidate.id<selected.id))) selected=candidate;
      if(selected.mass==0) break;
      if(holes.size()>=(size_t)All.BHMaximumCount) Terminate("Eligible BH seed exceeds BHMaximumCount");
      MyFloat stored_seed_velocity[3]={};
      if(All.BHSeedingMode==2) {
        // Explicit mode: nearest-first donors in the configured proper kernel
        // aperture, each bounded by BHSourceFraction. No partial seed commits.
        const double length_unit=All.UnitLength_in_cm/All.HubbleParam;
        const double aperture=All.BHKernelRadiusProperKpc*3.0856775814913673e21/length_unit/All.cf_atime;
        if(aperture>=All.BoxSize/2)Terminate("BH seed reservoir aperture exceeds unique periodic region");
        const size_t maximum_records=std::min(size_t(INT_MAX)/sizeof(research_bh_seed::gas),
          size_t(All.MaxMemSize*1024.*1024.*.0625/1024.));
        std::vector<research_bh_seed::gas> gas_records;
        auto collect_gas=[&](){
          std::vector<research_bh_seed::gas> records;
          bool within_bound=true;
          for(int i=0;i<Sp.NumGas;i++)if(Sp.P[i].getType()==0 && Sp.P[i].getMass()>0) {
            double d[3];Sp.nearest_image_intpos_to_pos(Sp.P[i].IntPos,selected.pos,d);
            if(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]>=aperture*aperture && Sp.P[i].ID.get()!=selected.id)continue;
            if(records.size()>=maximum_records){within_bound=false;break;}
            research_bh_seed::gas record{};record.id=Sp.P[i].ID.get();record.mass=Sp.P[i].getMass();record.specific_internal_energy=Sp.get_utherm_from_entropy(i);
            for(int k=0;k<3;k++){record.offset[k]=d[k]*All.cf_atime;record.velocity[k]=Sp.P[i].Vel[k]/All.cf_atime;}
            for(int k=0;k<11;k++)record.mass_fractions[k]=Sp.P[i].ResearchSourceState.element_mass_fraction[k];
            records.push_back(record);
          }
          research_bh_seed::collective_require(within_bound,Communicator,"BH seed aperture exceeds local donor workspace");
          return records;
        };
        try {
          gas_records=collect_gas();
          auto prepared=research_bh_seed::prepare_collective(gas_records,selected.id,seedmass,All.BHSourceFraction,maximum_records,Communicator);
          if(!prepared.assembly.supported){unavailable_seed_carriers.insert(selected.id);continue;}
          long double old_sink_kinetic=0,new_sink_kinetic=0;
          for(int k=0;k<3;k++) {
            const double ideal=prepared.assembly.velocity[k];
            stored_seed_velocity[k]=MyFloat(ideal*All.cf_atime);
            const double actual=double(stored_seed_velocity[k])/All.cf_atime;
            old_sink_kinetic+=.5L*prepared.assembly.gravity_mass*ideal*ideal;
            new_sink_kinetic+=.5L*prepared.assembly.gravity_mass*actual*actual;
            prepared.assembly.velocity[k]=actual;
          }
          prepared.assembly.mixing_energy=double(static_cast<long double>(prepared.assembly.mixing_energy)+old_sink_kinetic-new_sink_kinetic);
          std::map<uint64_t,research_bh_seed::transfer> transfers;
          for(const auto &change:prepared.assembly.transfers)transfers.emplace(change.donor_id,change);
          // Stage every journal row and ledger update before the collective
          // commit check. Failures cannot leave a partially applied seed.
          size_t local_rows=0;
          for(int i=0;i<Sp.NumGas;i++)if(Sp.P[i].getType()==0 && (Sp.P[i].ID.get()==selected.id || transfers.count(Sp.P[i].ID.get())))++local_rows;
          research_bh_seed::collective_require((journal.size()+2*local_rows)*8192.<=All.MaxMemSize*1024.*1024.*.125,Communicator,"BH seed journal staging exceeds workspace");
          std::map<uint64_t,const research_bh_seed::gas*> local_records;
          for(const auto &record:gas_records)local_records.emplace(record.id,&record);
          std::vector<std::string> seed_rows;research_source_state seed_state{};bool local_valid=true;
          for(int i=0;i<Sp.NumGas;i++)if(Sp.P[i].getType()==0) {
            auto found=transfers.find(Sp.P[i].ID.get());
            if(found!=transfers.end()) {
              auto record=local_records.find(found->first);
              if(record==local_records.end())throw std::runtime_error("Missing local seed donor state");
              if(double(MyDouble(found->second.mass_after))!=found->second.mass_after)local_valid=false;
              seed_rows.push_back(research_bh_seed::donor_json(*record->second,found->second,selected.id,All.Time,uint64_t(seeds)+1,unit_mass,length_unit,All.UnitVelocity_in_cm_per_s));
            }
            if(Sp.P[i].ID.get()==selected.id) {
              seed_state=Sp.P[i].ResearchSourceState;
              research_bh_seed::store_assembly(seed_state,prepared.assembly,unit_mass,length_unit,All.UnitVelocity_in_cm_per_s);
              seed_rows.push_back(research_bh_seed::assembly_json(prepared.carrier,prepared.assembly,All.Time,uint64_t(seeds)+1,All.BHSourceFraction,All.BHKernelRadiusProperKpc,unit_mass,length_unit,All.UnitVelocity_in_cm_per_s));
              if(double(MyDouble(prepared.assembly.gravity_mass))!=prepared.assembly.gravity_mass)local_valid=false;
            }
          }
          bool journal_fits=(journal.size()+seed_rows.size())*8192.<=All.MaxMemSize*1024.*1024.*.125;
          for(const auto &row:seed_rows)if(row.size()>8192)journal_fits=false;
          research_bh_seed::collective_require(local_valid && journal_fits,Communicator,"BH seed stored state/journal exceeds representable transaction budget");
          journal.reserve(journal.size()+seed_rows.size());
          for(auto &row:seed_rows)row.push_back('\n');
          research_bh_seed::validate_collective_commit(prepared,collect_gas(),Communicator);
          for(int i=0;i<Sp.NumGas;i++)if(Sp.P[i].getType()==0) {
            auto found=transfers.find(Sp.P[i].ID.get());
            if(found!=transfers.end()) {
              Sp.P[i].setMass(found->second.mass_after);
#ifdef STARFORMATION
              Sp.SphP[i].MassMetallicity=Sp.P[i].getMass()*Sp.P[i].Metallicity;
#endif
            }
            if(Sp.P[i].ID.get()==selected.id)Sp.P[i].ResearchSourceState=seed_state;
          }
          for(auto &row:seed_rows)journal.emplace_back(std::move(row));
          selected.mass=prepared.assembly.gravity_mass;
          for(int k=0;k<3;k++)selected.vel[k]=prepared.assembly.velocity[k];
        }catch(const std::exception &error){Terminate("BH seed reservoir transaction failed: %s",error.what());}
      }
      mpi_printf("BH_SEED time=%.17g id=%llu gas_gravity_mass=%.17g seed_mass=%.17g hydrogen_cm3=%.17g rule=%s\n",
                 All.Time,(unsigned long long)selected.id,selected.mass,seedmass,selected.submass,All.BHSeedingMode==1 ? "density_exclusion" : "density_exclusion_reservoir");
      for(int i=0;i<Sp.NumGas;i++) if(Sp.P[i].getType()==0 && Sp.P[i].ID.get()==selected.id) {
#ifdef STARFORMATION
        Sp.TimeBinSfr[Sp.P[i].getTimeBinHydro()]-=Sp.SphP[i].Sfr;
#endif
        if(All.BHSeedingMode==2){Sp.P[i].setMass(selected.mass);for(int k=0;k<3;k++)Sp.P[i].Vel[k]=stored_seed_velocity[k];}
        Sp.P[i].setType(5); Sp.P[i].BHMass=seedmass;
        auto &state=Sp.P[i].ResearchSourceState;state.bh_event_counter=1;state.bh_last_source_time=All.Time;state.bh_state_version=All.BHDynamicsEnabled ? 1 : 0;state.bh_last_dynamics_time=All.Time;state.bh_spin[0]=All.BHSeedSpinX;state.bh_spin[1]=All.BHSeedSpinY;state.bh_spin[2]=All.BHSeedSpinZ;
        if(All.BHSeedingMode==1){std::ostringstream row;row<<std::setprecision(17)<<event_prefix("seed",selected.id,1)<<",\"previous_species\":0,\"gravity_mass_native\":"<<selected.mass<<",\"physical_mass_native\":"<<seedmass<<",\"rule\":\"density_exclusion\"}";research_bh::append(journal,row.str());}
        Sp.P[i].BHInflowRate=Sp.P[i].BHCumulativeRadiatedMass=Sp.P[i].BHCumulativeFeedbackEnergy=0;
        Sp.P[i].setSofteningClass(All.SofteningClassOfPartType[5]);
      }
      selected.submass=seedmass;selected.counter=1;selected.last_time=All.Time;selected.spin[0]=All.BHSeedSpinX;selected.spin[1]=All.BHSeedSpinY;selected.spin[2]=All.BHSeedSpinZ;
      holes.push_back(selected);
      seeds++;
    }
    Sp.TotNumGas-=seeds;
    std::sort(holes.begin(),holes.end(),[](const bh_record&a,const bh_record&b){return a.id<b.id;});
  } else if(All.BHSeedingMode != 0) Terminate("Unknown BHSeedingMode");
  if(holes.empty() && All.BHSeedingMode==0) Terminate("BHSeedingMode=0 requires explicit type-5 IC seeds");

  // Radius proper kpc -> native comoving length. No hidden maximum kernel.
  double radius = All.BHKernelRadiusProperKpc*3.0856775814913673e21*All.HubbleParam/All.UnitLength_in_cm/All.cf_atime;
  if(radius >= All.BoxSize/2) Terminate("BH kernel radius exceeds unique periodic region");
  double eps = All.BHRadiativeEfficiency, coupling = All.BHFeedbackCoupling;
  double consumed_total=0, radiated_total=0, feedback_total=0;
  int source_steps=0;
  for(auto &bh: holes) {
    double duration=research_bh::proper_seconds(bh.last_time>0 ? bh.last_time : All.TimeOld,All.Time)/unit_time;
    double remaining=duration, consumed=0, radiated=0, feedback=0,advected=0,kinetic_total=0;
    double measured_density=0,measured_sound=0,measured_relative=0,measured_j=0,measured_suppression=1,measured_eddington=0;
    int owner_index=-1;
    for(int i=0;i<Sp.NumPart;i++) if(Sp.P[i].getType()==5 && Sp.P[i].ID.get()==bh.id) owner_index=i;
    while(remaining > 0) {
      if(++source_steps > 100000) Terminate("BH source subcycling exhausted; reduce MaxSizeTimestep");
      std::vector<double> weights(Sp.NumGas,0);
      double sums[5]={}, global[5]={};
      for(int i=0;i<Sp.NumGas;i++) if(Sp.P[i].getType()==0) {
        double d[3]; Sp.nearest_image_intpos_to_pos(Sp.P[i].IntPos,bh.pos,d);
        double r=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
        double w=Sp.P[i].getMass()*kernel(r,radius);
        weights[i]=w; sums[0]+=w;
        if(w>0) {
          double u=Sp.get_utherm_from_entropy(i);
          if(!(u>0) || !std::isfinite(u)) Terminate("BH gas kernel requires positive finite internal energy");
          sums[1]+=w*u;
          for(int k=0;k<3;k++) sums[k+2]+=w*Sp.P[i].Vel[k]/All.cf_atime;
        }
      }
      MPI_Allreduce(sums,global,5,MPI_DOUBLE,MPI_SUM,Communicator);
      if(global[0]<=0) break;
      double umean=global[1]/global[0], vmean[3], relative2=0;
      for(int k=0;k<3;k++) {vmean[k]=global[k+2]/global[0]; relative2+=std::pow(vmean[k]-bh.vel[k],2);}
      double density=global[0]*All.cf_a3inv*All.UnitDensity_in_cgs*All.HubbleParam*All.HubbleParam;
      double cs2=GAMMA*GAMMA_MINUS1*umean*All.UnitVelocity_in_cm_per_s*All.UnitVelocity_in_cm_per_s;
      double rel2=relative2*All.UnitVelocity_in_cm_per_s*All.UnitVelocity_in_cm_per_s;
      double physical_mass=bh.submass*unit_mass;
      double bondi=4*M_PI*All.BHBondiBoost*grav_cgs*grav_cgs*physical_mass*physical_mass*density/std::pow(cs2+rel2,1.5);
      double edd=All.BHEddingtonFactor*4*M_PI*grav_cgs*physical_mass*proton/(eps*thomson*2.99792458e10);
      double angular[4]={},globalangular[4]={};
      for(int i=0;i<Sp.NumGas;i++)if(weights[i]>0){double d[3];Sp.nearest_image_intpos_to_pos(Sp.P[i].IntPos,bh.pos,d);
        research_bh::vec r{},v{};for(int k=0;k<3;k++){r[k]=d[k]*All.cf_atime*All.UnitLength_in_cm/All.HubbleParam;v[k]=(Sp.P[i].Vel[k]/All.cf_atime-vmean[k])*All.UnitVelocity_in_cm_per_s;}
        auto j=research_bh::cross(r,v);for(int k=0;k<3;k++)angular[k]+=weights[i]*j[k];angular[3]+=weights[i]*research_bh::norm(r);}
      MPI_Allreduce(angular,globalangular,4,MPI_DOUBLE,MPI_SUM,Communicator);
      research_bh::vec meanj{};for(int k=0;k<3;k++)meanj[k]=globalangular[k]/global[0];double jnorm=research_bh::norm(meanj),effective_radius=globalangular[3]/global[0];
      double rotation=effective_radius>0 ? jnorm/effective_radius : 0,suppression=1;
      if(All.BHAngularMomentumModel==1 && rotation>0)suppression=std::min(1.,pow(sqrt(cs2)/rotation,3)/All.BHViscosityParameter);
      double selected=std::min(bondi*suppression,edd),rate=selected*unit_time/unit_mass;
      measured_density=density;measured_sound=sqrt(cs2);measured_relative=sqrt(rel2);measured_j=jnorm;measured_suppression=suppression;measured_eddington=selected/edd;
      if(!std::isfinite(rate) || rate<0) Terminate("Invalid BH accretion rate");
      if(rate==0) break;
      // Limit fractional mass removal, BH growth and gas thermal-energy
      // change per source substep. Rates are recomputed after every update.
      double capacity=All.BHSourceFraction*bh.submass/(1-eps);
      double timestep=std::min(remaining,capacity/rate);
      for(int i=0;i<Sp.NumGas;i++) if(weights[i]>0) {
        double fraction=weights[i]/global[0], mass=Sp.P[i].getMass(), u=Sp.get_utherm_from_entropy(i);
        timestep=std::min(timestep,All.BHSourceFraction*mass/(rate*fraction));
        capacity=std::min(capacity,All.BHSourceFraction*mass/fraction);
        // Relative kinetic mixing supplies a conservative upper bound on
        // the extra dissipative heat when donors join the sink momentum.
        double dv2=0;
        for(int k=0;k<3;k++) dv2+=std::pow(Sp.P[i].Vel[k]/All.cf_atime-bh.vel[k],2);
        double heat_per_mass=coupling*eps*c*c+(1-eps)*dv2/2;
        if(heat_per_mass>0) {
          timestep=std::min(timestep,All.BHSourceFraction*mass*u/(rate*fraction*heat_per_mass));
          capacity=std::min(capacity,All.BHSourceFraction*mass*u/(fraction*heat_per_mass));
        }
      }
      double globalstep;
      MPI_Allreduce(&timestep,&globalstep,1,MPI_DOUBLE,MPI_MIN,Communicator);
      double globalcapacity;
      MPI_Allreduce(&capacity,&globalcapacity,1,MPI_DOUBLE,MPI_MIN,Communicator);
      if(!(globalstep>0) || remaining-globalstep==remaining) Terminate("BH source timestep underflow");
      if(bh.counter==UINT64_MAX)Terminate("BH source event counter overflow");++bh.counter;
      const double requested=rate*globalstep, oldmass=bh.mass, oldphysical=bh.submass;
      const double oldpending=bh.pending_mass, oldpendingcorrection=bh.pending_correction;
      const double oldgravitycorrection=bh.gravity_correction, oldphysicalcorrection=bh.physical_correction;
      research_bh_accretion::compensated_value demand;
      try {demand=research_bh_accretion::add_increment({oldpending,oldpendingcorrection},requested);}
      catch(const std::exception &error){Terminate("Invalid BH accretion integral: %s",error.what());}
      const double available=demand.value-demand.correction;
      if(!(std::isfinite(available) && available>=0 && std::isfinite(globalcapacity) && globalcapacity>0))
        Terminate("Invalid pending BH accretion demand/capacity");
      double allocation=std::min(available,globalcapacity),donor[6]={},globaldonor[6]={};
      research_bh::vec oldvelocity{{bh.vel[0],bh.vel[1],bh.vel[2]}};
      std::vector<double> beforemass(Sp.NumGas),beforeu(Sp.NumGas);
      std::vector<research_bh::vec> beforevelocity(Sp.NumGas);
      std::vector<MyDouble> aftermass(Sp.NumGas);
      for(int i=0;i<Sp.NumGas;i++) if(weights[i]>0) {
        beforemass[i]=Sp.P[i].getMass();beforeu[i]=Sp.get_utherm_from_entropy(i);
        for(int k=0;k<3;k++)beforevelocity[i][k]=Sp.P[i].Vel[k]/All.cf_atime;
      }
      bool planned=false;
      for(int attempt=0;attempt<8 && !planned;attempt++) {
        std::fill(donor,donor+6,0.);
        for(int i=0;i<Sp.NumGas;i++) if(weights[i]>0) {
          research_bh_accretion::donor_plan<MyDouble> plan;
          try {plan=research_bh_accretion::plan_donor(MyDouble(beforemass[i]),allocation*weights[i]/global[0],All.BHSourceFraction);}
          catch(const std::exception &error){Terminate("Invalid BH donor quantization: %s",error.what());}
          aftermass[i]=plan.after;double dm=plan.removed,v2=0;
          donor[5]+=dm;
          for(int k=0;k<3;k++){double v=beforevelocity[i][k];donor[k]+=dm*v;v2+=v*v;}
          donor[3]+=dm*v2/2;donor[4]+=dm*beforeu[i];
        }
        MPI_Allreduce(donor,globaldonor,6,MPI_DOUBLE,MPI_SUM,Communicator);
        // Validate the global budget before changing any particle. A bounded
        // replan handles MPI summation rounding; unused demand is retained.
        planned=globaldonor[5]>=0 && globaldonor[5]<=available && globaldonor[5]<=globalcapacity;
        if(!planned)allocation*=.5;
      }
      if(!planned)Terminate("BH donor plan cannot respect its global pending/source-fraction budget");
      const double inflow=globaldonor[5];
      research_bh_accretion::compensated_value pending;
      try {pending=research_bh_accretion::add_increment(demand,-inflow);}
      catch(const std::exception &error){Terminate("Invalid residual BH demand: %s",error.what());}
      if(pending.value-pending.correction<0)Terminate("BH donor plan overdraws pending demand");
      bh.pending_mass=pending.value;bh.pending_correction=pending.correction;
      auto record_integral=[&]() {
        if(owner_index<0)return;
        std::ostringstream row;row<<std::setprecision(17)<<event_prefix("accretion_integral",bh.id,bh.counter)
          <<",\"version\":1,\"requested_inflow_mass_native\":"<<requested<<",\"applied_inflow_mass_native\":"<<inflow<<",\"retained_gain_mass_native\":"<<(1-eps)*inflow
          <<",\"pending_mass_before_native\":"<<oldpending<<",\"pending_mass_after_native\":"<<bh.pending_mass
          <<",\"pending_mass_correction_before_native\":"<<oldpendingcorrection<<",\"pending_mass_correction_after_native\":"<<bh.pending_correction
          <<",\"gravity_mass_correction_before_native\":"<<oldgravitycorrection<<",\"gravity_mass_correction_after_native\":"<<bh.gravity_correction
          <<",\"physical_mass_correction_before_native\":"<<oldphysicalcorrection<<",\"physical_mass_correction_after_native\":"<<bh.physical_correction
          <<",\"mass_before_native\":"<<oldmass<<",\"mass_after_native\":"<<bh.mass
          <<",\"physical_mass_before_native\":"<<oldphysical<<",\"physical_mass_after_native\":"<<bh.submass<<'}';
        research_bh::append(journal,row.str());
      };
      if(inflow==0) {record_integral();remaining=std::max(0.,remaining-globalstep);continue;}
      for(int i=0;i<Sp.NumGas;i++) if(weights[i]>0) {
        Sp.P[i].setMass(aftermass[i]);
#ifdef STARFORMATION
        Sp.SphP[i].MassMetallicity=Sp.P[i].getMass()*Sp.P[i].Metallicity;
#endif
        if(!(Sp.P[i].getMass()>0)) Terminate("BH consumption produced nonpositive gas mass");
      }
      double oldkin=0,newkin=0;
      try {
        auto gravity=research_bh_accretion::add_increment({bh.mass,bh.gravity_correction},(1-eps)*inflow);
        auto physical=research_bh_accretion::add_increment({bh.submass,bh.physical_correction},(1-eps)*inflow);
        bh.mass=gravity.value;bh.gravity_correction=gravity.correction;
        bh.submass=physical.value;bh.physical_correction=physical.correction;
      } catch(const std::exception &error){Terminate("BH mass accumulation failed: %s",error.what());}
      // Spherical Bondi inflow supplies zero intrinsic spin torque. Preserve
      // dimensional BH spin angular momentum rather than creating it with M^2.
      for(int k=0;k<3;k++)bh.spin[k]*=pow(oldphysical/bh.submass,2);
      for(int k=0;k<3;k++) {
        oldkin+=oldmass*bh.vel[k]*bh.vel[k]/2;
        bh.vel[k]=(oldmass*bh.vel[k]+(1-eps)*globaldonor[k])/bh.mass;
        newkin+=bh.mass*bh.vel[k]*bh.vel[k]/2;
      }
      double mixing=std::max(0.,oldkin+(1-eps)*globaldonor[3]-newkin);
      double deposited=coupling*eps*inflow*c*c;
      double kinetic=0;
      if(All.BHFeedbackMode==1 && deposited>0){
        std::vector<research_bh::vec> direction(Sp.NumGas);double moments[5]={},gmoments[5]={};
        for(int i=0;i<Sp.NumGas;i++)if(weights[i]>0){double d[3];Sp.nearest_image_intpos_to_pos(Sp.P[i].IntPos,bh.pos,d);research_bh::vec r{{d[0],d[1],d[2]}};
          if(research_bh::norm(r)==0)for(int k=0;k<3;k++)r[k]=2*double(research_bh::keyed_phase(Sp.P[i].ID.get(),bh.id,bh.counter+uint64_t(k),0)>>11)*std::ldexp(1.0,-53)-1;
          direction[i]=research_bh::mul(research_bh::unit(r),weights[i]/global[0]/Sp.P[i].getMass());double mass=Sp.P[i].getMass();moments[0]+=mass;moments[4]++;
          for(int k=0;k<3;k++)moments[k+1]+=mass*direction[i][k];}
        MPI_Allreduce(moments,gmoments,5,MPI_DOUBLE,MPI_SUM,Communicator);
        if(gmoments[4]<2)Terminate("Kinetic BH feedback requires at least two distinct measured gas recipients");
        double quadratic[2]={},gquadratic[2]={};for(int i=0;i<Sp.NumGas;i++)if(weights[i]>0){double mass=Sp.P[i].getMass();research_bh::vec v{};for(int k=0;k<3;k++){direction[i][k]-=gmoments[k+1]/gmoments[0];v[k]=Sp.P[i].Vel[k]/All.cf_atime;}quadratic[0]+=.5*mass*research_bh::dot(direction[i],direction[i]);quadratic[1]+=mass*research_bh::dot(v,direction[i]);}
        MPI_Allreduce(quadratic,gquadratic,2,MPI_DOUBLE,MPI_SUM,Communicator);if(!(gquadratic[0]>0))Terminate("Degenerate kinetic feedback momentum geometry");
        double radical=sqrt(gquadratic[1]*gquadratic[1]+4*gquadratic[0]*deposited),scale=gquadratic[1]>=0 ? 2*deposited/(radical+gquadratic[1]) : (radical-gquadratic[1])/(2*gquadratic[0]);
        for(int i=0;i<Sp.NumGas;i++)if(weights[i]>0)for(int k=0;k<3;k++)Sp.P[i].Vel[k]+=scale*direction[i][k]*All.cf_atime;
        kinetic=deposited;
      }
      for(int i=0;i<Sp.NumGas;i++) if(weights[i]>0) {
        double du=(deposited-kinetic+mixing)*weights[i]/global[0]/Sp.P[i].getMass();
        Sp.set_entropy_from_utherm(Sp.get_utherm_from_entropy(i)+du,i);
        Sp.SphP[i].DtEntropy=0;
        Sp.SphP[i].set_thermodynamic_variables();
        research_bh::vec velocity{};for(int k=0;k<3;k++)velocity[k]=Sp.P[i].Vel[k]/All.cf_atime;
        const char *kind=Sp.P[i].getMass()<beforemass[i] ? "gas_accretion" : "gas_feedback";
        std::ostringstream row;row<<std::setprecision(17)<<event_prefix(kind,Sp.P[i].ID.get(),bh.counter)<<",\"source_id\":"<<bh.id<<",\"mass_before_native\":"<<beforemass[i]<<",\"mass_after_native\":"<<Sp.P[i].getMass()<<",\"velocity_before_native\":"<<research_bh::vector_json(beforevelocity[i])<<",\"velocity_after_native\":"<<research_bh::vector_json(velocity)<<",\"internal_energy_before_native\":"<<beforeu[i]<<",\"internal_energy_after_native\":"<<Sp.get_utherm_from_entropy(i)<<'}';research_bh::append(journal,row.str());
      }
      if(owner_index>=0){research_bh::vec velocity{{bh.vel[0],bh.vel[1],bh.vel[2]}},radiationP{{eps*globaldonor[0],eps*globaldonor[1],eps*globaldonor[2]}};
        std::ostringstream row;row<<std::setprecision(17)<<event_prefix("accretion",bh.id,bh.counter)<<",\"mass_before_native\":"<<oldmass<<",\"mass_after_native\":"<<bh.mass<<",\"physical_mass_before_native\":"<<oldphysical<<",\"physical_mass_after_native\":"<<bh.submass<<",\"inflow_mass_native\":"<<inflow<<",\"radiated_mass_native\":"<<eps*inflow<<",\"radiated_momentum_native\":"<<research_bh::vector_json(radiationP)<<",\"velocity_before_native\":"<<research_bh::vector_json(oldvelocity)<<",\"velocity_after_native\":"<<research_bh::vector_json(velocity)<<",\"thermal_energy_native\":"<<(deposited-kinetic)<<",\"kinetic_energy_native\":"<<kinetic<<",\"advected_internal_energy_native\":"<<globaldonor[4]<<",\"density_g_cm3\":"<<density<<",\"sound_speed_cm_s\":"<<sqrt(cs2)<<",\"relative_speed_cm_s\":"<<sqrt(rel2)<<",\"specific_angular_momentum_cm2_s\":"<<jnorm<<",\"rotational_suppression\":"<<suppression<<",\"eddington_ratio\":"<<selected/edd<<",\"mixing_energy_native\":"<<mixing<<'}';research_bh::append(journal,row.str());}
      consumed+=inflow; radiated+=eps*inflow; feedback+=deposited;advected+=globaldonor[4];kinetic_total+=kinetic;
      record_integral();
      remaining=std::max(0.,remaining-globalstep);
    }
    if(owner_index>=0) {
      auto &p=Sp.P[owner_index];
      p.ResearchSourceState.bh_event_counter=bh.counter;p.ResearchSourceState.bh_last_source_time=All.Time;
      auto &state=p.ResearchSourceState;state.bh_advected_internal_energy+=advected;state.bh_kinetic_feedback_energy+=kinetic_total;
      state.bh_accretion_roundoff_version=1;state.bh_accretion_pending_mass=bh.pending_mass;state.bh_accretion_pending_correction=bh.pending_correction;
      state.bh_gravity_mass_correction=bh.gravity_correction;state.bh_physical_mass_correction=bh.physical_correction;
      state.bh_density_g_cm3=measured_density;state.bh_sound_speed_cm_s=measured_sound;state.bh_relative_speed_cm_s=measured_relative;
      state.bh_specific_angular_momentum_cm2_s=measured_j;state.bh_rotational_suppression=measured_suppression;state.bh_eddington_ratio=measured_eddington;
      p.setMass(bh.mass); p.BHMass=bh.submass; p.BHInflowRate=duration>0 ? consumed/duration : 0;
      p.BHCumulativeRadiatedMass+=radiated; p.BHCumulativeFeedbackEnergy+=feedback;
      for(int k=0;k<3;k++){p.Vel[k]=bh.vel[k]*All.cf_atime;p.ResearchSourceState.bh_spin[k]=bh.spin[k];}
    }
    consumed_total+=consumed; radiated_total+=radiated; feedback_total+=feedback;
  }
  mpi_printf("BH_SOURCE time=%.17g holes=%zu substeps=%d inflow=%.17g radiated_mass=%.17g feedback_energy=%.17g [native units]\n",
             All.Time,holes.size(),source_steps,consumed_total,radiated_total,feedback_total);
  research_bh::publish(journal,Communicator);
  if(consumed_total > 0 || seeds>0) {
    // A changed gas mass invalidates SPH densities and cached remote tree
    // points. Preserve specific thermal energy through a fresh density solve;
    // retaining the old entropy would spuriously cool the consumed gas.
    for(int i=0;i<Sp.NumGas;i++) if(Sp.P[i].getType()==0)
      Sp.SphP[i].BHSourceInternalEnergy=Sp.get_utherm_from_entropy(i);
    NgbTree.treefree();
    Domain.domain_free();
    Domain.domain_decomposition(STANDARD);
    NgbTree.treeallocate(Sp.NumGas,&Sp,&Domain);
    NgbTree.treebuild(Sp.NumGas,NULL);
    NgbTree.compute_densities();
    NgbTree.update_maxhsml();
    for(int i=0;i<Sp.NumGas;i++) if(Sp.P[i].getType()==0) {
      Sp.set_entropy_from_utherm(Sp.SphP[i].BHSourceInternalEnergy,i);
      Sp.SphP[i].set_thermodynamic_variables();
    }
  }
}
#endif
