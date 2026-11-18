/* Research2 conservative, synchronized delayed stellar sources.
 * Equations and table conventions: Wiersma et al. 2009 (0902.1535).
 * This is original GADGET-4 integration code, not a port of NuPyCEE.
 * Composition, IMF, lifetimes, yields, prompt subtraction and DTD are pinned
 * by the compiled table. No nearest-neighbor metallicity policy is implicit.
 */
#include "gadgetconfig.h"
#ifdef STELLAR_EVOLUTION
#ifdef RESEARCH_GRACKLE
#include "research_primordial_network.h"
#endif
#if !defined(STARFORMATION) || !defined(COOLING) || !defined(FORCE_EQUAL_TIMESTEPS) || defined(HIERARCHICAL_GRAVITY) || defined(LEAN) || defined(ISOTHERM_EQS) || defined(PRESSURE_ENTROPY_SPH)
#error "STELLAR_EVOLUTION requires STARFORMATION, COOLING, synchronized steps and density-entropy SPH"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include "../main/simulation.h"
#include "../time_integration/driftfac.h"
#include "../data/research_stellar_birth.h"
#include "research_stellar_returns.h"
#include "research_restart_fingerprint.h"

namespace {
const double solar_mass=1.98847e33, seconds_per_year=31557600., kpc=3.0856775814913673e21;
struct source_table {
  int nz=0,na=0,edge=0;
  bool population_present=false;
  std::vector<std::array<double,2>> population;
  uint64_t crc=0;
  double beta=0,pattern[11]={};
  std::vector<double> z,age;
  std::vector<std::array<double,13>> data;
  double remnant(double metallicity,double years) const {
    if(!population_present)return -1;
    double zvalue=std::max(z.front(),std::min(z.back(),metallicity));int zr=nz>1 ? std::max(1,std::min(nz-1,int(std::upper_bound(z.begin(),z.end(),zvalue)-z.begin()))) : 0,zl=std::max(0,zr-1);
    double fz=zr==zl ? 0 : (zvalue-z[zl])/(z[zr]-z[zl]);int ar=std::max(1,std::min(na-1,int(std::upper_bound(age.begin(),age.end(),years)-age.begin())));double fa=(years-age[ar-1])/(age[ar]-age[ar-1]);
    return (1-fz)*((1-fa)*population[zl*na+ar-1][1]+fa*population[zl*na+ar][1])+fz*((1-fa)*population[zr*na+ar-1][1]+fa*population[zr*na+ar][1]);
  }
  const std::array<double,13>& at(int zi,int ai,int channel) const {return data[(zi*na+ai)*3+channel];}
  std::array<double,13> cumulative(double metallicity,double years,int channel) const {
    if(!std::isfinite(metallicity) || metallicity<0 || metallicity>1 || !std::isfinite(years) || years<0 || years>age.back())
      Terminate("Stellar source coordinate outside compiled model domain");
    if(!edge && (metallicity<z.front() || metallicity>z.back())) Terminate("Stellar metallicity outside pinned yield grid");
    const double zz=std::max(z.front(),std::min(z.back(),metallicity));
    int zr=nz>1 ? std::max(1,std::min(nz-1,int(std::upper_bound(z.begin(),z.end(),zz)-z.begin()))) : 0;
    int zl=std::max(0,zr-1);
    const double fz=zr==zl ? 0 : (zz-z[zl])/(z[zr]-z[zl]);
    const int ar=std::max(1,std::min(na-1,int(std::upper_bound(age.begin(),age.end(),years)-age.begin())));
    const double fa=(years-age[ar-1])/(age[ar]-age[ar-1]);
    std::array<double,13> out{};
    for(int e=0;e<13;e++) out[e]=(1-fz)*((1-fa)*at(zl,ar-1,channel)[e]+fa*at(zl,ar,channel)[e])+
                                   fz*((1-fa)*at(zr,ar-1,channel)[e]+fa*at(zr,ar,channel)[e]);
    return out;
  }
};

source_table &table() {
  static source_table t;
  if(t.nz)return t;
  uint64_t expected=0,bytes=0,crc=0;
  if(!research_stellar_seed(All.StellarEvolutionTableCRC32,expected) || expected>UINT32_MAX ||
     !research_stellar_file_census(All.StellarEvolutionTableFile,bytes,crc) || crc!=expected || bytes>128*1024*1024)
    Terminate("Stellar table CRC/size mismatch");
  std::ifstream f(All.StellarEvolutionTableFile);
  std::string schema;int ne=0;
  if(!(f>>schema>>t.nz>>t.na>>ne>>t.beta>>t.edge) || (schema!="RESEARCH_STELLAR_TABLE_V1" && schema!="RESEARCH_STELLAR_TABLE_V2") || ne!=11 ||
      t.nz<1 || t.nz>64 || t.na<3 || t.na>8192 || (t.edge!=0 && t.edge!=1) || t.beta!=All.FactorSN)
    Terminate("Stellar table schema or SH03 prompt fraction mismatch");
  if(double(t.nz)*t.na*(3*13+2)*sizeof(double)>All.MaxMemSize*1024.*1024.*.125)
    Terminate("Stellar source table exceeds native allocation allowance");
  t.population_present=schema=="RESEARCH_STELLAR_TABLE_V2";
  t.crc=crc;t.z.resize(t.nz);t.age.resize(t.na);t.data.resize(size_t(t.nz)*t.na*3);
  for(double &v:t.z)if(!(f>>v) || !std::isfinite(v) || v<0 || v>1)Terminate("Invalid stellar metallicity axis");
  for(double &v:t.age)if(!(f>>v) || !std::isfinite(v) || v<0)Terminate("Invalid stellar age axis");
  if(t.age[0]!=0)Terminate("Stellar source model must begin at age zero");
  for(int i=1;i<t.nz;i++)if(t.z[i]<=t.z[i-1])Terminate("Nonmonotone stellar metallicity axis");
  for(int i=1;i<t.na;i++)if(t.age[i]<=t.age[i-1])Terminate("Nonmonotone stellar age axis");
  double metals=0;
  for(int e=0;e<11;e++){if(!(f>>t.pattern[e]) || !std::isfinite(t.pattern[e]) || t.pattern[e]<0)Terminate("Invalid stellar abundance pattern");if(e>=2)metals+=t.pattern[e];}
  if(fabs(t.pattern[0]+t.pattern[1]-1)>1.e-12 || fabs(metals-1)>1.e-12)Terminate("Stellar initial abundance pattern does not normalize");
  for(int zi=0;zi<t.nz;zi++)for(int ai=0;ai<t.na;ai++)for(int ch=0;ch<3;ch++) {
    int iz,ia,ic;if(!(f>>iz>>ia>>ic) || iz!=zi || ia!=ai || ic!=ch)Terminate("Stellar table row ordering mismatch");
    auto &row=t.data[(zi*t.na+ai)*3+ch];double sum=0;
    for(int e=0;e<13;e++) {
      if(!(f>>row[e]) || !std::isfinite(row[e]) || row[e]<0)Terminate("Invalid stellar cumulative source value");
      if(e>=1 && e<=11)sum+=row[e];
      if(ai>0 && row[e]+1.e-12<t.at(zi,ai-1,ch)[e])Terminate("Stellar cumulative source regressed");
      if(ai==0 && row[e]!=0)Terminate("Stellar delayed source has a birth-time impulse");
    }
    if(fabs(sum-row[0])>1.e-12*std::max(sum,1.e-12))Terminate("Stellar source mass/element closure mismatch");
  }
  if(t.population_present){t.population.resize(size_t(t.nz)*t.na);
    for(int zi=0;zi<t.nz;zi++)for(int ai=0;ai<t.na;ai++){int iz,ia;auto &p=t.population[zi*t.na+ai];
      if(!(f>>iz>>ia>>p[0]>>p[1]) || iz!=zi || ia!=ai || !std::isfinite(p[0]+p[1]) || p[0]<0 || p[1]<0)Terminate("Invalid stellar population table");
      double retained=1-t.beta;for(int ch=0;ch<3;ch++)retained-=t.at(zi,ai,ch)[0];
      if(p[1]>retained+1.e-12 || (ai==0 && (p[0]!=0 || p[1]!=0)) || (ai>0 && p[0]<t.population[zi*t.na+ai-1][0]))Terminate("Stellar remnant/live mass closure failed");
    }}
  std::string trailing;if(f>>trailing)Terminate("Unexpected trailing stellar table data");
  for(int zi=0;zi<t.nz;zi++)for(int ai=0;ai<t.na;ai++) {
    double total=0;for(int ch=0;ch<3;ch++)total+=t.at(zi,ai,ch)[0];
    if(total>=1-t.beta)Terminate("Stellar delayed returns exhaust retained birth mass");
  }
  return t;
}

double proper_years(double before,double after) {
  if(!std::isfinite(before) || !std::isfinite(after) || before<0 || after<before)
    Terminate("Invalid stellar proper-age interval");
  if(!All.ComovingIntegrationOn)return (after-before)*All.UnitTime_in_s/seconds_per_year;
  if(before<=0)Terminate("Comoving stellar age needs a positive birth scale factor");
  if(before==after)return 0;
  // Integrate the actual native H(a), including its selected external
  // cosmology table. Delta ln(a) alone is never used as elapsed time.
  double lo=std::log(before),hi=std::log(after),previous=-1;
  for(int n=16;n<=16384;n*=2) {
    double sum=0,dx=(hi-lo)/n;
    for(int i=0;i<=n;i++) {
      double h=driftfac::hubble_function(std::exp(lo+dx*i));
      if(!std::isfinite(h) || h<=0)Terminate("Invalid native H(a) for stellar source age");
      sum+=(i==0 || i==n ? 1 : i%2 ? 4 : 2)/h;
    }
    double value=sum*dx/3;
    if(previous>=0 && fabs(value-previous)<=1.e-10*std::max(value,1.e-20))
      return value*All.UnitTime_in_s/All.HubbleParam/seconds_per_year;
    previous=value;
  }
  Terminate("Native stellar proper-age quadrature did not converge");return 0;
}

double kernel(double r,double h) {
  double q=r/h;if(q>=1)return 0;
  return 8/M_PI/(h*h*h)*(q<.5 ? 1-6*q*q+6*q*q*q : 2*std::pow(1-q,3));
}

struct star_record {
  uint64_t id,counter,model;
  MyIntPosType pos[3];
  double vel[3],mass,gross,birth,last,metallicity;
};

std::string event_line(uint64_t id,uint64_t source,uint64_t counter,int species,
    double before,double after,const double *vb,const double *va,double ub,double ua,
    const double *eb,const double *ea,double energy,double age0,double age1) {
  std::ostringstream s;s<<std::setprecision(17)<<id<<'\t'<<source<<'\t'<<counter<<'\t'<<species
    <<'\t'<<All.Time<<'\t'<<age0<<'\t'<<age1<<'\t'<<before<<'\t'<<after;
  for(int k=0;k<3;k++)s<<'\t'<<vb[k];for(int k=0;k<3;k++)s<<'\t'<<va[k];
  s<<'\t'<<ub<<'\t'<<ua;
  for(int e=0;e<11;e++)s<<'\t'<<eb[e];for(int e=0;e<11;e++)s<<'\t'<<ea[e];
  s<<'\t'<<energy<<'\n';return s.str();
}

void publish_journal(const std::vector<std::string>& rows,MPI_Comm comm) {
  uint64_t local=rows.size(),total=0;
  MPI_Allreduce(&local,&total,1,MPI_UINT64_T,MPI_SUM,comm);if(!total)return;
  if(All.StellarReturnCommitCount==UINT64_MAX || All.StellarReturnEventCount>UINT64_MAX-total)
    Terminate("Stellar source journal counter overflow");
  int rank,ranks;MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&ranks);
  uint64_t sequence=All.StellarReturnCommitCount+1,bytes=0,crc=0;
  char name[160];snprintf(name,sizeof(name),"stellar_returns_%020llu.rank_%06d.of_%06d.tsv",(unsigned long long)sequence,rank,ranks);
  std::string directory=All.OutputDir,filename=directory+"/"+name;
  int good=1;
  if(local)good=research_stellar_publish_file(filename,[&](FILE*f){
    for(const auto &row:rows)if(fwrite(row.data(),1,row.size(),f)!=row.size())return false;return true;},bytes,crc);
  int allgood=0;MPI_Allreduce(&good,&allgood,1,MPI_INT,MPI_MIN,comm);
  if(!allgood)Terminate("Stellar source payload publication failed");
  uint64_t tuple[3]={local,bytes,crc};std::vector<uint64_t> census(size_t(ranks)*3);
  MPI_Allgather(tuple,3,MPI_UINT64_T,census.data(),3,MPI_UINT64_T,comm);
  if(rank==0) {
    char commit[160];snprintf(commit,sizeof(commit),"stellar_returns_%020llu.commit.json",(unsigned long long)sequence);
    good=research_stellar_publish_file(directory+"/"+commit,[&](FILE*f){
      fprintf(f,"{\"schema\":\"native-stellar-returns-v1\",\"sequence\":%llu,\"previous_count\":%llu,\"cumulative_count\":%llu,\"event_count\":%llu,\"native_time\":%.17g,\"comoving\":%d,\"table_crc32\":%llu,\"source_fingerprint\":\"%s\",\"unit_mass_g\":%.17g,\"unit_velocity_cm_s\":%.17g,\"unit_time_s\":%.17g,\"hubble_param\":%.17g,\"mpi_ranks\":%d,\"parts\":[",
        (unsigned long long)sequence,(unsigned long long)All.StellarReturnEventCount,
        (unsigned long long)(All.StellarReturnEventCount+total),(unsigned long long)total,All.Time,All.ComovingIntegrationOn,
        (unsigned long long)table().crc,research_restart_fingerprint,All.UnitMass_in_g,All.UnitVelocity_in_cm_per_s,All.UnitTime_in_s,All.HubbleParam,ranks);
      for(int r=0;r<ranks;r++) {
        snprintf(name,sizeof(name),"stellar_returns_%020llu.rank_%06d.of_%06d.tsv",(unsigned long long)sequence,r,ranks);
        fprintf(f,"%s{\"rank\":%d,\"rows\":%llu,\"bytes\":%llu,\"crc32\":%llu,\"filename\":",r ? "," : "",r,
          (unsigned long long)census[3*r],(unsigned long long)census[3*r+1],(unsigned long long)census[3*r+2]);
        if(census[3*r])fprintf(f,"\"%s\"}",name);else fprintf(f,"null}");
      }
      fprintf(f,"]}\n");return !ferror(f);},bytes,crc);
  }
  MPI_Bcast(&good,1,MPI_INT,0,comm);if(!good)Terminate("Stellar source commit publication failed");
  All.StellarReturnCommitCount=sequence;All.StellarReturnEventCount+=total;
}
}

bool research_stellar_set_composition(research_source_state &state,double metallicity) {
  if(!std::isfinite(metallicity) || metallicity<0 || metallicity>1)return false;
  auto &t=table();double old=0;for(int e=2;e<11;e++)old+=state.element_mass_fraction[e];
  if(state.chemical_state_version==0) {
    state.element_mass_fraction[0]=(1-metallicity)*t.pattern[0];state.element_mass_fraction[1]=(1-metallicity)*t.pattern[1];
    for(int e=2;e<11;e++)state.element_mass_fraction[e]=metallicity*t.pattern[e];
    state.chemical_state_version=1;
  } else {
    if(state.chemical_state_version!=1 || metallicity+1.e-6<old)return false;
    double delta=metallicity-old,light=state.element_mass_fraction[0]+state.element_mass_fraction[1];
    if(delta>0) {
      if(delta>light)return false;
      for(int e=0;e<2;e++)state.element_mass_fraction[e]*=(light-delta)/light;
      for(int e=2;e<11;e++)state.element_mass_fraction[e]+=delta*t.pattern[e];
    }
  }
  return true;
}

double research_stellar_elapsed_years(double before,double after) {return proper_years(before,after);}

void sim::stellar_initialize_composition(void) {
  if(!All.ComovingIntegrationOn && All.HubbleParam!=1.)
    Terminate("Isolated stellar-source profile requires explicit HubbleParam=1 unit convention");
  auto &t=table();(void)t;
  for(int i=0;i<Sp.NumGas;i++)if(Sp.P[i].getType()==0 && Sp.P[i].getMass()>0)
    if(!research_stellar_set_composition(Sp.P[i].ResearchSourceState,Sp.P[i].Metallicity))
      Terminate("Gas elemental composition cannot be initialized or synchronized");
#ifdef RESEARCH_GRACKLE
  for(int i=0;i<Sp.NumGas;i++)if(Sp.P[i].getType()==0 && Sp.P[i].getMass()>0) {
    auto &state=Sp.P[i].ResearchSourceState;
    double existing=0;for(int k=0;k<6;k++)existing+=state.grackle_species_mass_fraction[k]+state.grackle_molecular_mass_fraction[k];
    if(existing>0 && !state.grackle_state_version)
      Terminate("Nonzero imported primordial species require a recorded chemical state version; explicit IC migration is needed");
    if(existing==0 && !state.grackle_state_version) {
      const double ratio=All.GracklePrimordialChemistry==3?All.GrackleDeuteriumToHydrogenRatio:0;
      try {research_primordial::seed(state,All.GrackleInitialSpecies,ratio,All.GracklePrimordialChemistry);}
      catch(const std::exception &error){Terminate("Primordial gas seed: %s",error.what());}
    }
  }
  research_grackle_initialize();
#endif
}

void sim::stellar_source_step(void) {
  auto &t=table();
  if(!std::isfinite(All.StellarSourceRadiusProperKpc) || All.StellarSourceRadiusProperKpc<=0 ||
      !std::isfinite(All.StellarSourceFraction) || All.StellarSourceFraction<=0 || All.StellarSourceFraction>.1 ||
      All.StellarMaximumCount<1 || All.StellarMaximumSubcycles<1)
    Terminate("Invalid native stellar source controls");
  All.set_cosmo_factors_for_current_time();
  const double a=All.ComovingIntegrationOn ? All.Time : 1.;
  const double h=All.ComovingIntegrationOn ? All.HubbleParam : 1.;
  const double mass_unit=All.UnitMass_in_g/h,velocity_unit=All.UnitVelocity_in_cm_per_s;
  const double radius=All.StellarSourceRadiusProperKpc*kpc*h/All.UnitLength_in_cm/a;
#ifdef PERIODIC
  if(radius>=All.BoxSize/2)Terminate("Stellar source kernel exceeds unique periodic region");
#endif
  std::vector<star_record> local;
  for(int i=0;i<Sp.NumPart;i++)if(Sp.P[i].getType()==4 && Sp.P[i].getMass()>0) {
    auto &state=Sp.P[i].ResearchSourceState;
    if(state.stellar_birth_state_version!=1)Terminate("Delayed source requires measured native SSP birth state, including legacy particles");
    if(state.stellar_model_crc32 && state.stellar_model_crc32!=t.crc)Terminate("Stellar source model changed across restart");
    if(state.stellar_last_source_time<state.stellar_birth_time || state.stellar_last_source_time>All.Time)
      Terminate("Stellar source clock outside birth/current interval");
    star_record s{};s.id=Sp.P[i].ID.get();s.mass=Sp.P[i].getMass();s.gross=state.stellar_initial_mass;
    s.birth=state.stellar_birth_time;s.last=state.stellar_last_source_time;s.metallicity=state.stellar_birth_metallicity;
    s.counter=state.stellar_event_counter;s.model=t.crc;
    for(int k=0;k<3;k++){s.pos[k]=Sp.P[i].IntPos[k];s.vel[k]=Sp.P[i].Vel[k]/a;}
    local.push_back(s);
  }
  if(local.size()>size_t(INT_MAX)/sizeof(star_record))Terminate("Stellar source MPI send overflow");
  int bytes=int(local.size()*sizeof(star_record));std::vector<int> sizes(NTask),offsets(NTask);
  MPI_Allgather(&bytes,1,MPI_INT,sizes.data(),1,MPI_INT,Communicator);
  long long total=0;for(int r=0;r<NTask;r++) {
    if(sizes[r]<0 || sizes[r]%sizeof(star_record) || total+sizes[r]>INT_MAX)Terminate("Stellar source MPI receive overflow");
    offsets[r]=int(total);total+=sizes[r];
  }
  if(total/sizeof(star_record)>size_t(All.StellarMaximumCount))Terminate("Stellar source replicated census exceeds declared maximum");
  if(double(total)*3>All.MaxMemSize*1024.*1024.*.25)
    Terminate("Replicated stellar source census exceeds its native memory allocation allowance");
  std::vector<star_record> stars(total/sizeof(star_record));
  MPI_Allgatherv(local.data(),bytes,MPI_BYTE,stars.data(),sizes.data(),offsets.data(),MPI_BYTE,Communicator);
  std::sort(stars.begin(),stars.end(),[](const star_record &x,const star_record &y){return x.id<y.id;});
  for(size_t j=1;j<stars.size();j++)if(stars[j].id==stars[j-1].id)Terminate("Duplicate live SSP identity");
  std::vector<std::string> rows;bool changed=false;size_t row_bytes=0;
  auto emit=[&](const std::string &row) {
    if(row.size()>4096 || double(row_bytes+row.size()+sizeof(std::string))>All.MaxMemSize*1024.*1024.*.125)
      Terminate("Stellar source event buffer exceeds declared native allocation allowance");
    row_bytes+=row.size()+sizeof(std::string);rows.push_back(row);
  };
  for(auto &star:stars) {
    const double age0=proper_years(star.birth,star.last),age1=proper_years(star.birth,All.Time);
    double pending[13]={};
    for(int ch=0;ch<3;ch++) {
      auto lo=t.cumulative(star.metallicity,age0,ch),hi=t.cumulative(star.metallicity,age1,ch);
      for(int e=0;e<13;e++) {
        double value=(hi[e]-lo[e])*star.gross*mass_unit/solar_mass;
        if(value<0 || !std::isfinite(value))Terminate("Invalid stellar source cumulative interval");
        pending[e]+=value;
      }
    }
    for(int e=0;e<12;e++)pending[e]*=solar_mass/mass_unit;
    pending[12]/=mass_unit*velocity_unit*velocity_unit;
    if(pending[0]>=star.mass)Terminate("Stellar source would exhaust SSP gravity mass");
    int owner=-1;for(int i=0;i<Sp.NumPart;i++)if(Sp.P[i].getType()==4 && Sp.P[i].ID.get()==star.id)owner=i;
    int subcycles=0;bool supported=true;
    while(pending[0]>0 || pending[12]>0) {
      if(++subcycles>All.StellarMaximumSubcycles)Terminate("Stellar source subcycling limit exhausted");
      std::vector<double> weights(Sp.NumGas);double sum=0,global=0;
      for(int i=0;i<Sp.NumGas;i++)if(Sp.P[i].getType()==0 && Sp.P[i].getMass()>0) {
        double d[3];Sp.nearest_image_intpos_to_pos(Sp.P[i].IntPos,star.pos,d);
        weights[i]=Sp.P[i].getMass()*kernel(std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]),radius);sum+=weights[i];
      }
      MPI_Allreduce(&sum,&global,1,MPI_DOUBLE,MPI_SUM,Communicator);
      if(global<=0){supported=false;break;}
      double fraction=1.;if(pending[0]>0)fraction=std::min(fraction,All.StellarSourceFraction*star.mass/pending[0]);
      for(int i=0;i<Sp.NumGas;i++)if(weights[i]>0) {
        double w=weights[i]/global,m=Sp.P[i].getMass(),u=Sp.get_utherm_from_entropy(i),dv2=0;
        for(int k=0;k<3;k++)dv2+=std::pow(Sp.P[i].Vel[k]/a-star.vel[k],2);
        if(pending[0]>0)fraction=std::min(fraction,All.StellarSourceFraction*m/(pending[0]*w));
        double heat=w*(pending[12]+.5*pending[0]*dv2);
        if(heat>0)fraction=std::min(fraction,All.StellarSourceFraction*m*std::max(u,All.MinEgySpec)/heat);
      }
      double applied;MPI_Allreduce(&fraction,&applied,1,MPI_DOUBLE,MPI_MIN,Communicator);
      if(!std::isfinite(applied) || applied<=0 || pending[0]*(1-applied)==pending[0] && pending[12]*(1-applied)==pending[12])
        Terminate("Stellar source substep underflow");
      if(star.counter==UINT64_MAX)Terminate("Stellar per-source event counter overflow");
      star.counter++;double emitted[13];for(int e=0;e<13;e++)emitted[e]=pending[e]*applied;
      for(int i=0;i<Sp.NumGas;i++)if(weights[i]>0) {
        auto &p=Sp.P[i];auto &state=p.ResearchSourceState;
        if(state.chemical_state_version!=1)Terminate("Stellar deposition recipient has no eleven-element state");
        double w=weights[i]/global,m=p.getMass(),dm=emitted[0]*w,newmass=m+dm;
        double before[3],after[3],eb[11],ea[11],u=Sp.get_utherm_from_entropy(i),dv2=0;
        for(int k=0;k<3;k++){before[k]=p.Vel[k]/a;after[k]=(m*before[k]+dm*star.vel[k])/newmass;dv2+=std::pow(before[k]-star.vel[k],2);}
        const double du=(emitted[12]*w+.5*m*dm/newmass*dv2)/newmass;
        double newu=(m*u)/newmass+du,metal=0;
        for(int e=0;e<11;e++){eb[e]=m*state.element_mass_fraction[e];ea[e]=eb[e]+emitted[e+1]*w;state.element_mass_fraction[e]=ea[e]/newmass;if(e>=2)metal+=state.element_mass_fraction[e];}
#ifdef RESEARCH_GRACKLE
        // Existing isotope/ion/molecule reservoirs are extensive quantities.
        // Stellar returns add neutral H/He; these yields do not replenish D.
        for(int e=0;e<6;e++){state.grackle_species_mass_fraction[e]*=m/newmass;state.grackle_molecular_mass_fraction[e]*=m/newmass;}
        state.grackle_species_mass_fraction[0]+=emitted[1]*w/newmass;
        state.grackle_species_mass_fraction[2]+=emitted[2]*w/newmass;
#endif
        p.setMass(newmass);for(int k=0;k<3;k++)p.Vel[k]=a*after[k];
        p.Metallicity=Sp.SphP[i].Metallicity=metal;Sp.SphP[i].MassMetallicity=newmass*metal;
        Sp.set_entropy_from_utherm(newu,i);Sp.SphP[i].DtEntropy=0;
#ifdef RESEARCH_GRACKLE
        research_grackle_refresh_thermodynamics(&Sp,i);
#endif
        Sp.SphP[i].set_thermodynamic_variables();
        emit(event_line(p.ID.get(),star.id,star.counter,0,m,newmass,before,after,u,newu,eb,ea,emitted[12]*w,age0,age1));
      }
      double old=star.mass;star.mass-=emitted[0];
      if(owner>=0) {
        auto &p=Sp.P[owner];auto &state=p.ResearchSourceState;
        double eb[11]={},ea[11]={};for(int e=0;e<11;e++){state.stellar_returned_elements[e]+=emitted[e+1];ea[e]=emitted[e+1];}
        p.setMass(star.mass);state.stellar_returned_mass+=emitted[0];state.stellar_injected_energy+=emitted[12];state.stellar_event_counter=star.counter;
        emit(event_line(star.id,star.id,star.counter,4,old,star.mass,star.vel,star.vel,0,0,eb,ea,emitted[12],age0,age1));
      }
      for(int e=0;e<13;e++)pending[e]=applied==1 ? 0 : pending[e]*(1-applied);
      changed=true;
    }
    if(supported && owner>=0) {
      Sp.P[owner].ResearchSourceState.stellar_last_source_time=All.Time;
      Sp.P[owner].ResearchSourceState.stellar_model_crc32=t.crc;
      auto &state=Sp.P[owner].ResearchSourceState;
      if(t.population_present){state.stellar_remnant_mass=star.gross*t.remnant(star.metallicity,age1);
        if(state.stellar_remnant_mass>star.mass*(1+1.e-12))Terminate("Native remnant exceeds actual SSP mass");
        state.stellar_live_mass=std::max(0.,star.mass-state.stellar_remnant_mass);state.stellar_population_version=1;
      }

    } else if(!supported) {
      // A partially applied source cannot be retried using the same cumulative
      // interval. Abort before a checkpoint instead of duplicating its ejecta.
      if(subcycles>1)Terminate("Stellar recipient support vanished during source subcycling");
      mpi_printf("STELLAR_SOURCE_PENDING time=%.17g source=%llu reason=no_gas_support\n",All.Time,(unsigned long long)star.id);
    }
  }
  publish_journal(rows,Communicator);
  if(changed) {
    // Rebuild cached densities after mass transfer, preserving specific gas
    // energy through the native neighbor solve and MPI domain migration.
    for(int i=0;i<Sp.NumGas;i++)if(Sp.P[i].getType()==0)Sp.P[i].ResearchSourceState.source_internal_energy=Sp.get_utherm_from_entropy(i);
    NgbTree.treefree();Domain.domain_free();Domain.domain_decomposition(STANDARD);
    NgbTree.treeallocate(Sp.NumGas,&Sp,&Domain);NgbTree.treebuild(Sp.NumGas,NULL);NgbTree.compute_densities();NgbTree.update_maxhsml();
    for(int i=0;i<Sp.NumGas;i++)if(Sp.P[i].getType()==0) {
      Sp.set_entropy_from_utherm(Sp.P[i].ResearchSourceState.source_internal_energy,i);Sp.SphP[i].set_thermodynamic_variables();
    }
  }
}
#endif
