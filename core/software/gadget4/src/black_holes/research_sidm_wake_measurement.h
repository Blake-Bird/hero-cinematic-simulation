/* Read-only native fine-DM aperture monopoles at a synchronized BH probe.
 * Calls the actual gravtree softening kernel. No tree/PM/Ewald equivalence
 * or calibrated production-force enablement is inferred from this diagnostic.
 */
#ifndef RESEARCH_SIDM_WAKE_MEASUREMENT_H
#define RESEARCH_SIDM_WAKE_MEASUREMENT_H
#include "gadgetconfig.h"
#include "research_sidm_drag_source.h"
#include "../gravtree/gravtree.h"
#include "../gravity/ewald.h"
#include <type_traits>
#include <limits>
namespace research_sidm_wake_measurement {
struct shell {
  std::uint64_t particle_count=0;
  double mass_msun=0,minimum_support_proper_kpc=0,maximum_support_proper_kpc=0;
  research_sidm_drag_source::vec acceleration_cm_s2{};
  research_sidm_drag_source::vec periodic_correction_cm_s2{};
  research_sidm_drag_source::vec pm_short_range_cm_s2{};
};
struct measurement {
  std::uint64_t carrier_id=0,complete_dm_count=0,pair_evaluations=0;
  double native_time=0,physical_bh_mass_msun=0,gravity_carrier_mass_msun=0;
  research_sidm_drag_source::vec probe_position_proper_kpc{},stored_gravity_acceleration_cm_s2{};
  std::vector<shell> shells;
  bool periodic_correction_available=false;
  bool pm_short_range_available=false;
  bool closed_total_comparison_available=false;
  std::array<std::uint64_t,7> source_species_counts{};
  research_sidm_drag_source::vec full_dm_monopoles_cm_s2{},full_dm_periodic_correction_cm_s2{};
};
template<class Particles,class Gravity,class Softening>
measurement measure(Particles&particles,Gravity&gravity,Softening softening,MPI_Comm comm,
    std::uint64_t carrier_id,long long timeline,double native_time,const std::vector<double>&edges,
    double mass_to_msun,double length_to_proper_kpc,double native_acceleration_to_cgs,double native_gravity_constant,
    std::uint64_t maximum_particles,std::uint64_t maximum_work,std::uint64_t maximum_workspace_bytes)
{
  using namespace research_sidm_drag_source;
  using position_type=typename std::remove_reference<decltype(particles.P[0].IntPos[0])>::type;
  bool okay=carrier_id!=0&&timeline>=0&&edges.size()>=2&&edges.size()<=4096&&std::isfinite(native_time)&&
    std::isfinite(mass_to_msun)&&mass_to_msun>0&&std::isfinite(length_to_proper_kpc)&&length_to_proper_kpc>0&&
    std::isfinite(native_acceleration_to_cgs)&&native_acceleration_to_cgs>0&&std::isfinite(native_gravity_constant)&&native_gravity_constant>0&&maximum_particles>0&&maximum_work>0;
  for(std::size_t i=0;i<edges.size();i++)okay=okay&&std::isfinite(edges[i])&&edges[i]>=0&&(!i||edges[i]>edges[i-1]);
  require_collective(okay,comm,"Invalid bounded native wake measurement inputs");
  std::array<unsigned long long,6> bounds{{carrier_id,maximum_particles,maximum_work,maximum_workspace_bytes,edges.size(),static_cast<unsigned long long>(timeline)}},lo{},hi{};
  MPI_Allreduce(bounds.data(),lo.data(),6,MPI_UNSIGNED_LONG_LONG,MPI_MIN,comm);MPI_Allreduce(bounds.data(),hi.data(),6,MPI_UNSIGNED_LONG_LONG,MPI_MAX,comm);
  std::array<double,5> units{{native_time,mass_to_msun,length_to_proper_kpc,native_acceleration_to_cgs,native_gravity_constant}},units_lo{},units_hi{};
  MPI_Allreduce(units.data(),units_lo.data(),5,MPI_DOUBLE,MPI_MIN,comm);MPI_Allreduce(units.data(),units_hi.data(),5,MPI_DOUBLE,MPI_MAX,comm);
  require_collective(lo==hi&&units_lo==units_hi,comm,"MPI native wake inputs disagree");
  int periodic_mode=0,periodic_low=0,periodic_high=0;
#ifdef PERIODIC
  periodic_mode=gravity.DoEwald?1:0;
#endif
  MPI_Allreduce(&periodic_mode,&periodic_low,1,MPI_INT,MPI_MIN,comm);MPI_Allreduce(&periodic_mode,&periodic_high,1,MPI_INT,MPI_MAX,comm);
  require_collective(periodic_low==periodic_high,comm,"MPI native periodic force modes disagree");
  int pm_mode=0,pm_low=0,pm_high=0;
#ifdef PMGRID
  pm_mode=gravity.DoPM;
#endif
  MPI_Allreduce(&pm_mode,&pm_low,1,MPI_INT,MPI_MIN,comm);MPI_Allreduce(&pm_mode,&pm_high,1,MPI_INT,MPI_MAX,comm);
  require_collective(pm_low==pm_high,comm,"MPI native PM force modes disagree");
  bool pm_available=false;
#ifdef PMGRID
  pm_available=(pm_mode&(TREE_ACTIVE_CUTTOFF_BASE_PM|TREE_ACTIVE_CUTTOFF_HIGHRES_PM))!=0;
#endif
  int rank=0,ranks=0;MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&ranks);
  int index=-1,local_owner=0,total_owner=0,owner_rank=-1;
  std::uint64_t local_dm=0;
  std::array<unsigned long long,7> local_species{},global_species{};
  for(int i=0;i<particles.NumPart;i++){if(particles.P[i].ID.get()==carrier_id){index=i;local_owner++;}unsigned kind=particles.P[i].getType();local_species[std::min(kind,6u)]++;if(kind==1)local_dm++;}
  MPI_Allreduce(local_species.data(),global_species.data(),7,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);
  MPI_Allreduce(&local_owner,&total_owner,1,MPI_INT,MPI_SUM,comm);int owner=index>=0?rank:-1;MPI_Allreduce(&owner,&owner_rank,1,MPI_INT,MPI_MAX,comm);
  unsigned long long local=local_dm,total=0;MPI_Allreduce(&local,&total,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);
  const std::size_t bins=edges.size()-1;
  require_collective(total_owner==1&&total>0&&total<=maximum_particles&&total<=maximum_work/(1+periodic_mode+int(pm_available))&&total<=INT_MAX&&
    32.L*total+4096.L*bins+64.L*ranks<=maximum_workspace_bytes,comm,"Native wake census/workspace exceeds its complete declared bounds");
  struct probe {std::array<position_type,3> integer_position;vec position,stored_acceleration;double softening,mass,physical;int inside=0;bool gravity_active=false;};
  probe state{};
  if(index>=0){auto&p=particles.P[index];double position[3];particles.intpos_to_pos(p.IntPos,position);
    okay=p.getType()==5&&p.get_Ti_Current()==timeline&&p.getMass()>0&&p.BHMass>0;
    state.mass=p.getMass()*mass_to_msun;state.physical=p.BHMass*mass_to_msun;state.softening=softening(p);
    for(int i=0;i<particles.TimeBinsGravity.NActiveParticles;i++)if(particles.TimeBinsGravity.ActiveParticleList[i]==index)state.gravity_active=true;
#ifdef RESEARCH_BH_WAKE_DIAGNOSTIC
    state.gravity_active=state.gravity_active&&p.ResearchWakeGravityFrameVersion==1&&p.ResearchWakeGravityTimeline==timeline&&
      p.ResearchWakeGravityType==5&&p.ResearchWakeGravityMass==p.getMass()&&p.ResearchWakeGravitySoftening==state.softening;
    for(int k=0;k<3;k++)state.gravity_active=state.gravity_active&&p.ResearchWakeGravityPosition[k]==p.IntPos[k]&&p.ResearchWakeGravityAcceleration[k]==p.GravAccel[k];
#else
    state.gravity_active=false;
#endif
#if defined(PMGRID) && defined(PLACEHIGHRESREGION)
    state.inside=p.InsideOutsideFlag;
#endif
    for(int k=0;k<3;k++){state.integer_position[k]=p.IntPos[k];state.position[k]=position[k]*length_to_proper_kpc;state.stored_acceleration[k]=p.GravAccel[k]*native_acceleration_to_cgs;}}
  require_collective(okay,comm,"Native wake probe is not live and synchronized");
  static_assert(std::is_trivially_copyable<probe>::value,"Native wake probe broadcast cannot own pointers");
  MPI_Bcast(&state,sizeof(state),MPI_BYTE,owner_rank,comm);
  require_collective(finite(state.position)&&finite(state.stored_acceleration)&&std::isfinite(state.softening)&&state.softening>0&&
    std::isfinite(state.mass)&&state.mass>0&&std::isfinite(state.physical)&&state.physical>0,
    comm,"Native wake probe softening/phase is invalid");
  std::vector<unsigned long long> ids,all_ids;std::vector<int> counts,offsets;std::vector<double> wire,wire_lo,wire_hi,values;
  std::vector<unsigned long long> shell_counts;measurement result;bool allocated=true;
  std::vector<double> minima,maxima,global_values,global_minima,global_maxima;std::vector<unsigned long long> global_counts;
  try{ids.reserve(local_dm);all_ids.resize(total);counts.resize(ranks);offsets.resize(ranks);wire=edges;wire_lo.resize(edges.size());wire_hi.resize(edges.size());
    values.assign(10*bins,0);shell_counts.assign(bins,0);result.shells.resize(bins);
    minima.assign(bins,std::numeric_limits<double>::infinity());maxima.assign(bins,0);global_values.resize(10*bins);
    global_minima.resize(bins);global_maxima.resize(bins);global_counts.resize(bins);}
  catch(const std::exception&){allocated=false;}
  require_collective(allocated,comm,"Native wake measurement allocation failed collectively");
  MPI_Allreduce(wire.data(),wire_lo.data(),int(edges.size()),MPI_DOUBLE,MPI_MIN,comm);MPI_Allreduce(wire.data(),wire_hi.data(),int(edges.size()),MPI_DOUBLE,MPI_MAX,comm);
  require_collective(wire_lo==wire_hi,comm,"MPI native wake shell edges disagree");
  for(int i=0;i<particles.NumPart;i++)if(particles.P[i].getType()==1)ids.push_back(particles.P[i].ID.get());
  int count=int(ids.size());MPI_Allgather(&count,1,MPI_INT,counts.data(),1,MPI_INT,comm);int offset=0;
  for(int i=0;i<ranks;i++){offsets[i]=offset;offset+=counts[i];}
  MPI_Allgatherv(ids.data(),count,MPI_UNSIGNED_LONG_LONG,all_ids.data(),counts.data(),offsets.data(),MPI_UNSIGNED_LONG_LONG,comm);
  std::sort(all_ids.begin(),all_ids.end());require_collective(all_ids.front()!=0&&std::adjacent_find(all_ids.begin(),all_ids.end())==all_ids.end(),comm,"Native wake fine-DM identities alias globally");
  std::array<double,6> full_local{},full_global{};
  for(int i=0;i<particles.NumPart;i++){auto&p=particles.P[i];if(p.getType()!=1)continue;
    double delta[3];particles.nearest_image_intpos_to_pos(p.IntPos,state.integer_position.data(),delta);double radius=std::hypot(delta[0],std::hypot(delta[1],delta[2]));
    double proper=radius*length_to_proper_kpc;okay=okay&&p.get_Ti_Current()==timeline&&std::isfinite(radius)&&std::isfinite(proper)&&std::isfinite(p.getMass())&&p.getMass()>0;
    if(!okay)continue;
    auto at=std::upper_bound(edges.begin(),edges.end(),proper);
    bool aperture=at!=edges.begin()&&at!=edges.end();std::size_t bin=aperture?std::size_t(at-edges.begin()-1):0;
    double source_h=softening(p),h=std::max(state.softening,source_h);okay=okay&&std::isfinite(source_h)&&source_h>=0&&std::isfinite(h)&&h>0&&std::isfinite(p.getMass())&&p.getMass()>0;
    if(!okay)continue;
    typename Gravity::gfactors factors;double inverse=radius>0?1/radius:0;
    gravity.get_gfactors_monopole(factors,radius,h,inverse);
    if(aperture)values[10*bin]+=p.getMass()*mass_to_msun;
    for(int k=0;k<3;k++){double value=-p.getMass()*factors.fac1*inverse*delta[k]*native_gravity_constant*native_acceleration_to_cgs;
      full_local[k]+=value;if(aperture)values[10*bin+1+k]+=value;}
#ifdef PERIODIC
    if(periodic_mode){
      ewald_data correction;Ewald.ewald_gridlookup(p.IntPos,state.integer_position.data(),ewald::POINTMASS,correction);
      for(int k=0;k<3;k++){double value=p.getMass()*correction.D1phi[k]*native_gravity_constant*native_acceleration_to_cgs;
        full_local[k+3]+=value;if(aperture)values[10*bin+4+k]+=value;}
    }
#endif
#ifdef PMGRID
    if(pm_available&&aperture){
      const typename Gravity::mesh_factors*mesh=&gravity.mf[LOW_MESH];
#ifdef PLACEHIGHRESREGION
      if((pm_mode&TREE_ACTIVE_CUTTOFF_HIGHRES_PM)&&p.InsideOutsideFlag==FLAG_INSIDE&&state.inside==FLAG_INSIDE)mesh=&gravity.mf[HIGH_MESH];
#endif
      typename Gravity::gfactors short_factors;
      if(!gravity.modify_gfactors_pm_monopole(short_factors,radius,inverse,mesh)){
        gravity.get_gfactors_monopole(short_factors,radius,h,inverse);
        for(int k=0;k<3;k++)values[10*bin+7+k]+=-p.getMass()*short_factors.fac1*inverse*delta[k]*native_gravity_constant*native_acceleration_to_cgs;
      }
    }
#endif
    if(aperture){shell_counts[bin]++;minima[bin]=std::min(minima[bin],h*length_to_proper_kpc);maxima[bin]=std::max(maxima[bin],h*length_to_proper_kpc);}
  }
  require_collective(okay,comm,"Native wake aperture includes stale or invalid fine-DM state");
  MPI_Allreduce(full_local.data(),full_global.data(),6,MPI_DOUBLE,MPI_SUM,comm);
  for(double value:full_global)okay=okay&&std::isfinite(value);
  require_collective(okay,comm,"Full fine-DM native force reductions overflow");
  MPI_Allreduce(values.data(),global_values.data(),int(values.size()),MPI_DOUBLE,MPI_SUM,comm);
  MPI_Allreduce(shell_counts.data(),global_counts.data(),int(bins),MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);
  MPI_Allreduce(minima.data(),global_minima.data(),int(bins),MPI_DOUBLE,MPI_MIN,comm);MPI_Allreduce(maxima.data(),global_maxima.data(),int(bins),MPI_DOUBLE,MPI_MAX,comm);
  for(std::size_t i=0;i<bins;i++){auto&s=result.shells[i];s.particle_count=global_counts[i];s.mass_msun=global_values[10*i];
    for(int k=0;k<3;k++){s.acceleration_cm_s2[k]=global_values[10*i+1+k];s.periodic_correction_cm_s2[k]=global_values[10*i+4+k];s.pm_short_range_cm_s2[k]=global_values[10*i+7+k];}
    s.minimum_support_proper_kpc=s.particle_count?global_minima[i]:0;s.maximum_support_proper_kpc=global_maxima[i];
    okay=okay&&std::isfinite(s.mass_msun)&&finite(s.acceleration_cm_s2)&&finite(s.periodic_correction_cm_s2)&&finite(s.pm_short_range_cm_s2)&&std::isfinite(s.minimum_support_proper_kpc)&&std::isfinite(s.maximum_support_proper_kpc);}
  require_collective(okay,comm,"Native wake reductions overflow");
  result.carrier_id=carrier_id;result.complete_dm_count=total;result.native_time=native_time;
  for(const auto&s:result.shells)result.pair_evaluations+=s.particle_count;
  result.physical_bh_mass_msun=state.physical;result.gravity_carrier_mass_msun=state.mass;
  result.probe_position_proper_kpc=state.position;result.stored_gravity_acceleration_cm_s2=state.stored_acceleration;
  result.periodic_correction_available=periodic_mode!=0;result.pm_short_range_available=pm_available;
  for(int k=0;k<3;k++){result.full_dm_monopoles_cm_s2[k]=full_global[k];result.full_dm_periodic_correction_cm_s2[k]=full_global[k+3];}
  for(int k=0;k<7;k++)result.source_species_counts[k]=global_species[k];
  result.closed_total_comparison_available=state.gravity_active&&periodic_mode&&global_species[5]==1;
  for(int kind:{0,2,3,4,6})result.closed_total_comparison_available=result.closed_total_comparison_available&&global_species[kind]==0;
#if defined(PMGRID) || defined(EXTERNALGRAVITY) || defined(HIERARCHICAL_GRAVITY)
  result.closed_total_comparison_available=false;
#endif
  return result;
}
}
#endif
