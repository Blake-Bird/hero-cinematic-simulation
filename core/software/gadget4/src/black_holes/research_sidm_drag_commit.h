/* Native precision-aware transactional source commit. No scheduling policy.
 * Prepare all allocations, native casts and rounded conservation before any
 * particle write. Commit rechecks exact live bindings collectively, then uses
 * only scalar assignments. The caller must hold particle ownership stable.
 */
#ifndef RESEARCH_SIDM_DRAG_COMMIT_H
#define RESEARCH_SIDM_DRAG_COMMIT_H
#include "research_sidm_drag_coupling.h"
#include <type_traits>
#include <utility>
#include <cstring>
namespace research_sidm_drag_commit {
using research_sidm_drag_source::vec;
template<class Velocity,class Position> struct update {
  int index;std::uint64_t id;double native_mass;int kind;
  std::array<Position,3> position;
  std::array<Velocity,3> previous,next;
};
template<class Velocity,class Position> struct packet {
  std::vector<update<Velocity,Position>> local;
  int carrier_index=-1;
  research_source_state previous_state{},next_state{};
  vec rounded_momentum_residual{},rounded_angular_residual{};
  double rounded_energy_residual=0;
  double momentum_budget=0,angular_budget=0,energy_budget=0;
};
template<class Particles,class Position> auto prepare(Particles&particles,const std::vector<int>&recipient_indices,
    const std::vector<std::array<Position,3>>&original_native_positions,int carrier_index,const std::array<Position,3>&original_carrier_native_position,
    std::uint64_t carrier_id,const std::vector<research_sidm_drag_source::bath_particle>&original_bath,
    const research_sidm_drag_coupling::staged_result&staged,const research_source_state&previous_state,
    const research_source_state&next_state,double native_mass_to_msun,double native_velocity_to_km_s,
    const research_sidm_drag_source::limits&bound,MPI_Comm comm)
 -> packet<typename std::remove_reference<decltype(particles.P[0].Vel[0])>::type,
           typename std::remove_reference<decltype(particles.P[0].IntPos[0])>::type> {
  using namespace research_sidm_drag_source;
  using velocity_type=typename std::remove_reference<decltype(particles.P[0].Vel[0])>::type;
  using position_type=typename std::remove_reference<decltype(particles.P[0].IntPos[0])>::type;
  static_assert(std::is_same<Position,position_type>::value,"Drag position bindings must preserve the exact native integer type");
  packet<velocity_type,position_type> out{};
  bool okay=std::isfinite(native_mass_to_msun)&&native_mass_to_msun>0&&std::isfinite(native_velocity_to_km_s)&&native_velocity_to_km_s>0&&
    recipient_indices.size()==original_bath.size()&&original_native_positions.size()==original_bath.size()&&original_bath.size()==staged.conservative.local_velocities.size()&&
    carrier_index>=-1&&carrier_index<particles.NumPart&&carrier_id!=0&&original_bath.size()<=bound.maximum_particles&&
    original_bath.size()<=bound.maximum_workspace_bytes/1024&&previous_state.bh_sidm_drag_counter==staged.source_counter&&
    next_state.bh_sidm_drag_counter==staged.next_source_counter&&next_state.bh_sidm_drag_version==1&&
    previous_state.bh_sidm_drag_last_proper_seconds==staged.proper_start_seconds&&next_state.bh_sidm_drag_last_proper_seconds==staged.proper_end_seconds&&
    previous_state.bh_sidm_drag_model_crc32==staged.response_checksum&&next_state.bh_sidm_drag_model_crc32==staged.response_checksum;
  for(int i=0;i<4;i++)okay=okay&&previous_state.bh_sidm_drag_model_sha256[i]==staged.response_model_words[i]&&
    next_state.bh_sidm_drag_model_sha256[i]==staged.response_model_words[i]&&previous_state.bh_sidm_drag_signature_sha256[i]==staged.response_signature_words[i]&&
    next_state.bh_sidm_drag_signature_sha256[i]==staged.response_signature_words[i];
  require_collective(okay,comm,"Invalid native drag commit units, census, state or workspace");
  std::array<double,2> units{{native_mass_to_msun,native_velocity_to_km_s}},low{},high{};
  MPI_Allreduce(units.data(),low.data(),2,MPI_DOUBLE,MPI_MIN,comm);MPI_Allreduce(units.data(),high.data(),2,MPI_DOUBLE,MPI_MAX,comm);
  require_collective(low==high,comm,"MPI owners disagree on native drag commit conversion");
  bool allocated=true;std::vector<int> checked_indices;
  try{out.local.resize(recipient_indices.size()+(carrier_index>=0?1:0));checked_indices=recipient_indices;if(carrier_index>=0)checked_indices.push_back(carrier_index);}catch(const std::exception&){allocated=false;}
  require_collective(allocated,comm,"Native drag commit workspace allocation failed");
  int owner=carrier_index>=0?1:0,owners=0;MPI_Allreduce(&owner,&owners,1,MPI_INT,MPI_SUM,comm);
  require_collective(owners==1,comm,"Native drag carrier must have exactly one MPI owner");
  auto bind=[&](std::size_t row,int index,std::uint64_t id,int kind,double mass,const vec&old_velocity,const vec&new_velocity,const std::array<Position,3>&old_position){
    if(index<0||index>=particles.NumPart){okay=false;return;}
    const auto&p=particles.P[index];auto&target=out.local[row];
    target.index=index;target.id=id;target.kind=kind;target.native_mass=p.getMass();
    okay=okay&&p.ID.get()==id&&p.getType()==kind&&p.getMass()*native_mass_to_msun==mass;
    for(int k=0;k<3;k++){
      target.position[k]=old_position[k];target.previous[k]=p.Vel[k];okay=okay&&p.IntPos[k]==old_position[k];
      okay=okay&&double(p.Vel[k])*native_velocity_to_km_s==old_velocity[k];
      double converted=new_velocity[k]/native_velocity_to_km_s;
      if(!std::isfinite(converted)||std::abs(converted)>std::numeric_limits<velocity_type>::max()){okay=false;continue;}
      target.next[k]=static_cast<velocity_type>(converted);okay=okay&&std::isfinite(double(target.next[k]));
    }
  };
  for(std::size_t i=0;i<recipient_indices.size();i++)bind(i,recipient_indices[i],original_bath[i].id,1,original_bath[i].mass,
    original_bath[i].velocity,staged.conservative.local_velocities[i],original_native_positions[i]);
  if(carrier_index>=0){bind(recipient_indices.size(),carrier_index,carrier_id,5,staged.gravity_carrier_mass_msun,
    staged.original_carrier_velocity_km_s,staged.conservative.carrier_velocity,original_carrier_native_position);
    okay=okay&&std::memcmp(&particles.P[carrier_index].ResearchSourceState,&previous_state,sizeof(previous_state))==0;}
  std::sort(checked_indices.begin(),checked_indices.end());
  okay=okay&&std::adjacent_find(checked_indices.begin(),checked_indices.end())==checked_indices.end();
  require_collective(okay,comm,"Native drag commit changed an original particle identity, mass, type, velocity or source state");
  std::array<double,4> local_moments{};
  for(const auto&p:original_bath){local_moments[0]+=p.mass;for(int k=0;k<3;k++)local_moments[k+1]+=p.mass*p.velocity[k];}
  auto moments=sum(local_moments,comm);double total_mass=moments[0]+staged.gravity_carrier_mass_msun;
  require_collective(std::isfinite(total_mass)&&total_mass>0,comm,"Native drag commit has invalid global mass");
  vec com{};for(int k=0;k<3;k++)com[k]=(moments[k+1]+staged.gravity_carrier_mass_msun*staged.original_carrier_velocity_km_s[k])/total_mass;
  std::array<double,10> residual{};
  auto measure=[&](const vec&old_velocity,const vec&new_velocity,const vec&relative_position,double mass){
    vec dv=add(new_velocity,scale(old_velocity,-1)),impulse=scale(dv,mass),angular=cross(relative_position,impulse);
    for(int k=0;k<3;k++){residual[k]+=impulse[k];residual[k+3]+=angular[k];}
    double work=mass*(dot(add(old_velocity,scale(com,-1)),dv)+.5*dot(dv,dv));
    residual[6]+=work;residual[7]+=norm(impulse);residual[8]+=norm(angular);residual[9]+=std::abs(work);
  };
  for(std::size_t i=0;i<original_bath.size();i++){
    vec rounded{};for(int k=0;k<3;k++)rounded[k]=double(out.local[i].next[k])*native_velocity_to_km_s;
    measure(original_bath[i].velocity,rounded,add(original_bath[i].position,scale(staged.original_carrier_position_proper_kpc,-1)),original_bath[i].mass);
  }
  if(carrier_index>=0){vec rounded{};for(int k=0;k<3;k++)rounded[k]=double(out.local.back().next[k])*native_velocity_to_km_s;
    measure(staged.original_carrier_velocity_km_s,rounded,vec{},staged.gravity_carrier_mass_msun);}
  auto global=sum(residual,comm);for(double x:global)okay=okay&&std::isfinite(x);
  out.rounded_momentum_residual={{global[0],global[1],global[2]}};out.rounded_angular_residual={{global[3],global[4],global[5]}};out.rounded_energy_residual=global[6];
  out.momentum_budget=bound.momentum_floor+bound.relative_tolerance*global[7];out.angular_budget=bound.angular_floor+bound.relative_tolerance*global[8];
  out.energy_budget=bound.energy_floor+bound.relative_tolerance*global[9];
  okay=okay&&std::isfinite(out.momentum_budget)&&std::isfinite(out.angular_budget)&&std::isfinite(out.energy_budget)&&
    norm(out.rounded_momentum_residual)<=out.momentum_budget&&norm(out.rounded_angular_residual)<=out.angular_budget&&std::abs(global[6])<=out.energy_budget;
  require_collective(okay,comm,"Rounded native storage violates calibrated source conservation budgets");
  out.carrier_index=carrier_index;out.previous_state=previous_state;out.next_state=next_state;return out;
}
template<class Particles,class Velocity,class Position> void commit(Particles&particles,const packet<Velocity,Position>&prepared,MPI_Comm comm){
  using research_sidm_drag_source::require_collective;
  bool okay=true;
  for(const auto&entry:prepared.local){
    if(entry.index<0||entry.index>=particles.NumPart){okay=false;continue;}
    const auto&p=particles.P[entry.index];okay=okay&&p.ID.get()==entry.id&&p.getType()==entry.kind&&p.getMass()==entry.native_mass;
    for(int k=0;k<3;k++)okay=okay&&p.IntPos[k]==entry.position[k]&&p.Vel[k]==entry.previous[k];
  }
  if(prepared.carrier_index>=0){
    okay=okay&&prepared.carrier_index<particles.NumPart;
    if(prepared.carrier_index<particles.NumPart)okay=okay&&std::memcmp(&particles.P[prepared.carrier_index].ResearchSourceState,&prepared.previous_state,sizeof(research_source_state))==0;
  }
  require_collective(okay,comm,"Native drag commit is stale; no particles were modified");
  // Ownership/reordering must remain locked by the caller across this region.
  // No allocation, callback, file I/O or throwing calculation follows preflight.
  for(const auto&entry:prepared.local)for(int k=0;k<3;k++)particles.P[entry.index].Vel[k]=entry.next[k];
  if(prepared.carrier_index>=0)particles.P[prepared.carrier_index].ResearchSourceState=prepared.next_state;
}
}
#endif
