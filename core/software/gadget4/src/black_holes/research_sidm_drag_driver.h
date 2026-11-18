/* Actual native particle extraction -> calibrated stage -> rounded commit.
 * A production selector supplies source-bound context and an initialized
 * model clock. This entry point does not infer background roles or enable a
 * new force model. Ownership and synchronized drift remain caller contracts.
 */
#ifndef RESEARCH_SIDM_DRAG_DRIVER_H
#define RESEARCH_SIDM_DRAG_DRIVER_H
#include "research_sidm_drag_commit.h"
#include "research_sidm_drag_schedule.h"
namespace research_sidm_drag_driver {
struct context {
  std::string actual_fixed_signature;
  std::uint64_t carrier_id,seed;
  long long native_timeline;
  double native_begin,native_end,proper_begin_seconds,proper_end_seconds;
  double proper_clock_absolute_tolerance_seconds,proper_clock_relative_tolerance;
  double native_mass_to_msun,native_velocity_to_km_s,native_length_to_proper_kpc,proper_box_kpc,erg_per_msun_km2_s2;
  double expected_physical_bh_mass_msun,expected_gravity_carrier_mass_msun;
  research_sidm_drag_source::vec expected_probe_position_proper_kpc,background_mean_velocity_km_s;
  double background_density_msun_kpc3;
  std::array<double,6> background_covariance_km2_s2;
  double recipient_inner_radius_proper_kpc,recipient_outer_radius_proper_kpc,recipient_gravity_mass_msun;
  research_sidm_drag_coupling::source_limits limits;
};
struct diagnostics {
  std::uint64_t carrier_id,source_counter,next_source_counter,recipient_count;
  double proper_begin_seconds,proper_end_seconds,heat_msun_km2_s2,rounded_energy_residual;
  double momentum_budget,angular_budget,energy_budget,reference_error_cm_s2,heldout_error_cm_s2;
  research_sidm_drag_source::vec acceleration_cm_s2,rounded_momentum_residual,rounded_angular_residual;
};
template<class Particles> diagnostics step(Particles&particles,const research_sidm_drag_response::state_response&response,
    const context&ctx,MPI_Comm comm){
  using namespace research_sidm_drag_source;
  using position_type=typename std::remove_reference<decltype(particles.P[0].IntPos[0])>::type;
  bool okay=response.sha256_verified()&&ctx.carrier_id!=0&&std::isfinite(ctx.native_mass_to_msun)&&ctx.native_mass_to_msun>0&&
    std::isfinite(ctx.native_velocity_to_km_s)&&ctx.native_velocity_to_km_s>0&&std::isfinite(ctx.native_length_to_proper_kpc)&&ctx.native_length_to_proper_kpc>0&&
    std::isfinite(ctx.proper_box_kpc)&&ctx.proper_box_kpc>0&&std::isfinite(ctx.recipient_inner_radius_proper_kpc)&&ctx.recipient_inner_radius_proper_kpc>=0&&
    std::isfinite(ctx.recipient_outer_radius_proper_kpc)&&ctx.recipient_outer_radius_proper_kpc>ctx.recipient_inner_radius_proper_kpc&&
    ctx.recipient_outer_radius_proper_kpc<ctx.proper_box_kpc/2&&std::isfinite(ctx.recipient_gravity_mass_msun)&&ctx.recipient_gravity_mass_msun>0;
  require_collective(okay,comm,"Native calibrated source context has invalid physical units or aperture");
  std::array<double,13> settings{{ctx.native_mass_to_msun,ctx.native_velocity_to_km_s,ctx.native_length_to_proper_kpc,ctx.proper_box_kpc,
    ctx.recipient_inner_radius_proper_kpc,ctx.recipient_outer_radius_proper_kpc,ctx.recipient_gravity_mass_msun,
    ctx.expected_physical_bh_mass_msun,ctx.expected_gravity_carrier_mass_msun,ctx.expected_probe_position_proper_kpc[0],
    ctx.expected_probe_position_proper_kpc[1],ctx.expected_probe_position_proper_kpc[2],ctx.erg_per_msun_km2_s2}},settings_low{},settings_high{};
  for(double x:settings)okay=okay&&std::isfinite(x);
  require_collective(okay&&ctx.native_timeline>=0,comm,"Native calibrated context contains nonfinite state or invalid timeline");
  MPI_Allreduce(settings.data(),settings_low.data(),13,MPI_DOUBLE,MPI_MIN,comm);MPI_Allreduce(settings.data(),settings_high.data(),13,MPI_DOUBLE,MPI_MAX,comm);
  require_collective(settings_low==settings_high,comm,"MPI owners disagree on native source units, aperture or calibrated BH state");
  std::array<unsigned long long,3> keys{{ctx.carrier_id,ctx.seed,static_cast<unsigned long long>(ctx.native_timeline)}},keys_low{},keys_high{};
  MPI_Allreduce(keys.data(),keys_low.data(),3,MPI_UNSIGNED_LONG_LONG,MPI_MIN,comm);MPI_Allreduce(keys.data(),keys_high.data(),3,MPI_UNSIGNED_LONG_LONG,MPI_MAX,comm);
  require_collective(keys_low==keys_high,comm,"MPI owners disagree on native source identity, seed or timeline");
  int owner_index=-1,owner_rank=-1,rank=0;MPI_Comm_rank(comm,&rank);
  for(int i=0;i<particles.NumPart;i++)if(particles.P[i].ID.get()==ctx.carrier_id){if(owner_index>=0)okay=false;owner_index=i;}
  int owner_count=owner_index>=0?1:0,total_owners=0;MPI_Allreduce(&owner_count,&total_owners,1,MPI_INT,MPI_SUM,comm);
  int local_owner=owner_index>=0?rank:-1;MPI_Allreduce(&local_owner,&owner_rank,1,MPI_INT,MPI_MAX,comm);
  require_collective(okay&&total_owners==1,comm,"Calibrated native carrier identity has no unique owner");
  struct carrier_record {std::array<position_type,3> native_position;vec position,velocity;double mass,physical;research_source_state state;};
  carrier_record carrier{};
  if(owner_index>=0){auto&p=particles.P[owner_index];okay=p.getType()==5&&p.getMass()>0&&p.BHMass>0&&p.get_Ti_Current()==ctx.native_timeline;
    carrier.state=p.ResearchSourceState;carrier.mass=p.getMass()*ctx.native_mass_to_msun;carrier.physical=p.BHMass*ctx.native_mass_to_msun;
    double position[3];particles.intpos_to_pos(p.IntPos,position);
    for(int k=0;k<3;k++){carrier.native_position[k]=p.IntPos[k];carrier.position[k]=position[k]*ctx.native_length_to_proper_kpc;carrier.velocity[k]=p.Vel[k]*ctx.native_velocity_to_km_s;}}
  require_collective(okay,comm,"Native calibrated carrier is not live or synchronized");
  static_assert(std::is_trivially_copyable<carrier_record>::value,"Native carrier broadcast must not own pointers");
  MPI_Bcast(&carrier,sizeof(carrier),MPI_BYTE,owner_rank,comm);
  okay=finite(carrier.position)&&finite(carrier.velocity)&&carrier.mass==ctx.expected_gravity_carrier_mass_msun&&carrier.physical==ctx.expected_physical_bh_mass_msun&&
    carrier.position==ctx.expected_probe_position_proper_kpc&&carrier.state.bh_sidm_drag_version==1&&carrier.state.bh_sidm_drag_last_native_time==ctx.native_begin&&
    carrier.state.bh_sidm_drag_last_proper_seconds==ctx.proper_begin_seconds;
  require_collective(okay,comm,"Actual native BH phase/mass/clock differs from calibrated source binding");
  // Count first, then allocate only the declared complete local fine-DM bath.
  auto selected=[&](int i,vec&position){
    auto&p=particles.P[i];if(p.getType()!=1)return false;
    double delta[3];particles.nearest_image_intpos_to_pos(p.IntPos,carrier.native_position.data(),delta);
    vec relative{};for(int k=0;k<3;k++){relative[k]=delta[k]*ctx.native_length_to_proper_kpc;position[k]=carrier.position[k]+relative[k];}
    double radius=norm(relative);return radius>=ctx.recipient_inner_radius_proper_kpc&&radius<ctx.recipient_outer_radius_proper_kpc;
  };
  std::size_t count=0;for(int i=0;i<particles.NumPart;i++){vec position{};if(selected(i,position))count++;}
  unsigned long long local_count=count,global_count=0;MPI_Allreduce(&local_count,&global_count,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);
  int ranks=0;MPI_Comm_size(comm,&ranks);auto workspace=ctx.limits.conservative.maximum_workspace_bytes;
  bool bounded=global_count>=3&&global_count<=ctx.limits.conservative.maximum_particles&&global_count<=INT_MAX&&global_count<=workspace/64;
  if(bounded){auto remaining=workspace-64*global_count;bounded=count<=remaining/2048;
    if(bounded){remaining-=2048*count;bounded=static_cast<unsigned long long>(ranks)<=remaining/32;}}
  require_collective(bounded,
    comm,"Actual native source extraction exceeds its local workspace/census bound");
  std::vector<int> indices;std::vector<bath_particle> bath;std::vector<std::array<position_type,3>> native_positions;bool allocated=true;
  try{indices.reserve(count);bath.reserve(count);native_positions.reserve(count);}catch(const std::exception&){allocated=false;}
  require_collective(allocated,comm,"Native source census allocation failed");
  for(int i=0;i<particles.NumPart;i++){vec position{};if(!selected(i,position))continue;auto&p=particles.P[i];
    if(bath.size()>=count){okay=false;continue;}
    okay=okay&&p.ID.get()!=0&&p.get_Ti_Current()==ctx.native_timeline&&p.getMass()*ctx.native_mass_to_msun==ctx.recipient_gravity_mass_msun;
    bath_particle record{};record.id=p.ID.get();record.mass=p.getMass()*ctx.native_mass_to_msun;record.position=position;
    std::array<position_type,3> native_position{};for(int k=0;k<3;k++){record.velocity[k]=p.Vel[k]*ctx.native_velocity_to_km_s;native_position[k]=p.IntPos[k];}
    indices.push_back(i);bath.push_back(record);native_positions.push_back(native_position);
  }
  require_collective(okay&&bath.size()==count,comm,"Native aperture contains coarse, stale or invalid recipient particles");
  auto staged=research_sidm_drag_coupling::stage(response,ctx.actual_fixed_signature,bath,ctx.carrier_id,carrier.mass,carrier.position,carrier.velocity,
    ctx.background_mean_velocity_km_s,ctx.background_density_msun_kpc3,ctx.background_covariance_km2_s2,ctx.proper_begin_seconds,ctx.proper_end_seconds,
    ctx.seed,carrier.state.bh_sidm_drag_counter,ctx.limits,comm);
  auto next_state=research_sidm_drag_coupling::proposed_committed_state(carrier.state,staged,response,ctx.native_begin,ctx.native_end,ctx.erg_per_msun_km2_s2,comm);
  auto prepared=research_sidm_drag_commit::prepare(particles,indices,native_positions,owner_index,carrier.native_position,ctx.carrier_id,bath,staged,
    carrier.state,next_state,ctx.native_mass_to_msun,ctx.native_velocity_to_km_s,ctx.limits.conservative,comm);
  diagnostics out{};out.carrier_id=ctx.carrier_id;out.source_counter=staged.source_counter;out.next_source_counter=staged.next_source_counter;out.recipient_count=global_count;
  out.proper_begin_seconds=ctx.proper_begin_seconds;out.proper_end_seconds=ctx.proper_end_seconds;out.heat_msun_km2_s2=staged.conservative.heat;
  out.acceleration_cm_s2=staged.response.acceleration;out.rounded_momentum_residual=prepared.rounded_momentum_residual;
  out.rounded_angular_residual=prepared.rounded_angular_residual;out.rounded_energy_residual=prepared.rounded_energy_residual;
  out.momentum_budget=prepared.momentum_budget;out.angular_budget=prepared.angular_budget;out.energy_budget=prepared.energy_budget;
  out.reference_error_cm_s2=staged.response.reference_error;out.heldout_error_cm_s2=staged.response.heldout_error;
  research_sidm_drag_commit::commit(particles,prepared,comm);return out;
}
}
#endif
