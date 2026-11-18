#include "gadgetconfig.h"
#ifdef BLACK_HOLES
#include <climits>
#include <cstring>
#include <set>
#include "../main/simulation.h"
#include "research_bh_dynamics.h"
#include "research_bh_seed_storage.h"
#include "research_bh_accretion_roundoff.h"
#include "research_sidm_drag_source.h"
#include "research_sidm_drag_response.h"
#include "research_sidm_drag_state_response.h"
#include "research_sidm_drag_coupling.h"
#include "research_sidm_drag_commit.h"
#include "research_sidm_drag_context.h"
#include "research_sidm_wake_schedule.h"
#include <cstdlib>
using namespace research_bh;
void sim::black_hole_sidm_wake_schedule_step(void){
#ifdef RESEARCH_BH_WAKE_DIAGNOSTIC
#if !defined(FORCE_EQUAL_TIMESTEPS) || !defined(OUTPUT_EXACT_SYNCHRONIZED) || !defined(PERIODIC) || !defined(SELFGRAVITY)
#error "Native wake schedules require synchronized periodic self-gravitating evolution"
#endif
#if !defined(RESEARCH_BH_WAKE_DIAGNOSTIC_SHA256) || !defined(RESEARCH_BH_WAKE_DIAGNOSTIC_MAX_BYTES) || !defined(RESEARCH_BH_WAKE_DIAGNOSTIC_MAX_ROWS)
#error "Native wake diagnostics require compiled schedule SHA and allocation bounds"
#endif
#define RESEARCH_WAKE_RAW_TEXT(x) #x
#define RESEARCH_WAKE_TEXT(x) RESEARCH_WAKE_RAW_TEXT(x)
  const char*path=std::getenv("RESEARCH_BH_WAKE_DIAGNOSTIC_SCHEDULE");
  const std::string expected=RESEARCH_WAKE_TEXT(RESEARCH_BH_WAKE_DIAGNOSTIC_SHA256);
  research_sidm_drag_coupling::same_identity(expected,Communicator);
  research_sidm_drag_source::require_collective(path&&path[0],Communicator,"Native wake schedule scientific environment is missing");
  research_sidm_wake_schedule::schedule schedule;bool okay=true;
  try{schedule=research_sidm_wake_schedule::read(path,expected,RESEARCH_BH_WAKE_DIAGNOSTIC_MAX_BYTES,
    RESEARCH_BH_WAKE_DIAGNOSTIC_MAX_ROWS,All.MaxMemSize*1024.*1024.);}
  catch(const std::exception&){okay=false;}
  research_sidm_drag_source::require_collective(okay,Communicator,"Native wake schedule parse/allocation failed on an MPI owner");
  std::array<double,4> clocks{{All.TimeOld,All.Time,All.TimeBegin,All.TimeMax}},low{},high{};
  MPI_Allreduce(clocks.data(),low.data(),4,MPI_DOUBLE,MPI_MIN,Communicator);MPI_Allreduce(clocks.data(),high.data(),4,MPI_DOUBLE,MPI_MAX,Communicator);
  research_sidm_drag_source::require_collective(low==high,Communicator,"MPI wake scheduling clocks disagree");
  for(const auto&row:schedule.rows){
    research_sidm_drag_source::require_collective(row.native_time-All.TimeBegin>2*schedule.epoch_tolerance&&row.native_time<=All.TimeMax,
      Communicator,"Wake schedule epoch is outside the native measurement interval");
    if(row.native_time<=All.TimeOld+schedule.epoch_tolerance||row.native_time>All.Time+schedule.epoch_tolerance)continue;
    research_sidm_drag_source::require_collective(std::abs(row.native_time-All.Time)<=schedule.epoch_tolerance&&row.native_time>All.TimeBegin&&row.native_time<=All.TimeMax,
      Communicator,"Native source step missed its declared synchronized wake epoch");
    research_sidm_drag_source::require_collective(schedule.epoch_tolerance<(All.Time-All.TimeOld)/4,
      Communicator,"Wake epoch tolerance cannot distinguish actual adjacent native source steps");
    black_hole_sidm_publish_wake_measurement(row.carrier_id,row.edges,schedule.maximum_particles,schedule.maximum_work,schedule.workspace,
      row.measurement_id,schedule.publication,schedule.physics_signature);
  }
#undef RESEARCH_WAKE_TEXT
#undef RESEARCH_WAKE_RAW_TEXT
#endif
}
research_sidm_wake_measurement::measurement sim::black_hole_sidm_wake_measurement(std::uint64_t carrier_id,
    const std::vector<double>&edges,std::uint64_t maximum_particles,std::uint64_t maximum_work,std::uint64_t maximum_workspace_bytes){
  double mass=All.UnitMass_in_g/All.HubbleParam/solar,length=All.UnitLength_in_cm*All.cf_atime/All.HubbleParam/kpc;
  double acceleration=All.UnitVelocity_in_cm_per_s*All.UnitVelocity_in_cm_per_s*All.HubbleParam/
    (All.UnitLength_in_cm*All.cf_atime*All.cf_atime);
  bool valid=!edges.empty()&&std::isfinite(length)&&length>0&&std::isfinite(mass)&&mass>0&&
    std::isfinite(acceleration)&&acceleration>0&&std::isfinite(All.MaxMemSize)&&All.MaxMemSize>0&&
    maximum_workspace_bytes<=.25L*All.MaxMemSize*1024.L*1024.L;
#ifdef PERIODIC
  valid=valid&&edges.back()<All.BoxSize*length/2;
#else
  valid=false;
#endif
  research_sidm_drag_source::require_collective(valid,Communicator,"Native wake diagnostic needs bounded unique-image periodic gravity support");
  const auto softening=[&](particle_data&particle){
    unsigned index=particle.getSofteningClass();
    if(index>=sizeof(All.ForceSoftening)/sizeof(All.ForceSoftening[0]))return std::numeric_limits<double>::quiet_NaN();
    return All.ForceSoftening[index];
  };
  return research_sidm_wake_measurement::measure(Sp,GravTree,softening,Communicator,carrier_id,All.Ti_Current,All.Time,
    edges,mass,length,acceleration,All.G,maximum_particles,maximum_work,maximum_workspace_bytes);
}
namespace {
std::string wake_measurement_json(const research_sidm_wake_measurement::measurement&value,const std::vector<double>&edges,
    std::uint64_t measurement_id,const std::string&physics){
  std::ostringstream out;out<<std::setprecision(17)
    <<"{\"schema\":\"native-bh-wake-monopole-diagnostic-v1\",\"native_time\":"<<value.native_time
    <<",\"native_timeline_decimal\":\""<<All.Ti_Current<<"\",\"carrier_id_decimal\":\""<<value.carrier_id
    <<"\",\"measurement_id_decimal\":\""<<measurement_id<<"\",\"complete_dm_count_decimal\":\""<<value.complete_dm_count
    <<"\",\"pair_evaluations_decimal\":\""<<value.pair_evaluations<<"\",\"physical_bh_mass_msun\":"<<value.physical_bh_mass_msun
    <<",\"gravity_carrier_mass_msun\":"<<value.gravity_carrier_mass_msun<<",\"probe_position_proper_kpc\":"<<vector_json(value.probe_position_proper_kpc)
    <<",\"stored_grav_accel_cm_s2\":"<<vector_json(value.stored_gravity_acceleration_cm_s2)
    <<",\"native_physics_signature\":\""<<physics<<"\",\"force_scope\":\"native_kernel_aperture_monopoles_excludes_tree_pm_ewald\",\"radial_edges_proper_kpc\":[";
  for(std::size_t i=0;i<edges.size();i++){if(i)out<<',';out<<edges[i];}
  out<<"],\"periodic_correction_available\":"<<(value.periodic_correction_available?"true":"false")
    <<",\"periodic_correction_scope\":\"actual_native_ewald_aperture_sources; excludes_pm_tree_and_unselected_matter\""
    <<",\"pm_short_range_available\":"<<(value.pm_short_range_available?"true":"false")
    <<",\"pm_short_range_scope\":\"actual_native_cutoff_monopoles; excludes_long_range_mesh_and_tree_approximation\""
    <<",\"closed_total_comparison_available\":"<<(value.closed_total_comparison_available?"true":"false")
    <<",\"closed_total_comparison_scope\":\"tracked_gravity_frame_periodic_single_bh_fine_dm_only_no_pm_external_or_hierarchy\""
    <<",\"full_dm_monopoles_cm_s2\":"<<vector_json(value.full_dm_monopoles_cm_s2)
    <<",\"full_dm_periodic_correction_cm_s2\":"<<vector_json(value.full_dm_periodic_correction_cm_s2)<<",\"source_species_counts_decimal\":[";
  for(int i=0;i<7;i++){if(i)out<<',';out<<'"'<<value.source_species_counts[i]<<'"';}out<<"],\"shells\":[";
  for(std::size_t i=0;i<value.shells.size();i++){const auto&s=value.shells[i];if(i)out<<',';
    out<<"{\"particle_count_decimal\":\""<<s.particle_count<<"\",\"mass_msun\":"<<s.mass_msun
      <<",\"minimum_support_proper_kpc\":"<<s.minimum_support_proper_kpc<<",\"maximum_support_proper_kpc\":"<<s.maximum_support_proper_kpc
      <<",\"acceleration_cm_s2\":"<<vector_json(s.acceleration_cm_s2)
      <<",\"periodic_correction_cm_s2\":"<<vector_json(s.periodic_correction_cm_s2)
      <<",\"pm_short_range_cm_s2\":"<<vector_json(s.pm_short_range_cm_s2)<<'}';}
  out<<"]}\n";return out.str();
}
}
void sim::black_hole_sidm_publish_wake_measurement(std::uint64_t carrier_id,const std::vector<double>&edges,
    std::uint64_t maximum_particles,std::uint64_t maximum_work,std::uint64_t maximum_workspace_bytes,
    std::uint64_t measurement_id,std::uint64_t maximum_publication_bytes,const std::string&actual_physics_signature){
  using research_sidm_drag_source::require_collective;
  research_sidm_drag_coupling::same_identity(actual_physics_signature,Communicator);
  std::array<unsigned long long,2> bounds{{measurement_id,maximum_publication_bytes}},lo{},hi{};
  MPI_Allreduce(bounds.data(),lo.data(),2,MPI_UNSIGNED_LONG_LONG,MPI_MIN,Communicator);MPI_Allreduce(bounds.data(),hi.data(),2,MPI_UNSIGNED_LONG_LONG,MPI_MAX,Communicator);
  require_collective(lo==hi&&measurement_id>0&&maximum_publication_bytes>=8192.L+2048.L*edges.size()&&
    maximum_publication_bytes<=.125L*All.MaxMemSize*1024.L*1024.L,Communicator,"Native wake publication identity/allocation invalid");
  auto value=black_hole_sidm_wake_measurement(carrier_id,edges,maximum_particles,maximum_work,maximum_workspace_bytes);
  int rank=0;MPI_Comm_rank(Communicator,&rank);bool okay=true;
  if(rank==0)try {
    std::string payload=wake_measurement_json(value,edges,measurement_id,actual_physics_signature);
    if(payload.size()>maximum_publication_bytes)throw std::runtime_error("Native wake serialized record exceeds allocation");
    std::ostringstream name;name<<All.OutputDir<<"/wake_monopole_"<<All.Ti_Current<<'_'<<carrier_id<<'_'<<measurement_id<<".json";
    std::uint64_t bytes=0,crc=0;
    okay=research_stellar_publish_file(name.str(),[&](FILE*f){return fwrite(payload.data(),1,payload.size(),f)==payload.size();},bytes,crc);
    if(okay){std::string seal=research_asset_sha256::bytes(payload)+"\n";
      okay=research_stellar_publish_file(name.str()+".sha256",[&](FILE*f){return fwrite(seal.data(),1,seal.size(),f)==seal.size();},bytes,crc);}
  } catch(const std::exception&){okay=false;}
  require_collective(okay,Communicator,"Native wake immutable record/seal publication failed");
}
namespace {
void check_drag_launch_budget(const research_sidm_drag_context::launch_settings&settings,MPI_Comm comm){
  long double bytes=8.L*settings.maximum_context_bytes+8.L*settings.maximum_table_bytes+
    4096.L*settings.maximum_states+settings.maximum_schedule_workspace_bytes;
  research_sidm_drag_source::require_collective(std::isfinite(All.MaxMemSize)&&All.MaxMemSize>0&&
    bytes<=.25L*All.MaxMemSize*1024.L*1024.L,comm,"Calibrated launch asset/schedule allocation exceeds the actual native rank budget");
}
research_sidm_drag_driver::context bind_drag_assets(const research_sidm_drag_context::asset_bundle&assets,
    const std::string&physics,double native_begin,double proper_begin,double proper_end,MPI_Comm comm){
  research_sidm_drag_coupling::same_identity(physics,comm);
  research_sidm_drag_driver::context context{};bool okay=true;
  try{context=assets.context_binding().bind(assets.response(),physics,All.Time,native_begin,All.Time,proper_begin,proper_end,
    All.Ti_Current,All.UnitMass_in_g/All.HubbleParam/solar,All.UnitVelocity_in_cm_per_s/All.cf_atime/1e5,
    All.UnitLength_in_cm*All.cf_atime/All.HubbleParam/kpc,solar*1e10);}
  catch(const std::exception&){okay=false;}
  research_sidm_drag_source::require_collective(okay,comm,"Actual native epoch/physics cannot bind the calibrated launch context");
  research_sidm_drag_source::require_collective(context.limits.conservative.maximum_workspace_bytes<=.5L*All.MaxMemSize*1024.L*1024.L,
    comm,"Calibrated source workspace exceeds the actual native rank budget");
  return context;
}
}
void sim::black_hole_sidm_drag_initialize_assets(const research_sidm_drag_context::launch_settings&settings,
    const std::string&actual_native_physics_signature,double proper_origin_seconds,bool restore){
  check_drag_launch_budget(settings,Communicator);
  auto assets=research_sidm_drag_context::load_collective(settings,Communicator);
  auto context=bind_drag_assets(*assets,actual_native_physics_signature,All.Time,proper_origin_seconds,proper_origin_seconds,Communicator);
  int mode=restore?1:0,low=0,high=0;MPI_Allreduce(&mode,&low,1,MPI_INT,MPI_MIN,Communicator);MPI_Allreduce(&mode,&high,1,MPI_INT,MPI_MAX,Communicator);
  research_sidm_drag_source::require_collective(low==high,Communicator,"MPI calibrated source fresh/restore modes disagree");
  if(restore)black_hole_sidm_drag_restore(assets->response(),context);
  else black_hole_sidm_drag_initialize(assets->response(),context);
}
std::vector<research_sidm_drag_driver::diagnostics> sim::black_hole_sidm_drag_advance_assets(
    const research_sidm_drag_context::launch_settings&settings,const std::string&actual_native_physics_signature,
    double native_begin,double proper_begin_seconds,double proper_end_seconds){
  check_drag_launch_budget(settings,Communicator);
  auto assets=research_sidm_drag_context::load_collective(settings,Communicator);
  auto context=bind_drag_assets(*assets,actual_native_physics_signature,native_begin,proper_begin_seconds,proper_end_seconds,Communicator);
  return black_hole_sidm_drag_advance(assets->response(),context,assets->context_binding().maximum_subcycles(),settings.maximum_schedule_workspace_bytes);
}
void sim::black_hole_sidm_drag_restore(const research_sidm_drag_response::state_response&response,
    const research_sidm_drag_driver::context&context){
  using research_sidm_drag_source::require_collective;
  require_collective(response.sha256_verified()&&(All.RestartFlag==RST_RESUME||All.RestartFlag==RST_STARTFROMSNAP)&&
    context.native_begin==All.Time&&context.native_timeline==All.Ti_Current&&context.actual_fixed_signature==response.fixed_signature(),
    Communicator,"Calibrated source restore requires an actual native continuation boundary");
  research_sidm_drag_coupling::same_identity(response.product_identity(),Communicator);
  research_sidm_drag_coupling::same_identity(response.fixed_signature(),Communicator);
  unsigned long long id=context.carrier_id,low=0,high=0;MPI_Allreduce(&id,&low,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,Communicator);
  MPI_Allreduce(&id,&high,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,Communicator);require_collective(low==high&&id!=0,Communicator,"MPI restore carrier IDs disagree");
  auto model=research_sidm_drag_coupling::identity_words(response.product_identity());
  auto signature=research_sidm_drag_coupling::identity_words(response.fixed_signature());
  double mass_conversion=All.UnitMass_in_g/All.HubbleParam/solar,length_conversion=All.UnitLength_in_cm*All.cf_atime/All.HubbleParam/kpc;
  int local_count=0,total_count=0;bool okay=true;
  for(int i=0;i<Sp.NumPart;i++)if(Sp.P[i].ID.get()==id){local_count++;auto&p=Sp.P[i];const auto&s=p.ResearchSourceState;
    okay=okay&&p.getType()==5&&p.getMass()>0&&p.BHMass>0&&p.get_Ti_Current()==All.Ti_Current&&
      s.bh_sidm_drag_version==1&&s.bh_sidm_drag_counter<UINT64_MAX&&s.bh_sidm_drag_model_crc32==response.checksum()&&
      s.bh_sidm_drag_last_native_time==All.Time&&s.bh_sidm_drag_last_proper_seconds==context.proper_begin_seconds&&
      std::isfinite(s.bh_sidm_drag_heat_erg)&&s.bh_sidm_drag_heat_erg>=0&&
      std::isfinite(s.bh_sidm_drag_reference_error_cm_s2)&&s.bh_sidm_drag_reference_error_cm_s2>=0&&
      std::isfinite(s.bh_sidm_drag_heldout_error_cm_s2)&&s.bh_sidm_drag_heldout_error_cm_s2>=0;
    double position[3];Sp.intpos_to_pos(p.IntPos,position);
    okay=okay&&p.getMass()*mass_conversion==context.expected_gravity_carrier_mass_msun&&
      p.BHMass*mass_conversion==context.expected_physical_bh_mass_msun;
    for(int k=0;k<3;k++)okay=okay&&position[k]*length_conversion==context.expected_probe_position_proper_kpc[k];
    for(int k=0;k<4;k++)okay=okay&&s.bh_sidm_drag_model_sha256[k]==model[k]&&s.bh_sidm_drag_signature_sha256[k]==signature[k];
    for(double value:s.bh_sidm_drag_last_acceleration_cm_s2)okay=okay&&std::isfinite(value);
  }
  MPI_Allreduce(&local_count,&total_count,1,MPI_INT,MPI_SUM,Communicator);
  double origin=context.proper_begin_seconds,origin_low=0,origin_high=0;
  MPI_Allreduce(&origin,&origin_low,1,MPI_DOUBLE,MPI_MIN,Communicator);MPI_Allreduce(&origin,&origin_high,1,MPI_DOUBLE,MPI_MAX,Communicator);
  okay=okay&&std::isfinite(origin)&&origin_low==origin_high;
  require_collective(okay&&total_count==1,Communicator,"Restored calibrated source has stale model/clock/state or no unique carrier");
  // No state initialization and no duplicate initialization journal row:
  // the persistent counter, cumulative heat and previous events are retained.
}
void sim::black_hole_sidm_drag_initialize(const research_sidm_drag_response::state_response&response,
    const research_sidm_drag_driver::context&context){
  using research_sidm_drag_source::require_collective;
  require_collective(response.sha256_verified()&&All.RestartFlag==RST_BEGIN&&All.Time==All.TimeBegin&&context.native_begin==All.Time&&
    context.native_timeline==All.Ti_Current&&context.carrier_id!=0&&std::isfinite(context.proper_begin_seconds)&&
    context.actual_fixed_signature==response.fixed_signature(),Communicator,"SIDM drag initialization requires the actual fresh run boundary/model");
  research_sidm_drag_coupling::same_identity(response.product_identity(),Communicator);
  research_sidm_drag_coupling::same_identity(response.fixed_signature(),Communicator);
  unsigned long long id=context.carrier_id,low_id=0,high_id=0;MPI_Allreduce(&id,&low_id,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,Communicator);
  MPI_Allreduce(&id,&high_id,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,Communicator);
  double origin=context.proper_begin_seconds,low_origin=0,high_origin=0;
  MPI_Allreduce(&origin,&low_origin,1,MPI_DOUBLE,MPI_MIN,Communicator);MPI_Allreduce(&origin,&high_origin,1,MPI_DOUBLE,MPI_MAX,Communicator);
  require_collective(low_id==high_id&&low_origin==high_origin,Communicator,"MPI drag initialization identity/clock disagreement");
  int index=-1,local_count=0,total_count=0;bool okay=true;
  for(int i=0;i<Sp.NumPart;i++)if(Sp.P[i].ID.get()==context.carrier_id){index=i;local_count++;}
  MPI_Allreduce(&local_count,&total_count,1,MPI_INT,MPI_SUM,Communicator);
  require_collective(total_count==1,Communicator,"Fresh calibrated BH does not have a unique native owner");
  research_source_state proposed{};
  if(index>=0){auto&p=Sp.P[index];double native_position[3];Sp.intpos_to_pos(p.IntPos,native_position);
    double mass_conversion=All.UnitMass_in_g/All.HubbleParam/solar,length_conversion=All.UnitLength_in_cm*All.cf_atime/All.HubbleParam/kpc;
    okay=p.getType()==5&&p.getMass()>0&&p.BHMass>0&&p.get_Ti_Current()==All.Ti_Current&&
      p.getMass()*mass_conversion==context.expected_gravity_carrier_mass_msun&&p.BHMass*mass_conversion==context.expected_physical_bh_mass_msun;
    for(int k=0;k<3;k++)okay=okay&&native_position[k]*length_conversion==context.expected_probe_position_proper_kpc[k];
    try{proposed=research_sidm_drag_coupling::initialize_clock(p.ResearchSourceState,response,All.Time,origin);}catch(const std::exception&){okay=false;}}
  require_collective(okay,Communicator,"Fresh calibrated source clock/state initialization failed");
  if(index>=0)Sp.P[index].ResearchSourceState=proposed;
  int rank=0;MPI_Comm_rank(Communicator,&rank);std::vector<std::string> rows;
  if(rank==0){std::ostringstream event;event<<std::setprecision(17)<<"{\"kind\":\"sidm_drag_initialize\",\"native_time\":"<<All.Time
    <<",\"particle_id\":"<<context.carrier_id<<",\"counter\":1,\"initialized_source_counter\":0,\"proper_origin_seconds\":"<<origin
    <<",\"response_product_sha256\":\""<<response.product_identity()<<"\",\"fixed_response_signature\":\""<<response.fixed_signature()
    <<"\",\"table_crc32\":"<<response.checksum()<<'}';append(rows,event.str());}
  publish(rows,Communicator);
}
void sim::black_hole_sidm_drag_transition_response(const research_sidm_drag_response::state_response&old_response,
    const research_sidm_drag_response::state_response&new_response,const research_sidm_drag_driver::context&context){
  using research_sidm_drag_source::require_collective;
  research_sidm_drag_coupling::same_identity(old_response.product_identity(),Communicator);
  research_sidm_drag_coupling::same_identity(new_response.product_identity(),Communicator);
  research_sidm_drag_coupling::same_identity(new_response.fixed_signature(),Communicator);
  require_collective(context.actual_fixed_signature==new_response.fixed_signature()&&
    old_response.fixed_signature()==new_response.fixed_signature()&&context.native_begin==All.Time&&
    context.native_end==All.Time&&context.native_timeline==All.Ti_Current&&context.carrier_id!=0,
    Communicator,"Calibrated response transition requires a synchronized unchanged physical-definition boundary");
  std::array<unsigned long long,3> identity{{context.carrier_id,old_response.checksum(),new_response.checksum()}},low{},high{};
  MPI_Allreduce(identity.data(),low.data(),3,MPI_UNSIGNED_LONG_LONG,MPI_MIN,Communicator);
  MPI_Allreduce(identity.data(),high.data(),3,MPI_UNSIGNED_LONG_LONG,MPI_MAX,Communicator);
  double proper=context.proper_begin_seconds,proper_low=0,proper_high=0;
  MPI_Allreduce(&proper,&proper_low,1,MPI_DOUBLE,MPI_MIN,Communicator);MPI_Allreduce(&proper,&proper_high,1,MPI_DOUBLE,MPI_MAX,Communicator);
  require_collective(low==high&&proper_low==proper_high&&std::isfinite(proper)&&context.proper_end_seconds==proper,
    Communicator,"MPI response transition identity/clock disagreement");
  std::array<double,18> physical{{context.expected_physical_bh_mass_msun,context.expected_gravity_carrier_mass_msun,
    context.expected_probe_position_proper_kpc[0],context.expected_probe_position_proper_kpc[1],context.expected_probe_position_proper_kpc[2],
    context.background_mean_velocity_km_s[0],context.background_mean_velocity_km_s[1],context.background_mean_velocity_km_s[2],
    context.background_density_msun_kpc3,context.background_covariance_km2_s2[0],context.background_covariance_km2_s2[1],
    context.background_covariance_km2_s2[2],context.background_covariance_km2_s2[3],context.background_covariance_km2_s2[4],
    context.background_covariance_km2_s2[5],context.limits.maximum_reference_error_cm_s2,context.limits.maximum_heldout_error_cm_s2,context.proper_box_kpc}},physical_low{},physical_high{};
  MPI_Allreduce(physical.data(),physical_low.data(),18,MPI_DOUBLE,MPI_MIN,Communicator);
  MPI_Allreduce(physical.data(),physical_high.data(),18,MPI_DOUBLE,MPI_MAX,Communicator);
  bool finite_context=true;for(double value:physical)finite_context=finite_context&&std::isfinite(value);
  require_collective(finite_context&&physical_low==physical_high&&context.limits.maximum_reference_error_cm_s2>=0&&
    context.limits.maximum_heldout_error_cm_s2>=0,Communicator,"MPI transition physical context/error budgets differ");
  int index=-1,local_count=0,total_count=0;bool okay=true;research_source_state proposed{};
  double mass_conversion=All.UnitMass_in_g/All.HubbleParam/solar,length_conversion=All.UnitLength_in_cm*All.cf_atime/All.HubbleParam/kpc;
  for(int i=0;i<Sp.NumPart;i++)if(Sp.P[i].ID.get()==context.carrier_id){index=i;local_count++;}
  MPI_Allreduce(&local_count,&total_count,1,MPI_INT,MPI_SUM,Communicator);
  require_collective(total_count==1,Communicator,"Response transition carrier does not have a unique native owner");
  if(index>=0){auto&p=Sp.P[index];double position[3];Sp.intpos_to_pos(p.IntPos,position);
    okay=p.getType()==5&&p.get_Ti_Current()==All.Ti_Current&&p.getMass()*mass_conversion==context.expected_gravity_carrier_mass_msun&&
      p.BHMass*mass_conversion==context.expected_physical_bh_mass_msun;
    for(int k=0;k<3;k++)okay=okay&&position[k]*length_conversion==context.expected_probe_position_proper_kpc[k];
    try{std::array<double,10> query{};
      for(int k=0;k<3;k++)query[k]=p.Vel[k]*All.UnitVelocity_in_cm_per_s/All.cf_atime/1e5-context.background_mean_velocity_km_s[k];
      query[3]=context.background_density_msun_kpc3;
      for(int k=0;k<6;k++)query[k+4]=context.background_covariance_km2_s2[k];
      auto support=new_response.evaluate(query,context.actual_fixed_signature);
      okay=okay&&support.reference_error<=context.limits.maximum_reference_error_cm_s2&&
        support.heldout_error<=context.limits.maximum_heldout_error_cm_s2;
      proposed=research_sidm_drag_coupling::transition_response(p.ResearchSourceState,old_response,new_response,All.Time,proper);}
    catch(const std::exception&){okay=false;}}
  require_collective(okay,Communicator,"Response transition persistent state or physical carrier binding is invalid");
  unsigned long long counter=index>=0?proposed.bh_sidm_drag_counter:0,global_counter=0;
  double heat=index>=0?proposed.bh_sidm_drag_heat_erg:0,global_heat=0;
  MPI_Allreduce(&counter,&global_counter,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,Communicator);
  MPI_Allreduce(&heat,&global_heat,1,MPI_DOUBLE,MPI_SUM,Communicator);
  int rank=0;MPI_Comm_rank(Communicator,&rank);std::vector<std::string> rows;
  try{if(rank==0){std::ostringstream event;event<<std::setprecision(17)
    <<"{\"kind\":\"sidm_drag_response_transition\",\"native_time\":"<<All.Time<<",\"particle_id\":"<<context.carrier_id
    <<",\"counter\":"<<global_counter+1<<",\"retained_source_counter\":"<<global_counter<<",\"proper_origin_seconds\":"<<proper
    <<",\"retained_heat_erg\":"<<global_heat<<",\"previous_response_product_sha256\":\""<<old_response.product_identity()
    <<"\",\"response_product_sha256\":\""<<new_response.product_identity()<<"\",\"fixed_response_signature\":\""<<new_response.fixed_signature()
    <<"\",\"previous_table_crc32\":"<<old_response.checksum()<<",\"table_crc32\":"<<new_response.checksum()<<'}';append(rows,event.str());}}
  catch(const std::exception&){okay=false;}
  require_collective(okay,Communicator,"Response transition journal allocation failed before state commit");
  if(index>=0)Sp.P[index].ResearchSourceState=proposed;
  publish(rows,Communicator);
}
void sim::black_hole_sidm_drag_transition_assets(const research_sidm_drag_context::launch_settings&old_settings,
    const research_sidm_drag_context::launch_settings&new_settings,const std::string&actual_native_physics_signature,
    double proper_boundary_seconds){
  check_drag_launch_budget(old_settings,Communicator);check_drag_launch_budget(new_settings,Communicator);
  long double combined=8.L*(static_cast<long double>(old_settings.maximum_context_bytes)+new_settings.maximum_context_bytes+
    old_settings.maximum_table_bytes+new_settings.maximum_table_bytes)+4096.L*(static_cast<long double>(old_settings.maximum_states)+new_settings.maximum_states);
  research_sidm_drag_source::require_collective(combined<=.25L*All.MaxMemSize*1024.L*1024.L,
    Communicator,"Two-response transition assets exceed the native allocation");
  auto old_assets=research_sidm_drag_context::load_collective(old_settings,Communicator);
  auto new_assets=research_sidm_drag_context::load_collective(new_settings,Communicator);
  auto old_context=bind_drag_assets(*old_assets,actual_native_physics_signature,All.Time,proper_boundary_seconds,proper_boundary_seconds,Communicator);
  auto new_context=bind_drag_assets(*new_assets,actual_native_physics_signature,All.Time,proper_boundary_seconds,proper_boundary_seconds,Communicator);
  research_sidm_drag_source::require_collective(old_context.carrier_id==new_context.carrier_id&&old_context.seed==new_context.seed&&
    old_context.expected_probe_position_proper_kpc==new_context.expected_probe_position_proper_kpc&&
    old_context.expected_physical_bh_mass_msun==new_context.expected_physical_bh_mass_msun&&
    old_context.expected_gravity_carrier_mass_msun==new_context.expected_gravity_carrier_mass_msun&&
    old_context.background_mean_velocity_km_s==new_context.background_mean_velocity_km_s&&
    old_context.background_density_msun_kpc3==new_context.background_density_msun_kpc3&&
    old_context.background_covariance_km2_s2==new_context.background_covariance_km2_s2&&
    old_context.recipient_inner_radius_proper_kpc==new_context.recipient_inner_radius_proper_kpc&&
    old_context.recipient_outer_radius_proper_kpc==new_context.recipient_outer_radius_proper_kpc&&
    old_context.recipient_gravity_mass_msun==new_context.recipient_gravity_mass_msun&&old_context.proper_box_kpc==new_context.proper_box_kpc,
    Communicator,"Response replacement changes the measured physical background, carrier, recipient or random stream");
  black_hole_sidm_drag_transition_response(old_assets->response(),new_assets->response(),new_context);
}
std::vector<research_sidm_drag_driver::diagnostics> sim::black_hole_sidm_drag_advance(
    const research_sidm_drag_response::state_response&response,const research_sidm_drag_driver::context&context,
    std::size_t maximum_subcycles,std::uint64_t maximum_schedule_workspace_bytes){
  research_sidm_drag_source::require_collective(std::isfinite(context.native_begin)&&std::isfinite(context.native_end)&&
    context.native_begin>=All.TimeOld&&context.native_begin<context.native_end&&context.native_end==All.Time&&
    std::isfinite(context.proper_clock_absolute_tolerance_seconds)&&context.proper_clock_absolute_tolerance_seconds>=0&&
    std::isfinite(context.proper_clock_relative_tolerance)&&context.proper_clock_relative_tolerance>=0&&context.proper_clock_relative_tolerance<=1e-6,
    Communicator,"Automatic source advance must cover a finite actual native interval with explicit clock limits");
  std::array<double,2> proper{{context.proper_begin_seconds,context.proper_end_seconds}},proper_low{},proper_high{};
  MPI_Allreduce(proper.data(),proper_low.data(),2,MPI_DOUBLE,MPI_MIN,Communicator);
  MPI_Allreduce(proper.data(),proper_high.data(),2,MPI_DOUBLE,MPI_MAX,Communicator);
  double duration=proper_seconds(context.native_begin,context.native_end);
  double budget=context.proper_clock_absolute_tolerance_seconds+context.proper_clock_relative_tolerance*duration;
  research_sidm_drag_source::require_collective(proper_low==proper_high&&std::isfinite(duration)&&duration>0&&
    std::isfinite(proper[0])&&std::isfinite(proper[1])&&proper[1]>proper[0]&&std::isfinite(budget)&&budget>=0&&
    std::abs((proper[1]-proper[0])-duration)<=budget,Communicator,"Complete supplied source interval differs from the actual native proper clock");
  auto plan=black_hole_sidm_drag_plan(context.native_begin,context.proper_begin_seconds,context.limits.maximum_proper_step_seconds,
    maximum_subcycles,maximum_schedule_workspace_bytes,context.proper_clock_absolute_tolerance_seconds,context.proper_clock_relative_tolerance);
  research_sidm_drag_source::require_collective(!plan.empty()&&plan.front().native_begin==context.native_begin&&
    plan.back().native_end==context.native_end&&std::abs(plan.back().proper_end_seconds-context.proper_end_seconds)<=budget,
    Communicator,"Automatic source plan loses the supplied complete endpoint");
  research_sidm_drag_source::require_collective(plan.size()<=maximum_schedule_workspace_bytes/(sizeof(research_sidm_drag_driver::diagnostics)+128),
    Communicator,"Automatic source diagnostics and schedule exceed declared workspace");
  std::vector<research_sidm_drag_driver::diagnostics> results;bool allocated=true;
  research_sidm_drag_driver::context subcycle{};
  try{results.reserve(plan.size());subcycle=context;}catch(const std::exception&){allocated=false;}
  research_sidm_drag_source::require_collective(allocated,Communicator,"Automatic native source diagnostic allocation failed");
  for(const auto&interval:plan){subcycle.native_begin=interval.native_begin;subcycle.native_end=interval.native_end;
    subcycle.proper_begin_seconds=interval.proper_begin_seconds;subcycle.proper_end_seconds=interval.proper_end_seconds;
    results.push_back(black_hole_sidm_drag_source_step(response,subcycle));}
  return results;
}
std::vector<research_sidm_drag_schedule::interval> sim::black_hole_sidm_drag_plan(double native_begin,double proper_origin_seconds,
    double maximum_step_seconds,std::size_t maximum_subcycles,std::uint64_t maximum_workspace_bytes,
    double absolute_clock_tolerance_seconds,double relative_clock_tolerance){
  research_sidm_drag_source::require_collective(native_begin>=All.TimeOld&&native_begin<All.Time,Communicator,"Calibrated source plan is outside the actual native step");
  std::array<double,6> settings{{native_begin,All.Time,proper_origin_seconds,maximum_step_seconds,absolute_clock_tolerance_seconds,relative_clock_tolerance}},low{},high{};
  MPI_Allreduce(settings.data(),low.data(),6,MPI_DOUBLE,MPI_MIN,Communicator);MPI_Allreduce(settings.data(),high.data(),6,MPI_DOUBLE,MPI_MAX,Communicator);
  std::array<unsigned long long,2> bounds{{maximum_subcycles,maximum_workspace_bytes}},bounds_low{},bounds_high{};
  MPI_Allreduce(bounds.data(),bounds_low.data(),2,MPI_UNSIGNED_LONG_LONG,MPI_MIN,Communicator);MPI_Allreduce(bounds.data(),bounds_high.data(),2,MPI_UNSIGNED_LONG_LONG,MPI_MAX,Communicator);
  research_sidm_drag_source::require_collective(low==high&&bounds_low==bounds_high,Communicator,"MPI owners disagree on calibrated source planning inputs");
  std::vector<research_sidm_drag_schedule::interval> plan;bool okay=true;
  try{plan=research_sidm_drag_schedule::partition(native_begin,All.Time,proper_origin_seconds,maximum_step_seconds,maximum_subcycles,
      maximum_workspace_bytes,absolute_clock_tolerance_seconds,relative_clock_tolerance,[](double a,double b){return proper_seconds(a,b);});}
  catch(const std::exception&){okay=false;}
  research_sidm_drag_source::require_collective(okay,Communicator,"Calibrated source clock partition/allocation failed on an MPI owner");
  unsigned long long count=plan.size(),minimum_count=0,maximum_count=0;
  MPI_Allreduce(&count,&minimum_count,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,Communicator);MPI_Allreduce(&count,&maximum_count,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,Communicator);
  research_sidm_drag_source::require_collective(minimum_count==maximum_count,Communicator,"MPI source plans differ in subcycle count");
  std::vector<research_sidm_drag_schedule::interval> reference;
  try{reference=plan;}catch(const std::exception&){okay=false;}
  research_sidm_drag_source::require_collective(okay,Communicator,"Native source plan agreement workspace allocation failed");
  static_assert(sizeof(research_sidm_drag_schedule::interval)==4*sizeof(double),"Source interval wire layout changed");
  MPI_Bcast(reference.data(),static_cast<int>(reference.size()*sizeof(reference[0])),MPI_BYTE,0,Communicator);
  research_sidm_drag_source::require_collective(std::memcmp(reference.data(),plan.data(),plan.size()*sizeof(plan[0]))==0,Communicator,"Native proper source plans differ across MPI owners");
  return plan;
}
research_sidm_drag_driver::diagnostics sim::black_hole_sidm_drag_source_step(
    const research_sidm_drag_response::state_response &response,const research_sidm_drag_driver::context &context){
  if(context.native_timeline!=All.Ti_Current||context.native_begin<All.TimeOld||context.native_end>All.Time||context.native_end<=context.native_begin)
    Terminate("Calibrated BH source context differs from the current native clock");
  double mass_conversion=All.UnitMass_in_g/All.HubbleParam/solar,velocity_conversion=All.UnitVelocity_in_cm_per_s/All.cf_atime/1e5;
  double length_conversion=All.UnitLength_in_cm*All.cf_atime/All.HubbleParam/kpc;
  if(context.native_mass_to_msun!=mass_conversion||context.native_velocity_to_km_s!=velocity_conversion||
      context.native_length_to_proper_kpc!=length_conversion||context.proper_box_kpc!=All.BoxSize*length_conversion||context.erg_per_msun_km2_s2!=solar*1e10)
    Terminate("Calibrated BH source conversion differs from actual native units");
  if(!std::isfinite(context.proper_clock_absolute_tolerance_seconds)||context.proper_clock_absolute_tolerance_seconds<0||
      !std::isfinite(context.proper_clock_relative_tolerance)||context.proper_clock_relative_tolerance<0||context.proper_clock_relative_tolerance>1e-6)
    Terminate("Invalid calibrated BH source proper-clock budget");
  double actual_duration=proper_seconds(context.native_begin,context.native_end),supplied_duration=context.proper_end_seconds-context.proper_begin_seconds;
  double clock_budget=context.proper_clock_absolute_tolerance_seconds+context.proper_clock_relative_tolerance*actual_duration;
  if(!std::isfinite(actual_duration)||actual_duration<=0||!std::isfinite(supplied_duration)||!std::isfinite(clock_budget)||
      std::abs(actual_duration-supplied_duration)>clock_budget)Terminate("Calibrated source duration differs from actual native cosmological/proper clock");
  auto result=research_sidm_drag_driver::step(Sp,response,context,Communicator);
  int rank=0;MPI_Comm_rank(Communicator,&rank);std::vector<std::string> rows;
  if(rank==0){std::ostringstream event;event<<std::setprecision(17)
    <<"{\"kind\":\"sidm_drag\",\"native_time\":"<<All.Time<<",\"particle_id\":"<<result.carrier_id<<",\"counter\":"<<result.next_source_counter
    <<",\"previous_counter\":"<<result.source_counter<<",\"recipient_count\":"<<result.recipient_count
    <<",\"source_native_begin\":"<<context.native_begin<<",\"source_native_end\":"<<context.native_end<<",\"proper_begin_seconds\":"<<result.proper_begin_seconds<<",\"proper_end_seconds\":"<<result.proper_end_seconds
    <<",\"response_product_sha256\":\""<<response.product_identity()<<"\",\"fixed_response_signature\":\""<<response.fixed_signature()<<"\",\"table_crc32\":"<<response.checksum()
    <<",\"heat_msun_km2_s2\":"<<result.heat_msun_km2_s2<<",\"acceleration_cm_s2\":"<<vector_json(result.acceleration_cm_s2)
    <<",\"rounded_momentum_residual_msun_km_s\":"<<vector_json(result.rounded_momentum_residual)
    <<",\"rounded_angular_residual_msun_kpc_km_s\":"<<vector_json(result.rounded_angular_residual)
    <<",\"rounded_energy_residual_msun_km2_s2\":"<<result.rounded_energy_residual
    <<",\"momentum_budget_msun_km_s\":"<<result.momentum_budget<<",\"angular_budget_msun_kpc_km_s\":"<<result.angular_budget<<",\"energy_budget_msun_km2_s2\":"<<result.energy_budget
    <<",\"reference_error_cm_s2\":"<<result.reference_error_cm_s2<<",\"heldout_error_cm_s2\":"<<result.heldout_error_cm_s2<<'}';
    append(rows,event.str());}
  publish(rows,Communicator);return result;
}
namespace {
struct record {uint64_t id;MyIntPosType pos[3];double mass,physical,support,radiated_mass,feedback_energy;vec velocity;research_source_state state;};
inline std::string prefix(const char*kind,uint64_t id,uint64_t counter){std::ostringstream s;s<<std::setprecision(17)<<"{\"kind\":\""<<kind<<"\",\"native_time\":"<<All.Time<<",\"particle_id\":"<<id<<",\"counter\":"<<counter;return s.str();}
}
void sim::black_hole_dynamics_step(void){
  if(!All.BHDynamicsEnabled)return;
  if(!(All.BHBackgroundRadiusProperKpc>0 && All.BHMinimumImpactProperKpc>0 && All.BHStellarHardeningCoefficient>=0 &&
       All.BHBinaryMaximumEccentricity>=0 && All.BHBinaryMaximumEccentricity<=.05 && All.BHDynamicsFraction>0 && All.BHDynamicsFraction<=.01 && All.BHDynamicsMaximumSubcycles>0))Terminate("Invalid native BH dynamics profile");
  uint64_t seed;if(!research_stellar_seed(All.BHRecoilSeed,seed))Terminate("BH recoil seed must be a uint64 decimal string");
  double massunit=All.UnitMass_in_g/All.HubbleParam,properlength=All.UnitLength_in_cm*All.cf_atime/All.HubbleParam;
  double radius=All.BHBackgroundRadiusProperKpc*kpc/properlength;
  if(radius>=All.BoxSize/2)Terminate("BH dynamics support exceeds unique periodic region");
  std::vector<record> local;std::vector<std::string> rows;
  for(int i=0;i<Sp.NumPart;i++)if(Sp.P[i].getType()==5 && Sp.P[i].getMass()>0 && Sp.P[i].ID.get()!=0){
    auto &p=Sp.P[i];auto &s=p.ResearchSourceState;
    if(!s.bh_state_version && All.RestartFlag==RST_STARTFROMSNAP)Terminate("Snapshot restart is missing BH dynamical state");
    if(!s.bh_state_version){s.bh_state_version=1;s.bh_spin[0]=All.BHSeedSpinX;s.bh_spin[1]=All.BHSeedSpinY;s.bh_spin[2]=All.BHSeedSpinZ;s.bh_last_dynamics_time=All.TimeOld;}
    if(s.bh_state_version!=1 || s.bh_last_dynamics_time>All.Time)Terminate("Invalid BH dynamics restart state");
    record b{};b.id=p.ID.get();b.mass=p.getMass();b.physical=p.BHMass;b.radiated_mass=p.BHCumulativeRadiatedMass;b.feedback_energy=p.BHCumulativeFeedbackEnergy;b.state=s;b.support=All.ForceSoftening[p.getSofteningClass()]*properlength;
    for(int k=0;k<3;k++){b.pos[k]=p.IntPos[k];b.velocity[k]=p.Vel[k]/All.cf_atime*All.UnitVelocity_in_cm_per_s;}local.push_back(b);
  }
  if(local.size()>size_t(INT_MAX)/sizeof(record))Terminate("BH local MPI record overflow");
  int bytes=local.size()*sizeof(record);std::vector<int> sizes(NTask),offsets(NTask);MPI_Allgather(&bytes,1,MPI_INT,sizes.data(),1,MPI_INT,Communicator);
  long long total=0;for(int r=0;r<NTask;r++){if(total>INT_MAX)Terminate("BH replicated MPI record overflow");offsets[r]=total;total+=sizes[r];}
  if(total>INT_MAX || total/sizeof(record)>size_t(All.BHMaximumCount) || total>All.MaxMemSize*1024.*1024.*.25)Terminate("BH dynamics census exceeds allocation allowance");
  std::vector<record> holes(total/sizeof(record));MPI_Allgatherv(local.data(),bytes,MPI_BYTE,holes.data(),sizes.data(),offsets.data(),MPI_BYTE,Communicator);
  std::sort(holes.begin(),holes.end(),[](const record&a,const record&b){return a.id<b.id;});
  for(size_t j=1;j<holes.size();j++)if(holes[j].id==holes[j-1].id)Terminate("Duplicate BH dynamics identity");
  auto owner=[&](uint64_t id){for(int i=0;i<Sp.NumPart;i++)if(Sp.P[i].getType()==5 && Sp.P[i].ID.get()==id)return i;return -1;};
  auto write=[&](record &b){int i=owner(b.id);if(i>=0){auto &p=Sp.P[i];
#ifdef PERIODIC
    MyIntPosType previous[3],next[3];Sp.intpos_to_intpos(p.IntPos,previous);Sp.intpos_to_intpos(b.pos,next);
    for(int k=0;k<3;k++)if(previous[k]!=next[k]) {
#if defined(GRAVITY_TALLBOX) || defined(LONG_X_BITS) || defined(LONG_Y_BITS) || defined(LONG_Z_BITS)
      if(p.ResearchPeriodicImages.status==research_periodic_images::supported)
        p.ResearchPeriodicImages.status=research_periodic_images::ambiguous_relocation;
#else
      // b.pos changes only through the explicitly nearest-image BH COM
      // assignment below. Record that operation at commit, not from outputs.
      research_periodic_images::nearest_relocation(p.ResearchPeriodicImages,k,previous[k],next[k],BITS_FOR_POSITIONS);
#endif
    }
#endif
    p.ResearchSourceState=b.state;p.BHMass=b.physical;p.BHCumulativeRadiatedMass=b.radiated_mass;p.BHCumulativeFeedbackEnergy=b.feedback_energy;p.setMass(b.mass);for(int k=0;k<3;k++){p.Vel[k]=b.velocity[k]/All.UnitVelocity_in_cm_per_s*All.cf_atime;p.IntPos[k]=b.pos[k];}}};
  /* Collisionless bath velocity kicks have exactly zero mass-weighted momentum.
   * Solve their quadratic amplitude from the requested extensive heat; IDs and
   * persistent counters define directions, independently of MPI order. */
  auto heat_bath=[&](const std::vector<int>&indices,double energy,uint64_t id,uint64_t counter){
    if(energy==0)return vec{};double sums[4]={},global[4]={};std::vector<vec> kicks(indices.size());
    for(size_t j=0;j<indices.size();j++){int i=indices[j];double m=Sp.P[i].getMass()*massunit;sums[0]+=m;
      for(int k=0;k<3;k++){uint64_t bits=keyed_phase(Sp.P[i].ID.get(),id,counter+uint64_t(k),seed);kicks[j][k]=2*double(bits>>11)*std::ldexp(1.0,-53)-1;sums[k+1]+=m*kicks[j][k];}}
    MPI_Allreduce(sums,global,4,MPI_DOUBLE,MPI_SUM,Communicator);if(!(global[0]>0))Terminate("BH scattering energy lacks a measured bath");
    double ab[2]={},gab[2]={};for(size_t j=0;j<indices.size();j++){int i=indices[j];double m=Sp.P[i].getMass()*massunit;vec v{};
      for(int k=0;k<3;k++){kicks[j][k]-=global[k+1]/global[0];v[k]=Sp.P[i].Vel[k]/All.cf_atime*All.UnitVelocity_in_cm_per_s;}ab[0]+=.5*m*dot(kicks[j],kicks[j]);ab[1]+=m*dot(v,kicks[j]);}
    MPI_Allreduce(ab,gab,2,MPI_DOUBLE,MPI_SUM,Communicator);if(!(gab[0]>0) || energy<0)Terminate("BH conservative bath kick has no independent degrees of freedom");
    double radical=sqrt(gab[1]*gab[1]+4*gab[0]*energy),scale=gab[1]>=0 ? 2*energy/(radical+gab[1]) : (radical-gab[1])/(2*gab[0]);
    auto center=std::find_if(holes.begin(),holes.end(),[&](const record&x){return x.id==id;});if(center==holes.end())Terminate("BH bath lacks its exact center identity");
    vec angular{},globalangular{};
    for(size_t j=0;j<indices.size();j++){int i=indices[j];double dr[3];Sp.nearest_image_intpos_to_pos(Sp.P[i].IntPos,center->pos,dr);vec r{{dr[0]*properlength,dr[1]*properlength,dr[2]*properlength}};
      angular=add(angular,cross(r,mul(kicks[j],Sp.P[i].getMass()*massunit*scale)));for(int k=0;k<3;k++)Sp.P[i].Vel[k]+=scale*kicks[j][k]/All.UnitVelocity_in_cm_per_s*All.cf_atime;}
    MPI_Allreduce(angular.data(),globalangular.data(),3,MPI_DOUBLE,MPI_SUM,Communicator);return globalangular;
  };
  /* DF uses only impact parameters below the true force support. The force is
   * set by physical BH mass, while velocity response uses composite gravity mass.
   * Measured slow particles receive opposite momentum and the dissipated heat. */
  for(auto &b:holes)if(!b.state.bh_companion_id){
    double remaining=proper_seconds(b.state.bh_last_dynamics_time,All.Time);int steps=0;
    while(remaining>0){if(++steps>All.BHDynamicsMaximumSubcycles)Terminate("BH friction source budget exhausted");
      double bulk[4]={},gbulk[4]={};std::vector<int> candidates,slow;
      for(int i=0;i<Sp.NumPart;i++)if(Sp.P[i].getType()>=1 && Sp.P[i].getType()<=4 && Sp.P[i].getMass()>0){double d[3];Sp.nearest_image_intpos_to_pos(Sp.P[i].IntPos,b.pos,d);double rr=0;for(double x:d)rr+=x*x;
        if(rr<radius*radius){candidates.push_back(i);double m=Sp.P[i].getMass()*massunit;bulk[0]+=m;for(int k=0;k<3;k++)bulk[k+1]+=m*Sp.P[i].Vel[k]/All.cf_atime*All.UnitVelocity_in_cm_per_s;}}
      MPI_Allreduce(bulk,gbulk,4,MPI_DOUBLE,MPI_SUM,Communicator);if(!(gbulk[0]>0))break;
      vec mean{};for(int k=0;k<3;k++)mean[k]=gbulk[k+1]/gbulk[0];double speed=norm(add(b.velocity,mul(mean,-1)));if(speed==0)break;
      double sums[5]={},gs[5]={};for(int i:candidates){vec v{};for(int k=0;k<3;k++)v[k]=Sp.P[i].Vel[k]/All.cf_atime*All.UnitVelocity_in_cm_per_s;
        double d[3];Sp.nearest_image_intpos_to_pos(Sp.P[i].IntPos,b.pos,d);double dist=0;for(double x:d)dist+=x*x;
        if(norm(add(v,mul(mean,-1)))<speed && dist*properlength*properlength<b.support*b.support){slow.push_back(i);double m=Sp.P[i].getMass()*massunit;sums[0]+=m;for(int k=0;k<3;k++)sums[k+1]+=m*v[k];sums[4]++;}}
      MPI_Allreduce(sums,gs,5,MPI_DOUBLE,MPI_SUM,Communicator);if(gs[4]<2 || gs[0]==0)break;
      for(int k=0;k<3;k++)mean[k]=gs[k+1]/gs[0];vec relative=add(b.velocity,mul(mean,-1));speed=norm(relative);if(speed==0)break;
      double minimpact=std::max(All.BHMinimumImpactProperKpc*kpc,G*b.physical*massunit/(speed*speed));if(minimpact>=b.support)break;
      double rho=gs[0]/(4*M_PI/3*pow(b.support,3)),force=4*M_PI*G*G*pow(b.physical*massunit,2)*rho*log(b.support/minimpact)/(speed*speed);
      double dt=std::min(remaining,All.BHDynamicsFraction*speed/(force/(b.mass*massunit)+force/gs[0]));if(!(dt>0) || remaining-dt==remaining)Terminate("BH friction timestep underflow");
      vec impulse=mul(relative,-force*dt/speed),before=b.velocity;
      b.velocity=add(b.velocity,mul(impulse,1/(b.mass*massunit)));
      for(int i:slow)for(int k=0;k<3;k++)Sp.P[i].Vel[k]-=impulse[k]/gs[0]/All.UnitVelocity_in_cm_per_s*All.cf_atime;
      // Evaluate relative kinetic work directly: subtracting two extensive
      // bulk kinetic energies destroys precision for a small friction kick.
      double heat=-dot(impulse,relative)-.5*dot(impulse,impulse)*(1/(b.mass*massunit)+1/gs[0]);
      if(!(heat>=0) || !std::isfinite(heat))Terminate("BH friction energy budget has invalid heat");
      if(b.state.bh_event_counter==UINT64_MAX)Terminate("BH event counter overflow");++b.state.bh_event_counter;heat_bath(slow,heat,b.id,b.state.bh_event_counter);b.state.bh_friction_work_erg+=heat;
      if(owner(b.id)>=0)append(rows,prefix("friction",b.id,b.state.bh_event_counter)+",\"impulse_g_cm_s\":"+vector_json(impulse)+",\"bath_heat_erg\":"+scalar_json(heat)+",\"velocity_before_cm_s\":"+vector_json(before)+",\"velocity_after_cm_s\":"+vector_json(b.velocity)+"}");
      remaining-=dt;
    }b.state.bh_last_dynamics_time=All.Time;write(b);
  }
  /* Deterministic disjoint pairing. A captured bound, hard, quasi-circular pair
   * becomes a shared COM gravity carrier; its internal orbit is stored privately.
   * This prevents a softened two-body force from duplicating private hardening. */
  for(size_t j=0;j<holes.size();j++){auto &b=holes[j];if(b.mass<=0 || b.state.bh_companion_id)continue;
    for(size_t k=j+1;k<holes.size();k++){auto &d=holes[k];if(d.mass<=0 || d.state.bh_companion_id)continue;
      double dr[3];Sp.nearest_image_intpos_to_pos(d.pos,b.pos,dr);vec r{},v=add(d.velocity,mul(b.velocity,-1));for(int z=0;z<3;z++)r[z]=dr[z]*properlength;
      double distance=norm(r),support=std::min(b.support,d.support);if(!(distance>0 && distance<support))continue;
      double mu=G*(b.physical+d.physical)*massunit,E=.5*dot(v,v)-mu/distance;
      double resolved_energy=.5*dot(v,v)+mu*softened_potential(distance,std::max(b.support,d.support));if(E>=0 || resolved_energy>=0)continue;
      vec angular=cross(r,v);if(norm(angular)==0)continue;double a=-mu/(2*E),ecc=norm(add(mul(cross(v,angular),1/mu),mul(r,-1/distance)));
      if(ecc>All.BHBinaryMaximumEccentricity || a>=support)continue;
      double env[5]={},genv[5]={};for(int i=0;i<Sp.NumPart;i++)if(Sp.P[i].getType()==4 && Sp.P[i].getMass()>0){double dx[3];Sp.nearest_image_intpos_to_pos(Sp.P[i].IntPos,b.pos,dx);double rr=0;for(double x:dx)rr+=x*x;if(rr<radius*radius){double m=Sp.P[i].getMass()*massunit;env[0]+=m;vec sv{};for(int z=0;z<3;z++){sv[z]=Sp.P[i].Vel[z]/All.cf_atime*All.UnitVelocity_in_cm_per_s;env[z+1]+=m*sv[z];}env[4]+=m*dot(sv,sv);}}
      MPI_Allreduce(env,genv,5,MPI_DOUBLE,MPI_SUM,Communicator);double dispersion=0;if(genv[0]>0){dispersion=genv[4]/genv[0];for(int z=0;z<3;z++)dispersion-=pow(genv[z+1]/genv[0],2);dispersion=std::max(0.,dispersion/3);}
      if(dispersion>0 && a>G*b.physical*d.physical*massunit/(4*(b.physical+d.physical)*dispersion))continue;
      vec comv=mul(add(mul(b.velocity,b.mass),mul(d.velocity,d.mass)),1/(b.mass+d.mass));double xyz[3];Sp.intpos_to_pos(b.pos,xyz);for(int z=0;z<3;z++){xyz[z]+=dr[z]*d.mass/(b.mass+d.mass);xyz[z]=fmod(fmod(xyz[z],All.BoxSize)+All.BoxSize,All.BoxSize);}Sp.pos_to_intpos(xyz,b.pos);for(int z=0;z<3;z++)d.pos[z]=b.pos[z];
      double reduced_gravity=b.mass*d.mass/(b.mass+d.mass)*massunit,private_binding=-G*b.physical*d.physical*massunit*massunit/(2*a);
      double pair_potential_before=G*b.mass*d.mass*massunit*massunit*softened_potential(distance,std::max(b.support,d.support));
      double pair_potential_after=G*b.mass*d.mass*massunit*massunit*softened_potential(0,std::max(b.support,d.support));
      b.state.bh_binary_constraint_energy_erg+=.5*reduced_gravity*dot(v,v)+pair_potential_before-pair_potential_after-private_binding;
      vec angular_correction=mul(angular,reduced_gravity-b.physical*d.physical/(b.physical+d.physical)*massunit);
      for(int z=0;z<3;z++)b.state.bh_binary_angular_reservoir[z]+=angular_correction[z];
      b.velocity=d.velocity=comv;b.state.bh_companion_id=d.id;d.state.bh_companion_id=b.id;b.state.bh_binary_a_cm=a;b.state.bh_binary_eccentricity=ecc;b.state.bh_binary_last_time=All.Time;b.state.bh_binary_previous_physical_mass_g[0]=b.physical*massunit;b.state.bh_binary_previous_physical_mass_g[1]=d.physical*massunit;
      if(b.state.bh_binary_counter==UINT64_MAX)Terminate("BH binary capture counter overflow");++b.state.bh_binary_counter;
      for(int z=0;z<3;z++){b.state.bh_binary_axis[z]=unit(angular)[z];b.state.bh_binary_radial[z]=unit(r)[z];b.state.bh_binary_spin2[z]=d.state.bh_spin[z];}
      b.state.bh_binary_phase=2*M_PI*double(keyed_phase(b.id,d.id,b.state.bh_binary_counter,seed)>>11)*std::ldexp(1.0,-53);
      if(owner(b.id)>=0){std::ostringstream s;s<<std::setprecision(17)<<prefix("binary_capture",b.id,b.state.bh_binary_counter)<<",\"companion_id\":"<<d.id<<",\"semimajor_axis_cm\":"<<a<<",\"eccentricity\":"<<ecc<<",\"private_binding_energy_erg\":"<<-G*b.physical*d.physical*massunit*massunit/(2*a)<<",\"constraint_reservoir_energy_erg\":"<<b.state.bh_binary_constraint_energy_erg<<",\"angular_reservoir_g_cm2_s\":"<<vector_json(vec{{b.state.bh_binary_angular_reservoir[0],b.state.bh_binary_angular_reservoir[1],b.state.bh_binary_angular_reservoir[2]}})<<",\"capture_model\":\"quasi_circular_COM_carrier\"}";append(rows,s.str());}
      write(b);write(d);break;
    }
  }
  for(auto &b:holes)if(b.mass>0 && b.state.bh_companion_id && b.id<b.state.bh_companion_id){
    auto found=std::find_if(holes.begin(),holes.end(),[&](const record &x){return x.id==b.state.bh_companion_id && x.mass>0;});if(found==holes.end() || found->state.bh_companion_id!=b.id)Terminate("BH binary companion reciprocity failed");auto &d=*found;
    /* Re-project any resolved differential drift to COM, preserving total
     * linear momentum. The discarded relative coarse motion is recorded. */
    vec comv=mul(add(mul(b.velocity,b.mass),mul(d.velocity,d.mass)),1/(b.mass+d.mass));
    vec coarse_relative=add(b.velocity,mul(d.velocity,-1));
    double constraint_heat=.5*massunit*b.mass*d.mass/(b.mass+d.mass)*dot(coarse_relative,coarse_relative);
    if(constraint_heat< -1.e-12*massunit*(b.mass+d.mass)*std::max(1.,dot(comv,comv)))Terminate("BH COM constraint energy became negative");
    b.state.bh_binary_constraint_energy_erg+=std::max(0.,constraint_heat);b.velocity=d.velocity=comv;
    double dx[3],xyz[3];Sp.nearest_image_intpos_to_pos(d.pos,b.pos,dx);Sp.intpos_to_pos(b.pos,xyz);for(int z=0;z<3;z++){xyz[z]+=dx[z]*d.mass/(b.mass+d.mass);xyz[z]=fmod(fmod(xyz[z],All.BoxSize)+All.BoxSize,All.BoxSize);}Sp.pos_to_intpos(xyz,b.pos);for(int z=0;z<3;z++)d.pos[z]=b.pos[z];
    double remaining=proper_seconds(b.state.bh_binary_last_time,All.Time),m1=b.physical*massunit,m2=d.physical*massunit,a=b.state.bh_binary_a_cm,e=b.state.bh_binary_eccentricity;
    double previous_mass=b.state.bh_binary_previous_physical_mass_g[0]+b.state.bh_binary_previous_physical_mass_g[1];
    if(!(previous_mass>0))Terminate("BH binary restart lacks its adiabatic mass reference");
    // Slow spherical accretion: a M_total is invariant in the declared
    // orbit-averaged model, and eccentricity remains unchanged.
    a*=previous_mass/(m1+m2);b.state.bh_binary_previous_physical_mass_g[0]=m1;b.state.bh_binary_previous_physical_mass_g[1]=m2;
    vec s1{},s2{},L{},radial{};for(int z=0;z<3;z++){s1[z]=b.state.bh_spin[z];s2[z]=d.state.bh_spin[z];L[z]=b.state.bh_binary_axis[z];radial[z]=b.state.bh_binary_radial[z];}
    double gate=10*G*(m1+m2)/(c*c);int steps=0;
    while(remaining>0 && a>gate){if(++steps>All.BHDynamicsMaximumSubcycles)Terminate("BH binary hardening source budget exhausted");
      std::vector<int> stars;double env[6]={},genv[6]={};for(int i=0;i<Sp.NumPart;i++)if(Sp.P[i].getType()==4 && Sp.P[i].getMass()>0){double dd[3];Sp.nearest_image_intpos_to_pos(Sp.P[i].IntPos,b.pos,dd);double rr=0;for(double x:dd)rr+=x*x;if(rr<radius*radius){stars.push_back(i);double m=Sp.P[i].getMass()*massunit;env[0]+=m;env[5]++;vec v{};for(int z=0;z<3;z++){v[z]=Sp.P[i].Vel[z]/All.cf_atime*All.UnitVelocity_in_cm_per_s;env[z+1]+=m*v[z];}env[4]+=m*dot(v,v);}}
      MPI_Allreduce(env,genv,6,MPI_DOUBLE,MPI_SUM,Communicator);double variance=0;if(genv[0]>0){variance=genv[4]/genv[0];for(int z=0;z<3;z++)variance-=pow(genv[z+1]/genv[0],2);variance=std::max(0.,variance/3);}
      double hard=(variance>0 && genv[5]>=2) ? All.BHStellarHardeningCoefficient*G*genv[0]/(4*M_PI/3*pow(radius*properlength,3))/sqrt(variance) : 0;
      double coeff=G*G*G*m1*m2*(m1+m2)/pow(c,5),gw=64./5*coeff/pow(a,3)*(1+73./24*e*e+37./96*pow(e,4))/pow(1-e*e,3.5),stellar=hard*a*a;
      double de=-304./15*coeff/pow(a,4)*e*(1+121./304*e*e)/pow(1-e*e,2.5);vec ds1,ds2,dL;spin_rhs(m1,m2,a,s1,s2,L,ds1,ds2,dL);
      double spinrate=std::max({norm(ds1),norm(ds2),norm(dL)}),dt=std::min(remaining,All.BHDynamicsFraction*a/(gw+stellar));
      if(spinrate>0)dt=std::min(dt,All.BHDynamicsFraction/spinrate);dt=std::min(dt,(a-gate)/(gw+stellar));if(!(dt>0) || remaining-dt==remaining)Terminate("BH hardening timestep underflow");
      double next=std::max(gate,a-(gw+stellar)*dt),energy=G*m1*m2/2*(1/next-1/a),starenergy=energy*stellar/(gw+stellar);
      if(b.state.bh_event_counter==UINT64_MAX)Terminate("BH binary event counter overflow");++b.state.bh_event_counter;
      double reduced=m1*m2/(m1+m2),next_e=std::max(0.,e+de*dt);
      double oldL=reduced*sqrt(G*(m1+m2)*a*(1-e*e)),newL=reduced*sqrt(G*(m1+m2)*next*(1-next_e*next_e));
      double gw_a=std::max(next,a-gw*dt),gwL=reduced*sqrt(G*(m1+m2)*gw_a*(1-next_e*next_e));
      vec radiatedL=mul(L,std::max(0.,oldL-gwL)),stellarL=mul(L,oldL-newL-dot(radiatedL,L)),actualBathL{};
      if(starenergy>0)actualBathL=heat_bath(stars,starenergy,b.id,b.state.bh_event_counter);
      for(int z=0;z<3;z++){b.state.bh_binary_gw_angular_momentum[z]+=radiatedL[z];b.state.bh_binary_angular_reservoir[z]+=stellarL[z]-actualBathL[z];}b.state.bh_binary_stellar_energy_erg+=starenergy;b.state.bh_binary_gw_energy_erg+=energy-starenergy;
      advance_spins(m1,m2,(a+next)/2,dt,s1,s2,L,radial,b.state.bh_binary_phase);a=next;e=next_e;remaining-=dt;
    }
    b.state.bh_binary_a_cm=a;b.state.bh_binary_eccentricity=e;b.state.bh_binary_last_time=All.Time;for(int z=0;z<3;z++){b.state.bh_spin[z]=s1[z];d.state.bh_spin[z]=s2[z];b.state.bh_binary_axis[z]=L[z];b.state.bh_binary_radial[z]=radial[z];}
    if(a<=gate){
      remnant f=fit(m1,m2,s1,s2,L,radial,b.state.bh_binary_phase);double newgravity=(b.mass+d.mass)*massunit-f.radiated;vec kick=mul(f.kick,f.mass/newgravity),newv=add(comv,kick),radiationP=add(mul(comv,f.radiated),mul(f.kick,-f.mass));
      double private_binding=-G*m1*m2/(2*a);
      double kinetic_before=.5*(b.mass+d.mass)*massunit*dot(comv,comv),kinetic_after=.5*newgravity*dot(newv,newv);
      // The fitted mass deficit includes inspiral from infinity. The private
      // orbit already radiated/heated its bath down to the merger gate; retain
      // only post-gate radiation to avoid double-counting the private inspiral.
      double radiation_energy=f.radiated*c*c+private_binding+kinetic_before-kinetic_after;
      if(!(newgravity>=f.mass && radiation_energy>0) || norm(radiationP)*c>radiation_energy)Terminate("BH fitted recoil violates composite/radiation energy-momentum budget");
      // Retain the removed softened COM pair potential in the constraint
      // reservoir rather than silently losing it or heating ambient gas.
      b.state.bh_binary_constraint_energy_erg+=G*b.mass*d.mass*massunit*massunit*softened_potential(0,std::max(b.support,d.support));
      vec orbital=mul(L,m1*m2/(m1+m2)*sqrt(G*(m1+m2)*a*(1-e*e)));
      vec initial_spin=mul(add(mul(s1,m1*m1),mul(s2,m2*m2)),G/c),final_spin=mul(f.spin,G*f.mass*f.mass/c);
      vec merger_angular=add(add(orbital,initial_spin),mul(final_spin,-1));
      for(int z=0;z<3;z++)b.state.bh_binary_gw_angular_momentum[z]+=merger_angular[z];
      uint64_t counter=b.state.bh_event_counter;if(counter==UINT64_MAX)Terminate("BH merger counter overflow");++counter;
      if(owner(b.id)>=0){std::ostringstream s;s<<std::setprecision(17)<<prefix("merger",b.id,counter)<<",\"parent_ids\":["<<b.id<<','<<d.id<<"],\"gravity_mass_before_g\":["<<b.mass*massunit<<','<<d.mass*massunit<<"],\"physical_mass_before_g\":["<<m1<<','<<m2<<"],\"gravity_mass_after_g\":"<<newgravity<<",\"physical_mass_after_g\":"<<f.mass<<",\"radiated_mass_g\":"<<f.radiated<<",\"radiated_energy_erg\":"<<radiation_energy<<",\"radiated_momentum_g_cm_s\":"<<vector_json(radiationP)<<",\"velocity_before_cm_s\":"<<vector_json(comv)<<",\"velocity_after_cm_s\":"<<vector_json(newv)<<",\"physical_recoil_cm_s\":"<<vector_json(f.kick)<<",\"composite_recoil_cm_s\":"<<vector_json(kick)<<",\"spin\":"<<vector_json(f.spin)<<",\"fit_reference_gm_c2\":10,\"rest_mass_deficit_energy_erg\":"<<f.radiated*c*c<<",\"private_binding_energy_erg\":"<<private_binding<<",\"composite_kinetic_before_erg\":"<<kinetic_before<<",\"composite_kinetic_after_erg\":"<<kinetic_after<<",\"post_gate_radiated_angular_momentum_g_cm2_s\":"<<vector_json(merger_angular)<<",\"inspiral_gw_energy_erg\":"<<b.state.bh_binary_gw_energy_erg<<",\"stellar_scattering_energy_erg\":"<<b.state.bh_binary_stellar_energy_erg<<'}';append(rows,s.str());}
      b.radiated_mass+=d.radiated_mass+f.radiated/massunit;b.feedback_energy+=d.feedback_energy;
      try {research_bh_seed::merge_storage(b.state,d.state);}
      catch(const std::exception &error){Terminate("Invalid BH seed ledger during merger: %s",error.what());}
      const auto previous_roundoff=b.state;
      try {research_bh_accretion::merge_state(b.state,d.state);}
      catch(const std::exception &error){Terminate("Invalid BH accretion integral during merger: %s",error.what());}
      if(owner(b.id)>=0) {
        std::ostringstream row;row<<std::setprecision(17)<<prefix("accretion_merge",b.id,counter)
          <<",\"version\":1,\"parent_ids\":["<<b.id<<','<<d.id<<']'
          <<",\"pending_mass_before_native\":["<<previous_roundoff.bh_accretion_pending_mass<<','<<d.state.bh_accretion_pending_mass<<']'
          <<",\"pending_mass_correction_before_native\":["<<previous_roundoff.bh_accretion_pending_correction<<','<<d.state.bh_accretion_pending_correction<<']'
          <<",\"gravity_mass_correction_before_native\":["<<previous_roundoff.bh_gravity_mass_correction<<','<<d.state.bh_gravity_mass_correction<<']'
          <<",\"physical_mass_correction_before_native\":["<<previous_roundoff.bh_physical_mass_correction<<','<<d.state.bh_physical_mass_correction<<']'
          <<",\"pending_mass_after_native\":"<<b.state.bh_accretion_pending_mass
          <<",\"pending_mass_correction_after_native\":"<<b.state.bh_accretion_pending_correction
          <<",\"gravity_mass_correction_after_native\":"<<b.state.bh_gravity_mass_correction
          <<",\"physical_mass_correction_after_native\":"<<b.state.bh_physical_mass_correction<<'}';
        append(rows,row.str());
      }
      b.state.bh_feedback_reservoir+=d.state.bh_feedback_reservoir;
      b.state.bh_advected_internal_energy+=d.state.bh_advected_internal_energy;
      b.state.bh_kinetic_feedback_energy+=d.state.bh_kinetic_feedback_energy;
      b.state.bh_friction_work_erg+=d.state.bh_friction_work_erg;
      b.state.bh_binary_constraint_energy_erg+=d.state.bh_binary_constraint_energy_erg;
      for(int z=0;z<3;z++){b.state.bh_binary_angular_reservoir[z]+=d.state.bh_binary_angular_reservoir[z];b.state.bh_binary_gw_angular_momentum[z]+=d.state.bh_binary_gw_angular_momentum[z];}
      b.mass=newgravity/massunit;b.physical=f.mass/massunit;b.velocity=newv;b.state.bh_event_counter=counter;b.state.bh_parent_ids[0]=b.id;b.state.bh_parent_ids[1]=d.id;
      if(std::max(b.state.bh_generation,d.state.bh_generation)==UINT64_MAX)Terminate("BH ancestry generation overflow");b.state.bh_generation=std::max(b.state.bh_generation,d.state.bh_generation)+1;b.state.bh_companion_id=0;for(int z=0;z<3;z++)b.state.bh_spin[z]=f.spin[z];
      int loser=owner(d.id);if(loser>=0){Sp.P[loser].setMass(0);Sp.P[loser].BHMass=0;Sp.P[loser].ID.set(0);Sp.P[loser].ResearchSourceState={};Sp.P[loser].ResearchSourceState.bh_retired_particle_id=d.id;Sp.P[loser].ResearchSourceState.bh_retirement_survivor_id=b.id;Sp.P[loser].ResearchSourceState.bh_retirement_native_time=All.Time;}d.mass=d.physical=0;
    }else if(owner(b.id)>=0){std::ostringstream s;s<<std::setprecision(17)<<prefix("binary_hardening",b.id,b.state.bh_event_counter)<<",\"companion_id\":"<<d.id<<",\"semimajor_axis_cm\":"<<a<<",\"eccentricity\":"<<e<<",\"gw_energy_erg\":"<<b.state.bh_binary_gw_energy_erg<<",\"stellar_energy_erg\":"<<b.state.bh_binary_stellar_energy_erg<<",\"constraint_reservoir_energy_erg\":"<<b.state.bh_binary_constraint_energy_erg<<",\"angular_reservoir_g_cm2_s\":"<<vector_json(vec{{b.state.bh_binary_angular_reservoir[0],b.state.bh_binary_angular_reservoir[1],b.state.bh_binary_angular_reservoir[2]}})<<",\"gw_angular_momentum_g_cm2_s\":"<<vector_json(vec{{b.state.bh_binary_gw_angular_momentum[0],b.state.bh_binary_gw_angular_momentum[1],b.state.bh_binary_gw_angular_momentum[2]}})<<'}';append(rows,s.str());}
    write(b);if(d.mass>0)write(d);
  }
  publish(rows,Communicator);
}
#endif
