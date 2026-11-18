/* Staged native calibrated-response -> conservative-source connection.
 * The caller supplies a measured unperturbed background and separately
 * selects actual live recipient particles. No background is invented from
 * the perturbed recipient bath. No native particle or counter is committed.
 */
#ifndef RESEARCH_SIDM_DRAG_COUPLING_H
#define RESEARCH_SIDM_DRAG_COUPLING_H
#include "research_sidm_drag_source.h"
#include "research_sidm_drag_state_response.h"
#include "../data/research_source_state.h"
#include <limits>
namespace research_sidm_drag_coupling {
using research_sidm_drag_source::vec;
struct source_limits {
  research_sidm_drag_source::limits conservative;
  double maximum_proper_step_seconds,maximum_reference_error_cm_s2,maximum_heldout_error_cm_s2;
};
struct staged_result {
  research_sidm_drag_source::result conservative;
  research_sidm_drag_response::state_response::evaluation response;
  std::array<double,10> measured_query;
  double proper_start_seconds,proper_end_seconds;
  std::uint64_t source_counter,next_source_counter;
  std::array<std::uint64_t,4> response_model_words,response_signature_words;
  std::uint32_t response_checksum;
  double gravity_carrier_mass_msun;
  vec original_carrier_position_proper_kpc,original_carrier_velocity_km_s;
};
inline std::array<std::uint64_t,4> identity_words(const std::string&identity){
  if(!research_sidm_drag_response::sha256_token(identity))throw std::runtime_error("Invalid persistent drag SHA256 identity");
  std::array<std::uint64_t,4> words{};
  for(std::size_t i=0;i<64;i++){char c=identity[i];unsigned value=c<='9'?c-'0':c-'a'+10;words[i/16]=(words[i/16]<<4)|value;}
  return words;
}
inline research_source_state initialize_clock(const research_source_state&previous,
    const research_sidm_drag_response::state_response&response,double native_time,double proper_seconds){
  if(previous.bh_sidm_drag_version!=0||previous.bh_sidm_drag_counter!=0||previous.bh_sidm_drag_heat_erg!=0||
      !std::isfinite(native_time)||!std::isfinite(proper_seconds))throw std::runtime_error("Drag clock initialization would overwrite active state");
  bool inactive=previous.bh_sidm_drag_model_crc32==0&&previous.bh_sidm_drag_last_native_time==0&&previous.bh_sidm_drag_last_proper_seconds==0&&
    previous.bh_sidm_drag_reference_error_cm_s2==0&&previous.bh_sidm_drag_heldout_error_cm_s2==0;
  for(int i=0;i<4;i++)inactive=inactive&&previous.bh_sidm_drag_model_sha256[i]==0&&previous.bh_sidm_drag_signature_sha256[i]==0;
  for(double x:previous.bh_sidm_drag_last_acceleration_cm_s2)inactive=inactive&&x==0;
  if(!inactive)throw std::runtime_error("Inactive drag state contains retained model/clock data; initialization cannot erase it");
  research_source_state next=previous;next.bh_sidm_drag_version=1;next.bh_sidm_drag_counter=0;
  next.bh_sidm_drag_model_crc32=response.checksum();auto model=identity_words(response.product_identity()),signature=identity_words(response.fixed_signature());
  for(int i=0;i<4;i++){next.bh_sidm_drag_model_sha256[i]=model[i];next.bh_sidm_drag_signature_sha256[i]=signature[i];}
  next.bh_sidm_drag_last_native_time=native_time;next.bh_sidm_drag_last_proper_seconds=proper_seconds;
  next.bh_sidm_drag_heat_erg=0;next.bh_sidm_drag_reference_error_cm_s2=0;next.bh_sidm_drag_heldout_error_cm_s2=0;
  for(double&x:next.bh_sidm_drag_last_acceleration_cm_s2)x=0;return next;
}
inline research_source_state proposed_committed_state(const research_source_state&previous,const staged_result&transaction,
    const research_sidm_drag_response::state_response&response,double native_begin,double native_end,
    double erg_per_msun_km2_s2,MPI_Comm comm){
  using research_sidm_drag_source::require_collective;
  bool okay=previous.bh_sidm_drag_version==1&&previous.bh_sidm_drag_counter==transaction.source_counter&&
    transaction.source_counter<std::numeric_limits<std::uint64_t>::max()&&transaction.next_source_counter==transaction.source_counter+1&&
    previous.bh_sidm_drag_last_native_time==native_begin&&previous.bh_sidm_drag_last_proper_seconds==transaction.proper_start_seconds&&
    std::isfinite(native_begin)&&std::isfinite(native_end)&&native_end>native_begin&&
    std::isfinite(erg_per_msun_km2_s2)&&erg_per_msun_km2_s2>0&&
    previous.bh_sidm_drag_model_crc32==response.checksum()&&std::isfinite(previous.bh_sidm_drag_heat_erg)&&previous.bh_sidm_drag_heat_erg>=0;
  auto model=identity_words(response.product_identity()),signature=identity_words(response.fixed_signature());
  for(int i=0;i<4;i++)okay=okay&&previous.bh_sidm_drag_model_sha256[i]==model[i]&&previous.bh_sidm_drag_signature_sha256[i]==signature[i]&&
    transaction.response_model_words[i]==model[i]&&transaction.response_signature_words[i]==signature[i];
  okay=okay&&transaction.response_checksum==response.checksum();
  double heat=previous.bh_sidm_drag_heat_erg+transaction.conservative.heat*erg_per_msun_km2_s2;
  okay=okay&&std::isfinite(heat)&&heat>=previous.bh_sidm_drag_heat_erg&&
    std::isfinite(transaction.proper_end_seconds)&&transaction.proper_end_seconds>transaction.proper_start_seconds&&
    std::isfinite(transaction.response.reference_error)&&transaction.response.reference_error>=0&&
    std::isfinite(transaction.response.heldout_error)&&transaction.response.heldout_error>=0;
  for(double x:transaction.response.acceleration)okay=okay&&std::isfinite(x);
  require_collective(okay,comm,"Drag commit proposal changes persistent model, counter, clock or cumulative energy");
  std::array<double,4> settings{{native_begin,native_end,erg_per_msun_km2_s2,heat}},low{},high{};
  MPI_Allreduce(settings.data(),low.data(),4,MPI_DOUBLE,MPI_MIN,comm);MPI_Allreduce(settings.data(),high.data(),4,MPI_DOUBLE,MPI_MAX,comm);
  require_collective(low==high,comm,"MPI owners disagree on drag commit clock/energy conversion");
  research_source_state next=previous;next.bh_sidm_drag_counter=transaction.next_source_counter;
  next.bh_sidm_drag_last_native_time=native_end;next.bh_sidm_drag_last_proper_seconds=transaction.proper_end_seconds;next.bh_sidm_drag_heat_erg=heat;
  next.bh_sidm_drag_reference_error_cm_s2=transaction.response.reference_error;next.bh_sidm_drag_heldout_error_cm_s2=transaction.response.heldout_error;
  for(int i=0;i<3;i++)next.bh_sidm_drag_last_acceleration_cm_s2[i]=transaction.response.acceleration[i];return next;
}
inline research_source_state transition_response(const research_source_state&previous,
    const research_sidm_drag_response::state_response&old_response,const research_sidm_drag_response::state_response&new_response,
    double native_time,double proper_time){
  if(!old_response.sha256_verified()||!new_response.sha256_verified()||
      old_response.fixed_signature()!=new_response.fixed_signature()||old_response.product_identity()==new_response.product_identity()||
      previous.bh_sidm_drag_version!=1||previous.bh_sidm_drag_counter==UINT64_MAX||
      previous.bh_sidm_drag_model_crc32!=old_response.checksum()||previous.bh_sidm_drag_last_native_time!=native_time||
      previous.bh_sidm_drag_last_proper_seconds!=proper_time||!std::isfinite(native_time)||!std::isfinite(proper_time)||
      !std::isfinite(previous.bh_sidm_drag_heat_erg)||previous.bh_sidm_drag_heat_erg<0)
    throw std::runtime_error("Response transition changes physical definition or stale persistent source boundary");
  auto old_model=identity_words(old_response.product_identity()),new_model=identity_words(new_response.product_identity());
  auto signature=identity_words(old_response.fixed_signature());
  for(int i=0;i<4;i++)if(previous.bh_sidm_drag_model_sha256[i]!=old_model[i]||previous.bh_sidm_drag_signature_sha256[i]!=signature[i])
    throw std::runtime_error("Response transition does not match the retained original model identity");
  if(!std::isfinite(previous.bh_sidm_drag_reference_error_cm_s2)||previous.bh_sidm_drag_reference_error_cm_s2<0||
      !std::isfinite(previous.bh_sidm_drag_heldout_error_cm_s2)||previous.bh_sidm_drag_heldout_error_cm_s2<0)
    throw std::runtime_error("Response transition cannot erase invalid retained source errors");
  for(double value:previous.bh_sidm_drag_last_acceleration_cm_s2)if(!std::isfinite(value))
    throw std::runtime_error("Response transition cannot erase invalid retained acceleration");
  research_source_state next=previous;
  for(int i=0;i<4;i++)next.bh_sidm_drag_model_sha256[i]=new_model[i];
  next.bh_sidm_drag_model_crc32=new_response.checksum();
  // Errors/acceleration describe the last applied impulse, so retain them
  // across metadata-only model changes. The transition journal links that
  // impulse to its previous response. Clock, counter and heat never reset.
  return next;
}
inline void same_identity(const std::string&identity,MPI_Comm comm){
  using research_sidm_drag_source::require_collective;
  require_collective(research_sidm_drag_response::sha256_token(identity),comm,"Invalid calibrated source identity");
  std::array<unsigned char,64> bytes{},low{},high{};std::copy(identity.begin(),identity.end(),bytes.begin());
  MPI_Allreduce(bytes.data(),low.data(),64,MPI_UNSIGNED_CHAR,MPI_MIN,comm);
  MPI_Allreduce(bytes.data(),high.data(),64,MPI_UNSIGNED_CHAR,MPI_MAX,comm);
  require_collective(low==high,comm,"MPI ranks disagree on calibrated source identity");
}
inline staged_result stage(const research_sidm_drag_response::state_response&response,
    const std::string&actual_fixed_signature,const std::vector<research_sidm_drag_source::bath_particle>&local_recipients,
    std::uint64_t carrier_id,double gravity_carrier_mass_msun,const vec&carrier_position_proper_kpc,
    const vec&carrier_velocity_peculiar_km_s,const vec&background_mean_velocity_km_s,
    double background_density_msun_kpc3,const std::array<double,6>&background_covariance_km2_s2,
    double proper_start_seconds,double proper_end_seconds,std::uint64_t seed,std::uint64_t counter,
    const source_limits&bound,MPI_Comm comm){
  using namespace research_sidm_drag_source;
  bool okay=std::isfinite(proper_start_seconds)&&std::isfinite(proper_end_seconds)&&proper_end_seconds>proper_start_seconds&&
    std::isfinite(bound.maximum_proper_step_seconds)&&bound.maximum_proper_step_seconds>0&&
    std::isfinite(bound.maximum_reference_error_cm_s2)&&bound.maximum_reference_error_cm_s2>=0&&
    std::isfinite(bound.maximum_heldout_error_cm_s2)&&bound.maximum_heldout_error_cm_s2>=0&&counter<std::numeric_limits<std::uint64_t>::max();
  double dt=proper_end_seconds-proper_start_seconds;
  okay=okay&&std::isfinite(dt)&&dt>0&&dt<=bound.maximum_proper_step_seconds&&finite(background_mean_velocity_km_s)&&finite(carrier_velocity_peculiar_km_s)&&
    std::isfinite(background_density_msun_kpc3)&&background_density_msun_kpc3>0;
  for(double value:background_covariance_km2_s2)okay=okay&&std::isfinite(value);
  require_collective(okay,comm,"Invalid proper source interval, measured background or source counter");
  same_identity(actual_fixed_signature,comm);same_identity(response.fixed_signature(),comm);same_identity(response.product_identity(),comm);
  unsigned checksum=response.checksum(),minimum_checksum=0,maximum_checksum=0;
  MPI_Allreduce(&checksum,&minimum_checksum,1,MPI_UNSIGNED,MPI_MIN,comm);MPI_Allreduce(&checksum,&maximum_checksum,1,MPI_UNSIGNED,MPI_MAX,comm);
  require_collective(minimum_checksum==maximum_checksum,comm,"MPI ranks use different calibrated state table bytes");
  vec relative=add(carrier_velocity_peculiar_km_s,scale(background_mean_velocity_km_s,-1));
  std::array<double,10> query{{relative[0],relative[1],relative[2],background_density_msun_kpc3,
    background_covariance_km2_s2[0],background_covariance_km2_s2[1],background_covariance_km2_s2[2],
    background_covariance_km2_s2[3],background_covariance_km2_s2[4],background_covariance_km2_s2[5]}};
  std::array<double,15> replicated{};std::copy(query.begin(),query.end(),replicated.begin());
  replicated[10]=proper_start_seconds;replicated[11]=proper_end_seconds;replicated[12]=bound.maximum_proper_step_seconds;
  replicated[13]=bound.maximum_reference_error_cm_s2;replicated[14]=bound.maximum_heldout_error_cm_s2;
  std::array<double,15> low{},high{};
  MPI_Allreduce(replicated.data(),low.data(),15,MPI_DOUBLE,MPI_MIN,comm);MPI_Allreduce(replicated.data(),high.data(),15,MPI_DOUBLE,MPI_MAX,comm);
  require_collective(low==high,comm,"MPI ranks disagree on measured source state/clock/error limits");
  research_sidm_drag_response::state_response::evaluation evaluated{};bool evaluated_okay=true;
  try{evaluated=response.evaluate(query,actual_fixed_signature);}catch(const std::exception&){evaluated_okay=false;}
  require_collective(evaluated_okay,comm,"Calibrated source query is unsupported on at least one MPI owner");
  okay=evaluated.reference_error<=bound.maximum_reference_error_cm_s2&&evaluated.heldout_error<=bound.maximum_heldout_error_cm_s2;
  require_collective(okay,comm,"Calibrated source exceeds its declared reference/held-out force budgets");
  // kick rechecks actual recipient COM work, source fraction, identities,
  // geometry and rounded conservation before returning staged velocities.
  staged_result out{};
  out.gravity_carrier_mass_msun=gravity_carrier_mass_msun;out.original_carrier_position_proper_kpc=carrier_position_proper_kpc;
  out.original_carrier_velocity_km_s=carrier_velocity_peculiar_km_s;
  out.response_model_words=identity_words(response.product_identity());out.response_signature_words=identity_words(response.fixed_signature());out.response_checksum=response.checksum();
  out.conservative=kick(local_recipients,carrier_id,gravity_carrier_mass_msun,carrier_position_proper_kpc,
    carrier_velocity_peculiar_km_s,evaluated.acceleration,dt,seed,counter,bound.conservative,comm);
  out.response=evaluated;out.measured_query=query;out.proper_start_seconds=proper_start_seconds;out.proper_end_seconds=proper_end_seconds;
  out.source_counter=counter;out.next_source_counter=counter+1;return out;
}
}
#endif
