/* Directed hybrid source: half small-angle drag/diffusion, full sampled
 * large-angle source, half small-angle drag/diffusion. Both channels derive
 * from the same total cross section and angular distribution; no channel is
 * independently normalized to the full cross section. The splitting and
 * finite-small-angle errors require independent controls.
 */
#ifndef RESEARCH_HYBRID_SIDM_PAIR_H
#define RESEARCH_HYBRID_SIDM_PAIR_H
#include "research_frequent_pair.h"

namespace research_sidm {
struct hybrid_pair_result
{
  pair_result aggregate;
  bool rare_scattered=false;
  unsigned frequent_updates=0;
};
inline hybrid_pair_result apply_hybrid_pair(pair_particle &left,pair_particle &right,const rare_parameters &parameters,
                                          double split_angle,double maximum_transfer_depth,double maximum_moment_ratio,
                                          std::uint64_t integer_interval,std::uint64_t &persistent_attempt,
                                          double native_time,double scale_factor,std::uint32_t logging_rank,
                                          std::uint64_t &persistent_update_sequence,
                                          research_sidm_event_log *rare_log,research_sidm_event_log *frequent_log)
{
  if(!std::isfinite(split_angle) || split_angle<=0 || split_angle>=3.1415926535897932384626433832795)
    throw std::runtime_error("Hybrid SIDM requires a nonempty small/large-angle partition");
  if(!rare_log || !frequent_log || rare_log==frequent_log)
    throw std::runtime_error("Hybrid SIDM requires separate native rare and effective diffusion records");
  // Each frequent half checks its current angular moments and approximation
  // domain. Avoid an unconditional table query for zero-speed/disabled pairs.
  rare_parameters half=parameters;
  half.physical_dt_seconds=parameters.physical_dt_seconds/2;
  require_positive(half.physical_dt_seconds,"Hybrid SIDM half source interval below finite time precision");
  hybrid_pair_result result;
  const auto first=apply_frequent_pair(left,right,half,split_angle,maximum_transfer_depth,maximum_moment_ratio,
      integer_interval,persistent_attempt,native_time,scale_factor,logging_rank,persistent_update_sequence,frequent_log);
  const auto rare=apply_rare_pair(left,right,parameters,integer_interval,persistent_attempt,
      native_time,scale_factor,logging_rank,persistent_update_sequence,rare_log,split_angle);
  half.random_stage=parameters.random_stage+1;
  const auto second=apply_frequent_pair(left,right,half,split_angle,maximum_transfer_depth,maximum_moment_ratio,
      integer_interval,persistent_attempt,native_time,scale_factor,logging_rank,persistent_update_sequence,frequent_log);
  result.rare_scattered=rare.scattered;
  result.frequent_updates=unsigned(first.scattered)+unsigned(second.scattered);
  result.aggregate.scattered=rare.scattered || result.frequent_updates;
  result.aggregate.optical_depth=first.optical_depth+rare.optical_depth+second.optical_depth;
  result.aggregate.event_probability=rare.event_probability;
  result.aggregate.relative_speed_before_kms=first.relative_speed_before_kms;
  result.aggregate.relative_speed_after_kms=second.scattered?second.relative_speed_after_kms:
      rare.scattered?rare.relative_speed_after_kms:first.scattered?first.relative_speed_after_kms:first.relative_speed_before_kms;
  if(!std::isfinite(result.aggregate.optical_depth))throw std::runtime_error("Hybrid SIDM measured channel exposure overflow");
  return result;
}
}
#endif
