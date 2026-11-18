/* Native directed small-angle drag plus transverse momentum diffusion.
 * Relative velocity shrinks along its old direction by transfer_depth, and
 * receives an isotropic transverse increment restoring the removed kinetic
 * energy. The mass-weighted pair COM is retained by commit_pair_rotation.
 * This is a small-angle effective operator, not sampled microscopic events.
 * Reference: Fischer et al. 2012.10277; Arido et al. 2410.07175.
 */
#ifndef RESEARCH_FREQUENT_SIDM_PAIR_H
#define RESEARCH_FREQUENT_SIDM_PAIR_H
#include "research_rare_pair.h"
#include "research_angular_transport.h"

namespace research_sidm {
inline pair_result apply_frequent_pair(pair_particle &left,pair_particle &right,const rare_parameters &parameters,
                                      double small_angle_cutoff,double maximum_transfer_depth,double maximum_moment_ratio,
                                      std::uint64_t integer_interval,std::uint64_t &persistent_attempt,
                                      double native_time,double scale_factor,std::uint32_t logging_rank,
                                      std::uint64_t &persistent_update_sequence,research_sidm_event_log *log)
{
  source_cpu_scope resource_scope(parameters.resources,2);
  if(left.id>=right.id)throw std::runtime_error("Frequent SIDM requires one canonical unordered pair");
  require_positive(left.mass_msun,"Frequent SIDM left macro mass invalid");
  require_positive(right.mass_msun,"Frequent SIDM right macro mass invalid");
  require_positive(parameters.physical_dt_seconds,"Frequent SIDM actual physical interval invalid");
  require_positive(parameters.cross_section_per_mass_cm2_g,"Frequent SIDM cross section invalid",true);
  require_positive(parameters.differential_normalization,"Frequent SIDM differential normalization invalid",true);
  require_positive(parameters.differential_speed_scale,"Frequent SIDM differential speed mapping invalid");
  require_positive(parameters.kernel_overlap_inverse_cm3,"Frequent SIDM kernel overlap invalid",true);
  require_positive(parameters.macro_mass_relative_tolerance,"Frequent SIDM macro mass tolerance invalid",true);
  require_positive(parameters.conservation_relative_tolerance,"Frequent SIDM conservation bound invalid");
  require_positive(parameters.energy_absolute_floor,"Frequent SIDM energy floor invalid");
  require_positive(parameters.momentum_absolute_floor,"Frequent SIDM momentum floor invalid");
  require_positive(maximum_transfer_depth,"Frequent SIDM source-step limit invalid");
  require_positive(maximum_moment_ratio,"Frequent SIDM angular approximation limit invalid");
  if(maximum_transfer_depth>=1 || maximum_moment_ratio>=2)
    throw std::runtime_error("Frequent SIDM controls lie outside small-increment physical domain");
  if(left.mass_msun!=right.mass_msun)
    throw std::runtime_error("Frequent SIDM unequal-resolution pair requires calibrated weighted operator");
  if(persistent_attempt==std::numeric_limits<std::uint64_t>::max())
    throw std::runtime_error("Frequent SIDM persistent update counter exhausted");
  vector3 relative;
  for(int k=0;k<3;k++) {
    if(!std::isfinite(left.velocity_kms[k]) || !std::isfinite(right.velocity_kms[k]))
      throw std::runtime_error("Frequent SIDM native pair velocity invalid");
    relative[k]=left.velocity_kms[k]-right.velocity_kms[k];
  }
  const double speed=norm(relative);
  const bool disabled=parameters.differential_model?parameters.differential_normalization==0:
      parameters.cross_section_per_mass_cm2_g==0;
  if(speed==0 || disabled || parameters.kernel_overlap_inverse_cm3==0) {
    pair_result result;result.relative_speed_before_kms=speed;result.relative_speed_after_kms=speed;
    ++persistent_attempt;return result;
  }
  const auto moments=pair_angular_transport(parameters,speed,small_angle_cutoff);
  if(moments.neglected_moment_ratio>maximum_moment_ratio)
    throw std::runtime_error("Frequent SIDM angular model exceeds the frozen neglected-moment ratio; use rare or split controls");
  const double sigma=parameters.differential_model?parameters.differential_normalization*parameters.differential_model->total_cross_section(pair_differential_speed(parameters,speed)):parameters.cross_section_per_mass_cm2_g;
  const long double total_depth=static_cast<long double>(sigma)*left.mass_msun*
      1.98847e33L*speed*1.e5L*parameters.kernel_overlap_inverse_cm3*parameters.physical_dt_seconds;
  const long double transfer_depth=total_depth*moments.small_transfer;
  if(!std::isfinite(total_depth) || !std::isfinite(transfer_depth) || transfer_depth>maximum_transfer_depth)
    throw std::runtime_error("Frequent SIDM actual drag increment exceeds its frozen source-step limit");
  pair_result result;result.relative_speed_before_kms=speed;result.optical_depth=double(total_depth*moments.small_probability);
  if(!std::isfinite(result.optical_depth))throw std::runtime_error("Frequent SIDM recorded exposure exceeds finite double precision");
  if(speed==0 || transfer_depth==0) {++persistent_attempt;return result;}
  const double cosine=double(1-transfer_depth);
  if(cosine==1)throw std::runtime_error("Frequent SIDM drag below finite rotation precision; no source update was fabricated");
  const double phi=6.283185307179586476925286766559*
      pair_uniform(parameters.operator_seed,left.id,right.id,integer_interval,parameters.random_substep,8,parameters.random_stage);
  result=commit_pair_rotation(left,right,parameters,cosine,phi,native_time,scale_factor,
      logging_rank,persistent_update_sequence,log);
  result.optical_depth=double(total_depth*moments.small_probability);
  ++persistent_attempt;
  return result;
}
}
#endif
