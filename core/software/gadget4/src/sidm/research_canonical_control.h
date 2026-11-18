/* Native canonical control adapter. A driver owns x=r/a, u=a*v_pec,
 * d tau=dt/a^2 and supplies a frozen positive scale factor per source step.
 * This adapter transforms the physical elastic law and proper kernel support;
 * it does not relabel canonical event velocities as physical measurements.
 */
#ifndef RESEARCH_SIDM_CANONICAL_CONTROL_H
#define RESEARCH_SIDM_CANONICAL_CONTROL_H
#include "research_rare_pair.h"
#include "research_rare_substeps.h"
#include "research_canonical_clock.h"
#include <cmath>
#include <stdexcept>

namespace research_sidm {
inline rare_parameters canonical_pair_parameters(const rare_parameters &physical,double scale,double canonical_speed,
    double canonical_dt_seconds,double canonical_overlap_inverse_cm3)
{
  if(!std::isfinite(scale) || scale<=0 || !std::isfinite(canonical_speed) || canonical_speed<0)
    throw std::runtime_error("Canonical SIDM source state outside finite positive clock domain");
  if(physical.differential_speed_scale!=1)
    throw std::runtime_error("Canonical SIDM law cannot be mapped twice");
  rare_parameters result=physical;
  const double factor=1/(scale*scale);
  if(!std::isfinite(canonical_dt_seconds) || canonical_dt_seconds<=0 || !std::isfinite(canonical_overlap_inverse_cm3) || canonical_overlap_inverse_cm3<0 || factor<=0)
    throw std::runtime_error("Canonical SIDM interval, overlap or scale precision invalid");
  result.differential_speed_scale=1/scale;
  result.differential_normalization=physical.differential_normalization*factor;
  result.cross_section_per_mass_cm2_g=physical.cross_section_per_mass_cm2_g*factor;
  result.screening_velocity_kms=physical.screening_velocity_kms*scale;
  result.physical_dt_seconds=canonical_dt_seconds;
  result.kernel_overlap_inverse_cm3=canonical_overlap_inverse_cm3;
  result.energy_absolute_floor=physical.energy_absolute_floor*scale*scale;
  result.momentum_absolute_floor=physical.momentum_absolute_floor*scale;
  result.native_velocity_unit_kms=physical.native_velocity_unit_kms*scale;
  // The source driver supplies canonical velocities. Absolute conservation
  // floors and the physical-template native rounding unit transform with u.
  for(double value:{factor,result.differential_speed_scale,result.differential_normalization,result.cross_section_per_mass_cm2_g,result.screening_velocity_kms,
      result.energy_absolute_floor,result.momentum_absolute_floor,result.native_velocity_unit_kms})
    if(!std::isfinite(value) || value<0)throw std::runtime_error("Canonical SIDM law normalization overflow");
  for(double value:{result.differential_speed_scale,result.energy_absolute_floor,result.momentum_absolute_floor,result.native_velocity_unit_kms})
    if(value<=0)throw std::runtime_error("Canonical SIDM conservation or rounding scale underflows");
  if((physical.differential_normalization>0 && result.differential_normalization==0)
      || (physical.cross_section_per_mass_cm2_g>0 && result.cross_section_per_mass_cm2_g==0)
      || ((physical.screened_rutherford || physical.born_moller) && result.screening_velocity_kms<=0))
    throw std::runtime_error("Canonical SIDM positive physical law underflows");
  pair_differential_speed(result,canonical_speed);
  return result;
}
// Prepare the analytic amplitude from the retained native physical law at
// u/a, so a velocity-dependent normalization cannot accidentally be mapped
// as a constant total cross section. A table keeps its original conditional
// angular distribution and identity; only speed and amplitude transform.
inline rare_parameters canonical_native_pair_parameters(rare_parameters physical,int angular_kind,
    double scale,double canonical_speed,double canonical_dt_seconds,double canonical_overlap_inverse_cm3)
{
  if(!std::isfinite(scale) || scale<=0 || !std::isfinite(canonical_speed) || canonical_speed<0
      || angular_kind<0 || angular_kind>3)
    throw std::runtime_error("Canonical native physical-law selection invalid");
  if(physical.screened_rutherford!=(angular_kind==1 || angular_kind==2) || physical.born_moller!=(angular_kind==2)
      || (angular_kind==2 && !physical.folded_identical_particle)
      || (angular_kind==3 && (!physical.differential_model || physical.differential_model!=native_differential_model()))
      || (angular_kind!=3 && physical.differential_model))
    throw std::runtime_error("Canonical native physical and angular law identities disagree");
  require_positive(physical.differential_normalization,"Canonical native physical normalization invalid",true);
  const double speed=canonical_speed/scale;
  if(!std::isfinite(speed) || (canonical_speed>0 && speed==0))
    throw std::runtime_error("Canonical native physical speed conversion outside precision");
  physical.cross_section_per_mass_cm2_g=canonical_speed==0?0:native_total_cross_section(angular_kind,
      physical.differential_normalization,speed,physical.screening_velocity_kms);
  return canonical_pair_parameters(physical,scale,canonical_speed,canonical_dt_seconds,canonical_overlap_inverse_cm3);
}
inline double canonical_kernel_support_cm(double proper_support_cm,double scale)
{
  if(!std::isfinite(proper_support_cm) || proper_support_cm<=0 || !std::isfinite(scale) || scale<=0)
    throw std::runtime_error("Canonical SIDM kernel support outside physical domain");
  const double radius=proper_support_cm/scale;
  if(!std::isfinite(radius) || radius<=0)throw std::runtime_error("Canonical SIDM support conversion overflow");
  return radius;
}
inline long double canonical_sigma_speed_bound(int angular_kind,long double physical_normalization,
    long double canonical_speed_upper,long double physical_screening_speed,long double scale)
{
  if(!std::isfinite(scale) || scale<=0 || !std::isfinite(canonical_speed_upper) || canonical_speed_upper<0)
    throw std::runtime_error("Canonical SIDM rate support invalid");
  const long double value=native_sigma_speed_bound(angular_kind,physical_normalization,canonical_speed_upper/scale,physical_screening_speed)/scale;
  if(!std::isfinite(value) || value<0)throw std::runtime_error("Canonical SIDM rate upper bound overflow");
  return value;
}
// The particles and overlap callback must contain canonical u and x support.
// Bounds use a^-2 sigma(u/a)*u, not a physical-law evaluation at u. Energy
// floors are transformed exactly as in canonical_pair_parameters above.
template<class Container,class Overlap>
rare_substep_plan plan_canonical_substep(const Container &particles,Overlap overlap,
    double remaining_canonical_seconds,double physical_normalization,int angular_kind,
    double physical_screening_velocity,double scale,double pair_probability_limit,
    double particle_probability_limit,double conservation_tolerance,double physical_energy_floor,
    std::uint64_t remaining_pair_visits,unsigned maximum_pair_rotations=1)
{
  if(!std::isfinite(scale) || scale<=0 || !std::isfinite(physical_energy_floor) || physical_energy_floor<=0)
    throw std::runtime_error("Canonical SIDM planner epoch or physical energy floor invalid");
  const long double floor=static_cast<long double>(physical_energy_floor)*scale*scale;
  if(!std::isfinite(floor) || floor<=0 || floor>std::numeric_limits<double>::max()
      || static_cast<double>(floor)<=0)
    throw std::runtime_error("Canonical SIDM planner energy floor outside storage precision");
  return plan_rare_substep_with_rate_bound(particles,overlap,remaining_canonical_seconds,physical_normalization,
      angular_kind,physical_screening_velocity,pair_probability_limit,particle_probability_limit,
      conservation_tolerance,static_cast<double>(floor),remaining_pair_visits,maximum_pair_rotations,
      [&](long double upper) {
        return canonical_sigma_speed_bound(angular_kind,physical_normalization,upper,physical_screening_velocity,scale);
      });
}

struct canonical_source_interval {
  double begin_tau_seconds=0,end_tau_seconds=0,canonical_dt_seconds=0;
  double begin_scale_factor=0,end_scale_factor=0,frozen_scale_factor=0;
  double logarithmic_scale_span=0,maximum_inverse_bracket_relative_width=0;
  double begin_inverse_bracket_relative_width=0,end_inverse_bracket_relative_width=0;
  double maximum_clock_absolute_error_estimate_seconds=0;
  std::size_t function_evaluations=0,inverse_iterations=0;
};

// Static native time is tau*h/UnitTime. Resolve both actual endpoints and
// the time midpoint using ONE aggregate clock-work allocation. The source
// A whole-interval frozen driver uses frozen_scale_factor and the default
// span gate. An adaptive driver explicitly disables that whole gate, then
// applies limit_canonical_expansion_step and verifies every selected substep.
// Neither policy certifies a transport truncation-error bound.
inline canonical_source_interval canonical_native_source_interval(const canonical_control_clock &clock,
    double native_begin,double native_end,double time_unit_seconds,double hubble_parameter,
    double maximum_relative_scale_span,std::size_t maximum_clock_work,bool enforce_whole_interval_span=true)
{
  if(!std::isfinite(native_begin) || native_begin<0 || !std::isfinite(native_end) || native_end<=native_begin)
    throw std::runtime_error("Canonical native source interval must have increasing nonnegative times");
  for(double value:{time_unit_seconds,hubble_parameter,maximum_relative_scale_span})
    if(!std::isfinite(value) || value<=0)
      throw std::runtime_error("Canonical native source unit or expansion budget invalid");
  if(maximum_relative_scale_span>=1 || maximum_clock_work<1)
    throw std::runtime_error("Canonical native source numerical allocation invalid");
  const long double unit=static_cast<long double>(time_unit_seconds)/hubble_parameter;
  const long double begin=static_cast<long double>(native_begin)*unit,end=static_cast<long double>(native_end)*unit;
  const long double duration=(static_cast<long double>(native_end)-native_begin)*unit;
  for(long double value:{unit,begin,end,duration})
    if(!std::isfinite(value) || value<0 || value>std::numeric_limits<double>::max())
      throw std::runtime_error("Canonical native elapsed clock conversion overflow");
  canonical_source_interval result;
  result.begin_tau_seconds=static_cast<double>(begin);result.end_tau_seconds=static_cast<double>(end);
  result.canonical_dt_seconds=static_cast<double>(duration);
  const double middle=result.begin_tau_seconds+(result.end_tau_seconds-result.begin_tau_seconds)/2;
  if(result.canonical_dt_seconds<=0 || result.end_tau_seconds<=result.begin_tau_seconds
      || middle<=result.begin_tau_seconds || middle>=result.end_tau_seconds)
    throw std::runtime_error("Canonical source interval below elapsed-clock precision");
  const auto epoch=[&](double tau) {
    if(result.function_evaluations>=maximum_clock_work)
      throw std::runtime_error("Canonical source aggregate inverse-clock work exhausted");
    const auto value=clock.epoch(tau,maximum_clock_work-result.function_evaluations);
    if(value.function_evaluations>maximum_clock_work-result.function_evaluations
        || value.iterations>std::numeric_limits<std::size_t>::max()-result.inverse_iterations)
      throw std::runtime_error("Canonical source inverse-clock accounting overflow");
    result.function_evaluations+=value.function_evaluations;result.inverse_iterations+=value.iterations;
    result.maximum_inverse_bracket_relative_width=std::max(result.maximum_inverse_bracket_relative_width,value.bracket_relative_width);
    result.maximum_clock_absolute_error_estimate_seconds=std::max(result.maximum_clock_absolute_error_estimate_seconds,value.clock_absolute_error_estimate_seconds);
    return value;
  };
  const auto first=epoch(result.begin_tau_seconds),last=epoch(result.end_tau_seconds),center=epoch(middle);
  result.begin_scale_factor=first.scale_factor;result.end_scale_factor=last.scale_factor;
  result.begin_inverse_bracket_relative_width=first.bracket_relative_width;
  result.end_inverse_bracket_relative_width=last.bracket_relative_width;
  result.frozen_scale_factor=center.scale_factor;
  if(result.end_scale_factor<result.begin_scale_factor || result.frozen_scale_factor<result.begin_scale_factor
      || result.frozen_scale_factor>result.end_scale_factor)
    throw std::runtime_error("Canonical inverse clocks do not preserve source epoch ordering");
  result.logarithmic_scale_span=std::log(result.end_scale_factor)-std::log(result.begin_scale_factor);
  // Include the numerical brackets in the declared scale-span allocation.
  // Brackets are solver estimates, not certified intervals of the exact a.
  const double span=result.logarithmic_scale_span+std::log1p(first.bracket_relative_width)
      +std::log1p(last.bracket_relative_width);
  if(!std::isfinite(span) || span<0 || (enforce_whole_interval_span && span>std::log1p(maximum_relative_scale_span)))
    throw std::runtime_error("Canonical frozen source law exceeds its expansion span allocation");
  return result;
}
struct canonical_expansion_cap {
  double seconds=0,maximum_clock_error_seconds=0,maximum_inverse_bracket_relative_width=0;
  double end_tau_seconds=0,end_scale_factor=0,end_inverse_bracket_relative_width=0;
  std::size_t function_evaluations=0;
};
inline canonical_expansion_cap limit_canonical_expansion_step(const canonical_control_clock&clock,
    double begin_tau,double end_tau,double begin_scale,double begin_bracket,double maximum_relative_span,std::size_t maximum_work)
{
  if(!std::isfinite(begin_tau)||!std::isfinite(end_tau)||end_tau<=begin_tau||!std::isfinite(begin_scale)||begin_scale<=0||
      !std::isfinite(begin_bracket)||begin_bracket<0||!std::isfinite(maximum_relative_span)||maximum_relative_span<=0||maximum_relative_span>=1||maximum_work<1)
    throw std::runtime_error("Invalid bounded canonical expansion-step domain");
  canonical_expansion_cap result;double endpoint=end_tau;
  for(unsigned trial=0;trial<64;trial++) {
    if(result.function_evaluations>=maximum_work)throw std::runtime_error("Canonical expansion limiter clock work exhausted");
    const auto epoch=clock.epoch(endpoint,maximum_work-result.function_evaluations);
    if(epoch.function_evaluations>maximum_work-result.function_evaluations)throw std::runtime_error("Canonical expansion limiter work accounting overflow");
    result.function_evaluations+=epoch.function_evaluations;
    result.maximum_clock_error_seconds=std::max(result.maximum_clock_error_seconds,epoch.clock_absolute_error_estimate_seconds);
    result.maximum_inverse_bracket_relative_width=std::max(result.maximum_inverse_bracket_relative_width,epoch.bracket_relative_width);
    if(epoch.scale_factor<begin_scale)throw std::runtime_error("Canonical expansion limiter inverse epochs regress");
    const double span=std::log(epoch.scale_factor)-std::log(begin_scale)+std::log1p(begin_bracket)+std::log1p(epoch.bracket_relative_width);
    if(!std::isfinite(span)||span<0)throw std::runtime_error("Canonical expansion limiter span invalid");
    if(span<=std::log1p(maximum_relative_span)) {
      result.seconds=endpoint-begin_tau;result.end_tau_seconds=endpoint;result.end_scale_factor=epoch.scale_factor;
      result.end_inverse_bracket_relative_width=epoch.bracket_relative_width;return result;
    }
    const double next=begin_tau+(endpoint-begin_tau)/2;
    if(!std::isfinite(next)||next<=begin_tau||next>=endpoint)throw std::runtime_error("Canonical expansion limiter loses elapsed-clock precision");
    endpoint=next;
  }
  throw std::runtime_error("Canonical expansion limiter cannot resolve its span within 64 bounded halvings");
}
}
#endif
