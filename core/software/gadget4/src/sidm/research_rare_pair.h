/* First-party rare elastic SIDM pair update and native measurement hook.
 * Physical rates use a supplied, measured kernel overlap in cm^-3. A native
 * neighbor/scheduling driver must enumerate each unordered pair once per
 * interval, provide actual physical dt, and checkpoint attempt/event counters.
 * This primitive does not implement frequent diffusion or a domain exchange.
 * Reference: Robertson et al. SIDM pair algorithms; Fischer et al. 2403.00739.
 */
#ifndef RESEARCH_RARE_SIDM_PAIR_H
#define RESEARCH_RARE_SIDM_PAIR_H
#include "../io/research_sidm_event_log.h"
#include "research_angular_transport.h"
#include "research_folded_transport.h"
#include "research_moller_transport.h"
#include "research_resource_measurements.h"
#include "sidm.h"
#include "sidm_random.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace research_sidm {
using vector3=std::array<double,3>;
struct pair_particle
{
  std::uint64_t id;
  double mass_msun;
  vector3 velocity_kms;
};
struct rare_parameters
{
  std::uint64_t operator_seed;
  double cross_section_per_mass_cm2_g;
  double kernel_overlap_inverse_cm3;
  double physical_dt_seconds;
  double maximum_event_probability;
  double macro_mass_relative_tolerance;
  double conservation_relative_tolerance;
  double energy_absolute_floor;
  double momentum_absolute_floor;
  bool screened_rutherford;
  double screening_velocity_kms;
  double cdf_absolute_tolerance;
  double native_velocity_unit_kms;
  int native_velocity_precision_bits;
  bool folded_identical_particle;
  bool born_moller;
  source_resource_measurements *resources;
  const differential_table *differential_model=nullptr;
  double differential_normalization=1;
  std::uint64_t random_substep=0,random_stage=0;
  double differential_speed_scale=1;
};
inline double pair_differential_speed(const rare_parameters &parameters,double speed)
{
  if(!std::isfinite(parameters.differential_speed_scale) || parameters.differential_speed_scale<=0 || !std::isfinite(speed) || speed<0)
    throw std::runtime_error("SIDM physical differential speed mapping invalid");
  const double physical=speed*parameters.differential_speed_scale;
  if(!std::isfinite(physical))throw std::runtime_error("SIDM mapped differential speed overflow");
  return physical;
}
struct pair_result
{
  bool scattered=false;
  double optical_depth=0,event_probability=0,scattering_cosine=0;
  double relative_speed_before_kms=0,relative_speed_after_kms=0;
  double pair_energy_relative_change=0,pair_momentum_relative_change=0;
};
inline angular_transport_moments pair_angular_transport(const rare_parameters &parameters,double speed,double cutoff)
{
  if(parameters.differential_model)return parameters.differential_model->moments(pair_differential_speed(parameters,speed),cutoff,parameters.folded_identical_particle);
  if(parameters.born_moller) {
    if(!parameters.folded_identical_particle)throw std::runtime_error("Born Moller requires identical-particle folded convention");
    return moller_angular_transport(speed,parameters.screening_velocity_kms,cutoff,parameters.cdf_absolute_tolerance);
  }
  return physical_angular_transport(parameters.folded_identical_particle,parameters.screened_rutherford,speed,
      parameters.screening_velocity_kms,cutoff,parameters.cdf_absolute_tolerance);
}
inline std::uint64_t mix(std::uint64_t value)
{
  value+=UINT64_C(0x9e3779b97f4a7c15);
  value=(value^(value>>30))*UINT64_C(0xbf58476d1ce4e5b9);
  value=(value^(value>>27))*UINT64_C(0x94d049bb133111eb);
  return value^(value>>31);
}
inline double uniform_open(std::uint64_t seed,std::uint64_t left,std::uint64_t right,
                           std::uint64_t integer_interval,std::uint64_t attempt,std::uint64_t lane)
{
  return pair_uniform(seed,left,right,integer_interval,attempt,lane);
}
inline double norm(const vector3 &value)
{return std::hypot(value[0],std::hypot(value[1],value[2]));}
inline double dot(const vector3 &left,const vector3 &right)
{return left[0]*right[0]+left[1]*right[1]+left[2]*right[2];}
inline vector3 cross(const vector3 &left,const vector3 &right)
{return {{left[1]*right[2]-left[2]*right[1],left[2]*right[0]-left[0]*right[2],left[0]*right[1]-left[1]*right[0]}};}
inline void require_positive(double value,const char *name,bool zero=false)
{
  if(!std::isfinite(value) || (zero?value<0:value<=0)) throw std::runtime_error(name);
}
inline double directed_cosine(double quantile,double speed,const rare_parameters &parameters)
{
  if(!parameters.screened_rutherford)return 2*quantile-1;
  const double ratio=speed/parameters.screening_velocity_kms;
  const double a=0.5*ratio*ratio;
  if(!std::isfinite(a) || !std::isfinite(1+2*a*quantile))
    throw std::runtime_error("SIDM angular model outside finite inverse-CDF domain");
  // Analytic inverse of F(mu)=(1+mu)/(2[1+a(1-mu)]).
  return 1-2*(1-quantile)/(1+2*a*quantile);
}
inline pair_result commit_pair_rotation(pair_particle &left,pair_particle &right,const rare_parameters &parameters,
                                        double mu,double phi,double native_time,double scale_factor,
                                        std::uint32_t logging_rank,std::uint64_t &persistent_event_sequence,
                                        research_sidm_event_log *log)
{
  if(left.mass_msun!=right.mass_msun)throw std::runtime_error("SIDM elastic numerical representatives must have exactly equal scattering weights");
  if(!std::isfinite(mu) || mu < -1 || mu > 1 || !std::isfinite(phi))
    throw std::runtime_error("SIDM prescribed pair rotation outside angular support");
  vector3 relative;
  for(int k=0;k<3;k++)relative[k]=left.velocity_kms[k]-right.velocity_kms[k];
  const double speed=norm(relative);
  if(!std::isfinite(speed) || speed<=0)throw std::runtime_error("SIDM prescribed rotation requires finite nonzero relative speed");
  pair_result result;result.relative_speed_before_kms=speed;
  vector3 direction,axis={{0,0,0}};
  for(int k=0;k<3;k++)direction[k]=relative[k]/speed;
  int least=0;for(int k=1;k<3;k++)if(std::abs(direction[k])<std::abs(direction[least]))least=k;
  axis[least]=1;
  vector3 tangent=cross(direction,axis);double length=norm(tangent);
  for(double &value:tangent)value/=length;
  vector3 transverse=cross(direction,tangent),rotated;
  const double sine=std::sqrt(std::max(0.,(1-mu)*(1+mu)));
  for(int k=0;k<3;k++)rotated[k]=mu*direction[k]+sine*(std::cos(phi)*tangent[k]+std::sin(phi)*transverse[k]);
  length=norm(rotated);for(double &value:rotated)value/=length;
  const double total_mass=left.mass_msun+right.mass_msun;
  if(!std::isfinite(total_mass))throw std::runtime_error("SIDM pair mass overflow");
  pair_particle new_left=left,new_right=right;
  vector3 momentum_before,momentum_after;
  double energy_before=0,energy_after=0;
  for(int k=0;k<3;k++) {
    const double center=0.5*left.velocity_kms[k]+0.5*right.velocity_kms[k];
    new_left.velocity_kms[k]=center+0.5*speed*rotated[k];
    new_right.velocity_kms[k]=center-0.5*speed*rotated[k];
    if(parameters.native_velocity_unit_kms>0) {
      if(!std::isfinite(parameters.native_velocity_unit_kms) ||
          (parameters.native_velocity_precision_bits!=32 && parameters.native_velocity_precision_bits!=64))
        throw std::runtime_error("SIDM native velocity storage conversion is invalid");
      const auto retained=[&](double physical) {
        const double native=physical/parameters.native_velocity_unit_kms;
        return parameters.native_velocity_unit_kms*(parameters.native_velocity_precision_bits==32?double(float(native)):native);
      };
      new_left.velocity_kms[k]=retained(new_left.velocity_kms[k]);
      new_right.velocity_kms[k]=retained(new_right.velocity_kms[k]);
    }
    momentum_before[k]=left.mass_msun*left.velocity_kms[k]+right.mass_msun*right.velocity_kms[k];
    momentum_after[k]=left.mass_msun*new_left.velocity_kms[k]+right.mass_msun*new_right.velocity_kms[k];
    energy_before+=0.5*(left.mass_msun*left.velocity_kms[k]*left.velocity_kms[k]+right.mass_msun*right.velocity_kms[k]*right.velocity_kms[k]);
    energy_after+=0.5*(left.mass_msun*new_left.velocity_kms[k]*new_left.velocity_kms[k]+right.mass_msun*new_right.velocity_kms[k]*new_right.velocity_kms[k]);
  }
  vector3 momentum_change,relative_after;
  for(int k=0;k<3;k++) {momentum_change[k]=momentum_after[k]-momentum_before[k];relative_after[k]=new_left.velocity_kms[k]-new_right.velocity_kms[k];}
  result.pair_energy_relative_change=std::abs(energy_after-energy_before)/std::max(std::abs(energy_before),parameters.energy_absolute_floor);
  result.pair_momentum_relative_change=norm(momentum_change)/std::max(norm(momentum_before),parameters.momentum_absolute_floor);
  if(!std::isfinite(result.pair_energy_relative_change) || !std::isfinite(result.pair_momentum_relative_change) ||
      result.pair_energy_relative_change>parameters.conservation_relative_tolerance || result.pair_momentum_relative_change>parameters.conservation_relative_tolerance)
    throw std::runtime_error("SIDM native pair update failed its predeclared conservation bound");
  if(persistent_event_sequence>=((std::uint64_t(1)<<48)-1))throw std::runtime_error("SIDM persistent event identity exhausted");
  research_sidm_event_record record;
  record.event_id=research_sidm_event_record::global_id(logging_rank,persistent_event_sequence);
  record.particle_ids={{left.id,right.id}};record.mass_msun={{left.mass_msun,right.mass_msun}};
  record.native_time=native_time;record.scale_factor=scale_factor;
  for(int k=0;k<3;k++) {
    record.velocity_before_kms[k]=left.velocity_kms[k];record.velocity_before_kms[k+3]=right.velocity_kms[k];
    record.velocity_after_kms[k]=new_left.velocity_kms[k];record.velocity_after_kms[k+3]=new_right.velocity_kms[k];
  }
  // A logging failure is fatal before the caller can continue evolution or seal
  // a generation. The native driver must not catch it and reuse partial state.
  if(log)log->record_applied_update(record);
  left=new_left;right=new_right;++persistent_event_sequence;
  result.scattered=true;result.scattering_cosine=dot(relative,relative_after)/(speed*norm(relative_after));
  result.relative_speed_after_kms=norm(relative_after);
  return result;
}
inline pair_result apply_rare_pair(pair_particle &left,pair_particle &right,
                                  const rare_parameters &parameters,
                                  std::uint64_t integer_interval,std::uint64_t &persistent_attempt,
                                  double native_time,double scale_factor,
                                  std::uint32_t logging_rank,std::uint64_t &persistent_event_sequence,
                                  research_sidm_event_log *log,double minimum_scattering_angle=0)
{
  source_cpu_scope resource_scope(parameters.resources,1);
  if(left.id>=right.id)throw std::runtime_error("SIDM driver must supply one canonical unordered pair");
  require_positive(left.mass_msun,"SIDM left macro mass invalid");
  require_positive(right.mass_msun,"SIDM right macro mass invalid");
  require_positive(parameters.cross_section_per_mass_cm2_g,"SIDM cross section invalid",true);
  require_positive(parameters.differential_normalization,"SIDM differential normalization invalid",true);
  require_positive(parameters.differential_speed_scale,"SIDM differential speed mapping invalid");
  require_positive(parameters.kernel_overlap_inverse_cm3,"SIDM actual kernel overlap invalid",true);
  require_positive(parameters.physical_dt_seconds,"SIDM actual physical interval invalid");
  require_positive(parameters.maximum_event_probability,"SIDM probability limit invalid");
  if(parameters.maximum_event_probability>=1)throw std::runtime_error("SIDM probability bound must be smaller than one");
  require_positive(parameters.macro_mass_relative_tolerance,"SIDM macro mass comparison tolerance invalid",true);
  require_positive(parameters.conservation_relative_tolerance,"SIDM conservation bound invalid");
  require_positive(parameters.cdf_absolute_tolerance,"SIDM inverse-CDF error bound invalid");
  require_positive(parameters.energy_absolute_floor,"SIDM energy floor invalid");
  require_positive(parameters.momentum_absolute_floor,"SIDM momentum floor invalid");
  if(parameters.screened_rutherford)require_positive(parameters.screening_velocity_kms,"SIDM screening velocity invalid");
  if(!std::isfinite(native_time) || !std::isfinite(scale_factor) || scale_factor<=0)
    throw std::runtime_error("SIDM measured native clock invalid");
  if(left.mass_msun!=right.mass_msun)
    throw std::runtime_error("Unequal-resolution SIDM macro pair requires a calibrated weighted operator; coarse support must be guarded");
  vector3 relative;
  for(int axis=0;axis<3;axis++) {
    if(!std::isfinite(left.velocity_kms[axis]) || !std::isfinite(right.velocity_kms[axis]))
      throw std::runtime_error("SIDM measured pair velocity invalid");
    relative[axis]=left.velocity_kms[axis]-right.velocity_kms[axis];
  }
  pair_result result;
  const double speed=norm(relative);
  result.relative_speed_before_kms=speed;
  if(persistent_attempt==std::numeric_limits<std::uint64_t>::max())
    throw std::runtime_error("SIDM persistent pair attempt counter exhausted");
  // A zero relative speed or disabled normalization has exactly zero rate.
  // Do not query an arbitrary table outside its positive velocity coverage
  // for an interaction that cannot occur.
  const bool disabled=parameters.differential_model?parameters.differential_normalization==0:
      parameters.cross_section_per_mass_cm2_g==0;
  if(speed==0 || disabled || parameters.kernel_overlap_inverse_cm3==0) {
    result.relative_speed_after_kms=speed;++persistent_attempt;return result;
  }
  // Equal-weight macro-particle hazard: (sigma/m) m_macro v_relative g_ij dt.
  // The cross-section amplitude must be evaluated at this actual relative speed.
  const double sigma=parameters.differential_model?parameters.differential_normalization*parameters.differential_model->total_cross_section(pair_differential_speed(parameters,speed)):parameters.cross_section_per_mass_cm2_g;
  result.optical_depth=sigma*left.mass_msun*1.98847e33*
      speed*1e5*parameters.kernel_overlap_inverse_cm3*parameters.physical_dt_seconds;
  if(!std::isfinite(result.optical_depth))throw std::runtime_error("SIDM rate overflow");
  if(minimum_scattering_angle!=0 || parameters.folded_identical_particle) {
    const auto split=pair_angular_transport(parameters,speed,minimum_scattering_angle);
    result.optical_depth*=split.large_probability;
  }
  result.event_probability=result.optical_depth; // small-step Bernoulli P=Gamma*dt
  if(result.event_probability>parameters.maximum_event_probability)
    throw std::runtime_error("SIDM native interval exceeds the declared probability limit; refine/subcycle before committing");
  if(persistent_attempt==std::numeric_limits<std::uint64_t>::max())throw std::runtime_error("SIDM persistent pair attempt counter exhausted");
  const std::uint64_t attempt=persistent_attempt;
  if(speed==0 || pair_uniform(parameters.operator_seed,left.id,right.id,integer_interval,parameters.random_substep,0,parameters.random_stage)>=result.event_probability) {
    ++persistent_attempt;return result;
  }
  const double quantile=pair_uniform(parameters.operator_seed,left.id,right.id,integer_interval,parameters.random_substep,1,parameters.random_stage);
  const double mu=parameters.differential_model?parameters.differential_model->sample_cosine(pair_differential_speed(parameters,speed),quantile,minimum_scattering_angle,parameters.folded_identical_particle):parameters.born_moller?
      moller_large_angle_cosine(quantile,speed,parameters.screening_velocity_kms,minimum_scattering_angle,parameters.cdf_absolute_tolerance):parameters.folded_identical_particle?
      folded_large_angle_cosine(quantile,speed,parameters.screened_rutherford,parameters.screening_velocity_kms,
          minimum_scattering_angle,parameters.cdf_absolute_tolerance):minimum_scattering_angle==0?directed_cosine(quantile,speed,parameters):
      directed_large_angle_cosine(quantile,speed,parameters.screened_rutherford,parameters.screening_velocity_kms,
          minimum_scattering_angle,parameters.cdf_absolute_tolerance);
  const double angular_a=parameters.screened_rutherford?0.5*std::pow(speed/parameters.screening_velocity_kms,2):0.;
  const double recovered=(1+mu)/(2*(1+angular_a*(1-mu)));
  if(!parameters.differential_model && minimum_scattering_angle==0 && !parameters.folded_identical_particle &&
      (!std::isfinite(recovered) || std::abs(recovered-quantile)>parameters.cdf_absolute_tolerance))
    throw std::runtime_error("SIDM native inverse-CDF precision exceeds its frozen error bound");
  const double phi=6.283185307179586476925286766559*pair_uniform(parameters.operator_seed,left.id,right.id,integer_interval,parameters.random_substep,2,parameters.random_stage);
  auto committed=commit_pair_rotation(left,right,parameters,mu,phi,native_time,scale_factor,
      logging_rank,persistent_event_sequence,log);
  committed.optical_depth=result.optical_depth;committed.event_probability=result.event_probability;
  ++persistent_attempt;
  return committed;
}

} // namespace research_sidm
#endif
