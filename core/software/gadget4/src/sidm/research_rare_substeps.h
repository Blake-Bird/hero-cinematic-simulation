/* Bounded rare-source substeps with a per-particle hazard limit.
 * Step selection uses pre-draw rate upper bounds. It does not reject random
 * outcomes, clip event probabilities, or choose dt from accepted collisions.
 * Spatial kernels are frozen by the caller over this source kick; gravity and
 * drift splitting errors still require independent integration controls.
 */
#ifndef RESEARCH_RARE_SIDM_SUBSTEPS_H
#define RESEARCH_RARE_SIDM_SUBSTEPS_H
#include "research_rare_pair.h"
#include <vector>

namespace research_sidm {
struct rare_substep_plan
{
  // Historical member name retained for physical source drivers. A mapped
  // caller receives canonical seconds and uses the frame-neutral accessor.
  double physical_dt_seconds=0;
  double maximum_particle_rate_bound=0;
  double maximum_pair_rate_bound=0;
  std::uint64_t inspected_pairs=0;
  double source_dt_seconds()const{return physical_dt_seconds;}
};
template<class Container,class Overlap,class SigmaSpeedBound>
rare_substep_plan plan_rare_substep_with_rate_bound(const Container &particles,Overlap overlap,
                                  double remaining_seconds,double cross_section_normalization,
                                  int angular_kind,double screening_velocity,
                                  double pair_probability_limit,double particle_probability_limit,
                                  double conservation_tolerance,double energy_absolute_floor,
                                  std::uint64_t remaining_pair_visits,unsigned maximum_pair_rotations,
                                  SigmaSpeedBound sigma_speed_bound)
{
  require_positive(remaining_seconds,"SIDM source interval invalid");
  require_positive(cross_section_normalization,"SIDM source cross section invalid",true);
  require_positive(pair_probability_limit,"SIDM pair probability limit invalid");
  require_positive(particle_probability_limit,"SIDM particle probability limit invalid");
  require_positive(conservation_tolerance,"SIDM conservation tolerance invalid");
  require_positive(energy_absolute_floor,"SIDM source energy floor invalid");
  if(angular_kind<0 || angular_kind>3)
    throw std::runtime_error("SIDM source rate planner angular model unknown");
  if(maximum_pair_rotations<1 || maximum_pair_rotations>3)
    throw std::runtime_error("SIDM source rotation count outside the declared rare/frequent/hybrid schedule");
  if(pair_probability_limit>=1 || particle_probability_limit>=1)
    throw std::runtime_error("SIDM source probability bounds must lie below one");
  if(angular_kind==1 || angular_kind==2)require_positive(screening_velocity,"SIDM source screening scale invalid");
  const std::uint64_t n=particles.size();
  if(n && n-1>std::numeric_limits<std::uint64_t>::max()/n)
    throw std::runtime_error("SIDM source pair count overflow");
  const std::uint64_t pairs=n*(n? n-1:0)/2;
  // Reserve both rate planning and application passes before any update.
  if(pairs>remaining_pair_visits/2)
    throw std::runtime_error("SIDM source rate/application work exceeds the frozen pair-visit allocation");
  long double kinetic=0;
  for(const auto &row:particles) {
    require_positive(row.value.mass_msun,"SIDM source macro mass invalid");
    for(double velocity:row.value.velocity_kms) {
      if(!std::isfinite(velocity))throw std::runtime_error("SIDM source velocity invalid");
      kinetic+=(.5L*row.value.mass_msun)*velocity*velocity;
    }
  }
  // Each pair is committed at most once during this substep. Include its
  // allowed roundoff energy increase in the global speed upper bound. This
  // avoids choosing the step from velocities that later pairs can change.
  const long double growth=std::exp(static_cast<long double>(pairs)*maximum_pair_rotations*
      std::log1p(static_cast<long double>(conservation_tolerance)));
  const long double energy_upper=(kinetic+energy_absolute_floor)*growth;
  if(!std::isfinite(energy_upper) || !std::isfinite(growth))
    throw std::runtime_error("SIDM source conservative energy/rate bound overflow");
  std::vector<long double> particle_rates(n,0);
  rare_substep_plan result;
  for(std::size_t left=0;left<particles.size();left++)for(std::size_t right=left+1;right<particles.size();right++) {
    ++result.inspected_pairs;
    const double kernel=overlap(left,right);
    require_positive(kernel,"SIDM planned kernel overlap invalid",true);
    if(kernel==0 || cross_section_normalization==0)continue;
    const long double ml=particles[left].value.mass_msun,mr=particles[right].value.mass_msun;
    // Cauchy bound on |v_i-v_j| from total kinetic energy, in km/s.
    const long double speed_upper=std::sqrt(2*energy_upper*(1/ml+1/mr));
    if(!std::isfinite(speed_upper) || speed_upper<0)
      throw std::runtime_error("SIDM planned speed bound outside finite domain");
    // The supplied law must bound sigma(speed)*speed in the SAME velocity
    // frame and clock as these particles, overlaps and remaining_seconds.
    // Reject a negative bound before a zero/tiny kernel could conceal it.
    const long double sigma_v=sigma_speed_bound(speed_upper);
    if(!std::isfinite(sigma_v) || sigma_v<0)
      throw std::runtime_error("SIDM supplied cross-section speed bound invalid");
    // Equal-weight pair primitive uses the canonical left mass. Keep exactly
    // that mass in both participants' accumulated event hazard bounds.
    const long double rate=sigma_v*1.e5L*ml*1.98847e33L*kernel;
    if(!std::isfinite(rate) || rate<0 || rate>std::numeric_limits<double>::max())
      throw std::runtime_error("SIDM planned pair rate outside finite domain");
    particle_rates[left]+=rate;particle_rates[right]+=rate;
    result.maximum_pair_rate_bound=std::max(result.maximum_pair_rate_bound,double(rate));
  }
  long double maximum=0;
  for(long double rate:particle_rates)maximum=std::max(maximum,rate);
  if(!std::isfinite(maximum) || maximum>std::numeric_limits<double>::max())
    throw std::runtime_error("SIDM total particle rate bound overflow");
  result.maximum_particle_rate_bound=double(maximum);
  long double dt=remaining_seconds;
  // Margin avoids accepting a dt rounded just above the strict probability
  // limit. It is not a substitute for the primitive's actual pair check.
  const long double margin=1-64*std::numeric_limits<double>::epsilon();
  if(maximum>0)dt=std::min(dt,margin*(-std::log1p(-static_cast<long double>(particle_probability_limit)))/maximum);
  if(result.maximum_pair_rate_bound>0)dt=std::min(dt,
    margin*static_cast<long double>(pair_probability_limit)/result.maximum_pair_rate_bound);
  result.physical_dt_seconds=double(dt);
  if(!std::isfinite(result.physical_dt_seconds) || result.physical_dt_seconds<=0)
    throw std::runtime_error("SIDM planned substep is below the finite positive clock domain");
  return result;
}
// Preserve the physical-clock entry point for existing reference controls.
// Canonical controls use the same scheduling/conservation implementation,
// supplying their transformed law through the explicit bound entry point.
template<class Container,class Overlap>
rare_substep_plan plan_rare_substep(const Container &particles,Overlap overlap,
                                  double remaining_seconds,double cross_section_normalization,
                                  int angular_kind,double screening_velocity,
                                  double pair_probability_limit,double particle_probability_limit,
                                  double conservation_tolerance,double energy_absolute_floor,
                                  std::uint64_t remaining_pair_visits,unsigned maximum_pair_rotations=1)
{
  return plan_rare_substep_with_rate_bound(particles,overlap,remaining_seconds,cross_section_normalization,
      angular_kind,screening_velocity,pair_probability_limit,particle_probability_limit,
      conservation_tolerance,energy_absolute_floor,remaining_pair_visits,maximum_pair_rotations,
      [&](long double speed_upper) {
        return native_sigma_speed_bound(angular_kind,cross_section_normalization,speed_upper,screening_velocity);
      });
}
}
#endif
