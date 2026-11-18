/* Current native distributed kinetic/rate bounds for synchronized sources.
 * No replicated global particle census is needed. Candidate workers send rate
 * contributions to native endpoint owners, then agree on one physical dt.
 */
#ifndef RESEARCH_SIDM_DISTRIBUTED_RATES_H
#define RESEARCH_SIDM_DISTRIBUTED_RATES_H
#include "research_pair_exchange.h"

namespace research_sidm {
struct distributed_rate_plan
{
  double physical_dt_seconds=0,maximum_particle_rate=0,maximum_pair_rate=0;
  std::uint64_t global_pairs=0;
  pair_exchange_measurements exchange;
};
inline distributed_rate_plan plan_distributed_rates(MPI_Comm communicator,const std::vector<pair_particle> &local,
                                                   const std::vector<distributed_pair> &pairs,
                                                   double remaining_seconds,double normalization,int angular_kind,double screening_velocity,
                                                   double pair_probability,double particle_probability,double conservation_tolerance,
                                                   double energy_floor,unsigned maximum_pair_rotations,
                                                   const pair_exchange_limits &limits)
{
  int ranks=0;MPI_Comm_size(communicator,&ranks);
  bool valid=std::isfinite(remaining_seconds) && remaining_seconds>0 && std::isfinite(normalization) && normalization>=0 &&
      std::isfinite(pair_probability) && pair_probability>0 && pair_probability<1 && std::isfinite(particle_probability) &&
      particle_probability>0 && particle_probability<1 && std::isfinite(conservation_tolerance) && conservation_tolerance>0 &&
      std::isfinite(energy_floor) && energy_floor>0 && maximum_pair_rotations>=1 && maximum_pair_rotations<=3 &&
      (angular_kind==0 || angular_kind==3 || (std::isfinite(screening_velocity) && screening_velocity>0)) &&
      pairs.size()<=UINT64_MAX/std::uint64_t(ranks);
  long double kinetic=0;
  std::set<std::uint64_t> owned;
  for(const auto &particle:local) {
    valid=valid && owned.insert(particle.id).second && std::isfinite(particle.mass_msun) && particle.mass_msun>0;
    for(double velocity:particle.velocity_kms) {
      valid=valid && std::isfinite(velocity);
      kinetic+=(.5L*particle.mass_msun)*velocity*velocity;
    }
  }
  valid=valid && std::isfinite(kinetic);
  require_collective(communicator,valid,"SIDM distributed rate plan changes native state, probability or physical domain");
  long double total_kinetic=0;
  if(MPI_Allreduce(&kinetic,&total_kinetic,1,MPI_LONG_DOUBLE,MPI_SUM,communicator)!=MPI_SUCCESS)
    throw std::runtime_error("SIDM native kinetic-energy rate reduction failed");
  distributed_rate_plan result;
  const std::uint64_t count=pairs.size();
  if(MPI_Allreduce(&count,&result.global_pairs,1,MPI_UINT64_T,MPI_SUM,communicator)!=MPI_SUCCESS)
    throw std::runtime_error("SIDM distributed pair census reduction failed");
  const long double growth=std::exp(static_cast<long double>(result.global_pairs)*maximum_pair_rotations*
      std::log1p(static_cast<long double>(conservation_tolerance)));
  const long double upper_energy=(total_kinetic+energy_floor)*growth;
  require_collective(communicator,std::isfinite(upper_energy) && upper_energy>=0,
      "SIDM distributed conservative rate bound exceeds finite physical domain");
  struct contribution {std::uint64_t id;long double rate;};
  std::vector<std::pair<int,contribution>> outgoing;
  require_collective(communicator,pairs.size()<=limits.maximum_exchange_records/2,
      "SIDM distributed endpoint rate contributions exceed the exchange allocation");
  outgoing.reserve(2*pairs.size());
  long double largest_pair=0;
  for(const auto &pair:pairs) {
    require_positive(pair.left_mass_msun,"SIDM distributed left physical mass invalid");
    require_positive(pair.right_mass_msun,"SIDM distributed right physical mass invalid");
    const long double speed=std::sqrt(2*upper_energy*(1/pair.left_mass_msun+1/pair.right_mass_msun));
    const long double sigma_v=native_sigma_speed_bound(angular_kind,normalization,speed,screening_velocity);
    const long double rate=sigma_v*1.e5L*pair.left_mass_msun*1.98847e33L*pair.kernel_overlap_inverse_cm3;
    if(!std::isfinite(rate) || rate<0)throw std::runtime_error("SIDM distributed candidate rate overflow");
    largest_pair=std::max(largest_pair,rate);
    outgoing.push_back({pair.left_owner,{pair.left_id,rate}});
    outgoing.push_back({pair.right_owner,{pair.right_id,rate}});
  }
  const auto contributions=exchange_pair_records(communicator,outgoing,limits,result.exchange);
  std::map<std::uint64_t,long double> particle_rates;
  valid=true;
  for(const auto &row:contributions.values) {
    valid=valid && owned.count(row.id) && std::isfinite(row.rate) && row.rate>=0;
    particle_rates[row.id]+=row.rate;
  }
  long double largest_particle=0;
  for(const auto &row:particle_rates) {valid=valid && std::isfinite(row.second);largest_particle=std::max(largest_particle,row.second);}
  require_collective(communicator,valid,"SIDM distributed rate contribution has absent owner or nonfinite total hazard");
  long double global_pair=0,global_particle=0;
  MPI_Allreduce(&largest_pair,&global_pair,1,MPI_LONG_DOUBLE,MPI_MAX,communicator);
  MPI_Allreduce(&largest_particle,&global_particle,1,MPI_LONG_DOUBLE,MPI_MAX,communicator);
  long double dt=remaining_seconds;
  const long double margin=1-128*std::numeric_limits<double>::epsilon();
  if(global_pair>0)dt=std::min(dt,margin*static_cast<long double>(pair_probability)/global_pair);
  if(global_particle>0)dt=std::min(dt,margin*(-std::log1p(-static_cast<long double>(particle_probability)))/global_particle);
  result.physical_dt_seconds=double(dt);result.maximum_particle_rate=double(global_particle);result.maximum_pair_rate=double(global_pair);
  require_collective(communicator,std::isfinite(result.physical_dt_seconds) && result.physical_dt_seconds>0 &&
      std::isfinite(result.maximum_particle_rate) && std::isfinite(result.maximum_pair_rate),
      "SIDM distributed source dt/rate exceeds finite recorded physical precision");
  return result;
}
}
#endif
