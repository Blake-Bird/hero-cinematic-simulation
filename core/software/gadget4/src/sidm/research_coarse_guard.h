/* Measured SIDM coarse support and omitted-scattering bounds.
 * Fine/coarse pairs supply diagnostics but do not receive an uncalibrated
 * unequal-weight scattering update. Native gravity still includes coarse DM.
 * Contributions are routed to the actual fine particle owner before sources
 * commit. Bounds use actual compact kernel overlap and physical macro masses.
 */
#ifndef RESEARCH_SIDM_COARSE_GUARD_H
#define RESEARCH_SIDM_COARSE_GUARD_H
#include "research_pair_exchange.h"

namespace research_sidm {
struct coarse_support_measurement
{
  double kernel_coarse_support_fraction=0;
  double omitted_scattering_upper_bound=0;
  double coarse_kernel_mass_density_msun_cm3=0,fine_kernel_mass_density_msun_cm3=0;
};
struct coarse_support_products
{
  std::map<std::uint64_t,coarse_support_measurement> particles;
  pair_exchange_measurements exchange;
};
inline coarse_support_products measure_distributed_coarse_support(MPI_Comm communicator,
    const std::vector<pair_particle> &all_local,const std::set<std::uint64_t> &fine_owned,
    const std::vector<distributed_pair> &support_pairs,std::uint64_t local_scattering_pairs,
    unsigned maximum_pair_rotations,double source_seconds,double cross_section_normalization,
    int angular_kind,double screening_velocity,double conservation_tolerance,double energy_floor,
    const pair_exchange_limits &limits)
{
  int ranks=0;MPI_Comm_size(communicator,&ranks);
  bool valid=std::isfinite(source_seconds) && source_seconds>0 && std::isfinite(cross_section_normalization) &&
      cross_section_normalization>=0 && (angular_kind==0 || angular_kind==3 || (std::isfinite(screening_velocity) && screening_velocity>0)) &&
      std::isfinite(conservation_tolerance) && conservation_tolerance>0 && std::isfinite(energy_floor) && energy_floor>0 &&
      maximum_pair_rotations>=1 && maximum_pair_rotations<=3 && local_scattering_pairs<=UINT64_MAX/std::uint64_t(ranks);
  long double kinetic=0;
  for(const auto &particle:all_local) {
    valid=valid && std::isfinite(particle.mass_msun) && particle.mass_msun>0;
    for(double velocity:particle.velocity_kms) {valid=valid && std::isfinite(velocity);kinetic+=(.5L*particle.mass_msun)*velocity*velocity;}
  }
  require_collective(communicator,valid && std::isfinite(kinetic),"SIDM coarse-support physical state or native source bounds invalid");
  long double global_kinetic=0;
  std::uint64_t global_pairs=0;
  MPI_Allreduce(&kinetic,&global_kinetic,1,MPI_LONG_DOUBLE,MPI_SUM,communicator);
  MPI_Allreduce(&local_scattering_pairs,&global_pairs,1,MPI_UINT64_T,MPI_SUM,communicator);
  const long double energy_upper=(global_kinetic+energy_floor)*std::exp(static_cast<long double>(global_pairs)*maximum_pair_rotations*
      std::log1p(static_cast<long double>(conservation_tolerance)));
  require_collective(communicator,std::isfinite(energy_upper) && energy_upper>=0,
      "SIDM omitted-scattering speed bound exceeds finite physical domain");
  struct contribution {std::uint64_t id;long double fine_density,coarse_density,omitted_depth;};
  require_collective(communicator,support_pairs.size()<=limits.maximum_exchange_records/2,
      "SIDM coarse-support contributions exceed the exchange allocation");
  coarse_support_products result;
  std::vector<std::pair<int,contribution>> outgoing;
  outgoing.reserve(2*support_pairs.size());
  for(const auto &pair:support_pairs) {
    if(pair.left_type==1 && pair.right_type==1) {
      outgoing.push_back({pair.left_owner,{pair.left_id,pair.right_mass_msun*pair.kernel_overlap_inverse_cm3,0,0}});
      outgoing.push_back({pair.right_owner,{pair.right_id,pair.left_mass_msun*pair.kernel_overlap_inverse_cm3,0,0}});
      continue;
    }
    if(!((pair.left_type==1 && pair.right_type==2) || (pair.left_type==2 && pair.right_type==1)))
      throw std::runtime_error("SIDM coarse-support pair has an unsupported native species");
    const bool left_fine=pair.left_type==1;
    const double fine_mass=left_fine?pair.left_mass_msun:pair.right_mass_msun;
    const double coarse_mass=left_fine?pair.right_mass_msun:pair.left_mass_msun;
    const long double speed_upper=std::sqrt(2*energy_upper*(1/fine_mass+1/coarse_mass));
    const long double sigma_v=native_sigma_speed_bound(angular_kind,cross_section_normalization,speed_upper,screening_velocity);
    const long double density=static_cast<long double>(coarse_mass)*pair.kernel_overlap_inverse_cm3;
    const long double omitted=density*1.98847e33L*sigma_v*1.e5L*source_seconds;
    if(!std::isfinite(omitted) || omitted<0 || !std::isfinite(density))
      throw std::runtime_error("SIDM omitted coarse-scattering bound overflow");
    outgoing.push_back({left_fine?pair.left_owner:pair.right_owner,{left_fine?pair.left_id:pair.right_id,0,density,omitted}});
  }
  const auto incoming=exchange_pair_records(communicator,outgoing,limits,result.exchange);
  struct totals {long double fine=0,coarse=0,depth=0;};
  std::map<std::uint64_t,totals> accumulated;
  valid=true;
  for(const auto &row:incoming.values) {
    valid=valid && fine_owned.count(row.id) && std::isfinite(row.fine_density) && row.fine_density>=0 &&
        std::isfinite(row.coarse_density) && row.coarse_density>=0 && std::isfinite(row.omitted_depth) && row.omitted_depth>=0;
    auto &target=accumulated[row.id];target.fine+=row.fine_density;target.coarse+=row.coarse_density;target.depth+=row.omitted_depth;
  }
  require_collective(communicator,valid,"SIDM coarse support contribution has absent fine owner or nonphysical values");
  for(std::uint64_t id:fine_owned) {
    const auto &total=accumulated[id];const long double density=total.fine+total.coarse;
    coarse_support_measurement measurement;
    measurement.kernel_coarse_support_fraction=density>0?double(total.coarse/density):0;
    measurement.omitted_scattering_upper_bound=double(total.depth);
    measurement.fine_kernel_mass_density_msun_cm3=double(total.fine);
    measurement.coarse_kernel_mass_density_msun_cm3=double(total.coarse);
    for(double value:{measurement.kernel_coarse_support_fraction,measurement.omitted_scattering_upper_bound,
        measurement.fine_kernel_mass_density_msun_cm3,measurement.coarse_kernel_mass_density_msun_cm3})
      if(!std::isfinite(value) || value<0)throw std::runtime_error("SIDM retained coarse-support measurement overflow");
    result.particles.emplace(id,measurement);
  }
  return result;
}
}
#endif
