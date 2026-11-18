/* Research2: non-mutating seed assembly planner. All masses, proper offsets,
 * peculiar velocities and specific internal energies must share declared
 * physical units. Caller must commit every mass/energy/element/angular ledger
 * atomically before enabling a reservoir seeding mode. No gas is deleted here.
 */
#ifndef RESEARCH_BH_SEED_RESERVOIR_H
#define RESEARCH_BH_SEED_RESERVOIR_H
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

namespace research_bh_seed {
using vector3=std::array<double,3>;
using elements=std::array<double,11>;
struct gas {
  std::uint64_t id;
  double mass,specific_internal_energy;
  vector3 offset,velocity;
  elements mass_fractions;
};
struct transfer {
  std::uint64_t donor_id;
  double mass_before,mass_after,removed_mass;
};
struct plan {
  bool supported=false;
  double requested_seed_mass=0,gravity_mass=0,mass_shortfall=0;
  vector3 velocity{},captured_orbital_angular_momentum{};
  elements captured_element_mass{};
  double captured_internal_energy=0,mixing_energy=0;
  std::vector<transfer> transfers;
};
inline double dot(vector3 a,vector3 b) {
  double value=0;for(int k=0;k<3;k++)value+=a[k]*b[k];return value;
}
inline void validate(const gas &p) {
  if(!std::isfinite(p.mass) || p.mass<=0 || !std::isfinite(p.specific_internal_energy) || p.specific_internal_energy<0)
    throw std::invalid_argument("Seed reservoir requires positive gas mass and nonnegative finite specific internal energy");
  for(int k=0;k<3;k++)if(!std::isfinite(p.offset[k]) || !std::isfinite(p.velocity[k]))
    throw std::invalid_argument("Seed reservoir requires finite proper offsets and peculiar velocities");
  long double total=0;
  for(double x:p.mass_fractions) {
    if(!std::isfinite(x) || x<0 || x>1)throw std::invalid_argument("Seed donor element mass fraction outside physical support");
    total+=x;
  }
  // Unrepresented mass may remain; represented fractions must not be renormalized.
  if(total>1)throw std::invalid_argument("Seed donor represented elements exceed total mass");
}
inline plan prepare(const gas &carrier,std::vector<gas> donors,double seed_mass,
                    double maximum_donor_fraction,std::size_t maximum_donors) {
  validate(carrier);
  if(!std::isfinite(seed_mass) || seed_mass<=0 || !std::isfinite(maximum_donor_fraction) ||
      maximum_donor_fraction<=0 || maximum_donor_fraction>.1 || maximum_donors==0 || donors.size()>maximum_donors)
    throw std::invalid_argument("Explicit supported seed mass, donor-fraction bound and donor allocation required");
  if(dot(carrier.offset,carrier.offset)!=0)throw std::invalid_argument("Seed carrier must define the local proper-offset origin");
  std::set<std::uint64_t> ids{carrier.id};
  for(const auto &p:donors) {
    validate(p);
    if(!ids.insert(p.id).second)throw std::invalid_argument("Seed reservoir exact identities duplicated");
  }
  // Deterministic nearest-first allocation; caller supplies a physically
  // selected aperture. This rule is explicit, not a density/kernel inference.
  std::sort(donors.begin(),donors.end(),[](const gas&a,const gas&b){
    double x=dot(a.offset,a.offset),y=dot(b.offset,b.offset);
    return x<y || (x==y && a.id<b.id);
  });
  plan output;output.requested_seed_mass=seed_mass;
  long double mass=carrier.mass,thermal=carrier.mass*carrier.specific_internal_energy;
  std::array<long double,3> momentum{},angular{};
  std::array<long double,11> chemical{};
  long double kinetic=.5L*carrier.mass*dot(carrier.velocity,carrier.velocity);
  for(int k=0;k<3;k++)momentum[k]=carrier.mass*carrier.velocity[k];
  for(int k=0;k<11;k++)chemical[k]=carrier.mass*carrier.mass_fractions[k];
  for(const auto &p:donors) {
    if(mass>=seed_mass)break;
    long double need=static_cast<long double>(seed_mass)-mass;
    double proposed=std::min(double(need),maximum_donor_fraction*p.mass);
    double after=p.mass-proposed,actual=p.mass-after;
    // Round toward retaining gas if subtraction would remove more than the
    // explicit donor cap. This preserves the cap, not a widened tolerance.
    if(actual>maximum_donor_fraction*p.mass) {
      after=std::nextafter(after,p.mass);actual=p.mass-after;
    }
    if(!(after>0) || actual<0 || actual>maximum_donor_fraction*p.mass)
      throw std::runtime_error("Rounded seed donor transfer exceeds declared bound");
    if(actual==0)continue;
    output.transfers.push_back({p.id,p.mass,after,actual});mass+=actual;
    thermal+=actual*p.specific_internal_energy;kinetic+=.5L*actual*dot(p.velocity,p.velocity);
    for(int k=0;k<3;k++)momentum[k]+=actual*p.velocity[k];
    for(int k=0;k<11;k++)chemical[k]+=actual*p.mass_fractions[k];
    angular[0]+=actual*(p.offset[1]*p.velocity[2]-p.offset[2]*p.velocity[1]);
    angular[1]+=actual*(p.offset[2]*p.velocity[0]-p.offset[0]*p.velocity[2]);
    angular[2]+=actual*(p.offset[0]*p.velocity[1]-p.offset[1]*p.velocity[0]);
  }
  output.gravity_mass=double(mass);output.mass_shortfall=double(std::max(0.L,static_cast<long double>(seed_mass)-mass));
  output.supported=mass>=seed_mass;
  long double sink_kinetic=0;
  for(int k=0;k<3;k++) {
    output.velocity[k]=double(momentum[k]/mass);
    output.captured_orbital_angular_momentum[k]=double(angular[k]);
    sink_kinetic+=.5L*mass*output.velocity[k]*output.velocity[k];
  }
  output.captured_internal_energy=double(thermal);
  output.mixing_energy=double(kinetic-sink_kinetic);
  for(int k=0;k<11;k++)output.captured_element_mass[k]=double(chemical[k]);
  // Tiny representational negatives are retained, never silently clamped.
  if(!std::isfinite(output.gravity_mass) || !std::isfinite(output.captured_internal_energy) || !std::isfinite(output.mixing_energy))
    throw std::runtime_error("Seed reservoir extensive accounting overflow");
  return output;
}
}
#endif
