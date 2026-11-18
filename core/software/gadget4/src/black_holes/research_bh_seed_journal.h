#ifndef RESEARCH_BH_SEED_JOURNAL_H
#define RESEARCH_BH_SEED_JOURNAL_H
#include "research_bh_seed_reservoir.h"
#include <iomanip>
#include <sstream>
#include <locale>
namespace research_bh_seed {
inline void journal_units(double mass_g,double length_cm,double velocity_cm_s) {
  if(!(std::isfinite(mass_g)&&mass_g>0&&std::isfinite(length_cm)&&length_cm>0&&std::isfinite(velocity_cm_s)&&velocity_cm_s>0))
    throw std::invalid_argument("Seed journal requires explicit positive physical units");
}
template<std::size_t N> inline std::string array_json(const std::array<double,N> &values,double factor=1) {
  std::ostringstream out;out.imbue(std::locale::classic());out<<std::setprecision(17)<<'[';
  for(std::size_t k=0;k<N;k++) {
    const double value=values[k]*factor;if(!std::isfinite(value))throw std::invalid_argument("Nonfinite seed journal vector");
    if(k)out<<',';out<<value;
  }
  out<<']';return out.str();
}
inline std::string donor_json(const gas &before,const transfer &change,std::uint64_t source_id,
    double time,std::uint64_t transaction,double mass_g,double length_cm,double velocity_cm_s) {
  validate(before);journal_units(mass_g,length_cm,velocity_cm_s);
  if(!source_id || !transaction || before.id==source_id || before.id!=change.donor_id || before.mass!=change.mass_before ||
      !(change.mass_after>0 && change.mass_after<before.mass) || before.mass-change.mass_after!=change.removed_mass || !std::isfinite(time))
    throw std::invalid_argument("Inconsistent seed donor journal transfer");
  std::ostringstream out;out.imbue(std::locale::classic());out<<std::setprecision(17);
  out<<"{\"kind\":\"seed_donor\",\"native_time\":"<<time<<",\"particle_id\":"<<before.id<<",\"counter\":1,\"source_id\":"<<source_id<<",\"transaction_index\":"<<transaction
     <<",\"mass_before_native\":"<<before.mass<<",\"mass_after_native\":"<<change.mass_after
     <<",\"velocity_before_native\":"<<array_json(before.velocity)<<",\"velocity_after_native\":"<<array_json(before.velocity)
     <<",\"internal_energy_before_native\":"<<before.specific_internal_energy<<",\"internal_energy_after_native\":"<<before.specific_internal_energy
     <<",\"offset_proper_cm\":"<<array_json(before.offset,length_cm)<<",\"element_mass_before_g\":"<<array_json(before.mass_fractions,before.mass*mass_g)
     <<",\"element_mass_after_g\":"<<array_json(before.mass_fractions,change.mass_after*mass_g)<<'}';
  return out.str();
}
inline std::string assembly_json(const gas &carrier,const plan &assembly,double time,std::uint64_t transaction,
    double donor_fraction,double aperture_kpc,double mass_g,double length_cm,double velocity_cm_s) {
  validate(carrier);journal_units(mass_g,length_cm,velocity_cm_s);
  if(!carrier.id || !transaction || !assembly.supported || !std::isfinite(time) ||
     !(std::isfinite(donor_fraction)&&donor_fraction>0&&donor_fraction<=.1&&std::isfinite(aperture_kpc)&&aperture_kpc>0))
    throw std::invalid_argument("Unsupported seed assembly journal transaction");
  std::ostringstream out;out.imbue(std::locale::classic());out<<std::setprecision(17);
  out<<"{\"kind\":\"seed_assembly\",\"native_time\":"<<time<<",\"particle_id\":"<<carrier.id<<",\"counter\":1,\"transaction_index\":"<<transaction<<",\"previous_species\":0"
     <<",\"mass_before_native\":"<<carrier.mass<<",\"mass_after_native\":"<<assembly.gravity_mass<<",\"physical_mass_native\":"<<assembly.requested_seed_mass
     <<",\"velocity_before_native\":"<<array_json(carrier.velocity)<<",\"velocity_after_native\":"<<array_json(assembly.velocity)
     <<",\"internal_energy_before_native\":"<<carrier.specific_internal_energy<<",\"captured_internal_energy_native\":"<<assembly.captured_internal_energy<<",\"mixing_energy_native\":"<<assembly.mixing_energy
     <<",\"captured_element_mass_g\":"<<array_json(assembly.captured_element_mass,mass_g)
     <<",\"captured_orbital_angular_momentum_g_cm2_s\":"<<array_json(assembly.captured_orbital_angular_momentum,mass_g*length_cm*velocity_cm_s)
     <<",\"carrier_element_mass_before_g\":"<<array_json(carrier.mass_fractions,carrier.mass*mass_g)
     <<",\"donor_count\":"<<assembly.transfers.size()<<",\"maximum_donor_fraction\":"<<donor_fraction<<",\"aperture_proper_kpc\":"<<aperture_kpc<<'}';
  return out.str();
}
}
#endif
