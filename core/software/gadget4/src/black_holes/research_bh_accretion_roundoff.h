/* Finite-precision primitives for the native accretion transaction.
 * A donor plan never removes more than its requested/fraction-limited budget.
 * Unrepresented demand belongs to the caller's persistent source integral;
 * it must not be counted as accreted mass, radiation or feedback.
 */
#ifndef RESEARCH_BH_ACCRETION_ROUNDOFF_H
#define RESEARCH_BH_ACCRETION_ROUNDOFF_H
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace research_bh_accretion {
template<class Mass> struct donor_plan {
  Mass after;
  double removed;
  double unapplied;
};

template<class Mass>
donor_plan<Mass> plan_donor(Mass before, double requested, double maximum_fraction) {
  static_assert(std::is_same<Mass,float>::value || std::is_same<Mass,double>::value,
                "Native accretion plans require the actual particle mass type");
  const double original=double(before);
  if(!(std::isfinite(original) && original>0 && std::isfinite(requested) && requested>=0 &&
       std::isfinite(maximum_fraction) && maximum_fraction>0 && maximum_fraction<=.1))
    throw std::domain_error("Invalid finite native accretion donor budget");
  const double budget=std::min(requested,maximum_fraction*original);
  Mass after=Mass(original-budget);
  double removed=original-double(after);
  // Nearest rounding can otherwise remove one ULP when the requested budget
  // is smaller than that ULP. Move back toward the original mass so the
  // transaction only spends represented, available demand.
  if(removed>budget) {
    after=std::nextafter(after,before);
    removed=original-double(after);
  }
  if(!(double(after)>0 && after<=before && removed>=0 && removed<=budget))
    throw std::runtime_error("Native accretion donor quantization did not respect its budget");
  return {after,removed,requested-removed};
}

struct compensated_value {
  double value;
  double correction;
};

inline compensated_value add_increment(compensated_value before,double increment) {
  if(!(std::isfinite(before.value) && std::isfinite(before.correction) && std::isfinite(increment)))
    throw std::domain_error("Nonfinite native accretion compensated increment");
  // Keep each IEEE operation explicit even in an optimized native build.
  volatile double effective=increment-before.correction;
  volatile double after=before.value+effective;
  volatile double change=after-before.value;
  volatile double correction=change-effective;
  if(!(std::isfinite(after) && std::isfinite(correction)))
    throw std::overflow_error("Native accretion compensated increment overflow");
  return {after,correction};
}

template<class State> void merge_state(State &survivor,const State &other) {
  if(survivor.bh_accretion_roundoff_version!=1 || other.bh_accretion_roundoff_version!=1)
    throw std::domain_error("BH merger lacks versioned accretion integral state");
  auto pending=add_increment({survivor.bh_accretion_pending_mass,survivor.bh_accretion_pending_correction},other.bh_accretion_pending_mass);
  pending=add_increment(pending,-other.bh_accretion_pending_correction);
  const double gravity=survivor.bh_gravity_mass_correction+other.bh_gravity_mass_correction;
  const double physical=survivor.bh_physical_mass_correction+other.bh_physical_mass_correction;
  if(!(std::isfinite(gravity) && std::isfinite(physical) && pending.value-pending.correction>=0))
    throw std::domain_error("BH merger accretion integral overflow/negative pending demand");
  survivor.bh_accretion_pending_mass=pending.value;survivor.bh_accretion_pending_correction=pending.correction;
  survivor.bh_gravity_mass_correction=gravity;survivor.bh_physical_mass_correction=physical;
}
}
#endif
