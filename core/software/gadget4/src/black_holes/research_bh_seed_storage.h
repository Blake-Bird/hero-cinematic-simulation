/* Extensive, physical-CGS seed assembly history. These capture ledgers are
 * separate from Kerr spin, radiative losses and resolved binary reservoirs. */
#ifndef RESEARCH_BH_SEED_STORAGE_H
#define RESEARCH_BH_SEED_STORAGE_H
#include "../data/research_source_state.h"
#include "research_bh_seed_reservoir.h"
#include <cmath>
#include <stdexcept>
namespace research_bh_seed {
inline void validate_storage(const research_source_state &s) {
  if(s.bh_seed_reservoir_version>1)throw std::invalid_argument("Unsupported seed ledger version");
  const double scalar[]={s.bh_seed_assembled_mass_g,s.bh_seed_internal_energy_erg,s.bh_seed_mixing_energy_erg};
  for(double x:scalar)if(!std::isfinite(x) || (!s.bh_seed_reservoir_version && x!=0))
    throw std::invalid_argument("Invalid/unversioned seed ledger scalar");
  if(s.bh_seed_assembled_mass_g<0 || s.bh_seed_internal_energy_erg<0 ||
     (s.bh_seed_reservoir_version && s.bh_seed_assembled_mass_g<=0))
    throw std::invalid_argument("Invalid extensive seed ledger mass/energy");
  long double elements=0;
  for(double x:s.bh_seed_element_mass_g) {
    if(!std::isfinite(x) || x<0 || (!s.bh_seed_reservoir_version && x!=0))
      throw std::invalid_argument("Invalid/unversioned seed element ledger");
    elements+=x;
  }
  if(elements>s.bh_seed_assembled_mass_g)throw std::invalid_argument("Seed ledger elements exceed assembly mass");
  for(double x:s.bh_seed_orbital_angular_momentum_cgs)
    if(!std::isfinite(x) || (!s.bh_seed_reservoir_version && x!=0))
      throw std::invalid_argument("Invalid/unversioned seed angular ledger");
}
inline void store_assembly(research_source_state &state,const plan &assembly,
    double physical_mass_unit_g,double proper_length_unit_cm,double peculiar_velocity_unit_cm_s) {
  validate_storage(state);
  if(state.bh_seed_reservoir_version || !assembly.supported ||
      !(std::isfinite(physical_mass_unit_g) && physical_mass_unit_g>0 &&
        std::isfinite(proper_length_unit_cm) && proper_length_unit_cm>0 &&
        std::isfinite(peculiar_velocity_unit_cm_s) && peculiar_velocity_unit_cm_s>0))
    throw std::invalid_argument("Seed assembly requires supported plan, unused ledger and explicit physical units");
  research_source_state next=state;
  const double energy_unit=physical_mass_unit_g*peculiar_velocity_unit_cm_s*peculiar_velocity_unit_cm_s;
  const double angular_unit=physical_mass_unit_g*proper_length_unit_cm*peculiar_velocity_unit_cm_s;
  next.bh_seed_reservoir_version=1;
  next.bh_seed_assembled_mass_g=assembly.gravity_mass*physical_mass_unit_g;
  next.bh_seed_internal_energy_erg=assembly.captured_internal_energy*energy_unit;
  next.bh_seed_mixing_energy_erg=assembly.mixing_energy*energy_unit;
  for(int k=0;k<11;k++)next.bh_seed_element_mass_g[k]=assembly.captured_element_mass[k]*physical_mass_unit_g;
  for(int k=0;k<3;k++)next.bh_seed_orbital_angular_momentum_cgs[k]=assembly.captured_orbital_angular_momentum[k]*angular_unit;
  validate_storage(next);state=next;
}
inline void merge_storage(research_source_state &survivor,const research_source_state &retired) {
  validate_storage(survivor);validate_storage(retired);
  // Prepare and validate before touching the survivor. The rest of its native
  // source state remains byte-for-byte unchanged by this operation.
  research_source_state next=survivor;
  next.bh_seed_reservoir_version=survivor.bh_seed_reservoir_version || retired.bh_seed_reservoir_version;
  next.bh_seed_assembled_mass_g+=retired.bh_seed_assembled_mass_g;
  next.bh_seed_internal_energy_erg+=retired.bh_seed_internal_energy_erg;
  next.bh_seed_mixing_energy_erg+=retired.bh_seed_mixing_energy_erg;
  for(int k=0;k<11;k++)next.bh_seed_element_mass_g[k]+=retired.bh_seed_element_mass_g[k];
  for(int k=0;k<3;k++)next.bh_seed_orbital_angular_momentum_cgs[k]+=retired.bh_seed_orbital_angular_momentum_cgs[k];
  validate_storage(next);survivor=next;
}
}
#endif
