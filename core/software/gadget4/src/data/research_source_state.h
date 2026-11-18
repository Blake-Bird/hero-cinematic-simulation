/* Restart storage for source integrators. Physics producers update these fields
 * in their own blocks; zero initialization does not enable any physical model.
 * No pointers, platform-sized counters or random-generator process globals. */
#ifndef RESEARCH_SOURCE_STATE_H
#define RESEARCH_SOURCE_STATE_H
#include "gadgetconfig.h"
#include <stdint.h>
#include <type_traits>
struct research_source_state
{
  double sidm_last_update_time;
  double sidm_optical_depth;
  uint64_t sidm_collision_count;
  uint64_t sidm_random_counter;
  double sidm_kernel_coarse_support_fraction;
  double sidm_omitted_scattering_upper_bound;
  uint64_t sidm_coarse_guard_version;
  double stellar_live_mass, stellar_remnant_mass;
  uint64_t stellar_population_version;
  double stellar_last_source_time;
  double stellar_returned_mass;
  double stellar_injected_energy;
  double stellar_returned_elements[11];
  double element_mass_fraction[11];
  double stellar_birth_element_fraction[11];
  uint64_t stellar_birth_composition_version;
  uint64_t chemical_state_version;
  uint64_t stellar_model_crc32;
  double source_internal_energy;
  double grackle_species_mass_fraction[6]; /* HI,HII,HeI,HeII,HeIII,e(m_H equivalent); e excluded from baryon mass */
  double chemical_last_update_time, chemical_radiated_energy;
  uint64_t grackle_state_version, grackle_data_crc32;
  uint64_t stellar_event_counter;
  /* SH03 SSP birth state; version zero denotes missing legacy state. */
  double stellar_initial_mass;
  double stellar_initial_locked_mass;
  double stellar_prompt_returned_mass;
  double stellar_birth_time;
  double stellar_birth_scale_factor;
  double stellar_birth_metallicity;
  uint64_t stellar_parent_gas_id;
  uint64_t stellar_random_counter;
  uint64_t stellar_birth_state_version;
  uint64_t stellar_birth_mode;
  uint64_t bh_retired_particle_id, bh_retirement_survivor_id;
  double bh_retirement_native_time;
  uint64_t bh_state_version, bh_companion_id, bh_parent_ids[2];
  uint64_t bh_generation, bh_binary_counter;
  double bh_spin[3], bh_binary_spin2[3], bh_binary_axis[3], bh_binary_radial[3];
  double bh_binary_a_cm, bh_binary_eccentricity, bh_binary_phase;
  double bh_binary_previous_physical_mass_g[2];
  double bh_binary_constraint_energy_erg;
  double bh_binary_angular_reservoir[3], bh_binary_gw_angular_momentum[3];
  double bh_binary_last_time, bh_binary_gw_energy_erg, bh_binary_stellar_energy_erg;
  double bh_last_dynamics_time, bh_friction_work_erg;
  double bh_last_source_time;
  double bh_feedback_reservoir;
  double bh_advected_internal_energy, bh_kinetic_feedback_energy;
  double bh_density_g_cm3, bh_sound_speed_cm_s, bh_relative_speed_cm_s;
  double bh_specific_angular_momentum_cm2_s, bh_rotational_suppression, bh_eddington_ratio;
  uint64_t bh_event_counter;
  /* Seed assembly ledger in physical CGS; version zero is inactive.
   * Orbital capture is separate from intrinsic Kerr spin and binary state. */
  uint64_t bh_seed_reservoir_version;
  double bh_seed_assembled_mass_g, bh_seed_internal_energy_erg, bh_seed_mixing_energy_erg;
  double bh_seed_element_mass_g[11], bh_seed_orbital_angular_momentum_cgs[3];
  /* Molecular chemistry extension; atomic lanes above remain unchanged. */
  double grackle_molecular_mass_fraction[6]; /* HM,H2I,H2II,DI,DII,HDI */
  double grackle_gamma_pressure, grackle_gamma_sound;
  double grackle_temperature_kelvin, grackle_mean_molecular_weight;
  double grackle_inverse_cooling_time_per_s; /* positive net cooling; zero equilibrium; negative net heating */
  /* Calibrated SIDM BH source transaction state; version zero is inactive. */
  uint64_t bh_sidm_drag_version, bh_sidm_drag_counter, bh_sidm_drag_model_crc32;
  uint64_t bh_sidm_drag_model_sha256[4], bh_sidm_drag_signature_sha256[4];
  double bh_sidm_drag_last_native_time, bh_sidm_drag_last_proper_seconds;
  double bh_sidm_drag_heat_erg, bh_sidm_drag_reference_error_cm_s2, bh_sidm_drag_heldout_error_cm_s2;
  double bh_sidm_drag_last_acceleration_cm_s2[3];
  /* Versioned finite-precision source integral; these are native mass units.
   * Pending demand is not accreted matter. Both halves of compensated sums
   * must migrate and restart with the carrier. */
  uint64_t bh_accretion_roundoff_version;
  double bh_accretion_pending_mass, bh_accretion_pending_correction;
  double bh_gravity_mass_correction, bh_physical_mass_correction;
};
static_assert(std::is_trivially_copyable<research_source_state>::value,
              "Native source restart state must contain no pointers or owning objects");
#endif
