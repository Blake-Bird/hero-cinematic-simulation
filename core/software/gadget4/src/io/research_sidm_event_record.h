/* First-party SIDM diagnostic record. A native physical operator supplies
 * measured pre/post states; this record never generates scattering events.
 * Physical km/s and Msun values remove ambiguous scale-factor velocity rules.
 * Persistent sequence ownership belongs to the native restart state. */
#ifndef RESEARCH_SIDM_EVENT_RECORD_H
#define RESEARCH_SIDM_EVENT_RECORD_H
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

struct research_sidm_event_record
{
  std::uint64_t event_id;
  std::array<std::uint64_t,2> particle_ids;
  std::array<double,6> velocity_before_kms, velocity_after_kms;
  std::array<double,2> mass_msun;
  double native_time, scale_factor;

  static std::uint64_t global_id(std::uint32_t rank,std::uint64_t persistent_sequence)
  {
    if(rank >= (1u<<16) || persistent_sequence >= (std::uint64_t(1)<<48))
      throw std::runtime_error("SIDM diagnostic event identity capacity exhausted");
    return (std::uint64_t(rank)<<48)|persistent_sequence;
  }
  void require_physical() const
  {
    if(particle_ids[0] == particle_ids[1] || !std::isfinite(native_time) ||
       !std::isfinite(scale_factor) || scale_factor <= 0)
      throw std::runtime_error("Invalid SIDM event identity or native clock");
    for(double value:velocity_before_kms)
      if(!std::isfinite(value)) throw std::runtime_error("Nonfinite SIDM event pre-velocity");
    for(double value:velocity_after_kms)
      if(!std::isfinite(value)) throw std::runtime_error("Nonfinite SIDM event post-velocity");
    for(double value:mass_msun)
      if(!std::isfinite(value) || value <= 0) throw std::runtime_error("Invalid SIDM event mass");
  }
};
#endif
