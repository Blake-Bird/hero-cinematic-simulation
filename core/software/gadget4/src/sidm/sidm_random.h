/* Decomposition-independent counter-based pair draws.
 * Random123 Philox4x64-10, D. E. Shaw Research BSD license retained in
 * third_party/random123/LICENSE. No rank or event-log sequence enters a key.
 */
#ifndef RESEARCH_SIDM_RANDOM_H
#define RESEARCH_SIDM_RANDOM_H
#include "../../third_party/random123/include/Random123/philox.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace research_sidm {
enum class draw_purpose : std::uint64_t {
  accept=0, polar_angle=1, azimuth=2, diffusion_azimuth=8,
  pair_priority=16, control_population=17,
  bh_drag_heat_x=32,bh_drag_heat_y=33,bh_drag_heat_z=34
};
inline std::uint64_t pair_random_bits(std::uint64_t seed,std::uint64_t left,std::uint64_t right,
                                     std::uint64_t interval,std::uint64_t substep,std::uint64_t purpose,
                                     std::uint64_t stage=0)
{
  if(left==right)throw std::runtime_error("SIDM counter RNG requires distinct particle identities");
  if(left>right)std::swap(left,right);
  // The whole canonical uint64 pair, operator interval and substep occupy the
  // counter without lossy pair hashing. Purpose/stage domain-separate keys.
  r123::Philox4x64 generator;
  r123::Philox4x64::ctr_type counter={{left,right,interval,substep}};
  r123::Philox4x64::key_type key={{seed,purpose | (stage<<32)}};
  if(purpose>UINT32_MAX || stage>UINT32_MAX)
    throw std::runtime_error("SIDM RNG draw domain exceeds exact key lanes");
  return generator(counter,key)[0];
}
inline double pair_uniform(std::uint64_t seed,std::uint64_t left,std::uint64_t right,
                           std::uint64_t interval,std::uint64_t substep,std::uint64_t purpose,
                           std::uint64_t stage=0)
{
  // Exactly representable half-bin yields 0<u<1, including extreme outputs.
  return (double(pair_random_bits(seed,left,right,interval,substep,purpose,stage)>>12)+.5)/4503599627370496.;
}
}
#endif
