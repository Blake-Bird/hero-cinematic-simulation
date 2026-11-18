/* Exact periodic winding accumulated at native drifts, not reconstructed
 * from sparsely saved positions. Independent of the internal domain shift. */
#ifndef RESEARCH_PERIODIC_IMAGES_H
#define RESEARCH_PERIODIC_IMAGES_H
#ifndef RESEARCH_PERIODIC_IMAGES_STANDALONE
#include "gadgetconfig.h"
#endif
#include <cstdint>
#include <limits>
#include <cmath>
#include <climits>

struct research_periodic_image_state
{
  std::uint64_t version, status;
  double anchor_native_time;
  std::int64_t offsets[3];
};

namespace research_periodic_images
{
enum { unavailable=0, supported=1, counter_overflow=2, invalid_drift=3, ambiguous_relocation=4 };

inline void initialize(research_periodic_image_state &s, double time)
{
  s={}; s.version=1; s.status=std::isfinite(time)?supported:unavailable; s.anchor_native_time=time;
}

inline bool valid(const research_periodic_image_state &s)
{
  return s.version==1 && s.status<=ambiguous_relocation && std::isfinite(s.anchor_native_time);
}

inline bool increment(std::int64_t &value, bool negative, std::uint64_t amount)
{
  const std::uint64_t high=std::uint64_t(1)<<63;
  const std::uint64_t bits=static_cast<std::uint64_t>(value);
  const std::uint64_t capacity=negative ? bits-high : std::uint64_t(INT64_MAX)-bits;
  if(amount>capacity)return false;
  const std::uint64_t result=negative ? bits-amount : bits+amount;
  value=result<high ? static_cast<std::int64_t>(result) : INT64_MIN+static_cast<std::int64_t>(result-high);
  return true;
}

template <typename U> inline U mask(unsigned bits)
{
  return bits==sizeof(U)*CHAR_BIT ? ~U(0) : (U(1)<<bits)-1;
}

template <typename U, typename S>
inline bool drift(research_periodic_image_state &s, int axis, U physical_before, S delta, unsigned bits)
{
  if(s.version!=1 || s.status!=supported)return false;
  if(axis<0 || axis>2 || bits<1 || bits>sizeof(U)*CHAR_BIT){s.status=invalid_drift;return false;}
  const bool negative=delta<0;
  const U magnitude=negative ? U(0)-static_cast<U>(delta) : static_cast<U>(delta);
  const U before=physical_before&mask<U>(bits);
  U crossings=0;
  if(bits==sizeof(U)*CHAR_BIT)
    crossings=negative ? U(before<magnitude) : U(magnitude>U(~U(0)-before));
  else
    {
      const U remainder=magnitude&mask<U>(bits), period=U(1)<<bits;
      crossings=(magnitude>>bits)+(negative ? U(before<remainder) : U(remainder>=period-before));
    }
  if(crossings>static_cast<U>(UINT64_MAX) || !increment(s.offsets[axis],negative,static_cast<std::uint64_t>(crossings)))
    {s.status=counter_overflow;return false;}
  return true;
}

template <typename U, typename S>
inline bool drift_nonperiodic_axis(research_periodic_image_state &s, int axis, U before, S delta, unsigned bits)
{
  if(s.version!=1 || s.status!=supported)return false;
  research_periodic_image_state trial=s;
  if(!drift(trial,axis,before,delta,bits) || trial.offsets[axis]!=s.offsets[axis])
    {s.status=invalid_drift;return false;}
  // A tallbox open axis has no periodic images. An integer-domain overflow
  // makes its position unavailable rather than inventing an open-axis wrap.
  return true;
}

/* Only for a native operation explicitly defined to move within the nearest
 * image (the BH center-of-mass assignment). This is not a snapshot unwrapper. */
template <typename U>
inline bool nearest_relocation(research_periodic_image_state &s, int axis, U before, U after, unsigned bits)
{
  if(s.version!=1 || s.status!=supported)return false;
  if(axis<0 || axis>2 || bits<1 || bits>sizeof(U)*CHAR_BIT){s.status=ambiguous_relocation;return false;}
  before&=mask<U>(bits); after&=mask<U>(bits);
  const U half=U(1)<<(bits-1), distance=before>after ? before-after : after-before;
  if(distance==half){s.status=ambiguous_relocation;return false;}
  if(distance>half && !increment(s.offsets[axis],after>before,1))
    {s.status=counter_overflow;return false;}
  return true;
}
}
#endif
