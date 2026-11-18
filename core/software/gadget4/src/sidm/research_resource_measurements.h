/* Native source resource telemetry. Measurements never change source physics. */
#ifndef RESEARCH_SIDM_RESOURCE_MEASUREMENTS_H
#define RESEARCH_SIDM_RESOURCE_MEASUREMENTS_H
#include <cmath>
#include <cstdint>
#include <limits>
#include <sys/resource.h>
#include <time.h>
namespace research_sidm {
struct source_resource_measurements
{
  double pair_cpu_seconds=0,rare_cpu_seconds=0,frequent_cpu_seconds=0;
  std::uint64_t rare_calls=0,frequent_calls=0,exchange_buffer_peak_bytes=0;
  bool cpu_available=true;
};
inline bool thread_cpu_now(long double &seconds)
{
#ifdef CLOCK_THREAD_CPUTIME_ID
  struct timespec stamp;
  if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&stamp)==0) {
    seconds=static_cast<long double>(stamp.tv_sec)+stamp.tv_nsec/1.e9L;
    return std::isfinite(seconds);
  }
#endif
  return false;
}
class source_cpu_scope
{
  source_resource_measurements *measurements;
  int channel;
  long double begin=0;
  bool available=false;
public:
  source_cpu_scope(source_resource_measurements *value,int kind):measurements(value),channel(kind)
  {
    if(measurements)available=thread_cpu_now(begin);
  }
  ~source_cpu_scope() noexcept
  {
    if(!measurements)return;
    long double end=0;
    if(!available || !thread_cpu_now(end) || end<begin) {measurements->cpu_available=false;return;}
    const double elapsed=double(end-begin);
    double &total=channel==0?measurements->pair_cpu_seconds:channel==1?measurements->rare_cpu_seconds:measurements->frequent_cpu_seconds;
    if(!std::isfinite(total+elapsed)) {measurements->cpu_available=false;return;}
    total+=elapsed;
    if(channel) {
      std::uint64_t &calls=channel==1?measurements->rare_calls:measurements->frequent_calls;
      if(calls==UINT64_MAX)measurements->cpu_available=false;else ++calls;
    }
  }
  source_cpu_scope(const source_cpu_scope &)=delete;
  source_cpu_scope &operator=(const source_cpu_scope &)=delete;
};
inline bool process_peak_rss_bytes(std::uint64_t &bytes)
{
  struct rusage usage;
  if(getrusage(RUSAGE_SELF,&usage)!=0 || usage.ru_maxrss<0)return false;
#if defined(__linux__)
  const std::uint64_t multiplier=1024;
#elif defined(__APPLE__)
  const std::uint64_t multiplier=1;
#else
  // POSIX does not specify the units of this field. Never guess them.
  return false;
#endif
#if defined(__linux__) || defined(__APPLE__)
  if(std::uint64_t(usage.ru_maxrss)>UINT64_MAX/multiplier)return false;
  bytes=std::uint64_t(usage.ru_maxrss)*multiplier;
  return true;
#endif
}
}
#endif
