/* Bounded source-clock partition. No particle writes or model extrapolation. */
#ifndef RESEARCH_SIDM_DRAG_SCHEDULE_H
#define RESEARCH_SIDM_DRAG_SCHEDULE_H
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
namespace research_sidm_drag_schedule {
struct interval {double native_begin,native_end,proper_begin_seconds,proper_end_seconds;};
template<class Clock> std::vector<interval> partition(double begin,double end,double proper_origin,
    double maximum_step_seconds,std::size_t maximum_subcycles,std::uint64_t maximum_workspace_bytes,
    double absolute_clock_tolerance_seconds,double relative_clock_tolerance,Clock clock){
  if(!std::isfinite(begin)||!std::isfinite(end)||end<=begin||!std::isfinite(proper_origin)||
      !std::isfinite(maximum_step_seconds)||maximum_step_seconds<=0||maximum_subcycles<1||maximum_subcycles>4096||
      maximum_subcycles>maximum_workspace_bytes/128||!std::isfinite(absolute_clock_tolerance_seconds)||absolute_clock_tolerance_seconds<0||
      !std::isfinite(relative_clock_tolerance)||relative_clock_tolerance<0||relative_clock_tolerance>1e-6)
    throw std::runtime_error("Invalid bounded calibrated source schedule");
  double whole=clock(begin,end);
  if(!std::isfinite(whole)||whole<=0)throw std::runtime_error("Native source proper interval is not positive/finite");
  if(whole/maximum_step_seconds>maximum_subcycles)throw std::runtime_error("Calibrated source interval exceeds complete subcycle budget");
  std::vector<interval> plan;plan.reserve(maximum_subcycles);
  for(std::size_t count=1;;count=std::min(maximum_subcycles,2*count)){
    plan.clear();double left=begin,proper_left=proper_origin,total=0;bool short_enough=true;
    for(std::size_t i=0;i<count;i++){
      double right=i+1==count?end:begin+(end-begin)*double(i+1)/double(count);
      if(!std::isfinite(right)||right<=left)throw std::runtime_error("Native source subcycle endpoint loses precision");
      double duration=clock(left,right),proper_right=proper_left+duration;
      if(!std::isfinite(duration)||duration<=0||!std::isfinite(proper_right)||proper_right<=proper_left)
        throw std::runtime_error("Proper source subcycle clock loses precision or regresses");
      double stored_budget=absolute_clock_tolerance_seconds+relative_clock_tolerance*duration;
      if(!std::isfinite(stored_budget)||std::abs((proper_right-proper_left)-duration)>stored_budget)
        throw std::runtime_error("Stored proper source endpoints lose the declared clock accuracy");
      short_enough=short_enough&&duration<=maximum_step_seconds;
      plan.push_back({left,right,proper_left,proper_right});total+=duration;left=right;proper_left=proper_right;
    }
    double budget=absolute_clock_tolerance_seconds+relative_clock_tolerance*whole;
    if(!std::isfinite(total)||!std::isfinite(budget)||std::abs(total-whole)>budget)
      throw std::runtime_error("Partitioned native source clock differs from whole interval");
    if(short_enough)return plan;
    if(count==maximum_subcycles)throw std::runtime_error("Native source clock cannot meet its bounded subcycle duration");
  }
}
}
#endif
