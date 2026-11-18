/* Physical elapsed time for the actual GADGET cosmological background.
 * Native cosmological time is scale factor, not elapsed seconds. Integrate
 * dt = d(log a)/H(a) using the native Hubble callback and explicit bounds.
 * Adaptive Simpson disagreement is a numerical error estimate, not a
 * certified bound for arbitrary nonsmooth cosmology tables.
 */
#ifndef RESEARCH_SIDM_PHYSICAL_CLOCK_H
#define RESEARCH_SIDM_PHYSICAL_CLOCK_H
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace research_sidm {
struct physical_clock_result
{
  double elapsed_seconds=0,estimated_absolute_error_seconds=0;
  std::size_t function_evaluations=0,accepted_panels=0;
};
template<class Hubble>
physical_clock_result cosmological_elapsed_seconds(double a_begin,double a_end,double time_unit_seconds,
                                                  double hubble_parameter,Hubble hubble,
                                                  double absolute_tolerance_seconds,double relative_tolerance,
                                                  std::size_t maximum_panels,int canonical_scale_power=0)
{
  for(double value:{a_begin,a_end,time_unit_seconds,hubble_parameter,absolute_tolerance_seconds,relative_tolerance})
    if(!std::isfinite(value) || value<=0)throw std::runtime_error("SIDM physical clock domain invalid");
  if(a_end<=a_begin || relative_tolerance>=1 || maximum_panels<2)
    throw std::runtime_error("SIDM physical clock needs increasing epochs and explicit integration bounds");
  if(canonical_scale_power!=0 && canonical_scale_power!=2)
    throw std::runtime_error("SIDM clock selects proper time or explicit canonical d tau=dt/a^2 only");
  physical_clock_result result;
  const auto integrand=[&](double log_a) {
    ++result.function_evaluations;
    const double a=std::exp(log_a),h=hubble(a);
    if(!std::isfinite(h) || h<=0)throw std::runtime_error("SIDM native background has a nonpositive Hubble rate");
    const double value=time_unit_seconds/hubble_parameter/h/(canonical_scale_power==2?a*a:1.);
    if(!std::isfinite(value) || value<=0)throw std::runtime_error("SIDM physical clock integrand overflow");
    return value;
  };
  struct panel {double left,right,fl,fm,fr,coarse,tolerance;};
  const double left=std::log(a_begin),right=std::log(a_end),middle=left+(right-left)/2;
  if(middle<=left || middle>=right)throw std::runtime_error("SIDM cosmological interval below native log-clock precision");
  const double fl=integrand(left),fm=integrand(middle),fr=integrand(right);
  const double coarse=(right-left)*(fl+4*fm+fr)/6;
  const double tolerance=absolute_tolerance_seconds+relative_tolerance*std::abs(coarse);
  if(!std::isfinite(coarse) || !std::isfinite(tolerance))throw std::runtime_error("SIDM physical elapsed-time estimate overflow");
  std::vector<panel> pending;
  pending.push_back({left,right,fl,fm,fr,coarse,tolerance});
  std::size_t inspected=0;
  long double sum=0,error=0;
  while(!pending.empty()) {
    if(++inspected>maximum_panels)throw std::runtime_error("SIDM native physical-time quadrature work allocation exhausted");
    const panel current=pending.back();pending.pop_back();
    const double center=current.left+(current.right-current.left)/2;
    const double quarter=current.left+(center-current.left)/2,three_quarter=center+(current.right-center)/2;
    if(quarter<=current.left || quarter>=center || three_quarter<=center || three_quarter>=current.right)
      throw std::runtime_error("SIDM physical-time refinement below finite epoch precision");
    const double fq=integrand(quarter),ft=integrand(three_quarter);
    const double first=(center-current.left)*(current.fl+4*fq+current.fm)/6;
    const double second=(current.right-center)*(current.fm+4*ft+current.fr)/6;
    const double refined=first+second,disagreement=std::abs(refined-current.coarse)/15;
    if(!std::isfinite(refined) || !std::isfinite(disagreement))throw std::runtime_error("SIDM native clock refinement overflow");
    if(disagreement<=current.tolerance) {
      const double corrected=refined+(refined-current.coarse)/15;
      if(!std::isfinite(corrected) || corrected<=0)throw std::runtime_error("SIDM native clock accepted nonphysical elapsed time");
      sum+=corrected;error+=disagreement;++result.accepted_panels;
    } else {
      if(pending.size()+2>maximum_panels-inspected)
        throw std::runtime_error("SIDM native clock panel allocation cannot resolve the declared tolerance");
      pending.push_back({center,current.right,current.fm,ft,current.fr,second,current.tolerance/2});
      pending.push_back({current.left,center,current.fl,fq,current.fm,first,current.tolerance/2});
    }
  }
  result.elapsed_seconds=double(sum);result.estimated_absolute_error_seconds=double(error);
  if(!std::isfinite(result.elapsed_seconds) || result.elapsed_seconds<=0)
    throw std::runtime_error("SIDM measured cosmological source interval overflow");
  return result;
}
}
#endif
