/* First-party compact SIDM kernel convolution in physical units.
 * g_ij = integral W_i(x) W_j(x-r_ij) d^3x, units cm^-3.
 * Cubic spline support radius H: W=8/(pi H^3) f(r/H),
 * f(q)=1-6q^2+6q^3 below 1/2; 2(1-q)^3 up to 1.
 * Piecewise polynomial radial convolution uses Gauss-Legendre rules exact
 * for each polynomial segment in real arithmetic (no sampled lookup table).
 * Reference: Rocha et al. 2013, MNRAS 430, 81; TangoSIDM appendix A.
 */
#ifndef RESEARCH_SIDM_KERNEL_OVERLAP_H
#define RESEARCH_SIDM_KERNEL_OVERLAP_H
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace research_sidm {
namespace kernel_detail {
constexpr long double pi=3.141592653589793238462643383279502884L;
inline long double spline(long double q)
{
  if(q<0 || q>=1)return 0;
  if(q<.5L)return 1+q*q*(-6+6*q);
  const long double t=1-q;
  return 2*t*t*t;
}
template<class Function>
inline long double gauss8(Function function,long double left,long double right)
{
  static const long double nodes[4]={.183434642495649804939476142360184L,
    .525532409916328985817739049189246L,.796666477413626739591553936475831L,
    .960289856497536231683560868569473L};
  static const long double weights[4]={.362683783378361982965150449277195L,
    .313706645877887287337962201986601L,.222381034453374470544355994426240L,
    .101228536290376259152531354309962L};
  const long double half=(right-left)/2,center=left+half;
  long double sum=0;
  for(int i=0;i<4;i++)sum+=weights[i]*(function(center-half*nodes[i])+function(center+half*nodes[i]));
  return half*sum;
}
inline long double radial_primitive_difference(long double left,long double right,long double support)
{
  left=std::max(0.L,left);right=std::min(support,right);
  if(right<=left)return 0;
  // Integrate s f(s/H) without subtracting nearly equal fifth-degree
  // antiderivatives. Three-point Gauss integrates this degree-four polynomial.
  const long double split=support/2;
  const auto segment=[support](long double lo,long double hi) {
    const long double half=(hi-lo)/2,center=lo+half;
    const long double offset=half*.774596669241483377035853079956480L;
    const auto value=[support](long double s){return s*spline(s/support);};
    return half*((5.L/9)*(value(center-offset)+value(center+offset))+(8.L/9)*value(center));
  };
  if(left<split && right>split)return segment(left,split)+segment(split,right);
  return segment(left,right);
}
inline long double coincident(long double a,long double b)
{
  std::vector<long double> edges={0,std::min(a,b)};
  for(long double q:{a/2,b/2})if(q>0 && q<edges[1])edges.push_back(q);
  std::sort(edges.begin(),edges.end());
  long double sum=0;
  for(std::size_t i=1;i<edges.size();i++)sum+=gauss8([a,b](long double r) {
    return r*r*spline(r/a)*spline(r/b);
  },edges[i-1],edges[i]);
  return 256*sum/(pi*a*a*a*b*b*b);
}
}

inline double uniform_ball_kernel_overlap(double distance_cm,double left_radius_cm,double right_radius_cm)
{
  for(double value:{left_radius_cm,right_radius_cm})if(!std::isfinite(value) || value<=0)
    throw std::runtime_error("SIDM uniform kernel support must be finite and positive");
  if(!std::isfinite(distance_cm) || distance_cm<0)throw std::runtime_error("SIDM physical pair separation invalid");
  const long double scale=std::max(left_radius_cm,right_radius_cm);
  const long double a=left_radius_cm/scale,b=right_radius_cm/scale,d=distance_cm/scale;
  if(d>=a+b)return 0;
  long double volume;
  if(d<=std::abs(a-b))volume=(4*kernel_detail::pi/3)*std::pow(std::min(a,b),3);
  else {
    const long double sum=a+b,difference=a-b;
    volume=kernel_detail::pi*(sum-d)*(sum-d)*(d*d+2*d*sum-3*difference*difference)/(12*d);
  }
  const long double normalization=(4*kernel_detail::pi/3)*(4*kernel_detail::pi/3)*a*a*a*b*b*b;
  const long double result=(volume/normalization)/scale/scale/scale;
  if(!std::isfinite(result) || result<0 || result>std::numeric_limits<double>::max())
    throw std::runtime_error("SIDM uniform kernel convolution outside finite physical domain");
  return double(result);
}

inline double cubic_spline_kernel_overlap(double distance_cm,double left_radius_cm,double right_radius_cm,
                                         double near_origin_relative_bound)
{
  for(double value:{left_radius_cm,right_radius_cm,near_origin_relative_bound})
    if(!std::isfinite(value) || value<=0)throw std::runtime_error("SIDM cubic kernel support/error bound invalid");
  if(near_origin_relative_bound>=1 || !std::isfinite(distance_cm) || distance_cm<0)
    throw std::runtime_error("SIDM cubic kernel physical domain invalid");
  const long double scale=std::max(left_radius_cm,right_radius_cm);
  const long double a=left_radius_cm/scale,b=right_radius_cm/scale,d=distance_cm/scale;
  if(d>=a+b)return 0;
  if(std::min(a,b)<1.e-6L)
    throw std::runtime_error("SIDM cubic kernel support ratio outside declared finite arithmetic domain");
  long double dimensionless=0;
  if(d==0)dimensionless=kernel_detail::coincident(a,b);
  else if(d<1.e-8L) {
    const long double origin=kernel_detail::coincident(a,b);
    // Translation bound using integral W_i=1 and max |grad W_j|.
    // Choose the larger support for the tighter bound; f' has magnitude <=2.
    const long double lipschitz=16*d/(kernel_detail::pi*std::pow(std::max(a,b),4));
    if(lipschitz<=near_origin_relative_bound*origin)dimensionless=origin;
    else if(d<128*std::numeric_limits<long double>::epsilon())
      throw std::runtime_error("SIDM kernel translation below finite precision with requested origin bound unmet");
  }
  if(dimensionless==0) {
    std::vector<long double> edges={0,a};
    // Split at every knot of f(r/a), the absolute lower limit, and both
    // primitive limits crossing b/2 or b. The remaining integrand is polynomial.
    for(long double q:{a/2,d,b-d,b/2-d,d-b/2,d-b,d+b/2,d+b})
      if(q>0 && q<a)edges.push_back(q);
    std::sort(edges.begin(),edges.end());
    edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    long double sum=0;
    for(std::size_t i=1;i<edges.size();i++)sum+=kernel_detail::gauss8([a,b,d](long double r) {
      return r*kernel_detail::spline(r/a)*kernel_detail::radial_primitive_difference(std::abs(r-d),r+d,b);
    },edges[i-1],edges[i]);
    dimensionless=128*sum/(kernel_detail::pi*d*a*a*a*b*b*b);
  }
  const long double result=dimensionless/scale/scale/scale;
  if(!std::isfinite(result) || result<0 || result>std::numeric_limits<double>::max())
    throw std::runtime_error("SIDM cubic kernel convolution outside finite physical domain");
  return double(result);
}
inline double measured_kernel_overlap(int kind,double distance_cm,double left_radius_cm,double right_radius_cm,
                                      double near_origin_relative_bound)
{
  if(kind==0)return uniform_ball_kernel_overlap(distance_cm,left_radius_cm,right_radius_cm);
  if(kind==1)return cubic_spline_kernel_overlap(distance_cm,left_radius_cm,right_radius_cm,near_origin_relative_bound);
  throw std::runtime_error("SIDM native kernel kind must be uniform ball (0) or cubic spline (1)");
}
}
#endif
