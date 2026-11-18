/* Unpolarized identical spin-1/2 Born Yukawa scattering, including exchange
 * interference. Yang & Yu 2205.03392 equations 2-3; Arido et al. 2410.07175.
 * sigma0 denotes the Rutherford low-velocity total normalization. The Moller
 * total tends to sigma0/2, with the identical-final-state factor included once.
 * Folded u=|cos(theta)| removes the unobservable exchange of final-state labels.
 */
#ifndef RESEARCH_SIDM_MOLLER_TRANSPORT_H
#define RESEARCH_SIDM_MOLLER_TRANSPORT_H
#include "research_folded_transport.h"
namespace research_sidm {
inline long double moller_parameter(double speed,double scale)
{
  if(!std::isfinite(speed) || speed<0 || !std::isfinite(scale) || scale<=0)
    throw std::runtime_error("Born Moller velocity domain invalid");
  const long double ratio=static_cast<long double>(speed)/scale,a=ratio*ratio/2;
  if(!std::isfinite(a) || !std::isfinite(a*a))throw std::runtime_error("Born Moller anisotropy exceeds finite domain");
  return a;
}
inline long double moller_primitive(long double u,long double a)
{
  if(u<0 || u>1)throw std::runtime_error("Born Moller folded cosine invalid");
  if(a==0)return u/2;
  const long double c=1+a,plus=1+a*(1-u),minus=1+a*(1+u);
  // log1p form avoids subtracting nearly equal logarithms near isotropy/u=0.
  const long double interference=std::log1p(2*a*u/plus)/(4*a*c);
  const long double result=u/(plus*minus)-interference;
  if(!std::isfinite(result) || result<0)throw std::runtime_error("Born Moller primitive lost positive support");
  return result;
}
inline double moller_total_factor(double speed,double scale)
{return double(moller_primitive(1,moller_parameter(speed,scale)));}
inline long double moller_cumulative(long double u,long double a)
{return moller_primitive(u,a)/moller_primitive(1,a);}

inline long double moller_simple_power(int power,long double a,long double limit,bool reverse,long double tolerance)
{
  if(limit==0)return 0;
  const long double denominator=reverse?1+2*a:1;
  const long double ratio=(reverse?a/denominator:-a)*limit;
  if(reverse || std::abs(ratio)<.25L) {
    long double sum=0,scale=1;
    const long double factor=std::pow(limit,power+1)/denominator;
    for(int order=0;order<256;++order) {
      sum+=scale/(power+order+1);
      const long double next=std::abs(scale*ratio)/(power+order+2);
      if(std::abs(ratio)<1 && factor*next/(1-std::abs(ratio))<=tolerance/8)return factor*sum;
      scale*=ratio;
    }
    throw std::runtime_error("Born Moller interference moment exhausted bounded series");
  }
  long double value=std::log1p(a*limit)/a;
  for(int n=1;n<=power;++n)value=std::pow(limit,n)/(a*n)-value/a;
  const long double allowance=512*std::numeric_limits<long double>::epsilon()*
      (1+std::pow(limit,power+1))*(1+1/a+1/(a*a*a));
  if(!std::isfinite(value) || value<0 || allowance>tolerance)
    throw std::runtime_error("Born Moller interference moment cannot meet requested precision");
  return value;
}
inline angular_transport_moments moller_angular_transport(double speed,double scale,double angle,double tolerance)
{
  folded_angular_transport(true,speed,scale,angle,tolerance);
  const long double a=moller_parameter(speed,scale),total=moller_primitive(1,a);
  const long double sine=std::sin(static_cast<long double>(angle)/2),cutoff=2*sine*sine;
  const auto moment=[&](int power,long double limit) {
    const long double precision=tolerance*total/16;
    const long double interference=(moller_simple_power(power,a,limit,false,precision)+
        moller_simple_power(power,a,limit,true,precision))/(2*(1+a));
    long double direct;
    if(power==0)direct=limit/(2*(1+a*limit))+limit/(2*(1+2*a-a*limit));
    else direct=(angular_power_integral(power,a,limit,precision)+
        folded_reverse_power_integral(power,a,limit,precision))/(1+2*a);
    const long double result=(direct-interference/2)/total;
    if(!std::isfinite(result) || result<0)throw std::runtime_error("Born Moller moment lost nonnegative support");
    return double(result);
  };
  angular_transport_moments result;
  result.small_probability=moment(0,cutoff);
  result.large_probability=double(moller_cumulative(1-cutoff,a));
  result.small_transfer=moment(1,cutoff);result.small_second_moment=moment(2,cutoff);
  result.full_transfer=moment(1,1);result.full_second_moment=moment(2,1);
  result.small_viscosity=2*result.small_transfer-result.small_second_moment;
  result.full_viscosity=2*result.full_transfer-result.full_second_moment;
  result.neglected_moment_ratio=result.small_transfer?result.small_second_moment/result.small_transfer:0;
  for(double value:{result.small_probability,result.large_probability,result.small_transfer,result.small_second_moment,
      result.full_transfer,result.full_second_moment,result.small_viscosity,result.full_viscosity,result.neglected_moment_ratio})
    if(!std::isfinite(value) || value<0)throw std::runtime_error("Born Moller transport outside physical precision domain");
  return result;
}
inline double moller_large_angle_cosine(double quantile,double speed,double scale,double angle,double tolerance)
{
  if(!std::isfinite(quantile) || quantile<=0 || quantile>=1)
    throw std::runtime_error("Born Moller quantile must be strictly inside (0,1)");
  if(!std::isfinite(angle) || angle<0 || angle>1.57079632679489661923 || !std::isfinite(tolerance) || tolerance<=0)
    throw std::runtime_error("Born Moller inverse CDF domain invalid");
  const long double a=moller_parameter(speed,scale),maximum=std::cos(static_cast<long double>(angle));
  const long double mass=moller_primitive(maximum,a);
  if(mass<=0)throw std::runtime_error("Born Moller large-angle distribution has empty support");
  long double low=0,high=maximum;
  for(int iteration=0;iteration<128;++iteration) {
    const long double middle=(low+high)/2;
    const long double recovered=moller_primitive(middle,a)/mass;
    if(recovered<quantile)low=middle;else high=middle;
    const double retained=double(middle);
    if(std::abs(moller_primitive(retained,a)/mass-quantile)<=tolerance)return retained;
  }
  throw std::runtime_error("Born Moller retained inverse CDF exhausted bounded precision/work");
}
}
#endif
