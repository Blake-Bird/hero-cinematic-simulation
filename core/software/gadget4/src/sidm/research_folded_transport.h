/* Folded identical-particle convention for isotropic/Rutherford angular laws.
 * u=|cos(theta)|, x=1-u. The folded density is p(u)+p(-u), retaining the full
 * total cross section. This is a folded Rutherford model, not quantum Moller
 * scattering: no t/u-channel interference is invented by folding labels.
 */
#ifndef RESEARCH_SIDM_FOLDED_TRANSPORT_H
#define RESEARCH_SIDM_FOLDED_TRANSPORT_H
#include "research_angular_transport.h"

namespace research_sidm {
inline long double folded_cumulative(long double u,long double a)
{
  if(u<=0)return 0;if(u>=1)return 1;
  return u*(1+2*a)/((1+a*(1-u))*(1+a*(1+u)));
}
inline long double folded_reverse_power_integral(int power,long double a,long double limit,long double tolerance)
{
  if(limit==0)return 0;
  const long double b=a/(1+2*a),ratio=b*limit;
  const long double factor=std::pow(limit,power+1)/(2*(1+2*a));
  long double sum=0,scale=1;
  for(int order=0;order<256;order++) {
    sum+=scale*(order+1)/(order+power+1);
    const long double next=scale*ratio*(order+2)/(order+power+2);
    const long double geometric=ratio*(order+3)/(order+2);
    if(geometric<1 && factor*next/(1-geometric)<=tolerance/4)return factor*sum;
    scale*=ratio;
  }
  throw std::runtime_error("SIDM folded reverse angular moment exceeds bounded series allocation");
}
inline angular_transport_moments folded_angular_transport(bool screened,double speed,double screening_velocity,
                                                        double split_angle,double tolerance)
{
  // Reuse physical-domain checks; directed moments are not reused as folded
  // moments, because backward events have a small folded deflection too.
  directed_angular_transport(screened,speed,screening_velocity,0,tolerance);
  if(!std::isfinite(split_angle) || split_angle<0 || split_angle>1.5707963267948966192313216916398)
    throw std::runtime_error("SIDM folded split angle must lie in [0,pi/2]");
  const long double ratio=screened?static_cast<long double>(speed)/screening_velocity:0,a=ratio*ratio/2;
  const long double sine=std::sin(static_cast<long double>(split_angle)/2),cutoff=2*sine*sine;
  const auto moment=[&](int power,long double limit) {
    return angular_power_integral(power,a,limit,tolerance/2)+folded_reverse_power_integral(power,a,limit,tolerance/2);
  };
  angular_transport_moments result;
  result.large_probability=double(folded_cumulative(1-cutoff,a));
  // Compute the small-angle probability without subtracting a near-one CDF.
  result.small_probability=double((1+2*a)*cutoff/(2*(1+a*cutoff))+
      cutoff/(2*(1+2*a-a*cutoff)));
  result.small_transfer=double(moment(1,cutoff));result.small_second_moment=double(moment(2,cutoff));
  result.full_transfer=double(moment(1,1));result.full_second_moment=double(moment(2,1));
  result.small_viscosity=2*result.small_transfer-result.small_second_moment;
  result.full_viscosity=2*result.full_transfer-result.full_second_moment;
  result.neglected_moment_ratio=result.small_transfer>0?result.small_second_moment/result.small_transfer:0;
  for(double value:{result.small_probability,result.large_probability,result.small_transfer,result.small_second_moment,
      result.full_transfer,result.full_second_moment,result.small_viscosity,result.full_viscosity,result.neglected_moment_ratio})
    if(!std::isfinite(value) || value<0)throw std::runtime_error("SIDM folded moments lost finite physical support");
  return result;
}
inline double folded_large_angle_cosine(double quantile,double speed,bool screened,double screening_velocity,
                                       double split_angle,double tolerance)
{
  if(!std::isfinite(quantile) || quantile<=0 || quantile>=1)
    throw std::runtime_error("SIDM folded quantile must lie strictly between zero and one");
  const auto moments=folded_angular_transport(screened,speed,screening_velocity,split_angle,tolerance);
  if(moments.large_probability<=0)throw std::runtime_error("SIDM folded large-angle sample has empty support");
  const long double ratio=screened?static_cast<long double>(speed)/screening_velocity:0,a=ratio*ratio/2;
  const long double q=static_cast<long double>(quantile)*moments.large_probability;
  const long double center=1+a,coefficient=1+2*a;
  const long double discriminant=std::hypot(coefficient,2*q*a*center);
  const long double u=2*q*center*center/(coefficient+discriminant);
  if(!std::isfinite(u) || u<0 || u>1 || std::abs(folded_cumulative(u,a)/moments.large_probability-quantile)>tolerance)
    throw std::runtime_error("SIDM folded inverse CDF exceeds its frozen physical precision");
  const double retained=double(u);
  if(std::abs(folded_cumulative(retained,a)/moments.large_probability-quantile)>tolerance)
    throw std::runtime_error("SIDM retained folded scattering cosine exceeds its native precision requirement");
  return retained;
}
inline angular_transport_moments physical_angular_transport(bool folded,bool screened,double speed,double screening_velocity,
                                                          double cutoff,double tolerance)
{
  return folded?folded_angular_transport(screened,speed,screening_velocity,cutoff,tolerance):
      directed_angular_transport(screened,speed,screening_velocity,cutoff,tolerance);
}
}
#endif
