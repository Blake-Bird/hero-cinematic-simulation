/* First-party angular moments for the native directed angular models.
 * x=1-cos(theta), p(x)=(1+2a)/(2(1+a*x)^2), a=v^2/(2w^2).
 * Truncated moments retain the normalization of the full physical cross
 * section: they are not conditional moments divided by the small-angle mass.
 * Rutherford convention is directed/distinguishable, not symmetrized Moller.
 * Reference: Arido, Fischer & Garny 2410.07175, appendices A and B.
 */
#ifndef RESEARCH_SIDM_ANGULAR_TRANSPORT_H
#define RESEARCH_SIDM_ANGULAR_TRANSPORT_H
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace research_sidm {
struct angular_transport_moments
{
  double small_probability=0,large_probability=1;
  double small_transfer=0,small_second_moment=0,small_viscosity=0;
  double full_transfer=0,full_second_moment=0,full_viscosity=0;
  double neglected_moment_ratio=0;
};
inline long double angular_power_integral(int power,long double a,long double limit,long double absolute_tolerance)
{
  if(limit==0)return 0;
  const long double x=a*limit,normalization=(1+2*a)/2;
  if(a==0)return std::pow(limit,power+1)/(2*(power+1));
  if(x<.25L) {
    // Convergent alternating series with an explicit next-term bound; this
    // avoids cancellation in the isotropic/low-speed limit.
    long double sum=0,scale=1;
    const long double factor=normalization*std::pow(limit,power+1);
    for(int order=0;order<256;order++) {
      const long double term=scale*(order+1)/(order+power+1);
      sum+=term;
      const long double next=std::abs(scale*x)*(order+2)/(order+power+2);
      if(factor*next<=absolute_tolerance/4)return factor*sum;
      scale*=-x;
    }
    throw std::runtime_error("SIDM angular moment series exceeded its bounded work allocation");
  }
  const long double logarithm=std::log1p(x),fraction=x/(1+x);
  long double numerator,denominator;
  if(power==1) {numerator=logarithm-fraction;denominator=a*a;}
  else if(power==2) {numerator=x+fraction-2*logarithm;denominator=a*a*a;}
  else throw std::runtime_error("SIDM angular transport supports first and second moments only");
  const long double result=normalization*numerator/denominator;
  const long double roundoff=256*std::numeric_limits<long double>::epsilon()*normalization*
      (std::abs(x)+std::abs(fraction)+2*std::abs(logarithm))/denominator;
  if(!std::isfinite(result) || result<0 || !std::isfinite(roundoff) || roundoff>absolute_tolerance)
    throw std::runtime_error("SIDM angular moment outside requested finite arithmetic precision");
  return result;
}
inline angular_transport_moments directed_angular_transport(bool screened,double speed_kms,double screening_velocity_kms,
                                                          double split_angle_radians,double absolute_tolerance)
{
  const long double pi=3.1415926535897932384626433832795L;
  if(!std::isfinite(speed_kms) || speed_kms<0 || !std::isfinite(split_angle_radians) ||
      split_angle_radians<0 || split_angle_radians>pi || !std::isfinite(absolute_tolerance) || absolute_tolerance<=0)
    throw std::runtime_error("SIDM angular moment physical domain invalid");
  if(screened && (!std::isfinite(screening_velocity_kms) || screening_velocity_kms<=0))
    throw std::runtime_error("SIDM angular transport screening scale invalid");
  const long double ratio=screened?static_cast<long double>(speed_kms)/screening_velocity_kms:0;
  const long double a=ratio*ratio/2;
  const long double sine=std::sin(static_cast<long double>(split_angle_radians)/2),cutoff=2*sine*sine;
  if(!std::isfinite(a))throw std::runtime_error("SIDM angular anisotropy overflow");
  angular_transport_moments result;
  result.small_probability=double((1+2*a)*cutoff/(2*(1+a*cutoff)));
  result.large_probability=double((2-cutoff)/(2*(1+a*cutoff)));
  result.small_transfer=double(angular_power_integral(1,a,cutoff,absolute_tolerance));
  result.small_second_moment=double(angular_power_integral(2,a,cutoff,absolute_tolerance));
  result.full_transfer=double(angular_power_integral(1,a,2,absolute_tolerance));
  result.full_second_moment=double(angular_power_integral(2,a,2,absolute_tolerance));
  result.small_viscosity=2*result.small_transfer-result.small_second_moment;
  result.full_viscosity=2*result.full_transfer-result.full_second_moment;
  result.neglected_moment_ratio=result.small_transfer>0?result.small_second_moment/result.small_transfer:0;
  for(double value:{result.small_probability,result.large_probability,result.small_transfer,result.small_second_moment,
      result.full_transfer,result.full_second_moment,result.small_viscosity,result.full_viscosity,result.neglected_moment_ratio})
    if(!std::isfinite(value) || value<0)throw std::runtime_error("SIDM angular moments lost finite nonnegative physical support");
  return result;
}
inline double directed_large_angle_cosine(double quantile,double speed_kms,bool screened,double screening_velocity_kms,
                                         double split_angle_radians,double absolute_tolerance)
{
  if(!std::isfinite(quantile) || quantile<=0 || quantile>=1)
    throw std::runtime_error("SIDM large-angle quantile must lie strictly between zero and one");
  const auto moments=directed_angular_transport(screened,speed_kms,screening_velocity_kms,split_angle_radians,absolute_tolerance);
  if(moments.large_probability<=0)throw std::runtime_error("SIDM large-angle distribution has empty physical support");
  const long double ratio=screened?static_cast<long double>(speed_kms)/screening_velocity_kms:0,a=ratio*ratio/2;
  const long double u=static_cast<long double>(quantile)*moments.large_probability;
  const long double mu=1-2*(1-u)/(1+2*a*u);
  const long double recovered=(1+mu)/(2*(1+a*(1-mu)));
  if(!std::isfinite(mu) || std::abs(recovered/moments.large_probability-quantile)>absolute_tolerance)
    throw std::runtime_error("SIDM conditional large-angle inverse CDF exceeds its frozen precision");
  return double(mu);
}
}
#endif
