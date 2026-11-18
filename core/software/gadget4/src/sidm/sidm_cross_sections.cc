#include "gadgetconfig.h"
#ifdef RESEARCH_SIDM_REFERENCE
#include "sidm.h"
#include "research_moller_transport.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace research_sidm {
namespace {
const double pi=3.1415926535897932384626433832795;
// FIPS 180-4 SHA-256; retained model bytes are checked before parsing. This
// avoids trusting a mutable table merely because its header names a hash.
std::string sha256(const std::string &input)
{
  static const std::uint32_t k[64]={
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  std::vector<unsigned char> data(input.begin(),input.end());
  const std::uint64_t bits=std::uint64_t(data.size())*8;data.push_back(0x80);
  while(data.size()%64!=56)data.push_back(0);
  for(int i=7;i>=0;--i)data.push_back(static_cast<unsigned char>(bits>>(8*i)));
  std::array<std::uint32_t,8> h={{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
  const auto rotate=[](std::uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));};
  for(std::size_t offset=0;offset<data.size();offset+=64) {
    std::uint32_t w[64];
    for(int i=0;i<16;i++)w[i]=(std::uint32_t(data[offset+4*i])<<24)|(std::uint32_t(data[offset+4*i+1])<<16)|
        (std::uint32_t(data[offset+4*i+2])<<8)|data[offset+4*i+3];
    for(int i=16;i<64;i++) {
      const auto s0=rotate(w[i-15],7)^rotate(w[i-15],18)^(w[i-15]>>3);
      const auto s1=rotate(w[i-2],17)^rotate(w[i-2],19)^(w[i-2]>>10);
      w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
    for(int i=0;i<64;i++) {
      const auto s1=rotate(e,6)^rotate(e,11)^rotate(e,25),choice=(e&f)^(~e&g);
      const auto t1=z+s1+choice+k[i]+w[i];
      const auto s0=rotate(a,2)^rotate(a,13)^rotate(a,22),majority=(a&b)^(a&c)^(b&c),t2=s0+majority;
      z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
  }
  std::ostringstream output;output<<std::hex<<std::setfill('0');for(auto value:h)output<<std::setw(8)<<value;
  return output.str();
}
double interpolate(const std::vector<double> &x,const std::vector<double> &y,double value)
{
  if(value<=x.front())return y.front();if(value>=x.back())return y.back();
  const auto upper=std::upper_bound(x.begin(),x.end(),value)-x.begin();
  const double fraction=(value-x[upper-1])/(x[upper]-x[upper-1]);
  return (1-fraction)*y[upper-1]+fraction*y[upper];
}
// Exact segment polynomial integration through degree three. Differential
// density is linear; transfer and viscosity moments are quadratic at most.
long double weighted_segment(double lo,double hi,double ylo,double yhi,int moment)
{
  const long double a=lo,b=hi,slope=(yhi-ylo)/(b-a),intercept=ylo-slope*a;
  const long double m0=intercept*(b-a)+slope*(b*b-a*a)/2;
  const long double m1=intercept*(b*b-a*a)/2+slope*(b*b*b-a*a*a)/3;
  const long double m2=intercept*(b*b*b-a*a*a)/3+slope*(b*b*b*b-a*a*a*a)/4;
  if(moment==0)return m0;if(moment==1)return m0-m1;
  if(moment==2)return m0-2*m1+m2;return m0-m2;
}
}
differential_table differential_table::read(const std::string &path,const std::string &model_hash,const std::string &table_hash)
{
  std::ifstream source(path,std::ios::binary);if(!source)throw std::runtime_error("Native SIDM differential table missing");
  source.seekg(0,std::ios::end);const auto size=source.tellg();
  if(size<=0 || size>std::streamoff(256*1024*1024))throw std::runtime_error("Native SIDM table exceeds finite retained allocation");
  source.seekg(0);std::string bytes(static_cast<std::size_t>(size),'\0');
  if(!source.read(&bytes[0],size))throw std::runtime_error("Native SIDM table read failed");
  if(table_hash.size()!=64 || sha256(bytes)!=table_hash)throw std::runtime_error("Native SIDM differential table SHA256 changed");
  std::istringstream input(bytes);std::string magic,declared;std::uint64_t nv=0,nmu=0;
  differential_table result;input>>magic>>declared>>nv>>nmu>>result.low_policy_>>result.high_policy_>>
      result.absolute_tolerance_>>result.relative_tolerance_;
  if(magic!="SIDM_DIFFERENTIAL_TABLE_V2" || declared!=model_hash || model_hash.size()!=64 || nv<2 || nmu<3 ||
      nv>4000000/nmu || result.low_policy_<0 || result.low_policy_>2 || result.high_policy_<0 || result.high_policy_>2 ||
      !std::isfinite(result.absolute_tolerance_) || result.absolute_tolerance_<=0 ||
      !std::isfinite(result.relative_tolerance_) || result.relative_tolerance_<=0 || result.relative_tolerance_>=1)
    throw std::runtime_error("Native SIDM differential model header or interpolation contract invalid");
  result.velocities_.resize(nv);result.cosines_.resize(nmu);result.differential_.resize(nv*nmu);
  for(auto &value:result.velocities_)input>>value;for(auto &value:result.cosines_)input>>value;
  for(auto &value:result.differential_)input>>value;
  if(!input)throw std::runtime_error("Native SIDM differential table truncated or nonnumeric");
  std::string trailing;if(input>>trailing)throw std::runtime_error("Native SIDM differential table has undeclared trailing fields");
  for(std::size_t i=0;i<nv;i++)if(!std::isfinite(result.velocities_[i]) || result.velocities_[i]<=0 ||
      (i && result.velocities_[i]<=result.velocities_[i-1]))throw std::runtime_error("Native SIDM table velocity support invalid");
  for(std::size_t i=0;i<nmu;i++)if(!std::isfinite(result.cosines_[i]) || (i && result.cosines_[i]<=result.cosines_[i-1]))
    throw std::runtime_error("Native SIDM table angular support invalid");
  if(result.cosines_.front()!=-1 || result.cosines_.back()!=1)throw std::runtime_error("Native SIDM table must cover all scattering cosines");
  for(double value:result.differential_)if(!std::isfinite(value) || value<0)
    throw std::runtime_error("Native SIDM differential kernel must be finite nonnegative");
  result.model_hash_=model_hash;result.table_hash_=table_hash;return result;
}
std::vector<double> differential_table::row(double speed) const
{
  if(!std::isfinite(speed) || speed<0)throw std::runtime_error("Native SIDM table received invalid relative speed");
  const auto count=cosines_.size();std::vector<double> result(count);
  if(speed<velocities_.front() || speed>velocities_.back()) {
    const bool low=speed<velocities_.front();const int policy=low?low_policy_:high_policy_;
    if(policy==0)throw std::runtime_error("Native SIDM relative speed exceeds declared differential table support");
    if(policy==2)return result;
    const std::size_t offset=low?0:(velocities_.size()-1)*count;
    std::copy(differential_.begin()+offset,differential_.begin()+offset+count,result.begin());return result;
  }
  std::size_t upper=std::min<std::size_t>(std::upper_bound(velocities_.begin(),velocities_.end(),speed)-velocities_.begin(),velocities_.size()-1);
  const auto lower=upper-1;const double fraction=std::log(speed/velocities_[lower])/std::log(velocities_[upper]/velocities_[lower]);
  for(std::size_t i=0;i<count;i++)result[i]=(1-fraction)*differential_[lower*count+i]+fraction*differential_[upper*count+i];
  return result;
}
double differential_table::differential_cross_section(double speed,double cosine) const
{
  if(!std::isfinite(cosine) || cosine<-1 || cosine>1)throw std::runtime_error("Native SIDM table cosine outside support");
  return interpolate(cosines_,row(speed),cosine);
}
double differential_table::total_cross_section(double speed) const
{
  const auto density=row(speed);long double total=0;
  for(std::size_t i=1;i<cosines_.size();i++)total+=(cosines_[i]-cosines_[i-1])*(density[i]+density[i-1])*pi;
  if(!std::isfinite(total) || total>std::numeric_limits<double>::max())throw std::runtime_error("Native SIDM total cross section overflow");
  return double(total);
}
angular_transport_moments differential_table::moments(double speed,double angle,bool folded) const
{
  const double maximum=folded?pi/2:pi;
  if(!std::isfinite(angle) || angle<0 || angle>maximum)throw std::runtime_error("Native SIDM table moment split invalid");
  const auto original=row(speed);std::vector<double> edges=cosines_;
  const double cutoff=std::cos(angle);edges.push_back(0);edges.push_back(cutoff);if(folded)edges.push_back(-cutoff);
  std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
  long double total=0,small=0,first=0,second=0,viscosity=0,small_first=0,small_second=0,small_viscosity=0;
  for(std::size_t i=1;i<edges.size();i++) {
    double lo=edges[i-1],hi=edges[i],yl=interpolate(cosines_,original,lo),yh=interpolate(cosines_,original,hi);
    const bool is_small=(folded?std::abs((lo+hi)/2):(lo+hi)/2)>cutoff;
    if(folded && hi<=0) {const double old=lo;lo=-hi;hi=-old;std::swap(yl,yh);}
    const auto m0=weighted_segment(lo,hi,yl,yh,0)*2*pi,m1=weighted_segment(lo,hi,yl,yh,1)*2*pi;
    const auto m2=weighted_segment(lo,hi,yl,yh,2)*2*pi,mv=weighted_segment(lo,hi,yl,yh,3)*2*pi;
    total+=m0;first+=m1;second+=m2;viscosity+=mv;
    if(is_small){small+=m0;small_first+=m1;small_second+=m2;small_viscosity+=mv;}
  }
  angular_transport_moments result;if(total==0){result.large_probability=0;return result;}
  result.small_probability=double(small/total);result.large_probability=double((total-small)/total);
  result.small_transfer=double(small_first/total);result.small_second_moment=double(small_second/total);
  result.small_viscosity=double(small_viscosity/total);result.full_transfer=double(first/total);
  result.full_second_moment=double(second/total);result.full_viscosity=double(viscosity/total);
  result.neglected_moment_ratio=small_first>0?double(small_second/small_first):0;
  // Allow only finite roundoff at segment endpoints, never a negative kernel.
  for(double *value:{&result.small_probability,&result.large_probability,&result.small_transfer,&result.small_second_moment,
      &result.small_viscosity,&result.full_transfer,&result.full_second_moment,&result.full_viscosity,&result.neglected_moment_ratio}) {
    if(!std::isfinite(*value) || *value < -absolute_tolerance_)throw std::runtime_error("Native SIDM table moments lost physical support");
    *value=std::max(0.,*value);
  }
  return result;
}
double differential_table::sample_cosine(double speed,double uniform,double minimum_angle,bool folded) const
{
  const double maximum=folded?pi/2:pi;
  if(!std::isfinite(uniform) || uniform<0 || uniform>1 || !std::isfinite(minimum_angle) || minimum_angle<0 || minimum_angle>maximum)
    throw std::runtime_error("Native SIDM table angle draw outside declared domain");
  const auto original=row(speed);const double upper=std::cos(minimum_angle),lower=folded?0:-1;
  std::vector<double> edges={lower,upper};
  for(double cosine:cosines_) {if(cosine>lower && cosine<upper)edges.push_back(cosine);
    if(folded && -cosine>lower && -cosine<upper)edges.push_back(-cosine);}
  std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
  std::vector<double> density;for(double mu:edges)density.push_back(interpolate(cosines_,original,mu)+
      (folded?interpolate(cosines_,original,-mu):0));
  long double total=0;for(std::size_t i=1;i<edges.size();i++)total+=(edges[i]-edges[i-1])*(density[i]+density[i-1])/2;
  if(total<=0 || !std::isfinite(total))throw std::runtime_error("Native SIDM conditional angle distribution has no support");
  long double remaining=uniform*total;
  for(std::size_t i=1;i<edges.size();i++) {
    const long double width=edges[i]-edges[i-1],mass=width*(density[i]+density[i-1])/2;
    if(remaining>mass && i+1<edges.size()){remaining-=mass;continue;}
    const long double slope=(density[i]-density[i-1])/width;
    const long double root=std::sqrt(std::max(0.L,static_cast<long double>(density[i-1])*density[i-1]+2*slope*remaining));
    const long double denominator=density[i-1]+root,delta=denominator>0?2*remaining/denominator:0;
    return double(std::max(static_cast<long double>(edges[i-1]),std::min(static_cast<long double>(edges[i]),edges[i-1]+delta)));
  }
  throw std::runtime_error("Native SIDM table inverse CDF exhausted its finite angular support");
}
long double differential_table::maximum_sigma_speed(long double upper) const
{
  if(!std::isfinite(upper) || upper<0)throw std::runtime_error("Native SIDM table rate bound speed invalid");
  // For a log-linear sigma, sigma(v) <= max(endpoint sigma). Multiplying that
  // by the interval's largest attainable speed is a conservative bound even
  // when sigma*v has an interior extremum. Error policies are enforced for
  // actual pair evaluations; a loose energy bound need not lie inside table.
  long double maximum=0;
  for(std::size_t i=0;i<velocities_.size();i++) {
    const double sigma=total_cross_section(velocities_[i]);
    const long double speed=std::min(upper,static_cast<long double>(i+1<velocities_.size()?velocities_[i+1]:velocities_[i]));
    maximum=std::max(maximum,static_cast<long double>(sigma)*speed);
  }
  if(upper>velocities_.back() && high_policy_==1)maximum=std::max(maximum,total_cross_section(velocities_.back())*upper);
  return maximum;
}
}
#endif
