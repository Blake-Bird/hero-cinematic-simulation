/* First-party reference adapter, authored for this research workspace.
 * The surrounding OpenGadget3 project is GPL-3.0-or-later; no native GADGET-4
 * transport, neighbor, scheduling or random code is used by this adapter.
 * OpenGadget3 independently integrates/samples the common differential law.
 */
#ifndef RESEARCH_OPENGADGET3_DIFFERENTIAL_TABLE_HPP
#define RESEARCH_OPENGADGET3_DIFFERENTIAL_TABLE_HPP
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace scatter {
namespace research_reference {
class table
{
  std::vector<double> velocity,mu,differential;
  int low=0,high=0;
 public:
  table()
  {
    const char *path=std::getenv("RESEARCH_SIDM_REFERENCE_TABLE");
    if(!path || !*path)throw std::runtime_error("Independent differential reference requires retained table environment path");
    std::ifstream input(path);std::string magic,hash;std::size_t nv=0,nmu=0;double absolute=0,relative=0;
    input>>magic>>hash>>nv>>nmu>>low>>high>>absolute>>relative;
    if(!input || magic!="SIDM_DIFFERENTIAL_TABLE_V2" || hash.size()!=64 || nv<2 || nmu<3 || nv>4000000/nmu ||
        low<0 || low>2 || high<0 || high>2 || !std::isfinite(absolute) || absolute<=0 || !std::isfinite(relative) || relative<=0 || relative>=1)
      throw std::runtime_error("Independent differential reference table header invalid");
    velocity.resize(nv);mu.resize(nmu);differential.resize(nv*nmu);
    for(double &x:velocity)input>>x;for(double &x:mu)input>>x;for(double &x:differential)input>>x;
    if(!input)throw std::runtime_error("Independent differential reference table truncated");
    for(std::size_t i=0;i<nv;i++)if(!std::isfinite(velocity[i]) || velocity[i]<=0 || (i && velocity[i]<=velocity[i-1]))
      throw std::runtime_error("Independent differential reference velocity coverage invalid");
    for(std::size_t i=0;i<nmu;i++)if(!std::isfinite(mu[i]) || (i && mu[i]<=mu[i-1]))
      throw std::runtime_error("Independent differential reference angular coverage invalid");
    if(mu.front()!=-1 || mu.back()!=1)throw std::runtime_error("Independent differential reference requires complete angular support");
    for(double x:differential)if(!std::isfinite(x) || x<0)throw std::runtime_error("Independent differential reference kernel negative or nonfinite");
  }
  double evaluate(double speed,double cosine) const
  {
    if(!std::isfinite(speed) || speed<0 || !std::isfinite(cosine) || cosine<-1 || cosine>1)
      throw std::runtime_error("Independent differential reference outside physical domain");
    std::size_t vi=0,vj=0;double vf=0;
    if(speed<velocity.front() || speed>velocity.back()) {
      const bool before=speed<velocity.front();const int policy=before?low:high;
      if(policy==0)throw std::runtime_error("Independent differential reference velocity policy rejects extrapolation");
      if(policy==2)return 0;vi=vj=before?0:velocity.size()-1;
    } else {
      vj=std::min<std::size_t>(std::upper_bound(velocity.begin(),velocity.end(),speed)-velocity.begin(),velocity.size()-1);vi=vj-1;
      vf=std::log(speed/velocity[vi])/std::log(velocity[vj]/velocity[vi]);
    }
    const std::size_t mj=std::max<std::size_t>(1,std::min<std::size_t>(std::upper_bound(mu.begin(),mu.end(),cosine)-mu.begin(),mu.size()-1));
    const std::size_t mi=mj-1;const double mf=(cosine-mu[mi])/(mu[mj]-mu[mi]);
    const auto angular=[&](std::size_t v){return (1-mf)*differential[v*mu.size()+mi]+mf*differential[v*mu.size()+mj];};
    return (1-vf)*angular(vi)+vf*angular(vj);
  }
};
}
inline double research_table_scattering(double theta,double vrel,double velocity_unit_kms)
{
  if(!std::isfinite(velocity_unit_kms) || velocity_unit_kms<=0)throw std::runtime_error("Independent differential reference velocity unit invalid");
  static const research_reference::table retained;
  // OpenGadget3's interface is d(sigma/m)/dcos(theta), rather than per sr.
  return 6.283185307179586476925286766559*retained.evaluate(vrel*velocity_unit_kms,std::cos(theta));
}
REGISTER_FUNC(research_table_scattering,theta,vrel,velocity_unit_kms);
}
#endif
