#include "gadgetconfig.h"
#ifdef RESEARCH_SIDM_REFERENCE
#include "sidm.h"
#include "research_moller_transport.h"
#include <cmath>
#include <memory>
#include <stdexcept>
namespace research_sidm {
namespace {std::unique_ptr<differential_table> retained_native_model;}
const differential_table *native_differential_model(){return retained_native_model.get();}
void prepare_native_differential_model(int kind,const char *path,const char *model_hash,const char *table_hash)
{
  if(kind!=3){retained_native_model.reset();return;}
  if(!path || !*path || !model_hash || !table_hash)throw std::runtime_error("Native SIDM retained differential table identity missing");
  if(retained_native_model && retained_native_model->model_hash()==model_hash && retained_native_model->table_hash()==table_hash)return;
  retained_native_model.reset(new differential_table(differential_table::read(path,model_hash,table_hash)));
}
double native_total_cross_section(int kind,double normalization,double speed,double scale)
{
  if(!std::isfinite(normalization) || normalization<0 || !std::isfinite(speed) || speed<0)
    throw std::runtime_error("Native SIDM physical cross section input invalid");
  if(normalization==0)return 0;
  if(kind==3) {
    if(!retained_native_model)throw std::runtime_error("Native SIDM differential table was not prepared");
    return normalization*retained_native_model->total_cross_section(speed);
  }
  if(kind==0)return normalization;
  if(!std::isfinite(scale) || scale<=0)throw std::runtime_error("Native SIDM angular screening scale invalid");
  if(kind==1)return normalization/(1+std::pow(speed/scale,2));
  if(kind==2)return normalization*moller_total_factor(speed,scale);
  throw std::runtime_error("Unknown native SIDM differential model");
}
long double native_sigma_speed_bound(int kind,long double normalization,long double upper,long double scale)
{
  if(!std::isfinite(normalization) || normalization<0 || !std::isfinite(upper) || upper<0)
    throw std::runtime_error("Native SIDM rate bound input invalid");
  if(normalization==0 || upper==0)return 0;
  if(kind==3) {
    if(!retained_native_model)throw std::runtime_error("Native SIDM rate bound lacks retained differential model");
    return normalization*retained_native_model->maximum_sigma_speed(upper);
  }
  if(kind==0)return normalization*upper;
  // Moller total <= Rutherford total for the declared unpolarized symmetry
  // normalization. This bound remains valid for exchange interference.
  const long double velocity=std::min(upper,scale);
  return normalization*velocity/(1+(velocity/scale)*(velocity/scale));
}
}
#endif
