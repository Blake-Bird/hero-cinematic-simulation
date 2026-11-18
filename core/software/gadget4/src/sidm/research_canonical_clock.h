/* Source-bound expanding background for a canonical gravity-free control.
 * Tabulated E(a) retains the native log-E/log-a interpolation and integrates
 * each power-law piece analytically. Ordinary matter/radiation/curvature/Lambda
 * backgrounds use the bounded native clock quadrature. Inverse lookup has
 * explicit clock, epoch, iteration and aggregate-work budgets, no extrapolation.
 */
#ifndef RESEARCH_SIDM_CANONICAL_CLOCK_H
#define RESEARCH_SIDM_CANONICAL_CLOCK_H
#include "research_physical_clock.h"
#include "../black_holes/research_asset_sha256.h"
#include <fstream>
#include <sstream>
#include <string>
#include <limits>

namespace research_sidm {
struct canonical_epoch_result {
  double scale_factor=0,canonical_elapsed_seconds=0,clock_absolute_error_estimate_seconds=0;
  double bracket_relative_width=0;
  std::size_t iterations=0,function_evaluations=0;
};
class canonical_control_clock {
  int kind_=0;
  double h0_=0,begin_=0,end_=0,absolute_=0,relative_=0,epoch_relative_=0;
  double matter_=0,lambda_=0,radiation_=0,curvature_=0;
  std::size_t panels_=0,inverse_=0;
  std::vector<double> a_,e_;
  std::string physical_model_,background_,asset_;
  static bool hash(const std::string &s) {
    if(s.size()!=64)return false;
    for(char c:s)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
    return true;
  }
 public:
  static canonical_control_clock read(const std::string &path,const std::string &expected_sha,
      const std::string &expected_model,std::size_t maximum_bytes,std::size_t maximum_rows) {
    if(!hash(expected_sha) || !hash(expected_model) || maximum_bytes<128 || maximum_rows<2)
      throw std::runtime_error("Canonical clock asset identity or allocation invalid");
    std::ifstream file(path,std::ios::binary);
    if(!file)throw std::runtime_error("Canonical clock asset is unavailable");
    file.seekg(0,std::ios::end);const auto length=file.tellg();
    if(length<128 || static_cast<unsigned long long>(length)>maximum_bytes)
      throw std::runtime_error("Canonical clock asset exceeds byte allocation");
    file.seekg(0);std::string bytes(static_cast<std::size_t>(length),'\0');
    file.read(&bytes[0],static_cast<std::streamsize>(bytes.size()));
    if(!file || file.peek()!=std::char_traits<char>::eof() || research_asset_sha256::bytes(bytes)!=expected_sha)
      throw std::runtime_error("Canonical clock actual asset bytes differ from their SHA256");
    canonical_control_clock result;std::istringstream input(bytes);std::string magic;std::size_t rows=0;
    if(!(input>>magic>>result.physical_model_>>result.background_>>result.kind_>>result.h0_>>result.begin_>>result.end_
        >>result.absolute_>>result.relative_>>result.panels_>>result.inverse_>>result.epoch_relative_>>rows
        >>result.matter_>>result.lambda_>>result.radiation_>>result.curvature_) || magic!="RESEARCH_SIDM_CANONICAL_CLOCK_V1"
        || result.physical_model_!=expected_model || !hash(result.background_) || (result.kind_!=0&&result.kind_!=1)
        || rows>maximum_rows || (result.kind_==0&&rows!=0) || (result.kind_==1&&rows<2))
      throw std::runtime_error("Canonical clock asset schema or original background binding invalid");
    for(double value:{result.h0_,result.begin_,result.end_,result.absolute_,result.relative_,result.epoch_relative_})
      if(!std::isfinite(value) || value<=0)throw std::runtime_error("Canonical clock positive physical bound invalid");
    for(double value:{result.matter_,result.lambda_,result.radiation_,result.curvature_})
      if(!std::isfinite(value))throw std::runtime_error("Canonical clock background coefficient invalid");
    if(result.end_<=result.begin_ || result.relative_>=1 || result.epoch_relative_>=1 || result.panels_<2 || result.inverse_<1)
      throw std::runtime_error("Canonical clock ordered domain or numerical budget invalid");
    result.a_.resize(rows);result.e_.resize(rows);
    for(std::size_t i=0;i<rows;i++) {
      if(!(input>>result.a_[i]>>result.e_[i]) || !std::isfinite(result.a_[i]) || !std::isfinite(result.e_[i]) || result.a_[i]<=0 || result.e_[i]<=0 || (i&&result.a_[i]<=result.a_[i-1]))
        throw std::runtime_error("Canonical clock native expansion knots invalid");
    }
    std::string extra;if(input>>extra)throw std::runtime_error("Canonical clock has undeclared trailing fields");
    if(rows && (result.begin_<result.a_.front() || result.end_>result.a_.back()))throw std::runtime_error("Canonical clock domain escapes original expansion table");
    result.asset_=expected_sha;return result;
  }
  physical_clock_result elapsed(double end,std::size_t maximum_work) const {
    if(!std::isfinite(end) || end<begin_ || end>end_)throw std::runtime_error("Canonical clock cannot extrapolate its source-bound epoch domain");
    physical_clock_result result;if(end==begin_)return result;
    if(kind_==0) {
      if(maximum_work<7)throw std::runtime_error("Canonical background quadrature work exhausted");
      const auto budget=std::min(panels_,(maximum_work-3)/2);
      return cosmological_elapsed_seconds(begin_,end,1.,1.,[&](double a) {
        const double squared=matter_/(a*a*a)+curvature_/(a*a)+radiation_/(a*a*a*a)+lambda_;
        if(!std::isfinite(squared) || squared<=0)throw std::runtime_error("Canonical original background has no positive finite Hubble rate");
        return h0_*std::sqrt(squared);
      },absolute_,relative_,budget,2);
    }
    double current=begin_;long double total=0;
    while(current<end) {
      if(result.function_evaluations>=maximum_work)throw std::runtime_error("Canonical exact table integration work exhausted");
      ++result.function_evaluations;++result.accepted_panels;
      auto upper=std::upper_bound(a_.begin(),a_.end(),current);
      if(upper==a_.begin() || upper==a_.end())throw std::runtime_error("Canonical clock cannot identify its original table interval");
      const std::size_t hi=static_cast<std::size_t>(upper-a_.begin()),lo=hi-1;
      const double next=std::min(end,a_[hi]);
      if(next<=current)throw std::runtime_error("Canonical clock interval below finite epoch precision");
      const long double left=std::log(static_cast<long double>(current)),dx=std::log(static_cast<long double>(next))-left;
      const long double slope=(std::log(static_cast<long double>(e_[hi]))-std::log(static_cast<long double>(e_[lo])))/(std::log(static_cast<long double>(a_[hi]))-std::log(static_cast<long double>(a_[lo])));
      const long double loge=std::log(static_cast<long double>(e_[lo]))+slope*(left-std::log(static_cast<long double>(a_[lo])));
      const long double exponent=-2-slope,z=exponent*dx;
      const long double factor=z==0?dx:dx*std::expm1(z)/z;
      const long double part=std::exp(-2*left-std::log(static_cast<long double>(h0_))-loge)*factor;
      if(!std::isfinite(part) || part<=0)throw std::runtime_error("Canonical exact table interval overflows");
      total+=part;current=next;
    }
    result.elapsed_seconds=static_cast<double>(total);
    result.estimated_absolute_error_seconds=256*std::numeric_limits<double>::epsilon()*std::abs(result.elapsed_seconds)*result.accepted_panels;
    if(!std::isfinite(result.elapsed_seconds) || !std::isfinite(result.estimated_absolute_error_seconds)
        || result.estimated_absolute_error_seconds>absolute_+relative_*std::abs(result.elapsed_seconds))
      throw std::runtime_error("Canonical exact table clock exceeds arithmetic budget");
    return result;
  }
  canonical_epoch_result epoch(double tau,std::size_t maximum_work) const {
    if(!std::isfinite(tau) || tau<0)throw std::runtime_error("Canonical elapsed time invalid");
    canonical_epoch_result result;result.scale_factor=begin_;
    if(tau==0)return result;
    auto full=elapsed(end_,maximum_work);result.function_evaluations=full.function_evaluations;
    // Permit only a positive endpoint disagreement within the declared clock
    // estimate/tolerance. The physical epoch stays at the source-bound endpoint;
    // the retained error includes the actual disagreement, never extrapolation.
    const double endpoint_budget=absolute_+relative_*std::abs(tau)+full.estimated_absolute_error_seconds;
    if(!std::isfinite(endpoint_budget) || tau-full.elapsed_seconds>endpoint_budget)
      throw std::runtime_error("Canonical inverse clock cannot escape endpoint uncertainty");
    if(tau>=full.elapsed_seconds){result.scale_factor=end_;result.canonical_elapsed_seconds=full.elapsed_seconds;result.clock_absolute_error_estimate_seconds=full.estimated_absolute_error_seconds+std::abs(tau-full.elapsed_seconds);return result;}
    double low=std::log(begin_),high=std::log(end_);
    for(std::size_t iteration=0;iteration<inverse_;iteration++) {
      if(result.function_evaluations>=maximum_work)throw std::runtime_error("Canonical inverse clock aggregate work exhausted");
      const double middle=low+(high-low)/2,a=std::exp(middle);
      if(middle<=low || middle>=high)throw std::runtime_error("Canonical inverse clock below finite precision");
      const auto clock=elapsed(a,maximum_work-result.function_evaluations);result.function_evaluations+=clock.function_evaluations;
      result.iterations=iteration+1;result.scale_factor=a;result.canonical_elapsed_seconds=clock.elapsed_seconds;
      result.clock_absolute_error_estimate_seconds=clock.estimated_absolute_error_seconds;result.bracket_relative_width=std::expm1(high-low);
      if(result.bracket_relative_width<=epoch_relative_ && std::abs(clock.elapsed_seconds-tau)<=absolute_+relative_*std::abs(tau)+clock.estimated_absolute_error_seconds)return result;
      if(clock.elapsed_seconds<tau)low=middle;else high=middle;
    }
    throw std::runtime_error("Canonical inverse clock fails its declared epoch/time budgets");
  }
  const std::string &asset_sha256()const{return asset_;}
  const std::string &background_sha256()const{return background_;}
  double initial_scale_factor()const{return begin_;}
  double final_scale_factor()const{return end_;}
};
}
#endif
