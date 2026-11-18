/* Measured native wake state response. No production force-equivalence claim.
 * Convex feasibility is checked in all ten original scaled state coordinates.
 * Iteration exhaustion rejects the query; no bounding-box extrapolation.
 */
#ifndef RESEARCH_SIDM_DRAG_STATE_RESPONSE_H
#define RESEARCH_SIDM_DRAG_STATE_RESPONSE_H
#include "research_sidm_drag_response.h"
#include "research_asset_sha256.h"
#include <limits>
#include <functional>
namespace research_sidm_drag_response {
class state_response {
  std::string signature,response_identity;
  std::uint32_t table_checksum=0;
  bool verified_table_sha=false;
  std::array<double,10> center{},scale{};
  std::vector<std::array<double,10>> states;
  std::vector<double> reference_errors;
  std::vector<int> active;
  std::vector<std::array<double,3>> coefficients;
  double tolerance=0,heldout_error=0,lipschitz=0;
  std::size_t iterations=0;
  static double number(std::istringstream&in){double x;if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("Incomplete or nonfinite native wake state table");return x;}
public:
  state_response(const std::string&path,std::size_t maximum_bytes,std::size_t maximum_states,
      std::size_t maximum_iterations,std::uint32_t expected_crc,const std::string&expected_sha,const std::string&expected_signature,
      const std::string&expected_table_sha=""){
    if(maximum_bytes<128||maximum_states<3||maximum_iterations<1||!sha256_token(expected_sha)||!sha256_token(expected_signature))
      throw std::runtime_error("Invalid bounded native wake state intake");
    std::ifstream file(path,std::ios::binary|std::ios::ate);if(!file)throw std::runtime_error("Missing native wake state table");
    auto length=file.tellg();if(length<128||static_cast<unsigned long long>(length)>maximum_bytes)throw std::runtime_error("Native wake state table byte bound exceeded");
    std::string bytes(static_cast<std::size_t>(length),'\0');file.seekg(0);
    if(!file.read(&bytes[0],static_cast<std::streamsize>(bytes.size()))||crc32(bytes)!=expected_crc)throw std::runtime_error("Native wake state table checksum differs");
    if(!expected_table_sha.empty()&&(!sha256_token(expected_table_sha)||research_asset_sha256::bytes(bytes)!=expected_table_sha))
      throw std::runtime_error("Native wake state table SHA256 bytes differ from pinned intake");
    verified_table_sha=!expected_table_sha.empty();
    std::istringstream in(bytes);std::string magic,sha;std::size_t n=0,k=0;
    if(!(in>>magic>>sha>>signature>>n>>k>>iterations)||magic!="RESEARCH2_WAKE_STATE_RESPONSE_V1"||sha!=expected_sha||signature!=expected_signature||
        n<3||n>maximum_states||n>bytes.size()/16||k<1||k>10||n<=k+1||iterations<1||iterations>maximum_iterations)
      throw std::runtime_error("Native wake state identity/count/work mismatch");
    response_identity=sha;table_checksum=expected_crc;
    tolerance=number(in);heldout_error=number(in);
    if(tolerance<=0||tolerance>1e-6||heldout_error<0)throw std::runtime_error("Invalid native wake state residual budget");
    for(double&x:center)x=number(in);
    for(double&x:scale){x=number(in);if(x<=0)throw std::runtime_error("Nonpositive native wake state coordinate scale");}
    active.resize(k);std::array<bool,10> used{};
    for(int&x:active){if(!(in>>x)||x<0||x>=10||used[x])throw std::runtime_error("Invalid repeated native wake state fit coordinate");used[x]=true;}
    coefficients.resize(k+1);for(auto&row:coefficients)for(double&x:row)x=number(in);
    states.resize(n);reference_errors.resize(n);
    for(std::size_t i=0;i<n;i++){
      for(double&x:states[i]){x=number(in);lipschitz+=x*x;}
      reference_errors[i]=number(in);if(reference_errors[i]<0)throw std::runtime_error("Negative native reference force error");
    }
    if(!std::isfinite(lipschitz)||lipschitz<=0)throw std::runtime_error("Native state domain has unsupported numeric conditioning");
    std::string extra;if(in>>extra)throw std::runtime_error("Unexpected trailing native wake state table payload");
  }
  struct evaluation {std::array<double,3> acceleration{};double reference_error=0,heldout_error=0,domain_residual=0;std::size_t iterations=0;};
  const std::string& product_identity()const{return response_identity;}
  const std::string& fixed_signature()const{return signature;}
  std::uint32_t checksum()const{return table_checksum;}
  bool sha256_verified()const{return verified_table_sha;}
  evaluation evaluate(const std::array<double,10>&physical,const std::string&actual_signature)const{
    if(actual_signature!=signature)throw std::runtime_error("Native wake state query changes fixed physical definition");
    for(double x:physical)if(!std::isfinite(x))throw std::runtime_error("Nonfinite physical wake state query");
    double speed=std::hypot(std::hypot(physical[0],physical[1]),physical[2]);
    if(physical[3]<=0||!std::isfinite(speed)||speed<=0)throw std::runtime_error("Nonpositive density or unsupported bath-relative speed");
    // Bounded Jacobi diagonalization avoids weak determinant-only PSD gates.
    double cscale=1;for(int i=4;i<10;i++)cscale=std::max(cscale,std::abs(physical[i]));
    double a=physical[4]/cscale,b=physical[5]/cscale,c=physical[6]/cscale,
      d=physical[7]/cscale,e=physical[8]/cscale,f=physical[9]/cscale;
    double matrix[3][3]={{a,d,e},{d,b,f},{e,f,c}};bool diagonalized=false;
    for(int sweep=0;sweep<64;sweep++){
      int p=0,q=1;double off=std::abs(matrix[0][1]);
      for(int i=0;i<3;i++)for(int j=i+1;j<3;j++)if(std::abs(matrix[i][j])>off){off=std::abs(matrix[i][j]);p=i;q=j;}
      if(off<=1e-15){diagonalized=true;break;}
      double angle=.5*std::atan2(2*matrix[p][q],matrix[q][q]-matrix[p][p]),cs=std::cos(angle),sn=std::sin(angle);
      double pp=matrix[p][p],qq=matrix[q][q],pq=matrix[p][q];
      matrix[p][p]=cs*cs*pp-2*cs*sn*pq+sn*sn*qq;matrix[q][q]=sn*sn*pp+2*cs*sn*pq+cs*cs*qq;matrix[p][q]=matrix[q][p]=0;
      for(int i=0;i<3;i++)if(i!=p&&i!=q){double ip=matrix[i][p],iq=matrix[i][q];
        matrix[i][p]=matrix[p][i]=cs*ip-sn*iq;matrix[i][q]=matrix[q][i]=sn*ip+cs*iq;}
    }
    if(!diagonalized||std::min(matrix[0][0],std::min(matrix[1][1],matrix[2][2]))<-1e-12)throw std::runtime_error("Unsupported physical bath covariance");
    std::array<double,10> query{};for(int j=0;j<10;j++){query[j]=(physical[j]-center[j])/scale[j];if(!std::isfinite(query[j]))throw std::runtime_error("Scaled native state query overflows");}
    std::vector<double> weights(states.size(),1./states.size()),trial(states.size()),ordered(states.size());
    evaluation out{};bool supported=false;
    for(std::size_t step=0;step<=iterations;step++){
      std::array<double,10> residual{};for(int j=0;j<10;j++)residual[j]=-query[j];
      double sum=0;for(std::size_t i=0;i<states.size();i++){sum+=weights[i];for(int j=0;j<10;j++)residual[j]+=weights[i]*states[i][j];}
      double error=std::abs(sum-1.);for(double x:residual)error=std::max(error,std::abs(x));
      if(!std::isfinite(error))throw std::runtime_error("Native state domain solve overflows");
      if(error<=tolerance){out.domain_residual=error;out.iterations=step;supported=true;break;}
      if(step==iterations)break;
      for(std::size_t i=0;i<states.size();i++){
        double gradient=0;for(int j=0;j<10;j++)gradient+=states[i][j]*residual[j];
        trial[i]=weights[i]-gradient/lipschitz;if(!std::isfinite(trial[i]))throw std::runtime_error("Native convex-weight update overflows");
      }
      ordered=trial;std::sort(ordered.begin(),ordered.end(),std::greater<double>());
      double prefix=0,theta=0;bool threshold=false;
      for(std::size_t i=0;i<ordered.size();i++){prefix+=ordered[i];double candidate=(prefix-1.)/(i+1);
        if(ordered[i]>candidate){theta=candidate;threshold=true;}}
      if(!threshold||!std::isfinite(theta))throw std::runtime_error("Native simplex projection is unsupported");
      sum=0;for(std::size_t i=0;i<weights.size();i++){weights[i]=std::max(0.,trial[i]-theta);sum+=weights[i];}
      if(!std::isfinite(sum)||sum<=0)throw std::runtime_error("Native convex weights have no positive normalization");
      for(double&w:weights)w/=sum;
    }
    if(!supported)throw std::runtime_error("Native state is outside measured convex support or iteration budget exhausted");
    out.acceleration=coefficients[0];
    for(std::size_t i=0;i<active.size();i++)for(int j=0;j<3;j++)out.acceleration[j]+=query[active[i]]*coefficients[i+1][j];
    for(std::size_t i=0;i<weights.size();i++)out.reference_error+=weights[i]*reference_errors[i];out.heldout_error=heldout_error;
    for(double x:out.acceleration)if(!std::isfinite(x))throw std::runtime_error("Native affine force response overflows");
    if(!std::isfinite(out.reference_error))throw std::runtime_error("Native reference allowance overflows");
    return out;
  }
};
}
#endif
