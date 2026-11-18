/* Bounded intake of a measured, conditional wake speed response.
 * This reader does not establish native force equivalence or extend the
 * recorded physical state domain. The caller binds that domain separately.
 */
#ifndef RESEARCH_SIDM_DRAG_RESPONSE_H
#define RESEARCH_SIDM_DRAG_RESPONSE_H
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace research_sidm_drag_response {
inline std::uint32_t crc32(const std::string&bytes){
  std::uint32_t crc=UINT32_C(0xffffffff);
  for(unsigned char value:bytes){crc^=value;for(int bit=0;bit<8;bit++)crc=(crc>>1)^(UINT32_C(0xedb88320)&(0-(crc&1)));}
  return crc^UINT32_C(0xffffffff);
}
inline bool sha256_token(const std::string&value){return value.size()==64&&std::all_of(value.begin(),value.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});}
class speed_response {
  std::vector<double> speed,amplitude,mean_error,reference_error;
  std::vector<int> supported;
  std::array<double,3> direction{};
  std::string response_sha,signature;
public:
  speed_response(const std::string&path,std::size_t maximum_bytes,std::size_t maximum_knots,std::uint32_t expected_crc,
      const std::string&expected_response_sha,const std::string&expected_signature){
    if(maximum_knots<3||maximum_bytes<128||!sha256_token(expected_response_sha)||!sha256_token(expected_signature))
      throw std::runtime_error("Invalid bounded wake-response intake definition");
    std::ifstream file(path,std::ios::binary|std::ios::ate);if(!file)throw std::runtime_error("Cannot open retained wake-response table");
    auto length=file.tellg();if(length<128||static_cast<unsigned long long>(length)>maximum_bytes)throw std::runtime_error("Wake-response table exceeds its byte bound or is incomplete");
    std::string bytes(static_cast<std::size_t>(length),'\0');file.seekg(0);
    if(!file.read(&bytes[0],static_cast<std::streamsize>(bytes.size()))||crc32(bytes)!=expected_crc)
      throw std::runtime_error("Wake-response table bytes differ from the pinned intake checksum");
    std::istringstream input(bytes);std::string magic;std::size_t count=0;
    if(!(input>>magic>>response_sha>>signature>>count)||magic!="RESEARCH2_WAKE_SPEED_RESPONSE_V1"||response_sha!=expected_response_sha||signature!=expected_signature||count<3||count>maximum_knots||count>bytes.size()/8)
      throw std::runtime_error("Wake-response identity, format or knot bound mismatch");
    if(!(input>>direction[0]>>direction[1]>>direction[2]))throw std::runtime_error("Wake-response direction is missing");
    double direction_norm=std::hypot(std::hypot(direction[0],direction[1]),direction[2]);
    if(!std::isfinite(direction_norm)||std::abs(direction_norm-1)>1e-12)throw std::runtime_error("Wake-response direction is unsupported");
    speed.resize(count);amplitude.resize(count);mean_error.resize(count);reference_error.resize(count);supported.resize(count-1);
    for(std::size_t row=0;row<count;row++){
      if(!(input>>speed[row]>>amplitude[row]>>mean_error[row]>>reference_error[row])||!std::isfinite(speed[row])||speed[row]<=0||!std::isfinite(amplitude[row])||amplitude[row]<0||!std::isfinite(mean_error[row])||mean_error[row]<0||!std::isfinite(reference_error[row])||reference_error[row]<0||(row&&speed[row]<=speed[row-1]))
        throw std::runtime_error("Nonfinite, negative or unordered measured wake-response knot");
    }
    for(int&value:supported)if(!(input>>value)||(value!=0&&value!=1))throw std::runtime_error("Wake-response interval support is missing or invalid");
    std::string extra;if(input>>extra)throw std::runtime_error("Unexpected trailing wake-response payload");
  }
  struct evaluation {std::array<double,3> acceleration;double mean_error,reference_error;std::size_t interval;};
  evaluation evaluate(const std::array<double,3>&relative_velocity_km_s,const std::string&actual_fixed_signature)const{
    if(actual_fixed_signature!=signature)throw std::runtime_error("Native wake-response query changes its recorded physical definition");
    double value=std::hypot(std::hypot(relative_velocity_km_s[0],relative_velocity_km_s[1]),relative_velocity_km_s[2]);
    if(!std::isfinite(value)||value<speed.front()||value>speed.back())throw std::runtime_error("Native wake-response extrapolation is unsupported");
    for(int axis=0;axis<3;axis++)if(std::abs(relative_velocity_km_s[axis]/value-direction[axis])>1e-12)throw std::runtime_error("Native wake-response query changes its measured direction");
    std::size_t right=std::upper_bound(speed.begin(),speed.end(),value)-speed.begin();
    std::size_t left=std::min(right?right-1:0,speed.size()-2);
    if(!supported[left])throw std::runtime_error("Native wake-response interval lacks held-out support");
    double w=(value-speed[left])/(speed[left+1]-speed[left]);evaluation out{};out.interval=left;
    double a=(1-w)*amplitude[left]+w*amplitude[left+1];
    out.mean_error=(1-w)*mean_error[left]+w*mean_error[left+1];out.reference_error=(1-w)*reference_error[left]+w*reference_error[left+1];
    for(int axis=0;axis<3;axis++)out.acceleration[axis]=-a*relative_velocity_km_s[axis]/value;
    return out;
  }
};
}
#endif
