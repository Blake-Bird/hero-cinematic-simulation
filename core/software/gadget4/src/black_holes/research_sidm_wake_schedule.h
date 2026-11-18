/* Stateless immutable diagnostic scheduling. No drag source is enabled. */
#ifndef RESEARCH_SIDM_WAKE_SCHEDULE_H
#define RESEARCH_SIDM_WAKE_SCHEDULE_H
#include "research_asset_sha256.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace research_sidm_wake_schedule {
struct row {double native_time;std::uint64_t carrier_id,measurement_id;std::vector<double> edges;};
struct schedule {
  std::string physics_signature;double epoch_tolerance=0;
  std::uint64_t maximum_particles=0,maximum_work=0,workspace=0,publication=0;
  std::vector<row> rows;
};
inline std::uint64_t integer(std::istringstream&in){
  std::string token;if(!(in>>token)||token.empty()||token.size()>20)throw std::runtime_error("Incomplete wake schedule integer");
  std::uint64_t value=0;for(char c:token){if(c<'0'||c>'9'||value>(UINT64_MAX-(c-'0'))/10)throw std::runtime_error("Invalid wake schedule uint64");value=value*10+(c-'0');}return value;
}
inline bool identity(const std::string&value){if(value.size()!=64)return false;for(char c:value)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;return true;}
inline schedule read(const std::string&path,const std::string&expected_sha,std::size_t maximum_bytes,std::size_t maximum_rows,double memory_bytes){
  if(!identity(expected_sha)||maximum_bytes<1||maximum_rows<1||!std::isfinite(memory_bytes)||memory_bytes<=0)
    throw std::runtime_error("Invalid wake schedule asset/allocation");
  std::ifstream file(path,std::ios::binary|std::ios::ate);if(!file)throw std::runtime_error("Native wake schedule missing");
  const auto length=file.tellg();if(length<1||static_cast<unsigned long long>(length)>maximum_bytes)
    throw std::runtime_error("Native wake schedule byte limit exceeded");
  std::string bytes(std::size_t(length),'\0');file.seekg(0);
  if(!file.read(&bytes[0],std::streamsize(bytes.size()))||research_asset_sha256::bytes(bytes)!=expected_sha)
    throw std::runtime_error("Native wake schedule SHA differs from its compiled identity");
  std::istringstream input(bytes);std::string magic;schedule result;
  if(!(input>>magic>>result.physics_signature)||magic!="RESEARCH2_WAKE_SCHEDULE_V1"||!identity(result.physics_signature))
    throw std::runtime_error("Invalid native wake schedule physics identity");
  result.maximum_particles=integer(input);result.maximum_work=integer(input);result.workspace=integer(input);result.publication=integer(input);
  if(!(input>>result.epoch_tolerance)||!std::isfinite(result.epoch_tolerance)||result.epoch_tolerance<0||result.maximum_particles<1||result.maximum_work<1||result.workspace<1024||result.publication<8192)
    throw std::runtime_error("Native wake schedule source limits invalid");
  const auto count=integer(input);
  if(count<1||count>maximum_rows||count>bytes.size()/8||16.L*bytes.size()+256.L*count>memory_bytes/8)
    throw std::runtime_error("Native wake schedule row/workspace allocation exceeded");
  result.rows.reserve(count);
  for(std::uint64_t i=0;i<count;i++){row value;
    if(!(input>>value.native_time)||!std::isfinite(value.native_time)||value.native_time<0)
      throw std::runtime_error("Invalid native wake scheduled epoch");
    value.carrier_id=integer(input);value.measurement_id=integer(input);const auto edges=integer(input);
    if(!value.carrier_id||!value.measurement_id||edges<2||edges>4096||8192.L+2048.L*edges>result.publication)
      throw std::runtime_error("Native wake scheduled identity/shell/publication support invalid");
    value.edges.resize(edges);for(std::size_t j=0;j<edges;j++)if(!(input>>value.edges[j])||!std::isfinite(value.edges[j])||value.edges[j]<0||(j&&value.edges[j]<=value.edges[j-1]))
      throw std::runtime_error("Native wake scheduled shell edges invalid");
    if(i&&value.native_time-result.rows.back().native_time<=2*result.epoch_tolerance)
      throw std::runtime_error("Wake scheduled epochs must be increasing and distinguishable");
    result.rows.push_back(value);
  }
  std::string extra;if(input>>extra)throw std::runtime_error("Unexpected trailing wake schedule fields");return result;
}
}
#endif
