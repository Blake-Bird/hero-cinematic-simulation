/* Source-bound physical context intake. The launch layer pins the context
 * SHA256; native intake verifies both context and table bytes plus CRCs.
 */
#ifndef RESEARCH_SIDM_DRAG_CONTEXT_H
#define RESEARCH_SIDM_DRAG_CONTEXT_H
#include "research_sidm_drag_driver.h"
#include <memory>
namespace research_sidm_drag_context {
struct launch_settings {
  std::string context_path,context_sha256,table_path;
  std::uint32_t context_crc32=0;
  std::size_t maximum_context_bytes=0,maximum_table_bytes=0,maximum_states=0,maximum_domain_iterations=0;
  std::uint64_t maximum_schedule_workspace_bytes=0;
};
class intake {
  std::string product,signature,physics,table_sha;
  std::uint64_t table_crc=0,carrier=0,seed=0,maximum_particles=0,workspace=0,subcycles=0;
  std::array<double,20> physical{};
  std::array<double,11> limits{};
  static std::uint64_t integer(std::istringstream&in){
    std::string token;if(!(in>>token)||token.empty()||token.size()>20)throw std::runtime_error("Incomplete native context integer");
    std::uint64_t value=0;for(char c:token){if(c<'0'||c>'9'||value>(UINT64_MAX-(c-'0'))/10)throw std::runtime_error("Invalid native context uint64");value=value*10+(c-'0');}return value;
  }
public:
  intake(const std::string&path,std::size_t maximum_bytes,std::uint32_t expected_crc,const std::string&expected_context_sha){
    using research_sidm_drag_response::sha256_token;
    if(maximum_bytes<128)throw std::runtime_error("Native context byte bound is too small");
    std::ifstream file(path,std::ios::binary|std::ios::ate);if(!file)throw std::runtime_error("Native source context is missing");
    auto length=file.tellg();if(length<128||static_cast<unsigned long long>(length)>maximum_bytes)throw std::runtime_error("Native source context byte bound exceeded");
    std::string bytes(static_cast<std::size_t>(length),'\0');file.seekg(0);
    if(!file.read(&bytes[0],static_cast<std::streamsize>(bytes.size()))||research_sidm_drag_response::crc32(bytes)!=expected_crc)throw std::runtime_error("Native context checksum differs");
    if(!sha256_token(expected_context_sha)||research_asset_sha256::bytes(bytes)!=expected_context_sha)throw std::runtime_error("Native context SHA256 differs from pinned launch intake");
    std::istringstream in(bytes);std::string magic;
    if(!(in>>magic>>product>>signature>>physics>>table_sha)||magic!="RESEARCH2_WAKE_CONTEXT_V1"||
        !sha256_token(product)||!sha256_token(signature)||!sha256_token(physics)||!sha256_token(table_sha))throw std::runtime_error("Invalid native context model identities");
    table_crc=integer(in);carrier=integer(in);seed=integer(in);maximum_particles=integer(in);workspace=integer(in);subcycles=integer(in);
    if(table_crc>UINT32_MAX||carrier==0||maximum_particles<3||workspace<1024||subcycles<1||subcycles>4096)throw std::runtime_error("Native context bounds/identity are invalid");
    for(double&x:physical)if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("Incomplete/nonfinite native physical context");
    for(double&x:limits)if(!(in>>x)||!std::isfinite(x)||x<0)throw std::runtime_error("Incomplete/nonfinite native source limits");
    if(physical[0]<0||physical[4]<=0||physical[5]<=0||physical[9]<=0||physical[16]<0||physical[17]<=physical[16]||physical[18]<=0||physical[19]<=0||physical[17]>=physical[19]/2)
      throw std::runtime_error("Native physical mass/density/aperture context is unsupported");
    if(limits[0]<=0||limits[3]<=0||limits[3]>=1||limits[4]<=0||limits[4]>=1||limits[5]<=0||limits[5]>=1||limits[6]<=0||limits[7]<=0||limits[8]<=0||limits[10]>1e-6)
      throw std::runtime_error("Native source numerical context is unsupported");
    std::string extra;if(in>>extra)throw std::runtime_error("Unexpected trailing native source context");
  }
  std::uint64_t maximum_subcycles()const{return subcycles;}
  const std::string& table_sha256()const{return table_sha;}
  std::uint32_t table_checksum()const{return static_cast<std::uint32_t>(table_crc);}
  const std::string& response_identity()const{return product;}
  const std::string& fixed_signature()const{return signature;}
  research_sidm_drag_driver::context bind(const research_sidm_drag_response::state_response&response,
      const std::string&actual_physics_signature,double actual_particle_epoch,double native_begin,double native_end,
      double proper_begin,double proper_end,long long timeline,double mass_to_msun,double velocity_to_km_s,
      double length_to_proper_kpc,double erg_per_msun_km2_s2)const{
    if(actual_physics_signature!=physics||actual_particle_epoch!=physical[0]||response.product_identity()!=product||
        response.fixed_signature()!=signature||response.checksum()!=table_crc)throw std::runtime_error("Actual native model/epoch differs from source-bound context");
    research_sidm_drag_driver::context out{};out.actual_fixed_signature=signature;out.carrier_id=carrier;out.seed=seed;out.native_timeline=timeline;
    out.native_begin=native_begin;out.native_end=native_end;out.proper_begin_seconds=proper_begin;out.proper_end_seconds=proper_end;
    out.native_mass_to_msun=mass_to_msun;out.native_velocity_to_km_s=velocity_to_km_s;out.native_length_to_proper_kpc=length_to_proper_kpc;out.erg_per_msun_km2_s2=erg_per_msun_km2_s2;
    for(int i=0;i<3;i++){out.expected_probe_position_proper_kpc[i]=physical[i+1];out.background_mean_velocity_km_s[i]=physical[i+6];}
    out.expected_physical_bh_mass_msun=physical[4];out.expected_gravity_carrier_mass_msun=physical[5];out.background_density_msun_kpc3=physical[9];
    for(int i=0;i<6;i++)out.background_covariance_km2_s2[i]=physical[i+10];
    out.recipient_inner_radius_proper_kpc=physical[16];out.recipient_outer_radius_proper_kpc=physical[17];out.recipient_gravity_mass_msun=physical[18];out.proper_box_kpc=physical[19];
    out.limits.maximum_proper_step_seconds=limits[0];out.limits.maximum_reference_error_cm_s2=limits[1];out.limits.maximum_heldout_error_cm_s2=limits[2];
    out.limits.conservative.fraction=limits[3];out.limits.conservative.inertia_ratio=limits[4];out.limits.conservative.relative_tolerance=limits[5];
    out.limits.conservative.momentum_floor=limits[6];out.limits.conservative.energy_floor=limits[7];out.limits.conservative.angular_floor=limits[8];
    out.proper_clock_absolute_tolerance_seconds=limits[9];out.proper_clock_relative_tolerance=limits[10];
    out.limits.conservative.maximum_particles=maximum_particles;out.limits.conservative.maximum_workspace_bytes=workspace;return out;
  }
};
class asset_bundle {
  intake binding;
  research_sidm_drag_response::state_response model;
public:
  asset_bundle(const std::string&context_path,std::size_t maximum_context_bytes,std::uint32_t context_crc,const std::string&context_sha,
      const std::string&table_path,std::size_t maximum_table_bytes,std::size_t maximum_states,std::size_t maximum_iterations)
    :binding(context_path,maximum_context_bytes,context_crc,context_sha),
     model(table_path,maximum_table_bytes,maximum_states,maximum_iterations,binding.table_checksum(),binding.response_identity(),binding.fixed_signature(),binding.table_sha256()){}
  const intake& context_binding()const{return binding;}
  const research_sidm_drag_response::state_response& response()const{return model;}
};
inline std::unique_ptr<asset_bundle> load_collective(const launch_settings&settings,MPI_Comm comm){
  using research_sidm_drag_source::require_collective;
  research_sidm_drag_coupling::same_identity(settings.context_sha256,comm);
  std::array<unsigned long long,6> bounds{{settings.context_crc32,settings.maximum_context_bytes,settings.maximum_table_bytes,
    settings.maximum_states,settings.maximum_domain_iterations,settings.maximum_schedule_workspace_bytes}},low{},high{};
  MPI_Allreduce(bounds.data(),low.data(),6,MPI_UNSIGNED_LONG_LONG,MPI_MIN,comm);
  MPI_Allreduce(bounds.data(),high.data(),6,MPI_UNSIGNED_LONG_LONG,MPI_MAX,comm);
  require_collective(low==high&&settings.maximum_context_bytes>=128&&settings.maximum_table_bytes>=128&&
    settings.maximum_states>=3&&settings.maximum_states<=4096&&settings.maximum_domain_iterations>=1&&
    settings.maximum_domain_iterations<=1000000&&settings.maximum_schedule_workspace_bytes>=1024,
    comm,"MPI calibrated launch bounds disagree or exceed the supported asset domain");
  std::unique_ptr<asset_bundle> assets;bool okay=true;
  try{assets.reset(new asset_bundle(settings.context_path,settings.maximum_context_bytes,settings.context_crc32,settings.context_sha256,
    settings.table_path,settings.maximum_table_bytes,settings.maximum_states,settings.maximum_domain_iterations));}
  catch(const std::exception&){okay=false;}
  require_collective(okay,comm,"An MPI owner cannot load the pinned calibrated context/table assets");
  research_sidm_drag_coupling::same_identity(assets->response().product_identity(),comm);
  research_sidm_drag_coupling::same_identity(assets->response().fixed_signature(),comm);
  return assets;
}
}
#endif
