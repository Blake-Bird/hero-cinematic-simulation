/* Explicit mapped-frame snapshot lineage. Ordinary builds must reject these
 * ICs rather than evolving their canonical velocities with a physical law. */
#ifndef RESEARCH_SIDM_CANONICAL_SNAPSHOT_H
#define RESEARCH_SIDM_CANONICAL_SNAPSHOT_H
#include <hdf5.h>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace research_sidm {
inline std::string canonical_header_hash(hid_t header,const char *name)
{
  const hid_t attribute=H5Aopen(header,name,H5P_DEFAULT);
  if(attribute<0)throw std::runtime_error("Canonical snapshot lineage hash missing");
  const hid_t type=H5Aget_type(attribute),space=H5Aget_space(attribute);
  if(type<0 || space<0 || H5Tget_class(type)!=H5T_STRING || H5Sget_simple_extent_npoints(space)!=1) {
    if(type>=0)H5Tclose(type);if(space>=0)H5Sclose(space);H5Aclose(attribute);
    throw std::runtime_error("Canonical snapshot lineage needs scalar text hashes");
  }
  std::string result;bool valid=true;
  if(H5Tis_variable_str(type)>0) {
    char *value=nullptr;
    if(H5Aread(attribute,type,&value)<0 || !value)valid=false;
    else {
      // h5py writes variable-length scalar strings in the original IC.
      // Read that representation and release HDF5-owned memory immediately.
      std::size_t length=0;while(length<65 && value[length])++length;
      if(length!=64)valid=false;else result.assign(value,length);
    }
    if(value)H5free_memory(value);
  } else {
    const std::size_t length=H5Tget_size(type);
    std::array<char,66> value{};
    if(length<64 || length>65 || H5Aread(attribute,type,value.data())<0)valid=false;
    else {
      result.assign(value.data(),64);
      if(length==65 && value[64]!='\0')valid=false;
    }
  }
  H5Tclose(type);H5Sclose(space);H5Aclose(attribute);
  for(char c:result)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))valid=false;
  if(!valid || result.size()!=64)throw std::runtime_error("Canonical snapshot lineage hash invalid");
  return result;
}
inline void require_canonical_snapshot(hid_t header,const char *clock_sha,const char *model_sha,
    const char *physical_state_sha,double initial_a)
{
  const hid_t attribute=H5Aopen(header,"Research_sidm_canonical_control_schema",H5P_DEFAULT);
  if(attribute<0)throw std::runtime_error("Canonical native build requires an explicitly mapped IC/snapshot");
  const hid_t type=H5Aget_type(attribute),space=H5Aget_space(attribute);
  unsigned long long version=0;
  const bool valid=type>=0 && space>=0 && H5Tget_class(type)==H5T_INTEGER
      && H5Sget_simple_extent_npoints(space)==1 && H5Aread(attribute,H5T_NATIVE_ULLONG,&version)>=0;
  if(type>=0)H5Tclose(type);if(space>=0)H5Sclose(space);H5Aclose(attribute);
  if(!valid || version!=1)throw std::runtime_error("Canonical snapshot frame schema invalid");
  if(canonical_header_hash(header,"Research_sidm_canonical_clock_SHA256")!=clock_sha
      || canonical_header_hash(header,"Research_sidm_physical_model_hash")!=model_sha
      || canonical_header_hash(header,"Research_sidm_original_physical_state_SHA256")!=physical_state_sha)
    throw std::runtime_error("Canonical snapshot lineage differs from its native launch parameters");
  const hid_t epoch=H5Aopen(header,"Research_sidm_canonical_initial_scale_factor",H5P_DEFAULT);
  if(epoch<0)throw std::runtime_error("Canonical snapshot initial physical epoch missing");
  const hid_t epoch_type=H5Aget_type(epoch),epoch_space=H5Aget_space(epoch);double value=0;
  const bool epoch_valid=epoch_type>=0 && epoch_space>=0 && H5Tget_class(epoch_type)==H5T_FLOAT
      && H5Sget_simple_extent_npoints(epoch_space)==1 && H5Aread(epoch,H5T_NATIVE_DOUBLE,&value)>=0;
  if(epoch_type>=0)H5Tclose(epoch_type);if(epoch_space>=0)H5Sclose(epoch_space);H5Aclose(epoch);
  if(!epoch_valid || !std::isfinite(initial_a) || initial_a<=0 || value!=initial_a)
    throw std::runtime_error("Canonical snapshot initial physical epoch differs from native launch");
}
}
#endif
