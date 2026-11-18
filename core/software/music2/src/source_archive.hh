// Immutable on-disk numerical inputs for independent hierarchy comparisons.
// Values retain native computational units and actual ghost cells. This
// archive does not assert point/cell-average semantics or science acceptance.
#pragma once
#include "mesh.hh"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <limits>

namespace music_source_archive {
template<typename Hierarchy>uint64_t estimated_bytes(const Hierarchy& grid,unsigned halo,unsigned potential_scalar_bytes=8){
 if(potential_scalar_bytes!=8&&potential_scalar_bytes!=16)throw std::runtime_error("Unsupported native archived potential scalar width");
 long double total=24+14*28;
 for(unsigned level=grid.levelmin();level<=grid.levelmax();++level){
  long double cells=1,ghost_cells=1;
  for(int a=0;a<3;++a){cells*=grid.size(level,a);ghost_cells*=grid.size(level,a)+2*halo;}
  total+=14*52+8*8*cells+potential_scalar_bytes*6*ghost_cells;
 }
 if(!std::isfinite(total)||total>std::numeric_limits<uint64_t>::max())
  throw std::runtime_error("Native source archive estimate overflow");
 return uint64_t(total);
}
class Writer {
 std::FILE *file_=nullptr;uint64_t written_=0,limit_;bool wide_format_;
 void bytes(const void *data,size_t count){
  if(count>limit_-written_)throw std::runtime_error("Native source archive exceeds declared disk budget");
  if(std::fwrite(data,1,count,file_)!=count)throw std::runtime_error("Native source archive write failed");
  written_+=count;
 }
 template<typename T>void scalar(T value){bytes(&value,sizeof(value));}
public:
 Writer(const char *filename,uint64_t limit,bool wide_format=false):limit_(limit),wide_format_(wide_format){
  static_assert(sizeof(double)==8&&std::numeric_limits<double>::is_iec559,"IEEE binary64 source archive required");
  if(limit<24)throw std::runtime_error("Invalid native source archive disk allocation");
  file_=std::fopen(filename,"wbx");
  if(!file_)throw std::runtime_error("Refusing existing or unwritable native source archive");
  try{bytes(wide_format_?"MUSICS02":"MUSICS01",8);scalar(uint32_t(0x01020304));scalar(uint32_t(wide_format_?2:1));}
  catch(...){std::fclose(file_);file_=nullptr;throw;}
 }
 Writer(const Writer&)=delete;Writer& operator=(const Writer&)=delete;
 ~Writer(){if(file_)std::fclose(file_);}
 // Quantities: 1 matter source, 2 linear potential, 3 quadratic source,
 // 4 quadratic potential, 5 compensated relative density, 6 velocity source,
 // 7 velocity potential. Species follows the actual gas/DM preparation pass.
 void field(uint32_t quantity,uint32_t species,const grid_hierarchy& field,unsigned halo){
  if(!file_||quantity<1||quantity>7||species>1||halo>16||field.levelmax()>30)
   throw std::runtime_error("Invalid native source archive field request");
  bytes("FIELD001",8);scalar(quantity);scalar(species);
  scalar(uint32_t(field.levelmin()));scalar(uint32_t(field.levelmax()));scalar(uint32_t(halo));
  for(unsigned level=field.levelmin();level<=field.levelmax();++level){
   auto &grid=*field.get_grid(level);scalar(uint32_t(level));
   for(int a=0;a<3;++a)scalar(uint64_t(grid.size(a)));
   for(int a=0;a<3;++a)scalar(int64_t(field.offset_abs(level,a)));
   double buffer[8192];size_t used=0;
   for(int64_t i=-int64_t(halo);i<int64_t(grid.size(0))+halo;++i)
    for(int64_t j=-int64_t(halo);j<int64_t(grid.size(1))+halo;++j)
     for(int64_t k=-int64_t(halo);k<int64_t(grid.size(2))+halo;++k){
      double value=grid(i,j,k);
      if(!std::isfinite(value))throw std::runtime_error("Nonfinite native archived source field");
      buffer[used++]=value;
      if(used==8192){bytes(buffer,sizeof(buffer));used=0;}
     }
   if(used)bytes(buffer,used*sizeof(double));
  }
 }
#ifdef MUSIC_WIDE_MIXED_POTENTIAL
 // FIELD128 stores the original IEEE754 binary128 samples, including ghosts.
 // Never pass through double or platform-dependent long double serialization.
 void field_wide(uint32_t quantity,uint32_t species,const GridHierarchy<__float128>& field,unsigned halo){
  static_assert(sizeof(__float128)==16 && __FLT128_MANT_DIG__==113 && __FLT128_MAX_EXP__==16384,
                "IEEE binary128 source archive required");
  if(!file_||!wide_format_||(quantity!=2&&quantity!=4&&quantity!=7)||species>1||halo!=4||field.levelmax()>30)
   throw std::runtime_error("Invalid wide native source archive field request");
  bytes("FIELD128",8);scalar(quantity);scalar(species);
  scalar(uint32_t(field.levelmin()));scalar(uint32_t(field.levelmax()));scalar(uint32_t(halo));
  for(unsigned level=field.levelmin();level<=field.levelmax();++level){
   auto &grid=*field.get_grid(level);scalar(uint32_t(level));
   for(int a=0;a<3;++a)scalar(uint64_t(grid.size(a)));
   for(int a=0;a<3;++a)scalar(int64_t(field.offset_abs(level,a)));
   __float128 buffer[4096];size_t used=0;
   for(int64_t i=-int64_t(halo);i<int64_t(grid.size(0))+halo;++i)
    for(int64_t j=-int64_t(halo);j<int64_t(grid.size(1))+halo;++j)
     for(int64_t k=-int64_t(halo);k<int64_t(grid.size(2))+halo;++k){
      const __float128 value=grid(i,j,k);
      if(!__builtin_isfinite(value))throw std::runtime_error("Nonfinite native archived wide potential");
      buffer[used++]=value;
      if(used==4096){bytes(buffer,sizeof(buffer));used=0;}
     }
   if(used)bytes(buffer,used*sizeof(__float128));
  }
 }
#endif
 void finish(){
  if(!file_)throw std::runtime_error("Native source archive is already closed");
  bytes("END00001",8);
  const int flush=std::fflush(file_),closed=std::fclose(file_);file_=nullptr;
  if(flush||closed)throw std::runtime_error("Native source archive completion failed");
 }
};
}
