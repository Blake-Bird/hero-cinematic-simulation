// Named write-only native integrator diagnostic. No struct padding is serialized.
#ifndef RESEARCH_RESTART_INTEGRATOR_CAPSULE_H
#define RESEARCH_RESTART_INTEGRATOR_CAPSULE_H
#include "gadgetconfig.h"
#include <cstdint>
#include <cstring>
#include "../data/particle_data.h"
namespace research_restart_integrator {
static const char layout_sha256[]="b85cc6fa256d4cd355caa19b744d8a9520af584c6b3a344a18271c4decb1e191";
static const int floating_columns=0
#define RESEARCH_INTEGRATOR_FLOAT(name,value) +1
#include "research_restart_integrator_float.inc"
#undef RESEARCH_INTEGRATOR_FLOAT
;
static const int integer_columns=9;
static const int coordinate_lanes=12;
static const int storage_lanes=coordinate_lanes+2*integer_columns;
static const char active_float_columns[]=
#define RESEARCH_INTEGRATOR_FLOAT(name,value) name ";"
#include "research_restart_integrator_float.inc"
#undef RESEARCH_INTEGRATOR_FLOAT
"";
inline void store_lane(uint32_t value,int *target) {
  static_assert(sizeof(int)==sizeof(uint32_t),"Native diagnostic lane needs int32");
  std::memcpy(target,&value,sizeof(value));
}
inline void store_uint64(uint64_t value,int *target) {
  store_lane(static_cast<uint32_t>(value),target);
  store_lane(static_cast<uint32_t>(value>>32),target+1);
}
inline void pack_floating(particle_data &p,double *output) {
  int cursor=0;
#define RESEARCH_INTEGRATOR_FLOAT(name,value) output[cursor++]=static_cast<double>(value);
#include "research_restart_integrator_float.inc"
#undef RESEARCH_INTEGRATOR_FLOAT
}
inline void pack_integer(particle_data &p,int *output) {
  static_assert(sizeof(MyIntPosType)*8==BITS_FOR_POSITIONS,"Integer coordinate storage changed");
  static_assert(BITS_FOR_POSITIONS==32 || BITS_FOR_POSITIONS==64 || BITS_FOR_POSITIONS==128,"Unsupported coordinate width");
  for(int axis=0;axis<3;axis++) {
    MyIntPosType value=p.IntPos[axis];
    for(int lane=0;lane<4;lane++) {
      store_lane(static_cast<uint32_t>(value),output+4*axis+lane);
#if BITS_FOR_POSITIONS>32
      value>>=32;
#else
      value=0;
#endif
    }
  }
  uint64_t values[integer_columns]={
    static_cast<uint64_t>(static_cast<int64_t>(p.get_Ti_Current())),
    static_cast<uint64_t>(static_cast<int64_t>(p.TimeBinGrav)),
#ifndef LEAN
    static_cast<uint64_t>(static_cast<int64_t>(p.TimeBinHydro)),
#else
    0,
#endif
    static_cast<uint64_t>(p.getType()),static_cast<uint64_t>(p.getSofteningClass()),
    static_cast<uint64_t>(static_cast<int64_t>(p.GravCost)),
#ifdef REARRANGE_OPTION
    static_cast<uint64_t>(p.TreeID),
#else
    0,
#endif
#if defined(PMGRID) && defined(PLACEHIGHRESREGION)
    static_cast<uint64_t>(p.InsideOutsideFlag),
#else
    0,
#endif
#if defined(FORCETEST) && defined(FORCETEST_FIXEDPARTICLESET)
    static_cast<uint64_t>(p.SelectedFlag)
#else
    0
#endif
  };
  for(int index=0;index<integer_columns;index++)store_uint64(values[index],output+coordinate_lanes+2*index);
}
} // namespace research_restart_integrator
#endif
