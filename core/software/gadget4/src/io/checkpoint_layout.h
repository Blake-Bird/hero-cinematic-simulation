/* Versioned raw restart ABI guard. Check before restoring All or allocating P.
 * Same-sized source/configuration changes are rejected by the generated digest. */
#ifndef RESEARCH_CHECKPOINT_LAYOUT_H
#define RESEARCH_CHECKPOINT_LAYOUT_H
#include "gadgetconfig.h"
#include <stdint.h>
#include <string.h>
struct native_checkpoint_layout
{
  char magic[8];
  uint64_t version, global_bytes, particle_bytes, sph_bytes, ranks, endian, compiler;
  char source_config_sha256[64];
};
static_assert(sizeof(native_checkpoint_layout) == 128, "Unexpected checkpoint header layout");
inline native_checkpoint_layout native_checkpoint_layout_make(uint64_t global_bytes,
  uint64_t particle_bytes, uint64_t sph_bytes, uint64_t ranks, const char *fingerprint)
{
  native_checkpoint_layout result = {};
  memcpy(result.magic, "R2RST001", 8);
  result.version = 1;
  result.global_bytes = global_bytes;
  result.particle_bytes = particle_bytes;
  result.sph_bytes = sph_bytes;
  result.ranks = ranks;
  result.endian = UINT64_C(0x0102030405060708);
  result.compiler = UINT64_C(14695981039346656037);
  for(const char *p = __VERSION__; *p; ++p)
    result.compiler = (result.compiler ^ (unsigned char)*p) * UINT64_C(1099511628211);
  memcpy(result.source_config_sha256, fingerprint, 64);
  return result;
}
inline bool native_checkpoint_layout_equal(const native_checkpoint_layout &saved,
                                           const native_checkpoint_layout &expected)
{
  return memcmp(&saved, &expected, sizeof(saved)) == 0;
}
#endif
