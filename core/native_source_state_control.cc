#include "gadgetconfig.h"
#include "data/particle_data.h"
#include "io/checkpoint_layout.h"
#include "data/research_stellar_birth.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

// Use the exact production particle struct and the same raw byten storage used
// by restart.cc. No simulated source physics: these are serialization controls.
int main(int argc, char **argv)
{
  if(argc != 2) return 2;
  particle_data saved;
  if(saved.ResearchSourceState.sidm_collision_count || saved.ResearchSourceState.bh_event_counter)
    return 3;
  saved.ID.set(ID_MAX);
  saved.ResearchSourceState.sidm_last_update_time = 0.375;
  saved.ResearchSourceState.sidm_optical_depth = 0.125;
  saved.ResearchSourceState.sidm_collision_count = UINT64_MAX - 1;
  saved.ResearchSourceState.sidm_random_counter = UINT64_MAX;
  saved.ResearchSourceState.stellar_last_source_time = 0.25;
  saved.ResearchSourceState.stellar_returned_mass = 3.25;
  saved.ResearchSourceState.stellar_injected_energy = 9.5;
  saved.ResearchSourceState.stellar_event_counter = UINT64_C(9223372036854775809);
  saved.ResearchSourceState.stellar_initial_mass = 2.5;
  saved.ResearchSourceState.stellar_initial_locked_mass = 2.25;
  saved.ResearchSourceState.stellar_prompt_returned_mass = .25;
  saved.ResearchSourceState.stellar_birth_time = .125;
  saved.ResearchSourceState.stellar_birth_scale_factor = .125;
  saved.ResearchSourceState.stellar_birth_metallicity = .03125;
  saved.ResearchSourceState.stellar_parent_gas_id = ID_MAX;
  saved.ResearchSourceState.stellar_random_counter = UINT64_MAX - 1;
  saved.ResearchSourceState.stellar_birth_state_version = 1;
  saved.ResearchSourceState.stellar_birth_mode = 2;
  for(int i = 0; i < 11; ++i) saved.ResearchSourceState.stellar_returned_elements[i] = i + 0.5;
  saved.ResearchSourceState.bh_last_source_time = 0.3;
  saved.ResearchSourceState.bh_feedback_reservoir = 18.125;
  saved.ResearchSourceState.bh_event_counter = UINT64_MAX;
#ifdef STARFORMATION
  saved.StellarAge = 0.125;
  saved.Metallicity = 0.03125;
#endif
#ifdef BLACK_HOLES
  saved.BHMass = 12.5;
  saved.BHInflowRate = 1.25;
  saved.BHCumulativeRadiatedMass = 3.75;
  saved.BHCumulativeFeedbackEnergy = 8.25;
#endif
  const char *fingerprint = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  native_checkpoint_layout expected = native_checkpoint_layout_make(4096, sizeof(particle_data), 512, 1, fingerprint);
  FILE *f = fopen(argv[1], "wb");
  if(!f) return 4;
  if(fwrite(&expected, sizeof(expected), 1, f) != 1 || fwrite(&saved, sizeof(saved), 1, f) != 1 || fclose(f)) return 5;
  particle_data restored;
  native_checkpoint_layout header = {};
  f = fopen(argv[1], "rb");
  if(!f || fread(&header, sizeof(header), 1, f) != 1 || !native_checkpoint_layout_equal(header, expected)) return 6;
  if(fread(&restored, sizeof(restored), 1, f) != 1 || fclose(f)) return 7;
  if(restored.ID.get() != ID_MAX || memcmp(&saved.ResearchSourceState, &restored.ResearchSourceState, sizeof(research_source_state))) return 8;
  for(unsigned lane = 0; lane < 3; ++lane)
    if(research_stellar_uniform(UINT64_MAX, saved.ID.get(), saved.ResearchSourceState.stellar_random_counter, lane) !=
       research_stellar_uniform(UINT64_MAX, restored.ID.get(), restored.ResearchSourceState.stellar_random_counter, lane)) return 13;
#ifdef STARFORMATION
  if(saved.StellarAge != restored.StellarAge || saved.Metallicity != restored.Metallicity) return 9;
#endif
#ifdef BLACK_HOLES
  if(saved.BHMass != restored.BHMass || saved.BHInflowRate != restored.BHInflowRate ||
     saved.BHCumulativeRadiatedMass != restored.BHCumulativeRadiatedMass ||
     saved.BHCumulativeFeedbackEnergy != restored.BHCumulativeFeedbackEnergy) return 10;
#endif
  header.source_config_sha256[0] ^= 1;
  if(native_checkpoint_layout_equal(header, expected)) return 11;
  header = expected; header.particle_bytes++;
  if(native_checkpoint_layout_equal(header, expected)) return 12;
  printf("PASS native particle source-state roundtrip; particle_bytes=%zu; source_state_bytes=%zu\n", sizeof(saved), sizeof(research_source_state));
  return 0;
}
