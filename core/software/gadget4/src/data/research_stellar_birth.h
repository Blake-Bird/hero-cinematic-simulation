/* Research2 native SH03 birth bookkeeping and keyed draws.
 * Philox4x32-10: Salmon et al. SC11; round convention checked against
 * DEShawResearch/Random123 published known-answer vectors. This independently
 * written scalar implementation is not a linked Random123 installation.
 * Model reference: Springel & Hernquist 2003, astro-ph/0206393.
 */
#ifndef RESEARCH_STELLAR_BIRTH_H
#define RESEARCH_STELLAR_BIRTH_H
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string>
#include <vector>
#include <mpi.h>
#include <zlib.h>
#include "research_source_state.h"

inline bool research_stellar_seed(const char *text, uint64_t &seed)
{
  seed = 0;
  if(!text || !*text) return false;
  for(const char *p = text; *p; ++p)
    {
      if(*p < '0' || *p > '9') return false;
      const uint64_t digit = *p - '0';
      if(seed > (UINT64_MAX - digit) / 10) return false;
      seed = seed * 10 + digit;
    }
  return true;
}

inline void research_stellar_philox(uint32_t c[4], uint32_t k[2])
{
  for(int round = 0; round < 10; ++round)
    {
      const uint64_t p0 = UINT64_C(0xd2511f53) * c[0];
      const uint64_t p1 = UINT64_C(0xcd9e8d57) * c[2];
      const uint32_t next[4] = {uint32_t(p1 >> 32) ^ c[1] ^ k[0], uint32_t(p1),
                               uint32_t(p0 >> 32) ^ c[3] ^ k[1], uint32_t(p0)};
      for(int j = 0; j < 4; ++j) c[j] = next[j];
      k[0] += UINT32_C(0x9e3779b9);
      k[1] += UINT32_C(0xbb67ae85);
    }
}

/* Lanes 0,1,2: metal partition, channel choice, conversion acceptance.
 * Each positive-SFR opportunity consumes one persisted counter value.
 * Exact uint64 IDs and counters occupy distinct halves of the input. */
inline double research_stellar_uniform(uint64_t seed, uint64_t id, uint64_t opportunity, unsigned lane)
{
  if(lane > 2) return NAN;
  uint32_t c[4] = {uint32_t(id), uint32_t(id >> 32), uint32_t(opportunity), uint32_t(opportunity >> 32)};
  uint32_t k[2] = {uint32_t(seed), uint32_t(seed >> 32)};
  research_stellar_philox(c, k);
  return (double(c[lane]) + 0.5) / 4294967296.0;
}

inline bool research_stellar_record_birth(research_source_state &s, uint64_t parent,
                                         double locked_mass, double time, double scale_factor,
                                         double metallicity, double beta, uint64_t mode)
{
  if(!isfinite(locked_mass) || locked_mass <= 0 || !isfinite(time) || time < 0 ||
     !isfinite(scale_factor) || scale_factor < 0 || !isfinite(metallicity) ||
     metallicity < 0 || metallicity > 1 || !isfinite(beta) || beta <= 0 || beta >= 1 ||
     (mode != 1 && mode != 2) || s.stellar_birth_state_version != 0) return false;
  const double initial = locked_mass / (1 - beta);
  if(!isfinite(initial)) return false;
  s.stellar_initial_mass = initial;
  s.stellar_initial_locked_mass = locked_mass;
  s.stellar_live_mass=locked_mass;s.stellar_remnant_mass=0;s.stellar_population_version=0;
  s.stellar_prompt_returned_mass = initial - locked_mass;
  s.stellar_birth_time = time;
  s.stellar_birth_scale_factor = scale_factor; /* zero denotes isolated evolution */
  s.stellar_birth_metallicity = metallicity;
  for(int e=0;e<11;++e) s.stellar_birth_element_fraction[e]=s.element_mass_fraction[e];
  s.stellar_birth_composition_version=s.chemical_state_version;
  s.stellar_parent_gas_id = parent;
  s.stellar_birth_state_version = 1;
  s.stellar_birth_mode = mode; /* 1 whole-gas conversion, 2 partial spawn */
  s.stellar_last_source_time = time;
  s.stellar_event_counter = 1;
  return true;
}
/* A row is a measured instantaneous source operation, not an inferred change
 * between snapshots. P.Vel is the native canonical velocity (a*v_pec in
 * comoving evolution). Prompt recycling below is virtual SH03 bookkeeping. */
struct research_stellar_birth_event
{
  int particle_index;
  uint64_t ssp_id, parent_id, mode, opportunity_counter;
  double birth_time, scale_factor, metallicity, gross_mass, locked_mass, prompt_mass;
  double gas_mass_before, gas_mass_after;
  double gas_velocity_before[3], gas_velocity_after[3], ssp_velocity[3];
};

inline std::string research_stellar_path(const std::string &directory, const std::string &name)
{
  return directory + (directory.empty() || directory.back() != '/' ? "/" : "") + name;
}

/* Bounded CRC/byte census. CRC detects accidental corruption; Python products
 * additionally pin the exact files with SHA-256. Neither is authentication. */
inline bool research_stellar_file_census(const char *path, uint64_t &bytes, uint64_t &crc)
{
  FILE *f = fopen(path, "rb");
  if(!f) return false;
  unsigned char block[65536]; bytes = 0; uLong checksum = crc32(0L, Z_NULL, 0);
  size_t n;
  while((n = fread(block, 1, sizeof(block), f)))
    {
      if(bytes > UINT64_MAX - n) { fclose(f); return false; }
      bytes += n; checksum = crc32(checksum, block, n);
    }
  const bool good = !ferror(f); const int closed = fclose(f);
  crc = uint64_t(checksum); return good && closed == 0;
}

inline bool research_stellar_same_file(const char *first, const char *second)
{
  struct stat st;
  if(lstat(second, &st) != 0 || !S_ISREG(st.st_mode)) return false;
  const int fd = open(second, O_RDONLY | O_NOFOLLOW);
  if(fd < 0) return false;
  FILE *b = fdopen(fd, "rb");
  if(!b) { close(fd); return false; }
  FILE *a = fopen(first, "rb");
  if(!a) { fclose(b); return false; }
  unsigned char x[65536], y[65536]; bool same = true;
  for(;;)
    {
      const size_t nx = fread(x, 1, sizeof(x), a), ny = fread(y, 1, sizeof(y), b);
      if(nx != ny || memcmp(x, y, nx) != 0) { same = false; break; }
      if(nx == 0) { same = !ferror(a) && !ferror(b); break; }
    }
  if(fclose(a) != 0) same = false;
  if(fclose(b) != 0) same = false;
  return same;
}

/* Exclusive publication in the existing output directory. Identical replay
 * is permitted; a divergent existing file is never replaced. All staging
 * files are removed on normal success/failure. Interrupted staging files are
 * uncommitted and cannot satisfy a reader's commit record. */
template<class Writer>
inline bool research_stellar_publish_file(const std::string &destination, Writer writer, uint64_t &bytes, uint64_t &crc)
{
  std::string pattern = destination + ".stage.XXXXXX";
  std::vector<char> stage(pattern.begin(), pattern.end()); stage.push_back(0);
  int fd = mkstemp(stage.data());
  if(fd < 0) return false;
  FILE *f = fdopen(fd, "wb");
  if(!f) { close(fd); unlink(stage.data()); return false; }
  bool good = writer(f);
  if(fflush(f) != 0 || ferror(f) || fsync(fileno(f)) != 0) good = false;
  if(fclose(f) != 0) good = false;
  if(good) good = research_stellar_file_census(stage.data(), bytes, crc);
  if(good && link(stage.data(), destination.c_str()) != 0)
    good = errno == EEXIST && research_stellar_same_file(stage.data(), destination.c_str());
  if(unlink(stage.data()) != 0) good = false;
  if(good)
    {
      const size_t slash = destination.rfind('/');
      const std::string directory = slash == std::string::npos ? "." : destination.substr(0, slash);
      fd = open(directory.c_str(), O_RDONLY);
      if(fd < 0) good = false;
      else { if(fsync(fd) != 0) good = false; if(close(fd) != 0) good = false; }
    }
  return good;
}

inline std::string research_stellar_chunk_name(uint64_t sequence, int rank, int ranks)
{
  char text[128];
  snprintf(text, sizeof(text), "ssp_birth_%020llu.rank_%06d.of_%06d.tsv", (unsigned long long)sequence, rank, ranks);
  return text;
}

/* All ranks must call this routine. No-event steps do not create files. Global
 * counters advance only after payload and commit-directory fsync succeeds. */
inline bool research_stellar_publish_journal(const std::string &directory,
    const std::vector<research_stellar_birth_event> &events, uint64_t &sequence, uint64_t &event_count,
    long long ti_current, double native_time, int comoving, double unit_mass, double unit_time,
    double unit_velocity, double unit_length, double hubble, double beta, uint64_t seed,
    const char *source_fingerprint, MPI_Comm communicator)
{
  int rank, ranks; MPI_Comm_rank(communicator, &rank); MPI_Comm_size(communicator, &ranks);
  uint64_t local[3] = {uint64_t(events.size()), 0, 0}, total;
  MPI_Allreduce(local, &total, 1, MPI_UINT64_T, MPI_SUM, communicator);
  if(total == 0) return true;
  int good = sequence < UINT64_MAX && total <= UINT64_MAX - event_count && ti_current >= 0 &&
    isfinite(native_time) && native_time >= 0 && (comoving == 0 || comoving == 1) && (!comoving || native_time > 0) &&
    isfinite(unit_mass) && unit_mass > 0 && isfinite(unit_time) && unit_time > 0 &&
    isfinite(unit_velocity) && unit_velocity > 0 && isfinite(unit_length) && unit_length > 0 &&
    isfinite(hubble) && hubble > 0 && isfinite(beta) && beta > 0 && beta < 1 &&
    source_fingerprint && strlen(source_fingerprint) == 64;
  if(good) for(const char *p = source_fingerprint; *p; ++p)
    if(!((*p >= '0' && *p <= '9') || (*p >= 'a' && *p <= 'f'))) good = 0;
  for(const auto &e : events)
    {
      if((e.mode != 1 && e.mode != 2) || e.opportunity_counter == 0 || e.birth_time != native_time ||
         e.scale_factor != (comoving ? native_time : 0) || e.metallicity < 0 || e.metallicity > 1 ||
         e.gross_mass <= 0 || e.locked_mass <= 0 || e.prompt_mass < 0 || e.gas_mass_before <= 0 || e.gas_mass_after < 0 ||
         (e.mode == 1 && (e.ssp_id != e.parent_id || e.gas_mass_after != 0)) ||
         (e.mode == 2 && (e.ssp_id == e.parent_id || e.gas_mass_after <= 0))) good = 0;
      const double scalars[] = {e.birth_time, e.scale_factor, e.metallicity, e.gross_mass,
        e.locked_mass, e.prompt_mass, e.gas_mass_before, e.gas_mass_after};
      for(double value : scalars) if(!isfinite(value)) good = 0;
      for(int k = 0; k < 3; ++k)
        if(!isfinite(e.gas_velocity_before[k]) || !isfinite(e.gas_velocity_after[k]) || !isfinite(e.ssp_velocity[k])) good = 0;
    }
  int all_good; MPI_Allreduce(&good, &all_good, 1, MPI_INT, MPI_MIN, communicator);
  if(!all_good) return false;
  if(local[0])
    good = research_stellar_publish_file(research_stellar_path(directory, research_stellar_chunk_name(sequence, rank, ranks)),
      [&](FILE *f) {
        if(fprintf(f, "ssp_id\tparent_id\tmode\topportunity_counter\tbirth_time\tscale_factor\tmetallicity\tgross_mass\tlocked_mass\tprompt_mass\tgas_mass_before\tgas_mass_after\tgas_v_before_x\tgas_v_before_y\tgas_v_before_z\tgas_v_after_x\tgas_v_after_y\tgas_v_after_z\tssp_v_x\tssp_v_y\tssp_v_z\n") < 0) return false;
        for(const auto &e : events)
          {
            if(fprintf(f, "%llu\t%llu\t%llu\t%llu", (unsigned long long)e.ssp_id, (unsigned long long)e.parent_id,
                (unsigned long long)e.mode, (unsigned long long)e.opportunity_counter) < 0) return false;
            const double values[] = {e.birth_time, e.scale_factor, e.metallicity, e.gross_mass,
              e.locked_mass, e.prompt_mass, e.gas_mass_before, e.gas_mass_after,
              e.gas_velocity_before[0], e.gas_velocity_before[1], e.gas_velocity_before[2],
              e.gas_velocity_after[0], e.gas_velocity_after[1], e.gas_velocity_after[2],
              e.ssp_velocity[0], e.ssp_velocity[1], e.ssp_velocity[2]};
            for(double value : values) if(fprintf(f, "\t%.17g", value) < 0) return false;
            if(fputc('\n', f) == EOF) return false;
          }
        return true;
      }, local[1], local[2]);
  MPI_Allreduce(&good, &all_good, 1, MPI_INT, MPI_MIN, communicator);
  if(!all_good) return false;
  std::vector<uint64_t> census(rank == 0 ? size_t(ranks) * 3 : 0);
  MPI_Gather(local, 3, MPI_UINT64_T, rank == 0 ? census.data() : NULL, 3, MPI_UINT64_T, 0, communicator);
  if(rank == 0)
    {
      char name[96]; snprintf(name, sizeof(name), "ssp_birth_%020llu.commit.json", (unsigned long long)sequence);
      uint64_t bytes, crc;
      good = research_stellar_publish_file(research_stellar_path(directory, name), [&](FILE *f) {
        if(fprintf(f, "{\"schema\":\"native-ssp-birth-journal-v1\",\"sequence\":%llu,\"previous_event_count\":%llu,\"event_count\":%llu,\"cumulative_event_count\":%llu,\"ti_current\":%lld,\"native_time\":%.17g,\"comoving\":%d,\"seed\":\"%llu\",\"source_fingerprint\":\"%s\",\"unit_mass_g\":%.17g,\"unit_time_s\":%.17g,\"unit_velocity_cm_s\":%.17g,\"unit_length_cm\":%.17g,\"hubble_param\":%.17g,\"factor_sn\":%.17g,\"velocity_convention\":\"native canonical P.Vel; peculiar=P.Vel/a when comoving\",\"ranks\":[",
            (unsigned long long)sequence, (unsigned long long)event_count, (unsigned long long)total,
            (unsigned long long)(event_count + total), ti_current, native_time, comoving,
            (unsigned long long)seed, source_fingerprint, unit_mass, unit_time, unit_velocity, unit_length, hubble, beta) < 0) return false;
        for(int r = 0; r < ranks; ++r)
          {
            if(fprintf(f, "%s{\"rank\":%d,\"count\":%llu,\"bytes\":%llu,\"crc32\":%llu,\"name\":",
                r ? "," : "", r, (unsigned long long)census[3*r],
                (unsigned long long)census[3*r+1], (unsigned long long)census[3*r+2]) < 0) return false;
            if(census[3*r]) { if(fprintf(f, "\"%s\"}", research_stellar_chunk_name(sequence, r, ranks).c_str()) < 0) return false; }
            else if(fprintf(f, "null}") < 0) return false;
          }
        return fprintf(f, "]}\n") >= 0;
      }, bytes, crc);
    }
  MPI_Bcast(&good, 1, MPI_INT, 0, communicator);
  if(!good) return false;
  event_count += total; ++sequence; return true;
}
#endif
