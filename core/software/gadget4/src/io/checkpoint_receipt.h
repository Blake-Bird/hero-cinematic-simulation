/* Native checkpoint receipt validation. CRC32 detects accidental corruption;
 * it does not authenticate raw checkpoint ABI or an adversarial replacement. */
#ifndef GADGET_CHECKPOINT_RECEIPT_H
#define GADGET_CHECKPOINT_RECEIPT_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <zlib.h>

struct native_checkpoint_receipt
{
  unsigned long long generation;
  unsigned long long size_bytes;
  double scale_factor; // generic native clock value; v1 retains the historical field name
  int comoving;
  unsigned long crc;
};

inline bool native_checkpoint_verify(const char *payload, const char *metadata,
                                     int expected_rank, int expected_ranks,
                                     native_checkpoint_receipt &receipt)
{
  FILE *info = fopen(metadata, "r");
  if(!info) return false;
  char line[2048], hex[9] = {0};
  int rank = -1, ranks = -1, consumed = 0;
  bool okay = fgets(line, sizeof(line), info) != NULL;
  int extra = fgetc(info);
  if(fclose(info) != 0 || extra != EOF) okay = false;
  if(!okay) return false;
  if(strstr(line, "\"generation\":-") || strstr(line, "\"generation\":+") ||
     strstr(line, "\"size_bytes\":-") || strstr(line, "\"size_bytes\":+")) return false;
  int fields = sscanf(line, "{\"schema\":\"gadget-rank-checkpoint-v1\",\"rank\":%d,\"ranks\":%d,\"generation\":%llu,\"a\":%lf,\"size_bytes\":%llu,\"crc32\":\"%8[0123456789abcdef]\"}%n",
                      &rank, &ranks, &receipt.generation, &receipt.scale_factor,
                      &receipt.size_bytes, hex, &consumed);
  receipt.comoving = 1;
  if(fields != 6 || consumed == 0)
    {
      consumed = 0;
      fields = sscanf(line, "{\"schema\":\"gadget-rank-checkpoint-v2\",\"rank\":%d,\"ranks\":%d,\"generation\":%llu,\"native_time\":%lf,\"size_bytes\":%llu,\"crc32\":\"%8[0123456789abcdef]\"}%n",
                      &rank, &ranks, &receipt.generation, &receipt.scale_factor,
                      &receipt.size_bytes, hex, &consumed);
      receipt.comoving = 0;
    }
  if(fields != 6 || consumed == 0 || rank != expected_rank || ranks != expected_ranks ||
     receipt.generation == 0 || receipt.size_bytes == 0 ||
     !isfinite(receipt.scale_factor) || receipt.scale_factor < 0 || (receipt.comoving && receipt.scale_factor == 0) || strlen(hex) != 8) return false;
  for(char *p = line + consumed; *p; p++) if(!isspace((unsigned char)*p)) return false;
  receipt.crc = strtoul(hex, NULL, 16);
  FILE *input = fopen(payload, "rb");
  if(!input) return false;
  unsigned char block[65536];
  unsigned long crc = crc32(0L, Z_NULL, 0);
  unsigned long long bytes = 0;
  size_t count;
  while((count = fread(block, 1, sizeof(block), input)) != 0)
    { crc = crc32(crc, block, count); bytes += count; }
  okay = !ferror(input);
  if(fclose(input) != 0) okay = false;
  return okay && bytes == receipt.size_bytes && crc == receipt.crc;
}
#endif
