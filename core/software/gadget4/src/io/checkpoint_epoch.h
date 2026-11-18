/* Canonical time metadata only. No modification to GADGET dynamics.
 * Shared contract: interpolate CLASS proper time linearly in a, never extrapolate.
 * The transport CRC32 does not provide cryptographic authenticity. */
#ifndef GADGET_CHECKPOINT_EPOCH_H
#define GADGET_CHECKPOINT_EPOCH_H
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <zlib.h>

inline bool native_checkpoint_epoch_stage(const char *output_dir, const char *payload,
                                         int rank, unsigned long long generation,
                                         double a, char *stage, size_t capacity, bool comoving = true)
{
  if(!comoving)
    {
      if(!isfinite(a) || a<0 || snprintf(stage,capacity,"%s.epoch.json.tmp.%d",payload,(int)getpid()) >= (int)capacity) return false;
      FILE *metadata=fopen(stage,"w");
      if(!metadata) return false;
      int written=fprintf(metadata,"{\"schema\":\"native-checkpoint-epoch-v2\",\"rank\":%d,\"generation\":%llu,\"native_time\":%.17g,\"clock_representation\":\"native_time_L_over_V\",\"cosmic_time_status\":\"NOT_APPLICABLE\",\"cosmic_time_gyr\":null}\n",rank,generation,a);
      bool okay=written>0 && fflush(metadata)==0 && fsync(fileno(metadata))==0;
      if(fclose(metadata)!=0) okay=false;
      return okay;
    }
  char clock_path[4096], source[65] = {0}, line[256], checksum[9] = {0};
  snprintf(clock_path, sizeof(clock_path), "%s/canonical-clock.tsv", output_dir);
  FILE *clock = fopen(clock_path, "r");
  bool available = false;
  double time = 0, previous_a = 0, previous_t = -1;
  if(clock)
    {
      int consumed = 0;
      if(!fgets(line,sizeof(line),clock) ||
         sscanf(line,"# native-canonical-clock-v1 %64[0123456789abcdef] %8[0123456789abcdef]%n",source,checksum,&consumed)!=2 ||
         strlen(source)!=64 || strlen(checksum)!=8 ||
         (line[consumed]!='\n' && line[consumed]!='\0')) { fclose(clock); return false; }
      unsigned long crc = crc32(0L,Z_NULL,0);
      int count = 0;
      while(fgets(line,sizeof(line),clock))
        {
          double x,t;
          int end=0;
          crc = crc32(crc,(unsigned char*)line,strlen(line));
          if(sscanf(line,"%lf %lf%n",&x,&t,&end)!=2 ||
             (line[end]!='\n' && line[end]!='\0') || !isfinite(x) || !isfinite(t) ||
             x<=previous_a || t<0 || (count && t<=previous_t)) { fclose(clock); return false; }
          if(x==a) { time=t; available=true; }
          if(count && previous_a<a && a<x)
            { time=previous_t+(t-previous_t)*(a-previous_a)/(x-previous_a); available=true; }
          previous_a=x; previous_t=t; count++;
        }
      bool okay=!ferror(clock) && count>=2 && crc==strtoul(checksum,NULL,16);
      if(fclose(clock)!=0 || !okay || !available) return false; /* advertised clock cannot disappear */
    }
  else if(errno!=ENOENT) return false;
  if(!isfinite(a) || a<=0) return false;
  if(snprintf(stage,capacity,"%s.epoch.json.tmp.%d",payload,(int)getpid()) >= (int)capacity) return false;
  FILE *metadata=fopen(stage,"w");
  if(!metadata) return false;
  int written;
  if(available)
    written=fprintf(metadata,"{\"schema\":\"native-checkpoint-epoch-v1\",\"rank\":%d,\"generation\":%llu,\"a\":%.17g,\"z\":%.17g,\"cosmic_time_status\":\"AVAILABLE\",\"cosmic_time_gyr\":%.17g,\"clock_source_sha256\":\"%s\"}\n",rank,generation,a,1/a-1,time,source);
  else
    written=fprintf(metadata,"{\"schema\":\"native-checkpoint-epoch-v1\",\"rank\":%d,\"generation\":%llu,\"a\":%.17g,\"z\":%.17g,\"cosmic_time_status\":\"UNAVAILABLE\",\"cosmic_time_gyr\":null,\"clock_source_sha256\":null}\n",rank,generation,a,1/a-1);
  bool okay=written>0 && fflush(metadata)==0 && fsync(fileno(metadata))==0;
  if(fclose(metadata)!=0) okay=false;
  return okay;
}
#endif
