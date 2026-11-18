/* Project instrumentation; compiled without -finstrument-functions. */
#include "research_native_call_trace.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <limits.h>

namespace {
const size_t capacity = 65536;
struct slot { uintptr_t address; uint64_t calls; };
slot entries[capacity];
unsigned lock_word;
int active, failed;
uint64_t dropped;
FILE *output;
uintptr_t executable_base;
thread_local bool inside;

void lock() { while(__atomic_exchange_n(&lock_word, 1u, __ATOMIC_ACQUIRE)) {} }
void unlock() { __atomic_store_n(&lock_word, 0u, __ATOMIC_RELEASE); }
void fail(const char *message) { fprintf(stderr, "Native call trace: %s\n", message); failed = 1; }

void finish()
{
  lock();
  __atomic_store_n(&active, 0, __ATOMIC_RELEASE);
  if(output)
    {
      for(size_t i=0; i<capacity; ++i)
        if(entries[i].address)
          {
            Dl_info info;
            if(dladdr(reinterpret_cast<void *>(entries[i].address), &info) &&
               reinterpret_cast<uintptr_t>(info.dli_fbase) == executable_base)
              fprintf(output, "CALL\t%llx\t%llu\n",
                      (unsigned long long)entries[i].address,
                      (unsigned long long)entries[i].calls);
            else
              fprintf(output, "EXTERNAL\t%llx\t%llu\n",
                      (unsigned long long)entries[i].address,
                      (unsigned long long)entries[i].calls);
          }
      fprintf(output, "END\t%s\t%llu\n", failed ? "INCOMPLETE" : "NORMAL_EXIT",
              (unsigned long long)dropped);
      if(fflush(output) || fsync(fileno(output))) fail("cannot persist completed trace");
      if(fclose(output)) fail("cannot close trace");
      output = 0;
    }
  unlock();
}
}

extern "C" void __cyg_profile_func_enter(void *function, void *) __attribute__((no_instrument_function));
extern "C" void __cyg_profile_func_exit(void *, void *) __attribute__((no_instrument_function));
extern "C" void __cyg_profile_func_enter(void *function, void *)
{
  if(!__atomic_load_n(&active, __ATOMIC_ACQUIRE) || inside) return;
  inside = true;
  lock();
  if(__atomic_load_n(&active, __ATOMIC_ACQUIRE))
    {
      uintptr_t address = reinterpret_cast<uintptr_t>(function);
      size_t index = (address >> 4) % capacity;
      size_t probe = 0;
      for(; probe<capacity; ++probe, index=(index+1)%capacity)
        if(!entries[index].address || entries[index].address == address)
          {
            entries[index].address = address;
            if(entries[index].calls == UINT64_MAX) ++dropped;
            else ++entries[index].calls;
            break;
          }
      if(probe == capacity) ++dropped;
    }
  unlock();
  inside = false;
}
extern "C" void __cyg_profile_func_exit(void *, void *) {}

void research_native_trace_begin(int rank, void *entrypoint)
{
  const char *directory = getenv("RESEARCH_NATIVE_TRACE_DIRECTORY");
  if(!directory || !*directory) return;
  Dl_info info;
  if(!dladdr(entrypoint, &info)) { fail("cannot resolve executable load base"); return; }
  executable_base = reinterpret_cast<uintptr_t>(info.dli_fbase);
  char binary[PATH_MAX], trace[PATH_MAX];
  int n=snprintf(binary,sizeof(binary),"%s/native-rank%d-pid%ld.binary",directory,rank,(long)getpid());
  int m=snprintf(trace,sizeof(trace),"%s/native-rank%d-pid%ld.trace",directory,rank,(long)getpid());
  if(n<0 || m<0 || n>=(int)sizeof(binary) || m>=(int)sizeof(trace)) { fail("trace path too long"); return; }
  int source=open("/proc/self/exe",O_RDONLY);
  struct stat st;
  if(source<0) { fail("Linux /proc/self/exe unavailable"); return; }
  if(fstat(source,&st) || st.st_size<1 || st.st_size>512LL*1024*1024)
    { close(source); fail("executable exceeds retention limit"); return; }
  int target=open(binary,O_WRONLY|O_CREAT|O_EXCL,0600);
  if(target<0) { close(source); fail("cannot retain exclusive executable"); return; }
  char buffer[65536];
  ssize_t size;
  off_t copied=0;
  bool good=true;
  while((size=read(source,buffer,sizeof(buffer)))>0)
    {
      ssize_t offset=0;
      while(offset<size)
        {
          ssize_t written=write(target,buffer+offset,size-offset);
          if(written<=0) { good=false; break; }
          offset+=written;
        }
      if(!good) break;
      copied+=size;
      if(copied>st.st_size) { good=false; break; }
    }
  if(size<0 || copied!=st.st_size || fsync(target)) good=false;
  if(close(target)) good=false;
  close(source);
  if(!good) { unlink(binary); fail("executable retention failed"); return; }
  int fd=open(trace,O_WRONLY|O_CREAT|O_EXCL,0600);
  if(fd<0) { fail("cannot create exclusive trace"); return; }
  output=fdopen(fd,"w");
  if(!output) { close(fd); fail("cannot open trace stream"); return; }
  fprintf(output,"NATIVE_CALL_TRACE_V1\nRANK\t%d\nBASE\t%llx\nBINARY\t%s\nSCOPE\tafter_MPI_rank_until_normal_process_exit\n",
          rank,(unsigned long long)executable_base,strrchr(binary,'/')+1);
  if(atexit(finish)) { fclose(output); output=0; fail("cannot register trace finalizer"); return; }
  __atomic_store_n(&active,1,__ATOMIC_RELEASE);
}
