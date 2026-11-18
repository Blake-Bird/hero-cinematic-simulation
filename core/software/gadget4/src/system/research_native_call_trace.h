#ifndef RESEARCH_NATIVE_CALL_TRACE_H
#define RESEARCH_NATIVE_CALL_TRACE_H
/* Optional Linux/GCC-compatible function-entry instrumentation. */
void research_native_trace_begin(int rank, void *entrypoint) __attribute__((no_instrument_function));
#endif
