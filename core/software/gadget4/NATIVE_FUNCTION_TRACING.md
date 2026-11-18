# Optional native callable tracing

Add `RESEARCH_NATIVE_CALL_TRACE` to the actual GADGET configuration before
building. The Makefile adds debug information and compiler function-entry hooks;
the recorder translation unit explicitly disables its own instrumentation.
This option is for recording reachability, and adds runtime overhead.

Before launching the executable, set `RESEARCH_NATIVE_TRACE_DIRECTORY` to an
existing writable directory. Every MPI process retains its actual Linux
`/proc/self/exe` binary and an exclusive rank/PID trace. Executables exceeding
512 MiB are rejected. Calls are counted after MPI rank initialization, including
calls from threads, in a bounded 65,536-entry table. Process aborts and table or
counter overflow do not yield a complete trace. Ghost/listener ranks are included
when they terminate normally. A trace closes at process exit and does not certify
that the simulation reached its requested final time.

The decoder is available through `python -m core.native_function_trace`, with
`--request` and `--output`. A request has schema
`native-function-trace-request-v1` and three pinned `{path, sha256}` inputs:
`trace`, `build_record`, and `addr2line` (the executable GNU-compatible tool).

The build record has schema `native-trace-build-record-v1`, `binary` as a pinned
`{path, sha256}` reference, `source_root` as the compilation source root, and
`source_hashes` mapping absolute source paths to their actual build-time SHA-256
values. These bindings must come from the build that produced the executable;
this decoder does not establish clean-build reproduction. Retained executed
binary bytes must match that pinned binary exactly. The decoder handles ELF
executables and PIE load addresses, resolves recorded functions in bounded
batches, and refuses to credit files absent from the source bindings. Keep debug
information in the retained binary. Unknown debug locations remain unresolved.

A repository audit can consume a JSON list of pinned decoded reports through
`--native-traces`. Changing a trace, executable, source, decoder, or symbol tool
invalidates its evidence. Missing execution evidence is not proof that a path is
dead, and native tracing does not establish scientific acceptance.

Compiler hook reference:
https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
