# Native SIDM resource measurements

Completed source event logs now retain native per-interval resource telemetry
for both the distributed spatial backend and the one-rank all-pair reference.
The existing `native_sidm_resources` workflow stage decodes these records from
the completed native stage's retained files.

CPU scopes use the calling thread's CPU clock. The dispatch scope covers each
actual pair operation, while separate rare and frequent primitive scopes record
their calls and CPU time, including event-log writes. A hybrid dispatch records
one rare call and two frequent calls. Returned data include measured CPU seconds
per primitive call; the resource estimator can explicitly select the hybrid
composition and account for all three calls. CPU-clock failure produces an
unavailable measurement and does not change the particle evolution.

`getrusage(RUSAGE_SELF)` supplies the rank process's resident-memory high-water
mark through interval completion. Linux KiB and macOS byte units are converted
explicitly. Unsupported units remain unavailable. These peaks include earlier
process work. The decoder publishes the largest rank peak and the sum of rank
peaks as an upper bound, not as a simultaneous measurement of MPI memory.

Exchange instrumentation records the largest simultaneously retained vector
capacity payload within a single exchange call, including outgoing records,
packed records, received records/origins and count/offset/cursor vectors. This
measurement excludes allocator metadata, MPI-private buffers and caller maps.
The independent process high-water measurement includes resident allocations.

Dynamic kernel and angular lookup table bytes are zero for the implemented
analytic kernel and CDF operators. Quadrature constants are executable/static
storage, outside the dynamic lookup-table metric. These zeros cannot be reused
for a future tabulated operator without updating the producer and schema.

Rare and frequent hybrid logs retain the same interval telemetry. The decoder
requires it to agree and counts it once. Completed checkpoint-replay logs retain
the original producer's timings; replay performance is not substituted for those
measurements. Older logs can still contribute their measured counters, but do
not acquire missing CPU or memory measurements.

References:
https://man7.org/linux/man-pages/man2/getrusage.2.html
https://man7.org/linux/man-pages/man3/clock_gettime.3.html
