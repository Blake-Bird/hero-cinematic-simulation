# Troubleshooting and recovery

Use the saved draft, native logs, or workflow record for the stage that failed. Keep the
original inputs and published results while diagnosing the problem.

Start with the failing stage, not the last image you can find:

1. Confirm the active Python environment and exact command.
2. Read that attempt's terminal/native log and saved status or workflow record.
3. Identify whether the cause is a dependency, input, resource or numerical check.
4. Keep the failed attempt; retry with the corrected prerequisite and a fresh
   destination where publication is immutable.

| Failure stage | First place to look |
|---|---|
| Installation/import | [Dependency symptoms](#diagnose-the-symptom-before-retrying) and [software guide](software.md) |
| Questionnaire | [Interrupted setup](#setup-was-interrupted) |
| Native run or checkpoint | [Evolution failure](#native-evolution-failed) and [early successful exit](#a-native-process-exits-successfully-before-the-final-snapshot) |
| Missing measurement or figure | [Blocked requests](#an-output-request-is-blocked), then the current workflow record |
| Field replay or allocation | [AGAMA replay](#saved-agama-coefficients-fail-the-field-replay) or [renderer memory](#figure-export-exceeds-its-memory-allocation) |
| Broken navigation | [Results verification](#the-results-page-is-missing-or-has-broken-links) |

## Setup was interrupted

Repeat the connected command with the same `--draft PATH`, or continue particle setup
with `gadget --cosmology BUNDLE --draft PATH`. Use `:list` and `:edit N` to review saved
answers. A draft contains paths and physical settings; preserve it with the run records.

## An output request is blocked

Read the launch review or saved workflow's applicability decisions. Supply the named
measurement request and physical assets. Selecting hosts or particle IDs may require
inspecting completed snapshots first. `available` runs ready products and records
omissions; strict requests still need their prerequisites.

## A paired-atlas comparison refuses to resume

The core paired renderer checks its sealed `core_render_receipt.json` against the
current cache data, measurement records, renderer source and original PNG/PDF/SVG,
CSV and NPZ exports. A missing receipt, changed input or damaged export stops reuse.
Keep that comparison for diagnosis and render into a fresh atlas/output directory;
do not manufacture a seal for old files. Direct `render_pair` calls require an empty
destination. Partial failed renders stay unsealed and are not silently overwritten.

Separate scripts can append gallery entries to `comparison_manifest.json`. The core
receipt covers the original renderer's outputs; consult each added product's source
record for its own provenance. Successful core reuse does not verify those additions
or establish scientific convergence.

## Native evolution failed

Inspect the native log and checkpoint state. Analyse only complete saved snapshot sets.
Successful post-processing does not repair an interrupted or invalid evolution.

Native continuation uses the original frozen run:

```bash
python commands.py gadget --run-dir /absolute/path/to/RUN --resume
```

Use checkpoint recovery only through its documented route. Do not replace an executable
or rewrite checksums to force a checkpoint onto a different build.

## Analysis or rendering failed

Open the exact workflow record, read the stage's failure reason, and follow its retry
guidance. Native continuation and radial-shape selective resume retain their existing
restrictions. Other retries may publish a fresh generation; requests with immutable
output destinations need a new destination.

For a platform-specific Blender failure, verify that platform's actual executable,
saved scene and render log. A successful render on another machine does not resolve
the failing attempt.

## Saved AGAMA coefficients fail the field replay

If `verify_halo_structure` reports “Reloaded host coefficients do not reproduce
their measured field”, preserve the rejected product and its coefficient files.
Compare each component before/after export at the exact validation positions,
then compare their radii with the declared radial grid. A retained triaxial
stellar control reproduced its interpolated field inside a 0.05–10 kpc grid,
but the reloaded exterior extrapolation differed by about 0.2% near the grid
edge. This exceeded the serialization replay check even though the independent
force budget was looser. The guard correctly rejected publication.

For a new experiment, declare a grid domain containing every intended field
query, including the calibration and withheld validation locations, before
running it. The retained eight-epoch assembly control uses an outer grid
radius of 40 proper kpc and passes the same replay and field checks. This does
not validate exterior extrapolation. Keep the original science tolerances and
source hashes; do not repair a failed receipt by rewriting either. The
[shape guide](learning-guide.md#read-shape-and-environment-measurements-from-their-arrays)
explains the distinct roles of field validation and shape support.

## Figure export exceeds its memory allocation

A measured product can pass its scientific checks and still be too large for
the chosen renderer allocation. For example, the known-profile splashback
control's 43-MiB environment JSON exceeds the exporter's conservative 1-GiB
workspace bound. Keep the completed product and retry only the export with a
larger explicit `memory_gib`/`--memory-gib` allocation that fits the machine.
The retained export passed with 4 GiB. Read the command's `--help` for its
specific argument; the Python exporter uses `memory_gib`.

A larger allocation supplies workspace. It does not resolve failed field,
feature, calibration or support checks, and those settings remain unchanged.

## The results page is missing or has broken links

Rebuild navigation without running simulation or analysis:

```bash
python commands.py results /absolute/path/to/RUN --verify
```

The command reports absent files, checksum mismatches, external paths, and unsupported
older manifests. Keep the entire run together when moving it. See [outputs](outputs.md)
for the navigation rules.

## Storage is cluttered

Keep environments, bytecode, Finder metadata, native builds and generated caches
out of the portable source. Creating an archive is optional. Keep builds, receipts,
snapshots, checkpoints, physical tables, licensing, and scientific source records in
their declared locations.

A failed simulation is not a disposable cache. Consult [the architecture
guide](architecture.md#locations-and-ownership) before moving or removing run data.

## Diagnose the symptom before retrying

| Symptom | What to inspect | Next step |
|---|---|---|
| MUSIC writes gas velocities but no gas coordinates or IDs | The staged `[setup]` section, actual gas datasets and structural audit | Use the updated production template with explicit `do_SPH=yes`. MUSIC reads this switch before output-plugin initialization. Preserve the rejected generation; stage and generate a fresh parent. The learning guide gives the minimum-grid reproduction. |
| Process aborts with `OMP: Error #15` and two `libomp` runtimes | The optional native libraries imported into that Python process | Run the three regression commands in the README as separate processes. The lab's library profile also uses isolated children. Keep the abort log; do not enable `KMP_DUPLICATE_LIB_OK` to suppress the conflict. |
| `ModuleNotFoundError` for an optional package | Active Python and the selected dependency profile | Activate the external environment; install the workflow's profile from [software](software.md). |
| monofonIC cannot find MPI FFTW headers or symbols | Build log, active MPI compiler and FFTW development packages | Ubuntu needs both `libfftw3-dev` and `libfftw3-mpi-dev`. On a cluster, load matching MPI/FFTW modules before using `--no-system-packages`. Retry the build with a consistent toolchain. |
| Grackle worker cannot resolve headers or library | `pkg-config --modversion grackle`, the installed `grackle.pc` and the compiler log | Follow the [external-prefix build](software.md#build-optional-grackle-and-inspect-its-cooling-control). Set `PKG_CONFIG_PATH` to its actual `lib/pkgconfig` or `lib64/pkgconfig`; verify version and double precision before retrying. |
| MPI launch fails before any native step | Launcher stderr, rank count, executable capability and MPI/compiler/library identity | Use the MPI installation that built the executable and fit the declared CPU allocation. A launcher failure does not establish a solver defect; keep it separate from a completed evolution. |
| Process is killed or a renderer raises `MemoryError` | Native/worker log, declared allocation and scheduler/OS memory report | Reduce the separately configured control size or increase its explicit allocation within machine limits. Keep the original result and distinguish a resource failure from failed numerical acceptance. |
| `No module named bpy` | Whether a Blender-only helper was run with ordinary Python | Use the public Blender command or run the helper with `blender --background --python SCRIPT.py`. |
| `int()` received `None` in a hand-written configuration | The traceback's actual setting and the example's unset values | Fill the required numerical setting through setup; `core/config.yaml` is a template containing null placeholders. |
| No gas-temperature map | Map status, valid gas energies/densities, source temperature convention, star-forming/EOS exclusions | Check whether the selected gas supports a physical temperature. Do not relabel an effective EOS variable as measured thermal temperature. |
| No stars/BH/halos in a tiny run | Saved species counts, source configuration and finder particle threshold | Verify whether the model was enabled and whether the population actually formed. Increase duration/resolution only through a new experiment. |
| Consistent Trees reports `SKIPPED` | Number of usable saved Rockstar catalogues and objects per epoch | Supply adequate multi-epoch catalogues; a successful skip is not successful tree construction. |
| Rockstar reports insufficient worker slots | CPU budget and periodic finder requirements | Use the required resources or retain configured FOF/SUBFIND; do not exceed the machine's resource budget. |
| A movie exists but the current attempt failed | Exact workflow attempt, product state, source hashes | Open Status and use its retry command. An older MP4 does not satisfy the new attempt. |
| A nested zoom contains more particles than requested for the tiny uniform grid | Refined species counts and coarse boundary particles | This is expected geometry; inspect the zoom contamination controls rather than discarding boundary particles. |
| A sparse Blender render looks empty | Point count, camera and frame, species visibility and sampling | Inspect the `.blend` and a preview. Low particle counts can be visually sparse; verify actual points before changing display settings. |
| A gas-free run crashes at `SPH-INIT-ENTROPY` | Native source revision and the actual global gas count | The maintained initializer skips entropy-tree work when the global gas count is zero. Rebuild from the updated source in a new run; preserve the failed generation and its log. A rank with no local gas must still participate when gas exists on another rank. |
| A file hash changed | Whether a source snapshot/configuration was edited after publication | Restore the original input or create a new fixture/run and regenerate its products. Never patch a receipt to make the check pass. |
| Paired atlas rejects a cached epoch | Its metadata seal and cache binding: source files/inventory, run metadata, map grid and implementation hashes | Choose a fresh measurement destination and regenerate from the retained inputs. Older caches without these bindings are retained for review, but cannot be reused by the updated producer. Do not add a seal or edit bindings to bypass rejection. |
| HDF5 reader reports an absent shard or dataset | Complete snapshot-set listing, publication marker and required field/species contract | Recover the original complete generation or resume its native run through the supported checkpoint route. Do not concatenate unrelated generations or replace absent arrays with zeros. |
| Optional lab rejects an output inside the repository | Resolved output path, including symlink targets | Choose an external sibling such as `../research2-runtime/optional-lab/NEW_RUN`. The guard applies to the CLI, server and advanced workers. |
| Paired power reports no common supported bins | Saved physical k values, phase-comparison k limit, actual epochs and every realization's masks | A 4³ IC mesh can have no overlap with the native spectrum. Use a separately configured larger control if needed; do not extend phase support by assertion. |
| A posterior figure exists but the sampler reports `CHAIN_TOO_SHORT` | Original chain, burn-in, acceptance fractions and autocorrelation diagnostics | Treat the image as execution evidence. Run a deliberately budgeted longer experiment and independent ensembles before inference. |
| NFW concentration is `MEASURED` but χ² is large | Measured shell densities, fit domain, particle support, degrees of freedom and residuals | Optimizer completion does not validate NFW. Retain poor-fit diagnostics and reconsider the model/domain through a new request. |
| Truncated distance-ladder fit reports BFGS precision loss | Original observations, covariance, priors, selection likelihood and full traceback | Preserve the failed selected fit. A complete likelihood is appropriate only for a separately defined complete sample; switching it is not a repair for selected data. |

For an execution-only exercise, the [learning guide](learning-guide.md) explains the
required input/output chain. The [retained smoke report](smoke-test.md) distinguishes
fixed fixture mistakes from software failures and untested branches.

### A native process exits successfully before the final snapshot

GADGET can return exit code zero when its CPU time limit triggers a restart.
Read the end of the native log and check that every requested snapshot has
a complete publication marker. For a time-limited run, keep the restart and
use the documented continuation command with a suitable time allowance.
Keep the original timeline and check the actual saved scale factor after
continuation. For a new control with closely spaced output epochs, set the
maximum step below the minimum spacing in the native time variable. In
comoving evolution this spacing is measured in `ln(a)`.
