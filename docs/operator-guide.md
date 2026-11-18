# Operator guide

Use this guide for installation, saved-run analysis, visual tools, and native
continuation. The [README](../README.md) is the starting point; the [full-run
walkthrough](architecture.md) explains how the stages connect.

## Choose a task

| What you want to do | Follow this section |
|---|---|
| Install and activate the environment | [Ubuntu / WSL2 setup](#install-on-ubuntu-2404-or-wsl2) |
| Start a new physical simulation | [Connected run](#start-a-fresh-connected-run) |
| Inspect or analyse completed snapshots | [Saved-run analysis](#analyse-saved-outputs-including-an-interrupted-run) |
| Produce a scene, map, portable export or movie | [Visual tools](#movies-portable-data-and-interactive-blender) |
| Find output files and their required inputs | [Output catalogue](#output-catalogue) |
| Diagnose native cooling or a failed stage | [Cooling controls](#native-cooling-correction-and-verification) · [Troubleshooting](troubleshooting.md) |
| Understand which stages are connected | [Workflow and separate experiments](#what-connects-and-what-needs-a-separate-experiment) |
| Locate source, runtime state and checkpoints | [Filesystem layout](#filesystem-layout) |
| Check or transfer an update | [Validation and packaging](#check-and-package-an-update) |

The shortest route is **install → activate → configure one run → inspect saved
epochs → plan analysis → analyse → open results**. The visual and research
recipes later in this guide are optional; run only the ones you selected.
For a fast software check before a simulation, use the
[32-particle learning walkthrough](learning-guide.md#start-with-a-32-particle-walkthrough).

Commands below are examples to run in the installed environment. Replace run, asset, and
destination placeholders with your own paths. Existing named runs and immutable
publications must be preserved.

`RUN`, `RUN_DIR` and `MAP_BUNDLE` below stand for existing directories. `KEY`,
`ID`, `N` and `ROW` must come from your saved data or command help.
`NEW_DIRECTORY` means a new external destination for that product. Replace
these placeholders before copying a command; they are not supplied presets.
Run `python commands.py COMMAND --help` to inspect arguments without launching
the selected workflow. Unless a block is labelled PowerShell, run it in the
activated Linux/macOS shell from the source root.

Find a command quickly in the [complete command reference](command-reference.md).
For an explanation of what each backend takes in and returns, read the
[learning guide](learning-guide.md). Inspect the [smoke report](smoke-test.md) for
retained execution examples and their limitations.

## Install on Ubuntu 24.04 or WSL2

From **Windows PowerShell**, first enter Ubuntu:

```powershell
wsl -d Ubuntu-24.04
```

Run the remaining shell commands **inside Ubuntu**, as your normal user. Keep source and
run data in the Linux filesystem, for example `/home/YOUR_USER/`, rather than the
Windows-mounted `/mnt/c/` tree. Replace the archive path below with the location of your
transferred archive:

```bash
cd ~
unzip /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/research2-source.zip
cd ~/research2
ls commands.py scripts/setup.sh
bash scripts/setup.sh
source ../research2-runtime/environments/default/bin/activate
```

The official package contains its own top-level `research2/` folder; do not add another
`-d research2` when extracting it. If you made a ZIP manually, find the folder
containing `commands.py` and `scripts/`, then `cd` there. A missing `scripts/setup.sh`
usually means the terminal is in the wrong folder.

Setup installs Ubuntu development libraries, creates an external Python environment,
installs the `base` Python dependency profile, and builds CLASS, AxiCLASS, and
monofonIC. Additional profiles and engines are explicit choices. GADGET is compiled
separately for each run's physical configuration. Setup needs Internet access for system
and Python packages. Read the final error if a step fails; do not start the simulation
after an incomplete installation.

Before continuing, confirm that activation uses the external environment:

```bash
python -c 'import sys; print(sys.executable)'
python commands.py --help
```

The first command should print the selected runtime environment's Python path.
The second should list Research2 commands. An import/build failure here belongs
to installation; read the setup log before configuring a run.

Optional setup choices:

```bash
# Install every declared Python profile, including streams/SBI/cosmography.
# This can require substantial build time and storage.
bash scripts/setup.sh --profile all --engines all

# Use existing system development libraries (for example on a cluster).
bash scripts/setup.sh --no-system-packages
```

Profiles are declared in [dependencies.json](../configs/runtime/dependencies.json).
Installing all profiles does not enable all physical models. Grackle needs a compatible
built library and cooling table; SPS needs its physical model assets; Blender must be
installed separately if scene creation/rendering is requested.

Setup is an installation step. In each later Ubuntu terminal, activate the same
environment before using `python commands.py ...`. Do not copy a macOS `.venv` or native
executable to Linux.

## Start a fresh connected run

After installation and activation, from the source folder:

```bash
python commands.py run
```

This collects a new cosmology, runs CLASS or AxiCLASS, hands the **exact newly created
bundle** to the GADGET driver, collects simulation settings, generates or prepares the
configured initial conditions, evolves, then runs the selected post-processing. It does
not search for an arbitrary newest run. A failure in an earlier stage stops the chain.
Existing named runs are not overwritten.

Keep the run path printed by the driver. Inspect these files before choosing
additional analysis; restart filenames and snapshot shard counts depend on the
native configuration.

| File or folder in the run | What to verify |
|---|---|
| `config.yaml` | Selected cosmology, physics, resolution, resources and products |
| `manifest.json` | Original input/build/source bindings for this run |
| `input/param.txt` | Native parameter values and output destinations actually supplied |
| `input/output_times.txt` | The requested native snapshot schedule |
| `data/` and native logs | Complete saved snapshot sets, actual epochs and solver outcome |

A completed native evolution and completed analysis have separate statuses.
The saved workflow report states which requested products succeeded, failed
or still need inputs. Use [outputs](outputs.md) for navigation and
[troubleshooting](troubleshooting.md) for a failed or interrupted stage.

You choose the cosmology and simulation names, physical models, resolution, resources,
output redshifts, numerical tolerances, and products. Several related numerical inputs
share a comma-separated prompt. Read the printed field order and units; every value
remains explicit. Invalid rows are retried. Exact integer seeds are never rounded
through floating point. Advanced physical source settings still have individual prompts
where that makes units clearer.

For example, a prompt listing `H0, Omega_b, Omega_cdm` expects three numbers in that
order. These are **Omega density fractions**, not physical densities `omega =
Omega*h^2`. Printed numbers in documentation are not run settings.

At **Graphs and outputs**, choose one of:

| Choice | Behavior |
|---|---|
| `available` | Run every product whose saved-data prerequisites and explicit requests are ready. Print and record omissions. |
| `all` | Request every physically applicable product; missing measurement requests prevent a complete workflow. |
| Comma-separated names | Request exactly those products; unavailable explicit selections are errors. |

`available` is the broad automatic saved-data route. It cannot invent stellar host
selections, zoom apertures, posterior samples, merger trees, or missing native fields.
Inspect the printed omissions and the workflow JSON to see what actually ran. Selecting
`both` halo finders plus `histories`, `all`, or `available` also enables Consistent
Trees during fresh-run preparation. The saved `analysis.requested_outputs` records the
selection. Broad selections also set the core analysis scope to `all`. Actual
applicability is resolved from saved particles and catalogue support after evolution.

Choose `fof` halo masses with linking length `0.2` for the additional HMF-fit
comparisons. Supply enough linear-spectrum k coverage for the chosen mass range.

Independent stages remain available:

```bash
python commands.py class
python commands.py gadget --cosmology /absolute/path/to/linear_run/spectrum.json
python commands.py gadget --help
```

For repeatable user-authored inputs, `run --linear-config COSMOLOGY.json --answers
SIMULATION.json` accepts the same input schemas as the individual commands. The
simulation answers must name the new cosmology bundle. These files are settings you
author, not presets. Use `run --help` for output paths and `--state-root`.

## Analyse saved outputs, including an interrupted run

Completed snapshot sets can be inspected without restarting GADGET. Start with a
read-only applicability plan, then run the supported analysis:

```bash
python commands.py analysis /absolute/path/to/run --graphs available --dry-run
python commands.py analysis /absolute/path/to/run --graphs available
python commands.py results /absolute/path/to/run --verify
```

If native halo catalogues have not been generated, request the configured finder
prerequisites as well:

```bash
python commands.py analysis /absolute/path/to/run --graphs available --find-halos
```

Open `RUN/results/index.html` after analysis. Its Status page is the place to
check omissions and failures before interpreting an image. `--dry-run` prints
the applicability plan; it does not create the scientific measurements.

This runs configured finders; it does not enable an unconfigured finder or change a
frozen simulation's tree settings. Missing or unsuccessful prerequisites are retained as
failures rather than reported as successful figures.

The normal workflow covers core scientific analysis plus requested maps, BH histories,
stellar measurements, zoom audits and Blender exports. Core products include
`hmf,pk,concentration,correlation,bias,spatial,subhmf,subhalos,spin,
profiles,shape,hydro,galaxies,histories`. Use `analysis --help` for request files.

The legacy `analyze_run(..., full_analysis=True)` route is available through the public
CLI:

```bash
python commands.py analysis /absolute/path/to/run --full-analysis
# Explicitly archive previous results before regenerating that legacy suite:
python commands.py analysis /absolute/path/to/run --full-analysis --overwrite
```

That **legacy core suite** does not invoke every workflow product. Use the normal
`--graphs` route for BH histories, maps, movies, XDMF, planetarium and Blender.
Selected-product analysis publishes fresh `results/analysis_products_*/` bundles and
keeps prior bundles. `--overwrite` is only for the legacy full-analysis route; it
archives the previous results folder. Repeated failed legacy attempts can consume disk
space, so inspect the retained folders.

After a cooling failure, analysing completed snapshots does not repair cooling, finish
missing epochs, or prove that the physical run was sound. Inspect the native log and
failed particle state before restarting. Do not replace invalid temperatures with zero.
Histogram bin padding only resolves floating-point geometry; it never creates
measurements. Entirely invalid temperatures are reported as unavailable.

## Movies, portable data and interactive Blender

The saved-run `--graphs available` route includes ready visualization products:
`movie,gas_movie,temperature_movie,stars_movie,halo_movie,xdmf,planetarium,
blender,blender_movie`. An explicit subset is also supported:

```bash
python commands.py analysis /absolute/path/to/run --graphs maps,movie,gas_movie,temperature_movie,stars_movie,halo_movie,xdmf,planetarium,blender_movie
```

`blender_movie` builds one interactive `.blend`, preview PNGs and the animated MP4.
Selecting both `blender` and `blender_movie` builds the scene once. Cycles denoising is
disabled in generated scenes and saved-scene renders so Blender builds without
OpenImageDenoise can render. Saved-scene rendering also mutes compositor denoise nodes
in memory; it does not rewrite the original `.blend`. Low-sample output can therefore
appear noisy.

For a sealed saved scene, `render-scene --prepare-only` writes a `request.json`
and a `launch.json` containing the exact headless Blender command. The scene
and its publication must be available on the rendering machine; the preparation
does not by itself render or validate the image. After an interrupted render,
the Blender-side renderer can continue recorded frames by adding `--resume` at
the end of that command, after the existing `--output .../frames` argument.
For example, using the same Blender binary and paths from `launch.json`:

```bash
blender --background --python-exit-code 1 --python /path/to/research2/core/scene_render.py -- /path/to/render/request.json --output /path/to/render/frames --resume
```

The checkpoint checks the request, sealed scene, Blender version, renderer
source, selected device, and each recorded frame hash before continuing. It
retains per-attempt failure records. If a frame exists without a checkpoint
record (for example, Blender was interrupted while writing it), move that
file to a safe location outside the frames directory before resuming; it will
not be silently overwritten or accepted as complete. A changed request,
renderer, or device requires a fresh output. The wall-time limit and logging in
the original launcher are not reapplied by this direct Blender command; apply
the appropriate scheduler or process limit and retain its log. Run only one
renderer against a frame directory at a time; the checkpoint is not a lock. Recovery has
not yet been verified by an actual interrupted render.

### Queues and batch analysis

For multiple simulations, run `python commands.py run` and enter the number of
simulations at the first prompt, or use `python commands.py run --count 3`. Each run is
configured separately; its linear spectrum is prepared during configuration, then the
particle simulations run sequentially. A queue report records each outcome under the
runtime's `simulation_queues` directory. Open that queue's `index.html` for run links
and status, or inspect `queue_report.json` for details. Incomplete analysis is recorded
separately from successful simulation execution.

For saved simulations, `python commands.py analysis` accepts one or several
comma-separated run numbers when selecting interactively. For explicit paths, use
`python commands.py analysis-batch RUN_A RUN_B --graphs hmf --output NEW_DIRECTORY`.
Each run receives its own normal analysis; the first is the default comparison
reference. The batch attempts an HMF overlay when both analyses contain HMFs,
power-spectrum and halo-correlation overlays when selected and available, and paired
dark-matter maps where their geometry permits it. The batch report lists unavailable
comparisons and individual failures. Specialized per-run requests are still handled
individually. The batch output also contains an `index.html` linking to available
individual figures and comparison results. Its run overview table also exports
`run_overview.csv`, with configured cosmology, physics, resolution and requested versus
analyzed redshifts. With three or more runs, the batch also attempts one combined
overlay per selected scientific product (`hmf`, `pk`, `correlation`). Each overlay
includes every selected run at a common saved epoch, exports PNG/PDF and measured-curve
CSV files, and records unavailable comparisons rather than dropping runs.

### Void boundary histories

To measure boundaries from the saved density volumes of a `void-explorer` bundle, use
`python commands.py void-boundary-history RUN VOID_BUNDLE --output NEW_DIRECTORY`. This
exports periodic six-connected below-threshold masks, volume histories, adjacent-epoch
overlap, a CSV table and an HTML results page. Tracking maximizes nonzero mesh-cell
Jaccard overlap backward from the final selected region and stops at a missing overlap.
It measures Eulerian density excursions: large connected regions and splitting/merging
can make void identity ambiguous.

### Scenes and halo histories

The combined scene uses the current sealed analysis bundle's PNG figures as well as
registered gallery figures. Saved particles, actual catalogue overlays and exact-ID
interpolation remain the source; absent galaxies/subhalos are never invented. Halo
movies request the history analysis and consume its exact current
`analysis_products.h5`, through the selected analysis bundle. Native merger-tree data
are required to measure those histories.

When descendant-linked Rockstar or Consistent-Trees histories are available, the history
analysis also exports `merger_tree.html` and `merger_tree.json` in its results
directory. Open the HTML locally and select a node to inspect its halo ID, snapshot,
redshift, and virial mass. The view follows the largest selected final halo with a
measured main history and includes its linked progenitors across adjacent available
catalogues. It does not infer missing links or show halos excluded by the particle
floor; a node-limit warning marks any truncated display. IDs in the JSON are strings to
preserve exact values. To select another final halo, run `python commands.py
select-merger-tree RUN_DIR ANALYSIS_BUNDLE --halo-id EXACT_ID --output NEW_DIRECTORY`.
The ID is the exact ID in the final saved tree catalogue; it must pass the configured
particle floor and have at least one resolved progenitor. The selected-tree directory
can also replace the analysis bundle argument to `merger-tree-map`,
`halo-formation-movie`, and `halo-collision-movie`; those commands verify the selected
tree and its source analysis before use.

When at least two tracked halos have similar final masses, history analysis also creates
a formation atlas. It selects the most populated factor-of-two mass group and shows each
saved main branch alongside the interpolated half-mass crossing, where one is bracketed
by saved epochs.

Tree nodes include saved catalogue positions when available; selecting one in the HTML
shows its XY position in the box. `python commands.py halo-formation-movie
ANALYSIS_BUNDLE MAP_BUNDLE --output NEW_DIRECTORY --map-key KEY` animates a saved
density map beside the linked tree. The bundles must describe the same run and unchanged
source data. Markers use saved catalogue positions and respect the map's periodic slab.

```bash
python commands.py halo-collision-movie ANALYSIS_BUNDLE MAP_BUNDLE --output NEW_DIRECTORY --map-key KEY
```

This command selects two linked progenitor branches with at least two shared saved
positions and shows a periodic close-up, separation, and available peculiar velocities.
Use `--descendant-id ID` to select a particular merger. Both halos must be inside the
map slab at displayed epochs; this is a linked encounter view, not a collision-causality
measurement.

Analysis exports `snapshot_epochs.html` and `snapshot_epochs.csv`: a readable epoch
table and a full-precision CSV with actual and requested redshifts, scale factors,
available cosmic times and product statuses. Unavailable values are shown explicitly;
cosmic times follow the recorded run clock.

The halo-profile analysis also exports a radial phase-space portrait for the most
massive selected halo with adequate particle data. It shows sampled dark matter within
two R200c, with radial peculiar velocities relative to the weighted mean within R200c.
It does not add Hubble flow. The accompanying `*.phase_data.npz` contains the histogram
and bin edges.

### Native source overlays

For explicit full-file native source overlays, the current sealed inventory is
`core/patches/gadget4-source/native-source-current.json`. The older `native-source.json`
remains available for its historical audit references; it contains an older cooling
implementation. Overlay installation requires explicit prior hashes and does not itself
build or validate the simulation.

### Comparison movies

A published matched comparison-map bundle can be animated with `python commands.py
comparison-movie BUNDLE --output NEW_DIRECTORY --map-key KEY`. Choose `KEY` from the
bundle's `comparison_maps_summary.json`. The movie uses saved paired epochs in
chronological order and fixed comparison scales; each epoch gets equal display time
rather than an implied physical-time cadence. Matched epochs or random seeds alone do
not prove matching initial phases. Both source-run contracts are checked before and
after rendering under the run locks. Supply `--runs RUN_A RUN_B` if their paths cannot
be inferred.

### Timeline and multifield movies

For a single run, `python commands.py timeline-wall MAP_BUNDLE --output NEW_DIRECTORY
--map-key KEY` creates a PNG/PDF timeline of saved maps. `KEY` is the map-series key in
`scale.json`; the series must use global fixed normalization. When there are more
snapshots than panels, the figure samples the saved epochs evenly by index and records
which ones it shows. The map series and source-run provenance are checked under the run
lock; use `--run RUN_DIR` if the source run cannot be inferred from the saved contract.

```bash
python commands.py multifield-movie MAP_BUNDLE --output NEW_DIRECTORY --map-key dm_xy --map-key gas_xy
```

This command animates two to four globally normalized field series from one map bundle.
Use the exact keys in `scale.json`, including any slab suffix. Missing maps are labeled
in the movie; no missing field or intermediate state is invented. Temperature and
stellar panels need their respective saved fields and map products.

### Particle cohorts and camera paths

```bash
python commands.py particle-origin-movie RUN --output NEW_DIRECTORY --anchor-snapshot N --host-row ROW
```

This command traces a fixed deterministic sample of exact type-1 FOF member IDs through
saved snapshots. The output includes the movie, source hashes, and the exact matched IDs
and positions in HDF5. It uses equal display time per saved epoch and does not infer
motion between outputs; the trace represents material continuity, not persistent
halo-core identity.

```bash
python commands.py stellar-stream-movie RUN STELLAR_HISTORY_BUNDLE --output NEW_DIRECTORY
```

This command animates the fixed exact-ID candidate population from an existing
stellar-history product. Optional `--center X Y Z --width-kpc W` controls its fixed
comoving view. Membership colors distinguish particles in saved SUBFIND subhalos from
those outside them; being outside those subhalos does not establish that a star is
unbound from the host or has tidal origin.

```bash
python commands.py velocity-flow RUN --snapshot N --output NEW_DIRECTORY
```

This command creates a dark-matter surface-density map with mass-weighted
peculiar-velocity arrows. The saved numeric grid includes the particle count and support
mask; arrows are omitted below the declared count threshold. The velocity excludes
Hubble expansion and no smoothing is applied.

```bash
python commands.py allsky-map RUN --snapshot N --output NEW_DIRECTORY
```

This command creates an observer-centered all-sky matter map. Optional `--observer X Y
Z` uses comoving Mpc/h. The map bins represented mass per solid angle from one nearest
periodic copy, with the sampled IDs and numeric raster saved alongside the image. It is
a simulated matter distribution, not emitted light or a telescope observation.

`python commands.py cosmic-flythrough RUN --output NEW_DIRECTORY --snapshot N --host-row
ROW --start X Y Z` flies a camera toward a saved FOF center. Add `--waypoint X Y Z` as
needed to choose a route, in comoving Mpc/h. The camera moves through one fixed snapshot
with one nearest periodic copy of a sampled dark-matter population; the route does not
identify a filament automatically or imply particle evolution. Its perspective rasters,
particle sample, camera positions and display choices are retained.

```bash
python commands.py resolution-reveal RUN_A MAP_BUNDLE_A RUN_B MAP_BUNDLE_B --snapshot-a A --snapshot-b B --map-key KEY --output NEW_DIRECTORY
```

This command creates a shared-scale side-by-side figure and an interactive wipe for two
saved maps. The maps must share the field, projection, comoving box and slab, and their
redshifts must match within the specified tolerance. The view alone does not prove
matching initial phases, identical physics, or numerical convergence.

```bash
python commands.py filament-flythrough RUN_DIR --snapshot N --host-row ROW --output NEW_DIRECTORY
```

This command finds a connected 3-D route through the saved dark-matter density field's
periodic Hessian ridge classification, then flies along it toward the selected host. The
final segment connects the nearest classified ridge cell to the halo. This is a grid-,
smoothing-, and threshold-dependent filament proxy, not a persistence-filtered skeleton.
The camera moves through one saved epoch; it does not depict time evolution. Route cells
and the source contract are saved.

### Curated galleries

```bash
python commands.py results-gallery REQUEST.json --output NEW_DIRECTORY
```

This command builds an offline gallery from an explicit list of existing result files.
The request uses schema `visual-results-gallery-request-v1`, a title, intro, and `items`
with `title`, `category`, `description`, and `path`; optional `preview`, `contract`, and
`note` fields add an image, source record, and context. Paths may be absolute or
relative to the request file. The gallery links to the originals and records their
hashes at creation; it does not duplicate or change scientific outputs, and it does not
establish scientific acceptance. Use `python commands.py results-gallery
GALLERY_DIRECTORY --verify` to check the saved page and directly linked file hashes.
This does not check every dependency inside a linked interactive result.

### Tree-linked maps

```bash
python commands.py merger-tree-map RUN_DIR ANALYSIS_BUNDLE MAP_BUNDLE --map-key KEY --output NEW_DIRECTORY
```

This command connects the interactive merger tree to the corresponding saved density
maps. Selecting a halo displays the map from its actual saved epoch and marks the halo's
catalogue position when it lies inside the map slab. Missing maps, missing positions and
halos outside the slab are identified separately. The tree and maps must have verified
provenance from the same run.

### Metallicity maps

```bash
python commands.py metallicity-maps RUN_DIR MAP_BUNDLE --source-key gas_xy --output NEW_DIRECTORY
```

This command adds synchronized gas-metallicity maps to a fresh copy of a published map
bundle. It uses the saved dimensionless `PartType0/Metallicity` metal mass fraction and
deposits the CIC ratio `sum(m Z) / sum(m)`. The measured arrays retain zero metallicity;
the logarithmic movie labels an all-zero epoch explicitly. The generated map key can be
passed to `multifield-movie` alongside the original density and temperature keys.
Missing metallicity or a missing epoch is never replaced with a fabricated measurement.
The `multifield-movie` command verifies the source simulation provenance before and
after rendering and holds the run lock during publication. If the run cannot be inferred
from the map bundle, provide `--run RUN_DIR` explicitly.

### Matched halo fingerprints

```bash
python commands.py halo-fingerprint RUN_A RUN_B --snapshot-a N --snapshot-b M --host-row ROW --output NEW_DIRECTORY
```

This command finds the reciprocal-best counterpart to a selected FOF host using exact
shared dark-matter particle IDs. It exports matched host-centered maps, a
surface-density difference map, radial density profiles, satellite counts, and the
shared-ID fractions. Different runs require equal declared lattice, seed and box plus a
passing measured initial-phase agreement; the same-run case is labeled as a control.
Phase checks use the exact, checksum-verified initial-condition shards from the recorded
GADGET handoff when available. Maps use a common comoving cube and do not by themselves
demonstrate convergence or a causal model effect.

### Voids and particle continuity

```bash
python commands.py void-explorer RUN_DIR --output NEW_DIRECTORY
```

This command measures a periodic, mass-weighted dark-matter density mesh for every saved
snapshot, smooths it, and links the latest snapshot's deepest underdense minimum
backward to nearby minima. Its offline page contains centered density slices, an
underdense-radius history, and a fly-through of the latest saved snapshot. The link is a
proximity heuristic, not proven void identity; the radius is a spherical, smoothed-grid
proxy rather than a watershed boundary. Epochs that cannot be linked are marked
unavailable, and a radius that reaches the periodic-aperture limit is marked as a lower
bound. Grid, smoothing, threshold, matching distance, and movie controls are
configurable in `--help`.

```bash
python commands.py void-particle-history RUN_DIR VOID_BUNDLE --radius-mpc-h R --output NEW_DIRECTORY
```

This command adds an exact-ID continuity table and figure for equal-size periodic
spherical apertures centered on the matched minima. It reports both overlap fractions
and the Jaccard index, including missing anchor IDs. The requested radius must fit
within the supported density-based aperture at each displayed epoch; unsupported epochs
are marked. Shared particles provide continuity evidence but do not prove that the void
boundary, its filaments, or a watershed identity persist.

### Halo anatomy

```bash
python commands.py halo-anatomy RUN_DIR --snapshot N --host-row ROW --output NEW_DIRECTORY
```

This command creates an offline interactive depth cutaway for a native FOF host, with
catalogue subhalo centers, a projected R200c circle, and measured radial dark-matter
density and velocity-dispersion profiles. The cutaway uses all saved type-1 particles in
a 4 R200c-wide cube; radial profiles include every particle within 2 R200c, including
material not bound to the selected host. The selected aperture must fit inside half the
periodic box. Velocity dispersion is shown only for shells with at least ten particles.
The output includes numerical arrays and a source-bound provenance contract.

### Parameter exploration

```bash
python commands.py parameter-explorer REQUEST.json --output NEW_DIRECTORY
```

This command builds an offline selector for 2–30 explicitly listed saved runs. The
request uses schema `parameter-explorer-request-v1`, a dotted `parameter` such as
`cosmology.H0`, and a `runs` list whose entries each contain `run`, `map_bundle`,
`map_key`, `snapshot`, and `analysis_bundle`. Each view includes the saved map, run
information, and a halo mass function when the analysis has one. Other settings may
differ, so the view alone does not establish causation.

### Lensing maps

```bash
python commands.py lens-map RUN MAP_BUNDLE --output NEW_DIRECTORY --map-key KEY --snapshot N --lens-distance-kpc DL --source-distance-kpc DS --lens-source-distance-kpc DLS --background-msun-kpc2 SIGMA
```

This command produces a single-plane convergence and shear figure from a saved mass
projection. The three angular-diameter distances are explicit proper kpc inputs. The
periodic contrast and uniform mass-sheet assumption are recorded; this is not a full
light cone, instrument model, or observational-noise simulation.

### Planetarium movies

```bash
python commands.py planetarium-movie RUN --output NEW_DIRECTORY
```

This command creates a fixed-scale, 180-degree dome movie of saved snapshots. Use
repeated `--snapshot` options to choose epochs and `--observer X Y Z` for a comoving
Mpc/h position. Each epoch has equal display time; no intermediate simulation states,
emitted light, light cone, or projector warp are implied. Numeric dome rasters and their
source-particle samples are retained beside the movie. Each snapshot is sampled
independently, so individual displayed particles are not tracked.

### Portable single-epoch stream inspector

Export a sealed stream product to a fresh external directory, then copy that
inspector directory to a machine with Blender if desired. These commands are a
future usage example; the new path has not been runtime-verified.

```sh
python commands.py stream-products inspector /scratch/study/stream/product.json --output /scratch/study/stream-inspector --memory-gib 2
blender --background --python-exit-code 1 --python core/blender.py -- stream-inspector /scratch/study/stream-inspector/inspector.json --output /scratch/study/stream-inspector/stream.blend --maximum-points 100000 --marker-radius-kpc 0.01 --velocity-scale-kpc-per-kms 0.002 --maximum-velocity-vectors 200
```

Set the point and memory limits for the actual sample; `--maximum-points`
rejects larger samples rather than silently reducing them. The source report
metadata and selected arrays are included and hash-checked in new inspector
bundles. The original full numeric audit and simulation inputs are not copied:
retain those separately for scientific reproduction. Build the `.blend` after
transferring the inspector bundle, because object metadata records the load-time
archive path. Older bundles continue to use their original source-product path.

The optional velocity lines use saved km/s components multiplied by the explicit
display scale in proper kpc per km/s. They are not trajectory predictions or
measured displacements. Both vector options are required together; omit both
for points alone. The first IDs in exact uint64 order are displayed, which can
be a biased subsample. The line object stores these IDs as decimal strings and
records the scale. It does not infer a systemic velocity subtraction or frame;
consult the original product's frame metadata. Product schema identifies native
particles, resimulated particles, Gala or StreaMAX tracers where that provenance
is direct; derived or older products can remain `UNSPECIFIED`. This label is a
provenance description, not classification or scientific acceptance. Supported
track portions remain separated at unsupported bins.

### Display settings and failure reporting

Maps use the saved `visualization.map_fields` and `map_projections`. Density movies use
XY sequences. Planetarium exports one fresh directory containing all domemaster frames,
numeric rasters and particle samples, frame provenance, `timeline.json`, and
`planetarium.mp4`. Its default is 2048 square pixels, DM, 27-degree tilt and the saved
Blender particle budget (300000 if absent). Optional `visualization.planetarium`
settings are `field`, `resolution`, `max_particles`, `tilt`, and `observer_mpc_h`;
movies use `movie_fps`. These display defaults do not alter the simulation. Movie frame
spacing is uniform across saved outputs; provenance preserves each physical epoch.

Visual stages record exceptions separately and continue independent exports. The final
workflow remains `INCOMPLETE` and exits nonzero when requested stages fail, unless
`--allow-incomplete` was explicitly supplied. A failed scientific analysis also remains
recorded while independent visual stages are attempted. Files, logs and staged failures
are retained; prior successful exports are kept. A complete product report describes
generated outputs; scientific acceptance requires its own declared evidence.

### Blender compatibility

Blender compatibility paths support legacy and layered animation Actions, Blender 5
compositor node groups, and the image/video media-type switch. The exact binary particle
PLY reader preserves vertex order and every exported float attribute, validates payload
size and finiteness, and rejects corruption. AUTO Cycles device selection retains CPU
fallback when GPU discovery fails. Historical execution checks used Blender 4.5.1. The
workstation Blender 5.2 configuration and full 53-snapshot dataset still require
validation there. Full-resolution movies and hundreds of gallery previews can take
substantial time and disk.

## Output catalogue

See [the maintained output catalogue](outputs.md) for all single-run products, required
inputs, scientific figures and exports.

## Native cooling correction and verification

The primordial cooling code previously forced neutrality below its 10 K table boundary
while the cooling-rate evaluator retained table-edge photoionization. That inconsistent
molecular weight could oscillate or pass a small-step test without satisfying the gas
energy equation. Both paths now use the same endpoint rate coefficients. Actual
temperature and internal energy are not floored. The temperature closure uses a
physically bracketed, safeguarded solve with an energy-residual check; electron
convergence is tighter. The implicit energy solve also has bounded local bracket
searches and a residual acceptance test. It searches from the initial energy in the
direction of the net source, retaining the original thermal-branch selection policy.
Invalid states still stop with diagnostics; cooling is not disabled to continue.

Below the rate table, endpoint coefficients remain an approximation of this atomic
cooling model. The change does not introduce molecular cooling, modify BH seed
thresholds, raise BH count limits, or establish source-model convergence.

Run the compiled shipped-method controls with a **new** evidence directory:

```bash
python -m core.native_cooling_control ../research2-runtime/evidence/cooling-check
python verification.py NativeCoolingTests -v
```

The control compiles the shipped C++ rate-table, abundance, temperature and energy
methods. It checks 1,344 temperature/ionization states, 2,400 implicit steps with and
without cosmological Compton terms, an independent analytic cold-photoionization
balance, unit invariance and invalid-input rejection. The comparison test reproduces
failure using the retained upstream methods. Reports bind the source hashes and retain
compiler/execution output. A C++ compiler is required; these controls are distinct from
a full MPI simulation.

Fresh runs build a new executable because the cache key includes native source bytes.
Existing checkpoint continuation deliberately retains the original executable. Do not
replace it or rewrite checksums to force an old checkpoint onto a changed binary; that
needs an explicit validated continuation workflow.

## Terminal logging

Avoid nested `python -c` commands from PowerShell. Enter WSL once and use the public CLI
above. For a log with the correct pipeline exit status in Ubuntu:

```bash
set -o pipefail
mkdir -p ../research2-runtime/logs
python -u commands.py analysis /absolute/path/to/run --graphs available 2>&1 | tee ../research2-runtime/logs/analysis.log
```

## What connects, and what needs a separate experiment

Bundled software is a toolbox. Each backend serves a declared workflow; running every
executable in every simulation would mix incompatible estimators/models.

| Product or backend | Actual trigger / required input |
|---|---|
| CLASS / AxiCLASS | Selected cosmology in `run` or `class`; saved P(k), transfers, background and provenance |
| GADGET / IC generator | `gadget` or `run`, with the selected DM/hydro and native IC route |
| FOF / SUBFIND | Native host/subhalo catalogues for halo products |
| ROCKSTAR / Consistent Trees | `both` plus configured temporal-tree production; measured multi-epoch catalogues |
| VELOCIraptor | Separate `velociraptor` command or an authored campaign selecting it |
| HMF and structural statistics | Saved resolved host/subhalo catalogues and selected `--graphs` products |
| Power / correlation / bias | Saved compatible full-volume measurements and configured estimators |
| Hydro / stars / galaxies | Actual gas/star particles, supported units and selected source physics |
| Black-hole histories | Native BH physics enabled, saved source fields, `black_holes` product |
| DisPerSE / curvature proxy | Selected map/filament method and its native tools or field inputs |
| Grackle / stellar returns | Explicit enabled physical model, library/table inputs and matching native build |
| Stellar components / histories | Explicit host/search or exact-ID history JSON requests |
| Scientific zooms | `trace-zoom` / `zoom-campaign` or authored zoom workflow with region/resolution controls |
| Zoom contamination | `zoom_contamination` plus `--zoom-regions` request |
| Paired comparisons | `compare-runs` with two compatible completed runs |
| Inference / SBI / cosmography | Authored likelihood, data, prior, sampler or campaign inputs |
| AGAMA / Gala / StreaMAX | Selected reconstructed/static-host/stream experiment; not an automatic box-simulation stage |
| FSPS | Selected stellar-population synthesis, with compatible model assets |
| Blender | `blender` product or command, installed executable and supported saved epochs |
| Movies / dome frames | Selected single-run movie/planetarium outputs; advanced camera experiments use separate authored requests |

A camera zoom does not run a scientific zoom simulation. A small box does not
automatically resolve stars or BH source scales. `all` does not launch extra universes,
create posterior samples, or generate two scientific zooms.

Find domain entry points with `python commands.py --help`. The main campaign routes are
`native-prepare`, `workflow`, `zoom-campaign`, `streams`, and `flagship`; each has
`--help`. Preparation writes inputs and recipes; it does not establish that native
execution occurred.

## Additional HMF fitting functions

[core/hmf_fitting_functions.py](../core/hmf_fitting_functions.py) adds Press-Schechter,
Sheth-Tormen, Reed (2007), Watson FOF, Jenkins and Warren comparisons whenever
compatible `hmf` analysis runs. Existing HMF figures and exports retain their names.
Each compatible snapshot adds ten separate figures: abundance overlay, measurement/fit
residuals, fit/ST ratios, multiplicity, expected counts, cumulative finite-window
abundance, signed Poisson deviance, mass contribution, model spread and coverage. Two
CSV tables export their numerical values. Cumulative points are correlated and stop at
the highest exported mass edge; mass contribution uses bin-center masses. Deviance
residuals are diagnostics, not a fit acceptance test. CSV/JSON sidecars record
predictions, Poisson intervals, masks, assumptions and input hashes. Find them under
`hmf_fitting_functions/` inside the analysis bundle (or `results/` for the legacy core
route).

Predictions integrate over measured mass bins and use checksummed saved linear power and
growth tables. Missing k coverage is masked. FOF b=0.2 is required; SO masses are not
converted. Dashed segments identify extrapolation outside coordinate ranges in [HMFcalc
Table 1](https://arxiv.org/abs/1306.6721). EDE/WDM, hydro and SIDM uses are
extrapolations of CDM DMO fits; background growth rescaling assumes scale-independent
growth. Massive-neutrino runs are skipped. Model spread is not an uncertainty estimate.
FOF particle-mass and finite-box calibration corrections are not applied. Warren's mass
range is approximate in the code's Msun/h convention.

## Filesystem layout

```text
parent/
  research2/                      portable source
    commands.py                   public command dispatcher
    bootstrap.py                  dependency installation
    verification.py               grouped regression checks
    scripts/setup.sh              Ubuntu/WSL2 installation
    configs/                      dependency declarations and request templates
    docs/                         workflows, equations and generated references
    tests/                        application regression and standalone checks
    optional_tests/               runnable teaching controls, atlas and notebook
    core/                         physics, adapters, analysis and display
      software/                   bundled third-party source and physical assets
      patches/                    native integration payloads
      research/                   controls, checkpoints and campaigns
      stream/                     stream experiments and measurements
      music/                      IC / multilevel numerical infrastructure
  research2-runtime/              machine-specific writable state
    environments/default/         Linux or macOS Python environment
    builds/                       native executables / build receipts
    cache/                        reusable generated assets
    runs/                         cosmologies, snapshots, catalogues and results
    evidence/                     retained validation evidence
```

Keeping runtime state outside source is intentional: source archives remain portable,
native builds stay host-specific, and updating source need not move large simulation
datasets. Set `RESEARCH2_STATE_ROOT` to another **absolute, external** directory before
setup and future commands if desired. For a custom root, activate
`$RESEARCH2_STATE_ROOT/environments/default/bin/activate`. Explicit output paths can
override defaults.

The source layout is usable but still has technical debt: `commands.py`,
`core/analysis.py` and `verification.py` are large consolidated modules, and
`core/__init__.py` retains historical import aliases described by
`configs/core_source_layout.json`. A module-by-module migration is safer than moving all
files during a scientific bug fix. Keep bundled upstream sources, licenses, numerical
tables and patch manifests together.

Stored physical quantities follow their dataset unit contracts, including a/h scaling.
Particle IDs remain exact integers. GADGET stored velocities generally need the snapshot
scale-factor conversion. Coordinate, mass-definition, source and epoch conventions are
recorded with measurements; a missing convention is not supplied by a figure label.

## Check and package an update

These are optional transfer instructions. The current completion task does not
create an archive. Keep validation and archive generation as separate commands.
With the project environment activated:

```bash
python verification.py NativeCoolingTests WorkstationRegressionTests HMFFittingFunctionTests ModularCosmologyTests SavedUnitTests PortablePackageTests -v
python commands.py package --output ../research2-source-updated.zip
python commands.py package --verify ../research2-source-updated.zip
```

Use a new archive filename. Official packaging excludes local environments, compiled
binaries, caches and generated run outputs, and retains required
source/assets/licensing. Do not ZIP `.venv` for another operating system.

Unpack an update into a **new source location** while existing jobs are running; retain
their original code and environment. Do not overwrite active source, run data,
checkpoints or provenance records. Bind the new source to the chosen external runtime
root deliberately. Workstation-only patches must be compared and reconciled before
replacing that checkout.

Native continuation uses frozen run inputs, not a new interactive configuration:

```bash
python commands.py gadget --run-dir /absolute/path/to/run --resume
```

Prepared campaigns use their `workflow ... --resume` route. Native controls, convergence
studies, model validation and performance measurements remain separate work; no fixed
runtime estimate or end-to-end guarantee follows from these Python checks alone.
