# Optional observable lab

The `optional_tests/` folder is a local teaching interface and reproducible numerical
lab. It is optional: the simulation, analysis and ordinary verification commands do
not depend on its server, notebook or reference images.

It has three purposes:

- Run tiny calculations and show their inputs, Python implementations, equations,
  measured results, assertions, arrays and source hashes.
- Explain all 102 registered graph recipes and expose retained examples where actual
  producer-to-render evidence exists. Missing inputs remain visible.
- Help a new reader move between a figure, its units and numerical sidecar, the source
  code and a notebook that replays the calculation.

## Choose a profile

| Lab profile | Steps | Requirements beyond base analysis |
|---|---:|---|
| `quick` | 7 | None; 32 total synthetic particles by default |
| `visual` | 7 | Same calculations; choose `--particles 1024` for larger images |
| `native` | 8 | Installed GADGET build toolchain/MPI |
| `libraries` | 19 | Twelve installed optional library backends and their physical assets |
| `combined` | 20 | Library backends plus the native toolchain |
| `engines` | 31 | Combined prerequisites plus the eleven native-package workers below |
| `physics` | 10 | Native toolchain for three fixed gas/star/BH controls |
| `figures` | 10 | `halo-structure` dependency profile for shapes/splashback; flyby needs base only |
| `complete` | 37 | Engines, physics and figure-control prerequisites together |

Missing requirements are reported as `UNAVAILABLE`; a numerical or execution error
is `FAIL`. Neither status counts as a pass. Counts above are checklist steps, not
particle counts or all 102 graph recipes.

Follow [interface setup](#start-the-interface), [the seven measurements](#read-the-seven-quick-steps),
[CLI runs](#run-without-a-server), [notebook replay](#use-the-notebook),
[execution evidence](#read-per-file-execution-evidence) or
[portability/removal](#scope-portability-and-removal) as needed.

## Read per-file execution evidence

Each fresh run writes `python-calls.json` and includes the same record in
`run.json`. The source map distinguishes three kinds of evidence:

- **Listed in passed step:** a declared implementation link for a completed check.
- **Observed Python calls:** function names, source line numbers and call counts
  recorded in the parent runner's main thread during the measurements.
- **Archived regression calls:** the separately recorded tests and source revision
  from the retained regression campaign; fixtures or mocks may apply.

Imports and class definitions have separate counters. They are not counted as
function calls. Source hashes are taken before the run and compared afterward;
the detail panel also compares the captured revision with your current file.
Observed function links open the source at the recorded line number.
The recorder logs no function arguments. The normal lab entrypoint disables
bytecode writes so launching it does not recreate source-tree Python caches. Generator resumptions can contribute
multiple call events. This is execution evidence, not full branch coverage or
proof that every function in a file ran.

Native C/C++ code, other threads and the isolated library workers are outside
this parent-thread trace. Their actual execution is established by the native
campaign and worker receipts, rather than by inventing Python call counts.

## Start the interface

From the source root, use the installed Research2 environment:

```sh
source ../research2-runtime/environments/default/bin/activate
python optional_tests/lab.py serve
```

Open the printed `http://127.0.0.1:8765` URL. Use that exact host; `localhost` is not
an alias accepted by this local server. Choose another free port with `--port 8766`
if needed. The interface uses local CSS, JavaScript and images. It fetches no remote
fonts, scripts, datasets or videos.

The default **quick** lab uses **32 total particles**. It does not set the cosmological
`particles_per_dimension` option, where 32 would mean 32³ particles per generated
species. The visual lab defaults to 1,024 total particles; larger choices are bounded
at 8,192. Both profiles use explicitly synthetic inputs and the same numerical
measurements. Neither profile creates movies.

The native profile adds one real 32-particle GADGET self-gravity control. It
requires the native compiler/MPI toolchain and installed base dependencies. Its
numerical run is small, but a cold native build may take minutes. This profile does
not run CLASS, MUSIC, FSPS, Blender, or every other engine. The software page keeps
availability, fresh execution and archived evidence separate.

The **Quick + regenerate shape/splashback/flyby examples · 10 steps** option
(`--profile figures`) regenerates three advanced examples from shipped code.
It creates its own synthetic HDF5 and uses the production host/profile,
AGAMA reconstruction and registered figure pipelines. It does not need the
original campaign directories or archived native snapshots. The portable flyby was
also checked from `/tmp` in macOS and Ubuntu, with no AGAMA installed in the
Ubuntu image. It does not launch a native solver or require a halo finder.

```sh
python optional_tests/lab.py run --profile figures --particles 32 --seed 42 --output ../research2-runtime/optional-lab/my-figure-controls
python optional_tests/advanced_controls.py shapes --output ../research2-runtime/optional-lab/my-shapes --seed 42 --memory-gib 4
python optional_tests/advanced_controls.py splashback --output ../research2-runtime/optional-lab/my-splashback --seed 42 --memory-gib 4
python optional_tests/advanced_controls.py orbits --output ../research2-runtime/optional-lab/my-flyby --seed 42 --memory-gib 4
```

The CLI runner and server also reject output paths that resolve inside the
source tree, including symlink aliases, before creating their directories.

Use new external directories. For a standalone call outside the repository,
use the absolute script path. The worker locates `core/` and configuration
assets relative to its own file, and rejects outputs inside the source tree.
The standalone `orbits` control requires base NumPy/SciPy, h5py, YAML and plotting
dependencies. Shapes/splashback additionally require the installed `halo-structure`
profile and pinned AGAMA. On Ubuntu/WSL, install that profile
with `bash scripts/setup.sh --profile halo-structure --engines none` before
activating the environment. The worker itself installs/builds/downloads nothing.

| Fresh control | Fixed input and acceptance | Output |
|---|---|---|
| `shapes` | 6,000 authored points, b/a=0.70, c/a=0.45 and successive 0.15-rad rotations; original iteration/bootstrap support, ratio/twist recovery and field/reload checks | Original HDF5, structure JSON/NPZ, shape/profile figures and human report |
| `splashback` | 32,768 authored mass-quantile samples of a known density valley; original seven-bin GLS, covariance/feature support and preset radius/depth/width recovery | Original HDF5, analytic reference arrays, structure/environment products, slope/depth/width figure and human report |
| `orbits` | 32 particles in two authored apertures across three epochs; original graph/image/clock/Hermite intake and closed-form infall/pericentre recovery within 10⁻⁶ Myr | Original HDF5/catalogues, graph, image offsets, candidate events/brackets and figure |

These counts are separate from `--particles 32`, which still controls the quick
fixture. Each worker has a 900-second bound and an explicit 4-GiB estimated
workspace allocation in the interface; that is not a hard RSS limit. The
standalone `--memory-gib` option accepts 0.5–16; an insufficient allocation can
retain a failure after measurement, especially for the large environment JSON.
A parent worker allows 930 seconds for completion and receipts.

Open `index.html` in each control output, or the parent run's `report.html`.
`predeclared.json` records the input formula/geometry and recovery tolerances;
`receipt.json` records status, source hashes, measurements and artifact hashes.
All retained diagnostic images come from the original exporters. The notebook
replays physical units, shell tensors and the derivative/covariance identities
from these fresh outputs without rerunning AGAMA. The complete profile includes
these three controls; it remains a fixed walkthrough rather than every scientific
experiment or resolution campaign. Other advanced atlas examples still have
their individual original source/input requirements.

The **libraries** profile runs the seven quick steps plus twelve isolated
optional-backend calculations. It uses the active Python environment and
locally installed source/data profiles. Each child has a 120-second time
limit, an independent log, a finite JSON receipt and its actual numerical
arrays. Missing dependencies report `UNAVAILABLE`; other errors report `FAIL`,
and subsequent checks continue. Cancellation stops the owned process group.
This profile installs nothing and creates no videos. SBI training-event logs are
saved beneath the external run directory and included in its artifact hashes;
the check does not create `sbi-logs` in the source folder.

Choose **Quick + twelve libraries + native GADGET · 20 steps** to run both
families in one checklist. The `combined` profile executes seven quick
measurements, one actual 32-particle GADGET self-gravity control, and all twelve
library calculations. It requires the installed optional libraries and native
compiler/MPI toolchain; a cold GADGET build can take minutes. Later checks
continue after an individual failure, and missing Python dependencies remain
`UNAVAILABLE`. The native control always contains 32 particles, even if the
quick measurements use a larger selected fixture. Other native engines retain
their separate campaign requirements and are not run by this profile.

The native step shows its measured force, energy and angular-momentum relative
errors beside their declared thresholds. It reads actual output times from the
verified numerical archive. Those times use the isolated control's native
length/velocity clock, not Myr or cosmological scale factor. Its saved analysis
receipt and NPZ expose the original normalizations, reference floors and array
contracts. GADGET and MPI appear separately in the software evidence table.

```sh
python optional_tests/lab.py run --profile combined --particles 32 --output ../research2-runtime/optional-lab/my-combined-32
```

The **Native gas, stellar births and BH accretion · 10 steps** choice
(`--profile physics`) adds three real GADGET source controls to the seven
quick numerical steps. The **Complete · 37 steps** choice (`--profile complete`)
combines the 31-step engines profile below, these three physics controls, and
the three known-input figure controls above.
Each physics worker allows 900 seconds including a cold build, with a
120-second bound on its native numerical run; the parent allows 930 seconds
for worker shutdown and receipts. These profiles reuse the installed toolchain
and do not install dependencies.

```sh
python optional_tests/lab.py run --profile physics --particles 32 --seed 42 --output ../research2-runtime/optional-lab/my-physics-32
python optional_tests/lab.py run --profile complete --particles 32 --seed 42 --output ../research2-runtime/optional-lab/my-complete-32
```

Use a new output directory for each command. The selected `--particles 32`
controls the quick fixture; each physics control has a fixed 4³ = **64 gas
particles**, and the BH control adds **one explicitly seeded 10-solar-mass BH**.
The stellar control starts with no stars. Its seed is passed to both the IC
configuration and native stochastic birth operator. Different seeds may change
the number of births.

| Physics control | Measurement and interpretation | Retained evidence |
|---|---|---|
| Cooling gas | Three isolated native epochs; constant mass and IDs, positive gas energy, and a decrease in mass-weighted specific energy | Original HDF5 snapshots, `arrays.npz`, `physics.png` and `receipt.json` |
| Native stellar births | Stars must actually form with `SSPBirthMode=1`; the production history audit checks mass and momentum with relative tolerance 10⁻⁸ and absolute floors 10⁻⁶ M☉ and M☉ km/s | Original birth/parent IDs, census, `history.json`, numerical arrays and figure |
| Seeded BH accretion | Positive physical BH mass growth and inflow in M☉/year; original gas IC is retained, and a separately hashed compressed IC supplies the accretion fixture | Compression transform, original/executed IC hashes, native snapshots, physical mass/inflow arrays and figure |

The stellar run's internal energy is an effective multiphase EOS quantity;
it is not a literal single-temperature cooling curve. Delayed stellar returns
are disabled in this control. The BH run uses a toy all-baryon background
Ωm = Ωb = 0.3, h = 0.7 and a = 0.5 to satisfy its native mean-density constraint.
It tests accretion onto an existing seed, not seed formation, and does not
reconcile the full BH feedback/source budget. Physical BH mass and gravitational
particle mass are distinct. The notebook shows the relevant equations, units,
raw arrays and source chain for each control.

The retained macOS complete run passed **36 of 37 steps in 171.649 seconds**;
only the local Linux-only OpenGadget3 build was unavailable. That independent
32-particle scattering worker passed separately in Linux. Its outputs can be
selected for notebook replay with `RESEARCH2_LAB_SIDM` as described below.

The **Combined + native engines, trees, topology and independent SIDM · 31 steps** option adds eleven
native-package workers to that checklist (`--profile engines`). It does not
install or compile these extra packages. Build the selected engines through
setup (`class,axiclass,monofonic,music2,rockstar,velociraptor,consistent-trees,disperse`) and install
Blender separately. The IC route also requires MPI and its FFTW
libraries; the Ubuntu setup script includes them. Missing executables stay `UNAVAILABLE`; stale
or invalid installations fail with their original diagnostics. Each worker
has a 180-second budget and retains its logs and outputs outside the source.

For Grackle, expose an existing shared-library installation before launching
this profile (including when launching the dashboard):

```sh
export PKG_CONFIG_PATH="/absolute/path/to/grackle-prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
pkg-config --modversion grackle
python optional_tests/lab.py serve
```

The version must be 3.4.1, with installed public headers and a C compiler
(`cc`, or the executable named by `CC`). The worker compiles only
[`native_grackle.c`](../optional_tests/native_grackle.c) into the new external
run directory and links it against that shared library. It never installs or
builds Grackle. Missing prerequisites are `UNAVAILABLE`; compilation/native
errors are `FAIL` with retained logs. The control writes two 32-cell epochs in
cgs units. Grackle's cooling time is signed `u/(du/dt)`, negative for net cooling;
use its absolute value as a timescale. H/He mass-fraction sums are checked against
0.76/0.24 with absolute tolerance 1e-10. This atomic control requires no cooling
table; the separate native GADGET chemistry campaigns retain their network,
table and stellar-model requirements.

| Native package | Actual calculation | Inspect these outputs |
|---|---|---|
| CLASS | Production reusable-bundle driver runs LCDM linear spectra at z=49 and z=0, with k coverage through 2 h/Mpc | Sealed `spectrum.json`, raw solver inputs/outputs, power/background tables, actual arrays and power figure |
| AxiCLASS | Same route for an explicit axion EDE model: f_EDE=0.05, z_c=3500, theta_i=2.8 rad, n=3 | Separate EDE bundle, source/executable hashes, physical unit conventions and figures |
| Blender | Production materials, point geometry and lighting/camera helpers build 32 seeded display points; CPU Cycles renders one 256×192 PNG at four samples | Editable `particles.blend`, `preview.png`, original coordinates and invocation receipts; a fresh Blender process reopens the saved scene and checks all 32 coordinates |
| monofonIC | Fresh CLASS transfers/background feed the production two-species 1LPT generator; 4³ gas plus 4³ DM particles | Original HDF5, staging/configuration/logs, finite coordinates/velocities, bounds, 128 unique IDs, cosmological gas fraction, numerical arrays and XY plot |
| Grackle | Installed shared Grackle 3.4.1 evolves 32 constant-density atomic H/He cells for 1e11 seconds, without UV, metals, molecules or dust | Original cgs CSV/NPZ, compile/native invocation receipts, signed cooling times, thermal/ionization plot and H/He mass-conservation residuals |
| MUSIC | Production native transfer-control CLI runs ten uniform/nested/periodic two-fluid 1LPT realizations and seven expected input rejections | Original configurations/transfers/HDF5/logs, normalization and conservation audit, nested 10,753-particle arrays and full/central XY views |
| Rockstar | Production periodic finder plus native parent-label utility reads two explicit-unit 32-DM-particle epochs; eight available CPU slots required | Native catalogues, per-epoch intake, exact IDs/membership when available, saved row counts and diagnostic image |
| VELOCIraptor | Production native driver reads the final 32-DM fixture, with reversible dense ID transport, eight-neighbor controls, no substructure search or unbinding | Sealed native report and original artifacts, actual object/membership intake, array archive and diagnostic image |
| Consistent Trees | Production tree construction from two analytic 128-object catalogues at a=0.99 and 1.0 | Actual hlists, 256 native nodes/128 forward links, exact IDs, array archive and track diagnostic; no particles generated |
| DisPerSE | Production CIC/log-density, native mse/skelconv and periodic skeleton projection from 64 DM samples on a 16³ mesh | Sealed native skeleton/topology/FITS products, original arrays and XY occupancy; one-cell density and one-iteration display smoothing |
| OpenGadget3 | Explicit existing Linux rare/directed build evolves 32 physical beam particles at three epochs with gravity disabled and a high-rate toy elastic law | Original population, model, verified build/input lineage, native snapshots, exact-ID arrays, mass/momentum/energy residuals and a velocity-change plot; missing builds are UNAVAILABLE |

The two Boltzmann solvers evolve linear fields, not particles. The Blender
check uses arbitrary display coordinates and an explicitly enlarged glyph
radius; it is not stellar formation, calibrated luminosity or a native
snapshot export. It produces no video. The IC control independently runs fresh
CLASS transfers and monofonIC with 64 gas plus 64 DM particles. Its count is
fixed, separate from the selected quick-fixture count. Grackle cells are also
independent of this count. MUSIC uses analytic synthetic transfer functions
for normalization, with 8,192 particles per uniform realization and 10,753 per
nested/periodic realization. Its coarse boundary particles represent combined
matter. The nested picture shows the actual native particle load, not a
resolved target halo. The finder fixtures repeat the same synthetic coordinates
and stored velocities at a=0.5 and a=1.0; they do not represent particle
evolution. Rockstar reads both epochs; VELOCIraptor reads a=1.0. Objects may
be absent. Their mass definitions/settings differ, so these are execution
checks, not halo matching or finder-agreement measurements. Independent SIDM
checks require an existing build: set `RESEARCH2_OPENGADGET3_BUILD` to its
completed directory containing `build-capability.json` and
`source/Config.research.sh`. See [software](software.md#independent-opengadget3-control).
Consistent Trees uses analytic catalogue inputs rather than the sparse
32-particle finder output. DisPerSE needs at least 64 DM particles; its 64-sample
fixture meets that production requirement. Their original links and skeleton
outputs are replayed in a dedicated notebook section. The 18-case numerical gravity matrix and dedicated gas/star/BH/SIDM controls
remain separate campaigns outside this fixed quick profile.

```sh
python optional_tests/lab.py run --profile engines --particles 32 --output ../research2-runtime/optional-lab/my-native-packages
```

| Backend check | Actual calculation | Saved result and interpretation |
|---|---|---|
| Agama | Pinned physical-unit Plummer force at 1 proper kpc for 1 Msun and scale 1 proper kpc; compare with −G/(2√2) | Position and acceleration in proper kpc and (km/s)²/kpc |
| Gala | Short seeded spray in a static Hernquist host | Proper positions/velocities and release arrays; no native star formation |
| StreaMAX | 32 generated particles over eight release epochs in its static Plummer reference | Original float32 phase-space arrays in the adapter's physical convention |
| FSPS | 1-Gyr SSP spectrum and SDSS g/r magnitudes using pinned data | Wavelength in angstrom, spectrum in backend SSP normalization, magnitudes and a spectrum PNG |
| emcee | Eight walkers, ten steps on a one-parameter Gaussian likelihood | Chain and log probability; convergence is untested |
| dynesty | Short Gaussian nested-sampling calculation | Samples and log-evidence history; iteration-limited accuracy is untested |
| cloudpickle | Serialize a function, reload it and evaluate f(3)=9 | Actual serialized bytes and recovered value |
| SBI | Train the real density estimator on 32 noisy toy forward samples | Input parameters/observations and model weights; no calibration or deployment claim |
| lenstronomy | Four SIS deflections with Einstein radius 1 arcsec | Ray coordinates and deflections in arcsec, compared with the exact unit-radius result |
| PyVista | Render 32 seeded points offscreen | Coordinates and decoded PNG; geometric display units |
| pyrecon | Reconstruct 32 tracers with 128 randoms on an 8³ mesh | Original/shifted positions in comoving Mpc; no BAO detection or accuracy claim |
| pycorr / Corrfunc | Count pairs and estimate six separation/angle bins for the same small uniform geometry | Dimensionless ξ(s,μ), separation bins in Mpc and angular bins; no BAO fit |

The Software & Evidence page has a search field for archived receipts. It searches
backend names, scope, status and nested recorded calculation/output fields;
matching counts stay visible. These remain original-workstation receipts, not
fresh tests of your environment.

Library inputs are separate explicit controls. Increasing the quick lab's
particle count does not increase the fixed stream, sampler or reconstruction
budgets in this profile. The native CLASS/AxiCLASS, IC, chemistry, finder and
Blender campaigns retain their separate engine/input requirements and archived
receipts; the libraries profile does not execute those native engines.

## Read the seven quick steps

| Step | What actually happens | Evidence |
|---|---|---|
| Fixture | Seed a truncated Plummer-shaped geometry, velocities, positive masses and exact IDs across gas, DM, stellar and BH marker species. Write explicit isolated physical units. | `synthetic_snapshot.hdf5`, `particles.json` |
| Units | Read each species through `core.data.read_species_phase_space`. Check every original ID, position, velocity and mass. IDs exceed 2⁶⁰ to catch float identity loss. | Per-step assertions and implementation hash in `run.json` |
| Density | Sum masses in spherical shells and divide by measured shell volumes. Project pixel mass and divide by pixel area. Verify mass closure and production NGP count conservation. | `density_arrays.npz`, `density.png` |
| Power | Call `core.analysis.fft_power_spectrum`, retaining its NGP mesh, rFFT multiplicity and volume normalization. | `power_arrays.npz`, `power.png` |
| Viewing zoom | Measure 100, 20 and 5 proper-kpc projected views from the same particles. Check selected mass against pixel-area integrals. This adds no particles and is not MUSIC IC refinement. | `zoom_arrays.npz`, `zoom.png` |
| Orbits | Integrate exactly N tracers through a fixed isochrone potential with the production DOP853 stream integrator. Check finite outputs and energy drift on nine saved epochs. | `orbit_arrays.npz`, `orbits.png` |
| Covariance | Call the production mass-weighted finite-population velocity covariance and check the declared finite-set definition. | `velocity_arrays.npz` |

A synthetic stellar marker is not a formed SSP. A BH marker is not a source-certified
accreting or merging BH. The tracer arc is an imposed-potential calculation without
self-gravity. The dashboard illustration is geometric artwork, separately labeled
from the numerical runs. These distinctions keep the walkthrough useful without
misrepresenting a tiny fixture as astrophysical evolution.

The shell density uses Msun and proper kpc. The optional lab FFT uses inverse proper
kpc and proper kpc³; native cosmological graph products have their own Mpc/h and
physical Mpc conventions. The lab NGP spectrum includes sampling/assignment effects
and does not substitute for GADGET's folded, CIC-deconvolved estimator.

## Run without a server

Choose a new output directory; earlier runs are never overwritten:

```sh
python optional_tests/lab.py run --particles 32 --output ../research2-runtime/optional-lab/my-first-32
python optional_tests/lab.py run --profile visual --particles 1024 --output ../research2-runtime/optional-lab/my-first-1024
python optional_tests/lab.py run --profile native --particles 32 --output ../research2-runtime/optional-lab/my-native-32
python optional_tests/lab.py run --profile libraries --particles 32 --output ../research2-runtime/optional-lab/my-library-checks
```

Open the resulting `report.html` directly for an offline report. `run.json` gives
step status, measured quantities, source function/line/hash, time and artifact hashes.
`FAIL`, `UNAVAILABLE`, and cancellation are distinct outcomes. A successful subset
never becomes an “everything passed” claim.

The source map inventories first-party application, verification and optional lab
Python files. It reads source docstrings and the [reviewed API explanations](python-api-notes.md)
without importing solvers. Reviewed descriptions must match their implementation's
syntax-tree hash; stale bindings stop catalogue generation. Select a public function
to open its specific explanation at the top of the detail panel, above the source.
A file changed since the inventory
was loaded requires a refresh before its explanation is shown. Missing descriptions
stay explicitly undocumented.

“Called in this
run” records the listed implementation calls; it does not imply full file or branch
coverage. Where a retained regression trace exists, the source map has a separate
archived badge. Open the file to see function call counts, recorded test IDs and the
original trace checksum. Historical calls can come from fixtures or mocks. Their
line numbers refer to the traced run. When a source checksum was recorded at execution,
the detail panel reports whether it matches the current file. Calls can include
failing tests, so use the software summaries to inspect test outcomes. Vendored software belongs to its backend's source and license inventory.

## Use the notebook

The generated notebook is `optional_tests/reference_gallery/Walkthrough.ipynb`.
Download it from the interface or open it in a Jupyter-capable editor using the same
Research2 Python environment. A Jupyter frontend is optional and is not installed
by the lab. If needed, install the frontend and kernel into your activated
external Python environment:

```sh
python -m pip install jupyterlab ipykernel
```

Then launch the notebook from the repository root:

```sh
python -m jupyter lab optional_tests/reference_gallery/Walkthrough.ipynb
```

It locates the repository from its working directory, selects the newest optional
run, replays density from the actual saved shell masses/volumes, inspects FFT
normalization, checks the saved orbit energy drift and opens original figure contracts.
Set `RESEARCH2_LAB_RUN` to an explicit results directory to choose another run.
For a CLI run this is the directory passed to `--output`. For a dashboard job it
is the job's `results/` directory, containing `run.json`, rather than the server's
whole output root. For example, set it before starting Jupyter:

```sh
export RESEARCH2_LAB_RUN="/absolute/path/to/COMPLETED_RUN"
```

Set `RESEARCH2_LAB_FIGURE` to a `family/variant` pair to inspect a specific
archived figure in the final cell. The default opens the impact geometry control.
The cell checks image, array and manifest checksums, displays the saved image,
and prints the original axis labels and each array's shape, type and numeric range.
The implementation chain links the producer and renderer to their source locations.
When the selected run used the libraries profile, another cell verifies each
backend receipt and numerical archive against `run.json`, prints the actual
array shapes/types and scope, and displays the separately calculated FSPS SSP
spectrum. These checks read saved outputs without retraining or rerunning engines.

For a completed native or combined run, the native cell additionally verifies
the original report and NPZ hashes, recomputes total energy and angular momentum
from saved particle arrays, and plots the original energy drift. The notebook
has 155 cells, including eighteen executable cells. All eighteen passed on the
retained 37-step macOS complete run plus separately completed Linux worker selected with
`RESEARCH2_LAB_SIDM`. The macOS run has 36 passes and an explicit missing
OpenGadget3 build; the Linux worker supplies its actual scattering outputs. The native equations use the isolated control's declared units rather than assuming a cosmological clock. The solver
cell verifies the original CLASS/AxiCLASS bundles and replays their dimensionless
power handoff; the Blender section checks the saved scene verification receipt
and displays the original still. It also checks the 128-particle monofonIC
archive, unique IDs and gas fraction. A dedicated tree/topology section checks
the original native hlists and forward links, then reconstructs DisPerSE
occupancy from its saved periodic polylines. It reads these outputs without
starting another solver or renderer. The gas/star/BH section additionally
checks original HDF5 and array hashes, recomputes mass-weighted gas energy,
replays the stellar history audit, inspects native birth IDs and plots the
measured BH physical mass gain. In the retained seed-42 run, gas specific energy
decreased by about 5.64%, nine stars formed from an initially all-gas population,
and the seeded BH gained about 2.481 × 10⁻⁶ M☉. These tiny measured responses
establish execution of the selected controls, not resolved astrophysical accuracy.

The gas-phase atlas example comes from the actual 64-particle Grackle cloud.
Its horizontal axis is proper density in g/cm³, its vertical axis is native
single-phase temperature in K, and color is the sum of gas mass in M☉ per
physical bin. Color is neither probability density nor the number of particles.
Broad fixed bin edges make this tiny uniform control occupy very few bins;
they do not imply a broad thermodynamic distribution. The exporter excludes
star-forming effective-EOS gas and retains explicit out-of-range counts.
The notebook's figure-selection cell prints the original labels and arrays.

The shape examples include a 6,000-point known triaxial geometry, with
b/a = 0.70, c/a = 0.45 and adjacent 0.15-radian rotations about z. Production
shell measurements recover the ratios and about 8.594° major/intermediate-axis
twists. A distinct periodic-filament control adds 8,448 background carriers;
only the original 6,000 type-1 samples define its selected host shape/spin.
The background supplies a filament environment on a 33³ mesh. Full-range
alignment views use a cosine axis from 0 to 1, preserving the original arrays
and masks. Their near-perfect alignment is specified by this test geometry,
not discovered in a simulated galaxy.

The two-epoch web-history example uses 14,448 particles per authored epoch,
explicit synthetic identity correspondence and the toy clock
t(a) = 1 + 2(a − 0.5) Gyr, H(a) = 1/(2a Gyr), over
0.5 ≤ a ≤ 0.500005. Its six clock nodes provide a conditional interpolation
error envelope; the selected maximum interval is 0.02 Myr. The measured local
filament axis stays fixed, so the supported plotted change is zero. This is a
clock/identity/measurement walkthrough, not a dynamical or cosmological result.
The earlier isolated knot and unbounded-clock attempts remain recorded as
unsupported controls. See the [shape and environment reading guide](learning-guide.md#read-shape-and-environment-measurements-from-their-arrays)
for the equations, file responsibilities and how to interpret masks.

Three assembly examples now use eight authored epochs of 8,192 DM and 2,048
stellar particles each. The production profile/fit functions recover the
prescribed NFW scale radii, measure enclosed stellar masses and stellar mass
tensors, and pass those measurements through the original graph/assembly
estimators. The SFR is zero from a complete empty gas census. Changing particle
masses and scale radii are prescribed inputs, not a simulated formation history
or mass-conserving evolution. The field grid extends to 40 proper kpc to contain
its calibration and withheld validation queries; original coefficient replay
checks pass at all epochs. The earlier failed exterior replay is retained.

The orbital-event example is smaller: 32 authored particles, two aperture
centres and three epochs. A straight flyby with impact parameter 0.5 kpc and
speed 0.4 kpc/Myr crosses a fixed 1.5-kpc test boundary at
1001.464466094 Myr and reaches pericentre at 1005 Myr. The production graph,
clock, image-offset and Hermite-polynomial event chain recovers those times.
Its event brackets and interpolation-uncertified status remain visible. There
is no physical merger or native finder/ancestry measurement in this control.
See the [assembly reading guide](learning-guide.md#assembly-histories-identity-first-then-measurements)
for equations, commands and the per-file calculation chain.

The splashback example uses 32,768 deterministic mass-quantile samples from
a known analytic density profile, with a seeded permutation over spherical
directions. Its prescribed slope minimum is at 1 kpc, depth is 3 and
half-prominence log-radius width is about 0.471. Production readers, profiles,
GLS derivatives, covariance propagation, feature checks and renderer recover
0.9655 kpc, 2.894 and 0.4908. The preset primary derivative window is seven
bins; the five-bin alternative remains uncertainty-unresolved, and the
nine-bin alternative is retained separately. The control samples 0.2–4 kpc
and searches only 0.5–2 kpc; the inner truncation is not a second physical
feature. It is an authored recovery test, not an evolved halo.

Its full external environment metadata require an explicit 4-GiB export
allocation under the production workspace guard. The original 1-GiB rejection
is retained. The teaching folder contains the original rendered figure,
plotted numerical sidecar and figure contract, rather than all of that run's
native-format inputs. See the [splashback reading guide](learning-guide.md#splashback-diagnostic-density-slope-then-feature-support)
for units, covariance, prominence/width definitions and the Python chain.

The action-change example comes from a fresh pair of actual 32-particle
native stellar self-gravity runs, each with total carrier mass 10⁴ M☉ and
identical initial velocities [0.01, 0.05, 0.02] km/s. The native gravity constant
matches the declared physical units. Only stellar softening differs (0.01
versus 0.30 kpc); actual input/state hashes and all three saved epochs remain
paired. A frozen lmax=mmax=0 projection of the initial particles defines a
shared, explicitly spherical diagnostic potential. It is not the full native
nonspherical force, so its actions need not be conserved along those trajectories.

All 32 particles have action support at 0, 3.9111689 and 7.8223378 Myr.
The three plotted curves are the baseline-centred mean squares for Jr, Jz and
Jphi, in proper kpc² (km/s)². The final values are approximately
[2.4103×10⁻¹⁰, 1.0089×10⁻¹¹, 8.5428×10⁻¹⁰]. Independent AGAMA spherical
actions agree with the radial-quadrature curves to about 3.63×10⁻⁸ relative
over all 192 branch/epoch/particle states. A separate reading view gives the
components physical names while retaining the original arrays and masks.
This is a numerical force-resolution comparison, not a stationary diffusion
coefficient, astrophysical stream detection or unique perturber identification.
The [paired-action reading guide](learning-guide.md#paired-action-changes-what-the-diffusion-labelled-plot-measures)
shows the calculation, masks and differences between the saved arrays.

All 102 recipe entries show their exact registered input requirements. Archived
PNG/NPZ examples are actual original outputs, not newly generated replacements.

Rebuild the notebook and source inventory after editing the lab:

```sh
python optional_tests/build_assets.py
```

Only the campaign owner needs the optional `--campaign EVIDENCE_DIRECTORY` import.
It copies one original figure, its actual plotted arrays and its figure contract per
available recipe. It checks the original numerical sidecar hash, includes no movies,
and leaves unexecuted recipes empty. Archived metadata retain original workstation
paths for lineage; the original native inputs are not all included in this teaching
folder. Replaying a native producer still requires its actual inputs.

## Scope, portability and removal

The interface binds only to IPv4 loopback and accepts fixed lab profiles, not arbitrary
shell commands. Outputs stay under the selected external `--output-root`. Run and
cancel actions require the current server token. Stop the server with Ctrl-C; it
terminates its own active job group and retains the outputs.

The ordinary source packager excludes notebook and generated media formats.
For transfer, keep the teaching folder beside the matching Research2 source;
it needs that source and environment. A future optional archive can be prepared
through `python optional_tests/package.py --help`; creating an archive is not
needed to run or review the lab. No reference image certifies a recipient's machine.

The quick CLI/server and notebook were also checked in a relocated full checkout:
two fresh 32-particle runs passed, all bundled example assets matched their hashes,
and all 18 notebook code cells executed with explicit skips for absent backend
outputs. That check reused installed dependencies. On another machine, install the
required environment first. Outside the source root, use the absolute lab-script
path; launch Jupyter from the copied root and set `RESEARCH2_LAB_RUN` to the chosen
external results directory. Full native/optional installation has separate checks.

After review, remove individual optional runtime run directories, or remove the
optional teaching folder itself. Existing native build caches and research runs have
separate ownership; the lab does not delete them. See the [retained campaign report](smoke-test.md)
for the broader native/software evidence and remaining graph coverage gaps.
