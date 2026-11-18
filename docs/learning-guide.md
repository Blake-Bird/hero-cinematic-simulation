# Learn the workflows and inspect what they produced

Start here when you want to understand the software rather than copy commands. The
[operator guide](operator-guide.md) gives the longer command sequences;
[command reference](command-reference.md) lists the actual parser arguments;
[Python reference](python-reference.md) links every application file and its public
signatures. The [architecture source map](architecture.md#python-source-map) identifies
callers. These are complementary: a library helper is usually called by a workflow,
not launched as `python core/something.py`.

To regenerate the known-shape, density-valley and analytic flyby demonstrations, use the
optional `figures` profile or the standalone commands in the [lab guide](optional-tests.md#start-the-interface).
[`advanced_controls.py`](../optional_tests/advanced_controls.py) creates its own
explicit-unit HDF5 and calls the same production readers and exporters described
below. Its reports contain inputs, preset recovery criteria, arrays and original
images. These controls need installed dependencies but no author-specific campaign
folder. They are separate from an evolved cosmological simulation.

## Find the walkthrough you need

| Goal | Start here |
|---|---|
| Try a tiny calculation first | [32-particle walkthrough](#start-with-a-32-particle-walkthrough) |
| Install and complete a first simulation | [First successful run](#your-first-successful-run) |
| Understand each software package's role | [Inputs, implementation and outputs](#what-goes-in-what-the-software-does-and-what-comes-out) |
| Trace a parent selection into a refinement | [Zoom checks and their evidence](#follow-the-zoom-checks-in-the-order-their-evidence-allows) |
| Interpret shape ratios, twists and alignments | [Shape and environment arrays](#read-shape-and-environment-measurements-from-their-arrays) |
| Read a density-slope feature | [Splashback diagnostic](#splashback-diagnostic-density-slope-then-feature-support) |
| Follow mass, stars and orbital events across epochs | [Assembly histories](#assembly-histories-identity-first-then-measurements) |
| Understand the action-change curves | [Paired action changes](#paired-action-changes-what-the-diffusion-labelled-plot-measures) |
| Decide whether a result supports your conclusion | [Evaluate an output](#how-to-decide-whether-an-output-is-useful) |

## Start with a 32-particle walkthrough

After installing and activating the Research2 environment, run this from the
repository root. Choose a new output name if it already exists:

```sh
python optional_tests/lab.py run --profile quick --particles 32 --seed 42 --output ../research2-runtime/optional-lab/first-quick-32
```

The terminal prints the status, report path and each step's result. With the base
dependencies installed, expect seven passing numerical steps and `COMPLETE`.
Open `../research2-runtime/optional-lab/first-quick-32/report.html` to inspect
the original pictures and measurements. This exercise creates synthetic HDF5;
it does not launch GADGET or generate a cosmological initial condition.

| Inspect | What it answers |
|---|---|
| `run.json` | Which seven steps ran, their results, software states and artifact hashes |
| `density_arrays.npz` | Which shell masses and volumes produced the density curve |
| `zoom_arrays.npz` | Which existing particles and pixel-area integrals produced the close views |
| `python-calls.json` | Which parent-thread Python functions ran, with source hashes; child/native execution has separate scope |
| `report.html` | A portable visual report with the original generated images |

For an interface, use `python optional_tests/lab.py serve` and open its printed
`http://127.0.0.1:8765` address. The [lab guide](optional-tests.md) covers other
profiles and dependencies. Its **fresh controls** regenerate their inputs and
results on your machine; the **archived atlas** shows original saved examples
with their arrays and source records. Viewing an archived image does not run its
producer. The **notebook replay** reads those numerical results and checks the
explained calculations; it does not silently start a simulation.

## Your first successful run

1. Install from the source folder using `bash scripts/setup.sh`, then activate
   `source ../research2-runtime/environments/default/bin/activate`.
2. Run `python commands.py run --count 1 --draft /absolute/path/to/new-setup.json`.
   Read the units beside each prompt. Choose a small cubic particle grid for an
   execution test, a short redshift interval, and a few saved epochs. The native
   generator's particle count is **N cubed**; entering 32 means 32,768 particles per
   generated species, not 32 total. A hydro load can contain both DM and gas.
3. Start with collisionless matter and ready outputs. Add cooling, star formation,
   black holes, chemistry, and synthetic light through their individual settings
   after the simpler run works. Each additional model has assets and parameters.
4. Keep the printed run path. Inspect its `config.yaml`, `manifest.json`,
   `input/param.txt`, `input/output_times.txt`, native log, and `data/` snapshots.
5. Inspect the analysis plan with
   `python commands.py analysis /absolute/path/to/RUN --graphs available --dry-run`.
   Execute the same command without `--dry-run`, then run
   `python commands.py results /absolute/path/to/RUN --verify`.
6. Open `RUN/results/index.html`. Select the latest attempt, inspect Status, then
   follow a figure to its numerical data and source record. A previous attempt's
   picture is not evidence that the current attempt succeeded.

The terminal draft stores prompt answers for continuation. The run configuration
records solver choices. The manifest binds inputs and produced artifacts. A `.blend`
contains an editable visual scene. These files serve different purposes.

## What goes in, what the software does, and what comes out

### CLASS and AxiCLASS: the linear universe

**Input:** cosmology, primordial perturbations, species settings, redshift and wavenumber
coverage. **Implementation:** the selected Boltzmann solver evolves linear background
and perturbation equations; `core/linear.py` and `core/cosmology.py` prepare the inputs,
invoke the executable and collect its outputs. CLASS serves ordinary selected models;
AxiCLASS serves the EDE branch.

**Output:** a `spectrum.json` bundle with spectrum/transfer/background artifacts and
checksums. The particle stage consumes this bundle. Inspect model, epoch, k coverage,
normalization and transfer columns before interpreting a later power ratio. A linear
spectrum is a prediction before nonlinear particle evolution, not a halo catalogue.

**Use:** the connected `run` command orchestrates this stage. For an individual linear
experiment use `python commands.py class --help`; preserve the returned bundle rather
than supplying an arbitrary older directory to another run.

### monofonIC, MUSIC and the native IC route: particles from fields

**Input:** linear transfers, box geometry, seed/phases, species, initial redshift and LPT
order. **Implementation:** `core/initial_conditions.py` validates the handoff;
`core/evolution.py` drives monofonIC or the supported GADGET generator route;
`core/zoom.py` and `core/research/generator_controls.py` handle MUSIC and zoom controls.
Displacement and velocity fields become particle positions, velocities, masses and IDs.

**Output:** initial particle HDF5 plus generation/handoff records. Inspect species counts,
mass sums, box bounds, IDs, units, scale factor and transfer provenance. The same random
seed across different generators does not prove identical phases.

**Use:** choose the generator during `run`. For parent-to-zoom work consult
`trace-zoom --help`, `zoom --help`, `stage-zoom-inputs --help` and
`audit-zoom-phases --help`. A zoom needs a parent selection and boundary particles;
shrinking a uniform grid does not create a nested zoom.

### GADGET-4: evolution, gas, stars and black holes

**Input:** valid initial particles, cosmological background, compile features, native
parameters and any cooling/yield/chemistry assets. **Implementation:**
`core/evolution.py` generates the build options and parameter file and starts GADGET.
The native gravity/time-integration/SPH/source routines evolve the particles.

**Output:** complete saved-epoch HDF5 files, diagnostics, restart/checkpoint files,
configured catalogue products and source journals. Inspect the native process exit,
requested versus saved epochs, finite fields, particle/mass accounting and physical
source ledgers. A snapshot must include all its shards.

Gas dynamics, cooling, star formation, stellar returns and BH accretion are separate
models. A short star-formation-enabled run may produce no stars. A BH particle can
exist without substantial accretion. See [calculations](calculations.md) for the
carrier/subgrid mass distinction and mass/energy accounting.

**Use:** `run` connects ICs and evolution. `gadget --help` exposes the individual driver;
continuation must use the original run's compatible executable/checkpoints. Source
controls use `prepare-source-controls --help` and `native-source-events --help`.

### FOF, SUBFIND, Rockstar, Consistent Trees and VELOCIraptor: structure

**Input:** particle snapshots and finder settings; tree builders need compatible
multi-epoch catalogues. **Implementation:** GADGET's FOF/SUBFIND run through compiled
features; `core/halos.py` invokes and reads the external finders and trees.
FOF groups nearby particles, SUBFIND identifies bound substructure, Rockstar uses
phase space, Consistent Trees builds temporal links, and VELOCIraptor supplies its
selected structure-finding route.

**Output:** catalogues, memberships, host/subhalo properties and, when built, temporal
links. Check mass definition, linking length, membership IDs, resolved-particle floor
and snapshot association. A tiny uniform test can legitimately contain no resolved
halos. Tree success also needs actual links; successful process launch is insufficient.

**Use:** configure native finders before evolution; inspect `halos --help`,
`velociraptor --help`, `compare-finders --help` and `select-merger-tree --help` for
saved-data routes. Histories require saved temporal information, not two unrelated
catalogue files with matching row numbers.

### Grackle and FSPS: two different baryonic calculations

**Grackle input:** gas thermodynamic/composition state, chemistry network, radiation and
cooling tables. It supplies the selected chemistry/cooling calculation through the
native bridge (`core/evolution.py`, native `research_grackle.cc`). **Output:** updated
chemical/thermal state and cooling diagnostics. Check network/table identity,
composition, energy and conservation. Installing Grackle does not enable it in a run.

**FSPS input:** saved stellar ages, metallicities, initial-mass convention, population
model data, IMF and requested filters. `core/stellar_population.py` calls the verified
population-synthesis backend and normalizes its spectra to the stellar population.
**Output:** model spectra, luminosities and filter products. Inspect model libraries,
age/metallicity support, filter definitions, mass normalization and any dust model.
FSPS computes light; it does not create the stars or evolve native gas.

**Use:** install the `sps` profile and provide its model assets, then select the relevant
stellar observer workflow; see `export-stellar-observer --help` and
`observe-stellar-catalogue --help`. Display emission/glow in Blender is not an FSPS
luminosity measurement.

### Agama, Gala and StreaMAX: potentials, orbits and streams

**Input:** a defined potential/host history, progenitor state, mass/size, physical units,
time sampling and random seed where needed. `core/action_angle.py`, `core/halos.py`
and `core/stream_resimulation.py` connect these backends. Agama evaluates selected
potentials/actions; Gala supplies its spray adapter; StreaMAX supplies an explicit
static-host reference. Research2 also has a self-gravitating collisionless stripping
integrator. Those approximations answer different questions.

**Output:** orbit/stream phase-space samples, diagnostics and comparison products.
Inspect time direction, units, release prescription, particle count, host evolution
and support of the selected approximation. Spray particles are a model of stripping,
not a native self-gravitating star cluster.

**Use:** install `streams`; consult `streams --help`, `stream-products --help` and the
stream sections in the operator guide. A renderer called `stellar-stream-movie` needs
an already measured stream selection; it does not itself run Gala.

### SIDM and OpenGadget3: scattering and an independent control

**Input:** particle state, cross-section/scattering model, interaction parameters and
resource controls. Research2's native SIDM route evolves the configured interaction;
`core/sidm_forward.py` and `core/sidm.py` support explicitly authored OpenGadget3
comparison campaigns. An independent executable needs its own compatible build and
request, not the GADGET-4 parameter file.

**Output:** changed particle phase space, event/operator diagnostics and matched-control
comparisons. Compare collisionless and scattering cases with matched initial states;
measure momentum/energy and convergence. Few collisions in a tiny stochastic test can
be expected. Increasing the rate to test execution changes the experiment.

**Use:** inspect `sidm-angular-control --help`, `sidm-control-products --help` and the
SIDM campaign instructions. Keep independent-engine results identified by backend.

### DisPerSE: a topological skeleton

**Input:** the selected density grid and persistence/smoothing settings.
`core/visualization.py` invokes `mse`/`skelconv` for the explicit `disperse_grid` route.
**Output:** a skeleton/filament representation and projection products. Check grid
coordinate conversion, periodicity and persistence threshold. This differs from a
heuristic image overlay and from a time-linked particle trajectory.

**Use:** install the `disperse` engine, configure the native extraction route, and
request the corresponding map/filament products. See the visualization commands in
the operator guide and `filament-flythrough --help`.

### dynesty, emcee, SBI and cloudpickle: inference

**Input:** specified priors, observations and uncertainties, model/forward simulations,
selection and seed. `core/stream/inference.py` loads the likelihood samplers and
serializer; `core/stream/sbi.py` manages selected simulator-based training.
**Output:** chains/nested-sampling results or a trained density estimator, plus settings
and diagnostic products. cloudpickle serializes supported state; it is not a sampler.

Inspect convergence, effective sample size, prior bounds, simulator support and
calibration before interpreting a posterior. A ten-step chain or two training epochs
proves calculation execution only. See the inference commands in
[command reference](command-reference.md) and the inference/stream operator sections.

For an SBI campaign, follow the artifacts in this order:

1. The forward signature binds the physical definition, parameter bindings and
   source bytes. Its accepted domain must have executed force, integration,
   progenitor, observation, selection and discrepancy controls. An analytic
   tracer control can support its own narrow numerical domain; it does not
   accept a native stellar-stream model.
2. The dataset records independent parent/lineage components and splits those
   components before fitting. Public training tensors mask held-out parameter
   truth. Keep the separate `heldout_truth.json` for evaluation only.
3. Summary or graph training uses the training rows, selects a state using
   validation rows, and freezes `posterior.pkl` with its hash and backend
   sources. The summary computes moments; the graph performs message passing
   using observed coordinates. Neither receives native membership labels.
4. Calibration samples the frozen posterior for the independent test cases.
   SBC compares randomized parameter ranks with a uniform distribution.
   Coverage counts whether the declared credible interval contains each
   held-out truth, and reports a binomial uncertainty interval. The acceptance
   checks use that interval, not just closeness of the central coverage estimate.
5. Deployment requires accepted calibration for the exact frozen model. Its
   OOD check measures distance from training summaries and the fraction of
   coordinates outside training ranges. Unsupported observations produce no
   posterior draws. A graph model also needs the paired held-out log-score
   comparison against an independently calibrated summary baseline.

The retained short harmonic example completes steps one through four and fails
calibration. Open its rank and coverage figures in the optional atlas to inspect
the original masks, counts, units and controls. A rejected calibration is an
actionable result: it leaves deployment disabled. Its 17 independent test
groups cannot establish narrow coverage accuracy, and three training epochs
also leave model accuracy unresolved. A larger study needs fresh independent
test cases and training controls frozen before looking at their truth.

The separate larger harmonic example used 1,536 fresh initial-state ensembles
and passed summary-model calibration on 613 independent test groups. Its
deliberately distant observation then failed OOD support and produced no
posterior samples. The graph candidate on that same fixed split failed
calibration, leaving its gain evaluation disabled. Compare the original
software receipts to see the difference between completing an evaluation,
passing calibration and accepting an observation for inference.

### lenstronomy, pyrecon, pycorr and Corrfunc: observations and large-scale statistics

**Input:** lens/observation geometry or measured tracer/random catalogues, selection,
weights, distance convention and estimator settings. `core/observations.py` connects
lenstronomy's selected lens calculation; `core/bao.py` drives reconstruction and pair
counting. pyrecon reconstructs selected density/displacement fields; pycorr assembles
the correlation estimator using Corrfunc's pair counting.

**Output:** lensing/observation products or reconstructed positions, counts,
correlation measurements and selected likelihood products. Check angular/physical
units, random selection, estimator normalization, mesh/smoothing and covariance.
A cosmological snapshot is not a survey observation without the stated mapping.

**Use:** install `cosmography`; consult `cosmography --help`, `ingest-bao --help`,
`bao-model --help` and their operator sections. These workflows require their own
requests and data in addition to a cosmological box.

### NumPy/SciPy, matplotlib, HDF5 and the movie stack

NumPy/SciPy implement array operations, integration/interpolation, transforms and spatial
queries in the analysis modules. h5py reads and writes scientific HDF5; PyYAML reads
configuration. matplotlib draws measured arrays. Pillow handles raster images;
imageio/FFmpeg encode frames. PyVista provides selected interactive/offscreen displays.
Their outputs are arrays, files or visualizations used by the domain workflow; they
are not separate physical simulation stages.

For a scientific plot, inspect its plotted-data sidecar and `.figure.json` as well as
PNG/PDF/SVG. For maps inspect the map's numerical bundle and projection geometry. For
movies inspect source epochs, exact-ID matching and frame manifest. MP4 compression
is unsuitable for recovering the original measurements.

### Blender: editable scenes and rendered exports

**Input:** Research2's exported PLY/arrays and scene/source manifests, species selection,
camera/timeline and display settings. `core/blender.py`, `core/blender_scene.py` and
`core/scene_render.py` build and render scenes using Blender's Python API.
**Output:** editable `.blend`, previews, frame/render records and requested MP4.
Inspect species/point counts, camera, timeline, source record and rendered frames.
Sparse particles can produce a sparse picture even when rendering worked.

**Use:** install Blender separately and select `blender` or `blender_movie` in analysis.
The optional engine control instead makes a small still and editable scene, with
a fresh-process load check and no video encoding.
Run `python commands.py blender --help` for the public entry point. Helpers importing
`bpy` must run inside Blender, for example `blender --background --python SCRIPT.py`;
the ordinary project Python cannot import Blender's bundled API. Scene lights and
particle glyph size are display choices unless a separate physical-light product is
explicitly supplied.

## Zoom initial conditions: from a parent selection to checked particles

### Follow the zoom checks in the order their evidence allows

A viewing zoom and an initial-condition zoom have different inputs. The lab's
projected close views select the same existing particles at smaller image
scales. MUSIC refinement instead needs a parent simulation, a late-time
selection traced back to its original Lagrangian coordinates, matching coarse
modes, and coarse boundary matter around the refined region.

The tracing request uses a centre in **comoving kpc/h**, a selection radius in
**proper kpc**, and padding in **comoving kpc/h**. At scale factor a and Hubble
parameter h, a proper radius R becomes a native comoving radius R h/a. The
reader applies periodic minimum-image distances before selecting particles.
It then matches exact integer particle IDs to the initial load. A zero ID can
be valid; converting IDs to floating point can merge distinct particles.

| File / callable | What it does | Product and interpretation |
|---|---|---|
| [`core/zoom.py:trace_aperture`](../core/zoom.py#L189) | Reads complete target snapshots and the original IC, selects the declared aperture or supplied exact members, matches IDs and retains their initial q coordinates | `region.json`, selected-particle HDF5 and region figures; a traced refinement proposal with original membership, not proof of a clean evolved target |
| [`core/zoom.py:trace_catalogue_host`](../core/zoom.py#L591) | Starts with a native FOF host or selected galaxy, its actual epoch/centre/radius and membership, then calls the same tracing layer | Catalogue-linked selection with explicit radius factor; catalogue identity does not remove the need for padding/coverage checks |
| [`core/zoom.py:audit_music_ic`](../core/zoom.py#L1144) | Inspects original generated fields, declared units/species, exact IDs, refinement-level and Lagrangian metadata | Structural IC audit; field presence does not establish transfer normalization, phase matching or evolved convergence |
| [`core/zoom.py:audit_target_coverage`](../core/zoom.py#L953) | Verifies the exact traced region and generated IC, then intersects coarse material cells with conservative guards around the selected initial q points | Original intersecting coarse IDs and pass/fail static coverage; no inference about later contamination |
| [`core/zoom.py:compare_density_phases`](../core/zoom.py#L1726) | Checks common cosmology/epoch/coarse grid and restored integer-cell shifts, reads coarse-noise arrays and measures independent large-scale total-matter modes | Noise equality, squared coherence, weighted phase RMS, amplitude ratio and mode arrays; a density-phase check with separately stated limits |
| [`core/zoom.py:audit_zoom_velocity`](../core/zoom.py#L1865) | Reads delivered bulk velocity/density modes and the signed CLASS transfer ratio; restores the recorded global momentum shift and subtracts separately stored thermal velocities when present | Measured/predicted divergence arrays, relative L2 residual and count of measurable modes; no relic thermal-distribution validation |
| [`core/zoom.py:radial_particle_census`](../core/zoom.py#L2553) | Reads the complete evolved native shard set in bounded chunks, preserving unequal masses and actual saved softenings | Radial mass/count/second-mass moments and source records for contamination/heating/relaxation diagnostics |

For common complex density modes p and z, the squared coherence is
|Σ p* z|² / [(Σ|p|²)(Σ|z|²)]. The phase RMS weights
arg(z p*)² by |p|². A uniform amplitude rescaling can have perfect coherence
and zero phase error, so inspect the separately recorded amplitude norm ratio.
The implemented density-phase gate requires bitwise equality of restored coarse
noise, sufficient independent modes, and the declared coherence/phase bounds.
Zero perturbations cannot establish phase matching. Identical seeds across
cosmologies or generators do not replace these measurements.

The velocity audit compares the delivered total-matter velocity divergence
with density × (signed transfer θ/δ) × c/(1000 h), using the native Fourier
and storage conventions. It does not replace this ratio with a generic LCDM
growth-rate approximation for an EDE model. The residual is
||measured − predicted||₂ / ||predicted||₂. A missing or vanishing denominator,
insufficient measurable modes or incompatible frames is rejected explicitly.

Use `python commands.py trace-zoom REQUEST.json` for an aperture request, or
add `--catalogue-host` for the catalogue-host request schema. Stage the selected
inputs with `stage-zoom-inputs REQUEST.json`; the request supplies its destinations
and physical settings. To compare generated phases, the actual argument order is:

```sh
python commands.py audit-zoom-phases /absolute/path/to/parent.hdf5 /absolute/path/to/zoom.hdf5 /absolute/path/to/parent-noise.bin /absolute/path/to/zoom-noise.bin /absolute/path/to/new-phase-audit.json --memory-gib 1
```

Every path must identify the corresponding original product. Noise caches must
be the native-endian MUSIC full-double cube: three uint32
dimensions followed by n³ float64 values, for a total size of 12 + 8n³ bytes.
They are not HDF5 files or arbitrary random arrays.
Consult the command reference and request examples before staging. Structural
success, static coarse-cell coverage, phase/velocity agreement and evolved
resolution acceptance are separate results. Preserve all of them when comparing
a parent and refinement run.

### Reproduce the minimum-grid MUSIC parent control

This checks the production staging and native-generation path without evolving
a galaxy. Supply your installed MUSIC executable and a completed schema-2 CLASS
`spectrum.json` at z=49, with transfers covering the requested mesh. The retained
example uses a 64 Mpc/h box, parent/refinement level 5 and seed 150. Level 5 means
32³ gas particles plus 32³ DM particles, **65,536 total**, rather than 32 total.
The shared template explicitly enables `setup.do_SPH`: MUSIC reads this switch
before its output plugin starts, and needs it to write gas positions and IDs.

First obtain the admission proof for the exact binary:

```sh
python commands.py native-music-transfer /absolute/path/to/music-proof --binary /absolute/path/to/MUSIC --prepare --nested
```

Save this request as `parent-request.json`, replacing the two paths:

```json
{
  "spectrum_manifest": "/absolute/path/to/linear-bundle/spectrum.json",
  "destination": "/absolute/path/to/music-parent/staged/native-inputs",
  "box_mpc_h": 64.0,
  "parent_level": 5,
  "refinement_level": 5,
  "seeds_by_level": {"5": 150},
  "padding_cells": 2,
  "poisson_accuracy": 1e-14,
  "lpt_order": 1
}
```

Stage and execute it in fresh destinations:

```sh
python commands.py stage-zoom-inputs parent-request.json
python commands.py zoom /absolute/path/to/music-parent/staged/native-inputs/parent.conf /absolute/path/to/music-parent/generated --binary /absolute/path/to/MUSIC --native-validation /absolute/path/to/music-proof/native-transfer-suite.json --threads 2 --audit-memory-gib 1 --native-memory-gib 2 --native-disk-gib 4 --native-wall-hours 0.1 --structural-only
```

Read `generated/generation.json`, `physical-audit.json`, `velocity-audit.json`
and `generation.log`. The retained execution passed mass, exact-ID, momentum,
barycenter and species-velocity checks. The generation status is
`GENERATED_STRUCTURAL_PASS`; density/velocity-mode tests remain pending and
the parent phase status is `INCOMPLETE_NO_PARENT_PHASES`. A passing structural
control therefore still has `science_acceptance: NOT_ACCEPTED`.

The same folder structure supplies the native fixture for the refinement
staging regression. It checks inherited seeds, a newly seeded level, resource
caps and rejected receipt mutations; it does not generate or evolve that child:

```sh
RESEARCH_NATIVE_ZOOM_BRANCH=/absolute/path/to/music-parent python verification.py BlockIntegrationTests.test_actual_music_refinement_staging_and_caps -v
```

## Read shape and environment measurements from their arrays

A picture of a stretched object does not tell you which spatial window or
population was measured. The shape pipeline starts with a fixed centre, exact
particle IDs, positive masses in M☉, positions in proper kpc and a declared
complete support radius. `aperture_all` means all selected species in that
aperture; it does not claim that every particle is gravitationally bound.

For the `mass_quadrupole` estimator, the weighted second-moment tensor has the
form Iᵢⱼ = Σₚ mₚ xₚᵢ xₚⱼ / Σₚ mₚ. Its ordered eigenvalues λ₀ ≥ λ₁ ≥ λ₂ give
b/a = √(λ₁/λ₀) and c/a = √(λ₂/λ₀). A `spherical_shell` measures the moment of
a spherical cut. The `iterative_ellipsoidal_shell` option repeatedly selects
particles using the fitted ellipsoid; its boundary semiaxes preserve the volume
of the original radial window. These are different selections and can yield
different ratios for the same particle distribution.

The major/intermediate/minor eigenvectors are columns of `axes_columns`. Their
signs have no physical preference. Adjacent-shell twist therefore uses
θⱼ = arccos(|eⱼ,inner · eⱼ,outer|), reported in degrees. A nearly spherical
sample can have stable axis ratios and poorly determined directions. The
bootstrap refits the original samples and records per-axis orientation support;
finite eigenvectors alone do not make a twist measurement supported.

| Implementation | Inputs and operation | Return or saved result |
|---|---|---|
| [`load_native_host_component_samples`](../core/halos.py#L590) | Run, snapshot, host row, species/membership definitions and proper support radius; verifies HDF5 publication/units, applies the periodic minimum-image aperture and preserves exact IDs | Complete selected component phase-space arrays and source/epoch context |
| [`radial_density_shapes`](../core/halos.py#L4561) | Proper offsets, positive masses, radial edges and explicit iteration/bootstrap controls; measures every requested shell | Dictionary containing shell ratios, axis columns, boundary geometry, counts, uncertainties and failure/support states |
| [`radial_shape_twists`](../core/halos.py#L62) | Adjacent measured shell records and the name of their orientation-support flag | One record per adjacent pair, with unsigned angles or unresolved entries; no interpolation across missing shells |
| [`publish_halo_structure`](../core/halos.py#L712) | Complete structure request; reads original samples, fits the selected AGAMA field and compares its forces/tides with direct particle sums at separate calibration and held-out positions | Structure JSON, numerical arrays and reloadable potential coefficients; density-shape and field-validation states remain separate |
| [`publish_native_environment`](../core/analysis.py#L1143) | Full periodic snapshot, mesh/smoothing settings, threshold and measured structure products | CIC density, spectral tidal field, web classes, conditional alignments and original support masks in JSON/NPZ products |
| [`measure_filament_axis_alignment`](../core/analysis.py#L227) | Measured spin/shape vectors, local filament axes and their boolean support; normalizes nonzero vectors and calculates unsigned dot products | Cosines, retained support and separately generated isotropic comparison draws |
| [`export_native_structure_figures`](../core/halo_products.py) / [`export_native_web_figures`](../core/halo_products.py#L390) | Verified structure/environment product; adapts its original arrays and masks to registered figure recipes | PNG/PDF/SVG, plotted-array NPZ sidecars and figure manifests under the output gallery |

The environment field has its own spatial meaning. CIC assigns particle mass
to a periodic grid; the specified Gaussian smoothing acts on Fourier modes.
The zero-mean periodic Poisson solve produces a Hessian whose positive
eigenvalues denote compression. The T-web class counts eigenvalues above the
recorded dimensionless threshold: zero is a void, one a sheet, two a filament
and three a knot. A filament direction is the least-compressive eigenvector,
with an eigenvalue-gap check. These classified voxel regions differ from a
DisPerSE ridge skeleton.

Alignment plots use |v̂ · êfilament|, ranging from 0 (perpendicular) to 1
(parallel). A uniformly random unoriented vector has mean cosine 1/2. For
spin, v is the measured enclosed angular momentum Σ m x × vpeculiar; for
shape it is the supported shell major axis. A zero-spin aperture, unresolved
axis or non-filament environment produces a masked entry. The isotropic draws
are a directional comparison, not a simulation convergence error bar.

Inspect `density_shapes`, `radial_twists`, `field_validation_status` and
`halo_measurements` in the product JSON, then compare the plotted point to
its array sidecar. A passed field comparison checks the declared reconstruction
and sampled locations. It does not establish converged halo evolution.
The [smoke report](smoke-test.md) records the known-geometry controls and their
measured results; their HDF5 samples are explicitly authored synthetic inputs.

## Splashback diagnostic: density, slope, then feature support

The plotted slope is d ln ρ / d ln r. Both logarithms use the same radius and
density units throughout, so the derivative is dimensionless. The underlying
shell density is mass divided by proper shell volume; its mass-shot-noise
error is √(Σ m²)/V. The native environment producer converts these errors into
a diagonal log-density covariance with entries (σρ/ρ)². This is a conditional
noise model, not a total uncertainty or convergence estimate.

[`uncertainty_aware_log_density_derivative`](../core/analysis.py#L278) takes
increasing proper radii, measured densities, effective counts and the full
supplied covariance. Within each contiguous supported segment it fits a local
polynomial to ln ρ as a function of ln r by generalized least squares. The
linear derivative operator D gives slope = D ln ρ and propagated covariance
Cslope = D Clogρ Dᵀ. Empty/underpopulated bins split the segment; a fitting
window cannot bridge them. Its result includes the operator, covariance,
standard errors and boolean support, so the plotted derivative can be replayed.

[`locate_supported_slope_feature`](../core/analysis.py#L332) searches the
supported slope for valleys inside an explicit radius range. Prominence is
the depth relative to the lower adjacent shoulder. The half-prominence width
is ln(router/rinner), a dimensionless log-radius interval rather than a width
in kpc. Multiple supported valleys remain ambiguous. The uncertainty calculation
perturbs the measured slopes using their correlated Gaussian covariance and
retains the fraction of draws with exactly one acceptable feature. The
`bootstrap_realizations` setting names those draws; it does not resample
particles or rerun a simulation.

[`publish_native_environment`](../core/analysis.py#L1143) joins these calculations
to the original structure product, epoch and component profile. It retains
alternative derivative-window results separately. Finally,
[`export_native_splashback_figures`](../core/halo_products.py#L235) reads and
verifies that product and its source structure, then produces three panels:
the measured slope with propagated errors, supported prominence, and supported
half-prominence width. The x axis of each is proper radius in kpc. A rendered
slope can be supported while the feature's depth/width remain unavailable.
This nonparametric diagnostic differs from a fitted turnover-based splashback
model; the two definitions should not be compared as identical parameters.

The optional atlas's known-profile example uses an authored density with
logarithmic slope −2 − 3 exp[−(ln r)²/(2w²)], w=0.2, over its sampled radial
domain. Its true minimum is at r=1 kpc, prominence is 3 and ideal half-prominence
width is 2√(2 ln 2)w ≈ 0.471. The particle sample is truncated to 0.2–4 kpc;
the rising slope near the inner cut is a sampling-boundary feature, outside
the requested 0.5–2 kpc search. Interpret the recovered radius together with
its original bin bracket, derivative window and covariance. The demonstration
checks recovery from an input profile; it does not establish physical halo
splashback or resolution convergence.

## Assembly histories: identity first, then measurements

The optional flyby is reproducible without campaign files using
`python optional_tests/advanced_controls.py orbits --output /absolute/path/to/new-external-output`.
[`orbital_control.py`](../optional_tests/orbital_control.py) creates complete
explicit-unit particle/catalogue epochs, authored identity links and image offsets,
then invokes the production graph/trajectory/event functions below. Its receipt
and offline report distinguish known-input recovery from physical orbit validation.

A mass history follows one branch in an identity graph. Catalogue row 0 at two
epochs does not by itself establish that the object is the same. The graph
records epoch-qualified object keys and descendant evidence. The branch reader
follows an explicit main progenitor, or a sufficiently separated transferred
mass contribution. Missing detections, ambiguous parents and excessive saved
time gaps remain masked.

The branch's mass definition must stay fixed. Its plotted accretion rate is
Δln M / Δln a, a dimensionless finite-interval quantity, rather than M☉/year.
The x coordinate for this rate belongs to the interval between two saved epochs.
A half-mass time is a linear crossing within a supported mass bracket; that
bracket measures saved cadence. A half-mass crossing outside the saved range
cannot be recovered from the final mass alone.

Concentration uses a separate measurement. The native structure function first
finds a single bracket for the total enclosed mean density equal to
Δ × ρreference. It interpolates RΔ on the recorded radial grid, then fits the
selected component's shell densities to
ρ(r) = ρs / [(r/rs)(1 + r/rs)²]. The plotted c = RΔ/rs is dimensionless.
The fit requires at least five supported bins with positive density/error and
the requested effective particle count. Its covariance is conditional on the
NFW model and shell shot noise. It does not include uncertainty in RΔ, model
choice, resolution or host membership.

Stellar assembly plots the measured stellar mass, not the number of individual
stars. In real runs, each stellar particle is a population. The specific SFR
is SFR [M☉/year] divided by stellar mass [M☉], giving year⁻¹. Zero SFR is a
valid quiescent measurement when the source census establishes it; absent SFR
is unavailable. The quenching threshold classifies the saved state. It does
not reconstruct an unsaved quenching time or establish a physical mechanism.
Morphology uses the eigenvalues of the original stellar mass tensor inside the
recorded aperture, with the same axis-ratio interpretation as the shape guide.

| File / callable | What it consumes and implements | What you receive |
|---|---|---|
| [`core/halo_campaign.py:structure_assembly_attributes`](../core/halo_campaign.py#L318) | Original structure report/arrays plus exact component, profile-bin and shape selections; carries measured spin, shapes, concentration and optional stellar tensor into graph attributes | Attribute dictionary with measurement definitions; unavailable measurements stay absent |
| [`core/assembly.py:reconstruct_quality_masked_main_branch`](../core/assembly.py#L13) | Identity graph, target key and branch clock/cadence settings; follows original ancestry and quality | Branch report, aligned mass/stellar/spin arrays, per-epoch quality and interval masks |
| [`core/assembly.py:measure_assembly`](../core/assembly.py#L93) | That measured history and event settings | Growth rates, cadence-bracketed half-mass crossings, spin changes, specific SFR and stripping/quiescent candidates |
| [`core/halo_campaign.py:native_assembly_history_runner`](../core/halo_campaign.py#L517) | Verified graph and complete template; derives times/scales from graph epochs, runs the estimators and preserves mass definitions | `assembly.json`, its numerical archive, result receipt and registered figures |
| [`core/assembly.py:export_native_assembly_figures`](../core/assembly.py#L306) | Completed assembly product and the exact source graph | Mass, growth, concentration, spin, shape, stellar and quenching/morphology plots with original support |
| [`core/flagship_orbits.py:measure_native_assembly_orbits`](../core/flagship_orbits.py#L96) | Original snapshot/catalogue graph, recorded integer periodic images, proper clock and interval flags | Relative Hermite trajectories and orbital event candidates, plus separately identified progenitor joins |
| [`core/assembly.py:orbital_assembly_events`](../core/assembly.py#L195) | Object/host worldlines and a declared host-radius history | Pericentre/apocentre extrema of squared separation and inward radius crossings; saved brackets remain in every candidate |

To render an existing valid assembly product, activate the environment and run
from the repository root:

```sh
python commands.py export-assembly-figures /absolute/path/to/assembly.json /absolute/path/to/identity-graph /absolute/path/to/new-output --memory-gib 1
```

The output directory must be new. It contains a measurement index and a
`gallery/` with the plots, numerical sidecars and manifests. This command
renders measurements; it does not generate a graph or evolve particles.

Orbital positions are global proper coordinates after applying the recorded
periodic images. Their time derivative includes both peculiar motion and
Hubble expansion: dxproper/dt = vpeculiar + H xproper. The code converts these
velocities to the worldline clock before solving polynomial extrema. A
progenitor join at a saved epoch is an ancestry event, not a measured physical
coalescence time. An interpolated pericentre remains a candidate until its
trajectory error has an independent certificate.

The optional atlas includes explicitly authored assembly and flyby controls.
Their graphs describe known synthetic identity correspondence. Their changing
masses and fitted scale radii were prescribed; only the subsequent measurements,
fits, event reconstruction and rendering were executed. This keeps them useful
for learning the pipeline without labeling them as native stellar formation or
converged halo evolution. The [smoke report](smoke-test.md) records the expected
numbers and original failures.

## Paired action changes: what the diffusion-labelled plot measures

The `action_diffusion` recipe plots mean squared paired action changes. It does
not fit a stationary diffusion coefficient. The comparison starts with the same
exact particle cohort, initial state, saved clock and fixed physical coordinate
chart in both native branches. A shared random seed is insufficient: the
production comparison checks the original particle IDs, masses and state hashes.
Changing the first selected epoch after the branches have diverged fails that
initial-state check.

For a stationary spherical diagnostic potential, each particle's specific
energy is E = |v|²/2 + Φ(r), and L = |x × v|. The action calculation solves
for its radial turning points and integrates
Jr = (1/π) ∫[rperi,rapo] √[2(E − Φ(r)) − L²/r²] dr.
The other spherical components are Jz = L − |Lz| and Jφ = Lz. Actions are
per unit mass, in proper kpc km/s. Positions, velocities, potential origin,
field domain and potential units must describe the same physical chart.
The spherical formula is inappropriate for a general nonspherical native field
unless an explicitly stated spherical approximation defines the diagnostic.

For branch difference ΔJᵢ(t) = Jintervention,ᵢ(t) − Jreference,ᵢ(t), the plotted
component α is Qα(t) = meanᵢ{[ΔJᵢα(t) − ΔJᵢα(t₀)]²}. Its units are
proper kpc² (km/s)². The mean includes particles supported in both branches
at that epoch **and** the first saved epoch. Its particle count can therefore
change with support; check those counts beside the curve. Q(t₀) is zero by
definition. The code does not divide Q by elapsed time or infer a transport
coefficient, and a finite change does not identify a unique perturber.

The saved `paired_action_change_kpc_kms` array holds the uncentred branch
difference ΔJ(t). The separate `paired_action_mean_square_kpc2_kms2` array
holds the centred Q(t) above. This distinction matters when replaying a
plotted point from the arrays.

| Implementation | Calculation / validation | Returned or retained result |
|---|---|---|
| [`core/streams.py:publish_native_stream_worldlines`](../core/streams.py#L761) | Reads exact selected stars from completed native generations at the requested saved epochs, with original units and execution receipts | Physical positions/velocities/masses, exact IDs, times and source/initial-state bindings |
| [`core/streams.py:publish_native_stream_comparison`](../core/streams.py#L1312) | Requires identical initial selected states, species, clock/frame and persistent IDs; calculates actual paired position/velocity differences | Comparison product with original cohort weights and both source histories |
| [`core/stream/workflow.py:_potential`](../core/stream/workflow.py#L140) | Loads measured coefficients and requires passed field validation; spherical adapters enforce the declared radial domain | Common frozen field adapter and original backend coefficients/sources |
| [`core/action_angle.py:measure_actions`](../core/action_angle.py#L469) | Selects a method from the explicit symmetry/time domain; spherical quadrature checks turning points and integral diagnostics | Actions, angles/frequencies where supported, boolean support and per-particle unresolved reasons |
| [`core/native_stream_responses.py:paired_action_evolution`](../core/native_stream_responses.py#L121) | Intersects current and baseline branch support, calculates ΔJ and the baseline-centred mean square | Raw paired differences, per-particle masks and per-component Q(t) |
| [`core/native_stream_responses.py:native_response_runner`](../core/native_stream_responses.py#L134) | Replays paired native evidence, verifies a common frame, freezes the reference track, measures binned responses and optional actions | `stream-responses.json`, numerical archive, source records and registered response figures |

Radial/circular singular limits, unbound particles, incomplete turning domains
and failed quadrature remain unresolved. An unsupported initial action cannot
be replaced by zero to create a later change. Likewise, an unavailable host
potential is an explicit missing input. A source-bound monopole can define a
useful common comparison chart, but its actions need not be conserved along
particles evolved in a different, nonspherical force field.

## How to decide whether an output is useful

For each product follow this chain: selected physical model → actual saved input →
measurement and units → numerical arrays → figure/movie → source record. Compare a
few numbers by hand: particle count and mass, an epoch's `a=1/(1+z)`, histogram counts
and volume normalization, or a plotted point against its saved array. Record unavailable
inputs instead of replacing them with illustrative values.

The [smoke report](smoke-test.md) states what this checkout actually executed. Small
controls establish execution and specific measured checks. Research conclusions need
appropriate populations, numerical convergence and independent reference comparisons.

### Read a failed calibration alongside successful execution

The expanded harmonic control in the smoke report passed 12,288 exact oscillator
checks before training. Both density estimators then trained, froze and produced
held-out ranks, credible intervals and log scores. Both failed calibration on
899 independent groups. That outcome distinguishes a working forward/inference
pipeline from an accepted posterior: the saved model can execute and still
under-cover the true parameters. Its training settings and held-out set stay
fixed; changing a credible mass, tolerance or source receipt would not repair it.
The graph-versus-summary gain calculation requires both calibrations to pass,
so the corresponding recipe remains unavailable for that campaign. Inspect the
original rank and coverage figures and the recorded failure conditions first.
