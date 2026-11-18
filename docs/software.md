# Software and dependencies

Each backend has a defined consumer. The selected workflow determines which software
runs; installing a package does not enable its physical model. Installation integrity
and successful scientific execution are separate checks.

For each backend’s inputs, implementation, output files, inspection steps and public
entry points, use the [learning guide](learning-guide.md#what-goes-in-what-the-software-does-and-what-comes-out).
The [command reference](command-reference.md) supplies parser arguments, and the
[smoke report](smoke-test.md) supplies measured execution evidence.

| Your next task | Start here |
|---|---|
| Choose dependencies for a workflow | [Installation profiles](#installation-profiles) |
| Install or extend an environment | [Setup commands](#install-only-the-workflows-you-need) |
| Understand a native executable | [Native roles and drivers](#native-engines-and-their-roles) |
| Understand a Python distribution | [Package consumers](#python-packages-and-their-roles) |
| Run the standalone cooling check | [Grackle build and outputs](#build-optional-grackle-and-inspect-its-cooling-control) |
| Run an independent Linux SIDM check | [OpenGadget3 control](#independent-opengadget3-control) |

Run the commands from the repository root. After setup, activate the environment
before using the `python` commands in this guide:

```sh
source ../research2-runtime/environments/default/bin/activate
```

Installation profiles select **dependencies**. The optional lab's `quick`,
`figures`, `libraries` and `complete` profiles select **tests**; they are a
different set of names. For example, install `halo-structure` to supply AGAMA
for the lab's `figures` checks. The [lab prerequisite table](optional-tests.md)
maps each test profile to its required dependencies and native installations.

## Native engines and their roles

| Component | When it is used | Code that connects it |
|---|---|---|
| CLASS | Selected non-EDE linear cosmology | `core/linear.py`; `core/cli.py:run_pipeline_main` |
| AxiCLASS | Selected EDE linear cosmology | Same linear driver, selected by model |
| GADGET-4 | Configured particle/physical-source evolution | `core/evolution.py:prepare_and_run`, `run_gadget` |
| monofonIC | Selected collisionless IC generation | `core/initial_conditions.py`, `core/evolution.py` |
| MUSIC | Selected parent/zoom IC generation and native handoff | `core/zoom.py`, `core/research/generator_controls.py` and the IC handoff readers |
| FOF/SUBFIND | Native halo/host/subhalo catalogues | Compiled GADGET feature choices; `core/halos.py` readers |
| Rockstar | Independent phase-space finder when configured | `core/halos.py`; finder workflow in `commands.py` |
| Consistent Trees | Explicit temporal catalogue/tree production | `core/halos.py`; fresh-run analysis/tree configuration |
| VELOCIraptor | Requested stellar/phase-space or finder experiment | `core/halos.py` stellar finder adapter and `velociraptor` command |
| DisPerSE | Explicit `disperse_grid` extraction | `core/visualization.py` native `mse`/`skelconv` route |
| Grackle | Selected compatible native chemistry/cooling network | `core/evolution.py` physical/library validation; GADGET `research_grackle.cc` |
| Blender | Requested interactive scene, preview or scene movie | `commands.py:build_scene`; `core/blender.py`, `core/blender_scene.py`, `core/scene_render.py` |
| Blacklight (optional, separately installed) | Prepared Kerr ray tracing and independent GRMHD/analytic emission display | [Blacklight handoff](blacklight.md); source pin, external build and scientific status are recorded separately |
| OpenGadget3 | Authored independent rare/frequent SIDM control | `core/sidm_forward.py`, `core/sidm.py` control sections, explicit campaign build/request inputs |
| Pristine upstream GADGET and native patches | Source identity, patch/release construction and supported native controls | `core/release.py`, `core/native.py`, source-bound build/control routes |

For native H₂/HD cooling, the Grackle adapter transports the
protium, helium and deuterium reservoirs with the particle state.
Grackle 3.4.1 expects a deuterium-to-hydrogen mass ratio. Its consistency
pass normalizes H and He against the solver density before accounting
for deuterium and excludes the H in HD from its H sum. The adapter
checks that convention, converts the returned fractions to the original
reservoirs, then checks nucleus and charge conservation. The snapshot
reader uses the native energy unit and applies the Hubble exponent only
to comoving runs.

A backend is **connected** when its driver exists. It runs only for a selected workflow
with its required inputs. `all` cannot run CLASS and AxiCLASS as one cosmology, invent
an independent comparison run, create a posterior, or enable unselected chemistry/source
physics. Build tools are used to produce executables; they are not additional simulation
stages.

The Snakemake, Blender sequence-loader, pynbody, NSphere and ChaNGa full trees had no
application runtime consumer. Their unused checkouts are archived outside the source;
pinned reference descriptors and licenses remain. These references explain
algorithm/provenance history and are not downloaded or built by setup. The small SIDM
method-reference assets remain scientific references, not a second simulator. The
pristine GADGET tree has an actual release/patch consumer and is retained for that
purpose.

## Build optional Grackle and inspect its cooling control

With the C/Fortran compilers, CMake, pkg-config and HDF5 development files
installed (the Ubuntu setup script supplies the build prerequisites), build the
retained Grackle source into an external prefix:

```sh
cmake -S core/software/chemistry/grackle -B ../research2-runtime/builds/grackle-shared -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGRACKLE_USE_DOUBLE=ON -DGRACKLE_USE_OPENMP=OFF -DGRACKLE_EXAMPLES=OFF -DGRACKLE_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX="$PWD/../research2-runtime/grackle-install"
cmake --build ../research2-runtime/builds/grackle-shared --parallel 2
cmake --install ../research2-runtime/builds/grackle-shared
export PKG_CONFIG_PATH="$PWD/../research2-runtime/grackle-install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
pkg-config --modversion grackle
python optional_tests/native_checks.py grackle --output ../research2-runtime/optional-lab/my-atomic-cooling --seed 42
```

Choose a new output directory. Some systems install under `lib64`; use the
actual directory containing `grackle.pc`. Version 3.4.1 and double precision
match this control. The worker compiles its small C caller against the installed
public API, records library/header/input hashes and native invocations, and
writes `cooling.csv`, `arrays.npz`, `cooling.png` and a status receipt. It evolves
32 independent atomic H/He cells at fixed density, with no lookup table, UV,
metals, molecules or dust. This checks native library execution and elemental
mass conservation; it does not enable chemistry in GADGET. Production chemistry
still requires its explicitly selected network, table and compatible native
build. See the [optional lab guide](optional-tests.md) for the integrated checklist
and [learning guide](learning-guide.md#grackle-and-fsps-two-different-baryonic-calculations)
for the production data flow.

## Python packages and their roles

The following table covers every distribution explicitly declared in
`configs/runtime/dependencies.json`; transitive dependencies are selected by the
packages' own build/runtime requirements.

| Distribution | Consumer / reason it is installed |
|---|---|
| numpy | Particle/field/measurement arrays throughout the scientific modules |
| scipy | Numerical integration, interpolation, FFT, spatial statistics and distributions in analysis/physics modules |
| matplotlib | Scientific plots and sidecar export in `core/figures.py` and domain plotters |
| PyYAML | `config.yaml` collection/reading in command, evolution and analysis paths |
| h5py | Complete snapshot/catalogue and scientific HDF5 intake/output |
| Pillow | Image loading, sizing and raster/frame handling in visual/movie products |
| imageio | Reading/writing encoded frames and MP4 products in visual pipelines |
| imageio-ffmpeg | FFmpeg encoder used by imageio's requested movie routes |
| pyvista | Explicit interactive volume/particle display routes in `core/visualization.py` |
| packaging | Dependency requirement/version validation in `bootstrap.py` and native backend checks |
| setuptools, wheel | Source-package build tooling for selected pinned optional Python backends |
| agama | `core/halos.py:pinned_agama_backend`; selected potential/action/stream calculations |
| gala | `core/stream_resimulation.py:gala_spray_adapter` and its verified backend loader |
| StreaMAX | `core/stream_resimulation.py:streamamax_reference` for the explicit static-host stream reference |
| fsps | `core/stellar_population.py:pinned_fsps`; requested population synthesis, not native star creation |
| dynesty, emcee | Selected samplers loaded by `core/stream/inference.py:pinned_sampler_backend` |
| cloudpickle | Pinned serialization in `core/stream/inference.py` and `core/stream/sbi.py` |
| sbi | Selected amortized inference/training in `core/stream/sbi.py`; requires its accepted forward inputs |
| lenstronomy | Explicit mass-plane/lensed-observation and cosmography fitting in `core/observations.py` |
| pyrecon | Selected BAO reconstruction in `core/bao.py:apply_recorded_bao_reconstruction` |
| pycorr, Corrfunc | Pinned survey pair counting in `core/bao.py:pycorr_measurement` |
| pytest | Optional `validation` profile for recipient-side checks; not a simulation dependency |

Installing the `base` profile does not install the stream, SPS, inference or cosmography
profiles. Every optional declared source package has a selected consumer; absence from
the ordinary box-run path is not evidence that it is unused software. A backend import
alone does not establish a successful run.

## Installation profiles

`configs/runtime/dependencies.json` is the authoritative declaration of profiles,
package requirements, source revisions, and checksums. Setup follows that declaration.
The profile names below are the names accepted by `--profile`.

| Desired calculation | Dependency profile | What to inspect after the calculation |
|---|---|---|
| Ordinary box analysis and plotting | `base` | Run manifest, saved arrays and figures |
| Potential, action and halo-structure measurements | `halo-structure` | Potential serialization, measurement arrays and support flags |
| Stream experiments | `streams` | Selected adapter's trajectories and its source/units record |
| Stellar population synthesis | `sps` | Population spectrum or band luminosities and model-asset provenance |
| Posterior sampling | `inference` | Samples, likelihood configuration and diagnostics |
| Simulator-based inference | `sbi` | Training/forward-input records, posterior and acceptance checks |
| Lensing, BAO or survey pair counting | `cosmography` | Selected observation/reconstruction product and its input contract |
| Recipient-side regression checks | `validation` | Test results, fixture requirements and explicit skips |
| Halo, SPS and sampling dependencies together | `flagship` | The outputs of the workflows actually selected |
| Every declared dependency group | `all` | Check the selected workflow's products; installation creates no scientific result |

These are product families, not guaranteed filenames. Follow the relevant
[worked software chain](learning-guide.md#what-goes-in-what-the-software-does-and-what-comes-out)
for the exact inputs, commands, units and outputs, then use the
[output catalogue](outputs.md) to inspect the run. Model assets are separate
from Python packages: in particular, FSPS needs its retained physical tables,
and production Grackle chemistry needs the selected network and compatible
data/build configuration. Preserve those assets during cleanup.

### Base analysis (`base`)

**Includes profiles:** none.

- numpy>=2.0
- scipy>=1.14
- matplotlib>=3.9
- PyYAML>=6
- h5py>=3.11
- Pillow>=10
- imageio>=2.35
- imageio-ffmpeg>=0.5
- pyvista>=0.49,<0.50
- packaging>=24

### Stream experiments (`streams`)

**Includes profiles:** base, halo-structure.

- gala — `core/software/streams/gala`
- StreaMAX — `core/software/streams/StreaMAX`

### Stellar population synthesis (`sps`)

**Includes profiles:** base.

- fsps — `core/software/sps/python-fsps`

### Inference (`inference`)

**Includes profiles:** base.

- dynesty — `core/software/inference/dynesty`
- emcee — `core/software/inference/emcee`
- cloudpickle — `core/software/inference/cloudpickle`

### Simulator-based inference (`sbi`)

**Includes profiles:** base, inference.

- sbi — `core/software/inference/sbi`

### Cosmography (`cosmography`)

**Includes profiles:** base.

- lenstronomy — `core/software/cosmography/lenstronomy`
- pycorr — `core/software/cosmography/pycorr`
- pyrecon — `core/software/cosmography/pyrecon`
- Corrfunc — `core/software/cosmography/Corrfunc`

### Halo structure (`halo-structure`)

**Includes profiles:** base.

- setuptools>=75
- wheel>=0.44
- agama — `core/software/streams/agama`

The optional `figures` lab profile and `advanced_controls.py` use this profile
to regenerate known-shape and density-valley examples. They execute the pinned
AGAMA fit/serialization/field checks and production measurements, without native
particle evolution. See [the lab commands](optional-tests.md#start-the-interface).

### Validation (`validation`)

**Includes profiles:** base.

- pytest>=8,<10

### Combined profile (`flagship`)

**Includes profiles:** halo-structure, sps, inference.

## Source references and retained records

The [Python source map](architecture.md#python-source-map) explains application
connections, dynamically loaded sections, scene scripts, and verification-only code. The
portable source contains that maintained guide rather than generated inventory reports
or machine-specific cache listings. Scientific data-contract checks, conservation
ledgers and per-run source records remain because actual readers and acceptance/recovery
workflows consume them.

## Install only the workflows you need

Default setup installs the base analysis packages and builds CLASS, AxiCLASS and
monofonIC. Both cosmology solvers are available for the explicit model choice; a
particular run uses one. GADGET compiles for that run's physical configuration. For a
single cosmology, reduce `--engines` further to the solver you use plus monofonIC.
Optional commands do not run simply because their backend is installed.

Ubuntu prerequisites include **both** `libfftw3-dev` and `libfftw3-mpi-dev`.
The first supplies serial FFTW; the MPI-enabled monofonIC build also requires
the second. The setup script installs both. If a cluster provides libraries
through modules, load the MPI-enabled FFTW development files before selecting
`--no-system-packages`.

```bash
# LCDM box simulations and saved-data figures:
bash scripts/setup.sh --profile base --engines class,monofonic
# Add the independently selected halo/tree tools:
bash scripts/setup.sh --profile base --engines rockstar,consistent-trees,velociraptor
# Explicit MUSIC IC or topology workflows:
bash scripts/setup.sh --profile base --engines music2,disperse
# Python stream experiments; retain installed native engines:
bash scripts/setup.sh --profile streams --engines none
# Every optional profile and native engine (explicit opt-in):
bash scripts/setup.sh --profile all --engines all
```

OpenGadget3 remains an optional executable backend for independent SIDM campaigns. Only
attribution/pinned descriptors remain for the five archived reference-only checkouts;
setup does not install them.

A fresh Ubuntu 24.04 ARM64 container passed base installation, all three default
native builds and source/executable hash verification after the MPI FFTW repair.
The optional native profile separately compiled and evolved a real 32-particle
GADGET control on Linux. See the smoke report for commands and receipts. WSL,
other architectures and the full optional-profile installation remain separate
validation targets.


## Independent OpenGadget3 control

The optional worker uses an existing Linux rare/directed reference build. Start
from [the retained build request](../configs/research/opengadget3_rare_build.example.json),
replace its example paths/checksums with your actual source, compiler, MPI,
library/header dependency manifest and resource allocations, and run the
[source-bound build command](command-reference.md#sidm-reference-build):

```sh
python commands.py sidm-reference-build /absolute/path/to/reference-build-request.json --output ../research2-runtime/my-independent-build
export RESEARCH2_OPENGADGET3_BUILD="$PWD/../research2-runtime/my-independent-build/reference-build"
python optional_tests/native_checks.py opengadget3 --output ../research2-runtime/optional-lab/my-independent-32 --seed 42
```

The worker does not build dependencies. Its production driver passes short
relative native filenames from the owned input directory, so long external
output roots do not overflow the pinned engine's 100-character filename buffers.
It verifies the actual completed build
capability and compile/input hashes through the production runner. A stale
capability fails with its changed input; do not edit its hashes to make it
pass. Build a new reference directory instead. The supplied build recipe
requires Linux. The worker's high-rate 100 cm²/g toy law runs 32 beam particles
with gravity disabled and retains exact IDs at three epochs. It checks mass,
momentum and kinetic energy; changed-velocity particles are not counted as
scattering events. The optional notebook accepts `RESEARCH2_LAB_SIDM` pointing
to a completed standalone worker directory, so its saved Linux outputs can be
inspected without executing that binary on another platform.
