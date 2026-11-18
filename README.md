# Research2

Research2 connects cosmological initial conditions, native particle simulations,
scientific measurements, and visual exports. It supports CLASS/AxiCLASS, GADGET-4, halo
catalogues, and selected gas, stellar, black-hole, SIDM, zoom, stream, and observation
workflows.

Start with a connected run, or analyse snapshots you have already saved. Physical models
and numerical settings are chosen explicitly. Export status and scientific validation
are recorded separately.

Follow the [completion checklist](docs/completion-plan.md) for the remaining work
and the evidence required before this task can be called complete.

## Choose a starting point

| Your goal | Start here |
|---|---|
| Learn the software and follow a first run | [Hands-on learning guide](docs/learning-guide.md) |
| Run a small calculation and inspect the evidence | [Optional observable lab](docs/optional-tests.md) |
| Deploy to HPC Slurm cluster (512³ production / 32³ validation) | [HPU cluster guide](TRANSFER_AND_CLUSTER_GUIDE.md) |
| Run an end-to-end automated study | [Study roadmap](docs/one-command-research-roadmap.md) · [Study driver](docs/study.md) |
| Analyse existing snapshots | [Saved-run commands](#analyse-saved-snapshots) |
| Find an argument or understand a Python file | [Command reference](docs/command-reference.md) · [File/callable reference](docs/python-reference.md) |
| See what actually ran and what remains | [Smoke report](docs/smoke-test.md) · [Release status](docs/finalization.md) |

The optional lab connects equations, units, Python source, numerical arrays and
original figures. Its quick profile defaults to **32 total synthetic particles**.
For cosmological generators, `particles_per_dimension` is a cubic-grid setting:
`4` means **64 particles per species**, and `32` means **32,768 per species**.

| Optional lab choice | What it runs |
|---|---|
| `quick` / `visual` | Seven numerical measurements; larger synthetic views for the visual profile |
| `combined` | Quick measurements, twelve library calculations and one actual 32-particle GADGET control; 20 steps |
| `engines` | Adds cosmology, IC, Blender, chemistry, finder, tree, topology and independent SIDM workers; 31 steps |
| `physics` | Quick measurements plus native cooling, stellar births and seeded BH accretion; 10 steps |
| `figures` | Quick measurements plus portable shape, density-valley and flyby examples; 10 steps |
| `complete` | All of the above check families in one 37-step checklist |

Backend counts differ: the gas controls start with 64 particles; the BH case adds
one seeded sink. Other IC/catalogue/mesh controls use their documented counts.
Missing dependencies remain `UNAVAILABLE`. See the [lab guide](docs/optional-tests.md)
for setup, exact commands, resource bounds, outputs and notebook use.

## Start a connected run

Inside Ubuntu 24.04 or WSL2, from the extracted source folder:

```bash
bash scripts/setup.sh
source ../research2-runtime/environments/default/bin/activate
python commands.py run --count 1 --draft /absolute/path/to/setup.json
```

Setup installs the base Python profile and the default linear/IC engines. See
[software](docs/software.md) for optional profiles and engine selection.

The terminal walkthrough saves your answers and supports help, back/edit, and
continuation. Enter a schedule such as `10, 5*, 2, 1*, 0*` to save all five redshifts
and highlight 5, 1, and 0. Highlights preserve the full saved schedule and movie
timeline.

## Analyse saved snapshots

Inspect applicability before starting analysis:

```bash
# Using run_analysis.py CLI:
python run_analysis.py /absolute/path/to/RUN --graphs available --dry-run
python run_analysis.py /absolute/path/to/RUN --graphs available --find-halos

# Or using commands.py:
python commands.py analysis /absolute/path/to/RUN --graphs available --dry-run
python commands.py analysis /absolute/path/to/RUN --graphs available
python commands.py results /absolute/path/to/RUN --verify
```

Open `RUN/results/index.html` for **Highlights**, **By redshift**, **Evolution**,
**Movies**, **Data**, and **Status**. Each attempt retains its own outputs and failure
record.

Choose `available` to run products with ready inputs and record omissions. Choose `all`
to request every physically applicable product, including those that require additional
measurement requests. Neither option enables new physics.

## How the pipeline works

```text
Settings
  → CLASS/AxiCLASS spectra and background
  → validated initial particles
  → GADGET gravity and selected gas/star/BH physics
  → saved HDF5 epochs and configured catalogues
  → measured arrays
  → figures, movies, and results/index.html
```

Hydro, star formation, delayed stellar returns, black holes, and synthetic stellar light
are separate choices. Graphs read saved particle/catalogue fields or supplied
experimental products, apply their unit and selection conventions, and measure
quantities before plotting. Numerical data and source records accompany the figures.

The [full walkthrough](docs/architecture.md#a-full-hydro-stars-and-black-hole-run)
explains the Python/native files, HDF5 fields, and graph-making pseudocode.
[Calculations](docs/calculations.md) covers the governing equations;
[software](docs/software.md) identifies each backend's consumer.

For CLASS/HMF predictions, selecting full GADGET experiments and learning a
fitting function from measured halo counts, follow the [HMF research workflow](docs/hmf-research.md).
It explains training inputs, cosmology splits and resolution comparisons.

## Documentation

| Guide | Use it to understand |
|---|---|
| [Configuration](docs/configuration.md) | Prompts, saved answers, redshifts, highlights, and preflight |
| [Architecture](docs/architecture.md) | The complete data flow and every first-party Python file |
| [Calculations](docs/calculations.md) | Equations, units, estimators, cuts, and uncertainty limits |
| [Figure reference](docs/figure-contracts.md) | The 102 registered graph recipes and their input requirements |
| [Software](docs/software.md) | Backend roles, dependency profiles, and optional installation |
| [Outputs](docs/outputs.md) | Result navigation, naming, movies, and the single-run catalogue |
| [Troubleshooting](docs/troubleshooting.md) | Interrupted setup, failed stages, recovery, and storage |
| [Operator guide](docs/operator-guide.md) | Installation and detailed commands for individual workflows |
| [Release status](docs/finalization.md) | Completed work, historical evidence, and remaining validation |

## Share the source

Write the portable archive outside the source folder:

```bash
python commands.py package --output /absolute/path/to/research2-source.zip
```

The archive includes first-party code and guides, declared backend source, required
physical tables, and licenses. It excludes local environments, compiled binaries, run
outputs, caches, and retired downloads. The recipient runs setup for their machine.
Example JSON files require their own asset paths, run paths, and physical settings.

## Validate after transfer

With the installed environment active, the recipient can run:

```bash
export PYTHONDONTWRITEBYTECODE=1
python -m unittest discover -s tests -v
python verification.py -v
python -m unittest discover -s optional_tests -p 'test_*.py' -v
python -m pytest -p no:cacheprovider tests/test_empty_catalogue_shard.py tests/test_central_point_potential.py tests/test_blender_portable_scene.py -q
```

The first two commands run the application regression tests; the third checks the
optional lab. The fourth runs five standalone pytest checks that `unittest` does
not collect; it requires pytest and Agama for the central-potential check.
Tests that need retained native fixtures report skips when those
fixtures are absent. These checks include fixtures and mocks. To execute fresh
32-particle calculations and inspect their outputs, use the optional lab above.

Keep the source folder for code, configuration/examples, documentation, tests,
required backend sources, licenses and physical tables. Installed environments,
builds, simulation outputs and logs belong in the external `research2-runtime`
folder. The optional teaching images, arrays and notebook stay together under
`optional_tests/`. Large FSPS stellar tables are required inputs, not caches.

Fresh execution results, retained artifacts and remaining coverage gaps are listed in
the [smoke-test report](docs/smoke-test.md). Platform-specific and scientific
acceptance checks are tracked in [release status](docs/finalization.md).

Tiny runs verify execution and specific numerical checks. Resolved halo,
galaxy, black-hole and cosmological conclusions require appropriate populations,
convergence and independent validation.

For the experimental bounded study driver, including dry-run, provenance and recovery limits, see the [study guide](docs/study.md).
