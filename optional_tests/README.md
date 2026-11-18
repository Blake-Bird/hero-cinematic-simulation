# Optional observable lab

Run small calculations, inspect their equations and source calls, and replay saved
figures in a notebook. This folder is optional; ordinary simulations and analysis
work without it. It needs the matching repository and installed dependencies.

## Start here

From the repository root:

```sh
source ../research2-runtime/environments/default/bin/activate
python optional_tests/lab.py serve
```

Open the printed `http://127.0.0.1:8765` URL. Choose **Quick** for seven checks
using **32 total synthetic particles**. The report shows each calculation,
assertion, unit, Python source and output. These are production readers and
measurements on an explicit fixture; species markers do not demonstrate star
formation or BH evolution.

For the same quick checks without a server:

```sh
python optional_tests/lab.py run --profile quick --particles 32 --seed 42 --output ../research2-runtime/optional-lab/my-first-32
```

Choose a new directory outside the source tree, including resolved symlink paths.
Open its `report.html`; inspect `run.json`, `python-calls.json`, original HDF5 and
numerical NPZ arrays. Parent-thread calls, isolated workers and archived tests
are separate evidence. `FAIL` and `UNAVAILABLE` remain visible.

## Choose the next walkthrough

| Goal | Lab profile | What it adds |
|---|---|---|
| Larger density and viewing-zoom images | `visual --particles 1024` | Same seven calculations; no extra physical resolution or movies |
| A real 32-particle self-gravity run | `native` | One GADGET control; needs the native compiler/MPI toolchain |
| Check installed optional Python libraries | `libraries` | Twelve isolated calculations; needs their dependencies and assets |
| Regenerate shape, density-slope and flyby examples | `figures` | Three fixed known-input controls; AGAMA needed for shapes/splashback |
| Native engines, libraries, gas, births, BH and figures | `complete` | 37 steps with all prerequisites; see the full guide before running |

`--particles` controls the quick fixture. Backend inputs have their own fixed
counts: gas controls use 64 particles, shapes use 6,000, and the density-valley
control uses 32,768. Native tools must already be installed. Workers install
nothing. Missing dependencies stay unavailable; a cold GADGET build may take
minutes. These profiles generate no videos.

Read the [full guide](../docs/optional-tests.md) for all nine profiles,
requirements, exact backend calculations, resource bounds, output files,
[notebook setup](../docs/optional-tests.md#use-the-notebook) and removal.
Open [Walkthrough.ipynb](reference_gallery/Walkthrough.ipynb) in the same Python
environment; it reads saved arrays and receipts rather than silently rerunning
engines. Archived graph examples need their own original inputs for a fresh
producer run. Graph-gain calibration remains unresolved.

## Keep or remove

Keep this folder as `optional_tests/` beside `core/`, `configs/` and `docs/` when
moving it; it is not a standalone simulator. Runtime outputs stay external.
Stop the server with Ctrl-C before removing the optional folder or its selected
runtime directories. Preserve research runs, checkpoints, physical tables and
licenses according to their separate ownership rules. An archive is optional.
