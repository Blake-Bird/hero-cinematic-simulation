# Architecture and full-run walkthrough

This guide follows a run from configuration to saved measurements and visual outputs.
Start with the walkthrough, then use the source map when you need to locate an
implementation.

- [Full hydro, stellar, and black-hole run](#a-full-hydro-stars-and-black-hole-run)
- [Locations and ownership](#locations-and-ownership)
- [Python source map](#python-source-map)
- [How historical module names load](#how-historical-module-names-load)
- [Portability and source hygiene](#portability-and-source-hygiene)

`commands.py` dispatches public commands. `core/cli.py` coordinates a fresh run, and
`core/setup_walkthrough.py` collects and reviews settings. A configuration contains
explicit physical and numerical choices; a run name is an identifier.

1. CLASS or AxiCLASS produces a new, checksummed `spectrum.json` bundle containing
   linear spectra, transfers and background histories.
2. The particle driver binds that exact bundle, validates the chosen physical
   models, prepares ICs, compiles the matching GADGET configuration and evolves.
3. Saved native snapshot sets and configured finders supply analysis. Complete
   shards, exact particle IDs and dataset unit contracts are checked by readers.
4. `process_run` resolves output applicability and records a workflow attempt.
   Scientific measurements, optional host requests and visual exports use their
   corresponding adapters. Independent optional failures are recorded and later
   exports continue.
5. `core/run_results.py` publishes navigation from that attempt's recorded
   artifacts. It preserves previous attempts, does not select an unrelated newest
   directory, and keeps older figures separate from a failed current attempt.

## A full hydro, stars and black-hole run

Start with `python commands.py run --count 1 --draft /absolute/path/to/setup.json`. A
**full run** evolves the models and produces the outputs you select. Each physical
source model must be enabled explicitly. In particular, gas, star formation, delayed
stellar returns, black holes and synthetic stellar light are separate choices with
different input requirements.

### 1. Settings and the initial universe

`core/cli.py:run_pipeline_main` collects the cosmology through `core/linear.py`. CLASS
handles the standard selected cosmologies; AxiCLASS handles the EDE branch. These solve
the linear cosmological perturbation and background problem, producing spectra, species
transfer functions and expansion history. Their `spectrum.json` identifies the actual
input settings, output files and checksums. It is passed directly to particle
preparation; the particle driver uses that exact bundle rather than searching older
output directories.

`commands.py:_gather_answers` collects box size, particles per dimension, seed, physical
models, numerical tolerances, resources, saved redshifts and display settings.
`core/setup_walkthrough.py` saves responses, supports help/back/edit, and shows the
review. `commands.py:build_config` translates answers into the configuration consumed by
the solver and readers. Native option validation in `core/evolution.py` rejects unknown
or incompatible settings.

For hydro choose `physics=hydro`. The gas model is one of:

| Choice | What evolves |
|---|---|
| `adiabatic` | Gas dynamics and gravitational evolution; radiative cooling and star formation are disabled. |
| `cooling` | Gas plus the selected radiative cooling/UV model; no star-formation conversion. |
| `star_formation` | Gas plus the Springel–Hernquist multiphase/star-formation prescription and its explicitly entered parameters. |

Black holes are a separate opt-in with seed rules, kernel radius, efficiencies, source
limits and allocation cap. Delayed stellar evolution and Grackle chemistry also require
explicit compatible model/library/table settings. `available` and `all` select
**analysis products**, not new physical source models.

### 2. From spectra to particles and a native executable

`core/evolution.py:prepare_and_run` binds the selected linear bundle, prepares or
validates initial conditions, checks the physical assets and resource limits, and
compiles the matching GADGET executable. `core/initial_conditions.py` owns
linear-bundle/IC validation. monofonIC supplies its selected collisionless route; MUSIC
or the supported native generator/handoff route supplies the explicitly configured
alternative. Hydro requires compatible baryon and dark-matter fields; an unvalidated
split of a collisionless load is not a replacement for the selected hydro transfer
prescription.

`gadget_config_text` selects compiled features; `gadget_param_text` writes native
parameter values. These are different inputs: compiling a cooling feature does not
supply its physical table, and a JSON option cannot add an absent compiled feature. The
run saves `config.yaml`, `manifest.json`, `input/param.txt`, `input/output_times.txt`,
its build/IC records and the native executable/build configuration used. Native and
Python build state is machine-specific and lives outside the portable source where
supported.

Requested redshifts become scale factors `a=1/(1+z)` in the native output list. A list
such as `10,5*,2,1*,0*` still saves all five epochs. The stars are display highlights,
not instructions to discard the other snapshots.

### 3. What the evolving particles represent

GADGET advances gravity and the configured gas/source physics. Relevant source locations
are below; the equation details and conventions are in [calculations](calculations.md).

| Component | Implementation that governs the evolution |
|---|---|
| Gravity, force split and integration | GADGET `src/gravity/`, `src/time_integration/`; configuration/build driver in `core/evolution.py` |
| Gas/SPH and pressure forces | GADGET `src/sph/`, with the selected density/entropy and numerical controls |
| Primordial radiative cooling and UV table | `src/cooling_sfr/cooling.cc`, `cooling.h`, and the selected TREECOOL asset |
| Multiphase star-forming gas | `src/cooling_sfr/sfr_eos.cc` |
| Gas-to-star conversion and committed birth state | `src/cooling_sfr/starformation.cc`, `src/data/research_stellar_birth.h` |
| Delayed stellar mass/element/energy return | `src/cooling_sfr/research_stellar_returns.cc` and its pinned model table |
| Optional Grackle network | `src/cooling_sfr/research_grackle.cc`, compatible compiled Grackle library and declared cooling table |
| BH seeds, gas accretion and feedback | `src/black_holes/black_holes.cc`, seed-reservoir headers and the selected optional dynamics headers |
| HDF5 field names and stored-unit factors | `src/io/snap_io.cc` |

All `src/` paths in this table are under `core/software/gadget4/`. They describe the
maintained native source; its pristine upstream and patch payloads are retained for
source identity and portable release generation.

A star particle represents a coeval stellar population with a birth time, initial mass
and metallicity. It is not a resolved individual star or a direct stellar image.
Optional FSPS synthesis later converts population properties into model luminosities
using separately selected stellar libraries. It does not advance the simulation or
replace the native source history.

A type-5 BH carrier has a gravitational mass and a separate subgrid BH mass. The
dense-gas seed routes either convert an eligible gas carrier or assemble a bounded donor
reservoir according to the selected rule. Accretion removes gas, grows the subgrid mass
after radiative losses, and couples the chosen feedback. The ordinary interactive
thermal model does not enable BH mergers. Authored optional binary/dynamics experiments
have their own configuration and scope.

### 4. What is saved, and what the graphs actually read

Snapshots are HDF5 sets in `RUN/data/`, including sharded `snapdir_###` sets when
configured. Header metadata and dataset unit attributes are read together. The storage
number is an identifier; `Header/Time` and the epoch contract establish physical time. A
missing shard or required field is an unavailable measurement, not a zero-valued result.

| Data in a saved snapshot or catalogue | Consumer and resulting product |
|---|---|
| `PartType1` coordinates, velocities, IDs and masses | Matter maps, spatial/velocity measurements and selected particle analyses |
| `PartType0` density, internal energy, composition/electron state and masses | Gas density/temperature distributions, phase diagrams, projected gas maps and supported gas budgets |
| Gas `StarFormationRate` | Whole-snapshot SFR and its time history, after physical-unit conversion |
| `PartType4` coordinates, current masses, IDs and birth fields | Stellar maps/counts/ages; host membership and trajectories when explicitly requested |
| `SSPInitialMass`, `SSPInitialLockedMass`, `SSPBirthNativeTime`, `SSPBirthMetallicity`, parent IDs and source state | Native birth/return accounting and optional population synthesis with the actual run clock |
| `PartType5` `Masses`, `BH_Mass`, `BH_InflowRate`, `BH_RadiatedMass`, `BH_FeedbackEnergy` and source fields | Measured BH mass/inflow/luminosity/energy histories |
| Native FOF/SUBFIND masses, positions, radii, membership and subhalo links | HMFs, concentrations, supported profiles/shapes, halo overlays and subhalo exports |
| Configured Rockstar/Consistent Trees catalogues | Independent finder comparisons and measured descendant/main-progenitor histories |
| Native matter-spectrum output and saved estimator metadata | Matter power curves, selected fold bands, noise treatment and mode-count uncertainties |

`core/data.py` applies the stored a/h and physical-unit contracts. `core/analysis.py`
measures supported scientific products; `core/halos.py`, `core/stars.py` and
`core/black_holes.py` supply specialized catalogue/source readers. Particle IDs are
exact integers, including when their values exceed floating-point exact precision.
Several older logical reader modules are sections of these physical files; the source
map below explains how those names resolve.

### 5. From saved data to a scientific figure

There are three related production paths:

1. **Scientific statistics:** `commands.py:process_run` resolves prerequisites,
   calls `analyze_run(..., products=...)`, and publishes a fresh
   `analysis_products_*` generation. Measurements and status go into
   `analysis_products.h5` and `analysis_products_summary.json`. Domain plotters
   use those measured arrays; `core/figures.py:export_figure` handles the display,
   formats and numerical/metadata sidecars.
2. **Particle maps and time movies:** `core/visual_pipeline.py` dispatches
   `core/visualization.py`. Maps measure/project actual saved particles for the
   selected field, plane and depth. Movies encode their recorded saved frames.
   A halo movie takes the current selected history table. Missing map frames for
   particles that exist are failures; star-free epochs can be omitted
   with their reasons recorded.
3. **Optional measured/campaign figures:** authored requests provide host IDs,
   apertures, supplied posterior samples, matched runs or observation geometry.
   Their producer contracts identify their measurements, not just a callable
   renderer. See [all registered graph contracts](figure-contracts.md) and
   [the output catalogue](outputs.md).

A static HMF example makes the data path concrete:

```text
read the complete saved halo catalogue for an epoch
read its declared mass definition and units
select finite positive masses passing the configured particle-support cut
count those masses in the configured mass bins
compute volume V = box_side**3
compute abundance = counts / (V * bin_width_in_log10_mass)
compute supported count intervals / spatial resampling statistics
plot bin centers, abundance, uncertainty and explicit empty-bin upper limits
export PNG/PDF/SVG, plotted arrays and figure metadata
record files in this analysis generation's summary
link them from this run's results page
```

The actual histogram implementation is `core/analysis.py:hmf_arrays`; the support cuts,
uncertainty and plot preparation are separate functions. Fits are predictions from the
saved linear inputs, not additional halo detections.

A gas temperature–density plot uses actual gas density and internal energy plus the
source's composition/EOS conventions, then bins the converted values. It never infers
temperature from the colour of a previously drawn image. A BH luminosity panel uses
saved source-step inflow and the selected radiative efficiency; it does not solve for a
resolved accretion-disk spectrum.

`export_figure` writes a `.figure.json` record, `.plotdata.npz` arrays and the selected
image formats. The record retains axes, scientific metadata, masks, layout information
and artifact hashes. Drawn artist geometry is distinguished from the source physical
measurements. Maps and movies have their own source, numeric and publication contracts;
a compressed movie is not the authoritative measurement array.

### 6. Results, failures and navigation

`core/run_results.py` consumes recorded workflow/artifact manifests. It publishes
`RUN/results/index.html` with Highlights, By redshift, Evolution, Movies, Data and
Status. An epoch page such as `snap_012_z_1.000` uses both the storage ID and rounded
actual redshift; metadata retains the unrounded value. History products and movies span
epochs and therefore also have run-level views.

Each workflow attempt keeps its own artifacts. The latest failed attempt stays visible
as failed; older analysis is selectable separately. Standalone old analysis summaries
and movie manifests are explicitly historical exports, not proof that the current
workflow completed. Files are linked once rather than copied into every redshift folder.
Native snapshot names stay stable for readers.

Independent optional render/measurement failures are recorded and later exports
continue. Status includes the reason and retry guidance. Tasks explicitly selected for
later host/sample input stay pending. Run completion, product completion and scientific
acceptance are separate states.

### End-to-end pseudocode

This is a reading guide, not a second implementation or a set of physical defaults:

```text
collect_and_review_cosmology()
linear_bundle = run_selected_CLASS_or_AxiCLASS()
collect_and_review_simulation_settings(linear_bundle)
    choose hydro model, stellar sources and BH model explicitly
    choose saved redshifts, display highlights, products and optional requests
validate_configuration_assets_resources_and_output_location()
bind_exact_linear_bundle_and_prepare_or_validate_initial_conditions()
compile_GADGET_for_the_selected_physics()
evolve_and_save_complete_native_epochs()
plan_products_and_catalogue_prerequisites_from_saved_content()
run_configured_catalogue_prerequisites()
gate_downstream_products_on_actual_finder_results()
resolve_ready_measurement_and_export_work()
for each ready scientific product:
    read -> verify units/IDs/epoch -> measure -> write arrays/status -> plot
for each ready independent optional export:
    attempt export; record failure and continue other independent exports
publish_workflow_status_and_retry_guidance()
publish_results_homepage_from_recorded_artifacts()
```

A process abort, incomplete shard set, missing library, unavailable host request or
absent physical species takes its corresponding failure/pending/inapplicable path.
Pseudocode does not override those branches in the real implementation.

## Locations and ownership

Portable source lives in `research2/`; writable environments, native builds, reusable
caches, runs and evidence live in the external `research2-runtime/`.
`RESEARCH2_STATE_ROOT` selects another external root. Native checkpoints and published
products retain their original provenance and ownership rules.

### How historical module names load

`core/__init__.py` reads the six aliases in `configs/core_source_layout.json` and
installs an import finder. For example, importing `core.sidm_analysis` loads the
`# @research2-module sidm_analysis` section of `core/sidm.py`. The corresponding
end marker bounds its code. The loader pads preceding lines so tracebacks retain
physical source line numbers, and applies the declared Python future flags.

The short name `sidm_analysis` and qualified name `core.sidm_analysis` resolve to
the same module object. Different logical sections have separate module dictionaries,
even when their text shares a physical file. This preserves their independent
helpers and constants. `core.source_path(name)` locates the physical file;
`core.source_section(name)` returns the section text and its first line for tooling.
Importing a section executes its Python module initialization; it does not launch
the native simulation workflow.

Identical helper names across those sections are not sufficient evidence of dead
code. Inspect aliases, CLI registrations, dynamic lookups and subprocess callers
before removal. The layout file retains its original migration inventory; use the
current source reference for the complete file inventory.

The schedule parser and result-navigation code are separate modules. Large scientific
consolidated modules are retained: moving them requires dedicated import, provenance and
numerical equivalence checks. See the finalization checklist for outstanding refactoring
and validation.

## Execution and acceptance

A workflow's product status describes exports. Scientific acceptance is separate and
needs its declared controls and evidence. The run homepage exposes workflow records,
missing inputs and failure messages. Installation receipts establish installation
integrity; they do not establish successful scientific execution.

## Python source map

For input signatures and implementation documentation, see the generated
[Python reference](python-reference.md). For invocation arguments, see the
[command reference](command-reference.md). Regenerate both with
`python scripts/build_reference.py` after changing interfaces.

The tables below cover the 188 core/root implementation files: 186 under `core/`
plus `commands.py` and `bootstrap.py`. The wider application inventory has 265
files when the root verification dispatcher, 56 scripts and 20 optional-lab files
are included. Their per-file entries are in the [Python reference](python-reference.md).
Vendored upstream Python and tests are separate from these application counts.
The conceptual connections below are based on
parsed imports, registered logical aliases, literal dynamic-module names and explicit
source/script references. An incoming reference shows a connection, not execution in
every run. `commands.py` and `bootstrap.py` are entry points; optional command/workflow
modules run only when selected.

Package initializers are imported through Python package loading. Some helpers
are called by scripts or explicit Python APIs rather than registered commands;
`core/native_cooling_control.py` is a verification-only native numerical control.
The `verification.py` dispatcher and files under `tests/` serve recipient-side
verification. This map describes connections; actual execution and source-compatible
regression evidence are documented in the [smoke report](smoke-test.md).

Several modules are copied into scenes or loaded from a named workflow rather than
imported in an ordinary `import` statement. Historical logical aliases must be resolved
through `configs/core_source_layout.json`; deleting one of those sections because no
same-named `.py` file exists would break the loader.

All 188 implementation files have module docstrings. 1,648 of 3,809 function/method
definitions have docstrings; that count includes nested/private helpers and is not a
claim of complete function documentation. The largest scientific files still use
consolidated, named sections. Their responsibilities are documented, but splitting them
safely remains future work. Physical equations, unit boundaries and source/measurement
ownership are explained above and in the calculation guide. Test methods primarily
document their behavior through descriptive names and assertions.

### Entry points and orchestration

| File | Responsibility | Connected from |
|---|---|---|
| [bootstrap.py](../bootstrap.py) | Python profile installation and native setup forwarding | `scripts/setup.sh` |
| [commands.py](../commands.py) | Public command dispatch and run/analysis orchestration | Public entry point |
| [core/__init__.py](../core/__init__.py) | Package initialization and historical logical-module loading | Python package loading |
| [core/batch_analysis.py](../core/batch_analysis.py) | Sequential saved-run analysis and explicit pairwise comparison publication | `commands.py` |
| [core/batch_comparison.py](../core/batch_comparison.py) | Compare selected-run halo mass functions without replacing individual analysis | `core/batch_analysis.py` |
| [core/batch_results.py](../core/batch_results.py) | Human-readable navigation for sequential analysis and comparison outputs | `core/batch_analysis.py` |
| [core/batch_simulation.py](../core/batch_simulation.py) | Collect multiple interactive run configurations, then execute sequentially | `core/cli.py` |
| [core/campaigns.py](../core/campaigns.py) | Prepare native campaign recipes through one command, without evolving runs | `commands.py`, `core/response_campaign.py` |
| [core/cli.py](../core/cli.py) | Small public command handlers, consolidated from root launch scripts | `commands.py`, `core/workflow.py` |
| [core/cli_prompts.py](../core/cli_prompts.py) | Explicit grouped numerical prompts; no recipes or inferred parameter values | `commands.py`, `core/linear.py` |
| [core/data.py](../core/data.py) | Native units, exact identities, data contracts and scientific publication | `commands.py`, `core/analysis.py` |
| [core/epochs.py](../core/epochs.py) | Saved output schedules and display highlights; neither changes native names | `commands.py`, `core/analysis.py` |
| [core/foundation.py](../core/foundation.py) | Shared units, durable file publication and scientific product contracts | `commands.py`, `core/allsky_map.py` |
| [core/graph_producer_registry.py](../core/graph_producer_registry.py) | Reconcile all 102 registered graphs against actual callable producer routes | `core/cli.py`, `core/workflow.py` |
| [core/native.py](../core/native.py) | Native software identity, staging, builds and owned process execution | `commands.py`, `bootstrap.py` |
| [core/numeric.py](../core/numeric.py) | Bounded NumPy storage: streamed NPZ contracts and disk-backed publication | `core/data.py`, `core/merger_tree_map.py` |
| [core/release.py](../core/release.py) | Portable source selection and the corresponding native GADGET patch payload | `commands.py`, `core/cosmography.py` |
| [core/resources.py](../core/resources.py) | CPU, memory, storage estimates and external runtime locations | `commands.py`, `core/analysis_figure_campaign.py` |
| [core/results_gallery.py](../core/results_gallery.py) | Build an offline, curated navigation page for existing scientific outputs | `commands.py` |
| [core/run_results.py](../core/run_results.py) | Manifest-driven, relocatable navigation across immutable workflow attempts | `commands.py` |
| [core/run_status.py](../core/run_status.py) | Human-readable recovery commands derived from a workflow's explicit requests | `commands.py` |
| [core/setup_walkthrough.py](../core/setup_walkthrough.py) | Resumable terminal questionnaires and prelaunch review without physics presets | `commands.py`, `core/batch_simulation.py` |
| [core/simulation_queue_index.py](../core/simulation_queue_index.py) | Readable index for an interactive sequential simulation queue | `core/batch_simulation.py` |
| [core/snapshot_tables.py](../core/snapshot_tables.py) | Human-readable and machine-readable tables of recorded snapshot epochs | `core/analysis.py`, `core/run_results.py` |
| [core/suite_comparison.py](../core/suite_comparison.py) | Overlay selected measurements from every run at a common saved epoch | `core/batch_analysis.py` |
| [core/workflow.py](../core/workflow.py) | Content-bound, capability-aware product DAG with durable stage generations | `commands.py`, `core/analysis_figure_campaign.py` |

### Physics and measurements

| File | Responsibility | Connected from |
|---|---|---|
| [core/action_angle.py](../core/action_angle.py) | Actions, angles, frequencies and explicit orbit-validity diagnostics | `core/native_stream_responses.py`, `core/research/display.py` |
| [core/analysis.py](../core/analysis.py) | Snapshot/catalogue measurements and scientific bundle publication | `commands.py`, `core/analysis_figure_campaign.py` |
| [core/analysis_figure_campaign.py](../core/analysis_figure_campaign.py) | Connect original cosmography/stream analysis campaigns to registered figures | `core/cli.py`, `core/flagship_analysis_profile.py` |
| [core/apertures.py](../core/apertures.py) | Measured native selection apertures and their contamination histories | `commands.py`, `core/workflow.py` |
| [core/assembly.py](../core/assembly.py) | Native halo/galaxy assembly, orbital events, workflow stages and figures | `core/cli.py`, `core/campaigns.py` |
| [core/bao.py](../core/bao.py) | BAO catalogue measurements, template fits and published compressed likelihoods | `core/cli.py`, `core/cosmography.py` |
| [core/baryon_products.py](../core/baryon_products.py) | Measured native gas, stellar-return, metal and population products | `core/workflow.py`, `core/flagship_products.py` |
| [core/black_hole_ledger.py](../core/black_hole_ledger.py) | Native black-hole seed and accretion ledgers, with conservation replay | `core/workflow.py`, `core/black_holes.py` |
| [core/black_holes.py](../core/black_holes.py) | BH parameter contracts, reference equations, and saved-source histories | `commands.py`, `core/blender.py` |
| [core/cosmography.py](../core/cosmography.py) | Authored cosmography campaigns and measured ladder/full-lens intake | `commands.py`, `core/cli.py` |
| [core/cosmology.py](../core/cosmology.py) | Cosmological configuration, backgrounds, and shared calculations | `commands.py`, `core/analysis.py` |
| [core/evolution.py](../core/evolution.py) | Native configuration, IC handoff, execution, and checkpoint continuation | `commands.py`, `core/analysis.py` |
| [core/expansion_products.py](../core/expansion_products.py) | Post-fit expansion recovery, explicit domain decisions and original figures | `core/cli.py`, `core/workflow.py` |
| [core/first_stars.py](../core/first_stars.py) | Native first-cooling/birth analysis and resolved-cloud products | `core/workflow.py`, `core/assembly.py` |
| [core/gravity_fields.py](../core/gravity_fields.py) | Host motion, reconstructed fields and extended gravity components | `core/matched_perturber_figures.py`, `core/sidm.py` |
| [core/halo_campaign.py](../core/halo_campaign.py) | Prepare halo product campaigns and measure exact recovered descendants | `core/workflow.py`, `core/campaigns.py` |
| [core/halo_products.py](../core/halo_products.py) | Measured halo structure, splashback and cosmic-web products and figures | `core/cli.py`, `core/flagship_history.py` |
| [core/halos.py](../core/halos.py) | Finder adapters, catalogue readers, and halo/stellar structure measurements | `commands.py`, `core/action_angle.py` |
| [core/hmf_fitting_functions.py](../core/hmf_fitting_functions.py) | Additional, analysis-only HMF comparisons; never changes a simulation or old plot | `core/analysis.py` |
| [core/initial_conditions.py](../core/initial_conditions.py) | IC growth/transfer calculations and linear/particle consistency checks | `commands.py`, `core/analysis.py` |
| [core/linear.py](../core/linear.py) | Standalone CLASS/AxiCLASS linear bundles and their public command | `commands.py`, `core/batch_simulation.py` |
| [core/matched.py](../core/matched.py) | Matched cosmology/object comparisons, population statistics and workflows | `commands.py`, `core/cli.py` |
| [core/matched_perturber_figures.py](../core/matched_perturber_figures.py) | Registered BH/extended-profile figures from the existing template engine | `core/workflow.py` |
| [core/native_baryon_campaign.py](../core/native_baryon_campaign.py) | Native baryon histories and cooling/birth/cloud recipes through existing estimators | `core/workflow.py`, `core/campaigns.py` |
| [core/native_bh_campaign.py](../core/native_bh_campaign.py) | Native BH history/journal/image producers and executable graph recipes | `core/workflow.py`, `core/campaigns.py` |
| [core/native_convergence_campaign.py](../core/native_convergence_campaign.py) | Connect prepared native cases to the original measured convergence engine | `core/campaigns.py` |
| [core/native_convergence_catalogue.py](../core/native_convergence_catalogue.py) | Finder-sensitive catalogue mass profiles with fixed spatial jackknife bins | `core/workflow.py`, `core/native_convergence_campaign.py` |
| [core/native_cooling_control.py](../core/native_cooling_control.py) | Verification-only isolated native cooling control | `verification.py` |
| [core/native_gadget_ancestry.py](../core/native_gadget_ancestry.py) | Read GADGET's recorded SUBFIND descendant/progenitor tables exactly | `core/workflow.py`, `core/flagship_campaigns.py` |
| [core/native_scene_overlay.py](../core/native_scene_overlay.py) | Original encounter histories to global saved overlays and local event views | `core/workflow.py`, `core/research/display.py` |
| [core/native_stream_responses.py](../core/native_stream_responses.py) | Native paired histories → existing fixed-track response estimators | `core/workflow.py`, `core/response_campaign.py` |
| [core/observation_data.py](../core/observation_data.py) | Observed-data contracts, covariance fits and immutable estimator products | `core/workflow.py`, `core/bao.py` |
| [core/observation_geometry.py](../core/observation_geometry.py) | Native observation geometry, stellar attenuation and survey measurements | `core/workflow.py`, `core/apertures.py` |
| [core/observations.py](../core/observations.py) | Background expansion and observer/light-cone calculations; Doppler/aberration transforms, lens planes, time-delay distances and selected H₀ fits | `core/analysis.py`, `core/analysis_figure_campaign.py` |
| [core/periodic_images.py](../core/periodic_images.py) | Read native image counters and bind them to exact measured worldline rows | `core/workflow.py`, `core/flagship_products.py` |
| [core/power.py](../core/power.py) | Read, compare and export linear and measured native power spectra | `core/cli.py`, `core/workflow.py` |
| [core/response_campaign.py](../core/response_campaign.py) | Prepare native response campaigns and bind their actual generated inputs | `commands.py`, `core/workflow.py` |
| [core/sidm.py](../core/sidm.py) | SIDM membership/history and matched-control analysis; isolated and homogeneous scattering experiments, forward observables and calibrated joint-response products | `commands.py`, `core/bao.py` |
| [core/sidm_campaign.py](../core/sidm_campaign.py) | SIDM campaign preparation, native measurement requests and resource intake | `commands.py`, `core/workflow.py` |
| [core/sidm_forward.py](../core/sidm_forward.py) | Matched native SIDM/CDM lensing and stellar-stream forward responses | `core/workflow.py`, `core/sidm_recovery.py` |
| [core/sidm_measurements.py](../core/sidm_measurements.py) | Measured SIDM shell, core, component and satellite diagnostics | `core/workflow.py`, `core/sidm.py` |
| [core/sidm_physics.py](../core/sidm_physics.py) | Elastic SIDM angular models, native tables and explicit engine parameters | `commands.py`, `core/workflow.py` |
| [core/sidm_products.py](../core/sidm_products.py) | SIDM history, matched contrasts, derived figures and campaign export routes | `core/cli.py`, `core/workflow.py` |
| [core/sidm_recovery.py](../core/sidm_recovery.py) | Recovered native target -> original SIDM membership/history producers | `core/workflow.py`, `core/campaigns.py` |
| [core/stars.py](../core/stars.py) | Native birth/source histories, species census, and conservation readers | `commands.py`, `core/analysis.py` |
| [core/stellar_observations.py](../core/stellar_observations.py) | Stellar observation models: spectral-age chronometers and distance ladders | `core/cosmography.py`, `core/observation_data.py` |
| [core/stellar_population.py](../core/stellar_population.py) | Native stellar/gas history intake and pinned FSPS population synthesis | `core/workflow.py`, `core/assembly.py` |
| [core/stellar_returns.py](../core/stellar_returns.py) | Delayed stellar-return input support, native state and campaign stages | `commands.py`, `core/workflow.py` |
| [core/zoom.py](../core/zoom.py) | Parent-particle tracing and MUSIC nested-grid staging; exact-ID coverage/contamination checks, density/velocity phase comparisons and hierarchy/window covariance controls | `commands.py`, `core/apertures.py` |

### Figures and visual exports

| File | Responsibility | Connected from |
|---|---|---|
| [core/allsky_map.py](../core/allsky_map.py) | Equal-solid-angle sky maps of saved simulation matter around an observer | `commands.py` |
| [core/blender.py](../core/blender.py) | Scene construction, animation compatibility, and measured inspector support | `commands.py`, `core/cli.py` |
| [core/blender_scene.py](../core/blender_scene.py) | Portable measured display packages and exact-identity epoch scene intake | `commands.py`, `core/blender.py` |
| [core/comparison_movie.py](../core/comparison_movie.py) | Encode matched, measured comparison maps into a provenance-bound movie | `commands.py` |
| [core/cosmic_flythrough.py](../core/cosmic_flythrough.py) | A perspective camera travelling through one saved cosmological snapshot | `commands.py`, `core/filament_flythrough.py` |
| [core/display_fresh_load.py](../core/display_fresh_load.py) | Executed inside a freshly loaded Blender process only when requested | `core/blender_scene.py`, `core/foundation.py` |
| [core/display_runtime.py](../core/display_runtime.py) | Standard-library Blender runtime embedded in .blend; measured epochs only | `core/blender.py`, `core/blender_scene.py` |
| [core/dome_movie.py](../core/dome_movie.py) | Fixed-scale planetarium movie from saved, source-bound dome rasters | `commands.py` |
| [core/figures.py](../core/figures.py) | Shared scientific figure export, plotted-array contracts and graph recipes | `commands.py`, `core/analysis.py` |
| [core/filament_flythrough.py](../core/filament_flythrough.py) | Cinematic path along a periodic Hessian density-ridge proxy to a halo | `commands.py` |
| [core/formation_atlas.py](../core/formation_atlas.py) | Small-multiple atlas of comparable-mass halo assembly histories | `core/analysis.py` |
| [core/halo_anatomy.py](../core/halo_anatomy.py) | Interactive, saved-snapshot cutaway and measured radial halo profiles | `commands.py` |
| [core/halo_collision_movie.py](../core/halo_collision_movie.py) | Close-up of two linked progenitor branches using measured saved positions | `commands.py` |
| [core/halo_fingerprint.py](../core/halo_fingerprint.py) | Exact-ID matched host comparison with common physical map and profile grids | `commands.py` |
| [core/halo_formation_movie.py](../core/halo_formation_movie.py) | Animate saved density maps and their linked, positioned halo ancestry | `commands.py` |
| [core/halo_phase_portrait.py](../core/halo_phase_portrait.py) | Radius–radial peculiar-velocity portrait of a saved halo particle sample | `core/analysis.py` |
| [core/lens_map.py](../core/lens_map.py) | Thin-plane convergence and shear from a saved projected mass map | `commands.py` |
| [core/merger_tree_map.py](../core/merger_tree_map.py) | Connect an existing verified merger tree to its saved projected matter maps | `commands.py` |
| [core/merger_tree_visual.py](../core/merger_tree_visual.py) | Build and display a descendant-linked halo ancestry from saved catalogues | `core/analysis.py`, `core/select_merger_tree.py` |
| [core/metallicity_maps.py](../core/metallicity_maps.py) | Add mass-weighted gas metallicity maps to a sealed saved-map bundle | `commands.py` |
| [core/multifield_movie.py](../core/multifield_movie.py) | Synchronized saved-field projections, with explicit missing panels | `commands.py` |
| [core/parameter_explorer.py](../core/parameter_explorer.py) | Offline explorer of explicitly supplied, source-verified simulation runs | `commands.py` |
| [core/particle_origin_movie.py](../core/particle_origin_movie.py) | Trace exact native host members through saved cosmological snapshots | `commands.py`, `core/void_particle_history.py` |
| [core/research_inspector.py](../core/research_inspector.py) | Measured object inspector embedded in the saved scene, standard library only | `core/blender.py`, `core/blender_scene.py` |
| [core/resolution_reveal.py](../core/resolution_reveal.py) | Compare two saved projected maps without assuming matched phases | `commands.py` |
| [core/scene_render.py](../core/scene_render.py) | Render a sealed saved scene with explicit display and allocation settings | `core/cli.py`, `core/workflow.py` |
| [core/select_merger_tree.py](../core/select_merger_tree.py) | Export the measured ancestry of an explicitly selected final halo | `commands.py`, `core/halo_collision_movie.py` |
| [core/stellar_stream_movie.py](../core/stellar_stream_movie.py) | Animate saved exact-ID stellar candidate histories without inferring origin | `commands.py` |
| [core/timeline_wall.py](../core/timeline_wall.py) | Create a fixed-scale cosmic timeline from a sealed saved-map bundle | `commands.py` |
| [core/velocity_flow.py](../core/velocity_flow.py) | Saved dark-matter surface density and mass-weighted peculiar-velocity flow | `commands.py` |
| [core/visual_pipeline.py](../core/visual_pipeline.py) | Saved-run visualization dispatch with independent, explicitly recorded failures | `commands.py` |
| [core/visualization.py](../core/visualization.py) | Saved-particle maps, chronological movies and portable visual exports | `commands.py`, `core/allsky_map.py` |
| [core/void_boundary_history.py](../core/void_boundary_history.py) | Measure periodic density-excursion boundaries and their Eulerian overlap | `commands.py` |
| [core/void_explorer.py](../core/void_explorer.py) | Explore measured underdensities in periodic saved dark-matter snapshots | `commands.py` |
| [core/void_particle_history.py](../core/void_particle_history.py) | Measure exact dark-matter particle overlap between saved void-centered apertures | `commands.py` |

### Flagship adapters

| File | Responsibility | Connected from |
|---|---|---|
| [core/flagship_abundance.py](../core/flagship_abundance.py) | Native stellar-mass and Vmax abundances with declared catalogue selections | `core/cli.py`, `core/workflow.py` |
| [core/flagship_analysis_profile.py](../core/flagship_analysis_profile.py) | Portable published-BAO branches over the original CLASS and analysis engines | `core/workflow.py`, `core/flagship_launch.py` |
| [core/flagship_bh.py](../core/flagship_bh.py) | Registered BH histories from the existing native BH analysis producer | `core/cli.py`, `core/workflow.py` |
| [core/flagship_bh_response.py](../core/flagship_bh_response.py) | Matched BH-branch density/shape figures from existing native structure products | `core/workflow.py`, `core/flagship_products.py` |
| [core/flagship_campaign_profile.py](../core/flagship_campaign_profile.py) | Freeze original production or matched SIDM compiler output for the launcher | `core/flagship_launch.py` |
| [core/flagship_campaigns.py](../core/flagship_campaigns.py) | Bind selected flagship profiles to the existing native workflow runners | `core/cli.py`, `core/workflow.py` |
| [core/flagship_history.py](../core/flagship_history.py) | Connect native campaign artifacts to the existing material history estimators | `core/cli.py`, `core/workflow.py` |
| [core/flagship_inference.py](../core/flagship_inference.py) | Six registered inference figures from original completed inference products | `core/cli.py`, `core/workflow.py` |
| [core/flagship_launch.py](../core/flagship_launch.py) | A frozen launch interface over the existing scientific workflow engine | `core/cli.py`, `core/flagship_analysis_profile.py` |
| [core/flagship_native_births.py](../core/flagship_native_births.py) | Saved SUBFIND ownership and recorded SSP births for native cooling joins | `core/flagship_native_branches.py` |
| [core/flagship_native_branches.py](../core/flagship_native_branches.py) | Original recorded native ancestry to assembly, cooling and web histories | `core/workflow.py` |
| [core/flagship_orbits.py](../core/flagship_orbits.py) | Native catalogue ancestry to existing orbital reconstruction and event plots | `core/cli.py`, `core/workflow.py` |
| [core/flagship_parent_preparation.py](../core/flagship_parent_preparation.py) | Prepare native parent followups from an actual completed CLASS stage | `core/workflow.py`, `core/flagship_profiles.py` |
| [core/flagship_portable.py](../core/flagship_portable.py) | Resolve portable source assets through existing builders and admission gates | `core/flagship_analysis_profile.py`, `core/flagship_launch.py` |
| [core/flagship_production_campaigns.py](../core/flagship_production_campaigns.py) | Production bridge recipes delegated to the existing native workflow engine | `core/cli.py`, `core/workflow.py` |
| [core/flagship_production_products.py](../core/flagship_production_products.py) | Compose original production histories with existing measurement/display followups | `core/flagship_campaign_profile.py` |
| [core/flagship_products.py](../core/flagship_products.py) | Connect native stream measurements to existing science and display products | `core/cli.py`, `core/workflow.py` |
| [core/flagship_profiles.py](../core/flagship_profiles.py) | Profile input validation; resource suitability never certifies science | `core/cli.py`, `core/workflow.py` |
| [core/flagship_retained.py](../core/flagship_retained.py) | Bind postprocessing and rendering to original completed workflow receipts | `core/workflow.py` |
| [core/flagship_run_report.py](../core/flagship_run_report.py) | Report original workflow evidence without promoting it to science acceptance | `core/flagship_launch.py` |
| [core/flagship_runtime_display.py](../core/flagship_runtime_display.py) | Completed native history inventories to original gallery and scene producers | `core/cli.py`, `core/workflow.py` |
| [core/flagship_runtime_history.py](../core/flagship_runtime_history.py) | Extend completed runtime zoom generations with original native history DAGs | `core/workflow.py`, `core/flagship_profiles.py` |
| [core/flagship_runtime_measurements.py](../core/flagship_runtime_measurements.py) | Connect completed native histories to the existing measurement/export stages | `core/workflow.py`, `core/flagship_production_products.py` |
| [core/flagship_zoom_preparation.py](../core/flagship_zoom_preparation.py) | Trace actual completed parent targets and prepare bounded native zoom stages | `core/workflow.py`, `core/flagship_profiles.py` |

### MUSIC numerical infrastructure

| File | Responsibility | Connected from |
|---|---|---|
| [core/music/__init__.py](../core/music/__init__.py) | MUSIC planning, numerical operators, response propagation and noise models | Python package loading |
| [core/music/adjoint.py](../core/music/adjoint.py) | Adjoints from MUSIC windows through potentials, sources and shared noise | `core/music/flow.py`, `core/music/windows.py` |
| [core/music/archive.py](../core/music/archive.py) | Retained native MUSIC fields, hierarchy loading and discrete comparisons | `core/flagship_products.py`, `core/music/flow.py` |
| [core/music/boundary.py](../core/music/boundary.py) | Build and use MUSIC's native flux-matched boundary map and transpose | `core/cli.py`, `core/flagship_portable.py` |
| [core/music/covariance.py](../core/music/covariance.py) | Verified MUSIC noise moments, window covariance and Gaussian checks | `core/flagship_products.py`, `core/music/flow.py` |
| [core/music/flow.py](../core/music/flow.py) | MUSIC flow preparation, shared-noise responses and pre-evolution checks | `commands.py`, `core/cli.py` |
| [core/music/phases.py](../core/music/phases.py) | Native MUSIC phase formats and explicit frozen/seeded model verification | `core/cli.py`, `core/workflow.py` |
| [core/music/poisson.py](../core/music/poisson.py) | Coupled MUSIC Poisson equations, independent references and failure intake | `core/cli.py`, `core/flagship_products.py` |
| [core/music/precision.py](../core/music/precision.py) | Exact binary128 storage and its explicit native arithmetic adapter | `core/workflow.py`, `core/music/archive.py` |
| [core/music/replay.py](../core/music/replay.py) | Independent replay from native MUSIC noise to sources and IC components | `core/flagship_products.py`, `core/music/adjoint.py` |
| [core/music/resources.py](../core/music/resources.py) | MUSIC hierarchy census and phase, response and solver allocation plans | `core/cli.py`, `core/flagship_profiles.py` |
| [core/music/seeded.py](../core/music/seeded.py) | MUSIC local seeded refinement, GSL replay and fixed-prefix moments | `core/music/covariance.py`, `core/music/flow.py` |
| [core/music/staging.py](../core/music/staging.py) | Stage native MUSIC cases and bind evolution to their actual generated mesh | `commands.py`, `core/workflow.py` |
| [core/music/windows.py](../core/music/windows.py) | Finite MUSIC particle-window measurements and their native input responses | `core/flagship_products.py`, `core/music/adjoint.py` |

### Research workflows

| File | Responsibility | Connected from |
|---|---|---|
| [core/research/__init__.py](../core/research/__init__.py) | Research workflows and measurements, grouped by responsibility | Python package loading |
| [core/research/capabilities.py](../core/research/capabilities.py) | Decide which workflows their declared physics and saved inputs can support | `commands.py`, `core/workflow.py` |
| [core/research/checkpoints.py](../core/research/checkpoints.py) | Publish restart markers from coherent native rank checkpoint generations | `commands.py`, `core/flagship_campaigns.py` |
| [core/research/controls.py](../core/research/controls.py) | Native control build profiles and physical input states, without execution | `commands.py`, `core/evolution.py` |
| [core/research/convergence.py](../core/research/convergence.py) | Plan controlled native variants, materialize their inputs and compare measurements | `commands.py`, `core/workflow.py` |
| [core/research/display.py](../core/research/display.py) | Assemble measured displays, graph galleries and native scene attachments | `commands.py`, `core/workflow.py` |
| [core/research/errors.py](../core/research/errors.py) | Measured error budgets, signed residuals and correlated native ensembles | `commands.py`, `core/workflow.py` |
| [core/research/findings.py](../core/research/findings.py) | Optional history of reported failures and the measurements used to resolve them | `commands.py`, `core/workflow.py` |
| [core/research/generator_controls.py](../core/research/generator_controls.py) | Explicit numerical controls for the native MUSIC transfer interface | `commands.py`, `core/flagship_campaigns.py` |
| [core/research/measurements.py](../core/research/measurements.py) | Measure native observables and collect paired controls into error-budget workflows | `commands.py`, `core/flagship_history.py` |
| [core/research/pilots.py](../core/research/pilots.py) | Prepare and inspect representative native pilot campaigns | `commands.py`, `core/workflow.py` |
| [core/research/release.py](../core/research/release.py) | Prepare and inspect an explicitly requested portable-release control campaign | `commands.py`, `core/research/restart_campaign.py` |
| [core/research/restart_campaign.py](../core/research/restart_campaign.py) | Optional native restart campaigns: freeze, pause/resume, compare | `core/workflow.py`, `core/flagship_history.py` |
| [core/research/restarts.py](../core/research/restarts.py) | Compare retained native states before and after restart | `commands.py`, `core/workflow.py` |
| [core/research/zooms.py](../core/research/zooms.py) | Prepare, recover and measure bounded native zoom campaigns | `commands.py`, `core/workflow.py` |

### Streams and inference

| File | Responsibility | Connected from |
|---|---|---|
| [core/stream/__init__.py](../core/stream/__init__.py) | Stream experiment adapters; import the selected module explicitly | Python package loading |
| [core/stream/encoders.py](../core/stream/encoders.py) | Observable-only graph/summary embeddings for stellar-stream inference | `core/stream/sbi.py` |
| [core/stream/encounters.py](../core/stream/encounters.py) | Matched perturber templates and resolved compound stream encounters | `core/native_stream_responses.py`, `core/sidm.py` |
| [core/stream/experiments.py](../core/stream/experiments.py) | Physical stream experiments, selection and campaign stage dispatch | `core/stream/sbi.py`, `core/stream/workflow.py` |
| [core/stream/inference.py](../core/stream/inference.py) | Recorded priors, selection-aware likelihoods and real posterior products | `core/bao.py`, `core/cosmography.py` |
| [core/stream/measurements.py](../core/stream/measurements.py) | Native stream measurements, stellar context, support spectra and figures | `core/flagship_history.py`, `core/research/display.py` |
| [core/stream/native_encounters.py](../core/stream/native_encounters.py) | Prepare, display and refine encounters from actual native histories | `core/workflow.py`, `core/campaigns.py` |
| [core/stream/perturbers.py](../core/stream/perturbers.py) | Exact native ancestry and bound memberships → encounter worldlines | `core/workflow.py`, `core/flagship_history.py` |
| [core/stream/sbi.py](../core/stream/sbi.py) | Domain-gated SBI generation, frozen models, calibration and stage dispatch | `core/campaigns.py`, `core/stream/experiments.py` |
| [core/stream/trajectories.py](../core/stream/trajectories.py) | Continuous, bounded encounter geometry in proper kpc/km/s/Myr | `core/analysis.py`, `core/assembly.py` |
| [core/stream/workflow.py](../core/stream/workflow.py) | Connected public workflow for blocks 16–30, with durable scientific products | `commands.py`, `core/apertures.py` |
| [core/stream_catalogue.py](../core/stream_catalogue.py) | Native stellar debris catalogue with measured ancestry and calibrated ambiguity | `core/flagship_products.py`, `core/stream/sbi.py` |
| [core/stream_common.py](../core/stream_common.py) | Shared stream contracts; all lengths proper kpc, velocities km/s, times Myr | `core/workflow.py`, `core/action_angle.py` |
| [core/stream_resimulation.py](../core/stream_resimulation.py) | Live self-gravitating stellar stripping and calibrated release adapters | `core/flagship_orbits.py`, `core/stream/experiments.py` |
| [core/stream_resolution.py](../core/stream_resolution.py) | Stellar-stream sampling, force support and actual output-cadence gates | `commands.py`, `core/workflow.py` |
| [core/stream_tracks.py](../core/stream_tracks.py) | Principal stream curves, transported frames and local observables | `core/flagship_products.py`, `core/native_stream_responses.py` |
| [core/streams.py](../core/streams.py) | Shared initial stream states and integrations; track/density/velocity measurements, native counterfactual IC/run campaigns, worldlines and matched response comparisons | `commands.py`, `core/analysis.py` |

### Paired atlases and HMF research

| File | Responsibility | Connected from |
|---|---|---|
| [core/flow_movie.py](../core/flow_movie.py) | Advect display tracers through a measured projected peculiar-velocity field; export films and source records | `scripts/build_paired_flow_movies.py` |
| [core/gadget_hmf_data.py](../core/gadget_hmf_data.py) | Export measured full-box halo counts and snapshot metadata as learning targets | `core/hmf_learning.py` |
| [core/halo_story.py](../core/halo_story.py) | Synchronize a positioned merger tree, measured density maps and labelled saved-epoch film | `commands.py:halo_story_main`, `scripts/build_paired_halo_stories.py` |
| [core/hmf_campaign.py](../core/hmf_campaign.py) | Run per-epoch linear predictions and cosmology screening; export candidates and prediction tables | `core/cli.py`, `core/gadget_hmf_data.py` |
| [core/hmf_learning.py](../core/hmf_learning.py) | Build simulation-count datasets, hold out cosmologies and fit a Poisson polynomial baseline | `core/cli.py:hmf_learning_main` |
| [core/paired_atlas.py](../core/paired_atlas.py) | Read native epochs into physical-unit caches; compare initial conditions and preserve exact sample IDs | Paired measurement scripts and the two atlas helpers below |
| [core/paired_atlas_figures.py](../core/paired_atlas_figures.py) | Plot paired epoch caches and export exact comparison tables | Explicit Python API `render_pair` / `render_available` |
| [core/paired_atlas_site.py](../core/paired_atlas_site.py) | Build portable atlas navigation, topic pages, tables and a local viewer | `scripts/refresh_paired_atlas.py` |

The paired-atlas helpers require completed run/cache inputs; they do not evolve a
new simulation. The [HMF research guide](hmf-research.md) distinguishes analytic
predictions, measured simulation targets and learned fits.

### Verification sources

`verification.py` contains the grouped domain/control suites and is exposed by the
public verification command. Standalone test modules are listed below so they are not
mistaken for unused simulation software.

| File | Purpose |
|---|---|
| [tests/test_allsky_map.py](../tests/test_allsky_map.py) | allsky map behavior and failure cases |
| [tests/test_batch_analysis.py](../tests/test_batch_analysis.py) | batch analysis behavior and failure cases |
| [tests/test_blender_denoising.py](../tests/test_blender_denoising.py) | blender denoising behavior and failure cases |
| [tests/test_comparison_movie.py](../tests/test_comparison_movie.py) | comparison movie behavior and failure cases |
| [tests/test_cosmic_flythrough.py](../tests/test_cosmic_flythrough.py) | cosmic flythrough behavior and failure cases |
| [tests/test_dome_movie.py](../tests/test_dome_movie.py) | dome movie behavior and failure cases |
| [tests/test_filament_flythrough.py](../tests/test_filament_flythrough.py) | filament flythrough behavior and failure cases |
| [tests/test_finalization.py](../tests/test_finalization.py) | finalization behavior and failure cases |
| [tests/test_formation_atlas.py](../tests/test_formation_atlas.py) | formation atlas behavior and failure cases |
| [tests/test_halo_anatomy.py](../tests/test_halo_anatomy.py) | halo anatomy behavior and failure cases |
| [tests/test_halo_collision_movie.py](../tests/test_halo_collision_movie.py) | halo collision movie behavior and failure cases |
| [tests/test_halo_fingerprint.py](../tests/test_halo_fingerprint.py) | halo fingerprint behavior and failure cases |
| [tests/test_halo_phase_portrait.py](../tests/test_halo_phase_portrait.py) | halo phase portrait behavior and failure cases |
| [tests/test_ic_phase_handoff.py](../tests/test_ic_phase_handoff.py) | ic phase handoff behavior and failure cases |
| [tests/test_lens_map.py](../tests/test_lens_map.py) | lens map behavior and failure cases |
| [tests/test_merger_tree_map.py](../tests/test_merger_tree_map.py) | merger tree map behavior and failure cases |
| [tests/test_merger_tree_visual.py](../tests/test_merger_tree_visual.py) | merger tree visual behavior and failure cases |
| [tests/test_metallicity_maps.py](../tests/test_metallicity_maps.py) | metallicity maps behavior and failure cases |
| [tests/test_multifield_movie.py](../tests/test_multifield_movie.py) | multifield movie behavior and failure cases |
| [tests/test_parameter_explorer.py](../tests/test_parameter_explorer.py) | parameter explorer behavior and failure cases |
| [tests/test_particle_origin_movie.py](../tests/test_particle_origin_movie.py) | particle origin movie behavior and failure cases |
| [tests/test_resolution_reveal.py](../tests/test_resolution_reveal.py) | resolution reveal behavior and failure cases |
| [tests/test_results_gallery.py](../tests/test_results_gallery.py) | results gallery behavior and failure cases |
| [tests/test_select_merger_tree.py](../tests/test_select_merger_tree.py) | select merger tree behavior and failure cases |
| [tests/test_snapshot_tables.py](../tests/test_snapshot_tables.py) | snapshot tables behavior and failure cases |
| [tests/test_stellar_stream_movie.py](../tests/test_stellar_stream_movie.py) | stellar stream movie behavior and failure cases |
| [tests/test_timeline_wall.py](../tests/test_timeline_wall.py) | timeline wall behavior and failure cases |
| [tests/test_velocity_flow.py](../tests/test_velocity_flow.py) | velocity flow behavior and failure cases |
| [tests/test_void_explorer.py](../tests/test_void_explorer.py) | void explorer behavior and failure cases |
| [tests/test_void_particle_history.py](../tests/test_void_particle_history.py) | void particle history behavior and failure cases |

The reference generator [scripts/build_reference.py](../scripts/build_reference.py)
rebuilds the separate command and Python appendices from current source declarations.
It imports the dispatcher to locate handlers but does not execute their workflows.

### Working with consolidated source

Use the `# @research2-module`/end markers and the layout declaration to locate a
historical logical module. Keep its globals and import namespace intact. Source pins and
figure/campaign provenance include implementation hashes, so even a numerical-equivalent
refactor belongs to a new source version. Large numerical moves require a later
validation run. Import and documentation checks establish the loader behavior;
numerical equivalence requires the relevant execution controls.

## Portability and source hygiene

For a future archive release, `core/release.py` selects portable files,
retains scientific tables and attribution,
excludes generated state, and records source identity. Local `.venv` and
installed/native builds belong to the current machine and are not part of the release.
This cleanup does not require creating an archive. Source/compiled code pins are
not interchangeable: comments and refactors can change
source identity even when no numerical formula changes.

The remaining reference-only folders contain attribution or pinned method records, not
unused executable checkouts. Upstream source, tests and ancillary files may also be
retained where selected package builds or their pinned source inventories require them;
they should not be deleted merely because the box simulation does not import them.
Conservation/epoch/identity records are actual reader/recovery inputs. Generated cache
listings and ad-hoc audit reports are not first-party source deliverables.

Fresh local execution is described in the [smoke report](smoke-test.md).
Historical results, platform gaps and scientific acceptance remain tracked in the
[maintained checklist](finalization.md).
