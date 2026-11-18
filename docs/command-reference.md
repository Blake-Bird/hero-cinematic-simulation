# Command reference

Generated from `commands.COMMANDS` and the handlers’ argparse declarations by `python scripts/build_reference.py`. Run commands from the source folder with the installed environment active.

```bash
python commands.py --help
python commands.py COMMAND --help
```

This is a lookup reference, not a sequence to run from top to bottom. The [operator guide](operator-guide.md) supplies workflow order and examples. Request JSON examples live in `configs/`; choose the file named by the relevant workflow guide. Paths, models, units, selected hosts, resource limits, and destinations must describe your experiment. Never reuse a published destination for a new experiment.

Options below are source declarations, including conditional branches; the handler’s live help resolves its actual parser, required arguments, choices, and defaults. Simple imported-function forwarders are followed to their parser owner without invoking the workflow. Dynamic expressions remain visible as source expressions; use subcommand help for their concrete values.

**192 registered commands.**

- [allsky-map](#allsky-map)
- [analysis](#analysis)
- [analysis-batch](#analysis-batch)
- [artwork](#artwork)
- [audit-black-hole-events](#audit-black-hole-events)
- [audit-data-contract](#audit-data-contract)
- [audit-graph-producers](#audit-graph-producers)
- [audit-hydro-events](#audit-hydro-events)
- [audit-hydro-history](#audit-hydro-history)
- [audit-hydro-ssp](#audit-hydro-ssp)
- [audit-ic](#audit-ic)
- [audit-native-handoff](#audit-native-handoff)
- [audit-tree-integrity](#audit-tree-integrity)
- [audit-zoom-phases](#audit-zoom-phases)
- [audit-zoom-regions](#audit-zoom-regions)
- [bao-model](#bao-model)
- [black-hole-environment](#black-hole-environment)
- [blender](#blender)
- [bootstrap](#bootstrap)
- [build-identity-graph](#build-identity-graph)
- [checkpoint-marker](#checkpoint-marker)
- [class](#class)
- [clean-release](#clean-release)
- [combine-graph-galleries](#combine-graph-galleries)
- [compare-finders](#compare-finders)
- [compare-runs](#compare-runs)
- [comparison-movie](#comparison-movie)
- [compile-pilot-family](#compile-pilot-family)
- [compile-release-campaign](#compile-release-campaign)
- [compile-stellar-model](#compile-stellar-model)
- [conditional-h0](#conditional-h0)
- [convergence-budgets](#convergence-budgets)
- [convergence-campaign](#convergence-campaign)
- [convergence-measurement](#convergence-measurement)
- [cosmic-flythrough](#cosmic-flythrough)
- [cosmography](#cosmography)
- [cosmography-report](#cosmography-report)
- [display-products](#display-products)
- [dm-wake-context](#dm-wake-context)
- [dm-wake-contrast](#dm-wake-contrast)
- [dm-wake-ensemble](#dm-wake-ensemble)
- [dm-wake-force](#dm-wake-force)
- [dm-wake-kernel-comparison](#dm-wake-kernel-comparison)
- [dm-wake-native-table](#dm-wake-native-table)
- [dm-wake-response](#dm-wake-response)
- [dm-wake-schedule](#dm-wake-schedule)
- [dm-wake-state-response](#dm-wake-state-response)
- [dm-wake-state-table](#dm-wake-state-table)
- [dome](#dome)
- [environment](#environment)
- [error-budget](#error-budget)
- [error-ensemble](#error-ensemble)
- [error-residuals](#error-residuals)
- [event-cadence](#event-cadence)
- [export-abundance-figures](#export-abundance-figures)
- [export-analysis-figures](#export-analysis-figures)
- [export-assembly-figures](#export-assembly-figures)
- [export-bh-figures](#export-bh-figures)
- [export-convergence-figures](#export-convergence-figures)
- [export-hubble-figures](#export-hubble-figures)
- [export-inference-figures](#export-inference-figures)
- [export-merger](#export-merger)
- [export-power-figures](#export-power-figures)
- [export-sidm-figures](#export-sidm-figures)
- [export-splashback-figures](#export-splashback-figures)
- [export-stellar-observer](#export-stellar-observer)
- [export-stream-figures](#export-stream-figures)
- [export-structure-figures](#export-structure-figures)
- [filament-flythrough](#filament-flythrough)
- [find-galaxies](#find-galaxies)
- [finder-campaign](#finder-campaign)
- [findings](#findings)
- [flagship](#flagship)
- [flagship-products](#flagship-products)
- [gadget](#gadget)
- [halo-anatomy](#halo-anatomy)
- [halo-collision-movie](#halo-collision-movie)
- [halo-fingerprint](#halo-fingerprint)
- [halo-formation-movie](#halo-formation-movie)
- [halo-story](#halo-story)
- [halo-structure](#halo-structure)
- [halos](#halos)
- [hmf](#hmf)
- [hmf-learning](#hmf-learning)
- [hmf-scan](#hmf-scan)
- [hydro-comparison](#hydro-comparison)
- [ingest-bao](#ingest-bao)
- [ingest-cosmography](#ingest-cosmography)
- [inspect-poisson-failure](#inspect-poisson-failure)
- [isolated-control-history](#isolated-control-history)
- [lens-map](#lens-map)
- [lensing](#lensing)
- [matched-hydro-pilot](#matched-hydro-pilot)
- [measure-halo-shapes](#measure-halo-shapes)
- [measure-stream-heating](#measure-stream-heating)
- [measure-zoom-campaign](#measure-zoom-campaign)
- [measured-scene](#measured-scene)
- [merger-tree-map](#merger-tree-map)
- [metallicity-maps](#metallicity-maps)
- [multifield-movie](#multifield-movie)
- [music-boundary](#music-boundary)
- [music-capacity](#music-capacity)
- [music-flow-prepare](#music-flow-prepare)
- [music-frozen-model](#music-frozen-model)
- [music-phase-plan](#music-phase-plan)
- [music-seeded-model](#music-seeded-model)
- [native-counterfactual-campaign](#native-counterfactual-campaign)
- [native-counterfactual-ic](#native-counterfactual-ic)
- [native-error-envelopes](#native-error-envelopes)
- [native-late-threeway](#native-late-threeway)
- [native-music-mixed2lpt](#native-music-mixed2lpt)
- [native-music-transfer](#native-music-transfer)
- [native-prepare](#native-prepare)
- [native-sidm-zoom-support](#native-sidm-zoom-support)
- [native-source-events](#native-source-events)
- [native-source-overlay](#native-source-overlay)
- [native-stream-comparison](#native-stream-comparison)
- [native-stream-worldlines](#native-stream-worldlines)
- [observe-stellar-catalogue](#observe-stellar-catalogue)
- [package](#package)
- [parameter-explorer](#parameter-explorer)
- [particle-origin-movie](#particle-origin-movie)
- [physical-target](#physical-target)
- [pilots](#pilots)
- [planetarium-movie](#planetarium-movie)
- [prepare-analysis-figures](#prepare-analysis-figures)
- [prepare-flagship-campaign](#prepare-flagship-campaign)
- [prepare-flagship-profile](#prepare-flagship-profile)
- [prepare-generated-flow](#prepare-generated-flow)
- [prepare-source-controls](#prepare-source-controls)
- [production-resources](#production-resources)
- [propose-tree-repair](#propose-tree-repair)
- [query-identity-graph](#query-identity-graph)
- [release-blender-control](#release-blender-control)
- [render-scene](#render-scene)
- [repair-receipt](#repair-receipt)
- [research-acceptance](#research-acceptance)
- [resolution-reveal](#resolution-reveal)
- [resource-compare](#resource-compare)
- [restart-comparison](#restart-comparison)
- [results](#results)
- [results-gallery](#results-gallery)
- [run](#run)
- [run-spec](#run-spec)
- [scientific-figures](#scientific-figures)
- [select-merger-tree](#select-merger-tree)
- [sidm-angular-control](#sidm-angular-control)
- [sidm-canonical-population](#sidm-canonical-population)
- [sidm-control-campaign](#sidm-control-campaign)
- [sidm-control-clock-mapping](#sidm-control-clock-mapping)
- [sidm-control-comparison](#sidm-control-comparison)
- [sidm-control-ensemble](#sidm-control-ensemble)
- [sidm-control-expectation](#sidm-control-expectation)
- [sidm-control-history](#sidm-control-history)
- [sidm-control-native-clock](#sidm-control-native-clock)
- [sidm-control-plan](#sidm-control-plan)
- [sidm-control-population](#sidm-control-population)
- [sidm-control-products](#sidm-control-products)
- [sidm-control-refinement](#sidm-control-refinement)
- [sidm-model](#sidm-model)
- [sidm-reference-build](#sidm-reference-build)
- [sidm-refinement-measurement](#sidm-refinement-measurement)
- [sidm-response-campaign](#sidm-response-campaign)
- [sidm-science](#sidm-science)
- [stage-zoom-inputs](#stage-zoom-inputs)
- [start-epoch-campaign](#start-epoch-campaign)
- [stellar-stream-movie](#stellar-stream-movie)
- [stream-products](#stream-products)
- [streams](#streams)
- [timeline-wall](#timeline-wall)
- [trace-zoom](#trace-zoom)
- [twofluid-eulerian-seed](#twofluid-eulerian-seed)
- [twofluid-native-comparison](#twofluid-native-comparison)
- [twofluid-native-family](#twofluid-native-family)
- [twofluid-native-ic](#twofluid-native-ic)
- [twofluid-nonlinear](#twofluid-nonlinear)
- [twofluid-particles](#twofluid-particles)
- [twofluid-projection-refinement](#twofluid-projection-refinement)
- [twofluid-refinement](#twofluid-refinement)
- [twofluid-refinement-campaign](#twofluid-refinement-campaign)
- [twofluid-relative](#twofluid-relative)
- [twofluid-window](#twofluid-window)
- [velociraptor](#velociraptor)
- [velocity-flow](#velocity-flow)
- [verify](#verify)
- [void-boundary-history](#void-boundary-history)
- [void-explorer](#void-explorer)
- [void-particle-history](#void-particle-history)
- [workflow](#workflow)
- [workflow-products](#workflow-products)
- [zoom](#zoom)
- [zoom-campaign](#zoom-campaign)

## allsky-map

`python commands.py allsky-map --help`

Implementation: [commands.py](../commands.py), `allsky_map_main`.

Select DM or gas from one saved epoch and render an observer-centred sky with
the requested equal-solid-angle grid and bounded sample. The observer is in
comoving Mpc/h, defaulting to box centre. Print the published map path; its
scalar is represented matter mass per steradian, not observed stellar flux
or a time-evolving light cone.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--snapshot` | See live help and implementation. | type=int; required=True |
| main | `--field` | See live help and implementation. | choices=['dm', 'gas']; default='dm' |
| main | `--observer` | Observer in comoving Mpc/h; default is box center | type=float; nargs=3 |
| main | `--nlon` | See live help and implementation. | type=int; default=360 |
| main | `--nlat` | See live help and implementation. | type=int; default=180 |
| main | `--max-particles` | See live help and implementation. | type=int; default=1000000 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## analysis

`python commands.py analysis --help`

Implementation: [commands.py](../commands.py), `run_analysis_main`.

Parse command arguments and dispatch this module's documented workflow.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | nargs='?'; type=Path |
| main | `--graphs` | See live help and implementation. |  |
| main | `--full-analysis` | Legacy core analysis suite; maps/BH/Blender remain separate workflow products | action='store_true' |
| main | `--overwrite` | With --full-analysis only: archive previous results before replacing them | action='store_true' |
| main | `--dry-run` | Inspect saved-run applicability without publishing or executing products | action='store_true' |
| main | `--state-root` | External runtime state directory | type=Path |
| main | `--find-halos` | run configured halo finders before analysis | action='store_true' |
| main | `--allow-incomplete` | retain an INCOMPLETE record and allow exit 0 when selected products are unavailable | action='store_true' |
| main | `--stellar-search` | JSON object: snapshot, host (FOF row), controls, memory_gib, threads | type=Path |
| main | `--stellar-history` | JSON object: catalogue, snapshots, candidate_ids, memory_gib; optional output_directory | type=Path |
| main | `--zoom-regions` | JSON object: regions, fine_dm_mass_msun, maximum_coarse_mass_fraction; optional output_directory | type=Path |
| main | `--research-spec` | Versioned physical observable and acceptance specification JSON | type=Path |
| main | `--evidence` | Measured acceptance evidence keyed product/test | type=Path |
| main | `--inference-diagnostics` | JSON object: samples_npz, metadata_json, output_directory | type=Path |
| main | `--radial-shapes` | Native saved host/snapshot and complete radial density-shape request JSON | type=Path |
| main | `--resume` | Retained failed/interrupted radial_shapes-only workflow record | type=Path |

## analysis-batch

`python commands.py analysis-batch --help`

Implementation: [commands.py](../commands.py), `analysis_batch_main`.

Analyze selected saved runs sequentially and compare each to a reference.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `runs` | Saved run paths; omit to select from the run list | nargs='*'; type=Path |
| main | `--graphs` | Comma-separated graph selection, available, or all |  |
| main | `--output` | Fresh batch-report directory | type=Path; required=True |
| main | `--reference` | One-based reference run number (default: first) | type=int; default=1 |
| main | `--state-root` | See live help and implementation. | type=Path |
| main | `--find-halos` | See live help and implementation. | action='store_true' |
| main | `--allow-incomplete` | See live help and implementation. | action='store_true' |
| main | `--no-comparisons` | See live help and implementation. | action='store_true' |

## artwork

`python commands.py artwork --help`

Implementation: [core/cli.py](../core/cli.py), `artwork_main`.

Process measured outputs from an existing saved scene campaign.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `operation` | See live help and implementation. | choices=['parent-export', 'diagnostics', 'encode'] |
| main | `--run` | See live help and implementation. | required=True; type=Path |

## audit-black-hole-events

`python commands.py audit-black-hole-events --help`

Implementation: [commands.py](../commands.py), `audit_hydro_history_main`.

Publish or independently verify a multi-epoch native gas/star/BH accounting
audit. For publication, bind the request bytes before and after the audit;
print status, epoch/interval/issue counts and physics acceptance. Exit nonzero
unless the accounting status is PASS. Audit success is scoped to the measured
census/journals and tolerances, not every physical model or resolution.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--verify` | See live help and implementation. | action='store_true' |

## audit-data-contract

`python commands.py audit-data-contract --help`

Implementation: [commands.py](../commands.py), `audit_data_contract_main`.

Audit a saved native snapshot or a contracted in-run product and write fresh
JSON evidence. Product modes select HDF5/table/numeric/IC/image/scene/epoch
verifiers, with optional explicit HDF5 semantics or snapshot chunk checks.
Separate recovery modes replay a named native-set generation, IC transaction,
catalogue publication or supported product generation under the relevant lock;
these modes can restore publication files. Enforce mode compatibility and
in-run paths, retain failures before raising, and keep integrity status separate
from scientific acceptance. Recovery does not repeat the original IC transform
or rerun a finder/evolution.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `--snapshot` | See live help and implementation. | type=int |
| main | `--recover-native-set` | Explicit native HDF5 base path inside the run | type=Path |
| main | `--native-set-generation` | Committed native generation to restore; required with --recover-native-set |  |
| main | `--recover-ic-transaction` | Replay a durable pending IC commit; do not repeat its transform | type=Path |
| main | `--recover-catalogue-snapshot` | Replay a pending native catalogue/order publication without rerunning finding | type=int |
| main | `--product` | Audit a saved contracted product inside this run | type=Path |
| main | `--product-kind` | See live help and implementation. | choices=('hdf5', 'numeric-table', 'numeric-audit', 'native-ic-audit', 'blender-scene', 'scientific-image', 'epoch-selection') |
| main | `--recover-generation` | Explicit replay of a sealed product generation; requires --product and --product-kind | type=Path |
| main | `--require-chunk-checksums` | Reject nonempty native snapshot fields without Fletcher32 protection | action='store_true' |
| main | `--require-scientific-semantics` | Require explicit sealed units, frame and epoch for every numeric HDF5 dataset | action='store_true' |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## audit-graph-producers

`python commands.py audit-graph-producers --help`

Implementation: [core/cli.py](../core/cli.py), `audit_graph_producers_main`.

Audit exact102 graph producer routes without running scientific producers.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--registry` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## audit-hydro-events

`python commands.py audit-hydro-events --help`

Implementation: [commands.py](../commands.py), `audit_hydro_events_main`.

Audit committed native birth events between the requested saved epochs,
checking that the request hash stays unchanged across publication. With
`--verify`, read/replay the existing output audit instead. Print status,
event count and physical reconciliation. This examines original journal events;
it neither launches star formation nor converts an empty event list into births.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--verify` | See live help and implementation. | action='store_true' |

## audit-hydro-history

`python commands.py audit-hydro-history --help`

Implementation: [commands.py](../commands.py), `audit_hydro_history_main`.

Publish or independently verify a multi-epoch native gas/star/BH accounting
audit. For publication, bind the request bytes before and after the audit;
print status, epoch/interval/issue counts and physics acceptance. Exit nonzero
unless the accounting status is PASS. Audit success is scoped to the measured
census/journals and tolerances, not every physical model or resolution.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--verify` | See live help and implementation. | action='store_true' |

## audit-hydro-ssp

`python commands.py audit-hydro-ssp --help`

Implementation: [commands.py](../commands.py), `audit_hydro_ssp_main`.

Publish an audit of native SSP birth masses, exact parent identities and
epochs from a request, or verify the existing audit at `--output`. Print status,
particle/support counts, legacy unavailable count and science acceptance.
It reads committed birth state; legacy missing fields remain unavailable
rather than becoming fabricated stellar ages or formed masses.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--verify` | See live help and implementation. | action='store_true' |

## audit-ic

`python commands.py audit-ic --help`

Implementation: [commands.py](../commands.py), `audit_ic_main`.

Read a JSON object of validator keyword arguments and audit a linear bundle,
delivered IC realization or generator comparison. Decode requested complex
density/velocity predictions for realization checks and print finite JSON.
Return 0 for PASS, 2 for INCOMPLETE and 1 otherwise. This validates supplied
sources and measurements; it does not create the input realizations.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `kind` | See live help and implementation. | choices=('bundle', 'realization', 'compare') |
| main | `request` | JSON keyword arguments for the selected validator | type=Path |

## audit-native-handoff

`python commands.py audit-native-handoff --help`

Implementation: [commands.py](../commands.py), `audit_native_handoff_main`.

Normally load simulation YAML and revalidate an IC against the named native
generation report, publishing the handoff audit at its destination with explicit
audit memory. Alternatively verify a retained numerical-control receipt; that
mode rejects IC arguments and exits nonzero unless PASS. It checks actual
source/build/measurement bindings without silently substituting another IC or
executing a new native control.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `configuration` | Simulation YAML with native_generation_report | type=Path; nargs='?' |
| main | `initial_ic` | See live help and implementation. | type=Path; nargs='?' |
| main | `destination` | See live help and implementation. | type=Path; nargs='?' |
| main | `--audit-memory-gib` | See live help and implementation. | type=float; default=1 |
| main | `--numerical-control-receipt` | Verify all retained native control builds, inputs and measurements | type=Path |

## audit-tree-integrity

`python commands.py audit-tree-integrity --help`

Implementation: [commands.py](../commands.py), `audit_tree_integrity_main`.

Load explicit native nodes/links or import a requested Consistent Trees series,
then require frozen diagnostic thresholds, epoch and source bindings. Retain
original native links, measure disappearance/swap/jump/fragmentation flags and
publish a numeric audit with exact node IDs at the destination. Print status
and flag count. Alternative repairs belong in a separate proposal product.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path |
| main | `--from-consistent-trees` | See live help and implementation. | action='store_true' |

## audit-zoom-phases

`python commands.py audit-zoom-phases --help`

Implementation: [commands.py](../commands.py), `audit_zoom_phases_main`.

Read parent/zoom IC and coarse-noise paths, destination and memory allocation,
then run the delivered density-phase comparison and print its retained report.
Return 0 only for PASS, otherwise 1. This audits the supplied 1LPT realization
pair and preserves FAIL evidence rather than asserting phase agreement from
matching seeds.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parent_ic` | See live help and implementation. | type=Path |
| main | `zoom_ic` | See live help and implementation. | type=Path |
| main | `parent_noise` | See live help and implementation. | type=Path |
| main | `zoom_noise` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1 |

## audit-zoom-regions

`python commands.py audit-zoom-regions --help`

Implementation: [commands.py](../commands.py), `audit_zoom_regions_main`.

Audit requested saved/tracked science apertures and sampling, or explicitly
compare observable products against a sealed saved-aperture resolution budget.
Comparison mode verifies the budget seal/provenance, selects exactly one audited
epoch and publishes under the run lock. Print the result and exit nonzero unless
PASS. This evaluates original selections and evidence; it does not refine ICs
or imply convergence outside the measured budget.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `request` | See live help and implementation. | type=Path |
| main | `--compare-resolution` | Compare contracted native observable products against a sealed saved-aperture budget | action='store_true' |

## bao-model

`python commands.py bao-model --help`

Implementation: [core/cli.py](../core/cli.py), `bao_model_main`.

Evaluate one frozen model against the original compressed BAO covariance.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--prepare-inference` | Validate the model, explicit sampler settings and installed backend without sampling. | action='store_true' |

## black-hole-environment

`python commands.py black-hole-environment --help`

Implementation: [commands.py](../commands.py), `black_hole_environment_main`.

Read a stable request and measure selected BH host binding, wandering and
DM radial histories from sealed native data. Publish the environment history,
recheck request bytes and print schema, epoch count and physics acceptance.
The command analyzes saved history rather than generating accretion, host
motion or a feedback intervention.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## blender

`python commands.py blender --help`

Implementation: [commands.py](../commands.py), `run_blender_main`.

Collect export settings and create a portable Blender scene manifest.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | nargs='?'; type=Path |
| main | `--max-points` | See live help and implementation. | type=int |
| main | `--quality` | See live help and implementation. | choices=['preview', 'high'] |
| main | `--frames-per-snapshot` | See live help and implementation. | type=int |
| main | `--build` | build and save the combined .blend scene | action='store_true' |
| main | `--preview` | build the scene and render a scientific preview PNG | action='store_true' |
| main | `--radial-shapes` | Exact native radial density-shape request JSON for verified FOF/epoch joins | type=Path |
| main | `--measured-display` | Sealed measured shapes/tides/streams/events/SIDM inspector manifest | type=Path |
| main | `--scene-attachments` | Source-bound native stream timelines and exact-context scientific gallery request | type=Path |
| main | `--output` | Fresh scene bundle directory outside the retained native run | type=Path |

## bootstrap

`python commands.py bootstrap --help`

Implementation: [commands.py](../commands.py), `bootstrap_main`.

Build requested engines or inspect the host prerequisites.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--check` | See live help and implementation. | action='store_true' |
| main | `--build-all` | build both Boltzmann solvers, IC generator, halo/tree tools and stellar STF | action='store_true' |
| main | `--build-class` | See live help and implementation. | action='store_true' |
| main | `--build-axiclass` | See live help and implementation. | action='store_true' |
| main | `--build-monofonic` | See live help and implementation. | action='store_true' |
| main | `--build-music2` | See live help and implementation. | action='store_true' |
| main | `--build-rockstar` | See live help and implementation. | action='store_true' |
| main | `--build-consistent-trees` | See live help and implementation. | action='store_true' |
| main | `--build-velociraptor` | See live help and implementation. | action='store_true' |
| main | `--build-disperse` | See live help and implementation. | action='store_true' |
| main | `--jobs` | See live help and implementation. | type=int; default=min(4, cpu_thread_budget()) |

## build-identity-graph

`python commands.py build-identity-graph --help`

Implementation: [commands.py](../commands.py), `build_identity_graph_main`.

Bind a request and source hashes, then publish an indexed exact-identity graph.
`--from-native` delegates native subhalo ownership intake; explicit graph mode
adds declared epochs, typed nodes, ancestry/parent-zoom/other edges and optional
SSP or hydro audits. Print node/edge counts and status. `--reuse-completed`
requests verified reuse; graph publication alone does not establish physical
ancestry beyond its supplied evidence.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path |
| main | `--reuse-completed` | See live help and implementation. | action='store_true' |
| main | `--from-native` | See live help and implementation. | action='store_true' |

## checkpoint-marker

`python commands.py checkpoint-marker --help`

Implementation: [core/research/checkpoints.py](../core/research/checkpoints.py), `checkpoint_marker_main`.

Parse required --run, --marker, --workspace, --mpi-ranks and --executable-sha256, plus optional --minimum-generation defaulting to one. Publish the verified immutable checkpoint marker and print its result as JSON; normal completion returns None. This command observes already written native checkpoints and does not start or resume GADGET.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--run` | See live help and implementation. | type=Path; required=True |
| main | `--marker` | See live help and implementation. | type=Path; required=True |
| main | `--workspace` | See live help and implementation. | type=Path; required=True |
| main | `--mpi-ranks` | See live help and implementation. | type=int; required=True |
| main | `--executable-sha256` | See live help and implementation. | required=True |
| main | `--minimum-generation` | See live help and implementation. | type=int; default=1 |

## class

`python commands.py class --help`

Implementation: [core/linear.py](../core/linear.py), `run_class_main`.

Resolve the compiled solver and generate self-contained run products.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--config` | JSON object with the interactive prompt keys | type=Path |
| main | `--output` | Explicit parent output directory for the isolated named cosmology run | type=Path |
| main | `--binary` | Explicit compiled CLASS/AxiCLASS executable; native provenance remains mandatory | type=Path |

## clean-release

`python commands.py clean-release --help`

Implementation: [core/research/release.py](../core/research/release.py), `clean_release_main`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--archive` | See live help and implementation. | type=Path; required=True |
| main | `--campaign` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## combine-graph-galleries

`python commands.py combine-graph-galleries --help`

Implementation: [core/cli.py](../core/cli.py), `combine_graph_galleries_main`.

Combine original measured graph galleries with all102 producer/status entries.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## compare-finders

`python commands.py compare-finders --help`

Implementation: [commands.py](../commands.py), `compare_finders_main`.

Read exactly two requested saved catalogues and matching thresholds under the
run lock. Intake native GADGET, Rockstar or a sealed VELOCIraptor catalogue,
checking the selected epoch, then publish their measured disagreements at the
destination and print finite JSON. This compares existing catalogue results;
it does not launch the finders or erase their different mass/membership definitions.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `request` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path |

## compare-runs

`python commands.py compare-runs --help`

Implementation: [commands.py](../commands.py), `compare_runs_main`.

Write the paired analysis products from existing analysis.h5 files.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_a` | See live help and implementation. | type=Path |
| main | `run_b` | See live help and implementation. | type=Path |
| main | `--name` | See live help and implementation. | default='comparison' |

## comparison-movie

`python commands.py comparison-movie --help`

Implementation: [commands.py](../commands.py), `comparison_movie_main`.

Encode an already published comparison-map series selected by its exact
summary key. Supply optional source-run paths only when provenance cannot
resolve them, plus explicit playback and memory settings; print the output.
This consumes measured comparison products, not an implicit phase match or
new pair of native simulations.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `bundle` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--map-key` | Exact map key from comparison_maps_summary.json | required=True |
| main | `--fps` | See live help and implementation. | type=int; default=24 |
| main | `--seconds-per-snapshot` | See live help and implementation. | type=int; default=1 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |
| main | `--runs` | Source runs when they cannot be inferred from provenance | type=Path; nargs=2 |

## compile-pilot-family

`python commands.py compile-pilot-family --help`

Implementation: [core/research/pilots.py](../core/research/pilots.py), `pilot_recipes_main`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--request` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## compile-release-campaign

`python commands.py compile-release-campaign --help`

Implementation: [core/research/release.py](../core/research/release.py), `release_campaign_recipes_main`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--request` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## compile-stellar-model

`python commands.py compile-stellar-model --help`

Implementation: [commands.py](../commands.py), `compile_stellar_model_main`.

Compile explicit pinned yield/lifetime inputs into the delayed-return table,
or verify an existing compiled model with `--verify`. `--preflight-only` instead
writes source-row/IMF/age-support diagnostics and returns nonzero for failure;
verification and preflight are mutually exclusive. Bind request bytes across
work and print implementation/validation/table metadata. This compiles a data
model, not a native executable or stellar evolution run.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--verify` | See live help and implementation. | action='store_true' |
| main | `--preflight-only` | Write complete source-row and IMF/age support audit; do not compile or change yields | action='store_true' |

## conditional-h0

`python commands.py conditional-h0 --help`

Implementation: [core/cli.py](../core/cli.py), `conditional_h0_main`.

Fit conditional H0 to measured BAO with fixed expansion shape and drag scale.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--bao-output` | See live help and implementation. | required=True; type=Path |
| main | `--output` | See live help and implementation. | required=True; type=Path |
| main | `--settings` | Explicit JSON: h0_bounds_km_s_mpc, sampler, histogram_bins, memory_gib | required=True; type=Path |

## convergence-budgets

`python commands.py convergence-budgets --help`

Implementation: [core/research/measurements.py](../core/research/measurements.py), `convergence_budget_collection_main`.

Read --request through a bounded 16 MiB JSON intake and publish the convergence-budget collection relative to the request file’s parent. Pass --execute and --resume to the publisher and print its JSON result. This CLI connects the collection workflow; its returned execution and science statuses determine what actually completed.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--request` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--execute` | See live help and implementation. | action='store_true' |
| main | `--resume` | See live help and implementation. | action='store_true' |

## convergence-campaign

`python commands.py convergence-campaign --help`

Implementation: [core/research/convergence.py](../core/research/convergence.py), `convergence_main`.

Read an explicit --campaign JSON and invoke execute_and_measure with required --output, the campaign’s parent as request base and optional --resume. Print the resulting controlled-variant measurement report. This CLI can launch actual evolution/measurement through that campaign; it returns no explicit status value and propagates uncaught validation/execution failures.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--campaign` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--resume` | See live help and implementation. | action='store_true' |

## convergence-measurement

`python commands.py convergence-measurement --help`

Implementation: [core/research/measurements.py](../core/research/measurements.py), `convergence_measurement_main`.

Read --parameters JSON, produce the native convergence measurement and write required --output plus its SHA-256 seal. Reject an existing output to retain prior evidence. Normal completion returns None; failures propagate. This command does not launch the native convergence family.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--parameters` | See live help and implementation. | required=True; type=Path |
| main | `--output` | See live help and implementation. | required=True; type=Path |

## cosmic-flythrough

`python commands.py cosmic-flythrough --help`

Implementation: [commands.py](../commands.py), `cosmic_flythrough_main`.

Render a camera path through one saved snapshot toward a selected catalogue
host, using an explicit start, optional waypoints and frame/image/FOV/sampling
budgets. Publish and print the output. The command may encode a film; it changes
the view through existing data, not the simulated epoch or particle dynamics.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--snapshot` | See live help and implementation. | type=int; required=True |
| main | `--host-row` | See live help and implementation. | type=int; required=True |
| main | `--start` | See live help and implementation. | type=float; nargs=3; required=True |
| main | `--waypoint` | See live help and implementation. | type=float; nargs=3; action='append' |
| main | `--frames` | See live help and implementation. | type=int; default=240 |
| main | `--width` | See live help and implementation. | type=int; default=1280 |
| main | `--height` | See live help and implementation. | type=int; default=720 |
| main | `--fps` | See live help and implementation. | type=int; default=24 |
| main | `--max-particles` | See live help and implementation. | type=int; default=300000 |
| main | `--fov-degrees` | See live help and implementation. | type=float; default=60.0 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## cosmography

`python commands.py cosmography --help`

Implementation: [commands.py](../commands.py), `cosmography_main`.

Forward to the selected cosmography CLI, which reads a request JSON, resolves
its relative inputs against that file and executes the declared expansion/
observer campaign into a required output. Return the delegate's result; it
prints the campaign record. Physical likelihoods, observations and resources
must come from that request rather than an automatic box-run inference.

Parser owner: [core/cosmography.py](../core/cosmography.py#L486), `cosmography_main`. The wrapper forwards to this function.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## cosmography-report

`python commands.py cosmography-report --help`

Implementation: [commands.py](../commands.py), `cosmography_report_main`.

Forward to the post-fit comparison exporter. Its CLI requires a JSON product
with role `post_fit_evaluation` and an output directory, then publishes comparison
figures and prints their locations. It renders frozen fit/evaluation results;
it does not execute a new fit or sample a posterior.

Parser owner: [core/cosmography.py](../core/cosmography.py#L496), `cosmography_report_main`. The wrapper forwards to this function.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `comparison` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## display-products

`python commands.py display-products --help`

Implementation: [core/research/display.py](../core/research/display.py), `display_products_main`.

Read required --request JSON and --output, resolve referenced inputs relative to the request file’s parent and invoke assemble_measured_display. Print the assembly receipt as formatted JSON; normal completion returns None. This command produces portable measured scene inputs and does not launch Blender.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--request` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## dm-wake-context

`python commands.py dm-wake-context --help`

Implementation: [core/black_holes.py](../core/black_holes.py), `dm_wake_context_main`.

Read the context parameters JSON, set a default request_base from its parent and publish source-bound native context assets to --output. Return zero on publication and propagate validation/I/O failures. Output launch metadata is preparation evidence, not a completed source-operator integration.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## dm-wake-contrast

`python commands.py dm-wake-contrast --help`

Implementation: [core/black_holes.py](../core/black_holes.py), `dm_wake_contrast_main`.

Load resolution-contrast parameters, default request_base to their JSON parent and publish to --output. Return zero after the three-or-more-level contrast producer completes. Numerical support/rejection remains in the saved product and exceptions propagate.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## dm-wake-ensemble

`python commands.py dm-wake-ensemble --help`

Implementation: [core/black_holes.py](../core/black_holes.py), `dm_wake_ensemble_main`.

Read ensemble parameters JSON, supply the parent as default request_base and publish means/uncertainty from the selected measured wake contrasts. Return zero on publication and propagate errors. This analyzes existing realizations rather than launching independent runs.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## dm-wake-force

`python commands.py dm-wake-force --help`

Implementation: [core/black_holes.py](../core/black_holes.py), `dm_wake_force_main`.

Read force-measurement parameters JSON and publish original native DM wake forces/bath moments under the declared probe operator to --output. Return zero after successful publication and propagate errors. Unlike other wake CLI adapters, this function does not insert a default request_base; required path/base controls must already be supplied by its parameters.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## dm-wake-kernel-comparison

`python commands.py dm-wake-kernel-comparison --help`

Implementation: [core/black_holes.py](../core/black_holes.py), `dm_wake_kernel_comparison_main`.

Read parameter JSON, default request_base to that file’s parent when absent and publish the native kernel/reference comparison to --output. Return zero after publication; rejected comparisons remain encoded in their product status and exceptions propagate. The CLI does not rerun native evolution to manufacture a diagnostic.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## dm-wake-native-table

`python commands.py dm-wake-native-table --help`

Implementation: [core/black_holes.py](../core/black_holes.py), `dm_wake_native_table_main`.

Read a speed-table parameter JSON, default request_base to its parent and publish a bounded native-readable response table to --output. Return zero on successful publication, with errors propagated; the table retains its measured speed/signature restrictions.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## dm-wake-response

`python commands.py dm-wake-response --help`

Implementation: [core/black_holes.py](../core/black_holes.py), `dm_wake_response_main`.

Load speed-response calibration parameters, default request_base to the file parent and export the local response to --output. Return zero after publication; inspect output interval support for calibration outcome rather than using process success as scientific acceptance.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## dm-wake-schedule

`python commands.py dm-wake-schedule --help`

Implementation: [core/black_holes.py](../core/black_holes.py), `dm_wake_schedule_main`.

Read a diagnostic schedule request JSON and required --output, invoke publish_dm_wake_schedule and return zero after successful publication. Validation/publication errors propagate. This CLI materializes the bounded native diagnostic schedule; it does not run GADGET or compare wake forces.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## dm-wake-state-response

`python commands.py dm-wake-state-response --help`

Implementation: [core/black_holes.py](../core/black_holes.py), `dm_wake_state_response_main`.

Read state-response parameter JSON, default its request base to the file parent and publish the fit to --output. Return zero after publication and propagate errors. Supported versus unresolved calibration is recorded in the output product, rather than inferred from the CLI exit code alone.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## dm-wake-state-table

`python commands.py dm-wake-state-table --help`

Implementation: [core/black_holes.py](../core/black_holes.py), `dm_wake_state_table_main`.

Read state-table parameters, default relative input resolution to the JSON parent and publish to required --output. Return zero after successful export, leaving the native-domain/production limitations in the receipt; exceptions propagate.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## dome

`python commands.py dome --help`

Implementation: [core/visualization.py](../core/visualization.py), `fulldome_planetarium_main`.

Export a dome image and its complete scientific/display provenance.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | nargs='?'; type=Path |
| main | `--state-root` | External runtime state containing runs | type=Path |
| main | `--snapshot` | See live help and implementation. | type=int |
| main | `--tilt` | See live help and implementation. | type=float; default=DEFAULT_TILT_DEG |
| main | `--res` | See live help and implementation. | type=int; default=DEFAULT_RESOLUTION |
| main | `--supersample` | See live help and implementation. | action='store_true' |
| main | `--field` | See live help and implementation. | choices=['dm', 'gas', 'gas_temperature']; default='dm' |
| main | `--observer-mpc-h` | See live help and implementation. | nargs=3; type=float |
| main | `--max-particles` | See live help and implementation. | type=int; default=3000000 |
| main | `--cmap` | See live help and implementation. | default=None |
| main | `--output` | See live help and implementation. | type=Path |

## environment

`python commands.py environment --help`

Implementation: [commands.py](../commands.py), `environment_main`.

Publish or verify a native periodic field/web/splashback product; `--history`
selects the graph-bound multi-epoch producer/verifier. Optional field plots
require a single-epoch product and explicit smoothing index, with optional
halo selection. Bind the request on publication and print status/count metadata.
Scientific acceptance and local feature support remain the producer's recorded
states, not consequences of a completed renderer.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--verify` | See live help and implementation. | action='store_true' |
| main | `--history` | Publish or verify graph-bound multi-epoch reorientation and splashback histories | action='store_true' |
| main | `--figure-prefix` | See live help and implementation. | type=Path |
| main | `--figure-smoothing-index` | See live help and implementation. | type=int |
| main | `--figure-halo-index` | See live help and implementation. | type=int |

## error-budget

`python commands.py error-budget --help`

Implementation: [core/research/errors.py](../core/research/errors.py), `error_budget_main`.

Load the positional request JSON containing keyword arguments for publish_budget_and_domain, publish to required --output and print the result. Normal completion returns None. Measured component values and optional joint covariance must already be present; this command evaluates them rather than generating control measurements.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## error-ensemble

`python commands.py error-ensemble --help`

Implementation: [core/research/errors.py](../core/research/errors.py), `error_ensemble_main`.

Bound --request metadata to 16 MiB, load its parameters and set request_base to the file parent. Publish the measured correlated ensemble at --output and print its JSON publication result. Normal completion returns None; retained-input and publication errors propagate.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--request` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## error-residuals

`python commands.py error-residuals --help`

Implementation: [core/research/errors.py](../core/research/errors.py), `error_residuals_main`.

Load required --request JSON after a 16 MiB metadata bound, resolve request_base from its parent, publish native control residuals to --output and print the publication record. Normal completion returns None; this command does not execute missing native variants.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--request` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## event-cadence

`python commands.py event-cadence --help`

Implementation: [commands.py](../commands.py), `run_event_cadence_main`.

Choose one mutually exclusive operation: merge/quantize requested output times,
plan followups, derive history candidates, execute a checkpoint branch or verify
a retained branch receipt. Nonverification modes require a destination; default
planning writes fresh JSON, while execute mode launches the requested branch.
Print the report. Planning, execution and verified output coverage remain
separate operations with their own input contracts.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path; nargs='?' |
| main | `--execute-branch` | See live help and implementation. | action='store_true' |
| main | `--plan-followups` | See live help and implementation. | action='store_true' |
| main | `--from-history` | See live help and implementation. | action='store_true' |
| main | `--verify-branch` | See live help and implementation. | action='store_true' |

## export-abundance-figures

`python commands.py export-abundance-figures --help`

Implementation: [core/cli.py](../core/cli.py), `export_abundance_figures_main`.

Export native stellar-mass and Vmax abundance from a frozen input request.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `output_directory` | See live help and implementation. |  |
| main | `--kind` | See live help and implementation. | choices=('subhalo', 'host', 'matched'); default='subhalo' |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## export-analysis-figures

`python commands.py export-analysis-figures --help`

Implementation: [core/cli.py](../core/cli.py), `export_analysis_figures_main`.

Connect sealed native analysis datasets to registered scientific figures.

Each recipe names existing datasets and their units. Missing measurements remain
pending. Rendering does not establish physical resolution or convergence.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `analysis_bundle` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |
| main | `--dpi` | See live help and implementation. | type=int; default=160 |

## export-assembly-figures

`python commands.py export-assembly-figures --help`

Implementation: [core/cli.py](../core/cli.py), `export_assembly_figures_main`.

Export registered assembly figures from a source-bound native ancestry stage.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `product` | See live help and implementation. |  |
| main | `identity_graph_directory` | See live help and implementation. |  |
| main | `output_directory` | See live help and implementation. |  |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## export-bh-figures

`python commands.py export-bh-figures --help`

Implementation: [core/cli.py](../core/cli.py), `export_bh_figures_main`.

Export registered growth, feedback and accretion-ratio figures from native BH histories.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `native_bh_directory` | See live help and implementation. |  |
| main | `destination` | See live help and implementation. |  |
| main | `--run-directory` | See live help and implementation. |  |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## export-convergence-figures

`python commands.py export-convergence-figures --help`

Implementation: [core/cli.py](../core/cli.py), `export_convergence_figures_main`.

Export registered convergence figures from pinned native measurements.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path |
| main | `--memory-gib` | See live help and implementation. | type=float; default=2.0 |

## export-hubble-figures

`python commands.py export-hubble-figures --help`

Implementation: [core/cli.py](../core/cli.py), `export_hubble_figures_main`.

Export registered Hubble figures from original measured/frozen products.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## export-inference-figures

`python commands.py export-inference-figures --help`

Implementation: [core/cli.py](../core/cli.py), `export_inference_figures_main`.

Export registered inference graphs from completed original producer products.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## export-merger

`python commands.py export-merger --help`

Implementation: [core/visualization.py](../core/visualization.py), `export_gadget_merger_main`.

Parse command arguments and dispatch this module's documented workflow.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--snapshots` | directory containing GADGET HDF5 snapshot_NNN files | required=True; type=Path |
| main | `--output` | directory for sampled PLY data and manifest | required=True; type=Path |
| main | `--max-particles` | maximum stable ParticleID sample per species | type=int; default=250000 |
| main | `--dark-type` | GADGET particle type for collisionless halo particles | type=int; default=1 |
| main | `--stellar-type` | GADGET particle type for stellar disk/bulge particles | type=int; default=2 |
| main | `--gas-type` | GADGET particle type for gas (included only when present) | type=int; default=0 |
| main | `--black-hole-type` | GADGET particle type for BH markers (included only when present) | type=int; default=5 |
| main | `--length-unit-kpc` | physical kpc per coordinate code unit if HDF5 does not declare UnitLength_in_cm | type=float |
| main | `--assume-noncomoving` | allow snapshots without a ComovingIntegrationOn header after you verify the parameter file | action='store_true' |
| main | `--allow-periodic` | allow periodic inputs after reviewing for box-edge splits | action='store_true' |

## export-power-figures

`python commands.py export-power-figures --help`

Implementation: [core/cli.py](../core/cli.py), `export_power_figures_main`.

Export source-bound linear, evolved or matched matter-power products.

Default positional usage still consumes a sealed native analysis bundle.
Use --kind linear with spectrum.json, or --kind matched with a frozen
flagship-matched-power-v1 request containing measured joint covariance.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `input` | See live help and implementation. | type=Path |
| main | `output_directory` | See live help and implementation. | type=Path |
| main | `--kind` | See live help and implementation. | choices=('linear', 'evolved', 'matched', 'ensemble'); default='evolved' |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## export-sidm-figures

`python commands.py export-sidm-figures --help`

Implementation: [core/cli.py](../core/cli.py), `export_sidm_figures_main`.

Export native SIDM history, fixed-cohort rates and source-bound extra figures.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `operation` | See live help and implementation. | choices=['history', 'scattering', 'centroids', 'extended', 'matched'] |
| main | `input` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path |
| main | `--run-directory` | See live help and implementation. | type=Path |
| main | `--memory-gib` | See live help and implementation. | type=float; default=2.0 |

## export-splashback-figures

`python commands.py export-splashback-figures --help`

Implementation: [core/cli.py](../core/cli.py), `export_splashback_figures_main`.

Export registered splashback diagnostics from a verified native environment.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `product` | See live help and implementation. |  |
| main | `output_directory` | See live help and implementation. |  |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## export-stellar-observer

`python commands.py export-stellar-observer --help`

Implementation: [commands.py](../commands.py), `export_stellar_observer_main`.

Read an observer-export JSON, construct its explicit ObserverFrame and export
the requested saved stellar epoch from the run. Print the publication report.
This produces truth-space observer quantities for exact saved IDs; observational
noise, cuts and missingness belong to the separate observation request route.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `request` | See live help and implementation. | type=Path |

## export-stream-figures

`python commands.py export-stream-figures --help`

Implementation: [core/cli.py](../core/cli.py), `export_stream_figures_main`.

Export registered stream graphs from a sealed measured coordinate product.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `product` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path |
| main | `--memory-gib` | See live help and implementation. | type=float; default=2.0 |

## export-structure-figures

`python commands.py export-structure-figures --help`

Implementation: [core/cli.py](../core/cli.py), `export_structure_figures_main`.

Export registered profiles from a verified native-halo-structure-v1 product.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `product` | See live help and implementation. |  |
| main | `output_directory` | See live help and implementation. |  |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## filament-flythrough

`python commands.py filament-flythrough --help`

Implementation: [commands.py](../commands.py), `filament_flythrough_main`.

Select one saved snapshot and native host row, pass the declared grid/path/
sampling/render budgets to the connected density-ridge flythrough renderer,
and print its output. This can encode a camera film through one saved density
field; camera motion is not cosmic time evolution or a DisPerSE topology
measurement. It does not launch another particle simulation.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `--snapshot` | See live help and implementation. | type=int; required=True |
| main | `--host-row` | See live help and implementation. | type=int; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--grid` | See live help and implementation. | type=int; default=48 |
| main | `--max-host-distance-fraction` | See live help and implementation. | type=float; default=0.15 |
| main | `--path-length-fraction` | See live help and implementation. | type=float; default=0.35 |
| main | `--frames` | See live help and implementation. | type=int; default=240 |
| main | `--fps` | See live help and implementation. | type=int; default=24 |
| main | `--max-particles` | See live help and implementation. | type=int; default=100000 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## find-galaxies

`python commands.py find-galaxies --help`

Implementation: [commands.py](../commands.py), `find_galaxies_main`.

Calculate SUBFIND-bound stellar/gas masses, SFR, and formation epoch.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | nargs='?'; type=Path |
| main | `--snapshot` | See live help and implementation. | action='append'; type=int |
| main | `--minimum-stars` | See live help and implementation. | type=int |
| main | `--minimum-dm` | See live help and implementation. | type=int |
| main | `--stellar-host` | search all stars in this zero-based FOF host row with STF | type=int |
| main | `--linking-length-kpc` | required proper physical stellar linking distance | type=float |
| main | `--stream-minimum-particles` | See live help and implementation. | type=int; default=20 |
| main | `--velocity-neighbors` | See live help and implementation. | type=int; default=32 |
| main | `--physical-neighbors` | See live help and implementation. | type=int; default=256 |
| main | `--cell-fraction` | See live help and implementation. | type=float; default=0.01 |
| main | `--outlier-threshold` | See live help and implementation. | type=float; default=2.5 |
| main | `--significance-level` | See live help and implementation. | type=float; default=1.0 |
| main | `--velocity-ratio` | See live help and implementation. | type=float; default=2.0 |
| main | `--opening-angle-pi` | velocity opening angle divided by pi | type=float; default=0.1 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=4.0 |
| main | `--threads` | See live help and implementation. | type=int; default=1 |

## finder-campaign

`python commands.py finder-campaign --help`

Implementation: [commands.py](../commands.py), `run_finder_campaign_main`.

Read a saved-run finder request and delegate its native finder campaign,
printing the resulting record. `--dry-run` selects the campaign planning path;
ordinary execution may launch the requested finder stages and publish their
outputs. Finder availability and measured catalogue contents remain explicit
in the delegated result.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `request` | See live help and implementation. | type=Path |
| main | `--dry-run` | See live help and implementation. | action='store_true' |

## findings

`python commands.py findings --help`

Implementation: [core/research/findings.py](../core/research/findings.py), `findings_main`.

Parse record, fix, close, reopen, list or revalidate with required --store and a JSON --request for every action except list. Dispatch to FindingStore and print the result; revalidate defaults request_base beside the request and executes the required workflows. Normal completion returns None. Closure uses complete current receipt checks rather than a manually assigned success flag.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `action` | See live help and implementation. | choices=['record', 'fix', 'close', 'reopen', 'list', 'revalidate'] |
| main | `--store` | See live help and implementation. | type=Path; required=True |
| main | `--request` | See live help and implementation. | type=Path |

## flagship

`python commands.py flagship --help`

Implementation: [core/cli.py](../core/cli.py), `flagship_main`.

Unified preparation, execution, recovery, analysis and rendering entrypoint.

Parser owner: [core/flagship_launch.py](../core/flagship_launch.py#L233), `main`. The wrapper forwards to this function.

Parser group `operation` accepts `test`, `run`, `postprocess`, `render`, `resume`. Inspect one with `python commands.py flagship SUBCOMMAND --help`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| prepare | `--request` | Your explicit campaign settings; no bundled run preset is selected | type=Path; required=True |
| prepare | `--output` | See live help and implementation. | type=Path; required=True |
| prepare | `--campaign-request` | See live help and implementation. | type=Path |
| prepare | `--downstream-request` | See live help and implementation. | type=Path |
| prepare | `--memory-gib` | See live help and implementation. | type=float |
| prepare | `--disk-gib` | See live help and implementation. | type=float |
| prepare | `--mpi-ranks` | See live help and implementation. | type=int |
| prepare | `--omp-threads` | See live help and implementation. | type=int |
| prepare | `--build-jobs` | See live help and implementation. | type=int |
| prepare | `--maximum-stage-count` | See live help and implementation. | type=int |
| operation | `--prepared` | See live help and implementation. | type=Path; required=True |
| operation | `--output` | See live help and implementation. | type=Path; required=True |
| operation | `--dry-run` | See live help and implementation. | action='store_true' |
| operation | `--source-output` | See live help and implementation. | type=Path |
| status | `--output` | See live help and implementation. | type=Path; required=True |

## flagship-products

`python commands.py flagship-products --help`

Implementation: [core/cli.py](../core/cli.py), `flagship_products_main`.

Native stream measurements, registered figures, Blender timeline and yields.

Parser owner: [core/flagship_products.py](../core/flagship_products.py#L1222), `flagship_products_main`. The wrapper forwards to this function.

Parser group `kind` accepts `sidm-matched-figures`, `sidm-history-figures`, `web-history-figures`, `web-figures`, `cloud-figures`, `first-star-figures`, `metal-history-figures`, `population-history-figures`, `mass-return-figures`, `sfr-history-figures`, `gas-phase-figures`, `bh-dm-response`, `bh-recoil-figures`, `bh-pair-figures`, `bh-merger-figures`, `bh-figures`, `analysis-figures`, `assembly-orbits`, `assembly-event-figures`, `abundance`, `host-abundance`, `matched-host-abundance`, `assembly-figures`, `splashback-figures`, `structure-figures`, `power-linear`, `power-evolved`, `power-matched`, `power-ensemble`, `encounter-figures`, `encounter-catalogue`, `encounter-frame`, `encounter-images`, `ic-species-window-modes`, `ic-quadratic-assembly`, `ic-poisson-equations`, `ic-gradient-operators`, `ic-boundary-reference`, `ic-transfer-replay`, `ic-compensated-windows`, `ic-flow-windows`, `ic-flow-noise`, `ic-flow-covariance`, `ic-compensated-covariance`, `stream-derived`, `stream-catalogue`, `stream-epoch`, `stream-figures`, `stream-timeline`, `yield-audit`, `convergence-figures`, `profile-preflight`. Inspect one with `python commands.py flagship-products SUBCOMMAND --help`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| kind | `input` | See live help and implementation. | type=Path |
| kind | `--output` | See live help and implementation. | type=Path; required=True |
| kind | `--memory-gib` | See live help and implementation. | type=float; default=2.0 |
| kind | `--request` | See live help and implementation. | type=Path; required=True |
| kind | `--maximum-saved-vertices` | See live help and implementation. | type=int; default=2000000 |
| kind | `--output-directory` | See live help and implementation. | type=Path; required=True |
| kind | `--configuration` | See live help and implementation. | type=Path; required=True |
| kind | `--source-index` | See live help and implementation. | type=Path; required=True |
| kind | `--identity-graph` | See live help and implementation. | type=Path; required=True |
| kind | `--run-directory` | See live help and implementation. | type=Path |
| kind | `--run-directory` | See live help and implementation. | type=Path; required=True |

## gadget

`python commands.py gadget --help`

Implementation: [commands.py](../commands.py), `run_gadget_main`.

Run the physics engine; later stages can be launched independently.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--answers` | JSON object with simulation answers for repeatable runs | type=Path |
| main | `--draft` | Persist and resume interactive configuration at this JSON path | type=Path |
| main | `--cosmology` | required spectrum.json or its run folder |  |
| main | `--run-dir` | Explicit output directory instead of the suite run-name directory | type=Path |
| main | `--resume` | Continue the same frozen native run from verified rank checkpoints | action='store_true' |
| main | `--recover-checkpoint` | Complete verified checkpoint recovery before native resume | action='store_true' |
| main | `--dry-run` | validate and stage a GADGET plan without evolving | action='store_true' |
| main | `--ic-only` | generate and validate ICs, then stop | action='store_true' |
| main | `--existing-ic` | Use delivered existing ICs instead of generating a new load | type=Path |
| main | `--native-generation-report` | Require revalidated native MUSIC parent/zoom handoff | type=Path |
| main | `--validated-public-ic` | Bind an original validated public hydro IC-only generation and its transfer source; all native IC gates run again | type=Path |
| main | `--zoom-softening-mass-ratios` | Explicit sorted coarse/fine-DM mass ratios, comma separated |  |
| main | `--prepare-only` | Validate an existing native handoff and build GADGET without evolution | action='store_true' |
| main | `--evolution-only` | Evolve the selected physics; leave saved-output analysis to its workflow stages | action='store_true' |
| main | `--research-spec` | Versioned research specification; gates resolved-stream feasibility before launch | type=Path |
| main | `--numerical-controls` | Execute a declared analytic/native reliability matrix | type=Path |
| main | `--resume-controls` | Reuse a current completed matching control matrix | action='store_true' |

## halo-anatomy

`python commands.py halo-anatomy --help`

Implementation: [commands.py](../commands.py), `halo_anatomy_main`.

Select a saved snapshot and host row and build its interactive cutaway and
radial-profile output with the requested grid/layer/memory settings. Print the
published location. It displays measured saved particles and profiles, rather
than generating a new halo or adding unresolved internal structure.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `--snapshot` | See live help and implementation. | type=int; required=True |
| main | `--host-row` | See live help and implementation. | type=int; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--grid` | See live help and implementation. | type=int; default=96 |
| main | `--layers` | See live help and implementation. | type=int; default=24 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## halo-collision-movie

`python commands.py halo-collision-movie --help`

Implementation: [commands.py](../commands.py), `halo_collision_movie_main`.

Use saved ancestry and map bundles to animate two linked progenitor branches
near a selected merger, optionally choosing an exact descendant ID. Pass the
map key and playback/memory settings and print the output. The film displays
recorded geometry; physical collision interpretation still needs the original
trajectory, source and resolution evidence.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `analysis_bundle` | See live help and implementation. | type=Path |
| main | `map_bundle` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--map-key` | See live help and implementation. | required=True |
| main | `--descendant-id` | Optional exact tree-node ID to select a merger |  |
| main | `--fps` | See live help and implementation. | type=int; default=24 |
| main | `--seconds-per-snapshot` | See live help and implementation. | type=int; default=1 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## halo-fingerprint

`python commands.py halo-fingerprint --help`

Implementation: [commands.py](../commands.py), `halo_fingerprint_main`.

Compare a selected host between two saved runs/snapshots through reciprocal
exact-ID membership matching. Pass shared-count/fraction, grid, epoch tolerance
and memory controls to the fingerprint producer; publish and print the result.
A descriptive matched-host display is not a convergence or causal comparison
unless the broader experiment supplies that evidence.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_a` | See live help and implementation. | type=Path |
| main | `run_b` | See live help and implementation. | type=Path |
| main | `--snapshot-a` | See live help and implementation. | type=int; required=True |
| main | `--snapshot-b` | See live help and implementation. | type=int; required=True |
| main | `--host-row` | See live help and implementation. | type=int; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--min-shared` | See live help and implementation. | type=int; default=10 |
| main | `--min-fraction` | See live help and implementation. | type=float; default=0.1 |
| main | `--grid` | See live help and implementation. | type=int; default=128 |
| main | `--redshift-tolerance` | See live help and implementation. | type=float; default=1e-06 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## halo-formation-movie

`python commands.py halo-formation-movie --help`

Implementation: [commands.py](../commands.py), `halo_formation_movie_main`.

Animate measured saved halo ancestry beside its associated density-map series,
using an exact map key and explicit timing/memory settings. Publish and print
the result. It reads existing tree/map products and does not rerun formation
physics or invent missing progenitor positions.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `analysis_bundle` | See live help and implementation. | type=Path |
| main | `map_bundle` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--map-key` | See live help and implementation. | required=True |
| main | `--fps` | See live help and implementation. | type=int; default=24 |
| main | `--seconds-per-snapshot` | See live help and implementation. | type=int; default=1 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## halo-story

`python commands.py halo-story --help`

Implementation: [commands.py](../commands.py), `halo_story_main`.

Connect a selected positioned tree, recorded density contract and original
density directory to a synchronized labelled map/tree/film output. Pass the
run binding and explicit playback rate/hold duration and print the published
result. It uses saved source geometry and ancestry rather than fabricating
undetected early progenitors.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run` | See live help and implementation. | type=Path |
| main | `--tree` | See live help and implementation. | type=Path; required=True |
| main | `--density-contract` | See live help and implementation. | type=Path; required=True |
| main | `--density-directory` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--fps` | See live help and implementation. | type=int; default=24 |
| main | `--hold-seconds` | See live help and implementation. | type=int; default=1 |

## halo-structure

`python commands.py halo-structure --help`

Implementation: [commands.py](../commands.py), `halo_structure_main`.

Publish complete native component profiles and validated proper host fields
from a request, binding its hash and command source. `--verify` instead reloads
and verifies the existing output; an optional figure prefix requests profile/
validation plots. Print measurement, field-validation and science statuses.
The command measures or replays saved particles, not a new halo evolution.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--verify` | Verify and reload an existing sealed structure product | action='store_true' |
| main | `--figure-prefix` | Publish a source-bound PNG/PDF profile and validation figure | type=Path |

## halos

`python commands.py halos --help`

Implementation: [commands.py](../commands.py), `run_halos_main`.

Allow halo finding to be repeated without touching GADGET evolution.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `--finder` | See live help and implementation. | choices=['rockstar', 'fof', 'both'] |
| main | `--min-particles` | See live help and implementation. | type=int |
| main | `--campaign-request` | explicit native finder/tree stages and saved snapshot selections | type=Path |
| main | `--fof-linking-length` | required FOF linking length in mean inter-particle spacings | type=float |
| main | `--dry-run` | See live help and implementation. | action='store_true' |

## hmf

`python commands.py hmf --help`

Implementation: [core/cli.py](../core/cli.py), `hmf_main`.

Forward the HMF/HMF-scan CLI to the linear prediction campaign driver. Its
request selects cosmology, epochs, masses/formulas and optional sampling; it
can run CLASS/AxiCLASS or reuse matching saved linear epochs, then publish
tables, plots and candidate records. Return the delegate's completion code.
These formula abundances are screening predictions, not measured GADGET counts
or automatically launched particle simulations.

Parser owner: [core/hmf_campaign.py](../core/hmf_campaign.py#L435), `hmf_main`. The wrapper forwards to this function.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--spec` | See live help and implementation. | required=True; type=Path |
| main | `--output` | See live help and implementation. | required=True; type=Path |
| main | `--workers` | See live help and implementation. | type=int; default=1 |
| main | `--binary` | See live help and implementation. | type=Path |
| main | `--linear-bundle` | HMF only: reuse exact epochs from a verified commands.py class bundle | type=Path |

## hmf-learning

`python commands.py hmf-learning --help`

Implementation: [core/cli.py](../core/cli.py), `hmf_learning_main`.

Forward to the measured-HMF learning CLI for native catalogue export, dataset
construction, training or prediction operations. The workflow keeps simulation
counts/volumes and cosmology-level splits distinct from analytic screening
predictions, and writes each operation's requested products. Return the
underlying result; this wrapper supplies neither training targets nor fitted
coefficients by itself.

Parser owner: [core/hmf_learning.py](../core/hmf_learning.py#L443), `learning_main`. The wrapper forwards to this function.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--run` | See live help and implementation. | type=Path |
| main | `--runs-root` | Recursively export many saved GADGET runs | type=Path |
| export-gadget | `--output` | See live help and implementation. | type=Path; required=True |
| export-gadget | `--snapshots` | See live help and implementation. | type=int; nargs='+' |
| build | `--spec` | See live help and implementation. | type=Path; required=True |
| build | `--output` | See live help and implementation. | type=Path; required=True |
| train | `--dataset` | See live help and implementation. | type=Path; required=True |
| train | `--output` | See live help and implementation. | type=Path; required=True |
| train | `--degree` | See live help and implementation. | type=int; choices=(1, 2); default=2 |
| train | `--features` | See live help and implementation. | nargs='+' |
| predict | `--model` | See live help and implementation. | type=Path; required=True |
| predict | `--features-csv` | See live help and implementation. | type=Path; required=True |
| predict | `--output` | See live help and implementation. | type=Path; required=True |
| predict | `--allow-extrapolation` | See live help and implementation. | action='store_true' |

## hmf-scan

`python commands.py hmf-scan --help`

Implementation: [core/cli.py](../core/cli.py), `hmf_main`.

Forward the HMF/HMF-scan CLI to the linear prediction campaign driver. Its
request selects cosmology, epochs, masses/formulas and optional sampling; it
can run CLASS/AxiCLASS or reuse matching saved linear epochs, then publish
tables, plots and candidate records. Return the delegate's completion code.
These formula abundances are screening predictions, not measured GADGET counts
or automatically launched particle simulations.

Parser owner: [core/hmf_campaign.py](../core/hmf_campaign.py#L435), `hmf_main`. The wrapper forwards to this function.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--spec` | See live help and implementation. | required=True; type=Path |
| main | `--output` | See live help and implementation. | required=True; type=Path |
| main | `--workers` | See live help and implementation. | type=int; default=1 |
| main | `--binary` | See live help and implementation. | type=Path |
| main | `--linear-bundle` | HMF only: reuse exact epochs from a verified commands.py class bundle | type=Path |

## hydro-comparison

`python commands.py hydro-comparison --help`

Implementation: [commands.py](../commands.py), `hydro_comparison_main`.

With `--design`, publish a fresh sealed matched-physics configuration design.
Otherwise measure the requested saved radial hydro comparison. Both modes
hash-check the request across work and print schema/physics-acceptance metadata.
Neither mode launches evolution, and a valid design is not an executed matched
comparison.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--design` | See live help and implementation. | action='store_true' |

## ingest-bao

`python commands.py ingest-bao --help`

Implementation: [core/cli.py](../core/cli.py), `ingest_bao_main`.

Publish a pinned measured BAO input without fitting cosmological parameters.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## ingest-cosmography

`python commands.py ingest-cosmography --help`

Implementation: [core/cli.py](../core/cli.py), `ingest_cosmography_main`.

Validate measured ladder/full-lens inputs without executing an inference fit.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path |

## inspect-poisson-failure

`python commands.py inspect-poisson-failure --help`

Implementation: [core/cli.py](../core/cli.py), `inspect_poisson_failure_main`.

Inspect a saved native failed Poisson field without changing its source.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `snapshot` | See live help and implementation. | type=Path |
| main | `--order` | See live help and implementation. | type=int; choices=(2, 4, 6); required=True |
| main | `--memory-gib` | See live help and implementation. | type=float; default=0.25 |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## isolated-control-history

`python commands.py isolated-control-history --help`

Implementation: [core/sidm.py](../core/sidm.py), `isolated_history_main`.

Read required --parameters JSON and measure the isolated native conservation/collapse history. Require a fresh --output, then atomically write the record and sibling .sha256 seal. Normal completion returns None; invalid inputs raise. A written record retains PASS/FAIL control gates separately from execution completion, and the CLI does not evolve a halo.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--parameters` | See live help and implementation. | required=True; type=Path |
| main | `--output` | See live help and implementation. | required=True; type=Path |

## lens-map

`python commands.py lens-map --help`

Implementation: [commands.py](../commands.py), `lens_map_main`.

Compute a thin-lens convergence/shear display from a selected saved mass map.
Require the snapshot/map key, three distances in kpc and background surface
mass density in Msun/kpc², then publish and print the output. These supplied
geometry/calibration inputs are not a fitted cosmology, lens posterior or
new ray-traced simulation.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run` | See live help and implementation. | type=Path |
| main | `map_bundle` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--map-key` | See live help and implementation. | required=True |
| main | `--snapshot` | See live help and implementation. | type=int; required=True |
| main | `--lens-distance-kpc` | See live help and implementation. | type=float; required=True |
| main | `--source-distance-kpc` | See live help and implementation. | type=float; required=True |
| main | `--lens-source-distance-kpc` | See live help and implementation. | type=float; required=True |
| main | `--background-msun-kpc2` | See live help and implementation. | type=float; required=True |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## lensing

`python commands.py lensing --help`

Implementation: [core/black_holes.py](../core/black_holes.py), `main`.

Parse command arguments and dispatch this module's documented workflow.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--panorama` | equirectangular all-sky source image | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; default=Path('black_holes.png') |
| main | `--mass-solar` | black-hole mass in solar masses | type=float; required=True |
| main | `--observer-rs` | static observer radius in Schwarzschild radii | type=float; default=50.0 |
| main | `--fov-deg` | horizontal perspective field of view | type=float; default=60.0 |
| main | `--width` | See live help and implementation. | type=int; default=1600 |
| main | `--height` | See live help and implementation. | type=int; default=900 |
| main | `--radial-samples` | more samples resolve photon-ring winding more finely | type=int |
| main | `--roll-deg` | rotate the panorama horizontally | type=float; default=0.0 |

## matched-hydro-pilot

`python commands.py matched-hydro-pilot --help`

Implementation: [core/matched.py](../core/matched.py), `matched_hydro_pilot_main`.

Read --request JSON after a 16 MiB metadata bound and compile the matched hydro pilot into required --output, resolving pinned request assets relative to the request file’s parent. Print the design record and return None on normal completion. This CLI prepares the workflow without executing its simulations.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--request` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## measure-halo-shapes

`python commands.py measure-halo-shapes --help`

Implementation: [commands.py](../commands.py), `measure_halo_shapes_main`.

Read a run and radial-shape request and publish its measured native-host
shape product; `--resume` verifies/reuses an unchanged publication. Print shell
statuses, profiles and scientific acceptance. Exit nonzero unless the result
status is MEASURED. The selected support sphere/FOF/SUBFIND convention remains
the request's measurement definition, not automatic global host binding.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `request` | See live help and implementation. | type=Path |
| main | `--resume` | verify and reuse an unchanged published request | action='store_true' |

## measure-stream-heating

`python commands.py measure-stream-heating --help`

Implementation: [commands.py](../commands.py), `measure_stream_heating_main`.

Load a JSON request and publish the numerical heating calibration from its
paired native granular/smooth controls at the requested output. Print the
reported status. This delegates analysis of the specified controls; the wrapper
does not launch new evolutions or itself convert a reported non-PASS into
an explicit exit code.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## measure-zoom-campaign

`python commands.py measure-zoom-campaign --help`

Implementation: [core/research/zooms.py](../core/research/zooms.py), `zoom_diagnostics_main`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--run` | See live help and implementation. | type=Path; required=True |
| main | `--recovery` | See live help and implementation. | type=Path; required=True |
| main | `--request` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## measured-scene

`python commands.py measured-scene --help`

Implementation: [core/blender_scene.py](../core/blender_scene.py), `scene_main`.

Parse a manifest and required output path. By default package its verified
display assets and print the receipt. With `--build`, require Blender's bpy and
a fresh .blend path, create the sibling .assets package, attach measured display
objects to the active scene, set its relative asset base and save the .blend
under the native lock. It writes display/scene files; building or packaging
alone is not a fresh-process replay or native scientific calculation.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `manifest` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--build` | See live help and implementation. | action='store_true' |

## merger-tree-map

`python commands.py merger-tree-map --help`

Implementation: [commands.py](../commands.py), `merger_tree_map_main`.

Join a run's verified ancestry analysis to a saved map bundle using an exact
map key and explicit memory allocation. Publish the connected tree/map display
at the selected output and print its result. The underlying density arrays,
tree IDs and epochs remain the source measurements.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `analysis_bundle` | See live help and implementation. | type=Path |
| main | `map_bundle` | See live help and implementation. | type=Path |
| main | `--map-key` | See live help and implementation. | required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## metallicity-maps

`python commands.py metallicity-maps --help`

Implementation: [commands.py](../commands.py), `metallicity_maps_main`.

Add gas metallicity projections to a selected saved structure-map bundle.
Pass the source map key, run, destination and memory budget to the producer
and print its JSON record. It measures saved gas fields in the map's geometry;
it does not infer metallicity from image colour or evolve chemical enrichment.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `map_bundle` | See live help and implementation. | type=Path |
| main | `--source-key` | See live help and implementation. | required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## multifield-movie

`python commands.py multifield-movie --help`

Implementation: [commands.py](../commands.py), `multifield_movie_main`.

Encode synchronized panels from two to four explicitly repeated saved-map keys.
Pass the recorded bundle, optional source run, destination and playback/memory
settings; print the result. Field availability and epoch coverage come from
the map contracts, rather than copying a neighbouring field into a missing panel.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `bundle` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--map-key` | Map key from scale.json; repeat two to four times | action='append'; required=True |
| main | `--fps` | See live help and implementation. | type=int; default=24 |
| main | `--seconds-per-snapshot` | See live help and implementation. | type=int; default=1 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |
| main | `--run` | Source run if it cannot be identified from map provenance | type=Path |

## music-boundary

`python commands.py music-boundary --help`

Implementation: [core/cli.py](../core/cli.py), `music_boundary_main`.

Build the native MUSIC boundary response adapter without running ICs.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--compiler` | See live help and implementation. |  |
| main | `--include` | See live help and implementation. | type=Path; action='append'; default=[] |

## music-capacity

`python commands.py music-capacity --help`

Implementation: [core/cli.py](../core/cli.py), `music_capacity_main`.

Plan native generation, response, coupled-solver and covariance resources.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## music-flow-prepare

`python commands.py music-flow-prepare --help`

Implementation: [core/cli.py](../core/cli.py), `music_flow_prepare_main`.

Freeze native window launch settings in complete public evolution answers.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## music-frozen-model

`python commands.py music-frozen-model --help`

Implementation: [core/cli.py](../core/cli.py), `music_frozen_model_main`.

Verify retained complete MUSIC fields against their original seed streams.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## music-phase-plan

`python commands.py music-phase-plan --help`

Implementation: [core/cli.py](../core/cli.py), `music_phase_plan_main`.

Preflight native tiled phase memory and cache storage on retained geometry.

Parser owner: [core/music/resources.py](../core/music/resources.py#L420), `phase_resources_main`. The wrapper forwards to this function.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## music-seeded-model

`python commands.py music-seeded-model --help`

Implementation: [core/cli.py](../core/cli.py), `music_seeded_model_main`.

Verify fixed-prefix local MUSIC caches against their declared native seed streams.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## native-counterfactual-campaign

`python commands.py native-counterfactual-campaign --help`

Implementation: [core/streams.py](../core/streams.py), `native_counterfactual_campaign_main`.

Read a native live-response campaign request and compile its follow-up workflow into --output using the request parent as base. Return zero after preparation. Compilation creates executable recipes; it does not establish that their native runs or comparisons completed.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## native-counterfactual-ic

`python commands.py native-counterfactual-ic --help`

Implementation: [core/streams.py](../core/streams.py), `native_counterfactual_ic_main`.

Read native intervention parameter JSON and required --output, invoke publish_native_counterfactual_ic and return zero on publication. Exceptions propagate. This CLI does not insert a request_base default or launch evolution; parameter paths must satisfy the publisher’s input rules.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## native-error-envelopes

`python commands.py native-error-envelopes --help`

Implementation: [core/research/errors.py](../core/research/errors.py), `native_error_envelopes_main`.

Require --request and --output, enforce an eight-times-metadata expansion estimate within 16 MiB, load parameters and set request_base to the request parent. Publish the native systematic envelope and print its JSON result. Normal completion returns None; absent or invalid retained controls/certificates raise.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--request` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## native-late-threeway

`python commands.py native-late-threeway --help`

Implementation: [core/streams.py](../core/streams.py), `native_late_threeway_main`.

Read three-way parameters JSON, default its request base to the file parent and publish to required --output. Return zero after publication; the saved product’s restart-budget status remains authoritative and errors propagate. This CLI measures existing parent/fresh-control/intervention differences rather than evolving those runs.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## native-music-mixed2lpt

`python commands.py native-music-mixed2lpt --help`

Implementation: [core/research/generator_controls.py](../core/research/generator_controls.py), `native_music_mixed2lpt_main`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--binary` | See live help and implementation. | type=Path; required=True |
| main | `--monofonic` | See live help and implementation. | type=Path; required=True |
| main | `--legacy-proof` | See live help and implementation. | type=Path; required=True |

## native-music-transfer

`python commands.py native-music-transfer --help`

Implementation: [core/research/generator_controls.py](../core/research/generator_controls.py), `native_music_transfer_main`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `root` | See live help and implementation. | type=Path |
| main | `--binary` | See live help and implementation. | type=Path |
| main | `--prepare` | See live help and implementation. | action='store_true' |
| main | `--nested` | See live help and implementation. | action='store_true' |

## native-prepare

`python commands.py native-prepare --help`

Implementation: [core/campaigns.py](../core/campaigns.py), `main`.

Parse native-prepare domain arguments and lazily import only the selected campaign producer. Common domains consume a JSON request and fresh destination; responses binds a design/template hash, while joint-response consumes original IC manifest/execution references. Resolve SIDM and joint-response assets beside their request/design file where explicitly supported. Print the prepared result as JSON and return zero. This adapter prepares campaign recipes; native evolution is launched separately through their workflow.

Parser group `name` accepts `assembly`, `baryons`, `black-holes`, `convergence`, `encounters`, `halos`, `power`, `response-branch`, `sbi`, `sidm`, `web`, `recovered-sidm`, `joint-ics`. Inspect one with `python commands.py native-prepare SUBCOMMAND --help`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| name | `request` | JSON campaign request | type=Path |
| name | `destination` | new recipe directory | type=Path |
| responses | `design` | See live help and implementation. | type=Path |
| responses | `template` | See live help and implementation. | type=Path |
| responses | `destination` | See live help and implementation. | type=Path |
| joint-response | `design` | See live help and implementation. | type=Path |
| joint-response | `--ic-manifest` | See live help and implementation. | type=Path; required=True |
| joint-response | `--ic-execution` | See live help and implementation. | type=Path; required=True |
| joint-response | `--output` | See live help and implementation. | type=Path; required=True |

## native-sidm-zoom-support

`python commands.py native-sidm-zoom-support --help`

Implementation: [core/apertures.py](../core/apertures.py), `native_zoom_support_main`.

Parse required `--request` and `--output` paths, bound the request-file intake,
load its JSON and print the numeric publisher's JSON result. The file contains
the full parameter envelope: `native_stage_receipt`, `native_run_directory` and
the inner `request` object, not only the inner request. Publication reads verified
native support fields and writes the destination audit/arrays. Validation or
integrity errors propagate. A measured FAIL remains a report status; this handler
does not itself map it to a nonzero exit code or launch native evolution.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--request` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## native-source-events

`python commands.py native-source-events --help`

Implementation: [commands.py](../commands.py), `native_source_events_main`.

Read a hash-stable physical-source event request and publish the linked
journal/event-cadence product at the selected output. Recheck the request after
publication and print event count and cadence acceptance. This connects retained
source journals to measured windows; it does not execute those source operators.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## native-source-overlay

`python commands.py native-source-overlay --help`

Implementation: [commands.py](../commands.py), `native_source_overlay_main`.

Without a source root, verify and describe a sealed native full-file overlay.
Explicit installation additionally requires a stable prior-file hash mapping
and writes the overlay into the selected source root, then prints changed-file
count and build-execution status. Installation changes source files but does
not compile, test or run the native solver.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `manifest` | See live help and implementation. | type=Path |
| main | `--source-root` | See live help and implementation. | type=Path |
| main | `--expected-hashes` | See live help and implementation. | type=Path |

## native-stream-comparison

`python commands.py native-stream-comparison --help`

Implementation: [core/streams.py](../core/streams.py), `native_stream_comparison_main`.

Read paired native comparison parameters, default request_base to the JSON parent and publish to --output. Return zero on publication with errors propagated. Compare actual saved self-gravitating branches; this command does not replace them with imposed-field trajectories.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## native-stream-worldlines

`python commands.py native-stream-worldlines --help`

Implementation: [core/streams.py](../core/streams.py), `native_stream_worldline_main`.

Read native stream-history parameters JSON and publish selected original trajectories to --output. Return zero after publication and propagate errors. The adapter supplies no default request_base and does not integrate new orbits.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## observe-stellar-catalogue

`python commands.py observe-stellar-catalogue --help`

Implementation: [commands.py](../commands.py), `observe_stellar_catalogue_main`.

Read a saved observer-truth directory and JSON observation request, then call
the mock-observation exporter with its declared correlated errors, missingness
and cuts. Print the resulting publication record. It produces observations from
supplied stellar truth; it does not evolve stars or infer a new truth catalogue.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `truth_directory` | See live help and implementation. | type=Path |
| main | `request` | See live help and implementation. | type=Path |

## package

`python commands.py package --help`

Implementation: [core/release.py](../core/release.py), `package_suite_main`.

Package the current sources to the requested external ZIP destination.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--output` | See live help and implementation. | type=Path |
| main | `--verify` | verify an existing release before extraction | type=Path |

## parameter-explorer

`python commands.py parameter-explorer --help`

Implementation: [commands.py](../commands.py), `parameter_explorer_main`.

Read an explicit saved-run comparison request, build its offline parameter
explorer at the required destination, and reject a request hash change during
export. Print the result path. It organizes the supplied runs and measurements;
it does not interpolate an uncomputed cosmology or launch a parameter scan.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## particle-origin-movie

`python commands.py particle-origin-movie --help`

Implementation: [commands.py](../commands.py), `particle_origin_movie_main`.

Choose the exact DM members of a saved anchor host and trace those identities
through saved epochs, with explicit sample and playback bounds. Publish the
origin-history film/output and print its result. This follows a selected
material cohort; it neither changes membership to improve appearance nor
creates unsaved early states.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--anchor-snapshot` | See live help and implementation. | type=int; required=True |
| main | `--host-row` | See live help and implementation. | type=int; required=True |
| main | `--max-particles` | See live help and implementation. | type=int; default=50000 |
| main | `--fps` | See live help and implementation. | type=int; default=24 |
| main | `--seconds-per-snapshot` | See live help and implementation. | type=int; default=1 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## physical-target

`python commands.py physical-target --help`

Implementation: [core/zoom.py](../core/zoom.py), `physical_target_main`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `mode` | See live help and implementation. | choices=('support', 'model-control', 'measurement', 'budget', 'convergence') |
| main | `--parameters` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## pilots

`python commands.py pilots --help`

Implementation: [core/research/pilots.py](../core/research/pilots.py), `pilots_main`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--campaign` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--resume` | See live help and implementation. | action='store_true' |
| main | `--dry-run` | See live help and implementation. | action='store_true' |

## planetarium-movie

`python commands.py planetarium-movie --help`

Implementation: [commands.py](../commands.py), `planetarium_movie_main`.

Render the requested saved epochs/field into a fixed-scale dome output using
the supplied observer, raster resolution, tilt, sampling and timing budgets.
Print the resulting publication. This command can produce a dome movie from
saved data; the projection is not by itself a cosmological light cone or
an emitted-light calculation.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--field` | See live help and implementation. | choices=['dm', 'gas', 'gas_temperature']; default='dm' |
| main | `--snapshot` | Saved snapshot; repeat to select epochs | type=int; action='append' |
| main | `--observer` | See live help and implementation. | type=float; nargs=3 |
| main | `--resolution` | See live help and implementation. | type=int; default=512 |
| main | `--tilt` | See live help and implementation. | type=float; default=0.0 |
| main | `--max-particles` | See live help and implementation. | type=int; default=1000000 |
| main | `--fps` | See live help and implementation. | type=int; default=24 |
| main | `--seconds-per-snapshot` | See live help and implementation. | type=int; default=1 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## prepare-analysis-figures

`python commands.py prepare-analysis-figures --help`

Implementation: [core/cli.py](../core/cli.py), `prepare_analysis_figures_main`.

Prepare original observational/inference campaign and registered graph stages.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path |

## prepare-flagship-campaign

`python commands.py prepare-flagship-campaign --help`

Implementation: [core/cli.py](../core/cli.py), `prepare_flagship_campaign_main`.

Prepare selected native campaign recipes using the existing workflow engine.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--matched` | Combine prepared CDM/SIDM branches with a shared IC producer per target. | action='store_true' |
| main | `--production` | Connect retained production bridge staging and prepared evolution settings. | action='store_true' |
| main | `--history` | Extend a prepared native campaign with material recovery and actual epoch coverage. | action='store_true' |

## prepare-flagship-profile

`python commands.py prepare-flagship-profile --help`

Implementation: [core/cli.py](../core/cli.py), `prepare_flagship_profile_main`.

Freeze selected flagship profile settings after actual resource preflight.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `profile` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--case` | Select a case ID; repeat for any number of cases. Default: all supplied cases. | action='append' |
| main | `--sidm-settings` | Prepare paired CDM/SIDM profiles using this explicit native SIDM configuration. | type=Path |
| main | `--freeze-matched` | Read a prepared matched-pair manifest and freeze both branches with current resource checks. | action='store_true' |
| main | `--geometry-only` | Estimate a pinned future production geometry; does not freeze launch settings or generate ICs. | action='store_true' |
| main | `--hardware-requirements` | Prepare larger-hardware requirements while retaining the selected production targets and resolution. | action='store_true' |
| main | `--stage-production` | Stage production inputs from the retained parent phase prefix; never generate production ICs. | action='store_true' |
| main | `--plan-production` | Run current-generator mesh-only planning on completed staged production inputs. | action='store_true' |
| main | `--production-answers` | Materialize complete native production evolution settings without launching. | action='store_true' |
| main | `--headroom-fraction` | Provisioning headroom assumption for --hardware-requirements (default: 0.25). | type=float; default=0.25 |

## prepare-generated-flow

`python commands.py prepare-generated-flow --help`

Implementation: [commands.py](../commands.py), `prepare_generated_flow_main`.

Prepare evolution answers from the exact newly generated native mesh.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--request` | See live help and implementation. | type=Path; required=True |
| main | `--generation-report` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## prepare-source-controls

`python commands.py prepare-source-controls --help`

Implementation: [commands.py](../commands.py), `prepare_source_controls_main`.

Read and hash-check an explicit native-source control request, then write its
flat hydro/stellar/BH IC and build-profile products under the selected prefix.
Recheck request stability and print file/execution metadata. Preparation writes
inputs only; build and evolution remain separate requested actions.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output-prefix` | See live help and implementation. | type=Path; required=True |

## production-resources

`python commands.py production-resources --help`

Implementation: [core/resources.py](../core/resources.py), `resources_main`.

Parse a resource request, required fresh `--output` path and optional
`--measurements` JSON. Estimate production resources and, when supplied, update
the plan from actual stage receipts; atomically write the plan JSON and print
it. Reject an existing report rather than overwriting history. This CLI does
not allocate a machine, change the requested resolution/ranks or run the
scientific campaign.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--measurements` | See live help and implementation. | type=Path |

## propose-tree-repair

`python commands.py propose-tree-repair --help`

Implementation: [commands.py](../commands.py), `propose_tree_repair_main`.

Publish proposed alternative links against a supplied original audit, rationale
and thresholds, binding the request and command source hashes. Print proposal
status and diagnostic count. With `--verify`, instead verify the existing
destination proposal. It does not replace the original native tree or silently
promote proposed ancestry to accepted history.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path |
| main | `--verify` | See live help and implementation. | action='store_true' |

## query-identity-graph

`python commands.py query-identity-graph --help`

Implementation: [commands.py](../commands.py), `query_identity_graph_main`.

Read a verified indexed graph and an explicit JSON query/arguments pair.
Allow only members-at-epoch, descendants, bound-owner, birth-parent or
parent-zoom queries; invoke that query and print finite JSON, converting array
results to lists. It reads indexed identity evidence without modifying the
graph or inferring missing links.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `directory` | See live help and implementation. | type=Path |
| main | `request` | See live help and implementation. | type=Path |

## release-blender-control

`python commands.py release-blender-control --help`

Implementation: [core/research/release.py](../core/research/release.py), `release_blender_control_main`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--workflow` | See live help and implementation. | type=Path; required=True |
| main | `--scene-product` | See live help and implementation. | required=True |
| main | `--scene-artifact` | See live help and implementation. | required=True |
| main | `--blender` | See live help and implementation. | required=True |
| main | `--wall-seconds` | See live help and implementation. | type=float; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## render-scene

`python commands.py render-scene --help`

Implementation: [core/cli.py](../core/cli.py), `render_scene_main`.

Forward to the sealed-scene renderer CLI. It verifies the saved scene, requires
explicit projection, image/frame and resource controls, and writes a prepared
request/launch record to a fresh output. Default execution launches the selected
Blender binary for frames; `--prepare-only` stops at preparation. Return the
underlying result. It renders an existing scene rather than evolving particles.

Parser owner: [core/scene_render.py](../core/scene_render.py#L90), `prepare_render_main`. The wrapper forwards to this function.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `scene` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | required=True; type=Path |
| main | `--projection` | See live help and implementation. | choices=['active', 'perspective', 'dome']; required=True |
| main | `--scene-name` | See live help and implementation. |  |
| main | `--width` | See live help and implementation. | type=int; required=True |
| main | `--height` | See live help and implementation. | type=int; required=True |
| main | `--samples` | See live help and implementation. | type=int; required=True |
| main | `--frame-start` | See live help and implementation. | type=int; required=True |
| main | `--frame-end` | See live help and implementation. | type=int; required=True |
| main | `--fps` | See live help and implementation. | type=int; required=True |
| main | `--memory-gib` | See live help and implementation. | required=True; type=float |
| main | `--disk-gib` | See live help and implementation. | required=True; type=float |
| main | `--device` | See live help and implementation. | default='AUTO'; choices=['AUTO', 'CPU', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL'] |
| main | `--wall-seconds` | See live help and implementation. | required=True; type=int |
| main | `--blender` | See live help and implementation. | type=Path |
| main | `--prepare-only` | See live help and implementation. | action='store_true' |

## repair-receipt

`python commands.py repair-receipt --help`

Implementation: [core/research/findings.py](../core/research/findings.py), `repair_receipt_main`.

Parse required --request and --output, bound request JSON to 16 MiB and set request_base to the request file parent. Publish the scoped receipt, print it and return zero for PASS or one for failed measured predicates. Invalid/stale evidence raises; successful publication is separate from finding closure.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--request` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## research-acceptance

`python commands.py research-acceptance --help`

Implementation: [commands.py](../commands.py), `run_research_acceptance_main`.

Read a provenance-bearing manifest and explicit evidence, require config/data/
source hash fields, and check the saved run through `acceptance_report`.
Write a fresh acceptance JSON and print its status/path. Return 0 only for
ACCEPTED, otherwise 1. It evaluates retained evidence without rerunning analysis
or overwriting the original manifest and products.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `--manifest` | See live help and implementation. | required=True; type=Path |
| main | `--evidence` | See live help and implementation. | required=True; type=Path |
| main | `--output` | See live help and implementation. | required=True; type=Path |

## resolution-reveal

`python commands.py resolution-reveal --help`

Implementation: [commands.py](../commands.py), `resolution_reveal_main`.

Compare one selected saved projection from each of two run/map bundles.
Require explicit snapshots, common map key, epoch tolerance and memory budget;
publish the reveal display and print the result. It presents the selected
resolution contrast without automatically proving phase matching or numerical
convergence.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_a` | See live help and implementation. | type=Path |
| main | `bundle_a` | See live help and implementation. | type=Path |
| main | `run_b` | See live help and implementation. | type=Path |
| main | `bundle_b` | See live help and implementation. | type=Path |
| main | `--snapshot-a` | See live help and implementation. | type=int; required=True |
| main | `--snapshot-b` | See live help and implementation. | type=int; required=True |
| main | `--map-key` | See live help and implementation. | required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--redshift-tolerance` | See live help and implementation. | type=float; default=1e-06 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## resource-compare

`python commands.py resource-compare --help`

Implementation: [core/resources.py](../core/resources.py), `resource_comparison_main`.

Publish measured/estimated resources without changing run controls.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `plan` | See live help and implementation. | type=Path |
| main | `--receipts` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## restart-comparison

`python commands.py restart-comparison --help`

Implementation: [core/research/restarts.py](../core/research/restarts.py), `restart_comparison_main`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--request` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## results

`python commands.py results --help`

Implementation: [core/run_results.py](../core/run_results.py), `results_main`.

Rebuild saved-run navigation without running analysis or simulation.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `--verify` | Check recorded artifact hashes as well as existence | action='store_true' |

## results-gallery

`python commands.py results-gallery --help`

Implementation: [commands.py](../commands.py), `results_gallery_main`.

Create an offline gallery from an explicitly selected-results JSON, resolving
relative inputs against that request's directory and checking its hash after
publication. Creation requires `--output`; `--verify` instead takes a gallery
directory and rejects an output argument. Print the created location or
verification JSON. This indexes supplied outputs rather than rerunning their
scientific producers.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | Request JSON, or gallery directory when --verify is used | type=Path |
| main | `--output` | See live help and implementation. | type=Path |
| main | `--verify` | Check gallery page and recorded target-file hashes | action='store_true' |

## run

`python commands.py run --help`

Implementation: [core/cli.py](../core/cli.py), `run_pipeline_main`.

Run a fresh user-configured linear cosmology, then hand it to GADGET.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--linear-config` | Your CLASS/AxiCLASS answers JSON; otherwise prompt | type=Path |
| main | `--answers` | Your GADGET answers JSON; cosmology must match the new bundle | type=Path |
| main | `--state-root` | External runtime root for every stage | type=Path |
| main | `--linear-output` | Parent for new named cosmology; defaults to runtime/runs/linear | type=Path |
| main | `--run-dir` | Explicit new particle-run directory | type=Path |
| main | `--binary` | Explicit provenance-bound compiled linear solver | type=Path |
| main | `--draft` | Saved terminal questionnaire; resume with the same path | type=Path |
| main | `--count` | Number of simulations to configure and run sequentially | type=int |

## run-spec

`python commands.py run-spec --help`

Implementation: [core/workflow.py](../core/workflow.py), `run_spec_main`.

Author canonical migrated specifications and automatic product plans.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `mode` | See live help and implementation. | choices=('plan', 'migrate', 'catalogue') |
| main | `request` | See live help and implementation. | type=Path; nargs='?' |
| main | `--explicit` | Required explicit run fields when migrating a v1 science specification | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## scientific-figures

`python commands.py scientific-figures --help`

Implementation: [commands.py](../commands.py), `scientific_figures_main`.

Forward to the registered scientific-figure CLI. Rendering requires a measured
array request, output and memory budget; optional organized layout separates
formats by family. A mutually exclusive registry-export mode only writes the
catalogue and returns success without rendering. The wrapper creates neither
missing measurements nor accepted physical results from a recipe name.

Parser owner: [core/research/display.py](../core/research/display.py#L1345), `scientific_figures_main`. The wrapper forwards to this function.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | nargs='?' |
| main | `--output` | See live help and implementation. |  |
| main | `--memory-gib` | See live help and implementation. | type=float |
| main | `--export-registry` | Write the current variant catalogue without rendering or claiming measurement coverage. | type=Path |
| main | `--organized` | Separate PDF/PNG/SVG by family and retain numeric products under products/scientific_figures. | action='store_true' |

## select-merger-tree

`python commands.py select-merger-tree --help`

Implementation: [commands.py](../commands.py), `select_merger_tree_main`.

Select an exact final halo ID from the saved run and analysis bundle, then
export its descendant-linked ancestry to the required output and print the
result path. Selection uses retained catalogues/history rather than synthesizing
progenitors or treating a catalogue row as a universal object identity.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `analysis_bundle` | See live help and implementation. | type=Path |
| main | `--halo-id` | See live help and implementation. | required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-angular-control

`python commands.py sidm-angular-control --help`

Implementation: [core/sidm.py](../core/sidm.py), `angular_control_main`.

Read required --request JSON and measure saved native scattering angles against the physical reference at a fresh --output. If no numeric publication was produced, write the returned JSON and its SHA seal directly. Normal completion returns None; existing outputs and invalid measurements raise. Paths inside the request are not automatically rebased to the request file by this adapter.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--request` | See live help and implementation. | required=True; type=Path |
| main | `--output` | See live help and implementation. | required=True; type=Path |

## sidm-canonical-population

`python commands.py sidm-canonical-population --help`

Implementation: [core/sidm.py](../core/sidm.py), `canonical_population_main`.

Read positional canonical-population parameters JSON, default request_base to the file parent and materialize exact original expanding SIDM particles as native canonical x,u ICs at required --output. Return zero after publication. This command writes source-bound initial conditions and does not evolve them.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-control-campaign

`python commands.py sidm-control-campaign --help`

Implementation: [core/sidm.py](../core/sidm.py), `control_campaign_main`.

Read the positional request JSON and compile a physical or mapped-expanding native SIDM control/paired-realization workflow into --output, using the request parent as base. Return zero after design publication. The command prepares concrete recipes; it does not itself execute the native controls.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-control-clock-mapping

`python commands.py sidm-control-clock-mapping --help`

Implementation: [core/sidm.py](../core/sidm.py), `control_clock_mapping_main`.

Read positional parameters JSON, supply its parent as request_base when absent and publish the expanding-control canonical clock/law/state mapping to required --output. Return zero after publication and propagate input/measurement errors. No native mapped driver is launched by this command.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-control-comparison

`python commands.py sidm-control-comparison --help`

Implementation: [core/sidm.py](../core/sidm.py), `control_comparison_main`.

Read positional comparison parameters JSON, default request_base to the file parent and publish the actual retained control contrast at required --output. Return zero after publication; the product’s comparison_status remains the numerical decision. This command does not manufacture a reference trajectory or run missing refinements.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-control-ensemble

`python commands.py sidm-control-ensemble --help`

Implementation: [core/sidm.py](../core/sidm.py), `control_ensemble_main`.

Read positional ensemble parameters JSON, default request_base to the file parent and publish measured joint covariance and paired-realization differences from retained native SIDM comparisons at --output. Return zero on publication. Missing or incompatible realizations raise rather than being replaced by invented samples.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-control-expectation

`python commands.py sidm-control-expectation --help`

Implementation: [core/sidm.py](../core/sidm.py), `control_expectation_main`.

Load positional expectation parameters JSON, default request_base to the file parent and publish native beam/thermal/saved-relaxation measurements at required --output. Return zero on publication and retain the producer’s actual status/scope. This CLI reads completed controls; it does not interpolate an unsaved relaxation time or launch a replacement native run.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-control-history

`python commands.py sidm-control-history --help`

Implementation: [core/sidm.py](../core/sidm.py), `control_history_main`.

Load positional parameters JSON, default request_base to the parameter file’s parent and publish measured dedicated SIDM history at required --output. Return zero on completion. The publisher reads the actual completed Gadget4/OpenGadget3 run and retains its physical clock/model/source contract; this CLI does not launch either engine.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-control-native-clock

`python commands.py sidm-control-native-clock --help`

Implementation: [core/sidm.py](../core/sidm.py), `control_native_clock_main`.

Read positional parameters JSON, default request_base to its file parent and publish the source-bound native canonical-clock asset into required --output. Return zero on completion. The CLI creates clock inputs and a report rather than executing or accepting a mapped native experiment.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-control-plan

`python commands.py sidm-control-plan --help`

Implementation: [core/sidm_campaign.py](../core/sidm_campaign.py), `sidm_control_plan_main`.

Author the complete required native control matrix; execute no cases.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-control-population

`python commands.py sidm-control-population --help`

Implementation: [core/sidm.py](../core/sidm.py), `control_population_main`.

Read positional population parameters JSON and publish physical SIDM control populations plus matched native ICs to required --output. Return zero after publication. This adapter does not insert a request_base default and does not launch evolution; explicit paths/settings must satisfy the population producer.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-control-products

`python commands.py sidm-control-products --help`

Implementation: [core/sidm.py](../core/sidm.py), `sidm_control_products_main`.

Select reference-cdf or native-events from the positional kind, read --request JSON and require a fresh --output. A physical CDF reference rejects --native-stage-receipt, while actual event export requires that completed evolution receipt. Invoke the selected producer; normal completion returns None. This CLI keeps independently authored physical reference distributions separate from measured native event records.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `kind` | See live help and implementation. | choices=['reference-cdf', 'native-events'] |
| main | `--request` | See live help and implementation. | required=True; type=Path |
| main | `--output` | See live help and implementation. | required=True; type=Path |
| main | `--native-stage-receipt` | See live help and implementation. | type=Path |

## sidm-control-refinement

`python commands.py sidm-control-refinement --help`

Implementation: [core/sidm.py](../core/sidm.py), `control_refinement_main`.

Read the positional request JSON and compile the independent three-or-more-level native SIDM timestep/kernel/hybrid/probability refinement workflow into --output, resolving assets relative to the request parent. Return zero after compilation. Native evolution and multi-level measured acceptance occur only when the resulting workflow is executed.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-model

`python commands.py sidm-model --help`

Implementation: [core/sidm_physics.py](../core/sidm_physics.py), `sidm_model_main`.

Materialize a declared differential model, without native execution.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `model` | See live help and implementation. | type=Path |
| main | `--table` | See live help and implementation. | type=Path; required=True |
| main | `--grid` | JSON containing explicit speed_kms and cos_theta arrays for analytic models | type=Path |
| main | `--output` | Immutable model/table/source descriptor | type=Path; required=True |

## sidm-reference-build

`python commands.py sidm-reference-build --help`

Implementation: [core/sidm.py](../core/sidm.py), `reference_build_main`.

Read positional parameters and require a fresh --output for the pinned independent OpenGadget3 reference build. Under a native run lock, set request_base/destination and a context carrying the declared jobs, OpenMP, memory and disk limits, then invoke reference_build_runner. Return zero after the build operation. This is an explicit compilation entry point; completed build/invocation/capability receipts, not this command name, establish which executable was produced.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-refinement-measurement

`python commands.py sidm-refinement-measurement --help`

Implementation: [core/sidm.py](../core/sidm.py), `control_refinement_measurement_main`.

Read positional refinement parameters JSON, set request_base to the file parent when absent and publish the measured native multi-level paired SIDM refinement product to --output. Return zero on publication. This adapter reduces existing ensembles; it does not execute the refinement simulations.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## sidm-response-campaign

`python commands.py sidm-response-campaign --help`

Implementation: [core/sidm.py](../core/sidm.py), `joint_response_campaign_main`.

Read the positional request JSON and compile a complete native joint-response/held-out workflow into --output using the request parent as asset base. Only --execute passes the generated workflow to execute_workflow under output/execution. Return zero on completion. Preparing the design alone launches no native runs; actual execution and calibration receipts determine scientific support.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--execute` | Execute the generated production workflow; design generation alone launches nothing | action='store_true' |

## sidm-science

`python commands.py sidm-science --help`

Implementation: [core/sidm.py](../core/sidm.py), `sidm_science_main`.

Select one of ten science modes from a positional mode/request JSON and required --output, resolving request assets relative to that file. Membership, satellite, history and forward_vector also require both native-stage-receipt and native-run-directory. Call the selected publisher, write a sibling .ref.json with its SHA-256 and return zero. Import/run through the documented core.sidm_analysis logical namespace or public commands; successful publication retains each product’s support and acceptance status.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `mode` | See live help and implementation. | choices=('membership', 'satellite', 'history', 'matched', 'observations', 'derived', 'forward_vector', 'joint_response', 'joint_calibration', 'calibrated_response') |
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--native-stage-receipt` | See live help and implementation. | type=Path |
| main | `--native-run-directory` | See live help and implementation. | type=Path |

## stage-zoom-inputs

`python commands.py stage-zoom-inputs --help`

Implementation: [commands.py](../commands.py), `stage_zoom_inputs_main`.

Load a JSON staging request and delegate compatible species-transfer/MUSIC
input staging. `--resolution-refinement` selects bounded refinement staging
from an audited parent and saved-aperture budget. Print the resulting staged
product record. These request-driven input products are distinct from the
native zoom-generation command and its subsequent physical acceptance checks.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--resolution-refinement` | Stage a bounded higher-resolution child from an audited parent and saved-aperture budget | action='store_true' |

## start-epoch-campaign

`python commands.py start-epoch-campaign --help`

Implementation: [commands.py](../commands.py), `run_start_epoch_campaign_main`.

Dispatch explicit start-epoch campaign build, verification, branch staging,
branch execution, phase audit, saved measurement, density-mode/analysis-product
measurement or comparison operations. Requests bind branch IDs, outputs,
executions and IC phase evidence; execute mode passes the supplied command
argument list to the owned branch runner. Print status and return 1 for FAIL,
2 for INCOMPLETE, otherwise 0. A prepared campaign does not imply its branches
have executed or converged.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| build | `--spec` | See live help and implementation. | type=Path; required=True |
| build | `--output` | See live help and implementation. | type=Path; required=True |
| build | `--resume` | See live help and implementation. | action='store_true' |
| verify | `--campaign` | See live help and implementation. | type=Path; required=True |
| stage | `--campaign` | See live help and implementation. | type=Path; required=True |
| stage | `--branch` | See live help and implementation. | required=True |
| stage | `--output` | See live help and implementation. | type=Path; required=True |
| stage | `--wdm-request` | See live help and implementation. | type=Path |
| execute | `--campaign` | See live help and implementation. | type=Path; required=True |
| execute | `--branch` | See live help and implementation. | required=True |
| execute | `--run-directory` | See live help and implementation. | type=Path; required=True |
| execute | `--kind` | See live help and implementation. | choices=('evolution', 'ic_generation', 'numerical_control'); required=True |
| execute | `command` | See live help and implementation. | nargs=argparse.REMAINDER |
| phases | `--campaign` | See live help and implementation. | type=Path; required=True |
| phases | `--request` | See live help and implementation. | type=Path; required=True |
| phases | `--output` | See live help and implementation. | type=Path; required=True |
| measure | `--campaign` | See live help and implementation. | type=Path; required=True |
| measure | `--request` | See live help and implementation. | type=Path; required=True |
| measure | `--output` | See live help and implementation. | type=Path; required=True |
| measure-density-modes | `--campaign` | See live help and implementation. | type=Path; required=True |
| measure-density-modes | `--request` | See live help and implementation. | type=Path; required=True |
| measure-density-modes | `--output` | See live help and implementation. | type=Path; required=True |
| measure-analysis | `--campaign` | See live help and implementation. | type=Path; required=True |
| measure-analysis | `--request` | See live help and implementation. | type=Path; required=True |
| measure-analysis | `--output` | See live help and implementation. | type=Path; required=True |
| compare | `--campaign` | See live help and implementation. | type=Path; required=True |
| compare | `--measurements` | See live help and implementation. | nargs='+'; type=Path; required=True |
| compare | `--output` | See live help and implementation. | type=Path; required=True |
| compare | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## stellar-stream-movie

`python commands.py stellar-stream-movie --help`

Implementation: [commands.py](../commands.py), `stellar_stream_movie_main`.

Animate an existing exact-ID stellar candidate history from its run/bundle.
Pass optional fixed centre and width in comoving kpc, particle bound, frame rate
and seconds per saved snapshot; print the published output. This encodes saved
candidate trajectories, without establishing tidal origin from appearance.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run` | See live help and implementation. | type=Path |
| main | `history_bundle` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--center` | Fixed viewing center in comoving kpc | type=float; nargs=3 |
| main | `--width-kpc` | Comoving display width, at most the simulation box | type=float |
| main | `--max-particles` | See live help and implementation. | type=int; default=50000 |
| main | `--fps` | See live help and implementation. | type=int; default=24 |
| main | `--seconds-per-snapshot` | See live help and implementation. | type=int; default=1 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## stream-products

`python commands.py stream-products --help`

Implementation: [core/cli.py](../core/cli.py), `stream_products_main`.

Read a measured stream product and select either figure export or Blender
inspector-asset export, with explicit output and memory. Invoke the corresponding
producer and print its JSON-safe result. Inspector assets are data for a later
scene loader; this command does not fit a new stream or silently launch native
evolution or a Blender render.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `kind` | See live help and implementation. | choices=('figures', 'inspector') |
| main | `product` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--memory-gib` | See live help and implementation. | type=float; required=True |

## streams

`python commands.py streams --help`

Implementation: [commands.py](../commands.py), `streams_main`.

Forward to the stream-workflow CLI, which loads a request, resolves inputs
relative to that request and runs its selected catalogue/stripping/geometry/
action or inference stages into the required output. Return the delegate's
result and let it print finite JSON. Which scientific work executes depends
on the explicit stage graph and its original input/acceptance contracts.

Parser owner: [core/stream/workflow.py](../core/stream/workflow.py#L510), `streams_main`. The wrapper forwards to this function.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | required=True; type=Path |

## timeline-wall

`python commands.py timeline-wall --help`

Implementation: [commands.py](../commands.py), `timeline_wall_main`.

Build a static fixed-scale cosmic timeline from one exact map key in a saved
bundle, with an explicit panel cap, memory allocation and optional source run.
Publish and print the output. It lays out recorded epochs rather than adding
interpolated simulations or changing each panel's physical scale independently.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `bundle` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--map-key` | Exact key from scale.json | required=True |
| main | `--max-panels` | See live help and implementation. | type=int; default=6 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |
| main | `--run` | Source run if it cannot be inferred from map provenance | type=Path |

## trace-zoom

`python commands.py trace-zoom --help`

Implementation: [commands.py](../commands.py), `trace_zoom_main`.

Read keyword arguments from JSON and trace a saved DM aperture back to parent
IC identities. With `--catalogue-host`, select the actual catalogue-host tracing
route instead of an arbitrary aperture. Print the delegated trace product/report;
selection and output locations come from the request. Tracing does not itself
establish a delivered refined realization or converged zoom evolution.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--catalogue-host` | Select actual native FOF host/galaxy identity rather than an arbitrary centre | action='store_true' |

## twofluid-eulerian-seed

`python commands.py twofluid-eulerian-seed --help`

Implementation: [core/initial_conditions.py](../core/initial_conditions.py), `twofluid_eulerian_seed_main`.

Read positional parameters JSON and required --output, invoke the Eulerian seed producer and return zero. The command measures original full-box gas/DM cell averages; it does not insert request_base or automatically rebase embedded paths beside the parameters file.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## twofluid-native-comparison

`python commands.py twofluid-native-comparison --help`

Implementation: [core/initial_conditions.py](../core/initial_conditions.py), `twofluid_native_comparison_main`.

Read positional parameters JSON, default request_base to its parent, publish the original evolved-native-versus-nonlinear-reference comparison to required --output and return zero. Explicit base and source references remain authoritative; the command does not replace unavailable native output with generated fixtures.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## twofluid-native-family

`python commands.py twofluid-native-family --help`

Implementation: [core/initial_conditions.py](../core/initial_conditions.py), `twofluid_native_family_main`.

Read positional parameters JSON and required --output, default request_base to the file parent and execute the native/reference grid-family producer. Return zero. The request selects which original family evidence is measured; this entry point does not itself launch a fresh native parameter sweep.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## twofluid-native-ic

`python commands.py twofluid-native-ic --help`

Implementation: [core/initial_conditions.py](../core/initial_conditions.py), `twofluid_native_ic_main`.

Parse required --parameters JSON and --output, default request_base to the parameters file’s parent and invoke the native two-fluid IC writer. Normal completion returns None. Explicit request_base is preserved; the producer owns original projection/generation and thermochemical-state checks.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--parameters` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## twofluid-nonlinear

`python commands.py twofluid-nonlinear --help`

Implementation: [core/initial_conditions.py](../core/initial_conditions.py), `twofluid_nonlinear_main`.

Read positional parameters JSON and required --output, default request_base to the parameters parent and execute the conservative nonlinear two-fluid grid producer. Return zero. The retained seed and request controls determine its scoped physical model; the CLI does not invent a native comparison or convergence result.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## twofluid-particles

`python commands.py twofluid-particles --help`

Implementation: [core/initial_conditions.py](../core/initial_conditions.py), `twofluid_particle_main`.

Parse required --parameters JSON and --output, default request_base to the parameters parent and publish native-q/ID projection using the request’s uniform-point or finite-volume material-cell mode. Normal completion returns None. This preserves actual native load identity rather than substituting an arbitrary particle sample.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--parameters` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## twofluid-projection-refinement

`python commands.py twofluid-projection-refinement --help`

Implementation: [core/initial_conditions.py](../core/initial_conditions.py), `twofluid_projection_refinement_main`.

Read positional parameters JSON and required --output, default request_base to the file parent and measure finite-volume native projection refinement. The request selects exact-ID, nested hierarchy or retained-flow overlap semantics; return zero after publication. This comparison does not independently certify the underlying species-flow model.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## twofluid-refinement

`python commands.py twofluid-refinement --help`

Implementation: [core/initial_conditions.py](../core/initial_conditions.py), `twofluid_refinement_main`.

Read positional parameters JSON and required --output, default request_base beside the file and execute conservative adjacent nonlinear two-fluid grid refinement measurement. Return zero. This compares declared retained levels rather than constructing a new independently validated physical seed.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## twofluid-refinement-campaign

`python commands.py twofluid-refinement-campaign --help`

Implementation: [core/initial_conditions.py](../core/initial_conditions.py), `twofluid_refinement_campaign_main`.

Read positional campaign request JSON and required --output, then compile with request_base fixed to the request file’s parent. Return zero after recipe publication. The compiler supports declared nonlinear grid, operative timestep, dissipation or supplied native hierarchy comparisons; this CLI prepares their workflow rather than executing it.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `request` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## twofluid-relative

`python commands.py twofluid-relative --help`

Implementation: [core/initial_conditions.py](../core/initial_conditions.py), `twofluid_relative_main`.

Read positional parameters JSON, default request_base to that file’s parent, publish finite-epoch two-fluid relative-mode grids to required --output and return zero. Explicit request_base is retained. This runs the declared perturbative grid producer rather than a native particle simulation.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `parameters` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## twofluid-window

`python commands.py twofluid-window --help`

Implementation: [core/initial_conditions.py](../core/initial_conditions.py), `twofluid_window_main`.

Parse required --parameters JSON and --output, default request_base to the parameters file parent and invoke the material-window comparison producer. Normal completion returns None. The comparison uses actual delivered particles and identically windowed retained flow integrals, rather than substituting an unwindowed theory curve.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--parameters` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |

## velociraptor

`python commands.py velociraptor --help`

Implementation: [commands.py](../commands.py), `run_velociraptor_main`.

Read a saved-run JSON request and execute the pinned native VELOCIraptor
finder at the required destination, passing the producer's exact-ID transport
settings. Print status and resulting object count. This launches a finder on
existing particles; successful execution can legitimately yield an empty catalogue.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `request` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path |

## velocity-flow

`python commands.py velocity-flow --help`

Implementation: [commands.py](../commands.py), `velocity_flow_main`.

Map one saved DM snapshot's density and mass-weighted peculiar-velocity flow
in the selected xy/xz/yz projection, applying grid, particle-support and optional
slice controls. Publish at `--output` and print the result. It measures saved
velocities; displayed arrows do not integrate future particle motion.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--snapshot` | See live help and implementation. | type=int; required=True |
| main | `--projection` | See live help and implementation. | choices=['xy', 'xz', 'yz']; default='xy' |
| main | `--grid` | See live help and implementation. | type=int; default=32 |
| main | `--minimum-particles` | See live help and implementation. | type=int; default=3 |
| main | `--slice-center` | See live help and implementation. | type=float |
| main | `--slice-thickness` | See live help and implementation. | type=float |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## verify

`python commands.py verify --help`

Implementation: [commands.py](../commands.py), `verify_main`.

Forward to the verification dispatcher. It runs unittest selections or explicit
retained domain groups, with group mode allowing only the supported verbosity
switches, and exits according to test success. Suites can use fixtures, mocks
and selected native controls; a passing test count does not imply every backend,
branch or scientific regime executed.

Parser owner: [verification.py](../verification.py#L12087), `main`. The wrapper forwards to this function.

Verification delegates test selection to `unittest.main`. Use `python commands.py verify --help` for standard test-name, verbosity, fail-fast and discovery options. The additional `--group NAME` switch selects a retained domain group; repeat it to combine groups. With `--group`, only `-v`/`--verbose` or `-q`/`--quiet` may accompany the groups. See `TEST_GROUPS` in the parser-owner source for the available names.

## void-boundary-history

`python commands.py void-boundary-history --help`

Implementation: [commands.py](../commands.py), `void_boundary_history_main`.

Use a saved run and void bundle to measure periodic density-excursion boundaries
and adjacent-epoch mesh overlap within the memory allocation. Publish the history
output and print its result. This quantifies Eulerian region boundaries/overlap,
which is distinct from exact particle-cohort history or a proven void genealogy.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `void_bundle` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## void-explorer

`python commands.py void-explorer --help`

Implementation: [commands.py](../commands.py), `void_explorer_main`.

Read saved periodic DM epochs and build the requested underdensity explorer
using explicit mesh/smoothing/threshold/matching and sampling/movie budgets.
Print the published output. The producer can create diagnostic films; threshold
regions and heuristic matches do not automatically establish continuous physical
void identity or a new cosmological evolution.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--grid` | See live help and implementation. | type=int; default=48 |
| main | `--smoothing-cells` | See live help and implementation. | type=float; default=1.5 |
| main | `--threshold` | See live help and implementation. | type=float; default=0.8 |
| main | `--match-fraction` | See live help and implementation. | type=float; default=0.125 |
| main | `--frames` | See live help and implementation. | type=int; default=32 |
| main | `--fps` | See live help and implementation. | type=int; default=16 |
| main | `--max-particles` | See live help and implementation. | type=int; default=100000 |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## void-particle-history

`python commands.py void-particle-history --help`

Implementation: [commands.py](../commands.py), `void_particle_history_main`.

Measure exact DM-particle overlap between saved void-centred apertures using
the run, void bundle, explicit radius in Mpc/h and memory budget. Publish the
requested history output and print its result. Aperture overlap is a measured
material diagnostic, not by itself persistent physical void identity.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `run_dir` | See live help and implementation. | type=Path |
| main | `void_bundle` | See live help and implementation. | type=Path |
| main | `--radius-mpc-h` | See live help and implementation. | type=float; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--memory-gib` | See live help and implementation. | type=float; default=1.0 |

## workflow

`python commands.py workflow --help`

Implementation: [core/workflow.py](../core/workflow.py), `workflow_main`.

Parse exactly one --answers or --campaign request plus required --output, optional --resume and --dry-run. Read JSON, resolve relative inputs using the request’s parent and invoke execute_workflow. Print the result; return zero for dry-run or COMPLETE execution and one for an incomplete returned workflow. Uncaught validation/execution failures propagate rather than being converted into successful status.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--answers` | See live help and implementation. | type=Path |
| main | `--campaign` | See live help and implementation. | type=Path |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--resume` | See live help and implementation. | action='store_true' |
| main | `--dry-run` | See live help and implementation. | action='store_true' |

## workflow-products

`python commands.py workflow-products --help`

Implementation: [core/workflow.py](../core/workflow.py), `workflow_products_main`.

Read a retained workflow through load_workflow_stage_results and print per-stage execution state and product manifest IDs, paths, hashes and schema versions. Direct readers to each product’s current domain/acceptance receipts. This CLI exposes existing scientific products without rerunning the workflow, granting acceptance or producing new figures.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `workflow` | See live help and implementation. | type=Path |

## zoom

`python commands.py zoom --help`

Implementation: [commands.py](../commands.py), `run_zoom_main`.

Run the supplied MUSIC binary on the selected staged configuration and publish
its structurally audited zoom ICs. Require a native-validation input and either
both parent IC/noise paths or explicit `--structural-only`; separate audit/native
memory, disk and wall budgets remain visible. Print the generation report.
This launches real IC generation, but structural-only operation and incomplete
phase/physical gates never confer scientific acceptance.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `configuration` | See live help and implementation. | type=Path |
| main | `destination` | See live help and implementation. | type=Path |
| main | `--binary` | See live help and implementation. | type=Path; required=True |
| main | `--native-validation` | See live help and implementation. | type=Path; required=True |
| main | `--traced-region` | See live help and implementation. | type=Path |
| main | `--threads` | See live help and implementation. | type=int; default=2 |
| main | `--audit-memory-gib` | See live help and implementation. | type=float; default=1 |
| main | `--native-memory-gib` | Native mesh working-set estimate budget, separate from Python audit RAM | type=float; default=4 |
| main | `--native-disk-gib` | Cap native output/cache estimate; default checks available disk | type=float |
| main | `--native-wall-hours` | Stop native generation at this wall-time budget; partial ICs fail | type=float |
| main | `--parent-ic` | Compatible saved parent MUSIC IC for required phase matching | type=Path |
| main | `--parent-noise` | Saved parent coarse white-noise file | type=Path |
| main | `--structural-only` | Explicitly allow missing parent phases; never grants science acceptance | action='store_true' |

## zoom-campaign

`python commands.py zoom-campaign --help`

Implementation: [core/research/zooms.py](../core/research/zooms.py), `zoom_campaign_main`.

| Parser / subcommand | Argument | Meaning | Parser constraints |
|---|---|---|---|
| main | `--campaign` | See live help and implementation. | type=Path; required=True |
| main | `--output` | See live help and implementation. | type=Path; required=True |
| main | `--resume` | See live help and implementation. | action='store_true' |
