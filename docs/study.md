# Bounded study driver

`python commands.py study /absolute/path/study.json --dry-run` prints a read-only
plan. `python commands.py study /absolute/path/study.json` then runs the
implemented stages in order. This interface is new and has only been statically
inspected; do not treat a plan or a successful process as scientific validation.
Existing standalone commands remain available.

Create a study JSON outside the source tree, for example:

```json
{
  "schema": "research2-study-v1",
  "answers": "/scratch/my-study/gadget-answers.json",
  "run_dir": "/scratch/my-study/native",
  "state_dir": "/scratch/my-study/state",
  "analysis": {
    "graphs": ["radial_shapes"],
    "requests": {"radial_shapes": "/scratch/my-study/radial-shapes.json"},
    "find_halos": false
  },
  "visualization": {"max_points": 100000, "quality": "preview", "frames_per_snapshot": 8},
  "requested_integrations": []
}
```

The answers file uses the existing `gadget --answers` contract, including its
`cosmology` path to a completed spectrum manifest (unless `linear` below is selected) and all simulation choices,
seeds, physics, output epochs and resource settings. Relative paths in that file
are resolved by the existing command from the repository directory; use absolute
paths. Relative paths in the study file are relative to the study file. The
analysis request is optional and accepts explicit graph names and the same JSON
requests as `commands.py analysis`; it cannot predict whether a saved run contains
the necessary stars, catalogues, or other fields. Choose a different output or
omit the analysis stage when its measured prerequisites are not yet available.
The scene stage only exports a portable bundle and requires at least two snapshots;
it does not build or render a Blender scene.

## Starting with a linear cosmology

Instead of providing a completed spectrum, add a `linear` object to the study:

```json
"linear": {"config": "/scratch/my-study/linear.json", "binary": "/path/to/class"}
```

The `binary` key is optional; when omitted the existing `class` command resolves
the configured native installation. The linear JSON uses the existing
`commands.py class --config` contract, for example the following illustrative
flat ΛCDM settings (choose and justify settings for the study):

```json
{
  "name": "initial-cosmology", "model": "lcdm", "H0": 67.4,
  "Omega_b": 0.049, "Omega_cdm": 0.265, "A_s": 2.1e-9,
  "n_s": 0.965, "tau_reio": 0.054, "z_start": 99,
  "k_max_h_mpc": 100
}
```

Omit `cosmology` from the GADGET answers file, or set it to the exact generated
`STATE/linear/NAME/spectrum.json` (or its parent). The driver derives a new
`STATE/gadget-answers.json`, preserving every other answer, and runs CLASS or
AxiCLASS before GADGET. The original answers are never rewritten. A conflicting
authored cosmology is rejected. The solver's sealed bundle and generated answers
are checked before reuse; an incomplete linear run is preserved and is not
automatically retried. Choosing a new state directory is required after inspecting
and preserving a failed linear attempt. The underlying solver checks its own
settings, executable provenance and physical coverage during execution; the dry
run checks only the model and safe output name and cannot certify these checks.
The `linear` stage and its GADGET handoff have not been run or validated.

## Selecting a parent target

Add a `target_selection` object to select FOF candidates from the actual
completed native run. For example, the following chooses the most populated
host at snapshot 8 with at least 1,000 DM particles:

```json
"target_selection": {
  "snapshot": 8,
  "memory_gib": 4,
  "minimum_dm_particles": 1000,
  "maximum_targets": 1,
  "ranking": "dm_count_descending",
  "host_rows": []
}
```

The required values are `snapshot` and `memory_gib`. Optional
`maximum_dm_particles` sets an upper count, and `host_rows` explicitly
prioritizes one or more epoch-local catalogue rows. Override count cannot exceed
`maximum_targets`; overrides outside the automatic count rules remain marked.
The count floor is a selection choice, not a convergence criterion. A complete,
compatible group-ordered FOF catalogue and snapshot must actually exist. When
`analysis.find_halos` is selected, selection follows that stage; otherwise it
follows evolution and does not itself run a halo finder.

The dry run reports this stage but does not read a future catalogue or select a
target. During execution the study saves `STATE/target-selection.json`, a
source-hashed native selection record, and preserves each selector stdout and stderr log. Reuse checks the
selection bytes, native source hashes and original settings. Selection alone
does not launch MUSIC or prove parent phases, target recovery, or contamination
acceptance; add the explicit `zoom` configuration below to request the campaign. To bind this preview to a separately prepared zoom campaign, use its
`selection_preview` contract with this file path and its exact SHA-256; the
campaign checks the preview against a fresh selection before proceeding. A failed selection can be retried with `--retry-postprocessing` after
inspecting the retained evidence; an existing differing selection is never
silently overwritten. This new path has not been run or scientifically validated.

## Running a physical zoom in the study

A study can invoke the existing physical MUSIC zoom campaign after selecting a
parent. This route currently requires an **already generated native MUSIC parent
IC and generation report**, and its actual coarse parent-noise file. It does not
turn an ordinary monofonIC parent into a phase-matched MUSIC parent or generate
the parent MUSIC IC from the linear stage. The GADGET answers must select a
supported native MUSIC handoff; the existing IC and generator validation gates
remain authoritative.

Add these objects to the study alongside `target_selection`:

```json
"parent_ic": {
  "path": "/scratch/parent-generation/ics_gadget.hdf5",
  "sha256": "REPLACE_WITH_EXACT_64_CHARACTER_SHA256",
  "generation_report": "/scratch/parent-generation/generation.json",
  "generation_report_sha256": "REPLACE_WITH_EXACT_64_CHARACTER_SHA256"
},
"zoom": {"template": "/scratch/my-study/zoom-template.json"}
```

Create the zoom template from the complete
[zoom campaign example](../configs/research/block40_zoom_campaign.example.json).
Set the following fields to the literal string `${study}`: `parent.run_dir`,
`parent.initial_ic`, `parent.source_manifest`, `parent.spectrum_manifest`, and
those last three names in `parent.sha256`. Omit `selection_preview` or set it to
`${study}`. Set `parent.snapshot`, `selection_rules` and `explicit_host_rows`
to exactly match the study target selection, including its explicit count and
ranking defaults. Set `parent.noise` and `parent.sha256.noise` to the actual
coarse parent-noise file and its predeclared digest. Paths in the template are
relative to the template's directory, even though the derived request is saved
under the study state directory. Fill all remaining campaign settings, native
MUSIC executable/control and zoom-answer pins, seeds, physics, resources,
refinement variants, diagnostic inputs and acceptance requirements explicitly;
the driver does not invent or relax them. Recipe and per-level phase-noise
input paths and hashes are checked where supplied. Use `sha256sum PATH` on Linux
or `shasum -a 256 PATH` on macOS to record asset hashes.

The dry run shows the planned native evolution, target selection and zoom
command; it can check the explicit fixed asset pins, but cannot select a target
or check a future parent IC handoff. During execution the study verifies the
completed parent manifest's single IC against the supplied IC hash, then writes
`STATE/zoom-campaign.request.json` with the actual IC, source manifest, spectrum
and selection paths and hashes. The campaign rechecks the preview against the
live catalogue, traces parent material, stages and evolves the explicit
refinement variants, and measures their declared adequacy. Outputs are in
`STATE/zoom/`; execution completion and region adequacy remain separate from
scientific acceptance. The study seals the request, campaign outputs and
campaign input bindings for conservative reuse.

If a zoom fails or is interrupted, inspect its study log and campaign evidence.
Reissue the study command with `--resume-zoom` to invoke the campaign's own
source-bound resume; failed attempt directories are retained. A nonempty zoom
directory without its campaign receipt is not adopted automatically. A parent
or source mismatch still requires a new study; this option does not rerun
completed parent evolution. This integration, including its resume, has only
been statically inspected and has not been run.

## Independent Blacklight imagery

An optional `blacklight` object requests the explicitly sourced Blacklight bridge.
Install and prepare its pinned checkout, executable, input and physical provenance
as described in [Blacklight imaging](blacklight.md), then add:

```json
"blacklight": {
  "source": "/scratch/research2-runtime/blacklight",
  "binary": "/scratch/research2-runtime/blacklight/bin/blacklight",
  "binary_sha256": "REPLACE_WITH_EXACT_64_CHARACTER_SHA256",
  "input": "/scratch/blacklight-science/input.txt",
  "provenance": "/scratch/blacklight-science/provenance.json",
  "memory_gib": 8,
  "wall_seconds": 3600,
  "display_max": 1e-9,
  "display_stretch": 10,
  "palette": "ember"
}
```

The scientific inputs are a separately supplied GRMHD snapshot or an explicitly
labeled analytic illustration. The study does not infer horizon-scale plasma
from GADGET or imply a physical connection to its large-scale BH. The plan
records the exact bridge command and input hashes; the bridge enforces its source
revision, executable hash, physical-input declarations and resource bounds. It
records the raw intensity, display mapping, image and provenance in
`STATE/blacklight/attempt_001/`; the bridge's science status stays `NOT_ACCEPTED`.
An external binary hash and source checkout do not prove that binary was built
from that source. Building a Blender scene from the image remains a separate
explicit step in the Blacklight guide.

A failed or interrupted calculation is retained. After inspection, use
`--retry-blacklight` to launch a fresh numbered attempt; it does not overwrite
the failed output. Completed outputs and recorded inputs are checked on reuse.
This stage and its retry have not been executed or validated. Stages are run in
order, so a preceding failure stops the study before this independent image stage.

The plan reports exact argument vectors, inputs, expected products, selected
answers, installed Python package versions, tool paths, source and input hashes,
and cost caveats. It does not validate the full native configuration or linked
spectrum payloads, estimate measured resource consumption, or check scientific
acceptance. The dry run cannot select a target before catalogue data exist; the
optional execution stage has its own native catalogue requirements. Native CLI preflights remain authoritative. Put names
of additional desired integrations in `requested_integrations`; unconfigured
integrations are reported as unsupported and block execution instead of being
silently omitted. The current
driver can invoke the explicitly configured physical zoom campaign described
above; it does not automatically connect stream models or inference. Use their standalone documented workflows with explicit data.

The native run and state directories must be outside the source tree and separate from each other. It retains a
locked study record, exact command lines, logs, timestamps and per-stage status.
Successful evolution is sealed by hashing the run's configuration, manifest,
input and data trees; it also seals the analysis results and scene output directories,
plus absolute paths named in the analysis artifact record. Other unlisted external
artifacts are not sealed. This can take substantial time for large runs. Re-invocation
checks every originally sealed file before reusing evolution. A completed analysis
also seals the entire resulting native input/data tree, including files added by
catalogue finding; only that exact post-analysis tree is accepted on reuse. Changing
or replacing an original sealed file or adding unsealed files stops reuse. If halo
finding was interrupted after publishing files, preserve and inspect its outputs;
the driver refuses automatic recovery until the native tree is safely restored or
the study is started in a new run directory. Changed source, inputs, Python package
versions, or original native evidence also stop reuse. This is a conservative integrity
check, not a proof of scientific correctness. Source hashes cover Python files only; C/C++ sources, native executables, linked spectrum payloads, physical tables and system libraries are not fully inventoried. Output products have their own workflow records
and provenance where provided by their underlying commands.

A failed evolution is never automatically launched again. Inspect the native
manifest and logs; the standalone `gadget --resume --run-dir RUN` has its own
checkpoint validation, but this study driver cannot currently adopt a manually
resumed evolution. For a failed postprocessing stage, inspect its log and output
first; `--retry-postprocessing` reruns it without rerunning completed evolution.
Some specialized output destinations may need a fresh request. If a partial scene
folder exists, move it to a safe location before retrying. Existing study records
and original logs are retained. The study always records `NOT_ACCEPTED` for science
until separate scientific validation is performed.

For a future cluster smoke run, install the dependencies and compiled engines as
described in the [operator guide](operator-guide.md), prepare a completed spectrum or a linear configuration,
a bounded answers file and scratch directories, then run the dry-run command on
the actual target environment. Review its arguments, memory and wall-time
settings, spectrum and asset paths, MPI setup, quotas and outputs. Submit the same
command without `--dry-run` through the site's scheduler when validation is
authorized; preserve the scheduler log and `state/study.json`. This guide does
not claim the cluster run has been performed.

### Future scheduler smoke procedure

After validation is authorized, on the target cluster select a small, fresh
scratch location and prepare the completed spectrum and native answers using
[operator-guide.md](operator-guide.md#start-a-fresh-connected-run). Supply a
study JSON as above with paths valid on the compute node. The exact allocation,
modules, MPI launcher and resource values must match the cluster and native
answers; the following is a submission template, not a measured sizing claim:

```sh
cd /path/to/research2
source /path/to/research2-runtime/environments/default/bin/activate
python commands.py study /scratch/my-study/study.json --dry-run > /scratch/my-study/plan.json
# Review plan.json and site resource settings before submitting.
cat > /scratch/my-study/smoke.slurm <<'SLURM'
#!/bin/bash
#SBATCH --job-name=research2-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/my-study/slurm-%j.out
set -euo pipefail
cd /path/to/research2
source /path/to/research2-runtime/environments/default/bin/activate
python commands.py study /scratch/my-study/study.json
SLURM
# Edit paths, modules, launcher and allocations for the target site first.
sbatch /scratch/my-study/smoke.slurm
```

The sample scheduler values are placeholders, not resource guarantees; nested
MPI settings must agree with the site's scheduler policy. Preserve the scheduler
log, study record, native manifest, each stage log, analysis receipts and scene
bundle. Check selected output epochs and particle IDs, configuration and actual
products, then deliberately exercise postprocessing recovery on an expendable
failed or interrupted stage. Verify the Blender bundle on a machine with Blender;
the study scene stage does not create a `.blend` file. A smoke result does not
establish convergence, calibration, physical budgets or production resource use.
