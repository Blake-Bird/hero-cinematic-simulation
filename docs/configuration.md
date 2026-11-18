# Configuration and saved answers

The connected run uses a numbered terminal walkthrough:

```bash
python commands.py run --count 1 --draft /absolute/path/to/setup.json
```

It covers cosmology, physical models, resources, saved epochs, output products, optional
requests, and launch review. Numerical questions show units and bounds; integer fields
retain exact precision.

## Keep the configuration files distinct

| File | What it stores | How to use it |
|---|---|---|
| Draft named by `--draft` | Numbered questionnaire responses and phase state | Repeat the same command to continue collecting/reviewing answers |
| User-authored cosmology/answers JSON | Settings in the command's input schema | Pass through the corresponding `--config`, `--linear-config` or `--answers` interface; see live help |
| `RUN/config.yaml` | The selected run's solver, physics, resources and output settings | Inspect the frozen run; a changed file is not a compatible checkpoint continuation |
| `RUN/manifest.json` | Input, build and source records used by that run | Follow the actual source bindings when validating or analysing it |
| Product request JSON | A specific host/particle/aperture/observation selection and its controls | Supply it to the named analysis or campaign command |

A questionnaire draft is not a simulation checkpoint. A request template is not
ready data: replace its illustrative paths, selections and physical settings.
For a first run, use the walkthrough; for exact schema/argument details use the
[command reference](command-reference.md) and [operator guide](operator-guide.md).

## Save, edit, and continue

Responses are saved immediately. These commands are available at questionnaire prompts:

| Command | Action |
|---|---|
| `:help` | Explain how to answer and edit the questionnaire |
| `:list` | Show saved answers and their numbers |
| `:back` | Revisit the previous answer |
| `:edit N` | Revisit answer N and collect subsequent answers again |
| `:quit` | Save the draft and stop configuration |

Editing discards later answers so a changed physical branch cannot inherit stale
settings. Only input collection is replayed; replay does not restart a solver. Each
settings review requires a fresh `yes`.

Repeat the command with the same draft path to continue. A completed linear bundle is
reused only when the configuration, solver checksum, and output root match and the saved
bundle passes its integrity checks. To continue particle setup against an existing
cosmology bundle, use the public command:

```sh
python commands.py gadget --cosmology /absolute/path/to/linear/spectrum.json --draft /absolute/path/to/particle-setup.json
```

Use a separate draft for each unrelated run. Continuing an existing native simulation
uses its checkpoint route, described in [troubleshooting](troubleshooting.md), rather
than the setup questionnaire.

## Saved epochs and highlights

| Schedule | Meaning |
|---|---|
| `10, 5*, 2, 1*, 0*` | Save all five epochs; highlight 5, 1, and 0 |
| `1:0:0.1` | Save the inclusive range from 1 to 0 in steps of 0.1 |

Ranges require a positive step and must reach the endpoint exactly. You can select
additional highlights after entering a range. Every highlight must belong to the saved
schedule. Redshifts must lie between zero and the IC starting redshift, with at most
1,100 saved epochs.

An answers JSON can include `"highlight_redshifts": [5, 1, 0]`. The frozen configuration
stores `simulation.highlight_redshifts`; older configurations default to no highlights.
Highlighting changes navigation, not native output times, analysis scope, or movie
coverage.

## Select outputs and optional tasks

| Selection | Behavior |
|---|---|
| `available` | Run saved-data products with ready inputs and record omissions |
| `all` | Request every physically applicable product and require its additional inputs |
| Explicit names | Request those products; unavailable selections are errors |

Setup lists the products and asks for JSON requests where measurements need host
selections, particle IDs, apertures, or supplied samples. Enter `later` to explicitly
defer such a task. Ready products can then run, while the deferred task remains pending
and the workflow remains `INCOMPLETE`.

After making the saved-data selections, run the deferred analysis with its request file.
Requests that have not been explicitly deferred must satisfy preflight. See
[outputs](outputs.md) for the product catalogue.

## Launch review and other interfaces

The launch review checks the output location, estimated snapshot/IC storage, MPI,
selected Blender availability, and missing requests. Native build, physical-asset, and
resource-admission checks still apply. Planning estimates do not establish physical
resolution, convergence, or render duration.

Answers files retain their existing schema, with optional highlight and deferred output
fields. `gadget --dry-run`, `gadget --ic-only`, native continuation, and campaign requests
keep their separate behavior; inspect `python commands.py gadget --help` for
their scope. Specialized experiments use their own request contracts and live help.

## Small execution controls

`particles_per_dimension` is a grid side length: 4 gives 64 particles per generated
species. Hydro and refined zooms add species/boundary populations. A 32-particle
isolated control uses an explicitly supplied particle state instead.

`core/config.yaml` contains unset template values. It is not a ready-to-run physical
experiment: copying it and filling only the box size can leave required finder,
source, or export parameters null. Prefer the setup walkthrough, review the final
configuration. A short schedule with multiple saved outputs supports history checks;
movie tests additionally need their renderer and selected display settings.
See the [learning guide](learning-guide.md) and [smoke report](smoke-test.md).
