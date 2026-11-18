# Inspect and select parent targets for a physical zoom

The `select-zoom-targets` command reads a saved native GADGET FOF catalogue
and its matching group-ordered snapshots. It prints a source-bound, epoch-local
selection without generating initial conditions or evolving a simulation. This
interface and the additional selection metadata were statically inspected; they
have not been run as part of this work. Existing zoom workflows remain available.

For a completed parent run containing the requested snapshot and complete FOF
catalogue shards, from the Research2 directory use:

```sh
python commands.py select-zoom-targets /scratch/my-study/native --snapshot 8 --memory-gib 4 > /scratch/my-study/target-selection.json
```

The default selects one row with the largest DM membership among halos with at
least 1,000 DM particles and positive measured R200c, with the native row number
breaking ties. This is a candidate support floor, not a scientific convergence
criterion. Increase `--minimum-dm-particles`, add `--maximum-dm-particles`, set
`--maximum-targets`, or use `--ranking catalogue_order` to make a selection
appropriate for the question and allocation. Choose the threshold before
inspecting results if the selection is part of a hypothesis test.

An explicit saved-epoch FOF row has priority; set the target budget high enough
for every repeated override:

```sh
python commands.py select-zoom-targets /scratch/my-study/native --snapshot 8 --memory-gib 4 --host-row 12 --host-row 23 --maximum-targets 2 > /scratch/my-study/explicit-targets.json
```

Overrides may lie outside the automatic DM count limits. Their recorded
`automatic_rule_qualified` and `selection_issues` make the deviation visible;
a nonpositive R200c cannot be traced, and a row without collisionless members
cannot produce a physical trace. A selected catalogue row is an identity only
within its source catalogue and epoch: later correspondence requires the
existing exact-particle or tree-based continuity checks. The record includes
catalogue and snapshot SHA-256 hashes, scale factor, box and coordinates,
wrapped center, proper and comoving R200c, DM and stellar support counts, and
M200c only when that catalogue field is available and positive. It does not
infer M200c from total FOF mass or claim an absent galaxy is resolved. Reading
and hashing large snapshot shards can require substantial time and I/O.

To bind a reviewed preview to a physical zoom, copy its exact `rules` and
`explicit_overrides` into the campaign's `selection_rules` and
`explicit_host_rows`, respectively. Add this optional field to the campaign
request, replacing the digest with the SHA-256 of the preview file:

```json
"selection_preview": {
  "path": "/scratch/my-study/target-selection.json",
  "sha256": "REPLACE_WITH_64_LOWERCASE_HEX_DIGITS"
}
```

Obtain the digest with `shasum -a 256 /scratch/my-study/target-selection.json`
on macOS or `sha256sum /scratch/my-study/target-selection.json` on Linux.
The campaign verifies the file hash, recomputes the selection from its actual
parent run and catalogue, and requires the full preview and fresh selection to
match before tracing or generation. The exact run path, epoch, rules, overrides,
source hashes and selected records must agree. Changing an automatically
selected row into an explicit override changes the selection record; produce a
new preview with that override first. The preview is optional for existing
campaigns, whose selection is still recorded and bound into the campaign
signature. This interface has been statically inspected only.

Use [the campaign example](../configs/research/block40_zoom_campaign.example.json)
as a schema starting point, not as runnable data: supply the actual parent IC,
source manifest, retained noise, spectrum, generator and control hashes,
refinement seeds, native evolution and diagnostic inputs, output epochs, and
resource limits. Execute `python commands.py zoom-campaign --campaign
/path/to/filled-campaign.json --output /scratch/my-study/zooms` only after
validation is authorized.

The existing campaign traces selected host collisionless membership and an
explicit R200c aperture back to parent IC material, stages the nested MUSIC
hierarchy and phase inputs, generates and evolves each configured zoom,
recovers descendant material, and evaluates declared contamination, force and
observable bounds. Success on those declared controls is not a blanket science
acceptance or a substitute for convergence studies. Inspect the retained
selection, traced-region, material-map, recovery, adequacy and provenance
records. Missing parent fields, unsupported models, inadequate resolution,
phase fidelity and coarse contamination remain explicit limitations, not
synthetic replacements. See the [research benchmark plan](research-benchmark-plan.md)
for later validation and the [one-command roadmap](one-command-research-roadmap.md)
for the integration still required.
