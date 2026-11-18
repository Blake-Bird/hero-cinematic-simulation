# Release status

**Updated:** November 7, 2026. This is the maintained implementation and validation
checklist. Machine-specific logs and build evidence belong outside the source tree.
The current small-run campaign is under
`research2-runtime/evidence/readme-smoke-20261104/`; earlier release checks remain
under `research2-runtime/evidence/finalization-20261103/`.

The requested final review is still in progress. Bounded local execution is
established below; portable reruns, documentation gaps and scientific validation
remain open in the completion plan.

The [completion plan](completion-plan.md) is the current ordered checklist for
remaining documentation, recipient-side reruns, scientific gaps and closing checks.
Its items require actual evidence before being marked complete.

## Current evidence at a glance

The November 4–6 [smoke report](smoke-test.md) contains the software-by-software
calculations, source revisions, original failures and subsequent controls.
Receipts below live outside the source tree and require the retained runtime
folder. None represents universal branch coverage or astrophysical convergence.

| Check | Recorded result | Receipt |
|---|---|---|
| Complete optional walkthrough | 37 steps; 36 local passes, Linux-only OpenGadget3 unavailable locally; 171.649 s, 1,147 verified artifacts | [Complete run](../../research2-runtime/evidence/readme-smoke-20261104/native-complete-portable-orbits-http-summary.json) |
| Independent SIDM | Separate real 32-particle Linux scattering execution | [Linux control](../../research2-runtime/evidence/readme-smoke-20261104/native-independent-sidm-linux-summary.json) |
| Current external-output guard | Ten figures-profile checks passed; 18.18 s, 297 artifacts; direct and symlink source paths rejected | [Current HTTP run](../../research2-runtime/evidence/readme-smoke-20261104/external-output-figures-http-summary.json) |
| Larger visual controls | 512- and 1024-particle runs: 14 step passes; density and all viewing-zoom mass integrals independently checked; 24 declared artifacts verified | [Visual-profile receipt](../../research2-runtime/evidence/readme-smoke-20261104/final-visual-profile-summary.json) |
| Optional-lab browser | Edge desktop/narrow walkthrough, equation/source links, graph/notebook downloads and saved report checked; 14 quick-step passes and 12 interface tests; narrow software-badge overflow fixed | [Browser receipt](../../research2-runtime/evidence/readme-smoke-20261104/browser-lab-review-summary.json) |
| Source-map integration | Initial 642 reviewed notes matched the live server response; selected function prose verified in Edge; 14 interface tests, including stale-binding rejection. Later note additions retain their separate AST/catalogue checks. | [Source-map receipt](../../research2-runtime/evidence/readme-smoke-20261104/source-map-notes-summary.json) |
| Relocated checkout | CLI/server launched from `/tmp`; two 32-particle quick runs pass, 308 source hashes and 303 archived assets verified; all 18 notebook code cells execute with absent-backend skips | [Relocation receipt](../../research2-runtime/evidence/readme-smoke-20261104/relocated-checkout-summary.json), [Notebook replay](../../research2-runtime/evidence/readme-smoke-20261104/optional-walkthrough-relocated-summary.json) |
| Regression aggregate | 861 unittest passes; fixture skips resolved by actual reruns; five additional pytest passes retain separate evidence | [Suite aggregate](../../research2-runtime/evidence/readme-smoke-20261104/retained-tests-external-output-summary.json) |
| Paired-atlas cache reuse | Seven current focused tests passed: fresh 32-particle HDF5 measurement, unchanged reuse and stale/corrupt cache rejection; catalogue readers are fixtures | [Cache regression](../../research2-runtime/evidence/readme-smoke-20261104/paired-atlas-cache-summary.json) |
| Isolated fixed-mass units | Corrected an erroneous division by h; two tests with real 32-particle HDF5 fixtures pass across fixed/explicit storage and multiple h values | [Unit regression](../../research2-runtime/evidence/readme-smoke-20261104/isolated-history-units-summary.json) |
| Paired-figure resume | Seven focused tests pass: nine actual plots from authored 32-particle populations, independent numeric checks and stale/corrupt/incomplete render rejection; no native evolution | [Render regression](../../research2-runtime/evidence/readme-smoke-20261104/paired-figure-resume-summary.json) |
| Notebook replay | 155 cells, all 18 code cells passed against the complete run plus separate Linux outputs | [Notebook receipt](../../research2-runtime/evidence/readme-smoke-20261104/optional-walkthrough-portable-orbits-summary.json) |
| Graph examples | 101 of 102 rendered with supported plotted entries; graph gain calibration-blocked | [Recipe evidence](../../research2-runtime/evidence/readme-smoke-20261104/graph-coverage.json) |
| Portable flyby | Same 32-particle analytic control passed from another directory on macOS and Ubuntu, with read-only source and no AGAMA in Ubuntu | [Platform checks](../../research2-runtime/evidence/readme-smoke-20261104/portable-orbits-platform-summary.json) |
| Documentation | Current syntax/link results, exact counts and scope | [Latest check](../../research2-runtime/evidence/readme-smoke-20261104/documentation-check.json) |
| Source hygiene | Caches, old environment/logs and legacy binaries moved outside source; required source, data and licenses retained | [Cleanup receipt](../../research2-runtime/evidence/readme-smoke-20261104/final-source-cleanup-summary.json) |

The 37-step run predates the external-output guard change. Its unchanged numerical
workers retain their original evidence; the ten-step run tests current guard/server
behavior. Core/application regression evidence is reused only with the recorded
source compatibility, and current teaching changes have separate interface checks.
The paired-atlas cache producer has since changed and has its own focused receipt;
the earlier aggregate does not certify that changed implementation. Old epoch caches
without the new metadata seal and source/grid/code bindings require a fresh destination.
The isolated conservation-history reader also has a separately tested mass-unit
correction; earlier aggregate/native receipts do not certify that changed reader.
The paired core figure renderer now requires its own sealed receipt before reuse,
binding the original data, code and exports. Its focused receipt covers original
renderer products; extra gallery producers retain separate provenance requirements.
The earlier aggregate does not certify this changed resume implementation either.
These rows do not replace the closing checks on the finished revision.

### What the walkthrough exercises

The fixed profiles execute real small calculations with GADGET/MPI, twelve Python
backends, CLASS/AxiCLASS, Blender, monofonIC, Grackle, MUSIC, Rockstar,
VELOCIraptor, Consistent Trees and DisPerSE. Independent OpenGadget3 runs in Linux.
Separate controls exercise gas cooling, native stellar births and seeded BH
accretion, plus known-shape, density-slope and flyby measurements. The
[profile guide](optional-tests.md#choose-a-profile) lists exact prerequisites,
fixed particle/cell/catalogue counts, output files and scope. Increasing the quick
fixture count does not change every backend's count.

Fresh Ubuntu 24.04 ARM64 checks cover base installation, the default three engine
builds and a separate native 32-particle run. They do not certify all optional
profiles, other architectures or WSL. The optional lab now has actual desktop/narrow
Edge layout and navigation checks; the main research-run homepage remains separate.

The figure controls include synthetic clocks/ancestry and known analytic inputs.
The native paired-action example uses two real 32-particle runs, physical G and a
shared diagnostic monopole, with independent AGAMA comparison over 192 states.
It measures conditional action increments, not a stationary diffusion coefficient.
Graph gain remains blocked: the original harmonic study accepted the summary model
and rejected the graph model; a separately predeclared expanded study rejected
both. Preserve those decisions and their held-out data.

### Documentation and cleanup scope

The generated reference inventories 265 application Python files and 192 commands.
Live help checks cover 180 directly parsed handlers; twelve delegated/interactive
handlers are identified separately. The architecture map now distinguishes 186 core
files, 188 core/root implementation files and the wider application count. Source
connections and callable resolution remain distinct from execution coverage.

New outputs, builds, environments and recovery copies stay in the external runtime.
Required FSPS/chemistry tables, vendored source, licenses and optional teaching
assets remain. No new ZIP or long movie export is required. The final source scan,
reference refresh and finished-revision reruns remain in the completion plan.

## Scope of the previous static review

The preceding source and documentation review used file reading, Python syntax trees,
declared references, Markdown links, and package-content inspection. It did **not** run
tests, simulations, renders, or movie encoders. Historical execution results below apply
to an earlier source revision.

- [x] Explain the connected run, HDF5 inputs, graph production, and export flow
  in the README and architecture guide, with pseudocode.
- [x] Document gravity/gas, implicit cooling, SH03 star formation, stellar returns,
  Bondi/Eddington accretion, feedback accounting, and FSPS normalization using
  the corresponding implementations.
- [x] Account for all 180 implementation files and the verification sources.
  The native cooling control is intentionally verification-only.
- [x] Review module/function documentation and source organization. All
  implementation files have module docstrings; the source map records remaining
  function-documentation and large-file limitations.
- [x] Identify runtime, build, release, or verification consumers for each declared
  Python distribution and native backend.
- [x] Archive five unused reference checkouts, retaining attribution and pinned
  references. Remove the redundant inventory-report helper.
- [x] Remove obsolete imports and duplicate map-helper source declarations.
- [x] Exclude Git directories and file-based submodule links from the portable
  archive while preserving working-copy Git metadata.
- [x] Prevent a workflow-owned movie from appearing again as a standalone
  historical export in navigation.
- [x] Refresh the layout inventory to include all 178 core Python files while
  preserving the six historical aliases and the migration-history digest.
- [x] Parse all 211 first-party Python files and inspect the declared definition
  locations of all 102 graph recipes without executing their producers.
- [x] Resolve first-party documentation links and declared dependency paths.

## Implemented behavior

- [x] Numbered terminal setup with units, help, saved drafts, back/edit, and review.
- [x] Configuration continuation that reuses a verified linear bundle only when
  its configuration, solver hash, and output root match.
- [x] Exact inclusive redshift ranges and starred highlights. Older runs default
  to no highlights; highlights preserve analysis and movie coverage.
- [x] Launch review for output locations, storage estimates, MPI, selected Blender
  availability, and specialized requests, alongside the existing native gates.
- [x] Explicitly deferred measurements remain pending while ready products run.
- [x] Independent optional export failures are recorded while later exports
  continue; workflow records include product states and retry guidance.
- [x] Results homepage with attempt selection, Highlights, By redshift, Evolution,
  Movies, Data, and Status. Native snapshot and publication names remain stable.
- [x] Recorded manifests and historical summaries drive navigation. Missing files,
  hash mismatches, external paths, and oversized metadata are reported.
- [x] Correct the batch test's macOS temporary-path alias comparison while
  retaining its failure-reporting assertions.
- [x] Accept highlights and recorded output selections through the native
  configuration validator.
- [x] Default installation uses the base Python profile and selected native
  engines; broader profiles and all-engine setup are explicit choices.
- [x] Move embedded generated builds and the root Ewald cache outside the source;
  preserve environments, physical tables, licensing, checkpoints, and evidence.
- [x] Include maintained first-party guides in the portable package and exclude
  environments, binaries, caches, run outputs, and retired downloads.

## Historical execution evidence

These results predate the latest source/documentation changes. They are retained as
evidence of their original scope, not as a fresh validation of this revision.

| Check | Recorded result |
|---|---|
| Full `verification.py` suite | 748 checks run; 5 skipped; no failures |
| Standalone tests | 65 passed |
| Focused changed-workflow checks | 26 passed |
| Closing package/native-manifest/workstation/output checks | 23 passed |
| Main saved-run navigation | 13 LCDM and 5 EDE attempts indexed; no navigation issues |
| Generated HTML links | 1,900 checked; no missing targets |
| Representative HMF image | Labels, units, uncertainty legend, and annotations inspected |
| Source package | Generation, byte integrity, and packaged command startup exercised |

Shell setup syntax/help were checked. The Ubuntu setup script was not used to install
packages on the Mac. Large metadata was linked with explicit intake-limit notes. HTML
link checks do not establish browser rendering.

## Validation still required

- [ ] Install and exercise representative single/batch workflows on fresh
  Ubuntu/WSL with broader optional profiles and the recipient's architecture.
  The ARM64 Ubuntu base/default-engine checks above do not establish those installations.
- [ ] Review the main research-run results homepage in a browser and inspect each
  figure/movie family against appropriate data. Optional-lab browser checks above
  cover its walkthrough, atlas and offline quick report, not this broader homepage.
- [ ] Complete physical examples for halo collisions, stellar tidal streams,
  matched-model fingerprints, and independent-resolution comparisons.
- [ ] Validate void continuity, boundary interpretation, and numerical convergence
  with suitable data and controlled realizations.

The old unfinished 2,040-frame WSL export is historical evidence, not work to
restart for this request. Its original records remain retained.

## Engineering follow-up

- [ ] Improve private/helper documentation during future domain changes.
- [ ] Split large consolidated modules incrementally, preserving aliases,
  numerical results, and provenance, with a dedicated validation run.
- [ ] Review per-function reachability before claiming zero dead functions.
  File-level connections alone cannot prove a function safe to delete when
  workflows can be loaded dynamically or supplied by a user.

