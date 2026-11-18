# Completion plan

This is the working checklist for the README, documentation, optional walkthrough,
execution proof and cleanup request. Work proceeds from top to bottom. The goal
stays active until every required open item has evidence supporting completion.

A checked box means the stated bounded result was inspected. It does not mean
universal correctness, every code branch, or publication-grade physics. A failed
or unavailable check stays open unless the requirement itself permits that result.
New defects found during these checks become additional numbered items here.

## Starting evidence

These completed checks establish the starting point; they do not replace the
closing checks after subsequent edits. Evidence is outside the repository at
`../research2-runtime/evidence/readme-smoke-20261104/`.

- [x] B01 — Read the user's attached audit and retain its installation, regression,
  numerical, physical, convergence, reproducibility and usability requirements.
- [x] B02 — Generate the source/command inventory: 265 application Python files
  and 192 command handlers. File listings remain separate from execution coverage.
- [x] B03 — Verify the recorded regression aggregate: 861 unique unittest passes,
  including 12 current optional-interface checks; preserve original suite hashes,
  fixture skips and their actual reruns. Evidence: `retained-tests-external-output-summary.json`.
- [x] B04 — Execute the retained complete 37-step walkthrough: 36 local passes,
  one local OpenGadget3 unavailable, with its actual independent execution separately
  passed in Linux. Evidence: `native-complete-portable-orbits-http-summary.json`
  and `native-independent-sidm-linux-summary.json`.
- [x] B05 — Execute the current ten-step figures profile after the external-output
  guard change: ten passes and 297 artifact hashes checked. Evidence:
  `external-output-figures-http-summary.json`.
- [x] B06 — Execute all 18 cells of code in the 155-cell notebook against the full
  saved walkthrough, and separately against the current figures run. Absent results
  take explicit skip branches in the figures-only replay. Evidence:
  `optional-walkthrough-portable-orbits-summary.json` and
  `optional-walkthrough-external-output-summary.json`.
- [x] B07 — Verify 101 registered figure recipes have supported entries in their
  selected examples. The remaining graph-gain recipe is calibration-blocked.
  Evidence: `graph-coverage.json`; this is not scientific acceptance of 101 models.
- [x] B08 — Run the portable 32-particle flyby from another working directory on
  macOS and in Ubuntu with read-only source and no AGAMA. Evidence:
  `portable-orbits-platform-summary.json`.
- [x] B09 — Enforce external output directories in the CLI, server and advanced
  worker, including resolved symlink paths; retain boundary tests.
- [x] B10 — Fix loop-declared CLI argument documentation and compare the actual
  phase-audit arguments with live help. Evidence: `reference-argument-expansion-summary.json`.
- [x] B11 — Scan disposable source-tree files and revalidate eight native
  installations; preserve required sources, licenses and physical tables.
  Evidence: `final-source-cleanup-summary.json`. A closing scan is still required.

## 1. Finish the documentation, one file at a time

For each file below: read the whole document; lead with its purpose; make commands
and their prerequisites easy to find; explain inputs, implementation, outputs and
units where relevant; remove redundant prose; check its navigation and examples
against current source. A list of signatures alone does not finish an explanation.

- [x] D01 — [README](../README.md): concise entry point; separate first simulation,
  existing-run analysis and optional demonstrations; put detailed profile inventories
  in their guide; explain the cubic particle-count setting before the first run.
  Completed: replaced the long profile narrative with task/profile tables, moved
  particle-count units ahead of setup, kept detailed requirements in the lab guide,
  and removed repeated closing caveats. README commands retain their original syntax.
- [x] D02 — [Learning guide](learning-guide.md): coherent reading order; equations
  next to interpretation, units and source chains; clear distinction between fresh
  runnable controls and archived examples.
  Completed: read the full guide, kept the backend overview uninterrupted, added
  a tested seven-step quick start and artifact-reading table, separated archive/
  replay/execution scopes and checked mentioned command names. Evidence:
  `learning-guide-editorial-summary.json`; local links passed after reordering.
- [x] D03 — [Operator guide](operator-guide.md): installation-to-results sequence;
  clear prerequisites, placeholders, expected files and recovery routes for commands;
  optional research experiments easy to distinguish from a first run.
  Completed: read the full guide, added task navigation, explicit placeholder/
  environment checks and expected run files; separated visual recipes from the
  first-run path and moved logs into external runtime storage. All 31 explicitly
  named public commands passed bounded live help. Evidence:
  `operator-guide-editorial-summary.json`.
- [x] D04 — [Configuration](configuration.md): draft versus frozen config, prompt
  editing, schedule/highlights, output selection and continuation explained together.
  Completed: reviewed all sections against the questionnaire, launch review and
  epoch parser; added draft/answers/config/manifest/request distinctions and the
  full continuation command. Six schedule/highlight examples/rejections passed;
  gadget help confirms the documented switches. Evidence:
  `configuration-editorial-summary.json`.
- [x] D05 — [Software](software.md): each dependency's role, installation profile,
  actual consumer, assets, invocation, output and tested platform scope.
  Completed: read the full guide; added task navigation, activation, workflow-to-
  dependency/product mapping and the distinction between dependency and test
  profiles. All nine declared profiles match their requirements/source paths;
  setup and native-worker help passed. Existing linked execution receipts retain
  their platform limits. Evidence: `software-guide-editorial-summary.json`.
- [x] D06 — [Architecture](architecture.md): concise conceptual pipeline before the
  source map; update historical counts and explain dynamic logical-module loading.
  Completed: read the full guide; added eight missing core-file entries, reconciled
  the 186-core/188-implementation/265-application scopes and current docstring counts,
  explained alias identity, section boundaries and separate globals, and removed
  stale static-only wording. All six aliases passed fresh-interpreter source/identity/
  isolation checks; core table coverage is complete. Evidence:
  `architecture-editorial-summary.json`.
- [x] D07 — [Calculations](calculations.md): check equation symbols, dimensions,
  normalization, estimator definitions and implementation references; link detailed
  worked examples instead of duplicating them.
  Completed: read the full guide; clarified logarithmic HMF normalization and bin
  averaging, cgs cooling/BH dimensions, density-map integration and the separate
  specific-angular-momentum/Bullock-spin definitions. Linked the worked notebook
  and replaced stale static-review language. Four bounded numerical checks passed
  for histogram units, top-hat/radius limits, BH conversions and a Fourier-mode
  potential/acceleration/Hessian. Broader native/scientific audits remain S05–S07.
  Evidence: `calculations-editorial-summary.json`.
- [x] D08 — [Figure contracts](figure-contracts.md): every one of 102 recipes has
  required inputs, producer, units, support and interpretation; mark the unresolved
  graph-gain example visibly and explain how its acceptance gate works.
  Completed: read the full guide; added per-recipe axes from 101 retained examples
  after verifying their physical-array hashes, family interpretation guidance and
  a visible unresolved graph-gain gate. Corrected five overbroad cosmology input
  descriptions in the guide/registry and the SFR measurement description. All 102
  rows match current callable routes and required measurements. Evidence:
  `figure-contracts-editorial-summary.json`, `figure-contracts-editorial-inputs.json`.
- [x] D09 — [Outputs](outputs.md): readable product catalogue; navigation, original
  arrays, manifests and workflow states; show how to inspect an actual plotted point.
  Completed: read the full guide; added task navigation, current execution/browser
  scope and an executable original-array inspection example. The exact snippet
  verified a retained input-power archive and printed its first supported point
  with units and source hashes. Evidence: `outputs-editorial-summary.json`.
- [x] D10 — [Troubleshooting](troubleshooting.md): actionable symptom → evidence →
  recovery; distinguish missing dependencies, bad inputs, memory limits and numerical
  rejection; preserve original results when retrying.
  Completed: read the full guide; added a stage-based recovery path and MPI/FFTW,
  Grackle, memory, shard and external-output symptoms. Kept numerical rejections
  distinct from resource failures and removed archive/old-movie instructions from
  routine recovery. Three parser checks and setup prerequisites passed. Evidence:
  `troubleshooting-editorial-summary.json`.
- [x] D11 — [Optional lab guide](optional-tests.md): clear profile comparison,
  current counts, dependencies, resource bounds, commands, reports, notebook and removal.
  Completed: read the full guide; added a nine-profile decision table and exact
  CLI/dashboard notebook-directory selection. Counts match the runner's actual
  expression and current worker lists; four parser checks passed. Evidence:
  `optional-guides-editorial-summary.json`.
- [x] D12 — [Optional folder README](../optional_tests/README.md): short installation,
  launch and first-result path; link the full guide instead of repeating its inventory.
  Completed: replaced duplicated backend details with activation, serve/quick-run,
  report files, next-profile choices, notebook and removal instructions. Quick-run
  execution reuses the unchanged seven-step D02 evidence; parser checks and current
  guide hashes are in `optional-guides-editorial-summary.json`.
- [x] D13 — [HMF research](hmf-research.md): prediction, simulation selection and
  learned fits have distinct inputs/outputs; density conventions and held-out
  cosmology splits remain explicit.
  Completed: read the full guide; added a three-workflow input/result table and
  separated production 512³/1024³ examples from tiny execution controls. Checked
  density conversion, simulation-count provenance, exposure and resolution/split
  conventions against the implementations; all six documented command/subcommand
  help checks passed. Evidence: `hmf-guide-editorial-summary.json`.
- [x] D14 — [Release status](finalization.md): one maintained current summary;
  separate old evidence, unresolved work and broader engineering follow-up;
  remove misleading statements implying a current check was static-only.
  Completed: read the full report; replaced repeated campaign narrative with a
  compact receipt-linked summary, retained the old static-review scope/counts as
  history and removed the old WSL movie export from required work. Scientific,
  recipient and closing gaps remain explicit. Evidence:
  `release-reports-editorial-summary.json`.
- [x] D15 — [Smoke report](smoke-test.md): current results easy to find; historical
  failures remain traceable; every numerical claim points to a receipt and its scope;
  avoid a long chronological narrative hiding the current answer.
  Completed: read the full report; added the latest results first and preserved
  original failures/milestones in a labelled expandable history. Checked summary
  counts against original receipts, rehashed all 1,444 artifacts across the two
  retained runs and verified notebook source/executed-file hashes. No native rerun
  or finished-revision certification implied. Evidence:
  `release-reports-editorial-summary.json`.
- [x] D16 — [Command reference](command-reference.md): regenerate, inspect all 192
  entries, resolve blank/dynamic/delegated arguments and check examples against help.
  Completed: regenerated both references; checked all 192 command sections against
  their parser owners and declarations. The generator follows twelve simple
  forwarders, expands literal loop flags/type pairs, documents concrete loop-created
  parser choices and explains verification's unittest/group interface. All twelve
  delegated help probes passed; original 180 direct-help outputs were rehashed with
  the unchanged dispatcher. Six focused string/type/group cases passed. Evidence:
  `command-reference-editorial-summary.json`, `command-reference-groups-summary.json`.
- [ ] D17 — [Python reference](python-reference.md): regenerate; every application
  file has a useful role and invocation/caller explanation. Resolve missing descriptions
  of public callables using actual implementation, including side effects and outputs.
  Use `callable-documentation-gaps.json` to track the current backlog; do not fill it
  with name-based boilerplate or silently declare undocumented functions complete.
  Current coverage: **2,407 of 2,739 public top-level definitions** across the
  regenerated 265-file application inventory, including scripts and optional helpers.
  Of these, 957 descriptions are manually reviewed implementation-bound notes across
  170 files; the remainder use source docstrings. **332 gaps across 44 files remain.**
  Methods, private and nested definitions are outside this count. The original
  backlog for this scope was 1,289; the older 961 count used a narrower inventory.

  `commands.py`, `core/cli.py` and all 120 public definitions in consolidated SIDM
  now have descriptions. Reviewed domains also include workflow preparation versus
  execution, native histories, scientific graph calculations/units/support, measured
  scene assets and archive navigation/storage behavior. All public top-level
  definitions in `core/initial_conditions.py` now have descriptions. The latest 17
  close SBI acceptance/prior/calibration/OOD adapters and public IC/paired-response
  orchestration. They distinguish source-signature preparation from actual training,
  preserve rejected calibration/support and bind native contrasts to original
  histories, physical clocks and exact IC identity.
  Command entries reuse reviewed notes
  when their source handler has no docstring. Documentation coverage is not execution
  or branch-coverage evidence. Repairs discovered during review have separate
  S09–S11 evidence; the latest documentation batch changes no numerical code.

  Evidence: `python-api-notes-editorial-summary.json` (binding/rejection checks)
  and `python-api-notes-public-cli-summary.json` (unchanged parser/source checks),
  plus `python-api-notes-response-campaigns-summary.json` (current coverage and
  bindings). Earlier domain receipts remain retained. The current backlog
  is `callable-documentation-gaps.json`.
  The initial 642 reviewed notes also passed live optional source-map transport;
  selecting a function shows its description above the source. Actual HTTP/Edge
  checks and 14 interface tests verify transport, current hashes and stale-binding
  rejection. Evidence: `source-map-notes-summary.json`. The current catalogue reader
  exposes all 957 reviewed descriptions; 332 gaps remain explicit. No numerical
  workflow was executed for the latest documentation-only batch.
  Two generated-gallery JavaScript tests also pass for single/multiple terms,
  whitespace, case, unmatched terms and clearing filters. These use Node.js with a
  DOM test double; they do not replace browser layout checks. The emitted regex was
  already correct, so no renderer repair was needed. Evidence: `gallery-search-summary.json`.
- [ ] D18 — Check the guide set together: consistent terminology, heading hierarchy,
  units, code-block languages, tables, relative links and no contradictory counts.

## 2. Finish recipient-side reproducibility

An archived image is useful teaching material, but it is not a portable rerun.
Inventory and address the remaining gaps before calling the walkthrough complete.

- [ ] P01 — Produce a per-recipe reproducibility matrix for all 102 entries:
  shipped fresh command, archived replay, extra physical input required, or unresolved.
  Include the precise input files/software needed and evidence for each classification.
- [ ] P02 — Make the supported assembly examples reproducible with shipped code:
  concentration, stellar mass, morphology and their original graph/clock requirements.
  Exercise them in a new external directory and verify the measured arrays.
- [ ] P03 — Make the supported filament alignment/history examples reproducible
  with shipped code; preserve synthetic ancestry, field thresholds, source hashes and
  bounded-clock requirements; test the original rejection cases too.
- [ ] P04 — Make the paired native action example reproducible with shipped code:
  two real 32-particle runs, identical initial states, physical G, common projected
  diagnostic potential, original action support and independent numerical comparison.
- [ ] P05 — Resolve the remaining nonportable examples identified by P01, including
  source-physics budgets/light, stream/inference/observation and convergence examples.
  Supply real commands or required input contracts; never replace physical measurements
  with fabricated data to make the matrix green.
- [x] P06 — Check the visual profile at 512/1024 particles: units, density/zoom mass
  integrals, useful larger images and actual artifacts; retain the small defaults.
  Both current CLI runs passed all seven steps (14 passes). Independently checked
  exact HDF5/JSON identities, saved shell estimators, full projection and all three
  cropped mass integrals; verified 24 declared artifact hashes and implementation
  hashes. Inspected density/zoom images at both counts and 1024-particle power/orbits.
  These are synthetic viewing controls, not native refinement or convergence proof.
  Evidence: `final-visual-profile-summary.json`; outputs remain external.
- [x] P07 — Run the optional folder in a temporary relocated checkout with the
  matching sources, without making a ZIP; detect absolute workstation-path assumptions
  and confirm stored examples/reports/notebook work as documented.
  Cloned the full checkout to a new external location and launched CLI/server from
  `/tmp`. Two new 32-particle quick runs passed seven checks each; all 24 declared
  artifact hashes match. Verified 308 first-party source hashes, all 642 reviewed
  source-map notes, 102 registry entries and 303 assets for 101 archived examples.
  All 18 shipped notebook code cells executed, plus one audit cell proving core,
  catalogue and reference-builder imports resolve inside the relocated checkout.
  Missing native/library outputs retain skip branches. Existing external Python and
  notebook tools were reused; fresh optional installation and full backend relocation
  remain outside this bounded check. No source mutations/caches or ZIP were created.
  Evidence: `relocated-checkout-summary.json`, `optional-walkthrough-relocated-summary.json`.
- [x] P08 — Inspect desktop and narrow optional-lab browser layouts, navigation,
  checklist, equation/source links, software states and artifact downloads.
  Actual Edge review at 1440×1000 and 390×844 checked all six sections, density
  equation/source drill-down, populated/missing atlas examples, current/archived
  execution labels, notebook/array downloads and the saved report. Two UI-launched
  32-particle quick runs passed seven checks each; 24 declared artifact hashes match.
  Fixed long software badges causing 11-pixel narrow-page overflow; measured
  client/scroll width now both 390. All 12 optional-interface tests passed. Saved
  screenshots/AX evidence are external; temporary tabs, viewport and server cleaned.
  This covers the optional lab, not the main research-run homepage or every browser.
  Evidence: `browser-lab-review-summary.json`.

## 3. Close scientific and audit gaps

These requirements cannot be checked off merely because a tiny run finished.
Use the original criteria and retain unsuccessful results.

- [ ] S01 — Address graph-baseline-gain calibration using a valid predeclared
  independent experiment. Both compared posterior families must pass their original
  checks before producing the actual registered gain figure. Do not tune on held-out
  truth, weaken criteria or relabel a diagnostic illustration as an accepted gain.
  Current evidence is rejected; completion time is unknown.
- [ ] S02 — Audit every software/backend against actual execution evidence and
  document remaining experimental/unvalidated capability states. Import success is
  not a backend calculation; child/native work needs its own outputs and receipts.
- [ ] S03 — Audit every application file and requested workflow against observed
  calls and actual runs. Resolve unexplained missing execution; distinguish mocked
  tests, native subprocesses and unexecuted branches. Do not claim every function works
  from file-level coverage alone.
- [ ] S04 — Verify seeded repetition and deliberate input/artifact corruption
  detection on representative end-to-end workflows, with exact IDs and justified
  numerical tolerances; preserve original and tampered copies separately.
  The bounded core paired-figure resume repair has separate S11 evidence;
  extra gallery producers and broader end-to-end workflows still need this audit.
- [ ] S05 — Complete the attachment's physical-source evidence audit: hydro,
  cooling, stochastic star formation, returns, feedback, BH and SIDM ledgers, boundaries
  and analytic/reference controls. Positive BH growth alone is not budget reconciliation.
- [ ] S06 — Complete the attachment's numerical/comparison evidence audit:
  gravity and linear growth, Fourier normalization, HMF units/counts/fit conventions,
  timestep/softening/mass resolution, finite volume and matched cosmological phases.
  Existing numerical matrices prove only their recorded domains.
- [ ] S07 — Check remaining physical-example/acceptance requirements from the
  attachment, including collisions, streams, matched models, void continuity and
  independent resolution. Record missing evidence as incomplete and obtain it.
- [ ] S08 — Verify the intended recipient installation and representative single/
  batch workflows on fresh Ubuntu/WSL with required optional dependencies. Existing
  base/engine checks and the reused Ubuntu flyby image are partial evidence.
- [x] S09 — Fix the paired-atlas cache reuse gap found during source review.
  Cached epochs now bind run metadata, complete native source inventory/hashes,
  map grid and relevant implementations, and require a sealed metadata record.
  Seven focused tests passed using real 32-particle HDF5 payloads with fixture
  catalogue readers: unchanged reuse, changed inputs/grid/code, source mutation
  during measurement and corrupt/legacy/incomplete cache rejection. Earlier caches
  remain preserved and require a new destination. This bounded repair does not
  replace S04's broader end-to-end repeat/corruption audit.
  Evidence: `paired-atlas-cache-summary.json`.
- [x] S10 — Correct isolated conservation-history fixed-mass conversion.
  Native noncomoving output has zero a/h exponents; the fixed `MassTable`
  fallback incorrectly divided by h. Equivalent 32-particle fixed/explicit HDF5
  fixtures reproduced the error (320 versus 160 Msun at h=0.5). After correction,
  two focused tests passed across h=0.25/0.5/0.7/1, checking mass, kinetic/potential
  energy, momentum, angular momentum, centre of mass, core effective count and IDs.
  Native evolution/build kernels are unchanged. These authored reduction controls
  do not establish full source budgets or replace the final current-revision suites.
  Evidence: `isolated-history-units-summary.json`, with original red/green logs.
- [x] S11 — Repair paired-figure resume validation without changing reductions.
  Bind original records, numeric caches, renderer sources and original exports in
  a sealed core-render receipt. Reject changed/legacy/incomplete comparisons while
  preserving their files; recheck inputs before publishing a new render. Seven tests
  passed, including actual nine-figure exports from authored 32-particle populations,
  independent mass/abundance/velocity checks and input mutation during rendering.
  Additional gallery entries remain compatible and require their own provenance
  checks. This does not replace S04 or the final regression.
  Evidence: `paired-figure-resume-summary.json`, red/green logs and retained exports.

## 4. Closing checks on the finished revision

Run these after the required edits above. Repeat a check only if a later change
invalidates it or reveals a new concern.

- [ ] V01 — Rebuild source/command references, notebook and optional inventories;
  record hashes and reconcile the reproducibility/capability matrices.
- [ ] V02 — Run relevant current regression suites in separate processes; supply
  required native fixtures; resolve skips and failures transparently. Run the separate
  pytest functions too. Record exact source compatibility for any reused evidence.
- [ ] V03 — Execute the final complete optional profile with installed requirements;
  verify every numerical assertion, software receipt and artifact hash. Document any
  platform-specific separately executed worker without disguising unavailable states.
- [ ] V04 — Execute every notebook code cell on the final full run and its portable
  recipient scenario; check arrays/units/equations/images, with skips scoped explicitly.
- [ ] V05 — Check local Markdown links/anchors, actual command examples, Python and
  JavaScript syntax, figure readability, missing assets and portable HTML navigation.
- [ ] V06 — Inspect the final source tree for caches, Finder/history files,
  obsolete duplicates and compiled/runtime artifacts. Keep required data, licenses,
  source and optional teaching assets. Keep new outputs/builds/recovery copies external.
  Do not create a ZIP or resume the old long movie export.
- [ ] V07 — Reconcile every explicit request and attached audit requirement with
  the final files and receipts; update this plan and release status. Remaining failed,
  unavailable or unverified required items prevent goal completion.
- [ ] V08 — Deliver a concise walkthrough with exact start commands, documentation,
  notebook and result links, what each software produced, and material limits. Mark
  the active goal complete only after the evidence supports every required item above.

## Scope notes

The user explicitly requested no new ZIP and no new long videos. The old unfinished
2,040-frame WSL export mentioned in historical release notes is not a new task to
restart here. Preserve its original records. Likewise, routine cleanup does not
mean deleting required FSPS/chemistry tables, licenses, scientific inputs or evidence.

Large-module refactoring is not needed merely to make the folder look clean.
Documentation and reachability still require inspection, but removing a function
requires evidence that it is unused; dynamic imports and user-authored workflows
must be considered. No universal mathematical or software guarantee is implied by
a passing checklist. The goal's broad scientific requirements remain open wherever
their evidence is incomplete.
