# Retained software smoke test — November 4–7, 2026

This campaign executes small calculations, native particle controls, analysis and visual
exports on the local macOS workstation, with a separate Linux container for the
independent SIDM engine. Its outputs are retained for inspection and later deletion.
Tiny controls establish execution and their stated numerical assertions. They do not
establish astrophysical convergence, parameter-space coverage, or publication acceptance.

## Latest recorded results

Start with these receipts. The detailed records below preserve earlier milestones,
failed attempts and their original source revisions; their old step/test/notebook
counts are historical. Documentation and registry-description edits after these
runs do not turn them into new executions. The finished-revision checks remain
open in the [completion plan](completion-plan.md).

| Check | Result and scope | Receipt |
|---|---|---|
| Complete optional profile | 37 steps; 36 local passes, OpenGadget3 unavailable locally; 171.649 seconds and 1,147 verified artifacts | [Complete HTTP run](../../research2-runtime/evidence/readme-smoke-20261104/native-complete-portable-orbits-http-summary.json) |
| Independent OpenGadget3 | Separate real 32-particle Linux worker passed | [Linux scattering](../../research2-runtime/evidence/readme-smoke-20261104/native-independent-sidm-linux-summary.json) |
| External-output guard revision | Ten figures-profile steps passed; 18.18 seconds, 297 verified artifacts; current CLI/server output confinement | [Figures HTTP run](../../research2-runtime/evidence/readme-smoke-20261104/external-output-figures-http-summary.json) |
| Optional-lab browser review | Actual Edge desktop/narrow navigation, equation/source chain, atlas, notebook/array downloads and offline report; 14 UI-launched quick-step passes and 12 interface tests; narrow software-badge overflow corrected | [Browser review](../../research2-runtime/evidence/readme-smoke-20261104/browser-lab-review-summary.json) |
| Source-map integration | Initial 642 reviewed descriptions matched the actual bootstrap response; selected equation/units prose visible above source in Edge; 14 interface tests. Later note additions have separate AST/catalogue checks. | [Source-map checks](../../research2-runtime/evidence/readme-smoke-20261104/source-map-notes-summary.json) |
| Relocated recipient workflow | Full checkout copied without ZIP; CLI/server run from `/tmp`, 14 new quick-step passes, 308 matching source files and 303 archived assets; all 18 notebook code cells execute with scoped skips using existing installed dependencies | [Relocation checks](../../research2-runtime/evidence/readme-smoke-20261104/relocated-checkout-summary.json) |
| Larger visual controls | 512/1024-particle visual profiles: 14 step passes, independently checked density/zoom mass integrals and 24 verified declared artifacts | [Visual checks](../../research2-runtime/evidence/readme-smoke-20261104/final-visual-profile-summary.json) |
| Reader repairs | Seven paired-atlas cache tests and two isolated fixed-mass unit tests; real 32-particle HDF5 reduction fixtures, with scope and original red/green evidence retained | [Cache checks](../../research2-runtime/evidence/readme-smoke-20261104/paired-atlas-cache-summary.json), [Mass-unit checks](../../research2-runtime/evidence/readme-smoke-20261104/isolated-history-units-summary.json) |
| Regression evidence | 861 unique unittest passes after actual fixture reruns; five separately retained pytest passes give 866 recorded checks | [Unittest aggregate](../../research2-runtime/evidence/readme-smoke-20261104/retained-tests-external-output-summary.json), [pytest record](../../research2-runtime/evidence/readme-smoke-20261104/pytest-clean-final-summary.json) |
| Notebook | All 18 code cells in 155 cells passed on the complete run plus actual Linux outputs; saved-output replay, not another solver launch | [Full replay](../../research2-runtime/evidence/readme-smoke-20261104/optional-walkthrough-portable-orbits-summary.json) |
| Figures-only notebook | All 18 code cells executed on the ten-step run; absent library/native results use explicit skip branches | [Scoped replay](../../research2-runtime/evidence/readme-smoke-20261104/optional-walkthrough-external-output-summary.json) |
| Registered recipes | 101 of 102 rendered with supported entries in selected examples; graph gain remains calibration-blocked | [Per-recipe evidence](../../research2-runtime/evidence/readme-smoke-20261104/graph-coverage.json) |
| Portable analytic flyby | macOS and Ubuntu from another working directory; read-only source/no network/no AGAMA in the Ubuntu check | [Platform receipt](../../research2-runtime/evidence/readme-smoke-20261104/portable-orbits-platform-summary.json) |
| Documentation | 265 application Python files and 192 command handlers inventoried; latest link/syntax counts in the receipt | [Latest documentation check](../../research2-runtime/evidence/readme-smoke-20261104/documentation-check.json) |

For a fresh small run, use the [optional profile guide](optional-tests.md#choose-a-profile).
It distinguishes fixed backend inputs from the 32-particle quick fixture. For each
registered figure's inputs, units and interpretation, use the
[figure reference](figure-contracts.md). To trace one actual plotted point, follow
[the array inspection example](outputs.md#inspect-an-output-from-the-smoke-campaign).

**Still incomplete:** graph-gain calibration, the remaining portable example
wrappers, full recipient optional installation, main research-run homepage review and
broader scientific/source-budget/convergence audits. The tiny runs and historical
movies below do not close those requirements. No new ZIP or long video is needed.

## Open the evidence

On the workstation used for this run:

- [Evidence gallery](../../research2-runtime/evidence/readme-smoke-20261104/index.html)
  links calculation plots, decoded Blender frames, movies, receipts, Python coverage
  and all 102 graph routes.
- [Connected results page](../../research2-runtime/runs/readme-smoke-20261104/synthetic-corrected/results/index.html)
  exposes saved redshifts, completed products and earlier failed attempts.
- [Artifact inventory](../../research2-runtime/evidence/readme-smoke-20261104/artifact-inventory.json)
  records main-run files, sizes, SHA256 hashes and image/movie decoding checks.
- [Python coverage](../../research2-runtime/evidence/readme-smoke-20261104/file-coverage.json)
  lists every first-party application file and observed regression function calls.
- [Graph coverage](../../research2-runtime/evidence/readme-smoke-20261104/graph-coverage.json)
  distinguishes connected routes from observed regression calls and complete scientific
  recipe execution.

These links intentionally point outside the portable source. Recipients need the
retained runtime/evidence folders as well as the source to inspect this campaign.
The public [learning guide](learning-guide.md), [command reference](command-reference.md)
and [Python reference](python-reference.md) remain useful without those local artifacts.

## Historical campaign records

The expandable record preserves the original software table and subsequent
verification milestones. Counts and words such as “current” or “now” inside it
refer to each milestone, not the latest summary above. Existing videos are
historical artifacts; they are not instructions to create new ones.

<details>
<summary>Read original calculations, failures, repairs and verification milestones</summary>

## What actually ran

| Software / workflow | Inputs and observed result | Evidence under the campaign evidence folder |
|---|---|---|
| NumPy, SciPy, matplotlib, YAML, HDF5, Pillow, imageio/FFmpeg | Actual numerical, configuration, snapshot, plot, raster and encoded movie work in the workflows below | `artifact-inventory.json`, `software-calculations.png`, workflow logs |
| CLASS | Fresh LCDM and WDM spectra, transfers and background bundles | `linear-summary.json`, `linear.log` |
| AxiCLASS | Fresh EDE spectrum/transfer/background calculation | Same linear summary/log; inspect its `ede` entry |
| GADGET gravity | 32-particle isolated control and restart comparison passed | `gravity-summary.json`, `native.log`; full receipts in `runs/.../gravity32/` |
| GADGET cosmological evolution | Separate 64-particle LCDM and WDM loads each saved three finite-position epochs, with FOF/SUBFIND enabled | `cosmo-particles-summary.json`; `runs/.../evolve64-no-thermal-*` |
| Native gravity matrix | Refreshed all 18 native cases and eight comparisons after the gas-free entropy guard. All passed. Seven receipt-backed regression checks then passed in 4.007 seconds against that new campaign. Earlier campaign receipts remain separately retained. | `native-matrix-vacuum-guard-summary.json`, `native-matrix-vacuum-guard-regression.log`; `runs/.../native-matrix-vacuum-guard/campaign.json` |
| Registered timestep contrast | Compared the measured relative orbit-position errors of the fixed 28-tracer Plummer controls, changing only timestep 0.002 → 0.001. At saved elapsed time 0.4 the maximum relative error fell from 5.42 × 10⁻⁸ to 1.35 × 10⁻⁸. All five times entered the registered plot. This deterministic finite-tracer statistic has no estimated sampling covariance; it is not a population uncertainty calculation. | `extend_timestep_convergence.py`, `registered-timestep-convergence.log`; `runs/.../registered-native-timestep-convergence/` |
| Registered softening contrast | Evolved two 32-particle Plummer loads with identical initial IDs and positions. At softening 0.01 and 0.02, all four saved RMS accelerations matched exactly. A second matched pair at 0.01 and 0.30 gave first-epoch RMS accelerations of 0.327895 and 0.250716 in native acceleration units, a 23.54% decrease. Both contrasts rendered. | `registered-softening-convergence.log`, `registered-softening-wide.log`, `extend_softening_convergence.py`, `extend_softening_wide.py` |
| Registered start-redshift contrast | Ran fresh CLASS, N-GenIC and GADGET branches with 64 particles and the same LCDM settings and seed, starting at redshifts 49 and 59. At common output redshifts 48.5 and 48 the earlier-start branch gave RMS peculiar speeds lower by 0.60149 and 0.59281 km/s. Both plotted samples are supported. The contrast exceeded the chosen 1% tolerance at both epochs. | `extend_start_redshift.py`, `registered-start-redshift.log` |
| Registered mass-resolution contrast | Ran 32- and 64-particle isolated GADGET Plummer controls with the same total mass, input model, seed, force controls and four saved times. Macro mass halved from 0.03125 to 0.015625 in native units. The initial mass-weighted RMS radii were 2.580982 and 2.778363 native lengths. The contrast includes finite sampling of the initial profile at each resolution. | `extend_mass_resolution.py`, `registered-mass-resolution.log` |
| Registered finder variation | Compared the native Rockstar and VELOCIraptor catalogue row counts on the same two hash-matched 512-DM synthetic snapshots. Rockstar reported one row and VELOCIraptor reported zero at each saved scale factor. The figure shows VELOCIraptor minus Rockstar. This checks finder variation with recorded settings and leaves tree variation and matched-halo accuracy open. | `extend_finder_variation.py`, `registered-finder-count-variation.log` |
| Registered native worldlines and impact geometry | Evolved a fresh sealed 32-particle isolated GADGET Plummer control. Chose the closest initial pair and extracted both exact IDs through four saved epochs. Both 3D worldlines rendered. The shared-frame Hermite search retained two boundary candidates and two refinement requests. Zero impact samples were supported because the control has no independent interpolation certificate. The pair consists of DM particles. | `extend_native_encounter_worldlines.py`, `verify_native_encounter_worldlines.py`, `registered-native-encounter-worldlines-verified.log` |
| Registered supported impact geometry | Evolved 28 harmonic-force tracers through nine saved epochs in a sealed GADGET stage. The selected exact-ID pair had an interior closest approach of 0.394530 proper kpc at 1281.39 Myr and a relative speed of 1.02151 km/s. A closed-form oscillator bound and the measured endpoint residuals gave a combined position error bound of 3.02 × 10⁻⁵ proper kpc. The supported point rendered. | `extend_harmonic_encounter.py`, `registered-harmonic-encounter.log` |
| Registered paired native stellar responses | Evolved the same 32 seeded stars with native self-gravity at softenings 0.01 and 0.30. The fixed curve and four common interior bins supplied 12 supported bin/epoch samples at 0, 3.91117 and 7.82234 Myr. Density ratios remained one; measured velocities and transverse centroids changed. The largest individual velocity difference was 0.01760 km/s. Before/after and density-deficit figures plotted supported samples. The action-change figure rendered with all samples masked because no measured host action field was supplied. This is an artificial stellar filament and numerical force-resolution contrast, with zero conditioned model-error covariance. | `extend_native_stellar_responses.py`, `native-stellar-softening-response-summary.json`; `runs/.../registered-native-stellar-softening-response-32-readable/` |
| Registered native assembly event timeline | Followed two native central SUBFIND objects through three snapshots and used the second object’s measured bound half-mass radius as the reference boundary. Both intervals passed the saved-epoch support check for each track. No pericenter, infall, apocenter or ancestry join entered the timeline, which rendered empty. The clustered control provides a negative case with no recorded merger. | `extend_native_assembly_events.py`, `registered-native-assembly-events.log` |
| Registered native perturber mass profile | Evolved a 64-particle clustered cosmological control with GADGET FOF, SUBFIND and the native merger-tree option. Followed two native descendant rows over three epochs and summed the actual masses of 11 bound members at each epoch. The profiles reach 1.77476 × 10¹⁴ Msun at the outer radius. A recorded image counter from the exact most-bound particle supplies each catalogue-center image. The clustered ICs are synthetic. | `extend_native_subfind_tree.py`, `verify_native_perturber_profile.py`, `registered-native-perturber-profile-verified.log` |
| Registered matched compact and extended encounters | Integrated the same 32 tracers in a harmonic host and two imposed equal-mass Plummer flybys. Both runs kept the same initial state, trajectory and output clock. All eight final bins had paired support, with maximum difference in bin mean velocity of 4.0947 km/s. The original attempt caught a harmonic Hessian shape error, which was fixed and checked against the exact Cartesian derivatives. | `extend_matched_perturbers.py`, `registered-matched-perturber.log`, `harmonic-hessian-regression.log` |
| Registered 3D stream and progenitor | Integrated a seeded 32-star Plummer cluster in a static Hernquist host with direct softened self-gravity. At 160 Myr the iterative self-binding census found 26 bound stars and six originally bound stars that were now unbound. The registered exporter plotted their measured positions in proper kpc. This synthetic control leaves the native stellar-catalogue intake unverified for this recipe. | `extend_stream_progenitor.py`, `registered-live-stream-progenitor-final.log` |
| Registered output cadence contrast | Computed RMS radius at all five saved native Plummer times. Withheld two snapshots and compared their saved values to interpolation from the three retained times. Errors were 6.21 × 10⁻¹¹ and 4.84 × 10⁻¹⁰ in native length units. The statistic and masks are preserved with the plotted arrays. | `extend_output_cadence.py`, `registered-output-cadence.log` |
| SIDM and OpenGadget3 | GADGET-4 and independent Linux OpenGadget3 each evolved 32-particle controls. The separate OpenGadget3 high-rate case changed 19 particle velocities with relative kinetic-energy drift of about 2 × 10⁻¹⁶. | `sidm-summary.json`, `opengadget-summary.json`, `opengadget-high-rate-summary.json` |
| Native SIDM event reconciliation | Matched the one saved event journal entry to the two exact participant counter increments across three snapshots. The diagnostic preserves all 32 fixed IDs, exposure increments, and the elapsed clock. The registered scattering-rate route requires a sealed native halo history and per-step guard evidence. | `extend_sidm_scattering.py`, `sidm-fixed-cohort.log`, `runs/.../sidm32-fixed-cohort/measurement.json` |
| Native SIDM elastic pair kinematics | Audited all 32 contiguous journal intervals, then matched each saved interval’s exact per-ID counter changes. The one event deflected IDs 14 and 32 by 100.42 degrees in relative-velocity space. Relative speed stayed 100 km/s, pair momentum residual was zero, and relative kinetic-energy residual was 2.35 × 10⁻¹⁶. | `audit_sidm_pair.py`, `sidm-pair-kinematics-summary.json` |
| Registered native SIDM halo histories | Fresh distributed-operator GADGET evolution used 24 DM particles plus eight seeded collisionless stars in explicitly synthetic compact ICs. Three snapshots supplied central density, saved phase classification, density/dispersion/anisotropy, stellar-axis alignment, kinetic heat flux and kernel diagnostics: 12 figures across six recipes. The dimensional isotropic cross section was 10⁻²⁰ cm²/g, with legacy normalization one. All original particles persisted. A separate scattering-rate figure retained both intervals with source-bound per-step guard and clock support. Rare-event counts were zero; mass-weighted optical-depth rates were 1.99002 × 10⁻⁹ and 1.05749 × 10⁻⁹ Gyr⁻¹. The tiny shell floor is three particles; the four-bin core-fit floor remains unchanged. These controls establish execution, not a resolved SIDM core, collapse or conduction law. | `extend_native_sidm_halo32_source_bound.py`, `native-sidm-halo32-source-bound-summary.json`, `native-sidm-halo32-guard-audit-summary.json`; `runs/.../registered-native-sidm-halo32-source-bound-*` |
| Registered matched SIDM/CDM controls | Replayed the same 32-particle IC bytes with the scattering operator enabled and disabled. The actual CIC phase check passed the explicit five-mode smoke floor. All 24 first-saved DM members matched exactly. Seven figures compare central density and shell density/variance, retaining four supported shell/epoch bins and three central epochs. The tiny cross section generated no scattering events. | `extend_native_sidm_cdm_pair32.py`, `native-sidm-cdm-pair32-summary.json`; `runs/.../registered-native-sidm-cdm-pair32/` |
| Registered bound-object retention | Measured native SUBFIND row zero at the first and last SIDM epochs. All 24 DM particles remained bound, with original bound-mass retention 1.0 and zero evaporation candidates. The source is a synthetic central object: it exercises the satellite binding/retention algorithm without establishing an orbiting satellite. The direct softened DM self-potential excludes stellar mass and host tides. | `extend_native_sidm_binding32.py`, `native-sidm-binding32-summary.json`; `runs/.../registered-native-sidm-binding32/` |
| Registered matched native lens response | Projected both actual native mass populations into common 8 × 8 CIC planes at three saved redshifts. Distances came from their original CLASS background, with a declared source redshift of 60. Nearest-pixel count support requires two particles. Six figures retained the supported surface-density and convergence differences, all zero in this no-scattering control. No H₀ estimator or posterior was supplied. The first adapter omitted transitive native input bindings; the verified adapter checks the original arrays and adds their complete unchanged bindings. | `extend_native_sidm_lens_pair32.py`, `verify_native_sidm_lens_pair32.py`, `native-sidm-lens-pair32-summary.json`; `runs/.../registered-native-sidm-lens-pair32-verified/` |
| Registered native centroid and ancestry history | Measured FOF-centered structures at all three SIDM epochs. The unique central SUBFIND descendant and its native FOF mapping establish the saved object chain. Stellar/DM centroid distances were 0.017715, 0.003517 and 0.016029 proper kpc. The figure retains those offsets and an empty native-merger lane. The native chain records no merger. | `extend_native_sidm_centroids32.py`, `native-sidm-centroids32-summary.json`; `runs/.../registered-native-sidm-centroids32/` |
| Native calibrated SIDM/BH response control | Completed 32 native evolutions, each with 31 DM particles and one gravitating BH. Independently varied SIDM amplitude and initial BH mass; measured final BH mass from saved snapshots. Two training realizations and two independent held-out realizations at six parameter points passed the declared local-response checks. The scaled response has singular values 100 and 0: the mass direction is resolved and scattering remains unconstrained. The covariance uses an explicit 10,000 Msun² regularization floor. This verifies a null/identity response and registered degeneracy figure, not baryonic feedback, an astrophysical cross-section constraint, or posterior coverage. | `run_native_sidm_compact_mass_response32_retry.py`, `native-sidm-compact-mass-degeneracy32-summary.json`; `runs/.../registered-native-sidm-compact-mass-degeneracy32/` |
| Registered SIDM stellar-cohort response | Read the eight seeded stars through all three actual matched SIDM/CDM snapshots. Fixed a projection axis from the first baseline sample and used the same baseline origins, peculiar bulk velocities and two proper-arclength bins in both branches. Bin populations were 2/6, 2/6 and 3/5; all six bin/epoch samples passed the explicit two-particle floor. The registered exporter produced density and parallel-peculiar-velocity contrasts, both zero. These stars are an explicitly artificial cohort, not detected tidal debris. This tests the measurement/contrast/export path and leaves native stream finding, stellar birth/escape history, resolved perturber profiles and physical attribution unproven. | `extend_native_sidm_stellar_cohort32.py`, `native-sidm-stellar-cohort32-summary.json`; `runs/.../registered-native-sidm-stellar-cohort32/` |
| Native cooling | Compiled cooling numerical control passed | `cooling-summary.json` |
| Gas/star-formation configuration | A 64-gas-particle SH03-enabled native control saved three epochs | `source-summary.json`; `runs/.../sh03_equilibrium-retry2` |
| Native BH accretion configuration | Gas plus an explicitly supplied BH evolved through three saved epochs | `source-summary.json`; `runs/.../bh_accretion-retry2` |
| Native BH pair separation | A separate 64-gas, two-seeded-BH control saved both exact IDs at three epochs. The exporter measured periodic minimum-image separations of 0.14285714, 0.14286000 and 0.14286286 proper kpc. This is a positional measurement; it does not establish binary binding or a merger. | `extend_bh_pair.py`, `registered-native-bh-pair.log`; `runs/.../registered-native-bh-pair/pair-separation.json` |
| Registered matched BH DM response | Evolved byte-identical ICs containing 31 DM particles and one seeded BH, with the native BH source enabled and disabled. Both branches retain the same gravitating BH; there is no gas or BH dynamics. The final FOF objects match all 31 DM IDs reciprocally, and the five-mode phase audit passes. Two density/shape figures retain both proper shells. All 64 paired bootstrap draws supplied complete density and shape vectors. Both measured responses and their paired difference errors are zero. This verifies a gas-free negative control, not a feedback-induced halo change. The covariance intake needed a four-GiB allocation for its sealed structure metadata. | `run_native_bh_vacuum_pair32.py`, `export_native_bh_vacuum_pair32.py`, `native-bh-vacuum-pair32-summary.json`; `runs/.../registered-native-bh-vacuum-pair32-response/` |
| Native gas-free pressure-entropy initialization | The original 32-particle disabled-BH branch crashed in the pressure-entropy initializer with global gas count zero. A global-count guard now skips the nonexistent gas tree. Both one-rank branches then saved three snapshots. A separate two-rank run preserved every exact ID and agreed with the one-rank states at the recorded tolerances. The native source digest and executable hash match their actual build signature. The updated 37-file overlay passes all eight archive/replay/install checks. | `native-bh-vacuum-pair32-entropy-crash.log`, `verify_native_vacuum_rank2.py`, `native-vacuum-rank2-summary.json`, `native-overlay-vacuum-guard-regression.log` |
| monofonIC | Generated the selected 128-particle two-species IC load from the fresh LCDM bundle | `monofonic-summary.json` |
| MUSIC | Executed uniform, nested and periodic transfer-normalization controls | `zoom-retry.log`; `runs/.../music-controls/native-transfer-suite.json` |
| Production 1LPT MUSIC staging | A 64 Mpc/h, level-5 parent used the actual retained CLASS bundle. The first 20 Mpc/h request failed transfer-coverage preflight. The initial covered native run returned zero but omitted gas positions/IDs; its structural audit rejected it. The shared template now explicitly enables `do_SPH` before MUSIC plugin initialization. With identical transfer bytes and only that native setting changed, a fresh run produced 32³ gas plus 32³ DM particles and passed mass, exact-ID, momentum, barycenter and species-velocity checks. The formerly skipped refinement-staging regression and 27 related checks passed. Density/velocity-mode and parent-phase validation remain incomplete; this is not resolved zoom convergence. | `native-music-gas-fix-summary.json`, `native-music-staging-fixture-summary.json`, `native-music-gas-staging-regression.log`, `native-music-fixture-replay-regression.log`; `runs/.../native-music-staging-fixture-gas/` |
| Nested MUSIC → GADGET | 2,744 gas + 2,744 refined DM + 5,265 boundary particles saved three epochs with finite coordinates | `zoom-evolution-summary.json` |
| Rockstar | Executed a periodic finder on the explicit synthetic particle fixture | `finder-summary.json`; finder native bundle |
| Consistent Trees | The sparse Rockstar example correctly skipped; a separate synthetic 128-object, two-epoch catalogue completed tree construction | `finder-summary.json`, `tree-summary.json`, `tree.log` |
| VELOCIraptor | Executed on the small supplied particle fixture; its resulting object list is empty | `finder-extra.log`; `runs/.../finder-fixture2/velociraptor/` |
| DisPerSE | Native extraction and skeleton projection executed | `disperse-retry.log`, `disperse-projection.npz`; native extraction cache/receipts |
| Grackle | Fresh CMake build and actual native chemistry/cooling example; finite temperature, pressure, gamma and dust temperature | `grackle-configure.log`, `grackle-build.log`, `grackle-execution.log` |
| GADGET + Grackle H₂/HD | Built and linked the native 64-gas control against shared precision-64 Grackle 3.4.1. The standard build required two missing internal header guards to be registered. GADGET wrote the initial molecular thermodynamics snapshot, then halted at the H/He/D conservation check on the first cooling update. The gas-only control used an inactive synthetic stellar-return table to satisfy build-time requirements. | `grackle-shared.log`, `grackle-cloud-yield-rejection.log`, `grackle-cloud-macro-rejection.log`, `runs/.../grackle-cloud-native-64-retry/native.log` |
| Registered native primordial cloud | Corrected the Grackle mass-ratio handoff and converted the solver normalization back to the transported H/He/D reservoirs before the existing conservation check. The rerun saved three epochs. Across 64 matched gas IDs, the final relative drift in any nucleus reservoir was at most 3.02 × 10⁻¹². Molecular fractions and cumulative radiated energy changed, and all 64 points and one shape-supported centroid entered the registered plot. The controlled gas box used an inactive synthetic stellar-return table and contained zero stars. | `verify_grackle_cloud.py`, `registered-grackle-cloud-verified.log`, `grackle-cloud-nucleus-audit.json` |
| Registered cooling and birth timeline | Measured 128 positive radiative-loss intervals across the 64 exact gas IDs in the evolved Grackle snapshots. The cooling-to-free-fall ratios ranged from 782 to 1,010. All events were excluded at a ratio cut of one. A separate broad cut of 10⁶ exercised the plotted cooling branch. The native journal census contained zero births, so the birth lane is empty and the host-ancestry join remains untested with positive data. | `extend_cooling_timeline.py`, `registered-grackle-cooling-timeline.log` |
| Registered primordial cloud support gate | Read a saved SH03 gas snapshot with Grackle network disabled and ran the registered morphology exporter. It rendered the unsupported status with zero plotted gas points or cloud centroids. | `extend_cloud_negative.py`, `registered-sh03-cloud-negative.log` |
| Agama | Plummer force at (1,0,0), giving Fx ≈ −0.35355339 in the declared model units | `agama-calculation.npz`, `optional-summary.json` |
| Gala | Actual spray adapter returned 36 phase-space samples | `gala-calculation.npz`, `streams-optional-summary.json` |
| StreaMAX | Actual static-host reference returned 20 stream samples | `StreaMAX-calculation.npz`, same stream summary |
| FSPS | A 1 Gyr population produced a 5,994-element spectrum and two selected filter magnitudes | `fsps-calculation.npz`, `optional-summary.json` |
| emcee | Eight walkers × ten steps for a toy Gaussian likelihood; saved chain/log probabilities | `emcee-calculation.npz` |
| dynesty | Deliberately short toy Gaussian nested sampling; saved samples/evidence estimates | `dynesty-calculation.npz` |
| cloudpickle | Serialized/deserialized a calculation and recovered f(3)=9 | `cloudpickle-calculation.npz` |
| SBI | Trained a small density estimator for two epochs on 32 toy forward samples | `sbi-density-estimator.pt`, `streams-optional-summary.json` |
| Registered SBI rank and coverage calibration | Executed 64 independent initial-state experiments with 32 massless tracers each in an analytic harmonic field, then trained the actual pinned SBI summary posterior for three epochs. The forward-domain receipt is restricted to that defined numerical control: fixed initial tracers, Cartesian velocity, Gaussian noise, complete selection and zero model discrepancy by construction. Before training, 192 endpoint/midpoint force and exact-oscillator comparisons passed, with maximum velocity error 4.27 × 10⁻¹⁴ km/s. Parent/lineage splits withheld 17 independent test groups; truth was read only after model freezing. The unchanged 90% credible-coverage criteria rejected calibration: its 95% binomial interval was [0.2781, 0.7702]. The rank and coverage figures preserve that rejection. No native stream or observational acceptance is implied, and deployment stays disabled. | `run_harmonic_sbi_calibration32_device.py`, `harmonic-sbi-calibration32-summary.json`; `runs/.../registered-harmonic-sbi-*` |
| Independent harmonic SBI calibration | A separate study fixed new training settings before evaluation and drew 1,536 fresh initial-state ensembles, each with 32 massless tracers. All 4,608 exact oscillator comparisons passed; the largest velocity residual was 5.69 × 10⁻¹⁴ km/s. The summary model trained for 100 epochs / 1,100 optimizer steps. Its 613 independent held-out groups passed the original calibration criteria: the 95% coverage interval was [0.9030, 0.9460] for nominal 90% credible regions. The specified Bonferroni-adjusted coverage and SBC tests also passed. This accepts this synthetic analytic numerical domain only. | `run_harmonic_sbi_calibration32_independent.py`, `harmonic-sbi-calibration32-independent-summary.json`; `runs/.../registered-independent-harmonic-sbi-*` |
| Original SBI out-of-domain gate | After the independent summary calibration passed, supplied 32 synthetic observed velocities of 1,000 km/s with the same declared Gaussian covariance. The original OOD reducer measured squared Mahalanobis distance 126,188.77 and outside-range fraction 1/7, exceeding the frozen thresholds 25 and 0.1. It reported OUT_OF_DOMAIN and wrote zero posterior draws; the registered support figure rendered. | `run_harmonic_sbi_ood32_independent.py`, `harmonic-sbi-independent-ood-summary.json`; `runs/.../registered-independent-harmonic-ood/` |
| Independent graph candidate calibration | The predeclared message-passing candidate trained on the same fresh dataset/split, freezing after 96 epochs / 1,056 optimizer steps. On the same 613 held-out groups its 95% coverage interval was [0.8385, 0.8937], failing the unchanged coverage-equivalence condition and adjusted coverage test. Its original ranks and coverage were exported, but graph-baseline gain was not evaluated because both models must pass calibration. This campaign's held-out set was not reused for tuning or a retry. | `run_harmonic_sbi_graph32_independent.py`, `harmonic-sbi-graph32-independent-summary.json`; `runs/.../registered-independent-harmonic-graph-*` |
| Expanded predeclared harmonic comparison | A separate fixed study used 4,096 fresh initial states, 32 tracers each, and 12,288 closed oscillator checks; maximum velocity residual was 6.40 × 10⁻¹⁴ km/s. Training budgets were fixed before the new held-out truths: up to 300 epochs / 15,000 optimizer steps, with validation early stopping. Both frozen models failed the unchanged calibration criteria on 899 held-out groups. The summary 95% coverage interval was [0.8425, 0.8881]; graph interval [0.8508, 0.8953], with a rejected coverage test. The original ranks and coverage figures are exported; paired graph gain and deployment remain blocked. No tuning or retry uses these held-out cases. Earlier accepted summary calibration applies only to its original campaign. | `harmonic-sbi-expanded-predeclared.json`, `harmonic-sbi-calibration32-expanded-summary.json`, `harmonic-sbi-graph32-expanded-diagnostics-summary.json`; `runs/.../registered-expanded-harmonic-*` |
| lenstronomy | Singular-isothermal-sphere deflection calculation returned the expected unit deflection | `lenstronomy-calculation.npz` |
| Registered lens time-delay posterior | Rendered a seeded 32 × 32 synthetic two-image lens with lenstronomy and sampled the joint image and delay likelihood with 12 walkers. Retained 840 draws. Mean time-delay distance was 2302.17 Mpc and mean H₀ was 71.74 km/s/Mpc with sample standard deviation 2.61 km/s/Mpc. The compact control fixes the source shape and assumes an SIS lens and a fixed expansion shape. Chain convergence and independent coverage remain open. | `extend_lens_mcmc.py`, `registered-lens-imaging-mcmc.log` |
| pyrecon | Reconstruction executed on 512 synthetic tracers and 2,048 randoms | `bao-summary.json`, `bao.log` |
| pycorr/Corrfunc | Actual pair counting and correlation estimation returned finite selected bins | Same BAO summary/log |
| PyVista | Rendered 32 supplied points offscreen to a PNG | `optional-summary.json`; `runs/.../pyvista.png` |
| Blender species scenes | DM, gas, stars and BH each built an editable scene and encoded a two-frame MP4; first frames decoded and retained | `blender-native.log`, `blender-*-frame.png`; `runs/.../synthetic1028/render-*` |
| Connected analysis/export | Kinematics, hydro, maps, four species/temperature movies, XDMF, planetarium and combined Blender scene/movie completed from a corrected synthetic fixture | `analysis-final.log`; latest `synthetic-corrected/results/workflow_*.json` |
| Registered power, HMF and BH recipes | Produced two linear figures, three evolved power figures, three BH figures, and nine native analysis figures from a six-cluster 64-particle control. The clustered control supplied three saved epochs for differential HMF, cumulative HMF and evolved power. | `registered-linear.log`, `registered-evolved.log`, `registered-bh-retry.log`, `clustered-hmf-retry.log` |
| Matched HMF ratio | An independent execution of the same 64-particle GADGET input reproduced all six halos in the same occupied mass bin. The 4³ mesh has five common low-k modes and initially failed the default 20-mode gate. A separate smoke-only configuration explicitly requires five modes and measured coherence 1.0. The occupied bin has a ratio of 1.0. | `registered-matched-hmf.log`, `registered-matched-hmf-small-mesh.log` |
| Native paired power ensemble | Two independent N-GenIC parent seeds, each evolved twice with identical physics and IC bytes within the pair. The 64-particle attempt had no spectral bins inside its measured phase range. Four new 512-particle executions supplied two common bins: ratio 1, residual 0. The estimator retains signed, shot-noise-subtracted powers and computes the full sample covariance from the two parent realizations. Two parents do not calibrate that covariance. | `extend_power_ensemble_512.py`, `retry_power_ensemble_512.py`, `registered-power-ensemble-512-retry.log`; `runs/.../registered-native-paired-power-512-ensemble/ensemble-power.json` |
| Mock distance ladder | Generated 84 explicit mock indicators across an anchor host, calibrator host and Hubble-flow host. The complete-sample joint fit returned H₀ = 67.86 ± 6.58 km/s/Mpc for inputs generated at 70. The interval figure uses the fit uncertainty, with no posterior samples invented. The separate magnitude-truncated attempt failed optimizer precision checks. | `extend_mock_ladder_complete.py`, `registered-mock-ladder-complete.log`, `registered-mock-ladder.log` |
| Published BAO and actual short posterior | Ingested the 13 shipped DESI measurements and their original full covariance. A fresh run of the frozen CLASS binary measured drag sound horizon 147.070889 Mpc. The background comparison rendered all original rows. Eight emcee walkers × 20 steps, discarding four, supplied 128 retained two-parameter distance-dilation draws and their registered CDF figure. The sampler reports `CHAIN_TOO_SHORT`. This is neither an H₀ posterior nor a CMB likelihood. | `extend_published_bao.py`, `registered-published-bao.log`; `runs/.../published-bao-posterior-campaign/bao/` |
| Registered native host census | The six-cluster control supplied six measured FOF groups. The reliability and rare high-redshift figures use its saved group counts, masses, configured particle cut, and periodic volume. | `registered-hosts.log`, `extend_native_graphs.py` |
| Registered Gala stream figures | A computed Hernquist-potential spray supplied 130 tracers from one tail, with equal display mass weights. The coordinate and geometry exporters produced 19 figures across 13 recipes. Nine of 16 local bins had support. Density and track power remained masked, and bifurcation entries are candidate diagnostics. | `extend_stream_figures_longer.py`, `registered-gala-longer.log` |
| Registered stream spectra and actions | Reused the same sealed Gala tail with the example minimum of eight supported spectral bins. Nine bins qualified and both spectra plotted nine modes. A separate isochrone Gala control supplied 36 tracers. The retry resolved all three actions and angles for 27 tracers, with a maximum absolute action difference of 4.73 × 10⁻¹³ proper kpc km/s against analytic isochrone actions on the supported subset. Nine remained unresolved. | `registered-gala-power.log`, `registered-gala-actions-retry.log` |
| Registered assembly histories | Eight synthetic orbit snapshots supplied measured DM aperture mass, spin, and shape. The harness declares synthetic ancestry and passes the sealed identity graph to the standard assembly runner. The transition fit recorded no supported breakpoint. Concentration and stellar histories rendered with their missing values masked. | `extend_assembly.py`, `registered-orbit-assembly-retry.log` |
| Registered structure and T-web | Recomputed halo structure, environment, and environment history from eight retained synthetic orbit snapshots. Seven structure recipes and the filament map plotted supported entries. Orientation twist, splashback, spin and shape alignment, and web evolution ran with zero supported entries. | `extend_structure_web.py`, `registered-orbit-web-final.log`, `registered-orbit-splashback-final.log` |
| Conditional concentration fit | Remeasured the same 256-particle synthetic snapshot on finer radial shells. With an explicitly artificial reference-density definition, eight measured shells supplied a conditional NFW fit: R_delta = 1.6322 proper kpc, r_s = 0.4472 proper kpc, concentration 3.6501. χ² = 43.23 for six degrees of freedom indicates poor fit quality. The particle fixture was not generated as an NFW halo; optimizer completion does not establish model adequacy. | `extend_concentration.py`, `registered-concentration.log`; `runs/.../conditional-concentration-input/structure.json` |
| Native stellar light | A native SH03-only control evolved 64 gas particles and one seeded SSP through three saved epochs. The hydro audit passed. FSPS produced a spectrum and supported luminosity and SDSS g−r color at each epoch. Bolometric luminosity was about 1.28 million solar luminosities. The run supplied the stellar particle at initialization. | `extend_native_ssp.py`, `registered-native-seeded-ssp.log` |
| Native stellar and Vmax abundance | Added 12 initial stellar tracers to the six-cluster 64-DM control, preserving total IC gravitating mass, and evolved three epochs. Native SUBFIND supplied six stellar-bearing subhalos. All six enter the stellar-mass histogram. A second Vmax request includes their measured high speeds; the first retained request correctly reports six objects above its range. These extreme synthetic clusters are numerical fixtures, not credible galaxy populations. Missing formation times and SFR stay unavailable. | `extend_native_abundance.py`, `registered-native-abundance.log`; `runs/.../registered-native-seeded-stellar-abundance-wide-vmax/` |
| Mock expansion-method comparison | Executed eight frozen fits: local lightcone-distance and distance-ladder estimators across four independent measurement-noise trials on fixed mock geometry. A separate post-fit stage measured recovery and the full cross-method covariance. The 20-realization accuracy budget remains unmet. These are not four independent cosmological volumes. | `extend_mock_method_comparison.py`, `registered-mock-method-comparison.log` |
| Physical stream posterior prediction | Integrated eight computed Gala tracers through an isochrone forward model, sampled one mass parameter with eight emcee walkers × 20 steps, then generated eight Gaussian predictive replications. The registered posterior and predictive figures completed. The chain remains too short; the eight-draw predictive tail estimate of zero is not calibrated evidence of model rejection. Positional observables are explicitly normalized by 1 proper kpc. | `extend_stream_predictive_retry.py`, `registered-physical-stream-predictive-retry.log` |
| Compact native BH binary attempt | The isolated preflight was rejected by the supported comoving-clock requirement. An overly strict step request then failed integer time resolution. A separate standard-step comoving run recorded one binary capture and 9,896 hardening rows before the native CPU time limit stopped evolution. The process returned zero and wrote a restart, but it did not reach the final requested snapshot. The journal audit also rejected the absent stellar event counters in that BH-only build. No registered merger or recoil proof is credited. | `extend_native_bh_merger_standard_step.py`, `registered-native-bh-merger-standard-step.log` |
| Audited native BH merger and recoil | A short comoving run used 64 gas particles and two black holes with physical masses of 1 million and 500,000 Msun. The gravitating masses started 10% higher. The combined SH03 and BH build saved all three requested snapshots and passed the hydro journal audit. The solver recorded one merger and a 141.93 km/s composite kick. The registered timeline retained one event and the recoil plot retained two supported samples. This near-threshold setup tests the configured subgrid merger operator. | `extend_native_bh_merger_sh03_short.py`, `registered-native-bh-merger-sh03-short.log`, `export_bh_short_recoil_retry.py`, `registered-native-bh-recoil-short-retry.log` |
| Delayed-return source preflight | The shipped example yield request rejected six source rows against its 2% ejecta/remnant mass-closure tolerance. The sealed diagnostic records the original rows and errors. This remains a separate input limitation for delayed-return evolution. | `runs/.../stellar-model-preflight.json` |
| Registered hydro histories | Three SH03 snapshots supplied the SFR, delayed-return ledger, and metals history exports. The ledger retains zero events. The gas phase figure rendered with zero supported temperature samples because the snapshots lack the required element composition. The metals figure preserves this missing gas composition separately from the ledger. | `registered-gas-phase.log`, `registered-hydro-histories.log` |
| Additional visual products | All-sky, velocity flow, lens map, multifield movie, timeline wall and void explorer executed | `extra-visual-summary.json`, `extra-visual-retry.log` |
| Metallicity | Mass-weighted metallicity projection completed after supplying full dimensionless field metadata in a separate fixture | `metallicity-summary.json`, `metallicity-complete.log` |
| Regression suites | 74 standalone + 748 verification tests; no errors/failures, five fixture-related skips | `retained-tests-summary.json`, `retained-tests.log` |
| Clean Ubuntu base installation | Started a fresh Ubuntu 24.04 ARM64 container from the recorded base image, installed apt prerequisites and ran the setup script with base profile and `--engines none`. A newly created venv completed all seven 32-particle quick steps in 1.668 seconds. All 11 output hashes, installation receipt, frozen dependencies and public command help were checked. The final wrapper incorrectly expected aggregate `PASS`; the actual result was `COMPLETE` with every step passing. A read-only verifier corrected that assertion without rerunning or rewriting the original results. This does not cover default native-engine builds, optional profiles or WSL. | `clean-ubuntu-base-summary.json`, `verify_clean_ubuntu_base.py`, `clean-ubuntu-base.log`; `clean-ubuntu-base/` |
| Clean Ubuntu default engines | The initial default build succeeded for CLASS/AxiCLASS but rejected monofonIC because serial FFTW lacks MPI support. Setup now includes `libfftw3-mpi-dev`. A separate container and new runtime installed the base environment and built all three default engines successfully. The public native resolver verified each declared source file and executable hash. Original failed build/log retained. This is ARM64 Ubuntu 24.04, not WSL or all optional profiles. | `clean-ubuntu-missing-fftw-mpi-summary.json`, `clean-ubuntu-default-fixed-summary.json`, `clean-ubuntu-default-fixed.log` |
| Clean Ubuntu native 32-particle lab | The portable native profile compiled GADGET and passed all eight steps. Its actual isolated Plummer evolution saved three epochs with 32 exact particles each, at times 0, 0.02 and 0.04 in the declared native analytic clock. A separate container verified all artifact hashes, full native campaign source/build/output closure, particle counts and finite positions. | `clean-ubuntu-native32-summary.json`, `clean-ubuntu-native32-verification.log`; `clean-ubuntu-base/native-32/` |
| Public command help | All 180 directly parsed handlers returned successful live help in isolated bounded subprocesses. Twelve delegated/interactive handlers were excluded explicitly. The source reader retained all 763 argument declarations across 192 handlers and now labels subcommands rather than merging repeated flags. These are help/documentation checks, not scientific command executions. | `command-help-current-summary.json`, `command-reference-subcommands-summary.json` |
| Python inventory | The generated reference now includes 262 application/reference-generator/optional-lab Python files, including the current HMF additions. The original import inventory checked 185 ordinary/logical targets, with the Blender-only import exercised separately inside Blender. Optional lab checks and fresh calls have their own receipts. New files in the source map are not silently counted as executed. | `import-summary.json`, `blender-import.log`, `file-coverage.json`; optional run `run.json` files |
| Portable optional lab | Seven numerical steps passed on 32 particles through the browser in about two seconds. The 1,024-particle visual run passed in about two seconds. A separate 32-particle GADGET self-gravity profile passed all eight steps. The local authority/path tests passed. All nine notebook code cells executed in a real Jupyter kernel. | `optional-lab-*` logs; `optional-walkthrough-executed.ipynb`; external `research2-runtime/optional-lab/` outputs |
| Fresh optional-library profile | The downloadable lab now has a libraries profile: seven quick measurements plus twelve isolated optional calculations. Agama, Gala, StreaMAX, FSPS, emcee, dynesty, cloudpickle, SBI, lenstronomy, PyVista, pyrecon and pycorr/Corrfunc all calculated successfully. The complete CLI run passed all 19 steps in 38.871 seconds. A separate actual HTTP launch/status/artifact run also passed all 19 steps, checked every artifact hash and served the FSPS PNG. Six focused tests cover confinement, authority, inventory claims, missing-site-package detection and continuing after a backend failure. Each check has arrays, scope, measured assertions and its own process log/receipt. Native engines remain separate profiles/campaigns. | `optional-libraries-complete-32.log`, `optional-library-profile-http-summary.json`, `optional-library-profile-complete-regression.log`; external `optional-lab/libraries-complete-32/` and `optional-lab/libraries-http-32/` |
| Graph registry | All 102 declared producer/render routes resolve to callables | `graph-routes.json`; complete scientific recipe execution is not implied |

`runs/.../` in the table means `research2-runtime/runs/readme-smoke-20261104/`.
Native snapshots and source receipts remain there; compact summaries and scripts live
under `research2-runtime/evidence/readme-smoke-20261104/`.

## What the small run does and does not demonstrate

The visual fixture explicitly contains synthetic DM, gas, stars and BH particles.
Those particles exercise readers, measurements and exports; they were not formed by
the tiny cosmological run. FSPS and stream inputs are explicit separate calculations.
The synthetic background used by visual fixtures is also not a precision cosmology.

The tiny native star-formation-enabled control establishes execution with that model
enabled; it does not by itself show a resolved stellar population or galaxy. The BH
control establishes native evolution with a BH source configuration; accretion,
feedback and conservation require inspecting the saved fields/journals and appropriate
numerical controls. The WDM evolution explicitly excludes thermal-velocity sampling
because 64 particles are inadequate for its statistical gate.

Some halo catalogues are empty at this resolution. Rockstar process success and
VELOCIraptor process success do not prove resolved objects or finder agreement.
Consistent Trees' synthetic catalogue control establishes tree execution, not realistic
merger-history recovery. The nested MUSIC control uses zero-transfer normalization
inputs, so it tests the handoff/evolution path rather than resolved galaxy assembly.

The initial five regression skips are listed in `retained-tests-summary.json`. The new native
gravity matrix supplied the measured receipt for the skipped control class, and all seven
checks in that class subsequently passed. The historical fixture gaps were later supplied; the complete rerun below has no skips.

At this milestone, the coverage table recorded 90 recipes with supported plotted entries, eleven
rendered recipes with zero supported samples, and one recipe awaiting direct execution
evidence. Many
require matched simulations, host/member selections, source journals, observations,
calibrated encounters or inference/convergence products. Their individual routes and
coverage gaps appear in the graph coverage file. Creating generic arrays under those
names would not test their scientific producer chains. Likewise, Python imports and
observed regression calls do not establish execution of every branch in every file.

## Failures encountered and corrected in the disposable harness

Earlier attempts and their workflow records remain visible; they are not counted as
successful outputs. Corrections included:

- Launch Blender-only helpers inside Blender, where `bpy` exists.
- Fill template configuration placeholders, including the HMF member floor used by
  scene-history selection. Use at least two frames per saved interval for animation.
- Use a non-star-forming gas fixture for the thermal-temperature map; nonzero SFR and
  effective-EOS exclusions prevented the original temperature export.
- Use a SUBFIND link-neighbor count below the smallest halo membership in the synthetic six-cluster case. The inherited 20-neighbor setting failed for its 10–11 particle groups.
- Supply complete dimensionless metallicity unit metadata, including length, mass
  and velocity exponents, in a separate fixture before publishing new maps.
- Preserve MPI wrapper executable names when pinning tools; those aliases select
  language-specific compilation behavior.
- Use a short Linux run path for OpenGadget3. The long path overflowed the native output-list filename buffer.
- Quarantine a Finder `.DS_Store` file from the pinned OpenGadget3 source tree. It was
  outside the complete retained-source manifest; source-integrity checks correctly
  rejected the extra file. Its bytes remain in the campaign evidence folder.
- Increase the paired-power control from 4³ to 8³ particles when its native power
  bins and measured IC phase range have no overlap. Retain the unsupported smaller
  attempt. A typo in the second disposable request's schema was corrected in a
  separate retry script; all four native outputs were reused unchanged.

The magnitude-truncated mock distance-ladder fit remains a failed attempt: BFGS
reported precision loss and the producer rejected its gradient. The successful
complete-sample example separately sets all magnitude limits to 100 and declares
the complete likelihood. This choice is appropriate for that effectively uncut
mock sample; it is not a recovery rule for a magnitude-selected observational sample.

No physical-input checks were weakened to make the fixtures pass. Published native
research runs were not modified.

## Reproduce or remove this campaign

Activate the installed environment, read the retained harness before running it, and
choose fresh destinations. Several harnesses intentionally reject existing output
folders. Reusing the same campaign path is not an overwrite/replay interface.

The ordinary public verification commands remain:

```bash
python -m unittest discover -s tests -v
python verification.py -v
python commands.py audit-graph-producers --output /absolute/path/to/new-graph-audit.json
python commands.py results /absolute/path/to/RUN --verify
python scripts/build_reference.py
```

This campaign's `retained_tests.py` additionally keeps temporary fixtures and records
Python call events; many regression cases intentionally use mocks or invalid inputs.
Its retained directory is not a collection of accepted scientific runs.

The campaign run and evidence directories can be removed after review. Native build
caches created outside them and the explicitly named Linux container have separate
ownership and should be reviewed before cleanup. Existing environments, unrelated
research runs, source tables and licenses are not disposable campaign outputs.


The refreshed walkthrough executed all nine code cells against the new
32-particle optional run in `optional-lab/traceability-32`. The last cell
verified the selected impact figure's image, arrays and manifest hashes,
displayed the plot, and printed its producer, renderer, units and array ranges.
The notebook execution used the retained optional notebook tools outside the
base environment. Its executed copy is
`optional-walkthrough-traceability-executed.ipynb` in the evidence directory.

The expanded notebook subsequently executed all ten code cells against the
actual 19-step HTTP libraries run. Its new cell checked the backend receipt
and array hashes, printed their shapes/types and scopes, and displayed the
calculated FSPS spectrum. The final cell verified and displayed the retained
OOD diagnostic. See optional-walkthrough-libraries-summary.json and the
executed optional-walkthrough-libraries-executed.ipynb.

The walkthrough at this milestone had 132 cells, including eleven equation explanations
and ten executable code cells. A fresh execution of all ten code cells selected
the native SIDM/BH response figure and verified its original arrays, image and
manifest. The new response explanation covers the Jacobian, Cholesky whitening,
parameter scales, singular values and unresolved modes. See
`optional-walkthrough-local-response-summary.json` and its executed notebook.

An earlier regression refresh ran 825 tests in 271 seconds. It recorded
819 passes, five skips and one native overlay mismatch. The distributed
full-file overlay omitted the current header-guard changes and chemistry
normalization adapter. Its refresh includes the required primordial-network
header. All eight tests in NativeSSPBirthTests passed after that repair,
including byte-for-byte archive comparison, installation and rejection of
unbound local edits. At that point the complete suite had not yet been repeated after the archive
repair. See retained-tests-refresh-summary.json and
regression-refresh-resolution-summary.json for the original result and repair
verification. The skipped native campaign class was subsequently supplied
the fresh 18-case, eight-comparison campaign receipt, and all seven of its
checks passed in 3.629 seconds. Four other fixture skips remain. See
native-matrix-current-summary.json and native-matrix-current-regression.log.

The preceding isolated suites completed on matching source hashes: 93
application tests, 756 main verification tests and six optional-interface tests.
Together they recorded **854 passes, one skip, no failures and no errors**.
The remaining skip needs a specific native MUSIC staging fixture. The fresh
`native-matrix-final-source` campaign supplied all 18 cases and eight comparisons;
its seven receipt checks and three particle/restart fixture checks passed within
the complete verification suite. See `retained-tests-current-summary.json`, its
three component receipts and `python-execution-current.json`.

The first combined-process attempt aborted with duplicate OpenMP runtimes after
loading optional libraries together. The isolated suites use separate Python
processes without `KMP_DUPLICATE_LIB_OK`. The first isolated verification attempt
then rejected the earlier native receipt because `commands.py` had changed.
A new native campaign, followed by the complete verification rerun, resolved
that source mismatch. Both failed attempts remain retained. The final merged
summary verifies that its Python source hashes still match the files tested.

After fixing the 1LPT gas-displacement template and supplying the production
MUSIC parent fixture, a new complete verification run passed all 756 tests
in 271.397 seconds with no skips. The application suite passed 93 tests;
the optional interface suite passed seven, including the new archive-provenance
selection check. The merged result is **856 passes, zero skips, zero failures
and zero errors**, with matching implementation hashes across all three suites.
See `retained-tests-music-complete-summary.json` and
`python-execution-music-complete.json`. The optional atlas now imports a
completed merged trace and its exact source pins instead of defaulting to the
older regression refresh. Earlier attempts and skips remain historical records.

A compiled normalization audit exercised 432 combinations of hydrogen,
helium, deuterium, HD fraction and solver mass. It recovered the requested
nucleus reservoirs with maximum relative error 2.22 × 10⁻¹⁶ and rejected
an invalid solver reservoir. The source and binary hashes are in
grackle-normalization-grid-summary.json.

The new native SIDM history exposed an unsupported-rate export bug: NaN
summary values could not be written into finite JSON metadata. The exporter
now uses null for absent metadata and preserves numerical NaNs and false
support masks in the arrays. A regression executes the actual interval
estimator, numeric writer and figure exporter for both complete but
physically unsupported and incomplete cohorts; only the native file intake
is substituted. Both cases passed. The subsequent source-bound native halo
run exported its actual rates successfully. See sidm-rate-export-regression.log.
Earlier failed smoke attempts retain their normalization and SUBFIND-neighbor
rejections. The final 32-particle request uses 16 density neighbors so its
SUBFIND target fits inside its 24-particle DM population.

The actual SBI campaign found that the pinned backend asks every embedding
for its first parameter's device. The deterministic summary encoder had no
parameters and raised StopIteration. It now exposes an empty frozen device
marker, adding zero trainable values. The focused regression checks the
unchanged seven numerical summary outputs and device/dtype propagation.
The new actual backend training and held-out calibration then completed.
See summary-embedding-device-regression.log and harmonic-sbi-calibration32-device.log.

The five module-level pytest functions are collected separately from unittest.
All five passed with no skips in 1.96 seconds, covering empty catalogue shards,
central-point potential persistence and portable Blender scene receipt integrity.
The complete recorded total is therefore **861 passing checks**: 856 unittest
checks plus five pytest functions. See `pytest-functions-summary.json` and its
JUnit XML. These fixture checks do not add native Blender rendering coverage.

## Final source cleanup and external log ownership

The old in-tree `.venv`, pytest/bytecode caches, generated `sbi-logs`, Finder
metadata and duplicate `README.html` were moved to external recovery storage.
The manifest is `main-folder-cleanup-summary.json`; required sources, physical
tables, licenses, Git provenance and verification tests were retained. The
optional teaching examples remain in `optional_tests/`.

The optional SBI worker now supplies an explicit external TensorBoard writer.
The library runner hashes its nested training logs as run artifacts. A fresh
complete libraries run passed all 19 steps in 43.974 seconds and verified all
51 artifact hashes, with no source-tree `sbi-logs` directory. See
`final-clean-libraries-summary.json`. The earlier 861 regression checks retain
their original source revisions; the final cleanup rerun has separate receipts.

The final post-cleanup rerun passed all 756 main checks in 275.684 seconds,
all 93 application checks, all seven optional-interface checks and all five
standalone pytest functions: **861 passes, zero skips, no failures or errors**.
Source pins were rechecked after completion. See
`retained-tests-clean-final-summary.json`, `python-execution-clean-final.json`
and `pytest-clean-final-summary.json`. These checks use the stated fixtures and
mocks; they do not establish every physical model's scientific convergence.

## Combined optional checklist

The new `combined` profile runs the seven quick measurements, one actual native
GADGET 32-particle control and twelve optional-library calculations in one
20-step checklist. The first HTTP-launched run passed in 75.270 seconds. A
second run with the expanded native diagnostic display passed all 20 steps
in 74.708 seconds and verified all 54 artifacts, including the original native
analysis and numerical archive. The native result displays actual output times,
force/energy/angular residuals and their unchanged thresholds. GADGET and MPI
appear separately in the software evidence table. These runs do not execute
other native engines.

The updated notebook has 134 cells. All eleven executable cells passed on the
second combined run, including a direct replay of energy/angular-momentum
residuals from original native masses, positions, velocities and potentials.
Seven focused interface tests passed. See
`optional-combined-diagnostics-http-summary.json`,
`optional-walkthrough-combined-summary.json` and
`retained-tests-combined-notebook-interface-summary.json`.

## Observed first-party calls in the teaching run

The traced combined HTTP run passed all 20 steps in 75.864 seconds and verified
55 artifacts. Its main-thread observer recorded 10,039 function call events
across 22 first-party Python files. Module imports and class definitions have
separate counts; no native or isolated-worker calls are inferred. Captured
source hashes matched at the end of the run. The source viewer hashes the
exact current bytes returned by its confined read route.

Eight focused tests passed, including real fixture execution, an unused
function/class, source mutation and profiler restoration. Their updated merged
receipt records 857 unittest passes; the five standalone pytest checks make
862 recorded passing checks, with no failures/errors/skips. The notebook now
has 136 cells; all twelve code cells passed on this actual traced run. It
checks `python-calls.json` against the run receipt, lists observed functions
and current source matches, and replays native conservation residuals.
See `observed-combined-http-summary.json`,
`retained-tests-observed-summary.json`, `observed-call-ui-summary.json` and
`optional-walkthrough-observed-summary.json`.

The normal CLI was also launched without `-B` and with
`PYTHONDONTWRITEBYTECODE` removed. All seven quick checks passed, recording
3,902 calls across eleven files, and no root/optional Python caches were
created. The entrypoint sets bytecode suppression before its own imports.
See `default-clean-observed-quick32-summary.json`.

A final current-source combined HTTP run then passed all 20 steps in
75.355 seconds with all 55 artifacts verified. It again observed 10,039 calls
across 22 files, whose captured hashes still match the checkout. Its source
panel links each observed function to the recorded line and distinguishes
changed revisions. See `observed-final-combined-http-summary.json` and
`optional-walkthrough-observed-final-summary.json`.


## Expanded optional engines walkthrough

The HTTP-launched `engines` profile passed all **23 steps** in 86.611 seconds,
with **208 artifacts** checked against their recorded hashes. Alongside the
seven quick controls, native 32-particle GADGET and twelve library calculations,
it ran actual CLASS and AxiCLASS at z=49 and z=0, and Blender with 32 synthetic
display points. Each Boltzmann epoch produced 115 power modes. Blender rendered
one 256×192 four-sample still, saved an editable scene, and reopened it in a
fresh process to compare every point coordinate. A deliberate Python exception
also confirmed that the Blender invocation returns a failing exit code.
These are execution and structural controls, not astrophysical acceptance.

The CLASS and AxiCLASS workers separately passed in the clean Ubuntu ARM64
runtime with their verified native binaries and sealed two-epoch bundles.
A runtime without Blender correctly returned `UNAVAILABLE`, without installing
software or substituting another renderer. MUSIC, monofonIC, chemistry and halo
finders retain their separate campaign evidence; this profile does not yet run
them. Browser layout was unverified at this milestone; the later optional-lab
desktop/narrow checks are recorded above.

The regenerated notebook has **139 cells**, including **13 executable cells**.
All thirteen passed on this retained engines run, including original density
arrays, native energy/angular momentum replay, library receipts, the sealed
linear spectra and their dimensionless power handoff, Blender scene receipts,
and a registered density-profile figure contract. Source references now cover
256 application Python files and 192 commands. The merged current-source
regression receipt passed 857 tests without skips; the five standalone pytest
checks retain their separate earlier receipt. The eight interface tests were
rerun after the optional native worker changes.

Evidence: `native-packages-checked-http-summary.json`,
`native-package-linux-summary.json`, `native-package-availability-summary.json`,
`native-blender-python-exit-summary.json`,
`retained-tests-native-packages-summary.json`, and
`optional-walkthrough-native-packages-summary.json`. The last Blender helper
change expanded only its module documentation; a separate fresh native worker
run passed afterward in `native-blender-final-documentation.log`.


### Fresh monofonIC integration

The engines profile now includes a fourth native-package worker, for **24
steps**. A fresh HTTP run passed all 24 in **90.578 seconds**, retaining and
hash-checking **270 artifacts**. It added a production CLASS → staged species
transfer/background → monofonIC 1LPT calculation. The 4³ grid creates **64 gas
plus 64 DM particles**, independent of the selected 32-particle quick fixture.
Inspection checked finite phase-space fields, periodic bounds, 128 unique
integer IDs and a gas mass fraction of 0.1573033580, against the configured
0.1573033708 (absolute tolerance 1e-6). The HDF5, configuration, original logs,
arrays and comoving-kpc/h XY figure are retained. This is an initial-condition
structural check; it does not evolve the particles or establish phase accuracy.

All thirteen code cells in the 139-cell notebook passed against this new run,
including the added IC array/ID/gas-fraction replay. The references now cover
257 application Python files. Eight focused interface checks and the merged
857-test current-source regression receipt passed. See
`native-monofonic-http-summary.json`,
`optional-walkthrough-native-monofonic-summary.json`, and
`retained-tests-native-monofonic-summary.json`.


The new monofonIC worker also passed on Ubuntu ARM64 using the container with
the corrected setup prerequisites, with original HDF5/NPZ values checked again.
The first attempt deliberately used the older base image and failed because
`libfftw3_mpi.so.3` was absent; that failed receipt and native log remain intact.
The passing run used the environment produced by the existing corrected setup,
without changing code or weakening checks. See `native-monofonic-linux-summary.json`,
`optional-native-monofonic-linux.log` and
`optional-native-monofonic-fixed-linux.log`.


### Atomic Grackle control in the optional checklist

The engines profile now adds a fixed 32-cell Grackle atomic H/He cooling
calculation, for **25 steps**. A fresh HTTP launch passed all 25 in **92.455
seconds**, with **280 artifacts** hash-checked. The worker compiled the shipped
small C control against the installed shared Grackle 3.4.1 public API and
retained compile/native invocations, the original CSV, numerical arrays and a
temperature/ionization figure. It evolves independent fixed-density cells for
1e11 seconds, without UV, metals, molecules, dust or cooling lookup tables;
it is separate from the production GADGET chemistry bridge. All 32 cells
cooled, with fractional specific-energy changes from −0.9384 to −0.8752.
Maximum H/He mass-fraction residuals were 2.22e-16 and 5.55e-17, below the
unchanged absolute 1e-10 tolerance. These controls do not establish convergence.

The first checker incorrectly required positive cooling times. The native
library computes energy divided by its signed energy-rate: net cooling gives
negative values, and its upstream evolution helper takes the absolute value
for a timestep magnitude. The checker was corrected to require the negative
sign for this cooling-only setup. The original failed receipt remains in
`native-grackle-first`; the corrected standalone and integrated runs both
passed. This corrected a checker convention, not the native solver.

With the pkg-config search confined to an empty directory, the actual worker
returned `UNAVAILABLE`, without a compiled executable or fabricated arrays.
Missing prerequisites are explicit; the worker installs no dependencies.
[Software installation](software.md#build-optional-grackle-and-inspect-its-cooling-control)
now includes the external shared-library build and environment commands.

The 139-cell notebook passed all thirteen executable cells on this latest run,
including the original signed cooling-time and H/He conservation replay. The
eight focused interface tests and merged 857-test current-source regression
receipt passed. Source references now cover 258 Python files and 192 commands.
The C caller is included by the optional source packager; no new ZIP was made.
Evidence: `native-grackle-http-summary.json`,
`native-grackle-unavailable-summary.json`,
`retained-tests-native-grackle-summary.json`, and
`optional-walkthrough-native-grackle-summary.json`. This new standalone
Grackle worker was exercised on macOS; a Linux execution of this worker is
not claimed. MUSIC, finders and independent SIDM remain separate campaigns.


### MUSIC nested controls in the optional checklist

The engines profile now includes the production MUSIC transfer-control CLI,
for **26 steps**. A fresh HTTP launch passed all 26 in **97.343 seconds**,
with **406 artifacts** verified. MUSIC delivered ten uniform/nested/periodic
1LPT realizations and rejected seven deliberately invalid requests for their
expected diagnostics. Uniform realizations contain 4,096 gas plus 4,096 DM
particles; each nested/periodic realization contains 2,744 gas, 2,744 fine DM
and 5,265 coarse matter particles (**10,753 total**). The particle counts and
analytic synthetic transfer origin are explicit; these are not 32-particle
or target-halo zoom claims.

The original controls check unchanged Lagrangian coordinates and IDs, velocity
doubling, zero-transfer velocities, continuity units and periodic frame
restoration. Mass, exact-ID, momentum and barycenter audits passed for all ten
delivered realizations. The largest uniform/nested velocity-displacement
relative residual was 4.281e-14, within the existing 1e-9 gate. Original input
configurations, transfer tables, HDF5, native logs, the audit and numerical
arrays remain linked to their hashes. The new full/central XY plot reads the
same native nested coordinates; it does not interpolate additional particles.
Expected rejection attempts can retain partial native outputs; their status
remains rejected, separate from the ten delivered realizations.

All thirteen calculation cells in the 139-cell notebook passed on this run.
The MUSIC cell verifies original input/output hashes against both run and
native-suite receipts, checks the nested arrays against original HDF5 and
shows its actual counts and units. Eight interface tests and the merged
857-test current-source regression receipt passed. References now cover
259 Python files and 192 commands. The new worker was executed on macOS;
Linux execution of this wrapper is not claimed. Finders and independent SIDM
remain separate campaigns, and the graph calibration gap remains open.
Evidence: `native-music-http-summary.json`,
`retained-tests-native-music-summary.json`, and
`optional-walkthrough-native-music-summary.json`.


### Fresh native finder controls and binary cleanup

The optional engines profile now includes Rockstar and VELOCIraptor, for **28
steps**. The final post-cleanup HTTP run passed all 28 in **102.898 seconds**,
with **599 artifacts** verified. Each finder receives a fresh explicit-unit
32-DM snapshot fixture with the same seed and exact IDs. The two saved epochs
repeat coordinates/stored velocities at a=0.5 and a=1.0; they are not an evolved
simulation or a true halo history. Rockstar reads both epochs and returned
one then zero objects. VELOCIraptor reads a=1.0 and returned zero. Native empty
catalogues remain empty; neither successful execution nor row-count agreement
establishes resolved halo accuracy. Rockstar's m200c convention and native
assigned membership differ from VELOCIraptor's selected mass without unbinding.

The driver retains original native reports, labelled Rockstar tables,
per-epoch intake JSON, exact IDs, arrays and a diagnostic picture. Rockstar's
periodic path requires eight available CPU slots; VELOCIraptor uses two threads
and eight-neighbor/minimum-size controls. The notebook initially exposed an
incorrect metadata path in the new wrapper: production stages Rockstar under
an attempt directory. The wrapper now obtains the actual selected directory
through `rockstar_directory`; the original failed notebook/log is retained.
All thirteen calculation cells in the 139-cell notebook passed on the final
run, including original reports, native artifact seals and catalogue counts.

A byte-signature source audit found **22 legacy macOS native/debug binaries**
that extension-only checks had missed. AxiCLASS still used its in-tree copy;
it received a real external build before the old executable was moved out.
The binaries and complete debug bundles (26 files, 6,509,373 bytes) were moved
to external recovery storage with hashes. Rockstar was rebuilt externally
because its previous source inventory included generated debug metadata. All
eight installed native resolver records now verify their source/executable
hashes under external build paths. A full vendored byte-signature rescan finds
no ELF/Mach-O/static archives outside Git metadata. The post-cleanup 28-step
run exercised the new AxiCLASS and Rockstar binaries; this was not only a
presence check.

The first missing-Rockstar probe unexpectedly discovered the legacy binary and
ran successfully; it is retained as execution evidence, not an absence test.
After cleanup, workers with an uninstalled external runtime reported
`UNAVAILABLE` for both finders without fabricated numerical output. Eight
interface tests and the merged source-bound 857-test regression receipt passed.
Generated references cover 260 application Python files and 192 commands.
No ZIP was created. These new finder workers were exercised on macOS; Linux
execution of the wrappers is not claimed. Consistent Trees, DisPerSE and
independent SIDM still have separate campaign checks.

Evidence: `native-finders-cleaned-http-summary.json`,
`optional-walkthrough-native-finders-cleaned-summary.json`,
`retained-tests-native-finders-cleaned-summary.json`,
`native-finder-unavailable-summary.json`,
`vendor-native-binary-cleanup-moves.json`, and
`native-binaries-post-cleanup-summary.json`.


### Native tree and filament walkthrough

The engines profile now adds Consistent Trees and DisPerSE, for **30 steps**.
A fresh HTTP launch passed all 30 in **105.746 seconds**, retaining **684
hash-verified artifacts**. Consistent Trees received two explicitly synthetic
128-object catalogues at a=0.99 and a=1.0. The production driver wrote actual
hlists containing 256 unique native node IDs and 128 forward descendant links.
Each link's target exists at the native recorded later scale factor. These
are catalogue objects, not particles or detected halos; the supplied Np=1000
is a synthetic property, and no physical merger history is claimed.

DisPerSE received 64 DM samples, meeting its production minimum, on a 16³
periodic CIC mesh. Its dimensionless field is log10(1 + smoothed rho / mean rho),
with one-cell Gaussian density smoothing, persistence cut 0.1 and one
iteration of display-geometry smoothing. Native mse/skelconv produced **55
skeleton points and six polylines**; their XY projection occupies **33 cells**.
Original FITS, native skeleton, unsmoothed topology, VTP, HDF5, arrays and sealed
contracts are retained separately. The diagnostic image shows boolean projected
skeleton occupancy, not density, physical thickness or a resolved cosmic web.

The notebook now has **141 cells**, including **14 executable cells**. All
fourteen passed on this run. The new section compares original native hlists
with saved integer IDs, scales, positions and masses, checks link endpoints,
verifies the filament contract's artifact hashes, compares original skeleton
HDF5 with the array archive, and reconstructs periodic XY occupancy from saved
polylines. It launches no new native program. The fixed small controls do not
establish physical accuracy or convergence.

Both workers also returned `UNAVAILABLE` with no substitute arrays when tested
against an uninstalled external runtime. Eight interface tests and the merged
source-bound 857-test regression receipt passed with no skips/failures/errors.
Generated references cover **261 application Python files and 192 commands**.
These new wrappers were exercised on macOS; their Linux execution is not
claimed. Independent SIDM and the broader native physics matrix remain separate
campaigns, and the rejected graph-calibration gate remains unresolved.
Evidence: `native-structure-http-summary.json`,
`optional-walkthrough-native-structure-summary.json`,
`retained-tests-native-structure-summary.json`, and
`native-structure-unavailable-summary.json`.


### Independent SIDM worker and long-path repair

The optional interface now includes a 31st worker for an explicitly selected
Linux OpenGadget3 rare/directed reference build. Its current macOS HTTP run
completed in **109.904 seconds**, with **30 PASS**, one explicit OpenGadget3
**UNAVAILABLE**, and **686 hash-verified artifacts**. The independent worker
separately passed in Linux with 32 beam particles at native times 0, 0.0001
and 0.0002. It uses a deliberately high toy total cross section of 100 cm²/g
and gravity is disabled. Changed-velocity particle counts were **0, 11, 25**;
these are not event counts. Exact IDs and masses were preserved. Maximum
relative kinetic-energy and momentum residuals were **2.22e-16 and 1.43e-16**,
below unchanged 1e-10 limits. Forty-one owned output artifacts are hash-bound
in the worker result. This is not an inter-engine or calibrated-rate comparison.

The old reference build was correctly rejected because a recorded Python
input had changed. A fresh actual source-bound rebuild then exposed a native
buffer overflow: the pinned program copies filenames into char[100] buffers.
The crash address resolved to its main parameter-file handling. Production
`homogeneous_control_runner` now supplies `param.txt`, the IC basename,
`../data/` and `output_times.txt` relative to the owned input cwd. The original
absolute assets remain bound by the invocation receipt. No native-source
manifest, scientific threshold or old receipt was edited to bypass a check.
A second fresh rebuild against the fixed Python driver passed, and its long
external output-root run completed successfully. Original rejection/crash logs
remain preserved. The first build-command attempt also used an incorrect
command name and launched no compilation; the corrected command was the
existing `sidm-reference-build` interface.

The **143-cell notebook passed all 15 code cells**, including actual Linux
scattering artifacts selected with `RESEARCH2_LAB_SIDM`. It verifies their
retained hashes, exact IDs and saved momenta/energies, recomputes conservation
and shows the original velocity-change image without executing the native
binary on macOS. The broad macOS run remains explicit about the missing local
build. The software guide documents the completed-build environment variable
and source-bound build command.

The production change prompted a fresh full application/verification/interface
regression run. The verification process initially lacked its fixture environment
and recorded four individual skips plus one class-level skip (seven test cases).
All **11 affected cases** then passed with the actual retained native campaign,
restart, resolution and MUSIC fixtures. The aggregate verifies current source
hashes, original discovery/skip counts and the exact rerun set: **756 unique
verification tests**, plus **93 application** and **8 interface**, all covered
and passing (**857 total**, no unresolved skips/failures/errors). This combines
real test runs; it does not rewrite the original skipped-run receipt. The five
standalone pytest checks retain their separate unchanged-source earlier receipt.

Evidence: `native-independent-http-summary.json`,
`native-independent-sidm-linux-summary.json`,
`optional-walkthrough-native-independent-summary.json`,
`retained-tests-independent-shortpaths-complete-summary.json`,
`retained-tests-independent-shortpaths-complete-verification-summary.json`,
`native-independent-sidm-linux-ready.log`,
`native-independent-sidm-linux-current.log`, and
`native-independent-sidm-linux-shortpaths.log`. Generated references now cover
262 application Python files and 192 commands. Dedicated gas/star/BH physics
checklist integration and the rejected graph-calibration gate remain open.


## November 6: complete physics checklist and notebook replay

The current HTTP-driven `complete` profile finished **34 steps in 155.336 s**:
**33 PASS**, one explicit local OpenGadget3 **UNAVAILABLE**, with **862 artifacts**
verified. The independently executed Linux scattering control remains PASS;
its outputs were included in the same notebook replay. No new movies or ZIPs
were created.

Three newly integrated controls executed real GADGET source physics. All start
with 64 gas particles; the BH control adds one explicit 10 M☉ seed. At seed 42,
cooling retained its IDs and mass while mean specific energy fell from
10¹⁴ to 9.4358993 × 10¹³ erg/g. The stellar run began with zero stars, then
produced six and nine native SH03 births at the later epochs; all final
`SSPBirthMode` values were 1 and the production mass/momentum history audit
passed. Its effective EOS energy is not a single gas temperature. The BH's
physical mass grew from 10 to 10.000002481181017 M☉ with positive native inflow.
The toy BH background has Ωm = Ωb = 0.3, h = 0.7; the separately compressed IC
and original IC are both retained with hashes. This checks seeded accretion,
not BH formation or complete source-budget conservation. Stellar delayed
returns are disabled in this small birth control.

The new worker retains a dependency-unavailable receipt even without Python
site packages. That check and the eight existing interface tests passed.
The source-checked regression aggregate now contains **858 unique passing
unittest cases** (756 verification, 93 application, nine interface), with no
unresolved skips, failures or errors. The five separate pytest passes retain
their earlier unchanged-source receipt. Production core files did not change
for this optional integration; the aggregate checks their original hashes.

The expanded **148-cell notebook passed all 16 code cells** against the actual
complete run and separately saved Linux outputs. It replays original units,
arrays, birth bookkeeping and mass gain, without launching another engine.
Generated references cover **263 application Python files and 192 commands**;
observed regression calls cover 89 files and are not full branch coverage.
The graph baseline-gain recipe remains rejected by its calibration gate.

Evidence: `native-complete-physics-http-summary.json`,
`retained-tests-native-physics-interface-summary.json`,
`retained-tests-native-physics-summary.json`,
`python-execution-native-physics.json`, and
`optional-walkthrough-native-physics-summary.json`, all in the external campaign
evidence directory. Earlier receipts describe their own earlier source revisions.


### Supported gas-phase example and final source scan

The registered single-phase gas plot was rerun using the actual retained
`grackle-cloud-native-64-conserved` outputs, with the production hydro history
audit and gas-phase exporter. All three epochs have **64 supported gas particles**,
no entries outside the fixed display edges, measured native temperatures,
composition and zero star-formation rates. No effective-EOS energy was relabeled
as temperature. The color is original gas mass per physical density/temperature
bin. This raises supported recipe coverage to **91**, with **ten** still entirely
masked and the graph-gain recipe still unexecuted. Total rendered recipe
coverage remains **101 of 102**. The earlier masked output remains preserved.
Evidence: `native-grackle-cloud-phase-summary.json` and its native history,
original array sidecars and production gallery.

The full source-tree cache/history scan found no remaining disposable entries;
eight installed native resolver records revalidated their source and executable
hashes. Required sources/tables/licenses and optional teaching assets remain;
outputs/builds/environments stay external. Evidence:
`final-source-cleanup-summary.json`. No ZIP was created.


### Known shape, filament alignment and bounded-history controls

Four previously all-masked recipes now have supported measurements. A new
6,000-point authored HDF5 fixture contains five Fibonacci shells with true
axis ratios b/a = 0.7 and c/a = 0.45, each rotated by an additional 0.15 rad
about z. The production structure reader, AGAMA reconstruction, shell fitter,
bootstrap and registered renderer measured four shells. All requested axis
orientations were resolved; ratios matched within 10⁻⁴ and major/intermediate
adjacent twists matched 8.5943669° within the predeclared 0.02° bound. Minor-axis
rotations were about 0.000457°. The separate direct-force/tide validation passed.
This measures authored geometry, not evolved or converged halo structure.

A rotating version has Ωz = 1 km/s/kpc. Its isolated environment classified the
host centre as a class-3 knot at both smoothing scales, correctly leaving
filament alignment unsupported. A distinct analytic periodic filament was then
supplied as 8,448 type-2 carriers of 10⁵ M☉ each; the selected shape/spin samples
remain the original 6,000 type-1 carriers of 1 M☉ each. The full 14,448-particle
snapshot passes production T-web measurements on a 33³ mesh with original
threshold 0.1 and eigenvalue-gap floor 0.001, at smoothing lengths 0 and 5 comoving
kpc. Both spin- and shape-alignment figures have eight supported series entries
(four apertures shown as lines and points), not eight independent measurements.
The original automatic-axis renders remain retained. Separate display recipes
set cosine limits [0,1] to show their full physical range, with original arrays,
support masks and measurement products unchanged.

The initial history harness omitted the graph's required proper position and
failed before publication. Those logs remain preserved. A separate new history
run supplies actual per-epoch proper centres, explicit authored identity links
and a consistent local analytic clock: t(a)=1+2(a−0.5) Gyr and H(a)=1/(2a Gyr).
Six nodes span a=0.5 to 0.500005. The production clock computes its conditional
numerical envelope, and both smoothing-scale history figures have one supported
saved interval under a 0.02 Myr bound. The local filament axis change is measured
as zero. The clock and snapshots are explicitly synthetic; no Boltzmann solver,
native evolution, halo detection or physical ancestry is claimed for this control.

Supported recipe coverage is now **95**, with **six** entirely masked and the
graph-gain recipe still gated; total rendered coverage remains **101 of 102**.
Evidence: `resolved-shape-control-summary.json`,
`rotating-shape-control-summary.json`, `isolated-knot-alignment-rejection-summary.json`,
`filament-shape-control-summary.json`, `filament-web-control-summary.json`,
`bounded-alignment-views-summary.json`, and `bounded-filament-history-summary.json`.
All raw inputs, requests, numerical products and original failures remain external.
Production Python sources did not change for these controls; the recorded
regression/source bindings remain applicable to their stated suites.


### Measured assembly histories and analytic orbital events

An eight-epoch synthetic HDF5 control now exercises the production host reader,
AGAMA expansion, radial profiles, NFW fits, morphology tensor, identity graph,
main-branch estimator and assembly renderer. Each epoch contains **8,192 DM +
2,048 stellar particles**. Prescribed rs grows from 1 to 1.35 proper kpc;
measured rs is 1.0013348 to 1.3523579 kpc, with all eight fits inside the
predeclared 5% recovery bound. DM mass sums grow from 8,192 to 11,059.2 M☉ and
stellar sums from 102.4 to 245.76 M☉, matching the authored arrays. These mass
changes are prescribed inputs, not a claim of mass-conserving dynamics or
native stellar formation. SFR=0 comes from a complete empty gas census.

The original 0.05–10 kpc expansion failed coefficient replay. Component-level
checks show that its stellar multipole reload differs in the exterior region
(about 0.2% near 10 kpc), while interior interpolation agrees closely. The
verifier remains unchanged. A new experiment declares a 40-kpc outer grid
containing every calibration/withheld-validation query. All eight field
validation and serialized-coefficient replay checks then pass. This establishes
the stated interior domain; exterior extrapolation remains unverified. Both
sets of inputs and the original rejection remain preserved. No checksum or
science threshold was rewritten.

The completed run's harness initially looked one level too high for the gallery
index. A separate verification script read the actual completed products,
rechecked all eight structures and the graph, and produced its summary without
editing the source-pinned helper or rerunning the measurements. The public
`export-assembly-figures` command also ran successfully on that stage and exact
graph, exporting all eight figures into a new directory.

Supported samples now appear in concentration history (eight), stellar assembly
(eight), and quenching/morphology (24 across the sSFR and two morphology series).
The resulting synthetic aperture concentration changes from about 4.261 to 3.591
using the explicit toy density reference 0.1 M☉/kpc³ and Δ=200. The graph's
identity correspondence and measurement-quality scope are explicitly synthetic;
no resolved native central galaxy or measured ancestry is claimed.

A separate **32-particle**, three-epoch authored flyby exercises the native-format
catalogue/graph/orbit intake and event exporter. Proper derivatives include the
recorded Hubble term. With a 0.5-kpc impact parameter, speed 0.4 kpc/Myr and
fixed 1.5-kpc boundary, the producer finds infall at **1001.4644660940672 Myr**
and pericentre at **1005 Myr**, radius **0.5 kpc**, within the fixed 10⁻⁶ Myr
analytic recovery tolerance. There are no progenitor joins or physical mergers.
Saved brackets and the interpolation-uncertified classification remain present;
these are reconstructed candidates on synthetic input, not certified native
orbital evolution. The original native event example with no detected events
remains unchanged.

Supported recipe coverage is now **99**, with **two** entirely masked
(splashback and action diffusion) and graph-baseline gain still gated. Total
rendered recipe coverage remains **101 of 102**. Evidence:
`assembly-coefficients-diagnostic.json`, `measured-assembly-grid40-summary.json`,
`measured-assembly-cli-summary.json`, and `analytic-orbital-events-summary.json`.
The rejected first run and misplaced-index harness logs remain recorded alongside
the successful bounded-domain and verification receipts.


### Known-density splashback recovery

A new **32,768-particle** synthetic control samples the declared density
ln ρ = −2 ln r − 3w√(π/2) erf[ln r/(√2 w)], w=0.2, over 0.2–4 proper kpc.
Mass-quantile inverse-CDF radii and a seeded independent permutation over
Fibonacci directions are retained with the analytic arrays. It is an authored
radial-density fixture, not native halo evolution. The primary seven-bin GLS
window, search interval 0.5–2 kpc, 64 covariance draws, fixed seed and recovery
bounds were recorded before measurement. No support mask was relaxed.

Production native-format host/profile intake, calibrated AGAMA field replay,
local GLS derivatives and the feature estimator measured a unique valley at
**0.9654851 kpc**, prominence **2.8943698**, and half-prominence ln-radius width
**0.4907564**. The analytic values are 1 kpc, 3 and 0.4709640; all lie within
the preset 10% radius / 20% depth / 20% width recovery tolerances. The original
radius-bin bracket is [0.8914250, 1.0456981] kpc. The slope near the inner
sampling cut is a boundary feature outside the declared search interval.

Covariance perturbations gave one acceptable feature in **55/64 draws**
(success fraction 0.859375), with nine ambiguous draws retained. The fractional
radius-error estimate is 0.04154. The five-bin alternative remains
`UNCERTAINTY_UNRESOLVED`; seven- and nine-bin alternatives are `MEASURED`, with
supported radii spanning 0.9654851–1.0456981 kpc. The prescribed primary window
was not chosen after viewing those alternatives. These are conditional Gaussian
covariance diagnostics, not independent particle realizations or convergence.

The completed 43-MiB environment product initially exceeded the exporter's
conservative 1-GiB workspace bound. A separate export-only retry with an explicit
**4-GiB allocation** passed, preserving all original measurements, requests,
checks and the rejection log. Its slope/depth/width figure has **35 supported
series entries**. The atlas now has **100 supported recipes**, one entirely
masked action-diffusion recipe, and calibration-gated graph gain. Total rendered
coverage remains **101 of 102**. Evidence: `known-splashback-control-summary.json`,
`known-splashback-control.log`, `known-splashback-render.log`, and the original
predeclared settings, numerical products and gallery.


### Native paired action changes with an independent numerical check

The remaining action-diffusion recipe now has supported data. The first attempt
selected only the two later epochs of the earlier zero-velocity control; the
native comparison correctly rejected its changed first selected state. That
failure remains recorded. A new pair of **32-particle native self-gravity runs**
uses identical initial phase space, IDs and masses, total carrier mass **10⁴ M☉**,
initial velocity [0.01, 0.05, 0.02] km/s, and physical gravity constant
**GravityConstantInternal = 43010.47329314801** for kpc, 10¹⁰ M☉ and km/s units.
The initial masses and 32 exact IDs match, total mass stays 10⁴ M☉ at every
epoch, and the original native execution receipts verify.

The only differing physical parameters are stellar softenings **0.01 and
0.30 kpc**. The other parameter differences are the owned IC/output/schedule
paths. All actual IC arrays match exactly; initial-state hashes are identical.
The maximum later branch velocity difference is **0.0007655422 km/s**. These
are seeded stellar carriers in an artificial filament, not native star
formation, a detected stream, or a physical perturber intervention.

A frozen **lmax=mmax=0 monopole** is fitted from the reference's original initial
stellar census. An explicitly recorded unit/frame adapter stores those same
physical offsets and masses in a new native-format host fixture. Source/frame
bindings, direct-field calibration and coefficient replay pass. The monopole
is a declared common diagnostic chart, not a claim that the original native
nonspherical forces have spherical invariants. No late epoch is omitted.

The response pipeline uses actual native epochs **0, 3.9111688867 and
7.8223377734 Myr**, and all **32 particles** have action support at all three.
The final mean-square paired changes for (Jr,Jz,Jphi) are
[**2.4102703×10⁻¹⁰**, **1.0089476×10⁻¹¹**, **8.5428089×10⁻¹⁰**]
proper kpc² (km/s)². Baseline values are zero by the stated centring operation.
The plotted quantity is not divided by elapsed time and is not a stationary
diffusion coefficient. The original raw paired ΔJ arrays and support are retained.

An independent check uses AGAMA spherical actions on **all 192 particle states**
(32 particles × three epochs × two branches). It reconstructs the centred curves
from those actions, without using the production radial-quadrature results as
inputs. All agree within the predeclared 5% relative / 5×10⁻¹⁶ absolute comparison
bounds; the actual largest relative curve difference is **3.62555×10⁻⁸**.
Both engines use the same saved diagnostic field and physical chart. This is
an independent numerical calculation for that field, not independent evidence
that the approximation describes astrophysical stream dynamics.

The registered action figure has nine supported series entries (three components
at three epochs). A separate display recipe labels Jr/Jz/Jphi and the measured
quantity clearly; original data and production render remain unchanged. Recipe
coverage is now **101 of 102, all 101 with supported plotted entries**. Only
graph-baseline gain remains gated by failed calibration. Evidence:
`native-stellar-action-pair-summary.json`, `native-paired-action-control-summary.json`,
`paired-actions-independent-summary.json`, `paired-actions-independent-arrays.npz`,
`paired-actions-independent-predeclared.json`, and `action-reading-view-summary.json`.
The first helper launch lacked its source import path and ran no native job;
its failure log and the corrected launch with explicit `PYTHONPATH` are retained.


### Portable figure workers, full 36-step profile and fresh notebook replay

The shipped `optional_tests/advanced_controls.py` now regenerates the known-shape
and density-valley controls without the author's external campaign helpers or
fixtures. It creates complete explicit-unit HDF5, exact uint64 IDs, a clearly
synthetic catalogue centre/clock, initial input and source records, predeclared
recovery settings, production structure/environment products, registered
figures, an offline `index.html` and hashed `receipt.json`. It rejects source-tree
outputs, preserves failures and reports absent dependencies as `UNAVAILABLE`.
It installs/builds/downloads nothing and creates no movies or ZIPs.

Both standalone commands passed; the shape command was launched from `/tmp`
using the script's absolute path to check working-directory independence. The
actual counts are **6,000** and **32,768**, separate from the quick fixture count.
The new `figures` profile has **nine steps** (seven quick plus these two workers),
and `complete` now has **36**. Interface menus, validated HTTP profile selection,
CLI help, software receipts and equations include these controls. Each worker
uses a 900-second bound, parent limit 930 seconds and an explicit 4-GiB estimated
workspace. The package whitelist already includes the new Python helper; no
archive was generated.

The fully configured macOS HTTP run finished in **170.51 seconds** with
**35 PASS**, **one explicit OpenGadget3 UNAVAILABLE**, and **1,104 verified
artifacts**. Shape and density-valley workers passed in 7.291 and 6.375 seconds.
All earlier native/library/physics checks also passed locally except the
Linux-only independent build, whose actual separate Linux evidence remains
retained. The first 36-step run omitted `PKG_CONFIG_PATH`, so its Grackle check
correctly remained unavailable; its expectation assertion failed and the
original result/log remain. A preliminary relaunch used a nonexistent example
prefix and was interrupted. The final launch used the actual shared-library
prefix after `pkg-config --modversion grackle` returned **3.4.1**. No dependency
was installed or rebuilt and no missing result was relabeled as a pass.

The two new interface tests check the no-site-packages unavailable report and
source-output rejection before any directory is created. All **11 optional
interface tests passed**. The updated aggregate has **860 passing unittest
cases**: 756 verification, 93 application and 11 interface. The old core/application
runs have no optional-file call/import events and their nonoptional bytes still
match. Their unexecuted optional-file pins are explicitly excluded from reuse;
the changed teaching files have fresh interface pins. Original fixture skip and
11-case native rerun receipts remain preserved. Five standalone pytest passes
retain their separate earlier unchanged-source record.

Generated references now cover **264 application Python files and 192 commands**.
The refreshed **152-cell notebook passed all 17 code cells** on the actual
36-step run and the retained separate Linux outputs. Its new control cell
checks worker/parent hashes, replays original coordinate/mass cgs conversions,
recomputes ellipsoidal-window tensor ratios and verifies the saved linear GLS
derivative and propagated covariance identities. It displays the original
fresh plots without launching another engine. The final atlas cell also reads
the actual paired-action example. Figure support remains **101 of 102**, with
only graph-baseline gain calibration-gated.

Evidence: `native-complete-portable-figures-http-summary.json`,
`portable-figures-environment.json`, `retained-tests-portable-figures-summary.json`,
`retained-tests-portable-figures-interface-summary.json`,
`optional-walkthrough-portable-figures-summary.json`, `portable-shapes-6000.log`,
`portable-splashback-32768.log`, and the original preconfiguration/cancellation
logs. Other advanced archived examples still need their individual original
inputs or additional portable wrappers; browser layout and general scientific
acceptance remain outside these bounded checks.


### Portable 32-particle flyby, 37-step profile and Ubuntu execution

`optional_tests/orbital_control.py` now creates the analytic flyby without the
external campaign's snapshot/catalogue templates. The shared worker accepts
`advanced_controls.py orbits`; it generates three complete explicit-unit HDF5
sets, two authored catalogue apertures, graph correspondence, periodic-image
records and a consistent analytic clock. Production trajectory/event functions
recover infall **1001.4644660940672 Myr** and pericentre **1005 Myr**, within
the preset 10⁻⁶ Myr bound. All 32 particle IDs and physical fields are original
worker outputs. The fixed 1.5-kpc boundary, synthetic identity evidence and
interpolation-uncertified status remain explicit. No solver, native finder,
physical merger or measured ancestry is claimed.

The standalone command passed from `/tmp` on macOS and inside the retained
Ubuntu amd64 validation image, with source mounted read-only and network
disabled. The Linux image has no AGAMA module. Both output artifact maps
verified, and event times agree within the declared bound. This demonstrates
the standalone flyby with base dependencies; it does not certify the full
optional profile on Linux/WSL or constitute a fresh installation.

The `figures` profile now has **10 steps**; `complete` has **37**. The current
configured macOS HTTP run passed **36 checks in 171.649 seconds**, with only
the local Linux-only OpenGadget3 worker unavailable. Its separate Linux native
execution remains recorded. The HTTP harness verified **1,147 artifacts**,
including every flyby worker hash, original candidate times and software state.
The source-checked regression aggregate remains **860 unittest passes**: 756
verification, 93 application and 11 freshly rerun optional-interface tests.
Unchanged core-suite compatibility and original native fixture reruns retain
their separate receipts.

The refreshed **155-cell notebook passed all 18 code cells**. The new flyby
section recomputes relative proper derivatives from saved fields, derives speed,
impact parameter and host boundary, calculates closed-form infall/pericentre
times and checks the reported candidates. It also confirms no merger entries
and retains the uncertified-trajectory interpretation. Generated references
cover **265 application Python files and 192 commands**. Atlas support remains
**101 of 102**, with graph gain calibration-gated. No production core source
changed, no dependency was installed and no archive/movie was generated.

Evidence: `native-complete-portable-orbits-http-summary.json`,
`portable-orbits-platform-summary.json`, `retained-tests-portable-orbits-summary.json`,
`retained-tests-portable-orbits-interface-summary.json`,
`optional-walkthrough-portable-orbits-summary.json`, `portable-orbits-32.log`,
`portable-orbits-from-tmp-32.log`, and `portable-orbits-linux-32.log`.


### External-output enforcement and loop-declared command arguments

The lab CLI and server now enforce the documented external-output rule before
creating a directory. They expand/resolve the requested path and reject any
location inside the source tree, including an external symlink alias resolving
back into it. A new test covers both constructors and both direct/aliased paths.
All **12 optional-interface tests passed**; the compatibility-checked aggregate
is **861 unittest passes**. Production numerical/native sources did not change.
The reused core suites have no optional/reference-generator events, their
remaining source hashes match, and the changed files have fresh source pins.

The current `figures` HTTP profile passed **10/10 steps in 18.18 seconds**,
verifying **297 artifacts** and the flyby report route. The first harness used
an artifact URL relative to the output instead of the job container; its final
route assertion failed after all numerical steps passed. The corrected harness
uses `/artifacts/JOB/results/...`; the original failure/log remain retained.
This fresh run verifies current runner/server behavior. The earlier 37-step
native/library run remains evidence for its stated source revision; unchanged
numerical worker checks are not relabeled as a fresh complete run.

The reference generator had emitted a blank argument for the five paths declared
inside `audit_zoom_phases_main`'s literal loop. It now expands that literal loop
to `parent_ic`, `zoom_ic`, `parent_noise`, `zoom_noise`, and `destination`.
Unresolved expressions are shown as expressions, rather than blank rows. Four
focused checks passed, including comparison of the actual generated names to
public CLI help, literal-loop expansion, dynamic-name disclosure and flag aliases.
These checks exercise the generator, not scientific simulation. The zoom reading
guide now explains tracing, exact IDs, proper/comoving units, static coverage,
coherence/phase/amplitude distinctions and delivered velocity-transfer checks.

Evidence: `external-output-figures-http-summary.json`,
`retained-tests-external-output-summary.json`,
`retained-tests-external-output-interface-summary.json`,
`reference-argument-expansion-summary.json`, and their original retained logs.

</details>
