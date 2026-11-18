# Research benchmark and validation plan

This is a proposed, unexecuted validation program, not a claim of parity with
any suite. See [smoke-test.md](smoke-test.md) for the scope and evidence of
retained runs and [completion-plan.md](completion-plan.md) for unresolved work.
A smoke run establishes workflow behavior only. Freeze the observable,
configuration, selection, tolerances, and analysis code before inspecting the
candidate result; preserve rejected runs and the original acceptance gates.

## Context from published suites

| Suite | Relevant published capability | Implication for a fair comparison |
| --- | --- | --- |
| [AbacusSummit](https://arxiv.org/abs/2110.11398) | Large gravity-only cosmology suite, force-accuracy characterization, snapshots, lightcones, and halo products. | Compare identical cosmology, epoch, box, halo definition, force and sampling conventions; measure accuracy and cost rather than assuming its published force error applies here. |
| [FLAMINGO](https://arxiv.org/abs/2306.04024) | Hydrodynamic volumes, explicitly calibrated feedback, model variants, convergence and observational comparisons. | Disclose calibration observables and distinguish held-out comparisons; matching a hydrodynamic prediction requires compatible physics, selection and resolution. |
| [MillenniumTNG](https://arxiv.org/abs/2210.10059) | Large hydrodynamic and gravity-only volumes, paired realizations, merger trees, lightcones and clustering convergence. | Compare matched statistics and report pairing, covariance, finite-volume and baryonic effects. |
| [FIRE-2](https://arxiv.org/abs/1702.06148) | Galaxy simulations with a detailed assessment of feedback and numerical sensitivity. | Compare physics and resolution explicitly; evaluate feedback conservation and the time and spatial scales relevant to the observable. |
| [Auriga](https://arxiv.org/abs/2401.08750) | Public galaxy zoom snapshots, catalogues, trees and initial conditions with convergence and galaxy comparisons. | Audit zoom contamination, identity, galaxy selection and numerical convergence before comparing resolved structure. |

Research2 has multiple workflows and retained bounded controls, but these are
not equivalent to the scale, calibration or published validation of these suites.
No performance or novelty ranking follows from the feature inventory.

## Predeclared measurements for later execution

1. **Initial conditions and force:** on common cosmology and phases compare
   transfer functions and growth with independent linear calculations; measure
   force residual distributions against a trusted direct-force reference over
   declared positions and scales, including tails and softening. Record units,
   force convention, particle sampling, reference precision and hardware.
2. **Large-scale structure:** compare power spectra, halo mass functions and
   clustering in common bins and redshifts, with matched mass definitions,
   finder configuration, particle cuts, shot noise and volume. Report bin counts,
   cross-realization covariance, finite-volume and resolution sensitivity. A
   matched phase pair does not replace independent realizations for covariance.
3. **Zoom:** repeat the same target at multiple refinement levels and buffers.
   Record phase and exact-ID correspondence where defined, cosmology, mass
   resolution, softening and low-resolution contamination versus radius. Compare
   the target's mass, density profile, substructure and merger history only over
   a declared uncontaminated, adequately resolved domain.
4. **Baryons and compact objects:** reconcile mass, momentum, energy and metal
   source/sink ledgers for each enabled build and time interval. Check stellar
   return, cooling, star formation, SIDM and black-hole terms independently and
   in combination. Compare galaxy/BH observables to specified observational
   selections and calibration data, accounting for model uncertainty. A growth
   curve alone is not a closed budget.
5. **Streams:** establish stellar membership and ancestry and compare measured
   native debris to independently identified samples; report support, selection,
   cadence and fit-domain force residuals. Repeat at multiple resolutions and
   epochs, and compare static tracer models and evolving fields as distinct
   experiments, not as the same particles or an automatic accuracy control.
6. **Inference and visuals:** perform held-out coverage and rank checks with
   independent realizations, including selection and model discrepancy; retain
   failed calibration and gate downstream gain claims. Check plot arrays and
   visual assets against the same measured IDs, epochs, frame and units. Validate
   a black-hole ray tracer against published geodesic/redshift cases separately
   from any emission or artistic model.

For every comparison publish raw values and units, selection, support counts,
uncertainty or reason it is unavailable, input and software hashes, failures,
resource measurements, and a machine-readable result. Establish numerical
acceptance tolerances for the specific scientific question using an independent
reference and required error budget *before* running the campaign. Do not infer
convergence from a single small run or from matching a single bin.

## Strong-field imaging candidate

The existing `core/black_holes.py` Schwarzschild null-geodesic mapping and
`core/blender.py` scene consume a sealed image and metadata. That calculation
explicitly supplies neither Kerr spin nor plasma emission. [Blacklight](https://github.com/c-white/blacklight)
is a possible separate GR ray-tracing and GRMHD post-processing backend: its
repository identifies the Unlicense, and its [method paper](https://arxiv.org/abs/2203.15963)
describes the code and validation. Its [documentation](https://github.com/c-white/blacklight/wiki/Background-and-Introduction)
describes its intended GRMHD image use. This is an evaluated candidate, not an
installed, pinned, or integrated dependency; platform builds, input formats,
maintenance, numerical controls and license obligations must be checked at a
specific commit before adoption.

A predictive handoff requires a separately supplied compatible horizon-scale
GRMHD snapshot and explicit metric, spin, units, observer geometry, plasma and
electron prescription, frequency and radiative-transfer settings. Galaxy-scale
GADGET BH data do not determine those inputs. Preserve source and parameter
hashes, and distinguish a prescribed illustrative disk from a prediction. First
compare geodesics and redshifts with published cases and test resolution and
radiative-transfer convergence. Only then adapt the resulting sealed imagery to
a Blender scene; the current Schwarzschild-specific intake cannot be relabeled
as a Kerr implementation.
