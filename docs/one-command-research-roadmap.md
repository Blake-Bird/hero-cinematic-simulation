# One-command Research2 roadmap

## Target

Give Research2 one study file with the starting cosmology, physical models, target selection, resolution, output cadence, and requested measurements and scenes. Run the study with one command. Publish a results site with figures, machine-readable measurements, source records, Blender assets, movies, and a status for every requested product. The experimental entry point is `python commands.py study study.json`; its integrated paths and limits are described below.

## Current implementation status (2026-11-13)

- The experimental [study driver](study.md) plans and can invoke native evolution,
  selected analysis, native FOF target selection and portable Blender export,
  optionally starting with a CLASS/AxiCLASS configuration. Its dry run, restart and outputs
  remain runtime-unverified; it explicitly blocks unconnected requests.
- The existing native, MUSIC staging, saved-output analysis, stream and Blender
  commands remain available. Their contracts and retained evidence are described
  in [architecture](architecture.md), [software](software.md) and
  [smoke-test](smoke-test.md). Their presence does not establish automatic
  end-to-end integration or science acceptance.
- A source-bound, explicitly selected native stellar reference can feed the
  existing static Gala and StreaMAX reference adapters; see
  [native stream reference handoff](native-stream-reference-handoff.md). It
  can optionally consume a spherical NFW host fitted from a native structure
  product, with withheld finite-source force residuals. Continuous-orbit support,
  GADGET periodic forces and scientific acceptance remain unverified.
- A [zoom target preview](zoom-target-selection.md) records selected catalogue
  rows, native source hashes, and selection evidence. A campaign may bind a
  reviewed preview by hash and require an exact match to its live reselection;
  this interface remains runtime-unverified.
- Saved Blender scene rendering has opt-in frame recovery; see the
  [operator guide](operator-guide.md). The separate [Blacklight bridge](blacklight.md)
  prepares a pinned Kerr ray-tracing/GRMHD handoff and an explicitly false-color
  Blender display from separately supplied data. It has not been built or run,
  and is not yet connected to the study or a cinematic sequence.
- These new paths have only been statically inspected. No tests, simulation,
  cluster job or render was run for this work. The unchecked items below remain
  required; source code or preparation alone is not validation.

## Before the first cluster smoke run

- [ ] Choose a bounded first study with an explicit success condition. Use a small parent run plus a known stellar input for the stream handoff if the small parent produces too few stars. Label the fixture and native products in the results.
- [ ] Add a study driver over the existing native run, zoom, stream, and analysis commands. Resolve dependencies, record exact configuration and seeds, and resume from verified completed stages. Keep the existing standalone commands available.
- [ ] Add a dry-run plan showing which executable each stage will call, its inputs, and its outputs. Check modules, MPI, compiled features, cooling and FSPS assets, disk quota, RAM, wall time, checkpoint location, and headless Blender availability on the intended platform.
- [ ] Define a snapshot handoff with cosmology and clock, proper/comoving units, host center and velocity, orientation, exact particle IDs, membership, stellar ages and metallicities, and source hashes. Require the actual catalogue and fields needed by each stage.
- [ ] Add an automatic host and progenitor selection policy with explicit ID overrides. Save the chosen object, epoch, mass, size, position, velocity, and selection evidence. Report insufficient particle support or ambiguous ancestry in the run summary.
- [ ] Wire the handoff through the stages below for the smoke study. Use bounded quick-look exports and preserve logs for setup, native evolution, science analyses, and rendering. Verify checkpoint and export recovery in the smoke run when testing is authorized.

## Connect simulation, streams, measurements, and Blender

- [ ] **GADGET → host field.** Fit the requested host components from saved dark matter, gas, stars, and supported central BH terms. Record the fit's spatial domain and withheld force residuals. Keep snapshot-by-snapshot fields for the evolving-host route.
- [ ] **Host field → Gala and StreaMAX.** Translate a declared frozen epoch into each adapter's supported host model. Pass the same measured progenitor state and physical units wherever the models permit. Compare approximated forces over the orbit and record unsupported components or mismatched release prescriptions. The input manifest should make each reference reproducible.
- [ ] **Native stellar debris and evolving-host model.** Follow exact saved stellar IDs and identify stripping candidates with binding and ancestry evidence. Use the existing forward integrator for the separate evolving-field experiment. Carry provenance for GADGET stars, Gala/StreaMAX tracers, and resimulated progenitor particles into every downstream product.
- [ ] **Measurements → figures and 3-D.** Measure tracks, widths, density, velocity, and supported actions on named samples. Save common coordinate frames, quality masks, uncertainties, plot arrays, and reusable scene assets. Give Blender the same physical center, clock, and track geometry. Check that scene exports and quantitative figures agree on particle IDs, epochs, and units.
- [ ] **Use each engine where its calculation belongs.** Select CLASS or AxiCLASS for each cosmology and the appropriate IC generator for each stage. Schedule alternative finders and independent SIDM runs when the study asks for cross-checks. Activate FSPS, lensing, BAO, and inference when the required inputs exist. Give every requested stage an explicit completed, skipped, unsupported, or failed status with a reason.
- [ ] **Make recovery routine.** Cache by input and software hashes. Retry figures and renders from saved products. Allow headless cluster runs to export portable assets for later GPU rendering. Publish the stage log and recovery command alongside the results.

The [research benchmark plan](research-benchmark-plan.md) sets out proposed comparisons and evidence requirements; none is newly executed or accepted by this plan.

## Before production-scale science

- [ ] Run the first authorized cluster smoke end to end, including scene load or a rendered preview. Verify native completion, exact-ID joins, common frames, stage statuses, and restart behavior. Inspect measured peak memory, disk, and wall time before sizing production.
- [ ] Establish baseline agreement on common inputs. Check linear spectra, growth, force accuracy, mass functions, power spectra, and halo counts against independent calculations and published references in overlapping regimes. Match mass definition, cosmology, and selection.
- [ ] Run a resolution, timestep, force-softening, volume, and independent-seed ladder appropriate to each claimed observable. For zooms verify target identity, parent-phase fidelity, contamination, and the usable radial and mass range. Freeze acceptance tolerances before looking at the scientific result.
- [ ] Validate gas, stellar-return, chemistry, SIDM, and BH mass, momentum, energy, and metal budgets for the specific combined build. Benchmark galaxy sizes, stellar masses, star formation, BH growth, and feedback against suitable observations or published suites before making population claims.
- [ ] Recover a known stream in a realistic galaxy zoom. Compare native debris, the evolving-host forward model, and static Gala/StreaMAX references. Check sensitivity to cadence, potential fit, sample selection, host evolution, and numerical resolution.
- [ ] For inference claims, validate posterior coverage on independent held-out realizations. Keep the existing graph-gain acceptance gate in place until both compared models pass.

## For publication-quality and cinematic visuals

- [ ] Build one end-to-end hero sequence from a single traceable study. Move from the cosmic web to the target halo, then to resolved stellar debris. Use shared clocks and camera anchors, stable display sampling, and consistent color and exposure across comparable shots.
- [ ] Provide a publication mode with physical units, uncertainty, scale bars, and downloadable arrays. Provide a cinematic mode with camera direction, lighting, and compositing settings recorded. Review temporal flicker, interpolation artifacts, visibility of faint structures, frame quality, and scientific interpretation independently.
- [ ] For telescope-like images, add and validate the relevant passbands, dust or radiative transfer, projection, PSF, and noise. Label the underlying field or emission model for each image.
- [ ] For an *Interstellar*-class black-hole shot, add a separate Kerr ray-tracing pipeline. Validate lensing geometry and redshift against published reference cases. For a physically predicted disk or jet image, supply independently modeled plasma and emission, such as GRMHD plus relativistic radiative transfer. GADGET's BH history can provide large-scale context and supported parameters; the horizon-scale model requires its own physical inputs.

## For broader scientific use

- [ ] Pick one publishable question and one benchmark the project can improve on. A candidate is whether a controlled change in a dark-matter model changes subhalo survival or a resolved stream observable beyond the numerical and realization scatter. Release the null controls, uncertainty budget, exact configs, and the measurements behind the figures.
- [ ] Have an independent group reproduce the benchmark from a clean install and use it on a different target. Publish reference outputs, pinned environments, data formats, scaling and cost measurements, known limits, a citation, and versioned releases.
- [ ] Add individual stellar evolution through an explicit MESA-scale handoff only for studies that require it. Add disk or planetary dynamics as a specialist workflow with explicit initial conditions and validation. A galaxy-scale star particle supplies population context; an individual planetary system requires additional initial data or assumptions.

## Changes and removals

- [ ] Reuse the existing native, stream, analysis, and Blender implementations. Move repeated workflow glue into the study driver where the callers share the same input and recovery contract.
- [ ] Replace ambiguous `all` language with a preview of the selected physics, reference runs, and actual deliverables. Keep a distinction in the results between native measurements, model-generated tracers, and display-derived quantities.
- [ ] Remove duplicate wrappers or obsolete assets only after checking CLI entry points, dynamic imports, shipped examples, and provenance consumers. Preserve licenses, published inputs, benchmark fixtures, failed scientific controls, and references used to verify reproducibility. No deletion is required to start the integration.
