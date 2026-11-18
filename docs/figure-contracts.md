# Scientific figure reference

This reference describes the **102 registered graph recipes** in
[`graph_producers.json`](../configs/research/graph_producers.json). It lists what each
recipe measures, which inputs it needs, and the producer that supplies it. A registered
route does not establish that a particular run produced the graph.

For the 32 ordinary single-run output selections, use the [output
catalogue](outputs.md#single-run-product-catalogue). Specialized recipes below can
require additional simulations, selected objects, observations, or inference products.

Execution coverage for this checkout is reported per route in the
[smoke-test evidence](smoke-test.md). A callable check, a renderer-only fixture and
a complete measurement-to-render run are different forms of evidence.

**Current unresolved recipe:** `graph_baseline_gain` has no accepted retained
example. Both posterior families must pass their original independent calibration
before its gain calculation can run. The retained calibration failed; diagnostic
rank/coverage figures remain useful evidence of that failure. See
[the calibration gate](#inference) and [remaining scientific work](completion-plan.md).

## Shared requirements

Every recipe preserves the source numerical product or execution record, exact
identities, saved epoch, target selection, and physical/statistical unit conventions.
The renderer uses the producer's per-dataset units and rejects incompatible axes or
undeclared conversions.

Uncertainty and support come from the measured covariance, uncertainty arrays, and
boolean masks. Missing measurements remain unavailable or pending. Rendering cannot
supply missing physics or establish scientific acceptance.

Use the producer-specific request or campaign constructor. The public routes include
`flagship-products`, `export-hubble-figures`, `export-inference-figures`, and the
dedicated preparers; each retains its own input contract.

All entries use `core.research.display.export_scientific_suite` as the registered
renderer. Their file contract includes PNG, PDF, SVG, `physical_arrays.npz`,
`figure.json`, and `gallery.json`.

## Find a family

Each table lists the measurement, producer and **axes from retained examples**.
Read an axis pair as x → y; triples are x/y/z coordinates. `1` means dimensionless.
Multiple pairs describe different panels or producer routes. They illustrate the
actual exported units, rather than imposing one unit system on every future input.
For example, the native-analysis power route keeps h units while the dedicated
power exporter converts to Mpc. Always use the dataset contract for your own run.
Convergence controls can deliberately use native L/V units; their saved conversion
must be supplied before comparing them with a physical-time plot.

- [Cosmology](#cosmology) — 5 recipes
- [Abundance](#abundance) — 7 recipes
- [Halo structure](#halo-structure) — 10 recipes
- [Assembly histories](#assembly-histories) — 9 recipes
- [Cosmic web](#cosmic-web) — 4 recipes
- [Stellar streams](#stellar-streams) — 15 recipes
- [Encounters](#encounters) — 7 recipes
- [Black holes](#black-holes) — 8 recipes
- [Gas and stellar populations](#gas-and-stellar-populations) — 8 recipes
- [Self-interacting dark matter](#self-interacting-dark-matter) — 13 recipes
- [Hubble measurements](#hubble-measurements) — 4 recipes
- [Inference](#inference) — 6 recipes
- [Convergence](#convergence) — 6 recipes

## Cosmology

| Recipe ID | Required measurement | Producer route | Example axes |
|---|---|---|---|
| `input_power` | Input species/total spectra and normalization | `core.power.export_linear_power` | `1 / Mpc` → `Mpc^3` |
| `evolved_power` | Saved-epoch species/total spectra with assignment correction | `core.power.export_evolved_power` | `1 / Mpc` → `Mpc^3`<br>`h/Mpc` → `(Mpc/h)^3` |
| `power_ratios` | Matched epoch, cosmology and bin masks for spectrum ratios | `core.power.export_power_ensemble` | `1 / Mpc` → `1` |
| `transfer_comparison` | Signed density and velocity transfer columns on identical k grid | `core.power.export_linear_power` | `1 / Mpc` → `Mpc^-1`<br>`1 / Mpc` → `1` |
| `matched_residuals` | Matched phases, units and residual covariance | `core.power.export_power_ensemble` | `1 / Mpc` → `Mpc^3` |

**Input requirements**

- **`input_power`, `transfer_comparison`:** A validated schema 2 linear bundle with its original power/transfer files, source hashes, normalization and actual solver epoch. These exports do not require an evolved native run or paired covariance.
- **`evolved_power`:** A verified native analysis bundle containing saved power arrays, mode counts, errors and estimator metadata. The dedicated producer checks the source run and converts h-scaled k/P to physical Mpc units.
- **`power_ratios`, `matched_residuals`:** A `native-paired-power-ensemble-v1` request with independent parent realizations, original evolved spectra, common epochs/grids/estimator policy and measured within-pair IC phase agreement. The producer measures the full joint covariance from those realizations; independence needs its separate generation-design evidence.

**Read the plots:** Input power is a deterministic solver prediction. Evolved
power is a saved particle-field estimate. The paired plots show mean(B)/mean(A)
and mean(B)−mean(A), with support restricted to their measured phase range.
Density transfers retain their sign; velocity-transfer panels have inverse-length
units. See [power conventions](calculations.md#matter-power-correlation-and-bias).

## Abundance

| Recipe ID | Required measurement | Producer route | Example axes |
|---|---|---|---|
| `differential_hmf` | Resolved halo counts, volume and log-mass bins | `core.figures.export_native_analysis_figures` | `Msun/h` → `(Mpc/h)^-3` |
| `cumulative_hmf` | Same catalogue, cumulative thresholds and volume | `core.figures.export_native_analysis_figures` | `Msun/h` → `(Mpc/h)^-3` |
| `hmf_ratio` | Matched catalogues, common mass definition and completeness | `core.flagship_abundance.export_matched_host_abundance` | `Msun` → `1` |
| `reliability_limits` | Particle-count, contamination and finder completeness cuts | `core.flagship_abundance.export_native_host_abundance` | `Msun` → `particle count` |
| `stellar_mass_function` | Native stellar masses grouped by resolved galaxy | `core.flagship_abundance.export_native_abundance` | `Msun` → `count / comoving Mpc^3 / dex` |
| `vmax_function` | Resolved Vmax catalogue and common completeness | `core.flagship_abundance.export_native_abundance` | `km/s` → `count / comoving Mpc^3 / dex` |
| `rare_high_z_abundance` | Actual high-redshift outputs, volume and counting uncertainty | `core.flagship_abundance.export_native_host_abundance` | `Msun` → `count / comoving Mpc^3` |

**Input requirements**

- **`differential_hmf`, `cumulative_hmf`:** A verified analysis bundle containing the relevant `snap_*/hmf/data` datasets: differential mass/count/abundance arrays or cumulative thresholds and abundance. Dataset records must declare quantities, units, epochs and provenance; unrelated power/profile/history datasets do not replace missing HMF measurements.
- **`hmf_ratio`:** Two native host censuses with the same physical epoch, h convention, volume, mass definition, selection cut, and physical bins. Require a passing IC phase check and the existing paired jackknife measurements.
- **`reliability_limits`, `rare_high_z_abundance`:** Complete native group catalogues and the configured mass/count selection. Bind the ICs, periodic full-parent volume, and saved epoch to their recorded sources.
- **`stellar_mass_function`, `vmax_function`:** Pinned native galaxy/SUBFIND exports at the same epoch and identity namespace, with the original quality/count masks and verified full-parent volume.

**Read the plots:** Differential abundance is count/(volume × dex width);
cumulative abundance counts halos above a threshold inside the measured mass
window. The ratio uses a shared physical mass grid. Particle-count cuts define
which bins can be measured; they do not establish numerical convergence.
See [the HMF calculation](calculations.md#halo-abundances-and-reference-fits).

## Halo structure

| Recipe ID | Required measurement | Producer route | Example axes |
|---|---|---|---|
| `density_profile` | Bound membership, center and radial density bins | `core.halo_products.export_native_structure_figures` | `proper kpc` → `Msun/(proper kpc)^3` |
| `circular_velocity` | Enclosed total/species mass and proper radii | `core.halo_products.export_native_structure_figures` | `proper kpc` → `km/s` |
| `velocity_dispersion` | Frame-subtracted radial velocity moments and support | `core.halo_products.export_native_structure_figures` | `proper kpc` → `km/s` |
| `anisotropy` | Radial/tangential velocity moments with finite-count errors | `core.halo_products.export_native_structure_figures` | `proper kpc` → `1` |
| `concentration_mass` | Resolved profile fits, mass definition and fit covariance | `core.halo_products.export_native_structure_figures` | `Msun` → `1` |
| `radial_axis_ratios` | Iterative radial inertia tensors and sampling errors | `core.halo_products.export_native_structure_figures` | `proper kpc` → `1` |
| `triaxiality` | Ordered shape eigenvalues and degeneracy masks | `core.halo_products.export_native_structure_figures` | `proper kpc` → `1` |
| `axis_orientation_twist` | Eigenvector continuity, sign convention and degeneracy masks | `core.halo_products.export_native_structure_figures` | `proper kpc` → `deg` |
| `density_potential_shape` | Density shape and independently measured potential shape | `core.halo_products.export_native_structure_figures` | `proper kpc` → `1` |
| `splashback_slope_radius_depth_width` | Resolved outer radial slopes, fit support and feature uncertainty | `core.halo_products.export_native_splashback_figures` | `proper kpc` → `1` |

**Input requirements**

- **`density_profile`, `circular_velocity`, `velocity_dispersion`, `anisotropy`, `concentration_mass`, `radial_axis_ratios`, `triaxiality`, `axis_orientation_twist`, `density_potential_shape`:** A verified `native-halo-structure-v1` product with component profiles, density/equipotential shape grids, sampling masks, and the conditional NFW fit definitions.
- **`splashback_slope_radius_depth_width`:** A verified `native-environment-v1` product joined to the exact host structure, retaining the slope covariance and feature status.

**Read the plots:** Radius is proper. Density divides shell mass by shell volume,
and circular speed uses enclosed mass. Axis ratios approach one for a round
distribution; orientation is meaningful only outside the exported degeneracy
masks. Splashback panels show the measured logarithmic slope and supported
feature depth/width; an unresolved feature stays unresolved. The
[worked notebook](../optional_tests/reference_gallery/Walkthrough.ipynb) replays
the known-shape and density-slope controls from their original arrays.

## Assembly histories

| Recipe ID | Required measurement | Producer route | Example axes |
|---|---|---|---|
| `mass_scale_factor` | Exact main-progenitor IDs, scale factor and consistent mass | `core.assembly.export_native_assembly_figures` | `1` → `Msun` |
| `log_mass_accretion` | Time derivatives on supported mass histories | `core.assembly.export_native_assembly_figures` | `1` → `1` |
| `fast_slow_transition` | Measured accretion transition and fit uncertainty | `core.assembly.export_native_assembly_figures` | `1` → `Msun` |
| `spin_history` | Consistent spin definition along a verified tree | `core.assembly.export_native_assembly_figures` | `1` → `1` |
| `concentration_history` | Supported profile fits along the same tree | `core.assembly.export_native_assembly_figures` | `1` → `1` |
| `shape_history` | Shape tensors and orientation continuity along tree | `core.assembly.export_native_assembly_figures` | `1` → `1` |
| `pericenter_infall_merger_events` | Worldline extrema, infall definitions and actual merger links | `core.flagship_orbits.measure_native_assembly_orbits` → `core.flagship_orbits.export_native_assembly_event_figures` | `Myr` → `1` |
| `stellar_assembly` | Star birth IDs, in-situ/ex-situ membership and birth times | `core.assembly.export_native_assembly_figures` | `1` → `Msun` |
| `quenching_morphology` | SFR history, morphology estimator and frozen quenching definition | `core.assembly.export_native_assembly_figures` | `1` → `1 / year`<br>`1` → `1` |

**Input requirements**

- **`mass_scale_factor`, `log_mass_accretion`, `fast_slow_transition`, `spin_history`, `concentration_history`, `shape_history`, `stellar_assembly`, `quenching_morphology`:** A published `halo-main-branch-v1` stage and its exact identity graph. Retain saved epochs, source bindings, integer IDs, and quality/cadence masks.
- **`pericenter_infall_merger_events`:** The native graph and original snapshot/catalogue sources, measured quality, physical clocks, hash-verified recorded images, and original host-radius definition. Cadence and support flags must be explicit.

**Read the plots:** Follow the same branch along scale factor, not snapshot
number. A mass jump or fitted transition needs its branch/support record.
Event ordinates encode event categories; physical event times and definitions
remain in the numerical product. A pericenter or infall does not imply a merger.

## Cosmic web

| Recipe ID | Required measurement | Producer route | Example axes |
|---|---|---|---|
| `filament_map` | Periodic density field, smoothing scale and web classifier | `core.halo_products.export_native_web_figures` | `comoving kpc` → `comoving kpc` → `comoving kpc` |
| `spin_alignment` | Native spins and local filament axes with selection masks | `core.halo_products.export_native_web_figures` | `proper kpc` → `1` |
| `shape_alignment` | Native shape axes and web eigenvectors | `core.halo_products.export_native_web_figures` | `proper kpc` → `1` |
| `web_evolution` | Same web definition and smoothing across saved times | `core.halo_products.export_native_web_history` | `Myr` → `degrees` |

**Input requirements**

- **`filament_map`, `spin_alignment`, `shape_alignment`:** A verified `native-environment-v1` product containing periodic T-web nodes/labels and native halo-structure joins. Keep optional graph arcs, recorded images, and unique-assignment masks distinct from the T-web classification.
- **`web_evolution`:** A verified `native-environment-history-v1` product with persistent host graph nodes, saved proper-time brackets, unoriented axis changes, and accepted intervals bounded by the clock.

**Read the plots:** The map shows classified mesh nodes. Alignment is an axis
comparison with the producer's sign/absolute-cosine convention, not a signed
filament arrow. Time-dependent angles require both a supported local axis and
accepted time interval; their absence is not a zero-degree alignment.

## Stellar streams

| Recipe ID | Required measurement | Producer route | Example axes |
|---|---|---|---|
| `stream_progenitor_3d` | Native stellar IDs, progenitor membership and 3D samples | `core.stream.measurements.export_stream_support_figures` | `proper kpc` → `proper kpc` → `proper kpc` |
| `intrinsic_track` | Supported intrinsic principal curve and arc length | `core.flagship_products.export_registered_stream_figures` | `proper kpc` → `proper kpc` → `proper kpc` |
| `projected_track` | Declared observer, units and projection of native track | `core.flagship_products.derive_native_stream_product` → `core.flagship_products.export_registered_stream_projection_figures` | `proper kpc` → `proper kpc` |
| `curvature` | Track derivatives with curvature uncertainty | `core.flagship_products.derive_native_stream_product` → `core.flagship_products.export_registered_stream_projection_figures` | `proper kpc` → `1 / proper kpc` |
| `convexity` | Track orientation and signed-convexity convention | `core.flagship_products.derive_native_stream_product` → `core.flagship_products.export_registered_stream_projection_figures` | `proper kpc` → `1` |
| `linear_density` | Selection-corrected counts/mass per arc-length bin | `core.flagship_products.export_registered_stream_figures` | `proper kpc` → `Msun / proper kpc` |
| `width` | Transverse distribution, estimator and unresolved-width masks | `core.flagship_products.export_registered_stream_figures` | `proper kpc` → `proper kpc` |
| `velocity` | Stream-frame velocity mean and component definitions | `core.flagship_products.export_registered_stream_figures` | `proper kpc` → `km/s` |
| `dispersion` | Intrinsic velocity scatter, sampling uncertainty and masks | `core.flagship_products.export_registered_stream_figures` | `proper kpc` → `km / s` |
| `gaps` | Matched reference density, gap definition and significance | `core.flagship_products.export_registered_stream_figures` | `proper kpc` → `1` |
| `spurs` | Supported transverse residual peaks against matched track | `core.flagship_products.export_registered_stream_figures` | `proper kpc` → `proper kpc` |
| `bifurcations` | Separate wrap/branch membership and persistence evidence | `core.stream.measurements.measured_track_support_products` → `core.flagship_products.export_registered_stream_figures` | `proper kpc` → `proper kpc` |
| `action_angle` | Supported potential domain, action solver and error controls | `core.stream.measurements.export_stream_support_figures` | `radian` → `proper kpc km/s` |
| `density_power` | Density spectrum with window/noise treatment | `core.flagship_products.export_registered_stream_figures` | `1 / proper kpc` → `proper kpc` |
| `track_power` | Track residual spectrum with coordinate and window controls | `core.stream.measurements.measured_track_support_products` → `core.flagship_products.export_registered_stream_figures` | `1 / proper kpc` → `proper kpc^3` |

**Input requirements**

- **`stream_progenitor_3d`:** Native stellar IDs, progenitor membership, and three-dimensional samples. The numerical product must declare the source epoch/namespace, physical quantities, support, and uncertainty.
- **`intrinsic_track`, `linear_density`, `width`, `velocity`, `dispersion`, `gaps`, `spurs`, `density_power`:** A `stream-coordinates-v1` product with measured wraps, per-dataset `numeric_data` contracts, bootstrap arrays, and support masks.
- **`projected_track`, `curvature`, `convexity`:** A `stream-coordinates-v1` product passed through the stream geometry/observer stage, with an explicit viewing basis, host centre/covariance, local widths, and support.
- **`bifurcations`:** Local two-mode fork proposals. These do not by themselves establish persistent physical branches or a causal encounter.
- **`action_angle`:** A supported potential domain, action solver, and numerical error controls. The numerical product must declare its source epoch/namespace, physical quantities, support, and uncertainty.
- **`track_power`:** Raw two-component power in transverse bin-centroid residuals about the fitted curve, with Hann/support windows. Conditional noise from width covariance and effective count is published separately. Units are proper kpc³; no noise/baseline fit or subtraction is applied.

**Read the plots:** Track distance is proper arclength; width and spur amplitudes
are transverse lengths. Density is mass per arclength, and velocity/dispersion
describe the stream frame. Power uses inverse arclength and its recorded window.
Fork/gap candidates are measured diagnostics whose persistence or cause needs
additional evidence. The action panel uses the explicitly selected potential;
failed quadratures remain masked.

## Encounters

| Recipe ID | Required measurement | Producer route | Example axes |
|---|---|---|---|
| `worldlines_3d` | Persistent perturber IDs and stream worldlines in one frame | `core.flagship_products.native_encounter_frame` → `core.flagship_products.measure_native_encounters` → `core.flagship_products.export_native_encounter_figures` | `proper kpc` → `proper kpc` → `proper kpc` |
| `impact_velocity_geometry` | Certified closest approach, relative speed and orientation | `core.flagship_products.native_encounter_frame` → `core.flagship_products.measure_native_encounters` → `core.flagship_products.export_native_encounter_figures` | `Myr` → `proper kpc`<br>`km / s` → `proper kpc` |
| `perturber_mass_profile` | Time-dependent bound profile and mass-definition support | `core.stream.perturbers.native_perturber_history_runner` | `proper kpc` → `Msun` |
| `before_after_controls` | Same tracers, times and host with declared intervention | `core.native_stream_responses.native_response_runner` | `proper kpc` → `Msun / proper kpc`<br>`proper kpc` → `km/s`<br>`proper kpc` → `proper kpc` |
| `gap_growth` | Same gap definition through time and matched baseline | `core.native_stream_responses.native_response_runner` | `Myr` → `1` |
| `action_diffusion` | Supported action histories and matched stochastic/control variance | `core.native_stream_responses.native_response_runner` | `Myr` → `proper kpc^2 (km/s)^2` |
| `bh_extended_subhalo_comparison` | Matched compact/extended perturber masses, orbits and force resolution | `core.matched_perturber_figures.matched_template_figures_runner` | `proper kpc` → `Msun / proper kpc`<br>`proper kpc` → `km/s`<br>`proper kpc` → `proper kpc` |

**Input requirements**

- **`worldlines_3d`, `impact_velocity_geometry`:** A `native-encounter-frame-history-v1` product with explicit interval flags and a frame-definition hash. The `stream-encounters-v1` product retains refinement requests and unresolved intervals.
- **`perturber_mass_profile`:** The exact native descendant graph, bound memberships, gravitating-particle census, recorded images, and converted stream frame.
- **`before_after_controls`, `gap_growth`, `action_diffusion`:** A replayed native paired comparison using the exact selected stellar cohort and the same physical coordinate chart tied to its sources; measured host actions are optional.
- **`bh_extended_subhalo_comparison`:** Executed matched imposed-field templates with the same design, realization, initial cohort, geometry, and bins.

**Read the plots:** Separate saved worldline positions from certified continuous
closest approaches. Paired responses compare the same tracers and coordinate
chart. `action_diffusion` in the retained control is an action-increment second
moment versus elapsed time, with units of action squared; it is not divided by
time and is not an independently measured diffusion coefficient.

## Black holes

| Recipe ID | Required measurement | Producer route | Example axes |
|---|---|---|---|
| `bh_growth` | Persistent native BH IDs, mass and saved times | `core.flagship_bh.export_native_bh_figures` | `1` → `Msun` |
| `eddington_ratio` | Native accretion rate and consistent Eddington convention | `core.flagship_bh.export_native_bh_figures` | `1` → `1` |
| `pair_separation` | Actual BH pair identities and proper separation | `core.flagship_bh.export_native_bh_pair_separation` | `1` → `proper kpc` |
| `merger_timeline` | Actual merger event records and ancestry | `core.flagship_bh.export_native_bh_merger_timeline` | `Myr` → `1` |
| `recoil_trajectory` | Implemented kick physics and native post-kick trajectory | `core.flagship_bh.export_native_bh_recoil_trajectory` | `proper kpc` → `proper kpc` → `proper kpc` |
| `feedback_energy` | Native accumulated feedback energy and time convention | `core.flagship_bh.export_native_bh_figures` | `1` → `erg` |
| `dm_density_response` | Matched BH feedback intervention and DM profiles | `core.flagship_bh_response.export_native_bh_dm_response` | `proper kpc` → `Msun/(proper kpc)^3` |
| `dm_shape_response` | Matched feedback intervention and resolved DM shape tensors | `core.flagship_bh_response.export_native_bh_dm_response` | `proper kpc` → `1` |

**Input requirements**

- **`bh_growth`, `eddington_ratio`, `merger_timeline`, `feedback_energy`:** A native BH history tied to its recorded sources, with exact namespace/ID/epoch and agreement with the original field summaries. Merger timelines also require independently replayed native hydro history and a complete committed BH journal.
- **`pair_separation`:** A native BH history for two exact IDs in the same declared namespace. Use saved proper coordinates/epochs and the periodic-build convention, retaining missing values.
- **`recoil_trajectory`:** A native BH history and independently replayed complete hydro journal for the exact selected merger counter and survivor. Include hash-verified images at saved epochs, the physical clock, and subsequent ancestry masks.
- **`dm_density_response`, `dm_shape_response`:** Verified native structures and replayed, reciprocal, unique membership matches. Require passing phase agreement; matching epochs, proper bins, species, and estimator; a BH-only configuration contrast; and full joint covariance tied to the exact source hashes.

**Read the plots:** Growth follows subgrid BH mass. The inflow/cap ratio compares
a source-step mean with a cap evaluated at saved mass; feedback energy is an
accumulated source field. Pair separation alone does not establish binding or
coalescence. Merger and recoil panels require their actual event journal.

## Gas and stellar populations

| Recipe ID | Required measurement | Producer route | Example axes |
|---|---|---|---|
| `gas_phase_diagram` | Native gas density, temperature, mass and physical units | `core.baryon_products.export_native_gas_phase` | `g/cm^3` → `K` |
| `star_formation_history` | Complete native gas-SFR census, physical units and saved epochs | `core.baryon_products.export_native_sfr_history` | `1` → `Msun/yr` |
| `mass_return` | Verified delayed-return table and native deposited mass ledger | `core.baryon_products.export_native_mass_return` | `1` → `Msun` |
| `metals_history` | Element-resolved stellar/gas metal ledgers and closure | `core.baryon_products.export_native_metal_history` | `1` → `Msun` |
| `luminosity_history` | Pinned SPS model, native age/Z/mass and filter response | `core.baryon_products.export_native_population_history` | `Myr` → `Lsun` |
| `color_history` | Same SPS assumptions and explicit passbands | `core.baryon_products.export_native_population_history` | `Myr` → `AB mag` |
| `first_cooling_birth` | Saved cooling/birth events with sufficient time resolution | `core.first_stars.export_native_first_star_figures` | `Myr` → `1` |
| `primordial_cloud_morphology` | Native cloud membership and resolved gas morphology | `core.first_stars.export_native_cloud_figures` | `proper kpc` → `proper kpc` → `proper kpc` |

**Input requirements**

- **`gas_phase_diagram`:** A hash-verified native hydro history and explicit positive physical bin edges, retaining availability masks and excluding effective-EOS gas where the measurement requires it.
- **`star_formation_history`:** A verified native hydro history with a complete gas-SFR census, matching original summary, and saved cosmological clock. Use `build_star_formation_history`, preserving gaps and measured zero rates.
- **`mass_return`:** A verified committed stellar-return journal and complete snapshot counters. Keep signed stellar-source loss separate from gas-recipient gain since the first selected epoch; exclude prompt-return bookkeeping.
- **`metals_history`:** An independently verified `native-hydro-history-v1` product with complete eleven-element gas fractions, SSP initial locked birth masses/metallicities, and delayed-return source/recipient rows. Current stellar chemical mass remains explicitly unavailable for a nonempty stellar census.
- **`luminosity_history`, `color_history`:** A `native-population-history-request-v1` request with hash-verified native hydro history and pinned FSPS configuration/filter assets. Require all native type-4 birth states, the proper clock, and exact namespace. Incomplete SSP support masks whole-census totals; intrinsic rest-frame AB colours use summed band Lnu.
- **`first_cooling_birth`:** A `first-star-environments-v1` product tied to its original sources. Cooling/birth times must match the numerical rows; preserve namespace identities, saved intervals, and host-ancestry brackets.
- **`primordial_cloud_morphology`:** A primordial-cloud product tied to its original sources, with measured proper Cartesian points/centroids and shape support. Retain selected gas IDs, namespace, and epoch where available, including empty results for unsupported physics.

**Read the plots:** The phase diagram bins gas mass in density and temperature;
check whether effective-EOS gas is excluded. SFR comes from saved gas source
rates, while delayed returns use their committed journal. Luminosity sums
initial-mass-normalized populations; AB colour is formed from summed band flux
densities. A seeded stellar population can exercise light synthesis without
demonstrating a new star-formation event.

## Self-interacting dark matter

| Recipe ID | Required measurement | Producer route | Example axes |
|---|---|---|---|
| `sidm_core_evolution` | Native scattering-enabled resolved core history | `core.sidm_products.export_native_sidm_history` | `Gyr` → `Msun / proper kpc^3` |
| `collapse_phase` | Resolved density/dispersion evolution and collapse criterion | `core.sidm_products.export_native_sidm_history` | `Gyr` → `1` |
| `scattering_rates` | Native event/rate ledger and physical cross-section model | `core.sidm_products.export_native_scattering_rates` | `Gyr` → `Gyr^-1` |
| `sidm_cdm_controls` | Matched SIDM/CDM phases, settings and catalogue selection | `core.sidm_products.export_native_sidm_matched` | `Gyr` → `Msun / proper kpc^3`<br>`Gyr` → `1`<br>`proper kpc` → `(km / s)^2`<br>`proper kpc` → `Msun / proper kpc^3`<br>`proper kpc` → `1` |
| `sidm_density_dispersion_anisotropy` | Resolved native SIDM profiles and velocity tensors | `core.sidm_products.export_native_sidm_history` | `proper kpc` → `(km / s)^2`<br>`proper kpc` → `Msun / proper kpc^3`<br>`proper kpc` → `1` |
| `sidm_shape_baryon_misalignment` | SIDM/baryon axes, angular degeneracy and shared frame | `core.sidm_products.export_native_sidm_history` | `Gyr` → `degrees` |
| `sidm_heat_flux` | Measured energy flux with sign convention and conservation | `core.sidm_products.export_native_sidm_history` | `proper kpc` → `erg / cm^2 / s` |
| `sidm_kernel_support` | Actual neighbor/kernel support and particle sampling | `core.sidm_products.export_native_sidm_history` | `proper kpc` → `1` |
| `sidm_satellite_survival` | Persistent satellite IDs, disruption and completeness | `core.sidm_products.export_native_sidm_extended` | `Gyr` → `Msun`<br>`Gyr` → `1` |
| `sidm_merger_centroid_offsets` | Matched merging components, native centroids and errors | `core.sidm_products.export_native_sidm_centroids` | `Gyr` → `proper kpc` |
| `sidm_stream_profile_response` | Matched native streams with resolved SIDM perturber profiles | `core.sidm_products.export_native_sidm_extended` | `1` → `Msun / proper kpc`<br>`1` → `1`<br>`1` → `km / s` |
| `sidm_lensing_h0_response` | Native mass/lightcone/lens products plus supported inference | `core.sidm_products.export_native_sidm_extended` | `1` → `Msun / proper kpc^2`<br>`1` → `1` |
| `sidm_baryonic_degeneracy` | Matched baryonic interventions and SIDM parameter support | `core.sidm_products.export_native_sidm_extended` | `1` → `dimensionless in declared scaled parameter coordinates` |

**Input requirements**

- **`sidm_core_evolution`, `collapse_phase`, `scattering_rates`, `sidm_cdm_controls`, `sidm_density_dispersion_anisotropy`, `sidm_shape_baryon_misalignment`, `sidm_heat_flux`, `sidm_kernel_support`, `sidm_satellite_survival`, `sidm_merger_centroid_offsets`, `sidm_stream_profile_response`, `sidm_lensing_h0_response`:** Native target, model, clock, and identity products tied to their original sources, with unchanged dimensional support. Missing native support cannot be treated as a detection.
- **`sidm_baryonic_degeneracy`:** Native target, model, clock, and identity products tied to their original sources, with unchanged dimensional support and an independent held-out calibration.

**Read the plots:** Keep collision participations, integrated exposure and rates
distinct. A squared velocity dispersion has squared velocity units. Heat flux
needs its source sign convention. Relative and floor-scaled responses use the
saved denominator/floor; a small number alone does not establish insensitivity.
Collapsed cores, disrupted satellites and baryonic degeneracy require their
individual support and acceptance decisions.

## Hubble measurements

| Recipe ID | Required measurement | Producer route | Example axes |
|---|---|---|---|
| `hubble_method_comparison` | Actual method likelihoods/posteriors with common convention | `core.expansion_products.export_hubble_figures` | `1` → `km/s/Mpc` |
| `distance_ladder` | Observed/calibrated distance-ladder data and likelihood | `core.expansion_products.export_hubble_figures` | `1` → `km/s/Mpc` |
| `bao_cmb_background` | Pinned BAO/CMB likelihood and background model | `core.expansion_products.export_hubble_figures` | `1` → `1` |
| `time_delay_posteriors` | Actual time-delay lens likelihood, nuisance and posterior samples | `core.expansion_products.export_hubble_figures` | `Mpc` → `1`<br>`km/s/Mpc` → `1` |

**Input requirements**

- **All recipes in this family:** A completed cosmography/stream inference product tied to its original sources, with physical prior/observable units, covariance, diagnostics, independent calibration, and out-of-domain support. See `configs/research/analysis_branch_input_contracts.json`.

**Read the plots:** H₀ estimates use km/s/Mpc, while time-delay distances use Mpc.
The retained `bao_cmb_background` example compares measured BAO distance ratios
with a fixed CLASS prediction; its registry name does not add a CMB likelihood.
Read method labels and source likelihoods before comparing intervals.

## Inference

| Recipe ID | Required measurement | Producer route | Example axes |
|---|---|---|---|
| `actual_posterior` | Actual validated likelihood/simulator posterior samples | `core.flagship_inference.export_inference_figures` | `Msun` → `1`<br>`1` → `1` |
| `posterior_predictive` | Held-out forward predictions and observed-data contract | `core.flagship_inference.export_inference_figures` | `1` → `1` |
| `sbc_ranks` | Independent calibration simulations, truths and ranks | `core.flagship_inference.export_inference_figures` | `1` → `1` |
| `coverage` | Held-out truths, credible regions and sampling uncertainty | `core.flagship_inference.export_inference_figures` | `1` → `1` |
| `ood_support` | Training/support domain and measured out-of-domain scores | `core.flagship_inference.export_inference_figures` | `1` → `1` |
| `graph_baseline_gain` | Same held-out lineage split, baseline and graph-model metric | `core.flagship_inference.export_inference_figures` | Unresolved; intended log-score/index and bootstrap CDF panels use `1` |

The retained harmonic-control example executes the actual SBI summary model
and held-out rank/coverage evaluation. Its forward-domain acceptance applies
only to an analytic, massless-tracer numerical exercise with Gaussian noise
and complete selection. Calibration failed, and that result is plotted. A
successful figure export does not enable posterior deployment: the OOD
posterior route requires independently accepted calibration, and graph gain
requires both original posterior families to pass their calibration checks.

The required order is: freeze the models and held-out lineage split → evaluate
independent calibration for each family → require both original acceptance
decisions → compare paired held-out log scores → apply the predeclared bootstrap
gain threshold. The current rejected calibration stops this sequence before
gain production. Do not weaken thresholds or reuse held-out truth for tuning
to make this recipe render.

**Input requirements**

- **All recipes in this family:** A completed cosmography/stream inference product tied to its original sources, with physical prior/observable units, covariance, diagnostics, independent calibration, and out-of-domain support. See `configs/research/analysis_branch_input_contracts.json`.

**Read the plots:** Posterior panels show weighted CDFs in each parameter's units.
Predictive panels compare observed and replicated covariance-weighted
discrepancies. SBC shows randomized rank/PIT CDFs; coverage shows fractions and
binomial intervals. OOD shows the original distance/outside-range metrics.
These are different diagnostics; neither a plotted CDF nor an OOD rejection
establishes accepted posterior calibration.

## Convergence

| Recipe ID | Required measurement | Producer route | Example axes |
|---|---|---|---|
| `mass_resolution` | Same physical targets across mass levels, matched epochs | `core.flagship_products.export_convergence_figures` | `native elapsed time L/V` → `native length L` |
| `softening` | Resolved same-target softening ladder and observables | `core.flagship_products.export_convergence_figures` | `native elapsed time L/V` → `native acceleration V^2/L` |
| `timestep` | Resolved same-target integration-step ladder and observables | `core.flagship_products.export_convergence_figures` | `native elapsed time L/V` → `1` |
| `output_cadence` | Dense saved outputs and measured thinning/interpolation error | `core.flagship_products.export_convergence_figures` | `native elapsed time L/V` → `native length L` |
| `start_redshift` | Same phases and final observables across start epochs | `core.flagship_products.export_convergence_figures` | `1` → `km/s` |
| `finder_tree_variation` | Same native snapshots through finder/tree alternatives | `core.flagship_products.export_convergence_figures` | `1` → `count` |

**Input requirements**

- **All recipes in this family:** The `compare_campaign` comparison contract, measurements verified against their recorded hashes, and physical `bin_coordinate` values.

**Read the plots:** Each control changes the named numerical setting while
holding its declared inputs fixed. Read the observable and reference direction
from the comparison record: the retained examples include RMS-radius,
acceleration and orbit-error contrasts. Their axes need not be the changed
setting itself. Acceptance applies to that observable, target and tolerance.

## Read the result with its data

Keep the numerical arrays, masks, figure metadata, and source records together. The
image is a presentation of those measurements. A visually complete panel may still have
restricted calibration, incomplete support, or an unresolved scientific interpretation;
the corresponding records describe those limits.


## Reproduce a halo structure and web export

The structure producer reads the selected host's catalogue and particles. It
publishes shell statistics, shape measurements, support masks, and the source
hashes in a sealed numeric product. The environment producer uses the saved
snapshot and the exact structure product selected in its request to compute
the mesh, classify the tidal field, and evaluate halo alignment and splashback.

```sh
python commands.py halo-structure STRUCTURE_REQUEST.json --output STRUCTURE.json
python commands.py flagship-products structure-figures STRUCTURE.json --output STRUCTURE_FIGURES
python commands.py environment ENVIRONMENT_REQUEST.json --output ENVIRONMENT.json
python commands.py flagship-products web-figures ENVIRONMENT.json --output WEB_FIGURES
python commands.py flagship-products splashback-figures ENVIRONMENT.json --output SPLASHBACK_FIGURES --memory-gib 8
```

Use fresh output paths and keep the measurement JSON with its numeric sidecar
directory. Set the memory budget for the size of the environment metadata.
Open each export's `gallery/index.html` and inspect the series support counts
in `gallery/gallery.json`. A classified voxel map can have supported regions
while the same control provides zero supported halo alignments. Web evolution
additionally needs an identity graph and saved intervals with an accepted
proper-time clock. The synthetic orbit smoke control exercises those checks.
See the [smoke report](smoke-test.md).


## Inspect an assembly history

The assembly runner reads a sealed identity graph. Its branch estimator uses
explicit main-progenitor links or measured bound mass transfer, saved epochs,
and per-observation quality flags. It preserves gaps and writes the branch,
interval rates, and fitted-transition result to the assembly product. Export
a sealed product with its exact graph directory.

```sh
python commands.py flagship-products assembly-figures ASSEMBLY.json --identity-graph GRAPH_DIRECTORY --output FIGURE_DIRECTORY
```

Inspect the fit status in the assembly JSON before interpreting the transition
plot. That figure can show the measured mass sequence even when the fit finds
no supported breakpoint. The disposable orbit control uses a fixed proper DM
aperture and fixture ancestry. Its measured trends exercise the exporter,
while physical merger histories require ancestry from a native catalogue.


## Export stream coordinates and viewing geometry

The coordinate stage selects a population and fits an intrinsic track. It saves
the position, velocity, transverse covariance, sampling uncertainty, and
support masks by wrap. Geometry uses the fitted track and its supported width
bins to measure projected tracks, curvature, and convexity in specified views.

```sh
python commands.py flagship-products stream-figures COORDINATES.json --output COORDINATE_FIGURES
python commands.py flagship-products stream-derived COORDINATES.json --request GEOMETRY_REQUEST.json --output GEOMETRY.json
python commands.py flagship-products stream-figures GEOMETRY.json --output GEOMETRY_FIGURES
```

Use a `flagship-derived-stream-v1` request with `kind: geometry` and the view
controls in [the stream campaign example](../configs/research/blocks16_20.example.json).
The geometry stage requires at least two resolved width bins. A measurable
track can still have too few supported bins for power. The retained Gala
control exercises both cases. Its equal tracer weights set the normalization
of the density panels, and its feature candidates are computed diagnostics
from the simulated spray.


The stream power controls default to requiring every spatial bin. The
[native stream epoch example](../configs/research/flagship_native_stream_epoch.example.json)
declares eight as its minimum. The second Gala control uses that setting and
retains the Hann window and mask so the nine contributing bins are visible
in the sidecar. The action-angle route uses a separate, explicitly declared
potential. Its sealed arrays retain failed quadrature rows and, for the
isochrone control, differences against the analytic actions. Keep the original
solver settings with the product when comparing runs.


## Compare paired halo mass functions

Create one host census per run with `flagship-products host-abundance`. The
`flagship-matched-host-abundance-v1` request binds those two JSON products by
path and SHA-256 and specifies shared physical mass bins. Run the paired
export after both censuses succeed.

```sh
python commands.py flagship-products matched-host-abundance REQUEST.json --output HMF_RATIO_FIGURES
```

The exporter checks the saved scale factor, mass definition, particle cut,
volume, and the Fourier phases of the original IC payloads. Its arrays retain
the counts for both runs and mask unsupported ratios. The disposable repeat
uses the same ICs and a five-mode minimum specific to its tiny mesh. The
repeated catalogue exercises reproducibility at that size.

## Reproduce stellar light histories

Run the hydro history audit on at least two saved epochs. Bind its sealed
JSON by path and SHA-256 in a `native-population-history-request-v1` request.
Set the FSPS data root, source hashes, expected libraries, filters, and
age and metallicity domains. The synthesis settings in the
[stellar light example](../configs/research/block29_stellar_light.example.json)
show the expected units and model choices.

```sh
python commands.py audit-hydro-history HYDRO_REQUEST.json --output HYDRO_HISTORY.json
python commands.py flagship-products population-history-figures POPULATION_REQUEST.json --output POPULATION_FIGURES
```

The exporter reads the birth state for each saved stellar particle and
normalizes its spectrum by the initial formed mass. It saves one synthesis
product per epoch, then sums bolometric luminosities and band flux densities
before calculating the integrated AB color. The history sidecars preserve
the support flags for each epoch. A seeded stellar particle exercises this
light pipeline without a new birth event. The delayed-return model has its
own yield-table preflight and requires accepted source rows before evolution.


## Inspect SIDM scattering histories

A registered scattering-rate plot takes a sealed `sidm-halo-history-v1`
product and its original run directory. The halo-history producer requires
a completed evolution receipt and an exact measured membership selection
for each saved epoch. Use the same run and snapshots when exporting.

```sh
python commands.py export-sidm-figures scattering HALO_HISTORY.json SCATTERING_FIGURES --run-directory ORIGINAL_RUN
```

The reducer follows the first saved fine-DM cohort by exact ID, subtracts
the original unsigned interaction counters, and divides by the saved proper
time interval. It separately reduces the integrated exposure, which includes
the operator's optical depth. The output keeps the per-step guard and clock
support masks. The 32-particle homogeneous smoke control reconciles event
and participation counts and writes diagnostic rates. Halo membership and
the halo workflow receipt are separate inputs to the registered route.

The retained halo smoke example supplies that separate route with 24 DM
particles and eight seeded stars. Its actual source-bound version-one guards
and clock bounds support both saved intervals. Zero event participations
and nonzero optical depth remain separate plotted quantities. Its compact
synthetic ICs and tiny cross section exercise execution; they do not resolve
a SIDM core. Unsupported JSON metadata is null; the corresponding numerical
NaNs and support masks remain in the sidecar.

## Measure the separation of two saved black holes

Produce the sealed history with `core.black_holes.analyze_black_holes(RUN)`;
its return value names the history directory. Read the exact integer IDs from
`history.hdf5` and the black-hole identity namespace from `provenance.json`.
The pair request has this form; replace the IDs and namespace with your values:

```json
{
  "schema": "native-bh-pair-separation-v1",
  "particle_ids": [1152921504606847042, 1152921504606847041],
  "namespace": "COPY_THE_HISTORY_BLACK_HOLE_NAMESPACE"
}
```

```sh
python commands.py flagship-products bh-pair-figures BH_HISTORY_DIRECTORY --request PAIR.json --run-directory ORIGINAL_RUN --output PAIR_FIGURES
```

The [pair exporter](../core/flagship_bh.py) joins those IDs at each saved epoch,
reads the original coordinates and periodic box, and converts both to proper
kpc. For periodic runs it measures the minimum-image distance. Its sidecar
preserves positions, box sizes, IDs, scale factors and support masks. Missing
members remain gaps. The control starts at 0.1 and 0.9 of the box; the wrapped
separation is 0.2 box lengths. Small plotted changes may use axis offsets, so
inspect the sidecar for absolute distances. Separation alone does not establish
binary binding, coalescence or recoil; those recipes require their own events.

## Build a measured paired-power ensemble

Generate at least two independent parent IC realizations. Evolve both comparison
models from each parent, preserving the seed, original ICs, configuration and
native manifest. Export selected native spectra with `flagship-products
power-evolved` before building the ensemble request.

```sh
python commands.py flagship-products power-ensemble ENSEMBLE_REQUEST.json --output ENSEMBLE_FIGURES --memory-gib 2
```

The `native-paired-power-ensemble-v1` request contains `minimum_realizations`,
`denominator_floor_mpc3`, `independence_design`, and `realizations`. Each
realization supplies a unique `realization_id`, unique `parent_phase_namespace`,
and `left`/`right` path-and-SHA256 bindings to `snap_NNN-power.json`. The
independence design supplies `assertion: independent_parent_random_streams`,
a description of the generation procedure and a bound evidence file.
The producer checks distinct seeds and IC payload hashes across parents, plus
the measured within-pair phase agreement. A namespace cannot prove independence.

The [ensemble producer](../core/power.py) intersects support across parents,
requires identical saved epochs, physical k grids and estimator policies,
then calculates the full covariance of the ordered `[P_left, P_right]` vector.
It divides the sample covariance by the number of parents for the covariance
of the means. Cross-bin and left/right blocks enter ratio and residual errors.
These are ratios of means, not means of individual ratios. Signed
shot-noise-subtracted powers remain signed; a ratio near one of two negative
estimates does not establish detection of physical clustering.

The 4³ control had no saved spectral bins inside its measured phase range.
The 8³ control supplied two. Identical-physics repeats measured ratio one
and residual zero. Two parents give a rank-deficient covariance estimate
and cannot establish ensemble precision.

## Plot an actual distance fit or posterior

The [ladder example](../configs/research/block42_distance_ladder.example.json)
declares host, indicator-prescription and anchor inputs separately. Supply
those assets before executing the campaign:

```sh
python commands.py cosmography CAMPAIGN.json --output COSMOGRAPHY_OUTPUT
python commands.py export-hubble-figures HUBBLE_REQUEST.json --output HUBBLE_FIGURES
python commands.py export-inference-figures INFERENCE_REQUEST.json --output INFERENCE_FIGURES
```

A `native-hubble-figure-request-v1` ladder request names `variant:
distance_ladder` and binds the original campaign stage's `product.json` under
`product`. The exporter verifies its generation manifest, producer sources and
input hashes. It plots the jointly fitted H₀ and interval. That interval is a
fit uncertainty, not invented posterior draws. Use `complete` likelihood only
for a complete sample. A selected sample needs its selection likelihood;
silently switching this choice is not a recovery for a failed selected fit.

For `variant: bao_cmb_background`, supply `bao_request` instead: bindings for
the measured compressed BAO product, frozen linear bundle, and actual CLASS
drag execution, log and invocation. The
[shipped provenance](../configs/observations/bao/source-provenance.json)
binds the original DESI tables. The exporter retains all measured ratios and
their covariance, comparing them with one fixed CLASS prediction. Despite the
registry name, this route does not add a CMB likelihood.

A sampled BAO campaign uses `kind: bao_compressed_posterior`, ordered physical
priors for `alpha_perp` and `alpha_parallel`, explicit emcee settings and a
checkpoint directory. Bind its published `product.json` in a
`native-inference-figure-request-v1` request with `variant: actual_posterior`.
The plot reads retained weighted draws and preserves sampler diagnostics.
These dimensionless distance dilations relative to the fixed model are not
a full cosmological or H₀ posterior. A tiny chain can render while correctly
reporting that it is too short for inference.

## Interpret concentration and timestep smoke plots

The structure request's optional `concentration` block must name its actual
`reference_density_msun_kpc3`, `overdensity_delta`, `definition_label`,
`fit_min_kpc`, `fit_max_kpc`, `rs_min_kpc`, `rs_max_kpc` and
`minimum_effective_count`. The producer brackets the total overdensity radius
from measured enclosed masses. It then fits the NFW log density on at least
five supported component shells. A unique radius crossing and an interior,
rank-two fit are required. The concentration plot uses total overdensity mass
on its x axis and the component fit's R_delta/r_s on its y axis. The uncertainty
is conditional on the supplied profile model and reference density. Inspect
χ², degrees of freedom and fit bounds before interpreting it; a numerical
`MEASURED` status does not show that NFW is an adequate density model.

The `flagship-products convergence-figures` request binds original measured
records and declares one changed control, common bins, units, phase and
observable definition. Specify absolute/relative tolerances, a dimensional
floor, uncertainty multiplier and covariance tolerance before comparison.
Paired random estimators need their measured cross covariance. An exact
deterministic statistic on a fixed tracer set can instead have zero statistical
covariance if that scope is stated explicitly. It still has numerical error
and says nothing about an unmeasured population. The retained timestep example
plots differences in the maximum relative orbit-position error across the
same 28 Plummer tracers at five actual saved times. It is a timestep control,
not a resolution or cosmological convergence study.

### Native BH merger and recoil

Start with a run that saves the native BH event journal, exact particle IDs,
periodic image counters and a hash-bound expansion history. Build the
snapshot BH catalogue and audit the hydro journal for the requested epochs.
The merger timeline selects audited rows with `kind=merger` and retains
the native source-step bracket. The plotted event time is the end of that
step. For recoil, select the survivor ID and event counter from the same
row, then publish its original periodic image counters. The trajectory
exporter joins this selection to the saved BH catalogue and unwraps the
measured positions using the recorded images. The saved data include the
physical kick, the kick of the gravitating particle and their units.
A short control can supply only a few points after the event. The retained
control supplies two and is documented in the [smoke test report](smoke-test.md).

### Softening and saved-output cadence

For the softening contrast, use identical initial particles and hold the
integrator, force settings, seed and output times fixed. The retained
32-particle control computes the RMS norm of the native acceleration over
the complete particle set at each saved time, then subtracts the reference
run's RMS. The acceleration unit is the native velocity unit squared
divided by the native length unit. A zero contrast at small softenings
can occur when no pair reaches the softened part of the force law.
The wider retained pair (0.01 and 0.30) gives a measurable decrease
in RMS acceleration across all four saved epochs. Each pair tests the
force response of the fixed load at the specified softenings.

For the cadence contrast, measure the same statistic at all saved epochs
before withholding any snapshots. The retained Plummer control computes
the RMS radius of the fixed tracer set, keeps every other saved epoch
plus the endpoints, and linearly interpolates the RMS radius at the
withheld times. The plot shows reconstructed minus measured radius.
Retained endpoints test the interpolation boundary condition. The
withheld points measure cadence error in this statistic on this orbit.
A different statistic, particularly a sharp event, may require a
different output schedule.


### Three-dimensional stellar debris and progenitor

The 32-star live control samples an isotropic Plummer distribution and
evolves the particles with kick–drift–kick integration in a static Hernquist
host. The host potential is `−GM/(r+a)`. At each saved epoch, the remnant
estimator iteratively removes particles whose specific kinetic energy
relative to the remnant plus its softened self-potential is nonnegative.
The plotted debris mask is the intersection of the initially bound and
currently unbound sets. Current bound members are shown separately.
The three axes are host-centered proper kpc. The live calculation and
registered exporter are exercised on this control. For a production
plot, use the source-bound native stellar catalogue and its recorded
progenitor labels and membership masks.


### Matched compact and extended encounter

The 32-tracer control evolves identical initial positions, velocities and
masses in two imposed fields with SciPy DOP853. Both use a harmonic host and
the same 10⁷ solar-mass flyby trajectory. The compact and extended Plummer
scale radii are 0.005 and 0.25 proper kpc. Each field uses the potential
`−GM/√(r²+b²)`. The observer and fixed track are shared.

Project each saved particle onto the same track and bin by arclength.
Line density is the sum of tracer mass divided by bin width, in solar
masses per proper kpc. The other panels show the norm of the difference
in mass-weighted transverse centroids and the norm of the difference in
mass-weighted local velocities. A bin enters the comparison only when both
models meet the minimum particle count. This experiment measures the response
to imposed profile compactness. Native BH and subhalo ancestry and encounter
calibration require separate evidence.


### Finder variation on a fixed snapshot pair

The finder control verifies that both codes consumed the same snapshot
checksums and used the recorded scale factors. It counts rows in each
complete catalogue and plots each count minus the Rockstar count at the
same scale factor. An empty catalogue contributes zero when the finder
completed and its output checksums verify. The measured count is a census
conditional on the exact inputs and settings. Its zero sampling
covariance does not represent uncertainty across halo populations.
Use this figure to inspect differences between finder configurations.
A matched-halo comparison additionally needs shared particle memberships
and aligned mass definitions. A tree comparison needs the same catalogue
sequence linked under controlled alternative settings.


### Mass resolution in a fixed Plummer model

The 32- and 64-particle isolated GADGET runs share the analytic Plummer
profile, total mass, seed, force settings and saved times. Doubling the
count halves the native particle mass. The generator resamples the
radial quantiles at each count, so particle IDs cannot be paired between
the two resolutions.

At each saved time, compute the mass-weighted center
`x_cm = Σ m_i x_i / Σ m_i`, then compute
`R_rms = √[Σ m_i |x_i − x_cm|² / Σ m_i]` in native length units.
The plot subtracts the 32-particle value at the same time. The initial
difference measures finite sampling of the same model. Later differences
combine that sampling with the response to evolution at each resolution.


### Cosmological start redshift

Use CLASS to regenerate the input spectrum and transfer for each start,
then use N-GenIC with the same seed and evolve the two loads with the
same native settings. Compare only the common saved epochs. The
64-particle control checks exact particle IDs and accepts actual scale
factors within 10⁻¹⁰ of each requested scale factor.

Convert each GADGET stored velocity to peculiar km/s with `√a` and the
header velocity unit. Subtract the mass-weighted mean velocity and take
the mass-weighted RMS of the residual speeds. The curve for the earlier
start subtracts the result from the later start at the same output
redshift. The reported contrast is conditional on this finite particle
load. A value outside the comparison tolerance leaves convergence open.


### Sampled imaging and time-delay posterior

The synthetic control uses a singular isothermal sphere to ray-trace
a Sérsic source and a point source through a Gaussian PSF. It adds
seeded noise to 32 × 32 image pixels, astrometric positions and the
relative arrival time. The estimator reads those observations from
a separate file. The lens family and source shape are fixed for this
small test. The Einstein radius and the time-delay distance are free.

For an angular Fermat-potential difference in radians squared,
`Δt = D_dt Δφ / c`, where `D_dt = (1 + z_l) D_l D_s / D_ls`
and the distances are angular-diameter distances. The fit samples
`D_dt` in Mpc using both imaging and relative-delay likelihoods.
For the declared fixed dimensionless expansion shape, `H₀ D_dt`
is constant, so each sampled distance maps to one H₀ in km/s/Mpc.
The plots show the empirical cumulative probability of each set
of retained draws. Their widths include the uncertainty represented
by this fit. Mass-profile freedom, independent coverage and convergence
require their own controls.


### Primordial cloud support

The cloud producer requires native non-equilibrium chemistry, H₂ cooling,
HD cooling and measured single-phase temperature. It compares each
particle's thermal Jeans length against its saved smoothing length and
its Jeans mass against the kernel mass. Only supported particles above
the density threshold enter the spatial neighbor graph. For each
connected group, the mass-weighted position covariance gives the axes
through the square roots of its eigenvalues.

The SH03 negative control reads a saved native snapshot with the
primordial network disabled. The exporter retains an empty geometry and
the support status. The initial Grackle network-3 run halted at the nucleus conservation
check. After correcting the normalization handoff, the rerun evolved
64 gas particles across three saved epochs. The final plot uses their
saved Cartesian coordinates and one mass-weighted centroid. The gas
box verifies chemistry, measurement and plotting. Cloud collapse and
fragmentation require their own controls.


### Cooling and stellar birth timeline

For each exact gas ID present at adjacent saved epochs, take the change
in cumulative native radiated energy `ΔE_rad` over the measured time
interval `Δt`. For positive loss, the estimator computes
`t_cool = m u Δt / ΔE_rad` from the saved gas mass and specific internal
energy, and compares it to `t_ff = √[3π/(32Gρ)]`. Jeans support separates
resolved and unresolved cooling candidates. The x-coordinate is the end
of the saved interval in Myr. The numerical sidecar retains both ends.

The gas-only control measures ratios of roughly 782–1,010. A unit ratio
cut yields zero fast-cooling candidates. The atlas uses a deliberately
broad cut of 10⁶ to exercise the cooling lane. There are 64 exact gas
IDs at each of two intervals, so the 128 event markers overlap at their
two saved end times. The birth lane stays empty because the native
birth-journal census is empty. A positive birth entry requires the
recorded SSP application time, exact native gas parent, unique bound
host, and a supported host assembly interval. A positive cooling-to-birth
join additionally requires a supported cooling interval for that parent.


### SIDM event journals and fixed-cohort rates

For each native event, convert the two saved velocities and masses
using the event datasets' unit attributes. The scattering angle is the
angle between the pre-kick and post-kick relative velocity vectors.
An elastic kick conserves the vector momentum `Σ m_i v_i` and the
kinetic energy `Σ m_i |v_i|² / 2`. Check both directly on the event
row, where the pair is isolated from changes between saved snapshots.

For the fixed cohort, count each participating exact ID in the source
journal and reconcile it against that particle's unsigned snapshot
counter increment over the same interval. A single pair event gives
two participations. Divide each counter increment by elapsed Gyr and
average with initial-mass weights. The integrated optical-depth curve
divides each exposure increment by the same elapsed interval. The
homogeneous control verifies these diagnostic calculations. The
registered halo figure additionally requires a completed source-bound
halo history and per-step operator admission.


### Saved native worldlines and encounter geometry

The 32-particle isolated control selects the closest pair in the first
snapshot. The extractor follows those two exact integer IDs through all
four saved outputs. The frame adapter subtracts the declared origin
and uses the native elapsed-time conversion. The 3D figures show the
saved coordinates in proper kpc. Each point touches a recorded interval
and the display draws no interpolated segments.

The encounter search uses cubic Hermite trajectories with the saved
positions and velocities as endpoint constraints. It minimizes squared
separation on each eligible time bracket and records relative speed in
km/s at the candidate time. Finite observation boundaries can yield
candidate minima. The impact plot requires an independent interpolation
certificate before displaying a supported point. The control retains
both candidates and their refinement requests in the original data.


The separate harmonic control exercises the supported impact branch.
For each coordinate, the independent reference is
`x(t) = x₀ cos(ωt) + (v₀/ω) sin(ωt)`, in native length and time units.
Over a saved interval of width `h`, cubic Hermite truncation is bounded
by `h⁴ max|x⁽⁴⁾|/384`. For the oscillator, the fourth-derivative
maximum is bounded by `ω⁴ √[x₀² + (v₀/ω)²]`. Add the largest endpoint
position error and `h/4` times the largest endpoint velocity error
before combining the Cartesian coordinate bounds. The two particles'
bounds add to the reported separation error. This certificate applies
to the prescribed oscillator and the retained cadence.


### Native bound perturber mass profiles

GADGET writes FOF/SUBFIND catalogues and a separate descendant table.
Read the descendant row before selecting the next observation. The
SUBFIND offset and length select exact bound particle IDs from each
group-ordered snapshot. The control verifies that the catalogue center
matches the native most-bound particle and uses that particle's saved
integer image counter for the center's temporal frame.

Within each epoch, take the local proper displacement from the center
with a minimum-image correction. Sort the distances and compute
`M(≤R) = Σ m_i` over bound members with distance at most `R`.
The x-axis uses proper kpc and the y-axis uses solar masses. The
outermost sample recovers the full 11-member mass in each saved
epoch. The controlled clustered ICs test membership, ancestry,
and rendering. The profile is conditional on the tiny particle census.


### Native orbital assembly timeline

Select exact object and reference-halo observations at the same saved
epochs. Use descendant records to establish continuity and saved image
counters to convert the positions to one proper Cartesian frame. The
position derivative is peculiar velocity plus `H(a) r`, with consistent
distance units. An interval enters the search only when its endpoint
observations pass the selected quality gate and the saved cadence is
supported.

Pericenter and apocenter candidates are minima and maxima of squared
relative separation along the Hermite interpolants. Infall candidates
cross the declared reference radius inward. A merger marker records a
separate native progenitor joining the selected descendant at a saved
epoch. The actual coalescence time is unresolved by that record.

The clustered smoke control uses two central SUBFIND objects and the
second object's bound half-mass radius as the reference boundary.
The two supported intervals contain no recorded event in the selected
pair. This verifies the empty timeline through native graph, clock,
image and figure intake. A positive merger or infall control requires
a different trajectory.
