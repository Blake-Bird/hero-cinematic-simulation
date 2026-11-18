# Calculations and numerical conventions

This guide explains the implemented measurement families and their numerical
conventions. The [full-run walkthrough](architecture.md) connects them to the simulation
and export stages. The [figure reference](figure-contracts.md) lists the measurements
required by each registered graph.

Native models and specialized campaigns also retain their own parameters, references,
controls, and data contracts. Use the recorded model and estimator for a particular run;
these equations do not establish its calibration or convergence.

- [Epochs, coordinates, and IDs](#epochs-coordinates-and-ids)
- [Halo abundances and reference fits](#halo-abundances-and-reference-fits)
- [Power, correlation, and bias](#matter-power-correlation-and-bias)
- [Halo structure and histories](#halo-structure-membership-and-histories)
- [Gravity and gas evolution](#gravity-and-gas-evolution)
- [Cooling, star formation, and returns](#cooling-star-formation-and-stellar-return)
- [Black holes](#black-hole-accretion-masses-and-feedback)
- [Stellar populations and light](#stellar-populations-and-synthetic-light)
- [Maps and movies](#maps-field-morphology-and-movies)
- [Specialized observations and inference](#specialized-observations-inference-and-comparisons)
- [Execution evidence and limits](#execution-controls-versus-scientific-validation)

For worked calculations from original arrays, open the
[optional notebook](../optional_tests/reference_gallery/Walkthrough.ipynb) using the
[notebook instructions](optional-tests.md). The [learning guide](learning-guide.md)
connects those examples to their producers. This page supplies conventions to use
when interpreting an actual run, rather than a second set of solver settings.

## Epochs, coordinates and IDs

For cosmological snapshots, `a = 1/(1+z)`. Requested and saved physical epochs are
recorded separately. Cosmic time comes from the run's saved cosmological clock; missing
clock information is not replaced by zero or a different cosmology. Physical lengths are
`a` times comoving lengths. Quantities expressed per `h` retain that convention until an
explicit unit conversion. Particle IDs remain integers throughout selection,
trajectories and joins, including IDs above 2^53.

`core/data.py` owns dataset conventions and conversions; `core/cosmology.py` owns
clock/background helpers. GADGET's stored velocity conversion includes the snapshot
scale factor; use the reader's declared conversion rather than treating raw stored
vectors as physical km/s. Source fields can carry different a/h powers. Verification
references: `SavedUnitTests`, `SnapshotScheduleTests`, `DataContractTests` and
`ExactOutputTimelineTests` in `verification.py`.

## Halo abundances and reference fits

For counts N in mass bin width Δlog10(M) and comoving volume V, `dn/dlog10(M) = N/(V
Δlog10(M))`; `dn/dM = N/(V ΔM)` uses linear bin widths. Cumulative abundance sums counts
above the selected threshold within the recorded finite mass window. Mass definitions
remain distinct: FOF, M200c, M200b and virial masses are not interchanged by plot
labeling. Minimum halo particle cuts set a measurement support floor, not a convergence
guarantee.

HMF uncertainty includes central Garwood Poisson intervals and, where supported,
periodic spatial delete-one jackknife measurements. Their availability and finite-box
limits are exported; absent uncertainty is not zero uncertainty. Largest halo mass and
rare-halo abundance histories use saved physical epochs.

Reference predictions integrate the saved linear spectrum using the spherical top-hat
window to obtain `σ²(R) = ∫ k²P(k)W²(kR) dk/(2π²)` with `M = 4πρ̄R³/3`. Multiplicity
predictions use `dn/dln(M) = (ρ̄/M) f(σ) dln(σ⁻¹)/dln(M)`.

The logarithm convention matters: `dn/dlog10(M) = ln(10) * dn/dln(M)`.
For `k` in h/Mpc and `P(k)` in (Mpc/h)³, use `R` in Mpc/h, `M` in
Msun/h and the mean density in (Msun/h)/(Mpc/h)³. Then `kR`, `σ²` and
`f(σ)` are dimensionless, while abundance per logarithmic mass interval has
units (Mpc/h)⁻³. The top-hat window is
`W(q) = 3*(sin(q)-q*cos(q))/q³`, with limit `W(0)=1`;
`core/cosmology.py` handles the small-q limit and mass-to-radius conversion.

`core/hmf_fitting_functions.py:binned_predictions` averages each prediction
over the actual log10 mass bin using 32 Simpson panels. A curve evaluated only
at a bin centre is a different approximation. The measured histogram's
`core/analysis.py:hmf_arrays` instead divides its integer count by the actual
volume and bin width. For example, two halos in a one-dex bin in a
10 Mpc/h box give `2/(10³*1) = 0.002 (Mpc/h)⁻³`.

`core/hmf_fitting_functions.py` implements PS, ST, Reed, Watson FOF, Jenkins and Warren
coefficients and calibration masks. It requires compatible FOF b=0.2 inputs, checks
spectrum coverage, and records extrapolations. It does not supply an SO conversion, a
massive-neutrino fit or a model-spread uncertainty. Verification references:
`HMFFittingFunctionTests`, `FinderComparisonTests`, `AnalysisContractTests`.

## Matter power, correlation and bias

`δ = ρ/ρ̄ − 1`; the spectrum and correlation describe Fourier and real-space two-point
density statistics. The maintained native-power route consumes GADGET mass-weighted
CIC-deconvolved folded spectra. Saved output reports its fold/mesh selection, shot-noise
subtraction and exact mode counts. Its diagonal Gaussian approximation is `σ(Psignal) =
sqrt(2/Nmodes)(Psignal + Pshot)`; this excludes nonlinear covariance and numerical
resolution error. Species auto-spectra are not additive because total power also
includes cross terms.

Correlation and bias use their configured saved-data estimators, selections and
geometry. Do not mix halo and matter normalization or independently selected k/r ranges.
See `core/power.py` and `core/analysis.py`, plus each exported HDF5 estimator contract.
Verification references: `SavedUnitTests`, `AnalysisDatasetSemanticsTests` and
batch/suite comparison tests. Matched phases and convergence require separate evidence,
beyond an overlay.

## Halo structure, membership and histories

Density profiles divide shell mass by shell volume; velocity dispersions use supported
member velocities and the documented centering convention. Concentration, axis ratios
and radial shapes require valid native positions, radii, memberships and explicit
fit/sampling choices. Shape uncertainties come from the requested resampling procedure
and support, not a cosmetic band. Inspect the spin field's definition: specific
angular momentum has units of length times velocity. The enclosed
`bullock_spin_enclosed` measurement instead uses `|J|/(sqrt(2)*M*V_c*R)` and
is dimensionless; it is available only where its enclosed mass and radius support
the calculation. These two quantities are not interchangeable.

Histories follow measured descendant links; trajectories and stellar streams join exact
IDs across epochs. Periodic displacement uses the declared wrapping convention. Missing
links/epochs are retained as gaps. Formation-time, merger, void-boundary and collision
diagnostics require their stated temporal support; proximity or overlap alone does not
prove physical identity. Sources: `core/halos.py`, `core/analysis.py`,
`core/stellar_stream_movie.py`, `core/void_boundary_history.py`. Verification references
include `HostTrackingTests`, `HaloStructureTests`, radial-shape groups and corresponding
tests in `tests/`.

## Gravity and gas evolution

For a cosmological periodic box, use comoving position x, scale factor a and physical
peculiar velocity v. The continuum interpretation of the gravity solver is

```text
x_dot = v/a
v_dot + H(a) v = -gradient_x(Phi)/a
laplacian_x(Phi) = 4*pi*G*a^2*(rho_proper - mean_rho_proper)
```

GADGET implements its configured Tree/PM split, softening, time bins and integration
variables; the equations above are not an instruction to replace that native integrator
with a Python Euler loop. `core/evolution.py` maps explicit tolerances, mesh and
softening values into the build and parameter file. IC displacements and velocities are
prepared by the selected IC route from the bound species transfers, seed and starting
epoch. The native gravity, SPH and integration source lives under
`core/software/gadget4/src/`.

For a single-phase ideal-gas interpretation, `P = (gamma-1) rho u` and `T = (gamma-1) mu
m_p u/k_B`, with physical specific internal energy u in cm²/s². The reader must obtain
mu from the declared composition/ionization or native thermodynamic state and convert
u's stored units first. This formula does not turn a multiphase effective EOS into a
resolved single-temperature measurement. `core/data.py`, `core/analysis.py`,
`core/stars.py` and `core/stellar_population.py` retain the applicable field,
composition, epoch and availability conventions.

## Cooling, star formation and stellar return

The radiative source step uses heating/cooling functions and the actual gas state. In
the maintained primordial solver, `cooling_sfr/cooling.cc:DoCooling` solves the implicit
source equation

```text
u_new - u_old - (n_H^2/rho) * Lambda_net(u_new, rho, electron_state, UV) * dt = 0
```

Here Lambda_net is the code's signed net source function in its converted cgs
convention: positive values heat and negative values cool. With `n_H` in cm⁻³,
`rho` in g/cm³, `Lambda_net` in erg cm³/s and `dt` in seconds, the source
increment has units erg/g, the same as specific internal energy `u`.
The native routine validates its inputs and solves for the new energy; it is
not a post-processing correction to saved temperatures. Hydrodynamic
compression/expansion and gravity are separate integrator contributions. The optional
Grackle branch calls the selected compatible library/network and model table through
`research_grackle.cc`; it must not silently substitute a different cooling model.
Invalid/nonfinite gas states are unavailable measurements or solver failures, not
zero-temperature gas.

For the selected Springel–Hernquist model, `cooling_sfr/sfr_eos.cc` computes the
physical-density threshold, hot/cold components and cold-cloud fraction x. The
corresponding gas consumption time and net star-formation rate are

```text
t_star(rho) = MaxSfrTimescale * sqrt(rho_threshold/rho_proper)
M_cloud = x * M_gas
SFR = (1-beta) * M_cloud/t_star, where beta = FactorSN
```

The routine bounds the working time by its source interval where implemented, requires
the configured eligibility/overdensity conditions and converts the reported rate to
Msun/year. The native model also calculates evaporation, effective energy and
relaxation; these are explicit source-model choices, not values fitted to the final
graph.

`cooling_sfr/starformation.cc` converts eligible gas stochastically. For the current
whole-carrier conversion, the probability is derived from

```text
hazard = SFR_native * dt_native / M_gas_native
probability = 1 - exp(-hazard)
```

The code evaluates this with `-expm1(-hazard)` and uses a particle ID, declared stellar
seed and opportunity counter for its random draws. Birth state records the parent
identity, native birth time, gross initial mass, locked mass and birth composition. A
star particle is a coeval population, so its present mass, initial formed mass and
number of represented physical stars are different quantities.

The optional delayed-return operator reads a pinned age/metallicity table in
`research_stellar_returns.cc`. Schematically a source interval applies `Delta_m =
M_initial * [R(age_2,Z)-R(age_1,Z)]`, where R denotes the table's cumulative delivered
fraction under its recorded IMF, yields, lifetimes and prompt-return subtraction.
Element and energy increments use the corresponding cumulative tables. The actual code
retains source counters and mass/element/energy bookkeeping; adding an independent
prompt return again would double count it. `core/stellar_returns.py` reconciles these
saved native increments.

Gas `StarFormationRate` fields give the whole-snapshot SFR after unit conversion;
`core/analysis.py:build_star_formation_history` assembles only supported rates at their
actual redshifts. Birth cohorts and stellar ages use the run's cosmological clock or its
declared isolated physical clock, rather than the snapshot number.

## Black-hole accretion, masses and feedback

The native accretion/source implementation is
`core/software/gadget4/src/black_holes/black_holes.cc`. Its Python unit-explicit
reference is `core/black_holes.py:accretion_rates`. With subgrid BH mass M, measured gas
density rho, sound speed c_s, relative speed v_rel and declared boost alpha, the base
rates are

```text
Bondi inflow = 4*pi*alpha*G^2*M^2*rho / (c_s^2 + v_rel^2)^(3/2)
Eddington cap = f_Edd * 4*pi*G*M*m_p / (epsilon_r*sigma_T*c)
inflow = min(Bondi inflow * rotational_suppression, Eddington cap)
BH growth = (1-epsilon_r) * inflow
model luminosity = epsilon_r * inflow * c^2
coupled feedback power = epsilon_f * epsilon_r * inflow * c^2
```

In these equations use a single coherent unit system: cgs mass in g, density
in g/cm³, velocities in cm/s and `G` in cm³/(g s²) give inflow in g/s and
luminosity/power in erg/s. The Python reference accepts mass in Msun, converts
it internally and returns explicitly named Msun/year and erg/s fields.
`alpha`, `f_Edd`, `epsilon_r`, `epsilon_f` and rotational suppression are
dimensionless; their selected values belong to the saved model configuration.

Default rotational suppression is one. An explicitly selected angular-momentum model
applies its declared C_visc and measured support; thermal and optional kinetic feedback
have distinct native operators. The actual source step further limits changes by
available donors, the selected source fraction and subcycling. It records applied
inflow; the analytic Bondi estimate is not permission to remove absent gas.

Seeding rules and reservoir assembly define the relationship between gravity carrier
mass and subgrid BH mass. A conversion does not imply that all carrier mass is already
the physical BH. Source accounting includes consumed gas, retained BH mass, radiated
mass-equivalent and coupled/escaping energy. Optional binary/dynamics settings must be
read separately; the ordinary interactive thermal profile does not enable mergers.

`core/black_holes.py:analyze_black_holes` reads complete type-5 snapshot fields, keeps
each exact BH identity, converts rates/masses/energies and writes the history and
panels. In particular `BH_InflowRate` is a source-step mean, while `BH_Mass` is the
saved-epoch subgrid mass. Its diagnostic inflow/cap ratio compares that mean with a cap
evaluated at the saved mass; it is not automatically an instantaneous source-step
Eddington ratio. Model luminosity is not a resolved accretion-disk SED, observed flux or
emitted-light image.

## Stellar populations and synthetic light

`core/stellar_population.py:stellar_sps_truth` derives ages using native birth times and
the bound physical clock. `pinned_fsps` checks the selected FSPS wrapper, IMF, filter,
isochrone, spectral and other physical assets before synthesis. For the chosen intrinsic
SSP libraries the normalization is one initially formed solar mass, so the schematic
population sum is

```text
L_filter = sum_over_populations(M_initial * l_filter(age, metallicity; model))
```

Use gross initially formed mass, not the surviving current particle mass, for this
normalization. Native locked/current mass and FSPS surviving mass stay separate. Missing
age/metallicity coverage stays unavailable; it is not clipped into the model domain.
Dust, observer geometry and instrument effects belong to their explicit observation
requests. Basic stellar-density maps do not require FSPS, and installing FSPS does not
cause it to run on every snapshot.

Verification entry points for these implementations include
`NativeCoolingTests`, `NativeSSPBirthTests`, native SSP journal groups,
`BlackHoleReferenceTests` and `BlackHoleAnalysisTests`. See the
[smoke report](smoke-test.md) for executed suites, native controls and their source
compatibility. A reference-test pass and a native source-budget reconciliation
provide different evidence; check the recorded scope of each result.

## Maps, field morphology and movies

Surface-density maps deposit selected mass onto a configured projection grid; slab maps
also apply the explicit line-of-sight selection. Temperature maps use the declared
mass-weighted temperature estimator. Smoothing, pixels, periodic geometry, percentile
normalization and global/per-epoch scales are recorded. Hessian morphology is a
density-curvature diagnostic. DisPerSE topology is a separate selected native method and
is not implied by a filament-like image.

For a mass map, `Sigma_pixel = deposited_mass / pixel_area`; integrating
`sum(Sigma_pixel * pixel_area)` recovers the selected deposited mass, including
the producer's masks and assignment convention. A volume-density grid instead
divides by cell volume. Keep proper/comoving area or volume consistent with the
coordinate and unit record. Display smoothing and colour scaling do not change
which physical estimator the original arrays represent.

For the periodic peculiar-potential solver, `∇x²Φ = 4πG a²ρ̄proper δ`; the homogeneous
mode is removed. Proper acceleration is `−∇xΦ/a`; the proper Hessian is `∂xi∂xjΦ/a²`.
The source has an explicit Gaussian filter, with no automatic CIC deconvolution. The
implementation checks its Poisson trace identity and rejects unsupported
grids/allocations.

Movies retain their own selected saved-epoch coverage, source records and
interpolation/display assumptions. Equal frame spacing does not imply equal physical
time. Star-free epochs may be explicitly omitted; missing maps for present species
remain failures. Highlight selection does not thin any timeline. Sources:
`core/visualization.py`, `core/visual_pipeline.py`,
`core/analysis.py:solve_periodic_peculiar_potential`. Verification references:
`VisualWorkflowRegressionTests`, `AllOutputConnectionTests`, movie tests.

## Specialized observations, inference, and comparisons

Lensing, sky maps, stellar observations, SIDM inference, observer errors and stream
response experiments require their authored geometry, physical arrays, likelihoods,
priors or sample inputs. Their specific input, unit, uncertainty and source contracts
appear in [the 102-graph catalogue](figure-contracts.md). No generic box-simulation
`all` creates those independent experiments.

Every figure's numeric arrays and metadata are the reference for the values actually
plotted. PNG/SVG/PDF are presentations; axis labels alone do not verify units. Reproduce
numerical acceptance with the named test/campaign and exact saved inputs, using the
figure together with its numerical data and source record.

## Execution controls versus scientific validation

The [retained smoke campaign](smoke-test.md) exercises selected numerical calculations
and native/visual paths. Its tiny particle populations and short sampler/training
runs cannot establish convergence, covariance accuracy, empirical-fit calibration,
or a preferred cosmological model. Review the named controls and measured inputs
for the particular equation or estimator before using an exported figure as evidence.
