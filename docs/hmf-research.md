# Three workflows: predict, select simulations, learn from GADGET

The main research path is **full GADGET simulations → halo catalogues → measured
halo counts → a newly learned fitting function**. CLASS/AxiCLASS and published HMF
formulas are a cheap screening stage for choosing simulations. Their predicted
abundances are not training targets or default training inputs.

| Task | Input | Result |
|---|---|---|
| [Predict reference abundances](#1-run-cosmology-and-hmf-formulas-without-gadget) | Cosmology, epochs, mass bins and requested formulas | Linear spectra and formula predictions |
| [Choose simulations](#2-find-cosmologies-where-formulas-disagree-unusually-strongly) | Parameter design and formula-eligibility masks | Disagreement diagnostics and candidate configurations |
| [Learn from measured halos](#3-train-a-new-fitting-function-using-full-gadget-measurements) | Completed full-box GADGET catalogues and their metadata | Measured counts, held-out splits and a learned equation |

Use new output directories and an activated Research2 environment. The 512³/1024³
examples below describe production research inputs, not the small smoke tests.
For a fast execution walkthrough, start with the
[32-particle optional lab](optional-tests.md#choose-a-profile); it does not establish
HMF resolution or precision.

## 1. Run cosmology and HMF formulas without GADGET

```bash
python commands.py class --config /absolute/path/linear.json --output /absolute/path/linear
python commands.py hmf --spec configs/hmf_single.example.json --output /absolute/path/hmf
```

`class` is the existing standalone cosmology command. `hmf` runs only
CLASS/AxiCLASS and the requested reference fitting functions at your masses and
redshifts. It saves spectra, tables and plots. No particles or GADGET run is needed.
`hmf` uses nested physical densities `omega_i=Omega_i*h²`; the existing `class`
JSON uses density fractions `Omega_i`. Do not interchange them.

To reuse exact epochs from an existing standalone CLASS bundle:

```bash
python commands.py hmf --spec /absolute/path/hmf.json --linear-bundle /absolute/path/linear/name --output /absolute/path/hmf-reused
```

The cosmology must match the verified bundle. Only saved epochs can be reused;
other redshifts require fresh spectra. For EDE, set `config.cosmology.model=ede`
and provide `config.ede`, for example:

```json
{"parameterization":"axion","f_ede":0.08,"z_c":3500,"theta_i":2.8,"n_axion":3}
```

## 2. Find cosmologies where formulas disagree unusually strongly

```bash
python commands.py hmf-scan --spec configs/hmf_scan.example.json --output /absolute/path/scan --workers 4
```

Set parameter intervals in `sampling.bounds`, redshifts in `redshifts` and mass
windows in `mass_edges_msun_h`. Choose `latin_hypercube` or `uniform` sampling
and a reproducible seed. Each process has a private solver directory and uses
one OpenMP thread. `timeout_seconds` bounds each solver invocation. Failed
samples retain logs; a partial campaign exits nonzero and reports its failures.
These samples explore the intervals you declare; they are not a posterior.

For exact parameter values on a Cartesian grid, replace `sampling` with:

```json
{
  "method":"grid",
  "grid":{"cosmology.H0":[62,66,70,74],"cosmology.n_s":[0.94,0.96,0.98]},
  "max_samples":10000
}
```

This runs all 12 combinations. Do not provide `count` or `bounds` for a grid.
EDE parameters can be sampled as `ede.f_ede`, `ede.z_c`, `ede.theta_i` and
`ede.n_axion` when the model is EDE.

At each fixed mass bin and redshift, compute the reference-formula spread
`S=log10(max(predicted abundance)/min(predicted abundance))`. Compare S to other
sampled cosmologies with **exactly the same set of eligible formulas** there.
The default candidate threshold is:

```text
S > median(S) + max(3 * 1.4826 * MAD(S), 0.05 dex)
```

At least five eligible cosmologies are required. This flags unusually large
formula disagreement. A cosmology with more halos under every formula is not
selected merely because its abundance is large. A uniformly large disagreement
throughout the scan is also not unusual under this rule. The MAD rule is a
selection heuristic, not a statistical significance or confidence level.

Change `selection.minimum_cosmologies`, `mad_multiplier`, and
`minimum_excess_dex` if needed. `selection.criterion=absolute` selects formula
ratios exceeding `flag_ratio`; `either` accepts either criterion. `unusual` is
the default. Abundance-versus-ensemble scores remain separate diagnostics.

Outputs:

- `predictions.csv`: every formula, cosmology, epoch and mass bin, including masks.
- `scores.csv`: formula spreads, peer baselines, thresholds and flags.
- `gadget_candidates.json`: one entry per selected cosmology, with triggering bins.
- `candidates/cosmo_*/cosmology.json`: its exact physical configuration.
- `candidates/cosmo_*/linear-config.json`: input for the existing `class` command,
  when that command can represent all its physical parameters exactly.
- `hmf_z*.png`, solver directories and `report.json`: plots, provenance and failures.

Run a selected cosmology through the normal full GADGET workflow, for example:

```bash
python commands.py class --config /absolute/path/scan/candidates/cosmo_00010/linear-config.json --output /absolute/path/selected-linear
python commands.py gadget --cosmology /absolute/path/selected-linear/cosmo_00010
```

Choose the box, seed, **512 or 1024 particles per dimension** (512³ or 1024³),
accuracy, full output schedule and halo measurements in the normal GADGET setup.
The selection record preserves the mass/redshift windows to investigate.
`selection.gadget_start_redshift` controls the handoff's initial epoch; it must
precede the measurement epochs. Generic solver extras or changed radiation
settings are retained in `cosmology.json`; the handoff marks when the narrower
standalone `class` interface cannot reproduce them. Never silently drop them.
The scan does not launch a large particle run automatically.

Only k-covered bins wholly within each formula's listed mass/σ/redshift ranges
participate. Those ranges come from the existing HMF implementation: Jenkins
stops at z=5, Warren is z=0, and Reed/Watson stop at z=30. PS/ST have no listed
coordinate limit here; that is not validation at arbitrary redshift. All current
screening formulas use FOF b=0.2, with no SO mass conversion. EDE predictions
are marked extrapolations even inside the coordinate masks. See the
[HMF calibration-mask convention](https://hmf.readthedocs.io/en/v3.7.2/_autosummary/fitting_functions/hmf.mass_function.fitting_functions.FittingFunction.html).
A candidate identifies a simulation worth investigating, not the correct formula.

## 3. Train a new fitting function using full GADGET measurements

Run the full particle simulations and halo finder first. Then export directly
from their saved catalogues:

```bash
python commands.py hmf-learning export-gadget --run /absolute/path/GADGET_512 --output /absolute/path/measurements/run512
python commands.py hmf-learning export-gadget --run /absolute/path/GADGET_1024 --output /absolute/path/measurements/run1024
```

Or collect many simulations recursively in one command:

```bash
python commands.py hmf-learning export-gadget --runs-root /absolute/path/full-gadget-runs --output /absolute/path/measurements
python commands.py hmf-learning build --spec /absolute/path/measurements/dataset-request.json --output /absolute/path/dataset
python commands.py hmf-learning train --dataset /absolute/path/dataset --output /absolute/path/new-fit --degree 2
```

Here “HMF data” means **counting actual simulated halos in mass bins**. It does
not mean values generated by an existing HMF fitting function. A target row is
`(cosmology, redshift, mass window, measured halo count, simulation volume)`.
The exporter reads native GADGET FoF/SUBFIND host masses and particle counts,
applies your particle cut and bins the measured masses. Empty bins remain zero.
It checks snapshot/catalogue epoch agreement, complete catalogue shard totals,
box size and the full uniform DMO particle load declared in the snapshot header.
It does not read all particle positions into memory. It hashes the measured
catalogues and retains the relevant snapshot headers and file sizes.

The current direct exporter is for full periodic uniform DMO runs, not selected
zoom volumes or hydro. It retains **cosmology, seed, box, particles per dimension,
particle mass, redshift, mass definition and resolution cut**. `--snapshots`
selects saved snapshot numbers. A batch collection reports failed runs explicitly;
review them before using the successful subset's dataset request.

Keep raw snapshots/catalogues in HDF5. Use the exported **CSV counts plus JSON
metadata** for ML. Keep FoF b=0.2, SO 200c, 200m (`m200b`) and virial datasets
separate. The halo finder and mass definition are separate choices. This direct
exporter uses native GADGET Group fields; running Rockstar as well does not
change their origin. External Rockstar measurements may still use the legacy
portable `metadata.json` + `hmf.csv` importer with a declared finder and host-only,
complete-volume counts. Mixed finders/physics/mass definitions are rejected.

For several resolutions of the same cosmology, seed, box, epoch and mass bin,
the dataset writes `resolution_comparison.csv` comparing counts to the finest
available resolution. By default `resolution_policy=finest_per_bin` trains on
the finest of those matched measurements, so a 512³/1024³ pair is not counted
as two independent realizations. Different seeds and volumes remain separate.
This does not establish convergence; inspect the comparison and run additional
resolution/volume checks. `retain_all` is an explicit alternative.

The default split holds out entire cosmologies (at least five for an automatic
approximately 60/20/20 split). All their seeds, snapshots and resolutions stay
together. You can instead provide train/validation/test folders:

```json
{
  "source_roots":[
    {"path":"/absolute/path/measured-training","split":"train"},
    {"path":"/absolute/path/measured-validation","split":"validation"},
    {"path":"/absolute/path/measured-testing","split":"test"}
  ],
  "resolution_policy":"finest_per_bin"
}
```

The existing supervised trainer learns an explicit positive empirical equation
for abundance directly from these GADGET counts, using a Poisson likelihood
with exposure `volume*log10(M_upper/M_lower)`. Degree 1 or 2 chooses a polynomial
basis for log abundance. This is a newly fitted empirical function, not a
recalibration of Sheth–Tormen, Watson, etc. `model.json` records its coefficients,
features, domain and validation/test results; `evaluation.csv` compares predicted
counts to the simulation counts. It is a starting model, not automatic symbolic
discovery or a demonstrated precision fit.

Inputs include mass, redshift, cosmological densities, primordial spectrum and
EDE parameters. Optional `linear_features.enabled=true` in the dataset request
adds physical σ(M,z), variance/effective slope, H/H₀, D(z)/D(0) and growth rate
from epoch-specific CLASS solves. Those are physical descriptors; the target
still comes exclusively from GADGET. Existing formula-abundance columns are
excluded from default training inputs. Additional independently computed physical
`feature_*` columns require explicit `additional_physical_features` selection.
Compare reference formulas with the learned fit separately on held-out GADGET
measurements. Do not use measured target-derived quantities as inputs.

Training does not imply accuracy at every redshift or in every EDE cosmology.
Assess mass/redshift/cosmology residuals, resolution and volume convergence,
cosmic covariance and independent tests before making precision claims. The
current Poisson baseline omits full covariance and uses bin-center features
for bin-average rates. Prediction rejects inputs outside training feature bounds
unless explicitly allowed. A rectangular feature bound cannot prove joint
parameter-space coverage. Nonlinear EDE accuracy also depends on the particle
model; AxiCLASS initial conditions alone do not establish it.

## Checks

```bash
python -m unittest discover -s tests -p test_hmf_research.py -v
python verification.py HMFFittingFunctionTests ModularCosmologyTests CosmologyProductTests -v
```

Tests include grid coverage, usual-versus-unusual disagreement, GADGET handoff,
native HDF5 host-count export, zero bins, incomplete catalogues, resolution
pair handling, cosmology leakage, target provenance, learned-equation recovery
and prediction bounds. Tiny/synthetic fixtures establish implementation behavior;
they do not stand in for full 512³ or 1024³ simulation validation.
