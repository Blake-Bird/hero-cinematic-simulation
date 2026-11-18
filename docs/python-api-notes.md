# Reviewed Python API explanations

These notes supplement missing source docstrings in the generated Python reference.
They describe inspected implementations, inputs, units, outputs and effects.
The generator checks each hidden implementation binding before using its prose;
review the note when that definition changes. Runtime scientific code does not
read this file. Descriptions are documentation, not execution or acceptance evidence.

## core/action_angle.py::SphericalFieldAdapter

<!-- implementation-sha256: 68b13679cb9503c18b381225eda7cc6955cb030af93c383a10caa926fc1341ba -->

Wrap a calibrated field as a stationary spherical potential, with radii in
proper kpc and an explicit escape-energy boundary. Radial queries evaluate the
field on the positive x axis; Cartesian queries preserve the supplied positions.
Both reject radii outside the declared domain. The adapter returns the wrapped
field's potential/force/Hessian values and writes no files; declaring spherical
symmetry here does not independently establish that the fitted field is spherical.

## core/action_angle.py::classify_potential_domain

<!-- implementation-sha256: 6badf3ac4555538cda0ba0adc3ae6f0cc7cc33fd0fdaba328d6de4b86eac0817 -->

Validate the declared symmetry and Boolean time-dependence flag, then return
the action-method decision and its retained symmetry evidence. Spherical fields
select radial quadrature; stationary axisymmetric fields select AGAMA's
Staeckel-fudge route; general/triaxial and time-dependent axisymmetric fields
select orbit diagnostics only. Exact action invariants are claimed only for a
stationary spherical declaration. This chooses a method; it performs no force,
orbit or action calculation and writes no files.

## core/action_angle.py::spherical_actions_angles

<!-- implementation-sha256: a2aeb878c6946227853cbfe2944e82a2a4bf686c9cb1bb76ce86588805ca42a6 -->

Take aligned N-by-3 positions in proper kpc and velocities in km/s, an explicit
spherical potential and turning-point/quadrature controls. For each supported
bound state, bracket radial turning points and integrate the radial action and
orbital time/azimuth; return `(report, arrays)` with ordered Jr/Jz/Jphi actions
in kpc km/s, angles in radians, frequencies in km/s/kpc, turning radii, Jr error
estimates and separate action/angle masks. Unbound, radial-degenerate or failed
quadrature rows retain NaNs and findings. Analytically recognized circular
states can retain valid actions while radial phase remains undefined. No files
or external solver processes are created.

## core/action_angle.py::agama_adapter

<!-- implementation-sha256: ec51fdbed5d1e534c6ef9664305e241608614f1f6bbe3f9758c0a2f2557344d9 -->

Use the verified pinned AGAMA backend on N-by-6 Cartesian phase-space points
assembled from the sample positions/velocities and an explicitly supplied AGAMA
potential. Require a spherical or axisymmetric declaration; optionally pass a
positive focal distance in proper kpc. Return backend/domain provenance and
N-by-3 actions, angles and frequencies in the shared kpc km/s, radian and
km/s/kpc convention, with finite-value masks. It loads and executes the native
Python backend in the current process; it does not write an action product or
certify the declared symmetry or approximation accuracy.

## core/action_angle.py::action_history

<!-- implementation-sha256: 06d55a851d5450330702577c6e848d8ca569984a481f9fea45c89ec922da3d84 -->

Integrate the initial Cartesian sample over at least 64 uniformly spaced Myr
epochs, then measure actions at each epoch when the declared domain permits
an action estimator. Return `(report, arrays)` containing orbit states, force
evaluation counts, first/second-half Cartesian spectral peaks and their finite
duration floor; supported action histories additionally retain action drift,
angle-rate/cadence masks and differences from a second integration with stricter
tolerances and maximum step. Time-dependent spherical measurements freeze the
host at each epoch. Spectral peaks are not identified as canonical frequencies,
and refinement differences are kept separate from physical drift. Work is
in-memory, with AGAMA calls when that route is selected; no files are published.

## core/action_angle.py::measure_action_distribution

<!-- implementation-sha256: c06ed280c8a750c6bf5cc9af72a30c10a5dbb88dfefcdf0b6809a1e4c3e95ec6 -->

Reduce N-by-3 action vectors using aligned finite nonnegative weights. Exclude
nonfinite action rows and return the weighted mean in kpc km/s, finite-set
covariance in kpc² (km/s)², effective count and unavailable count. With a
shape-matched reference, also reduce the aligned action differences and their
covariance. The caller must establish exact identity/order correspondence; equal
array shapes alone cannot establish that correspondence. This is an in-memory
weighted statistic, not an estimated diffusion coefficient or a file export.

## core/action_angle.py::measure_actions

<!-- implementation-sha256: f58ae0d61b349c2cb61131ebb8b0997ef51e1be5fb4f09291a9406357d072d05 -->

Classify the supplied potential-domain record and measure one sample using
spherical quadrature or the explicit AGAMA potential. Domains permitting only
orbit diagnostics return unavailable action/angle/frequency arrays and false
support masks. Where the analytic control implements closed actions, retain
those values and measured-minus-analytic differences. Return `(report, arrays)`
with Jr/Jz/Jphi ordering, the action normalization, angle convention and domain
metadata; publication and independent numerical/physical acceptance remain the
caller's responsibility. It writes no files.

## core/allsky_map.py::make_allsky_map

<!-- implementation-sha256: e7571f31a60868c9759e95cb5328fc586974e06033692b5a29e2a0f10e09c61f -->

Read a saved DM or gas snapshot under the native-run lock, using the supplied
observer in Mpc/h, bounded particle sample and memory allocation. Bin represented
mass uniformly in longitude and sin(latitude), divide by each equal pixel solid
angle, and display log10 mass per steradian in Msun/h/sr on a Mollweide map.
Observer-coincident particles are excluded and counted. Verify source provenance,
then publish a fresh directory with `allsky.png`, `allsky.pdf`, original sample
and grid arrays in `allsky_data.npz`, and a hashed `allsky_contract.json`; return
the PNG path. Failed staged publication retains a failure record. This maps one
nearest periodic copy of matter, not emitted stellar light, observed flux or
a time-evolving light cone.

## core/analysis.py::measure_filament_proximity

<!-- implementation-sha256: 38dcc2497c3b43a215a30cc4995956b4471e44591359b225f03e8d152926becc -->

Take finite N-by-3 query positions and classified voxel positions in comoving
kpc. A periodic KD-tree measures distance to the nearest retained voxel node;
return that distance, a nonnegative lower bound using the producer's voxel-error
bound, the node distance as the upper bound and a support mask. An empty region
returns NaNs and false support. This is proximity to a discretized region, not
a fitted physical filament centreline or proper-distance measurement. It writes
no files.

## core/analysis.py::verify_native_environment

<!-- implementation-sha256: 2f614e5a103b33d23b090556c708524e03d66bfa34fc82846919ac0fbdfaec6f -->

Read and verify a `native-environment-v1` numeric product, every recorded
source hash, the source-run provenance and the complete published native HDF5
snapshot set. Recursively verify each halo-structure product named by its
request. Return the verified report or raise on a schema/integrity failure;
no environment is recomputed and no files are published.

## core/analysis.py::verify_environment_history

<!-- implementation-sha256: c9ab219eb34d230e7fd94d7e5b3aa5d68c28bf008c279a8311cd33cdd5e61b2e -->

Verify a `native-environment-history-v1` numeric product and every recorded
source hash, then verify each observation's original environment product and
its transitive snapshot/structure inputs. Return the report after integrity
checks. This reads existing evidence; it does not calculate a history or grant
physical continuity or convergence acceptance.

## core/analysis.py::halo_particle_counts

<!-- implementation-sha256: 203d682aebff47cd58bd4e2a0c29ac790502a845f4edacea3631597644db884f -->

Read native group particle counts for a saved snapshot. Prefer the `GroupLen`
vector; if absent, sum the columns of a two-dimensional `GroupLenType` array.
Return a one-dimensional int64 count vector, or `None` when neither field is
available. Malformed per-type dimensions raise instead of becoming zero counts.
The function reads the catalogue and writes no files.

## core/analysis.py::rockstar_masses

<!-- implementation-sha256: 6757c77af28146854e872080919420e51c7d8f26509579cd1d2b1cd9260511ba -->

Read the saved Rockstar table and select `mfof`, `m200c`, `m200b` or `mvir`
according to the requested definition. Return only finite positive entries in
the table's mass convention, or `None` for an absent table/unsupported or missing
mass column. It neither converts mass definitions nor assigns a new mass unit,
and writes no files.

## core/analysis.py::rare_halo_table

<!-- implementation-sha256: 9b7ab6f2e5a80bd46ed718bf1e36c1216d5c7b80cd2c3a94b3130b59ad447d89 -->

Convert pipeline-native masses from 10¹⁰ Msun/h to Msun/h and discard nonfinite
or nonpositive entries. Return inclusive counts `M >= threshold` for every
configured `rare_halo_thresholds_msun_h`, together with the maximum mass and
finite positive halo count; the maximum is `None` for an empty census. These
are counts in the selected input catalogue, not a volume-normalized abundance
or an extrapolation beyond its mass window. No files are written.

## core/analysis.py::hydro_diagnostics

<!-- implementation-sha256: fa3d2ec5be186879bba143ea339b63452ef51d52ce2899caeab143b0691b865b -->

Read a bounded gas sample and a separate full gas-SFR census from the saved
snapshot shards. Convert density using its a/h/cgs attributes to hydrogen-nucleus
number density with the implementation's fixed hydrogen mass fraction 0.76;
read explicitly Kelvin-labelled temperature or derive it from internal energy
and electron abundance, retaining conversion fallbacks and unavailable reasons.
Exclude effective-EOS/star-forming gas from thermal temperature measurements.
Write density/temperature distribution and phase figures under `RUN/results`,
and dispatch supported hydro field maps. Return histogram arrays, status/count
records and raw plus physically converted SFR totals in Msun/year where metadata
supports them. This ordinary-analysis phase diagram retains raw sample counts
and inverse-inclusion-weighted count estimates, rather than the registered
gas-phase exporter's mass-per-bin statistic. It performs no gas evolution.

## core/analysis.py::plot_mass_accretion_histories

<!-- implementation-sha256: cbe93c0946c87d43ca5a01c92d5911e096279e67ab1b26cece8c6600ce443efc -->

For a completed history result, export the formation atlas and plot each
provided main-progenitor track against scale factor under `RUN/results`.
Use the supplied `mass_msun_h` values, retain final halo IDs in legends and write
`mass_accretion_histories.png` through the project figure exporter. If supported
formation redshifts exist, also write `formation_redshift.png` against final
mass. It returns `None` and produces nothing for an absent/incomplete result;
it does not reconstruct ancestry or independently validate the input mass definition.

## core/analysis.py::plot_merger_histories

<!-- implementation-sha256: c8bbce81433bbd1ee8ef88aa4d86bef6b5fc879f23b3496047be62f7dc5438b8 -->

For a completed history result, pass its positioned tree to the merger-tree
exporter and plot the supplied secondary/main progenitor mass ratios against
descendant redshift. Save `RUN/results/halo_merger_mass_ratios.png`, marking the
1:4 reference and retaining an explicit empty-event annotation when appropriate.
Return `None`; an absent/incomplete history is a no-op. This displays existing
linked events and does not infer mergers from spatial proximity.

## core/analysis.py::plot_time_evolution

<!-- implementation-sha256: c6e2ad37c6be8ed5b945fc05570e8ab2c8f53c5684274211f1b38759d8d97410 -->

Read saved redshifts and rare-halo summaries, order them by redshift and write
`maximum_halo_mass_vs_z.png` when finite positive maxima exist. Read the run's
configured mass thresholds and write `rare_halo_abundance_vs_z.png` when any
threshold counts are available. The latter ordinate is halo count in the box,
not count per volume despite the historical filename; mass maxima use Msun/h.
The function writes figures, closes an unused threshold figure and returns
`None`; it does not interpolate missing epochs or run another finder.

## core/analysis.py::verify_paired_analysis_report

<!-- implementation-sha256: acec73bced4e42c49492207a60e7275a2c07eb4aafd570394835b0e82e70193c -->

Verify a completed paired-report publication pointer, its manifest SHA256 and
schema, and the referenced immutable generation under `paired_reports/`. Check
every recorded artifact hash, the public PDF hash and that the manifest did not
change while reading. Return the manifest record; raise for incomplete state,
path escape or changed assets. It reads publication files without regenerating
the PDF or accepting the scientific comparison.

## core/apertures.py::validate_aperture_contamination

<!-- implementation-sha256: 5ef18a275c1a87072527148ab96bb352de2c3cfd114e628b077f35f89d705eda -->

Validate a finite numeric workspace of at least 96 MiB, a positive integer
aperture-count bound and a coarse-mass-fraction criterion in [0,1]. Return
`None` or raise before measurement. This checks declared controls only; it
neither reads particle masses nor establishes that an aperture meets the criterion.

## core/apertures.py::aperture_contamination_runner

<!-- implementation-sha256: 845322cc436e3a31e1f936ecd21052bb9b4225a93393f2e53c26f962817431e2 -->

Read a bounded `native-science-apertures-v1` product, verify its snapshot,
IC/lineage/source hashes and epoch, and measure actual native coarse-matter
contamination in each supported sphere. Centres use original-box comoving
kpc/h; radii use proper kpc. Unresolved selections retain reasons without
invented measurements, and overlapping spheres are never summed into a union.
Write `contamination.json` in the owned workflow generation and return its
report; a measured criterion failure is written before raising. An execution
result does not grant scientific acceptance.

## core/apertures.py::validate_catalogue_apertures

<!-- implementation-sha256: 6e0343cc1ac4c53266ed8ddf28ef11fcfb8191455cd4b3cd02b54f384a6baf67 -->

Require a positive integer maximum aperture count and positive finite numeric
memory in GiB. Return `None` on valid controls or raise on invalid values.
This performs no catalogue intake, classification or geometry measurement.

## core/apertures.py::native_catalogue_apertures_runner

<!-- implementation-sha256: d2f3e2de8994fcaa412ad7fea76c19dd3268e4bad0d391bb488c51ea3297e9df -->

Read a stellar catalogue and verify its original native membership, exact IDs,
positions, masses, cosmological frame and source hashes. Support either a
full native host catalogue or its explicitly selected type-4 worldline cohort;
recheck the latter against the original stellar census. Build mass-centred
bounding spheres through `catalogue_apertures`, with unresolved progenitor or
uncalibrated stream selections retained. Write `apertures.json` in the workflow
generation and return the report. This reads existing evolution; it does not
create stars or infer missing ancestry.

## core/apertures.py::validate_encounter_apertures

<!-- implementation-sha256: 1e8e095331e04b29cb2f5586c7423636965bfb10cce230e2f3947d4d55842c73 -->

Require an integer saved snapshot label >=0, a positive integer aperture-count
bound and positive finite numeric memory in GiB. Return `None` or raise on
invalid controls. It does not certify an encounter or read its worldlines.

## core/apertures.py::native_encounter_apertures_runner

<!-- implementation-sha256: 18102b822893a7569e75e4073284e2e4421f75d122e36bf2e574eaffa50904e6 -->

Verify the encounter catalogue, converted frame and original periodic type-4
worldlines against the exact saved native stellar census, images, clock and
namespace. At the requested snapshot, create a search-radius sphere centred
on the original saved stellar position only for a certified event whose time
bracket contains that snapshot. Convert its centre to wrapped comoving kpc/h;
retain the search radius in proper kpc and event/saved times separately in Myr.
Write generation `apertures.json` and return the report; unsupported geometry,
missing events and out-of-bracket epochs remain unresolved. This does not
measure density at the interpolated event time.

## core/apertures.py::validate_recovered_apertures

<!-- implementation-sha256: 6b5b700fd72fd2ccbd3a046b399a6dba1d9e5eba4644eee7cb6e13933e513042 -->

Validate positive integer particle/aperture bounds, finite memory of at least
288 MiB, distinct nonnegative integer snapshot labels and a coarse-mass-fraction
criterion in [0,1]. Return `None` or raise. These are intake/resource controls,
not a descendant match or contamination measurement.

## core/apertures.py::validate_recovered_catalogue_controls

<!-- implementation-sha256: 8cf0597cc7b57af05a202ef4e7dc0152c89ef918e1cb187a6cbc1e36041b41d6 -->

Require exactly the supported frozen stellar-classifier controls: integer
population/neighbor/edge bounds, positive proper-kpc and km/s link scales,
probability/precision floors in [0,1] and a Boolean progenitor-splitting policy.
Reject an edge budget exceeding its declared memory share. Return `None`;
valid settings alone do not provide classifier calibration or detect a stream.

## core/apertures.py::recovered_apertures_runner

<!-- implementation-sha256: 92bb6706090191d9732b1f7e958c6f22e317f7128830e7e1c2ea671ef481910e -->

Follow a verified descendant-recovery product over requested native snapshots,
using a separately hashed frozen catalogue-control file. For matched hosts
with stars, run the production stellar catalogue stage and publish its numeric
product, derive supported progenitor/stream spheres, optionally add measured
encounter spheres, then measure contamination for each epoch. Write per-snapshot
catalogue/selection/contamination products and `contamination-history.json` in
the owned generation; return the history report. Unmatched or star-free hosts
and missing encounter products retain unresolved rows. This is post-processing
of saved evolution, with explicit memory/count bounds and source checks.

## core/apertures.py::validate_zoom_support

<!-- implementation-sha256: b3f24e3349475856f36f904cbc8a77981adc94b51bc11639e150c0e40585874f -->

Validate the inner `native-sidm-zoom-support-request-v1` request: positive
streaming/count/memory bounds, a saved snapshot, distinct exact uint64-range
selected IDs and a nonempty namespace. Check dimensionless coarse-kernel and
omitted-scattering limits and require cumulative-since-initialization omitted
bound scope; reject a workspace estimate exceeding memory. Return `None` or
raise. It does not read or accept the native support fields.

## core/apertures.py::native_zoom_support_runner

<!-- implementation-sha256: f6022c3d9062513b1be07d273397a87be69ba04a706607ba8b02cdd78441f6e5 -->

Require the requested support workspace to fit the workflow allocation and
its destination to resolve inside the owned generation. Delegate publication
to `publish_native_zoom_support` and return its result. That publisher streams
selected native fine-DM guard fields, preserves exact-ID order, compares measured
maxima with predeclared dimensionless limits and writes the numeric audit.
The runner does not evolve particles or turn an optical-depth bound into an
observed collision count.

## core/apertures.py::native_zoom_support_main

<!-- implementation-sha256: f93f4b60dfac7ca01ca320d2bf5d8e3f521f41cc41c1699555a146c55b27f7d9 -->

Parse required `--request` and `--output` paths, bound the request-file intake,
load its JSON and print the numeric publisher's JSON result. The file contains
the full parameter envelope: `native_stage_receipt`, `native_run_directory` and
the inner `request` object, not only the inner request. Publication reads verified
native support fields and writes the destination audit/arrays. Validation or
integrity errors propagate. A measured FAIL remains a report status; this handler
does not itself map it to a nonzero exit code or launch native evolution.

## core/analysis_figure_campaign.py::prepare_analysis_figure_campaign

<!-- implementation-sha256: db6e253e34240a037db47bf569e19438496455cb97c910280d4888bfd0a0e16d -->

Validate an `analysis-figure-campaign-request-v1` request binding an original
cosmography or stream/inference campaign, resources, wall budget and distinct
figure/producer selections. Verify source hashes, require each variant's
original producer kind, and compile a workflow containing the analysis command
followed by its dependent registered figure adapters. Preserve BAO likelihood,
linear-bundle and drag-execution bindings explicitly. Write `workflow.json`
and `manifest.json` in a fresh directory and return the preparation record.
This prepares recipes only: it does not execute analysis, sample a posterior,
render a figure or accept calibration.

## core/batch_analysis.py::run_batch_analysis

<!-- implementation-sha256: b30af58b3be2d1f26af10ddbedb13eb58c97338352f792e1602f01745b35ec6e -->

Analyze a nonempty unique list of saved runs sequentially with the selected
ordinary products; specialized per-run request products must be run separately.
Write each run's new analysis outputs and maintain a fresh batch destination's
`batch_report.json` and `index.html`. Optionally compare the chosen reference
with every other run and build all-run overlays for suites of three or more,
using supported HMF/power/correlation and DM-map products. Return the report;
individual errors are retained while later runs continue, and interruption
marks active work before re-raising. Comparisons do not establish phase matching,
numerical convergence or causation.

## core/batch_comparison.py::compare_selected_hmf

<!-- implementation-sha256: 849478f17fc4c951740a1c84da642ab6a7b81daa6a7418005cb75630cd600eff -->

Under both native-run locks, verify the two selected analysis bundles belong
to their runs and share an HMF mass definition and abundance-unit string.
Choose the latest low-redshift pair within tolerance, retain positive finite
measured bins and overlay their source HMF curves without rebinning or a new
halo finder. Publish a fresh directory with `hmf_comparison.png` and hashed
`comparison.json`, recheck original bundles, and return the PNG path. Failures
retain a staged record. Bin selections/resolution follow each source analysis;
this descriptive overlay is not the matched-phase HMF-ratio experiment.

## core/batch_simulation.py::configure_and_run_queue

<!-- implementation-sha256: 92d26a7e2562819675820df92085b83f8a6eeb170ee706d86f05419a25b4673c -->

Interactively configure at least two distinct simulations, first running each
selected CLASS/AxiCLASS spectrum during configuration because particle setup
needs that bundle. Save answers and a queue report/index under the external
state root, then invoke GADGET runs sequentially with their own destinations.
Record native completion separately from each saved analysis workflow's status;
continue after individual run errors, preserve reports and restore `sys.argv`.
Return 0 only when every run and its analysis are complete, otherwise 1;
configuration/interruption errors are recorded and re-raised. This function
launches real solver work rather than only drafting a queue.

## core/setup_walkthrough.py::draft_path

<!-- implementation-sha256: 8fde20e628d6846890c68a0badbade35cffaf59120ca7b8daf192594489f6b52 -->

Resolve an explicitly supplied draft path after expanding `~`; otherwise
return `drafts/interactive-setup.json` under the configured external runtime
state root. It returns a `Path`; neither branch creates directories or writes
the draft. Questionnaire persistence belongs to the caller.

## core/resources.py::require_resources

<!-- implementation-sha256: d814a6e1e466873c2ce187c785583c45df3a8292b3268d03f521073902b2fa5d -->

Compare a production plan with the requested MPI ranks, threads, declared
memory/disk and optional per-rank memory, plus available local CPU slots and
disk space. An explicit scheduler topology delegates to the topology-aware
check; optional available-memory bytes add a measured local limit. Return a
PASS record with the plan hash and no configuration changes, or raise with
resource deficits. Disk space is read from the nearest existing destination
ancestor; no directory, report, process placement or resolution is created
or adjusted.

## core/resources.py::validate_sampling_options

<!-- implementation-sha256: 9a490e54ef67e28f93b291380793501875cb8d6c1da2135fb03a10a8589c6c4b -->

Accept only interval/count bounds for local process observation. Require a
finite numeric interval between 0.05 and 5 seconds and positive integer maxima
for retained samples, retained processes and scanned `/proc` entries; omitted
values use the sampler defaults. Return `None` or raise. Validation does not
start an observer, sample memory or constrain a native process's allocation.

## core/resources.py::LocalProcessResourceSampler

<!-- implementation-sha256: 8159415f59a50e169aabead1f445501be52ce3b6fad297130d79b32c21c6b7e1 -->

Observe one owned Linux process session through `/proc`, keyed by PID and
start ticks to distinguish reused identities. `start(pid)` takes an initial
sample and starts a bounded-interval daemon observer when supported; `stop()`
joins it and writes the JSON report to the configured output, including retained
samples/processes, sampled RSS peak, CPU time, bounds and observation errors.
On unsupported platforms or an unreadable process, retain UNAVAILABLE coverage.
This is approximate local sampling: shared pages may be counted repeatedly,
remote/escaped processes are unmeasured, and it neither enforces memory limits
nor measures a guaranteed exact peak. Constructing it alone starts no thread.

## core/resources.py::resources_main

<!-- implementation-sha256: fdf737738a258e06fbcf56c1e9c6bc24a6b460aa098864a8955f95107a5765e6 -->

Parse a resource request, required fresh `--output` path and optional
`--measurements` JSON. Estimate production resources and, when supplied, update
the plan from actual stage receipts; atomically write the plan JSON and print
it. Reject an existing report rather than overwriting history. This CLI does
not allocate a machine, change the requested resolution/ranks or run the
scientific campaign.

## core/baryon_products.py::hydro_baryon_inputs

<!-- implementation-sha256: 237767e79e22963fad54cefcc0a2b67501a474320072fcad4d70559113416182 -->

Verify the native hydro-history product through the BH/source-event binding
reader, then reread its numerical arrays within the allocated memory share.
Require the reread report to equal the independently verified report. Return
`(report, arrays, source_hashes)` with both sets of bindings. This reads saved
census/journal products; it neither evolves gas nor publishes another product.

## core/baryon_products.py::export_native_sfr_history

<!-- implementation-sha256: 8cf483a7d7e759700faf3b48d84434f8b79f5445ce670e2218739b133829a123 -->

Verify a saved hydro census and reduce each complete gas population through
the existing SFR-history estimator. Publish `star-formation-history.json`, its
numeric audit and the registered gallery in a fresh destination; return the
gallery exporter result. The ordinate is the whole-snapshot instantaneous sum
of gas rates in Msun/year versus actual scale factor, not a volume-normalized
cosmic SFR density or a count of stellar births. Measured zero rates remain
zero; incomplete epochs retain NaNs and false support, breaking plotted lines.

## core/baryon_products.py::export_native_mass_return

<!-- implementation-sha256: 971d005852d3990355c87e0774667615342e95374cf45d6d92441b5eace2139c -->

Read the verified hydro history and aggregate committed delayed-return rows,
keeping stellar source losses and gas recipient gains as separate signed Msun
ledgers. Write `mass-return.json`, its numeric audit and the registered gallery
in a fresh destination. The first selected snapshot is the zero baseline;
returns before it are unknown. Prompt SH03 recycling is excluded, signed
values are not clipped and discrepancies are not repaired by plotting. Return
the gallery result; yield validity and source-budget acceptance remain separate.

## core/baryon_products.py::gas_phase_arrays

<!-- implementation-sha256: 0ce4d4efa6b9a19e91861ba46b4aa3475a187004a9185ac8d86adbf0a1834420 -->

Select one epoch's aligned native identity, species, mass, density, temperature
and SFR arrays, checking their declared units and Boolean availability masks.
Return the original census plus gas/effective-EOS/support masks. Supported
thermal gas requires measured zero SFR, available density and temperature, and
positive finite mass, proper density and temperature. Preserve excluded rows
rather than replacing them; this performs no histogramming, publication or
thermal evolution.

## core/baryon_products.py::gas_phase_inputs

<!-- implementation-sha256: 40c5a6deb3139ddb25ca8759e2ec454ae2d5d84a590ec2b314ee82b11b618ef9 -->

Validate a `native-gas-phase-request-v1` request with strictly increasing
positive physical density and temperature edges, each containing 2–4,097 values.
Check histogram workspace against the memory share and verify the bound hydro
history SHA256 before reading it. Return `(report, arrays, source_hashes, edges)`.
No gas is evolved and no figure or new product is written.

## core/baryon_products.py::export_native_gas_phase

<!-- implementation-sha256: e12da6cf3971392a6484680d2aebb94ec00665479e259a848f4b5448397c6d1b -->

For each saved hydro epoch, derive the single-phase gas support mask and keep
a separate in-display-range mask using the requested physical edges. Publish
per-epoch `gas-NNN.json` audits with exact IDs, original density/temperature/mass
and masks, then render the registered gallery. Axes are proper g/cm³ and K;
colour sums original gas mass in Msun per bin, with no probability normalization.
Retain effective-EOS exclusions and supported particles outside the display
range. Return the gallery result and reject an existing destination.

## core/baryon_products.py::native_metal_history_arrays

<!-- implementation-sha256: 5692159663437778145337fed5ed36ba461e2b0af0b24ca551c9cd577105af11 -->

Reduce aligned saved gas composition and stellar birth fields using the fixed
eleven-element order. Require gas fractions to close, then calculate gas element
masses and mark whole-gas totals unavailable when coverage is incomplete.
Retained SSP birth metals use initial locked mass times birth metallicity;
current stellar chemical inventory stays unavailable for a nonempty stellar
census. Separately accumulate committed stellar ejecta and gas-recipient element
gains since the first selected epoch. Return arrays versus increasing scale
factor, including subtotals and completeness/count masks. No files or inferred
global gas-plus-star conservation total are produced.

## core/baryon_products.py::export_native_metal_history

<!-- implementation-sha256: ef4c218b275b31b688189bf59b935e5841d75826d36f8e04a858304c84062f8f -->

Verify the native hydro history and write `metals-history.json` with element
inventories, birth bookkeeping, separate delayed-return ledgers and explicit
missingness. Render gas, stellar and return panels in a fresh registered gallery;
return its result. Metal mass excludes H and He and uses Msun versus actual
scale factor. Current stellar chemistry is not inferred from initial birth
metallicity, and neither normalization nor closure repair is applied.

## core/baryon_products.py::population_inputs

<!-- implementation-sha256: e9421c857ea1501c6bedeae0710cf0e07fd9636c43050cc7b208732eb5ab86e4 -->

Validate a `native-population-history-request-v1` request, hash-check and read
its hydro history, and initialize the verified FSPS model from the synthesis
backend settings. Require two distinct colour filters present in that model.
Return the hydro report, combined source bindings and model metadata. Backend
initialization loads installed FSPS/data assets; this function does not itself
synthesize the epoch populations or publish history files.

## core/baryon_products.py::export_native_population_history

<!-- implementation-sha256: 86b3dceb34b3115933c16603eea4c47533e3622d5dbb22d6f4953805ea92a40b -->

For each verified saved epoch, read native SSP birth states and clock-derived
ages, synthesize intrinsic FSPS light and publish `sps-NNN.json` audits with
original ages/metallicities. Require the same pinned model throughout and
strictly increasing saved proper times in Myr. Write `population-history.json`
and the luminosity/AB-colour gallery; return the gallery result. Whole-census
luminosity is masked if any SSP is unsupported; a supported subtotal remains
separate. Empty populations have zero bolometric light and undefined colour.
Colours use summed band luminosities, not the mean of individual magnitudes.

## core/stellar_population.py::gas_epoch

<!-- implementation-sha256: 65dd5a2d8483dbe86a762891988d932efac8e7edb03f3b1cb8a568d6a79b60e5 -->

Read a saved native hydro census, select gas rows and calculate composition/
electron-based mean molecular weight, using the native EOS value where available.
Join caller-supplied measured smoothing lengths by exact ID, require complete
positive proper-kpc coverage, and attach the actual cosmological or isolated
clock in Myr. Return the gas/host/namespace/physics record and source hashes.
The caller supplies host mapping and resolution fields; particle separation
is not substituted for a missing smoothing measurement. No files are published.

## core/stellar_population.py::native_births

<!-- implementation-sha256: 9096b30ac5affb543819a178a7fcd20c268d9ca9688b1604cb0bc5289fc0e813 -->

Read the committed SSP birth-journal interval between two matching native
source-model states, using their commit/event counters and recorded parameters.
Convert its birth times to Myr with the run's cosmological clock or isolated
time unit. Return birth records and source hashes: namespace-qualified exact
SSP IDs, parent IDs, observation epochs, gross/locked birth masses in Msun and
metallicity. Use the explicit end-epoch fallback only when supplied, retaining
that origin label. This reads actual birth events and creates no new stars
or publication files.

## core/stellar_population.py::pinned_fsps

<!-- implementation-sha256: 1825206889a96d9acffe7872e8f624fa5e25922c44e6baaddf3033c41d25a13f -->

Verify retained SPS source pins, the installed FSPS distribution/import path,
`SPS_HOME`, declared physical-table hashes and complete model-asset coverage.
Require intrinsic SSP settings, an explicit IMF and the requested compiled
library identities. Initialize and return `(StellarPopulation, model_metadata)`
with normalization and source hashes. This loads the real installed backend
and physical tables; it does not install packages, run native star formation
or publish a spectrum. The returned population is configured for one initially
formed solar mass, distinct from a native particle's current mass.

## core/stellar_population.py::synthesize_stellar_light

<!-- implementation-sha256: 56a25ef95acae17682cdcdf4ec7ec2792e03f90c57a4c87ceed4c2ff872da25f -->

Validate exact unique type-4 SSP identities, native gross birth provenance,
positive formed masses and aligned present masses/ages/metallicities. Convert
Myr to Gyr, retain separate age/metallicity support masks and require requested
domains to lie inside the actual pinned FSPS tables. For each supported SSP,
scale its spectrum, bolometric luminosity and model surviving mass by gross
initial formed mass; convert absolute AB magnitudes to band Lnu. Return report
and arrays containing spectra in Lsun/angstrom, band luminosities in erg/s/Hz,
AB colours and native/model mass fields. Unsupported rows stay NaN; no model
clipping or publication files are created, and native evolution is not advanced.

## core/black_hole_ledger.py::event_state

<!-- implementation-sha256: 79376434663078e7484aaf1a03413eef44124dab6d076c6fdfc52302067a94da -->

Extract the four recorded native accretion-compensation values for the chosen
`before` or `after` side: pending mass, pending correction, gravity correction
and physical-mass correction. With an index, select one parent's values from
a merger row. Return a tuple without unit conversion, arithmetic validation
or file I/O; the caller must supply an already ingested journal schema.

## core/black_hole_ledger.py::merge_values

<!-- implementation-sha256: c5294aa929f7a6e80f813208467ed4e8b973f56830b6c37cbb858166b298a556 -->

Combine two four-value native compensation states. Replay Kahan addition for
the represented pending demand, including the second state's correction, then
sum the gravity and physical-mass corrections separately. Return the merged
tuple. Values remain in native mass units; pending demand is not applied mass
transfer or feedback. This performs no journal lookup or publication.

## core/black_hole_ledger.py::validate_roundoff_event

<!-- implementation-sha256: 4bd7efac3e025d8ac9dfefbb26a5aa4b6a39210bdded82994d308058644c6549 -->

Replay one ingested version-one accretion-integral or accretion-merge event.
Check pending demand, requested/applied inflow and retained gain, reconstruct
compensated gravity/subgrid mass sums, and require the recorded after-state
to match the native arithmetic exactly with finite nonnegative represented
demand. Return `None` or raise; zero applied inflow retains the prior mass
operands. This validates numerical bookkeeping, not a complete physical
mass/energy budget, and writes no files.

## core/black_hole_ledger.py::allocate_roundoff_state

<!-- implementation-sha256: 41db77f9370499b445708fc44d96faadd36de2f37c83c087483b60900c9565b7 -->

Allocate zero-filled arrays of the requested count for the BH accretion
version, analysis mass/correction columns and untouched native compensation
columns. The version array is uint64; other arrays are floating point. Return
the dictionary. Version-zero allocation is inactive/legacy state, not evidence
that accretion was measured. It performs no unit conversion or file writes.

## core/flagship_bh.py::bh_figure_input_bindings

<!-- implementation-sha256: bd470642cc2c863a858d5789156f1d451676b100ab82b999c49abd437b8eccc9 -->

Verify a native BH history directory's provenance checksum, HDF5 payload and
run contract within the memory allocation. Use an explicitly supplied run or
require exactly one matching run among the recorded parameter path's parents;
verify every saved snapshot hash. Return `(record, history_path, run_path,
source_hashes)`. This reads the completed history and its original build/input
bindings; it does not reconstruct a BH history or render figures.

## core/flagship_bh.py::export_native_bh_figures

<!-- implementation-sha256: 360202349c50e205d2f3d801fd21383cccae1edbebb61a8577ba4fd318e4ee27 -->

Verify the saved BH catalogue and match every summary observation to its exact
ID, epoch, HDF5 fields, units and namespace. Require increasing positive scale
factors, then publish one `bh-ID.json` track per BH and a registered gallery
for subgrid mass, accumulated coupled energy and mean-inflow/saved-mass
Eddington-cap ratio. Units are Msun, erg and dimensionless; absent epochs stay
NaN/false. Return the gallery result in a fresh destination. This exports saved
source results and does not create accretion, infer mergers or turn the mean
inflow ratio into an instantaneous luminosity ratio.

## core/flagship_bh.py::bh_event_input_bindings

<!-- implementation-sha256: c863cde653acc2ff22216d49c8113c8e86fab7c3d0e2eada0062541e11d752f5 -->

Bound native hydro-journal metadata intake, read its numeric product and require
the recorded replay memory to fit the supplied allocation. Independently replay
the existing `native-hydro-history-v1` verifier and require an identical report.
Return `(report, source_hashes)` including numerical-audit bindings. Integrity
replay preserves the original accounting status; reading a consistent report
does not relabel an accounting failure as accepted physics. No files are written.

## core/flagship_bh.py::export_native_bh_merger_timeline

<!-- implementation-sha256: 45ce0f6e12fa3303e3d0e4832f51ffd4923e76f5d16f675179e33aee05deb9a6 -->

Verify the hydro/source journal and convert its original merger source-step
brackets to proper Myr using the bound cosmological clock or isolated unit.
Publish `native-merger-timeline.json` with exact parent/survivor IDs, original
rows and brackets, then render event ordinal versus applied time in a fresh
gallery. The plotted time is the source-step end; continuous occurrence remains
bracketed. An empty complete journal remains an explicit zero-merger result.
Return the gallery result; no merger is inferred from pair proximity.

## core/flagship_bh.py::export_native_bh_recoil_trajectory

<!-- implementation-sha256: 42689fecff17ed5cc43a22fc97edf6caf85147892a613072ee4739d528961676 -->

Select an exact BH survivor/event counter in a verified history and hydro merger
journal, and verify a separate recorded-image product bound to all saved epochs.
Read the original proper-kpc coordinates, unwrap using those images once, and
retain supported samples after the selected merger only until later ancestry
change. Publish `recoil-trajectory.json` and a registered 3D gallery; return its
result. Missing event/image/epoch support stays masked. There is no event-time
interpolation, speculative extrapolation or claim that the saved displacement
isolates a causal recoil response.

## core/stellar_returns.py::validate_stellar_model_stage

<!-- implementation-sha256: 09798fe33201f3f43906206944993e310a579fd08b2cce4bc1347c9365d45cbe -->

Require positive finite numeric memory in GiB for stellar-model preparation.
Return `None` or raise before request intake. It does not read yield tables,
compile a stellar model or validate mass/element/energy closure.

## core/stellar_returns.py::validate_stellar_branch_stage

<!-- implementation-sha256: 0cb27906584d42e05502f5fb251dd61625d4cf9c8f2c0ca3a84e9eb9322a5774 -->

Require an explicit `prompt_only` or `delayed_returns` mode. Delayed returns
need a model-manifest reference and exactly the supported kernel/source-fraction/
count/subcycle control keys; prompt-only settings must not carry either delayed
model or controls. Return `None` or raise. This checks branch configuration
shape, not model-file integrity, each control's numerical validity or evolution.

## core/stellar_returns.py::validate_stellar_history_stage

<!-- implementation-sha256: 4eda0eb02c05dde6355e107677a77e934754337305a9db0c32cd2c52fd776eee -->

Require at least two distinct nonnegative integer snapshot labels, positive
finite memory in GiB and valid native hydro-accounting tolerances. Return
`None` or raise. The actual census/journal audit and independent replay belong
to the history runner; this validator neither runs them nor publishes a result.

## bootstrap.py::paths

<!-- implementation-sha256: c67887574fdcd9a5d7d973becba53663f14c212e129f4944a8dcd58bac76b85e -->

Resolve the external state root from an explicit path, `RESEARCH2_STATE_ROOT`
or the sibling `research2-runtime` default. Reject a root inside the source and
return `(state_root, state_root/environments/default)` as Paths. This resolves
locations only; it creates no environment or directories.

## bootstrap.py::dependency_plan

<!-- implementation-sha256: d1553686e4fdb979297197a6259b3247c97bb20aaf4df91da15a30f53a4d8d86 -->

Read `configs/runtime/dependencies.json`, validate its schema and traverse the
selected profile's dependencies before its own packages. `all` selects every
profile; completed groups are visited once and unknown/cyclic groups raise.
Return the ordered package dictionaries and full declaration record. This is
an installation plan, not a package import, installation or backend calculation.

## bootstrap.py::install_packages

<!-- implementation-sha256: 592612cfad123fb06a05c7472a1a98a8aad28b7ebf1d69dbcc86575fd9a21858 -->

Install each declared package through the target interpreter's pip subprocess.
For bundled packages, verify their source hashes, copy source to an external
build directory, supply missing build-only README metadata there and apply only
declared pip/SCM options; macOS survey packages can receive libomp flags.
After installation, inventory installed bytes with that interpreter and return
bundled-package provenance records. Registry requirements may download packages;
local source builds can compile code. This function changes the environment and
external build/cache state, not the retained source tree or simulation results.

## bootstrap.py::main

<!-- implementation-sha256: 6f9c1fc4ba0c663dacd4c2ce72f03912f34d035b25f9daf50d2ef6d1e63bc0f4 -->

Parse setup actions, resolve the dependency plan and external environment,
and support a dry-run JSON plan without installation. `--install` creates or
validates the venv, installs packages sequentially and atomically records partial/
complete progress with package and interpreter provenance, preserving previous
receipts. Check/build actions first verify the selected Python profile inside
that environment, then delegate requested native setup flags to `commands.py
bootstrap`. Return the action's exit code. Native Windows and too-old Python
are rejected; successful setup does not represent a scientific simulation.

## core/assembly.py::orbital_assembly_events

<!-- implementation-sha256: aa8f02343b0d1ad52856cd16654f5009354aef2ff10034e42405b28a88ab884a -->

Construct object/host worldlines in a common inertial frame and search valid
overlapping Hermite intervals for relative-distance minima and optional maxima.
Find infall from inward roots of distance² minus the linearly evolving host
radius²; also inspect shared knot boundaries and deduplicate nearby same-kind
events. Return candidate times/brackets in Myr, radii in proper kpc, method
metadata and the two worldlines' certificate status. Interpolation itself adds
no event-error certificate or physical merger evidence. No files are written.

## core/assembly.py::execute_assembly_stage

<!-- implementation-sha256: 9b3722292034b1d8f6cb19a6fdc53e6bf93d58e66752e50308051a9047a7a0f9 -->

Dispatch an assembly-history, orbital-event, first-star-environment or
primordial-cloud stage using its declared inputs and controls. History stages
reconstruct a quality-masked branch, measure events and fit a conditional
accretion transition; first-star stages can read real gas/birth journals or
supplied records before joining cooling, births and ancestry. Return
`(report, arrays, namespace, epoch)` for the workflow publisher; only the cloud
route supplies its report epoch here. Input helpers read source products and
record bindings. This dispatcher does not run a native evolution.

## core/assembly.py::export_native_assembly_figures

<!-- implementation-sha256: 99f246a51b10d51fd9bdefcffc914a755a96f62a3142ff41c928b899e2870ab4 -->

Verify a main-branch product against the exact identity graph, source files,
node attributes, units and increasing saved cosmological clocks. Preserve
original IDs and node/interval masks, inserting unsupported display rows where
intervals are missing. Write `assembly.json`, numeric sidecars and `products.json`
plus eight registered histories for mass, accretion, spin, concentration,
shape, stars, morphology and conditional transition. Return the fresh gallery
result. Orbital-event plotting remains a separate product; unavailable fits
and gaps are retained rather than interpolated away.

## core/bao.py::bao_prediction

<!-- implementation-sha256: a83e3b5ea0d12411d02eebe832830d9f471d7b0c125b3854b90187ed1dc9d516 -->

Evaluate a supplied expansion history at aligned positive redshifts. Compute
transverse distance DM, radial distance DH=c/H and DV=(z*DH*DM²)^(1/3), then
select each row's requested DM/DH/DV quantity and divide by the explicit positive
drag sound horizon in Mpc. Return dimensionless predicted ratios in input row
order. It does not fit parameters, calculate the sound horizon or write files.

## core/bao.py::parse_drag_sound_horizon

<!-- implementation-sha256: ee2f35b52bb8bde59c6f9f453da024e9e0e03727f3e7dc38aabd236113ac4a37 -->

Parse exactly one CLASS log block containing baryon-drag redshift and its
comoving sound horizon. Require finite positive values and return `(z_drag,
r_s_mpc)`; ambiguous, absent or invalid output raises. This reads a supplied
text string, not a solver executable, and retains the precision of the printed
CLASS summary rather than adding unseen digits.

## core/bao.py::publish_conditional_bao

<!-- implementation-sha256: 8b4f99c81cc1e0fe171074d2283510a268ef55ee78df79c79073bd689fed8c48 -->

Verify and evaluate the bound published BAO measurements against one frozen
CLASS background and its actual drag execution through `conditional_bao_inputs`.
Publish the resulting numeric audit to a fresh destination, retaining all
original measurement rows, residuals, full covariance and source hashes, and
return the audit result. These dimensionless distance ratios supply a conditional
likelihood; no parameters are fitted, posterior sampled or CMB likelihood added.

## core/bao.py::validate_bao_inference_settings

<!-- implementation-sha256: e3c084c3dbcb2630bdaded31e29305da9556b0dd92a208ec3e91b515a93deba5 -->

Construct and return a ParameterSpace with exactly ordered positive,
dimensionless physical priors for `alpha_perp` and `alpha_parallel`. Require
explicit emcee settings with bounded integer walkers/steps/burn-in/thinning/seed,
a uint32 seed, surviving post-burn samples and retention within the memory
estimate. Invalid settings raise. This validates configuration without sampling
a chain or establishing convergence or parameter-space coverage.

## core/bao.py::dilate_bao_ratios

<!-- implementation-sha256: e6bc79192f1ec9218cdea7b059837ba7f5a0d634b8805db919fa4a071e7153ee -->

Scale aligned baseline distance ratios using positive finite transverse and
radial dilations. DM/rs uses alpha_perp, DH/rs uses alpha_parallel, and DV/rs
uses (alpha_perp²*alpha_parallel)^(1/3). Require exactly those two parameter keys
and the original quantity-vector dimensions. Return the scaled array; no
background, sound-horizon or likelihood calculation is performed here.

## core/blender.py::native_catalogue_inspector_main

<!-- implementation-sha256: 9b47164512e12b45d6fb6cbc6cdb8876555cfc1bbf109cbdae2b965c7bbb1d5e -->

Run inside Blender, taking an input overlay JSON and output .blend path after
`--`. Clear the current scene objects, create synthetic selectable catalogue
markers and assert their count, bound half-mass scale, metadata, hidden-render
semantics and same-host/epoch selection behavior. Register the inspector, save
the scene and write a sibling JSON PASS receipt. This deliberately edits the
current Blender scene and exercises a supplied inspection fixture; it neither
evolves particles nor establishes halo-finder accuracy.

## core/blender.py::main

<!-- implementation-sha256: c01c9eb610862b9c10188172639da6388fe38c6322db3bffe6b4ae9fb211fef8 -->

Select a registered scene command from explicit arguments or Blender's `--`
argument tail, defaulting to the ordinary blender command. Temporarily rewrite
`sys.argv`, invoke the selected callback and restore the previous arguments in
a finally block; return the callback result. Scene creation, saving or rendering
belongs to that callback. This is in-process dispatch; callbacks using bpy
need Blender's Python environment.

## core/blender.py::stream_inspector_main

<!-- implementation-sha256: 35b53a1ef16dfe7db9512754fa8c9aeeaf45d9d1f74c42655fde2a726d6ddbc0 -->

Run inside Blender with a measured-stream manifest, required output .blend,
point-count bound and marker radius in kpc. Parse Blender's `--` argument tail,
load the measured inspector points/tracks with those controls, create the output
parent directory and save the active scene. The loader supplies source checks;
this helper adds display assets and writes a Blender file, not a new stream
evolution or stellar catalogue.

## core/blender_scene.py::load_verified_display_products

<!-- implementation-sha256: 5390f0504ed52c2c09716f3eca994774ae680c297db6a423548d0870315af12f -->

Read a sealed `measured-display-v1` manifest and verify explicit coordinate
conventions, at least two saved epochs in increasing native-clock/frame order,
unique exact object identities and supported kinds. Check confined hashes for
lifecycle/source links, flat measured geometry layers, SIDM diagnostic overlays
and assets, then validate the lifecycle index. Return the manifest dictionary
or raise on an integrity/contract failure. It reads files without copying assets,
constructing Blender geometry or inventing missing object states.

## core/blender_scene.py::join_exact_object_states

<!-- implementation-sha256: b1802221c5f90e13029478d83f011a3473e1b68b9c2b452cb9c6edcb16990810 -->

Group each timeline's saved object records by the manifest's exact identity
key. Return a dictionary of tracks retaining frame, snapshot, epoch contract
and original record at every observed state. It does not interpolate missing
states, verify source hashes or create Blender objects; use a verified manifest
when consuming external display products.

## core/blender_scene.py::visibility_schedule

<!-- implementation-sha256: bf315bb37315573266f8de7053acb265414e482c966216c1b077bee3194c22ec -->

Build the sorted union of exact object identities in the timeline and return
per-identity frame/presence records for every saved display epoch. Presence means
the identity appears in that epoch's object list, without inferring existence
between detections. This is an in-memory display schedule; source validation
and actual scene visibility keyframes belong to other helpers.

## core/blender_scene.py::package_display_products

<!-- implementation-sha256: d0a36c2b5dd40d81809a8b6fa16132e6b63de3c02a802763cfbad84aeabb5b91 -->

Verify the display manifest, lock a fresh destination and copy referenced
scientific assets with hash-based names or validated numeric-bundle paths.
Rewrite only package-relative links, preserve object/lifecycle/diagnostic bindings,
then recheck the original seal. Write sealed `display.json`, assets and the
portable display-runtime/inspector scripts; return hashes and a COMPLETE packaging
receipt. Existing destinations are rejected. The receipt explicitly says fresh
scene loading has not executed; packaging itself creates no Blender scene,
render, movie or ZIP.

## core/blender_scene.py::scene_main

<!-- implementation-sha256: 6b165273f6e8ddd0d22f993e2378472e425f61693a493c6725b256a8819f2364 -->

Parse a manifest and required output path. By default package its verified
display assets and print the receipt. With `--build`, require Blender's bpy and
a fresh .blend path, create the sibling .assets package, attach measured display
objects to the active scene, set its relative asset base and save the .blend
under the native lock. It writes display/scene files; building or packaging
alone is not a fresh-process replay or native scientific calculation.

## commands.py::audit_ic_main

<!-- implementation-sha256: e1ae87d696b912776a11184bf9c52a024c98afc0489923b5496a214bb43b36bd -->

Read a JSON object of validator keyword arguments and audit a linear bundle,
delivered IC realization or generator comparison. Decode requested complex
density/velocity predictions for realization checks and print finite JSON.
Return 0 for PASS, 2 for INCOMPLETE and 1 otherwise. This validates supplied
sources and measurements; it does not create the input realizations.

## commands.py::observe_stellar_catalogue_main

<!-- implementation-sha256: 8f5f59b5ae865b0edede86033b71ca878f0cf15a268c35427fe5bfcee4273fc2 -->

Read a saved observer-truth directory and JSON observation request, then call
the mock-observation exporter with its declared correlated errors, missingness
and cuts. Print the resulting publication record. It produces observations from
supplied stellar truth; it does not evolve stars or infer a new truth catalogue.

## commands.py::parse_products

<!-- implementation-sha256: 4aeb03ce4570c281ed06443223348d25c3ff36d17d24d80c81afc0e14ad6854c -->

Normalize a comma-separated string or iterable of product names by stripping
whitespace, lowercasing and preserving first occurrence order. Accept `all` or
`available` only as the sole selection; otherwise require nonempty names from
DESCRIPTIONS. Return the list or raise. This validates names only, without
checking physical applicability or launching products.

## commands.py::choose_products

<!-- implementation-sha256: 10f60f6fb361d0236211ad659d898a8cfb0001627b1d82bf073184d6b0b391ff -->

Print the available product descriptions and explain the strict `all` versus
ready-subset `available` choices. Repeatedly prompt for an explicit selection,
validate it through `parse_products` and display validation errors until valid.
Return selected names; no simulation, applicability plan or output is created.

## commands.py::audit_tree_integrity_main

<!-- implementation-sha256: b30ccd5f09d116c576313adbfc63086661da7328dd0b2ab894e15c9d1aa20a55 -->

Load explicit native nodes/links or import a requested Consistent Trees series,
then require frozen diagnostic thresholds, epoch and source bindings. Retain
original native links, measure disappearance/swap/jump/fragmentation flags and
publish a numeric audit with exact node IDs at the destination. Print status
and flag count. Alternative repairs belong in a separate proposal product.

## commands.py::propose_tree_repair_main

<!-- implementation-sha256: 1b8206224b238680f318b342679bf9de523eabd9e709b93fc23419ffa8dfd777 -->

Publish proposed alternative links against a supplied original audit, rationale
and thresholds, binding the request and command source hashes. Print proposal
status and diagnostic count. With `--verify`, instead verify the existing
destination proposal. It does not replace the original native tree or silently
promote proposed ancestry to accepted history.

## commands.py::measure_halo_shapes_main

<!-- implementation-sha256: d8c89eb41c6d0f511cd24770240047935a565249966b7d55272be250dbd61744 -->

Read a run and radial-shape request and publish its measured native-host
shape product; `--resume` verifies/reuses an unchanged publication. Print shell
statuses, profiles and scientific acceptance. Exit nonzero unless the result
status is MEASURED. The selected support sphere/FOF/SUBFIND convention remains
the request's measurement definition, not automatic global host binding.

## commands.py::run_research_acceptance_main

<!-- implementation-sha256: 21b2d1c08b04b773e87ae2276b01dff1c21a032ce11e5b54158f46efa7ce6835 -->

Read a provenance-bearing manifest and explicit evidence, require config/data/
source hash fields, and check the saved run through `acceptance_report`.
Write a fresh acceptance JSON and print its status/path. Return 0 only for
ACCEPTED, otherwise 1. It evaluates retained evidence without rerunning analysis
or overwriting the original manifest and products.

## commands.py::compare_finders_main

<!-- implementation-sha256: e5d63d67227a17558ab376bd3051f103fa8237e58b0bcccb64310c76db9df907 -->

Read exactly two requested saved catalogues and matching thresholds under the
run lock. Intake native GADGET, Rockstar or a sealed VELOCIraptor catalogue,
checking the selected epoch, then publish their measured disagreements at the
destination and print finite JSON. This compares existing catalogue results;
it does not launch the finders or erase their different mass/membership definitions.

## commands.py::build_identity_graph_main

<!-- implementation-sha256: 10dbf093abee4d31b2665ed5fd63216639d4db523bcfb234d65cfc7fc70647ee -->

Bind a request and source hashes, then publish an indexed exact-identity graph.
`--from-native` delegates native subhalo ownership intake; explicit graph mode
adds declared epochs, typed nodes, ancestry/parent-zoom/other edges and optional
SSP or hydro audits. Print node/edge counts and status. `--reuse-completed`
requests verified reuse; graph publication alone does not establish physical
ancestry beyond its supplied evidence.

## commands.py::run_event_cadence_main

<!-- implementation-sha256: 9142714a8dbeb53b3a289fab7abe96f4c7970d648b1b09f0ec43874abfec211c -->

Choose one mutually exclusive operation: merge/quantize requested output times,
plan followups, derive history candidates, execute a checkpoint branch or verify
a retained branch receipt. Nonverification modes require a destination; default
planning writes fresh JSON, while execute mode launches the requested branch.
Print the report. Planning, execution and verified output coverage remain
separate operations with their own input contracts.

## commands.py::trace_zoom_main

<!-- implementation-sha256: 608feea944b8459b3d9bcf37a45509979f483593de13f3ab823721a580475be6 -->

Read keyword arguments from JSON and trace a saved DM aperture back to parent
IC identities. With `--catalogue-host`, select the actual catalogue-host tracing
route instead of an arbitrary aperture. Print the delegated trace product/report;
selection and output locations come from the request. Tracing does not itself
establish a delivered refined realization or converged zoom evolution.

## commands.py::audit_native_handoff_main

<!-- implementation-sha256: 0ed0fa69cc839354486b991be45fa801b59b30caeddb3e69bfa4b12e31563ef8 -->

Normally load simulation YAML and revalidate an IC against the named native
generation report, publishing the handoff audit at its destination with explicit
audit memory. Alternatively verify a retained numerical-control receipt; that
mode rejects IC arguments and exits nonzero unless PASS. It checks actual
source/build/measurement bindings without silently substituting another IC or
executing a new native control.

## commands.py::stage_zoom_inputs_main

<!-- implementation-sha256: 940c5b827bf66091e5d31d2894fbc9dee952ffef1d7f4603bee14ff46cda2ca2 -->

Load a JSON staging request and delegate compatible species-transfer/MUSIC
input staging. `--resolution-refinement` selects bounded refinement staging
from an audited parent and saved-aperture budget. Print the resulting staged
product record. These request-driven input products are distinct from the
native zoom-generation command and its subsequent physical acceptance checks.

## commands.py::run_start_epoch_campaign_main

<!-- implementation-sha256: 3646578ef4bfcbb8b58a289d343068d6e2e2ccdfd6817e7056405830908b223b -->

Dispatch explicit start-epoch campaign build, verification, branch staging,
branch execution, phase audit, saved measurement, density-mode/analysis-product
measurement or comparison operations. Requests bind branch IDs, outputs,
executions and IC phase evidence; execute mode passes the supplied command
argument list to the owned branch runner. Print status and return 1 for FAIL,
2 for INCOMPLETE, otherwise 0. A prepared campaign does not imply its branches
have executed or converged.

## commands.py::ask_required

<!-- implementation-sha256: 5981611e8d4136d363824877c55636e35e5019bfe86cb6a5e452d48352e6641b -->

Prompt until the user enters a nonempty stripped string, printing a reminder
for blank input. Return the string. This blocks on terminal input and supplies
no default value, numeric conversion or file persistence.

## commands.py::required_number

<!-- implementation-sha256: 8da58cb586bcde8bf185e1474b2f1c0b36f403267eb0a93310812d219f39064d -->

Prompt through `ask_required` until a finite float lies inside the inclusive
minimum/maximum bounds. Print a range reminder for conversion, nonfinite or
out-of-range failures and return the accepted float. No configuration is
persisted by this helper.

## commands.py::required_integer

<!-- implementation-sha256: 29c62cc627bbe72cf0e0b882571002e7176bd8bfd2fa61f7afc3fa8d122b3475 -->

Prompt for a required value and delegate exact integer/range validation to
`integer_value`, retrying with the validator's error text. Return the accepted
integer inside the supplied bounds. It reads terminal input but does not save
answers or choose a numerical default.

## commands.py::required_choice

<!-- implementation-sha256: 0c0250ca14aa84dde06e44ab8a209640f56ef4fd5c9ada3b00627b55b671452d -->

Normalize the supplied option labels and each nonempty user response to
lowercase, then prompt until the response exactly matches a normalized option.
Return that lowercase string and print the allowed choices on failure. This
selects a label only; the caller applies and persists its scientific meaning.

## commands.py::audit_data_contract_main

<!-- implementation-sha256: 138f7650b58f7bc1c38fe0a39669703f41ae06c53ac578a00f3276ae6bcf3395 -->

Audit a saved native snapshot or a contracted in-run product and write fresh
JSON evidence. Product modes select HDF5/table/numeric/IC/image/scene/epoch
verifiers, with optional explicit HDF5 semantics or snapshot chunk checks.
Separate recovery modes replay a named native-set generation, IC transaction,
catalogue publication or supported product generation under the relevant lock;
these modes can restore publication files. Enforce mode compatibility and
in-run paths, retain failures before raising, and keep integrity status separate
from scientific acceptance. Recovery does not repeat the original IC transform
or rerun a finder/evolution.

## commands.py::audit_zoom_phases_main

<!-- implementation-sha256: 339d03eb38ffddaf8a73dcbfea11f39a0ce2dbccfcdc0084896f56fe4e8498b6 -->

Read parent/zoom IC and coarse-noise paths, destination and memory allocation,
then run the delivered density-phase comparison and print its retained report.
Return 0 only for PASS, otherwise 1. This audits the supplied 1LPT realization
pair and preserves FAIL evidence rather than asserting phase agreement from
matching seeds.

## commands.py::query_identity_graph_main

<!-- implementation-sha256: 11e87219ca28cfa0d97fa2ff873fd79d3cafa2c24a280583a1a34c0f8c261fc8 -->

Read a verified indexed graph and an explicit JSON query/arguments pair.
Allow only members-at-epoch, descendants, bound-owner, birth-parent or
parent-zoom queries; invoke that query and print finite JSON, converting array
results to lists. It reads indexed identity evidence without modifying the
graph or inferring missing links.

## commands.py::run_finder_campaign_main

<!-- implementation-sha256: 675310078d083ae59b7e1d06b782f346b56c1d8c87354e54f621f0d77ccdf68d -->

Read a saved-run finder request and delegate its native finder campaign,
printing the resulting record. `--dry-run` selects the campaign planning path;
ordinary execution may launch the requested finder stages and publish their
outputs. Finder availability and measured catalogue contents remain explicit
in the delegated result.

## commands.py::export_stellar_observer_main

<!-- implementation-sha256: f5291cd9eca693f8aa96f65ec4507a93682654c33dc8052b2b5b67f16a78b9d6 -->

Read an observer-export JSON, construct its explicit ObserverFrame and export
the requested saved stellar epoch from the run. Print the publication report.
This produces truth-space observer quantities for exact saved IDs; observational
noise, cuts and missingness belong to the separate observation request route.

## commands.py::audit_hydro_ssp_main

<!-- implementation-sha256: 6ffed33145bdb0fc648fc31329fd071340340574891d04d902e1e67037f96a51 -->

Publish an audit of native SSP birth masses, exact parent identities and
epochs from a request, or verify the existing audit at `--output`. Print status,
particle/support counts, legacy unavailable count and science acceptance.
It reads committed birth state; legacy missing fields remain unavailable
rather than becoming fabricated stellar ages or formed masses.

## commands.py::audit_hydro_events_main

<!-- implementation-sha256: 699cbcf7625fb0726c4820c1ed487dd5284132d4d29005bf1e3d926df8bbe60e -->

Audit committed native birth events between the requested saved epochs,
checking that the request hash stays unchanged across publication. With
`--verify`, read/replay the existing output audit instead. Print status,
event count and physical reconciliation. This examines original journal events;
it neither launches star formation nor converts an empty event list into births.

## commands.py::audit_hydro_history_main

<!-- implementation-sha256: 1e668808bfc5a56226d7a737699f60506136206659451a90618b899747433b67 -->

Publish or independently verify a multi-epoch native gas/star/BH accounting
audit. For publication, bind the request bytes before and after the audit;
print status, epoch/interval/issue counts and physics acceptance. Exit nonzero
unless the accounting status is PASS. Audit success is scoped to the measured
census/journals and tolerances, not every physical model or resolution.

## commands.py::select_merger_tree_main

<!-- implementation-sha256: 8f8c678ca155259be24f9562b232299bdc5827df8b7e656377e3066c2854368c -->

Select an exact final halo ID from the saved run and analysis bundle, then
export its descendant-linked ancestry to the required output and print the
result path. Selection uses retained catalogues/history rather than synthesizing
progenitors or treating a catalogue row as a universal object identity.

## commands.py::void_particle_history_main

<!-- implementation-sha256: eadf93b757e8cc7ac24bb1c5f0bacca0b94e6f782865f1fe15c9ec1ba6d0740c -->

Measure exact DM-particle overlap between saved void-centred apertures using
the run, void bundle, explicit radius in Mpc/h and memory budget. Publish the
requested history output and print its result. Aperture overlap is a measured
material diagnostic, not by itself persistent physical void identity.

## commands.py::filament_flythrough_main

<!-- implementation-sha256: e25c8960a6b241614b171f6512f486e72cd0dc1367d2764901e3db32a23fe2bd -->

Select one saved snapshot and native host row, pass the declared grid/path/
sampling/render budgets to the connected density-ridge flythrough renderer,
and print its output. This can encode a camera film through one saved density
field; camera motion is not cosmic time evolution or a DisPerSE topology
measurement. It does not launch another particle simulation.

## commands.py::results_gallery_main

<!-- implementation-sha256: 0199423ff275705a6a943435647ba08e94db0c6795629e7a1126f847c7d18ad3 -->

Create an offline gallery from an explicitly selected-results JSON, resolving
relative inputs against that request's directory and checking its hash after
publication. Creation requires `--output`; `--verify` instead takes a gallery
directory and rejects an output argument. Print the created location or
verification JSON. This indexes supplied outputs rather than rerunning their
scientific producers.

## commands.py::merger_tree_map_main

<!-- implementation-sha256: a102b7c11ff6e636ea6c731096fd4a54ef2886e03c45540520127af790187425 -->

Join a run's verified ancestry analysis to a saved map bundle using an exact
map key and explicit memory allocation. Publish the connected tree/map display
at the selected output and print its result. The underlying density arrays,
tree IDs and epochs remain the source measurements.

## commands.py::metallicity_maps_main

<!-- implementation-sha256: 300fbe0813e3e65a72c34dcf58326a92aeeb47d9e567fab1dac1af56a037ab08 -->

Add gas metallicity projections to a selected saved structure-map bundle.
Pass the source map key, run, destination and memory budget to the producer
and print its JSON record. It measures saved gas fields in the map's geometry;
it does not infer metallicity from image colour or evolve chemical enrichment.

## commands.py::halo_fingerprint_main

<!-- implementation-sha256: 55d20fb7e6f2c3d6f22616e47e8f8a03641134634bdc38588c6ca375c9d4d10f -->

Compare a selected host between two saved runs/snapshots through reciprocal
exact-ID membership matching. Pass shared-count/fraction, grid, epoch tolerance
and memory controls to the fingerprint producer; publish and print the result.
A descriptive matched-host display is not a convergence or causal comparison
unless the broader experiment supplies that evidence.

## commands.py::void_explorer_main

<!-- implementation-sha256: 560ce5ff3299e017a276026b87fd6ee7a5eea750f46a741141ad6f432239f415 -->

Read saved periodic DM epochs and build the requested underdensity explorer
using explicit mesh/smoothing/threshold/matching and sampling/movie budgets.
Print the published output. The producer can create diagnostic films; threshold
regions and heuristic matches do not automatically establish continuous physical
void identity or a new cosmological evolution.

## commands.py::void_boundary_history_main

<!-- implementation-sha256: 53ab1b8d2f187c142c6821237728ab5c5dd29517c46a954e96062c44ec36b068 -->

Use a saved run and void bundle to measure periodic density-excursion boundaries
and adjacent-epoch mesh overlap within the memory allocation. Publish the history
output and print its result. This quantifies Eulerian region boundaries/overlap,
which is distinct from exact particle-cohort history or a proven void genealogy.

## commands.py::halo_anatomy_main

<!-- implementation-sha256: 57357d7277a3b0beb7f9fba90f911fcd206d39225faf1a78cf8fafd843abc2e6 -->

Select a saved snapshot and host row and build its interactive cutaway and
radial-profile output with the requested grid/layer/memory settings. Print the
published location. It displays measured saved particles and profiles, rather
than generating a new halo or adding unresolved internal structure.

## commands.py::parameter_explorer_main

<!-- implementation-sha256: 86063a3da4c7163e07417ae7dcadc6cdb9da88e114ea9d6aa8a596a5830bbe0a -->

Read an explicit saved-run comparison request, build its offline parameter
explorer at the required destination, and reject a request hash change during
export. Print the result path. It organizes the supplied runs and measurements;
it does not interpolate an uncomputed cosmology or launch a parameter scan.

## commands.py::resolution_reveal_main

<!-- implementation-sha256: 9cf6e34c421608f8d84a1b87dc7b11adfe160971c5eeb1ea0678b0abc5d28288 -->

Compare one selected saved projection from each of two run/map bundles.
Require explicit snapshots, common map key, epoch tolerance and memory budget;
publish the reveal display and print the result. It presents the selected
resolution contrast without automatically proving phase matching or numerical
convergence.

## commands.py::cosmic_flythrough_main

<!-- implementation-sha256: 28eef7faaa073309dfb8f528ec5362f7958e8b7a4bb01e007c76e00d26b3ff5f -->

Render a camera path through one saved snapshot toward a selected catalogue
host, using an explicit start, optional waypoints and frame/image/FOV/sampling
budgets. Publish and print the output. The command may encode a film; it changes
the view through existing data, not the simulated epoch or particle dynamics.

## commands.py::halo_story_main

<!-- implementation-sha256: fd3c86d3942ff4b59bfb2cd899dd9178d0021e341a4a9d2e7081e0476239e737 -->

Connect a selected positioned tree, recorded density contract and original
density directory to a synchronized labelled map/tree/film output. Pass the
run binding and explicit playback rate/hold duration and print the published
result. It uses saved source geometry and ancestry rather than fabricating
undetected early progenitors.

## commands.py::lens_map_main

<!-- implementation-sha256: d587b8b2887fcd93d3d1e6f65e4b3e85e6869b5fe91af91494b03a028542242b -->

Compute a thin-lens convergence/shear display from a selected saved mass map.
Require the snapshot/map key, three distances in kpc and background surface
mass density in Msun/kpc², then publish and print the output. These supplied
geometry/calibration inputs are not a fitted cosmology, lens posterior or
new ray-traced simulation.

## commands.py::stellar_stream_movie_main

<!-- implementation-sha256: 2129fdea333038ec3335bb2014aaf90cb3c1e0cef9a0835967ce891db42ad33b -->

Animate an existing exact-ID stellar candidate history from its run/bundle.
Pass optional fixed centre and width in comoving kpc, particle bound, frame rate
and seconds per saved snapshot; print the published output. This encodes saved
candidate trajectories, without establishing tidal origin from appearance.

## commands.py::planetarium_movie_main

<!-- implementation-sha256: e7466ef807131d9c9030e43481e3866d56be9aa75f1d880b4b24b586a4a2c7b1 -->

Render the requested saved epochs/field into a fixed-scale dome output using
the supplied observer, raster resolution, tilt, sampling and timing budgets.
Print the resulting publication. This command can produce a dome movie from
saved data; the projection is not by itself a cosmological light cone or
an emitted-light calculation.

## commands.py::halo_collision_movie_main

<!-- implementation-sha256: daf71d2a50d53babd6b160dcb7bd349f9a35f97be2d4212dc127a2d05729237f -->

Use saved ancestry and map bundles to animate two linked progenitor branches
near a selected merger, optionally choosing an exact descendant ID. Pass the
map key and playback/memory settings and print the output. The film displays
recorded geometry; physical collision interpretation still needs the original
trajectory, source and resolution evidence.

## commands.py::halo_formation_movie_main

<!-- implementation-sha256: 1123a921d887b5430fdc9ef59b2638d170656d95f27496d5ef4aae2f05b7d1ba -->

Animate measured saved halo ancestry beside its associated density-map series,
using an exact map key and explicit timing/memory settings. Publish and print
the result. It reads existing tree/map products and does not rerun formation
physics or invent missing progenitor positions.

## commands.py::allsky_map_main

<!-- implementation-sha256: f2ef6d4d6da21ada209a169831609ba1d66a950517b4aa7ab935a1a4ddbafe94 -->

Select DM or gas from one saved epoch and render an observer-centred sky with
the requested equal-solid-angle grid and bounded sample. The observer is in
comoving Mpc/h, defaulting to box centre. Print the published map path; its
scalar is represented matter mass per steradian, not observed stellar flux
or a time-evolving light cone.

## commands.py::velocity_flow_main

<!-- implementation-sha256: 417a4ac9cf064624fef7016b7e1d0c19bffb9fcb16519e73a708c97a0555452b -->

Map one saved DM snapshot's density and mass-weighted peculiar-velocity flow
in the selected xy/xz/yz projection, applying grid, particle-support and optional
slice controls. Publish at `--output` and print the result. It measures saved
velocities; displayed arrows do not integrate future particle motion.

## commands.py::particle_origin_movie_main

<!-- implementation-sha256: 32a384cc93d2b39d2b0a75de0363ef223072c7e1ba01227d9029c3166bcffaa2 -->

Choose the exact DM members of a saved anchor host and trace those identities
through saved epochs, with explicit sample and playback bounds. Publish the
origin-history film/output and print its result. This follows a selected
material cohort; it neither changes membership to improve appearance nor
creates unsaved early states.

## commands.py::multifield_movie_main

<!-- implementation-sha256: 36b38280b7613785c5448d79ca384ad8cff629a3fafc26951b558bbedd287c7b -->

Encode synchronized panels from two to four explicitly repeated saved-map keys.
Pass the recorded bundle, optional source run, destination and playback/memory
settings; print the result. Field availability and epoch coverage come from
the map contracts, rather than copying a neighbouring field into a missing panel.

## commands.py::timeline_wall_main

<!-- implementation-sha256: 84f0a18e49ae283aa35d3345ac0147d1d6e4e21f0611d539ba5e476227926914 -->

Build a static fixed-scale cosmic timeline from one exact map key in a saved
bundle, with an explicit panel cap, memory allocation and optional source run.
Publish and print the output. It lays out recorded epochs rather than adding
interpolated simulations or changing each panel's physical scale independently.

## commands.py::comparison_movie_main

<!-- implementation-sha256: 01ced9f9b3d6dd884dbd3b0b5bc894d073803fcb88601b1cbf63c0cb611f9fe7 -->

Encode an already published comparison-map series selected by its exact
summary key. Supply optional source-run paths only when provenance cannot
resolve them, plus explicit playback and memory settings; print the output.
This consumes measured comparison products, not an implicit phase match or
new pair of native simulations.

## commands.py::native_source_overlay_main

<!-- implementation-sha256: 63b458c53b94689358eb0816cc137a39c6a1c18cb37365b5d123d2ac8936ac98 -->

Without a source root, verify and describe a sealed native full-file overlay.
Explicit installation additionally requires a stable prior-file hash mapping
and writes the overlay into the selected source root, then prints changed-file
count and build-execution status. Installation changes source files but does
not compile, test or run the native solver.

## commands.py::native_source_events_main

<!-- implementation-sha256: 044a83a6046dbe0735c7fdc186725ad3dae344e900266bd1fc43c49c6d51617b -->

Read a hash-stable physical-source event request and publish the linked
journal/event-cadence product at the selected output. Recheck the request after
publication and print event count and cadence acceptance. This connects retained
source journals to measured windows; it does not execute those source operators.

## commands.py::prepare_source_controls_main

<!-- implementation-sha256: 4cc3155a407357479fce8d9a69b4876525bcba41481f9d2f45e1a3406e4fd808 -->

Read and hash-check an explicit native-source control request, then write its
flat hydro/stellar/BH IC and build-profile products under the selected prefix.
Recheck request stability and print file/execution metadata. Preparation writes
inputs only; build and evolution remain separate requested actions.

## commands.py::black_hole_environment_main

<!-- implementation-sha256: 88f892630933738941416cede8a30881916347070ec281ce53a4148e264a8720 -->

Read a stable request and measure selected BH host binding, wandering and
DM radial histories from sealed native data. Publish the environment history,
recheck request bytes and print schema, epoch count and physics acceptance.
The command analyzes saved history rather than generating accretion, host
motion or a feedback intervention.

## commands.py::hydro_comparison_main

<!-- implementation-sha256: 3290a335a4881d4fff09fe0b60a8b7792b183fb553de2aa575cfae8a9111fa10 -->

With `--design`, publish a fresh sealed matched-physics configuration design.
Otherwise measure the requested saved radial hydro comparison. Both modes
hash-check the request across work and print schema/physics-acceptance metadata.
Neither mode launches evolution, and a valid design is not an executed matched
comparison.

## commands.py::compile_stellar_model_main

<!-- implementation-sha256: 4fffa2517b86b1f2f5cb407eba7c79882cf7c37a7cb18bfaa169818bc1b303d7 -->

Compile explicit pinned yield/lifetime inputs into the delayed-return table,
or verify an existing compiled model with `--verify`. `--preflight-only` instead
writes source-row/IMF/age-support diagnostics and returns nonzero for failure;
verification and preflight are mutually exclusive. Bind request bytes across
work and print implementation/validation/table metadata. This compiles a data
model, not a native executable or stellar evolution run.

## commands.py::measure_stream_heating_main

<!-- implementation-sha256: 6e33a8d56a7ac41e4266977cd236e320b1a249be95060b5336d2dc786b6f3f19 -->

Load a JSON request and publish the numerical heating calibration from its
paired native granular/smooth controls at the requested output. Print the
reported status. This delegates analysis of the specified controls; the wrapper
does not launch new evolutions or itself convert a reported non-PASS into
an explicit exit code.

## commands.py::halo_structure_main

<!-- implementation-sha256: af85dad3d823b9a0a6cc037c5a3b599ee7cd0c95d9afe717780e45161e509d0c -->

Publish complete native component profiles and validated proper host fields
from a request, binding its hash and command source. `--verify` instead reloads
and verifies the existing output; an optional figure prefix requests profile/
validation plots. Print measurement, field-validation and science statuses.
The command measures or replays saved particles, not a new halo evolution.

## commands.py::environment_main

<!-- implementation-sha256: 8c24c3763c1d895835acaad009f5fca94e029fea512847bac8a9a0223323cae6 -->

Publish or verify a native periodic field/web/splashback product; `--history`
selects the graph-bound multi-epoch producer/verifier. Optional field plots
require a single-epoch product and explicit smoothing index, with optional
halo selection. Bind the request on publication and print status/count metadata.
Scientific acceptance and local feature support remain the producer's recorded
states, not consequences of a completed renderer.

## commands.py::audit_zoom_regions_main

<!-- implementation-sha256: af542cee63cb5f50b025a27b328ee3cc4ece2c6e7489ecc4d578e6abf11bb9b2 -->

Audit requested saved/tracked science apertures and sampling, or explicitly
compare observable products against a sealed saved-aperture resolution budget.
Comparison mode verifies the budget seal/provenance, selects exactly one audited
epoch and publishes under the run lock. Print the result and exit nonzero unless
PASS. This evaluates original selections and evidence; it does not refine ICs
or imply convergence outside the measured budget.

## commands.py::run_velociraptor_main

<!-- implementation-sha256: 910f5fea50c79584379760eb90667fdca7eec9a58482bd50dabaa73596579339 -->

Read a saved-run JSON request and execute the pinned native VELOCIraptor
finder at the required destination, passing the producer's exact-ID transport
settings. Print status and resulting object count. This launches a finder on
existing particles; successful execution can legitimately yield an empty catalogue.

## commands.py::run_zoom_main

<!-- implementation-sha256: 67f12479f8fe6e29337718ecf6cdbd11f22baa6fbfd53ca01cb9b334289c04e8 -->

Run the supplied MUSIC binary on the selected staged configuration and publish
its structurally audited zoom ICs. Require a native-validation input and either
both parent IC/noise paths or explicit `--structural-only`; separate audit/native
memory, disk and wall budgets remain visible. Print the generation report.
This launches real IC generation, but structural-only operation and incomplete
phase/physical gates never confer scientific acceptance.

## commands.py::cosmography_main

<!-- implementation-sha256: ab51da259e516017e4701121e080d61cf43e22526f8f8ea262026cc083c52bd0 -->

Forward to the selected cosmography CLI, which reads a request JSON, resolves
its relative inputs against that file and executes the declared expansion/
observer campaign into a required output. Return the delegate's result; it
prints the campaign record. Physical likelihoods, observations and resources
must come from that request rather than an automatic box-run inference.

## commands.py::cosmography_report_main

<!-- implementation-sha256: c1b5e46d83b47f59c6fc72fb9a5df141fb4f4e44cf47fc5d8980211a949b7363 -->

Forward to the post-fit comparison exporter. Its CLI requires a JSON product
with role `post_fit_evaluation` and an output directory, then publishes comparison
figures and prints their locations. It renders frozen fit/evaluation results;
it does not execute a new fit or sample a posterior.

## commands.py::streams_main

<!-- implementation-sha256: 37c8bc335cc3629544cc1e14fd7ea226c63cb59113b36f39d556351672591937 -->

Forward to the stream-workflow CLI, which loads a request, resolves inputs
relative to that request and runs its selected catalogue/stripping/geometry/
action or inference stages into the required output. Return the delegate's
result and let it print finite JSON. Which scientific work executes depends
on the explicit stage graph and its original input/acceptance contracts.

## commands.py::scientific_figures_main

<!-- implementation-sha256: a76f8ae1fc1ea9f846b3a9b3d490c6ca848b0a54aeda3cc0cebe545ed1476994 -->

Forward to the registered scientific-figure CLI. Rendering requires a measured
array request, output and memory budget; optional organized layout separates
formats by family. A mutually exclusive registry-export mode only writes the
catalogue and returns success without rendering. The wrapper creates neither
missing measurements nor accepted physical results from a recipe name.

## commands.py::verify_main

<!-- implementation-sha256: 12f55164fd28681297c5b4b99e011130e5dc9239cfc0f41ca13c2c7178824674 -->

Forward to the verification dispatcher. It runs unittest selections or explicit
retained domain groups, with group mode allowing only the supported verbosity
switches, and exits according to test success. Suites can use fixtures, mocks
and selected native controls; a passing test count does not imply every backend,
branch or scientific regime executed.

## core/cli.py::hmf_main

<!-- implementation-sha256: 7d3604978f41e7d77a878b38caa6113b6b4e8bc5667e9fb82284920a9dcf31aa -->

Forward the HMF/HMF-scan CLI to the linear prediction campaign driver. Its
request selects cosmology, epochs, masses/formulas and optional sampling; it
can run CLASS/AxiCLASS or reuse matching saved linear epochs, then publish
tables, plots and candidate records. Return the delegate's completion code.
These formula abundances are screening predictions, not measured GADGET counts
or automatically launched particle simulations.

## core/cli.py::hmf_learning_main

<!-- implementation-sha256: 3dc35e83b2a0e3279aaa33eefaf124cadec1e53c9cec67924ec1b01ab13b1698 -->

Forward to the measured-HMF learning CLI for native catalogue export, dataset
construction, training or prediction operations. The workflow keeps simulation
counts/volumes and cosmology-level splits distinct from analytic screening
predictions, and writes each operation's requested products. Return the
underlying result; this wrapper supplies neither training targets nor fitted
coefficients by itself.

## core/cli.py::render_scene_main

<!-- implementation-sha256: fb8e8fcca495eb7285d2ab84537a12773bb66a1955eb00c092e01c42729aaa1b -->

Forward to the sealed-scene renderer CLI. It verifies the saved scene, requires
explicit projection, image/frame and resource controls, and writes a prepared
request/launch record to a fresh output. Default execution launches the selected
Blender binary for frames; `--prepare-only` stops at preparation. Return the
underlying result. It renders an existing scene rather than evolving particles.

## core/cli.py::stream_products_main

<!-- implementation-sha256: 2b5da5b03d4b2f398943fd283d38ba17d4b2397d8d5bf89b24cba8c650f28585 -->

Read a measured stream product and select either figure export or Blender
inspector-asset export, with explicit output and memory. Invoke the corresponding
producer and print its JSON-safe result. Inspector assets are data for a later
scene loader; this command does not fit a new stream or silently launch native
evolution or a Blender render.

## core/cosmography.py::cosmography_main

<!-- implementation-sha256: 6513fabb2146adb6c5f71852318ff337d612eb78f7ad38c1ebf43e13e572dde8 -->

Read the CLI request JSON and dispatch its selected expansion-observer campaign stages. Resolve relative campaign inputs against the request file’s parent and pass the original request path for provenance. Write the campaign’s requested products beneath --output and print its result as JSON; this adapter returns no explicit status value.

## core/cosmography.py::cosmography_report_main

<!-- implementation-sha256: 58b1ad2584cebfa6d2d9d1ac18b9462471a8fa47f24d310fc4718b495bdc41ed -->

Render figures from an already frozen comparison JSON whose role must be post_fit_evaluation. Pass that product to publish_comparison_figures with the requested output directory and print the figure records. This reporting operation consumes prior inference results; it does not run a new cosmological fit.

## core/cosmology.py::doctor

<!-- implementation-sha256: 2e1ac78aa572535b5775309aac57083339521112f21f6a8f35147981433598c1 -->

Print the discovered CLASS and AxiCLASS executable paths, or MISSING for each unavailable engine. Discovery uses executable(); this diagnostic returns no structured result and does not launch a solver, check its numerical outputs or demonstrate that a native calculation works.

## core/gadget_hmf_data.py::export_gadget_measurements

<!-- implementation-sha256: 06ea3ad105b0dcbdb7e0fc2be405fbb0384020d4dc786d4338808b6ce502f8cf -->

Export measured host-halo counts from a dark-matter-only, non-zoom native GADGET run. Validate snapshot box, particle census, uniform particle mass and complete matching-epoch group catalogues; apply the configured particle-count cut and histogram catalogue masses in Msun/h, retaining zero-count bins. In a fresh output write hmf.csv and metadata.json with volume in (Mpc/h)^3, source/config/catalogue hashes and the CSV digest; return metadata. Snapshot headers and file sizes are recorded rather than full particle-position hashes. The cut is an eligibility rule, not a demonstrated resolution limit.

## core/hmf_campaign.py::write_csv

<!-- implementation-sha256: 00415070a591efb0d148b975ab0e2f8b51c9b5c8cc8d91a3a2161f673ce04da3 -->

Write nonempty dictionary rows as a UTF-8 CSV, taking column order from the first row. The caller supplies an existing parent directory. Opening in write mode replaces an existing file; empty rows raise ValueError. This low-level writer neither validates scientific units nor creates provenance receipts.

## core/hmf_campaign.py::validate_request

<!-- implementation-sha256: 4458d052ceca66e6f1964ec91a6b55572f0007fcd5902f36e72306f455dc8e32 -->

Check an HMF screening request before solver work: supported LCDM/EDE model, positive finite declared cosmology values, no supported massive-neutrino branch, no protected overrides through solver extras, distinct nonnegative epochs, ordered positive mass edges and known unique fitting formulas. Validate k coverage, timeout and disagreement controls, then delegate EDE parameter checks. Return None or raise; no files or predictions are produced.

## core/hmf_campaign.py::predict_sample

<!-- implementation-sha256: 2dac19dee0cc7ce283cd922142ae86e5c65152aed146a3afbd852ad69d21cf1f -->

Evaluate one validated cosmology in a fresh sample directory. Save request.json, obtain a separate linear spectrum for each requested epoch from CLASS/AxiCLASS or an exact verified saved bundle, and integrate the selected HMF formulas over mass bins. Write predictions.csv and return its rows: masses in Msun/h, abundance per dex in h^3/Mpc^3, sigma, slope and coverage/calibration flags. EDE rows remain labelled extrapolation; these are formula predictions rather than measured simulation counts.

## core/hmf_campaign.py::sampled_requests

<!-- implementation-sha256: 13b69a0c9002543528f877b27b48d8a889e0b89f711353ac83ec7d6aac522211 -->

Construct independent request dictionaries from a screening specification. Support seeded Latin-hypercube or uniform draws over allowed cosmology/EDE bounds, or an explicit finite Cartesian grid bounded by max_samples. Grid count must match its product if supplied. Reject incompatible model paths, malformed ranges and invalid resulting requests; return the list without running solvers or writing files.

## core/hmf_campaign.py::plot_campaign

<!-- implementation-sha256: fe8281d47815b972145d36870788d97cca21c22618061de3edc005545d882742 -->

Use headless Matplotlib to save one hmf_zNNN.png per sorted redshift in the supplied output directory. Plot positive predicted abundances only where both k coverage and the formula’s listed calibration range permit support, with logarithmic mass [Msun/h] and abundance [(h/Mpc)^3 per dex] axes. Close each figure; these screening plots show formula spread, with EDE explicitly labelled extrapolation.

## core/hmf_campaign.py::run_campaign

<!-- implementation-sha256: 6a2e9226d1a8bd0aca6a246f7ce9affb6386d6cb9c21b1ff39404fb286cc6b34 -->

Sample and validate requests, then run predict_sample in a bounded process pool with separate sample directories under a fresh campaign output. Preserve individual failures and aggregate sorted predictions, per-epoch PNGs, disagreement scores, candidate configuration records and report.json. Return COMPLETE, PARTIAL or FAILED in the report according to available rows and failed samples. Candidate selection proposes later GADGET experiments; it neither launches particle simulations nor manufactures learning targets, and scientific acceptance remains unset.

## core/hmf_campaign.py::hmf_main

<!-- implementation-sha256: ac09b9f64746d19dab06d2ca4513edd22d225dce92ea05eb97df5473d0b31aeb -->

Parse --spec/--output plus worker, solver-binary and saved-linear-bundle overrides, load the JSON request and invoke run_campaign. Print the report without its potentially lengthy candidates list. Return zero only for COMPLETE, otherwise one; validation and uncaught I/O failures propagate. Use this CLI for linear HMF screening, with separate commands for native measurements and training.

## core/hmf_learning.py::cosmology_features

<!-- implementation-sha256: 33bf0b2233ffbdc79d2c2b8b357c552ca563ad9a90ef803d87c934ebc429b154 -->

Extract the declared LCDM/EDE cosmology into numeric model features, including log10(A_s), H0 and physical density parameters; EDE-specific fields are zero for LCDM. Validate supported models and key finite/positive quantities. Return the feature dictionary and a SHA-derived grouping identity built from declared physical cosmology, including relevant defaults/extras. The grouping prevents one cosmology leaking across dataset splits, even when some identifying parameters are excluded from the learned feature set.

## core/hmf_learning.py::build_dataset

<!-- implementation-sha256: 25846c99cdd7a4808ebde0d94cb282c181d831a7bf7933aea4a2044a3e132730 -->

Combine explicitly selected native or portable simulation-count sources into a fresh dataset directory. Reject duplicate measurements, incompatible finder/physics/mass conventions and inconsistent optional features; apply the declared resolution-selection policy. Keep each cosmology wholly within one train/validation/test split, requiring at least five groups for automatic splitting. Optionally enrich rows with linear-theory features. Write dataset.csv, metadata.json with hashes, split counts and units, and resolution_comparison.csv when applicable; return metadata. Exposure is volume times logarithmic bin width, not an analytic HMF target.

## core/hmf_learning.py::powers_for

<!-- implementation-sha256: 804fd46096c857aceb058daab5ab7399568cc46a7228ac355a448a54c351441c -->

Return polynomial feature-index tuples up to the requested degree, beginning with an empty tuple for the intercept. combinations_with_replacement includes repeated powers and cross terms. This describes basis structure only; it does not standardize features, fit coefficients or write files.

## core/hmf_learning.py::design

<!-- implementation-sha256: 59c4fefe604a98b81388cd41695173aab886ac4d768a2c18a3dfdaed9981f84c -->

Build a NumPy polynomial design matrix from a two-dimensional feature array and the tuples returned by powers_for. Each column multiplies the indexed features; the empty tuple supplies an all-ones intercept. Return the column stack. Scaling, feature selection and validity checks belong to the training or prediction caller.

## core/hmf_learning.py::fit_poisson

<!-- implementation-sha256: 02b07451936d6e1c2e7f5fb0a91abca60e8ab03e7ad9c3c17df2f2ca721dd6ec -->

Fit log-density coefficients for count data with log(exposure) as an offset and a ridge penalty excluding the intercept. Initialize the intercept from total count divided by total exposure, reject zero total counts, and use SciPy trust-exact optimization with analytic gradient/Hessian and overflow guards. Return finite converged coefficients or raise. Aligned numeric inputs and exposure conventions are caller responsibilities; this objective does not include cosmic covariance.

## core/hmf_learning.py::poisson_deviance

<!-- implementation-sha256: d96d4827abd70676dcfc141315c809e848609cad162976841507ac8ab0ba3600 -->

Return twice the mean per-bin Poisson deviance contribution: mu−n+n log(n/mu), with the logarithmic term omitted for zero counts and small negative roundoff clipped away. The result is dimensionless and assumes valid compatible counts and positive expected counts supplied by the caller. No fit, covariance estimate or file output occurs.

## core/hmf_learning.py::train_dataset

<!-- implementation-sha256: 6b7ea62020c0d47409670de9811ba33c3fe2a2e9db05cc2bfc99312ed973a54b -->

Verify a hashed measured dataset and its three nonempty splits, then fit a degree-one or degree-two polynomial log-density model. Standardize using training data only, remove constant active dimensions, fit four ridge choices on training counts and select by validation deviance; report test performance separately. Write model.json and evaluation.csv to a fresh directory and return the model. Predictions approximate bin-averaged densities through bin-centre features; rectangular training bounds and positive rates do not certify physical accuracy or full count covariance.

## core/hmf_learning.py::predict_density

<!-- implementation-sha256: 79b7fc24147befed7817a49cfa717b5a6ce147af79dd9d553ff3d526228f54ae -->

Order numeric feature rows by a saved model’s feature list, reject nonfinite values and flag violations of its training bounding box, including constant features. Reject extrapolation unless explicitly allowed. Apply stored training scales, active dimensions, polynomial terms and coefficients, guard overflow and return positive density predictions plus the outside-domain mask. It performs no I/O; passing a rectangular bound does not prove joint training coverage.

## core/hmf_learning.py::learning_main

<!-- implementation-sha256: 0502eddc39f100764ec3e12cd7e3a0a4b6e76846db803bd213f66784eb9ad99c -->

Dispatch measured-HMF CLI operations: export native GADGET catalogue counts, build a source-bound dataset, train a polynomial Poisson baseline, or predict from a model and feature CSV. Write each operation’s requested products and print its summary; prediction CSVs retain input fields and add abundance in h^3/Mpc^3 per dex plus extrapolation flags. Multi-run export returns failure unless its report is COMPLETE; other successful branches return zero and validation errors propagate.

## core/observations.py::ExpansionHistory

<!-- implementation-sha256: e7f8f1ff1bf6cd099bde40f248280a60e1a0483886daea4f1bb3f4e8406030d4 -->

Represent a sampled solver background H(z) in km/s/Mpc with curvature and an optional absolute-time anchor. Require at least four ordered samples beginning at z=0 and preserve read-only copies. PCHIP interpolates H without extrapolation; quadrature computes chi=c integral dz/H in Mpc and lookback time in Gyr. Curvature supplies transverse distances, with angular and luminosity distances derived from them. Absolute cosmic time requires the high-z anchor. from_class converts CLASS H from 1/Mpc and records the source hash; construction alone runs no solver.

## core/observations.py::export_background_expansion

<!-- implementation-sha256: 74428e923053a58fdec051f65ced0a07113fd0057140ea3bea87ce01e134458b -->

Return a serializable privileged_expansion_truth dictionary evaluated on the history’s saved redshift grid: H and H0 in km/s/Mpc, radial/luminosity distances in physical Mpc without h, lookback time in Gyr, curvature, source hash and equation labels. Include cosmic time only when an absolute-time anchor exists. This returns truth for later controlled comparisons; it does not write a file or fit observed data.

## core/observations.py::Observer

<!-- implementation-sha256: f8a7e07024593a85772d4114d5937bfaba7d83f253319c8b5f1dd9523e784e1e -->

Store an observer identifier, three-component position in Mpc and peculiar velocity in km/s, environment and selection rule. Validate/copy finite vector inputs and reject velocities at or above the speed of light. This record does not select an observer from particles, evolve a trajectory or produce observations by itself.

## core/observations.py::doppler_factor

<!-- implementation-sha256: 5df9b315f42d440a2cf489917c33db667a0f97ea315997198bd88bd532b2e755 -->

Compute the relativistic line-of-sight factor (1+v dot n/c)/sqrt(1−|v|^2/c^2) for peculiar velocities in km/s and caller-supplied sightline directions. Reject superluminal velocities and return the broadcast NumPy result. Direction normalization and geometric conventions are supplied by the caller; this factor alone is not a complete cosmological redshift.

## core/observations.py::fit_distance_redshift_relation

<!-- implementation-sha256: 4e6066458fed8865a2393a928c7b6cb2c2400267bd66d4b0f23c86249eb9d7fd -->

Fit ln(H0) by generalized least squares to selected distance/redshift observations within an explicitly supported cosmographic redshift range. Apply independently supplied flow corrections and combine log-distance errors, redshift/flow variance and optional calibration covariance. Return a frozen local-observer result with object IDs, nuisance/selection metadata, H0 in km/s/Mpc and its transformed 68% interval. This uses the declared q0/j0 shape and observations, rather than reading simulation truth as a fitted target.

## core/observations.py::time_delay_distance

<!-- implementation-sha256: 542f29acdf9d87eefccb9f0d6b7236fa57286caca8c86ff975e444698ad06790 -->

Return D_dt=(1+z_lens) D_l D_s/D_ls as a scalar in physical Mpc using the supplied expansion history’s angular-diameter distances. Endpoint ordering and background coverage are enforced by those distance methods. This geometry calculation performs no lens fit, ray tracing or file output.

## core/observations.py::fermat_delay_days

<!-- implementation-sha256: a8183b12142cc3abbc953f40b0c6005ca56e3ff43c14766de8c4bccc5d47a996 -->

Convert a Fermat-potential difference in arcsec^2 and time-delay distance in Mpc to arrival-time difference in days, multiplying by the squared arcsec-to-radian factor and distance/light-speed/time conversions. Return a NumPy array or scalar-like result. Units and physical validity of the supplied inputs are caller responsibilities.

## core/observations.py::construct_mass_planes

<!-- implementation-sha256: 8524d8c78869b77d045ccbc76536a9a02f30d2bde4014dd60d07312f43ec6b4d -->

Project a nonempty list of simulated mass components through _particle_plane and sort the resulting planes by redshift. Return lens_mass_truth with source redshift and an explicit convention: proper Mpc without h, angular potential in arcsec^2 and source-normalized deflections for isolated softened particles. Resolution remains UNRESOLVED_NOT_EXECUTED; this construction neither fits lens observations nor establishes numerical convergence.

## core/observations.py::render_lensed_observations

<!-- implementation-sha256: a72ac4bec4b29c13e1379dc74f461ee082533b75fb66f3100169d3c8d83596d7 -->

Use pinned lenstronomy to ray-trace supplied physical mass planes and source light, requiring at least two time-delay images. With the declared seed, simulate correlated delay noise in days, noisy astrometry in arcsec and a PSF/exposure-aware image with background plus model-dependent pixel variance. Return canonical estimator observations, instrument/selection data and backend revision, preserving full delay covariance and optional kinematic/line-of-sight constraints for fitting. This produces synthetic observations in memory, not a posterior or scientific acceptance.

## core/observations.py::fit_lenstronomy_model

<!-- implementation-sha256: 37d4157d175352d797cebc700153344e7cb1bd0b10b9d04d99f37ca0f1c77ad3 -->

Fit a declared lens family to observation-only image, astrometry and delay inputs with pinned lenstronomy, requiring sampled D_dt and a delay likelihood. Add explicitly configured correlated-delay/kinematic likelihoods and line-of-sight priors when present; restore NumPy’s prior random state after seeded sampling. Require enough positive-distance posterior samples, optionally evaluate kappa profiles, and transform D_dt samples to H0 using the declared expansion shape. Return a frozen inference result with samples and units; convergence and independent coverage remain explicitly unexecuted.

## core/power.py::export_linear_power

<!-- implementation-sha256: ad16957d88350415254a686e0e39182867fab08031bfd2c88c15e9788268807f -->

Verify a saved linear solver bundle, epoch, hashes and explicit CLASS units before writing a fresh figure export. Convert k from h/Mpc to 1/Mpc by multiplying h and P from (Mpc/h)^3 to Mpc^3 by dividing h^3. Publish input-power.json, signed-transfers.json and the species reference table, preserving signed density transfers and velocity transfers in Mpc^-1; return the scientific gallery export. Memory and source-integrity checks apply; solver convergence is not established by this export.

## core/power.py::export_evolved_power

<!-- implementation-sha256: cb031e2ae305e72a57a6dfeafde8a1400507b241c9a11396606264aaff20165e -->

Read a source-verified native analysis bundle and its original run configuration. For each saved power spectrum, require aligned epoch/unit-labelled estimator arrays, integer mode counts, raw and shot-noise power, errors and fold metadata. Convert k/P to physical units and mark support only for finite power/errors with nonnegative error and occupied bins. Write per-snapshot power products, products.json and a gallery in a fresh destination; return the gallery record. Preserve native estimator policies without claiming numerical convergence.

## core/power.py::export_matched_power

<!-- implementation-sha256: f1709993a9fcfa61d17b5070883c959f3b840f27b3fb9edf67759c7ef059c2eb -->

Compare two hash-bound evolved native power products at identical actual epochs, physical k bins and estimator policies. Require a full ordered joint covariance in Mpc^6 bound to those exact products and independently check their actual initial-condition phases. Propagate covariance to B/A and B−A, restricting support to both inputs and the measured phase range. Write matched-power.json and ratio/residual gallery products in a fresh destination; return the gallery record. No interpolation or covariance fabrication fills missing measurements.

## core/power.py::export_power_ensemble

<!-- implementation-sha256: e71f202f1a3e2add30aec31e483663a75ef015c88f7307af81f31194738338f1 -->

Read verified paired native realizations with common epoch, bins and policies, measured within-pair phase support and distinct seed/IC identities. Compute the full ordered [P_A,P_B] sample covariance divided by realization count for the ensemble means, then their ratio and difference with propagated errors. Write ensemble-power.json, products.json and its gallery; return the gallery record. Support is the intersection across realizations. Independence remains conditional on source-generation design, and these exports do not establish scientific acceptance.

## core/stream_common.py::finite

<!-- implementation-sha256: 1077d8ae1f55f6c74bb89ef81f43826f6fd7e247bf8df4c6df9f70e33d1c174b -->

Convert an input to a floating NumPy array and require every element to be finite. If shape is provided, require that exact shape; include name in validation errors. Return the array, which may share storage with an already compatible input. This helper validates numbers and dimensions, leaving physical units and interpretation to its caller.

## core/stream_common.py::positive

<!-- implementation-sha256: 1b2503bac4566540ad72652c823d0a44ecd3a0ea5e17d67e5f846510ed82a2da -->

Require a finite strictly positive numeric scalar with NumPy floating, signed-integer or unsigned-integer dtype, then return a Python float. Reject arrays, booleans and other nonnumeric values. The supplied name identifies failed controls such as memory, length or timescale; units are defined by the calling workflow.

## core/stream_common.py::integer

<!-- implementation-sha256: 386fcecf7e64d6707c848329870aa7a67598b7cffd8effac73e594357ae04b59 -->

Require an explicit built-in Python int at least minimum and return it unchanged. Booleans, floating representations and NumPy integer scalars are rejected by the exact type check. This protects discrete counts and seeds from implicit coercion; the default minimum is one.

## core/stream_common.py::increasing

<!-- implementation-sha256: b44b509a666569abce8b8abd7d9b19169cf3fd09b8e09d612501be2b14210416 -->

Convert values through finite() and require a one-dimensional sequence with at least minimum entries and strictly positive successive differences. Return the floating array; duplicate or reversed samples raise. Callers use this for times or bin edges and remain responsible for the chosen physical units.

## core/stream_common.py::phase_space

<!-- implementation-sha256: 48178d32a1c61b3a935d720d2142acd0e322be1bf7f92307aca6675bd14cb9df -->

Validate an aligned particle sample: N-by-3 positions in proper kpc, matching velocities in km/s, positive measured masses in Msun and unique exact uint64 particle IDs. Require a nonblank identity namespace. Return positions, velocities, masses and IDs in the original row order; empty samples are allowed when their shapes are valid. This checks the supplied sample rather than loading or converting native snapshots.

## core/stream_common.py::seeded_rng

<!-- implementation-sha256: 9d8663378bec327222538faf5e4ad903b6554f7443903fcdba540b5c54af7286 -->

Create a NumPy Philox Generator from a nonnegative explicit integer seed and JSON-serializable labels. Hash the canonical JSON payload with SHA-256 and use its first 128 bits as the generator seed, providing reproducible labelled substreams without changing NumPy’s global RNG. Return the generator; labels distinguish separate stochastic operations.

## core/stream_common.py::json_safe

<!-- implementation-sha256: d7c36478c72df32518dfb17749309dbd292aea3c09b66b50e3a2a1ec24da545f -->

Recursively convert NumPy arrays/scalars to Python values, tuples to lists and dictionary keys to strings. Replace nonfinite Python floats with None for missing numeric values. Return the converted structure without writing JSON; unsupported custom objects pass through and may still need handling by the serializer.

## core/gravity_fields.py::ComponentField

<!-- implementation-sha256: d73e13332b9230059c1c31fd54c4a052c9f590fae297dc803c23800b4c1f5ced -->

Combine named imposed gravity components with matching accounting records. Require each component’s gravity_count to equal one, known physical roles and no overlapping declared native source IDs. evaluate(x,t) adds each component’s potential [(km/s)^2], acceleration [(km/s)^2 per proper kpc] and potential Hessian [(km/s)^2 per proper kpc^2], validating aligned finite arrays. This evaluates an imposed field; it does not evolve live gravitating particles or prove undeclared source identities are distinct.

## core/gravity_fields.py::FrozenField

<!-- implementation-sha256: 7bcb96214921174eb447ed7053e86e6e1d5984b97d68d5f4eeb37ad685716526 -->

Wrap a field at one finite epoch. Every evaluate(x,t) call delegates to the underlying field with that saved epoch, ignoring the requested time while retaining the original position queries and output units. This supplies a static diagnostic control without copying or re-evolving the component.

## core/gravity_fields.py::construct_component_control

<!-- implementation-sha256: fbb7fc9d0b774cf2f1328540a52d528fc9e8d25c9bbd0f8d556b3bc0e6b72866 -->

Build a declared intervention on an existing ComponentField: keep the full field, retain selected components, subtract them, freeze selected epochs, fix translating-host centres, or scale selected masses by a positive nonunit multiplier. Validate distinct existing component names and operation-specific inputs, preserve accounting for retained components and reject an empty result. Return the original full field or a new composite wrapper; this constructs a counterfactual force model rather than running its trajectories.

## core/gravity_fields.py::fourier_torque_diagnostics

<!-- implementation-sha256: ac809f3dacf01ed2778fb89a3df613af4d6d40fc7da43467cfd9cdaf5bc58c66 -->

Bin particles by cylindrical radius and return mass-weighted cosine/sine moments for azimuthal modes m=1 through 4 plus actual total z torque sum m(x cross a)_z in Msun (km/s)^2. Validate aligned finite positions/accelerations/masses, nonnegative masses and increasing radial edges; include the final upper edge. Empty or zero-mass bins retain NaNs and false support. Moments diagnose asymmetry; resonance identification requires separate orbital frequencies.

## core/stellar_observations.py::composite_spectrum

<!-- implementation-sha256: 527d21e0d50c06d4b9f87d3dbd4cb35d97a2586e3b2ee967c608ae62967c4981 -->

Sum the library spectra for positive formed-mass weights at supplied ages in Gyr and positive metallicities. Require aligned age/weight/metallicity arrays with nonnegative weights and positive total, nonnegative E(B−V), and a nonnegative attenuation curve on every library wavelength. Apply L_lambda times 10^(−0.4 E(B−V) k_lambda) and return the finite spectrum in the library’s luminosity-density normalization. Zero-weight components skip library interpolation. This combines existing population spectra without running FSPS or writing an observation product.

## core/stellar_observations.py::distance_modulus

<!-- implementation-sha256: 4d314f7dd3b7ee19c18c75957e9a471c4b332a49b102a4474b4ded53d579e8fa -->

Return the astronomical distance modulus 5 log10(D_L/Mpc)+25 for a positive finite luminosity distance or array of distances in physical Mpc. The result is in magnitudes. This is the distance conversion alone; redshift-dependent spectral, extinction and filter corrections are supplied elsewhere.

## core/stream_tracks.py::PrincipalCurve

<!-- implementation-sha256: fff18dc9ac67641b752beb0955f40c68940ecf1c1e70628d2ef443e738045e58 -->

Store fitted Cartesian splines, particle parameters, sampled grid and actual arclength, track positions and transported tangent/normal/binormal axes, support, fit variance in kpc^2 and diagnostics. evaluate(parameter, derivative) returns the three spline components or derivatives as an N-by-3 array. Construction stores an already fitted result; fitting, convergence and endpoint support are determined by fit_uncertainty_aware_principal_curve.

## core/stream_tracks.py::separate_by_ancestry_and_phase_space

<!-- implementation-sha256: b156e098438b93a04fb4a65b85702deb74ab4e33c7f99c45287027c23ea7a1f5 -->

Split an aligned particle sample first by supplied exact ancestry labels and optional unwrapped orbital turns, then by connected components in a position/velocity-scaled six-dimensional neighbour graph. Enforce positive linking scales and a maximum graph-edge budget. Return group records and per-particle group labels. Without unwrapped phase, report coherent components rather than assigning physical wrap identity from proximity alone; this partition precedes principal-curve fitting.

## core/stream_tracks.py::fit_uncertainty_aware_principal_curve

<!-- implementation-sha256: 08fd1469f452c6c1268d7156294fb629c4223fb1433c2b6afedf2624cffe6469 -->

Fit one resolved stream wrap from particle phase space using either a six-dimensional minimum-spanning-tree backbone or supplied ordering parameters. Alternate mass-weighted centroid spline fitting and nearest-curve projection, weighting residuals by centroid uncertainty. Preserve ordering anchors outside fitted centroid bounds, compute actual arclength by integrating spline speed and transport an orthonormal frame. Return PrincipalCurve with iteration history, variance and support excluding unresolved endpoints/reversals or nonconvergence. Position units are proper kpc; smoothing is a dimensionless weighted-residual budget.

## core/stream_tracks.py::project_to_arclength_transverse_frame

<!-- implementation-sha256: 564c787c6a09c38d6892e73bc92d8fa38b500d35fd6fd75715cdb0a5ae4c1d87 -->

Project sample positions onto the fitted spline and interpolate its actual arclength and transported axes. Reorthogonalize interpolated axes before resolving position residuals into two transverse coordinates and velocities into three local components. Return particle IDs, arclength/transverse/residual distances in proper kpc, local velocities in km/s and interpolated projection support, preserving row order. Endpoint-clipped projections remain subject to the curve’s support mask.

## core/stream_tracks.py::choose_bins_from_resolution_and_sampling

<!-- implementation-sha256: 4219303660c9ed109a1a5a654f0cb858b8889383a0957a9378602b0edbd8873f -->

Choose uniform arclength edges from finite nonempty samples. Limit the bin count by maximum_bins, total sample count divided by minimum_count, and total extent divided by physical resolution_kpc; require at least two bins. Return the evenly spaced edges in kpc. Empty individual bins survive: the global count bound does not guarantee every bin has enough measured particles.

## core/stream_tracks.py::measure_local_observables

<!-- implementation-sha256: f6e7ea08ff042bc0ef29d1129cf6755cfb03fed34c08e5ff80680bc52a003e82 -->

Measure a stream in aligned local coordinates, requiring the original exact particle order. Weight masses by membership_probability/selection_probability; divide bin mass by arclength width to obtain Msun/kpc density and propagate weighted sampling plus membership variance. Compute local transverse and velocity covariances with conditional particle-bootstrap errors, effective counts and projection support. Compare density to a Gaussian-smoothed baseline to propose gaps/caustics, and test offsets/two-mode evidence for spurs/forks. Return a report and arrays, including Hann-windowed relative-density power in kpc with its shot/window terms and additional measured-support products. Candidate features and conditional bootstrap errors require further controls and whole-track uncertainty.

## core/stream_tracks.py::stable_track_derivatives

<!-- implementation-sha256: aa3ed9a097f745c78520885ad183ebbfe9f8c0ed07490c196efee61719621069 -->

Fit cubic smoothing splines independently to a finite planar or three-dimensional track on at least five increasing parameter samples. Require positive smoothing variance and positive aligned weights; use a residual budget of smoothing_variance_kpc2 times sample count. Return fitted positions and first, second and third parameter derivatives as four arrays. Derivative units follow the caller’s parameterization; subsequent curvature calculations remove that parameter dependence.

## core/stream_tracks.py::viewing_basis

<!-- implementation-sha256: 249e523b034445182544c8500d36a49b6f4140ee6b507441822ba231d0b72bf5 -->

Return a three-row orthonormal basis [horizontal, vertical, line of sight] from a finite nonzero viewing direction. Use a least-aligned Cartesian axis for the default up vector, or validate the supplied up direction; reject parallel up/line-of-sight vectors. Row-vector positions project by multiplication with the basis transpose. This defines geometry without changing the particle population.

## core/stream_tracks.py::fibonacci_viewing_ensemble

<!-- implementation-sha256: 3e4d2314c21f628454b3f8426d34273d09d1907730575321ee50b22c28838840 -->

Return count approximately uniform unit directions on the sphere using midpoint z samples and the golden-angle Fibonacci construction. Require an explicit integer count of at least two. This deterministic direction set uses no random generator; multiple views of one stream remain views of the same physical system.

## core/stream_tracks.py::measure_viewing_ensemble

<!-- implementation-sha256: ffde440a21ff9cf0b48442391f9bc1281e0138156f13369973930825dd618213 -->

Project a fitted curve and its spatial-width covariance into each supplied viewing direction. Compute curvature and host-relative convexity using local width gradients, track/selection masks and finite-width/tangent exclusions. Seed uncertainty draws for host-centre covariance and optionally refit independent point-covariance track draws; calculate convex probability conditional on at least two supported draws and identify sufficiently long confident segments. Return geometry/report arrays with proper-kpc distances, inverse-kpc curvature and per-view support. Views are not independent halos; the geometric diagnostic alone does not identify a unique force model or SIDM cross section.

## core/stream/trajectories.py::orbital_interpolation_contract

<!-- implementation-sha256: caba798873f1f72fed9876cf7a7689f959e5bf809622c7a2295289013369d260 -->

Return the loaded orbit interpolation settings, their SHA-256 and the repository-relative configuration path. Rehash the settings file first and reject changes since import, requiring a restart before producing events. This identifies the interpolation algorithm; it does not supply a trajectory error certificate or independently validate a physical orbit.

## core/stream/trajectories.py::real_roots

<!-- implementation-sha256: 44cfaddf37294ca53d57702a1b13d1874b6b143743d1f6e6ec6db167b8df659a -->

Trim polynomial coefficients with tolerance 1e-16, solve its roots, and return sorted real components strictly inside (lo,hi). Accept a root as real using the configured relative imaginary tolerance. Endpoints are excluded and repeated roots are not explicitly deduplicated. This numerical helper assumes compatible polynomial/input coordinates; callers normalize physical intervals where required.

## core/stream/trajectories.py::Segment

<!-- implementation-sha256: 79bd03d37acb6e79e04e8b86de5649ab6b0ec28417bd3ea3bd8730dbe8bf0919 -->

Hold one Worldline interval, its integer interval index and lower/upper four-dimensional time-space bounds for the broad-phase search. start and stop read the corresponding saved proper times from the worldline. This immutable record stores supplied bounds; SpacetimeBoundingIndex constructs them from supported cubic trajectory intervals rather than Segment validating them itself.

## core/stream/trajectories.py::build_spacetime_bounding_index

<!-- implementation-sha256: 369cd84f4d9214b03d217fc91341463a92232f8316dc619661adf118356a2789 -->

Construct and return SpacetimeBoundingIndex from worldlines and forwarded controls, including search padding in proper kpc, leaf size and maximum retained segments. The index includes supported intervals and their cubic swept bounds in time and space. It identifies possible overlaps for later close-approach refinement; building it does not establish an encounter or write a product.

## core/stream/trajectories.py::query_stream_world_tube

<!-- implementation-sha256: 792b680e648d2d0e381aa20f36ea6d6d3d04f0da73412f0460b34ae1d5afdba4 -->

Yield candidate tuples (stream, stream interval index, perturber, perturber interval index) for overlapping supported spacetime bounds. Skip candidates with the same namespace-qualified identity as the stream and raise when the emitted-pair count exceeds maximum_pairs. This generator performs the broad-phase search only; spatially overlapping boxes still require continuous minimum-separation calculations.

## core/stream/trajectories.py::reconstruct_encounters

<!-- implementation-sha256: 70a1d07063a0656e70ad43662e22be2414753cfcc8c9baa89a54deaaf2a49e82 -->

Validate unique stream/perturber worldlines, build a bounded swept-worldtube index and solve close-approach candidates within the search radius. Reject nonstationary interior catalogue knots, deduplicate event times and refine geometry with the declared covariance draws. Mark events requiring refinement when interpolation certification is absent, sampling error exceeds the proper-kpc budget or coverage ends at the minimum; retain native branch requests and invalid intervals. Return a report plus times [Myr], impacts/errors [kpc], relative velocities [km/s] and support arrays. Worldline errors are treated independently; cross-worldline/shared-frame covariance is omitted.

## core/stream/encounters.py::require_resolved_binding_and_worldlines

<!-- implementation-sha256: 23b4ce0bddadc215a074fd820127e11843424f3445bf1f9693622843699e7517 -->

Require a nested system labelled RESOLVED_BOUND with binding source hashes, passed resolution gates and parent profiles excluding child mass. Construct both EvolvingPerturber models, require a common inertial frame and valid trajectory coverage throughout the requested time span, and reject missing/disruption intervals. Elastic-SIDM records also need evaporation and disruption histories. Return the parent/child models; this consumes recorded binding evidence rather than deriving bound status from their proximity.

## core/stream/encounters.py::measure_multiple_features

<!-- implementation-sha256: e1858afff02799203b7a6fb669636f241845769429eacacd332125389dabb668 -->

Find candidate density gaps in each contiguous run of finite density-ratio and transverse-offset bins. Rank peaks of 1−ratio by prominence, enforce physical separation using bin-centre distances in kpc and return sorted features with gap depth, prominence, two-component spur offset and the finite-bin mask. Missing bins split searches rather than becoming detections. The result lists alternative explanations; several features alone do not establish a bound compound perturber.

## core/stream/encounters.py::observed_phase_space

<!-- implementation-sha256: 259de9b581394f1066d9f3d56d8e903ef08683051d80a5f3aed025988a5827fe -->

Transform the final saved position/velocity state of an integrated stream through the supplied ObserverFrame. Require a finite N-by-6 output aligned with particle IDs and return observer coordinates: longitude/latitude [rad], distance [kpc], radial velocity [km/s] and two proper motions [mas/year]. This adapter uses the last integration state; it does not simulate observational noise or selection.

## core/stream/encounters.py::generate_template_ensemble

<!-- implementation-sha256: 5a9d703f3fe9968a483659ac3487f0112485b24352b326948e76d3e95d63ca38 -->

Freeze one common initial stream and evolve every declared matched perturber template in a host that excludes those candidates. Enforce template-count and retained-history memory budgets, measure track density/velocity and transform final states to observer coordinates. Optional centred log-mass perturbations estimate sensitivity, wrapping longitude differences at the angular seam. Return template provenance/forward records and prefixed arrays, including the shared initial-state hash. Templates are imposed-field forward experiments; their classification remains UNCALIBRATED until separate held-out evaluation.

## core/stream/encounters.py::calibrate_template_classifier

<!-- implementation-sha256: c7c0da53dc130d8aaabb73b9233401ec00c99f1fc79868c53f6a5845ba7c568f -->

Fit Gaussian feature-class models using disjoint named training and held-out realization groups. Require at least three distinct training groups per class, shrink sample covariances toward their diagonals and add the supplied positive-definite covariance floor; class priors follow training row counts. Evaluate held-out probabilities and append an ambiguous confusion-matrix column below the confidence threshold. Return calibration metadata and numeric model/evaluation arrays, leaving scientific acceptance unset. Group separation is required but physical independence of supplied realizations still depends on their generation.

## core/stream/inference.py::estimator_only

<!-- implementation-sha256: 3e702b2728b994eade3d5d979bd3dafc18adff08eaa98b971c80b06fad832b06 -->

Recursively inspect dictionaries and list/tuple contents for keys in TRUTH_KEYS, raising if privileged truth fields enter estimator inputs. Return None when the check passes. This is a named-field boundary check rather than a general information-flow proof; arbitrary custom objects and unrecognized aliases are not recursively inspected.

## core/stream/inference.py::selected_observation_loglikelihood

<!-- implementation-sha256: 8effb826156044b1f95793bc5a4ceed6e314d25db69234ba6803b9bb3af91429 -->

Evaluate a Gaussian observation likelihood on object-by-coordinate data, excluding declared missing entries and adding model-error covariance. Wrap residuals for declared periodic columns using a local tangent Gaussian. Optional signal/background membership mixtures require independent object covariance blocks. Apply the declared unselected, conditional-detection or Poisson-population normalization and add missingness log probabilities; return a scalar log likelihood, or negative infinity for invalid detection normalization/probabilities. Broad angular likelihoods need another family; this function does not generate predictions or fit parameters.

## core/stream/inference.py::PosteriorModel

<!-- implementation-sha256: 7067c5b716c9f3e97abb8a73444bd7ab460336cf760394d7d0b088446711e722 -->

Combine a ParameterSpace, forward callable, observation dictionary and selection callable. On construction reject named privileged truth fields in the observations. log_likelihood maps an allowed parameter vector to predictions and selection, rejecting unsupported selections before evaluating selected_observation_loglikelihood; log_posterior adds the parameter prior. This model adapter performs no sampling or file writes by itself.

## core/stream/inference.py::fit_sampler

<!-- implementation-sha256: b6e3e7ca4ec42fb417b36495eb4a7234ae8ae01232fbef61f9a8556796fd08ce -->

Run the pinned dynesty or emcee backend under a run-directory lock, binding continuation to code, data, priors, sampler settings and installed assets. Require explicit resume for an existing binding and matching checkpoints. Dynesty writes a checkpoint and weighted posterior/evidence arrays; emcee writes chain.h5 and returns burn/thin-selected samples, weights and available autocorrelation diagnostics. Write posterior.json and return report/arrays. Resource or stopping limits, short chains and unavailable independent-ensemble R-hat remain explicit; a written chain does not establish convergence or calibration.

## core/stream/inference.py::posterior_predictive

<!-- implementation-sha256: aa91bf5873de0157888481b00bedf9f7f0e206c4ec87e317f4102636594f3a01 -->

Draw parameter rows according to supplied posterior weights and run the forward model for each. Generate Gaussian replicated observations with observational plus model-error covariance; selected cases require an observable detection rule and bounded rejection sampling. On nonmissing coordinates compare observed and replicated Mahalanobis discrepancies, returning their tail probability and replication/mean/index arrays. The observed count stays fixed, including for a selected population; this is a diagnostic rather than an acceptance decision.

## core/stream/inference.py::shape_coordinates

<!-- implementation-sha256: 395069d5bb390a47d3d710dfd067a05a04b7c4e36c246586d38840b5681d88f5 -->

Validate aligned finite axis ratios satisfying 0<c/a<=b/a<=1 and return both ratios plus triaxiality T=(1−(b/a)^2)/(1−(c/a)^2). Mark T unsupported and leave it NaN when its denominator is at most 1e-12, including spherical degeneracy. This transforms already measured or inferred ratios without estimating halo shape from particles.

## core/stream/inference.py::covariate_design

<!-- implementation-sha256: 2fb7ecdc9d8d14483e68ba7924b391598544ac0c5a25857fc13f9167aa29a713 -->

Reject named privileged truth fields in a covariate record, then assemble [1, standardized log_mass_msun, log_radius_kpc, redshift, environment] using frozen four-component centres/scales. Require finite values and positive scales. Return the five-element regression design vector. Environment must already have a numeric encoding; this helper neither learns normalization nor extracts covariates from a simulation.

## core/stream/inference.py::hierarchical_reweighting

<!-- implementation-sha256: 7d4b20e1e9bd9ae7eb796257d984c07b1dd99a0fe144b20c835e8a6c67e36521 -->

Combine individual posterior samples through importance ratios of population shape density to each sample’s original shape prior, retaining posterior weights. Require matching shape definitions and one joint posterior per halo to avoid double counting multiple streams/shells. Reject insufficient importance effective sample size, absent population support or unsupported per-object selection normalization. Return the summed selection-corrected log likelihood and per-halo diagnostics, or negative infinity with the failure reason. This uses supplied physical/selection models; it does not establish population calibration.

## core/stream/inference.py::HierarchicalPosterior

<!-- implementation-sha256: ad0748811966af765eb237e2e0e758fd3b6658354e3c7d4a48ce614803584533 -->

Adapt a ParameterSpace, individual posteriors, ShapePopulation and population-selection callable to sampler methods. log_likelihood maps hyperparameters and delegates to hierarchical_reweighting with the required effective-sample threshold; log_posterior first enforces prior support and adds the prior. Construction alone neither reweights samples nor runs inference, and likelihood diagnostics remain available from the underlying reweighting routine.

## core/stream/workflow.py::Inputs

<!-- implementation-sha256: 23836315343b34c04cdeba9bed096dbaf2bb07780b83649449a401b2a5a4c909 -->

Resolve campaign input files relative to the request base or an earlier published stage; reject stage references escaping their generation and record file hashes. Read JSON and sealed numeric products within the stage memory budget. sample() selects an epoch, optional candidate/wrap and aligned proper-kpc/km/s/Msun rows with exact identities and a particle cap. host() reconstructs measured fields and satellites while enforcing exclusion accounting. This reader supplies validated inputs to stage producers; it does not publish stages or execute a campaign itself.

## core/stream/workflow.py::execute_stream_stage

<!-- implementation-sha256: 77f71373867584524ed2477f865b7fe0b0c60b44460c45aeea1b4be93ca0d819 -->

Execute one stream stage using Inputs and the campaign’s resource bounds. Dispatch catalogue extraction, classifier calibration, wrap/curve/local measurements, viewing geometry, live progenitor evolution, calibrated spray, actions/history, Lyapunov and observer transforms; delegate experiment kinds to execute_experiment_stage. Retain per-wrap insufficient/unresolved states and explicit support masks. Optional independent checks use Gala, StreaMAX or AGAMA through their actual adapters. Return (report, arrays, namespace, epoch) for the campaign publisher. Live stages may write checkpoints under a lock; this dispatcher does not itself seal the common stage product or certify physical convergence.

## core/stream/workflow.py::run_stream_campaign

<!-- implementation-sha256: 6bc2e365ed8413448b8f54cadbe37ee945f9c39f79cba8e7159c179f4b14733f -->

Validate stage order/references and run stages sequentially under a destination lock. Reuse a published stage only when its request, resources, code hashes, input hashes and sealed numeric payload still verify. For new stages, compute in a temporary generation, enforce output workspace, publish product.json with numeric arrays/units, then stage.json and an atomic stage directory. Preserve failure.json in the temporary generation and rethrow on failure. Write campaign.json and return the campaign result after all stages; execution COMPLETE remains separate from scientific acceptance. Relative input resolution comes from base, not the current working directory.

## core/stream/workflow.py::streams_main

<!-- implementation-sha256: 132adbf54dbbb6212c92324d812ccea2ad4590e05559f362ee28374b7abec649 -->

Parse the stream campaign request JSON and required --output directory. Resolve the request path, use its parent as the input base, call run_stream_campaign with that path for provenance and print the JSON-safe result. This is the commands.py streams CLI adapter; it returns no explicit status value, while campaign exceptions propagate rather than being printed as a successful result.

## core/stream/experiments.py::parameterize_candidate_record

<!-- implementation-sha256: 0b659db74f59b8b7edb7c534dbab2126bb06b22ad39a7ec1c94e73b38222b201 -->

Deep-copy an evolving perturber record and apply explicit positive mass/radius scale factors to its worldline masses and supported profile fields. Optionally replace positions/velocities with a local linear flyby specified in proper kpc, km/s and Myr, requiring transverse impact/velocity vectors, or apply a fixed positional offset. Return the altered record with diagnostic support/provenance labels for rescaling or the model flyby. These transformations parameterize a forward model; they do not become newly measured native trajectories.

## core/stream/experiments.py::build_component_field

<!-- implementation-sha256: cd0d7697389b650db0bb167c8f31c6d5ab0b34891abd2d0ec9eca1b3c1e96e97 -->

Construct named gravity components from a JSON definition using measured hosts/coefficients, analytic spherical or halo-shape fields, evolving perturbers or baryonic differentiated potentials. Apply optional common rotation/translation wrappers and verify declared measured provenance files through Inputs. Return a ComponentField with explicit source accounting and the requested actual host moving frame, rejecting unknown kinds or a moving-frame name without a host. Construction resolves models and assets rather than evolving particles.

## core/stream/experiments.py::physical_posterior

<!-- implementation-sha256: 132d57b8c2f9c2aa3005ebf47f6fdb55b751919608dc6d3cc0b33e0baf24b677 -->

Build a physical PosteriorModel from stage priors, forward-definition, observations and selection inputs. Optionally replace the initial stream with an epoch-selected measured sample; impose campaign memory/particle bounds and reject an oversized joint covariance. Resolve component assets before checkpoint binding, require the prior-name set to match declared forward bindings, then attach the physical forward and selection adapters. Return the model without starting a sampler or writing posterior files.

## core/stream/experiments.py::execute_experiment_stage

<!-- implementation-sha256: e2d7424a16dbdb8ccafa4579c13f3e5ea514dc4910a042fdd4a6c82800ce60ab -->

Dispatch a connected experiment stage, delegating observation, assembly, matched-model and SBI families to their producers. Other branches handle initial states/worldlines, continuous encounters and certificate assets, counterfactual/template/compound experiments, selected physical or hierarchical inference, posterior prediction, held-out recovery, independent-chain diagnostics and native checkpoint branches. Read earlier sealed products through Inputs and pass campaign resource limits into integrations. Return (report, arrays, namespace, epoch), leaving common publication to run_stream_campaign. Sampler checkpoints and requested native branches can write external products; measured support and scientific acceptance remain separate from stage execution.

## core/stream_resimulation.py::load_live_checkpoint

<!-- implementation-sha256: 293bd6e0d40f3eee65f40fa7194d6b15e65b70f65140b450990bf9bf64436d3b -->

Read a numeric live-stream restart only after checking the metadata seal, restart schema, exact caller binding and archive SHA-256. Bound metadata and expanded NPY sizes by memory_gib, require unique flat numeric archive members and load with pickle disabled. Reject scalar/array state-key collisions and return the merged state. Detailed identity, clock and release-history consistency is subsequently checked by evolve_self_gravitating_progenitor; this reader does not advance the simulation.

## core/stream_resimulation.py::evolve_self_gravitating_progenitor

<!-- implementation-sha256: fa730dc3e2218294123907cf863ce6224d7229707558ba0351a51be237b65a48 -->

Evolve a nonempty stellar progenitor with direct softened self-gravity plus a prescribed host using adaptive kick-drift-kick steps, in proper kpc, km/s and Myr. Enforce particle/memory/step limits and reject dark-matter-bearing progenitors needing a native SIDM route. At saved outputs measure the self-bound remnant, retain first bound-to-unbound time brackets and release states, and report energy plus explicit-time host work in Msun (km/s)^2. Optional bound checkpoints preserve identities, ancestry and stripping clocks; resume returns the suffix beginning at a requested boundary. Return report/history arrays. The parent host receives no backreaction and convergence is not established here.

## core/stream_resolution.py::validate_stream_resolution

<!-- implementation-sha256: bbdc6ddb787f8ad94b20c79c8f31ae3ab9b4425e8b593e9daaad34cdba4ca9d9 -->

Validate the exact stream-resolution request fields: proper width, velocity scales, progenitor/perturber masses, member-count minima and softening/cadence ratios. Require positive numeric values, explicit integer member minima of at least three, and ratio maxima no greater than one. Return the same dictionary or raise. This checks planning controls; actual saved sampling, cadence and observable convergence are evaluated by the downstream feasibility and measurement routines.

## core/stream/encoders.py::ObservableEncoder

<!-- implementation-sha256: f5538b4f761ba6f3f0f534b8b409ccbcdb1e5d698372ad3aa48893a74e27b65e -->

Pack observed coordinates and their full joint covariance into a fixed-size float32 vector for downstream SBI encoders. Require fixed unique columns, positive frozen scales, bounded node count, explicit boolean missingness, finite unmasked values and positive-semidefinite covariance. Normalize coordinates/covariance, wrap declared periodic columns, replace missing coordinate values with their centres and retain coordinate/padding masks plus local covariance blocks. output_size includes all padded node features and the padded full covariance matrix. Named privileged truth keys are rejected. This class packs observations; it does not train a graph network or calibrate a posterior.

## core/figures.py::semantic_cmap

<!-- implementation-sha256: be172b5ffbb9b516c76f6dba138962b46511997aa5312ecec7cbefb45261d94b -->

Return the configured colormap name for density/cosmic-web, temperature, sequential, difference or categorical data. Unknown semantics raise ValueError instead of silently choosing a palette. This maps quantity meaning to a palette identifier; registration and plotting occur elsewhere.

## core/figures.py::verify_figure_export

<!-- implementation-sha256: d4e40f376a19e7a7e0e7114556604b037b5b44900ce19441c1838f29bdf6d3e5 -->

Verify a figure JSON manifest against its adjacent .sha256 file, then require every listed artifact to remain within the export directory and match its recorded hash. Open NPZ arrays with pickle disabled and validate the declared plot-data array contract when present. Return the manifest or raise. This detects changed/missing numerical and image assets; it does not visually inspect readability or establish physical accuracy.

## core/figures.py::normalized_weights

<!-- implementation-sha256: a719a26fdf27117db0f42a35c1e815e4b46f35532bdcac19f877764173d0c6a6 -->

Return uniform weights when none are supplied, otherwise require a finite nonnegative vector of length count with positive support. Divide by the maximum before normalizing the sum to avoid overflow from large common scales. Return the normalized NumPy vector. The caller supplies a nonempty population; this helper does not interpret weight units or calculate an estimator.

## core/figures.py::verify_figure_review

<!-- implementation-sha256: 0b4fa9ff74fed9a97eacf36604f21c79aeef3423c85803b6cba3a686da2a5a52 -->

Read figure_review.json only after checking figure_review.sha256, then verify that each listed review artifact resolves inside the directory and matches its saved digest. Return the review record. This is file-integrity verification of an existing review package, not a new human or browser assessment of its images.

## core/foundation.py::canonical_digest

<!-- implementation-sha256: b58ae8e83daf5fc0790bb4c668ba2736567b8e057073d504226bf9f297eafb5f -->

Return the SHA-256 of a JSON-serializable value encoded with sorted keys and compact separators. Disallow NaN/infinity through allow_nan=False so invalid numeric payloads cannot acquire an ordinary canonical digest. Python/NumPy scientific objects must be converted by the caller; this routine does not read or hash files.

## core/graph_producer_registry.py::audit_graph_producers

<!-- implementation-sha256: 9a6577cfe98429e3f897dd06d281215c1b124d484fe84b95abe9936ad8ef1209 -->

Check that the bounded producer-registry JSON contains exactly the 102 registered family/variant pairs, complete input/measurement/unit/support descriptions and the required image/numeric artifacts. Import and inspect every declared first-party producer/renderer, requiring a callable within the repository; record source paths, signatures and hashes. Return the route audit. This checks callable connections and metadata, without invoking any producer or proving that its measurements/figures can execute.

## core/native.py::native_run_lock_fds

<!-- implementation-sha256: e74b9ca40fe7681a4281de4fe3cfeced3c65a58a8c047f0fc90ee96112e29efe -->

Return a one-element tuple containing the currently held run-lock file descriptor only when this process and thread own that lock; otherwise return an empty tuple. Resolve the lock identity and inspect its in-process registry under the guard. Callers can preserve an owned descriptor across native launch; this helper neither acquires a lock nor proves another process is idle.

## core/native.py::verify_invocation_inputs

<!-- implementation-sha256: 43d9077399a98d9c92116276ddd2c9199035c0fa9d11d5c6cf88696d75df120b -->

Rehash every path in a native invocation record’s input_hashes and raise when any digest differs. Return None after all files match; missing files propagate their read error. This checks the declared invocation inputs at the time of the call, without executing a binary or inspecting its output.

## core/native.py::NativeParticleCapacityError

<!-- implementation-sha256: f3cafa3de7210881877806d248a363f6e9d1bdbb235b0e66fdc3c0430e02b6a3 -->

Raise a MemoryError subtype carrying the rejected capacity report as .report. The message includes total particle count, allocated MPI ranks and the minimum rank requirement for native particle-array indexing. Constructing this exception does not modify the simulation or increase its allocation.

## core/native.py::validate_native_particle_capacity

<!-- implementation-sha256: eb31f9c02cc6a75fdee183fbf71a14e0315cb215cc22df6fd1360f4e4536a578 -->

Require an explicit positive MPI rank count and a complete or planned-uniform six-species integer particle census. Compute the native particle-index requirement and return a NECESSARY_CAPACITY_PASS report, or raise NativeParticleCapacityError with the failed report. This bounds signed-integer particle-array capacity only; memory, domain imbalance, tree/source buffers, scheduler resources and physical validity need separate checks. No native execution occurs.

## core/native.py::preflight_native_ic_capacity

<!-- implementation-sha256: ed7a47d1427680eda23b3204c5c142384fd1e2bdbc27df104f957c11ca7f3989 -->

Read the initial-condition particle census from the supplied paths through initial_condition_census, then apply validate_native_particle_capacity to the requested MPI ranks. Return its capacity report or propagate census/capacity errors. This inspects IC metadata for an indexing prerequisite; it neither evolves particles nor validates the entire native resource allocation.

## core/numeric.py::validate_response_workspace

<!-- implementation-sha256: 9355bb9d913a6a9970f5b6df4ea939e406d66e76e0d498c22147422b93c97c76 -->

Require an explicit dictionary of positive finite numeric GiB allocations: disk_gib for one publication, or exactly source_disk_gib and noise_disk_gib when combined=True. Reject booleans, missing/extra keys and invalid values. Return None on success. This validates declared storage controls without checking free disk space or allocating files.

## core/numeric.py::response_disk_bound

<!-- implementation-sha256: dfbff3837a8b297ff5f6d6d1e7f8cc6a812b23c52e6b3c3856d4b7b7cb408df2 -->

Estimate retained response publication bytes from a nonnegative integer payload size and positive integer array count. Include an NPY workspace, conservative archive/header overhead and an additional MiB. Return the byte bound used by preflight_response_disk. This is a publication-storage estimate; it is not a measured compressed size or a process-memory bound.

## core/numeric.py::preflight_response_disk

<!-- implementation-sha256: 6ceb90310cb08849a3f598ed18da0e6c673cc40b689b0a0289e4dd197b61e302 -->

Validate disk_gib, compute the response spool/publication byte bound and compare it with both the declared allocation and free space at the nearest existing destination ancestor. Raise before output when either is insufficient; otherwise return byte/count/allocation estimates. This check creates no files and reserves no disk, so later concurrent usage can still change available space.

## core/research/capabilities.py::CapabilityRegistry

<!-- implementation-sha256: 0d29e10726d2da86e64c2b27990c456efd3097b487bf5ca5f87b9182b13c0fcb -->

Store uniquely named Capability records. from_spec resolves executable, asset and build-receipt paths relative to a supplied base; require() checks the requested records, optional parameter domains and compiled feature names. signature() hashes the declared binary/features/assets/domain/profile/build-receipt identities in sorted name order. These operations supply workflow prerequisites and reuse identities; registering a record or computing its signature does not run its scientific backend.

## core/research/capabilities.py::available_species

<!-- implementation-sha256: daed63814218d018b9e6cd8a6515afdc65ec0a20d5c5e93fccceca6b24321614 -->

Infer configured analysis species from the request’s supported physics/interaction fields: dark matter by default, gas for baryonic modes, stars with enabled formation, black holes with enabled BH physics and SIDM for its configured interaction. Reject stellar or BH requests without a gas domain. Return a set of labels. These are configuration possibilities, not a census proving that saved snapshots actually contain those particles.

## core/research/capabilities.py::resolve_applicable_products

<!-- implementation-sha256: efbd1c6eb1aa3371d0438d7b818144c544262f1740c55b1493d34ba0f6765ac4 -->

Resolve all or an explicit product list against classify_product decisions for the supplied inventory/catalogue. Deduplicate requests, reject unknown names or all mixed with explicit products, and return applicable selections, missing-input reasons and explicit unavailable decisions. All alone reports inapplicable products through decisions without treating them as explicitly unavailable requests. This plans analysis eligibility; it neither runs producers nor creates missing physical inputs.

## core/observation_data.py::array

<!-- implementation-sha256: b89f04890b74b4153b5c0cf796b31cddd71ef46de327f6f4892159f8c7afbad3 -->

Convert input to a floating NumPy array while rejecting Boolean/complex dtypes, nonfinite values, a mismatched optional dimension count and nonpositive values when positive=True. Return the array with the caller’s name in validation errors. Units and quantity semantics remain the caller’s responsibility; this helper does not convert native h-scaled measurements.

## core/observation_data.py::canonical

<!-- implementation-sha256: 19bf47d216c1c79b95aedef7dad7e725c063fa62a4070c87f1fb9dc20953d52e -->

Recursively convert NumPy arrays/scalars, Paths, mappings and sequences to ordinary JSON-oriented Python structures. Stringify mapping keys and reject extended-precision NumPy scalars that cannot be reduced through item() without losing their explicit representation. Return the converted value. Unlike stream_common.json_safe, this routine does not replace nonfinite floats with None; the downstream digest/JSON policy determines their rejection.

## core/observation_data.py::reject_truth

<!-- implementation-sha256: ff0bf49d5bdb42b1551621492b0d87b754baa15348511f9e2c8286df8229052a -->

Recursively inspect mappings and list/tuple contents, rejecting forbidden truth/background keys and names beginning truth_, native_ or true_. Include the traversed field trail in PermissionError. Return None when clear. This named-field boundary supports observation-only estimation; it does not inspect arbitrary custom objects or establish that every possible information alias is absent.

## core/observation_data.py::observed_catalogue

<!-- implementation-sha256: 723f4e7c5e51f855ba2b4d021c6a010fedaaff99ac1592ed3c178a21b30ae3b0 -->

Build an estimator_observations dictionary from aligned columns with exact unique object IDs, an identity namespace, explicit units and selection/calibration/provenance records. Reject named truth fields and unit strings containing supported native h-scaling markers; validate an optional object-count square covariance. Return canonicalized metadata in memory, without saving a file. Column-specific physical requirements and a more detailed likelihood covariance layout belong to the chosen estimator.

## core/observation_data.py::read_observations

<!-- implementation-sha256: 91d9f67df8afe46eb314d14ced2bd5c6266d277d76efecd9509ec223e09e8b5d -->

Read an observation JSON path or accept an in-memory product, reject named privileged truth and hidden background fields recursively, require role estimator_observations and validate exact object IDs when columns are present. Return the product. This reader enforces the observation boundary; it does not comprehensively validate every method-specific column, covariance or selection rule, which fitting routines must check.

## core/observation_data.py::freeze_result

<!-- implementation-sha256: b5d27d00ba597337e70edf22249242e1c15feca64c290ebee21d4bd7b7572ca8 -->

Create a completed estimator-result dictionary containing fit/observables, selection, exact object identities, namespace, model/nuisance/environment and calibration metadata. Require a named method and a namespace for nonempty IDs, reject privileged truth fields and attach a canonical content fingerprint. Return the frozen payload in memory; scientific acceptance defaults to unresolved and validation receipts start empty. This does not write an immutable directory or certify the fit.

## core/observation_data.py::read_frozen

<!-- implementation-sha256: f924e971988839a3257d99592181de64ea9740cd4d47c325a4e5e7c679c11b33 -->

Read a result JSON path or copy the top-level mapping, require the completed frozen-result schema/role and reject privileged truth fields. Remove the saved frozen_hash temporarily, recompute the fingerprint and reject changed content; restore the hash in the returned dictionary. This verifies a completed estimator artifact for later evaluation, without granting truth access or rerunning its estimator.

## core/observation_data.py::gaussian_observable

<!-- implementation-sha256: 4531bd21403d91084a73c05a49c6241bb32777a0b6ee049f65f2b338233517aa -->

Return a scalar observable record with finite mean, positive finite variance, sigma=sqrt(variance), caller-supplied units and optional redshift. Its default 68% interval is mean±sigma. This summarizes a Gaussian uncertainty convention; it does not estimate variance from samples or guarantee that a non-Gaussian posterior has that coverage.

## core/release.py::portable_native_tree

<!-- implementation-sha256: 117afafb4d67ef8b42e755d47cb0bfdfea8e6871ef152e3dea7c520b30e5291a -->

Inventory portable regular native source/assets beneath a root, filtering configured cache/build directories, excluded suffixes, Finder metadata and validation/release records. Reject selected symlink/nonregular files and control-character filenames; require a nonempty inventory including LICENSE and Makefile. Return relative-name-to-Path bindings without copying files. Compiled native artifacts are omitted by native_file(); traversal does not follow symlink directories.

## core/release.py::tree_hashes

<!-- implementation-sha256: 972d53c5cc9c49a741cb0d2dc6dc74966425009b99f1b536a19020d3d6b37b60 -->

Return SHA-256 values for every relative file selected by portable_native_tree. Inherit its source/license checks and build/cache exclusions. This creates a source-content inventory without copying, archiving or executing native software; omitted files are outside the digest scope.

## core/release.py::build_native_patch_payload

<!-- implementation-sha256: 17499ed93e9bd55cff05a447af506d0fed099c88cd0abf7182e6f74262a00cdd -->

Compare retained pristine and shipped portable native source trees after verifying a full pinned commit and upstream provenance/file hashes. Copy normalized file modes into a temporary comparison tree and run git diff --no-index with binary support. Return full-source.patch and manifest.json as bytes, recording source inventories, deleted files, patch digest and strip count two. Require an actual delta; temporary files are cleaned automatically. This authors a reproducible patch payload without creating a release archive or demonstrating patch application.

## core/formation_atlas.py::select_comparable_histories

<!-- implementation-sha256: aefff15429747a0cbd336f08ed137fa8b82da32224324ef593ecc254dff8c12d -->

Select the largest available group of positive finite final-mass histories with at least two track entries and masses within mass_factor, capped by max_halos. Sort candidates by descending final mass then halo ID and keep the first group when sizes tie. Return the selected history records, possibly empty or shorter than the requested cap. This is a comparable-mass display selection, without establishing matched phases, merger identity or physical convergence.

## core/simulation_queue_index.py::write_queue_index

<!-- implementation-sha256: 6c2dc99536605b05a0310d0b5a17a10dc9c70bbba1aa1a8744d34020e6f824c3 -->

Write an atomic index.html for a simulation queue report and return its path. Build escaped run/status/error cards linking existing linear spectra, saved inputs, result pages and optional analysis records via relative URLs; mark absent targets as not created. Include separate simulation/analysis states and a detailed-report link. This renders an existing queue report without launching, retrying or validating the runs it lists.

## core/comparison_movie.py::make_comparison_movie

<!-- implementation-sha256: cf331c6561ddade34d2884a78f7c89dfc0cf34e845ea9e3fed39df69a5ab02cb -->

Create a paired saved-map movie after resolving exactly two source runs and verifying their comparison provenance under run locks. The renderer requires a complete fixed-scale map series at two or more ordered saved epochs and aligned A/B/ratio grids. Publish paired_comparison.mp4 and movie_contract.json in a fresh directory, showing A, B and log10(B/A) with field units and comoving Mpc/h axes; return the movie path. Every saved epoch receives equal display time, without interpolated states. Matching epochs/seeds alone does not prove matched initial phases.

## core/dome_movie.py::make_dome_movie

<!-- implementation-sha256: 830e28ecf2b3d35f0a390970a4f4f246b52c52a8b54c1753080dfab84e2313bd -->

Project sampled saved dark matter, gas or gas temperature onto a 180-degree equidistant fisheye dome. Require at least two distinct saved epochs and bounded raster/sample/movie/memory controls; keep one global logarithmic display scale across epochs. In a fresh directory publish planetarium.mp4, per-snapshot numeric samples/rasters and planetarium_contract.json, then return the movie path. Scalars are angular mass density [Msun/h/sr] or mass-weighted temperature [K]. Each epoch is sampled independently and shown for equal display time; this projection has no emitted-light or light-cone calculation.

## core/filament_flythrough.py::make_filament_flythrough

<!-- implementation-sha256: 7f66f5c6e5e7b6f2873d75c56ac8fc5a58e897befc07bf5645bbf97011db3e0d -->

Find a camera route toward a selected native FOF host using a periodic CIC dark-matter density mesh and the configured smoothed Hessian ridge thresholds. Verify matching host/snapshot frames and bounded mesh/sample memory, select a connected ridge-cell route and delegate the flight to make_cosmic_flythrough. Publish ridge_route.npz, filament_contract.json and the flight subdirectory in a fresh destination; return flight/cosmic_flythrough.mp4. Route positions are comoving Mpc/h. This grid/threshold-dependent camera path is a filament-like proxy rather than a persistence-filtered skeleton or physical orbit.

## core/multifield_movie.py::make_multifield_movie

<!-- implementation-sha256: 1eed7b78cde56d8bce8b39ff1b8103a8b0ab011e39eda4c3773feb463922a04c -->

Verify a saved map bundle against its source run under a lock, then render two to four distinct fields with common projection, box, slab and window geometry. Require global fixed scales per field and at least two saved epochs; show unavailable panels explicitly and distinguish measured zero metallicity from absent data. Publish multifield_evolution.mp4 and movie_contract.json in a fresh directory and return the movie path. Axes remain comoving Mpc/h and each field retains its mass-density, temperature or metal-fraction units. Saved epochs receive equal display time without interpolation.

## core/lens_map.py::make_lens_map

<!-- implementation-sha256: 818fac90b126562ef5fa75edc85a2eb1a2c1116d917be7a40c426549cd5145eb -->

Read a verified scalar mass-surface-density map at a matching native snapshot epoch. Convert comoving h Msun/Mpc^2 to proper Msun/kpc^2 by h/(a^2·10^6), derive proper pixel size and call thin_lens_from_surface with supplied lens/source distances and background sheet. Publish lens_map.png/PDF, lens_data.npz and lens_contract.json in a fresh directory; return the PNG path. The dimensionless convergence and shear use periodic contrast plus an explicit uniform sheet. This is a single supplied-distance thin plane without a light cone, instrument or noisy source population.

## core/timeline_wall.py::make_timeline_wall

<!-- implementation-sha256: 0eb2c193a5fbdf115dc652623e394b11e2bd4ac7367beb71786aeebe7471b14b -->

Resolve and verify the source run of a saved map bundle under a lock, then build a static timeline from one complete globally normalized series. Require consistent field, geometry and units across at least two actual saved epochs; if needed, select evenly spaced indices up to the panel limit. Publish cosmic_timeline.png/PDF and timeline_contract.json in a fresh destination and return the PNG path. Panels share one color scale and comoving Mpc/h axes; selection does not synthesize epochs or imply equal physical-time spacing.

## core/velocity_flow.py::make_velocity_flow

<!-- implementation-sha256: c906a716378f4de96611649501da6f81ef675fd1d5b2b6f8aa907b3e008b2e5b -->

Stream native dark-matter positions, masses and unit-converted peculiar velocities into projected full-box or periodic-slab cells. Compute mass-weighted two-component mean velocities, excluding cells below min_particles or without mass; add no Hubble flow. Publish velocity_flow.png/PDF, flow_data.npz and flow_contract.json in a fresh directory and return the PNG path. The background is log surface density [h Msun/Mpc^2], axes are comoving Mpc/h and the arrow reference is km/s. Validate shard frames, explicit unit metadata and resource bounds before publication.

## core/results_gallery.py::create_results_gallery

<!-- implementation-sha256: 3adfbe70d9667fe5a5fe2c0660ba1a0b230c45722800a32c86b9162d8fab2ed2 -->

Build a local results portal from one to 500 explicitly selected existing files, with titles, categories, descriptions and optional previews/JSON source records. Resolve paths relative to base, validate allowed media types, hash sources and render escaped searchable-by-category cards. Publish index.html and gallery_manifest.json to a fresh directory and return the index path. The gallery links to original files rather than copying them, so those relative locations must remain available. This navigation layer does not run the listed workflows or certify their scientific results.

## core/stellar_stream_movie.py::make_stellar_stream_movie

<!-- implementation-sha256: d8e32b579722f45f3231d9c31f9e73fbc0bc1638595b5b6c6ac9c413b22b1699 -->

Render exact candidate stellar material from a sealed history.hdf5/provenance bundle. Select one deterministic particle-ID sample shared across at least two saved epochs, verify presence/membership fields and display three periodic projections around a fixed comoving-kpc centre. Publish stellar_candidate_evolution.mp4, per-snapshot numeric samples and stellar_movie_contract.json in a fresh directory; return the movie path. Colors distinguish saved subhalo membership, outside all subhalos and missing catalogues. These candidate particles are not automatically a tidal stream, and outside SUBFIND does not mean unbound to the host.

## core/particle_origin_movie.py::stable_cohort

<!-- implementation-sha256: 78efd7e67f0013af2411fba75711008e7cc39645f3c429d6ab9169f82cca1dfa -->

Validate a nonempty exact unique uint64 ID vector and a positive integer display cap. Select up to maximum IDs by deterministic 64-bit hash with ID tie-breaking, then return the selected IDs in numeric order. This preserves a fixed display cohort across epochs without random state; it does not identify persistent objects or match separate simulation namespaces.

## core/particle_origin_movie.py::make_particle_origin_movie

<!-- implementation-sha256: 9d521029d7c91c68bda1d07f22a2504652e9817afa6f841fad3858ee1e8c2c04 -->

Select a deterministic subset of one anchor FOF host’s exact type-1 member IDs and locate those same IDs across the original run’s saved snapshots. Verify run/IC lineage, box, epochs, unit conventions and source hashes; retain matched counts per epoch. In a fresh directory publish particle_origin.mp4, tracked_particles.hdf5 and origin_contract.json, returning the movie path. Three full-box projections use comoving Mpc/h and equal display time per saved epoch. This demonstrates same-run material continuity, without establishing persistent halo-core or cross-run identity.

## core/resolution_reveal.py::make_resolution_reveal

<!-- implementation-sha256: b78cd75caef9ec20de6522f465b57f2f8d9edf480d660bf7a3465c9c540c5bfa -->

Compare two verified saved map grids with matching field, projection, box, measurement units, slab geometry and epochs within the supplied tolerance. Derive one shared logarithmic display range and publish run_a.png/run_b.png, resolution_reveal.png/PDF, an interactive divider HTML and resolution_contract.json in a fresh directory; return the HTML path. The comparison displays complete periodic grids even when a source figure had a crop. It does not independently verify common initial phases, identical physics or numerical convergence.

## core/halo_phase_portrait.py::export_phase_portrait

<!-- implementation-sha256: 2aab40e10db86aa36703e67ceb67c72f614c4c9b1a4270ee0fa1d2f0f97d43fb -->

Render an existing halo phase-space histogram as a logarithmic radius/radial-velocity count map, using r/R200c and peculiar radial km/s with no Hubble flow. Export the figure and its metadata through export_figure, save the supplied portrait arrays beside it as .phase_data.npz and return path; a None portrait returns None. The caller supplies populated positive count bins and the measured weighted-mean bulk convention. This function plots already estimated counts rather than extracting particles or calculating the histogram.

## core/select_merger_tree.py::export_selected_tree

<!-- implementation-sha256: 690a04b971745014b661e16894a829808b56b45a0907238467336e92b1051d4c -->

Select an exact nonnegative decimal halo ID from a source-verified analysis bundle belonging to the requested run. Reconstruct its available main-progenitor tree using saved snapshot metadata, require a complete tree rooted at that ID and delegate tree rendering. Publish the tree assets and selected_tree_contract.json in a fresh destination, reverify sources and return merger_tree.html. Unavailable progenitors raise rather than producing an empty successful tree; catalogue links and particle-floor support retain their measured scope.

## core/halo_anatomy.py::make_halo_anatomy

<!-- implementation-sha256: c008f6ea6c3bbc90f2acd988cce693d8251dec3133db93a94a9c09b50620d3b8 -->

Build an interactive dark-matter cutaway around a measured host: equal-thickness cube slices extending ±2 R200c, shell density and 3-D mass-weighted peculiar velocity RMS, plus available catalogue subhalo centres. Read all relevant type-1 particles in chunks, enforce frame/memory/aperture bounds and omit dispersion in shells with fewer than ten particles. Publish slice PNGs, radial_profiles.png, anatomy_data.npz, halo_anatomy.html and anatomy_contract.json; return the HTML path. Coordinates are comoving Mpc/h, cell masses Msun/h and shell density is divided by a^3. Shells include neighboring/unbound material; this is not a bound-members-only profile.

## core/halo_fingerprint.py::common_aperture_histogram

<!-- implementation-sha256: 81bec4fdde5d39cad6100a68b6ffc4a7efc7bc8463764445c405ae0f9ebf29cc -->

Apply a periodic minimum-image offset around center, select particles inside a cube of half-extent extent and return its XY weighted mass histogram, radial-shell masses and shell counts. Validate finite aligned N-by-3 positions and positive masses; box/extent/grid/edges controls are supplied by the caller. Outputs retain input mass/length units. No pixel-area, shell-volume or scale-factor normalization occurs in this helper.

## core/halo_fingerprint.py::make_halo_fingerprint

<!-- implementation-sha256: 2fb42115a61012379b08f9b78fbb978570747b58ef1d9e7fc7503cdc4a2bda85 -->

Match a selected FOF host across two runs by unique reciprocal-best exact dark-matter ID overlap after checking seed/lattice/box declarations and, for distinct runs, measured initial-phase agreement. At matching saved epochs measure host-centred maps and shell densities using one common comoving aperture, including all local type-1 material. Publish fingerprint maps/profiles/data, halo_fingerprint.html and fingerprint_contract.json in a fresh directory; return the HTML path. The B−A image is a surface-density difference rather than significance; same-run controls and cross-run phase scope remain explicit, without claiming convergence or causal isolation.

## core/halo_collision_movie.py::make_halo_collision_movie

<!-- implementation-sha256: b7f61c0a014484c450440bafff9037148bdac1a4b462e7d6af444a4715d53cb0 -->

Select two linked progenitor branches from a verified merger tree and overlay their saved positions on matching globally normalized maps from the same run. Require at least two usable pair epochs with consistent projection/slab geometry and both halos inside any selected slab. Publish halo_collision.mp4 and movie_contract.json in a fresh destination, showing periodic pair-centred maps, measured separation versus scale factor and available peculiar-velocity arrows. Return the movie path. Equal display time, connected saved points and display-scaled arrows do not establish a continuous orbit, core identity or collision causality.

## core/metallicity_maps.py::make_metallicity_bundle

<!-- implementation-sha256: 1555f411b61467d0b98a028b18b4c7478c4f7ea93b36871e1fdae24217893875 -->

Extend a verified structure-map bundle with gas metallicity on the same saved-epoch geometry. Read native gas mass and dimensionless saved metal fractions, requiring consistent populated shards; compute periodic CIC sum(mZ)/sum(m). Preserve measured zeros separately from absent-gas NaNs and record missing epochs. Copy original bundle assets, add numeric HDF5/figure metadata, update global scale.json and publish a new map_contract.json in a fresh directory. Return bundle/key/output/missing records. This creates numerical map products for downstream renderers; it does not invent enrichment where no metallicity was saved.

## core/music/boundary.py::verify_boundary_adapter_build

<!-- implementation-sha256: 484578c54fecfbd44404356593362727a7b4a827da58978d38e1e79c7d1c6dc6 -->

Verify a boundary-adapter build receipt against its adjacent checksum, expected schema/BUILT state, ABI version one and eight-byte scalar format. Rehash every recorded source and the adapter binary, rejecting changed assets. Return the resolved binary path and receipt dictionary. This validates the retained build identity; it neither compiles/loads the adapter nor measures boundary-force accuracy.

## core/music/flow.py::validate_flow_noise_request

<!-- implementation-sha256: d2b5e7c2d195adf948e6f4c59a6bda3bbfa0d2333b45a325d3cb59af09d62992 -->

Validate the full-flow coupled-noise request schema and required nonempty configuration/IC/mesh/noise paths. Require a checksum-bound boundary-adapter receipt and explicit bounded restart/maximum-iteration integers; validate optional publication disk controls. Return None or raise before execution. Paths are syntactically declared here; byte integrity and compatibility are checked by flow_noise_input_bindings.

## core/music/flow.py::flow_noise_input_bindings

<!-- implementation-sha256: 87c3b0d25e497423529f65693207c774c1bdcff1f89c43216eb9f9754b99b63f -->

Verify a successful original full-flow window-response product and its source bindings, plus the requested sealed boundary-adapter build. Require configuration, IC, mesh and every convolution-level native noise cache to match the original response’s hashes. Return combined pins for product/seal/arrays, adapter and source files, rejecting conflicting bindings. This establishes exact inputs for subsequent response calculations without running a new MUSIC solve.

## core/music/flow.py::validate_original_flow_launch_request

<!-- implementation-sha256: eb808f3325a583a70fbff387e525a2dc506d1914c0e97d87a2475d36e81ae796 -->

Validate the nested original-flow launch controls by composing the existing window, coupled-noise and covariance request validators. Require explicit combined source/noise publication workspace when requested or needed for fixed-prefix conditioning, and mode vectors spanning all three spatial axes. Return None or raise before output. Schema validation preserves numerical tolerances; it does not turn an unexecuted noise-model declaration into verified launch evidence.

## core/music/poisson.py::read_failure_snapshot

<!-- implementation-sha256: 56489a3955a75fa1044835edaafd5c6edaddaa8403c5fa670074493f51a32a52 -->

Read a bounded native MUSICPS1 Poisson failure snapshot with versioned little-endian metadata and contiguous level geometry. Validate level/halo ranges, positive tolerance, payload lengths and absence of trailing bytes, then expose each level’s original source, returned source and potential as read-only float64 memory maps. Return the path, reported error/tolerance and level records. This preserves failed-solver evidence for diagnosis; it does not rerun the solve, assert finite field values or relabel failure as success.

## core/music/precision.py::Binary128Array

<!-- implementation-sha256: 051cf6a1b469a022a6073573f2a997dd59f22582726ff34707d76ce7cde3e193 -->

Store a three-dimensional finite IEEE754 binary128 grid as immutable uint64 word pairs with an explicit little/big word-order convention. Reject NaN/infinity encodings and implicit NumPy conversion. words() returns a read-only view, bits() preserves one cell’s exact representation and fraction() decodes its value as an exact rational, including subnormals. This avoids assuming NumPy longdouble is binary128; arithmetic requires the explicit native adapter rather than a silent precision fallback.

## core/music/precision.py::wide_derivative

<!-- implementation-sha256: d80201a8a4314b4601a0a5ce5eba883975eac06196ce0f40ebb592e0b1cc9629 -->

Apply the explicit native binary128 reference to a Binary128Array with real retained ghost cells. Validate level/order/halo and memory controls, then compute a gradient, negative Laplacian (poisson) or quadratic Hessian source on the interior. Native arithmetic uses the dyadic computational-coordinate scale 2^level and rounds only final observables to float64, rejecting invalid/out-of-range results. Return the scalar or three-component interior array. Loading the adapter may build/reuse its external-runtime reference library; conversion to physical units belongs to the caller.

## core/music/precision.py::wide_dot

<!-- implementation-sha256: c7766de324cb06416d5ee676f15332eb151987596d930e9b961818e783390e93 -->

Compute the weighted sum of an original Binary128Array with an aligned real weight grid, within the declared memory budget. Convert weights explicitly to contiguous float64 and call the native binary128 compensated-sum reference, rounding the final scalar to binary64 with range/finite checks. Return that scalar; errors propagate from the adapter. The underlying field words are retained rather than converted, and adapter preparation may build/reuse the external native reference.

## core/music/seeded.py::validate_seeded_covariance

<!-- implementation-sha256: 714ec97b0d6629ad84e5a190f1663cacc30f162e30dca73417c11b5bd470b65f -->

Require exactly disk_gib and chunk_cells for local seeded covariance work: positive finite numeric disk allocation and an explicit positive integer cell chunk. Reject Boolean allocations and malformed settings. Return None on success. This validates controls without reserving storage, drawing innovations or estimating a covariance.

## core/music/seeded.py::verified_seeded_window_moments

<!-- implementation-sha256: 3a058cd66d56618a44d82def318a775b84d2021bdbd73d51d3dc20f9843a411b -->

Compute observable means/covariance for a verified local seeded model with explicitly fixed retained prefix. Require matching hierarchy geometry, successful model checks and hash-identical native caches, then stream response rows through fixed-prefix moment accumulation within memory/disk/chunk budgets. Create the owned innovation/moment workspace and add its hashes plus implementation bindings to pins. Return mean, covariance, cache bindings and metadata. Overlapping cases retain shared innovation identities; this routine computes per-hierarchy moments rather than a joint covariance across separate cases.

## core/music/staging.py::generated_flow_bindings

<!-- implementation-sha256: a670006c15cab5255e5daf3bb2d326d17138370f5f7d2795f6d0c91958d84d26 -->

Validate a generated-flow preparation request and its checksum-bound physical answers template. Require exactly one existing noise-model verification or explicit local-model-generation branch; the latter also requires fixed-prefix conditioning, seeded covariance settings, verified GSL and boundary-adapter assets. Validate reusable launch controls with an intake-only window and return input/code hashes, delegating existing-model bindings as needed. No ICs or runtime proof are generated here: the temporary future-proof identity used for intake is never scientific evidence.

## core/music/windows.py::validate_compensated_window_request

<!-- implementation-sha256: 0dadc69c3714eec1d66d069cf0685cfa455a04b6375d250b7ddbc20772816603 -->

Validate a compensated-window request with mesh/noise paths, exactly representable integer modes and nonempty uniquely named windows. Accept either all_fine or explicit nonempty integer computational-cell boxes with ordered bounds. Return None or raise. This checks declarations before response generation; actual mesh coverage, noise files and measurements are validated by the producing workflow.

## core/music/windows.py::validate_flow_window_request

<!-- implementation-sha256: 7657cd689ebaf36c29c4e745af260f16168e3b32baf8fc1daf7d44efaff562fd -->

Validate full-flow window controls: mesh/noise paths, unique density/velocity quantities, exactly representable integer modes and distinct safe window IDs. Each window selects unique original matter species from types 0/1/2 and either all or explicitly listed dyadic levels 2–20. Validate optional response disk allocation and return None or raise. These selectors describe the response to compute; they do not establish that a saved mesh contains the requested species/levels.

## core/music/covariance.py::validate_window_covariance_request

<!-- implementation-sha256: 32ef8a91a94bc70eb91f29953aa85a548329d9f663e50ec1c06b369ecf72f56e -->

Validate a compensated-window covariance request with a checksum-bound noise-model verification and explicit parent conditioning: fixed, ensemble or fixed_prefix. Require seeded covariance workspace settings only for fixed_prefix and reject them for the other modes. Return None or raise before reading a model. This preserves the distinction between conditional and ensemble moments without drawing fields or fitting covariance.

## core/music/covariance.py::window_covariance_input_bindings

<!-- implementation-sha256: 812eef7638c6bcfd41591b2f9f85c49be625a5880b757cb6e00d1137d212e1d3 -->

Validate the covariance request, then verify a successful compensated-window response and supported successful noise-model product, including numeric seals, archives and source hashes. Reject contradictory shared provenance and return the combined input pins. This establishes the original response/model identity; the following moment calculation checks actual cache values and conditioning.

## core/music/covariance.py::verified_window_noise_moments

<!-- implementation-sha256: cd678d1e90c73a5b4e86dbbf0183f56e5e1ba5cd4e706210e6a867dbb0e7b2cb -->

Compute moments from a supported verified noise model for the original response hierarchy. Delegate local fixed-prefix models to the streamed seeded routine with an owned workspace. For full-frozen models require disjoint verified innovation streams, match each actual cache bitwise to its original full-field coordinates and build innovation-response weights. Return mean/covariance, cache bindings and algorithm metadata, conditioning on the base field for fixed or including its ensemble variation otherwise. These finite-model moments do not establish continuum boundary accuracy or Gaussian acceptance.

## core/music/covariance.py::validate_flow_covariance_request

<!-- implementation-sha256: 9ee54c54f03c2b2a7558857f57d5b76bfd578963b4a9e9f3ed9d2a6d5c2488f7 -->

Require the full-flow covariance schema, then validate its remaining controls through validate_window_covariance_request with the corresponding compensated schema substituted. Return None or propagate validation errors. This reuses the same explicit noise-model and parent-conditioning rules without accepting a compensated response in place of a full-flow one.

## core/music/covariance.py::flow_covariance_input_bindings

<!-- implementation-sha256: 39c385f070dc81d5f95c3872c90faf4d1774a78dd9d4b7e5ede0ddc60b3c124d -->

Validate the full-flow covariance request and verify a successful native-music-full-flow-noise-response product together with its supported verified noise model. Return merged source/product/seal/archive hashes, rejecting changed assets or contradictory provenance. This input binding precedes covariance evaluation and performs no new native solve.

## core/music/covariance.py::publish_flow_window_covariance

<!-- implementation-sha256: 673b0a190070b65fba647da693a3e259561703524817eb67717a5bfc4495f3fa -->

Read a verified full-flow noise response and model, compute the requested conditional/ensemble finite moments and retain observed real channels with their per-channel physical units. Stream fixed-prefix rows and publish a sibling innovation workspace when required. Export mean, covariance, correlation, eigenvalues and centered observations through the numeric audit publisher. Mark MOMENTS_MEASURED only when finite positive variances and the dimensionless correlation condition bound pass; otherwise publish UNRESOLVED_COVARIANCE with missing correlation values as appropriate. Return the publication result; no Gaussian acceptance, physical boundary certification or evolution launch is authorized.

## core/music/resources.py::native_boundary_count_profile

<!-- implementation-sha256: eb789efca2153198e636547e34bd9cbfafdc0b5defd3497a6c85a4cc488e9532 -->

Load a verified native boundary adapter and invoke its symbolic operation counter for order two, four or six. Return cubic ghost-cell and face-pair node counts, endpoint policy and build/binary/source identities. This calls the compiled count routine, not a Poisson solve; the counts describe arithmetic/allocation structure rather than measured runtime or total process memory.

## core/music/resources.py::coupled_geometry

<!-- implementation-sha256: fcec1d65b900d085ba35e61a226ceba3e286e2cb172adf8b569684048f5b8aaf -->

Validate a nested native hierarchy with consecutive levels, even bounded interior shapes, aligned integer origins and four parent interior cells of coupling padding. Require the coarsest level to be a whole periodic cube at origin zero. Return parallel lists of levels, shapes, origins and parent-relative offsets. This geometry check allocates no field arrays and does not test boundary accuracy.

## core/music/resources.py::coupled_workspace_plan

<!-- implementation-sha256: 380f06edf132f0f04829d671abe20bc9c7ae102186ae2881c2e432cfd1d8ae1e -->

Estimate coupled Poisson construction and solve workspace from validated hierarchy geometry, bounded Krylov restart/iteration controls and compiled native boundary operation counts. Return per-interface allocations, base/Krylov workspace, separate construction/solve totals and their maximum in bytes. A single periodic level needs no boundary counter. This plan performs no field solve; allocator/FFTW process memory remains uncalibrated and is outside a guaranteed RSS bound.

## core/music/resources.py::noise_response_resources

<!-- implementation-sha256: 185625cc517961b32dfd74e968f8ba571eca1a9b4076854bd233e3cfd5324d69 -->

Estimate retained noise-response storage from the actual convolution hierarchy, a positive channel count and explicit source/metadata byte counts. Require a positive even convolution margin, include padded refined cache shapes, and return input residency, numeric payload/array counts and a conservative response publication disk bound. Coupled-operator workspace is additional. This computes sizes without creating response arrays or reserving disk.

## core/music/resources.py::hybrid_workspace_requirements

<!-- implementation-sha256: 15f557ee0c9d80721fda1bb5f798a85f8ddb2854cf6281736170729f1d658e57 -->

Validate bounded second-order source settings against a binary64 native Poisson hierarchy with a full supported periodic base. Estimate captured-source and extended-FFT workspace, compare their maximum and extended side with declared limits, and return a hashed geometry/settings/source report. Status distinguishes feasible workspace geometry from insufficient allocation. These native resource inequalities preserve numerical tolerances but do not execute generation or establish actual peak memory.

## core/music/resources.py::rectangular_particle_census

<!-- implementation-sha256: 3fead6814899401b1537a975587557d35cdf1bc3a53faea2f4e3a06f92e7254b -->

Compute the exact particle count implied by a native unmasked box/Arepo hierarchy, reserving type two for combined coarse matter. Validate configuration levels, even integer nested geometry and whole-parent-cell containment; count leaf cells and require exact integer periodic-volume closure on the finest lattice. Return fine dark-matter, optional fine-gas and coarse counts plus configuration/geometry/source hashes; unsupported region/output routes return UNSUPPORTED. No particles are emitted, and actual IC headers remain the runtime census authority.

## core/music/resources.py::publish_generation_capacity

<!-- implementation-sha256: d48671ddac5bfcbb36c67bbf8759f265abe3ae74242a5214b9c5e8e172640d3e -->

Read hash-bound configurations/meshes and a verified boundary build, then estimate sequential MUSIC generation plus original-window handoff memory/disk requirements for each case. Optionally include explicitly assumed aggregate evolution memory, snapshot/checkpoint storage and necessary particle-index rank floors. Verify noise/dependency hashes and write a fresh JSON report with ESTIMATED or REQUIRES_LARGER_ALLOCATION status, retaining all-case disk totals. Return the report. This plans resources without granting hardware allocation or evolution launch; catalogues, downstream figures/rendering, numerical acceptance and measured peak usage remain separate.

## core/flagship_launch.py::prepare_launch

<!-- implementation-sha256: c61dccb67313d99aa4b1a5f4dff97cf57c557009147b89576eb15669792be4d8 -->

Materialize the selected caller-authored launch profile, including campaign/portable/connected variants, and freeze its resource allocation and input/source hashes. Build or reuse explicitly selected native engines, compile/preflight each operation and write operation.workflow.json files plus sealed launch.json in a fresh destination. Production campaign preparation records static-DAG scope where runtime admission is still required; other paths use workflow dry-run checks. Return the PREPARED record. Preparation may create sibling input assets and build products, but does not execute the miniature or requested scientific workflow; failures retain preparation-failure.json.

## core/flagship_launch.py::load_launch

<!-- implementation-sha256: 37c7439f1bcd06704d694843028467593df51ec03b160860f6a97b0e52d9fab9 -->

Read a prepared launch directory or launch.json path, require the prepared schema/status and verify its launch.sha256 seal. Rehash every bound source and operation workflow, rejecting changed inputs. Return the manifest path and launch record. This loads an existing preparation without building engines, checking actual hardware allocation or running its stages.

## core/flagship_launch.py::execute_launch

<!-- implementation-sha256: 931d69198c365b3f1428b31a90df88af371d5f9cd155dfb29cf82ee979bbf7fd -->

Verify a frozen launch and select test/run/postprocess/render, allowing test only for the miniature profile. Resolve retained source-workflow references, preserve the exact launch/operation/allocation binding on resume and invoke execute_workflow with the frozen request base. A dry run delegates planning without writing a launch binding; actual execution writes the binding and publishes a verified run report, including on workflow failure when possible. Return scheduler and report status separately; missing required connections can make a scheduler-complete run INCOMPLETE. Original workflow exceptions remain authoritative if reporting also fails.

## core/flagship_launch.py::main

<!-- implementation-sha256: 4af0288ec39c7b5863c788e9fb49e13ae18164fbcd975e366ac69c610449a9b9 -->

Dispatch the frozen-launch CLI: prepare explicit request settings, inspect status, execute an available operation, or resume the operation recorded in its output binding. Accept resource overrides during preparation, optional campaign/downstream requests and retained source-output references. Print the result JSON. Return zero for preparation/status/dry-run or a COMPLETE executed result, otherwise one; uncaught validation/execution errors propagate. This adapter supplies no implicit bundled simulation preset.

## core/flagship_parent_preparation.py::validate_parent_preparation

<!-- implementation-sha256: 03e928d93269ad963abfb37c39b56b4f9653d94d7ef8d52326de3a5111d1fc14 -->

Validate the native parent-preparation schema, safe prefix, explicit hydro answers and fixed hierarchy settings. Require parent level 5–20, positive integer seed and second-order LPT, with answer box/resolution/seed matching staging. Check positive execution allocations, exact generator/admission keys and a continuation dictionary or null. Return None or raise. This validates request structure before parent_preparation_inputs checks the actual solver bundle, binary and admission evidence.

## core/flagship_parent_preparation.py::parent_preparation_inputs

<!-- implementation-sha256: 4937334560a24e8dc5ed710a6ef89d854150cd6e79f2a7aee9793be40379c10f -->

Bind a parent request to an actual completed CLASS public-command stage that owns its linear bundle. Verify bundle hash/geometry, stage payload/source/input hashes, exact generator/admission references, optional frozen parent noise and native MUSIC mixed-fluid second-order validation. Rehash accumulated pins and return them. This proves which retained inputs/admission receipts the preparation consumes; it does not generate parent ICs or evolve a parent run.

## core/flagship_parent_preparation.py::parent_preparation_runner

<!-- implementation-sha256: 9d0ab89751ca9127a1dd6e6abb7d7bf064bb733c08a9656f75d000eeef12b677 -->

Validate the prepared parent’s inputs against workflow resources, stage same-level MUSIC parent configuration/assets and write answers.json referring to the verified linear bundle. Construct follow-up public-command stages for zoom generation and evolution-only GADGET, including checkpoint policy and exact generated-IC/report references; resolve optional continuation bindings to their actual parent outputs. Write parent-preparation.json with RECIPES_PREPARED and return follow-ups/measurement metadata. The runner prepares these stages rather than executing native parent evolution itself.

## core/flagship_native_branches.py::native_branch_inputs

<!-- implementation-sha256: 8602c6bdf049d8284527296c46808a4674df808014b7655ff014fc492fcc0454 -->

Read one owned native history case and its checksum-bound branch design under explicit memory/metadata limits. Require complete ancestry/assembly/cooling/web controls, positive graph/measurement census bounds and cadence; bind optional cooling/web design files. Return the selected case record, parsed design and combined source pins. This validates branch inputs without creating ancestry graphs or measuring histories.

## core/flagship_native_branches.py::native_branch_runner

<!-- implementation-sha256: 34dd05f8b76e2f51fdbaacf53c5830d2c8d7a98e4ec3f98bb909441909b5fa3f -->

Analyze saved epochs from one owned native run, requiring its original recovery stage to cover that exact epoch set. If final target recovery is ambiguous, publish UNRESOLVED_TARGET without choosing another host. Otherwise construct native FOF ancestry, attach supported structure observations and measure the assembly history; optional branches add recorded stellar-birth/cooling and multi-epoch web measurements. Publish products/branch-result.json with original graph/assembly references, source pins and unresolved records. Return its result. These measurements consume evolved native outputs; no new particle evolution or unsupported host substitution occurs.

## core/flagship_retained.py::retained_operation_inputs

<!-- implementation-sha256: e865d3cbcb26467edfd9b81ba0b4ffc45165e42532fe428e0d4ec4d0e5669a43 -->

Verify a hash-bound COMPLETE original workflow for retained postprocess/render operations within memory/stage limits. Require every stage generation/receipt/fingerprint to remain owned by that workflow, verify its sources and artifacts, and locate exactly one original runtime measurement preparation. Return that preparation, completed receipt mapping and merged pins. This establishes reusable completed inputs without rerunning measurements or rendering.

## core/flagship_retained.py::retained_operation_runner

<!-- implementation-sha256: b0b478aceda2083dda52ec6ea6d822962c333d402cce26ea6409fe468d228313 -->

Prepare a retained operation from the verified original workflow. For render, resolve completed case references into the original display continuation; for postprocess, prepare a fresh measurement stage whose continuation uses that display request. Assign safe stage IDs, bind all retained inputs, write retained-operation.json and return the follow-up recipe plus source-stage count. Execution belongs to the workflow scheduler; this runner does not claim those new measurements or renders have finished.

## core/flagship_run_report.py::publish_run_report

<!-- implementation-sha256: 84f1a4bb732f5117895d745a10d08ebc9ef4c04cd6ed1456adcf1b1d4b9a7f1d -->

Build run-report.json and its checksum from verified completed stage receipts owned by a workflow. Within metadata bounds, inventory native saved outputs, audit/support records, all-102 graph galleries, scenes/renders, unit/frame/identity contracts, storage and sampled resource coverage. In connected modes, require the expected producers and declared scene/render follow-ups; downgrade scheduler COMPLETE to INCOMPLETE when connections are missing. Return the report, preserving unresolved measurements and scientific exclusions. This summarizes retained evidence without rerunning science, visually inspecting figures or certifying machine-wide peak memory.

## core/scene_render.py::validate_render_request

<!-- implementation-sha256: dd2c49cb59c268b7880b14dd1fd33582ed0fb62f76872f1e888bec36ae63c994 -->

Validate a complete saved-scene render request: exact sealed scene reference, positive integer raster/sampling/frame/FPS controls, ordered frames, explicit projection/device and positive memory/disk allocations. Require square dome output and a scene name or active selection. Reject raster or conservative PNG-sequence estimates exceeding allocations. Return None or raise before loading Blender data. These provisional bounds exclude scene geometry and do not guarantee actual renderer RSS.

## core/scene_render.py::render_saved_scene

<!-- implementation-sha256: 521456e34bde09fa8035e9a1cf6daf318bc5214cc350d11068878ed9628a24ed -->

Run inside Blender to verify and open a sealed original scene with automatic scripts disabled, then register only the application’s named packed inspector/display runtimes when present. Select the requested saved scene and unambiguous perspective/dome/active camera within saved frame coverage. Configure Cycles/device, sampling and 16-bit RGBA PNG output; render the requested frames to a fresh destination, enforce actual output disk usage and reverify the source publication. Write sealed render.json and return its receipt, retaining failure.json on error. It renders saved geometry/clocks/identities without saving altered source-scene bytes or establishing scientific resolution.

## core/scene_render.py::main

<!-- implementation-sha256: 120d12215388f8a663148867e4b11fe638cf13166eb0dde01afefa1015d4c99c -->

Parse a render request JSON and required --output path from Blender’s arguments after -- when present, then invoke render_saved_scene. This adapter runs in the Blender Python environment where bpy is available. It returns no explicit value and lets validation/render failures propagate; it does not prepare a scene or launch Blender from an ordinary Python process.

## core/sidm_campaign.py::validate_sidm_resources

<!-- implementation-sha256: b44fbe17e02ad013d737d3df1940712f2bfd9b8467f71b1fbdec2ec483391840 -->

Validate an actual native SIDM resource-log request: positive maximum log count and expected MPI ranks, bounded distinct event files, positive metadata memory and unique finite increasing native-coordinate intervals. Reject ranks beyond the native event identity capacity and metadata exceeding allocation. Return None or raise. These are supplied completed-interval/log controls, not an elapsed physical-time conversion or a whole-run cost prediction.

## core/sidm_campaign.py::native_sidm_resources_runner

<!-- implementation-sha256: 8a76444e760599abcd8f92400fe3883fbd7458dc649794796ab6fb5681a7260b -->

Enforce that the resource decoder’s requested memory fits its workflow allocation, measure the supplied native SIDM resource records and write native-sidm-resources.json in the stage generation. Return measured quantities together with explicit scientific exclusions for unmeasured items. This decodes retained native evidence; it does not launch a SIDM simulation or extrapolate a machine-wide peak from partial logs.

## core/sidm_physics.py::sidm_physical_model

<!-- implementation-sha256: 92151c28bd8761e1f9d5c72d07e280b4796d407e3c19c48bccbbf18e17e545be -->

Read the exact hash-bound physical angular-scattering model within a 256-MiB metadata cap. For sidm-elastic-model-v2, validate through ElasticSIDMModel and require the requested particle convention. Legacy supported isotropic/screened-Rutherford/Born-Moller models require their applicable velocity scale and approximation/spin/normalization rules; Born-Moller additionally requires folded identical particles. Return the model and its role/path/hash/size record. This intake validates a declared kernel model without simulating collisions or measuring transport observables.

## optional_tests/build_assets.py::markdown

<!-- implementation-sha256: 6bc26454ecc85f7dc6f863b6cf55e315b9abf24320a718653c510fa9801eb1a4 -->

Return a minimal Jupyter Markdown-cell dictionary from text, preserving source line endings and starting with empty metadata. This only constructs a cell in memory; build_notebook assembles and writes the notebook, and no Markdown is rendered here.

## optional_tests/build_assets.py::code

<!-- implementation-sha256: a4be786e64de120ac87b5269cb1b4dc3f1a2f0e6e6bf461379f4273e21144d58 -->

Return an unexecuted Jupyter code-cell dictionary with preserved source lines, empty metadata/outputs and execution_count=None. This authors notebook content without running the supplied Python. Notebook execution and verification are separate steps.

## optional_tests/software_checks.py::main

<!-- implementation-sha256: f24cf8496faaaa7a027cddeb60ac4658ba4fed9fced444c99192f22e7f45acae -->

Run one explicitly selected optional Python-backend calculation with a required seed in a fresh output directory. Write receipt.json with PASS and calculate() outputs, UNAVAILABLE for ImportError, or FAIL with traceback for other exceptions; print a compact status. Exit one only for FAIL, so a zero process code must still be interpreted alongside the receipt’s availability state. This worker executes a bounded backend calculation rather than claiming success from an import alone.

## optional_tests/test_lab.py::LocalLabTests

<!-- implementation-sha256: f3b6f0b7625fda8d6048f93262c811ed0a238d732be2a517e598dd15b5d162cb -->

Contain the optional lab’s focused unittest checks for source-tree output rejection, unavailable-backend receipts, call-trace identity, continued reporting after worker failures, confined paths, truthful live-versus-archived inventory and loopback/token/origin handling. Use temporary directories, mocks and a local HTTP server where appropriate. Run through python -m unittest discover -s optional_tests -p test_*.py -v. These checks exercise serving/reporting boundaries; numerical/native workflow evidence remains in separate lab runs, and documenting the class does not execute its tests. Also verify that reviewed equations/units appear in the source map without granting execution status and that changed implementation bindings are rejected before the catalogue is served.

## scripts/serve_results.py::ResultsHandler

<!-- implementation-sha256: 073d2471d7b5c02b15fd8b011021c82f023909f1bb7c55f6d648d4a98f4f5b61 -->

Extend SimpleHTTPRequestHandler with single byte-range file responses for local media seeking. Parse explicit/open-ended/suffix ranges, return 206 with bounded Content-Range/Length for valid requests or 416 for invalid intervals, and stream only that byte span. Other requests use the standard handler. This serves files from the configured directory; it supplies no simulation execution or scientific validation.

## scripts/serve_results.py::main

<!-- implementation-sha256: 05cb3b22c73a2570946bfc59ee530a08686e237b0fd183df8a2d566cdb83267b -->

Serve an existing results directory on 127.0.0.1 using the range-capable threaded ResultsHandler. Accept --port and optional --open, print the START_HERE.html URL and optionally launch it in the default browser. Run until interrupted, then close the server. This is a local atlas viewer; it does not create the expected entry page or verify its assets.

## scripts/verify_results.py::Links

<!-- implementation-sha256: 43e674b1a00f0b7d226842301a1b4119185064afcedffd08213195c6c6168505 -->

Collect nonempty href, src and poster attribute values from HTML start tags into a list. This lightweight HTMLParser subclass does not resolve URLs, check anchors, execute JavaScript or discover arbitrary dynamically constructed links; audit() handles the collected paths.

## scripts/verify_results.py::audit

<!-- implementation-sha256: 18134c6a974c3047fde52723f7943e0befca79d8fa68cf890437fa8c761630d1 -->

Audit an existing single-run atlas under root/analysis. Check local HTML asset paths, decode linked raster images and sample the first/middle/near-final frames of linked MP4s; include known dynamically selected halo-story map PNGs. Optional base_url adds concurrent HTTP HEAD length checks. Write analysis/06_Data_and_provenance/Loading_and_media_audit.json, print a compact summary and return whether no failures occurred. This expects that atlas layout and report parent already exist; sampled decoding and link checks do not replace visual review or full-video decoding.

## scripts/verify_paired_assets.py::Assets

<!-- implementation-sha256: 8b9cb39152d767ab552f729b61a7818af3835ca698446e19cfdeb3eaf161436a -->

Collect href, src, poster and data-film attributes from HTML into a unique reference set. This parser supports the paired-atlas asset audit but does not render a page or discover all JavaScript-generated requests.

## scripts/verify_paired_assets.py::public_path

<!-- implementation-sha256: 11786559fed8b4db1232ccd80604aae91f93059c31690cb7c7afa6cfcb6795a3 -->

Classify a path relative to the supplied atlas root as public when none of its components starts with a dot and its top-level section is not Source. Return a Boolean; paths outside root raise through relative_to. This is an inventory filter, not an HTTP access-control policy.

## scripts/verify_paired_assets.py::main

<!-- implementation-sha256: bf22c6dae7ebf56502884910bae9e96e4d5009f2f05d5459c361cba71442ad92 -->

Audit a paired atlas’s public HTML references and linked image headers, then inspect public movies using ffprobe and verify 64-byte HTTP range responses from the required --origin. Require H.264/yuv420p; optionally create absent JPEG posters with FFmpeg. Write 07_Status_and_Provenance/asset_audit.json and print counts. Return True on failures, which the CLI uses as exit status one. This depends on an already serving atlas and existing report layout; image/header/range checks do not constitute visual QA or complete movie decoding.

## scripts/verify_paired_measurements.py::main

<!-- implementation-sha256: af0b75f2e06a83311bc0a6969b90925a5c4bcc2d29153fcc54e4f488a6e9c358 -->

Verify the paired atlas’s EDE and LCDM measurement manifests each contain exactly snapshots 000–085. Rehash each numeric epoch and its recorded native sources, require expected public epoch pages/projections, and compare retained original run manifest/config hashes with relocation bindings. Write all_epoch_source_audit.json under 07_Status_and_Provenance after success. This is a specific 86-epoch campaign audit; it neither generates measurements nor applies unchanged to a tiny optional lab run.

## scripts/verify_paired_video_decode.py::main

<!-- implementation-sha256: acb6d14b47c6316f2a52dd95cd60200d34bbb4e3cffc2e0ce1a3984e6b37fa99 -->

Decode every nonhidden MP4 under a paired atlas’s 04_Movies through FFmpeg’s null output, with video-only error-fatal processing and a bounded four-worker pool. Write full_video_decode_audit.json and print the film count/status. Return failure if no films were found or any decoder failed. This reads existing movies through their final frames without creating new videos, inspecting scientific arrays or judging visual composition.

## scripts/audit_paired_graph_registry.py::main

<!-- implementation-sha256: e589fd8e5e03c2c5f46d973a003633bbef7d54576c209944b55b444672145076 -->

Combine the current registered graph catalogue with this script’s curated paired-DMO family/variant status tables. Require unique complete registry entries, then write registered_graph_coverage.csv/JSON under the atlas status folder, including required measurements and producer routes. Print the variant count. Status labels describe the intended retained atlas evidence and missing physics; this script does not inspect every claimed product or execute producers, so its inventory alone is not fresh execution proof.

## scripts/materialize_paired_share_links.py::main

<!-- implementation-sha256: de764984d5037402721c1d99415373392405f20dc85ef20d77b016480f672d29 -->

Replace atlas symlinks to internal regular files with atomically staged hard links to those same payloads. Reject external targets or directory links and write share_file_materialization.json recording replaced paths/targets/sizes. The change removes a dependency on symlink-aware sharing while retaining shared file storage on that volume; later edits to hard-linked content affect its linked aliases. This mutates an existing atlas and creates no ZIP or scientific results.

## scripts/build_paired_blender.py::main

<!-- implementation-sha256: 03ceb4371cd3d26d5d5a37e098cf3da88c5c5ae474398d68243c6b86ed0382b8 -->

Build an EDE or LCDM paired-atlas Blender scene from its retained native workspace and lexicographically latest detailed analysis bundle. Reuse an existing combined manifest, optionally finish a verified sealed DM timeline with --resume-dm, or export up to 80,000 particles at high quality and four frames per snapshot. Call build_scene with preview enabled and animation rendering disabled, then print export/build paths. This campaign-specific scene builder consumes real retained assets; it is not the optional lab’s 32-particle default.

## scripts/build_paired_blender_entry.py::builder

<!-- implementation-sha256: bb84ae5848d5cd75bce33e3b9a55b8e4b79fa2a49fa0fa0f388ab73305cd3ca0 -->

Configure the readable saved-universe view on Blender’s current scene and save it to the supplied output path. The surrounding Blender-only script provides the publication wrapper, model-specific destination and verified manifest; this callback itself ignores source/render arguments. Importing that script executes its top-level publication path, so the callback is not a standalone ordinary-Python entry point.

## scripts/refresh_paired_native.py::main

<!-- implementation-sha256: e766eca33ee4fe5f53ed1c02ff10f4cf4eeab51145db2ef3b8a161409ffa9934 -->

Locate the selected EDE/LCDM retained workspace in a paired atlas, read the lexicographically latest prior analysis summary and rerun analyze_selected_products for the same product list and saved snapshot IDs without a PDF report. This refreshes measurements through the current analysis implementation; it does not merely rewrite hashes or create new particle evolution. A matching pre-existing analysis summary is required.

## scripts/seal_paired_share_scenes.py::main

<!-- implementation-sha256: 587fbaf7927cf878656da91102ff80033078dab310dc470327c5721af8ff1200 -->

Update the two paired models’ atlas scene publications to compact direct-hash records. Copy the current blender.py helper beside each model’s scenes, retain prior publication metadata under Source/Storage_cleanup, hash the existing aliases and write/verify portable publication records for the main and readable-entry scenes. Publish scene_share_integrity.json. This changes helper/publication metadata around existing scene payloads; it does not rebuild geometry, render frames or remove rollback media itself.

## scripts/prune_paired_share_backups.py::main

<!-- implementation-sha256: 2dfeea3716125869a04cbc53d75b8545027d6a2b5e953b1d1c86e9fb0ac77162 -->

Delete superseded MP4/JPEG/PNG review backups from the paired atlas’s two named previous-publication folders, recording their hashes/sizes and writing metadata-retention notes. Update guided void-film contracts to mark removed original media/cache retention, then append removals to the existing storage_cleanup.json and print removed MiB. This is a mutating campaign cleanup that relies on retained final outputs and existing report structure; it does not itself reverify final-film correctness before deleting backups.

## core/flagship_abundance.py::logarithmic_abundance

<!-- implementation-sha256: 4e5e03954f9e436aa29873e7f360cf2f03d0dba0ea614b21add762a3c219647c -->

Histogram positive finite object values passing an explicit Boolean selection into ordered positive bins. Return counts, geometric bin centres, density N/(V·delta log10 value) and 68.27% central Garwood interval endpoints, retaining zero-count upper limits. Also return selected/in-range/out-of-range counts and both native/effective selection masks. V is comoving Mpc^3 without h, so density is count/Mpc^3/dex. Counting intervals omit cosmic variance, completeness and numerical convergence; a selected object outside the bin range does not contribute to the histogram.

## core/flagship_abundance.py::abundance_input_bindings

<!-- implementation-sha256: 14b2ac697ef8c6b56fed0f4b7c1780b3179181387c2207b85628016ccf237c89 -->

Verify the requested native subhalo export and galaxy catalogue against the run and their source contracts. Require a complete uniform periodic parent for the whole-box comoving volume and collect original config/manifest, catalogue, IC, solver and implementation hashes. Return the subhalo report, galaxy path, volume, normalization and source pins. This binds measured abundance inputs without evaluating histograms or generating galaxies.

## core/flagship_abundance.py::export_native_abundance

<!-- implementation-sha256: b9ee2e7fab9494733db2674940b09286db1c4564c0cc3112621f379a768fbeb5 -->

Produce stellar-mass and Vmax functions from verified uniform-parent native catalogues in a fresh destination. Require aligned exact catalogue identities, matching epoch/unit contracts and explicit particle-count/availability/resolution cuts; convert stellar Msun/h masses by dividing h. Publish per-function numeric audits, products.json and a scientific gallery, returning the gallery record. Populations are bound SUBFIND stellar masses and available native Vmax rows, including centrals, with density per comoving Mpc^3/dex. Zero counts and Garwood endpoints survive; catalogue selection alone does not establish completeness.

## core/flagship_abundance.py::matched_host_input_bindings

<!-- implementation-sha256: 9b1ac706144c829edade56a10b66571d42c82e7839ec75b6a0c4fbf74a9be79d -->

Load the two hash-bound native host-abundance products within split memory budgets, require their expected schema and verify their source-run provenance. Recompute uniform periodic parent volume/normalization and reject changes from each saved host census. Return the paired reports/arrays/configurations/volumes and combined pins. This verifies source and normalization inputs; actual epoch, mass-cut and initial-phase compatibility is checked by export_matched_host_abundance.

## core/flagship_abundance.py::export_matched_host_abundance

<!-- implementation-sha256: afbb1d1bc25da7726354194b6f5081e6b33e7ba51f0829ff155af628dba074a9 -->

Compare native selected host counts in common physical-Msun bins, requiring matching actual epochs, mass definitions, particle cuts, h/volume and measured initial IC phases. Use the existing shared-bin estimator, retaining conditional Poisson intervals and paired spatial-jackknife covariance separately; empty unsupported census ratios remain NaN with explicit masks. Publish matched-hmf.json and the B/A scientific gallery to a fresh destination and return the gallery result. Densities use comoving Mpc^3 without h. The two uncertainty estimates are not added or represented as full cosmic covariance.

## core/flagship_orbits.py::native_orbit_inputs

<!-- implementation-sha256: b37012ebd447181232ef51d9c8dfd730b05ebc09fd8d45234a5547ec06309c13 -->

Verify an identity graph plus a hash-bound recorded-image product naming that exact graph fingerprint and ordered object/host epoch keys. Collect graph arrays/index, run metadata, actual saved snapshot and proper-time source hashes within the epoch/metadata limits. Return the graph, recorded image offsets and source pins. This supplies explicit periodic image history rather than guessing unwrapping from sparse nearest-image positions; it does not calculate orbital events.

## core/flagship_orbits.py::measure_native_assembly_orbits

<!-- implementation-sha256: 78d7f8b7a025ce308d2c820c727624770525c10824ffff71345e28a62200b7af -->

Construct object/host worldlines from exact native graph observations, recorded periodic images and bound cosmological proper-time clocks. Require aligned saved epochs, source ownership and bounded cadence/search workspace; combine peculiar velocity with H times proper position for the coordinate derivative. Measure orbital candidates and retain native ancestry joins separately, with saved brackets, gaps and refinement status. Export an orbital-event numeric audit to a fresh destination, using Myr, proper kpc and km/s. Interpolation error is unavailable and ancestry-join time is a saved-epoch observation, not resolved coalescence.

## core/flagship_orbits.py::export_native_assembly_event_figures

<!-- implementation-sha256: 84a297de54be46ab0b6209b47a052542d2caf1bd0f303ab32eb9e7c318801cd2 -->

Read a source-verified native assembly orbital-event product and export its registered assembly scatter figure. Plot proper time [Myr] against event codes: pericenter 0, infall 1, ancestry join 2 and apocenter 3, distinguishing reconstructed candidates from original native joins. Return the scientific-suite export. The figure preserves INTERPOLATION_UNCERTIFIED status and does not turn displayed candidates into certified passages.

## core/flagship_history.py::validate_material_map

<!-- implementation-sha256: 8e60c7f81035bbdbb4c2230e990165fc5c470b59ba3c721198f09913e9201a9a -->

Validate positive memory and comoving box size, an exact parent refinement level from one to twenty and a positive integer material-tracer cap. Return None or raise. These are material-map control checks; this function does not read tracer IDs, determine ancestry or produce a parent-to-zoom mapping.

## core/flagship_history.py::validate_material_recovery

<!-- implementation-sha256: 094e97d348153113b3bdaf219d4cfb8e6b78f9a0c9eb4a5198065eb871c325ae -->

Validate a distinct nonempty saved-snapshot list and a matching distinct requested-redshift list, plus positive memory and a nonnegative epoch-error tolerance. Require explicit material/shared-particle thresholds with the runner-up ambiguity bound below the minimum accepted material fraction. Return None or raise. This checks recovery rules without selecting a descendant from native catalogues.

## core/flagship_history.py::material_recovery_runner

<!-- implementation-sha256: 07c66b42b1d6cdaa89ee32fb78e8cb84e8decdaa381dff429c3a869fbd3eeef1 -->

Read a parent-to-zoom material map and its owned uint64 tracer-ids.npy, verifying source hashes, count and memory bounds. Recover descendants across requested native snapshots using the declared material rules. Write recovery.json before the separate requested-epoch coverage gate so failed coverage does not hide recovery evidence, then write epoch-coverage.json and reverify inputs. Return recovery/coverage status with unresolved scientific acceptance. This uses exact material identities rather than choosing a convenient substitute host.

## core/flagship_history.py::validate_contamination_history

<!-- implementation-sha256: d5f66f04a7a4c367c0406a336cf186e35e63002eb0740902264826cb216b2691 -->

Require positive memory and R200c aperture factor, at least 96 MiB for the native catalogue/streaming workspace and a coarse-mass fraction limit in [0,1]. Return None or raise before measuring contamination. This validates allocations and the declared threshold; it does not inspect actual coarse particles or establish a clean aperture.

## core/native_convergence_catalogue.py::catalogue_profile_inputs

<!-- implementation-sha256: ef6266f0860fb18272d22831f46d70d28db100d41c89f6695e156705ee7e02fe -->

Verify catalogue/snapshot sources are owned by the completed native stage, check unit conventions and add native publication-marker hashes. If a paired baseline is supplied, verify its measurement seal and retain its source bindings. Return combined pins. This binds a convergence-profile input set without computing a statistic or proving convergence.

## core/native_convergence_catalogue.py::catalogue_profile

<!-- implementation-sha256: eeee245af66c7265d269621d31c2ac8261e692057dab4347ea5b669391cd78bc -->

Measure summed catalogued host M200c per fixed proper shell volume, not particle density within one halo. Require common physical bin/centre/box definitions, native epochs and at least 100 fine-DM particles per host, excluding coarse types two/three. Compute fixed-octant delete-one jackknife replicates/covariance with object/effective-count support gates; optional matched baseline adds paired cross-covariance after partition checks. Return report, values/covariance/support arrays and source pins, using Msun/proper kpc^3. Overlapping host masses remain summed; uncertainty is conditional catalogue sampling rather than cosmic variance or cross-finder identity proof.

## core/native_convergence_catalogue.py::catalogue_profile_runner

<!-- implementation-sha256: abac4c0596eaa21f0672c3158634278ef311bf068117757d201e91207adf8125 -->

Enforce the stage memory allocation, call catalogue_profile and publish catalogue-profile.json through the numeric audit writer with explicit density/covariance units and epoch/source bindings. Return its supported-bin count and unaccepted-science state. This measures one profile; a numerical convergence conclusion requires the declared comparisons and acceptance controls.

## core/stream_catalogue.py::build_stream_catalogue

<!-- implementation-sha256: 6b33fc9f7437118a4a48188e317ad8f632f643cb4eebac8a11f4c541666eedd9 -->

Discover coherent debris candidates from validated proper-kpc/km/s/Msun phase space and optional aligned ancestry/binding masks. Fit stream/shell/tail morphology alternatives and apply a supplied calibrated classifier; absent calibration or insufficient resolution remains ambiguous. Return candidate metadata plus original aligned particles, labels, binding masks and feature arrays/units. Missing ancestry stays unavailable, and wrap identity requires further track/phase history. This constructs a candidate catalogue rather than certifying every coherent structure as a tidal stream.

## core/paired_atlas.py::epoch_name

<!-- implementation-sha256: e592337fb84a9727a22096345c88dc053ae5bce82b2900974690881a119428a9 -->

Return a display/storage label combining the run name, three-decimal redshift token and zero-padded snapshot number; negative redshift is clamped to zero for the label. The token substitutes p for the decimal point. This is naming only: rounded labels do not validate or replace the saved physical epoch, and the caller supplies a suitable run name.

## core/paired_atlas.py::physical_frame

<!-- implementation-sha256: fdbee7475dec46d0db5ea65bfacc54b1a92602cface5fecf421bb83bbac7e42b -->

Read h=H0/100 and convert the configured box from comoving Mpc/h to comoving Mpc by dividing h. Require positive finite resulting h/box and return both scalars. This is a unit conversion for atlas measurements, not an epoch-dependent proper-distance conversion or a cosmological background solve.

## core/paired_atlas.py::measure_run

<!-- implementation-sha256: ffa51cd780edc104006291d18b64f15794e6f652f78d12510570e6fa992e58ba -->

Measure saved dark-matter epochs into non-h-scaled atlas arrays: three mass-per-area projections, a 64^3 mass cube, particle-speed statistics, catalogue masses/positions/counts and available native power. Save each NPZ, a sealed epoch JSON and a run measurement manifest/CSV; retain every-64th-row exact-ID samples for display. Reuse requires the current grid, run metadata, implementation and complete source inventory/hashes to match the sealed cache binding; reject legacy, corrupt or partial caches without overwriting them. Lengths are comoving Mpc, masses Msun, peculiar velocities km/s and P(k) Mpc^3. Full speed arrays are retained for statistics; native catalogue discovery is not replaced by a simulated population.

## core/halos.py::snapshot_numbers

<!-- implementation-sha256: 1eeb10c7738d2bdd914711927e2816bbd297a8a413a59301648ab5abdaab80e5 -->

Return the saved snapshot labels discovered by snapshot_sets as a list, preserving the discovery mapping’s order. This lists available storage epochs; it does not open all particle arrays or convert labels to redshift/proper time.

## core/halos.py::snapshot_sets

<!-- implementation-sha256: 93a567a14d6745f4f05113be043ce58ffef1d6b3f412fff7f95ab944550a0350 -->

Discover snapshot sets beneath run_dir/data using the shared analysis.discover_snapshot_sets reader. Return its mapping from saved snapshot labels to their file/shard lists. Discovery is separate from the full native dataset/unit/provenance verification performed by consuming workflows.

## core/halos.py::snapshot_file

<!-- implementation-sha256: e5c723d4baf56f245405ed3b7bd7276acf9c06a09a04fd91c3f2faaaed3bf364 -->

Look up int(snap) in the discovered snapshot sets and return its first file, or None when absent. This convenience accessor does not concatenate a multi-shard snapshot; callers needing the complete epoch must use its full file list.

## core/halos.py::snapshot_blocks

<!-- implementation-sha256: 840394d2207c2de5ac155166dc6c0a2960c264e4949c21de358cc548c51cc0df -->

Return the number of discovered files/shards for int(snap), or zero when no matching saved set exists. This is a file-count lookup, not a particle census or completeness certificate.

## core/halos.py::softening_kpc_h

<!-- implementation-sha256: ca237deec5bfa45a2fd452c292a46e0bce7929b3cc5fc65860437cc6d5e2789d -->

Return the explicit simulation softening as float, or derive the automatic value as one fortieth of uniform particle spacing: box_mpc_h·1000/N in comoving kpc/h. Imported noncubic sets must provide explicit softening. This helper supplies the Rockstar configuration scale; physical/time-dependent force support and value validation belong to the configured workflow.

## core/halos.py::subhalo_table

<!-- implementation-sha256: 6b312d4ba16de9549564be179cb5acc166d6f6c0cd9df1e9b2357375f577a28f -->

Read all native SUBFIND rows at a saved snapshot, retaining centrals and unresolved entries. Verify complete catalogue/snapshot shards, common epoch/box/h and native units, bounded memory, required/optional field consistency, integer identities and species-mass closure. Return a report with arrays, source/epoch contracts and native units: masses 10^10 Msun/h, positions/radii comoving kpc/h and SUBFIND peculiar km/s without sqrt(a). Missing catalogue returns SKIPPED; malformed partial fields raise. Rows are epoch-local and require trees for cross-epoch identity.

## core/halos.py::identity

<!-- implementation-sha256: 87dd117ed1fcfeca99e73d78b6df4b1ad197c5f8b180c8b4bfac18650bdc3f02 -->

Validate one value through exact_uint64_ids and return its exact identity as a Python int. This prevents a floating-point round trip for large particle/object IDs. It does not assign a namespace or establish that an ID is present in a particular catalogue.

## core/halos.py::node_key

<!-- implementation-sha256: e2c6198092b142da7524e1c0626a9556b4669fc4b2d71ec0d5cdbe3d052eb826 -->

Return a namespace-qualified graph key (namespace, exact uint64-compatible identity, epoch). Require a nonblank string namespace and an explicit nonnegative Python integer saved storage index. This constructs an observation identity; it does not infer a physical timestamp or a descendant link.

## core/halos.py::verify_identity_graph

<!-- implementation-sha256: 742ce25c8e9471ee27d3893d63620bd25746615021039932a61878f0ff622283 -->

Verify the identity-graph report seal, COMPLETE schema/status, SQLite digest, numeric node archive contract and every recorded source hash. Return the report. This protects the saved graph’s integrity; it does not reconstruct missing edges or independently validate the physical interpretation of recorded relationships.

## core/halos.py::IdentityQueries

<!-- implementation-sha256: 5d71615e2da89e8978db114e61b36575441f5e9edac30403e38778b5c52f9ef1 -->

Query an already published exact-identity graph, rechecking integrity before each read-only SQLite query. Expose epoch member IDs, measured bound owners, traversed descendant/birth/merger/return/stripping links, birth parents and parent-to-zoom matches. Return exact uint64 members or namespace-qualified records with explicit known/ambiguous/unavailable states. Queries preserve recorded evidence rather than inventing ownership from proximity or editing graph/tree links.

## core/halos.py::validate_catalogue

<!-- implementation-sha256: 443839c2bdff965bcef87f9b8e73b5ec1b891b4802c4943885e0e76f12c461e1 -->

Validate a normalized finder catalogue’s exact fields, finder/object/particle namespaces, actual scale factor, proper periodic box, aligned object phase coordinates and available nonnegative mass/radius/Vmax values. Require exact unique object/member/host IDs, retained quality flags, explicit mass definitions and unchanged bound sources. Return the input dictionary. This validates supplied records without identifying new halos or making two finders’ different mass/radius definitions interchangeable.

## core/halos.py::match_by_exact_membership_and_phase_space

<!-- implementation-sha256: 8f2bb795b0ed551591e1e5d30f8bcabf4fa515db2ef1e1e6f5316abce05a57f9 -->

Compare two normalized independent finder catalogues at the same actual epoch and proper periodic box, using one particle namespace and bounded pair search. Apply periodic position/velocity gates, then exact shared-member count/fraction thresholds when available; optionally retain phase-only candidates. Promote only unique reciprocal best overlap merits and preserve ties/ambiguity. Return candidates, disagreements and unmatched IDs, comparing mass/radius values only for matching definitions. COMPLETE describes comparison execution; no native mass or tree link is overwritten.

## core/halos.py::publish_finder_disagreements

<!-- implementation-sha256: 9eec0514111f36e224a5d53c4ef6a59df2127224681a675d24a269e12b16f00e -->

Run the exact-membership/phase-space comparison and publish its candidate IDs, distances [proper kpc] and velocity differences [km/s] with catalogue namespaces and source hashes through export_numeric_audit. Return the publication result. The full comparison report preserves ambiguity and incompatible mass definitions; the export does not repair a finder catalogue or select a new physical truth.

## core/halos.py::verify_tree_repair_proposal

<!-- implementation-sha256: cbb0b7f2f885c37a0d424fb9db3648cb79027e37156d2a0df55f540b17e199af -->

Verify a separate tree-repair numeric-audit product and all of its source hashes. Reopen the original audit and require its original node/link evidence to match that retained in the proposal. Return the proposal or raise. This checks a suggested repair’s provenance without applying it or replacing the native tree.

## core/halos.py::add_stellar_history

<!-- implementation-sha256: b46d6759b2270ca8c02058f687483640e8a9470634f619c4700b259a8b5d557d -->

Add exact stellar-history observations and measured ownership/continuity edges to an in-memory graph. Verify the HDF5 payload, native sources and workspace bound; sort epochs by saved scale factor, preserve missing/unbound/unavailable-membership sentinels and keep SSP identity links separate from unavailable birth-parent evidence. Reject duplicate observations and return source pins after rechecking the history file. This mutates the supplied graph but does not publish it or infer unrecorded progenitors.

## core/halos.py::consistent_tree_series

<!-- implementation-sha256: 31970e4cadc5bc70ca1ea9fcff1d22bb01da69be52197d0b85256970a745c3aa -->

Normalize native Consistent Trees tables against an explicit scale-factor-to-saved-snapshot mapping, checking actual snapshot clocks, h, box and bounded table memory. Convert masses to Msun and positions/box to proper kpc; retain measured peculiar velocities, original finder IDs and phantom flags when present. Return namespaced nodes, native descendant links and source hashes. Unmapped descendant epochs remain explicitly unresolved; missing measurements are not replaced by finite differences or equal row numbers.

## core/halos.py::read_velociraptor_catalogue

<!-- implementation-sha256: f3c3a9075779f7da78b3e38b863664bff424452dd16dbd3319103cd373912e77 -->

Read a published VELOCIraptor report after checking its seal, listed artifact digests and successful invocation status. Verify original or published-relocated invocation inputs, the native source manifest and normalized catalogue contract, then return the catalogue. This validates a retained native result without rerunning the finder or proving scientific convergence.

## core/halos.py::run_velociraptor

<!-- implementation-sha256: 28fe4d9a1ab6a921bc1647c43e7ec70599c3f04f381cba2311f4283840147d56 -->

Run the pinned native VELOCIraptor DM finder on a complete saved snapshot with explicit search/unbinding controls, thread/memory/time bounds and native ownership locks. Convert to proper kpc, peculiar km/s and Msun in a temporary transport HDF5, using dense transport IDs plus a reversible original-ID map. Retain native log/invocation, decode measured properties and bound memberships back to exact native IDs, then publish sealed catalogue/report assets in a fresh destination. Empty selection requires an explicit native marker. Return the report; build equivalence and scientific acceptance remain separate.

## core/evolution.py::add_wdm_thermal_velocities

<!-- implementation-sha256: 42520267e14b8de710c7d7396ff1a2ee8faf421f12278a555a689fa66177c4d9 -->

Apply the configured WDM thermal-velocity transform through transactional_ic_transform on a unique shard set. The delegate draws the CLASS-normalized relic field keyed by exact particle IDs and removes its full-population mass-weighted bulk. Transform staged copies and return the transactional publication result; failed validation preserves original ICs, and repeat thermal assignment is rejected. This changes initial velocities rather than running WDM evolution.

## core/evolution.py::rescale_monofonic_ic

<!-- implementation-sha256: f4d36d224733e16708072d9644d188839a23cc7c92757c76223792e8fb243d96 -->

Validate a positive finite target box in Mpc/h and transactionally convert monofonIC IC positions/header lengths to the run’s GADGET kpc/h convention, including periodic wrapping and scalar-header normalization in the delegate. Return the transform publication result. Staging protects originals on failure; this is an IC unit conversion, not a resimulation or cosmological growth adjustment.

## core/evolution.py::number

<!-- implementation-sha256: 7e66ad5d5dbcea822ff725189890fff3e05d71c8a2edcad91bb067691bfaae32 -->

Return a finite numeric scalar as float, optionally requiring positivity. Accept Python/NumPy integer and floating scalars while rejecting booleans and other types; include the supplied label in errors. Physical units and allowed domain beyond positivity are defined by the calling clock/configuration routine.

## core/evolution.py::validate_clock

<!-- implementation-sha256: 363b35710fc65682deee2a4676794147ea92633c8b8c36fbe2eabccac5f9df3e -->

Validate a native clock with ordered begin/end, positive maximum timestep, explicit comoving Boolean and supported 29/60-bit integer representation. Compute its tick interval in ln(time) for comoving scale factors or linear native time otherwise. Return interval and the legal power-of-two subdivision, with two-tick handling for exact_synchronized output. This plans a clock grid without observing actual native output times.

## core/evolution.py::quantize_native_output_time

<!-- implementation-sha256: f25e10724c04a5dde71c14921815bf623c07cd2083610e30527e055ace119fc3 -->

Map a requested time inside a validated native clock to its planned integer output tick. Use nearest allowed subdivision for timestep_grid, or the synchronized even-tick convention with exact final endpoint for exact_synchronized. Return requested/planned time, tick and quantization error; actual_output_time remains None and status PLANNED. This does not assert that a native run has saved the requested state.

## core/evolution.py::validate_user_schedule

<!-- implementation-sha256: d8f610a793dde5d284e50d6862367c65266a350ca7a9e7747f748f7d47b841bb -->

Require explicit start/end/backbone redshifts, a decreasing distinct backbone ending exactly at z_end and one to 1100 entries within the chosen redshift range. Return increasing scale factors 1/(1+z). This validates the user’s epoch schedule; native serialization/quantization and actual saved-time verification happen later.

## core/evolution.py::merge_and_quantize_native_output_times

<!-- implementation-sha256: d46d895f637fabcfe96930a469332503551b1c7861b76cb6beea9fe5fd90b3b6 -->

Merge an increasing backbone with distinct declared event windows within the native clock and output budget. Add points uniformly in ln(a) for comoving clocks or linear native time otherwise, round to the actual serialization precision and quantize each requested time. Return planned unique outputs plus collisions where different requests land on one tick. Window maximum_interval uses that native/log-clock coordinate, not automatically Myr; this produces planning metadata without executing evolution.

## core/evolution.py::event_window

<!-- implementation-sha256: cfce59cdd6552cd0f111eecd8b6badfeacc422e29937001aa7297fabc99d7458 -->

Convert a supported CANDIDATE event’s finite input bracket and contained predicted proper time into a refinement request. Require positive time-error budget and set maximum_interval_myr to twice that budget. Return the named Myr window, preserving its predicted time. This proposes finer output sampling rather than confirming the event or converting it to native scale-factor ticks.

## core/evolution.py::saved_outputs_resolve

<!-- implementation-sha256: 318b0007b204336384995a38a2edb402b031ee3d596e19eec3db8fc4590c863f -->

Test whether finite increasing actual saved times bracket the entire uncertain event window and whether the largest enclosed gap fits maximum_interval_myr. Return cadence status, measured bracket/gap and half-gap sampling-time bound. Coverage and cadence are necessary sampling support; the event must still be remeasured from finer history before physical acceptance.

## core/evolution.py::choose_restart

<!-- implementation-sha256: 34f6984006906a85d4f8dc15c93068e2d13bbd4e14bff7127cdcc2008ac17831 -->

Verify candidate native checkpoint directories for the requested MPI ranks and optional physical clock representation. Retain valid checkpoints no later than the event-window start, select the latest by saved clock with path tie-breaking and return its report plus rejected-candidate reasons. If none qualifies return NO_AVAILABLE_RESTART. This selects an existing restart boundary without launching a branch or suppressing invalid-checkpoint evidence.

## core/evolution.py::parameter_values

<!-- implementation-sha256: 748808a4bf9dd19de1c79aa6e4ba771e70157ff909920c54cb855c4dbfb39df2 -->

Read a native parameter text file into string key/value pairs. Skip blank lines and lines beginning with percent/hash comments, requiring exactly two tokens per active line and no duplicate keys. Return the mapping or raise on ambiguity. Numeric conversion, unit interpretation and supported parameter validation remain caller responsibilities.

## core/evolution.py::require_distinct_valid_epochs

<!-- implementation-sha256: 2f33f61e8dc5a1e705dfb5d1c282fe9ba629bb82e81c96ea674ec713329191e8 -->

Validate two to 64 positive start redshifts through the shared strict numeric check and reject pairs differing by at most 1e-8, the solver epoch precision gate. Return converted values in input order. This only validates candidate start epochs; it does not sort them, generate linear bundles or match evolved outputs.

## core/evolution.py::validate_start_epoch_request

<!-- implementation-sha256: e6bfa4db6f1890c53a30e7293fd8f5abed40279bb7b18387fa5101544549beb3 -->

Validate a bounded start-redshift campaign with an explicitly selected reference, exact distinct uint64 seeds and at most 256 branches. Require common later scale factors, narrow unambiguous epoch tolerance, named observable units/bins and numerical tolerance rules. Check configuration, one bound linear bundle per chosen start and optional complete seed/noise bindings. Return the request. These controls define a planned convergence comparison; validating them does not execute or accept its branches.

## core/evolution.py::verify_start_epoch_campaign

<!-- implementation-sha256: 22adf84d7fde79adcfaf5989b352a0b6eb76324a78c8516faaf919f632e6b617 -->

Verify a PREPARED start-epoch campaign’s seal, canonical request digest, contained artifact hashes and original source bindings. Revalidate branch configuration start/seed/unchanged physics, shared phase realization and exact-epoch linear bundle geometry. Rehash the campaign after reading and return its report. This checks a preparation’s integrity and consistency rather than proving native branch evolution or convergence.

## core/evolution.py::verify_start_epoch_phase_audit

<!-- implementation-sha256: 1fa6ef57815d43987880f83d924750239723473929d81e9313661e3937b71dff -->

Read a numeric-audit phase product, require the start-epoch phase schema, exact campaign record, overall PASS and PASS for the selected branch, then rehash its recorded sources. Return the phase report. This consumes an actual bound phase measurement; it does not infer phase invariance from seed equality or rerun the comparison.

## core/workflow.py::StageDefinition

<!-- implementation-sha256: 84ca61490dac6662f9ef5a58ac6c0e36ca676de53c24e009f130d3912e20df95 -->

Describe one registered workflow stage kind: its runner callable, implementation paths, required capabilities/parameters and optional validator. The frozen dataclass records how execution is dispatched and fingerprinted; constructing it does not validate a request, invoke the runner or establish that a backend is installed.

## core/workflow.py::StageRegistry

<!-- implementation-sha256: 039e7dd3a1d54404119dd895bdb9c933d52c1b0c228897757a9ca0ecd2d9a39c -->

Store uniquely named StageDefinitions and reject unknown/duplicate kinds. source_hashes() binds each definition’s declared files plus shared dispatch/resource/science readers and research helpers, adding dependent family modules as needed. Return definitions or source-digest mappings to the planner/executor. Registration supplies executable routes, not runtime success or physical acceptance; changes to these bound sources invalidate reusable stage fingerprints.

## core/workflow.py::safe_id

<!-- implementation-sha256: 4c7b29388af698038618397b0705250c6d74e27b38616f425d6d61889e092437 -->

Validate and return a nonempty string of at most 128 characters whose hyphens can be replaced by underscores to form a Python identifier. This supplies explicit stage/product names without path separators; it does not create an output or resolve a dependency.

## core/workflow.py::positive_number

<!-- implementation-sha256: 4a3376e60e06d35993e7a347ae3cde802e7fb75051884603440d493eee153b59 -->

Require a positive finite Python int/float, excluding booleans, and optionally an exact built-in int for discrete resources. Return the original numeric value or raise with the resource name. Units and machine feasibility belong to the corresponding resource control and preflight.

## core/workflow.py::path_at

<!-- implementation-sha256: ab55f9847f7b511f5dc66a1e8c6f1141a2b6568d62f6be1d4c1a1240c1bdccbc -->

Expand the user component of a filesystem path and resolve it absolutely, interpreting relative paths beneath base. Return a Path without requiring existence or confinement to base. Consumers add the ownership/existence/hash checks appropriate to that input.

## core/workflow.py::tree_hashes

<!-- implementation-sha256: f0ac7a66316dc5520f43cf42f08f13c4fcbc432d725f17792c5add7c9b6b68f7 -->

Return a sorted relative-path SHA-256 inventory of regular stage payload files beneath root. Reject symlinks and exclude stage receipt/seal files and native-run lock metadata. This inventory detects added, removed or changed payload files when compared with a completed receipt; it does not inspect their numerical contents or infer scientific acceptance.

## core/workflow.py::compile_requested_product_dag

<!-- implementation-sha256: 99bf184ce3fa2fb76b0e8dbd7ba0cd80a10e429c9b67e34cdcfc385ff02ad945 -->

Compile schema-version-two requests when needed, then validate the workflow’s resources, follow-up limits, supported physics, registered stage kinds, required parameters, product paths/contracts and dependencies. Resolve explicit requested products, including all through the physics-aware product catalogue, and topologically order only their required producer/dependency stages. Reject ambiguous producers, unsafe names, unknown references, cycles and excess stage/depth budgets. Return the research-plan dictionary without executing runners; validators requiring unresolved stage/file inputs are deferred until their inputs can be checked.

## core/workflow.py::resolve_values

<!-- implementation-sha256: aef53743bb50efade3a388beecee985f8a714a42033386b444bb6a4f525463a6 -->

Recursively resolve a stage’s parameters using completed receipts and its request base. Stage artifacts must exist inside the producing generation and can return a path, bounded JSON/field selection or SHA-256. Explicit input/json_input references resolve from base, while strings substitute generation/request_base/Python/suite placeholders. Return the resolved value. This resolution is separate from completed-receipt verification and does not run a producer to supply an absent artifact.

## core/workflow.py::verify_completed_receipt

<!-- implementation-sha256: a1de219737655b4cf4d981b461846c6aad4f6699349a837bcdbea2137d46247e -->

Require a sealed COMPLETE research-stage receipt located inside its declared immutable generation. Rehash every contained payload and compare the complete payload inventory, then verify recorded implementation, scientific-intake and input hashes. Return the receipt or reject changed/incomplete evidence. This proves the saved stage’s integrity for reuse; the receipt’s scientific/numerical domain assessments remain separate.

## core/workflow.py::WorkflowContext

<!-- implementation-sha256: 74b02141469c4189394af4e8ee30cd91baadcd8be630445f0c45edb5652f3782 -->

Carry execution state shared by the workflow: destination, request base, resource/capability declarations and completed receipts, with the current generation, optional restart checkpoint and stage ID. This mutable dataclass is passed to runners so their outputs and dependencies remain owned by the executing stage. Construction alone performs no work or resource allocation.

## core/workflow.py::execute_stage

<!-- implementation-sha256: dc1e041cabf9e812be8dc28900f9b5dcc86311390e6fc6f8a09d6d129619906a -->

Execute one registered stage after checking capabilities and fingerprinting its request, inputs, sources and resources. Reuse only an identical verified completed receipt on resume; otherwise create a permanent generation or select a matching failed/interrupted generation with verified native checkpoint shards under its declared policy. Resolve/validate parameters, run the producer and verify every declared output/scientific reader contract. Recheck inputs/code, hash/fsync payloads and publish sealed COMPLETE receipt/index records. Preserve FAILED/INTERRUPTED receipts on exceptions. Return execution evidence and follow-up requests; completion does not grant scientific acceptance.

## core/workflow.py::append_budgeted_child_stages

<!-- implementation-sha256: f3afd80aecea8faffad1e7fc1503d96e2e0431ceb58b51fe946ea97bd58a2ab0 -->

Validate a batch of requested follow-up stages against remaining child/stage counts, unique safe IDs and maximum dependency depth. Add the parent dependency and references to known earlier stages; later children may depend on earlier children in the same batch. Extend the in-memory plan only after normalization and return the updated total child count. The caller still recompiles/preflights the resulting plan before executing any new stage.

## core/workflow.py::execute_workflow

<!-- implementation-sha256: 46a9c59c69c284cc8a56600dd37ce70effeb3d08b3e2ba6fd2b8d5e2a9117ce7 -->

Compile and preflight the requested product plan and capability registry; dry_run returns that plan without running stages. Actual execution locks the destination, writes workflow.json state, runs/reuses stages sequentially and validates bounded dynamic follow-ups before admitting them. Preserve old specifications/history and original failure/interruption state. Record unfulfilled follow-ups/requirements as INCOMPLETE; otherwise execution is COMPLETE, with scientific exclusions retained separately. Optionally publish resource-plan comparisons and return the workflow receipt. This coordinates concrete producers rather than replacing their own numerical acceptance gates.

## core/workflow.py::public_command_runner

<!-- implementation-sha256: 6a7751ade70e5e71ba73ffb67f949eed20f1af67b0fb53889521fb80969f4d48 -->

Invoke an allowed commands.py command through the workflow’s subprocess runner, excluding recursive workflow/pilot/cleanup entry points. GADGET requires explicit answers matching stage MPI/thread/memory choices and optional canonical constraints. For mutating analysis/display inputs, optionally make a size-bounded private run copy inside the generation and record it, rejecting directory-symlink copy routes. Pass wall/resource-sampling controls through, preserving failed native checkpoints where applicable. Return command_runner’s result; this can launch actual native/Python work and write private-run/log/receipt artifacts.

## core/workflow.py::json_function_runner

<!-- implementation-sha256: 164c4c9dda8a49c85484b0c78e1f4ef6168e8b9547b8a68aa00b76b326719f32 -->

Return a runner closure that imports a named module/function when executed, calls it with resolved keyword parameters and writes its result to the configured JSON filename in the stage generation. Return the result to the scheduler when it is a dictionary, otherwise an empty result dictionary. Creating the closure does not import/call the target or create files; target side effects apply when the returned runner executes.

## core/workflow.py::display_products_runner

<!-- implementation-sha256: a6db0f334a9efa035d9c9b067754b7723c51a143aa9e2ac1ab14d207e066749b -->

Enforce the display request’s memory ceiling against the frozen stage allocation and require its destination inside the owned generation. Call assemble_measured_display with the explicit request base, write display-assembly.json and return the result. This executes the selected measured-display assembly; availability and scientific interpretation remain governed by its underlying products.

## core/workflow.py::error_ensemble_runner

<!-- implementation-sha256: 1e9343332cd324846744e0190485d594f96970ade631685bb3f6cabfbd9983ab -->

Validate an explicit integer ensemble workspace within the stage memory allocation and require publication inside its permanent generation. Call publish_correlated_ensemble with the resolved parameters, write ensemble-publication.json and return the result. This publishes the supplied realization/systematic-control ensemble through its existing scientific routine; it does not create independent physical realizations by labelling them.

## core/workflow.py::default_registry

<!-- implementation-sha256: 27f456ac600ff510e1ec0695c121847f0655508640f510263beb3b123505cec0 -->

Construct the workflow’s StageRegistry, binding registered scientific runners to required parameters, validators and source files. Include native preparation/history, IC/two-fluid, BH/SIDM, streams/encounters, apertures/assembly/web, observation/inference, convergence/errors, public commands, displays and small JSON-function adapters; delegate additional SIDM registration to its module. Return the registry. This imports stage definitions and establishes available dispatch routes, but does not execute those producers or establish that their external software, physical inputs or calibration are available.

## core/workflow.py::workflow_main

<!-- implementation-sha256: 0a6ba4f7b736e4aa156f5719013153125161d4b724aff89ac26e5555c3a3c0d2 -->

Parse exactly one --answers or --campaign request plus required --output, optional --resume and --dry-run. Read JSON, resolve relative inputs using the request’s parent and invoke execute_workflow. Print the result; return zero for dry-run or COMPLETE execution and one for an incomplete returned workflow. Uncaught validation/execution failures propagate rather than being converted into successful status.

## core/workflow.py::validate_product_contract_definitions

<!-- implementation-sha256: 24385529a1e639667d220f0253286882f779804df4efbf4acac545bce24a3106 -->

Require each scientific product-contract descriptor to name a declared stage output and a known existing reader. Validate explicit distinct snapshot schedules for native series and distinct contained relative shard paths for native HDF5 sets. Return the contract mapping. This checks descriptor structure during planning; read_stage_product later opens actual artifacts and verifies payload/semantic contracts.

## core/workflow.py::read_stage_product

<!-- implementation-sha256: 587a0ce0771e0a401c87cd5e21253c58d943ebb87652a3f95395842d270cf1c5 -->

Read a declared scientific product only inside its owned generation using the selected native snapshot/set, numeric-audit, model-array, stream, measured-display or Blender-scene verifier. Enforce applicable memory bounds, complete shard/publication rules and contained references, then return INTAKE_VERIFIED with hashes of the verified inputs/assets. This is structural/semantic/payload verification through existing readers; it does not establish convergence or accept a scientific model merely because the file opens.

## core/workflow.py::workflow_products_main

<!-- implementation-sha256: c619d384105beb912aadf20edd5f37c8be5cb96c932f26b16e84edc785522a02 -->

Read a retained workflow through load_workflow_stage_results and print per-stage execution state and product manifest IDs, paths, hashes and schema versions. Direct readers to each product’s current domain/acceptance receipts. This CLI exposes existing scientific products without rerunning the workflow, granting acceptance or producing new figures.

## core/research/controls.py::control_config_text

<!-- implementation-sha256: bedd5d807142084f3950cb4efe12bd10a4be8f3775a239a1a7864bbece15ee40 -->

Generate newline-separated native compile options for an explicit control profile. Support collisionless analytic/tree/TreePM modes, isolated or homogeneous physical SIDM variants, canonical SIDM transport and delegated hydro/source profiles. Validate exact profile controls and retain double precision, exact IDs and relevant restart/output features. Return Config.sh text without compiling or running it. Diagnostic-only collisionless variants change output/build bytes while retaining the tested equations; they do not implement a nonzero SIDM control.

## core/research/controls.py::validate_collision_control

<!-- implementation-sha256: 165847dfd58dfd1901c2d9719f6c0fe95618eeac42b6e54018afea40a1e6707c -->

Validate a physical one/two-Plummer-population collision request: exact particle/ID/seed bounds, finite physical units/state, explicit saved Gyr clocks and timestep/softening/resource controls. Require a valid native SIDM model and compatible macro masses when enabled, bounded MPI ranks and one worker thread per rank. Return total particle count or raise. This establishes valid requested inputs; no phase-space sample, collision evolution or conservation result is generated.

## core/research/controls.py::prepare_plummer_collision

<!-- implementation-sha256: e4956699c1f762ba146c2d61a0529d3fb3d6f7269e02d5453d887d164df62953 -->

Write immutable physical Plummer initial conditions in a fresh directory using the requested Philox seed and exact IDs. Sample the untruncated radial mass CDF and isotropic conditional velocities with v^2/v_escape^2 distributed as Beta(3/2,9/2); convert proper-kpc/km/s/Msun/Gyr inputs to explicit native units. Reject samples outside the chosen box rather than clipping/reseeding, and retain finite-population centre/momentum fluctuations. Publish request.json, collision-ic.hdf5 and collision-ic.json; return the IC receipt. This prepares analytic physical initial states, not evolved collision evidence.

## core/research/controls.py::collision_input_bindings

<!-- implementation-sha256: f56f869220ba73184c460b247fbb9a59498bf4469bc0914cc7ab58de6e18be07 -->

Validate the physical collision request and bind its finite-size base configuration by SHA-256. If SIDM is enabled, additionally require the exact physical angular-model asset hash. Return the input pins. This checks retained inputs without building GADGET or measuring a collision.

## core/research/controls.py::validate_collision_stage

<!-- implementation-sha256: db3f111de337a9cf8d3dceb54d8d131f16acc95acfadde65602440b6e4edf6c5 -->

Validate the collision-control request and require base_configuration to contain exactly a path and SHA-256. Return None or raise. File content verification belongs to collision_input_bindings; this stage-shape validator does not execute native physics.

## core/research/controls.py::physical_collision_runner

<!-- implementation-sha256: cc2bf6ba8e9947b9ef197f94a4b878581f216cea01ef54c2ce5a74d490bbd0a0 -->

Run a dedicated isolated DM/optional-SIDM Plummer control inside a workflow generation. Preserve the explicit resource allocation, reject incompatible baryonic/cosmological base settings, prepare ICs, build the requested native profile and invoke MPI GADGET with supplied physical units/clocks. Require all requested snapshot publications, correct isolated times and unchanged particle census. Write collision-evolution.json and return evolved-particle/snapshot counts. Status EVOLVED_ONLY records execution; rare-event measurements, physical budgets and scientific acceptance remain downstream requirements.

## core/research/controls.py::positive

<!-- implementation-sha256: 1c0c47d53d7ecc6cfdb7c049d08f5047758f3123410380634a0eab62b2237b5b -->

Require a finite positive Python int/float, excluding Boolean values, and return a float. Use the supplied name in validation errors. This scalar validator supplies no unit conversion or upper resource bound.

## core/research/controls.py::potential_parameters

<!-- implementation-sha256: 644ebb1e90f91a9f1b388d8633304e9d301b814b6ff5a5def13a2f58fdc5f1c7 -->

Require exactly kind, gm, scale, axis_y and axis_z for a supported Plummer, Hernquist, NFW or harmonic analytic control. Convert the four numeric parameters through positive() and return a new dictionary. Consistent native/control units are supplied by the caller; this routine neither fits a measured halo nor evaluates the potential.

## core/research/controls.py::analytic_field

<!-- implementation-sha256: b1d31dfefee8faafbdf3fb79cae58bc8b2b2ec48e97af5bc684aa898ef508909 -->

Evaluate the requested analytic potential, acceleration and tidal tensor (acceleration Jacobian, negative potential Hessian) at finite Cartesian positions. Support anisotropic harmonic and spherical Plummer/Hernquist/NFW models, excluding central cusps where required. NFW enclosed mass uses independent quadrature at small radius to avoid subtractive cancellation. Return arrays in the consistent control-unit convention: potential length^2/time^2, acceleration length/time^2 and tides 1/time^2. This reference does not include a live particle self-field or numerical evolution.

## core/research/controls.py::analytic_orbits

<!-- implementation-sha256: 0befae69d6e8bdc06c42b7dfec28b0dad504ac278b4aa7a05a4a8216e8594d65 -->

Integrate finite N-by-3 initial states in the analytic_field potential at strictly increasing requested times using SciPy DOP853 with tight fixed tolerances. Solve dx/dt=v and dv/dt=a in consistent caller-supplied control units, requiring successful coverage of every time. Return position and velocity histories. This is an independent continuous-potential reference, separate from the native particle/tree/PM integrator.

## core/research/controls.py::stencil_tides

<!-- implementation-sha256: d2c3aa3c630bc8a5ff7e28c2d57f41c9c3ce673de5c4208efcaf5c2762f227f7 -->

Estimate the acceleration Jacobian from a finite 28-by-3 acceleration array arranged as four seven-probe central stencils. Return three centred spatial differences per probe group, dividing opposing accelerations by 2·step. The caller supplies the matching nonzero length step and exact probe ordering; output units are acceleration/length. This computes a numerical tidal estimate without generating forces or certifying stencil convergence.

## core/research/controls.py::residual_summary

<!-- implementation-sha256: d656d75c5c4eeaf7498c3a7c4b799325870e0eb753d514f80e6c0e4136b587a2 -->

Summarize nonempty finite residual/reference arrays with per-row vector/tensor Frobenius norms. Normalize by max(reference norm, positive dimensional absolute_floor), then return absolute maximum/RMS, relative maximum/quantiles and the worst row/residual. This avoids componentwise division at zero reference components. The summary is descriptive; acceptance requires independently declared tolerances and a compatible reference/floor unit.

## core/research/controls.py::isolated_invariants

<!-- implementation-sha256: ac9fdf58b4aa88fa589e1b027f776340751e2935909e6ea714b40867188c29c6 -->

Return specific energy |v|^2/2+Phi(x) and specific angular momentum x cross v for supplied states in the declared analytic potential. Units follow the common control position/velocity convention. These are diagnostic values; conservation applies only where the potential’s time dependence/symmetry permits it, rather than being guaranteed by this calculation.

## core/research/controls.py::validate_control_request

<!-- implementation-sha256: ea5d2bcd2ee88edf27d9c016906eccb332eaba32b6dd6fbc5957cf49dfbfb43f -->

Validate and deep-copy a bounded native numerical-control campaign. Require explicit CPU/build/memory/wall/disk limits, one to 64 uniquely named cases with supported profiles/potentials, exact seeds, bounded particles and saved clocks, and tolerances declared before execution. Validate rank/restart/diagnostic/settings comparisons so unrelated inputs remain fixed. Analytic stencils require 28 probes; periodic cases require supported cosmological times. Return the copied request without building/running a case or asserting its tolerances pass.

## core/research/controls.py::fixture

<!-- implementation-sha256: eee4b1569de657e08fe392c4051ed7025cc4d0137338e38ddd5fa71923f7ac08 -->

Construct deterministic control phase space: exact analytic stencil probes, a periodic perturbed cubic lattice, or seeded isolated finite-particle samples from the selected profile. Return positions, velocities, masses, exact uint64 IDs above 2^53 and a control box size in the native control-unit convention. Isolated samples include prescribed modest rotation and finite-population centering; this numerical fixture is not evidence that a sampled smooth halo is equilibrated or evolved.

## core/research/controls.py::run_control_campaign

<!-- implementation-sha256: 63fc57a4594621dd941dec7afa7fe3fa39c38e83c1567c7797d2407debc051fa -->

Execute the validated native control matrix under a destination lock, binding its request and complete native build/source inputs. Build and invoke each GADGET case, optionally pause/verify/resume its checkpoint, analyze actual outputs and compare declared paired variants. Retain Ewald cache identity and every case/build/output hash; write sealed campaign.json plus execution state. Resume reuses only an identical verified completed campaign. Return PASS or FAIL from the original case/comparison decisions; results cover the recorded matrix and do not establish arbitrary production or SIDM-enabled accuracy.

## core/research/controls.py::verify_control_campaign

<!-- implementation-sha256: b8a762de8aa5035703b17db834f1974bde7f4e512f043495943525eb5139584d -->

Verify a completed sealed native-control campaign, revalidate its request and require the current complete native build-input inventory to match the retained source closure. Rehash every source/artifact and inspect each case numeric audit against its recorded status/hash. Return the verified report even when its scientific/control status is FAIL; this is integrity verification rather than permission to relabel rejected results or execute new cases.

## core/research/convergence.py::queue_bounded_refinement

<!-- implementation-sha256: 247de7bffc73ee68484b65a0ad118b5b06cc2a59940e6eba711dc5ff5bbd8b10 -->

Select decisions in their supplied order while remaining child count and maximum depth permit; retain excluded decisions with a recorded-bound-exhausted reason. Reject an already queued count beyond the frozen limit. Return children, unresolved decisions and a hash of the parent plan without running refinements or modifying the supplied plan.

## core/research/convergence.py::validate_native_configuration

<!-- implementation-sha256: c7949fb1710aebab4caf7b685b83482019f469bf1b0ebde6f73a1cf3a8465ec4 -->

Validate explicit JSON/GADGET-text configuration bindings and a positive byte workspace. Require a real phase_seed binding, supported scientific control names, unique complete native key paths and positive finite unit scales. Reject resource-field writes, seed rescaling and unsupported alternate specification paths; start-epoch controls may bind a regenerated cosmology bundle. Return the parameters. This checks how controls will be applied without silently changing hardware allocation or writing native input files.

## core/research/convergence.py::configuration_input_bindings

<!-- implementation-sha256: ccb3afbad0cf2703c0f168b0b79fa9c3220b799b3bce0dd0f9bb06720f2bc185 -->

Validate configuration controls and hash-bind the frozen baseline input, bounded variant metadata and exact specification named by that metadata. Return their resolved source pins or raise when a retained input differs. This supplies fingerprint inputs before materialization; it does not render a variant or establish evolved equivalence.

## core/research/convergence.py::materialize_native_configuration

<!-- implementation-sha256: 02ea5c7e6557982d1085b99451d7b7d22001709cb93fca83c3c99209c2161713 -->

Read immutable convergence variant metadata/specification and the hash-bound baseline JSON or GADGET text within byte limits. Apply exactly the planned control dimensions plus phase/operator seeds through explicit native bindings and unit scales, rejecting undeclared changes, seed coercion, missing paths or text-field injection. Recheck sources and return rendered input text plus applied-field/provenance metadata. The result is INPUT_MATERIALIZATION_ONLY; it does not evolve or measure the numerical variant.

## core/research/convergence.py::native_configuration_runner

<!-- implementation-sha256: 677a8d34bd33d8a3b054fbbdcd84540b8ed2fdc6d251f623bad559055ddb3154 -->

Require configuration materialization to fit the stage workspace, write the returned native-control-input.json or .txt and native-control-configuration.json in the generation, and add the written input hash/path. Return the count of applied native control fields. This prepares an executable input for a later native stage rather than claiming that the controlled comparison has run.

## core/research/convergence.py::convergence_main

<!-- implementation-sha256: b8704d433f1a186f0a7f437c708577d14fa25434bb092f3760dd7cb9e900c67a -->

Read an explicit --campaign JSON and invoke execute_and_measure with required --output, the campaign’s parent as request base and optional --resume. Print the resulting controlled-variant measurement report. This CLI can launch actual evolution/measurement through that campaign; it returns no explicit status value and propagates uncaught validation/execution failures.

## core/research/convergence.py::compare_observable_convergence

<!-- implementation-sha256: d281580dd5b47ab1c69a50d605ae1b08fe13a939f6d394bcad346eeb73256cb0 -->

Compare unique measured runs with a named reference, identical experiment/observable/unit definitions and exactly the declared changed numerical control. Intersect support/finite masks and compute differences with covariance: C_current+C_reference minus both cross terms for paired data, or require explicit independence. Validate joint/difference PSD without fabricating missing covariance. Return per-run tolerance decisions and difference/covariance/support arrays. Agreement uses absolute+relative tolerance on supported values; one passing control is not full physical acceptance and source hashes must be verified by the surrounding intake.

## core/research/convergence.py::convergence_figure_inputs

<!-- implementation-sha256: a7e808d16a473f241d5658ac5e7e36ad720cfd3b3f9a4c8f5d4d064f046a2f55 -->

Verify at least two hash-bound completed native measurement products against their exact owning convergence-measurement receipts and seals. Require each original native realization receipt to be a verified public-command stage, collect sources and bound metadata/covariance intake. Rehash the resulting pins and return them. This establishes measured producer ownership for figures; it does not create missing convergence runs or derive acceptance from an image.

## core/research/convergence.py::convergence_figure_runner

<!-- implementation-sha256: ac7547ca81eb46ec9be828391a03853564498aa161d2bf8c169afc5be24dee09 -->

Enforce the workflow memory allocation, verify original convergence figure inputs and export_convergence_figures into the generation’s figures directory with the request base. Recheck all input pins after export and return the figure count with scientific acceptance unset. This renders already measured controlled comparisons, leaving their original support/tolerance outcomes intact.

## core/research/convergence.py::convergence_phase_runner

<!-- implementation-sha256: 2006c6b2c835861a5b03fbc65a2fca0043a715cc0faf6fb24bb390e58206a8e8 -->

Enforce the stage memory limit and delegate actual common-phase/input checks to convergence_phase_inputs. Write phase-pair.json recording the control, common frozen levels, identical traced-parent cohort and source pins, then return the checked level count/PASS. This records the verified pairing prerequisite; it does not evolve variants or conclude observable convergence.

## core/black_holes.py::wake_diagnostic_settings

<!-- implementation-sha256: 7212c4266e914fa24d004caf196212833c76fa47c956ce09273905024e2d01f6 -->

Validate an exact schedule path/SHA-256 and positive bounded maximum byte/row controls. Reject symlink schedule inputs, invalid digests, changed bytes and oversized files; return the resolved original schedule path. This checks diagnostic intake settings rather than parsing every row or executing a native wake measurement.

## core/black_holes.py::dm_wake_schedule_main

<!-- implementation-sha256: e6cfcddc10c03e70ad36e170802a0f43e1e8aad35d4bbed8d642c4df6f89fa74 -->

Read a diagnostic schedule request JSON and required --output, invoke publish_dm_wake_schedule and return zero after successful publication. Validation/publication errors propagate. This CLI materializes the bounded native diagnostic schedule; it does not run GADGET or compare wake forces.

## core/black_holes.py::finite_number

<!-- implementation-sha256: 71a8117319f16496bc00a87cfe59100cf55ef87516e2138266243984731f00f8 -->

Require a finite Python int/float, excluding Boolean values, at least minimum or strictly above it when strict=True. Return a float with named physical-range errors. Units and upper bounds are supplied by the caller; this helper performs no unit conversion.

## core/black_holes.py::validate_dm_wake_force_request

<!-- implementation-sha256: ac5d2e8dd0ded620bbe489bf8f7a54390dd271e2a9f5227d7d21d6bfadd33410 -->

Validate supported wake-force request versions: explicit physical probe state for v1, exact original BH particle ID for v2/v3 and an additional reference/native support discrepancy budget for v3. Require nonempty particle namespace, bounded snapshot/census/chunk work, positive memory/support radius and ordered proper-kpc shell edges. Return the request. This validates measurement controls without reading native particles or certifying the force operator.

## core/black_holes.py::dm_wake_force_input_bindings

<!-- implementation-sha256: 5b9d067b7e8103d88e3c456d95ab1c3fb2420afba3a970ae1ca19892566f03c6 -->

Validate the wake-force request and return the shared native-stage input bindings from _stage_native_bindings. This requires the retained producer authority used by later measurements; it does not compute an aperture force or infer execution from a file listing.

## core/black_holes.py::validate_dm_wake_kernel_comparison_request

<!-- implementation-sha256: 2ee53a00e5dc3d2289fe0c49ee091ce5b82fbd41c42c203232ec507f5fe658f4 -->

Validate exact native-diagnostic/reference-force bindings and predeclared epoch, probe-position, relative-mass and shell-acceleration error limits. Enforce bounded diagnostic bytes/shell count and memory, with optional closed-total-force error budget. Return the request. This checks comparison controls; the publisher subsequently verifies the actual unchanged BH/DM census and measured arrays.

## core/black_holes.py::dm_wake_kernel_comparison_input_bindings

<!-- implementation-sha256: 9c2e0963570bba152286c3c80662d5ec5d4eb84759731b00736872c555e72c18 -->

Return source pins obtained by the full _wake_kernel_comparison_inputs reader for the requested native diagnostic and original saved-state reference. This prepares exact comparison inputs without evaluating force residuals or creating a publication.

## core/black_holes.py::publish_dm_wake_kernel_comparison

<!-- implementation-sha256: 827c8648dcb797dc4f5960679cd4f965e69440b9c1767c8a14499156e39ae2ce -->

Compare actual native BH aperture-kernel diagnostics with the original saved-state reference. Require exact carrier/DM counts, matched native epoch/probe state/masses, identical radial edges and shell census; compare shell mass/acceleration against frozen limits. Retain optional native Ewald/PM-short-range terms separately, with explicit availability masks, and allow closed total-force comparison only for its narrowly declared captured frame/census. Publish residual arrays in cm/s^2, masses in Msun and supports in proper kpc via a numeric audit. Return the export with supported/unresolved status; this does not establish arbitrary tree/PM force equivalence or production drag acceptance.

## core/black_holes.py::dm_wake_kernel_comparison_runner

<!-- implementation-sha256: 8e899cd2f599875399f71ee06adfbac132972fc8f0d0af6f0744b208615c4d3d -->

Require kernel-comparison memory within the stage allocation and a destination inside its generation, then return publish_dm_wake_kernel_comparison. This workflow adapter publishes the measured comparison; its returned support decision retains the original operator scope.

## core/black_holes.py::dm_wake_kernel_comparison_main

<!-- implementation-sha256: 9782891e63c8925e6ff36bb116e7d8911f11f7131243d2ebe269c74cd7969cc0 -->

Read parameter JSON, default request_base to that file’s parent when absent and publish the native kernel/reference comparison to --output. Return zero after publication; rejected comparisons remain encoded in their product status and exceptions propagate. The CLI does not rerun native evolution to manufacture a diagnostic.

## core/black_holes.py::validate_dm_wake_contrast_request

<!-- implementation-sha256: 7c50640688a7e445a9f4e8bd6bffeaffa441e5027bab070802ec341918d27216 -->

Validate matched perturbed/background force products at three or more actual resolution levels. Require explicit reference residual budget and bounded memory; v1 uses a supplied nonzero relative probe velocity, while v2 uses measured-bath mismatch/minimum-speed limits. Optionally retain a native pair-support discrepancy budget. Return the request without calculating a contrast or declaring resolution convergence.

## core/black_holes.py::dm_wake_contrast_input_bindings

<!-- implementation-sha256: 143995239535d14c04e58b5d2eca0a1cfef7952ee49b5034c04c5e97d9559943 -->

Validate the resolution-contrast request and load each perturbed/background product through the shared source-bound reader under split memory budgets. Return combined pins. This input pass does not subtract force vectors or choose a resolution acceptance outcome.

## core/black_holes.py::validate_dm_wake_ensemble_request

<!-- implementation-sha256: 4376e7e1a9a8532648b2547650c75c104755993c1e49c9c26d4fd0abeb1f009b -->

Validate a bounded multi-realization wake contrast ensemble with at least two requested realizations, exact observable/resolution indices and predeclared mean/reference/transverse/deceleration budgets. Version two adds measured bath velocity/density/covariance consistency limits and dimensional floors. Bound the complete covariance workspace and return the request. Declared realization lists alone do not establish independence; the measurement producer checks their native identities.

## core/black_holes.py::dm_wake_ensemble_input_bindings

<!-- implementation-sha256: 80a05a20cf8ded03defa8a0f50383460c4bb7c5cd4c4bd711f6912ad9f94c210 -->

Validate the wake ensemble request and source-verify every retained resolution-contrast product within a split input budget. Return merged pins for fingerprinting and later covariance measurement. This does not compute an ensemble mean or replace missing native realizations.

## core/black_holes.py::validate_dm_wake_response_request

<!-- implementation-sha256: 02e31149148987b9ffa803ee01c3fb1afcf1f5955d9e163a0b7daa1516f1bfd4 -->

Validate speed-response calibration controls with at least three training ensembles and one held-out ensemble, each bounded by maximum_points. Require positive memory/prediction floor and explicit absolute/relative prediction-error and mean-error controls. Return the request. Actual speed ordering, fixed-physics consistency and held-out support are determined by the response publisher.

## core/black_holes.py::dm_wake_response_input_bindings

<!-- implementation-sha256: 6a4b7e9080190f45ba77f49b2c8e97fcd27b386f14e0f9a7016406560eed5463 -->

Validate the speed-response request, read all training and held-out ensemble products through _load_product and return their combined source pins. This binds calibration evidence before fitting; it neither constructs knots nor evaluates a query speed.

## core/black_holes.py::evaluate_dm_wake_response

<!-- implementation-sha256: d35b4dd308e2dfeb76426425e44368580c4e4d9b2db44098f557cc5e143deb91 -->

Read a calibrated speed-response product with the requested fixed physical/operator signature. Require a finite nonzero relative velocity aligned with its measured direction, within the speed-knot range and inside a held-out-supported interval. Linearly interpolate the deceleration amplitude and return the opposing acceleration [cm/s^2] plus interval/source/scope metadata. Reject extrapolation; this evaluates an existing empirical response rather than evolving a native wake.

## core/black_holes.py::validate_dm_wake_state_response_request

<!-- implementation-sha256: 8cc6e724197f8dc77faa521423c3c1d0407eefe379c1d6f1df73b4d533bd4d6f -->

Validate bounded training/held-out product and realization counts, distinct active physical coordinates and ten positive dimensional scales. Require explicit design-conditioning, convex-domain residual and prediction-error budgets within supported limits, and a sufficient declared fit workspace. Return the request. This checks controls before loading actual state clouds or fitting vector accelerations.

## core/black_holes.py::dm_wake_state_response_input_bindings

<!-- implementation-sha256: 883042c10a62174fa2d2c84c6ce7397e9d6517743f80002fbc954d2cf26c7376 -->

Validate state-response controls and load every training/held-out ensemble reference with the shared verified reader. Return the accumulated source hashes. This input binding precedes native seed separation, design conditioning and held-out prediction checks in the publisher.

## core/black_holes.py::publish_dm_wake_state_response

<!-- implementation-sha256: 1195844cba030bd5d87c670a0703b4d664323b2a5a301c6534121744c30312e3 -->

Fit an overdetermined local affine vector-force model to supported measured warm-bath ensembles with fixed unmodeled BH/DM physics, epoch and resolution. Keep training/held-out phase/operator seed axes disjoint, normalize ten state coordinates and require design conditioning. For each held-out query require measured convex-domain support and an error allowance including prediction plus reference budgets. Publish coefficients, realized states/forces, convex weights, residuals and support masks. Return the numeric audit with supported/unresolved status; empirical local calibration does not grant arbitrary mass scaling or global coverage.

## core/black_holes.py::evaluate_dm_wake_state_response

<!-- implementation-sha256: e98c3ebac6a43ec8f7345dcfd1193de5ba488141c98496a7362be1b1b1cecf28 -->

Require an overall held-out-supported state response with the requested fixed signature and query workspace. Convert relative velocity, density and velocity covariance to its physical coordinate vector, enforce the measured convex domain and evaluate the saved affine force. Return acceleration [cm/s^2], convex weights, interpolated reference error and empirical held-out residual bound/source pins. Dissipative work must still be required independently by a conservative source caller.

## core/black_holes.py::dm_wake_state_response_runner

<!-- implementation-sha256: 43323081db575367200d42f7461d40db0c7171b560b54327edc2fedfba5b3ff1 -->

Check the state-fit memory against workflow allocation and confine its destination to the generation, then invoke publish_dm_wake_state_response. Return the actual calibration publication; this adapter does not change held-out budgets or mark unsupported states accepted.

## core/black_holes.py::dm_wake_state_response_main

<!-- implementation-sha256: 96d99beed29d2edaf0484ab0cc2eb8354207b3872f76b14c376901541872e625 -->

Read state-response parameter JSON, default its request base to the file parent and publish the fit to --output. Return zero after publication and propagate errors. Supported versus unresolved calibration is recorded in the output product, rather than inferred from the CLI exit code alone.

## core/black_holes.py::validate_dm_wake_context_request

<!-- implementation-sha256: f378b9368e7e8c94882a23208ee0ec747da9a4c2d9a81792871fbf4f402a09d3 -->

Validate a native source context referencing response, table and background-force products. Require an exact uint64 seed, bounded particles/context/source/schedule/subcycle work and complete numerical limits for velocity changes, conditioning, conservation and proper clocks. Enforce memory and supported tolerance ranges; return the request. This validates intake settings without enabling drag or stepping a source operator.

## core/black_holes.py::dm_wake_context_input_bindings

<!-- implementation-sha256: 7e57c10cd64866b6fec8b0a3766e326153a2d5f13eeda6d376434d358e35a370 -->

Validate context settings and verify the response, response-table and background-force products through _load_product. Return source pins under the declared memory budget. Matching actual operator/epoch/BH state and materialized table identity are checked during context publication.

## core/black_holes.py::publish_dm_wake_context

<!-- implementation-sha256: 5f5bda6437a5fd0b5728608da2bc91b4a291a64ecc7ad8b4d34a02b8d480b240 -->

Materialize a measured native state-response context only from a supported response/table pair and matching background operator, physics, epoch and original BH identity. Read bath density/velocity/covariance, require periodic nearest-image aperture support and verify the original regular table asset. Write sibling context/table/launch-assets files with SHA-256/CRC32 and explicit workspace/clock/conservation limits, then publish a numeric context-size audit. Return the export. Launch must still bind actual physics/current epoch; production enablement and native force equivalence remain unestablished.

## core/black_holes.py::dm_wake_context_runner

<!-- implementation-sha256: f1ae34e5512df29828e5690c16153b09ec49a179bf8339562437016a994d90f6 -->

Require context memory within the workflow budget and its publication inside the owned generation, then return publish_dm_wake_context. This writes measured context assets while preserving their production-enablement restrictions; it does not run native drag updates.

## core/black_holes.py::dm_wake_context_main

<!-- implementation-sha256: 624ade7f992ad6a393a4e524dd9562ba911b495a5a085ec294c2df66b9af7d45 -->

Read the context parameters JSON, set a default request_base from its parent and publish source-bound native context assets to --output. Return zero on publication and propagate validation/I/O failures. Output launch metadata is preparation evidence, not a completed source-operator integration.

## core/black_holes.py::validate_dm_wake_state_table_request

<!-- implementation-sha256: 9451ad7694d3b7729201960ccecbf852071f513240489a11c0b68d6c481bfb43 -->

Validate an explicit state-response reference and positive memory/serialization controls, bounding training states and native domain-solver iterations. Require at least three allowed states and enough table/workspace allocation. Return the request without loading coefficients or writing a native-readable table.

## core/black_holes.py::dm_wake_state_table_input_bindings

<!-- implementation-sha256: 0187e2a36cf270eeff3dd7cf1e3098ce4684db10a68db1645989c10af58ca880 -->

Validate state-table controls and return verified bindings from its response product under the split memory budget. This checks the response identity before serialization rather than proving that native domain evaluation has run.

## core/black_holes.py::publish_dm_wake_state_table

<!-- implementation-sha256: 5d95eebc3d59e4e2cc139fa594e1a178fa568fbcb41e87932ee5326e05e31b1b -->

Serialize a fully held-out-supported affine state response, requiring complete ten-coordinate scaling, unique active axes, overdetermined finite coefficients, reference errors and held-out support. Write a bounded ASCII sibling table with normalized training coordinates, coefficients/domain tolerances and original response signature, plus a numeric audit recording SHA-256/CRC32 and byte count. Return the export. Native projected-gradient domain evaluation remains separately residual/iteration gated; table publication does not establish production force equivalence.

## core/black_holes.py::dm_wake_state_table_runner

<!-- implementation-sha256: 172b3230dd7375e3e5bbb9a4bd6802c676c9786de434619cf4063e2e902c759e -->

Enforce state-table memory and generation ownership, then invoke publish_dm_wake_state_table. Return its serialization/audit result without widening the calibrated coordinate domain or running the native table reader.

## core/black_holes.py::dm_wake_state_table_main

<!-- implementation-sha256: 88dc989d588a40f8beb77b8fb4f68e5afcd700db946a77c61adb0a5a03c7d227 -->

Read state-table parameters, default relative input resolution to the JSON parent and publish to required --output. Return zero after successful export, leaving the native-domain/production limitations in the receipt; exceptions propagate.

## core/black_holes.py::validate_dm_wake_native_table_request

<!-- implementation-sha256: 718b4008ab6971d72f5b9f9ea47ba52e3cc3d6f3970f71c7b13edc18ae228c52 -->

Validate a measured speed-response reference, positive memory and explicit maximum knot/table-byte bounds. Require capacity for at least three knots and a minimum-sized serialized table, rejecting excessive estimated workspace. Return the request without exporting or evaluating it.

## core/black_holes.py::dm_wake_native_table_input_bindings

<!-- implementation-sha256: 12458a80131862a584abe9fc8f62eca746797b3ee939797986de002f2481607d -->

Validate speed-table controls and return the loaded response product’s source bindings within its input budget. This preserves the exact calibration identity before writing the native table; interval support remains in the response arrays.

## core/black_holes.py::publish_dm_wake_native_table

<!-- implementation-sha256: 4e36f7caad22096347e75cea59d29b74a9b8c1a90487b1df6320d4f9a5884b26 -->

Export finite increasing positive speed knots, nonnegative deceleration/error arrays, direction and per-interval calibration support from the calibrated speed-response product. Write a bounded sibling ASCII table and numeric audit with original signature, SHA-256/CRC32 and physical units, retaining unsupported intervals explicitly. Return the export. Serialization does not convert a false support mask into a calibrated native operating range.

## core/black_holes.py::dm_wake_native_table_runner

<!-- implementation-sha256: 960aea9fc335e5b57f7b08dab44d2175cebc499c05a6fe35fca5b927752146bd -->

Require speed-table export memory within the workflow allocation and a destination in its generation, then return publish_dm_wake_native_table. This owns the serialized product without launching a BH/DM simulation.

## core/black_holes.py::dm_wake_native_table_main

<!-- implementation-sha256: c8744c60cc77675cea9cff3610d62a67bc33108cb91cf04db75c54125f000d1a -->

Read a speed-table parameter JSON, default request_base to its parent and publish a bounded native-readable response table to --output. Return zero on successful publication, with errors propagated; the table retains its measured speed/signature restrictions.

## core/black_holes.py::dm_wake_response_runner

<!-- implementation-sha256: effbd0e68232075c63b8800ddd998d9ec230ed713fd27c7a4afefc20fd594d61 -->

Confine speed-response calibration to the generation and enforce its declared memory, then return publish_dm_wake_response. This runs the training/held-out calibration producer while leaving its original interval decisions intact.

## core/black_holes.py::dm_wake_response_main

<!-- implementation-sha256: fb9f3bb929846df8bb003dd1fb0ba1d1ab3a748a7975517ebeeff8128e4af4e1 -->

Load speed-response calibration parameters, default request_base to the file parent and export the local response to --output. Return zero after publication; inspect output interval support for calibration outcome rather than using process success as scientific acceptance.

## core/black_holes.py::dm_wake_ensemble_runner

<!-- implementation-sha256: f70fa853f049337c47203b30813fdf28359ec3ea7a14fac4a607a9f43bd39236 -->

Check ensemble memory and destination ownership before invoking publish_dm_wake_ensemble. Return its measured native-resolution ensemble result, retaining uncertainty and scalar-drag support decisions rather than changing acceptance thresholds.

## core/black_holes.py::dm_wake_ensemble_main

<!-- implementation-sha256: a67fa8b0c13eeb20d3f31b12a0db43087ae22a0a1d3dd43c02a8a5429e7cfffa -->

Read ensemble parameters JSON, supply the parent as default request_base and publish means/uncertainty from the selected measured wake contrasts. Return zero on publication and propagate errors. This analyzes existing realizations rather than launching independent runs.

## core/black_holes.py::dm_wake_contrast_runner

<!-- implementation-sha256: bdd0716ecbc3a500069ec7751d3428e9321d1888571aeffa708ec413121ca23a -->

Require contrast memory within the stage budget and output inside its generation, then return publish_dm_wake_contrast. This computes the selected measured resolution-pair contrast; it does not generate missing background/perturbed native runs.

## core/black_holes.py::dm_wake_contrast_main

<!-- implementation-sha256: e656c13e8f212585bfac6f72512260886f8477be98375aba3e8ae463c1b408c5 -->

Load resolution-contrast parameters, default request_base to their JSON parent and publish to --output. Return zero after the three-or-more-level contrast producer completes. Numerical support/rejection remains in the saved product and exceptions propagate.

## core/black_holes.py::dm_wake_force_runner

<!-- implementation-sha256: 89da74b9ad5197df1c378ef1d0f82a1ffdcce75e70eceacde27d0265c75f0dc9 -->

Require the complete native wake-force request to fit stage memory and its destination to remain inside the generation. Invoke publish_dm_wake_force and return its original measured force/bath publication; no evolution is launched by this adapter.

## core/black_holes.py::dm_wake_force_main

<!-- implementation-sha256: 530e7d225f22d08b9e0fbd1969de37e2b6465c5fe515230fdfc84d3018a9d469 -->

Read force-measurement parameters JSON and publish original native DM wake forces/bath moments under the declared probe operator to --output. Return zero after successful publication and propagate errors. Unlike other wake CLI adapters, this function does not insert a default request_base; required path/base controls must already be supplied by its parameters.

## core/black_holes.py::measured_epoch

<!-- implementation-sha256: 4311c2076e5d19dd3e0b988a21e1dfd0ae476f9cf4d3ba1c64e88232c05dcb20 -->

Read complete native BH snapshot shards after validating snapshot/unit/runtime choices against the requested BH settings. Require aligned finite type-five fields with explicit conversion metadata, converting positions to proper kpc, peculiar velocities to km/s, masses to Msun, rates to Msun/year and cumulative energy to cgs. Remove dead zero-ID/zero-mass carriers, reject duplicate living IDs or subgrid mass exceeding gravity-carrier mass, and rehash sources. Return aligned arrays plus epoch/source records, including an empty BH population when no live objects were saved.

## core/black_holes.py::bh_figure

<!-- implementation-sha256: d830239dd16dcb415ccd686f74fa8d787f660b736a81a24ce3f52e9cd950b37f -->

Verify the history HDF5 payload and render six native BH history panels versus scale factor: subgrid mass, mean inflow, model bolometric luminosity, gravity-carrier mass, mean inflow/Eddington cap and cumulative coupled energy. Show up to twelve IDs ranked by their largest recorded subgrid mass; retain the full catalogue elsewhere. Export PNG/PDF with history/source/epoch/unit metadata and close the figure. This plots measured histories and model luminosity rather than simulating an accretion-disk spectrum or establishing physical acceptance.

## core/streams.py::freeze_common_initial_stream_state

<!-- implementation-sha256: 813b95ea99d069f5ff841e7588a139ed905425d3ba4aedabd5acc9e034397287 -->

Validate the particle sample, copy its proper-kpc positions, km/s velocities, Msun masses and exact IDs, and hash their bytes together with the namespace. Make copied arrays read-only and return the common state. This preserves identical initial conditions across imposed-field interventions; it does not freeze the external potential or integrate trajectories.

## core/streams.py::integrate_stream

<!-- implementation-sha256: 3555c2648606dd5879668f653926b016620ec5ad78f97214e04c129cece156b9 -->

Integrate bounded collisionless tracer chunks with DOP853 in an inertial imposed field at requested Myr outputs. Convert km/s consistently to kpc/Myr, enforce tolerances/maximum step/force-call/memory limits and return histories of positions, velocities, potential, specific energy, angular momentum and torque. Optional host coordinates subtract measured centre motion and centre acceleration for host-relative torque. Return report/arrays; self-gravity is absent and convergence remains unexecuted. Import through core.stream_forward_model, since the physical historical streams.py file rejects direct import.

## core/streams.py::measure_track_density_velocity

<!-- implementation-sha256: 90a4b606b6aba1a1d1c80d8bedd4accf73170af66c956c6c982e75cabe5227f4 -->

Project each evolved state onto the supplied track and bin arclength in proper kpc. Return weighted linear density [Msun/kpc], counts, transverse centroid/covariance, mean local velocity [km/s] and count-based support; unsupported moments remain NaN. Optional weights must align and be nonnegative. Retain a Lagrangian ordering Jacobian relative to initial labels, without assuming a folded multistream track has a unique inverse.

## core/streams.py::compare_control_effects

<!-- implementation-sha256: 046a9115c884d7392cfadd681af1b2be0287622bf5f997a5f8632b09cb302445 -->

Require identical particle order and output times, then subtract reference from counterfactual positions/velocities. Bootstrap the final six-component mean and add an explicit positive-semidefinite model-error covariance. Return comparison metadata and difference arrays, with density ratios/support where measured. Covariance axes mix kpc and km/s as declared; this paired imposed-field response alone does not identify a unique dark perturber.

## core/streams.py::run_counterfactuals

<!-- implementation-sha256: a13c5ce70e119d80c0b45a1cb9f77a9bd30fc73c14b7887e1cd5443051e508a6 -->

Freeze one initial stream and separately evolve each declared component intervention, using collisionless tracers or the live stellar-progenitor reference. Measure track observables, optional actions in a declared field, Fourier/torque and pattern histories. Pack intervention arrays plus paired differences against the named reference and return report/arrays. Preserve requests for native live-host branches when backreaction is needed; a prescribed host does not supply self-consistent wake/deformation response.

## core/streams.py::validate_native_counterfactual_ic_request

<!-- implementation-sha256: a80ed9712a644ca8d400c118929ace57d5d605256462ed1c85847e79687a2cad -->

Validate exact native intervention controls for DM removal, nonunit positive mass scaling or a late unchanged-state control. Require exact disjoint affected/stream IDs, species/namespace, bounded particle/chunk/memory/disk work and a saved snapshot for late requests; late collisionless branches cannot carry gas streams. Return the request. This defines which initial-state change is allowed without writing or evolving ICs.

## core/streams.py::native_counterfactual_ic_input_bindings

<!-- implementation-sha256: 43b5f3d8214f3b278b260bb2598adaf643dfbb0835965d1e18f07e0d8d98f086 -->

Bind a validated intervention to its completed native-stage sources and exact original regular single-file IC. For late-state requests require that file to be the selected native snapshot. Return source pins, rejecting changed bytes or unrelated producer ownership. This input check does not perform the intervention.

## core/streams.py::native_counterfactual_ic_runner

<!-- implementation-sha256: e798665ef53215eeb4cc1ad3eb9c7e6d22aa257bbd835cf132e4badd22794fd2 -->

Enforce memory/disk allocation and generation-contained destination, then return publish_native_counterfactual_ic. This workflow adapter materializes the explicit changed or retained native state; it does not evolve the branch.

## core/streams.py::native_counterfactual_ic_main

<!-- implementation-sha256: 3e9106603853d9a996b2d932301b4cbd6964a6b816760d04b83ed0d3d6e35314 -->

Read native intervention parameter JSON and required --output, invoke publish_native_counterfactual_ic and return zero on publication. Exceptions propagate. This CLI does not insert a request_base default or launch evolution; parameter paths must satisfy the publisher’s input rules.

## core/streams.py::validate_native_counterfactual_run_request

<!-- implementation-sha256: d78a472de4bbbdcd5c3dfc18b6bd14610a6884bacec53a14b5278dbe17a4350c -->

Validate an explicit branch-execution request with hash-bound intervention lineage and MPI launcher, positive rank/thread/storage/time limits and saved-output byte/count bounds. Return the request. Actual parent configuration/binary/IC compatibility is checked by the binding reader before execution.

## core/streams.py::native_counterfactual_run_input_bindings

<!-- implementation-sha256: aee01f70b3895fe971db64fddf312f7a612bad703ef0df527cf7cdb12185d36b -->

Verify original completed native-stage ownership, parent configuration/binary/parameters/compile flags and the sealed intervention lineage. Require original MPI/OpenMP decomposition, self-gravity and exact synchronized outputs; reject SIDM branches lacking a regenerated support/model contract. Verify the changed IC and executable pinned launcher, then return request, parent/config/lineage/IC/launcher and source pins. This prepares original-engine execution without invoking it.

## core/streams.py::validate_native_stream_worldline_request

<!-- implementation-sha256: 96c6037c4a880961490a58f4a0da491cbe1fd7c872edcb48ef5e8a223eb9232e -->

Validate exact stream species/namespace and a bounded unique particle-ID selection across at least two saved snapshots. Permit the specific deferred parent/branch snapshot-list references used by the scheduler, with worst-case workspace checks. Return the request; this does not read trajectories or infer missing identities.

## core/streams.py::native_stream_worldline_input_bindings

<!-- implementation-sha256: dc56cfcb83d25e776601309d9d929070322c0a44f4650a93536d3ec08c97261b -->

Validate the native worldline request and return the shared completed-native-stage source bindings. This preserves the actual producer authority before selected particle histories are read; no interpolation or native evolution occurs.

## core/streams.py::native_stream_worldline_runner

<!-- implementation-sha256: 4ad26f3b5f8b8b59c93be7855d505df508e12b0e1fc0eef7ef1839afc210ed66 -->

Enforce stream-history memory within the workflow allocation and destination inside its generation, then invoke publish_native_stream_worldlines. Return the selected native history publication without assigning a new physical stream identity from proximity.

## core/streams.py::native_stream_worldline_main

<!-- implementation-sha256: dede50617e979b08cf4bf4722839047d7aa55bbea8a0b4a8c9f4c2030c942e01 -->

Read native stream-history parameters JSON and publish selected original trajectories to --output. Return zero after publication and propagate errors. The adapter supplies no default request_base and does not integrate new orbits.

## core/streams.py::native_counterfactual_campaign_main

<!-- implementation-sha256: 4fcdfacfb4983489669550b1f84eba77ae30cbfeb4da2dd1f6a92c0b80066fee -->

Read a native live-response campaign request and compile its follow-up workflow into --output using the request parent as base. Return zero after preparation. Compilation creates executable recipes; it does not establish that their native runs or comparisons completed.

## core/streams.py::validate_native_counterfactual_campaign_request

<!-- implementation-sha256: a1ef1575465d2384ade1d3d524ea3dca1eff257b269793f43fe2a65571e36dab -->

Validate the original live-response campaign structure, bounded intervention count and typed membership/resource/measurement controls; delegate late saved-state requests to their dedicated validator. Optional host-reflex controls must provide valid worldline selection and comparison limits. Return the request. Detailed ownership and source bindings are checked separately before compiling concrete branches.

## core/streams.py::native_counterfactual_campaign_input_bindings

<!-- implementation-sha256: a08ef4f6295e346a95a5685340aa006e96acf15ae0567dca91f8923bdf1cb768 -->

Validate the campaign, delegate late variants when applicable and verify counterfactual, live-host and component-membership products. Hash-bind parent stage/IC/launcher and its completed native run sources. Return pins. This establishes inputs for the automatic branch recipes without executing them.

## core/streams.py::validate_native_counterfactual_result_request

<!-- implementation-sha256: d126c055cdc7b3f6cf9eeaaca53001013c0ad3491312820f2444cb1b37091498 -->

Require the complete native intervention result schema with comparison, run report/receipt, host membership and positive census/support/memory controls. Return the request or raise. This validates requested evidence structure rather than concluding that live-host response was measured.

## core/streams.py::native_counterfactual_result_input_bindings

<!-- implementation-sha256: 47d3a12d1ef69feb47de3e4fa0688bb5a25000de97ca7217d687701cac1bdcbe -->

Verify comparison and host-membership numeric products plus exact run report/stage-receipt references relative to request_base. Return the combined source hashes. The result producer subsequently interprets native evidence; this reader does not fabricate missing host support.

## core/streams.py::native_counterfactual_result_runner

<!-- implementation-sha256: 7069d4b9d4d01bf46628c4ae01f33ee47defeabc3450f0fafcd6d68bb66aed42 -->

Enforce stage memory and contained destination, then publish_native_counterfactual_result and return its actual evidence report. This adapter summarizes a completed native intervention/comparison rather than launching a new run.

## core/streams.py::validate_native_stream_comparison_request

<!-- implementation-sha256: 9fa7e03f9486e8b2566e8a7d01e98f3d2f7f7f7ac92cd343eecb431b025610fc -->

Validate paired native stream references, proper-time tolerance, bounded saved/particle work and periodic displacement support strictly below half a box. Optional host-reflex records require an exact anchor, supported host census, local host/stream radii and combined memory bound. Return the request. Actual ID/time/frame equality is checked by the comparison producer.

## core/streams.py::native_stream_comparison_input_bindings

<!-- implementation-sha256: e07f4a8756e11eda09e3270bc2dd892d67f77d926b9de029afa69ede906ea03e -->

Validate paired-stream controls and source-verify both histories, including both host histories when reflex measurement is requested. Return combined pins under split memory budgets. This does not subtract trajectories or establish support from labels alone.

## core/streams.py::native_stream_comparison_runner

<!-- implementation-sha256: f15a25c6112a038b8425ed8832e5b930d9c6e5bf83caaa1450a5ffb75c3f1e12 -->

Require comparison memory within the stage allocation and output within its generation, then return publish_native_stream_comparison. The producer retains measured epoch/identity/periodic support checks; this adapter does not loosen them.

## core/streams.py::native_stream_comparison_main

<!-- implementation-sha256: 5020ac9aee41eed60a99c75cebc16db54ee79eb78e430e1d1140d18ab6fba6eb -->

Read paired native comparison parameters, default request_base to the JSON parent and publish to --output. Return zero on publication with errors propagated. Compare actual saved self-gravitating branches; this command does not replace them with imposed-field trajectories.

## core/streams.py::validate_native_late_campaign_request

<!-- implementation-sha256: 56561bdbe4d599dd584db24e5bd76a201fa22c4d8e06141bfa5641ea5f2b5cea -->

Validate the complete saved-state parent/control/intervention family. Require one explicit changed late state, an identical retained stream and matching run/comparison controls; generate the unchanged control later. Optional host-relative frames need a distinct unchanged same-namespace DM cohort with anchor/support. Bound all retained states/runs/measurements, resource decomposition, stages and follow-up depth. Return the request. This validates a full decomposition plan without executing any branch.

## core/streams.py::native_late_campaign_input_bindings

<!-- implementation-sha256: 0bcd0e8c2e254eece5c351520174033848ee878491b9007b50fbbb13c9981c52 -->

Validate the late campaign and bind its exact parent receipt, original saved-state intervention inputs and pinned launcher relative to request_base. Return the original native source pins. This checks the retained origin of the late experiment without modifying its source snapshot.

## core/streams.py::validate_native_late_threeway_request

<!-- implementation-sha256: 208ec8836e894fea50ab9fed74163001d10839287788893d56773bca136cf6ba -->

Validate the declared inertial or host-relative three-way comparison, exact references to three differences/two native runs and bounded saved-particle workspace. Require positive signal floors and nonnegative restart-error limits, with restart contamination fraction below one. Return the request. These frozen budgets do not demonstrate that the measured differences satisfy them.

## core/streams.py::native_late_threeway_input_bindings

<!-- implementation-sha256: 182e30004f3ef7906cad7a016888a68bc766176361ccb34e6bf63fd71323ec0c -->

Return source pins from the full _late_threeway_inputs reader, which loads the exact comparison/native-run family. This binds evidence for later decomposition without computing new differences or correcting a restart residual.

## core/streams.py::publish_native_late_threeway

<!-- implementation-sha256: 8dd8e2cfbcebe85c24865e9390bdc8c4b9908609b3a788d5d7312d16393647a1 -->

Require identical selected IDs, initial masses, namespace, epoch/domain arrays and optional host-frame definition across parent/control/intervention comparisons. Decompose total difference into unchanged restart plus intervention signal, checking closure at a roundoff-scaled bound with periodic position handling. Compare restart norms against both absolute ceilings and a fraction of the signal floor; publish all differences, closure, limits and support decisions in proper kpc/km/s. Return a numeric audit with supported/unresolved reinitialization status. No cancellation correction, independent-resolution accuracy or global causal acceptance is inferred.

## core/streams.py::native_late_threeway_runner

<!-- implementation-sha256: 107a701ebe37c32c7e3df40274cffb53ad2be865dc0c3367a9248295ea5acc6e -->

Check the three-way producer’s memory and output ownership against its workflow generation, then return publish_native_late_threeway. This performs the retained decomposition without rerunning the parent or subtracting a corrective trajectory from its evidence.

## core/streams.py::native_late_threeway_main

<!-- implementation-sha256: bba40f263607ebbf16370e1b873a316e722377d5bdf512b3014cf706276c32a6 -->

Read three-way parameters JSON, default its request base to the file parent and publish to required --output. Return zero after publication; the saved product’s restart-budget status remains authoritative and errors propagate. This CLI measures existing parent/fresh-control/intervention differences rather than evolving those runs.

## core/research/measurements.py::validate_measurement_request

<!-- implementation-sha256: f9631c52f1d1cd93ee896da28e0a263a693d30dc37cb25de2ef0d0bfa6fecba9 -->

Validate the frozen native-convergence measurement schema, explicit particle/bin/memory bounds, unique bins, comparison epoch and native-parameter conversion paths. For spherical profiles require the estimator’s exact physical units, increasing proper-kpc edges, fixed centre/bulk velocity, explicit species and sampling thresholds, fixed octant jackknife partitions and any periodic half-box restriction. Return the request or raise before reading particles; numeric-product mode delegates its product-specific checks to the reader.

## core/research/measurements.py::measure_numeric_product

<!-- implementation-sha256: c25b5326480364b9f1d596b10e4c2da1cef1fd2a9aa3eb918f2485551cec78f9 -->

Bound the expanded numeric archive against declared memory, verify its audit, and require the exact frozen observable hash, units and bins. Copy the named value, covariance and boolean-support arrays from arrays.npz; require a recorded uncertainty method/scope. Return those arrays, measured epoch, original source bindings and uncertainty metadata. This reads a measured product rather than estimating covariance from invented samples.

## core/research/measurements.py::produce_convergence_measurement

<!-- implementation-sha256: de67c8fb342f99e75d4d515722294d79dc3751d18b8371c492ea3b5951ea1542 -->

Verify the planned variant/specification, actual native seed/control settings and completed evolution receipt, then measure a spherical profile or read an audited numeric product. Require the declared epoch, array shapes, finite supported values and positive-semidefinite supported covariance. Controlled variants must bind their exact baseline; derive paired cross covariance from common jackknife replicates or require the retained numeric cross covariance, and check the joint covariance. Rehash sources and return a MEASUREMENT_ONLY record with unsupported values/covariances null. Units follow the frozen observable; measurement completion alone does not establish convergence.

## core/research/measurements.py::convergence_measurement_runner

<!-- implementation-sha256: b34363d174633db189db8c39580706cd35a923d760f07ce0e713f7b97c79ea3d -->

Require requested measurement memory within the workflow allocation, produce the native convergence measurement, and atomically write convergence-measurement.json plus its SHA-256 seal inside the generation. Return bin counts and explicit insufficient-sampling exclusions for workflow reporting. This adapter measures existing native outputs; it does not evolve them.

## core/research/measurements.py::convergence_measurement_main

<!-- implementation-sha256: ac0c993b268a7ea4c6324baaa5936ad2ab1dd2197e0edfc944eb1a71f538ac2b -->

Read --parameters JSON, produce the native convergence measurement and write required --output plus its SHA-256 seal. Reject an existing output to retain prior evidence. Normal completion returns None; failures propagate. This command does not launch the native convergence family.

## core/research/measurements.py::convergence_budget_collection_main

<!-- implementation-sha256: b8a06c5185b60f295adf9d93786f700a1f44442bacba452aebab41a6424ccebf -->

Read --request through a bounded 16 MiB JSON intake and publish the convergence-budget collection relative to the request file’s parent. Pass --execute and --resume to the publisher and print its JSON result. This CLI connects the collection workflow; its returned execution and science statuses determine what actually completed.

## core/research/measurements.py::validate_isolated_observable

<!-- implementation-sha256: 3217ab94707ee27a01e3fff0168f3fcea26951e61cb42c9f18b3d8bbf3efee32 -->

Require a named supported saved-history estimator with its exact units, nonempty distinct physical bins and explicit positive memory/bin bounds. Bound dense covariance workspace and require the uncertainty scope conditional_deterministic_saved_history. Raise on invalid input; normal completion returns None. This scope describes reduction of retained outputs rather than native-realization scatter.

## core/research/measurements.py::isolated_observable_input_bindings

<!-- implementation-sha256: b06dcb6d401390fd0e72400e3aa6a367debfc7105c2b621d0928315d6736e36f -->

Validate the saved-history observable and obtain the retained history’s source pins. If a paired baseline is supplied, enforce its memory bound and SHA-256 publication seal, then include both baseline and seal hashes. Return the combined input bindings before observable reduction.

## core/research/measurements.py::publish_isolated_observable

<!-- implementation-sha256: 725d313f5b745b9760fba485a9c03660beccd64710a718a0be44cc13c6786941 -->

Reduce a completed isolated-halo history under the exact declared aperture, frame, gauge, boundary and collapse definition. For each requested saved epoch retain energy, relative drift, core density or core mass; unavailable epochs and unsupported cores remain excluded. For persistent crossing use the midpoint and half-width of the actual saved Gyr bracket, excluding unbracketed collapse. Publish values, intervals, support and conditional covariance through a numeric audit; supported covariance is zero only for this deterministic saved-product reduction, not total physical error. Optional paired conditional baselines supply zero cross covariance after compatibility checks. Rehash sources before publication and return the audit.

## core/research/measurements.py::isolated_observable_runner

<!-- implementation-sha256: 50f492b437f6a1a767d009e361bf63915d073c1464de6a08cf16ae1f198bce94 -->

Enforce saved-history measurement memory within the workflow allocation and destination ownership inside its generation, then return publish_isolated_observable. This stage publishes the retained observable and its uncertainty scope without rerunning native dynamics.

## core/research/errors.py::publish_budget_and_domain

<!-- implementation-sha256: 083e989e93a5df165fe06ec5b053698bbc6b122e5885431fa27a99ce54b51345 -->

Require a fresh destination, evaluate the masked measured error budget with optional joint covariance, and atomically write error-budget.json, frozen-requirement.json and supported-domain.json. Return the evaluated budget. The destination is immutable evidence; scientific acceptance comes from the evaluator’s per-bin result rather than successful file creation.

## core/research/errors.py::publish_correlated_ensemble

<!-- implementation-sha256: 1154913d752ddabb1f6aff1c79491035d62183e1051edc95a34adcfad7ca81f2 -->

Lock a fresh destination, measure the retained correlated ensemble and publish ensemble.json plus separate request, measurements and joint_covariance JSON files. Also write error-budget-input.json for the next evaluation stage. Return publication paths and source hashes with execution COMPLETE and science acceptance NOT_ESTABLISHED; this publication does not itself accept the error budget.

## core/research/errors.py::error_ensemble_main

<!-- implementation-sha256: 3fc3362392e6c11b90dcd3ffdb6830b7ff9211f1cc99c156099b11839475960d -->

Bound --request metadata to 16 MiB, load its parameters and set request_base to the file parent. Publish the measured correlated ensemble at --output and print its JSON publication result. Normal completion returns None; retained-input and publication errors propagate.

## core/research/errors.py::native_residual_runner

<!-- implementation-sha256: a5af45281efe656fdfc5c97bcd7ac32dc28b9aa5080b4e9d3b70b5bb8399d606 -->

Require a positive integer residual workspace within the workflow memory allocation and a destination inside its generation. Publish the native control residual, write residual-publication.json and return its publication record. This stage measures retained contrasts rather than launching control runs.

## core/research/errors.py::produce_native_control_residual

<!-- implementation-sha256: 9a64661ae58f9a640658843d766c6e004c5969ac8f0a7890f1af08a120fb17bb -->

Bind a frozen error requirement to an actual completed native realization. For every component require the permitted control, distinct baseline/variant runs, exactly one changed control and matching physical bins, units, model, observable, phases and unrelated operator seed. Form signed variant-minus-reference residuals only on intersecting support, sum deterministic interval widths and retain nulls elsewhere. Optional direct SIDM baseline samples require an operator-seed ensemble and the explicitly frozen random-error target. Rehash sources and return component-by-bin samples, support, seeds and paired-family identity with science acceptance NOT_ESTABLISHED. Cross-realization covariance is measured later, not inferred from one contrast.

## core/research/errors.py::publish_native_control_residual

<!-- implementation-sha256: 8bae3c0b23c51a836ef6534ab12d1668b4c0df2e3948c1a57a939e4ce4b280ef -->

Lock a fresh immutable destination, produce the retained native control residual and atomically write native-control-residual.json. Return the product path, source hashes and COMPLETE publication status while keeping science acceptance NOT_ESTABLISHED.

## core/research/errors.py::error_residuals_main

<!-- implementation-sha256: eb4dc1411683879ffe4deb2d028e2be9fbfdaf57c216f630e794d6fa48bfd894 -->

Load required --request JSON after a 16 MiB metadata bound, resolve request_base from its parent, publish native control residuals to --output and print the publication record. Normal completion returns None; this command does not execute missing native variants.

## core/research/errors.py::execute_budget_stage

<!-- implementation-sha256: 66e293b1d83face97b7f8a2c565881064d2940016478ab17c7ffca45ad54b282 -->

Bound dense correlated-covariance workspace against stage memory, evaluate the masked error budget and construct bounded refinement requests for the explicit parent stage. Optionally append reassessment, then write the budget, frozen requirement and supported domain in the generation. Return follow-ups, unresolved findings/exclusions and supported/accepted bin counts. COMPLETE execution retains the evaluator’s separate scientific acceptance; dispatch requests do not mean child simulations have run.

## core/research/errors.py::error_budget_main

<!-- implementation-sha256: 252d204f35d6bc167e8c91cde0df1e86ef53dc334e249db201e3d2583f0b6426 -->

Load the positional request JSON containing keyword arguments for publish_budget_and_domain, publish to required --output and print the result. Normal completion returns None. Measured component values and optional joint covariance must already be present; this command evaluates them rather than generating control measurements.

## core/research/errors.py::validate_envelope_parameters

<!-- implementation-sha256: 89b0fa525ce23eb6e82bb110162ab0d4cfffcd0831404ed8876145f403931a51 -->

Require a frozen error requirement, permitted component/control mapping and explicit bound scope matching the component’s declared systematic target. Certified-reference mode also requires the exact physical-target hash. Check positive memory/pair limits, nonempty bounded pairs, covariance workspace and exact reference/variant/certificate keys. Raise on invalid input; normal completion returns None. Validation alone does not certify the reference.

## core/research/errors.py::envelope_input_bindings

<!-- implementation-sha256: 27cfd37f4080eb9b51a2631c669620da319f3e34310512f8b1ffeb1f21ed8094 -->

Validate native envelope parameters, read each retained reference, variant and required certificate under the memory bound, and merge their source hashes. Reject conflicting hashes for the same path and return the bindings; no envelope is calculated here.

## core/research/errors.py::measure_native_error_envelope

<!-- implementation-sha256: 391fb25d15313dffb1da8b02b684869c89c65a29e1c9e3f66a26c37c212f384a -->

Compare exact native baseline/variant pairs at one model, observable definition and control setting, rejecting duplicate pairs. Use the paired contrast error bound plus both deterministic input widths; certified-reference mode additionally verifies the target, actual reference provenance/domain and adds its absolute upper bound. Intersect support across all pairs and retain the maximum bound per commonly supported bin. Rehash sources and return bias_bound in observable units, retained pair bounds and the measured domain. A declared control contrast remains CONTROL_ONLY; certified mode is limited to the certificate domain. The frozen sigma multiplier does not by itself imply a confidence level.

## core/research/errors.py::publish_native_error_envelope

<!-- implementation-sha256: 317cff3f2ad79e09ce9a609d9b5039a998f6adc3fcb98328dff0427dc6ede92b -->

Lock a fresh immutable destination, measure the native systematic envelope and atomically write systematic-envelope.json. Return its product path, COMPLETE execution status and the measurement’s actual science-acceptance scope.

## core/research/errors.py::native_error_envelope_runner

<!-- implementation-sha256: f5446cbe421848a8d04736dcc46a282aab5d183fd19e03735579cac1a049c9de -->

Require envelope memory within the workflow allocation and its output destination within the generation, then return publish_native_error_envelope. This adapter publishes a measured bound without expanding its physical or certificate domain.

## core/research/errors.py::native_error_envelopes_main

<!-- implementation-sha256: d1fd17fbb9a000c50d1067403283765f7bd3250a352ec9c7d247650c28f7632a -->

Require --request and --output, enforce an eight-times-metadata expansion estimate within 16 MiB, load parameters and set request_base to the request parent. Publish the native systematic envelope and print its JSON result. Normal completion returns None; absent or invalid retained controls/certificates raise.

## core/display_fresh_load.py::main

<!-- implementation-sha256: bdadfcebc788c543d8ee8d7560dea65d5b0341fb56631633bd2620146de677a1 -->

Run inside Blender after freshly opening the saved scene. Read the output path after Blender’s -- separator, explicitly execute packed display-runtime and object-inspector text, and test visible namespace/ID identities plus primary/layer geometry at every saved epoch. Check hold-last-state frames, hidden objects before/after coverage, inspector registration and every manifest asset hash. Write a blender-fresh-load-v1 JSON receipt including the blend hash, then raise if any check failed. This verifies saved-scene reload behavior without relying on automatic script execution; it does not rerun physical dynamics.

## core/display_runtime.py::object_key

<!-- implementation-sha256: fd153089737f5cc4ccf9291915c26dc8cfa5e0fcf3e5dceada1693dd334197c3 -->

Return the exact namespace plus colon plus stringified object_id from a measured display record. This identity keeps objects from different native namespaces separate; no proximity matching or ID remapping occurs.

## core/display_runtime.py::apply_epoch

<!-- implementation-sha256: 7645227283f0dcc1322a0206b3b32012c52e3cf7a0561b9f624fe3c09a1b16f6 -->

Parse and cache the scene’s embedded display timeline, select the latest saved epoch at or before the current frame, and update visibility only for objects owned by that display generation. Hide them outside measured coverage. When enabled, follow the active exact namespace/ID and geometry-layer role into the selected epoch. Update current epoch/snapshot scene metadata and return None. Intermediate frames hold measured state; this operation does not interpolate positions, create births or calculate physical substeps.

## core/display_runtime.py::frame_change

<!-- implementation-sha256: f8ca3171e1f62bb0ad647324b83b9691adcff2115fc40bec23ea849c3e686d4a -->

Blender frame-change callback: import bpy and apply the saved display epoch only when the scene has a research_display_manifest. Normal completion returns None. This callback changes display visibility and epoch metadata, not simulation state.

## core/display_runtime.py::load_post

<!-- implementation-sha256: 37c4d459eefc5b426f1f902fcb5319ded2a95c4291bd54618e2af3bf3c34eefe -->

After Blender loads a file, clear the parsed-epoch cache, apply the current saved epoch to every scene carrying a display manifest and reinstall persistent handlers. Return None; scenes without measured display metadata are left alone.

## core/display_runtime.py::register_handlers

<!-- implementation-sha256: 50c9381689f0dcf4ae578cb07767872cac308b46761aa86b5f130f476103532c -->

Install persistent frame_change_post and load_post callbacks, first removing prior handlers carrying the same display tag or recognized module/name. Return None. This prevents repeated runtime loading from accumulating duplicate callbacks while retaining unrelated Blender handlers.

## core/display_runtime.py::install_runtime

<!-- implementation-sha256: 952bbf9cae4fc85c98161b510ba18f37d95d44b59aac3ee82e01709a8168b4fd -->

Install one bounded measured-display generation into the active Blender scene. Validate object/geometry/optional vertex budgets, allocate a generation-owned collection, store the manifest, physical coordinate units, lifecycle index and asset base, then create epoch objects and measured geometry layers with exact identities/source links. Add snapshot timeline markers, embed the runtime Python text and register handlers. Optionally preserve an existing clock after requiring measured coverage to fit it; otherwise use the measured endpoints. Apply the initial epoch and return None. Mesh/curve geometry can be symbolic markers, and curve thickness is a display choice; only supplied measured points, axes and tensors encode those physical shapes.

## core/flow_movie.py::sample_flow

<!-- implementation-sha256: 48809e7496180ac155201293ba5ebc0628e8ce6922e46f932f6abd2e96f69368 -->

Bilinearly sample the two projected velocity components at normalized periodic display positions, using grid-wrap coordinates offset to cell centres. Also interpolate the boolean support field and return velocities plus a validity mask requiring support weight at least 0.999. NaN component cells are filled with zero for interpolation, so callers must honor the returned mask. Values retain the input velocity units; these positions are illustrative tracer coordinates.

## core/flow_movie.py::make_paired_flow_movies

<!-- implementation-sha256: 5bcdf54c4943b01fad962be92c83c16537c14775515e78655d96de862395e7aa -->

Read hash-bound EDE/LCDM Peculiar_velocity_XY fields from the paired atlas’s fixed model folders and use their hard-coded h values to prepare density backgrounds. Animate 1200 seeded display tracers in each frozen projected km/s field with periodic midpoint steps, common speed scaling and respawn at unsupported cells or display ages. Write individual and comparison MP4s, a midpoint Poster.png and film_source.json receipts with field/movie/algorithm hashes, then return movie paths. Snapshot/redshift select and label the saved product; animation time is illustrative, and tracer paths are not native particle histories. This archival movie producer is separate from the video-free optional lab.

## core/halo_formation_movie.py::tree_layout

<!-- implementation-sha256: db2670e97405754222ff13b8840c7bf7c9c25a836bdb05fb5cf12b1ddf07bd19 -->

Require each indexed merger-tree edge to point from a later-listed progenitor to an earlier-listed descendant. Traverse nodes in reverse order, assign sequential x coordinates to leaves and place descendants at the mean x of their recorded progenitors. Return dimensionless diagram coordinates; they are branch-layout positions rather than spatial halo coordinates.

## core/halo_formation_movie.py::make_halo_formation_movie

<!-- implementation-sha256: 014e1976153d000e773be2b2e3e1707ed6701c0a509fa363dd179c0fcb370457 -->

Verify a positioned linked tree and globally normalized map publication from the same run/box, with at least two distinct saved epochs and consistent projection, slab and map window. Render saved density beside the linked ancestry diagram; overlay only finite saved catalogue positions inside the periodic slab and retain exact branch/ID labels. Repeat each measured frame for fps times seconds_per_snapshot, without interpolating halos. Publish halo_formation.mp4 and movie_contract.json in a fresh destination after rechecking inputs, returning the movie path. Map coordinates are comoving Mpc/h; the diagram uses scale factor. Enforce raster-memory bounds and retain staging failure.json on error. This producer is not run by the video-free optional lab.

## core/cosmic_flythrough.py::unwrap_camera_path

<!-- implementation-sha256: 9e27d63b5eb8b5a64ef38f3ff9bd2ead4d514204d7374e0f6887df77b4a7f6d6 -->

Validate at least two finite xyz waypoints and a positive periodic box, wrap points into the box and cumulatively unwrap minimum-image displacements. Reject consecutive waypoints that coincide within box times 1e-10. Return continuous camera coordinates in the input length units; this is a presentation path rather than a particle orbit.

## core/cosmic_flythrough.py::camera_basis

<!-- implementation-sha256: 6dea5402f8b0245912c2f277e1f60b5431455d050c1ce8e13eb4447142d85c13 -->

Normalize a nonzero forward direction, choose a reference up axis away from its pole and return perpendicular right/up unit vectors using cross products. The helper assumes a valid finite direction; perspective_mass_raster validates it before calling. This defines a camera basis and carries no physical units.

## core/cosmic_flythrough.py::density_palette

<!-- implementation-sha256: 75945ac4ed284309d1b3eafd9b648f89f60161d521e5b42f64d3a7ca44a3cac9 -->

Return the custom navy-to-cyan-to-cream Matplotlib colormap used by fly-through density displays. This maps normalized display values to colors; it does not compute density or stellar emission.

## core/cosmic_flythrough.py::perspective_mass_raster

<!-- implementation-sha256: 2e6662e25bb181cf9581083de52bf70929cccb5801845854c5288e6033187ac3 -->

Validate aligned finite positions and positive represented masses, finite camera/nonzero direction and a positive periodic box. Project minimum-image particle displacements through the requested perspective field, retain depths between 0.005 and 0.85 box lengths and points inside the image, then mass-weight a 2D histogram and apply 0.7-pixel Gaussian display smoothing. Return the float32 raster and visible sample count. Raster values represent sampled mass per display pixel before boundary smoothing, not surface density per physical area or radiative intensity.

## core/cosmic_flythrough.py::make_cosmic_flythrough

<!-- implementation-sha256: 53999c74bfdacbfff74f448d436161c4c7c02109e8b3fbc70defa3034e8a81a0 -->

Lock a saved run, bind the exact host catalogue row and a bounded DM sample, and construct an eased periodic waypoint camera path ending at a finite halo standoff. Build a complete 128-cubed, smoothed density/mean volume from saved DM and average perspective rays; overlay sampled mass glints with fixed film-wide display exposure. Write an MP4, preview, original particle/camera arrays, volume, per-frame rasters and a hash-bound flythrough_contract.json into a staged fresh destination. Recheck particle/catalogue provenance before publication and return the movie path; retain failure.json on error. Lengths are comoving Mpc/h, and camera motion through this frozen snapshot is not physical evolution or emitted light. Frame/sample/raster-memory limits bound presentation work. This movie command is separate from the video-free optional tests.

## core/flagship_runtime_display.py::validate_scene_options

<!-- implementation-sha256: 1133a9fadc0dee85d6e229cca3a7ab3ea1b093f533c941cd3c3a3c754f42d469 -->

Require the complete explicit scene sampling, selection, geometry, marker and encounter-overlay controls. Validate positive bounded counts, preview/high quality and distinct render IDs with complete ordered frame ranges, dimensions, samples, fps, device and wall-time settings; dome renders must be square. Marker fields declare kpc or Mpc/h in their names. Raise on invalid options; normal completion returns None. Resource allocation and measured input ownership are checked separately.

## core/flagship_runtime_display.py::display_preparation_inputs

<!-- implementation-sha256: 142ecdb0b3d284ea0c3e1e04473e1034ab3f4da2188db306e44142a28ebb070d -->

Require the exact flagship runtime-display request schema and resource/metadata/figure limits, validate scene options and estimated render storage, then read the completed history-case inventory. Add this implementation’s hash and return case records plus source bindings. This checks retained inputs and capacity without creating Blender scenes or executing render recipes.

## core/flagship_runtime_display.py::display_preparation_runner

<!-- implementation-sha256: ff9a855af7ad4e230953d877c6d1f72e54498c7032a18c73a0eb4b16c36cce60 -->

Require requested memory/disk within the workflow allocation and inspect owned completed history metadata. Build follow-up stream inspectors, a combined scientific gallery and per-case base/encounter scene recipes, retaining unresolved history states explicitly. Bind all recipe inputs to original source hashes and write display-preparation.json with RECIPES_PREPARED and native_scene_execution NOT_EXECUTED. Return follow-ups and gallery/scene counts. Preparing recipes does not establish that Blender or renders completed.

## core/flagship_runtime_display.py::runtime_scene_runner

<!-- implementation-sha256: 6e42d12a6774d052dffbf3f41c89fe66e7419072355514e486ff785245a10e99 -->

Bind the completed native evolution and verified combined gallery, optionally retain private analysis-copy lineage, and select views tied to the actual run. Under scene/render memory and disk bounds, export attachments and build blender_universe.blend, then invoke each requested render-scene subprocess with its own wall limit, log and invocation receipt. Skip unsupported native-focus renders as unresolved rather than inventing a centre. Rehash original evolution inputs, write scene-result.json with scene/manifest hashes and rendered/unresolved IDs, and return counts. COMPLETE publication retains science acceptance NOT_ACCEPTED; saved measurements and display geometry keep their separate meanings.

## core/data.py::run_epoch_contract

<!-- implementation-sha256: 681661e28884d40fc72ef0469a24638e55c4c2b9cde7677bbdc2c2908d114c82 -->

Require an exact nonnegative snapshot storage index, read any requested redshift schedule from manifest.json and convert that entry to a requested scale factor. Return epoch_contract for the supplied actual scale factor, using the caller’s clock or the run’s proper-time clock, plus snapshot_storage_index. The requested epoch is retained separately from the actual saved epoch; the storage index is not a physical time. This helper writes no files.

## core/data.py::assert_ic_publication_complete

<!-- implementation-sha256: d10b1844e766a290e6198e5c99b9ca24df7237083894030d9364366a90e355c1 -->

Inspect every .ic-transform-*.json journal beside the supplied IC paths. Raise unless every encountered journal has schema ic-transform-v1 and status COMPLETE; normal completion returns None. This prevents use during an unfinished recorded transform but does not checksum all transformed payloads; replay_ic_transform verifies the transaction and its archived/current bytes.

## core/data.py::seal_dataset_semantics

<!-- implementation-sha256: 2a8e22811233d346060ee8e252adbc2eb586c644538f15481d76ffd70b189ae9 -->

Hash the declared scientific-contract attributes of an open HDF5 dataset and store the digest in its semantics_sha256 attribute, returning None. The seal includes units, shape/dtype declarations, quantity, epoch, frame, provenance and identity metadata covered by _SEMANTIC_ATTRIBUTES. It does not checksum the numeric payload or unrelated producer/group attributes; payload/container publication checks provide those separate bindings.

## core/data.py::verify_native_hdf5_set

<!-- implementation-sha256: 0087f929861d9ac79b606ad3c95cb784c6ee9e99811afa0b76a0f0f51705242e -->

Inspect an explicit nonempty unique HDF5 shard list for a common publication base, finite common native time, consistent generation/shard count and declared shard indices. If a publication marker exists, require its COMPLETE schema/generation/count and unchanged bytes; require_publication rejects a missing marker. Check file identity/size/mtime stability during inspection and return COMPLETE, COPIED_GENERATION_UNSEALED or LEGACY_UNSEALED with generation/count/time and science acceptance NOT_ACCEPTED. This structural publication check does not itself read/checksum all particle payloads or infer whether native time is scale factor.

## core/data.py::verify_native_checkpoints

<!-- implementation-sha256: 05716da7cf846be41827da91f0b7d9a91373ca92fe300a1db90854a3d83cfde5 -->

Require exactly one nonempty canonical restart.N file per original MPI rank. If no checksum receipts exist, return LEGACY_UNVERIFIED unless require_receipts demands them; partial receipts fail. Otherwise verify every rank’s CRC32, stable payload/receipt, native state layout, epoch metadata and common generation/clock/layout, then return VERIFIED_NATIVE_CHECKSUM with file hashes and clock fields. CRC32 detects accidental corruption, not authenticity. Isolated native L/V time is not cosmological scale factor, and proper cosmic time needs its separately bound run clock.

## core/data.py::publish_hdf5_product

<!-- implementation-sha256: 46c24e4e9a2d03181db2301a35d5f6a73349bc57b915db7bf5975648da04ef43 -->

Optionally attach semantic specifications for every numeric dataset in a staged HDF5 file, then require its dataset contracts/payloads to verify. Archive the staged new generation and any previous public bytes under .hdf5_product_history with a hashed publication manifest. Publish a PENDING pointer, atomically copy the retained generation to the public destination and change the pointer to COMPLETE. Return the publication record with science acceptance NOT_ACCEPTED; failures preserve available staging evidence. This operation can replace the public file while retaining its previous generation; it is not a physical calculation.

## core/data.py::verify_hdf5_product

<!-- implementation-sha256: b415540973ef2c25eb91dab5e11a332affdcea2efb9bc3532de118a743f306b0 -->

Verify a published HDF5 product under the declared read-memory bound. Without a publication pointer, check available dataset payload contracts and explicitly return LEGACY_UNSEALED. With a pointer, require COMPLETE, a generation contained in .hdf5_product_history, the original publication-manifest hash, matching product identity and matching public/retained whole-file hashes, then verify strict dataset payload contracts and return the publication record. Pending, redirected or changed products raise; a successful integrity check does not grant scientific acceptance.

## core/data.py::replay_catalogue_publication

<!-- implementation-sha256: b1e82185bef461a0cf470e57eaf58e31e42dcd5b0e0ea19703e661f7df24b24d -->

Load the recorded catalogue transaction and immutable plan/archive for an owned nonsymlink data directory. Validate the whole current destination set against permitted previous/next hashes before changing anything. If already COMPLETE, require the exact next generation and return its record; otherwise copy missing next-generation members, remove only planned obsolete members, verify final hashes and atomically write COMPLETE journal/archive receipts. Return the completed record with science acceptance NOT_ACCEPTED. Replay commits recorded bytes without rerunning a finder; externally changed destinations are rejected.

## core/matched.py::require_matched_run_family

<!-- implementation-sha256: e34b7c52377beb331ddf002f575c499d861d925489aadd534d1135299e9f8487 -->

Require unique run IDs and locate the declared reference. Across all runs enforce identical Lagrangian namespace, primordial phase hash, comoving no-h box and resolution signature, with masses in Msun_no_h and lengths in comoving_Mpc_no_h. Require actual changes in background, linear-transfer and interaction signatures to equal the declared contrast axes, plus source provenance. Return the reference record; this metadata check does not itself demonstrate statistical independence or checksum all source files.

## core/matched.py::compare_common_observables

<!-- implementation-sha256: 27f6c119fc41fe6db741194b5122e89cbd802ae6f3b6e26894e0bd7f899cf638 -->

For each declared observable, retain only matched objects passing both reliability masks, exact estimator/unit definitions and all support bins. Form right-minus-left vectors; combine measurement covariance as C_left+C_right−C_cross−C_cross.T when paired cross covariance is supplied, requiring positive semidefiniteness. Return report/arrays containing individual and mean deltas, object-sampling covariance divided by match count, propagated mean-measurement covariance and the supplied matching covariance. A single object leaves sampling covariance unknown. Reject incompatible pairs explicitly; this within-realization scatter is not independent-realization covariance.

## core/matched.py::ensemble_matched_covariance

<!-- implementation-sha256: c2ce3a9d8d98f69dc56c67307ea17c42bcefdda72c2fbf9d48ef60b3645bfa56 -->

Require at least the declared minimum and two uniquely labeled realization products. Compute the finite mean_difference array’s sample covariance, its ensemble mean and covariance divided by realization count for the mean. Return those arrays and count. Units follow the supplied differences; unique IDs prevent duplicate labels but do not independently prove that the underlying simulations are statistically independent.

## core/matched.py::cic_power_spectrum

<!-- implementation-sha256: aed02f2e324dda0a2b53bdb84745b5a8b98e227140bc3743fa97977e57329651 -->

Deposit positive Msun masses periodically on a bounded comoving-Mpc no-h CIC mesh, normalize its overdensity and divide the real FFT by mesh_size cubed. Correct Fourier amplitudes by the CIC window, multiply squared amplitudes by box volume, and bin modes below the declared Nyquist fraction with real-FFT multiplicities. Return raw and optionally mass-weighted-Poisson-subtracted P(k), mode counts/support and edges, plus estimator metadata. k is inverse comoving Mpc and P is comoving Mpc cubed, with no h. Negative subtracted estimates remain; mode support and this window correction do not establish aliasing/discreteness convergence.

## core/matched.py::contrast_population_functions

<!-- implementation-sha256: 40ea215276ddb1ddc3ff0935d7033a8e306d108044a3be182b7a9e697db0f753 -->

Require identical population bin edges. Return right/left density ratios only where both populations support the bin and the left density is positive, plus right-minus-left differences. Propagate difference covariance from the two diagonal shot variances minus the supplied paired covariance and its transpose; reject eigenvalues below the declared tolerance. Difference units are count per comoving Mpc cubed per dex and covariance has squared units; the returned shared support must be honored.

## core/matched.py::execute_matched_stage

<!-- implementation-sha256: f6ea9366db1ae1dda0b1092c7511dc09be29ca4e2a7e4c27c6c74f17348c3ed8 -->

Dispatch a declared research stage to observable-convergence comparison, native/explicit-particle CIC power, supported population functions/contrasts, Lagrangian-matched cosmologies, realization covariance, stream thermometry or discovery controls. Verify relevant retained source hashes, native units and estimator definitions before computing; matched cosmologies retain WDM reliability screens and exact patch assignments. Return report, numeric arrays, a matched:stage_id namespace and None for the optional final field. The dispatcher produces measured products with explicit acceptance status; it does not turn one realization into an independent ensemble or launch missing native runs.

## core/matched.py::compile_matched_hydro_pilot

<!-- implementation-sha256: ceb1e4a77d869556af03eb3a0885aa1ac4e81fa88a084ab93ff8272d5238bbcd -->

Compile a bounded immutable workflow from complete hydro answers, pinned shared IC shards and a phase/noise asset. Require positive exact seed/resolution/resources, common saved history indices and a retained comparison snapshot. Variants may change only declared BH, stellar-evolution, chemistry or hydro-model sections, must actually differ, and must retain all common numerical settings. Write per-variant answers and native gadget stages, optional supported galaxy-measurement stages and a final matched-hydro measurement stage. Return the hashed design/workflow paths with NOT_EXECUTED and NOT_ESTABLISHED statuses. Compilation writes recipes; native evolution, source-budget closure and convergence remain separate execution requirements.

## core/matched.py::matched_hydro_measurement_runner

<!-- implementation-sha256: 59d922e91a20f099f48ee778df49fa135eb1a13a3274f7d80034381d31448788 -->

Require the matched-hydro destination to lie within the workflow generation and its requested memory to fit the stage allocation. Invoke measure_hydro_comparison on the retained request and return its publication result. This adapter measures completed native histories rather than evolving the physical variants.

## core/matched.py::matched_hydro_pilot_main

<!-- implementation-sha256: d113a547b1f6cabedb126346e359be39c488fa20465f9de2ad36323030ed0a29 -->

Read --request JSON after a 16 MiB metadata bound and compile the matched hydro pilot into required --output, resolving pinned request assets relative to the request file’s parent. Print the design record and return None on normal completion. This CLI prepares the workflow without executing its simulations.

## core/observation_geometry.py::read_native_gravitating_particles

<!-- implementation-sha256: 6a0404c615b29626cce195b3b85b0eb7ab7f4894b43ed6cde99bd0fc8e60daf3 -->

Read the complete explicitly selected native gravitating species from a sealed snapshot shard set, within particle/memory bounds. Require common shard epoch/unit parameters, exact unique uint64 IDs and finite positive retained mass, excluding only zero-ID/zero-mass BH tombstones. Apply dataset a/h/cgs factors to obtain proper kpc, peculiar km/s and Msun; fixed cosmological MassTable values include 1/h, while isolated code masses use UnitMass directly. Rehash original shards/publication markers and return metadata plus aligned IDs/species/phase-space/mass arrays, adding no-h comoving Mpc positions only for comoving runs. Epoch metadata distinguishes scale factor from isolated native elapsed time. Gravitating BH composite mass is retained; no luminosity is assigned.

## core/observation_geometry.py::read_gadget_lightcone

<!-- implementation-sha256: be70601e1822671ca9f7fb4d2c3b35f1e4c83f097b529bc46cfddb4a06c45288 -->

Read one cosmological native GADGET light-cone file with exact IDs and per-particle Ascale, bounded by maximum_particles. Convert signed observer-relative positions, a-dependent velocities and dataset/fixed-table masses to proper kpc, peculiar km/s and Msun. Use each emission scale to derive no-h comoving positions/radius, angular-diameter distance a*r, luminosity distance a_observer squared*r/a and cosmological redshift a_observer/a−1. Retain the recorded observer origin, rehash the file and return metadata/arrays. Repeated native IDs in different periodic emission replicas are permitted; the reader neither deduplicates them into independent sources nor applies an extra origin subtraction to already observer-relative coordinates.

## core/observation_geometry.py::background_lightcone

<!-- implementation-sha256: bc8ef226fd09cebc5d4d46df614b160881b57baea9a79bb31a36fde1cb3cd8f3 -->

Require increasing physical Myr/background scale-factor samples, source hashes, observer time inside the clock, bounded sample intervals and a passing independently supplied distance-interpolation certificate within the requested kpc error budget. Integrate c*dt/a by cumulative trapezoids, subtract from the observer’s integral and retain past-cone knots through the observer. Return scale factors, no-h comoving-kpc distances, source pins, interpolation bound and curvature metadata. The represented distance model is flat FLRW; the intersection stage separately rejects nonzero curvature. This helper does not generate or independently certify the input clock.

## core/observation_geometry.py::emission_intersections

<!-- implementation-sha256: 55744bee2b23fb0dedb49cfc4a0b8db0dc5964141d56abfb3d3b5ffb67d4daf3 -->

For flat FLRW, intersect exact recorded periodic-image particle tracks with the measured past light cone. Require no-h comoving-kpc/peculiar-km/s/Myr units, aligned scale-factor clocks, integer image offsets and declared interval support/error bounds. Build cubic-Hermite positions with dx_comoving/dt=v_peculiar/a, split at background knots and solve the squared-distance polynomial for cone crossings in each requested replica and field of view. Retain unsupported intervals and deduplicate boundary events by namespace/identity/replica/time. Return events/arrays with emission epoch, comoving/physical distances, interpolated mass and FLRW plus special-relativistic Doppler redshifts; gravitational shifts are omitted. Replicas are observations of the same volume, not independent realizations. Interpolated crossings retain the supplied trajectory/background error scope.

## core/observation_geometry.py::trace_multiple_planes

<!-- implementation-sha256: d48e9d285337505f5fc28b95cd21e5d0b94d8d63b693bc5a3a6ed45ae8672f2c -->

Require N-by-2 angular rays in radians and measured lens planes strictly ordered in positive comoving distance before a source, using flat-FLRW distance ratios. At each plane subtract preceding physical deflections, bilinearly interpolate that plane’s reduced angular deflection and divide out its source-distance ratio. Map rays to source beta using the observer-to-source ratios, accumulating a support mask for finite in-plane interpolation. Return the original rays, per-plane positions, source positions and support with a report. Out-of-grid rays remain unsupported/NaN; this geometric thin-lens mapping does not calculate strong-field BH appearance or time delays.

## core/sidm.py::validate_sidm_history_request

<!-- implementation-sha256: 27ca7902cda1056afe2b0b07452e2593e7db1bcdeaf96de8349dde3589f19a4f -->

Require a versioned bounded halo-history request with explicit particle/epoch/halo/shell/central support limits and memory. Validate unique saved snapshot labels, increasing measured elapsed Gyr, sealed selections, exact phase/background/linear/baryonic matching hashes and a namespaced uint64 halo identity. Permit collisionless or elastic_sidm, requiring native scattering diagnostics for SIDM. Require species/component mappings to label only fine type-1 DM as dm; coarse type-2 remains a separate gravity/support component. Return the request without reading/evolving particles. Import through core.sidm_analysis, since the physical consolidated sidm.py rejects direct import.

## core/sidm.py::validate_sidm_membership_request

<!-- implementation-sha256: 624dc97f69abbeeba83198a434be7941a96659464ca2927b3b8976578fe406aa -->

Validate the native host-membership request schema, exact nonnegative snapshot/host row, positive particle ceiling, GiB workspace and proper-kpc support aperture. Require matched-halo metadata and particle namespace. Return the request; actual native-stage ownership, complete membership and physical measurements are checked by the publisher.

## core/sidm.py::sidm_input_bindings

<!-- implementation-sha256: 48e9a88d6156cd2428590bf90c15f5e15432a7e8ba43e3395f62787c9abd8b69 -->

Dispatch the typed SIDM science request to its validator, bind this implementation and relevant sources, and include completed native-stage ownership for membership/satellite/history/forward-vector requests. Load the exact sealed products needed by matched histories, observation contrasts, derived products, response samples or independent calibration. Merge and return source hashes under the declared memory bound; unknown schemas or conflicting/changed products raise. This provenance intake does not execute the requested estimator or native dynamics.

## core/sidm.py::validate_sidm_satellite_request

<!-- implementation-sha256: e5bbcbddfc04197106bff068e4150cbad57b019b8f915c880f6b23d374f02deb -->

Require the native satellite request schema, exact snapshot/subhalo row, explicit particle/pair/chunk limits, positive workspace and proper-kpc potential softening. Require a boolean native-diagnostic policy, satellite/particle identities and the supported cohort policy. The implemented potential gauge must be instantaneous_satellite_escape_zero; a tidal-saddle escape model needs its own measurement. Return the request without deciding which particles remain bound.

## core/sidm.py::validate_sidm_matched_request

<!-- implementation-sha256: 0f4c539930aa09d1bc2d639728f76a1f0f8ffb1730896d403ba488cad26b0c66 -->

Validate a bounded family of at least two distinctly labeled SIDM/collisionless histories with an explicit baseline. Require unique supported observable names, a positive dimensional absolute floor for each, and minimum exact-DM overlap count/fraction with fraction at most one. Return the request; physical estimator/epoch/identity and collisionless-baseline checks follow when products are loaded.

## core/sidm.py::publish_sidm_matched

<!-- implementation-sha256: 4fb68314ee07697db36dfc98918d413d029ba646d74d090132032f403ddebc91 -->

Read completed sealed halo histories within the combined memory bound and require a collisionless baseline. For each comparison require identical primordial/background/halo matching, physical estimator/aperture definitions, particle namespace, actual Gyr epochs and proper-kpc radial bins. Check first-saved exact fine-DM ID overlap, intersect each observable’s appropriate support masks and publish difference plus difference/max(abs(baseline), dimensional_floor), leaving unsupported values NaN. Flag changed baryonic controls as combined contrasts rather than interaction-only results. Return a numeric audit with units, selected-pair overlap and exclusions; covariance across independent IC/operator realizations and unique cross-section inference remain separate.

## core/sidm.py::validate_sidm_observations_request

<!-- implementation-sha256: 399f21eb795b82f2c57a316a31974ee8293ee8ba5a160f34f3422df88cdff48a -->

Require a bounded nonempty distinctly identified list of stream, lensing or H0-estimator comparisons, explicit matching controls and nonempty observable selectors. Each observable needs units and a positive dimensional floor; workspace must be positive. Return the request. Input producers must later demonstrate the matching declaration and array units; consumer labels alone cannot create matched observations.

## core/sidm.py::publish_sidm_observations

<!-- implementation-sha256: 1c1f5750f757b51919baf34c48e9e90e6ad7df4abbfdbf93d99010ba41b9eaf1 -->

Read sealed collisionless/SIDM observational products and require both producers’ matched_experiment declarations to equal the requested geometry/selection/nuisance controls. Require aligned array shapes, exact source units and finite common-supported values. Publish supported right-minus-left and floor-scaled relative contrasts with original source bindings and return a numeric audit. The H0 scope is an estimator response at fixed background: scattering does not replace the expansion history, and baryonic/native/estimator biases require separate controls.

## core/sidm.py::validate_sidm_derived_request

<!-- implementation-sha256: b81f042ca7d19ca8bc83b2a0a9d187546aefe5579cafdf21a846d16c55351f38 -->

Validate the explicit derived product type: satellite, degeneracy, stream_template or lens_plane. Require positive GiB workspace, one to 64 retained inputs and nonempty physical settings. Return the request; the typed producer must subsequently verify the actual scientific inputs rather than synthesizing them from this declaration.

## core/sidm.py::validate_forward_vector_request

<!-- implementation-sha256: d965fabd917b0ee2febf3cb36f20fb6696b5bc1ccea62ef9bb33941a5c9bbd4d -->

Require a native forward-observable-vector schema, bounded distinctly named observable selectors, a nonempty native parameter-binding map and an exact key match between parameter values and bindings. Require a named realization and positive workspace. Return the request without measuring observables or proving independence from its label; the producer binds the original native configuration, realization and saved products.

## core/sidm.py::validate_joint_response_request

<!-- implementation-sha256: 2112696d4914cfedcb2d115316b44e02df0bb90160ba5bd4cadab4af49e28160 -->

Validate training-v2 or calibrated-v1 native joint-response requests with distinct parameter names, finite baseline/scales, positive scales and at least two strictly decreasing positive derivative steps per parameter. Bound measured samples and require at least two realizations. Training forbids pre-attached calibration receipts; calibrated publication requires independent calibration receipts. Return the request. Finite-difference stability and actual independence are evaluated from the retained native products, not inferred from request validity.

## core/sidm.py::validate_joint_calibration_request

<!-- implementation-sha256: 9bb38bf51d729a0aa22d43749131f0f917712c5ae98bd60b70e50fe1bc0c663d -->

Require a versioned held-out response-calibration request, positive workspace, nonempty bounded baseline/varied pairs and at least two independent realizations per point. Validate nonnegative paired-mean sampling allowance, positive whitened residual tolerance and a design singular-ratio threshold strictly below one. Return the request; actual seed disjointness, two-sided parameter coverage and measured prediction residuals are checked by measure_joint_response_calibration.

## core/sidm.py::publish_joint_response_calibration

<!-- implementation-sha256: 9a10274401c741151d1f11c18565a2fd82a967a09361c188ebf4f339e91b6051 -->

Measure joint-response calibration from original disjoint native training/held-out products and publish the returned report, residual/support/design arrays and source hashes as a numeric audit. Residuals retain per-observable physical units, offsets retain declared parameter units and support/design/norm arrays are dimensionless. Return the audit; its local supported domain does not imply global posterior coverage.

## core/sidm.py::validate_calibrated_response_request

<!-- implementation-sha256: 6ab759c0a77892e2c8e769d1445a38b847bb9c0e568014d4646ba0f5f803302b -->

Require the typed calibrated-response schema, positive memory and a bounded nonempty list of independent measured calibration products. Return the request. This shallow structural validation does not substitute for replaying each calibration’s retained measurements or matching its training-response fingerprint.

## core/sidm.py::publish_calibrated_joint_response

<!-- implementation-sha256: 8856a9945513e656cdee6a9e2275ea295c2309482497d5f0e913555394a9a2c9 -->

Load an actual native training-response estimate, replay/verify each measured independent calibration and require the same response fingerprint, original training-product hash and observable domain. Intersect training support with every verified calibration mask. Publish the original response arrays with the resulting domain, calibration receipt pins and SUPPORTED_LOCAL_RESPONSE or UNRESOLVED status, returning a numeric audit. The scope is step-stable local derivatives checked at measured held-out points; no global posterior or unsampled-domain coverage is inferred.

## core/sidm.py::joint_response_campaign_main

<!-- implementation-sha256: c9db6405cbdf782863a1b37855654483b69035804c25dc6dd53df8118d4772e1 -->

Read the positional request JSON and compile a complete native joint-response/held-out workflow into --output using the request parent as asset base. Only --execute passes the generated workflow to execute_workflow under output/execution. Return zero on completion. Preparing the design alone launches no native runs; actual execution and calibration receipts determine scientific support.

## core/sidm.py::joint_calibration_runner

<!-- implementation-sha256: 8d0fdaf899f0a6f22d44ec611d1242847a1e8c89d5d3ab0b65a7d5b58d77571d -->

Publish measured held-out native response calibration through the shared SIDM workflow adapter. Enforce request memory within allocation and destination inside the generation, write a sibling .ref.json with the published product hash and return the audit. Calibration support remains conditional on its retained held-out measurements.

## core/sidm.py::calibrated_response_runner

<!-- implementation-sha256: c8db3efffb0bf1a03ed9d9c38f0b38d5a94948df7d9ad3c34d85856727a65e6d -->

Use the shared owned-generation/workspace adapter to publish the training response restricted by verified independent calibrations. Write the product’s sibling .ref.json hash reference and return its audit. A completed stage can still retain an UNRESOLVED calibrated response domain.

## core/sidm.py::forward_vector_runner

<!-- implementation-sha256: baf10d95afd1c18899cee5551aac7ff1823e6ddc116d97a01668e87c80de9d9e -->

Publish the actual native forward-observable vector through the SIDM generation/memory guard, write its sibling .ref.json with SHA-256 and return the audit. This stage reads bound native observables and parameter settings; it does not evolve a new parameter sample.

## core/sidm.py::joint_response_runner

<!-- implementation-sha256: cbdbb727e07ef307daba3850cec54a7bde41414e8c758acd05d03ba7297ea95e -->

Run the native joint-response estimator through the shared memory/output-ownership adapter, write a sibling .ref.json and return the publication. The request determines training or calibrated mode; stage completion does not bypass derivative stability or independent-calibration requirements.

## core/sidm.py::sidm_history_runner

<!-- implementation-sha256: 521113405ab8388b278ef5e4ba4b971f456f3936e08ddf373ed6b845ba3df037 -->

Measure and publish the bound SIDM halo history through the shared workflow memory and generation-containment guard. Write a sibling .ref.json with the product SHA-256 and return the audit. This stage reduces existing native histories and retains their support/diagnostic scope.

## core/sidm.py::sidm_membership_runner

<!-- implementation-sha256: b82cb6d5634c419455a87ce5c2519325f9614bc616ce3e70948f5d18f663ec39 -->

Publish the actual native halo membership product through the shared SIDM workspace/output-ownership guard. Write a sibling .ref.json and return the audit. Membership and aperture support come from the producer’s native selection, not this adapter or a configured SIDM switch.

## core/sidm.py::sidm_satellite_runner

<!-- implementation-sha256: b1b17363ffb87e3dfb7eec0a8156ed583bfb577fc0fefdc470ec1d8ec83909e9 -->

Publish native satellite binding/survival measurements through the common SIDM workflow guard, requiring workspace within allocation and output inside the generation. Write the sibling .ref.json hash reference and return the audit; the producer’s exact cohort and escape-potential definitions remain authoritative.

## core/sidm.py::sidm_matched_runner

<!-- implementation-sha256: 0663fa068a90a1465a207808cd7952e41fce0058fc5d75c57e3268cba12347f5 -->

Publish matched native halo-history contrasts through the shared workflow guard, write a sibling .ref.json and return the audit. This adapter preserves the publisher’s common epochs, ID-overlap support and matched/unmatched baryonic interpretation; it does not supply independent ensemble covariance.

## core/sidm.py::sidm_observations_runner

<!-- implementation-sha256: dff3d0d2fdca772a6a689d3bfcf2b448f0655790c975d66295f5cb50acc3b2a6 -->

Publish measured matched stream/lens/H0-estimator contrasts under the shared SIDM memory and generation-ownership limits. Write a sibling .ref.json and return the audit. The original producers must declare matching observations and units; the adapter does not construct a missing physical comparison.

## core/sidm.py::sidm_derived_runner

<!-- implementation-sha256: 3c2e4c04b45379dca33ba0a145f086f6b775b5a0795cef55b60a1a9bc16f6d23 -->

Invoke the requested supported derived SIDM science publisher under shared memory/output ownership limits, write the sibling .ref.json hash reference and return the audit. Product-specific physical settings and input support stay with the underlying satellite/degeneracy/template/lens producer.

## core/sidm.py::register_sidm_stages

<!-- implementation-sha256: 025af8155f18d3e30967a50d6f4468de32cc05e7a9d86c04b3977ef91330647f -->

Register ten SIDM stage definitions for membership, satellite, history, matched/observational/derived products, forward vectors, response training, held-out calibration and calibrated response. Bind their implementation-source paths and schema validators; all require request, request_base and destination, and native consumers additionally require native receipt/run directory. Return None. Registration makes workflow routes available without executing them.

## core/sidm.py::sidm_science_main

<!-- implementation-sha256: 9a2b16e1730a792e42c7ff00b1a2d552f27e8ddac53f6d740476c37b15efcb55 -->

Select one of ten science modes from a positional mode/request JSON and required --output, resolving request assets relative to that file. Membership, satellite, history and forward_vector also require both native-stage-receipt and native-run-directory. Call the selected publisher, write a sibling .ref.json with its SHA-256 and return zero. Import/run through the documented core.sidm_analysis logical namespace or public commands; successful publication retains each product’s support and acceptance status.

## core/sidm.py::validate_isolated_history_request

<!-- implementation-sha256: d4b56226892ca61d67c08f9ebc2cec5490387a3648d1bcaef5b0a3458f70917d -->

Require the versioned isolated-halo history schema, explicit positive particle/snapshot/chunk/core/persistence limits and at least three distinct saved snapshots with increasing requested native times. Validate GiB memory, proper-kpc core aperture, collapse density ratio above one, epoch/drift tolerances and dimensional energy/angular floors. Require isolated nonperiodic zero-at-infinity potential, a fixed-inertial or measured-COM frame and either diagnostic-only or predeclared-bound angular policy. Return the request. These assumptions exclude cosmological, hydro, external-force and periodic energy balances.

## core/sidm.py::isolated_history_input_bindings

<!-- implementation-sha256: 3ab0f4d4781f144d92729f75f2e91ff154f1e3ff2519a1df847d11bbe7d90696 -->

Verify the completed native evolution receipt and hash-bind its original source/input/delivered files. Require every selected original snapshot and publication marker to belong to that exact generation. Verify the owned native Config.sh enables self-gravity plus evaluated/saved potential and excludes periodic/external-gravity flags; verify any run manifest is owned too. Return the source map. This is provenance/build support for the isolated half-sum potential-energy estimator, not a conservation result.

## core/sidm.py::measure_isolated_control_history

<!-- implementation-sha256: 9ad1547ebdfe7f9b9fb7457bb75800dae44c5ff304e800dc9daee79db9890278 -->

Measure actual saved epochs of a completed isolated DM-only closed system using full fine/coarse-DM particle census and declared core aperture/frame. Require requested native times, unchanged exact IDs and common time units. Reduce E=sum(m*v²)/2+sum(m*phi)/2 in the fixed native inertial frame; angular momentum uses the declared origin/velocity frame. Masses are Msun, energies Msun*(km/s)², angular momentum Msun*proper-kpc*km/s and elapsed time Gyr. Compute core mass/volume and count/effective-count support, then maximum mass/energy/angular drift relative to initial values with declared dimensional floors. Return per-epoch data, predeclared control gates and a persistent saved-crossing collapse bracket after rehashing sources. CONTROL_ONLY and deterministic saved-history uncertainty remain explicit; no unsaved collapse time, cosmological energy gate or independent numerical accuracy is inferred. Fixed MassTable and per-particle Masses use the same isolated unit convention without h scaling. Import via core.sidm_control_measurements.

## core/sidm.py::isolated_history_runner

<!-- implementation-sha256: 180f8a4a43d7c228b11ff0aab330a8a0a9497d82f0578892a27abde12690b1f2 -->

Require requested isolated-history memory within the stage allocation, measure the retained native history and atomically write isolated-control-history.json plus its SHA-256 seal in the generation. Return maximum drift measurements and findings for failed predeclared gates. Stage execution can complete with failed control gates; the adapter does not rerun or repair the native trajectories.

## core/sidm.py::isolated_history_main

<!-- implementation-sha256: b8d317ec059f9561ca3058ef7fd34213b76ea64e1bdd9b8135487543deba7feb -->

Read required --parameters JSON and measure the isolated native conservation/collapse history. Require a fresh --output, then atomically write the record and sibling .sha256 seal. Normal completion returns None; invalid inputs raise. A written record retains PASS/FAIL control gates separately from execution completion, and the CLI does not evolve a halo.

## core/sidm.py::validate_control_clock_mapping_request

<!-- implementation-sha256: b949853d4f7a0074cca84b234117978d52161653200d6b56a4c4827a805d2e8a -->

Require the exact canonical clock-mapping schema and complete key set, bounded increasing nonnegative canonical-speed nodes in km/s and directed cosine nodes spanning [-1,1]. Validate epoch/speed/angle/quadrature-work ceilings, positive memory/time tolerances and fractional clock tolerance below one. Require exact path/SHA references for history and physical law, with an optional native differential table, allowing deferred stage references during planning. Bound the epoch-speed-angle workspace and return the request. This declaration does not demonstrate equivalence between expanding and static transport. Import via core.sidm_control_measurements.

## core/sidm.py::control_clock_mapping_input_bindings

<!-- implementation-sha256: ab1bd90f926418477f9f0c0ccefe4118cc1ed1f28f346d0fd67c486d3cc29c32 -->

Validate the canonical mapping request, read its sealed native history under one-third of the memory allocation and retain its source pins. Verify any physical-model/differential-table references against their original hashes and metadata workspace bounds, then return the merged bindings. This reader does not construct a tau clock or evolve mapped particles.

## core/sidm.py::validate_control_native_clock_request

<!-- implementation-sha256: 6aa44da2bfaca91d9030006c62dee24b25c18a4503d3798e2ce9d0f24535236b -->

Require the exact native canonical-clock asset schema with a sealed mapping reference, finite positive memory and clock/inverse tolerances, fractional tolerances below one and explicit row/byte/panel/inverse-iteration limits. Bound ASCII and row workspace before returning the request. A valid asset request is not a verified inverse-clock execution.

## core/sidm.py::control_native_clock_input_bindings

<!-- implementation-sha256: 433e9a8442a9b97729d28098f0bc53f4e5b17dd31ca6364b19c803132eff1cfb -->

Validate native-clock parameters, load the sealed current canonical mapping and its original history under bounded memory, and require an expanding sidm-physical-control-history-v1 source with request-base lineage. Return all source bindings. Older mappings without the recorded request base are rejected rather than interpreted relative to an arbitrary shell directory.

## core/sidm.py::publish_control_native_clock

<!-- implementation-sha256: 3397e59aa0a31aa0dcb700adbfc782886e9f73599e7e55b398082a71bc3fca3f -->

From a source-bound canonical mapping and its original expanding native history, require the exact population, engine and increasing saved scale-factor domain. Retain the actual native H0/background coefficients or verified native log-E/log-a table and physical-law binding. Write a fresh canonical-clock.txt forward/inverse asset plus clock.json, including model/background/source hashes and the native output schedule for tau=integral(dt/a²). Check quadrature plus schedule-rounding error against the frozen seconds/relative budget and retain the native time-unit conversion. Return the report with science acceptance NOT_ESTABLISHED. Asset publication does not execute mapped transport, certify inverse-clock accuracy downstream or establish distribution equivalence.

## core/sidm.py::control_native_clock_runner

<!-- implementation-sha256: 199e58e56687cedba938e67e75ab7ce2daedc9ef933f421e6c5c59440c49be22 -->

Require requested native canonical-clock memory within workflow allocation and destination inside the owned generation. Return publish_control_native_clock, which writes the original-background/law-bound ASCII asset and report. This stage prepares the native clock input without running the mapped collision operator.

## core/sidm.py::control_native_clock_main

<!-- implementation-sha256: 9060424fbbcbd809c1d2f9f3aabedd23a7782ba118baf03538389bd4f341fd96 -->

Read positional parameters JSON, default request_base to its file parent and publish the source-bound native canonical-clock asset into required --output. Return zero on completion. The CLI creates clock inputs and a report rather than executing or accepting a mapped native experiment.

## core/sidm.py::control_clock_mapping_runner

<!-- implementation-sha256: e6b0c272f80414cc77e9840bea4556f49fc869c6f249eec91e361bcb8c919662 -->

Enforce canonical mapping memory within the stage allocation and output containment in the workflow generation, then return publish_control_clock_mapping. The publication relates x=r/a, u=a*v_pec, tau and the time-dependent effective scattering law at measured nodes; the adapter does not turn that mapping into executed transport equivalence.

## core/sidm.py::control_clock_mapping_main

<!-- implementation-sha256: 3dde478089b992720a311bd221bc07ff5478bb87e173f0d99a91db95361839d4 -->

Read positional parameters JSON, supply its parent as request_base when absent and publish the expanding-control canonical clock/law/state mapping to required --output. Return zero after publication and propagate input/measurement errors. No native mapped driver is launched by this command.

## core/sidm.py::control_history_runner

<!-- implementation-sha256: 7fb1ee3621cb0d5d1b0dfacbd1f806cf50ba93f9345eef6895800000dc8b4f3e -->

Require the dedicated SIDM control-history workspace to fit the workflow allocation and destination to remain inside its generation. Return publish_control_history on the original completed native run. This adapter measures saved Gadget4/OpenGadget3 control outputs; it neither evolves them nor supplies missing native diagnostics.

## core/sidm.py::control_history_main

<!-- implementation-sha256: 09ab5b310726f7e8b8a6d38358e508886a2cf9941302f97e126c725f49a2bfe5 -->

Load positional parameters JSON, default request_base to the parameter file’s parent and publish measured dedicated SIDM history at required --output. Return zero on completion. The publisher reads the actual completed Gadget4/OpenGadget3 run and retains its physical clock/model/source contract; this CLI does not launch either engine.

## core/sidm.py::validate_control_comparison_request

<!-- implementation-sha256: cc7d44c4926bba3b2a75cffd90e49aca4e5e40a4d1524e2163427b7a1a0c32ac -->

Require the exact frozen physical-control comparison schema, distinct selected metrics with dimensional absolute/relative limits and positive reference floors, adequately sampled cohorts, memory and proper-time matching tolerance. Permit explicit independent-engine, same-engine refinement, rare/frequent overlap, physical/comoving balance or mapped-distribution scopes. Balance-only scope permits mass and canonical energy, not arbitrary distribution equivalence; mapped scope also requires inverse-bracket/scale-factor error budgets below one. Validate sealed left/right/law/optional-table references and return the request without evaluating any contrast.

## core/sidm.py::publish_control_comparison

<!-- implementation-sha256: 145693a563409a9f8d8a261c1954e2009043c30c04cdfc39431ce65eda798154 -->

Read two distinct completed native control histories and bind both to the same recorded physical population, identity namespace, cohort census and actual physical law. Enforce the declared engine/operator scope, common histogram bins and proper epochs with clock-error allowances; mapped comparisons additionally check original background, inverse brackets and scale factors. Build CDFs from complete cohort counts or compare selected moments/exact-ID saved particle metrics. Apply abs(delta)<=absolute_limit+relative_limit*max(abs(left),abs(right),floor); periodic particle positions use minimum-image differences/displacements. Balance-only comparisons must also bound each run’s own mass/canonical-energy drift, so equal bad drifts cannot pass. Publish differences, dimensional allowed errors and per-element decisions as a numeric audit, returning PASS_DECLARED_LIMITS or FAIL_DECLARED_LIMITS metadata. These single-realization saved-output limits do not establish iid confidence, independent-ensemble agreement or general pathwise/convergence acceptance.

## core/sidm.py::control_comparison_runner

<!-- implementation-sha256: 4db09deb41d627ec3386bf25c10b1aec966d69ef061e59f398fabc4b453a6bdc -->

Require comparison memory within the stage allocation and its output inside the owned workflow generation, then return publish_control_comparison. The published decision refers to the exact frozen scope/metric limits on original native histories; successful workflow execution can retain FAIL_DECLARED_LIMITS.

## core/sidm.py::control_comparison_main

<!-- implementation-sha256: 72ae4e195314ce20801061fba1a81030398efa680685820bd48c871bf783e877 -->

Read positional comparison parameters JSON, default request_base to the file parent and publish the actual retained control contrast at required --output. Return zero after publication; the product’s comparison_status remains the numerical decision. This command does not manufacture a reference trajectory or run missing refinements.

## core/sidm.py::validate_control_ensemble_request

<!-- implementation-sha256: e0a7ea53202fdd3741e06e2f3efc685980386ef9b3452ed169ac0ff5c7eb41f5 -->

Require the exact paired-native-control ensemble schema, at least two retained comparison references within declared minimum/maximum realization bounds, a positive observable ceiling and GiB workspace. Bound the full joint covariance and realization-vector allocation before returning the request. The structural count is not evidence of actual independent seeds; the ensemble producer checks the retained realization family.

## core/sidm.py::control_ensemble_input_bindings

<!-- implementation-sha256: efafb5f4d89e673eca0e5854825c67bf6a92c86009461f056d6315ef86a4294f -->

Validate the paired ensemble request, load every selected sealed comparison under one-quarter of its memory allocation and merge the original source hashes. Return those bindings before covariance production. Reading several references neither computes the ensemble nor makes duplicate/dependent native realizations independent.

## core/sidm.py::control_ensemble_runner

<!-- implementation-sha256: 1a609463c3c3cea3e899684e3eb2dc8270dce4317669bc061b85de649fdc27c7 -->

Enforce paired-ensemble workspace within workflow memory and generation-contained output, then return publish_control_ensemble. This stage measures the retained joint covariance/paired-realization differences; its producer’s independence, physical family and support requirements remain authoritative.

## core/sidm.py::control_ensemble_main

<!-- implementation-sha256: 4e596c8801ccf29b29bda369ee5f3b39d06eb45a17e7dce9af0e6f3365448ba6 -->

Read positional ensemble parameters JSON, default request_base to the file parent and publish measured joint covariance and paired-realization differences from retained native SIDM comparisons at --output. Return zero on publication. Missing or incompatible realizations raise rather than being replaced by invented samples.

## core/sidm.py::validate_control_refinement_measurement_request

<!-- implementation-sha256: fe3b9c8e69044da9b897ec88a3befae4cd7594777956bb484854dad5cc00f15a -->

Require the exact refinement-measurement schema and a supported operative native axis: timestep, kernel support, split angle, pair/particle probability, transfer depth or scale-factor fractional step. Require at least three strictly decreasing positive levels and one retained paired ensemble for each adjacent pair. Validate memory/observable/level limits, nonnegative mean-error multiplier and a positive sampling-error budget fraction below one. Return the request; empirical refinement support is measured later from the original ensembles.

## core/sidm.py::control_refinement_measurement_input_bindings

<!-- implementation-sha256: 6a4d36a7061f5afd4f26bf39416b69fe6e19151bbbff0edd9d8a466d60a5036b -->

Validate multi-level SIDM refinement parameters and read every adjacent-pair ensemble as a sealed numeric product within one-quarter of the allocated workspace. Return merged original input hashes. This intake does not infer a convergence order or extrapolate beyond measured levels.

## core/sidm.py::control_refinement_measurement_runner

<!-- implementation-sha256: 8c1ee21de5fefa09b8bf9141b5ca0d65e0a4b65c301187914b351e3babac538c -->

Require native multi-level refinement measurement memory within the workflow allocation and destination within its generation. Return publish_control_refinement_measurement on the actual paired ensembles. Preparing/finishing this stage does not bypass the producer’s sampling-error and measured-level support decisions.

## core/sidm.py::control_refinement_measurement_main

<!-- implementation-sha256: 64416828e037d5034fb5b6de5dbe91f00053f513fdb5da5496c9af83fe4bebce -->

Read positional refinement parameters JSON, set request_base to the file parent when absent and publish the measured native multi-level paired SIDM refinement product to --output. Return zero on publication. This adapter reduces existing ensembles; it does not execute the refinement simulations.

## core/sidm.py::control_expectation_runner

<!-- implementation-sha256: 246eb111a21cf175f9de9f2b036ae007696e1b3e7c76dfb1994d7efe907ac2e1 -->

Enforce physical-expectation workspace within the stage allocation and publication inside its owned generation, then return publish_control_expectation. The underlying producer measures beam expectations, thermal stationarity and saved relaxation from original native controls; the adapter supplies no new particles, collision events or reference distribution.

## core/sidm.py::control_expectation_main

<!-- implementation-sha256: ee778b5e9bcc1e5271a40a658d1d82121d55b95892f7c2ff12ce904340241209 -->

Load positional expectation parameters JSON, default request_base to the file parent and publish native beam/thermal/saved-relaxation measurements at required --output. Return zero on publication and retain the producer’s actual status/scope. This CLI reads completed controls; it does not interpolate an unsaved relaxation time or launch a replacement native run.

## core/sidm.py::control_history_input_bindings

<!-- implementation-sha256: a08ec0d6a29fa414730c34f866b161a17975bc822611a696a839fc9d615997dc -->

Read the actual native execution bindings and sealed physical/canonical population, requiring the correct schema. Match the evolved InitCondFile hash to both the population and invocation input, and require exact native units, box, starting time and physical/comoving mode. Mapped histories additionally require owned completed run metadata, the original physical/canonical state identities, event namespace, parameter/build/invocation paths and clock/law bindings. Attach measured invocation environment/cwd to the retained population and return request, parameters, flags, source hashes, population and arrays. This intake verifies provenance before history reduction; it does not rerun the control. Import through core.sidm_control_measurements.

## core/sidm.py::control_expectation_input_bindings

<!-- implementation-sha256: 7a735c6461d28195a5a15687aebe55ca6b7e234fb2bb43105ba63a9bec8f88ed -->

Return source pins from the full physical-expectation input reader. That reader validates the completed native history, original physical population/namespace, bounded particles/pair workspace and dimensional v2 elastic model. Where the native run used a differential table, retain its exact evolved interpolant rather than substituting a new analytic law. This function binds inputs without publishing or measuring a relaxation result.

## core/sidm.py::angular_control_input_bindings

<!-- implementation-sha256: c90e88cd0f2a1d9009dc2488df8bda5f638ea8e282dbd4621af2c300158cb916 -->

Bind the supplied native-event and reference-CDF JSON paths to their requested hashes and memory limits. Merge numeric-audit source bindings, reject conflicting hashes, and retain each report, sibling SHA seal and numeric-bundle file hash. Return the complete source map. These references support the angular measurement; reading them does not certify the empirical angular distribution.

## core/sidm.py::angular_control_runner

<!-- implementation-sha256: 81ed072126804157a94f024d5a22559c2cfcd0630a5ffac4faa81463c19f8b19 -->

Require native angular measurement memory within the workflow allocation and measure against the requested physical CDF at generation/sidm-angular-control.json. If the producer did not already publish numeric_data, atomically write its JSON and SHA seal. Return the empirical CDF error upper bound, native event count and explicit unmeasured-error exclusions. Completion retains the measurement’s actual support/limitations rather than establishing operator accuracy everywhere.

## core/sidm.py::angular_control_main

<!-- implementation-sha256: 0f88fd3d472453bea7e1fb059262bf5032b06285af9b6451df900b41dd8d93c2 -->

Read required --request JSON and measure saved native scattering angles against the physical reference at a fresh --output. If no numeric publication was produced, write the returned JSON and its SHA seal directly. Normal completion returns None; existing outputs and invalid measurements raise. Paths inside the request are not automatically rebased to the request file by this adapter.

## core/sidm.py::validate_angular_control_stage

<!-- implementation-sha256: b75c3b85b783101156ddd510fb75292b72f5731b17658cde4a607a66d20def3a -->

Validate v1/v2 native angular-control schemas, positive memory/speed, nonnegative CDF tolerance, bounded event work and exact event/reference path-hash objects. Require directed cosine bins spanning [-1,1] or folded identical-particle absolute-cosine bins spanning [0,1]. Per-event-speed mode requires v2, bounded conditional evaluations/support, full probability bins, nonnegative increasing speed bins, a tiny declared cosine-roundoff allowance and an angular cutoff leaving nonempty rare support. Bound v2 histogram/event workspace. Raise on invalid requests; normal completion returns None without measuring events.

## core/sidm.py::validate_cdf_request

<!-- implementation-sha256: b7d355154d1ee77aab03910ecb29adefcb33210aeb8ec1a6cb564e4ffaab06c0 -->

Require a versioned physical reference-CDF request with bounded initial/maximum nodes and refinement rounds, allocated memory, nonnegative fixed relative speed and optional speed-window width in km/s. Require a CDF tolerance above the retained floating-point allowance, explicit directed/folded cosine convention and a valid minimum angle, below pi/2 for folded support. Require the exact physical-model path/hash and return the request. This validates reference construction limits, not a native sampling outcome. Import through core.sidm_control_products.

## core/sidm.py::cdf_input_bindings

<!-- implementation-sha256: b2fb5eb3917a4578fe0fc660046661ed7e62892c9ad0d8586f8b02cb1a8dbd7e -->

Validate the physical reference-CDF request, resolve its model path from the process working directory, and require unchanged model bytes within the metadata workspace bound. Return model and consolidated SIDM implementation hashes. The function does not integrate a CDF or launch native software.

## core/sidm.py::publish_reference_cdf

<!-- implementation-sha256: 044dec710e9139c02ba67aa067fe30b010774bc34638a8fa04c57a216029054a -->

Construct a normalized fixed-speed physical angular CDF, delegating dimensional v2 elastic models to the differential-law publisher. Legacy supported models are isotropic, screened Rutherford and explicitly normalized nonrelativistic unpolarized spin-half Born Moller; Moller requires identical-particle folding. Retain the declared rare-angle cutoff and normalization probability. Adaptively refine cosine nodes until the linear-interpolation curvature bound plus floating-point allowance meets the requested tolerance, rejecting exhausted node/round budgets without relaxing it. Publish dimensionless nodes, probabilities and cell bounds, with any speed-window variation bound and source/model hashes. Return a REFERENCE_ONLY product; these analytic reference arrays are not native scattering samples.

## core/sidm.py::validate_events_request

<!-- implementation-sha256: e8b7c732528d5dcc414617924e2d7dc9917b00f3a39c5a603f7bf4f5a4892cc8 -->

Require a physical or mapped native-event export schema, nonempty actual event-file selection, positive bounded chunk/selected-event memory work and explicit speed window, dimensional energy/momentum floors and directed/folded angle convention. Validate the actual hybrid cutoff; mapped exports additionally require exact run-publication reference and an aggregate logged-event bound. Return the request without reading or fabricating any event. The requested speed window and cutoff define which logged rows may be measured.

## core/sidm.py::event_input_bindings

<!-- implementation-sha256: 0f72b2ea5441a4f8f7c41f22aab7b8df23589649825096de21e7e85dc3147c96 -->

Verify the completed native-stage receipt and require each unique event-log shard to be its exact owned delivered payload. Return receipt, original input/source and event-file pins. Mapped exports additionally verify completed run/clock ownership, model/event namespace, successful native invocation, original parameters and the gravity-free canonical build flags. This provenance reader rejects unrelated files or relabeled mapped frames before event reduction; it does not claim native logging was complete.

## core/sidm.py::export_native_events

<!-- implementation-sha256: 031b2d726a20cdeef1a4dd897d9cb643973a25444e8eab3281f74ed53fa2e345 -->

Read completed owned rare-elastic-pair HDF5 logs in bounded chunks, requiring one exact event namespace, physical angular model/particle convention and actual hybrid split across shards. Convert saved a/h/cgs metadata into physical peculiar km/s and Msun; mapped logs also validate original clock, frozen/substep source epochs and budgets, then convert canonical u to v_peculiar using each recorded scale factor. Select nonzero relative-velocity pairs in the declared speed window, retain exact uint64 event/two-particle IDs, and derive directed or folded scattering cosine from pre/post relative velocities. Publish speeds, native times, scale factors and pair energy/momentum relative residuals with dimensional floors; mapped products add tau endpoints in seconds. Reject empty selections, repeated event IDs, incompatible frames and changed sources. Return MEASURED_ONLY metadata and numeric arrays. This covers selected logged rare events, not continuous frequent diffusion updates, all microscopic collisions, completeness of logging or general rate/convergence acceptance.

## core/sidm.py::validate_canonical_population_request

<!-- implementation-sha256: f406330e760ed1def24a09fae3cfe19e2f832576f8ae99a162c8e1ff4f0a7d15 -->

Require the complete canonical-population request schema with exact original population/native-clock references, positive particle/asset/chunk bounds, chunk size within particle limit and explicit GiB memory/disk budgets. Bound original/mapped state, ASCII clock and HDF5 chunk work before returning the request. This declaration does not materialize mapped particles or establish equivalence to the expanding native run.

## core/sidm.py::canonical_population_input_bindings

<!-- implementation-sha256: 1077ab3eb50b5be79f8dc109ae53a51ff2c2f11605ec48ae31c71ae714f9697e -->

Load the completed sealed original physical population and current native-clock manifest under memory bounds. Require matching physical-state hash and event namespace; verify regular clock asset bytes, model/background identifiers, row census and actual H0/scale-factor fields against the manifest. Recheck original clock source hashes and reject conflicts with population lineage. Return request, population, arrays, clock and source pins. This intake binds the mapping’s origin without writing ICs or evolving a new representation.

## core/sidm.py::canonical_population_runner

<!-- implementation-sha256: d3a23ee93e119fdae6e08f579d6869db593857b933bfec56a239ca431a4ee07a -->

Require requested canonical-population memory/disk within the workflow allocation and destination inside the owned generation. Return publish_canonical_population, which materializes the exact original expanding particles as native canonical x,u ICs. The stage creates inputs rather than measuring completed transport equivalence.

## core/sidm.py::canonical_population_main

<!-- implementation-sha256: 6195cd5c3c6e4bdf0724287f6fbb6f241a89287ddf98793a5fe664e48584430a -->

Read positional canonical-population parameters JSON, default request_base to the file parent and materialize exact original expanding SIDM particles as native canonical x,u ICs at required --output. Return zero after publication. This command writes source-bound initial conditions and does not evolve them.

## core/sidm.py::reference_build_main

<!-- implementation-sha256: a9bdd8f1b1c668e416114df239ae41af43c43b9d91b8541133dce999486ff234 -->

Read positional parameters and require a fresh --output for the pinned independent OpenGadget3 reference build. Under a native run lock, set request_base/destination and a context carrying the declared jobs, OpenMP, memory and disk limits, then invoke reference_build_runner. Return zero after the build operation. This is an explicit compilation entry point; completed build/invocation/capability receipts, not this command name, establish which executable was produced.

## core/sidm.py::control_run_input_bindings

<!-- implementation-sha256: c28e8d98cc8c78c0991752f14ffcfe49fc6f6981dc2a0203c4ad1ba4e28448cf -->

Validate a physical/canonical native control run and load the original population, requiring particle limits, starting output epoch, physical box/softening compatibility and at least 512 MiB per rank after reserve. Hash-bind base configuration, elastic law, original IC and any independent build inputs. Canonical runs also preserve the source clock/model/background domain, physical-state namespace, unit/census lineage and bounded native clock asset. Independent-engine runs require a completed build capability, unchanged executable/configuration and successful source-bound compile invocation. Return request, population and original source pins without launching the native engine.

## core/sidm.py::mapped_campaign_plan_input_bindings

<!-- implementation-sha256: 70dc30b2e71a166e43e9f71a2b303da403d915f40471b9ca85de009e4452553f -->

Validate the mapped-campaign plan, load each retained population/reference history within allocated memory and bind its canonical run’s original base configuration and elastic model. Verify any shared native differential table hash and return the combined sources. This preparation reads the declared physical lineage; it neither compiles the follow-up campaign nor executes mapped runs.

## core/sidm.py::control_refinement_main

<!-- implementation-sha256: 1a96a87466c0e611495053579678ffa89a1b37dfab271626070576eaede78c33 -->

Read the positional request JSON and compile the independent three-or-more-level native SIDM timestep/kernel/hybrid/probability refinement workflow into --output, resolving assets relative to the request parent. Return zero after compilation. Native evolution and multi-level measured acceptance occur only when the resulting workflow is executed.

## core/sidm.py::control_campaign_main

<!-- implementation-sha256: afb9830e69a5d5907bb4230300c8a03a2ecd20ff2ae7de408da8368f63b866f2 -->

Read the positional request JSON and compile a physical or mapped-expanding native SIDM control/paired-realization workflow into --output, using the request parent as base. Return zero after design publication. The command prepares concrete recipes; it does not itself execute the native controls.

## core/sidm.py::control_population_runner

<!-- implementation-sha256: 643d78df6a348f9d6ea83330e9b8fa9fa5a582d2e90eca93a8c708ff882ae992 -->

Validate the physical SIDM control-population parameters, enforce memory/disk within the stage allocation and require publication inside its workflow generation. Return publish_control_population. This creates explicit authored physical populations and matched native ICs without evolving them.

## core/sidm.py::control_population_main

<!-- implementation-sha256: 1d181876f38479e4330adf23afe8af0ea3bb923ed76af6a0106a578daebb2a4a -->

Read positional population parameters JSON and publish physical SIDM control populations plus matched native ICs to required --output. Return zero after publication. This adapter does not insert a request_base default and does not launch evolution; explicit paths/settings must satisfy the population producer.

## core/sidm.py::reference_cdf_runner

<!-- implementation-sha256: 3f75aebf5734c559756017e7e288049abb113ab94f8468b2aac6d18175179280 -->

Require reference-CDF memory within the workflow allocation, publish the physical angular reference at generation/sidm-reference-cdf.json and return its maximum CDF absolute-error bound and node count. The reference product remains an analytic construction, not evidence that a native angular sampler passed.

## core/sidm.py::native_events_runner

<!-- implementation-sha256: 25bd16827f69198868233acc33cee5b963e5260f94997b3fa9f53c03a49bcbcf -->

Require native-event measurement memory within the workflow allocation, export the exact completed-stage event selection to generation/sidm-native-events.json and return event count plus maximum pair energy/momentum relative changes. The adapter reads actual logged pair states; changed-particle counts and frequent diffusion updates are not substituted for discrete events.

## core/sidm.py::sidm_control_products_main

<!-- implementation-sha256: 57ec225bfefb9a77adb6d51e57b69c58df304e502710de0c787ccf9f66d99a58 -->

Select reference-cdf or native-events from the positional kind, read --request JSON and require a fresh --output. A physical CDF reference rejects --native-stage-receipt, while actual event export requires that completed evolution receipt. Invoke the selected producer; normal completion returns None. This CLI keeps independently authored physical reference distributions separate from measured native event records.

## core/flagship_profiles.py::validate_profile_resources

<!-- implementation-sha256: 8b57889d88429b41b83ece9e4dee71454e88722791de1d6b17c9f3beb7280a40 -->

Validate positive finite memory and per-particle estimates, integer process/thread/job/epoch/checkpoint limits, and allocation against machine RAM and CPU slots. Reject MPI × OpenMP or build-job oversubscription. Return None on success; this checks declared resource settings without measuring runtime consumption.

## core/flagship_profiles.py::publish_production_requirements

<!-- implementation-sha256: a3caa89112c2d6c60275d2681be6801278b401d2d98daaab3192a92ebc2cb25c -->

Require a fresh destination and preflight production geometry in a temporary workspace. Retain a written planning receipt even when geometry exceeds the planning memory bound, then size generation, evolution and storage requirements with the requested headroom. Atomically publish and return the requirement report. These provisioning estimates do not allocate hardware or establish measured runtime capacity.

## core/flagship_profiles.py::re_safe_id

<!-- implementation-sha256: a471328fade185d81f7b0e2d31e44c07384bba76d4af815fcbf97226e1320143 -->

Return whether the identifier matches an alphanumeric first character followed only by alphanumeric characters, underscores or hyphens. This is a Boolean format check and does not create files or validate scientific settings.

## core/flagship_profiles.py::connected_launch_profile

<!-- implementation-sha256: 5449861ed1a39fb1d3e246c5f5c63a9a817f7ca78ba97aed8076a81d8df0f5d8 -->

Deep-copy a connected miniature/local-preparation request, merge resource overrides, resolve and hash-bind assets relative to the supplied base, and validate nested budgets, CLASS settings and parent/zoom/display contracts. Build the initial CLASS/parent workflow plus deferred zoom, history, measurement and display continuations; align answer-file process/thread/memory settings with the allocation. Return the launch-profile dictionary after initial DAG validation. This constructs recipes in memory; follow-up native preflight waits for the actual upstream products.

## core/flagship_portable.py::native_admission

<!-- implementation-sha256: 9e6a99ac10cd176a9548bf39bfe310e1f7a10be3963b361c240d43806bf20fe4 -->

Reuse a supplied pinned native admission report or create a fresh control directory and invoke the public native-music-transfer and native-music-mixed2lpt commands with the selected binaries. Save invocation/log files and enforce subprocess timeouts. Validate both native MUSIC and mixed-LPT proof, then return its path and SHA-256. Fresh admission actually executes controls; a reused report is checked rather than rerun.

## core/flagship_portable.py::materialize_portable_profile

<!-- implementation-sha256: 492dde81f8f3d639feb064cb3648bcf68a099ba9f8ee49113dad97f9e809bb14 -->

Validate a connected or locally seeded request, resolve its assets/resources and create a fresh workspace. Build the requested CLASS/MUSIC/monofonIC engines, obtain native admission, build the boundary adapter and require a passing frozen-noise replay. Prepare combined answer/flow templates and substitute pinned engine, admission and phase assets into connected-request.json. Return the connected request and path, retaining portable-preparation.json; failures retain failure.json. Preparation runs builds and controls, while the full miniature evolution remains NOT_EXECUTED and science NOT_ACCEPTED.

## core/flagship_production_products.py::connect_production_products

<!-- implementation-sha256: 357ad73920eaecb71126a4b80208f336b919b66364d7c96c9b51397447e3d294 -->

Read pinned production campaign/workflow inputs into a fresh product workspace and require their original allocation/storage gate. Preserve native-stage resource settings; optional postprocessing overrides may increase memory/disk budgets. Prepare history, measurement and display continuations against the original CLASS and per-case lineage, rejecting paired branches and incomplete required products. Compile the combined workflow, recheck source pins and write its manifest. Return prepared recipe references with native execution NOT_EXECUTED and science NOT_ACCEPTED; no evolution or plot rendering happens here.

## core/flagship_production_campaigns.py::validate_production_admission_contract

<!-- implementation-sha256: e97a5d6c01576bc41111ca500cdf2327651d3889805189cd08733ea24267ff36 -->

Require positive generation memory/storage/thread settings, an explicit expected allocation with valid MPI/OpenMP/CPU-slot relationships, and nonempty exact staging references. Validate the optional allocation reference and return None. This validates the requested admission contract; actual assigned resources and staging bytes are checked by the admission reader.

## core/flagship_production_campaigns.py::production_admission_runner

<!-- implementation-sha256: c494f1aacc144f9fe72b4c4fe988a240fcedf03e21a78404e5063d081b69f2ed -->

Run production admission using the stage destination as output parent, write generation/production-admission.json and return the admission result. The delegated check verifies allocation, source/staging pins, host capacity and disk availability; it does not evolve the production simulation.

## core/flagship_production_campaigns.py::prepare_production_campaign

<!-- implementation-sha256: 69dee752d9281f75da3ac90ff38c0e7b4f57afab71de2a6d8f9baa0b436ff048 -->

Validate pinned production staging, hardware settings, generator admission, traced host regions and frozen phase/seed lineage. Create a fresh campaign with an admission gate, per-case native zoom-generation recipes and evolution-only GADGET recipes; optionally prepare restaging/generated-flow recipes and checkpoint policies. Compile and publish workflow/manifest plus check/run/resume commands. Return RECIPES_PREPARED references and outstanding requirements. An absent allocation keeps launch_ready false; preparing recipes does not execute native generation or evolution.

## core/flagship_analysis_profile.py::portable_analysis_profile

<!-- implementation-sha256: 170988cfd07bee0090315695af2fdef3a83ed0366523dc67656bc3578b922d43 -->

Validate the exact BAO observation/inference request, bounded resources and pinned CLASS source configuration. In a fresh workspace build CLASS, then construct an initial CLASS stage and a runtime BAO preparation stage; validate their DAG and return the resolved profile/path. Retain failure.json on errors. Engine preparation is real, while CLASS physics and posterior inference run later through the workflow.

## core/flagship_analysis_profile.py::runtime_bao_inputs

<!-- implementation-sha256: 7fe0d8055f9a2bffc1677a2fc3b16b2f7c7c7dad6224924d329231de9b40fb20 -->

Validate runtime BAO parameters and the original owned public CLASS bundle/linear receipt. Resolve and hash-bind the observed mean, covariance and provenance references under their intake schema. Return the CLASS bundle and source pins. This binds input lineage without deriving a drag scale, fitting a posterior or producing a graph.

## core/flagship_analysis_profile.py::runtime_bao_runner

<!-- implementation-sha256: 65f3687dba251a0ff27b6c223fb00e5db18fc6634dec8ff607a677f90ff0dda1 -->

Check workflow memory/disk, ingest the published BAO observation and verify the frozen CLASS binary/configuration. Run a bounded CLASS drag calibration with only output root and thermodynamics verbosity changed; retain invocation, logs, execution metadata and parsed sound horizon. Prepare compressed-BAO or posterior recipes and figure/gallery continuations, then recheck source hashes and write bao-preparation.json. Return follow-ups and variant counts. Drag calibration executes here; posterior sampling and plotting remain deferred, with science NOT_ACCEPTED.

## core/flagship_runtime_history.py::runtime_history_inputs

<!-- implementation-sha256: b20cc1e4f491d7aa8109d204944136efabc49c7eac2c28daf68d92e2915d6ea9 -->

Require the miniature/local-preparation/SIDM request schema and an owned zoom-preparation receipt covering the exact configured case matrix. Verify each original generated zoom and passing linear/phase audits, matching host, refinement, trace, configuration and staged assets. Bind parent answers, generator, flow, history and combined templates and recheck all hashes. Return planned preparation, staged cases and pins; this validates existing lineage without evolving another run.

## core/flagship_runtime_history.py::runtime_history_runner

<!-- implementation-sha256: 8a7459a4adc7ef9f8e615e2f30547b0f8500ee9711ade68ee45b73362ffa1614 -->

Use the actual shared CLASS bundle and resource-adjusted combined answers to prepare a native campaign. Reuse completed generation stage definitions only after matching their generation parameters, and construct evolution-only follow-ups for the remaining work. Prepare per-case histories/restart diagnostics and bind their references into the continuation, rejecting paired branches. Recheck sources and write runtime-history-preparation.json. Return new follow-ups plus reuse counts; newly prepared native evolution is NOT_EXECUTED and convergence remains pending.

## core/flagship_runtime_measurements.py::measurement_preparation_inputs

<!-- implementation-sha256: 82f4525b14fd31a0772c6cf20c4ca0cd09a7025fdb663a8f6ccab675413fdecb -->

Validate measurement limits/options and owned history-case references, including explicit analysis-export and black-hole/recoil selections. Bind the original CLASS bundle either through its public receipt or through matching actual evolution inputs and saved spectrum configurations. Verify power inputs and optional baryon/web/branch designs, then recheck hashes. Return case records and source pins without executing measurements or manufacturing missing inputs.

## core/flagship_runtime_measurements.py::measurement_preparation_runner

<!-- implementation-sha256: ffbf3c71185285b1946fb7a0052d7a86ae6f23412d04a1a4d2453b7dc57f5278 -->

Check allocated memory and require actual saved epochs plus one original hydro/source history for each case. Prepare star-formation, mass-return and metal figures, common linear power and requested analysis, black-hole, baryon, web or branch products. Analysis copies retain verified private-run lineage; unavailable required products are recorded as unresolved. Bind case references into the continuation and recheck source hashes. Publish measurement-preparation.json and return follow-ups/counts. Measurement execution stays NOT_EXECUTED until those recipes run.

## core/flagship_zoom_preparation.py::validate_zoom_preparation

<!-- implementation-sha256: 0ad0fb00fe260c048134d78b1d78b22a0812bc26d5f2b53b693863474ab2fa75 -->

Require the complete actual-parent zoom request, a safe prefix, nonnegative snapshot, unique host rows and increasing distinct refinement levels from 5 through 20. Validate explicit trace padding/aperture, bounded generation/audit settings, generator/admission references and an optional continuation dictionary. Return None; lineage and asset bytes are checked during input intake.

## core/flagship_zoom_preparation.py::zoom_preparation_inputs

<!-- implementation-sha256: 76ee02e22ddbbf32102eb05d54edd76e9f52961ebfa0dace3b5f224e13478664 -->

Validate the request and require an owned completed native parent evolution and original contracted uniform-parent IC generation. Match its initial-condition hash, retained parent noise, seed hierarchy, frozen phase prefix, linear descriptor and admitted generator. Reject inconsistent refinements or changed source bytes. Return pins, parent configuration, parent level and original IC path. Intake validates existing products without selecting hosts or generating refined ICs.

## core/flagship_zoom_preparation.py::zoom_preparation_runner

<!-- implementation-sha256: 38350d6d06bb3b8d3e0edfb2475197f278bf60d7a60fd4eb70ee9e57ba403916 -->

Check allocation, select actual catalogue hosts and encode the retained parent noise into a frozen prefix, comparing any declared prefix bitwise. Trace each host back to original ICs; stage each requested refinement, apply its traced region and run native resource planning. Enforce geometry-based memory and aggregate retained-storage bounds. Publish zoom-preparation.json and generation/continuation recipes with source pins and resource estimates. Return follow-ups and target/case counts. Host tracing and planning execute here; native zoom generation remains NOT_EXECUTED and science NOT_ACCEPTED.

## core/campaigns.py::main

<!-- implementation-sha256: ccc2f502ab684f3bd5775a9fc0062c019383fa1b5547c1f5b2914be65cc4b01e -->

Parse native-prepare domain arguments and lazily import only the selected campaign producer. Common domains consume a JSON request and fresh destination; responses binds a design/template hash, while joint-response consumes original IC manifest/execution references. Resolve SIDM and joint-response assets beside their request/design file where explicitly supported. Print the prepared result as JSON and return zero. This adapter prepares campaign recipes; native evolution is launched separately through their workflow.

## core/parameter_explorer.py::parameter_value

<!-- implementation-sha256: fa818044f12b43d8479fc93cbe8a0541972014f3a9ffc196eeec759a1d4d64e0 -->

Walk a nonempty dotted configuration path through dictionaries. Return None for a missing component, otherwise accept null, text, Boolean, integer or finite floating-point values; reject containers and nonfinite floats. This extracts a display label without interpolating parameters or running a simulation.

## core/parameter_explorer.py::make_parameter_explorer

<!-- implementation-sha256: c41147b4aa7d722d15ab940bbc212d351125d38bb10bfa0b6f8d75b9c682bb16 -->

Require two to thirty explicit runs and a fresh output directory. Lock runs, verify source-bound analysis/map publications, namespace and epoch agreement, and load unadorned square log10 maps within the declared memory budget. Require common field/projection/units and reject temperature without a common representation contract. Render maps with a shared 1st-to-99.9th percentile display range and supported saved HMF curves, retaining units, geometry, parameter labels and source hashes. Reverify sources, publish an offline HTML selector with PNGs and explorer_contract.json, and return the HTML path. Differences are descriptive: the selected runs need not have matched phases, epochs or other parameters.

## core/paired_atlas_figures.py::save

<!-- implementation-sha256: cfb4777dc2d3eeff4b229648c5ffe66e7e1a39135e48362bf693b871567bc6f7 -->

Apply the paired-atlas dark Matplotlib style, create the destination parent and save the supplied figure as PNG, PDF and SVG at the same base path. Close the figure; when arrays are supplied, also write a compressed NPZ beside it. Return None. This exports an already constructed plot rather than computing its physical measurements.

## core/paired_atlas_figures.py::render_pair

<!-- implementation-sha256: e5795c63b733bb8d2e3e230c6e1a81a643bdb9ede6c786bff1b16abbcf766a96 -->

Load EDE/LCDM numeric epoch caches into a fresh or empty output directory and require matching comoving box and redshift. Produce differential FOF/subhalo abundances, cumulative host counts, supported power/Vmax curves, speed distributions and three projected density comparisons. Use a shared 300-particle mass floor, count/volume/log-bin-width abundances and Poisson count intervals; power ratios interpolate only overlapping positive bins. Density displays use periodic two-cell Gaussian smoothing and shared colors, with log10 ratios rather than significance. Save PNG/PDF/SVG and relevant NPZ/CSV arrays. Recheck input/renderer hashes before publishing comparison_manifest.json and a sealed core_render_receipt.json binding records, numeric caches, source code and original exports; return the manifest. Partial outputs remain unsealed on failure. The comparison does not establish convergence, joint covariance or an EDE-only causal effect.

## core/paired_atlas_figures.py::render_available

<!-- implementation-sha256: 449352f16d78339820285112e9de184a8a60ff8ab5ec11af3ee2d95d1ad64f8a -->

Scan atlas measurement JSON files for EDE/LCDM snapshots present in both runs and near configured priority redshifts. For existing comparisons require a sealed core-render receipt matching current record content, numeric cache bytes, renderer sources, original manifest fields and every original image/table/array export. Reject stale, legacy, incomplete or changed outputs without overwriting them. Otherwise call render_pair and print progress. Write paired_figure_progress.json only after all selected epochs succeed and return the comparison list. Separate scripts may append gallery entries; their additional products require independent provenance checks outside this core receipt.
## core/native_gadget_ancestry.py::recorded_link_sources

<!-- implementation-sha256: bafbc7144ca32c29ff8871a3bad40ff9c8ffface43b45be3ba6abc496a569a50 -->

Return absolute path/SHA-256 bindings for the six bundled GADGET descendant/progenitor writer, operator and run-loop source files used to interpret recorded ancestry. This hashes source bytes without compiling or running GADGET; a completed evolution must independently bind the same sources.

## core/native_gadget_ancestry.py::native_link_table

<!-- implementation-sha256: fcb4298824ece2a9ee816ccafd27e9f6203fbf7bb888d0db034faecd45bca2d7 -->

Read an original saved desc or prog SUBFIND table under predeclared row/memory bounds. Locate only the native catalogue output tree, reject duplicate single/sharded publications or shard gaps, and require every file hash in the completed evolution inventory. Validate header census and exact unique row IDs; preserve signed -1 links as unavailable through a separate Boolean mask. Return links and availability sorted by original row ID plus file pins. This reads native recorded relationships rather than inferring them by mass or proximity.

## core/native_gadget_ancestry.py::build_native_fof_ancestry

<!-- implementation-sha256: 6de4eae714a141a4c6f7eb30285c2bc4f8dbe170926f9d86ef71cc8349a7439e -->

Require consecutive actual saved snapshots, bounded graph/catalogue work and native writer sources owned by the completed evolution. Read original SUBFIND spans and FOF catalogues, require increasing cosmological clocks, and create epoch-local halo nodes with proper-kpc positions, Msun inclusive masses and particle-count quality flags. Project previous rank-zero centrals through recorded native descendant/progenitor links into the next FOF membership; retain unresolved links and ambiguity. Recheck all source hashes, publish the IdentityGraph and return its report plus unresolved records. These edges describe FOF membership joins, not continuous galaxy coalescence or measured particle transfer.

## core/research/checkpoints.py::publish_native_checkpoint_marker

<!-- implementation-sha256: fd64d9408b599f2d64280a889deb3c1b3f9651f339e11c3c97086f49ca3f7ac4 -->

Require the run and fresh immutable marker inside the owned workspace, with the original MPI count and executable hash. Under the marker lock verify all native rank checkpoint payloads, checksum/layout receipts, minimum generation and epoch sidecars. Bind original executable, parameters, compiled options, schedule, ICs, physical assets and configuration; rehash them and reread the checkpoint set to reject replacement during publication. Atomically write the marker and SHA seal, returning COMPLETE integrity metadata. Checkpoint completeness does not establish scientific acceptance or completed evolution.

## core/research/checkpoints.py::checkpoint_marker_main

<!-- implementation-sha256: 21d2b887ee21a2251c727b7c8ab5d700e4f8360a95970c9ed9ebee9265bec17a -->

Parse required --run, --marker, --workspace, --mpi-ranks and --executable-sha256, plus optional --minimum-generation defaulting to one. Publish the verified immutable checkpoint marker and print its result as JSON; normal completion returns None. This command observes already written native checkpoints and does not start or resume GADGET.

## core/research/checkpoints.py::validate_evolution_checkpoint_policy

<!-- implementation-sha256: 4050a340cac1b0e2c429944e1c35597de1967d4ae559326b1fbb40f9d35336f9 -->

Return immediately when a workflow row has no checkpoint_resume policy. Otherwise require the public gadget command, the exact registered marker/resume argv and exactly one --run-dir argument pointing to {generation}/run. Raise for mismatched policies; return None on success. This checks the continuation recipe without reading checkpoint shards or granting restart admission.

## core/flagship_inference.py::validate_inference_figure_stage

<!-- implementation-sha256: 601634f46985a9b743ef7552b86c5a2fe09ef0532d0915de2c5307ae69c5dcd9 -->

Require positive finite non-Boolean memory and an exact native-inference-figure-request-v1 request containing variant and original product reference. Accept only actual_posterior, posterior_predictive, sbc_ranks, coverage, ood_support or graph_baseline_gain. Return None on success. Structural validation does not read the original inference product or establish statistical acceptance.

## core/flagship_inference.py::inference_figure_runner

<!-- implementation-sha256: 2ab7a4f90fe79a1f0b09a812160cc0eb724a76bf22c10a305cf366b1932a962e -->

Validate the selected inference figure request and require its memory within the workflow allocation. Export figures under generation/figures, write inference-figures.json with variant/count and science NOT_ACCEPTED, then return COMPLETE execution measurements. Completion means the display export ran; it does not fit, calibrate or accept an inference model.

## core/flagship_inference.py::inference_product_inputs

<!-- implementation-sha256: e58843ed5264f260188edbb3873706662ad0cce81d8fe9b366dd94c222e2fc44 -->

Require an exact path/hash reference to original product.json within a bounded workspace. For a cosmography publication, verify its generation manifest, sampled BAO posterior, original numeric arrays and dimensionless dilation priors. Otherwise require a completed stream-stage receipt, replay its request/resource/code/source fingerprint, read the numeric product and verify all original producer/source hashes including the workflow owner. Return report, arrays, original stage and pins. This loads actual retained inference output without generating new posterior draws.

## core/flagship_inference.py::inference_figure_arrays

<!-- implementation-sha256: 8f707bdc5dbe0aad9b59d3de3ce1c936453dbf2aa0b5c8306ec9bcd8c76161be -->

Verify the original completed inference product and return plotted arrays, units, metadata, panel descriptions and source pins for one registered variant. Posterior panels sort actual samples and cumulatively sum normalized nonnegative weights, preserving parameter units; point estimates cannot replace chains. Predictive panels replay observed/replicated discrepancy tail probability. SBC and coverage panels use retained independent held-out rows, randomized PIT and original coverage intervals. OOD panels replay recorded thresholds; graph-gain panels replay paired score means and bootstrap acceptance probabilities. Failed support/calibration remains explicit. The function writes no files and grants no new statistical acceptance.

## core/flagship_inference.py::export_inference_figures

<!-- implementation-sha256: fabdbe7b9ac193673f16a7f156df7a1c44808bfc5e242d2041845ca69c8ce0ad -->

Build the selected inference arrays from their original producer and require a fresh destination. Publish inference-measurements.json with numeric arrays, units, source hashes and original acceptance decisions, then recheck all sources. Render an organized scientific gallery at 160 dpi with up to four scatter panels per figure and return its manifest. Axes retain the source parameter/statistic units; export does not resample a posterior or change failed calibration into acceptance.

## core/matched_perturber_figures.py::validate_matched_template_figures

<!-- implementation-sha256: 263ccfa96b08ee83032b5fe0d6c0f0272bd0beade127cb1ebe07fdb8a2b15b5c -->

Require an integer maximum_pairs of at least one, integer dpi from 32 through 2400 and finite memory of at least 0.1 GiB. Return None after checking these declared export bounds. Actual paired template identity, force model, geometry and array units are verified by the reader/runner.

## core/matched_perturber_figures.py::matched_template_inputs

<!-- implementation-sha256: cc2018727417205f4d1e85788c6af84f4322a42cb57a3db03fcd3882e07e6920 -->

Validate export bounds and limit metadata intake to one-quarter of the declared workspace. Read the original stream product and numeric-audit source bindings, requiring schema stream-perturber-templates-v1, COMPLETE execution and producer kind perturber_templates. Return report, numeric arrays and pins. This reads imposed-field forward experiments, not native BH histories.

## core/matched_perturber_figures.py::matched_template_input_bindings

<!-- implementation-sha256: 29e87c8c4a5985e2d2fb31930d882ebcd05b8b45f177f754d72a3f41f4f071e7 -->

Return the source-hash map from matched_template_inputs after its bounds, completed-producer and numeric-product checks. This preflight exposes dependencies for workflow reuse without rendering figures or rerunning the forward experiments.

## core/matched_perturber_figures.py::matched_template_figures_runner

<!-- implementation-sha256: 2f70a0f1a73b860d1360eba86793b5c566129ca2adcf48fab60d9155080d60ca -->

Group original templates by design and realization, reject duplicate IDs and bound BH-versus-CDM/WDM/SIDM pair counts. Require identical initial cohort, trajectory geometry, particle IDs, masses, times, bins and initial phase space, with declared force categories and physical units. Intersect support and form extended-minus-BH line-density, transverse-centroid and velocity differences; final centroid/velocity panels show vector norms. Publish numeric bridges and three-panel galleries with original units and conditional support. Return the gallery manifest. These are matched imposed-field predictions, with no inferred native BH ancestry, independent ensemble covariance or calibrated classification.

## core/flagship_products.py::flagship_product_input_bindings

<!-- implementation-sha256: 8ce7569816df2d590c9848dec06957d4451d70e4f6e2f5b0619e1a7ff8e2bcda -->

Dispatch by operation to the domain-specific input reader and return original source hashes for workflow/cache binding. Specialized paths verify native history, web, baryon, BH, power, abundance and analysis lineage; IC audits bind original configurations, transfer data and sealed source archives where required. The generic numeric-audit path binds product/NPZ and original sources, plus selected catalogue, graph, calibration or recorded-image dependencies. Request JSON intake is memory-bounded where explicitly handled. This gathers verified dependencies without executing the selected measurement/export.

## core/flagship_products.py::flagship_product_runner

<!-- implementation-sha256: 94a1a314d67ae4f4c012e538f2357be3138f3bf15cca4e1517270a81181e0dd2 -->

Require the resolved destination inside the owned workflow generation and positive finite memory within its allocation. Dispatch the selected operation to its actual measurement or exporter, passing original product/request plus required run, identity graph, MUSIC configuration/source index or timeline vertex bound. Return that producer’s result. Routes include native scientific figures, stream/encounter measurements, power and abundance products, IC operator audits and flow covariance. It executes the requested operation rather than merely preparing recipes; domain-specific producers retain their own provenance, support and acceptance rules.

## core/flagship_products.py::validate_flagship_product

<!-- implementation-sha256: 134c5dcb10b0fbcf504baba64d48a66774b60635eaa6119894e90cf9e3b53e89 -->

Require a supported product operation and positive finite non-Boolean workspace. Validate operation-specific schemas and controls: IC modes/configuration/source index, flow/noise/window covariance requests, assembly identity graph, stream/encounter request schemas and a positive exact timeline vertex ceiling. Return None on success. This is request validation; it does not execute exporters or replace their original-input and scientific-support checks.

## core/flagship_products.py::flagship_products_main

<!-- implementation-sha256: 8fd471d33679bf20073e3a9cf313f1c87655476141d467b2b707d15d78fa19f8 -->

Parse a product subcommand with input and required --output, adding explicit --request, original MUSIC --configuration/--source-index, --identity-graph or --run-directory where that route needs them. Ordinary routes default to 2 GiB and reject nonpositive/nonfinite memory; timeline exports default to two million saved vertices. Dispatch to the selected native product, IC audit, profile preflight, convergence figure or yield audit and print JSON-safe results. Return 2 for unresolved flow/covariance solves, 1 for reported FAIL, otherwise 0; exceptions propagate. Request-relative rebasing is explicit only in routes that provide it.

## core/native_stream_responses.py::validate_native_response

<!-- implementation-sha256: 08ff87cca558e9592ffdb26cc264f94274794fd3a88e47e7f94f185545f20043 -->

Require non-Boolean finite native-response memory of at least 0.1 GiB and return None. The frozen request, original paired histories, physical frames and source bindings are read by the subsequent intake; this small validator alone does not admit a response measurement.

## core/native_stream_responses.py::native_response_input_bindings

<!-- implementation-sha256: 26da0c1e92bbf839f990735fae950ce74d6aeb9e2e093e0757df0835cc496e80 -->

Load and verify the frozen response request, completed original paired stellar comparison and both physical frame products without replaying the paired comparison. Require exact cohort/clock/mass equality, original self-gravitating histories and shared frame definition, including recorded periodic images and proper position-derivative velocity conversion. If actions are requested, load their declared measured potential and add its source pins. Return source hashes without publishing responses or measuring actions.

## core/native_stream_responses.py::native_response_runner

<!-- implementation-sha256: 1655d251e2e992d17edb3cf49485b43986ad47a2fffaebeba946a8d95ae463f5 -->

Check workflow memory, replay the original paired comparison and require its exact stellar cohort, saved clock and common physical chart. Fit one supported reference principal curve at the requested epoch, freeze its track/interior bins and original masses, then measure both branches at every epoch. Publish line-density ratios/deficits, centroid/velocity differences, bootstrap comparisons and feature proposals with common support masks. Optional source-bound host actions produce paired action differences and mean squared changes relative to the first saved epoch; unavailable actions remain unresolved. Write stream-responses.json, optionally export figures, and return the product. A failed track fit publishes an unresolved result; measured deficits do not establish a detected causal gap, and squared action changes are not stationary diffusion coefficients.

## core/native_stream_responses.py::export_native_response_figures

<!-- implementation-sha256: 8430e8cdea62fe1cec4dbcff9efa5ee969c5f31755448a181abb8cccb65a2203 -->

Read the original native-stream-responses-v1 product and require a fresh output directory. Export numeric bridges and galleries for first/final fixed-track density, transverse offsets and local velocity changes; saved-time signed density deficits; and supported mean squared paired action changes. Preserve proper-kpc, km/s, Myr and squared-action units plus original support masks. Missing measurements render explicit unsupported series. Return the scientific gallery manifest, retaining synthetic-control labels and science NOT_ACCEPTED rather than assigning causal or convergence acceptance.

## core/merger_tree_map.py::connect_tree_to_maps

<!-- implementation-sha256: f98e96a9d467549941b05cebcfd85682428af3d1b16bb031fc9228e92cb5d26a -->

Under the native-run lock verify the original tree analysis and map publication share the run namespace, box and saved epochs. Require unique unadorned density maps with a fixed global color scale and bounded HDF5 display arrays. Recolor saved maps into PNGs and add click/keyboard tree-node navigation to the HTML, showing a projected center ring only inside the map slab; unavailable epochs/positions stay explicit. Reverify source contracts and publish a fresh directory with tree_with_map.html, map images and hashed tree_map_contract.json. Return the HTML path. This links existing tree/map measurements without deriving new ancestry or certifying scientific validity.

## core/native_cooling_control.py::method_source

<!-- implementation-sha256: fe197182332a2a719d273ea4ceb2a2dd9bc7efd360be87249ef45d3492ef2abd -->

Locate a named double/void coolsfr C++ method and return its original signature and brace-delimited body. Raise when missing or when braces do not balance. The scanner counts raw brace characters; it is tailored to the retained native source and is not a general C++ parser. It does not compile or execute the method.

## core/native_cooling_control.py::control_source

<!-- implementation-sha256: c6a7d219c084e17678dc4729b7c842a10de0f0103bcba22876f566990b045ee3 -->

Read the retained cooling.cc (or supplied alternative) and bundled cooling.h. Extract six native cooling/rate/temperature methods and four structures verbatim into a standalone C++ harness, replacing MPI/particle I/O and global unit plumbing. Return source text containing thermodynamic closure/species checks, analytic low-temperature hydrogen balance, physical-unit invariance, implicit-energy residual sweeps and invalid-input controls. The reproduce mode selects the narrow cold-temperature case; the normal sweep exercises the broader grid. This constructs a control source string without compiling it or representing a full hydrodynamic run.

## core/native_cooling_control.py::run_control

<!-- implementation-sha256: 4e651826fe981f1544226c53ee0ec3190e3b5c05ff6151f33353ea84c25544b7 -->

Require a fresh output directory and an available c++ compiler. Write the isolated native cooling harness, compile with C++11/O2 and bundled headers under a 60-second timeout, then execute the requested mode under another 60-second timeout. Retain build/stdout/stderr logs and report.json with native/source hashes, command, return code and parsed closure/residual measurements or failure text. Return the report; compilation failures raise. This executes shipped cooling methods with isolated plumbing, not a cosmological evolution or restart-compatibility proof.

## scripts/freeze_paired_rebuild_source.py::main

<!-- implementation-sha256: f89e7a180614f795108692a8012d40648a883137c7414cf192cbd29c6f31da51 -->

Parse the atlas path and copy selected first-party Python/JSON source, script/configuration files and eight named regression tests into Source/Rebuild_code, preserving relative paths and verifying each copied hash. Write source_manifest.json with checksums/sizes and a rebuild README describing archival N128 analysis/viewing scripts. Print the copied file count; return None. This overwrites selected snapshot paths rather than requiring a fresh directory. It excludes bundled native engines and installed environments, and the fixed selection is not a complete portable source distribution.

## scripts/pack_paired_blender_scenes.py::main

<!-- implementation-sha256: 863ebb6af8d3b9839456323cff18bc4ebdd02a819d78dedf4e2aa93103a17b56 -->

For the fixed EDE/LCDM rich and entry .blend paths, run the installed macOS Blender in background mode to save a compressed staged scene and freshly inspect it. Require passing load status, unchanged object count, frame range and initial frame, plus 86 saved epochs. Replace the original scene only when the staged file is smaller; retain compression logs, before/after hashes/sizes and load reports, then refresh scene-audit hashes. Return None. This archival maintenance command rewrites existing scenes and uses a hard-coded Blender application path; these structural checks do not compare every scene property or rendered pixel.

## scripts/pack_paired_xdmf_inputs.py::repack

<!-- implementation-sha256: d90993962d71203b211e3fc1798c092f680d90a72543a5c7fcce3cdb093ae5d4 -->

Consume an HDF5 path, original expected hash and packing-receipt path. Resume a receipt-matching packed file or recover its verified staging file; otherwise verify the original hash. Recursively copy groups/datasets/attributes into gzip-level-1 shuffled storage in 65,536-row chunks, then compare tree names, shape, dtype, attributes and every numeric payload byte before replacement. Write a recoverable receipt with before/after hashes and dataset inventory, replace the original and return the receipt. Failed staging is removed. This targets ordinary numeric XDMF HDF5 inputs; it is not a general preservation guarantee for arbitrary HDF5 links or storage features.

## scripts/pack_paired_xdmf_inputs.py::main

<!-- implementation-sha256: 75bd8ebbe1efa7081c485a721112d968ddcc765adb7588f8c25774d85d791a13 -->

Parse the fixed paired atlas path and process EDE/LCDM ParaView XDMF input sets with four worker processes. Retain the original contract, repack inputs through per-file recovery receipts, replace input hashes while preserving original-export hashes and publish lossless packing status/totals. Remove per-file receipts after successful model publication and print saved size. A contract already marked LOSSLESS_VERIFIED is skipped without rechecking files in this entry point. This is in-place archival storage maintenance, not a native rerun or a fresh scientific export.

## scripts/prepare_paired_analysis.py::prepare

<!-- implementation-sha256: 1651e76bc3c0ff1c1ce15704b218a011659ad5c92e9230b7088ebcc9e941517c -->

Create a local analysis view of a transferred native run: link its data, copy config.yaml unchanged and rewrite manifest path keys/values under the recorded original run prefix only when the local target exists. Retain original manifest/configuration hashes and a relocation list. Link input assets, rebasing halo_finding.json content separately, and return the new directory. An existing destination is returned immediately without revalidation. This convenience view preserves original native IC prefix tokens and does not verify the whole run or make existing destinations current.

## core/halo_products.py::export_native_structure_figures

<!-- implementation-sha256: af8391f2890bca83641635c3eaec993c5d3c245b64b85d06fe9066e6cdec9919 -->

Read a bounded original halo-structure product, require the native verifier to reproduce its report and preserve component/total profile units and exact support masks. Export shell density with mass-shot-noise bars, spherical-equivalent circular speed at outer edges, sqrt(trace(C)/3) peculiar population dispersion and local spherical anisotropy beta. Add available density/equipotential shape and conditional concentration products without refitting; missing products remain pending. Recheck source hashes, write products.json and numeric bridges, then return the organized scientific gallery. Radius is proper kpc, mass Msun and velocities km/s; these displays do not establish simulation convergence.

## core/halo_products.py::export_native_splashback_figures

<!-- implementation-sha256: 57ec95a080216730bead44c432ca9c94c838ee0a9c70f16d3602e755120290ec -->

Verify the original native environment and each bound halo structure share the actual run, snapshot and scale factor. Read original proper-kpc radii, logarithmic density slopes, full covariance and Boolean support; use sqrt(diag(covariance)) for displayed slope errors. Retain the measured valley candidates, prominence, half-prominence ln-radius widths and sampling percentiles without refitting. Publish per-host/smoothing numeric bridges, products.json and three-panel slope/depth/width galleries in a fresh directory after source rechecks. Return the gallery manifest. Ambiguous/unresolved features remain masked and science NOT_ACCEPTED.

## core/halo_products.py::web_inputs

<!-- implementation-sha256: 3f153867822bdab3b2f74dc6905a9c4289b0f2e6fe518cead822dd9860140b0c -->

Read the original environment product using half the declared memory, require equality with verify_native_environment and merge numeric-audit source bindings. Return report, arrays and source pins. This verifies saved web measurements without recomputing the tidal field or deriving filament identities.

## core/halo_products.py::web_region_arrays

<!-- implementation-sha256: 007eb026c677ba77b7981d30037ca00c1b72971c09f430ad11095bc79be8181f -->

Extract one smoothing field’s original filament-class voxel positions and uint64 component labels. Require comoving-kpc/identity units, the declared cubic mesh shape and exact equality to positive-labelled spectral grid nodes times box/mesh spacing. Return x/y/z and nonzero component IDs plus their identity namespace. These are product-local connected voxel regions, not ridge skeletons or cross-epoch filament ancestry.

## core/halo_products.py::web_alignment_arrays

<!-- implementation-sha256: da9c72f74480632229c294bb169f94e144cae51eb4050aab251655c8ff94dee8 -->

Resolve the halo’s unambiguous original order at smoothing index zero and extract its spin/shape unsigned alignment cosine and Boolean support. Use native total-profile outer radii for spin or requested shell outer edges for shape, retaining positive proper-kpc apertures and cosines in [0,1] on support. Return radii, cosines, masks and original isotropic-null mean array. This adapts conditional saved measurements without estimating a new alignment significance.

## core/halo_products.py::topology_alignment_arrays

<!-- implementation-sha256: 0469b21465d70c6638f8fa890f1d9731040a4c893d07fc1ff2004b777d4ce915 -->

Read a halo’s original spin/shape aperture radii, unique-assignment unsigned cosine and Boolean support from the native topology product, plus retained candidate-zero isotropic-null means. Validate proper-kpc versus dimensionless units, aligned arrays and supported cosines in [0,1]. Return these display arrays. Unique support excludes ambiguous assignments; the retained null is not a newly estimated covariance for the unique-selected sample.

## core/halo_products.py::export_native_web_figures

<!-- implementation-sha256: 0b064e8504c37b505baaaaf07f5d5d5980a0b583d38fdaba8400f37d07a6f86e -->

Verify the saved environment and create a fresh gallery of periodic class-2 T-web voxel regions in comoving kpc, with original component identities. When supplied, read sealed native skeleton arcs and recorded spatial images in comoving Mpc/h, retaining separate arcs with NaN breaks. Export original halo spin/shape alignment cosines against either local T-web axes or uniquely assigned native arc tangents at proper-kpc apertures. Recheck all source hashes and return numeric bridges plus organized figures. Synthetic controls retain their label; voxel regions, arc geometry and halo histories are distinct products, and export grants no convergence or filament-ancestry acceptance.

## core/halo_products.py::web_history_inputs

<!-- implementation-sha256: ab800547ea06b5b0326cd776b1eeddf0eb171f2c377f46a2b7e45a40ddd21a62 -->

Read the saved web history using half the declared memory, require equality with verify_environment_history and merge original numeric-audit source hashes. Return report, arrays and pins without deriving new time interpolation or reorientation events.

## core/halo_products.py::web_history_arrays

<!-- implementation-sha256: 7c32ea5ec0516cb69e64b54a39ba7f434e91452b8bb1ffb7325b12a71d971c17 -->

Extract one smoothing field’s saved proper-time clock, adjacent time brackets, unoriented axis-change angles and original clock-bounded accepted-interval mask. Require strictly increasing finite Myr times, brackets exactly equal to consecutive saved endpoints, aligned Boolean masks and supported angles from zero to 90 degrees. Return interval-end times, full brackets, degrees and masks. The interval end is a display coordinate, not a measured continuous event time.

## core/halo_products.py::export_native_web_history

<!-- implementation-sha256: 3d3a693949835f55e817d88144ef742d1ca758cdec1271a42cb0a57c99b9d785 -->

Verify the original history and require a fresh destination. For each smoothing scale publish a numeric bridge and scatter plot of supported local T-web axis changes in degrees against saved interval end in Myr, retaining full time brackets, persistent host graph nodes and original cadence/clock decisions. Recheck source hashes and return the organized gallery manifest. This tracks local unoriented web axes along a host history; it does not infer cross-epoch voxel-component ancestry or a continuous reorientation time.

## core/native_bh_campaign.py::bh_history_inputs

<!-- implementation-sha256: abfb893e297f86e55f75863a558644dee9287b8642274ddec5aba3ac94196af0 -->

Require positive finite memory, actual saved snapshots and a saved-epoch census within maximum_saved_epochs. Return SHA-256 bindings for config.yaml, manifest.json and every native snapshot shard. This inventories existing BH-history inputs without measuring BH growth or requiring that a merger occurred.

## core/native_bh_campaign.py::native_bh_history_runner

<!-- implementation-sha256: 32a4267cea0e218f2c534daa70ad0789f66b5c068256570d4229fe0118d05a78 -->

Require history memory within the workflow allocation, bind original run/snapshot hashes and run analyze_black_holes into generation/history. Export the native BH history figures into generation/figures, recheck all input hashes and write history-result.json with measurement and figure count. Return COMPLETE execution metadata with science NOT_ACCEPTED; this analyzes retained output rather than evolving a new BH run.

## core/native_bh_campaign.py::publish_recoil_images

<!-- implementation-sha256: 7b3c86c155dcae2bb86820f683132a15c0d62e2d2ed70e3d4e0b7007f95fad59 -->

Verify the original BH history/run and exact survivor ID, then scan native type-5 snapshot chunks at the recorded epochs. Read retained per-drift periodic image offsets, support and anchor clock for that identity; reject duplicate identities. Inconsistent supported anchors invalidate all support, while absent legacy state remains explicitly unsupported. Rehash sources and publish the numeric recoil-image product with namespace and row epoch contracts. No temporal winding is inferred from wrapped positions.

## core/native_bh_campaign.py::native_bh_event_figures_runner

<!-- implementation-sha256: 2f88bc042697dca74c80ffc123d87352c19cf946e33661828fad43a84056ff74 -->

Verify BH history and hydro/source journal refer to the same run within allocated memory. Use explicit observed pair IDs or the first two sorted observed IDs for separation display, export original merger events and select an explicit recoil or the first available journal merger. Read native image counters before recoil export; absent identities/events remain unresolved rather than invented. Write event-figures.json and return selection, actual merger count and gallery totals with science NOT_ACCEPTED. A displayed pair is not automatically a bound binary.

## core/native_bh_campaign.py::bh_event_stage_inputs

<!-- implementation-sha256: 994094655e0bac78252b494b83b26f5f795428b0a82dd7deef9c280f8a843772 -->

Return combined source pins from the original BH history reader and hydro/BH-event journal reader. This supplies stage dependencies without rendering pair, merger or recoil figures. The execution adapter separately checks their common run.

## core/native_bh_campaign.py::bh_matched_stage_inputs

<!-- implementation-sha256: e748cd4aba27be4659c803bd5210828d4527198810e82954c04d77bc788ef104 -->

Verify the frozen pair-request hash, read it and attach the actual joint-covariance path/hash with the matched-response schema. Run response_inputs to verify exact counterpart, phases, original structures and ordered covariance lineage. Return the pair-request and producer source pins without exporting a response figure.

## core/native_bh_campaign.py::native_bh_response_runner

<!-- implementation-sha256: abe371e4f85618d05b87b167fbe2acdbb388852b584361b7a5b3c46bd9d7c8a2 -->

Require finite positive memory within workflow allocation and unchanged frozen pair-request bytes. Bind the produced joint covariance, write generation/response-request.json and execute the native matched-DM response exporter under generation/response. Return COMPLETE execution with figure count and science NOT_ACCEPTED. This measures/displays the retained branch contrast; it does not evolve either branch.

## core/native_bh_campaign.py::prepare_native_bh_campaign

<!-- implementation-sha256: bdbaf275f28c6083edc3ce71c650fd3754f072e25c43fed44f53bcf8cfb0f07b -->

Validate the exact campaign schema, finite resources, hydro tolerances and snapshot list equal to the actual saved set. Check the configured native BH output contract and source-bound matched covariance request, requiring the displayed run to be one matched branch. Build and validate five stages for history, source journal, event figures, paired covariance and DM response. Recheck source/registered code hashes and publish pair-request, request, workflow and manifest in a fresh directory. Return RECIPES_PREPARED for eight BH variants; preparation does not execute them or guarantee pairs, mergers, recoil or supported shape measurements.

## core/flagship_bh_response.py::response_pair_inputs

<!-- implementation-sha256: c90134a5713590ef4c0df6cab0abcf5de711381383b2d0832f27dcff59e81ce1 -->

Load hash-bound original left/right halo structures and finder comparison under memory limits. Replay exact membership/phase-space matching and require one reciprocal unique counterpart with matching centers, source snapshots and actual epoch. Require common proper shell edges, type-1 DM component selection and shape controls, equal non-BH configuration branches and exactly one enabled BH branch. Bind original run/IC assets and require passing actual IC phase comparison. Return both reports/arrays, counterpart, phase result and source pins; this intake alone does not establish physical causal attribution.

## core/flagship_bh_response.py::response_inputs

<!-- implementation-sha256: 412d05a7fefa2e9bbfe9941e33e94ed957a47ca74a703b322f35f937366f7409 -->

Run the full matched-pair intake, then read the hash-bound joint covariance numeric product. Require its native-bh-response-joint-covariance-v1 schema and exact ordered left/right original product hashes, merging its source pins. Return structures/arrays, covariance/arrays, counterpart, phase comparison and sources. Covariance units, dimensions, positive semidefiniteness and supported response vectors are checked during measurement propagation.

## core/flagship_bh_response.py::export_native_bh_dm_response

<!-- implementation-sha256: 2542e8e8fe228e861878994dcfa93eff3ac27d306be0140ab95e93a5e7663c1a -->

Require a fresh destination, bounded memory and a matched-DM response request. Verify original structures, unique counterpart, phases and full joint covariance; propagate right-minus-left density and two shape-ratio contrasts through the existing common-observable estimator. Recheck sources, publish matched-dm-response.json and density/shape error-bar galleries against proper-kpc radius, then return the gallery manifest. Density units are Msun/(proper kpc)^3 and shape ratios dimensionless. Errors condition on the selected counterpart with zero matching covariance; one host supplies no object-sampling covariance, and physical attribution/convergence remains unaccepted.

## core/flagship_bh_response.py::covariance_input_bindings

<!-- implementation-sha256: 4e489f427ed96c8a5646f952c71fb55446c52c51c8489db9b8d1b1689bc227f4 -->

Read and hash-verify a bounded covariance request with explicit bootstrap realization, seed and success-fraction controls. Run the matched-pair intake at half allocation and require identical nonempty multisets of original native IC file hashes, beyond passing phase comparison. Return request, loaded pair and source pins. This admits shared-ID bootstrap inputs but does not compute covariance.

## core/flagship_bh_response.py::native_bh_covariance_runner

<!-- implementation-sha256: f71df6656f9a25c948d5d02d4e97456e4850aa6b1d6d84e15536620b5f27962a -->

Check stage memory, verify the frozen paired request and extract original parent particle IDs, proper-kpc offsets and Msun masses from both structures. Require equal support radii and run paired_structure_bootstrap with original edges/shape controls at half memory. A shared multinomial draw on the union of IDs preserves A/B sampling cross covariance; complete-draw support determines whether covariance is available. Rehash sources and publish generation/joint-covariance.json with exact ordered structure hashes, vectors, masks and units. Return the numeric product; this is conditional particle-sampling covariance, not volume variance or dynamical convergence.

## core/expansion_products.py::load_frozen_estimator_products

<!-- implementation-sha256: 2df7a34ab3ca06ca6fd577095ba1c3ed56071c5e3894f196e276ff7b00d5ee68 -->

Read every supplied frozen estimator result, require its recorded method to match the dictionary key and reject duplicate (method, realization_id, frozen_hash) identities. Return the flattened verified product list. This intake does not refit estimators, access recovery truth or make repeated observers independent realizations.

## core/expansion_products.py::compare_methods_conditioned_on_halo_environment_and_model

<!-- implementation-sha256: b0f2e5fbb0592321309c92c7080da9740799ddb2690584523699f818f5f2f99f -->

Group frozen estimator results by fingerprints of their environment and explicit comparison_condition. For each group return its environment/condition, the distinct estimator-model hashes and bias/precision/coverage/redshift recovery computed against post-fit truth. Different model hashes remain listed within a shared condition; this grouping is not an assertion that their models are identical.

## core/expansion_products.py::paired_method_tensions

<!-- implementation-sha256: 039b0f63d0df780a6c68ac1a9c64e539fa50fa296e55cab6e73a3fe92a4b4b5f -->

Compute method means from the supplied aligned joint realization values. For every method pair use sample-mean covariance to form variance Cii+Cjj-2Cij, require it positive and return mean difference, its standard deviation and signed difference/standard-deviation pull. This uses supplied paired covariance rather than adding errors as independent; the pull alone is not a calibrated discovery probability.

## core/expansion_products.py::comparison_report_files

<!-- implementation-sha256: 2e42445f100db7656084e5f906f03fe0a0979c30bb9256ef5abae20c6b3bdae7 -->

Serialize recovery summary rows into comparison.csv and an escaped standalone report.html, returning a filename-to-text dictionary without writing files. Include method, observable/redshift, product and independent-realization counts, bias, quoted error, scatter and coverage, plus follow-up requests. Explain that H0 and H(z) remain separate and repeated observers/replicas are not independent simulations.

## core/expansion_products.py::expansion_acceptance_runner

<!-- implementation-sha256: 5dbca1dec5f286fa4fafa09f37ba45a42f1b0f3ac7970cc099d05ed4525b0ae3 -->

Require a positive integer memory_bytes within workflow allocation, read the retained comparison relative to request_base and evaluate its frozen acceptance requirements. Write generation/expansion-acceptance.json with retained source bindings, then return accepted-domain count and remaining scientific exclusions. Acceptance uses the exact declared recovery domains; merely finishing this stage does not accept missing or failed domains.

## core/expansion_products.py::validate_hubble_figure_stage

<!-- implementation-sha256: e51466423185d0e1103f7b3ad3925ae43c0bc8d4694deb142af70c6e4b525d64 -->

Require positive finite non-Boolean memory and native-hubble-figure-request-v1 with one registered variant. Require exactly schema/variant/bao_request for conditional BAO, otherwise schema/variant/product for distance ladder, sampled time-delay posterior or method comparison. Return None; original producer/source validation is performed during intake.

## core/expansion_products.py::hubble_figure_runner

<!-- implementation-sha256: 48399c2e4585ad74cbcd4d16a84158ba97b40c64d18e4a43585c1b65183ef77f -->

Validate the selected Hubble graph request and require its memory within workflow allocation. Export under generation/figures, write hubble-figures.json with actual variant/count and science NOT_ACCEPTED, then return COMPLETE execution measurements. This exports retained results without fitting a new expansion model.

## core/expansion_products.py::cosmography_generation_inputs

<!-- implementation-sha256: 608e23a9edab35b8923efb303f22090ba9eda2dd65ee1dfe38aea84adb1d5530 -->

Require an exact original nonsymlink product.json path/hash and bounded product/manifest metadata. Verify a completed cosmography-generation-v1 manifest, its request/input/source fingerprint, every contained payload hash, producer code and original measurement/model input hashes. Require the product to belong to that publication and validate frozen-estimator semantics where applicable. Return report, original manifest and source pins without evaluating a new likelihood or posterior.

## core/expansion_products.py::hubble_figure_arrays

<!-- implementation-sha256: ae993a014365db041256c47ac15fcf95503d97bbb1fb6c779d90ab7d376e7b42 -->

Return arrays, units, original-source metadata and pins for one Hubble variant. Conditional BAO retains full measured covariance/availability and uses sqrt(diagonal) only for displayed bars against the fixed CLASS prediction; no CMB likelihood or H0 posterior is supplied. Ladder export retains original H0 and its 68% interval in km/s/Mpc. Time-delay export requires actual sampled full-lens H0 and Mpc distance draws, replaying their H0 summary and empirical CDF. Method comparison keeps post-fit H0 and H(z) residuals separate with original quoted errors. No files are written and no posterior is fabricated.

## core/expansion_products.py::export_hubble_figures

<!-- implementation-sha256: c6acee563043a8653e02247a372551538d7dbc18e36d378680b66fbf10c927bd -->

Require the Hubble request schema and a fresh output directory. Build original-source arrays and publish hubble-measurements.json with units, support and provenance; construct ladder interval, lens posterior CDF, conditional BAO comparison or separate H0/H(z) recovery panels. Recheck all source hashes and render an organized 160-dpi gallery with up to four panels per figure. Return the gallery manifest, preserving original calibration/selection limits and science NOT_ACCEPTED.

## core/native_scene_overlay.py::prepare_native_scene_overlay

<!-- implementation-sha256: 266263580f355bb01a618955e88186647bf402bc225e2d5fd0b7dcb1434b1455 -->

Verify the bounded stream/perturber/encounter references and original native run/clock against the saved DM scene timeline. Transform each supported saved worldline endpoint from its proper local chart to global comoving Mpc/h using that endpoint’s own origin and scale factor. Publish exact identity-bearing point/line products, a measured-display build and identity map, attaching the display reference to combined. Keep reconstructed encounter positions in separate local proper-kpc event views because no continuous global chart origin is inferred. Enforce object/vertex bounds, recheck sources and return counts/pins. Marker radii/line widths are symbolic display sizes; this prepares scene assets without new dynamics or rendering.

## core/paired_atlas_site.py::rel

<!-- implementation-sha256: 5ecc38073f7804231d42f7bba41502fcba5deec5cd4bf90ceffe87b4b239d75e -->

Return the filesystem-relative path from base to path with forward slashes for HTML links. This formats paths without checking existence, containment or source hashes.

## core/paired_atlas_site.py::link

<!-- implementation-sha256: 6053c1bf750c1ff541dab6b946015f355eac09fb993428bdbf2975b1c09de6f1 -->

Create parent directories and a relative symlink to the resolved source when the target is absent. Keep a symlink already resolving to that source; replace a symlink pointing elsewhere. An existing regular file/directory is left untouched. Return None. This is navigation linking, not source verification or a guarantee that an existing ordinary target is current.

## core/paired_atlas_site.py::page

<!-- implementation-sha256: a805f399eefd0e210b5263f4935519cb2b5f3508502e9207c1993e1419f06235 -->

Atomically write an atlas HTML page with the shared responsive CSS, fixed section navigation and escaped document title. Insert supplied body markup, search/priority filtering and video-load error guidance. Create parent directories and overwrite the page path. Return None. The caller supplies trusted body HTML; this renderer neither measures data nor verifies navigation targets.

## core/paired_atlas_site.py::human

<!-- implementation-sha256: 8b92ca81df583c1b18d0207bde542c6017c5af228676ff127434936a0995d92a -->

Remove a recognized paired/model redshift-and-snapshot filename prefix and replace underscores with spaces for a display title. Return the resulting text without reading a file or interpreting its physical units.

## core/paired_atlas_site.py::figure_card

<!-- implementation-sha256: 57dbab45f0c67f4e00399a9b468d7ae1b0dd21b83ded919b2d17486f9c56c8ca -->

Return escaped figure-card HTML with a preview, humanized title/caption and relative links to existing PNG/PDF/SVG/CSV/NPZ/plotted-array/source siblings. This checks sibling existence rather than their checksums or scientific support; it writes no files.

## core/paired_atlas_site.py::native_spatial_gallery

<!-- implementation-sha256: 69db1d5b0cc4a71426451f9d6f335cf6acb18ff48e9743bd073b2de3c507c49c -->

Return False when product/gallery/gallery.json is absent. Otherwise read its figure entries, build cards linking available formats/physical arrays/source records and display the sum of recorded supported entries across panel series. Write product/index.html through the shared page renderer and return True. The displayed count is plotted-series entries, not unique particles or a fresh measurement; source files are not revalidated here.

## core/paired_atlas_site.py::build_model_explorers

<!-- implementation-sha256: 0e5cdbed91bdc4191506909c7d71e9119dc0f2a3ef74f3725226375552104b26 -->

Build a model’s spatial-explorer and halo-history HTML indexes from existing images/pages and native scientific gallery manifests. Prefer readable halo cutaways when present, include void histories and finder comparisons, and link saved history-bundle figure siblings into the model history folder. Return None. This creates navigation/pages/symlinks using the fixed archive layout; it does not calculate halo properties or rerun analyses.

## core/paired_atlas_site.py::build_epoch_navigation

<!-- implementation-sha256: 19b2113aa334373754838ed686e6901f61e50ad69a422bbe517918310759f443 -->

Read existing EDE/LCDM measurement manifests and numeric caches. For each saved epoch create missing three-view DM preview PNGs with periodic two-cell Gaussian smoothing and fixed log10 surface-density limits 10.6–12.9 Msun/Mpc², plus a missing interactive sampled-particle page. Link priority-epoch figures/tables from the lexicographically latest native analysis bundle and build model/epoch/explorer/history pages. Return per-model summary rows with page/priority fields. Existing previews are reused without freshness checks; fixed N128 archive text is descriptive, not a general run census or numerical audit.

## core/paired_atlas_site.py::build_tables

<!-- implementation-sha256: 8fbe3ae42c0249a61767e4cf7be77e9bde7bc226a4232e98332a6eec6b0756cc -->

Join supplied EDE/LCDM summary rows by snapshot, write the all-epoch comparison CSV and build the numerical-table HTML index. Include host-count floors, largest FOF mass, subhalo count and particle-speed statistics with EDE/LCDM ratios; zero denominators are explicit UNDEFINED_ZERO_DENOMINATOR. Return comparison rows. Matching snapshot labels are assumed to exist in both models; this table assembly does not independently establish matching epochs or source freshness.

## core/paired_atlas_site.py::film_library

<!-- implementation-sha256: 722e2c80e5749224965e7c52f60ff76dc2a1912b9e9722bbbd3faf3d9aea2f76 -->

Discover visible, nonstaged MP4s in the fixed model movie folders, read optional film_source.json and create searchable film cards plus one on-demand HTML video player. Link existing posters, downloads, tree/void pages and per-model movie indexes; label frozen spatial tours separately from saved-time evolution. Return the discovered film paths. This indexes existing films without encoding video, checking codecs or verifying their source hashes.

## core/paired_atlas_site.py::install_local_viewer

<!-- implementation-sha256: c4e76859b94182b3309c9bed219f11c70f0266eb7ac154ec508420531464b1d0 -->

Copy the repository’s scripts/serve_results.py into the atlas Source directory and atomically write an executable Open_atlas.command launcher. The launcher changes to its own folder and starts python3 on an automatically assigned local port with browser opening. Return None; this installs the helper but does not start the server or verify a recipient’s Python installation.

## core/paired_atlas_site.py::build_fitting_galleries

<!-- implementation-sha256: 9a6c2b527e6415f068de77e036caf918f5e67c1f7119cdb7898b3b939cff27f6 -->

Return immediately if highlighted fitting-function coverage is absent. Otherwise scan comparison manifests for fitting-function PNGs, require each referenced per-model table hash to match coverage, link native diagnostic siblings and write epoch/model/paired fitting galleries. Keep the 300-particle floor, empty-bin limits, native axis conventions and CDM-to-EDE extrapolation context visible. Return None. Only the referenced table hashes are checked here; navigation generation does not recalibrate fits or certify all image assets.

## core/paired_atlas_site.py::build_topic_hubs

<!-- implementation-sha256: 37b03868253207220f62938457c4e45f9b0bb8dc0f80f3c2f738439804a60dc3 -->

Write Spatial_explorers.html and Halo_histories.html with direct links to the fixed EDE/LCDM model collections. Return None. These are static navigation hubs and do not test that every linked product exists or is scientifically supported.

## core/paired_atlas_site.py::refresh_site

<!-- implementation-sha256: 2f2a6848032f49f009500cf1398587c21dfea2f91f521b73ab371837bced20b1 -->

Rebuild the fixed N128 paired-atlas fitting, epoch, topic, comparison, table, movie, Blender/ParaView and coverage navigation from retained manifests/files. Install the local viewer, write START_HERE.html and catalog.json, and overwrite the atlas README with its layout/source guidance; print the entry-page path and return None. Existing preview/interactive assets may be reused by helper functions. This archive-specific publisher uses fixed timeline/box/priority descriptions and does not rerun simulations, render Blender, encode films or provide a comprehensive data/provenance audit.

## core/halo_campaign.py::prepare_native_halo_campaign

<!-- implementation-sha256: 3b8ae585f2af99ace201325caa3a5e2203a6853ca5d39acff28e1d61297c4c82 -->

Validate the exact halo campaign request and finite memory/disk allocation in a fresh destination. Bind a prepared assembly campaign with its orbital recipe, preserve inherited stages/resources/dependencies and add explicit analysis, abundance, structure and splashback exporters. Require exactly 26 variant routes backed by compiled producer stages, merge rather than erase inherited input pins and recheck registered source hashes. Write request/workflow/manifest and return RECIPES_PREPARED. This connects executable recipes; actual populations, field support and convergence remain runtime requirements.

## core/halo_campaign.py::validate_structure_template

<!-- implementation-sha256: cad5b3b0300cbded1064905b004412f5251d137748b1cd33292329656e18ab69 -->

Require the complete native-halo-structure-v1 template field set without run/snapshot/host selectors and finite positive template memory within the supplied bound. Return the template unchanged. Physical estimator controls retain the native producer’s execution-time validation; this adapter does not choose a host or measure a profile.

## core/halo_campaign.py::validate_recovered_structure

<!-- implementation-sha256: 04b0568c884212e149c23385a3c6178a8da83729c900e0a8102f9e245dcdb408 -->

Require a nonnegative exact snapshot index, explicit Boolean export_figures, finite positive memory and a lowercase 64-character SHA-256 template digest. Return None. Recovery lineage, original source bytes and the actual descendant are verified by the runner.

## core/halo_campaign.py::recovered_structure_runner

<!-- implementation-sha256: 6c3edb9547be75b506a90edd1698d8ea08fb04d165e3e01bfd6f755f1fcf2d95 -->

Verify the bounded frozen structure template and exact material-descendant recovery report/source hashes. Select exactly one record at the requested actual epoch; unresolved/disrupted/ambiguous recovery publishes UNRESOLVED_TARGET without a fallback host. For MATCHED, require the exact catalogue row, source generation and scale factor, then add only run/snapshot/host selectors to the template. Publish native structure and optional figures, recheck sources and retain status updates; failures/interruption record diagnostics before rethrowing. Return the measurement status with science NOT_ACCEPTED and no launch authorization.

## core/halo_campaign.py::validate_assembly_measurements

<!-- implementation-sha256: a92a8345b8d334884f2e6a554bacf07effd8c83edc37a2d979d29a5f25712b93 -->

Require finite positive assembly memory and exact positive maximum_nodes, maximum_edges and maximum_measurements. Return None. Actual graph census, template/source bindings and observation alignment are checked during input intake.

## core/halo_campaign.py::assembly_measurement_inputs

<!-- implementation-sha256: 33815126a611a3e346b5d544c442e2c6570a51c9c91cb36848c988b35cd7294d -->

Verify the hash-bound bounded observation template and original identity graph, including declared node/edge/measurement ceilings and decompressed graph workspace. Join each structure to its exact native FOF namespace/row/snapshot and original clock, rejecting duplicates, cross-finder row substitutions and conflicting attributes. Extract supported structure properties and optional matched central-galaxy attributes; require fixed physical spin/shape/morphology definitions across observations. Recheck sources and return graph, per-node updates and pins. This prepares measured attributes without publishing a modified graph.

## core/halo_campaign.py::native_assembly_measurements_runner

<!-- implementation-sha256: 791e27a89d7e88ca4abc70138256fb24029db3af1df29c71a98892198d05f3aa -->

Require assembly memory within workflow allocation and verify original graph/measurement inputs. Rebuild the graph with original epochs, node kinds/statuses, edges and evidence, adding only the validated per-node measurement attributes. Publish generation/identity-graph and measurements.json with old/new fingerprints and attached-observation count; return the summary. Existing quality/ancestry are preserved rather than promoted by profile support, and science remains NOT_ACCEPTED.

## core/halo_campaign.py::native_assembly_history_inputs

<!-- implementation-sha256: b3ba86cf0117b951435065b4db4336aa7ac2abe1ef4d3b814eae66ae4ddb973a -->

Verify finite positive memory, bounded graph/template metadata and decompressed graph arrays, plus the original history-template hash. Require the exact target/branch/events/accretion_fit template schema and reject authored epoch-time or scale-factor overrides: clocks come from graph epochs. Bind graph report/seal/arrays/database, original sources and cosmic-time assets. Return graph, template and pins without deriving assembly history.

## core/halo_campaign.py::native_web_epoch_runner

<!-- implementation-sha256: f333a3b0172eea3fdf04d1c766c58106afa2858009f44bc8aaae2770ffc64578 -->

Require web memory within allocation and bind an odd-mesh/filter design, exact native snapshot publication and same-epoch halo structure through native_web_epoch_inputs. Publish and verify generation/web-environment.json, then export web figures. Return COMPLETE execution, figure count, massive-particle row count and explicit filtered-density/eigengap/spin/shape support. This actually measures the retained field; it does not force filament detections or grant scientific acceptance.

## core/halo_campaign.py::native_web_history_runner

<!-- implementation-sha256: b7a4c1551d556b7c3f6d40a372fe72e3d35247884e3929aa5f98a4ba8696353b -->

Require history memory within allocation and verify each original environment/structure observation against persistent native graph ancestry and proper clocks. Publish and verify generation/web-history.json under the declared maximum interval in Myr, then export its history gallery. Return COMPLETE execution with figure count and the original clock-bounded support policy. This follows local field observations along a host history, not cross-epoch voxel-component identities.

## core/research/display.py::assemble_measured_display

<!-- implementation-sha256: f08559ad960d906e84e5516f4e8e798dba1b5a61db3ea7b906d558309fdf3e45 -->

Require a bounded measured-display-build-v1 request with at least two saved epochs and explicit frame, object, geometry, measurement and optional lifecycle limits. Under the destination lock, require a fresh generation, resolve producer references and verify sealed numeric reports, units, coordinates, exact uint64 namespaces and Boolean support. Copy original product bundles, validate shape/tidal/stream geometry, join same-object SIDM diagnostics and retain supported lifecycle events. Enforce aggregate memory/object/vertex counts, rehash every source and publish/verify display.json with its seal and assembly.json. Return the assembly receipt; failures retain failure.json. This prepares measured Blender inputs without rendering or supplying new dynamics.

## core/research/display.py::display_products_main

<!-- implementation-sha256: 0ec602fe09ccf8b686ca269de11e11b9fd1a16b81bac1019c992577334005549 -->

Read required --request JSON and --output, resolve referenced inputs relative to the request file’s parent and invoke assemble_measured_display. Print the assembly receipt as formatted JSON; normal completion returns None. This command produces portable measured scene inputs and does not launch Blender.

## core/research/display.py::validate_combined_gallery_stage

<!-- implementation-sha256: fff942250d009b0e111e737cab2bbe98c3d82c8107d973396d07e57cce76a2cf -->

Require combined-scientific-gallery-request-v1 with finite positive non-Boolean memory_gib and disk_gib. Return None. Full gallery/scene lists, figure limits, source publications and scientific context are verified during input intake.

## core/research/display.py::combined_gallery_runner

<!-- implementation-sha256: 6dea196d5c8e977bb4f298c6af705d6c8baeea15fb055e9b7389c1d47af7ce26 -->

Validate the combined-gallery request and require its memory/disk within workflow allocation. Combine original measured galleries under generation/gallery and verify the resulting sealed publication. Return COMPLETE execution with 102 registered variants, actual exported-view count and count of variants still pending measured inputs. Available exports retain science NOT_ACCEPTED; registered capability is not measurement coverage.

## core/research/display.py::gallery_inputs

<!-- implementation-sha256: 1a99855741547395f855beae195f1aae779546ddd10f3a90e2081480dced9b3b -->

Validate explicit gallery/scene references and bounded memory/disk/figure counts, then audit registered producer routes. Verify each original figure manifest, all three image formats, plotted/physical array contracts, source hashes and gallery scientific identity/selection; deduplicate by manifest hash. For optional scenes require sealed publication and an exact context binding plus inclusion of the same measured PNG. Recheck all pins and return figure records, source map and registry. This verifies dependencies before copying; a matching title alone cannot connect a scene to a measurement.

## core/research/display.py::combine_graph_galleries

<!-- implementation-sha256: 5c67b2db46a528fff39287b34c38716a7883731a301f9b15543c50ae68b86179 -->

Verify original galleries/scenes and require a fresh destination. Copy each original numerical/figure bundle into its manifest-hash directory, reverify figure and physical-array contracts and retain original contexts and scene links. Build all 102 registry entries, marking variants without supplied views PENDING_MEASURED_INPUT rather than generating substitutes. Rehash sources, write searchable index.html and seal gallery.json with the copied-file inventory. Return the combined record. This packages existing measurements without recalculating them or accepting unsupported science.

## core/research/display.py::verify_combined_gallery

<!-- implementation-sha256: e3d92234ba4614938904693f3e5aebdac82d842746abc297a02f7b61c9a6d384 -->

Verify the gallery SHA seal, exact 102 unique current registry entries, every inventoried artifact and the sealed navigation page. Revalidate each view’s original manifest, bounded array expansion, physical-array contract, formats, context, identity and science status; require complete inventory and consistent measured/pending statuses. With verify_sources=True also rehash original inputs and verify linked scene publications, exact context bindings and included PNG. Return the record. Disabling source checks verifies retained bundle integrity only, not current external provenance or scientific correctness.

## core/research/display.py::write_combined_gallery

<!-- implementation-sha256: c9587ba6c53910104f3229bf3d92937180d5d2f7bf25966635a04c4120cdf2e6 -->

Atomically write the combined index.html from supplied registry entries/views. Include each measured image, original formats/manifest/physical arrays, expandable target/epoch/selection context and exact-context scene links; show missing-measurement reasons otherwise. Add a case-insensitive filter requiring every whitespace-separated term. Return None. This renders navigation from an already prepared record and does not verify its artifacts.

## core/research/display.py::prepare_native_scene_selections

<!-- implementation-sha256: 515524a285ff1dbc42a55a23825131d3004614dbc34342a4709e610b687351af -->

Verify the scene’s saved DM timeline and source-bound progenitor membership/contamination products from the same run and exact epochs. Copy original stellar products and create identity-bearing inspector arrays with a one-epoch proper-to-global-comoving conversion, retaining source coordinates and integer display images. Attach glyph overlays only over their saved-frame interval; record unobserved membership explicitly. Convert measured contamination apertures to scene units, adding neighboring periodic sphere images where needed while counting all rendered copies against limits. Recheck sources, publish selections.json and attach its reference to combined. Glyphs/copies highlight existing particles/apertures and must not be counted as additional mass.

## core/research/display.py::export_scientific_suite

<!-- implementation-sha256: d1450ba2b62fe6cbcb588a487f5849e61425e083b006144477987445cc96f2d1 -->

Read declared sealed numeric source products under a combined decompression/memory budget. Require unique safe figure IDs, registered family/variant and bounded panel count; draw explicit measured representations with declared unit conversions and exact support/finite/log-domain masks. Retain the actual plotted arrays, units, recipe, selection counts and provenance in physical_arrays.npz/contracts and figure manifests; export PNG/PDF/SVG, optionally organized by format/family. Reject existing per-figure numeric publications, record absent variants/convergence observables as pending and write gallery.json/index.html. Return the suite receipt. Empty supported domains stay visibly unavailable rather than acquiring fabricated data.

## core/research/display.py::write_manifest_gallery

<!-- implementation-sha256: 827bb88b3ac1513228ea81e1e1583e209bf4c3023bfbc5d812e77c0ed10f35c9 -->

Atomically write a searchable scientific-suite index.html from supplied figure records. Include escaped figure/science labels, PNG previews, format downloads, manifests, physical arrays and pending-measurement entries. Filter by catalogue ID, family, variant or status with all whitespace-separated case-insensitive terms. Return None. Page rendering does not validate source hashes or change recorded support/acceptance.

## core/research/display.py::scientific_figures_main

<!-- implementation-sha256: b07d3247529954d55d4bfc6c9cdd4de4a07dee9463b26bb49c47a280e51e80fd -->

With --export-registry alone, write the current registered variant catalogue and return zero without rendering. Otherwise require request, --output and --memory-gib, read the request JSON and optionally enable --organized outputs. Resolve source paths relative to the request parent, execute export_scientific_suite and return zero. The command renders supplied sealed measured arrays; exporting the registry is not proof that every variant has data.

## core/initial_conditions.py::SecondOrderGrowth

<!-- implementation-sha256: bc015c0cf2670627ecb32353dba815aec588cbccb0fa5ed85ee0cf88a94a5450 -->

Frozen dataclass carrying the evaluated scale factor, second-order displacement/velocity coefficients, linear growth rate, seed scale factor/kind and ODE relative tolerance. second_order_growth fills the coefficient fields with -E and -dE/dln(a) in its target-normalized growth convention. This record stores the result and seed provenance; constructing it does not integrate a growth equation or validate a cosmology.

## core/initial_conditions.py::validate_twofluid_eulerian_seed_request

<!-- implementation-sha256: 5da3332d7db22e157bebb46ccd6f37fd8744feefcaf8b56fa6f52b2f50ccf3af -->

Require the exact native Eulerian-seed schema, optional Boolean pressure measurement, nonnegative snapshot and positive integer grid/census/chunk/cell-support bounds. Require at least four cells per grid axis, a namespace and finite positive memory, velocity-dispersion and relative-mass budgets. Reject grid/census workspace exceeding declared memory and return the request. Native snapshot lineage and measured cell support are checked by the producer.

## core/initial_conditions.py::twofluid_eulerian_seed_input_bindings

<!-- implementation-sha256: b4161d461f31076d7553e7b48a60c54920ba1c5a5f349a23195f024357798a59 -->

Validate the Eulerian seed request and obtain original completed-native-stage source bindings. When expected_native_source_hashes is supplied, require a nonempty mapping whose hashes all match those stage bindings. Return the source map. This checks retained inputs against campaign design without reducing native particles into cell averages.

## core/initial_conditions.py::twofluid_eulerian_seed_runner

<!-- implementation-sha256: 45878013a14c395e16e37da7469eb9d6b1c2e385472079b9826f61a8016b4998 -->

Require the request memory within workflow allocation and the resolved destination inside the owned generation. Return publish_twofluid_eulerian_seed on the original parameters. This executes the native snapshot-to-cell-average measurement; it is not a recipe-only adapter.

## core/initial_conditions.py::twofluid_eulerian_seed_main

<!-- implementation-sha256: 82b0e79c37e325f080307576dc27df44315c5d60c6f0ed7353ec91819bf305ff -->

Read positional parameters JSON and required --output, invoke the Eulerian seed producer and return zero. The command measures original full-box gas/DM cell averages; it does not insert request_base or automatically rebase embedded paths beside the parameters file.

## core/initial_conditions.py::validate_twofluid_native_ic_request

<!-- implementation-sha256: dc5aa1fca29541a2bd70840e9c8df9f3c3348c8c4bbe63ffd103480d6353dbb1 -->

Require the typed native-IC request and exact particle-projection/base-generation references, with explicit preservation of original native gas thermochemical state. Validate finite positive memory/disk, positive chunk rows and optional nonempty retained dataset-unit declarations. Validate any frozen window-comparison template after inserting a placeholder generation reference; templates cannot supply their own replaceable native_generation. Return the request without writing ICs.

## core/initial_conditions.py::twofluid_native_ic_input_bindings

<!-- implementation-sha256: 56b2ce0165da66857fad28d367ef4fe8d9387dc213ad78fd593fe14aff05da49 -->

Validate the writer request and read the sealed native-q particle projection. Require a frozen window-comparison launch template for finite-volume material cells. Resolve exactly one original q-load source from its hash, verify the declared native generation and require its IC bytes to be that same load. Return projection/native-generation pins including every original generation artifact. Equivalent-looking ICs, IDs or noise realizations are not interchangeable.

## core/initial_conditions.py::twofluid_native_ic_runner

<!-- implementation-sha256: b16ca3c6b1aaa7358f979eae8e4d6076cde77e669f045a11e3cc5820d685df24 -->

Require requested writer memory/disk within workflow resources and destination inside the owned generation. Return publish_twofluid_native_ic, executing the immutable transformed-IC writer with original q/ID and gas-state provenance. The writer’s scoped checks do not grant general physical acceptance.

## core/initial_conditions.py::twofluid_native_ic_main

<!-- implementation-sha256: acb87e17455c497c4a69062061c4c9ab86c1a2004d42f8e88275e5882a3e035d -->

Parse required --parameters JSON and --output, default request_base to the parameters file’s parent and invoke the native two-fluid IC writer. Normal completion returns None. Explicit request_base is preserved; the producer owns original projection/generation and thermochemical-state checks.

## core/initial_conditions.py::validate_twofluid_window_request

<!-- implementation-sha256: 5787784d004342c5a7120d1733e852ce121bd67f0429dc13bbb036c06edabf0c -->

Require the typed material-window request, positive exact particle/mode/window/chunk/phase-work bounds and finite positive memory/density/momentum/quadrature tolerances. Bound k times cell width by pi/2, require at least two increasing Gaussian orders from 2 through 32, distinct nonzero integer wavevectors and uniquely named windows over explicit native species and dyadic levels 1–20. Validate the generation path/hash reference and return the request. This validates comparison controls without measuring delivery error.

## core/initial_conditions.py::twofluid_window_input_bindings

<!-- implementation-sha256: 7080f7448b3246f71924668ff88af4b3e8de4f2d478b4c8b81817cdda0f9f4c2 -->

Validate the comparison request and hash-bound transformed native generation. Verify its actual IC/projection lineage and require finite-volume material-cell mode. Load original relative fields and return their pins merged with generation sources, generation JSON/seal and actual IC hash. This binds delivery-comparison inputs without evaluating Fourier density/momentum residuals.

## core/initial_conditions.py::twofluid_window_runner

<!-- implementation-sha256: 2ca6d3636deaca723b2b2113cd31c1fd624799ea588b70bfda0643b707ff8daf -->

Require material-window comparison memory within workflow allocation and publication inside the owned generation. Return publish_twofluid_window_comparison for the frozen request. This executes the scoped particle-versus-retained-flow comparison; it does not certify the physical seed or later target contamination.

## core/initial_conditions.py::twofluid_window_main

<!-- implementation-sha256: 54f6e405b2a0b588d49094ab25cf658c73871b2c35081d55db3e2e9d68518800 -->

Parse required --parameters JSON and --output, default request_base to the parameters file parent and invoke the material-window comparison producer. Normal completion returns None. The comparison uses actual delivered particles and identically windowed retained flow integrals, rather than substituting an unwindowed theory curve.

## core/initial_conditions.py::twofluid_relative_input_bindings

<!-- implementation-sha256: 66e71a899a39b1d0a6d481044331799370b7bd9faf795a4fbc09045846a5a775 -->

Load the initial_fields product relative to request_base under the request memory bound and return its original source pins. This reader verifies the retained numeric publication; exact seed epoch and required field units are checked when publish_twofluid_relative_product executes.

## core/initial_conditions.py::twofluid_relative_runner

<!-- implementation-sha256: 76b345bc14737f796a7674888b54021fac8e883950f94bfed2cfe23eb70d100e -->

Require request memory within workflow allocation and resolved output inside the owned stage generation. Return publish_twofluid_relative_product, executing finite-epoch two-fluid grid evolution from the retained seed fields. The producer preserves the reference/target scale factors and physical array units.

## core/initial_conditions.py::twofluid_relative_main

<!-- implementation-sha256: 352e355bd19d343164af281bbe83a2c57ec470e12c80cc6da438b5f44db2d2fd -->

Read positional parameters JSON, default request_base to that file’s parent, publish finite-epoch two-fluid relative-mode grids to required --output and return zero. Explicit request_base is retained. This runs the declared perturbative grid producer rather than a native particle simulation.

## core/initial_conditions.py::apply_native_wdm_thermal

<!-- implementation-sha256: 6a98ac359c30eb74da18672e90bd4ad5f7817e5315b65e4031273f86266eaedf -->

Require an explicitly enabled thermal-relic WDM model and unsigned 64-bit random seed. Through the transactional IC transformer, require one complete native IC, measure its mass-weighted thermal bulk and write exact-ID-keyed Fermi-Dirac kicks with that bulk removed; coarse combined baryon/relic particles use the relic mass fraction. Retain the IC/algorithm-bound native_wdm sidecar and return full verification, which recomputes kicks, inventory and momentum closure. This modifies IC velocities with the physical sampler; coarse phase space and later target resolution remain unresolved and science NOT_ACCEPTED.

## core/initial_conditions.py::twofluid_native_family_runner

<!-- implementation-sha256: 2103b3030120c289a47e2dd2f23e82cb8987f6e077636cd96c6651a51cbba178 -->

Require the grid-family request memory within workflow allocation and destination inside the owned generation. Return publish_twofluid_native_family, executing the requested reference-grid or independent native timestep/force-family trend measurement. The adapter does not promote a family trend into general convergence acceptance.

## core/initial_conditions.py::twofluid_native_family_main

<!-- implementation-sha256: 4e6dced18a4958e62efd8d2740fcc2dc85bdefa7a78c7efa8f47cf391abca159 -->

Read positional parameters JSON and required --output, default request_base to the file parent and execute the native/reference grid-family producer. Return zero. The request selects which original family evidence is measured; this entry point does not itself launch a fresh native parameter sweep.

## core/initial_conditions.py::twofluid_native_comparison_runner

<!-- implementation-sha256: c806115f556b5f1bb3e14b91665089fec5a3cc3af0acd5bf25f6772447603e98 -->

Require native-comparison request memory within workflow allocation and output inside the owned generation. Return publish_twofluid_native_comparison on original parameters, measuring evolved native cell fields against the retained nonlinear reference. This executes comparison rather than native evolution.

## core/initial_conditions.py::twofluid_native_comparison_main

<!-- implementation-sha256: ef67ec2e2f307f7b9f13845f9b7442bb5502b78456ebf12a95d13647433111ae -->

Read positional parameters JSON, default request_base to its parent, publish the original evolved-native-versus-nonlinear-reference comparison to required --output and return zero. Explicit base and source references remain authoritative; the command does not replace unavailable native output with generated fixtures.

## core/initial_conditions.py::twofluid_refinement_runner

<!-- implementation-sha256: 4d99b4e25e9ec61761e379ef40d68f8dcd11e777898017936026cc2f93da8a27 -->

Require nonlinear-refinement request memory within workflow allocation and destination within the owned generation. Return publish_twofluid_refinement, executing the conservative adjacent-grid comparison on retained levels. The producer’s original support and frozen error tolerances govern the result.

## core/initial_conditions.py::twofluid_refinement_main

<!-- implementation-sha256: ba68d6ecf6b598639afe27c0c372171c31ec0bcf16ab43b267d92f3f0ab6219d -->

Read positional parameters JSON and required --output, default request_base beside the file and execute conservative adjacent nonlinear two-fluid grid refinement measurement. Return zero. This compares declared retained levels rather than constructing a new independently validated physical seed.

## core/initial_conditions.py::twofluid_refinement_campaign_runner

<!-- implementation-sha256: 50e61401cfc041ac16ee768d94157358224b5931c985f5222fcb97390b5fb6cf -->

Require exact MPI/OpenMP allocation and memory/disk/stage ceilings within the current workflow. Compile the requested nonlinear refinement campaign into an owned destination, then prefix child stage IDs, dependencies, output IDs and stage references with the parent stage ID. Write followups.json with result-stage mappings and return children plus design path/hash and planned results. This expands executable follow-up recipes; returning them does not mean the child calculations have run.

## core/initial_conditions.py::twofluid_refinement_campaign_main

<!-- implementation-sha256: a4c172ee9f2b227dac103fd3856d940b91725126bcc5c885cb73e8e16d6aacf3 -->

Read positional campaign request JSON and required --output, then compile with request_base fixed to the request file’s parent. Return zero after recipe publication. The compiler supports declared nonlinear grid, operative timestep, dissipation or supplied native hierarchy comparisons; this CLI prepares their workflow rather than executing it.

## core/initial_conditions.py::twofluid_nonlinear_runner

<!-- implementation-sha256: 2cbbc6e0d1314e362172a2311122317d4457fdaeb3257a455bb16baf3827e3df -->

Require nonlinear-reference memory within workflow allocation and destination inside the owned stage generation. Return publish_twofluid_nonlinear_product, executing the conservative nonlinear single-stream two-fluid reference producer. This grid reference is distinct from a self-gravitating native particle run.

## core/initial_conditions.py::twofluid_nonlinear_main

<!-- implementation-sha256: ca5d6a55600b43d59cffefa87c6756977b164012ce94df9978db32095ade4d1c -->

Read positional parameters JSON and required --output, default request_base to the parameters parent and execute the conservative nonlinear two-fluid grid producer. Return zero. The retained seed and request controls determine its scoped physical model; the CLI does not invent a native comparison or convergence result.

## core/initial_conditions.py::twofluid_particle_runner

<!-- implementation-sha256: 64fd89dc577ef6c86dadb831e978295152858491488ab359d087693d78b58d2e -->

Require particle-projection request memory within workflow allocation and output inside the owned generation. Return publish_twofluid_particle_projection, evaluating retained finite-epoch species fields on the actual native q-load and exact identities. Projection does not itself write the final transformed GADGET IC; that is a separate producer.

## core/initial_conditions.py::twofluid_particle_main

<!-- implementation-sha256: 4d54da3ddc6ff76201768eb0d4c2d27ca97d869c95425f7fb442f28d53b2d50e -->

Parse required --parameters JSON and --output, default request_base to the parameters parent and publish native-q/ID projection using the request’s uniform-point or finite-volume material-cell mode. Normal completion returns None. This preserves actual native load identity rather than substituting an arbitrary particle sample.

## core/initial_conditions.py::twofluid_projection_refinement_runner

<!-- implementation-sha256: 553f3fe1aea8be2465c2b960f583463850c2d33f9e3fea4ef73ed11f2960a27e -->

Require projection-refinement request memory within workflow allocation and destination inside the owned generation. Return publish_twofluid_projection_refinement, executing the retained projection comparison with its declared identity/hierarchy/overlap semantics. Output support and acceptance remain those of the original producer.

## core/initial_conditions.py::twofluid_projection_refinement_main

<!-- implementation-sha256: 946628bff23633d9245461652b158ece3f1ecfd308536ec363f0f4f912af7af4 -->

Read positional parameters JSON and required --output, default request_base to the file parent and measure finite-volume native projection refinement. The request selects exact-ID, nested hierarchy or retained-flow overlap semantics; return zero after publication. This comparison does not independently certify the underlying species-flow model.

## core/initial_conditions.py::publish_twofluid_nonlinear_product

<!-- implementation-sha256: 4bd6e95349eaa2338d50262e882b5dc026f0d0699780765f942c98c192b607d7 -->

Read the sealed native Eulerian seed under one-quarter memory and require exact CDM/baryon order, cell-average convention, reference epoch, box, baryon fraction, h and matter density. Verify dimensionless density and km/s peculiar velocity units, execute the nonlinear reference solver and retain original seed fields plus native seed provenance. Publish numeric arrays at the target scale factor with physical units, source hashes and periodic Eulerian frame; return the product. This executes a scoped single-stream fluid reference, not native particle evolution.

## core/initial_conditions.py::validate_twofluid_native_timestep_campaign_request

<!-- implementation-sha256: b35d1c74d834b38bc088ae8d6ee0265781c6b15489f3c355293625167635a80d -->

Validate the versioned native timestep, tree-force, uniform-softening, PM-grid or artificial-viscosity launch campaign. Require at least three operative control levels: decreasing positive caps/coefficients, increasing power-of-two PM grids, or decreasing paired softening limits at a fixed turnover epoch. Require a pinned complete IC set, original snapshot-zero seed, native/reference/family controls and explicit wall/disk resources. Reuse their validators and bound retained memory, disk, stage count and follow-up depth; return the request. Actual control execution and logged histories are checked later.

## core/initial_conditions.py::twofluid_native_timestep_campaign_input_bindings

<!-- implementation-sha256: c8716a52ba3f5f9f2ba4b0ff4a4cfc25ed5f3dfd082eb1948381f95bf9579267 -->

Validate the native launch family, resolve answer/IC/generation references relative to request_base and require their exact hashes. Resolve the actual complete GADGET IC shard set and require equality to the declared files, binding native publication markers as well. Bound answer/generation metadata and add the relevant public route, workflow, unit and source-layout implementation hashes. Return sources before launch compilation.

## core/initial_conditions.py::compile_twofluid_native_timestep_campaign

<!-- implementation-sha256: 5e79e17e47b847a9b758673888d2e36372205b2eff4d346004c2f93d455c7bd1 -->

Require a fresh design destination and verified launch assets. Clone the common native answer template per control level, validate the public route preserves the requested operative setting, and construct independent GADGET launches with the same pinned IC generation. Add actual initial/target cell measurements, per-run nonlinear references/comparisons and target-wise family stages whose epochs bind runtime measured fields. Validate the DAG, recheck sources and write workflow.json/design.json. Return NOT_EXECUTED design metadata; compilation does not launch native processes or establish physical acceptance.

## core/initial_conditions.py::validate_twofluid_pressure_history_request

<!-- implementation-sha256: 0f654cfde029d4ce64b41b4f32599ee6caeca41046798c86925424c381c391fc -->

Require the exact saved-pressure-history schema with two or more bounded comparison references, explicit epoch/grid ceilings and finite positive memory. Validate each product reference and reject retained-mask workspace exceeding allocation. Return the request without accepting pressure support at any epoch.

## core/initial_conditions.py::publish_twofluid_pressure_history

<!-- implementation-sha256: 5c3c10fe909fe459172dc993d29d5bb44e62a75c1abb887a8bcfbe4e782042f0 -->

Load each completed native/reference comparison and its original initial, native and nonlinear products. Require strictly increasing actual epochs, unique snapshot labels, fixed run/seed/grid/pressure/reference definitions and frozen limits. Retain exact initial/target/combined pressure masks, replay their conjunction and pressure-model binding, and mark an epoch supported only when all pressure masks and the original comparison status pass. Publish saved-epoch arrays and lineage with science NOT_ESTABLISHED. This measures saved support only, excluding unsaved thermodynamics, pressure-work/source-energy closure, SPH force and shock acceptance.

## core/initial_conditions.py::validate_twofluid_native_family_request

<!-- implementation-sha256: ec9fe12978fe6bf4e23dd89ad87c28ce80ee9edeb6ceb802cf9e4e4a72f053b8 -->

Dispatch pressure-history requests to their validator; otherwise validate the versioned grid, native timestep, force or hydro comparison family. Require at least three product levels, finite dimensional trend floors, adjacent-error ratio at most one and bounded grid/memory work. Native-control versions require the matching refinement axis, supported parameter and bounded timebin logs/records/IC shards. Return the request; declared levels alone do not prove independent launches or operative refinement.

## core/initial_conditions.py::twofluid_native_family_input_bindings

<!-- implementation-sha256: 18272b15122d7b8f613ebf68988393162e10566bd20f98ad3a2b663bc8d826e3 -->

Validate the family and load source pins from every retained comparison. For native timestep/force/hydro versions also load original initial/evolved measurements and bind their actual launch/timebin evidence through the shared native evidence reader. Return the merged source map. This gathers dependencies without measuring the family trend.

## core/initial_conditions.py::publish_twofluid_native_family

<!-- implementation-sha256: 732e03ddefdc8cc0a6e41feee6f7d975c256546cc36f3bdd8b11b700df61408b -->

Verify distinct completed comparisons under a fixed physical/reference/limit definition. Grid mode requires increasing matched native/reference grids; native-control modes require identical launch IC bytes and initial fields, fixed grids and actual timebin evidence. Enforce operative decreasing timesteps or tightened force/hydro controls, keeping unrelated realized histories fixed where required. Reduce complete species density/momentum/velocity and relative-velocity errors to mean/max vectors; require original support and adjacent errors below ratio-times-previous plus dimensional floors. Publish arrays, masks, control histories and scoped status. Reference-grid, softening, PM and viscosity sensitivities retain separate limits and do not establish continuum or universal native convergence.

## core/initial_conditions.py::validate_twofluid_native_campaign_request

<!-- implementation-sha256: 44fc499ed889b16060bf33fe4543b5a3d9639b84ee99d665234e9888ef8af447 -->

Require an original completed-native seed template, increasing bounded comparison grids and distinct target snapshots separate from the seed. Validate dimensional native/reference comparison limits, reference schema and optional grid-family/pressure-history requests. Require pressure measurements when selected and reserve generated source references for the compiler. Bound per-stage memory plus aggregate retained disk, stage count and follow-up depth; return the request. This validates a retained-run comparison design without evolving a new native run.

## core/initial_conditions.py::validate_twofluid_native_comparison_request

<!-- implementation-sha256: 0546595e9ee37cfb686f105247f5bcc6e43e10741079187461de7f80baceae4f -->

Require exact initial-seed/reference/native-measurement references and a bounded full-grid workspace. Validate finite nonnegative density, momentum, velocity, relative-velocity and dispersion budgets, positive memory and a species-mass relative tolerance strictly between zero and one. Optional pressure controls retain dimensional coefficient/spread/acceleration limits and Boolean energy/expansion selectors; expansion work requires energy diagnostics. Return the request; source physics/epochs and measured masks are checked by the producer.

## core/initial_conditions.py::twofluid_native_comparison_input_bindings

<!-- implementation-sha256: fbe92c7b64218f6a97904e975c7edca3759ed0701766bd330287cb230f9e2533 -->

Validate the comparison request and load the sealed initial seed, nonlinear reference and evolved native measurement relative to request_base, each within one-quarter memory. Return their merged source pins without computing residuals or accepting the comparison.

## core/initial_conditions.py::validate_twofluid_refinement_request

<!-- implementation-sha256: 3514be705aa5a26d2eb34e88d61eb7314b06a987d57ca846a564334e0e3e3e57 -->

Validate versioned nonlinear grid, timestep or dissipation refinement with at least three actual product levels and bounded grid work. Require finite dimensional/nonnegative error budgets, positive normalization floors and an adjacent-error ratio at most one. Material versions additionally require bounded sampling chunks and explicit displacement, velocity, deformation and Jacobian tolerances. Return the request after reserving half the allocation for complete adjacent products/restriction; actual operative levels are checked during measurement.

## core/initial_conditions.py::twofluid_refinement_input_bindings

<!-- implementation-sha256: d06259d5b340888b6580ce8f607b2d52cf7006beab325d33c63ece44be8f7476 -->

Validate the nonlinear refinement request, load each retained level relative to request_base under one-quarter memory and return merged source hashes. This intake does not restrict grids, compare material maps or compute convergence errors.

## core/initial_conditions.py::publish_twofluid_refinement

<!-- implementation-sha256: 971828983c80dd0a4e135e3caf0f6b65920204aa9283a3817a0fbde5a68500f8 -->

Verify distinct original nonlinear products with fixed native seed, physical model, epoch and cell convention. Compare adjacent species density and density-weighted momentum conservatively, reconstructing velocity after cell-volume restriction; same-grid controls retain the same seed and require operative step or dissipation changes. Require initial restricted mass/momentum agreement, then evaluate normalized density/momentum and dimensional relative-velocity errors against frozen limits and nonincreasing ratios. Material versions also sample unwrapped displacement, velocity, deformation and Jacobian differences at coarse labels. Dissipation mode verifies identical accepted step histories and actual flux ledgers. Publish arrays/masks and scoped status, without continuum, native-evolution or general physical acceptance.

## core/initial_conditions.py::validate_twofluid_hierarchy_campaign_request

<!-- implementation-sha256: 1476dd5009d4c7e726340c7ac5cd04e8506ac7813debdce0c237c9e51b686c90 -->

Require one fixed material-flow reference and at least three distinct pinned native loads with explicit namespaces. Validate finite resources, stage/follow-up bounds and a frozen finite-volume projection template whose flow/load/namespace fields belong to the campaign. Validate hierarchy refinement limits using generated placeholders and require projection/comparison memory within allocation. Return the request without generating loads or proving geometric hierarchy support.

## core/initial_conditions.py::twofluid_hierarchy_campaign_input_bindings

<!-- implementation-sha256: eb815a141e3a6a238b5f45d573bc937a694a8e16ae838b10b16d012e2b910110 -->

Validate the hierarchy campaign and, for each supplied existing native load, build the corresponding projection request against the common flow. Verify it through twofluid_particle_input_bindings and merge sources; unresolved future load paths are rejected. Return pins without projecting any particles.

## core/initial_conditions.py::compile_twofluid_hierarchy_campaign

<!-- implementation-sha256: 5041ccaa9e0439db4b785f2511e95b11a01a8a0b62d51bcd80bf5a5db0fa2d4c -->

Verify existing original loads and the common sealed material flow, resolving their paths relative to request_base. Construct one finite-volume particle-projection stage per load plus a hierarchy-refinement stage bound to their generated path/hash references. Validate the DAG and write workflow.json/design.json in a fresh destination. Return PLANNED_NOT_EXECUTED with source pins and result-stage identity. Geometry support is evaluated by the actual comparison; no native generation or evolution runs here.

## core/initial_conditions.py::validate_twofluid_refinement_campaign_request

<!-- implementation-sha256: 8c6b70d739ceca00c89972394ce777150fb1c339fc6b10343285c6bf0a6645be -->

Dispatch native-control, retained-native comparison and hierarchy schemas to their specific validators. Otherwise require at least three bounded grid, timestep or Rusanov-dissipation levels, explicit resources and native seed/nonlinear/refinement templates with campaign-owned level references. Enforce matching comparison-axis/version semantics and stage memory/count limits; optional finite-volume projection templates must retain a fixed original native load and matching refinement axis. Return the request without compiling or executing stages.

## core/initial_conditions.py::twofluid_refinement_campaign_input_bindings

<!-- implementation-sha256: c31acf93974f6e2be3c335b697f9c3d6ac07b2e5ebc1651b4eae699db3bf433a -->

Dispatch native-control campaigns to launch-asset intake and hierarchy campaigns to original-load intake. Other campaigns resolve the original native run/receipt beside request_base and bind the seed generation; projected variants additionally require the fixed existing native-load path/hash. Return source pins without materializing the workflow or accepting its numerical controls.

## core/initial_conditions.py::compile_twofluid_refinement_campaign

<!-- implementation-sha256: 783003cb1832df80f5b0da1feda9d4d2fd11dbd4e50781e7c9f7d5ca936199b2 -->

Dispatch specialized native launch, retained-native comparison and hierarchy designs to their compilers. For nonlinear grid/timestep/dissipation designs, pin the actual original seed, construct seed/evolution levels, optional same-load finite-volume projections and their final refinement stages. Same-grid controls reuse one seed and vary only the requested step cap or dissipation multiplier. Validate the DAG and publish workflow/design in a fresh destination with planned result stages and source pins. Return PLANNED_NOT_EXECUTED; child calculations and native/physical acceptance remain pending.

## core/initial_conditions.py::validate_twofluid_particle_request

<!-- implementation-sha256: fc605aebe7212038542c7be19e1203bceb38efdba83fc4eddb203af939065eeb -->

Require the versioned native-q projection schema, explicit finite-volume mode/quadrature bound for v2, positive exact particle/chunk limits and a nonempty original namespace. Validate positive finite memory, displacement/velocity/material-weight tolerances and a mass-rescaling fraction below one. Return the request. Native load identity, actual retained flow and material support are checked during intake/projection.

## core/initial_conditions.py::twofluid_particle_input_bindings

<!-- implementation-sha256: fa16f24ed6b7ac601718056018a43f313359376f97ec6e49397182ee76078ab3 -->

Validate the projection request and read the evolved CDM/baryon fields from a supported sealed relative/nonlinear schema. Verify material field structure; full nonlinear maps require v2 finite-volume cells. Resolve the original native load relative to request_base, require its expected bytes and add it to the source pins. Return sources without interpolating or integrating particle projections.

## core/initial_conditions.py::validate_twofluid_projection_refinement_request

<!-- implementation-sha256: 6d48d980697cf2f4c1bb374518cc043f33b99bc9ff0c6e59e4c584bfb7a3604e -->

Require a complete versioned finite-volume projection comparison and a supported grid/timestep/hierarchy axis. Validate at least three exact product references, particle/level/chunk bounds, version-specific matching/overlap quadrature limits and finite dimensional error budgets. Require positive mass/velocity floors, adjacent ratio at most one and enough memory for adjacent products/sorting. Return the request; actual shared-ID, nested or overlap semantics are verified by the producer.

## core/initial_conditions.py::twofluid_projection_refinement_input_bindings

<!-- implementation-sha256: 77ba460cff13ed3d2eebf4f7195071ef712b9d4a527601e4d3e8fb74d593b2da -->

Validate the projection refinement request and load each sealed projection level plus its referenced original relative-flow product under bounded memory. Return merged source hashes, discarding loaded arrays between levels. This binds the comparison’s actual projection/flow lineage without computing restriction or overlap errors.

## core/research/findings.py::FindingStore

<!-- implementation-sha256: 249930d843e7a4236bfe649295a6b01b3ef27257fb3d8f805dbde7298c6c0eb6 -->

Maintain optional immutable, hash-chained failure/fix/closure events under a directory lock. Record artifact, trigger, severity, requirement, reproduction, exact sources and affected products; expand downstream dependencies without narrowing recorded scope. A fix remains FIXED_PENDING_REVALIDATION until current passing receipts cover the complete affected closure and every frozen measured condition. Recheck closure freshness against source/payload hashes, support explicit reopening and list unresolved critical findings. This store tracks supplied evidence; ordinary simulation/analysis does not create it automatically, and CLOSED does not certify an entire physical model.

## core/research/findings.py::validate_repair_parameters

<!-- implementation-sha256: 8c203dbfbbb030341d358257eaf0dab91fa774f268b34f8cd9e8986ecf730c95 -->

Require safe finding/product IDs, exact current fix-event and dependency-graph hashes, positive integer metadata memory and a valid frozen revalidation requirement. Require one supplied measurement reference per declared measurement in the same order and return parameters. Source freshness, producer identity, actual rerun time and predicate outcomes are checked by the receipt producer.

## core/research/findings.py::repair_receipt_runner

<!-- implementation-sha256: ab8e011779a8275efa5bbf8d595f4551eb377a3c49df51c2e5e399ecb6b7fdfb -->

Require the repair destination inside the owned scientific workflow generation and metadata bytes within stage memory. Publish the fix-bound receipt from actual retained rerun measurements, then return total/passed condition counts and explicit scientific exclusions when any frozen condition fails. It does not close the finding or infer a pass merely from stage completion.

## core/research/findings.py::repair_receipt_main

<!-- implementation-sha256: 18a4fe307a1e0f23702ee181a9eae5fcbb7df519a129d32fe16d7d68fe52bb56 -->

Parse required --request and --output, bound request JSON to 16 MiB and set request_base to the request file parent. Publish the scoped receipt, print it and return zero for PASS or one for failed measured predicates. Invalid/stale evidence raises; successful publication is separate from finding closure.

## core/research/findings.py::findings_main

<!-- implementation-sha256: 93f33f4765b558b0e3627f42a410c4019b03ce1f125b79700bb57ea7159e9055 -->

Parse record, fix, close, reopen, list or revalidate with required --store and a JSON --request for every action except list. Dispatch to FindingStore and print the result; revalidate defaults request_base beside the request and executes the required workflows. Normal completion returns None. Closure uses complete current receipt checks rather than a manually assigned success flag.

## core/research/restart_campaign.py::lifecycle_tolerance_catalog

<!-- implementation-sha256: 1f70560addab640039fefda895eb37d79092200145b2bf34760afe08e60be0b9 -->

From compiled STARFORMATION, STELLAR_EVOLUTION and BLACK_HOLES features, enumerate active lifecycle journal families and floating commit/event fields. Include original BH native columns and their public physical-unit counterparts while excluding exact integer fields. Return family names and zero absolute/relative tolerances for every active float field. This authors strict restart reproducibility controls, not physical conservation budgets.

## core/research/restart_campaign.py::validate_campaign_restart_comparison

<!-- implementation-sha256: e4c3787f223a2b16c4deb09e968711463e6077545f40cfe8c30bdd62fa86695e -->

Require an exact nonnegative saved snapshot, finite positive comparison memory and distinct uninterrupted/resumed run arguments. Return None. Actual completed runs, identical inputs, checkpoint ownership and full selected state are verified by the runner.

## core/research/restart_campaign.py::campaign_restart_comparison_runner

<!-- implementation-sha256: 4f0654da4581d1cedf034fd054554f82c649a12ec933cb476beeddfb5842fb44 -->

Verify the frozen bounded comparison template and completed native runs: the reference has no resume attempts, while the candidate has an actual completed resume. Require matching binary, ICs, compiled options, schedules and physical/numerical parameters, allowing only declared pause and path/resource bookkeeping differences. Verify the resume invocation binds a real checkpoint marker and every interrupted checkpoint payload is retained in its hash-bound archive. Rebind actual run/journal locations, prepare the complete typed comparison, execute compare_native_restart and recheck sources. Write restart-comparison.json; a non-PASS result raises with measurements retained. Numerical restart agreement remains separate from scientific accuracy.

## core/research/restart_campaign.py::validate_restart_exercise

<!-- implementation-sha256: 0e9622e7cce7f5edc1941ff96f7578072f9aa23f02ebc50cf8d2b32f609a0ef4 -->

Require a positive integer pause_step no greater than one million, exact nonnegative snapshot and finite positive memory/wall limits. Return None. This checks declared exercise bounds without creating a checkpoint or admitting any existing restart files.

## core/research/restart_campaign.py::restart_exercise_runner

<!-- implementation-sha256: d605c5f791ed513485a8e233ea78efb1ed4abc92c637876d5988ef33b3af5400 -->

Validate bounded answers/template inputs and freeze the complete comparison before either branch runs. Preserve original physics while enabling source capsules and setting reference pause zero versus a declared positive candidate pause. Execute the uninterrupted native branch, require the candidate to stop after publishing a verified checkpoint strictly inside its timeline, then resume through the registered owned checkpoint policy. Recheck original inputs and run the complete campaign restart comparison; return its measured result. A missing/interior-invalid checkpoint or unexpected completed candidate fails rather than simulating a restart.

## core/research_inspector.py::install_inspector

<!-- implementation-sha256: 894d1fc21b6c08ae39857906ffbf96a7c377d785bd0d367ae17ccb55218e9cf5 -->

Copy this module’s source plus register() into the Blender text block research_object_inspector.py, mark it as an executable module and register the inspector in the current Blender process. Reuse/replace that text block by name. Return None. The saved scene can reload the embedded inspector without scientific Python packages; this function does not load measurements or render a scene.

## core/research_inspector.py::register

<!-- implementation-sha256: e14ab90ff0be9d19a9905ab1d7f7c4f37230b30af14fc05d5e074fab2e9452f5 -->

Register or replace the Blender Research sidebar panel and operators for measured epochs, exact identities and source links. Show saved clock/units, support, lifecycle detections/events, geometry layers, parents and SIDM macro diagnostics without interpreting catalogue absence as physical death. Restrict selection to the scene owner and exact identity; source opening requires an allowed package-contained link and matching streamed SHA-256. Return None. This UI exposes retained measurement provenance and symbolic geometry; it does not calculate new dynamics.

## core/suite_comparison.py::compare_suite

<!-- implementation-sha256: 7551f8036381bc3620394c5c6dd51ef39b0c24d9300c9e6f560b59132e9a79c7 -->

Lock at least two distinct runs and verify one source-bound analysis bundle per run. Select the lowest-redshift completed HMF, power or correlation epoch common to all runs within tolerance; require common HMF mass definitions and saved array units. Plot each run’s own finite bins without interpolation, retain signed correlation values and export comparison.png/PDF, curves.csv and a hashed comparison.json into a fresh publication. Reverify sources before publishing and return the PNG path. This descriptive overlay does not infer matched phases, common binning, causal parameter effects or convergence.

## core/stars.py::verify_native_ssp_birth_audit

<!-- implementation-sha256: 087d28f0ea9b5df8ee83942f8b5b81d3cbb476940594cc38085137d51051b4b9 -->

Verify the sealed numeric product has schema native-ssp-birth-audit-v1, rehash every retained source and validate its run-bound product provenance. Return the audit report. Integrity verification preserves the audit’s actual decisions; it does not independently prove the stochastic birth prescription or convergence.

## core/stars.py::verify_native_ssp_birth_events

<!-- implementation-sha256: d5164d6a1bd9c959c0b238c0e9d5c4203060898e2fa9a3b8fc920091d5dd888d -->

Verify the original numeric audit and native-ssp-birth-events-audit-v1 schema, rehash recorded sources and require current run provenance. Return the event audit. This validates the retained event publication rather than inferring births from snapshot particle-count changes.

## core/stars.py::verify_native_hydro_history

<!-- implementation-sha256: 5156d33fa0c71dffc5710639831617e22221f3358cb6ae1b08dfd14a14f7c608 -->

Verify the sealed hydro-history product, original source hashes and run provenance, requiring a complete epoch/interval sequence. Read retained census/event arrays and replay every interval’s mass/momentum/species accounting, original committed stellar-return/BH rows, BH merger lineage and detailed stellar-return state reconciliation. Require exact dtype/shape/value agreement for saved residuals and decisions, matching snapshot endpoints and consistent aggregate PASS/FAIL with physics NOT_ACCEPTED. Return the verified report. This replays the stated accounting contract; it does not turn accounting consistency into full physical validation.

## core/stars.py::exact_ids

<!-- implementation-sha256: 588aaa682cccea5e9409de7b5cbfbf7dcf61f1504b274fbe7152ed862804468c -->

Return exact_uint64_ids with allow_empty=False, rejecting an empty or invalid identity selection under the shared canonical ID contract. This preserves exact integer identities rather than converting through floating-point particle labels; it reads no native files.

## core/stars.py::history_figure

<!-- implementation-sha256: 079664b26e166b3eb910092b2e0aa0e44e50a24be6d1cc7ea3b0b6e20fd5301f -->

Verify history.hdf5 dataset payloads and export a four-panel saved stellar-history PNG/PDF. Plot selected-star presence fraction and largest epoch-local bound-subhalo overlap against scale factor, plus first/last selected XY positions in proper kpc after epoch-local periodic centering. Retain epoch/data/source metadata and explicit synthetic-fixture labeling. Return None. The spatial panels use a display frame, not temporal winding or a demonstrated tidal-origin history.

## core/stars.py::candidate_morphology

<!-- implementation-sha256: 50c9e4b3bdfcd29cf508a2b5fc765a4bffad2c354e88b9e16784fbede017f2fd -->

Require at least three saved members, finite phase space and positive masses. Form a minimum-image chart anchored on the first member; return UNAVAILABLE if its span makes the periodic frame ambiguous. Compute mass-weighted centroid, proper-position covariance eigenvectors/RMS axes, b/a and c/a, and velocity population covariance/RMS. Return the measurement dictionary with comoving-kpc center, proper-kpc extents and km/s velocities. RMS axes are covariance scales, not full lengths or calibrated stream widths, and morphology alone does not establish a tidal stream.

## core/stars.py::add_stream_diagnostics

<!-- implementation-sha256: 94b577db8e265ffc8950c0624bfbf87813b63ed30fffff6c7ed5ab0f743ff9e3 -->

Read exact selected IDs/candidate labels and each saved history epoch, bounding catalogue, tree and summary workspace. Compute candidate morphology, bound-subhalo overlap counts and nearest saved nonmember-centre context. For adjacent epochs write contracted HDF5 transition masks and selected-star overlap links using aligned identities, retaining the actual saved-time bracket. Store StreamDiagnosticsJSON in the open output and return the diagnostics. These links concern selected stars rather than complete progenitor ancestry; centre proximity is not an impact parameter or an inferred between-output flyby.

## core/stream/measurements.py::validate_stream_measurement_template

<!-- implementation-sha256: 9d0146c5a0faa9e1eeef81b05a835edf51da9c0eb506f76f879e1eb87b92a043 -->

Require the exact native catalogue measurement template and complete intrinsic wrap/curve/bin/observable/bootstrap controls. Validate explicit figure selection and dpi 32–2400; optional geometry, observer and actions must be complete dictionaries or null. Validate an observer frame when supplied and require actions to use a source-bound measured coefficient/time-dependent host potential. Return the template; missing optional inputs remain unavailable rather than replaced by assumed fields.

## core/stream/measurements.py::validate_catalogue_stream_measurements

<!-- implementation-sha256: 1ee01ebfdbdd11b78a25a755feb6f0a6b2f2a2e7d3204dd70ec65ff0ba93bf21 -->

Require positive exact maximum_particles and maximum_candidates, plus finite non-Boolean memory of at least 0.1 GiB. Return None. This checks intake bounds; native catalogue lineage and calibrated selection/ancestry support are verified by the runner.

## core/stream/measurements.py::catalogue_stream_measurements_runner

<!-- implementation-sha256: 1778fedf39c921deeca98e96b9c475115d028f311ba55681e8bdbaf8cbf811ad -->

Verify the frozen measurement template and original native tidal catalogue, bounding particle/candidate counts and reusing actual native aperture intake. Publish a selection-context sidecar for every candidate, with optional exact-ID birth context; unsupported classification/ancestry selections remain unresolved. For supported candidates publish progenitor views and execute original per-wrap coordinate, density/width/velocity/power and optional geometry/observer/action producers. Retain every wrap/view mask and explicit missing-input/error reasons; figures are exported only where supported and selected. Rehash sources and write stream-measurements.json. Morphology and support do not establish a causal gap or population acceptance.

## core/stream/measurements.py::publish_native_stream_birth_context

<!-- implementation-sha256: 91249d6a96ac07b6e792f4a97150578de66bf88099ffab5fd6d7068be54bf650 -->

Verify passing source-certified hydro accounting for the same run, particle namespace and exact saved snapshot under bounded metadata/array intake. Join selected distinct uint64 stellar IDs to the actual census and require original birth-field dtype, alignment, physical units and parent namespace. Preserve version-zero fields behind an unavailable mask, reject unknown versions or future births and publish the numeric context with source hashes. Return report and selected arrays. No second unit conversion, age/metallicity imputation or inferred stripping time is applied.

## core/stream/perturbers.py::validate_perturber_history

<!-- implementation-sha256: 8bc7b851f577646d8744250ab3429bbccc70aa4e8b6b856a997e1c5f34caa609 -->

Require positive exact particle/node/track ceilings and finite memory of at least 0.1 GiB. Return None. Actual graph, physical chart, bound membership, recorded image and interval support are verified by the history reader/runner.

## core/stream/perturbers.py::native_perturber_history_runner

<!-- implementation-sha256: 74b426efbdb8de844b57dc6d5359e09f2066e8718b3f05d6c609938b67e17b89 -->

Verify the frozen selected-track template, original identity graph and common saved stream frame. At each supported native observation require detected bound membership, exact graph/snapshot ownership and physical phase coordinates; sum actual selected member masses and enclosed profiles in proper kpc/Msun. Use recorded periodic images and proper position-derivative velocities, retaining sparse unavailable observations. Intervals require both endpoints, unique native descendant continuity, caller support and bounded cadence. Publish perturber-history.json with original interpolation certificates and unresolved reasons. Equal catalogue rows, nearest images and assumed analytic profiles never establish continuity.

## core/stream/native_encounters.py::validate_encounter_display

<!-- implementation-sha256: c9dccae867998038eded5889e43be4c7320f071f7c58b2b62a98b31601134348 -->

Require positive exact maximum_objects, maximum_vertices and frame_stride plus finite non-Boolean memory of at least 0.1 GiB. Return None; source identity, clocks and geometry are checked during input intake and display assembly.

## core/stream/native_encounters.py::encounter_display_inputs

<!-- implementation-sha256: 624efae966d38a22bb02c2fee57b2e5ce7e7136724b67864d0fab280858c9592 -->

Validate display bounds and read bounded stream, perturber and encounter numeric products with their source hashes. Require both histories in native encounter-frame format with identical epochs/frame-definition hash, and require the encounter catalogue to bind the exact history bytes. Return report/array tuples and merged pins without creating display geometry.

## core/stream/native_encounters.py::encounter_display_input_bindings

<!-- implementation-sha256: 5a16b9e0eb6893122687b42f1e54a473a46ea1208d4a40be640b7bba31a1739d -->

Return the source-pin map from encounter_display_inputs after its bound, common-frame/epoch and exact catalogue-history checks. This dependency intake does not assemble scene assets or reconstruct additional events.

## core/stream/native_encounters.py::native_encounter_display_runner

<!-- implementation-sha256: ad607a32fdd699c0cf0844ce29f8324653bd2cf4f3281a5a256977bf76d56982 -->

Verify allocated memory and exact native histories, then publish identity-bearing point/two-endpoint line products at saved epochs in their proper-kpc chart. Preserve original observation/interval support, frame stride and track-anchor identity; assemble the sealed measured display without trajectory resampling. Separately publish reconstructed-impacts.json with original event times, positions and geometry masks, explicitly distinguishing them from saved snapshot centers. Write encounter-display.json and return its display/event references; this prepares scene inputs rather than rendering Blender or certifying event geometry.

## core/stream/native_encounters.py::validate_encounter_refinement

<!-- implementation-sha256: 80d2d02d65c5c04967c6895f15d97c4f62ff19539cf340fb44410fdcd5d5dc16 -->

Require exact positive ranks/threads, nonnegative branch ceiling, 2–1100 maximum outputs and finite positive memory/disk/wall/timestep limits. Require distinct explicit retained checkpoint-directory strings. Return None. This validates declared refinement bounds without admitting any checkpoint or launching a branch.

## core/stream/native_encounters.py::refinement_input_bindings

<!-- implementation-sha256: 44aaca850891ee99f48f0798989ef8d26ad3787dc3acde45f6e050e8dde78437 -->

Verify bounded original encounter/stream products and merge their numeric source pins. Bind the parent native parameter file, compiled options and executable. For every selected checkpoint directory require verified rank payload/checksum receipts and retain available epoch sidecar hashes. Return dependencies without preparing output times or executing refinement.

## core/stream/native_encounters.py::native_encounter_refinement_runner

<!-- implementation-sha256: 5b3e70f44289710c469815f010a0be3b87d2df4cb3b9c731b16b678c07373d2c -->

Bind encounter requests to the original native run and saved physical clock, requiring synchronized outputs and no timestep increase. For each original event bracket subtract background-clock uncertainty from the requested cadence, choose a verified earlier checkpoint and quantize a bounded output schedule. Reject unsupported clocks, insufficient checkpoint/cadence or exhausted branch/output limits with explicit decisions. Write eligible branch requests and refinement-plan.json after source rechecks; return RECIPES_PREPARED with the execution command. This plans branches only: finer output cadence itself does not certify trajectories or event geometry.

## core/stream/native_encounters.py::prepare_native_encounter_campaign

<!-- implementation-sha256: 11d6df1c09008ac7bfe282f0ce1aabfece618dbd2cdb8befe79945364bf45879 -->

Verify a completed self-gravitating type-4 native stream history, exact run, bounded original graph, pinned perturber template and explicit frame/worldline selections. Build recipes for periodic images where needed, common frame, bound perturber histories, encounter catalogue, registered figures and measured display; optionally add checkpoint-refinement planning. Compile dependency closure, bind retained/code hashes and write request/workflow/manifest in a fresh directory. Return RECIPES_PREPARED. It does not execute these stages or manufacture a matched causal response from encounter geometry.

## core/stream/native_encounters.py::validate_recovered_encounters

<!-- implementation-sha256: 89d7c704c6b9c547d54ee23ce66305679d4537f34f7abc2ee2a7660082ad37ad -->

Require finite non-Boolean recovered-encounter memory of at least 0.1 GiB and return None. The loader separately verifies template, original measurement epochs, exact cohorts and optional perturber assets.

## core/stream/native_encounters.py::validate_recovered_encounter_template

<!-- implementation-sha256: 9372822f6127aad864c50c596eb64af751f4a5d90fdeeb04a602a9239b8d8f92 -->

Require the exact recovered-encounter template, at least two distinct nonnegative snapshot labels and positive exact cohort/selected/census/node/track ceilings. Optional graph/perturber assets must be path/hash objects or null. Require a named fixed chart with finite per-snapshot origin/velocity vectors and positive interval limit in Myr. Return the template without inferring source trajectories or a continuous chart between outputs.

## core/stream/native_encounters.py::recovered_encounter_input_bindings

<!-- implementation-sha256: 096869e4191a8c74312e3f4d1bb4b683006b893dc85368f29bcaee607de79844 -->

Return the full pin map from the recovered-encounter loader. It verifies original epoch measurement reports, exact uint64 cohorts and namespace-qualified progenitor associations, graph/template inputs and interpolation certificate sources under memory bounds. This intake does not execute the encounter campaign or infer continuity from candidate row numbers.

## core/stream/native_encounters.py::recovered_native_encounters_runner

<!-- implementation-sha256: 4b49a706df441bea0988f3daee858a836b1095acf5592018d79631ea6b2c3482 -->

Verify the recovered measurement history and keep each complete supported reference-epoch stellar cohort with its exact per-ID progenitor association. Require exactly one later catalogue selection containing that unchanged cohort; later-added stars remain outside it. Missing/ambiguous continuity, perturber assets or recorded periodic images remain unresolved. For eligible cohorts publish native worldlines, prepare and execute the encounter workflow, then retain product/receipt references and recheck sources. Write encounter-history.json with support status. It does not fabricate interpolation certificates, infer temporal winding or establish causal attribution.

## core/sidm_forward.py::lensing_controls

<!-- implementation-sha256: da0977933afc212679444ac1d1ceb279adb4187c3b0795e42e73b20bc47e06e1 -->

Read hash-bound bounded native lens controls and validate positive source redshift/cylinder dimensions, pixel/count ceilings and coarse-mass fraction. Require a finite nondegenerate projection basis, explicit mass sheet and distinct supported observables with exact surface-density, angular or dimensionless units and positive dimensional floors. Return settings. These geometry/sampling controls do not fit an observed lens or estimate H0.

## core/sidm_forward.py::lensing_inputs

<!-- implementation-sha256: 97b32c4eb96909041f7d416a83ad2f25c6dda787610bce816ec630b62d555699 -->

Verify the completed recovered CDM/SIDM target results and pair, retaining unresolved statuses. For measured branches require contained hash-bound native histories and membership products with the original matching, saved epochs and actual collisionless/elastic-SIDM model. Verify the paired history binds these same targets and rehash all sources. Return settings, targets, pair, histories and pins without depositing a lens map.

## core/sidm_forward.py::native_recovered_lensing_runner

<!-- implementation-sha256: 9dca283b79185f1d546d7a719e6ab56d915a79335c314c69b45bfe6c1fc4b501 -->

Verify recovered targets under stage memory and require a common flat, original CLASS background and matching saved epochs. For each valid observer/lens/source ordering read complete species-resolved native apertures, compute physical distances and execute original thin-lens mass-map producers. Screen support using noncoarse counts and both cylinder/nearest-pixel and actual CIC-deposited coarse fractions. Publish supported planes, paired observational contrasts, registered figures and sidm-lensing-result.json. Unavailable pairs/geometry stay unresolved; H0 is NOT_ESTIMATED and no CMB/time-delay likelihood or lens posterior is generated.

## core/sidm_forward.py::stream_controls

<!-- implementation-sha256: 38c2712743a28eca85978fa50111f13d7c104148a055addaf9ece1887ec11f59 -->

Read hash-bound bounded native SIDM stream controls. Require positive candidate/pair/particle ceilings, exact shared-particle rules with runner-up fraction below retained fraction, increasing interior reference-track bin fractions and explicit measurement counts/resolution. Validate distinct line-density, centroid, width-covariance or velocity observables with their physical units and positive floors. Return settings without selecting or comparing stellar populations.

## core/sidm_forward.py::stream_inputs

<!-- implementation-sha256: dcc9da725600a4a36f22dd3da3f0269206bb1282cc230a4923ebe1b84e5f18df -->

Verify the completed recovered pair and its actual native matched histories, plus owned completed recovered-stream measurement receipts and contained products for both branches. Require distinct saved labels and current source hashes. Build an explicit bridge between original run-specific particle namespaces only through identical verified IC byte sets; subsequent stellar matching still requires equal recorded birth state. Return controls, histories/measurements, owning roots, bridge and pins without measuring a stellar contrast.

## core/sidm_forward.py::native_recovered_stream_response_runner

<!-- implementation-sha256: 02dcb8db33e3dd97ddefb18fbaf40400ac8b9ec8c0b3c7dc7b19ad3c30f73e6c -->

Verify paired recovered targets, exact owned stellar measurements and shared-IC namespace bridge within stage memory. Match supported wraps by mutual exact-ID overlap, require equal recorded birth states and use common initial locked masses as frozen weights. Freeze a contiguous supported CDM track and interior physical bins, measure both selected common cohorts and publish supported profile contrasts plus figures. Rehash sources and write sidm-stream-result.json; missing pair, cohort, birth or track support stays unresolved. This is a conditional common-cohort contrast at saved epochs, not a full-population or uniquely causal SIDM response.

## core/sidm_recovery.py::native_recovered_sidm_runner

<!-- implementation-sha256: bc76d49570cc62f23395c6dd45089ba307ee2d1edd936cb2276a4e4cd0da2662 -->

Verify actual recovered hosts, completed native lineage, physical controls, model and shared matching identity under stage allocation. With fewer than two accepted observations publish UNRESOLVED_TARGET_HISTORY. Otherwise measure complete native memberships at accepted saved hosts, publish the original SIDM/collisionless history and export its history/scattering figures as applicable. Retain missing targets, recheck sources and write sidm-history-result.json. Sparse observations do not establish unresolved collapse timing or scientific acceptance.

## core/sidm_recovery.py::recovered_sidm_pair_inputs

<!-- implementation-sha256: 3cee4c018233255ef0ec538328befdf62b5c231be339c1256bf872d4d68bd880 -->

Verify the original paired-control hash/schema and both completed recovered-target results. Measured histories must remain contained in their original generations and pass native source validation with actual collisionless versus elastic-SIDM model labels. Preserve unresolved branch histories and recheck sources. Return controls, results and pins without producing a matched contrast.

## core/sidm_recovery.py::native_recovered_sidm_pair_runner

<!-- implementation-sha256: 933cd0fdd678e0220d1c2aa56311156cf657569e08aad3ee86542a90d784fe21 -->

Check allocated memory and verified recovered branch histories. If either branch lacks a measured target history, write an explicit unresolved pair result. Otherwise execute the original matched estimator and native-source/actual-IC-phase-checked figure exporter, then write sidm-pair-result.json with matched product and variants. Return COMPLETE execution measurements while retaining science NOT_ACCEPTED; geometry or labels cannot substitute for the original matching gates.

## core/sidm_recovery.py::centroid_controls

<!-- implementation-sha256: 671813f428e285373ff04eaa74a855824b82d89a72127be051fc54c36011673d -->

Verify bounded pinned centroid/ancestry settings, positive graph/census ceilings and every consecutive saved snapshot in increasing order. Require explicit branch-length, physical cadence and transfer-margin controls, with clocks derived later from native outputs. Verify and validate the original structure template within memory and return settings. This authors selection/measurement controls without reconstructing ancestry.

## core/sidm_recovery.py::recovered_centroid_inputs

<!-- implementation-sha256: ee2ac8e3523d5caf847e663661ab36179f4ee5b5f966e24ea4b1b5337cf7b6c1 -->

Verify the completed original recovered target and pinned centroid/structure controls. For measured targets require contained source-verified history with the same native receipt, matching, actual model and membership epoch sequence; all selected epochs must appear in the controls. Preserve an unresolved target without supplying a replacement history. Rehash sources and return settings, target, optional history and pins.

## core/sidm_recovery.py::native_recovered_centroid_runner

<!-- implementation-sha256: 03b2160c19dea9bfd0adb0e2442c7d2e66f9383eefb0c1ab3463edcc65d1b114 -->

Require an actual MERGERTREE-enabled native run and build recorded FOF ancestry over consecutive saved outputs. Check that original recovered observations lie on the supported native main branch; absent/different ancestry remains unresolved rather than replaced by material recovery. Reconstruct the quality-masked branch with actual clocks, measure original halo structures and export centroid offsets with saved FOF membership-join events. Write sidm-centroid-result.json and return measurements. These joins are not continuous galaxy/subhalo coalescence times; source and convergence limits remain explicit.

## core/sidm_recovery.py::satellite_controls

<!-- implementation-sha256: f71d5f24f3b89fcd5a4b3f2c2f902db92ed9a2be2d18ed449eea8c580a4b876f -->

Verify the bounded pinned satellite template and exact census, material-retention, runner-up and shared-particle criteria. Require fine-DM-only binding controls, validate their pair-work/softening/gauge contract and require survival count/mass floors using the same potential gauge. Return controls without selecting a satellite or interpreting missing membership as disruption.

## core/sidm_recovery.py::satellite_inputs

<!-- implementation-sha256: 6c1707faf65092e2b597634d26c84ea5019369cc82162dc4087fd9d9f7da745f -->

Verify completed original target status, satellite controls and all retained sources. For measured targets require that the target pins include the actual completed native evolution, then verify each contained host membership product matches its exact saved host/snapshot. Return controls, target and pins. Unresolved histories remain available as explicit status, not fabricated satellite populations.

## core/sidm_recovery.py::accepted_satellite_match

<!-- implementation-sha256: 7f4df107a47d96b799bd923f50e06a30826505771a9f8121e7a26b58baafd3cc -->

Return False when no candidate subhalo row exists. Otherwise require the frozen shared-count/material-fraction/runner-up rules, a strict best overlap, noncentral rank, membership in the recovered host and zero coarse type-2 particles. Return the resulting Boolean. This accepts a same-run fixed-cohort satellite observation; it is not an independent merger-tree identity or survival measurement.

## core/sidm_recovery.py::native_recovered_satellite_runner

<!-- implementation-sha256: 9e8e858210e26136be6cfebd12d16802eaa6d09e0034970b84d525c7b436bb09 -->

Verify target/native sources and select the first supported clean noncentral anchor deterministically by largest bound fine-DM count, breaking ties by original row. Freeze its exact initial bound cohort and match that cohort at later saved hosts under the original retention/runner-up rules. Measure supported bound populations with native diagnostics and derive survival against the original anchor, exporting registered figures. Missing/ambiguous/other-host observations remain unknown, never zero retention. Rehash sources and write sidm-satellite-result.json. This measures fixed-cohort survival under declared binding controls, not collision causality or universal disruption.

## core/sidm_products.py::export_native_sidm_history

<!-- implementation-sha256: 4c6495ffbc5aa77f8e389aa647be942954e8bf37982b510086b4df178e309145 -->

Verify the original native history and export fixed-aperture central density, saved phase categories, DM/baryon unsigned axis misalignments and per-epoch shell density/variance/anisotropy, kinetic heat flux and kernel diagnostics. Publish numeric bridges with original physical units, exact support and unchanged classification; finite failing kernel diagnostics remain visible rather than relabeled as passing. Recheck sources and return an organized scientific gallery in a fresh directory. Synthetic origin and actual DM model stay labeled; no continuous collapse time or convergence acceptance is inferred.

## core/sidm_products.py::sidm_matched_inputs

<!-- implementation-sha256: 137c25616eabf1a8612f84bbff7690d37893f92b61c872d0e2201fc01b21de32 -->

Load a completed original matched product and bind every original history to its actual native run. Require a genuinely collisionless baseline and elastic-SIDM comparison branches, and measure their actual original IC phases within memory bounds. Replay the original matched producer in temporary storage, requiring identical comparisons/matching metadata and all array dtypes, shapes and values. Return report, arrays, pins, phase measurements and histories. Sealed labels alone cannot substitute for native-source and numerical replay.

## core/sidm_products.py::export_native_sidm_matched

<!-- implementation-sha256: 0d7d4eefaaec3b42f58624e034f79854b461270a6e37d9244a888c9024d74bdb -->

Verify and replay the original matched product, require a fresh output directory and export each central-history or radial-shell observable. Retain signed branch-minus-baseline differences and the original floor-scaled signed difference, not an ordinary ratio, with exact paired support and actual IC phase evidence. Publish numeric bridges and two-panel galleries, rechecking sources. Return the gallery manifest; no independent-realization covariance or new uncertainty bars are inferred.

## core/sidm_products.py::extended_inputs

<!-- implementation-sha256: b50a50c128b80abf8295ff62d054d1495883af361915bca66c0a04ee35ed458e -->

Validate the exact satellite/degeneracy/observational export request and original completed numeric producer/source binding. Require every supplied native run and receipt to be owned by the product; observational comparisons additionally need a collisionless baseline, enabled SIDM branches and passing actual IC phase measurements. Replay the original calibrated derived/observational producer and require identical arrays, units, namespaces and result metadata. Return report, arrays, pins and phase evidence without rendering.

## core/sidm_products.py::export_native_sidm_extended

<!-- implementation-sha256: eddc3be5be132f9faa2e51427aa7b5a313086c7bc8aa6d13f77051cd80b4184e -->

Verify/replay the original extended product and require a fresh destination. Export fixed-cohort satellite retention/current bound mass, calibrated scaled-parameter SVD values/modes, or original supported stellar/lens/H0-estimator signed contrasts in retained bin order. Preserve unresolved modes, masks, physical units, floor scaling, calibration and actual phase evidence in numeric bridges. Recheck sources and return organized figures. This is display adaptation of original scoped results, without a newly fitted posterior, independent covariance or collision-causality acceptance.

## core/sidm_products.py::native_scattering_history

<!-- implementation-sha256: ed7542644c5d3886bff45e229a25a2070b37f36513f6e8ffc9f835e801349b2a -->

Verify an actual elastic-SIDM history and track every ID in its fixed first-saved fine-DM cohort through full snapshots, including particles leaving the aperture. Read exact uint64 counters, exposure, masses and per-step guard state; bind the native operator source and require original coarse/omitted-hazard caps plus a finite clock envelope with positive lower interval duration. Produce complete-cohort diagnostic rates and separately masked physically admitted interval rates, retaining missing IDs and full time brackets. Return report, arrays and pins. Rare-event participations differ from rare-plus-frequent optical-depth exposure; no Poisson equality or unresolved microscopic collision count is inferred.

## core/sidm_products.py::export_native_scattering_rates

<!-- implementation-sha256: cdf79f9032c34bf4841acdd2a544b0a83aaa276d9b61f5fb05a9a64d5cd850c1 -->

Measure the verified fixed-cohort scattering history and require a fresh output directory. Publish scattering-rates.json with exact IDs, saved/interval diagnostics, Gyr/Gyr^-1 units and original missing/support policy. Render separate diagnostic-completeness and physical-admission panels, each showing rare-event participation versus optical-depth rates. Return the gallery manifest. Unsupported rates remain NaN in arrays/null in JSON rather than being normalized over only present particles.

## core/sidm_products.py::centroid_inputs

<!-- implementation-sha256: ae1857909d4e4b72e62e4486ba925a6d37360ac67833a0f95517e8a852c9c222 -->

Verify the original SIDM history, bounded identity graph and source-bound main-branch assembly product. Replay branch/merger selection and require one exact structure/graph observation for each history epoch, matching the same native run, branch, FOF membership, center and physical clock. Merge original numerical source pins and return history, replayed assembly, ordered keys and pins. Matching timestamps or host labels alone do not establish a valid centroid/event join.

## core/sidm_products.py::export_native_sidm_centroids

<!-- implementation-sha256: a6f8be1681540be3d211309f8968633ae404259826eca7a03377cd9b2c41e8dc -->

Verify the original history/assembly/structure join and require a fresh destination. Publish centroid-events.json retaining component order, native graph keys, original aperture and saved FOF join definition. Plot each supported baryon-minus-DM centroid offset in proper kpc against actual Gyr time, separately showing all measured epochs and recorded native joins. A DM-only run receives an explicit unavailable-centroid panel. Recheck sources and return the gallery; simultaneous offsets do not establish merger causality or continuous coalescence time.

## core/stream/sbi.py::forward_signature

<!-- implementation-sha256: 0f301abe91b46fc52670665c4072b165167834680340961c8b8d6ca5de020dc1 -->

Return SHA-256 of canonical JSON-safe forward definitions, parameter bindings and code/source hashes. Sorted compact serialization makes the identity independent of mapping insertion order. This signature identifies exact supplied content; it does not execute or accept the forward model.

## core/stream/sbi.py::require_forward_model_accepted_for

<!-- implementation-sha256: 09c83417979608954861ea0f5e3cf183e925bf05bb71b1e795634fbba67a117c -->

Require an ACCEPTED accepted-forward-domain-v1 receipt matching the expected forward signature, domain ID and observable contract. Require passing force, time-integration, progenitor, observation, selection and discrepancy controls, rehashing all retained control sources. Require each prior’s bounds inside the accepted parameter bounds and return the receipt. A training run or matching domain label cannot replace these physical/observational controls.

## core/stream/sbi.py::build_recorded_torch_prior

<!-- implementation-sha256: 162d8f12b385a7258d92c811d1a8cda4c72e5f54145f5df8636192ae4d50be43 -->

Build a Torch vector distribution from the supplied validated parameter records. Sample independent bounded uniform, log-uniform or inverse-CDF truncated-normal coordinates and compute their normalized joint log density, returning minus infinity outside the recorded bounds. Preserve parameter order and expose independent interval support. This constructs a prior distribution, not a posterior or scientific-domain acceptance; the caller supplies valid recorded prior specifications.

## core/stream/sbi.py::graph_baseline_gain

<!-- implementation-sha256: 22d68510ecb2fdfce3fcb4d9cdd0094bc935d80a57b603f3cc426955e296fd67 -->

Require identical frozen held-out cases, accepted independent calibration for graph and summary models, and the same selected independent test rows. Form paired graph-minus-baseline held-out log scores and bootstrap their mean by resampling paired groups with a keyed seed. Return the measured mean, fraction of bootstrap means above the declared minimum, threshold decision and original paired/bootstrap arrays. This conditional gain test does not establish deployment beyond the calibrated domain.

## core/stream/sbi.py::fit_ood_reference

<!-- implementation-sha256: 3ba839c3fe70a3c8361d4be71b0e2a0ca8d51fd857e23697ef3e168ecccf334b -->

From supplied finite training features compute their mean, sample covariance plus a positive diagonal floor, and componentwise min/max padded by the requested amount. Return these arrays with a training-only scope label. No held-out truth is read here; the caller must supply training observations only.

## core/stream/sbi.py::evaluate_ood

<!-- implementation-sha256: 87215f0566f918464ffb7bad9195e38e80e6ede2ed812fe1737427adadd2eb5d -->

Require finite feature/reference arrays, Cholesky-whiten the feature-minus-training-mean residual and calculate its squared Mahalanobis distance. Compute the fraction of summary coordinates outside the retained padded ranges; return both metrics and support only when both declared thresholds pass. This is a training-domain screen rather than a probability that the physical model is correct.

## core/stream/sbi.py::infer_calibrated_posterior

<!-- implementation-sha256: 0f63d373c8e601bd835b85ccc63be0a6a8bccd69618eb2dc25db3602542fb883 -->

Require accepted independent calibration bound to the actual frozen model, plus accepted gain for graph architecture. Reject privileged truth in the observation, encode it and evaluate its summary against the training-only OOD reference. Unsupported input returns OUT_OF_DOMAIN with empty posterior arrays. Otherwise load the frozen model, seed Torch and draw the requested posterior samples with equal weights. Return report/arrays, preserving calibration and science NOT_ACCEPTED; successful inference does not extend the accepted domain.

## core/stream/sbi.py::execute_sbi_stage

<!-- implementation-sha256: a4a4b13b0a8abf807b62c309c26f4e9d1814f4f238124a85ffc85ebc79719589 -->

Dispatch signature, dataset, training, calibration, paired gain or posterior stages through the recorded Inputs reader. Signature construction resolves forward assets/code without calling simulators; datasets require accepted physical domains and keep held-out truth separate. Training binds the actual dataset and frozen backend/serializer, fitting OOD statistics from training rows only. Calibration verifies the frozen model/dataset before reading evaluation truth; deployment retains calibration, graph-gain and OOD checks. Return report, arrays, stage namespace and no single epoch. This executes the selected operation rather than merely listing capability.

## core/stream/sbi.py::prepare_native_sbi_branch

<!-- implementation-sha256: e053abf27eff48efa9d18b3de77712128116ff36cc40c45a75d27fa38dbabb9e -->

Verify pinned design/acceptance/observation/selection assets, absolute forward definitions and original parameter/encoder bindings. Compute actual lineage-disjoint split feasibility without inventing parents, require summary baseline plus explicit experimental graph training and validate the unchanged independent held-out floor. Execute source-signature construction twice to check relocation identity and accepted forward-domain coverage. Publish dataset/training/calibration/gain/posterior recipes plus a separate designated-design physical posterior/predictive and six figure routes. Return RECIPES_PREPARED; preparation executes signatures only, with no forward simulations, training or calibration.

## core/response_campaign.py::public_ic_inputs

<!-- implementation-sha256: 810a49c5ddc1af80a2927b55a9da0f69c35edfda8284f1e9ad93ed61d33afc5b -->

Require a bounded original completed public-command receipt owning an actual IC_STAGE_COMPLETE run with generated/validated ICs and a passing saved power gate. Verify the exact phase seed, uniform hydro monofonIC route, complete owned IC shard set, original transfer-validation file and linear bundle. Recheck all pins and return a validated-public-native-ic-v1 contract plus sources. This adapter records original generation evidence; reuse must still execute the native transfer/velocity/power gates, and independent seeds alone do not establish empirical independence.

## core/response_campaign.py::public_ic_runner

<!-- implementation-sha256: b51f9035eca5cc43487df2e2ca5b99ae5da977ad06f91bb47c17b4b65eae3be4 -->

Require public-IC intake memory within workflow allocation, verify original generation through public_ic_inputs and publish generation/validated-ic.json with its source pins. Return COMPLETE measurements containing phase seed and shard count with science NOT_ACCEPTED. This publishes a verified IC contract rather than generating new particles or evolving a run.

## core/response_campaign.py::load_public_ic

<!-- implementation-sha256: 35355107675a3a4f5600bfdb2d62b1d8e579492154717f5f2490332641ebf93c -->

Read a bounded validated-public-native-ic-v1 capsule and verify its original producer receipt. Reconstruct the contract from the actual IC-only generation and require every generated field to match, rehash recorded sources and reject capsule changes during intake. Return report and merged pins. A stored PASS alone is insufficient for reuse.

## core/response_campaign.py::validate_paired_response

<!-- implementation-sha256: 09a4b83ce785b173edcded93afb38b53385ad9abb2ecdc30c9c4890d4b9533d8 -->

Require finite non-Boolean paired-response memory of at least 0.1 GiB and return None. The frozen template, exact stellar histories, physical clocks, periodic topology and intervention lineage are checked during input intake.

## core/response_campaign.py::paired_response_input_bindings

<!-- implementation-sha256: b2ae31c9e02a196d279c2993e5960b9fdff6eda1303a172c48a3d0577ad5c03d -->

Load the frozen box-origin response template and original completed paired stellar comparison, verifying both histories’ matching physical epochs/topology and optional original intervention result. If actions are requested, load the source-bound measured potential and add its dependencies. Return the merged pins without converting frames or measuring a response.

## core/response_campaign.py::native_paired_response_runner

<!-- implementation-sha256: d5007193bcbc842b9c31c8738841518c810613dfee7e6718c26983693a05d6ab -->

Verify allocated memory and original paired histories/template/lineage. Publish original periodic image support where needed and convert each branch to the same fixed native box-origin proper chart; missing image continuity yields an unresolved result. Otherwise write response-request.json and execute the original native response estimator, retaining its product/variant status. Recheck sources and publish paired-response.json with original intervention evidence. This measures a conditional exact-cohort response without extending the intervention’s physical/causal domain.

## core/response_campaign.py::prepare_native_response_campaign

<!-- implementation-sha256: 5fe9c35d249632e97d454a6f4778c9156721b73cca35a3c3896f023d444a20da -->

Verify an original live/late native counterfactual design, its workflow and a pinned response template. Preserve original stages and add a response consumer for each declared native paired-comparison artifact, binding the corresponding intervention/threeway result where present. Reject stage collisions or ambiguous ownership, compile the extended DAG and recheck registered code/input hashes. Publish workflow/manifest in a fresh destination and return RECIPES_PREPARED. Preparation executes no native evolution and retains original lineage/domain checks.

## core/response_campaign.py::response_target_inputs

<!-- implementation-sha256: 6e8440036d4db4552fd0163b2121b5bc545630bf09e60cdc02b4d0d6661643bf -->

Validate the pinned target controls and bind actual completed current/baseline native runs. Require each original IC shard to belong to its evolution receipt, identical complete IC hash sets within the realization, equal phase seeds and equal SIDM operator seeds. Add control pins and return settings, run/config maps, sources and a common IC-derived fine-DM namespace. This establishes target identity inputs; it does not select a baseline halo or measure its response observables.

