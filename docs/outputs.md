# Results and exports

After saved-run analysis, open `RUN/results/index.html`. Rebuild it without simulation
or analysis:

```bash
python commands.py results /absolute/path/to/RUN --verify
```

The attempt selector defaults to the latest workflow record, even if it failed. Previous
attempts remain selectable. Figures come from recorded manifests; missing or changed
files appear as navigation issues. `--verify` hashes artifacts where their manifests
supply checksums; existence-only indexing is faster and is the automatic default. This
does not certify scientific validity.

| Task | Read next |
|---|---|
| Locate run files and generations | [Export tree](#the-export-tree-and-its-interface) |
| Choose ordinary analysis products | [32-product catalogue](#single-run-product-catalogue) |
| Find a specialized scientific graph | [102-recipe reference](figure-contracts.md) |
| Navigate a tree, density map and film together | [Linked halo stories](#readable-fly-throughs-and-linked-halo-stories) |
| Trace one plotted value back to its array | [Inspect an output](#inspect-an-output-from-the-smoke-campaign) |

- **Highlights:** recorded figures for starred saved epochs.
- **By redshift:** topic-grouped epoch pages, exact requested/actual redshift and
  links to run movies. Consult each movie's source record for epoch coverage.
- **Evolution:** multi-epoch figures and figures without a single-epoch contract.
- **Movies:** recorded animations from that attempt.
- **Data:** numerical arrays, tables and portable/interactive exports.
- **Status:** requested products, applicability decisions, missing inputs,
  failures and retry guidance.

Navigation folders use `snap_012_z_1.000`; the snapshot ID prevents collisions when
redshifts round to the same display value. Raw `snapshot_###` names and published
artifact filenames remain stable for native readers and provenance. Navigation links to
one authoritative artifact, without duplicating large data. Relative links survive
relocation of the entire run. External artifacts are reported rather than silently
rebound. Unsupported historical manifests remain accessible through their workflow
records and are listed as navigation issues.

Existing workflow records and standalone published analysis summaries can be indexed. A
standalone summary is labeled `HISTORICAL_ANALYSIS_ONLY`; it cannot establish that a
whole workflow completed. Unrecorded files are not guessed into a generation. Large
metadata is linked without parsing above the 32 MiB intake bound; the page lists that
limitation. Epoch metadata comes from contracts; no epoch is inferred from a filename or
a rounded plot title.

## The export tree and its interface

```text
RUN/
  config.yaml, manifest.json       frozen run inputs and native provenance
  input/                          native parameters, ICs and output schedule
  data/                           authoritative native snapshots/catalogues
  results/
    index.html                    start here; select one analysis attempt
    navigation.json               machine-readable navigation, not new measurements
    workflow_*Z.json              outcomes, missing prerequisites and retry guidance
    analysis_products_*/          one immutable measurement/figure generation
      analysis_products.h5        measured scientific arrays and contracts
      analysis_products_summary.json
      *.png, *.pdf, *.svg         selected figure formats
      *.figure.json              axes, physical metadata and recorded artifacts
      *.plotdata.npz              arrays/geometry actually drawn
      snapshot_epochs.csv/html    actual/requested epoch table
      paper_figures/              declared multi-panel scientific story recipes
    structure_maps/maps_*/        selected particle-field/projection bundles
    movies/*/                     frame/source contracts and encoded movies
    planetarium/timeline_*/       domemaster frames, numeric/source records, movie
    3d/*/                         scene/preview/particle export generation
    navigation/ATTEMPT/snap_###_z_*/index.html
                                  topic-grouped epoch links; no duplicate particle data
```

Optional measurements also retain their declared destinations, such as a BH history or
host-specific request bundle. The homepage links them from their workflow record. These
destinations are published bundles with recorded paths and hashes. Use the navigation
layer to browse them; renaming their files can break references and integrity checks.

The page separates historical analysis, standalone old exports and a current workflow. A
movie already recorded by a workflow is not added again as a new standalone historical
attempt. A failed latest attempt remains visible as failed. Its page does not borrow
successful figures from an older attempt. Image-format alternatives share one figure
card; source records/numerical sidecars remain accessible through Data and their links.

The terminal interface's product selection applies to actual supported data. A single
run can supply many epochs; multiple runs each retain their own homepage, while
queue/batch pages link to the run homepages and their comparison products. Comparison
products retain their participating runs and epoch selections. Each multi-epoch movie
records its own coverage.

The [smoke report](smoke-test.md) records executed publication, artifact-route,
figure and native checks with their actual scope. The optional lab has separately
verified desktop/narrow Edge navigation and a saved quick-report view; browser
inspection of the main research-run homepage remains open. Large metadata and
unsupported older manifests are reported as the navigation limitations above.

## Single-run product catalogue

`all` resolves the 32 registered single-run products below against saved data. It is
strict: missing additional request files can stop execution before any products launch.
`available` runs the ready subset and records omissions. Neither word creates absent
particles, missing catalogues, new simulations or posterior samples. The number of files
depends on snapshot count, configured projections, resolved halos, available fields and
requested figure formats.

| Product | Figures and exports |
|---|---|
| `hmf` | Differential dn/dlog10M; dn/dM; cumulative abundance; analytic ST check; saved-unit HMF; maximum halo mass and rare-halo abundance versus redshift; ten reference-fit plots below. |
| `finder` | HMF overlays against Rockstar and mass-bin ratios, for the configured native halo mass definition. |
| `concentration` | Halo concentration versus mass. |
| `pk` | Matter power spectrum; separate particle-type spectra when present; saved-unit P(k) curve. |
| `correlation` | Two-point spatial correlation function. |
| `bias` | Halo bias relative to the matter field. |
| `kinematics` | Particle and halo velocity distributions. |
| `spatial` | Three-dimensional halo spatial distribution figure. |
| `subhmf` | Subhalo mass function. |
| `subhalos` | Full native SUBFIND catalogue: CSV/HDF5, host links, sizes, masses, species counts and measured kinematics; a data export, not a separate figure recipe. |
| `spin` | Specific angular momentum distribution; this is not an automatically inferred dimensionless spin parameter. |
| `profiles` | Dark-matter density and velocity-dispersion profiles; gas density, temperature and velocity-dispersion profiles when fields support them. |
| `shape` | Halo shape/triaxiality diagnostics. |
| `radial_shapes` | Radial density-shape, axis-ratio and orientation panels; measured arrays and sampling support. Requires an explicit radial-shape request. |
| `hydro` | Gas density and temperature distributions; temperature-density phase diagram; gas/star baryon fractions; compression, signed cooling/heating and vorticity maps when native fields exist; star-formation history; stellar count, birth cohorts and age history when supported. |
| `galaxies` | Measured stellar galaxy catalogue in HDF5; supplies galaxy overlays. Does not fabricate a separate luminosity or synthetic galaxy-image plot. |
| `histories` | Main-progenitor mass-accretion curves; half-mass formation redshifts; merger histories and exact history tables. Requires measured descendant links. |
| `stellar_components` | Host stellar surface-density/phase-space figure, stellar-component diagnostic panels, exact membership catalogue and Blender particle assets. Requires a saved host/search request. |
| `stellar_histories` | Selected stellar history figure and stream diagnostics; exact-ID trajectories and tabulated measurements. Requires an explicit catalogue/particle/epoch request. |
| `black_holes` | Six-panel histories: subgrid mass, inflow, model luminosity, gravity mass, inflow/Eddington-cap ratio and cumulative coupled energy; HDF5 catalogue. |
| `zoom_contamination` | Contamination figure and saved aperture/host measurements. Requires a zoom selection and tolerances. |
| `maps` | DM, gas, temperature, stellar and optional total-density maps for configured fields/projections/epochs; structure-story composites and configured measured filament overlays. |
| `blender` | Interactive combined .blend, preview PNG and section previews; particles, available halo/subhalo/galaxy overlays, scientific HUD and inspector, camera rig and verified figure gallery. |
| `inference_diagnostics` | Supplied-sample corner distributions and conditional prediction/residual panels when provided. Requires an inference request; no posterior is generated from snapshots by this option. |
| `movie` | DM density/cosmic-web MP4. |
| `gas_movie` | Gas-density MP4. |
| `temperature_movie` | Gas-temperature MP4. |
| `stars_movie` | Stellar-density MP4; records star-free omitted epochs. |
| `halo_movie` | Main-progenitor growth and merger MP4 from the current selected history table. |
| `xdmf` | Portable XDMF/HDF5 particle and field export for external 3-D tools; no automatic external viewer launch. |
| `planetarium` | Square equidistant domemaster frames, MP4, sampled particle/numeric raster data and epoch/provenance records. |
| `blender_movie` | The interactive Blender scene and previews plus its animated 3-D MP4. Selected alongside blender, it builds once. |

Compatible HMF analysis adds these ten fitting-function figures per snapshot:

1. Differential abundance overlay.
2. Measurement/fit residual panels.
3. Ratios to Sheth–Tormen.
4. Multiplicity functions.
5. Expected and measured bin counts.
6. Finite-mass-window cumulative abundance.
7. Signed Poisson deviance residuals.
8. Bin-center mass contribution.
9. Largest/smallest model spread (not an uncertainty band).
10. k-coverage/calibration-range map.

References: Press–Schechter, Sheth–Tormen, Reed 2007, Watson FOF, Jenkins and Warren.
Fits require FOF b=0.2 and supported linear inputs. Other mass conventions are
explicitly skipped, not converted by assumption. CSV tables retain the plotted values
and masks; diagnostic use is not scientific acceptance.

The analysis also attempts two cosmic-web recipes at the latest supported epoch:
`cosmic_web_3d_morphology` (density-Hessian environment morphology) and
`filament_detection_sequence` (nine-panel particle/density/ridge sequence). These are
morphology diagnostics; they are not automatically a DisPerSE result. A failed recipe appears in the parent
skipped/incomplete report. Scientific figure exports include
PNG and the configured PDF/SVG formats, plot data/metadata, plus relevant sealed
HDF5/CSV/JSON products. Map/film/Blender exports retain their own provenance and product
formats.

Separate experiments are outside this single-run `all`: comparisons between multiple
runs/cosmologies, convergence and start-epoch campaigns, isolated merger simulations,
relativistic lensing, observer-error experiments and explicit research/stream-response
campaigns. Those need their own inputs and commands. Live 3-D viewing is an explicit
launch action; the saved Blender scene and XDMF are the automatic 3-D exports.


## Readable fly-throughs and linked halo stories

The original `cosmic-flythrough` and `filament-flythrough` renderers now use a smoothed
full-DM density volume with sampled-particle highlights, fixed exposure within a film,
an eased camera path, a target standoff, a labelled FOF target and an XY camera overview.
These are saved-snapshot morphology views. Their artificial colors are not stellar light,
and camera travel is not time evolution. Exposures are adjusted between films.

The `histories` merger figure is a branching descendant-linked tree. The previous merger
mass-ratio scatter remains available as `halo_merger_mass_ratios`. H0 follows the most
massive incoming progenitor; H1, H2… are display branch aliases. Exact catalogue IDs
remain epoch-local, and FOF row numbers are never equated with Consistent-Trees IDs.

To link a sealed positioned tree to recorded full-box density arrays and an offline film:

```bash
python commands.py halo-story RUN --tree TREE_JSON \
  --density-contract DENSITY_MOVIE_CONTRACT_JSON \
  --density-directory ATLAS_DIRECTORY --output NEW_STORY_DIRECTORY
```

The current density adapter consumes `focused-dmo-density-movie-v1` source records and
their saved `log10_projected_density_over_mean` NPZ arrays. Tree provenance and run
namespace, source HDF5 checksums, epochs and saved positions must match. It never
invents missing early progenitors. The HTML embeds its state, uses local map PNGs and
a local MP4, and shares branch colors and labels across all three views. Node selection
updates the map, exact-ID details and movie time; branch selection jumps to a recorded
detection if that branch is absent at the current epoch.

For reliable MP4 seeking across browsers, serve the entire run using:

```bash
python scripts/serve_results.py RUN --port 0 --open
```

The server binds only to 127.0.0.1 and implements byte-range requests. The prepared run
also includes an `Open_atlas.command` launcher and a standalone copy of the server.
Use `scripts/verify_results.py RUN --url http://127.0.0.1:PORT` to audit reachable local
links, decoded images, first/middle/last movie frames and HTTP asset responses.

## Inspect an output from the smoke campaign

The [smoke report](smoke-test.md) links retained local examples. For each figure,
compare its PNG/PDF/SVG with its `.figure.json` and plotted-data arrays; for maps,
inspect the referenced numerical HDF5/NPZ and projection metadata. For a movie,
inspect its frame/epoch contract and decode at least one frame. For Blender, open
the `.blend` alongside its source/publication record. Source data, measured arrays
and visual exports have different roles; retain all three when reviewing a result.

For a registered two-dimensional line figure, the example below verifies the
physical-array archive and prints the first supported point in its first series.
Pass its `.figure.json` file, such as the optional atlas's input-power figure.
It reads the numerical sidecar rather than estimating a value from image pixels:

```sh
python - /absolute/path/to/FIGURE.figure.json <<'PY'
from pathlib import Path
import hashlib, json, sys
import numpy as np

figure = Path(sys.argv[1]).resolve()
metadata = json.loads(figure.read_text())['scientific_metadata']
contract = metadata['physical_array_contract']
archive = (figure.parent / contract['path']).resolve()
if hashlib.sha256(archive.read_bytes()).hexdigest() != contract['sha256']:
    raise ValueError('Physical-array archive changed')
panel = metadata['recipe']['panels'][0]
prefix = panel['panel_id'] + '__series_000_'
with np.load(archive, allow_pickle=False) as arrays:
    supported = np.flatnonzero(arrays[prefix + 'supported'])
    if not len(supported):
        raise ValueError('This series has no supported point')
    i = int(supported[0])
    print('selection:', panel['selection'])
    print('row:', i)
    print('x:', arrays[prefix + 'x'][i], panel['x_units'])
    print('y:', arrays[prefix + 'y'][i], panel['y_units'])
    print('source hashes:', metadata['source_hashes'])
PY
```

Follow those source records to the original numerical product and its producer.
Ordinary analysis `.plotdata.npz` files instead describe plotted arrays/artist
geometry; their authoritative scientific measurements remain in
`analysis_products.h5`. Use each sidecar's own schema, dataset names and masks.
The [notebook](../optional_tests/reference_gallery/Walkthrough.ipynb) provides
worked density, zoom, shape and event calculations from original arrays.
