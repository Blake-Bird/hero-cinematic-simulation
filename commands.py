"""One command interface for simulation, analysis, audits and portable setup."""
from __future__ import annotations

import sys as _entry_sys
from core import cli as _small_cli

# Seed the existing public command table before optional scientific imports.
# These commands can dispatch without loading unrelated domain dependencies.
COMMANDS = {
    'study': lambda: __import__('core.study', fromlist=['main']).main(),
    'hmf': _small_cli.hmf_main,
    'hmf-scan': _small_cli.hmf_main,
    'hmf-learning': _small_cli.hmf_learning_main,
    'run': _small_cli.run_pipeline_main,
    'audit-graph-producers': _small_cli.audit_graph_producers_main,
    'combine-graph-galleries': _small_cli.combine_graph_galleries_main,
    'export-hubble-figures': _small_cli.export_hubble_figures_main,
    'export-inference-figures': _small_cli.export_inference_figures_main,
    'export-abundance-figures': _small_cli.export_abundance_figures_main,
    'export-assembly-figures': _small_cli.export_assembly_figures_main,
    'export-bh-figures': _small_cli.export_bh_figures_main,
    'export-convergence-figures': _small_cli.export_convergence_figures_main,
    'export-splashback-figures': _small_cli.export_splashback_figures_main,
    'export-stream-figures': _small_cli.export_stream_figures_main,
    'export-structure-figures': _small_cli.export_structure_figures_main,
    'flagship': _small_cli.flagship_main,
    'ingest-cosmography': _small_cli.ingest_cosmography_main,
    'prepare-analysis-figures': _small_cli.prepare_analysis_figures_main,
    'bao-model': _small_cli.bao_model_main,
    'flagship-products': _small_cli.flagship_products_main,
    'music-boundary': _small_cli.music_boundary_main,
    'music-capacity': _small_cli.music_capacity_main,
    'music-flow-prepare': _small_cli.music_flow_prepare_main,
    'music-frozen-model': _small_cli.music_frozen_model_main,
    'music-phase-plan': _small_cli.music_phase_plan_main,
    'music-seeded-model': _small_cli.music_seeded_model_main,
    'ingest-bao': _small_cli.ingest_bao_main,
    'prepare-flagship-campaign': _small_cli.prepare_flagship_campaign_main,
    'prepare-flagship-profile': _small_cli.prepare_flagship_profile_main,
    'export-power-figures': _small_cli.export_power_figures_main,
    'export-sidm-figures': _small_cli.export_sidm_figures_main,
    'render-scene': _small_cli.render_scene_main,
    'conditional-h0': _small_cli.conditional_h0_main,
    'export-analysis-figures': _small_cli.export_analysis_figures_main,
    'artwork': _small_cli.artwork_main,
    'inspect-poisson-failure': _small_cli.inspect_poisson_failure_main,
}
if __name__ == '__main__' and len(_entry_sys.argv)>1 and _entry_sys.argv[1] in COMMANDS:
    _selected_command = _entry_sys.argv.pop(1)
    _entry_sys.argv[0] += ' '+_selected_command
    raise SystemExit(COMMANDS[_selected_command]())

# Preparation should not import unrelated plotting or simulation backends.
if __name__ == '__main__' and _entry_sys.argv[1:2] == ['native-prepare']:
    from core.campaigns import main as _prepare_main
    raise SystemExit(_prepare_main(_entry_sys.argv[2:]))
if __name__ == '__main__' and _entry_sys.argv[1:2] == ['package']:
    from core.release import package_suite_main as _package_main
    del _entry_sys.argv[1]
    raise SystemExit(_package_main())

if __name__ == '__main__' and _entry_sys.argv[1:2] == ['class']:
    from core.linear import run_class_main
    del _entry_sys.argv[1]
    raise SystemExit(run_class_main())

if __name__ == '__main__' and _entry_sys.argv[1:2] in (['native-music-transfer'], ['native-music-mixed2lpt']):
    from core.research.generator_controls import native_music_transfer_main, native_music_mixed2lpt_main
    _control = _entry_sys.argv.pop(1)
    raise SystemExit((native_music_transfer_main if _control == 'native-music-transfer' else native_music_mixed2lpt_main)())

# ========================================================================
# Shared command arguments and portable setup

import argparse
import hashlib
import json
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from core.foundation import atomic_text, file_sha256
from core.native import directory_owner

from core.release import (
    SOURCE_ROOT as package_suite_ROOT,
    package_suite_main, release_inventory, source_files,
)

import importlib
import shutil
import subprocess
import sys

bootstrap_SUITE = Path(__file__).resolve().parent
SOFTWARE = bootstrap_SUITE / "core/software"
sys.path.insert(0, str(bootstrap_SUITE / "core"))
from core.resources import cpu_thread_budget, runtime_paths

PACKAGES = ("numpy", "scipy", "matplotlib", "yaml", "h5py", "imageio", "imageio_ffmpeg", "PIL", "pyvista")
TOOLS = ("make", "cc", "c++", "mpicc", "mpicxx", "cmake", "pkg-config", "perl")


def check_environment() -> None:
    """Check imports and build tools; missing prerequisites stop the command."""
    if os.name == "nt":
        raise RuntimeError("Run this suite inside Ubuntu on WSL2. Native Windows builds are not supported; use Ubuntu 24.04 in WSL2.")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required")
    missing = []
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            print(f"Python {name}: {getattr(module, '__version__', 'available')}")
        except ImportError:
            missing.append(name)
    for command in TOOLS:
        path = shutil.which(command)
        print(f"Tool {command}: {path or 'missing'}")
        if not path:
            missing.append(command)
    # Require development headers/link flags, rather than accepting the mere
    # presence of a package manager. GADGET also accepts hdf5-openmpi metadata.
    libraries = ("gsl", "fftw3", "hdf5") + (("libtirpc",) if sys.platform.startswith("linux") else ())
    for library in libraries:
        result = subprocess.run(["pkg-config", "--exists", library], check=False) if shutil.which("pkg-config") else None
        if result is None or result.returncode:
            if library == "hdf5" and result is not None and subprocess.run(["pkg-config", "--exists", "hdf5-openmpi"], check=False).returncode == 0:
                continue
            missing.append(f"{library} development library")
    blender = shutil.which("blender")
    if not blender and Path("/Applications/Blender.app/Contents/MacOS/Blender").is_file():
        blender = "/Applications/Blender.app/Contents/MacOS/Blender"
    print(f"Blender: {blender or 'install for .blend scene creation/rendering'}")
    if missing:
        raise RuntimeError("Install prerequisites: " + ", ".join(missing))




def bootstrap_main() -> None:
    """Build requested engines or inspect the host prerequisites."""
    import json
    from core.native import build_engine
    parser = argparse.ArgumentParser(description="Check dependencies and build the bundled, pinned scientific engines.\n\nRun after bootstrap.py installs the selected dependency profile. This builds executables for the current\nmachine; native binaries are deliberately excluded from the portable release.\nGADGET is built by commands.py gadget with each run's actual physics configuration.\nThe eight engine sources are already bundled; no additional repository is fetched.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--build-all", action="store_true", help="build both Boltzmann solvers, IC generator, halo/tree tools and stellar STF")
    for name in ("class", "axiclass", "monofonic", "music2", "rockstar", "consistent-trees", "velociraptor", "disperse"):
        parser.add_argument("--build-" + name, action="store_true")
    parser.add_argument("--jobs", type=int, default=min(4, cpu_thread_budget()))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    check_environment()
    for name in ("class", "axiclass", "monofonic", "music2", "rockstar", "consistent-trees", "velociraptor", "disperse"):
        if args.build_all or getattr(args, "build_" + name.replace("-", "_")):
            build_engine(name, args.jobs)
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from initial_conditions import validate_linear_bundle,validate_ic_realization,compare_ic_generators


def _complex_prediction(value):
    if value is None:
        return None
    if not isinstance(value,dict) or set(value) != {'real','imag'}:
        raise ValueError('Fourier prediction must declare real and imag arrays')
    real,imag = np.asarray(value['real'],dtype=float),np.asarray(value['imag'],dtype=float)
    if real.shape != imag.shape or real.ndim != 1 or not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError('Fourier prediction requires aligned finite real/imag vectors')
    return real+1j*imag


def audit_ic_main():
    parser=argparse.ArgumentParser(description='Verify exact-epoch linear bundles or measure delivered IC physical controls.')
    parser.add_argument('kind',choices=('bundle','realization','compare'))
    parser.add_argument('request',type=Path,help='JSON keyword arguments for the selected validator')
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    if not isinstance(request,dict):
        raise ValueError('IC audit request must be a JSON object')
    if args.kind == 'bundle':
        report=validate_linear_bundle(**request)
    elif args.kind == 'realization':
        for key in ('expected_density','expected_velocity'):
            if key in request:
                request[key]=_complex_prediction(request[key])
        report=validate_ic_realization(**request)
    else:
        report=compare_ic_generators(**request)
    print(json.dumps(report,indent=2,allow_nan=False))
    return 0 if report['status']=='PASS' else 2 if report['status']=='INCOMPLETE' else 1

# ========================================================================
# observe_stellar_catalogue

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from stars import export_mock_observations


def observe_stellar_catalogue_main():
    parser = argparse.ArgumentParser(description='Apply declared correlated errors, missingness and cuts to a saved stellar truth export.')
    parser.add_argument('truth_directory', type=Path)
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    print(json.dumps(export_mock_observations(args.truth_directory,
                    **json.loads(args.request.read_text())), indent=2))
import datetime

run_analysis_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(run_analysis_ROOT), str(run_analysis_ROOT / "core")]
import matplotlib
matplotlib.use("Agg")
from analysis import ANALYSIS_PRODUCTS, analyze_run, snapshot_ids as run_analysis_snapshot_ids
from core.native import native_run_owner

DESCRIPTIONS = {
    "hmf": "Halo abundance versus mass, with counting uncertainties",
    "finder": "Compare available halo catalogues and mass definitions",
    "concentration": "Resolved halo concentrations",
    "pk": "Matter power spectrum and measured shot noise",
    "correlation": "Spatial two-point correlations",
    "bias": "Halo bias relative to the matter field",
    "kinematics": "Measured velocities and velocity dispersions",
    "spatial": "Halo spatial distributions",
    "subhmf": "Subhalo mass functions",
    "subhalos": "Every SUBFIND row: host, bound mass, size, species counts/masses and kinematics; CSV and HDF5",
    "spin": "Halo angular momentum and spin",
    "profiles": "Measured radial density profiles",
    "shape": "Halo shapes from particle distributions",
    "radial_shapes": "Proper-radius density-moment shells, sampling errors and orientation flags; needs --radial-shapes JSON",
    "hydro": "Gas thermodynamics and star formation; needs gas/star data",
    "galaxies": "Stellar galaxy catalogues; needs actual star particles",
    "histories": "Halo growth histories; needs merger-tree catalogues",
    "stellar_components": "Stellar phase-space candidates and measured morphology; needs explicit host/search settings",
    "stellar_histories": "Exact saved candidate IDs across epochs and bound-subhalo memberships; needs --stellar-history JSON",
    "black_holes": "Measured native BH mass, mean inflow, model luminosity and feedback histories",
    "zoom_contamination": "Saved science-region coarse-matter fractions and particle sampling; needs --zoom-regions JSON",
    "maps": "Projected mass-density structure maps and selected overlays",
    "blender": "Particle and catalogue timeline for the Blender scene",
    "inference_diagnostics": "Supplied parameter samples: corner densities, prediction envelopes and residuals; needs --inference-diagnostics JSON",
}

from core.visual_pipeline import VISUAL_PRODUCTS
DESCRIPTIONS.update(VISUAL_PRODUCTS)


def parse_products(values):
    if isinstance(values, str):
        values = values.split(",")
    selected = list(dict.fromkeys(str(v).strip().lower() for v in values))
    if selected in (["all"], ["available"]):
        return selected
    if not selected or any(v not in DESCRIPTIONS for v in selected):
        raise ValueError("Choose available, all, or a comma list of: " + ", ".join(DESCRIPTIONS))
    return selected


def plan_saved_analysis(run_dir, products, *, requests=None):
    """Read-only applicability plan shared by public saved-run analysis."""
    from core.research.capabilities import inspect_run_capabilities, resolve_applicable_products
    requests = requests or {}
    inventory = inspect_run_capabilities(run_dir)
    catalogue = {name: {'required_species': ['dm'], 'requires_catalogues': name in {
        'hmf', 'finder', 'concentration', 'correlation', 'bias', 'spatial', 'subhmf', 'spin', 'profiles', 'shape', 'subhalos'},
        'requires_complete_volume': name in {'hmf', 'pk', 'correlation', 'bias'}} for name in DESCRIPTIONS}
    power_epochs = [row['snapshot'] for row in inventory['snapshot_epochs']]
    catalogue['pk'].update(inputs_ready=bool(power_epochs) and all(
        (Path(run_dir)/'data/powerspecs'/('powerspec_%03d.txt' % epoch)).is_file()
        for epoch in power_epochs), input_reason='Saved native matter power spectra required at every selected epoch')
    for name in ('hydro',):
        catalogue[name]['required_species'] = ['gas']
    for name in ('galaxies', 'stellar_components', 'stellar_histories'):
        catalogue[name]['required_species'] = ['stars']
    star_epochs = [row['snapshot'] for row in inventory['snapshot_epochs']
                   if inventory['species_by_epoch'].get(str(row['snapshot']),{}).get('stars',0)]
    for name in ('galaxies','stellar_components'):
        catalogue[name].update(requires_catalogues=True,catalogue_epochs=star_epochs)
    search = requests.get('stellar_search')
    if isinstance(search,dict) and type(search.get('snapshot')) is int:
        catalogue['stellar_components']['catalogue_epochs'] = [search['snapshot']]
        catalogue['stellar_components']['requested_epochs'] = [search['snapshot']]
    catalogue['radial_shapes']['requires_catalogues'] = True
    shapes = requests.get('radial_shapes')
    if isinstance(shapes,dict):
        if shapes.get('schema')=='radial-shape-batch-v1' and isinstance(shapes.get('selections'),list):
            epochs = [row.get('snapshot') for row in shapes['selections'] if isinstance(row,dict)]
            if all(type(epoch) is int for epoch in epochs):epochs = list(dict.fromkeys(epochs))
        else:epochs = [shapes.get('snapshot')]
        catalogue['radial_shapes'].update(catalogue_epochs=epochs,requested_epochs=epochs)
    catalogue['stellar_histories']['minimum_epochs'] = 2
    history = requests.get('stellar_history')
    if isinstance(history,dict) and 'snapshots' in history:
        catalogue['stellar_histories']['requested_epochs'] = history['snapshots']
    catalogue['black_holes'].update(required_species=['black_holes'], required_features=['black_holes'])
    catalogue['zoom_contamination']['requires_zoom'] = requests.get('zoom_regions') is None
    for name, key in [('radial_shapes', 'radial_shapes'), ('stellar_components', 'stellar_search'),
                      ('stellar_histories', 'stellar_history'), ('zoom_contamination', 'zoom_regions'),
                      ('inference_diagnostics', 'inference_diagnostics')]:
        catalogue[name].update(inputs_ready=requests.get(key) is not None, input_reason='Explicit '+key+' request required')
    catalogue['histories'].update(minimum_epochs=2, requires_catalogues=True)
    for name in ('blender','blender_movie','movie','gas_movie','temperature_movie','stars_movie','halo_movie','planetarium'):
        catalogue[name]['minimum_epochs'] = 2
    for name in ('gas_movie','temperature_movie'):
        catalogue[name]['required_species'] = ['gas']
    catalogue['stars_movie']['required_species'] = ['stars']
    catalogue['halo_movie']['requires_catalogues'] = True
    if requests.get('find_halos'):
        for row in catalogue.values():
            if row.get('requires_catalogues'):
                row['requires_catalogues'] = False
                row['prerequisite_ids'] = ['halo_finders']

    # Posterior assets are conditional: a run without a requested inference
    # experiment has no posterior to diagnose, regardless of particle species.
    requested = parse_products(products)
    if requested in (['all'], ['available']) or any(name in requested for name in ('blender','blender_movie')):
        from visualization import _find_blender_executable
        for name in ('blender','blender_movie'):
            catalogue[name].update(inputs_ready=bool(_find_blender_executable()),
                input_reason='Blender executable is missing; install it before requesting scene construction')
    if requests.get('inference_diagnostics') is None and requested in (['all'], ['available']):
        catalogue['inference_diagnostics']['conditional_input_absent'] = True
    plan = resolve_applicable_products(['all'] if requested == ['available'] else requested, inventory, catalogue)
    if requested == ['available']:
        plan['requested'] = requested
        plan['omitted_missing_inputs'] = dict(plan['missing_inputs'])
        plan['missing_inputs'] = {}
    plan['capabilities'] = inventory
    # The history reader supports physical trajectories with -3/unknown bound
    # membership when catalogues are absent. Preserve that explicit partial
    # measurement instead of suppressing supported phase-space history.
    history = requests.get('stellar_history')
    if isinstance(history,dict) and isinstance(history.get('snapshots'),list):
        missing = [epoch for epoch in history['snapshots']
                   if inventory['species_by_epoch'].get(str(epoch),{}).get('stars',0)
                   and not inventory['catalogue_support'].get(str(epoch),False)]
        if missing and plan['decisions']['stellar_histories']['status']=='APPLICABLE':
            plan['partial_support'] = {'stellar_histories':{
                'supported':'saved selected stellar phase-space history',
                'unresolved':'bound-subhalo membership when selected stars are present at epochs without native catalogues',
                'missing_catalogue_epochs':missing,'missing_membership_code':-3}}
    return plan


def choose_products():
    print("\nScientific outputs")
    print("  available      Every supported output with ready inputs; record omissions")
    print("  all            Every physically applicable output; require extra requests")
    for key, description in DESCRIPTIONS.items():
        print(f"  {key:14s} {description}")
    while True:
        try:
            return parse_products(ask_required("Graphs and outputs [comma-separated names, available, or all]"))
        except ValueError as exc:
            print(exc)


def configure_analysis_outputs(cfg, products):
    """Bind requested workflow outputs to core settings before freezing a run."""
    selected = parse_products(products)
    analysis = cfg['analysis']
    analysis['requested_outputs'] = selected
    if selected in (['all'], ['available']):
        analysis.update(all=True, products=[])
    else:
        scientific = [name for name in selected if name in ANALYSIS_PRODUCTS]
        if scientific:
            analysis.update(all=False, products=scientific)
    if ('histories' in selected or 'halo_movie' in selected or selected in (['all'], ['available'])) and cfg['halos']['rockstar']:
        cfg['halos']['consistent_trees'] = True


def choose_stellar_search(run_dir):
    """Choose a saved epoch/host and explicit physical STF controls interactively."""
    from halos import _group_files, _read_catalogue_series, _stellar_search_controls
    from core.resources import cpu_thread_budget, runtime_paths
    available = run_analysis_snapshot_ids(run_dir)
    if required_choice("Search a measured stellar host now", ["run", "skip"]) == "skip":
        return None
    print("Saved snapshot numbers: " + ", ".join(map(str, available)))
    while True:
        snapshot = required_integer("Snapshot number", min(available), max(available))
        if snapshot in available:
            break
        print("Choose one of the saved snapshots.")
    paths = _group_files(Path(run_dir), snapshot)
    if not paths:
        print("No FOF host catalogue is available at this epoch.")
        return None
    counts = _read_catalogue_series(paths, "Group", "GroupLenType")[:, 4]
    hosts = sorted((i for i, count in enumerate(counts) if count >= 128),
                   key=lambda i: int(counts[i]), reverse=True)
    if not hosts:
        print("No host has the 128 stars needed for 8 background cells of at least 16 stars.")
        return None
    print("Most populated stellar hosts [FOF row: star count]: " +
          ", ".join(f"{i}: {int(counts[i])}" for i in hosts[:20]))
    while True:
        host = required_integer("Zero-based FOF host row", 0, len(counts)-1)
        if counts[host] >= 128:
            break
        print("That host has fewer than the required 128 stars.")
    while True:
        controls = {
            "minimum_particles": required_integer("Minimum candidate members", 3, 2147483647),
            "velocity_neighbors": required_integer("Velocity-density neighbors [starting value 32]", 4, 2147483647),
            "physical_neighbors": required_integer("Physical neighbor pool [starting value 256]", 5, 2147483647),
            "cell_fraction": required_number("Background cell fraction [starting value 0.01]", 1e-12, 1),
            "outlier_threshold": required_number("Velocity-density outlier threshold [starting value 2.5]", 1e-12, 1e6),
            "significance_level": required_number("Candidate significance threshold [starting value 1]", 1e-12, 1e6),
            "velocity_ratio": required_number("Maximum linked speed ratio [starting value 2; >1]", 1+1e-12, 1e6),
            "opening_angle_pi": required_number("Linked velocity opening angle / pi [starting value 0.1]", 1e-12, 1),
            "linking_length_kpc": required_number("Physical linking length [proper kpc; choose for resolved structure]", 1e-12, 1e9),
        }
        try:
            controls = _stellar_search_controls(controls)
            n = int(counts[host])
            if n <= controls["physical_neighbors"] or controls["minimum_particles"] > n:
                raise ValueError("Host is too small for those neighbor/member counts")
            if not 16 <= n * controls["cell_fraction"] <= n/8:
                raise ValueError("Background needs at least 16 stars/cell and 8 cells/host")
            break
        except ValueError as exc:
            print(exc)
    return {"snapshot": snapshot, "host": host, "controls": controls,
            "memory_gib": required_number("STF memory budget [GiB]", 1e-6, 1e6),
            "threads": required_integer("STF CPU threads", 1, cpu_thread_budget())}


def validate_stellar_search(settings):
    """Reject malformed requests before a long simulation or analysis begins."""
    from halos import _stellar_search_controls
    from commands import integer_value
    from core.resources import cpu_thread_budget, runtime_paths
    import math
    required = {"snapshot", "host", "controls", "memory_gib", "threads"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("Stellar search requires exactly: " + ", ".join(sorted(required)))
    result = dict(settings)
    for key in ("snapshot", "host"):
        result[key] = integer_value(settings[key], key, 0, 2147483647)
    result["threads"] = integer_value(settings["threads"], "STF threads", 1, cpu_thread_budget())
    memory = settings["memory_gib"]
    if isinstance(memory, (bool, str)) or not isinstance(memory, (int, float)) or not math.isfinite(memory) or memory <= 0:
        raise ValueError("STF memory_gib must be a finite positive number")
    result["controls"] = _stellar_search_controls(settings["controls"])
    return result


def gate_finder_descendants(run_dir, selected, status, *, allow_incomplete):
    """Recheck produced catalogue support; failed prerequisites block consumers.

    Independent requested measurements may proceed only under explicit
    allow-incomplete. Finder completion labels alone never prove that all
    requested epochs have usable catalogue publications.
    """
    decisions = status.get('applicability',{}).get('decisions',{})
    dependent = [name for name in selected if 'halo_finders' in
                 decisions.get(name,{}).get('prerequisite_ids',[])]
    if not dependent:return selected
    outcomes = status.get('halo_finders',{})
    if not isinstance(outcomes,dict):
        raise ValueError('Finder prerequisite must return named stage outcomes')
    failures = {name:row.get('reason',row.get('status','MISSING_STATUS'))
                if isinstance(row,dict) else 'MALFORMED_OUTCOME'
                for name,row in outcomes.items()
                if not isinstance(row,dict) or row.get('status')!='COMPLETE'}
    requests = dict(status['requests'],find_halos=False)
    after = plan_saved_analysis(run_dir,selected,requests=requests)
    status['post_finder_applicability'] = after
    blocked = {}
    for name in dependent:
        decision = after['decisions'][name]
        if failures:
            blocked[name] = 'Finder prerequisite failed: '+json.dumps(failures,sort_keys=True)
        elif decision['status']!='APPLICABLE':
            blocked[name] = 'Produced catalogue support unavailable: '+decision['reason']
    if blocked:
        status.setdefault('blocked_products',{}).update(blocked)
        status['unavailable'].update(blocked)
        if not allow_incomplete:
            raise RuntimeError('Analysis prerequisites incomplete: '+json.dumps(blocked,sort_keys=True))
    return [name for name in selected if name not in blocked]


def validate_analysis_requests(products, *, stellar_search=None, stellar_history=None,
                               zoom_regions=None, inference_diagnostics=None,
                               radial_shapes=None, allow_incomplete=False):
    """Check request structure before evolution or analysis work starts.

    This validates declared settings only. Saved host membership, snapshot
    availability, and physical measurement support are checked on real output.
    """
    selected = parse_products(products)
    if selected in (['all'], ['available']):
        selected = list(DESCRIPTIONS)
    if inference_diagnostics is not None and 'inference_diagnostics' not in selected:
        raise ValueError('Inference diagnostics request requires inference_diagnostics product')
    if not allow_incomplete:
        missing = [flag for product, value, flag in (
            ("stellar_components", stellar_search, "--stellar-search"),
            ("stellar_histories", stellar_history, "--stellar-history"),
            ("zoom_contamination", zoom_regions, "--zoom-regions"),
            ("inference_diagnostics", inference_diagnostics, "--inference-diagnostics"),
            ('radial_shapes',radial_shapes,'--radial-shapes'),
        ) if product in selected and value is None]
        if missing:
            raise ValueError("Complete analysis requires " + ", ".join(missing) +
                             "; provide requests or select the products appropriate to this run")
    if zoom_regions is not None:
        required = {'fine_dm_mass_msun', 'maximum_coarse_mass_fraction'}
        if 'zoom_contamination' not in selected:
            raise ValueError('--zoom-regions requires zoom_contamination product')
        if (not isinstance(zoom_regions, dict) or required-set(zoom_regions)
                or set(zoom_regions)-required-{'regions','host_tracking','output_directory','minimum_fine_dm_count','minimum_stellar_count','resolution_budget'}
                or ('regions' in zoom_regions)==('host_tracking' in zoom_regions)):
            raise ValueError('Zoom audit requires mass/tolerance and exactly one of regions or host_tracking; optional output directory, sampling thresholds and resolution_budget')
    if stellar_search is not None:
        if "stellar_components" not in selected:
            raise ValueError("Stellar search requires the stellar_components product")
        stellar_search = validate_stellar_search(stellar_search)
    if stellar_history is not None:
        required = {"catalogue", "snapshots", "candidate_ids", "memory_gib"}
        if "stellar_histories" not in selected:
            raise ValueError("Stellar history requires the stellar_histories product")
        if not isinstance(stellar_history, dict) or set(stellar_history) - required - {"output_directory"} or required - set(stellar_history):
            raise ValueError("Stellar history requires catalogue, snapshots, candidate_ids, memory_gib; optional output_directory")
    if radial_shapes is not None:
        if 'radial_shapes' not in selected:
            raise ValueError('--radial-shapes requires radial_shapes product')
        from halos import validate_shape_request
        radial_shapes=validate_shape_request(radial_shapes)
    return {"stellar_search": stellar_search, "stellar_history": stellar_history,
            "zoom_regions": zoom_regions, "inference_diagnostics": inference_diagnostics,
            "radial_shapes": radial_shapes}


def _execute_run(run_dir, products, *, find_halos=False, stellar_search=None, stellar_history=None, zoom_regions=None,
                 allow_incomplete=False, inference_diagnostics=None, radial_shapes=None, resume=False, status, destination):
    """Use saved simulation settings; record unavailable physics explicitly."""
    import yaml
    from figures import apply_plot_style
    run_dir = Path(run_dir).expanduser().resolve()
    selected = parse_products(products)
    if selected in (['all'], ['available']):
        raise ValueError('Resolve all against saved-run capabilities before execution')
    requests = validate_analysis_requests(selected, stellar_search=stellar_search,
        stellar_history=stellar_history, zoom_regions=zoom_regions,
        inference_diagnostics=inference_diagnostics, radial_shapes=radial_shapes,
        allow_incomplete=allow_incomplete)
    stellar_search = requests["stellar_search"]
    radial_shapes = requests["radial_shapes"]
    if not run_analysis_snapshot_ids(run_dir):
        raise ValueError(f"No completed GADGET snapshots in {run_dir}")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    apply_plot_style(cfg.get("visualization", {}))
    from core.foundation import atomic_json
    def checkpoint(stage):
        status['current_stage'] = stage
        atomic_json(destination, status)
    if find_halos:
        checkpoint('halo_finders')
        from halos import run_all
        status["halo_finders"] = run_all(run_dir, cfg)
        if not isinstance(status['halo_finders'],dict):
            raise ValueError('Finder prerequisite must return named stage outcomes')
        for name,outcome in status['halo_finders'].items():
            if not isinstance(outcome,dict) or outcome.get('status')!='COMPLETE':
                status['unavailable']['halo_finder_'+name] = (
                    outcome.get('reason') or outcome.get('status') or 'MISSING_STATUS'
                    if isinstance(outcome,dict) else 'MALFORMED_OUTCOME')
        selected = gate_finder_descendants(run_dir,selected,status,allow_incomplete=allow_incomplete)
        atomic_json(destination,status)
    scientific = [p for p in selected if p in ANALYSIS_PRODUCTS]
    if 'halo_movie' in selected and 'histories' not in scientific:
        scientific.append('histories')
    if scientific:
        checkpoint('scientific_analysis')
        try:
            summary = analyze_run(run_dir, products=scientific, include_pdf_report=False)
            if not isinstance(summary, dict) or not summary.get("output_directory"):
                raise RuntimeError("Analysis did not return its result directory")
            status["artifacts"]["analysis"] = str(summary["output_directory"])
            status["unavailable"].update(summary.get("skipped", {}))
        except Exception as exc:
            status['unavailable']['scientific_analysis']=f'{type(exc).__name__}: {exc}'
            checkpoint('scientific_analysis_failed')
    if "stellar_components" in selected:
        try:
            checkpoint('stellar_components')
            if stellar_search is None:
                status["unavailable"]["stellar_components"] = (
                    "Supply --stellar-search JSON with snapshot, host, controls, memory_gib and threads; "
                    "a proper-kpc linking length and measured host selection are required")
            else:
                from halos import find_stellar_substructures
                status["artifacts"]["stellar_components"] = find_stellar_substructures(
                    run_dir, **stellar_search)
        except Exception as exc:
            status['unavailable']['stellar_components'] = f'{type(exc).__name__}: {exc}'
            checkpoint('stellar_components_failed')
    if 'radial_shapes' in selected:
        try:
            checkpoint('radial_shapes')
            if radial_shapes is None:
                status['unavailable']['radial_shapes']='Supply --radial-shapes JSON with saved snapshot/host, support, windows and estimator/sampling controls'
            else:
                from halos import export_shape_request,plot_radial_shapes
                shape_directory=Path(radial_shapes['output_directory'])
                shape_result=export_shape_request(run_dir,radial_shapes,resume=resume)
                batch=shape_result.get('schema')=='radial-shape-batch-product-v1'
                report_path=shape_directory/('batch.json' if batch else 'report.json')
                artifact={'directory':str(shape_directory),'report':str(report_path),
                          'status':shape_result['status'],'science_acceptance':'NOT_ACCEPTED',
                          'report_sha256': file_sha256(report_path)}
                if batch:
                    artifact.update(figure_registry=shape_result['figure_registry'],inspector_references=shape_result['inspector_references'])
                    status['artifacts']['radial_shapes']=artifact
                    if shape_result['status']!='MEASURED':
                        status['unavailable']['radial_shapes']='Unmeasured batch bins or insufficient sampling; all profiles and figures retained'
                else:
                    status['artifacts']['radial_shapes']=artifact
                atomic_json(destination,status)
                if not batch:
                    figure_directory=shape_directory.with_name(shape_directory.name+'_figures')
                    if shape_result['status']!='MEASURED':
                        status['unavailable']['radial_shapes']='Unmeasured radial density-shape shells; failed bins and figures retained'
                    elif radial_shapes['bootstrap_realizations'] and any(s['sampling_uncertainty']['status']!='ESTIMATED' for s in shape_result['shells']):
                        status['unavailable']['radial_shapes']='Requested shape sampling uncertainty has insufficient bootstrap success; outputs retained'
                    checkpoint('radial_shapes_figures')
                    try:
                        if resume and figure_directory.exists():
                            saved = json.loads((figure_directory/'report.json').read_text())
                            if (saved.get('source_report_sha256') != artifact['report_sha256']
                                    or saved.get('implementation_sha256') != file_sha256(run_analysis_ROOT/'core/halos.py')
                                    or set(saved.get('product_sha256', {})) != {'radial_shapes.png', 'radial_shapes.pdf'}
                                    or any(file_sha256(figure_directory/name) != sha for name, sha in saved['product_sha256'].items())):
                                raise ValueError('Shape figure checksums or source changed')
                        else:
                            plot_radial_shapes(shape_directory,figure_directory)
                        artifact['figures']=str(figure_directory)
                    except Exception as exc:
                        artifact['figure_failure']={'type':type(exc).__name__,'message':str(exc)}
                        status['unavailable']['radial_shapes_figures']='Plotting failed; measured parent/report remain retained'
                    atomic_json(destination,status)
        except Exception as exc:
            status['unavailable']['radial_shapes'] = f'{type(exc).__name__}: {exc}'
            checkpoint('radial_shapes_failed')
    if "stellar_histories" in selected:
        try:
            checkpoint('stellar_histories')
            if stellar_history is None:
                status["unavailable"]["stellar_histories"] = "Supply --stellar-history JSON with catalogue, snapshots, candidate_ids and memory_gib"
            else:
                from stars import trace_stellar_candidates
                status["artifacts"]["stellar_histories"] = trace_stellar_candidates(run_dir, **stellar_history)
        except Exception as exc:
            status['unavailable']['stellar_histories'] = f'{type(exc).__name__}: {exc}'
            checkpoint('stellar_histories_failed')
    if "black_holes" in selected:
        try:
            checkpoint('black_holes')
            if not cfg.get("black_holes", {}).get("enabled", False):
                status["unavailable"]["black_holes"]="Native BH model is disabled; Newtonian markers cannot supply accretion histories"
            else:
                from black_holes import analyze_black_holes, NoBlackHoleData
                try:
                    status["artifacts"]["black_holes"]=analyze_black_holes(run_dir)
                except NoBlackHoleData as exc:
                    status['unavailable']['black_holes'] = str(exc)
        except Exception as exc:
            status['unavailable']['black_holes'] = f'{type(exc).__name__}: {exc}'
            checkpoint('black_holes_failed')
    if 'zoom_contamination' in selected:
        try:
            checkpoint('zoom_contamination')
            if zoom_regions is None:
                status['unavailable']['zoom_contamination'] = 'Supply --zoom-regions JSON with saved science apertures and contamination tolerance'
            else:
                from zoom import audit_zoom_regions
                status['artifacts']['zoom_contamination'] = audit_zoom_regions(run_dir, **zoom_regions)
                if status['artifacts']['zoom_contamination'].get('status') != 'PASS':
                    status['unavailable']['zoom_contamination'] = 'Saved-region contamination acceptance failed; audit artifacts retained'
        except Exception as exc:
            status['unavailable']['zoom_contamination'] = f'{type(exc).__name__}: {exc}'
            checkpoint('zoom_contamination_failed')
    from core.visual_pipeline import execute_visual_products
    execute_visual_products(run_dir, selected, cfg, status, checkpoint, radial_shapes=radial_shapes)
    if 'inference_diagnostics' in selected:
        try:
            checkpoint('inference_diagnostics')
            if inference_diagnostics is None:
                status['unavailable']['inference_diagnostics'] = 'Supply --inference-diagnostics JSON with samples_npz, metadata_json and output_directory'
            else:
                from figures import export_diagnostics
                status['artifacts']['inference_diagnostics'] = export_diagnostics(inference_diagnostics)
        except Exception as exc:
            status['unavailable']['inference_diagnostics'] = f'{type(exc).__name__}: {exc}'
            checkpoint('inference_diagnostics_failed')
    status["status"] = "INCOMPLETE" if status["unavailable"] else "COMPLETE"
    status['product_status'] = status['status']
    status['current_stage'] = 'products_finished'
    status["allow_incomplete"] = bool(allow_incomplete)
    from core.run_status import attach_retry_commands
    attach_retry_commands(status)
    atomic_json(destination, status)
    print(f"\nExports: {run_dir / 'results'}\nOutput record: {destination}")
    if status["unavailable"] and not allow_incomplete:
        raise RuntimeError("Requested products are incomplete; retained artifacts and reasons in " + str(destination))
    return status


@native_run_owner
def process_run(run_dir, products, *, find_halos=False, stellar_search=None, stellar_history=None,
                zoom_regions=None, allow_incomplete=False, research_spec=None, evidence=None, inference_diagnostics=None, radial_shapes=None,
                resume_from=None, deferred_products=()):
    """Retain failure/interruption state; product completeness never grants science acceptance."""
    from core.foundation import atomic_json, acceptance_report, file_sha256 as digest, validate_spec
    run_dir = Path(run_dir).expanduser().resolve()
    previous = None
    if resume_from is not None:
        previous = json.loads(Path(resume_from).read_text())
        stale_running = False
        if previous.get('status') == 'RUNNING':
            pid = previous.get('owner_pid')
            if type(pid) is not int or pid <= 0:
                raise ValueError('Cannot safely resume RUNNING record without a valid owner PID')
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                stale_running = True
            except PermissionError:
                pass
            if not stale_running:
                raise ValueError('Cannot resume a workflow while its owner process is still alive')
        if (previous.get('run') != str(run_dir) or previous.get('products') != ['radial_shapes']
                or parse_products(products) != ['radial_shapes'] or find_halos
                or any(v is not None for v in (stellar_search, stellar_history, zoom_regions, inference_diagnostics))
                or previous.get('requests', {}).get('find_halos')
                or (previous.get('status') not in ('FAILED','INTERRUPTED','INCOMPLETE') and not stale_running)):
            raise ValueError('Selective resume currently requires an interrupted/failed radial_shapes-only workflow without halo finding')
        if radial_shapes is None:
            radial_shapes = previous['requests']['radial_shapes']
        from halos import validate_shape_request
        if validate_shape_request(radial_shapes) != validate_shape_request(previous['requests']['radial_shapes']):
            raise ValueError('Resume requires the exact original radial shape request')
        if research_spec is None:
            research_spec = previous.get('research_spec')
        elif research_spec != previous.get('research_spec'):
            raise ValueError('Resume requires the original research specification')
    destination = run_dir / 'results' / ('workflow_' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    status = {'run': str(run_dir), 'products': [], 'status': 'RUNNING',
              'workflow_record': str(destination),
              'owner_pid': os.getpid(),
              'product_status': 'RUNNING', 'science_status': 'NOT_ACCEPTED',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'current_stage': 'preflight', 'artifacts': {}, 'unavailable': {},
              'requests': {'find_halos': find_halos, 'stellar_search': stellar_search,
                           'stellar_history': stellar_history, 'zoom_regions': zoom_regions,
                           'inference_diagnostics': inference_diagnostics,'radial_shapes':radial_shapes},
              'resume': {'automatic_resume_safe': False,
                         'instruction': 'Inspect this record and stage outputs before rerunning; immutable exports may require a new output directory.'}}
    if previous is not None:
        status['resumed_from'] = {'path': str(Path(resume_from).resolve()), 'sha256': digest(resume_from)}
    atomic_json(destination, status)
    try:
        selected = parse_products(products)
        status['requested_products'] = selected
        applicability = plan_saved_analysis(run_dir, selected, requests=status['requests'])
        from core.setup_walkthrough import REQUESTS
        deferred = set(deferred_products)
        if deferred - set(REQUESTS) or (selected not in (['all'], ['available']) and deferred - set(selected)):
            raise ValueError('Only explicitly selected specialized products may be deferred')
        status['deferred_outputs'] = sorted(deferred)
        for name in deferred:
            applicability['selected'] = [p for p in applicability['selected'] if p != name]
            applicability['missing_inputs'].pop(name, None)
            applicability['explicit_unavailable'].pop(name, None)
            if applicability['decisions'][name]['status'] not in ('INAPPLICABLE', 'NOT_APPLICABLE'):
                applicability['decisions'][name] = {'status': 'PENDING', 'reason': 'Explicitly deferred during setup; select saved data and supply its measurement request'}
                status['unavailable'][name] = applicability['decisions'][name]['reason']
        status['applicability'] = applicability
        candidates = set(DESCRIPTIONS) if selected in (['all'], ['available']) else set(selected)
        for name, detail in applicability['decisions'].items():
            if name in candidates and name not in applicability['selected']:
                print("  Omitted %s: %s (%s)" % (name, detail['reason'], detail['status']))
        print("Analysis products: " + ", ".join(applicability['selected']))
        if applicability['explicit_unavailable']:
            raise ValueError('Explicit analysis products unavailable: '+json.dumps(
                applicability['explicit_unavailable'],sort_keys=True))
        if applicability['missing_inputs'] and not allow_incomplete:
            raise ValueError('Applicable analysis inputs missing: '+json.dumps(applicability['missing_inputs'],sort_keys=True))
        status['unavailable'].update(applicability['missing_inputs'])
        selected = applicability['selected']
        if not selected and not deferred:
            raise ValueError('No supported analysis products selected')
        if not selected and deferred:
            status.update(status='INCOMPLETE', product_status='INCOMPLETE', current_stage='awaiting_saved_selections')
            atomic_json(destination, status)
            return status
        status['products'] = selected
        # Validate radial requests before hashing data, running finders or analysis.
        if radial_shapes is not None:
            if 'radial_shapes' not in selected:
                raise ValueError('--radial-shapes requires radial_shapes product')
            from halos import validate_shape_request
            radial_shapes = validate_shape_request(radial_shapes)
            status['requests']['radial_shapes'] = radial_shapes
        if not allow_incomplete:
            missing = [flag for product, value, flag in (
                ('stellar_components', stellar_search, '--stellar-search'),
                ('stellar_histories', stellar_history, '--stellar-history'),
                ('zoom_contamination', zoom_regions, '--zoom-regions'),
                ('inference_diagnostics', inference_diagnostics, '--inference-diagnostics'),
                ('radial_shapes', radial_shapes, '--radial-shapes'))
                if product in selected and value is None]
            if missing:
                raise ValueError('Complete analysis requires ' + ', '.join(missing))
        status['resume'] = {'automatic_resume_safe': selected == ['radial_shapes'] and not find_halos,
                            'instruction': 'For failed/interrupted radial_shapes-only workflows: commands.py analysis RUN --graphs radial_shapes --resume THIS_RECORD; unchanged published measurements are verified and reused.'}
        if research_spec is not None:
            status['research_spec'] = research_spec
        if research_spec is not None:
            from core.foundation import legacy_science_view
            research_spec = legacy_science_view(research_spec)
            validate_spec(research_spec, selected)
            if 'validation_campaign' in research_spec:
                from core.foundation import validation_campaign_record
                campaign_record=validation_campaign_record(research_spec['validation_campaign'])
                campaign_path=destination.with_name(destination.stem+'_deferred_campaign.json')
                atomic_json(campaign_path,campaign_record)
                status['artifacts']['deferred_validation_campaign']=str(campaign_path)
            manifest = {'schema_version': 1, 'research_spec': research_spec, 'products': selected,
                        'requests': status['requests'], 'config_sha256': digest(run_dir / 'config.yaml'),
                        'data_sha256': {str(p.relative_to(run_dir)): digest(p) for p in sorted(
                            (run_dir / 'data').rglob('*')) if p.is_file()},
                        'source_sha256': {str(p.relative_to(run_analysis_ROOT)): digest(p) for p in sorted(
                            list(run_analysis_ROOT.glob('*.py')) + list((run_analysis_ROOT / 'core').glob('*.py')) + list((run_analysis_ROOT / '5_blender').glob('*.py')))}}
            manifest_path = destination.with_name(destination.stem + '_manifest.json')
            atomic_json(manifest_path, manifest)
            status['artifacts']['research_manifest'] = str(manifest_path)
        result = _execute_run(run_dir, selected, find_halos=find_halos, stellar_search=stellar_search,
                              stellar_history=stellar_history, zoom_regions=zoom_regions,
                              allow_incomplete=allow_incomplete, inference_diagnostics=inference_diagnostics,
                              radial_shapes=radial_shapes,
                              resume=previous is not None,
                              status=status, destination=destination)
        if research_spec is not None and research_spec.get('cosmography_campaign'):
            from core.cosmography import run_cosmography_campaign
            cosmography = run_cosmography_campaign(research_spec['cosmography_campaign'],
                run_dir / 'results' / ('cosmography_' + destination.stem), request_base=run_dir)
            result['artifacts']['cosmography_campaign'] = cosmography
        if research_spec is not None:
            report = acceptance_report(manifest, evidence or {}, run_dir)
            if result['product_status'] != 'COMPLETE':
                report['status'] = 'NOT_ACCEPTED'
                report['product_failure'] = dict(result['unavailable'])
            report_path = destination.with_name(destination.stem + '_acceptance.json')
            atomic_json(report_path, report)
            result['science_status'] = report['status']
            result['artifacts']['science_acceptance'] = str(report_path)
        else:
            result['science_reason'] = 'No declared research specification/evidence; COMPLETE describes generated products only.'
        atomic_json(destination, result)
        return result
    except BaseException as exc:
        if status['status'] == 'RUNNING' or status['status'] == 'COMPLETE':
            status['status'] = 'INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'FAILED'
        if status['product_status'] == 'RUNNING':
            status['product_status'] = status['status']
        status['science_status'] = 'NOT_ACCEPTED'
        status['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        atomic_json(destination, status)
        raise
    finally:
        from core.run_status import attach_retry_commands
        attach_retry_commands(status)
        atomic_json(destination, status)
        from core.run_results import build_run_results
        try:
            status['results_index'] = str(build_run_results(run_dir))
            atomic_json(destination, status)
        except Exception as navigation_error:
            status['navigation_error'] = {'type': type(navigation_error).__name__, 'message': str(navigation_error)}
            atomic_json(destination, status)


def analysis_batch_main():
    """Analyze selected saved runs sequentially and compare each to a reference."""
    parser=argparse.ArgumentParser(description=analysis_batch_main.__doc__)
    parser.add_argument('runs',nargs='*',type=Path,help='Saved run paths; omit to select from the run list')
    parser.add_argument('--graphs',help='Comma-separated graph selection, available, or all')
    parser.add_argument('--output',type=Path,required=True,help='Fresh batch-report directory')
    parser.add_argument('--reference',type=int,default=1,help='One-based reference run number (default: first)')
    parser.add_argument('--state-root',type=Path);parser.add_argument('--find-halos',action='store_true')
    parser.add_argument('--allow-incomplete',action='store_true');parser.add_argument('--no-comparisons',action='store_true')
    args=parser.parse_args()
    roots=args.runs or select_runs(runtime_paths(run_analysis_ROOT,args.state_root).runs_root,'config.yaml')
    products=parse_products(args.graphs) if args.graphs else choose_products()
    from core.batch_analysis import run_batch_analysis
    report=run_batch_analysis(roots,products,args.output,reference=args.reference-1,find_halos=args.find_halos,
                              allow_incomplete=args.allow_incomplete,compare=not args.no_comparisons)
    print(json.dumps({'status':report['status'],'report':str(args.output.resolve()/'batch_report.json')},indent=2))
    if report.get('results_index'):print('Open batch results: '+report['results_index'])
    if report['status']!='COMPLETE':raise SystemExit(1)


def run_analysis_main():
    """Parse command arguments and dispatch this module's documented workflow."""
    parser = argparse.ArgumentParser(description='Analyze saved snapshots and export explicitly selected scientific figures.')
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--graphs", help="comma list ("+", ".join(DESCRIPTIONS)+"), available (ready inputs), or all (strict)")
    parser.add_argument('--full-analysis', action='store_true',
                        help='Legacy core analysis suite; maps/BH/Blender remain separate workflow products')
    parser.add_argument('--overwrite', action='store_true',
                        help='With --full-analysis only: archive previous results before replacing them')
    parser.add_argument('--dry-run', action='store_true', help='Inspect saved-run applicability without publishing or executing products')
    parser.add_argument('--state-root', type=Path, help='External runtime state directory')
    parser.add_argument("--find-halos", action="store_true", help="run configured halo finders before analysis")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="retain an INCOMPLETE record and allow exit 0 when selected products are unavailable")
    parser.add_argument("--stellar-search", type=Path,
                        help="JSON object: snapshot, host (FOF row), controls, memory_gib, threads")
    parser.add_argument("--stellar-history", type=Path,
                        help="JSON object: catalogue, snapshots, candidate_ids, memory_gib; optional output_directory")
    parser.add_argument('--zoom-regions', type=Path,
                        help='JSON object: regions, fine_dm_mass_msun, maximum_coarse_mass_fraction; optional output_directory')
    parser.add_argument('--research-spec', type=Path, help='Versioned physical observable and acceptance specification JSON')
    parser.add_argument('--evidence', type=Path, help='Measured acceptance evidence keyed product/test')
    parser.add_argument('--inference-diagnostics', type=Path, help='JSON object: samples_npz, metadata_json, output_directory')
    parser.add_argument('--radial-shapes',type=Path,help='Native saved host/snapshot and complete radial density-shape request JSON')
    parser.add_argument('--resume',type=Path,help='Retained failed/interrupted radial_shapes-only workflow record')
    args = parser.parse_args()
    from core.resources import runtime_paths
    paths = runtime_paths(run_analysis_ROOT, args.state_root)
    selected_runs = [args.run_dir] if args.run_dir else select_runs(paths.runs_root, "config.yaml")
    if len(selected_runs)>1:
        if (args.full_analysis or args.overwrite or args.dry_run or args.resume or args.stellar_search or args.stellar_history
                or args.zoom_regions or args.research_spec or args.evidence or args.inference_diagnostics or args.radial_shapes):
            parser.error('Multi-run selection supports normal selected graphs; run specialized requests individually')
        products=parse_products(args.graphs) if args.graphs else choose_products()
        while True:
            answer=input('Comparison reference within your selection [1]: ').strip() or '1'
            if answer.isdigit() and 1<=int(answer)<=len(selected_runs):
                reference=int(answer)-1;break
            print('Choose a number from 1 to %d.'%len(selected_runs))
        import uuid
        destination=paths.state_root/'batch_analysis'/('analysis_'+uuid.uuid4().hex)
        from core.batch_analysis import run_batch_analysis
        report=run_batch_analysis(selected_runs,products,destination,reference=reference,find_halos=args.find_halos,
                                  allow_incomplete=args.allow_incomplete,compare=True)
        print('Batch report: '+str(destination/'batch_report.json'))
        if report.get('results_index'):print('Open batch results: '+report['results_index'])
        if report['status']!='COMPLETE':raise SystemExit(1)
        return 0
    run_dir=selected_runs[0]
    if args.overwrite and not args.full_analysis:
        parser.error('--overwrite requires --full-analysis; selected exports publish fresh bundles')
    if args.full_analysis:
        if (args.graphs or args.resume or args.stellar_search or args.stellar_history or args.zoom_regions
                or args.inference_diagnostics or args.radial_shapes or args.research_spec or args.evidence):
            parser.error('--full-analysis cannot be combined with selected-output or research request flags')
        if args.dry_run:
            parser.error('Use --graphs available --dry-run to inspect saved-run support')
        if args.find_halos:
            from halos import run_all
            from analysis import load_cfg
            outcomes = run_all(run_dir, load_cfg(run_dir))
            if any(row.get('status') != 'COMPLETE' for row in outcomes.values()):
                raise RuntimeError('Halo prerequisites incomplete: ' + json.dumps(outcomes, default=str))
        analyze_run(run_dir, overwrite=args.overwrite, full_analysis=True)
        return 0
    products = parse_products(args.graphs) if args.graphs else choose_products()
    stellar_search = json.loads(args.stellar_search.read_text()) if args.stellar_search else None
    stellar_history = json.loads(args.stellar_history.read_text()) if args.stellar_history else None
    zoom_regions = json.loads(args.zoom_regions.read_text()) if args.zoom_regions else None
    research_spec = json.loads(args.research_spec.read_text()) if args.research_spec else None
    evidence = json.loads(args.evidence.read_text()) if args.evidence else None
    inference_diagnostics = json.loads(args.inference_diagnostics.read_text()) if args.inference_diagnostics else None
    radial_shapes=json.loads(args.radial_shapes.read_text()) if args.radial_shapes else None
    if args.dry_run:
        print(json.dumps(plan_saved_analysis(run_dir, products, requests={
            'find_halos':args.find_halos,
            'stellar_search': stellar_search, 'stellar_history': stellar_history,
            'zoom_regions': zoom_regions, 'radial_shapes': radial_shapes,
            'inference_diagnostics': inference_diagnostics}), indent=2))
        return 0
    if args.resume:
        prior = json.loads(args.resume.read_text())
        if research_spec is None:
            research_spec = prior.get('research_spec')
    if evidence is not None and research_spec is None:
        parser.error('--evidence requires --research-spec')
    if stellar_search is not None and "stellar_components" not in products:
        parser.error("--stellar-search requires stellar_components in --graphs")
    if "stellar_components" in products and stellar_search is None and not args.graphs:
        stellar_search = choose_stellar_search(run_dir)
    process_run(run_dir, products, find_halos=args.find_halos, stellar_search=stellar_search,
                stellar_history=stellar_history, zoom_regions=zoom_regions, allow_incomplete=args.allow_incomplete,
                research_spec=research_spec, evidence=evidence, inference_diagnostics=inference_diagnostics,
                radial_shapes=radial_shapes, resume_from=args.resume)
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from halos import detect_disappearance_swap_jump_fragmentation
from core.data import export_numeric_audit
from core.foundation import file_sha256


def audit_tree_integrity_main():
    parser=argparse.ArgumentParser(description='Publish source-bound native tree diagnostics while retaining every original link.');parser.add_argument('request',type=Path);parser.add_argument('destination',type=Path);parser.add_argument('--from-consistent-trees',action='store_true')
    args=parser.parse_args();data=json.loads(args.request.read_text())
    if args.from_consistent_trees:
        from halos import consistent_tree_series
        if set(data)!={'run_dir','native_intake','thresholds','epoch_contract'}:raise ValueError('Declare native tree intake and diagnostics')
        imported=consistent_tree_series(data['run_dir'],**data['native_intake'])
        data={**imported,'thresholds':data['thresholds'],'epoch_contract':data['epoch_contract']}
        data.pop('namespace')
    if set(data)!={'nodes','links','thresholds','source_hashes','epoch_contract'}:raise ValueError('Native nodes/links, frozen limits, source hashes and actual epoch required')
    if data['thresholds'].get('link_origin','native_tree')!='native_tree':raise ValueError('Original audit requires original native links; publish repairs separately')
    result=detect_disappearance_swap_jump_fragmentation(data['nodes'],data['links'],**data['thresholds'])
    sources={**data['source_hashes'],str(args.request.resolve()):file_sha256(args.request),str(Path(__file__).resolve()):file_sha256(__file__),str(Path(__file__).resolve().parent/'core/halos.py'):file_sha256(Path(__file__).resolve().parent/'core/halos.py')}
    report=export_numeric_audit(args.destination,result,{'node_ids':np.array([n['id'] for n in data['nodes']],dtype=np.uint64)},
        {'node_ids':{'units':'identity','quantity':'epoch-qualified native node identities; see original_nodes for namespaces','identity_namespace':'tree:epoch-qualified-native'}},
        algorithm=__file__,context={'epoch':data['epoch_contract'],'selection':'declared native tree nodes'},source_hashes=sources)
    print(json.dumps({'status':report['status'],'diagnostic_flags':len(report['flags'])},indent=2))


def propose_tree_repair_main():
    parser = argparse.ArgumentParser(description='Publish reviewable alternative tree links while retaining the original native audit.')
    parser.add_argument('request', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    from halos import publish_tree_repair_proposal, verify_tree_repair_proposal
    if args.verify:
        print(json.dumps({'status': verify_tree_repair_proposal(args.destination)['status']}, indent=2))
        return
    before = file_sha256(args.request)
    request = json.loads(args.request.read_text())
    if set(request) != {'original_audit', 'proposed_links', 'rationale', 'thresholds'}:
        raise ValueError('Declare original_audit, proposed_links, rationale and thresholds')
    report = publish_tree_repair_proposal(args.destination, **request,
        request_source_hashes={str(args.request.resolve()): before,
                              str(Path(__file__).resolve()): file_sha256(__file__)})
    print(json.dumps({'status': report['status'], 'diagnostic_flags': len(report['proposed_flags'])}, indent=2))

# ========================================================================
# measure_halo_shapes

sys.path.insert(0, str(Path(__file__).resolve().parent/'core'))
from halos import export_shape_request


def measure_halo_shapes_main():
    parser=argparse.ArgumentParser(description='Measure radial density shapes around an exact saved native host centre.\n\nparent_selection: complete_support_sphere (default), fof_members, subfind_bound_associated or subfind_central_bound. SUBFIND self-binding/FOF association does not establish global host binding.')
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('request',type=Path)
    parser.add_argument('--resume',action='store_true',help='verify and reuse an unchanged published request')
    args=parser.parse_args()
    result=export_shape_request(args.run_dir,json.loads(args.request.read_text()),resume=args.resume)
    print(json.dumps({'status':result['status'],'science_acceptance':result['science_acceptance'],
                      'shell_status':[s['status'] for s in result.get('shells',[])],
                      'profiles':result.get('profiles',[])},indent=2))
    if result['status']!='MEASURED':
        raise SystemExit(1)

# ========================================================================
# run_research_acceptance

run_research_acceptance_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(run_research_acceptance_ROOT / 'core'))
from core.foundation import acceptance_report, atomic_json


def run_research_acceptance_main():
    parser = argparse.ArgumentParser(description='Check saved research evidence without rerunning or overwriting analysis.')
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite acceptance evidence')
    manifest = json.loads(args.manifest.read_text())
    evidence = json.loads(args.evidence.read_text())
    if not {'config_sha256', 'data_sha256', 'source_sha256'} <= set(manifest):
        parser.error('Use a provenance-bearing research manifest generated by commands.py analysis')
    report = acceptance_report(manifest, evidence, args.run_dir)
    atomic_json(args.output, report)
    print(report['status'] + ': ' + str(args.output))
    return 0 if report['status'] == 'ACCEPTED' else 1

# ========================================================================
# compare_finders
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from halos import gadget_catalogue,rockstar_catalogue
from halos import publish_finder_disagreements
from core.native import native_run_lock


def compare_finders_main():
    parser=argparse.ArgumentParser(description='Compare requested saved native finder catalogues and preserve all disagreements.');parser.add_argument('run_dir',type=Path);parser.add_argument('request',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();request=json.loads(args.request.read_text())
    if set(request)!={'snapshot','catalogues','matching'} or len(request['catalogues'])!=2:raise ValueError('Declare two independent catalogue selections and matching thresholds')
    with native_run_lock(args.run_dir,operation='finder comparison'):
        catalogues=[]
        for selection in request['catalogues']:
            if selection.get('finder')=='velociraptor':
                from halos import read_velociraptor_catalogue
                if set(selection)!={'finder','directory'}:raise ValueError('Select a sealed native VELOCIraptor directory')
                catalogue=read_velociraptor_catalogue(selection['directory'])
                if catalogue['epoch']['snapshot_storage_index']!=request['snapshot']:raise ValueError('VELOCIraptor epoch mismatch')
                catalogues.append(catalogue)
                continue
            function=rockstar_catalogue if selection.get('finder')=='rockstar' else gadget_catalogue
            options=dict(selection)
            if function is rockstar_catalogue:options.pop('finder')
            catalogues.append(function(args.run_dir,request['snapshot'],**options))
        result=publish_finder_disagreements(args.destination,*catalogues,request['matching'])
    print(json.dumps(result,indent=2,allow_nan=False))

# ========================================================================
# build_identity_graph
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from halos import IdentityGraph
from core.foundation import ProductRef, StageRequest


def build_identity_graph_main():
    parser=argparse.ArgumentParser(description='Publish an exact, source-bound ownership graph and indexed query product.');parser.add_argument('request',type=Path);parser.add_argument('destination',type=Path);parser.add_argument('--reuse-completed',action='store_true');parser.add_argument('--from-native',action='store_true')
    args=parser.parse_args();request_hash=file_sha256(args.request);data=json.loads(args.request.read_text())
    stage_request=StageRequest('ownership_graph',data,(ProductRef('request',args.request,request_hash,1),),('ownership_graph',))
    stage_request.verify_inputs()
    bindings={str(args.request.resolve()):request_hash,str(Path(__file__).resolve()):file_sha256(__file__),
        str(Path(__file__).resolve().parent/'core/foundation.py'):file_sha256(Path(__file__).resolve().parent/'core/foundation.py')}
    if args.from_native:
        from halos import export_subhalo_ownership
        report=export_subhalo_ownership(destination=args.destination,reuse_completed=args.reuse_completed,request_source_hashes=bindings,**data)
        print(json.dumps({'status':report['status'],'nodes':len(report['nodes']),'edges':len(report['edges'])},indent=2))
        return
    required={'limits','epochs','nodes','edges','source_hashes'}
    if not required<=set(data) or set(data)-required-{'ssp_birth_audits','native_source_histories'}:raise ValueError('Declare graph limits, shared epochs, typed observations/edges and source bindings')
    graph=IdentityGraph(**data['limits'])
    for row in data['epochs']:graph.add_epoch(row['storage_index'],row['epoch_contract'])
    by_epoch = {}
    for row in data['nodes']:
        by_epoch.setdefault(row['epoch'], []).append({key: value for key, value in row.items() if key != 'epoch'})
    for epoch, observations in by_epoch.items(): graph.add_epoch_nodes(epoch, observations)
    from halos import ANCESTRY
    graph.add_ancestry_edges([edge for edge in data['edges'] if edge['kind'] in ANCESTRY])
    graph.add_parent_zoom_edges([edge for edge in data['edges'] if edge['kind'] == 'parent_zoom'])
    for edge in data['edges']:
        if edge['kind'] not in ANCESTRY | {'parent_zoom'}:
            graph.add_edge(edge['source'],edge['target'],edge['kind'],evidence=edge['evidence'])
    sources={**data['source_hashes'],**bindings}
    from halos import add_native_ssp_birth_audit
    for intake in data.get('ssp_birth_audits',[]):
        sources.update(add_native_ssp_birth_audit(graph,**intake))
    from halos import add_native_hydro_source_history
    for intake in data.get('native_source_histories',[]):
        sources.update(add_native_hydro_source_history(graph,**intake))
    report=graph.publish_indexed_queries(args.destination,source_hashes=sources,reuse_completed=args.reuse_completed)
    print(json.dumps({'status':report['status'],'nodes':len(report['nodes']),'edges':len(report['edges'])},indent=2))
import math
import yaml

run_gadget_SUITE = Path(__file__).resolve().parent
sys.path.insert(0, str(run_gadget_SUITE))
from evolution import (mean_particle_spacing_kpc_h, particle_mass_msun_h,
                      prepare_and_run, validate_supported_options,
                      validate_hydro_configuration, HYDRO_STAR_FORMATION_PARAMETERS,
                      HYDRO_SPH_PARAMETERS)


def spectrum_path(value: str | None) -> Path:
    """Resolve a named completed run or an explicit spectrum.json path."""
    if not value:
        from core.resources import runtime_paths
        value = str(select_run(runtime_paths(run_gadget_SUITE).runs_root / "linear", "spectrum.json"))
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path /= "spectrum.json"
    record = load_json(path)
    if record.get("status") != "complete":
        raise ValueError(f"Cosmology is incomplete: {path}")
    return path


def gather_answers(source: Path) -> dict:
    """Collect simulation settings with persisted answers and review."""
    from core.setup_walkthrough import collect_simulation
    return collect_simulation(source, outputs=False)


def _gather_answers(source: Path) -> dict:
    """Collect physical inputs and explicit numerical/figure prescriptions."""
    from core.cli_prompts import required_values
    info = load_json(source)
    start = float(info["parameters"]["z_start"])
    print("\n=== GADGET-4: PARTICLES, HALOS AND SCIENTIFIC OUTPUTS ===")
    print(f"Cosmology and IC redshift come from {source}; z_start={start:g}.")
    print("Every question requires an explicit answer; no run preset is selected.")
    print("\n[2/7] Physical configuration")
    answers = {
        "cosmology": str(source),
        "name": ask_required("Run name [identifier]"),
        "box_mpc_h": required_number("L, periodic comoving box side [Mpc/h]", 0.1, 10000),
        "particles_per_dimension": required_integer("N, particles per side [integer; N^3 dark matter particles]", 32, 2048),
        "physics": required_choice("Evolved species [dm: collisionless matter; hydro: gas plus matter, with gas physics chosen below]", ["dm", "hydro"]),
        "seed": required_integer("Random-phase seed [integer]", 0, 2147483647),
    }
    print("\n[3/7] Resources")
    answers['mpi_ranks'] = required_integer("MPI processes [integer; parallel CPU workers]", 1, 1024)
    print("\n[4/7] Saved epochs and numerical controls")
    while True:
        answers['schedule'] = ask_required(f"Snapshot redshifts z [dimensionless; comma list, z* for highlights, or first:last:step; 0 <= z <= {start:g}]")
        try:
            redshift_schedule(answers['schedule'], start)
            break
        except ValueError as error:
            print(error)
    box, n = answers["box_mpc_h"], answers["particles_per_dimension"]
    # All solver controls below come from explicit answers. Do not fill a
    # standard/high run recipe and then ask for the same values again.
    answers["accuracy"] = "custom"
    answers["num_files_per_snapshot"] = required_integer("HDF5 shards per snapshot [1 to MPI process count]", 1, answers["mpi_ranks"])
    answers.update(required_values("Gravity mesh and tree", [
        ("pm_grid", "force-grid cells per side", int, 64, 16384),
        ("asmth", "Tree/PM split [PM cells]", float, 1e-12, 100),
        ("multipole_order", "tree multipole order", int, 1, 5)]))
    answers.update(required_values("Gravitational softenings", [
        ("softening_kpc_h", "comoving softening [kpc/h]", float, 1e-12, 1000*box),
        ("softening_max_physical_kpc_h", "maximum physical softening [kpc/h]", float, 1e-12,
         lambda row: row['softening_kpc_h'])]))
    answers.update(required_values("Timesteps", [
        ("max_size_timestep_ln_a", "maximum Delta ln(a)", float, 1e-15, 1),
        ("min_size_timestep_ln_a", "minimum safety threshold Delta ln(a); 0 disables", float, 0,
         lambda row: row['max_size_timestep_ln_a'])]))
    answers.update(required_values("Force and integration tolerances", [
        ("err_tol_force_acc", "relative force tolerance", float, 1e-12, 10),
        ("err_tol_int_accuracy", "integration accuracy coefficient", float, 1e-12, 10),
        ("err_tol_theta", "tree opening angle", float, 1e-12, 10)]))
    answers.update(required_values("Checkpoints and execution budget", [
        ("restart_every_seconds", "checkpoint interval [wall seconds]", int, 60, 604800),
        ("time_limit_cpu_seconds", "wall seconds; checkpoint/stop at 85%", float, 1, 1e12)]))
    answers.update(required_values("Resources", [
        ("omp_threads", "OpenMP threads per MPI process", int, 1, 1024),
        ("build_jobs", "parallel compiler jobs", int, 1, 1024),
        ("memory_limit_gb", "total simulation memory [GiB across MPI ranks]", float, 1, 1e6)]))
    if answers["physics"] == "hydro":
        answers["courant_factor"] = required_number("SPH Courant coefficient [dimensionless]", 1e-12, 1)
        answers["desired_neighbors"] = required_integer("SPH target gas neighbors [count]", 1, n**3 - 1)
        answers["neighbor_deviation"] = required_number("SPH neighbor-count tolerance [count]", 0, answers["desired_neighbors"] - 1e-12)
        answers["artificial_viscosity"] = required_number("SPH artificial viscosity coefficient [dimensionless]", 0, 100)
        answers["initial_gas_temperature"] = required_number("Gas initial temperature [K; used only for zero-energy IC gas]", 0, 1e12)
        answers["hydro_model"] = required_choice(
            "Gas model [adiabatic; cooling: TREECOOL cooling/UV; star_formation: also Springel-Hernquist stars]",
            ["adiabatic", "cooling", "star_formation"])
        if answers["hydro_model"] != "adiabatic":
            table = required_choice("Cooling/UV table [bundled: pinned TREECOOL; custom: supply a file]", ["bundled", "custom"])
            if table == "custom":
                answers["treecool_file"] = ask_required("TREECOOL file path [GADGET ionization/heating table]")
        if answers["hydro_model"] == "star_formation":
            labels = {
                "max_sfr_timescale": "Gas consumption time at threshold [native GADGET time units]",
                "temp_clouds": "Cold-cloud temperature [K]",
                "temp_supernova": "Supernova temperature [K]",
                "factor_evp": "Cloud evaporation parameter [dimensionless]",
                "factor_sn": "Short-lived massive-star mass fraction [0 < beta < 1]",
                "crit_phys_density": "Critical hydrogen number density [cm^-3; 0 requests model-derived threshold]",
                "crit_overdensity": "Minimum comoving baryon overdensity [dimensionless]",
            }
            for key in HYDRO_STAR_FORMATION_PARAMETERS:
                answers[key] = required_number(labels[key], 0 if key == "crit_phys_density" else 1e-12,
                                               1 - 1e-12 if key == "factor_sn" else 1e30)
        answers["galaxy_min_star_particles"] = required_integer("Resolved galaxy minimum bound star particles [count]", 1, 1000000000)
        answers["galaxy_min_dm_particles"] = required_integer("Resolved galaxy minimum bound DM particles [count]", 1, 1000000000)
        bh_mode = required_choice("Black-hole source model [disabled; density_exclusion: one-particle dense-gas seed; density_exclusion_reservoir: bounded nearest-first donor assembly within the declared kernel aperture]", ["disabled", "density_exclusion", "density_exclusion_reservoir"])
        if bh_mode != "disabled":
            print("BH mode uses synchronized timesteps and density-entropy SPH. Seeds convert gas particles into sinks; gravity mass retains the unresolved reservoir. Merging is disabled in this model.")
            answers["black_holes"] = {
                "enabled": True, "seeding_mode": bh_mode,
                "radiative_efficiency": required_number("BH radiative efficiency [0 < epsilon_r < 1]", 1e-6, 1-1e-6),
                "feedback_coupling": required_number("BH thermal feedback coupling [0 <= epsilon_f <= 1]", 0, 1),
                "bondi_boost": required_number("Declared unresolved Bondi density boost [dimensionless; 1 is unboosted]", 1e-12, 1e12),
                "eddington_factor": required_number("Maximum inflow / Eddington inflow [dimensionless]", 1e-12, 1e6),
                "kernel_radius_proper_kpc": required_number("BH accretion/feedback and reservoir-donor aperture [proper kpc; resolve and test this scale]", 1e-9, 1000*box),
                "seed_mass_msun": required_number("BH seed mass [Msun; assembled from bounded donors]" if bh_mode=="density_exclusion_reservoir" else "BH seed mass [Msun; must fit within one gas particle]", 1, 1e15),
                "seed_density_hydrogen_cm3": required_number("Seed minimum proper hydrogen density [cm^-3]", 1e-12, 1e12),
                "seed_exclusion_proper_kpc": required_number("Seed exclusion radius around existing BHs [proper kpc]", 1e-9, 1000*box),
                "source_fraction": required_number("Maximum donor fraction and source change per substep [<=0.1; test reduction]", 1e-6, .1),
                "maximum_count": required_integer("Maximum global BH count [allocation cap; excess eligible seeds stop the run]", 1, 100000),
            }
    if answers["physics"] == "hydro" and answers["hydro_model"] == "star_formation":
        answers.update(gather_stellar_source_settings())
    answers["sidm"] = gather_sidm_settings(answers["physics"])
    answers["fof_linking_length"] = required_number("FoF linking length / mean particle spacing [dimensionless; between 0 and 1]", 1e-12, 1 - 1e-12)
    answers["halo_finder"] = required_choice("Halo finders [fof: GADGET FoF+SUBFIND; both: also run the Rockstar phase-space finder]", ["fof", "both"])
    answers["hmf_mass_definition"] = required_choice("Halo mass [fof: linked mass; m200c: 200×critical density; m200b: 200×mean density; mvir: virial overdensity]", ["fof", "m200c", "m200b", "mvir"])
    answers["hmf_min_particles"] = required_integer("Minimum halo particles [integer; resolution cut, not proof of convergence]", 2, 1000000)
    answers.update(required_values("Halo abundance bins", [
        ("hmf_min_mass_msun_h", "lower mass [Msun/h]", float, 1, 1e30),
        ("hmf_max_mass_msun_h", "upper mass [Msun/h]; must exceed lower mass", float,
         lambda row: math.nextafter(row['hmf_min_mass_msun_h'], math.inf), 1e30),
        ("hmf_mass_bins", "logarithmic mass bins", int, 3, 200)]))
    answers["density_grid"] = required_integer("Projected mass-map pixels per side [integer]", 32, 8192)
    answers["story_projections"] = required_choice("Projection planes [axes in the comoving box]", ["xy", "xz", "yz", "xy,xz,yz"])
    answers["slice_mode"] = required_choice("Projection depth [full_box integrates all mass; slab restricts depth]", ["full_box", "slab"])
    if answers["slice_mode"] == "slab":
        answers["slice_thickness_mpc_h"] = required_number("Line-of-sight slab thickness [Mpc/h]", 1e-6, box)
        answers["slice_center_mpc_h"] = required_number("Periodic line-of-sight slab centre [Mpc/h]", 0, box)
    answers["figure_dpi"] = required_integer("Raster export resolution [dots per inch]", 150, 1200)
    answers["figure_formats"] = required_choice("Export file formats", ["png", "png,pdf", "png,pdf,svg"])
    answers["density_scale_mode"] = required_choice("Density colour limits [global: identical across time; per_snapshot: rescaled each output]", ["global", "per_snapshot"])
    answers["density_low_percentile"] = required_number("Lower density display percentile [% of pixels]", 0, 99)
    answers["density_high_percentile"] = required_number("Upper density display percentile [% of pixels]", 1, 100)
    filament_options = ["hessian_proxy"]
    if shutil.which("mse") and shutil.which("skelconv"):
        filament_options.append("disperse_grid")
    answers["filament_method"] = required_choice(
        "Available filament extractors [hessian_proxy: local density-curvature estimate; disperse_grid: persistent topology]",
        filament_options)
    answers["filament_grid_size"] = required_integer("Filament density cells per side [integer]", 32, 160)
    answers["filament_smoothing_mpc_h"] = required_number("Gaussian density smoothing sigma [Mpc/h]", 1e-12, min(box, 8*box/answers['filament_grid_size']))
    if answers["filament_method"] == "hessian_proxy":
        answers["filament_density_percentile"] = required_integer("Occupied-cell density threshold [% percentile]", 50, 99)
        answers["filament_curvature_threshold_factor"] = required_number("Curvature threshold / RMS curvature [dimensionless]", 0.01, 10)
    else:
        answers["disperse_persistence_cut"] = required_number("Persistence cut [log10(1+rho/rho_mean)]", 1e-12, 10)
        answers["disperse_smoothing_cells"] = required_integer("Skeleton display smoothing iterations [integer]", 0, 20)
        answers['disperse_output_coordinate_representation']=required_choice(
            'Verified skelconv output coordinates [comoving_mpc_h or grid_vertex_coordinates]',
            ['comoving_mpc_h','grid_vertex_coordinates'])
    return answers


def gather_sidm_settings(physics):
    """Ask for the actual scattering model and native operator controls.

    A model file specifies angular cross sections; it is scientific input,
    not a run preset. Limits bound numerical work and approximation error.
    Existing native validators still check the complete model and resources.
    """
    from core.foundation import file_sha256
    if required_choice("Enable elastic self-interacting dark matter?", ["yes", "no"]) == "no":
        return {"enabled": False}
    result = {"schema": "native-sidm-configuration-v1", "enabled": True}
    while True:
        path = Path(ask_required("SIDM angular/differential model [JSON file path]")).expanduser()
        if path.is_file() and not path.is_symlink():
            result["physical_model"] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
            break
        print("Choose an existing regular model file, not a symbolic link.")
    backends = ["distributed_spatial"] if physics != "dm" else ["distributed_spatial", "all_pair_reference"]
    result["backend"] = required_choice("Pair search [all-pair reference requires one MPI rank]", backends)
    result["scheme"] = required_choice("Scattering operator", ["rare", "frequent", "hybrid"])
    result["kernel_kind"] = required_choice("Interaction kernel", ["uniform_ball", "cubic_spline"])
    result["particle_convention"] = required_choice("Angular particle convention [must match the model]", ["directed", "folded_identical"])
    while True:
        namespace = ask_required("SIDM event identity namespace [1–128 printable non-space ASCII characters]")
        if len(namespace) <= 128 and all(33 <= ord(c) <= 126 for c in namespace):
            result["event_identity_namespace"] = namespace
            break
        print("Use at most 128 printable ASCII characters without spaces.")
    # These endpoints bound accepted input; none supplies an answer/default.
    positive = math.nextafter(0., 1.)
    finite_max = sys.float_info.max
    below_one = math.nextafter(1., 0.)
    below_two = math.nextafter(2., 0.)
    angle_limit = math.pi/2 if result["particle_convention"] == "folded_identical" else math.pi
    controls = (
        ("kernel_support_proper_kpc", "Fine-particle kernel support [proper kpc]", positive, finite_max),
        ("cross_section_normalization_cm2_per_g", "Analytic cross-section normalization [cm^2/g; dimensional tables require 1, or 0 for a collisionless control]", 0., finite_max),
        ("maximum_pair_probability", "Maximum pair-event probability [fraction <1]", positive, below_one),
        ("macro_mass_relative_tolerance", "Scattering representative mass tolerance [relative]", 0., finite_max),
        ("conservation_relative_tolerance", "Pair momentum/energy conservation tolerance [relative]", positive, finite_max),
        ("energy_absolute_floor", "Energy denominator floor [Msun (km/s)^2, physical convention before canonical conversion]", positive, finite_max),
        ("momentum_absolute_floor", "Momentum denominator floor [Msun km/s, physical convention before canonical conversion]", positive, finite_max),
        ("cdf_absolute_tolerance", "Angular CDF inversion tolerance [absolute probability]", positive, finite_max),
        ("kernel_origin_relative_bound", "Kernel origin approximation bound [relative <1]", positive, below_one),
        ("maximum_particle_probability", "Maximum cumulative particle-event probability [fraction <1]", positive, below_one),
        ("maximum_transfer_depth", "Maximum transfer optical depth per substep [dimensionless <1]", positive, below_one),
        ("maximum_neglected_moment_ratio", "Maximum neglected angular moment ratio [dimensionless <2]", positive, below_two),
        ("split_angle_radians", "Rare/frequent angular split [radians, inside the selected angular domain]", positive, math.nextafter(angle_limit, 0.)),
        ("clock_absolute_tolerance_seconds", "Physical scattering-clock quadrature tolerance [seconds]", positive, finite_max),
        ("clock_relative_tolerance", "Scattering-clock quadrature tolerance [relative <1]", positive, below_one),
        ("maximum_scale_factor_fractional_change", "Maximum fractional scale-factor change per source step [dimensionless]", positive, finite_max),
        ("exchange_memory_mib", "Per-rank pair-exchange workspace [MiB; must fit the native allocation]", positive, finite_max),
        ("coarse_kernel_support_proper_kpc", "Coarse-particle guard kernel support [proper kpc]", positive, finite_max),
        ("maximum_kernel_coarse_support_fraction", "Maximum coarse support in a fine-particle kernel [fraction]", 0., 1.),
        ("maximum_omitted_scattering_upper_bound", "Maximum cumulative omitted-scattering upper bound [dimensionless]", 0., finite_max),
    )
    for key, label, low, high in controls:
        result[key] = required_number(label, low, high)
    for key in ("maximum_particles", "maximum_events_per_step", "maximum_substeps", "maximum_clock_panels", "maximum_local_pairs", "maximum_matching_rounds"):
        result[key] = required_integer(key.replace("_", " ") + " [exact count]", 2 if key == "maximum_clock_panels" else 1, 2**31-1)
    for key in ("operator_seed", "maximum_pair_visits", "maximum_exchange_records"):
        # Never round seeds or identity/work counters through floating point.
        result[key] = required_integer(key.replace("_", " ") + " [exact uint64]", 0 if key == "operator_seed" else 1, 2**64-1)
    selection = required_choice("Saved event population", ["all", "aggregate_only", "hashed_particle_ids"])
    result["event_audit_population"] = {"selection": selection}
    if selection == "hashed_particle_ids":
        modulus = required_integer("Exact particle-ID hash modulus [integer]", 2, 2**31-1)
        result["event_audit_population"].update(modulus=modulus, residue=required_integer("Retained hash residue [integer]", 0, modulus-1))
    return result


def gather_stellar_source_settings():
    """Collect optional delayed stellar returns and chemistry in physical units.

    The selected stellar model already records IMF/yield assumptions and its
    prompt return fraction. Native validation checks that fraction against the
    star-formation model; no new table is invented by these questions.
    """
    from core.foundation import file_sha256

    def selected_file(label):
        while True:
            path = Path(ask_required(label)).expanduser()
            if path.is_file() and not path.is_symlink():
                return str(path.resolve())
            print("Choose an existing regular file, not a symbolic link.")

    def enabled(label):
        return required_choice(label, ["yes", "no"]) == "yes"

    stellar = {"enabled": enabled("Enable delayed stellar mass/metal returns?")}
    chemistry = {"enabled": False}
    if not stellar["enabled"]:
        return {"stellar_evolution": stellar, "chemistry": chemistry}
    manifest = selected_file("Compiled stellar return model manifest [JSON file path]")
    stellar.update(
        model_manifest=manifest, model_sha256=file_sha256(manifest),
        kernel_radius_proper_kpc=required_number("Stellar return deposition radius [proper kpc]", 1e-12, 1e12),
        source_fraction=required_number("Maximum stellar source change per subcycle [fraction; <=0.1]", 1e-12, .1),
        maximum_count=required_integer("Maximum stellar population particles [allocation bound]", 1, 10000000),
        maximum_subcycles=required_integer("Maximum stellar return subcycles [iteration bound]", 1, 1000000),
    )
    chemistry["enabled"] = enabled("Enable native Grackle chemistry and cooling?")
    if chemistry["enabled"]:
        network = required_integer("Primordial network [0 equilibrium, 1 atomic, 2 H2, 3 H2/HD]", 0, 3)
        metal = enabled("Include metal cooling?")
        uv = enabled("Include a UV background? Select matching data below")
        data = selected_file("Grackle cooling table [HDF5 file path]")
        library = selected_file("Built Grackle precision64 library [file path]")
        chemistry.update(
            primordial_network=network, metal_cooling=metal, uv_background=uv,
            data_file=data, data_sha256=file_sha256(data),
            library_file=library, library_sha256=file_sha256(library),
            upstream_revision="af7939494ce65007887ada7b98d1813df6843346",
        )
        # Number fractions count H species per H nucleus, He species per He
        # nucleus, and D species per H nucleus. Molecular species count once
        # as molecules; the native validator applies their nuclear factors.
        ratio = required_number("Primordial D/H number ratio [dimensionless]", 1e-15, 1e-3) if network == 3 else 0.
        chemistry["deuterium_hydrogen_number_ratio"] = ratio
        species = ("HI", "HII", "HeI", "HeII", "HeIII", "HM", "H2I", "H2II", "DI", "DII", "HDI")
        active = set(species[:5])
        if network >= 2:
            active.update(("HM", "H2I", "H2II"))
        if network == 3:
            active.update(("DI", "DII", "HDI"))
        chemistry["initial_species_number_fractions"] = {
            name: required_number("Initial " + name + " number fraction [relative to " +
                ("He nuclei" if name.startswith("He") else "H nuclei") + "]", 0., 1.) if name in active else 0.
            for name in species
        }
    return {"stellar_evolution": stellar, "chemistry": chemistry}


def build_config(answers: dict) -> tuple[dict, Path]:
    """Assemble explicit LCDM/EDE settings and validate the selected spectrum provenance."""
    if not isinstance(answers, dict):
        raise ValueError("--answers must contain a JSON object")
    equal_steps=answers.get('force_equal_timesteps',False)
    if type(equal_steps) is not bool:raise ValueError('force_equal_timesteps requires an explicit boolean')
    native_trees=answers.get('native_mergertree',False)
    if type(native_trees) is not bool:raise ValueError('native_mergertree requires an explicit boolean')
    diagnostics=answers.get('output_acceleration_diagnostics',False)
    if type(diagnostics)is not bool:raise ValueError('output_acceleration_diagnostics requires an explicit boolean')
    required_answers = {
        "cosmology", "name", "box_mpc_h", "particles_per_dimension", "seed",
        "schedule", "mpi_ranks", "num_files_per_snapshot", "pm_grid", "asmth",
        "multipole_order", "softening_kpc_h", "softening_max_physical_kpc_h",
        "max_size_timestep_ln_a", "err_tol_force_acc", "err_tol_int_accuracy", "err_tol_theta",
        "restart_every_seconds", "time_limit_cpu_seconds", "hmf_mass_definition",
        "hmf_min_mass_msun_h", "hmf_max_mass_msun_h", "hmf_mass_bins", "hmf_min_particles",
        "density_grid", "filament_method", "filament_grid_size", "filament_smoothing_mpc_h",
        "slice_mode", "story_projections", "figure_dpi", "figure_formats",
        "density_scale_mode", "density_low_percentile", "density_high_percentile",
    }
    missing = sorted(required_answers - set(answers))
    if missing:
        raise ValueError("--answers must explicitly provide: " + ", ".join(missing))
    optional_answers = {
        "deferred_outputs", "highlight_redshifts", "physics", "accuracy", "halo_finder", "graphs", "stellar_search", "stellar_history", "zoom_regions", "native_flow_window_comparison",
        "force_equal_timesteps", "native_mergertree", "consistent_trees", "rockstar_wall_seconds", "fof_linking_length", "allow_incomplete_analysis", "restart_source_capsule", "restart_exercise_pause_step", "black_holes", "sidm", "stellar_evolution", "chemistry", "min_size_timestep_ln_a", "research_spec", "inference_diagnostics", "radial_shapes",
        "cpu_threads", "omp_threads", "build_jobs", "memory_limit_gb", "courant_factor", "output_acceleration_diagnostics",
        "hydro_model", "treecool_file", "galaxy_min_star_particles", "galaxy_min_dm_particles",
        "filament_density_percentile", "filament_curvature_threshold_factor",
        "disperse_persistence_cut", "disperse_smoothing_cells",
        "disperse_density_smoothing_cells", "disperse_output_coordinate_representation",
        "disperse_timeout_s", "disperse_memory_gib",
        "blender_sequence_max_points", "blender_frames_per_snapshot", "blender_render_quality", "movie_fps",
        "catalogue_min_particles", "hmf_diagnostic_thresholds",
        "ic_engine_hydro", "lpt_order_dm", "lpt_order_hydro", "ic_engine_dm",
        "particle_mass_msun_h", "mass_threshold_100_particles_msun_h",
        "mass_threshold_400_particles_msun_h", "mass_threshold_1000_particles_msun_h",
        "slice_thickness_mpc_h", "slice_center_mpc_h",
        *HYDRO_STAR_FORMATION_PARAMETERS, *HYDRO_SPH_PARAMETERS,
    }
    unknown = sorted(set(answers) - required_answers - optional_answers)
    if unknown:
        raise ValueError("Unsupported --answers key(s): " + ", ".join(unknown))
    manifest = spectrum_path(str(answers["cosmology"]))
    info = load_json(manifest)
    linear = info["parameters"]
    solver_parameters = info["class_parameters"]["start"]
    start = float(linear["z_start"])
    schedule = redshift_schedule(str(answers["schedule"]), start)
    from core.epochs import highlight_redshifts
    highlights = highlight_redshifts(str(answers["schedule"]), start, answers.get("highlight_redshifts", []))
    box = float(answers["box_mpc_h"])
    if not math.isfinite(box) or box <= 0:
        raise ValueError("Box length must be a positive finite value in Mpc/h")
    # This is also the effective fine resolution of an existing MUSIC zoom,
    # not a claim that a uniform N^3 load can fit the selected allocation.
    # Native census/resource and MUSIC handoff checks remain launch gates.
    n = integer_value(answers["particles_per_dimension"], "Particles per side N", 32, 2**20)
    seed = integer_value(answers["seed"], "Random seed", 0, 2147483647)
    mpi_ranks = integer_value(answers["mpi_ranks"], "MPI ranks", 1, 1024)
    snapshot_files = integer_value(answers["num_files_per_snapshot"], "Snapshot HDF5 files", 1, mpi_ranks)
    pm_grid = integer_value(answers["pm_grid"], "Particle-mesh cells per side", 64, 16384)
    multipole_order = integer_value(answers["multipole_order"], "Tree-force multipole order", 1, 5)
    density_grid = integer_value(answers["density_grid"], "Dark-matter map pixels per side", 32, 8192)
    filament_grid_size = integer_value(answers["filament_grid_size"], "Filament field cells per side", 32, 160)
    physics = str(answers.get("physics", "dm")).lower()
    if physics not in {"dm", "hydro"}:
        raise ValueError("physics must be dm or hydro")
    mass_min = float(answers["hmf_min_mass_msun_h"])
    mass_max = float(answers["hmf_max_mass_msun_h"])
    mass_bins = integer_value(answers["hmf_mass_bins"], "Number of fixed HMF mass bins", 3, 200)
    hmf_min_particles = integer_value(answers["hmf_min_particles"], "Minimum particles per HMF halo", 2, 1000000)
    if not (math.isfinite(mass_min) and math.isfinite(mass_max) and 0 < mass_min < mass_max):
        raise ValueError("HMF mass limits must be finite, positive, and strictly increasing")
    filament_method = str(answers["filament_method"]).lower()
    if filament_method not in {"disperse_grid", "hessian_proxy"}:
        raise ValueError("filament_method must be disperse_grid or hessian_proxy")
    smoothing_mpc_h = float(answers["filament_smoothing_mpc_h"])
    if not math.isfinite(smoothing_mpc_h) or smoothing_mpc_h < 0:
        raise ValueError("filament_smoothing_mpc_h must be finite and non-negative")
    if filament_method == "hessian_proxy" and smoothing_mpc_h <= 0:
        raise ValueError("filament_smoothing_mpc_h must be positive for the Hessian filament proxy")
    if filament_method == "disperse_grid":
        from core.native import native_engine_directory
        disperse_dir = None
        try:
            disperse_dir = native_engine_directory('disperse')
        except Exception:
            pass
        mse_ok = shutil.which("mse") or (disperse_dir and (disperse_dir / "build/src/mse/mse").is_file())
        skelconv_ok = shutil.which("skelconv") or (disperse_dir and (disperse_dir / "build/src/mse/skelconv").is_file())
        if not (mse_ok and skelconv_ok):
            raise RuntimeError("DisPerSE mse/skelconv executables are missing; install them or explicitly choose hessian_proxy")
        required_filament_answers = {"disperse_persistence_cut", "disperse_smoothing_cells","disperse_output_coordinate_representation"}
        missing_filament_answers = sorted(required_filament_answers - set(answers))
        if missing_filament_answers:
            raise ValueError("--answers must explicitly provide: " + ", ".join(missing_filament_answers))
        persistence_cut = float(answers["disperse_persistence_cut"])
        smoothing_iterations = integer_value(answers["disperse_smoothing_cells"], "DisPerSE display-line smoothing iterations", 0, 20)
        if not math.isfinite(persistence_cut) or not 0 < persistence_cut <= 10:
            raise ValueError("disperse_persistence_cut must be finite and in (0, 10]")
        if answers['disperse_output_coordinate_representation'] not in ('comoving_mpc_h','grid_vertex_coordinates'):
            raise ValueError('disperse_output_coordinate_representation must be comoving_mpc_h or grid_vertex_coordinates')
    else:
        required_filament_answers = {"filament_density_percentile", "filament_curvature_threshold_factor"}
        missing_filament_answers = sorted(required_filament_answers - set(answers))
        if missing_filament_answers:
            raise ValueError("--answers must explicitly provide: " + ", ".join(missing_filament_answers))
        percentile = float(answers["filament_density_percentile"])
        curvature_factor = float(answers["filament_curvature_threshold_factor"])
        if not math.isfinite(percentile) or not percentile.is_integer() or not 50 <= percentile <= 99:
            raise ValueError("filament_density_percentile must be an integer in [50, 99]")
        if not math.isfinite(curvature_factor) or not 0 < curvature_factor <= 10:
            raise ValueError("filament_curvature_threshold_factor must be finite and in (0, 10]")
        if smoothing_mpc_h <= 0:
            raise ValueError("Hessian proxy extraction requires positive physical density smoothing")
    softening = float(answers["softening_kpc_h"])
    if not math.isfinite(softening) or softening <= 0:
        raise ValueError("softening_kpc_h must be finite and positive")
    physical_softening = float(answers["softening_max_physical_kpc_h"])
    if not math.isfinite(physical_softening) or not 0 < physical_softening <= softening:
        raise ValueError("maximum physical softening must be positive and no larger than comoving softening")
    gadget_numerics = {
        key: float(answers[key]) for key in (
            "asmth", "max_size_timestep_ln_a", "err_tol_force_acc",
            "err_tol_int_accuracy", "err_tol_theta", "time_limit_cpu_seconds",
        )
    }
    if len(schedule) > 1:
        min_spacing = min(math.log((1.0 + left) / (1.0 + right))
                          for left, right in zip(schedule, schedule[1:]))
        if gadget_numerics["max_size_timestep_ln_a"] > min_spacing:
            gadget_numerics["max_size_timestep_ln_a"] = round(min_spacing * 0.98, 6)
    if any(not math.isfinite(value) or value <= 0 for value in gadget_numerics.values()):
        raise ValueError("GADGET numerical controls and time limit must be finite and positive")
    minimum_step = answers.get("min_size_timestep_ln_a", 0.0)
    if isinstance(minimum_step, bool):
        raise ValueError("min_size_timestep_ln_a must be a finite non-negative number")
    minimum_step = float(minimum_step)
    if not math.isfinite(minimum_step) or not 0 <= minimum_step <= gadget_numerics["max_size_timestep_ln_a"]:
        raise ValueError("min_size_timestep_ln_a must lie between zero and max_size_timestep_ln_a")
    restart_seconds = integer_value(
        answers["restart_every_seconds"], "Restart checkpoint interval [seconds]", 60, 604800,
    )
    if answers["slice_mode"] not in {"full_box", "slab"}:
        raise ValueError("slice_mode must be full_box or slab")
    if answers["slice_mode"] == "slab":
        if "slice_thickness_mpc_h" not in answers or "slice_center_mpc_h" not in answers:
            raise ValueError("slab maps require explicit slice_thickness_mpc_h and slice_center_mpc_h")
        if float(answers["slice_thickness_mpc_h"]) > box:
            raise ValueError("slice_thickness_mpc_h cannot exceed the periodic box length")
    low_percentile = float(answers["density_low_percentile"])
    high_percentile = float(answers["density_high_percentile"])
    if not (math.isfinite(low_percentile) and math.isfinite(high_percentile)
            and 0 <= low_percentile < high_percentile <= 100):
        raise ValueError("Map display percentiles must satisfy 0 <= lower < upper <= 100")
    figure_formats = [value.strip().lower() for value in str(answers["figure_formats"]).split(",")]
    if not figure_formats or any(value not in {"png", "pdf", "svg"} for value in figure_formats):
        raise ValueError("figure_formats must be a comma-separated selection of png, pdf, and svg")
    figure_formats = list(dict.fromkeys(figure_formats))
    if answers["density_scale_mode"] not in {"global", "per_snapshot"}:
        raise ValueError("density_scale_mode must be global or per_snapshot")
    story_projections = [value.strip() for value in str(answers["story_projections"]).split(",")]
    if not story_projections or any(value not in {"xy", "xz", "yz"} for value in story_projections):
        raise ValueError("story_projections must select from xy, xz, and yz")
    if len(set(story_projections)) != len(story_projections):
        raise ValueError("story_projections cannot repeat a projection")
    figure_dpi = integer_value(answers["figure_dpi"], "Raster figure resolution [DPI]", 150, 1200)
    # N-GenIC samples the full 3D Fourier cube; the existing validator uses
    # a margin beyond sqrt(3)*k_Nyquist to avoid edge extrapolation.
    spectrum = np.loadtxt(manifest.parent / "pk_class.txt", comments="#", ndmin=2)
    k_fund = 2 * math.pi / box
    k_nyq = math.pi * n / box
    required_min = k_fund / 8
    required_max = max(2.5 * k_nyq, 1.35 * math.sqrt(3) * k_nyq)
    if spectrum[0, 0] > required_min or spectrum[-1, 0] < required_max:
        raise ValueError(
            f"Selected cosmology spans k={spectrum[0,0]:g}–{spectrum[-1,0]:g} h/Mpc; "
            f"this box and mesh require coverage through {required_min:g}–{required_max:g}. "
            "Rerun Component 1 with a larger k_max or choose a larger box/smaller mesh."
        )
    cfg = yaml.safe_load(get_config_path().read_text(encoding="utf-8"))
    if 'restart_source_capsule' in answers:
        if type(answers['restart_source_capsule']) is not bool:raise ValueError('Restart source capsule setting must be boolean')
        cfg['simulation']['restart_source_capsule']=answers['restart_source_capsule']
    if 'restart_exercise_pause_step' in answers:
        step=answers['restart_exercise_pause_step']
        if type(step) is not int or not 0<=step<=1000000 or answers.get('restart_source_capsule') is not True:
            raise ValueError('Explicit bounded restart exercise step and full native restart state diagnostics required')
        cfg['simulation']['restart_exercise_pause_step']=step
    if 'sidm' in answers:
        from core.sidm_physics import sidm_native_parameters
        sidm_native_parameters(answers['sidm'],physics=physics)
        cfg['sidm']=dict(answers['sidm'])
    for section in ('stellar_evolution','chemistry'):
        if section in answers:
            if not isinstance(answers[section],dict):raise ValueError(section+' source settings must be an explicit mapping')
            cfg[section]=dict(answers[section])
    if "black_holes" in answers:
        from black_holes import native_parameters
        native_parameters(answers["black_holes"])
        if answers["black_holes"].get("enabled", False) and answers["black_holes"]["seeding_mode"] not in {"density_exclusion","density_exclusion_reservoir"}:
            raise ValueError("Generated uniform-grid ICs require an explicit dense-gas BH seeding mode; ic_particles is currently a direct native validation mode")
        cfg["black_holes"] = dict(answers["black_holes"])
    run_dir = named_path(str(answers["name"]), runtime_paths().runs_root)
    cfg["project"].update(name=str(answers["name"]), runs_dir=str(run_dir))
    cfg["spectrum"].update(source="modular", file=str(manifest), format="class_pk",
                           k_unit="h/Mpc", p_unit="(Mpc/h)^3", redshift=start)
    h = float(linear["H0"]) / 100.0
    cfg["cosmology"].update(model=info["model"], engine="axiclass" if info["model"] == "ede" else "class",
                            H0=float(linear["H0"]), omega_b=float(linear["Omega_b"]) * h**2,
                            omega_cdm=float(linear["Omega_cdm"]) * h**2,
                            A_s=float(linear["A_s"]), n_s=float(linear["n_s"]),
                            tau_reio=float(linear["tau_reio"]),
                            N_ur=float(solver_parameters["N_ur"]),
                            T_cmb=float(solver_parameters["T_cmb"]),
                            N_ncdm=0, m_ncdm_eV=0.0)
    cfg["ede"]["enabled"] = info["model"] == "ede"
    if info["model"] == "ede":
        cfg["ede"].update(
            parameterization="axion",
            f_ede=float(linear["f_ede"]),
            z_c=10.0 ** float(linear["log10_zc"]),
            theta_i=float(linear["theta_i"]),
            n_axion=float(linear["n_axion"]),
        )
    cfg['simulation']['force_equal_timesteps']=equal_steps
    cfg["simulation"].update(box_mpc_h=box, particles_per_dimension=n,
                             start_redshift=start, start_redshift_guess=start,
                             random_seed=seed, physics=physics,
                             lpt_order_dm=answers.get("lpt_order_dm", 2),
                             lpt_order_hydro=answers.get("lpt_order_hydro", 3),
                             ic_engine_dm=answers.get("ic_engine_dm", "ngenic"),
                             ic_engine_hydro=answers.get("ic_engine_hydro", "monofonic"),
                             synchronized_outputs=True, output_format="hdf5",
                             num_files_per_snapshot=snapshot_files,
                             pm_grid=pm_grid, asmth=gadget_numerics["asmth"],
                             multipole_order=multipole_order,
                             softening_kpc_h=softening,
                             softening_max_physical_kpc_h=physical_softening,
                             max_size_timestep_ln_a=gadget_numerics["max_size_timestep_ln_a"],
                             min_size_timestep_ln_a=minimum_step,
                             err_tol_force_acc=gadget_numerics["err_tol_force_acc"],
                             err_tol_int_accuracy=gadget_numerics["err_tol_int_accuracy"],
                             err_tol_theta=gadget_numerics["err_tol_theta"],
                             restart_every_seconds=restart_seconds,
                             time_limit_cpu_seconds=gadget_numerics["time_limit_cpu_seconds"],
                             snapshot_z_max=schedule[0],
                             snapshot_z_min=schedule[-1], snapshot_dz=1.0,
                             snapshot_redshift_list=schedule, highlight_redshifts=highlights, end_redshift=schedule[-1])
    cfg["simulation"]["accuracy"] = answers.get("accuracy", "custom")
    cfg['simulation']['output_acceleration_diagnostics']=diagnostics
    cfg["hardware"]["mpi_ranks"] = mpi_ranks
    for key in ("cpu_threads", "omp_threads", "build_jobs"):
        if key in answers:
            cfg["hardware"][key] = integer_value(answers[key], key, 1, 1024)
    if "memory_limit_gb" in answers:
        memory = float(answers["memory_limit_gb"])
        if isinstance(answers["memory_limit_gb"], bool) or not math.isfinite(memory) or memory <= 0:
            raise ValueError("memory_limit_gb must be finite and positive")
        cfg["hardware"]["memory_limit_gb"] = memory
    if answers.get('native_flow_window_comparison') is not None:
        from core.music.flow import validate_original_flow_launch_request
        validate_original_flow_launch_request(answers['native_flow_window_comparison'])
        if physics!='hydro':raise ValueError('Native flow window comparison requires hydro answers')
        cfg['simulation']['native_flow_window_comparison']=json.loads(json.dumps(answers['native_flow_window_comparison']))
    cfg["hydro"]["courant_factor"] = answers.get("courant_factor", 0.1) if physics == "hydro" else None
    if physics == "hydro":
        gas_model = answers.get("hydro_model", "star_formation")
        if gas_model not in {"adiabatic", "cooling", "star_formation"}:
            raise ValueError("hydro_model must be adiabatic, cooling or star_formation")
        cfg["hydro"].update(cooling=gas_model != "adiabatic", uv_background=gas_model != "adiabatic",
                            star_formation=gas_model == "star_formation")
        for key in (*HYDRO_STAR_FORMATION_PARAMETERS, *HYDRO_SPH_PARAMETERS, "treecool_file"):
            if key in answers:
                cfg["hydro"][key] = answers[key]
        if cfg["hydro"].get("treecool_file") is not None:
            raw_path = cfg["hydro"]["treecool_file"]
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("treecool_file must be a non-empty file path or null")
            cfg["hydro"]["treecool_file"] = str(Path(raw_path).expanduser().resolve())
        validate_hydro_configuration(cfg, require_runtime_assets=False)
        for answer_key, config_key, default in (("galaxy_min_star_particles", "min_star_particles", 100),
                                                ("galaxy_min_dm_particles", "min_dm_particles", 300)):
            cfg["galaxies"][config_key] = integer_value(answers.get(answer_key, default), answer_key, 1, 1000000000)
    finder = answers.get("halo_finder", "fof")
    if finder not in {"fof", "rockstar", "both"}:
        raise ValueError("halo_finder must be fof, rockstar or both")
    linking=answers.get('fof_linking_length',.2)
    if type(linking) not in (int,float) or not math.isfinite(linking) or not 0<linking<1:
        raise ValueError('fof_linking_length must be a finite fraction in (0,1)')
    cfg["halos"].update(fof=True, subfind=True, rockstar=finder in {"rockstar", "both"}, consistent_trees=False,
                         min_particles=hmf_min_particles, fof_linking_length=float(linking))
    cfg['halos']['native_mergertree']=native_trees
    consistent=answers.get('consistent_trees',False)
    if type(consistent) is not bool or (consistent and finder not in {'rockstar','both'}):
        raise ValueError('consistent_trees requires an explicit boolean and Rockstar')
    cfg['halos']['consistent_trees']=consistent
    rockstar_wall=answers.get('rockstar_wall_seconds',3600.)
    if type(rockstar_wall) not in (int,float) or not math.isfinite(rockstar_wall) or rockstar_wall<=0:
        raise ValueError('Positive finite Rockstar execution budget required')
    cfg['halos']['rockstar_wall_seconds']=float(rockstar_wall)
    cfg["galaxies"]["enabled"] = physics == "hydro"
    from core.setup_walkthrough import REQUESTS
    deferred = answers.get('deferred_outputs', [])
    if not isinstance(deferred, list) or any(p not in REQUESTS for p in deferred):
        raise ValueError('deferred_outputs must list specialized measurement products')
    cfg['analysis']['deferred_outputs'] = list(dict.fromkeys(deferred))
    cfg["analysis"].update(
        all=False,
        products=["hmf"],
        hmf_mass_definition=str(answers["hmf_mass_definition"]).lower(),
        hmf_min_particles=hmf_min_particles,
        hmf_mass_range_msun_h=[mass_min, mass_max],
        mass_bins=mass_bins,
        density_grid=density_grid,
    )
    visualization = cfg["visualization"]
    cell_size = box / filament_grid_size
    visualization.update(
        figure_formats=figure_formats,
        figure_dpi=figure_dpi,
        map_fields=["dm", "gas", "gas_temperature", "stars"] if physics == "hydro" else ["dm"],
        map_projections=story_projections,
        density_scale_mode=str(answers["density_scale_mode"]),
        density_display_percentiles=[low_percentile, high_percentile],
        density_cmap="cividis",
        filament_color="#b88045",
        structure_story_figure=True,
        structure_story_projections=story_projections,
        structure_map_overlays=["filaments"],
        filament_overlay=True,
        filament_method=filament_method,
        filament_grid_size=filament_grid_size,
        filament_smoothing_mpc_h=smoothing_mpc_h,
        disperse_density_smoothing_cells=smoothing_mpc_h / cell_size,
        blender_render=False,
    )
    if filament_method == "disperse_grid":
        visualization["disperse_persistence_cut"] = float(answers["disperse_persistence_cut"])
        visualization["disperse_smoothing_cells"] = smoothing_iterations
        visualization['disperse_output_coordinate_representation']=answers['disperse_output_coordinate_representation']
        if not shutil.which("mse") and disperse_dir:
            visualization["mse_executable"] = str(disperse_dir / "build/src/mse/mse")
        if not shutil.which("skelconv") and disperse_dir:
            visualization["skelconv_executable"] = str(disperse_dir / "build/src/mse/skelconv")
    else:
        visualization["disperse_smoothing_cells"] = None
        visualization["disperse_persistence_cut"] = None
        visualization["filament_density_percentile"] = integer_value(
            answers["filament_density_percentile"], "Hessian proxy occupied-cell density percentile", 50, 99,
        )
        visualization["filament_curvature_threshold_factor"] = float(answers["filament_curvature_threshold_factor"])
    if answers["slice_mode"] == "slab":
        visualization["slice_thickness_mpc_h"] = float(answers["slice_thickness_mpc_h"])
        visualization["slice_center_mpc_h"] = float(answers["slice_center_mpc_h"])
        visualization["slice_mode"] = "slab"
    else:
        visualization["slice_thickness_mpc_h"] = None
        visualization["slice_center_mpc_h"] = None
        visualization["slice_mode"] = "full_box"
    validate_supported_options(cfg)
    return cfg, run_dir


def run_gadget_main() -> None:
    """Run the physics engine; later stages can be launched independently."""
    parser = argparse.ArgumentParser(description='Choose a Component 1 cosmology, generate ICs, and evolve with GADGET-4.')
    parser.add_argument("--answers", type=Path, help="JSON object with simulation answers for repeatable runs")
    parser.add_argument("--draft", type=Path, help="Persist and resume interactive configuration at this JSON path")
    parser.add_argument("--cosmology", help="required spectrum.json or its run folder")
    parser.add_argument('--run-dir',type=Path,help='Explicit output directory instead of the suite run-name directory')
    parser.add_argument('--resume',action='store_true',help='Continue the same frozen native run from verified rank checkpoints')
    parser.add_argument('--recover-checkpoint',action='store_true',help='Complete verified checkpoint recovery before native resume')
    parser.add_argument("--dry-run", action="store_true", help="validate and stage a GADGET plan without evolving")
    parser.add_argument("--ic-only", action="store_true", help="generate and validate ICs, then stop")
    parser.add_argument('--existing-ic',type=Path,help='Use delivered existing ICs instead of generating a new load')
    parser.add_argument('--native-generation-report',type=Path,help='Require revalidated native MUSIC parent/zoom handoff')
    parser.add_argument('--validated-public-ic',type=Path,help='Bind an original validated public hydro IC-only generation and its transfer source; all native IC gates run again')
    parser.add_argument('--zoom-softening-mass-ratios',help='Explicit sorted coarse/fine-DM mass ratios, comma separated')
    parser.add_argument('--prepare-only',action='store_true',help='Validate an existing native handoff and build GADGET without evolution')
    parser.add_argument('--evolution-only',action='store_true',help='Evolve the selected physics; leave saved-output analysis to its workflow stages')
    parser.add_argument('--research-spec', type=Path, help='Versioned research specification; gates resolved-stream feasibility before launch')
    parser.add_argument('--numerical-controls', type=Path, help='Execute a declared analytic/native reliability matrix')
    parser.add_argument('--resume-controls', action='store_true', help='Reuse a current completed matching control matrix')
    args = parser.parse_args()
    if args.resume:
        if not args.run_dir or any((args.answers,args.cosmology,args.existing_ic,args.native_generation_report,args.research_spec,
                args.dry_run,args.ic_only,args.prepare_only,args.evolution_only,args.numerical_controls,args.resume_controls,args.zoom_softening_mass_ratios,args.validated_public_ic)):
            parser.error('--resume requires only --run-dir and optional --recover-checkpoint; frozen run inputs are retained')
        from core.evolution import resume_gadget_run
        from core.native import native_run_lock
        with native_run_lock(args.run_dir,operation='public native checkpoint continuation'):
            result=resume_gadget_run(args.run_dir,recover_checkpoint=args.recover_checkpoint)
        print(__import__('json').dumps(result,indent=2))
        return
    if args.recover_checkpoint:parser.error('--recover-checkpoint requires --resume')
    if args.numerical_controls:
        if not args.run_dir or any((args.answers,args.cosmology,args.existing_ic,args.native_generation_report,args.research_spec,args.dry_run,args.ic_only,args.prepare_only,args.zoom_softening_mass_ratios,args.validated_public_ic)):
            parser.error('--numerical-controls requires --run-dir and cannot be mixed with production run options')
        from core.research.controls import run_control_campaign
        report=run_control_campaign(load_json(args.numerical_controls),args.run_dir,resume=args.resume_controls)
        print(__import__('json').dumps(report,indent=2))
        if report['status']!='PASS':raise SystemExit(1)
        return
    if args.resume_controls:parser.error('--resume-controls requires --numerical-controls')
    stage_only=args.dry_run or args.ic_only or args.prepare_only or args.evolution_only
    if args.native_generation_report and not args.existing_ic:
        parser.error('--native-generation-report requires --existing-ic')
    if args.validated_public_ic and (not args.existing_ic or args.native_generation_report or args.ic_only or args.prepare_only):
        parser.error('--validated-public-ic requires --existing-ic and its separate public IC route; omit MUSIC report,IC-only andprepare-only')
    if args.prepare_only and (not args.existing_ic or not args.native_generation_report or args.dry_run or args.ic_only):
        parser.error('--prepare-only requires --existing-ic --native-generation-report; omit dry-run/IC-only')
    if args.existing_ic and args.ic_only:
        parser.error('--existing-ic cannot be combined with --ic-only')
    if args.zoom_softening_mass_ratios and not args.native_generation_report:
        parser.error('--zoom-softening-mass-ratios requires --native-generation-report')
    if args.answers:
        answers = load_json(args.answers)
    else:
        from core.setup_walkthrough import collect_simulation
        answers = collect_simulation(spectrum_path(args.cosmology), args.draft, outputs=not stage_only)
    if args.research_spec:
        if 'research_spec' in answers:
            raise ValueError('Supply research_spec in answers or --research-spec, not both')
        answers['research_spec'] = load_json(args.research_spec)
    from commands import (choose_products, process_run, parse_products, choose_stellar_search,
                              validate_stellar_search)
    products = answers.get("graphs")
    if products is None and not stage_only:
        products = choose_products()
    if products is None and answers.get('radial_shapes') is not None:
        raise ValueError('A radial_shapes request requires selecting the radial_shapes product')
    if products is not None:
        products = parse_products(products)
        answers['graphs'] = products
    if products is not None:
        if not args.answers and not stage_only:
            for product, key in (("stellar_histories", "stellar_history"),
                                 ("zoom_contamination", "zoom_regions"),
                                 ("inference_diagnostics", "inference_diagnostics"),
                                 ("radial_shapes", "radial_shapes")):
                if product in products and product not in answers.get('deferred_outputs', []) and answers.get(key) is None:
                    while True:
                        path = Path(ask_required("Your " + product + " measurement request [JSON file path]")).expanduser()
                        try:
                            answers[key] = load_json(path)
                            break
                        except (OSError, ValueError) as error:
                            print(error)
        # Missing-request policy below distinguishes explicit saved settings
        # from a later stellar host selection that needs actual catalogues.
        requests = validate_analysis_requests(products,
            **{key: answers.get(key) for key in ("stellar_search", "stellar_history",
                "zoom_regions", "inference_diagnostics", "radial_shapes")},
            allow_incomplete=True)
        answers.update({key: value for key, value in requests.items() if value is not None})
    cfg, run_dir = build_config(answers)
    if args.run_dir:
        run_dir=args.run_dir.expanduser().resolve()
        cfg['project']['runs_dir']=str(run_dir)
    if args.native_generation_report:
        generation_path=args.native_generation_report.expanduser().resolve()
        # The finite-epoch materializer owns its second-order species flow;
        # original MUSIC generations keep the order actually configured.
        generation_info=json.loads(generation_path.read_text())
        if generation_info.get('schema')=='twofluid-native-ic-generation-v1':
            native_order=2
        else:
            from zoom import _configuration
            native_order=2 if _configuration(generation_path.parent/'music.conf').getboolean('setup','use_2LPT',fallback=False) else 1
        cfg['simulation'].update(ic_engine_hydro='native_music',lpt_order_hydro=native_order,
                                 native_generation_report=str(args.native_generation_report.expanduser().resolve()))
        if args.zoom_softening_mass_ratios:
            cfg['simulation']['zoom_softening_mass_ratios']=[float(value) for value in args.zoom_softening_mass_ratios.split(',')]
        validate_supported_options(cfg)
    if args.validated_public_ic:
        from core.response_campaign import apply_public_ic
        apply_public_ic(cfg,args.validated_public_ic,args.existing_ic)
        validate_supported_options(cfg)
    research_spec = answers.get('research_spec')
    if research_spec is not None:
        from core.foundation import validate_spec, atomic_json, file_sha256 as digest
        from core.stream_resolution import configured_stream_preflight
        from core.foundation import legacy_science_view
        research_spec = legacy_science_view(research_spec)
        validate_spec(research_spec, products)
        spectrum_manifest = spectrum_path(str(answers['cosmology']))
        spectrum_info = load_json(spectrum_manifest)
        background_relative = spectrum_info['source_files']['start']['background']
        background = spectrum_manifest.parent / background_relative
        expected = spectrum_info['checksums_sha256'].get(background_relative)
        if expected is None or digest(background) != expected:
            raise ValueError('Stream preflight CLASS background is changed or missing from spectrum checksums')
        feasibility = configured_stream_preflight(research_spec, cfg, background)
        # Keep a failed plan outside the still-uncreated simulation directory.
        plan_path = run_dir.parent / (run_dir.name + '_research_preflight.json')
        if plan_path.exists():
            raise FileExistsError('Research preflight already exists; choose a new run name: ' + str(plan_path))
        atomic_json(plan_path, {'schema_version': 1, 'run': str(run_dir), 'research_spec': research_spec,
                                'config': cfg, 'stream_feasibility': feasibility, 'science_status': 'NOT_ACCEPTED'})
        if feasibility['status'] == 'FAIL':
            raise ValueError('Requested stellar-stream scales fail preflight; details retained in ' + str(plan_path))
    allow_incomplete = answers.get("allow_incomplete_analysis", False)
    if not isinstance(allow_incomplete, bool):
        raise ValueError("allow_incomplete_analysis must be a boolean")
    if products and not allow_incomplete and not stage_only:
        required = [("stellar_histories", "stellar_history"), ("zoom_contamination", "zoom_regions"),
                    ('inference_diagnostics', 'inference_diagnostics'),('radial_shapes','radial_shapes')]
        if args.answers:
            required.append(("stellar_components", "stellar_search"))
        missing = [name for product, name in required if product in products and product not in answers.get('deferred_outputs', []) and answers.get(name) is None]
        if missing:
            raise ValueError("Complete analysis requires request settings before evolution: " + ", ".join(missing))
        if "black_holes" in products and not cfg.get("black_holes", {}).get("enabled"):
            raise ValueError("black_holes output requested but native BH physics is disabled")
    if products:
        configure_analysis_outputs(cfg, products)
        print("Requested workflow outputs: " + ", ".join(products))
    particle_mass = particle_mass_msun_h(cfg, dm_only=cfg["simulation"]["physics"] == "dm")
    spacing = mean_particle_spacing_kpc_h(cfg)
    print(f"Cosmology: {cfg['spectrum']['file']}")
    print(f"Run: {run_dir} | {cfg['simulation']['particles_per_dimension']}^3 | {cfg['simulation']['physics']}")
    print(f"Snapshots z: {cfg['simulation']['snapshot_redshift_list']}")
    from core.resources import mpi_ranks as effective_ranks, omp_threads, memory_limit_gb
    print(f"Effective resources: {effective_ranks(cfg)} MPI processes, {omp_threads(cfg)} threads/process, {memory_limit_gb(cfg):g} GiB total budget")
    print(f"Timestep controls: minimum safety threshold={cfg['simulation']['min_size_timestep_ln_a']:g}, maximum={cfg['simulation']['max_size_timestep_ln_a']:g} in ln(a)")
    print(f"Particle mass: {particle_mass:.8g} Msun/h | mean spacing: {spacing:.8g} kpc/h")
    print(f"Box: {cfg['simulation']['box_mpc_h']:.8g} Mpc/h | map pixel: {cfg['simulation']['box_mpc_h'] / cfg['analysis']['density_grid']:.8g} Mpc/h")
    print(f"Filament cell: {cfg['simulation']['box_mpc_h'] / cfg['visualization']['filament_grid_size']:.8g} Mpc/h")
    if not stage_only:
        from core.setup_walkthrough import launch_review
        launch_review(cfg, run_dir, products, answers)
    try:
        result = prepare_and_run(cfg, run_dir, model=cfg["cosmology"]["model"],
                                 dry_run=args.dry_run, ic_only=args.ic_only,existing_ic=args.existing_ic,
                                 prepare_only=args.prepare_only)
    except Exception:
        print("Simulation/preparation failed; automatic post-processing was not started.", file=sys.stderr)
        print("After resolving the failure, inspect any completed saved snapshots with:", file=sys.stderr)
        print("  python commands.py analysis %s --graphs available --dry-run" % __import__('shlex').quote(str(run_dir)), file=sys.stderr)
        raise
    print(f"Simulation stage: {result}")
    if not stage_only:
        stellar_search = answers.get("stellar_search")
        if "stellar_components" in products and "stellar_components" not in answers.get("deferred_outputs", []) and stellar_search is None and not args.answers:
            stellar_search = choose_stellar_search(run_dir)
        process_run(run_dir, products, find_halos=True, stellar_search=stellar_search,
                    stellar_history=answers.get("stellar_history"), zoom_regions=answers.get("zoom_regions"),
                    allow_incomplete=allow_incomplete or bool(answers.get('deferred_outputs')), research_spec=research_spec,
                    inference_diagnostics=answers.get('inference_diagnostics'),radial_shapes=answers.get('radial_shapes'),
                    **({'deferred_products': answers['deferred_outputs']} if answers.get('deferred_outputs') else {}))

# ========================================================================
# run_event_cadence
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from evolution import execute_checkpoint_branch, merge_and_quantize_native_output_times,plan_event_followups,publish_history_candidates,verify_event_branch


def run_event_cadence_main():
    parser=argparse.ArgumentParser(description='Plan event outputs or execute an immutable native checkpoint branch.')
    parser.add_argument('request',type=Path)
    parser.add_argument('destination',type=Path,nargs='?')
    parser.add_argument('--execute-branch',action='store_true')
    parser.add_argument('--plan-followups',action='store_true')
    parser.add_argument('--from-history',action='store_true')
    parser.add_argument('--verify-branch',action='store_true')
    args=parser.parse_args()
    if sum((args.execute_branch,args.plan_followups,args.from_history,args.verify_branch))>1:parser.error('Choose one cadence operation')
    if args.verify_branch:
        if args.destination is not None:parser.error('Branch verification takes only its lineage receipt')
        report=verify_event_branch(args.request)
        print(json.dumps({'status':report['status'],'outputs':len(report['measured_outputs'])},indent=2))
        return
    if args.destination is None:parser.error('This operation requires a fresh destination')
    request=json.loads(args.request.read_text())
    if args.execute_branch:
        report=execute_checkpoint_branch(destination=args.destination,**request)
    elif args.plan_followups:
        report=plan_event_followups(destination=args.destination,**request)
    elif args.from_history:
        report=publish_history_candidates(destination=args.destination,**request)
    else:
        if args.destination.exists():raise FileExistsError(args.destination)
        report=merge_and_quantize_native_output_times(**request)
        atomic_json(args.destination,report)
    print(json.dumps(report,indent=2,allow_nan=False))
import re

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent / "core")]

from core.linear import run_class_main



























# ========================================================================
# trace_zoom

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from zoom import trace_aperture
from zoom import trace_catalogue_host


def trace_zoom_main():
    parser = argparse.ArgumentParser(description='Trace a saved DM aperture to its recorded parent IC particle region.')
    parser.add_argument("request", type=Path)
    parser.add_argument('--catalogue-host', action='store_true', help='Select actual native FOF host/galaxy identity rather than an arbitrary centre')
    args = parser.parse_args()
    operation = trace_catalogue_host if args.catalogue_host else trace_aperture
    print(json.dumps(operation(**json.loads(args.request.read_text())), indent=2))

# ========================================================================
# audit_native_handoff

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from zoom import verify_native_handoff


def audit_native_handoff_main():
    parser=argparse.ArgumentParser(description='Revalidate a native parent/zoom IC against its actual model-matched sources.')
    parser.add_argument('configuration',type=Path,nargs='?',help='Simulation YAML with native_generation_report')
    parser.add_argument('initial_ic',type=Path,nargs='?')
    parser.add_argument('destination',type=Path,nargs='?')
    parser.add_argument('--audit-memory-gib',type=float,default=1)
    parser.add_argument('--numerical-control-receipt',type=Path,help='Verify all retained native control builds, inputs and measurements')
    args=parser.parse_args()
    if args.numerical_control_receipt:
        if any((args.configuration,args.initial_ic,args.destination)):
            parser.error('Control verification cannot be mixed with IC handoff arguments')
        from core.research.controls import verify_control_campaign
        report=verify_control_campaign(args.numerical_control_receipt)
        print(json.dumps(report,indent=2))
        if report['status']!='PASS':raise SystemExit(1)
        return
    if not all((args.configuration,args.initial_ic,args.destination)):
        parser.error('IC handoff requires configuration, initial_ic and destination')
    cfg=yaml.safe_load(args.configuration.read_text())
    report=verify_native_handoff(args.initial_ic,cfg['simulation']['native_generation_report'],cfg,
                                args.destination,memory_gib=args.audit_memory_gib)
    print(json.dumps(report,indent=2))
import h5py

run_blender_SUITE = Path(__file__).resolve().parent
sys.path[:0] = [str(run_blender_SUITE), str(run_blender_SUITE / "core")]
import visualization as visualize
from visualization import make_blender_timeline_loader, snapshot_ids as run_blender_snapshot_ids
from core.foundation import copy_blender_runtime


def indexed_blender_figure_gallery(run_dir, export_root):
    """Copy only registered figure PNGs, never recursively discover render output."""
    from core.foundation import verify_artifact_index, run_namespace_binding, atomic_copy
    run, destination = Path(run_dir).resolve(), Path(export_root).resolve()
    results = run/'results'
    index_path = results/'artifact-index.json'
    if not index_path.is_file():
        return {'status':'MISSING_ARTIFACT_INDEX','entries':[],
            'reason':'Figure exports must be registered before inclusion in the Blender gallery'}
    ownership = run_namespace_binding(run)
    if ownership['status'] != 'DECLARED':
        return {'status':'UNRESOLVED_RUN_OWNERSHIP','entries':[],
            'reason':'Legacy run ownership is undeclared; no gallery source association is inferred'}
    before = file_sha256(index_path)
    index = verify_artifact_index(results)
    if index['run_namespace'] != ownership['run_namespace']:
        raise ValueError('Blender gallery index belongs to another run namespace')
    selected = [row for row in index['artifacts'] if
        row['relative_path'].startswith('figures/png/') and Path(row['relative_path']).suffix.lower()=='.png']
    entries = []
    for row in selected:
        path = results/row['relative_path']
        copied = destination/'figures'/(row['sha256']+'.png')
        if not copied.is_file() or file_sha256(copied)!=row['sha256']:
            atomic_copy(path,copied)
        if file_sha256(copied)!=row['sha256'] or file_sha256(path)!=row['sha256']:
            raise ValueError('Indexed Blender figure changed during gallery copying')
        entries.append({'path':copied.relative_to(destination).as_posix(),'title':path.stem,
            'source':path.relative_to(run).as_posix(),'sha256':row['sha256'],
            'artifact_id':row['artifact_id'],'catalogue_id':row['catalogue_id'],
            'support_status':row['support_status']})
    if file_sha256(index_path)!=before:
        raise ValueError('Artifact index changed during Blender gallery export')
    return {'status':'COPIED_REGISTERED_FIGURES' if entries else 'NO_REGISTERED_FIGURES',
        'entries':entries,'source_index':{'relative_path':'artifact-index.json','sha256':before},
        'scope':'Registered figure PNGs only; numeric sources and scientific acceptance remain in the run index'}


def verified_analysis_figure_gallery(run_dir, analysis_directory, export_root):
    """Copy PNGs explicitly listed in this run's sealed selected-analysis bundle."""
    from analysis import verify_analysis_bundle
    from core.foundation import atomic_copy
    run, source, target = map(lambda x:Path(x).resolve(),(run_dir,analysis_directory,export_root))
    record=verify_analysis_bundle(source,verify_sources=True)
    if Path(record['source_run']).resolve()!=run:
        raise ValueError('Selected analysis gallery belongs to another run')
    before=file_sha256(source/'analysis_provenance.json')
    entries=[]
    for row in record['output_artifacts']:
        path=source/row['relative_path']
        if path.suffix.lower()!='.png':continue
        copied=target/'figures'/(row['sha256']+'.png')
        atomic_copy(path,copied)
        if file_sha256(copied)!=row['sha256'] or file_sha256(path)!=row['sha256']:
            raise ValueError('Selected analysis gallery figure changed during copy')
        entries.append({'path':copied.relative_to(target).as_posix(),'title':path.stem,
            'source':str(path),'sha256':row['sha256'], 'support_status':'SEALED_ANALYSIS_OUTPUT',
            'analysis_provenance_sha256':before})
    if file_sha256(source/'analysis_provenance.json')!=before:
        raise ValueError('Selected analysis provenance changed during gallery copy')
    return entries


@native_run_owner
def export(run_dir: Path, max_points: int, quality: str, frames_per_snapshot: int, *, radial_shapes=None,scene_attachments=None,output_directory=None,analysis_directory=None) -> Path:
    """Reuse exact-ID and periodic-wrap exporters for DM, gas, and stars."""
    run_dir=Path(run_dir).resolve()
    if len(run_blender_snapshot_ids(run_dir)) < 2:
        raise ValueError("A cinematic timeline needs at least two saved snapshots")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    cfg["visualization"].update(blender_render=False, blender_timeline=True,
                                blender_interpolate_motion=True,
                                blender_frames_per_snapshot=frames_per_snapshot,
                                blender_render_quality=quality)
    # Supply render settings explicitly without changing recorded run physics.
    requested_species = ["dm"] if cfg["simulation"]["physics"] == "dm" else ["dm", "gas", "stars"]
    particle_types = {"dm": 1, "gas": 0, "stars": 4}
    species = [item for item in requested_species if any(
        _snapshot_has_particle_type(run_dir, snap, particle_types[item])
        for snap in run_blender_snapshot_ids(run_dir))]
    unavailable_species = {item: "No particles of this species in any saved output"
                           for item in requested_species if item not in species}
    if any(_snapshot_has_particle_type(run_dir, snap, 5) for snap in run_blender_snapshot_ids(run_dir)):
        species.append("black_holes")
    manifests = {}
    shape_report = None
    shape_profile_reports = []
    if radial_shapes is not None:
        from halos import verify_shape_request,shape_reports
        shape_report = verify_shape_request(run_dir, radial_shapes)
        shape_profile_reports = list(shape_reports(run_dir,radial_shapes))
    import datetime,uuid
    base=Path(output_directory).expanduser().resolve() if output_directory is not None else run_dir/"results/3d"/("universe_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8])
    if output_directory is not None and base.is_relative_to(run_dir):
        raise ValueError('Explicit scene output must be outside the retained native run')
    base.mkdir(parents=True,exist_ok=False)
    from analysis import main_progenitor_histories, read_snapshot_header, redshift_from_header
    epochs = {str(snap): {"redshift": redshift_from_header(read_snapshot_header(run_dir, snap)[0])}
              for snap in run_blender_snapshot_ids(run_dir)}
    histories = main_progenitor_histories(run_dir, cfg, epochs)
    # Join only explicit original catalogue IDs at the same saved epoch.
    # Consistent Trees IDs occupy a separate namespace from Rockstar IDs.
    lookup = {}
    if histories.get("status") == "COMPLETE":
        for history in histories["histories"]:
            for row in history["track"]:
                lookup.setdefault((row["snapshot"], row["original_halo_id"]), history)
    def transform(timeline):
        if shape_report is not None:
            from halos import join_radial_shapes_to_timeline
            for profile_report in shape_profile_reports:
                join_radial_shapes_to_timeline(profile_report, timeline)
        for epoch in timeline["snapshots"]:
            for detection in epoch.get("halo_overlays", {}).get("rockstar", {}).get("items", []):
                history = lookup.get((epoch["snapshot"], detection["id"]))
                if history is not None:
                    detection["merger_history"] = history

    for item in species:
        loader=make_blender_timeline_loader(run_dir,item,max_points=max_points,config=cfg,manifest_transform=transform,output_directory=base/item)
        manifests[item]=str((loader.parent/f"blender_{item}_timeline.json").relative_to(base))
    combined = {"schema_version": 1, "run_dir": str(run_dir), "species_manifests": manifests,
                "quality": quality, "frames_per_snapshot": frames_per_snapshot,
                "unavailable_species": unavailable_species,
                "merger_histories": histories,
                "scene_note": "Visual interpolation joins matching ParticleIDs; intermediate frames are not simulation timesteps. Optional black_holes are actual GADGET PartType5 particles, displayed as Newtonian mass markers without event horizons or relativistic light bending."}
    combined["species_manifest_sha256"]={kind:file_sha256(base/name) for kind,name in manifests.items()}
    combined["science_acceptance"]="NOT_ACCEPTED"
    combined['hubble_parameter']=float(cfg['cosmology']['H0'])/100 if 'H0' in cfg['cosmology'] else None
    destination = base / "blender_universe.json"
    if shape_report is not None:
        from halos import shape_profile_requests
        combined['radial_density_shape_profiles']=[]
        for request in shape_profile_requests(radial_shapes):
            source=Path(request['output_directory'])
            copied=base/'radial_shapes'/file_sha256(source/'report.json')
            copied.mkdir(parents=True,exist_ok=True)
            for name in ('report.json','report.sha256','parent.npz'):
                shutil.copy2(source/name,copied/name)
            combined['radial_density_shape_profiles'].append({'report':str((copied/'report.json').relative_to(base)),
                'sha256':file_sha256(copied/'report.json'),'parent_sha256':file_sha256(copied/'parent.npz'),
                'snapshot':request['snapshot'],'host_row':request['host_row'],'science_acceptance':'NOT_ACCEPTED'})
        if len(combined['radial_density_shape_profiles'])==1:
            combined['radial_density_shapes']=combined['radial_density_shape_profiles'][0]
    gallery = indexed_blender_figure_gallery(run_dir,base)
    combined['figure_gallery'] = gallery.pop('entries')
    combined['figure_gallery_status'] = gallery
    if analysis_directory is not None:
        combined['figure_gallery'].extend(verified_analysis_figure_gallery(run_dir,analysis_directory,base))
        combined['figure_gallery_status']['selected_analysis_directory']=str(Path(analysis_directory).resolve())
    if scene_attachments is not None:
        from core.research.display import attach_scene_products
        attach_scene_products(scene_attachments,combined,base)
    fixture_record = run_dir / "native-validation.json"
    if fixture_record.is_file():
        combined["validation_fixture"] = json.loads(fixture_record.read_text())["fixture"]
    from core.foundation import atomic_json
    atomic_json(destination,combined)
    (base/"blender_universe.json.sha256").write_text(file_sha256(destination)+"\n")
    return destination


def _snapshot_has_particle_type(run_dir: Path, snapshot: int, particle_type: int) -> bool:
    """Check the actual HDF5 groups in one output for non-empty GADGET particles."""
    _header, files = visualize.read_snapshot_header(run_dir, int(snapshot))
    group = "PartType%d" % int(particle_type)
    for path in files:
        with h5py.File(path, "r") as source:
            if group in source and "Coordinates" in source[group] and len(source[group]["Coordinates"]):
                return True
    return False


@directory_owner("manifest")
def build_scene(manifest: Path, *, preview=False, render=False):
    """Build a Blender scene from saved exported particle assets; rendering changes display only."""
    from visualization import _find_blender_executable
    executable = _find_blender_executable()
    if not executable:
        raise RuntimeError("Blender executable is missing; exported data remain available")
    runner = manifest.parent / "build_research_scene.py"
    # Copy the builder and its inspector next to the exported data for portability.
    import shutil
    copy_blender_runtime(manifest.parent)
    from core.foundation import atomic_text
    runner_code = ("from pathlib import Path\nimport sys\nbase=Path(__file__).resolve().parent\n"
                   'sys.path.insert(0,str(base))\nfrom blender import build_blender_universe_build as build\n'
                   "manifest=base/'blender_universe.json'\nimport json\n"
                   "settings=json.loads(manifest.read_text())\n"
                   "build(manifest, tier='research' if settings['quality']=='high' else 'viewport', "
                   "frames_per_snap=settings['frames_per_snapshot'], preview=" + repr(bool(preview)) +
                   ", render=" + repr(bool(render)) + ")\n")
    atomic_text(runner, runner_code)
    log = manifest.parent / "blender_build.log"
    with log.open("w") as stream:
        from core.native import run_native_command
        run_native_command([executable, "--background", "--python-exit-code", "1", "--python", str(runner)],
            cwd=manifest.parent, receipt_path=manifest.parent / "blender_build.invocation.json",
            inputs=[manifest], stdout=stream, stderr=subprocess.STDOUT, check=True)
    if not (manifest.parent / "blender_universe.blend").is_file():
        raise RuntimeError(f"Blender did not save the scene; inspect {log}")
    if preview and not (manifest.parent / "blender_preview.png").is_file():
        raise RuntimeError(f"Blender did not render the preview; inspect {log}")
    if render and not (manifest.parent / "blender_universe_movie.mp4").is_file():
        raise RuntimeError(f"Blender did not render the movie; inspect {log}")
    from blender import verify_scene_publication
    verify_scene_publication(manifest.parent / "blender_universe.blend")
    print(f"Blender scene: {manifest.parent / 'blender_universe.blend'}")


def run_blender_main() -> None:
    """Collect export settings and create a portable Blender scene manifest."""
    parser = argparse.ArgumentParser(description='Export aligned particle timelines and halo overlays for one combined scene.')
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--quality", choices=["preview", "high"])
    parser.add_argument("--frames-per-snapshot", type=int)
    parser.add_argument("--build", action="store_true", help="build and save the combined .blend scene")
    parser.add_argument("--preview", action="store_true", help="build the scene and render a scientific preview PNG")
    parser.add_argument('--radial-shapes',type=Path,help='Exact native radial density-shape request JSON for verified FOF/epoch joins')
    parser.add_argument('--measured-display', type=Path, help='Sealed measured shapes/tides/streams/events/SIDM inspector manifest')
    parser.add_argument('--scene-attachments',type=Path,help='Source-bound native stream timelines and exact-context scientific gallery request')
    parser.add_argument('--output',type=Path,help='Fresh scene bundle directory outside the retained native run')
    args = parser.parse_args()
    run_dir = (args.run_dir or select_run(runtime_paths().runs_root, "config.yaml")).resolve()
    points = args.max_points if args.max_points is not None else required_integer("Maximum sampled particles per species [integer; stable ParticleID sampling]", 100, 5000000)
    quality = args.quality or required_choice("Render quality", ["preview", "high"])
    frames = args.frames_per_snapshot if args.frames_per_snapshot is not None else required_integer("Frames between saved snapshots [integer; visual interpolation only]", 2, 240)
    if not 100 <= points <= 5_000_000 or not 2 <= frames <= 240:
        raise ValueError("Point budget must be 100–5,000,000 and frames 2–240")
    radial_shapes = json.loads(args.radial_shapes.read_text()) if args.radial_shapes else None
    attachments=json.loads(args.scene_attachments.read_text()) if args.scene_attachments else None
    if args.measured_display and attachments and attachments.get('encounter_overlay') is not None:
        raise ValueError('Native encounter overlay already supplies the measured-display generation; select one source rather than replacing its measurements')
    manifest = export(run_dir, points, quality, frames, radial_shapes=radial_shapes,scene_attachments=attachments,output_directory=args.output)
    if args.measured_display:
        from core.blender_scene import package_display_products
        package_display_products(args.measured_display, manifest.parent/'measured_display')
        combined = json.loads(manifest.read_text())
        combined['measured_display'] = {'path': 'measured_display/display.json', 'sha256': file_sha256(manifest.parent/'measured_display/display.json')}
        atomic_json(manifest, combined)
        manifest.with_name(manifest.name+'.sha256').write_text(file_sha256(manifest)+'\n')
    print(f"Combined Blender manifest: {manifest}")
    if args.build or args.preview:
        build_scene(manifest, preview=args.preview)

# ========================================================================
# stage_zoom_inputs

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from zoom import stage_zoom_inputs


def prepare_generated_flow_main():
    """Prepare evolution answers from the exact newly generated native mesh."""
    parser=argparse.ArgumentParser(description=prepare_generated_flow_main.__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--generation-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from core.music.staging import prepare_generated_flow_answers
    result=prepare_generated_flow_answers(args.generation_report,json.loads(args.request.read_text()),args.output)
    print(json.dumps(result,indent=2))
    return 0


def stage_zoom_inputs_main():
    parser=argparse.ArgumentParser(description='Stage CLASS/AxiCLASS species transfers and compatible MUSIC parent/zoom inputs.')
    parser.add_argument('request',type=Path)
    parser.add_argument('--resolution-refinement',action='store_true',help='Stage a bounded higher-resolution child from an audited parent and saved-aperture budget')
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    if args.resolution_refinement:
        from zoom import stage_resolution_refinement
        result=stage_resolution_refinement(**request)
    else:
        result=stage_zoom_inputs(**request)
    print(json.dumps(result,indent=2))

# ========================================================================
# run_start_epoch_campaign
run_start_epoch_campaign_ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(run_start_epoch_campaign_ROOT/'core'))
from evolution import (build_start_epoch_campaign,stage_start_epoch_branch,
    write_start_epoch_measurements,compare_start_epoch_campaign,verify_start_epoch_campaign,
    audit_start_epoch_phases,execute_start_epoch_branch)


def run_start_epoch_campaign_main():
    parser=argparse.ArgumentParser(description='Prepare, stage, measure, compare or verify user-selected start-epoch branches.')
    sub=parser.add_subparsers(dest='operation',required=True)
    build=sub.add_parser('build');build.add_argument('--spec',type=Path,required=True);build.add_argument('--output',type=Path,required=True)
    build.add_argument('--resume',action='store_true')
    check=sub.add_parser('verify');check.add_argument('--campaign',type=Path,required=True)
    stage=sub.add_parser('stage');stage.add_argument('--campaign',type=Path,required=True)
    stage.add_argument('--branch',required=True);stage.add_argument('--output',type=Path,required=True)
    stage.add_argument('--wdm-request',type=Path)
    execute=sub.add_parser('execute');execute.add_argument('--campaign',type=Path,required=True)
    execute.add_argument('--branch',required=True);execute.add_argument('--run-directory',type=Path,required=True)
    execute.add_argument('--kind',choices=('evolution','ic_generation','numerical_control'),required=True)
    execute.add_argument('command',nargs=argparse.REMAINDER)
    phases=sub.add_parser('phases');phases.add_argument('--campaign',type=Path,required=True)
    phases.add_argument('--request',type=Path,required=True);phases.add_argument('--output',type=Path,required=True)
    measure=sub.add_parser('measure');measure.add_argument('--campaign',type=Path,required=True)
    measure.add_argument('--request',type=Path,required=True);measure.add_argument('--output',type=Path,required=True)
    density=sub.add_parser('measure-density-modes');density.add_argument('--campaign',type=Path,required=True)
    density.add_argument('--request',type=Path,required=True);density.add_argument('--output',type=Path,required=True)
    producers=sub.add_parser('measure-analysis');producers.add_argument('--campaign',type=Path,required=True)
    producers.add_argument('--request',type=Path,required=True);producers.add_argument('--output',type=Path,required=True)
    compare=sub.add_parser('compare');compare.add_argument('--campaign',type=Path,required=True)
    compare.add_argument('--measurements',nargs='+',type=Path,required=True);compare.add_argument('--output',type=Path,required=True)
    compare.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    load=lambda path:json.loads(path.read_text())
    if args.operation=='build':
        spec=load(args.spec);result=build_start_epoch_campaign(spec,spec['start_epoch_campaign']['redshifts'],args.output,resume=args.resume)
    elif args.operation=='verify':result=verify_start_epoch_campaign(args.campaign)
    elif args.operation=='stage':result=stage_start_epoch_branch(args.campaign,args.branch,args.output,
        wdm_request=load(args.wdm_request) if args.wdm_request else None)
    elif args.operation=='execute':
        command=args.command[1:] if args.command[:1]==['--'] else args.command
        result=execute_start_epoch_branch(args.campaign,args.branch,command,args.run_directory,execution_kind=args.kind)
    elif args.operation=='phases':
        request=load(args.request)
        result=audit_start_epoch_phases(args.campaign,request['realizations'],args.output,
            modes=request['modes'],maximum_phase_error=request.get('maximum_phase_error',.05),memory_gib=request.get('memory_gib',1.))
    elif args.operation=='measure':
        request=load(args.request)
        result=write_start_epoch_measurements(args.campaign,request['branch_id'],request['snapshots'],request['values'],args.output,
            ic_phase_audit=request['ic_phase_audit'],execution_receipt=request.get('execution_receipt'),execution_kind=request.get('execution_kind'))
    elif args.operation=='measure-analysis':
        from evolution import write_start_epoch_analysis_measurements
        request=load(args.request)
        result=write_start_epoch_analysis_measurements(args.campaign,request['branch_id'],request['products'],args.output,
            ic_phase_audit=request['ic_phase_audit'],execution_receipt=request.get('execution_receipt'),
            execution_kind=request.get('execution_kind'),memory_gib=request.get('memory_gib',1.))
    elif args.operation=='measure-density-modes':
        from evolution import write_start_epoch_density_modes
        request=load(args.request)
        result=write_start_epoch_density_modes(args.campaign,request['branch_id'],request['snapshot_sets'],
            request['modes'],args.output,ic_phase_audit=request['ic_phase_audit'],
            execution_receipt=request.get('execution_receipt'),execution_kind=request.get('execution_kind'),
            species=tuple(request.get('species',[0,1])),memory_gib=request.get('memory_gib',1.))
    else:result=compare_start_epoch_campaign(args.campaign,args.measurements,args.output,memory_gib=args.memory_gib)
    status=result['status'];print(status)
    return 1 if status=='FAIL' else 2 if status=='INCOMPLETE' else 0
from decimal import Decimal, InvalidOperation

suite_common_ROOT = Path(__file__).resolve().parent
CORE = suite_common_ROOT / "core"

# Import the shipped root entrypoints and shared core implementations.
for p in reversed([str(suite_common_ROOT), str(CORE)]):
    if p not in sys.path:
        sys.path.insert(0, p)


def get_config_path() -> Path:
    """Find the shipped base configuration at core/config.yaml."""
    if (CORE / "config.yaml").is_file():
        return CORE / "config.yaml"
    raise FileNotFoundError("Base configuration is missing: " + str(CORE / "config.yaml"))



def ask_required(label: str) -> str:
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        print("An explicit value is required.")


def required_number(label: str, minimum: float, maximum: float) -> float:
    while True:
        value = ask_required(label)
        try:
            parsed = float(value)
            if math.isfinite(parsed) and minimum <= parsed <= maximum:
                return parsed
        except ValueError:
            pass
        print(f"Enter a finite value from {minimum:g} to {maximum:g}.")


def required_integer(label: str, minimum: int, maximum: int) -> int:
    while True:
        value = ask_required(label)
        try:
            return integer_value(value, label, minimum, maximum)
        except ValueError as exc:
            print(exc)


def required_choice(label: str, options: list[str]) -> str:
    choices = [str(option).lower() for option in options]
    while True:
        value = ask_required(f"{label} ({'/'.join(choices)})").lower()
        if value in choices:
            return value
        print("Choose " + ", ".join(choices) + ".")



def integer_value(value, label: str, minimum: int, maximum: int) -> int:
    """Validate exact integers against inclusive bounds without float rounding.

    Decimal parsing accepts integer-valued decimal/scientific strings while
    preserving large Python/JSON integers and particle identifiers above 2^53.
    It cannot restore precision already lost by a caller's floating-point input.
    """
    if isinstance(value,float) and abs(value)>=2**53:
        raise ValueError(f"{label} requires an exact integer or decimal string above the safe floating-point range.")
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not a boolean.")
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer from {minimum} to {maximum}.") from exc
    if not numeric.is_finite() or numeric != numeric.to_integral_value() or not minimum <= numeric <= maximum:
        raise ValueError(f"{label} must be an integer from {minimum} to {maximum}.")
    return int(numeric)



def named_path(name: str, parent: Path) -> Path:
    """Prevent traversal and accidental overwrite of a named run directory."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Run name must contain 1–64 letters, digits, underscores, or hyphens")
    target = parent / name
    if target.exists():
        raise FileExistsError(f"Run already exists: {target}")
    return target


def select_runs(parent: Path, required: str) -> list[Path]:
    """Select one or more completed run folders by comma-separated numbers."""
    from analysis import snapshot_ids
    candidates=[]
    if parent.exists():
        for path in parent.iterdir():
            if path.is_dir() and (path/required).is_file() and (required!='config.yaml' or snapshot_ids(path)):
                candidates.append(path)
    candidates.sort(key=lambda path:path.stat().st_mtime,reverse=True)
    if not candidates:raise FileNotFoundError(f'No runs containing {required} in {parent}')
    for index,path in enumerate(candidates,1):print(f'  {index}. {path.name}')
    while True:
        answer=ask_required('Select run numbers (for example 1,2,3)')
        fields=[x.strip() for x in answer.split(',')]
        if fields and all(x.isdigit() and 1<=int(x)<=len(candidates) for x in fields) and len(set(fields))==len(fields):
            indices=[int(x)-1 for x in fields]
            if len(set(indices))==len(indices):return [candidates[i] for i in indices]
        print(f'Enter distinct comma-separated numbers from 1 to {len(candidates)}.')


def select_run(parent: Path, required: str) -> Path:
    """Show completed run folders and return the user's selection."""
    # Share the scientific reader's filename discovery, including snapdir_NNN
    # and numbered shards. A directory listing reads no simulation arrays.
    from analysis import snapshot_ids
    candidates = []
    if parent.exists():
        for path in parent.iterdir():
            if not path.is_dir() or not (path / required).is_file():
                continue
            if required == 'spectrum.json':
                status=path/'execution_status.json'
                if status.is_file() and json.loads(status.read_text()).get('status')!='COMPLETE':continue
                if json.loads((path/required).read_text()).get('status')!='complete':continue
            if required == "config.yaml" and not snapshot_ids(path):
                continue
            candidates.append(path)
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No runs containing {required} in {parent}")
    for index, path in enumerate(candidates, 1):
        print(f"  {index}. {path.name}")
    while True:
        answer = ask_required("Select run number")
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            chosen=candidates[int(answer)-1]
            if required=='spectrum.json':load_json(chosen/required)
            return chosen
        print(f"Enter an integer from 1 to {len(candidates)}.")


def load_json(path: Path) -> dict:
    """Read a JSON object and give a useful error for missing metadata."""
    path=Path(path)
    if path.name=='spectrum.json':
        from core.linear import verify_spectrum_directory
        return verify_spectrum_directory(path.parent)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def redshift_schedule(spec: str, start_z: float) -> list[float]:
    """Expand the saved schedule; stars affect navigation only."""
    from core.epochs import parse_schedule
    return parse_schedule(spec, start_z)[0]

# ========================================================================
# audit_data_contract
sys.path[:0]=[str(Path(__file__).resolve().parent/'core')]
from core.data import audit_native_snapshot_contract, replay_ic_transform


def audit_data_contract_main():
    parser=argparse.ArgumentParser(description='Audit complete saved native payloads and canonical time/units/ID provenance.')
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--snapshot',type=int)
    parser.add_argument('--recover-native-set', type=Path, help='Explicit native HDF5 base path inside the run')
    parser.add_argument('--native-set-generation', help='Committed native generation to restore; required with --recover-native-set')
    parser.add_argument('--recover-ic-transaction',type=Path,help='Replay a durable pending IC commit; do not repeat its transform')
    parser.add_argument('--recover-catalogue-snapshot',type=int,help='Replay a pending native catalogue/order publication without rerunning finding')
    parser.add_argument('--product',type=Path,help='Audit a saved contracted product inside this run')
    parser.add_argument('--product-kind',choices=('hdf5','numeric-table','numeric-audit','native-ic-audit','blender-scene','scientific-image','epoch-selection'))
    parser.add_argument('--recover-generation',type=Path,help='Explicit replay of a sealed product generation; requires --product and --product-kind')
    parser.add_argument('--require-chunk-checksums',action='store_true',help='Reject nonempty native snapshot fields without Fletcher32 protection')
    parser.add_argument('--require-scientific-semantics', action='store_true', help='Require explicit sealed units, frame and epoch for every numeric HDF5 dataset')
    parser.add_argument('--memory-gib',type=float,default=1.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.recover_native_set is not None:
        if not args.native_set_generation:
            parser.error('--recover-native-set requires --native-set-generation')
        if (args.product is not None or args.snapshot is not None or args.recover_ic_transaction or
                args.recover_catalogue_snapshot is not None or args.require_chunk_checksums or args.recover_generation):
            parser.error('Native set recovery cannot be combined with other modes')
        base = args.recover_native_set.expanduser()
        if not base.is_absolute(): base = args.run_dir / base
        base = base.resolve()
        if not base.is_relative_to(args.run_dir.resolve()):
            parser.error('Native recovery base must be inside the run')
        if args.output.exists(): raise FileExistsError('Choose a fresh audit output')
        from core.data import recover_native_hdf5_set
        try:
            from core.native import native_run_lock
            with native_run_lock(args.run_dir, operation="native snapshot set recovery"):
                result = recover_native_hdf5_set(base, args.native_set_generation)
            atomic_json(args.output, result)
        except Exception as exc:
            atomic_json(args.output, {'status': 'FAIL', 'science_acceptance': 'NOT_ACCEPTED',
                        'failure': {'type': type(exc).__name__, 'message': str(exc)}})
            raise
        print('Native set generation restored (science NOT_ACCEPTED)')
        return
    if args.native_set_generation is not None:
        parser.error('--native-set-generation requires --recover-native-set')
    if args.product is None and (args.product_kind is not None or args.recover_generation is not None):
        parser.error('--product-kind/--recover-generation require --product')
    if args.product is not None:
        if args.snapshot is not None or args.recover_ic_transaction or args.recover_catalogue_snapshot is not None:
            parser.error('Product audit/recovery cannot be combined with snapshot, IC or catalogue modes')
        if args.product_kind is None:parser.error('--product requires --product-kind')
        if args.product_kind in ('numeric-audit','native-ic-audit','epoch-selection') and args.recover_generation is not None:
            parser.error('Immutable numeric audits have no public replay mode; keep failed bytes and choose a fresh audit output')
    if args.require_scientific_semantics and (args.product is None or args.product_kind != 'hdf5'):
        parser.error('--require-scientific-semantics requires --product and --product-kind hdf5')
    if args.require_chunk_checksums and (args.snapshot is None or args.product is not None):
        parser.error('--require-chunk-checksums requires the snapshot audit mode')
    if args.output.exists():
        raise FileExistsError('Keep previous audit evidence; choose a new output path')
    if args.product is not None:
        root=args.run_dir.expanduser().resolve()
        product=args.product.expanduser()
        if not product.is_absolute():product=root/product
        product=product.resolve()
        if not product.is_relative_to(root):parser.error('Product must be inside the declared run root')
        if args.product_kind=='epoch-selection':
            from core.cosmology import verify_start_epoch_selection as verify
        elif args.product_kind=='scientific-image':
            from blender import verify_image_publication as verify, recover_image_publication as recover
        elif args.product_kind=='blender-scene':
            from blender import verify_scene_publication as verify, recover_scene_publication as recover
        elif args.product_kind=='hdf5':
            from core.data import verify_hdf5_product as verify,recover_hdf5_product as recover
        elif args.product_kind=='native-ic-audit':
            from zoom import verify_music_ic_audit as verify
        elif args.product_kind=='numeric-audit':
            from core.data import verify_numeric_audit as verify
        else:
            from core.data import verify_numeric_table as verify,recover_numeric_table as recover
        try:
            if args.recover_generation is not None:
                generation=args.recover_generation.expanduser()
                if not generation.is_absolute():generation=root/generation
                generation=generation.resolve()
                if not generation.is_relative_to(root):parser.error('Recovery generation must be inside the declared run root')
                record=recover(product,generation)
            else:record=verify(product)
            if args.require_scientific_semantics:
                import math
                if not math.isfinite(args.memory_gib) or args.memory_gib <= 0:
                    raise ValueError('Scientific reader memory must be finite and positive')
                from core.data import read_scientific_product
                reader=read_scientific_product(product, memory_bytes=int(args.memory_gib*1024**3))
                record={**record, 'scientific_datasets':reader.datasets, 'semantics_status':'VERIFIED'}
            result={'status':'LEGACY_UNSEALED' if record.get('status')=='LEGACY_UNSEALED' else 'INTEGRITY_VERIFIED',
                'science_acceptance':'NOT_ACCEPTED','product':str(product),'product_kind':args.product_kind,'record':record}
            atomic_json(args.output,result)
        except Exception as exc:
            atomic_json(args.output,{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
                'product':str(product),'failure':{'type':type(exc).__name__,'message':str(exc)}})
            raise
        print('Saved product contract:',result['status'],'(science NOT_ACCEPTED)')
        return
    if args.recover_catalogue_snapshot is not None:
        if args.snapshot is not None or args.recover_ic_transaction:
            parser.error('Catalogue recovery cannot be combined with IC recovery or snapshot audit')
        from core.data import replay_catalogue_publication
        result=replay_catalogue_publication(args.run_dir,args.recover_catalogue_snapshot)
        atomic_json(args.output,result)
        print('Native catalogue publication replay:',result['status'],'(science NOT_ACCEPTED)')
        return
    if args.recover_ic_transaction:
        if args.snapshot is not None:
            parser.error('--recover-ic-transaction cannot be combined with --snapshot')
        journal=args.recover_ic_transaction.expanduser()
        if not journal.is_absolute():journal=args.run_dir/journal
        journal=journal.resolve()
        if not journal.is_relative_to(args.run_dir.resolve()):
            parser.error('IC transaction must be inside the declared run root')
        from core.native import native_run_lock
        with native_run_lock(journal.parent,operation='IC publication recovery'):
            result=replay_ic_transform(journal)
        atomic_json(args.output,result)
        print('IC publication replay:',result['status'],'(science NOT_ACCEPTED)')
        return
    if args.snapshot is None:parser.error('--snapshot is required for a snapshot audit')
    try:
        result=audit_native_snapshot_contract(args.run_dir,args.snapshot,memory_gib=args.memory_gib,require_chunk_checksums=args.require_chunk_checksums)
    except Exception as exc:
        atomic_json(args.output,{'status':'FAIL','science_acceptance':'NOT_ACCEPTED',
            'run':str(args.run_dir.resolve()),'snapshot':args.snapshot,
            'failure':{'type':type(exc).__name__,'message':str(exc)}})
        raise
    atomic_json(args.output,result)
    print('Saved native data contract:',result['status'],'(science NOT_ACCEPTED)')

# ========================================================================
# audit_zoom_phases

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from zoom import compare_density_phases


def audit_zoom_phases_main():
    parser = argparse.ArgumentParser(description='Audit native 1LPT parent/refinement density phases; retains FAIL evidence.')
    for name in ("parent_ic", "zoom_ic", "parent_noise", "zoom_noise", "destination"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--memory-gib", type=float, default=1)
    args = parser.parse_args()
    report = compare_density_phases(
        args.parent_ic,
        args.zoom_ic,
        args.parent_noise,
        args.zoom_noise,
        args.destination,
        memory_gib=args.memory_gib,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1

# ========================================================================
# query_identity_graph
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from halos import IdentityQueries

def query_identity_graph_main():
 p=argparse.ArgumentParser(description='Read verified exact ownership/ancestry queries from a published graph.');p.add_argument('directory',type=Path);p.add_argument('request',type=Path)
 a=p.parse_args();r=json.loads(a.request.read_text())
 if set(r)!={'query','arguments'} or r['query'] not in {'members_at_epoch','descendants','bound_owner','birth_parent','parent_zoom_match'}:raise ValueError('Choose an explicit indexed identity query')
 value=getattr(IdentityQueries(a.directory),r['query'])(**r['arguments'])
 if hasattr(value,'tolist'):value=value.tolist()
 print(json.dumps(value,indent=2,allow_nan=False))

# ========================================================================
# run_finder_campaign
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from evolution import run_saved_finder_campaign


def run_finder_campaign_main():
    p=argparse.ArgumentParser(description='Run explicitly requested native finder stages on an owned saved run.');p.add_argument('run_dir',type=Path);p.add_argument('request',type=Path);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();request=json.loads(args.request.read_text())
    result=run_saved_finder_campaign(args.run_dir,request,dry_run=args.dry_run)
    print(json.dumps(result,indent=2))

# ========================================================================
# export_stellar_observer

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from stars import ObserverFrame, export_stellar_observer_epoch


def export_stellar_observer_main():
    parser = argparse.ArgumentParser(description='Export exact saved stellar IDs to a declared local observer truth catalogue.')
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    request['frame'] = ObserverFrame(**request['frame'])
    print(json.dumps(export_stellar_observer_epoch(args.run_dir, **request), indent=2))

# ========================================================================
# audit_zoom_regions

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from zoom import audit_zoom_regions


def audit_hydro_ssp_main():
    parser=argparse.ArgumentParser(description='Audit native SSP birth masses, epochs, exact parents and legacy missingness.')
    parser.add_argument('request',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    from stars import audit_native_ssp_birth_state,verify_native_ssp_birth_audit
    report=verify_native_ssp_birth_audit(args.output) if args.verify else audit_native_ssp_birth_state(json.loads(args.request.read_text()),args.output)
    print(json.dumps({k:report[k] for k in ('status','particle_count','supported_count','legacy_birth_unavailable_count','science_acceptance')}))


def audit_hydro_events_main():
    parser=argparse.ArgumentParser(description='Audit committed native SSP birth events between two actual output epochs.')
    parser.add_argument('request',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    from stars import audit_native_ssp_birth_events,verify_native_ssp_birth_events
    if args.verify:report=verify_native_ssp_birth_events(args.output)
    else:
        checksum=file_sha256(args.request)
        report=audit_native_ssp_birth_events(json.loads(args.request.read_text()),args.output)
        if file_sha256(args.request)!=checksum:raise ValueError('SSP event request changed during audit')
    print(json.dumps({'status':report['status'],'event_count':report['event_count'],'physical_reconciliation':report['physical_reconciliation']},indent=2))


def audit_hydro_history_main():
    parser=argparse.ArgumentParser(description='Reconcile complete native gas/star/BH censuses and measure multi-epoch gas histories.')
    parser.add_argument('request',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    from stars import audit_native_hydro_history,verify_native_hydro_history
    if args.verify:report=verify_native_hydro_history(args.output)
    else:
        checksum=file_sha256(args.request)
        request=json.loads(args.request.read_text())
        if file_sha256(args.request)!=checksum:raise ValueError('Hydro history request changed while reading')
        report=audit_native_hydro_history(request,args.output)
        if file_sha256(args.request)!=checksum:raise ValueError('Hydro history request changed during publication')
    print(json.dumps({'status':report['status'],'epochs':len(report['epochs']),
        'intervals':len(report['intervals']),'issues':sum(len(x['issues']) for x in report['intervals']),
        'physics_acceptance':report['physics_acceptance']},indent=2))
    if report['status']!='PASS':raise SystemExit(1)


def select_merger_tree_main():
    parser=argparse.ArgumentParser(description='Export one exact final-halo ancestry from saved descendant catalogues.')
    parser.add_argument('run_dir',type=Path);parser.add_argument('analysis_bundle',type=Path);parser.add_argument('--halo-id',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from core.select_merger_tree import export_selected_tree
    print(export_selected_tree(args.run_dir,args.analysis_bundle,args.output,halo_id=args.halo_id))


def void_particle_history_main():
    parser=argparse.ArgumentParser(description='Measure exact particle overlap between saved void-centered apertures.')
    parser.add_argument('run_dir',type=Path);parser.add_argument('void_bundle',type=Path);parser.add_argument('--radius-mpc-h',type=float,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.void_particle_history import make_void_particle_history
    print(make_void_particle_history(args.run_dir,args.void_bundle,args.output,radius_mpc_h=args.radius_mpc_h,memory_gib=args.memory_gib))


def filament_flythrough_main():
    parser=argparse.ArgumentParser(description='Fly along a connected 3-D density-ridge proxy toward a saved halo.')
    parser.add_argument('run_dir',type=Path);parser.add_argument('--snapshot',type=int,required=True);parser.add_argument('--host-row',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grid',type=int,default=48);parser.add_argument('--max-host-distance-fraction',type=float,default=.15);parser.add_argument('--path-length-fraction',type=float,default=.35);parser.add_argument('--frames',type=int,default=240);parser.add_argument('--fps',type=int,default=24);parser.add_argument('--max-particles',type=int,default=100000);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.filament_flythrough import make_filament_flythrough
    print(make_filament_flythrough(args.run_dir,args.output,snapshot=args.snapshot,host_row=args.host_row,grid=args.grid,max_host_distance_fraction=args.max_host_distance_fraction,path_length_fraction=args.path_length_fraction,frames=args.frames,fps=args.fps,max_particles=args.max_particles,memory_gib=args.memory_gib))


def results_gallery_main():
    parser=argparse.ArgumentParser(description='Create an offline gallery of explicitly selected local simulation results.')
    parser.add_argument('request',type=Path,help='Request JSON, or gallery directory when --verify is used');parser.add_argument('--output',type=Path)
    parser.add_argument('--verify',action='store_true',help='Check gallery page and recorded target-file hashes')
    args=parser.parse_args()
    from core.results_gallery import create_results_gallery,verify_results_gallery
    if args.verify:
        if args.output is not None:parser.error('--verify does not use --output')
        print(json.dumps(verify_results_gallery(args.request),indent=2));return
    if args.output is None:parser.error('--output is required when creating a gallery')
    before=file_sha256(args.request);request=json.loads(args.request.read_text())
    result=create_results_gallery(request,args.output,base=args.request.resolve().parent)
    if file_sha256(args.request)!=before:raise ValueError('Gallery request changed during publication')
    print(result)


def merger_tree_map_main():
    parser=argparse.ArgumentParser(description='Connect a verified merger tree to saved density maps.')
    parser.add_argument('run_dir',type=Path);parser.add_argument('analysis_bundle',type=Path);parser.add_argument('map_bundle',type=Path);parser.add_argument('--map-key',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.merger_tree_map import connect_tree_to_maps
    print(connect_tree_to_maps(args.run_dir,args.analysis_bundle,args.map_bundle,args.output,map_key=args.map_key,memory_gib=args.memory_gib))


def metallicity_maps_main():
    parser=argparse.ArgumentParser(description='Add gas metallicity projections to a saved structure-map bundle.')
    parser.add_argument('run_dir',type=Path);parser.add_argument('map_bundle',type=Path);parser.add_argument('--source-key',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.metallicity_maps import make_metallicity_bundle
    result=make_metallicity_bundle(args.run_dir,args.map_bundle,args.output,source_key=args.source_key,memory_gib=args.memory_gib)
    print(json.dumps(result,indent=2,default=str))


def halo_fingerprint_main():
    parser=argparse.ArgumentParser(description='Compare reciprocal exact-ID matched host halos.')
    parser.add_argument('run_a',type=Path);parser.add_argument('run_b',type=Path);parser.add_argument('--snapshot-a',type=int,required=True);parser.add_argument('--snapshot-b',type=int,required=True);parser.add_argument('--host-row',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--min-shared',type=int,default=10);parser.add_argument('--min-fraction',type=float,default=.1);parser.add_argument('--grid',type=int,default=128);parser.add_argument('--redshift-tolerance',type=float,default=1e-6);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.halo_fingerprint import make_halo_fingerprint
    print(make_halo_fingerprint(args.run_a,args.run_b,args.output,snapshot_a=args.snapshot_a,snapshot_b=args.snapshot_b,host_row=args.host_row,min_shared=args.min_shared,min_fraction=args.min_fraction,grid=args.grid,redshift_tolerance=args.redshift_tolerance,memory_gib=args.memory_gib))


def void_explorer_main():
    parser=argparse.ArgumentParser(description='Explore underdense regions in saved periodic dark-matter snapshots.')
    parser.add_argument('run_dir',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grid',type=int,default=48);parser.add_argument('--smoothing-cells',type=float,default=1.5);parser.add_argument('--threshold',type=float,default=.8);parser.add_argument('--match-fraction',type=float,default=.125)
    parser.add_argument('--frames',type=int,default=32);parser.add_argument('--fps',type=int,default=16);parser.add_argument('--max-particles',type=int,default=100000);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.void_explorer import make_void_explorer
    print(make_void_explorer(args.run_dir,args.output,grid=args.grid,smoothing_cells=args.smoothing_cells,threshold=args.threshold,match_fraction=args.match_fraction,frames=args.frames,fps=args.fps,max_particles=args.max_particles,memory_gib=args.memory_gib))


def void_boundary_history_main():
    parser=argparse.ArgumentParser(description='Measure periodic underdensity boundaries and adjacent-epoch mesh overlap.')
    parser.add_argument('run_dir',type=Path);parser.add_argument('void_bundle',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.void_boundary_history import make_boundary_history
    print(make_boundary_history(args.run_dir,args.void_bundle,args.output,memory_gib=args.memory_gib))


def halo_anatomy_main():
    parser=argparse.ArgumentParser(description='Build an interactive cutaway and radial profiles for a saved halo.')
    parser.add_argument('run_dir',type=Path);parser.add_argument('--snapshot',type=int,required=True);parser.add_argument('--host-row',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--grid',type=int,default=96);parser.add_argument('--layers',type=int,default=24);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.halo_anatomy import make_halo_anatomy
    print(make_halo_anatomy(args.run_dir,args.output,snapshot=args.snapshot,host_row=args.host_row,grid=args.grid,layers=args.layers,memory_gib=args.memory_gib))


def parameter_explorer_main():
    parser=argparse.ArgumentParser(description='Build an offline explorer from explicit saved-run bundles.')
    parser.add_argument('request',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args();before=file_sha256(args.request);request=json.loads(args.request.read_text())
    from core.parameter_explorer import make_parameter_explorer
    result=make_parameter_explorer(request,args.output,memory_gib=args.memory_gib)
    if file_sha256(args.request)!=before:raise ValueError('Explorer request changed during export')
    print(result)


def resolution_reveal_main():
    parser=argparse.ArgumentParser(description='Compare saved projections from two resolution runs.')
    parser.add_argument('run_a',type=Path);parser.add_argument('bundle_a',type=Path);parser.add_argument('run_b',type=Path);parser.add_argument('bundle_b',type=Path)
    parser.add_argument('--snapshot-a',type=int,required=True);parser.add_argument('--snapshot-b',type=int,required=True)
    parser.add_argument('--map-key',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--redshift-tolerance',type=float,default=1e-6);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.resolution_reveal import make_resolution_reveal
    print(make_resolution_reveal(args.run_a,args.bundle_a,args.snapshot_a,args.run_b,args.bundle_b,args.snapshot_b,
          args.output,map_key=args.map_key,redshift_tolerance=args.redshift_tolerance,memory_gib=args.memory_gib))


def cosmic_flythrough_main():
    parser=argparse.ArgumentParser(description='Fly an explicit camera path through a saved snapshot toward a catalogue halo.')
    parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--snapshot',type=int,required=True);parser.add_argument('--host-row',type=int,required=True)
    parser.add_argument('--start',type=float,nargs=3,required=True,metavar=('X','Y','Z'))
    parser.add_argument('--waypoint',type=float,nargs=3,action='append',metavar=('X','Y','Z'))
    parser.add_argument('--frames',type=int,default=240);parser.add_argument('--width',type=int,default=1280);parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--fps',type=int,default=24);parser.add_argument('--max-particles',type=int,default=300000)
    parser.add_argument('--fov-degrees',type=float,default=60.);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.cosmic_flythrough import make_cosmic_flythrough
    print(make_cosmic_flythrough(args.run,args.output,snapshot=args.snapshot,host_row=args.host_row,start=args.start,
          waypoints=args.waypoint,frames=args.frames,width=args.width,height=args.height,fps=args.fps,
          max_particles=args.max_particles,fov_degrees=args.fov_degrees,memory_gib=args.memory_gib))


def halo_story_main():
    parser=argparse.ArgumentParser(description='Link an exact saved merger tree to labelled maps and a synchronized offline film.')
    parser.add_argument('run',type=Path)
    parser.add_argument('--tree',type=Path,required=True)
    parser.add_argument('--density-contract',type=Path,required=True)
    parser.add_argument('--density-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fps',type=int,default=24)
    parser.add_argument('--hold-seconds',type=int,default=1)
    args=parser.parse_args()
    from core.halo_story import export_linked_halo_story
    print(export_linked_halo_story(args.tree,args.density_contract,args.density_directory,args.output,
                                  run_dir=args.run,fps=args.fps,hold_seconds=args.hold_seconds))


def lens_map_main():
    parser=argparse.ArgumentParser(description='Compute a thin-lens convergence/shear map from saved projected mass.')
    parser.add_argument('run',type=Path);parser.add_argument('map_bundle',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--map-key',required=True);parser.add_argument('--snapshot',type=int,required=True)
    parser.add_argument('--lens-distance-kpc',type=float,required=True);parser.add_argument('--source-distance-kpc',type=float,required=True)
    parser.add_argument('--lens-source-distance-kpc',type=float,required=True);parser.add_argument('--background-msun-kpc2',type=float,required=True)
    parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.lens_map import make_lens_map
    print(make_lens_map(args.run,args.map_bundle,args.output,map_key=args.map_key,snapshot=args.snapshot,
          lens_kpc=args.lens_distance_kpc,source_kpc=args.source_distance_kpc,lens_source_kpc=args.lens_source_distance_kpc,
          background_msun_kpc2=args.background_msun_kpc2,memory_gib=args.memory_gib))


def stellar_stream_movie_main():
    parser=argparse.ArgumentParser(description='Animate a saved exact-ID stellar candidate history.')
    parser.add_argument('run',type=Path);parser.add_argument('history_bundle',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--center',type=float,nargs=3,metavar=('X','Y','Z'),help='Fixed viewing center in comoving kpc')
    parser.add_argument('--width-kpc',type=float,help='Comoving display width, at most the simulation box')
    parser.add_argument('--max-particles',type=int,default=50000);parser.add_argument('--fps',type=int,default=24)
    parser.add_argument('--seconds-per-snapshot',type=int,default=1);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.stellar_stream_movie import make_stellar_stream_movie
    print(make_stellar_stream_movie(args.run,args.history_bundle,args.output,center=args.center,width_kpc=args.width_kpc,
          max_particles=args.max_particles,fps=args.fps,seconds_per_snapshot=args.seconds_per_snapshot,memory_gib=args.memory_gib))


def planetarium_movie_main():
    parser=argparse.ArgumentParser(description='Render a saved-epoch, fixed-scale dome movie.')
    parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--field',choices=['dm','gas','gas_temperature'],default='dm')
    parser.add_argument('--snapshot',type=int,action='append',help='Saved snapshot; repeat to select epochs')
    parser.add_argument('--observer',type=float,nargs=3,metavar=('X','Y','Z'))
    parser.add_argument('--resolution',type=int,default=512);parser.add_argument('--tilt',type=float,default=0.)
    parser.add_argument('--max-particles',type=int,default=1000000);parser.add_argument('--fps',type=int,default=24)
    parser.add_argument('--seconds-per-snapshot',type=int,default=1);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.dome_movie import make_dome_movie
    print(make_dome_movie(args.run,args.output,field=args.field,snapshots=args.snapshot,observer=args.observer,
          resolution=args.resolution,tilt=args.tilt,max_particles=args.max_particles,fps=args.fps,
          seconds_per_snapshot=args.seconds_per_snapshot,memory_gib=args.memory_gib))


def halo_collision_movie_main():
    parser=argparse.ArgumentParser(description='Animate two linked halo progenitor branches near a merger.')
    parser.add_argument('analysis_bundle',type=Path);parser.add_argument('map_bundle',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--map-key',required=True)
    parser.add_argument('--descendant-id',help='Optional exact tree-node ID to select a merger')
    parser.add_argument('--fps',type=int,default=24);parser.add_argument('--seconds-per-snapshot',type=int,default=1);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.halo_collision_movie import make_halo_collision_movie
    print(make_halo_collision_movie(args.analysis_bundle,args.map_bundle,args.output,map_key=args.map_key,
          descendant_id=args.descendant_id,fps=args.fps,seconds_per_snapshot=args.seconds_per_snapshot,memory_gib=args.memory_gib))


def halo_formation_movie_main():
    parser=argparse.ArgumentParser(description='Animate saved halo ancestry beside its matched density maps.')
    parser.add_argument('analysis_bundle',type=Path);parser.add_argument('map_bundle',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--map-key',required=True)
    parser.add_argument('--fps',type=int,default=24);parser.add_argument('--seconds-per-snapshot',type=int,default=1)
    parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.halo_formation_movie import make_halo_formation_movie
    print(make_halo_formation_movie(args.analysis_bundle,args.map_bundle,args.output,map_key=args.map_key,
          fps=args.fps,seconds_per_snapshot=args.seconds_per_snapshot,memory_gib=args.memory_gib))


def allsky_map_main():
    parser=argparse.ArgumentParser(description='Create an observer-centered simulation matter sky.')
    parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--snapshot',type=int,required=True);parser.add_argument('--field',choices=['dm','gas'],default='dm')
    parser.add_argument('--observer',type=float,nargs=3,metavar=('X','Y','Z'),help='Observer in comoving Mpc/h; default is box center')
    parser.add_argument('--nlon',type=int,default=360);parser.add_argument('--nlat',type=int,default=180)
    parser.add_argument('--max-particles',type=int,default=1000000);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.allsky_map import make_allsky_map
    print(make_allsky_map(args.run,args.output,snapshot=args.snapshot,field=args.field,observer=args.observer,
                          nlon=args.nlon,nlat=args.nlat,max_particles=args.max_particles,memory_gib=args.memory_gib))


def velocity_flow_main():
    parser=argparse.ArgumentParser(description='Map saved dark-matter density and peculiar velocity flow.')
    parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--snapshot',type=int,required=True);parser.add_argument('--projection',choices=['xy','xz','yz'],default='xy')
    parser.add_argument('--grid',type=int,default=32);parser.add_argument('--minimum-particles',type=int,default=3)
    parser.add_argument('--slice-center',type=float);parser.add_argument('--slice-thickness',type=float);parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.velocity_flow import make_velocity_flow
    print(make_velocity_flow(args.run,args.output,snapshot=args.snapshot,projection=args.projection,grid=args.grid,
          min_particles=args.minimum_particles,slice_center=args.slice_center,slice_thickness=args.slice_thickness,memory_gib=args.memory_gib))


def particle_origin_movie_main():
    parser=argparse.ArgumentParser(description='Trace exact saved host-member dark matter through snapshot epochs.')
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--anchor-snapshot',type=int,required=True)
    parser.add_argument('--host-row',type=int,required=True)
    parser.add_argument('--max-particles',type=int,default=50000)
    parser.add_argument('--fps',type=int,default=24)
    parser.add_argument('--seconds-per-snapshot',type=int,default=1)
    parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.particle_origin_movie import make_particle_origin_movie
    print(make_particle_origin_movie(args.run,args.output,anchor_snapshot=args.anchor_snapshot,host_row=args.host_row,
          max_particles=args.max_particles,fps=args.fps,seconds_per_snapshot=args.seconds_per_snapshot,memory_gib=args.memory_gib))


def multifield_movie_main():
    parser=argparse.ArgumentParser(description='Animate synchronized saved field maps.')
    parser.add_argument('bundle',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--map-key',action='append',required=True,help='Map key from scale.json; repeat two to four times')
    parser.add_argument('--fps',type=int,default=24)
    parser.add_argument('--seconds-per-snapshot',type=int,default=1)
    parser.add_argument('--memory-gib',type=float,default=1.)
    parser.add_argument('--run',type=Path,help='Source run if it cannot be identified from map provenance')
    args=parser.parse_args()
    from core.multifield_movie import make_multifield_movie
    print(make_multifield_movie(args.bundle,args.output,map_keys=args.map_key,fps=args.fps,
          seconds_per_snapshot=args.seconds_per_snapshot,memory_gib=args.memory_gib,run_dir=args.run))


def timeline_wall_main():
    parser=argparse.ArgumentParser(description='Build a cosmic timeline from a saved fixed-scale map bundle.')
    parser.add_argument('bundle',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--map-key',required=True,help='Exact key from scale.json')
    parser.add_argument('--max-panels',type=int,default=6)
    parser.add_argument('--memory-gib',type=float,default=1.)
    parser.add_argument('--run',type=Path,help='Source run if it cannot be inferred from map provenance')
    args=parser.parse_args()
    from core.timeline_wall import make_timeline_wall
    print(make_timeline_wall(args.bundle,args.output,map_key=args.map_key,
                             max_panels=args.max_panels,memory_gib=args.memory_gib,run_dir=args.run))


def comparison_movie_main():
    parser=argparse.ArgumentParser(description='Animate a published matched comparison-map series.')
    parser.add_argument('bundle',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--map-key',required=True,help='Exact map key from comparison_maps_summary.json')
    parser.add_argument('--fps',type=int,default=24)
    parser.add_argument('--seconds-per-snapshot',type=int,default=1)
    parser.add_argument('--memory-gib',type=float,default=1.)
    parser.add_argument('--runs',type=Path,nargs=2,metavar=('RUN_A','RUN_B'),help='Source runs when they cannot be inferred from provenance')
    args=parser.parse_args()
    from core.comparison_movie import make_comparison_movie
    print(make_comparison_movie(args.bundle,args.output,map_key=args.map_key,fps=args.fps,
          seconds_per_snapshot=args.seconds_per_snapshot,memory_gib=args.memory_gib,runs=args.runs))


def native_source_overlay_main():
    parser=argparse.ArgumentParser(description='Describe or explicitly install a sealed native full-file source overlay; never build/test/run.')
    parser.add_argument('manifest',type=Path);parser.add_argument('--source-root',type=Path);parser.add_argument('--expected-hashes',type=Path)
    args=parser.parse_args()
    from evolution import native_source_overlay_manifest,install_native_source_overlay
    if args.source_root is None:
        if args.expected_hashes is not None:raise ValueError('Prior hashes apply only to explicit installation')
        record,_=native_source_overlay_manifest(args.manifest);print(json.dumps(record,indent=2));return
    if args.expected_hashes is None:raise ValueError('Installation requires an exact prior-file hash mapping')
    checksum=file_sha256(args.expected_hashes);prior=json.loads(args.expected_hashes.read_text())
    if file_sha256(args.expected_hashes)!=checksum:raise ValueError('Prior source mapping changed while reading')
    report=install_native_source_overlay(args.manifest,args.source_root,expected_existing_hashes=prior)
    print(json.dumps({'status':report['status'],'files':len(report['files']),'build_execution':report['build_execution']},indent=2))


def native_source_events_main():
    parser=argparse.ArgumentParser(description='Connect sealed physical source journals to bounded event/cadence windows.')
    parser.add_argument('request',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();checksum=file_sha256(args.request);request=json.loads(args.request.read_text())
    from evolution import publish_native_source_events
    if file_sha256(args.request)!=checksum:raise ValueError('Source event request changed while reading')
    report=publish_native_source_events(request,args.output)
    if file_sha256(args.request)!=checksum:raise ValueError('Source event request changed during publication')
    print(json.dumps({'events':len(report['events']),'cadence_acceptance':report['cadence_acceptance']},indent=2))


def prepare_source_controls_main():
    parser=argparse.ArgumentParser(description='Write flat native hydro/stellar/BH control ICs and build profiles; no build or simulation is launched.')
    parser.add_argument('request',type=Path);parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args();checksum=file_sha256(args.request);request=json.loads(args.request.read_text())
    from core.research.controls import prepare_native_source_control
    if file_sha256(args.request)!=checksum:raise ValueError('Source control request changed while reading')
    report=prepare_native_source_control(request,args.output_prefix)
    if file_sha256(args.request)!=checksum:raise ValueError('Source control request changed during publication')
    print(json.dumps({'execution_status':report['execution_status'],'files':report['files']},indent=2))


def black_hole_environment_main():
    parser=argparse.ArgumentParser(description='Measure selected BH host binding, wandering and DM radial evolution from sealed native history.')
    parser.add_argument('request',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();checksum=file_sha256(args.request);request=json.loads(args.request.read_text())
    from black_holes import publish_black_hole_environment_history
    if file_sha256(args.request)!=checksum:raise ValueError('BH environment request changed while reading')
    report=publish_black_hole_environment_history(request,args.output)
    if file_sha256(args.request)!=checksum:raise ValueError('BH environment request changed during publication')
    print(json.dumps({'schema':report['schema'],'epochs':len(report['epochs']),'physics_acceptance':report['physics_acceptance']},indent=2))


def hydro_comparison_main():
    parser=argparse.ArgumentParser(description='Prepare matched physics configurations or measure saved radial hydro comparisons; never launch an evolution.')
    parser.add_argument('request',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--design',action='store_true')
    args=parser.parse_args();checksum=file_sha256(args.request);request=json.loads(args.request.read_text())
    if file_sha256(args.request)!=checksum:raise ValueError('Hydro comparison request changed while reading')
    from stars import hydro_comparison_design,measure_hydro_comparison
    if args.design:
        from core.foundation import atomic_json, atomic_text
        report=hydro_comparison_design(**request)
        if args.output.exists():raise FileExistsError('Keep prior comparison design')
        atomic_json(args.output,report);atomic_text(args.output.with_name(args.output.name+'.sha256'),file_sha256(args.output)+'\n')
    else:report=measure_hydro_comparison(request,args.output)
    if file_sha256(args.request)!=checksum:raise ValueError('Hydro comparison request changed during publication')
    print(json.dumps({'schema':report['schema'],'physics_acceptance':report['physics_acceptance']},indent=2))


def compile_stellar_model_main():
    parser=argparse.ArgumentParser(description='Compile pinned total stellar yields/lifetimes into an explicit native delayed-return model.')
    parser.add_argument('request',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--preflight-only',action='store_true',help='Write complete source-row and IMF/age support audit; do not compile or change yields')
    args=parser.parse_args()
    from stars import compile_stellar_return_model,read_stellar_return_model
    if args.verify and args.preflight_only:parser.error('--verify and --preflight-only are mutually exclusive')
    if args.verify:report=read_stellar_return_model(args.output)[0]
    else:
        checksum=file_sha256(args.request);request=json.loads(args.request.read_text())
        if file_sha256(args.request)!=checksum:raise ValueError('Stellar model request changed while reading')
        if args.preflight_only:
            from core.stellar_returns import stellar_return_input_preflight
            report=stellar_return_input_preflight(request,args.output)
            if file_sha256(args.request)!=checksum:raise ValueError('Stellar model request changed during preflight')
            print(json.dumps({'status':report['status'],'readiness':report['readiness'],
                'failed_yield_row_count':report['failed_yield_row_count'],'failures':report['failures'],'audit':str(args.output.resolve())},indent=2))
            return 0 if report['status']=='PASS' else 1
        report=compile_stellar_return_model(request,args.output)
        if file_sha256(args.request)!=checksum:raise ValueError('Stellar model request changed during compilation')
    print(json.dumps({'implementation_status':report['implementation_status'],
        'validation_status':report['validation_status'],'table':report['table'],
        'physics_acceptance':report['physics_acceptance']},indent=2))


def measure_stream_heating_main():
    parser = argparse.ArgumentParser(description='Measure numerical stream heating from paired native granular and smooth controls.')
    parser.add_argument('request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from zoom import publish_stream_heating_calibration
    report = publish_stream_heating_calibration(json.loads(args.request.read_text()), args.output)
    print(report['status'])


def halo_structure_main():
    parser = argparse.ArgumentParser(description='Measure complete native component profiles and validated proper host fields.')
    parser.add_argument('request',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true',help='Verify and reload an existing sealed structure product')
    parser.add_argument('--figure-prefix',type=Path,help='Publish a source-bound PNG/PDF profile and validation figure')
    args = parser.parse_args()
    from halos import publish_halo_structure,verify_halo_structure,plot_halo_structure
    if args.verify:
        report = verify_halo_structure(args.output)
    else:
        before = file_sha256(args.request)
        report = publish_halo_structure(json.loads(args.request.read_text()),args.output,
            request_source_hashes={str(args.request.resolve()):before,str(Path(__file__).resolve()):file_sha256(__file__)})
    if args.figure_prefix:
        plot_halo_structure(args.output,args.figure_prefix)
    print(json.dumps({'status':report['status'],'field_validation_status':report['field_validation_status'],
                     'science_acceptance':report['science_acceptance']},indent=2))


def environment_main():
    parser = argparse.ArgumentParser(description='Measure native periodic peculiar fields, signed tides, web classifications and splashback.')
    parser.add_argument('request',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--history',action='store_true',help='Publish or verify graph-bound multi-epoch reorientation and splashback histories')
    parser.add_argument('--figure-prefix',type=Path)
    parser.add_argument('--figure-smoothing-index',type=int)
    parser.add_argument('--figure-halo-index',type=int)
    args = parser.parse_args()
    if args.figure_prefix is not None and (args.history or args.figure_smoothing_index is None):
        parser.error('Field figures require an explicit --figure-smoothing-index and a single-epoch product')
    from analysis import publish_native_environment,verify_native_environment,publish_environment_history,verify_environment_history
    publisher = publish_environment_history if args.history else publish_native_environment
    verifier = verify_environment_history if args.history else verify_native_environment
    if args.verify:
        report = verifier(args.output)
    else:
        before = file_sha256(args.request)
        report = publisher(json.loads(args.request.read_text()),args.output,
            request_source_hashes={str(args.request.resolve()):before,str(Path(__file__).resolve()):file_sha256(__file__)})
    if args.figure_prefix is not None:
        from analysis import plot_native_environment
        plot_native_environment(args.output,args.figure_prefix,smoothing_index=args.figure_smoothing_index,
                                halo_index=args.figure_halo_index)
    print(json.dumps({'status':report['status'],'science_acceptance':report['science_acceptance'],
                     'smoothing_fields':len(report.get('smoothing_fields',[])),
                     'halo_measurements':len(report.get('halo_measurements',report.get('epoch_contracts',[])))},indent=2))


def audit_zoom_regions_main():
    parser=argparse.ArgumentParser(description='Audit saved declared or exactly tracked host apertures and tracer sampling.')
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('request',type=Path)
    parser.add_argument('--compare-resolution',action='store_true',help='Compare contracted native observable products against a sealed saved-aperture budget')
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    if args.compare_resolution:
        from core.native import native_run_lock
        from zoom import publish_resolution_convergence
        from core.foundation import file_sha256
        if set(request)!={'budget_receipt','snapshot','comparisons','destination'}:
            raise ValueError('Resolution comparison needs budget_receipt, snapshot, comparisons and destination')
        path=Path(request['budget_receipt']).resolve()
        if path.with_name('provenance.sha256').read_text().strip()!=file_sha256(path):raise ValueError('Saved budget seal changed')
        budget=json.loads(path.read_text())
        from core.foundation import verify_product_provenance
        verify_product_provenance(args.run_dir,budget['data_contract'])
        rows=[r for r in budget['records'] if r['snapshot']==request['snapshot']]
        if len(rows)!=1:raise ValueError('Choose one actually audited saved epoch')
        resolution=rows[0]['resolution_budget']
        with native_run_lock(args.run_dir,operation='resolution_convergence'):
            result=publish_resolution_convergence(request['destination'],resolution['fingerprint'],resolution['request'],request['comparisons'])
    else:
        result=audit_zoom_regions(args.run_dir,**request)
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':
        raise SystemExit(1)

# ========================================================================
# run_velociraptor
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from halos import run_velociraptor

def run_velociraptor_main():
 p=argparse.ArgumentParser(description='Execute pinned native DM finding with reversible exact-ID transport.');p.add_argument('run_dir',type=Path);p.add_argument('request',type=Path);p.add_argument('destination',type=Path)
 a=p.parse_args();result=run_velociraptor(a.run_dir,a.destination,**json.loads(a.request.read_text()))
 print(json.dumps({'status':result['status'],'objects':len(result['catalogue']['objects'])},indent=2))

# ========================================================================
# run_zoom

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from zoom import generate_zoom


def run_zoom_main():
    parser = argparse.ArgumentParser(description='Generate structurally audited MUSIC2 zoom ICs; scientific gates remain explicit.')
    parser.add_argument("configuration", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--native-validation", type=Path, required=True)
    parser.add_argument("--traced-region", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--audit-memory-gib", type=float, default=1)
    parser.add_argument('--native-memory-gib', type=float, default=4,
                        help='Native mesh working-set estimate budget, separate from Python audit RAM')
    parser.add_argument('--native-disk-gib', type=float,
                        help='Cap native output/cache estimate; default checks available disk')
    parser.add_argument('--native-wall-hours', type=float,
                        help='Stop native generation at this wall-time budget; partial ICs fail')
    parser.add_argument('--parent-ic', type=Path, help='Compatible saved parent MUSIC IC for required phase matching')
    parser.add_argument('--parent-noise', type=Path, help='Saved parent coarse white-noise file')
    parser.add_argument('--structural-only', action='store_true', help='Explicitly allow missing parent phases; never grants science acceptance')
    args = parser.parse_args()
    if (args.parent_ic is None) != (args.parent_noise is None):
        parser.error('--parent-ic and --parent-noise must be supplied together')
    if args.parent_ic is None and not args.structural_only:
        parser.error('Supply --parent-ic and --parent-noise, or explicitly request --structural-only')
    report = generate_zoom(
        args.configuration,
        args.binary,
        args.destination,
        native_validation=args.native_validation,
        traced_region=args.traced_region,
        threads=args.threads,
        memory_gib=args.audit_memory_gib,
        parent_ic=args.parent_ic,
        parent_noise=args.parent_noise,
        native_memory_gib=args.native_memory_gib,
        native_disk_gib=args.native_disk_gib,
        native_wall_hours=args.native_wall_hours,
    )
    print(json.dumps(report, indent=2))
import csv

find_galaxies_SUITE = Path(__file__).resolve().parent
sys.path[:0] = [str(find_galaxies_SUITE), str(find_galaxies_SUITE / "core")]
from analysis import snapshot_ids as find_galaxies_snapshot_ids
from halos import build_galaxy_catalogues, find_stellar_substructures


CSV_FIELDS = {
    "GalaxyID": "GalaxyID", "SubhaloID": "SubhaloID", "HostFOFID": "HostFOFID",
    "CentralFlag": "CentralFlag", "Resolved": "Resolved",
    "DMMass_Msun_h": "DMMass", "StellarMass_Msun_h": "StellarMass",
    "GasMass_Msun_h": "GasMass", "GasFraction": "GasFraction",
    "SFR_Msun_year": "SFR", "FormationRedshift": "FormationRedshift",
    "StellarHalfMassRadius_Mpc_h": "StellarHalfMassRadius",
    "NDM": "NDM", "NGas": "NGas", "NStars": "NStars",
    "ApertureRadius_Mpc_h": "ApertureRadius_Mpc_h",
    "ApertureStellarMass_Msun_h": "ApertureStellarMass",
    "ApertureGasMass_Msun_h": "ApertureGasMass",
    "ApertureGasFraction": "ApertureGasFraction",
    "ApertureSFR_Msun_year": "ApertureSFR",
    "ApertureFormationRedshift": "ApertureFormationRedshift",
    "ApertureNStars": "ApertureNStars", "ApertureNGas": "ApertureNGas",
}
VECTOR_FIELDS = {
    "x_Mpc_h": ("Position", 0), "y_Mpc_h": ("Position", 1), "z_Mpc_h": ("Position", 2),
    "vx_km_s": ("Velocity", 0), "vy_km_s": ("Velocity", 1), "vz_km_s": ("Velocity", 2),
}


def export_galaxy_csvs(run_dir: Path, snapshots: list[int]) -> list[dict]:
    """Write unit-labelled, spreadsheet-friendly views beside HDF5 catalogues."""
    import numpy as np

    output = run_dir / "results" / "galaxy_catalogues"
    output.mkdir(parents=True, exist_ok=True)
    fields = list(CSV_FIELDS) + list(VECTOR_FIELDS)
    records = []
    for snapshot in snapshots:
        source = run_dir / "results" / f"galaxies_{int(snapshot):03d}.hdf5"
        if not source.is_file():
            continue
        with h5py.File(source, "r") as handle:
            available = {key: np.asarray(handle[key]) for key in CSV_FIELDS.values()
                         if key in handle and isinstance(handle[key], h5py.Dataset)}
            available.update({key: np.asarray(handle[key]) for key in {name for name, _ in VECTOR_FIELDS.values()}
                              if key in handle and isinstance(handle[key], h5py.Dataset)})
            if "GalaxyID" not in available:
                raise ValueError(f"Galaxy catalogue is missing GalaxyID: {source}")
            count = len(available["GalaxyID"])
            path = output / f"snapshot_{int(snapshot):03d}_galaxies.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index in range(count):
                    row = {}
                    for column, dataset in CSV_FIELDS.items():
                        row[column] = available[dataset][index].item() if dataset in available else ""
                    for column, (dataset, component) in VECTOR_FIELDS.items():
                        row[column] = available[dataset][index, component].item() if dataset in available else ""
                    writer.writerow(row)
            records.append({"snapshot": int(snapshot), "rows": count, "source_hdf5": str(source),
                            "csv": str(path), "redshift": float(handle.attrs.get("redshift", float("nan"))),
                            "scale_factor": float(handle.attrs.get("scale_factor", float("nan")))})
    manifest = output / "catalogue_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "darkmattersim-galaxy-csv-v1",
        "mass_unit": "Msun/h", "length_unit": "Mpc/h", "velocity_unit": "km/s",
        "sfr_unit": "Msun/year",
        "bound_fields": "SUBFIND-bound subhalo membership",
        "aperture_fields": "bound PartType0/4 particles within 0.1 times host R200c",
        "snapshots": records,
    }, indent=2) + "\n", encoding="utf-8")
    return records


def find_galaxies_main() -> None:
    """Calculate SUBFIND-bound stellar/gas masses, SFR, and formation epoch."""
    parser = argparse.ArgumentParser(description='Identify resolved hydro galaxies in existing SUBFIND subhalos.')
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--snapshot", action="append", type=int)
    parser.add_argument("--minimum-stars", type=int)
    parser.add_argument("--minimum-dm", type=int)
    parser.add_argument("--stellar-host", type=int, help="search all stars in this zero-based FOF host row with STF")
    parser.add_argument("--linking-length-kpc", type=float, help="required proper physical stellar linking distance")
    parser.add_argument("--stream-minimum-particles", type=int, default=20)
    parser.add_argument("--velocity-neighbors", type=int, default=32)
    parser.add_argument("--physical-neighbors", type=int, default=256)
    parser.add_argument("--cell-fraction", type=float, default=0.01)
    parser.add_argument("--outlier-threshold", type=float, default=2.5)
    parser.add_argument("--significance-level", type=float, default=1.)
    parser.add_argument("--velocity-ratio", type=float, default=2.)
    parser.add_argument("--opening-angle-pi", type=float, default=0.1, help="velocity opening angle divided by pi")
    parser.add_argument("--memory-gib", type=float, default=4.)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    run_dir = (args.run_dir or select_run(runtime_paths().runs_root, "config.yaml")).resolve()
    if args.stellar_host is not None:
        if not args.snapshot or len(args.snapshot) != 1 or args.linking_length_kpc is None:
            parser.error("STF requires one --snapshot and an explicit --linking-length-kpc")
        controls = {key: getattr(args, key) for key in (
            "velocity_neighbors", "physical_neighbors", "cell_fraction", "outlier_threshold",
            "significance_level", "velocity_ratio", "opening_angle_pi", "linking_length_kpc")}
        controls["minimum_particles"] = args.stream_minimum_particles
        result = find_stellar_substructures(run_dir, args.snapshot[0], args.stellar_host,
                                            controls=controls, memory_gib=args.memory_gib,
                                            threads=args.threads)
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    stars = args.minimum_stars if args.minimum_stars is not None else required_integer("Minimum resolved star particles [integer]", 0, 100000)
    dm = args.minimum_dm if args.minimum_dm is not None else required_integer("Minimum resolved dark matter particles [integer]", 0, 100000)
    if not (0 <= stars <= 100000 and 0 <= dm <= 100000):
        raise ValueError("Particle thresholds must be 0–100000")
    snapshots = args.snapshot or find_galaxies_snapshot_ids(run_dir)
    if not snapshots:
        raise FileNotFoundError("No completed HDF5 snapshots were found")
    result = build_galaxy_catalogues(run_dir, snapshots,
                                     minimum_star_particles=stars, minimum_dm_particles=dm)
    result["csv_catalogues"] = export_galaxy_csvs(run_dir, snapshots)
    print(json.dumps(result, indent=2))
sys.path[:0] = [str(Path(__file__).resolve().parent / "core")]
from analysis import compare_runs


def compare_runs_main():
    """Write the paired analysis products from existing analysis.h5 files."""
    parser = argparse.ArgumentParser(description="Compare two runs with the shared catalogue, fixed-bin HMF and phase readers.\n\nUses core/analyze.py for consistent mass definitions, redshift matching,\nGarwood counting intervals and explicit resolution cuts. No substitute power\nspectrum is generated when a run's CLASS spectrum is absent.")
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--name", default="comparison")
    args = parser.parse_args()
    missing = [run for run in (args.run_a, args.run_b) if not (run / "results/analysis.h5").is_file()]
    if missing:
        parser.error("Paired analysis requires the full results/analysis.h5 bundle. Selected graph bundles have a different schema. Create full analysis in a separate run copy, or explicitly archive existing results with core/analyze.py --compare-with ... --overwrite. Missing: " + ", ".join(map(str, missing)))
    print(compare_runs(args.run_a.resolve(), args.run_b.resolve(), name=args.name))

run_halos_SUITE = Path(__file__).resolve().parent
sys.path[:0] = [str(run_halos_SUITE), str(run_halos_SUITE / "core")]
from halos import run_all
from analysis import rockstar_host_table, snapshot_ids as run_halos_snapshot_ids


def ensure_rockstar_tools() -> None:
    """Build the bundled HDF5 reader and parent-ID helper when needed.

    Rockstar's plain ``make`` binary cannot read GADGET HDF5 snapshots. The
    link dependency is checked before every run so an older local build is
    upgraded once, while a working build is left untouched.
    """
    source = run_halos_SUITE / "core" / "software" / "rockstar"
    binary = source / "rockstar"
    linked = subprocess.run(
        ["otool", "-L", str(binary)] if sys.platform == "darwin" else ["ldd", str(binary)],
        capture_output=True, text=True, check=False,
    ) if binary.is_file() else None
    if linked is None or linked.returncode != 0 or "hdf5" not in linked.stdout.lower():
        if not shutil.which("pkg-config"):
            raise RuntimeError("pkg-config and the HDF5 development library are required to build Rockstar")
        flags = subprocess.check_output(["pkg-config", "--cflags", "--libs", "hdf5"], text=True).strip()
        subprocess.run(["make", "with_hdf5", f"HDF5_FLAGS=-DH5_USE_16_API -DENABLE_HDF5 {flags}"],
                       cwd=source, check=True)
    if not (source / "util" / "find_parents").is_file():
        subprocess.run(["make", "parents"], cwd=source, check=True)


def export_catalogues(run_dir: Path) -> list[str]:
    """Write compact, unit-labelled CSV views of finished Rockstar catalogs.

    Rockstar uses -1 for a host PID. Shift IDs by one so zero can represent
    a host's ParentID without conflating it with Rockstar halo ID zero.
    """
    paths = []
    out = run_dir / "results" / "rockstar_catalogues"
    out.mkdir(parents=True, exist_ok=True)
    for snap in run_halos_snapshot_ids(run_dir):
        table = rockstar_host_table(run_dir, snap)
        if not table or "id" not in table or "pid" not in table:
            continue
        fields = {
            "HaloID": "id", "ParentID": "pid", "Mvir_Msun_h": "mvir",
            "Rvir_kpc_h": "rvir", "x_Mpc_h": "x", "y_Mpc_h": "y", "z_Mpc_h": "z",
            "vx_km_s": "vx", "vy_km_s": "vy", "vz_km_s": "vz",
            "Spin": "spin", "Concentration": "c",
        }
        available = {label: key for label, key in fields.items() if key in table}
        path = out / f"snapshot_{snap:03d}_halos.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(available))
            writer.writeheader()
            for index in range(len(table["id"])):
                row = {label: table[key][index].item() for label, key in available.items()}
                row["HaloID"] = int(row["HaloID"]) + 1
                row["ParentID"] = max(0, int(row["ParentID"]) + 1)
                writer.writerow(row)
        paths.append(str(path))
    return paths


def run_halos_main() -> None:
    """Allow halo finding to be repeated without touching GADGET evolution."""
    parser = argparse.ArgumentParser(description='Run Rockstar 6D halos, GADGET FOF/SUBFIND, or both on saved snapshots.')
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--finder", choices=["rockstar", "fof", "both"])
    parser.add_argument("--min-particles", type=int)
    parser.add_argument('--campaign-request',type=Path,help='explicit native finder/tree stages and saved snapshot selections')
    parser.add_argument("--fof-linking-length", type=float,
                        help="required FOF linking length in mean inter-particle spacings")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.campaign_request:
        from evolution import run_saved_finder_campaign
        result=run_saved_finder_campaign(run_dir,json.loads(args.campaign_request.read_text()),dry_run=args.dry_run)
        print(json.dumps(result,indent=2));return
    if args.finder is None or args.min_particles is None:parser.error('Declare --finder and --min-particles, or --campaign-request')
    finder = args.finder
    minimum = args.min_particles
    if not 2 <= minimum <= 100000:
        raise ValueError("Minimum halo particle count must be 2–100000")
    if finder in {"fof", "both"}:
        if args.fof_linking_length is None:
            parser.error("FOF halo finding requires an explicit --fof-linking-length")
        if not 0 < args.fof_linking_length < 1:
            raise ValueError("FOF linking length must be finite and in (0, 1) mean spacings")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    hmf_minimum = int(cfg["analysis"]["hmf_min_particles"])
    if minimum != hmf_minimum:
        raise ValueError(
            "--min-particles must equal analysis.hmf_min_particles (%d) so the finder and HMF selection agree"
            % hmf_minimum
        )
    cfg["halos"].update(fof=finder in {"fof", "both"},
                         subfind=finder in {"fof", "both"},
                         rockstar=finder in {"rockstar", "both"},
                         consistent_trees=False, min_particles=minimum)
    if finder in {"fof", "both"}:
        cfg["halos"]["fof_linking_length"] = float(args.fof_linking_length)
    if finder in {"rockstar", "both"} and not args.dry_run:
        ensure_rockstar_tools()
    status = run_all(run_dir, cfg, dry_run=args.dry_run)
    if not args.dry_run:
        status["rockstar_csv"] = export_catalogues(run_dir)
    print(json.dumps(status, indent=2, default=str))

def cosmography_main():
    from core.cosmography import cosmography_main as execute
    return execute()

def cosmography_report_main():
    from core.cosmography import cosmography_report_main as execute
    return execute()

def streams_main():
    from core.stream.workflow import streams_main as execute
    return execute()


def scientific_figures_main():
    from core.research.display import scientific_figures_main as execute
    return execute()

from core.workflow import workflow_main, run_spec_main
from core.workflow import workflow_products_main
from core.research.zooms import zoom_campaign_main, select_zoom_targets_main
from core.research.zooms import zoom_diagnostics_main
from core.research.pilots import pilots_main
from core.research.release import clean_release_main
from core.research.checkpoints import checkpoint_marker_main
from core.blender_scene import scene_main
from core.research.display import display_products_main
from core.research.generator_controls import native_music_transfer_main, native_music_mixed2lpt_main
from core.research.errors import error_budget_main
from core.research.convergence import convergence_main
from core.research.findings import findings_main
from core.resources import resources_main, resource_comparison_main
from core.research.errors import error_ensemble_main
from core.research.errors import error_residuals_main
from core.research.errors import native_error_envelopes_main
from core.apertures import native_zoom_support_main
from core.matched import matched_hydro_pilot_main
from core.research.pilots import pilot_recipes_main
from core.research.restarts import restart_comparison_main
from core.research.release import release_campaign_recipes_main
from core.research.release import release_blender_control_main
from core.research.measurements import convergence_budget_collection_main
from core.research.measurements import convergence_measurement_main
from core.sidm_control_measurements import isolated_history_main, angular_control_main, control_history_main, control_comparison_main, control_ensemble_main
from core.sidm_control_measurements import control_expectation_main
from core.sidm_control_measurements import control_clock_mapping_main
from core.sidm_control_measurements import control_native_clock_main
from core.sidm_control_measurements import control_refinement_measurement_main
from core.sidm_control_products import sidm_control_products_main
from core.sidm_control_products import control_population_main
from core.sidm_control_products import canonical_population_main
from core.sidm_control_products import control_campaign_main
from core.sidm_control_products import reference_build_main
from core.sidm_control_products import control_refinement_main
from core.sidm_analysis import sidm_science_main
from core.sidm_analysis import joint_response_campaign_main
from core.sidm_physics import sidm_model_main
from core.initial_conditions import twofluid_relative_main
from core.initial_conditions import twofluid_nonlinear_main
from core.initial_conditions import twofluid_eulerian_seed_main
from black_holes import dm_wake_force_main
from black_holes import dm_wake_kernel_comparison_main
from black_holes import dm_wake_schedule_main
from black_holes import dm_wake_state_response_main
from black_holes import dm_wake_state_table_main
from black_holes import dm_wake_context_main
from core.initial_conditions import twofluid_refinement_main
from core.initial_conditions import twofluid_native_comparison_main
from core.initial_conditions import twofluid_native_family_main
from core.initial_conditions import twofluid_projection_refinement_main
from core.initial_conditions import twofluid_refinement_campaign_main
from black_holes import dm_wake_contrast_main
from black_holes import dm_wake_ensemble_main
from black_holes import dm_wake_response_main
from black_holes import dm_wake_native_table_main
from core.stream_forward_model import native_stream_worldline_main
from core.stream_forward_model import native_stream_comparison_main
from core.stream_forward_model import native_late_threeway_main
from core.stream_forward_model import native_counterfactual_ic_main
from core.stream_forward_model import native_counterfactual_campaign_main
from core.initial_conditions import twofluid_particle_main
from core.initial_conditions import twofluid_native_ic_main
from core.initial_conditions import twofluid_window_main
from core.zoom import physical_target_main
from core.sidm_campaign import sidm_control_plan_main
from core.research.findings import repair_receipt_main

COMMANDS.update({
    'run-spec': run_spec_main,
    'twofluid-relative': twofluid_relative_main,
    'twofluid-nonlinear': twofluid_nonlinear_main,
    'twofluid-refinement': twofluid_refinement_main,
    'twofluid-native-comparison': twofluid_native_comparison_main,
    'twofluid-native-family': twofluid_native_family_main,
    'twofluid-projection-refinement': twofluid_projection_refinement_main,
    'twofluid-refinement-campaign': twofluid_refinement_campaign_main,
    'twofluid-eulerian-seed': twofluid_eulerian_seed_main,
    'dm-wake-force': dm_wake_force_main,
    'dm-wake-kernel-comparison': dm_wake_kernel_comparison_main,
    'dm-wake-schedule': dm_wake_schedule_main,
    'dm-wake-contrast': dm_wake_contrast_main,
    'dm-wake-ensemble': dm_wake_ensemble_main,
    'dm-wake-response': dm_wake_response_main,
    'dm-wake-state-response': dm_wake_state_response_main,
    'dm-wake-state-table': dm_wake_state_table_main,
    'dm-wake-context': dm_wake_context_main,
    'dm-wake-native-table': dm_wake_native_table_main,
    'native-stream-worldlines': native_stream_worldline_main,
    'native-stream-comparison': native_stream_comparison_main,
    'native-late-threeway': native_late_threeway_main,
    'native-counterfactual-ic': native_counterfactual_ic_main,
    'native-counterfactual-campaign': native_counterfactual_campaign_main,
    'twofluid-particles': twofluid_particle_main,
    'twofluid-native-ic': twofluid_native_ic_main,
    'twofluid-window': twofluid_window_main,
    'physical-target': physical_target_main,
    'sidm-science': sidm_science_main,
    'sidm-response-campaign': joint_response_campaign_main,
    'sidm-model': sidm_model_main,
    'sidm-control-plan': sidm_control_plan_main,
    'sidm-control-population': control_population_main,
    'sidm-canonical-population': canonical_population_main,
    'sidm-control-campaign': control_campaign_main,
    'sidm-control-history': control_history_main,
    'sidm-control-clock-mapping': control_clock_mapping_main,
    'sidm-control-native-clock': control_native_clock_main,
    'sidm-control-comparison': control_comparison_main,
    'sidm-control-ensemble': control_ensemble_main,
    'sidm-control-expectation': control_expectation_main,
    'sidm-reference-build': reference_build_main,
    'sidm-control-refinement': control_refinement_main,
    'sidm-refinement-measurement': control_refinement_measurement_main,
    'repair-receipt': repair_receipt_main,
    'error-budget': error_budget_main,
    'error-ensemble': error_ensemble_main,
    'error-residuals': error_residuals_main,
    'native-error-envelopes': native_error_envelopes_main,
    'native-sidm-zoom-support': native_zoom_support_main,
    'matched-hydro-pilot': matched_hydro_pilot_main,
    'compile-pilot-family': pilot_recipes_main,
    'restart-comparison': restart_comparison_main,
    'compile-release-campaign': release_campaign_recipes_main,
    'release-blender-control': release_blender_control_main,
    'convergence-budgets': convergence_budget_collection_main,
    'convergence-campaign': convergence_main,
    'convergence-measurement': convergence_measurement_main,
    'isolated-control-history': isolated_history_main,
    'sidm-angular-control': angular_control_main,
    'sidm-control-products': sidm_control_products_main,
    'findings': findings_main,
    'production-resources': resources_main,
    'resource-compare': resource_comparison_main,
    'workflow': workflow_main,
    'workflow-products': workflow_products_main,
    'zoom-campaign': zoom_campaign_main,
    'select-zoom-targets': select_zoom_targets_main,
    'measure-zoom-campaign': zoom_diagnostics_main,
    'pilots': pilots_main,
    'clean-release':clean_release_main,
    'checkpoint-marker':checkpoint_marker_main,
    'measured-scene':scene_main,
    'display-products':display_products_main,
    'native-music-transfer':native_music_transfer_main,
    'native-music-mixed2lpt':native_music_mixed2lpt_main,
    'scientific-figures': scientific_figures_main,
    'streams': streams_main,
    'stream-products': _small_cli.stream_products_main,
    'analysis': run_analysis_main,
    'analysis-batch': analysis_batch_main,
    'audit-data-contract': audit_data_contract_main,
    'audit-ic': audit_ic_main,
    'audit-native-handoff': audit_native_handoff_main,
    'audit-tree-integrity': audit_tree_integrity_main,
    'propose-tree-repair': propose_tree_repair_main,
    'audit-zoom-phases': audit_zoom_phases_main,
    'audit-zoom-regions': audit_zoom_regions_main,
    'blender': run_blender_main,
    'bootstrap': bootstrap_main,
    'build-identity-graph': build_identity_graph_main,
    'class': run_class_main,
    'cosmography': cosmography_main,
    'cosmography-report': cosmography_report_main,
    'compare-finders': compare_finders_main,
    'compare-runs': compare_runs_main,
    'event-cadence': run_event_cadence_main,
    'export-stellar-observer': export_stellar_observer_main,
    'find-galaxies': find_galaxies_main,
    'finder-campaign': run_finder_campaign_main,
    'gadget': run_gadget_main,
    'halos': run_halos_main,
    'measure-halo-shapes': measure_halo_shapes_main,
    'measure-stream-heating': measure_stream_heating_main,
    'audit-hydro-ssp': audit_hydro_ssp_main,
    'audit-hydro-events': audit_hydro_events_main,
    'audit-hydro-history': audit_hydro_history_main,
    'audit-black-hole-events': audit_hydro_history_main,
    'compile-stellar-model': compile_stellar_model_main,
    'hydro-comparison': hydro_comparison_main,
    'black-hole-environment': black_hole_environment_main,
    'prepare-source-controls': prepare_source_controls_main,
    'native-source-events': native_source_events_main,
    'native-source-overlay': native_source_overlay_main,
    'comparison-movie': comparison_movie_main,
    'timeline-wall': timeline_wall_main,
    'multifield-movie': multifield_movie_main,
    'particle-origin-movie': particle_origin_movie_main,
    'velocity-flow': velocity_flow_main,
    'allsky-map': allsky_map_main,
    'halo-formation-movie': halo_formation_movie_main,
    'halo-story': halo_story_main,
    'halo-collision-movie': halo_collision_movie_main,
    'planetarium-movie': planetarium_movie_main,
    'stellar-stream-movie': stellar_stream_movie_main,
    'lens-map': lens_map_main,
    'cosmic-flythrough': cosmic_flythrough_main,
    'resolution-reveal': resolution_reveal_main,
    'parameter-explorer': parameter_explorer_main,
    'halo-anatomy': halo_anatomy_main,
    'void-explorer': void_explorer_main,
    'void-boundary-history': void_boundary_history_main,
    'halo-fingerprint': halo_fingerprint_main,
    'metallicity-maps': metallicity_maps_main,
    'merger-tree-map': merger_tree_map_main,
    'results-gallery': results_gallery_main,
    'filament-flythrough': filament_flythrough_main,
    'void-particle-history': void_particle_history_main,
    'select-merger-tree': select_merger_tree_main,
    'halo-structure': halo_structure_main,
    'environment': environment_main,
    'observe-stellar-catalogue': observe_stellar_catalogue_main,
    'package': package_suite_main,
    'query-identity-graph': query_identity_graph_main,
    'research-acceptance': run_research_acceptance_main,
    'stage-zoom-inputs': stage_zoom_inputs_main,
    'prepare-generated-flow': prepare_generated_flow_main,
    'start-epoch-campaign': run_start_epoch_campaign_main,
    'trace-zoom': trace_zoom_main,
    'velociraptor': run_velociraptor_main,
    'zoom': run_zoom_main,
})


def verify_main():
    from verification import main as verify
    return verify()

from black_holes import SCENE_COMMANDS as _BLACK_HOLE_COMMANDS
from visualization import SCENE_COMMANDS as _VISUAL_COMMANDS
COMMANDS.update({
    'verify': verify_main,
    'lensing': _BLACK_HOLE_COMMANDS['schwarzschild-lensing'],
    'export-merger': _VISUAL_COMMANDS['export-gadget-merger'],
    'dome': _VISUAL_COMMANDS['fulldome-planetarium'],
})

from core.campaigns import main as native_prepare_main
COMMANDS['native-prepare'] = native_prepare_main
from core.run_results import results_main
COMMANDS['results'] = results_main

def main(argv=None):
    """Dispatch a command while preserving its argument parser and exit status."""
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {'--help', '-h'}:
        print('Usage: python commands.py COMMAND [arguments]\n')
        for name in sorted(COMMANDS):
            print('  ' + name)
        return 0
    name = args.pop(0)
    if name not in COMMANDS:
        raise SystemExit('Unknown command: ' + name + '; use --help for commands')
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()) + ' ' + name, *args]
        return COMMANDS[name]()
    finally:
        sys.argv = previous

if __name__ == '__main__':
    raise SystemExit(main())
