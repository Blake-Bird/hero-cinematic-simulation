"""Consolidated sidm implementation source.

Sections retain independent Python namespaces through core.__init__.
Import their established core.<section> names; public commands use those names.
"""

if __name__ == '__main__':
    import sys as _bundle_sys
    import runpy as _bundle_runpy
    from pathlib import Path as _BundlePath
    _bundle_sys.path.insert(0, str(_BundlePath(__file__).resolve().parents[1]))
    import core as _bundle_core
    if len(_bundle_sys.argv) >= 3 and _bundle_sys.argv[1] == '--module':
        _logical_module = _bundle_sys.argv[2]
        del _bundle_sys.argv[1:3]
    else:
        _logical_module = 'sidm_analysis'
    if (_logical_module not in _bundle_core._module_layout
            or _bundle_core.source_path(_logical_module).resolve() != _BundlePath(__file__).resolve()):
        raise SystemExit('Select a documented logical module owned by this source bundle')
    _namespace = _bundle_runpy.run_module('core.' + _logical_module, run_name='__main__', alter_sys=True)
    _entry_name = {'sidm_analysis': 'sidm_science_main'}.get(_logical_module)
    if _entry_name is not None:
        raise SystemExit(_namespace[_entry_name]() or 0)
    raise SystemExit(0)
raise ImportError('Import an established core.<section> namespace or use commands.py')







# @research2-module sidm_analysis
"""SIDM science from complete, source-bound native particle histories.

Every estimator preserves its declared aperture, measured epoch and support.
Shell velocity moments measure a coarse-grained phase-space heat flux; macro
event counts and cumulative hazards are separate numerical diagnostics.
Cross-section inference belongs to independently calibrated forward models.
"""


import argparse
import copy
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from .foundation import canonical_digest, file_sha256, KPC_CM, SECONDS_PER_YEAR, G_MPC_KMS2_MSUN, atomic_json
from .data import exact_uint64_ids, export_numeric_audit, verify_numeric_audit
from .data import numeric_audit_arrays_path, numeric_audit_source_bindings
from .foundation import merge_source_bindings as _merge
from .sidm_physics import _positive
from .sidm_measurements import HISTORY_FIELDS, SIDM_FIELDS, _finite, _integer, _supported, classify_sidm_history, component_centroids_and_misalignment, measure_core_definitions, measure_sidm_shells, satellite_survival_and_transport

G_KPC_KMS2_MSUN = G_MPC_KMS2_MSUN * 1000
KMS_TO_KPC_GYR = 1e5 * SECONDS_PER_YEAR * 1e9 / KPC_CM


def _path(base, value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(base) / p).resolve()


def _load_product(reference, base, memory_bytes):
    """Read a sealed numeric product with bounded expanded arrays and sources."""
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
        raise ValueError('Exact path/SHA256 measured product reference required')
    if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, (int,float)) or not math.isfinite(memory_bytes) or memory_bytes <= 0:
        raise ValueError('Positive finite product read workspace required')
    p = _path(base, reference['path'])
    if p.stat().st_size * 8 > memory_bytes:
        raise MemoryError('Scientific product metadata exceeds its allocation')
    if file_sha256(p) != reference['sha256']:
        raise ValueError('Scientific product reference changed')
    # Bound ZIP expansion before the ordinary contract reader materializes it.
    raw = json.loads(p.read_text())
    if raw.get('schema') == 'cosmography-generation-v1':
        result = _load_cosmography_observables(p, raw, memory_bytes)
        if file_sha256(p) != reference['sha256']:
            raise ValueError('Cosmography generation changed during intake')
        return result
    archive = numeric_audit_arrays_path(p, raw)
    expected = p.with_name(p.name+'.data')
    if expected.is_symlink() or archive.parent != expected:
        raise ValueError('Scientific numeric arrays must use their original sibling bundle')
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if len(members) > 4096 or sum(row.file_size for row in members) * 6 > memory_bytes:
            raise MemoryError('Scientific arrays exceed the expanded workspace')
    report = verify_numeric_audit(p)
    sources = numeric_audit_source_bindings(report)
    _merge(sources, {str(p): reference['sha256'], str(p.with_name(p.name+'.sha256')):
                     file_sha256(p.with_name(p.name+'.sha256'))})
    directory = archive.parent
    for entry in directory.rglob('*'):
        if entry.is_symlink():
            raise ValueError('Scientific product bundles require regular retained assets')
        if entry.is_file():
            _merge(sources, {str(entry.resolve()): file_sha256(entry)})
    with np.load(archive, allow_pickle=False) as f:
        # NPZ member reads already allocate independent arrays.
        arrays = {name: f[name] for name in f.files}
    if file_sha256(p) != reference['sha256']:
        raise ValueError('Scientific product changed during intake')
    return report, arrays, sources


def _load_cosmography_observables(path, manifest, memory_bytes):
    """Bridge actual frozen Hubble estimators, preserving their source units."""
    from .observation_data import read_frozen
    from .data import verify_hdf5_dataset_payloads
    sources = {str(path): file_sha256(path)}
    for name, digest in manifest['files_sha256'].items():
        file = (path.parent / name).resolve()
        if not file.is_relative_to(path.parent) or file.is_symlink() or file_sha256(file) != digest:
            raise ValueError('Actual frozen Hubble generation payload changed')
        sources[str(file)] = digest
    product = path.parent / 'product.json'
    if product.stat().st_size * 8 > memory_bytes:
        raise MemoryError('Frozen Hubble estimator metadata exceeds SIDM workspace')
    payload = read_frozen(product)
    if payload.get('execution_status') != 'COMPLETED':
        raise ValueError('SIDM estimator response requires an actual completed frozen fit')
    numeric = path.parent / 'numeric.hdf5'
    verify_hdf5_dataset_payloads(numeric, require_contract=True, memory_bytes=memory_bytes)
    _merge(sources, manifest['input_hashes'])
    for name, digest in manifest['source_hashes'].items():
        code = Path(name)
        if not code.is_absolute():
            code = (Path(__file__).parent.parent / name if code.parts[0] in {'core', 'configs'}
                else Path(__file__).parent / name)
        if file_sha256(code) != digest:
            raise ValueError('Frozen Hubble estimator owning source changed')
        _merge(sources, {str(code.resolve()): digest})
    for name, digest in sources.items():
        if file_sha256(name) != digest:
            raise ValueError('Frozen Hubble estimator input/source changed')
    arrays, datasets = {}, {}
    for observable, value in payload['observables'].items():
        mean = _finite([value['value']], (1,), 'frozen cosmography observable')
        sigma = _finite([value['sigma']], (1,), 'frozen cosmography uncertainty')
        if sigma[0] <= 0:
            raise ValueError('Frozen cosmography uncertainty must remain positive')
        arrays[observable] = mean
        arrays[observable + '_sigma'] = sigma
        arrays[observable + '_supported'] = np.ones(1, bool)
        for name in (observable, observable + '_sigma'):
            datasets[name] = {'units': value['units'], 'quantity': 'actual frozen estimator ' + name}
        datasets[observable + '_supported'] = {'units': 'dimensionless', 'quantity': 'completed finite frozen-fit measurement support; scientific calibration separate'}
    report = {'schema': 'cosmography-frozen-observable-adapter-v1', 'execution_status': 'COMPLETE',
        'matched_experiment': manifest.get('matched_experiment'), 'method': payload['method'],
        'frozen_hash': payload['frozen_hash'], 'source_hashes': sources,
        'numeric_data': {'array_contract': {'datasets': datasets}},
        'scientific_acceptance': payload['scientific_acceptance'],
        'measurement_support_scope': 'completed numerical fit; inference coverage and physics support require independent evidence'}
    return report, arrays, sources


def _source_code_bindings():
    from . import data, halos, observation_geometry, zoom, gravity_fields, stream_common, stream_forward_model, sidm_physics, sidm_measurements
    from core.stream import encounters
    return {str(Path(m.__file__).resolve()): file_sha256(m.__file__)
            for m in (data, halos, observation_geometry, zoom, encounters, gravity_fields,
                stream_common, stream_forward_model, sidm_physics, sidm_measurements)} | {str(Path(__file__).resolve()): file_sha256(__file__)}


def _native_diagnostics(root, snapshot, selected_ids, request):
    """Read fine-DM counters and support fields in the requested exact-ID order.

    Counts remain uint64; exposure and support fractions are dimensionless.
    Scattering support length converts native dataset units to proper kpc.
    Every selected identity must occur once when fields are available. Optional
    absent fields return an empty mapping; partially read arrays are not exposed.
    """
    import h5py
    from .halos import _snapshot_files
    from .data import dataset_to_cgs_factor

    selected = exact_uint64_ids(selected_ids, allow_empty=True)
    workspace = _positive(request['memory_gib'], 'native diagnostic workspace') * 1024**3
    if type(request['require_sidm_diagnostics']) is not bool:
        raise ValueError('Native diagnostics requirement must be Boolean')
    count = len(selected)
    # Account for retained fields, exact IDs, index and masks before streaming.
    if count * 256 > workspace:
        raise MemoryError('Exact SIDM diagnostic identity join exceeds its workspace')
    if not count:
        if request['require_sidm_diagnostics']:
            raise ValueError('Native SIDM diagnostics require a nonempty fine-DM selection')
        return {}
    chunk_rows = min(131072, int((workspace-count*256)//256))
    if chunk_rows < 1:
        raise MemoryError('Native SIDM diagnostics have no remaining streaming workspace')
    order = np.argsort(selected)
    sorted_ids = selected[order]
    found = np.zeros(count, bool)
    available = True
    values = {name: np.empty(count, np.uint64 if name == 'interaction_count' else float)
              for name in SIDM_FIELDS}
    for path in _snapshot_files(Path(root), snapshot):
        with h5py.File(path, 'r') as handle:
            parameters = handle['Parameters'].attrs
            scale = float(handle['Header'].attrs['Time']) if int(parameters['ComovingIntegrationOn']) else 1.
            hubble = float(parameters['HubbleParam'])
            # Fine type-1 particles scatter; coarse gravity particles do not
            # define another elastic species or a zero-exposure population.
            group = handle.get('PartType1')
            if group is None:
                continue
            rows = len(group['ParticleIDs'])
            for start in range(0, rows, chunk_rows):
                stop = min(rows, start+chunk_rows)
                ids = exact_uint64_ids(group['ParticleIDs'][start:stop])
                location = np.searchsorted(sorted_ids, ids)
                keep = location < count
                keep[keep] = sorted_ids[location[keep]] == ids[keep]
                if not keep.any():
                    continue
                targets = order[location[keep]]
                if found[targets].any():
                    raise ValueError('Native SIDM selected identity repeats across chunks or shards')
                found[targets] = True
                if not all(field in group for field in SIDM_FIELDS.values()):
                    available = False
                    continue
                for name, field in SIDM_FIELDS.items():
                    dataset = group[field]
                    if dataset.shape != (rows,):
                        raise ValueError('Native SIDM field is not aligned to fine-DM identities: '+field)
                    value = dataset[start:stop][keep]
                    if name == 'interaction_count':
                        value = exact_uint64_ids(value, unique=False)
                    else:
                        value = _finite(value, (len(targets),), field)
                        if np.any(value < 0) or (name == 'coarse_support_fraction' and np.any(value > 1)):
                            raise ValueError('Native SIDM diagnostic leaves its physical domain: '+field)
                        if name == 'support_length_proper_kpc':
                            factor = _positive(dataset_to_cgs_factor(dataset.attrs, scale, hubble), 'SIDM support length conversion')
                            value = value * factor / KPC_CM
                            if not np.isfinite(value).all():
                                raise FloatingPointError('Native SIDM support length conversion overflow')
                    values[name][targets] = value
    if not available or not found.any():
        if request['require_sidm_diagnostics']:
            raise ValueError('Native SIDM snapshot fields required for requested interaction science')
        return {}
    if not found.all():
        raise ValueError('Native SIDM diagnostic census misses selected exact identities')
    return values


def validate_sidm_history_request(request):
    if request.get('schema') != 'sidm-halo-history-request-v1':
        raise ValueError('Versioned native SIDM history request required')
    for name in ('maximum_particles', 'maximum_epochs', 'maximum_halos', 'minimum_shell_particles', 'minimum_central_particles'):
        _integer(request[name], name)
    _positive(request['memory_gib'], 'history workspace')
    if type(request['require_sidm_diagnostics']) is not bool:
        raise ValueError('SIDM diagnostic requirement must be explicit boolean')
    epochs = request['epochs']
    if not isinstance(epochs, list) or not epochs or len(epochs) > request['maximum_epochs']:
        raise ValueError('Nonempty bounded native epochs required')
    if len({row['snapshot'] for row in epochs}) != len(epochs):
        raise ValueError('Native epoch selection contains duplicates')
    for row in epochs:
        _integer(row['snapshot'], 'native snapshot', 0)
        _positive(row['elapsed_time_gyr'], 'measured elapsed time', zero=True)
        if not row.get('selection'):
            raise ValueError('Each epoch needs sealed native halo membership and center measurements')
    times = [row['elapsed_time_gyr'] for row in epochs]
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError('Native physical times must increase, never inferred from output numbers')
    match = request['matching']
    required = {'phase_seed', 'linear_bundle_sha256', 'background_sha256', 'halo_match_namespace', 'halo_match_id', 'baryon_control', 'baryon_control_sha256'}
    if not required <= set(match) or not isinstance(match['baryon_control'], str) or not match['baryon_control']:
        raise ValueError('SIDM history needs exact primordial/background/halo/baryon matching rules')
    _integer(match['phase_seed'], 'IC phase seed', 0)
    if (not isinstance(match['halo_match_namespace'], str) or not match['halo_match_namespace']
            or type(match['halo_match_id']) is not int or not 0 <= match['halo_match_id'] <= 2 ** 64 - 1):
        raise ValueError('Exact common matched halo identity required')
    for name in ('linear_bundle_sha256', 'background_sha256', 'baryon_control_sha256'):
        if not isinstance(match[name], str) or len(match[name]) != 64 or any(c not in '0123456789abcdef' for c in match[name]):
            raise ValueError('Exact lowercase SHA256 matched background/linear inputs required')
    if request['dark_matter_model'] not in {'collisionless', 'elastic_sidm'}:
        raise ValueError('Explicit nonlinear interaction model required')
    if request['dark_matter_model'] == 'elastic_sidm' and not request['require_sidm_diagnostics']:
        raise ValueError('SIDM interaction science cannot omit native scattering diagnostics')
    mapping = request['species_components']
    if set(mapping) != set(map(str, request['species'])) or mapping.get('1') != 'dm':
        raise ValueError('Declare exactly the native fine-DM and other component species mappings')
    if any(label == 'dm' and species != '1' for species, label in mapping.items()):
        raise ValueError('SIDM science DM profiles use fine type-1 representatives; coarse type-2 is a separate gravity/support component')
    return request


def _stage_native_bindings(parameters):
    from .workflow import verify_completed_receipt
    receipt_path = Path(parameters['native_stage_receipt']).expanduser().resolve()
    if receipt_path.stat().st_size * 8 > parameters['request']['memory_gib'] * 1024 ** 3:
        raise MemoryError('Native evolution receipt exceeds allocated metadata workspace')
    receipt = verify_completed_receipt(receipt_path)
    root = Path(receipt['generation']).resolve()
    native = Path(parameters['native_run_directory']).expanduser().resolve()
    if not native.is_relative_to(root):
        raise ValueError('Native history directory lies outside its completed evolution generation')
    sources = {str(root / name): digest for name, digest in receipt['files_sha256'].items()}
    _merge(sources, receipt['source_hashes'])
    _merge(sources, receipt['input_hashes'])
    _merge(sources, {str(receipt_path): file_sha256(receipt_path)})
    request = parameters['request']
    if request.get('matching'):
        import yaml
        config = native / 'config.yaml'
        if sources.get(str(config)) != file_sha256(config):
            raise ValueError('SIDM matching requires the original owned native configuration')
        cfg = yaml.safe_load(config.read_text())
        matching = request['matching']
        if cfg['simulation']['random_seed'] != matching['phase_seed']:
            raise ValueError('SIDM matching relabels the actual primordial phase seed')
        if request.get('dark_matter_model') is not None:
            actual_sidm = cfg.get('sidm', {}).get('enabled', False)
            if type(actual_sidm) is not bool or actual_sidm != (request['dark_matter_model'] == 'elastic_sidm'):
                raise ValueError('SIDM science relabels the actual evolved nonlinear interaction model')
        for name in ('linear_bundle_sha256', 'background_sha256'):
            if matching[name] not in sources.values():
                raise ValueError('SIDM matching references an unbound original linear/background product: ' + name)
        baryon_signature = canonical_digest({'physics': cfg['simulation']['physics'],
            **{name: cfg.get(name, {}) for name in ('hydro', 'black_holes', 'stellar_evolution', 'chemistry',
                'external_potential', 'baryonic_force', 'stellar_sources')}})
        if matching.get('baryon_control_sha256') != baryon_signature:
            raise ValueError('Matched baryonic control must hash the actual saved native physics')
    return sources


def validate_sidm_membership_request(request):
    if request.get('schema') != 'sidm-native-membership-request-v1':
        raise ValueError('Native SIDM host-membership producer request required')
    for name in ('snapshot', 'host_row'):
        _integer(request[name], name, 0)
    _integer(request['maximum_particles'], 'complete native census ceiling')
    _positive(request['memory_gib'], 'native membership workspace')
    _positive(request['support_radius_proper_kpc'], 'complete host aperture')
    if not request.get('matching') or not request.get('particle_namespace'):
        raise ValueError('Native membership requires exact matched halo and native particle namespace')
    return request


def publish_sidm_membership(parameters, destination):
    """Measured FOF-center aperture census, with actual native clock/velocity.

    This deliberately declares an aperture selection. It does not label every
    enclosed particle as bound to a satellite. Satellite-bound selections use
    the separate exact SUBFIND membership contract.
    """
    from .observation_geometry import read_native_gravitating_particles
    from .zoom import read_host_catalogue
    request = validate_sidm_membership_request(parameters['request'])
    sources = _stage_native_bindings(parameters)
    population, particles = read_native_gravitating_particles(parameters['native_run_directory'], request['snapshot'],
        species=request['species'], maximum_particles=request['maximum_particles'], memory_gib=request['memory_gib'], namespace=request['particle_namespace'])
    catalogue = read_host_catalogue(parameters['native_run_directory'], request['snapshot'], memory_gib=request['memory_gib'], host_row=request['host_row'])
    a, box, h = catalogue['frame']
    center = np.asarray(catalogue['positions'][request['host_row']], float) * a / h
    proper_box = float(box * a / h)
    aperture = request['support_radius_proper_kpc']
    if aperture >= proper_box / 2:
        raise ValueError('Host aperture must remain inside half the native proper periodic box')
    offsets = particles['position_proper_kpc'] - center
    offsets -= proper_box * np.rint(offsets / proper_box)
    selected = np.linalg.norm(offsets, axis=1) <= aperture
    if not selected.any():
        raise ValueError('Measured native halo aperture is physically empty')
    mass = particles['mass_msun'][selected]
    bulk = np.average(particles['velocity_peculiar_kms'][selected], axis=0, weights=mass)
    originals = dict(population['source_hashes'])
    for row in catalogue['sources'] + catalogue['snapshot_sources']:
        _merge(originals, {row['path']: row['sha256']})
    if any(sources.get(path) != sha for path, sha in originals.items()):
        raise ValueError('Halo center/membership source is outside actual completed native evolution')
    _merge(sources, originals)
    _merge(sources, _source_code_bindings())
    epoch = population['epoch']
    physical_time = epoch.get('cosmic_time_gyr') if epoch.get('canonical_time') != 'isolated_native_elapsed_time' else epoch['elapsed_time_gyr']
    if physical_time is None or not math.isfinite(physical_time):
        raise ValueError('Native host membership requires a measured background-bound physical clock')
    arrays = {'member_particle_ids': particles['particle_ids'][selected], 'center_proper_kpc': center,
        'bulk_velocity_km_s': bulk, 'member_species': particles['species'][selected],
        'particle_ids': particles['particle_ids'][selected], 'species': particles['species'][selected],
        'position_proper_kpc': center + offsets[selected], 'mass_msun': mass,
        'velocity_peculiar_kms': particles['velocity_peculiar_kms'][selected]}
    report = {'schema': 'native-halo-membership-measurement-v1', 'execution_status': 'COMPLETE',
        'snapshot': request['snapshot'], 'matching': request['matching'], 'request': request, 'epoch': epoch,
        'elapsed_time_gyr': float(physical_time), 'time_origin': 'cosmic age' if epoch.get('cosmic_time_gyr') is not None else 'native isolated time zero',
        'periodic_box_proper_kpc': proper_box, 'native_particle_source_hashes': originals, 'source_hashes': sources,
        'membership_definition': 'all requested native species inside declared proper FOF-center aperture; substructure retained',
        'bulk_definition': 'mass-weighted peculiar velocity of the same complete aperture'}
    specifications = {name: {'units': 'proper kpc' if name in {'center_proper_kpc', 'position_proper_kpc'} else 'km / s' if name in {'bulk_velocity_km_s', 'velocity_peculiar_kms'}
        else 'Msun' if name == 'mass_msun' else 'dimensionless', 'quantity': 'native measured ' + name.replace('_', ' ')} for name in arrays}
    for name in ('particle_ids', 'member_particle_ids'):
        specifications[name]['units'] = 'identity'
        specifications[name]['identity_namespace'] = request['particle_namespace']
    return export_numeric_audit(destination, report, arrays, specifications,
        algorithm=Path(__file__).resolve(), context={'epoch': epoch, 'coordinates': {'frame': 'native FOF measured proper center'}}, source_hashes=sources)


def sidm_input_bindings(parameters):
    request = parameters['request']
    kind = request.get('schema')
    base = parameters.get('request_base', '.')
    memory = _positive(request['memory_gib'], 'SIDM workspace') * 1024 ** 3
    references = []
    sources = _source_code_bindings()
    if kind == 'sidm-native-satellite-request-v1':
        validate_sidm_satellite_request(request)
        _merge(sources, _stage_native_bindings(parameters))
        references = [request['original_membership']]
    elif kind == 'sidm-native-membership-request-v1':
        validate_sidm_membership_request(request)
        _merge(sources, _stage_native_bindings(parameters))
    elif kind == 'sidm-halo-history-request-v1':
        validate_sidm_history_request(request)
        _merge(sources, _stage_native_bindings(parameters))
        references = [row['selection'] for row in request['epochs']]
    elif kind == 'sidm-matched-science-request-v1':
        validate_sidm_matched_request(request)
        references = [row['product'] for row in request['histories']]
    elif kind == 'sidm-observational-comparison-request-v1':
        validate_sidm_observations_request(request)
        references = [ref for row in request['comparisons'] for ref in (row['collisionless'], row['sidm'])]
    elif kind == 'sidm-derived-science-request-v1':
        validate_sidm_derived_request(request)
        references = request['inputs']
    elif kind == 'native-forward-observable-vector-request-v1':
        validate_forward_vector_request(request)
        _merge(sources, _stage_native_bindings(parameters))
        references = [row['product'] for row in request['observables']]
    elif kind in {'sidm-joint-response-request-v1','sidm-joint-response-training-request-v2'}:
        validate_joint_response_request(request)
        references = request['samples']
        references=[*references,*request.get('independent_calibration_receipts',[])]
    elif kind=='native-joint-response-calibration-request-v1':
        validate_joint_calibration_request(request)
        references=[request['training_response'],*[row[key] for row in request['pairs'] for key in ('baseline','varied')]]
    elif kind=='sidm-calibrated-joint-response-request-v1':
        validate_calibrated_response_request(request)
        references=[request['training_response'],*request['calibrations']]
    else:
        raise ValueError('Unknown SIDM science request')
    for ref in references:
        _, _, binding = _load_product(ref, base, memory)
        _merge(sources, binding)
    return sources


def validate_sidm_satellite_request(request):
    if request.get('schema') != 'sidm-native-satellite-request-v1':
        raise ValueError('Versioned native SIDM satellite-binding request required')
    for name in ('snapshot', 'subhalo_row'):
        _integer(request[name], name, 0)
    for name in ('maximum_particles', 'maximum_bound_particles', 'maximum_pair_evaluations', 'potential_chunk_rows'):
        _integer(request[name], name)
    _positive(request['memory_gib'], 'satellite measurement workspace')
    _positive(request['potential_softening_proper_kpc'], 'satellite potential softening')
    if type(request['require_sidm_diagnostics']) is not bool:
        raise ValueError('Satellite native diagnostics requirement must be explicit')
    if request['potential_gauge'] != 'instantaneous_satellite_escape_zero':
        raise ValueError('Implemented satellite self-potential uses zero at infinity; tidal saddles need a separate measured model')
    if request.get('cohort_policy', 'legacy_source_population') not in {'legacy_source_population', 'initial_bound_only'}:
        raise ValueError('Unknown native satellite original population policy')
    if not request.get('satellite_identity') or not request.get('particle_namespace'):
        raise ValueError('Source-bound satellite and native identity namespaces required')
    return request


def publish_sidm_satellite(parameters, destination):
    """Produce exact SUBFIND binding and conservative evaporation candidates.

    Self-potential is a measured bound-mass Plummer-softened direct sum with
    zero at infinity. Host tides and cosmological energy terms are excluded
    explicitly; energy crossings remain candidates for matched controls.
    """
    import h5py
    from .halos import _group_files, _snapshot_files, _bound_particle_slice
    from .observation_geometry import read_native_gravitating_particles
    request = validate_sidm_satellite_request(parameters['request'])
    sources = sidm_input_bindings(parameters)
    native_sources = _stage_native_bindings(parameters)
    original, original_arrays, _ = _load_product(request['original_membership'], parameters.get('request_base', '.'), request['memory_gib'] * 1024 ** 3)
    if original.get('schema') not in {'native-halo-membership-measurement-v1', 'native-satellite-binding-measurement-v1'}:
        raise ValueError('Satellite original members require an actual source-bound native population')
    original_ids = exact_uint64_ids(original_arrays.get('member_particle_ids', original_arrays.get('particle_ids')))
    original_namespace = original.get('particle_namespace', original.get('request', {}).get('particle_namespace'))
    if original_namespace != request['particle_namespace']:
        raise ValueError('Satellite transport changes the original native identity namespace')
    native = Path(parameters['native_run_directory']).resolve()
    catalogues = _group_files(native, request['snapshot'])
    snapshots = _snapshot_files(native, request['snapshot'])
    if not catalogues or not snapshots:
        raise ValueError('Native satellite producer needs the complete evolved SUBFIND and ordered snapshot')
    remaining = request['subhalo_row']
    offsets = lengths = None
    for path in catalogues:
        if native_sources.get(str(path.resolve())) != file_sha256(path):
            raise ValueError('Native satellite catalogue is outside its completed evolution generation')
        with h5py.File(path, 'r') as f:
            group = f.get('Subhalo')
            n = len(group['SubhaloLenType']) if group is not None else 0
            if remaining < n:
                offsets = np.asarray(group['SubhaloOffsetType'][remaining], np.uint64)
                lengths = np.asarray(group['SubhaloLenType'][remaining], np.uint64)
                break
            remaining -= n
    if offsets is None:
        raise ValueError('Selected actual SUBFIND row does not exist; disruption cannot be replaced by a manufactured row')
    count = sum(int(lengths[k]) for k in request['species'])
    if int(lengths[2]):
        raise ValueError('Selected satellite contains coarse DM; expand the fine zoom region before SIDM satellite science')
    if count > request['maximum_bound_particles']:
        raise MemoryError('Native satellite bound population exceeds its direct-potential budget')
    bound_parts = []
    for kind in request['species']:
        if int(lengths[kind]):
            row = _bound_particle_slice(snapshots, kind, int(offsets[kind]), int(lengths[kind]), ('ParticleIDs',))
            if row['ParticleIDs'] is None:
                raise ValueError('Native SUBFIND span lacks exact member IDs')
            bound_parts.append(exact_uint64_ids(row['ParticleIDs']))
    bound_ids = exact_uint64_ids(np.concatenate(bound_parts) if bound_parts else np.empty(0, np.uint64))
    if request.get('cohort_policy') == 'initial_bound_only':
        if original['schema'] == 'native-satellite-binding-measurement-v1':
            original_bound = np.asarray(original_arrays['bound'])
            if original_bound.dtype != np.dtype(bool) or original_bound.shape != original_ids.shape:
                raise ValueError('Initial satellite bound cohort flags must align with exact original IDs')
            original_ids = original_ids[original_bound]
        else:
            # A host aperture seeds provenance, not the satellite cohort. The
            # first cohort is the complete actual bound satellite population.
            original_ids = bound_ids
    population, particles = read_native_gravitating_particles(native, request['snapshot'], species=request['species'],
        maximum_particles=request['maximum_particles'], memory_gib=request['memory_gib'], namespace=request['particle_namespace'])
    if any(native_sources.get(path) != sha for path, sha in population['source_hashes'].items()):
        raise ValueError('Satellite particle measurements lie outside the actual completed native generation')
    sources.update(population['source_hashes'])
    current_ids = particles['particle_ids']
    if not np.isin(bound_ids, current_ids).all():
        raise ValueError('Actual satellite bound census is not contained in requested native species')
    select = np.isin(current_ids, np.union1d(original_ids, bound_ids))
    ids = current_ids[select]
    bounded = np.isin(ids, bound_ids)
    x, velocity, mass = particles['position_proper_kpc'][select], particles['velocity_peculiar_kms'][select], particles['mass_msun'][select]
    if not bounded.any():
        raise ValueError('Empty actual satellite binding selection remains unresolved; no potential/center is manufactured')
    with h5py.File(snapshots[0], 'r') as f:
        a = float(f['Header'].attrs['Time']) if int(f['Parameters'].attrs['ComovingIntegrationOn']) else 1.
        h = float(f['Parameters'].attrs['HubbleParam'])
        from .data import dataset_to_cgs_factor
        group = f.get('PartType' + str(request['species'][0]))
        if group is None:
            group = next(f['PartType' + str(kind)] for kind in request['species'] if 'PartType' + str(kind) in f)
        proper_box = float(f['Header'].attrs['BoxSize']) * dataset_to_cgs_factor(group['Coordinates'].attrs, a, h) / KPC_CM
        periodic = int(f['Parameters'].attrs.get('PeriodicBoundariesOn', 0)) != 0
    pivot = x[np.flatnonzero(bounded)[0]].copy()
    delta = x - pivot
    if periodic and proper_box > 0:
        delta -= proper_box * np.rint(delta / proper_box)
    center_offset = np.average(delta[bounded], axis=0, weights=mass[bounded])
    center = pivot + center_offset
    offsets = delta - center_offset
    bulk = np.average(velocity[bounded], axis=0, weights=mass[bounded])
    if len(ids) * int(bounded.sum()) > request['maximum_pair_evaluations']:
        raise MemoryError('Native satellite self-potential exceeds the predeclared direct-sum pair ceiling')
    chunk = request['potential_chunk_rows']
    if chunk * int(bounded.sum()) * 64 + len(ids) * 512 > request['memory_gib'] * 1024 ** 3:
        raise MemoryError('Direct satellite self-potential workspace exceeds explicit allocation')
    potential = np.empty(len(ids))
    bound_x, bound_mass = offsets[bounded], mass[bounded]
    softening = request['potential_softening_proper_kpc']
    for start in range(0, len(ids), chunk):
        stop = min(start + chunk, len(ids))
        distance = np.linalg.norm(offsets[start:stop, None, :] - bound_x[None, :, :], axis=2)
        inverse = 1 / np.sqrt(distance ** 2 + softening ** 2)
        inverse[ids[start:stop, None] == ids[bounded][None, :]] = 0
        potential[start:stop] = -G_KPC_KMS2_MSUN * (inverse @ bound_mass)
    energy = 0.5 * np.sum((velocity - bulk) ** 2, axis=1) + potential
    dm_ids = ids[np.isin(particles['species'][select], (1, 2))]
    diagnostics = _native_diagnostics(native, request['snapshot'], dm_ids, request)
    arrays = {'particle_ids': ids, 'mass_msun': mass, 'bound': bounded, 'specific_energy_km2_s2': energy,
        'specific_self_potential_km2_s2': potential, 'center_proper_kpc': center, 'bulk_velocity_km_s': bulk}
    if diagnostics and len(dm_ids) == len(ids):
        arrays['interaction_count'] = diagnostics['interaction_count']
    if request.get('host_center_proper_kpc') is not None:
        host = _finite(request['host_center_proper_kpc'], (3,), 'measured host center')
        orbit = center - host
        if periodic and proper_box > 0:
            orbit -= proper_box * np.rint(orbit / proper_box)
        arrays['host_offset_proper_kpc'] = orbit
        arrays['host_relative_velocity_km_s'] = bulk - _finite(request['host_bulk_velocity_km_s'], (3,), 'measured host velocity')
    epoch = population['epoch']
    time = epoch.get('cosmic_time_gyr') if epoch.get('cosmic_time_gyr') is not None else epoch.get('elapsed_time_gyr')
    if time is None:
        raise ValueError('Native satellite measurement lacks its actual physical clock')
    report = {'schema': 'native-satellite-binding-measurement-v1', 'execution_status': 'COMPLETE', 'request': request,
        'satellite_identity': request['satellite_identity'], 'particle_namespace': request['particle_namespace'], 'epoch': epoch,
        'elapsed_time_gyr': time, 'potential_gauge': request['potential_gauge'], 'source_hashes': sources,
        'binding_definition': 'actual native SUBFIND ordered member spans',
        'energy_definition': 'instantaneous bound-mass Plummer-softened satellite self-potential plus peculiar COM-relative kinetic energy; self term excluded',
        'energy_exclusions': ['host tidal work', 'cosmological terms', 'time-dependent potential work', 'binding finder smoothing prescription differences']}
    specs = {name: {'units': '(km / s)^2' if 'energy' in name or 'potential' in name else 'proper kpc' if 'kpc' in name
        else 'km / s' if 'km_s' in name else 'Msun' if name == 'mass_msun' else 'dimensionless',
        'quantity': 'actual native satellite ' + name.replace('_', ' ')} for name in arrays}
    specs['particle_ids']['units'] = 'identity'
    specs['particle_ids']['identity_namespace'] = request['particle_namespace']
    return export_numeric_audit(destination, report, arrays, specs, algorithm=Path(__file__).resolve(),
        context={'epoch': epoch, 'coordinates': {'frame': 'instantaneous native satellite proper COM'}}, source_hashes=sources)


def publish_sidm_history(parameters, destination):
    """Public native producer: complete selected membership at exact epochs."""
    from .observation_geometry import read_native_gravitating_particles
    request = validate_sidm_history_request(parameters['request'])
    base = parameters.get('request_base', '.')
    memory = request['memory_gib'] * 1024 ** 3
    sources = sidm_input_bindings(parameters)
    native_sources = _stage_native_bindings(parameters)
    records, packed, first_members = [], {}, None
    scientific_definitions = {name: request[name] for name in ('radial_edges_proper_kpc',
        'minimum_shell_particles', 'minimum_effective_particles', 'minimum_resolved_radius_proper_kpc',
        'minimum_shape_eigenvalue_gap_fraction', 'central_aperture_proper_kpc', 'minimum_central_particles',
        'maximum_coarse_support_fraction', 'maximum_omitted_scattering_upper_bound', 'core_fit', 'component_centroids', 'collapse')}
    for epoch in request['epochs']:
        selection, sa, bindings = _load_product(epoch['selection'], base, memory)
        sources.update(bindings)
        if selection.get('schema') != 'native-halo-membership-measurement-v1':
            raise ValueError('SIDM halo intake requires measured native membership, not manufactured positions')
        if selection.get('snapshot') != epoch['snapshot'] or selection.get('matching') != request['matching']:
            raise ValueError('SIDM membership snapshot or exact matched halo identity differs')
        if selection.get('elapsed_time_gyr') != epoch['elapsed_time_gyr']:
            raise ValueError('Requested elapsed time must equal the measured source clock')
        # Membership measurements must bind original evolved bytes, not only code.
        original = selection.get('native_particle_source_hashes', {})
        if not original or any(native_sources.get(path) != sha for path, sha in original.items()):
            raise ValueError('Membership lacks original native completed evolution source bindings')
        selected_ids = exact_uint64_ids(sa['member_particle_ids'])
        if len(selected_ids) > request['maximum_particles']:
            raise MemoryError('Measured halo membership exceeds requested complete particle budget')
        center = _finite(sa['center_proper_kpc'], (3,), 'measured halo center')
        bulk = _finite(sa['bulk_velocity_km_s'], (3,), 'measured halo bulk velocity')
        population, particles = read_native_gravitating_particles(parameters['native_run_directory'], epoch['snapshot'],
            species=request['species'], maximum_particles=request['maximum_particles'], memory_gib=request['memory_gib'],
            namespace=request['particle_namespace'])
        sources.update(population['source_hashes'])
        if any(native_sources.get(path) != sha for path, sha in population['source_hashes'].items()):
            raise ValueError('SIDM science uses snapshots outside the retained native generation')
        order = np.argsort(particles['particle_ids'])
        locations = np.searchsorted(particles['particle_ids'][order], selected_ids)
        if np.any(locations >= len(order)) or not np.array_equal(particles['particle_ids'][order][locations], selected_ids):
            raise ValueError('Measured halo membership has missing native particle IDs')
        indices = order[locations]
        x = particles['position_proper_kpc'][indices] - center
        box = selection.get('periodic_box_proper_kpc')
        if box is not None:
            box = _positive(box, 'measured proper periodic box')
            x -= box * np.rint(x / box)
        components = {}
        definitions = request['species_components']
        if set(definitions) != set(map(str, request['species'])):
            raise ValueError('Each requested native species needs exactly one component name')
        for name in sorted(set(definitions.values())):
            kinds = [int(k) for k, label in definitions.items() if label == name]
            keep = np.isin(particles['species'][indices], kinds)
            components[name] = {'particle_ids': selected_ids[keep], 'offsets_proper_kpc': x[keep],
                'velocities_peculiar_km_s': particles['velocity_peculiar_kms'][indices][keep],
                'masses_msun': particles['mass_msun'][indices][keep]}
        if 'dm' not in components:
            raise ValueError('SIDM halo science requires a complete declared DM population')
        dm = components['dm']
        if first_members is None:
            first_members = dm['particle_ids'].copy()
        dm['sidm'] = _native_diagnostics(parameters['native_run_directory'], epoch['snapshot'], dm['particle_ids'], request)
        if request['dark_matter_model'] == 'collisionless' and not dm['sidm']:
            # Exact zero belongs to the verified disabled native operator. A
            # missing SIDM output in an enabled run never receives this value.
            count = len(dm['particle_ids'])
            dm['sidm'] = {name: np.zeros(count, np.uint64 if name == 'interaction_count' else float) for name in SIDM_FIELDS}
        shell_request = {**request, 'bulk_velocity_km_s': bulk.tolist()}
        measurement, arrays = measure_sidm_shells(dm, shell_request)
        core, extra = measure_core_definitions(dm, arrays, request)
        arrays.update(extra)
        core_radius = request['central_aperture_proper_kpc']
        boundary = int(np.searchsorted(arrays['radial_edges_proper_kpc'], core_radius, side='right') - 1)
        valid_boundary = 0 <= boundary < len(arrays['radius_proper_kpc']) and core_radius < arrays['radial_edges_proper_kpc'][-1]
        heat = float(arrays['heat_luminosity_erg_s'][boundary]) if valid_boundary and arrays['transport_supported'][boundary] else None
        record = {'snapshot': epoch['snapshot'], 'elapsed_time_gyr': epoch['elapsed_time_gyr'], 'epoch': population['epoch'],
            'halo_id': request['matching']['halo_match_id'], 'center_proper_kpc': center.tolist(), 'core': core,
            'core_boundary_heat_luminosity_erg_s': heat,
            'core_boundary_heat_radius_proper_kpc': float(arrays['radius_proper_kpc'][boundary]) if valid_boundary else None,
            'core_boundary_heat_aperture_bracket_proper_kpc': arrays['radial_edges_proper_kpc'][boundary:boundary + 2].tolist() if valid_boundary else None,
            'components': component_centroids_and_misalignment(components, request['component_centroids']), 'measurement': measurement}
        records.append(record)
        for name, array in arrays.items():
            packed.setdefault(name, []).append(array)
        if sum(v.nbytes for values in packed.values() for v in values) * 6 > memory:
            raise MemoryError('Retained SIDM history and next epoch exceed the explicit workspace')
    arrays = {key: np.stack(value) for key, value in packed.items()}
    arrays['elapsed_time_gyr'] = np.asarray([row['elapsed_time_gyr'] for row in records])
    arrays['snapshots'] = np.asarray([row['snapshot'] for row in records], np.int64)
    arrays['halo_ids'] = np.full(len(records), request['matching']['halo_match_id'], np.uint64)
    arrays['centers_proper_kpc'] = np.asarray([row['center_proper_kpc'] for row in records])
    arrays['central_density_msun_kpc3'] = np.asarray([row['core']['central_aperture']['density_msun_kpc3'] for row in records])
    arrays['central_supported'] = np.asarray([row['core']['central_aperture']['supported'] for row in records], bool)
    arrays['first_saved_dm_member_ids'] = first_members
    report = {'schema': 'sidm-halo-history-v1', 'execution_status': 'COMPLETE', 'science_acceptance': 'NOT_ACCEPTED',
        'matching': request['matching'], 'dark_matter_model': request['dark_matter_model'], 'request': request,
        'physical_definitions': scientific_definitions, 'physical_definition_sha256': canonical_digest(scientific_definitions),
        'epochs': records, 'phase_classification': classify_sidm_history(records, request['collapse']),
        'native_stage_receipt': str(Path(parameters['native_stage_receipt']).resolve()), 'source_hashes': sources,
        'inference_policy': 'forward-model comparisons and independent calibration required; one core radius or axis ratio does not determine a cross section'}
    specs = _history_specifications(arrays, epochs=[row['epoch'] for row in records],
        halo_namespace=request['matching']['halo_match_namespace'])
    specs['first_saved_dm_member_ids'].update(identity_namespace=request['particle_namespace'],
        quantity='exact native DM identity roster at the first saved aperture; not a claim of IC Lagrangian membership')
    from core.research.display import measured_epoch_series_annotation
    specs['first_saved_dm_member_ids']['annotations']['epoch_series'] = measured_epoch_series_annotation([row['epoch'] for row in records], axis=None)
    return export_numeric_audit(destination, report, arrays, specs, algorithm=Path(__file__).resolve(),
        context={'epoch': {'saved_native_epochs': [row['epoch'] for row in records]},
            'coordinates': {'position_units': 'proper kpc', 'velocity_convention': 'peculiar km/s', 'origin': 'exact measured halo center'}}, source_hashes=sources)


def _history_specifications(arrays, *, epochs=None, halo_namespace=None):
    units = {**HISTORY_FIELDS, 'radius_proper_kpc': 'proper kpc', 'radial_edges_proper_kpc': 'proper kpc',
        'halo_ids':'identity','first_saved_dm_member_ids':'identity',
        'slope_radius_proper_kpc': 'proper kpc', 'mean_velocity_km_s': 'km / s', 'elapsed_time_gyr': 'Gyr',
        'central_density_msun_kpc3': 'Msun / proper kpc^3', 'centers_proper_kpc': 'proper kpc'}
    specifications = {name: {'units': units.get(name, 'dimensionless'), 'quantity': 'measured ' + name.replace('_', ' '),
        'annotations': {'meaning': name.replace('_', ' '), 'time_axis': 0,
            'unsupported_policy': 'explicit boolean support; unsupported estimators are never accuracy claims'}} for name in arrays}
    if halo_namespace and 'halo_ids' in specifications:
        specifications['halo_ids']['identity_namespace'] = halo_namespace
    if epochs:
        from core.research.display import measured_epoch_series_annotation
        for specification in specifications.values():
            specification['annotations']['epoch_series'] = measured_epoch_series_annotation(epochs, axis=0)
    return specifications


def validate_sidm_matched_request(request):
    if request.get('schema') != 'sidm-matched-science-request-v1':
        raise ValueError('Versioned matched CDM/SIDM science request required')
    _positive(request['memory_gib'], 'matched science workspace')
    _integer(request['maximum_histories'], 'matched history ceiling', 2)
    histories = request['histories']
    if not isinstance(histories, list) or not 2 <= len(histories) <= request['maximum_histories']:
        raise ValueError('Bounded matched histories required')
    labels = [row['label'] for row in histories]
    if len(set(labels)) != len(labels) or request['baseline_label'] not in labels:
        raise ValueError('Distinct history labels and explicit collisionless baseline required')
    if not request['observables'] or len(set(request['observables'])) != len(request['observables']):
        raise ValueError('Distinct selected SIDM observables required')
    if any(name not in HISTORY_FIELDS and name != 'central_density_msun_kpc3' for name in request['observables']):
        raise ValueError('Unsupported matched SIDM observable')
    for name in request['observables']:
        _positive(request['absolute_floors'][name], 'observable dimensional floor ' + name)
    _integer(request['matching_controls']['minimum_shared_dm_ids'], 'matched native DM overlap count')
    fraction = _positive(request['matching_controls']['minimum_dm_overlap_fraction'], 'matched native DM overlap fraction')
    if fraction > 1:
        raise ValueError('Matched native ID overlap fraction cannot exceed one')
    return request


def publish_sidm_matched(parameters, destination):
    request = validate_sidm_matched_request(parameters['request'])
    base = parameters.get('request_base', '.')
    memory = request['memory_gib'] * 1024 ** 3
    sources = _source_code_bindings()
    products = {}
    retained_array_bytes = 0
    for row in request['histories']:
        report, arrays, binding = _load_product(row['product'], base, memory)
        if report.get('schema') != 'sidm-halo-history-v1' or report.get('execution_status') != 'COMPLETE':
            raise ValueError('Matched family needs actually completed native halo measurements')
        sources.update(binding)
        retained_array_bytes += sum(value.nbytes for value in arrays.values())
        if retained_array_bytes * 6 > memory:
            raise MemoryError('Combined matched history arrays exceed the explicit family allocation')
        products[row['label']] = (report, arrays)
    baseline, ba = products[request['baseline_label']]
    if baseline['dark_matter_model'] != 'collisionless':
        raise ValueError('Matched baseline must be collisionless')
    rows, packed = [], {}
    for row_index, row in enumerate(request['histories']):
        label = row['label']
        if label == request['baseline_label']:
            continue
        report, arrays = products[label]
        for key in ('phase_seed', 'linear_bundle_sha256', 'background_sha256', 'halo_match_namespace', 'halo_match_id'):
            if report['matching'][key] != baseline['matching'][key]:
                raise ValueError('Matched SIDM family changes exact primordial/halo identity: ' + key)
        if report['physical_definition_sha256'] != baseline['physical_definition_sha256']:
            raise ValueError('Matched profiles use different physical estimators/apertures/support rules')
        if report['request']['particle_namespace'] != baseline['request']['particle_namespace']:
            raise ValueError('Matched DM histories change the exact native identity namespace')
        first = exact_uint64_ids(ba['first_saved_dm_member_ids'])
        second = exact_uint64_ids(arrays['first_saved_dm_member_ids'])
        shared = len(np.intersect1d(first, second, assume_unique=True))
        minimum_overlap = min(shared / len(first) if len(first) else 0., shared / len(second) if len(second) else 0.)
        controls = request['matching_controls']
        if shared < controls['minimum_shared_dm_ids'] or minimum_overlap < controls['minimum_dm_overlap_fraction']:
            raise ValueError('Caller halo labels fail the measured exact native DM overlap requirement')
        if not np.array_equal(arrays['elapsed_time_gyr'], ba['elapsed_time_gyr']) or not np.array_equal(arrays['radius_proper_kpc'], ba['radius_proper_kpc']):
            raise ValueError('Matched histories require the same actual epochs and radial bins; no nearest substitution')
        baryons_equal = report['matching']['baryon_control_sha256'] == baseline['matching']['baryon_control_sha256']
        comparison = {'label': label, 'baseline_label': request['baseline_label'], 'matched_baryons': baryons_equal,
            'native_identity_overlap': {'shared_dm_ids': shared, 'minimum_directional_overlap_fraction': minimum_overlap,
                'definition': 'exact first-saved DM aperture census at common native epochs and IC identity namespace',
                'scope': 'selected pair identity support; a global catalogue match is needed for population completeness/split ambiguity'},
            'interpretation': 'matched interaction contrast' if baryons_equal else 'combined baryonic/interaction contrast; separate matched baryonic controls required',
            'observables': {}}
        for name in request['observables']:
            current, previous = np.asarray(arrays[name], float), np.asarray(ba[name], float)
            if current.shape != previous.shape:
                raise ValueError('Matched observable dimensions differ')
            support_key = ('central_supported' if name == 'central_density_msun_kpc3' else 'enclosed_mass_supported' if name == 'enclosed_mass_msun'
                else 'shape_supported' if name.startswith('axis_ratio')
                else 'transport_supported' if name.startswith('heat_') else 'scattering_supported' if 'interaction' in name or 'exposure' in name or 'coarse' in name or 'omitted' in name else 'profile_supported')
            mask = _supported(arrays[support_key], current.shape, support_key) & _supported(ba[support_key], current.shape, support_key)
            mask &= np.isfinite(current) & np.isfinite(previous)
            difference = np.where(mask, current - previous, np.nan)
            scale = np.maximum(np.abs(previous), request['absolute_floors'][name])
            relative = np.where(mask, difference / scale, np.nan)
            if np.isinf(difference).any() or np.isinf(relative).any():
                raise FloatingPointError('Matched physical observable contrast overflow')
            prefix = 'comparison_%04d_' % row_index + name
            packed[prefix + '_difference'] = difference
            packed[prefix + '_relative_difference'] = relative
            packed[prefix + '_supported'] = mask
            comparison['observables'][name] = {'supported_bins': int(mask.sum()), 'total_bins': mask.size,
                'difference_array': prefix + '_difference', 'relative_difference_array': prefix + '_relative_difference',
                'support_array': prefix + '_supported', 'absolute_floor': request['absolute_floors'][name],
                'units': HISTORY_FIELDS.get(name, 'Msun / proper kpc^3'),
                'uncertainty': 'paired measurement only; independent IC/operator ensembles and numerical/baryonic budgets remain separate'}
        rows.append(comparison)
    packed['elapsed_time_gyr'] = ba['elapsed_time_gyr']
    packed['radius_proper_kpc'] = ba['radius_proper_kpc']
    report = {'schema': 'sidm-matched-science-v1', 'execution_status': 'COMPLETE', 'request': request,
        'comparisons': rows, 'matching': baseline['matching'], 'source_hashes': sources,
        'inference_policy': 'no unique cross-section inference from one core or shape; use calibrated joint forward models',
        'scientific_exclusions': [{'label': row['label'], 'reason': 'unmatched baryonic control'} for row in rows if not row['matched_baryons']]}
    specs = {}
    for comparison in rows:
        for value in comparison['observables'].values():
            specs[value['difference_array']] = {'units': value['units']}
            specs[value['relative_difference_array']] = {'units': 'dimensionless'}
            specs[value['support_array']] = {'units': 'dimensionless'}
    specs.update({'elapsed_time_gyr': {'units': 'Gyr'}, 'radius_proper_kpc': {'units': 'proper kpc'}})
    for name, specification in specs.items():
        specification['quantity'] = 'measured matched SIDM ' + name.replace('_', ' ')
    return export_numeric_audit(destination, report, packed, specs, algorithm=Path(__file__).resolve(),
        context={'epoch': baseline['numeric_data']['array_contract'].get('epoch'), 'coordinates': {'frame': 'matched proper halo measurements'}}, source_hashes=sources)


def forward_model_degeneracy(jacobian, covariance, parameter_names, observable_names, request):
    """Whiten a calibrated joint response; report unresolved parameter modes."""
    j = _finite(jacobian, (len(observable_names), len(parameter_names)), 'forward response Jacobian')
    cov = _finite(covariance, (len(observable_names), len(observable_names)), 'joint observable covariance')
    if not parameter_names or not observable_names or len(set(parameter_names)) != len(parameter_names) or len(set(observable_names)) != len(observable_names):
        raise ValueError('Distinct declared parameter and observable definitions required')
    if not request.get('independent_calibration_receipts'):
        raise ValueError('Degeneracy assessment requires retained independent forward calibration references')
    scales = _finite(request['parameter_scales'], (len(parameter_names),), 'physical parameter scales')
    if np.any(scales <= 0) or not np.allclose(cov, cov.T, rtol=1e-12, atol=0):
        raise ValueError('Positive parameter scales and symmetric covariance required')
    chol = np.linalg.cholesky(cov)
    whitened = np.linalg.solve(chol, j * scales[None, :])
    _, singular, right = np.linalg.svd(whitened, full_matrices=True)
    tolerance = _positive(request['minimum_whitened_singular_value'], 'response rank threshold')
    spectrum = np.zeros(len(parameter_names))
    spectrum[:len(singular)] = singular
    resolved = spectrum >= tolerance
    return {'schema': 'sidm-forward-degeneracy-v1', 'parameter_names': parameter_names, 'observable_names': observable_names,
        'singular_values': spectrum.tolist(), 'parameter_modes_rows': right.tolist(), 'resolved_modes': resolved.tolist(),
        'status': 'LOCALLY_IDENTIFIABLE' if resolved.all() else 'DEGENERATE_LOCAL_RESPONSE',
        'parameter_scale_definition': request['parameter_scales'], 'calibration_receipts': request['independent_calibration_receipts'],
        'scope': 'local calibrated linear response with declared covariance; posterior validity still requires independent coverage'}


def sidm_perturber_profile(history_report, arrays, request):
    """Use actual evolved enclosed masses in the existing stream backend."""
    if history_report.get('schema') != 'sidm-halo-history-v1':
        raise ValueError('SIDM stream templates require measured evolved profile history')
    times = _finite(request['worldline']['times_myr'], name='measured perturber times')
    elapsed = np.asarray(arrays['elapsed_time_gyr']) * 1000
    if not np.array_equal(times, elapsed):
        raise ValueError('Stream worldline times differ from actual SIDM profile measurements')
    if times.ndim != 1 or len(times) < 2 or arrays['enclosed_mass_msun'].shape[1] < 2:
        raise ValueError('SIDM stream Worldline/PCHIP requires at least two measured epochs and radii')
    profiles = []
    for i in range(len(times)):
        support = _supported(arrays['enclosed_mass_supported'][i], arrays['enclosed_mass_msun'][i].shape, 'enclosed mass support')
        radial_edges = arrays['radial_edges_proper_kpc'][i][1:]
        mass = arrays['enclosed_mass_msun'][i]
        # Removing unresolved interior shells would fabricate enclosed masses.
        if not support.all() or not np.isfinite(mass).all() or np.any(np.diff(mass) < 0):
            raise ValueError('SIDM stream template needs the full supported declared force aperture')
        profiles.append({'kind': 'enclosed_mass', 'radius_kpc': radial_edges.tolist(),
            'enclosed_mass_msun': mass.tolist(), 'mass_msun': float(mass[-1]),
            'minimum_supported_radius_kpc': float(radial_edges[0]),
            'inner_policy': 'constant_density_core',
            'outer_boundary': 'declared retained finite mass aperture'})
    template = {'category': 'sidm_subhalo', 'profiles': profiles,
        'support_label': 'simulation_derived',
        'provenance': {'measured_native_history': True, 'source_hashes': history_report['source_hashes'],
            'physical_definition_sha256': history_report['physical_definition_sha256']},
        'mass_matching_policy': 'require_actual_measured_mass', 'component_id': request['component_id']}
    from core.stream.encounters import matched_perturber_record
    return matched_perturber_record(request['worldline'], template)


def sidm_native_lens_plane(particles, request):
    """Project actual species masses through the existing lensing contract."""
    from .observation_geometry import species_lens_plane
    report, arrays = species_lens_plane(particles, request)
    report['sidm_profile_policy'] = 'actual native species-resolved evolved mass; no core-to-cross-section inversion'
    report['sidm_baryonic_control'] = request['sidm_baryonic_control']
    return report, arrays


def validate_sidm_observations_request(request):
    if request.get('schema') != 'sidm-observational-comparison-request-v1':
        raise ValueError('Versioned SIDM stream/lens/H0 observational comparison required')
    _positive(request['memory_gib'], 'observational comparison workspace')
    maximum = _integer(request['maximum_comparisons'], 'observational comparison ceiling')
    rows = request['comparisons']
    if not isinstance(rows, list) or not rows or len(rows) > maximum:
        raise ValueError('Bounded actual matched observational products required')
    if len({row['comparison_id'] for row in rows}) != len(rows):
        raise ValueError('Distinct observational comparison identifiers required')
    for row in rows:
        if row['family'] not in {'streams', 'lensing', 'h0_estimator'} or not row['observables']:
            raise ValueError('Supported actual observational family and observable maps required')
        if not row['matching']:
            raise ValueError('Observational matched geometry/selection/nuisance controls required')
        for observable in row['observables']:
            _positive(observable['absolute_floor'], 'observational dimensional floor')
            if not observable.get('units'):
                raise ValueError('Explicit physical observational units required')
    return request


def publish_sidm_observations(parameters, destination):
    request = validate_sidm_observations_request(parameters['request'])
    memory = request['memory_gib'] * 1024 ** 3
    sources, arrays, records, specs = _source_code_bindings(), {}, [], {}
    for i, row in enumerate(request['comparisons']):
        cdm, ca, cs = _load_product(row['collisionless'], parameters.get('request_base', '.'), memory)
        sidm, sa, ss = _load_product(row['sidm'], parameters.get('request_base', '.'), memory)
        sources.update(cs); sources.update(ss)
        # The producer must declare matched controls. A consumer cannot relabel
        # observer positions, estimator/nuisance models or stream histories.
        if cdm.get('matched_experiment') != row['matching'] or sidm.get('matched_experiment') != row['matching']:
            raise ValueError('Matched observational controls differ from producer declarations')
        comparisons = []
        for j, observable in enumerate(row['observables']):
            name = observable['array']
            support_name = observable['support_array']
            left, right = np.asarray(ca[name], float), np.asarray(sa[name], float)
            if left.shape != right.shape:
                raise ValueError('Matched stream/lens/H0 estimator bins differ')
            support = _supported(ca[support_name], left.shape, 'observational baseline support') & _supported(sa[support_name], left.shape, 'observational SIDM support')
            for report in (cdm, sidm):
                contract = report['numeric_data']['array_contract']['datasets'][name]
                if contract['units'] != observable['units']:
                    raise ValueError('Observational units differ from declared exact source arrays')
            if np.any(~np.isfinite(left[support])) or np.any(~np.isfinite(right[support])):
                raise ValueError('Supported observational measurements must be finite')
            prefix = 'comparison_%04d_observable_%04d' % (i, j)
            arrays[prefix + '_difference'] = np.where(support, right - left, np.nan)
            arrays[prefix + '_relative_difference'] = np.where(support, (right - left) / np.maximum(np.abs(left), observable['absolute_floor']), np.nan)
            arrays[prefix + '_supported'] = support
            specs[prefix + '_difference'] = {'units': observable['units']}
            specs[prefix + '_relative_difference'] = {'units': 'dimensionless'}
            specs[prefix + '_supported'] = {'units': 'dimensionless'}
            comparisons.append({'observable': name, 'array_prefix': prefix, 'supported_bins': int(support.sum()),
                'total_bins': support.size, 'units': observable['units'], 'absolute_floor': observable['absolute_floor']})
        records.append({'comparison_id': row['comparison_id'], 'family': row['family'], 'observables': comparisons,
            'matching': row['matching'], 'interpretation': 'measured matched forward contrast; native model, baryonic and estimator biases require separate controls'})
    report = {'schema': 'sidm-observational-comparison-v1', 'execution_status': 'COMPLETE', 'request': request,
        'comparisons': records, 'source_hashes': sources, 'h0_scope': 'estimator response at fixed background; scattering does not itself replace the cosmological expansion history'}
    for name, specification in specs.items():
        specification['quantity'] = 'measured matched observational ' + name.replace('_', ' ')
    return export_numeric_audit(destination, report, arrays, specs, algorithm=Path(__file__).resolve(),
        context={'epoch': {'meaning': 'producer-declared matched experiment epochs'}, 'coordinates': {'frame': 'producer-declared stream/lens/observer frame'}}, source_hashes=sources)


def validate_sidm_derived_request(request):
    if request.get('schema') != 'sidm-derived-science-request-v1' or request.get('product') not in {'satellite', 'degeneracy', 'stream_template', 'lens_plane'}:
        raise ValueError('Explicit supported SIDM derived science request required')
    _positive(request['memory_gib'], 'derived SIDM workspace')
    if not isinstance(request['inputs'], list) or not request['inputs'] or len(request['inputs']) > 64:
        raise ValueError('Bounded retained scientific inputs required')
    if not isinstance(request.get('settings'), dict) or not request['settings']:
        raise ValueError('Explicit derived SIDM physical settings required')
    return request


def publish_sidm_derived(parameters, destination):
    """Publish satellite, calibrated degeneracy, stream or lens forward products."""
    request = validate_sidm_derived_request(parameters['request'])
    memory = request['memory_gib'] * 1024 ** 3
    sources, products = _source_code_bindings(), []
    retained_array_bytes = 0
    for reference in request['inputs']:
        report, arrays, binding = _load_product(reference, parameters.get('request_base', '.'), memory)
        retained_array_bytes += sum(value.nbytes for value in arrays.values())
        if retained_array_bytes * 6 > memory:
            raise MemoryError('Combined derived SIDM arrays exceed the explicit allocation')
        products.append((report, arrays)); sources.update(binding)
    kind, settings = request['product'], request['settings']
    arrays, specs = {}, {}
    if kind == 'satellite':
        if len(products) != 2:
            raise ValueError('Satellite transport requires initial and current native bound measurements')
        initial, current = products
        for report, values in products:
            if report.get('schema') != 'native-satellite-binding-measurement-v1':
                raise ValueError('Satellite diagnostics require measured binding/energy/native-ID ancestry')
            if report.get('potential_gauge') != settings['potential_gauge']:
                raise ValueError('Satellite potential gauges differ from the actual bound producer')
        if initial[0]['satellite_identity'] != current[0]['satellite_identity'] or initial[0]['particle_namespace'] != current[0]['particle_namespace']:
            raise ValueError('Satellite transport changes exact ownership/native identity namespace')
        if current[0]['elapsed_time_gyr'] <= initial[0]['elapsed_time_gyr']:
            raise ValueError('Satellite state pair must have increasing actual physical time')
        result = satellite_survival_and_transport(initial[1], current[1], settings)
        arrays['retained_original_bound_mass_fraction'] = np.asarray([result['retained_original_bound_mass_fraction']])
        arrays['current_bound_mass_msun'] = np.asarray([result['current_bound_mass_msun']])
        arrays['resolved_survival_supported'] = np.asarray([result['status'] == 'RESOLVED_SURVIVOR'])
        specs = {'retained_original_bound_mass_fraction': {'units': 'dimensionless'},
            'current_bound_mass_msun': {'units': 'Msun'}, 'resolved_survival_supported': {'units': 'dimensionless'}}
    elif kind == 'degeneracy':
        if len(products) != 1 or products[0][0].get('schema') != 'calibrated-joint-forward-response-v1':
            raise ValueError('SIDM degeneracy requires an actual independently calibrated joint response product')
        calibration, values = products[0]
        if calibration['independent_calibration_receipts'] != settings['independent_calibration_receipts']:
            raise ValueError('Consumer changed the producer independent calibration scope')
        for reference in calibration['independent_calibration_receipts']:
            measured,mask,binding=_read_joint_response_calibration(reference,parameters.get('request_base','.'),parameters['request']['memory_gib']*1024**3)
            if measured['response_fingerprint']!=calibration['response_fingerprint']:
                raise ValueError('Degeneracy consumer changes the independently calibrated native response')
            if np.any(np.asarray(values['observable_supported'],bool)&~mask):
                raise ValueError('Degeneracy consumer includes independently unsupported response bins')
            sources.update(binding)
        domain=calibration.get('response_domain_indices', list(range(len(calibration['observable_names']))))
        if (not isinstance(domain,list) or not domain or len(set(domain))!=len(domain)
                or any(type(i)is not int or not 0<=i<len(calibration['observable_names']) for i in domain)):
            raise ValueError('Calibrated joint response has no stable supported observable domain')
        if 'observable_supported' in values:
            support=_supported(values['observable_supported'],(len(calibration['observable_names']),),'calibrated response support')
            if np.flatnonzero(support).tolist()!=domain:
                raise ValueError('Joint response domain differs from actual derivative-support arrays')
        if settings['parameter_scales']!=calibration.get('parameter_scales',settings['parameter_scales']):
            raise ValueError('Consumer changes the calibrated physical parameter coordinates')
        result = forward_model_degeneracy(values['jacobian'][domain], values['covariance'][np.ix_(domain,domain)],
            calibration['parameter_names'], [calibration['observable_names'][i] for i in domain], settings)
        result['observable_domain_indices']=domain
        arrays = {'whitened_singular_values': np.asarray(result['singular_values']),
            'parameter_modes_rows': np.asarray(result['parameter_modes_rows']), 'resolved_modes': np.asarray(result['resolved_modes'])}
        specs = {name: {'units': 'dimensionless in declared scaled parameter coordinates'} for name in arrays}
    elif kind == 'stream_template':
        if len(products) != 1:
            raise ValueError('SIDM stream template requires one complete measured profile history')
        history, values = products[0]
        result = sidm_perturber_profile(history, values, settings)
        arrays = {'elapsed_time_gyr': values['elapsed_time_gyr'], 'radius_proper_kpc': values['radial_edges_proper_kpc'][:, 1:],
            'enclosed_mass_msun': values['enclosed_mass_msun'], 'profile_supported': values['profile_supported']}
        specs = {'elapsed_time_gyr': {'units': 'Gyr'}, 'radius_proper_kpc': {'units': 'proper kpc'},
            'enclosed_mass_msun': {'units': 'Msun'}, 'profile_supported': {'units': 'dimensionless'}}
    else:
        if len(products) != 1 or products[0][0].get('schema') not in {'native-lensing-particle-measurement-v1', 'native-halo-membership-measurement-v1'}:
            raise ValueError('SIDM lens forward product requires actual evolved species-resolved lens particles')
        particle_report, values = products[0]
        if particle_report['schema'] == 'native-halo-membership-measurement-v1':
            if not np.array_equal(_finite(settings['center_proper_kpc'], (3,)), values['center_proper_kpc']):
                raise ValueError('Lens plane center differs from measured native halo center')
            required_radius = math.sqrt(2 * settings['half_width_proper_kpc'] ** 2 + settings['half_depth_proper_kpc'] ** 2)
            if required_radius > particle_report['request']['support_radius_proper_kpc']:
                raise ValueError('Measured native aperture does not cover the requested complete lens cylinder')
        particles = {**values, 'source_hashes': particle_report['source_hashes']}
        result, arrays = sidm_native_lens_plane(particles, settings)
        for name in arrays:
            unit = 'Msun / proper kpc^2' if 'surface_density' in name else 'rad' if 'rad' in name else 'dimensionless'
            specs[name] = {'units': unit}
    report = {'schema': 'sidm-derived-science-v1', 'execution_status': 'COMPLETE', 'request': request,
        'product': kind, 'result': result, 'source_hashes': sources, 'matched_experiment': request.get('matched_experiment'),
        'inference_policy': 'forward contrasts retain baryonic and numerical degeneracies; no unique microscopic inference from one structural observable'}
    for name, specification in specs.items():
        specification['quantity'] = 'measured derived SIDM ' + name.replace('_', ' ')
    return export_numeric_audit(destination, report, arrays, specs, algorithm=Path(__file__).resolve(),
        context={'epoch': products[-1][0].get('epoch', {'saved_history': True}), 'coordinates': {'frame': 'source-declared native proper component/observer frame'}}, source_hashes=sources)


def validate_forward_vector_request(request):
    if request.get('schema') != 'native-forward-observable-vector-request-v1':
        raise ValueError('Actual native forward observable-vector request required')
    _positive(request['memory_gib'], 'native response workspace')
    maximum = _integer(request['maximum_observables'], 'observable vector ceiling')
    rows = request['observables']
    if not isinstance(rows, list) or not rows or len(rows)>maximum or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Distinct bounded actual source observable selectors required')
    if not isinstance(request['parameter_bindings'], dict) or not request['parameter_bindings']:
        raise ValueError('Response parameters must bind actual native configuration/model fields')
    if set(request['parameters']) != set(request['parameter_bindings']):
        raise ValueError('Every response parameter needs its actual native binding')
    if not isinstance(request.get('realization_id'), str) or not request['realization_id']:
        raise ValueError('Recorded independent native realization identity required')
    return request


def _forward_epoch_signature(value):
    """Compare physical epochs, not the paths of their independently owned clocks."""
    if isinstance(value,dict):
        bookkeeping={'cosmic_time_source','source_hashes','native_stage_receipt','snapshot_number_role','requested_scale_factor',
            'requested_cosmic_time_gyr','cosmic_time_numerical_error_gyr'}
        return {key:_forward_epoch_signature(item) for key,item in value.items() if key not in bookkeeping}
    if isinstance(value,list):return [_forward_epoch_signature(item) for item in value]
    return value


def _forward_fixed_physics_signature(config, bindings, owned):
    """Hold native physics fixed except the explicitly varied coordinates."""
    fixed=copy.deepcopy({key:config.get(key,{}) for key in ('cosmology','ede','wdm','sidm','chemistry','black_holes','stellar_evolution','hydro','simulation')})
    fixed['simulation'].pop('random_seed',None);fixed['sidm'].pop('operator_seed',None)
    fixed['sidm'].pop('event_identity_namespace',None)
    for name in ('monofonic_transfer_validation_source','monofonic_background_validation_source','native_generation_report',
            'output_format','restart_every_seconds','num_files_per_snapshot','time_limit_cpu_seconds'):
        fixed['simulation'].pop(name,None)
    assets={}
    def replace_path(value,path,name):
        for key in path[:-1]:value=value[key]
        if path[-1] not in value:raise ValueError('Response coordinate is absent from actual native physics')
        value[path[-1]]={'varied_response_coordinate':name}
    for name,binding in bindings.items():
        if 'native_config_path' in binding:
            path=binding['native_config_path']
            if path[0] not in fixed or path in [['simulation','random_seed'],['sidm','operator_seed']]:
                raise ValueError('Response coordinates must describe physical native fields, not realization/bookkeeping settings')
            replace_path(fixed,path,name)
        else:
            path=Path(binding['native_asset']).expanduser().resolve()
            if owned.get(str(path))!=file_sha256(path):raise ValueError('Response model asset is outside original evolved inputs')
            assets.setdefault(str(path),json.loads(path.read_text()))
            replace_path(assets[str(path)],binding['model_path'],name)
    linked=set()
    def normalize(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256'} and str(Path(value['path']).expanduser().resolve()) in assets:
                path=str(Path(value['path']).expanduser().resolve());linked.add(path)
                return {'fixed_response_asset_model':assets[path]}
            return {key:normalize(item) for key,item in value.items()}
        if isinstance(value,list):return [normalize(item) for item in value]
        return value
    fixed=normalize(fixed)
    if linked!=set(assets):raise ValueError('A response asset must be linked to an actual evolved native physical configuration')
    return canonical_digest(fixed)


def publish_forward_observable_vector(parameters, destination):
    """Select actual scalar native observables without relabeling model parameters."""
    import yaml
    request=validate_forward_vector_request(parameters['request'])
    original=_stage_native_bindings(parameters)
    sources={**original,**_source_code_bindings()}
    root=Path(parameters['native_run_directory']).resolve()
    config=yaml.safe_load((root/'config.yaml').read_text())
    def at(value, path):
        if not isinstance(path,list) or not path or any(not isinstance(key,str) for key in path):
            raise ValueError('Exact nested native parameter binding required')
        for key in path:
            if not isinstance(value,dict) or key not in value:
                raise ValueError('Response parameter is absent from actual native model: '+'.'.join(path))
            value=value[key]
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
            raise ValueError('Differentiable response parameters must be finite native numeric values')
        return value
    actual={}
    for name,binding in request['parameter_bindings'].items():
        if set(binding)=={'native_config_path'}:
            value=at(config,binding['native_config_path'])
        elif set(binding)=={'native_asset','model_path'}:
            asset_path=Path(binding['native_asset']).expanduser().resolve()
            if original.get(str(asset_path)) != file_sha256(asset_path):
                raise ValueError('Response parameter model is outside actual evolved native asset bindings')
            value=at(json.loads(asset_path.read_text()),binding['model_path'])
        else:
            raise ValueError('Unsupported native response parameter binding')
        if value != request['parameters'][name]:
            raise ValueError('Requested response parameter differs from actual evolved model: '+name)
        actual[name]=value
    native_seed=config['simulation']['random_seed']
    operator_seed=config.get('sidm',{}).get('operator_seed')
    values=np.full(len(request['observables']),np.nan);supported=np.zeros(len(values),bool)
    definitions=[];epochs=[]
    memory=request['memory_gib']*1024**3
    for i,row in enumerate(request['observables']):
        report,arrays,binding=_load_product(row['product'],parameters['request_base'],memory)
        sources.update(binding)
        upstream=report.get('native_stage_receipt')
        if upstream is None or str(Path(upstream).resolve()) != str(Path(parameters['native_stage_receipt']).resolve()):
            raise ValueError('Response vector measurement does not bind the requested actual native evolution')
        array=np.asarray(arrays[row['array']])
        indices=row['indices']
        if not isinstance(indices,list) or len(indices)!=array.ndim or any(type(k)is not int or not 0<=k<n for k,n in zip(indices,array.shape)):
            raise ValueError('Each response observable selects one exact supported scalar bin')
        contract=report['numeric_data']['array_contract']['datasets'][row['array']]
        if contract['units'] != row['units']:
            raise ValueError('Native forward vector units differ from original producer units')
        mask=_supported(arrays[row['support_array']],array.shape,'forward observable support')
        supported[i]=bool(mask[tuple(indices)])
        value=float(array[tuple(indices)])
        if supported[i] and not math.isfinite(value):
            raise ValueError('Supported native forward observable is nonfinite')
        if supported[i]:values[i]=value
        definitions.append({'name':row['name'],'units':row['units'],'array':row['array'],'indices':indices,
            'quantity_definition':contract.get('quantity_definition'),
            'physical_definition':report.get('physical_definitions',report.get('observable_definition_hash'))})
        epochs.append(report.get('epoch',report.get('epochs')))
    report={'schema':'native-forward-observable-vector-v1','execution_status':'COMPLETE',
        'parameters':actual,'parameter_bindings':request['parameter_bindings'],'observable_definitions':definitions,
        'observable_definition_hash':canonical_digest(definitions),'realization_id':request['realization_id'],
        'phase_seed':native_seed,'sidm_operator_seed':operator_seed,'measurement_epochs':epochs,
        'measurement_epoch_signature':_forward_epoch_signature(epochs),
        'fixed_physics_fingerprint':_forward_fixed_physics_signature(config,request['parameter_bindings'],original),
        'native_stage_receipt':str(Path(parameters['native_stage_receipt']).resolve()),'source_hashes':sources,
        'science_acceptance':'NOT_ESTABLISHED','realization_scope':'actual native IC/operator realization, never a fabricated analysis vector'}
    return export_numeric_audit(destination,report,{'observable_value':values,'observable_supported':supported},
        {'observable_value':{'units':'per-observable units declared in observable_definitions','quantity':'actual selected native scalar observable vector'},
         'observable_supported':{'units':'dimensionless','quantity':'exact producer-supported selected scalar bins'}},
        algorithm=Path(__file__).resolve(),context={'epoch':{'measured_source_epochs':epochs},'coordinates':{'frame':'source-declared observable frames'}},source_hashes=sources)


def validate_joint_response_request(request):
    if request.get('schema') not in {'sidm-joint-response-request-v1','sidm-joint-response-training-request-v2'}:
        raise ValueError('Versioned calibrated joint native response request required')
    _positive(request['memory_gib'],'joint response workspace')
    names=request['parameter_names']
    if not isinstance(names,list) or not names or len(set(names))!=len(names):
        raise ValueError('Distinct native response parameter names required')
    baseline=_finite(request['baseline_parameters'],(len(names),),'baseline parameters')
    scales=_finite(request['parameter_scales'],baseline.shape,'parameter scales')
    if np.any(scales<=0):raise ValueError('Positive physical response parameter scales required')
    if set(request['difference_steps'])!=set(names):raise ValueError('Every response axis requires frozen difference steps')
    for name in names:
        steps=_finite(request['difference_steps'][name],name=name+' difference steps')
        if steps.ndim!=1 or len(steps)<2 or np.any(steps<=0) or np.any(np.diff(steps)>=0):
            raise ValueError('At least two strictly decreasing positive derivative steps required')
    training=request['schema']=='sidm-joint-response-training-request-v2'
    if training and request.get('independent_calibration_receipts'):
        raise ValueError('Training estimates precede independent measured calibration')
    if not training and not request.get('independent_calibration_receipts'):
        raise ValueError('Independent calibration receipts required before local identifiability publication')
    if not isinstance(request['samples'],list) or not request['samples'] or len(request['samples'])>_integer(request['maximum_samples'],'native response sample ceiling'):
        raise ValueError('Bounded actual measured native response samples required')
    _integer(request['minimum_realizations'],'independent native response realization floor',2)
    return request


def publish_joint_response(parameters, destination):
    """Estimate common-seed finite differences of actual native ensemble means."""
    import hashlib
    request=validate_joint_response_request(parameters['request'])
    memory=request['memory_gib']*1024**3;base=parameters['request_base']
    names=request['parameter_names'];baseline=np.asarray(request['baseline_parameters'],float)
    point_count=1+2*sum(len(request['difference_steps'][name]) for name in names)
    if point_count*len(names)*128>memory or point_count*request['minimum_realizations']>request['maximum_samples']:
        raise MemoryError('Complete native derivative design exceeds its frozen sample/workspace allocation')
    sources=_source_code_bindings();groups={};seen=set();seed_keys=set();definition=None;epochs=None;physics=None
    for reference in request['samples']:
        report,arrays,binding=_load_product(reference,base,memory)
        if report.get('schema')!='native-forward-observable-vector-v1' or report.get('execution_status')!='COMPLETE':
            raise ValueError('Joint response requires actual completed native observable-vector products')
        identity=(report['phase_seed'],report['sidm_operator_seed'],report['realization_id'])
        if set(report['parameters'])!=set(names):
            raise ValueError('Joint response contains unrecorded or missing physical parameter directions')
        point=tuple(float(report['parameters'][name]) for name in names)
        seed_identity=(point,report['phase_seed'],report['sidm_operator_seed'])
        if seed_identity in seed_keys:
            raise ValueError('Repeated IC/operator seeds cannot be relabeled as independent realizations')
        seed_keys.add(seed_identity)
        key=(point,identity)
        if key in seen:raise ValueError('Repeated native realization would bias response/covariance')
        seen.add(key)
        if definition is None:definition=report['observable_definition_hash'];definitions=report['observable_definitions'];epochs=report['measurement_epochs'];epoch_signature=_forward_epoch_signature(epochs)
        if report['observable_definition_hash']!=definition or _forward_epoch_signature(report['measurement_epochs'])!=epoch_signature:
            raise ValueError('Joint response changes physical observables, apertures or actual saved epochs')
        current=report.get('fixed_physics_fingerprint')
        if current is None or physics is not None and current!=physics:raise ValueError('Joint response changes undeclared native physical coordinates')
        physics=current
        value=np.asarray(arrays['observable_value'],float)
        support=_supported(arrays['observable_supported'],value.shape,'joint response support')
        if value.ndim!=1 or len(value)!=len(definitions):raise ValueError('Native joint observable dimensions differ')
        groups.setdefault(point,{})[identity]=(value,support)
        sources.update(binding)
        if len(seen)*(len(value)+len(names))*256+len(value)**2*64+len(value)*len(names)*128>memory:raise MemoryError('Joint response ensemble/covariance exceeds workspace')
    needed=[tuple(baseline)]
    for j,name in enumerate(names):
        for step in request['difference_steps'][name]:
            for sign in (-1,1):
                point=baseline.copy();point[j]+=sign*step;needed.append(tuple(point))
    if any(point not in groups for point in needed):raise ValueError('Actual response family omits a baseline or frozen derivative point')
    if set(groups)!=set(needed):raise ValueError('Joint response contains undeclared native parameter points')
    identities=set.intersection(*(set(groups[point]) for point in needed))
    if len(identities)<request['minimum_realizations']:raise ValueError('Insufficient common independent native response realizations')
    identities=sorted(identities,key=repr)
    if len({i[0] for i in identities})!=len(identities) or len({i[1] for i in identities if i[1] is not None}) not in {0,len(identities)}:
        raise ValueError('Independent response realizations require distinct actual IC seeds and active SIDM operator seeds')
    # Model-point pairing uses the same seed axes. Correlated derived bins and
    # cross-observable terms are retained, not replaced by diagonal errors.
    supports=np.asarray([groups[p][i][1] for p in needed for i in identities])
    common=np.all(supports,axis=0)
    indices=np.flatnonzero(common)
    if not len(indices):raise ValueError('Joint response has no common supported physical domain')
    x=np.asarray([groups[tuple(baseline)][i][0][indices] for i in identities])
    if not np.isfinite(x).all():raise ValueError('Supported native response baseline is nonfinite')
    covariance=np.atleast_2d(np.cov(x,rowvar=False,ddof=1))
    floor=_finite(request['observable_variance_floors'],(len(definitions),),'dimensional covariance floors')[indices]
    if np.any(floor<=0):raise ValueError('Positive predeclared dimensional observable variance floors required')
    # Explicit floors are disclosed regularization, not measured ensemble noise.
    covariance=covariance+np.diag(floor)
    jacobian=np.full((len(definitions),len(names)),np.nan);errors=np.full_like(jacobian,np.nan);stable=np.zeros_like(jacobian,bool)
    step_reports=[]
    for j,name in enumerate(names):
        derivatives=[]
        for step in request['difference_steps'][name]:
            plus=baseline.copy();minus=baseline.copy();plus[j]+=step;minus[j]-=step
            contrasts=np.asarray([(groups[tuple(plus)][i][0][indices]-groups[tuple(minus)][i][0][indices])/(2*step) for i in identities])
            if not np.isfinite(contrasts).all():raise ValueError('Supported paired native derivative is nonfinite')
            mean=contrasts.mean(axis=0);se=contrasts.std(axis=0,ddof=1)/math.sqrt(len(identities))
            derivatives.append((mean,se))
            step_reports.append({'parameter':name,'step':step,'mean_derivative':mean.tolist(),'paired_standard_error':se.tolist()})
        mean,se=derivatives[-1];delta=np.abs(mean-derivatives[-2][0])
        tolerance=_finite(request['derivative_absolute_tolerances'][name],(len(definitions),),'physical derivative tolerance')[indices]
        relative=_positive(request['derivative_relative_tolerance'],'relative derivative tolerance',zero=True)
        if np.any(tolerance<0):raise ValueError('Derivative absolute tolerances must be nonnegative')
        jacobian[indices,j]=mean;errors[indices,j]=se
        stable[indices,j]=delta<=tolerance+relative*np.abs(mean)
    accepted=common & np.all(stable,axis=1)
    response_fingerprint=canonical_digest({'parameter_names':names,'baseline':baseline.tolist(),'scales':request['parameter_scales'],
        'definitions':definitions,'epochs':epoch_signature,'fixed_physics':physics,'realizations':[list(i) for i in identities],
        'jacobian_sha256':hashlib.sha256(jacobian.astype('<f8').tobytes()).hexdigest(),
        'covariance_sha256':hashlib.sha256(covariance.astype('<f8').tobytes()).hexdigest(),'common_indices':indices.tolist()})
    training=request['schema']=='sidm-joint-response-training-request-v2'
    if not training:
        for reference in request['independent_calibration_receipts']:
            calibration,mask,binding=_read_joint_response_calibration(reference,base,memory)
            if calibration['response_fingerprint']!=response_fingerprint:raise ValueError('Measured calibration belongs to a different actual response estimate')
            accepted&=mask;sources.update(binding)
    response_indices=np.flatnonzero(accepted)
    report={'schema':'native-joint-forward-response-estimate-v1' if training else 'calibrated-joint-forward-response-v1','execution_status':'COMPLETE',
        'response_fingerprint':response_fingerprint,'fixed_physics_fingerprint':physics,'observable_definition_hash':definition,'measurement_epochs':epochs,
        'measurement_epoch_signature':epoch_signature,
        'parameter_names':names,'observable_names':[r['name'] for r in definitions], 'observable_definitions':definitions,
        'baseline_parameters':baseline.tolist(),'parameter_scales':request['parameter_scales'],
        'derivative_steps':step_reports,'covariance_domain_indices':indices.tolist(),'response_domain_indices':response_indices.tolist(),
        'covariance_scope':'measured independent native realization covariance plus explicit dimensional variance floors',
        'observable_variance_floors':request['observable_variance_floors'],
        'native_realizations':[list(identity) for identity in identities],
        'independent_calibration_receipts':request.get('independent_calibration_receipts',[]),'source_hashes':sources,
        'science_acceptance':'NOT_ESTABLISHED','local_response_scope':'uncalibrated step-stable ensemble-mean derivative' if training else 'step-stable derivative with independently measured native local prediction error; no global posterior/coverage claim'}
    dense=np.full((len(definitions),len(definitions)),np.nan);dense[np.ix_(indices,indices)]=covariance
    arrays={'jacobian':jacobian,'jacobian_standard_error':errors,'covariance':dense,'observable_supported':accepted,
        'derivative_step_supported':stable,'baseline_observable_mean':np.where(common,np.mean(np.asarray([groups[tuple(baseline)][i][0] for i in identities]),axis=0),np.nan)}
    units={name:{'units':'per-observable and per-parameter physical units declared in report','quantity':'measured '+name.replace('_',' ')} for name in arrays}
    return export_numeric_audit(destination,report,arrays,units,algorithm=Path(__file__).resolve(),
        context={'epoch':{'measured_source_epochs':epochs},'coordinates':{'frame':'declared joint native observables'}},source_hashes=sources)


def validate_joint_calibration_request(request):
    if request.get('schema')!='native-joint-response-calibration-request-v1':raise ValueError('Measured held-out native joint-response calibration request required')
    _positive(request['memory_gib'],'held-out calibration workspace')
    pairs=request['pairs']
    if not isinstance(pairs,list) or not pairs or len(pairs)>_integer(request['maximum_pairs'],'held-out pair ceiling'):
        raise ValueError('Bounded actual baseline/varied native vector pairs required')
    _integer(request['minimum_realizations_per_point'],'held-out independent realization floor',2)
    _positive(request['mean_error_multiplier'],'declared paired-mean sampling allowance',zero=True)
    _positive(request['maximum_whitened_mean_residual'],'joint response residual tolerance')
    ratio=_positive(request['minimum_design_singular_ratio'],'held-out design rank threshold')
    if ratio>=1:raise ValueError('Held-out design singular ratio must be below one')
    return request


def measure_joint_response_calibration(parameters):
    """Measure local prediction error on disjoint actual IC/operator ensembles.

    Calibration uses paired native observable differences and never invents
    response vectors. It measures the supplied parameter points; no global
    posterior coverage or unsampled parameter-domain accuracy is inferred.
    """
    request=validate_joint_calibration_request(parameters['request']);base=parameters['request_base'];memory=request['memory_gib']*1024**3
    training,estimate,sources=_load_product(request['training_response'],base,memory/3)
    if training.get('schema')!='native-joint-forward-response-estimate-v1':raise ValueError('Held-out calibration must follow an uncalibrated native training estimate')
    names=training['parameter_names'];baseline=np.asarray(training['baseline_parameters'],float);n=len(training['observable_names'])
    scales=np.asarray(training['parameter_scales'],float);box=_finite(request['parameter_box_half_widths'],baseline.shape,'held-out physical parameter bounds')
    if np.any(box<=0):raise ValueError('Held-out parameter bounds must be positive')
    absolute=_finite(request['maximum_absolute_mean_residual'],(n,),'dimensional held-out mean residual limits')
    relative=_finite(request['maximum_relative_mean_residual'],(n,),'relative held-out mean residual limits')
    if np.any(absolute<0) or np.any(relative<0):raise ValueError('Held-out residual limits must be nonnegative')
    domain=np.asarray(training['response_domain_indices'],int);support=_supported(estimate['observable_supported'],(n,),'training response support').copy()
    if not np.array_equal(domain,np.flatnonzero(support)) or not len(domain):raise ValueError('Held-out response requires a nonempty exact step-stable training domain')
    training_phases={i[0] for i in training['native_realizations']};training_operators={i[1] for i in training['native_realizations'] if i[1] is not None}
    grouped={};identities={};heldout=[]
    for pair in request['pairs']:
        records=[]
        for key in ('baseline','varied'):
            report,arrays,binding=_load_product(pair[key],base,memory/3);sources.update(binding)
            if report.get('schema')!='native-forward-observable-vector-v1' or report.get('execution_status')!='COMPLETE':raise ValueError('Held-out calibration requires completed original native forward vectors')
            if report['observable_definition_hash']!=training['observable_definition_hash'] or _forward_epoch_signature(report['measurement_epochs'])!=training['measurement_epoch_signature'] or report.get('fixed_physics_fingerprint')!=training['fixed_physics_fingerprint']:
                raise ValueError('Held-out native vectors change observables, epochs or undeclared physics')
            if set(report['parameters'])!=set(names):raise ValueError('Held-out native parameter directions differ from training')
            point=np.asarray([report['parameters'][name] for name in names],float)
            seed=(report['phase_seed'],report['sidm_operator_seed'])
            if seed[0] in training_phases or seed[1] is not None and seed[1] in training_operators:
                raise ValueError('Held-out calibration reuses a training IC or active operator seed')
            value=np.asarray(arrays['observable_value'],float);mask=_supported(arrays['observable_supported'],(n,),'held-out native bin support')
            if value.shape!=(n,) or not np.isfinite(value[mask]).all():raise ValueError('Held-out native observable vector is invalid')
            records.append((point,seed,value,mask))
        first,last=records
        if first[1]!=last[1] or not np.array_equal(first[0],baseline):raise ValueError('Held-out contrasts require the same actual native seeds and baseline parameters')
        delta=last[0]-baseline
        if not np.isfinite(delta).all() or not np.any(delta) or np.any(np.abs(delta)>box):raise ValueError('Held-out perturbation lies outside its frozen physical bounds')
        key=tuple(delta);seen=identities.setdefault(key,set())
        if first[1] in seen:raise ValueError('Repeated held-out native pair cannot count as another realization')
        seen.add(first[1]);heldout.append(first[1]);grouped.setdefault(key,[]).append((last[2]-first[2],first[3]&last[3]))
        if len(heldout)*(n+len(names))*256+n*n*64>memory:raise MemoryError('Held-out response ensembles exceed workspace')
    points=sorted(grouped)
    if len(points)*len(names)*256+min(len(points),len(names))**2*256>memory/2:raise MemoryError('Held-out scaled design and SVD exceed workspace')
    design=np.asarray(points)/scales
    singular=np.linalg.svd(design,compute_uv=False)
    if len(singular)<len(names) or singular[-1]/singular[0]<request['minimum_design_singular_ratio'] or not np.all(np.any(design>0,axis=0)&np.any(design<0,axis=0)):
        raise ValueError('Held-out design must resolve both signs of every parameter direction')
    if len(names)>1 and not np.any(np.count_nonzero(design,axis=1)>1):raise ValueError('Joint response calibration needs actual joint parameter perturbations')
    jacobian=np.asarray(estimate['jacobian'])[domain];covariance=np.asarray(estimate['covariance'])[np.ix_(domain,domain)]
    if not np.isfinite(jacobian).all() or not np.isfinite(covariance).all():raise ValueError('Supported training response/covariance must be finite')
    factor=np.linalg.cholesky(covariance);means=np.full((len(points),n),np.nan);errors=np.full_like(means,np.nan)
    supported=np.zeros_like(means,bool);joint=np.full(len(points),np.nan);counts=[]
    for j,point in enumerate(points):
        rows=grouped[point];seeds=identities[point];counts.append(len(rows))
        if len({i[0] for i in seeds})!=len(seeds) or len({i[1] for i in seeds if i[1] is not None}) not in {0,len(seeds)}:
            raise ValueError('Held-out covariance requires distinct actual IC and active SIDM operator seeds')
        common=np.all([row[1] for row in rows],axis=0)&support
        if len(rows)<request['minimum_realizations_per_point']:continue
        contrast=np.asarray([row[0] for row in rows]);prediction=np.asarray(estimate['jacobian'])@np.asarray(point)
        residual=contrast[:,domain]-prediction[domain];mean=residual.mean(axis=0);se=residual.std(axis=0,ddof=1)/math.sqrt(len(rows))
        means[j,domain]=mean;errors[j,domain]=se
        allowed=absolute[domain]+relative[domain]*np.abs(prediction[domain])+request['mean_error_multiplier']*se
        supported[j,domain]=common[domain]&(np.abs(mean)<=allowed)
        if bool(common[domain].all()):
            joint[j]=float(np.linalg.norm(np.linalg.solve(factor,mean)))
            if joint[j]>request['maximum_whitened_mean_residual']:supported[j,:]=False
        else:supported[j,:]=False
    accepted=support&np.all(supported,axis=0)
    retained=copy.deepcopy(request)
    for ref in [retained['training_response'],*[r[key] for r in retained['pairs'] for key in ('baseline','varied')]]:ref['path']=str(_path(base,ref['path']))
    report={'schema':'measured-native-joint-response-calibration-v1','execution_status':'COMPLETE','request':retained,
        'response_fingerprint':training['response_fingerprint'],'observable_definition_hash':training['observable_definition_hash'],
        'measurement_epochs':training['measurement_epochs'],'fixed_physics_fingerprint':training['fixed_physics_fingerprint'],
        'parameter_names':names,'baseline_parameters':baseline.tolist(),'parameter_scales':scales.tolist(),
        'training_response':retained['training_response'],'heldout_native_seeds':[list(i) for i in sorted(set(heldout),key=repr)],
        'measured_parameter_offsets':[list(p) for p in points],'realizations_per_point':counts,
        'status':'SUPPORTED_LOCAL_RESPONSE' if bool(accepted.any()) else 'UNRESOLVED','source_hashes':sources,
        'scope':'local ensemble-mean prediction on the explicitly measured held-out native parameter points; no global coverage claim','science_acceptance':'NOT_ESTABLISHED'}
    arrays={'calibration_supported':accepted,'point_observable_supported':supported,'paired_mean_prediction_residual':means,
        'paired_residual_mean_standard_error':errors,'whitened_mean_residual_norm':joint,'parameter_offsets':np.asarray(points),
        'design_singular_values':singular}
    return report,arrays,sources


def publish_joint_response_calibration(parameters,destination):
    report,arrays,sources=measure_joint_response_calibration(parameters)
    return export_numeric_audit(destination,report,arrays,{name:{'units':'per-observable physical units' if 'residual' in name and 'norm' not in name else 'declared physical parameter units' if name=='parameter_offsets' else 'dimensionless',
        'quantity':'measured held-out native '+name.replace('_',' ')} for name in arrays},algorithm=Path(__file__).resolve(),
        context={'epoch':{'measured_source_epochs':report['measurement_epochs']},'coordinates':{'frame':'actual native joint observable coordinates'}},source_hashes=sources)


def _read_joint_response_calibration(reference,base,memory):
    report,arrays,sources=_load_product(reference,base,memory)
    if report.get('schema')!='measured-native-joint-response-calibration-v1' or report.get('execution_status')!='COMPLETE':
        raise ValueError('A completed workflow receipt is not measured native response calibration')
    if report['request']['memory_gib']*1024**3>memory:raise MemoryError('Measured calibration recomputation exceeds the current allocated workspace')
    actual,values,binding=measure_joint_response_calibration({'request':report['request'],'request_base':'.'})
    if (actual['response_fingerprint']!=report['response_fingerprint'] or actual['training_response']!=report['training_response']
            or actual['heldout_native_seeds']!=report['heldout_native_seeds'] or not np.array_equal(arrays['calibration_supported'],values['calibration_supported'])):
        raise ValueError('Calibration decision differs from actual retained held-out native measurements')
    sources.update(binding)
    return report,np.asarray(values['calibration_supported'],bool),sources


def validate_calibrated_response_request(request):
    if request.get('schema')!='sidm-calibrated-joint-response-request-v1':raise ValueError('Typed training-response plus independent-calibration request required')
    _positive(request['memory_gib'],'calibrated response workspace')
    if not isinstance(request['calibrations'],list) or not 1<=len(request['calibrations'])<=_integer(request['maximum_calibrations'],'calibration ceiling'):
        raise ValueError('Bounded measured independent calibration products required')
    return request


def publish_calibrated_joint_response(parameters,destination):
    request=validate_calibrated_response_request(parameters['request']);memory=request['memory_gib']*1024**3;base=parameters['request_base']
    report,arrays,sources=_load_product(request['training_response'],base,memory/2)
    if report.get('schema')!='native-joint-forward-response-estimate-v1':raise ValueError('Calibrated response requires an actual native training estimate')
    supported=np.asarray(arrays['observable_supported'],bool).copy();references=[]
    for reference in request['calibrations']:
        measured,mask,binding=_read_joint_response_calibration(reference,base,memory)
        if measured['response_fingerprint']!=report['response_fingerprint'] or measured['training_response']['sha256']!=request['training_response']['sha256']:
            raise ValueError('Independent calibration is bound to a different native response estimate')
        if mask.shape!=supported.shape:raise ValueError('Calibration changes the response observable domain')
        supported&=mask;sources.update(binding);references.append({'path':str(_path(base,reference['path'])),'sha256':reference['sha256']})
    arrays['observable_supported']=supported
    report={**report,'schema':'calibrated-joint-forward-response-v1','response_domain_indices':np.flatnonzero(supported).tolist(),
        'independent_calibration_receipts':references,'source_hashes':sources,'calibration_status':'SUPPORTED_LOCAL_RESPONSE' if supported.any() else 'UNRESOLVED',
        'local_response_scope':'step-stable derivatives with independently measured held-out native local prediction error; no global posterior/coverage claim'}
    return export_numeric_audit(destination,report,arrays,{name:{'units':'per-observable and per-parameter physical units declared in report',
        'quantity':'measured calibrated '+name.replace('_',' ')} for name in arrays},algorithm=Path(__file__).resolve(),
        context={'epoch':{'measured_source_epochs':report['measurement_epochs']},'coordinates':{'frame':'declared joint native observables'}},source_hashes=sources)


def compile_joint_response_campaign(request,destination,*,request_base):
    """Materialize the complete native response/calibration DAG without running it.

    Each actual phase realization supplies retained native ICs with its MUSIC
    generation report or its validated public IC-only capsule. Pairing reuses
    that complete state within a realization;
    training and held-out realizations have disjoint IC/operator seed axes.
    Scientific measurements are registered producer templates, not fabricated
    vectors or caller-supplied accuracy flags.
    """
    from .workflow import safe_id,path_at,default_registry,compile_requested_product_dag
    from core.research.capabilities import validate_native_answers
    if request.get('schema')!='native-joint-response-campaign-request-v1':raise ValueError('Explicit native joint-response campaign request required')
    resources=request['resources'];memory=_positive(resources['memory_gib'],'campaign workspace')*1024**3
    maximum=_integer(resources['maximum_stage_count'],'campaign stage ceiling');wall=_positive(request['wall_seconds_per_native_run'],'native per-run wall allocation')
    destination=Path(destination).resolve();base=Path(request_base).resolve();sources=_source_code_bindings()
    def asset(reference):
        path=path_at(base,reference['path'])
        if not path.is_file() or path.is_symlink() or file_sha256(path)!=reference['sha256']:raise ValueError('Native campaign retained asset changed: '+str(path))
        sources[str(path)]=reference['sha256'];return path
    answers_path=asset(request['base_answers'])
    if answers_path.stat().st_size*8>memory:raise MemoryError('Base native campaign answers exceed metadata allocation')
    answers=json.loads(answers_path.read_text());base_config=validate_native_answers(answers,resources)
    definitions=request['parameter_bindings'];names=request['training']['parameter_names'];baseline=_finite(request['training']['baseline_parameters'],(len(names),),'native campaign baseline coordinates')
    if not names or len(set(names))!=len(names):raise ValueError('Distinct nonempty native campaign coordinates required')
    if set(definitions)!=set(names):raise ValueError('Every campaign response coordinate needs explicit native and answer/model bindings')
    training=copy.deepcopy(request['training']);training['schema']='sidm-joint-response-training-request-v2';training['samples']=[]
    training.pop('independent_calibration_receipts',None);training['memory_gib']=resources['memory_gib']
    calibration=copy.deepcopy(request['calibration']);calibration['memory_gib']=resources['memory_gib'];calibration['pairs']=[]
    calibration['schema']='native-joint-response-calibration-request-v1'
    if len(names)*len(names)*128>memory:raise MemoryError('Native campaign parameter grid exceeds workspace')
    derivative_points=[tuple(baseline)]
    for j,name in enumerate(names):
        steps=_finite(training['difference_steps'][name],name=name+' campaign derivative steps')
        if steps.ndim!=1 or len(steps)<2 or np.any(steps<=0) or np.any(np.diff(steps)>=0):raise ValueError('Frozen decreasing positive derivative steps required')
        for step in steps:
            for sign in (-1,1):
                point=baseline.copy();point[j]+=sign*step;derivative_points.append(tuple(point))
    if len(set(derivative_points))!=len(derivative_points) or not np.isfinite(derivative_points).all():
        raise ValueError('Derivative offsets must produce distinct finite native parameter points')
    offsets=_finite(request['heldout_parameter_offsets'],name='actual held-out parameter offsets')
    if offsets.ndim!=2 or offsets.shape[1]!=len(names) or len(offsets)<2 or np.any(np.all(offsets==0,axis=1)) or len({tuple(row) for row in offsets})!=len(offsets):
        raise ValueError('Distinct explicit nonzero held-out parameter points required')
    box=_finite(calibration['parameter_box_half_widths'],baseline.shape,'held-out campaign bounds')
    if np.any(box<=0) or np.any(np.abs(offsets)>box):raise ValueError('Native held-out design exceeds its frozen positive parameter bounds')
    scales=_finite(training['parameter_scales'],baseline.shape,'native campaign coordinate scales')
    if np.any(scales<=0):raise ValueError('Campaign coordinate scales must be positive')
    singular=np.linalg.svd(offsets/scales,compute_uv=False)
    if (len(singular)<len(names) or singular[-1]/singular[0]<calibration['minimum_design_singular_ratio']
            or not np.all(np.any(offsets>0,axis=0)&np.any(offsets<0,axis=0)) or len(names)>1 and not np.any(np.count_nonzero(offsets,axis=1)>1)):
        raise ValueError('The complete held-out campaign must resolve signed joint parameter directions before evolution')
    populations={};phase_seen=set();operator_seen=set();case_ids=set()
    for role,minimum in (('training',training['minimum_realizations']),('heldout',calibration['minimum_realizations_per_point'])):
        minimum=_integer(minimum,role+' realization floor',2);rows=request['realizations'][role]
        if not isinstance(rows,list) or len(rows)<minimum:raise ValueError('Native campaign omits the required independent '+role+' realizations')
        populations[role]=[]
        for row in rows:
            identity=safe_id(row['realization_id']);phase=row['phase_seed'];operator=row.get('sidm_operator_seed')
            if identity in case_ids or type(phase)is not int or not 0<=phase<=2**64-1 or phase in phase_seen:raise ValueError('Native response realizations require distinct exact IC seed identities')
            case_ids.add(identity);phase_seen.add(phase)
            if answers.get('sidm',{}).get('enabled',False):
                if type(operator)is not int or not 0<=operator<=2**64-1 or operator in operator_seen:raise ValueError('Native SIDM response realizations require distinct exact operator seeds')
                operator_seen.add(operator)
            elif operator is not None:raise ValueError('Collisionless campaign cannot label inactive operator seeds as stochastic realizations')
            if set(row['initial_conditions'])=={'validated_public_ic'}:
                from core.response_campaign import load_public_ic
                capsule=asset(row['initial_conditions']['validated_public_ic']);published,pins=load_public_ic(capsule,memory_gib=resources['memory_gib']);sources.update(pins)
                if published['phase_seed']!=phase:raise ValueError('Response realization relabels actual public IC generator seed')
                initial={key:copy.deepcopy(published[key]) for key in ('primary','files')}
                initial['validated_public_ic']=copy.deepcopy(row['initial_conditions']['validated_public_ic'])
                row={**row,'initial_conditions':initial,'phase_asset':copy.deepcopy(published['phase_asset'])}
                generation=None
            else:
                generation=asset(row['initial_conditions']['native_generation_report'])
            primary=asset(row['initial_conditions']['primary']);files=[asset(ref) for ref in row['initial_conditions']['files']]
            if primary not in files or len(set(files))!=len(files):raise ValueError('Complete native IC shards must include their primary shard exactly once')
            phase_asset=asset(row['phase_asset'])
            populations[role].append({**row,'primary':str(primary),'generation':None if generation is None else str(generation),'phase_asset_path':str(phase_asset)})
    templates=request['measurement_stages'];registry=default_registry()
    if not isinstance(templates,list) or not templates:raise ValueError('Native response requires concrete registered scientific measurement producers')
    template_ids=[safe_id(row['stage_id']) for row in templates]
    if len(set(template_ids))!=len(template_ids):raise ValueError('Repeated native response measurement template IDs')
    allowed={'sidm_membership','sidm_satellite','sidm_history','sidm_derived','physical_target_support','physical_target_measurement','native_response_target'}
    if any(row['kind'] not in allowed for row in templates):raise ValueError('Response measurement templates must use registered native scientific producers')
    jobs=len(populations['training'])*len(derivative_points)+len(populations['heldout'])*(len(offsets)+1)
    if len(populations['training'])*len(derivative_points)>training['maximum_samples'] or len(populations['heldout'])*len(offsets)>calibration['maximum_pairs']:
        raise ValueError('Full campaign exceeds its frozen training/calibration sample limits')
    degeneracy=request.get('degeneracy_settings')
    if degeneracy is not None:
        if (set(degeneracy)!={'parameter_scales','minimum_whitened_singular_value'}
                or degeneracy['parameter_scales']!=training['parameter_scales']):raise ValueError('Degeneracy must retain the calibrated physical parameter scales')
        _positive(degeneracy['minimum_whitened_singular_value'],'local response mode threshold')
    if jobs*(len(templates)+2)+3+(2 if degeneracy is not None else 0)>maximum or jobs*len(names)*512>memory:raise MemoryError('Complete native response campaign exceeds its frozen stage/workspace bounds')
    if destination.exists():raise FileExistsError('Keep prior native response campaign design')
    destination.mkdir(parents=True);stages=[];native_rows=[];model_cache={};baseline_native={}
    def at(value,path):
        if not isinstance(path,list) or not path or any(not isinstance(key,str) for key in path):raise ValueError('Exact nested native response field paths required')
        for key in path:value=value[key]
        return value
    def put(value,path,item):
        at(value,path)
        for key in path[:-1]:value=value[key]
        value[path[-1]]=copy.deepcopy(item)
    def product(sid,filename):return {'path':{'stage':sid,'artifact':filename},'sha256':{'stage':sid,'artifact':filename,'load':'sha256'}}
    def materialize(point,index):
        point_key=tuple(point)
        if point_key in model_cache:return copy.deepcopy(model_cache[point_key])
        current=copy.deepcopy(answers);bindings={};model_assets={}
        for j,name in enumerate(names):
            definition=definitions[name];answer_path=definition['answer_path']
            if definition['kind']=='answer_field':
                if answer_path[0] in {'name','seed','cosmology','schedule','particles_per_dimension','mpi_ranks','omp_threads','memory_limit_gb'}:
                    raise ValueError('Native response campaign varies nonlinear physical coordinates, not IC/resource/clock bookkeeping')
                put(current,answer_path,float(point[j]));bindings[name]=copy.deepcopy(definition['native_binding'])
            elif definition['kind']=='model_asset_field':
                original=at(current,answer_path)
                path=asset(original);key=str(path)
                model_assets.setdefault(key,{'model':json.loads(path.read_text()),'answer_paths':[],'coordinates':[]})
                entry=model_assets[key];put(entry['model'],definition['model_path'],float(point[j]));entry['answer_paths'].append(answer_path);entry['coordinates'].append((name,definition['model_path']))
            else:raise ValueError('Response campaign supports exact native answer fields or linked model-asset fields')
        for k,entry in enumerate(model_assets.values()):
            path=destination/f'point-{index:04d}-model-{k:03d}.json';atomic_json(path,entry['model']);sha=file_sha256(path);sources[str(path)]=sha
            for answer_path in entry['answer_paths']:put(current,answer_path,{'path':str(path),'sha256':sha})
            for name,model_path in entry['coordinates']:bindings[name]={'native_asset':str(path),'model_path':model_path}
        model_cache[point_key]=(current,bindings)
        return copy.deepcopy(model_cache[point_key])
    def emit(role,realization,point,index):
        prefix=safe_id(f'{role}_{realization["realization_id"]}_p{index:04d}');native_id=prefix+'_native'
        current,bindings=materialize(point,index);current['name']=prefix;current['seed']=realization['phase_seed']
        if current.get('sidm',{}).get('enabled',False):current['sidm']['operator_seed']=realization['sidm_operator_seed']
        cfg=validate_native_answers(current,resources)
        for j,name in enumerate(names):
            binding=bindings[name]
            actual=at(cfg,binding['native_config_path']) if 'native_config_path' in binding else at(json.loads(Path(binding['native_asset']).read_text()),binding['model_path'])
            if actual!=point[j]:raise ValueError('Generated native answer mapping does not apply the actual requested response coordinate: '+name)
        answer_path=destination/(prefix+'.answers.json');atomic_json(answer_path,current);sources[str(answer_path)]=file_sha256(answer_path)
        case_inputs={str(answers_path):sources[str(answers_path)],str(answer_path):sources[str(answer_path)]}
        ic_source=realization['initial_conditions'].get('native_generation_report',realization['initial_conditions'].get('validated_public_ic'))
        for reference in [realization['initial_conditions']['primary'],*realization['initial_conditions']['files'],ic_source,realization['phase_asset']]:
            path=path_at(base,reference['path']);case_inputs[str(path)]=reference['sha256']
        for binding in bindings.values():
            if 'native_asset' in binding:case_inputs[binding['native_asset']]=sources[binding['native_asset']]
        native_arguments=['--answers',{'input':str(answer_path)},'--existing-ic',{'input':realization['primary']}]
        if realization['generation'] is not None:native_arguments+=['--native-generation-report',{'input':realization['generation']}]
        else:
            from core.response_campaign import apply_public_ic
            capsule=path_at(base,realization['initial_conditions']['validated_public_ic']['path']);apply_public_ic(cfg,capsule,realization['primary'])
            native_arguments+=['--validated-public-ic',{'input':str(capsule)}]
        native_arguments+=['--evolution-only','--run-dir','{generation}/run']
        stages.append({'stage_id':native_id,'kind':'public_command','parameters':{'command':'gadget','arguments':native_arguments,
            'wall_seconds':wall},'depends_on':[],'input_files':[{'path':path,'sha256':sha} for path,sha in case_inputs.items()],
            'outputs':{native_id:'run'},'product_contracts':{native_id:{'reader':'native_snapshot_series','snapshots':request['snapshots']}}})
        scope={sid:prefix+'_'+sid for sid in template_ids}
        realization_key=(role,realization['realization_id'])
        if index==0:baseline_native[realization_key]=native_id
        baseline_id=baseline_native[realization_key]
        context={'native_run_directory':{'stage':native_id,'artifact':'run'},'native_stage_receipt':{'stage':native_id,'artifact':'stage-receipt.json'},
            'baseline_run_directory':{'stage':baseline_id,'artifact':'run'},'baseline_stage_receipt':{'stage':baseline_id,'artifact':'stage-receipt.json'},
            'point_parameters':dict(zip(names,map(float,point))),'native_parameter_bindings':bindings,'realization_id':realization['realization_id'],
            'phase_seed':realization['phase_seed'],'sidm_operator_seed':realization.get('sidm_operator_seed')}
        if request.get('matching_template'):
            matching=copy.deepcopy(request['matching_template']);matching['phase_seed']=realization['phase_seed']
            matching['linear_bundle_sha256']=file_sha256(cfg['spectrum']['file'])
            matching['baryon_control_sha256']=canonical_digest({'physics':cfg['simulation']['physics'],
                **{name:cfg.get(name,{}) for name in ('hydro','black_holes','stellar_evolution','chemistry','external_potential','baryonic_force','stellar_sources')}})
            context['native_matching']=matching
        def expand(value):
            if isinstance(value,dict):
                if set(value)=={'campaign_value'}:
                    key=value['campaign_value']
                    if key not in context:raise ValueError('Unknown native campaign template substitution: '+key)
                    return copy.deepcopy(context[key])
                if 'stage' in value and set(value)<={'stage','artifact','load','field'}:
                    if value['stage'] not in scope:raise ValueError('Measurement template references an undeclared local producer')
                    return {**value,'stage':scope[value['stage']]}
                return {key:expand(item) for key,item in value.items()}
            if isinstance(value,list):return [expand(item) for item in value]
            return value
        for template in templates:
            row=expand(copy.deepcopy(template));row['stage_id']=scope[template['stage_id']]
            row['depends_on']=[native_id,*[scope[sid] for sid in template.get('depends_on',[])]]
            row['outputs']={prefix+'_'+key:value for key,value in template['outputs'].items()}
            if row.get('product_contracts'):row['product_contracts']={prefix+'_'+key:value for key,value in row['product_contracts'].items()}
            request_directory=Path(row['parameters'].get('request_base',str(base)))
            row['parameters']['request_base']=str((base/request_directory if not request_directory.is_absolute() else request_directory).resolve())
            stages.append(row)
        vector=expand(copy.deepcopy(request['forward_vector']));vector.update(parameters=context['point_parameters'],parameter_bindings=bindings,
            realization_id=realization['realization_id'],memory_gib=resources['memory_gib'])
        vector_id=prefix+'_vector';stages.append({'stage_id':vector_id,'kind':'native_forward_vector',
            'parameters':{'request':vector,'request_base':str(base),**{key:context[key] for key in ('native_run_directory','native_stage_receipt')},'destination':'{generation}/vector.json'},
            'depends_on':[native_id,*scope.values()],'outputs':{vector_id:'vector.json'}})
        native_rows.append({'stage_id':native_id,'role':role,'point':list(point),'phase_seed':realization['phase_seed'],
            'sidm_operator_seed':realization.get('sidm_operator_seed'),'realization_id':realization['realization_id'],'vector_stage':vector_id})
        return product(vector_id,'vector.json')
    for realization in populations['training']:
        for index,point in enumerate(derivative_points):training['samples'].append(emit('training',realization,point,index))
    start=len(derivative_points)
    for realization in populations['heldout']:
        origin=emit('heldout',realization,tuple(baseline),0)
        for index,offset in enumerate(offsets):calibration['pairs'].append({'baseline':origin,'varied':emit('heldout',realization,tuple(baseline+offset),start+index)})
    stages.append({'stage_id':'response_training','kind':'sidm_joint_response','parameters':{'request':training,'request_base':str(base),'destination':'{generation}/response.json'},
        'depends_on':[row['vector_stage'] for row in native_rows if row['role']=='training'],'outputs':{'response_training':'response.json'}})
    calibration['training_response']=product('response_training','response.json')
    stages.append({'stage_id':'response_calibration','kind':'sidm_joint_calibration','parameters':{'request':calibration,'request_base':str(base),'destination':'{generation}/calibration.json'},
        'depends_on':['response_training',*[row['vector_stage'] for row in native_rows if row['role']=='heldout']],'outputs':{'response_calibration':'calibration.json'}})
    stages.append({'stage_id':'calibrated_response','kind':'sidm_calibrated_response','parameters':{'request':{'schema':'sidm-calibrated-joint-response-request-v1',
        'memory_gib':resources['memory_gib'],'maximum_calibrations':1,'training_response':product('response_training','response.json'),
        'calibrations':[product('response_calibration','calibration.json')]},'request_base':str(base),'destination':'{generation}/calibrated-response.json'},
        'depends_on':['response_training','response_calibration'],'outputs':{'calibrated_joint_response':'calibrated-response.json'}})
    requested=['calibrated_joint_response']
    if degeneracy is not None:
        stages.append({'stage_id':'baryonic_degeneracy','kind':'sidm_derived','parameters':{'request':{
            'schema':'sidm-derived-science-request-v1','memory_gib':resources['memory_gib'],'product':'degeneracy',
            'inputs':[product('calibrated_response','calibrated-response.json')],
            'settings':{**copy.deepcopy(degeneracy),'independent_calibration_receipts':[product('response_calibration','calibration.json')]}},
            'request_base':str(base),'destination':'{generation}/degeneracy.json'},'outputs':{'baryonic_degeneracy':'degeneracy.json'}})
        stages.append({'stage_id':'export_baryonic_degeneracy','kind':'native_sidm_export','parameters':{'operation':'extended',
            'product':{'stage':'baryonic_degeneracy','artifact':'degeneracy.json'},
            'run_directory':{'stage':native_rows[0]['stage_id'],'artifact':'run'},'memory_gib':resources['memory_gib'],
            'native_runs':[{'run_directory':{'stage':row['stage_id'],'artifact':'run'},'stage_receipt':{'stage':row['stage_id'],'artifact':'stage-receipt.json'}} for row in native_rows],
            'expected_variants':['sidm_baryonic_degeneracy'],'request':{'schema':'native-sidm-extended-figures-v1','operation':'degeneracy',
                'producer_base_directory':str(base),'native_runs':[],'phase_baseline_run_directory':None}},
            'outputs':{'sidm_baryonic_degeneracy':'figures'}})
        requested.append('sidm_baryonic_degeneracy')
    workflow={'schema':'research-workflow-v1','resources':resources,'followup_limits':request['followup_limits'],
        'physics_spec':{'cosmology_model':'lcdm' if base_config['cosmology']['model']=='wdm' else base_config['cosmology']['model'],
            'dark_matter':{'linear_model':'wdm' if base_config['cosmology']['model']=='wdm' or base_config.get('wdm',{}).get('enabled',False) else 'cdm',
                'interactions':{'kind':'elastic_sidm' if base_config.get('sidm',{}).get('enabled',False) else 'collisionless'}},
            **({'combination_specification':base_config['combination_specification'],'combination_model_sha256':canonical_digest({name:base_config.get(name,{}) for name in ('cosmology','ede','wdm','sidm')})} if 'combination_specification' in base_config else {})},
        'stages':stages,'requested_products':requested}
    compile_requested_product_dag(workflow,registry,request_base=base)
    atomic_json(destination/'workflow.json',workflow)
    report={'schema':'native-joint-response-campaign-design-v1','workflow':str(destination/'workflow.json'),'workflow_sha256':file_sha256(destination/'workflow.json'),
        'native_runs':native_rows,'native_run_count':jobs,'source_hashes':sources,'execution_status':'NOT_EXECUTED','science_acceptance':'NOT_ESTABLISHED',
        'scope':'complete native nonlinear response and disjoint held-out prediction calibration DAG; delivered ICs retain the common linear cosmology/initial state'}
    atomic_json(destination/'campaign-design.json',report);return report


def joint_response_campaign_main():
    parser=argparse.ArgumentParser(description='Materialize a complete native joint-response and held-out calibration workflow')
    parser.add_argument('request',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true',help='Execute the generated production workflow; design generation alone launches nothing')
    args=parser.parse_args();path=args.request.resolve();request=json.loads(path.read_text())
    result=compile_joint_response_campaign(request,args.output,request_base=path.parent)
    if args.execute:
        from .workflow import execute_workflow
        execute_workflow(json.loads(Path(result['workflow']).read_text()),args.output.resolve()/'execution',request_base=path.parent)
    return 0


def joint_calibration_runner(parameters,context):
    return _runner(publish_joint_response_calibration,parameters,context)


def calibrated_response_runner(parameters,context):
    return _runner(publish_calibrated_joint_response,parameters,context)


def forward_vector_runner(parameters,context):
    return _runner(publish_forward_observable_vector,parameters,context)


def joint_response_runner(parameters,context):
    return _runner(publish_joint_response,parameters,context)


def _runner(producer, parameters, context):
    if parameters['request']['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('SIDM science exceeds allocated workflow workspace')
    destination = Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):
        raise ValueError('SIDM science product escapes owned generation')
    report = producer(parameters, destination)
    atomic_json(destination.with_name(destination.name + '.ref.json'), {'path': str(destination), 'sha256': file_sha256(destination)})
    return report


def sidm_history_runner(parameters, context):
    return _runner(publish_sidm_history, parameters, context)


def sidm_membership_runner(parameters, context):
    return _runner(publish_sidm_membership, parameters, context)


def sidm_satellite_runner(parameters, context):
    return _runner(publish_sidm_satellite, parameters, context)


def sidm_matched_runner(parameters, context):
    return _runner(publish_sidm_matched, parameters, context)


def sidm_observations_runner(parameters, context):
    return _runner(publish_sidm_observations, parameters, context)


def sidm_derived_runner(parameters, context):
    return _runner(publish_sidm_derived, parameters, context)


def register_sidm_stages(registry):
    from .workflow import StageDefinition
    paths = tuple(Path(path) for path in _source_code_bindings())
    for kind, runner, validator, extras in (
        ('sidm_membership', sidm_membership_runner, validate_sidm_membership_request, ('native_stage_receipt', 'native_run_directory')),
        ('sidm_satellite', sidm_satellite_runner, validate_sidm_satellite_request, ('native_stage_receipt', 'native_run_directory')),
        ('sidm_history', sidm_history_runner, validate_sidm_history_request, ('native_stage_receipt', 'native_run_directory')),
        ('sidm_matched', sidm_matched_runner, validate_sidm_matched_request, ()),
        ('sidm_observations', sidm_observations_runner, validate_sidm_observations_request, ()),
        ('sidm_derived', sidm_derived_runner, validate_sidm_derived_request, ()),
        ('native_forward_vector', forward_vector_runner, validate_forward_vector_request, ('native_stage_receipt','native_run_directory')),
        ('sidm_joint_response', joint_response_runner, validate_joint_response_request, ()),
        ('sidm_joint_calibration',joint_calibration_runner,validate_joint_calibration_request,()),
        ('sidm_calibrated_response',calibrated_response_runner,validate_calibrated_response_request,())):
        def validate(parameters, v=validator):
            return v(parameters['request'])
        registry.register(StageDefinition(kind, runner, paths,
            required_parameters=('request', 'request_base', 'destination', *extras), validator=validate))


def sidm_science_main():
    parser = argparse.ArgumentParser(description='Measure SIDM native halo histories and matched stream/lens/H0 responses.')
    parser.add_argument('mode', choices=('membership', 'satellite', 'history', 'matched', 'observations', 'derived', 'forward_vector', 'joint_response','joint_calibration','calibrated_response'))
    parser.add_argument('request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-stage-receipt', type=Path)
    parser.add_argument('--native-run-directory', type=Path)
    args = parser.parse_args()
    path = args.request.expanduser().resolve()
    parameters = {'request': json.loads(path.read_text()), 'request_base': str(path.parent)}
    if args.mode in {'membership', 'satellite', 'history', 'forward_vector'}:
        if args.native_stage_receipt is None or args.native_run_directory is None:
            parser.error('history requires both native-stage-receipt and native-run-directory')
        parameters.update(native_stage_receipt=str(args.native_stage_receipt.resolve()),
            native_run_directory=str(args.native_run_directory.resolve()))
    producer = {'membership': publish_sidm_membership, 'history': publish_sidm_history, 'matched': publish_sidm_matched,
        'satellite': publish_sidm_satellite, 'observations': publish_sidm_observations, 'derived': publish_sidm_derived,
        'forward_vector': publish_forward_observable_vector, 'joint_response': publish_joint_response,
        'joint_calibration':publish_joint_response_calibration,'calibrated_response':publish_calibrated_joint_response}[args.mode]
    destination = args.output.expanduser().resolve()
    producer(parameters, destination)
    atomic_json(destination.with_name(destination.name + '.ref.json'), {'path': str(destination), 'sha256': file_sha256(destination)})
    return 0
# @research2-end-module sidm_analysis


# @research2-module sidm_control_measurements
"""Measured isolated-halo conservation and collapse controls from native outputs.

This producer reads saved full-system GADGET particle/potential fields. It does
not evolve a stand-in halo, extrapolate an unsaved collapse time, or turn a
configured SIDM switch into evidence that scattering occurred. Total energy
uses the actual native gravitational potential with its declared zero gauge.
Hydro, external forcing and cosmological expansion require different balances
and are excluded from this isolated closed-system control.
"""

from pathlib import Path
import hashlib
import json
import math
import numpy as np
from .foundation import atomic_json, atomic_text, canonical_digest, file_sha256, KPC_CM, MSUN_G
from .native import original_native_snapshot_files

ENERGY_UNITS = 'Msun (km / s)^2'
ANGULAR_UNITS = 'Msun proper kpc km / s'


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError('Explicit finite '+('nonnegative ' if allow_zero else 'positive ')+name+' required')
    return float(value)


def validate_isolated_history_request(request):
    if request.get('schema') != 'isolated-halo-control-history-request-v1':
        raise ValueError('Declared isolated halo control history required')
    for key in ('maximum_particles', 'maximum_snapshots', 'chunk_rows', 'minimum_core_particles', 'minimum_consecutive_crossings'):
        if type(request[key]) is not int or request[key] < 1:
            raise ValueError('Explicit positive integer history bound required: '+key)
    snapshots = request['snapshots']
    if not isinstance(snapshots, list) or not 3 <= len(snapshots) <= request['maximum_snapshots'] or len(set(snapshots)) != len(snapshots) or any(type(s) is not int or s < 0 for s in snapshots):
        raise ValueError('At least three distinct bounded saved snapshots required')
    if request['minimum_consecutive_crossings'] > len(snapshots)-1:
        raise ValueError('Collapse persistence exceeds the available saved history')
    _positive(request['memory_gib'], 'history memory')
    _positive(request['core_radius_proper_kpc'], 'fixed central density aperture')
    _positive(request['collapse_density_ratio'], 'declared collapse density ratio')
    if request['collapse_density_ratio'] <= 1:
        raise ValueError('Collapse threshold must exceed initial measured core density')
    _positive(request['epoch_tolerance'], 'native epoch tolerance', True)
    for key in ('energy_absolute_floor', 'angular_absolute_floor', 'mass_relative_tolerance', 'energy_relative_tolerance', 'angular_relative_tolerance'):
        _positive(request[key], key, key.endswith('tolerance'))
    if request['potential_gauge'] != 'isolated_zero_at_infinity' or request['boundary_condition'] != 'isolated_nonperiodic':
        raise ValueError('Native isolated zero-at-infinity potential and nonperiodic boundary required')
    if request['angular_policy'] not in {'diagnostic_only', 'predeclared_drift_bound'}:
        raise ValueError('Angular momentum is either diagnostic or has an explicit predeclared drift bound')
    frame = request['center_frame']
    if not isinstance(frame, dict) or frame.get('mode') not in {'fixed_inertial', 'center_of_mass'}:
        raise ValueError('Fixed inertial or measured center-of-mass frame required')
    if frame['mode'] == 'fixed_inertial':
        for key in ('position_proper_kpc', 'velocity_kms'):
            value = np.asarray(frame[key], float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError('Finite declared inertial origin and velocity required')
    times = request['requested_native_times']
    if not isinstance(times, list) or len(times) != len(snapshots) or any(isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) for t in times) or np.any(np.diff(times) <= 0):
        raise ValueError('Every saved snapshot must have an increasing declared native time')
    return request




def isolated_history_input_bindings(parameters):
    from .data import native_publication_markers
    from .workflow import verify_completed_receipt
    request = validate_isolated_history_request(parameters['request'])
    receipt_path = Path(parameters['native_stage_receipt']).resolve()
    if receipt_path.stat().st_size*8 > request['memory_gib']*1024**3:
        raise MemoryError('Native receipt exceeds the history workspace')
    receipt = verify_completed_receipt(receipt_path)
    root = Path(receipt['generation']).resolve()
    owned = {str(root/name): sha for name, sha in receipt['files_sha256'].items()}
    sources = {str(receipt_path):file_sha256(receipt_path), **receipt['source_hashes'], **receipt['input_hashes'], **owned}
    for snapshot in request['snapshots']:
        files = original_native_snapshot_files(Path(parameters['native_run_directory']), snapshot)
        for path in [*files, *native_publication_markers(files)]:
            name = str(path.resolve())
            if owned.get(name) != file_sha256(path):
                raise ValueError('Control history snapshot is outside the actual completed evolution generation')
    native_root = Path(parameters['native_run_directory']).resolve()
    configuration = native_root/'input/Config.sh'
    if owned.get(str(configuration)) != file_sha256(configuration):
        raise ValueError('Control history requires its actual owned native build configuration')
    flags = {line.split('#',1)[0].strip().split('=',1)[0].strip() for line in configuration.read_text().splitlines()}
    if not {'SELFGRAVITY','EVALPOTENTIAL','OUTPUT_POTENTIAL'} <= flags:
        raise ValueError('Native control build must provide self-gravity and saved gravitational potential')
    if 'PERIODIC' in flags or any(flag.startswith('EXTERNALGRAVITY') for flag in flags):
        raise ValueError('Isolated half-sum potential energy excludes periodic or external-gravity builds')
    manifest = native_root/'manifest.json'
    if manifest.exists() and owned.get(str(manifest)) != file_sha256(manifest):
        raise ValueError('Native control manifest is outside its evolved generation')
    return sources


class _Compensated:
    """Neumaier summation of bounded chunk reductions."""
    def __init__(self, shape=()):
        self.total = np.zeros(shape, float)
        self.correction = np.zeros(shape, float)
    def add(self, value):
        value = np.asarray(value, float)
        combined = self.total+value
        self.correction += np.where(abs(self.total) >= abs(value), (self.total-combined)+value, (value-combined)+self.total)
        self.total = combined
    def value(self):
        return self.total+self.correction


def _native_epoch(root, snapshot, request, *, center=None, velocity_frame=None):
    import h5py
    from .data import verify_native_hdf5_set, snapshot_total_counts, dataset_to_cgs_factor, exact_uint64_ids
    files = original_native_snapshot_files(root, snapshot)
    verify_native_hdf5_set(files, require_publication=True)
    with h5py.File(files[0], 'r') as f:
        counts = snapshot_total_counts(f['Header'].attrs)
        if any(int(counts[k]) for k in (0, 3, 4, 5)):
            raise ValueError('Isolated elastic DM conservation control excludes gas and other source species')
        count = int(counts[1])+int(counts[2])
    if not count or count > request['maximum_particles'] or count*32+request['chunk_rows']*256 > request['memory_gib']*1024**3:
        raise MemoryError('Full-system exact identity and chunk workspace exceed history allocation')
    mass_sum, kinetic, potential_sum = _Compensated(), _Compensated(), _Compensated()
    position_moment, momentum, angular = _Compensated((3,)), _Compensated((3,)), _Compensated((3,))
    core_mass, core_mass2 = _Compensated(), _Compensated()
    core_count = 0
    identities, native_time, unit_time, parameter_signature = [], None, None, None
    for file in files:
        with h5py.File(file, 'r') as f:
            header, params = f['Header'].attrs, f['Parameters'].attrs
            if int(params['ComovingIntegrationOn']) != 0 or int(params.get('PeriodicBoundariesOn', 0)) != 0:
                raise ValueError('Cosmological or periodic outputs cannot use an isolated energy gate')
            current = float(header['Time'])
            time_unit = float(params['UnitTime_in_s'])
            h = float(params['HubbleParam'])
            signature = tuple(float(params[k]) for k in ('UnitMass_in_g','UnitLength_in_cm','UnitVelocity_in_cm_per_s','UnitTime_in_s','HubbleParam'))
            if not math.isfinite(current) or not math.isfinite(time_unit) or time_unit <= 0 or h <= 0 or parameter_signature is not None and signature != parameter_signature or native_time is not None and current != native_time:
                raise ValueError('Native control shards change the epoch or physical units')
            native_time, unit_time, parameter_signature = current, time_unit, signature
            for kind in (1, 2):
                n = int(header['NumPart_ThisFile'][kind])
                if not n:
                    continue
                group = f['PartType'+str(kind)]
                for name in ('Coordinates','Velocities','ParticleIDs','Potential'):
                    if name not in group or len(group[name]) != n:
                        raise ValueError('Native conservation requires complete '+name+' output')
                for start in range(0, n, request['chunk_rows']):
                    sl = slice(start, min(start+request['chunk_rows'], n))
                    ids = exact_uint64_ids(group['ParticleIDs'][sl], allow_empty=False)
                    identities.append(ids)
                    x = np.asarray(group['Coordinates'][sl],float)*dataset_to_cgs_factor(group['Coordinates'].attrs,1.,h)/KPC_CM
                    v = np.asarray(group['Velocities'][sl],float)*dataset_to_cgs_factor(group['Velocities'].attrs,1.,h)/1e5
                    phi = np.asarray(group['Potential'][sl],float)*dataset_to_cgs_factor(group['Potential'].attrs,1.,h)/1e10
                    if 'Masses' in group:
                        m = np.asarray(group['Masses'][sl],float)*dataset_to_cgs_factor(group['Masses'].attrs,1.,h)/MSUN_G
                    else:
                        # Noncomoving native output has zero a/h exponents,
                        # including when a fixed MassTable replaces Masses.
                        m = np.full(len(ids),float(header['MassTable'][kind])*float(params['UnitMass_in_g'])/MSUN_G)
                    if x.shape != (len(ids),3) or v.shape != x.shape or phi.shape != m.shape or not all(np.isfinite(a).all() for a in (x,v,phi,m)) or np.any(m <= 0):
                        raise ValueError('Nonfinite or incomplete native conservation payload')
                    mass_sum.add(m.sum(dtype=float)); kinetic.add(.5*np.sum(m*np.sum(v*v,axis=1),dtype=float))
                    potential_sum.add(.5*np.sum(m*phi,dtype=float))
                    position_moment.add(np.sum(m[:,None]*x,axis=0,dtype=float))
                    momentum.add(np.sum(m[:,None]*v,axis=0,dtype=float))
                    angular.add(np.sum(m[:,None]*np.cross(x,v),axis=0,dtype=float))
                    if center is not None:
                        distance = np.linalg.norm(x-center,axis=1)
                        mask = distance <= request['core_radius_proper_kpc']
                        core_count += int(mask.sum()); core_mass.add(m[mask].sum(dtype=float)); core_mass2.add(np.dot(m[mask],m[mask]))
    ids = np.concatenate(identities)
    ids.sort()
    if len(ids) != count or np.any(ids[1:] == ids[:-1]):
        raise ValueError('Native control does not retain an exact unique full-system census')
    identity_hash = hashlib.sha256(np.asarray(ids,dtype='<u8').tobytes()).hexdigest()
    total_mass = float(mass_sum.value()); p = momentum.value(); r = position_moment.value()/total_mass
    u = p/total_mass if velocity_frame is None else velocity_frame
    c = r if center is None else center
    # Express total orbital angular momentum in the requested moving frame.
    l = angular.value()-np.cross(c,p)-total_mass*np.cross(r,u)+total_mass*np.cross(c,u)
    # Energy conservation is measured in one fixed native inertial frame.
    # Subtracting epoch-dependent COM kinetic energy would hide bulk drift.
    return {'snapshot':snapshot,'native_time_code':native_time,'elapsed_time_gyr':native_time*unit_time/31557600/1e9,
        'mass_msun':total_mass,'kinetic_energy':float(kinetic.value()),'potential_energy':float(potential_sum.value()),
        'total_energy':float(kinetic.value()+potential_sum.value()),'orbital_angular_momentum':l.tolist(),
        'linear_momentum':p.tolist(),'center_of_mass_proper_kpc':r.tolist(),'velocity_of_mass_kms':(p/total_mass).tolist(),
        'core_mass_msun':float(core_mass.value()),'core_particle_count':core_count,
        'core_effective_particles':float(core_mass.value()**2/core_mass2.value()) if core_mass2.value() > 0 else 0.,
        'identity_census_sha256':identity_hash,'particle_count':count,'time_unit_seconds':unit_time}


def _collapse_bracket(rows, request):
    reference = rows[0]['core_density_msun_proper_kpc3']
    if not rows[0]['core_supported'] or reference <= 0:
        return {'status':'UNSUPPORTED','reason':'initial measured core density lacks declared support'}
    threshold = reference*request['collapse_density_ratio']
    persistence = request['minimum_consecutive_crossings']
    for index in range(1,len(rows)-persistence+1):
        window = rows[index:index+persistence]
        previous = rows[index-1]
        if previous['core_supported'] and previous['core_density_msun_proper_kpc3'] < threshold and all(q['core_supported'] and q['core_density_msun_proper_kpc3'] >= threshold for q in window):
            if any(not row['core_supported'] for row in rows[:index]):
                return {'status':'UNSUPPORTED','threshold_density_msun_proper_kpc3':threshold,
                    'reason':'unsupported earlier saved epochs may hide an earlier persistent crossing'}
            return {'status':'BRACKETED','threshold_density_msun_proper_kpc3':threshold,
                'native_time_interval':[previous['native_time_code'],window[0]['native_time_code']],
                'elapsed_time_gyr_interval':[previous['elapsed_time_gyr'],window[0]['elapsed_time_gyr']],
                'consecutive_saved_crossings':persistence,'interpolation':'NONE; only observed saved-time bracket is retained'}
    if any(not q['core_supported'] for q in rows):
        return {'status':'UNSUPPORTED','threshold_density_msun_proper_kpc3':threshold,'reason':'unsupported epochs may hide a crossing'}
    return {'status':'NOT_BRACKETED','threshold_density_msun_proper_kpc3':threshold,
        'observed_native_time_interval':[rows[0]['native_time_code'],rows[-1]['native_time_code']],
        'reason':'no sustained saved-epoch crossing; future collapse is not inferred'}


def measure_isolated_control_history(parameters):
    request = validate_isolated_history_request(parameters['request'])
    sources = isolated_history_input_bindings(parameters)
    root = Path(parameters['native_run_directory']).resolve()
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('physics') not in {'dm','dmo','collisionless'}:
            raise ValueError('Conservation control requires actual DM-only native evolution')
    rows = []
    frame = request['center_frame']
    for snapshot, requested_time in zip(request['snapshots'],request['requested_native_times']):
        if frame['mode'] == 'center_of_mass':
            first_pass = _native_epoch(root,snapshot,request)
            center = np.asarray(first_pass['center_of_mass_proper_kpc'])
            velocity = np.asarray(first_pass['velocity_of_mass_kms'])
        else:
            center = np.asarray(frame['position_proper_kpc'],float)
            velocity = np.asarray(frame['velocity_kms'],float)
        row = _native_epoch(root,snapshot,request,center=center,velocity_frame=velocity)
        if abs(row['native_time_code']-requested_time) > request['epoch_tolerance']:
            raise ValueError('Actual native output time differs from the declared control schedule')
        if rows and (row['native_time_code'] <= rows[-1]['native_time_code'] or row['identity_census_sha256'] != rows[0]['identity_census_sha256'] or row['time_unit_seconds'] != rows[0]['time_unit_seconds']):
            raise ValueError('Closed-system history changes exact particle census or time units')
        row['core_density_msun_proper_kpc3'] = row['core_mass_msun']/(4*np.pi/3*request['core_radius_proper_kpc']**3)
        row['core_supported'] = bool(row['core_particle_count'] >= request['minimum_core_particles'] and row['core_effective_particles'] >= request['minimum_core_particles'])
        row['measured_center_proper_kpc'] = center.tolist()
        rows.append(row)
    first = rows[0]
    energy_floor = max(abs(first['total_energy']),request['energy_absolute_floor'])
    angular_floor = max(np.linalg.norm(first['orbital_angular_momentum']),request['angular_absolute_floor'])
    for row in rows:
        row['mass_relative_drift'] = abs(row['mass_msun']-first['mass_msun'])/first['mass_msun']
        row['energy_relative_drift'] = abs(row['total_energy']-first['total_energy'])/energy_floor
        row['angular_relative_drift'] = float(np.linalg.norm(np.asarray(row['orbital_angular_momentum'])-first['orbital_angular_momentum'])/angular_floor)
    maximum = {key:max(q[key] for q in rows) for key in ('mass_relative_drift','energy_relative_drift','angular_relative_drift')}
    gates = {'mass':maximum['mass_relative_drift'] <= request['mass_relative_tolerance'],
        'energy':maximum['energy_relative_drift'] <= request['energy_relative_tolerance']}
    if request['angular_policy'] == 'predeclared_drift_bound':
        gates['angular'] = maximum['angular_relative_drift'] <= request['angular_relative_tolerance']
    for path, sha in sources.items():
        if file_sha256(path) != sha:
            raise ValueError('Native source changed during conservation/collapse measurement')
    return {'schema':'isolated-halo-control-history-v1','request_hash':canonical_digest(request),'request':request,
        'epochs':rows,'maximum_measured_drift':maximum,'control_gates':gates,
        'status':'PASS' if all(gates.values()) else 'FAIL','execution_status':'COMPLETE',
        'collapse':_collapse_bracket(rows,request),'source_hashes':sources,
        'native_stage_receipt':str(Path(parameters['native_stage_receipt']).resolve()),
        'units':{'energy':ENERGY_UNITS,'orbital_angular_momentum':ANGULAR_UNITS,'linear_momentum':'Msun km / s',
            'density':'Msun / proper kpc^3','elapsed_time':'Gyr'},
        'potential_gauge':request['potential_gauge'],'angular_policy':request['angular_policy'],
        'science_acceptance':'CONTROL_ONLY','scope':'actual saved epochs of this isolated DM-only closed system; no cosmological energy gate',
        'uncertainty':'deterministic saved-history drift; independent numerical/operator/phase controls supply ensemble uncertainty'}


def isolated_history_runner(parameters, context):
    if parameters['request']['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('Isolated history exceeds declared workflow allocation')
    record = measure_isolated_control_history(parameters)
    path = context.generation/'isolated-control-history.json'
    atomic_json(path,record); atomic_text(path.with_name(path.name+'.sha256'),file_sha256(path)+'\n')
    return {'measurements':record['maximum_measured_drift'],
        'findings':[{'metric':name,'reason':'predeclared isolated drift bound failed'} for name,ok in record['control_gates'].items() if not ok]}


def isolated_history_main():
    import argparse
    parser = argparse.ArgumentParser(description='Measure native isolated halo conservation and collapse history')
    parser.add_argument('--parameters',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep previous native control evidence; choose a new output')
    parameters = json.loads(args.parameters.read_text())
    report = measure_isolated_control_history(parameters)
    atomic_json(args.output,report);atomic_text(args.output.with_name(args.output.name+'.sha256'),file_sha256(args.output)+'\n')


def _numeric_control(reference, memory_gib, sources):
    """Load an immutable numeric producer product, with expanded-byte bounds."""
    import zipfile
    from .data import verify_numeric_audit, numeric_audit_arrays_path, numeric_audit_source_bindings
    path = Path(reference['path']).resolve()
    if file_sha256(path) != reference['sha256']:
        raise ValueError('Native control product changed')
    archive = path.with_name(path.name+'.data')/'arrays.npz'
    with zipfile.ZipFile(archive) as packed:
        if sum(q.file_size for q in packed.infolist())*8 > memory_gib*1024**3:
            raise MemoryError('Native control arrays exceed declared workspace')
    report = verify_numeric_audit(path)
    for name, sha in numeric_audit_source_bindings(report).items():
        if name in sources and sources[name] != sha:
            raise ValueError('Native control products have conflicting provenance')
        sources[name] = sha
    sources[str(path)] = file_sha256(path)
    sources[str(path.with_name(path.name+'.sha256'))] = file_sha256(path.with_name(path.name+'.sha256'))
    for file in archive.parent.rglob('*'):
        if file.is_file():
            sources[str(file.resolve())] = file_sha256(file)
    with np.load(numeric_audit_arrays_path(path,report),allow_pickle=False) as arrays:
        return report,{name:arrays[name].copy() for name in arrays.files}


def validate_control_history_request(parameters):
    """Read only original gravity-free control trajectories, with exact IDs."""
    request=parameters['request']
    keys={'schema','population','engine','snapshots','maximum_particles','maximum_snapshots','maximum_bins',
        'chunk_rows','memory_gib','speed_bin_edges_kms','velocity_bin_edges_kms','epoch_tolerance',
        'relative_mass_tolerance','clock_absolute_tolerance_seconds','clock_relative_tolerance','maximum_clock_evaluations'}
    mapped=isinstance(request,dict) and request.get('schema')=='sidm-canonical-control-history-request-v1'
    if mapped:keys.add('mapped_run')
    if isinstance(request,dict) and 'particle_paths' in request:keys.add('particle_paths')
    if not isinstance(request,dict) or set(request)!=keys or request['schema'] not in {'sidm-physical-control-history-request-v1','sidm-canonical-control-history-request-v1'}:
        raise ValueError('Complete dedicated physical control history request required')
    if request['engine'] not in {'gadget4','opengadget3'}:raise ValueError('Original Gadget4 or independent OpenGadget3 outputs required')
    if mapped:
        if request['engine']!='gadget4':raise ValueError('Mapped history requires the actual canonical Gadget4 source profile')
        if not isinstance(request['mapped_run'],dict) or set(request['mapped_run'])!={'path','sha256'}:
            raise ValueError('Mapped history needs its actual source-bound native run publication')
    for name in ('maximum_particles','maximum_snapshots','maximum_bins','chunk_rows','maximum_clock_evaluations'):
        if type(request[name])is not int or request[name]<1:raise ValueError('Exact positive control history bound required: '+name)
    for name in ('memory_gib','epoch_tolerance','relative_mass_tolerance','clock_absolute_tolerance_seconds','clock_relative_tolerance'):
        _positive(request[name],name)
    if request['clock_relative_tolerance']>=1 or request['relative_mass_tolerance']>=1:
        raise ValueError('Fractional clock/mass tolerances must be below one')
    snapshots=request['snapshots']
    deferred=isinstance(snapshots,dict) and set(snapshots)=={'stage','artifact','load','field'} and snapshots.get('load')=='json' and snapshots.get('field')==['snapshots']
    if not deferred and (not isinstance(snapshots,list) or not 2<=len(snapshots)<=request['maximum_snapshots']):
        raise ValueError('At least two bounded original saved epochs required')
    previous=None
    for row in (() if deferred else snapshots):
        if not isinstance(row,dict) or set(row)!={'files','native_time'} or not isinstance(row['files'],list) or not row['files'] or len(row['files'])>request['maximum_particles']:
            raise ValueError('Each saved epoch needs its complete explicit original shard set and native time')
        time=_positive(row['native_time'],'native saved time',True)
        if previous is not None and time<=previous:raise ValueError('Saved native times must be increasing')
        previous=time
    for name in ('speed_bin_edges_kms','velocity_bin_edges_kms'):
        values=np.asarray(request[name],float)
        if values.ndim!=1 or not 2<=len(values)<=request['maximum_bins']+1 or not np.isfinite(values).all() or np.any(np.diff(values)<=0):
            raise ValueError('Explicit bounded ordered physical velocity bins required: '+name)
        if name=='speed_bin_edges_kms' and values[0]!=0:raise ValueError('Speed bins must start at zero')
    selected=0
    if 'particle_paths' in request:
        from .data import exact_uint64_ids
        paths=request['particle_paths']
        if not isinstance(paths,dict) or set(paths)!={'particle_ids','maximum_selected_particles'}:
            raise ValueError('Particle paths require exact original IDs and an explicit selection ceiling')
        ceiling=paths['maximum_selected_particles']
        if type(ceiling)is not int or not 1<=ceiling<=request['maximum_particles']:
            raise ValueError('Positive particle selection ceiling within the original census bound required')
        if not isinstance(paths['particle_ids'],list) or not 1<=len(paths['particle_ids'])<=ceiling:
            raise ValueError('Nonempty bounded explicit particle ID selection required')
        selected=len(exact_uint64_ids(paths['particle_ids'],allow_empty=False))
    saved=request['maximum_snapshots'] if deferred else len(snapshots)
    if request['maximum_particles']*720+saved*(request['maximum_bins']*8192+selected*512)>request['memory_gib']*1024**3:
        raise MemoryError('Exact-ID state, cohort statistics and history publication exceed allocation')
    return request


def _control_history_path(parameters,value):
    if not isinstance(value,str) or not value:raise ValueError('Resolved native control path required')
    return (Path(parameters['request_base'])/value).expanduser().resolve()


def _control_execution(parameters):
    """Bind parameters and outputs to a completed, actually invoked engine."""
    from .workflow import verify_completed_receipt
    from core.native import verify_invocation_inputs
    from .evolution import parameter_values
    request=validate_control_history_request(parameters);memory=request['memory_gib']*1024**3
    receipt_path=_control_history_path(parameters,parameters['native_stage_receipt'])
    if receipt_path.stat().st_size*8>memory:raise MemoryError('Control receipt exceeds metadata allocation')
    receipt=verify_completed_receipt(receipt_path);generation=Path(receipt['generation']).resolve()
    owned={str((generation/name).resolve()):digest for name,digest in receipt['files_sha256'].items()}
    sources={str(receipt_path):file_sha256(receipt_path),**receipt['source_hashes'],**receipt['input_hashes']}
    paths={name:_control_history_path(parameters,parameters[name]) for name in ('parameter_file','build_configuration','invocation_receipt')}
    for path in paths.values():
        if owned.get(str(path))!=file_sha256(path) or path.stat().st_size*8>memory:
            raise ValueError('Control configuration/invocation must be inside its actual completed generation')
        sources[str(path)]=owned[str(path)]
    invocation=json.loads(paths['invocation_receipt'].read_text())
    if invocation.get('schema')!='native-command-invocation-v1' or invocation.get('status')!='RETURNED' or invocation.get('returncode')!=0:
        raise ValueError('Control history requires an actual successful native invocation')
    verify_invocation_inputs(invocation);sources.update(invocation['input_hashes'])
    for name,value in invocation.get('scientific_environment',{}).items():
        if name!='RESEARCH_SIDM_REFERENCE_TABLE' or invocation['input_hashes'].get(value)!=file_sha256(value):
            raise ValueError('Scientific invocation environment lacks its retained physical input')
    cwd=Path(invocation['cwd']).resolve()
    arguments=[(Path(str(arg)) if Path(str(arg)).is_absolute() else cwd/str(arg)).resolve() for arg in invocation['command']]
    if paths['parameter_file'] not in arguments or str(paths['parameter_file']) not in invocation['input_hashes']:
        raise ValueError('The supplied control parameter file was not used by the actual engine invocation')
    flags={line.split('#',1)[0].strip().split('=',1)[0].strip() for line in paths['build_configuration'].read_text().splitlines()}
    if request['engine']=='gadget4':
        if 'SELFGRAVITY' in flags or any(name.startswith(('EXTERNALGRAVITY','PMGRID')) for name in flags):
            raise ValueError('Homogeneous elastic control history excludes gravitational forcing')
        if not {'PERIODIC','FORCE_EQUAL_TIMESTEPS'}<=flags:raise ValueError('Homogeneous native controls require periodic synchronized evolution')
        if ('RESEARCH_SIDM_CANONICAL_CONTROL' in flags)!=(request['schema']=='sidm-canonical-control-history-request-v1'):
            raise ValueError('Native history adapter differs from the actual canonical or physical build frame')
    else:
        if not {'NOGRAVITY','PERIODIC'}<=flags or any(name.startswith(('PMGRID','EXTERNALGRAVITY')) for name in flags):
            raise ValueError('Independent homogeneous controls require explicit periodic NOGRAVITY builds')
        if 'DARKENERGY' in flags or 'TIMEDEPDE' in flags:raise ValueError('Independent custom expansion needs its own retained clock adapter')
    params=parameter_values(paths['parameter_file'])
    table=params.get('CosmologyTableFile','none')
    if table.lower() not in {'none',''}:
        table_path=Path(table)
        params['CosmologyTableFile']=str((table_path if table_path.is_absolute() else cwd/table_path).resolve())
    if params.get('SIDMNativeTablePath','none').lower() not in {'none',''}:
        table_path=Path(params['SIDMNativeTablePath'])
        params['SIDMNativeTablePath']=str((table_path if table_path.is_absolute() else cwd/table_path).resolve())
    comoving=int(params['ComovingIntegrationOn'])
    if comoving not in (0,1):raise ValueError('Control clock must be explicitly physical or comoving')
    for name in ('UnitLength_in_cm','UnitMass_in_g','UnitVelocity_in_cm_per_s','HubbleParam','BoxSize'):
        _positive(float(params[name]),name)
    for row in request['snapshots']:
        files=[_control_history_path(parameters,path) for path in row['files']]
        if len(set(files))!=len(files):raise ValueError('Repeated native control snapshot shard')
        for path in files:
            if owned.get(str(path))!=file_sha256(path):raise ValueError('Control snapshot is outside its actual completed evolution generation')
            sources[str(path)]=owned[str(path)]
    return request,params,flags,sources,invocation


def _control_elapsed_seconds(times,params,engine,sources,request,*,canonical_scale_power=0):
    """Use the actual native expansion and h-scaled proper-time convention."""
    from .foundation import H100_S
    unit_time=float(params['UnitLength_in_cm'])/float(params['UnitVelocity_in_cm_per_s']);h=float(params['HubbleParam'])
    start=float(params['TimeBegin'])
    if any(time<start for time in times):raise ValueError('A saved control precedes the actual native start')
    if canonical_scale_power not in (0,2):raise ValueError('Select proper time or explicit canonical transport clock')
    if not int(params['ComovingIntegrationOn']):
        if canonical_scale_power:raise ValueError('Canonical expanding clock requires actual cosmological evolution')
        return (np.asarray(times)-start)*unit_time/h,np.zeros(len(times))
    if start<=0:raise ValueError('Cosmological control requires a positive actual initial scale factor')
    table=params.get('CosmologyTableFile','none');knots=[];log_a=log_e=None
    if table.lower() not in {'none',''}:
        path=Path(table).resolve()
        if engine!='gadget4' or sources.get(str(path))!=file_sha256(path):raise ValueError('Native expansion table must be bound by its actual invocation')
        if path.stat().st_size*32>request['memory_gib']*1024**3:raise MemoryError('Native expansion table exceeds clock workspace')
        values=np.loadtxt(path,ndmin=2)
        if values.shape[1]!=4 or len(values)<2 or not np.isfinite(values).all() or np.any(values<=0) or np.any(np.diff(values[:,0])<=0):
            raise ValueError('Complete actual ordered native a,E,D,f expansion table required')
        if start<values[0,0] or max(times)>values[-1,0]:raise ValueError('Saved controls escape the actual native expansion support')
        log_a=np.log(values[:,0]);log_e=np.log(values[:,1]);knots=log_a.tolist()
    h0=float(params['Hubble'])*h/unit_time if engine=='gadget4' else H100_S*h
    _positive(h0,'actual native physical H0')
    om=float(params['Omega0']);ol=float(params['OmegaLambda']);radiation=float(params.get('OmegaRadiation',0.));curvature=1-om-ol-radiation
    evaluations=0
    def integrand(log_scale):
        nonlocal evaluations
        evaluations+=1
        if evaluations>request['maximum_clock_evaluations']:raise RuntimeError('Physical control clock evaluation budget exhausted')
        scale=math.exp(log_scale)
        e=math.exp(float(np.interp(log_scale,log_a,log_e))) if log_a is not None else math.sqrt(om/scale**3+curvature/scale**2+radiation/scale**4+ol)
        value=1/(scale**canonical_scale_power*h0*e)
        if not math.isfinite(value) or value<=0:raise ValueError('Native control expansion has no finite positive proper-time derivative')
        return value
    from scipy.integrate import quad
    seconds=[];errors=[]
    for time in times:
        left=math.log(start);right=math.log(time);boundaries=[left,*[k for k in knots if left<k<right],right]
        total=0.;error=0.
        for lo,hi in zip(boundaries,boundaries[1:]):
            if hi==lo:continue
            value,estimate=quad(integrand,lo,hi,epsabs=request['clock_absolute_tolerance_seconds']/max(1,len(boundaries)-1),
                epsrel=request['clock_relative_tolerance'],limit=max(1,min(1000,request['maximum_clock_evaluations']//21)))
            total+=value;error+=estimate
        if error>request['clock_absolute_tolerance_seconds']+request['clock_relative_tolerance']*abs(total):
            raise ValueError('Actual native physical clock does not meet its declared quadrature tolerance')
        seconds.append(total);errors.append(error)
    return np.asarray(seconds),np.asarray(errors)


def _mapped_control_epochs(times,params,sources,request):
    """Invert the actual retained canonical clock, then measure proper time.

    This reads the native clock's physical background, not the static launch
    template. Brackets/quadrature errors are estimates, not interval proofs.
    All inverse and proper-clock evaluations share one request work budget.
    """
    path=Path(params['SIDMCanonicalClockPath']).resolve()
    if sources.get(str(path))!=params['SIDMCanonicalClockSHA256'] or file_sha256(path)!=params['SIDMCanonicalClockSHA256']:
        raise ValueError('Mapped history clock lacks the original invocation binding')
    if path.stat().st_size>int(params['SIDMCanonicalMaximumAssetBytes']) or path.stat().st_size*32>request['memory_gib']*1024**3/8:
        raise MemoryError('Mapped history clock metadata exceeds allocation')
    words=path.read_text(encoding='ascii').split()
    if len(words)<17 or words[0]!='RESEARCH_SIDM_CANONICAL_CLOCK_V1' or words[1]!=params['SIDMAngularModelHash']:
        raise ValueError('Mapped history clock model/schema differs from actual native source')
    kind=int(words[3]);h0,a0,a1,absolute,relative=map(float,words[4:9])
    panels,inverse=int(words[9]),int(words[10]);epoch_relative=float(words[11]);rows=int(words[12])
    matter,lambda_,radiation,curvature=map(float,words[13:17])
    if kind not in (0,1) or rows<0 or rows>int(params['SIDMCanonicalMaximumRows']) or len(words)!=17+2*rows or (kind==0 and rows) or (kind==1 and rows<2):
        raise ValueError('Mapped history clock table support/row count invalid')
    for value in (h0,a0,a1,absolute,relative,epoch_relative):_positive(value,'mapped clock bound')
    if a1<=a0 or relative>=1 or epoch_relative>=1 or panels<2 or inverse<1 or not all(math.isfinite(value) for value in (matter,lambda_,radiation,curvature)):
        raise ValueError('Mapped history background domain or inverse allocation invalid')
    if a0!=float(params['SIDMCanonicalInitialScaleFactor']) or float(params['TimeBegin'])!=0 or int(params['ComovingIntegrationOn'])!=0:
        raise ValueError('Mapped history initial native/physical clock binding invalid')
    table=np.asarray([float(value) for value in words[17:]],float).reshape(rows,2)
    if rows and (not np.isfinite(table).all() or np.any(table<=0) or np.any(np.diff(table[:,0])<=0) or a0<table[0,0] or a1>table[-1,0]):
        raise ValueError('Mapped history clock escapes its original positive native expansion table')
    loga=np.log(table[:,0]) if rows else None;loge=np.log(table[:,1]) if rows else None
    used=0
    def charge():
        nonlocal used
        used+=1
        if used>request['maximum_clock_evaluations']:raise RuntimeError('Mapped history aggregate clock work exhausted')
    def forward(a,power):
        if a<a0 or a>a1 or not math.isfinite(a):raise ValueError('Mapped history clock cannot extrapolate')
        if a==a0:return 0.,0.
        left,right=math.log(a0),math.log(a)
        if rows:
            boundaries=[left,*[value for value in loga if left<value<right],right]
            total=np.longdouble(0.)
            for lo,hi in zip(boundaries,boundaries[1:]):
                charge();mid=lo+(hi-lo)/2;index=int(np.searchsorted(loga,mid,side='right')-1)
                if not 0<=index<rows-1:raise ValueError('Mapped history cannot identify its native expansion interval')
                slope=np.longdouble(loge[index+1]-loge[index])/np.longdouble(loga[index+1]-loga[index])
                left_e=np.longdouble(loge[index])+slope*np.longdouble(lo-loga[index]);dx=np.longdouble(hi-lo)
                z=(-power-slope)*dx;factor=dx if z==0 else dx*np.expm1(z)/z
                part=np.exp(-power*np.longdouble(lo)-np.log(np.longdouble(h0))-left_e)*factor
                if not np.isfinite(part) or part<=0:raise ArithmeticError('Mapped native table clock interval overflows')
                total+=part
            value=float(total);error=256*np.finfo(float).eps*abs(value)*max(1,len(boundaries)-1)
        else:
            from scipy.integrate import quad
            def integrand(log_scale):
                charge();scale=math.exp(log_scale)
                squared=matter/scale**3+curvature/scale**2+radiation/scale**4+lambda_
                if not math.isfinite(squared) or squared<=0:raise ValueError('Mapped original background lacks a positive Hubble rate')
                value=1/(scale**power*h0*math.sqrt(squared))
                if not math.isfinite(value) or value<=0:raise ArithmeticError('Mapped original clock derivative overflows')
                return value
            result=quad(integrand,left,right,epsabs=min(absolute,request['clock_absolute_tolerance_seconds']),
                epsrel=min(relative,request['clock_relative_tolerance']),limit=max(1,min(panels,1000,request['maximum_clock_evaluations']//21)),full_output=1)
            if len(result)!=3:raise ValueError('Mapped background quadrature fails its declared clock budgets')
            value,error=result[:2]
        if not math.isfinite(value) or value<=0 or not math.isfinite(error) or error<0 or error>request['clock_absolute_tolerance_seconds']+request['clock_relative_tolerance']*value:
            raise ValueError('Mapped clock estimate fails its request tolerance')
        return value,error
    unit=float(params['UnitLength_in_cm'])/float(params['UnitVelocity_in_cm_per_s'])/float(params['HubbleParam'])
    tau=np.asarray(times,float)*unit
    if not np.isfinite(tau).all() or np.any(tau<0) or np.any(np.diff(tau)<=0):raise ValueError('Mapped history needs increasing finite canonical elapsed times')
    full,full_error=forward(a1,2);scales=[];proper=[];proper_errors=[];inverse_errors=[];brackets=[]
    for target in tau:
        endpoint_budget=min(absolute,request['clock_absolute_tolerance_seconds'])+min(relative,request['clock_relative_tolerance'])*abs(target)+full_error
        if target>full+endpoint_budget:raise ValueError('Mapped saved history escapes its retained canonical clock domain and endpoint uncertainty')
        if target==0:a=a0;error=0.;width=0.
        elif target>=full:a=a1;error=full_error+abs(target-full);width=0.
        else:
            lo,hi=math.log(a0),math.log(a1)
            for iteration in range(inverse):
                middle=lo+(hi-lo)/2
                if middle<=lo or middle>=hi:raise ArithmeticError('Mapped inverse clock falls below epoch precision')
                a=math.exp(middle);value,error=forward(a,2);width=math.expm1(hi-lo)
                if width<=epoch_relative and abs(value-target)<=min(absolute,request['clock_absolute_tolerance_seconds'])+min(relative,request['clock_relative_tolerance'])*target+error:break
                if value<target:lo=middle
                else:hi=middle
            else:raise ValueError('Mapped inverse epoch lookup exhausts its actual native iteration allocation')
        elapsed,proper_error=forward(a,0)
        scales.append(a);proper.append(elapsed);proper_errors.append(proper_error);inverse_errors.append(error);brackets.append(width)
    return {'scale_factor':np.asarray(scales),'elapsed_proper_seconds':np.asarray(proper),
        'clock_quadrature_error_seconds':np.asarray(proper_errors),'canonical_elapsed_seconds':tau,
        'canonical_clock_quadrature_error_seconds':np.asarray(inverse_errors),'inverse_epoch_bracket_relative_width':np.asarray(brackets),
        'clock_function_evaluations':used,'clock_background_hash':words[2]}


def control_history_input_bindings(parameters):
    request,params,flags,sources,invocation=_control_execution(parameters)
    from .sidm_analysis import _load_product
    population,arrays,binding=_load_product(request['population'],parameters['request_base'],request['memory_gib']*1024**3/2)
    mapped=request['schema']=='sidm-canonical-control-history-request-v1'
    if population.get('schema')!=('sidm-canonical-control-population-v1' if mapped else 'sidm-control-population-v1'):
        raise ValueError('History must retain its actual physical or canonical population')
    sources.update(binding)
    if mapped:
        from .workflow import verify_completed_receipt
        run_path=_control_history_path(parameters,request['mapped_run']['path'])
        receipt=verify_completed_receipt(_control_history_path(parameters,parameters['native_stage_receipt']))
        owned={str((Path(receipt['generation'])/name).resolve()):sha for name,sha in receipt['files_sha256'].items()}
        if owned.get(str(run_path))!=request['mapped_run']['sha256'] or file_sha256(run_path)!=request['mapped_run']['sha256'] or run_path.stat().st_size*16>request['memory_gib']*1024**3:
            raise ValueError('Mapped history run publication must belong to its actual native stage payload')
        run=json.loads(run_path.read_text());clock=population['native_clock']
        if run.get('schema')!='sidm-canonical-native-run-v1' or run.get('execution_status')!='COMPLETE' or run['canonical_state_sha256']!=population['canonical_state_sha256'] or run['physical_state_sha256']!=population['original_physical_state_sha256'] or run['event_identity_namespace']!=population['event_identity_namespace']:
            raise ValueError('Mapped history changes the original run/population state or identity')
        for name in ('parameter_file','build_configuration','invocation_receipt'):
            if Path(run[name]).resolve()!=_control_history_path(parameters,parameters[name]):raise ValueError('Mapped history points to another native '+name)
        actual_clock=Path(params['SIDMCanonicalClockPath']).resolve()
        if params['SIDMCanonicalClockSHA256']!=clock['sha256'] or params['SIDMAngularModelHash']!=clock['physical_model_hash'] or params['SIDMCanonicalPhysicalStateSHA256']!=population['original_physical_state_sha256'] or params['SIDMEventIdentityNamespace']!=population['event_identity_namespace'] or owned.get(str(actual_clock))!=clock['sha256'] or invocation['input_hashes'].get(str(actual_clock))!=clock['sha256']:
            raise ValueError('Mapped history clock/model/physical population lacks its original native binding')
        if float(params['SIDMCanonicalInitialScaleFactor'])!=population['initial_scale_factor'] or params['SIDMReferenceBackendKind']!='0':
            raise ValueError('Mapped history changes its initial physical epoch or native source backend')
        sources[str(run_path)]=request['mapped_run']['sha256'];sources[str(actual_clock)]=clock['sha256']
    original=Path(params['InitCondFile'])
    if not original.is_absolute():original=Path(invocation['cwd'])/original
    if not original.is_file():original=Path(str(original)+'.hdf5')
    original=original.resolve()
    if file_sha256(original)!=population['native_ic']['sha256'] or invocation['input_hashes'].get(str(original))!=file_sha256(original):
        raise ValueError('Control did not evolve the supplied population native IC')
    sources[str(original)]=file_sha256(original)
    u=population['native_units'] if mapped else population['request']['native_units']
    initial_clock={'mode':'physical'} if mapped else population['request']['clock']
    for key,expected in (('UnitLength_in_cm',u['length_cm']),('UnitMass_in_g',u['mass_g']),
            ('UnitVelocity_in_cm_per_s',u['velocity_cm_s']),('HubbleParam',u['hubble_parameter']),
            ('BoxSize',population['native_ic']['box_size_code']),('TimeBegin',population['native_ic']['native_time'])):
        if not math.isclose(float(params[key]),expected,rel_tol=1e-12,abs_tol=0.):raise ValueError('Executed control differs from its physical population '+key)
    if bool(int(params['ComovingIntegrationOn']))!=(initial_clock['mode']=='comoving'):
        raise ValueError('Native control changes its original clock representation')
    population['measured_invocation_environment']=invocation.get('scientific_environment',{})
    population['measured_invocation_cwd']=invocation['cwd']
    return request,params,flags,sources,population,arrays


def publish_control_history(parameters,destination):
    """Read original Gadget4/OpenGadget3 elastic populations and cohort histories."""
    import h5py
    from .data import exact_uint64_ids,snapshot_total_counts,dataset_to_cgs_factor,verify_native_hdf5_set,native_publication_markers,export_numeric_audit
    request,params,flags,sources,population,initial=control_history_input_bindings(parameters)
    mapped=request['schema']=='sidm-canonical-control-history-request-v1'
    ids=exact_uint64_ids(initial['particle_ids']);order=np.argsort(ids);ids=ids[order]
    labels=np.asarray(initial['cohort_index'],int)[order];original_mass=np.asarray(initial['mass_msun'],float)[order]
    count=len(ids);cohorts=len(population['cohorts']);epochs=len(request['snapshots'])
    if count>request['maximum_particles'] or np.any((labels<0)|(labels>=cohorts)):raise ValueError('Control original cohort census exceeds its declared support')
    if not np.array_equal(np.bincount(labels,minlength=cohorts),[row['particle_count'] for row in population['cohorts']]):
        raise ValueError('Physical population cohort labels contradict the retained original census')
    original_position=np.asarray(initial['original_position_proper_kpc'] if mapped else initial['position_proper_kpc'],float)[order]
    comoving=bool(int(params['ComovingIntegrationOn']));h=float(params['HubbleParam'])
    L=float(params['UnitLength_in_cm']);V=float(params['UnitVelocity_in_cm_per_s']);M=float(params['UnitMass_in_g'])
    native_times=np.asarray([row['native_time'] for row in request['snapshots']],float)
    mapped_clock=_mapped_control_epochs(native_times,params,sources,request) if mapped else None
    if mapped and (np.any(np.diff(mapped_clock['scale_factor'])<=0) or np.any(np.diff(mapped_clock['elapsed_proper_seconds'])<=0)):
        raise ValueError('Mapped saved epochs cannot be distinguished at the declared inverse-clock precision')
    if abs(native_times[0]-float(params['TimeBegin']))>request['epoch_tolerance']:raise ValueError('Control history must include its actual initial saved epoch')
    shapes={'cohort_mass_msun':(epochs,cohorts),'cohort_mean_velocity_kms':(epochs,cohorts,3),
        'cohort_covariance_kms2':(epochs,cohorts,3,3),'cohort_kinetic_energy':(epochs,cohorts),'cohort_canonical_kinetic_energy':(epochs,cohorts),
        'cohort_rare_event_participations':(epochs,cohorts),'cohort_microscopic_exposure':(epochs,cohorts),
        'cohort_speed_bin_counts':(epochs,cohorts,len(request['speed_bin_edges_kms'])+1),
        'cohort_velocity_bin_counts':(epochs,cohorts,3,len(request['velocity_bin_edges_kms'])+1)}
    arrays={name:np.zeros(shape,dtype=np.uint64 if name.endswith('bin_counts') else float) for name,shape in shapes.items()}
    path_slots=None;path_ids=None
    if 'particle_paths' in request:
        path_ids=np.sort(exact_uint64_ids(request['particle_paths']['particle_ids'],allow_empty=False))
        selected_at=np.searchsorted(ids,path_ids)
        if np.any(selected_at>=count) or not np.array_equal(ids[selected_at],path_ids):
            raise ValueError('Selected particle path IDs are absent from the original physical population')
        path_slots=np.full(count,-1,dtype=np.int64);path_slots[selected_at]=np.arange(len(path_ids))
        initial_a=population['initial_scale_factor'] if mapped else population['request']['clock']['scale_factor'] if comoving else 1.
        path_box=population['canonical_box_kpc'] if mapped else population['request']['box_proper_kpc']/initial_a
        if not math.isfinite(path_box) or path_box<=0:raise ValueError('Particle paths lack a finite canonical periodic domain')
        arrays.update(particle_ids=path_ids,particle_cohort_index=labels[selected_at].astype(np.int64),
            particle_mass_msun=original_mass[selected_at],particle_canonical_box_kpc=np.asarray(path_box))
        for name in ('particle_position_canonical_kpc','particle_velocity_peculiar_kms','particle_velocity_canonical_kms'):
            arrays[name]=np.empty((epochs,len(path_ids),3))
    diagnostic_availability=[];actual_scales=[];actual_times=[];participations=[];previous_rare=None;previous_exposure=None
    for epoch,row in enumerate(request['snapshots']):
        files=[_control_history_path(parameters,path) for path in row['files']]
        if request['engine']=='gadget4':
            verify_native_hdf5_set(files,require_publication=True)
            for marker in native_publication_markers(files):sources[str(marker.resolve())]=file_sha256(marker)
        velocity=np.empty((count,3));mass=np.empty(count);seen=np.zeros(count,dtype=bool)
        rare=np.zeros(count,dtype=np.uint64);exposure=np.zeros(count);available=None;actual=None;scale=None;precision=np.finfo(float).eps
        for path in files:
            with h5py.File(path,'r') as handle:
                header=handle['Header'].attrs;totals=snapshot_total_counts(header)
                if request['engine']=='gadget4':
                    if 'Config' not in handle or 'Parameters' not in handle:raise ValueError('Native control must retain its actual compiled flags and parameters')
                    actual_flags=set(handle['Config'].attrs)
                    if 'SELFGRAVITY' in actual_flags or any(name.startswith(('EXTERNALGRAVITY','PMGRID')) for name in actual_flags) or not {'PERIODIC','FORCE_EQUAL_TIMESTEPS'}<=actual_flags:
                        raise ValueError('Saved native build differs from the declared gravity-free synchronized control')
                    if ('RESEARCH_SIDM_CANONICAL_CONTROL' in actual_flags)!=mapped:
                        raise ValueError('Saved native snapshot changes the declared canonical frame')
                    if mapped:
                        for name,value in {'Research_sidm_canonical_control_schema':1,'Research_sidm_canonical_initial_scale_factor':population['initial_scale_factor'],
                                'Research_sidm_canonical_clock_SHA256':population['native_clock']['sha256'],
                                'Research_sidm_physical_model_hash':population['native_clock']['physical_model_hash'],
                                'Research_sidm_original_physical_state_SHA256':population['original_physical_state_sha256'],
                                'Research_sidm_position_frame':'canonical_x=r/a','Research_sidm_velocity_frame':'canonical_u=a*v_pec',
                                'Research_sidm_clock_frame':'tau_integral_dt_over_a_squared'}.items():
                            actual_value=header.get(name)
                            if isinstance(actual_value,bytes):actual_value=actual_value.decode('ascii')
                            if actual_value!=value:raise ValueError('Mapped saved snapshot loses frame/lineage: '+name)
                    for key in ('UnitLength_in_cm','UnitMass_in_g','UnitVelocity_in_cm_per_s','HubbleParam','ComovingIntegrationOn','BoxSize'):
                        if not math.isclose(float(handle['Parameters'].attrs[key]),float(params[key]),rel_tol=1e-12):
                            raise ValueError('Saved native parameters differ from the actually invoked configuration')
                    for key,value in params.items():
                        if not key.startswith('SIDM'):continue
                        if key not in handle['Parameters'].attrs:raise ValueError('Saved control is missing its actual SIDM parameter: '+key)
                        saved=handle['Parameters'].attrs[key]
                        if isinstance(saved,bytes):saved=saved.decode('utf-8')
                        if key.endswith('Path') and value.lower() not in {'none',''}:
                            # Relative spelling is allowed only when it resolves
                            # to the same actually invoked source-bound table.
                            recorded=Path(str(saved))
                            if not recorded.is_absolute():
                                recorded=Path(population['measured_invocation_cwd'])/recorded
                            equal=str(recorded.resolve())==value
                        elif key.endswith(('Hash','SHA256','Seed')) or key.endswith('Path'):
                            equal=str(saved)==value
                        else:
                            try:equal=math.isclose(float(saved),float(value),rel_tol=1e-12,abs_tol=0.)
                            except (ValueError,TypeError):equal=str(saved)==value
                        if not equal:raise ValueError('Saved native SIDM physics differs from its actual invocation: '+key)
                if int(totals[1])!=count or any(int(totals[k]) for k in (0,2,3,4,5)) or int(header['NumFilesPerSnapshot'])!=len(files):
                    raise ValueError('Saved control requires a complete one-species original census')
                current=float(header['Time']);a=float(mapped_clock['scale_factor'][epoch]) if mapped else current if comoving else 1.
                if not math.isfinite(current) or abs(current-row['native_time'])>request['epoch_tolerance'] or actual is not None and current!=actual:
                    raise ValueError('Control shards change or miss their actual requested epoch')
                actual=current;scale=a
                if not math.isclose(float(header['HubbleParam']),h,rel_tol=1e-12) or not math.isclose(float(header['BoxSize']),float(params['BoxSize']),rel_tol=1e-12):
                    raise ValueError('Saved native units/periodic domain differ from the executed parameters')
                number=int(header['NumPart_ThisFile'][1])
                if number==0:continue
                group=handle['PartType1'];fields=('ParticleIDs','Coordinates','Velocities')
                if any(name not in group or len(group[name])!=number for name in fields):raise ValueError('Native control particle columns are incomplete')
                if group['Velocities'].dtype.kind!='f':raise ValueError('Native velocity storage must be floating point')
                precision=max(precision,np.finfo(group['Velocities'].dtype).eps)
                present=all(name in group for name in ('SIDMInteractionCount','SIDMExposure'))
                if available is not None and available!=present:raise ValueError('Control shards change SIDM diagnostic availability')
                available=present
                for start in range(0,number,request['chunk_rows']):
                    sl=slice(start,min(start+request['chunk_rows'],number));current_ids=exact_uint64_ids(group['ParticleIDs'][sl])
                    at=np.searchsorted(ids,current_ids)
                    if np.any(at>=count) or not np.array_equal(ids[at],current_ids) or len(np.unique(at))!=len(at) or np.any(seen[at]):raise ValueError('Control snapshot changes or duplicates original particle IDs')
                    seen[at]=True
                    if request['engine']=='gadget4':
                        native_a=1. if mapped else a
                        factor_v=dataset_to_cgs_factor(group['Velocities'].attrs,native_a,h)/1e5
                        factor_x=dataset_to_cgs_factor(group['Coordinates'].attrs,native_a,h)/KPC_CM
                        expected_v=V/1e5 if mapped else V*math.sqrt(a)/1e5
                        expected_x=L/h/KPC_CM if mapped else L*a/h/KPC_CM
                        if not math.isclose(factor_v,expected_v,rel_tol=1e-12) or not math.isclose(factor_x,expected_x,rel_tol=1e-12):
                            raise ValueError('Native control coordinate/velocity metadata contradict the actual Gadget storage convention')
                    else:factor_v=V*math.sqrt(a)/1e5;factor_x=L*a/h/KPC_CM
                    v=np.asarray(group['Velocities'][sl],float)*factor_v;x=np.asarray(group['Coordinates'][sl],float)*factor_x
                    if mapped:v=v/a;x=x*a
                    if 'Masses' in group:
                        factor_m=dataset_to_cgs_factor(group['Masses'].attrs,a,h)/MSUN_G if request['engine']=='gadget4' else M/h/MSUN_G
                        m=np.asarray(group['Masses'][sl],float)*factor_m
                    else:m=np.full(len(at),float(header['MassTable'][1])*M/h/MSUN_G)
                    if v.shape!=(len(at),3) or x.shape!=v.shape or m.shape!=(len(at),) or not all(np.isfinite(value).all() for value in (v,x,m)) or np.any(m<=0):
                        raise ValueError('Native physical control state is nonfinite or misaligned')
                    if np.any(np.abs(m-original_mass[at])>request['relative_mass_tolerance']*original_mass[at]):raise ValueError('Elastic controls change original macro masses')
                    if epoch==0:
                        if group['Coordinates'].dtype.kind!='f':raise ValueError('Native coordinate storage must be floating point')
                        box=population['canonical_box_kpc']*population['initial_scale_factor'] if mapped else population['request']['box_proper_kpc'];shift=x-original_position[at]
                        shift-=box*np.rint(shift/box)
                        if np.any(np.abs(shift)>64*np.finfo(group['Coordinates'].dtype).eps*box):
                            raise ValueError('Initial saved periodic coordinates differ from the original physical population')
                    velocity[at]=v;mass[at]=m
                    if path_slots is not None:
                        chosen=path_slots[at]>=0;slots=path_slots[at[chosen]]
                        arrays['particle_position_canonical_kpc'][epoch,slots]=np.remainder(x[chosen]/a,path_box)
                        arrays['particle_velocity_peculiar_kms'][epoch,slots]=v[chosen]
                        arrays['particle_velocity_canonical_kms'][epoch,slots]=a*v[chosen]
                    if present:
                        values=exact_uint64_ids(group['SIDMInteractionCount'][sl],unique=False,allow_empty=True)
                        depth=np.asarray(group['SIDMExposure'][sl],float)
                        if values.shape!=(len(at),) or depth.shape!=values.shape or not np.isfinite(depth).all() or np.any(depth<0):raise ValueError('Actual SIDM count/exposure state is invalid')
                        rare[at]=values;exposure[at]=depth
        if not seen.all():raise ValueError('Control shard list omits original particles')
        if epoch==0:
            initial_velocity=np.asarray(initial['original_velocity_peculiar_kms'] if mapped else initial['velocity_peculiar_kms'],float)[order]
            # Snapshot precision can differ across engines; the finite roundoff
            # allowance follows actual file precision rather than a free fit.
            allowance=64*precision*np.maximum(np.abs(initial_velocity),np.maximum(np.max(np.abs(initial_velocity),axis=0),1.))
            if np.any(np.abs(velocity-initial_velocity)>allowance):raise ValueError('Initial saved control velocity differs from the original physical population')
        if available:
            if previous_rare is not None and (np.any(rare<previous_rare) or np.any(exposure<previous_exposure)):
                raise ValueError('Persistent native SIDM counters/exposure decrease along the original trajectory')
            previous_rare=rare.copy();previous_exposure=exposure.copy()
        elif previous_rare is not None:raise ValueError('Native control loses persistent SIDM diagnostics between saved epochs')
        actual_times.append(actual);actual_scales.append(scale);diagnostic_availability.append(bool(available));totals=[]
        for cohort in range(cohorts):
            chosen=labels==cohort;weights=mass[chosen];v=velocity[chosen];total=weights.sum();mean=np.sum(weights[:,None]*v,axis=0)/total
            delta=v-mean;energy=.5*np.sum(weights*np.sum(v*v,axis=1))
            arrays['cohort_mass_msun'][epoch,cohort]=total;arrays['cohort_mean_velocity_kms'][epoch,cohort]=mean
            arrays['cohort_covariance_kms2'][epoch,cohort]=np.einsum('n,ni,nj->ij',weights,delta,delta)/total
            arrays['cohort_kinetic_energy'][epoch,cohort]=energy;arrays['cohort_canonical_kinetic_energy'][epoch,cohort]=scale**2*energy
            def histogram(values,edges):
                return np.bincount(np.searchsorted(np.asarray(edges),values,side='right'),minlength=len(edges)+1).astype(np.uint64)
            arrays['cohort_speed_bin_counts'][epoch,cohort]=histogram(np.linalg.norm(v,axis=1),request['speed_bin_edges_kms'])
            for axis in range(3):arrays['cohort_velocity_bin_counts'][epoch,cohort,axis]=histogram(v[:,axis],request['velocity_bin_edges_kms'])
            # Keep exact counts as decimal strings; float arrays are views only.
            exact=sum(int(value) for value in rare[chosen]);totals.append(str(exact) if available else None)
            arrays['cohort_rare_event_participations'][epoch,cohort]=float(exact) if available else np.nan
            arrays['cohort_microscopic_exposure'][epoch,cohort]=float(exposure[chosen].sum()) if available else np.nan
        participations.append(totals)
    for name,value in arrays.items():
        if name in {'cohort_rare_event_participations','cohort_microscopic_exposure'}:
            supported=np.asarray(diagnostic_availability,dtype=bool)
            if not np.isfinite(value[supported]).all():raise ValueError('Native diagnostic accumulation overflows: '+name)
        elif not np.isfinite(value).all():raise ValueError('Native cohort history accumulation overflows: '+name)
    native_times=np.asarray(actual_times,float)
    if mapped:elapsed,clock_errors=mapped_clock['elapsed_proper_seconds'],mapped_clock['clock_quadrature_error_seconds']
    else:elapsed,clock_errors=_control_elapsed_seconds(native_times,params,request['engine'],sources,request)
    arrays.update(elapsed_proper_seconds=elapsed,clock_quadrature_error_seconds=clock_errors,native_time_code=native_times,
        scale_factor=np.asarray(actual_scales),sidm_diagnostics_available=np.asarray(diagnostic_availability,dtype=bool),
        speed_bin_edges_kms=np.asarray(request['speed_bin_edges_kms']),velocity_bin_edges_kms=np.asarray(request['velocity_bin_edges_kms']))
    parameter_hash=canonical_digest(params)
    report={'schema':'sidm-physical-control-history-v1','execution_status':'COMPLETE','request':request,'engine':request['engine'],
        'request_base':str(Path(parameters['request_base']).resolve()),
        'population_physical_state_sha256':population['original_physical_state_sha256'] if mapped else population['physical_state_sha256'],'identity_namespace':population['event_identity_namespace'],
        'phase_seed':population['phase_seed'] if mapped else population['request']['seed'],'operator_seed':params.get('SIDMOperatorSeed') if request['engine']=='gadget4' else params.get('MSIDM_SEED'),
        'cohorts':population['cohorts'],'cohort_particle_counts':[row['particle_count'] for row in population['cohorts']],
        'comoving':comoving,'native_parameter_sha256':parameter_hash,'native_parameters':params,
        'native_build_flags':sorted(flags),
        'invocation_environment':population['measured_invocation_environment'],
        'exact_rare_event_participations_decimal':participations,
        'diagnostic_definition':'Gadget4 SIDMInteractionCount is rare event participation; SIDMExposure is microscopic optical depth, including frequent contributions; missing independent fields remain unavailable',
        'histogram_definition':'inertial physical velocity/speed; explicit first underflow and last overflow bins; interior bins [left,right)',
        'energy_definition':'physical kinetic energy and a^2 times physical kinetic energy, separately retained; no potential or fabricated heat-flux measurement',
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    if path_ids is not None:
        report['particle_paths']={'schema':'sidm-exact-id-saved-particle-paths-v1','selected_particle_count':len(path_ids),
            'particle_ids_decimal':[str(int(value)) for value in path_ids],'identity_namespace':report['identity_namespace'],
            'canonical_box_kpc':path_box,'position_definition':'wrapped x=r/a in [0,L); a=1 for static physical controls',
            'velocity_definition':'physical peculiar v and canonical u=a*v, in km/s',
            'sampling_scope':'actual saved epochs only; no inferred winding number or continuous path between snapshots'}
    if mapped:
        for name in ('canonical_elapsed_seconds','canonical_clock_quadrature_error_seconds','inverse_epoch_bracket_relative_width'):arrays[name]=mapped_clock[name]
        report.update(mapped_transport=True,mapped_run=request['mapped_run'],canonical_state_sha256=population['canonical_state_sha256'],
            native_clock=population['native_clock'],clock_function_evaluations=mapped_clock['clock_function_evaluations'],
            original_background_hash=mapped_clock['clock_background_hash'],expanding_background=True,
            mapped_history_definition='physical r=a*x and v=u/a at inverse-clock snapshot epoch; native time remains canonical tau code')
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    units={name:'dimensionless' for name in arrays}
    units.update(cohort_mass_msun='Msun',cohort_mean_velocity_kms='km / s',cohort_covariance_kms2='(km / s)^2',
        cohort_kinetic_energy=ENERGY_UNITS,cohort_canonical_kinetic_energy=ENERGY_UNITS,elapsed_proper_seconds='s',clock_quadrature_error_seconds='s',
        native_time_code='native time code',speed_bin_edges_kms='km / s',velocity_bin_edges_kms='km / s')
    if mapped:units.update(canonical_elapsed_seconds='s',canonical_clock_quadrature_error_seconds='s')
    if path_ids is not None:
        units.update(particle_position_canonical_kpc='kpc',particle_canonical_box_kpc='kpc',particle_mass_msun='Msun',
            particle_velocity_peculiar_kms='km / s',particle_velocity_canonical_kms='km / s',particle_ids='identity')
    specifications={name:{'units':units[name],'quantity':'original native control '+name} for name in arrays}
    if path_ids is not None:specifications['particle_ids']['identity_namespace']=report['identity_namespace']
    return export_numeric_audit(destination,report,arrays,specifications,
        algorithm=source,context={'epoch':{'canonical_time':'per-row actual native time','elapsed_proper_seconds_array':'elapsed_proper_seconds'},
            'coordinates':{'frame':'inertial proper periodic Cartesian; peculiar velocity'}},source_hashes=sources)


def validate_control_clock_mapping_request(parameters):
    request=parameters['request']
    keys={'schema','history','physical_model','native_differential_table','canonical_speed_grid_kms','cosine_grid',
        'maximum_epochs','maximum_speeds','maximum_angles','maximum_clock_evaluations','memory_gib',
        'clock_absolute_tolerance_seconds','clock_relative_tolerance'}
    if not isinstance(request,dict) or set(request)!=keys or request.get('schema')!='sidm-canonical-clock-mapping-request-v1':
        raise ValueError('Complete canonical expanding-control mapping request required')
    for name in ('maximum_epochs','maximum_speeds','maximum_angles','maximum_clock_evaluations'):
        if type(request[name])is not int or request[name]<2:raise ValueError('Exact integer canonical mapping work bound of at least two required: '+name)
    for name in ('memory_gib','clock_absolute_tolerance_seconds','clock_relative_tolerance'):_positive(request[name],name)
    if request['clock_relative_tolerance']>=1:raise ValueError('Clock fractional tolerance must be below one')
    for name,maximum in (('canonical_speed_grid_kms','maximum_speeds'),('cosine_grid','maximum_angles')):
        values=np.asarray(request[name],float)
        if values.ndim!=1 or not 2<=len(values)<=request[maximum] or not np.isfinite(values).all() or np.any(np.diff(values)<=0):raise ValueError('Explicit bounded strictly increasing canonical mapping grid required: '+name)
    if request['canonical_speed_grid_kms'][0]<0:raise ValueError('Canonical relative speed must be nonnegative')
    if request['cosine_grid'][0]!=-1 or request['cosine_grid'][-1]!=1:raise ValueError('Complete directed cosine support [-1,1] required; particle folding is retained by the physical model')
    for name in ('history','physical_model','native_differential_table'):
        reference=request[name]
        if name=='native_differential_table' and reference is None:continue
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('Exact retained mapping input reference required: '+name)
        for field in ('path','sha256'):
            value=reference[field]
            if isinstance(value,dict) and 'stage' in value:continue
            if not isinstance(value,str) or not value or field=='sha256' and (len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):raise ValueError('Complete path/SHA mapping input required: '+name)
    rows=request['maximum_epochs']*len(request['canonical_speed_grid_kms'])*len(request['cosine_grid'])
    if rows*128>request['memory_gib']*1024**3/2:raise MemoryError('Canonical law/state mapping workspace exceeds allocation')
    return request


def control_clock_mapping_input_bindings(parameters):
    from .sidm_analysis import _load_product
    request=validate_control_clock_mapping_request(parameters);sources=_load_product(request['history'],parameters['request_base'],request['memory_gib']*1024**3/3)[2]
    for name in ('physical_model','native_differential_table'):
        reference=request[name]
        if reference is None:continue
        path=(Path(parameters['request_base'])/reference['path']).resolve()
        if path.stat().st_size*16>request['memory_gib']*1024**3/3 or file_sha256(path)!=reference['sha256']:raise ValueError('Canonical mapping input changed or exceeds workspace: '+name)
        sources[str(path)]=reference['sha256']
    return sources


def publish_control_clock_mapping(parameters,destination):
    """Produce the actual tau clock and time-dependent law for a static x,u representation.

    x=r/a, u=a*v_pec and d tau=dt/a^2 remove free-streaming
    Hubble drag. Fixed canonical density and u then require
    d(sigma_eff/m)/dOmega=a^-2 d(sigma/m)/dOmega(u/a,mu).
    The law is time dependent; a static unmodified cross section is not
    a physical/comoving equivalence control.
    """
    from .sidm_analysis import _load_product
    from .sidm_physics import ElasticSIDMModel
    from .data import export_numeric_audit
    request=validate_control_clock_mapping_request(parameters);base=parameters['request_base'];memory=request['memory_gib']*1024**3
    sources=control_clock_mapping_input_bindings(parameters)
    history,values,binding=_load_product(request['history'],base,memory/3);sources.update(binding)
    if history.get('schema')!='sidm-physical-control-history-v1' or history.get('execution_status')!='COMPLETE' or history.get('comoving')is not True:
        raise ValueError('Actual completed expanding native homogeneous control history required')
    model_path=(Path(base)/request['physical_model']['path']).resolve();spec=json.loads(model_path.read_text())
    model=ElasticSIDMModel(spec)
    scheme=_control_physical_model_binding(history,spec,request['native_differential_table'],base,sources)
    a=np.asarray(values['scale_factor'],float);times=np.asarray(values['native_time_code'],float)
    if (a.ndim!=1 or not 2<=len(a)<=request['maximum_epochs'] or times.shape!=a.shape or not np.isfinite(a).all()
            or np.any(a<=0) or np.any(np.diff(a)<=0) or not np.array_equal(a,times)):
        raise ValueError('Actual increasing scale-factor native epochs required')
    contracts=history['numeric_data']['array_contract']['datasets']
    for name,unit in (('cohort_mass_msun','Msun'),('cohort_mean_velocity_kms','km / s'),('cohort_covariance_kms2','(km / s)^2'),
            ('cohort_kinetic_energy',ENERGY_UNITS),('elapsed_proper_seconds','s')):
        if contracts[name]['units']!=unit:raise ValueError('Canonical mapping input physical units differ: '+name)
    clock_request=dict(history['request'])
    for name in ('memory_gib','maximum_clock_evaluations','clock_absolute_tolerance_seconds','clock_relative_tolerance'):clock_request[name]=request[name]
    tau,tau_error=_control_elapsed_seconds(times,history['native_parameters'],history['engine'],sources,clock_request,canonical_scale_power=2)
    if tau.shape!=a.shape or tau[0]!=0 or np.any(np.diff(tau)<=0):raise ValueError('Canonical transport clock is not resolved and increasing')
    speed=np.asarray(request['canonical_speed_grid_kms'],float);mu=np.asarray(request['cosine_grid'],float)
    differential=np.empty((len(a),len(speed),len(mu)));physical_total=np.empty((len(a),len(speed)));effective_total=np.empty_like(physical_total)
    for row,scale in enumerate(a):
        physical_speed=speed/scale
        differential[row]=np.asarray(model.differential_cross_section_per_mass(physical_speed[:,None],mu[None,:]),float)/scale**2
        physical_total[row]=np.asarray([model.total_cross_section_per_mass(float(v)) for v in physical_speed])
        effective_total[row]=physical_total[row]/scale**2
    mass=np.asarray(values['cohort_mass_msun'],float);mean=np.asarray(values['cohort_mean_velocity_kms'],float);cov=np.asarray(values['cohort_covariance_kms2'],float)
    if mass.shape!=(len(a),len(history['cohorts'])) or mean.shape!=(*mass.shape,3) or cov.shape!=(*mass.shape,3,3) or np.any(mass<=0):raise ValueError('Complete actual cohort state mapping required')
    box_code=float(history['native_parameters']['BoxSize']);unit_length=float(history['native_parameters']['UnitLength_in_cm']);h=float(history['native_parameters']['HubbleParam'])
    box=box_code*unit_length/h/KPC_CM
    density=mass.sum(axis=1)/box**3*MSUN_G/KPC_CM**3
    physical_rate=density[:,None]/a[:,None]**3*physical_total*(speed[None,:]/a[:,None])*1e5
    canonical_rate=density[:,None]*effective_total*speed[None,:]*1e5
    arrays={'scale_factor':a,'elapsed_proper_seconds':values['elapsed_proper_seconds'],'canonical_elapsed_seconds':tau,
        'canonical_clock_quadrature_error_seconds':tau_error,'canonical_speed_grid_kms':speed,'cosine_grid':mu,
        'effective_differential_cm2_g_sr':differential,'physical_total_cm2_g':physical_total,'effective_total_cm2_g':effective_total,
        'cohort_mass_msun':mass,'cohort_canonical_mean_velocity_kms':a[:,None,None]*mean,
        'cohort_canonical_covariance_kms2':a[:,None,None,None]**2*cov,
        'cohort_canonical_kinetic_energy':a[:,None]**2*values['cohort_kinetic_energy'],
        'homogeneous_physical_rate_per_second':physical_rate,'homogeneous_canonical_rate_per_second':canonical_rate,
        'rate_mapping_absolute_residual_per_second':abs(canonical_rate-a[:,None]**2*physical_rate)}
    if any(not np.isfinite(value).all() for value in arrays.values()) or np.any(differential<0) or np.any(effective_total<0):raise ArithmeticError('Canonical physical-law/state mapping overflows')
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={'schema':'sidm-canonical-clock-mapping-v1','execution_status':'COMPLETE','request':request,'actual_engine':history['engine'],'actual_scheme':scheme,
        'request_base':str(Path(base).resolve()),
        'population_physical_state_sha256':history['population_physical_state_sha256'],'identity_namespace':history['identity_namespace'],
        'canonical_box_kpc':box,'physical_model_sha256':request['physical_model']['sha256'],'particle_convention':model.particle_convention,
        'mapping':{'position':'x=r/a','velocity':'u=a*v_pec','clock':'d tau=dt/a^2','differential_law':'f_eff(tau,u,mu)=a(tau)^-2*f(u/a(tau),mu)'},
        'sampling':'actual saved native epochs; declared canonical speed/cosine nodes; no interpolation error or transport timestep support inferred between nodes',
        'rate_definition':'homogeneous total microscopic optical-depth rate at each supplied canonical relative speed, not measured kernel pair counts or frequent-update counts',
        'scope':'mapped clock, physical-law nodes and measured canonical cohort states for gravity-free elastic expanding controls; no mapped native driver execution or distribution equivalence acceptance',
        'method_references':['https://arxiv.org/abs/astro-ph/9710119','https://wwwmpa.mpa-garching.mpg.de/gadget4/06_snapshotformat/'],
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    units={name:'dimensionless' for name in arrays}
    units.update(elapsed_proper_seconds='s',canonical_elapsed_seconds='s',canonical_clock_quadrature_error_seconds='s',canonical_speed_grid_kms='km / s',
        effective_differential_cm2_g_sr='cm^2 / g / sr',physical_total_cm2_g='cm^2 / g',effective_total_cm2_g='cm^2 / g',cohort_mass_msun='Msun',
        cohort_canonical_mean_velocity_kms='km / s',cohort_canonical_covariance_kms2='(km / s)^2',cohort_canonical_kinetic_energy=ENERGY_UNITS,
        homogeneous_physical_rate_per_second='1 / s',homogeneous_canonical_rate_per_second='1 / s',rate_mapping_absolute_residual_per_second='1 / s')
    return export_numeric_audit(destination,report,arrays,{name:{'units':units[name],'quantity':'actual canonical SIDM control mapping '+name} for name in arrays},algorithm=source,source_hashes=sources,
        context={'epoch':{'canonical_time':'canonical_elapsed_seconds','actual_scale_factor_array':'scale_factor'},'coordinates':{'frame':'x=r/a; u=a*v_pec; fixed periodic canonical box'}})


def validate_control_native_clock_request(parameters):
    request=parameters['request']
    keys={'schema','mapping','maximum_rows','maximum_asset_bytes','memory_gib','clock_absolute_tolerance_seconds',
        'clock_relative_tolerance','maximum_clock_panels','maximum_inverse_iterations','inverse_scale_factor_relative_tolerance'}
    if not isinstance(request,dict) or set(request)!=keys or request.get('schema')!='sidm-native-canonical-clock-request-v1':raise ValueError('Complete native canonical clock asset request required')
    for name,minimum in (('maximum_rows',2),('maximum_asset_bytes',128),('maximum_clock_panels',2),('maximum_inverse_iterations',1)):
        if type(request[name])is not int or request[name]<minimum:raise ValueError('Exact positive native canonical clock allocation required: '+name)
    for name in ('memory_gib','clock_absolute_tolerance_seconds','clock_relative_tolerance','inverse_scale_factor_relative_tolerance'):_positive(request[name],name)
    if max(request['clock_relative_tolerance'],request['inverse_scale_factor_relative_tolerance'])>=1:raise ValueError('Canonical native clock fractional budgets must be below one')
    reference=request['mapping']
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('Exact sealed canonical mapping required')
    for name in ('path','sha256'):
        value=reference[name]
        if isinstance(value,dict) and 'stage' in value:continue
        if not isinstance(value,str) or not value or name=='sha256' and (len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):raise ValueError('Complete mapping path/SHA required')
    if request['maximum_asset_bytes']*8>request['memory_gib']*1024**3/3 or request['maximum_rows']*256>request['memory_gib']*1024**3/3:raise MemoryError('Native clock ASCII/row workspace exceeds allocation')
    return request


def control_native_clock_input_bindings(parameters):
    from .sidm_analysis import _load_product
    request=validate_control_native_clock_request(parameters);memory=request['memory_gib']*1024**3
    mapping,_,sources=_load_product(request['mapping'],parameters['request_base'],memory/3)
    if mapping.get('schema')!='sidm-canonical-clock-mapping-v1' or 'request_base' not in mapping:raise ValueError('Current source-bound canonical mapping required; regenerate older products')
    history,_,binding=_load_product(mapping['request']['history'],mapping['request_base'],memory/3);sources.update(binding)
    if history.get('schema')!='sidm-physical-control-history-v1' or history.get('comoving')is not True:raise ValueError('Actual expanding native background history required')
    return sources


def publish_control_native_clock(parameters,destination):
    from .sidm_analysis import _load_product
    from .foundation import H100_S
    request=validate_control_native_clock_request(parameters);memory=request['memory_gib']*1024**3
    mapping,arrays,sources=_load_product(request['mapping'],parameters['request_base'],memory/3)
    if mapping.get('schema')!='sidm-canonical-clock-mapping-v1' or 'request_base' not in mapping:raise ValueError('Current canonical mapping source/clock lineage required')
    history,_,binding=_load_product(mapping['request']['history'],mapping['request_base'],memory/3);sources.update(binding)
    if history.get('schema')!='sidm-physical-control-history-v1' or history.get('comoving')is not True or history['engine']!=mapping['actual_engine']:
        raise ValueError('Canonical native clock changes the original expanding engine/background')
    if history['population_physical_state_sha256']!=mapping['population_physical_state_sha256']:raise ValueError('Canonical clock changes the original physical population')
    params=history['native_parameters'];a=np.asarray(arrays['scale_factor'],float)
    if a.ndim!=1 or len(a)<2 or not np.isfinite(a).all() or np.any(a<=0) or np.any(np.diff(a)<=0) or a[0]!=float(params['TimeBegin']):raise ValueError('Actual native canonical start/end epochs required')
    unit_time=float(params['UnitLength_in_cm'])/float(params['UnitVelocity_in_cm_per_s']);h=float(params['HubbleParam'])
    h0=float(params['Hubble'])*h/unit_time if history['engine']=='gadget4' else H100_S*h
    _positive(h0,'actual canonical clock H0')
    matter=float(params['Omega0']);lambda_=float(params['OmegaLambda']);radiation=float(params.get('OmegaRadiation',0.));curvature=1-matter-lambda_-radiation
    if not all(math.isfinite(v) for v in (matter,lambda_,radiation,curvature)):raise ValueError('Finite original native background coefficients required')
    table=params.get('CosmologyTableFile','none');knots=np.empty((0,2));kind=0
    background=canonical_digest({'physical_H0_per_second':h0,'matter':matter,'lambda':lambda_,'radiation':radiation,'curvature':curvature})
    if table.lower()not in ('none',''):
        path=Path(table).resolve()
        if history['engine']!='gadget4' or sources.get(str(path))!=file_sha256(path):raise ValueError('Clock expansion table lacks its actual native invocation binding')
        if path.stat().st_size*32>memory/3:raise MemoryError('Original native expansion table exceeds allocation')
        values=np.loadtxt(path,ndmin=2)
        if values.shape[1]!=4 or not 2<=len(values)<=request['maximum_rows'] or not np.isfinite(values).all() or np.any(values<=0) or np.any(np.diff(values[:,0])<=0) or a[0]<values[0,0] or a[-1]>values[-1,0]:
            raise ValueError('Actual native log-E/log-a clock support is incomplete or outside allocation')
        knots=values[:,:2];kind=1;background=file_sha256(path);sources[str(path)]=background
    model_path=(Path(mapping['request_base'])/mapping['request']['physical_model']['path']).resolve()
    if sources.get(str(model_path))!=file_sha256(model_path) or model_path.stat().st_size*16>memory/3:raise ValueError('Native canonical clock physical law lost its source binding')
    spec=json.loads(model_path.read_text());model_hash=canonical_digest(spec)
    _control_physical_model_binding(history,spec,mapping['request']['native_differential_table'],mapping['request_base'],sources)
    rows=['RESEARCH_SIDM_CANONICAL_CLOCK_V1',model_hash,background,str(kind),
        ' '.join(format(value,'.17g') for value in (h0,a[0],a[-1],request['clock_absolute_tolerance_seconds'],request['clock_relative_tolerance'])),
        ' '.join(str(request[name]) for name in ('maximum_clock_panels','maximum_inverse_iterations'))+' '+format(request['inverse_scale_factor_relative_tolerance'],'.17g')+' '+str(len(knots)),
        ' '.join(format(value,'.17g') for value in (matter,lambda_,radiation,curvature))]
    rows.extend(' '.join(format(value,'.17g') for value in row) for row in knots)
    payload='\n'.join(rows)+'\n'
    if len(payload.encode('ascii'))>request['maximum_asset_bytes']:raise MemoryError('Complete native clock asset exceeds declared byte bound')
    schedule_request=dict(mapping['request'])
    schedule_request['clock_absolute_tolerance_seconds']=min(schedule_request['clock_absolute_tolerance_seconds'],request['clock_absolute_tolerance_seconds'])
    schedule_request['clock_relative_tolerance']=min(schedule_request['clock_relative_tolerance'],request['clock_relative_tolerance'])
    seconds,schedule_errors=_control_elapsed_seconds(a,params,history['engine'],sources,schedule_request,canonical_scale_power=2)
    output_times=seconds*h/unit_time
    roundtrip=output_times*unit_time/h
    rounding=np.abs(roundtrip-seconds)
    if not np.isfinite(output_times).all() or output_times[0]!=0 or np.any(np.diff(output_times)<=0):
        raise ValueError('Original physical epochs are indistinguishable in the native canonical clock')
    allowed=request['clock_absolute_tolerance_seconds']+request['clock_relative_tolerance']*np.abs(seconds)
    if np.any(schedule_errors+rounding>allowed):raise ValueError('Canonical schedule quadrature/rounding exceeds its frozen native time budget')
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Retain previous native canonical clock assets')
    destination.mkdir(parents=True);asset=destination/'canonical-clock.txt';atomic_text(asset,payload)
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    for relative in ('sidm/research_canonical_clock.h','sidm/research_physical_clock.h','black_holes/research_asset_sha256.h'):
        native_source=source.parent/'software/gadget4/src'/relative;sources[str(native_source)]=file_sha256(native_source)
    report={'schema':'sidm-native-canonical-clock-v1','execution_status':'COMPLETE','request':request,
        'request_base':str(Path(parameters['request_base']).resolve()),
        'population_physical_state_sha256':mapping['population_physical_state_sha256'],'identity_namespace':mapping['identity_namespace'],
        'physical_H0_per_second':h0,
        'native_output_times':output_times.tolist(),'output_scale_factors':a.tolist(),
        'canonical_output_seconds':seconds.tolist(),'canonical_schedule_error_estimate_seconds':(schedule_errors+rounding).tolist(),
        'native_schedule_time_unit_seconds':unit_time/h,
        'schedule_definition':'original expanding saved physical epochs integrated at clock-asset tolerance; actual inverse-clock agreement measured downstream',
        'clock_asset':str(asset),'clock_asset_sha256':file_sha256(asset),'clock_asset_bytes':asset.stat().st_size,
        'physical_model_hash':model_hash,'original_background_hash':background,'background_kind':'native_logE_logA_table' if kind else 'native_matter_radiation_curvature_lambda',
        'actual_scale_factor_begin':float(a[0]),'actual_scale_factor_end':float(a[-1]),'canonical_box_kpc':mapping['canonical_box_kpc'],
        'source_hashes':sources,'scope':'original background forward/inverse canonical-clock asset and physical law binding; no mapped transport execution or numerical equivalence acceptance',
        'science_acceptance':'NOT_ESTABLISHED'}
    atomic_json(destination/'clock.json',report);return report


def control_native_clock_runner(parameters,context):
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native canonical clock exceeds allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Native clock asset escapes owned generation')
    return publish_control_native_clock(parameters,destination)


def control_native_clock_main():
    import argparse
    parser=argparse.ArgumentParser(description='Export source-bound native canonical background forward/inverse clock assets')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text());parameters.setdefault('request_base',str(path.parent))
    publish_control_native_clock(parameters,args.output);return 0


def control_clock_mapping_runner(parameters,context):
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Canonical control mapping exceeds allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Canonical mapping escapes owned generation')
    return publish_control_clock_mapping(parameters,destination)


def control_clock_mapping_main():
    import argparse
    parser=argparse.ArgumentParser(description='Produce source-bound canonical clock/law/state mappings for expanding elastic SIDM controls')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text());parameters.setdefault('request_base',str(path.parent))
    publish_control_clock_mapping(parameters,args.output);return 0


def control_history_runner(parameters,context):
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Control history exceeds its workflow allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Control history escapes its owned generation')
    return publish_control_history(parameters,destination)


def control_history_main():
    import argparse
    parser=argparse.ArgumentParser(description='Measure dedicated SIDM controls from original completed Gadget4/OpenGadget3 runs')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.parameters.resolve();parameters=json.loads(source.read_text())
    parameters.setdefault('request_base',str(source.parent));publish_control_history(parameters,args.output)
    return 0


_CONTROL_PATH_METRICS={'particle_position_canonical_kpc','particle_velocity_peculiar_kms','particle_velocity_canonical_kms'}


def _control_metric_unit(metric):
    return {'cohort_mass_msun':'Msun','cohort_mean_velocity_kms':'km / s','cohort_covariance_kms2':'(km / s)^2',
        'cohort_kinetic_energy':ENERGY_UNITS,'cohort_canonical_kinetic_energy':ENERGY_UNITS,
        'particle_position_canonical_kpc':'kpc','particle_velocity_peculiar_kms':'km / s',
        'particle_velocity_canonical_kms':'km / s'}.get(metric,'dimensionless')


def _control_particle_path_binding(report,arrays):
    """Keep exact row identity and the saved, wrapped phase-space convention."""
    from .data import exact_uint64_ids
    paths=report.get('particle_paths')
    if not isinstance(paths,dict) or paths.get('schema')!='sidm-exact-id-saved-particle-paths-v1':
        raise ValueError('Particle comparison requires actual selected native path publications on both sides')
    ids=exact_uint64_ids(arrays['particle_ids'],allow_empty=False)
    if ids.dtype!=np.dtype('uint64') or np.any(ids[1:]<=ids[:-1]) or len(ids)!=paths['selected_particle_count']:
        raise ValueError('Particle path publication changes exact sorted identity support')
    if paths['identity_namespace']!=report['identity_namespace'] or paths['particle_ids_decimal']!=[str(int(value)) for value in ids]:
        raise ValueError('Particle path namespace/decimal identity declaration differs from exact rows')
    declared=exact_uint64_ids(report['request']['particle_paths']['particle_ids'],allow_empty=False)
    if not np.array_equal(np.sort(declared),ids):raise ValueError('Saved particle rows differ from the original requested selection')
    box=np.asarray(arrays['particle_canonical_box_kpc'],float)
    if box.shape!=() or not np.isfinite(box) or float(box)<=0 or float(box)!=paths['canonical_box_kpc']:
        raise ValueError('Particle path canonical periodic domain is invalid')
    scale=np.asarray(arrays['scale_factor'],float);epochs=len(arrays['elapsed_proper_seconds'])
    if scale.shape!=(epochs,) or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('Saved particle paths lack actual physical expansion epochs')
    for name in _CONTROL_PATH_METRICS:
        value=np.asarray(arrays[name],float)
        if value.shape!=(epochs,len(ids),3) or not np.isfinite(value).all():raise ValueError('Incomplete saved particle phase-space rows: '+name)
    position=arrays['particle_position_canonical_kpc']
    if np.any(position<0) or np.any(position>=float(box)):raise ValueError('Saved positions violate the declared periodic wrapping')
    mass=np.asarray(arrays['particle_mass_msun'],float);cohorts=np.asarray(arrays['particle_cohort_index'])
    if mass.shape!=(len(ids),) or not np.isfinite(mass).all() or np.any(mass<=0) or cohorts.shape!=mass.shape or cohorts.dtype.kind not in 'iu' or np.any(cohorts<0) or np.any(cohorts>=len(report['cohort_particle_counts'])):
        raise ValueError('Selected particle mass/cohort lineage is incomplete')
    expected=scale[:,None,None]*arrays['particle_velocity_peculiar_kms']
    if np.any(np.abs(expected-arrays['particle_velocity_canonical_kms'])>64*np.finfo(float).eps*np.maximum(np.abs(expected),1.)):
        raise ValueError('Particle canonical velocity differs from the actual physical v and a')
    return ids,float(box)


def validate_control_comparison_request(parameters):
    request=parameters['request']
    keys={'schema','left','right','physical_model','native_differential_table','scope','metrics','maximum_absolute_error',
        'maximum_relative_error','relative_error_floor','minimum_particles_per_cohort','memory_gib','time_tolerance_seconds'}
    mapped=isinstance(request,dict) and request.get('scope')=='physical_comoving_mapped_distributions'
    if mapped:keys.update({'maximum_scale_factor_relative_error','maximum_inverse_epoch_bracket_relative_width'})
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='sidm-physical-control-comparison-request-v1':
        raise ValueError('Complete frozen native physical control comparison required')
    if request['scope'] not in {'independent_engine_trajectories','same_engine_refinement','rare_frequent_hybrid_overlap','physical_comoving_balance','physical_comoving_mapped_distributions'}:
        raise ValueError('Declare the physical control comparison scope')
    _positive(request['memory_gib'],'control comparison memory');_positive(request['time_tolerance_seconds'],'physical epoch matching',True)
    if mapped:
        for name in ('maximum_scale_factor_relative_error','maximum_inverse_epoch_bracket_relative_width'):
            _positive(request[name],name)
            if request[name]>=1:raise ValueError('Mapped physical epoch uncertainty budgets must be smaller than one')
    if type(request['minimum_particles_per_cohort'])is not int or request['minimum_particles_per_cohort']<2:
        raise ValueError('Positive effective control cohort census required')
    metrics=request['metrics']
    allowed={'cohort_mass_msun','cohort_mean_velocity_kms','cohort_covariance_kms2','cohort_kinetic_energy',
        'cohort_canonical_kinetic_energy','cohort_speed_cdf','cohort_velocity_cdf'}|_CONTROL_PATH_METRICS
    if not isinstance(metrics,list) or not metrics or len(set(metrics))!=len(metrics) or set(metrics)-allowed:
        raise ValueError('Explicit distinct supported physical control metrics required')
    for key in ('maximum_absolute_error','maximum_relative_error','relative_error_floor'):
        if not isinstance(request[key],dict) or set(request[key])!=set(metrics):raise ValueError('Every metric needs a frozen dimensional error definition')
        for name,value in request[key].items():_positive(value,key+'/'+name,key!='relative_error_floor')
    if request['scope']=='physical_comoving_balance' and set(metrics)-{'cohort_mass_msun','cohort_canonical_kinetic_energy'}:
        raise ValueError('Different expanding/static trajectories require a mapped physical driver; this comparison scope measures only mass and canonical energy balance')
    for key in ('left','right','physical_model','native_differential_table'):
        reference=request[key]
        if key=='native_differential_table' and reference is None:continue
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:
            raise ValueError('Exact sealed control input reference required: '+key)
        # Workflow references are resolved before input loading, but may remain
        # structured paths/digests during declaration validation.
        for name in ('path','sha256'):
            value=reference[name]
            if isinstance(value,dict) and 'stage' in value:continue
            if not isinstance(value,str) or not value:raise ValueError('Nonempty control input reference required: '+key+'/'+name)
            if name=='sha256' and (len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('Exact lowercase SHA256 required: '+key)
    return request


def control_stage_input_bindings(kind,parameters):
    """Retain original native inputs even when histories are cached stages."""
    if kind=='sidm_control_history':return control_history_input_bindings(parameters)[3]
    if kind!='sidm_control_comparison':raise ValueError('Unknown physical control producer')
    from .sidm_analysis import _load_product
    request=validate_control_comparison_request(parameters);base=parameters['request_base'];sources={}
    for name in ('left','right'):
        _,_,binding=_load_product(request[name],base,request['memory_gib']*1024**3/3);sources.update(binding)
    for name in ('physical_model','native_differential_table'):
        reference=request[name]
        if reference is None:continue
        path=(Path(base)/reference['path']).resolve()
        if file_sha256(path)!=reference['sha256']:raise ValueError('Physical comparison input changed: '+name)
        sources[str(path)]=reference['sha256']
    return sources


def _control_physical_model_binding(report,model,table,base,sources):
    """Bind a common physical law to each engine's actual retained parameters."""
    from .sidm_physics import opengadget3_model_parameters
    params=report['native_parameters'];flags=set(report['native_build_flags'])
    if report['engine']=='gadget4':
        if 'RESEARCH_SIDM_REFERENCE' not in flags:raise ValueError('Measured native control lacks its compiled SIDM operator')
        if params.get('SIDMAngularModelHash')!=canonical_digest(model):raise ValueError('Measured native control evolved another physical angular model')
        convention={'directed':0,'folded_identical':1}.get(model.get('particle_convention','directed'))
        if convention is None or int(params['SIDMReferenceParticleConvention'])!=convention:
            raise ValueError('Actual native particle convention differs from the common physical law')
        if model.get('schema')=='sidm-elastic-model-v2' and float(params['SIDMReferenceCrossSectionNormalizationCm2PerG'])!=1:
            raise ValueError('Native dimensional model normalization does not match the common physical law')
        if model.get('schema')=='sidm-physical-angular-model-v1':
            normalization=model.get('cross_section_normalization_cm2_per_g')
            if normalization is None or not math.isclose(float(params['SIDMReferenceCrossSectionNormalizationCm2PerG']),float(normalization),rel_tol=1e-12,abs_tol=0.):
                raise ValueError('Legacy control must retain its actual dimensional cross section normalization')
        scheme={0:'rare',1:'frequent',2:'hybrid'}[int(params['SIDMReferenceSchemeKind'])]
        if model.get('schema')=='sidm-elastic-model-v2':
            path=Path(params['SIDMNativeTablePath']).resolve()
            if sources.get(str(path))!=params['SIDMNativeTableSHA256']:raise ValueError('Control model table has no exact actual execution binding')
        return scheme
    rare='rSIDM' in flags;frequent='fSIDM' in flags
    if not rare and not frequent:raise ValueError('Independent control lacks a compiled rare/frequent transport operator')
    scheme='hybrid' if rare and frequent else 'rare' if rare else 'frequent'
    retained=None
    if table is not None:
        path=(Path(base)/table['path']).resolve()
        if file_sha256(path)!=table['sha256']:raise ValueError('Common independent differential table changed')
        actual=report.get('invocation_environment',{}).get('RESEARCH_SIDM_REFERENCE_TABLE')
        if actual is not None:
            actual=Path(actual).resolve()
            if sources.get(str(actual))!=table['sha256'] or file_sha256(actual)!=table['sha256']:
                raise ValueError('Actual independent table differs from the common physical interpolant')
            path=actual
        retained={'path':str(path),'sha256':table['sha256']};sources[str(path)]=table['sha256']
    expected=opengadget3_model_parameters(model,velocity_unit_kms=float(params['UnitVelocity_in_cm_per_s'])/1e5,
        split_angle_radians=float(params['MSIDM_thetaC_fwd']) if scheme=='hybrid' else math.pi/4,
        scheme=scheme,retained_differential_table=retained)
    if params.get('MSIDM_CrossSection')!=expected['MSIDM_CrossSection']:raise ValueError('Independent control evolved a different differential-law class')
    actual={}
    for pair in params['MSIDM_DCS_params'].split(';'):
        name,value=pair.split('=',1)
        if name in actual:raise ValueError('Duplicate independent physical-model parameter')
        actual[name]=float(value)
    if set(actual)!=set(expected['cross_section_parameters']) or any(not math.isclose(actual[name],value,rel_tol=1e-12,abs_tol=0.) for name,value in expected['cross_section_parameters'].items()):
        raise ValueError('Independent cross section normalization/velocity parameters differ from the common physical law')
    if not set(expected['required_reference_build_features'])<=flags:raise ValueError('Independent transport configuration differs from the actual comparison convention')
    if not math.isclose(float(params['MSIDM_thetaC_bwd']),expected['MSIDM_thetaC_bwd'],rel_tol=1e-12,abs_tol=0.):
        raise ValueError('Independent backwards angular domain differs from the common particle convention')
    if not math.isclose(float(params['MSIDM_thetaC_fwd']),expected['MSIDM_thetaC_fwd'],rel_tol=1e-12,abs_tol=0.):
        raise ValueError('Independent forward angular domain differs from the actual transport scheme')
    if expected['environment']:
        # An environment-selected table needs an executed environment binding,
        # not merely a file that happens to share the IC's parent directory.
        if report.get('invocation_environment',{}).get('RESEARCH_SIDM_REFERENCE_TABLE')!=expected['environment']['RESEARCH_SIDM_REFERENCE_TABLE']:
            raise ValueError('Independent table-selected run needs a recorded actual invocation environment')
    return scheme


def _expanding_control_background_hash(report,sources):
    """Identity of the actual reference background used by the clock exporter."""
    from .foundation import H100_S
    params=report['native_parameters']
    if not report['comoving'] or report.get('mapped_transport',False):
        raise ValueError('An original expanding native reference history is required')
    table=params.get('CosmologyTableFile','none')
    if str(table).lower() not in {'none',''}:
        path=Path(table).resolve()
        if report['engine']!='gadget4' or sources.get(str(path))!=file_sha256(path):
            raise ValueError('Original expansion table lacks its actual native execution binding')
        return file_sha256(path)
    unit=float(params['UnitLength_in_cm'])/float(params['UnitVelocity_in_cm_per_s']);h=float(params['HubbleParam'])
    h0=float(params['Hubble'])*h/unit if report['engine']=='gadget4' else H100_S*h
    matter=float(params['Omega0']);lambda_=float(params['OmegaLambda']);radiation=float(params.get('OmegaRadiation',0.))
    return canonical_digest({'physical_H0_per_second':h0,'matter':matter,'lambda':lambda_,
        'radiation':radiation,'curvature':1-matter-lambda_-radiation})


def publish_control_comparison(parameters,destination):
    from .sidm_analysis import _load_product
    from .data import export_numeric_audit
    request=validate_control_comparison_request(parameters);base=parameters['request_base'];sources={};memory=request['memory_gib']*1024**3
    if request['left']['sha256']==request['right']['sha256']:raise ValueError('A control product cannot serve as its own independent comparison')
    model_path=(Path(base)/request['physical_model']['path']).resolve()
    if file_sha256(model_path)!=request['physical_model']['sha256'] or model_path.stat().st_size*16>memory:
        raise ValueError('Control comparison physical model changed or exceeds workspace')
    model=json.loads(model_path.read_text());sources[str(model_path)]=file_sha256(model_path)
    reports=[];values=[];schemes=[]
    for key in ('left','right'):
        report,arrays,binding=_load_product(request[key],base,memory/3);sources.update(binding)
        if report.get('schema')!='sidm-physical-control-history-v1' or report.get('execution_status')!='COMPLETE':raise ValueError('Comparison requires actual complete native control measurements')
        schemes.append(_control_physical_model_binding(report,model,request['native_differential_table'],base,sources))
        reports.append(report);values.append(arrays)
    left,right=reports;lv,rv=values
    mapped_scope=request['scope']=='physical_comoving_mapped_distributions'
    epoch_diagnostics={}
    if left['population_physical_state_sha256']!=right['population_physical_state_sha256'] or left['identity_namespace']!=right['identity_namespace']:
        raise ValueError('Paired physical controls must start from the same exact recorded physical population')
    if left['cohort_particle_counts']!=right['cohort_particle_counts'] or min(left['cohort_particle_counts'])<request['minimum_particles_per_cohort']:
        raise ValueError('Control comparisons require identical adequately sampled cohort selections')
    if request['scope']=='independent_engine_trajectories' and {left['engine'],right['engine']}!={'gadget4','opengadget3'}:
        raise ValueError('Independent engine comparison requires both actual native implementations')
    if request['scope']=='same_engine_refinement' and left['engine']!=right['engine']:raise ValueError('Numerical refinement must retain the same engine')
    if request['scope']=='rare_frequent_hybrid_overlap' and len(set(schemes))!=2:raise ValueError('Transport overlap comparison needs distinct actual operators')
    if request['scope']=='physical_comoving_balance' and left['comoving']==right['comoving']:raise ValueError('Physical/comoving balance needs distinct actual clock representations')
    if mapped_scope:
        mapped_indices=[index for index,report in enumerate(reports) if report.get('mapped_transport')is True]
        if len(mapped_indices)!=1:raise ValueError('Mapped distribution comparison requires one mapped and one original expanding native history')
        index=mapped_indices[0];mapped=reports[index];reference=reports[1-index]
        if schemes[0]!=schemes[1]:raise ValueError('Representation equivalence requires the same physical collision scheme')
        if mapped['engine']!='gadget4' or mapped['comoving'] or not mapped.get('expanding_background'):
            raise ValueError('Mapped history does not retain the canonical source/expanding background contract')
        if mapped['original_background_hash']!=_expanding_control_background_hash(reference,sources):
            raise ValueError('Mapped and original native controls use different actual expanding backgrounds')
        if mapped['native_clock']['physical_model_hash']!=canonical_digest(model):
            raise ValueError('Mapped representation changes its clock-bound physical law')
        brackets=np.asarray(values[index]['inverse_epoch_bracket_relative_width'],float)
        if brackets.shape!=values[index]['scale_factor'].shape or not np.isfinite(brackets).all() or np.any(brackets<0) or np.any(brackets>request['maximum_inverse_epoch_bracket_relative_width']):
            raise ValueError('Mapped snapshot inverse uncertainty exceeds its frozen comparison budget')
        epoch_diagnostics['mapped_inverse_epoch_bracket_relative_width']=brackets
    for name in ('speed_bin_edges_kms','velocity_bin_edges_kms'):
        if not np.array_equal(lv[name],rv[name]):raise ValueError('Control histograms change their actual physical measurement bins')
    times_left=lv['elapsed_proper_seconds'];times_right=rv['elapsed_proper_seconds']
    if times_left.shape!=times_right.shape:raise ValueError('Control comparison needs a complete common physical epoch schedule')
    if request['scope']!='physical_comoving_balance':
        allowance=request['time_tolerance_seconds']+lv['clock_quadrature_error_seconds']+rv['clock_quadrature_error_seconds']
        if np.any(np.abs(times_left-times_right)>allowance):raise ValueError('Native control trajectories do not share measured proper epochs')
        if mapped_scope:
            first=np.asarray(lv['scale_factor'],float);last=np.asarray(rv['scale_factor'],float)
            if first.shape!=last.shape or not np.isfinite(first).all() or not np.isfinite(last).all() or np.any(first<=0) or np.any(last<=0):
                raise ValueError('Mapped physical epoch support is incomplete')
            relative=np.abs(last-first)/np.maximum(first,last)
            if np.any(relative>request['maximum_scale_factor_relative_error']):
                raise ValueError('Mapped native histories miss their predeclared physical expansion epochs')
            epoch_diagnostics['scale_factor_relative_difference']=relative
        elif not np.array_equal(lv['scale_factor'],rv['scale_factor']):raise ValueError('Trajectory comparison must match actual physical expansion epochs')
    output={};decisions={};counts=np.asarray(left['cohort_particle_counts'],float)
    path_metrics=set(request['metrics'])&_CONTROL_PATH_METRICS
    if path_metrics:
        left_ids,path_box=_control_particle_path_binding(left,lv);right_ids,right_box=_control_particle_path_binding(right,rv)
        if not np.array_equal(left_ids,right_ids) or path_box!=right_box:
            raise ValueError('Particle path comparisons require identical exact selections and canonical boxes')
        if not np.array_equal(lv['particle_cohort_index'],rv['particle_cohort_index']):
            raise ValueError('Particle path comparison changes original cohort membership')
        first_mass=np.asarray(lv['particle_mass_msun'],float);last_mass=np.asarray(rv['particle_mass_msun'],float)
        mass_tolerance=min(left['request']['relative_mass_tolerance'],right['request']['relative_mass_tolerance'])
        if np.any(np.abs(last_mass-first_mass)>mass_tolerance*np.minimum(first_mass,last_mass)):
            raise ValueError('Particle path comparison changes original selected macro masses')
        output['particle_ids']=left_ids;output['particle_cohort_index']=lv['particle_cohort_index']
        output['particle_canonical_box_kpc']=np.asarray(path_box)
    for name in request['metrics']:
        if name.endswith('_cdf'):
            field=name.replace('_cdf','_bin_counts')
            denominator=counts.reshape((1,len(counts),*([1]*(lv[field].ndim-2))))
            for histogram in (lv[field],rv[field]):
                expected=counts.reshape((1,len(counts),*([1]*(histogram.ndim-3))))
                if np.any(np.sum(histogram,axis=-1)!=expected):raise ValueError('Saved physical histogram omits or duplicates original cohort particles')
            first=np.cumsum(lv[field],axis=-1)/denominator;last=np.cumsum(rv[field],axis=-1)/denominator
        else:first=np.asarray(lv[name],float);last=np.asarray(rv[name],float)
        if first.shape!=last.shape or not np.isfinite(first).all() or not np.isfinite(last).all():raise ValueError('Measured comparison support is incomplete: '+name)
        if request['scope']=='physical_comoving_balance':
            # Compare each trajectory's conservation residual, not an expanding
            # cohort's heat history to a different static physical experiment.
            first=first.sum(axis=1);last=last.sum(axis=1);first=first-first[0];last=last-last[0]
        delta=last-first
        if name=='particle_position_canonical_kpc':
            delta-=path_box*np.rint(delta/path_box)
            # Relative scales use each path's displacement from its initial
            # saved point, rather than arbitrary coordinates relative to zero.
            first=first-first[0];last=last-last[0]
            first-=path_box*np.rint(first/path_box);last-=path_box*np.rint(last/path_box)
        reference=np.maximum(np.maximum(np.abs(first),np.abs(last)),request['relative_error_floor'][name])
        allowed=request['maximum_absolute_error'][name]+request['maximum_relative_error'][name]*reference
        passed=np.abs(delta)<=allowed
        if request['scope']=='physical_comoving_balance':
            # Equal drift in both runs is still a conservation failure. Each
            # balance must independently satisfy the dimensional frozen limit.
            left_scale=np.maximum(np.abs(lv[name].sum(axis=1)[0]),request['relative_error_floor'][name])
            right_scale=np.maximum(np.abs(rv[name].sum(axis=1)[0]),request['relative_error_floor'][name])
            left_limit=request['maximum_absolute_error'][name]+request['maximum_relative_error'][name]*left_scale
            right_limit=request['maximum_absolute_error'][name]+request['maximum_relative_error'][name]*right_scale
            passed &= (np.abs(first)<=left_limit)&(np.abs(last)<=right_limit)
            output[name+'__left_balance']=first;output[name+'__right_balance']=last
        output[name+'__difference']=delta;output[name+'__allowed_error']=allowed;output[name+'__within_limits']=passed
        decisions[name]={'status':'PASS_LIMITS' if passed.all() else 'FAIL_LIMITS','maximum_absolute_difference':float(np.max(np.abs(delta))),
            'maximum_relative_difference':float(np.max(np.abs(delta)/reference))}
    output['left_elapsed_proper_seconds']=times_left;output['right_elapsed_proper_seconds']=times_right
    output.update(epoch_diagnostics)
    report={'schema':'sidm-physical-control-comparison-v1','execution_status':'COMPLETE','request':request,
        'request_base':str(Path(parameters['request_base']).resolve()),
        'actual_engines':[left['engine'],right['engine']],'actual_schemes':schemes,'decisions':decisions,
        'comparison_status':'PASS_DECLARED_LIMITS' if all(row['status']=='PASS_LIMITS' for row in decisions.values()) else 'FAIL_DECLARED_LIMITS',
        'population_physical_state_sha256':left['population_physical_state_sha256'],
        'scope':'source-bound single-realization physical trajectory/bin limits; no iid confidence, independent ensemble agreement or global scientific acceptance',
        'physical_comoving_scope':'mass and total canonical energy balance only; arbitrary expanding/static distribution equivalence needs a mapped transport driver',
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    if path_metrics:
        report['particle_path_comparison']={'schema':'sidm-exact-id-saved-particle-comparison-v1',
            'identity_namespace':left['identity_namespace'],'selected_particle_count':len(left_ids),
            'metrics':sorted(path_metrics),'canonical_box_kpc':path_box,
            'position_difference':'right minus left, minimum-image canonical Cartesian components',
            'position_relative_scale':'minimum-image displacement from each initial saved point; componentwise frozen limits',
            'sampling_scope':'exact-ID comparisons at matched saved epochs; no winding reconstruction or between-output trajectory accuracy',
            'stochastic_scope':'paired empirical particle differences; matching seeds do not imply identical stochastic events or pathwise equivalence'}
    if mapped_scope:
        report.update(physical_comoving_scope='physical cohort moments, kinetic energies, distribution CDFs and explicitly selected saved particle metrics at matched physical epochs; one mapped and one original expanding native execution',
            representation_scope='canonical mapped versus original expanding transport; selected exact-ID saved paths measured' if path_metrics else 'canonical mapped transport versus original expanding transport; particle-by-particle path agreement is not measured',
            independent_native_engines=left['engine']!=right['engine'],
            epoch_uncertainty_scope='clock quadrature and inverse brackets are estimates; explicit epoch budgets do not certify propagated statistic uncertainty',
            original_background_hash=mapped['original_background_hash'])
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    units={name:'dimensionless' for name in output}
    for metric in request['metrics']:
        unit=_control_metric_unit(metric)
        units[metric+'__difference']=unit;units[metric+'__allowed_error']=unit
        if request['scope']=='physical_comoving_balance':
            units[metric+'__left_balance']=unit;units[metric+'__right_balance']=unit
    units.update(left_elapsed_proper_seconds='s',right_elapsed_proper_seconds='s')
    if path_metrics:units.update(particle_ids='identity',particle_canonical_box_kpc='kpc')
    specifications={name:{'units':units[name],'quantity':'measured control comparison '+name} for name in output}
    if path_metrics:specifications['particle_ids']['identity_namespace']=left['identity_namespace']
    return export_numeric_audit(destination,report,output,specifications,
        algorithm=source,context={'epoch':{'canonical_time':'measured native proper epochs'},'coordinates':{'frame':'actual physical peculiar velocity/cohort membership'}},source_hashes=sources)


def control_comparison_runner(parameters,context):
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Control comparison exceeds workflow allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Control comparison escapes the owned generation')
    return publish_control_comparison(parameters,destination)


def control_comparison_main():
    import argparse
    parser=argparse.ArgumentParser(description='Compare actual physical SIDM control trajectories under frozen limits')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text())
    parameters.setdefault('request_base',str(path.parent));publish_control_comparison(parameters,args.output)
    return 0


def validate_control_ensemble_request(parameters):
    request=parameters['request']
    keys={'schema','comparisons','minimum_realizations','maximum_realizations','maximum_observables','memory_gib'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='sidm-paired-control-ensemble-request-v1':
        raise ValueError('Complete paired native control ensemble request required')
    for name in ('minimum_realizations','maximum_realizations','maximum_observables'):
        if type(request[name])is not int or request[name]<1:raise ValueError('Exact ensemble bound required: '+name)
    if request['minimum_realizations']<2 or request['minimum_realizations']>request['maximum_realizations']:
        raise ValueError('At least two bounded actual paired realizations required')
    references=request['comparisons']
    if not isinstance(references,list) or not request['minimum_realizations']<=len(references)<=request['maximum_realizations']:
        raise ValueError('Complete bounded ensemble comparison selection required')
    _positive(request['memory_gib'],'paired ensemble memory')
    if request['maximum_observables']**2*64+request['maximum_observables']*len(references)*64>request['memory_gib']*1024**3:
        raise MemoryError('Declared full joint covariance and realization vectors exceed allocation')
    return request


def control_ensemble_input_bindings(parameters):
    from .sidm_analysis import _load_product
    request=validate_control_ensemble_request(parameters);sources={}
    for reference in request['comparisons']:
        _,_,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
    return sources


def publish_control_ensemble(parameters,destination):
    """Measure paired-realization bias/scatter and full joint covariance.

    No particle/event independence, Gaussian confidence interval or posterior
    coverage is inferred from distinct initial/operator RNG seed axes.
    """
    from .sidm_analysis import _load_product
    from .data import export_numeric_audit
    request=validate_control_ensemble_request(parameters);base=parameters['request_base'];memory=request['memory_gib']*1024**3
    sources={};vectors=[];phase_seeds=set();left_seeds=set();right_seeds=set();physical_states=set();signature=None;layout=None;epoch_arrays=None;decisions=[]
    paired_identities=[];allowed_vectors=[]
    for reference in request['comparisons']:
        report,arrays,binding=_load_product(reference,base,memory/4);sources.update(binding)
        if report.get('schema')!='sidm-physical-control-comparison-v1' or report.get('execution_status')!='COMPLETE':
            raise ValueError('Paired ensemble needs actual complete control comparisons')
        comparison=report['request'];histories=[]
        for side in ('left','right'):
            history,_,binding=_load_product(comparison[side],report.get('request_base',base),memory/4);sources.update(binding)
            if history.get('schema')!='sidm-physical-control-history-v1':raise ValueError('Ensemble lacks its actual history seed/physics provenance')
            histories.append(history)
        left,right=histories
        if left['phase_seed']!=right['phase_seed'] or left['population_physical_state_sha256']!=right['population_physical_state_sha256']:
            raise ValueError('Within-realization comparisons must retain the identical actual phase population')
        phase=left['phase_seed'];state=left['population_physical_state_sha256']
        def operator_seed(value):
            if isinstance(value,bool) or not isinstance(value,(str,int)) or not str(value).isdigit() or not 0<=int(value)<2**64:
                raise ValueError('Ensemble needs actual exact native uint64 operator seed provenance')
            return int(value)
        ls=operator_seed(left['operator_seed']);rs=operator_seed(right['operator_seed'])
        if phase in phase_seeds or ls in left_seeds or rs in right_seeds or state in physical_states:
            raise ValueError('Separate realizations need distinct retained initial states and phase/operator seed axes')
        phase_seeds.add(phase);left_seeds.add(ls);right_seeds.add(rs);physical_states.add(state)
        def fixed_history(history):
            excluded={'InitCondFile','OutputDir','OutputListFilename','SIDMNativeTablePath','SIDMOperatorSeed','MSIDM_SEED'}
            mapped=history.get('mapped_transport')is True
            if mapped:excluded.update({'SIDMCanonicalClockPath','SIDMCanonicalPhysicalStateSHA256'})
            params={key:value for key,value in history['native_parameters'].items() if key not in excluded}
            population_reference=history['request']['population']
            population,_,binding=_load_product(population_reference,history.get('request_base',base),memory/4);sources.update(binding)
            representation=None
            if mapped:
                if population.get('schema')!='sidm-canonical-control-population-v1' or population['original_physical_state_sha256']!=history['population_physical_state_sha256']:
                    raise ValueError('Mapped ensemble lacks its original physical population lineage')
                representation={'request':{key:value for key,value in population['request'].items() if key not in {'population','native_clock'}},
                    'initial_scale_factor':population['initial_scale_factor'],'canonical_box_kpc':population['canonical_box_kpc'],
                    'native_units':population['native_units'],'clock_model':population['native_clock']['physical_model_hash'],
                    'clock_background':population['native_clock']['original_background_hash'],
                    'clock_asset_sha256':population['native_clock']['sha256']}
                original_base=population.get('request_base',history.get('request_base',base))
                # Resolve the original reference relative to the mapped producer,
                # never relative to this later ensemble's generation directory.
                original,_,binding=_load_product(population['original_population'],original_base,memory/4);sources.update(binding)
                if original.get('schema')!='sidm-control-population-v1' or original['physical_state_sha256']!=history['population_physical_state_sha256']:
                    raise ValueError('Mapped ensemble original population identity differs')
                population=original
            population_request={key:value for key,value in population['request'].items() if key not in {'seed','identity_namespace'}}
            return {'engine':history['engine'],'params':params,'flags':history['native_build_flags'],'cohorts':history['cohort_particle_counts'],
                'population_request':population_request,'mapped_representation':representation,
                'particle_selection':None if 'particle_paths' not in history['request'] else sorted(history['request']['particle_paths']['particle_ids'])}
        definition={key:value for key,value in comparison.items() if key not in {'left','right','physical_model','native_differential_table'}}
        definition['physical_model_sha256']=comparison['physical_model']['sha256']
        definition['native_table_sha256']=None if comparison['native_differential_table']is None else comparison['native_differential_table']['sha256']
        current_signature=canonical_digest({'comparison':definition,'histories':[fixed_history(left),fixed_history(right)],
            'actual_engines':report['actual_engines'],'actual_schemes':report['actual_schemes']})
        if signature is not None and signature!=current_signature:raise ValueError('Paired ensemble changes unrelated native physics, sampling or numerical definitions')
        signature=current_signature;current_layout=[];pieces=[];allowed_pieces=[];offset=0
        for metric in comparison['metrics']:
            values=np.asarray(arrays[metric+'__difference'],float)
            if not np.isfinite(values).all():raise ValueError('Ensemble has an unsupported measured coordinate')
            unit=_control_metric_unit(metric)
            allowed=np.asarray(arrays[metric+'__allowed_error'],float)
            if allowed.shape!=values.shape or not np.isfinite(allowed).all() or np.any(allowed<0):
                raise ValueError('Ensemble lacks the complete frozen dimensional coordinate limits')
            current_layout.append({'metric':metric,'units':unit,'shape':list(values.shape),'offset':offset,'size':values.size});offset+=values.size;pieces.append(values.ravel());allowed_pieces.append(allowed.ravel())
            if metric in _CONTROL_PATH_METRICS:
                from .data import exact_uint64_ids
                selected_ids=exact_uint64_ids(arrays['particle_ids'],allow_empty=False)
                if values.shape!=(len(arrays['left_elapsed_proper_seconds']),len(selected_ids),3):
                    raise ValueError('Particle ensemble coordinate shape differs from saved exact-ID paths')
                current_layout[-1].update(particle_ids_decimal=[str(int(value)) for value in selected_ids],
                    particle_cohort_index=np.asarray(arrays['particle_cohort_index']).tolist(),
                    axis_definition='saved epoch, exact sorted particle ID, Cartesian component',
                    position_boundary='minimum-image periodic difference; no winding recovery' if metric=='particle_position_canonical_kpc' else None)
        if offset>request['maximum_observables']:raise MemoryError('Actual paired observable vector exceeds its covariance ceiling')
        if layout is not None and layout!=current_layout:raise ValueError('Ensemble changes physical coordinate layout')
        layout=current_layout
        current_epochs=[arrays['left_elapsed_proper_seconds'],arrays['right_elapsed_proper_seconds']]
        if epoch_arrays is not None and any(not np.array_equal(a,b) for a,b in zip(epoch_arrays,current_epochs)):
            raise ValueError('Ensemble changes its measured physical output epochs')
        epoch_arrays=current_epochs;vectors.append(np.concatenate(pieces));allowed_vectors.append(np.concatenate(allowed_pieces));decisions.append(report['comparison_status'])
        paired_identities.append({'phase_seed':phase,'left_operator_seed':str(ls),'right_operator_seed':str(rs),'population_physical_state_sha256':state})
    values=np.asarray(vectors);mean=values.mean(axis=0);centered=values-mean
    covariance=centered.T@centered/(len(values)-1)
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():raise ValueError('Paired ensemble accumulation overflows')
    output={'realization_differences':values,'mean_difference':mean,'sample_covariance':covariance,
        'standard_error_of_mean':np.sqrt(np.maximum(np.diag(covariance),0.)/len(values)),
        'minimum_declared_allowed_error':np.min(np.asarray(allowed_vectors),axis=0),
        'left_elapsed_proper_seconds':epoch_arrays[0],'right_elapsed_proper_seconds':epoch_arrays[1]}
    report={'schema':'sidm-paired-control-ensemble-v1','execution_status':'COMPLETE','request':request,'fixed_experiment_signature':signature,
        'request_base':str(Path(base).resolve()),'ordered_realization_identities':paired_identities,
        'coordinate_layout':layout,'phase_seeds':sorted(phase_seeds),'left_operator_seeds':[str(value) for value in sorted(left_seeds)],'right_operator_seeds':[str(value) for value in sorted(right_seeds)],
        'realization_count':len(values),'realization_limit_statuses':decisions,
        'statistical_definition':'sample mean and unbiased paired-realization joint covariance; standard error assumes independent realizations; no Gaussian intervals, event independence or global acceptance',
        'science_acceptance':'NOT_ESTABLISHED','source_hashes':sources}
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    units={name:'per-coordinate physical units retained in source comparisons and coordinate_layout' for name in output}
    units.update(sample_covariance='product of coordinate units',left_elapsed_proper_seconds='s',right_elapsed_proper_seconds='s')
    return export_numeric_audit(destination,report,output,{name:{'units':units[name],'quantity':'measured paired control ensemble '+name} for name in output},
        algorithm=source,context={'epoch':{'canonical_time':'actual saved proper output times'},'coordinates':{'frame':'aligned native physical control coordinates'}},source_hashes=sources)


def control_ensemble_runner(parameters,context):
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Paired ensemble exceeds its workflow workspace')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Paired ensemble escapes its owned generation')
    return publish_control_ensemble(parameters,destination)


def control_ensemble_main():
    import argparse
    parser=argparse.ArgumentParser(description='Measure joint covariance and paired-realization differences from actual native SIDM comparisons')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text())
    parameters.setdefault('request_base',str(path.parent));publish_control_ensemble(parameters,args.output)
    return 0


def validate_control_refinement_measurement_request(parameters):
    request=parameters['request']
    keys={'schema','parameter_path','levels','ensembles','maximum_levels','maximum_observables','memory_gib','mean_error_multiplier','maximum_sampling_error_fraction'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='sidm-refinement-measurement-request-v1':
        raise ValueError('Complete source-bound native refinement measurement request required')
    paths={('maximum_timestep_native',),('sidm','kernel_support_proper_kpc'),('sidm','split_angle_radians'),
        ('sidm','maximum_pair_probability'),('sidm','maximum_particle_probability'),('sidm','maximum_transfer_depth'),
        ('sidm','maximum_scale_factor_fractional_change')}
    path=request['parameter_path']
    if not isinstance(path,list) or any(not isinstance(key,str) for key in path) or tuple(path) not in paths:
        raise ValueError('Refinement measurement requires an operative native numerical axis')
    for key in ('maximum_levels','maximum_observables'):
        if type(request[key])is not int or request[key]<1:raise ValueError('Exact positive refinement bound required')
    levels=request['levels'];ensembles=request['ensembles']
    if not isinstance(levels,list) or not 3<=len(levels)<=request['maximum_levels'] or not isinstance(ensembles,list) or len(ensembles)!=len(levels)-1:
        raise ValueError('Every adjacent pair of at least three levels needs its actual paired ensemble')
    for level in levels:_positive(level,'operative native level')
    if any(right>=left for left,right in zip(levels,levels[1:])):raise ValueError('Native refinement levels must decrease strictly')
    _positive(request['memory_gib'],'refinement workspace');_positive(request['mean_error_multiplier'],'declared mean error multiplier',True)
    fraction=_positive(request['maximum_sampling_error_fraction'],'sampling budget fraction')
    if fraction>=1:raise ValueError('Sampling uncertainty must consume less than the complete dimensional budget')
    if request['maximum_levels']*request['maximum_observables']*192>request['memory_gib']*1024**3:
        raise MemoryError('Declared refinement vectors exceed workspace')
    return request


def control_refinement_measurement_input_bindings(parameters):
    from .sidm_analysis import _load_product
    request=validate_control_refinement_measurement_request(parameters);sources={}
    for reference in request['ensembles']:
        _,_,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
    return sources


def publish_control_refinement_measurement(parameters,destination):
    """Measure paired adjacent-level and coarse-to-finest discrepancies.

    Statistical multipliers are declared operational budgets, not confidence
    levels. No extrapolated continuum solution or convergence order is assumed.
    Each saved history proves that precisely the declared native axis changed.
    """
    from .sidm_analysis import _load_product
    from .data import export_numeric_audit
    request=validate_control_refinement_measurement_request(parameters);base=parameters['request_base'];memory=request['memory_gib']*1024**3
    vectors=[];limits=[];sources={};layout=None;identity_order=None;epochs=None;previous_right=None;statuses=[];native_axis=None;realized=[]
    path=tuple(request['parameter_path']);ignored={'InitCondFile','OutputDir','OutputListFilename','SIDMNativeTablePath','CosmologyTableFile','SIDMCanonicalClockPath'}
    for index,reference in enumerate(request['ensembles']):
        ensemble,arrays,binding=_load_product(reference,base,memory/4);sources.update(binding)
        if ensemble.get('schema')!='sidm-paired-control-ensemble-v1' or ensemble.get('execution_status')!='COMPLETE':
            raise ValueError('Refinement requires actual complete paired native ensemble measurements')
        current=np.asarray(arrays['realization_differences'],float);allowed=np.asarray(arrays['minimum_declared_allowed_error'],float)
        identities=ensemble.get('ordered_realization_identities')
        if current.ndim!=2 or current.shape[0]<2 or current.shape[1]>request['maximum_observables'] or allowed.shape!=(current.shape[1],) or not np.isfinite(current).all() or not np.isfinite(allowed).all() or np.any(allowed<0):
            raise ValueError('Complete finite paired refinement coordinates and frozen dimensional limits required')
        if not isinstance(identities,list) or len(identities)!=len(current) or any(row['left_operator_seed']!=row['right_operator_seed'] for row in identities):
            raise ValueError('Refinement must retain the exact operator seed within every paired realization')
        if identity_order is not None and identities!=identity_order:raise ValueError('Adjacent refinement ensembles change realization ordering or actual populations')
        if layout is not None and ensemble['coordinate_layout']!=layout:raise ValueError('Refinement changes its actual observable layout')
        actual_epochs=[arrays['left_elapsed_proper_seconds'],arrays['right_elapsed_proper_seconds']]
        if epochs is not None and any(not np.array_equal(a,b) for a,b in zip(epochs,actual_epochs)):
            raise ValueError('Refinement changes actual measured physical output epochs')
        identity_order=identities;layout=ensemble['coordinate_layout'];epochs=actual_epochs
        comparisons=ensemble['request']['comparisons'];ebase=ensemble.get('request_base',base)
        if len(comparisons)!=len(current):raise ValueError('Ensemble does not retain every original paired comparison')
        current_right=[];realized_pair=[]
        for realization,comparison_ref in enumerate(comparisons):
            comparison,_,binding=_load_product(comparison_ref,ebase,memory/4);sources.update(binding)
            if comparison.get('schema')!='sidm-physical-control-comparison-v1' or comparison['request']['scope']!='same_engine_refinement':
                raise ValueError('Refinement requires measured same-engine native trajectories')
            histories=[];cbase=comparison.get('request_base',base)
            for side in ('left','right'):
                history,_,binding=_load_product(comparison['request'][side],cbase,memory/4);sources.update(binding)
                if history.get('schema')!='sidm-physical-control-history-v1':raise ValueError('Refinement lacks its original native history')
                histories.append(history)
            left,right=histories
            if left['engine']!=right['engine'] or left['native_build_flags']!=right['native_build_flags']:
                raise ValueError('Numerical refinement changes the native engine or compile profile')
            if (left['phase_seed']!=identity_order[realization]['phase_seed'] or right['phase_seed']!=left['phase_seed']
                    or str(left['operator_seed'])!=identity_order[realization]['left_operator_seed']
                    or str(right['operator_seed'])!=identity_order[realization]['right_operator_seed']
                    or left['population_physical_state_sha256']!=identity_order[realization]['population_physical_state_sha256']
                    or right['population_physical_state_sha256']!=left['population_physical_state_sha256']):
                raise ValueError('Paired refinement rows differ from their actual native seed/population provenance')
            axis='MaxSizeTimestep' if path==('maximum_timestep_native',) else {
                'kernel_support_proper_kpc':'SIDMReferenceKernelRadiusProperKpc','split_angle_radians':'SIDMReferenceSplitAngleRadians',
                'maximum_pair_probability':'SIDMReferenceMaximumEventProbability','maximum_particle_probability':'SIDMReferenceMaximumParticleProbability',
                'maximum_transfer_depth':'SIDMReferenceMaximumTransferDepth',
                'maximum_scale_factor_fractional_change':'SIDMReferenceMaximumScaleFactorFractionalChange'}[path[-1]]
            if left['engine']=='opengadget3':
                if path not in {('maximum_timestep_native',),('sidm','split_angle_radians')}:
                    raise ValueError('Independent engine axis is not mapped to an operative native setting')
                if path==('sidm','split_angle_radians'):axis='MSIDM_thetaC_fwd'
            if native_axis is not None and native_axis!=axis:raise ValueError('Refinement family changes native parameter ownership')
            native_axis=axis
            for table_key in ('SIDMNativeTablePath','CosmologyTableFile','SIDMCanonicalClockPath'):
                table_hashes=[]
                for history in histories:
                    value=history['native_parameters'].get(table_key,'none')
                    if str(value).lower() in {'none',''}:table_hashes.append(None);continue
                    table_path=Path(value).resolve();checksum=file_sha256(table_path)
                    if history['source_hashes'].get(str(table_path))!=checksum:
                        raise ValueError('Native refinement table is not bound to the actual history inputs')
                    table_hashes.append(checksum);sources[str(table_path)]=checksum
                if table_hashes[0]!=table_hashes[1]:raise ValueError('Refinement changes the actual physical/expansion table')
            lp={key:value for key,value in left['native_parameters'].items() if key not in ignored}
            rp={key:value for key,value in right['native_parameters'].items() if key not in ignored}
            derived=set()
            if axis=='MSIDM_thetaC_fwd' and lp.get('MSIDM_thetaC_bwd')!=rp.get('MSIDM_thetaC_bwd'):
                # Folded identical-particle hybrid cutoffs move together; this
                # is the same angular axis, not an unrelated numerical change.
                if any(not math.isclose(float(p['MSIDM_thetaC_bwd']),float(p[axis]),rel_tol=1e-12,abs_tol=0.) for p in (lp,rp)):
                    raise ValueError('Independent backwards cutoff changes outside the mapped folded hybrid axis')
                derived={'MSIDM_thetaC_bwd'}
            if set(lp)!=set(rp) or axis not in lp or any(lp[key]!=rp[key] for key in lp if key!=axis and key not in derived):
                raise ValueError('Native refinement changes an unrelated operative parameter')
            for actual,expected in ((lp[axis],request['levels'][index]),(rp[axis],request['levels'][index+1])):
                if not math.isclose(float(actual),expected,rel_tol=1e-12,abs_tol=0.):raise ValueError('Native operative level differs from the declared refinement design')
            expansion_axis=path==('sidm','maximum_scale_factor_fractional_change')
            if expansion_axis and not (left.get('mapped_transport') and right.get('mapped_transport')):
                raise ValueError('Source expansion refinement requires actual mapped native histories')
            if path in {('maximum_timestep_native',),('sidm','maximum_scale_factor_fractional_change')} and (left.get('mapped_transport') or right.get('mapped_transport')):
                if not left.get('mapped_transport') or not right.get('mapped_transport'):raise ValueError('Mapped timestep refinement cannot change representation')
                durations=[]
                for history in histories:
                    reference=history['mapped_run'];run_path=(Path(history['request_base'])/reference['path']).resolve()
                    if run_path.stat().st_size*16>memory/4 or file_sha256(run_path)!=reference['sha256']:
                        raise ValueError('Mapped realized-step run publication changed or exceeds allocation')
                    run=json.loads(run_path.read_text());sources[str(run_path)]=reference['sha256']
                    diagnostics=run.get('source_step_diagnostics',{})
                    if diagnostics.get('schema')!='sidm-canonical-source-step-diagnostics-v1' or diagnostics.get('execution_status')!='COMPLETE':
                        raise ValueError('Mapped refinement requires complete realized source intervals')
                    duration=_positive(diagnostics['maximum_native_duration'],'realized native maximum interval')
                    if run['physical_state_sha256']!=history['population_physical_state_sha256']:
                        raise ValueError('Realized timestep diagnostics change original physical population')
                    if duration>float(history['native_parameters']['MaxSizeTimestep'])+diagnostics['time_tolerance_native']:
                        raise ValueError('Realized native interval exceeds its declared numerical limit')
                    if expansion_axis:
                        records=diagnostics.get('intervals',diagnostics.get('records',[]))
                        if not records or any(row.get('source_expansion_policy')!='adaptive_substep_span_with_inverse_brackets' for row in records):
                            raise ValueError('Expansion refinement requires actual adaptive source-span diagnostics')
                        span=max(_positive(row['canonical_largest_substep_scale_fraction'],'realized source expansion span',True) for row in records)
                        if span>float(history['native_parameters'][axis])*(1+32*np.finfo(float).eps):raise ValueError('Realized source span exceeds its declared limit')
                        durations.append(span)
                    else:durations.append(duration)
                realized_pair.append(durations)
            lref=comparison['request']['left'];rref=comparison['request']['right']
            def absolute_ref(ref):return {'path':str((Path(cbase)/ref['path']).resolve()),'sha256':ref['sha256']}
            if previous_right is not None and absolute_ref(lref)!=previous_right[realization]:
                raise ValueError('Adjacent pairs do not share the exact measured intermediate history')
            current_right.append(absolute_ref(rref))
        if realized_pair:
            if len(realized_pair)!=len(current):raise ValueError('Realized mapped timestep coverage is incomplete across paired realizations')
            realized.append(realized_pair)
        previous_right=current_right;vectors.append(current);limits.append(allowed);statuses.append(ensemble['realization_limit_statuses'])
        if sum(value.nbytes for value in vectors)*8>memory:raise MemoryError('Retained refinement contrasts exceed allocation')
    values=np.asarray(vectors);mean=values.mean(axis=1);error=values.std(axis=1,ddof=1)/math.sqrt(values.shape[1]);allowed=np.asarray(limits)
    upper=np.abs(mean)+request['mean_error_multiplier']*error
    sampling=error<=request['maximum_sampling_error_fraction']*allowed
    measured=(upper<=allowed)&sampling
    # Pairing preserves cross-level stochastic covariance. Summing paired
    # per-realization contrasts telescopes exactly through the shared histories.
    cumulative=np.sum(values,axis=0);total_mean=cumulative.mean(axis=0);total_error=cumulative.std(axis=0,ddof=1)/math.sqrt(len(cumulative))
    output={'adjacent_mean_difference':mean,'adjacent_standard_error':error,'declared_allowed_error':allowed,
        'adjacent_operational_upper_discrepancy':upper,'adjacent_sampling_resolved':sampling,'adjacent_within_budget':measured,
        'finest_pair_within_budget':measured[-1],'all_adjacent_pairs_within_budget':np.all(measured,axis=0),
        'coarse_to_finest_mean_difference':total_mean,'coarse_to_finest_standard_error':total_error,
        'levels':np.asarray(request['levels']),'elapsed_proper_seconds':epochs[0]}
    operative=True
    if realized:
        if len(realized)!=len(vectors):raise ValueError('Realized mapped timestep coverage is incomplete across levels')
        actual=np.asarray(realized,float);reduced=actual[:,:,1]<actual[:,:,0]
        expansion_axis=path==('sidm','maximum_scale_factor_fractional_change')
        output['realized_maximum_source_scale_spans' if expansion_axis else 'realized_maximum_native_intervals']=actual
        output['realized_source_span_reduced' if expansion_axis else 'realized_interval_reduced']=reduced
        operative=bool(reduced.all())
    if any(not np.isfinite(value).all() for value in output.values()):raise ValueError('Refinement measurement accumulation overflows')
    report={'schema':'measured-sidm-homogeneous-refinement-v1','execution_status':'COMPLETE','request':request,
        'request_base':str(Path(base).resolve()),'native_parameter':native_axis,'coordinate_layout':layout,
        'ordered_realization_identities':identity_order,'individual_limit_statuses':statuses,
        'status':'ALL_ADJACENT_MEAN_BUDGETS_SUPPORTED' if bool(measured.all()) and operative else 'UNRESOLVED_OR_EXCEEDS_BUDGET',
        'realized_timestep_operative':operative if realized and path==('maximum_timestep_native',) else None,
        'realized_source_span_operative':operative if realized and path==('sidm','maximum_scale_factor_fractional_change') else None,
        'scope':'measured finite-level ensemble-mean differences; no continuum extrapolation, assumed convergence order, confidence coverage or global acceptance',
        'science_acceptance':'NOT_ESTABLISHED','source_hashes':sources}
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    return export_numeric_audit(destination,report,output,{name:{'units':'s' if name=='elapsed_proper_seconds' else 'dimensionless' if name.endswith(('resolved','budget','reduced')) or name=='realized_maximum_source_scale_spans' or name=='levels' and path==('sidm','maximum_scale_factor_fractional_change') else 'native operative parameter units' if name in {'levels','realized_maximum_native_intervals'} else 'per-coordinate physical units in coordinate_layout',
        'quantity':'measured native refinement '+name} for name in output},algorithm=source,
        context={'epoch':{'canonical_time':'actual measured native proper epochs'},'coordinates':{'frame':'paired physical control observable coordinates'}},source_hashes=sources)


def control_refinement_measurement_runner(parameters,context):
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Refinement measurement exceeds workflow allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Refinement measurement escapes its owned generation')
    return publish_control_refinement_measurement(parameters,destination)


def control_refinement_measurement_main():
    import argparse
    parser=argparse.ArgumentParser(description='Measure actual native multi-level paired SIDM refinement histories')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.parameters.resolve();parameters=json.loads(source.read_text())
    parameters.setdefault('request_base',str(source.parent));publish_control_refinement_measurement(parameters,args.output)
    return 0


def validate_control_expectation_request(parameters):
    """Freeze physical expectations separately from measured native histories."""
    request=parameters['request']
    keys={'schema','history','physical_model','native_differential_table','experiments',
        'maximum_particles','maximum_pair_visits','pair_chunk_rows','memory_gib',
        'maximum_thin_optical_depth','relaxation_fraction','minimum_consecutive_outputs','limits'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='sidm-control-expectation-request-v1':
        raise ValueError('Complete physical expectation and relaxation declaration required')
    experiments=request['experiments']
    if not isinstance(experiments,list) or not experiments or len(set(experiments))!=len(experiments) or set(experiments)-{'beam_rate','thermal_stationarity','relaxation'}:
        raise ValueError('Select distinct beam-rate, thermal-stationarity or relaxation measurements')
    for key in ('maximum_particles','maximum_pair_visits','pair_chunk_rows','minimum_consecutive_outputs'):
        if type(request[key])is not int or request[key]<1:raise ValueError('Positive exact expectation bound required: '+key)
    _positive(request['memory_gib'],'expectation memory')
    for key in ('maximum_thin_optical_depth','relaxation_fraction'):
        _positive(request[key],key)
        if request[key]>=1:raise ValueError('Fractional physical expectation domain must be below one: '+key)
    required={'mass_relative','momentum_absolute_msun_kms','energy_relative',
        'mean_velocity_absolute_kms','covariance_absolute_kms2','cdf_absolute','rare_exposure_relative','participation_floor'}
    if not isinstance(request['limits'],dict) or set(request['limits'])!=required:
        raise ValueError('Freeze every dimensional physical expectation limit')
    for key,value in request['limits'].items():_positive(value,key,key!='participation_floor')
    for key in ('history','physical_model','native_differential_table'):
        ref=request[key]
        if key=='native_differential_table' and ref is None:continue
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('Exact expectation source reference required: '+key)
        for member in ('path','sha256'):
            value=ref[member]
            if isinstance(value,dict) and 'stage' in value:continue
            if not isinstance(value,str) or not value:raise ValueError('Expectation input reference is empty')
            if member=='sha256' and (len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('Expectation input requires lowercase SHA256')
    return request


def _control_expectation_inputs(parameters):
    from .sidm_analysis import _load_product
    from .sidm_physics import ElasticSIDMModel, read_native_differential_model
    request=validate_control_expectation_request(parameters);base=Path(parameters['request_base']).resolve()
    memory=request['memory_gib']*1024**3
    history,arrays,sources=_load_product(request['history'],base,memory/4)
    if history.get('schema')!='sidm-physical-control-history-v1' or history.get('execution_status')!='COMPLETE':
        raise ValueError('Expectations require an actual original native physical-control history')
    population,initial,binding=_load_product(history['request']['population'],history['request_base'],memory/4)
    sources.update(binding)
    if population.get('schema')!='sidm-control-population-v1' or population['physical_state_sha256']!=history['population_physical_state_sha256'] or population['event_identity_namespace']!=history['identity_namespace']:
        raise ValueError('Expectations do not identify the evolved physical population')
    count=len(initial['particle_ids'])
    if count>request['maximum_particles'] or count*512+request['pair_chunk_rows']*2048>memory/4:
        raise MemoryError('Physical expectation population/pair workspace exceeds its declared allocation')
    path=(base/request['physical_model']['path']).resolve()
    if file_sha256(path)!=request['physical_model']['sha256'] or path.stat().st_size*32>memory/4:
        raise ValueError('Physical expectation model changed or exceeds metadata allocation')
    model=json.loads(path.read_text());sources[str(path)]=request['physical_model']['sha256']
    scheme=_control_physical_model_binding(history,model,request['native_differential_table'],base,sources)
    if model.get('schema')!='sidm-elastic-model-v2':
        raise ValueError('Physical expectations require an explicit dimensional v2 elastic model')
    physical=ElasticSIDMModel(model)
    # Use the exact evolved interpolant whenever the native route used a table.
    if history['engine']=='gadget4':
        table_path=Path(history['native_parameters']['SIDMNativeTablePath']).resolve()
        table_sha=history['native_parameters']['SIDMNativeTableSHA256']
    else:
        selected=history.get('invocation_environment',{}).get('RESEARCH_SIDM_REFERENCE_TABLE')
        table_path=Path(selected).resolve() if selected else None
        table_sha=sources.get(str(table_path)) if selected else None
    if table_path is not None:
        if sources.get(str(table_path))!=table_sha:raise ValueError('Expectation table lacks actual native source binding')
        declared=request['native_differential_table']
        if declared is not None:
            declared_path=(base/declared['path']).resolve()
            if declared['sha256']!=table_sha or file_sha256(declared_path)!=table_sha:
                raise ValueError('Declared expectation table differs from the actual native interpolant')
            sources[str(declared_path)]=table_sha
        physical=read_native_differential_model(table_path,model_sha256=canonical_digest(model),table_sha256=table_sha,
            particle_convention=model.get('particle_convention','directed'),memory_bytes=int(memory/4))
    return request,history,arrays,population,initial,physical,scheme,sources


def control_expectation_input_bindings(parameters):
    return _control_expectation_inputs(parameters)[-1]


def _saved_relaxation_bracket(times,values,fraction,persistence):
    """A first persistent saved crossing, with no interpolated e-folding fit."""
    if not np.isfinite(values).all() or values[0]<=0:
        return {'status':'UNSUPPORTED','reason':'no positive finite initial relaxation amplitude'}
    below=values<=fraction*values[0]
    for index in range(1,len(times)-persistence+1):
        if np.all(below[index:index+persistence]):
            return {'status':'SAVED_CROSSING','lower_seconds':float(times[index-1]),'upper_seconds':float(times[index]),
                'threshold':float(fraction*values[0]),'consecutive_outputs':persistence,
                'scope':'first persistent crossing sampled by this saved schedule; excursions between outputs are unresolved'}
    return {'status':'NOT_BRACKETED','last_elapsed_seconds':float(times[-1]),'threshold':float(fraction*values[0])}


def publish_control_expectation(parameters,destination):
    """Measure conservation, Maxwell stationarity, beam rates and relaxation.

    Thin-beam predictions use the actual initial finite velocity population in
    a homogeneous continuum. Native spatial sampling and scattered trajectories
    remain measured effects, not silently included in this reference model.
    """
    from scipy.special import ndtr
    from scipy.stats import ncx2
    from .data import export_numeric_audit
    request,history,measured,population,initial,physical,scheme,sources=_control_expectation_inputs(parameters)
    times=np.asarray(measured['elapsed_proper_seconds'],float);scale=np.asarray(measured['scale_factor'],float)
    mass=np.asarray(measured['cohort_mass_msun'],float);mean=np.asarray(measured['cohort_mean_velocity_kms'],float)
    covariance=np.asarray(measured['cohort_covariance_kms2'],float);energy=np.asarray(measured['cohort_canonical_kinetic_energy'],float)
    counts=np.asarray(history['cohort_particle_counts'],float);epochs,cohorts=mass.shape
    if epochs<2 or request['minimum_consecutive_outputs']>epochs-1 or np.any(np.diff(times)<=0) or times[0]!=0 or np.any(scale<=0):
        raise ValueError('Ordered original initial epoch and a bounded saved persistence schedule required')
    if mean.shape!=(epochs,cohorts,3) or covariance.shape!=(epochs,cohorts,3,3) or energy.shape!=mass.shape or counts.shape!=(cohorts,) or np.any(counts<2):
        raise ValueError('Complete aligned measured physical cohort histories required')
    expansion=scale/scale[0];canonical_mean=mean*expansion[:,None,None]
    canonical_covariance=covariance*expansion[:,None,None,None]**2
    total_mass=mass.sum(axis=1);momentum=np.sum(mass[:,:,None]*canonical_mean,axis=1);total_energy=energy.sum(axis=1)
    if total_energy[0]<=0:raise ValueError('Control needs a positive initial kinetic-energy scale')
    output={'elapsed_proper_seconds':times,'scale_factor':scale,
        'total_mass_relative_drift':total_mass/total_mass[0]-1,
        'total_canonical_momentum_delta_msun_kms':momentum-momentum[0],
        'total_canonical_energy_relative_drift':total_energy/total_energy[0]-1,
        'cohort_canonical_mean_delta_kms':canonical_mean-canonical_mean[0],
        'cohort_canonical_covariance_delta_kms2':canonical_covariance-canonical_covariance[0]}
    limits=request['limits'];decisions={
        'mass':bool(np.max(np.abs(output['total_mass_relative_drift']))<=limits['mass_relative']),
        'momentum':bool(np.max(np.linalg.norm(output['total_canonical_momentum_delta_msun_kms'],axis=1))<=limits['momentum_absolute_msun_kms']),
        'energy':bool(np.max(np.abs(output['total_canonical_energy_relative_drift']))<=limits['energy_relative'])}
    relaxation={};references={}
    if 'relaxation' in request['experiments']:
        pair_indices=[(i,j) for i in range(cohorts) for j in range(i+1,cohorts)]
        amplitude=np.asarray([np.linalg.norm(canonical_mean[:,i]-canonical_mean[:,j],axis=1) for i,j in pair_indices]).T if pair_indices else np.empty((epochs,0))
        variance=np.trace(canonical_covariance,axis1=2,axis2=3)/3
        anisotropy=np.linalg.norm(canonical_covariance-variance[:,:,None,None]*np.eye(3),axis=(2,3))
        output.update(cohort_pair_indices=np.asarray(pair_indices,dtype=np.int64).reshape(-1,2),
            cohort_pair_bulk_separation_kms=amplitude,cohort_covariance_anisotropy_kms2=anisotropy,
            cohort_variance_kms2=variance)
        for index,(i,j) in enumerate(pair_indices):
            relaxation['bulk_'+str(i)+'_'+str(j)]=_saved_relaxation_bracket(times,amplitude[:,index],request['relaxation_fraction'],request['minimum_consecutive_outputs'])
            difference=np.abs(variance[:,i]-variance[:,j])
            relaxation['variance_'+str(i)+'_'+str(j)]=_saved_relaxation_bracket(times,difference,request['relaxation_fraction'],request['minimum_consecutive_outputs'])
        for i in range(cohorts):
            relaxation['anisotropy_'+str(i)]=_saved_relaxation_bracket(times,anisotropy[:,i],request['relaxation_fraction'],request['minimum_consecutive_outputs'])
    if 'thermal_stationarity' in request['experiments']:
        components=population['request']['components'];first=components[0]
        dispersion=np.asarray(first['dispersion_1d_kms'],float);bulk=np.asarray(first['bulk_velocity_kms'],float)
        if dispersion.shape!=(3,) or np.any(dispersion<=0) or not np.all(dispersion==dispersion[0]):
            raise ValueError('Thermal stationarity requires an isotropic positive Maxwell population')
        for row in components:
            if row['dispersion_1d_kms']!=first['dispersion_1d_kms'] or row['bulk_velocity_kms']!=first['bulk_velocity_kms'] or row['remove_sample_bulk']!=first['remove_sample_bulk']:
                raise ValueError('Different cohort temperatures/bulk constraints describe relaxation rather than equilibrium')
        sigma=float(dispersion[0]);speed_edges=np.asarray(measured['speed_bin_edges_kms'],float)
        velocity_edges=np.asarray(measured['velocity_bin_edges_kms'],float)
        speed_expected=[];velocity_expected=[]
        for factor in expansion:
            speed_expected.append(np.where(speed_edges<0,0.,ncx2.cdf((np.maximum(speed_edges,0)*factor/sigma)**2,3,float(np.dot(bulk,bulk)/sigma**2))))
            velocity_expected.append(ndtr((velocity_edges[None,:]*factor-bulk[:,None])/sigma))
        speed_expected=np.asarray(speed_expected);velocity_expected=np.asarray(velocity_expected)
        observed_speed=np.cumsum(measured['cohort_speed_bin_counts'],axis=-1)[...,:-1]/counts[None,:,None]
        observed_velocity=np.cumsum(measured['cohort_velocity_bin_counts'],axis=-1)[...,:-1]/counts[None,:,None,None]
        speed_residual=observed_speed-speed_expected[:,None,:];velocity_residual=observed_velocity-velocity_expected[:,None,:,:]
        output.update(maxwell_speed_cdf=speed_expected,maxwell_velocity_cdf=velocity_expected,
            cohort_maxwell_speed_cdf_residual=speed_residual,cohort_maxwell_velocity_cdf_residual=velocity_residual,
            speed_bin_edges_kms=speed_edges,velocity_bin_edges_kms=velocity_edges)
        decisions.update(thermal_mean=bool(np.max(np.abs(output['cohort_canonical_mean_delta_kms']))<=limits['mean_velocity_absolute_kms']),
            thermal_covariance=bool(np.max(np.abs(output['cohort_canonical_covariance_delta_kms2']))<=limits['covariance_absolute_kms2']),
            thermal_speed_cdf=bool(np.max(np.abs(speed_residual))<=limits['cdf_absolute']),
            thermal_velocity_cdf=bool(np.max(np.abs(velocity_residual))<=limits['cdf_absolute']))
        references['thermal']='Unconditioned continuum Maxwell law with declared bulk and dispersion. Conditioned finite-cohort means and energy constraints are retained sampling effects; empirical limits are not iid confidence levels.'
    if 'beam_rate' in request['experiments']:
        if history['comoving']:raise ValueError('Frozen physical beam rates require a nonexpanding native clock; mapped expanding rate integration is separate scope')
        velocity=np.asarray(initial['velocity_peculiar_kms'],float);labels=np.asarray(initial['cohort_index'],int);n=len(velocity)
        if velocity.shape!=(n,3) or labels.shape!=(n,) or np.any((labels<0)|(labels>=cohorts)) or not np.array_equal(np.bincount(labels,minlength=cohorts),counts):
            raise ValueError('Actual initial finite-population beam labels and velocities required')
        visits=n*(n-1)//2
        if visits>request['maximum_pair_visits']:raise MemoryError('Exact finite velocity-pair beam rate exceeds the declared pair visit ceiling')
        macro=float(population['request']['macro_mass_msun']);volume=(float(population['request']['box_proper_kpc'])*KPC_CM)**3
        rates=np.zeros(cohorts);factor=macro*MSUN_G/volume*1e5
        for i in range(n-1):
            for begin in range(i+1,n,request['pair_chunk_rows']):
                end=min(n,begin+request['pair_chunk_rows']);speed=np.linalg.norm(velocity[begin:end]-velocity[i],axis=1)
                values=np.zeros_like(speed);moving=speed>0
                # Zero relative speed has zero collision rate even when the
                # dimensional table declares an error policy below its grid.
                values[moving]=np.asarray(physical.total_cross_section_per_mass(speed[moving]),float)*speed[moving]*factor
                if not np.isfinite(values).all() or np.any(values<0):raise ValueError('Finite nonnegative covered microscopic beam rates required')
                rates[labels[i]]+=values.sum();rates+=np.bincount(labels[begin:end],weights=values,minlength=cohorts)
        predicted=times[:,None]*rates[None,:];thin=np.max(predicted/counts[None,:],axis=1)<=request['maximum_thin_optical_depth']
        available=np.asarray(measured['sidm_diagnostics_available'],bool)
        exposure=np.asarray(measured['cohort_microscopic_exposure'],float)
        supported=thin&available
        measured_thin=np.zeros(epochs,dtype=bool)
        for i in np.flatnonzero(available):
            measured_thin[i]=bool(np.max((exposure[i]-exposure[0])/counts)<=request['maximum_thin_optical_depth'])
        supported&=measured_thin
        rare_supported=supported&(scheme=='rare')
        residual=np.full_like(predicted,np.nan);exposure_residual=np.full_like(predicted,np.nan)
        for i in np.flatnonzero(supported):
            exposure_residual[i]=(exposure[i]-exposure[0])-predicted[i]
        for i in np.flatnonzero(rare_supported):
            exact=history['exact_rare_event_participations_decimal']
            delta=[int(exact[i][j])-int(exact[0][j]) for j in range(cohorts)]
            if any(value<0 for value in delta):raise ValueError('Native rare-participation history decreases')
            if any(value>2**53 for value in delta):
                rare_supported[i]=False
                continue
            residual[i]=np.asarray(delta,float)-(exposure[i]-exposure[0])
        denominator=np.maximum(predicted,limits['participation_floor'])
        output.update(initial_cohort_participation_rate_per_second=rates,thin_beam_expected_participations=predicted,
            thin_beam_supported=supported,rare_participation_supported=rare_supported,rare_participation_minus_measured_exposure=residual,
            measured_exposure_minus_thin_beam_prediction=exposure_residual,
            beam_exposure_relative_residual=exposure_residual/denominator)
        decisions['beam_rate']=bool(np.all(np.abs((exposure_residual/denominator)[supported])<=limits['rare_exposure_relative'])) if np.any(supported[1:]) else None
        if scheme=='rare':
            decisions['rare_participation_exposure']=bool(np.all(np.abs((residual/np.maximum(exposure-exposure[0],limits['participation_floor']))[rare_supported])<=limits['rare_exposure_relative'])) if np.any(rare_supported[1:]) else None
        references['beam']='Exact initial unordered finite velocity pairs, both endpoints counted, homogeneous spatial continuum density; no self pairs. Linear-in-time frozen-population reference limited by declared thin optical depth. Frequent/hybrid microscopic exposure is not a rare-event expectation.'
    for name,array in output.items():
        if name in {'rare_participation_minus_measured_exposure','measured_exposure_minus_thin_beam_prediction','beam_exposure_relative_residual'}:continue
        if not np.isfinite(array).all():raise ValueError('Physical expectation accumulation is nonfinite: '+name)
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={'schema':'sidm-control-expectation-v1','execution_status':'COMPLETE','request':request,
        'request_base':str(Path(parameters['request_base']).resolve()),'engine':history['engine'],'scheme':scheme,
        'population_physical_state_sha256':history['population_physical_state_sha256'],
        'cohorts':history['cohorts'],'empirical_limit_decisions':decisions,
        'control_status':'FAIL' if any(value is False for value in decisions.values()) else 'UNRESOLVED' if any(value is None for value in decisions.values()) else 'PASS',
        'saved_relaxation_brackets':relaxation,
        'reference_definitions':references,'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED',
        'scope':'Measured single-realization homogeneous controls; unsupported diagnostics remain masked. No independent-event inference, universal exponential relaxation, mapped clock equivalence or global acceptance.'}
    units={name:'dimensionless' for name in output}
    units.update(elapsed_proper_seconds='s',total_canonical_momentum_delta_msun_kms='Msun km / s',
        cohort_canonical_mean_delta_kms='km / s',cohort_canonical_covariance_delta_kms2='(km / s)^2',
        cohort_pair_bulk_separation_kms='km / s',cohort_covariance_anisotropy_kms2='(km / s)^2',cohort_variance_kms2='(km / s)^2',
        initial_cohort_participation_rate_per_second='1 / s',speed_bin_edges_kms='km / s',velocity_bin_edges_kms='km / s')
    return export_numeric_audit(destination,report,output,{name:{'units':units[name],'quantity':'physical control expectation '+name} for name in output},
        algorithm=source,context={'epoch':{'canonical_time':'actual saved elapsed proper seconds'},
        'coordinates':{'frame':'inertial peculiar velocities; expansion-corrected moments use a/a_initial'}},source_hashes=sources)


def control_expectation_runner(parameters,context):
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Physical expectations exceed workflow allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Physical expectation publication escapes its generation')
    return publish_control_expectation(parameters,destination)


def control_expectation_main():
    import argparse
    parser=argparse.ArgumentParser(description='Measure physical beam expectations, thermal stationarity and saved relaxation from original native SIDM controls')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text())
    parameters.setdefault('request_base',str(path.parent));publish_control_expectation(parameters,args.output)
    return 0


def _conditional_angular_cdf(physical, retained, speed, points, *, folded, split):
    """Exact analytic or piecewise-quadratic CDF on the native rare domain.

    The table route integrates the actual log-speed/linear-cosine interpolant.
    Scaling by the largest density avoids spurious zero normalization for
    small dimensional cross sections. This routine never draws new events.
    """
    points=np.asarray(points,float);lo=0. if folded else -1.;hi=math.cos(split)
    if not np.isfinite(points).all() or not lo<hi<=1 or not math.isfinite(speed) or speed<0:
        raise ValueError('Finite speed and a nonempty actual rare-angle domain required')
    x=np.clip(points,lo,hi)
    if retained is not None:
        base=retained.cosines;row=retained._table_row(speed)
        full_knots=np.unique(np.r_[lo,1.,np.abs(base) if folded else base])
        full_knots=full_knots[(full_knots>=lo)&(full_knots<=1)]
        density=np.interp(full_knots,base,row)
        if folded:density+=np.interp(-full_knots,base,row)
        scale=float(density.max())
        if not math.isfinite(scale) or scale<=0:raise ValueError('Recorded native event has no positive differential support at its actual speed')
        density=density/scale
        full_mass=float(np.sum(np.diff(full_knots)*(density[:-1]+density[1:])/2))
        knots=np.unique(np.r_[full_knots[full_knots<hi],hi]);width=np.diff(knots)
        y=np.interp(knots,full_knots,density).astype(np.longdouble)
        masses=width.astype(np.longdouble)*(y[:-1]+y[1:])/2
        cumulative=np.r_[np.longdouble(0),np.cumsum(masses,dtype=np.longdouble)]
        normalization=cumulative[-1]
        if not np.isfinite(normalization) or normalization<=0 or not math.isfinite(full_mass) or full_mass<=0:
            raise ValueError('Actual speed/split leaves no finite positive native rare-angle probability')
        cell=np.clip(np.searchsorted(knots,x,side='right')-1,0,len(width)-1)
        distance=(x-knots[cell]).astype(np.longdouble)
        slope=(y[1:]-y[:-1])/width
        value=(cumulative[cell]+y[cell]*distance+slope[cell]*distance**2/2)/normalization
        result=np.asarray(value,float);result[points<=lo]=0.;result[points>=hi]=1.
        probability=float(normalization/full_mass)
        derivative=float(y.max()/normalization)
        # Conservative accumulation/evaluation allowance relative to the
        # retained angular support. Native sampling/model errors are separate.
        error=(64*len(knots)+512)*np.finfo(float).eps/probability
        operations=len(full_knots)+len(knots)+len(points)
    else:
        kind=physical['kind'];a=0. if kind=='isotropic' else .5*(speed/physical['velocity_scale_kms'])**2
        if not math.isfinite(a) or not math.isfinite(a*(1+2*a)):
            raise ValueError('Event anisotropy is outside the finite analytic reference domain')
        if kind=='born_moller':
            if not folded:raise ValueError('Born-Moller events require the identical-particle folded convention')
            from .sidm_physics import moller_primitive as primitive
            total=float(primitive(1.,a));normalization=float(primitive(hi,a))
            if normalization<=0 or not math.isfinite(normalization):raise ValueError('Actual folded Moller domain has no support')
            result=np.asarray(primitive(x,a)/normalization,float);probability=normalization/total
            plus=1+a*(1-hi);center=1+a
            derivative=float((1/plus**2+1/center**2+1/(2*plus*center))/normalization)
        elif folded:
            def primitive(u):return u*(1+2*a)/((1+a*(1-u))*(1+a*(1+u)))
            probability=float(primitive(hi));result=primitive(x)/probability
            derivative=float((1+2*a)/(1+a*(1-hi))**2/probability)
        else:
            def primitive(u):return (1+u)/(2*(1+a*(1-u)))
            probability=float(primitive(hi));result=primitive(x)/probability
            derivative=float((1+2*a)/(2*(1+a*(1-hi))**2*probability))
        error=512*np.finfo(float).eps/probability;operations=len(points)
    if not np.isfinite(result).all() or not math.isfinite(error) or not math.isfinite(derivative) or probability<=0:
        raise ValueError('Conditional angular calculation lost finite normalized support')
    allowance=max(error,512*np.finfo(float).eps)
    if np.any((result < -allowance)|(result > 1+allowance)):
        raise ValueError('Conditional angular CDF left its normalized domain')
    return np.clip(result,0.,1.),error,derivative,probability,operations


def _empirical_uniform_discrepancy(probability):
    ordered=np.sort(np.asarray(probability,float));n=len(ordered)
    if n==0:return None
    return float(max(np.max(np.arange(1,n+1)/n-ordered),np.max(ordered-np.arange(n)/n)))


def _measure_speed_conditioned_angles(request, events, values, reference, sources, destination):
    from .sidm_physics import read_native_differential_model
    angle=np.asarray(values['scattering_cosine'],float);speed=np.asarray(values['relative_speed_before_kms'],float)
    identity=np.asarray(values['native_event_id']);physical=reference['physical_model'];retained=None
    if identity.dtype!=np.dtype('uint64'):raise ValueError('Conditional diagnostics preserve exact uint64 native event identities')
    if speed.shape!=angle.shape or not np.isfinite(speed).all() or np.any(speed<=0):
        raise ValueError('Conditional diagnostics require each event\'s actual positive pre-collision relative speed')
    if len(angle)*1024>request['memory_gib']*1024**3:raise MemoryError('Conditional event arrays exceed their declared workspace')
    if physical.get('schema')=='sidm-elastic-model-v2':
        native=reference['native_table'];path=Path(native['path']).resolve()
        retained=read_native_differential_model(path,model_sha256=request['angular_model_hash'],table_sha256=native['sha256'],
            particle_convention='folded_identical' if request['cosine_convention']=='identical_particle_folded_absolute_cosine' else 'directed',
            memory_bytes=int(request['memory_gib']*1024**3/4))
        sources[str(path)]=native['sha256']
    folded=request['cosine_convention']=='identical_particle_folded_absolute_cosine';split=request.get('minimum_scattering_angle_radians',0.)
    hi=math.cos(split);lo=0. if folded else -1.;roundoff=request['cosine_roundoff_tolerance']
    if np.any(angle>hi+roundoff):raise ValueError('Recorded native angle is outside its actual hybrid rare domain')
    edges=np.asarray(request['cosine_bin_edges'],float);speed_edges=np.asarray(request['speed_bin_edges_kms'],float)
    if np.any(speed<speed_edges[0]) or np.any(speed>speed_edges[-1]):raise ValueError('Actual event speeds extend beyond the declared diagnostic bins')
    ns=len(speed_edges)-1;nb=len(edges)-1
    speed_bin=np.searchsorted(speed_edges,speed,side='right')-1;speed_bin[speed==speed_edges[-1]]=ns-1
    transform=np.empty(len(angle));error=np.empty(len(angle));rare_probability=np.empty(len(angle));evaluations=0
    expected_counts=np.zeros((ns,nb));unique,group=np.unique(speed,return_inverse=True)
    # Exact-speed grouping reduces repeated work without binning or replacing
    # any recorded speed with a nominal/averaged speed.
    order=np.argsort(group,kind='stable');counts=np.bincount(group,minlength=len(unique));offset=0
    for j,current_speed in enumerate(unique):
        rows=order[offset:offset+counts[j]];offset+=counts[j]
        queries=np.r_[np.clip(angle[rows],lo,hi),edges]
        cells=len(retained.cosines)*3 if retained is not None else 1
        anticipated=cells+len(queries)
        if evaluations+anticipated>request['maximum_cdf_evaluations']:
            raise MemoryError('Conditional native CDF work exceeds its frozen evaluation ceiling')
        cdf,bound,derivative,probability,operations=_conditional_angular_cdf(physical,retained,float(current_speed),queries,folded=folded,split=split)
        evaluations+=operations
        if evaluations>request['maximum_cdf_evaluations']:raise MemoryError('Actual conditional CDF integration exceeded its work allocation')
        transform[rows]=cdf[:len(rows)];error[rows]=bound+derivative*roundoff;rare_probability[rows]=probability
        per_bin=np.diff(cdf[len(rows):])
        if np.any(per_bin < -bound) or abs(float(per_bin.sum())-1)>2*bound:
            raise ValueError('Conditional cosine-bin probabilities do not preserve the actual rare distribution')
        per_bin=np.maximum(per_bin,0.)
        for b in np.unique(speed_bin[rows]):expected_counts[b]+=int(np.sum(speed_bin[rows]==b))*per_bin
    discrepancy=_empirical_uniform_discrepancy(transform);bound=float(error.max());limit=request['empirical_cdf_tolerance']
    measured_counts=np.zeros((ns,nb),dtype=np.int64);speed_count=np.bincount(speed_bin,minlength=ns)
    speed_discrepancy=np.full(ns,np.nan);speed_error=np.full(ns,np.nan);speed_supported=np.zeros(ns,bool)
    for b in range(ns):
        selected=speed_bin==b;measured_counts[b]=np.histogram(angle[selected],edges)[0]
        if speed_count[b]:
            speed_discrepancy[b]=_empirical_uniform_discrepancy(transform[selected]);speed_error[b]=float(error[selected].max())
            speed_supported[b]=speed_count[b]>=request['minimum_events_per_speed_bin']
    probability_edges=np.asarray(request['probability_bin_edges'],float)
    arrays={'native_event_id':identity.astype(np.uint64,copy=False),'relative_speed_before_kms':speed,'scattering_cosine':angle,
        'native_time_code':np.asarray(values['native_time_code'],float),
        'conditional_cdf_probability':transform,'conditional_cdf_absolute_error_bound':error,'conditional_rare_probability':rare_probability,
        'speed_bin_edges_kms':speed_edges,'cosine_bin_edges':edges,'native_speed_bin_count':speed_count,
        'native_speed_cosine_bin_count':measured_counts,'expected_speed_cosine_bin_count':expected_counts,
        'speed_bin_uniform_cdf_discrepancy':speed_discrepancy,'speed_bin_reference_error_bound':speed_error,'speed_bin_supported':speed_supported,
        'probability_bin_edges':probability_edges,'probability_bin_count':np.histogram(transform,probability_edges)[0]}
    passed=discrepancy+bound<=limit and bool(speed_supported.all()) and bool(np.all((speed_discrepancy+speed_error)[speed_supported]<=limit))
    report={'schema':'native-sidm-angular-control-v2','execution_status':'COMPLETE','request_hash':canonical_digest(request),
        'cdf_evaluation':'per_event_speed','event_count':len(angle),'status':'PASS' if passed else 'UNRESOLVED',
        'control_status':'PASS' if passed else 'UNRESOLVED',
        'maximum_empirical_cdf_difference':discrepancy,'reference_cdf_absolute_error_bound':bound,
        'speed_window_cdf_variation_bound':0.,'empirical_cdf_error_upper_bound':discrepancy+bound,
        'angular_model_hash':request['angular_model_hash'],'relative_speed_kms':request['relative_speed_kms'],
        'cosine_convention':request['cosine_convention'],'minimum_scattering_angle_radians':split,
        'actual_differential_table_sha256':events.get('actual_differential_table_sha256'),'actual_cdf_evaluations':evaluations,
        'speed_bins':[{'interval_kms':speed_edges[b:b+2].tolist(),'event_count':int(speed_count[b]),'supported':bool(speed_supported[b]),
            'empirical_cdf_error_upper_bound':float(speed_discrepancy[b]+speed_error[b]) if speed_count[b] else None} for b in range(ns)],
        'native_bin_count':measured_counts.sum(axis=0).tolist(),'reference_bin_probability':(expected_counts.sum(axis=0)/len(angle)).tolist(),
        'empirical_bin_probability':(measured_counts.sum(axis=0)/len(angle)).tolist(),'cosine_bin_edges':edges.tolist(),
        'probability_residual':((measured_counts.sum(axis=0)-expected_counts.sum(axis=0))/len(angle)).tolist(),
        'source_hashes':sources,'science_acceptance':'CONTROL_ONLY','control_scope':'conditional angle distribution of retained actual native events',
        'statistical_scope':'empirical uniformity of per-event conditional CDF transforms; no iid confidence level or independence inferred',
        'unmeasured_errors':['operator-realization scatter','unlogged native events','retained-table versus continuum-law interpolation error','physics beyond the retained differential model']}
    for name,sha in sources.items():
        if file_sha256(name)!=sha:raise ValueError('Speed-conditioned angular source changed during measurement')
    if destination is None:return report
    from .data import export_numeric_audit
    specifications={name:{'units':'km / s' if 'kms' in name else 'event count' if 'count' in name else 'dimensionless',
        'quantity':'native speed-conditioned angular diagnostic '+name.replace('_',' ')} for name in arrays}
    specifications['native_event_id']['units']='identity';specifications['native_event_id']['identity_namespace']=events['event_identity_namespace']
    specifications['native_time_code']['units']='native time code'
    event_fields={'native_event_id','relative_speed_before_kms','scattering_cosine','native_time_code','conditional_cdf_probability',
        'conditional_cdf_absolute_error_bound','conditional_rare_probability'}
    for name in set(arrays)-event_fields:
        specifications[name]['epoch']={'meaning':'aggregate over the retained native event population; axes describe speed, cosine or probability bins'}
    return export_numeric_audit(destination,report,arrays,specifications,algorithm=Path(__file__).resolve(),
        context={'epoch':{'status':'PER_RECORD','native_time_array':'native_time_code'},'coordinates':{'meaning':request['cosine_convention']}},source_hashes=sources)


def measure_native_angular_control(request, *, destination=None):
    """Measure saved native scattering angles against a retained reference CDF.

    Angles may be dependent in an evolving halo. We therefore publish their
    empirical discrepancy without assigning an iid KS/DKW confidence level.
    Independent-operator ensembles quantify scatter through block 32's native
    ensemble route. Missing native events cannot be replaced by Python draws.
    """
    validate_angular_control_stage({'request':request})
    memory = _positive(request['memory_gib'],'native angular workspace')
    if type(request['maximum_events']) is not int or request['maximum_events'] < 1:
        raise ValueError('Explicit native event bound required')
    sources = {}
    events,values = _numeric_control(request['events'],memory/2,sources)
    reference,table = _numeric_control(request['reference_cdf'],memory/2,sources)
    split = request.get('minimum_scattering_angle_radians',0.)
    if events.get('minimum_scattering_angle_radians',0.) != split or reference.get('minimum_scattering_angle_radians',0.) != split:
        raise ValueError('Native sampled angles and physical reference must retain the same actual angular partition')
    if events.get('schema') != 'native-sidm-scattering-events-v1' or reference.get('schema') != 'native-sidm-reference-cdf-v1':
        raise ValueError('Native event and physical reference CDF producer contracts required')
    for key in ('angular_model_hash','relative_speed_kms','cosine_convention'):
        if events[key] != reference[key] or events[key] != request[key]:
            raise ValueError('Angular control changes the physical model, speed or angle convention')
    if request['cosine_convention'] not in {'directed_cosine','identical_particle_folded_absolute_cosine'}:
        raise ValueError('Directed versus identical-particle folded angles must be explicit')
    physical = reference['physical_model']
    table_model = physical.get('schema') == 'sidm-elastic-model-v2'
    expected_kind = 'differential_table' if table_model else physical['kind']
    if events['actual_angular_kind'] != expected_kind or not table_model and physical['kind'] != 'isotropic' and events['actual_screening_velocity_kms'] != physical['velocity_scale_kms']:
        raise ValueError('Actual native angular operator differs from the independent physical reference')
    if table_model and events.get('actual_differential_table_sha256')!=reference['native_table']['sha256']:
        raise ValueError('Native angular events and reference CDF use different retained differential interpolants')
    angle = np.asarray(values['scattering_cosine'],float)
    identity = np.asarray(values['native_event_id'])
    if angle.ndim != 1 or not 1 <= len(angle) <= request['maximum_events'] or not np.isfinite(angle).all() or identity.shape != angle.shape or identity.dtype.kind not in 'iu' or len(np.unique(identity)) != len(identity):
        raise ValueError('Actual bounded native events need finite angles and distinct exact identities')
    if identity.dtype.kind == 'i' and np.any(identity < 0):
        raise ValueError('Native angular event IDs cannot use missing sentinels')
    lo,hi = (-1.,1.) if request['cosine_convention'] == 'directed_cosine' else (0.,1.)
    if np.any((angle < lo)|(angle > hi)):
        raise ValueError('Native scattering angles lie outside the declared physical convention')
    if request.get('cdf_evaluation','fixed_speed_reference')=='per_event_speed':
        return _measure_speed_conditioned_angles(request,events,values,reference,sources,destination)
    mu,cdf = np.asarray(table['cosine_nodes'],float),np.asarray(table['cdf_probability'],float)
    if mu.ndim != 1 or len(mu) < 2 or cdf.shape != mu.shape or not np.isfinite(mu).all() or not np.isfinite(cdf).all() or np.any(np.diff(mu) <= 0) or np.any(np.diff(cdf) < 0) or mu[0] != lo or mu[-1] != hi or cdf[0] != 0 or cdf[-1] != 1:
        raise ValueError('Physical reference CDF must be complete, monotone and normalized')
    sorted_angle = np.sort(angle)
    expected = np.interp(sorted_angle,mu,cdf)
    upper = np.arange(1,len(angle)+1)/len(angle)
    lower = np.arange(len(angle))/len(angle)
    discrepancy = float(max(np.max(upper-expected),np.max(expected-lower)))
    edges = np.asarray(request['cosine_bin_edges'],float)
    if edges.ndim != 1 or len(edges) < 2 or edges[0] != lo or edges[-1] != hi or np.any(np.diff(edges) <= 0):
        raise ValueError('Complete increasing declared cosine bins required')
    probabilities = np.diff(np.interp(edges,mu,cdf))
    counts = np.histogram(angle,edges)[0]
    measured = counts/len(angle)
    reference_bound = _positive(reference['cdf_absolute_error_bound'],'retained physical reference CDF error',True)
    tolerance = _positive(request['empirical_cdf_tolerance'],'predeclared angular empirical tolerance',True)
    native_speed_window = events.get('speed_absolute_tolerance_kms',0.)
    if native_speed_window > reference.get('speed_absolute_tolerance_kms',0.):
        raise ValueError('Native angle sample extends beyond the reference CDF speed domain')
    speed_bound = _positive(reference.get('speed_window_cdf_variation_bound',0.),'physical speed-window CDF variation',True)
    bound = discrepancy+reference_bound+speed_bound
    for name,sha in sources.items():
        if file_sha256(name) != sha:
            raise ValueError('Native angular control source changed during measurement')
    return {'schema':'native-sidm-angular-control-v1','execution_status':'COMPLETE',
        'request_hash':canonical_digest(request),'status':'PASS' if bound <= tolerance else 'UNRESOLVED',
        'event_count':len(angle),'maximum_empirical_cdf_difference':discrepancy,
        'reference_cdf_absolute_error_bound':reference_bound,'speed_window_cdf_variation_bound':speed_bound,
        'empirical_cdf_error_upper_bound':bound,
        'cosine_bin_edges':edges.tolist(),'native_bin_count':counts.tolist(),
        'empirical_bin_probability':measured.tolist(),'reference_bin_probability':probabilities.tolist(),
        'probability_residual':(measured-probabilities).tolist(),'source_hashes':sources,
        'angular_model_hash':request['angular_model_hash'],'relative_speed_kms':request['relative_speed_kms'],
        'cosine_convention':request['cosine_convention'],'science_acceptance':'CONTROL_ONLY',
        'statistical_scope':'saved native empirical distribution; event independence is not inferred',
        'unmeasured_errors':['operator-realization scatter','unlogged native events','relative-speed dependence outside this recorded speed']}


def angular_control_input_bindings(parameters):
    from .data import numeric_audit_source_bindings
    request = parameters['request']
    sources = {}
    for key in ('events','reference_cdf'):
        path = Path(request[key]['path']).resolve()
        if path.stat().st_size*8 > request['memory_gib']*1024**3:
            raise MemoryError('Native angular report exceeds declared input workspace')
        if file_sha256(path) != request[key]['sha256']:
            raise ValueError('Native angular input changed')
        report = json.loads(path.read_text())
        for name,sha in numeric_audit_source_bindings(report).items():
            if name in sources and sources[name] != sha:
                raise ValueError('Angular inputs disagree on their exact native sources')
            sources[name] = sha
        sources[str(path)] = file_sha256(path)
        seal = path.with_name(path.name+'.sha256')
        sources[str(seal)] = file_sha256(seal)
        for file in path.with_name(path.name+'.data').rglob('*'):
            if file.is_file():
                sources[str(file.resolve())] = file_sha256(file)
    return sources


def angular_control_runner(parameters,context):
    if parameters['request']['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('Angular control exceeds frozen stage workspace')
    path = context.generation/'sidm-angular-control.json'
    result = measure_native_angular_control(parameters['request'],destination=path)
    if 'numeric_data' not in result:
        atomic_json(path,result);atomic_text(path.with_name(path.name+'.sha256'),file_sha256(path)+'\n')
    return {'measurements':{'empirical_cdf_error_upper_bound':result['empirical_cdf_error_upper_bound'],
        'native_scattering_events':result['event_count']},
        'scientific_exclusions':result['unmeasured_errors']}


def angular_control_main():
    import argparse
    parser = argparse.ArgumentParser(description='Measure saved native SIDM angular distributions against a physical CDF')
    parser.add_argument('--request',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep native angular control history; choose a fresh output')
    result = measure_native_angular_control(json.loads(args.request.read_text()),destination=args.output)
    if 'numeric_data' not in result:
        atomic_json(args.output,result);atomic_text(args.output.with_name(args.output.name+'.sha256'),file_sha256(args.output)+'\n')


def validate_angular_control_stage(parameters):
    request = parameters['request']
    if request.get('schema') not in {'native-sidm-angular-control-request-v1','native-sidm-angular-control-request-v2'}:
        raise ValueError('Declared native SIDM angular control required')
    _positive(request['memory_gib'],'native angular allocation')
    _positive(request['relative_speed_kms'],'recorded physical relative speed')
    _positive(request['empirical_cdf_tolerance'],'declared CDF tolerance',True)
    if type(request['maximum_events']) is not int or request['maximum_events'] < 1:
        raise ValueError('Positive exact native event bound required')
    for key in ('events','reference_cdf'):
        if not isinstance(request[key],dict) or set(request[key]) != {'path','sha256'}:
            raise ValueError('Native angular inputs require an explicit path and hash')
    lo,hi = (-1.,1.) if request['cosine_convention'] == 'directed_cosine' else (0.,1.)
    if request['cosine_convention'] not in {'directed_cosine','identical_particle_folded_absolute_cosine'}:
        raise ValueError('Explicit physical angle convention required')
    edges = np.asarray(request['cosine_bin_edges'],float)
    if edges.ndim != 1 or not 2 <= len(edges) <= request['maximum_events']+1 or not np.isfinite(edges).all() or edges[0] != lo or edges[-1] != hi or np.any(np.diff(edges) <= 0):
        raise ValueError('Bounded increasing full-domain physical angle bins required')
    mode=request.get('cdf_evaluation','fixed_speed_reference')
    if mode not in {'fixed_speed_reference','per_event_speed'}:raise ValueError('Explicit fixed-speed or per-event conditional CDF evaluation required')
    if mode=='per_event_speed':
        if request['schema']!='native-sidm-angular-control-request-v2':raise ValueError('Per-event angular diagnostics require the v2 request')
        for name in ('maximum_cdf_evaluations','minimum_events_per_speed_bin'):
            if type(request[name])is not int or request[name]<1:raise ValueError('Positive exact conditional diagnostic bound required: '+name)
        tolerance=_positive(request['cosine_roundoff_tolerance'],'native angular reconstruction allowance',True)
        if tolerance>1e-8:raise ValueError('Angular roundoff allowance cannot hide a finite angular-domain error')
        split=request.get('minimum_scattering_angle_radians',0.)
        if isinstance(split,bool) or not isinstance(split,(float,int)) or not math.isfinite(split) or not 0<=split<(math.pi/2 if lo==0 else math.pi):
            raise ValueError('Conditional diagnostic cutoff must preserve the actual directed/folded rare domain')
        for name,left,right in (('probability_bin_edges',0.,1.),('speed_bin_edges_kms',None,None)):
            bins=np.asarray(request[name],float)
            if bins.ndim!=1 or not 2<=len(bins)<=request['maximum_events']+1 or not np.isfinite(bins).all() or np.any(np.diff(bins)<=0):
                raise ValueError('Bounded finite increasing conditional diagnostic bins required: '+name)
            if name=='probability_bin_edges' and (bins[0]!=left or bins[-1]!=right):raise ValueError('Conditional probability bins must span [0,1]')
            if name=='speed_bin_edges_kms' and bins[0]<0:raise ValueError('Speed bins cannot include negative relative speeds')
        if (len(request['cosine_bin_edges'])*len(request['speed_bin_edges_kms'])*64+request['maximum_events']*1024)>request['memory_gib']*1024**3:
            raise MemoryError('Conditional native diagnostic arrays exceed the declared memory allocation')
# @research2-end-module sidm_control_measurements


# @research2-module sidm_control_products
"""Physical angular reference tables and measured native scattering products.

CDF tables are independent analytic references, not evidence of native evolution.
Event products contain only records read from completed native HDF5 log shards;
angles, pair energy and momentum changes are reconstructed from their measured
pre/post velocities. This module does not generate replacement random events.
"""

from pathlib import Path
import json
import math
import numpy as np
from .foundation import canonical_digest, file_sha256, MSUN_G


def _positive(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError('Explicit finite '+name+' required')
    return float(value)


def _publish(destination,report,arrays,sources,units,context):
    from .data import export_numeric_audit
    source = Path(__file__).resolve()
    sources[str(source)] = file_sha256(source)
    physics_source = source.with_name('sidm_physics.py')
    sources[str(physics_source)] = file_sha256(physics_source)
    specifications = {name:{'units':units[name], 'annotations':{'meaning':name.replace('_',' ')},
        **({'identity_namespace':report['event_identity_namespace']} if name in {'native_event_id','particle_ids'} else {})} for name in arrays}
    return export_numeric_audit(destination,report,arrays,specifications,algorithm=source,
        context=context,source_hashes=sources)


def validate_cdf_request(request):
    if request.get('schema') != 'sidm-reference-cdf-request-v1':
        raise ValueError('Explicit physical SIDM CDF request required')
    for name in ('maximum_nodes','initial_nodes','maximum_refinement_rounds'):
        if type(request[name]) is not int or request[name] < (2 if name != 'maximum_refinement_rounds' else 1):
            raise ValueError('Bounded positive CDF construction setting required: '+name)
    if request['initial_nodes'] > request['maximum_nodes']:
        raise ValueError('Initial CDF grid exceeds the declared node bound')
    if request['maximum_nodes']*128 > _positive(request['memory_gib'],'CDF memory')*1024**3:
        raise MemoryError('Declared CDF node workspace exceeds its memory allocation')
    _positive(request['relative_speed_kms'],'physical relative speed',True)
    _positive(request.get('speed_absolute_tolerance_kms',0.),'reference speed window',True)
    tolerance = _positive(request['cdf_absolute_tolerance'],'CDF error tolerance')
    if tolerance <= 256*np.finfo(float).eps:
        raise ValueError('Declared CDF tolerance is below the retained floating-point allowance')
    if request['cosine_convention'] not in {'directed_cosine','identical_particle_folded_absolute_cosine'}:
        raise ValueError('Physical cosine convention required')
    split = request.get('minimum_scattering_angle_radians',0.)
    if isinstance(split,bool) or not isinstance(split,(int,float)) or not math.isfinite(split) or not 0 <= split < math.pi:
        raise ValueError('Explicit finite large-angle reference cutoff in [0,pi) required')
    if request['cosine_convention'] == 'identical_particle_folded_absolute_cosine' and split >= math.pi/2:
        raise ValueError('Folded native reference cutoff must leave a nonempty large-angle domain below pi/2')
    reference = request['physical_model']
    if not isinstance(reference,dict) or set(reference) != {'path','sha256'}:
        raise ValueError('Physical angular model requires its exact immutable source')
    return request


def cdf_input_bindings(parameters):
    request = validate_cdf_request(parameters['request'])
    path = Path(request['physical_model']['path']).resolve()
    if file_sha256(path) != request['physical_model']['sha256'] or path.stat().st_size*8 > request['memory_gib']*1024**3:
        raise ValueError('Physical angular model is changed or exceeds workspace')
    implementation=Path(__file__).with_name("sidm.py").resolve()
    return {str(path):file_sha256(path),str(implementation):file_sha256(implementation)}


def publish_reference_cdf(request,destination):
    request = validate_cdf_request(request)
    sources = cdf_input_bindings({'request':request})
    model = json.loads(Path(request['physical_model']['path']).read_text())
    if model.get('schema') == 'sidm-elastic-model-v2':
        return publish_differential_reference_cdf(request, model, sources, destination)
    if model.get('schema') != 'sidm-physical-angular-model-v1' or model['kind'] not in {'isotropic','screened_rutherford','born_moller'}:
        raise ValueError('Declared isotropic or screened Rutherford physical model required')
    if model['kind']=='born_moller' and (model.get('regime')!='nonrelativistic_born' or model.get('spin')!='unpolarized_spin_half' or model.get('normalization')!='rutherford_sigma0_including_identical_final_state_factor'):
        raise ValueError('Born Moller reference needs explicit approximation, spin and symmetry normalization')
    speed = request['relative_speed_kms']
    a = 0. if model['kind'] == 'isotropic' else .5*(speed/_positive(model['velocity_scale_kms'],'physical screening velocity'))**2
    if not math.isfinite(a) or not math.isfinite(a*(1+2*a)):
        raise ValueError('Angular anisotropy exceeds this finite analytic-reference domain')
    folded = request['cosine_convention'] == 'identical_particle_folded_absolute_cosine'
    moller = model['kind']=='born_moller'
    if moller and not folded:
        raise ValueError('Born Moller reference requires folded identical-particle angles')
    if moller:
        from .sidm_physics import moller_primitive, moller_curvature_upper_bound as curvature_upper_bound
        sources[str(Path(__file__).with_name('sidm.py').resolve())]=file_sha256(Path(__file__).with_name('sidm.py'))
    lo,hi = (0.,1.) if folded else (-1.,1.)
    split = request.get('minimum_scattering_angle_radians',0.)
    cutoff = 2*math.sin(split/2)**2
    cutoff_cosine = 1-cutoff
    large_probability = (2-cutoff)/(2*(1+a*cutoff))
    def folded_raw(mu):
        return mu*(1+2*a)/((1+a*(1-mu))*(1+a*(1+mu)))
    if moller:
        total_moller=float(moller_primitive(1.,a))
        conditional_moller=float(moller_primitive(cutoff_cosine,a))
        large_probability=conditional_moller/total_moller
    elif folded:
        large_probability = float(folded_raw(cutoff_cosine))
    if not math.isfinite(large_probability) or large_probability <= 0:
        raise ValueError('Requested large-angle reference has empty finite physical support')
    def directed(mu):
        if not split:
            return (1+mu)/(2*(1+a*(1-mu)))
        limited = np.minimum(mu,cutoff_cosine)
        return (1+limited)*(1+a*cutoff)/((2-cutoff)*(1+a*(1-limited)))
    def cumulative(mu):
        if moller:return moller_primitive(np.minimum(mu,cutoff_cosine),a)/conditional_moller
        return folded_raw(np.minimum(mu,cutoff_cosine))/large_probability if folded else directed(mu)
    nodes = np.linspace(lo,hi,request['initial_nodes'])
    if split:
        nodes = np.unique(np.r_[nodes,cutoff_cosine])
        if len(nodes) > request['maximum_nodes']:
            raise ValueError('Mandatory angular split knot exceeds the retained reference node allocation')
    rounding = 256*np.finfo(float).eps/large_probability
    if rounding >= request['cdf_absolute_tolerance']:
        raise ValueError('Conditional large-angle normalization exceeds the requested reference precision')
    cell_bounds = None
    for iteration in range(request['maximum_refinement_rounds']):
        # F''=a(1+2a)/(1+a(1-mu))^3 is positive and increasing.
        # The folded second derivative is F''(x)-F''(-x), bounded by F''(x).
        evaluation = np.minimum(nodes[1:],cutoff_cosine) if split else nodes[1:]
        curvature = curvature_upper_bound(evaluation,a,conditional_moller) if moller else a*(1+2*a)/(1+a*(1-evaluation))**3/large_probability
        if split:
            curvature = np.where(nodes[:-1] >= cutoff_cosine,0.,curvature)
        cell_bounds = curvature*np.diff(nodes)**2/8+rounding
        refine = cell_bounds > request['cdf_absolute_tolerance']
        if not refine.any():
            break
        if len(nodes)+int(refine.sum()) > request['maximum_nodes']:
            raise ValueError('Declared CDF grid bound cannot resolve this angular model; no silent tolerance relaxation')
        nodes = np.sort(np.r_[nodes,(nodes[:-1][refine]+nodes[1:][refine])/2])
    else:
        raise ValueError('CDF refinement exhausted its recorded round bound')
    probability = cumulative(nodes)
    probability[0],probability[-1] = 0.,1.
    if not np.isfinite(probability).all() or np.any(np.diff(probability) < 0):
        raise ValueError('Analytic CDF floating-point evaluation failed normalization/monotonicity')
    speed_width = request.get('speed_absolute_tolerance_kms',0.)
    variation = 0.
    if model['kind'] != 'isotropic' and speed_width:
        w = model['velocity_scale_kms']
        a_lower = .5*(max(0.,speed-speed_width)/w)**2
        a_upper = .5*((speed+speed_width)/w)**2
        # Directed |dF/da| <= 1/2; folded |dF/da| <= 1.
        variation = min(1.,max(a-a_lower,a_upper-a)*(4. if split and folded else 2. if split else 1. if folded else .5))
    if moller and speed_width:
        # Positive-polynomial log-derivative bound; conditioning increases
        # variation when the retained angular probability is small.
        unconditional=min(1.,32*math.log((1+a_upper)/(1+a_lower)))
        variation=1. if unconditional>=large_probability else min(1.,2*unconditional/(large_probability-unconditional))
    arrays = {'cosine_nodes':nodes,'cdf_probability':probability,'cdf_cell_absolute_error_bound':cell_bounds}
    report = {'schema':'native-sidm-reference-cdf-v1','reference_origin':'independent analytic physical model; not a native sampling result',
        'angular_model_hash':canonical_digest(model),'physical_model':model,
        'relative_speed_kms':speed,'cosine_convention':request['cosine_convention'],
        'minimum_scattering_angle_radians':split,'large_angle_probability_mass':large_probability,
        'anisotropy_parameter':a,'cdf_absolute_error_bound':float(cell_bounds.max()),
        'speed_absolute_tolerance_kms':speed_width,'speed_window_cdf_variation_bound':variation,
        'error_bound_method':'linear interpolation h^2 sup|F second derivative|/8 plus declared floating-point allowance',
        'floating_point_allowance':rounding,'grid_nodes':len(nodes),'refinement_rounds':iteration+1,
        'source_hashes':sources,'execution_status':'COMPLETE','status':'REFERENCE_ONLY',
        'physical_scope':'normalized angular distribution of the specified isotropic, screened Rutherford or exchange-interference Born Moller model at the recorded relative speed',
        'request_hash':canonical_digest(request)}
    return _publish(destination,report,arrays,sources,{key:'dimensionless' for key in arrays},
        {'epoch':{'status':'NOT_APPLICABLE','reason':'physical angular reference at fixed relative speed'},
         'coordinates':{'frame':'center-of-mass scattering cosine'},'physical_model_hash':canonical_digest(model)})


def publish_differential_reference_cdf(request, model, sources, destination):
    """Exact integration of the retained log-speed/linear-cosine interpolant.

    CDF interpolation bounds use the exact piecewise-constant second
    derivative of its piecewise-quadratic cumulative integral. The native
    materialized table, rather than a different analytic approximation, is
    the physical reference for every supplied v2 model.
    """
    from .sidm_physics import ElasticSIDMModel
    from .sidm_physics import _native_model_table
    elastic = ElasticSIDMModel(model)
    native = _native_model_table({'physical_model': request['physical_model']}, model)
    path = Path(native['path'])
    sources[str(path.resolve())] = native['sha256']
    lines = path.read_text().splitlines()
    nv, nmu, low, high = map(int, lines[2].split())
    if nv*nmu*64 > request['memory_gib']*1024**3:
        raise MemoryError('Materialized differential reference exceeds the declared workspace')
    speeds = np.fromstring(lines[4], sep=' ')
    cosine = np.fromstring(lines[5], sep=' ')
    differential = np.asarray([np.fromstring(line, sep=' ') for line in lines[6:]], float)
    if speeds.shape != (nv,) or cosine.shape != (nmu,) or differential.shape != (nv,nmu):
        raise ValueError('Native model table dimensions disagree with their exact materialization')
    spec = {'schema':'sidm-elastic-model-v2', 'kind':'differential_table',
        'supplied_cross_section':'total', 'particle_convention':elastic.particle_convention,
        'speed_kms':speeds.tolist(), 'cos_theta':cosine.tolist(),
        'differential_cm2_per_g_per_sr':differential.tolist(),
        'low_velocity_policy':{0:'error',1:'constant',2:'zero'}[low],
        'high_velocity_policy':{0:'error',1:'constant',2:'zero'}[high],
        'interpolation':{'absolute_tolerance_cm2_per_g':elastic.absolute_tolerance,
            'relative_tolerance':elastic.relative_tolerance}}
    retained = ElasticSIDMModel(spec)
    folded = request['cosine_convention'] == 'identical_particle_folded_absolute_cosine'
    if folded != (elastic.particle_convention == 'folded_identical'):
        raise ValueError('Differential CDF convention changes the actual native particle model')
    lo, hi = (0.,1.) if folded else (-1.,1.)
    cutoff = math.cos(request.get('minimum_scattering_angle_radians',0.))
    edges = np.unique(np.r_[lo, hi, cutoff, np.abs(cosine) if folded else cosine])
    edges = edges[(edges>=lo)&(edges<=hi)]
    model_knots = edges.copy()
    speed = request['relative_speed_kms']
    def distribution(v):
        y = retained.differential_cross_section_per_mass(v, model_knots)
        if folded:
            y = y + retained.differential_cross_section_per_mass(v, -model_knots)
        width = np.diff(model_knots)
        total = float(np.sum(width*(y[:-1]+y[1:])/2))
        domain = model_knots[1:] <= cutoff
        conditional = float(np.sum((width*(y[:-1]+y[1:])/2)[domain]))
        if not np.isfinite(y).all() or not math.isfinite(total) or not math.isfinite(conditional) or total <= 0 or conditional <= 0:
            raise ValueError('Declared differential angular domain has no positive physical support')
        masses = np.where(domain, width*(y[:-1]+y[1:])/2, 0.)
        cdf = np.r_[0., np.cumsum(masses)/conditional]
        cdf[-1] = 1.
        curvature = np.where(domain, np.abs(np.diff(y)/width)/conditional, 0.)
        return y, conditional, cdf, curvature, conditional/total
    y, normalization, cumulative, curvature, large_probability = distribution(speed)
    original_edges = edges.copy()
    original_cdf = cumulative.copy()
    rounding_bound = (16*len(model_knots)+256)*np.finfo(float).eps
    if rounding_bound >= request['cdf_absolute_tolerance']:
        raise ValueError('Differential CDF accumulation precision exceeds the requested tolerance')
    def evaluate(nodes):
        slots = np.minimum(np.searchsorted(original_edges,nodes,side='right')-1,len(original_edges)-2)
        slots = np.maximum(slots,0)
        delta = nodes-original_edges[slots]
        slope = np.diff(y)/np.diff(original_edges)
        extra = (y[slots]*delta + .5*slope[slots]*delta**2)/normalization
        return np.where(nodes>=cutoff,1.,original_cdf[slots]+extra)
    if len(edges) > request['maximum_nodes']:
        raise MemoryError('Actual differential knots exceed the declared diagnostic node ceiling')
    for iteration in range(request['maximum_refinement_rounds']):
        slots = np.maximum(0,np.minimum(np.searchsorted(original_edges,edges[:-1],side='right')-1,len(curvature)-1))
        cell_bounds = curvature[slots]*np.diff(edges)**2/8 + rounding_bound
        refine = cell_bounds > request['cdf_absolute_tolerance']
        if not refine.any():
            break
        if len(edges)+int(refine.sum()) > request['maximum_nodes']:
            raise MemoryError('Differential CDF precision exceeds the configured knot budget')
        edges = np.sort(np.r_[edges, .5*(edges[:-1]+edges[1:])[refine]])
    else:
        raise ValueError('Differential CDF refinement budget exhausted; requested tolerance was not established')
    probability = evaluate(edges)
    width = request.get('speed_absolute_tolerance_kms',0.)
    # Exact log-linear speed cells: conditional normalized CDFs vary
    # monotonically between cell endpoints; include all speed knots in the
    # requested window. No derivative approximation or assumed iid errors.
    variation = 0.
    for velocity in np.unique(np.r_[max(0.,speed-width),speed+width,
            speeds[(speeds>=max(0.,speed-width))&(speeds<=speed+width)]]):
        _, _, other, _, _ = distribution(float(velocity))
        # Include each quadratic extremum of the CDF difference.
        delta = other-original_cdf
        other_y, other_norm, _, _, _ = distribution(float(velocity))
        coefficients = other_y/other_norm-y/normalization
        derivative = np.diff(coefficients)/np.diff(original_edges)
        for k in range(len(delta)-1):
            variation=max(variation,abs(float(delta[k])),abs(float(delta[k+1])))
            if derivative[k] != 0 and original_edges[k] < cutoff:
                at=-coefficients[k]/derivative[k]
                if 0<at<original_edges[k+1]-original_edges[k]:
                    variation=max(variation,abs(float(delta[k]+coefficients[k]*at+.5*derivative[k]*at**2)))
    arrays={'cosine_nodes':edges,'cdf_probability':probability,'cdf_cell_absolute_error_bound':cell_bounds}
    report={'schema':'native-sidm-reference-cdf-v1','physical_model':model,
        'angular_model_hash':canonical_digest(model),'relative_speed_kms':speed,
        'cosine_convention':request['cosine_convention'],'minimum_scattering_angle_radians':request.get('minimum_scattering_angle_radians',0.),
        'speed_absolute_tolerance_kms':width,'speed_window_cdf_variation_bound':min(1.,variation),
        'cdf_absolute_error_bound':float(cell_bounds.max()),'native_table':native,
        'large_angle_probability':large_probability,'reference_origin':'exact retained differential interpolant; conditional quadratic cumulative integration',
        'source_hashes':sources,'execution_status':'COMPLETE','science_acceptance':'NOT_ESTABLISHED'}
    return _publish(destination,report,arrays,sources,{key:'dimensionless' for key in arrays},
        {'epoch':{'meaning':'model-defined relative-speed window'},'coordinates':{'meaning':request['cosine_convention']}})


def _event_text(attributes,name,default=None):
    value=attributes.get(name,default)
    if isinstance(value,bytes):value=value.decode('ascii')
    if not isinstance(value,str):raise ValueError('Native event text metadata missing or invalid: '+name)
    return value


def validate_events_request(request):
    if request.get('schema') not in {'native-sidm-event-export-request-v1','native-sidm-mapped-event-export-request-v1'}:
        raise ValueError('Native SIDM event export request required')
    mapped=request['schema']=='native-sidm-mapped-event-export-request-v1'
    if mapped:
        reference=request.get('mapped_run')
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:
            raise ValueError('Mapped event export requires its exact owned native run publication')
        if type(request.get('maximum_logged_events'))is not int or request['maximum_logged_events']<1:
            raise ValueError('Mapped event export requires a positive aggregate logged-row work bound')
    for key in ('maximum_events','chunk_rows'):
        if type(request[key]) is not int or request[key] < 1:
            raise ValueError('Explicit bounded event allocation required: '+key)
    memory = _positive(request['memory_gib'],'native event memory')*1024**3
    if request['maximum_events']*512+request['chunk_rows']*(1024 if mapped else 512) > memory:
        raise MemoryError('Native event selection and chunk workspace exceed the declared allocation')
    _positive(request['relative_speed_kms'],'selected physical relative speed')
    _positive(request['speed_absolute_tolerance_kms'],'speed selection tolerance',True)
    split = request.get('minimum_scattering_angle_radians',0.)
    if isinstance(split,bool) or not isinstance(split,(int,float)) or not math.isfinite(split) or not 0 <= split < math.pi:
        raise ValueError('Native event selection requires a finite actual angular cutoff in [0,pi)')
    if request['cosine_convention'] == 'identical_particle_folded_absolute_cosine' and split >= math.pi/2:
        raise ValueError('Folded hybrid sample cutoff must lie below pi/2')
    for key in ('energy_absolute_floor','momentum_absolute_floor'):
        _positive(request[key],key)
    if request['cosine_convention'] not in {'directed_cosine','identical_particle_folded_absolute_cosine'}:
        raise ValueError('Physical angle folding must be explicit')
    if not isinstance(request['event_files'],list) or not request['event_files']:
        raise ValueError('Actual completed native event log shards required')
    return request


def event_input_bindings(parameters):
    from .workflow import verify_completed_receipt
    request = validate_events_request(parameters['request'])
    receipt_path = Path(parameters['native_stage_receipt']).resolve()
    if receipt_path.stat().st_size*8 > request['memory_gib']*1024**3:
        raise MemoryError('Native event stage receipt exceeds declared workspace')
    receipt = verify_completed_receipt(receipt_path)
    root = Path(receipt['generation']).resolve()
    owned = {str(root/name):sha for name,sha in receipt['files_sha256'].items()}
    result = {str(receipt_path):file_sha256(receipt_path), **receipt['source_hashes'], **receipt['input_hashes']}
    paths = []
    for file in request['event_files']:
        path = Path(file).resolve()
        if owned.get(str(path)) != file_sha256(path):
            raise ValueError('Scattering log is outside the actual completed native evolution payload')
        paths.append(path);result[str(path)] = owned[str(path)]
    if len(set(paths)) != len(paths):
        raise ValueError('Repeated native scattering event shard')
    if request['schema']=='native-sidm-mapped-event-export-request-v1':
        run_path=Path(request['mapped_run']['path']).resolve();checksum=request['mapped_run']['sha256']
        if owned.get(str(run_path))!=checksum or file_sha256(run_path)!=checksum:
            raise ValueError('Mapped event run publication is outside the completed native stage payload')
        if run_path.stat().st_size*16>request['memory_gib']*1024**3:
            raise MemoryError('Mapped native run metadata exceeds event workspace')
        run=json.loads(run_path.read_text())
        if run.get('schema')!='sidm-canonical-native-run-v1' or run.get('execution_status')!='COMPLETE':
            raise ValueError('Actual completed mapped native run required')
        clock=run['native_clock'];clock_path=Path(clock['path']).resolve()
        if clock['physical_model_hash']!=request['angular_model_hash'] or owned.get(str(clock_path))!=clock['sha256'] or file_sha256(clock_path)!=clock['sha256']:
            raise ValueError('Mapped event clock/model differs from the completed native source payload')
        for name in ('parameter_file','build_configuration','invocation_receipt'):
            path=Path(run[name]).resolve()
            if owned.get(str(path))!=file_sha256(path):raise ValueError('Mapped native run loses owned '+name)
            result[str(path)]=owned[str(path)]
        invocation=json.loads(Path(run['invocation_receipt']).read_text())
        if invocation.get('status')!='RETURNED' or invocation.get('returncode')!=0 or invocation.get('input_hashes',{}).get(str(clock_path))!=clock['sha256']:
            raise ValueError('Mapped event clock lacks an actual successful native invocation binding')
        from core.native import verify_invocation_inputs
        from .evolution import parameter_values
        verify_invocation_inputs(invocation)
        result.update(invocation['input_hashes'])
        native=parameter_values(run['parameter_file'])
        if native.get('SIDMCanonicalClockSHA256')!=clock['sha256'] or native.get('SIDMAngularModelHash')!=request['angular_model_hash'] or native.get('SIDMEventIdentityNamespace')!=run['event_identity_namespace'] or native.get('ComovingIntegrationOn')!='0' or native.get('SIDMReferenceBackendKind')!='0':
            raise ValueError('Mapped native event parameters disagree with their original source clock/frame/model')
        if Path(native['SIDMCanonicalClockPath']).resolve()!=clock_path:
            raise ValueError('Mapped event source parameters reference another actual clock asset')
        features={line.strip().split('=',1)[0] for line in Path(run['build_configuration']).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')}
        if not {'RESEARCH_SIDM_CANONICAL_CONTROL','RESEARCH_SIDM_REFERENCE','RESEARCH_RESTART_STATE_CAPSULE','PERIODIC','FORCE_EQUAL_TIMESTEPS'}<=features or features & {'SELFGRAVITY','EXTERNALGRAVITY','PMGRID'}:
            raise ValueError('Mapped event source requires its gravity-free canonical native build profile')
        result[str(run_path)]=checksum;result[str(clock_path)]=clock['sha256']
    return result


def export_native_events(request,native_stage_receipt,destination):
    import h5py
    from .data import dataset_to_cgs_factor, exact_uint64_ids
    sources = event_input_bindings({'request':request,'native_stage_receipt':native_stage_receipt})
    mapped=request['schema']=='native-sidm-mapped-event-export-request-v1'
    mapped_run=json.loads(Path(request['mapped_run']['path']).read_text()) if mapped else None
    if mapped:
        from .evolution import parameter_values
        mapped_parameters=parameter_values(mapped_run['parameter_file'])
    retained = {key:[] for key in ('native_event_id','particle_ids','scattering_cosine','relative_speed_before_kms',
        'relative_speed_after_kms','pair_energy_relative_change','pair_momentum_relative_change','native_time_code')}
    retained['event_scale_factor']=[]
    if mapped:retained['canonical_tau_endpoint_seconds']=[]
    mapped_intervals=[];interval_keys=set()
    raw_count, selected_count, skipped_speed, zero_relative = 0,0,0,0
    namespace,angular_kind,screening_velocity,particle_convention = None,None,None,None
    differential_table_hash=None
    for file in request['event_files']:
        with h5py.File(file,'r') as f:
            if _event_text(f.attrs,'publication_state') != 'COMPLETE':
                raise ValueError('Native scattering log was not completely published')
            g = f['SIDMScatteringEvents']
            is_canonical=_event_text(g.attrs,'native_clock_coordinate','unspecified')=='canonical_native_time_code'
            if is_canonical!=mapped or (not mapped and _event_text(g.attrs,'velocity_frame','unspecified')=='canonical_u=a*v_pec'):
                raise ValueError('Native event frame differs from the requested physical or mapped adapter')
            if _event_text(g.attrs,'schema') != 'native-sidm-event-log-v1' or _event_text(g.attrs,'angular_model_hash') != request['angular_model_hash']:
                raise ValueError('Native event schema/model differs from the declared physical control')
            current_namespace = _event_text(g.attrs,'event_identity_namespace')
            if not isinstance(current_namespace,str) or not current_namespace or namespace is not None and current_namespace != namespace:
                raise ValueError('Native event log changes its exact global identity namespace')
            namespace = current_namespace
            if _event_text(g.attrs,'operator_kind') != 'rare_elastic_pair':
                raise ValueError('Continuous frequent diffusion steps are not discrete rare-scattering angles')
            native_split = float(g.attrs.get('split_angle_radians',0.))
            if native_split != request.get('minimum_scattering_angle_radians',0.):
                raise ValueError('Requested native angle sample changes the actual hybrid cutoff')
            convention = _event_text(g.attrs,'angular_particle_convention','directed_distinguishable')
            if convention not in {'directed_distinguishable','folded_identical_particle'} or particle_convention is not None and convention != particle_convention:
                raise ValueError('Native event shards change their actual angular particle convention')
            if convention == 'folded_identical_particle' and request['cosine_convention'] != 'identical_particle_folded_absolute_cosine':
                raise ValueError('Folded native samples cannot establish a directed angular distribution')
            if native_split and convention == 'directed_distinguishable' and request['cosine_convention'] != 'directed_cosine':
                raise ValueError('Truncated directed samples cannot be relabeled as a folded hybrid operator')
            particle_convention = convention
            kind = _event_text(g.attrs,'angular_kind')
            w = float(g.attrs['screening_velocity_kms'])
            if kind not in {'isotropic','screened_rutherford','born_moller','differential_table'} or not np.isfinite(w) or (kind not in {'isotropic','differential_table'} and w <= 0) or angular_kind is not None and (kind != angular_kind or w != screening_velocity):
                raise ValueError('Native log shards change the actual physical angular operator')
            angular_kind,screening_velocity = kind,w
            if kind=='differential_table':
                actual_hash=g.attrs.get('differential_table_sha256')
                if isinstance(actual_hash,bytes):actual_hash=actual_hash.decode('ascii')
                if (not isinstance(actual_hash,str) or len(actual_hash)!=64 or any(char not in '0123456789abcdef' for char in actual_hash)
                        or differential_table_hash is not None and differential_table_hash!=actual_hash):
                    raise ValueError('Native differential-table event shards lack one exact retained table identity')
                differential_table_hash=actual_hash
            h = _positive(f['Parameters'].attrs['HubbleParam'],'native event Hubble conversion')
            if mapped:
                for name,value in {'velocity_frame':'canonical_u=a*v_pec','position_frame':'canonical_x=r/a',
                        'source_clock_frame':'tau_integral_dt_over_a_squared','effective_differential_law':'a^-2*f_physical(u/a,mu)',
                        'canonical_clock_sha256':mapped_run['native_clock']['sha256'],
                        'physical_background_sha256':mapped_run['native_clock']['original_background_hash']}.items():
                    if _event_text(g.attrs,name)!=value:raise ValueError('Mapped native event lineage/frame differs: '+name)
                if namespace!=mapped_run['event_identity_namespace']:raise ValueError('Mapped event namespace changes the original population')
                native_begin=float(g.attrs['native_interval_begin']);native_end=float(g.attrs['native_interval_end'])
                tau_begin=float(g.attrs['canonical_tau_begin_seconds']);tau_end=float(g.attrs['canonical_tau_end_seconds'])
                frozen_a=float(g.attrs['frozen_physical_scale_factor']);begin_a=float(g.attrs['physical_scale_factor_begin']);end_a=float(g.attrs['physical_scale_factor_end'])
                time_unit=_positive(g.attrs['native_unit_time_s'],'mapped native time unit');native_h=_positive(g.attrs['native_hubble_parameter'],'mapped native h')
                for event_name,param_name in (('native_unit_mass_g','UnitMass_in_g'),('native_unit_length_cm','UnitLength_in_cm'),('native_hubble_parameter','HubbleParam')):
                    if float(g.attrs[event_name])!=float(mapped_parameters[param_name]):raise ValueError('Mapped event native units differ from launch: '+event_name)
                expected_time_unit=float(mapped_parameters['UnitLength_in_cm'])/float(mapped_parameters['UnitVelocity_in_cm_per_s'])
                if not math.isfinite(expected_time_unit) or abs(time_unit-expected_time_unit)>16*np.finfo(float).eps*max(time_unit,abs(expected_time_unit)):
                    raise ValueError('Mapped event native time unit differs from actual launch units')
                error=float(g.attrs['estimated_clock_error_seconds']);bracket=float(g.attrs['inverse_epoch_bracket_relative_width'])
                epoch_policy=_event_text(g.attrs,'canonical_source_epoch_policy') if 'canonical_source_epoch_policy' in g.attrs else 'whole_interval_midpoint'
                if epoch_policy not in {'whole_interval_midpoint','source_substep_left_endpoint'}:raise ValueError('Unknown mapped source freezing policy')
                if epoch_policy=='source_substep_left_endpoint':
                    expansion_policy=_event_text(g.attrs,'canonical_source_expansion_policy') if 'canonical_source_expansion_policy' in g.attrs else 'whole_native_interval_gate'
                    if expansion_policy not in {'whole_native_interval_gate','adaptive_substep_span_with_inverse_brackets'}:raise ValueError('Unknown mapped expansion-step policy')
                    work=float(g.attrs['canonical_source_clock_function_evaluations']);span=float(g.attrs['canonical_largest_substep_scale_fraction'])
                    step_error=float(g.attrs['canonical_substep_maximum_clock_error_seconds']);step_bracket=float(g.attrs['canonical_substep_maximum_inverse_bracket_relative_width'])
                    if not all(math.isfinite(value) for value in (work,span,step_error,step_bracket)) or work<0 or not work.is_integer() or work>int(mapped_parameters['SIDMCanonicalMaximumClockWork']) or span<0 or step_error<0 or step_bracket<0:
                        raise ValueError('Mapped actual substep clock work/domain diagnostics exceed their native allocation')
                    if _event_text(g.attrs,'canonical_kernel_radius_role')!='whole_interval_midpoint_reference_only; actual support is proper_radius/row_source_scale':
                        raise ValueError('Mapped per-substep kernel support metadata is ambiguous')
                    error=max(error,step_error);bracket=max(bracket,step_bracket)
                    span_limit=float(mapped_parameters['SIDMReferenceMaximumScaleFactorFractionalChange'])
                    if expansion_policy=='adaptive_substep_span_with_inverse_brackets' and span>span_limit*(1+32*np.finfo(float).eps):
                        raise ValueError('Mapped realized source expansion span exceeds its declared native limit')
                if not all(math.isfinite(value) for value in (native_begin,native_end,tau_begin,tau_end,frozen_a,begin_a,end_a,error,bracket)) or native_begin<0 or native_end<=native_begin or tau_begin<0 or tau_end<=tau_begin or begin_a<=0 or not begin_a<=frozen_a<=end_a or error<0 or bracket<0 or h!=native_h:
                    raise ValueError('Mapped native event source interval or inverse-clock estimates invalid')
                launch_begin=float(mapped_parameters['TimeBegin']);launch_end=float(mapped_parameters['TimeMax'])
                if native_begin<launch_begin or native_end>launch_end+32*np.finfo(float).eps*abs(launch_end):
                    raise ValueError('Mapped rare-event interval escapes the original native launch domain')
                for native,tau in ((native_begin,tau_begin),(native_end,tau_end)):
                    expected=native*time_unit/h
                    if not math.isfinite(expected) or abs(expected-tau)>32*np.finfo(float).eps*max(abs(expected),abs(tau),np.finfo(float).tiny):
                        raise ValueError('Mapped event tau endpoints disagree with actual native unit conversion')
                interval_key=(native_begin,native_end)
                if interval_key in interval_keys:raise ValueError('Repeated mapped rare-event source interval')
                interval_keys.add(interval_key)
                for name in ('VelocityBefore','VelocityAfter'):
                    if _event_text(g[name].attrs,'units')!='canonical a*v_pec km/s' or float(g[name].attrs['to_cgs'])!=1e5 or float(g[name].attrs['a_scaling'])!=0 or float(g[name].attrs['h_scaling'])!=0:
                        raise ValueError('Mapped event velocities lack their exact canonical unit contract')
                mapped_intervals.append({'native_begin':native_begin,'native_end':native_end,'tau_begin_seconds':tau_begin,'tau_end_seconds':tau_end,
                    'begin_scale_factor':begin_a,'end_scale_factor':end_a,'frozen_scale_factor':frozen_a,'clock_error_estimate_seconds':error,
                    'source_epoch_policy':epoch_policy,
                    'source_expansion_policy':expansion_policy if epoch_policy=='source_substep_left_endpoint' else 'whole_native_interval_gate',
                    'inverse_epoch_bracket_relative_width':bracket,'event_file':str(Path(file).resolve())})
            fields = ('EventID','ParticleIDs','VelocityBefore','VelocityAfter','Masses','Time')
            n = len(g['EventID'])
            if any(len(g[key]) != n for key in fields):
                raise ValueError('Native event columns do not retain one common row census')
            raw_count += n
            if mapped and raw_count>request['maximum_logged_events']:raise MemoryError('Mapped logged event work exceeds its aggregate row bound')
            for start in range(0,n,request['chunk_rows']):
                sl = slice(start,min(start+request['chunk_rows'],n))
                identities = exact_uint64_ids(g['EventID'][sl],allow_empty=True)
                ids = exact_uint64_ids(np.asarray(g['ParticleIDs'][sl]).reshape(-1),unique=False,allow_empty=True).reshape(-1,2)
                if ids.shape != (len(identities),2):
                    raise ValueError('Native events require exactly two original particle IDs per record')
                times = np.asarray(g['Time'][sl],float)
                a = np.asarray(g['ScaleFactor'][sl],float) if 'ScaleFactor' in g else np.ones(len(identities))
                if a.shape != times.shape or not np.isfinite(a).all() or np.any(a <= 0):
                    raise ValueError('Native event units require their actual positive scale factors')
                def physical(key,factor):
                    data = np.asarray(g[key][sl],float)
                    conversion = np.asarray([dataset_to_cgs_factor(g[key].attrs,float(epoch),h)/factor for epoch in a])
                    return data*conversion.reshape((-1,)+(1,)*(data.ndim-1))
                before,after,masses = physical('VelocityBefore',1e5),physical('VelocityAfter',1e5),physical('Masses',MSUN_G)
                if mapped:
                    if epoch_policy=='whole_interval_midpoint':
                        if np.any(a!=frozen_a) or np.any(times!=native_end):raise ValueError('Mapped event rows change the declared whole-interval frozen epoch')
                    elif np.any(a<begin_a) or np.any(a>end_a) or np.any(times<=native_begin) or np.any(times>native_end):
                        raise ValueError('Mapped event substep epoch/endpoints leave their actual source interval')
                    before=before/a[:,None,None];after=after/a[:,None,None]
                if before.shape != (len(identities),2,3) or after.shape != before.shape or masses.shape != (len(identities),2) or times.shape != (len(identities),) or not all(np.isfinite(q).all() for q in (before,after,masses,times)) or np.any(masses <= 0) or np.any(ids[:,0] == ids[:,1]):
                    raise ValueError('Native pre/post pair states are incomplete or nonphysical')
                relative_before,relative_after = before[:,0]-before[:,1],after[:,0]-after[:,1]
                speed_before,speed_after = np.linalg.norm(relative_before,axis=1),np.linalg.norm(relative_after,axis=1)
                nonzero = (speed_before > 0)&(speed_after > 0)
                selected = nonzero & (abs(speed_before-request['relative_speed_kms']) <= request['speed_absolute_tolerance_kms'])
                zero_relative += int((~nonzero).sum());skipped_speed += int((nonzero & ~selected).sum())
                selected_count += int(selected.sum())
                if selected_count > request['maximum_events']:
                    raise MemoryError('Native selected event count exceeds frozen publication bound')
                denominator = speed_before[selected]*speed_after[selected]
                cosine = np.clip(np.sum(relative_before[selected]*relative_after[selected],axis=1)/denominator,-1,1)
                if request['cosine_convention'] == 'identical_particle_folded_absolute_cosine':
                    cosine = abs(cosine)
                energy_before = .5*np.sum(masses*np.sum(before*before,axis=2),axis=1)
                energy_after = .5*np.sum(masses*np.sum(after*after,axis=2),axis=1)
                momentum_before = np.sum(masses[:,:,None]*before,axis=1)
                momentum_after = np.sum(masses[:,:,None]*after,axis=1)
                values = {'native_event_id':identities,'particle_ids':ids,'scattering_cosine':cosine,
                    'relative_speed_before_kms':speed_before,'relative_speed_after_kms':speed_after,
                    'pair_energy_relative_change':abs(energy_after-energy_before)/np.maximum(abs(energy_before),request['energy_absolute_floor']),
                    'pair_momentum_relative_change':np.linalg.norm(momentum_after-momentum_before,axis=1)/np.maximum(np.linalg.norm(momentum_before,axis=1),request['momentum_absolute_floor']),
                    'native_time_code':times,'event_scale_factor':a}
                if mapped:values['canonical_tau_endpoint_seconds']=times*time_unit/h
                for key,value in values.items():
                    retained[key].append(value if key == 'scattering_cosine' else value[selected])
    if not selected_count:
        raise ValueError('No actual measured rare-scattering events in the declared physical speed domain')
    arrays = {key:np.concatenate(rows) for key,rows in retained.items()}
    if len(np.unique(arrays['native_event_id'])) != selected_count:
        raise ValueError('Native selected event identities repeat across shards')
    if not all(np.isfinite(q).all() for q in arrays.values() if q.dtype.kind == 'f'):
        raise ValueError('Derived native event measurements overflow their physical numeric domain')
    for path,sha in sources.items():
        if file_sha256(path) != sha:
            raise ValueError('Native event source changed during publication')
    report = {'schema':'native-sidm-scattering-events-v1','angular_model_hash':request['angular_model_hash'],
        'relative_speed_kms':request['relative_speed_kms'],'speed_absolute_tolerance_kms':request['speed_absolute_tolerance_kms'],
        'cosine_convention':request['cosine_convention'],'event_identity_namespace':namespace,
        'minimum_scattering_angle_radians':request.get('minimum_scattering_angle_radians',0.),
        'actual_angular_kind':angular_kind,'actual_screening_velocity_kms':screening_velocity,
        'actual_particle_convention':particle_convention,
        'actual_differential_table_sha256':differential_table_hash,
        'event_count':selected_count,'logged_event_count':raw_count,'excluded_speed_count':skipped_speed,
        'unsupported_zero_relative_velocity_count':zero_relative,'source_hashes':sources,
        'native_stage_receipt':str(Path(native_stage_receipt).resolve()),'request_hash':canonical_digest(request),
        'maximum_pair_energy_relative_change':float(arrays['pair_energy_relative_change'].max()),
        'maximum_pair_momentum_relative_change':float(arrays['pair_momentum_relative_change'].max()),
        'execution_status':'COMPLETE','status':'MEASURED_ONLY',
        'measurement_definition':'angle between actual relative velocities before and after native elastic pair update',
        'coverage':'all logged records in the declared speed window; completeness of native logging is not inferred'}
    if mapped:
        report.update(input_velocity_frame='canonical_u=a*v_pec',physical_velocity_conversion='v_pec=u/actual_logged_source_scale_factor',
            mapped_run=request['mapped_run'],canonical_clock=mapped_run['native_clock'],mapped_source_intervals=mapped_intervals,
            event_epoch_definition='per-file declared freezing policy; row scale is the applied law epoch and row native time is its source/substep endpoint',
            science_acceptance='NOT_ESTABLISHED')
    units = {key:'dimensionless' for key in arrays}
    units.update(relative_speed_before_kms='km / s',relative_speed_after_kms='km / s',native_time_code='native time code')
    if mapped:units['canonical_tau_endpoint_seconds']='s'
    return _publish(destination,report,arrays,sources,units,
        {'epoch':{'status':'PER_RECORD','native_time_array':'native_time_code'},
         'coordinates':{'frame':'physical peculiar velocities; center-of-mass relative scattering angle'}})


def validate_control_population_request(parameters):
    """Define actual equal-mass populations, independently of an SIDM operator.

    Cohorts label initial velocity populations of one physical DM species.
    They are not unequal-mass target particles or different interacting species.
    The output is an initial state, never an evolution/control acceptance receipt.
    """
    request=parameters['request']
    required={'schema','kind','identity_namespace','seed','position_sampling','components',
        'box_proper_kpc','macro_mass_msun','clock','native_units','maximum_particles',
        'memory_gib','disk_gib','chunk_rows'}
    if not isinstance(request,dict) or set(request)!=required or request['schema']!='sidm-control-population-request-v1':
        raise ValueError('Complete typed SIDM physical control population required')
    if request['kind'] not in {'uniform_beam','thermal_equilibrium','physical_comoving','rare_frequent_hybrid'}:
        raise ValueError('Unknown dedicated SIDM population kind')
    if not isinstance(request['identity_namespace'],str) or not request['identity_namespace'].strip():
        raise ValueError('Explicit control particle identity namespace required')
    if type(request['seed'])is not int or not 0<=request['seed']<2**64:
        raise ValueError('Control population seed must be an exact uint64')
    for name in ('maximum_particles','chunk_rows'):
        if type(request[name])is not int or request[name]<1:raise ValueError('Positive exact population bound required: '+name)
    for name in ('box_proper_kpc','macro_mass_msun','memory_gib','disk_gib'):
        _positive(request[name],name)
    if request['position_sampling'] not in {'uniform_random','periodic_lattice'}:
        raise ValueError('Declare random homogeneous or periodic lattice positions')
    components=request['components']
    if not isinstance(components,list) or not 1<=len(components)<=16:
        raise ValueError('One to sixteen explicit velocity cohorts required')
    count=0;names=set()
    for component in components:
        if not isinstance(component,dict) or set(component)!={'name','particle_count','bulk_velocity_kms','dispersion_1d_kms','remove_sample_bulk'}:
            raise ValueError('Each cohort needs its count, bulk, Cartesian Gaussian dispersions and conditioning policy')
        name=component['name']
        if not isinstance(name,str) or not name or name in names:raise ValueError('Unique nonempty cohort names required')
        names.add(name)
        number=component['particle_count']
        if type(number)is not int or number<2:raise ValueError('Each velocity cohort requires at least two exact particles')
        count+=number
        for key in ('bulk_velocity_kms','dispersion_1d_kms'):
            if not isinstance(component[key],list) or len(component[key])!=3 or any(isinstance(x,bool) or not isinstance(x,(int,float)) for x in component[key]):
                raise ValueError('Cartesian velocity inputs must be three actual numeric values')
            value=np.asarray(component[key],float)
            if value.shape!=(3,) or not np.isfinite(value).all() or key=='dispersion_1d_kms' and np.any(value<0):
                raise ValueError('Finite Cartesian velocity specification required: '+key)
        if type(component['remove_sample_bulk'])is not bool:raise ValueError('Gaussian sample conditioning policy must be boolean')
    if count>request['maximum_particles'] or count>=2**32:
        raise MemoryError('Single-file native control census exceeds its explicit bound')
    if count*640+min(count,request['chunk_rows'])*256>request['memory_gib']*1024**3:
        raise MemoryError('Population, permutations, numeric publication and native chunk workspace exceed allocation')
    if count*512+1048576>request['disk_gib']*1024**3:
        raise MemoryError('Physical product and native IC exceed the declared disk allocation')
    if request['position_sampling']=='periodic_lattice':
        side=round(count**(1/3))
        if side**3!=count:raise ValueError('Periodic lattice requires a complete cubic particle census')
    if request['kind']=='uniform_beam' and len(components)!=2:
        raise ValueError('Uniform beam requires two explicit equal-mass velocity cohorts')
    if request['kind']=='thermal_equilibrium':
        sigma=np.asarray(components[0]['dispersion_1d_kms'],float)
        if len(components)!=1 or sigma[0]<=0 or not np.all(sigma==sigma[0]):
            raise ValueError('Thermal equilibrium requires one isotropic positive-temperature Gaussian population')
    try:
        density=count*request['macro_mass_msun']/request['box_proper_kpc']**3
    except (OverflowError,ZeroDivisionError) as exc:
        raise ValueError('Control physical density overflows its supported units') from exc
    if not math.isfinite(density) or density<=0:raise ValueError('Control physical density must remain positive and finite')
    clock=request['clock']
    if not isinstance(clock,dict) or set(clock)!={'mode','scale_factor','proper_time_seconds'} or clock['mode'] not in {'physical','comoving'}:
        raise ValueError('Explicit physical or comoving native clock required')
    _positive(clock['scale_factor'],'initial scale factor')
    _positive(clock['proper_time_seconds'],'initial proper time',True)
    if clock['mode']=='physical' and clock['scale_factor']!=1:
        raise ValueError('Physical native coordinates require scale_factor=1')
    units=request['native_units']
    if not isinstance(units,dict) or set(units)!={'length_cm','mass_g','velocity_cm_s','hubble_parameter','omega_m0','omega_lambda0','hubble0_km_s_mpc'}:
        raise ValueError('Complete native physical unit and background header definition required')
    for name in ('length_cm','mass_g','velocity_cm_s','hubble_parameter','hubble0_km_s_mpc'):_positive(units[name],name)
    for name in ('omega_m0','omega_lambda0'):_positive(units[name],name,True)
    if not math.isclose(units['hubble0_km_s_mpc'],100*units['hubble_parameter'],rel_tol=1e-12):
        raise ValueError('Native Hubble parameter and H0 must describe the same background')
    return request


def publish_control_population(parameters,destination):
    """Generate sealed proper phase space and a matched native Gadget HDF5 IC.

    The same physical product can be converted for physical/comoving controls
    and independent engines. The native IC has one equal-mass PartType1 species;
    original cohort membership remains in the sealed exact-ID product.
    """
    import h5py,os,uuid,shutil
    from .foundation import KPC_CM
    request=validate_control_population_request(parameters)
    destination=Path(destination).expanduser().resolve()
    native=destination.with_name(destination.stem+'-ics.hdf5')
    if destination.exists() or native.exists():raise FileExistsError('Keep previously generated control populations')
    destination.parent.mkdir(parents=True,exist_ok=True)
    count=sum(row['particle_count'] for row in request['components'])
    if count*512+1048576>shutil.disk_usage(destination.parent).free:raise MemoryError('Insufficient free space for the owned physical/native control population')
    rng=np.random.Generator(np.random.PCG64(request['seed']))
    box=float(request['box_proper_kpc'])
    if request['position_sampling']=='uniform_random':position=rng.uniform(0.,box,(count,3))
    else:
        side=round(count**(1/3));linear=np.arange(count,dtype=np.int64)
        position=np.column_stack((linear//(side*side),(linear//side)%side,linear%side)).astype(float)
        position=(position+.5)*(box/side)
        # Cohort membership must not occupy separate contiguous lattice slabs.
        position=position[rng.permutation(count)]
    velocity=np.empty((count,3));cohort=np.empty(count,dtype=np.uint8);summaries=[];offset=0
    for index,component in enumerate(request['components']):
        number=component['particle_count'];sl=slice(offset,offset+number)
        sigma=np.asarray(component['dispersion_1d_kms'],float);bulk=np.asarray(component['bulk_velocity_kms'],float)
        thermal=rng.normal(size=(number,3))*sigma
        removed=thermal.mean(axis=0) if component['remove_sample_bulk'] else np.zeros(3)
        velocity[sl]=thermal-removed+bulk;cohort[sl]=index
        actual=velocity[sl];mean=actual.mean(axis=0);centered=actual-mean
        with np.errstate(over='ignore',invalid='ignore'):
            covariance=centered.T@centered/number;expected=np.diag(sigma*sigma)
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all() or not np.isfinite(expected).all():
            raise ValueError('Generated cohort moments exceed finite physical velocity units')
        summaries.append({'name':component['name'],'particle_count':number,'measured_bulk_velocity_kms':mean.tolist(),
            'measured_population_covariance_kms2':covariance.tolist(),
            'removed_sample_velocity_kms':removed.tolist(),
            'distribution':'Gaussian draw conditioned on cohort sample mean' if component['remove_sample_bulk'] else 'independent Cartesian Gaussian draws',
            'expected_unconditioned_covariance_kms2':expected.tolist(),
            'finite_sample_conditioned_variance_factor':(number-1)/number if component['remove_sample_bulk'] else 1.})
        offset+=number
    if not np.isfinite(velocity).all():raise ValueError('Control Gaussian draw exceeds finite native velocity support')
    ids=np.arange(1,count+1,dtype=np.uint64)
    mass=np.full(count,float(request['macro_mass_msun']))
    clock=request['clock'];units=request['native_units'];comoving=clock['mode']=='comoving'
    a=float(clock['scale_factor']);h=float(units['hubble_parameter'])
    length=float(units['length_cm']);mass_unit=float(units['mass_g']);velocity_unit=float(units['velocity_cm_s'])
    native_time=a if comoving else float(clock['proper_time_seconds'])*h/(length/velocity_unit)
    native_box=box*KPC_CM*h/(length*a)
    native_mass=float(request['macro_mass_msun'])*MSUN_G*h/mass_unit
    factors={'Coordinates':KPC_CM*h/(length*a),'Velocities':1e5/(velocity_unit*math.sqrt(a)),
        'Masses':MSUN_G*h/mass_unit}
    if not np.isfinite([native_time,native_box,native_mass,*factors.values()]).all() or min(native_box,native_mass,*factors.values())<=0:
        raise ValueError('Native control unit conversion is nonfinite or underflows')
    temporary=native.with_name('.'+native.name+'.'+uuid.uuid4().hex)
    try:
        with h5py.File(temporary,'x') as handle:
            header=handle.create_group('Header');counts=np.zeros(6,dtype=np.uint32);counts[1]=count
            table=np.zeros(6);table[1]=native_mass
            header.attrs.update({'NumPart_ThisFile':counts,'NumPart_Total':counts,'NumPart_Total_HighWord':np.zeros(6,dtype=np.uint32),
                'MassTable':table,'Time':native_time,'Redshift':1/a-1 if comoving else 0.,'BoxSize':native_box,
                'NumFilesPerSnapshot':1,'Omega0':units['omega_m0'],'OmegaLambda':units['omega_lambda0'],'HubbleParam':h,
                'Flag_Sfr':0,'Flag_Cooling':0,'Flag_StellarAge':0,'Flag_Metals':0,'Flag_Feedback':0,'Flag_DoublePrecision':1,
                'Research_sidm_control_population_schema':1,'Research_sidm_control_request_sha256':canonical_digest(request)})
            params=handle.create_group('Parameters')
            params.attrs.update({'ComovingIntegrationOn':int(comoving),'PeriodicBoundariesOn':1,'HubbleParam':h,
                'Omega0':units['omega_m0'],'OmegaLambda':units['omega_lambda0'],'Hubble0_km_s_mpc':units['hubble0_km_s_mpc'],
                'UnitLength_in_cm':length,'UnitMass_in_g':mass_unit,'UnitVelocity_in_cm_per_s':velocity_unit,
                'UnitTime_in_s':length/velocity_unit})
            group=handle.create_group('PartType1')
            datasets={name:group.create_dataset(name,shape=(count,3) if name in {'Coordinates','Velocities'} else (count,),
                dtype='<u8' if name=='ParticleIDs' else '<f8') for name in ('Coordinates','Velocities','Masses','ParticleIDs')}
            for start in range(0,count,request['chunk_rows']):
                sl=slice(start,min(start+request['chunk_rows'],count))
                for name,values in (('Coordinates',position),('Velocities',velocity),('Masses',mass)):
                    converted=values[sl]*factors[name]
                    if not np.isfinite(converted).all():raise ValueError('Native converted particle state overflow')
                    datasets[name][sl]=converted
                datasets['ParticleIDs'][sl]=ids[sl]
            for name,factor,aexp,hexp in (('Coordinates',length,1 if comoving else 0,-1),
                    ('Velocities',velocity_unit,.5 if comoving else 0,0),('Masses',mass_unit,0,-1)):
                datasets[name].attrs.update({'to_cgs':factor,'CGSConversionFactor':factor,'a_scaling':aexp,'h_scaling':hexp,
                    'aexp-scale-exponent':aexp,'h-scale-exponent':hexp})
            datasets['ParticleIDs'].attrs['identity_namespace']=request['identity_namespace']
            handle.flush()
        # Link publication refuses replacement even if another publisher wins.
        os.link(temporary,native)
    finally:
        if temporary.exists():temporary.unlink()
    arrays={'particle_ids':ids,'cohort_index':cohort,'position_proper_kpc':position,'velocity_peculiar_kms':velocity,'mass_msun':mass}
    import hashlib
    state_digest=hashlib.sha256(canonical_digest({'schema':'sidm-control-physical-state-v1',
        'identity_namespace':request['identity_namespace'],'box_proper_kpc':box}).encode('ascii'))
    for name,dtype in (('particle_ids','<u8'),('cohort_index','u1'),('position_proper_kpc','<f8'),
            ('velocity_peculiar_kms','<f8'),('mass_msun','<f8')):
        state_digest.update(name.encode('ascii'))
        for start in range(0,count,request['chunk_rows']):
            state_digest.update(np.asarray(arrays[name][start:start+request['chunk_rows']],dtype=dtype).tobytes(order='C'))
    report={'schema':'sidm-control-population-v1','request':request,'kind':request['kind'],'execution_status':'COMPLETE',
        'physical_state_sha256':state_digest.hexdigest(),
        'event_identity_namespace':request['identity_namespace'],'particle_count':count,'cohorts':summaries,
        'physical_density_msun_kpc3':count*request['macro_mass_msun']/box**3,
        'native_ic':{'path':str(native),'sha256':file_sha256(native),'native_time':native_time,'box_size_code':native_box,
            'equal_macro_mass_code':native_mass,'velocity_storage':'peculiar/sqrt(a) in native velocity units' if comoving else 'physical native velocity units'},
        'operator_scope':'one equal-mass physical species; both cohorts respond; no infinite-mass fixed-target approximation',
        'evolution_status':'NOT_RUN','science_acceptance':'NOT_ESTABLISHED',
        'rng':{'implementation':'numpy PCG64','numpy_version':np.__version__,'seed':request['seed']}}
    return _publish(destination,report,arrays,{str(native):file_sha256(native)},
        {'particle_ids':'dimensionless','cohort_index':'dimensionless','position_proper_kpc':'proper kpc','velocity_peculiar_kms':'km / s','mass_msun':'Msun'},
        {'epoch':{'canonical_time':'scale_factor' if comoving else 'proper_seconds','actual_scale_factor':a,
            'proper_time_seconds':clock['proper_time_seconds']},'coordinates':{'frame':'proper periodic Cartesian; physical peculiar velocity'}})


def validate_canonical_population_request(parameters):
    request=parameters['request']
    keys={'schema','population','native_clock','maximum_particles','maximum_asset_bytes','chunk_rows','memory_gib','disk_gib'}
    if not isinstance(request,dict) or set(request)!=keys or request.get('schema')!='sidm-canonical-population-request-v1':raise ValueError('Complete original-population canonical native IC request required')
    for name in ('maximum_particles','maximum_asset_bytes','chunk_rows'):
        if type(request[name])is not int or request[name]<1:raise ValueError('Exact positive mapped population allocation required: '+name)
    if request['chunk_rows']>request['maximum_particles']:raise ValueError('Canonical native chunk exceeds particle bound')
    for name in ('memory_gib','disk_gib'):_positive(request[name],name)
    for name in ('population','native_clock'):
        reference=request[name]
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('Exact original population/clock reference required: '+name)
        for field in ('path','sha256'):
            value=reference[field]
            if isinstance(value,dict) and 'stage' in value:continue
            if not isinstance(value,str) or not value or field=='sha256' and (len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):raise ValueError('Complete native canonical path/SHA reference required: '+name)
    if request['maximum_particles']*1536+request['chunk_rows']*512+request['maximum_asset_bytes']*8>request['memory_gib']*1024**3/2:raise MemoryError('Original/mapped population and native chunk workspace exceed allocation')
    if request['maximum_particles']*768+request['maximum_asset_bytes']+1048576>request['disk_gib']*1024**3:raise MemoryError('Mapped native population exceeds disk allocation')
    return request


def canonical_population_input_bindings(parameters):
    from .sidm_analysis import _load_product
    request=validate_canonical_population_request(parameters);base=Path(parameters['request_base']);memory=request['memory_gib']*1024**3
    population,arrays,sources=_load_product(request['population'],base,memory/3)
    if population.get('schema')!='sidm-control-population-v1' or population.get('execution_status')!='COMPLETE':raise ValueError('Actual sealed original physical control population required')
    path=(base/request['native_clock']['path']).resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size*16>memory/3 or file_sha256(path)!=request['native_clock']['sha256']:raise ValueError('Native canonical clock manifest changed or exceeds allocation')
    clock=json.loads(path.read_text());sources[str(path)]=request['native_clock']['sha256']
    if clock.get('schema')!='sidm-native-canonical-clock-v1' or clock.get('execution_status')!='COMPLETE' or 'physical_H0_per_second' not in clock:
        raise ValueError('Current original-background native clock asset required; regenerate older assets')
    if clock['population_physical_state_sha256']!=population['physical_state_sha256'] or clock['identity_namespace']!=population['event_identity_namespace']:raise ValueError('Canonical clock and physical population lineage differ')
    asset=Path(clock['clock_asset']).resolve()
    if not asset.is_file() or asset.is_symlink() or asset.stat().st_size>request['maximum_asset_bytes'] or asset.stat().st_size!=clock['clock_asset_bytes'] or file_sha256(asset)!=clock['clock_asset_sha256']:raise ValueError('Actual native canonical clock bytes differ or exceed allocation')
    tokens=asset.read_text(encoding='ascii').split()
    if len(tokens)<17 or tokens[0]!='RESEARCH_SIDM_CANONICAL_CLOCK_V1' or tokens[1]!=clock['physical_model_hash'] or tokens[2]!=clock['original_background_hash']:
        raise ValueError('Native clock payload physical identities differ from its manifest')
    if not tokens[12].isdigit() or len(tokens)!=17+2*int(tokens[12]) or int(tokens[12])>clock['request']['maximum_rows']:
        raise ValueError('Native clock payload row census differs from its source-bound allocation')
    for index,name in ((4,'physical_H0_per_second'),(5,'actual_scale_factor_begin'),(6,'actual_scale_factor_end')):
        if float(tokens[index])!=float(clock[name]):raise ValueError('Native clock payload physical epoch/H0 differs from its manifest: '+name)
    for source,digest in clock['source_hashes'].items():
        original=Path(source).resolve()
        if not original.is_file() or file_sha256(original)!=digest:raise ValueError('Original canonical clock source changed: '+source)
        if str(original) in sources and sources[str(original)]!=digest:raise ValueError('Canonical clock/population source identities conflict')
        sources[str(original)]=digest
    sources[str(asset)]=clock['clock_asset_sha256']
    return request,population,arrays,clock,sources


def publish_canonical_population(parameters,destination):
    """Materialize exact original physical particles in x=r/a0, u=a0*v."""
    import h5py,os,uuid,shutil,hashlib
    from .data import exact_uint64_ids
    from .foundation import KPC_CM,H100_S
    request,population,initial,clock,sources=canonical_population_input_bindings(parameters)
    contracts=population['numeric_data']['array_contract']['datasets']
    for name,unit in (('particle_ids','dimensionless'),('cohort_index','dimensionless'),('position_proper_kpc','proper kpc'),('velocity_peculiar_kms','km / s'),('mass_msun','Msun')):
        if contracts[name]['units']!=unit:raise ValueError('Original canonical mapping physical units differ: '+name)
    if contracts['particle_ids'].get('identity_namespace')!=population['event_identity_namespace']:raise ValueError('Original particle ID namespace contract differs')
    ids=exact_uint64_ids(initial['particle_ids']);count=len(ids)
    if not 1<=count<=request['maximum_particles'] or count!=population['particle_count'] or count>2**32-1:raise MemoryError('Complete canonical native census exceeds its supported allocation')
    labels=np.asarray(initial['cohort_index']);position=np.asarray(initial['position_proper_kpc'],float);velocity=np.asarray(initial['velocity_peculiar_kms'],float);mass=np.asarray(initial['mass_msun'],float)
    if labels.shape!=(count,) or labels.dtype.kind not in 'iu' or position.shape!=(count,3) or velocity.shape!=position.shape or mass.shape!=(count,):raise ValueError('Exact aligned original cohort/phase/mass rows required')
    if not all(np.isfinite(value).all() for value in (position,velocity,mass)) or np.any(mass<=0) or not np.all(mass==mass[0]):raise ValueError('Canonical mapped elastic controls require finite equal-mass original particles')
    cohorts=population['cohorts']
    if not 1<=len(cohorts)<=16:raise ValueError('Original bounded control cohort definitions required')
    if np.any(labels>=len(cohorts)) or np.any(labels<0) or not np.array_equal(np.bincount(labels.astype(np.int64),minlength=len(cohorts)),[row['particle_count'] for row in cohorts]):raise ValueError('Canonical mapping changes original cohort membership')
    a=float(clock['actual_scale_factor_begin']);box=float(population['request']['box_proper_kpc']);units=population['request']['native_units']
    if not math.isfinite(a) or a<=0 or population['request']['clock']['mode']!='comoving' or float(population['request']['clock']['scale_factor'])!=a:raise ValueError('Original population must be at the actual canonical expanding start epoch')
    if np.any(position<0) or np.any(position>=box):raise ValueError('Original physical population must retain complete periodic positions')
    original_digest=hashlib.sha256(canonical_digest({'schema':'sidm-control-physical-state-v1','identity_namespace':population['event_identity_namespace'],'box_proper_kpc':box}).encode('ascii'))
    for name,dtype in (('particle_ids','<u8'),('cohort_index','u1'),('position_proper_kpc','<f8'),('velocity_peculiar_kms','<f8'),('mass_msun','<f8')):
        original_digest.update(name.encode('ascii'))
        for start in range(0,count,request['chunk_rows']):original_digest.update(np.asarray(initial[name][start:start+request['chunk_rows']],dtype=dtype).tobytes(order='C'))
    if original_digest.hexdigest()!=population['physical_state_sha256']:raise ValueError('Original typed particle phase-space identity differs from its physical state hash')
    h=float(units['hubble_parameter']);length=float(units['length_cm']);speed_unit=float(units['velocity_cm_s']);mass_unit=float(units['mass_g'])
    if not math.isclose(H100_S*h,float(clock['physical_H0_per_second']),rel_tol=1e-12):raise ValueError('Original native h/H0 convention differs from actual clock background')
    canonical_box=box/a
    if not math.isclose(canonical_box,float(clock['canonical_box_kpc']),rel_tol=1e-12):raise ValueError('Original physical box differs from measured canonical clock domain')
    x=position/a;u=a*velocity;box_code=canonical_box*KPC_CM*h/length;mass_code=mass[0]*MSUN_G*h/mass_unit
    factors={'Coordinates':KPC_CM*h/length,'Velocities':1e5/speed_unit,'Masses':MSUN_G*h/mass_unit}
    if not all(np.isfinite(value).all() for value in (x,u)) or not np.isfinite([box_code,mass_code,*factors.values()]).all() or min(box_code,mass_code,*factors.values())<=0:raise ArithmeticError('Canonical native state or unit conversion overflows')
    destination=Path(destination).resolve();native=destination.with_name(destination.stem+'-canonical-ics.hdf5');clock_copy=destination.with_name(destination.stem+'-canonical-clock.txt')
    if any(path.exists() for path in (destination,native,clock_copy)):raise FileExistsError('Retain prior mapped populations and native assets')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if count*768+clock['clock_asset_bytes']+1048576>shutil.disk_usage(destination.parent).free:raise MemoryError('Insufficient free space for mapped native input publication')
    temporary=native.with_name('.'+native.name+'.'+uuid.uuid4().hex);clock_temporary=clock_copy.with_name('.'+clock_copy.name+'.'+uuid.uuid4().hex)
    try:
        shutil.copyfile(clock['clock_asset'],clock_temporary)
        if file_sha256(clock_temporary)!=clock['clock_asset_sha256']:raise ValueError('Owned canonical clock copy changed during publication')
        with h5py.File(temporary,'x') as handle:
            header=handle.create_group('Header');counts=np.zeros(6,dtype=np.uint32);counts[1]=count;table=np.zeros(6);table[1]=mass_code
            header.attrs.update({'NumPart_ThisFile':counts,'NumPart_Total':counts,'NumPart_Total_HighWord':np.zeros(6,dtype=np.uint32),
                'MassTable':table,'Time':0.,'Redshift':0.,'BoxSize':box_code,'NumFilesPerSnapshot':1,
                'Omega0':units['omega_m0'],'OmegaLambda':units['omega_lambda0'],'HubbleParam':h,
                'Flag_Sfr':0,'Flag_Cooling':0,'Flag_StellarAge':0,'Flag_Metals':0,'Flag_Feedback':0,'Flag_DoublePrecision':1,
                'Research_sidm_canonical_control_schema':1,'Research_sidm_canonical_initial_scale_factor':a,
                'Research_sidm_canonical_clock_SHA256':clock['clock_asset_sha256'],'Research_sidm_physical_model_hash':clock['physical_model_hash'],
                'Research_sidm_original_physical_state_SHA256':population['physical_state_sha256'],
                'Research_sidm_control_request_sha256':canonical_digest(request)})
            params=handle.create_group('Parameters');params.attrs.update({'ComovingIntegrationOn':0,'PeriodicBoundariesOn':1,'HubbleParam':h,
                'UnitLength_in_cm':length,'UnitMass_in_g':mass_unit,'UnitVelocity_in_cm_per_s':speed_unit,'UnitTime_in_s':length/speed_unit,
                'Research_sidm_time_coordinate':'tau_seconds * h / native_UnitTime_in_s','Research_sidm_velocity_coordinate':'u=a*v_pec'})
            group=handle.create_group('PartType1')
            datasets={name:group.create_dataset(name,shape=(count,3) if name in ('Coordinates','Velocities') else (count,),dtype='<u8' if name=='ParticleIDs' else '<f8') for name in ('Coordinates','Velocities','Masses','ParticleIDs')}
            for start in range(0,count,request['chunk_rows']):
                sl=slice(start,min(count,start+request['chunk_rows']))
                for name,value in (('Coordinates',x),('Velocities',u),('Masses',mass)):
                    converted=value[sl]*factors[name]
                    if not np.isfinite(converted).all():raise ArithmeticError('Native mapped particle conversion overflows')
                    datasets[name][sl]=converted
                datasets['ParticleIDs'][sl]=ids[sl]
            for name,factor,hexp in (('Coordinates',length,-1),('Velocities',speed_unit,0),('Masses',mass_unit,-1)):
                datasets[name].attrs.update({'to_cgs':factor,'CGSConversionFactor':factor,'a_scaling':0,'h_scaling':hexp,'aexp-scale-exponent':0,'h-scale-exponent':hexp})
                datasets[name].attrs['research_frame']='canonical x=r/a' if name=='Coordinates' else 'canonical u=a*v_pec' if name=='Velocities' else 'actual original macro mass'
            datasets['ParticleIDs'].attrs['identity_namespace']=population['event_identity_namespace'];handle.flush()
        os.link(clock_temporary,clock_copy);os.link(temporary,native)
    finally:
        for path in (temporary,clock_temporary):
            if path.exists():path.unlink()
    arrays={'particle_ids':ids,'cohort_index':labels,'mass_msun':mass,'position_canonical_kpc':x,'velocity_canonical_kms':u,
        'original_position_proper_kpc':position,'original_velocity_peculiar_kms':velocity}
    digest=hashlib.sha256(canonical_digest({'schema':'sidm-canonical-state-v1','original_physical_state_sha256':population['physical_state_sha256'],
        'initial_scale_factor':a,'canonical_box_kpc':canonical_box,'clock_sha256':clock['clock_asset_sha256']}).encode('ascii'))
    for name,dtype in (('particle_ids','<u8'),('cohort_index','<u8'),('mass_msun','<f8'),('position_canonical_kpc','<f8'),('velocity_canonical_kms','<f8')):
        digest.update(name.encode('ascii'))
        for start in range(0,count,request['chunk_rows']):digest.update(np.asarray(arrays[name][start:start+request['chunk_rows']],dtype=dtype).tobytes(order='C'))
    sources[str(native)]=file_sha256(native);sources[str(clock_copy)]=file_sha256(clock_copy)
    report={'schema':'sidm-canonical-control-population-v1','execution_status':'COMPLETE','request':request,'request_base':str(Path(parameters['request_base']).resolve()),
        'original_population':request['population'],'original_physical_state_sha256':population['physical_state_sha256'],'canonical_state_sha256':digest.hexdigest(),
        'event_identity_namespace':population['event_identity_namespace'],'particle_count':count,'cohorts':cohorts,'phase_seed':population['request']['seed'],
        'initial_scale_factor':a,'canonical_box_kpc':canonical_box,'native_units':units,
        'native_ic':{'path':str(native),'sha256':sources[str(native)],'native_time':0.,'box_size_code':box_code,'equal_macro_mass_code':mass_code,
            'velocity_storage':'canonical u=a*v_pec in static native velocity units'},
        'native_clock':{'path':str(clock_copy),'sha256':sources[str(clock_copy)],'physical_model_hash':clock['physical_model_hash'],
            'original_background_hash':clock['original_background_hash'],'initial_scale_factor':a,'final_scale_factor':clock['actual_scale_factor_end']},
        'evolution_status':'NOT_RUN','scope':'exact original physical population represented in static canonical coordinates and clock; requires a mapped law/kernel source driver, no static unmodified-law run or trajectory acceptance',
        'science_acceptance':'NOT_ESTABLISHED'}
    return _publish(destination,report,arrays,sources,{'particle_ids':'dimensionless','cohort_index':'dimensionless','mass_msun':'Msun','position_canonical_kpc':'canonical kpc',
        'velocity_canonical_kms':'km / s','original_position_proper_kpc':'proper kpc','original_velocity_peculiar_kms':'km / s'},
        {'epoch':{'canonical_time':'tau_seconds','canonical_elapsed_seconds':0.,'actual_scale_factor':a},'coordinates':{'frame':'x=r/a; u=a*v_pec; original physical arrays separately retained'}})


def canonical_population_runner(parameters,context):
    request=parameters['request']
    if request['memory_gib']>context.resources['memory_gib'] or request['disk_gib']>context.resources['disk_gib']:raise MemoryError('Mapped native population exceeds allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Mapped native population escapes owned generation')
    return publish_canonical_population(parameters,destination)


def canonical_population_main():
    import argparse
    parser=argparse.ArgumentParser(description='Materialize exact original expanding SIDM particles as native canonical x,u initial conditions')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text());parameters.setdefault('request_base',str(path.parent))
    publish_canonical_population(parameters,args.output);return 0


def validate_reference_build_request(parameters):
    """A reproducible independent build recipe, never a fabricated capability."""
    request=parameters['request']
    keys={'schema','source_root','source_manifest','configuration','system_makefile','dependency_manifest',
        'tools','toolchain_environment','scheme','particle_convention','jobs','omp_threads','memory_gib','disk_gib','wall_seconds',
        'maximum_source_files','maximum_source_bytes','maximum_build_bytes'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='opengadget3-reference-build-request-v1':
        raise ValueError('Complete pinned independent reference-build recipe required')
    if request['scheme'] not in {'rare','frequent','hybrid'} or request['particle_convention'] not in {'directed','folded_identical'}:
        raise ValueError('Explicit independent operator and particle convention required')
    for key in ('jobs','omp_threads','maximum_source_files','maximum_source_bytes','maximum_build_bytes'):
        if type(request[key])is not int or request[key]<1:raise ValueError('Exact positive reference-build bound required: '+key)
    for key in ('memory_gib','disk_gib','wall_seconds'):_positive(request[key],key)
    if request['jobs']*1024**3>request['memory_gib']*1024**3:
        raise MemoryError('Independent compile reserves at least one GiB per parallel compiler job')
    if not isinstance(request['source_root'],str) or not request['source_root']:raise ValueError('Retained reference source directory required')
    if not isinstance(request['tools'],dict) or set(request['tools'])!={'make','mpi_cxx','perl','python3'}:
        raise ValueError('Pin make, MPI C++ wrapper, Perl and compile-flag Python executable identities')
    environment=request['toolchain_environment']
    allowed_environment={'PATH','LD_LIBRARY_PATH','LIBRARY_PATH','CPATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','OMPI_CXX','MPICH_CXX'}
    if not isinstance(environment,dict) or 'PATH' not in environment or set(environment)-allowed_environment:
        raise ValueError('Declare the bounded compiler/header/library search environment explicitly')
    if any(not isinstance(value,str) or not value or '\n' in value or '\x00' in value for value in environment.values()):
        raise ValueError('Nonempty single-line explicit toolchain environment values required')
    references=[request[key] for key in ('source_manifest','configuration','system_makefile','dependency_manifest')]+list(request['tools'].values())
    for reference in references:
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('Exact independent build asset reference required')
        for key in ('path','sha256'):
            value=reference[key]
            if isinstance(value,dict) and 'stage' in value:continue
            if not isinstance(value,str) or not value:raise ValueError('Empty independent build asset reference')
            if key=='sha256' and (len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('Exact lowercase independent build asset SHA256 required')
    return request


def reference_build_input_bindings(parameters):
    """Bind complete source bytes and explicit external toolchain dependencies."""
    import os,re
    request=validate_reference_build_request(parameters);base=Path(parameters['request_base']).resolve();sources={}
    def pin(reference,*,keep_alias=False):
        # MPI wrapper aliases select language-specific configuration by argv[0].
        # Preserve the requested executable name while hashing its actual bytes.
        path=Path(os.path.abspath(base/reference['path'])) if keep_alias else (base/reference['path']).resolve()
        if file_sha256(path)!=reference['sha256']:raise ValueError('Independent build input changed: '+str(path))
        sources[str(path)]=reference['sha256'];return path
    assets={key:pin(request[key]) for key in ('source_manifest','configuration','system_makefile','dependency_manifest')}
    tools={key:pin(ref,keep_alias=True) for key,ref in request['tools'].items()}
    if any(not os.access(path,os.X_OK) for path in tools.values()):raise ValueError('Pinned build tools must be executable')
    if any(re.fullmatch(r'[A-Za-z0-9_./+-]+',str(path)) is None for path in tools.values()):
        raise ValueError('This GNU-make toolchain profile requires tool paths without shell metacharacters or whitespace')
    for path in assets.values():
        if path.stat().st_size*32>request['memory_gib']*1024**3/4:raise MemoryError('Reference build metadata exceeds allocation')
    manifest=json.loads(assets['source_manifest'].read_text());root=(base/request['source_root']).resolve()
    if manifest.get('schema')!='opengadget3-retained-source-manifest-v1' or manifest.get('revision')!='7d90b03df29fa3c416b2b06c3fa6cc8c7821f1f1' or manifest.get('repository')!='https://gitlab.lrz.de/AstroCodes/OpenGadget3':
        raise ValueError('Pinned retained independent engine revision required')
    files=manifest.get('files_sha256')
    if not isinstance(files,dict) or not files or len(files)>request['maximum_source_files']:
        raise MemoryError('Bounded complete reference source file manifest required')
    size=0;expected=set()
    for relative,checksum in files.items():
        name=Path(relative)
        if name.is_absolute() or '..' in name.parts or relative!=name.as_posix():raise ValueError('Canonical relative reference source path required')
        path=root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():raise ValueError('Reference source requires regular contained files')
        if name.suffix in {'.o','.a','.so','.dylib','.pyc'} or name.name in {'gadgetconfig.hpp','compile_time_info.cpp','OpenGadget3','flagcheck.out','Makefile.local','Makefile.systype'}:
            raise ValueError('Independent source manifest must not include previous build products/local overrides')
        if file_sha256(path)!=checksum:raise ValueError('Retained independent source differs from its manifest')
        expected.add(relative);sources[str(path.resolve())]=checksum;size+=path.stat().st_size
        if size>request['maximum_source_bytes']:raise MemoryError('Independent source copy exceeds explicit byte ceiling')
    actual=set()
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        if '.git' in relative.parts:continue
        if path.is_symlink():raise ValueError('Retained independent source tree contains an unbound symbolic link')
        if path.is_file():
            actual.add(relative.as_posix())
            if len(actual)>request['maximum_source_files']:raise MemoryError('Independent source tree exceeds census ceiling')
    if actual!=expected:raise ValueError('Reference source manifest does not cover the complete retained tree')
    dependencies=json.loads(assets['dependency_manifest'].read_text())
    if dependencies.get('schema')!='native-build-dependency-manifest-v1' or not isinstance(dependencies.get('files_sha256'),dict) or not dependencies['files_sha256']:
        raise ValueError('Explicit MPI/compiler/system header and library dependency manifest required')
    if len(dependencies['files_sha256'])>request['maximum_source_files']:raise MemoryError('Independent build dependency census exceeds its ceiling')
    for name,checksum in dependencies['files_sha256'].items():pin({'path':name,'sha256':checksum})
    configuration=assets['configuration'].read_text();features={};pattern=re.compile(r'^([A-Za-z_][A-Za-z_0-9]*)(?:=([A-Za-z_0-9.+-]+))?$')
    for line in configuration.splitlines():
        line=line.split('#',1)[0].strip()
        if not line:continue
        matched=pattern.fullmatch(line)
        if matched is None or matched[1] in features:raise ValueError('One unambiguous independent compile switch per line required')
        features[matched[1]]=matched[2] or ''
    required={'PERIODIC','NOGRAVITY','HAVE_HDF5','LONGIDS','DOUBLEPRECISION',
              'INPUT_IN_DOUBLEPRECISION','OUTPUT_IN_DOUBLEPRECISION',
              'mSIDM','mSIDM_DIFFERENTIAL_CROSS_SECTION','DCS_USE_ZSLICE'}
    required|={'rSIDM'} if request['scheme']=='rare' else {'fSIDM'} if request['scheme']=='frequent' else {'rSIDM','fSIDM'}
    if request['particle_convention']=='folded_identical':required.add('rSIDM_SANGLE90')
    allowed=required|{'rSIDM_SANGLE90','mSIDM_NOOPENMP','mSIDM_NOTIMESTEPCRIT','mSIDM_NODOACTIVEONCE',
                      'NODELISTLENGTH_default','NODELISTLENGTH_PXP'}
    for name in ('NODELISTLENGTH_default','NODELISTLENGTH_PXP'):
        if name in features and (not features[name].isdigit() or not 1<=int(features[name])<=1024):
            raise ValueError('Bounded positive independent node-list allocation required')
    if not required<=set(features) or set(features)-allowed or features.get('DOUBLEPRECISION')!='1':
        raise ValueError('Independent homogeneous compile profile changes declared physics or storage precision')
    if ('rSIDM_SANGLE90' in features)!=(request['particle_convention']=='folded_identical'):
        raise ValueError('Independent compile profile changes the physical particle convention')
    if size+request['maximum_build_bytes']>request['disk_gib']*1024**3:raise MemoryError('Complete source plus compiler/output allowance exceeds declared disk')
    for owner in (Path(__file__).resolve(),Path(__file__).resolve().with_name('foundation.py'),*sorted((Path(__file__).parent/'research').glob('*.py'))):
        sources[str(owner)]=file_sha256(owner)
    return request,root,manifest,assets,tools,features,sources


def reference_build_runner(parameters,context):
    """Compile only during explicit future execution of the owned workflow stage."""
    import os,shutil,subprocess,sys,re
    from .foundation import atomic_json, atomic_text
    from core.native import run_native_command
    from core.research.capabilities import publish_native_build_capability
    from core.resources import LocalProcessResourceSampler
    request,source_root,manifest,assets,tools,features,sources=reference_build_input_bindings(parameters)
    if not sys.platform.startswith('linux'):raise ValueError('This independent reference build recipe requires a Linux MPI toolchain')
    if request['omp_threads']!=context.resources['omp_threads'] or request['jobs']>context.resources['mpi_ranks']*context.resources['omp_threads']:
        raise ValueError('Independent compiler parallelism exceeds its declared workflow CPU allocation')
    for key in ('memory_gib','disk_gib'):
        if request[key]>context.resources[key]:raise MemoryError('Reference build exceeds its workflow '+key)
    root=Path(parameters['destination']).resolve()
    if root.exists() or not root.is_relative_to(context.generation):raise ValueError('Independent build needs a fresh generation-owned directory')
    root.mkdir(parents=True);work=root/'source';work.mkdir();owned_sources={}
    if shutil.disk_usage(root).free<request['maximum_source_bytes']+request['maximum_build_bytes']:
        raise MemoryError('Insufficient free disk for the independent build allowance')
    for relative,checksum in manifest['files_sha256'].items():
        target=work/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source_root/relative,target)
        if file_sha256(target)!=checksum:raise ValueError('Reference source copy changed')
        owned_sources[str(target)]=checksum
    makefile=work/'Makefile';content=makefile.read_text()
    previous='SHA := $(shell git rev-parse HEAD)';recursive='make -f config-makefile'
    if content.count(previous)!=1 or content.count(recursive)!=1:raise ValueError('Pinned independent Makefile integration points changed')
    content=content.replace(previous,'SHA := '+manifest['revision']).replace(recursive,'$(MAKE) -f config-makefile')
    flagcheck=re.compile(r'(?m)^flagcheck\.out: \$\(CONFIG\)\n\t@if python3 -c[^\n]*\n(?:[^\n]*\n)*?\tfi\n')
    content,replaced=flagcheck.subn('flagcheck.out: $(CONFIG)\n\t$(RESEARCH_PYTHON) Python/check_flags.py $(CONFIG) $(OBJS) $(INCL) $(PM_OBJS)\n',content)
    if replaced!=1:raise ValueError('Pinned independent compile-flag prerequisite changed')
    atomic_text(makefile,content);owned_sources[str(makefile)]=file_sha256(makefile)
    configuration=work/'Config.research.sh';shutil.copyfile(assets['configuration'],configuration)
    system=work/'Build/Makefile.ResearchReference';shutil.copyfile(assets['system_makefile'],system)
    owned_sources[str(configuration)]=file_sha256(configuration);owned_sources[str(system)]=file_sha256(system)
    binary=work/'OpenGadget3';receipt=root/'compile-invocation.json';log=root/'compile.log'
    command=[str(tools['make']),'-j'+str(request['jobs']),'SYSTYPE=ResearchReference','CONFIG=Config.research.sh',
        'EXEC=OpenGadget3','CC='+str(tools['mpi_cxx']),'CXX='+str(tools['mpi_cxx']),'PERL='+str(tools['perl']),
        'RESEARCH_PYTHON='+str(tools['python3']),'OpenGadget3']
    # Configuration preprocessing in upstream uses a recursive make. Pin it
    # through $(MAKE), and retain the exact effective patched Makefile bytes.
    environment=dict(request['toolchain_environment'])
    environment.update(LANG='C',LC_ALL='C',OMP_NUM_THREADS=str(request['omp_threads']),OPENBLAS_NUM_THREADS='1')
    environment_path=root/'toolchain-environment.json';atomic_json(environment_path,environment)
    owned_sources[str(environment_path)]=file_sha256(environment_path)
    sampler=LocalProcessResourceSampler(root/'build-resource-samples.json')
    inputs=[Path(path) for path in {**sources,**owned_sources}]
    with log.open('wb') as stream:
        run_native_command(command,cwd=work,receipt_path=receipt,inputs=inputs,timeout=request['wall_seconds'],
            env=environment,stdout=stream,stderr=subprocess.STDOUT,process_observer=sampler)
    for path,checksum in owned_sources.items():
        if file_sha256(path)!=checksum:raise ValueError('Independent source changed during compilation')
    generated=work/'gadgetconfig.hpp'
    if not binary.is_file() or not os.access(binary,os.X_OK) or not generated.is_file():raise ValueError('Independent compile did not produce executable/configuration outputs')
    generated_features={}
    for line in generated.read_text().splitlines():
        matched=re.fullmatch(r'\s*#define\s+([A-Za-z_][A-Za-z_0-9]*)(?:\s+(.*?))?\s*',line)
        if matched and matched[1]!='CONFIG_FILE_NAME':generated_features[matched[1]]=matched[2] or ''
    if generated_features!=features:raise ValueError('Actual generated independent compile switches differ from the frozen profile')
    size=sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
    original_size=sum((source_root/name).stat().st_size for name in manifest['files_sha256'])
    if size-original_size>request['maximum_build_bytes'] or size>request['disk_gib']*1024**3:
        raise MemoryError('Independent source/build publication exceeds its declared byte/disk allocation')
    capability=root/'build-capability.json';profile={'engine':'opengadget3','revision':manifest['revision'],
        'scheme':request['scheme'],'particle_convention':request['particle_convention'],'configuration':features,
        'source_manifest_sha256':request['source_manifest']['sha256'],'dependency_manifest_sha256':request['dependency_manifest']['sha256'],
        'toolchain_environment_sha256':file_sha256(environment_path)}
    publish_native_build_capability(capability,name='opengadget3-homogeneous-'+request['scheme'],executable=binary,
        compiled_features=list(features),profile=profile,source_hashes={**sources,**owned_sources},invocation_receipt=receipt,
        domain={'boundary':'periodic','gravity':'disabled','species':'one equal-mass elastic DM species','precision':'double','identities':'uint64'})
    report={'schema':'opengadget3-reference-build-v1','execution_status':'COMPLETE','request':request,
        'capability':{'path':str(capability),'sha256':file_sha256(capability)},
        'configuration':{'path':str(configuration),'sha256':file_sha256(configuration)},
        'generated_configuration':{'path':str(generated),'sha256':file_sha256(generated)},
        'toolchain_environment':environment,'source_hashes':{**sources,**owned_sources},'science_acceptance':'NOT_ESTABLISHED_BY_BUILD'}
    atomic_json(root/'reference-build.json',report);return report


def reference_build_main():
    import argparse
    from types import SimpleNamespace
    from core.native import native_run_lock
    parser=argparse.ArgumentParser(description='Explicitly build the pinned independent OpenGadget3 homogeneous SIDM reference engine')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.parameters.resolve();parameters=json.loads(source.read_text());request=parameters['request']
    if args.output.exists():raise FileExistsError('Independent build publication cannot replace previous source/build evidence')
    with native_run_lock(args.output,operation='independent OpenGadget3 reference compile'):
        parameters.setdefault('request_base',str(source.parent));parameters['destination']=str(args.output.resolve()/'reference-build')
        context=SimpleNamespace(generation=args.output.resolve(),resources={key:request[key] for key in ('omp_threads','memory_gib','disk_gib')}|{'mpi_ranks':request['jobs']})
        reference_build_runner(parameters,context)
    return 0


def _canonical_source_step_diagnostics(root,values,limits,sources):
    """Read actual completed source intervals, including zero-event steps."""
    import h5py
    records=[];total_bytes=0
    for path in (Path(root)/'data').glob('sidm_events_*.hdf5'):
        if len(records)>=limits['maximum_intervals']:raise MemoryError('Actual source interval census exceeds allocation')
        if path.is_symlink() or not path.is_file():raise ValueError('Canonical source diagnostics require owned regular files')
        size=path.stat().st_size;total_bytes+=size
        if size>limits['maximum_file_bytes']:raise MemoryError('Native source log exceeds byte allocation')
        checksum=file_sha256(path);sources[str(path.resolve())]=checksum
        with h5py.File(path,'r') as handle:
            if _event_text(handle.attrs.get('publication_state',''))!='COMPLETE':raise ValueError('Incomplete canonical source interval log')
            group=handle['SIDMScatteringEvents'];attrs=group.attrs
            for name,expected in (('schema','native-sidm-event-log-v1'),('angular_model_hash',values['SIDMAngularModelHash']),
                ('event_identity_namespace',values['SIDMEventIdentityNamespace']),('native_clock_coordinate','canonical_native_time_code'),
                ('canonical_clock_sha256',values['SIDMCanonicalClockSHA256'])):
                if _event_text(attrs.get(name,''))!=expected:raise ValueError('Actual canonical source interval identity differs: '+name)
            fields=('native_interval_begin','native_interval_end','source_interval_seconds','smallest_source_step_seconds',
                'largest_source_step_seconds','physical_scale_factor_begin','physical_scale_factor_end','frozen_physical_scale_factor')
            row={name:float(attrs[name]) for name in fields}
            if not all(math.isfinite(value) for value in row.values()):raise ValueError('Nonfinite actual source interval diagnostics')
            row['source_epoch_policy']=_event_text(attrs.get('canonical_source_epoch_policy','whole_interval_midpoint'))
            if row['source_epoch_policy'] not in {'whole_interval_midpoint','source_substep_left_endpoint'}:raise ValueError('Unknown realized canonical source epoch policy')
            if row['source_epoch_policy']=='source_substep_left_endpoint':
                row['source_expansion_policy']=_event_text(attrs.get('canonical_source_expansion_policy','whole_native_interval_gate'))
                if row['source_expansion_policy'] not in {'whole_native_interval_gate','adaptive_substep_span_with_inverse_brackets'}:raise ValueError('Unknown realized source expansion policy')
                for name in ('canonical_source_clock_function_evaluations','canonical_largest_substep_scale_fraction',
                        'canonical_substep_maximum_clock_error_seconds','canonical_substep_maximum_inverse_bracket_relative_width'):
                    value=float(attrs[name])
                    if not math.isfinite(value) or value<0:raise ValueError('Nonfinite/negative realized canonical substep clock diagnostics')
                    row[name]=value
                work=row['canonical_source_clock_function_evaluations']
                if not work.is_integer() or work>int(values['SIDMCanonicalMaximumClockWork']):raise ValueError('Realized canonical clock work exceeds its native bound')
                span_limit=float(values['SIDMReferenceMaximumScaleFactorFractionalChange'])
                if row['source_expansion_policy']=='adaptive_substep_span_with_inverse_brackets' and row['canonical_largest_substep_scale_fraction']>span_limit*(1+32*np.finfo(float).eps):
                    raise ValueError('Realized canonical substep expansion span exceeds its native allocation')
            count=_event_text(attrs['source_substep_count'])
            if not count.isdigit() or not 1<=int(count)<=int(values['SIDMReferenceMaximumSubsteps']):raise ValueError('Actual source substep count exceeds its native allocation')
            row['substep_count']=int(count);row['path']=str(path.resolve());row['sha256']=checksum
            begin,end=row['native_interval_begin'],row['native_interval_end'];duration=end-begin
            smallest,largest=row['smallest_source_step_seconds'],row['largest_source_step_seconds'];seconds=row['source_interval_seconds']
            if begin<0 or duration<=0 or not 0<smallest<=largest<=seconds or row['physical_scale_factor_end']<row['physical_scale_factor_begin'] or row['physical_scale_factor_begin']<=0:
                raise ValueError('Actual source interval/substep extrema are inconsistent')
            unit=float(values['UnitLength_in_cm'])/float(values['UnitVelocity_in_cm_per_s'])/float(values['HubbleParam'])
            tolerance=limits['time_tolerance_native']
            if abs(seconds/unit-duration)>tolerance or duration>float(values['MaxSizeTimestep'])+tolerance:
                raise ValueError('Realized canonical native source interval differs from its actual clock/timestep bound')
            if row['substep_count']*smallest>seconds*(1+1e-12) or row['substep_count']*largest<seconds*(1-1e-12):
                raise ValueError('Source extrema cannot cover the declared source duration')
            row['native_duration']=duration;records.append(row)
    if not records:raise ValueError('Mapped native run lacks actual source intervals')
    records.sort(key=lambda row:row['native_interval_begin']);tolerance=limits['time_tolerance_native']
    if abs(records[0]['native_interval_begin']-float(values['TimeBegin']))>tolerance or abs(records[-1]['native_interval_end']-float(values['TimeMax']))>tolerance:
        raise ValueError('Source interval logs do not cover the completed native run')
    for left,right in zip(records,records[1:]):
        if abs(left['native_interval_end']-right['native_interval_begin'])>tolerance or right['native_interval_begin']<=left['native_interval_begin']:
            raise ValueError('Actual source intervals overlap, repeat or omit native evolution')
    return {'schema':'sidm-canonical-source-step-diagnostics-v1','execution_status':'COMPLETE','interval_count':len(records),
        'maximum_native_duration':max(row['native_duration'] for row in records),
        'minimum_native_duration':min(row['native_duration'] for row in records),
        'largest_source_step_seconds':max(row['largest_source_step_seconds'] for row in records),
        'smallest_source_step_seconds':min(row['smallest_source_step_seconds'] for row in records),
        'source_substep_count_decimal':str(sum(row['substep_count'] for row in records)),
        'maximum_relative_scale_span':max(row['physical_scale_factor_end']/row['physical_scale_factor_begin']-1 for row in records),
        'records':records,'retained_log_bytes':total_bytes,'time_tolerance_native':tolerance,
        'scope':'complete owned canonical source intervals and actual source-substep extrema; no trajectory/accuracy acceptance'}


def validate_control_run_request(parameters):
    """Declare a native homogeneous transport experiment, never run it here."""
    request=parameters['request']
    keys={'schema','engine','population','base_configuration','sidm','reference_build',
        'output_times_native','maximum_timestep_native','minimum_timestep_native','softening_proper_kpc',
        'output_time_absolute_tolerance_native','maximum_particles','maximum_snapshots','maximum_snapshot_bytes',
        'mpi_ranks','omp_threads','memory_gib','disk_gib','wall_seconds'}
    canonical=isinstance(request,dict) and request.get('schema')=='sidm-canonical-native-run-request-v1'
    if canonical:keys.add('canonical_clock_limits')
    if isinstance(request,dict) and 'source_diagnostic_limits' in request:keys.add('source_diagnostic_limits')
    if not isinstance(request,dict) or set(request)!=keys or request['schema'] not in {'sidm-homogeneous-native-run-request-v1','sidm-canonical-native-run-request-v1'}:
        raise ValueError('Complete dedicated homogeneous native evolution request required')
    if request['engine'] not in {'gadget4','opengadget3'}:raise ValueError('Dedicated native or independent engine required')
    if canonical:
        if request['engine']!='gadget4' or request['mpi_ranks']!=1 or request['reference_build']is not None:
            raise ValueError('Mapped native controls currently require the Gadget4 one-rank source profile')
        limits=request['canonical_clock_limits']
        if not isinstance(limits,dict) or set(limits)!={'maximum_asset_bytes','maximum_rows','maximum_clock_work'}:
            raise ValueError('Mapped native clock needs explicit byte/row/aggregate-work allocations')
        for key,minimum in (('maximum_asset_bytes',128),('maximum_rows',2),('maximum_clock_work',1)):
            if type(limits[key])is not int or not minimum<=limits[key]<=2147483647:
                raise ValueError('Mapped clock allocation must fit the native signed parameter domain: '+key)
    for key in ('maximum_particles','maximum_snapshots','maximum_snapshot_bytes','mpi_ranks','omp_threads'):
        if type(request[key])is not int or request[key]<1:raise ValueError('Exact positive native run bound required: '+key)
    for key in ('maximum_timestep_native','softening_proper_kpc','memory_gib','disk_gib','wall_seconds'):_positive(request[key],key)
    if canonical and limits['maximum_asset_bytes']+limits['maximum_rows']*16>request['memory_gib']*1024**3*.125:
        raise MemoryError('Mapped clock asset workspace exceeds its native memory reserve')
    for key in ('minimum_timestep_native','output_time_absolute_tolerance_native'):_positive(request[key],key,True)
    if request['minimum_timestep_native']>request['maximum_timestep_native']:raise ValueError('Native timestep bounds are reversed')
    if 'source_diagnostic_limits' in request:
        limits_source=request['source_diagnostic_limits']
        if not canonical or not isinstance(limits_source,dict) or set(limits_source)!={'maximum_intervals','maximum_file_bytes','memory_gib','time_tolerance_native'}:
            raise ValueError('Explicit canonical source diagnostic allocations required')
        for name in ('maximum_intervals','maximum_file_bytes'):
            if type(limits_source[name])is not int or limits_source[name]<1:raise ValueError('Exact positive source diagnostic allocation required')
        for name in ('memory_gib','time_tolerance_native'):_positive(limits_source[name],name)
        if limits_source['memory_gib']>request['memory_gib'] or limits_source['maximum_intervals']*4096>limits_source['memory_gib']*1024**3/2:
            raise MemoryError('Source interval metadata exceeds its explicit workspace')
    times=request['output_times_native']
    deferred=canonical and isinstance(times,dict) and set(times)=={'stage','artifact','load','field'} and times.get('load')=='json' and times.get('field')==['native_output_times']
    if not deferred:
        if not isinstance(times,list) or not 2<=len(times)<=request['maximum_snapshots']:raise ValueError('Complete bounded native output schedule required')
        for value in times:_positive(value,'native saved epoch',True)
        if any(right<=left for left,right in zip(times,times[1:])):raise ValueError('Native output epochs must increase')
        if canonical and times[0]!=0:raise ValueError('Mapped native output schedule starts at tau=0')
    if not isinstance(request['sidm'],dict) or request['sidm'].get('enabled')is not True:
        raise ValueError('Dedicated transport runner needs explicit complete enabled SIDM physics')
    if canonical and request['sidm'].get('backend')!='all_pair_reference':
        raise ValueError('Mapped native request must select the all-pair source backend explicitly')
    if request['engine']=='gadget4' and request['reference_build']is not None:raise ValueError('Gadget4 builds use the bundled source/profile')
    if request['engine']=='opengadget3':
        reference=request['reference_build']
        if not isinstance(reference,dict) or set(reference)!={'capability','configuration'}:raise ValueError('Independent run needs actual pinned build capability and configuration')
        # Public OpenGadget3 currently stores this seed as a signed native int.
        seed=request['sidm'].get('operator_seed')
        if not isinstance(seed,(str,int)) or isinstance(seed,bool) or not str(seed).isdigit() or not 0<=int(seed)<=2147483647:
            raise ValueError('Independent native SIDM seed must fit its actual signed integer domain')
    if request['maximum_snapshot_bytes']>request['disk_gib']*1024**3:raise MemoryError('Snapshot allowance exceeds native stage disk allocation')
    return request


def control_run_input_bindings(parameters):
    from .sidm_analysis import _load_product
    request=validate_control_run_request(parameters);base=Path(parameters['request_base']);sources={}
    if int(.85*request['memory_gib']*1024/request['mpi_ranks'])<512:
        raise MemoryError('Native control needs at least 512 MiB per rank after the declared reserve')
    population,_,binding=_load_product(request['population'],base,request['memory_gib']*1024**3/2);sources.update(binding)
    canonical=request['schema']=='sidm-canonical-native-run-request-v1'
    expected_schema='sidm-canonical-control-population-v1' if canonical else 'sidm-control-population-v1'
    if population.get('schema')!=expected_schema:raise ValueError('Native control requires its actual declared physical or canonical population')
    count=sum(row['particle_count'] for row in population['cohorts'])
    if count>request['maximum_particles']:raise MemoryError('Original native population exceeds the run particle bound')
    if request['output_times_native'][0]!=population['native_ic']['native_time']:
        raise ValueError('Native output schedule must begin at the original population epoch')
    if not canonical and population['request']['clock']['mode']=='comoving' and request['output_times_native'][-1]>1:
        raise ValueError('This native control expansion ends at or before scale factor one')
    box=population['canonical_box_kpc']*population['initial_scale_factor'] if canonical else population['request']['box_proper_kpc']
    if request['softening_proper_kpc']>=box/2:
        raise ValueError('Control softening must fit the original periodic region')
    references=[request['base_configuration'],request['sidm']['physical_model']]
    if request['reference_build']is not None:references.extend(request['reference_build'].values())
    for reference in references:
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('Exact native control asset reference required')
        path=(base/reference['path']).expanduser().resolve()
        if file_sha256(path)!=reference['sha256']:raise ValueError('Native control source asset changed')
        if path.stat().st_size*16>request['memory_gib']*1024**3:raise MemoryError('Native control source metadata exceeds allocation')
        sources[str(path)]=reference['sha256']
    from .sidm_physics import sidm_physical_model
    from copy import deepcopy
    physical=deepcopy(request['sidm'])
    physical['physical_model']['path']=str((base/physical['physical_model']['path']).resolve())
    sidm_physical_model(physical)
    if canonical:
        clock=population['native_clock'];asset=Path(clock['path']).resolve();limits=request['canonical_clock_limits']
        if request['sidm'].get('event_identity_namespace')!=population['event_identity_namespace']:
            raise ValueError('Mapped native events must preserve the original exact-ID population namespace')
        if request['sidm'].get('maximum_scale_factor_fractional_change',1)>=1:
            raise ValueError('Mapped native frozen-law expansion span must be smaller than one')
        units=population['native_units']
        for name in ('length_cm','mass_g','velocity_cm_s','hubble_parameter'):_positive(units[name],'mapped native '+name)
        if asset.stat().st_size>limits['maximum_asset_bytes'] or file_sha256(asset)!=clock['sha256']:
            raise ValueError('Mapped native run clock asset changed or exceeds allocation')
        tokens=asset.read_text(encoding='ascii').split()
        if len(tokens)<17 or tokens[0]!='RESEARCH_SIDM_CANONICAL_CLOCK_V1' or tokens[1]!=clock['physical_model_hash'] or tokens[2]!=clock['original_background_hash']:
            raise ValueError('Mapped native clock payload lost its physical-model/background identity')
        rows=int(tokens[12])
        if rows<0 or rows>limits['maximum_rows'] or len(tokens)!=17+2*rows:
            raise ValueError('Mapped native clock row allocation or payload length invalid')
        if float(tokens[5])!=population['initial_scale_factor'] or float(tokens[6])!=clock['final_scale_factor']:
            raise ValueError('Mapped native population changes the clock physical epoch domain')
        model=json.loads((base/request['sidm']['physical_model']['path']).read_text())
        if canonical_digest(model)!=clock['physical_model_hash']:
            raise ValueError('Mapped native run changes the original clock-bound physical law')
        if count!=population['particle_count'] or population['initial_scale_factor']!=clock['initial_scale_factor']:
            raise ValueError('Mapped native population census or initial epoch inconsistent')
        sources[str(asset)]=clock['sha256']
        for path,checksum in population.get('source_hashes',{}).items():
            if file_sha256(path)!=checksum:raise ValueError('Mapped population source lineage changed')
            sources[path]=checksum
    original=Path(population['native_ic']['path']).resolve()
    if file_sha256(original)!=population['native_ic']['sha256']:raise ValueError('Original native IC changed')
    sources[str(original)]=population['native_ic']['sha256']
    if request['reference_build']is not None:
        from core.native import verify_invocation_inputs
        capability_path=(base/request['reference_build']['capability']['path']).resolve()
        capability=json.loads(capability_path.read_text())
        if capability.get('schema')!='native-build-capability-v1' or capability.get('execution_status')!='COMPLETE':
            raise ValueError('Independent transport requires an actual completed build capability')
        executable=Path(capability['executable']).resolve()
        if file_sha256(executable)!=capability['executable_sha256']:raise ValueError('Independent executable differs from its actual build capability')
        receipt=capability['invocation_receipt'];invocation_path=Path(receipt['path']).resolve()
        if file_sha256(invocation_path)!=receipt['sha256']:raise ValueError('Independent compile invocation changed')
        invocation=json.loads(invocation_path.read_text())
        if invocation.get('status')!='RETURNED' or invocation.get('returncode')!=0:raise ValueError('Independent compile did not finish successfully')
        verify_invocation_inputs(invocation)
        configuration=(base/request['reference_build']['configuration']['path']).resolve()
        if str(configuration) not in {**capability['source_hashes'],**invocation['input_hashes']}:
            raise ValueError('Reference compile does not bind its actual supplied build configuration')
        for path,checksum in capability['source_hashes'].items():
            if file_sha256(path)!=checksum:raise ValueError('Independent build source changed')
            sources[path]=checksum
        sources.update(invocation['input_hashes']);sources[str(invocation_path)]=receipt['sha256'];sources[str(executable)]=capability['executable_sha256']
    return request,population,sources


def homogeneous_control_runner(parameters,context):
    """Build/materialize and evolve the actual sealed one-species IC population.

    This function runs only when the workflow executes the declared stage. It
    does not replace native trajectories with Python evolution or expectations.
    """
    import os,shutil,h5py,yaml
    from copy import deepcopy
    from .foundation import atomic_json, atomic_text, H100_S, KPC_CM
    from core.native import run_native_command
    from .evolution import build_gadget,gadget_param_text,parameter_values,gadget_ic_prefix
    from .sidm_physics import opengadget3_model_parameters
    request,population,sources=control_run_input_bindings(parameters);base=Path(parameters['request_base'])
    canonical=request['schema']=='sidm-canonical-native-run-request-v1'
    for name in ('mpi_ranks','omp_threads'):
        if request[name]!=context.resources[name]:raise ValueError('Control run changes frozen '+name)
    for name in ('memory_gib','disk_gib'):
        if request[name]>context.resources[name]:raise MemoryError('Control run exceeds frozen '+name)
    root=Path(parameters['destination']).resolve()
    if not root.is_relative_to(context.generation) or root.exists():raise ValueError('Control evolution needs a fresh owned run directory')
    root.mkdir(parents=True);(root/'input').mkdir();(root/'data').mkdir()
    if shutil.disk_usage(root).free<request['maximum_snapshot_bytes']+Path(population['native_ic']['path']).stat().st_size:
        raise MemoryError('Insufficient free disk for the frozen native transport allowance')
    initial=root/'input/control.hdf5';shutil.copyfile(population['native_ic']['path'],initial)
    if file_sha256(initial)!=population['native_ic']['sha256']:raise ValueError('Owned native IC copy differs from the sealed population')
    owned_clock=None
    if canonical:
        owned_clock=root/'input/canonical-clock.txt';shutil.copyfile(population['native_clock']['path'],owned_clock)
        if file_sha256(owned_clock)!=population['native_clock']['sha256']:raise ValueError('Owned mapped clock differs from its sealed population')
    model_ref=request['sidm']['physical_model'];model_path=(base/model_ref['path']).resolve();model=json.loads(model_path.read_text())
    owned_model=root/'input/physical-model.json';shutil.copyfile(model_path,owned_model)
    if file_sha256(owned_model)!=model_ref['sha256']:raise ValueError('Owned physical model copy differs from its frozen asset')
    settings=deepcopy(request['sidm']);settings['physical_model']={'path':str(owned_model),'sha256':model_ref['sha256']}
    units=population['native_units'] if canonical else population['request']['native_units']
    h=units['hubble_parameter'];length=units['length_cm'];velocity=units['velocity_cm_s']
    time_unit=length/velocity;comoving=False if canonical else population['request']['clock']['mode']=='comoving'
    a=population['initial_scale_factor'] if canonical else population['request']['clock']['scale_factor']
    configuration=(base/request['base_configuration']['path']).resolve();environment=os.environ.copy();environment['OMP_NUM_THREADS']=str(request['omp_threads'])
    environment.pop('RESEARCH_SIDM_REFERENCE_TABLE',None)
    if request['engine']=='gadget4':
        cfg=deepcopy(yaml.safe_load(configuration.read_text()))
        if cfg['cosmology'].get('model')!='lcdm' or cfg['cosmology'].get('gadget_background_table'):
            raise ValueError('Homogeneous control runner requires the explicitly supported analytic LCDM clock')
        for name in ('chemistry','black_holes','stellar_evolution','wdm'):
            if cfg.get(name,{}).get('enabled',False):raise ValueError('Single-species homogeneous transport excludes enabled '+name)
        cfg['hardware'].update(mpi_ranks=request['mpi_ranks'],omp_threads=request['omp_threads'],memory_limit_gb=request['memory_gib'],
            cpu_threads=request['mpi_ranks']*request['omp_threads'],build_jobs=request['mpi_ranks'])
        cfg['simulation'].update(physics='dm',num_files_per_snapshot=1,softening_kpc_h=request['softening_proper_kpc']*h/a,
            softening_max_physical_kpc_h=request['softening_proper_kpc']*h,restart_every_seconds=request['wall_seconds'],time_limit_cpu_seconds=request['wall_seconds'])
        cfg['cosmology']['H0']=100*h;cfg['halos'].update(fof=False,subfind=False);cfg['sidm']=settings
        profile={'mode':'homogeneous_sidm_canonical' if canonical else 'homogeneous_sidm_physical','sidm_enabled':True}
        binary=build_gadget(cfg,root,control_profile=profile)
        values={line.split()[0]:' '.join(line.split()[1:]) for line in gadget_param_text(cfg,root,initial).splitlines() if line.strip() and not line.lstrip().startswith(('%','#'))}
        values.update(Hubble=H100_S*time_unit,OmegaRadiation=0.,CosmologyTableFile='none',NativeControlPauseStep=0)
        values.pop('ActivePartFracForPMinsteadOfEwald',None)
        values.update(SofteningClassOfPartType1=1,SofteningComovingClass1=request['softening_proper_kpc']*h*KPC_CM/(a*length),
            SofteningMaxPhysClass1=request['softening_proper_kpc']*h*KPC_CM/length)
        if canonical:
            limits=request['canonical_clock_limits']
            values.update(SIDMCanonicalClockPath=str(owned_clock),SIDMCanonicalClockSHA256=population['native_clock']['sha256'],
                SIDMCanonicalPhysicalStateSHA256=population['original_physical_state_sha256'],
                SIDMCanonicalInitialScaleFactor=a,SIDMCanonicalMaximumAssetBytes=limits['maximum_asset_bytes'],
                SIDMCanonicalMaximumRows=limits['maximum_rows'],SIDMCanonicalMaximumClockWork=limits['maximum_clock_work'])
        atomic_json(root/'configuration.json',cfg)
        table=None
    else:
        capability=json.loads((base/request['reference_build']['capability']['path']).read_text());features=set(capability['compiled_features'])
        if not {'NOGRAVITY','PERIODIC'}<=features or any(name.startswith(('PMGRID','EXTERNALGRAVITY','DARKENERGY','TIMEDEPDE')) for name in features):
            raise ValueError('Independent control requires an actual periodic gravity-free analytic-clock build')
        if features & {'COOLING','SFR','BLACK_HOLES','HYDRO','MHD'}:raise ValueError('Independent homogeneous control has unrelated compiled source physics')
        compiled_scheme='hybrid' if {'rSIDM','fSIDM'}<=features else 'rare' if 'rSIDM' in features else 'frequent' if 'fSIDM' in features else None
        if compiled_scheme!=settings['scheme']:raise ValueError('Actual independent build evolves another transport scheme')
        binary=root/'input/OpenGadget3';shutil.copyfile(capability['executable'],binary);binary.chmod(0o755)
        if file_sha256(binary)!=capability['executable_sha256']:raise ValueError('Owned independent executable changed during materialization')
        atomic_text(root/'input/Config.sh','\n'.join(sorted(features))+'\n')
        shutil.copyfile(base/request['reference_build']['configuration']['path'],root/'input/reference-build-configuration.txt')
        atomic_json(root/'input/reference-build-capability.json',capability)
        values=parameter_values(configuration)
        table=None
        if model.get('schema')=='sidm-elastic-model-v2' and model.get('kind') not in {'constant_isotropic','screened_rutherford'}:
            from .sidm_physics import ElasticSIDMModel
            table=ElasticSIDMModel(model).to_native_table(root/'input/physical-model.native-sidm-table.dat')
        elif model.get('schema')=='sidm-elastic-model-v2' and model.get('kind')=='screened_rutherford' and model.get('supplied_cross_section')!='total':
            from .sidm_physics import ElasticSIDMModel
            table=ElasticSIDMModel(model).to_native_table(root/'input/physical-model.native-sidm-table.dat')
        mapped=opengadget3_model_parameters(model,velocity_unit_kms=velocity/1e5,split_angle_radians=settings['split_angle_radians'],
            scheme=settings['scheme'],retained_differential_table=None if table is None else {'path':table['path'],'sha256':table['sha256']})
        if not set(mapped['required_reference_build_features'])<=features:raise ValueError('Actual independent build cannot execute the requested physical operator')
        values.update({name:mapped[name] for name in ('MSIDM_CrossSection','MSIDM_DCS_params','MSIDM_thetaC_fwd','MSIDM_thetaC_bwd')})
        values['MSIDM_SEED']=int(settings['operator_seed']);environment.update(mapped['environment'])
        values.pop('OmegaRadiation',None);values.pop('CosmologyTableFile',None)
        values.update(SofteningHalo=request['softening_proper_kpc']*h*KPC_CM/(a*length),
            SofteningHaloMaxPhys=request['softening_proper_kpc']*h*KPC_CM/length)
    times=request['output_times_native']
    values.update(InitCondFile=gadget_ic_prefix(initial),OutputDir=str(root/'data')+'/',SnapshotFileBase='snapshot',
        OutputListFilename=str(root/'input/output_times.txt'),OutputListOn=1,ICFormat=3,SnapFormat=3,NumFilesPerSnapshot=1,
        TimeBegin=times[0],TimeMax=times[-1],TimeOfFirstSnapshot=times[0],ComovingIntegrationOn=int(comoving),
        BoxSize=population['native_ic']['box_size_code'],HubbleParam=h,Omega0=units['omega_m0'],OmegaLambda=units['omega_lambda0'],OmegaBaryon=0.,
        UnitLength_in_cm=length,UnitMass_in_g=units['mass_g'],UnitVelocity_in_cm_per_s=velocity,
        MaxSizeTimestep=request['maximum_timestep_native'],MinSizeTimestep=request['minimum_timestep_native'],
        TimeLimitCPU=request['wall_seconds'],CpuTimeBetRestartFile=request['wall_seconds'],
        GravityConstantInternal=0,
        MaxMemSize=int(.85*request['memory_gib']*1024/request['mpi_ranks']))
    # The pinned independent engine stores these filenames in char[100]. Its
    # cwd is the owned input directory, so short relative paths preserve the
    # exact same assets and work for long external output roots.
    if request['engine']=='opengadget3':
        values.update(InitCondFile=Path(gadget_ic_prefix(initial)).name,
                      OutputDir='../data/',OutputListFilename='output_times.txt')
    atomic_text(root/'input/param.txt','\n'.join(str(key)+' '+str(value) for key,value in values.items())+'\n')
    atomic_text(root/'input/output_times.txt','\n'.join(format(value,'.17g')+' 1' for value in times)+'\n')
    inputs=[initial,owned_model,root/'input/param.txt',root/'input/Config.sh',root/'input/output_times.txt',configuration]
    if owned_clock is not None:inputs.append(owned_clock)
    if table is not None:inputs.append(Path(table['path']))
    if values.get('SIDMNativeTablePath','none')!='none':inputs.append(Path(values['SIDMNativeTablePath']))
    if request['engine']=='opengadget3':inputs.extend([root/'input/reference-build-capability.json',root/'input/reference-build-configuration.txt'])
    with (root/'native.log').open('wb') as log:
        parameter_argument='param.txt' if request['engine']=='opengadget3' else str(root/'input/param.txt')
        run_native_command(['mpirun','-np',str(request['mpi_ranks']),str(binary),parameter_argument,'0'],
            cwd=root/'input',receipt_path=root/'native-invocation.json',inputs=inputs,timeout=request['wall_seconds'],env=environment,stdout=log,stderr=log)
    files=sorted((root/'data').glob('snapshot_*.hdf5'))
    if len(files)!=len(times):raise ValueError('Native control did not retain exactly the frozen saved output schedule')
    if sum(path.stat().st_size for path in files)>request['maximum_snapshot_bytes']:raise MemoryError('Actual native snapshots exceed the frozen disk allowance')
    if sum(path.stat().st_size for path in root.rglob('*') if path.is_file())>request['disk_gib']*1024**3:
        raise MemoryError('Actual native control products exceed the frozen stage disk allowance')
    rows=[]
    for path,expected in zip(files,times):
        with h5py.File(path,'r') as handle:
            actual=float(handle['Header'].attrs['Time'])
            if not math.isfinite(actual) or abs(actual-expected)>request['output_time_absolute_tolerance_native']:
                raise ValueError('Native control missed a requested output epoch')
            if canonical:
                header=handle['Header'].attrs
                for name,value in {'Research_sidm_canonical_control_schema':1,'Research_sidm_canonical_initial_scale_factor':a,
                        'Research_sidm_canonical_clock_SHA256':population['native_clock']['sha256'],
                        'Research_sidm_physical_model_hash':population['native_clock']['physical_model_hash'],
                        'Research_sidm_original_physical_state_SHA256':population['original_physical_state_sha256'],
                        'Research_sidm_position_frame':'canonical_x=r/a','Research_sidm_velocity_frame':'canonical_u=a*v_pec',
                        'Research_sidm_clock_frame':'tau_integral_dt_over_a_squared'}.items():
                    actual_value=header.get(name)
                    if isinstance(actual_value,bytes):actual_value=actual_value.decode('ascii')
                    if actual_value!=value:raise ValueError('Mapped native snapshot lost canonical lineage/frame: '+name)
        rows.append({'files':[str(path.resolve())],'native_time':actual})
    for path,checksum in sources.items():
        if file_sha256(path)!=checksum:raise ValueError('Frozen control input changed while the native engine ran')
    report={'schema':'sidm-canonical-native-run-v1' if canonical else 'sidm-homogeneous-native-run-v1','execution_status':'COMPLETE','engine':request['engine'],'request':request,
        'population':request['population'],'physical_state_sha256':population['original_physical_state_sha256'] if canonical else population['physical_state_sha256'],'snapshots':rows,
        'parameter_file':str(root/'input/param.txt'),'build_configuration':str(root/'input/Config.sh'),
        'invocation_receipt':str(root/'native-invocation.json'),'native_run_directory':str(root),
        'physical_model':{'path':str(owned_model),'sha256':file_sha256(owned_model)},
        'native_differential_table':({'path':values['SIDMNativeTablePath'],'sha256':values['SIDMNativeTableSHA256']}
            if request['engine']=='gadget4' and values.get('SIDMNativeTablePath','none')!='none' else
            None if table is None else {'path':table['path'],'sha256':table['sha256']}),
        'native_differential_table_metadata':table,
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    if canonical:
        report.update(request_base=str(base.resolve()),canonical_state_sha256=population['canonical_state_sha256'],initial_scale_factor=a,
            owned_native_ic={'path':str(initial),'sha256':file_sha256(initial)},
            event_identity_namespace=population['event_identity_namespace'],
            native_clock={'path':str(owned_clock),'sha256':file_sha256(owned_clock),
                'physical_model_hash':population['native_clock']['physical_model_hash'],
                'original_background_hash':population['native_clock']['original_background_hash']},
            frame={'positions':'x=r/a','velocities':'u=a*v_pec','native_time':'tau_seconds*h/UnitTime'},
            scope='actual mapped native control execution and retained outputs; physical history/equivalence acceptance requires separate products')
    if canonical and 'source_diagnostic_limits' in request:
        report['source_step_diagnostics']=_canonical_source_step_diagnostics(root,values,request['source_diagnostic_limits'],sources)
    atomic_json(root/'native-control.json',report)
    return report


def compile_mapped_control_campaign(request,destination,*,request_base):
    """Materialize mapped branches of sealed, completed expanding experiments.

    Original native runs remain explicit inputs. Every mapped output is produced
    by an owned workflow stage; original outputs are never simulated by the
    compiler. Schedules are frozen native canonical times supplied per case,
    and the comparison checks their actual inverse-clock physical epochs.
    """
    from copy import deepcopy
    from .foundation import atomic_json
    from .workflow import safe_id,positive_number,compile_requested_product_dag,default_registry
    from .sidm_analysis import _load_product
    from .sidm_control_measurements import (validate_control_clock_mapping_request,
        validate_control_native_clock_request,validate_control_history_request,
        validate_control_comparison_request,validate_control_ensemble_request)
    keys={'schema','resources','followup_limits','cases','clock_mapping_request','native_clock_request',
        'canonical_population_request','canonical_history_request','comparison_request',
        'publish_ensemble','maximum_ensemble_observables'}
    if not isinstance(request,dict) or set(request)!=keys or request.get('schema')!='sidm-mapped-control-campaign-request-v1':
        raise ValueError('Complete mapped native campaign request required')
    base=Path(request_base).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Keep previous mapped campaign designs')
    resources=request['resources']
    if not isinstance(resources,dict):raise ValueError('Explicit mapped workflow allocations required')
    for key in ('mpi_ranks','omp_threads','memory_gib','disk_gib','maximum_stage_count'):
        positive_number(resources[key],key,key in {'mpi_ranks','omp_threads','maximum_stage_count'})
    if resources['mpi_ranks']!=1:raise ValueError('Canonical native source route requires one MPI rank')
    cases=request['cases'];ensemble=request['publish_ensemble']
    if type(ensemble)is not bool:raise ValueError('Explicit ensemble publication selection required')
    if not isinstance(cases,list) or not cases or len(cases)*6+int(ensemble)>resources['maximum_stage_count']:
        raise MemoryError('Complete mapped campaign exceeds its frozen stage allocation')
    if ensemble and len(cases)<2:raise ValueError('Paired mapped ensemble requires at least two actual realizations')
    if type(request['maximum_ensemble_observables'])is not int or request['maximum_ensemble_observables']<1:
        raise ValueError('Positive mapped ensemble coordinate bound required')
    sources={str(Path(__file__).resolve()):file_sha256(__file__)}
    def pin(reference):
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'} or not isinstance(reference['path'],str):
            raise ValueError('Pin actual completed campaign inputs by path and SHA256')
        checksum=reference['sha256']
        if not isinstance(checksum,str) or len(checksum)!=64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ValueError('Exact lowercase campaign input SHA256 required')
        path=(base/reference['path']).resolve()
        if not path.is_file() or file_sha256(path)!=checksum:raise ValueError('Mapped campaign asset changed: '+str(path))
        sources[str(path)]=checksum
        return {'path':str(path),'sha256':checksum}
    def artifact(stage,path,load=None,field=None):
        result={'stage':stage,'artifact':path}
        if load is not None:result['load']=load
        if field is not None:result['field']=field
        return result
    def product(stage,path):
        return {'path':artifact(stage,path),'sha256':artifact(stage,path,'sha256')}
    def allocated(definition):
        for key in ('memory_gib','disk_gib'):
            if key in definition and definition[key]>resources[key]:raise MemoryError('Mapped stage exceeds '+key)
    stages=[];requested=[];comparison_products=[];names=set();states=set();phases=set();operators=set();definitions=[]
    for case in cases:
        required={'name','population','reference_history','canonical_run_request'}
        if not isinstance(case,dict) or not required<=set(case) or set(case)-required-{'schedule_mode'}:
            raise ValueError('Each mapped case needs its original population/history and complete native run request')
        schedule_mode=case.get('schedule_mode','explicit')
        if schedule_mode not in {'explicit','original_physical_epochs'}:raise ValueError('Select explicit or original-physical-epoch canonical scheduling')
        name=safe_id(case['name'])
        if name in names:raise ValueError('Repeated mapped campaign case identity')
        names.add(name);population_ref=pin(case['population']);history_ref=pin(case['reference_history'])
        population,_,binding=_load_product(population_ref,base,resources['memory_gib']*1024**3/4);sources.update(binding)
        history,_,binding=_load_product(history_ref,base,resources['memory_gib']*1024**3/4);sources.update(binding)
        if population.get('schema')!='sidm-control-population-v1' or population.get('execution_status')!='COMPLETE':
            raise ValueError('Mapped campaign requires a completed original physical population')
        if history.get('schema')!='sidm-physical-control-history-v1' or history.get('execution_status')!='COMPLETE' or history.get('comoving')is not True or history.get('mapped_transport'):
            raise ValueError('Mapped campaign requires actual completed original expanding native histories')
        state=population['physical_state_sha256'];phase=population['request']['seed']
        if state!=history['population_physical_state_sha256'] or population['event_identity_namespace']!=history['identity_namespace'] or phase!=history['phase_seed']:
            raise ValueError('Original campaign population/history physical identity differs')
        run=deepcopy(case['canonical_run_request'])
        if run.get('schema')!='sidm-canonical-native-run-request-v1':raise ValueError('Actual mapped native run schema required')
        run['base_configuration']=pin(run['base_configuration']);run['sidm']['physical_model']=pin(run['sidm']['physical_model'])
        run['sidm']['event_identity_namespace']=population['event_identity_namespace']
        seed=run['sidm']['operator_seed']
        if type(seed)is not int or not 0<=seed<2**64 or str(seed)!=str(history['operator_seed']):
            raise ValueError('Pair original and mapped runs using the same actual operator seed')
        if ensemble and (state in states or phase in phases or seed in operators):
            raise ValueError('Mapped ensemble requires distinct physical populations and phase/operator axes')
        states.add(state);phases.add(phase);operators.add(seed)
        ids={kind:safe_id(kind+'_'+name) for kind in ('mapping','clock','population','run','history','comparison')}
        mapping=deepcopy(request['clock_mapping_request']);mapping.update(history=history_ref,physical_model=run['sidm']['physical_model'])
        if mapping['native_differential_table']is not None:mapping['native_differential_table']=pin(mapping['native_differential_table'])
        validate_control_clock_mapping_request({'request':mapping});allocated(mapping)
        stages.append({'stage_id':ids['mapping'],'kind':'sidm_control_clock_mapping','parameters':{
            'request':mapping,'request_base':str(base),'destination':'{generation}/mapping.json'},'depends_on':[],
            'outputs':{ids['mapping']:'mapping.json'}})
        clock=deepcopy(request['native_clock_request']);clock['mapping']=product(ids['mapping'],'mapping.json')
        validate_control_native_clock_request({'request':clock});allocated(clock)
        stages.append({'stage_id':ids['clock'],'kind':'sidm_control_native_clock','parameters':{
            'request':clock,'request_base':str(base),'destination':'{generation}/native-clock'},'depends_on':[ids['mapping']],
            'outputs':{ids['clock']:'native-clock'}})
        mapped_population=deepcopy(request['canonical_population_request']);mapped_population.update(
            population=population_ref,native_clock=product(ids['clock'],'native-clock/clock.json'))
        validate_canonical_population_request({'request':mapped_population});allocated(mapped_population)
        if population['particle_count']>mapped_population['maximum_particles']:raise MemoryError('Original population exceeds mapped IC bound')
        stages.append({'stage_id':ids['population'],'kind':'sidm_canonical_population','parameters':{
            'request':mapped_population,'request_base':str(base),'destination':'{generation}/population.json'},
            'depends_on':[ids['clock']],'outputs':{ids['population']:'population.json'}})
        run['population']=product(ids['population'],'population.json')
        if schedule_mode=='original_physical_epochs':
            run['output_times_native']=artifact(ids['clock'],'native-clock/clock.json','json',['native_output_times'])
        validate_control_run_request({'request':run});allocated(run)
        if run['omp_threads']!=resources['omp_threads']:raise ValueError('Mapped case changes frozen OpenMP allocation')
        if population['particle_count']>run['maximum_particles']:raise MemoryError('Original population exceeds mapped native run bound')
        if run['canonical_clock_limits']['maximum_asset_bytes']<clock['maximum_asset_bytes'] or run['canonical_clock_limits']['maximum_rows']<clock['maximum_rows']:
            raise ValueError('Mapped native reader allocation is smaller than the clock producer contract')
        stages.append({'stage_id':ids['run'],'kind':'sidm_homogeneous_run','parameters':{
            'request':run,'request_base':str(base),'destination':'{generation}/native-control'},
            'depends_on':[ids['population'],ids['clock']],'outputs':{ids['run']:'native-control'}})
        mapped_history=deepcopy(request['canonical_history_request']);mapped_history.update(engine='gadget4',
            population=product(ids['population'],'population.json'),mapped_run=product(ids['run'],'native-control/native-control.json'),
            snapshots=artifact(ids['run'],'native-control/native-control.json','json',['snapshots']))
        if mapped_history.get('schema')!='sidm-canonical-control-history-request-v1':raise ValueError('Actual mapped history schema required')
        validate_control_history_request({'request':mapped_history});allocated(mapped_history)
        for key in ('speed_bin_edges_kms','velocity_bin_edges_kms'):
            if mapped_history[key]!=history['request'][key]:raise ValueError('Mapped campaign changes original distribution bins')
        if any(name.startswith('particle_') for name in request['comparison_request']['metrics']):
            original_selection=history['request'].get('particle_paths');mapped_selection=mapped_history.get('particle_paths')
            if original_selection is None or mapped_selection is None or sorted(original_selection['particle_ids'])!=sorted(mapped_selection['particle_ids']):
                raise ValueError('Mapped particle campaign must retain the original exact saved path selection')
        output_count=len(history['request']['snapshots']) if schedule_mode=='original_physical_epochs' else len(run['output_times_native'])
        if output_count!=len(history['request']['snapshots']) or output_count>min(mapped_history['maximum_snapshots'],run['maximum_snapshots']):
            raise ValueError('Canonical schedule must retain every original saved physical epoch within stage bounds')
        parameters={'request':mapped_history,'request_base':str(base),'destination':'{generation}/history.json',
            'native_stage_receipt':artifact(ids['run'],'stage-receipt.json')}
        for key in ('parameter_file','build_configuration','invocation_receipt'):
            parameters[key]=artifact(ids['run'],'native-control/native-control.json','json',[key])
        stages.append({'stage_id':ids['history'],'kind':'sidm_control_history','parameters':parameters,
            'depends_on':[ids['run'],ids['population']],'outputs':{ids['history']:'history.json'}})
        comparison=deepcopy(request['comparison_request']);comparison.update(left=product(ids['history'],'history.json'),
            right=history_ref,physical_model=run['sidm']['physical_model'],native_differential_table=deepcopy(mapping['native_differential_table']))
        if comparison.get('scope')!='physical_comoving_mapped_distributions':raise ValueError('Mapped campaign must compare actual physical distributions')
        validate_control_comparison_request({'request':comparison});allocated(comparison)
        stages.append({'stage_id':ids['comparison'],'kind':'sidm_control_comparison','parameters':{
            'request':comparison,'request_base':str(base),'destination':'{generation}/comparison.json'},
            'depends_on':[ids['history']],'outputs':{ids['comparison']:'comparison.json'}})
        requested.append(ids['comparison']);comparison_products.append(product(ids['comparison'],'comparison.json'))
        definitions.append({'name':name,'original_population':population_ref,'original_history':history_ref,
            'population_physical_state_sha256':state,'phase_seed':phase,'operator_seed':str(seed),
            'mapped_native_output_times':run['output_times_native'],'schedule_mode':schedule_mode,'stages':ids})
    if ensemble:
        definition={'schema':'sidm-paired-control-ensemble-request-v1','comparisons':comparison_products,
            'minimum_realizations':len(cases),'maximum_realizations':len(cases),
            'maximum_observables':request['maximum_ensemble_observables'],'memory_gib':resources['memory_gib']}
        validate_control_ensemble_request({'request':definition})
        stages.append({'stage_id':'mapped_ensemble','kind':'sidm_control_ensemble','parameters':{
            'request':definition,'request_base':str(base),'destination':'{generation}/ensemble.json'},
            'depends_on':requested.copy(),'outputs':{'mapped_ensemble':'ensemble.json'}})
        requested.append('mapped_ensemble')
    workflow={'schema':'research-workflow-v1','resources':deepcopy(resources),'followup_limits':deepcopy(request['followup_limits']),
        'physics_spec':{'cosmology_model':'lcdm','physics':'dm','dark_matter':{'linear_model':'cdm','interactions':{'kind':'elastic_sidm'}}},
        'stages':stages,'requested_products':requested}
    compile_requested_product_dag(workflow,default_registry(),request_base=base)
    destination.mkdir(parents=True);atomic_json(destination/'workflow.json',workflow)
    report={'schema':'sidm-mapped-control-campaign-design-v1','workflow':str(destination/'workflow.json'),
        'workflow_sha256':file_sha256(destination/'workflow.json'),'cases':definitions,'stage_count':len(stages),
        'native_run_count':len(cases),'ensemble_requested':ensemble,'source_hashes':sources,
        'execution_status':'NOT_EXECUTED','science_acceptance':'NOT_ESTABLISHED',
        'scope':'owned mapped native branches and paired physical-distribution measurements against supplied completed expanding runs',
        'schedule_scope':'explicit or original-physical-epoch canonical schedules from the owned clock producer; actual inverse-clock epoch agreement checked by comparison',
        'remaining_scope':'independent accuracy/freezing controls and broader control families; selected particle paths require actual saved measurements'}
    atomic_json(destination/'campaign-design.json',report);return report


def _mapped_refinement_plan_bounds(refinement,campaign):
    """Validate the complete static mapped refinement design before evolution."""
    from .workflow import safe_id
    from .sidm_control_measurements import validate_control_comparison_request,validate_control_refinement_measurement_request
    keys={'schema','levels','maximum_levels','comparison_request','measurement_limits'}
    if not isinstance(refinement,dict) or set(refinement)!=keys or refinement.get('schema') not in {'sidm-mapped-timestep-refinement-plan-v1','sidm-mapped-expansion-refinement-plan-v1'}:
        raise ValueError('Complete mapped refinement follow-up design required')
    expansion=refinement['schema']=='sidm-mapped-expansion-refinement-plan-v1'
    levels=refinement['levels'];ceiling=refinement['maximum_levels'];labels=set();values=[]
    if type(ceiling)is not int or ceiling<3 or not isinstance(levels,list) or not 3<=len(levels)<=ceiling:
        raise ValueError('At least three bounded mapped native timestep levels required')
    for level in levels:
        if not isinstance(level,dict) or set(level)!={'label','value'}:raise ValueError('Exact mapped refinement level label/value required')
        label=safe_id(level['label']);value=_positive(level['value'],'mapped refinement timestep')
        if label in labels:raise ValueError('Mapped refinement level names repeat')
        labels.add(label);values.append(value)
        for case in campaign['cases']:
            safe_id('mapped_plan__level_'+label+'__comparison_'+case['name'])
            run=case['canonical_run_request']
            if 'source_diagnostic_limits' not in run or (value>=1 if expansion else run['minimum_timestep_native']>value):
                raise ValueError('Mapped refinement needs realized source diagnostics and consistent minimum timesteps')
    if any(right>=left for left,right in zip(values,values[1:])):raise ValueError('Mapped maximum timestep refinement must decrease strictly')
    comparison=refinement['comparison_request'];validate_control_comparison_request({'request':comparison})
    if any(name.startswith('particle_') for name in comparison['metrics']) and 'particle_paths' not in campaign['canonical_history_request']:
        raise ValueError('Mapped particle refinement requires selected path history production')
    if comparison['scope']!='same_engine_refinement':raise ValueError('Mapped refinement must retain its actual native engine')
    if comparison['memory_gib']>campaign['resources']['memory_gib']:raise MemoryError('Mapped refinement comparison exceeds memory allocation')
    limits=refinement['measurement_limits']
    if not isinstance(limits,dict) or set(limits)!={'mean_error_multiplier','maximum_sampling_error_fraction'}:
        raise ValueError('Complete frozen mapped sampling/mean error budgets required')
    cases=len(campaign['cases'])
    if cases<2:raise ValueError('Mapped refinement requires separate original realizations')
    # Validate dimensional/statistical/workspace declarations with deferred
    # references; completed ensemble products resolve in the child workflow.
    measurement={'schema':'sidm-refinement-measurement-request-v1','parameter_path':['sidm','maximum_scale_factor_fractional_change'] if expansion else ['maximum_timestep_native'],
        'levels':values,'ensembles':[{'path':{'stage':'pending_'+str(i),'artifact':'ensemble.json'},
            'sha256':{'stage':'pending_'+str(i),'artifact':'ensemble.json','load':'sha256'}} for i in range(len(values)-1)],
        'maximum_levels':ceiling,'maximum_observables':campaign['maximum_ensemble_observables'],
        'memory_gib':campaign['resources']['memory_gib'],**limits}
    validate_control_refinement_measurement_request({'request':measurement})
    children=len(levels)*(6*cases+int(campaign['publish_ensemble']))+(len(levels)-1)*(cases+1)+1
    if children>campaign['resources']['maximum_stage_count']:raise MemoryError('Complete mapped refinement exceeds planner stage budget')
    return children,8


def validate_mapped_campaign_plan_request(parameters):
    """Declare budgeted planning without loading unresolved producer outputs."""
    request=parameters['request']
    required={'schema','resources','followup_limits','cases','clock_mapping_request','native_clock_request',
        'canonical_population_request','canonical_history_request','comparison_request','publish_ensemble','maximum_ensemble_observables'}
    if not isinstance(request,dict) or set(request)!=required or request.get('schema')!='sidm-mapped-control-campaign-request-v1':
        raise ValueError('Complete mapped campaign planning request required')
    from .workflow import positive_number,safe_id
    for name in ('mpi_ranks','omp_threads','memory_gib','disk_gib','maximum_stage_count'):
        positive_number(request['resources'][name],name,name in {'mpi_ranks','omp_threads','maximum_stage_count'})
    if request['resources']['mpi_ranks']!=1:raise ValueError('Mapped native planning requires one MPI rank')
    if type(request['publish_ensemble'])is not bool or type(request['maximum_ensemble_observables'])is not int or request['maximum_ensemble_observables']<1:
        raise ValueError('Explicit ensemble selection and coordinate ceiling required')
    cases=request['cases']
    if not isinstance(cases,list) or not cases or 6*len(cases)+int(request['publish_ensemble'])>request['resources']['maximum_stage_count']:
        raise MemoryError('Mapped planning request exceeds its stage allocation')
    if request['publish_ensemble'] and len(cases)<2:raise ValueError('Mapped ensemble requires separate original realizations')
    from .sidm_control_measurements import (validate_control_clock_mapping_request,validate_control_native_clock_request,
        validate_control_history_request,validate_control_comparison_request)
    validators={'clock_mapping_request':validate_control_clock_mapping_request,'native_clock_request':validate_control_native_clock_request,
        'canonical_population_request':validate_canonical_population_request,'canonical_history_request':validate_control_history_request,
        'comparison_request':validate_control_comparison_request}
    for name,validator in validators.items():
        definition=request[name];validator({'request':definition})
        for key in ('memory_gib','disk_gib'):
            if key in definition and definition[key]>request['resources'][key]:raise MemoryError('Mapped planner template exceeds '+key)
    if request['canonical_history_request']['schema']!='sidm-canonical-control-history-request-v1' or request['comparison_request']['scope']!='physical_comoving_mapped_distributions':
        raise ValueError('Mapped planner must publish mapped histories and physical distribution comparisons')
    names=set()
    for case in cases:
        required_case={'name','population','reference_history','canonical_run_request'}
        if not isinstance(case,dict) or not required_case<=set(case) or set(case)-required_case-{'schedule_mode'}:
            raise ValueError('Complete mapped original population/history/run case required')
        name=safe_id(case['name'])
        if name in names:raise ValueError('Mapped planning case names repeat')
        names.add(name)
        for key in ('population','reference_history'):
            if not isinstance(case[key],dict) or set(case[key])!={'path','sha256'}:raise ValueError('Exact mapped planning input references required')
        if case.get('schedule_mode','explicit') not in {'explicit','original_physical_epochs'}:raise ValueError('Unknown mapped scheduling mode')
        validate_control_run_request({'request':case['canonical_run_request']})
    if 'refinement' in parameters:_mapped_refinement_plan_bounds(parameters['refinement'],request)
    return request


def mapped_campaign_plan_input_bindings(parameters):
    from .sidm_analysis import _load_product
    request=validate_mapped_campaign_plan_request(parameters);base=Path(parameters['request_base']).resolve()
    memory=request['resources']['memory_gib']*1024**3;sources={}
    for case in request['cases']:
        for key in ('population','reference_history'):
            _,_,binding=_load_product(case[key],base,memory/4);sources.update(binding)
        run=case['canonical_run_request']
        for reference in (run['base_configuration'],run['sidm']['physical_model']):
            path=(base/reference['path']).resolve()
            if file_sha256(path)!=reference['sha256']:raise ValueError('Mapped planner native input changed')
            sources[str(path)]=reference['sha256']
    reference=request['clock_mapping_request']['native_differential_table']
    if reference is not None:
        path=(base/reference['path']).resolve()
        if file_sha256(path)!=reference['sha256']:raise ValueError('Mapped planner angular table changed')
        sources[str(path)]=reference['sha256']
    return sources


def mapped_campaign_plan_runner(parameters,context):
    """Emit owned children; native processes remain ordinary workflow stages."""
    from copy import deepcopy
    from .foundation import atomic_json
    from .workflow import safe_id
    request=validate_mapped_campaign_plan_request(parameters);allocation=request['resources']
    for key in ('mpi_ranks','omp_threads'):
        if allocation[key]!=context.resources[key]:raise ValueError('Mapped planner changes workflow '+key)
    for key in ('memory_gib','disk_gib','maximum_stage_count'):
        if allocation[key]>context.resources[key]:raise MemoryError('Mapped planner exceeds workflow '+key)
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Mapped planner escapes its owned generation')
    if 'refinement' in parameters:
        input_base=Path(parameters['request_base']).resolve()
        def absolute_inputs(value):
            if isinstance(value,list):return [absolute_inputs(item) for item in value]
            if isinstance(value,dict):
                result={key:absolute_inputs(item) for key,item in value.items()}
                if set(result)=={'path','sha256'} and isinstance(result['path'],str):result['path']=str((input_base/result['path']).resolve())
                return result
            return value
        template=context.generation/'mapped-refinement-input.json'
        atomic_json(template,absolute_inputs(deepcopy(request)))
        refinement=deepcopy(parameters['refinement']);refinement.update(schema='sidm-mapped-expansion-refinement-request-v1' if refinement['schema']=='sidm-mapped-expansion-refinement-plan-v1' else 'sidm-mapped-timestep-refinement-request-v1',
            campaign_template={'path':str(template),'sha256':file_sha256(template)})
        design=compile_mapped_control_refinement(refinement,destination,request_base=input_base)
    else:design=compile_mapped_control_campaign(request,destination,request_base=parameters['request_base'])
    workflow=json.loads(Path(design['workflow']).read_text());stages=workflow['stages']
    names={row['stage_id']:safe_id(context.stage_id+'__'+row['stage_id']) for row in stages}
    def rewrite(value):
        if isinstance(value,list):return [rewrite(item) for item in value]
        if isinstance(value,dict):
            result={key:rewrite(item) for key,item in value.items()}
            if result.get('stage') in names:result['stage']=names[result['stage']]
            return result
        return value
    children=[]
    for original in stages:
        row=rewrite(deepcopy(original));row['stage_id']=names[original['stage_id']]
        row['depends_on']=[names[value] for value in original.get('depends_on',[])]
        row['outputs']={safe_id(context.stage_id+'__'+key):value for key,value in original['outputs'].items()}
        children.append(row)
    requested=[names[value] for value in workflow['requested_products']]
    for case in design.get('cases',[]):case['stages']={key:names[value] for key,value in case['stages'].items()}
    if 'measurement_stage' in design:design['measurement_stage']=names[design['measurement_stage']]
    report={**design,'schema':'sidm-mapped-control-followup-plan-v1','parent_stage':context.stage_id,
        'requested_result_stages':requested,'followup_stage_count':len(children),'execution_status':'PLANNED_NOT_NATIVE_EXECUTED',
        'followup_budget_policy':'root workflow child/dependency-depth/stage/resource admission and preflight; no nested native execution'}
    atomic_json(destination/'followup-plan.json',report);atomic_json(destination/'followups.json',{'followups':children})
    return {'followups':children,'measurements':{'mapped_campaign_plan':str(destination/'followup-plan.json'),
        'mapped_campaign_plan_sha256':file_sha256(destination/'followup-plan.json'),'requested_result_stages':requested},
        'scientific_exclusions':[{'scope':'mapped SIDM campaign','reason':'Saved particle paths are measured only when selected; independent accuracy requires actual returned comparisons and controls'}]}


def compile_fresh_mapped_control_campaign(request,destination,*,request_base):
    """Compose original native experiments and a budgeted mapped planner."""
    from copy import deepcopy
    from .foundation import atomic_json
    from .workflow import compile_requested_product_dag,default_registry
    keys={'schema','original_campaign','original_variant','mapped_campaign_template'}
    if isinstance(request,dict) and 'refinement' in request:keys.add('refinement')
    if not isinstance(request,dict) or set(request)!=keys or request.get('schema')!='sidm-fresh-mapped-control-campaign-request-v1':
        raise ValueError('Complete fresh expanding-plus-mapped campaign request required')
    original=deepcopy(request['original_campaign']);mapped=deepcopy(request['mapped_campaign_template'])
    if original.get('schema')!='sidm-physical-control-campaign-request-v1' or original['population_request']['clock']['mode']!='comoving':
        raise ValueError('Fresh mapped campaign requires original expanding physical native experiments')
    matches=[row for row in original['variants'] if row['name']==request['original_variant']]
    if len(matches)!=1:raise ValueError('Select one actual original native variant to map')
    variant=matches[0];prototype=mapped.get('cases')
    if any(name.startswith('particle_') for name in mapped['comparison_request']['metrics']):
        original_selection=variant['history_request'].get('particle_paths');mapped_selection=mapped['canonical_history_request'].get('particle_paths')
        if original_selection is None or mapped_selection is None or sorted(original_selection['particle_ids'])!=sorted(mapped_selection['particle_ids']):
            raise ValueError('Fresh particle campaign needs the same explicit selection in original and mapped histories')
    if not isinstance(prototype,list) or len(prototype)!=1:raise ValueError('Supply exactly one complete mapped case prototype for expansion over original realizations')
    resources=original['resources'];mapped['resources']=deepcopy(resources);mapped['followup_limits']=deepcopy(original['followup_limits'])
    count=len(original['realizations']);children=6*count+int(mapped['publish_ensemble'])
    limits=original['followup_limits']
    if limits['maximum_children']<children:raise ValueError('Freeze enough root follow-up children for the complete mapped campaign')
    mapped['cases']=[]
    def product(stage,path):return {'path':{'stage':stage,'artifact':path},'sha256':{'stage':stage,'artifact':path,'load':'sha256'}}
    for index,seeds in enumerate(original['realizations']):
        case=deepcopy(prototype[0]);case.update(name='realization_'+str(index),
            population=product('population_'+str(index),'population.json'),
            reference_history=product('history_'+variant['name']+'_'+str(index),'history.json'),schedule_mode='original_physical_epochs')
        case['canonical_run_request']['sidm']['operator_seed']=seeds['operator_seed']
        for key in ('physical_model','particle_convention','cross_section_normalization_cm2_per_g','scheme'):
            case['canonical_run_request']['sidm'][key]=deepcopy(variant['run_request']['sidm'][key])
        mapped['cases'].append(case)
    planner_parameters={'request':mapped}
    child_depth=6+int(mapped['publish_ensemble'])
    if 'refinement' in request:
        planner_parameters['refinement']=deepcopy(request['refinement'])
        children,child_depth=_mapped_refinement_plan_bounds(request['refinement'],mapped)
        if limits['maximum_children']<children:raise ValueError('Root child allocation cannot contain the complete mapped refinement family')
    validate_mapped_campaign_plan_request(planner_parameters)
    destination=Path(destination).resolve();base=Path(request_base).resolve()
    if destination.exists():raise FileExistsError('Retain previous fresh paired campaign designs')
    # The original compiler pins its actual native assets and builds the complete
    # independent-engine population/run/history/comparison recipe as before.
    original_design=compile_control_campaign(original,destination/'original-design',request_base=base)
    model_reference=variant['run_request']['sidm']['physical_model']
    model_path=(base/model_reference['path']).resolve()
    if model_path.stat().st_size*16>resources['memory_gib']*1024**3/4:raise MemoryError('Fresh mapped physical model exceeds planner workspace')
    model=json.loads(model_path.read_text())
    needs_table=model.get('schema')=='sidm-elastic-model-v2' and (model.get('kind')!='constant_isotropic' and
        not (model.get('kind')=='screened_rutherford' and model.get('supplied_cross_section')=='total'))
    if needs_table and mapped['clock_mapping_request']['native_differential_table']is None:
        mapped['clock_mapping_request']['native_differential_table']={key:{'stage':'run_'+variant['name']+'_0',
            'artifact':'native-control/native-control.json','load':'json','field':['native_differential_table',key]} for key in ('path','sha256')}
    workflow=json.loads(Path(original_design['workflow']).read_text());dependencies=[
        'history_'+variant['name']+'_'+str(index) for index in range(count)]
    planner={'stage_id':'mapped_plan','kind':'sidm_mapped_campaign_plan','parameters':{
        'request':mapped,'request_base':str(base),'destination':'{generation}/mapped-design'},
        'depends_on':dependencies,'outputs':{'mapped_plan':'mapped-design'}}
    if 'refinement' in request:planner['parameters']['refinement']=deepcopy(request['refinement'])
    workflow['stages'].append(planner);workflow['requested_products'].append('mapped_plan')
    if len(workflow['stages'])+children>resources['maximum_stage_count']:raise MemoryError('Original and complete mapped follow-ups exceed root stage allocation')
    plan=compile_requested_product_dag(workflow,default_registry(),request_base=base)
    parent=next(row for row in plan['stages'] if row['stage_id']=='mapped_plan')
    if parent['depth']+child_depth>limits['maximum_depth']:
        raise ValueError('Freeze enough dependency depth for every mapped child and ensemble')
    atomic_json(destination/'workflow.json',workflow)
    report={'schema':'sidm-fresh-mapped-control-campaign-design-v1','workflow':str(destination/'workflow.json'),
        'workflow_sha256':file_sha256(destination/'workflow.json'),'original_design':original_design,
        'original_variant':variant['name'],'mapped_planner_stage':'mapped_plan','mapped_native_run_count':count*(len(request['refinement']['levels']) if 'refinement' in request else 1),
        'mapped_refinement_requested':'refinement' in request,
        'mapped_followup_stage_count':children,'complete_planned_stage_count':len(workflow['stages'])+children,
        'source_hashes':original_design['source_hashes'],'execution_status':'NOT_EXECUTED','science_acceptance':'NOT_ESTABLISHED',
        'scope':'fresh original native campaign followed by owned source-bound canonical runs and physical comparisons under one workflow budget'}
    atomic_json(destination/'campaign-design.json',report);return report


def compile_control_campaign(request,destination,*,request_base):
    """Build the complete population/run/history/comparison/ensemble DAG.

    Generated paths and hashes resolve from completed producer artifacts; this
    compiler emits no simulated output, passing receipt or invented build proof.
    """
    if isinstance(request,dict) and request.get('schema')=='sidm-fresh-mapped-control-campaign-request-v1':
        return compile_fresh_mapped_control_campaign(request,destination,request_base=request_base)
    if isinstance(request,dict) and request.get('schema')=='sidm-mapped-control-campaign-request-v1':
        return compile_mapped_control_campaign(request,destination,request_base=request_base)
    from copy import deepcopy
    from .foundation import atomic_json
    from .workflow import safe_id,compile_requested_product_dag,default_registry
    from .sidm_control_measurements import validate_control_history_request,validate_control_comparison_request,validate_control_expectation_request
    keys={'schema','resources','followup_limits','population_request','realizations','variants','comparisons','maximum_ensemble_observables'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='sidm-physical-control-campaign-request-v1':
        raise ValueError('Complete automatic physical-control campaign declaration required')
    base=Path(request_base).resolve();destination=Path(destination).resolve();resources=request['resources'];realizations=request['realizations']
    from .workflow import positive_number
    for key in ('mpi_ranks','omp_threads','memory_gib','disk_gib','maximum_stage_count'):
        positive_number(resources[key],key,key in {'mpi_ranks','omp_threads','maximum_stage_count'})
    if destination.exists():raise FileExistsError('Keep previous native campaign designs')
    if not isinstance(realizations,list) or len(realizations)<2 or len(realizations)>resources['maximum_stage_count']:
        raise ValueError('Explicit bounded independent phase/operator realizations required')
    phases=set();operators=set()
    for row in realizations:
        if not isinstance(row,dict) or set(row)!={'phase_seed','operator_seed'}:raise ValueError('Each realization needs its exact phase and operator seeds')
        for key in ('phase_seed','operator_seed'):
            if type(row[key])is not int or not 0<=row[key]<2**64:raise ValueError('Exact uint64 campaign seed required')
        if row['phase_seed'] in phases or row['operator_seed'] in operators:raise ValueError('Campaign realizations reuse a phase/operator seed axis')
        phases.add(row['phase_seed']);operators.add(row['operator_seed'])
    variants=request['variants'];comparisons=request['comparisons'];names=set();pair_names=set()
    if not isinstance(variants,list) or len(variants)<2 or not isinstance(comparisons,list) or not comparisons:
        raise ValueError('At least two native variants and an explicit paired comparison required')
    expectation_count=sum(isinstance(row,dict) and 'expectation_request' in row for row in variants)
    build_designs={canonical_digest(row['reference_build_request']) for row in variants if isinstance(row,dict) and 'reference_build_request' in row}
    if len(realizations)*(1+2*len(variants)+len(comparisons)+expectation_count)+len(comparisons)+len(build_designs)>resources['maximum_stage_count']:
        raise MemoryError('Complete native campaign exceeds its declared stage ceiling')
    if type(request['maximum_ensemble_observables'])is not int or request['maximum_ensemble_observables']<1:
        raise ValueError('Positive bounded ensemble coordinate ceiling required')
    population_request=deepcopy(request['population_request']);validate_control_population_request({'request':population_request})
    sources={str(Path(__file__).resolve()):file_sha256(__file__)}
    def pin(reference):
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('Exact retained campaign asset required')
        path=(base/reference['path']).resolve()
        if file_sha256(path)!=reference['sha256']:raise ValueError('Native campaign input asset changed')
        sources[str(path)]=reference['sha256'];return path
    for variant in variants:
        if not isinstance(variant,dict) or not {'name','run_request','history_request'}<=set(variant) or set(variant)-{'name','run_request','history_request','expectation_request','reference_build_request'}:
            raise ValueError('Explicit native run, history and optional frozen physical expectation variant required')
        name=safe_id(variant['name'])
        if name in names:raise ValueError('Repeated native campaign variant')
        names.add(name);run=variant['run_request'];pin(run['base_configuration']);pin(run['sidm']['physical_model'])
        if 'reference_build_request' in variant:
            recipe=variant['reference_build_request'];validate_reference_build_request({'request':recipe})
            if run['engine']!='opengadget3' or recipe['scheme']!=run['sidm']['scheme'] or recipe['particle_convention']!=run['sidm']['particle_convention']:
                raise ValueError('Independent build recipe changes the requested native operator/convention')
            for key in ('source_manifest','configuration','system_makefile','dependency_manifest'):pin(recipe[key])
            for reference in recipe['tools'].values():pin(reference)
        elif run['reference_build']is not None:
            for reference in run['reference_build'].values():pin(reference)
    by_name={variant['name']:variant for variant in variants}
    for comparison in comparisons:
        if not isinstance(comparison,dict) or set(comparison)!={'name','left','right','request'}:raise ValueError('Explicit named pair and frozen comparison definitions required')
        name=safe_id(comparison['name'])
        if name in pair_names or comparison['left'] not in names or comparison['right'] not in names or comparison['left']==comparison['right']:
            raise ValueError('Comparison must name two distinct actual variants once')
        pair_names.add(name)
        left=by_name[comparison['left']]['run_request'];right=by_name[comparison['right']]['run_request']
        if left['sidm']['physical_model']['sha256']!=right['sidm']['physical_model']['sha256']:
            raise ValueError('Native comparison variants change the actual physical angular model')
        if left['sidm']['cross_section_normalization_cm2_per_g']!=right['sidm']['cross_section_normalization_cm2_per_g'] or left['sidm']['particle_convention']!=right['sidm']['particle_convention']:
            raise ValueError('Native comparison variants change dimensional normalization or particle convention')
        scope=comparison['request']['scope']
        if scope=='independent_engine_trajectories' and {left['engine'],right['engine']}!={'gadget4','opengadget3'}:
            raise ValueError('Independent campaign comparison requires the two actual engines')
        if scope=='same_engine_refinement' and left['engine']!=right['engine']:raise ValueError('Refinement campaign changes its native engine')
        if scope=='rare_frequent_hybrid_overlap' and left['sidm']['scheme']==right['sidm']['scheme']:
            raise ValueError('Transport overlap campaign requires distinct actual schemes')
        if scope=='physical_comoving_balance':
            raise ValueError('This common-population campaign needs separate clock-representation populations for physical/comoving controls')
    def artifact(stage,path,load=None,field=None):
        value={'stage':stage,'artifact':path}
        if load is not None:value['load']=load
        if field is not None:value['field']=field
        return value
    def product(stage,path):return {'path':artifact(stage,path),'sha256':artifact(stage,path,'sha256')}
    stages=[];outputs={name:[] for name in pair_names};run_count=0;physical_outputs=[];build_ids={};variant_builds={}
    for variant in variants:
        if 'reference_build_request' not in variant:continue
        recipe=deepcopy(variant['reference_build_request']);signature=canonical_digest(recipe)
        if signature not in build_ids:
            build_id='build_'+variant['name'];build_ids[signature]=build_id
            stages.append({'stage_id':build_id,'kind':'sidm_reference_build','parameters':{'request':recipe,'request_base':str(base),
                'destination':'{generation}/reference-build'},'depends_on':[],'outputs':{build_id:'reference-build'}})
        variant_builds[variant['name']]=build_ids[signature]
    for index,seeds in enumerate(realizations):
        population_id='population_'+str(index);population=deepcopy(population_request);population['seed']=seeds['phase_seed']
        stages.append({'stage_id':population_id,'kind':'sidm_control_population','parameters':{'request':population,'request_base':str(base),
            'destination':'{generation}/population.json'},'depends_on':[],'outputs':{population_id:'population.json'}})
        histories={};runs={}
        for variant in variants:
            run_id='run_'+variant['name']+'_'+str(index);history_id='history_'+variant['name']+'_'+str(index)
            run=deepcopy(variant['run_request']);run['population']=product(population_id,'population.json');run['sidm']['operator_seed']=seeds['operator_seed']
            dependencies=[population_id]
            if variant['name'] in variant_builds:
                build_id=variant_builds[variant['name']];dependencies.append(build_id)
                run['reference_build']={label:{key:artifact(build_id,'reference-build/reference-build.json','json',[label,key]) for key in ('path','sha256')} for label in ('capability','configuration')}
            validate_control_run_request({'request':run})
            for key in ('mpi_ranks','omp_threads'):
                if run[key]!=resources[key]:raise ValueError('Campaign variant changes frozen '+key)
            for key in ('memory_gib','disk_gib'):
                if run[key]>resources[key]:raise MemoryError('Campaign variant exceeds frozen '+key)
            initial=population['clock'];units=population['native_units']
            native_start=initial['scale_factor'] if initial['mode']=='comoving' else initial['proper_time_seconds']*units['hubble_parameter']*units['velocity_cm_s']/units['length_cm']
            if run['output_times_native'][0]!=native_start:raise ValueError('Campaign native schedule changes the explicit population starting epoch')
            if sum(row['particle_count'] for row in population['components'])>run['maximum_particles']:
                raise MemoryError('Campaign original population exceeds its native run bound')
            stages.append({'stage_id':run_id,'kind':'sidm_homogeneous_run','parameters':{'request':run,'request_base':str(base),
                'destination':'{generation}/native-control'},'depends_on':dependencies,'outputs':{run_id:'native-control'}})
            history=deepcopy(variant['history_request']);history.update(engine=run['engine'],population=product(population_id,'population.json'),
                snapshots=artifact(run_id,'native-control/native-control.json','json',['snapshots']))
            validate_control_history_request({'request':history})
            parameters={'request':history,'request_base':str(base),'destination':'{generation}/history.json',
                'native_stage_receipt':artifact(run_id,'stage-receipt.json')}
            for key in ('parameter_file','build_configuration','invocation_receipt'):
                parameters[key]=artifact(run_id,'native-control/native-control.json','json',[key])
            stages.append({'stage_id':history_id,'kind':'sidm_control_history','parameters':parameters,'depends_on':[run_id,population_id],
                'outputs':{history_id:'history.json'}})
            histories[variant['name']]=history_id;runs[variant['name']]=run_id;run_count+=1
            if 'expectation_request' in variant:
                expectation_id='expectation_'+variant['name']+'_'+str(index)
                expectation=deepcopy(variant['expectation_request'])
                expectation.update(history=product(history_id,'history.json'),physical_model=deepcopy(run['sidm']['physical_model']),native_differential_table=None)
                actual_model=json.loads(pin(run['sidm']['physical_model']).read_text())
                independent_table=run['engine']=='opengadget3' and actual_model.get('schema')=='sidm-elastic-model-v2' and (
                    actual_model.get('kind')!='constant_isotropic' and not (actual_model.get('kind')=='screened_rutherford' and actual_model.get('supplied_cross_section')=='total'))
                if independent_table:
                    expectation['native_differential_table']={key:artifact(run_id,'native-control/native-control.json','json',['native_differential_table',key]) for key in ('path','sha256')}
                validate_control_expectation_request({'request':expectation})
                if expectation['memory_gib']>resources['memory_gib']:raise MemoryError('Campaign physical expectation exceeds memory allocation')
                stages.append({'stage_id':expectation_id,'kind':'sidm_control_expectation','parameters':{'request':expectation,'request_base':str(base),
                    'destination':'{generation}/expectation.json'},'depends_on':[history_id,run_id,population_id],
                    'outputs':{expectation_id:'expectation.json'}})
                physical_outputs.append(expectation_id)
        for pair in comparisons:
            comparison_id='comparison_'+pair['name']+'_'+str(index);definition=deepcopy(pair['request'])
            definition.update(left=product(histories[pair['left']],'history.json'),right=product(histories[pair['right']],'history.json'))
            original=by_name[pair['left']]['run_request']['sidm']['physical_model'];definition['physical_model']=deepcopy(original)
            model=json.loads(pin(original).read_text())
            needs_table=model.get('schema')=='sidm-elastic-model-v2' and (model.get('kind')!='constant_isotropic' and
                not (model.get('kind')=='screened_rutherford' and model.get('supplied_cross_section')=='total'))
            definition['native_differential_table']=None
            if needs_table:
                selected=pair['left'] if by_name[pair['left']]['run_request']['engine']=='opengadget3' else pair['right']
                run_id=runs[selected]
                definition['native_differential_table']={key:artifact(run_id,'native-control/native-control.json','json',['native_differential_table',key]) for key in ('path','sha256')}
            validate_control_comparison_request({'request':definition})
            if any(name.startswith('particle_') for name in definition['metrics']):
                selections=[by_name[pair[side]]['history_request'].get('particle_paths') for side in ('left','right')]
                if any(selection is None for selection in selections) or sorted(selections[0]['particle_ids'])!=sorted(selections[1]['particle_ids']):
                    raise ValueError('Physical particle campaign comparisons need common exact selected path IDs')
            stages.append({'stage_id':comparison_id,'kind':'sidm_control_comparison','parameters':{'request':definition,'request_base':str(base),
                'destination':'{generation}/comparison.json'},'depends_on':[histories[pair['left']],histories[pair['right']]],
                'outputs':{comparison_id:'comparison.json'}})
            outputs[pair['name']].append(product(comparison_id,'comparison.json'))
    requested=list(physical_outputs)
    for pair in comparisons:
        ensemble_id='ensemble_'+pair['name'];requested.append(ensemble_id)
        ensemble={'schema':'sidm-paired-control-ensemble-request-v1','comparisons':outputs[pair['name']],
            'minimum_realizations':len(realizations),'maximum_realizations':len(realizations),'maximum_observables':request['maximum_ensemble_observables'],
            'memory_gib':resources['memory_gib']}
        stages.append({'stage_id':ensemble_id,'kind':'sidm_control_ensemble','parameters':{'request':ensemble,'request_base':str(base),
            'destination':'{generation}/ensemble.json'},'depends_on':['comparison_'+pair['name']+'_'+str(i) for i in range(len(realizations))],
            'outputs':{ensemble_id:'ensemble.json'}})
    workflow={'schema':'research-workflow-v1','resources':resources,'followup_limits':request['followup_limits'],
        'physics_spec':{'cosmology_model':'lcdm','physics':'dm','dark_matter':{'linear_model':'cdm','interactions':{'kind':'elastic_sidm'}}},
        'stages':stages,'requested_products':requested}
    compile_requested_product_dag(workflow,default_registry(),request_base=base)
    destination.mkdir(parents=True)
    atomic_json(destination/'workflow.json',workflow)
    report={'schema':'sidm-physical-control-campaign-design-v1','workflow':str(destination/'workflow.json'),
        'workflow_sha256':file_sha256(destination/'workflow.json'),'native_run_count':run_count,'realization_count':len(realizations),
        'stage_count':len(stages),'physical_expectation_count':len(physical_outputs),'independent_build_count':len(build_ids),
        'source_hashes':sources,'execution_status':'NOT_EXECUTED','science_acceptance':'NOT_ESTABLISHED'}
    atomic_json(destination/'campaign-design.json',report);return report


def compile_mapped_control_refinement(request,destination,*,request_base):
    """Produce genuine mapped native timestep levels and paired measurements.

    Decreasing a rejection tolerance is not refinement. This family changes the
    operative native maximum timestep, keeping the expansion-span safety gate,
    physical law, exact original realizations and output schedule fixed.
    """
    from copy import deepcopy
    from .foundation import atomic_json
    from .workflow import safe_id,compile_requested_product_dag,default_registry
    from .sidm_control_measurements import validate_control_comparison_request,validate_control_refinement_measurement_request
    keys={'schema','campaign_template','levels','maximum_levels','comparison_request','measurement_limits'}
    if not isinstance(request,dict) or set(request)!=keys or request.get('schema') not in {'sidm-mapped-timestep-refinement-request-v1','sidm-mapped-expansion-refinement-request-v1'}:
        raise ValueError('Complete mapped native timestep refinement request required')
    expansion=request['schema']=='sidm-mapped-expansion-refinement-request-v1'
    base=Path(request_base).resolve();reference=request['campaign_template']
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('Pin the complete mapped campaign template')
    source=(base/reference['path']).resolve()
    if not source.is_file() or source.stat().st_size>16777216 or file_sha256(source)!=reference['sha256']:
        raise ValueError('Mapped refinement template changed or exceeds its metadata allocation')
    template=json.loads(source.read_text());base=source.parent
    validate_mapped_campaign_plan_request({'request':template})
    cases=template['cases'];count=len(cases);resources=template['resources']
    if count<2:raise ValueError('Mapped refinement needs separate paired phase/operator realizations')
    ceiling=request['maximum_levels'];levels=request['levels']
    if type(ceiling)is not int or ceiling<3 or not isinstance(levels,list) or not 3<=len(levels)<=ceiling:
        raise ValueError('At least three explicitly bounded operative timestep levels required')
    labels=set();values=[]
    for level in levels:
        if not isinstance(level,dict) or set(level)!={'label','value'}:raise ValueError('Each native timestep level needs an exact label and value')
        label=safe_id(level['label']);value=_positive(level['value'],'mapped native maximum timestep')
        if label in labels:raise ValueError('Repeated mapped timestep level')
        labels.add(label);values.append(value)
        for case in cases:
            if (value>=1 if expansion else case['canonical_run_request']['minimum_timestep_native']>value):raise ValueError('Refinement axis exceeds its native fractional span or timestep domain')
    if any(right>=left for left,right in zip(values,values[1:])):raise ValueError('Mapped native timestep levels must decrease strictly')
    limits=request['measurement_limits']
    if not isinstance(limits,dict) or set(limits)!={'mean_error_multiplier','maximum_sampling_error_fraction'}:
        raise ValueError('Freeze mapped mean-error and sampling-budget definitions')
    comparison=deepcopy(request['comparison_request'])
    if comparison.get('scope')!='same_engine_refinement':raise ValueError('Mapped timestep controls require measured same-engine comparison')
    validate_control_comparison_request({'request':comparison})
    if any(name.startswith('particle_') for name in comparison['metrics']) and 'particle_paths' not in template['canonical_history_request']:
        raise ValueError('Mapped particle refinement requires selected path history production')
    if comparison['memory_gib']>resources['memory_gib']:raise MemoryError('Mapped adjacent comparison exceeds allocation')
    estimate=len(levels)*(6*count+int(template['publish_ensemble']))+(len(levels)-1)*(count+1)+1
    if estimate>resources['maximum_stage_count']:raise MemoryError('Complete mapped level/realization/comparison family exceeds stage ceiling')
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Retain earlier mapped refinement designs')
    stages=[];requested=[];histories=[];designs=[]
    def artifact(stage,path,load=None):
        value={'stage':stage,'artifact':path}
        if load is not None:value['load']=load
        return value
    def product(stage,path):return {'path':artifact(stage,path),'sha256':artifact(stage,path,'sha256')}
    for level in levels:
        campaign=deepcopy(template)
        for case in campaign['cases']:
            if 'source_diagnostic_limits' not in case['canonical_run_request']:raise ValueError('Mapped timestep refinement requires actual source diagnostic allocations')
            if expansion:case['canonical_run_request']['sidm']['maximum_scale_factor_fractional_change']=level['value']
            else:case['canonical_run_request']['maximum_timestep_native']=level['value']
        design=compile_mapped_control_campaign(campaign,destination/('level-'+level['label']),request_base=base)
        workflow=json.loads(Path(design['workflow']).read_text());prefix='level_'+level['label']+'__'
        names={row['stage_id']:safe_id(prefix+row['stage_id']) for row in workflow['stages']}
        def rewrite(value):
            if isinstance(value,list):return [rewrite(item) for item in value]
            if isinstance(value,dict):
                result={key:rewrite(item) for key,item in value.items()}
                if result.get('stage') in names:result['stage']=names[result['stage']]
                return result
            return value
        for original in workflow['stages']:
            row=rewrite(deepcopy(original));row['stage_id']=names[original['stage_id']]
            row['depends_on']=[names[member] for member in original.get('depends_on',[])]
            row['outputs']={safe_id(prefix+key):value for key,value in original['outputs'].items()}
            stages.append(row)
        requested.extend(names[member] for member in workflow['requested_products'])
        histories.append([names['history_'+case['name']] for case in cases]);designs.append(design)
    ensembles=[]
    for index in range(len(levels)-1):
        comparisons=[];ids=[]
        for realization,case in enumerate(cases):
            identity=safe_id('adjacent_'+str(index)+'_'+case['name']);definition=deepcopy(comparison)
            definition.update(left=product(histories[index][realization],'history.json'),right=product(histories[index+1][realization],'history.json'),
                physical_model=deepcopy(case['canonical_run_request']['sidm']['physical_model']),
                native_differential_table=deepcopy(template['clock_mapping_request']['native_differential_table']))
            validate_control_comparison_request({'request':definition})
            stages.append({'stage_id':identity,'kind':'sidm_control_comparison','parameters':{'request':definition,'request_base':str(base),
                'destination':'{generation}/comparison.json'},'depends_on':[histories[index][realization],histories[index+1][realization]],
                'outputs':{identity:'comparison.json'}})
            ids.append(identity);comparisons.append(product(identity,'comparison.json'))
        identity='adjacent_ensemble_'+str(index)
        definition={'schema':'sidm-paired-control-ensemble-request-v1','comparisons':comparisons,
            'minimum_realizations':count,'maximum_realizations':count,'maximum_observables':template['maximum_ensemble_observables'],
            'memory_gib':resources['memory_gib']}
        stages.append({'stage_id':identity,'kind':'sidm_control_ensemble','parameters':{'request':definition,'request_base':str(base),
            'destination':'{generation}/ensemble.json'},'depends_on':ids,'outputs':{identity:'ensemble.json'}})
        ensembles.append(identity)
    measurement={'schema':'sidm-refinement-measurement-request-v1','parameter_path':['sidm','maximum_scale_factor_fractional_change'] if expansion else ['maximum_timestep_native'],'levels':values,
        'ensembles':[product(identity,'ensemble.json') for identity in ensembles],'maximum_levels':ceiling,
        'maximum_observables':template['maximum_ensemble_observables'],'memory_gib':resources['memory_gib'],**limits}
    validate_control_refinement_measurement_request({'request':measurement})
    identity='mapped_expansion_refinement' if expansion else 'mapped_timestep_refinement'
    stages.append({'stage_id':identity,'kind':'sidm_control_refinement_measurement','parameters':{'request':measurement,
        'request_base':str(base),'destination':'{generation}/refinement.json'},'depends_on':ensembles,'outputs':{identity:'refinement.json'}})
    requested.append(identity)
    workflow={**workflow,'stages':stages,'requested_products':requested}
    compile_requested_product_dag(workflow,default_registry(),request_base=base)
    atomic_json(destination/'workflow.json',workflow)
    report={'schema':'sidm-mapped-expansion-refinement-design-v1' if expansion else 'sidm-mapped-timestep-refinement-design-v1','workflow':str(destination/'workflow.json'),
        'workflow_sha256':file_sha256(destination/'workflow.json'),'levels':levels,'level_designs':designs,
        'complete_stage_count':len(stages),'native_run_count':len(levels)*count,'measurement_stage':identity,
        'source_hashes':{str(source):reference['sha256'],str(Path(__file__).resolve()):file_sha256(__file__)},
        'execution_status':'NOT_EXECUTED','science_acceptance':'NOT_ESTABLISHED',
        'scope':'operative native mapped timestep and frozen-law interval refinement; exact paired adjacent realizations/covariance retained',
        'remaining_scope':'no assumed convergence order or continuum extrapolation; selected saved-path budgets require actual measurements and do not certify independent-engine accuracy'}
    atomic_json(destination/'refinement-design.json',report);return report


def compile_control_refinement(request,destination,*,request_base):
    """Expand independent one-axis multi-level homogeneous refinement families.

    Every adjacent comparison retains the same physical population/model and
    operator seeds. This emits real native stages, not convergence inferred
    from configured tolerances or an assumed Richardson order.
    """
    if isinstance(request,dict) and request.get('schema') in {'sidm-mapped-timestep-refinement-request-v1','sidm-mapped-expansion-refinement-request-v1'}:
        return compile_mapped_control_refinement(request,destination,request_base=request_base)
    from copy import deepcopy
    from .foundation import atomic_json
    from .workflow import safe_id
    from .sidm_control_measurements import validate_control_comparison_request
    keys={'schema','campaign_template','base_variant','axes','comparison_request','maximum_levels_per_axis','measurement_limits'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='sidm-homogeneous-refinement-request-v1':
        raise ValueError('Complete explicit homogeneous refinement family request required')
    base=Path(request_base).resolve();reference=request['campaign_template']
    measurement_limits=request['measurement_limits']
    if not isinstance(measurement_limits,dict) or set(measurement_limits)!={'mean_error_multiplier','maximum_sampling_error_fraction'}:
        raise ValueError('Freeze the ensemble-mean error multiplier and sampling budget fraction in the refinement design')
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('Pin the complete native campaign template')
    source=(base/reference['path']).resolve()
    if file_sha256(source)!=reference['sha256']:raise ValueError('Refinement campaign template changed')
    if source.stat().st_size>16777216:raise MemoryError('Refinement template metadata exceeds its bounded domain')
    template=json.loads(source.read_text());base=source.parent;variants=template.get('variants')
    if not isinstance(variants,list):raise ValueError('Refinement needs an actual complete native campaign template')
    matches=[row for row in variants if row.get('name')==request['base_variant']]
    if len(matches)!=1:raise ValueError('Refinement base variant must identify one native run/history definition')
    original=matches[0];run=original['run_request'];engine=run['engine'];scheme=run['sidm']['scheme']
    ceiling=request['maximum_levels_per_axis']
    if type(ceiling)is not int or ceiling<3:raise ValueError('At least three bounded levels per refinement axis required')
    axes=request['axes'];names=set();paths=set();generated=[];comparisons=[];families=[]
    if not isinstance(axes,list) or not axes or len(axes)>8:raise ValueError('Select one to eight independent bounded refinement axes')
    resources=template.get('resources',{});stage_ceiling=resources.get('maximum_stage_count');realizations=template.get('realizations')
    if type(stage_ceiling)is not int or stage_ceiling<1 or not isinstance(realizations,list) or len(realizations)<2:
        raise ValueError('Refinement template requires bounded workflow resources and independent realizations')
    level_count=0
    for axis in axes:
        if not isinstance(axis,dict) or not isinstance(axis.get('levels'),list):raise ValueError('Explicit bounded axis level lists required')
        level_count+=len(axis['levels'])
    comparison_count=level_count-len(axes)
    estimate=len(realizations)*(1+2*level_count+comparison_count+level_count*int('expectation_request' in original))+comparison_count+len(axes)+int('reference_build_request' in original)
    if estimate>stage_ceiling:raise MemoryError('Complete refinement families exceed the frozen native stage ceiling before materialization')
    allowed={('maximum_timestep_native',),('sidm','kernel_support_proper_kpc'),('sidm','split_angle_radians'),
        ('sidm','maximum_pair_probability'),('sidm','maximum_particle_probability'),('sidm','maximum_transfer_depth')}
    definition=deepcopy(request['comparison_request'])
    if definition.get('scope')!='same_engine_refinement':raise ValueError('Each refinement comparison must retain its actual native engine')
    for axis in axes:
        if not isinstance(axis,dict) or set(axis)!={'name','parameter_path','levels'}:raise ValueError('Each independent axis needs its native parameter path and ordered levels')
        name=safe_id(axis['name']);path=axis['parameter_path']
        if not isinstance(path,list) or any(not isinstance(member,str) for member in path):raise ValueError('Explicit native run parameter path required')
        path=tuple(path)
        if name in names or path in paths or path not in allowed:raise ValueError('Distinct maintained operative refinement axes required')
        names.add(name);paths.add(path)
        if engine!='gadget4' and path not in {('maximum_timestep_native',),('sidm','split_angle_radians')}:
            raise ValueError('Independent engine kernel/probability controls need their actual independent numerical template; Gadget4 settings cannot relabel them')
        if path==('sidm','split_angle_radians') and scheme!='hybrid':raise ValueError('A split-angle refinement needs an actual hybrid operator')
        if path==('sidm','maximum_transfer_depth') and scheme not in {'frequent','hybrid'}:raise ValueError('Transfer-depth refinement needs an actual frequent contribution')
        if path in {('sidm','maximum_pair_probability'),('sidm','maximum_particle_probability')} and scheme not in {'rare','hybrid'}:
            raise ValueError('Rare probability refinements need an actual rare contribution')
        levels=axis['levels']
        if not isinstance(levels,list) or not 3<=len(levels)<=ceiling:raise ValueError('Every refinement axis requires at least three explicitly bounded levels')
        labels=set();values=[];members=[]
        for row in levels:
            if not isinstance(row,dict) or set(row)!={'label','value'}:raise ValueError('Each refinement level needs a safe label and exact positive value')
            label=safe_id(row['label']);value=_positive(row['value'],'native refinement level')
            if label in labels:raise ValueError('Repeated refinement level label')
            if path in {('sidm','maximum_pair_probability'),('sidm','maximum_particle_probability'),('sidm','maximum_transfer_depth')} and value>=1:
                raise ValueError('Probability/transfer-depth refinement levels must remain below one')
            if path==('sidm','split_angle_radians') and value>=(math.pi/2 if run['sidm']['particle_convention']=='folded_identical' else math.pi):
                raise ValueError('Split-angle refinement must retain a nonempty actual rare angular domain')
            if path==('sidm','kernel_support_proper_kpc') and value>=template['population_request']['box_proper_kpc']/2:
                raise ValueError('Kernel refinement must retain unique homogeneous periodic support')
            labels.add(label);values.append(value)
            variant=deepcopy(original);variant['name']=safe_id(name+'_'+label)
            target=variant['run_request']
            for member in path[:-1]:target=target[member]
            target[path[-1]]=row['value']
            validate_control_run_request({'request':variant['run_request']})
            # Change no other numerical/scientific request fields. Complete
            # native SIDM preflight later enforces the selected model domain.
            generated.append(variant);members.append(variant['name'])
        if any(right>=left for left,right in zip(values,values[1:])):
            raise ValueError('Homogeneous refinement levels must run coarse-to-fine with strictly decreasing operative parameter values')
        for index in range(len(members)-1):
            comparison=deepcopy(definition)
            validate_control_comparison_request({'request':comparison})
            comparisons.append({'name':safe_id(name+'_adjacent_'+str(index)),'left':members[index],'right':members[index+1],'request':comparison})
        families.append({'name':name,'parameter_path':list(path),'values':values,'variants':members,
            'comparison_definition':definition,'scope':'adjacent level differences and independent paired-realization covariance; no assumed convergence order'})
    variant_names=[row['name'] for row in generated]
    if len(set(variant_names))!=len(variant_names):raise ValueError('Refinement axis/level names collide across families')
    campaign=deepcopy(template);campaign['variants']=generated;campaign['comparisons']=comparisons
    # Validate the complete measurement requests before publishing any campaign.
    from .sidm_control_measurements import validate_control_refinement_measurement_request
    measurements=[]
    for family in families:
        ids=['ensemble_'+family['name']+'_adjacent_'+str(i) for i in range(len(family['values'])-1)]
        measurement={'schema':'sidm-refinement-measurement-request-v1','parameter_path':family['parameter_path'],'levels':family['values'],
            'ensembles':[{'path':{'stage':stage,'artifact':'ensemble.json','value':'path'},'sha256':{'stage':stage,'artifact':'ensemble.json','value':'sha256'}} for stage in ids],
            'maximum_levels':ceiling,'maximum_observables':template['maximum_ensemble_observables'],'memory_gib':resources['memory_gib'],**measurement_limits}
        validate_control_refinement_measurement_request({'request':measurement})
        measurements.append({'stage_id':'refinement_'+family['name'],'kind':'sidm_control_refinement_measurement',
            'parameters':{'request':measurement,'request_base':str(base),'destination':'{generation}/refinement.json'},
            'depends_on':ids,'outputs':{'refinement_'+family['name']:'refinement.json'}})
    report=compile_control_campaign(campaign,destination,request_base=base)
    workflow_path=Path(report['workflow']);workflow=json.loads(workflow_path.read_text())
    workflow['stages'].extend(measurements);workflow['requested_products'].extend(row['stage_id'] for row in measurements)
    from .workflow import compile_requested_product_dag,default_registry
    compile_requested_product_dag(workflow,default_registry(),request_base=base)
    atomic_json(workflow_path,workflow);report['workflow_sha256']=file_sha256(workflow_path)
    report['stage_count']=len(workflow['stages']);report['refinement_measurement_count']=len(measurements)
    atomic_json(Path(destination)/'campaign-design.json',report)
    design={'schema':'sidm-homogeneous-refinement-design-v1','request':request,'families':families,
        'workflow':report['workflow'],'workflow_sha256':report['workflow_sha256'],
        'template':{'path':str(source),'sha256':reference['sha256']},'execution_status':'NOT_EXECUTED',
        'source_hashes':{**report['source_hashes'],str(source):reference['sha256'],str(Path(__file__).resolve()):file_sha256(__file__)},
        'science_acceptance':'NOT_ESTABLISHED','refinement_measurements':[row['stage_id'] for row in measurements],
        'remaining_measurement':'Run the generated complete native level/realization and refinement measurement DAG; no execution was performed by this compiler.'}
    atomic_json(Path(destination)/'refinement-design.json',design);return design


def control_refinement_main():
    import argparse
    parser=argparse.ArgumentParser(description='Compile independent three-or-more-level native SIDM timestep/kernel/hybrid/probability refinement workflows')
    parser.add_argument('request',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.request.resolve()
    compile_control_refinement(json.loads(source.read_text()),args.output,request_base=source.parent)
    return 0


def control_campaign_main():
    import argparse
    parser=argparse.ArgumentParser(description='Compile native SIDM physical or mapped expanding-control and paired-realization workflows')
    parser.add_argument('request',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.request.resolve()
    compile_control_campaign(json.loads(source.read_text()),args.output,request_base=source.parent)
    return 0


def control_population_runner(parameters,context):
    request=validate_control_population_request(parameters)
    if request['memory_gib']>context.resources['memory_gib'] or request['disk_gib']>context.resources['disk_gib']:
        raise MemoryError('Control population exceeds its workflow allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Control population escapes the owned workflow generation')
    return publish_control_population(parameters,destination)


def control_population_main():
    import argparse
    parser=argparse.ArgumentParser(description='Generate physical SIDM control populations and matched native ICs without evolution')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();parameters=json.loads(args.parameters.read_text())
    publish_control_population(parameters,args.output)
    return 0


def reference_cdf_runner(parameters,context):
    if parameters['request']['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('Reference CDF exceeds workflow allocation')
    result = publish_reference_cdf(parameters['request'],context.generation/'sidm-reference-cdf.json')
    return {'measurements':{'cdf_absolute_error_bound':result['cdf_absolute_error_bound'],'cdf_nodes':result['grid_nodes']}}


def native_events_runner(parameters,context):
    if parameters['request']['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('Native events exceed workflow allocation')
    result = export_native_events(parameters['request'],parameters['native_stage_receipt'],context.generation/'sidm-native-events.json')
    return {'measurements':{'native_scattering_events':result['event_count'],
        'maximum_pair_energy_relative_change':result['maximum_pair_energy_relative_change'],
        'maximum_pair_momentum_relative_change':result['maximum_pair_momentum_relative_change']}}


def sidm_control_products_main():
    import argparse
    parser = argparse.ArgumentParser(description='Publish SIDM physical CDF references or actual native event measurements')
    parser.add_argument('kind',choices=['reference-cdf','native-events'])
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--native-stage-receipt',type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep previous native control products; choose a fresh output')
    request = json.loads(args.request.read_text())
    if args.kind == 'reference-cdf':
        if args.native_stage_receipt:
            parser.error('Physical CDF references do not assert native evolution')
        publish_reference_cdf(request,args.output)
    else:
        if not args.native_stage_receipt:
            parser.error('Actual native event export requires its completed evolution stage receipt')
        export_native_events(request,args.native_stage_receipt,args.output)
# @research2-end-module sidm_control_products












