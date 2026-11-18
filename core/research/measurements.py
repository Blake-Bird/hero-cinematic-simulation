"""Measure native observables and collect paired controls into error-budget workflows.

All scientific tolerances and input units are declared by the caller.
"""
from __future__ import annotations

import json
import numpy as np
import zipfile
from pathlib import Path
from core.foundation import canonical_digest, file_sha256, atomic_json, atomic_text, G_MPC_KMS2_MSUN, SECONDS_PER_YEAR
from core.research.errors import _retained_json, COMPONENT_CONTROLS
from copy import deepcopy
from core.native import native_run_lock




# Native convergence measurements, including paired spatial jackknife covariance.
PROFILE_UNITS = {
    'density': 'Msun / proper kpc^3',
    'enclosed_mass': 'Msun',
    'circular_speed': 'km / s',
    'velocity_dispersion_3d': 'km / s',
    'angular_momentum_magnitude': 'Msun proper kpc km / s',
}


def validate_measurement_request(request):
    if request.get('schema') != 'native-convergence-measurement-request-v1':
        raise ValueError('Explicit native convergence measurement request required')
    for key in ('maximum_bins', 'maximum_particles'):
        if type(request[key]) is not int or request[key] < 1:
            raise ValueError('Positive integer measurement bound required: '+key)
    memory = request['memory_gib']
    if isinstance(memory, bool) or not np.isfinite(memory) or memory <= 0:
        raise ValueError('Positive finite measurement workspace required')
    definition = request['observable_definition']
    if not isinstance(definition, dict) or not definition.get('observable'):
        raise ValueError('Frozen explicit observable definition required')
    bins = definition['bins']
    if not isinstance(bins, list) or not 1 <= len(bins) <= request['maximum_bins']:
        raise ValueError('Measurement bins exceed their explicit bound')
    if len({canonical_digest(q) for q in bins}) != len(bins):
        raise ValueError('Repeated observable domain bin')
    if len(bins)**2*64 > memory*1024**3:
        raise MemoryError('Measured dense covariance exceeds workspace')
    epoch = request['comparison_epoch']
    if not isinstance(epoch,dict) or epoch.get('coordinate') not in {'actual_scale_factor','native_time_code','elapsed_time_gyr'}:
        raise ValueError('Explicit measured epoch coordinate required')
    for key in ('value','absolute_tolerance'):
        if isinstance(epoch[key],bool) or not np.isfinite(epoch[key]) or (key == 'absolute_tolerance' and epoch[key] < 0):
            raise ValueError('Finite epoch value and nonnegative declared tolerance required')
    tolerance = request['covariance_tolerance']
    if isinstance(tolerance,bool) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Finite nonnegative covariance roundoff tolerance required')
    mappings = request['native_parameter_bindings']
    if not isinstance(mappings,dict):
        raise ValueError('Explicit native-manifest parameter bindings required')
    for name,row in mappings.items():
        if not isinstance(row.get('manifest_path'),list) or not row['manifest_path'] or any(not isinstance(k,str) or not k for k in row['manifest_path']):
            raise ValueError('Native parameter binding requires an exact manifest key path: '+name)
        factor = row.get('scale',1.)
        if isinstance(factor,bool) or not np.isfinite(factor) or factor <= 0:
            raise ValueError('Native parameter conversion factor must be explicit positive finite')
    mode = request['mode']
    if mode == 'spherical_profile':
        name = definition['estimator']
        if name not in PROFILE_UNITS or definition['units'] != PROFILE_UNITS[name]:
            raise ValueError('Unknown spherical estimator or incorrect physical units')
        edges = np.asarray(definition['radius_edges_proper_kpc'], float)
        if edges.shape != (len(bins)+1,) or not np.isfinite(edges).all() or edges[0] < 0 or np.any(np.diff(edges) <= 0):
            raise ValueError('Strict increasing proper-kpc radius edges required')
        for index,domain in enumerate(bins):
            if not isinstance(domain,dict) or domain.get('radius_proper_kpc') != edges[index:index+2].tolist():
                raise ValueError('Published profile bin must name its actual proper-kpc radius bounds')
        center = np.asarray(definition['center_proper_kpc'], float)
        velocity = np.asarray(definition['bulk_velocity_peculiar_kms'], float)
        if center.shape != (3,) or velocity.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(velocity).all():
            raise ValueError('Fixed measured center and velocity frame required')
        species = definition['species']
        if not isinstance(species, list) or not species or len(set(species)) != len(species) or any(type(k) is not int or not 0 <= k < 6 for k in species):
            raise ValueError('Distinct explicit native gravitating species required')
        for key in ('minimum_particles_per_bin', 'minimum_effective_particles', 'minimum_particles_per_jackknife_group'):
            value = definition[key]
            if isinstance(value, bool) or not np.isfinite(value) or value < 1:
                raise ValueError('Explicit positive sampling support threshold required: '+key)
        # Common fixed spatial partitions preserve pairing across mass resolution.
        if definition['jackknife_partition'] != 'fixed_center_octants':
            raise ValueError('Recorded fixed center octant jackknife partition required')
        if definition.get('periodic_box_proper_kpc') is not None:
            box = definition['periodic_box_proper_kpc']
            if isinstance(box, bool) or not np.isfinite(box) or box <= 0 or edges[-1] > box/2:
                raise ValueError('Periodic profile must fit within a positive half-box')
    elif mode != 'numeric_product':
        raise ValueError('Unknown native convergence measurement mode')
    return request


def _merge_sources(destination, values):
    for path, checksum in values.items():
        absolute = str(Path(path).resolve())
        if absolute in destination and destination[absolute] != checksum:
            raise ValueError('Convergence input source bindings disagree')
        destination[absolute] = checksum


def _load_json(path, memory_bytes):
    path = Path(path).resolve()
    if path.stat().st_size*8 > memory_bytes:
        raise MemoryError('Measurement JSON exceeds bounded workspace')
    before = file_sha256(path)
    value = json.loads(path.read_text())
    if before != file_sha256(path):
        raise ValueError('Measurement input changed while reading')
    return value, {str(path): before}


def measurement_input_bindings(parameters):
    """Freeze external numeric bundles and native snapshots before stage execution."""
    from core.halos import _snapshot_files
    from core.data import native_publication_markers, numeric_audit_source_bindings
    request = validate_measurement_request(parameters['request'])
    memory = request['memory_gib']*1024**3
    sources = {}
    _, binding = _load_json(parameters['variant_metadata'], memory)
    _merge_sources(sources, binding)
    manifest_path = parameters.get('native_run_manifest')
    if manifest_path is None and parameters.get('native_run_directory'):
        manifest_path = Path(parameters['native_run_directory'])/'manifest.json'
    if manifest_path is None:
        raise ValueError('Actual completed native run manifest is required')
    _, binding = _load_json(manifest_path,memory)
    _merge_sources(sources,binding)
    if parameters.get('native_stage_receipt'):
        from core.workflow import verify_completed_receipt
        receipt_path = Path(parameters['native_stage_receipt']).resolve()
        receipt = verify_completed_receipt(receipt_path)
        _merge_sources(sources, {str(receipt_path): file_sha256(receipt_path), **receipt['source_hashes'], **receipt['input_hashes']})
        _merge_sources(sources, {str(Path(receipt['generation'])/name): sha for name, sha in receipt['files_sha256'].items()})
    if request['mode'] == 'spherical_profile':
        files = _snapshot_files(Path(parameters['native_run_directory']), request['snapshot'])
        _merge_sources(sources, {str(p.resolve()): file_sha256(p) for p in [*files, *native_publication_markers(files)]})
    else:
        report, binding = _load_json(parameters['numeric_product'], memory)
        _merge_sources(sources, binding)
        seal = Path(parameters['numeric_product']).resolve().with_name(Path(parameters['numeric_product']).name+'.sha256')
        _merge_sources(sources,{str(seal):file_sha256(seal)})
        _merge_sources(sources, numeric_audit_source_bindings(report))
        archive = Path(parameters['numeric_product']).resolve().with_name(Path(parameters['numeric_product']).name+'.data')/'arrays.npz'
        for p in archive.parent.rglob('*'):
            if p.is_file():
                _merge_sources(sources, {str(p): file_sha256(p)})
    reference = parameters.get('paired_reference')
    if reference:
        path = Path(reference).resolve()
        seal = path.with_name(path.name+'.sha256')
        if seal.read_text().strip() != file_sha256(path):
            raise ValueError('Paired baseline measurement seal is absent or changed')
        _merge_sources(sources,{str(seal):file_sha256(seal)})
        record, binding = _load_json(reference, memory)
        _merge_sources(sources, binding)
        _merge_sources(sources, record['source_hashes'])
    return sources


def _estimate_profile(displacement, velocity, mass, edges, estimator, scale=1.):
    """Measure physical shells from proper kpc, peculiar km/s and solar masses.

    Density is M_shell / [(4 pi/3)(r_outer^3 - r_inner^3)]. Enclosed mass
    includes the unresolved inner sphere; v_c = sqrt(G M(<r) / r) is evaluated
    at each outer edge. The 1000 converts G's Mpc length unit to kpc.
    Sampling support uses N_eff = (sum m)^2 / sum(m^2), without changing the
    actual particle count. ``scale`` rescales extensive jackknife estimates.
    """
    radius = np.linalg.norm(displacement, axis=1)
    count = np.histogram(radius, edges)[0]
    shell_mass = np.histogram(radius, edges, weights=mass)[0]
    shell_mass2 = np.histogram(radius, edges, weights=mass*mass)[0]
    effective = np.divide(shell_mass**2, shell_mass2, out=np.zeros_like(shell_mass), where=shell_mass2 > 0)
    if estimator == 'density':
        values = shell_mass*scale / (4*np.pi/3*np.diff(edges**3))
    elif estimator in {'enclosed_mass', 'circular_speed'}:
        # Include all particles below the first edge, rather than losing core mass.
        inner = radius < edges[0]
        values = (np.cumsum(shell_mass)+mass[inner].sum())*scale
        count = np.cumsum(count)+int(inner.sum())
        variance_mass = np.cumsum(shell_mass2)+np.dot(mass[inner], mass[inner])
        effective = np.divide((values/scale)**2, variance_mass, out=np.zeros_like(values), where=variance_mass > 0)
        if estimator == 'circular_speed':
            values = np.sqrt(G_MPC_KMS2_MSUN*1000*values/edges[1:])
    elif estimator == 'angular_momentum_magnitude':
        angular = mass[:, None]*np.cross(displacement, velocity)
        vector = np.column_stack([np.histogram(radius, edges, weights=angular[:, k])[0] for k in range(3)])
        values = np.linalg.norm(vector*scale, axis=1)
    elif estimator == 'velocity_dispersion_3d':
        first = np.column_stack([np.histogram(radius, edges, weights=mass*velocity[:, k])[0] for k in range(3)])
        mean = np.divide(first, shell_mass[:, None], out=np.zeros_like(first), where=shell_mass[:, None] > 0)
        # Sum m |v - mean(v)|^2 directly. E[v^2] - E[v]^2 loses a small
        # dispersion when a shell has a large coherent streaming velocity.
        included = (radius >= edges[0]) & (radius <= edges[-1])
        bin_index = np.searchsorted(edges, radius[included], side='right') - 1
        bin_index = np.minimum(bin_index, len(shell_mass) - 1)  # last edge is included
        residual = velocity[included] - mean[bin_index]
        second = np.histogram(radius[included], edges,
                              weights=mass[included]*np.sum(residual*residual, axis=1))[0]
        dispersion2 = np.divide(second, shell_mass, out=np.zeros_like(second), where=shell_mass > 0)
        values = np.sqrt(dispersion2)
    else:
        raise ValueError('Unknown native spherical profile estimator: '+str(estimator))
    return values, count, effective


def measure_spherical_profile(parameters):
    """Measure a profile and its conditional eight-octant sampling covariance.

    For J=8 leave-one-octant-out profiles theta_j, the jackknife covariance is
    (J-1)/J sum_j (theta_j - mean)(theta_j - mean).T. This is a spatial
    sampling diagnostic of this snapshot; it does not estimate cosmic variance
    or independently seeded SIDM scatter.
    """
    from core.observation_geometry import read_native_gravitating_particles
    request = parameters['request']
    definition = request['observable_definition']
    report, particles = read_native_gravitating_particles(parameters['native_run_directory'], request['snapshot'],
        species=definition['species'], maximum_particles=request['maximum_particles'], memory_gib=request['memory_gib']/3,
        namespace=request['namespace'])
    displacement = particles['position_proper_kpc']-np.asarray(definition['center_proper_kpc'])
    box = definition.get('periodic_box_proper_kpc')
    if box is not None:
        displacement = (displacement+box/2)%box-box/2
    velocity = particles['velocity_peculiar_kms']-np.asarray(definition['bulk_velocity_peculiar_kms'])
    mass = particles['mass_msun']
    edges = np.asarray(definition['radius_edges_proper_kpc'], float)
    estimator = definition['estimator']
    value, counts, effective = _estimate_profile(displacement, velocity, mass, edges, estimator)
    octant = np.sum((displacement >= 0)*np.array([1, 2, 4]), axis=1)
    inside_outer_radius = np.linalg.norm(displacement, axis=1) <= edges[-1]
    replicates, masks, populations = [], [], []
    for label in range(8):
        keep = octant != label
        v, n, neff = _estimate_profile(displacement[keep], velocity[keep], mass[keep], edges, estimator, 8/7)
        replicates.append(v)
        masks.append((n >= definition['minimum_particles_per_bin']) & (neff >= definition['minimum_effective_particles']))
        populations.append(int(np.sum((octant == label) & inside_outer_radius)))
    replicates = np.asarray(replicates)
    supported = (counts >= definition['minimum_particles_per_bin']) & (effective >= definition['minimum_effective_particles'])
    supported &= np.logical_and.reduce(masks)
    supported &= min(populations) >= definition['minimum_particles_per_jackknife_group']
    residual = replicates-replicates.mean(axis=0)
    covariance = 7/8*residual.T@residual
    return value, covariance, supported, report['epoch'], report['source_hashes'], {
        'method': 'delete-one fixed spatial octant jackknife',
        'mass_rescaling': '8/7 for each deleted spatial octant; dispersion uses retained mass-weighted moments', 'replicate_values': replicates.tolist(),
        'replicate_labels': list(range(8)), 'replicate_supported': np.asarray(masks).tolist(),
        'partition_definition_hash': canonical_digest({k: definition.get(k) for k in
            ('center_proper_kpc', 'periodic_box_proper_kpc', 'radius_edges_proper_kpc', 'jackknife_partition')}),
        'covariance_factor': 7/8, 'native_particle_count': len(mass), 'bin_particle_count': counts.tolist(),
        'effective_particle_count': effective.tolist(), 'octant_particle_count': populations,
        'scope': 'conditional spatial sampling uncertainty; independent phase/operator scatter remains separate',
    }


def measure_numeric_product(parameters):
    from core.data import verify_numeric_audit, numeric_audit_arrays_path, numeric_audit_source_bindings
    request = parameters['request']
    definition = request['observable_definition']
    path = Path(parameters['numeric_product']).resolve()
    archive = path.with_name(path.name+'.data')/'arrays.npz'
    with zipfile.ZipFile(archive) as packed:
        if sum(q.file_size for q in packed.infolist())*8 > request['memory_gib']*1024**3:
            raise MemoryError('Expanded native measurement product exceeds workspace')
    report = verify_numeric_audit(path)
    if report.get('observable_definition_hash') != canonical_digest(definition) or report.get('observable') != definition['observable']:
        raise ValueError('Measured product does not bind the exact frozen observable definition')
    if report.get('units_of_observable') != definition['units'] or report.get('bins') != definition['bins']:
        raise ValueError('Measured product units/domain differ from the frozen request')
    with np.load(numeric_audit_arrays_path(path, report), allow_pickle=False) as arrays:
        value = arrays[request['value_array']].copy()
        covariance = arrays[request['covariance_array']].copy()
        supported = arrays[request['support_array']].copy()
    uncertainty = report['measurement_uncertainty']
    if not uncertainty.get('method') or not uncertainty.get('scope'):
        raise ValueError('Native measured uncertainty must record its method and physical scope')
    return value, covariance, supported, report['epoch'], numeric_audit_source_bindings(report), uncertainty


def produce_convergence_measurement(parameters):
    from core.research.errors import _psd
    request = validate_measurement_request(parameters['request'])
    memory = request['memory_gib']*1024**3
    sources = measurement_input_bindings(parameters)
    metadata, bindings = _load_json(parameters['variant_metadata'], memory)
    _merge_sources(sources, bindings)
    specification = Path(metadata['specification_path']).resolve()
    spec, bindings = _load_json(specification, memory)
    if bindings[str(specification)] != metadata['specification_sha256'] or spec != metadata['specification']:
        raise ValueError('Native measurement variant specification changed')
    _merge_sources(sources, bindings)
    manifest_path = parameters.get('native_run_manifest') or Path(parameters['native_run_directory'])/'manifest.json'
    manifest,binding = _load_json(manifest_path,memory)
    _merge_sources(sources,binding)
    if manifest.get('random_seed') != spec['seed'] or type(manifest.get('random_seed')) is not int:
        raise ValueError('Actual native evolution used a different IC phase seed')
    from core.research.convergence import CONTROLS
    applied = {}
    for name in metadata['planned_control_dimensions']:
        if name not in request['native_parameter_bindings']:
            raise ValueError('A planned control lacks an actual native parameter binding: '+name)
        row = request['native_parameter_bindings'][name]
        expected = spec
        for key in CONTROLS[name]:
            expected = expected[key]
        actual = manifest
        for key in row['manifest_path']:
            actual = actual[key]
        factor = row.get('scale',1.)
        if factor != 1.:
            if isinstance(expected,bool) or not isinstance(expected,(int,float)):
                raise ValueError('Numeric unit conversion cannot transform a nonnumeric native control')
            expected = expected*factor
        if canonical_digest(actual) != canonical_digest(expected):
            raise ValueError('Evolved native parameter differs from the planned control: '+name)
        applied[name] = {'manifest_path':row['manifest_path'],'native_value':actual}
    phase = spec['seed']
    if type(phase) is not int or phase < 0:
        raise ValueError('Native phase seed must retain its exact integer value')
    definition = request['observable_definition']
    measured = measure_spherical_profile(parameters) if request['mode'] == 'spherical_profile' else measure_numeric_product(parameters)
    value, covariance, mask, epoch, bindings, uncertainty = measured
    _merge_sources(sources, bindings)
    coordinate = request['comparison_epoch']['coordinate']
    actual = epoch.get(coordinate)
    if actual is None or not np.isfinite(actual) or abs(actual-request['comparison_epoch']['value']) > request['comparison_epoch']['absolute_tolerance']:
        raise ValueError('Actual native epoch lies outside the frozen comparison domain')
    n = len(definition['bins'])
    value, covariance, mask = np.asarray(value, float), np.asarray(covariance, float), np.asarray(mask)
    if value.shape != (n,) or covariance.shape != (n, n) or mask.shape != (n,) or mask.dtype.kind != 'b':
        raise ValueError('Native measurement value/covariance/support shapes disagree')
    indices = np.flatnonzero(mask)
    if not np.isfinite(value[mask]).all():
        raise ValueError('Supported native measurements must be finite')
    _psd(covariance[np.ix_(indices, indices)], len(indices), request['covariance_tolerance'], 'native measurement covariance')
    result = {'schema': 'native-convergence-measurement-v1',
        'run_id': metadata['run_id'], 'observable': definition['observable'],
        'observable_definition_hash': canonical_digest(definition), 'units': definition['units'],
        'bins': definition['bins'], 'epoch': request['comparison_epoch'], 'native_epoch': epoch,
        'common_experiment_signature': metadata['common_signature'],
        'common_model_family_hash': metadata['common_model_family_hash'],
        'phase_seed': phase, 'ic_phase_seed': phase,
        'sidm_operator_seed': spec.get('dark_matter', {}).get('interactions', {}).get('seed'),
        'changed_controls': metadata['changed_controls'], 'value': value.tolist(),
        'covariance': covariance.tolist(), 'supported': mask.tolist(),
        'measurement_uncertainty': uncertainty, 'applied_native_controls':applied,'source_hashes': sources,
        'execution_status': 'COMPLETE', 'science_acceptance': 'MEASUREMENT_ONLY'}
    if 'deterministic_absolute_error_bound' in uncertainty:
        bounds = np.asarray(uncertainty['deterministic_absolute_error_bound'],float)
        if bounds.shape != (n,) or not np.isfinite(bounds[mask]).all() or np.any(bounds[mask]<0):
            raise ValueError('Deterministic observable interval widths must retain finite nonnegative supported bounds')
        result['deterministic_absolute_error_bound'] = np.where(mask,bounds,None).tolist()
    if metadata['changed_controls']:
        result['control_value'] = metadata['control_value']
        if not parameters.get('paired_reference'):
            raise ValueError('Native controlled variants require their actual paired baseline measurement')
        reference, bindings = _load_json(parameters['paired_reference'], memory)
        _merge_sources(sources, bindings)
        for key in ('observable_definition_hash', 'units', 'bins', 'epoch', 'phase_seed', 'common_experiment_signature'):
            if reference[key] != result[key]:
                raise ValueError('Paired measurement changes an unrelated physical definition: '+key)
        reference_mask = np.asarray(reference['supported'])
        if reference_mask.dtype.kind != 'b' or reference_mask.shape != (n,):
            raise ValueError('Paired baseline requires exact boolean support for every bin')
        if reference.get('changed_controls') != [] or reference['run_id'] != metadata['baseline_run_id']:
            raise ValueError('Cross covariance requires the exact planned baseline')
        if request['mode'] == 'spherical_profile':
            previous = reference['measurement_uncertainty']
            for key in ('method', 'replicate_labels', 'partition_definition_hash', 'covariance_factor'):
                if previous[key] != uncertainty[key]:
                    raise ValueError('Paired jackknife changes the common spatial partition')
            left = np.asarray(uncertainty['replicate_values'], float)
            right = np.asarray(previous['replicate_values'], float)
            if left.shape != right.shape:
                raise ValueError('Paired native jackknife replicate domains differ')
            result['paired_cross_covariance'] = (uncertainty['covariance_factor']*(left-left.mean(axis=0)).T@(right-right.mean(axis=0))).tolist()
        else:
            cross = uncertainty.get('paired_cross_covariance')
            if cross is None or uncertainty.get('paired_reference_sha256') != file_sha256(parameters['paired_reference']):
                raise ValueError('Numeric product must retain its measured baseline cross covariance')
            result['paired_cross_covariance'] = cross
        paired = np.flatnonzero(mask & reference_mask)
        cross = np.asarray(result['paired_cross_covariance'], float)
        if cross.shape != (n, n):
            raise ValueError('Measured paired covariance domain differs')
        _psd(np.block([[covariance[np.ix_(paired, paired)], cross[np.ix_(paired, paired)]],
            [cross[np.ix_(paired, paired)].T, np.asarray(reference['covariance'])[np.ix_(paired, paired)]]]),
            2*len(paired), request['covariance_tolerance'], 'native paired joint covariance')
    if parameters.get('native_stage_receipt'):
        from core.workflow import verify_completed_receipt
        receipt_path = Path(parameters['native_stage_receipt']).resolve()
        receipt = verify_completed_receipt(receipt_path)
        root = Path(receipt['generation']).resolve()
        owned = {str(root/name): sha for name, sha in receipt['files_sha256'].items()}
        # Native sources must be actual delivered payloads of the evolved stage.
        if owned.get(str(Path(manifest_path).resolve())) != file_sha256(manifest_path):
            raise ValueError('Actual native run manifest is outside the completed evolution generation')
        native = measured[4]
        if request['mode'] == 'spherical_profile' and not all(owned.get(path) == sha for path, sha in native.items()):
            raise ValueError('Native profile sources are outside the declared completed evolution generation')
        if spec.get('dark_matter',{}).get('interactions',{}).get('kind') == 'elastic_sidm':
            mapping = request['native_parameter_bindings'].get('sidm_stochastic_realization')
            if mapping is None:
                raise ValueError('SIDM operator measurements must bind the actual native operator seed')
            native_seed = manifest
            for key in mapping['manifest_path']:
                native_seed = native_seed[key]
            if type(native_seed) is not int or native_seed != result['sidm_operator_seed']:
                raise ValueError('Actual SIDM operator seed differs from its planned realization')
        result['native_realization_receipt'] = str(receipt_path)
    else:
        raise ValueError('Convergence measurements require a completed native evolution stage receipt')
    for path, checksum in sources.items():
        if file_sha256(path) != checksum:
            raise ValueError('Convergence source changed during measurement')
    # Unsupported bins retain null values/covariance rather than fake zero errors.
    result['value'] = [float(v) if ok else None for v, ok in zip(value, mask)]
    result['covariance'] = [[float(covariance[i, j]) if mask[i] and mask[j] else None for j in range(n)] for i in range(n)]
    if 'paired_cross_covariance' in result:
        cross = np.asarray(result['paired_cross_covariance'], float)
        refmask = reference_mask
        result['paired_cross_covariance'] = [[float(cross[i, j]) if mask[i] and refmask[j] else None for j in range(n)] for i in range(n)]
    return result


def convergence_measurement_runner(parameters, context):
    if parameters['request']['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('Native measurement exceeds its workflow allocation')
    record = produce_convergence_measurement(parameters)
    atomic_json(context.generation/'convergence-measurement.json', record)
    atomic_text(context.generation/'convergence-measurement.json.sha256', file_sha256(context.generation/'convergence-measurement.json')+'\n')
    return {'measurements': {'supported_bins': sum(record['supported']), 'native_observable_bins': len(record['bins'])},
        'scientific_exclusions': [{'bin': b, 'reason': 'native sampling support is insufficient'} for b, ok in zip(record['bins'], record['supported']) if not ok]}


def convergence_measurement_main():
    import argparse
    parser = argparse.ArgumentParser(description='Measure an evolved native convergence observable with paired covariance')
    parser.add_argument('--parameters', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    parameters = json.loads(args.parameters.read_text())
    record = produce_convergence_measurement(parameters)
    if args.output.exists():
        raise FileExistsError('Keep native measurement evidence; choose a fresh output')
    atomic_json(args.output, record)
    atomic_text(args.output.with_name(args.output.name+'.sha256'), file_sha256(args.output)+'\n')


def attach_native_measurement_stage(workflow_template, settings):
    """Append a concrete measurement producer to the actual native evolution DAG.

    The campaign supplies immutable variant metadata and the real baseline
    product path. Native run directories remain inside the evolution stage's
    permanent generation, so hashes and reference pairing survive resume.
    """
    from copy import deepcopy
    from core.workflow import safe_id
    workflow = deepcopy(workflow_template)
    request = validate_measurement_request(settings['request'])
    native_id = safe_id(settings['native_stage_id'])
    sid = safe_id(settings.get('measurement_stage_id','native_measurement'))
    product_name = safe_id(settings.get('product_name','convergence_measurement'))
    reference_placeholder = safe_id(settings.get('reference_placeholder','reference_measurement'))
    if any(s['stage_id'] == sid or product_name in s['outputs'] for s in workflow['stages']):
        raise ValueError('Campaign already defines a convergence measurement producer')
    native = next((s for s in workflow['stages'] if s['stage_id'] == native_id),None)
    if native is None:
        raise ValueError('Native measurement refers to an absent evolution producer')
    if native['kind'] != 'public_command' or native['parameters']['command'] != 'gadget':
        raise ValueError('Built-in native measurement must depend on actual public GADGET evolution')
    artifact = settings['native_run_artifact']
    if artifact not in native['outputs'].values():
        raise ValueError('Native run directory must be a declared evolved product artifact')
    parameters = {'request':request, 'variant_metadata':{'input':'{variant_metadata}'},
        'native_stage_receipt':{'stage':native_id,'artifact':'stage-receipt.json'},
        'native_run_directory':{'stage':native_id,'artifact':artifact},
        'paired_reference':'{'+reference_placeholder+'}'}
    if request['mode'] == 'numeric_product':
        if settings.get('isolated_history'):
            history_settings = settings['isolated_history']
            history_id = safe_id(history_settings.get('stage_id','native_isolated_history'))
            observable_id = safe_id(history_settings.get('observable_stage_id','native_history_observable'))
            if history_id == observable_id or any(row['stage_id'] in {history_id,observable_id} for row in workflow['stages']):
                raise ValueError('Native history/observable stage identities collide')
            workflow['stages'].append({'stage_id':history_id,'kind':'isolated_control_history',
                'parameters':{'request':deepcopy(history_settings['request']),
                    'native_run_directory':{'stage':native_id,'artifact':artifact},
                    'native_stage_receipt':{'stage':native_id,'artifact':'stage-receipt.json'}},
                'outputs':{history_id+'_product':'isolated-control-history.json'},'depends_on':[native_id]})
            workflow['stages'].append({'stage_id':observable_id,'kind':'isolated_observable',
                'parameters':{'observable_definition':deepcopy(request['observable_definition']),
                    'history':{'receipt':{'stage':history_id,'artifact':'stage-receipt.json'},'product':history_id+'_product'},
                    'paired_reference':'{'+reference_placeholder+'}','memory_bytes':int(request['memory_gib']*1024**3),
                    'maximum_bins':request['maximum_bins'],'request_base':'{request_base}',
                    'destination':'{generation}/isolated-observable.json'},
                'outputs':{observable_id+'_product':'isolated-observable.json'},'depends_on':[history_id]})
            parameters['numeric_product'] = {'stage':observable_id,'artifact':'isolated-observable.json'}
        elif not settings.get('numeric_product'):
            raise ValueError('Numeric observables require their actual produced measurement bundle')
        else:
            parameters['numeric_product'] = deepcopy(settings['numeric_product'])
    stage = {'stage_id':sid, 'kind':'convergence_measurement', 'parameters':parameters,
        'outputs':{product_name:'convergence-measurement.json'},
        'depends_on':[native_id], 'resources':deepcopy(settings.get('resources',workflow['resources'])),
        'species':deepcopy(request['observable_definition'].get('species',[]))}
    workflow['stages'].append(stage)
    workflow['requested_products'] = list(dict.fromkeys([*workflow['requested_products'],product_name]))
    return workflow


# Connect completed native control families to measured correlated budgets.
def _bounded_json(path,memory_bytes):
    return _load_json(path, memory_bytes)[0]


def _family_inputs(family_path,request_base,memory_bytes):
    from core.workflow import path_at, verify_completed_receipt
    family_path = path_at(request_base,family_path)
    if file_sha256(family_path) != family_path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Completed convergence family publication seal changed')
    family = _bounded_json(family_path,memory_bytes)
    if family.get('schema') != 'executed-convergence-family-v1':
        raise ValueError('Collection requires an actually completed native convergence family')
    plan_path = path_at(request_base,family['plan_path'])
    scope_path = path_at(request_base,family['scope_path'])
    plan = _bounded_json(plan_path,memory_bytes)
    scope = _bounded_json(scope_path,memory_bytes)
    if canonical_digest(plan) != family['plan_hash'] or canonical_digest(scope) != family['scope_hash']:
        raise ValueError('Completed family no longer matches its frozen plan and scope')
    bindings = {str(path):file_sha256(path) for path in (family_path,plan_path,scope_path)}
    plan_runs = {row['run_id']:row for row in plan['runs']}
    if len(plan_runs) != len(plan['runs']):
        raise ValueError('Frozen convergence plan repeats a run identity')
    retained = {}
    for row in family['workflow_receipts']:
        identity = row['run_id']
        if identity not in plan_runs or identity in retained:
            raise ValueError('Completed family repeats a run or publishes an unplanned realization')
        measured,sources = _retained_json(row['measurement_reference'],request_base,memory_bytes)
        _merge_sources(bindings,sources)
        reference = row['measurement_reference']
        measurement_receipt = verify_completed_receipt(path_at(request_base,reference['receipt']))
        product = measurement_receipt['products'][reference['product']]
        product_path = Path(measurement_receipt['generation'])/product['path']
        if file_sha256(product_path) != row['measurement_sha256']:
            raise ValueError('Family measurement differs from its final retained publication')
        native_path = path_at(request_base,row['native_realization_receipt'])
        native = verify_completed_receipt(native_path)
        digest = file_sha256(native_path)
        if digest != row['native_realization_receipt_sha256'] or measured['source_hashes'].get(str(native_path)) != digest:
            raise ValueError('Family measurement does not bind its actual native evolution receipt')
        _merge_sources(bindings,{str(native_path):digest,**native['source_hashes'],**native['input_hashes']})
        _merge_sources(bindings,{str(Path(native['generation'])/name):value for name,value in native['files_sha256'].items()})
        branch = plan_runs[identity]
        if (measured['run_id'] != identity or measured['changed_controls'] != branch['changed_controls']
                or measured['common_experiment_signature'] != branch['common_signature']
                or measured['phase_seed'] != branch['ic_phase_seed']
                or measured.get('sidm_operator_seed') != branch['specification'].get('dark_matter',{}).get('interactions',{}).get('seed')
                or measured['common_model_family_hash'] != plan['common_model_family_hash']):
            raise ValueError('Retained native measurement differs from the frozen experiment branch')
        if branch['changed_controls'] and measured.get('control_value') != branch['control_value']:
            raise ValueError('Retained control value differs from the native experiment plan')
        retained[identity] = {'branch':branch,'measurement':measured,'reference':reference,
            'native_realization_receipt':str(native_path)}
    if set(retained) != set(plan_runs):
        raise ValueError('Completed family omits planned native measurements; incomplete runs cannot form an ensemble')
    return family,plan,scope,retained,bindings


def compile_convergence_budget_collection(request,*,request_base):
    """Build residual -> covariance -> budget DAGs from exact measured controls.

    Each component selector identifies one actual variant setting. Multiple
    levels require separate requests, preserving the numerical contrast. A
    singleton partition remains an explicit unresolved domain; it is never
    assigned zero stochastic variance.
    """
    from core.workflow import positive_number, path_at, safe_id
    if request.get('schema') != 'convergence-budget-collection-request-v1':
        raise ValueError('Expected convergence-budget-collection-request-v1')
    memory = positive_number(request['memory_bytes'],'collection workspace',integer=True)
    maximum_groups = positive_number(request['maximum_groups'],'collection group bound',integer=True)
    maximum_realizations = positive_number(request['maximum_realizations'],'ensemble realization bound',integer=True)
    if maximum_realizations < 2:
        raise ValueError('A correlated error ensemble requires at least two native realizations')
    resources = deepcopy(request['resources'])
    if memory > resources['memory_gib']*1024**3:
        raise MemoryError('Collection workspace exceeds explicit workflow memory')
    independence = request['independence_key']
    if independence not in {'ic_phase_seed','sidm_operator_seed'}:
        raise ValueError('Choose exactly one native stochastic independence axis')
    requirement = deepcopy(request['requirement'])
    if requirement.get('schema') != 'observable-error-requirement-v1':
        raise ValueError('Collection requires a predeclared physical error requirement')
    selectors = request['component_controls']
    stochastic_baseline = request.get('stochastic_baseline',False)
    if type(stochastic_baseline) is not bool:
        raise ValueError('Direct native stochastic baseline selection must be an exact boolean')
    if stochastic_baseline and independence != 'sidm_operator_seed':
        raise ValueError('Direct SIDM observable variance requires independent native operator seeds at fixed IC phase')
    if not isinstance(selectors,list) or (not selectors and not stochastic_baseline):
        raise ValueError('Collection needs an explicit measured component/control matrix')
    labels = []
    for selector in selectors:
        component,control = selector['component'],selector['control']
        if component in labels or control not in COMPONENT_CONTROLS.get(component,set()) or 'value' not in selector:
            raise ValueError('Each component must name one distinct recognized control and its exact variant value')
        labels.append(component)
    if stochastic_baseline:
        if 'sidm_stochastic_realization' in labels:
            raise ValueError('Do not count direct baseline stochastic variance and a stochastic pair contrast as the same component')
        target = 'baseline_plus_declared_signed_control_contrasts' if selectors else 'native_baseline_observable'
        if requirement.get('random_error_target') != target:
            raise ValueError('Frozen stochastic requirement must declare its exact observable covariance target: '+target)
        labels.append('sidm_stochastic_realization')
    if len(requirement['bins'])*len(labels)*maximum_realizations*128 > memory:
        raise MemoryError('Requested collection exceeds its declared bounded metadata workspace')
    family,plan,scope,retained,bindings = _family_inputs(request['family'],request_base,memory)
    controls = request['systematic_controls']
    if not isinstance(controls,list) or len(controls) != len(labels):
        raise ValueError('Each selected component requires a separate retained systematic envelope')
    control_refs = []
    envelope_stages = []
    for label,reference in zip(labels,controls):
        if isinstance(reference,dict) and set(reference) == {'native_envelope'}:
            from core.research.errors import validate_envelope_parameters
            recipe = reference['native_envelope']
            selected = next((row for row in selectors if row['component'] == label),None)
            if selected is None:
                raise ValueError('Generated envelope requires a selected actual native numerical control')
            pairs = []
            for group in plan['realization_groups']:
                baseline = retained[group['baseline_run_id']]
                candidates = [entry for entry in retained.values()
                    if entry['branch']['realization_group_id'] == group['realization_group_id']
                    and entry['branch']['changed_controls'] == [selected['control']]
                    and entry['branch'].get('control_value') == selected['value']]
                if len(candidates) != 1:
                    raise ValueError('Envelope recipe must identify exactly one actual variant per native baseline')
                variant = candidates[0]
                pair = {'reference':baseline['reference'],'variant':variant['reference']}
                if recipe['bound_scope'] == 'native_baseline_error_against_certified_reference':
                    certificate = recipe['certificates_by_variant'][variant['branch']['run_id']]
                    pair['certificate'] = {'receipt':str(path_at(request_base,certificate['receipt'])),
                        'product':certificate['product']}
                    _,certificate_sources = _retained_json(pair['certificate'],request_base,memory)
                    _merge_sources(bindings,certificate_sources)
                pairs.append(pair)
            sid = safe_id('systematic_'+label)
            product = sid+'_product'
            parameters = {'request':requirement,'component':label,'control':selected['control'],
                'bound_scope':recipe['bound_scope'],'pairs':pairs,'memory_bytes':memory,
                'maximum_pairs':recipe['maximum_pairs'],'request_base':str(Path(request_base).resolve()),
                'destination':'{generation}/envelope'}
            validate_envelope_parameters(parameters)
            envelope_stages.append({'stage_id':sid,'kind':'native_error_envelope','parameters':parameters,
                'outputs':{product:'envelope/systematic-envelope.json'}})
            control_refs.append({'receipt':{'stage':sid,'artifact':'stage-receipt.json'},'product':product})
            continue
        record,sources = _retained_json(reference,request_base,memory)
        _merge_sources(bindings,sources)
        if (record.get('component') != label or record.get('requirement_hash') != canonical_digest(requirement)
                or record.get('observable') != requirement['observable'] or record.get('units') != requirement['units']):
            raise ValueError('Systematic envelopes must retain selected component order and exact physical requirement')
        control_refs.append({'receipt':str(path_at(request_base,reference['receipt'])),'product':reference['product']})
    partitions = {}
    for group in plan['realization_groups']:
        fixed = group['ic_phase_seed'] if independence == 'sidm_operator_seed' else group['operator_realization_seed']
        independent = group['operator_realization_seed'] if independence == 'sidm_operator_seed' else group['ic_phase_seed']
        if type(independent) is not int or independent < 0:
            raise ValueError('The selected independence axis has no actual integer native seed')
        partition_key = canonical_digest({'independence_key':independence,'fixed_other_seed':fixed})
        partitions.setdefault(partition_key,{'fixed_other_seed':fixed,'members':[]})['members'].append(group)
    if len(partitions) > maximum_groups:
        raise ValueError('Crossed seed grid exceeds the declared collection group bound')
    stages,wanted,groups,unresolved = envelope_stages,[],[],[]
    # Freeze the campaign files as DAG inputs. Measurements/native payloads are
    # bound by the residual producer; systematic controls by ensemble intake.
    family_files = [{'path':path,'sha256':digest} for path,digest in bindings.items()
        if path in {str(path_at(request_base,request['family'])),family['plan_path'],family['scope_path']}]
    for key,partition in partitions.items():
        members = partition['members']
        if len(members) > maximum_realizations:
            raise ValueError('A seed partition exceeds its exact native realization bound')
        if len(members) < 2:
            unresolved.append({'independence_key':independence,'fixed_other_seed':partition['fixed_other_seed'],
                'realization_count':len(members),'bins':requirement['bins'],
                'reason':'fewer than two independent measured realizations; covariance remains unmeasured'})
            continue
        residual_refs,residual_ids = [],[]
        prefix = 'budget_'+key[:16]
        for index,group in enumerate(members):
            baseline = retained[group['baseline_run_id']]
            observed = baseline['measurement']
            if (observed['observable'] != requirement['observable'] or observed['units'] != requirement['units']
                    or observed['bins'] != requirement['bins']):
                raise ValueError('Native family observable, units or physical bins differ from the budget requirement')
            pairs = []
            for selector in selectors:
                candidates = [entry for entry in retained.values()
                    if entry['branch']['realization_group_id'] == group['realization_group_id']
                    and entry['branch']['changed_controls'] == [selector['control']]
                    and entry['branch'].get('control_value') == selector['value']]
                if len(candidates) != 1:
                    raise ValueError('Component selector must identify exactly one actually measured variant in every seed group')
                pairs.append({'component':selector['component'],'control':selector['control'],
                    'reference':baseline['reference'],'variant':candidates[0]['reference']})
            sid = safe_id(prefix+'_residual_'+str(index))
            product = sid+'_product'
            residual_parameters = {'request':requirement,'pairs':pairs,
                    'realization_id':group['realization_group_id'],
                    'native_realization_receipt':baseline['native_realization_receipt'],
                    'independence_key':independence,'memory_bytes':memory,'maximum_components':len(labels),
                    'request_base':str(Path(request_base).resolve()),'destination':'{generation}/residual'}
            if stochastic_baseline:
                residual_parameters['baseline_component'] = {'component':'sidm_stochastic_realization',
                    'reference':baseline['reference']}
            stages.append({'stage_id':sid,'kind':'native_control_residual','input_files':family_files,
                'parameters':residual_parameters,
                'outputs':{product:'residual/native-control-residual.json'}})
            residual_refs.append({'receipt':{'stage':sid,'artifact':'stage-receipt.json'},'product':product})
            residual_ids.append(sid)
        ensemble_id = prefix+'_ensemble'
        stages.append({'stage_id':ensemble_id,'kind':'error_ensemble',
            'depends_on':residual_ids+[row['stage_id'] for row in envelope_stages],
            'parameters':{'request':requirement,'realizations':residual_refs,'systematic_controls':control_refs,
                'independence_key':independence,'maximum_realizations':maximum_realizations,
                'memory_bytes':memory,'request_base':str(Path(request_base).resolve()),'destination':'{generation}/ensemble'},
            'outputs':{ensemble_id+'_product':'ensemble/ensemble.json'}})
        budget_id = prefix+'_acceptance'
        parameters = {'request':requirement,
            'measurements':{'stage':ensemble_id,'artifact':'ensemble/measurements.json','load':'json'},
            'joint_covariance':{'stage':ensemble_id,'artifact':'ensemble/joint_covariance.json','load':'json'},
            'refinement_routes':deepcopy(request.get('refinement_routes',{})),
            'refinement_limits':deepcopy(request.get('refinement_limits',
                {'maximum_children':0,'maximum_controls_per_bin':0}))}
        if request.get('reassessment'):
            parameters['reassessment'] = deepcopy(request['reassessment'])
        product = budget_id+'_product'
        stages.append({'stage_id':budget_id,'kind':'error_budget','depends_on':[ensemble_id],
            'parameters':parameters,'outputs':{product:'error-budget.json'}})
        wanted.append(product)
        groups.append({'independence_key':independence,'fixed_other_seed':partition['fixed_other_seed'],
            'realization_count':len(members),'budget_stage':budget_id,'budget_product':product,
            'ensemble_stage':ensemble_id,'residual_stages':residual_ids})
    if len(stages) > resources['maximum_stage_count']:
        raise ValueError('Concrete collection DAG exceeds the declared initial stage bound')
    workflow = {'schema':'research-workflow-v1','resources':resources,
        'followup_limits':deepcopy(request['followup_limits']),
        'physics_spec':deepcopy(scope['workflow_template'].get('physics_spec',{})),
        'requested_products':wanted,'stages':stages}
    return {'schema':'compiled-convergence-budget-collection-v1','workflow':workflow,'groups':groups,
        'unresolved_groups':unresolved,'requirement_hash':canonical_digest(requirement),
        'family_plan_hash':family['plan_hash'],'request_hash':canonical_digest(request),
        'source_hashes':bindings,'science_acceptance':'NOT_ESTABLISHED',
        'scope':'declared measured bins and seed partitions; no parameter or epoch extrapolation'}


def publish_convergence_budget_collection(request,destination,*,request_base,execute=False,resume=False):
    """Retain a reviewable workflow, optionally execute it on explicit invocation."""
    from core.workflow import execute_workflow, verify_completed_receipt
    destination = Path(destination).resolve()
    with native_run_lock(destination,operation='collect native convergence error budgets'):
        compiled = compile_convergence_budget_collection(request,request_base=request_base)
        destination.mkdir(parents=True,exist_ok=True)
        scope_path = destination/'collection-request.json'
        if scope_path.exists() and _bounded_json(scope_path,request['memory_bytes']) != request:
            raise ValueError('Existing collection changes frozen inputs; choose a new destination')
        if (destination/'collection-result.json').exists() and not resume:
            raise FileExistsError('Existing collection execution requires explicit resume')
        atomic_json(scope_path,request)
        atomic_json(destination/'collection-workflow.json',compiled['workflow'])
        atomic_json(destination/'collection-plan.json',compiled)
        if not execute:
            return {'schema':'convergence-budget-collection-publication-v1','execution_status':'NOT_EXECUTED',
                'workflow':str(destination/'collection-workflow.json'),'groups':compiled['groups'],
                'unresolved_groups':compiled['unresolved_groups'],'science_acceptance':'NOT_ESTABLISHED'}
        reports = []
        receipt = None
        atomic_json(destination/'collection-state.json',{
            'schema':'convergence-budget-collection-state-v1','execution_status':'RUNNING',
            'request_hash':compiled['request_hash'],'science_acceptance':'NOT_ESTABLISHED'})
        if compiled['groups']:
            try:
                receipt = execute_workflow(compiled['workflow'],destination/'workflow',request_base=request_base,resume=resume)
            except BaseException as error:
                # The underlying workflow retains failed generations and their
                # physical products. Keep an explicit collection failure too;
                # no partial group is silently promoted to an accepted budget.
                import time,uuid
                failure = {'schema':'convergence-budget-collection-state-v1',
                    'execution_status':'INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED',
                    'request_hash':compiled['request_hash'],'finished_unix':time.time(),
                    'failure':{'type':type(error).__name__,'message':str(error)},
                    'unresolved_groups':compiled['unresolved_groups'],
                    'science_acceptance':'NOT_ESTABLISHED'}
                history = destination/'failed-attempts'/uuid.uuid4().hex
                history.mkdir(parents=True,exist_ok=False)
                atomic_json(history/'collection-state.json',failure)
                atomic_json(destination/'collection-state.json',failure)
                raise
            if receipt.get('execution_status') != 'COMPLETE':
                raise RuntimeError('Measured budget collection workflow did not complete')
            for group in compiled['groups']:
                stage_path = Path(receipt['stages'][group['budget_stage']]['generation'])/'stage-receipt.json'
                stage = verify_completed_receipt(stage_path)
                report_path = Path(stage['generation'])/stage['products'][group['budget_product']]['path']
                report = _bounded_json(report_path,request['memory_bytes'])
                reports.append({**group,'budget':report,'receipt':str(stage_path),
                    'budget_path':str(report_path),'budget_sha256':file_sha256(report_path)})
        for path,digest in compiled['source_hashes'].items():
            if file_sha256(path) != digest:
                raise ValueError('Retained family input changed before budget collection publication')
        result = {'schema':'measured-convergence-budget-collection-v1','execution_status':'COMPLETE',
            'request_hash':compiled['request_hash'],'family_plan_hash':compiled['family_plan_hash'],
            'budgets':reports,'unresolved_groups':compiled['unresolved_groups'],'workflow_receipt':receipt,
            'science_acceptance':'DECLARED_DOMAINS_ONLY' if reports and not compiled['unresolved_groups']
                and all(row['budget']['science_acceptance'] == 'ACCEPTED_DECLARED_BINS' for row in reports)
                else 'UNRESOLVED','source_hashes':compiled['source_hashes']}
        import uuid
        generation = destination/'results'/uuid.uuid4().hex
        generation.mkdir(parents=True,exist_ok=False)
        atomic_json(generation/'collection-result.json',result)
        atomic_text(generation/'collection-result.sha256',file_sha256(generation/'collection-result.json')+'\n')
        atomic_json(destination/'collection-result.json',result)
        atomic_text(destination/'collection-result.sha256',file_sha256(destination/'collection-result.json')+'\n')
        atomic_json(destination/'collection-state.json',{
            'schema':'convergence-budget-collection-state-v1','execution_status':'COMPLETE',
            'request_hash':compiled['request_hash'],'result_generation':str(generation),
            'science_acceptance':result['science_acceptance']})
        return result


def convergence_budget_collection_main():
    import argparse
    parser = argparse.ArgumentParser(description='Connect completed native convergence families to correlated error budgets')
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    request = _bounded_json(args.request,16*1024**2)
    result = publish_convergence_budget_collection(request,args.output,
        request_base=args.request.resolve().parent,execute=args.execute,resume=args.resume)
    print(json.dumps(result,indent=2))


# Native saved-history observables for convergence campaigns.
ESTIMATORS = {
    'total_energy':('total_energy','Msun (km / s)^2'),
    'energy_relative_drift':('energy_relative_drift','dimensionless'),
    'mass_relative_drift':('mass_relative_drift','dimensionless'),
    'core_density':('core_density_msun_proper_kpc3','Msun / proper kpc^3'),
    'core_mass':('core_mass_msun','Msun'),
    'persistent_saved_crossing_bracket_midpoint':(None,'Gyr'),
}


def validate_isolated_observable(parameters):
    definition = parameters['observable_definition']
    if definition.get('estimator') not in ESTIMATORS or not definition.get('observable'):
        raise ValueError('Explicit supported native saved-history observable required')
    if definition['units'] != ESTIMATORS[definition['estimator']][1]:
        raise ValueError('Native saved-history estimator physical units differ')
    bins = definition['bins']
    if not isinstance(bins,list) or not bins or len({canonical_digest(row) for row in bins}) != len(bins):
        raise ValueError('Distinct nonempty physical saved-history bins required')
    for key in ('memory_bytes','maximum_bins'):
        if type(parameters[key]) is not int or parameters[key] <= 0:
            raise ValueError('Positive explicit isolated observable bound required: '+key)
    if len(bins) > parameters['maximum_bins'] or len(bins)**2*128 > parameters['memory_bytes']:
        raise MemoryError('Saved-history observable exceeds its declared bin/covariance workspace')
    if definition.get('uncertainty_scope') != 'conditional_deterministic_saved_history':
        raise ValueError('Declare conditional saved-product reduction separately from native realization scatter')


def isolated_observable_input_bindings(parameters):
    validate_isolated_observable(parameters)
    _,sources = _retained_json(parameters['history'],parameters['request_base'],parameters['memory_bytes'])
    reference = parameters.get('paired_reference')
    if reference:
        path = Path(reference).resolve()
        if path.stat().st_size*8 > parameters['memory_bytes']:
            raise MemoryError('Paired saved-history baseline exceeds workspace')
        if file_sha256(path) != path.with_name(path.name+'.sha256').read_text().strip():
            raise ValueError('Paired native observable publication seal changed')
        sources[str(path)] = file_sha256(path)
        sources[str(path.with_name(path.name+'.sha256'))] = file_sha256(path.with_name(path.name+'.sha256'))
    return sources


def publish_isolated_observable(parameters,destination):
    from core.data import export_numeric_audit
    from core.data import isolated_epoch_contract
    validate_isolated_observable(parameters)
    definition = parameters['observable_definition']
    history,sources = _retained_json(parameters['history'],parameters['request_base'],parameters['memory_bytes'])
    sources.update(isolated_observable_input_bindings(parameters))
    if history.get('schema') != 'isolated-halo-control-history-v1' or history.get('execution_status') != 'COMPLETE':
        raise ValueError('Saved-history observables require actually completed native history measurements')
    physical = history['request']
    for key in ('core_radius_proper_kpc','center_frame','potential_gauge','boundary_condition',
                'collapse_density_ratio','minimum_consecutive_crossings'):
        if definition.get(key) != physical[key]:
            raise ValueError('Frozen history observable changes its physical measurement definition: '+key)
    bins = definition['bins']
    n = len(bins)
    value,lower,upper = np.full(n,np.nan),np.full(n,np.nan),np.full(n,np.nan)
    supported = np.zeros(n,dtype=bool)
    exclusions = []
    estimator = definition['estimator']
    collapse_estimator = estimator == 'persistent_saved_crossing_bracket_midpoint'
    for index,domain in enumerate(bins):
        if collapse_estimator:
            if (n != 1 or domain.get('history_native_time_interval') !=
                    [history['epochs'][0]['native_time_code'],history['epochs'][-1]['native_time_code']]):
                raise ValueError('Collapse observable must retain the exact observed native history interval')
            bracket = history['collapse']
            if bracket['status'] != 'BRACKETED':
                exclusions.append({'bin':domain,'reason':bracket.get('reason','collapse time not bracketed'),
                    'collapse_status':bracket['status']})
                continue
            interval = np.asarray(bracket['elapsed_time_gyr_interval'],float)
            if interval.shape != (2,) or not np.isfinite(interval).all() or interval[1] <= interval[0]:
                raise ValueError('Native saved collapse bracket lacks a finite increasing physical interval')
            lower[index],upper[index] = interval
            value[index] = interval[0]+(interval[1]-interval[0])/2
            supported[index] = True
        else:
            if 'native_time_code' not in domain:
                raise ValueError('Every saved-history observable bin needs its actual measured native time')
            candidates = [row for row in history['epochs'] if row['native_time_code'] == domain['native_time_code']]
            if len(candidates) != 1:
                exclusions.append({'bin':domain,'reason':'declared native epoch is not present; no nearest-output substitution'})
                continue
            row = candidates[0]
            if estimator.startswith('core_') and not row['core_supported']:
                exclusions.append({'bin':domain,'reason':'actual native core aperture lacks declared particle support'})
                continue
            measured = row[ESTIMATORS[estimator][0]]
            if not np.isfinite(measured):
                raise ValueError('Supported saved-history observable is nonfinite')
            value[index]=lower[index]=upper[index]=measured
            supported[index]=True
    bound = np.where(supported,(upper-lower)/2,np.nan)
    # Zero covariance here describes a deterministic reduction of retained
    # outputs, not exact physical dynamics, infinite resolution or zero seed
    # scatter. Unsupported rows remain unknown, including unbracketed collapse.
    covariance = np.full((n,n),np.nan)
    positions = np.flatnonzero(supported)
    covariance[np.ix_(positions,positions)] = 0.
    uncertainty = {'method':'deterministic retained native saved-history reduction',
        'scope':'conditional_deterministic_saved_history; native seed scatter and numerical/model bias remain separate',
        'deterministic_absolute_error_bound':np.where(supported,bound,None).tolist(),
        'interval_lower':np.where(supported,lower,None).tolist(),'interval_upper':np.where(supported,upper,None).tolist(),
        'interval_interpretation':'saved persistent-crossing bracket' if collapse_estimator else 'actual retained native reduction',
        'floating_point_reduction_error':'not certified by this conditional covariance'}
    if parameters.get('paired_reference'):
        path = Path(parameters['paired_reference']).resolve()
        baseline = json.loads(path.read_text())
        if (baseline['observable_definition_hash'] != canonical_digest(definition) or baseline['bins'] != bins
                or baseline['measurement_uncertainty'].get('method') != uncertainty['method']):
            raise ValueError('Paired conditional baseline differs from its frozen history observable')
        previous_support = np.asarray(baseline['supported'])
        previous_covariance = np.asarray(baseline['covariance'],float)
        if previous_support.dtype.kind != 'b' or previous_support.shape != (n,) or previous_covariance.shape != (n,n):
            raise ValueError('Paired saved-product baseline dimensions differ')
        old_indices = np.flatnonzero(previous_support)
        if np.any(previous_covariance[np.ix_(old_indices,old_indices)] != 0):
            raise ValueError('Conditional saved-product covariance cannot replace a stochastic baseline covariance')
        cross = np.full((n,n),np.nan)
        cross[np.ix_(positions,old_indices)] = 0.
        uncertainty['paired_cross_covariance'] = np.where(np.isfinite(cross),cross,None).tolist()
        uncertainty['paired_reference_sha256'] = file_sha256(path)
    epoch = history['epochs'][-1]
    report = {'schema':'native-isolated-observable-v1','observable':definition['observable'],
        'observable_definition_hash':canonical_digest(definition),'units_of_observable':definition['units'],'bins':bins,
        'epoch':isolated_epoch_contract(epoch['native_time_code'],
            time_unit_gyr=epoch['time_unit_seconds']/(1e9*SECONDS_PER_YEAR)),
        'measurement_uncertainty':uncertainty,'scientific_exclusions':exclusions,
        'native_stage_receipt':history['native_stage_receipt'],'source_hashes':sources,
        'execution_status':'COMPLETE','science_acceptance':'MEASUREMENT_ONLY'}
    arrays = {'observable_value':value,'conditional_covariance':covariance,'observable_supported':supported,
        'interval_lower':lower,'interval_upper':upper,'deterministic_absolute_error_bound':bound}
    specifications = {name:{'units':definition['units']+' squared' if name == 'conditional_covariance'
        else 'dimensionless' if name == 'observable_supported' else definition['units'],
        'annotations':{'meaning':name.replace('_',' ')}} for name in arrays}
    source = Path(__file__).resolve()
    sources[str(source)] = file_sha256(source)
    for path,digest in sources.items():
        if file_sha256(path) != digest:
            raise ValueError('Native saved-history inputs changed before observable publication')
    return export_numeric_audit(destination,report,arrays,specifications,algorithm=source,
        context={'epoch':report['epoch'],'coordinates':{'frame':'declared isolated history measurement'}},source_hashes=sources)


def isolated_observable_runner(parameters,context):
    if parameters['memory_bytes'] > context.resources['memory_gib']*1024**3:
        raise MemoryError('Native saved-history observable exceeds allocated workspace')
    destination = Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):
        raise ValueError('Saved-history observable escapes owned workflow generation')
    return publish_isolated_observable(parameters,destination)
