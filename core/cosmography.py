"""Authored cosmography campaigns and measured ladder/full-lens intake.

Run selected estimators through commands.py. Requests name their physical
models, real inputs, resource bounds and output directories. Estimator physics
stays in its domain module; this owner handles intake, dispatch and publication.
Measured observation/model assets are preserved byte-for-byte, without inferred
calibration, covariance, selection or posterior samples.
"""
from pathlib import Path
import json
import math
import numpy as np
from .foundation import atomic_json, atomic_copy, file_sha256, MPC_CM
from .stream_common import finite, positive, integer
from .native import native_run_lock
from .observation_data import (canonical, publish_product, EvaluationReader, fingerprint, reject_truth)


STAGES={'expansion_truth','local_observer','distance_ladder','time_delay','time_delay_reduced','bao_compressed','bao_compressed_posterior',
        'bao','cosmic_chronometer','hubble_comparison','native_trajectories','lens_recovery'}


def validate_campaign_request(request):
    """Require authored stages, ordered dependencies and finite resource limits."""
    if request.get('schema') != 'cosmography-campaign-v1' or not request.get('experiment_id'):
        raise ValueError('Campaign requires schema cosmography-campaign-v1 and experiment_id')
    stages = request.get('stages', [])
    if not isinstance(stages, list) or not stages:
        raise ValueError('No cosmography stages requested')
    identifiers = set()
    for stage in stages:
        stage_id = stage.get('stage_id', '')
        if not isinstance(stage_id, str) or not stage_id or not stage_id.replace('-', '_').isidentifier() or stage_id in identifiers:
            raise ValueError('Stage IDs must be distinct safe path names')
        if stage.get('kind') not in STAGES:
            raise ValueError('Unknown cosmography stage')
        dependencies = stage.get('depends_on', [])
        if not isinstance(dependencies, list) or any(dep not in identifiers for dep in dependencies):
            raise ValueError('Stage dependencies must precede their consumers')
        identifiers.add(stage_id)
    resources = request['resources']
    positive(resources['memory_gib'], 'memory_gib')
    for key in ('maximum_stage_count', 'maximum_objects', 'maximum_spectral_pixels'):
        integer(resources[key], key)
    if len(stages) > resources['maximum_stage_count']:
        raise ValueError('Campaign stage budget exceeded')
    return request


class StageInputs:
    """Resolve one stage's actual inputs and retain their original identities."""

    def __init__(self, base, published, resource_spec):
        self.base = Path(base).resolve()
        self.published = published
        self.paths = []
        self.opened_hashes = {}
        self.resources = resource_spec
        self.loaded = []

    def path(self, reference):
        if isinstance(reference, dict) and 'stage' in reference:
            root = self.published[reference['stage']].resolve()
            path = (root / reference.get('artifact', 'product.json')).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Stage artifact escapes generation')
        else:
            path = Path(reference)
            if not path.is_absolute():
                path = self.base / path
            path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError('Required scientific input missing: ' + str(path))
        current = file_sha256(path)
        previous = self.opened_hashes.get(str(path))
        if previous is not None and previous != current:
            raise ValueError('Scientific input changed between stage reads: ' + str(path))
        if previous is None:
            self.paths.append(path)
            self.opened_hashes[str(path)] = current
        return path

    def bind_sources(self, sources):
        # Native/backend readers return the hashes of bytes they actually used.
        # Retain those bindings; never silently replace them at publication time.
        for name, expected in sources.items():
            path = self.path(name)
            if self.opened_hashes[str(path)] != expected:
                raise ValueError('Measured/backend source changed: ' + str(path))

    def _read_json_path(self, path):
        if path.stat().st_size * 4 > self.resources['memory_gib'] * 1024**3:
            raise MemoryError('Serialized scientific input exceeds bounded in-memory stage capacity')
        value = json.loads(path.read_text())
        self.loaded.append(value)
        estimate_stage_memory({'kind': 'input'}, self.loaded, self.resources)
        return value

    def json(self, reference):
        return self._read_json_path(self.path(reference))

    def history(self, reference):
        from .hubble_observers import ExpansionHistory
        if isinstance(reference, dict) and 'class_background' in reference:
            path = self.path(reference['class_background'])
            if path.stat().st_size * 32 > self.resources['memory_gib'] * 1024**3:
                raise MemoryError('CLASS background table exceeds stage workspace')
            return ExpansionHistory.from_class(path, omega_k=reference['omega_k'])
        path = self.path(reference)
        data = self._read_json_path(path)
        return ExpansionHistory(np.asarray(data['redshift']), np.asarray(data['H_kms_mpc']),
                                data['omega_k'], data.get('time_at_highest_redshift_gyr'), self.opened_hashes[str(path)])

    def library(self, reference):
        from zipfile import ZipFile
        from .stellar_observations import SpectralLibrary
        metadata = self.json(reference['metadata'])
        path = self.path(reference['npz'])
        # NPZ is compressed: bound expanded members, not the compressed file size.
        # Library construction retains an immutable copy of its numerical arrays.
        with ZipFile(path) as archive:
            expanded = sum(member.file_size for member in archive.infolist())
        if expanded * 2 > self.resources['memory_gib'] * 1024**3:
            raise MemoryError('Expanded spectral library exceeds stage workspace')
        library = SpectralLibrary.from_npz(path, metadata)
        if len(library.wavelength_angstrom) > self.resources['maximum_spectral_pixels']:
            raise MemoryError('Spectral library exceeds authored wavelength pixel bound')
        return library

    def bound_objects(self, count):
        integer(count, 'object census', 0)
        if count > self.resources['maximum_objects']:
            raise MemoryError('Requested object census exceeds campaign bound')


def native_particle_trajectories(run_dir, snapshots, particle_ids, *, particle_type, namespace, memory_gib):
    """Read selected native identities as no-h comoving Mpc and peculiar km/s.

    Use each dataset's saved cgs/a/h metadata. Proper coordinate conversions
    are divided by the actual scale factor to obtain comoving positions.
    Selected identities must exist at every epoch; births/disappearances are
    never interpolated. Minimum-image unwrapping assumes saved outputs resolve
    motion; it cannot measure an unrecorded extra whole-box winding.
    """
    import h5py
    from .halos import snapshot_sets, _snapshot_native_units
    from .data import (verify_native_hdf5_set, exact_uint64_ids, run_epoch_contract,
                       dataset_to_cgs_factor, native_publication_markers)

    requested = exact_uint64_ids(particle_ids, allow_empty=False)
    if type(particle_type) is not int or not 0 <= particle_type < 6 or not isinstance(namespace, str) or not namespace.strip():
        raise ValueError('Native trajectory request needs an explicit species and identity namespace')
    if not isinstance(snapshots, (list, tuple)) or len(snapshots) < 2:
        raise ValueError('At least two distinct native snapshots are required')
    for snapshot in snapshots:
        integer(snapshot, 'native trajectory snapshot', 0)
    if len(set(snapshots)) != len(snapshots):
        raise ValueError('Native trajectory snapshots must be distinct')
    limit = int(positive(memory_gib, 'trajectory memory_gib') * 1024**3)
    # Reserve headroom for selected arrays, reorder/unwrap temporaries and the
    # returned Python lists. Native I/O chunks use the remaining allocation.
    reserved = len(requested) * len(snapshots) * 512 + 16 * 1024**2
    if reserved > limit:
        raise MemoryError('Selected trajectory census exceeds declared memory budget')
    chunk = max(1, min(250000, (limit - reserved) // 256))
    order_ids = np.argsort(requested)
    sorted_ids = requested[order_ids]
    sets = snapshot_sets(run_dir)
    epochs, positions, velocities = [], [], []
    sources = {}
    box = None
    hubble = None
    for snapshot in snapshots:
        paths = sets.get(snapshot, [])
        verify_native_hdf5_set(paths, require_publication=True)
        length_unit, _, _ = _snapshot_native_units(run_dir, snapshot)
        for path in [*paths, *native_publication_markers(paths)]:
            sources[str(path.resolve())] = file_sha256(path)
        position = np.empty((len(requested), 3))
        velocity = np.empty_like(position)
        seen = np.zeros(len(requested), dtype=bool)
        epoch = None
        for path in paths:
            with h5py.File(path, 'r') as handle:
                header = handle['Header'].attrs
                scale = float(header['Time'])
                h = positive(float(header['HubbleParam']), 'native Hubble parameter')
                if epoch is None:
                    epoch = run_epoch_contract(run_dir, snapshot, scale)
                if abs(epoch['scale_factor'] - scale) > 1e-12:
                    raise ValueError('Native snapshot shards differ in epoch')
                if hubble is None:
                    hubble = h
                if not np.isclose(hubble, h, rtol=1e-12, atol=0):
                    raise ValueError('Trajectory snapshots change the fixed native Hubble unit parameter')
                current_box = positive(float(header['BoxSize']) * length_unit / h / MPC_CM, 'comoving box Mpc')
                if box is None:
                    box = current_box
                if not np.isclose(box, current_box, rtol=1e-10, atol=0):
                    raise ValueError('Trajectory snapshots differ in box size')
                group = handle.get('PartType' + str(particle_type))
                if group is None:
                    continue
                # Gadget snap_io.cc declares x_cgs=x*UnitLength*a/h and
                # v_cgs=v*UnitVelocity*sqrt(a). Read the saved factors, then
                # divide proper x by a to deliver no-h comoving coordinates.
                length = dataset_to_cgs_factor(group['Coordinates'].attrs, scale, h) / scale / MPC_CM
                speed = dataset_to_cgs_factor(group['Velocities'].attrs, scale, h) / 1e5
                for start in range(0, len(group['ParticleIDs']), chunk):
                    stop = start + chunk
                    available = exact_uint64_ids(group['ParticleIDs'][start:stop])
                    # Search the fixed requested census, not a newly sorted
                    # full join for every native chunk. Keep its original order.
                    slots = np.searchsorted(sorted_ids, available)
                    matched = slots < len(sorted_ids)
                    matched[matched] &= sorted_ids[slots[matched]] == available[matched]
                    rows = order_ids[slots[matched]]
                    if np.any(seen[rows]):
                        raise ValueError('Duplicate selected particle across native chunks or shards')
                    if len(rows):
                        coordinates = np.asarray(group['Coordinates'][start:stop])
                        speeds = np.asarray(group['Velocities'][start:stop])
                        if coordinates.shape != (len(available), 3) or speeds.shape != coordinates.shape:
                            raise ValueError('Native phase-space rows differ from particle identities')
                        position[rows] = finite(coordinates[matched] * length, (len(rows), 3), 'comoving coordinates')
                        velocity[rows] = finite(speeds[matched] * speed, (len(rows), 3), 'peculiar velocities')
                        seen[rows] = True
        if not seen.all():
            raise ValueError('Selected trajectories have missing/born/deleted particle identities')
        epochs.append(epoch)
        positions.append(position)
        velocities.append(velocity)

    redshift = np.asarray([1 / epoch['scale_factor'] - 1 for epoch in epochs])
    order = np.argsort(redshift)
    redshift = redshift[order]
    positions = np.asarray(positions)[order]
    velocities = np.asarray(velocities)[order]
    if np.any(np.diff(redshift) <= 0):
        raise ValueError('Native trajectory epochs must be distinct')
    delta = finite(np.diff(positions, axis=0), name='saved trajectory displacements')
    unwrapped = delta - box * np.rint(delta / box)
    # This guard rejects large resolved jumps. It does not prove that the
    # particle made no additional, unresolved whole-box traversal between saves.
    if not np.isfinite(unwrapped).all() or np.any(np.abs(unwrapped) > .45 * box):
        raise ValueError('Native cadence leaves ambiguous periodic trajectory unwrap')
    positions = finite(np.concatenate([positions[:1], positions[:1] + np.cumsum(unwrapped, axis=0)], axis=0), name='unwrapped trajectories')
    return {
        'role': 'simulation_trajectory_input', 'schema': 'native-observer-trajectories-v1',
        'namespace': namespace, 'object_id': requested.tolist(), 'redshift': redshift.tolist(),
        'position_mpc': positions.tolist(), 'velocity_kms': velocities.tolist(), 'box_mpc': [box] * 3,
        'actual_epochs': [epochs[i] for i in order], 'source_hashes': sources,
        'units': {'position': 'comoving Mpc without h', 'velocity': 'peculiar km/s'},
        'periodic_unwrap': 'minimum-image between saved outputs; additional whole-box winding is not measured',
    }


def execute_stage(stage, inputs):
    """Dispatch only the authored estimator and preserve its scientific inputs.

    Optional backends load inside their selected branch. Expansion truth is
    read only by post-fit comparison/recovery, after estimator fits are frozen.
    Native/backend source bindings and sampler outputs become stage inputs.
    """
    kind = stage['kind']
    parameters = stage.get('parameters', {})
    refs = stage.get('inputs', {})
    artifacts = {}
    rid = stage.get('realization_id')
    if kind in {'bao_compressed', 'bao_compressed_posterior'}:
        from .bao import conditional_bao_inputs
        request = {'schema': 'conditional-compressed-bao-request-v1', 'memory_gib': inputs.resources['memory_gib']}
        for key in ('observations', 'linear_bundle', 'drag_execution', 'drag_log', 'drag_invocation'):
            path = inputs.path(refs[key])
            request[key] = {'path': str(path), 'sha256': file_sha256(path)}
        if kind == 'bao_compressed_posterior':
            from .bao import prepare_bao_posterior
            from core.stream.inference import fit_sampler
            model, prepared, sources = prepare_bao_posterior(request, parameters)
            checkpoint = Path(parameters['checkpoint_directory'])
            if not checkpoint.is_absolute():
                checkpoint = inputs.base / checkpoint
            result, arrays = fit_sampler(model, parameters['sampler'], checkpoint, binding={'stage': stage, 'source_hashes': sources, 'resources': inputs.resources})
            result['bao_distance_dilation_definition'] = prepared
            inputs.path(checkpoint / 'posterior.json')
        else:
            result, arrays, sources = conditional_bao_inputs(request)
        inputs.bind_sources(sources)
        artifacts['numeric_arrays.json'] = json.dumps({'arrays': {k: v.tolist() for k, v in arrays.items()}, 'units': {k: 'dimensionless' for k in arrays}}, allow_nan=False)
        return (result, artifacts)
    if kind == 'expansion_truth':
        from .hubble_observers import export_background_expansion
        return (export_background_expansion(inputs.history(refs['background'])), artifacts)
    if kind == 'native_trajectories':
        ids = inputs.json(refs['particle_ids'])
        inputs.bound_objects(len(ids))
        run = Path(refs['run_dir'])
        run = run if run.is_absolute() else inputs.base / run
        product = native_particle_trajectories(run, parameters['snapshots'], ids, particle_type=parameters['particle_type'], namespace=parameters['namespace'], memory_gib=inputs.resources['memory_gib'])
        inputs.bind_sources(product['source_hashes'])
        return (product, artifacts)
    if kind == 'local_observer':
        from .hubble_observers import Observer, generate_past_lightcone_observations, fit_distance_redshift_relation
        if 'observations' in refs:
            observations = inputs.json(refs['observations'])
        else:
            history = inputs.history(refs['background'])
            trajectories = inputs.json(refs['trajectories'])
            inputs.bound_objects(len(trajectories['object_id']))
            o = parameters['observer']
            observer = Observer(o['observer_id'], np.asarray(o['position_mpc']), np.asarray(o['velocity_kms']), o['environment'], o['selection_rule'])
            generation = dict(parameters['generation'])
            if 'sky_cells' in generation:
                cells = generation.pop('sky_cells')
                generation['sky_mask'] = lambda ra, dec: any((a <= ra <= b and c <= dec <= d for a, b, c, d in cells))
            observations = generate_past_lightcone_observations(history, observer, trajectories, **generation)
        artifacts['observations.json'] = json.dumps(canonical(observations), indent=2, allow_nan=False)
        return (fit_distance_redshift_relation(observations, parameters['fit'], realization_id=rid), artifacts)
    if kind == 'distance_ladder':
        from .stellar_observations import IndicatorPrescription, sample_indicator_populations, render_indicator_measurements, jointly_fit_anchors_and_indicator_relations
        if 'observations' in refs:
            observations = inputs.json(refs['observations'])
        else:
            hosts = inputs.json(refs['hosts'])
            inputs.bound_objects(len(hosts['host_id']))
            model = IndicatorPrescription(**inputs.json(refs['indicator_model']))
            # Dense joint covariance and fit temporaries grow quadratically.
            # Reject an oversized authored population before allocating it;
            # do not reduce its requested counts. This is a headroom estimate.
            fit_object_limit=min(inputs.resources['maximum_objects'],math.isqrt(int(inputs.resources['memory_gib']*1024**3/256)))
            populations = sample_indicator_populations(hosts, model, parameters['populations'], seed=parameters['population_seed'], maximum_objects=fit_object_limit)
            inputs.bound_objects(len(populations['rows']))
            observations = render_indicator_measurements(populations, inputs.json(refs['anchors']), parameters['measurement'], seed=parameters['measurement_seed'])
        artifacts['observations.json'] = json.dumps(canonical(observations), indent=2, allow_nan=False)
        return (jointly_fit_anchors_and_indicator_relations(observations, parameters['fit'], realization_id=rid), artifacts)
    if kind in ('time_delay', 'time_delay_reduced'):
        from .time_delay_cosmography import construct_mass_planes, render_lensed_observations, fit_lenstronomy_model, fit_delay_distance
        if 'observations' in refs:
            observations = inputs.json(refs['observations'])
        else:
            history = inputs.history(refs['background'])
            if 'native_hosts' in refs:
                from .time_delay_cosmography import native_host_mass_planes
                native = inputs.json(refs['native_hosts'])
                run = Path(native['run_dir'])
                if not run.is_absolute():
                    run = inputs.base / run
                components, sources = native_host_mass_planes(run, native['hosts'], memory_gib=inputs.resources['memory_gib'])
                inputs.bind_sources(sources)
            else:
                components = inputs.json(refs['mass_planes'])
            inputs.bound_objects(sum((len(p['particle_id']) for p in components)))
            truth = construct_mass_planes(history, components, z_source=parameters['z_source'], grid_arcsec=parameters['grid_arcsec'])
            observations = render_lensed_observations(history, truth, inputs.json(refs['source']), parameters['instrument'], seed=parameters['seed'], z_lens=parameters['z_lens'], kinematics=inputs.json(refs['kinematics']) if 'kinematics' in refs else None, los_constraints=inputs.json(refs['los_constraints']) if 'los_constraints' in refs else None)
        artifacts['observations.json'] = json.dumps(canonical(observations), indent=2, allow_nan=False)
        fit = inputs.json(refs['lens_fit_model'])
        function = fit_lenstronomy_model if kind == 'time_delay' else fit_delay_distance
        return (function(observations, fit, realization_id=rid), artifacts)
    if kind == 'lens_recovery':
        from .time_delay_cosmography import compare_h0_shape_profile_and_external_convergence_recovery
        return (compare_h0_shape_profile_and_external_convergence_recovery(inputs.json(refs['posterior']), inputs.path(refs['lens_truth']), parameters['truth_sha256']), artifacts)
    if kind == 'bao':
        from .bao import build_selected_tracer_catalogue, generate_survey_window_random_catalogue, apply_recorded_bao_reconstruction, pycorr_measurement, fit_anisotropic_bao_template
        if 'correlation' in refs:
            correlation = inputs.json(refs['correlation'])
        else:
            parent = inputs.json(refs['parent_tracers'])
            inputs.bound_objects(len(parent['object_id']))
            tracers = build_selected_tracer_catalogue(parent, parameters['tracer_selection'], seed=parameters['seed'])
            inputs.bound_objects(parameters['random_count'])
            randoms = generate_survey_window_random_catalogue(inputs.json(refs['survey_window']), count=parameters['random_count'], seed=parameters['random_seed'])
            reconstruction = None
            if parameters.get('reconstruction'):
                reconstruction = apply_recorded_bao_reconstruction(tracers, randoms, parameters['reconstruction'])
                artifacts['reconstruction.json'] = json.dumps(canonical(reconstruction), allow_nan=False)
            correlation = pycorr_measurement(tracers, randoms, parameters['correlation'], reconstructed=reconstruction)
        artifacts['correlation.json'] = json.dumps(canonical(correlation), allow_nan=False)
        return (fit_anisotropic_bao_template(correlation, inputs.json(refs['template']), inputs.json(refs['ensemble']), parameters['fit'], realization_id=rid), artifacts)
    if kind == 'cosmic_chronometer':
        from .stellar_observations import render_passive_spectra, select_from_observed_spectra_and_catalogues, fit_spectral_ages, fit_differential_ages_with_covariance
        if 'ages' in refs:
            ages = inputs.json(refs['ages'])
        else:
            if 'observations' in refs:
                observations = inputs.json(refs['observations'])
            else:
                hosts = inputs.json(refs['hosts'])
                inputs.bound_objects(len(hosts['object_id']))
                observations = render_passive_spectra(hosts, inputs.library(refs['generating_sps']), parameters['instrument'], seed=parameters['seed'])
            pixels = len(observations['calibration']['wavelength_angstrom'])
            if pixels > inputs.resources['maximum_spectral_pixels']:
                raise MemoryError('Spectral pixel budget exceeded')
            passive = select_from_observed_spectra_and_catalogues(observations, parameters['passive_selection'])
            ages = fit_spectral_ages(passive, inputs.library(refs['fitting_sps']), parameters['spectral_fit'])
            artifacts['observations.json'] = json.dumps(canonical(passive), allow_nan=False)
        artifacts['ages.json'] = json.dumps(canonical(ages), allow_nan=False)
        return (fit_differential_ages_with_covariance(ages, parameters['differential_fit'], realization_id=rid), artifacts)
    if kind == 'hubble_comparison':
        from .expansion_products import compare_expansion_methods, comparison_report_files
        methods = {m: [inputs.path(p) for p in paths] for m, paths in refs['methods'].items()}
        truth = inputs.path(refs['expansion_truth'])
        reader = EvaluationReader(truth, parameters['truth_sha256'])
        product = compare_expansion_methods(methods, reader, parameters['accuracy_budgets'], minimum_shared_realizations=parameters['minimum_shared_realizations'])
        return (product, comparison_report_files(product))
    raise ValueError('No handler for requested cosmography stage')


def run_cosmography_campaign(request, output, *, request_base, request_path=None):
    """Publish selected stages or reuse their original immutable generations.

    Reuse checks the saved request, inputs, producer code and payload. Failed
    unpublished generations remain available for diagnosis. Estimator results
    reject truth-bearing fields before their final model/environment freeze.
    """
    request = validate_campaign_request(request)
    output = Path(output).resolve()
    published = {}
    records = []
    with native_run_lock(output, operation='cosmography-campaign'):
        for stage in request['stages']:
            inputs = StageInputs(request_base, published, request['resources'])
            if request_path:
                inputs.path(Path(request_path).resolve())
            destination = output / stage['stage_id']
            if destination.is_dir():
                manifest = json.loads((destination / 'manifest.json').read_text())
                if fingerprint(manifest['request']) != fingerprint(stage):
                    raise FileExistsError('Immutable stage request differs; choose a new generation output')
                for path, digest in manifest['input_hashes'].items():
                    if file_sha256(path) != digest:
                        raise ValueError('Completed stage input changed')
                for name, digest in manifest['source_hashes'].items():
                    source = Path(name)
                    if not source.is_absolute():
                        source = Path(__file__).parent.parent / source if source.parts[0] in {'core', 'configs'} else Path(__file__).parent / source
                    if file_sha256(source) != digest:
                        raise ValueError('Completed stage source changed')
                for name, digest in manifest['files_sha256'].items():
                    if file_sha256(destination / name) != digest:
                        raise ValueError('Completed stage payload changed')
                status = 'REUSED_IMMUTABLE_GENERATION'
            else:
                estimate_stage_memory(stage, [], request['resources'])
                product, files = execute_stage(stage, inputs)
                allocation = estimate_stage_memory(stage, inputs.loaded, request['resources'])
                if product.get('role') == 'frozen_estimator_result':
                    product['environment'] = stage.get('environment', product.get('environment', {}))
                    product['comparison_condition'] = stage.get('comparison_condition', {})
                    reject_truth(product)
                    product.pop('frozen_hash', None)
                    product['frozen_hash'] = fingerprint(product)
                files['resources.json'] = json.dumps(allocation, indent=2)
                for path, expected in inputs.opened_hashes.items():
                    if file_sha256(path) != expected:
                        raise ValueError('Scientific input changed during stage computation')
                manifest = publish_product(destination, product, inputs=inputs.paths, request=stage, files=files)
                status = 'COMPLETED'
            published[stage['stage_id']] = destination
            records.append({
                'stage_id': stage['stage_id'],
                'kind': stage['kind'],
                'execution_status': status,
                'manifest': str(destination / 'manifest.json'),
                'product': str(destination / 'product.json'),
                'scientific_acceptance': manifest['scientific_acceptance'],
            })
            atomic_json(output / 'campaign.json', {
                'schema': 'cosmography-campaign-result-v1',
                'experiment_id': request['experiment_id'],
                'stages': records,
                'execution_status': 'RUNNING',
                'scientific_acceptance': 'UNRESOLVED_NOT_EXECUTED',
            })
        result = {
            'schema': 'cosmography-campaign-result-v1',
            'experiment_id': request['experiment_id'],
            'stages': records,
            'execution_status': 'COMPLETED',
            'scientific_acceptance': 'UNRESOLVED_NOT_EXECUTED',
            'validation_receipts': [],
        }
        atomic_json(output / 'campaign.json', result)
        return result


def cosmography_main():
    import argparse
    parser=argparse.ArgumentParser(description='Execute selected expansion-observer science stages (blocks 41–46).')
    parser.add_argument('request',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.request.resolve()
    result=run_cosmography_campaign(json.loads(path.read_text()),args.output,
                                  request_base=path.parent,request_path=path)
    print(json.dumps(result,indent=2))


def cosmography_report_main():
    import argparse
    from .expansion_products import publish_comparison_figures
    parser=argparse.ArgumentParser(description='Render already published post-fit expansion comparisons.')
    parser.add_argument('comparison',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();product=json.loads(args.comparison.read_text())
    if product.get('role')!='post_fit_evaluation':raise ValueError('Expected frozen post-fit comparison')
    print(json.dumps({'figures':publish_comparison_figures(product,args.output)},indent=2))


def estimate_stage_memory(stage, loaded_inputs, resources):
    """Bound major arrays before execution; this is not a runtime peak guarantee."""
    parameters = stage.get('parameters', {})
    bytes_required = 16 * 1024**2
    if stage['kind'] == 'bao' and parameters.get('reconstruction'):
        mesh = parameters['reconstruction']['nmesh']
        if type(mesh) is int:
            shape = [integer(mesh, 'reconstruction mesh size')] * 3
        elif isinstance(mesh, (list, tuple)) and len(mesh) == 3:
            shape = [integer(size, 'reconstruction mesh axis') for size in mesh]
        else:
            raise ValueError('Reconstruction mesh requires one integer or three integer axes')
        # Python integer multiplication cannot wrap like a fixed-width np.prod.
        # Account for density/displacement/work meshes and pair-count headroom.
        bytes_required += 256 * math.prod(shape)
    for value in loaded_inputs:
        if isinstance(value, dict) and 'columns' in value:
            columns = value['columns']
            count = len(columns.get('object_id', []))
            if count > resources['maximum_objects']:
                raise MemoryError('Stage object budget exceeded')
            if 'flux' in columns:
                pixels = len(columns['flux'][0]) if count else 0
                if pixels > resources['maximum_spectral_pixels']:
                    raise MemoryError('Observed spectral pixel budget exceeded')
                bytes_required += count * (pixels * pixels * 8 + pixels * 64)
            else:
                bytes_required += count * 1024
        elif isinstance(value, dict) and 'sfh_grid' in value:
            bytes_required += len(value['sfh_grid']) * resources['maximum_spectral_pixels'] * 8
    if bytes_required > positive(resources['memory_gib'], 'memory_gib') * 1024**3:
        raise MemoryError('Cosmography stage major-array forecast exceeds memory budget')
    return {'estimated_major_array_bytes': bytes_required, 'memory_gib_limit': resources['memory_gib'],
            'scope': 'preflight array bound; backend/runtime peak may be higher'}


def ingest_cosmography_manifest(request, destination):
    """Validate measured ladder/lens assets without fitting or changing them.

    Units, original calibrations, selection and covariance are explicit. Ladder
    design construction checks indicator/anchor links and identifiability;
    lens intake requires original imaging, delays and an authored sampler.
    Binary copies preserve every source byte, including original line endings.
    """
    from .observation_data import read_observations, covariance, observed_catalogue
    from . import source_path
    required = {'schema', 'method', 'observations', 'provenance', 'fit_model', 'unit_contract', 'memory_gib'}
    if set(request) != required or request['schema'] != 'measured-cosmography-intake-v1' or request['method'] not in {'distance_ladder', 'time_delay'}:
        raise ValueError('Complete measured ladder or full-lens intake request required')
    memory = request['memory_gib']
    if type(memory) not in (int, float) or not math.isfinite(memory) or memory <= 0:
        raise ValueError('Positive measured-data intake allocation required')
    paths = {}
    pins = {str(Path(__file__).resolve()): file_sha256(__file__)}
    for key in ['observations', 'provenance', 'fit_model']:
        reference = request[key]
        if set(reference) != {'path', 'sha256'}:
            raise ValueError('Exact measured input/model/provenance binding required')
        supplied = Path(reference['path'])
        if supplied.is_symlink() or not supplied.is_file():
            raise ValueError('Measured cosmography inputs must be regular source assets')
        path = supplied.resolve()
        if path.stat().st_size * 32 > memory * 1024 ** 3 / 4:
            raise MemoryError('Measured cosmography input exceeds allocation')
        if file_sha256(path) != reference['sha256']:
            raise ValueError('Measured cosmography source changed')
        paths[key] = path
        pins[str(path)] = reference['sha256']
    provenance = json.loads(paths['provenance'].read_text())
    if provenance.get('schema') != 'measured-cosmography-provenance-v1' or provenance.get('source_origin') != 'published_observations' or (not provenance.get('citation')) or (not provenance.get('selection_definition')) or (not provenance.get('release')):
        raise ValueError('Explicit published source release, citation and measured selection provenance required')
    for key in ['observations', 'fit_model']:
        if provenance['files'].get(paths[key].name, {}).get('sha256') != pins[str(paths[key])]:
            raise ValueError('Measured provenance does not bind original observations and inference model')
    observed = read_observations(paths['observations'])
    model = json.loads(paths['fit_model'].read_text())
    units = request['unit_contract']
    if not observed.get('selection') or not observed.get('namespace'):
        raise ValueError('Original observation identity namespace and explicit selection required')
    if request['method'] == 'distance_ladder':
        from .stellar_observations import _ladder_design
        expected = {
            'magnitudes': 'mag',
            'photometric_covariance': 'mag^2',
            'anchor_distance_moduli': 'mag',
            'anchor_covariance': 'mag^2',
            'redshift': 'dimensionless',
            'period': 'log10(day)',
            'metallicity': 'dex',
            'flow_velocity': 'km/s',
        }
        if units != expected:
            raise ValueError('Explicit original ladder physical units required; no conversion guessed')
        if observed.get('schema') != 'cosmography-observations-v1':
            raise ValueError('Original cosmography observed catalogue schema required')
        observed_catalogue(observed['columns'], units=observed['units'], selection=observed['selection'], namespace=observed['namespace'], calibration=observed['calibration'], provenance=observed.get('provenance', {}))
        for key in ['magnitude', 'sigma_mag', 'intrinsic_sigma_mag', 'extinction_mag', 'crowding_mag', 'extinction_sigma_mag', 'crowding_sigma_mag']:
            if observed['units'].get(key) != 'mag':
                raise ValueError('Original ladder photometric column units must be mag')
        columns = observed['columns']
        count = len(columns['object_id'])
        if not count or count * count * 128 > memory * 1024 ** 3 / 2:
            raise MemoryError('Complete original ladder covariance exceeds allocation')
        if not observed.get('calibration'):
            raise ValueError('Original anchors, photometric/instrument calibration required')
        design, y, cov, names, *_ = _ladder_design(observed, model)
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError('Measured ladder design is not identifiable under its declared calibration/prior model')
        covariance(cov, len(y))
        dimensions = {'observed_indicators': count, 'fitted_coordinates': len(names)}
    else:
        from .time_delay_cosmography import _ddt_h0_coefficient
        expected = {
            'time_delay': 'day',
            'time_delay_covariance': 'day^2',
            'astrometry': 'arcsec',
            'pixel_transform': 'arcsec/pixel',
            'image_flux': 'declared instrument flux',
            'noise': 'same instrument flux',
            'exposure': 's',
            'redshift': 'dimensionless',
        }
        if units != expected:
            raise ValueError('Explicit full-lens delay/imaging/noise/astrometry units required')
        if observed.get('schema') != 'lens-observations-v1' or not 0 < observed['z_lens'] < observed['z_source']:
            raise ValueError('Original lens/source geometry required')
        joint = observed['joint_data']
        delays = np.asarray(joint['time_delays_measured'], float)
        if delays.ndim != 1 or not len(delays) or (not np.isfinite(delays).all()):
            raise ValueError('Actual measured lens delays required')
        if 'time_delay_covariance_days2' in joint:
            covariance(joint['time_delay_covariance_days2'], len(delays))
        else:
            sigma = np.asarray(joint['time_delays_uncertainties'], float)
            if sigma.shape != delays.shape or not np.isfinite(sigma).all() or np.any(sigma <= 0):
                raise ValueError('Original positive lens-delay uncertainties required')
        if not joint.get('multi_band_list'):
            raise ValueError('Full lens imaging, PSF and measured noise inputs required')
        pixels = 0
        for data, psf, numerics in joint['multi_band_list']:
            image = np.asarray(data['image_data'], float)
            pixels += image.size
            if pixels * 512 > memory * 1024 ** 3 / 2:
                raise MemoryError('Original full lens images exceed allocation')
            if image.ndim != 2 or not image.size or (not np.isfinite(image).all()) or (not psf):
                raise ValueError('Original finite lens image and PSF required')
            transform = np.asarray(data['transform_pix2angle'], float)
            if transform.shape != (2, 2) or not np.isfinite(transform).all() or np.linalg.det(transform) == 0:
                raise ValueError('Original nonsingular angular pixel transform required')
            if 'noise_map' in data:
                noise = np.asarray(data['noise_map'], float)
                if noise.shape != image.shape or not np.isfinite(noise).all() or np.any(noise <= 0):
                    raise ValueError('Original positive image noise map required')
            else:
                positive(data.get('background_rms', 0), 'measured image background RMS')
                positive(data.get('exposure_time', 0), 'measured image exposure time')
        if not model['constraints'].get('Ddt_sampling') or not model['likelihood'].get('time_delay_likelihood'):
            raise ValueError('Full lens inference must sample distance with actual delay likelihood')
        if type(model['minimum_posterior_samples']) is not int or model['minimum_posterior_samples'] < 2:
            raise ValueError('Actual full-lens posterior sampling floor required')
        if not model.get('model') or not model.get('parameters') or type(model.get('sampling_seed')) is not int or (model['sampling_seed'] < 0):
            raise ValueError('Complete original lens model, native parameter bindings and sampling seed required')
        sequence = model['fitting_sequence']
        if not isinstance(sequence, list) or not any((isinstance(row, list) and len(row) == 2 and (row[0] in {'MCMC', 'emcee', 'zeus'}) for row in sequence)):
            raise ValueError('Original full lens model must configure actual posterior sampling')
        _ddt_h0_coefficient(observed['z_lens'], observed['z_source'], model['expansion_shape'])
        dimensions = {'measured_delays': len(delays), 'image_pixels': pixels}
    for name in ['observation_data', 'stellar_observations', 'time_delay_cosmography', 'data', 'foundation']:
        path = source_path(name).resolve()
        pins[str(path)] = file_sha256(path)
    for p, h in pins.items():
        if file_sha256(p) != h:
            raise ValueError('Measured cosmography input changed during intake')
    out = Path(destination).resolve()
    if out.exists():
        raise FileExistsError('Preserve previous measured cosmography intake')
    out.mkdir(parents=True)
    for key, name in [('observations', 'observations.json'), ('fit_model', 'fit-model.json')]:
        if atomic_copy(paths[key], out / name) != pins[str(paths[key])]:
            raise ValueError('Measured cosmography source changed during publication')
    if file_sha256(out / 'observations.json') != pins[str(paths['observations'])] or file_sha256(out / 'fit-model.json') != pins[str(paths['fit_model'])]:
        raise ValueError('Measured data/model publication changed original bytes')
    result = {
        'schema': 'measured-cosmography-intake-result-v1',
        'status': 'INPUTS_VALIDATED',
        'method': request['method'],
        'observations': {'path': str(out / 'observations.json'), 'sha256': file_sha256(out / 'observations.json')},
        'fit_model': {'path': str(out / 'fit-model.json'), 'sha256': file_sha256(out / 'fit-model.json')},
        'unit_contract': units,
        'dimensions': dimensions,
        'source_hashes': pins,
        'source_origin': 'published_observations',
        'provenance': provenance,
        'fit_executed': False,
        'science_acceptance': 'NOT_ACCEPTED',
        'scope': 'Original measured observation/model bytes validated for existing estimator intake. No measurements, nuisance calibration, covariance or posterior draws invented.',
    }
    atomic_json(out / 'manifest.json', result)
    return result
