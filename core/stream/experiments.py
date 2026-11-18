"""Physical stream experiments, selection and campaign stage dispatch.

Coordinates are proper kpc, velocities km/s and clocks Myr in the declared
frame. Candidate rescaling is an explicit diagnostic model; it does not turn
an altered trajectory into a measured native orbit. Inference uses recorded
priors, selection normalization and real sampler products.
"""
from __future__ import annotations

from pathlib import Path
import copy
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation
from core.stream_common import finite, positive, digest, json_safe
from core.gravity_fields import ComponentField, EvolvingPerturber, DifferentiatedPotential

from core.stream.sbi import SBI_KINDS
from core.matched import MATCHED_KINDS
from core.assembly import ASSEMBLY_KINDS
from core.observation_geometry import OBSERVATION_KINDS

EXPERIMENT_KINDS = SBI_KINDS | MATCHED_KINDS | ASSEMBLY_KINDS | OBSERVATION_KINDS | {'encounters', 'counterfactuals', 'perturber_templates', 'template_calibration',
    'template_model_comparison', 'compound_selection', 'compound_encounter', 'compound_features',
    'compound_identifiability', 'stream_inference', 'halo_shape_inference', 'posterior_predictive',
    'heldout_recovery', 'independent_chain_diagnostics', 'live_control_comparison', 'native_encounter_branch', 'initial_stream_state', 'worldline_history'}


class OrientedField:
    """Rotate a component's potential, acceleration and tidal Hessian together.

    Row-vector positions use (x-center)R; acceleration returns via R.T and
    the Hessian via R H R.T. This preserves one physical force convention.
    """
    def __init__(self, field, record):
        self.field = field
        self.record = record
        self.rotation = Rotation.from_euler(record['euler_sequence'], finite(record['euler_angles_rad'], (3,))).as_matrix()
        self.center = finite(record['center_kpc'], (3,))
    def evaluate(self, x, t):
        phi, acc, hessian = self.field.evaluate((x-self.center)@self.rotation, t)
        return phi, acc@self.rotation.T, np.einsum('ij,njk,lk->nil', self.rotation, hessian, self.rotation)


def parameterize_candidate_record(row):
    record = copy.deepcopy(row['record'])
    line = record['worldline']
    factor = positive(row.get('mass_scale', 1.), 'candidate mass scale')
    scale = positive(row.get('radius_scale', 1.), 'candidate radius scale')
    line['mass_msun'] = (np.asarray(line['mass_msun'])*factor).tolist()
    for profile in record['profiles']:
        profile['mass_msun'] *= factor
        if profile['kind'] == 'enclosed_mass':
            profile['enclosed_mass_msun'] = (np.asarray(profile['enclosed_mass_msun'])*factor).tolist()
            profile['radius_kpc'] = (np.asarray(profile['radius_kpc'])*scale).tolist()
        for key in ('scale_kpc', 'truncation_radius_kpc'):
            if key in profile:
                profile[key] *= scale
    if row.get('encounter_geometry'):
        from core.stream_common import KMS_TO_KPC_MYR
        geometry = row['encounter_geometry']
        times = finite(line['times_myr'])
        velocity = finite(geometry['velocity_kms'], (3,))
        center = finite(geometry['crossing_position_kpc'], (3,))
        impact = finite(geometry['impact_vector_kpc'], (3,))
        if abs(np.dot(impact, velocity)) > 1e-8*max(1., np.linalg.norm(impact)*np.linalg.norm(velocity)):
            raise ValueError('Encounter impact vector must be transverse to relative flyby velocity')
        line['position_kpc'] = (center+impact+(times-geometry['crossing_time_myr'])[:, None]*velocity*KMS_TO_KPC_MYR).tolist()
        line['velocity_kms'] = np.broadcast_to(velocity, (len(times), 3)).tolist()
        record['support_label'] = 'parameterized_flyby_diagnostic'
        record['provenance']['trajectory_scope'] = 'linear local flyby model; not the unchanged native trajectory'
    elif row.get('trajectory_offset_kpc'):
        line['position_kpc'] = (np.asarray(line['position_kpc'])+finite(row['trajectory_offset_kpc'], (3,))).tolist()
    if factor != 1 or scale != 1:
        record['support_label'] = 'homologously_rescaled_diagnostic'
        record['provenance']['profile_scope'] = 'explicit mass/radius scaling of recorded profile family'
    return record


def build_component_field(definition, inputs):
    components = {}
    frames = {}
    for name, row in definition['components'].items():
        kind = row['kind']
        if kind == 'host':
            field = inputs.host(row['host'])
            frames[name] = field.frame
        elif kind == 'analytic_spherical':
            from core.action_angle import AnalyticSphericalPotential
            field = AnalyticSphericalPotential(**row['parameters'])
        elif kind == 'halo_shape':
            from core.gravity_fields import FlattenedLogPotential
            field = FlattenedLogPotential(**row['parameters'])
        elif kind == 'coefficients':
            from core.stream.workflow import _potential
            field, _ = _potential(inputs, row)
        elif kind == 'evolving_perturber':
            field = EvolvingPerturber(parameterize_candidate_record(row))
        elif kind == 'baryonic':
            frame = None
            if row.get('frame'):
                from core.gravity_fields import HostFrame
                frame = HostFrame(**row['frame'])
            field = DifferentiatedPotential(row['record'], frame=frame)
        else: raise ValueError('Unknown component field kind: '+kind)
        if row.get('orientation'):
            field = OrientedField(field, row['orientation'])
        components[name] = field
    for row in definition['components'].values():
        provenance = row.get('record', {}).get('provenance', {})
        # External measured asset hashes must refer to real files, not labels.
        for reference, expected in provenance.get('source_hashes', {}).items():
            path = inputs.path(reference)
            if digest(path) != expected:
                raise ValueError('Measured component source changed')
    moving = definition.get('moving_frame_component')
    if moving is not None and moving not in frames:
        raise ValueError('Moving frame must refer to an actual host component')
    return ComponentField(components, accounting=definition['accounting']), frames.get(moving)


def _sample_from_definition(definition):
    sample = definition['initial_stream']
    from core.data import exact_uint64_ids
    ids = exact_uint64_ids(sample['particle_ids'], allow_empty=False)
    return {**sample, 'position_kpc': np.asarray(sample['position_kpc'], float),
        'velocity_kms': np.asarray(sample['velocity_kms'], float), 'mass_msun': np.asarray(sample['mass_msun'], float),
        'particle_ids': ids}


def make_physical_forward(definition, bindings, inputs):
    """Bind sampled parameters to the physical integrator and chosen observables.

    The same declared frame, particle identities, units and covariance follow
    every evaluation. Live self-gravity is requested explicitly; observed and
    intrinsic coordinates are separate, selected output conventions.
    """
    from core.stream.inference import DeterministicForwardMap
    from core.stream_forward_model import integrate_stream
    from core.stream.encounters import observed_phase_space
    def evaluate(record):
        field, frame = build_component_field(record['field'], inputs)
        sample = _sample_from_definition(record)
        if record.get('progenitor_transform'):
            transform=record['progenitor_transform']
            center=np.average(sample['position_kpc'],axis=0,weights=sample['mass_msun'])
            bulk=np.average(sample['velocity_kms'],axis=0,weights=sample['mass_msun'])
            sample['position_kpc']=(sample['position_kpc']-center)*positive(transform['size_scale'],'progenitor size')+finite(transform['center_kpc'],(3,))
            sample['velocity_kms']=(sample['velocity_kms']-bulk)*positive(transform['dispersion_scale'],'progenitor dispersion')+finite(transform['bulk_velocity_kms'],(3,))
            sample['mass_msun']*=positive(transform['mass_scale'],'progenitor mass')
        if record.get('live_progenitor'):
            from core.stream_resimulation import evolve_self_gravitating_progenitor
            _, arrays = evolve_self_gravitating_progenitor(sample, field, record['times_myr'], record['live_progenitor'])
        else:
            _, arrays = integrate_stream(sample, field, record['times_myr'], record['integration'], moving_frame=frame)
        if record['observable']['kind'] == 'observed_phase_space':
            value = observed_phase_space(arrays, record['observer'])
        elif record['observable']['kind'] == 'intrinsic_phase_space':
            value = np.column_stack((arrays['position_kpc'][-1], arrays['velocity_kms'][-1]))
        elif record['observable']['kind'] == 'track_summary':
            from core.stream_forward_model import TrackCoordinates, measure_track_density_velocity
            measured = measure_track_density_velocity(arrays, TrackCoordinates(record['track']), record['measurement'])
            value = np.column_stack((measured['linear_density_msun_kpc'][-1], measured['centroid_kpc'][-1], measured['velocity_mean_kms'][-1]))
        else: raise ValueError('Unknown forward observable')
        columns = record['observable']['columns']
        rows = record['observable']['rows']
        value = value[np.ix_(rows, columns)]
        if record.get('observable_zero_point') is not None:
            value += finite(record['observable_zero_point'], (len(columns),))
        if record.get('observable_scale') is not None:
            value *= finite(record['observable_scale'], (len(columns),))
        size = value.size
        covariance = finite(record['model_error_covariance'], (size, size))
        if 'discrepancy_amplitude' in record:
            amplitude = positive(record['discrepancy_amplitude'], 'discrepancy_amplitude')
            covariance = covariance*amplitude**2
        return {'values': value, 'model_error_covariance': covariance,
            'missingness_log_probability': record.get('missingness_log_probability', np.zeros(len(value)))}
    return DeterministicForwardMap(definition, bindings, evaluate)


def selection_adapter(record, n):
    """Use declared fixed selection or interpolate independent injection data.

    Fixed normalization needs a parameter-independence justification. Grid
    interpolation rejects extrapolation, and effective injection count keeps
    poorly supported regions explicit in the posterior model.
    """
    mode = record['mode']
    if mode == 'unselected':
        return lambda theta, prediction: {'mode': mode, 'normalization': 1., 'observed_detection_probability': np.ones(n)}
    if record['kind'] == 'fixed':
        if not record.get('parameter_independence_justification'):
            raise ValueError('Fixed selection normalization requires justification')
        def selected(theta, prediction):
            return {'mode': mode, 'normalization': record['normalization'],
                'observed_detection_probability': record['observed_detection_probability'],
                'intrinsic_expected_count': theta.get(record.get('count_parameter'), record.get('intrinsic_expected_count')),
                'supported': record['supported'], **({'detection_rule':record['detection_rule']} if 'detection_rule' in record else {})}
        return selected
    if record['kind'] == 'injection_grid':
        names = record['parameter_names']
        axes = [finite(q) for q in record['axes']]
        if len(names) != len(axes) or any(q.ndim != 1 or len(q) < 2 or np.any(np.diff(q) <= 0) for q in axes):
            raise ValueError('Selection grid requires increasing parameter axes')
        alpha = RegularGridInterpolator(axes, finite(record['normalization_grid']), bounds_error=True)
        effective = RegularGridInterpolator(axes, finite(record['effective_sample_grid']), bounds_error=True)
        def selected(theta, prediction):
            point = [theta[name] for name in names]
            return {'mode': mode, 'normalization': float(alpha([point])[0]),
                'observed_detection_probability': record['observed_detection_probability'],
                'intrinsic_expected_count': theta.get(record.get('count_parameter'), record.get('intrinsic_expected_count')),
                'supported': float(effective([point])[0]) >= record['minimum_effective_samples'],
                **({'detection_rule':record['detection_rule']} if 'detection_rule' in record else {})}
        return selected
    raise ValueError('Selection adapter requires fixed or independent injection grid normalization')


def physical_posterior(stage, inputs):
    from core.stream.inference import ParameterSpace, PosteriorModel
    refs = stage['inputs']
    p = stage['parameters']
    definition = inputs.json(refs['forward_definition'])
    if 'initial_sample' in refs:
        sample, parent, _, _ = inputs.sample(refs['initial_sample'], epoch_index=p.get('epoch_index', -1))
        definition['initial_stream'] = json_safe(sample)
    # Resource bounds are shared with campaign authority.
    definition['integration']['memory_gib'] = inputs.resources['memory_gib']
    definition['integration']['maximum_particles'] = inputs.resources['maximum_particles']
    observations = inputs.json(refs['observations'])
    dimensions=np.asarray(observations['values']).size
    if dimensions*dimensions*8*6>inputs.resources['memory_gib']*1024**3:
        raise MemoryError('Joint likelihood covariance exceeds declared workspace')
    space = ParameterSpace(p['priors'])
    build_component_field(definition['field'], inputs)  # resolve measured assets before checkpoint binding
    forward = make_physical_forward(definition, p['bindings'], inputs)
    if set(space.names) != {r['name'] for r in p['bindings']}:
        raise ValueError('Every physical/nuisance prior requires a declared forward binding')
    selected = selection_adapter(inputs.json(refs['selection']), len(observations['values']))
    return PosteriorModel(space, forward, observations, selected)


def execute_experiment_stage(stage, inputs, working):
    if stage['kind'] in OBSERVATION_KINDS:
        from core.observation_geometry import execute_observation_stage
        return execute_observation_stage(stage,inputs,working)
    if stage['kind'] in ASSEMBLY_KINDS:
        from core.assembly import execute_assembly_stage
        return execute_assembly_stage(stage,inputs,working)
    if stage['kind'] in MATCHED_KINDS:
        from core.matched import execute_matched_stage
        return execute_matched_stage(stage,inputs,working)
    if stage['kind'] in SBI_KINDS:
        from core.stream.sbi import execute_sbi_stage
        return execute_sbi_stage(stage,inputs,working)
    kind = stage['kind']
    p = stage.get('parameters', {})
    refs = stage.get('inputs', {})
    namespace = p.get('namespace', 'experiment:'+stage['stage_id'])
    epoch = None
    if kind == 'initial_stream_state':
        from core.stream_common import phase_space
        sample = _sample_from_definition({'initial_stream': inputs.json(refs['sample'])})
        phase_space(sample)
        namespace = sample['namespace']
        report = {'schema': 'stream-initial-state-v1', 'source_scope': p['source_scope'],
            'origin': p['origin'], 'native_population_claim': False}
        arrays = {key: sample[key] for key in ('position_kpc', 'velocity_kms', 'mass_msun', 'particle_ids')}
    elif kind == 'worldline_history':
        from core.stream.trajectories import worldlines_from_history
        parent, values = inputs.product(refs['history'])
        lines = worldlines_from_history(parent, values, p['selections'], frame=p['frame'], maximum_interval_myr=p['maximum_interval_myr'])
        report = {'schema': 'stream-worldline-history-v1', 'worldlines': lines, 'origin': p['frame'], 'source_scope': parent['schema']}
        arrays = {'worldline_count': np.array([len(lines)])}
    elif kind == 'encounters':
        from core.stream.trajectories import reconstruct_encounters
        perturbers=inputs.json(refs['perturbers'])
        streams=inputs.json(refs['stream_worldlines'])
        if isinstance(perturbers,dict):
            perturbers=perturbers['worldlines']
        if isinstance(streams,dict):
            streams=streams['worldlines']
        for line in [*perturbers,*streams]:
            certificate=line.get('interpolation_certificate',{})
            resolved_certificate_sources={}
            for reference,expected in certificate.get('source_hashes',{}).items():
                path=inputs.path(reference)
                if digest(path)!=expected:
                    raise ValueError('Trajectory interpolation certificate source changed')
                resolved_certificate_sources[str(path)]=expected
            if resolved_certificate_sources:
                certificate['source_hashes']=resolved_certificate_sources
        report, arrays = reconstruct_encounters(perturbers, streams, p['controls'])
    elif kind == 'counterfactuals':
        from core.stream_forward_model import run_counterfactuals
        sample, parent, _, _ = inputs.sample(refs['initial_sample'], epoch_index=p.get('epoch_index', -1))
        namespace = sample['namespace']
        field, frame = build_component_field(inputs.json(refs['field']), inputs)
        request = copy.deepcopy(p['request'])
        request['integration'].update(memory_gib=inputs.resources['memory_gib'], maximum_particles=inputs.resources['maximum_particles'])
        if request.get('live_progenitor'):
            request['live_progenitor'].update(memory_gib=inputs.resources['memory_gib'], maximum_particles=inputs.resources['maximum_particles'])
        if request.get('actions'):
            from core.stream.workflow import _potential
            request['actions']['field_object'], _ = _potential(inputs, request['actions']['potential'])
        report, arrays = run_counterfactuals(sample, field, p['times_myr'], request, moving_frame=frame)
    elif kind == 'perturber_templates':
        from core.stream.encounters import generate_template_ensemble
        sample, _, _, _ = inputs.sample(refs['initial_sample'], epoch_index=p.get('epoch_index', -1))
        namespace = sample['namespace']
        field, frame = build_component_field(inputs.json(refs['host_field']), inputs)
        request = copy.deepcopy(p['request'])
        request['integration'].update(memory_gib=inputs.resources['memory_gib'], maximum_particles=inputs.resources['maximum_particles'])
        report, arrays = generate_template_ensemble(sample, field, inputs.json(refs['designs']), request, moving_frame=frame)
    elif kind == 'template_calibration':
        from core.stream.encounters import calibrate_template_classifier
        data = inputs.json(refs['training_features'])
        report, arrays = calibrate_template_classifier(**data, **p['controls'])
    elif kind == 'template_model_comparison':
        from core.stream.encounters import posterior_model_comparison
        products = [inputs.product(q) for q in refs['posteriors']]
        if any(r['sampler'] != 'dynesty' for r, _ in products):
            raise ValueError('Evidence comparison requires nested evidence estimates')
        report = {'schema': 'stream-template-model-comparison-v1', 'models': p['model_names']}
        arrays = posterior_model_comparison([r['diagnostics']['log_evidence'] for r, _ in products],
            [r['diagnostics']['log_evidence_error'] for r, _ in products], p['model_prior'])
        report['scope'] = arrays.pop('scope')
    elif kind == 'compound_selection':
        from core.stream.encounters import find_bound_nested_substructure
        if 'native_graph_directory' in refs:
            from core.halos import verify_identity_graph
            from core.stream.encounters import nested_edges_from_native_graph
            path=inputs.path(refs['native_graph_directory'])
            native=verify_identity_graph(path)
            for asset in ('report.json','index.sqlite'):
                inputs.sources[str(path/asset)]=digest(path/asset)
            graph=nested_edges_from_native_graph(native)
        else: graph=inputs.json(refs['identity_graph'])
        report = find_bound_nested_substructure(graph, inputs.json(refs['catalogues']), **p['controls'])
        arrays = {'resolved_count': np.array([len(report['resolved'])]), 'rejected_count': np.array([len(report['rejected'])])}
    elif kind == 'compound_encounter':
        from core.stream.encounters import evolve_compound_encounter
        sample, _, _, _ = inputs.sample(refs['initial_sample'])
        namespace = sample['namespace']
        field, frame = build_component_field(inputs.json(refs['host_field']), inputs)
        request = copy.deepcopy(p['request'])
        request['forward']['integration'].update(memory_gib=inputs.resources['memory_gib'], maximum_particles=inputs.resources['maximum_particles'])
        report, arrays = evolve_compound_encounter(sample, field, inputs.json(refs['nested_system']), request, moving_frame=frame)
    elif kind == 'compound_features':
        from core.stream.encounters import measure_multiple_features
        _, values = inputs.product(refs['compound'])
        name = p['intervention']
        report = measure_multiple_features(values[name+'__density_ratio'][-1],
            values[name+'__delta_centroid_kpc'][-1], values['bin_edges_kpc'], **p['controls'])
        arrays = {'feature_supported': report.pop('supported_bins')}
        report['schema'] = 'compound-feature-catalogue-v1'
    elif kind == 'compound_identifiability':
        from core.stream.encounters import identifiability_matrix
        data = inputs.json(refs['sensitivity_design'])
        result = identifiability_matrix(**data)
        report = {'schema': 'compound-identifiability-v1', 'meaning': result.pop('meaning')}
        arrays = result
    elif kind in {'stream_inference', 'halo_shape_inference'}:
        from core.stream.inference import fit_sampler
        if kind == 'stream_inference':
            model = physical_posterior(stage, inputs)
        else:
            from core.stream.inference import ShapePopulation, HierarchicalPosterior, injection_population_selection
            from core.stream.inference import ParameterSpace
            population = ShapePopulation(inputs.json(refs['population_definition']))
            individual = inputs.json(refs['individual_posteriors'])
            for item in individual:
                if 'posterior_product' in item:
                    parent, values = inputs.product(item.pop('posterior_product'))
                    item['samples'] = values['posterior_samples']
                    item['weights'] = values['posterior_weight']
                    prior_column = item.pop('original_shape_log_prior_array', 'original_shape_log_prior')
                    if prior_column not in values:
                        raise ValueError('Hierarchical intake requires original shape marginal prior, not full nuisance prior')
                    item['original_shape_log_prior'] = values[prior_column]
            selection = injection_population_selection(population, inputs.json(refs['selection_injections']), minimum_effective_samples=p['minimum_selection_effective_samples'])
            model = HierarchicalPosterior(ParameterSpace(p['priors']), individual, population, selection, p['minimum_reweighting_effective_samples'])
        directory = Path(p['checkpoint_directory'])
        if not directory.is_absolute():
            directory = inputs.base/directory
        # Bind code, input assets, prior and observation configuration before
        # checkpoint reuse. Backends add their own exact revision binding.
        report, arrays = fit_sampler(model, p['sampler'], directory, binding={'source_hashes': inputs.sources, 'stage': stage, 'resources':inputs.resources})
        inputs.sources.update(report['backend']['source_hashes'])
        if kind == 'halo_shape_inference':
            report['schema'] = 'halo-shape-population-posterior-v1'
        elif p.get('shape_marginal_priors'):
            from core.stream.inference import ParameterSpace
            subset = ParameterSpace(p['shape_marginal_priors'])
            for prior in subset.records:
                original = next(row for row in model.space.records if row['name'] == prior['name'])
                if original != prior:
                    raise ValueError('Recorded shape marginal prior differs from original separable inference prior')
            lookup = [model.space.names.index(name) for name in subset.names]
            arrays['original_shape_log_prior'] = np.array([subset.log_prior(q[lookup]) for q in arrays['posterior_samples']])
            report['shape_marginal_prior'] = subset.records
    elif kind == 'posterior_predictive':
        from core.stream.inference import posterior_predictive
        model = physical_posterior(stage, inputs)
        _, posterior = inputs.product(refs['posterior'])
        report, arrays = posterior_predictive(model, posterior, **p['controls'])
        report['schema'] = 'stream-posterior-predictive-v1'
    elif kind == 'heldout_recovery':
        from core.stream.inference import evaluate_heldout_recovery
        posteriors = {key: inputs.product(ref)[1] for key, ref in refs['posteriors'].items()}
        report = evaluate_heldout_recovery(posteriors, inputs.json(refs['privileged_evaluation_truth']), **p['controls'])
        arrays = {'coverage': np.asarray(report['coverage'])}
    elif kind == 'independent_chain_diagnostics':
        from core.stream.inference import split_rank_diagnostic
        products = [inputs.product(ref) for ref in refs['independent_ensembles']]
        # One walker trajectory from each independently initialized ensemble;
        # do not incorrectly count all coupled walkers as independent chains.
        chains = [values['chain'][:, p['walker_row'], :] for _, values in products]
        seeds = [r['binding']['sampler']['seed'] for r, _ in products]
        if len(set(seeds)) != len(seeds):
            raise ValueError('Independent ensemble seeds must differ')
        length = min(map(len, chains))
        arrays = {'rank_split_rhat': split_rank_diagnostic(np.array([q[-length:] for q in chains]))}
        report = {'schema': 'stream-independent-chain-diagnostics-v1', 'independent_seeds': seeds, 'retained_steps': length}
    elif kind == 'native_encounter_branch':
        from core.evolution import execute_checkpoint_branch
        request = copy.deepcopy(p['request'])
        request['checkpoint_directory'] = inputs.path(refs['checkpoint_directory'])
        destination = Path(p['branch_directory'])
        if not destination.is_absolute():
            destination = inputs.base/destination
        report = execute_checkpoint_branch(inputs.path(refs['native_run']), destination, **request)
        report['schema'] = 'stream-native-encounter-branch-v1'
        arrays = {'event_count': np.array([len(request['event_ids'])])}
    elif kind == 'live_control_comparison':
        from core.stream_forward_model import attach_live_control
        lr, live = inputs.product(refs['live'])
        rr, reference = inputs.product(refs['reference'])
        arrays = attach_live_control(reference, live, identity_namespace=rr['namespace'], source_namespace=lr['namespace'])
        report = {'schema': 'stream-live-host-control-v1', 'scope': 'matched native live branches', 'host_response': 'native evolved'}
    else: raise ValueError('Unsupported block21–25 experiment kind')
    report.setdefault('science_acceptance', 'NOT_ACCEPTED')
    if kind in {'counterfactuals','compound_encounter','perturber_templates'}:
        report['origin']='inertial proper Cartesian'
    return report, arrays, namespace, epoch
