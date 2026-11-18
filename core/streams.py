"""Remaining historical stream forward-model section; ordinary migration pending."""

raise ImportError('Use core.stream_forward_model or commands.py')

# @research2-module stream_forward_model
"""Full phase-space forward dynamics and paired diagnostic interventions.

DOP853 integrates inertial trajectories in the actual time-dependent field.
Self-gravitating stellar branches reuse block 18's live reference engine.
Imposed fields do not provide self-consistent host response; branch requests
and independently produced native live controls remain explicit products.
"""

import hashlib
import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from core.stream_common import finite, increasing, positive, integer, phase_space, KMS_TO_KPC_MYR, seeded_rng
from core.gravity_fields import construct_component_control, fourier_torque_diagnostics


def freeze_common_initial_stream_state(sample):
    x, v, m, ids = phase_space(sample)
    result = {'position_kpc': x.copy(), 'velocity_kms': v.copy(), 'mass_msun': m.copy(),
              'particle_ids': ids.copy(), 'namespace': sample['namespace']}
    result['state_sha256'] = hashlib.sha256(b''.join(q.tobytes() for q in (x, v, m, ids))+sample['namespace'].encode()).hexdigest()
    for value in result.values():
        if isinstance(value, np.ndarray): value.flags.writeable = False
    return result


def integrate_stream(sample, field, output_times_myr, controls, *, moving_frame=None):
    x, v, m, ids = phase_space(sample); times = increasing(output_times_myr)
    n = len(x); maximum = integer(controls['maximum_particles'], 'maximum_particles')
    if not n or n > maximum: raise MemoryError('Stream particle count exceeds declared bounds')
    chunk = min(n, integer(controls['integration_chunk_rows'], 'integration_chunk_rows'))
    estimate = n*len(times)*320 + chunk*6*8*40
    if estimate > positive(controls['memory_gib'], 'memory_gib')*1024**3: raise MemoryError('Forward histories exceed workspace budget')
    max_step = positive(controls['maximum_step_myr'], 'maximum_step_myr')
    rtol = positive(controls['relative_tolerance'], 'relative_tolerance')
    absolute = np.tile(np.r_[np.full(3, positive(controls['position_tolerance_kpc'], 'position_tolerance_kpc')),
        np.full(3, positive(controls['velocity_tolerance_kms'], 'velocity_tolerance_kms'))], chunk)
    maximum_calls = integer(controls['maximum_force_calls'], 'maximum_force_calls')
    positions = np.empty((len(times), n, 3)); velocities = np.empty_like(positions)
    positions[0] = x; velocities[0] = v; calls = 0
    for begin in range(0, n, chunk):
        end = min(n, begin+chunk); count = end-begin
        state = np.column_stack((x[begin:end], v[begin:end])).ravel()
        def rhs(t, value):
            nonlocal calls
            calls += 1
            if calls > maximum_calls: raise RuntimeError('Forward dynamics exhausted maximum_force_calls')
            q = value.reshape(count, 6)
            acc = field.evaluate(q[:, :3], t)[1]
            return np.column_stack((q[:, 3:]*KMS_TO_KPC_MYR, acc*KMS_TO_KPC_MYR)).ravel()
        for k in range(1, len(times)):
            integration = solve_ivp(rhs, (times[k-1], times[k]), state, method='DOP853',
                rtol=rtol, atol=absolute[:count*6], max_step=max_step, t_eval=[times[k]])
            if not integration.success: raise ArithmeticError('Forward integrator failed: '+integration.message)
            state = integration.y[:, -1]
            positions[k, begin:end] = state.reshape(count, 6)[:, :3]
            velocities[k, begin:end] = state.reshape(count, 6)[:, 3:]
    potentials = np.empty((len(times), n)); acceleration = np.empty_like(positions)
    for k, t in enumerate(times):
        potentials[k], acceleration[k], _ = field.evaluate(positions[k], t)
    angular_momentum = np.cross(positions, velocities)
    torque = np.cross(positions, acceleration)
    arrays = {'times_myr': times, 'position_kpc': positions, 'velocity_kms': velocities,
              'mass_msun': m.copy(), 'particle_ids': ids.copy(), 'potential_kms2': potentials,
              'specific_energy_kms2': .5*np.sum(velocities**2, axis=2)+potentials,
              'angular_momentum_kpc_kms': angular_momentum, 'torque_kms2': torque}
    if moving_frame is not None:
        local = [moving_frame.to_local(positions[k], velocities[k], t) for k, t in enumerate(times)]
        arrays['host_position_kpc'] = np.array([q[0] for q in local]); arrays['host_velocity_kms'] = np.array([q[1] for q in local])
        # Angular momentum and torque relative to the translating center need
        # center acceleration; velocity-relative axes alone cannot provide it.
        centers = np.array([moving_frame.state(t)[0] for t in times])
        bulk = np.array([moving_frame.state(t)[1] for t in times])
        center_acc = np.array([moving_frame.motion(t, 2)/KMS_TO_KPC_MYR**2 for t in times])
        arrays['host_angular_momentum_kpc_kms'] = np.cross(positions-centers[:, None], velocities-bulk[:, None])
        arrays['host_torque_kms2'] = np.cross(positions-centers[:, None], acceleration-center_acc[:, None])
    return {'schema': 'stream-forward-dynamics-v1', 'integrator': 'DOP853 inertial proper units',
        'force_calls': calls, 'controls': controls, 'self_gravity': False,
        'convergence_status': 'NOT_EXECUTED', 'action_interpretation': 'angular momentum recorded; actions require a justified frozen potential'}, arrays


class TrackCoordinates:
    """A frozen piecewise-linear reference track, shared across interventions.

    Closest-segment projection uses bounded batches and an exhaustive segment
    search. A changed stream cannot redefine coordinates to erase its spur.
    """
    def __init__(self, record):
        self.vertices = finite(record['position_kpc'])
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or len(self.vertices) < 2: raise ValueError('Reference track needs at least two 3D vertices')
        self.delta = np.diff(self.vertices, axis=0); self.length = np.linalg.norm(self.delta, axis=1)
        if np.any(self.length <= 0): raise ValueError('Distinct ordered track vertices required')
        self.arc = np.r_[0., np.cumsum(self.length)]
        self.tangent = self.delta/self.length[:, None]
        from core.stream_tracks import transport_frame
        self.tangent, self.normal, self.binormal, self.reversals = transport_frame(self.tangent)
        self.origin = record.get('origin', 'inertial')

    def project(self, positions, velocities, *, chunk_rows):
        x = finite(positions); v = finite(velocities, x.shape); n = len(x)
        s = np.empty(n); transverse = np.empty((n, 2)); velocity = np.empty((n, 3)); segments = np.empty(n, int)
        for start in range(0, n, chunk_rows):
            stop = min(n, start+chunk_rows)
            d = x[start:stop, None]-self.vertices[:-1]
            fraction = np.clip(np.einsum('nsi,si->ns', d, self.delta)/self.length**2, 0., 1.)
            residual = d-fraction[:, :, None]*self.delta
            index = np.argmin(np.sum(residual**2, axis=2), axis=1); rows = np.arange(stop-start)
            offset = residual[rows, index]; f = fraction[rows, index]
            basis = np.stack((self.tangent[index], self.normal[index], self.binormal[index]), axis=1)
            s[start:stop] = self.arc[index]+f*self.length[index]
            transverse[start:stop] = np.einsum('ni,nji->nj', offset, basis[:, 1:])
            velocity[start:stop] = np.einsum('ni,nji->nj', v[start:stop], basis)
            segments[start:stop] = index
        return {'s_kpc': s, 'transverse_kpc': transverse, 'local_velocity_kms': velocity, 'track_segment': segments}


def measure_track_density_velocity(arrays, track, controls, *, particle_weights=None):
    edges = increasing(controls['bin_edges_kpc']); bins = len(edges)-1
    epochs, particles = arrays['position_kpc'].shape[:2]
    weights = arrays['mass_msun'].copy() if particle_weights is None else finite(particle_weights, (particles,))
    if np.any(weights < 0): raise ValueError('Measurement weights must be nonnegative')
    chunk = integer(controls['projection_chunk_rows'], 'projection_chunk_rows')
    maximum = integer(controls['minimum_bin_particles'], 'minimum_bin_particles', 2)
    density = np.zeros((epochs, bins)); counts = np.zeros((epochs, bins), int)
    centroid = np.full((epochs, bins, 2), np.nan); width = np.full((epochs, bins, 2, 2), np.nan)
    velocity = np.full((epochs, bins, 3), np.nan); support = np.zeros((epochs, bins), bool)
    coordinates = []; compression = np.full((epochs, particles), np.nan)
    for k in range(epochs):
        projected = track.project(arrays['position_kpc'][k], arrays['velocity_kms'][k], chunk_rows=chunk)
        coordinates.append(projected['s_kpc'])
        labels = np.searchsorted(edges, projected['s_kpc'], side='right')-1
        labels[projected['s_kpc'] == edges[-1]] = bins-1
        for b in range(bins):
            use = (labels == b) & (weights > 0); count = int(use.sum()); counts[k, b] = count
            density[k, b] = weights[use].sum()/np.diff(edges)[b]
            if count < maximum: continue
            w = weights[use]/weights[use].sum(); y = projected['transverse_kpc'][use]
            center = np.einsum('n,ni->i', w, y); d = y-center
            centroid[k, b] = center; width[k, b] = np.einsum('n,ni,nj->ij', w, d, d)
            velocity[k, b] = np.einsum('n,ni->i', w, projected['local_velocity_kms'][use]); support[k, b] = True
        # This Jacobian is a Lagrangian ordering diagnostic, not an assumption
        # that a folded multistream caustic retains a unique arclength inverse.
        if k:
            order = np.argsort(coordinates[0]); reference = coordinates[0][order]
            unique = np.r_[True, np.diff(reference) > controls['label_resolution_kpc']]
            chosen = order[unique]
            if len(chosen) >= 3:
                compression[k, chosen] = np.gradient(projected['s_kpc'][chosen], reference[unique])
    return {'bin_edges_kpc': edges, 'bin_center_kpc': .5*(edges[1:]+edges[:-1]),
            'linear_density_msun_kpc': density, 'bin_count': counts, 'centroid_kpc': centroid,
            'width_covariance_kpc2': width, 'velocity_mean_kms': velocity, 'measurement_supported': support,
            's_kpc': np.array(coordinates), 'lagrangian_jacobian': compression}


def compare_control_effects(reference, counterfactual, *, model_error_covariance, bootstrap_replicates, seed):
    if not np.array_equal(reference['particle_ids'], counterfactual['particle_ids']): raise ValueError('Paired controls must preserve exact particle ordering')
    if not np.array_equal(reference['times_myr'], counterfactual['times_myr']): raise ValueError('Paired controls require identical output times')
    dx = counterfactual['position_kpc']-reference['position_kpc']; dv = counterfactual['velocity_kms']-reference['velocity_kms']
    final = np.column_stack((dx[-1], dv[-1])); n = len(final)
    error = finite(model_error_covariance, (6, 6))
    if not np.allclose(error, error.T) or np.linalg.eigvalsh(error).min() < -1e-12: raise ValueError('Model discrepancy covariance must be positive semidefinite')
    rng = seeded_rng(seed, 'paired-controls'); draws = integer(bootstrap_replicates, 'bootstrap_replicates', 2)
    means = np.array([final[rng.integers(0, n, n)].mean(axis=0) for _ in range(draws)])
    covariance = np.cov(means, rowvar=False)+error
    report = {'paired_final_mean': final.mean(axis=0), 'paired_mean_covariance': covariance,
        'covariance_coordinate_order': ['dx', 'dy', 'dz', 'dvx', 'dvy', 'dvz'],
        'model_error_covariance': error, 'bootstrap_replicates': draws,
        'interpretation': 'paired imposed-field response; not proof of unique dark attribution'}
    arrays = {'delta_position_kpc': dx, 'delta_velocity_kms': dv, 'paired_mean_covariance': covariance}
    for name in ('linear_density_msun_kpc', 'centroid_kpc', 'velocity_mean_kms', 'width_covariance_kpc2', 'angular_momentum_kpc_kms'):
        if name in reference and name in counterfactual: arrays['delta_'+name] = counterfactual[name]-reference[name]
    if 'linear_density_msun_kpc' in reference:
        base = reference['linear_density_msun_kpc']; changed = counterfactual['linear_density_msun_kpc']
        arrays['density_ratio'] = np.divide(changed, base, out=np.full_like(base, np.nan), where=base > 0)
        arrays['gap_supported'] = reference['measurement_supported'] & counterfactual['measurement_supported'] & (base > 0)
    return report, arrays


def run_counterfactuals(sample, full_field, times, request, *, moving_frame=None):
    frozen = freeze_common_initial_stream_state(sample)
    controls = request['integration']; results = {}; reports = {}; branches = []
    interventions = request['interventions']
    if len(interventions) > integer(request['maximum_interventions'], 'maximum_interventions'): raise ValueError('Intervention bound exceeded')
    retained=len(interventions)*len(times)*len(frozen['particle_ids'])*320
    if retained>controls['memory_gib']*1024**3: raise MemoryError('Paired intervention histories exceed declared memory')
    track = TrackCoordinates(request['track'])
    if len(track.delta)*controls['integration_chunk_rows']*64 > controls['memory_gib']*1024**3:
        raise MemoryError('Projection workspace exceeds memory budget')
    for intervention in interventions:
        name = intervention['name']
        if not name.replace('-', '_').isidentifier() or name in results: raise ValueError('Unique safe intervention names required')
        field = construct_component_control(full_field, intervention)
        if request.get('live_progenitor'):
            from core.stream_resimulation import evolve_self_gravitating_progenitor
            report, arrays = evolve_self_gravitating_progenitor(frozen, field, times, request['live_progenitor'])
        else:
            report, arrays = integrate_stream(frozen, field, times, controls, moving_frame=moving_frame)
        if moving_frame is not None and 'host_position_kpc' not in arrays:
            local=[moving_frame.to_local(arrays['position_kpc'][k],arrays['velocity_kms'][k],t) for k,t in enumerate(times)]
            arrays['host_position_kpc']=np.array([row[0] for row in local]);arrays['host_velocity_kms']=np.array([row[1] for row in local])
        arrays.update(measure_track_density_velocity(arrays, track, request['measurement']))
        if request.get('actions'):
            from core.action_angle import measure_actions
            action_field = request['actions']['field_object']
            action_reports = []; action_arrays = []
            for k, t in enumerate(times):
                state = {**frozen, 'position_kpc': arrays['position_kpc'][k], 'velocity_kms': arrays['velocity_kms'][k]}
                ar, aa = measure_actions(state, action_field, request['actions']['domain'], request['actions']['controls'])
                action_reports.append(ar); action_arrays.append(aa['actions_kpc_kms'])
            arrays['actions_kpc_kms'] = np.array(action_arrays); report['actions'] = action_reports
        if request.get('fourier_radial_edges_kpc'):
            rows = []
            for k, t in enumerate(times):
                x = arrays.get('host_position_kpc', arrays['position_kpc'])[k]
                acc = field.evaluate(arrays['position_kpc'][k], t)[1]
                if moving_frame is not None:
                    center_acc = moving_frame.motion(t, 2)/KMS_TO_KPC_MYR**2
                    acc = (acc-center_acc)@moving_frame.state(t)[2]
                rows.append(fourier_torque_diagnostics(x, acc, frozen['mass_msun'], request['fourier_radial_edges_kpc']))
            arrays['fourier_m1_m4'] = np.array([r['fourier_m1_m4'] for r in rows])
            arrays['radial_torque_z_msun_kms2'] = np.array([r['torque_z_msun_kms2'] for r in rows])
            arrays['fourier_supported']=np.array([r['fourier_supported'] for r in rows])
            report['torque_definition']='Sum m*(x cross acceleration)_z in the declared chart; moving chart subtracts origin acceleration and rotates axes, without rotating-frame fictitious terms'
        patterns={}
        for component_name,component in field.components.items():
            if hasattr(component,'pattern_speed_at'):
                patterns[component_name]={'phase_rad':component.phase,'times_myr':component.times,
                    'pattern_speed_kms_kpc':[component.pattern_speed_at(t) for t in times]}
                arrays[component_name+'__pattern_speed_kms_kpc']=np.array(patterns[component_name]['pattern_speed_kms_kpc'])
        report['pattern_histories']=patterns
        report.update({'intervention': intervention, 'initial_state_sha256': frozen['state_sha256'],
            'experiment_scope': 'imposed host field, live stellar progenitor' if request.get('live_progenitor') else 'imposed field, collisionless tracers',
            'component_accounting': field.accounting, 'host_response': 'prescribed measured evolution'})
        if intervention.get('requires_live_host_response'):
            branches.append({'name': name, 'status': 'NATIVE_BRANCH_REQUIRED', 'initial_state_sha256': frozen['state_sha256'],
                'intervention': intervention, 'reason': 'wake/deformation/backreaction must be evolved by the native live engine'})
        reports[name] = report; results[name] = arrays
    reference = request['reference_intervention']
    if reference not in results: raise ValueError('Reference intervention missing')
    packed = {}; comparisons = {}
    for name, values in results.items():
        packed.update({name+'__'+key: value for key, value in values.items()})
        if name != reference:
            comparison, delta = compare_control_effects(results[reference], values,
                model_error_covariance=request['model_error_covariance'], bootstrap_replicates=request['bootstrap_replicates'], seed=request['seed'])
            comparisons[name] = comparison; packed.update({name+'__'+key: value for key, value in delta.items()})
    # Baseline fields are also exposed for existing stream figure/observer consumers.
    packed.update(results[reference])
    return {'schema': 'stream-counterfactuals-v1', 'interventions': reports, 'comparisons': comparisons,
        'live_branch_requests': branches, 'initial_state_sha256': frozen['state_sha256'], 'reference_intervention': reference,
        'alternative_causes': request['alternative_causes'], 'science_acceptance': 'NOT_ACCEPTED'}, packed


def validate_native_counterfactual_ic_request(parameters):
    request=parameters['request']
    keys={'schema','original_ic','operation','removed_dm_particle_ids','stream_particle_ids','stream_species','particle_namespace',
        'maximum_particles','memory_gib','disk_gib','chunk_rows'}
    if not isinstance(request,dict):raise ValueError('Explicit source-bound native IC intervention request required')
    late=request.get('schema')=='native-counterfactual-ic-request-v3'
    control=late and request.get('operation')=='retain_collisionless_state'
    scaling=request.get('schema')=='native-counterfactual-ic-request-v2' or late and request.get('operation')=='scale_dm_particle_masses'
    if scaling:keys=(keys-{'removed_dm_particle_ids'})|{'affected_dm_particle_ids','mass_multiplier'}
    if control:keys.discard('removed_dm_particle_ids')
    if late:keys.add('source_snapshot')
    expected_operation='retain_collisionless_state' if control else 'scale_dm_particle_masses' if scaling else 'remove_dm_particles'
    if set(request)!=keys or request['schema'] not in {'native-counterfactual-ic-request-v1','native-counterfactual-ic-request-v2','native-counterfactual-ic-request-v3'} or request['operation']!=expected_operation:
        raise ValueError('Declare exact removal, DM mass scaling, or the late unchanged-state control')
    if scaling:
        multiplier=positive(request['mass_multiplier'],'DM mass multiplier')
        if multiplier==1:raise ValueError('A mass counterfactual must change its declared target masses')
    if late and (type(request['source_snapshot'])is not int or request['source_snapshot']<0):raise ValueError('Exact saved late-time source snapshot label required')
    for name in ('maximum_particles','chunk_rows'):
        if type(request[name])is not int or request[name]<1:raise ValueError('Exact positive intervention IC bounds required')
    for name in ('memory_gib','disk_gib'):positive(request[name],name)
    if type(request['stream_species'])is not int or not 0<=request['stream_species']<=4 or not isinstance(request['particle_namespace'],str) or not request['particle_namespace']:
        raise ValueError('Explicit unchanged stream species and namespace required')
    if late and request['stream_species']==0:raise ValueError('Late collisionless intervention cannot carry a gas stream population')
    from .data import exact_uint64_ids
    removed=exact_uint64_ids([] if control else request['affected_dm_particle_ids'] if scaling else request['removed_dm_particle_ids']);stream=exact_uint64_ids(request['stream_particle_ids'])
    if (not control and not len(removed)) or not len(stream) or len(np.unique(removed))!=len(removed) or len(np.unique(stream))!=len(stream) or np.intersect1d(removed,stream).size:
        raise ValueError('Distinct nonempty removed DM and unchanged stream identity sets required')
    if len(removed)+len(stream)>request['maximum_particles'] or request['maximum_particles']*512+request['chunk_rows']*1024>request['memory_gib']*1024**3:
        raise MemoryError('Intervention ID selection and row-copy workspace exceed allocation')
    return request


def native_counterfactual_ic_input_bindings(parameters):
    from pathlib import Path
    from .sidm_analysis import _stage_native_bindings
    from .foundation import file_sha256
    request=validate_native_counterfactual_ic_request(parameters);sources=_stage_native_bindings(parameters)
    reference=request['original_ic']
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('Pin the complete original single-file native IC')
    requested=Path(reference['path']).expanduser();path=requested.resolve()
    if requested.is_symlink() or sources.get(str(path))!=reference['sha256'] or file_sha256(path)!=reference['sha256']:
        raise ValueError('Counterfactual IC is outside its actual original native source generation')
    if request['schema']=='native-counterfactual-ic-request-v3':
        from .halos import _snapshot_files
        files=_snapshot_files(Path(parameters['native_run_directory']).resolve(),request['source_snapshot'])
        if len(files)!=1 or files[0].resolve()!=path:raise ValueError('Late-time intervention requires its complete original single-file saved native snapshot')
    return sources


def publish_native_counterfactual_ic(parameters,destination):
    """Copy original ICs and remove exact DM rows, leaving stream state intact.

    This changes the full-system matter content and is not a continuation of
    the same restart state or an inherited cosmological IC acceptance proof.
    """
    import h5py,json,shutil,yaml
    from pathlib import Path
    from .data import exact_uint64_ids,snapshot_total_counts
    from .foundation import atomic_json,atomic_copy,atomic_text,file_sha256
    request=validate_native_counterfactual_ic_request(parameters);sources=native_counterfactual_ic_input_bindings(parameters)
    root=Path(parameters['native_run_directory']).resolve();config_path=root/'config.yaml'
    if sources.get(str(config_path))!=file_sha256(config_path):raise ValueError('IC intervention lacks original native physics configuration')
    config=yaml.safe_load(config_path.read_text())
    if config.get('sidm',{}).get('enabled',False):raise ValueError('SIDM interventions need regenerated fine-population/coarse-support contracts; collisionless removal cannot inherit them')
    late=request['schema']=='native-counterfactual-ic-request-v3'
    control=late and request['operation']=='retain_collisionless_state'
    late_epoch=None;phase_copy_bytes=0
    if late:
        configuration=root/'input/Config.sh'
        if sources.get(str(configuration))!=file_sha256(configuration):raise ValueError('Late-time state lacks its actual native compile profile')
        flags={line.split('#',1)[0].strip().split('=',1)[0] for line in configuration.read_text().splitlines()}
        forbidden={'NGENIC','CREATE_GRID','GENERATE_GAS_IN_ICS','SPLIT_PARTICLE_TYPE','SECOND_ORDER_LPT_ICS',
            'COOLING','STARFORMATION','STELLAR_EVOLUTION','BLACK_HOLES','RESEARCH_GRACKLE','RESEARCH_SIDM_REFERENCE','SIDM',
            'OUTPUT_NON_SYNCHRONIZED_ALLOWED','ADAPTIVE_HYDRO_SOFTENING','INDIVIDUAL_GRAVITY_SOFTENING'}
        if flags&forbidden or not {'SELFGRAVITY','OUTPUT_EXACT_SYNCHRONIZED'}<=flags:
            raise ValueError('Late-time fresh intervention needs source-free collisionless synchronized evolution without IC regeneration or uncarried softening/source state')
    if request['operation']=='scale_dm_particle_masses':
        configuration=root/'input/Config.sh'
        if sources.get(str(configuration))!=file_sha256(configuration):raise ValueError('Mass intervention lacks its actual native compile profile')
        flags={line.split('#',1)[0].strip().split('=',1)[0] for line in configuration.read_text().splitlines()}
        if 'LEAN' in flags:raise ValueError('DM mass changes require a native per-particle mass profile; LEAN constant-mass state cannot represent them')
    original=Path(request['original_ic']['path']).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Keep previous native counterfactual IC generations')
    if original.stat().st_size+1024**2>request['disk_gib']*1024**3:raise MemoryError('Immutable intervention copy exceeds disk allocation')
    scaling=request['operation']=='scale_dm_particle_masses'
    removed=np.sort(exact_uint64_ids([] if control else request['affected_dm_particle_ids'] if scaling else request['removed_dm_particle_ids']));stream=np.sort(exact_uint64_ids(request['stream_particle_ids']))
    def selected_state(native):
        group=native['PartType'+str(request['stream_species'])];ids=exact_uint64_ids(group['ParticleIDs'][:]);order=np.argsort(ids);at=np.searchsorted(ids[order],stream)
        if np.any(at>=len(order)) or not np.array_equal(ids[order[at]],stream):raise ValueError('Original/changed IC does not retain every selected stream identity')
        rows=order[at];fields={}
        for name in ('Coordinates','Velocities','Masses'):
            if name in group:
                expected=(len(ids),) if name=='Masses' else (len(ids),3)
                if group[name].shape!=expected:raise ValueError('Original native selected stream field dimensions differ: '+name)
                fields[name]=np.asarray(group[name][:])[rows]
            elif name=='Masses':fields[name]=np.full(len(rows),native['Header'].attrs['MassTable'][request['stream_species']])
            else:raise ValueError('Selected native stream has incomplete original phase space')
        return fields
    with h5py.File(original,'r') as native:
        counts=snapshot_total_counts(native['Header'].attrs)
        if int(counts.sum())>request['maximum_particles'] or int(native['Header'].attrs.get('NumFilesPerSnapshot',1))!=1 or not np.array_equal(np.asarray(native['Header'].attrs['NumPart_ThisFile']),counts):
            raise ValueError('Intervention requires the complete bounded original single-file native IC census')
        if late:
            if int(counts[0]) or int(counts[5]) or 'Config' not in native or 'Parameters' not in native:
                raise ValueError('Late collisionless state cannot discard gas/BH/source physics or missing native configuration')
            actual_flags=set(native['Config'].attrs)
            if actual_flags&forbidden or not {'SELFGRAVITY','OUTPUT_EXACT_SYNCHRONIZED'}<=actual_flags:
                raise ValueError('Saved late-time native profile contradicts its collisionless synchronized state contract')
            from .evolution import parameter_values
            parameters_native=parameter_values(root/'input/param.txt');saved=native['Parameters'].attrs
            for name in ('ComovingIntegrationOn','HubbleParam','UnitLength_in_cm','UnitMass_in_g','UnitVelocity_in_cm_per_s','BoxSize'):
                if float(saved[name])!=float(parameters_native[name]):raise ValueError('Saved late state changes original native physical units/domain: '+name)
            late_epoch=float(native['Header'].attrs['Time'])
            if not math.isfinite(late_epoch) or not float(parameters_native['TimeBegin'])<late_epoch<float(parameters_native['TimeMax']):
                raise ValueError('Late-time intervention must lie strictly inside the actual parent evolution interval')
            all_ids=[]
            for species in (1,2,3,4):
                n=int(counts[species])
                if not n:continue
                group=native['PartType'+str(species)]
                for name in ('ParticleIDs','Coordinates','Velocities'):
                    expected=(n,) if name=='ParticleIDs' else (n,3)
                    if name not in group or group[name].shape!=expected:raise ValueError('Late native state has an incomplete required phase field')
                if 'Masses' in group and group['Masses'].shape!=(n,):raise ValueError('Late native mass column is incomplete')
                for name in ('ParticleIDs','Coordinates','Velocities','Masses'):
                    if name in group:phase_copy_bytes+=group[name].size*group[name].dtype.itemsize
                for begin in range(0,n,request['chunk_rows']):
                    for name in ('Coordinates','Velocities'):
                        values=np.asarray(group[name][begin:begin+request['chunk_rows']])
                        if values.dtype.kind!='f' or not np.isfinite(values).all():raise ValueError('Late native phase-space storage is not finite floating point')
                    values=np.asarray(group['Masses'][begin:begin+request['chunk_rows']],float) if 'Masses' in group else np.asarray([native['Header'].attrs['MassTable'][species]],float)
                    if not np.isfinite(values).all() or np.any(values<=0):raise ValueError('Late collisionless population has nonpositive/nonfinite masses')
                all_ids.append(exact_uint64_ids(group['ParticleIDs'][:],allow_empty=False))
            if len(np.unique(np.concatenate(all_ids)))!=int(counts.sum()):raise ValueError('Late collisionless state aliases native identities across species')
        initial_stream=selected_state(native);ids=exact_uint64_ids(native['PartType1/ParticleIDs'][:]);at=np.searchsorted(removed,ids)
        selected=(at<len(removed));selected[selected]&=removed[at[selected]]==ids[selected]
        if int(selected.sum())!=len(removed) or len(np.unique(ids))!=len(ids):raise ValueError('Removed identities are absent or aliased in the original DM IC')
        original_count=len(ids);retained=np.arange(original_count) if scaling else np.flatnonzero(~selected)
        if not len(retained):raise ValueError('A live-host DM counterfactual must retain a nonempty native DM population')
        specs=[]
        for name,dataset in native['PartType1'].items():
            if not isinstance(dataset,h5py.Dataset) or not dataset.shape or dataset.shape[0]!=original_count or dataset.dtype.hasobject:
                raise ValueError('DM intervention cannot reinterpret opaque or nonparticle native group state: '+name)
            row_bytes=dataset.dtype.itemsize*math.prod(dataset.shape[1:])
            if row_bytes*request['chunk_rows']*4>request['memory_gib']*1024**3/4:raise MemoryError('Native particle row shape exceeds intervention workspace')
            specs.append(name)
        mass=np.asarray(native['PartType1/Masses'][:],float) if 'Masses' in native['PartType1'] else np.full(len(ids),native['Header'].attrs['MassTable'][1])
        velocity=np.asarray(native['PartType1/Velocities'][:],float)
        if mass.shape!=(len(ids),) or velocity.shape!=(len(ids),3) or not np.isfinite(mass).all() or not np.isfinite(velocity).all() or np.any(mass<=0):
            raise ValueError('Original DM extensive phase-space state is invalid')
        row_copy_bytes=sum(len(retained)*(8 if scaling and name=='Masses' else native['PartType1/'+name].dtype.itemsize)*math.prod(native['PartType1/'+name].shape[1:]) for name in specs)
        if scaling and 'Masses' not in specs:row_copy_bytes+=len(ids)*8
        removed_mass=float(mass[selected].sum());removed_momentum=np.sum(mass[selected,None]*velocity[selected],axis=0)
        changed_mass=mass.copy()
        if scaling:
            changed_mass[selected]*=request['mass_multiplier']
            if not np.isfinite(changed_mass).all() or np.any(changed_mass<=0):raise ValueError('Scaled native DM masses leave finite positive support')
            if not math.isfinite(float(changed_mass.sum())) or not math.isfinite(removed_mass*(request['mass_multiplier']-1)) or not np.isfinite(removed_momentum*(request['mass_multiplier']-1)).all():
                raise ValueError('Scaled native extensive mass/momentum leaves finite support')
    if not math.isfinite(removed_mass) or (removed_mass<0 if control else removed_mass<=0) or not np.isfinite(removed_momentum).all():raise ValueError('Affected native extensive state is invalid')
    footprint=(phase_copy_bytes if late else original.stat().st_size)+row_copy_bytes+4*1024**2
    ancestor=destination.parent
    while not ancestor.exists():ancestor=ancestor.parent
    if footprint>request['disk_gib']*1024**3 or footprint>shutil.disk_usage(ancestor).free:
        raise MemoryError('Uncompressed immutable intervention output exceeds allocated/available disk')
    destination.mkdir(parents=True);output=destination/'ics_gadget.hdf5'
    if not late:atomic_copy(original,output)
    else:
        # Write fresh phase-space ICs, without snapshot capsules, stale
        # publication metadata, or source/integrator state masquerading as
        # a continuation. Native stored coordinate/velocity values are kept.
        with h5py.File(original,'r') as source,h5py.File(output,'w') as target:
            header=target.create_group('Header')
            names={'Time','Redshift','BoxSize','Omega0','OmegaLambda','HubbleParam','NumPart_ThisFile','NumPart_Total',
                'NumPart_Total_HighWord','MassTable','NumFilesPerSnapshot','Flag_DoublePrecision','Flag_Entropy_ICs'}
            for name in names:
                if name in source['Header'].attrs:header.attrs[name]=source['Header'].attrs[name]
            header.attrs['Flag_IC_Info']=0
            for species in (1,2,3,4):
                if not int(counts[species]):continue
                group=target.create_group('PartType'+str(species));parent_group=source[group.name]
                for name in ('ParticleIDs','Coordinates','Velocities','Masses'):
                    if name not in parent_group:continue
                    old=parent_group[name];new=group.create_dataset(name,shape=old.shape,dtype=old.dtype,chunks=True,fletcher32=True)
                    for begin in range(0,len(old),request['chunk_rows']):
                        new[begin:begin+request['chunk_rows']]=old[begin:begin+request['chunk_rows']]
                        if not np.array_equal(new[begin:begin+request['chunk_rows']],old[begin:begin+request['chunk_rows']]):
                            raise RuntimeError('Fresh late-state materialization changes stored collisionless phase/mass/identity values')
        specs=[name for name in specs if name in {'ParticleIDs','Coordinates','Velocities','Masses'}]
    with h5py.File(original,'r') as source,h5py.File(output,'r+') as native:
        group=native['PartType1']
        for name in specs:
            old=source['PartType1/'+name];attrs={} if late else dict(old.attrs);del group[name]
            new=group.create_dataset(name,shape=(len(retained),*old.shape[1:]),dtype='f8' if scaling and name=='Masses' else old.dtype,chunks=True,fletcher32=True)
            for begin in range(0,len(retained),request['chunk_rows']):
                rows=retained[begin:begin+request['chunk_rows']]
                new[begin:begin+len(rows)]=changed_mass[rows] if scaling and name=='Masses' else old[rows]
            for key,value in attrs.items():new.attrs[key]=value
        if scaling and 'Masses' not in specs:
            dataset=group.create_dataset('Masses',shape=changed_mass.shape,dtype='f8',chunks=True,fletcher32=True)
            for begin in range(0,len(changed_mass),request['chunk_rows']):dataset[begin:begin+request['chunk_rows']]=changed_mass[begin:begin+request['chunk_rows']]
        new_counts=np.asarray(counts,dtype=np.uint64).copy();new_counts[1]=len(retained)
        header=native['Header'].attrs;header['NumPart_ThisFile']=new_counts.astype(np.uint64);header['NumPart_Total']=(new_counts&np.uint64(0xffffffff)).astype(np.uint32)
        header['NumPart_Total_HighWord']=(new_counts>>np.uint64(32)).astype(np.uint32)
        if scaling:
            table=np.asarray(header['MassTable'],float).copy();table[1]=0;header['MassTable']=table
            if 'Research_fine_dm_mean_mass_native' in header:header['Research_fine_dm_mean_mass_native']=float(changed_mass.mean())
            if not np.array_equal(group['Masses'][:],changed_mass):raise RuntimeError('Materialized native mass changes differ from the declared multiplier')
        header['Research_counterfactual_ic_schema']=3 if late else 2 if scaling else 1;header['Research_counterfactual_operation']=request['operation']
        header['Research_counterfactual_parent_sha256']=request['original_ic']['sha256']
        changed_stream=selected_state(native)
        if any(not np.array_equal(initial_stream[name],changed_stream[name]) for name in initial_stream):raise RuntimeError('Intervention changed the original selected stream phase/mass state')
    if file_sha256(original)!=request['original_ic']['sha256']:raise RuntimeError('Original native IC changed during intervention copying')
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={'schema':'native-counterfactual-ic-lineage-v1','execution_status':'COMPLETE','request':request,
        'original_ic':{'path':str(original),'sha256':request['original_ic']['sha256']},'native_ic':{'path':str(output),'sha256':file_sha256(output)},
        'removed_dm_count':0 if scaling else len(removed),'removed_dm_ids':[] if scaling else removed.tolist(),'affected_dm_ids':removed.tolist(),
        'mass_multiplier':request['mass_multiplier'] if scaling else 0.,'unchanged_stream_ids':stream.tolist(),
        'original_counts':counts.tolist(),'intervention_counts':new_counts.tolist(),'removed_mass_native':0. if scaling else removed_mass,
        'removed_momentum_native':[0.,0.,0.] if scaling else removed_momentum.tolist(),
        'affected_original_mass_native':removed_mass,'system_mass_change_native':removed_mass*(request['mass_multiplier']-1) if scaling else -removed_mass,
        'system_momentum_change_native':(removed_momentum*(request['mass_multiplier']-1) if scaling else -removed_momentum).tolist(),
        'selected_stream_state':'BYTE_VALUE_IDENTICAL_PHASE_AND_MASS','physical_operation':
            ('scale declared DM particle masses at the original IC epoch; original IDs/positions/velocities and other particles unchanged' if scaling else
             'remove declared DM identities at the original IC epoch; background parameters and remaining particles unchanged'),
        'full_system_initial_state':'CHANGED_MATTER_CONTENT','inherited_cosmological_ic_acceptance':'NOT_ESTABLISHED',
        'science_acceptance':'NOT_ESTABLISHED','source_hashes':sources}
    if late:
        report.update(intervention_epoch={'schema':'native-saved-intervention-epoch-v1','source_snapshot':request['source_snapshot'],
            'native_time':late_epoch,'clock':'scale_factor' if int(parameters_native['ComovingIntegrationOn']) else 'native_time_code'},
            physical_operation=request['operation']+' at the original saved late-time epoch; remaining collisionless phase/mass state retained',
            native_state_policy='FRESH_PHASE_SPACE_REINITIALIZATION; NOT_PARENT_INTEGRATOR_RESTART',
            source_state_scope='source-free collisionless population; hydro/BH/SIDM and individual/adaptive softening excluded')
        if control:report.update(full_system_initial_state='UNCHANGED_COLLISIONLESS_PHASE_AND_MASS',
            physical_operation='retain all original collisionless particles at the saved epoch; fresh integrator initialization only',
            control_role='UNCHANGED_STATE_REINITIALIZATION_CONTROL; EFFECT_REQUIRES_NATIVE_COMPARISON')
    manifest=destination/'counterfactual-ic.json';atomic_json(manifest,report);atomic_text(manifest.with_name(manifest.name+'.sha256'),file_sha256(manifest)+'\n');return report


def native_counterfactual_ic_runner(parameters,context):
    from pathlib import Path
    for key in ('memory_gib','disk_gib'):
        if parameters['request'][key]>context.resources[key]:raise MemoryError('Native intervention exceeds allocated '+key)
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Native intervention escapes its owned generation')
    return publish_native_counterfactual_ic(parameters,destination)


def native_counterfactual_ic_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Materialize an explicit native DM-removal IC counterfactual while preserving the selected stream')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.parameters.resolve();publish_native_counterfactual_ic(json.loads(source.read_text()),args.output);return 0


def validate_native_counterfactual_run_request(parameters):
    request=parameters['request']
    keys={'schema','intervention_ic','mpi_launcher','mpi_ranks','omp_threads','memory_gib','disk_gib','wall_hours','maximum_snapshot_bytes'}
    if isinstance(request,dict) and 'maximum_snapshots' in request:keys.add('maximum_snapshots')
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-counterfactual-run-request-v1':
        raise ValueError('Complete bounded original-engine native counterfactual execution request required')
    for name in ('mpi_ranks','omp_threads','maximum_snapshot_bytes'):
        if type(request[name])is not int or request[name]<1:raise ValueError('Exact positive native branch resource bound required')
    for name in ('memory_gib','disk_gib','wall_hours'):positive(request[name],name)
    if type(request.get('maximum_snapshots',4096))is not int or not 2<=request.get('maximum_snapshots',4096)<=2**31-1:
        raise ValueError('Exact native saved-output count ceiling of at least two required')
    for name in ('intervention_ic','mpi_launcher'):
        if not isinstance(request[name],dict) or set(request[name])!={'path','sha256'}:raise ValueError('Pin the actual native branch '+name)
    return request


def _native_late_parent_output_map(parent,expected,sources,maximum_snapshots,memory_bytes):
    """Bind future native times to original parent storage labels/shards."""
    import h5py
    from .analysis import find_hdf5_files,validate_gadget_snapshot_set
    from .foundation import file_sha256
    groups={};file_count=0
    for path in find_hdf5_files(parent/'data','snapshot_'):
        file_count+=1
        if file_count*4096>memory_bytes/8:raise MemoryError('Parent output mapping exceeds metadata allocation')
        label=int(path.name.split('_')[1].split('.')[0]);groups.setdefault(label,[]).append(path)
        if len(groups)>maximum_snapshots:raise MemoryError('Parent saved-output mapping exceeds its declared count ceiling')
    by_time={}
    for label,paths in sorted(groups.items()):
        for path in paths:
            if sources.get(str(path.resolve()))!=file_sha256(path):raise ValueError('Parent snapshot mapping is outside its completed native generation')
        validate_gadget_snapshot_set(paths,label)
        with h5py.File(paths[0],'r') as native:time=float(native['Header'].attrs['Time'])
        if not math.isfinite(time) or time in by_time:raise ValueError('Parent snapshot mapping has nonfinite or duplicate physical epochs')
        by_time[time]={'parent_snapshot':label,'parent_native_time':time,'parent_files':{str(path.resolve()):sources[str(path.resolve())] for path in paths}}
    from bisect import bisect_left,bisect_right
    times=sorted(by_time);result=[]
    for time in expected:
        tolerance=(abs(time)*1e-12+1e-14)/(1-1e-12)
        matches=[saved for saved in times[bisect_left(times,time-tolerance):bisect_right(times,time+tolerance)] if math.isclose(saved,time,rel_tol=1e-12,abs_tol=1e-14)]
        if len(matches)!=1:raise ValueError('Each future late output must match one original parent saved epoch')
        result.append(by_time[matches[0]])
    if len({row['parent_snapshot'] for row in result})!=len(result):raise ValueError('Late output schedule aliases parent saved epochs')
    return result


def native_counterfactual_run_input_bindings(parameters):
    import json,os,yaml
    from pathlib import Path
    from .sidm_analysis import _stage_native_bindings
    from .foundation import file_sha256
    from core.resources import mpi_ranks, omp_threads
    request=validate_native_counterfactual_run_request(parameters);sources=_stage_native_bindings(parameters)
    parent=Path(parameters['native_run_directory']).resolve();config=parent/'config.yaml'
    if sources.get(str(config))!=file_sha256(config):raise ValueError('Native branch lacks its actual parent configuration')
    cfg=yaml.safe_load(config.read_text())
    if cfg.get('sidm',{}).get('enabled',False):raise ValueError('SIDM branch execution requires its regenerated intervention support/model contract')
    if request['mpi_ranks']!=mpi_ranks(cfg) or request['omp_threads']!=omp_threads(cfg):raise ValueError('Initial native branch profile preserves original MPI/OpenMP decomposition')
    for name in ('Gadget4','param.txt','Config.sh'):
        path=parent/'input'/name
        if sources.get(str(path))!=file_sha256(path):raise ValueError('Native branch requires its owned original '+name)
    flags={line.split('#',1)[0].strip().split('=',1)[0] for line in (parent/'input/Config.sh').read_text().splitlines()}
    if 'SELFGRAVITY' not in flags:raise ValueError('Live-host native branch requires its actual self-gravity compile profile')
    if 'OUTPUT_EXACT_SYNCHRONIZED' not in flags or 'OUTPUT_NON_SYNCHRONIZED_ALLOWED' in flags:
        raise ValueError('Matched native counterfactuals require the original exact synchronized-output profile')
    base=Path(parameters.get('request_base','.')).resolve();lineage_path=(base/request['intervention_ic']['path']).resolve()
    if file_sha256(lineage_path)!=request['intervention_ic']['sha256'] or lineage_path.with_name(lineage_path.name+'.sha256').read_text().strip()!=request['intervention_ic']['sha256']:
        raise ValueError('Native intervention lineage seal changed')
    if lineage_path.stat().st_size*16>request['memory_gib']*1024**3/4:raise MemoryError('Native intervention metadata exceeds allocation')
    lineage=json.loads(lineage_path.read_text())
    if lineage.get('schema')!='native-counterfactual-ic-lineage-v1' or lineage.get('execution_status')!='COMPLETE':raise ValueError('A supplied changed IC is not a completed physical intervention product')
    for path,digest in lineage['source_hashes'].items():
        if file_sha256(path)!=digest:raise ValueError('Native intervention source changed: '+path)
    original=Path(lineage['original_ic']['path']).resolve()
    if sources.get(str(original))!=lineage['original_ic']['sha256']:raise ValueError('Intervention was produced from a different original native system')
    initial=Path(lineage['native_ic']['path']).resolve()
    if file_sha256(initial)!=lineage['native_ic']['sha256']:raise ValueError('Native intervention IC payload changed')
    launcher=(base/request['mpi_launcher']['path']).resolve()
    if file_sha256(launcher)!=request['mpi_launcher']['sha256'] or not os.access(launcher,os.X_OK) or not os.access(parent/'input/Gadget4',os.X_OK):
        raise ValueError('Native branch requires executable pinned launcher and original engine')
    sources.update(lineage['source_hashes']);sources[str(lineage_path)]=request['intervention_ic']['sha256'];sources[str(initial)]=lineage['native_ic']['sha256'];sources[str(launcher)]=request['mpi_launcher']['sha256']
    return request,parent,cfg,lineage,initial,launcher,sources


def native_counterfactual_run_runner(parameters,context):
    """Evolve a fresh explicit IC intervention with the original native engine.

    Runs only when this stage is executed. No parent restart bytes are edited,
    and original IC acceptance is not inherited by a changed matter system.
    """
    import os,json,shutil,subprocess,h5py
    from pathlib import Path
    from .foundation import atomic_json, atomic_text, atomic_copy, file_sha256
    from core.native import run_native_command
    from .evolution import parameter_values,gadget_ic_prefix
    from .analysis import find_hdf5_files,validate_gadget_snapshot_set
    from core.resources import LocalProcessResourceSampler
    request,parent,cfg,lineage,initial,launcher,sources=native_counterfactual_run_input_bindings(parameters)
    for name in ('mpi_ranks','omp_threads'):
        if request[name]!=context.resources[name]:raise ValueError('Native branch changes frozen workflow '+name)
    for name in ('memory_gib','disk_gib'):
        if request[name]>context.resources[name]:raise MemoryError('Native branch exceeds workflow '+name)
    destination=Path(parameters['destination']).resolve()
    if destination.exists() or not destination.is_relative_to(context.generation):raise ValueError('Native branch needs a fresh owned run directory')
    original_ic=Path(lineage['original_ic']['path']).resolve()
    retained=[path for path in (parent/'input').iterdir() if path.is_file() and path.resolve()!=original_ic]
    footprint=sum(path.stat().st_size for path in retained)+initial.stat().st_size+request['maximum_snapshot_bytes']
    ancestor=destination.parent
    while not ancestor.exists():ancestor=ancestor.parent
    if footprint>request['disk_gib']*1024**3 or footprint>shutil.disk_usage(ancestor).free:raise MemoryError('Native branch inputs/output allowance exceed allocated/available disk')
    params=parameter_values(parent/'input/param.txt')
    late=lineage['request']['schema']=='native-counterfactual-ic-request-v3'
    if late:
        epoch=lineage.get('intervention_epoch')
        if not isinstance(epoch,dict) or epoch.get('schema')!='native-saved-intervention-epoch-v1' or lineage.get('native_state_policy')!='FRESH_PHASE_SPACE_REINITIALIZATION; NOT_PARENT_INTEGRATOR_RESTART':
            raise ValueError('Late native launch lacks an explicit saved epoch and fresh-state contract')
        start=float(epoch['native_time'])
        if not math.isfinite(start) or not float(params['TimeBegin'])<start<float(params['TimeMax']):raise ValueError('Late launch epoch is outside the original native evolution')
        params['TimeBegin']=format(start,'.17g')
    with h5py.File(initial,'r') as native:
        if float(native['Header'].attrs['Time'])!=float(params['TimeBegin']):raise ValueError('Intervention IC does not start at the preserved native clock epoch')
    # Resolve all scientific native file parameters before changing cwd. They
    # remain actual parent inputs unless an owned same-name copy is present.
    file_keys={'TreecoolFile','CosmologyTableFile','StellarEvolutionTableFile','GrackleDataFile'}
    for name in file_keys:
        value=params.get(name)
        if value and value.lower() not in {'none',''}:
            path=Path(value);path=(path if path.is_absolute() else parent/'input'/path).resolve()
            if sources.get(str(path))!=file_sha256(path):raise ValueError('Native physical input is unbound: '+name)
            params[name]=str(path)
    if int(params.get('OutputListOn','0'))!=1:raise ValueError('Native branch requires the original explicit output schedule')
    schedule=Path(params['OutputListFilename']);schedule=(schedule if schedule.is_absolute() else parent/'input'/schedule).resolve()
    if sources.get(str(schedule))!=file_sha256(schedule):raise ValueError('Native output schedule is not bound to the original evolution')
    scheduled=[]
    for line in schedule.read_text().splitlines():
        fields=line.split()
        if not fields or fields[0].startswith(('#','%')):continue
        if len(fields)>2 or len(fields)==2 and fields[1] not in ('0','1'):raise ValueError('Original output list has an unsupported row format')
        value=float(fields[0])
        if not math.isfinite(value):raise ValueError('Original output schedule is nonfinite')
        if (len(fields)==1 or fields[1]=='1') and float(params['TimeBegin'])<=value<=float(params['TimeMax']):scheduled.append(value)
    expected=sorted(set(scheduled))
    maximum_snapshots=request.get('maximum_snapshots',4096)
    if len(expected)>maximum_snapshots or len(expected)*8192>request['memory_gib']*1024**3/8:
        raise MemoryError('Native output schedule exceeds its saved-output/metadata allocation')
    if not expected or late and (expected[0]!=float(params['TimeBegin']) or len(expected)<2):
        raise ValueError('Late native launch requires its saved start and at least one subsequent original scheduled output')
    parent_output_map=_native_late_parent_output_map(parent,expected,sources,maximum_snapshots,request['memory_gib']*1024**3) if late else None
    destination.mkdir(parents=True);(destination/'input').mkdir();(destination/'data').mkdir()
    for path in retained:
        if sources.get(str(path))!=file_sha256(path):raise ValueError('Original retained native input changed')
        atomic_copy(path,destination/'input'/path.name)
    binary=destination/'input/Gadget4';binary.chmod(0o755);owned_ic=destination/'input/counterfactual.hdf5';atomic_copy(initial,owned_ic)
    if late:atomic_text(destination/'input/output_times.txt',''.join(format(value,'.17g')+' 1\n' for value in expected))
    else:atomic_copy(schedule,destination/'input/output_times.txt')
    params.update(InitCondFile=str(gadget_ic_prefix(owned_ic)),OutputDir=str(destination/'data')+'/',OutputListFilename=str(destination/'input/output_times.txt'),
        CpuTimeBetRestartFile=str(request['wall_hours']*3600),TimeLimitCPU=str(request['wall_hours']*3600),
        MaxMemSize=str(max(1,int(.85*request['memory_gib']*1024/request['mpi_ranks']))))
    param=destination/'input/param.txt';atomic_text(param,''.join(key+' '+str(value)+'\n' for key,value in params.items()))
    if late:
        import copy,yaml
        branch_config=copy.deepcopy(cfg)
        if int(params['ComovingIntegrationOn']):branch_config['simulation']['start_redshift']=1/float(params['TimeBegin'])-1
        atomic_text(destination/'config.yaml',yaml.safe_dump(branch_config,sort_keys=False))
    else:atomic_copy(parent/'config.yaml',destination/'config.yaml')
    parent_manifest=parent/'manifest.json'
    if sources.get(str(parent_manifest))!=file_sha256(parent_manifest):raise ValueError('Native branch needs its original source-bound background manifest')
    original_manifest=json.loads(parent_manifest.read_text());background=original_manifest.get('input_artifacts',{}).get('boltzmann_background')
    artifacts={}
    if background:
        path=Path(background['path']);path=(path if path.is_absolute() else parent/path).resolve()
        if sources.get(str(path))!=background['sha256'] or file_sha256(path)!=background['sha256']:raise ValueError('Native branch background clock changed')
        artifacts['boltzmann_background']={**background,'path':str(path)}
    atomic_json(destination/'manifest.json',{'schema':'native-counterfactual-run-clock-v1','input_artifacts':artifacts,
        'snapshot_redshifts_requested':[1/value-1 for value in expected] if late and int(params['ComovingIntegrationOn']) else original_manifest.get('snapshot_redshifts_requested'),'science_acceptance':'NOT_ESTABLISHED',
        'counterfactual_parent_run':str(parent),'counterfactual_ic_lineage':request['intervention_ic'],
        'state_origin':{'kind':'fresh_saved_collisionless_state','native_time':float(params['TimeBegin']),
            'source_snapshot':lineage['intervention_epoch']['source_snapshot']} if late else {'kind':'original_ic_intervention'}})
    invocation_inputs={str(path):file_sha256(path) for path in (binary,param,owned_ic,destination/'input/output_times.txt',destination/'input/Config.sh',launcher)}
    invocation_inputs.update({params[name]:sources[params[name]] for name in file_keys if params.get(name) and params[name].lower() not in {'none',''}})
    sampler=LocalProcessResourceSampler(destination/'resource-samples.json');environment=os.environ.copy();environment['OMP_NUM_THREADS']=str(request['omp_threads'])
    with (destination/'native.log').open('w') as log:
        run_native_command([str(launcher),'-np',str(request['mpi_ranks']),str(binary),str(param),'0'],cwd=destination/'input',
            receipt_path=destination/'invocation.json',inputs=list(invocation_inputs),env=environment,stdout=log,stderr=subprocess.STDOUT,
            timeout=request['wall_hours']*3600,process_observer=sampler)
    groups={}
    for path in find_hdf5_files(destination/'data','snapshot_'):
        label=int(path.name.split('_')[1].split('.')[0]);groups.setdefault(label,[]).append(path)
    if not groups:raise RuntimeError('Native counterfactual returned without original snapshot outputs')
    outputs=[];size=0
    for label,paths in sorted(groups.items()):
        validate_gadget_snapshot_set(paths,label)
        size+=sum(path.stat().st_size for path in paths)
        with h5py.File(paths[0],'r') as native:actual=float(native['Header'].attrs['Time'])
        outputs.append({'snapshot':label,'actual_native_time':actual,'files':{str(path.resolve()):file_sha256(path) for path in paths}})
    actual=[item['actual_native_time'] for item in outputs]
    if not expected or len(actual)!=len(expected) or not np.allclose(actual,expected,rtol=1e-12,atol=1e-14):
        raise RuntimeError('Native counterfactual did not retain its complete preserved exact output schedule')
    if size>request['maximum_snapshot_bytes'] or sum(path.stat().st_size for path in destination.rglob('*') if path.is_file())>request['disk_gib']*1024**3:
        raise MemoryError('Actual native branch outputs exceed their frozen storage bounds')
    if file_sha256(initial)!=lineage['native_ic']['sha256']:raise RuntimeError('Original intervention input changed during native execution')
    report={'schema':'native-counterfactual-run-v1','execution_status':'COMPLETE','request':request,'native_run_directory':str(destination),
        'parent_run':str(parent),'intervention_ic_lineage':request['intervention_ic'],'measured_outputs':outputs,
        'invocation_receipt':{'path':str(destination/'invocation.json'),'sha256':file_sha256(destination/'invocation.json')},
        'original_engine_sha256':sources[str(parent/'input/Gadget4')],'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED',
        'scope':'fresh original-engine collisionless IC-removal experiment; preserved physical/numerical parameters except owned input/output and resource limits; no inherited cosmological or causal accuracy acceptance'}
    if late:report.update(intervention_epoch=lineage['intervention_epoch'],native_state_policy=lineage['native_state_policy'],
        preserved_late_output_times=expected,parent_snapshot_labels=[row['parent_snapshot'] for row in parent_output_map],
        branch_snapshot_labels=[row['snapshot'] for row in outputs],
        run_role='UNCHANGED_STATE_REINITIALIZATION_CONTROL' if lineage['request']['operation']=='retain_collisionless_state' else 'LATE_COLLISIONLESS_INTERVENTION',
        parent_branch_output_pairs=[{**original,'branch_snapshot':changed['snapshot'],'branch_native_time':changed['actual_native_time'],
            'branch_files':changed['files']} for original,changed in zip(parent_output_map,outputs)],
        scope='fresh source-free collisionless late-state intervention with original future output times; parent integrator history is not preserved')
    atomic_json(destination/'counterfactual-run.json',report);return report


def validate_native_stream_worldline_request(parameters):
    request=parameters['request']
    keys={'schema','snapshots','species','particle_ids','particle_namespace','maximum_snapshots','maximum_selected_particles','maximum_census_particles','memory_gib'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-stream-worldline-request-v1':
        raise ValueError('Complete original native stream worldline measurement request required')
    for name in ('maximum_snapshots','maximum_selected_particles','maximum_census_particles'):
        integer(request[name],name)
        if type(request[name])is not int or request[name]<1:raise ValueError('Exact positive native stream bounds required')
    snapshots=request['snapshots']
    deferred=isinstance(snapshots,dict) and set(snapshots)=={'stage','artifact','load','field'} and snapshots.get('load')=='json' and snapshots.get('field') in (['parent_snapshot_labels'],['branch_snapshot_labels'])
    if not deferred and (not isinstance(snapshots,list) or not 2<=len(snapshots)<=request['maximum_snapshots'] or len(set(snapshots))!=len(snapshots) or any(type(value)is not int or value<0 for value in snapshots)):
        raise ValueError('At least two distinct original native snapshot labels required')
    if type(request['species'])is not int or not 0<=request['species']<=4 or not isinstance(request['particle_namespace'],str) or not request['particle_namespace']:
        raise ValueError('Explicit native stream species and exact namespace required')
    from .data import exact_uint64_ids
    ids=exact_uint64_ids(request['particle_ids'])
    if not 1<=len(ids)<=request['maximum_selected_particles'] or len(np.unique(ids))!=len(ids):raise ValueError('Distinct bounded exact stream identities required')
    memory=positive(request['memory_gib'],'native stream workspace')*1024**3
    if (request['maximum_snapshots'] if deferred else len(snapshots))*len(ids)*256+request['maximum_census_particles']*800>memory:raise MemoryError('Native stream trajectories and original census exceed allocation')
    return request


def native_stream_worldline_input_bindings(parameters):
    from .sidm_analysis import _stage_native_bindings
    validate_native_stream_worldline_request(parameters)
    return _stage_native_bindings(parameters)


def publish_native_stream_worldlines(parameters,destination):
    """Read the selected exact native stream population at actual saved epochs.

    Positions retain the native periodic box origin/wrapping. No ambiguous
    unwrapping, interpolated output epoch or fabricated self-consistent wake
    is introduced by this measurement layer.
    """
    import h5py
    from pathlib import Path
    from .observation_geometry import read_native_gravitating_particles
    from .halos import _snapshot_files
    from .data import exact_uint64_ids,export_numeric_audit,native_box_proper_kpc
    from .foundation import file_sha256,KPC_CM
    request=validate_native_stream_worldline_request(parameters);sources=native_stream_worldline_input_bindings(parameters)
    if not isinstance(request['snapshots'],list):raise ValueError('Native worldline output-label selectors must resolve before scientific measurement')
    ids=exact_uint64_ids(request['particle_ids']);ids=np.sort(ids);positions=[];velocities=[];masses=[];times=[];boxes=[];epochs=[]
    root=Path(parameters['native_run_directory']).resolve();configuration=root/'input/Config.sh'
    if sources.get(str(configuration))!=file_sha256(configuration):raise ValueError('Native stream lacks its original compile configuration')
    flags={line.split('#',1)[0].strip().split('=',1)[0] for line in configuration.read_text().splitlines()};periodic='PERIODIC' in flags
    clock_kind=None
    for snapshot in request['snapshots']:
        files=_snapshot_files(root,snapshot)
        if not files:raise FileNotFoundError('Original stream snapshot set is missing')
        native_box=None
        for path in files:
            if sources.get(str(path.resolve()))!=file_sha256(path):raise ValueError('Stream snapshot is outside its completed native generation')
            with h5py.File(path,'r') as native:
                value=native_box_proper_kpc(native['Header'].attrs,native['Parameters'].attrs)
                if not math.isfinite(value) or value<=0 or native_box is not None and value!=native_box:raise ValueError('Native stream shards change their physical box')
                native_box=value
        population,particles=read_native_gravitating_particles(root,snapshot,species=[request['species']],maximum_particles=request['maximum_census_particles'],
            memory_gib=request['memory_gib']/4,namespace=request['particle_namespace'])
        sources.update(population['source_hashes']);order=np.argsort(particles['particle_ids']);at=np.searchsorted(particles['particle_ids'][order],ids)
        if np.any(at>=len(order)) or not np.array_equal(particles['particle_ids'][order[at]],ids):
            raise ValueError('Original native stream snapshot loses a selected exact identity')
        selected=order[at];epoch=population['epoch'];kind=epoch['canonical_time']
        if clock_kind is not None and kind!=clock_kind:raise ValueError('Native stream changes its actual clock representation')
        clock_kind=kind;time=epoch.get('cosmic_time_gyr') if kind=='scale_factor' else epoch.get('elapsed_time_gyr')
        if time is None or not math.isfinite(time):raise ValueError('Native stream has no measured proper-time conversion')
        times.append(1000*time);epochs.append(epoch);boxes.append(native_box)
        positions.append(particles['position_proper_kpc'][selected]);velocities.append(particles['velocity_peculiar_kms'][selected]);masses.append(particles['mass_msun'][selected])
    if np.any(np.diff(times)<=0):raise ValueError('Native stream saved epochs must be strictly ordered in actual proper time')
    arrays={'particle_ids':ids,'position_kpc':np.asarray(positions),'velocity_kms':np.asarray(velocities),
        'mass_msun':np.asarray(masses),'times_myr':np.asarray(times),'periodic_box_proper_kpc':np.asarray(boxes)}
    initial={'position_kpc':arrays['position_kpc'][0],'velocity_kms':arrays['velocity_kms'][0],'mass_msun':arrays['mass_msun'][0],
        'particle_ids':ids,'namespace':request['particle_namespace']}
    frozen=freeze_common_initial_stream_state(initial);source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={'schema':'measured-native-stream-worldlines-v1','execution_status':'COMPLETE','request':request,'namespace':request['particle_namespace'],
        'epochs':epochs,'periodic':periodic,'initial_state_sha256':frozen['state_sha256'],'source_hashes':sources,
        'native_run_directory':str(root),'native_stage_receipt':{'path':str(Path(parameters['native_stage_receipt']).resolve()),
            'sha256':file_sha256(parameters['native_stage_receipt'])},
        'native_build_flags':sorted(flags),'native_self_gravity_enabled':'SELFGRAVITY' in flags,
        'scope':'actual original native positions/peculiar velocities and time-dependent masses for unchanged exact selected IDs; native periodic wrapping retained',
        'host_response':'native evolved source; causal intervention and convergence require separate paired evidence','science_acceptance':'NOT_ESTABLISHED'}
    units={'particle_ids':'identity','position_kpc':'proper kpc','velocity_kms':'km / s','mass_msun':'Msun','times_myr':'Myr','periodic_box_proper_kpc':'proper kpc'}
    specs={name:{'units':units[name],'quantity':'measured original native stream '+name} for name in arrays};specs['particle_ids']['identity_namespace']=request['particle_namespace']
    return export_numeric_audit(destination,report,arrays,specs,algorithm=source,source_hashes=sources,
        context={'epoch':{'canonical_time':clock_kind,'measured_epochs':epochs},'coordinates':{'frame':'native box origin; periodic wrapped proper positions and peculiar velocity'}})


def native_stream_worldline_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native stream measurement exceeds allocated memory')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Native stream measurement escapes its owned generation')
    return publish_native_stream_worldlines(parameters,destination)


def native_stream_worldline_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Measure selected original native stream trajectories at actual physical epochs')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.parameters.resolve();publish_native_stream_worldlines(json.loads(source.read_text()),args.output);return 0


def compile_native_counterfactual_campaign(request,destination,*,request_base):
    """Expand saved live-response requests into real native intervention stages.

    Component-to-particle mappings are actual source-bound DM memberships,
    never guessed from rendered fields. The original completed run supplies
    the native reference trajectory; changed branches get new ICs/executions.
    """
    if isinstance(request,dict) and request.get('schema')=='native-late-counterfactual-campaign-request-v1':
        return compile_native_late_counterfactual_campaign(request,destination,request_base=request_base)
    import copy,json
    from pathlib import Path
    from .sidm_analysis import _load_product,_stage_native_bindings
    from .foundation import atomic_json,file_sha256
    from .workflow import safe_id,compile_requested_product_dag,default_registry
    keys={'schema','counterfactual_product','component_memberships','parent_native_stage_receipt','parent_native_run_directory',
        'original_ic','stream_request','ic_limits','run_request','comparison_limits','live_host_membership','result_limits','resources','followup_limits','maximum_interventions'}
    if isinstance(request,dict) and 'host_reflex_request' in request:keys.add('host_reflex_request')
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-live-counterfactual-campaign-request-v1':
        raise ValueError('Complete automatic native live-response campaign request required')
    if type(request['maximum_interventions'])is not int or request['maximum_interventions']<1:raise ValueError('Bounded live intervention count required')
    base=Path(request_base).resolve();destination=Path(destination).resolve();resources=request['resources'];memory=resources['memory_gib']*1024**3
    if not isinstance(request['ic_limits'],dict) or set(request['ic_limits'])!={'maximum_particles','memory_gib','disk_gib','chunk_rows'}:
        raise ValueError('Freeze only the explicit native IC resource limits')
    if not isinstance(request['comparison_limits'],dict) or set(request['comparison_limits'])!={'memory_gib','maximum_snapshots','maximum_selected_particles','time_absolute_tolerance_myr','maximum_periodic_difference_box_fraction'}:
        raise ValueError('Freeze only the explicit native comparison limits')
    if not isinstance(request['result_limits'],dict) or set(request['result_limits'])!={'memory_gib','maximum_census_particles','minimum_host_particles'}:
        raise ValueError('Freeze the actual live-host evidence census limits')
    if destination.exists():raise FileExistsError('Keep previous native counterfactual campaign designs')
    pinned_sources={}
    def pin(reference):
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('Pin complete original native campaign assets')
        path=(base/reference['path']).resolve()
        if file_sha256(path)!=reference['sha256']:raise ValueError('Native campaign asset changed')
        pinned_sources[str(path)]=reference['sha256']
        return str(path)
    receipt=pin(request['parent_native_stage_receipt']);original_ic={'path':pin(request['original_ic']),'sha256':request['original_ic']['sha256']}
    parent=(base/request['parent_native_run_directory']).resolve();stream=copy.deepcopy(request['stream_request'])
    validate_native_stream_worldline_request({'request':stream})
    host_stream=None;host_limits=None
    if 'host_reflex_request' in request:
        host=request['host_reflex_request']
        if not isinstance(host,dict) or set(host)!={'worldline_request','anchor_particle_id','minimum_host_particles',
                'maximum_host_radius_box_fraction','maximum_stream_radius_box_fraction'}:
            raise ValueError('Complete optional native host-reflex worldline selection and limits required')
        host_stream=copy.deepcopy(host['worldline_request']);validate_native_stream_worldline_request({'request':host_stream})
        if host_stream['snapshots']!=stream['snapshots'] or host_stream['particle_namespace']!=stream['particle_namespace'] or host_stream['memory_gib']>resources['memory_gib']:
            raise ValueError('Campaign host and stream selections require common snapshots/namespace and bounded resources')
        host_limits={key:copy.deepcopy(value) for key,value in host.items() if key!='worldline_request'}
        host_limits['maximum_host_particles']=host_stream['maximum_selected_particles']
        provisional={'schema':'native-stream-paired-comparison-request-v1',**copy.deepcopy(request['comparison_limits']),
            'reference':{'path':'pending','sha256':'0'*64},'intervention':{'path':'pending','sha256':'0'*64},
            'host_reflex':{**host_limits,'reference':{'path':'pending','sha256':'0'*64},'intervention':{'path':'pending','sha256':'0'*64}}}
        validate_native_stream_comparison_request({'request':provisional})
        from .data import exact_uint64_ids
        host_ids=exact_uint64_ids(host_stream['particle_ids'],allow_empty=False)
        if len(host_ids)<host_limits['minimum_host_particles'] or host_limits['anchor_particle_id'] not in host_ids or np.intersect1d(host_ids,stream['particle_ids']).size:
            raise ValueError('Campaign host must have a supported anchor and be distinct from its selected stream')
    if stream['memory_gib']>resources['memory_gib']:raise MemoryError('Native campaign stream measurement exceeds allocation')
    parent_sources=_stage_native_bindings({'request':stream,'native_stage_receipt':receipt,'native_run_directory':str(parent)})
    forward,arrays,sources=_load_product(request['counterfactual_product'],base,memory/4);sources.update(parent_sources)
    if forward.get('schema')!='stream-counterfactuals-v1':raise ValueError('Native campaign requires the actual imposed-field counterfactual producer with saved live branch requests')
    if forward.get('namespace')!=stream['particle_namespace']:raise ValueError('Native campaign changes the imposed-field stream identity namespace')
    branches=forward.get('live_branch_requests')
    if not isinstance(branches,list) or not 1<=len(branches)<=request['maximum_interventions']:raise ValueError('Explicit bounded actual live-response branch requests required')
    from .data import exact_uint64_ids
    ids=exact_uint64_ids(stream['particle_ids'])
    if 'particle_ids' not in arrays or not np.array_equal(np.sort(exact_uint64_ids(arrays['particle_ids'])),np.sort(ids)):
        raise ValueError('Native campaign changes the selected imposed-field stream identity population')
    required_stages=1+5*len(branches)+(1+len(branches) if host_stream is not None else 0)
    if required_stages>resources['maximum_stage_count']:raise MemoryError('Complete intervention/run/measurement/comparison/evidence graph exceeds the frozen stage ceiling')
    memberships=request['component_memberships'];stages=[];requested=[];used=set();mappings=[]
    stages.append({'stage_id':'native_reference','kind':'native_stream_worldlines',
        'parameters':{'request':stream,'native_stage_receipt':receipt,'native_run_directory':str(parent),'destination':'{generation}/worldlines.json'},
        'outputs':{'native_reference':'worldlines.json'}})
    def artifact(stage,path,field=None):
        result={'stage':stage,'artifact':path}
        if field is not None:result.update(load='json',field=field)
        return result
    def product(stage,path):return {'path':artifact(stage,path),'sha256':{**artifact(stage,path),'load':'sha256'}}
    if host_stream is not None:
        stages.append({'stage_id':'native_host_reference','kind':'native_stream_worldlines',
            'parameters':{'request':copy.deepcopy(host_stream),'native_stage_receipt':receipt,'native_run_directory':str(parent),
                'destination':'{generation}/host-worldlines.json'},'outputs':{'native_host_reference':'host-worldlines.json'}})
    for branch in branches:
        name=safe_id(branch['name']);intervention=branch['intervention']
        if name in used or branch.get('status')!='NATIVE_BRANCH_REQUIRED' or intervention.get('operation') not in {'subtract','scale_mass'}:
            raise ValueError('Native campaign materializes explicit DM subtraction/mass-scaling requests; other live interventions need their own physical transformer')
        used.add(name);components=intervention.get('components')
        if not isinstance(components,list) or not components or len(set(components))!=len(components):raise ValueError('Live subtraction requires distinct recorded component identities')
        removals=[];references=[]
        for component in components:
            if component not in memberships:raise ValueError('Live component has no supplied actual native DM membership: '+component)
            measured,values,binding=_load_product(memberships[component],base,memory/4);sources.update(binding)
            if measured.get('schema')!='native-halo-membership-measurement-v1' or measured.get('request',{}).get('species')!=[1]:
                raise ValueError('Initial live subtraction mapping requires measured DM-only native memberships')
            if measured['request']['particle_namespace']!=stream['particle_namespace']:raise ValueError('Live component changes the original native particle namespace')
            original_sources=measured.get('native_particle_source_hashes',{})
            if not original_sources or any(parent_sources.get(path)!=checksum for path,checksum in original_sources.items()):
                raise ValueError('Live component membership is outside the actual parent native evolution')
            selected=exact_uint64_ids(values['member_particle_ids'])
            if np.intersect1d(selected,ids).size:raise ValueError('Live component removal overlaps the unchanged selected stream population')
            removals.extend(selected.tolist());references.append({'component':component,'membership':memberships[component],'definition':measured['membership_definition']})
        if not removals or len(set(removals))!=len(removals):raise ValueError('Native live components are empty or double-count removed matter')
        if host_stream is not None and np.intersect1d(host_ids,np.asarray(removals,np.uint64)).size:
            raise ValueError('Campaign intervention changes the selected unchanged host-reflex cohort')
        ic_id='ic_'+name;run_id='run_'+name;history_id='history_'+name;comparison_id='comparison_'+name
        ic={'schema':'native-counterfactual-ic-request-v1','original_ic':original_ic,'operation':'remove_dm_particles','removed_dm_particle_ids':sorted(removals),
            'stream_particle_ids':stream['particle_ids'],'stream_species':stream['species'],'particle_namespace':stream['particle_namespace'],**copy.deepcopy(request['ic_limits'])}
        if intervention['operation']=='scale_mass':
            ic.update(schema='native-counterfactual-ic-request-v2',operation='scale_dm_particle_masses',
                affected_dm_particle_ids=ic.pop('removed_dm_particle_ids'),mass_multiplier=intervention['mass_multiplier'])
        validate_native_counterfactual_ic_request({'request':ic})
        stages.append({'stage_id':ic_id,'kind':'native_counterfactual_ic','parameters':{'request':ic,'native_stage_receipt':receipt,
            'native_run_directory':str(parent),'destination':'{generation}/counterfactual-ic'},'outputs':{ic_id:'counterfactual-ic/counterfactual-ic.json'}})
        run=copy.deepcopy(request['run_request']);run['intervention_ic']=product(ic_id,'counterfactual-ic/counterfactual-ic.json')
        run['mpi_launcher']={'path':pin(run['mpi_launcher']),'sha256':run['mpi_launcher']['sha256']}
        validate_native_counterfactual_run_request({'request':run})
        stages.append({'stage_id':run_id,'kind':'native_counterfactual_run','parameters':{'request':run,'request_base':str(base),
            'native_stage_receipt':receipt,'native_run_directory':str(parent),'destination':'{generation}/counterfactual-run'},
            'depends_on':[ic_id],'outputs':{run_id:'counterfactual-run/counterfactual-run.json'}})
        stages.append({'stage_id':history_id,'kind':'native_stream_worldlines','parameters':{'request':copy.deepcopy(stream),
            'native_stage_receipt':artifact(run_id,'stage-receipt.json'),
            'native_run_directory':artifact(run_id,'counterfactual-run/counterfactual-run.json',['native_run_directory']),
            'destination':'{generation}/worldlines.json'},'depends_on':[run_id],'outputs':{history_id:'worldlines.json'}})
        comparison={'schema':'native-stream-paired-comparison-request-v1','reference':product('native_reference','worldlines.json'),
            'intervention':product(history_id,'worldlines.json'),**copy.deepcopy(request['comparison_limits'])}
        dependencies=['native_reference',history_id]
        if host_stream is not None:
            host_history_id='host_'+name
            stages.append({'stage_id':host_history_id,'kind':'native_stream_worldlines','parameters':{'request':copy.deepcopy(host_stream),
                'native_stage_receipt':artifact(run_id,'stage-receipt.json'),
                'native_run_directory':artifact(run_id,'counterfactual-run/counterfactual-run.json',['native_run_directory']),
                'destination':'{generation}/host-worldlines.json'},'depends_on':[run_id],'outputs':{host_history_id:'host-worldlines.json'}})
            comparison['host_reflex']={**copy.deepcopy(host_limits),'reference':product('native_host_reference','host-worldlines.json'),
                'intervention':product(host_history_id,'host-worldlines.json')}
            dependencies.extend(['native_host_reference',host_history_id])
        validate_native_stream_comparison_request({'request':comparison})
        stages.append({'stage_id':comparison_id,'kind':'native_stream_comparison','parameters':{'request':comparison,'request_base':str(base),
            'destination':'{generation}/comparison.json'},'depends_on':dependencies,'outputs':{comparison_id:'comparison.json'}})
        result_id='result_'+name
        result={'schema':'native-counterfactual-result-request-v1','comparison':product(comparison_id,'comparison.json'),
            'run_report':product(run_id,'counterfactual-run/counterfactual-run.json'),'run_stage_receipt':product(run_id,'stage-receipt.json'),
            'host_membership':copy.deepcopy(request['live_host_membership']),**copy.deepcopy(request['result_limits'])}
        validate_native_counterfactual_result_request({'request':result})
        stages.append({'stage_id':result_id,'kind':'native_counterfactual_result','parameters':{'request':result,'request_base':str(base),
            'destination':'{generation}/result.json'},'depends_on':[comparison_id,run_id,history_id],'outputs':{result_id:'result.json'}})
        requested.append(result_id);mappings.append({'branch':name,'original_intervention':intervention,'component_mappings':references,
            'physical_operation':ic['operation']+' at the original IC epoch','comparison_product':comparison_id,'native_result_product':result_id})
    workflow={'schema':'research-workflow-v1','resources':resources,'followup_limits':request['followup_limits'],'stages':stages,'requested_products':requested}
    compile_requested_product_dag(workflow,default_registry(),request_base=base)
    sources.update(pinned_sources)
    destination.mkdir(parents=True);atomic_json(destination/'workflow.json',workflow)
    report={'schema':'native-live-counterfactual-campaign-design-v1','request':request,'workflow':str(destination/'workflow.json'),
        'workflow_sha256':file_sha256(destination/'workflow.json'),'mapped_native_branches':mappings,'stage_count':len(stages),'source_hashes':sources,
        'execution_status':'NOT_EXECUTED','science_acceptance':'NOT_ESTABLISHED',
        'imposed_initial_state_sha256':forward['initial_state_sha256'],
        'scope':'automatic DM-membership subtraction IC/native-run/worldline/paired-comparison graph; component mappings explicit; no unperformed native execution or inherited causal/accuracy acceptance'}
    atomic_json(destination/'campaign-design.json',report);return report


def native_counterfactual_campaign_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Compile actual native live-response branches from saved stream counterfactual requests')
    parser.add_argument('request',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.request.resolve();compile_native_counterfactual_campaign(json.loads(source.read_text()),args.output,request_base=source.parent);return 0


def validate_native_counterfactual_campaign_request(parameters):
    request=parameters['request']
    if isinstance(request,dict) and request.get('schema')=='native-late-counterfactual-campaign-request-v1':
        return validate_native_late_campaign_request(parameters)
    keys={'schema','counterfactual_product','component_memberships','parent_native_stage_receipt','parent_native_run_directory',
        'original_ic','stream_request','ic_limits','run_request','comparison_limits','live_host_membership','result_limits','resources','followup_limits','maximum_interventions'}
    if isinstance(request,dict) and 'host_reflex_request' in request:keys.add('host_reflex_request')
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-live-counterfactual-campaign-request-v1':
        raise ValueError('Complete automatic native live-response request required')
    if type(request['maximum_interventions'])is not int or request['maximum_interventions']<1:raise ValueError('Exact positive native intervention ceiling required')
    for name in ('component_memberships','resources','stream_request','ic_limits','run_request','comparison_limits','result_limits','followup_limits'):
        if not isinstance(request[name],dict):raise ValueError('Native campaign requires a typed '+name+' mapping')
    if 'host_reflex_request' in request:
        host=request['host_reflex_request']
        if not isinstance(host,dict) or set(host)!={'worldline_request','anchor_particle_id','minimum_host_particles',
                'maximum_host_radius_box_fraction','maximum_stream_radius_box_fraction'}:
            raise ValueError('Complete host-reflex campaign request required')
        validate_native_stream_worldline_request({'request':host['worldline_request']})
        limits={key:value for key,value in host.items() if key!='worldline_request'}
        limits['maximum_host_particles']=host['worldline_request']['maximum_selected_particles']
        validate_native_stream_comparison_request({'request':{'schema':'native-stream-paired-comparison-request-v1',
            **request['comparison_limits'],'reference':{},'intervention':{},'host_reflex':{**limits,'reference':{},'intervention':{}}}})
    return request


def native_counterfactual_campaign_input_bindings(parameters):
    from pathlib import Path
    from .foundation import file_sha256
    from .sidm_analysis import _load_product,_stage_native_bindings
    request=validate_native_counterfactual_campaign_request(parameters)
    if request['schema']=='native-late-counterfactual-campaign-request-v1':return native_late_campaign_input_bindings(parameters)
    base=Path(parameters['request_base']).resolve();memory=request['resources']['memory_gib']*1024**3;sources={}
    for reference in [request['counterfactual_product'],request['live_host_membership'],*request['component_memberships'].values()]:
        _,_,binding=_load_product(reference,base,memory/4);sources.update(binding)
    for reference in (request['parent_native_stage_receipt'],request['original_ic'],request['run_request']['mpi_launcher']):
        path=(base/reference['path']).resolve()
        if file_sha256(path)!=reference['sha256']:raise ValueError('Automatic native campaign input changed')
        sources[str(path)]=reference['sha256']
    sources.update(_stage_native_bindings({'request':{'memory_gib':request['resources']['memory_gib']},
        'native_stage_receipt':str((base/request['parent_native_stage_receipt']['path']).resolve()),
        'native_run_directory':str((base/request['parent_native_run_directory']).resolve())}))
    return sources


def native_counterfactual_campaign_runner(parameters,context):
    """Publish namespaced native children through the existing follow-up budget.

    No nested workflow or hidden native command executes in this planner. The
    workflow engine preflights and executes admitted children normally, and
    retains budget exhaustion as an incomplete requirement.
    """
    import copy,json
    from pathlib import Path
    from .foundation import atomic_json,file_sha256
    from .workflow import safe_id
    request=copy.deepcopy(parameters['request']);allocation=request['resources']
    for name in ('mpi_ranks','omp_threads'):
        if allocation[name]!=context.resources[name]:raise ValueError('Automatic native campaign changes allocated '+name)
    for name in ('memory_gib','disk_gib','maximum_stage_count'):
        if allocation[name]>context.resources[name]:raise MemoryError('Automatic native campaign exceeds allocated '+name)
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Automatic native campaign escapes its owned generation')
    design=compile_native_counterfactual_campaign(request,destination,request_base=parameters['request_base'])
    workflow=json.loads(Path(design['workflow']).read_text());stages=workflow['stages']
    names={row['stage_id']:safe_id(context.stage_id+'__'+row['stage_id']) for row in stages}
    def rewrite(value):
        if isinstance(value,dict):
            result={key:rewrite(item) for key,item in value.items()}
            if 'stage' in result and result['stage'] in names:result['stage']=names[result['stage']]
            return result
        if isinstance(value,list):return [rewrite(item) for item in value]
        return value
    children=[]
    for original in stages:
        row=rewrite(copy.deepcopy(original));row['stage_id']=names[original['stage_id']]
        row['depends_on']=[names[value] for value in original.get('depends_on',[])]
        row['outputs']={safe_id(context.stage_id+'__'+key):value for key,value in original['outputs'].items()}
        children.append(row)
    result_ids=[names[value] for value in workflow['requested_products']]
    report={**design,'schema':'native-live-counterfactual-followup-plan-v1','parent_stage':context.stage_id,
        'planned_native_result_stages':result_ids,'followup_stage_count':len(children),
        'execution_status':'PLANNED_NOT_NATIVE_EXECUTED','followup_budget_policy':'inherited workflow child/depth/stage/resource limits; no nested bypass',
        'science_acceptance':'NOT_ESTABLISHED'}
    for mapping in report['mapped_native_branches']:
        for key in ('comparison_product','native_result_product'):mapping[key]=names[mapping[key]]
    atomic_json(destination/'followup-plan.json',report);atomic_json(destination/'followups.json',{'followups':children})
    return {'followups':children,'measurements':{'native_counterfactual_plan':str(destination/'followup-plan.json'),
        'native_counterfactual_plan_sha256':file_sha256(destination/'followup-plan.json'),'planned_result_stages':result_ids},
        'scientific_exclusions':[{'scope':'native interventions','reason':'Initial and source-free late collisionless branches are coded; measurements require actual returned native products. Hydro/BH/SIDM-safe state changes and independent causal accuracy remain.'}]}


def validate_native_counterfactual_result_request(parameters):
    request=parameters['request']
    keys={'schema','comparison','run_report','run_stage_receipt','host_membership','memory_gib','maximum_census_particles','minimum_host_particles'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-counterfactual-result-request-v1':
        raise ValueError('Complete actual native intervention/comparison/host-evidence result request required')
    for name in ('maximum_census_particles','minimum_host_particles'):
        if type(request[name])is not int or request[name]<1:raise ValueError('Exact positive live-host census bounds required')
    positive(request['memory_gib'],'native counterfactual result memory')
    return request


def native_counterfactual_result_input_bindings(parameters):
    from pathlib import Path
    from .foundation import file_sha256
    from .sidm_analysis import _load_product
    request=validate_native_counterfactual_result_request(parameters);base=Path(parameters['request_base']).resolve();sources={}
    for name in ('comparison','host_membership'):
        _,_,binding=_load_product(request[name],base,request['memory_gib']*1024**3/4);sources.update(binding)
    for name in ('run_report','run_stage_receipt'):
        reference=request[name];path=(base/reference['path']).resolve()
        if file_sha256(path)!=reference['sha256']:raise ValueError('Native result changes its actual '+name)
        sources[str(path)]=reference['sha256']
    return sources


def publish_native_counterfactual_result(parameters,destination):
    """Attach actual IC/native invocation lineage and retained live-host IDs.

    This establishes the explicitly executed initial-epoch removal experiment,
    not arbitrary late-time interventions, present bound-host membership,
    force-resolution convergence or global causal accuracy.
    """
    import json
    from pathlib import Path
    from .foundation import file_sha256
    from core.native import verify_invocation_inputs
    from .workflow import verify_completed_receipt
    from .sidm_analysis import _load_product
    from .observation_geometry import read_native_gravitating_particles
    from .data import exact_uint64_ids,export_numeric_audit
    request=validate_native_counterfactual_result_request(parameters);base=Path(parameters['request_base']).resolve();memory=request['memory_gib']*1024**3
    sources=native_counterfactual_result_input_bindings(parameters)
    comparison,arrays,binding=_load_product(request['comparison'],base,memory/4);sources.update(binding)
    if comparison.get('schema')!='measured-native-stream-paired-comparison-v1':raise ValueError('Native intervention result requires actual paired native stream measurements')
    receipt_path=(base/request['run_stage_receipt']['path']).resolve();receipt=verify_completed_receipt(receipt_path);generation=Path(receipt['generation']).resolve()
    owned={str((generation/name).resolve()):value for name,value in receipt['files_sha256'].items()}
    report_path=(base/request['run_report']['path']).resolve()
    if owned.get(str(report_path))!=request['run_report']['sha256'] or report_path.stat().st_size*16>memory/4:
        raise ValueError('Counterfactual run report is outside its actual completed evolution generation')
    run=json.loads(report_path.read_text())
    if run.get('schema')!='native-counterfactual-run-v1' or run.get('execution_status')!='COMPLETE':raise ValueError('Native counterfactual was not completed by the native runner')
    native=Path(run['native_run_directory']).resolve()
    if not native.is_relative_to(generation):raise ValueError('Counterfactual native directory escapes its completed generation')
    invocation=run['invocation_receipt'];invocation_path=Path(invocation['path']).resolve()
    if owned.get(str(invocation_path))!=invocation['sha256'] or file_sha256(invocation_path)!=invocation['sha256']:
        raise ValueError('Native counterfactual invocation is not owned by the completed generation')
    executed=json.loads(invocation_path.read_text());verify_invocation_inputs(executed)
    if executed.get('status')!='RETURNED' or executed.get('returncode')!=0:raise ValueError('Native result cannot inherit an unreturned/failed invocation')
    sources.update(executed['input_hashes']);sources[str(invocation_path)]=invocation['sha256']
    lineage_ref=run['intervention_ic_lineage'];lineage_path=Path(lineage_ref['path']).resolve()
    if file_sha256(lineage_path)!=lineage_ref['sha256']:raise ValueError('Executed native intervention lineage changed')
    lineage=json.loads(lineage_path.read_text());removed=exact_uint64_ids(lineage['removed_dm_ids']);stream=exact_uint64_ids(lineage['unchanged_stream_ids'])
    if lineage.get('schema')!='native-counterfactual-ic-lineage-v1' or lineage['request']['operation'] not in {'remove_dm_particles','scale_dm_particle_masses'} or not np.array_equal(np.sort(stream),arrays['particle_ids']):
        raise ValueError('Executed intervention does not preserve the measured paired stream selection')
    affected=exact_uint64_ids(lineage.get('affected_dm_ids',lineage['removed_dm_ids']))
    binary=native/'input/Gadget4';param=native/'input/param.txt';initial=native/'input/counterfactual.hdf5'
    from .evolution import parameter_values,gadget_ic_prefix
    command_paths=[(Path(value) if Path(value).is_absolute() else Path(executed['cwd'])/value).resolve() for value in executed['command']]
    if (binary.resolve() not in command_paths or param.resolve() not in command_paths
            or executed['input_hashes'].get(str(binary))!=run['original_engine_sha256']
            or executed['input_hashes'].get(str(initial))!=lineage['native_ic']['sha256']):
        raise ValueError('Completed invocation did not use the exact original engine and materialized changed IC')
    if parameter_values(param)['InitCondFile']!=str(gadget_ic_prefix(initial)) or executed['command'][-1]!='0':
        raise ValueError('Native result is not the declared fresh changed-IC execution')
    original,rv,reference_sources=_load_product(comparison['request']['reference'],comparison['request_base'],memory/4)
    changed,cv,changed_sources=_load_product(comparison['request']['intervention'],comparison['request_base'],memory/4)
    if (Path(changed['native_run_directory']).resolve()!=native or Path(original['native_run_directory']).resolve()!=Path(run['parent_run']).resolve()
            or changed['native_stage_receipt']['sha256']!=request['run_stage_receipt']['sha256']):
        raise ValueError('Paired stream histories refer to different actual native branch/parent generations')
    if any(owned.get(path)!=checksum and receipt['source_hashes'].get(path)!=checksum and receipt['input_hashes'].get(path)!=checksum for path,checksum in changed.get('source_hashes',{}).items() if str(native) in path):
        raise ValueError('Paired intervention trajectory is not drawn from this actual native branch')
    sources.update(reference_sources);sources.update(changed_sources);sources[str(lineage_path)]=lineage_ref['sha256']
    host,host_arrays,host_sources=_load_product(request['host_membership'],base,memory/4);sources.update(host_sources)
    if host.get('schema')!='native-halo-membership-measurement-v1' or host['request'].get('species')!=[1] or host['request']['particle_namespace']!=comparison['namespace']:
        raise ValueError('Live-host evidence requires the same-namespace measured parent DM population')
    if not host.get('native_particle_source_hashes') or any(run['source_hashes'].get(path)!=checksum for path,checksum in host['native_particle_source_hashes'].items()):
        raise ValueError('Live host is not measured from the actual reference parent evolution')
    retained=np.setdiff1d(exact_uint64_ids(host_arrays['member_particle_ids']),affected)
    if 'host_reflex' in comparison:
        selected_host=exact_uint64_ids(arrays['host_particle_ids'],allow_empty=False)
        if np.setdiff1d(selected_host,retained).size:raise ValueError('Reflex reference cohort is outside the unchanged source-bound host membership')
    if len(retained)<request['minimum_host_particles']:raise ValueError('Intervention leaves inadequate declared live host-particle support')
    counts=[]
    for root,parent_report,expected_sources in ((Path(run['parent_run']).resolve(),original,run['source_hashes']),(native,changed,owned)):
        branch_counts=[]
        for snapshot in parent_report['request']['snapshots']:
            population,particles=read_native_gravitating_particles(root,snapshot,species=[1],maximum_particles=request['maximum_census_particles'],
                memory_gib=request['memory_gib']/4,namespace=comparison['namespace'])
            if any(expected_sources.get(path)!=checksum for path,checksum in population['source_hashes'].items()):raise ValueError('Live-host census is outside the actual paired native histories')
            sources.update(population['source_hashes']);present=np.intersect1d(retained,particles['particle_ids'])
            if len(present)!=len(retained):raise ValueError('Native paired evolution loses an unchanged live-host DM identity')
            if root==native and np.intersect1d(removed,particles['particle_ids']).size:raise ValueError('Native intervention still contains a declared removed DM identity')
            branch_counts.append(len(present))
        counts.append(branch_counts)
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={**comparison,'schema':'measured-native-live-counterfactual-result-v1','request':request,'paired_comparison':request['comparison'],
        'request_base':str(base),
        'executed_run':request['run_report'],'executed_ic_lineage':lineage_ref,'host_membership':request['host_membership'],
        'causal_intervention_status':'SOURCE_BOUND_INITIAL_IC_DM_INTERVENTION_EXECUTED_WITH_RETAINED_LIVE_HOST_PARTICLES',
        'physical_intervention':lineage['physical_operation'],'full_system_initial_state':lineage['full_system_initial_state'],
        'host_definition':'unchanged traced original DM membership IDs after excluding explicitly affected matter; not current bound-host membership',
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    if lineage['request']['schema']=='native-counterfactual-ic-request-v3':
        if run.get('intervention_epoch')!=lineage.get('intervention_epoch'):raise ValueError('Late counterfactual result changes its executed intervention epoch')
        report.update(intervention_epoch=lineage['intervention_epoch'],native_state_policy=lineage['native_state_policy'],
            causal_intervention_status='SOURCE_BOUND_LATE_COLLISIONLESS_STATE_INTERVENTION; FRESH_REINITIALIZATION_EFFECT_REQUIRES_MATCHED_UNCHANGED_STATE_CONTROL',
            inherited_integrator_continuation='NOT_ESTABLISHED')
    arrays.update(retained_host_particle_ids=retained,reference_host_id_counts=np.asarray(counts[0],np.uint64),intervention_host_id_counts=np.asarray(counts[1],np.uint64))
    specs={name:{'units':'identity' if name.endswith('particle_ids') else 'dimensionless' if name.endswith('counts') else 'Myr' if 'times' in name else 'Msun' if 'msun' in name else 'km / s' if 'kms' in name else 'proper kpc',
        'quantity':'source-bound native intervention '+name} for name in arrays}
    for name in ('particle_ids','retained_host_particle_ids'):specs[name]['identity_namespace']=comparison['namespace']
    if 'host_reflex' in comparison:
        specs['host_particle_ids']['identity_namespace']=comparison['host_reflex']['host_namespace']
        for name in ('reference_host_velocity_covariance_kms2','intervention_host_velocity_covariance_kms2'):specs[name]['units']='(km / s)^2'
    return export_numeric_audit(destination,report,arrays,specs,algorithm=source,source_hashes=sources,
        context={'epoch':{'reference_epochs':comparison['reference_epochs'],'intervention_epochs':comparison['intervention_epochs']},'coordinates':{'frame':comparison['coordinate_definition']}})


def native_counterfactual_result_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native intervention result exceeds allocated memory')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Native intervention result escapes its owned generation')
    return publish_native_counterfactual_result(parameters,destination)


def validate_native_stream_comparison_request(parameters):
    request=parameters['request']
    keys={'schema','reference','intervention','memory_gib','maximum_snapshots','maximum_selected_particles',
        'time_absolute_tolerance_myr','maximum_periodic_difference_box_fraction'}
    if isinstance(request,dict) and 'host_reflex' in request:keys.add('host_reflex')
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-stream-paired-comparison-request-v1':
        raise ValueError('Complete actual native paired stream comparison request required')
    for name in ('maximum_snapshots','maximum_selected_particles'):
        if type(request[name])is not int or request[name]<1:raise ValueError('Exact positive paired stream bounds required')
    positive(request['memory_gib'],'paired stream memory')
    tolerance=request['time_absolute_tolerance_myr']
    if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('Finite nonnegative proper epoch matching tolerance required')
    fraction=positive(request['maximum_periodic_difference_box_fraction'],'periodic displacement fraction')
    if fraction>=.5:raise ValueError('Paired periodic differences must remain strictly below half the box')
    if 'host_reflex' in request:
        host=request['host_reflex']
        if not isinstance(host,dict) or set(host)!={'reference','intervention','anchor_particle_id','maximum_host_particles','minimum_host_particles',
                'maximum_host_radius_box_fraction','maximum_stream_radius_box_fraction'}:
            raise ValueError('Complete source-bound host reflex selection and spatial support limits required')
        for name in ('anchor_particle_id','maximum_host_particles','minimum_host_particles'):
            if type(host[name])is not int or not 1<=host[name]<2**64:raise ValueError('Exact positive host identity/census bound required')
        if not 2<=host['minimum_host_particles']<=host['maximum_host_particles']:raise ValueError('Host reflex needs a bounded supported particle cohort')
        for name in ('maximum_host_radius_box_fraction','maximum_stream_radius_box_fraction'):
            if positive(host[name],name)>=.5:raise ValueError('Host/stream local periodic support must remain below half the box')
        if request['maximum_snapshots']*(host['maximum_host_particles']*512+request['maximum_selected_particles']*1024)>request['memory_gib']*1024**3:
            raise MemoryError('Combined stream/host reflex coordinate workspace exceeds allocation')
    return request


def native_stream_comparison_input_bindings(parameters):
    from .sidm_analysis import _load_product
    request=validate_native_stream_comparison_request(parameters);sources={}
    for name in ('reference','intervention'):
        _,_,binding=_load_product(request[name],parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
    if 'host_reflex' in request:
        for name in ('reference','intervention'):
            _,_,binding=_load_product(request['host_reflex'][name],parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
    return sources


def _native_host_reflex_moments(report,values,stream_report,stream_values,request):
    """Actual selected host COM at saved epochs in an explicit local chart."""
    from .data import exact_uint64_ids
    limits=request['host_reflex'];ids=exact_uint64_ids(values['particle_ids'],allow_empty=False)
    epochs=len(stream_values['times_myr']);count=len(ids)
    if report.get('schema')!='measured-native-stream-worldlines-v1' or report.get('execution_status')!='COMPLETE' or report.get('native_self_gravity_enabled')is not True:
        raise ValueError('Host reflex requires actual self-gravitating native selected-particle histories')
    if report['namespace']!=stream_report['namespace'] or report['periodic']!=stream_report['periodic'] or report['native_stage_receipt']['sha256']!=stream_report['native_stage_receipt']['sha256']:
        raise ValueError('Host and stream histories must share the actual native branch, namespace and boundary frame')
    if not limits['minimum_host_particles']<=count<=limits['maximum_host_particles'] or np.intersect1d(ids,stream_values['particle_ids']).size:
        raise ValueError('Host reference cohort must be adequately sampled and distinct from the selected stream')
    anchor=np.flatnonzero(ids==limits['anchor_particle_id'])
    if len(anchor)!=1:raise ValueError('Host reflex anchor is absent from the exact native host cohort')
    for name,shape in (('position_kpc',(epochs,count,3)),('velocity_kms',(epochs,count,3)),('mass_msun',(epochs,count)),
            ('times_myr',(epochs,)),('periodic_box_proper_kpc',(epochs,))):
        if values[name].shape!=shape or not np.isfinite(values[name]).all():raise ValueError('Host reflex has incomplete saved physical state: '+name)
    if np.any(values['mass_msun']<=0) or np.any(values['periodic_box_proper_kpc']<=0) or np.any(np.diff(values['times_myr'])<=0):
        raise ValueError('Host reflex masses/domain/time support must remain positive and ordered')
    if np.any(np.abs(values['times_myr']-stream_values['times_myr'])>request['time_absolute_tolerance_myr']) or not np.array_equal(values['periodic_box_proper_kpc'],stream_values['periodic_box_proper_kpc']):
        raise ValueError('Host reflex and stream lack the same actual physical saved epochs/domain')
    if report['epochs']!=stream_report['epochs']:raise ValueError('Host/stream physical epoch definitions differ within a native branch')
    origin=values['position_kpc'][:,int(anchor[0]),:];relative=values['position_kpc']-origin[:,None,:]
    box=values['periodic_box_proper_kpc'][:,None,None]
    if report['periodic']:relative-=box*np.rint(relative/box)
    radius=np.linalg.norm(relative,axis=2)
    if report['periodic'] and np.any(radius>limits['maximum_host_radius_box_fraction']*box[:,:,0]):
        raise ValueError('Host cohort leaves the declared anchor-centred local periodic chart')
    mass=values['mass_msun'];total=mass.sum(axis=1)
    if not np.isfinite(total).all() or np.any(total<=0):raise ArithmeticError('Host extensive mass overflows')
    position=origin+np.einsum('en,eni->ei',mass,relative)/total[:,None]
    if report['periodic']:position=np.remainder(position,box[:,0,:])
    velocity=np.einsum('en,eni->ei',mass,values['velocity_kms'])/total[:,None]
    centred=values['velocity_kms']-velocity[:,None,:]
    covariance=np.einsum('en,eni,enj->eij',mass,centred,centred)/total[:,None,None]
    if not all(np.isfinite(value).all() for value in (position,velocity,covariance)):raise ArithmeticError('Host reflex moments overflow')
    return {'ids':ids,'position_kpc':position,'velocity_kms':velocity,'mass_msun':total,'covariance_kms2':covariance,
        'maximum_anchor_radius_kpc':radius.max(axis=1)}


def publish_native_stream_comparison(parameters,destination):
    """Measure paired native trajectories without relabeling wrapped positions.

    Periodic displacements are minimum-image differences at each common saved
    epoch. They do not infer an unwrapped long-time winding history. Means use
    the original common stream mass weights, retaining later mass differences.
    """
    from pathlib import Path
    from .sidm_analysis import _load_product
    from .data import export_numeric_audit
    from .foundation import file_sha256
    request=validate_native_stream_comparison_request(parameters);memory=request['memory_gib']*1024**3;sources={};products=[]
    if request['reference']['sha256']==request['intervention']['sha256']:raise ValueError('A native branch cannot serve as its own counterfactual')
    for name in ('reference','intervention'):
        report,values,binding=_load_product(request[name],parameters['request_base'],memory/4);sources.update(binding)
        if report.get('schema')!='measured-native-stream-worldlines-v1' or report.get('execution_status')!='COMPLETE' or report.get('native_self_gravity_enabled')is not True:
            raise ValueError('Native paired controls require original measured trajectories from self-gravitating native evolution')
        products.append((report,values))
    left,lv=products[0];right,rv=products[1]
    if left['namespace']!=right['namespace'] or left['initial_state_sha256']!=right['initial_state_sha256'] or left['periodic']!=right['periodic']:
        raise ValueError('Native controls change the unchanged initial selected stream population/frame')
    if not np.array_equal(lv['particle_ids'],rv['particle_ids']):raise ValueError('Native controls require exact persistent stream identity pairing')
    epochs,count=lv['position_kpc'].shape[:2]
    if (lv['position_kpc'].shape!=(epochs,count,3) or lv['velocity_kms'].shape!=(epochs,count,3)
            or lv['mass_msun'].shape!=(epochs,count) or lv['times_myr'].shape!=(epochs,)
            or lv['periodic_box_proper_kpc'].shape!=(epochs,)):
        raise ValueError('Native comparison requires complete vector, mass and physical clock dimensions')
    if epochs>request['maximum_snapshots'] or count>request['maximum_selected_particles'] or epochs*count*1024>memory:
        raise MemoryError('Complete paired stream arrays exceed allocation')
    for name in ('position_kpc','velocity_kms','mass_msun','times_myr','periodic_box_proper_kpc'):
        if lv[name].shape!=rv[name].shape or not np.isfinite(lv[name]).all() or not np.isfinite(rv[name]).all():raise ValueError('Native paired coordinates have incomplete support: '+name)
    if np.any(np.abs(lv['times_myr']-rv['times_myr'])>request['time_absolute_tolerance_myr']):raise ValueError('Native controls do not share the requested actual proper epochs')
    if len(left['epochs'])!=epochs or len(right['epochs'])!=epochs:raise ValueError('Native comparison lacks every actual saved epoch contract')
    for first,last in zip(left['epochs'],right['epochs']):
        if first['canonical_time']!=last['canonical_time'] or first.get('actual_scale_factor')!=last.get('actual_scale_factor'):
            raise ValueError('Native paired controls change physical expansion epochs or clock representation')
    if not np.array_equal(lv['periodic_box_proper_kpc'],rv['periodic_box_proper_kpc']):raise ValueError('Native paired controls change the actual physical boundary box')
    delta_x=rv['position_kpc']-lv['position_kpc'];raw_delta=delta_x.copy();delta_v=rv['velocity_kms']-lv['velocity_kms']
    if left['periodic']:
        box=lv['periodic_box_proper_kpc'][:,None,None]
        delta_x-=box*np.rint(delta_x/box)
        if np.any(np.linalg.norm(delta_x,axis=2)>request['maximum_periodic_difference_box_fraction']*box[:,:,0]):
            raise ValueError('Paired minimum-image differences leave the declared local comparison domain')
    weights=lv['mass_msun'][0]
    if weights.shape!=(count,) or np.any(weights<=0) or not np.array_equal(weights,rv['mass_msun'][0]):raise ValueError('Paired stream initial gravitating masses differ')
    total=float(weights.sum())
    if not math.isfinite(total) or total<=0:raise ArithmeticError('Initial paired stream extensive mass overflows')
    mean_x=np.einsum('n,eni->ei',weights,delta_x)/total;mean_v=np.einsum('n,eni->ei',weights,delta_v)/total
    output={'particle_ids':lv['particle_ids'],'reference_times_myr':lv['times_myr'],'intervention_times_myr':rv['times_myr'],
        'position_change_kpc':delta_x,'raw_wrapped_position_change_kpc':raw_delta,'velocity_change_kms':delta_v,
        'mass_change_msun':rv['mass_msun']-lv['mass_msun'],'initial_mass_msun':weights,
        'mean_position_change_kpc':mean_x,'mean_velocity_change_kms':mean_v,
        'rms_position_change_kpc':np.sqrt(np.einsum('n,en->e',weights,np.sum(delta_x*delta_x,axis=2))/total),
        'rms_velocity_change_kms':np.sqrt(np.einsum('n,en->e',weights,np.sum(delta_v*delta_v,axis=2))/total),
        'periodic_box_proper_kpc':lv['periodic_box_proper_kpc']}
    host_definition=None
    if 'host_reflex' in request:
        hosts=[];host_reports=[];host_values=[]
        for name,stream_report,stream_values in (('reference',left,lv),('intervention',right,rv)):
            host_report,host_arrays,binding=_load_product(request['host_reflex'][name],parameters['request_base'],memory/4);sources.update(binding)
            hosts.append(_native_host_reflex_moments(host_report,host_arrays,stream_report,stream_values,request))
            host_reports.append(host_report);host_values.append(host_arrays)
        first,last=hosts
        if not np.array_equal(first['ids'],last['ids']) or host_reports[0]['initial_state_sha256']!=host_reports[1]['initial_state_sha256']:
            raise ValueError('Host reflex pairing changes the unchanged original selected host cohort')
        if not np.array_equal(host_values[0]['mass_msun'][0],host_values[1]['mass_msun'][0]):raise ValueError('Host reflex pairing changes initial host masses')
        host_delta=last['position_kpc']-first['position_kpc'];local_positions=[]
        for stream_values,host in ((lv,first),(rv,last)):
            local=stream_values['position_kpc']-host['position_kpc'][:,None,:]
            if left['periodic']:
                box=lv['periodic_box_proper_kpc'][:,None,None];local-=box*np.rint(local/box)
                if np.any(np.linalg.norm(local,axis=2)>request['host_reflex']['maximum_stream_radius_box_fraction']*box[:,:,0]):
                    raise ValueError('Selected stream leaves the declared local host-relative periodic chart')
            local_positions.append(local)
        centred_x=local_positions[1]-local_positions[0]
        if left['periodic']:
            box=lv['periodic_box_proper_kpc'][:,None,None];host_delta-=box[:,0,:]*np.rint(host_delta/box[:,0,:])
            centred_x-=box*np.rint(centred_x/box)
            for value in (np.linalg.norm(host_delta,axis=1),np.linalg.norm(centred_x,axis=2).max(axis=1)):
                if np.any(value>request['maximum_periodic_difference_box_fraction']*box[:,0,0]):
                    raise ValueError('Host reflex or host-relative stream difference leaves the frozen local comparison domain')
        host_delta_v=last['velocity_kms']-first['velocity_kms'];centred_v=delta_v-host_delta_v[:,None,:]
        output.update(host_particle_ids=first['ids'],reference_host_position_kpc=first['position_kpc'],intervention_host_position_kpc=last['position_kpc'],
            reference_host_velocity_kms=first['velocity_kms'],intervention_host_velocity_kms=last['velocity_kms'],
            reference_host_mass_msun=first['mass_msun'],intervention_host_mass_msun=last['mass_msun'],
            reference_host_velocity_covariance_kms2=first['covariance_kms2'],intervention_host_velocity_covariance_kms2=last['covariance_kms2'],
            reference_host_anchor_radius_kpc=first['maximum_anchor_radius_kpc'],intervention_host_anchor_radius_kpc=last['maximum_anchor_radius_kpc'],
            host_position_change_kpc=host_delta,host_velocity_change_kms=host_delta_v,
            host_relative_position_change_kpc=centred_x,host_relative_velocity_change_kms=centred_v,
            mean_host_relative_position_change_kpc=np.einsum('n,eni->ei',weights,centred_x)/total,
            mean_host_relative_velocity_change_kms=np.einsum('n,eni->ei',weights,centred_v)/total,
            rms_host_relative_position_change_kpc=np.sqrt(np.einsum('n,en->e',weights,np.sum(centred_x*centred_x,axis=2))/total),
            rms_host_relative_velocity_change_kms=np.sqrt(np.einsum('n,en->e',weights,np.sum(centred_v*centred_v,axis=2))/total))
        host_definition={'schema':'measured-native-selected-host-reflex-v1','anchor_particle_id_decimal':str(request['host_reflex']['anchor_particle_id']),
            'selected_host_count':len(first['ids']),'host_namespace':host_reports[0]['namespace'],
            'host_initial_state_sha256':host_reports[0]['initial_state_sha256'],
            'centre_definition':'current-mass weighted selected host COM in the anchor-centred local chart; wrapped in periodic runs',
            'velocity_definition':'current-mass weighted selected host peculiar velocity; subtract each branch host velocity from its stream',
            'frame_definition':'translated host-relative saved coordinates; Cartesian axes retained, no rotating-frame or continuous-path inference',
            'scope':'selected host cohort reflex only; not automatic whole-halo membership, wake/deformation attribution or independent causal accuracy'}
    if any(not np.isfinite(value).all() for value in output.values()):raise ArithmeticError('Native paired stream accumulation overflows')
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={'schema':'measured-native-stream-paired-comparison-v1','execution_status':'COMPLETE','request':request,
        'request_base':str(Path(parameters['request_base']).resolve()),'namespace':left['namespace'],'initial_state_sha256':left['initial_state_sha256'],
        'reference_epochs':left['epochs'],'intervention_epochs':right['epochs'],'periodic':left['periodic'],
        'coordinate_definition':'same-epoch minimum-image native proper position differences' if left['periodic'] else 'native proper inertial position differences',
        'weight_definition':'unchanged original selected stream masses; later branch mass differences retained separately',
        'causal_intervention_status':'REQUIRES_NATIVE_BRANCH_LINEAGE_AND_HOST_POPULATION_EVIDENCE',
        'sampling_uncertainty':'NOT_ESTIMATED; independent native realizations and measured resolution/model errors remain required',
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    if host_definition is not None:report['host_reflex']=host_definition
    units={name:'identity' if name=='particle_ids' else 'Myr' if 'times' in name else 'Msun' if 'msun' in name else 'km / s' if 'kms' in name else 'proper kpc' for name in output}
    specs={name:{'units':units[name],'quantity':'paired original native stream '+name} for name in output};specs['particle_ids']['identity_namespace']=left['namespace']
    if host_definition is not None:
        specs['host_particle_ids'].update(units='identity',identity_namespace=host_definition['host_namespace'])
        for name in ('reference_host_velocity_covariance_kms2','intervention_host_velocity_covariance_kms2'):specs[name]['units']='(km / s)^2'
    return export_numeric_audit(destination,report,output,specs,algorithm=source,source_hashes=sources,
        context={'epoch':{'reference_measured_epochs':left['epochs'],'intervention_measured_epochs':right['epochs']},'coordinates':{'frame':report['coordinate_definition']}})


def native_stream_comparison_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native comparison exceeds allocated memory')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Native comparison escapes its owned generation')
    return publish_native_stream_comparison(parameters,destination)


def native_stream_comparison_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Compare actual matched self-gravitating native stream branches')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.parameters.resolve();parameters=json.loads(source.read_text());parameters.setdefault('request_base',str(source.parent))
    publish_native_stream_comparison(parameters,args.output);return 0


def validate_native_late_campaign_request(parameters):
    request=parameters['request'];keys={'schema','parent_native_stage_receipt','parent_native_run_directory','state_request',
        'stream_request','run_request','comparison_limits','threeway_limits','resources','followup_limits'}
    if isinstance(request,dict) and 'host_reflex_request' in request:keys.add('host_reflex_request')
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-late-counterfactual-campaign-request-v1':
        raise ValueError('Complete saved-state late native campaign specification required')
    state=request['state_request'];validate_native_counterfactual_ic_request({'request':state})
    if state['schema']!='native-counterfactual-ic-request-v3' or state['operation']=='retain_collisionless_state':
        raise ValueError('Late campaign requires one explicit changed-state intervention; unchanged control is generated automatically')
    validate_native_stream_worldline_request({'request':request['stream_request']})
    validate_native_counterfactual_run_request({'request':request['run_request']})
    stream=request['stream_request']
    if stream['particle_namespace']!=state['particle_namespace'] or stream['species']!=state['stream_species'] or sorted(stream['particle_ids'])!=sorted(state['stream_particle_ids']):
        raise ValueError('Late campaign changes its unchanged original selected stream definition')
    comparison={'schema':'native-stream-paired-comparison-request-v1','reference':{},'intervention':{},**request['comparison_limits']}
    validate_native_stream_comparison_request({'request':comparison})
    if set(request['comparison_limits'])!=set(comparison)-{'schema','reference','intervention'}:
        raise ValueError('Freeze only complete late pair comparison limits')
    three={'schema':'native-late-threeway-comparison-request-v1',**request['threeway_limits'],
        **{name:{'path':'pending','sha256':'0'*64} for name in ('restart_comparison','intervention_comparison','total_comparison','control_run','intervention_run')}}
    validate_native_late_threeway_request({'request':three})
    if set(request['threeway_limits'])!=set(three)-{'schema','restart_comparison','intervention_comparison','total_comparison','control_run','intervention_run'}:
        raise ValueError('Freeze only complete three-way measurement limits')
    if 'host_reflex_request' in request:
        host=request['host_reflex_request']
        if not isinstance(host,dict) or set(host)!={'worldline_request','anchor_particle_id','minimum_host_particles',
                'maximum_host_radius_box_fraction','maximum_stream_radius_box_fraction'}:raise ValueError('Complete late host-reflex request required')
        validate_native_stream_worldline_request({'request':host['worldline_request']})
        h=host['worldline_request']
        if h['particle_namespace']!=stream['particle_namespace'] or h['species']!=1 or set(h['particle_ids'])&set(stream['particle_ids']):
            raise ValueError('Late host frame requires a distinct original same-namespace DM cohort')
        affected=state.get('removed_dm_particle_ids',state.get('affected_dm_particle_ids',[]))
        if set(h['particle_ids'])&set(affected):raise ValueError('Late intervention changes its selected unchanged host reference')
        if len(h['particle_ids'])<host['minimum_host_particles'] or host['anchor_particle_id'] not in h['particle_ids']:
            raise ValueError('Late host frame lacks its exact anchor or selected particle support')
        limits={key:value for key,value in host.items() if key!='worldline_request'};limits['maximum_host_particles']=h['maximum_selected_particles']
        comparison['host_reflex']={**limits,'reference':{},'intervention':{}};validate_native_stream_comparison_request({'request':comparison})
    elif request['threeway_limits']['frame']=='host_relative':raise ValueError('Host-relative late measurement requires its actual host histories')
    resources=request['resources']
    for definition in (state,stream,request['run_request'],request['comparison_limits'],request['threeway_limits']):
        if definition['memory_gib']>resources['memory_gib']:raise MemoryError('Late campaign stage exceeds its frozen memory allocation')
    if 'host_reflex_request' in request and request['host_reflex_request']['worldline_request']['memory_gib']>resources['memory_gib']:
        raise MemoryError('Late host history exceeds campaign allocation')
    for name in ('mpi_ranks','omp_threads'):
        if request['run_request'][name]!=resources[name]:raise ValueError('Late campaign changes its frozen native decomposition')
    if max(state['disk_gib'],request['run_request']['disk_gib'])>resources['disk_gib']:raise MemoryError('Late native stage exceeds campaign disk allocation')
    saved=max(stream['maximum_snapshots'],request['comparison_limits']['maximum_snapshots'],request['threeway_limits']['maximum_snapshots'])
    selected=max(stream['maximum_selected_particles'],request['comparison_limits']['maximum_selected_particles'],request['threeway_limits']['maximum_selected_particles'])
    measured_bytes=saved*selected*8192
    if 'host_reflex_request' in request:
        measured_bytes+=3*request['host_reflex_request']['worldline_request']['maximum_snapshots']*request['host_reflex_request']['worldline_request']['maximum_selected_particles']*1024
    persistent_gib=2*state['disk_gib']+2*request['run_request']['disk_gib']+measured_bytes/1024**3
    if persistent_gib>resources['disk_gib']:raise MemoryError('Complete retained late state/run/measurement family exceeds campaign disk allocation')
    stages=14 if 'host_reflex_request' in request else 11
    if resources['maximum_stage_count']<stages or request['followup_limits']['maximum_children']<stages or request['followup_limits']['maximum_depth']<5:
        raise MemoryError('Late campaign needs its complete state/run/history/pair/measurement stage and follow-up budgets')
    return request


def native_late_campaign_input_bindings(parameters):
    from pathlib import Path
    from .foundation import file_sha256
    request=validate_native_late_campaign_request(parameters);base=Path(parameters['request_base']).resolve()
    receipt=(base/request['parent_native_stage_receipt']['path']).resolve()
    if file_sha256(receipt)!=request['parent_native_stage_receipt']['sha256']:raise ValueError('Late parent receipt identity changed')
    state=dict(request['state_request']);state['original_ic']={'path':str((base/state['original_ic']['path']).resolve()),'sha256':state['original_ic']['sha256']}
    sources=native_counterfactual_ic_input_bindings({'request':state,'native_stage_receipt':str(receipt),
        'native_run_directory':str((base/request['parent_native_run_directory']).resolve())})
    launcher=(base/request['run_request']['mpi_launcher']['path']).resolve()
    if file_sha256(launcher)!=request['run_request']['mpi_launcher']['sha256']:raise ValueError('Late campaign native launcher changed')
    sources[str(launcher)]=file_sha256(launcher);return sources


def compile_native_late_counterfactual_campaign(request,destination,*,request_base):
    """Build the full parent/control/changed saved-state experiment graph."""
    import copy,json
    from pathlib import Path
    from .foundation import atomic_json,file_sha256
    from .workflow import compile_requested_product_dag,default_registry
    request=validate_native_late_campaign_request({'request':request});base=Path(request_base).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Keep previous late campaign designs')
    sources=native_late_campaign_input_bindings({'request':request,'request_base':str(base)})
    parent=(base/request['parent_native_run_directory']).resolve();receipt=str((base/request['parent_native_stage_receipt']['path']).resolve())
    state=copy.deepcopy(request['state_request']);state['original_ic']['path']=str((base/state['original_ic']['path']).resolve())
    run=copy.deepcopy(request['run_request']);run['mpi_launcher']['path']=str((base/run['mpi_launcher']['path']).resolve())
    stages=[]
    def artifact(stage,path,field=None):
        ref={'stage':stage,'artifact':path}
        if field is not None:ref.update(load='json',field=field)
        return ref
    def product(stage,path):return {'path':artifact(stage,path),'sha256':{**artifact(stage,path),'load':'sha256'}}
    def add(identity,kind,parameters,output,dependencies=()):
        stages.append({'stage_id':identity,'kind':kind,'parameters':parameters,'depends_on':list(dependencies),'outputs':{identity:output}})
    for role in ('control','changed'):
        current=copy.deepcopy(state)
        if role=='control':
            current['operation']='retain_collisionless_state'
            for name in ('removed_dm_particle_ids','affected_dm_particle_ids','mass_multiplier'):current.pop(name,None)
        add(role+'_ic','native_counterfactual_ic',{'request':current,'native_stage_receipt':receipt,'native_run_directory':str(parent),
            'destination':'{generation}/state'},'state/counterfactual-ic.json')
        native=copy.deepcopy(run);native['intervention_ic']=product(role+'_ic','state/counterfactual-ic.json')
        add(role+'_run','native_counterfactual_run',{'request':native,'request_base':str(base),'native_stage_receipt':receipt,
            'native_run_directory':str(parent),'destination':'{generation}/run'},'run/counterfactual-run.json',[role+'_ic'])
    for role in ('parent','control','changed'):
        branch='control' if role=='parent' else role
        history=copy.deepcopy(request['stream_request']);history['snapshots']=artifact(branch+'_run','run/counterfactual-run.json',
            ['parent_snapshot_labels'] if role=='parent' else ['branch_snapshot_labels'])
        native_receipt=receipt if role=='parent' else artifact(role+'_run','stage-receipt.json')
        native_root=str(parent) if role=='parent' else artifact(role+'_run','run/counterfactual-run.json',['native_run_directory'])
        add(role+'_history','native_stream_worldlines',{'request':history,'native_stage_receipt':native_receipt,
            'native_run_directory':native_root,'destination':'{generation}/worldlines.json'},'worldlines.json',[branch+'_run'])
        if 'host_reflex_request' in request:
            host=copy.deepcopy(request['host_reflex_request']['worldline_request']);host['snapshots']=copy.deepcopy(history['snapshots'])
            add(role+'_host','native_stream_worldlines',{'request':host,'native_stage_receipt':native_receipt,'native_run_directory':native_root,
                'destination':'{generation}/host.json'},'host.json',[branch+'_run'])
    pair_ids={}
    for label,left,right in (('restart','parent','control'),('intervention','control','changed'),('total','parent','changed')):
        comparison={'schema':'native-stream-paired-comparison-request-v1',**copy.deepcopy(request['comparison_limits']),
            'reference':product(left+'_history','worldlines.json'),'intervention':product(right+'_history','worldlines.json')}
        dependencies=[left+'_history',right+'_history']
        if 'host_reflex_request' in request:
            host=request['host_reflex_request'];limits={key:copy.deepcopy(value) for key,value in host.items() if key!='worldline_request'}
            limits['maximum_host_particles']=host['worldline_request']['maximum_selected_particles']
            comparison['host_reflex']={**limits,'reference':product(left+'_host','host.json'),'intervention':product(right+'_host','host.json')}
            dependencies.extend([left+'_host',right+'_host'])
        identity=label+'_comparison';pair_ids[label]=identity
        add(identity,'native_stream_comparison',{'request':comparison,'request_base':str(base),'destination':'{generation}/comparison.json'},'comparison.json',dependencies)
    measurement={'schema':'native-late-threeway-comparison-request-v1',**copy.deepcopy(request['threeway_limits']),
        **{label+'_comparison':product(identity,'comparison.json') for label,identity in pair_ids.items()},
        'control_run':product('control_run','run/counterfactual-run.json'),'intervention_run':product('changed_run','run/counterfactual-run.json')}
    add('late_threeway','native_late_threeway',{'request':measurement,'request_base':str(base),'destination':'{generation}/threeway.json'},
        'threeway.json',[*pair_ids.values(),'control_run','changed_run'])
    workflow={'schema':'research-workflow-v1','resources':copy.deepcopy(request['resources']),'followup_limits':copy.deepcopy(request['followup_limits']),
        'physics_spec':{'cosmology_model':'lcdm','physics':'dm','dark_matter':{'linear_model':'cdm','interactions':{'kind':'collisionless'}}},
        'stages':stages,'requested_products':['late_threeway']}
    # Preserve the actual parent background/linear physics in capability intake.
    import yaml
    cfg=yaml.safe_load((parent/'config.yaml').read_text());background=cfg['cosmology']['model']
    workflow['physics_spec']['cosmology_model']='lcdm' if background=='wdm' else background
    workflow['physics_spec']['dark_matter']['linear_model']='wdm' if background=='wdm' or cfg.get('wdm',{}).get('enabled',False) else 'cdm'
    compile_requested_product_dag(workflow,default_registry(),request_base=base)
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    destination.mkdir(parents=True);atomic_json(destination/'workflow.json',workflow)
    report={'schema':'native-late-counterfactual-campaign-design-v1','request':request,'workflow':str(destination/'workflow.json'),
        'workflow_sha256':file_sha256(destination/'workflow.json'),'stage_count':len(stages),'native_run_count':2,
        'mapped_native_branches':[{'branch':'late_changed','comparison_product':'total_comparison','native_result_product':'late_threeway'}],
        'source_hashes':sources,'execution_status':'NOT_EXECUTED','science_acceptance':'NOT_ESTABLISHED',
        'scope':'source-free single-file collisionless parent/control/intervention graph with measured restart budgets; broader domains remain'}
    atomic_json(destination/'campaign-design.json',report);return report


def validate_native_late_threeway_request(parameters):
    request=parameters['request']
    keys={'schema','restart_comparison','intervention_comparison','total_comparison','control_run','intervention_run','frame',
        'memory_gib','maximum_snapshots','maximum_selected_particles','maximum_restart_position_error_kpc',
        'maximum_restart_velocity_error_kms','maximum_restart_signal_fraction','position_signal_floor_kpc','velocity_signal_floor_kms'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='native-late-threeway-comparison-request-v1':
        raise ValueError('Complete late parent/control/intervention comparison and frozen budgets required')
    if request['frame'] not in {'inertial','host_relative'}:raise ValueError('Declare the actual saved comparison frame')
    for name in ('maximum_snapshots','maximum_selected_particles'):
        if type(request[name])is not int or request[name]<(2 if name=='maximum_snapshots' else 1):raise ValueError('Bounded complete saved-epoch/particle support required')
    for name in ('memory_gib','position_signal_floor_kpc','velocity_signal_floor_kms'):positive(request[name],name)
    for name in ('maximum_restart_position_error_kpc','maximum_restart_velocity_error_kms','maximum_restart_signal_fraction'):
        value=request[name]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Finite nonnegative frozen restart budget required')
    if request['maximum_restart_signal_fraction']>=1:raise ValueError('Restart contamination budget must be smaller than the intervention signal scale')
    if request['maximum_snapshots']*request['maximum_selected_particles']*2048>request['memory_gib']*1024**3:
        raise MemoryError('Complete three-way saved-particle workspace exceeds allocation')
    for name in ('restart_comparison','intervention_comparison','total_comparison','control_run','intervention_run'):
        if not isinstance(request[name],dict) or set(request[name])!={'path','sha256'}:raise ValueError('Exact source-bound three-way product references required')
    return request


def _late_threeway_inputs(parameters):
    import json
    from pathlib import Path
    from .sidm_analysis import _load_product
    from .foundation import file_sha256
    from core.native import verify_invocation_inputs
    from .workflow import verify_completed_receipt
    from .evolution import parameter_values
    request=validate_native_late_threeway_request(parameters);base=Path(parameters['request_base']).resolve();memory=request['memory_gib']*1024**3
    sources={};comparisons={};runs={};definitions=[]
    for name in ('restart_comparison','intervention_comparison','total_comparison'):
        report,arrays,binding=_load_product(request[name],base,memory/6);sources.update(binding)
        if report.get('schema')!='measured-native-stream-paired-comparison-v1' or report.get('execution_status')!='COMPLETE':
            raise ValueError('Three-way measurement requires complete actual native comparisons')
        comparisons[name]=(report,arrays)
    restart=comparisons['restart_comparison'][0];intervention=comparisons['intervention_comparison'][0];total=comparisons['total_comparison'][0]
    for a,side_a,b,side_b in ((restart,'reference',total,'reference'),(restart,'intervention',intervention,'reference'),(intervention,'intervention',total,'intervention')):
        if a['request'][side_a]['sha256']!=b['request'][side_b]['sha256']:raise ValueError('Three-way comparisons do not share the same actual parent/control/intervention histories')
    if len({restart['request']['reference']['sha256'],restart['request']['intervention']['sha256'],intervention['request']['intervention']['sha256']})!=3:
        raise ValueError('Parent, unchanged control and intervention histories must be distinct actual products')
    parent,parent_arrays,binding=_load_product(restart['request']['reference'],restart['request_base'],memory/6);sources.update(binding)
    roundoff={name:np.max(np.abs(parent_arrays[field]),axis=1) for name,field in (('position','position_kpc'),('velocity','velocity_kms'))}
    for name,reference in (('control_run',restart['request']['intervention']),('intervention_run',intervention['request']['intervention'])):
        history,history_arrays,binding=_load_product(reference,restart['request_base'] if name=='control_run' else intervention['request_base'],memory/6);sources.update(binding)
        for quantity,field in (('position','position_kpc'),('velocity','velocity_kms')):
            roundoff[quantity]=np.maximum(roundoff[quantity],np.max(np.abs(history_arrays[field]),axis=1))
        if file_sha256(history['native_stage_receipt']['path'])!=history['native_stage_receipt']['sha256']:raise ValueError('Late native history receipt identity changed')
        receipt=verify_completed_receipt(history['native_stage_receipt']['path']);generation=Path(receipt['generation']).resolve()
        owned={str((generation/key).resolve()):value for key,value in receipt['files_sha256'].items()}
        path=(base/request[name]['path']).resolve()
        if owned.get(str(path))!=request[name]['sha256'] or file_sha256(path)!=request[name]['sha256'] or path.stat().st_size*16>memory/6:
            raise ValueError('Late run report is outside its actual completed native history generation')
        run=json.loads(path.read_text())
        if run.get('schema')!='native-counterfactual-run-v1' or run.get('execution_status')!='COMPLETE' or Path(run['native_run_directory']).resolve()!=Path(history['native_run_directory']).resolve() or Path(run['parent_run']).resolve()!=Path(parent['native_run_directory']).resolve():
            raise ValueError('Late comparison run/parent/history roles differ from their actual executions')
        invocation_path=Path(run['invocation_receipt']['path']).resolve()
        if owned.get(str(invocation_path))!=run['invocation_receipt']['sha256'] or file_sha256(invocation_path)!=run['invocation_receipt']['sha256']:
            raise ValueError('Late native invocation changed or is outside the actual completed generation')
        invocation=json.loads(invocation_path.read_text());verify_invocation_inputs(invocation)
        if invocation.get('status')!='RETURNED' or invocation.get('returncode')!=0:raise ValueError('Late three-way native branch did not complete its invocation')
        sources.update(invocation['input_hashes']);sources[str(invocation_path)]=file_sha256(invocation_path)
        lineage_path=Path(run['intervention_ic_lineage']['path']).resolve()
        if run['source_hashes'].get(str(lineage_path))!=run['intervention_ic_lineage']['sha256'] or file_sha256(lineage_path)!=run['intervention_ic_lineage']['sha256'] or lineage_path.stat().st_size*16>memory/6:
            raise ValueError('Late three-way source lineage changed or exceeds allocation')
        lineage=json.loads(lineage_path.read_text())
        if lineage.get('schema')!='native-counterfactual-ic-lineage-v1' or lineage['request']['schema']!='native-counterfactual-ic-request-v3' or lineage.get('intervention_epoch')!=run.get('intervention_epoch'):
            raise ValueError('Actual late saved-state native lineage required')
        operation=lineage['request']['operation']
        if (name=='control_run')!=(operation=='retain_collisionless_state') or operation not in {'retain_collisionless_state','remove_dm_particles','scale_dm_particle_masses'}:
            raise ValueError('Late three-way unchanged/changed native state roles are invalid')
        if lineage.get('native_state_policy')!='FRESH_PHASE_SPACE_REINITIALIZATION; NOT_PARENT_INTEGRATOR_RESTART' or (name=='control_run' and lineage.get('full_system_initial_state')!='UNCHANGED_COLLISIONLESS_PHASE_AND_MASS'):
            raise ValueError('Late comparison control cannot inherit an unrecorded state transformation')
        param=Path(run['native_run_directory'])/'input/param.txt';binary=Path(run['native_run_directory'])/'input/Gadget4'
        args=[(Path(value) if Path(value).is_absolute() else Path(invocation['cwd'])/value).resolve() for value in invocation['command']]
        initial=Path(lineage['native_ic']['path']).resolve();params=parameter_values(param)
        used_initial=Path(params['InitCondFile']+'.hdf5').resolve()
        if param.resolve() not in args or binary.resolve() not in args or invocation['command'][-1]!='0' or invocation['input_hashes'].get(str(binary.resolve()))!=run['original_engine_sha256'] or invocation['input_hashes'].get(str(used_initial))!=lineage['native_ic']['sha256'] or file_sha256(initial)!=lineage['native_ic']['sha256']:
            raise ValueError('Late native invocation did not use the declared original engine and materialized state')
        ignored={'InitCondFile','OutputDir','OutputListFilename','TimeLimitCPU','CpuTimeBetRestartFile','MaxMemSize'}
        definitions.append({'parameters':{key:value for key,value in params.items() if key not in ignored},
            'engine':run['original_engine_sha256'],'epoch':run['intervention_epoch'],'original_state':lineage['original_ic']['sha256']})
        for key,value in lineage['source_hashes'].items():
            if file_sha256(key)!=value:raise ValueError('Late three-way original state source changed')
        sources.update(lineage['source_hashes']);sources[str(lineage_path)]=file_sha256(lineage_path);sources[str(path)]=file_sha256(path)
        runs[name]=run
    if definitions[0]!=definitions[1]:raise ValueError('Late control/intervention changes unrelated native numerics, physical state origin or epoch')
    runs['roundoff_scales']=roundoff
    return request,comparisons,runs,sources


def native_late_threeway_input_bindings(parameters):
    return _late_threeway_inputs(parameters)[-1]


def publish_native_late_threeway(parameters,destination):
    from pathlib import Path
    from .data import exact_uint64_ids,export_numeric_audit
    from .foundation import file_sha256
    request,comparisons,runs,sources=_late_threeway_inputs(parameters)
    reports=[comparisons[name][0] for name in ('restart_comparison','intervention_comparison','total_comparison')]
    values=[comparisons[name][1] for name in ('restart_comparison','intervention_comparison','total_comparison')]
    ids=exact_uint64_ids(values[0]['particle_ids'],allow_empty=False);epochs=len(values[0]['reference_times_myr']);count=len(ids)
    if not 2<=epochs<=request['maximum_snapshots'] or count>request['maximum_selected_particles']:raise MemoryError('Three-way actual support exceeds declared bounds')
    for report,arrays in zip(reports,values):
        if report['namespace']!=reports[0]['namespace'] or report['initial_state_sha256']!=reports[0]['initial_state_sha256'] or report['periodic']!=reports[0]['periodic'] or not np.array_equal(ids,arrays['particle_ids']) or not np.array_equal(values[0]['initial_mass_msun'],arrays['initial_mass_msun']):
            raise ValueError('Three-way native comparisons change the common selected stream/frame/weights')
        for name in ('reference_times_myr','intervention_times_myr','periodic_box_proper_kpc'):
            if not np.array_equal(values[0][name],arrays[name]):raise ValueError('Three-way comparison requires exact common measured saved epochs/domain')
        if request['frame']=='host_relative':
            if 'host_reflex' not in report or report['host_reflex']!=reports[0].get('host_reflex'):
                raise ValueError('Three-way host-relative comparisons change the common selected host definition')
            for side in ('reference','intervention'):
                host_velocity=np.max(np.abs(arrays[side+'_host_velocity_kms']),axis=0)
                runs['roundoff_scales']['velocity']=np.maximum(runs['roundoff_scales']['velocity'],host_velocity[None,:])
    prefix='host_relative_' if request['frame']=='host_relative' else ''
    output={'particle_ids':ids,'times_myr':values[0]['reference_times_myr'],'initial_mass_msun':values[0]['initial_mass_msun']};decisions={}
    for quantity,unit,ceiling,floor in (('position','kpc','maximum_restart_position_error_kpc','position_signal_floor_kpc'),('velocity','kms','maximum_restart_velocity_error_kms','velocity_signal_floor_kms')):
        field=prefix+quantity+'_change_'+unit;vectors=[np.asarray(arrays[field],float) for arrays in values]
        if any(vector.shape!=(epochs,count,3) or not np.isfinite(vector).all() for vector in vectors):raise ValueError('Three-way measured coordinates are incomplete')
        restart,signal,total=vectors;closure=total-restart-signal
        if quantity=='position' and reports[0]['periodic']:
            box=values[0]['periodic_box_proper_kpc'][:,None,None];closure-=box*np.rint(closure/box)
        scale=np.maximum.reduce([np.abs(total),np.abs(restart),np.abs(signal),np.full(total.shape,request[floor])])
        scale=np.maximum(scale,runs['roundoff_scales'][quantity][:,None,:])
        if quantity=='position' and reports[0]['periodic']:scale=np.maximum(scale,values[0]['periodic_box_proper_kpc'][:,None,None])
        if np.any(np.abs(closure)>128*np.finfo(float).eps*scale):raise ValueError('Three-way differences do not close in the declared saved coordinate frame')
        error=np.linalg.norm(restart,axis=2);signal_scale=np.maximum(np.linalg.norm(signal,axis=2),request[floor])
        allowed=np.minimum(request[ceiling],request['maximum_restart_signal_fraction']*signal_scale);passed=error<=allowed
        output.update({quantity+'_restart_difference_'+unit:restart,quantity+'_intervention_difference_'+unit:signal,
            quantity+'_total_difference_'+unit:total,quantity+'_closure_residual_'+unit:closure,quantity+'_restart_allowed_'+unit:allowed,
            quantity+'_restart_within_limits':passed})
        decisions[quantity]={'status':'WITHIN_DECLARED_RESTART_BUDGET' if passed.all() else 'REINITIALIZATION_UNRESOLVED',
            'maximum_restart_norm':float(error.max()),'maximum_restart_signal_fraction':float(np.max(error/signal_scale))}
    report={'schema':'measured-native-late-threeway-comparison-v1','execution_status':'COMPLETE','request':request,'request_base':str(Path(parameters['request_base']).resolve()),
        'namespace':reports[0]['namespace'],'intervention_epoch':runs['control_run']['intervention_epoch'],'decisions':decisions,
        'status':'WITHIN_DECLARED_REINITIALIZATION_BUDGETS' if all(row['status']=='WITHIN_DECLARED_RESTART_BUDGET' for row in decisions.values()) else 'REINITIALIZATION_UNRESOLVED',
        'definition':'restart=unchanged fresh-parent; intervention=changed fresh-unchanged fresh; total=changed fresh-parent',
        'scope':'same-saved-epoch exact-ID source-bound decomposition; no cancellation correction, independent resolution/model accuracy or global causal acceptance',
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    specs={name:{'units':'identity' if name=='particle_ids' else 'Myr' if name=='times_myr' else 'Msun' if name=='initial_mass_msun' else 'dimensionless' if name.endswith('limits') else 'km / s' if name.endswith('kms') else 'proper kpc','quantity':'native late three-way '+name} for name in output}
    specs['particle_ids']['identity_namespace']=reports[0]['namespace']
    return export_numeric_audit(destination,report,output,specs,algorithm=source,source_hashes=sources,context={'epoch':{'canonical_time':'actual saved native epochs','measured_epochs':reports[0]['reference_epochs']},'coordinates':{'frame':request['frame']}})


def native_late_threeway_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Late three-way producer exceeds its workflow allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Late three-way producer escapes its owned generation')
    return publish_native_late_threeway(parameters,destination)


def native_late_threeway_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Measure actual parent/fresh-control/intervention native stream differences')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text());parameters.setdefault('request_base',str(path.parent))
    publish_native_late_threeway(parameters,args.output);return 0


def attach_live_control(reference, live, *, identity_namespace, source_namespace):
    """Compare native live branch products without relabelling imposed fields."""
    if identity_namespace != source_namespace: raise ValueError('Live controls must share exact native ID namespace')
    if not np.array_equal(reference['particle_ids'], live['particle_ids']): raise ValueError('Live control ID join required before comparison')
    if not np.array_equal(reference['times_myr'], live['times_myr']): raise ValueError('Matched physical epochs required')
    return {'position_change_kpc': live['position_kpc']-reference['position_kpc'],
            'velocity_change_kms': live['velocity_kms']-reference['velocity_kms']}


def standard_reflex_control_ladder(full_field, *, host_components, satellite_components, wake_components, reference_epoch_myr):
    """Static -> evolving fixed-center -> satellite -> measured reflex/wake.

    Components must already be disjoint measured fields. This ladder freezes
    translations without changing the body's actual measured deformation.
    A native live comparison supplies the independent backreaction control.
    """
    from core.gravity_fields import ComponentField, FrozenField, FixedCenterHost
    names = set(full_field.components)
    if set(host_components+satellite_components+wake_components) != names:
        raise ValueError('Every component must have one ladder role')
    if sum(map(len,(host_components,satellite_components,wake_components))) != len(names):
        raise ValueError('Reflex ladder roles overlap')
    result = {}
    def field(rows): return ComponentField(rows, accounting={name: full_field.accounting[name] for name in rows})
    result['static'] = field({name: FrozenField(full_field.components[name], reference_epoch_myr) for name in host_components})
    fixed = {name: FixedCenterHost(full_field.components[name], reference_epoch_myr) if hasattr(full_field.components[name], 'frame') else full_field.components[name] for name in host_components}
    result['time_dependent_fixed_center'] = field(fixed)
    result['satellites_fixed_center'] = field({**fixed, **{name: full_field.components[name] for name in satellite_components}})
    result['reflex'] = field({name: full_field.components[name] for name in host_components+satellite_components})
    result['reflex_wake'] = full_field
    return result
# @research2-end-module stream_forward_model












