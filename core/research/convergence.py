"""Plan controlled native variants, materialize their inputs and compare measurements.

All scientific tolerances and input units are declared by the caller.
"""
from __future__ import annotations

import json
import math
import numpy as np
from copy import deepcopy
from core.foundation import atomic_json, atomic_text, file_sha256, canonical_digest
from core.foundation import merge_source_bindings as _merge
from pathlib import Path


# Controlled experiment plans and measured, matched convergence contrasts.
CONTROLS = {
    'mass_resolution': ('simulation', 'grid'), 'softening': ('simulation', 'softening'),
    'timestep': ('simulation', 'timestep'), 'output_cadence': ('epochs', 'backbone_redshifts'),
    'integration_accuracy': ('simulation','err_tol_int_accuracy'),
    'gravity_force_accuracy': ('simulation','err_tol_force_acc'),
    'gravity_tree_opening': ('simulation','err_tol_theta'),
    'start_epoch': ('epochs', 'z_start'), 'finder': ('science', 'finder'),
    'sidm_kernel': ('dark_matter', 'interactions', 'kernel_support'),
    'sidm_kernel_shape': ('dark_matter', 'interactions', 'kernel_kind'),
    'sidm_kernel_origin_bound': ('dark_matter', 'interactions', 'kernel_origin_relative_bound'),
    'sidm_timestep': ('dark_matter', 'interactions', 'maximum_probability'),
    'sidm_angular_cdf': ('dark_matter', 'interactions', 'cdf_tolerance'),
    'sidm_frequent_approximation': ('dark_matter', 'interactions', 'frequent_tolerance'),
    'sidm_split_angle': ('dark_matter', 'interactions', 'split_angle'),
    'sidm_stochastic_realization': ('dark_matter', 'interactions', 'seed'),
    'subgrid_model': ('science', 'subgrid_model'), 'observer_selection': ('science', 'observer_selection')}


def _set_path(spec, path, value):
    node = spec
    for key in path[:-1]:
        if key not in node:
            raise ValueError('Control parameter path missing: ' + '.'.join(path))
        node = node[key]
    if path[-1] not in node:
        raise ValueError('Control path absent from baseline')
    node[path[-1]] = value


def construct_independent_resolution_and_model_controls(base_spec, request):
    """Vary one control; record IC phases separately from SIDM operator trials."""
    baseline = deepcopy(base_spec)
    if 'seed' not in baseline:
        raise ValueError('Explicit phase seed required')
    baseline_only = request.get('baseline_ensemble_only',False)
    if type(baseline_only) is not bool:
        raise ValueError('Baseline-only native ensemble selection must be an exact boolean')
    if not isinstance(request.get('control_values'),dict):
        raise ValueError('Control matrix must explicitly enumerate native control dimensions')
    if not request['control_values'] and not baseline_only:
        raise ValueError('Explicit control matrix required')
    if baseline_only and request['control_values']:
        raise ValueError('A baseline-only ensemble cannot silently include native control variants')
    seed_list = request['independent_seeds']
    if not seed_list or any(type(x) is not int or x < 0 for x in seed_list) or len(set(seed_list)) != len(seed_list):
        raise ValueError('Distinct independent realization phase seeds required')
    interactions = baseline.get('dark_matter',{}).get('interactions',{})
    operator_seeds = request.get('independent_operator_seeds')
    if baseline_only and len(seed_list) < 2 and operator_seeds is None:
        raise ValueError('Baseline-only ensembles need independently seeded native realizations')
    if operator_seeds is not None:
        if interactions.get('kind') != 'elastic_sidm':
            raise ValueError('Independent operator ensembles require native elastic SIDM')
        if not isinstance(operator_seeds,list) or len(operator_seeds) < 2 or any(type(x) is not int or not 0 <= x < 2**64 for x in operator_seeds) or len(set(operator_seeds)) != len(operator_seeds):
            raise ValueError('Operator ensemble requires at least two distinct exact uint64 seeds')
        if 'seed' not in interactions:
            raise ValueError('Operator ensemble cannot add an absent native SIDM seed control')
    family_spec = deepcopy(baseline)
    family_spec.pop('seed')
    family_spec.get('dark_matter',{}).get('interactions',{}).pop('seed',None)
    model_family_hash = canonical_digest(family_spec)
    maximum = request.get('maximum_runs')
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError('Convergence run bound must be an explicit positive integer')
    if operator_seeds is not None and maximum is None:
        raise ValueError('Crossed phase/operator campaigns require an explicit total run bound')
    result,groups = [],[]
    for seed in seed_list:
        for operator in operator_seeds if operator_seeds is not None else [None]:
            base = deepcopy(baseline)
            base['seed'] = seed
            if operator is not None:
                _set_path(base,CONTROLS['sidm_stochastic_realization'],operator)
            suffix = f'seed{seed}' + (f'_operator{operator}' if operator is not None else '')
            baseline_id = 'baseline_'+suffix
            group = {'realization_group_id':suffix,'baseline_run_id':baseline_id,
                'ic_phase_seed':seed,'operator_realization_seed':base.get('dark_matter',{}).get('interactions',{}).get('seed'),
                'common_model_family_hash':model_family_hash}
            groups.append(group)
            signature = canonical_digest(base)
            result.append({**group,'run_id':baseline_id,'specification':base,
                'changed_controls':[],'common_signature':signature})
            if maximum is not None and len(result) > maximum:
                raise ValueError('Convergence matrix exceeds its declared run bound')
            for control, values in request['control_values'].items():
                if control not in CONTROLS or not values:
                    raise ValueError('Unknown or empty control family: ' + control)
                if len({canonical_digest(v) for v in values}) != len(values):
                    raise ValueError('Duplicate control setting')
                if control in {'integration_accuracy','gravity_force_accuracy','gravity_tree_opening'}:
                    if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0 for value in values):
                        raise ValueError('Native gravity/integration accuracy controls require finite positive scalar settings')
                if control == 'sidm_stochastic_realization':
                    if interactions.get('kind') != 'elastic_sidm':
                        raise ValueError('SIDM stochastic controls require native elastic interactions')
                    if any(type(value) is not int or not 0 <= value < 2**64 for value in values):
                        raise ValueError('Explicit exact uint64 SIDM operator seeds required')
                family_start = len(result)
                for index, value in enumerate(values):
                    branch = deepcopy(base)
                    _set_path(branch, CONTROLS[control], value)
                    if branch == base:
                        continue
                    if control == 'start_epoch':
                        branches = request.get('regenerated_bundles', {})
                        key = str(value)
                        if key not in branches:
                            raise ValueError('Start epoch control requires exact regenerated transfer bundle')
                        branch['cosmology_bundle'] = branches[key]
                        branch['epoch_regeneration_lineage'] = {'parent': base['cosmology_bundle'], 'z_start': value,
                            'phase_seed': seed, 'mode': 'exact solver regeneration'}
                    result.append({**group,'run_id': f'{control}_{index}_{suffix}', 'specification': branch,
                        'changed_controls': [control], 'control_value': value, 'common_signature': signature})
                    if maximum is not None and len(result) > maximum:
                        raise ValueError('Convergence matrix exceeds its declared run bound')
                if len(result) == family_start:
                    raise ValueError('Control family contains no setting distinct from baseline: '+control)
    maximum = request.get('maximum_runs')
    if maximum is not None and (type(maximum) is not int or maximum < 1 or len(result) > maximum):
        raise ValueError('Convergence matrix exceeds its declared run bound')
    return {'schema': 'convergence-campaign-plan-v1', 'base_hash': canonical_digest(baseline),
        'common_model_family_hash':model_family_hash,'realization_groups':groups,
        'ensemble_axes':{'ic_phase_seeds':list(seed_list),'sidm_operator_seeds':operator_seeds,
            'pooling_policy':'compare within each group; ensemble intake declares one independence scope'},
        'frozen_requirement_hash': canonical_digest(request['requirement']), 'runs': result,
        'run_count': len(result), 'scientific_acceptance': 'UNRESOLVED'}


def _compare_supported_campaign(request, measurements):
    """Compare a paired difference in the observable's declared physical units.

    For d = X - R, C_d = C_X + C_R - C_XR - C_XR.T. Omitting the
    cross terms is allowed only for explicitly independent measurements.
    The tested bound is |d| + s sqrt(diag(C_d)); its allowed width is
    absolute_tolerance + relative_tolerance * max(|R|, dimensional_floor).
    Covariances have units squared; all widths have the observable's units.
    """
    from core.research.errors import _finite, _psd, _support_mask
    if request['control'] not in CONTROLS:
        raise ValueError('Unsupported convergence dimension')
    if len(measurements) < 2:
        raise ValueError('Two actual measured controlled runs required')
    identities = [q['run_id'] for q in measurements]
    if len(set(identities)) != len(identities):
        raise ValueError('Repeated run identity')
    try:
        reference = next(q for q in measurements if q['run_id'] == request['reference_run'])
    except StopIteration:
        raise ValueError('Reference run not present') from None
    value = _finite(reference['value']).reshape(-1)
    n = len(value)
    if n == 0:
        raise ValueError('Empty observable selection')
    dimensional_floor = np.broadcast_to(_finite(request['dimensional_floor']), (n,))
    absolute = np.broadcast_to(_finite(request['absolute_tolerance']), (n,))
    relative = np.broadcast_to(_finite(request['relative_tolerance']), (n,))
    if np.any(dimensional_floor <= 0) or np.any(absolute < 0) or np.any(relative < 0):
        raise ValueError('Positive floor and nonnegative tolerances required')
    tol = request['covariance_tolerance']
    reference_cov = _psd(reference['covariance'], n, tol, 'reference covariance')
    allowed = absolute + relative*np.maximum(np.abs(value), dimensional_floor)
    rows = []
    full_support = _support_mask(reference['supported'], (n,))
    if full_support.shape != (n,):
        raise ValueError('Invalid reference support mask')
    for record in measurements:
        for key in ('common_experiment_signature', 'observable_definition_hash', 'units', 'epoch', 'bins', 'phase_seed'):
            if record[key] != reference[key]:
                raise ValueError('Comparison changes unrelated science or phase: ' + key)
        expected = [] if record is reference else [request['control']]
        if record['changed_controls'] != expected:
            raise ValueError('Control comparison must vary exactly the declared dimension')
        if not record.get('source_hashes'):
            raise ValueError('Measured sources required')
        for path, checksum in record['source_hashes'].items():
            if file_sha256(path) != checksum:
                raise ValueError('Convergence source changed')
        current = _finite(record['value'], (n,))
        mask = _support_mask(record['supported'], (n,))
        if mask.shape != (n,):
            raise ValueError('Control support shape mismatch')
        mask = mask & _support_mask(reference['supported'], (n,))
        cov = _psd(record['covariance'], n, tol, 'control covariance') + reference_cov
        if record is reference:
            cov = np.zeros((n, n))
        elif 'paired_cross_covariance' in record:
            cross = _finite(record['paired_cross_covariance'], (n, n))
            # Verify the full paired joint covariance too, not only its contrast.
            _psd(np.block([[np.asarray(record['covariance'],float), cross], [cross.T, reference_cov]]), 2*n, tol, 'paired joint covariance')
            cov -= cross + cross.T
        elif record.get('independent_of_reference') is not True:
            raise ValueError('Paired runs require cross covariance or recorded independence')
        cov = _psd(cov, n, tol, 'difference covariance')
        delta = current-value
        multiplier = request['sigma_multiplier']
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError('Positive uncertainty multiplier required')
        bound = np.abs(delta) + multiplier*np.sqrt(np.maximum(np.diag(cov), 0))
        accepted = mask & (bound <= allowed)
        if record is not reference:
            full_support = full_support & accepted
        rows.append({'run_id': record['run_id'], 'control_value': record.get('control_value'),
            'difference': delta.tolist(), 'difference_covariance': cov.tolist(), 'supported': mask.tolist(),
            'accepted_bins': accepted.tolist(), 'error_bound': bound.tolist(), 'source_hashes': record['source_hashes']})
    return {'schema': 'measured-convergence-campaign-v1', 'request_hash': canonical_digest(request),
        'control': request['control'], 'units': reference['units'], 'rows': rows,
        'accepted_domain': [b for i, b in enumerate(reference['bins']) if full_support[i]],
        'unsupported_or_failed_domain': [b for i, b in enumerate(reference['bins']) if not full_support[i]],
        'status': 'PASS' if bool(full_support.all()) else 'UNRESOLVED',
        'science_acceptance': 'CONTROL_ONLY', 'scope': 'one declared numerical/model control, no arbitrary parameter extrapolation'}


def compare_campaign(request, measurements):
    """Compare only the shared measured subspace; retain every unsupported bin.

    Missing native measurements may carry NaN in explicitly unsupported bins.
    They never acquire zero residuals or uncertainty from projection. The
    covariance returned for each contrast is indexed by its declared subspace.
    """
    from core.research.errors import _finite, _psd, _support_mask
    if request['control'] not in CONTROLS or len(measurements) < 2:
        raise ValueError('Recognized control and at least two measured runs required')
    identities = [q['run_id'] for q in measurements]
    if len(set(identities)) != len(identities):
        raise ValueError('Repeated convergence run identity')
    reference = next((q for q in measurements if q['run_id'] == request['reference_run']),None)
    if reference is None:
        raise ValueError('Reference run not present')
    bins = reference['bins']
    if not isinstance(bins,list) or not bins or len({canonical_digest(q) for q in bins}) != len(bins):
        raise ValueError('Distinct explicit convergence domain bins required')
    n = len(bins)
    tolerance = request['covariance_tolerance']
    if isinstance(tolerance,bool) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Finite nonnegative covariance tolerance required')
    multiplier = request['sigma_multiplier']
    if isinstance(multiplier,bool) or not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Positive frozen uncertainty multiplier required')
    limits = {}
    for key in ('dimensional_floor','absolute_tolerance','relative_tolerance'):
        limits[key] = np.broadcast_to(_finite(request[key]),(n,))
        invalid = np.any(limits[key] <= 0) if key == 'dimensional_floor' else np.any(limits[key] < 0)
        if invalid:
            raise ValueError('Positive floor and nonnegative tolerances required')
    masks = []
    for record in measurements:
        for key in ('common_experiment_signature','observable_definition_hash','units','epoch','bins','phase_seed'):
            if record[key] != reference[key]:
                raise ValueError('Comparison changes unrelated science or phase: '+key)
        if (request['control'] != 'sidm_stochastic_realization'
                and record.get('sidm_operator_seed') != reference.get('sidm_operator_seed')):
            raise ValueError('Numerical comparison changes an unrelated SIDM operator seed')
        if record['changed_controls'] != ([] if record is reference else [request['control']]):
            raise ValueError('Convergence must vary exactly the declared dimension')
        sources = record.get('source_hashes',{})
        if not sources or any(file_sha256(path) != expected for path,expected in sources.items()):
            raise ValueError('Actual current convergence measurement sources required')
        values = np.asarray(record['value'],float)
        covariance = np.asarray(record['covariance'],float)
        mask = _support_mask(record['supported'],(n,))
        if values.shape != (n,) or covariance.shape != (n,n) or not np.isfinite(values[mask]).all():
            raise ValueError('Supported convergence values/covariance must match declared bins')
        indices = np.flatnonzero(mask)
        _psd(covariance[np.ix_(indices,indices)],len(indices),tolerance,'measured supported covariance')
        if record is not reference and 'paired_cross_covariance' not in record and record.get('independent_of_reference') is not True:
            raise ValueError('Paired controls require measured cross covariance or explicit independence')
        if 'paired_cross_covariance' in record and np.asarray(record['paired_cross_covariance']).shape != (n,n):
            raise ValueError('Paired cross covariance does not match declared bins')
        masks.append(mask)
    reference_mask = _support_mask(reference['supported'],(n,))
    for record,mask in zip(measurements,masks):
        if record is reference or 'paired_cross_covariance' not in record:
            continue
        paired = np.flatnonzero(mask & reference_mask)
        current_covariance = np.asarray(record['covariance'],float)[np.ix_(paired,paired)]
        reference_covariance = np.asarray(reference['covariance'],float)[np.ix_(paired,paired)]
        cross = np.asarray(record['paired_cross_covariance'],float)[np.ix_(paired,paired)]
        _psd(np.block([[current_covariance,cross],[cross.T,reference_covariance]]),2*len(paired),tolerance,'supported paired joint covariance')
    common = np.logical_and.reduce(masks)
    indices = np.flatnonzero(common)
    result = None
    if len(indices):
        projected_request = dict(request)
        projected_request.update({key:value[indices].tolist() for key,value in limits.items()})
        projected = []
        for record in measurements:
            covariance = np.asarray(record['covariance'],float)
            row = {**record,'value':np.asarray(record['value'],float)[indices].tolist(),
                'bins':[bins[i] for i in indices],'supported':[True]*len(indices),
                'covariance':covariance[np.ix_(indices,indices)].tolist()}
            if 'paired_cross_covariance' in record:
                cross = np.asarray(record['paired_cross_covariance'],float)
                row['paired_cross_covariance'] = cross[np.ix_(indices,indices)].tolist()
            projected.append(row)
        result = _compare_supported_campaign(projected_request,projected)
    projected_rows = {q['run_id']:q for q in result['rows']} if result else {}
    rows = []
    accepted_domain_indices = set(range(n)) if result else set()
    for record,mask in zip(measurements,masks):
        measured = projected_rows.get(record['run_id'])
        row = {**measured} if measured else {'run_id':record['run_id'],
            'control_value':record.get('control_value'),'source_hashes':record['source_hashes'],
            'difference_covariance':[]}
        for key in ('difference','error_bound','accepted_bins'):
            full = [False if key == 'accepted_bins' else None for _ in range(n)]
            if measured:
                for i,value in zip(indices,measured[key]):
                    full[int(i)] = value
            row[key] = full
        row.update(supported=common.tolist(),native_supported=mask.tolist(),covariance_domain_indices=indices.tolist())
        if record is not reference:
            accepted_domain_indices &= {i for i,accepted in enumerate(row['accepted_bins']) if accepted}
        rows.append(row)
    return {'schema':'measured-convergence-campaign-v1','request_hash':canonical_digest(request),
        'control':request['control'],'units':reference['units'],'rows':rows,
        'accepted_domain':[bin for i,bin in enumerate(bins) if i in accepted_domain_indices],
        'unsupported_or_failed_domain':[bin for i,bin in enumerate(bins) if i not in accepted_domain_indices],
        'status':'PASS' if len(accepted_domain_indices) == n else 'UNRESOLVED',
        'science_acceptance':'CONTROL_ONLY','scope':'one declared numerical/model control, no arbitrary parameter extrapolation',
        'missing_policy':'unsupported contrasts are null; covariance covers only covariance_domain_indices'}


def queue_bounded_refinement(plan, decisions, limits, already_queued=0):
    remaining = limits['maximum_children']-already_queued
    if remaining < 0:
        raise ValueError('Existing followups exceed frozen limit')
    chosen, unresolved = [], []
    for decision in decisions:
        if len(chosen) < remaining and decision['depth'] < limits['maximum_depth']:
            chosen.append(decision)
        else:
            unresolved.append({**decision, 'reason': 'recorded refinement bound exhausted'})
    return {'children': chosen, 'unresolved': unresolved, 'parent_plan_hash': canonical_digest(plan)}


def _publish_convergence_progress(destination,state):
    """Retain each orchestration transition; state is never resume proof."""
    import uuid
    from core.foundation import atomic_json, atomic_text
    history = destination/'progress-history'
    history.mkdir(parents=True,exist_ok=True)
    state['transition_index'] = state.get('transition_index',0)+1
    event = history/(state['attempt_id']+'-'+str(state['transition_index']).zfill(8)+'-'+uuid.uuid4().hex+'.json')
    atomic_json(event,state)
    atomic_text(event.with_suffix('.sha256'),file_sha256(event)+'\n')
    atomic_json(destination/'campaign-state.json',state)
    atomic_text(destination/'campaign-state.sha256',file_sha256(destination/'campaign-state.json')+'\n')


def execute_and_measure(base_spec, request, workflow_template, destination, *, request_base, resume=False):
    """Execute every planned variant through a real public evolution/measurement DAG.

    Template argv can reference {variant_spec}, {variant_run_id} and {seed}.
    It must publish a measured JSON product named convergence_measurement with
    all matching/signature/source fields consumed by compare_campaign.
    """
    from pathlib import Path
    from core.workflow import execute_workflow
    from core.foundation import atomic_json, atomic_text
    from core.native import native_run_lock
    from copy import deepcopy
    import uuid,time
    if request.get('native_configuration'):
        
        workflow_template = attach_native_configuration_stage(workflow_template,request['native_configuration'])
    if request.get('native_measurement'):
        from core.research.measurements import attach_native_measurement_stage
        workflow_template = attach_native_measurement_stage(workflow_template,request['native_measurement'])
    additional_claims = request.get('additional_observables',[])
    if not isinstance(additional_claims,list):
        raise ValueError('Additional observable claims must be an explicitly bounded list')
    if additional_claims:
        from core.workflow import safe_id
        from core.research.measurements import attach_native_measurement_stage
        maximum = request.get('maximum_observables')
        if type(maximum) is not int or not 1+len(additional_claims)<=maximum:
            raise ValueError('Requested observable family exceeds its explicit total observable bound')
        identities=set()
        for claim in additional_claims:
            cid=safe_id(claim['claim_id'])
            if cid in identities:raise ValueError('Repeated additional observable claim identity')
            identities.add(cid)
            if not isinstance(claim.get('requirement'),dict):raise ValueError('Every additional observable needs frozen comparison requirements')
            settings=deepcopy(claim['native_measurement'])
            settings['product_name']=safe_id('measurement_'+cid)
            settings['measurement_stage_id']=safe_id('measure_'+cid)
            settings['reference_placeholder']=safe_id('reference_measurement_'+cid)
            if settings.get('isolated_history'):
                settings['isolated_history']['stage_id']=safe_id('history_'+cid)
                settings['isolated_history']['observable_stage_id']=safe_id('history_observable_'+cid)
            workflow_template=attach_native_measurement_stage(workflow_template,settings)
    plan = construct_independent_resolution_and_model_controls(base_spec, request)
    destination = Path(destination).resolve()
    def substitute(value, replacements):
        if isinstance(value, dict):
            return {k: substitute(v, replacements) for k, v in value.items()}
        if isinstance(value, list):
            return [substitute(v, replacements) for v in value]
        if isinstance(value, str):
            for key, replacement in replacements.items():
                value = value.replace('{'+key+'}', str(replacement))
        return value
    measurements, receipts = [], []
    additional_measurements={claim['claim_id']:[] for claim in additional_claims}
    additional_receipts={claim['claim_id']:[] for claim in additional_claims}
    additional_paths={claim['claim_id']:{} for claim in additional_claims}
    retained_measurement_bytes=0
    collection_memory=workflow_template['resources']['memory_gib']*1024**3
    with native_run_lock(destination, operation='measured convergence controls'):
        destination.mkdir(parents=True, exist_ok=True)
        plan_path = destination/'plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError('Frozen convergence plan changed; select a new destination')
        scope_path = destination/'frozen-campaign-scope.json'
        scope = {'schema':'convergence-campaign-scope-v1','request':request,
            'workflow_template':workflow_template,'base_spec':base_spec,
            'request_base':str(Path(request_base).resolve())}
        if scope_path.exists() and json.loads(scope_path.read_text()) != scope:
            raise ValueError('Convergence workflow/request scope changed; select a new destination')
        if (destination/'campaign-state.json').exists() and not resume:
            raise FileExistsError('Existing convergence attempt requires explicit resume or a new destination')
        atomic_json(scope_path,scope)
        atomic_json(plan_path,plan)
        state = {'schema':'convergence-campaign-state-v1','attempt_id':uuid.uuid4().hex,
            'plan_hash':canonical_digest(plan),'scope_hash':canonical_digest(scope),
            'execution_status':'RUNNING','phase':'PREFLIGHT','active_run':None,
            'completed_preflights':[],'completed_runs':[],'completed_comparisons':[],
            'started_unix':time.time(),'resume_requested':bool(resume),
            'source_hashes':{str(Path(__file__).resolve()):file_sha256(__file__)},
            'science_acceptance':'NOT_ESTABLISHED',
            'resume_policy':'recheck actual sealed workflow receipts; progress is not execution evidence'}
        _publish_convergence_progress(destination,state)
        prepared = []
        try:
            for branch in plan['runs']:
                sid = branch['run_id']
                state['active_run'] = sid
                _publish_convergence_progress(destination,state)
                specification_path = destination/'specifications'/(sid+'.json')
                if specification_path.exists() and json.loads(specification_path.read_text()) != branch['specification']:
                    raise ValueError('Convergence specification changed; select new destination')
                atomic_json(specification_path, branch['specification'])
                metadata_path = destination/'specifications'/(sid+'.measurement-metadata.json')
                metadata = {**branch,'schema':'native-convergence-variant-metadata-v1',
                    'specification_path':str(specification_path),
                    'specification_sha256':file_sha256(specification_path),
                    'planned_control_dimensions':list(request['control_values'])}
                if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
                    raise ValueError('Frozen native measurement variant metadata changed')
                atomic_json(metadata_path,metadata)
                workflow = substitute(deepcopy(workflow_template), {'variant_spec': specification_path,
                    'variant_run_id': sid, 'variant_metadata':metadata_path,
                    'seed': branch['specification']['seed'], 'common_signature': branch['common_signature'],
                    'operator_seed':branch['specification'].get('dark_matter',{}).get('interactions',{}).get('seed'),
                    'realization_group_id':branch['realization_group_id'],
                    'baseline_run_id':branch['baseline_run_id'],'common_model_family_hash':plan['common_model_family_hash']})
                if 'convergence_measurement' not in workflow['requested_products']:
                    raise ValueError('Convergence workflow must request its actual measured science product')
                preflight = execute_workflow(workflow,destination/sid,request_base=request_base,dry_run=True)
                prepared.append((branch,specification_path,workflow,preflight))
                state['completed_preflights'].append(sid)
                _publish_convergence_progress(destination,state)
            atomic_json(destination/'preflight.json',{'schema':'convergence-family-preflight-v1',
                'plan_hash':canonical_digest(plan),'variants':[{'run_id':branch['run_id'],'plan':preflight}
                    for branch,_,_,preflight in prepared]})
            measurement_paths = {}
            state['phase'] = 'EVOLVE_AND_MEASURE'
            _publish_convergence_progress(destination,state)
            for branch,specification_path,workflow,preflight in prepared:
                sid = branch['run_id']
                state['active_run'] = sid
                _publish_convergence_progress(destination,state)
                reference = measurement_paths.get(branch['baseline_run_id'],'')
                if branch['changed_controls'] and any(q['kind'] == 'convergence_measurement' for q in workflow['stages']) and not reference:
                    raise ValueError('Paired native baseline must be measured before its variant')
                references={'reference_measurement':reference}
                for claim in additional_claims:
                    cid=claim['claim_id']
                    paired=additional_paths[cid].get(branch['baseline_run_id'],'')
                    if branch['changed_controls'] and not paired:
                        raise ValueError('Additional observable baseline must be measured before its actual native variant')
                    references['reference_measurement_'+cid]=paired
                workflow = substitute(workflow,references)
                receipt = execute_workflow(workflow, destination/sid, request_base=request_base, resume=resume)
                if receipt.get('execution_status') != 'COMPLETE':
                    raise RuntimeError('Convergence native workflow did not complete: '+sid)
                producer = next(q for q in workflow['stages'] if 'convergence_measurement' in q['outputs'])
                generation = Path(receipt['stages'][producer['stage_id']]['generation'])
                source = generation/producer['outputs']['convergence_measurement']
                if source.stat().st_size*8 > workflow['resources']['memory_gib']*1024**3:
                    raise MemoryError('Convergence measurement exceeds its declared input workspace')
                retained_measurement_bytes+=source.stat().st_size*8
                if retained_measurement_bytes>collection_memory:
                    raise MemoryError('Retained native observable family exceeds the frozen collection workspace')
                measured = json.loads(source.read_text())
                if measured['run_id'] != sid or measured['common_experiment_signature'] != branch['common_signature'] or measured['changed_controls'] != branch['changed_controls']:
                    raise ValueError('Measurement is not from planned controlled evolution')
                if type(measured['phase_seed']) is not int or measured['phase_seed'] != branch['specification']['seed']:
                    raise ValueError('Convergence measurement does not retain its planned phase seed')
                if request.get('independent_operator_seeds') is not None:
                    expected_operator = branch['specification']['dark_matter']['interactions']['seed']
                    if type(measured.get('sidm_operator_seed')) is not int or measured['sidm_operator_seed'] != expected_operator:
                        raise ValueError('SIDM ensemble measurement does not retain its actual planned operator seed')
                    if measured.get('common_model_family_hash') != plan['common_model_family_hash']:
                        raise ValueError('SIDM ensemble measurement changes its frozen model family')
                elif 'common_model_family_hash' in measured and measured['common_model_family_hash'] != plan['common_model_family_hash']:
                    raise ValueError('Convergence measurement changes its frozen model family')
                if branch['changed_controls'] and measured.get('control_value') != branch['control_value']:
                    raise ValueError('Convergence measurement control setting differs from planned native evolution')
                if measured['source_hashes'].get(str(specification_path)) != file_sha256(specification_path):
                    raise ValueError('Convergence measurement does not bind its exact variant specification')
                measurements.append(measured)
                measurement_paths[sid] = str(source)
                native_producers = [q for q in workflow['stages'] if q['kind'] == 'public_command'
                    and q['parameters'].get('command') == 'gadget']
                if len(native_producers) != 1:
                    raise ValueError('Measured convergence requires exactly one declared native evolution producer')
                native_generation = Path(receipt['stages'][native_producers[0]['stage_id']]['generation'])
                native_receipt_path = native_generation/'stage-receipt.json'
                receipts.append({'run_id': sid, 'workflow': receipt,
                    'measurement_sha256': file_sha256(source),
                    'measurement_reference': {'receipt':str(generation/'stage-receipt.json'),
                        'product':'convergence_measurement'},
                    'native_realization_receipt':str(native_receipt_path),
                    'native_realization_receipt_sha256':file_sha256(native_receipt_path)})
                for claim in additional_claims:
                    cid=claim['claim_id'];product='measurement_'+cid
                    extra_producer=next(row for row in workflow['stages'] if product in row['outputs'])
                    extra_generation=Path(receipt['stages'][extra_producer['stage_id']]['generation'])
                    extra_path=extra_generation/extra_producer['outputs'][product]
                    if extra_path.stat().st_size*8>workflow['resources']['memory_gib']*1024**3:
                        raise MemoryError('Additional observable product exceeds declared input workspace')
                    retained_measurement_bytes+=extra_path.stat().st_size*8
                    if retained_measurement_bytes>collection_memory:
                        raise MemoryError('Additional native observable family exceeds the frozen collection workspace')
                    extra=json.loads(extra_path.read_text())
                    expected={'run_id':sid,'common_experiment_signature':branch['common_signature'],
                        'changed_controls':branch['changed_controls'],'phase_seed':branch['specification']['seed'],
                        'common_model_family_hash':plan['common_model_family_hash']}
                    if any(extra.get(key)!=value for key,value in expected.items()):
                        raise ValueError('Additional observable differs from the planned shared native evolution branch')
                    definition=claim['native_measurement']['request']['observable_definition']
                    if extra.get('observable_definition_hash')!=canonical_digest(definition):
                        raise ValueError('Additional native measurement changes its frozen observable definition')
                    if extra['source_hashes'].get(str(specification_path))!=file_sha256(specification_path):
                        raise ValueError('Additional observable does not bind its actual native variant specification')
                    if branch['changed_controls'] and extra.get('control_value')!=branch['control_value']:
                        raise ValueError('Additional observable changes its declared native control value')
                    expected_seed=branch['specification'].get('dark_matter',{}).get('interactions',{}).get('seed')
                    if extra.get('sidm_operator_seed')!=expected_seed:
                        raise ValueError('Additional observable does not retain the actual native operator seed')
                    additional_measurements[cid].append(extra)
                    additional_paths[cid][sid]=str(extra_path)
                    additional_receipts[cid].append({'run_id':sid,'workflow':receipt,
                        'measurement_sha256':file_sha256(extra_path),
                        'measurement_reference':{'receipt':str(extra_generation/'stage-receipt.json'),'product':product},
                        'native_realization_receipt':str(native_receipt_path),
                        'native_realization_receipt_sha256':file_sha256(native_receipt_path)})
                state['completed_runs'].append({'run_id':sid,'measurement_path':str(source),
                    'measurement_sha256':file_sha256(source),'measurement_generation':str(generation)})
                _publish_convergence_progress(destination,state)
            state.update(phase='COMPARE',active_run=None)
            _publish_convergence_progress(destination,state)
            comparisons = []
            additional_comparisons={claim['claim_id']:[] for claim in additional_claims}
            for group in plan['realization_groups']:
                member_ids = {branch['run_id'] for branch in plan['runs']
                    if branch['realization_group_id'] == group['realization_group_id']}
                subset = [q for q in measurements if q['run_id'] in member_ids]
                for control in request['control_values']:
                    chosen = [q for q in subset if not q['changed_controls'] or q['changed_controls'] == [control]]
                    comparison_request = {**request['requirement'], 'control': control, 'reference_run': group['baseline_run_id']}
                    comparison = compare_campaign(comparison_request, chosen)
                    comparisons.append({**comparison,'realization_group':group})
                    state['completed_comparisons'].append({'realization_group_id':group['realization_group_id'],
                        'control':control,'status':comparison['status'],'comparison_hash':canonical_digest(comparison)})
                    _publish_convergence_progress(destination,state)
                    for claim in additional_claims:
                        cid=claim['claim_id']
                        extra_chosen=[row for row in additional_measurements[cid]
                            if row['run_id'] in member_ids and (not row['changed_controls'] or row['changed_controls']==[control])]
                        extra_comparison=compare_campaign({**claim['requirement'],'control':control,
                            'reference_run':group['baseline_run_id']},extra_chosen)
                        additional_comparisons[cid].append({**extra_comparison,'realization_group':group})
            result = {'schema': 'executed-convergence-family-v1', 'plan_hash': canonical_digest(plan),
                'plan_path':str(plan_path),'scope_path':str(scope_path),
                'scope_hash':canonical_digest(scope),
                'comparisons': comparisons, 'workflow_receipts': receipts,
                'status': 'PASS' if comparisons and all(q['status'] == 'PASS' for q in comparisons) else 'UNRESOLVED',
                'science_acceptance': 'DECLARED_CONTROLS_ONLY'}
            for completed in state['completed_runs']:
                if file_sha256(completed['measurement_path']) != completed['measurement_sha256']:
                    raise ValueError('Convergence measurement changed before final campaign publication')
            result['execution_attempt_id'] = state['attempt_id']
            result_directory = destination/'results'/state['attempt_id']
            result_directory.mkdir(parents=True,exist_ok=False)
            result_path = result_directory/'convergence-family.json'
            atomic_json(result_path,result)
            atomic_text(result_path.with_suffix('.sha256'),file_sha256(result_path)+'\n')
            additional_families={}
            for claim in additional_claims:
                cid=claim['claim_id'];rows=additional_comparisons[cid]
                extra_result={**result,'comparisons':rows,'workflow_receipts':additional_receipts[cid],
                    'claim_id':cid,'shared_native_family_hash':result['plan_hash'],
                    'status':'PASS' if rows and all(row['status']=='PASS' for row in rows) else 'UNRESOLVED'}
                for row in additional_receipts[cid]:
                    if file_sha256(additional_paths[cid][row['run_id']])!=row['measurement_sha256']:
                        raise ValueError('Additional native observable changed before its retained family publication')
                extra_directory=result_directory/'observables'/cid
                extra_directory.mkdir(parents=True,exist_ok=False)
                extra_path=extra_directory/'convergence-family.json'
                atomic_json(extra_path,extra_result)
                atomic_text(extra_path.with_suffix('.sha256'),file_sha256(extra_path)+'\n')
                additional_families[cid]={'path':str(extra_path),'status':extra_result['status']}
            atomic_json(destination/'convergence-family.json', result)
            atomic_text(destination/'convergence-family.sha256',file_sha256(destination/'convergence-family.json')+'\n')
            state['result_generation'] = str(result_directory)
            budget_collection = None
            additional_budgets={}
            if request.get('error_budget_collection'):
                from core.research.measurements import publish_convergence_budget_collection
                state.update(phase='COLLECT_ERROR_BUDGETS',active_run=None)
                _publish_convergence_progress(destination,state)
                collection_request = deepcopy(request['error_budget_collection'])
                if 'family' in collection_request:
                    raise ValueError('Connected error-budget request receives its actual native family; do not substitute another family')
                collection_request['family'] = str(result_path)
                # One immutable campaign result owns one collection. A resumed
                # native family may reuse stage receipts but publishes its own
                # current collection; previous measured budgets remain retained.
                collection_destination = result_directory/'error-budgets'
                budget_collection = publish_convergence_budget_collection(collection_request,
                    collection_destination,request_base=request_base,execute=True,resume=False)
                state['error_budget_collection'] = {
                    'result_path':str(collection_destination/'collection-result.json'),
                    'science_acceptance':budget_collection['science_acceptance']}
            for claim in additional_claims:
                if not claim.get('error_budget_collection'):continue
                from core.research.measurements import publish_convergence_budget_collection
                cid=claim['claim_id'];collection_request=deepcopy(claim['error_budget_collection'])
                state.update(phase='COLLECT_ERROR_BUDGETS',active_observable=cid)
                _publish_convergence_progress(destination,state)
                if 'family' in collection_request:
                    raise ValueError('Additional observable budget receives its actual retained native measurement family')
                collection_request['family']=additional_families[cid]['path']
                collection_destination=result_directory/'observables'/cid/'error-budgets'
                additional_budgets[cid]=publish_convergence_budget_collection(collection_request,collection_destination,
                    request_base=request_base,execute=True,resume=False)
            if additional_families:
                state['additional_observable_families']=additional_families
                state['additional_observable_budget_acceptance']={cid:record['science_acceptance'] for cid,record in additional_budgets.items()}
                index={'schema':'shared-native-observable-family-index-v1','plan_hash':result['plan_hash'],
                    'primary_family':str(result_path),'additional_families':additional_families,
                    'additional_budgets':additional_budgets,'science_acceptance':'INDIVIDUAL_DECLARED_DOMAINS_ONLY'}
                atomic_json(result_directory/'observable-families.json',index)
                atomic_text(result_directory/'observable-families.sha256',file_sha256(result_directory/'observable-families.json')+'\n')
            state.update(execution_status='COMPLETE',phase='COMPLETE',active_run=None,active_observable=None,
                finished_unix=time.time(),control_status=result['status'],
                science_acceptance=result['science_acceptance'],result_sha256=file_sha256(destination/'convergence-family.json'))
            _publish_convergence_progress(destination,state)
            return {**result,'error_budget_collection':budget_collection,'additional_observable_families':additional_families,
                'additional_observable_budgets':additional_budgets}
        except BaseException as error:
            state.update(execution_status='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED',
                finished_unix=time.time(),failure={'type':type(error).__name__,'message':str(error)})
            _publish_convergence_progress(destination,state)
            raise


# Materialize native control inputs from frozen convergence specifications.
RESOURCE_FIELDS = {'mpi_ranks','omp_threads','memory_limit_gb','memory_gib','disk_gib','maximum_stage_count'}


def _read_json(path,memory_bytes):
    path = Path(path).resolve()
    if path.stat().st_size*8 > memory_bytes:
        raise MemoryError('Native configuration metadata exceeds declared workspace')
    before = file_sha256(path)
    value = json.loads(path.read_text())
    if file_sha256(path) != before:
        raise ValueError('Native configuration source changed while reading')
    return value


def _at(value,path):
    for key in path:
        if not isinstance(value,dict) or key not in value:
            raise ValueError('Explicit native/specification field is absent: '+'.'.join(path))
        value = value[key]
    return value


def _set(value,path,replacement):
    node = value
    for key in path[:-1]:
        if not isinstance(node,dict) or key not in node:
            raise ValueError('Native configuration binding cannot create a missing schema branch')
        node = node[key]
    if path[-1] not in node:
        raise ValueError('Native configuration binding cannot invent an unsupported solver parameter')
    node[path[-1]] = replacement


def validate_native_configuration(parameters):
    if parameters['mode'] not in {'json','gadget_parameter_text'}:
        raise ValueError('Native input must be explicit JSON or GADGET parameter text')
    memory = parameters['memory_bytes']
    if type(memory) is not int or memory <= 0:
        raise ValueError('Explicit native configuration byte allocation required')
    bindings = parameters['bindings']
    if not isinstance(bindings,dict) or 'phase_seed' not in bindings:
        raise ValueError('Native configuration must apply the actual planned IC phase seed')
    paths = set()
    for name,rows in bindings.items():
        if name not in {*CONTROLS,'phase_seed'} or not isinstance(rows,list) or not rows:
            raise ValueError('Native input bindings must identify actual supported control dimensions')
        for row in rows:
            path = row['native_path']
            if not isinstance(path,list) or not path or any(not isinstance(k,str) or not k for k in path):
                raise ValueError('Native input binding requires a complete key path')
            if parameters['mode'] == 'gadget_parameter_text' and len(path) != 1:
                raise ValueError('Native text parameter bindings name one actual scalar parameter')
            if set(path)&RESOURCE_FIELDS:
                raise ValueError('Scientific controls cannot silently alter resource allocations')
            marker = tuple(path)
            if marker in paths:
                raise ValueError('Multiple controls write the same native input parameter')
            paths.add(marker)
            scale = row.get('scale',1.)
            if name in {'phase_seed','sidm_stochastic_realization'} and scale != 1.:
                raise ValueError('Native random seed identity cannot undergo a unit conversion')
            source = row.get('specification_path')
            if source is not None and not (name == 'start_epoch' and source == ['cosmology_bundle']):
                raise ValueError('Only regenerated start-epoch controls may bind the alternate exact transfer bundle')
            if isinstance(scale,bool) or not isinstance(scale,(float,int)) or not math.isfinite(scale) or scale <= 0:
                raise ValueError('Native input unit scale must be explicit positive finite')
    return parameters


def configuration_input_bindings(parameters):
    validate_native_configuration(parameters)
    baseline = Path(parameters['baseline_input']).resolve()
    metadata = Path(parameters['variant_metadata']).resolve()
    if file_sha256(baseline) != parameters['baseline_sha256']:
        raise ValueError('Frozen baseline native input changed')
    record = _read_json(metadata,parameters['memory_bytes'])
    specification = Path(record['specification_path']).resolve()
    if file_sha256(specification) != record['specification_sha256']:
        raise ValueError('Native input specification changed before stage fingerprinting')
    return {str(baseline):file_sha256(baseline),str(metadata):file_sha256(metadata),
        str(specification):file_sha256(specification)}


def materialize_native_configuration(parameters):
    validate_native_configuration(parameters)
    sources = configuration_input_bindings(parameters)
    metadata = _read_json(parameters['variant_metadata'],parameters['memory_bytes'])
    if metadata.get('schema') != 'native-convergence-variant-metadata-v1':
        raise ValueError('Native configuration requires immutable convergence variant metadata')
    spec = _read_json(metadata['specification_path'],parameters['memory_bytes'])
    specification_path = str(Path(metadata['specification_path']).resolve())
    if spec != metadata['specification'] or file_sha256(specification_path) != metadata['specification_sha256']:
        raise ValueError('Planned native variant specification changed')
    sources[specification_path] = file_sha256(specification_path)
    baseline_path = Path(parameters['baseline_input']).resolve()
    if baseline_path.stat().st_size*8 > parameters['memory_bytes']:
        raise MemoryError('Native baseline input exceeds declared materialization workspace')
    if parameters['mode'] == 'json':
        baseline = _read_json(baseline_path,parameters['memory_bytes'])
    else:
        baseline = {}
        for line in baseline_path.read_text().splitlines():
            row = line.split('%',1)[0].split('#',1)[0].strip()
            if not row:
                continue
            key,*values = row.split()
            if not values or key in baseline:
                raise ValueError('Native parameter text requires distinct explicitly valued fields')
            baseline[key] = ' '.join(values)
    if not isinstance(baseline,dict):
        raise ValueError('Native baseline input must be a field map')
    configuration = deepcopy(baseline)
    required = ['phase_seed',*metadata['planned_control_dimensions']]
    interactions = spec.get('dark_matter',{}).get('interactions',{})
    if interactions.get('kind') == 'elastic_sidm' and 'sidm_stochastic_realization' not in required:
        required.append('sidm_stochastic_realization')
    if any(name not in parameters['bindings'] for name in required):
        raise ValueError('A planned phase/operator/numerical control has no native input binding')
    if set(parameters['bindings'])-set(required):
        raise ValueError('Native materialization cannot alter undeclared control dimensions')
    applied = []
    for name in required:
        value = _at(spec,('seed',) if name == 'phase_seed' else CONTROLS[name])
        if name in {'phase_seed','sidm_stochastic_realization'} and (type(value) is not int or value < 0 or name == 'sidm_stochastic_realization' and value >= 2**64):
            raise ValueError('Native phase/operator seeds must retain exact declared integers')
        for row in parameters['bindings'][name]:
            scale = row.get('scale',1.)
            native = deepcopy(_at(spec,row['specification_path']) if row.get('specification_path') else value)
            if scale != 1.:
                if isinstance(native,bool) or not isinstance(native,(float,int)):
                    raise ValueError('Unit conversion requires a scalar numeric native parameter')
                native *= scale
            if isinstance(native,float) and not math.isfinite(native):
                raise ValueError('Native parameter conversion overflows its domain')
            if parameters['mode'] == 'gadget_parameter_text':
                if isinstance(native,(dict,list)) or native is None:
                    raise ValueError('Scalar native parameter cannot serialize a structured control')
                native = str(int(native)) if isinstance(native,bool) else str(native)
                if any(c in native for c in '\n\r%#'):
                    raise ValueError('Native parameter values cannot inject extra text fields')
            previous = deepcopy(_at(configuration,row['native_path']))
            _set(configuration,row['native_path'],native)
            applied.append({'control':name,'native_path':row['native_path'],
                'specification_value':value,'native_value':native,'baseline_value':previous,'scale':scale})
    if parameters['mode'] == 'json':
        rendered = json.dumps(configuration,indent=2,sort_keys=True,allow_nan=False)+'\n'
    else:
        rendered = '\n'.join(key+' '+str(value) for key,value in configuration.items())+'\n'
    if len(rendered.encode())*8 > parameters['memory_bytes']:
        raise MemoryError('Materialized native control input exceeds its workspace')
    for path,sha in sources.items():
        if file_sha256(path) != sha:
            raise ValueError('Native control input source changed during materialization')
    return rendered, {'schema':'native-control-configuration-v1','run_id':metadata['run_id'],
        'common_experiment_signature':metadata['common_signature'],'common_model_family_hash':metadata['common_model_family_hash'],
        'changed_controls':metadata['changed_controls'],'applied_native_bindings':applied,
        'baseline_sha256':parameters['baseline_sha256'],'variant_specification_sha256':metadata['specification_sha256'],
        'rendered_configuration_hash':canonical_digest(configuration),'source_hashes':sources,
        'execution_status':'COMPLETE','science_acceptance':'INPUT_MATERIALIZATION_ONLY'}


def native_configuration_runner(parameters,context):
    if parameters['memory_bytes'] > context.resources['memory_gib']*1024**3:
        raise MemoryError('Native input materialization exceeds frozen workflow memory')
    rendered,receipt = materialize_native_configuration(parameters)
    filename = 'native-control-input.json' if parameters['mode'] == 'json' else 'native-control-input.txt'
    path = context.generation/filename
    atomic_text(path,rendered)
    receipt['native_input_path'] = str(path);receipt['native_input_sha256'] = file_sha256(path)
    atomic_json(context.generation/'native-control-configuration.json',receipt)
    return {'measurements':{'applied_native_control_fields':len(receipt['applied_native_bindings'])}}


def attach_native_configuration_stage(workflow_template,settings):
    """Replace the evolution argument with an actual variant-specific input product."""
    from core.workflow import safe_id
    workflow = deepcopy(workflow_template)
    native_id = safe_id(settings['native_stage_id'])
    sid = safe_id(settings.get('configuration_stage_id','native_configuration'))
    if any(row['stage_id'] == sid for row in workflow['stages']):
        raise ValueError('Native control configuration stage identity already exists')
    native = next((row for row in workflow['stages'] if row['stage_id'] == native_id),None)
    if native is None or native['kind'] not in {'public_command','command'}:
        raise ValueError('Native input materialization requires an actual public/native command producer')
    parameters = {key:deepcopy(settings[key]) for key in ('mode','baseline_sha256','bindings','memory_bytes')}
    parameters.update(baseline_input={'input':settings['baseline_input']},variant_metadata={'input':'{variant_metadata}'})
    validate_native_configuration(parameters)
    argument_key = 'arguments' if native['kind'] == 'public_command' else 'argv'
    arguments = native['parameters'][argument_key]
    index = settings['input_argument_index']
    if type(index) is not int or not 0 <= index < len(arguments):
        raise ValueError('Native configuration must select its actual input-file argument')
    if native['kind'] == 'public_command' and native['parameters']['command'] == 'gadget':
        if parameters['mode'] != 'json' or index < 1 or arguments[index-1] != '--answers':
            raise ValueError('Public GADGET input binding must replace its --answers JSON')
    filename = 'native-control-input.json' if parameters['mode'] == 'json' else 'native-control-input.txt'
    arguments[index] = {'stage':sid,'artifact':filename}
    native['depends_on'] = list(dict.fromkeys([*native.get('depends_on',[]),sid]))
    workflow['stages'].insert(0,{'stage_id':sid,'kind':'native_control_configuration','parameters':parameters,
        'outputs':{'native_control_input':filename,'native_control_configuration':'native-control-configuration.json'}})
    return workflow


def preflight_configuration_stages(plan,context):
    """Inspect every produced native input before evolution consumes any variant."""
    from core.workflow import resolve_values
    configurations = {}
    for row in plan['stages']:
        if row['kind'] != 'native_control_configuration':
            continue
        parameters = resolve_values(row['parameters'],context)
        allocation = row.get('resources',context.resources)
        if parameters['memory_bytes'] > allocation['memory_gib']*1024**3:
            raise MemoryError('Native configuration materialization exceeds workflow memory')
        rendered,receipt = materialize_native_configuration(parameters)
        configurations[row['stage_id']] = (parameters,rendered,receipt)
    for row in plan['stages']:
        if row['kind'] != 'public_command' or row['parameters']['command'] != 'gadget':
            continue
        arguments = row['parameters'].get('arguments',[])
        if '--answers' not in arguments:
            continue
        position = arguments.index('--answers')+1
        if position >= len(arguments):
            raise ValueError('Native --answers input argument is missing')
        reference = arguments[position]
        if not isinstance(reference,dict) or reference.get('stage') not in configurations:
            continue
        parameters,rendered,receipt = configurations[reference['stage']]
        if parameters['mode'] != 'json' or reference.get('artifact') != 'native-control-input.json':
            raise ValueError('Native public evolution must consume its actual materialized JSON answer product')
        answers = json.loads(rendered)
        allocation = row.get('resources',context.resources)
        for key,resource in (('mpi_ranks','mpi_ranks'),('omp_threads','omp_threads'),('memory_limit_gb','memory_gib')):
            if answers.get(key) != allocation[resource]:
                raise ValueError('Materialized native answers differ from the allocated '+resource)
        from core.research.capabilities import validate_native_answers
        validate_native_answers(answers,allocation)
    return {sid:receipt for sid,(_,_,receipt) in configurations.items()}


def convergence_main():
    import argparse
    p = argparse.ArgumentParser(description='Evolve and measure controlled convergence variants')
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    q = json.loads(a.campaign.read_text())
    print(json.dumps(execute_and_measure(**q, destination=a.output, request_base=a.campaign.parent, resume=a.resume), indent=2))


# Measured convergence products for every declared scientific observable.
OBSERVABLE_CONTROLS={'mass_resolution','softening','timestep','output_cadence','start_redshift','finder_tree_variation'}


def compare_observable_convergence(records,request):
    from core.research.errors import _finite as finite, _psd, _support_mask
    if request['control'] not in OBSERVABLE_CONTROLS:raise ValueError('Unknown required convergence dimension')
    if not records:raise ValueError('Actual measured convergence products required')
    for key in ('absolute_tolerance','relative_tolerance','covariance_tolerance'):
        if isinstance(request[key], bool) or not np.isfinite(request[key]) or request[key]<0:raise ValueError('Finite nonnegative convergence tolerances required')
    if len({q['run_id'] for q in records})!=len(records):raise ValueError('Unique convergence run identities required')
    reference=next((q for q in records if q['run_id']==request['reference_run']),None)
    if reference is None:
        raise ValueError('The requested reference run is absent from the measured records')
    signature=reference['common_experiment_signature'];value=np.asarray(reference['value'],float);rows=[];difference=[];support=[];covariances=[]
    if value.ndim != 1 or not value.size:
        raise ValueError('Convergence values require a nonempty one-dimensional observable')
    for record in records:
        if record['common_experiment_signature']!=signature or record['observable_definition_hash']!=reference['observable_definition_hash'] or record['units']!=reference['units']:
            raise ValueError('Convergence contrast changes the observable, epoch, initial phases or unrelated physics')
        expected={request['control']} if record['run_id']!=reference['run_id'] else set()
        if set(record['changed_controls'])!=expected:raise ValueError('Convergence run must change exactly the declared control')
        if not record['source_hashes']:raise ValueError('Measured convergence source provenance required')
        y=np.asarray(record['value'],float)
        if y.shape!=value.shape:raise ValueError('Convergence observable shape differs')
        mask=_support_mask(record['supported'],value.shape)&_support_mask(reference['supported'],value.shape)
        mask&=np.isfinite(y)&np.isfinite(value)
        if mask.shape!=value.shape:raise ValueError('Convergence masks must share physical measurement bins')
        current_covariance = _psd(record['covariance'],value.size,request['covariance_tolerance'],'observable covariance')
        reference_covariance = _psd(reference['covariance'],value.size,request['covariance_tolerance'],'reference covariance')
        cov=current_covariance+reference_covariance
        if record['run_id']==reference['run_id']:cov=np.zeros_like(cov)
        elif record.get('paired_cross_covariance') is not None:
            cross=finite(record['paired_cross_covariance'],cov.shape)
            _psd(np.block([[current_covariance,cross],[cross.T,reference_covariance]]),
                 2*value.size,request['covariance_tolerance'],'paired observable covariance')
            cov-=cross+cross.T
        elif record.get('independent_of_reference') is not True:
            raise ValueError('Paired observable contrasts require cross covariance or explicit independence')
        cov = _psd(cov,value.size,request['covariance_tolerance'],'observable difference covariance')
        delta=y-value;allowed=request['absolute_tolerance']+request['relative_tolerance']*abs(value)
        rows.append({'run_id':record['run_id'],'control_value':record.get('control_value'),'supported_fraction':float(mask.mean()),'within_requested_tolerance':bool(np.all(abs(delta[mask])<=allowed[mask])) if mask.any() else False,'source_hashes':record['source_hashes']})
        difference.append(delta);support.append(mask);covariances.append(cov)
    return ({'schema':'observable-convergence-v1','observable':request['observable'],'control':request['control'],'reference_run':reference['run_id'],'rows':rows,
        'scope':'measured controlled differences and recorded covariance; passing one control is not full physical acceptance','science_acceptance':'NOT_ACCEPTED',
        'units':{'observable_difference':reference['units'],'difference_covariance':'('+reference['units']+')^2'}},
        {'observable_difference':np.asarray(difference),'difference_covariance':np.asarray(covariances),'comparison_supported':np.asarray(support)})


# Native convergence products


def _allocation(memory):
    if type(memory) not in (int, float) or not math.isfinite(memory) or memory <= 0:
        raise ValueError('Positive finite convergence workspace required')


def _owned(path, receipt_path, kind, pins):
    from core.workflow import verify_completed_receipt
    receipt_path = Path(receipt_path).resolve()
    receipt = verify_completed_receipt(receipt_path)
    path = Path(path).resolve(); root = Path(receipt['generation']).resolve()
    if receipt['kind'] != kind or not path.is_relative_to(root):
        raise ValueError('Convergence input requires its original completed producer')
    sha = file_sha256(path)
    if receipt['files_sha256'].get(path.relative_to(root).as_posix()) != sha:
        raise ValueError('Convergence product is outside its completed producer inventory')
    _merge(pins, {str(path): sha, str(receipt_path): file_sha256(receipt_path),
        str(receipt_path.with_suffix('.sha256')): file_sha256(receipt_path.with_suffix('.sha256'))})
    _merge(pins, receipt['source_hashes'])
    _merge(pins, receipt['input_hashes'])
    return receipt


def convergence_figure_inputs(parameters):
    memory = parameters['memory_gib']; _allocation(memory)
    request = parameters['request']; receipts = parameters['measurement_receipts']
    if len(receipts) != len(request['measurements']) or len(receipts) < 2:
        raise ValueError('Each native comparison measurement requires its producer receipt')
    pins = {}; bins_bytes = 0
    for reference, receipt_path in zip(request['measurements'], receipts):
        path = Path(reference['path']).resolve()
        if path.stat().st_size * 32 > memory * 1024**3 / 8:
            raise MemoryError('Convergence metadata exceeds allocation')
        if file_sha256(path) != reference['sha256']:
            raise ValueError('Native convergence measurement changed')
        _owned(path, receipt_path, 'convergence_measurement', pins)
        seal = path.with_name(path.name + '.sha256')
        if seal.read_text().strip() != reference['sha256']:
            raise ValueError('Native convergence measurement seal changed')
        _merge(pins, {str(seal): file_sha256(seal)})
        record = json.loads(path.read_text())
        if record.get('schema') != 'native-convergence-measurement-v1' or record.get('execution_status') != 'COMPLETE':
            raise ValueError('Actual completed native measurement required')
        native = Path(record['native_realization_receipt']).resolve()
        from core.workflow import verify_completed_receipt
        evolution = verify_completed_receipt(native)
        if evolution['kind'] != 'public_command':
            raise ValueError('Native convergence requires original native evolution authority')
        _merge(pins, {str(native): file_sha256(native)})
        _merge(pins, record['source_hashes'])
        bins_bytes += len(record['bins'])**2 * 128
    if bins_bytes > memory * 1024**3:
        raise MemoryError('Native comparison covariance exceeds allocation')
    for path, sha in pins.items():
        if file_sha256(path) != sha:
            raise ValueError('Native convergence source changed')
    return pins


def convergence_figure_runner(parameters, context):
    from core.flagship_products import export_convergence_figures
    if parameters['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('Convergence figures exceed stage allocation')
    pins = convergence_figure_inputs(parameters)
    result = export_convergence_figures(parameters['request'], context.generation / 'figures',
        request_base=context.base, memory_gib=parameters['memory_gib'])
    for path, sha in pins.items():
        if file_sha256(path) != sha:
            raise ValueError('Native convergence source changed during export')
    return {'measurements': {'exported_figures': len(result['figures']),
        'science_acceptance': 'NOT_ACCEPTED'}}


def convergence_phase_inputs(parameters):
    """Require actual common white noise and the same traced parent cohort."""
    from core.zoom import verify_generated_zoom, verify_traced_region, _configuration, _music_noise_files
    _allocation(parameters['memory_gib']); control = parameters['control']
    if control not in {'mass_resolution', 'softening', 'timestep', 'output_cadence', 'start_epoch', 'finder'}:
        raise ValueError('Unknown native convergence phase control')
    pins = {}; noises = {}; memberships = {}; ics = {}; parent_levels = {}
    for label in ('baseline', 'current'):
        path = Path(parameters[label + '_generation']).resolve()
        _owned(path, parameters[label + '_receipt'], 'public_command', pins)
        if path.stat().st_size * 32 > parameters['memory_gib'] * 1024**3 / 8:
            raise MemoryError('Native IC phase metadata exceeds allocation')
        report = verify_generated_zoom(path)
        if (report.get('publication_contract') != 'native-ic-generation-v1'
                or report.get('linear_audit_status') != 'PASS'
                or report.get('phase_audit', {}).get('status') != 'PASS'):
            raise ValueError('Native convergence requires actual accepted IC phase audits')
        _merge(pins, {str(path.parent / name): sha for name, sha in report['artifacts'].items()})
        seal = path.with_name(path.stem + '.sha256')
        _merge(pins, {str(seal): file_sha256(seal)})
        for name, sha in report['parent_phase_inputs'].items():
            if file_sha256(name) != sha: raise ValueError('Native parent phase input changed')
        _merge(pins, report['parent_phase_inputs'])
        region_ref = report['traced_region']; region_path = Path(region_ref['path']).resolve()
        if file_sha256(region_path) != region_ref['sha256']:
            raise ValueError('Native traced target changed')
        region = verify_traced_region(region_path)
        selection = region_path.parent / 'selected_particles.hdf5'
        if file_sha256(selection) != region['selected_particles_sha256']:
            raise ValueError('Native traced parent membership changed')
        memberships[label] = region['selected_particles_sha256']
        _merge(pins, {str(region_path): region_ref['sha256'], str(selection): memberships[label]})
        cfg_path = path.parent / 'music.conf'; cfg = _configuration(cfg_path)
        parent_levels[label] = cfg.getint('setup', 'levelmin')
        noise_files = _music_noise_files(cfg, cfg_path)
        noise_hashes = {str(noise): file_sha256(noise) for noise in set(noise_files.values())}
        noises[label] = {level: noise_hashes[str(noise)] for level, noise in noise_files.items()}
        _merge(pins, noise_hashes)
        ics[label] = report['artifacts']['ics_gadget.hdf5']
    if memberships['baseline'] != memberships['current']:
        raise ValueError('Convergence runs select different exact traced parent cohorts')
    common = sorted(set(noises['baseline']) & set(noises['current']))
    if (parent_levels['baseline'] != parent_levels['current'] or parent_levels['baseline'] not in common
            or any(noises['baseline'][level] != noises['current'][level] for level in common)):
        raise ValueError('Convergence runs lack identical actual common white-noise levels')
    if control not in {'mass_resolution', 'start_epoch'} and ics['baseline'] != ics['current']:
        raise ValueError('Non-IC numerical convergence must reuse identical actual initial conditions')
    for path, sha in pins.items():
        if file_sha256(path) != sha: raise ValueError('Convergence phase source changed')
    return pins, common


def convergence_phase_runner(parameters, context):
    if parameters['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('Convergence phase audit exceeds stage allocation')
    pins, common = convergence_phase_inputs(parameters)
    atomic_json(context.generation / 'phase-pair.json', {
        'schema': 'native-convergence-phase-pair-v1', 'status': 'PASS',
        'control': parameters['control'], 'common_frozen_levels': common,
        'target_match': 'IDENTICAL_TRACED_PARENT_COHORT_BYTES', 'source_hashes': pins,
        'science_acceptance': 'NOT_ACCEPTED'})
    return {'measurements': {'common_frozen_levels': len(common), 'phase_pair': 'PASS'}}
