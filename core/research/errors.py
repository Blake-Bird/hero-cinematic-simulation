"""Measured error budgets, signed residuals and correlated native ensembles.

All scientific tolerances and input units are declared by the caller.
"""
from __future__ import annotations

import json
import numpy as np
import math
from pathlib import Path
from core.foundation import atomic_json, atomic_text, canonical_digest, file_sha256
from core.native import native_run_lock
from core.research.convergence import compare_campaign, CONTROLS
from copy import deepcopy




# Correlated observable error budgets with frozen dimensional requirements.
ERROR_COMPONENTS = frozenset({
    'integration', 'mass_resolution', 'force_resolution', 'output_cadence',
    'gravity_force_accuracy','gravity_tree_opening',
    'start_epoch', 'finder', 'potential_reconstruction', 'subgrid_model',
    'observer_selection', 'estimator', 'finite_volume', 'initial_conditions',
    'sidm_kernel', 'sidm_timestep', 'sidm_angular_cdf', 'sidm_frequent_approximation',
    'sidm_split_angle', 'sidm_stochastic_realization', 'stream_heating', 'baryonic_model'})


def _finite(value, shape=None, name='value'):
    x = np.asarray(value, float)
    if shape is not None and x.shape != shape or not np.isfinite(x).all():
        raise ValueError('Invalid finite ' + name)
    return x


def _support_mask(value, shape):
    mask = np.asarray(value)
    if mask.dtype.kind != 'b' or mask.shape != shape:
        raise ValueError('Support masks must contain exact booleans at the declared measurement bins')
    return mask


def _psd(value, size, tolerance, name):
    x = _finite(value, (size, size), name)
    if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError('Finite nonnegative covariance tolerance required')
    if not np.allclose(x, x.T, rtol=0, atol=tolerance):
        raise ValueError('Asymmetric ' + name)
    if size and np.linalg.eigvalsh((x+x.T)/2).min() < -tolerance:
        raise ValueError('Non-PSD ' + name)
    return (x+x.T)/2


def _verify_sources(record):
    sources = record.get('source_hashes', {})
    if not sources:
        raise ValueError('Measured uncertainty requires exact source hashes')
    for path, expected in sources.items():
        if file_sha256(path) != expected:
            raise ValueError('Uncertainty source changed: ' + path)


def determine_supported_parameter_radius_epoch_domain(budget):
    """Publish exact supported cells and adjacent measured radial intervals.

    Equal descriptor fields define a group. Only exactly abutting accepted
    intervals are joined. An unmeasured epoch, parameter point, or radial gap
    is never filled by a bounding box or an interpolation assumption.
    """
    rows = budget['bins']
    identities = [canonical_digest(row['bin']) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('Repeated error-budget physical domain cell')
    supported,accepted,excluded = [],[],[]
    groups = {}
    for index,row in enumerate(rows):
        if type(row['supported']) is not bool or type(row['accepted']) is not bool or row['accepted'] and not row['supported']:
            raise ValueError('Physical domain needs exact coherent supported/accepted flags')
        cell = row['bin']
        if row['supported']:
            supported.append(cell)
        if not row['accepted']:
            excluded.append({'bin':cell,'bin_index':index,
                'reason':'measured error exceeds frozen requirement' if row['supported'] else 'native component support or evidence absent'})
            continue
        accepted.append(cell)
        if not isinstance(cell,dict):
            continue
        keys = [name for name in ('radius_proper_kpc','radius_kpc') if name in cell]
        if len(keys) != 1:
            continue
        key = keys[0]
        interval = cell[key]
        if (not isinstance(interval,list) or len(interval) != 2
                or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in interval)
                or interval[0] < 0 or interval[0] >= interval[1]):
            raise ValueError('Accepted radial cells require explicit finite increasing physical interval bounds')
        context = {name:value for name,value in cell.items() if name != key}
        signature = canonical_digest({'coordinate':key,'context':context})
        groups.setdefault(signature,{'radius_coordinate':key,'fixed_descriptor':context,'cells':[]})['cells'].append((interval,index))
    regions = []
    for group in groups.values():
        current = None
        for interval,index in sorted(group['cells'],key=lambda item:(item[0][0],item[0][1])):
            if current is not None and interval[0] < current['radius_interval'][1]:
                raise ValueError('Accepted radial bins overlap within one fixed epoch/parameter descriptor')
            if current is not None and interval[0] == current['radius_interval'][1]:
                current['radius_interval'][1] = interval[1]
                current['measured_bin_indices'].append(index)
            else:
                current = {'radius_coordinate':group['radius_coordinate'],'fixed_descriptor':group['fixed_descriptor'],
                    'radius_interval':list(interval),'measured_bin_indices':[index]}
                regions.append(current)
    return {'schema':'measured-error-support-domain-v1','requirement_hash':budget['requirement_hash'],
        'supported_cells':supported,'accepted_cells':accepted,'excluded_cells':excluded,
        'connected_radial_regions':regions,'interpolation_authorized':False,'extrapolation_authorized':False,
        'scope':'exact measured epoch/parameter descriptors; only adjacent accepted radial cells may be grouped'}


def evaluate_error_budget(request, measurements, joint_covariance=None):
    """Sum correlated random errors; conservatively add systematic envelopes.

    joint_covariance is component-major, shape (components*bins)^2. Supplying
    only component covariance requires an explicit independent_components
    statement. That statement is recorded as an assumption, never inferred.

    If e is the stacked component error, S = [I I ... I] sums components:
    C_total = S C_joint S.T. Add nonnegative systematic bounds directly,
    then s times the random standard deviation. This preserves correlation
    and avoids treating a systematic envelope as a random variance. Biases,
    reference values and absolute tolerances share request['units']; covariance
    entries have those units squared, and relative tolerance is dimensionless.
    """
    if not measurements:
        raise ValueError('At least one measured error component required')
    if request.get('schema') != 'observable-error-requirement-v1':
        raise ValueError('Frozen observable-error-requirement-v1 required')
    observable, units = request['observable'], request['units']
    reference = _finite(request['reference_value'], name='reference observable').reshape(-1)
    n = reference.size
    if n == 0:
        raise ValueError('An empty physical selection has no accepted error domain')
    if not isinstance(request.get('bins'), list) or len(request['bins']) != n:
        raise ValueError('Every observable entry requires an explicit corresponding domain bin')
    if len({canonical_digest(cell) for cell in request['bins']}) != n:
        raise ValueError('Error requirement repeats an epoch/radius/parameter cell')
    floor = np.broadcast_to(_finite(request['dimensional_floor'], name='dimensional floor'), (n,))
    absolute = np.broadcast_to(_finite(request['absolute_tolerance'], name='absolute tolerance'), (n,))
    relative = np.broadcast_to(_finite(request['relative_tolerance'], name='relative tolerance'), (n,))
    if np.any(floor <= 0) or np.any(absolute < 0) or np.any(relative < 0):
        raise ValueError('Positive dimensional floors and nonnegative tolerances required')
    sigma_multiplier = request['sigma_multiplier']
    if isinstance(sigma_multiplier, bool) or not isinstance(sigma_multiplier, (int, float)) or not math.isfinite(sigma_multiplier) or sigma_multiplier <= 0:
        raise ValueError('Positive frozen uncertainty multiplier required')
    tolerance = request['covariance_tolerance']
    labels = [q['component'] for q in measurements]
    if len(set(labels)) != len(labels) or any(x not in ERROR_COMPONENTS for x in labels):
        raise ValueError('Distinct recognized error components required')
    required = set(request['required_components'])
    if not required or not required <= ERROR_COMPONENTS:
        raise ValueError('Explicit mandatory error components required')
    masks, biases, blocks, domains = [], [], [], []
    for q in measurements:
        if q['observable'] != observable or q['units'] != units or q['requirement_hash'] != canonical_digest(request):
            raise ValueError('Measured component does not match the frozen requirement')
        _verify_sources(q)
        mask = _support_mask(q['supported'], (n,))
        if mask.shape != (n,):
            raise ValueError('Per-bin component support required')
        bias = _finite(q['bias_bound'], (n,), 'systematic bias bound')
        if np.any(bias < 0):
            raise ValueError('Systematic bias envelopes must be nonnegative')
        blocks.append(_psd(q['covariance'], n, tolerance, 'component covariance'))
        masks.append(mask)
        biases.append(bias)
        domain = q['domain']
        if not isinstance(domain, list):
            raise ValueError('Measured component domain must explicitly enumerate its actual bin descriptors')
        for index in np.flatnonzero(mask):
            if request['bins'][index] not in domain:
                raise ValueError('Component marks an unmeasured radius/epoch/parameter domain as supported')
        domains.append(domain)
    missing = sorted(required-set(labels))
    k = len(labels)
    if joint_covariance is None:
        if k > 1 and request.get('independent_components') is not True:
            raise ValueError('Full cross-component covariance or explicit independence required')
        joint = np.zeros((k*n, k*n))
        for i, block in enumerate(blocks):
            joint[i*n:(i+1)*n, i*n:(i+1)*n] = block
        covariance_assumption = 'explicit component independence' if k > 1 else 'single component'
    else:
        joint = _psd(joint_covariance, k*n, tolerance, 'joint component covariance')
        for i, block in enumerate(blocks):
            if not np.allclose(joint[i*n:(i+1)*n, i*n:(i+1)*n], block, rtol=0, atol=tolerance):
                raise ValueError('Joint diagonal block differs from measured component covariance')
        covariance_assumption = 'measured cross-component covariance supplied'
    sum_operator = np.tile(np.eye(n), (1, k))
    covariance = _psd(sum_operator@joint@sum_operator.T, n, tolerance, 'combined observable covariance')
    systematic = np.sum(biases, axis=0)
    uncertainty = systematic + sigma_multiplier*np.sqrt(np.maximum(np.diag(covariance), 0))
    allowed = absolute + relative*np.maximum(np.abs(reference), floor)
    supported = np.logical_and.reduce(masks)
    if missing:
        supported[:] = False
    accepted = supported & (uncertainty <= allowed)
    bins = request['bins']
    if len(bins) != n:
        raise ValueError('Explicit epoch/radius/parameter bin domain required')
    rows = []
    for i, domain in enumerate(bins):
        rows.append({'bin': domain, 'supported': bool(supported[i]), 'accepted': bool(accepted[i]),
                     'error_bound': float(uncertainty[i]), 'allowed_error': float(allowed[i]),
                     'systematic_bound': float(systematic[i]),
                     'random_sigma': float(np.sqrt(max(covariance[i,i], 0)))})
    refinements = []
    for i in np.flatnonzero(~accepted):
        candidates = sorted(zip(labels, np.asarray(biases)[:, i] + sigma_multiplier*np.sqrt(np.maximum([b[i,i] for b in blocks], 0))),
                            key=lambda x: (-x[1], x[0]))
        refinements.append({'bin': bins[i], 'components': [p[0] for p in candidates],
                            'reason': 'missing controls' if missing else 'unsupported bin' if not supported[i] else 'error budget exceeded'})
    return {'schema': 'observable-error-budget-v1', 'requirement_hash': canonical_digest(request),
        'observable': observable, 'units': units, 'components': labels, 'missing_components': missing,
        'covariance_assumption': covariance_assumption, 'joint_covariance': joint.tolist(),
        'combined_covariance': covariance.tolist(), 'component_domains': domains, 'bins': rows,
        'accepted_domain': [bins[i] for i in np.flatnonzero(accepted)],
        'science_acceptance': 'ACCEPTED_DECLARED_BINS' if bool(accepted.all()) else 'UNRESOLVED',
        'refinement_requests': refinements, 'source_hashes': [q['source_hashes'] for q in measurements]}


def estimate_correlated_error_components(samples, *, names, units, covariance_tolerance):
    """Ensemble paired differences: shape realizations x components x bins.

    Distinct realizations are required by the caller; matching phases within
    one realization retain cross-component correlation. No bootstrap replaces
    an independent native ensemble.

    C = sum_i (x_i - mean)(x_i - mean).T / (N - 1), in units squared.
    This estimates variation of one realization; covariance of the ensemble
    mean would instead be C/N and is not the quantity returned here.
    """
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names) or not isinstance(units, str) or not units.strip()):
        raise ValueError('Distinct component names and explicit common observable units required')
    values = _finite(samples, name='measured ensemble residuals')
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] != len(names):
        raise ValueError('At least two measured realizations x components x bins required')
    flat = values.reshape(values.shape[0], -1)
    covariance = np.atleast_2d(np.cov(flat, rowvar=False, ddof=1))
    _psd(covariance, flat.shape[1], covariance_tolerance, 'ensemble covariance')
    return {'components': list(names), 'units': units, 'mean_residual': values.mean(axis=0).tolist(),
            'joint_covariance': covariance.tolist(), 'independent_realizations': values.shape[0],
            'covariance_target': 'single realization residual, not covariance of the ensemble mean'}


def publish_budget_and_domain(destination, request, measurements, *, joint_covariance=None):
    from pathlib import Path
    from core.foundation import atomic_json
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Error budgets are immutable science generations')
    destination.mkdir(parents=True)
    result = evaluate_masked_error_budget(request, measurements, joint_covariance)
    atomic_json(destination/'error-budget.json', result)
    atomic_json(destination/'frozen-requirement.json', request)
    atomic_json(destination/'supported-domain.json',result['supported_parameter_radius_epoch_domain'])
    return result


def evaluate_masked_error_budget(request, measurements, joint_covariance=None):
    """Preserve failed native bins without inventing finite uncertainty values.

    Compute covariance only on the common actually supported bin subspace.
    Published covariance matrices carry explicit original-bin indices; unknown
    rows are represented by null scalar bounds rather than zero uncertainties.
    """
    from copy import deepcopy
    reference = np.asarray(request['reference_value'], float).reshape(-1)
    n = len(reference)
    if not n or len(request['bins']) != n or not measurements:
        raise ValueError('Nonempty measured error components and corresponding physical bins required')
    if request.get('schema') != 'observable-error-requirement-v1':
        raise ValueError('Frozen observable error requirement required')
    for label in ('observable','units'):
        if not isinstance(request.get(label),str) or not request[label]:
            raise ValueError('Explicit physical observable and units required')
    # Frozen requirement metadata remains finite even when native measurements
    # fail; unknown measured biases/covariances live behind support masks.
    canonical_digest(request)
    for key in ('dimensional_floor','absolute_tolerance','relative_tolerance'):
        values = np.broadcast_to(_finite(request[key],name=key),(n,))
        if np.any(values <= 0) if key == 'dimensional_floor' else np.any(values < 0):
            raise ValueError('Invalid predeclared dimensional floor or error tolerance')
    multiplier = request['sigma_multiplier']
    if isinstance(multiplier,bool) or not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Explicit positive finite uncertainty multiplier required')
    labels = [q['component'] for q in measurements]
    if len(set(labels)) != len(labels) or not set(labels) <= ERROR_COMPONENTS:
        raise ValueError('Distinct recognized measured error components required')
    required = set(request['required_components'])
    if not required or not required <= ERROR_COMPONENTS:
        raise ValueError('Explicit recognized mandatory error components required')
    missing = sorted(required-set(labels))
    supported = np.isfinite(reference)
    masks = []
    for q in measurements:
        if q['requirement_hash'] != canonical_digest(request) or q['observable'] != request['observable'] or q['units'] != request['units']:
            raise ValueError('Masked component does not bind the frozen physical requirement')
        _verify_sources(q)
        mask = _support_mask(q['supported'], (n,))
        bias = np.asarray(q['bias_bound'], float)
        covariance = np.asarray(q['covariance'], float)
        if bias.shape != (n,) or covariance.shape != (n,n):
            raise ValueError('Masked component dimensions differ from declared observable bins')
        if not np.isfinite(bias[mask]).all() or np.any(bias[mask] < 0):
            raise ValueError('Supported component biases must be finite nonnegative envelopes')
        positions = np.flatnonzero(mask)
        _psd(covariance[np.ix_(positions, positions)], len(positions), request['covariance_tolerance'], 'supported component covariance')
        if not isinstance(q['domain'], list) or any(request['bins'][i] not in q['domain'] for i in positions):
            raise ValueError('Component support lies outside its actual measured domains')
        masks.append(mask)
        supported &= mask
    if missing:
        supported[:] = False
    indices = np.flatnonzero(supported)
    result = None
    if len(indices):
        projected = deepcopy(request)
        projected['reference_value'] = reference[indices].tolist()
        projected['bins'] = [request['bins'][i] for i in indices]
        for key in ('dimensional_floor', 'absolute_tolerance', 'relative_tolerance'):
            values = np.broadcast_to(np.asarray(request[key],float), (n,))
            projected[key] = values[indices].tolist()
        rows = []
        for q in measurements:
            covariance = np.asarray(q['covariance'], float)
            rows.append({**q, 'requirement_hash':canonical_digest(projected),
                'bias_bound':np.asarray(q['bias_bound'],float)[indices].tolist(),
                'covariance':covariance[np.ix_(indices,indices)].tolist(),
                'supported':[True]*len(indices)})
        projected_joint = None
        if joint_covariance is not None:
            full = np.asarray(joint_covariance, float)
            k = len(measurements)
            if full.shape != (k*n,k*n):
                raise ValueError('Masked joint covariance dimensions do not match component-major bins')
            component_indices = np.concatenate([index*n+indices for index in range(k)])
            projected_joint = full[np.ix_(component_indices,component_indices)]
        result = evaluate_error_budget(projected, rows, projected_joint)
    rows_by_index = {int(index):row for index,row in zip(indices, result['bins'] if result else [])}
    full_rows = []
    refinements = list(result['refinement_requests']) if result else []
    for i, domain in enumerate(request['bins']):
        if i in rows_by_index:
            full_rows.append(rows_by_index[i])
        else:
            full_rows.append({'bin':domain,'supported':False,'accepted':False,
                'error_bound':None,'allowed_error':None,'systematic_bound':None,'random_sigma':None})
            refinements.append({'bin':domain,'components':list(dict.fromkeys(missing+labels)),
                'reason':'missing required controls' if missing else 'unsupported or unmeasured native bin'})
    report = {'schema':'observable-error-budget-v1','requirement_hash':canonical_digest(request),
        'observable':request['observable'],'units':request['units'],'components':labels,'missing_components':missing,
        'bins':full_rows,'accepted_domain':[q['bin'] for q in full_rows if q['accepted']],
        'combined_covariance':result['combined_covariance'] if result else [],
        'joint_covariance':result['joint_covariance'] if result else [],
        'covariance_domain_indices':indices.tolist(),
        'component_domains':[q['domain'] for q in measurements],
        'covariance_assumption':result['covariance_assumption'] if result else 'no common measured bin subspace',
        'science_acceptance':'ACCEPTED_DECLARED_BINS' if all(q['accepted'] for q in full_rows) else 'UNRESOLVED',
        'refinement_requests':refinements,'source_hashes':[q['source_hashes'] for q in measurements],
        'missing_policy':'unsupported scalar bounds are null; covariance covers only covariance_domain_indices'}
    report['supported_parameter_radius_epoch_domain'] = determine_supported_parameter_radius_epoch_domain(report)
    return report


# Paired measured-control ensembles for correlated observable error budgets.
def _retained_json(reference,base,memory_bytes):
    from core.workflow import path_at, verify_completed_receipt
    if not isinstance(reference,dict) or set(reference) != {'receipt','product'}:
        raise ValueError('Ensemble inputs need a sealed workflow receipt and declared product name')
    receipt_path = path_at(base,reference['receipt'])
    if receipt_path.stat().st_size*8 > memory_bytes:
        raise MemoryError('Ensemble workflow receipt exceeds input workspace')
    receipt = verify_completed_receipt(receipt_path)
    product = receipt['products'][reference['product']]
    if product['kind'] != 'file':
        raise ValueError('Ensemble source must be a retained measured JSON file')
    root = Path(receipt['generation']).resolve()
    path = (root/product['path']).resolve()
    if not path.is_relative_to(root) or path.stat().st_size*8 > memory_bytes:
        raise MemoryError('Ensemble measured product exceeds its bounded owned workspace')
    digest = file_sha256(path)
    if receipt['files_sha256'].get(path.relative_to(root).as_posix()) != digest:
        raise ValueError('Measured ensemble product is absent from the sealed workflow payload')
    record = json.loads(path.read_text())
    _verify_sources(record)
    bindings = {}
    groups = (record['source_hashes'], {str(path):digest,str(receipt_path):file_sha256(receipt_path)},
        receipt['source_hashes'], receipt['input_hashes'],
        {str(root/name):expected for name,expected in receipt['files_sha256'].items()})
    for group in groups:
        for source,expected in group.items():
            source = str(Path(source).resolve())
            if source in bindings and bindings[source] != expected:
                raise ValueError('Retained measured product and workflow sources conflict')
            bindings[source] = expected
    return record,bindings


def ensemble_input_bindings(realizations,systematic_controls,*,request_base,memory_bytes):
    """Pin measured sources and native receipt payloads for workflow resume."""
    from core.workflow import path_at, verify_completed_receipt
    if type(memory_bytes) is not int or memory_bytes <= 0:
        raise ValueError('Ensemble input workspace must be an explicit positive byte count')
    bindings = {}
    def merge(values):
        for path,expected in values.items():
            if path in bindings and bindings[path] != expected:
                raise ValueError('Ensemble input bindings conflict')
            bindings[path] = expected
    for reference in [*realizations,*systematic_controls]:
        record,sources = _retained_json(reference,request_base,memory_bytes)
        merge(sources)
        if record.get('schema') == 'native-control-residual-v1':
            receipt_path = path_at(request_base,record['native_realization_receipt'])
            if receipt_path.stat().st_size*8 > memory_bytes:
                raise MemoryError('Native realization receipt exceeds ensemble workspace')
            receipt = verify_completed_receipt(receipt_path)
            root = Path(receipt['generation']).resolve()
            merge({str(receipt_path):file_sha256(receipt_path),**receipt['source_hashes'],**receipt['input_hashes']})
            merge({str(root/name):expected for name,expected in receipt['files_sha256'].items()})
    return bindings


def measure_correlated_ensemble(request,realizations,systematic_controls,*,independence_key,
                                maximum_realizations,memory_bytes,request_base):
    """Build component-major covariance on bins observed in every paired sample.

    Residuals are signed variant-minus-reference differences in the frozen
    observable units. Sample covariance targets one native realization; it is
    never divided by the sample count. Independently measured systematic bias
    envelopes remain separate and are not replaced with an ensemble mean.
    """
    if request.get('schema') != 'observable-error-requirement-v1':
        raise ValueError('A frozen observable error requirement is required')
    if type(memory_bytes) is not int or memory_bytes <= 0 or type(maximum_realizations) is not int or maximum_realizations < 2:
        raise ValueError('Explicit positive ensemble workspace and realization bounds required')
    if not isinstance(realizations,list) or not 2 <= len(realizations) <= maximum_realizations:
        raise ValueError('At least two bounded independent native realizations required')
    if independence_key not in {'ic_phase_seed','sidm_operator_seed'}:
        raise ValueError('Explicit IC-phase or SIDM-operator independence scope required')
    requirement_hash = canonical_digest(request)
    n = len(request['bins'])
    if not n or not isinstance(systematic_controls,list) or not systematic_controls:
        raise ValueError('Nonempty physical bins and separately measured systematic controls required')
    controls = []
    sources = ensemble_input_bindings(realizations,systematic_controls,request_base=request_base,memory_bytes=memory_bytes)
    def retain(bindings):
        for path,digest in bindings.items():
            if path in sources and sources[path] != digest:
                raise ValueError('Ensemble sources conflict or changed during intake')
            sources[path] = digest
    for reference in systematic_controls:
        control,bindings = _retained_json(reference,request_base,memory_bytes)
        retain(bindings)
        if control.get('requirement_hash') != requirement_hash or control.get('observable') != request['observable'] or control.get('units') != request['units']:
            raise ValueError('Systematic control differs from the frozen physical requirement')
        controls.append(control)
    labels = [q['component'] for q in controls]
    if len(set(labels)) != len(labels) or not set(labels) <= ERROR_COMPONENTS:
        raise ValueError('Systematic controls must identify distinct recognized error components')
    k,m = len(labels),len(realizations)
    # Include dense published covariance, sample arrays, temporary reductions,
    # and conservative JSON/list expansion. Fail rather than reducing bins.
    estimated_bytes = 64*(k*n)**2+64*m*k*n
    if estimated_bytes > memory_bytes:
        raise MemoryError('Paired covariance and serialized products exceed ensemble workspace')
    samples = np.empty((m,k,n),dtype=float)
    deterministic_bounds = np.zeros((k,n),dtype=float)
    common = np.ones(n,dtype=bool)
    for control in controls:
        mask = _support_mask(control['supported'],(n,))
        bias = np.asarray(control['bias_bound'],dtype=float)
        if bias.shape != (n,) or not np.isfinite(bias[mask]).all() or np.any(bias[mask] < 0):
            raise ValueError('Supported systematic envelopes must be measured finite nonnegative values')
        if not isinstance(control.get('domain'),list) or any(request['bins'][i] not in control['domain'] for i in np.flatnonzero(mask)):
            raise ValueError('Systematic control support exceeds its measured domain')
        common &= mask
    identifiers,seeds,generations = set(),set(),set()
    family = None
    fixed_seed = None
    fixed_seed_initialized = False
    rows = []
    sample_definitions = None
    for index,reference in enumerate(realizations):
        record,bindings = _retained_json(reference,request_base,memory_bytes)
        retain(bindings)
        if record.get('schema') != 'native-control-residual-v1' or record.get('measurement_origin') != 'retained_native_control_products':
            raise ValueError('Ensemble requires actual retained native control residual products')
        if record.get('requirement_hash') != requirement_hash or record.get('components') != labels or record.get('bins') != request['bins'] or record.get('units') != request['units'] or record.get('observable') != request['observable']:
            raise ValueError('Paired residual components, bins or physical requirement differ')
        identity = record['realization_id']
        seed = record['independence'][independence_key]
        from core.workflow import path_at, verify_completed_receipt
        native_receipt_path = path_at(request_base,record['native_realization_receipt'])
        native_receipt = verify_completed_receipt(native_receipt_path)
        generation = str(Path(native_receipt['generation']).resolve())
        phase = record['independence'].get('ic_phase_seed')
        operator = record['independence'].get('sidm_operator_seed')
        if type(phase) is not int or phase < 0 or (operator is not None and (type(operator) is not int or not 0 <= operator < 2**64)):
            raise ValueError('Ensemble records must retain exact phase/operator seed declarations')
        unchanged = phase if independence_key == 'sidm_operator_seed' else operator
        if fixed_seed_initialized and unchanged != fixed_seed:
            raise ValueError('An ensemble cannot pool phase and operator seed changes under one independence scope')
        fixed_seed,fixed_seed_initialized = unchanged,True
        if Path(record['native_realization_generation']).resolve() != Path(generation):
            raise ValueError('Residual product changes its retained native realization generation')
        native_hash = file_sha256(native_receipt_path)
        if record['source_hashes'].get(str(native_receipt_path)) != native_hash:
            raise ValueError('Native realization receipt must be pinned in measured residual provenance')
        retain({str(native_receipt_path):native_hash})
        if not isinstance(identity,str) or not identity or type(seed) is not int or seed < 0 or not isinstance(generation,str) or not generation:
            raise ValueError('Each realization needs its exact native generation, identity and integer seed')
        if identity in identifiers or seed in seeds or generation in generations:
            raise ValueError('Repeated identities, seeds or native generations cannot inflate an ensemble')
        identifiers.add(identity);seeds.add(seed);generations.add(generation)
        model = record['paired_control_family_hash']
        if not isinstance(model,str) or len(model) != 64 or any(c not in '0123456789abcdef' for c in model):
            raise ValueError('Paired realizations require an exact frozen control-family hash')
        if family is not None and family != model:
            raise ValueError('Different model/control families cannot be pooled as stochastic realizations')
        family = model
        definitions = record.get('component_sample_definitions',
            {label:'signed_variant_minus_reference' for label in labels})
        expected = {label:('native_baseline_observable_centered_by_ensemble_covariance'
            if label == 'sidm_stochastic_realization' and record.get('residual_definition') == 'componentwise_native_samples'
            else 'signed_variant_minus_reference') for label in labels}
        if definitions != expected or record.get('residual_definition') not in {
                'signed_variant_minus_reference','componentwise_native_samples'}:
            raise ValueError('Every native sample component needs its exact common measurement definition')
        if record.get('residual_definition') == 'componentwise_native_samples' and (
                independence_key != 'sidm_operator_seed' or 'sidm_stochastic_realization' not in labels):
            raise ValueError('Direct SIDM stochastic samples require actual operator-seed baselines')
        if record.get('residual_definition') == 'componentwise_native_samples':
            target = 'native_baseline_observable' if len(labels) == 1 else 'baseline_plus_declared_signed_control_contrasts'
            if request.get('random_error_target') != target:
                raise ValueError('Stochastic sample covariance differs from the frozen physical claim target')
        if sample_definitions is not None and definitions != sample_definitions:
            raise ValueError('Native ensemble changes a component sample definition between realizations')
        sample_definitions = definitions
        values = np.asarray(record['residuals'],dtype=float)
        support = _support_mask(record['supported'],(k,n))
        if values.shape != (k,n) or not np.isfinite(values[support]).all():
            raise ValueError('Supported paired residuals must be finite component-by-bin measurements')
        samples[index] = values
        widths = np.asarray(record.get('deterministic_sample_bounds',np.zeros((k,n)).tolist()),float)
        if widths.shape != (k,n) or not np.isfinite(widths[support]).all() or np.any(widths[support]<0):
            raise ValueError('Native deterministic sample bounds differ from the actual supported ensemble bins')
        deterministic_bounds[support] = np.maximum(deterministic_bounds[support],widths[support])
        common &= support.all(axis=0)
        rows.append({'realization_id':identity,'seed':seed,'native_realization_generation':generation,
            'ic_phase_seed':phase,'sidm_operator_seed':operator})
    indices = np.flatnonzero(common)
    full_covariance = np.full((k*n,k*n),np.nan)
    mean = np.full((k,n),np.nan)
    combined_mean = np.full(n,np.nan)
    if len(indices):
        reduced = samples[:,:,indices]
        flattened = reduced.reshape(m,-1)
        covariance = np.atleast_2d(np.cov(flattened,rowvar=False,ddof=1))
        covariance = _psd(covariance,k*len(indices),request['covariance_tolerance'],'paired ensemble covariance')
        full_indices = np.concatenate([i*n+indices for i in range(k)])
        full_covariance[np.ix_(full_indices,full_indices)] = covariance
        mean[:,indices] = reduced.mean(axis=0)
        combined_mean[indices] = reduced.sum(axis=1).mean(axis=0)
    def nullable(values):
        return np.where(np.isfinite(values),values,None).tolist()
    measurements = []
    for index,control in enumerate(controls):
        block = full_covariance[index*n:(index+1)*n,index*n:(index+1)*n]
        covered = control.get('includes_deterministic_input_bounds',False)
        if type(covered) is not bool:
            raise ValueError('Systematic-envelope deterministic-width coverage must be an exact boolean')
        bias = np.asarray(control['bias_bound'],float)+(0 if covered else deterministic_bounds[index])
        measurements.append({**control,'supported':common.tolist(),
            'bias_bound':nullable(np.where(common,bias,np.nan)),
            'native_deterministic_sample_bound':nullable(np.where(common,deterministic_bounds[index],np.nan)),
            'deterministic_width_policy':'retained systematic envelope already includes widths' if covered else 'add maximum actual native sample width',
            'covariance':nullable(block),'source_hashes':dict(sources),
            'covariance_target':'single native realization baseline observable' if
                sample_definitions[control['component']] == 'native_baseline_observable_centered_by_ensemble_covariance'
                else 'single native realization residual',
            'ensemble_realization_count':m,'ensemble_independence_scope':independence_key})
        measurements[-1]['sample_definition'] = sample_definitions[control['component']]
        measurements[-1]['random_error_target'] = request.get('random_error_target','sum_of_declared_component_errors')
    for path,digest in sources.items():
        if file_sha256(path) != digest:
            raise ValueError('Ensemble input changed while covariance was measured')
    return {'schema':'correlated-error-ensemble-v1','request':request,'measurements':measurements,
        'joint_covariance':nullable(full_covariance),'covariance_domain_indices':indices.tolist(),
        'mean_residual_diagnostic':nullable(mean),'realizations':rows,'paired_control_family_hash':family,
        'component_sample_definitions':sample_definitions,
        'combined_estimator_mean_diagnostic':nullable(combined_mean),
        'combined_estimator_definition':request.get('random_error_target','sum_of_declared_component_errors'),
        'baseline_observable_mean_diagnostic':nullable(mean[labels.index('sidm_stochastic_realization')])
            if sample_definitions.get('sidm_stochastic_realization') == 'native_baseline_observable_centered_by_ensemble_covariance'
            else None,
        'independence_key':independence_key,'fixed_other_seed':fixed_seed,'source_hashes':sources,
        'random_error_target':request.get('random_error_target','sum_of_declared_component_errors'),
        'covariance_target':'single realization residual, not covariance of ensemble mean',
        'independence_status':'PRODUCER_DECLARED_DISTINCT_NATIVE_SEEDS',
        'science_acceptance':'NOT_ESTABLISHED'}


def publish_correlated_ensemble(destination,**parameters):
    destination = Path(destination).resolve()
    with native_run_lock(destination,operation='publish measured error ensemble'):
        if destination.exists():
            raise FileExistsError('Measured ensemble generations are immutable')
        result = measure_correlated_ensemble(**parameters)
        destination.mkdir(parents=True)
        atomic_json(destination/'ensemble.json',result)
        for key in ('request','measurements','joint_covariance'):
            product = destination/(key+'.json')
            atomic_json(product,result[key])
        budget_input = {key:result[key] for key in ('request','measurements','joint_covariance')}
        atomic_json(destination/'error-budget-input.json',budget_input)
        return {'schema':'correlated-error-ensemble-publication-v1','execution_status':'COMPLETE',
            'ensemble':str(destination/'ensemble.json'),'source_hashes':result['source_hashes'],
            'science_acceptance':'NOT_ESTABLISHED'}


def error_ensemble_main():
    import argparse
    parser = argparse.ArgumentParser(description='Measure covariance from retained paired native control ensembles')
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.request.stat().st_size > 16*1024**2:
        raise MemoryError('Ensemble request metadata exceeds the explicit 16 MiB intake bound')
    parameters = json.loads(args.request.read_text())
    parameters['request_base'] = args.request.resolve().parent
    print(json.dumps(publish_correlated_ensemble(args.output,**parameters),indent=2))


# Produce signed paired residuals from retained native convergence measurements.
COMPONENT_CONTROLS = {
    'integration':{'timestep','integration_accuracy'},'force_resolution':{'softening'},
    'gravity_force_accuracy':{'gravity_force_accuracy'},'gravity_tree_opening':{'gravity_tree_opening'},
    'mass_resolution':{'mass_resolution'},'output_cadence':{'output_cadence'},
    'start_epoch':{'start_epoch'},'finder':{'finder'},'subgrid_model':{'subgrid_model'},
    'observer_selection':{'observer_selection'},
    **{key:{key} for key in CONTROLS if key.startswith('sidm_')},
    'sidm_kernel':{'sidm_kernel','sidm_kernel_shape','sidm_kernel_origin_bound'},
}


def residual_input_bindings(parameters):
    """Bind every measured input and the actual baseline evolution generation."""
    from core.workflow import path_at, verify_completed_receipt
    base,memory = parameters['request_base'],parameters['memory_bytes']
    native_path = path_at(base,parameters['native_realization_receipt'])
    if native_path.stat().st_size*8 > memory:
        raise MemoryError('Native residual receipt exceeds declared intake workspace')
    native = verify_completed_receipt(native_path)
    sources = {str(native_path):file_sha256(native_path),**native['source_hashes'],**native['input_hashes']}
    sources.update({str(Path(native['generation'])/name):digest for name,digest in native['files_sha256'].items()})
    for pair in parameters['pairs']:
        for key in ('reference','variant'):
            _,bindings = _retained_json(pair[key],base,memory)
            for path,digest in bindings.items():
                if path in sources and sources[path] != digest:
                    raise ValueError('Native residual provenance changes between input bindings')
                sources[path] = digest
    if parameters.get('baseline_component') is not None:
        _,bindings = _retained_json(parameters['baseline_component']['reference'],base,memory)
        for path,digest in bindings.items():
            if path in sources and sources[path] != digest:
                raise ValueError('Native baseline sample changes residual provenance')
            sources[path] = digest
    return sources


def native_residual_runner(parameters,context):
    memory = parameters['memory_bytes']
    if type(memory) is not int or not 0 < memory <= context.resources['memory_gib']*1024**3:
        raise MemoryError('Residual workspace exceeds its explicit stage allocation')
    destination = Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):
        raise ValueError('Residual destination escapes its owned workflow generation')
    result = publish_native_control_residual(**parameters)
    atomic_json(context.generation/'residual-publication.json',result)
    return result


def produce_native_control_residual(request,pairs,*,realization_id,native_realization_receipt,
                                    independence_key,memory_bytes,maximum_components,request_base,
                                    baseline_component=None):
    from core.workflow import path_at, verify_completed_receipt
    if request.get('schema') != 'observable-error-requirement-v1':
        raise ValueError('Paired residuals require a frozen physical error requirement')
    if type(memory_bytes) is not int or memory_bytes <= 0 or type(maximum_components) is not int or maximum_components < 1:
        raise ValueError('Explicit positive residual workspace and component bounds required')
    component_count = len(pairs)+(baseline_component is not None) if isinstance(pairs,list) else 0
    if not isinstance(pairs,list) or not 1 <= component_count <= maximum_components:
        raise ValueError('A bounded nonempty measured component matrix is required')
    if not isinstance(realization_id,str) or not realization_id or independence_key not in {'ic_phase_seed','sidm_operator_seed'}:
        raise ValueError('Explicit realization identity and native independence scope required')
    n = len(request['bins'])
    if not n or component_count*n*128 > memory_bytes:
        raise MemoryError('Paired residual arrays and serialization exceed declared workspace')
    native_path = path_at(request_base,native_realization_receipt)
    if native_path.stat().st_size*8 > memory_bytes:
        raise MemoryError('Native realization receipt exceeds residual workspace')
    native = verify_completed_receipt(native_path)
    native_hash = file_sha256(native_path)
    sources = {str(native_path):native_hash,**native['source_hashes'],**native['input_hashes']}
    sources.update({str(Path(native['generation'])/name):digest for name,digest in native['files_sha256'].items()})
    labels,rows,masks,family,seeds = [],[],[],[],None
    deterministic_bounds = []
    reference_identity = None
    common_signature = None
    def retain(bindings):
        for path,digest in bindings.items():
            if path in sources and sources[path] != digest:
                raise ValueError('Paired residual input bindings conflict')
            sources[path] = digest
    for pair in pairs:
        component,control = pair['component'],pair['control']
        if component not in ERROR_COMPONENTS or control not in COMPONENT_CONTROLS.get(component,set()) or component in labels:
            raise ValueError('Each recognized error component must map to its actual native control dimension')
        reference,bindings = _retained_json(pair['reference'],request_base,memory_bytes)
        retain(bindings)
        variant,bindings = _retained_json(pair['variant'],request_base,memory_bytes)
        retain(bindings)
        if reference['run_id'] == variant['run_id'] or reference.get('changed_controls') != [] or variant.get('changed_controls') != [control]:
            raise ValueError('A residual requires distinct baseline and exactly one changed native control')
        # A numerical contrast must retain the SIDM operator seed as well as
        # the IC phases. The operator-seed control is the explicit exception.
        if (control != 'sidm_stochastic_realization'
                and reference.get('sidm_operator_seed') != variant.get('sidm_operator_seed')):
            raise ValueError('Numerical residual changes an unrelated SIDM operator realization')
        for key in ('common_experiment_signature','observable_definition_hash','units','epoch','bins','phase_seed','common_model_family_hash'):
            if reference.get(key) is None or reference[key] != variant.get(key):
                raise ValueError('Paired residual changes unrelated physical configuration: '+key)
        if reference['units'] != request['units'] or reference['bins'] != request['bins'] or reference.get('observable') != request['observable'] or variant.get('observable') != request['observable']:
            raise ValueError('Native paired measurements differ from the frozen observable or physical bins')
        if reference['source_hashes'].get(str(native_path)) != native_hash:
            raise ValueError('Paired baseline must bind the exact native realization receipt')
        current_identity = reference['run_id']
        if reference_identity is not None and reference_identity != current_identity:
            raise ValueError('All component contrasts must use the same paired native baseline')
        reference_identity = current_identity
        if common_signature is not None and common_signature != reference['common_experiment_signature']:
            raise ValueError('Component baselines do not share the recorded native experiment')
        common_signature = reference['common_experiment_signature']
        phase = reference['phase_seed']
        operator = reference.get('sidm_operator_seed')
        if type(phase) is not int or phase < 0 or (operator is not None and (type(operator) is not int or not 0 <= operator < 2**64)):
            raise ValueError('Native phase/operator seeds must retain their exact integer values')
        current_seeds = {'ic_phase_seed':phase,'sidm_operator_seed':operator}
        if current_seeds[independence_key] is None or (seeds is not None and seeds != current_seeds):
            raise ValueError('Paired component baselines change native seed settings or omit the selected scope')
        seeds = current_seeds
        model = reference['common_model_family_hash']
        if not isinstance(model,str) or len(model) != 64 or any(c not in '0123456789abcdef' for c in model):
            raise ValueError('Measured native controls require a frozen seed-independent model-family hash')
        if 'control_value' not in variant:
            raise ValueError('Variant measurement lacks its actual native control setting')
        family.append({'component':component,'control':control,'control_value':variant['control_value'],
            'model_family_hash':model,'observable_definition_hash':reference['observable_definition_hash']})
        left,right = np.asarray(reference['value'],float),np.asarray(variant['value'],float)
        if left.shape != (n,) or right.shape != (n,):
            raise ValueError('Paired measurements need one value per exact frozen physical bin')
        supported = _support_mask(reference['supported'],(n,)) & _support_mask(variant['supported'],(n,))
        if not np.isfinite(left[supported]).all() or not np.isfinite(right[supported]).all():
            raise ValueError('Supported native control measurements must be finite')
        difference = np.full(n,np.nan)
        with np.errstate(over='ignore',invalid='ignore'):
            difference[supported] = right[supported]-left[supported]
        if not np.isfinite(difference[supported]).all():
            raise ValueError('Native signed residual overflows its retained measurement representation')
        labels.append(component)
        rows.append(np.where(supported,difference,None).tolist())
        masks.append(supported.tolist())
        widths = np.zeros(n)
        for measured in (reference,variant):
            current = np.asarray(measured.get('deterministic_absolute_error_bound',[0.]*n),float)
            if current.shape != (n,) or not np.isfinite(current[supported]).all() or np.any(current[supported]<0):
                raise ValueError('Native paired deterministic error widths differ from supported physical bins')
            widths += current
        deterministic_bounds.append(np.where(supported,widths,None).tolist())
    definitions = {label:'signed_variant_minus_reference' for label in labels}
    if baseline_component is not None:
        if (not isinstance(baseline_component,dict) or set(baseline_component) != {'component','reference'}
                or baseline_component['component'] != 'sidm_stochastic_realization'
                or independence_key != 'sidm_operator_seed' or baseline_component['component'] in labels):
            raise ValueError('Direct SIDM variance requires a distinct baseline component and an operator-seed ensemble')
        target = 'baseline_plus_declared_signed_control_contrasts' if pairs else 'native_baseline_observable'
        if request.get('random_error_target') != target:
            raise ValueError('Direct native stochastic covariance requires its frozen observable target: '+target)
        baseline,bindings = _retained_json(baseline_component['reference'],request_base,memory_bytes)
        retain(bindings)
        if (baseline.get('schema') != 'native-convergence-measurement-v1' or baseline.get('changed_controls') != []
                or baseline.get('observable') != request['observable'] or baseline.get('units') != request['units']
                or baseline.get('bins') != request['bins']
                or baseline['source_hashes'].get(str(native_path)) != native_hash):
            raise ValueError('Stochastic samples must be actual matching native baseline observables')
        if reference_identity is not None and baseline['run_id'] != reference_identity:
            raise ValueError('Stochastic and numerical components must use the same native baseline')
        baseline_seeds = {'ic_phase_seed':baseline['phase_seed'],'sidm_operator_seed':baseline.get('sidm_operator_seed')}
        if (type(baseline_seeds['ic_phase_seed']) is not int or baseline_seeds['ic_phase_seed'] < 0
                or type(baseline_seeds['sidm_operator_seed']) is not int
                or not 0 <= baseline_seeds['sidm_operator_seed'] < 2**64
                or (seeds is not None and seeds != baseline_seeds)):
            raise ValueError('Direct stochastic sample lacks exact matching native phase/operator seeds')
        if common_signature is not None and baseline['common_experiment_signature'] != common_signature:
            raise ValueError('Direct baseline covariance changes the paired native experiment')
        seeds = baseline_seeds
        common_signature = baseline['common_experiment_signature']
        values = np.asarray(baseline['value'],float)
        mask = _support_mask(baseline['supported'],(n,))
        if values.shape != (n,) or not np.isfinite(values[mask]).all():
            raise ValueError('Supported native stochastic observable samples must be finite in exact physical bins')
        model = baseline.get('common_model_family_hash')
        definition = baseline.get('observable_definition_hash')
        if any(not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
                for value in (model,definition)):
            raise ValueError('Native stochastic samples require frozen model and observable definitions')
        component = baseline_component['component']
        family.append({'component':component,'control':'native_baseline_operator_ensemble',
            'model_family_hash':model,'observable_definition_hash':definition})
        labels.append(component)
        rows.append(np.where(mask,values,None).tolist())
        masks.append(mask.tolist())
        widths = np.asarray(baseline.get('deterministic_absolute_error_bound',[0.]*n),float)
        if widths.shape != (n,) or not np.isfinite(widths[mask]).all() or np.any(widths[mask]<0):
            raise ValueError('Native baseline deterministic uncertainty widths must remain supported physical bounds')
        deterministic_bounds.append(np.where(mask,widths,None).tolist())
        definitions[component] = 'native_baseline_observable_centered_by_ensemble_covariance'
    for path,digest in sources.items():
        if file_sha256(path) != digest:
            raise ValueError('Paired native source changed during residual production')
    return {'schema':'native-control-residual-v1','measurement_origin':'retained_native_control_products',
        'requirement_hash':canonical_digest(request),'observable':request['observable'],'units':request['units'],
        'components':labels,'bins':request['bins'],'realization_id':realization_id,'independence':seeds,
        'native_realization_generation':native['generation'],'native_realization_receipt':str(native_path),
        'paired_control_family_hash':canonical_digest({'requirement_hash':canonical_digest(request),'controls':family}),
        'paired_control_family':family,'common_experiment_signature':common_signature,
        'residual_definition':'componentwise_native_samples' if baseline_component is not None else 'signed_variant_minus_reference',
        'component_sample_definitions':definitions,'residuals':rows,'supported':masks,
        'deterministic_sample_bounds':deterministic_bounds,
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}


def publish_native_control_residual(destination,**parameters):
    destination = Path(destination).resolve()
    with native_run_lock(destination,operation='publish signed native control residuals'):
        if destination.exists():
            raise FileExistsError('Native residual product generations are immutable')
        result = produce_native_control_residual(**parameters)
        destination.mkdir(parents=True)
        atomic_json(destination/'native-control-residual.json',result)
        return {'schema':'native-control-residual-publication-v1','execution_status':'COMPLETE',
            'product':str(destination/'native-control-residual.json'),
            'source_hashes':result['source_hashes'],'science_acceptance':'NOT_ESTABLISHED'}


def error_residuals_main():
    import argparse
    parser = argparse.ArgumentParser(description='Produce paired residuals from retained native convergence measurements')
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.request.stat().st_size > 16*1024**2:
        raise MemoryError('Residual request metadata exceeds the 16 MiB intake bound')
    parameters = json.loads(args.request.read_text())
    parameters['request_base'] = args.request.resolve().parent
    print(json.dumps(publish_native_control_residual(args.output,**parameters),indent=2))


# Workflow refinement consumes newly measured errors; it never rescales old covariance.



def _substitute(value, replacements):
    if isinstance(value, dict):
        return {k: _substitute(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, replacements) for v in value]
    if isinstance(value, str):
        for key, replacement in replacements.items():
            value = value.replace('{'+key+'}', str(replacement))
    return value


def budget_refinement_requests(budget, routes, limits, *, parent_stage):
    """Deduplicate one physical control route across failed observable bins.

    The outer workflow enforces its own campaign-wide count/depth/resource
    limits as well. No automatic change to resolution, MPI ranks, epochs or
    model prescriptions occurs outside the recorded route specifications.
    """
    from core.workflow import safe_id
    safe_id(parent_stage)
    for key in ('maximum_children', 'maximum_controls_per_bin'):
        if type(limits[key]) is not int or limits[key] < 0:
            raise ValueError('Nonnegative explicit budget refinement limit required: '+key)
    if not set(routes) <= ERROR_COMPONENTS:
        raise ValueError('Refinement route names must identify actual error components')
    selected, unresolved = {}, []
    missing = budget['missing_components']
    for decision in budget['refinement_requests']:
        components = list(dict.fromkeys(missing+decision['components']))
        count = 0
        for component in components:
            if count >= limits['maximum_controls_per_bin']:
                unresolved.append({'bin': decision['bin'], 'component': component,
                    'reason': 'recorded per-bin control bound exhausted'})
                continue
            if component not in routes:
                unresolved.append({'bin': decision['bin'], 'component': component,
                    'reason': 'no recorded refinement producer for this component'})
                continue
            if component not in selected:
                if len(selected) >= limits['maximum_children']:
                    unresolved.append({'bin': decision['bin'], 'component': component,
                        'reason': 'recorded per-observable child count exhausted'})
                    continue
                selected[component] = []
            if decision['bin'] not in selected[component]:
                selected[component].append(decision['bin'])
            count += 1
    children = []
    budget_hash = canonical_digest(budget)
    for index, (component, domains) in enumerate(selected.items()):
        route = deepcopy(routes[component])
        if not route.get('kind') or not route.get('outputs') or not isinstance(route.get('parameters', {}), dict):
            raise ValueError('Refinement route must be a concrete registered product producer')
        sid = parent_stage+'_refine_'+str(index)+'_'+canonical_digest({'component': component, 'domains': domains})[:12]
        if len(sid) > 128:
            sid = 'refine_'+canonical_digest({'parent': parent_stage, 'component': component, 'domains': domains})[:40]
        route = _substitute(route, {'parent_stage': parent_stage, 'observable': budget['observable'],
            'component': component, 'budget_hash': budget_hash, 'refinement_id': sid})
        route['stage_id'] = sid
        route['refinement_evidence'] = {'parent_budget_hash': budget_hash, 'requirement_hash': budget['requirement_hash'],
            'observable': budget['observable'], 'component': component, 'requested_domains': domains,
            'reason': 'measured error budget requires additional native control'}
        if 'budget_input_parameter' in route:
            # Caller opts into a named parameter supported by its registered runner.
            key = route.pop('budget_input_parameter')
            if not isinstance(key, str) or not key or key in route['parameters']:
                raise ValueError('Refinement budget input parameter collides with explicit configuration')
            route['parameters'][key] = {'stage': parent_stage, 'artifact': 'error-budget.json'}
        children.append(route)
    return {'followups': children, 'unresolved_refinements': unresolved,
            'parent_budget_hash': budget_hash, 'limits': limits}


def execute_budget_stage(parameters, context):
    request = parameters['request']
    components = len(parameters['measurements'])
    bins = len(request['bins'])
    workspace = (components*bins)**2*8*4+bins*bins*8*4
    if workspace > context.resources['memory_gib']*1024**3:
        raise MemoryError('Dense correlated error-budget workspace exceeds the explicit stage memory allocation')
    budget = evaluate_masked_error_budget(request, parameters['measurements'], parameters.get('joint_covariance'))
    routes = parameters.get('refinement_routes', {})
    limits = parameters.get('refinement_limits', {'maximum_children': 0, 'maximum_controls_per_bin': 0})
    parent = context.stage_id
    if not parent:
        raise ValueError('Error-budget workflow stage requires an explicit stage identity')
    decisions = budget_refinement_requests(budget, routes, limits, parent_stage=parent)
    if parameters.get('reassessment'):
        decisions = append_budget_reassessment(decisions, parameters['reassessment'], parent_stage=parent)
    report = {**budget, 'refinement_dispatch': decisions,
              'random_error_target':request.get('random_error_target','sum_of_declared_component_errors'),
              'execution_status': 'COMPLETE', 'scientific_acceptance': budget['science_acceptance']}
    atomic_json(context.generation/'error-budget.json', report)
    atomic_json(context.generation/'frozen-requirement.json', request)
    atomic_json(context.generation/'supported-domain.json',budget['supported_parameter_radius_epoch_domain'])
    return {'followups': decisions['followups'], 'findings': decisions['unresolved_refinements'],
            'scientific_exclusions': decisions['unresolved_refinements'],
            'measurements': {'supported_bin_count': sum(q['supported'] for q in budget['bins']),
                             'accepted_bin_count': sum(q['accepted'] for q in budget['bins'])}}


def preflight_refinement_routes(parameters, context, registry):
    """Check every declared possible control producer before native evolution."""
    from core.workflow import positive_number, path_at, safe_id
    from core.foundation import file_sha256
    routes = parameters.get('refinement_routes', {})
    for component, row in routes.items():
        if component not in ERROR_COMPONENTS:
            raise ValueError('Unknown error-component refinement route')
        definition = registry.get(row['kind'])
        inputs = row.get('parameters', {})
        injected = row.get('budget_input_parameter')
        for key in definition.required_parameters:
            if key not in inputs and key != injected:
                raise ValueError('Refinement producer lacks required parameter: '+component+'/'+key)
        if not row.get('outputs'):
            raise ValueError('Refinement producer must declare actual products')
        for product, relative in row['outputs'].items():
            safe_id(product)
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts or relative in {'', '.'}:
                raise ValueError('Refinement product escapes generation')
        names = tuple(dict.fromkeys(definition.capabilities+tuple(row.get('capabilities', []))))
        context.capabilities.require(names, row.get('domain_parameters', {}), row.get('required_features', []))
        resources = row.get('resources', context.resources)
        for key in ('mpi_ranks', 'omp_threads', 'memory_gib', 'disk_gib'):
            value = positive_number(resources[key], key, key in {'mpi_ranks','omp_threads'})
            if value > context.resources[key]:
                raise ValueError('Refinement requires more resources than the declared campaign allocation')
        for asset in row.get('assets', [])+row.get('input_files', []):
            if file_sha256(path_at(context.base, asset['path'])) != asset['sha256']:
                raise ValueError('Refinement asset/input changed before native evolution')
    return {'possible_control_producers': sorted(routes), 'status': 'PREFLIGHTED',
            'scope': 'producer/capability/resource/input availability; measured need evaluated later'}


def append_budget_reassessment(dispatch, reassessment, *, parent_stage):
    """Connect explicit new measured inputs to a post-refinement budget producer.

    References of the form {refinement_component, artifact} resolve to the
    concrete child stage for that physical component. The reassessment must
    consume new measured covariance; this adapter never reuses old covariance
    for changed numerical/model controls by assumption.
    """
    from core.workflow import safe_id
    children = dispatch['followups']
    if not children:
        return dispatch
    by_component = {q['refinement_evidence']['component']:q['stage_id'] for q in children}
    def resolve(value):
        if isinstance(value, dict):
            if set(value) == {'refinement_component', 'artifact'}:
                component = value['refinement_component']
                if component not in by_component:
                    raise ValueError('Budget reassessment requires a component whose refinement was not dispatched')
                return {'stage': by_component[component], 'artifact': value['artifact'], 'load': 'json'}
            return {k:resolve(v) for k,v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value
    row = resolve(deepcopy(reassessment))
    if row.get('kind') != 'error_budget' or not row.get('outputs'):
        raise ValueError('Budget reassessment must use the measured error-budget producer and concrete outputs')
    params = row.get('parameters', {})
    if not all(key in params for key in ('request','measurements','joint_covariance')):
        raise ValueError('Budget reassessment requires its frozen requirement, new measurements and measured joint covariance')
    safe_id(row['stage_id'])
    row['depends_on'] = list(dict.fromkeys(row.get('depends_on', [])+[q['stage_id'] for q in children]+[parent_stage]))
    row['reassessment_evidence'] = {'parent_budget_hash':dispatch['parent_budget_hash'],
        'refined_components': sorted(by_component), 'parent_stage':parent_stage,
        'scope':'reassess actual new measured controls; no assumed variance scaling'}
    if len(children)+1 > dispatch['limits']['maximum_children']:
        return {**dispatch, 'unresolved_refinements':dispatch['unresolved_refinements']+[
            {'component':'budget_reassessment','reason':'recorded child count cannot accommodate reassessment producer'}]}
    return {**dispatch, 'followups':children+[row]}


def error_budget_main():
    import argparse
    p = argparse.ArgumentParser(description='Publish a frozen measured error budget with supported domains')
    p.add_argument('request', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    q = json.loads(a.request.read_text())
    print(json.dumps(publish_budget_and_domain(a.output, **q), indent=2))


# Systematic envelopes from retained native controls and reference certificates.
MODES = {'declared_native_control_contrast','native_baseline_error_against_certified_reference'}


def validate_envelope_parameters(parameters):
    request = parameters['request']
    component,control = parameters['component'],parameters['control']
    if request.get('schema') != 'observable-error-requirement-v1' or control not in COMPONENT_CONTROLS.get(component,set()):
        raise ValueError('Native envelope requires a frozen observable requirement and an actual component/control mapping')
    mode = parameters['bound_scope']
    if mode not in MODES or request.get('systematic_bound_targets',{}).get(component) != mode:
        raise ValueError('Native systematic-envelope interpretation must be frozen for this component')
    if mode == 'native_baseline_error_against_certified_reference':
        target = request.get('physical_target_definition_hash')
        if not isinstance(target,str) or len(target) != 64 or any(c not in '0123456789abcdef' for c in target):
            raise ValueError('Certified native error envelopes require an exact frozen physical target definition')
    for name in ('memory_bytes','maximum_pairs'):
        if type(parameters[name]) is not int or parameters[name] <= 0:
            raise ValueError('Explicit positive native envelope bound required: '+name)
    pairs = parameters['pairs']
    if not isinstance(pairs,list) or not 1 <= len(pairs) <= parameters['maximum_pairs']:
        raise ValueError('Native systematic envelopes require a bounded nonempty control sample')
    if parameters['memory_bytes'] < len(request['bins'])**2*len(pairs)*128:
        raise MemoryError('Native contrast covariance exceeds the explicit envelope workspace')
    for pair in pairs:
        expected = {'reference','variant'} | ({'certificate'} if mode != 'declared_native_control_contrast' else set())
        if not isinstance(pair,dict) or set(pair) != expected:
            raise ValueError('Each native pair must contain exactly its measured products and required reference certificate')


def envelope_input_bindings(parameters):
    validate_envelope_parameters(parameters)
    bindings = {}
    for pair in parameters['pairs']:
        for reference in pair.values():
            _,sources = _retained_json(reference,parameters['request_base'],parameters['memory_bytes'])
            for path,digest in sources.items():
                if path in bindings and bindings[path] != digest:
                    raise ValueError('Native systematic-envelope provenance conflicts')
                bindings[path] = digest
    return bindings


def measure_native_error_envelope(parameters):
    validate_envelope_parameters(parameters)
    request = parameters['request']
    component,control = parameters['component'],parameters['control']
    mode,memory = parameters['bound_scope'],parameters['memory_bytes']
    sources = envelope_input_bindings(parameters)
    n = len(request['bins'])
    if not n:
        raise ValueError('Empty native selections cannot supply a systematic-error domain')
    common = np.ones(n,dtype=bool)
    envelope = np.zeros(n,dtype=float)
    realizations,family,setting,definition = [],None,None,None
    identities = set()
    for pair in parameters['pairs']:
        baseline,_ = _retained_json(pair['reference'],parameters['request_base'],memory)
        variant,_ = _retained_json(pair['variant'],parameters['request_base'],memory)
        for record in (baseline,variant):
            if (record.get('schema') != 'native-convergence-measurement-v1'
                    or record.get('observable') != request['observable'] or record['units'] != request['units']
                    or record['bins'] != request['bins']):
                raise ValueError('Envelope inputs must be actual native measurements in the exact frozen physical bins')
        current_family = baseline['common_model_family_hash']
        current_setting = canonical_digest(variant.get('control_value'))
        current_definition = baseline['observable_definition_hash']
        if ((family is not None and current_family != family) or (setting is not None and current_setting != setting)
                or (definition is not None and definition != current_definition)
                or variant['common_model_family_hash'] != current_family):
            raise ValueError('Envelope cannot pool different physical models, observable definitions or control settings')
        family,setting,definition = current_family,current_setting,current_definition
        identity = (baseline['run_id'],variant['run_id'])
        if identity in identities:
            raise ValueError('Repeated native control pairs cannot expand an envelope sample')
        identities.add(identity)
        comparison = compare_campaign({**request,'control':control,'reference_run':baseline['run_id']},[baseline,variant])
        contrast = next(row for row in comparison['rows'] if row['run_id'] == variant['run_id'])
        mask = _support_mask(contrast['supported'],(n,))
        bound = np.asarray(contrast['error_bound'],float)
        for measured in (baseline,variant):
            deterministic = np.asarray(measured.get('deterministic_absolute_error_bound',[0.]*n),float)
            if deterministic.shape != (n,) or not np.isfinite(deterministic[mask]).all() or np.any(deterministic[mask]<0):
                raise ValueError('Native interval widths must remain finite nonnegative contrast bounds')
            bound = bound+deterministic
        if bound.shape != (n,) or not np.isfinite(bound[mask]).all() or np.any(bound[mask] < 0):
            raise ValueError('Supported native control contrasts require finite nonnegative measured envelopes')
        certificate_method = None
        if mode == 'native_baseline_error_against_certified_reference':
            certificate,_ = _retained_json(pair['certificate'],parameters['request_base'],memory)
            expected = {'schema':'native-reference-error-certificate-v1','component':component,
                'observable':request['observable'],'units':request['units'],'bins':request['bins'],
                'reference_run_id':variant['run_id'],'reference_value_hash':canonical_digest(variant['value']),
                'observable_definition_hash':definition,'common_model_family_hash':family,
                'target_definition_hash':request['physical_target_definition_hash']}
            if any(certificate.get(key) != value for key,value in expected.items()):
                raise ValueError('Reference certificate does not bound this actual native reference and physical target')
            if certificate.get('bound_kind') != 'absolute_error_upper_bound' or not isinstance(certificate.get('method'),str) or not certificate['method']:
                raise ValueError('A separately produced absolute reference bound and its physical method are required')
            certified = _support_mask(certificate['supported'],(n,))
            remainder = np.asarray(certificate['error_upper_bound'],float)
            if remainder.shape != (n,) or not np.isfinite(remainder[certified]).all() or np.any(remainder[certified] < 0):
                raise ValueError('Supported reference-error bounds must be finite and nonnegative')
            if any(request['bins'][i] not in certificate['domain'] for i in np.flatnonzero(certified)):
                raise ValueError('Reference-error certificate claims an unmeasured physical domain')
            # Pin the actual measured reference, rather than accepting a bound
            # for a different output that happens to reuse its run identity.
            if not all(certificate['source_hashes'].get(path) == digest for path,digest in variant['source_hashes'].items()):
                raise ValueError('Reference-error certificate does not bind the actual native reference provenance')
            mask &= certified
            bound = bound+remainder
            certificate_method = certificate['method']
        if not np.isfinite(bound[mask]).all():
            raise ValueError('Native physical envelope overflows its retained representation')
        common &= mask
        envelope[mask] = np.maximum(envelope[mask],bound[mask])
        realizations.append({'reference_run_id':baseline['run_id'],'variant_run_id':variant['run_id'],
            'phase_seed':baseline['phase_seed'],'operator_seed':baseline.get('sidm_operator_seed'),
            'control_value':variant['control_value'],'supported':mask.tolist(),
            'pair_bound':np.where(mask,bound,None).tolist(),'certificate_method':certificate_method})
    for path,digest in sources.items():
        if file_sha256(path) != digest:
            raise ValueError('Native envelope inputs changed before publication')
    return {'schema':'native-systematic-error-envelope-v1','component':component,
        'observable':request['observable'],'units':request['units'],'requirement_hash':canonical_digest(request),
        'supported':common.tolist(),'bias_bound':np.where(common,envelope,None).tolist(),
        'domain':[request['bins'][i] for i in np.flatnonzero(common)],
        'bound_scope':mode,'native_control':control,'common_model_family_hash':family,
        'includes_deterministic_input_bounds':True,
        'observable_definition_hash':definition,'retained_pairs':realizations,'source_hashes':sources,
        'sampling_uncertainty_policy':'frozen sigma multiplier on actual paired contrast covariance; no confidence level inferred',
        'parameter_scope':'declared measured control settings and retained native realizations only',
        'science_acceptance':'CONTROL_ONLY' if mode == 'declared_native_control_contrast' else 'REFERENCE_CERTIFICATE_DOMAIN_ONLY'}


def publish_native_error_envelope(destination,**parameters):
    destination = Path(destination).resolve()
    with native_run_lock(destination,operation='publish measured native systematic envelope'):
        if destination.exists():
            raise FileExistsError('Native systematic-envelope generations are immutable')
        result = measure_native_error_envelope(parameters)
        destination.mkdir(parents=True)
        atomic_json(destination/'systematic-envelope.json',result)
        return {'schema':'native-systematic-envelope-publication-v1','execution_status':'COMPLETE',
            'product':str(destination/'systematic-envelope.json'),'science_acceptance':result['science_acceptance']}


def native_error_envelope_runner(parameters,context):
    if parameters['memory_bytes'] > context.resources['memory_gib']*1024**3:
        raise MemoryError('Native error-envelope memory exceeds the allocated stage workspace')
    destination = Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):
        raise ValueError('Native envelope escapes its owned workflow generation')
    return publish_native_error_envelope(**parameters)


def native_error_envelopes_main():
    import argparse
    parser = argparse.ArgumentParser(description='Produce systematic envelopes from retained native controls')
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.request.stat().st_size*8 > 16*1024**2:
        raise MemoryError('Envelope request exceeds the bounded metadata intake')
    parameters = json.loads(args.request.read_text())
    parameters['request_base'] = args.request.resolve().parent
    print(json.dumps(publish_native_error_envelope(args.output,**parameters),indent=2))
