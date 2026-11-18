"""BAO catalogue measurements, template fits and published compressed likelihoods.

Positions use comoving Mpc without h in an explicitly authored fiducial geometry.
alpha_perp=(DM/rd)/(DM/rd)_fid and alpha_parallel=(DH/rd)/(DH/rd)_fid.
Uncalibrated ratios do not identify H0. Conditional fits retain their fixed
background/drag-scale assumptions; optional reconstruction/pair/sampler engines
load only when selected. Reference: DESI 2024 III, arXiv:2404.03000.
"""
import json
import re
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from .foundation import file_sha256, merge_source_bindings
from .stream_common import finite, positive, integer
from .observation_data import (array, covariance, exact_ids, read_observations,
    observed_catalogue, freeze_result, gaussian_observable, activate_pinned_backend, fingerprint, C_KMS)


def ingest_compressed_bao(request, destination):
    """Read the original ordered Gaussian BAO vector and full covariance.

    Each row retains its published redshift, dimensionless ratio and source
    index. Validate the covariance without rescaling, reordering, jitter or
    replacement. This intake does not fit H0 or generate posterior draws.
    """
    from .data import export_numeric_audit
    if request.get('schema') != 'compressed-bao-input-request-v1':
        raise ValueError('Versioned measured BAO input request required')
    limit = request['maximum_rows']
    if type(limit) is not int or not 1 <= limit <= 4096:
        raise ValueError('Bounded exact BAO row limit required')
    if request.get('source_origin') != 'published_observations':
        raise ValueError('Measured BAO intake requires explicit published source provenance')
    sources = {str(Path(__file__).resolve()): file_sha256(__file__)}
    paths = {}
    from . import source_path
    for name in ('observation_data', 'data', 'foundation'):
        path = source_path(name).resolve()
        sources[str(path)] = file_sha256(path)
    for name in ('mean', 'covariance', 'provenance'):
        reference = request[name]
        if set(reference) != {'path', 'sha256'}:
            raise ValueError('Exact measured input path/SHA256 required')
        path = Path(reference['path']).resolve()
        if path.stat().st_size > max(16 * 1024 ** 2, limit * limit * 32):
            raise MemoryError('Published BAO input exceeds bounded serialized intake')
        if file_sha256(path) != reference['sha256']:
            raise ValueError('Published BAO input changed: ' + name)
        paths[name] = path
        sources[str(path)] = reference['sha256']
    provenance = json.loads(paths['provenance'].read_text())
    if provenance.get('source_origin') != 'published_observations' or not provenance.get('commit'):
        raise ValueError('Pinned published BAO source provenance required')
    for name in ('mean', 'covariance'):
        entry = provenance['files'].get(paths[name].name, {})
        if entry.get('sha256') != sources[str(paths[name])]:
            raise ValueError('BAO source provenance does not bind the supplied file')
    rows = []
    for line in paths['mean'].read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        columns = line.split()
        if len(columns) != 3:
            raise ValueError('Original BAO source row requires redshift, value and quantity')
        z, value = (float(columns[0]), float(columns[1]))
        quantity = columns[2]
        if quantity not in {'DV_over_rs', 'DM_over_rs', 'DH_over_rs'} or not np.isfinite([z, value]).all() or z <= 0 or (value <= 0):
            raise ValueError('Unsupported or nonfinite measured BAO row')
        rows.append((z, value, quantity))
        if len(rows) > limit:
            raise MemoryError('Published BAO rows exceed selected intake bound')
    if not rows:
        raise ValueError('Published BAO input is empty')
    matrix = np.loadtxt(paths['covariance'], ndmin=2)
    validated = covariance(matrix, len(rows))
    if not np.array_equal(matrix, validated):
        raise ValueError('Measured covariance changed during validation')
    namespace = 'source-table-row:' + sources[str(paths['mean'])]
    arrays = {'source_row_index': np.arange(len(rows), dtype=np.uint64), 'redshift': np.array([r[0] for r in rows]), 'measured_ratio': np.array([r[1] for r in rows]), 'measured_covariance': matrix, 'input_available': np.ones(len(rows), dtype=bool)}
    specs = {name: {'units': 'identity' if name == 'source_row_index' else 'dimensionless', 'quantity': name, 'missing_policy': 'Original published rows retained; unavailable measurements are not invented'} for name in arrays}
    specs['source_row_index']['identity_namespace'] = namespace
    for path, digest in sources.items():
        if file_sha256(path) != digest:
            raise ValueError('Measured BAO input changed during conversion')
    report = {'schema': 'compressed-bao-observations-v1', 'source_origin': 'published_observations', 'role': 'estimator_observations', 'namespace': namespace, 'source_provenance': provenance, 'ordered_quantity_labels': [r[2] for r in rows], 'selection': 'All rows of the supplied published all-tracer likelihood, in original order', 'uncertainty': 'Original full measured Gaussian covariance; correlations preserved', 'calibration': 'Sound-horizon convention follows source labels; external calibration/model prediction required separately', 'science_acceptance': 'NOT_ACCEPTED', 'scope': 'Measured input ingestion only; no H0 fit or posterior generated'}
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Keep previous measured BAO input publication')
    return export_numeric_audit(destination, report, arrays, specs, algorithm=__file__, context={'epoch': {'observed_redshifts': 'per original source row'}}, source_hashes=sources)


def bao_prediction(history,redshift,quantities,sound_horizon_mpc):
    if isinstance(sound_horizon_mpc,bool) or not np.isfinite(sound_horizon_mpc) or sound_horizon_mpc<=0:
        raise ValueError('Positive explicit model drag sound horizon in Mpc required')
    z=np.asarray(redshift,float)
    if z.ndim!=1 or len(quantities)!=len(z) or np.any(~np.isfinite(z)) or np.any(z<=0):
        raise ValueError('Measured redshift/quantity row dimensions invalid')
    dm=history.transverse_distance(history.chi(z));dh=C_KMS/history.H(z)
    dv=np.cbrt(z*dh*dm**2)
    choices={'DM_over_rs':dm,'DH_over_rs':dh,'DV_over_rs':dv}
    if any(q not in choices for q in quantities):raise ValueError('Unknown source BAO quantity')
    return np.array([choices[q][i]/sound_horizon_mpc for i,q in enumerate(quantities)])


def parse_drag_sound_horizon(text):
    matches=re.findall(r'baryon drag stops at z\s*=\s*([0-9.eE+-]+)\s*\n\s*corresponding to conformal time\s*=\s*[0-9.eE+-]+\s*Mpc\s*\n\s*with comoving sound horizon rs\s*=\s*([0-9.eE+-]+)\s*Mpc',text)
    if len(matches)!=1:raise ValueError('One unambiguous CLASS baryon-drag sound horizon required')
    z,rs=map(float,matches[0])
    if not np.isfinite([z,rs]).all() or z<=0 or rs<=0:raise ValueError('Invalid CLASS drag-scale output')
    return z,rs


def conditional_bao_inputs(request):
    """Evaluate measured BAO against one frozen CLASS model, with no fit.

    DM, DH and DV are Mpc; dividing by the same model's drag horizon gives
    dimensionless predictions. Background, binary and drag invocation must
    agree with the selected linear bundle. Only output root/verbosity may
    differ in its drag request. The full original measured covariance enters
    the Gaussian likelihood; a CMB measurement likelihood is not supplied.
    """
    from .initial_conditions import validate_linear_bundle
    from .sidm_analysis import _load_product
    from .hubble_observers import ExpansionHistory
    from core.stream.inference import gaussian_loglikelihood
    if request.get('schema') != 'conditional-compressed-bao-request-v1':
        raise ValueError('Explicit conditional BAO model request required')
    memory = request['memory_gib']
    if isinstance(memory, bool) or not np.isfinite(memory) or memory <= 0:
        raise ValueError('Positive BAO workspace allocation required')
    sources = {str(Path(__file__).resolve()): file_sha256(__file__)}
    from . import source_path
    for name in ('hubble_observers', 'stream.inference', 'initial_conditions', 'cosmology', 'observation_data', 'data', 'foundation'):
        path = source_path(name).resolve()
        sources[str(path)] = file_sha256(path)

    def bound(reference):
        if set(reference) != {'path', 'sha256'}:
            raise ValueError('Exact conditional BAO source binding required')
        path = Path(reference['path']).resolve()
        if file_sha256(path) != reference['sha256']:
            raise ValueError('Conditional BAO source changed')
        merge_source_bindings(sources, {str(path): reference['sha256']})
        return path
    measured = bound(request['observations'])
    report, arrays, pins = _load_product(request['observations'], measured.parent, memory * 1024 ** 3 / 4)
    merge_source_bindings(sources, pins)
    if report.get('schema') != 'compressed-bao-observations-v1' or report.get('source_origin') != 'published_observations':
        raise ValueError('Pinned original published compressed BAO observations required')
    bundle = bound(request['linear_bundle'])
    validation = validate_linear_bundle(bundle, memory_gib=memory / 2)
    merge_source_bindings(sources, {r['path']: r['sha256'] for r in validation['sources'].values()})
    info = json.loads(bundle.read_text())
    background = (bundle.parent / info['source_files']['z0']['background']).resolve()
    if not background.is_relative_to(bundle.parent) or file_sha256(background) != info['checksums_sha256'][info['source_files']['z0']['background']]:
        raise ValueError('Original model background changed')
    sources[str(background)] = file_sha256(background)
    execution = bound(request['drag_execution'])
    log = bound(request['drag_log'])
    invocation = bound(request['drag_invocation'])
    receipt = json.loads(execution.read_text())
    native = json.loads(invocation.read_text())
    if receipt.get('status') != 'COMPLETE' or receipt.get('source_hashes', {}).get(str(bundle)) != file_sha256(bundle):
        raise ValueError('Drag calculation does not bind the selected frozen cosmology')
    if native.get('schema') != 'native-command-invocation-v1' or native.get('returncode') != 0 or native.get('status') != 'RETURNED':
        raise ValueError('Successful original CLASS drag invocation required')
    binary = Path(info['solver_binary']['path']).resolve()
    derived = Path(native['command'][1]).resolve()
    if native['command'] != [str(binary), str(derived)] or native['input_hashes'].get(str(binary)) != info['solver_binary']['sha256']:
        raise ValueError('Drag invocation changes the frozen CLASS binary')
    if not all((p.parent == execution.parent for p in [log, invocation, derived])):
        raise ValueError('Drag execution files belong to different runs')

    def settings(path):
        return {line.split('=', 1)[0].strip(): line.split('=', 1)[1].strip() for line in path.read_text().splitlines() if '=' in line and (not line.lstrip().startswith('#'))}
    original = (bundle.parent / info['source_files']['z0']['ini']).resolve()
    a, b = (settings(original), settings(derived))
    allowed = {'root', 'thermodynamics_verbose'}
    if any((a.get(k) != b.get(k) for k in set(a) | set(b) if k not in allowed)) or b.get('thermodynamics_verbose') != '1':
        raise ValueError('Drag calculation changes frozen physical or numerical settings')
    execution_sources = dict(receipt['source_hashes'])
    merge_source_bindings(execution_sources, native['input_hashes'])
    for p, digest in execution_sources.items():
        if file_sha256(p) != digest:
            raise ValueError('Original drag execution source changed')
        merge_source_bindings(sources, {p: digest})
    zdrag, rs = parse_drag_sound_horizon(log.read_text())
    parameters = info['class_parameters']['z0']
    omega_k = float(parameters.get('Omega_k', 0.0))
    history = ExpansionHistory.from_class(background, omega_k=omega_k)
    prediction = bao_prediction(history, arrays['redshift'], report['ordered_quantity_labels'], rs)
    contract = report['numeric_data']['array_contract']['datasets']
    for key in ['redshift', 'measured_ratio', 'measured_covariance']:
        if contract[key]['units'] != 'dimensionless':
            raise ValueError('Measured BAO unit contract differs')
    observed = finite(arrays['measured_ratio'], prediction.shape, 'measured BAO ratios')
    cov = covariance(arrays['measured_covariance'], len(observed), 'measured BAO covariance')
    available = np.asarray(arrays['input_available'])
    if available.dtype.kind != 'b' or available.shape != observed.shape or (not available.all()):
        raise ValueError('Conditional BAO likelihood requires every original measured row to be available')
    loglike, chi_square = gaussian_loglikelihood(observed, prediction, cov, return_chi_square=True)
    residual = observed - prediction
    values = {'redshift': arrays['redshift'], 'measured_ratio': observed, 'model_ratio': prediction, 'measured_covariance': cov, 'residual': residual, 'input_available': arrays['input_available']}
    for p, digest in sources.items():
        if file_sha256(p) != digest:
            raise ValueError('BAO model inputs changed during evaluation')
    result = {'schema': 'conditional-compressed-bao-likelihood-v1', 'role': 'conditional_model_likelihood', 'ordered_quantity_labels': report['ordered_quantity_labels'], 'model_drag_redshift': zdrag, 'model_sound_horizon_mpc': rs, 'drag_output_precision': 'CLASS summary prints six decimals; source bytes retained', 'log_likelihood': loglike, 'chi_square': chi_square, 'observed_rows': len(observed), 'calibration_scope': 'Drag scale predicted by the same frozen CLASS physical model; no CMB measurement or calibration likelihood included', 'uncertainty': 'Original full observed covariance; no free parameters fitted or degrees of freedom inferred', 'science_acceptance': 'NOT_ACCEPTED', 'posterior_generated': False, 'epoch': {'observed_redshifts': 'per original source row'}}
    return (result, values, sources)


def publish_conditional_bao(request,destination):
    from .data import export_numeric_audit
    result,values,sources=conditional_bao_inputs(request)
    if Path(destination).exists():raise FileExistsError('Keep previous conditional BAO evaluation')
    return export_numeric_audit(destination,result,values,
        {k:{'units':'dimensionless','quantity':k,'missing_policy':'All original published rows retained; no selection or renormalization'} for k in values},
        algorithm=__file__,context={'epoch':result['epoch']},source_hashes=sources)


def validate_bao_inference_settings(settings,memory_gib):
    from core.stream.inference import ParameterSpace
    memory_gib=positive(memory_gib,'BAO inference memory_gib')
    space=ParameterSpace(settings['priors'])
    if space.names!=('alpha_perp','alpha_parallel'):
        raise ValueError('Explicit ordered transverse/radial distance dilation priors required')
    for row in space.records:
        if row['units']!='dimensionless' or row['role']!='physical' or row['bounds'][0]<=0:
            raise ValueError('BAO distance dilations require positive dimensionless physical parameter bounds')
    sampler=settings['sampler']
    if sampler.get('sampler')!='emcee':raise ValueError('This bounded BAO profile requires explicit emcee settings')
    for name,minimum in [('walkers',4),('steps',1),('burn_in',0),('thin',1),('seed',0)]:
        if type(sampler.get(name)) is not int or sampler[name]<minimum:raise ValueError('Explicit bounded sampler integer required: '+name)
    if sampler['burn_in']>=sampler['steps']:raise ValueError('Sampler burn-in removes all steps')
    if sampler['seed']>=2**32:raise ValueError('emcee seed must fit its uint32 random-state domain')
    if sampler['walkers']*sampler['steps']*80*5>memory_gib*1024**3:raise MemoryError('BAO sampler retention exceeds allocation')
    return space


def dilate_bao_ratios(baseline,quantities,parameters):
    if set(parameters)!={'alpha_perp','alpha_parallel'}:raise ValueError('Exact two-parameter BAO dilation map required')
    transverse,radial=parameters['alpha_perp'],parameters['alpha_parallel']
    if not np.isfinite([transverse,radial]).all() or transverse<=0 or radial<=0:
        raise ValueError('Positive finite BAO distance dilations required')
    factors={'DM_over_rs':transverse,'DH_over_rs':radial,'DV_over_rs':np.cbrt(transverse**2*radial)}
    if any(q not in factors for q in quantities) or np.shape(baseline)!=(len(quantities),):raise ValueError('Original BAO quantity dimensions differ')
    return np.asarray(baseline)*np.array([factors[q] for q in quantities])


def prepare_bao_posterior(request, settings, *, verify_backend=True):
    """Build the existing posterior model; do not draw samples in preparation."""
    from core.stream.inference import PosteriorModel, pinned_sampler_backend
    result, values, sources = conditional_bao_inputs(request)
    space = validate_bao_inference_settings(settings, request['memory_gib'])
    if verify_backend:
        _, backend = pinned_sampler_backend('emcee')
        merge_source_bindings(sources, backend['source_hashes'])
    else:
        backend = None
    baseline = values['model_ratio']
    labels = result['ordered_quantity_labels']
    n = len(baseline)
    data = {'values': values['measured_ratio'][:, None], 'covariance': values['measured_covariance'], 'missing': np.zeros((n, 1), dtype=bool), 'source_origin': 'published_observations'}

    def forward(parameters):
        return {'values': dilate_bao_ratios(baseline, labels, parameters)[:, None]}

    def selection(parameters, prediction):
        return {'mode': 'unselected', 'normalization': 1.0, 'observed_detection_probability': np.ones(n), 'supported': True}
    model = PosteriorModel(space, forward, data, selection)
    center = space.transform(np.full(space.ndim, 0.5))
    loglike = model.log_likelihood(center)
    if not np.isfinite(loglike):
        raise ValueError('Prepared BAO likelihood is not finite at prior center')
    preparation = {'schema': 'prepared-bao-distance-dilation-posterior-v1', 'status': 'MODEL_PREPARED', 'parameter_priors': list(space.records), 'sampler': settings['sampler'], 'observed_rows': n, 'conditional_baseline': result, 'prior_center_log_likelihood': float(loglike), 'backend': backend, 'sampling_executed': False, 'science_acceptance': 'NOT_ACCEPTED', 'scope': 'Common dimensionless transverse/radial distance dilations relative to the frozen model across original redshifts; not H0 or a full cosmological model', 'selection': 'Published compressed likelihood treated as fixed selected data vector; no additional object-selection correction', 'remaining_validation': ['Sampler diagnostics and independent ensembles', 'Prior/support sensitivity', 'Posterior predictive checks', 'Calibration and coverage for this statistical model', 'Other inference and observational branches remain separate']}
    return (model, preparation, sources)


def publish_conditional_h0(bao_output, destination, *, settings=None):
    """Fit one late-time H0 scale to the original full-covariance BAO vector.

    With fixed E(z)=H(z)/H0 and fixed drag scale, DM, DH and DV scale as
    H0_reference/H0. This conditional fit keeps expansion shape, curvature
    shape and drag calibration fixed; it is not a full cosmology posterior.
    Settings expose the prior, MCMC budget, histogram grid and memory budget.
    """
    from core.foundation import atomic_json
    from core.workflow import verify_completed_receipt
    from core.hubble_observers import ExpansionHistory
    from core.stream.inference import ParameterSpace, PosteriorModel, fit_sampler
    from core.stream_common import json_safe
    required = {'memory_gib', 'h0_bounds_km_s_mpc', 'histogram_bins', 'sampler'}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError('Conditional H0 requires explicit memory, prior bounds, histogram and sampler settings')
    settings = dict(settings)
    sampler = dict(settings['sampler'])
    settings['sampler'] = sampler
    if set(sampler) != {'sampler', 'walkers', 'steps', 'burn_in', 'thin', 'seed', 'resume'}:
        raise ValueError('Explicit complete emcee settings required; no preset run is inserted')
    for name, minimum in [('walkers', 2), ('steps', 1), ('burn_in', 0), ('thin', 1), ('seed', 0)]:
        integer(sampler[name], name, minimum)
    if sampler['burn_in'] >= sampler['steps'] or sampler['seed'] >= 2 ** 32:
        raise ValueError('Sampler needs retained steps and a seed in the emcee uint32 domain')
    if sampler['sampler'] != 'emcee':
        raise ValueError('Conditional H0 currently requires the MCMC sampler')
    if sampler.get('resume') is not False:
        raise ValueError('Conditional H0 publishes a fresh analysis; sampler resume is not supported by this entry point')
    memory = positive(settings['memory_gib'], 'memory_gib')
    if sampler['walkers'] * sampler['steps'] * 80 * 5 > memory * 1024 ** 3:
        raise MemoryError('Conditional H0 retained chain exceeds the authored allocation')
    bounds = finite(settings['h0_bounds_km_s_mpc'], (2,), 'H0 bounds [km/s/Mpc]')
    if bounds[0] <= 0 or bounds[1] <= bounds[0]:
        raise ValueError('H0 prior bounds must be positive and ordered')
    bins = integer(settings['histogram_bins'], 'histogram_bins', 2)
    branch = Path(bao_output).resolve()
    state_path = branch / 'workflow.json'
    state = json.loads(state_path.read_text())
    owned = []
    for entry in state.get('stages', {}).values():
        generation = Path(entry['generation']).resolve()
        if not generation.is_relative_to(branch / 'generations'):
            raise ValueError('BAO stage escapes its original workflow')
        receipt_path = generation / 'stage-receipt.json'
        if file_sha256(receipt_path) != entry['receipt_sha256']:
            raise ValueError('BAO workflow receipt binding changed')
        receipt = verify_completed_receipt(receipt_path)
        if receipt['kind'] == 'flagship_runtime_bao':
            campaign_path = generation / 'campaign.json'
            if receipt['files_sha256'].get('campaign.json') != file_sha256(campaign_path):
                raise ValueError('Original BAO campaign is not owned by its completed producer')
            owned.append((campaign_path, receipt_path, receipt))
    if len(owned) != 1:
        raise ValueError('One original completed measured BAO preparation required')
    campaign_path, receipt_path, receipt = owned[0]
    campaign = json.loads(campaign_path.read_text())
    if campaign.get('schema') != 'cosmography-campaign-v1' or len(campaign['stages']) != 1 or campaign['stages'][0]['kind'] != 'bao_compressed':
        raise ValueError('Original published-observation BAO likelihood campaign required')
    request = {'schema': 'conditional-compressed-bao-request-v1', 'memory_gib': memory}
    for key in ('observations', 'linear_bundle', 'drag_execution', 'drag_log', 'drag_invocation'):
        path = Path(campaign['stages'][0]['inputs'][key]).resolve()
        request[key] = {'path': str(path), 'sha256': file_sha256(path)}
    baseline, values, pins = conditional_bao_inputs(request)
    if not np.asarray(values['input_available']).all():
        raise ValueError('Complete original published BAO vector required')
    bundle = Path(request['linear_bundle']['path'])
    info = json.loads(bundle.read_text())
    background = (bundle.parent / info['source_files']['z0']['background']).resolve()
    history = ExpansionHistory.from_class(background, omega_k=float(info['class_parameters']['z0'].get('Omega_k', 0.0)))
    h0_reference = float(np.asarray(history.H(np.array([0.0])))[0])
    if not np.isfinite(h0_reference) or h0_reference <= 0:
        raise ValueError('Positive reference H(0) in km/s/Mpc required')
    definition = {'model': 'H(z)=H0*E_reference(z); fixed reference spatial-curvature shape and r_drag', 'reference_H0_km_s_Mpc': h0_reference, 'fixed_sound_horizon_mpc': baseline['model_sound_horizon_mpc'], 'distance_scaling': 'DM, DH and DV scale by H0_reference/H0; r_drag held fixed', 'calibration': 'r_drag and E(z) are fixed CLASS model predictions, not an independently measured CMB calibration likelihood', 'not_marginalized': ['Expansion-shape parameters', 'Drag-scale uncertainty', 'Early-universe model uncertainty'], 'claim': 'Conditional H0 constraint from the original published full-covariance BAO data only; not a four-method H0 comparison'}
    space = ParameterSpace([{'name': 'H0', 'kind': 'uniform', 'bounds': bounds.tolist(), 'units': 'km/s/Mpc', 'role': 'physical'}])
    labels = baseline['ordered_quantity_labels']
    prediction = values['model_ratio']
    n = len(prediction)
    data = {'values': values['measured_ratio'][:, None], 'covariance': values['measured_covariance'], 'missing': np.zeros((n, 1), dtype=bool), 'source_origin': 'published_observations'}

    def forward(parameters):
        alpha = h0_reference / parameters['H0']
        return {'values': dilate_bao_ratios(prediction, labels, {'alpha_perp': alpha, 'alpha_parallel': alpha})[:, None]}

    def selection(parameters, predicted):
        return {'mode': 'unselected', 'normalization': 1.0, 'observed_detection_probability': np.ones(n), 'supported': True}
    model = PosteriorModel(space, forward, data, selection)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Preserve previous conditional H0 analysis')
    destination.mkdir(parents=True)
    merge_source_bindings(pins, {str(state_path): file_sha256(state_path), str(campaign_path): file_sha256(campaign_path), str(receipt_path): file_sha256(receipt_path), str(Path(__file__).resolve()): file_sha256(__file__)})
    report, draws = fit_sampler(model, sampler, destination / 'checkpoint', binding={'source_hashes': pins, 'resources': {'memory_gib': memory}, 'conditional_model': definition, 'settings': settings})
    samples = draws['posterior_samples'][:, 0]
    lo, median, hi = map(float, np.quantile(samples, [0.16, 0.5, 0.84]))
    density, edges = np.histogram(samples, bins=bins, density=True)
    grid = (edges[:-1] + edges[1:]) / 2
    arrays = {'h0_samples': samples, 'posterior_samples': draws['posterior_samples'], 'posterior_weight': draws['posterior_weight'], 'h0_grid': grid, 'posterior_density': density, 'measured_redshift': values['redshift'], 'measured_ratio': values['measured_ratio'], 'measured_covariance': values['measured_covariance'], 'model_ratio_at_median': dilate_bao_ratios(prediction, labels, {'alpha_perp': h0_reference / median, 'alpha_parallel': h0_reference / median})}
    units = {key: 'km/s/Mpc' if key in {'h0_samples', 'h0_grid', 'posterior_samples'} else '(km/s/Mpc)^-1' if key == 'posterior_density' else 'dimensionless' for key in arrays}
    from core.data import export_numeric_audit
    from core.research.display import export_scientific_suite
    from core import source_path
    for name in ('bao', 'stream.inference', 'hubble_observers', 'data', 'foundation'):
        path = source_path(name).resolve()
        pins[str(path)] = file_sha256(path)
    merge_source_bindings(pins, report['backend']['source_hashes'])
    if any((file_sha256(path) != digest for path, digest in pins.items())):
        raise ValueError('Conditional H0 inputs or sampler source changed')
    epoch = {'meaning': 'Original published observed redshift rows; no simulated H0 observations'}
    result = {'schema': 'conditional-bao-h0-posterior-v1', 'execution_status': 'COMPLETE', 'conditioning': definition, 'parameter_units': 'km/s/Mpc', 'median': median, 'equal_tail_68': [lo, hi], 'sampler': report, 'convergence': 'Original sampler diagnostics retained; independent ensembles and coverage are not established', 'science_acceptance': 'NOT_ACCEPTED', 'chain_written': True, 'epoch': epoch, 'settings': settings}
    result = json_safe(result)
    product = destination / 'h0-measurements.json'
    export_numeric_audit(product, result, arrays, {key: {'units': units[key], 'quantity': key} for key in arrays}, algorithm=__file__, context={'epoch': epoch}, source_hashes=pins)
    figure = {'figure_id': 'conditional-bao-h0', 'family': 'inference', 'variant': 'actual_posterior', 'title': 'Conditional H0 from measured BAO — fixed E(z) and drag scale', 'science_status': 'NOT_ACCEPTED', 'epoch': epoch, 'panels': [{'panel_id': 'h0', 'title': 'Fixed-model calibration; not a full cosmology posterior', 'quality_status': 'CONDITIONAL_MODEL', 'selection': definition['claim'], 'xlabel': 'H0 [km/s/Mpc]', 'ylabel': 'Posterior density [(km/s/Mpc)^-1]', 'x_units': units['h0_grid'], 'y_units': units['posterior_density'], 'series': [{'source': 'h0', 'kind': 'line', 'x': 'h0_grid', 'y': 'posterior_density', 'label': 'Actual MCMC draws; fixed calibration'}]}]}
    export_scientific_suite({'sources': {'h0': str(product)}, 'dpi': 180, 'maximum_panels_per_figure': 4, 'organized_outputs': True, 'figures': [figure]}, destination / 'figures', base=destination, memory_gib=memory)
    atomic_json(destination / 'summary.json', json_safe(result))
    return json_safe(result)


def build_selected_tracer_catalogue(parent, selection, *, seed):
    """Select tracers and apply the explicitly selected redshift-space geometry.

    For peculiar line-of-sight v in km/s, the comoving displacement is
    v/(a*H) in Mpc, with H in km/s/Mpc. Periodic wrapping uses the declared
    no-h comoving box. Simulation epoch H is omitted from estimator metadata.
    """
    ids = exact_ids(parent['object_id'])
    p = array(parent['position_mpc'], 'positions', 2)
    if p.shape != (len(ids), 3):
        raise ValueError('Tracer coordinates must be (N,3)')
    mask = np.ones(len(ids), dtype=bool)
    for field, (lo, hi) in selection.get('cuts', {}).items():
        values = finite(parent[field], (len(ids),), field)
        low, high = finite([lo, hi], (2,), field + ' cut bounds')
        if low > high:
            raise ValueError('Tracer cut bounds must be ordered')
        mask &= (values >= low) & (values <= high)
    rng = np.random.default_rng(integer(seed, 'tracer selection seed', 0))
    completeness = array(parent.get('completeness', np.ones(len(ids))), 'completeness', 1)
    if completeness.shape != ids.shape or np.any((completeness < 0) | (completeness > 1)):
        raise ValueError('Tracer completeness must be in [0,1]')
    mask &= rng.random(len(ids)) < completeness
    p = p[mask].copy()
    ident = ids[mask]
    mode = selection['rsd_geometry']
    if mode != 'none':
        v = finite(parent['peculiar_velocity_kms'], (len(ids), 3), 'tracer peculiar velocities')[mask]
        a = positive(selection['scale_factor'], 'RSD scale factor')
        h = positive(selection['epoch_H_kms_mpc'], 'RSD H [km/s/Mpc]')
        if mode == 'plane_parallel':
            los = finite(selection['los_vector'], (3,), 'RSD line of sight')
            los = los / positive(np.linalg.norm(los), 'RSD line-of-sight norm')
            p += (v @ los)[:, None] * los / (a * h)
        elif mode == 'radial':
            delta = p - finite(selection['observer_position_mpc'], (3,), 'observer position')
            distance = np.linalg.norm(delta, axis=1)
            if not np.isfinite(distance).all() or np.any(distance == 0):
                raise ValueError('Radial LOS undefined at observer or outside numeric range')
            los = delta / distance[:, None]
            p += np.sum(v * los, axis=1)[:, None] * los / (a * h)
        else:
            raise ValueError('Unknown RSD geometry')
    if 'box_mpc' in selection:
        box = finite(selection['box_mpc'], name='tracer periodic box')
        if box.shape not in ((), (3,)) or np.any(box <= 0):
            raise ValueError('Positive tracer periodic box required')
        p %= box
    p = finite(p, name='selected comoving tracer positions')
    if not len(ident):
        raise ValueError('Tracer selection is empty')
    recorded = {k: v for k, v in selection.items() if k != 'epoch_H_kms_mpc'}
    weights = finite(parent.get('weight', np.ones(len(ids))), (len(ids),), 'tracer weights')[mask]
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError('Selected tracer weights require positive total nonnegative support')
    columns = {'object_id': ident, 'position_mpc': p, 'weight': weights}
    return observed_catalogue(columns, units={'object_id': '1', 'position_mpc': 'Mpc', 'weight': '1'}, namespace=parent['namespace'], selection=dict(recorded, seed=seed, rsd_applied=mode != 'none', input_count=len(ids), selected_count=len(ident)))


def generate_survey_window_random_catalogue(window, *, count, seed):
    """Draw the authored survey window, not uniformly in radial distance.

    The radial CDF integrates n(r)*r^2 by trapezoids on the supplied grid and
    interpolates that recorded cumulative selection. A common density/length
    scale cancels from its normalization. Sky cells are weighted by solid
    angle times completeness, then sampled uniformly in RA and sin(dec).
    """
    if type(count) is not int or count < 2:
        raise ValueError('Random count must be an integer >=2')
    rng = np.random.default_rng(integer(seed, 'survey random seed', 0))
    if window['geometry'] == 'periodic_box':
        box = array(window['box_mpc'], 'box', 1, positive=True)
        if box.shape != (3,):
            raise ValueError('Periodic box needs three lengths')
        positions = rng.uniform(0, box, size=(count, 3))
    elif window['geometry'] == 'cutsky':
        from scipy.integrate import cumulative_trapezoid
        radius = array(window['radius_mpc'], 'radial selection', 1, positive=True)
        ndensity = array(window['number_density'], 'radial density', 1)
        if len(radius) < 2 or radius.shape != ndensity.shape or np.any(np.diff(radius) <= 0) or np.any(ndensity < 0):
            raise ValueError('Window radial sampling is invalid')
        if not np.any(ndensity > 0):
            raise ValueError('Window has zero radial selection')
        scaled_radius = radius / radius[-1]
        cdf = cumulative_trapezoid(ndensity / ndensity.max() * scaled_radius ** 2, x=scaled_radius, initial=0.0)
        if cdf[-1] <= 0:
            raise ValueError('Window has zero volume-weighted selection')
        cdf /= cdf[-1]
        cells = window['sky_cells']
        area = []
        for cell in cells:
            ra0, ra1, dec0, dec1 = finite(cell['bounds_deg'], (4,), 'sky cell bounds')
            comp = float(finite(cell['completeness'], (), 'sky completeness'))
            if not 0 <= ra0 < ra1 <= 360 or not -90 <= dec0 < dec1 <= 90 or (not 0 <= comp <= 1):
                raise ValueError('Sky mask cell bounds/completeness invalid')
            area.append(np.radians(ra1 - ra0) * (np.sin(np.radians(dec1)) - np.sin(np.radians(dec0))) * comp)
        for i, a in enumerate(cells):
            for b in cells[:i]:
                aa, bb = (a['bounds_deg'], b['bounds_deg'])
                if max(aa[0], bb[0]) < min(aa[1], bb[1]) and max(aa[2], bb[2]) < min(aa[3], bb[3]):
                    raise ValueError('Survey sky cells overlap')
        if sum(area) <= 0:
            raise ValueError('Sky selection is empty')
        choices = rng.choice(len(cells), size=count, p=np.asarray(area) / sum(area))
        positions = np.empty((count, 3))
        r = np.interp(rng.random(count), cdf, radius)
        for cell_index, cell in enumerate(cells):
            use = choices == cell_index
            n = int(use.sum())
            a, b, c, d = cell['bounds_deg']
            ra = np.radians(rng.uniform(a, b, n))
            mu = rng.uniform(np.sin(np.radians(c)), np.sin(np.radians(d)), n)
            positions[use] = r[use, None] * np.stack([np.sqrt(1 - mu * mu) * np.cos(ra), np.sqrt(1 - mu * mu) * np.sin(ra), mu], axis=-1)
        positions += finite(window['observer_position_mpc'], (3,), 'survey observer position')
    else:
        raise ValueError('Unsupported survey geometry')
    return observed_catalogue({'object_id': [f'random:{seed}:{i}' for i in range(count)], 'position_mpc': positions, 'weight': np.ones(count)}, units={'object_id': '1', 'position_mpc': 'Mpc', 'weight': '1'}, namespace=f'random:{seed}', selection={'window': window, 'seed': seed, 'count': count})


def _pair_catalogue(source):
    """Read aligned catalogue identities, no-h comoving Mpc and tracer weights."""
    product = read_observations(source)
    columns = product['columns']
    ids = exact_ids(columns['object_id'])
    positions = finite(columns['position_mpc'], (len(ids), 3), 'pair catalogue positions')
    weights = finite(columns['weight'], (len(ids),), 'pair catalogue weights')
    if product['units']['position_mpc'] != 'Mpc' or product['units']['weight'] != '1':
        raise ValueError('Pair catalogues require comoving Mpc without h and dimensionless weights')
    if not len(ids) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError('Pair catalogue requires positive total nonnegative tracer weight')
    return product, ids, positions, weights


def apply_recorded_bao_reconstruction(tracers, randoms, spec):
    """Run the selected pinned pyrecon algorithm with its authored geometry.

    RecSym shifts data and randoms by displacement+RSD; RecIso shifts randoms
    by displacement alone. The particle iterative algorithm requires the
    special 'data' lookup to evaluate shifts at reconstructed data positions.
    See the bundled pyrecon read_shifted_positions interfaces.
    """
    tracer, ids, positions, weights = _pair_catalogue(tracers)
    random, random_ids, random_positions, random_weights = _pair_catalogue(randoms)
    allowed = {'MultiGridReconstruction', 'IterativeFFTReconstruction',
               'IterativeFFTParticleReconstruction', 'PlaneParallelFFTReconstruction'}
    if spec['algorithm'] not in allowed or spec['convention'] not in {'RecSym', 'RecIso'}:
        raise ValueError('Declare a supported reconstruction algorithm and RecSym or RecIso')
    box = finite(spec['box_mpc'], name='reconstruction box')
    if box.shape not in ((), (3,)) or np.any(box <= 0):
        raise ValueError('Positive reconstruction box length or three lengths required')
    center = finite(spec['box_center_mpc'], (3,), 'reconstruction box center')
    growth = float(finite(spec['growth_rate'], (), 'reconstruction growth rate'))
    bias = float(finite(spec['bias'], (), 'tracer bias'))
    if bias == 0:
        raise ValueError('Reconstruction tracer bias cannot be zero')
    positive(spec['smoothing_mpc'], 'reconstruction smoothing Mpc')
    mesh = spec['nmesh']
    if type(mesh) is int:
        integer(mesh, 'reconstruction mesh')
    elif isinstance(mesh, (list, tuple)) and len(mesh) == 3:
        for size in mesh:
            integer(size, 'reconstruction mesh axis')
    else:
        raise ValueError('Reconstruction mesh requires one integer or three integer axes')
    backend, pin = activate_pinned_backend('pyrecon')
    reconstruction = getattr(backend, spec['algorithm'])(
        f=growth, bias=bias, los=spec['los'], nmesh=mesh, boxsize=box,
        boxcenter=center, dtype=spec.get('dtype', 'f8'))
    reconstruction.assign_data(positions, weights)
    reconstruction.assign_randoms(random_positions, random_weights)
    reconstruction.set_density_contrast(smoothing_radius=spec['smoothing_mpc'])
    reconstruction.run()
    data_input = 'data' if spec['algorithm'] == 'IterativeFFTParticleReconstruction' else positions
    shifted_data = finite(reconstruction.read_shifted_positions(data_input), positions.shape, 'reconstructed tracer positions')
    field = 'disp+rsd' if spec['convention'] == 'RecSym' else 'disp'
    shifted_randoms = finite(reconstruction.read_shifted_positions(random_positions, field=field), random_positions.shape, 'reconstructed random positions')
    return {
        'role': 'estimator_observations', 'schema': 'bao-reconstruction-v2',
        'data_positions_mpc': shifted_data.tolist(), 'shifted_randoms_mpc': shifted_randoms.tolist(),
        # Bind original geometry/weights/selection without duplicating their
        # large arrays in the reconstruction product. Pair counting receives
        # those original catalogues explicitly and verifies the bindings.
        'tracer_input_fingerprint': fingerprint(tracer),
        'random_input_fingerprint': fingerprint(random),
        'namespace': tracer['namespace'], 'object_ids': ids.tolist(),
        'random_namespace': random['namespace'], 'random_object_ids': random_ids.tolist(),
        'selection': tracer['selection'], 'reconstruction': spec, 'backend_revision': pin['commit'],
    }


def pycorr_measurement(tracers, randoms, spec, *, reconstructed=None):
    """Count xi(s,mu) using aligned data/random/shifted catalogues in Mpc.

    pycorr position_type='pos' accepts (N,3), as declared by the bundled counter.
    Shifted weights and exact identities must match the supplied randoms; a
    same-shaped reconstruction from another catalogue cannot be substituted.
    """
    tracer, ids, positions, weights = _pair_catalogue(tracers)
    random, random_ids, random_positions, random_weights = _pair_catalogue(randoms)
    shifted = None
    if reconstructed is not None:
        record = read_observations(reconstructed)
        if record.get('schema') != 'bao-reconstruction-v2' or record['namespace'] != tracer['namespace'] or record['random_namespace'] != random['namespace']:
            raise ValueError('Reconstruction must retain exact tracer and random namespaces')
        if not np.array_equal(exact_ids(record['object_ids']), ids) or not np.array_equal(exact_ids(record['random_object_ids']), random_ids):
            raise ValueError('Reconstruction particle identities/order differ from pair catalogues')
        if record['tracer_input_fingerprint'] != fingerprint(tracer) or record['random_input_fingerprint'] != fingerprint(random):
            raise ValueError('Reconstruction used different catalogue geometry, weights or selection')
        positions = finite(record['data_positions_mpc'], positions.shape, 'shifted data')
        shifted = finite(record['shifted_randoms_mpc'], random_positions.shape, 'shifted randoms')
    threads = integer(spec['threads'], 'pair counter threads')
    separation = array(spec['separation_edges_mpc'], 'separation edges', 1, positive=True)
    mu = array(spec['mu_edges'], 'mu edges', 1)
    if len(separation) < 2 or len(mu) < 2 or np.any(np.diff(separation) <= 0) or np.any(np.diff(mu) <= 0) or mu[0] < -1 or mu[-1] > 1:
        raise ValueError('Pair edges must increase; mu must lie in [-1,1]')
    kwargs = {
        'data_positions1': positions, 'data_weights1': weights,
        'randoms_positions1': random_positions, 'randoms_weights1': random_weights,
        'position_type': 'pos', 'los': spec['los'], 'engine': 'corrfunc', 'nthreads': threads,
    }
    if shifted is not None:
        kwargs.update(shifted_positions1=shifted, shifted_weights1=random_weights)
    if spec.get('box_mpc') is not None:
        box = finite(spec['box_mpc'], name='pair box')
        if box.shape not in ((), (3,)) or np.any(box <= 0):
            raise ValueError('Pair box must have positive scalar or three-axis extent')
        kwargs['boxsize'] = box
    _, counting_pin = activate_pinned_backend('Corrfunc')
    backend, pin = activate_pinned_backend('pycorr')
    result = backend.TwoPointCorrelationFunction('smu', (separation, mu), **kwargs)
    correlation = finite(result.corr, (len(separation)-1, len(mu)-1), 'measured pair correlation')
    return {
        'role': 'estimator_observations', 'schema': 'bao-correlation-v1',
        'xi_smu': correlation.tolist(),
        'separation_centers_mpc': (.5*separation[1:] + .5*separation[:-1]).tolist(),
        'mu_centers': (.5*mu[1:] + .5*mu[:-1]).tolist(),
        'separation_edges_mpc': separation.tolist(), 'mu_edges': mu.tolist(),
        'backend_revision': pin['commit'], 'counting_revision': counting_pin['commit'],
        'selection': tracer['selection'], 'namespace': tracer['namespace'],
        'object_ids': ids.tolist(), 'pair_settings': spec,
        'reconstruction': None if reconstructed is None else reconstructed['reconstruction'],
    }




def fit_anisotropic_bao_template(correlation, template, ensemble, spec, *, realization_id=None):
    """Fit transverse/radial BAO dilation, profiling seven linear nuisances.

    C is the supplied independent-mock covariance in separation-major, mu-minor
    order. Cholesky whitening and least squares profile amplitude/broadband
    terms without normal equations. The finite-mock likelihood is either the
    declared Student-t form or Hartlap-corrected Gaussian. Curvature errors are
    local approximations, not samples or demonstrated coverage.
    """
    observed = read_observations(correlation)
    volume = positive(spec['independent_volume_mpc3'], 'independent volume Mpc^3')
    minimum = positive(spec['minimum_independent_volume_mpc3'], 'minimum volume Mpc^3')
    replicated = spec.get('replicated_volume', False)
    if type(replicated) is not bool or replicated or volume < minimum:
        raise ValueError('BAO independent-volume requirement not met')
    separation = array(observed['separation_centers_mpc'], 'separation', 1, positive=True)
    mu = array(observed['mu_centers'], 'mu', 1)
    xi = finite(observed['xi_smu'], (len(separation), len(mu)), 'correlation')
    if len(separation) < 2 or len(mu) < 2 or np.any(np.diff(separation) <= 0) or np.any(np.diff(mu) <= 0) or mu[0] < -1 or mu[-1] > 1:
        raise ValueError('BAO correlation coordinates must increase with mu in [-1,1]')
    fit_range = finite(spec['fit_range_mpc'], (2,), 'BAO fit range')
    if fit_range[0] <= 0 or fit_range[1] <= fit_range[0]:
        raise ValueError('Positive ordered BAO fit range required')
    selected = (separation >= fit_range[0]) & (separation <= fit_range[1])
    ss, mm = np.meshgrid(separation[selected], mu, indexing='ij')
    data = xi[selected].ravel()
    bins = len(data)
    if bins < 9:
        raise ValueError('BAO fit needs at least nine bins for two dilations and seven nuisance coordinates')
    row_mask = np.repeat(selected, len(mu))
    # Unused bins may make the full sample covariance rank deficient. Only
    # the selected fit vector needs positive-definite covariance/precision.
    full_covariance = covariance(ensemble['covariance'], xi.size, semidefinite=True)
    measured_covariance = covariance(full_covariance[np.ix_(row_mask, row_mask)], bins, 'selected BAO covariance')
    cholesky = np.linalg.cholesky(measured_covariance)
    whitened_data = solve_triangular(cholesky, data, lower=True)
    mocks = integer(ensemble['n_realizations'], 'independent covariance realization count')
    realization_ids = exact_ids(ensemble['realization_ids'], 'independent covariance realization ID')
    if mocks <= bins + 2 or len(realization_ids) != mocks:
        raise ValueError('Insufficient distinct independent covariance realizations')
    template_s = array(template['separation_mpc'], 'template separation', 1, positive=True)
    template_mu = array(template['mu'], 'template mu', 1)
    template_xi = finite(template['xi_smu'], (len(template_s), len(template_mu)), 'template correlation')
    if len(template_s) < 2 or len(template_mu) < 2 or np.any(np.diff(template_s) <= 0) or np.any(np.diff(template_mu) <= 0):
        raise ValueError('BAO template coordinates must increase')
    interpolator = RegularGridInterpolator((template_s, template_mu), template_xi, bounds_error=True)
    bounds = finite(spec['alpha_bounds'], (2, 2), 'dilation bounds')
    initial = finite(spec['alpha_initial'], (2,), 'initial dilation')
    if np.any(bounds[:, 0] <= 0) or np.any(bounds[:, 1] <= bounds[:, 0]) or np.any(initial < bounds[:, 0]) or np.any(initial > bounds[:, 1]):
        raise ValueError('Positive ordered dilation bounds and an initial point within them required')
    likelihood = spec['covariance_likelihood']
    if likelihood not in {'student_t', 'hartlap_gaussian'}:
        raise ValueError('Declare student_t or hartlap_gaussian covariance likelihood')
    broadband = np.column_stack([item.ravel() for item in
        (np.ones_like(ss), 1/ss, 1/ss**2, mm**2, mm**2/ss, mm**2/ss**2)])

    def evaluate(alpha):
        transverse, radial = alpha
        transformed_s = ss * np.sqrt(transverse**2 * (1-mm**2) + radial**2 * mm**2)
        transformed_mu = radial * ss * mm / transformed_s
        if not np.isfinite(transformed_s).all() or not np.isfinite(transformed_mu).all():
            return None
        if np.any(transformed_s < template_s[0]) or np.any(transformed_s > template_s[-1]) or np.any(transformed_mu < template_mu[0]) or np.any(transformed_mu > template_mu[-1]):
            return None
        signal = interpolator(np.column_stack([transformed_s.ravel(), transformed_mu.ravel()]))
        design = np.column_stack([signal, broadband])
        whitened = solve_triangular(cholesky, design, lower=True)
        coefficients, _, rank, _ = np.linalg.lstsq(whitened, whitened_data, rcond=None)
        if rank != design.shape[1]:
            return None
        residual = whitened_data - whitened @ coefficients
        chi_square = float(residual @ residual)
        if not np.isfinite(chi_square):
            return None
        return chi_square, coefficients, design @ coefficients

    def cost(chi_square):
        # Parameter-independent normalization is omitted; this is not evidence.
        if likelihood == 'student_t':
            return mocks/2 * np.log1p(chi_square/(mocks-1))
        return .5 * (mocks-bins-2)/(mocks-1) * chi_square

    def objective(alpha):
        evaluated = evaluate(alpha)
        return 1e100 if evaluated is None else cost(evaluated[0])

    result = minimize(objective, initial, method='Nelder-Mead', bounds=bounds,
                      options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 10000})
    best = evaluate(result.x)
    if not result.success or result.fun > 1e90 or best is None:
        raise RuntimeError('Anisotropic BAO template fit failed or is outside template support')
    alpha = result.x
    step = 1e-4  # Dimensionless local-curvature stencil, not a run preset.
    if np.any(alpha-step < bounds[:, 0]) or np.any(alpha+step > bounds[:, 1]):
        raise ValueError('BAO optimum is too close to a dilation bound for a local Gaussian curvature error')

    def stencil(offset):
        evaluated = evaluate(alpha + offset)
        if evaluated is None:
            raise ValueError('BAO curvature stencil leaves identifiable template support')
        return cost(evaluated[0])

    e, f = np.array([step, 0.]), np.array([0., step])
    center = cost(best[0])
    hessian = np.empty((2, 2))
    hessian[0, 0] = (stencil(e)-2*center+stencil(-e))/step**2
    hessian[1, 1] = (stencil(f)-2*center+stencil(-f))/step**2
    hessian[0, 1] = hessian[1, 0] = (stencil(e+f)-stencil(e-f)-stencil(-e+f)+stencil(-e-f))/(4*step**2)
    curvature = np.linalg.cholesky(covariance(hessian, 2, 'BAO dilation curvature'))
    inverse_factor = solve_triangular(curvature, np.eye(2), lower=True)
    alpha_covariance = covariance(inverse_factor.T @ inverse_factor, 2, 'BAO dilation covariance')
    fiducial = finite([spec['fiducial_DM_over_rd'], spec['fiducial_DH_over_rd']], (2,), 'fiducial distance ratios')
    if np.any(fiducial <= 0):
        raise ValueError('Positive fiducial BAO distance ratios required')
    redshift = float(finite(spec['effective_redshift'], (), 'effective redshift'))
    if redshift < 0:
        raise ValueError('Effective BAO redshift must be nonnegative')
    ratios = fiducial * alpha
    ratio_covariance = fiducial[:, None] * alpha_covariance * fiducial[None, :]
    observables = {
        'DM_over_rd': gaussian_observable(ratios[0], ratio_covariance[0, 0], '1', redshift),
        'DH_over_rd': gaussian_observable(ratios[1], ratio_covariance[1, 1], '1', redshift),
    }
    calibration = spec.get('sound_horizon_calibration')
    if calibration is not None:
        rd = positive(calibration['rd_mpc'], 'calibrated drag horizon Mpc')
        sigma_rd = float(finite(calibration['sigma_mpc'], (), 'drag horizon uncertainty Mpc'))
        if sigma_rd < 0 or not calibration['source_hash']:
            raise ValueError('Sound-horizon calibration requires nonnegative uncertainty and a source binding')
        # First-order propagation assumes external r_drag is independent of the
        # fitted DH/r_drag ratio. This does not marginalize shared calibration.
        distance = ratios[1] * rd
        variance = ratio_covariance[1, 1]*rd**2 + ratios[1]**2*sigma_rd**2
        observables['H_z'] = gaussian_observable(C_KMS/distance, C_KMS**2*variance/distance**4, 'km/s/Mpc', redshift)
    fit = {
        'mean': alpha, 'covariance': alpha_covariance,
        'parameter_names': ['alpha_perp', 'alpha_parallel'], 'parameter_units': ['dimensionless', 'dimensionless'],
        'ratio_covariance': ratio_covariance, 'nuisance_coefficients': best[1], 'predicted_xi': best[2],
        'chi2': best[0], 'dof': bins-2-len(best[1]), 'negative_log_likelihood': float(result.fun),
        'likelihood_normalization': 'parameter-independent additive constant omitted; not model evidence',
        'n_independent_mocks': mocks, 'volume_mpc3': volume,
        'uncertainty': 'local curvature; independent coverage not executed',
        'calibration_propagation': 'first-order independent external r_drag' if calibration is not None else 'uncalibrated',
    }
    return freeze_result('bao', fit, observables=observables, selection=observed['selection'],
        object_ids=observed['object_ids'], namespace=observed['namespace'], model=spec,
        realization_id=realization_id, nuisance={'template_hash': fingerprint(template),
        'reconstruction': observed.get('reconstruction'), 'calibration': 'uncalibrated' if calibration is None else calibration})
