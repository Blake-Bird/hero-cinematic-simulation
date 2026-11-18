"""Recorded priors, selection-aware likelihoods and real posterior products.

Missing coordinates are marginalized through covariance submatrices. Shape
population inference divides out the original shape marginal prior and keeps
per-object selection explicit. A written chain is not scientific acceptance.
Optional sampler backends load only when a fit is requested.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import importlib
import importlib.metadata
import json
import sys
import numpy as np
from scipy.special import logsumexp, gammaln, betaln, expit
from scipy.stats import truncnorm
from scipy.integrate import quad
from core.stream_common import finite, positive, integer, seeded_rng, json_safe, digest
from core.foundation import atomic_json
from core.native import native_run_lock

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


PIN_FILE = _PROJECT_ROOT/'configs/inference.upstream.json'
TRUTH_KEYS = {'truth', 'true_parameters', 'injected_parameters', 'native_h0', 'generator_parameters', 'truth_labels'}


def estimator_only(value):
    if isinstance(value, dict):
        forbidden = TRUTH_KEYS.intersection(value)
        if forbidden:
            raise ValueError('Estimator input contains privileged truth fields: '+str(sorted(forbidden)))
        for v in value.values():
            estimator_only(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            estimator_only(v)


def pinned_sampler_backend(name):
    """Load actual pinned samplers from verified installed wheel inventories."""
    from core.native import verified_python_backend
    records = json.loads(PIN_FILE.read_text())['repositories']
    matches = [record for record in records if record['name']==name]
    if len(matches)!=1:
        raise ValueError('Unknown or repeated sampler backend: '+str(name))
    record = matches[0]
    root = _PROJECT_ROOT/record['directory']
    sources = {str(PIN_FILE):digest(PIN_FILE)}
    for relative,expected in record['source_sha256'].items():
        path = root/relative
        if digest(path)!=expected:
            raise ValueError('Pinned sampler source changed: '+relative)
        sources[str(path)] = expected
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise ImportError('Run bootstrap.py --install --profile inference before requesting '+name) from error
    binding = verified_python_backend(name,record,distribution=distribution)
    module = importlib.import_module(name)
    if str(Path(module.__file__).resolve()) not in binding['installed_assets']:
        raise ValueError('Loaded sampler is outside its verified distribution; start a clean process')
    sources.update(binding['source_hashes'])
    return module, {'name':name,'revision':record['revision'],'source_url':record['url'],
        'source_hashes':sources,'license_files':record['license_files'],
        'installed_origin':binding['origin'],
        'installed_asset_sha256':{Path(path).relative_to(Path(sys.prefix).resolve()).as_posix():checksum
            for path,checksum in binding['installed_assets'].items()}}



class ParameterSpace:
    """Named physical/nuisance priors with normalized densities and transforms.

    A log-uniform prior uses log(hi)-log(lo), avoiding overflow in hi/lo.
    Prior records keep units and roles attached to later posterior samples.
    """
    def __init__(self, records):
        self.records = tuple(copy.deepcopy(records))
        self.names = tuple(r['name'] for r in records)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('Distinct named parameters required')
        for row in self.records:
            if row['kind'] not in {'uniform', 'loguniform', 'truncated_normal'}:
                raise ValueError('Unsupported prior family')
            lo, hi = finite(row['bounds'], (2,))
            if lo >= hi or (row['kind'] == 'loguniform' and lo <= 0):
                raise ValueError('Invalid prior bounds')
            if row['kind'] == 'truncated_normal':
                positive(row['sigma'], 'prior sigma')
                finite(row['mean'], (), 'prior mean')
            if 'units' not in row or 'role' not in row:
                raise ValueError('Parameter units and physical/nuisance role required')

    @property
    def ndim(self):
        return len(self.records)

    def transform(self, cube):
        u = finite(cube, (self.ndim,))
        if np.any((u < 0)|(u > 1)):
            raise ValueError('Prior cube out of bounds')
        result = np.empty(self.ndim)
        for i, row in enumerate(self.records):
            lo, hi = row['bounds']
            kind = row['kind']
            if kind == 'uniform':
                result[i] = lo+(hi-lo)*u[i]
            elif kind == 'loguniform':
                result[i] = np.exp(np.log(lo)+(np.log(hi)-np.log(lo))*u[i])
            else:
                sigma = row['sigma']
                center = row['mean']
                result[i] = truncnorm.ppf(u[i], (lo-center)/sigma, (hi-center)/sigma, loc=center, scale=sigma)
        return result

    def log_prior(self, theta):
        theta = np.asarray(theta, float)
        if theta.shape != (self.ndim,) or not np.isfinite(theta).all():
            return -np.inf
        total = 0.
        for value, row in zip(theta, self.records):
            lo, hi = row['bounds']
            if not lo <= value <= hi:
                return -np.inf
            if row['kind'] == 'uniform':
                total -= np.log(hi-lo)
            elif row['kind'] == 'loguniform':
                total -= np.log(value)+np.log(np.log(hi)-np.log(lo))
            else:
                sigma = row['sigma']
                center = row['mean']
                total += truncnorm.logpdf(value, (lo-center)/sigma, (hi-center)/sigma, loc=center, scale=sigma)
        return float(total)

    def mapping(self, theta):
        return dict(zip(self.names, finite(theta, (self.ndim,))))


def gaussian_loglikelihood(observed, predicted, covariance, missing=None, *, return_chi_square=False):
    """Evaluate the Gaussian once; optionally retain its whitened chi-square.

    Default callers receive the log likelihood. Diagnostic consumers can reuse
    its quadratic form without factorizing the same covariance a second time.
    """
    y = np.asarray(observed, float).ravel()
    mu = np.asarray(predicted, float).ravel()
    if y.shape != mu.shape:
        raise ValueError('Prediction shape differs from data')
    mask = np.isfinite(y) if missing is None else ~np.asarray(missing, bool).ravel()
    if mask.shape != y.shape or np.any(mask & ~np.isfinite(y)):
        raise ValueError('Missingness mask inconsistent with observation')
    if not mask.any():
        return (0.,0.) if return_chi_square else 0.
    if not np.isfinite(mu[mask]).all():
        return (-np.inf,np.inf) if return_chi_square else -np.inf
    cov = finite(covariance, (len(y), len(y)))
    if not np.allclose(cov, cov.T):
        raise ValueError('Observation covariance must be symmetric')
    sub = cov[np.ix_(mask, mask)]
    root = np.linalg.cholesky(sub)
    residual = np.linalg.solve(root, y[mask]-mu[mask])
    chi_square=float(np.dot(residual,residual))
    loglike=float(-.5*chi_square-np.log(np.diag(root)).sum()-.5*mask.sum()*np.log(2*np.pi))
    return (loglike,chi_square) if return_chi_square else loglike


def selected_observation_loglikelihood(data, prediction, selection):
    y = np.asarray(data['values'], float)
    mu = finite(prediction['values'], y.shape)
    if y.ndim != 2:
        raise ValueError('Observations require objects by coordinates')
    n, d = y.shape
    missing = np.asarray(data['missing'], bool)
    if missing.shape != y.shape:
        raise ValueError('Missingness shape mismatch')
    covariance = finite(data['covariance'], (n*d, n*d))
    discrepancy = finite(prediction.get('model_error_covariance', np.zeros_like(covariance)), covariance.shape)
    if not np.allclose(discrepancy, discrepancy.T) or np.linalg.eigvalsh(discrepancy).min() < -1e-12:
        raise ValueError('Model error covariance must be positive semidefinite')
    periods = data.get('periodic_columns', {})
    # Wrap residuals in declared angular coordinates, then use the local
    # tangent Gaussian; broad angular likelihoods require wrapped families.
    adjusted = mu.copy()
    for key, period in periods.items():
        j = int(key)
        p = positive(period, 'coordinate period')
        residual = (y[:, j]-mu[:, j]+p/2)%p-p/2
        adjusted[:, j] = y[:, j]-residual
        adjusted[missing[:, j], j] = mu[missing[:, j], j]
    if 'membership_probability' in data:
        membership = finite(data['membership_probability'], (n,))
        if np.any((membership < 0)|(membership > 1)):
            raise ValueError('Invalid membership probabilities')
        total_cov = covariance+discrepancy
        block = np.zeros_like(total_cov)
        for i in range(n):
            block[i*d:(i+1)*d, i*d:(i+1)*d] = total_cov[i*d:(i+1)*d, i*d:(i+1)*d]
        if not np.allclose(total_cov, block):
            raise ValueError('Mixture likelihood requires independent object blocks; supply joint mixture model for correlations')
        background = finite(data['background_values'], y.shape)
        bg_cov = finite(data['background_covariance'], (n, d, d))
        result = 0.
        for i in range(n):
            signal = gaussian_loglikelihood(y[i], adjusted[i], block[i*d:(i+1)*d, i*d:(i+1)*d], missing[i])
            noise = gaussian_loglikelihood(y[i], background[i], bg_cov[i], missing[i])
            log_signal = np.log(membership[i]) if membership[i] else -np.inf
            log_noise = np.log1p(-membership[i]) if membership[i] < 1 else -np.inf
            result += np.logaddexp(log_signal+signal, log_noise+noise)
    else: result = gaussian_loglikelihood(y, adjusted, covariance+discrepancy, missing)
    detection = finite(selection['observed_detection_probability'], (n,))
    alpha = float(selection['normalization'])
    if not 0 < alpha <= 1 or np.any((detection <= 0)|(detection > 1)):
        return -np.inf
    mode = selection['mode']
    if mode == 'conditional_detected_sample':
        result += np.log(detection).sum()-n*np.log(alpha)
    elif mode == 'poisson_population':
        expected = positive(selection['intrinsic_expected_count'], 'intrinsic_expected_count')
        result += np.log(detection).sum()+n*np.log(expected)-expected*alpha-gammaln(n+1)
    elif mode != 'unselected':
        raise ValueError('Declare selection conditioning')
    missing_logp = finite(prediction.get('missingness_log_probability', np.zeros(n)), (n,))
    return float(result+missing_logp.sum())


def injection_selection_normalization(log_population_density, log_proposal_density, detection_probability, *, minimum_effective_samples):
    """Estimate alpha = mean[(population/proposal) * detection] over injections.

    Zero-support rows stay in the denominator. Importance ESS measures this
    estimate's support; it does not replace independent calibration.
    """
    minimum_effective_samples = positive(minimum_effective_samples, 'minimum_effective_samples')
    lp=np.asarray(log_population_density,float)
    if np.isnan(lp).any() or np.isposinf(lp).any():
        raise ValueError('Log population density may have finite values or physical -inf support')
    lq = finite(log_proposal_density, lp.shape)
    p = finite(detection_probability, lp.shape)
    if lp.ndim != 1 or np.any((p < 0)|(p > 1)):
        raise ValueError('Selection injections require one probability per independent proposal sample')
    if not np.any(p > 0):
        return {'normalization': 0., 'effective_samples': 0., 'supported': False}
    positive_rows = (p > 0)&np.isfinite(lp)
    if not positive_rows.any():
        return {'normalization':0.,'effective_samples':0.,'supported':False}
    logw = lp[positive_rows]-lq[positive_rows]+np.log(p[positive_rows])
    total = logsumexp(logw)
    alpha = np.exp(total-np.log(len(lp)))
    ess = float(np.exp(2*total-logsumexp(2*logw)))
    if alpha > 1+1e-8:
        raise ValueError('Injection selection estimate exceeds one; proposal/domain insufficient')
    return {'normalization': float(min(alpha, 1.)), 'effective_samples': ess,
        'supported': alpha > 0 and ess >= minimum_effective_samples, 'normalization_method': 'unbiased importance integral, not self-normalized prior weights'}


@dataclass
class PosteriorModel:
    space: ParameterSpace
    forward: object
    data: dict
    selection: object

    def __post_init__(self):
        estimator_only(self.data)

    def log_likelihood(self, theta):
        if not np.isfinite(self.space.log_prior(theta)):
            return -np.inf
        parameters = self.space.mapping(theta)
        prediction = self.forward(parameters)
        selected = self.selection(parameters, prediction)
        if not selected.get('supported', True):
            return -np.inf
        return selected_observation_loglikelihood(self.data, prediction, selected)

    def log_posterior(self, theta):
        prior = self.space.log_prior(theta)
        return prior+self.log_likelihood(theta) if np.isfinite(prior) else -np.inf


class DeterministicForwardMap:
    """Named parameter -> recorded physical config -> observed phase space.

    Only declared parameter bindings can change a model. Common tracer state
    and any stochastic release seed remain fixed across likelihood calls.
    Bindings are paths into a JSON model, not evaluated Python expressions.
    """
    def __init__(self, definition, parameters, evaluator):
        estimator_only(definition)
        self.definition = copy.deepcopy(definition)
        self.evaluator = evaluator
        self.bindings = parameters
        if len({r['name'] for r in parameters}) != len(parameters):
            raise ValueError('Duplicate forward parameter bindings')

    def __call__(self, theta):
        model = copy.deepcopy(self.definition)
        if set(theta) != {r['name'] for r in self.bindings}:
            raise ValueError('Parameter map does not match declared bindings')
        for row in self.bindings:
            paths = row.get('paths', [row.get('path')])
            if row.get('transform', 'identity') == 'exp':
                value = np.exp(theta[row['name']])
            elif row.get('transform', 'identity') == 'identity':
                value = theta[row['name']]
            else: raise ValueError('Unsupported parameter binding transform')
            for path in paths:
                if not path or any(str(q).startswith('__') for q in path):
                    raise ValueError('Explicit public config binding path required')
                target = model
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = float(value)
        return self.evaluator(model)


def split_rank_diagnostic(chains):
    """Split rank-normalized and folded R-hat for independent chains.

    Coupled emcee walkers are not independent chains and must not be passed
    here. Require separate ensembles to establish between-chain diagnostics.
    """
    from scipy.stats import rankdata, norm
    x = finite(chains)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[1] < 8:
        raise ValueError('At least two independent chains with eight draws required')
    half = x.shape[1]//2
    split = np.concatenate((x[:, :half], x[:, -half:]), axis=0)
    def statistic(values):
        mean = values.mean(axis=1)
        within = values.var(axis=1, ddof=1).mean(axis=0)
        between = half*mean.var(axis=0, ddof=1)
        variance = (half-1)/half*within+between/half
        return np.sqrt(np.divide(variance, within, out=np.full_like(within, np.inf), where=within > 0))
    values = split.reshape(-1, x.shape[-1])
    ranks = np.column_stack([rankdata(values[:, j]) for j in range(values.shape[1])])
    z = norm.ppf((ranks-.375)/(len(values)+.25)).reshape(split.shape)
    folded = np.abs(values-np.median(values, axis=0))
    ranks = np.column_stack([rankdata(folded[:, j]) for j in range(values.shape[1])])
    folded_z = norm.ppf((ranks-.375)/(len(values)+.25)).reshape(split.shape)
    return np.maximum(statistic(z), statistic(folded_z))


def fit_sampler(model, request, directory, *, binding):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sampler_name = request['sampler']
    module, backend = pinned_sampler_backend(sampler_name)
    if sampler_name == 'dynesty':
        import pickle
        from types import SimpleNamespace
        serializer, serializer_binding = pinned_sampler_backend('cloudpickle')
        # Upstream explicitly supports replacing its checkpoint pickler.
        # Physical forward adapters include closure-bound recorded inputs.
        module.utils.pickle_module = SimpleNamespace(dump=serializer.dump, load=pickle.load, PickleError=pickle.PickleError)
        backend['checkpoint_serializer'] = serializer_binding
        backend['source_hashes'].update(serializer_binding['source_hashes'])
    binding=copy.deepcopy(binding)
    if 'stage' in binding and 'sampler' in binding['stage'].get('parameters',{}):
        binding['stage']['parameters']['sampler'].pop('resume',None)
    sampler_signature={key:value for key,value in request.items() if key!='resume'}
    binding = json_safe({**binding, 'parameters': model.space.records, 'sampler': sampler_signature,
        'backend_revision': backend['revision'],
        'backend_installed_assets':backend['installed_asset_sha256'],
        'checkpoint_serializer_assets':backend.get('checkpoint_serializer',{}).get('installed_asset_sha256',{}),
        'sampler_implementation_sha256':digest(Path(__file__))})
    binding_path = directory/'binding.json'
    with native_run_lock(directory, operation='posterior sampler'):
        if binding_path.exists():
            if json.loads(binding_path.read_text()) != binding:
                raise ValueError('Sampler continuation code/data/prior binding changed')
            if not request.get('resume'):
                raise FileExistsError('Sampler directory already exists; choose explicit resume')
        else:
            if request.get('resume'):
                raise ValueError('Cannot resume without original binding')
            atomic_json(binding_path, binding)
        ndim = model.space.ndim
        seed = integer(request['seed'], 'seed', 0)
        resource=binding.get('resources',{})
        if resource:
            retained=(request.get('maximum_calls',0) if sampler_name=='dynesty' else request['walkers']*request['steps'])
            if retained*(ndim+8)*8*5>resource['memory_gib']*1024**3:
                raise MemoryError('Declared sampler retention exceeds campaign memory budget')
        if sampler_name == 'dynesty':
            checkpoint = directory/'dynesty.checkpoint'
            if request.get('resume'):
                # Trusted local pickle only: checkpoint is bound to this run.
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                sampler = module.DynamicNestedSampler.restore(str(checkpoint))
            else:
                sampler = module.DynamicNestedSampler(model.log_likelihood, model.space.transform, ndim,
                    nlive=integer(request['live_points'], 'live_points', 2*ndim+1),
                    bound=request['bound'], sample=request['sample'], rstate=np.random.default_rng(seed))
            sampler.run_nested(nlive_init=request['live_points'], dlogz_init=positive(request['dlogz'], 'dlogz'),
                maxcall=integer(request['maximum_calls'], 'maximum_calls'),
                maxiter=integer(request['maximum_iterations'], 'maximum_iterations'),
                n_effective=integer(request['effective_sample_target'], 'effective_sample_target'),
                checkpoint_file=str(checkpoint), checkpoint_every=request.get('checkpoint_seconds', 60),
                resume=bool(request.get('resume')), print_progress=False)
            result = sampler.results
            logweights = result.logwt-logsumexp(result.logwt)
            weights = np.exp(logweights)
            samples = result.samples
            arrays = {'posterior_samples': samples, 'posterior_weight': weights, 'log_likelihood': result.logl,
                'log_prior': np.array([model.space.log_prior(q) for q in samples]), 'log_evidence_history': result.logz,
                'log_evidence_error_history': result.logzerr}
            diagnostics = {'effective_samples': float(1/np.sum(weights**2)), 'log_evidence': float(result.logz[-1]),
                'log_evidence_error': float(result.logzerr[-1]), 'likelihood_calls': int(np.sum(result.ncall)),
                'sampling_status': 'RESOURCE_LIMIT_OR_STOPPING_RULE', 'rhat': 'not applicable to nested samples'}
        elif sampler_name == 'emcee':
            walkers = integer(request['walkers'], 'walkers', 2*ndim)
            backend_hdf = module.backends.HDFBackend(str(directory/'chain.h5'))
            rng = seeded_rng(seed, 'emcee-initial')
            initial = None
            if request.get('resume'):
                if not backend_hdf.initialized or backend_hdf.iteration == 0:
                    raise ValueError('No initialized chain to resume')
            else:
                backend_hdf.reset(walkers, ndim)
                initial = np.array([model.space.transform(rng.uniform(1e-12, 1-1e-12, ndim)) for _ in range(walkers)])
            sampler = module.EnsembleSampler(walkers, ndim, model.log_posterior, backend=backend_hdf)
            if not request.get('resume'):
                sampler.random_state = np.random.RandomState(seed).get_state()
            sampler.run_mcmc(initial, integer(request['steps'], 'steps'), progress=False)
            burn = integer(request['burn_in'], 'burn_in', 0)
            thin = integer(request['thin'], 'thin')
            chain = sampler.get_chain(discard=burn, thin=thin)
            if not len(chain):
                raise ValueError('Burn-in removes every posterior sample')
            samples = chain.reshape(-1, ndim)
            weights = np.full(len(samples), 1/len(samples))
            arrays = {'posterior_samples': samples, 'posterior_weight': weights, 'chain': chain,
                'log_posterior': sampler.get_log_prob(discard=burn, thin=thin).ravel(),
                'log_prior': np.array([model.space.log_prior(q) for q in samples])}
            arrays['log_likelihood']=arrays['log_posterior']-arrays['log_prior']
            try:
                tau = sampler.get_autocorr_time(discard=burn)
                ess = (backend_hdf.iteration-burn)*walkers/tau
                diagnostics = {'autocorrelation_steps': tau, 'effective_samples_by_parameter': ess,
                    'chain_length_over_tau': (backend_hdf.iteration-burn)/tau,
                    'sampling_status': 'AUTOCORRELATION_ESTIMATED'}
            except module.autocorr.AutocorrError as error:
                diagnostics = {'autocorrelation_steps': error.tau, 'sampling_status': 'CHAIN_TOO_SHORT', 'effective_samples': None}
            diagnostics.update({'acceptance_fraction': sampler.acceptance_fraction,
                'rhat': 'requires independent ensembles; coupled walkers are not independent chains'})
        else: raise ValueError('Supported pinned sampler names are dynesty and emcee')
        report = {'schema': 'stream-posterior-v1', 'sampler': sampler_name, 'parameter_names': model.space.names,
            'parameter_priors': model.space.records, 'backend': backend, 'diagnostics': diagnostics,
            'binding': binding, 'chain_written': True, 'science_acceptance': 'NOT_ACCEPTED',
            'calibration_status': 'NOT_EXECUTED', 'checkpoint_directory': str(directory.resolve())}
        # Individual posterior weights must remain attached to original priors
        # for later hierarchical importance reweighting.
        atomic_json(directory/'posterior.json', json_safe(report))
    return report, arrays


def observation_detection_probability(values, rule):
    """Recorded observable-space window/logistic selection; one per object."""
    x=finite(values)
    probability=np.full(len(x),float(rule.get('completeness',1.)))
    if np.any((probability<0)|(probability>1)):
        raise ValueError('Completeness must lie in [0,1]')
    for column, bounds in rule.get('windows',{}).items():
        j=int(column)
        lo,hi=finite(bounds,(2,))
        probability*=((x[:,j]>=lo)&(x[:,j]<=hi))
    if 'logistic' in rule:
        from scipy.special import expit
        record=rule['logistic']
        coefficient=finite(record['coefficients'],(x.shape[1],))
        probability*=expit(record['intercept']+x@coefficient)
    return probability


def posterior_predictive(model, posterior, *, draws, seed, maximum_selection_attempts=10000):
    samples = finite(posterior['posterior_samples'])
    weights = finite(posterior['posterior_weight'], (len(samples),))
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Valid posterior weights required')
    rng = seeded_rng(seed, 'posterior-predictive')
    index = rng.choice(len(samples), size=integer(draws, 'draws'), p=weights/weights.sum())
    predictions = []
    replicated = []
    discrepancy = []
    values = np.asarray(model.data['values'], float)
    missing = np.asarray(model.data['missing'], bool)
    mask = ~missing.ravel()
    observed = values.ravel()
    for i in index:
        prediction = model.forward(model.space.mapping(samples[i]))
        mu = finite(prediction['values'], values.shape)
        covariance = np.asarray(model.data['covariance'])+np.asarray(prediction.get('model_error_covariance', np.zeros_like(model.data['covariance'])))
        parameters=model.space.mapping(samples[i])
        selected=model.selection(parameters,prediction)
        if selected['mode']!='unselected' and 'detection_rule' not in selected:
            raise ValueError('Selected posterior predictive requires a declared observable-space detection rule')
        attempts=0
        while True:
            attempts+=1
            if attempts>maximum_selection_attempts:
                raise RuntimeError('Selected posterior predictive exhausted rejection budget')
            fake=rng.multivariate_normal(mu.ravel(),covariance).reshape(values.shape)
            if selected['mode']=='unselected':
                break
            probability=observation_detection_probability(fake,selected['detection_rule'])
            if np.all(rng.random(len(fake))<probability):
                break
        predictions.append(mu)
        replicated.append(fake)
        sub = covariance[np.ix_(mask, mask)]
        root = np.linalg.cholesky(sub)
        observed_residual = np.linalg.solve(root, observed[mask]-mu.ravel()[mask])
        fake_residual = np.linalg.solve(root, fake.ravel()[mask]-mu.ravel()[mask])
        discrepancy.append([np.dot(observed_residual, observed_residual), np.dot(fake_residual, fake_residual)])
    discrepancy = np.array(discrepancy)
    return {'posterior_predictive_tail_probability': float(np.mean(discrepancy[:, 1] >= discrepancy[:, 0])),
        'meaning': 'Gaussian replicated observations with joint observable selection rejection when requested; fixed observed count',
        'acceptance_status': 'NOT_ACCEPTED'}, {'predictive_mean': np.array(predictions), 'predictive_replication': np.array(replicated),
        'predictive_discrepancy': discrepancy, 'predictive_sample_row': index}


def evaluate_heldout_recovery(posterior_sets, frozen_truth, *, credible_mass):
    """Privileged evaluation only after fitting; does not feed truth to models."""
    if not 0 < credible_mass < 1:
        raise ValueError('Credible probability must lie within (0,1)')
    rows = []
    covered = []
    for identifier, posterior in posterior_sets.items():
        truth = finite(frozen_truth[identifier])
        x = finite(posterior['posterior_samples'])
        if x.shape[1:] != truth.shape:
            raise ValueError('Heldout truth/posterior shape mismatch')
        weight = finite(posterior['posterior_weight'], (len(x),)).copy()
        if np.any(weight < 0) or not np.isfinite(weight.sum()) or weight.sum() <= 0:
            raise ValueError('Heldout recovery requires finite nonnegative weights with positive total')
        weight /= weight.sum()
        intervals = []
        for j in range(x.shape[1]):
            order = np.argsort(x[:, j])
            cumulative = np.cumsum(weight[order])
            intervals.append(np.interp([(1-credible_mass)/2, (1+credible_mass)/2], cumulative, x[order, j]))
        intervals = np.array(intervals)
        mean = np.average(x, axis=0, weights=weight)
        coverage = (truth >= intervals[:, 0]) & (truth <= intervals[:, 1])
        covered.append(coverage)
        rows.append({'realization_id': identifier, 'bias': mean-truth, 'credible_interval': intervals, 'covered': coverage})
    return {'schema': 'stream-heldout-recovery-v1', 'credible_mass': credible_mass, 'cases': rows,
        'coverage': np.mean(covered, axis=0), 'evaluation_scope': 'independent realization truth supplied after freezing estimator',
        'science_acceptance': 'NOT_ACCEPTED'}


def shape_coordinates(b_over_a, c_over_a):
    b = finite(b_over_a)
    c = finite(c_over_a, b.shape)
    if np.any((b <= 0)|(b > 1)|(c <= 0)|(c > b)):
        raise ValueError('Require 0<c/a<=b/a<=1')
    denominator = 1-c*c
    t = np.divide(1-b*b, denominator, out=np.full_like(b, np.nan), where=denominator > 1e-12)
    return {'b_over_a': b, 'c_over_a': c, 'triaxiality': t, 'triaxiality_supported': denominator > 1e-12}


def covariate_design(record, normalization):
    estimator_only(record)
    names = ('log_mass_msun', 'log_radius_kpc', 'redshift', 'environment')
    values = finite([record[name] for name in names], (4,))
    center = finite(normalization['center'], (4,))
    scale = finite(normalization['scale'], (4,))
    if np.any(scale <= 0):
        raise ValueError('Positive frozen covariate scales required')
    return np.r_[1., (values-center)/scale]


class ShapePopulation:
    """Beta b/a and conditional Beta (c/a)/(b/a), with covariate means.

    This parameterization has the correct ordered-axis support and Jacobian.
    Orientation uses an axial Watson distribution relative to a recorded
    environmental axis, normalized over the full sphere. Full triaxial Euler
    orientation stays in individual nuisance posteriors unless modelled here.
    """
    def __init__(self, definition):
        self.definition = definition
        if definition['shape_definition'] not in {'density_shell', 'density_enclosed', 'potential_isosurface'}:
            raise ValueError('Population must declare density or potential shape estimator')

    def log_density(self, samples, columns, covariates, hyper):
        x = finite(samples)
        b = x[:, columns['b_over_a']]
        c = x[:, columns['c_over_a']]
        supported = (b > 0)&(b < 1)&(c > 0)&(c < b)
        result = np.full(len(x), -np.inf)
        design = covariate_design(covariates, self.definition['covariate_normalization'])
        beta_b = finite([hyper['b_mean_'+str(k)] for k in range(5)], (5,))
        beta_r = finite([hyper['ratio_mean_'+str(k)] for k in range(5)], (5,))
        mean_b = expit(np.dot(design, beta_b))
        mean_r = expit(np.dot(design, beta_r))
        # Saturation at exactly zero/one is outside a proper finite Beta law.
        # Reject this numerical hyperparameter point rather than clipping it.
        if not (0 < mean_b < 1 and 0 < mean_r < 1):
            return result
        concentration_b = positive(hyper['b_concentration'], 'b_concentration')
        concentration_r = positive(hyper['ratio_concentration'], 'ratio_concentration')
        a1, b1 = mean_b*concentration_b, (1-mean_b)*concentration_b
        a2, b2 = mean_r*concentration_r, (1-mean_r)*concentration_r
        q = b[supported]
        ratio = c[supported]/q
        density = (a1-1)*np.log(q)+(b1-1)*np.log1p(-q)-betaln(a1, b1)
        density += (a2-1)*np.log(ratio)+(b2-1)*np.log1p(-ratio)-betaln(a2, b2)-np.log(q)
        if 'axis_alignment_cosine' in columns:
            cosine = x[supported, columns['axis_alignment_cosine']]
            if np.any(abs(cosine) > 1):
                raise ValueError('Orientation cosines out of range')
            kappa = hyper['orientation_kappa']
            shift = max(0., kappa)
            norm = np.log(2*np.pi)+shift+np.log(quad(lambda u: np.exp(kappa*u*u-shift), -1., 1., epsabs=1e-10)[0])
            density += kappa*cosine*cosine-norm
        result[supported] = density
        return result


def hierarchical_reweighting(individual_posteriors, population, hyper, population_selection, *, minimum_effective_samples):
    minimum_effective_samples = positive(minimum_effective_samples, 'minimum_effective_samples')
    estimator_only(individual_posteriors)
    terms = []
    records = []
    halo_ids = set()
    for item in individual_posteriors:
        estimator_only(item.get('covariates', {}))
        halo = item['halo_id']
        if halo in halo_ids:
            raise ValueError('Multiple streams/shells in one halo require a joint posterior; avoid population double counting')
        halo_ids.add(halo)
        if item['shape_definition'] != population.definition['shape_definition']:
            raise ValueError('Density/potential or shell/enclosed shape definitions differ')
        samples = finite(item['samples'])
        logprior = finite(item['original_shape_log_prior'], (len(samples),))
        weights = finite(item['weights'], (len(samples),))
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('Posterior weights invalid')
        use = weights > 0
        logpopulation = population.log_density(samples, item['columns'], item['covariates'], hyper)
        logratio = np.log(weights[use]/weights.sum())+logpopulation[use]-logprior[use]
        integral = float(logsumexp(logratio))
        if not np.isfinite(integral):
            return -np.inf, {'status': 'NO_POPULATION_SUPPORT', 'halo_id': halo}
        normalized = logratio-integral
        ess = float(np.exp(-logsumexp(2*normalized)))
        if ess < minimum_effective_samples:
            return -np.inf, {'status': 'INDIVIDUAL_REWEIGHTING_UNRESOLVED', 'halo_id': halo, 'effective_samples': ess}
        selection = population_selection(hyper, item['covariates'])
        alpha = selection['normalization']
        if not selection.get('supported', False) or not 0 < alpha <= 1:
            return -np.inf, {'status': 'POPULATION_SELECTION_UNRESOLVED', 'halo_id': halo}
        term = integral-np.log(alpha)
        terms.append(term)
        records.append({'halo_id': halo, 'log_marginal_ratio': integral,
            'selection_normalization': alpha, 'importance_effective_samples': ess})
    if not terms:
        raise ValueError('No individual posteriors supplied')
    return float(np.sum(terms)), {'status': 'COMPUTED', 'individuals': records,
        'method': 'individual original-shape-prior importance ratios and per-object population selection normalization',
        'science_acceptance': 'NOT_ACCEPTED'}


@dataclass
class HierarchicalPosterior:
    space: ParameterSpace
    individual: list
    population: ShapePopulation
    selection: object
    minimum_effective_samples: int

    def log_likelihood(self, theta):
        result, _ = hierarchical_reweighting(self.individual, self.population, self.space.mapping(theta),
            self.selection, minimum_effective_samples=self.minimum_effective_samples)
        return result

    def log_posterior(self, theta):
        prior = self.space.log_prior(theta)
        return prior+self.log_likelihood(theta) if np.isfinite(prior) else -np.inf


def injection_population_selection(population, injections, *, minimum_effective_samples):
    """Condition on each recorded host's covariates, using independent injections."""
    minimum_effective_samples = positive(minimum_effective_samples, 'minimum_effective_samples')
    estimator_only(injections)
    by_id = {row['covariate_id']: row for row in injections['groups']}
    if len(by_id) != len(injections['groups']):
        raise ValueError('Distinct injection covariate groups required')
    def selection(hyper, covariates):
        row = by_id[covariates['covariate_id']]
        logp = population.log_density(row['samples'], row['columns'], covariates, hyper)
        # -inf means physical zero support. Keep these zero-weight injection
        # rows in the denominator rather than biasing the integral upward.
        return injection_selection_normalization(logp, row['log_proposal_density'],
            row['detection_probability'], minimum_effective_samples=minimum_effective_samples)
    return selection
