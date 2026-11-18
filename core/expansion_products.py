"""Post-fit expansion recovery, explicit domain decisions and original figures.

H0 and H(z) retain separate observable identities and km/s/Mpc units. Repeated
observers within one native realization form one statistical group. Recovery
truth is accessed only after estimator products are frozen. Execution success
and a declared domain's scientific acceptance remain separate.
"""
from collections import defaultdict
from pathlib import Path
import csv
import html
import io
import json
import math
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
from .foundation import atomic_json, canonical_digest, file_sha256
from .foundation import merge_source_bindings as _merge
from .observation_data import read_frozen, array, covariance, fingerprint


def load_frozen_estimator_products(methods):
    results=[]
    for method,products in methods.items():
        for path in products:
            product=read_frozen(path)
            if product['method']!=method:raise ValueError('Method and frozen result differ')
            results.append(product)
    seen=set()
    for product in results:
        identity=(product['method'],product['realization_id'],product['frozen_hash'])
        if identity in seen:raise ValueError('Repeated frozen result in comparison')
        seen.add(identity)
    return results




def _truth_at(truth, observable_name, measurement):
    """Read positive post-fit truth in km/s/Mpc, within its saved redshift grid."""
    if observable_name == 'H0':
        return float(array(truth['H0_kms_mpc'], 'evaluation H0', 0, positive=True))
    if observable_name == 'H_z' or observable_name.startswith('H_z:'):
        redshift = measurement['redshift']
        if redshift is None:
            raise ValueError('H(z) observable needs an actual redshift')
        redshift = float(array(redshift, 'evaluation redshift', 0))
        grid = array(truth['redshift'], 'evaluation redshifts', 1)
        expansion = array(truth['H_kms_mpc'], 'evaluation H(z)', 1, positive=True)
        if len(grid) < 2 or expansion.shape != grid.shape or np.any(np.diff(grid) <= 0):
            raise ValueError('Evaluation H(z) needs aligned increasing redshifts')
        if not grid[0] <= redshift <= grid[-1]:
            raise ValueError('H(z) comparison outside truth coverage')
        return float(PchipInterpolator(grid, expansion, extrapolate=False)(redshift))
    return None


def _wilson(successes,n,confidence=.68):
    if n<1:return None
    z=norm.ppf((1+confidence)/2);p=successes/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half),float(center+half)]


def measure_bias_precision_coverage_and_redshift_dependence(results, truth):
    """Measure residuals in km/s/Mpc and coverage over native realizations.

    Residual = estimate - post-fit truth; pull = residual/quoted sigma.
    Repeated observers are averaged within realization before computing mean
    bias, its sample standard error, or interval coverage. Wilson intervals
    apply only when every realization contributes one Bernoulli coverage draw.
    """
    measurements = []
    groups = defaultdict(list)
    for product in results:
        read_frozen(product)
        for key, measurement in product['observables'].items():
            target = _truth_at(truth, key, measurement)
            if target is None:
                continue
            value = float(measurement['value'])
            sigma = float(measurement['sigma'])
            lower, upper = measurement['interval_68']
            if not np.isfinite([value, sigma, lower, upper]).all() or sigma <= 0 or lower > upper:
                raise ValueError('Frozen observable needs finite value, positive sigma and ordered interval')
            row = {
                'method': product['method'],
                'observable': key,
                'redshift': measurement.get('redshift'),
                'value': value,
                'sigma': sigma,
                'target': target,
                'residual': value - target,
                'fractional_bias': (value - target) / target,
                'pull': (value - target) / sigma,
                'covered_68': bool(lower <= target <= upper),
                'realization_id': product['realization_id'],
                'frozen_hash': product['frozen_hash'],
                'environment': product['environment'],
                'model_hash': fingerprint(product['model']),
            }
            measurements.append(row)
            groups[row['method'], key, row['redshift'], row['model_hash'], fingerprint(row['environment'])].append(row)
    summaries = []
    for (method, key, z, model, environment), rows in groups.items():
        grouped = defaultdict(list)
        for row in rows:
            if row['realization_id'] is None:
                continue
            grouped[row['realization_id']].append(row)
        realization_bias = np.array([np.mean([r['residual'] for r in values]) for values in grouped.values()])
        count = len(grouped)
        coverage_values = np.array([np.mean([r['covered_68'] for r in values]) for values in grouped.values()])
        summary = {
            'method': method,
            'observable': key,
            'redshift': z,
            'model_hash': model,
            'environment_hash': environment,
            'n_products': len(rows),
            'n_independent_realizations': count,
            'mean_bias': float(realization_bias.mean()) if count else None,
            'mean_bias_definition': 'equal weight per independent native realization; average repeated observers within realization',
            'mean_quoted_sigma': float(np.mean([r['sigma'] for r in rows])),
            'coverage_68': None if not count else float(coverage_values.mean()),
            'bias_standard_error': None if count < 2 else float(realization_bias.std(ddof=1) / np.sqrt(count)),
            'realization_scatter': None if count < 2 else float(realization_bias.std(ddof=1)),
            'coverage_standard_error': None if count < 2 else float(coverage_values.std(ddof=1) / np.sqrt(count)),
            'coverage_status': 'UNRESOLVED' if count < 2 else 'EMPIRICAL_INDEPENDENT_REALIZATIONS',
        }
        if count and all((len(v) == 1 for v in grouped.values())):
            summary['coverage_wilson_68'] = _wilson(int(coverage_values.sum()), count)
        summaries.append(summary)
    return {'measurements': measurements, 'summaries': summaries}


def estimate_shared_sample_and_calibration_covariance(results, observable='H0', *, minimum_realizations, environment=None, model_hashes=None, redshift=None):
    """Estimate covariance across matched independent realization groups.

    Average repeated observers within each method/realization, then use the
    sample covariance C across shared realizations. C has (km/s/Mpc)^2 units;
    C/N is the covariance of the ensemble mean. Shared object/calibrator IDs
    are descriptive overlap counts and never a substitute for measured C.
    Different simulation conditions or within-method models cannot be pooled.
    """
    if type(minimum_realizations) is not int or minimum_realizations < 2:
        raise ValueError('Explicit minimum of at least two independent realizations required')
    groups = defaultdict(lambda: defaultdict(list))
    products = defaultdict(list)
    for p in results:
        matches = [key for key, value in p['observables'].items() if (key == observable or (observable == 'H_z' and key.startswith('H_z:'))) and (redshift is None or (value.get('redshift') is not None and np.isclose(value['redshift'], redshift, rtol=0, atol=1e-08)))]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError('Ambiguous cross-method observable/redshift match')
        matched_key = matches[0]
        if environment is not None and p['environment'] != environment:
            continue
        if model_hashes and fingerprint(p['model']) != model_hashes[p['method']]:
            continue
        rid = p['realization_id']
        if rid is None:
            raise ValueError('Cross-method empirical covariance needs explicit realization IDs')
        groups[p['method']][rid].append(p['observables'][matched_key]['value'])
        products[p['method']].append(p)
    methods = sorted(groups)
    if len(methods) < 2:
        raise ValueError('Cross-method covariance needs at least two methods')
    selected = [product for method in methods for product in products[method]]
    if len({fingerprint(product['environment']) for product in selected}) != 1 or len({fingerprint(product.get('comparison_condition', {})) for product in selected}) != 1:
        raise ValueError('Cross-method covariance requires one shared environment and simulation condition; compare conditions separately')
    if any((len({fingerprint(product['model']) for product in products[method]}) != 1 for method in methods)):
        raise ValueError('Cross-method covariance requires one declared estimator model per method')
    shared = set.intersection(*(set(groups[m]) for m in methods))
    if len(shared) < minimum_realizations or len(shared) <= len(methods):
        raise ValueError('Insufficient common independent realizations for cross-method covariance')
    ids = sorted(shared)
    x = np.asarray([[np.mean(groups[m][rid]) for m in methods] for rid in ids])
    c = covariance(np.cov(x, rowvar=False, ddof=1), len(methods))
    sigma = np.sqrt(np.diag(c))
    corr = c / sigma[:, None] / sigma[None, :]
    overlap = []
    for i, m in enumerate(methods):
        for n in methods[:i]:
            objects_m = {(p['namespace'], o) for p in products[m] for o in p['object_ids']}
            objects_n = {(p['namespace'], o) for p in products[n] for o in p['object_ids']}
            calibrators_m = {o for p in products[m] for o in p['calibration_ids']}
            calibrators_n = {o for p in products[n] for o in p['calibration_ids']}
            overlap.append({'methods': [m, n], 'shared_objects': len(objects_m & objects_n), 'shared_calibration_ids': sorted(calibrators_m & calibrators_n)})
    return {
        'observable': observable,
        'redshift': redshift,
        'methods': methods,
        'realization_ids': ids,
        'values': x.tolist(),
        'covariance': c.tolist(),
        'correlation': corr.tolist(),
        'sample_mean_covariance': (c / len(ids)).tolist(),
        'excluded_realizations': {m: sorted(set(groups[m]) - shared) for m in methods},
        'overlap': overlap,
        'n_independent_realizations': len(ids),
    }




def compare_methods_conditioned_on_halo_environment_and_model(results,truth):
    conditions=defaultdict(list)
    for p in results:
        conditions[(fingerprint(p['environment']),fingerprint(p.get('comparison_condition',{})))].append(p)
    return [{'environment':group[0]['environment'],'comparison_condition':group[0].get('comparison_condition',{}),
             'estimator_model_hashes':sorted({fingerprint(p['model']) for p in group}),
             'recovery':measure_bias_precision_coverage_and_redshift_dependence(group,truth)}
            for (environment,model),group in conditions.items()]


def paired_method_tensions(joint):
    x=np.asarray(joint['values']);c=np.asarray(joint['sample_mean_covariance']);means=x.mean(axis=0)
    rows=[]
    for i,m in enumerate(joint['methods']):
        for j,n in enumerate(joint['methods'][:i]):
            variance=c[i,i]+c[j,j]-2*c[i,j]
            if variance<=0:raise ValueError('Paired method difference variance is unresolved')
            rows.append({'methods':[m,n],'mean_difference':float(means[i]-means[j]),
                         'sigma_difference':float(np.sqrt(variance)),
                         'pull':float((means[i]-means[j])/np.sqrt(variance))})
    return rows


def build_followup_requests(recovery, budgets):
    """Report unmet authored budgets, including absent independent groups.

    No follow-up is launched here. A missing bias/coverage estimate remains
    unresolved; it must not crash reporting or be replaced with a zero bias.
    """
    followups = []
    for row in recovery['summaries']:
        controls = budgets.get(row['method'], {})
        reasons = []
        if row['n_independent_realizations'] < controls.get('minimum_realizations', 2):
            reasons.append('INDEPENDENT_REALIZATION_COVERAGE')
        bias = row['mean_bias']
        if bias is not None and abs(bias) > controls.get('maximum_absolute_bias', float('inf')):
            reasons.append('ESTIMATOR_BIAS')
        coverage = row['coverage_68']
        lower, upper = controls.get('coverage_range', [0, 1])
        if coverage is not None and not lower <= coverage <= upper:
            reasons.append('INTERVAL_COVERAGE')
        if reasons:
            limit = controls.get('followup_limit', 0)
            followups.append({
                'method': row['method'], 'observable': row['observable'], 'redshift': row['redshift'],
                'reasons': reasons,
                'requested_action': controls.get('refinement', 'increase independent realizations and inspect declared model mismatch'),
                'maximum_followups': limit, 'status': 'QUEUED' if limit > 0 else 'LIMIT_REACHED_UNRESOLVED',
            })
    return followups


def comparison_report_files(comparison):
    rows=comparison['recovery']['summaries'];csv_stream=io.StringIO()
    fields=['method','observable','redshift','n_products','n_independent_realizations','mean_bias',
            'mean_quoted_sigma','realization_scatter','coverage_68','coverage_status']
    writer=csv.DictWriter(csv_stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(row.get(k,'')))+'</td>' for k in fields)+'</tr>' for row in rows)
    report=('<!doctype html><html><meta charset="utf-8"><title>Expansion recovery</title>'
        '<style>body{font:16px system-ui;margin:2rem;color:#172333}table{border-collapse:collapse}'
        'td,th{padding:.6rem;border:1px solid #ccd}th{background:#eef}p{max-width:80ch}</style>'
        '<h1>Expansion recovery</h1><p>H0 and H(z) are reported separately. An H0 difference across redshift '
        'bins diagnoses the estimator and its model; it does not establish a varying fundamental H0.</p>'
        '<p>Execution success and scientific acceptance are separate. Coverage requires independent '
        'simulation realizations; shared observers and periodic replicas are not independent samples.</p>'
        '<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr></thead><tbody>'
        +table+'</tbody></table><h2>Follow-up requests</h2><pre>'+html.escape(json.dumps(comparison['followups'],indent=2))+'</pre></html>')
    return {'comparison.csv':csv_stream.getvalue(),'report.html':report}


def publish_comparison_figures(comparison,destination):
    """Generate shareable scientific plots when the user executes the report stage."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    rows=comparison['recovery']['measurements'];files=[]
    for family in ('H0','H_z'):
        subset=[r for r in rows if (r['observable']=='H0')==(family=='H0')]
        if not subset:continue
        fig,ax=plt.subplots(figsize=(9,5))
        methods=sorted({r['method'] for r in subset})
        for i,method in enumerate(methods):
            selected=[r for r in subset if r['method']==method]
            x=[i if family=='H0' else r['redshift'] for r in selected]
            ax.errorbar(x,[r['residual'] for r in selected],yerr=[r['sigma'] for r in selected],
                        fmt='o',alpha=.65,label=method,capsize=2)
        ax.axhline(0,color='black',lw=1);ax.set_ylabel('Recovered minus input expansion [km/s/Mpc]')
        ax.set_xlabel('Method' if family=='H0' else 'Redshift')
        if family=='H0':ax.set_xticks(range(len(methods)),methods,rotation=15)
        ax.legend();fig.tight_layout();path=destination/(family+'_recovery.png');fig.savefig(path,dpi=180);plt.close(fig)
        files.append(str(path))
    return files


def compare_expansion_methods(methods, evaluation_reader, budgets, *, minimum_shared_realizations):
    """Compare frozen estimators without allowing truth into their fits.

    Missing or incompatible joint covariance remains an explicit finding in a
    publishable comparison. This function never launches a queued refinement.
    """
    results = load_frozen_estimator_products(methods)
    truth = evaluation_reader.load_truth(results)
    recovery = measure_bias_precision_coverage_and_redshift_dependence(results, truth)
    source_hashes = {str(Path(__file__).resolve()): file_sha256(__file__)}
    for paths in methods.values():
        for source in paths:
            source = Path(source).resolve()
            source_hashes[str(source)] = file_sha256(source)
    joint = {}
    covariance_findings = []
    targets = set()
    for product in results:
        for key, value in product['observables'].items():
            if key == 'H0':
                targets.add(('H0', None))
            elif key == 'H_z' or key.startswith('H_z:'):
                targets.add(('H_z', value['redshift']))
    for key, z in sorted(targets, key=lambda x: (x[0], -1.0 if x[1] is None else x[1])):
        label = key if z is None else f'{key}@{z:.8g}'
        try:
            block = estimate_shared_sample_and_calibration_covariance(results, key, minimum_realizations=minimum_shared_realizations, redshift=z)
            block['paired_tensions'] = paired_method_tensions(block)
            joint[label] = block
        except ValueError as exc:
            covariance_findings.append({'observable': key, 'redshift': z, 'status': 'UNRESOLVED', 'reason': str(exc)})
    return {
        'schema': 'expansion-comparison-v1',
        'role': 'post_fit_evaluation',
        'recovery': recovery,
        'joint_covariance': joint,
        'covariance_findings': covariance_findings,
        'conditioned': compare_methods_conditioned_on_halo_environment_and_model(results, truth),
        'followups': build_followup_requests(recovery, budgets),
        'frozen_estimator_hashes': [p['frozen_hash'] for p in results],
        'truth_product_hash': evaluation_reader.expected_hash,
        'execution_status': 'COMPLETED',
        'source_hashes': source_hashes,
        'scientific_acceptance': 'UNRESOLVED_NOT_EXECUTED',
        'validation_receipts': [],
    }


DOMAIN_KEYS=('method','observable','redshift','model_hash','environment_hash')


def evaluate_expansion_acceptance(comparison, request):
    """Assess only exact, predeclared recovery domains.

    The bias bound is |mean_bias| + k*SE in the observable's km/s/Mpc units.
    Coverage averages observers within each independent realization first.
    Hoeffding bounds these [0,1] group means; Bonferroni splits the authored
    confidence across required domains. This is not a Gaussian error bar.
    """
    if comparison.get('schema') != 'expansion-comparison-v1' or comparison.get('role') != 'post_fit_evaluation' or comparison.get('execution_status') != 'COMPLETED':
        raise ValueError('Actual completed post-fit expansion recovery required')
    if request.get('schema') != 'expansion-acceptance-requirement-v1':
        raise ValueError('Frozen expansion acceptance requirements required')
    domains = request['required_domains']
    confidence = request['coverage_confidence']
    if not isinstance(domains, list) or not domains or len({canonical_digest(row) for row in domains}) != len(domains):
        raise ValueError('Distinct explicit expansion calibration domains required')
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or (not math.isfinite(confidence)) or (not 0 < confidence < 1):
        raise ValueError('Explicit coverage confidence in (0,1) required')
    if request.get('independence_scope') != 'distinct_native_realizations':
        raise ValueError('Expansion coverage requires distinct native realization groups')
    summaries = comparison['recovery']['summaries']
    rows = []
    family_alpha = (1 - confidence) / len(domains)
    for domain in domains:
        if set(domain) != set(DOMAIN_KEYS):
            raise ValueError('Expansion supported domain must retain exact method, observable, redshift, model and environment identity')
        matches = [row for row in summaries if all((row.get(key) == value for key, value in domain.items()))]
        if len(matches) != 1:
            rows.append({'domain': domain, 'status': 'UNRESOLVED', 'accepted': False, 'reason': 'exact measured recovery domain absent or duplicated'})
            continue
        measured = matches[0]
        control = request['method_requirements'][domain['method']]
        count = measured['n_independent_realizations']
        minimum = control['minimum_realizations']
        if type(count) is not int or count < 0 or type(minimum) is not int or (minimum < 2):
            raise ValueError('Explicit independent-realization counts required')
        multiplier = control['bias_sigma_multiplier']
        maximum_bias = control['maximum_absolute_bias']
        for value in (multiplier, maximum_bias):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (value < 0):
                raise ValueError('Frozen finite expansion bias controls required')
        if multiplier == 0:
            raise ValueError('Positive bias uncertainty multiplier required')
        coverage_range = control['coverage_range']
        if not isinstance(coverage_range, list) or len(coverage_range) != 2 or any((isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) for value in coverage_range)) or (not 0 <= coverage_range[0] < coverage_range[1] <= 1):
            raise ValueError('Frozen expansion coverage-equivalence range required')
        if count < minimum or measured['coverage_status'] != 'EMPIRICAL_INDEPENDENT_REALIZATIONS':
            rows.append({
                'domain': domain,
                'status': 'UNRESOLVED',
                'accepted': False,
                'reason': 'independent native realization count insufficient',
                'measured_count': count,
            })
            continue
        bias, se, coverage = (measured['mean_bias'], measured['bias_standard_error'], measured['coverage_68'])
        if any((isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) for value in (bias, se, coverage))) or se < 0 or (not 0 <= coverage <= 1):
            raise ValueError('Measured expansion bias, group uncertainty and coverage must be finite')
        bias_bound = abs(bias) + multiplier * se
        radius = math.sqrt(math.log(2 / family_alpha) / (2 * count))
        interval = [max(0.0, coverage - radius), min(1.0, coverage + radius)]
        observable = 'H_z' if domain['observable'].startswith('H_z:') else domain['observable']
        covariance_supported = not any((finding['observable'] == observable and finding.get('redshift') == domain['redshift'] for finding in comparison['covariance_findings']))
        accepted = bias_bound <= maximum_bias and interval[0] >= coverage_range[0] and (interval[1] <= coverage_range[1]) and covariance_supported
        rows.append({
            'domain': domain,
            'accepted': bool(accepted),
            'status': 'PASS' if accepted else 'FAIL',
            'measured_realization_count': count,
            'bias_upper_bound': bias_bound,
            'maximum_absolute_bias': maximum_bias,
            'group_average_coverage': coverage,
            'coverage_confidence_interval': interval,
            'required_coverage_range': coverage_range,
            'joint_covariance_supported': covariance_supported,
        })
    return {
        'schema': 'expansion-acceptance-v1',
        'execution_status': 'COMPLETE',
        'status': 'PASS' if all((row['accepted'] for row in rows)) else 'UNRESOLVED',
        'requirement_hash': canonical_digest(request),
        'comparison_hash': canonical_digest(comparison),
        'domains': rows,
        'accepted_domains': [row['domain'] for row in rows if row['accepted']],
        'remaining_exclusions': [row for row in rows if not row['accepted']],
        'coverage_uncertainty_method': 'bounded independent-realization group means; Hoeffding with Bonferroni domain-family correction',
        'bias_uncertainty_scope': 'frozen standard-error multiplier; no Gaussian confidence level inferred',
        'science_acceptance': 'DECLARED_RECOVERY_DOMAINS_ONLY',
    }


def expansion_acceptance_runner(parameters,context):
    from core.research.errors import _retained_json
    if type(parameters['memory_bytes']) is not int or not 0<parameters['memory_bytes']<=context.resources['memory_gib']*1024**3:
        raise MemoryError('Expansion acceptance exceeds allocated workspace')
    comparison,sources=_retained_json(parameters['comparison'],parameters['request_base'],parameters['memory_bytes'])
    result=evaluate_expansion_acceptance(comparison,parameters['request'])
    result['source_hashes']=sources
    atomic_json(context.generation/'expansion-acceptance.json',result)
    return {'measurements':{'accepted_domain_count':len(result['accepted_domains'])},'scientific_exclusions':result['remaining_exclusions']}


VARIANTS={'distance_ladder','time_delay_posteriors','hubble_method_comparison','bao_cmb_background'}


def validate_hubble_figure_stage(parameters):
    memory=parameters['memory_gib'];request=parameters['request']
    if type(memory) not in (int,float) or not math.isfinite(memory) or memory<=0:raise ValueError('Positive Hubble figure stage allocation required')
    if request.get('schema')!='native-hubble-figure-request-v1' or request.get('variant') not in VARIANTS:raise ValueError('Explicit registered Hubble figure request required')
    required={'schema','variant','bao_request' if request['variant']=='bao_cmb_background' else 'product'}
    if set(request)!=required:raise ValueError('Complete Hubble source request required')


def hubble_figure_runner(parameters,context):
    from .foundation import atomic_json
    validate_hubble_figure_stage(parameters)
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Hubble figure stage exceeds allocation')
    gallery=export_hubble_figures(parameters['request'],context.generation/'figures',memory_gib=parameters['memory_gib'])
    summary={'execution_status':'COMPLETE','variants':sorted({row['variant'] for row in gallery['figures']}),
        'figure_count':len(gallery['figures']),'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(context.generation/'hubble-figures.json',summary)
    return {'execution_status':'COMPLETE','measurements':{'hubble_figures':summary}}


def cosmography_generation_inputs(reference,*,memory_gib):
    from .observation_data import read_frozen,fingerprint
    from . import source_path
    if type(memory_gib) not in (int,float) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite Hubble figure allocation required')
    if set(reference)!={'path','sha256'}:raise ValueError('Exact original cosmography product binding required')
    path=Path(reference['path']).resolve();manifest=path.parent/'manifest.json'
    if path.name!='product.json' or not path.is_file() or path.is_symlink() or file_sha256(path)!=reference['sha256']:raise ValueError('Original owned cosmography product changed')
    if max(path.stat().st_size,manifest.stat().st_size)*24>memory_gib*1024**3/3:raise MemoryError('Original cosmography figure metadata exceeds allocation')
    original=json.loads(manifest.read_text());report=json.loads(path.read_text())
    if original.get('schema')!='cosmography-generation-v1' or original.get('execution_status')!='COMPLETED':raise ValueError('Completed original cosmography generation required')
    if fingerprint({'inputs':original['input_hashes'],'request':original['request'],'code':original['source_hashes']})!=original['fingerprint']:raise ValueError('Original cosmography publication fingerprint changed')
    pins={str(manifest):file_sha256(manifest)};root=Path(__file__).resolve().parents[1]
    for name,sha in original['files_sha256'].items():
        source=(path.parent/name).resolve()
        if not source.is_relative_to(path.parent) or file_sha256(source)!=sha:raise ValueError('Original cosmography generation payload changed or escaped')
        _merge(pins,{str(source):sha})
    if pins.get(str(path))!=reference['sha256']:raise ValueError('Cosmography product is outside its original publication')
    for name,sha in original['source_hashes'].items():
        source=Path(name);source=(root/source).resolve() if not source.is_absolute() else source.resolve()
        if file_sha256(source)!=sha:raise ValueError('Original cosmography producer source changed')
        _merge(pins,{str(source):sha})
    owner=source_path('observation_data').resolve()
    if pins.get(str(owner))!=file_sha256(owner):raise ValueError('Original cosmography publication source binding required')
    for source,sha in original['input_hashes'].items():
        if file_sha256(source)!=sha:raise ValueError('Original cosmography measurement/model input changed')
        _merge(pins,{source:sha})
    if report.get('role')=='frozen_estimator_result':read_frozen(report)
    return report,original,pins


def hubble_figure_arrays(request,*,memory_gib):
    variant=request['variant']
    if variant not in VARIANTS:raise ValueError('Explicit registered Hubble variant required')
    if variant=='bao_cmb_background':
        from .bao import conditional_bao_inputs
        if set(request)!={'schema','variant','bao_request'}:raise ValueError('Original conditional BAO request required')
        source={**request['bao_request'],'memory_gib':memory_gib};report,values,pins=conditional_bao_inputs(source)
        n=len(values['redshift']);cov=np.asarray(values['measured_covariance'],float)
        if cov.shape!=(n,n) or not np.isfinite(cov).all():raise ValueError('Original measured BAO covariance shape differs')
        support=np.asarray(values['input_available'])
        if support.dtype!=np.dtype(bool) or support.shape!=(n,):raise ValueError('Original boolean BAO row availability required')
        arrays={**values,'quoted_sigma':np.sqrt(np.diag(cov)),'supported':support}
        for label in sorted(set(report['ordered_quantity_labels'])):
            arrays['supported_'+label]=support&np.array([value==label for value in report['ordered_quantity_labels']])
        units={key:'dimensionless' for key in arrays}
        metadata={'original_producer':report,'ordered_quantity_labels':report['ordered_quantity_labels'],
            'scope':'Original measured full-covariance BAO likelihood against fixed CLASS model. No CMB measurement likelihood, H0 posterior or fitted cosmological parameters inferred.',
            'cmb_likelihood_status':'NOT_SUPPLIED','posterior_status':'NOT_GENERATED'}
        return arrays,units,metadata,pins
    if set(request)!={'schema','variant','product'}:raise ValueError('Original completed cosmography product reference required')
    report,manifest,pins=cosmography_generation_inputs(request['product'],memory_gib=memory_gib)
    metadata={'original_producer':report,'original_stage':manifest['request'],'science_acceptance':'NOT_ACCEPTED'}
    if variant=='distance_ladder':
        if report.get('schema')!='cosmography-frozen-result-v1' or report.get('method')!='distance_ladder':raise ValueError('Original frozen distance-ladder estimator required')
        measured=report['observables']['H0'];value=float(measured['value']);low,high=map(float,measured['interval_68'])
        sigma=float(measured['sigma'])
        if measured['units']!='km/s/Mpc' or not np.isfinite([value,low,high,sigma]).all() or not 0<low<=value<=high or sigma<=0:raise ValueError('Original physical H0 value and interval required')
        arrays={'estimate_index':np.array([0.]),'h0_mean':np.array([value]),'h0_sigma':np.array([sigma]),
            'interval_index':np.zeros(3),'h0_interval':np.array([low,value,high]),'supported':np.array([True]),'interval_supported':np.ones(3,bool)}
        units={key:'dimensionless' if 'index' in key or 'supported' in key else 'km/s/Mpc' for key in arrays}
        metadata.update(scope='Original jointly fitted ladder H0 and interval, with original calibration, covariance and selection. Local-curvature uncertainty is not relabeled as posterior draws.',posterior_status='NOT_GENERATED_BY_THIS_ADAPTER')
    elif variant=='time_delay_posteriors':
        fit=report.get('fit',{})
        if (report.get('schema')!='cosmography-frozen-result-v1' or report.get('method')!='time_delay'
                or not str(fit.get('posterior_status','')).startswith('SAMPLED')):raise ValueError('Actual full imaging/time-delay sampled posterior required; reduced Gaussian draws or point fits cannot substitute')
        h0=np.asarray(fit['H0_samples'],float);distance=np.asarray(fit['D_dt_mpc_samples'],float)
        if h0.ndim!=1 or len(h0)<2 or distance.shape!=h0.shape or not np.isfinite(h0).all() or not np.isfinite(distance).all() or np.any(h0<=0) or np.any(distance<=0):raise ValueError('Positive aligned actual lens posterior samples required')
        measured=report['observables']['H0']
        if measured['units']!='km/s/Mpc' or not np.isclose(np.mean(h0),measured['value'],rtol=1e-12,atol=0) or not np.isclose(np.std(h0,ddof=1),measured['sigma'],rtol=1e-12,atol=0):raise ValueError('Original lens H0 summary differs from its actual posterior draws')
        arrays={'h0_samples':h0,'distance_samples':distance,'h0_sorted':np.sort(h0),'distance_sorted':np.sort(distance),
            'cumulative_probability':np.arange(1,len(h0)+1,dtype=float)/len(h0),'supported':np.ones(len(h0),bool)}
        units={key:'km/s/Mpc' if key.startswith('h0') else 'Mpc' if key.startswith('distance') else 'dimensionless' for key in arrays}
        metadata.update(scope='Actual retained full lens-model MCMC draws; original expansion-shape, external-convergence, kinematic and selection assumptions retained. Convergence and independent coverage remain separate.',posterior_status=fit['posterior_status'])
    else:
        if report.get('schema')!='expansion-comparison-v1' or report.get('role')!='post_fit_evaluation':raise ValueError('Original post-fit expansion-method comparison required')
        rows=report['recovery']['measurements']
        if not rows:raise ValueError('Original comparison has no measured estimator recovery')
        names=sorted({row['method'] for row in rows});arrays={};units={}
        for label,selected in [('h0',[row for row in rows if row['observable']=='H0']),('hz',[row for row in rows if row['observable']=='H_z' or row['observable'].startswith('H_z:')])]:
            if not selected:continue
            x=np.array([names.index(row['method']) if label=='h0' else row['redshift'] for row in selected],float)
            residual=np.array([row['residual'] for row in selected],float);sigma=np.array([row['sigma'] for row in selected],float)
            if not np.isfinite(x).all() or not np.isfinite(residual).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):raise ValueError('Original H0/H(z) comparison values are invalid')
            arrays.update({label+'_coordinate':x,label+'_residual':residual,label+'_sigma':sigma,label+'_supported':np.ones(len(x),bool)})
            units.update({label+'_coordinate':'dimensionless',label+'_residual':'km/s/Mpc',label+'_sigma':'km/s/Mpc',label+'_supported':'dimensionless'})
        if not arrays:raise ValueError('Original comparison has no H0 or H(z) recovery rows')
        metadata.update(method_index_labels=names,scope='Original privileged post-fit recovery residuals. H0 and H(z) remain separate; shared-realization/calibration covariance findings and coverage limitations retained.')
    return arrays,units,metadata,pins


def export_hubble_figures(request,destination,*,memory_gib=1.):
    from .data import export_numeric_audit
    from core.research.display import export_scientific_suite
    if request.get('schema')!='native-hubble-figure-request-v1':raise ValueError('Explicit Hubble figure request required')
    arrays,units,metadata,pins=hubble_figure_arrays(request,memory_gib=memory_gib);out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Preserve previous Hubble figure export')
    out.mkdir(parents=True);_merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    variant=request['variant'];epoch={'meaning':'Original observational or post-fit source epochs; see original producer'}
    target=out/'hubble-measurements.json'
    export_numeric_audit(target,{'schema':'native-hubble-figure-product-v1','variant':variant,'epoch':epoch,'science_acceptance':'NOT_ACCEPTED',**metadata},arrays,
        {key:{'units':unit,'quantity':key,'missing_policy':'Original measured values/support; no fabricated posterior or calibration'} for key,unit in units.items()},algorithm=__file__,context={'epoch':epoch},source_hashes=pins)
    panels=[]
    def panel(name,title,x,y,xlabel,ylabel,support,kind='scatter',**extra):
        return {'panel_id':name,'title':title,'quality_status':'ORIGINAL_SOURCE_SCOPE','selection':metadata['scope'],
            'xlabel':xlabel+' ['+units[x]+']','ylabel':ylabel+' ['+units[y]+']','x_units':units[x],'y_units':units[y],
            'series':[{'source':'measurement','kind':kind,'x':x,'y':y,'support':support,'label':title,**extra}]}
    if variant=='distance_ladder':
        p=panel('ladder','Original ladder H0 interval; local fit uncertainty','interval_index','h0_interval','Ladder estimate index','H0','interval_supported','line')
        p['series'].append({'source':'measurement','kind':'scatter','x':'estimate_index','y':'h0_mean','support':'supported','label':'Original jointly fitted H0'})
        panels.append(p)
    elif variant=='time_delay_posteriors':
        for label,x,title in [('h0','h0_sorted','Full lens posterior: H0'),('ddt','distance_sorted','Full lens posterior: time-delay distance')]:
            panels.append(panel(label,title,x,'cumulative_probability','H0' if label=='h0' else 'Time-delay distance','Empirical cumulative probability','supported','line'))
    elif variant=='bao_cmb_background':
        for label in sorted(set(metadata['ordered_quantity_labels'])):
            p=panel(label,label+': measured BAO and fixed CLASS model','redshift','measured_ratio','Observed redshift',label,'supported_'+label,'errorbar',error='quoted_sigma')
            p['series'].append({'source':'measurement','kind':'scatter','x':'redshift','y':'model_ratio','support':'supported_'+label,'label':'Fixed CLASS prediction; no CMB likelihood'})
            panels.append(p)
    else:
        for label,title in [('h0','H0 recovery by estimator'),('hz','H(z) recovery at measured redshifts')]:
            if label+'_coordinate' in arrays:panels.append(panel(label,title,label+'_coordinate',label+'_residual','Method index' if label=='h0' else 'Measured redshift','Recovered minus original input',label+'_supported','errorbar',error=label+'_sigma'))
    if any(file_sha256(p)!=h for p,h in pins.items()):raise ValueError('Hubble source changed during export')
    return export_scientific_suite({'sources':{'measurement':str(target)},'dpi':160,'maximum_panels_per_figure':4,'organized_outputs':True,
        'figures':[{'figure_id':'native-'+variant,'family':'hubble','variant':variant,'title':variant.replace('_',' '),
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':panels}]},out/'gallery',base=out,memory_gib=memory_gib)
