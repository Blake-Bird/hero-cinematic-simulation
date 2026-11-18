"""Standalone linear HMF predictions and reproducible cosmology screening.

Every epoch uses its own solver P(k), not a scale-independent growth rescaling.
Formula spread and cosmology outliers are selection diagnostics, not errors.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import json
import itertools
import math
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import qmc

from core.foundation import atomic_json, file_sha256
from core.hmf_fitting_functions import LABELS, binned_predictions


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to export')
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_request(spec):
    c = spec['config']['cosmology']
    if c['model'] not in {'lcdm', 'ede'}:
        raise ValueError('This screening workflow supports lcdm and ede')
    for key in ('H0', 'omega_b', 'omega_cdm', 'A_s', 'n_s', 'tau_reio'):
        if not np.isfinite(float(c[key])) or float(c[key]) <= 0:
            raise ValueError('Positive finite cosmology.' + key + ' required')
    if int(c.get('N_ncdm', 0) or 0) != 0:
        raise ValueError('Massive-neutrino matter/sigma conventions are not supported here')
    protected = {'H0', 'h', 'omega_b', 'omega_cdm', 'Omega_b', 'Omega_cdm', 'A_s', 'sigma8',
                 'n_s', 'tau_reio', 'N_ncdm', 'm_ncdm', 'T_ncdm', 'N_ur', 'T_cmb'}
    extras = [c.get('extra', {}) or {}, spec['config'].get('ede', {}).get('extra_axiclass', {}) or {}]
    if any(protected.intersection(extra) for extra in extras):
        raise ValueError('Extra solver parameters cannot override declared cosmology features')
    z = np.asarray(spec['redshifts'], float)
    edges = np.asarray(spec['mass_edges_msun_h'], float)
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all() or np.any(z < 0) or len(np.unique(z)) != len(z):
        raise ValueError('Distinct finite nonnegative redshifts required')
    if edges.ndim != 1 or len(edges) < 3 or not np.isfinite(edges).all() or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError('At least two ordered positive mass bins required')
    fits = spec.get('fits', list(LABELS))
    if not fits or len(set(fits)) != len(fits) or set(fits) - LABELS.keys():
        raise ValueError('Unknown, repeated or empty fitting-function selection')
    for key in ('k_max_h_mpc', 'flag_ratio', 'timeout_seconds'):
        value = float(spec.get(key, {'k_max_h_mpc': 100, 'flag_ratio': 2, 'timeout_seconds': 1800}[key]))
        if not np.isfinite(value) or value <= (1 if key == 'flag_ratio' else 0):
            raise ValueError('Invalid ' + key)
    if c['model'] == 'ede':
        from core.cosmology import ede_parameters
        ede_parameters(spec['config'])


def solve_epoch(config, z, work, kmax, binary=None, timeout=1800):
    """Run a read-only solver executable in a private working directory.

    Shared solver source directories are never written; independent processes
    can execute concurrently. Binary and outputs are checksummed in each receipt.
    """
    from core.cosmology import class_common, ede_parameters, executable, write_ini, class_output_files
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    model = config['cosmology']['model']
    engine = 'axiclass' if model == 'ede' else 'class'
    binary = Path(binary).expanduser().resolve() if binary else executable(engine)
    if binary is None or not binary.is_file():
        raise FileNotFoundError(engine + ' executable missing; run setup or pass --binary')
    identity = file_sha256(binary)
    params = class_common(config, float(z), Path('linear_'), float(kmax))
    if model == 'ede':
        params.update(ede_parameters(config))
    write_ini(params, work/'input.ini')
    environment = os.environ.copy()
    environment['OMP_NUM_THREADS'] = '1'
    with (work/'solver.log').open('w') as log:
        subprocess.run([str(binary), 'input.ini'], cwd=work, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    if file_sha256(binary) != identity:
        raise ValueError('Solver executable changed during execution')
    outputs = class_output_files(work/'linear_', z)
    table = np.loadtxt(outputs['pk'], comments='#', ndmin=2)
    if (len(table) < 8 or not np.isfinite(table[:, :2]).all()
            or np.any(table[:, :2] <= 0) or np.any(np.diff(table[:, 0]) <= 0)):
        raise ValueError('Invalid linear spectrum')
    atomic_json(work/'receipt.json', {
        'schema': 'hmf-linear-epoch-v1', 'engine': engine, 'binary': str(binary),
        'binary_sha256': identity, 'parameters': params, 'redshift': float(z),
        'files': {p.name: file_sha256(p) for p in [work/'input.ini', work/'solver.log', *outputs.values()]},
        'science_acceptance': 'NOT_ACCEPTED'})
    return table[:, 0], table[:, 1]


def predict_sample(spec, sample_id, output, binary=None):
    validate_request(spec)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'request.json', spec)
    c = spec['config']['cosmology']
    omega = (float(c['omega_b']) + float(c['omega_cdm']))/(float(c['H0'])/100)**2
    edges = np.asarray(spec['mass_edges_msun_h'], float)
    rows = []
    for epoch, z in enumerate(spec['redshifts']):
        if spec.get('linear_bundle'):
            k, power = saved_bundle_epoch(spec, z, output/f'epoch_{epoch:03d}')
        else:
            k, power = solve_epoch(spec['config'], z, output/f'epoch_{epoch:03d}',
                                   spec.get('k_max_h_mpc', 100), binary, spec.get('timeout_seconds', 1800))
        fits, sigma, slope, covered = binned_predictions(k, power, omega, edges, z)
        for name in spec.get('fits', list(LABELS)):
            for i in range(len(edges)-1):
                rate = fits[name]['dn_dlog10M'][i]
                rows.append(dict(sample_id=sample_id, redshift=float(z), fit=name,
                    mass_lower_msun_h=edges[i], mass_upper_msun_h=edges[i+1],
                    mass_center_msun_h=np.sqrt(edges[i]*edges[i+1]),
                    dn_dlog10M_h3_mpc3=float(rate) if np.isfinite(rate) else '',
                    sigma=float(sigma[i]) if np.isfinite(sigma[i]) else '',
                    slope=float(slope[i]) if np.isfinite(slope[i]) else '',
                    k_covered=int(covered[i]), table_range=int(fits[name]['table_range_mask'][i]),
                    cosmology_scope='CDM_REFERENCE' if c['model'] == 'lcdm' else 'EDE_EXTRAPOLATION'))
    write_csv(output/'predictions.csv', rows)
    return rows


def saved_bundle_epoch(spec, z, output):
    """HMF-only reuse of an integrity-checked standalone CLASS bundle.

    Reuse requires the same physical solver parameters and an exact saved
    epoch. No growth extrapolation or density-convention conversion occurs.
    """
    from core.linear import verify_spectrum_directory
    from core.cosmology import class_common, ede_parameters, class_redshift_matches, pk_output_redshift
    root = Path(spec['linear_bundle']).expanduser().resolve()
    record = verify_spectrum_directory(root)
    desired = class_common(spec['config'], z, Path('unused_'), spec.get('k_max_h_mpc', 100))
    if spec['config']['cosmology']['model'] == 'ede':
        desired.update(ede_parameters(spec['config']))
    if record['model'] != spec['config']['cosmology']['model']:
        raise ValueError('Saved linear bundle model differs from the request')
    ignored = {'root', 'z_pk', 'P_k_max_h/Mpc', 'output', 'write background', 'write parameters', 'headers', 'format'}
    # These are irrelevant defaults for a solver with no ncdm species.
    ignored |= {'N_ncdm', 'm_ncdm', 'T_ncdm'}
    matches = []
    for epoch, files in record['source_files'].items():
        relative = Path(files['pk'])
        path = (root/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or relative.as_posix() not in record['checksums_sha256']:
            raise ValueError('Saved spectrum is outside the verified bundle')
        if class_redshift_matches(pk_output_redshift(path), z):
            matches.append((epoch, path))
    if not matches:
        raise ValueError('Requested redshift has no saved spectrum in the linear bundle')
    epoch, path = matches[0]
    actual = record['class_parameters'][epoch]
    if float(actual.get('N_ncdm', 0)) != 0:
        raise ValueError('Massive-neutrino saved spectra are unsupported')
    for key in (set(actual) | set(desired))-ignored:
        if key not in actual or key not in desired:
            raise ValueError('Saved cosmology parameter differs: '+key)
        try:
            same = np.isclose(float(actual[key]), float(desired[key]), rtol=1e-12, atol=0)
        except (ValueError, TypeError):
            same = str(actual[key]) == str(desired[key])
        if not same:
            raise ValueError('Saved cosmology parameter differs: '+key)
    table = np.loadtxt(path, comments='#', ndmin=2)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'receipt.json', dict(source_bundle=str(root), source_spectrum=str(path),
        spectrum_sha256=file_sha256(path), manifest_sha256=file_sha256(root/'spectrum.json'),
        redshift=float(z), method='Exact saved linear spectrum, no growth rescaling'))
    return table[:, 0], table[:, 1]


def score_rows(rows, flag_ratio=2):
    """Compare only positive, k-covered, listed-range bins on identical grids."""
    groups = {}
    for row in rows:
        if row['k_covered'] and row['table_range'] and row['dn_dlog10M_h3_mpc3'] != '' and row['dn_dlog10M_h3_mpc3'] > 0:
            key = (row['redshift'], row['mass_lower_msun_h'], row['mass_upper_msun_h'])
            groups.setdefault(key, []).append(row)
    scores = []
    for key, cell in sorted(groups.items()):
        # Compare cosmologies with the same formula, using median abundance.
        medians = {}
        for name in {r['fit'] for r in cell}:
            subset = [r for r in cell if r['fit'] == name]
            if len({r['sample_id'] for r in subset}) >= 3:
                medians[name] = float(np.median([r['dn_dlog10M_h3_mpc3'] for r in subset]))
        for sample in sorted({r['sample_id'] for r in cell}):
            subset = [r for r in cell if r['sample_id'] == sample]
            values = [r['dn_dlog10M_h3_mpc3'] for r in subset]
            log_spread = math.log(max(values))-math.log(min(values)) if len(values) >= 2 else None
            spread = math.exp(min(log_spread, 700)) if log_spread is not None else None
            ratios = [(math.log(r['dn_dlog10M_h3_mpc3'])-math.log(medians[r['fit']]), r['fit']) for r in subset if r['fit'] in medians]
            extreme = max(ratios, key=lambda p: abs(p[0])) if ratios else None
            factor = math.exp(min(abs(extreme[0]), 700)) if extreme else None
            scores.append(dict(sample_id=sample, redshift=key[0], mass_lower_msun_h=key[1],
                mass_upper_msun_h=key[2], eligible_fits=len(values), fit_max_over_min=spread,
                eligible_fit_names=','.join(sorted(r['fit'] for r in subset)),
                cosmology_factor_from_median=factor,
                cosmology_ratio_to_median=math.exp(max(-700, min(extreme[0], 700))) if extreme else None,
                fit_log10_spread=log_spread/math.log(10) if log_spread is not None else None,
                cosmology_log10_factor=abs(extreme[0])/math.log(10) if extreme else None,
                ratio_display_capped=bool((log_spread is not None and log_spread > 700) or (extreme and abs(extreme[0]) > 700)),
                extreme_fit=extreme[1] if extreme else None,
                flag_fit_disagreement=log_spread is not None and log_spread >= math.log(flag_ratio),
                flag_cosmology_outlier=extreme is not None and abs(extreme[0]) >= math.log(flag_ratio)))
    return scores


def mark_unusual_disagreement(scores, settings=None):
    """Flag excess *formula spread*, compared with the same fits at the same bin.

    Use a robust ensemble threshold in log10 space. Changes in which formulas
    are in domain must never masquerade as an unusual cosmology.
    """
    settings = settings or {}
    minimum = settings.get('minimum_cosmologies', 5)
    multiplier = float(settings.get('mad_multiplier', 3.))
    floor = float(settings.get('minimum_excess_dex', .05))
    if type(minimum) is not int or minimum < 3 or not np.isfinite([multiplier, floor]).all() or multiplier <= 0 or floor <= 0:
        raise ValueError('Disagreement selection needs >=3 cosmologies, positive MAD multiplier and excess floor')
    peers = {}
    for row in scores:
        key = (row['redshift'], row['mass_lower_msun_h'], row['mass_upper_msun_h'], row['eligible_fit_names'])
        if row['fit_log10_spread'] is not None:
            peers.setdefault(key, []).append(row)
        row.update(flag_unusual_disagreement=False, disagreement_peer_count=0,
                   usual_log10_spread=None, unusual_log10_threshold=None, disagreement_excess_dex=None)
    for cell in peers.values():
        values = np.array([r['fit_log10_spread'] for r in cell])
        median = float(np.median(values))
        mad = float(np.median(np.abs(values-median)))
        threshold = median + max(multiplier*1.4826*mad, floor)
        for row in cell:
            row.update(disagreement_peer_count=len(cell), usual_log10_spread=median,
                       unusual_log10_threshold=threshold,
                       disagreement_excess_dex=row['fit_log10_spread']-median,
                       flag_unusual_disagreement=len(cell) >= minimum and row['fit_log10_spread'] > threshold)
    return scores


def export_candidates(scores, requests, output, settings=None):
    """One record per cosmology, plus a reusable CLASS→GADGET handoff.

    This selects experiments; it does not start expensive particle simulations.
    """
    settings = settings or {}
    criterion = settings.get('criterion', 'unusual')
    if criterion not in {'unusual', 'absolute', 'either'}:
        raise ValueError('selection.criterion must be unusual, absolute or either')
    chosen = {}
    for row in scores:
        selected = ((criterion in {'unusual', 'either'} and row['flag_unusual_disagreement'])
                    or (criterion in {'absolute', 'either'} and row['flag_fit_disagreement']))
        if selected:
            chosen.setdefault(row['sample_id'], []).append(row)
    result = []
    for sample, triggers in chosen.items():
        request = requests[int(sample.rsplit('_', 1)[1])]
        folder = Path(output)/'candidates'/sample
        folder.mkdir(parents=True, exist_ok=False)
        c = request['config']['cosmology']
        h = float(c['H0'])/100
        start_z = float(settings.get('gadget_start_redshift', max(49, max(request['redshifts']))))
        if start_z < max(request['redshifts']):
            raise ValueError('GADGET start redshift must cover all requested measurement epochs')
        linear = dict(name=sample, model=c['model'], H0=float(c['H0']), Omega_b=float(c['omega_b'])/h**2,
            Omega_cdm=float(c['omega_cdm'])/h**2, A_s=float(c['A_s']), n_s=float(c['n_s']), tau_reio=float(c['tau_reio']),
            z_start=start_z, k_max_h_mpc=request.get('k_max_h_mpc', 100))
        if c['model'] == 'ede':
            e = request['config']['ede']
            linear.update(f_ede=e['f_ede'], log10_zc=math.log10(e['z_c']), theta_i=e['theta_i'], n_axion=e['n_axion'])
        # The existing standalone CLASS command has fixed radiation defaults
        # and no generic extra parameters. Never silently alter a candidate.
        compatible = (not c.get('extra') and not request['config'].get('ede', {}).get('extra_axiclass')
                      and float(c.get('T_cmb', 2.7255)) == 2.7255 and float(c.get('N_ur', 3.046)) == 3.046
                      and 5 <= start_z <= 1000)
        reason = None if compatible else 'Standalone class interface cannot preserve all requested solver settings'
        if compatible:
            from core.linear import validate
            try:
                validate(copy.deepcopy(linear))
            except ValueError as exc:
                compatible, reason = False, str(exc)
        atomic_json(folder/'cosmology.json', request['config'])
        if compatible:
            atomic_json(folder/'linear-config.json', linear)
        item = dict(sample_id=sample, triggers=triggers, config=request['config'],
            largest_excess_dex=max(r['disagreement_excess_dex'] or 0 for r in triggers),
            linear_config=str(folder/'linear-config.json') if compatible else None,
            handoff_status='READY_FOR_CLASS_THEN_GADGET' if compatible else 'USE_NATIVE_CONFIG_TO_PRESERVE_EXTRA_PARAMETERS',
            handoff_reason=reason,
            measurement_redshifts=request['redshifts'],
            next_step='Generate starting-epoch CLASS bundle, then run normal GADGET with your chosen box, 512/1024 particles per dimension, seed and halo catalogue settings.')
        atomic_json(folder/'selection.json', item)
        result.append(item)
    return sorted(result, key=lambda r: r['largest_excess_dex'], reverse=True)


def sampled_requests(spec):
    sampling = spec.get('sampling', {})
    n = sampling.get('count', 1)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('sampling.count must be a positive integer')
    bounds = sampling.get('bounds', {})
    allowed = {'cosmology.'+s for s in ('H0', 'omega_b', 'omega_cdm', 'A_s', 'n_s', 'tau_reio')}
    allowed |= {'ede.'+s for s in ('f_ede', 'z_c', 'theta_i', 'n_axion')}
    if set(bounds) - allowed:
        raise ValueError('Unsupported sampling path')
    if bounds and spec['config']['cosmology']['model'] != 'ede' and any(p.startswith('ede.') for p in bounds):
        raise ValueError('EDE bounds require model ede')
    keys = sorted(bounds)
    method = sampling.get('method', 'latin_hypercube')
    if method == 'grid':
        grid = sampling.get('grid', {})
        if bounds or not grid or set(grid)-allowed:
            raise ValueError('Grid sampling requires allowed parameter value lists in sampling.grid, without bounds')
        if spec['config']['cosmology']['model'] != 'ede' and any(p.startswith('ede.') for p in grid):
            raise ValueError('EDE grid requires model ede')
        grid_keys = sorted(grid)
        for key in grid_keys:
            values = np.asarray(grid[key], float)
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or len(np.unique(values)) != len(values):
                raise ValueError('Grid parameter values must be distinct finite numbers')
        total = math.prod(len(grid[k]) for k in grid_keys)
        limit = sampling.get('max_samples', 10000)
        if type(limit) is not int or limit < 1 or total > limit:
            raise ValueError('Grid exceeds sampling.max_samples')
        if 'count' in sampling and sampling['count'] != total:
            raise ValueError('Grid count is determined by its Cartesian product; omit count or match it')
        requests = []
        for values in itertools.product(*(grid[key] for key in grid_keys)):
            request = copy.deepcopy(spec)
            for path, value in zip(grid_keys, values):
                section, name = path.split('.')
                request['config'][section][name] = float(value)
            validate_request(request)
            requests.append(request)
        return requests
    if method not in {'latin_hypercube', 'uniform'}:
        raise ValueError('Use latin_hypercube, uniform or grid sampling')
    seed = sampling.get('seed', 42)
    unit = (qmc.LatinHypercube(len(keys), seed=seed).random(n) if keys and method == 'latin_hypercube'
            else np.random.default_rng(seed).random((n, len(keys))))
    for j, path in enumerate(keys):
        low, high = map(float, bounds[path])
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError('Finite ordered sampling bounds required: ' + path)
        unit[:, j] = low + unit[:, j]*(high-low)
    result = []
    for i in range(n):
        request = copy.deepcopy(spec)
        for j, path in enumerate(keys):
            section, name = path.split('.')
            request['config'][section][name] = float(unit[i, j])
        validate_request(request)
        result.append(request)
    return result


def plot_campaign(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for epoch, z in enumerate(sorted({r['redshift'] for r in rows})):
        fig, ax = plt.subplots(figsize=(9, 6))
        for name in sorted({r['fit'] for r in rows}):
            for index, sample in enumerate(sorted({r['sample_id'] for r in rows})):
                subset = [r for r in rows if r['fit'] == name and r['sample_id'] == sample and r['redshift'] == z]
                valid = [r for r in subset if r['table_range'] and r['k_covered'] and r['dn_dlog10M_h3_mpc3'] != '' and r['dn_dlog10M_h3_mpc3'] > 0]
                ax.plot([r['mass_center_msun_h'] for r in valid], [r['dn_dlog10M_h3_mpc3'] for r in valid],
                        '.', alpha=.5, label=LABELS[name] if index == 0 else None)
        ax.set(xscale='log', yscale='log', xlabel='Mass [Msun/h]', ylabel='dn/dlog10M [(h/Mpc)^3]',
               title=f'z={z:g}: listed-range bins only; diagnostic (EDE is extrapolation)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(output)/f'hmf_z{epoch:03d}.png', dpi=150)
        plt.close(fig)


def run_campaign(spec, output, workers=1, binary=None):
    requests = sampled_requests(spec)
    mark_unusual_disagreement([], spec.get('selection'))
    export_candidates([], requests, output, spec.get('selection'))
    start = spec.get('selection', {}).get('gadget_start_redshift')
    if start is not None and (not np.isfinite(float(start)) or float(start) < max(spec['redshifts'])):
        raise ValueError('Finite GADGET start redshift must cover requested epochs')
    if spec.get('linear_bundle') and (len(requests) != 1 or spec.get('sampling', {}).get('bounds')):
        raise ValueError('A saved linear bundle supports one fixed cosmology; remove sampling bounds')
    if isinstance(workers, bool) or workers < 1 or workers > (os.cpu_count() or 1):
        raise ValueError('workers must be between 1 and the available CPU count')
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'request.json', spec)
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(predict_sample, request, f'cosmo_{i:05d}', output/f'cosmo_{i:05d}', binary): i
                   for i, request in enumerate(requests)}
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception as exc:
                failures.append({'sample_id': f'cosmo_{futures[future]:05d}', 'error': str(exc)})
    rows.sort(key=lambda r: (r['sample_id'], r['redshift'], r['fit'], r['mass_lower_msun_h']))
    scores = mark_unusual_disagreement(score_rows(rows, spec.get('flag_ratio', 2)), spec.get('selection'))
    if rows:
        write_csv(output/'predictions.csv', rows)
        plot_campaign(rows, output)
    if scores:
        write_csv(output/'scores.csv', scores)
    candidates = export_candidates(scores, requests, output, spec.get('selection'))
    atomic_json(output/'gadget_candidates.json', dict(schema='hmf-gadget-candidates-v1', candidates=candidates,
        selection=spec.get('selection', {'criterion': 'unusual'}), training_data=False))
    report = dict(schema='hmf-screening-v1', status='FAILED' if not rows else 'PARTIAL' if failures else 'COMPLETE',
        requested_samples=len(requests), successful_samples=len(requests)-len(failures), failures=failures,
        eligible_cells=len(scores), candidates=candidates, science_acceptance='NOT_ACCEPTED',
        interpretation='Select GADGET experiments from unusually large disagreement BETWEEN formulas. Default: ensemble median + max(3*1.4826*MAD, 0.05 dex), >=5 peers with identical fit sets. Abundance outliers are separate diagnostics, not default candidate selectors. Solver predictions are not training measurements.')
    atomic_json(output/'report.json', report)
    return report


def hmf_main():
    parser = argparse.ArgumentParser(description='CLASS/AxiCLASS → HMF only; no particle simulation. See docs/hmf-research.md.')
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--linear-bundle', type=Path, help='HMF only: reuse exact epochs from a verified commands.py class bundle')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.linear_bundle:
        spec['linear_bundle'] = str(args.linear_bundle.expanduser().resolve())
    report = run_campaign(spec, args.output, args.workers, args.binary)
    print(json.dumps({k: v for k, v in report.items() if k != 'candidates'}, indent=2))
    return 0 if report['status'] == 'COMPLETE' else 1
