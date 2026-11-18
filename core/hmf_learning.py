"""Measured HMF datasets and an interpretable Poisson polynomial baseline.

Targets are simulation counts with exposure V*dlog10M. Analytic predictions
are never accepted as measured targets. Splits hold out entire cosmologies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import yaml

from core.foundation import atomic_json, file_sha256
from core.hmf_campaign import write_csv

BASE_FEATURES = ['log10_mass', 'log1p_redshift', 'H0', 'omega_b', 'omega_cdm',
                 'log10_A_s', 'n_s', 'f_ede', 'log10_z_c', 'theta_i', 'n_axion']
PHYSICAL_FEATURES = {'feature_sigma', 'feature_variance_slope', 'feature_n_eff',
                     'feature_H_over_H0', 'feature_growth_D', 'feature_growth_f'}


def cosmology_features(config):
    c = config['cosmology']
    if c['model'] not in {'lcdm', 'ede'}:
        raise ValueError('Dataset cosmology must be lcdm or ede')
    values = {key: float(c[key]) for key in ('H0', 'omega_b', 'omega_cdm', 'n_s')}
    values['log10_A_s'] = float(np.log10(float(c['A_s'])))
    values.update(f_ede=0., log10_z_c=0., theta_i=0., n_axion=0.)
    if c['model'] == 'ede':
        from core.cosmology import ede_parameters
        ede_parameters(config)
        e = config['ede']
        values.update(f_ede=float(e['f_ede']), log10_z_c=float(np.log10(e['z_c'])),
                      theta_i=float(e['theta_i']), n_axion=float(e['n_axion']))
    if not np.isfinite(list(values.values())).all() or any(values[k] <= 0 for k in ('H0', 'omega_b', 'omega_cdm')):
        raise ValueError('Invalid cosmology features')
    # Include all declared physical parameters, including unselected features.
    physical_c = {k: v for k, v in c.items() if k != 'engine'}
    defaults = dict(T_cmb=2.7255, N_ur=3.046, N_ncdm=0, m_ncdm_eV=0., T_ncdm=.71611)
    for key in ('H0', 'omega_b', 'omega_cdm', 'A_s', 'n_s', 'tau_reio', *defaults):
        value = c.get(key, defaults.get(key))
        physical_c[key] = float(value or 0.)
        if not np.isfinite(physical_c[key]):
            raise ValueError('Nonfinite cosmological parameter ' + key)
    physical_c['extra'] = c.get('extra') or {}
    physical_e = ({key: float(config['ede'][key]) for key in ('f_ede', 'z_c', 'theta_i', 'n_axion')}
                  if c['model'] == 'ede' else {})
    if c['model'] == 'ede':
        physical_e['extra_axiclass'] = config['ede'].get('extra_axiclass') or {}
    physical = {'cosmology': physical_c, 'ede': physical_e}
    group = hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest()
    return values, group


def load_source(source):
    """Read a native run, or a portable folder containing metadata.json + hmf.csv.

    Portable CSV uses the native results/hmf/snapshot_*_hmf.csv column names.
    Optional numeric feature_* columns travel to the dataset unchanged.
    """
    root = Path(source['path']).expanduser().resolve()
    if (root/'metadata.json').is_file():
        metadata_path = root/'metadata.json'
        meta = json.loads(metadata_path.read_text())
        if meta.get('target_kind') != 'simulation_counts':
            raise ValueError('metadata.target_kind must be simulation_counts')
        if meta.get('measurement_sha256') and file_sha256(root/'hmf.csv') != meta['measurement_sha256']:
            raise ValueError('Exported GADGET measurements changed')
        config = meta['config']
        finder, physics = meta['finder'], meta['physics']
        tables = [root/'hmf.csv']
    else:
        metadata_path = root/'config.yaml'
        config = yaml.safe_load(metadata_path.read_text())
        # Native results/hmf is produced from GADGET Group* fields, even when
        # a separate Rockstar catalogue was also requested.
        finder = 'gadget_fof_subfind'
        physics = config['simulation']['physics']
        meta = {'target_kind': 'simulation_counts'}
        tables = sorted((root/'results'/'hmf').glob('snapshot_*_hmf.csv'))
    if not tables or not finder or not physics:
        raise ValueError('Measured HMF tables, finder and physics must be declared: ' + str(root))
    features, group = cosmology_features(config)
    rows, hashes = [], {str(metadata_path): file_sha256(metadata_path)}
    for table in tables:
        hashes[str(table)] = file_sha256(table)
        with table.open(newline='') as stream:
            for raw in csv.DictReader(stream):
                lo, hi, count, volume, z, particle_mass, minimum = [float(raw[k]) for k in (
                    'mass_lower_msun_h', 'mass_upper_msun_h', 'halo_count', 'volume_mpc_h3',
                    'redshift', 'particle_mass_msun_h', 'minimum_halo_particles')]
                if (not np.isfinite([lo, hi, count, volume, z, particle_mass, minimum]).all()
                        or lo <= 0 or hi <= lo or count < 0 or count != int(count) or volume <= 0
                        or z < 0 or particle_mass <= 0 or minimum < 1 or minimum != int(minimum)):
                    raise ValueError('Invalid measured bin in ' + str(table))
                if lo < particle_mass*minimum:
                    continue
                definition = raw['mass_definition'].lower()
                if definition not in {'fof', 'm200c', 'm200b', 'mvir'}:
                    raise ValueError('Unknown halo mass definition')
                if definition == 'fof' and not np.isclose(float(config.get('halos', {}).get('fof_linking_length', np.nan)), .2):
                    raise ValueError('Declare halos.fof_linking_length=0.2 for this FOF dataset')
                simulation = config.get('simulation', {})
                row = dict(source=str(root), source_table=str(table), cosmology_group=group,
                    split=source.get('split', ''), finder=finder, physics=physics, mass_definition=definition,
                    model=config['cosmology']['model'], mass_lower_msun_h=lo, mass_upper_msun_h=hi,
                    redshift=z, count=int(count), volume_mpc_h3=volume,
                    particle_mass_msun_h=particle_mass, minimum_halo_particles=int(minimum),
                    particles_per_dimension=raw.get('particles_per_dimension', simulation.get('particles_per_dimension', '')),
                    random_seed=raw.get('random_seed', simulation.get('random_seed', '')),
                    box_size_mpc_h=raw.get('box_size_mpc_h', simulation.get('box_mpc_h', '')),
                    exposure=volume*np.log10(hi/lo), log10_mass=float(np.log10(np.sqrt(lo*hi))),
                    log1p_redshift=float(np.log1p(z)), **features)
                for key, value in raw.items():
                    if key.startswith('feature_'):
                        row[key] = float(value)
                        if not np.isfinite(row[key]):
                            raise ValueError('Nonfinite extra feature: ' + key)
                rows.append(row)
    return rows, hashes


def build_dataset(spec, output):
    rows, sources, seen, table_identities = [], {}, set(), set()
    entries = list(spec.get('sources', []))
    for folder in spec.get('source_roots', []):
        root = Path(folder['path']).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        discovered = {p.parent for p in root.rglob('config.yaml') if list((p.parent/'results'/'hmf').glob('snapshot_*_hmf.csv'))}
        discovered |= {p.parent for p in root.rglob('metadata.json') if (p.parent/'hmf.csv').is_file()}
        entries.extend(dict(path=str(p), **({'split': folder['split']} if 'split' in folder else {})) for p in sorted(discovered))
    for source in entries:
        path = str(Path(source['path']).expanduser().resolve())
        if path in seen:
            raise ValueError('Duplicate source folder')
        seen.add(path)
        loaded, hashes = load_source(source)
        for table in {r['source_table'] for r in loaded}:
            identity = (loaded[0]['cosmology_group'], hashes[table])
            if identity in table_identities:
                raise ValueError('Duplicate measured table content across source folders')
            table_identities.add(identity)
        rows.extend(loaded)
        sources.update(hashes)
    if not rows:
        raise ValueError('No resolved measured bins remain')
    conventions = {(r['finder'], r['physics'], r['mass_definition']) for r in rows}
    if len(conventions) != 1:
        raise ValueError('Build separate datasets for each finder, physics and mass definition')
    if any(set(r) != set(rows[0]) for r in rows):
        raise ValueError('Every source must supply the same optional feature columns')
    keys = [(r['cosmology_group'], r['source'], r['redshift'], r['mass_lower_msun_h'], r['mass_upper_msun_h']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate measured bins in source tables')
    rows, resolution_checks = select_resolution_rows(rows, spec.get('resolution_policy', 'finest_per_bin'))
    groups = sorted({r['cosmology_group'] for r in rows})
    declared = {r['split'] for r in rows}
    if declared == {''}:
        if len(groups) < 5:
            raise ValueError('Automatic train/validation/test split needs >=5 distinct cosmologies')
        np.random.default_rng(spec.get('seed', 42)).shuffle(groups)
        n_test = max(1, int(.2*len(groups)))
        n_val = max(1, int(.2*len(groups)))
        splits = {g: 'test' if i < n_test else 'validation' if i < n_test+n_val else 'train'
                  for i, g in enumerate(groups)}
        for row in rows:
            row['split'] = splits[row['cosmology_group']]
    elif declared != {'train', 'validation', 'test'}:
        raise ValueError('Declare all sources as train, validation or test, or leave every split automatic')
    for group in groups:
        if len({r['split'] for r in rows if r['cosmology_group'] == group}) != 1:
            raise ValueError('Cosmology leakage across splits')
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    if spec.get('linear_features', {}).get('enabled', False):
        enrich_linear_features(rows, output/'linear_features', spec['linear_features'])
    # A new empirical function learns simulation measurements. Existing
    # analytic formula outputs are excluded, even if present as feature_*.
    additional = set(spec.get('additional_physical_features', []))
    if any(not f.startswith('feature_') or any(token in f.lower() for token in ('fit', 'prediction', 'reference', 'target', 'count', 'density')) for f in additional):
        raise ValueError('Additional inputs must be physical features, not fitting formula outputs or targets')
    if additional-set(rows[0]):
        raise ValueError('Missing additional physical feature columns')
    physical = sorted((PHYSICAL_FEATURES | additional) & set(rows[0]))
    write_csv(output/'dataset.csv', rows)
    if resolution_checks:
        write_csv(output/'resolution_comparison.csv', resolution_checks)
    report = dict(schema='hmf-measured-dataset-v1', target_kind='simulation_counts',
        rows=len(rows), cosmologies=len(groups), conventions=dict(zip(('finder', 'physics', 'mass_definition'), next(iter(conventions)))),
        units={'mass': 'Msun/h', 'volume': '(Mpc/h)^3', 'abundance': '(h/Mpc)^3 per dex'},
        features=BASE_FEATURES + physical,
        additional_physical_features=sorted(additional),
        excluded_formula_or_unapproved_features=sorted(k for k in rows[0] if k.startswith('feature_') and k not in physical),
        target_origin='Measured simulation halo counts. No CLASS or existing fitting-function abundance targets or formula features.',
        resolution_policy=spec.get('resolution_policy', 'finest_per_bin'),
        matched_resolution_comparisons=len(resolution_checks),
        resolution_rows={str(n): sum(str(r['particles_per_dimension']) == str(n) for r in rows)
                         for n in sorted({str(r['particles_per_dimension']) for r in rows})},
        split_rows={s: sum(r['split'] == s for r in rows) for s in ('train', 'validation', 'test')},
        sources=sources, dataset_sha256=file_sha256(output/'dataset.csv'),
        limitations='Particle cuts do not establish convergence. Poisson likelihood omits cosmic covariance. No halo mass conversions. Different resolutions and seeds of one cosmology remain in one split.')
    atomic_json(output/'metadata.json', report)
    return report


def select_resolution_rows(rows, policy='finest_per_bin'):
    """Compare matched 512/1024 etc. counts without counting both as new seeds.

    Retain highest available resolution per same cosmology, seed, box, epoch,
    mass bin. Different seeds and volumes are distinct measurements. Rows
    without resolution metadata remain usable but cannot enter this check.
    """
    if policy not in {'finest_per_bin', 'retain_all'}:
        raise ValueError('resolution_policy must be finest_per_bin or retain_all')
    cells, retained, comparisons = {}, [], []
    for row in rows:
        if row['particles_per_dimension'] == '' or row['random_seed'] == '' or row['box_size_mpc_h'] == '':
            retained.append(row)
            continue
        n = float(row['particles_per_dimension'])
        if not np.isfinite(n) or n < 1 or n != int(n):
            raise ValueError('Invalid simulation resolution metadata')
        key = (row['cosmology_group'], str(row['random_seed']), float(row['box_size_mpc_h']),
               row['redshift'], row['mass_lower_msun_h'], row['mass_upper_msun_h'])
        cells.setdefault(key, []).append(row)
    for cell in cells.values():
        resolutions = [int(float(r['particles_per_dimension'])) for r in cell]
        if len(set(resolutions)) != len(resolutions):
            raise ValueError('Duplicate simulation realization at the same resolution, seed, box and epoch')
        finest = max(cell, key=lambda r: int(float(r['particles_per_dimension'])))
        retained.extend(cell if policy == 'retain_all' else [finest])
        for row in cell:
            if row is finest:
                continue
            comparisons.append(dict(cosmology_group=row['cosmology_group'], random_seed=row['random_seed'],
                box_size_mpc_h=row['box_size_mpc_h'], redshift=row['redshift'],
                mass_lower_msun_h=row['mass_lower_msun_h'], mass_upper_msun_h=row['mass_upper_msun_h'],
                lower_resolution=row['particles_per_dimension'], finest_resolution=finest['particles_per_dimension'],
                lower_count=row['count'], finest_count=finest['count'],
                count_ratio_to_finest=row['count']/finest['count'] if finest['count'] else '',
                lower_source=row['source'], finest_source=finest['source']))
    return retained, comparisons


def enrich_linear_features(rows, output, settings):
    """Add fresh epoch-specific sigma, spectral slope and expansion/growth.

    Reuse one solve for identical cosmology/epoch across simulation seeds.
    All native spectra, background tables and receipts remain in the dataset.
    """
    from core.hmf_campaign import solve_epoch
    from core.cosmology import compute_sigma_and_dlnsigma, class_background_table
    groups = {}
    for row in rows:
        groups.setdefault((row['cosmology_group'], row['redshift']), []).append(row)
    for index, key in enumerate(sorted(groups)):
        _, z = key
        cell = groups[key]
        root = Path(cell[0]['source'])
        config = (json.loads((root/'metadata.json').read_text())['config'] if (root/'metadata.json').is_file()
                  else yaml.safe_load((root/'config.yaml').read_text()))
        work = output/f'epoch_{index:06d}'
        k, power = solve_epoch(config, z, work, settings.get('k_max_h_mpc', 100),
                               settings.get('binary'), settings.get('timeout_seconds', 1800))
        c = config['cosmology']
        omega = (float(c['omega_b'])+float(c['omega_cdm']))/(float(c['H0'])/100)**2
        masses = np.array([10**r['log10_mass'] for r in cell])
        sigma, slope = compute_sigma_and_dlnsigma(k, power, masses, omega)
        background = list(work.glob('linear_*background.dat'))
        if len(background) != 1:
            raise ValueError('Expected one solver background table')
        columns, table = class_background_table(background[0])
        a = 1/(1+table[:, columns['z']])
        order = np.argsort(a)
        a, table = a[order], table[order]
        target = 1/(1+z)
        if a[0] > target or a[-1] < 1 or np.any(np.diff(a) <= 0):
            raise ValueError('Background does not cover the feature epoch and today')
        hubble = float(np.interp(np.log(target), np.log(a), table[:, columns['H']]))
        h0 = float(np.interp(0., np.log(a), table[:, columns['H']]))
        growth = float(np.interp(np.log(target), np.log(a), table[:, columns['growth_D']]))
        d0 = float(np.interp(0., np.log(a), table[:, columns['growth_D']]))
        rate = float(np.interp(np.log(target), np.log(a), table[:, columns['growth_f']]))
        for row, sig, derivative in zip(cell, sigma, slope):
            extra = dict(feature_sigma=float(sig), feature_variance_slope=float(derivative),
                         feature_n_eff=float(6*derivative-3), feature_H_over_H0=hubble/h0,
                         feature_growth_D=growth/d0, feature_growth_f=rate)
            if any(name in row for name in extra):
                raise ValueError('Linear enrichment would overwrite supplied features')
            if not np.isfinite(list(extra.values())).all():
                raise ValueError('Nonfinite solver features')
            row.update(extra)


def powers_for(n_features, degree):
    terms = [()]
    for d in range(1, degree+1):
        terms.extend(itertools.combinations_with_replacement(range(n_features), d))
    return terms


def design(x, powers):
    return np.column_stack([np.prod(x[:, term], axis=1) if term else np.ones(len(x)) for term in powers])


def fit_poisson(x, counts, exposure, ridge):
    if counts.sum() <= 0:
        raise ValueError('Training data contains no halos')
    initial = np.zeros(x.shape[1])
    initial[0] = np.log(counts.sum()/exposure.sum())

    def objective(beta):
        eta = x@beta + np.log(exposure)
        # Optimizer line searches can try enormous coefficients. Reject those
        # trials instead of clipping the physical model and its gradient.
        if np.max(eta) > 650:
            return np.inf, np.zeros_like(beta)
        mu = np.exp(eta)
        penalty = beta.copy()
        penalty[0] = 0
        return ((mu-counts*eta).sum()/len(counts) + .5*ridge*np.dot(penalty, penalty),
                x.T@(mu-counts)/len(counts) + ridge*penalty)
    def hessian(beta):
        mu = np.exp(np.minimum(x@beta + np.log(exposure), 650))
        diagonal = np.full(x.shape[1], ridge)
        diagonal[0] = 0
        return x.T@(mu[:, None]*x)/len(counts) + np.diag(diagonal)

    # Quadratic feature sets often include strongly correlated growth/slope
    # columns. An exact convex Poisson Hessian avoids slow quasi-Newton fits.
    result = minimize(objective, initial, jac=True, hess=hessian, method='trust-exact',
                      options={'maxiter': 1000, 'gtol': 1e-6})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Poisson fit did not converge: ' + result.message)
    return result.x


def poisson_deviance(counts, expected):
    term = expected-counts
    positive = counts > 0
    term[positive] += counts[positive]*np.log(counts[positive]/expected[positive])
    return float(2*np.maximum(term, 0).sum()/len(counts))


def train_dataset(dataset, output, degree=2, features=None):
    dataset = Path(dataset).expanduser().resolve()
    meta = json.loads((dataset/'metadata.json').read_text())
    if meta.get('schema') != 'hmf-measured-dataset-v1' or file_sha256(dataset/'dataset.csv') != meta['dataset_sha256']:
        raise ValueError('Measured dataset identity check failed')
    if degree not in (1, 2):
        raise ValueError('Baseline degree must be 1 or 2')
    with (dataset/'dataset.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    permitted = set(BASE_FEATURES) | PHYSICAL_FEATURES | set(meta.get('additional_physical_features', []))
    features = features or [f for f in meta['features'] if f in permitted]
    if not features or len(set(features)) != len(features) or set(features)-set(meta['features']):
        raise ValueError('Select unique declared numeric features')
    if set(features)-permitted:
        raise ValueError('Train on physical descriptors, not existing fitting-function abundance predictions')
    masks = {s: np.array([r['split'] == s for r in rows]) for s in ('train', 'validation', 'test')}
    if not all(mask.any() for mask in masks.values()):
        raise ValueError('All three splits must be nonempty')
    raw = np.array([[float(r[f]) for f in features] for r in rows])
    counts = np.array([float(r['count']) for r in rows])
    exposure = np.array([float(r['exposure']) for r in rows])
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite model features')
    mean, scale = raw[masks['train']].mean(axis=0), raw[masks['train']].std(axis=0)
    active = scale > 1e-12
    # Constant EDE features in an LCDM-only training set must not be silently
    # treated as learned EDE sensitivity at prediction time.
    scale = np.where(active, scale, 1.)
    powers = powers_for(int(active.sum()), degree)
    x = design(((raw-mean)/scale)[:, active], powers)
    trials = []
    for ridge in (1e-5, 1e-3, .1, 1.):
        beta = fit_poisson(x[masks['train']], counts[masks['train']], exposure[masks['train']], ridge)
        use = masks['validation']
        eta = x[use]@beta + np.log(exposure[use])
        dev = poisson_deviance(counts[use], np.exp(np.clip(eta, -700, 650)))
        trials.append((dev, ridge, beta))
    dev, ridge, beta = min(trials, key=lambda item: item[0])
    eta = x@beta
    if np.any(~np.isfinite(eta)) or np.max(eta) > 650:
        raise ValueError('Unstable fitted model')
    density = np.exp(np.maximum(eta, -700))
    selected = [f for f, keep in zip(features, active) if keep]
    limits = {f: [float(raw[masks['train'], j].min()), float(raw[masks['train'], j].max())] for j, f in enumerate(features)}
    outside = np.zeros(len(rows), bool)
    for j, f in enumerate(features):
        low, high = limits[f]
        outside |= (raw[:, j] < low-1e-12) | (raw[:, j] > high+1e-12)
    model = dict(schema='hmf-poisson-polynomial-v1', features=features, active_features=selected,
        mean=mean.tolist(), scale=scale.tolist(), powers=[list(p) for p in powers], coefficients=beta.tolist(),
        degree=degree, ridge=ridge, training_feature_bounds=limits, conventions=meta['conventions'],
        dataset_sha256=meta['dataset_sha256'], science_acceptance='NOT_ACCEPTED',
        equation='dn/dlog10M = exp(sum_t coefficient[t] * product_j standardized_feature[powers[t][j]])); standardized_feature=(feature-mean)/scale',
        likelihood='Poisson counts, mean=volume*dlog10M*predicted_bin_average_density; bin-center feature approximation',
        limitations='Interpretable baseline, not a precision calibration. No claim outside training features/mass/redshift. Constant features have no learned sensitivity. Cosmic covariance and bin averaging errors require scientific validation.',
        metrics={s: {'mean_poisson_deviance': poisson_deviance(counts[use], density[use]*exposure[use]),
                     'outside_training_feature_bounds': int(outside[use].sum()),
                     'rows': int(use.sum()), 'cosmologies': len({r['cosmology_group'] for r, keep in zip(rows, use) if keep})}
                 for s, use in masks.items()},
        validation_trials=[{'ridge': r, 'mean_poisson_deviance': d} for d, r, _ in trials])
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'model.json', model)
    evaluated = []
    for row, prediction, extrapolated in zip(rows, density, outside):
        evaluated.append(dict(source=row['source'], cosmology_group=row['cosmology_group'], split=row['split'],
            redshift=row['redshift'], mass_lower_msun_h=row['mass_lower_msun_h'], mass_upper_msun_h=row['mass_upper_msun_h'],
            count=row['count'], expected_count=float(prediction*float(row['exposure'])),
            predicted_dn_dlog10M_h3_mpc3=float(prediction), outside_training_feature_bounds=int(extrapolated)))
    write_csv(output/'evaluation.csv', evaluated)
    return model


def predict_density(model, feature_rows, *, allow_extrapolation=False):
    raw = np.array([[float(row[f]) for f in model['features']] for row in feature_rows])
    if not np.isfinite(raw).all():
        raise ValueError('Finite prediction features required')
    outside = np.zeros(len(raw), bool)
    for i, f in enumerate(model['features']):
        lo, hi = model['training_feature_bounds'][f]
        outside |= (raw[:, i] < lo-1e-12) | (raw[:, i] > hi+1e-12)
    if outside.any() and not allow_extrapolation:
        raise ValueError('Prediction outside training feature domain')
    active = [model['features'].index(f) for f in model['active_features']]
    standardized = (raw-np.asarray(model['mean']))/np.asarray(model['scale'])
    log_density = design(standardized[:, active], model['powers'])@np.asarray(model['coefficients'])
    if not np.isfinite(log_density).all() or np.max(log_density) > 650:
        raise ValueError('Unstable prediction')
    return np.exp(np.maximum(log_density, -700)), outside


def learning_main():
    parser = argparse.ArgumentParser(description='Measured HMF dataset → explicit Poisson fitting equation. See docs/hmf-research.md.')
    sub = parser.add_subparsers(dest='action', required=True)
    export = sub.add_parser('export-gadget', help='Measure host counts directly from saved full-box GADGET catalogues')
    export_source = export.add_mutually_exclusive_group(required=True)
    export_source.add_argument('--run', type=Path)
    export_source.add_argument('--runs-root', type=Path, help='Recursively export many saved GADGET runs')
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--snapshots', type=int, nargs='+')
    build = sub.add_parser('build')
    build.add_argument('--spec', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    train = sub.add_parser('train')
    train.add_argument('--dataset', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--degree', type=int, choices=(1, 2), default=2)
    train.add_argument('--features', nargs='+')
    predict = sub.add_parser('predict')
    predict.add_argument('--model', type=Path, required=True)
    predict.add_argument('--features-csv', type=Path, required=True)
    predict.add_argument('--output', type=Path, required=True)
    predict.add_argument('--allow-extrapolation', action='store_true')
    args = parser.parse_args()
    if args.action == 'export-gadget':
        from core.gadget_hmf_data import export_gadget_measurements, export_gadget_tree
        if args.runs_root:
            report = export_gadget_tree(args.runs_root, args.output, args.snapshots)
            print(json.dumps(report, indent=2))
            return 0 if report['status'] == 'COMPLETE' else 1
        else:
            report = export_gadget_measurements(args.run, args.output, args.snapshots)
            print(json.dumps({'target_origin': report['target_origin'], 'particles_per_dimension': report['particles_per_dimension'],
                              'particle_count': report['particle_count'], 'epochs': len(report['epochs'])}, indent=2))
    elif args.action == 'build':
        report = build_dataset(json.loads(args.spec.read_text()), args.output)
        print(json.dumps(report, indent=2))
    elif args.action == 'train':
        model = train_dataset(args.dataset, args.output, args.degree, args.features)
        print(json.dumps(model['metrics'], indent=2))
    else:
        model = json.loads(args.model.read_text())
        with args.features_csv.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        density, outside = predict_density(model, rows, allow_extrapolation=args.allow_extrapolation)
        if args.output.exists():
            raise FileExistsError(args.output)
        write_csv(args.output, [dict(**r, predicted_dn_dlog10M_h3_mpc3=float(d), extrapolation=int(o)) for r, d, o in zip(rows, density, outside)])
    return 0
