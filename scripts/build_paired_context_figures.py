#!/usr/bin/env python3
"""Plot the retained linear solver and saved nonlinear histories in common units."""
import argparse
import csv
import json
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json, file_sha256
from core.paired_atlas import _write_csv

MODELS = ('EDE', 'LCDM')
COLORS = ('#b65e1b', '#087d83')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    root = parser.parse_args().atlas.resolve()
    out = root / '01_Comparisons/EDE_vs_LCDM_z_0199_to_0000_cosmology_and_evolution'
    out.mkdir(exist_ok=True)
    source_records, figures = {}, []

    def save(fig, label):
        name = f'EDE_vs_LCDM_{label}'
        for suffix in ('png', 'svg', 'pdf'):
            fig.savefig(out / f'{name}.{suffix}', dpi=175)
        figures.append(name + '.png')
        plt.close(fig)

    data = {}
    for model in MODELS:
        key = model.lower()
        run = root.parent / f'dmo3_N128_L5_fast128_{key}'
        config = yaml.safe_load((run / 'config.yaml').read_text())
        h = float(config['cosmology']['H0'])/100
        path = run / 'input' / f'{key}_cosmology_products.h5'
        source_records[model] = {'path': str(path), 'sha256': file_sha256(path), 'h': h}
        with h5py.File(path) as f:
            bg, pk, tf = (f[k] for k in ('background', 'linear_power_spectrum', 'transfer_functions'))
            if not all(np.isclose(float(p.attrs['actual_redshift']), 199) for p in (pk, tf)):
                raise ValueError('Unexpected linear output redshift')
            columns = tf['columns']
            def column(prefix):
                return columns[next(k for k in columns if k.startswith(prefix))][:]
            data[model] = {'a': bg['scale_factor'][:], 'H': bg['H_km_s_Mpc'][:], 'D': bg['D1_CLASS'][:],
                           'f': bg['f1_CLASS'][:], 'pk_k': h * pk['k_h_Mpc'][:], 'P': pk['P_mpc_h3'][:] / h**3,
                           'tf_k': h * column('k_h_Mpc_'), 'd_m': column('d_tot_'),
                           'd_b': column('d_b_'), 'd_cdm': column('d_cdm_'),
                           'theta_b': column('t_b_'), 'theta_cdm': column('t_cdm_')}
        np.savez_compressed(out / f'{model}_z_0199_to_0000_native_solver_arrays.npz', **data[model])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    fig.suptitle('At the initial saved epoch · z = 199', fontsize=20)
    for model, color in zip(MODELS, COLORS):
        d = data[model]
        axes[0, 0].loglog(d['pk_k'], d['P'], color=color, label=model)
        for i, kind in enumerate(('d_m', 'd_b', 'd_cdm'), 1):
            ax = axes.flat[i]
            ax.semilogx(d['tf_k'], d[kind], color=color, label=model)
            ax.set_title({'d_m': 'Total solver density', 'd_b': 'Baryon solver species', 'd_cdm': 'Cold dark matter solver species'}[kind])
            ax.set_ylabel('Signed density transfer')
        _write_csv(out / f'{model}_z_0199_linear_power.csv', [
            {'k_mpc_inverse': k, 'power_mpc3': p} for k, p in zip(d['pk_k'], d['P'])])
        _write_csv(out / f'{model}_z_0199_signed_density_transfers.csv', [
            dict(zip(('k_mpc_inverse', 'total_solver_density', 'baryon', 'cold_dark_matter'), row))
            for row in zip(d['tf_k'], d['d_m'], d['d_b'], d['d_cdm'])])
    axes[0, 0].set(title='Linear matter power', ylabel='P(k) [Mpc³]')
    for ax in axes.flat:
        ax.set_xlabel('Comoving k [Mpc⁻¹]')
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    save(fig, 'z_0199_linear_power_and_signed_density_transfers')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    fig.suptitle('Signed velocity-divergence transfers · initial z = 199', fontsize=19)
    for model, color in zip(MODELS, COLORS):
        d = data[model]
        for ax, key, label in zip(axes, ('theta_b', 'theta_cdm'), ('Baryon θ', 'Cold dark matter θ')):
            ax.semilogx(d['tf_k'], d[key], color=color, label=model)
            ax.set(xlabel='Comoving k [Mpc⁻¹]', ylabel='Signed θ · native solver normalization', title=label)
            ax.grid(alpha=.2)
        _write_csv(out / f'{model}_z_0199_signed_velocity_divergence_transfers.csv', [
            {'k_mpc_inverse': k, 'theta_b_native': tb, 'theta_cdm_native': tc}
            for k, tb, tc in zip(d['tf_k'], d['theta_b'], d['theta_cdm'])])
    axes[0].legend(frameon=False)
    fig.supxlabel('Retained CLASS/AxiCLASS columns t_b and t_cdm, normalized to initial curvature = 1 in the original solver table.', fontsize=10)
    save(fig, 'z_0199_signed_velocity_divergence_transfers')

    a = np.geomspace(.005, 1., 600)
    shared = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    fig.suptitle('Expansion and linear growth · z = 199 to 0', fontsize=19)
    for model, color in zip(MODELS, COLORS):
        d = data[model]
        shared[model] = {key: np.interp(np.log(a), np.log(d['a']), d[key]) for key in ('H', 'D', 'f')}
        for ax, key, ylabel in zip(axes, ('H', 'D', 'f'), ('H [km s⁻¹ Mpc⁻¹]', 'D₁ (today = 1)', 'f₁')):
            ax.semilogx(a, shared[model][key], label=model, color=color)
            ax.set(xlabel='Scale factor a', ylabel=ylabel)
            ax.grid(alpha=.22)
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[0].legend()
    _write_csv(out / 'EDE_vs_LCDM_z_0199_to_0000_expansion_and_growth.csv', [
        {'scale_factor': x, 'redshift': 1/x-1, **{f'{key}_{model}': shared[model][key][i]
         for model in MODELS for key in ('H', 'D', 'f')}} for i, x in enumerate(a)])
    save(fig, 'z_0199_to_0000_expansion_and_growth')

    timeline = root / '06_Tables_and_Data/EDE_vs_LCDM_All_86_epochs_comparison.csv'
    with timeline.open() as stream:
        rows = list(csv.DictReader(stream))
    source_records['saved_output_comparison'] = {'path': str(timeline.relative_to(root)), 'sha256': file_sha256(timeline)}
    a = 1 / (1 + np.asarray([float(r['redshift_EDE']) for r in rows]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    fig.suptitle('How the two saved universes evolved · 86 outputs per model', fontsize=19)
    for ax, key, label in zip(axes.flat, ('fof_hosts_300', 'largest_fof_mass_msun', 'subhalo_count', 'particle_speed_rms_km_s'),
                              ('FOF hosts with ≥300 particles', 'Most massive FOF host [Msun]', 'Native SUBFIND rows', 'DM RMS peculiar speed [km/s]')):
        for model, color in zip(MODELS, COLORS):
            y = np.asarray([float(row[f'{key}_{model}']) for row in rows])
            ax.plot(a, y, label=model, color=color, linewidth=1.7)
        ax.axvline(1/15.4, color='#756a82', linestyle='--', linewidth=1, label='z = 14.4')
        ax.set(xscale='log', yscale='symlog', xlabel='Scale factor a', ylabel=label)
        ax.grid(alpha=.22)
    axes[0, 0].legend()
    save(fig, 'z_0199_to_0000_all_86_saved_epoch_evolution')
    atomic_json(out / 'context_manifest.json', {'sources': source_records, 'figures': figures,
        'tables': sorted(p.name for p in out.glob('*.csv')),
        'scope': 'Full cosmological parameter-set contrast. Common units; solver curves are deterministic. Background interpolated in log scale factor within source coverage.'})


if __name__ == '__main__':
    main()
