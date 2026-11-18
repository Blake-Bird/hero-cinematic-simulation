#!/usr/bin/env python3
"""Summarize the native z=0 FOF fitting-function checks with explicit support."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json, file_sha256
from core.paired_atlas import _write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    root = parser.parse_args().atlas.resolve()
    output = root / '01_Comparisons/EDE_vs_LCDM_z_0000p000_snap_085'
    records, scores, sources = [], [], {}
    fits = ['press_schechter', 'sheth_tormen', 'reed_2007', 'watson_fof', 'jenkins', 'warren']
    colors = ['#f3c97a', '#57d5bf', '#b79cec', '#f18b93', '#65b6ed', '#9dd077']
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True, layout='constrained')
    for model, h, ax in zip(('EDE', 'LCDM'), (.7483, .6781), axes):
        path, = (root / 'Source/Native_workspaces' / model).glob('results/analysis_products_*/hmf_fitting_functions/comparison_085.csv')
        sources[model] = {'path': str(path.relative_to(root)), 'sha256': file_sha256(path)}
        with path.open() as stream:
            bins = list(csv.DictReader(stream))
        ax.axhspan(.8, 1.2, color='#e5e5e5', alpha=.45, label='±20% guide')
        ax.axhline(1, color='#555555', linewidth=.9)
        for fit, color in zip(fits, colors):
            selected = []
            for row in bins:
                count = int(row['count'])
                supported = row['k_covered'] == '1' and row[f'{fit}_table_range'] == '1'
                expectation = float(row[f'{fit}_expected_count'])
                ratio = float(row[f'{fit}_measurement_over_fit'])
                record = {'model': model, 'fit': fit, 'mass_msun': float(row['mass_msun_h']) / h,
                          'bin_left_msun': float(row['bin_left_msun_h']) / h,
                          'bin_right_msun': float(row['bin_right_msun_h']) / h,
                          'measured_count': count, 'expected_count': expectation,
                          'measurement_over_fit': ratio, 'within_input_k_and_fit_range': supported,
                          'reviewed_bin': supported and count >= 5 and np.isfinite(ratio) and ratio > 0}
                records.append(record)
                if record['reviewed_bin']:
                    selected.append(record)
            ratios = np.asarray([r['measurement_over_fit'] for r in selected])
            scores.append({'model': model, 'fit': fit, 'supported_bins_with_at_least_5_halos': len(selected),
                           'median_measurement_over_fit': float(np.median(ratios)) if len(ratios) else None,
                           'median_abs_fractional_difference': float(np.median(abs(ratios-1))) if len(ratios) else None})
            if selected:
                ax.plot([r['mass_msun'] for r in selected], ratios, 'o', color=color, markeredgecolor='#25323b',
                        markeredgewidth=.4, label=fit.replace('_', ' ').title(), markersize=6)
        ax.set(xscale='log', yscale='log', title=model if model == 'EDE' else 'ΛCDM', xlabel='FOF bin centre [Msun]')
        ax.grid(axis='y', alpha=.24)
    axes[0].set_ylabel('Measured abundance / fitting function')
    handles, labels = {}, {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles[label] = handle
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=4, frameon=False)
    fig.suptitle('Do the z = 0 halo abundances follow the fitting functions?\n'
                 'Shown where the fit and input spectrum cover the bin and at least five FOF hosts were measured',
                 fontsize=15)
    stem = output / 'EDE_vs_LCDM_z_0000p000_snap_085_FOF_reference_fit_review'
    for suffix in ('.png', '.pdf', '.svg'):
        fig.savefig(stem.with_suffix(suffix), dpi=175)
    _write_csv(stem.with_suffix('.csv'), records)
    _write_csv(stem.with_name(stem.name + '_summary.csv'), scores)
    atomic_json(stem.with_suffix('.json'), {'sources': sources, 'minimum_count_per_reviewed_bin': 5,
                'scope': 'Descriptive agreement in native bins and within each fit validity range. Different fits can have different support. Single small volume; no sample-variance or convergence test.',
                'summary': scores})
    manifest_path = output / 'comparison_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for key, paths in [('figures', [stem.with_suffix('.png')]),
                       ('tables', [stem.with_suffix('.csv'), stem.with_name(stem.name + '_summary.csv')])]:
        manifest[key] = sorted(set(manifest[key]) | {path.name for path in paths})
    atomic_json(manifest_path, manifest)


if __name__ == '__main__':
    main()
