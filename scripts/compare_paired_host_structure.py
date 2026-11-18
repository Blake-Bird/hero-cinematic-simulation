#!/usr/bin/env python3
"""Compare the measured z=0 matched-host shell profiles."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json, file_sha256
from core.paired_atlas import epoch_name, _write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    root = parser.parse_args().atlas.resolve()
    label = epoch_name('EDE_vs_LCDM', 85, 0)
    out = root / '01_Comparisons' / label
    stem = out / f'{label}_Largest_matched_host_profiles'
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    fig.suptitle('Inside the most massive matched halo pair · z = 0', fontsize=19)
    quantities = [('density_msun_kpc3', 'DM density [Msun/kpc³]'),
                  ('circular_velocity_spherical_km_s', 'Spherical circular velocity [km/s]'),
                  ('velocity_dispersion_1d_km_s', '1D peculiar dispersion [km/s]'),
                  ('anisotropy_beta', 'Velocity anisotropy β'),
                  ('bullock_spin_enclosed', 'Enclosed Bullock spin λ′'),
                  ('shell_count', 'Particles in shell')]
    tables, sources = [], {}
    for model, parent, color in [('EDE', '02_EDE', '#b65e1b'), ('LCDM', '03_LCDM', '#087d83')]:
        report, = (root / parent / 'Spatial_explorers').glob(f'{model}_z_000p000/Native_structure_FOF_row_*/profile_contract.json')
        info = json.loads(report.read_text())
        path, = report.parent.glob('*density_and_kinematics.csv')
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        tables.extend({'model': model, 'FOF_row': info['FOF_row'], **row} for row in rows)
        sources[model] = {'table': str(path.relative_to(root)), 'sha256': file_sha256(path),
                          'host_contract': str(report.relative_to(root)), 'host_contract_sha256': file_sha256(report)}
        count = np.asarray([float(row['shell_count']) for row in rows])
        for ax, (key, ylabel) in zip(axes.flat, quantities):
            radius = 'radial_outer_edge_proper_kpc' if key in ('circular_velocity_spherical_km_s', 'bullock_spin_enclosed') else 'radius_proper_kpc'
            x = [float(row[radius]) for row in rows]
            y = np.asarray([float(row[key]) for row in rows])
            if key != 'shell_count':
                y = np.where(count >= 100, y, np.nan)
            ax.plot(x, y, 'o-', color=color, label=model if model == 'EDE' else 'ΛCDM', markersize=3)
            ax.axvline(info['R200c_proper_kpc'], linewidth=.8, linestyle=':', color=color)
            ax.set(xscale='log', yscale='log' if key in ('density_msun_kpc3','shell_count') else 'linear',
                   xlabel='Proper radius [kpc]', ylabel=ylabel)
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.supxlabel('Dotted lines mark each R200c. Profile curves show shells with at least 100 particles. Selection is all DM within each proper aperture.', fontsize=10)
    for ext in ('.png', '.pdf', '.svg'):
        fig.savefig(stem.with_suffix(ext), dpi=175)
    _write_csv(stem.with_suffix('.csv'), tables)
    atomic_json(stem.with_suffix('.json'), {'sources': sources, 'scope': 'Two independently sampled native radial grids shown in proper kpc. The pair was matched by exact initial particle identities.'})
    manifest_path = out / 'comparison_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for key, suffix in (('figures','.png'), ('tables','.csv')):
        manifest[key] = sorted(set(manifest[key]) | {stem.with_suffix(suffix).name})
    atomic_json(manifest_path, manifest)


if __name__ == '__main__':
    main()
