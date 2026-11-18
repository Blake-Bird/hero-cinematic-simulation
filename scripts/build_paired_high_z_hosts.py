#!/usr/bin/env python3
"""Map and tabulate the actual high-redshift FOF hosts in common coordinates."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json, file_sha256
from core.paired_atlas import epoch_name, _write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    root = parser.parse_args().atlas.resolve()
    snap, z = 21, 14.4
    label = epoch_name('EDE_vs_LCDM', snap, z)
    folder = root / '01_Comparisons' / label
    stem = folder / (label + '_Exploratory_FOF_host_locator')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), layout='constrained')
    fig.suptitle('Find the earliest catalogued hosts · z = 14.4', fontsize=20)
    table, sources = [], {}
    for model, ax in zip(('EDE', 'LCDM'), axes):
        data = root / '06_Tables_and_Data/Measurements' / model
        asset = data / (epoch_name(model, snap, z) + '.npz')
        source = json.loads(asset.with_suffix('.json').read_text())
        if source['numeric_sha256'] != file_sha256(asset):
            raise ValueError(f'Changed measurement {asset}')
        sources[model] = {'path': str(asset.relative_to(root)), 'sha256': source['numeric_sha256']}
        box = source['summary']['box_comoving_mpc']
        with np.load(asset, allow_pickle=False) as arrays:
            mass = arrays['fof_mass_msun']
            counts = arrays['fof_particle_count']
            positions = np.mod(arrays['fof_position_comoving_mpc'], box)
            field = gaussian_filter(arrays['dm_surface_density_msun_mpc2_xy'], 2., mode='wrap')
        visible = np.flatnonzero(counts >= 32)
        rows = visible[np.argsort(-mass[visible], kind='stable')]
        image = ax.imshow(np.log10(np.maximum(field, 10**11.33)), cmap='magma', vmin=11.33, vmax=11.74,
                          origin='lower', extent=(0, box, 0, box), interpolation='nearest')
        for rank, row in enumerate(rows, 1):
            x, y, depth = positions[row]
            table.append({'model': model, 'snapshot': snap, 'redshift': source['summary']['redshift'],
                'FOF_row_zero_based': int(row), 'rank_by_mass': rank, 'particle_count': int(counts[row]),
                'mass_msun': mass[row], 'x_comoving_mpc': x, 'y_comoving_mpc': y, 'z_comoving_mpc': depth,
                'passes_100_particles': bool(counts[row] >= 100), 'passes_300_particles': bool(counts[row] >= 300)})
        for members, color, size in ((rows[counts[rows] < 100], '#8fe5e3', 12), (rows[counts[rows] >= 100], '#ffec9f', 34)):
            ax.scatter(positions[members, 0], positions[members, 1], facecolors='none', edgecolors=color, s=size, linewidth=.85)
        for rank, row in enumerate(rows[:3]):
            x, y = positions[row, :2]
            offset_x = -72 if x > box*.72 else 21
            offset_y = (18 + rank * 11) * (-1 if y > box*.68 else 1)
            ax.annotate(f'FOF row {row}', xy=(x, y), xytext=(offset_x, offset_y), textcoords='offset points',
                        fontsize=9, color='white', bbox={'boxstyle': 'round,pad=.25', 'fc': '#111d2b', 'ec': '#ffe38b'},
                        arrowprops={'arrowstyle': '-', 'color': '#ffe38b', 'lw': .8})
        ax.set(title=f'{model if model == "EDE" else "ΛCDM"} · {len(rows)} hosts with ≥32 particles',
               xlabel='X [comoving Mpc]', ylabel='Y [comoving Mpc]', xlim=(0, box), ylim=(0, box))
    fig.colorbar(image, ax=axes, fraction=.045, label='log₁₀ DM surface density [Msun/Mpc²]')
    fig.supxlabel('Cyan rings: 32–99 particles. Gold rings: ≥100. Labels identify the three largest FOF rows in each saved catalogue.', fontsize=10)
    for ext in ('.png', '.pdf', '.svg'):
        fig.savefig(stem.with_suffix(ext), dpi=180)
    _write_csv(stem.with_suffix('.csv'), table)
    atomic_json(stem.with_suffix('.json'), {'sources': sources, 'selection': 'All native FOF rows with at least 32 particles.',
        'scope': 'Exploratory low-count catalogue and 2D projection. IDs are native epoch-local FOF row indices. All runs contain only dark matter.'})
    manifest_path = folder / 'comparison_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for key, ext in (('figures', '.png'), ('tables', '.csv')):
        manifest[key] = sorted(set(manifest[key]) | {stem.with_suffix(ext).name})
    atomic_json(manifest_path, manifest)


if __name__ == '__main__':
    main()
