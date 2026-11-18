#!/usr/bin/env python3
"""Compare the periodic T-web at two physical smoothing scales."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.foundation import atomic_json, file_sha256
from core.paired_atlas import _write_csv, epoch_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    args = parser.parse_args()
    root = args.atlas.resolve()
    classes = ('Void', 'Sheet', 'Filament', 'Knot')
    colors = ['#14263c', '#4b768d', '#8cd7be', '#edc27c']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-.5, 4), cmap.N)
    for snap, z, token in ((21, 14.4, '014p400'), (85, 0., '000p000')):
        rows, records = [], {}
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
        fig.suptitle(f'The dark-matter tidal web · z = {z:g}', fontsize=20)
        for model, folder, offset in [('EDE', '02_EDE', -.18), ('LCDM','03_LCDM', .18)]:
            product = root/folder/'Spatial_explorers'/f'{model}_z_{token}'/'Periodic_tidal_web/native_environment.json'
            record = json.loads(product.read_text())
            data = product.with_name(product.name+'.data')/'arrays.npz'
            records[model] = {'environment': str(product.relative_to(root)), 'sha256': file_sha256(product),
                              'arrays_sha256': file_sha256(data)}
            box = record['box_comoving_kpc']/1000
            with np.load(data, allow_pickle=False) as arrays:
                for i, smoothing in enumerate(record['request']['smoothing_comoving_kpc']):
                    web = arrays[f'smoothing_{i:02d}_web_web_class']
                    if web.ndim != 3 or np.any((web<0)|(web>3)):
                        raise ValueError('Unexpected native web classifier payload')
                    cell = web.shape[2]//2
                    ax = axes[i, 0 if model == 'EDE' else 1]
                    image = ax.imshow(web[:, :, cell].T, origin='lower', extent=(0, box, 0, box), cmap=cmap, norm=norm)
                    ax.set(xlabel='x [comoving Mpc]', ylabel='y [comoving Mpc]',
                           title=f'{model} · {smoothing:g} comoving kpc smoothing')
                    counts = np.bincount(web.ravel(), minlength=4)
                    fraction = counts/web.size
                    bars = axes[i,2]
                    bars.bar(np.arange(4)+offset, fraction, width=.34, alpha=.9,
                             color='#ae652d' if model=='EDE' else '#0e7476', label=model)
                    bars.set(xticks=np.arange(4), xticklabels=classes, ylabel='Box volume fraction', ylim=(0,1))
                    bars.grid(axis='y',alpha=.18)
                    rows.extend({'model':model,'snapshot':snap,'redshift':z,'smoothing_comoving_kpc':smoothing,
                                 'web_class':name,'mesh_cells':int(n),'box_volume_fraction':f,
                                 'slice_z_comoving_mpc':cell*box/web.shape[2]}
                                 for name,n,f in zip(classes,counts,fraction))
        axes[0,2].legend(frameon=False)
        cb=fig.colorbar(image, ax=axes[:,:2].ravel().tolist(), orientation='horizontal', shrink=.72, ticks=np.arange(4))
        cb.ax.set_xticklabels(classes)
        fig.supxlabel('Central XY slice from periodic tidal classification. Volume fractions use every mesh node. Both models use the same physical smoothing.',fontsize=10)
        label = epoch_name('EDE_vs_LCDM', snap, z)
        directory = root/'01_Comparisons'/label
        stem = directory/f'{label}_Tidal_web_classification_and_volume'
        for suffix in ('.png','.pdf','.svg'):
            fig.savefig(stem.with_suffix(suffix),dpi=175)
        plt.close(fig)
        _write_csv(stem.with_suffix('.csv'), rows)
        atomic_json(stem.with_suffix('.json'), {'sources':records,'classes':classes,
                    'scope':'Same comoving mesh box, 65^3 nodes, two common smoothing scales, fixed dimensionless tidal threshold .1.'})
        manifest_path = directory/'comparison_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        for key, suffix in (('figures','.png'),('tables','.csv')):
            manifest[key]=sorted(set(manifest[key])|{stem.with_suffix(suffix).name})
        atomic_json(manifest_path,manifest)


if __name__ == '__main__':
    main()
