#!/usr/bin/env python3
"""Link each measured merger tree to its native-epoch dark matter maps."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.analysis import analyze_selected_products
from core.foundation import atomic_json, file_sha256
from scripts.prepare_paired_analysis import prepare
from core.halo_story import export_linked_halo_story
from core.paired_atlas import epoch_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    parser.add_argument('model', choices=('EDE', 'LCDM'))
    parser.add_argument('--anchor', type=int, choices=(21, 85), default=85)
    args = parser.parse_args()
    root, model = args.atlas.resolve(), args.model
    workspace = root / 'Source/Native_workspaces' / f'{model}_history'
    if args.anchor == 21:
        workspace = prepare(root.parent / f'dmo3_N128_L5_fast128_{model.lower()}',
                            workspace.with_name(f'{model}_history_z14p4'))
        trees = list(workspace.glob('results/analysis_products_*/merger_tree.json'))
        if not trees:
            analyze_selected_products(workspace, ['histories'], include_pdf_report=False,
                                      snapshots=range(args.anchor + 1))
    trees = list(workspace.glob('results/analysis_products_*/merger_tree.json'))
    if not trees:
        raise ValueError(f'No resolved linked tree for {model}, anchor snapshot {args.anchor}')
    tree, = trees
    data = root / '06_Tables_and_Data/Measurements' / model
    maps = root / 'Source/Halo_story_maps' / model / f'anchor_{args.anchor:03d}'
    maps.mkdir(parents=True, exist_ok=True)
    frames = []
    for record in sorted(data.glob(f'{model}_z_*.json')):
        metadata = json.loads(record.read_text())
        summary = metadata['summary']
        if summary['snapshot'] > args.anchor:
            continue
        source = data / metadata['numeric_asset']
        if file_sha256(source) != metadata['numeric_sha256']:
            raise ValueError(f'Changed measured asset {source}')
        with np.load(source, allow_pickle=False) as arrays:
            field = gaussian_filter(arrays['dm_surface_density_msun_mpc2_xy'], 2., mode='wrap')
        image = np.log10(np.maximum(field / field.mean(), 1.e-6))
        name = epoch_name(model, summary['snapshot'], summary['redshift']) + '_relative_DM_XY.npz'
        np.savez_compressed(maps / name, log10_projected_density_over_mean=image)
        frames.append({'snapshot': summary['snapshot'], 'redshift': summary['redshift'],
                       'numeric_asset': name, 'sources': metadata['source_files'],
                       'measurement_record': str(record), 'measurement_sha256': metadata['numeric_sha256']})
    contract = maps / 'density_source.json'
    atomic_json(contract, {'schema': 'focused-dmo-density-movie-v1', 'frames': frames,
                           'measurement': 'Full type-1 projected mass, periodic two-pixel smoothing, normalized by its own box mean'})
    z = '014p400' if args.anchor == 21 else '000p000'
    dest = root / '04_Movies' / model / f'{model}_z_{z}_descendant_labelled_halo_merger_tree'
    if dest.exists():
        raise FileExistsError(f'Preserve the previous publication before rebuilding {dest}')
    print(export_linked_halo_story(tree, contract, maps, dest, run_dir=workspace), flush=True)


if __name__ == '__main__':
    main()
