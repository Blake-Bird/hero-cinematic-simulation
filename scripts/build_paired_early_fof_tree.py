#!/usr/bin/env python3
"""Build a low-count, exact-membership FOF ancestry anchored at z=14.4."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.analysis import _read_cross_run_halo_catalogue
from core.foundation import atomic_json, file_sha256
from core.halo_story import export_linked_halo_story
from core.merger_tree_visual import build_merger_tree


def descendant_ids(earlier, later):
    """One best child per parent, allowing several parents per child."""
    children = np.full(len(earlier['group_indices']), -1, dtype=np.int64)
    _, i, j = np.intersect1d(earlier['dm_ids'], later['dm_ids'], return_indices=True)
    if not len(i):
        return children
    pairs = np.column_stack((earlier['dm_group_indices'][i], later['dm_group_indices'][j]))
    edges, shared = np.unique(pairs, axis=0, return_counts=True)
    parent_rows = np.searchsorted(earlier['group_indices'], edges[:, 0])
    child_rows = np.searchsorted(later['group_indices'], edges[:, 1])
    supported = (shared >= 16) & (shared >= .5 * earlier['n_dm'][parent_rows])
    # Largest shared membership, then stable child row for exact ties.
    order = np.lexsort((-edges[:, 1], shared))
    for edge in order:
        if supported[edge]:
            children[parent_rows[edge]] = edges[edge, 1]
    return children


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    parser.add_argument('model', choices=('EDE', 'LCDM'))
    args = parser.parse_args()
    root, model = args.atlas.resolve(), args.model
    workspace = root / 'Source/Native_workspaces' / model
    source = root / '06_Tables_and_Data/Measurements' / model
    out = root / 'Source/Halo_story_maps' / model / 'fof_anchor_021'
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((source / 'measurement_manifest.json').read_text())['epochs'][:22]
    catalogues = [_read_cross_run_halo_catalogue(workspace, row['summary']['snapshot'], 'fof', 32, 2_100_000)
                  for row in rows]
    tree_inputs = []
    for i, (row, cat) in enumerate(zip(rows, catalogues)):
        epoch = row['summary']
        next_cat = catalogues[i+1] if i+1 < len(rows) else None
        tree_inputs.append({'ids': cat['group_indices'], 'mass': cat['mass_msun_h'],
            'valid': np.ones(len(cat['group_indices']), bool),
            'desc_ids': descendant_ids(cat, next_cat) if next_cat is not None else np.full(len(cat['group_indices']), -1),
            'desc_scale': np.full(len(cat['group_indices']), rows[i+1]['summary']['scale_factor'] if next_cat is not None else np.nan),
            'snapshot': epoch['snapshot'], 'redshift': epoch['redshift'], 'scale': epoch['scale_factor'],
            'position_mpc_h': np.mod(cat['position_mpc_h'], catalogues[-1]['box_mpc_h']), 'source': 'GADGET FOF exact native DM membership'})
    tree = build_merger_tree(tree_inputs, int(np.argmax(catalogues[-1]['mass_msun_h'])), box_mpc_h=catalogues[-1]['box_mpc_h'])
    if tree is None:
        raise ValueError(f'{model}: largest z=14.4 host has no supported adjacent FOF membership link')
    tree['scope'] = 'Exploratory FOF tree, minimum 32 particles per host. Each parent passes at least 16 exact native DM IDs and half its membership to one adjacent best child. IDs shown are FOF rows at that snapshot. Mass is total FOF mass.'
    tree_path = out / 'merger_tree.json'
    atomic_json(tree_path, tree)
    namespace = json.loads((workspace / 'manifest.json').read_text())['run_namespace']
    atomic_json(out / 'analysis_provenance.json', {'schema': 'paired-early-fof-ancestry-v1',
        'data_contract': {'run_namespace': namespace}, 'algorithm_sha256': file_sha256(__file__),
        'measurement_manifest_sha256': file_sha256(source / 'measurement_manifest.json'),
        'output_artifacts': [{'relative_path': tree_path.name, 'sha256': file_sha256(tree_path)}]})
    frames = []
    for row in rows:
        asset = source / row['numeric_asset']
        if file_sha256(asset) != row['numeric_sha256']:
            raise ValueError(f'Changed measurement {asset}')
        with np.load(asset, allow_pickle=False) as arrays:
            field = gaussian_filter(arrays['dm_surface_density_msun_mpc2_xy'], 2., mode='wrap')
        numeric = asset.name
        np.savez_compressed(out / numeric, log10_projected_density_over_mean=np.log10(np.maximum(field / field.mean(), 1.e-6)))
        epoch = row['summary']
        frames.append({'snapshot': epoch['snapshot'], 'redshift': epoch['redshift'],
                       'numeric_asset': numeric, 'sources': row['source_files']})
    density = out / 'density_source.json'
    atomic_json(density, {'schema': 'focused-dmo-density-movie-v1', 'frames': frames})
    dest = root / '04_Movies' / model / f'{model}_z_014p400_exploratory_FOF_ancestry'
    print(export_linked_halo_story(tree_path, density, out, dest, run_dir=workspace), flush=True)


if __name__ == '__main__':
    main()
