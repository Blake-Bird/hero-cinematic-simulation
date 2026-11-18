#!/usr/bin/env python3
"""Generate finder comparisons, shape figures and portable XDMF for both runs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.analysis import group_files, load_cfg, plot_halo_finder_comparison
from core.foundation import atomic_json, file_sha256
from core.halos import plot_radial_shapes
from core.visualization import make_xdmf
from scripts.prepare_paired_analysis import prepare


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    parser.add_argument('model', choices=('EDE', 'LCDM'))
    parser.add_argument('product', choices=('finder', 'shapes', 'xdmf'))
    args = parser.parse_args()
    root, model = args.atlas.resolve(), args.model
    native = root.parent / f'dmo3_N128_L5_fast128_{model.lower()}'
    workspaces = root / 'Source/Native_workspaces'
    if args.product == 'finder':
        work = prepare(native, workspaces / f'{model}_finder_M200c')
        config = work / 'config.yaml'
        if config.read_bytes() != (native / 'config.yaml').read_bytes():
            shutil.copy2(native / 'config.yaml', config)
        cfg = load_cfg(work)
        cfg['analysis']['hmf_mass_definition'] = 'm200c'
        (work / 'results').mkdir(exist_ok=True)
        result = {}
        for snap, z in ((21, '014p400'), (26, '012p000'), (30, '010p000'), (35, '008p000'),
                        (41, '006p000'), (54, '003p000'), (69, '001p000'), (85, '000p000')):
            record = plot_halo_finder_comparison(work, cfg, snap)
            model_dir = root / ('02_EDE' if model == 'EDE' else '03_LCDM')
            target = model_dir / 'Finder_comparison_M200c' / f'{model}_z_{z}_snap_{snap:03d}'
            target.mkdir(parents=True, exist_ok=True)
            artifacts = {}
            for item in (work / 'results').glob(f'*_{snap:03d}.*'):
                if item.is_file():
                    shutil.copy2(item, target / item.name)
                    artifacts[item.name] = file_sha256(item)
            arrays = {k: v for k, v in record.items() if isinstance(v, np.ndarray)}
            if arrays:
                np.savez_compressed(target / 'Finder_comparison_arrays.npz', **arrays)
            summary = json_value({k: v for k, v in record.items() if k not in arrays})
            atomic_json(target / 'Finder_comparison_source.json', {
                'selection': 'M200c for both catalogues; in-memory analysis override only',
                'source_configuration_sha256': file_sha256(config),
                'native_fof_shards': {str(p.resolve()): file_sha256(p) for p in group_files(work, snap)},
                'report': summary, 'figure_artifacts': artifacts})
            result[snap] = summary['status']
    elif args.product == 'shapes':
        source, = (native / 'results').glob('radial_shapes_[0-9]*/report.sha256')
        report = json.loads((source.parent / 'report.json').read_text())
        folder = root / ('02_EDE' if model == 'EDE' else '03_LCDM') / 'Spatial_explorers'
        output = folder / f'{model}_z_000p000' / 'Radial_density_shape'
        result = plot_radial_shapes(source.parent, output)
    else:
        work = prepare(native, workspaces / f'{model}_xdmf')
        result = make_xdmf(work)
        source = result[0].parent
        folder = root / '05_Blender_and_3D' / model / f'{model}_z_199_to_000_ParaView_XDMF'
        if folder.exists():
            raise FileExistsError(folder)
        shutil.move(source, folder)
        result = folder
    atomic_json(root / '07_Status_and_Provenance' / f'{model}_{args.product}_job.json',
                {'status': 'PUBLISHED', 'path': str(result)})
    print(result, flush=True)


if __name__ == '__main__':
    main()
