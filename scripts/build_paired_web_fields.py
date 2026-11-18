#!/usr/bin/env python3
"""Generate matched-scale periodic tidal-web maps and halo environment figures."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.analysis import publish_native_environment
from core.foundation import atomic_json, file_sha256
from core.halo_products import export_native_splashback_figures, export_native_web_figures
from scripts.prepare_paired_analysis import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    parser.add_argument('model', choices=('EDE', 'LCDM'))
    parser.add_argument('snapshot', type=int, choices=(21, 85))
    args = parser.parse_args()
    root, model, snapshot = args.atlas.resolve(), args.model, args.snapshot
    home = root / ('02_EDE' if model == 'EDE' else '03_LCDM')
    redshift = '014p400' if snapshot == 21 else '000p000'
    base = home / 'Spatial_explorers' / f'{model}_z_{redshift}'
    structures = sorted(base.glob('Native_structure_FOF_row_*/native_structure.json'))
    workspace = root / 'Source/Native_workspaces' / f'{model}_structure'
    prepare(root.parent / f'dmo3_N128_L5_fast128_{model.lower()}', workspace)
    template = Path(__file__).resolve().parents[1] / 'configs/environment.example.json'
    request = json.loads(template.read_text())
    request.update(run_directory=str(workspace), snapshot=snapshot, mesh_size=65,
                   smoothing_comoving_kpc=[140., 280.],
                   halo_structure_products=[str(p) for p in structures], memory_gib=4.)
    if structures:
        record = json.loads(structures[0].read_text())
        radius = record['source_context']['r200c_proper_kpc']
        request['splashback_controls']['search_interval_proper_kpc'] = [.55*radius, 1.4*radius]
    out = base / 'Periodic_tidal_web'
    request_path = out / 'environment_request.json'
    product = out / 'native_environment.json'
    if not product.exists():
        atomic_json(request_path, request)
        publish_native_environment(request, product, request_source_hashes={str(request_path): file_sha256(request_path)})
    web = out / 'Web_figures'
    if not web.exists():
        export_native_web_figures(product, web, memory_gib=4.)
    feature = out / 'Splashback_figures'
    if structures and not feature.exists():
        export_native_splashback_figures(product, feature, memory_gib=4.)
    print(product, flush=True)


if __name__ == '__main__':
    main()
