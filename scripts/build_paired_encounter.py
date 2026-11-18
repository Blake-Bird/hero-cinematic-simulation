#!/usr/bin/env python3
"""Build the repository's linked-halo encounter film with native map receipts."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.analysis import analyze_selected_products, load_cfg
from core.foundation import atomic_json, file_sha256
from core.halo_collision_movie import make_halo_collision_movie, select_merger_pair
from core.paired_atlas import _write_csv
from core.visualization import make_maps
from scripts.prepare_paired_analysis import prepare


def redshift_name(z):
    return f'{z:07.3f}'.replace('.', 'p')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    parser.add_argument('model', choices=('EDE', 'LCDM'))
    args = parser.parse_args()
    root, model = args.atlas.resolve(), args.model
    native = root.parent / f'dmo3_N128_L5_fast128_{model.lower()}'
    workspace = prepare(native, root/'Source/Native_workspaces'/f'{model}_encounter')
    bundles = sorted(workspace.glob('results/analysis_products_*/merger_tree.json'))
    if not bundles:
        analyze_selected_products(workspace, ['histories'], include_pdf_report=False, snapshots=list(range(64, 86)))
        bundles = sorted(workspace.glob('results/analysis_products_*/merger_tree.json'))
    tree = bundles[-1]
    pair = select_merger_pair(json.loads(tree.read_text()))
    start, end = (redshift_name(pair['rows'][i]['redshift']) for i in (0, -1))
    out = root/'04_Movies'/model/f'{model}_z_{start}_to_{end}_linked_halo_encounter'
    maps = []
    box = float(load_cfg(workspace)['simulation']['box_mpc_h'])
    for contract in workspace.glob('results/structure_maps/maps_*/map_contract.json'):
        scale = json.loads((contract.parent/'scale.json').read_text())
        if scale['slice']['thickness_mpc_h'] >= box:
            maps.append(contract)
    if not maps:
        make_maps(workspace, ['dm'], ['xy'], scale_mode='global', story_figure=False,
                  slice_thickness_mpc_h=box, slice_center_mpc_h=box/2)
        maps = sorted(workspace.glob('results/structure_maps/maps_*/map_contract.json'))
        maps = [p for p in maps if json.loads((p.parent/'scale.json').read_text())['slice']['thickness_mpc_h'] >= box]
    selected = sorted(maps)[-1].parent
    scale = json.loads((selected/'scale.json').read_text())
    map_key, = scale['maps']
    movie = make_halo_collision_movie(tree.parent, selected, out, map_key=map_key, memory_gib=3.)
    # Keep the original renderer's receipt and its named source map alongside a table.
    rows = [{k:v for k,v in row.items() if not isinstance(v,(tuple,list,dict))} for row in pair['rows']]
    _write_csv(out/'Saved_pair_kinematics.csv',rows)
    atomic_json(out/'film_source.json', {'title':'Linked halo encounter','model':model,
        'redshifts':f'z = {pair["rows"][0]["redshift"]:.3f} to {pair["rows"][-1]["redshift"]:.3f}',
        'published_sha256':file_sha256(movie),'render_contract':'movie_contract.json'})
    print(movie,flush=True)


if __name__ == '__main__':
    main()
