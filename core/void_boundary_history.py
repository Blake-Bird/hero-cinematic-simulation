"""Measure periodic density-excursion boundaries and their Eulerian overlap."""
from __future__ import annotations

import csv
import html
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import generate_binary_structure, label


def periodic_regions(field, threshold):
    """Label six-connected below-threshold cells, including periodic face links."""
    values = np.asarray(field)
    if values.ndim != 3 or len(set(values.shape)) != 1 or values.shape[0] < 2:
        raise ValueError('A cubic density mesh is required')
    if not np.isfinite(values).all() or np.any(values < 0) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Finite nonnegative density ratios and a threshold between zero and one required')
    regions, count = label(values < threshold, structure=generate_binary_structure(3, 1))
    parent = np.arange(count + 1)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for axis in range(3):
        left = np.take(regions, 0, axis=axis).ravel()
        right = np.take(regions, -1, axis=axis).ravel()
        valid = (left > 0) & (right > 0) & (left != right)
        for a, b in np.unique(np.column_stack((left[valid], right[valid])), axis=0):
            a, b = find(a), find(b)
            if a != b:
                parent[max(a, b)] = min(a, b)
    lookup = np.array([find(i) for i in range(count + 1)], dtype=regions.dtype)
    return lookup[regions]


def overlap_region(regions, later_mask):
    """Choose the region with greatest nonzero cell Jaccard overlap."""
    mask = np.asarray(later_mask, bool)
    if mask.shape != regions.shape or not mask.any():
        raise ValueError('A nonempty later mask on the same mesh is required')
    sizes = np.bincount(regions.ravel())
    shared = np.bincount(regions[mask], minlength=len(sizes))
    candidates = np.flatnonzero(shared[1:]) + 1
    if not len(candidates):
        return None
    later_count = int(mask.sum())
    scores = shared[candidates] / (sizes[candidates] + later_count - shared[candidates])
    order = np.lexsort((candidates, -shared[candidates], -scores))
    selected = int(candidates[order[0]])
    return selected, {
        'shared_cells': int(shared[selected]), 'later_cells': later_count,
        'jaccard': float(scores[order[0]]),
        'fraction_later': float(shared[selected] / later_count),
        'fraction_current': float(shared[selected] / sizes[selected]),
        'overlapping_component_count': int(len(candidates)),
        'runner_up_jaccard': float(scores[order[1]]) if len(order) > 1 else None,
    }


def make_boundary_history(run_dir, void_bundle, destination, *, memory_gib=1.):
    from core.foundation import atomic_json, file_sha256, publish_directory, verify_product_provenance
    from core.native import native_run_lock
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(run_dir).expanduser().resolve()
    bundle = Path(void_bundle).expanduser().resolve()
    out = Path(destination).expanduser().resolve()
    if out.exists():
        raise FileExistsError('Choose a fresh boundary-history directory')
    with native_run_lock(root, operation='void boundary history'):
        source = bundle / 'void_contract.json'
        digest = file_sha256(source)
        contract = json.loads(source.read_text())
        if contract.get('schema') != 'void-explorer-v1' or Path(contract['run']).resolve() != root:
            raise ValueError('Void explorer bundle must belong to the selected run')
        grid = int(contract['grid']); box = float(contract['box_mpc_h'])
        if not math.isfinite(memory_gib) or memory_gib <= 0 or grid**3 * 100 + 64*1024**2 > memory_gib*1024**3*.7:
            raise MemoryError('Boundary labels exceed declared memory budget')
        if not math.isfinite(box) or box <= 0:
            raise ValueError('Positive finite box size required')
        epochs = contract['epochs']
        if len(epochs) < 2 or epochs[-1].get('status') != 'MATCHED':
            raise ValueError('A latest selected void minimum and two epochs are required')
        for provenance in contract['source_provenance']:
            verify_product_provenance(root, provenance)
        out.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix='.void-boundary-', dir=out.parent))
        try:
            rows = []; later = None; tracked = True
            for index in range(len(epochs)-1, -1, -1):
                epoch = epochs[index]
                row = {key: epoch[key] for key in ('snapshot', 'scale_factor', 'redshift')}
                if not tracked:
                    row.update(status='UNMATCHED', reason='No continuous nonzero boundary overlap to the final selected region')
                    rows.append(row); continue
                name = 'density_%03d.npy' % epoch['snapshot']
                if file_sha256(bundle / name) != contract.get('artifacts', {}).get(name):
                    raise ValueError('Saved density volume changed: ' + name)
                field = np.load(bundle / name, allow_pickle=False)
                if field.shape != (grid, grid, grid):
                    raise ValueError('Density volume geometry differs from source contract')
                regions = periodic_regions(field, float(contract['threshold']))
                if later is None:
                    center = tuple(int(x) for x in epoch['center_grid'])
                    selected = int(regions[center]); metrics = {}
                    if selected == 0:
                        raise ValueError('Final minimum is outside the threshold excursion')
                else:
                    match = overlap_region(regions, later)
                    if match is None:
                        row.update(status='UNMATCHED', reason='No shared below-threshold mesh cells with the next saved epoch')
                        rows.append(row); tracked = False; later = None; continue
                    selected, metrics = match
                later = regions == selected
                count = int(later.sum()); volume = count * (box/grid)**3
                mask_name = 'boundary_%03d.npz' % epoch['snapshot']
                np.savez_compressed(stage / mask_name, mask=later)
                row.update(status='MEASURED', cells=count, volume_mpc_h3=volume,
                           volume_fraction=count/grid**3,
                           equivalent_radius_mpc_h=(3*volume/(4*math.pi))**(1/3),
                           mean_density_ratio=float(field[later].mean()), mask_file=mask_name, **metrics)
                rows.append(row)
            rows.reverse()
            with (stage/'boundary_history.csv').open('w', newline='') as stream:
                keys=['snapshot','scale_factor','redshift','status','cells','volume_mpc_h3','volume_fraction',
                      'equivalent_radius_mpc_h','mean_density_ratio','shared_cells','jaccard',
                      'fraction_later','fraction_current','overlapping_component_count','runner_up_jaccard','reason']
                writer=csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
                writer.writeheader(); writer.writerows(rows)
            fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor='#081321')
            try:
                x=[row['scale_factor'] for row in rows]
                for ax in axes:
                    ax.set_facecolor('#14283e'); ax.tick_params(colors='#d5e5f2'); ax.grid(alpha=.15)
                    ax.xaxis.label.set_color('#d5e5f2'); ax.yaxis.label.set_color('#d5e5f2')
                axes[0].plot(x,[row.get('volume_fraction',np.nan) for row in rows],'-o',color='#7ce6f2')
                axes[0].set_ylabel('Below-threshold box volume fraction')
                axes[0].set_title('Periodic underdensity boundary history',color='white')
                axes[1].plot(x,[row.get('jaccard',np.nan) for row in rows],'-o',color='#ffbf78')
                axes[1].set(xlabel='Scale factor',ylabel='Jaccard overlap with next saved epoch',ylim=(0,1.02))
                fig.tight_layout(); fig.savefig(stage/'boundary_history.png',dpi=180,facecolor=fig.get_facecolor())
            finally:
                plt.close(fig)
            scope='Six-connected periodic cells below the source smoothed-density threshold. Backward links maximize nonzero Eulerian cell Jaccard overlap; no motion correction or proven void identity. A large connected excursion can span much of the box. Equivalent radius describes volume, not a spherical boundary.'
            table=''.join('<tr>'+''.join('<td>'+html.escape(str(row.get(key,'—')))+'</td>' for key in ('snapshot','redshift','status','volume_fraction','jaccard'))+'</tr>' for row in rows)
            (stage/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Void boundary history</title><style>body{background:#081321;color:#e9f5ff;font:16px/1.6 system-ui;max-width:1000px;margin:32px auto;padding:20px}img{width:100%}a{color:#8de9ef}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #36516c}</style><h1>Void boundary history</h1><p>'+html.escape(scope)+'</p><p><a href="boundary_history.csv">CSV measurements</a> · <a href="boundary_contract.json">Source record</a></p><img src="boundary_history.png" alt="Boundary volume and adjacent-epoch overlap"><table><tr><th>Snapshot</th><th>Redshift</th><th>Status</th><th>Volume fraction</th><th>Jaccard</th></tr>'+table+'</table></html>',encoding='utf-8')
            for provenance in contract['source_provenance']:
                verify_product_provenance(root, provenance)
            if file_sha256(source) != digest:
                raise ValueError('Void source contract changed')
            atomic_json(stage/'boundary_contract.json',{
                'schema':'void-boundary-history-v1','science_acceptance':'NOT_ACCEPTED','run':str(root),
                'source_contract':str(source),'source_contract_sha256':digest,
                'algorithm_sha256':file_sha256(__file__),'grid':grid,'threshold':contract['threshold'],
                'scope':scope,'epochs':rows,'artifacts':{p.name:file_sha256(p) for p in stage.iterdir() if p.is_file()}})
            publish_directory(stage,out)
            return out/'index.html'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)})
            raise
