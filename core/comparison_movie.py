"""Encode matched, measured comparison maps into a provenance-bound movie."""
from __future__ import annotations
import json
import math
import tempfile
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_comparison_movie(bundle, destination, *, map_key, fps=24, seconds_per_snapshot=1, memory_gib=1., runs=None):
    from contextlib import ExitStack
    from core.foundation import file_sha256, verify_product_provenance
    from core.native import native_run_lock
    bundle=Path(bundle).expanduser().resolve();path=bundle/'comparison_contract.json'
    before=file_sha256(path);contract=json.loads(path.read_text());provenance=contract.get('provenance_by_run')
    if not isinstance(provenance,list) or len(provenance)!=2 or not all(isinstance(x,dict) for x in provenance):raise ValueError('Two source-run provenance records are required')
    if runs is None:
        runs=[]
        for source in provenance:
            candidates=set()
            for entry in source.get('input_hashes',[]):
                p=Path(entry.get('path','')).resolve()
                for parent in p.parents:
                    if parent.name=='data' and (parent.parent/'manifest.json').is_file():
                        candidates.add(parent.parent);break
            if len(candidates)!=1:raise ValueError('Cannot identify source run; supply both run paths')
            runs.append(candidates.pop())
    if not isinstance(runs,(list,tuple)) or len(runs)!=2:raise ValueError('Supply exactly two source run paths')
    roots=[Path(p).expanduser().resolve() for p in runs]
    with ExitStack() as locks:
        for root in sorted(set(roots)):locks.enter_context(native_run_lock(root,operation='paired comparison movie'))
        for root,source in zip(roots,provenance):verify_product_provenance(root,source)
        if file_sha256(path)!=before:raise ValueError('Source comparison contract changed')
        return _make_comparison_movie(bundle,destination,map_key=map_key,fps=fps,seconds_per_snapshot=seconds_per_snapshot,
            memory_gib=memory_gib,roots=roots,provenance=provenance,source_contract_hash=before)


def _make_comparison_movie(bundle, destination, *, map_key, fps, seconds_per_snapshot, memory_gib, roots, provenance, source_contract_hash):
    """Render saved matched maps in time order; no unsaved states are invented."""
    from core.foundation import atomic_json, file_sha256, publish_directory, verify_product_provenance
    root = Path(bundle).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError('Choose a fresh movie destination')
    if type(fps) is not int or not 1 <= fps <= 120 or type(seconds_per_snapshot) is not int or not 1 <= seconds_per_snapshot <= 60:
        raise ValueError('fps and seconds per snapshot must be positive bounded integers')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError('Positive finite movie memory allocation required')
    contract_path = root/'comparison_contract.json'
    contract_hash = file_sha256(contract_path)
    contract = json.loads(contract_path.read_text())
    if contract.get('schema') != 'comparison-map-publication-v1' or contract_hash!=source_contract_hash:
        raise ValueError('A published comparison-map bundle is required')
    inventory = {item['path']:item['sha256'] for item in contract['artifacts']}
    def checked(name):
        path = root/name
        if Path(name).name != name or name not in inventory or not path.is_file() or file_sha256(path) != inventory[name]:
            raise ValueError('Comparison input is missing, unlisted or changed: '+str(name))
        return path
    summary = json.loads(checked('comparison_maps_summary.json').read_text())
    mapping = summary['maps'].get(map_key)
    if not mapping or mapping.get('status') != 'COMPLETE' or not mapping.get('fixed_color_scale_across_matched_redshifts'):
        raise ValueError('Requested map series is absent or lacks fixed color scales')
    records = []
    for image in mapping['figure_files']:
        figure_name = str(Path(image).with_suffix('.figure.json'))
        figure = json.loads(checked(figure_name).read_text())
        meta = figure['scientific_metadata']
        asset = meta['numeric_asset']
        checked(asset)
        records.append({'numeric_asset':asset,'figure':figure_name,
                        'redshift_a':float(meta['redshift_a']),'redshift_b':float(meta['redshift_b']),
                        'snapshot_a':int(meta['snapshot_a']),'snapshot_b':int(meta['snapshot_b']),
                        'field':str(meta['field']), 'projection':str(meta['projection'])})
    if any(not math.isfinite(r[k]) or r[k]<-1e-10 for r in records for k in ('redshift_a','redshift_b')):raise ValueError('Invalid saved redshift')
    if len({(r['field'],r['projection']) for r in records})!=1:raise ValueError('Comparison field or projection changes across epochs')
    records.sort(key=lambda row: -row['redshift_a'])
    if len(records) < 2 or any(records[i]['redshift_a'] <= records[i+1]['redshift_a'] for i in range(len(records)-1)):
        raise ValueError('Movie requires at least two distinct ordered saved epochs')
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.paired-movie-',dir=destination.parent))
    output = stage/'paired_comparison.mp4'
    try:
        with imageio.get_writer(output,fps=fps,codec='libx264',quality=8,macro_block_size=16) as writer:
            for row in records:
                with h5py.File(checked(row['numeric_asset']),'r') as h:
                    shape = h['measured_a'].shape
                    if len(shape)!=2 or h['measured_b'].shape!=shape or h['log10_ratio_b_over_a'].shape!=shape:
                        raise ValueError('Comparison grids must be aligned two-dimensional arrays')
                    if np.prod(shape,dtype=np.int64)*64 > memory_gib*1024**3*.75:
                        raise MemoryError('Comparison frame exceeds movie memory allocation')
                    a,b,ratio = (np.asarray(h[k],dtype=float) for k in ('measured_a','measured_b','log10_ratio_b_over_a'))
                with np.errstate(divide='ignore',invalid='ignore'):
                    da = np.ma.masked_where(~np.isfinite(a)|(a<=0),np.log10(a))
                    db = np.ma.masked_where(~np.isfinite(b)|(b<=0),np.log10(b))
                dr = np.ma.masked_invalid(ratio)
                fig, axes = plt.subplots(1,3,figsize=(12.8,4.8),dpi=100)
                fig.subplots_adjust(left=.055,right=.93,bottom=.15,top=.84,wspace=.26)
                extent=(0,summary['box_mpc_h'],0,summary['box_mpc_h'])
                for ax,data,title in zip(axes,(da,db,dr),('Run A','Run B','log10(B/A)')):
                    lim = mapping['log10_ratio_limit']
                    image=ax.imshow(data,origin='lower',extent=extent,interpolation='nearest',
                        cmap='coolwarm' if ax is axes[2] else 'magma',
                        vmin=-lim if ax is axes[2] else mapping['vmin'],
                        vmax=lim if ax is axes[2] else mapping['vmax'])
                    ax.set_title(title,fontsize=11);ax.set_xlabel('comoving Mpc/h',fontsize=9);ax.set_ylabel('comoving Mpc/h',fontsize=9)
                    ax.tick_params(labelsize=8)
                    colorbar=fig.colorbar(image,ax=ax,fraction=.045,pad=.025)
                    colorbar.ax.tick_params(labelsize=8)
                    colorbar.set_label('log10(B/A)' if ax is axes[2] else
                        ('log10 temperature [K]' if row['field']=='gas_temperature' else 'log10 surface density [h Msun/Mpc²]'),fontsize=8)
                fig.suptitle('%s · %s projection · z A=%.5g / B=%.5g · snapshots %d / %d' %
                             (row['field'].replace('_',' ').title(),row['projection'].upper(),row['redshift_a'],row['redshift_b'],row['snapshot_a'],row['snapshot_b']),fontsize=12)
                fig.canvas.draw()
                frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
                for _ in range(fps*seconds_per_snapshot):writer.append_data(frame)
                plt.close(fig)
        for row in records:
            checked(row['numeric_asset']);checked(row['figure'])
        checked('comparison_maps_summary.json')
        if file_sha256(contract_path)!=contract_hash:raise ValueError('Source comparison contract changed')
        for run,source in zip(roots,provenance):verify_product_provenance(run,source)
        atomic_json(stage/'movie_contract.json',{'schema':'paired-comparison-movie-v1','science_acceptance':'NOT_ACCEPTED',
            'source_bundle':str(root),'source_runs':[str(p) for p in roots],'source_contract_sha256':contract_hash,'algorithm_sha256':file_sha256(__file__),'map_key':map_key,'frames':records,
            'fps':fps,'seconds_per_snapshot':seconds_per_snapshot,'movie_sha256':file_sha256(output),
            'timing_scope':'Equal display time per saved epoch; no interpolation or uniform physical-time claim.',
            'phase_scope':'Matching epochs and seeds do not by themselves prove matched initial phases.'})
        publish_directory(stage,destination)
        return destination/'paired_comparison.mp4'
    except BaseException as error:
        atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)})
        raise
