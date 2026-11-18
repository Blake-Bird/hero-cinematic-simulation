"""Synchronized saved-field projections, with explicit missing panels."""
from __future__ import annotations
import json
import math
import re
import tempfile
from pathlib import Path
import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_multifield_movie(bundle, destination, *, map_keys, fps=24, seconds_per_snapshot=1, memory_gib=1., run_dir=None):
    from core.foundation import file_sha256, verify_product_provenance
    from core.native import native_run_lock
    bundle=Path(bundle).expanduser().resolve()
    contract_path=bundle/'map_contract.json'; before=file_sha256(contract_path)
    contract=json.loads(contract_path.read_text())
    provenance=contract.get('provenance')
    if not isinstance(provenance,dict):raise ValueError('Map bundle lacks simulation provenance')
    if run_dir is None:
        roots=set()
        for entry in provenance.get('input_hashes',[]):
            path=Path(entry.get('path','')).resolve()
            for parent in path.parents:
                if parent.name=='data' and (parent.parent/'manifest.json').is_file():
                    roots.add(parent.parent);break
        if len(roots)!=1:raise ValueError('Cannot identify one source run; supply run_dir explicitly')
        run_dir=roots.pop()
    root=Path(run_dir).expanduser().resolve()
    with native_run_lock(root,operation='multi-field movie'):
        verify_product_provenance(root,provenance)
        if file_sha256(contract_path)!=before:raise ValueError('Source map contract changed')
        return _make_multifield_movie(bundle,destination,map_keys=map_keys,fps=fps,
            seconds_per_snapshot=seconds_per_snapshot,memory_gib=memory_gib,
            run_dir=root,source_provenance=provenance,source_contract_hash=before)


def _make_multifield_movie(bundle, destination, *, map_keys, fps, seconds_per_snapshot, memory_gib, run_dir, source_provenance, source_contract_hash):
    from core.foundation import atomic_json, file_sha256, publish_directory, verify_product_provenance
    root=Path(bundle).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh movie output directory')
    if not isinstance(map_keys,(list,tuple)) or not 2<=len(map_keys)<=4 or len(set(map_keys))!=len(map_keys):raise ValueError('Select two to four distinct map keys')
    if type(fps) is not int or not 1<=fps<=120 or type(seconds_per_snapshot) is not int or not 1<=seconds_per_snapshot<=60:raise ValueError('Positive bounded fps and display duration required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite memory allocation required')
    contract_path=root/'map_contract.json';contract_hash=file_sha256(contract_path);contract=json.loads(contract_path.read_text())
    if contract.get('schema')!='map-publication-v1' or contract_hash!=source_contract_hash:raise ValueError('An unchanged published structure-map bundle is required')
    inventory={r['path']:r['sha256'] for r in contract['artifacts']}
    def checked(name):
        if not isinstance(name,str) or Path(name).name!=name or name not in inventory:raise ValueError('Unlisted map asset: '+str(name))
        p=root/name
        if not p.is_file() or file_sha256(p)!=inventory[name]:raise ValueError('Map asset missing or changed: '+name)
        return p
    scales=json.loads(checked('scale.json').read_text())['maps'];series={};epochs={};geometry=None
    for key in map_keys:
        setting=scales.get(key)
        if not setting or setting.get('status')!='COMPLETE' or setting.get('normalization')!='global_fixed':raise ValueError('Each selected field requires available global fixed-scale maps: '+str(key))
        rows={}
        for name in inventory:
            if not re.fullmatch(re.escape(key)+r'_\d+\.figure\.json',name):continue
            meta=json.loads(checked(name).read_text()).get('scientific_metadata',{})
            if meta.get('overlay_layers')!=[] or meta.get('overlays') is None:continue
            asset=meta['numeric_asset'];checked(asset);snap=int(meta['snapshot']);z=float(meta['redshift'])
            if not math.isfinite(z) or z<0:raise ValueError('Saved epoch must have finite nonnegative redshift')
            geom=(meta['projection'],float(meta['box_size_mpc_h']),meta.get('map_window_mpc_h'),
                  *(meta['slice'].get(k) for k in ('thickness_mpc_h','center_mpc_h','line_of_sight_axis','assignment_kernel')))
            if geometry is None:geometry=geom
            if geom!=geometry:raise ValueError('All fields must share projection, box, slice and window')
            if snap in rows:raise ValueError('Duplicate field at saved snapshot')
            if snap in epochs and not math.isclose(epochs[snap],z,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('Field epoch metadata disagree')
            epochs[snap]=z
            rows[snap]={'figure':name,'numeric_asset':asset,'field':meta['field'],'colormap':meta['density_colormap'],'display_status':meta.get('display_status','AVAILABLE')}
        if not rows:raise ValueError('Selected series has no map data: '+key)
        series[key]=rows
    order=sorted(epochs,key=lambda s:-epochs[s])
    if len(order)<2 or any(epochs[order[i]]<=epochs[order[i+1]] for i in range(len(order)-1)):raise ValueError('At least two distinct saved epochs required')
    out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.multifield-',dir=out.parent));movie=stage/'multifield_evolution.mp4'
    try:
        with imageio.get_writer(movie,fps=fps,codec='libx264',quality=8,macro_block_size=16) as writer:
            for snap in order:
                fig,axes=plt.subplots(1,len(map_keys),figsize=(4.8*len(map_keys),5.44),dpi=100,squeeze=False,facecolor='#08111e')
                fig.subplots_adjust(left=.04,right=.97,top=.83,bottom=.16,wspace=.3)
                for ax,key in zip(axes.flat,map_keys):
                    ax.set_facecolor('#111f30');row=series[key].get(snap)
                    ax.set_title(key.replace('_',' '),color='white',fontsize=12);ax.grid(False)
                    if row is None:
                        ax.text(.5,.5,'No saved map\nat this epoch',color='#c6d4e5',ha='center',va='center',transform=ax.transAxes,fontsize=12)
                        ax.set_xticks([]);ax.set_yticks([]);continue
                    with h5py.File(checked(row['numeric_asset']),'r') as h:
                        dataset=h['log10_display']
                        if len(dataset.shape)!=2 or np.prod(dataset.shape,dtype=np.int64)*32*len(map_keys)>memory_gib*1024**3*.7:raise MemoryError('Multi-field maps exceed movie allocation')
                        data=dataset[:]
                    if row['display_status']=='ALL_SUPPORTED_GAS_ZERO_METALLICITY':
                        ax.text(.5,.5,'Measured gas metallicity\nis zero',color='#c6d4e5',ha='center',va='center',transform=ax.transAxes,fontsize=12)
                        ax.set_xticks([]);ax.set_yticks([]);continue
                    box=geometry[1];cmap=matplotlib.colormaps[row['colormap']].with_extremes(bad=matplotlib.colormaps[row['colormap']](0))
                    im=ax.imshow(np.ma.masked_invalid(data),origin='lower',extent=(0,box,0,box),interpolation='nearest',cmap=cmap,vmin=scales[key]['vmin'],vmax=scales[key]['vmax'])
                    if geometry[2] is not None:
                        x0,x1,y0,y1=geometry[2];ax.set_xlim(x0,x1);ax.set_ylim(y0,y1)
                    ax.set_xlabel(geometry[0][0]+' [comoving Mpc/h]',color='#c5d6e8',fontsize=9);ax.set_ylabel(geometry[0][1]+' [comoving Mpc/h]',color='#c5d6e8',fontsize=9)
                    ax.tick_params(colors='#c5d6e8',labelsize=8)
                    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.025);cb.ax.tick_params(colors='#c5d6e8',labelsize=8)
                    label = ('log10 temperature [K]' if row['field']=='gas_temperature' else
                             'log10 gas metal mass fraction' if row['field']=='gas_metallicity' else
                             'log10 surface density [h Msun/Mpc²]')
                    cb.set_label(label,color='#c5d6e8',fontsize=8)
                slab='full box' if geometry[3] is None else 'slab %.3g Mpc/h' % geometry[3]
                fig.suptitle('Saved multi-field projections · z=%.5g · snapshot %d · %s' % (epochs[snap],snap,slab),color='white',fontsize=16,y=.95)
                fig.canvas.draw();frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
                for _ in range(fps*seconds_per_snapshot):writer.append_data(frame)
                plt.close(fig)
        checked('scale.json')
        for rows in series.values():
            for row in rows.values():checked(row['figure']);checked(row['numeric_asset'])
        if file_sha256(contract_path)!=contract_hash:raise ValueError('Source map contract changed')
        verify_product_provenance(run_dir,source_provenance)
        atomic_json(stage/'movie_contract.json',{'schema':'multi-field-movie-v1','science_acceptance':'NOT_ACCEPTED',
            'source_bundle':str(root),'source_run':str(run_dir),'source_contract_sha256':contract_hash,'algorithm_sha256':file_sha256(__file__),
            'map_keys':list(map_keys),'epochs':[{'snapshot':s,'redshift':epochs[s],
                'fields':{k:series[k].get(s) for k in map_keys}} for s in order],
            'fps':fps,'seconds_per_snapshot':seconds_per_snapshot,'movie_sha256':file_sha256(movie),
            'timing_scope':'Equal display time per actual saved epoch; no interpolation or uniform physical-time claim.',
            'missing_scope':'An unavailable panel means no map was published at that epoch; it is not an assertion of zero physical signal.'})
        publish_directory(stage,out)
        return out/'multifield_evolution.mp4'
    except BaseException as error:
        atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)})
        raise
