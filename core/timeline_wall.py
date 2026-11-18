"""Create a fixed-scale cosmic timeline from a sealed saved-map bundle."""
from __future__ import annotations
import json
import math
import re
import tempfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_timeline_wall(bundle, destination, *, map_key, max_panels=6, memory_gib=1., run_dir=None):
    from core.foundation import file_sha256, verify_product_provenance
    from core.native import native_run_lock
    bundle=Path(bundle).expanduser().resolve();manifest=bundle/'map_contract.json'
    before=file_sha256(manifest);contract=json.loads(manifest.read_text());source=contract.get('provenance')
    if not isinstance(source,dict):raise ValueError('Map bundle lacks simulation provenance')
    if run_dir is None:
        roots=set()
        for entry in source.get('input_hashes',[]):
            for parent in Path(entry.get('path','')).resolve().parents:
                if parent.name=='data' and (parent.parent/'manifest.json').is_file():roots.add(parent.parent);break
        if len(roots)!=1:raise ValueError('Cannot identify source run; supply run_dir explicitly')
        run_dir=roots.pop()
    run=Path(run_dir).expanduser().resolve()
    with native_run_lock(run,operation='timeline wall'):
        verify_product_provenance(run,source)
        if file_sha256(manifest)!=before:raise ValueError('Source map contract changed')
        return _make_timeline_wall(bundle,destination,map_key=map_key,max_panels=max_panels,memory_gib=memory_gib,
                                   run_dir=run,source=source,source_hash=before)


def _make_timeline_wall(bundle, destination, *, map_key, max_panels, memory_gib, run_dir, source, source_hash):
    from core.foundation import atomic_json, file_sha256, publish_directory, verify_product_provenance
    root=Path(bundle).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh timeline output directory')
    if type(max_panels) is not int or not 2<=max_panels<=12:raise ValueError('Timeline requires 2–12 panels')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite memory allocation required')
    manifest=root/'map_contract.json';manifest_hash=file_sha256(manifest);contract=json.loads(manifest.read_text())
    if contract.get('schema')!='map-publication-v1' or manifest_hash!=source_hash:raise ValueError('An unchanged published structure-map bundle is required')
    inventory={row['path']:row['sha256'] for row in contract['artifacts']}
    def check(name):
        p=root/name
        if Path(name).name!=name or name not in inventory or not p.is_file() or file_sha256(p)!=inventory[name]:raise ValueError('Map bundle input is missing or changed: '+str(name))
        return p
    scale=json.loads(check('scale.json').read_text());settings=scale['maps'].get(map_key)
    if not settings or settings.get('status')!='COMPLETE' or settings.get('normalization')!='global_fixed':
        raise ValueError('Timeline requires a measured map series with global fixed normalization')
    records=[]
    for name in inventory:
        if not re.fullmatch(re.escape(map_key)+r'_\d+\.figure\.json',name):continue
        meta=json.loads(check(name).read_text()).get('scientific_metadata',{})
        if meta.get('overlays') is None or meta.get('overlay_layers')!=[]:continue
        numeric=meta.get('numeric_asset');check(numeric)
        records.append({'figure':name,'numeric_asset':numeric,'snapshot':int(meta['snapshot']),
                        'redshift':float(meta['redshift']),'field':meta['field'],'projection':meta['projection'],
                        'box_size_mpc_h':float(meta['box_size_mpc_h']),'colormap':meta['density_colormap'],
                        'slice':meta['slice'],'map_window_mpc_h':meta.get('map_window_mpc_h'),
                        'measured_quantity_units':meta.get('measured_quantity_units'),
                        'display_status':meta.get('display_status','AVAILABLE')})
    if any(not math.isfinite(row['redshift']) or row['redshift'] < -1e-10 for row in records):raise ValueError('Timeline requires valid saved redshifts')
    if len({row['snapshot'] for row in records})!=len(records):raise ValueError('Duplicate timeline snapshot')
    records.sort(key=lambda row:-row['redshift'])
    if len(records)<2 or any(records[i]['redshift']<=records[i+1]['redshift'] for i in range(len(records)-1)):
        raise ValueError('Timeline requires at least two distinct actual saved epochs')
    first=records[0]
    for row in records[1:]:
        if (any(row[k]!=first[k] for k in ('field','projection','box_size_mpc_h','colormap','map_window_mpc_h','measured_quantity_units'))
            or any(row['slice'].get(k)!=first['slice'].get(k) for k in ('thickness_mpc_h','center_mpc_h','line_of_sight_axis','assignment_kernel'))):
            raise ValueError('Timeline maps must share the same geometry, field and color map')
    source_epoch_count=len(records)
    if len(records)>max_panels:
        records=[records[i] for i in np.linspace(0,len(records)-1,max_panels,dtype=int)]
    cols=min(3,len(records));nrows=math.ceil(len(records)/cols)
    fig,axes=plt.subplots(nrows,cols,figsize=(4.5*cols,4.2*nrows),squeeze=False,facecolor='#08111e')
    fig.subplots_adjust(left=.055,right=.9,bottom=.095,top=.88,wspace=.08,hspace=.25)
    cmap=matplotlib.colormaps[first['colormap']].with_extremes(bad=matplotlib.colormaps[first['colormap']](0))
    img=None
    try:
        for ax,row in zip(axes.flat,records):
            with h5py.File(check(row['numeric_asset']),'r') as h:
                dataset=h['log10_display']
                if len(dataset.shape)!=2 or np.prod(dataset.shape,dtype=np.int64)*32*len(records)>memory_gib*1024**3*.75:raise MemoryError('Timeline maps exceed declared memory')
                data=dataset[:]
            box=row['box_size_mpc_h']
            img=ax.imshow(np.ma.masked_invalid(data),origin='lower',extent=(0,box,0,box),interpolation='nearest',cmap=cmap,vmin=settings['vmin'],vmax=settings['vmax'])
            if row['display_status']=='ALL_SUPPORTED_GAS_ZERO_METALLICITY':
                ax.text(.5,.5,'Measured gas metallicity\nis zero',color='white',ha='center',va='center',transform=ax.transAxes)
            ax.grid(False)
            if row['map_window_mpc_h'] is not None:
                x0,x1,y0,y1=row['map_window_mpc_h'];ax.set_xlim(x0,x1);ax.set_ylim(y0,y1)
            ax.set_title('z = %.4g  ·  snapshot %d' % (row['redshift'],row['snapshot']),color='#f1f6ff',fontsize=12,pad=7)
            ax.set_xlabel(row['projection'][0]+' [comoving Mpc/h]',color='#c5d6e8',fontsize=9)
            ax.set_ylabel(row['projection'][1]+' [comoving Mpc/h]',color='#c5d6e8',fontsize=9)
            ax.tick_params(colors='#c5d6e8',labelsize=8)
            for spine in ax.spines.values():spine.set_color('#7a91a8')
        for ax in list(axes.flat)[len(records):]:ax.set_visible(False)
        field_label={'dm':'Dark matter','gas':'Gas','stars':'Stars','total':'Total matter','gas_temperature':'Gas temperature'}.get(first['field'],first['field'])
        fig.suptitle('%s · %s projection through cosmic time' % (field_label,first['projection'].upper()),color='white',fontsize=19,y=.975)
        thickness=first['slice'].get('thickness_mpc_h')
        slab='full box' if thickness is None else 'slab %.3g Mpc/h at %.3g Mpc/h' % (thickness,first['slice']['center_mpc_h'])
        fig.text(.5,.92,'Saved outputs, earlier to later · shared fixed color scale · '+slab,color='#b8cadc',ha='center',fontsize=10)
        cax=fig.add_axes([.925,.18,.016,.63]);cbar=fig.colorbar(img,cax=cax)
        cbar.set_label('log10 temperature [K]' if first['field']=='gas_temperature' else 'log10 gas metal mass fraction' if first['field']=='gas_metallicity' else 'log10 surface density [h Msun/Mpc²]',color='#c5d6e8',fontsize=9)
        cbar.ax.tick_params(colors='#c5d6e8',labelsize=8)
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.timeline-',dir=out.parent))
        try:
            fig.savefig(stage/'cosmic_timeline.png',dpi=180,facecolor=fig.get_facecolor())
            fig.savefig(stage/'cosmic_timeline.pdf',facecolor=fig.get_facecolor())
            for row in records:check(row['numeric_asset']);check(row['figure'])
            check('scale.json')
            if file_sha256(manifest)!=manifest_hash:raise ValueError('Source map contract changed')
            verify_product_provenance(run_dir,source)
            atomic_json(stage/'timeline_contract.json',{'schema':'cosmic-timeline-wall-v1','science_acceptance':'NOT_ACCEPTED',
                'source_bundle':str(root),'source_run':str(run_dir),'source_contract_sha256':manifest_hash,'algorithm_sha256':file_sha256(__file__),
                'map_key':map_key,'epochs':records,'source_epoch_count':source_epoch_count,
                'selection':'Evenly spaced indices in actual time order when more than the panel limit; no synthesized epochs.',
                'color_limits_log10':[settings['vmin'],settings['vmax']],
                'artifacts':{name:file_sha256(stage/name) for name in ('cosmic_timeline.png','cosmic_timeline.pdf')}})
            publish_directory(stage,out)
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)})
            raise
    finally:
        plt.close(fig)
    return out/'cosmic_timeline.png'
