"""Compare two saved projected maps without assuming matched phases."""
from __future__ import annotations
import html
import json
import math
import tempfile
from contextlib import ExitStack
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def make_resolution_reveal(run_a,bundle_a,snapshot_a,run_b,bundle_b,snapshot_b,destination,*,map_key,redshift_tolerance=1e-6,memory_gib=1.):
    from core.foundation import atomic_json,file_sha256,verify_product_provenance,publish_directory
    from core.native import native_run_lock
    roots=[Path(run_a).expanduser().resolve(),Path(run_b).expanduser().resolve()];bundles=[Path(bundle_a).expanduser().resolve(),Path(bundle_b).expanduser().resolve()];out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh resolution-comparison directory')
    if any(type(s) is not int or s<0 for s in (snapshot_a,snapshot_b)):raise ValueError('Exact nonnegative saved snapshot labels required')
    if isinstance(redshift_tolerance,bool) or not isinstance(redshift_tolerance,(int,float)) or not math.isfinite(redshift_tolerance) or redshift_tolerance<0:raise ValueError('Finite nonnegative redshift tolerance required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite memory allocation required')
    with ExitStack() as owners:
        for root in sorted(set(roots)):owners.enter_context(native_run_lock(root,operation='resolution comparison'))
        loaded=[]
        for root,bundle,snapshot in zip(roots,bundles,(snapshot_a,snapshot_b)):
            manifest=bundle/'map_contract.json';mh=file_sha256(manifest);contract=json.loads(manifest.read_text())
            if contract.get('schema')!='map-publication-v1':raise ValueError('Two published structure-map bundles are required')
            verify_product_provenance(root,contract['provenance']);inventory={r['path']:r['sha256'] for r in contract['artifacts']}
            def checked(name):
                if not isinstance(name,str) or Path(name).name!=name or name not in inventory:raise ValueError('Unlisted map asset')
                p=bundle/name
                if not p.is_file() or file_sha256(p)!=inventory[name]:raise ValueError('Map asset missing or changed: '+name)
                return p
            scale=json.loads(checked('scale.json').read_text());settings=scale['maps'].get(map_key)
            if not settings or settings.get('status')!='COMPLETE':raise ValueError('Requested map series is unavailable')
            fig_name=map_key+'_%03d.figure.json'%snapshot;meta=json.loads(checked(fig_name).read_text())['scientific_metadata']
            if meta.get('overlay_layers')!=[] or int(meta['snapshot'])!=snapshot:raise ValueError('Unadorned map at requested snapshot required')
            numeric=checked(meta['numeric_asset'])
            with h5py.File(numeric,'r') as h:
                ds=h['log10_display']
                if len(ds.shape)!=2 or ds.shape[0]!=ds.shape[1] or np.prod(ds.shape,dtype=np.int64)*64*2>memory_gib*1024**3*.7:raise MemoryError('Comparison map grids exceed declared memory')
                array=np.asarray(ds[:],dtype=float)
            loaded.append({'root':root,'bundle':bundle,'manifest':manifest,'manifest_hash':mh,'contract':contract,
                'figure_name':fig_name,'figure_sha256':file_sha256(bundle/fig_name),'numeric_name':meta['numeric_asset'],
                'numeric_sha256':file_sha256(numeric),'metadata':meta,'settings':settings,'array':array,'snapshot':snapshot})
        a,b=loaded;ma,mb=a['metadata'],b['metadata']
        if any(ma[k]!=mb[k] for k in ('field','projection')) or not math.isclose(float(ma['box_size_mpc_h']),float(mb['box_size_mpc_h']),rel_tol=1e-10):raise ValueError('Maps must share field, projection and comoving box size')
        if ma.get('measured_quantity_units')!=mb.get('measured_quantity_units'):raise ValueError('Map measurement units differ')
        if ma['field']=='gas_temperature':
            ta=a['settings'].get('temperature_source_by_snapshot',{}).get(str(snapshot_a))
            tb=b['settings'].get('temperature_source_by_snapshot',{}).get(str(snapshot_b))
            if ta is None or ta!=tb:raise ValueError('Temperature representations must be available and match')
        geom=('thickness_mpc_h','center_mpc_h','line_of_sight_axis','assignment_kernel')
        if any(ma['slice'].get(k)!=mb['slice'].get(k) for k in geom):raise ValueError('Maps must share slab geometry and assignment')
        if abs(float(ma['redshift'])-float(mb['redshift']))>redshift_tolerance*max(1.,abs(float(ma['redshift'])),abs(float(mb['redshift']))):raise ValueError('Saved map epochs exceed the declared redshift tolerance')
        finite=[v['array'][np.isfinite(v['array'])] for v in loaded]
        if any(len(x)==0 for x in finite):raise ValueError('Both maps require finite measured pixels')
        low=min(float(np.percentile(x,1)) for x in finite);high=max(float(np.percentile(x,99.9)) for x in finite)
        if high<=low:high=np.nextafter(low,np.inf)
        palette=matplotlib.colormaps['magma'];out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.resolution-reveal-',dir=out.parent))
        try:
            for name,row in zip(('run_a.png','run_b.png'),loaded):
                data=row['array'];valid=np.isfinite(data);scaled=np.zeros_like(data);scaled[valid]=np.clip((data[valid]-low)/(high-low),0,1)
                rgb=palette(scaled,bytes=True)[...,:3];rgb[~valid]=0
                Image.fromarray(np.flipud(rgb)).resize((1024,1024),Image.Resampling.NEAREST).save(stage/name)
            fig,axes=plt.subplots(1,2,figsize=(12,6.8),facecolor='#08111e');fig.subplots_adjust(left=.055,right=.91,bottom=.12,top=.79,wspace=.14)
            box=float(ma['box_size_mpc_h']);im=None
            for ax,row,label in zip(axes,loaded,('A','B')):
                meta=row['metadata'];im=ax.imshow(row['array'],origin='lower',extent=(0,box,0,box),cmap='magma',vmin=low,vmax=high,interpolation='nearest');ax.grid(False)
                ax.set_title('%s · snapshot %d · N=%s³'%(label,row['snapshot'],meta.get('particles_per_dimension','?')),color='white',fontsize=11)
                ax.set_xlabel(meta['projection'][0]+' [comoving Mpc/h]',color='#c5d6e8');ax.set_ylabel(meta['projection'][1]+' [comoving Mpc/h]',color='#c5d6e8');ax.tick_params(colors='#c5d6e8',labelsize=8)
            colorbar=fig.colorbar(im,ax=axes,fraction=.025,pad=.03);colorbar.set_label('log10 value ['+str(ma.get('measured_quantity_units','unit unavailable'))+']',color='#c5d6e8');colorbar.ax.tick_params(colors='#c5d6e8',labelsize=8)
            fig.suptitle('Resolution comparison · %s · z=%.4g / %.4g'%(ma['field'],ma['redshift'],mb['redshift']),color='white',fontsize=16,y=.96)
            fig.text(.5,.89,'Shared display scale · matching epochs do not establish matched initial phases',ha='center',color='#b8cadc',fontsize=10)
            fig.savefig(stage/'resolution_reveal.png',dpi=180,facecolor=fig.get_facecolor());fig.savefig(stage/'resolution_reveal.pdf',facecolor=fig.get_facecolor());plt.close(fig)
            details=[]
            for label,row in zip(('A','B'),loaded):
                m=row['metadata'];details.append({'label':label,'run':str(row['root']),'bundle':str(row['bundle']),
                    'map_contract_sha256':row['manifest_hash'],'figure':row['figure_name'],'figure_sha256':row['figure_sha256'],
                    'numeric_asset':row['numeric_name'],'numeric_sha256':row['numeric_sha256'],'snapshot':row['snapshot'],
                    'redshift':m['redshift'],'particles_per_dimension':m.get('particles_per_dimension'),
                    'particle_mass_msun_h':m.get('particle_mass_msun_h'),'grid_shape':list(row['array'].shape)})
            def desc(d):return '%s · z=%.5g · N=%s³ · map %s pixels'%(d['label'],d['redshift'],d['particles_per_dimension'],' × '.join(map(str,d['grid_shape'])))
            page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Resolution reveal</title><style>body{background:#08111e;color:#edf5ff;font:16px system-ui;margin:0}main{max-width:960px;margin:auto;padding:28px}p{color:#b8cbdf}label{display:block;margin:18px 0}.view{position:relative;width:100%%;aspect-ratio:1;background:#000;overflow:hidden}.view img{position:absolute;width:100%%;height:100%%;image-rendering:pixelated}.upper{clip-path:inset(0 calc(100%% - var(--cut)) 0 0)}input{width:100%%}a{color:#8de9fa}</style><main><h1>Resolution reveal</h1><p>Drag the divider: B on the left, A on the right. Equal box, field and matched saved redshift; shared color scale. Phase matching is not established by this view.</p><p>%s<br>%s</p><label for="cut">Divider</label><input id="cut" type="range" min="0" max="100" value="50"><div class="view" id="view" style="--cut:50%%"><img src="run_a.png" alt="Run A projected map"><img class="upper" src="run_b.png" alt="Run B projected map"></div><p><a href="resolution_reveal.png">Open side-by-side figure</a></p></main><script>const s=document.getElementById('cut'),v=document.getElementById('view');s.addEventListener('input',()=>v.style.setProperty('--cut',s.value+'%%'));</script></html>'''%(html.escape(desc(details[0])),html.escape(desc(details[1])))
            (stage/'resolution_reveal.html').write_text(page,encoding='utf-8')
            for row in loaded:
                if file_sha256(row['manifest'])!=row['manifest_hash'] or file_sha256(row['bundle']/row['figure_name'])!=row['figure_sha256'] or file_sha256(row['bundle']/row['numeric_name'])!=row['numeric_sha256']:raise ValueError('Source map changed during comparison')
                verify_product_provenance(row['root'],row['contract']['provenance'])
            atomic_json(stage/'resolution_contract.json',{'schema':'resolution-reveal-v1','science_acceptance':'NOT_ACCEPTED','inputs':details,'map_key':map_key,
                'redshift_tolerance':redshift_tolerance,'display_log10_limits':[low,high],'algorithm_sha256':file_sha256(__file__),
                'render_scope':'Full periodic projected grid; any source figure display crop is not applied.',
                'interpretation':'Display comparison only; equal epochs and boxes do not establish common initial phases, identical physics, or numerical convergence.',
                'artifacts':{name:file_sha256(stage/name) for name in ('run_a.png','run_b.png','resolution_reveal.png','resolution_reveal.pdf','resolution_reveal.html')}})
            publish_directory(stage,out);return out/'resolution_reveal.html'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
