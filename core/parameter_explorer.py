"""Offline explorer of explicitly supplied, source-verified simulation runs."""
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
import yaml
from PIL import Image


def parameter_value(config,path):
    if not isinstance(path,str) or not path or any(not part for part in path.split('.')):raise ValueError('A dotted configuration parameter is required')
    value=config
    for part in path.split('.'):
        if not isinstance(value,dict) or part not in value:return None
        value=value[part]
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float) and math.isfinite(value):return value
    raise ValueError('Explorer parameter must be a finite scalar, text or boolean')


def make_parameter_explorer(request,destination,*,memory_gib=1.):
    from core.analysis import verify_analysis_bundle
    from core.foundation import atomic_json,file_sha256,verify_product_provenance,publish_directory
    from core.native import native_run_lock
    out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh explorer output directory')
    if not isinstance(request,dict) or set(request)!={'schema','parameter','runs'} or request['schema']!='parameter-explorer-request-v1':raise ValueError('A complete parameter-explorer request is required')
    parameter=request['parameter'];parameter_value({},parameter);items=request['runs']
    if not isinstance(items,list) or not 2<=len(items)<=30:raise ValueError('Select two to thirty explicit runs')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite explorer memory required')
    keys={'run','map_bundle','map_key','snapshot','analysis_bundle'}
    for item in items:
        if not isinstance(item,dict) or set(item)!=keys or type(item['snapshot']) is not int or item['snapshot']<0:raise ValueError('Each run needs exact run/map/analysis paths, map key and snapshot')
    roots=[Path(item['run']).expanduser().resolve() for item in items]
    with ExitStack() as owners:
        for root in sorted(set(roots)):owners.enter_context(native_run_lock(root,operation='parameter explorer'))
        prepared=[]
        for root,item in zip(roots,items):
            maps=Path(item['map_bundle']).expanduser().resolve();analysis=Path(item['analysis_bundle']).expanduser().resolve()
            analysis_record=verify_analysis_bundle(analysis,verify_sources=True)
            if Path(analysis_record['source_run']).resolve()!=root:raise ValueError('Analysis bundle belongs to another run')
            manifest=maps/'map_contract.json';manifest_hash=file_sha256(manifest);contract=json.loads(manifest.read_text())
            if contract.get('schema')!='map-publication-v1' or contract['provenance']['run_namespace']!=analysis_record['data_contract']['run_namespace']:raise ValueError('Map and analysis bundles belong to different run namespaces')
            verify_product_provenance(root,contract['provenance']);inventory={r['path']:r['sha256'] for r in contract['artifacts']}
            def checked(name):
                if not isinstance(name,str) or Path(name).name!=name or name not in inventory:raise ValueError('Unlisted map asset')
                p=maps/name
                if not p.is_file() or file_sha256(p)!=inventory[name]:raise ValueError('Map asset missing or changed: '+name)
                return p
            scale=json.loads(checked('scale.json').read_text());setting=scale['maps'].get(item['map_key'])
            if not setting or setting.get('status')!='COMPLETE':raise ValueError('Requested map series is unavailable')
            figure=item['map_key']+'_%03d.figure.json'%item['snapshot'];meta=json.loads(checked(figure).read_text())['scientific_metadata']
            if meta.get('overlay_layers')!=[] or int(meta['snapshot'])!=item['snapshot']:raise ValueError('Unadorned saved map at requested epoch required')
            with h5py.File(checked(meta['numeric_asset']),'r') as h:
                ds=h['log10_display']
                if len(ds.shape)!=2 or ds.shape[0]!=ds.shape[1] or np.prod(ds.shape,dtype=np.int64)*80*len(items)>memory_gib*1024**3*.7:raise MemoryError('Explorer map arrays exceed declared workspace')
                data=np.asarray(ds[:],dtype=float)
            config=yaml.safe_load((root/'config.yaml').read_text());value=parameter_value(config,parameter)
            summary_path=analysis/'analysis_products_summary.json'
            if not summary_path.is_file():raise ValueError('Selected-analysis summary is required')
            summary=json.loads(summary_path.read_text());epoch=summary['snapshots'].get(str(item['snapshot']))
            if epoch is None or not math.isclose(float(epoch['redshift']),float(meta['redshift']),rel_tol=1e-8,abs_tol=1e-8):raise ValueError('Analysis and map epochs differ')
            hmf=None;h5path=analysis/'analysis_products.h5'
            with h5py.File(h5path,'r') as h:
                key='snap_%03d/hmf/data'%item['snapshot']
                if key in h and all(name in h[key] for name in ('mass_msun_h','dn_dlogM','counts')):
                    g=h[key];x=np.asarray(g['mass_msun_h'][:],float);y=np.asarray(g['dn_dlogM'][:],float);counts=np.asarray(g['counts'][:])
                    status=g['status'][()] if 'status' in g else None
                    if isinstance(status,bytes):status=status.decode('utf-8')
                    if x.ndim!=1 or x.shape!=y.shape or counts.shape!=x.shape or counts.dtype.kind not in 'ui':raise ValueError('HMF arrays are malformed')
                    supported=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
                    if status=='COMPLETE' and np.any(supported):hmf={'mass':x[supported], 'density':y[supported], 'binned_count':int(np.sum(counts,dtype=np.uint64)),
                        'density_units':str(g['dn_dlogM'].attrs.get('units','units not declared'))}
            prepared.append({'root':root,'maps':maps,'analysis':analysis,'analysis_record':analysis_record,'manifest':manifest,'manifest_hash':manifest_hash,
                'contract':contract,'figure':figure,'figure_hash':file_sha256(maps/figure),'numeric':meta['numeric_asset'],'numeric_hash':file_sha256(maps/meta['numeric_asset']),
                'value':value,'meta':meta,'array':data,'epoch':epoch,'hmf':hmf})
        fields={(r['meta']['field'],r['meta']['projection'],r['meta'].get('measured_quantity_units')) for r in prepared}
        if len(fields)!=1:raise ValueError('Explorer maps must share a field, projection and measured units')
        if next(iter(fields))[0]=='gas_temperature':
            raise ValueError('Temperature explorer requires an explicit common representation contract; select a mass map')
        finite=[r['array'][np.isfinite(r['array'])] for r in prepared]
        if any(not len(x) for x in finite):raise ValueError('Each explorer map requires finite data')
        low=min(float(np.percentile(x,1)) for x in finite);high=max(float(np.percentile(x,99.9)) for x in finite)
        if high<=low:high=np.nextafter(low,np.inf)
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.parameter-explorer-',dir=out.parent));palette=matplotlib.colormaps['magma']
        try:
            records=[];cards=[]
            for index,row in enumerate(prepared):
                data=row['array'];valid=np.isfinite(data);normalized=np.zeros_like(data);normalized[valid]=np.clip((data[valid]-low)/(high-low),0,1)
                rgb=palette(normalized,bytes=True)[...,:3];rgb[~valid]=0
                image_name='map_%03d.png'%index;Image.fromarray(np.flipud(rgb)).resize((900,900),Image.Resampling.NEAREST).save(stage/image_name)
                hmf_name=None
                if row['hmf'] is not None:
                    hmf_name='hmf_%03d.png'%index;fig,ax=plt.subplots(figsize=(5.5,4),facecolor='#0f1e30');ax.set_facecolor('#172b41')
                    ax.loglog(row['hmf']['mass'],row['hmf']['density'],color='#6de2f0',marker='o',ms=3)
                    ax.set_xlabel('halo mass [Msun/h]',color='#d8e7f5');ax.set_ylabel('dn/dlog10 M ['+row['hmf']['density_units']+']',color='#d8e7f5');ax.tick_params(colors='#c5d6e8');ax.grid(alpha=.15)
                    fig.tight_layout();fig.savefig(stage/hmf_name,dpi=150,facecolor=fig.get_facecolor());plt.close(fig)
                m=row['meta'];epoch=row['epoch'];record={'index':index,'run':str(row['root']),'map_bundle':str(row['maps']),'analysis_bundle':str(row['analysis']),
                    'parameter_value':row['value'],'snapshot':int(m['snapshot']),'redshift':float(m['redshift']),'field':m['field'],'projection':m['projection'],
                    'box_mpc_h':m['box_size_mpc_h'],'slice':m['slice'],'map_grid_shape':list(data.shape),'particles_per_dimension':m.get('particles_per_dimension'),
                    'particle_mass_msun_h':m.get('particle_mass_msun_h'),'cosmic_time_gyr':(epoch.get('epoch_contract') or {}).get('cosmic_time_gyr'),
                    'binned_halo_count':None if row['hmf'] is None else row['hmf']['binned_count'],'hmf_scope':'Sum of saved HMF bins; resolution cuts and bin range follow the source analysis.',
                    'map_contract_sha256':row['manifest_hash'],'map_numeric_sha256':row['numeric_hash'],'analysis_provenance_sha256':file_sha256(row['analysis']/'analysis_provenance.json'),
                    'map_image':image_name,'hmf_image':hmf_name}
                records.append(record)
                def show(value):return 'Unavailable' if value is None else str(value)
                text=[('Run',row['root'].name),(parameter,show(row['value'])),('Snapshot',m['snapshot']),('Redshift','%.6g'%float(m['redshift'])),
                      ('Cosmic time [Gyr]',show(record['cosmic_time_gyr'])),('Box [comoving Mpc/h]',show(record['box_mpc_h'])),
                      ('Slab thickness [comoving Mpc/h]',show(m['slice'].get('thickness_mpc_h'))),
                      ('Slab center [comoving Mpc/h]',show(m['slice'].get('center_mpc_h'))),
                      ('Particles per dimension',show(record['particles_per_dimension'])),('Particle mass [Msun/h]',show(record['particle_mass_msun_h'])),
                      ('Halos in saved HMF bins',show(record['binned_halo_count']))]
                table=''.join('<tr><th scope="row">%s</th><td>%s</td></tr>'%(html.escape(str(k)),html.escape(str(v))) for k,v in text)
                graph='<img src="%s" alt="Saved halo mass function">'%hmf_name if hmf_name else '<p>No supported HMF data in the selected analysis bundle.</p>'
                cards.append('<section class="run" id="run-%d"%s><div class="grid"><div><img src="%s" alt="Saved projected matter map"><p>Full projected box footprint; shared log10 color limits %.4g to %.4g [%s].</p></div><div><table>%s</table>%s</div></div></section>'%(index,'' if index==0 else ' hidden',image_name,low,high,html.escape(str(m.get('measured_quantity_units','unit unavailable'))),table,graph))
            options=''.join('<option value="%d">%s</option>'%(r['index'],html.escape('%s = %s · %s'%(parameter,r['parameter_value'] if r['parameter_value'] is not None else 'Unavailable',Path(r['run']).name))) for r in records)
            page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Simulation parameter explorer</title><style>body{background:#08111e;color:#eff7ff;font:16px system-ui;margin:0}main{max-width:1500px;padding:28px;margin:auto}p{color:#b9cde0}select{background:#1b3048;color:white;font:inherit;padding:9px;max-width:100%%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:26px}.grid img{width:100%%;height:auto}.run[hidden]{display:none}table{border-collapse:collapse;width:100%%}th,td{padding:8px;border-bottom:1px solid #38506a;text-align:left}th{color:#a9c4dd}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><main><h1>Simulation parameter explorer</h1><p>Choose a saved run. Shared map colors show the same measured units; epochs, box geometry and other settings may differ. This view does not establish causal effects, matched phases or convergence.</p><label for="run-choice">Saved run: </label><select id="run-choice">%s</select>%s</main><script>document.getElementById('run-choice').addEventListener('change',e=>document.querySelectorAll('.run').forEach((v,i)=>v.hidden=String(i)!==e.target.value));</script></html>'''%(options,''.join(cards))
            (stage/'parameter_explorer.html').write_text(page,encoding='utf-8')
            for row in prepared:
                if file_sha256(row['manifest'])!=row['manifest_hash'] or file_sha256(row['maps']/row['figure'])!=row['figure_hash'] or file_sha256(row['maps']/row['numeric'])!=row['numeric_hash']:raise ValueError('Explorer map input changed')
                verify_analysis_bundle(row['analysis'],verify_sources=True);verify_product_provenance(row['root'],row['contract']['provenance'])
            artifacts={p.name:file_sha256(p) for p in stage.iterdir() if p.is_file()}
            atomic_json(stage/'explorer_contract.json',{'schema':'parameter-explorer-v1','science_acceptance':'NOT_ACCEPTED','request':request,'parameter':parameter,'runs':records,
                'display_log10_limits':[low,high],'algorithm_sha256':file_sha256(__file__),
                'interpretation':'Descriptive views of declared runs. Differences cannot be attributed solely to the displayed parameter without independent matching and convergence evidence.',
                'artifacts':artifacts})
            publish_directory(stage,out);return out/'parameter_explorer.html'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
