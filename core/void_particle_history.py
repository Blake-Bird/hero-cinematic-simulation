"""Measure exact dark-matter particle overlap between saved void-centered apertures."""
from __future__ import annotations
import html
import json
import math
import os
import tempfile
from pathlib import Path
from urllib.parse import quote
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sphere_ids(files,center,box,radius,*,max_ids,chunk_rows=200000):
    """Read every exact type-1 ID inside a periodic comoving sphere."""
    from core.data import exact_uint64_ids
    center=np.asarray(center,float)
    if center.shape!=(3,) or not np.isfinite(center).all() or not math.isfinite(box) or box<=0 or not 0<radius<box/2 or type(max_ids) is not int or max_ids<1:raise ValueError('Valid center, box, radius below half-box and ID limit required')
    result=[];total=0
    for path in files:
        with h5py.File(path,'r') as f:
            if 'PartType1' not in f:continue
            g=f['PartType1'];xyz=g['Coordinates'];ids=g['ParticleIDs'];n=len(ids)
            if ids.ndim!=1 or xyz.shape!=(n,3):raise ValueError('Particle IDs and coordinates do not align')
            for start in range(0,n,chunk_rows):
                stop=min(start+chunk_rows,n);p=np.asarray(xyz[start:stop],float)/1000.;identity=exact_uint64_ids(ids[start:stop],unique=False)
                if not np.isfinite(p).all() or np.any(p<0) or np.any(p>=box):raise ValueError('Invalid saved periodic positions')
                d=(p-center+box/2)%box-box/2;chosen=identity[np.einsum('ij,ij->i',d,d)<radius**2]
                total+=len(chosen)
                if total>max_ids:raise MemoryError('Aperture particle IDs exceed declared workspace')
                if len(chosen):result.append(chosen)
    merged=np.sort(np.concatenate(result)) if result else np.empty(0,dtype=np.uint64)
    if len(merged)>1 and np.any(merged[1:]==merged[:-1]):raise ValueError('Duplicate particle IDs in aperture')
    return merged


def overlap_metrics(anchor,current,present_count):
    a=np.asarray(anchor);b=np.asarray(current)
    if a.ndim!=1 or b.ndim!=1 or a.dtype.kind not in 'ui' or b.dtype.kind not in 'ui' or len(np.unique(a))!=len(a) or len(np.unique(b))!=len(b) or type(present_count) is not int or not 0<=present_count<=len(a):raise ValueError('Unique exact IDs and valid presence count required')
    shared=np.intersect1d(a,b,assume_unique=True);union=len(a)+len(b)-len(shared)
    return {'anchor_count':len(a),'current_count':len(b),'anchor_present':present_count,'shared_count':len(shared),
            'fraction_anchor':float(len(shared)/len(a)) if len(a) else None,'fraction_current':float(len(shared)/len(b)) if len(b) else None,
            'jaccard':float(len(shared)/union) if union else None},shared


def make_void_particle_history(run_dir,void_bundle,destination,*,radius_mpc_h,memory_gib=1.):
    from core.analysis import read_snapshot_header
    from core.data import validate_native_shard_counts
    from core.zoom import _run_lineage
    from core.native import native_run_lock
    from core.particle_origin_movie import matching_positions
    from core.foundation import atomic_json,file_sha256,verify_product_provenance,publish_directory
    root=Path(run_dir).expanduser().resolve();bundle=Path(void_bundle).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh void particle-history directory')
    if isinstance(radius_mpc_h,bool) or not isinstance(radius_mpc_h,(int,float)) or not math.isfinite(radius_mpc_h) or radius_mpc_h<=0:raise ValueError('A positive common aperture radius is required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<.25:raise ValueError('At least 0.25 GiB memory required')
    max_ids=int((memory_gib*1024**3*.7-64*1024**2)/160)
    if max_ids<1:raise MemoryError('Declared memory leaves no ID workspace')
    with native_run_lock(root,operation='void particle history'):
        contract_path=bundle/'void_contract.json';contract_hash=file_sha256(contract_path);contract=json.loads(contract_path.read_text())
        if contract.get('schema')!='void-explorer-v1' or Path(contract['run']).resolve()!=root:raise ValueError('Void bundle must belong to the selected run')
        source_page=bundle/'void_explorer.html'
        if file_sha256(source_page)!=contract.get('artifacts',{}).get('void_explorer.html'):raise ValueError('Void explorer page changed')
        for source in contract['source_provenance']:verify_product_provenance(root,source)
        lineage=_run_lineage(root);epochs=contract['epochs'];box=float(contract['box_mpc_h'])
        if not 0<radius_mpc_h<box/2:raise ValueError('Aperture radius must be below half the periodic box')
        if len(epochs)<2 or epochs[-1]['status']!='MATCHED':raise ValueError('Latest saved void minimum is unavailable')
        anchor_epoch=epochs[-1];anchor_limit=anchor_epoch.get('radius_mpc_h')
        if anchor_limit is None or radius_mpc_h>anchor_limit:raise ValueError('Common aperture exceeds the supported anchor void radius')
        source_files=[];prepared=[]
        for epoch in epochs:
            if epoch['status']!='MATCHED':prepared.append({'epoch':epoch,'files':None});continue
            if epoch.get('radius_mpc_h') is None or float(epoch['radius_mpc_h'])<radius_mpc_h:
                prepared.append({'epoch':epoch,'files':None,'unsupported':True});continue
            snap=int(epoch['snapshot']);header,files=read_snapshot_header(root,snap);validate_native_shard_counts(files,'NumPart_Total')
            if not math.isclose(float(header['Time']),float(epoch['scale_factor']),rel_tol=1e-10) or not math.isclose(float(header['BoxSize'])/1000.,box,rel_tol=1e-10):raise ValueError('Void and particle epoch or box differ')
            for p in files:
                with h5py.File(p) as f:
                    hdr=f['Header'].attrs;prefix=f['Parameters'].attrs.get('InitCondFile');prefix=prefix.decode() if isinstance(prefix,bytes) else prefix
                    if prefix!=lineage['prefix'] or not math.isclose(float(hdr['Time']),float(epoch['scale_factor']),rel_tol=1e-12) or not math.isclose(float(hdr['BoxSize'])/1000.,box,rel_tol=1e-12):raise ValueError('Particle shard differs from run IC namespace or epoch')
                source_files.append({'path':str(Path(p).resolve()),'sha256':file_sha256(p)})
            prepared.append({'epoch':epoch,'files':files})
        anchor=sphere_ids(prepared[-1]['files'],anchor_epoch['center_mpc_h'],box,radius_mpc_h,max_ids=max_ids)
        if not len(anchor):raise ValueError('Anchor aperture has no saved dark-matter particles')
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.void-particles-',dir=out.parent))
        try:
            rows=[]
            with h5py.File(stage/'particle_overlap.hdf5','w') as h:
                h.attrs['scope']='Exact same-run type-1 particle IDs in fixed comoving spherical apertures';h.create_dataset('anchor_ids',data=anchor,compression='gzip',fletcher32=True)
                for item in prepared:
                    epoch=item['epoch'];row={'snapshot':int(epoch['snapshot']),'redshift':float(epoch['redshift']),'scale_factor':float(epoch['scale_factor']),'center_mpc_h':epoch.get('center_mpc_h'),'void_radius_mpc_h':epoch.get('radius_mpc_h'),'void_radius_status':epoch.get('radius_status')}
                    if item['files'] is None:
                        row.update({'status':'APERTURE_UNSUPPORTED' if item.get('unsupported') else 'UNMATCHED',
                                    'reason':'Density-based radius does not support the requested aperture' if item.get('unsupported') else epoch.get('reason')});rows.append(row);continue
                    current=sphere_ids(item['files'],epoch['center_mpc_h'],box,radius_mpc_h,max_ids=max_ids);present,_=matching_positions(item['files'],anchor)
                    metrics,shared=overlap_metrics(anchor,current,int(len(present)));row.update({'status':'COMPLETE',**metrics});rows.append(row)
                    group=h.create_group('snapshot_%03d'%row['snapshot']);group.create_dataset('aperture_ids',data=current,compression='gzip',fletcher32=True);group.create_dataset('shared_anchor_ids',data=shared,compression='gzip',fletcher32=True)
            fig,ax=plt.subplots(figsize=(8,5),facecolor='#081321');ax.set_facecolor('#14283e');scales=[r['scale_factor'] for r in rows]
            for key,color,label in [('fraction_anchor','#7ce6f2','Fraction of final-region particles'),('fraction_current','#ffc276','Fraction of this epoch’s particles'),('jaccard','#c4a9ff','Jaccard overlap')]:
                values=[r.get(key) if r.get(key) is not None else np.nan for r in rows];ax.plot(scales,values,'-o',color=color,label=label)
            ax.set(xlabel='Scale factor',ylabel='Exact particle-ID overlap',ylim=(0,1.02));ax.tick_params(colors='#d5e5f2');ax.xaxis.label.set_color('#d5e5f2');ax.yaxis.label.set_color('#d5e5f2');ax.grid(alpha=.15);ax.legend(fontsize=8,labelcolor='#e5f4ff',facecolor='#14283e',edgecolor='#587188');fig.tight_layout();fig.savefig(stage/'void_particle_overlap.png',dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
            def show(value):return '—' if value is None else ('%.4g'%value if isinstance(value,float) else str(value))
            table=''.join('<tr>'+''.join('<td>'+html.escape(show(row.get(k)))+'</td>' for k in ('snapshot','redshift','status','current_count','anchor_present','shared_count','fraction_anchor','fraction_current','jaccard','void_radius_mpc_h','void_radius_status'))+'</tr>' for row in rows)
            explorer_link=html.escape(quote(os.path.relpath(source_page,out),safe='/'),quote=True)
            page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Void particle continuity</title><style>body{margin:0;background:#081321;color:#edf7ff;font:16px system-ui}main{max-width:1400px;margin:auto;padding:28px}p{color:#bbd0e1}a{color:#8fe6ec}img{width:min(100%%,950px)}.scroll{overflow:auto}table{border-collapse:collapse;width:100%%;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #345069;white-space:nowrap}th{color:#8fe6ec}</style><main><h1>Void particle continuity</h1><p><a href="%s">Open the corresponding void explorer</a></p><p>Exact same-run dark-matter IDs inside a fixed %.5g comoving Mpc/h radius around the linked density minima. The final saved aperture is the anchor.</p><img src="void_particle_overlap.png" alt="Particle overlap across saved epochs"><p>Overlap measures how much the sampled matter agrees between apertures; it does not establish that the void boundary or surrounding filaments are physically identical. Missing candidate minima remain unmatched. Radius estimates may be lower bounds and use a different, density-based definition.</p><div class="scroll"><table><thead><tr><th>Snapshot</th><th>Redshift</th><th>Status</th><th>Particles in aperture</th><th>Anchor IDs present</th><th>Shared IDs</th><th>Fraction of anchor</th><th>Fraction of epoch</th><th>Jaccard</th><th>Density radius [Mpc/h]</th><th>Radius status</th></tr></thead><tbody>%s</tbody></table></div></main></html>'''%(explorer_link,radius_mpc_h,table)
            (stage/'void_particle_history.html').write_text(page,encoding='utf8')
            for source in contract['source_provenance']:verify_product_provenance(root,source)
            if file_sha256(source_page)!=contract['artifacts']['void_explorer.html'] or file_sha256(contract_path)!=contract_hash or file_sha256(lineage['path'])!=lineage['sha256'] or any(file_sha256(s['path'])!=s['sha256'] for s in source_files):raise ValueError('Void or native input changed during publication')
            atomic_json(stage/'particle_history_contract.json',{'schema':'void-particle-history-v1','science_acceptance':'NOT_ACCEPTED','run':str(root),'void_bundle':str(bundle),'void_contract_sha256':contract_hash,'radius_comoving_mpc_h':radius_mpc_h,'anchor_snapshot':int(anchor_epoch['snapshot']),'epochs':rows,'source_sha256':source_files,'algorithm_sha256':file_sha256(__file__),'interpretation':'Exact particle-set overlaps within equal-size periodic spherical apertures centered on proximity-linked minima. This is evidence about matter continuity, not a proof of persistent void identity or a measurement of a watershed boundary.','artifacts':{p.name:file_sha256(p) for p in stage.iterdir() if p.is_file()}})
            publish_directory(stage,out);return out/'void_particle_history.html'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
