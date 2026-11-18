"""Connect an existing verified merger tree to its saved projected matter maps."""
from __future__ import annotations
import json
import math
import re
import tempfile
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image


def map_location(node,meta):
    """Return map marker and slab membership for a measured 3-D halo position."""
    p=node.get('position_mpc_h');box=float(meta['box_size_mpc_h']);projection=meta['projection'];slab=meta['slice']
    if p is None:return {'status':'POSITION_UNAVAILABLE'}
    p=np.asarray(p,float)
    if p.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(box) or box<=0 or np.any(p<0) or np.any(p>=box):return {'status':'POSITION_UNAVAILABLE'}
    axes={'xy':(0,1),'xz':(0,2),'yz':(1,2)}[projection];los=({0,1,2}-set(axes)).pop()
    if int(slab['line_of_sight_axis'])!=los:raise ValueError('Map line-of-sight axis disagrees with projection')
    thickness=slab.get('thickness_mpc_h');center=slab.get('center_mpc_h')
    inside=True if thickness is None else abs((p[los]-float(center)+box/2)%box-box/2)<=float(thickness)/2
    return {'status':'IN_SLAB' if inside else 'OUTSIDE_SLAB','x':float(p[axes[0]]/box*100),'y':float(100-p[axes[1]]/box*100)}


def connect_tree_to_maps(run_dir,analysis_bundle,map_bundle,destination,*,map_key,memory_gib=1.):
    from core.select_merger_tree import verify_tree_bundle
    from core.foundation import atomic_json,file_sha256,verify_product_provenance,publish_directory
    from core.native import native_run_lock
    root=Path(run_dir).expanduser().resolve();analysis=Path(analysis_bundle).expanduser().resolve();maps=Path(map_bundle).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh tree-map output directory')
    if not isinstance(map_key,str) or not re.fullmatch(r'[A-Za-z0-9_]+',map_key):raise ValueError('Valid saved map key required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite memory allocation required')
    with native_run_lock(root,operation='merger tree map'):
        record=verify_tree_bundle(analysis)
        if Path(record['source_run']).resolve()!=root:raise ValueError('Tree analysis belongs to another run')
        tree_path=analysis/'merger_tree.json';tree=json.loads(tree_path.read_text());page=(analysis/'merger_tree.html').read_text()
        if tree.get('schema')!='adjacent-merger-tree-v1' or page.count('<circle cx=')!=len(tree['nodes']):raise ValueError('Tree data and interactive page are inconsistent')
        manifest=maps/'map_contract.json';manifest_hash=file_sha256(manifest);contract=json.loads(manifest.read_text())
        if contract.get('schema')!='map-publication-v1' or contract['provenance']['run_namespace']!=record['data_contract']['run_namespace']:raise ValueError('Map and tree must belong to the same run namespace')
        verify_product_provenance(root,contract['provenance']);inventory={r['path']:r['sha256'] for r in contract['artifacts']}
        def checked(name):
            if not isinstance(name,str) or Path(name).name!=name or name not in inventory:raise ValueError('Unlisted map asset')
            p=maps/name
            if not p.is_file() or file_sha256(p)!=inventory[name]:raise ValueError('Map asset changed or missing')
            return p
        scale=json.loads(checked('scale.json').read_text())['maps'].get(map_key)
        if not scale or scale.get('status')!='COMPLETE' or scale.get('normalization')!='global_fixed':raise ValueError('Fixed global-scale map series required')
        series={}
        for name in inventory:
            if not re.fullmatch(re.escape(map_key)+r'_\d+\.figure\.json',name):continue
            meta=json.loads(checked(name).read_text())['scientific_metadata'];snap=int(meta['snapshot'])
            if snap in series or meta.get('overlay_layers')!=[] or meta.get('field') not in ('dm','total','gas','stars'):raise ValueError('Unique unadorned density maps required')
            if tree.get('box_mpc_h') is None or not math.isclose(float(tree['box_mpc_h']),float(meta['box_size_mpc_h']),rel_tol=1e-10):raise ValueError('Tree and map boxes differ')
            series[snap]={'figure':name,'metadata':meta,'numeric':meta['numeric_asset']};checked(meta['numeric_asset'])
        if not series:raise ValueError('Map series has no epochs')
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.tree-map-',dir=out.parent))
        try:
            used=set();locations=[]
            for node in tree['nodes']:
                row=series.get(node['snapshot'])
                if row is None:locations.append({'status':'MAP_UNAVAILABLE'});continue
                meta=row['metadata']
                if not math.isclose(float(node['redshift']),float(meta['redshift']),rel_tol=1e-6,abs_tol=1e-6):raise ValueError('Tree node and map redshifts differ')
                locations.append({'snapshot':node['snapshot'],**map_location(node,meta)});used.add(node['snapshot'])
            for snap in sorted(used):
                row=series[snap];meta=row['metadata']
                with h5py.File(checked(row['numeric']),'r') as f:
                    ds=f['log10_display']
                    if len(ds.shape)!=2 or np.prod(ds.shape,dtype=np.int64)*32>memory_gib*1024**3*.7:raise MemoryError('Map exceeds declared workspace')
                    value=np.asarray(ds[:],float)
                valid=np.isfinite(value);normal=np.zeros_like(value);normal[valid]=np.clip((value[valid]-float(scale['vmin']))/(float(scale['vmax'])-float(scale['vmin'])),0,1)
                color=matplotlib.colormaps[meta['density_colormap']](normal,bytes=True)[...,:3];color[~valid]=0
                Image.fromarray(np.flipud(color)).resize((768,768),Image.Resampling.NEAREST).save(stage/('map_%03d.png'%snap))
            payload=json.dumps(locations,separators=(',',':')).replace('<','\\u003c')
            insertion='''<section style="padding:20px 32px;background:#101e30"><h2>Selected halo in its saved density map</h2><p id="map-info">Select a tree node to inspect the simulation at its epoch.</p><div id="map-view" style="display:none;position:relative;width:min(520px,100%);aspect-ratio:1;background:#000"><img id="node-map" alt="Projected simulation density at the halo's saved epoch" style="width:100%;height:100%;position:absolute"><svg viewBox="0 0 100 100" style="position:absolute;inset:0;width:100%;height:100%;min-width:0;background:transparent;pointer-events:none"><circle id="map-marker" r="2" style="fill:none;stroke:#80ffff;stroke-width:.6"/></svg></div><p>Full projected box; shared fixed map color scale. A ring marks the projected halo center only when it lies within the displayed slab.</p></section>'''
            if '<div class="scroll">' not in page or '</body>' in page:raise ValueError('Unexpected tree HTML layout')
            page=page.replace('<div class="scroll">',insertion+'<div class="scroll">',1)
            script='''<script>const linked=%s;document.querySelectorAll('.scroll circle').forEach((c,i)=>{const select=()=>{const r=linked[i],view=document.getElementById('map-view'),info=document.getElementById('map-info'),marker=document.getElementById('map-marker');if(r.status==='MAP_UNAVAILABLE'){view.style.display='none';info.textContent='No matching saved density map at this epoch.';return}view.style.display='block';document.getElementById('node-map').src='map_'+String(r.snapshot).padStart(3,'0')+'.png';marker.setAttribute('visibility',r.status==='IN_SLAB'?'visible':'hidden');if(r.status==='IN_SLAB'){marker.setAttribute('cx',r.x);marker.setAttribute('cy',r.y)}info.textContent=r.status==='IN_SLAB'?'Halo center in the displayed map slab.':r.status==='OUTSIDE_SLAB'?'Halo center is outside this density-map slab.':'Halo position is unavailable in the source catalogue.'};c.addEventListener('click',select);c.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select()}})})</script>'''%payload
            page=page.replace('</html>',script+'</html>',1);(stage/'tree_with_map.html').write_text(page,encoding='utf-8')
            for row in series.values():checked(row['figure']);checked(row['numeric'])
            checked('scale.json')
            if file_sha256(manifest)!=manifest_hash:raise ValueError('Map contract changed')
            verify_product_provenance(root,contract['provenance']);verify_tree_bundle(analysis)
            atomic_json(stage/'tree_map_contract.json',{'schema':'tree-map-v1','science_acceptance':'NOT_ACCEPTED','run':str(root),'analysis_bundle':str(analysis),'analysis_provenance_sha256':file_sha256((Path(record.get('source_analysis_bundle',analysis))/'analysis_provenance.json')),'map_bundle':str(maps),'map_contract_sha256':manifest_hash,'map_key':map_key,'nodes':locations,'algorithm_sha256':file_sha256(__file__),'artifacts':{p.name:file_sha256(p) for p in stage.iterdir() if p.is_file()}})
            publish_directory(stage,out);return out/'tree_with_map.html'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
