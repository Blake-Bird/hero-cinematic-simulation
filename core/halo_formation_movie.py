"""Animate saved density maps and their linked, positioned halo ancestry."""
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


def tree_layout(tree):
    nodes=tree['nodes'];parents=[[] for _ in nodes]
    for a,b in tree['edges']:
        if not (0<=a<len(nodes) and 0<=b<a):raise ValueError('Tree edges must point from a later-listed progenitor to its earlier-listed descendant')
        parents[b].append(a)
    x=np.zeros(len(nodes));leaf=0
    for i in reversed(range(len(nodes))):
        if parents[i]:x[i]=np.mean([x[p] for p in parents[i]])
        else:x[i]=leaf;leaf+=1
    return x


def make_halo_formation_movie(analysis_bundle,map_bundle,destination,*,map_key,fps=24,seconds_per_snapshot=1,memory_gib=1.):
    from core.select_merger_tree import verify_tree_bundle
    from core.foundation import atomic_json,file_sha256,verify_product_provenance,publish_directory
    analysis=Path(analysis_bundle).expanduser().resolve();maps=Path(map_bundle).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh halo-formation movie directory')
    if type(fps) is not int or not 1<=fps<=120 or type(seconds_per_snapshot) is not int or not 1<=seconds_per_snapshot<=60:raise ValueError('Positive bounded movie timing required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite movie allocation required')
    analysis_record=verify_tree_bundle(analysis)
    tree_path=analysis/'merger_tree.json';tree=json.loads(tree_path.read_text());tree_hash=file_sha256(tree_path)
    if tree.get('schema')!='adjacent-merger-tree-v1' or not tree['nodes']:raise ValueError('Positioned linked merger tree required')
    contract_path=maps/'map_contract.json';contract_hash=file_sha256(contract_path);contract=json.loads(contract_path.read_text())
    if contract.get('schema')!='map-publication-v1':raise ValueError('Published map bundle required')
    if analysis_record['data_contract']['run_namespace']!=contract['provenance']['run_namespace']:raise ValueError('Tree and maps belong to different run namespaces')
    root=Path(analysis_record['source_run']);verify_product_provenance(root,contract['provenance'])
    inventory={r['path']:r['sha256'] for r in contract['artifacts']}
    def checked(name):
        if not isinstance(name,str) or Path(name).name!=name or name not in inventory:raise ValueError('Unlisted map asset: '+str(name))
        path=maps/name
        if not path.is_file() or file_sha256(path)!=inventory[name]:raise ValueError('Map asset missing or changed: '+name)
        return path
    scale=json.loads(checked('scale.json').read_text());settings=scale['maps'].get(map_key)
    if not settings or settings.get('status')!='COMPLETE' or settings.get('normalization')!='global_fixed':raise ValueError('A globally normalized map series is required')
    records=[]
    for name in inventory:
        if not re.fullmatch(re.escape(map_key)+r'_\d+\.figure\.json',name):continue
        meta=json.loads(checked(name).read_text()).get('scientific_metadata',{})
        if meta.get('overlay_layers')!=[]:continue
        checked(meta['numeric_asset']);records.append({'figure':name,'numeric_asset':meta['numeric_asset'],'snapshot':int(meta['snapshot']),
            'redshift':float(meta['redshift']),'box':float(meta['box_size_mpc_h']),'projection':meta['projection'],
            'colormap':meta['density_colormap'],'slice':meta['slice'],'map_window_mpc_h':meta.get('map_window_mpc_h')})
    records.sort(key=lambda r:-r['redshift'])
    if len(records)<2 or any(records[i]['redshift']<=records[i+1]['redshift'] for i in range(len(records)-1)):raise ValueError('At least two distinct saved map epochs required')
    first=records[0];box=tree.get('box_mpc_h')
    if box is None or not math.isclose(box,first['box'],rel_tol=1e-10):raise ValueError('Tree and map box sizes differ')
    from core.merger_tree_visual import branch_identity
    nodes=tree['nodes'];identity=branch_identity(tree);x=tree_layout(tree);a=np.asarray([n['scale_factor'] for n in nodes],float)
    if not np.isfinite(a).all() or np.any(a<=0) or not any('position_mpc_h' in n for n in nodes):raise ValueError('Tree has no usable saved positions')
    xmin,xmax=float(np.min(x)),float(np.max(x));margin=max(.5,(xmax-xmin)*.08)
    time_min=min(float(np.min(a)),min(1/(1+r['redshift']) for r in records))
    time_max=max(float(np.max(a)),max(1/(1+r['redshift']) for r in records))
    mapping={'xy':(0,1,2),'xz':(0,2,1),'yz':(1,2,0)};ix,iy,iz=mapping[first['projection']]
    for r in records:
        if any(r[k]!=first[k] for k in ('box','projection','colormap','map_window_mpc_h')) or any(r['slice'].get(k)!=first['slice'].get(k) for k in ('thickness_mpc_h','center_mpc_h','line_of_sight_axis','assignment_kernel')):raise ValueError('Map geometry changes across epochs')
        matched=[n for n in nodes if n['snapshot']==r['snapshot']]
        if any(not math.isclose(float(n['redshift']),r['redshift'],rel_tol=1e-8,abs_tol=1e-8) for n in matched):raise ValueError('Tree and map epochs disagree')
    out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.halo-formation-',dir=out.parent));movie=stage/'halo_formation.mp4'
    try:
        with imageio.get_writer(movie,fps=fps,codec='libx264',quality=8,macro_block_size=16) as writer:
            for r in records:
                with h5py.File(checked(r['numeric_asset']),'r') as h:
                    ds=h['log10_display']
                    if len(ds.shape)!=2 or np.prod(ds.shape,dtype=np.int64)*32>memory_gib*1024**3*.75:raise MemoryError('Formation map exceeds declared memory')
                    density=ds[:]
                fig,(spatial,diagram)=plt.subplots(1,2,figsize=(12.8,6.4),dpi=100,facecolor='#08111e',gridspec_kw={'width_ratios':[1.4,1]})
                fig.subplots_adjust(left=.06,right=.96,bottom=.12,top=.82,wspace=.25)
                cmap=matplotlib.colormaps[r['colormap']].with_extremes(bad=matplotlib.colormaps[r['colormap']](0))
                spatial.imshow(np.ma.masked_invalid(density),origin='lower',extent=(0,box,0,box),cmap=cmap,vmin=settings['vmin'],vmax=settings['vmax'],interpolation='nearest')
                if r['map_window_mpc_h'] is not None:
                    x0,x1,y0,y1=r['map_window_mpc_h'];spatial.set_xlim(x0,x1);spatial.set_ylim(y0,y1)
                visible=[];slab=r['slice'];thick=slab.get('thickness_mpc_h')
                for n in nodes:
                    p=n.get('position_mpc_h')
                    if n['snapshot']!=r['snapshot'] or p is None or len(p)!=3 or any(not math.isfinite(v) or not 0<=v<box for v in p):continue
                    if thick is not None:
                        dz=(p[iz]-slab['center_mpc_h']+box/2)%box-box/2
                        if abs(dz)>thick/2:continue
                    visible.append(n)
                if visible:
                    spatial.scatter([n['position_mpc_h'][ix] for n in visible],[n['position_mpc_h'][iy] for n in visible],s=42,facecolors='none',edgecolors='#77efff',linewidths=1.5)
                    for n in visible:
                        i = next(i for i, node in enumerate(nodes) if node is n)
                        spatial.annotate(identity[i]['alias']+' · ID '+n['id'],
                            (n['position_mpc_h'][ix],n['position_mpc_h'][iy]),xytext=(8,8),
                            textcoords='offset points',color=identity[i]['color'],fontsize=8,
                            bbox={'facecolor':'#08111e','edgecolor':identity[i]['color'],'alpha':.85,'pad':2})
                spatial.set_title('Saved density · %d linked halo positions'%len(visible),color='white',fontsize=11)
                spatial.set_xlabel(r['projection'][0]+' [comoving Mpc/h]',color='#c5d6e8');spatial.set_ylabel(r['projection'][1]+' [comoving Mpc/h]',color='#c5d6e8')
                spatial.tick_params(colors='#c5d6e8',labelsize=8);spatial.grid(False)
                diagram.set_facecolor('#14253a');current=1/(1+r['redshift']);active=a<=current+1e-10
                for parent,child in tree['edges']:
                    if active[parent] and active[child]:diagram.plot([x[parent],x[child]],[a[parent],a[child]],color='#6c91aa',lw=1)
                shown=np.flatnonzero(active)
                if len(shown):
                    masses=np.array([nodes[i]['mass_msun_h'] for i in shown]);sizes=16+130*np.sqrt(masses/max(n['mass_msun_h'] for n in nodes))
                    diagram.scatter(x[shown],a[shown],s=sizes,c=[identity[i]['color'] for i in shown],edgecolors='white',linewidths=.35,zorder=3)
                diagram.axhline(current,color='#f8cd77',ls='--',lw=1)
                diagram.set_xlim(xmin-margin,xmax+margin);diagram.set_ylim(time_max*1.02,max(0.,time_min*.98));diagram.set_xticks([])
                diagram.set_xlabel('branches',color='#c5d6e8');diagram.set_ylabel('scale factor a · time downward',color='#c5d6e8')
                diagram.set_title('Linked progenitors'+(' · truncated' if tree['truncated'] else ''),color='white',fontsize=11)
                diagram.tick_params(colors='#c5d6e8',labelsize=8);diagram.grid(alpha=.15)
                fig.suptitle('Halo formation · z=%.4g · snapshot %d'%(r['redshift'],r['snapshot']),color='white',fontsize=16,y=.95)
                fig.text(.5,.89,'Saved epochs only · circles show linked catalogued progenitors · no interpolated halo positions',ha='center',color='#b8cadc',fontsize=9)
                fig.canvas.draw();frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
                for _ in range(fps*seconds_per_snapshot):writer.append_data(frame)
                plt.close(fig)
                r['visible_position_count']=len(visible)
        checked('scale.json')
        for r in records:checked(r['figure']);checked(r['numeric_asset'])
        verify_tree_bundle(analysis);verify_product_provenance(root,contract['provenance'])
        if file_sha256(contract_path)!=contract_hash or file_sha256(tree_path)!=tree_hash:raise ValueError('Tree or map source changed')
        atomic_json(stage/'movie_contract.json',{'schema':'halo-formation-movie-v1','science_acceptance':'NOT_ACCEPTED',
            'analysis_bundle':str(analysis),'tree_sha256':tree_hash,'map_bundle':str(maps),'map_contract_sha256':contract_hash,
            'algorithm_sha256':file_sha256(__file__),'map_key':map_key,'frames':records,'fps':fps,'seconds_per_snapshot':seconds_per_snapshot,
            'movie_sha256':file_sha256(movie),'timing_scope':'Equal display time for actual saved epochs; no interpolation.',
            'position_scope':'Only saved linked catalogue positions; slab excludes halos outside the displayed periodic slice.'})
        publish_directory(stage,out);return out/'halo_formation.mp4'
    except BaseException as error:
        atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)})
        raise
