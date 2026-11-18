"""Close-up of two linked progenitor branches using measured saved positions."""
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


def select_merger_pair(tree,descendant_id=None):
    """Find a merger with two position-bearing main branches at multiple common epochs."""
    nodes=tree['nodes'];parents=[[] for _ in nodes]
    for parent,child in tree['edges']:
        if not (0<=child<parent<len(nodes)):raise ValueError('Invalid merger-tree edge')
        parents[child].append(parent)
    def branch(index):
        result=[]
        while True:
            result.append(index)
            if not parents[index]:break
            index=max(parents[index],key=lambda i:(nodes[i]['mass_msun_h'],-int(nodes[i]['id'])))
        return {nodes[i]['snapshot']:i for i in result}
    choices=[];box=tree.get('box_mpc_h')
    if box is None or not math.isfinite(box) or box<=0:raise ValueError('A finite tree box is required')
    for child,group in enumerate(parents):
        if len(group)<2 or (descendant_id is not None and str(nodes[child]['id'])!=str(descendant_id)):continue
        group=sorted(group,key=lambda i:(-nodes[i]['mass_msun_h'],int(nodes[i]['id'])))
        for first in range(len(group)-1):
            for second in range(first+1,len(group)):
                one,two=branch(group[first]),branch(group[second]);rows=[]
                for snap in one.keys() & two.keys():
                    a,b=nodes[one[snap]],nodes[two[snap]];pa=a.get('position_mpc_h');pb=b.get('position_mpc_h')
                    if pa is None or pb is None or not math.isclose(a['scale_factor'],b['scale_factor'],rel_tol=1e-10):continue
                    pa=np.asarray(pa,float);pb=np.asarray(pb,float)
                    if pa.shape!=(3,) or pb.shape!=(3,) or not np.isfinite(pa).all() or not np.isfinite(pb).all() or np.any(pa<0) or np.any(pb<0) or np.any(pa>=box) or np.any(pb>=box):continue
                    delta=(pb-pa+box/2)%box-box/2
                    va=a.get('velocity_peculiar_km_s');vb=b.get('velocity_peculiar_km_s')
                    va=np.asarray(va,float) if va is not None else None
                    vb=np.asarray(vb,float) if vb is not None else None
                    if va is not None and (va.shape!=(3,) or not np.isfinite(va).all()):va=None
                    if vb is not None and (vb.shape!=(3,) or not np.isfinite(vb).all()):vb=None
                    speed=None
                    if va is not None and vb is not None:
                        speed=float(np.linalg.norm(vb-va))
                    rows.append({'snapshot':int(snap),'scale_factor':float(a['scale_factor']),'redshift':float(a['redshift']),
                        'id_a':a['id'],'id_b':b['id'],'position_a_mpc_h':pa.tolist(),'position_b_mpc_h':pb.tolist(),
                        'velocity_a_km_s':None if va is None else va.tolist(),'velocity_b_km_s':None if vb is None else vb.tolist(),
                        'separation_comoving_mpc_h':float(np.linalg.norm(delta)),'relative_peculiar_speed_km_s':speed})
                rows.sort(key=lambda r:r['scale_factor'])
                if len(rows)>=2:
                    mass=min(nodes[group[first]]['mass_msun_h'],nodes[group[second]]['mass_msun_h'])
                    choices.append((len(rows),mass,child,rows))
    if not choices:raise ValueError('No linked pair has positions at two common saved epochs')
    _,_,child,rows=max(choices,key=lambda c:(c[0],c[1],-c[2]))
    return {'descendant_id':str(nodes[child]['id']),'descendant_snapshot':int(nodes[child]['snapshot']),'rows':rows}


def make_halo_collision_movie(analysis_bundle,map_bundle,destination,*,map_key,descendant_id=None,fps=24,seconds_per_snapshot=1,memory_gib=1.):
    from core.select_merger_tree import verify_tree_bundle
    from core.foundation import atomic_json,file_sha256,verify_product_provenance,publish_directory
    analysis=Path(analysis_bundle).resolve();maps=Path(map_bundle).resolve();out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Choose a fresh collision movie directory')
    if type(fps) is not int or not 1<=fps<=120 or type(seconds_per_snapshot) is not int or not 1<=seconds_per_snapshot<=60:raise ValueError('Positive bounded movie timing required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite movie allocation required')
    report=verify_tree_bundle(analysis);treepath=analysis/'merger_tree.json';treehash=file_sha256(treepath);tree=json.loads(treepath.read_text());pair=select_merger_pair(tree,descendant_id)
    contractpath=maps/'map_contract.json';contracthash=file_sha256(contractpath);contract=json.loads(contractpath.read_text());root=Path(report['source_run'])
    if contract.get('schema')!='map-publication-v1' or report['data_contract']['run_namespace']!=contract['provenance']['run_namespace']:raise ValueError('Tree and map bundles must belong to the same run')
    verify_product_provenance(root,contract['provenance']);inventory={v['path']:v['sha256'] for v in contract['artifacts']}
    def checked(name):
        if not isinstance(name,str) or Path(name).name!=name or name not in inventory:raise ValueError('Unlisted map asset')
        p=maps/name
        if not p.is_file() or file_sha256(p)!=inventory[name]:raise ValueError('Map asset missing or changed: '+name)
        return p
    setting=json.loads(checked('scale.json').read_text())['maps'].get(map_key)
    if not setting or setting.get('status')!='COMPLETE' or setting.get('normalization')!='global_fixed':raise ValueError('Globally normalized map series required')
    metadata={}
    for name in inventory:
        if not re.fullmatch(re.escape(map_key)+r'_\d+\.figure\.json',name):continue
        m=json.loads(checked(name).read_text()).get('scientific_metadata',{})
        if m.get('overlay_layers')!=[]:continue
        checked(m['numeric_asset']);metadata[int(m['snapshot'])]=(name,m)
    box=float(tree['box_mpc_h']);axes_map={'xy':(0,1,2),'xz':(0,2,1),'yz':(1,2,0)};frames=[]
    for row in pair['rows']:
        if row['snapshot'] not in metadata:continue
        name,m=metadata[row['snapshot']]
        if not math.isclose(float(m['box_size_mpc_h']),box,rel_tol=1e-10) or not math.isclose(float(m['redshift']),row['redshift'],rel_tol=1e-8,abs_tol=1e-8):raise ValueError('Tree and map epochs or box differ')
        projection=m['projection'];ix,iy,iz=axes_map[projection];slab=m['slice'];thick=slab.get('thickness_mpc_h')
        if thick is not None and any(abs((p[iz]-slab['center_mpc_h']+box/2)%box-box/2)>thick/2 for p in (row['position_a_mpc_h'],row['position_b_mpc_h'])):continue
        frames.append({**row,'figure':name,'numeric_asset':m['numeric_asset'],'projection':projection,'colormap':m['density_colormap'],
                       'slice_geometry':{k:slab.get(k) for k in ('thickness_mpc_h','center_mpc_h','line_of_sight_axis','assignment_kernel')}})
    if len(frames)<2:raise ValueError('At least two pair epochs must share maps and include both halos in the slab')
    if len({(r['projection'],r['colormap']) for r in frames})!=1:raise ValueError('Map projection or color map changes across epochs')
    if any(r['slice_geometry']!=frames[0]['slice_geometry'] for r in frames):raise ValueError('Map slab geometry changes across epochs')
    ix,iy,_=axes_map[frames[0]['projection']]
    projected=[]
    for row in frames:
        d=(np.asarray(row['position_b_mpc_h'])-np.asarray(row['position_a_mpc_h'])+box/2)%box-box/2
        projected.append(np.hypot(d[ix],d[iy]))
    half_width=min(box*.49,max(box*.03,max(projected)*1.3))
    all_scale=np.array([r['scale_factor'] for r in pair['rows']]);all_sep=np.array([r['separation_comoving_mpc_h'] for r in pair['rows']])
    out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.halo-collision-',dir=out.parent));movie=stage/'halo_collision.mp4'
    try:
        with imageio.get_writer(movie,fps=fps,codec='libx264',quality=8,macro_block_size=16) as writer:
            for row in frames:
                with h5py.File(checked(row['numeric_asset']),'r') as h:
                    ds=h['log10_display']
                    if len(ds.shape)!=2 or np.prod(ds.shape,dtype=np.int64)*32>memory_gib*1024**3*.75:raise MemoryError('Collision map exceeds declared memory')
                    density=ds[:]
                pa=np.array(row['position_a_mpc_h']);pb=np.array(row['position_b_mpc_h']);d=(pb-pa+box/2)%box-box/2
                center=np.mod(pa+d/2,box);positions=[center-d/2,center+d/2]
                fig,(spatial,graph)=plt.subplots(1,2,figsize=(12.8,6.4),dpi=100,facecolor='#08111e',gridspec_kw={'width_ratios':[1.2,1]});fig.subplots_adjust(left=.06,right=.96,bottom=.13,top=.82,wspace=.27)
                cmap=matplotlib.colormaps[row['colormap']].with_extremes(bad=matplotlib.colormaps[row['colormap']](0))
                for sx in (-box,0,box):
                    for sy in (-box,0,box):spatial.imshow(np.ma.masked_invalid(density),origin='lower',extent=(sx,sx+box,sy,sy+box),cmap=cmap,vmin=setting['vmin'],vmax=setting['vmax'],interpolation='nearest')
                for p,c,label in zip(positions,('#61e7fc','#ffc16a'),('A','B')):spatial.scatter(p[ix],p[iy],s=120,edgecolors=c,facecolors='none',linewidths=2,label=label)
                for p,v,c in zip(positions,(row['velocity_a_km_s'],row['velocity_b_km_s']),('#61e7fc','#ffc16a')):
                    if v is not None:
                        v=np.asarray(v,float);spatial.arrow(p[ix],p[iy],v[ix]/100*.12*half_width,v[iy]/100*.12*half_width,color=c,width=.003*half_width,length_includes_head=True)
                spatial.set_xlim(center[ix]-half_width,center[ix]+half_width);spatial.set_ylim(center[iy]-half_width,center[iy]+half_width);spatial.grid(False);spatial.tick_params(colors='#c5d6e8',labelsize=8)
                spatial.set_xlabel(row['projection'][0]+' [unwrapped comoving Mpc/h]',color='#c5d6e8',fontsize=9);spatial.set_ylabel(row['projection'][1]+' [unwrapped comoving Mpc/h]',color='#c5d6e8',fontsize=9)
                spatial.set_title('Linked pair · 100 km/s arrow = %.2g Mpc/h on display'%(.12*half_width),color='white',fontsize=10);spatial.legend(loc='upper right',fontsize=8)
                graph.set_facecolor('#14253a');seen=all_scale<=row['scale_factor']+1e-10;graph.plot(all_scale[seen],all_sep[seen],'-o',color='#74dced');graph.set_xlim(float(all_scale.min())*.98,float(all_scale.max())*1.02);graph.set_ylim(0,max(float(all_sep.max())*1.15,1e-12))
                graph.set_xlabel('scale factor a',color='#c5d6e8');graph.set_ylabel('3D periodic separation [comoving Mpc/h]',color='#c5d6e8');graph.tick_params(colors='#c5d6e8');graph.grid(alpha=.15)
                speed=row['relative_peculiar_speed_km_s'];graph.set_title('Relative peculiar speed: '+('unavailable' if speed is None else '%.3g km/s'%speed),color='white',fontsize=11)
                fig.suptitle('Linked halo encounter · z=%.4g · snapshot %d'%(row['redshift'],row['snapshot']),color='white',fontsize=16,y=.95)
                fig.text(.5,.89,'Saved catalogue positions and velocities · lines connect saved epochs · not a collision-causality claim',color='#b8cadc',ha='center',fontsize=9)
                fig.canvas.draw();frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
                for _ in range(fps*seconds_per_snapshot):writer.append_data(frame)
                plt.close(fig)
        checked('scale.json')
        for row in frames:checked(row['figure']);checked(row['numeric_asset'])
        verify_tree_bundle(analysis);verify_product_provenance(root,contract['provenance'])
        if file_sha256(treepath)!=treehash or file_sha256(contractpath)!=contracthash:raise ValueError('Source tree or map contract changed')
        atomic_json(stage/'movie_contract.json',{'schema':'linked-halo-encounter-movie-v1','science_acceptance':'NOT_ACCEPTED','analysis_bundle':str(analysis),
            'tree_sha256':treehash,'map_bundle':str(maps),'map_contract_sha256':contracthash,'algorithm_sha256':file_sha256(__file__),
            'pair':pair,'frames':frames,'movie_sha256':file_sha256(movie),'fps':fps,'seconds_per_snapshot':seconds_per_snapshot,
            'selection_scope':'Main branches of two linked progenitors at matching saved epochs; no inferred trajectory or core identity.',
            'position_scope':'Minimum-image 3D comoving separation; display repeats periodic map solely for a wrapped close-up.',
            'velocity_scope':'Native catalogue peculiar km/s; no Hubble flow. Arrow lengths are display scaling, not travelled distances.',
            'timing_scope':'Equal display time for saved epochs; no interpolated states.'})
        publish_directory(stage,out);return out/'halo_collision.mp4'
    except BaseException as error:
        atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)})
        raise
