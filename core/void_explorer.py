"""Explore measured underdensities in periodic saved dark-matter snapshots."""
from __future__ import annotations
import html
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
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, minimum_filter


def density_minima(field, threshold):
    """Find thresholded periodic 26-neighbor minima, including plateau nodes."""
    f=np.asarray(field,float)
    if f.ndim!=3 or len(set(f.shape))!=1 or f.shape[0]<8 or not np.isfinite(f).all() or not math.isfinite(threshold):raise ValueError('Finite cubic density field and threshold required')
    candidates=(f==minimum_filter(f,size=3,mode='wrap'))&(f<threshold)
    # Preserve reproducible candidates; a flat field does not define distinct void centers.
    if np.ptp(f)<=np.finfo(float).eps*max(1.,float(np.max(abs(f)))):return np.empty((0,3),dtype=int)
    return np.argwhere(candidates)


def nearest_minimum(field, candidates, previous, max_distance_cells):
    """Find the nearest candidate within a periodic gate; density breaks exact ties."""
    n=field.shape[0];candidates=np.asarray(candidates,int);previous=np.asarray(previous,int)
    if candidates.ndim!=2 or candidates.shape[1]!=3 or previous.shape!=(3,) or max_distance_cells<=0:raise ValueError('Valid minimum candidates and matching radius required')
    if not len(candidates):return None
    delta=(candidates-previous+n/2)%n-n/2;distance=np.linalg.norm(delta,axis=1);valid=np.flatnonzero(distance<=max_distance_cells)
    if not len(valid):return None
    order=np.lexsort((candidates[valid,2],candidates[valid,1],candidates[valid,0],field[tuple(candidates[valid].T)],distance[valid]));return candidates[valid[order[0]]]


def enclosed_density(field,center,box,threshold=.8):
    """Measure outermost threshold-supported spherical aperture on periodic mesh nodes."""
    f=np.asarray(field,float);n=f.shape[0];center=np.asarray(center,int)
    if f.ndim!=3 or f.shape!=(n,n,n) or center.shape!=(3,) or not np.isfinite(f).all() or np.any(f<0) or not math.isfinite(box) or box<=0 or not 0<threshold<1:raise ValueError('Valid density, center, box and underdensity threshold required')
    offsets=(np.arange(n)-center[:,None]+n/2)%n-n/2
    d2=offsets[0,:,None,None]**2+offsets[1,None,:,None]**2+offsets[2,None,None,:]**2
    # Integer-cell shells avoid pretending that the gridded boundary is exact.
    edges=np.arange(1,n//2,dtype=float);radii=edges*box/n
    means=np.asarray([np.mean(f[d2<=edge**2]) for edge in edges]);supported=np.flatnonzero(means<threshold)
    if not len(supported):return {'radius_mpc_h':None,'status':'UNRESOLVED','radii':radii,'means':means}
    last=int(supported[-1]);return {'radius_mpc_h':float(radii[last]),'status':'LOWER_BOUND' if last==len(radii)-1 else 'MEASURED_GRID_APERTURE','radii':radii,'means':means}


def make_void_explorer(run_dir,destination,*,grid=48,smoothing_cells=1.5,threshold=.8,match_fraction=.125,frames=32,fps=16,max_particles=100000,memory_gib=1.):
    from core.analysis import snapshot_ids,read_snapshot_header
    from core.data import validate_native_shard_counts,coordinate_contract
    from core.visualization import _particle_mass_density_grid,load_particles_for_fisheye
    from core.cosmic_flythrough import camera_samples,unwrap_camera_path,perspective_mass_raster
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,product_provenance,verify_product_provenance,publish_directory
    root=Path(run_dir).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh void explorer output directory')
    if type(grid) is not int or not 16<=grid<=128 or not 0<smoothing_cells<=8 or not 0<threshold<1 or not 0<match_fraction<=.25:raise ValueError('Grid 16–128, smoothing 0–8 cells, threshold 0–1 and match fraction 0–0.25 required')
    if type(frames) is not int or not 2<=frames<=1000 or type(fps) is not int or not 1<=fps<=120 or type(max_particles) is not int or not 1<=max_particles<=1000000:raise ValueError('Invalid movie controls')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(float,int)) or not math.isfinite(memory_gib) or memory_gib<=0 or grid**3*200+max_particles*350+128*1024**2>memory_gib*1024**3*.7:raise MemoryError('Void explorer exceeds declared memory')
    with native_run_lock(root,operation='void explorer'):
        snapshots=snapshot_ids(root)
        if len(snapshots)<2:raise ValueError('Void explorer requires at least two saved epochs')
        epochs=[];box=None
        for snap in snapshots:
            header,files=read_snapshot_header(root,snap);validate_native_shard_counts(files,'NumPart_Total')
            a=float(header['Time']);b=float(header['BoxSize'])/1000.
            if not math.isfinite(a) or a<=0 or not math.isfinite(b) or b<=0 or (box is not None and not math.isclose(box,b,rel_tol=1e-12)):raise ValueError('Saved epochs require a consistent periodic box')
            box=b
            for path in files:
                with h5py.File(path) as f:
                    if not math.isclose(float(f['Header'].attrs['Time']),a,rel_tol=1e-12) or not math.isclose(float(f['Header'].attrs['BoxSize'])/1000.,b,rel_tol=1e-12):raise ValueError('Snapshot shards disagree on frame')
            provenance=product_provenance(root,algorithm=__file__,helper_sources=[Path(__file__).with_name('analysis.py'),Path(__file__).with_name('visualization.py'),Path(__file__).with_name('cosmic_flythrough.py')],source_files=files,selection={'snapshot':snap,'grid':grid,'smoothing_cells':smoothing_cells,'threshold':threshold},coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='periodic simulation box',periodic_wrapping='periodic CIC',velocity_convention='not used'))
            epochs.append({'snapshot':int(snap),'scale_factor':a,'redshift':1/a-1,'files':files,'provenance':provenance})
        epochs.sort(key=lambda e:e['scale_factor'])
        if any(x['scale_factor']>=y['scale_factor'] for x,y in zip(epochs,epochs[1:])):raise ValueError('Saved epochs must have distinct scale factors')
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.void-explorer-',dir=out.parent))
        try:
            # Write each volume before tracking so memory stays bounded by one mesh.
            for e in epochs:
                density,metadata=_particle_mass_density_grid(e['files'],1,box,grid,return_density_ratio=True)
                smooth=gaussian_filter(density,smoothing_cells,mode='wrap');e['density_metadata']=metadata;e['volume']='density_%03d.npy'%e['snapshot'];np.save(stage/e['volume'],smooth,allow_pickle=False)
            previous=None
            for e in reversed(epochs):
                field=np.load(stage/e['volume'],allow_pickle=False);minima=density_minima(field,threshold)
                if previous is None:
                    if e is not epochs[-1] or len(minima)==0:e.update({'status':'UNMATCHED','reason':'No underdense minimum linked to the later selected candidate'});continue
                    index=np.argmin(field[tuple(minima.T)]);center=minima[index]
                else:center=nearest_minimum(field,minima,previous,grid*match_fraction)
                if center is None:e.update({'status':'UNMATCHED','reason':'No underdense local minimum within the matching distance'});previous=None;continue
                previous=center;profile=enclosed_density(field,center,box,threshold)
                e.update({'status':'MATCHED','center_grid':center.tolist(),'center_mpc_h':(center*box/grid).tolist(),'minimum_density_ratio':float(field[tuple(center)]),'radius_mpc_h':profile['radius_mpc_h'],'radius_status':profile['status'],'profile_radii_mpc_h':profile['radii'].tolist(),'profile_enclosed_density_ratio':profile['means'].tolist()})
            if epochs[-1]['status']!='MATCHED':raise ValueError('No underdense local minimum found in the latest snapshot; adjust threshold or smoothing')
            # Shared scale for all epoch slices; center-centered periodic view.
            lows=[];highs=[]
            for e in epochs:
                f=np.load(stage/e['volume'],allow_pickle=False);v=np.log10(np.maximum(f,1e-6));lows.append(float(np.percentile(v,1)));highs.append(float(np.percentile(v,99.5)))
            low=min(lows);high=max(highs)
            if high<=low:high=low+1
            palette=matplotlib.colormaps['magma']
            for e in epochs:
                if e['status']!='MATCHED':continue
                f=np.load(stage/e['volume'],allow_pickle=False);center=np.array(e['center_grid']);plane=np.roll(f,shift=(grid//2-center[0],grid//2-center[1]),axis=(0,1))[:,:,center[2]].T
                norm=np.clip((np.log10(np.maximum(plane,1e-6))-low)/(high-low),0,1);Image.fromarray(np.flipud(palette(norm,bytes=True)[...,:3])).resize((768,768),Image.Resampling.NEAREST).save(stage/('slice_%03d.png'%e['snapshot']))
            fig,ax=plt.subplots(figsize=(7,4),facecolor='#0b1828');ax.set_facecolor('#152b42');valid=[e for e in epochs if e['status']=='MATCHED' and e['radius_mpc_h'] is not None]
            if valid:ax.plot([e['scale_factor'] for e in epochs],[e.get('radius_mpc_h') if e.get('radius_mpc_h') is not None else np.nan for e in epochs],'-o',color='#7ae8f1')
            for e in valid:
                if e['radius_status']=='LOWER_BOUND':ax.annotate('≥',xy=(e['scale_factor'],e['radius_mpc_h']),color='white')
            ax.set(xlabel='Scale factor',ylabel='Underdense aperture [comoving Mpc/h]');ax.tick_params(colors='#d3e5f4');ax.xaxis.label.set_color('#d3e5f4');ax.yaxis.label.set_color('#d3e5f4');ax.grid(alpha=.15);fig.tight_layout();fig.savefig(stage/'void_evolution.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)
            latest=epochs[-1];target=np.array(latest['center_mpc_h']);start=np.mod(target+np.array([box*.24,0,0]),box);path=unwrap_camera_path([start,target],box);camera,directions=camera_samples(path,frames)
            coords,weights,sample_box,z,source,arrays=load_particles_for_fisheye(root,latest['snapshot'],'dm',max_particles,observer_mpc_h=[0,0,0],return_arrays=True)
            if not math.isclose(sample_box,box,rel_tol=1e-10):raise ValueError('Movie sample box differs')
            particles=np.mod(coords,box);records=[];lows=[];highs=[];width,height=768,432
            for i,(point,direction) in enumerate(zip(camera,directions)):
                raster,count=perspective_mass_raster(particles,weights,point,direction,box,width,height,60.)
                positive=raster[raster>0]
                if len(positive):v=np.log10(positive);lows.append(float(np.percentile(v,1)));highs.append(float(np.percentile(v,99.9)))
                name='camera_%04d.npy'%i;np.save(stage/name,raster,allow_pickle=False);records.append({'file':name,'visible_particles':count,'sha256':file_sha256(stage/name)})
            if not lows:raise ValueError('No particles visible on selected void camera path')
            movie_low=min(lows);movie_high=max(highs)
            if movie_high<=movie_low:movie_high=movie_low+1
            font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),14)
            with imageio.get_writer(stage/'void_flythrough.mp4',fps=fps,codec='libx264',quality=8,macro_block_size=16) as writer:
                for row in records:
                    raster=np.load(stage/row['file'],allow_pickle=False);positive=raster>0;normal=np.zeros_like(raster);normal[positive]=np.clip((np.log10(raster[positive])-movie_low)/(movie_high-movie_low),0,1)
                    rgb=palette(normal,bytes=True)[...,:3];rgb[~positive]=0;image=Image.fromarray(np.flipud(rgb).copy());draw=ImageDraw.Draw(image);draw.rectangle((0,0,width,27),fill=(5,12,22));draw.text((8,5),'Saved snapshot %d | z=%.4g | camera approaches density minimum'%(latest['snapshot'],z),font=font,fill=(220,239,250));writer.append_data(np.asarray(image))
            np.savez_compressed(stage/'movie_sample.npz',**arrays,camera_positions_unwrapped_mpc_h=camera,camera_directions=directions)
            for e in epochs:verify_product_provenance(root,e['provenance'])
            verify_product_provenance(root,source['data_contract'])
            public=[]
            for e in epochs:
                public.append({k:v for k,v in e.items() if k not in ('files','provenance')})
            payload=json.dumps(public,separators=(',',':')).replace('<','\\u003c');title=html.escape(root.name)
            page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cosmic void explorer</title><style>body{background:#081321;color:#e8f3ff;font:16px system-ui;margin:0}main{max-width:1400px;margin:auto;padding:28px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:30px}img,video{width:100%%}input{width:100%%}p{color:#bad0e1}.placeholder{aspect-ratio:1;background:#15283a;display:grid;place-items:center}[hidden]{display:none!important}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><main><h1>Cosmic void explorer</h1><p>%s · 3-D dark-matter density · periodic box %.4g comoving Mpc/h</p><div class="grid"><section><h2>Saved epochs</h2><img id="map" alt="Slice through the matched underdensity"><div id="missing" class="placeholder" hidden>No candidate could be linked at this epoch.</div><input id="epoch" type="range" min="0" max="%d" value="%d"><p id="info"></p><p>Centered XY slice through the 3-D density minimum; fixed color range log10(ρ/mean) %.3g to %.3g. White space in the history means a match was not established.</p></section><section><h2>Estimated underdense aperture</h2><img src="void_evolution.png" alt="Underdense aperture versus scale factor"><p>Radius is the outermost grid sphere with mean smoothed density below %.3g times the box mean, limited to less than half the box. A ≥ marker indicates the limit was reached. It is a gridded spherical proxy, not a watershed boundary.</p><h2>Fly into the latest minimum</h2><video controls preload="metadata" src="void_flythrough.mp4"></video><p>The camera moves through one saved snapshot; it does not depict evolution. Density-minimum matching between epochs is a proximity heuristic, not proof of void identity or filament formation.</p></section></div></main><script>const rows=%s,s=document.querySelector('#epoch'),im=document.querySelector('#map'),miss=document.querySelector('#missing'),info=document.querySelector('#info');function draw(){let r=rows[+s.value];im.hidden=r.status!=='MATCHED';miss.hidden=r.status==='MATCHED';if(!im.hidden)im.src='slice_'+String(r.snapshot).padStart(3,'0')+'.png';info.textContent=`Snapshot ${r.snapshot} · z=${r.redshift.toPrecision(4)} · ${r.status==='MATCHED'?`minimum ρ/mean=${r.minimum_density_ratio.toPrecision(4)} · radius ${r.radius_mpc_h===null?'unresolved':r.radius_mpc_h.toPrecision(4)+' Mpc/h ('+r.radius_status+')'}`:r.reason}`}s.addEventListener('input',draw);draw()</script></html>'''%(title,box,len(epochs)-1,len(epochs)-1,low,high,threshold,payload)
            (stage/'void_explorer.html').write_text(page,encoding='utf-8')
            atomic_json(stage/'void_contract.json',{'schema':'void-explorer-v1','science_acceptance':'NOT_ACCEPTED','run':str(root),'box_mpc_h':box,'grid':grid,'smoothing_cells':smoothing_cells,'threshold':threshold,'match_fraction_of_box':match_fraction,'epochs':public,'source_provenance':[e['provenance'] for e in epochs],'movie_source':source,'movie_frames':records,'movie_color_log10_limits':[movie_low,movie_high],'movie_fps':fps,'algorithm_sha256':file_sha256(__file__),'interpretation':'Periodic nearest-minimum linking is a heuristic, not validated identity; spherical threshold radius is a gridded proxy, not a watershed void. Fly-through shows one saved epoch only.','artifacts':{p.name:file_sha256(p) for p in stage.iterdir() if p.is_file()}})
            publish_directory(stage,out);return out/'void_explorer.html'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
