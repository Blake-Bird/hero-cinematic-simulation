"""Fixed-scale planetarium movie from saved, source-bound dome rasters."""
from __future__ import annotations
import math
import tempfile
from pathlib import Path
import imageio.v2 as imageio
import matplotlib
import numpy as np


def make_dome_movie(run_dir,destination,*,field='dm',snapshots=None,observer=None,resolution=512,tilt=0.,
                    max_particles=1000000,fps=24,seconds_per_snapshot=1,memory_gib=1.):
    from core.visualization import load_particles_for_fisheye,_angular_raster
    from core.analysis import snapshot_ids
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,verify_product_provenance,publish_directory
    root=Path(run_dir).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh planetarium movie directory')
    if field not in ('dm','gas','gas_temperature'):raise ValueError('Supported dome field required')
    if type(resolution) is not int or not 16<=resolution<=2048 or resolution%16:raise ValueError('Dome resolution must be a multiple of 16 from 16 to 2048')
    if type(max_particles) is not int or not 1<=max_particles<=4000000 or type(fps) is not int or not 1<=fps<=120 or type(seconds_per_snapshot) is not int or not 1<=seconds_per_snapshot<=60:raise ValueError('Bounded particle and movie controls required')
    if isinstance(tilt,bool) or not isinstance(tilt,(int,float)) or not math.isfinite(tilt):raise ValueError('Finite dome tilt required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite memory allocation required')
    if max_particles*256+resolution**2*100+64*1024**2>memory_gib*1024**3*.7:raise MemoryError('Dome sample and raster exceed declared memory')
    with native_run_lock(root,operation='planetarium movie'):
        available=snapshot_ids(root);wanted=available if snapshots is None else snapshots
        if not isinstance(wanted,list) or len(wanted)<2 or any(type(s) is not int or s not in available for s in wanted) or len(set(wanted))!=len(wanted):raise ValueError('Select at least two distinct existing saved snapshots')
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.dome-movie-',dir=out.parent));movie=stage/'planetarium.mp4'
        try:
            records=[];lows=[];highs=[];initial_box=None;temp_source=None
            yy,xx=np.ogrid[:resolution,:resolution];circle=(xx+.5-resolution/2)**2+(yy+.5-resolution/2)**2<=(resolution/2)**2
            for snap in wanted:
                coords,weights,box,z,source,inputs=load_particles_for_fisheye(root,snap,field,max_particles,observer_mpc_h=observer,return_arrays=True)
                if initial_box is None:initial_box=box
                if not math.isclose(box,initial_box,rel_tol=1e-10):raise ValueError('Saved dome snapshots have different box sizes')
                if field=='gas_temperature':
                    if temp_source is None:temp_source=source['temperature_source']
                    elif temp_source!=source['temperature_source']:raise ValueError('Temperature source changes between saved epochs')
                raster,visible=_angular_raster(coords,weights,resolution,tilt,field=='gas_temperature')
                occupied=circle&np.isfinite(raster)&(raster>0)
                if not np.any(occupied):raise ValueError('No particles project onto dome at snapshot %d'%snap)
                limits=np.percentile(np.log10(raster[occupied]),[1,99.9]);lows.append(float(limits[0]));highs.append(float(limits[1]))
                name='snapshot_%03d.npz'%snap;np.savez_compressed(stage/name,**inputs,angular_scalar_raster=raster,visible_sample=visible)
                records.append({'snapshot':snap,'scale_factor':float(source['epoch']['actual_scale_factor']),'redshift':float(z),
                    'numeric_asset':name,'numeric_sha256':file_sha256(stage/name),'source':source,'visible_sample_count':int(np.count_nonzero(visible))})
            records.sort(key=lambda r:r['scale_factor'])
            if any(records[i]['scale_factor']>=records[i+1]['scale_factor'] for i in range(len(records)-1)):raise ValueError('Dome epochs must be distinct and ordered')
            low=min(lows);high=max(highs)
            if high<=low:high=np.nextafter(low,np.inf)
            palette=matplotlib.colormaps['inferno' if field=='gas_temperature' else 'magma']
            with imageio.get_writer(movie,fps=fps,codec='libx264',quality=8,macro_block_size=16) as writer:
                for record in records:
                    with np.load(stage/record['numeric_asset'],allow_pickle=False) as saved:raster=saved['angular_scalar_raster']
                    occupied=circle&np.isfinite(raster)&(raster>0);normal=np.zeros_like(raster)
                    normal[occupied]=np.clip((np.log10(raster[occupied])-low)/(high-low),0,1)
                    rgb=palette(normal,bytes=True)[...,:3];rgb[~occupied]=0
                    for _ in range(fps*seconds_per_snapshot):writer.append_data(rgb)
            for record in records:
                verify_product_provenance(root,record['source']['data_contract'])
                if file_sha256(stage/record['numeric_asset'])!=record['numeric_sha256']:raise ValueError('Saved dome numeric data changed')
            atomic_json(stage/'planetarium_contract.json',{'schema':'planetarium-movie-v1','science_acceptance':'NOT_ACCEPTED',
                'run':str(root),'field':field,'epochs':records,'fps':fps,'seconds_per_snapshot':seconds_per_snapshot,
                'resolution':resolution,'tilt_degrees':tilt,'max_particles':max_particles,'algorithm_sha256':file_sha256(__file__),
                'projection':'180-degree equidistant fisheye; black outside the dome',
                'scalar':'mass-weighted gas temperature [K]' if field=='gas_temperature' else 'angular mass density [Msun/h/sr]',
                'display':'log10 with fixed global min/max of per-snapshot 1st/99.9th percentiles',
                'display_log10_limits':[low,high],'movie_sha256':file_sha256(movie),
                'sampling_scope':'Each saved snapshot is sampled independently; shared display scaling does not imply particle identity tracking.',
                'timing_scope':'Equal display time per saved epoch; no interpolated states or uniform physical-time claim.',
                'interpretation':'Simulation mass/temperature, not emitted light, a light cone, lensing or projector warp.'})
            publish_directory(stage,out);return out/'planetarium.mp4'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)})
            raise
