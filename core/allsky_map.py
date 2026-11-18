"""Equal-solid-angle sky maps of saved simulation matter around an observer."""
from __future__ import annotations
import math
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def angular_mass_grid(relative_positions, represented_mass, nlon, nlat):
    """Mass per steradian using uniform longitude and sin(latitude) bins."""
    p=np.asarray(relative_positions,dtype=float);w=np.asarray(represented_mass,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or w.shape!=(len(p),) or not np.isfinite(p).all() or not np.isfinite(w).all() or np.any(w<=0):raise ValueError('Finite aligned positions and positive represented masses required')
    if type(nlon) is not int or type(nlat) is not int or nlon<4 or nlat<2:raise ValueError('Valid angular resolution required')
    radius=np.linalg.norm(p,axis=1);visible=radius>0
    lon=np.arctan2(p[visible,1],p[visible,0]);sinlat=p[visible,2]/radius[visible]
    mass,lon_edges,sin_edges=np.histogram2d(lon,sinlat,bins=(nlon,nlat),range=((-np.pi,np.pi),(-1,1)),weights=w[visible])
    solid_angle=4*np.pi/(nlon*nlat)
    return {'mass_per_sr':mass/solid_angle,'longitude_edges_rad':lon_edges,'sin_latitude_edges':sin_edges,
            'excluded_observer_coincident_count':int(np.count_nonzero(~visible)),'solid_angle_sr_per_pixel':solid_angle}


def make_allsky_map(run_dir,destination,*,snapshot,field='dm',observer=None,nlon=360,nlat=180,max_particles=1000000,memory_gib=1.):
    from core.visualization import load_particles_for_fisheye
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,verify_product_provenance,publish_directory
    root=Path(run_dir).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh all-sky output directory')
    if type(snapshot) is not int or snapshot<0 or field not in ('dm','gas'):raise ValueError('Select a saved snapshot and dm or gas field')
    if type(nlon) is not int or type(nlat) is not int or not 16<=nlon<=2048 or not 8<=nlat<=1024:raise ValueError('Angular grid outside supported bounds')
    if type(max_particles) is not int or not 1<=max_particles<=4000000:raise ValueError('Positive bounded particle sample required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite memory allocation required')
    if max_particles*256+nlon*nlat*128+64*1024**2>memory_gib*1024**3*.75:raise MemoryError('All-sky sample and raster exceed declared memory')
    with native_run_lock(root,operation='all-sky map'):
        coords,weights,box,z,source,arrays=load_particles_for_fisheye(root,snapshot,field,max_particles,observer_mpc_h=observer,return_arrays=True)
        result=angular_mass_grid(coords,weights,nlon,nlat);mass=result['mass_per_sr']
        positive=mass[np.isfinite(mass)&(mass>0)]
        if not len(positive):raise ValueError('No nonzero angular mass is available')
        low,high=np.percentile(np.log10(positive),[1,99.9])
        if high<=low:high=np.nextafter(low,np.inf)
        with np.errstate(divide='ignore'):image=np.ma.masked_where(mass.T<=0,np.log10(mass.T))
        fig=plt.figure(figsize=(13,7),facecolor='#08111e');ax=fig.add_subplot(111,projection='mollweide',facecolor='#10243a')
        fig.subplots_adjust(left=.06,right=.88,bottom=.14,top=.82)
        cmap=matplotlib.colormaps['magma'].with_extremes(bad='#10243a')
        plot=ax.pcolormesh(result['longitude_edges_rad'],np.arcsin(result['sin_latitude_edges']),image,cmap=cmap,vmin=low,vmax=high,shading='flat',rasterized=True)
        ax.grid(color='#9fb4c9',alpha=.2,linewidth=.6);ax.tick_params(colors='#c5d6e8',labelsize=8)
        cb=fig.colorbar(plot,ax=ax,orientation='horizontal',fraction=.06,pad=.1);cb.set_label('log10 represented mass per steradian [Msun/h/sr]',color='#c5d6e8');cb.ax.tick_params(colors='#c5d6e8')
        fig.suptitle('%s · simulated all-sky matter map · z=%.4g'%('Dark matter' if field=='dm' else 'Gas',z),color='white',fontsize=19,y=.95)
        fig.text(.5,.89,'One nearest periodic copy · observer %s Mpc/h · sampled mass, not emitted light'%(', '.join('%.3g'%v for v in source['observer_mpc_h'])),ha='center',color='#b8cadc',fontsize=10)
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.allsky-',dir=out.parent))
        try:
            fig.savefig(stage/'allsky.png',dpi=180,facecolor=fig.get_facecolor());fig.savefig(stage/'allsky.pdf',facecolor=fig.get_facecolor())
            np.savez_compressed(stage/'allsky_data.npz',**arrays,**result)
            verify_product_provenance(root,source['data_contract'])
            atomic_json(stage/'allsky_contract.json',{'schema':'simulation-allsky-mass-v1','science_acceptance':'NOT_ACCEPTED',
                'source':source,'algorithm_sha256':file_sha256(__file__),'nlon':nlon,'nlat':nlat,
                'scalar':'inverse-inclusion-weighted mass per equal solid angle [Msun/h/sr]; not emitted light or an observed flux',
                'excluded_observer_coincident_count':result['excluded_observer_coincident_count'],
                'display_log10_limits':[float(low),float(high)],'solid_angle_sr_per_pixel':result['solid_angle_sr_per_pixel'],
                'artifacts':{name:file_sha256(stage/name) for name in ('allsky.png','allsky.pdf','allsky_data.npz')}})
            publish_directory(stage,out)
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)})
            raise
        finally:plt.close(fig)
    return out/'allsky.png'
