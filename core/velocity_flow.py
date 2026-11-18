"""Saved dark-matter surface density and mass-weighted peculiar-velocity flow."""
from __future__ import annotations
import math
import tempfile
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def flow_bins(position,velocity,mass,box,grid,axes,*,slice_axis=None,slice_center=None,slice_thickness=None):
    """Accumulate unsmoothed projected mass, particle count and mass-weighted velocity."""
    p=np.asarray(position,float);v=np.asarray(velocity,float);m=np.asarray(mass,float)
    if p.ndim!=2 or p.shape[1]!=3 or v.shape!=p.shape or m.shape!=(len(p),):raise ValueError('Particle positions, velocities and masses must align')
    if not np.isfinite(p).all() or not np.isfinite(v).all() or not np.isfinite(m).all() or np.any(m<=0):raise ValueError('Flow input contains invalid positions, velocities or masses')
    keep=np.ones(len(p),dtype=bool)
    if slice_thickness is not None:
        distance=(p[:,slice_axis]-slice_center+box/2)%box-box/2
        keep=np.abs(distance)<=slice_thickness/2
    xy=np.mod(p[keep][:,axes],box);vel=v[keep][:,axes];weight=m[keep]
    limits=((0,box),(0,box))
    hist=lambda w:np.histogram2d(xy[:,0],xy[:,1],bins=grid,range=limits,weights=w)[0]
    return hist(weight),hist(None),hist(weight*vel[:,0]),hist(weight*vel[:,1])


def make_velocity_flow(run_dir,destination,*,snapshot,projection='xy',grid=32,min_particles=3,
                       slice_center=None,slice_thickness=None,memory_gib=1.):
    from core.analysis import read_snapshot_header
    from core.visualization import _velocity_conversion_km_s
    from core.data import validate_native_shard_counts,coordinate_contract
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,product_provenance,verify_product_provenance,publish_directory,MSUN_G
    from halos import _unit_conversion
    root=Path(run_dir).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh velocity-flow directory')
    if type(snapshot) is not int or snapshot<0 or projection not in ('xy','xz','yz') or type(grid) is not int or not 8<=grid<=256 or type(min_particles) is not int or min_particles<1:raise ValueError('Valid snapshot, projection, grid 8–256 and positive support count required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<.25:raise ValueError('At least 0.25 GiB declared memory required')
    if (slice_center is None)!=(slice_thickness is None):raise ValueError('Supply both slab center and thickness, or neither')
    with native_run_lock(root,operation='velocity flow map'):
        header,files=read_snapshot_header(root,snapshot);validate_native_shard_counts(files,'NumPart_Total');_unit_conversion(files)
        box=float(header['BoxSize'])/1000.;a=float(header['Time'])
        if not math.isfinite(box) or box<=0 or not math.isfinite(a) or a<=0:raise ValueError('Valid native box and scale factor required')
        if slice_center is not None and (not math.isfinite(slice_center) or not 0<=slice_center<box or not math.isfinite(slice_thickness) or not 0<slice_thickness<=box):raise ValueError('Slab center/thickness must fit the periodic box')
        factor=_velocity_conversion_km_s(files,1,header)
        if factor is None:raise ValueError('Velocity unit metadata are required for peculiar km/s')
        axes={'xy':(0,1),'xz':(0,2),'yz':(1,2)}[projection];los=({0,1,2}-set(axes)).pop()
        contract=product_provenance(root,algorithm=__file__,helper_sources=[Path(__file__).with_name('visualization.py'),Path(__file__).with_name('analysis.py')],
            source_files=files,selection={'snapshot':snapshot,'projection':projection,'grid':grid,'minimum_particles':min_particles,'slice_center':slice_center,'slice_thickness':slice_thickness},
            coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='global simulation box',periodic_wrapping='periodic full box or specified slab',velocity_convention='mass-weighted peculiar km/s; no Hubble flow'))
        sums=[np.zeros((grid,grid),float) for _ in range(4)]
        for path in files:
            with h5py.File(path,'r') as h:
                hh=h['Header'].attrs
                if not math.isclose(float(hh['Time']),a,rel_tol=1e-12) or not math.isclose(float(hh['BoxSize'])/1000.,box,rel_tol=1e-12):raise ValueError('Snapshot shards disagree on epoch or box')
                if 'PartType1' not in h:continue
                p=h['PartType1'];xyz=p['Coordinates'];vel=p['Velocities'];mass=p.get('Masses');n=len(xyz)
                if xyz.shape!=(n,3) or vel.shape!=(n,3) or (mass is not None and mass.shape!=(n,)):raise ValueError('DM fields are not aligned')
                fixed=float(hh['MassTable'][1]);mass_unit=float(h['Parameters'].attrs['UnitMass_in_g'])/MSUN_G
                if mass is None and (not math.isfinite(fixed) or fixed<=0):raise ValueError('DM particle masses are unavailable')
                for start in range(0,n,200000):
                    stop=min(start+200000,n);p0=np.asarray(xyz[start:stop],float)/1000.;v0=np.asarray(vel[start:stop],float)*factor
                    m0=(np.asarray(mass[start:stop],float) if mass is not None else np.full(stop-start,fixed))*mass_unit
                    pieces=flow_bins(p0,v0,m0,box,grid,axes,slice_axis=los,slice_center=slice_center,slice_thickness=slice_thickness)
                    for target,value in zip(sums,pieces):target+=value
        mass,count,mx,my=sums;support=(count>=min_particles)&(mass>0)
        if not np.any(support):raise ValueError('No flow cells meet the requested particle support')
        vx=np.divide(mx,mass,out=np.full_like(mass,np.nan),where=support);vy=np.divide(my,mass,out=np.full_like(mass,np.nan),where=support)
        speed=np.hypot(vx[support],vy[support]);reference=float(np.median(speed))
        if reference<=0:reference=1.
        fig,ax=plt.subplots(figsize=(8.5,8),facecolor='#08111e');ax.set_facecolor('#0e2036')
        with np.errstate(divide='ignore'):density=np.ma.masked_where(mass.T<=0,np.log10(mass.T/(box/grid)**2))
        im=ax.imshow(density,origin='lower',extent=(0,box,0,box),cmap='magma',interpolation='nearest');ax.grid(False)
        centers=(np.arange(grid)+.5)*box/grid;X,Y=np.meshgrid(centers,centers,indexing='ij')
        arrows=ax.quiver(X[support],Y[support],vx[support],vy[support],angles='xy',scale_units='xy',scale=reference/(.045*box),color='#8ff6ff',width=.0025)
        ax.quiverkey(arrows,.78,1.035,reference,'%.3g km/s'%reference,labelpos='E',color='white',labelcolor='white',coordinates='axes')
        ax.set(xlim=(0,box),ylim=(0,box),xlabel=projection[0]+' [comoving Mpc/h]',ylabel=projection[1]+' [comoving Mpc/h]')
        ax.tick_params(colors='#c5d6e8');ax.xaxis.label.set_color('#c5d6e8');ax.yaxis.label.set_color('#c5d6e8')
        slab='full box' if slice_thickness is None else 'slab %.3g Mpc/h centered %.3g'%(slice_thickness,slice_center)
        ax.set_title('Dark-matter peculiar flow · z=%.4g · %s'%(1/a-1,slab),color='white',pad=24)
        cb=fig.colorbar(im,ax=ax,pad=.025);cb.set_label('log10 surface density [h Msun/Mpc²]',color='#c5d6e8');cb.ax.tick_params(colors='#c5d6e8')
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.velocity-flow-',dir=out.parent))
        try:
            fig.savefig(stage/'velocity_flow.png',dpi=180,facecolor=fig.get_facecolor(),bbox_inches='tight')
            fig.savefig(stage/'velocity_flow.pdf',facecolor=fig.get_facecolor(),bbox_inches='tight')
            np.savez_compressed(stage/'flow_data.npz',mass_msun_h=mass,particle_count=count,velocity_x_km_s=vx,velocity_y_km_s=vy,supported=support,cell_centers_mpc_h=centers)
            verify_product_provenance(root,contract)
            atomic_json(stage/'flow_contract.json',{'schema':'projected-velocity-flow-v1','science_acceptance':'NOT_ACCEPTED','provenance':contract,
                'redshift':1/a-1,'arrow_reference_km_s':reference,'supported_cells':int(support.sum()),
                'definition':'Unsmoothed cell mean of peculiar velocity weighted by particle mass; empty or low-count cells excluded. No Hubble flow.',
                'artifacts':{name:file_sha256(stage/name) for name in ('velocity_flow.png','velocity_flow.pdf','flow_data.npz')}})
            publish_directory(stage,out)
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)})
            raise
        finally:plt.close(fig)
    return out/'velocity_flow.png'
