"""Cinematic path along a periodic Hessian density-ridge proxy to a halo."""
from __future__ import annotations
import heapq
import itertools
import math
import tempfile
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter


def hessian_ridge_mask(density,*,sigma,threshold_factor,density_percentile):
    """Classify two-axis compressive log-density curvature on a periodic mesh."""
    f=np.asarray(density,float)
    if f.ndim!=3 or len(set(f.shape))!=1 or f.shape[0]<8 or not np.isfinite(f).all() or np.any(f<0) or not math.isfinite(sigma) or sigma<=0 or not math.isfinite(threshold_factor) or threshold_factor<=0 or not 0<=density_percentile<=100:raise ValueError('Valid periodic density, smoothing and threshold required')
    n=f.shape[0];log=np.log1p(f);h=np.empty((n,n,n,3,3),dtype=np.float32)
    for i in range(3):
        order=[0,0,0];order[i]=2;h[...,i,i]=gaussian_filter(log,sigma=sigma,order=order,mode='wrap')
        for j in range(i+1,3):
            order=[0,0,0];order[i]=order[j]=1;h[...,i,j]=h[...,j,i]=gaussian_filter(log,sigma=sigma,order=order,mode='wrap')
    eigen=np.linalg.eigvalsh(h);threshold=max(float(np.percentile(np.abs(eigen),65))*threshold_factor,np.finfo(float).eps)
    positive=f[f>0]
    if not len(positive):raise ValueError('No positive density for ridge classification')
    density_floor=float(np.percentile(positive,density_percentile));mask=(eigen[...,1]<-threshold)&(eigen[...,2]>=-threshold)&(f>=density_floor)
    return mask,{'curvature_threshold':threshold,'density_floor_ratio':density_floor,'ridge_cell_count':int(mask.sum())}


def ridge_route(mask,host_mpc_h,box,*,max_host_distance_mpc_h,path_length_mpc_h):
    """Select a connected, periodic ridge-voxel route ending nearest the host."""
    mask=np.asarray(mask,bool);n=mask.shape[0];host=np.asarray(host_mpc_h,float)
    if mask.ndim!=3 or mask.shape!=(n,n,n) or host.shape!=(3,) or not np.isfinite(host).all() or not math.isfinite(box) or box<=0 or not 0<max_host_distance_mpc_h<=box/2 or not 0<path_length_mpc_h<=box:raise ValueError('Valid ridge mesh, host and route limits required')
    coords=np.argwhere(mask)
    if not len(coords):raise ValueError('No resolved ridge cells')
    pos=coords*box/n;delta=(pos-host+box/2)%box-box/2;dist=np.linalg.norm(delta,axis=1);nearest=int(np.argmin(dist))
    if dist[nearest]>max_host_distance_mpc_h:raise ValueError('No ridge cell lies within the permitted host distance')
    seed=tuple(int(v) for v in coords[nearest]);neighbors=[(offset,math.sqrt(sum(v*v for v in offset))*box/n) for offset in itertools.product((-1,0,1),repeat=3) if offset!=(0,0,0)]
    distances={seed:0.};parents={};queue=[(0.,seed)]
    while queue:
        length,cell=heapq.heappop(queue)
        if length!=distances[cell]:continue
        for offset,step in neighbors:
            neighbor=tuple((cell[i]+offset[i])%n for i in range(3));new=length+step
            if not mask[neighbor] or new>path_length_mpc_h or new>=distances.get(neighbor,float('inf')):continue
            distances[neighbor]=new;parents[neighbor]=cell;heapq.heappush(queue,(new,neighbor))
    endpoint=max(distances,key=lambda c:(distances[c],c));length=distances[endpoint]
    if length<3*box/n:raise ValueError('Connected ridge near host is too short for a fly-through at this resolution')
    path=[endpoint]
    while path[-1]!=seed:path.append(parents[path[-1]])
    return np.asarray(path,dtype=int),{'host_to_ridge_distance_mpc_h':float(dist[nearest]),'ridge_path_length_mpc_h':float(length),'connected_cells_examined':len(distances)}


def make_filament_flythrough(run_dir,destination,*,snapshot,host_row,grid=48,max_host_distance_fraction=.15,path_length_fraction=.35,frames=48,fps=24,max_particles=100000,memory_gib=1.,width=1280,height=720):
    from core.analysis import load_cfg,read_snapshot_header
    from core.data import validate_native_shard_counts,coordinate_contract
    from core.visualization import _particle_mass_density_grid
    from core.cosmic_flythrough import make_cosmic_flythrough
    from core.zoom import read_host_catalogue
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,product_provenance,verify_product_provenance,publish_directory
    root=Path(run_dir).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh filament fly-through directory')
    if type(snapshot) is not int or snapshot<0 or type(host_row) is not int or host_row<0 or type(grid) is not int or not 16<=grid<=96 or not 0<max_host_distance_fraction<=.5 or not 0<path_length_fraction<=1:raise ValueError('Valid snapshot, host, grid and route fractions required')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<=0 or grid**3*240+max_particles*350+128*1024**2>memory_gib*1024**3*.7:raise MemoryError('Filament mesh and movie sample exceed allocation')
    with native_run_lock(root,operation='filament fly-through'):
        anchor=read_host_catalogue(root,snapshot,memory_gib=memory_gib,host_row=host_row);header,files=read_snapshot_header(root,snapshot);validate_native_shard_counts(files,'NumPart_Total')
        a,box_native,_=anchor['frame'];box=box_native/1000.;host=np.asarray(anchor['positions'][host_row],float)/1000.
        if not math.isclose(float(header['Time']),a,rel_tol=1e-12) or not math.isclose(float(header['BoxSize'])/1000.,box,rel_tol=1e-12):raise ValueError('Halo and snapshot frame disagree')
        for path in files:
            with h5py.File(path,'r') as f:
                attrs=f['Header'].attrs
                if not math.isclose(float(attrs['Time']),a,rel_tol=1e-12) or not math.isclose(float(attrs['BoxSize'])/1000.,box,rel_tol=1e-12):raise ValueError('Snapshot shards disagree on epoch or box')
        cfg=load_cfg(root)['visualization'];smoothing=float(cfg['filament_smoothing_mpc_h']);factor=float(cfg['filament_curvature_threshold_factor']);percentile=float(cfg['filament_density_percentile'])
        if not 0<smoothing<box/2:raise ValueError('Configured smoothing must be between zero and half the box')
        source=product_provenance(root,algorithm=__file__,helper_sources=[Path(__file__).with_name('analysis.py'),Path(__file__).with_name('visualization.py'),Path(__file__).with_name('cosmic_flythrough.py')],source_files=list(files)+list(anchor['catalogues']),selection={'snapshot':snapshot,'host_row':host_row,'grid':grid,'path_length_fraction':path_length_fraction,'max_host_distance_fraction':max_host_distance_fraction},coordinates=coordinate_contract(position_units='comoving Mpc/h',origin='periodic simulation box',periodic_wrapping='periodic CIC and ridge graph',velocity_convention='not used'))
        density,metadata=_particle_mass_density_grid(files,1,box,grid,return_density_ratio=True)
        mask,classification=hessian_ridge_mask(density,sigma=smoothing/(box/grid),threshold_factor=factor,density_percentile=percentile)
        cells,route=ridge_route(mask,host,box,max_host_distance_mpc_h=max_host_distance_fraction*box,path_length_mpc_h=path_length_fraction*box)
        points=cells*box/grid
        if np.linalg.norm((points[-1]-host+box/2)%box-box/2)<box*1e-10:points=points[:-1]
        if len(points)<2:raise ValueError('Ridge route has too few distinct camera points')
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.filament-flight-',dir=out.parent))
        try:
            np.savez_compressed(stage/'ridge_route.npz',ridge_mask=mask,route_grid_cells=cells,route_mpc_h=cells*box/grid,host_mpc_h=host)
            movie=make_cosmic_flythrough(root,stage/'flight',snapshot=snapshot,host_row=host_row,start=points[0].tolist(),waypoints=points[1:].tolist(),frames=frames,fps=fps,max_particles=max_particles,memory_gib=memory_gib,width=width,height=height)
            verify_product_provenance(root,source)
            atomic_json(stage/'filament_contract.json',{'schema':'filament-flythrough-v1','science_acceptance':'NOT_ACCEPTED','provenance':source,'snapshot':snapshot,'host_row':host_row,'redshift':1/a-1,'box_mpc_h':box,'grid':grid,'smoothing_mpc_h':smoothing,'threshold_factor':factor,'density_percentile':percentile,'classification':classification,'route':route,'density_deposition':metadata,'flight_contract_sha256':file_sha256(stage/'flight/flythrough_contract.json'),'movie_sha256':file_sha256(movie),'route_sha256':file_sha256(stage/'ridge_route.npz'),'interpretation':'A connected 26-neighbor path through Hessian filament-like ridge cells, followed by a final segment to the FOF host. This is a grid- and threshold-dependent ridge proxy, not a persistence-filtered filament skeleton or an evolved trajectory.'})
            publish_directory(stage,out);return out/'flight/cosmic_flythrough.mp4'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
