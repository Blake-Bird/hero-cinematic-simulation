"""A perspective camera travelling through one saved cosmological snapshot."""
from __future__ import annotations
import math
import tempfile
from pathlib import Path
import imageio.v2 as imageio
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter


def unwrap_camera_path(points,box):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<2 or not np.isfinite(points).all() or not math.isfinite(box) or box<=0:raise ValueError('Finite camera waypoints and positive box required')
    points=np.mod(points,box);delta=np.diff(points,axis=0);delta=(delta+box/2)%box-box/2
    length=np.linalg.norm(delta,axis=1)
    if np.any(length<=box*1e-10):raise ValueError('Consecutive camera waypoints must differ')
    return np.vstack([points[0],points[0]+np.cumsum(delta,axis=0)])


def camera_samples(path, count, *, stop_distance=0.):
    """Ease a presentation camera along an arc-length-resampled smooth path."""
    from scipy.interpolate import PchipInterpolator
    path = np.asarray(path, float)
    segments = np.diff(path, axis=0)
    length = np.linalg.norm(segments, axis=1)
    if type(count) is not int or count < 2 or np.any(length <= 0):
        raise ValueError('At least two samples and a nondegenerate path are required')
    if not math.isfinite(stop_distance) or stop_distance < 0:
        raise ValueError('Finite nonnegative camera stopping distance required')
    ends = np.r_[0, np.cumsum(length)]
    sample = np.linspace(0, ends[-1], max(256, len(path)*16))
    curve = PchipInterpolator(ends, path, axis=0)(sample)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 0]
    curve, arc = curve[keep], arc[keep]
    t = np.linspace(0, 1, count)
    eased = t*t*(3-2*t)
    limit = min(arc[-1]*.97, max(arc[-1]*.25, arc[-1]-stop_distance))
    distance = eased*limit
    camera = np.column_stack([np.interp(distance, arc, curve[:, j]) for j in range(3)])
    tangent = np.column_stack([np.interp(distance, arc, np.gradient(curve[:, j], arc)) for j in range(3)])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    return camera, tangent


def camera_basis(forward):
    forward = np.asarray(forward, float)
    forward = forward/np.linalg.norm(forward)
    up = np.array([0., 0., 1.]) if abs(forward[2]) < .98 else np.array([0., 1., 0.])
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    return right, np.cross(right, forward)


def display_limits(rasters):
    """Ignore Gaussian numerical tails; use one robust exposure for the film."""
    samples = []
    for raster in rasters:
        positive = np.asarray(raster)[np.asarray(raster) > 0]
        if not len(positive): continue
        threshold = max(float(positive.max())*1e-4, float(np.percentile(positive, 80))*.02)
        useful = positive[positive >= threshold]
        samples.append(np.log10(useful[::max(1, len(useful)//20000)]))
    if not samples: raise ValueError('No visible density along the camera path')
    low, high = np.percentile(np.concatenate(samples), [8, 99.5])
    return float(low), float(max(high, low+.01))


def density_palette():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('readable cosmic density',
        ['#071421', '#123c59', '#207e9e', '#66d7dc', '#cef1e4', '#ffcf82'])


def color_density(raster, low, high):
    """Fixed log exposure, subtle display glow and an explicit empty background."""
    positive = raster > 0
    normal = np.zeros_like(raster, dtype=float)
    normal[positive] = np.clip((np.log10(raster[positive])-low)/(high-low), 0, 1)
    rgb = density_palette()(normal)[..., :3]
    glow = gaussian_filter(np.clip(normal-.45, 0, 1), 3.)
    rgb = np.clip(rgb + glow[..., None]*np.array([.06, .09, .08]), 0, 1)
    return (rgb*255).astype(np.uint8)


def perspective_mass_raster(positions,weights,camera,forward,box,width,height,fov_degrees):
    p=np.asarray(positions,float);w=np.asarray(weights,float);camera=np.asarray(camera,float);forward=np.asarray(forward,float)
    if p.ndim!=2 or p.shape[1]!=3 or w.shape!=(len(p),) or not np.isfinite(p).all() or not np.isfinite(w).all() or np.any(w<=0):raise ValueError('Finite aligned positions and positive represented masses required')
    if camera.shape!=(3,) or forward.shape!=(3,) or not np.isfinite(camera).all() or not np.isfinite(forward).all() or np.linalg.norm(forward)==0 or not math.isfinite(box) or box<=0:
        raise ValueError('Finite camera position, nonzero direction and positive box required')
    forward=forward/np.linalg.norm(forward);up=np.array([0.,0.,1.])
    if abs(np.dot(forward,up))>.98:up=np.array([0.,1.,0.])
    right,up=camera_basis(forward)
    relative=(p-np.mod(camera,box)+box/2)%box-box/2;depth=relative@forward
    tangent=math.tan(math.radians(fov_degrees)/2);aspect=width/height
    valid=(depth>box*.005)&(depth<box*.85)
    xx=np.zeros(len(p));yy=np.zeros(len(p));xx[valid]=(relative[valid]@right)/(depth[valid]*tangent*aspect);yy[valid]=(relative[valid]@up)/(depth[valid]*tangent)
    valid &= (np.abs(xx)<1)&(np.abs(yy)<1)
    grid,_,_=np.histogram2d(yy[valid],xx[valid],bins=(height,width),range=((-1,1),(-1,1)),weights=w[valid])
    return gaussian_filter(grid.astype(np.float32),.7,mode='constant'),int(valid.sum())


def density_volume(files, box, grid=128):
    """Complete saved DM mass on a periodic NGP mesh; smooth only for display."""
    import h5py
    mass = np.zeros((grid, grid, grid), dtype=np.float64)
    for path in files:
        with h5py.File(path) as source:
            if 'PartType1' not in source: continue
            group = source['PartType1']; n = len(group['Coordinates'])
            fixed = float(source['Header'].attrs['MassTable'][1])
            for start in range(0, n, 200000):
                stop = min(start+200000, n)
                p = np.asarray(group['Coordinates'][start:stop], float)/1000.
                w = np.asarray(group['Masses'][start:stop], float) if 'Masses' in group else np.full(stop-start, fixed)
                if not np.isfinite(p).all() or not np.isfinite(w).all() or np.any(w<=0):
                    raise ValueError('Finite saved particle positions and positive masses required')
                mass += np.histogramdd(np.mod(p,box), bins=grid, range=[(0,box)]*3, weights=w)[0]
    if mass.mean() <= 0: raise ValueError('No saved DM for the volume')
    return gaussian_filter((mass/mass.mean()).astype(np.float32), 1.15, mode='wrap')


def perspective_density_raster(volume, camera, forward, box, width, height, fov_degrees, *, steps=28):
    """Average mean-normalized DM density on equal-length perspective rays.

    This is a smoothed morphology view, not a radiative-transfer calculation.
    The full periodic density mesh is retained alongside the camera records.
    """
    from scipy.ndimage import map_coordinates
    right, up = camera_basis(forward)
    fw = np.asarray(forward)/np.linalg.norm(forward)
    small_w=max(80,width//4);small_h=max(48,height//4)
    yy,xx=np.meshgrid(np.linspace(1,-1,small_h),np.linspace(-1,1,small_w),indexing='ij')
    tangent=math.tan(math.radians(fov_degrees)/2)
    rays=fw + xx[...,None]*tangent*(width/height)*right + yy[...,None]*tangent*up
    rays/=np.linalg.norm(rays,axis=-1)[...,None]
    result=np.zeros((small_h,small_w),dtype=np.float32)
    for depth in np.linspace(box*.015,box*.7,steps):
        points=(np.asarray(camera)+rays*depth)%box
        coords=(points*volume.shape[0]/box-.5).reshape(-1,3).T
        result+=map_coordinates(volume,coords,order=1,mode='grid-wrap').reshape(small_h,small_w)/steps
    # Match the scientific raster's lower-left origin; display uses one flip.
    return np.flipud(result).copy()


def make_cosmic_flythrough(run_dir,destination,*,snapshot,host_row,start,waypoints=None,frames=48,
                           width=1280,height=720,fps=24,max_particles=300000,fov_degrees=60.,memory_gib=1.):
    from PIL import Image,ImageDraw,ImageFont
    from core.visualization import load_particles_for_fisheye
    from core.zoom import read_host_catalogue
    from core.native import native_run_lock
    from core.foundation import atomic_json,file_sha256,verify_product_provenance,publish_directory
    root=Path(run_dir).expanduser().resolve();out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh fly-through directory')
    if type(snapshot) is not int or snapshot<0 or type(host_row) is not int or host_row<0:raise ValueError('Saved snapshot and exact host row required')
    if type(frames) is not int or not 2<=frames<=1000 or type(fps) is not int or not 1<=fps<=120 or type(max_particles) is not int or not 1<=max_particles<=1000000:raise ValueError('Bounded frame, fps and sample controls required')
    if type(width) is not int or type(height) is not int or not 64<=width<=1920 or not 64<=height<=1088 or width%16 or height%16:raise ValueError('Frame dimensions must be multiples of 16 within supported bounds')
    if isinstance(fov_degrees,bool) or not isinstance(fov_degrees,(int,float)) or not math.isfinite(fov_degrees) or not 10<=fov_degrees<=140:raise ValueError('Field of view must be 10–140 degrees')
    if isinstance(memory_gib,bool) or not isinstance(memory_gib,(int,float)) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite memory allocation required')
    if max_particles*350+width*height*80+64*1024**2>memory_gib*1024**3*.7:raise MemoryError('Fly-through sample and frame buffers exceed allocation')
    with native_run_lock(root,operation='cosmic fly-through'):
        anchor=read_host_catalogue(root,snapshot,memory_gib=memory_gib,host_row=host_row);target=np.asarray(anchor['positions'][host_row],float)/1000.
        coords,weights,box,z,source,arrays=load_particles_for_fisheye(root,snapshot,'dm',max_particles,observer_mpc_h=[0,0,0],return_arrays=True)
        if not math.isclose(box,anchor['frame'][1]/1000.,rel_tol=1e-10):raise ValueError('Host catalogue and particle box differ')
        points=[start]+list(waypoints or [])+[target];path=unwrap_camera_path(points,box)
        radius = float(anchor['radii'][host_row])/1000.
        stopping_distance = max(box*.065, radius*3.)
        camera,tangent=camera_samples(path,frames,stop_distance=stopping_distance)
        toward=path[-1]-camera;toward/=np.linalg.norm(toward,axis=1)[:,None]
        direction=.2*tangent+.8*toward;direction/=np.linalg.norm(direction,axis=1)[:,None]
        particles=np.mod(coords,box);anchors=anchor['sources']+anchor['snapshot_sources']
        out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.flythrough-',dir=out.parent));movie=stage/'cosmic_flythrough.mp4'
        try:
            np.savez_compressed(stage/'sampled_particles.npz',**arrays,wrapped_positions_mpc_h=particles,camera_positions_unwrapped_mpc_h=camera,camera_directions=direction)
            from core.analysis import read_snapshot_header
            _, volume_files = read_snapshot_header(root, snapshot)
            volume = density_volume(volume_files, box)
            np.savez_compressed(stage/'density_volume.npz',density_over_mean=volume,box_comoving_mpc_h=box)
            records=[]
            panel=min(max(180,width//4),width//3);view_width=width-panel
            for i,(position,look) in enumerate(zip(camera,direction)):
                point_raster,count=perspective_mass_raster(particles,weights,position,look,box,view_width,height,fov_degrees)
                raster=perspective_density_raster(volume,position,look,box,view_width,height,fov_degrees)
                point_name='particles_%04d.npy'%i;np.save(stage/point_name,point_raster,allow_pickle=False)
                name='frame_%04d.npy'%i;np.save(stage/name,raster,allow_pickle=False)
                records.append({'frame':i,'camera_unwrapped_mpc_h':position.tolist(),'direction':look.tolist(),
                    'visible_sample_count':count,'particle_raster':point_name,'particle_raster_sha256':file_sha256(stage/point_name),'raster':name,'raster_sha256':file_sha256(stage/name)})
            low,high=display_limits(np.load(stage/r['raster'],allow_pickle=False) for r in records)
            plow,phigh=display_limits(np.load(stage/r['particle_raster'],allow_pickle=False) for r in records)
            font_path=str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf')
            font=ImageFont.truetype(font_path,max(12,width//78));small=ImageFont.truetype(font_path,max(10,width//95));large=ImageFont.truetype(font_path,max(17,width//49))
            overview=np.histogram2d(particles[:,0],particles[:,1],bins=256,range=((0,box),(0,box)),weights=weights)[0].T
            overview=gaussian_filter(overview,1.2);ol,oh=display_limits([overview])
            size=panel-32;overview_image=Image.fromarray(np.flipud(color_density(overview,ol,oh))).resize((size,size),Image.Resampling.LANCZOS)
            with imageio.get_writer(movie,fps=fps,codec='libx264',quality=9,macro_block_size=16) as writer:
                for record in records:
                    raster=np.load(stage/record['raster'],allow_pickle=False)
                    base_rgb=Image.fromarray(np.flipud(color_density(raster,low,high))).resize((view_width,height),Image.Resampling.LANCZOS)
                    point_raster=np.load(stage/record['particle_raster'],allow_pickle=False)
                    glints=np.flipud(color_density(point_raster,plow,phigh)).astype(float)
                    rgb=np.clip(np.asarray(base_rgb,dtype=float)+.22*np.maximum(glints-np.array([7,20,33]),0),0,255).astype(np.uint8)
                    image=Image.new('RGB',(width,height),'#08131f');image.paste(Image.fromarray(rgb),(0,0))
                    draw=ImageDraw.Draw(image)
                    # Keep the labelled target in context; never fly through its center.
                    i=record['frame'];relative=path[-1]-camera[i];depth=float(relative@direction[i]);right,up=camera_basis(direction[i]);tan=math.tan(math.radians(fov_degrees)/2)
                    if depth>0:
                        tx=view_width/2+(relative@right)/(depth*tan*(view_width/height))*view_width/2
                        ty=height/2-(relative@up)/(depth*tan)*height/2
                        if 15<tx<view_width-15 and 75<ty<height-60:
                            rr=max(12,min(55,radius/depth/tan*height/2))
                            draw.ellipse((tx-rr,ty-rr,tx+rr,ty+rr),outline='#fff0bc',width=2)
                            draw.line((tx+rr,ty,tx+rr+18,ty-18),fill='#fff0bc',width=2)
                            draw.text((min(tx+rr+21,view_width-190),ty-40),'TARGET · FOF %d'%host_row,font=small,fill='#fff0bc',stroke_width=2,stroke_fill='#071421')
                    draw.rectangle((0,0,view_width,70),fill='#08131f');draw.text((22,10),'THE COSMIC WEB',font=large,fill='#ecf7ff')
                    draw.text((22,43),'z = %.4f  ·  saved snapshot %03d  ·  sampled dark matter'%(z,snapshot),font=small,fill='#a9cadc')
                    draw.rectangle((view_width,0,width,height),fill='#0f2233');sx=view_width+16
                    draw.text((sx,20),'WHERE AM I?',font=font,fill='#e9f7ff');image.paste(overview_image,(sx,58));draw=ImageDraw.Draw(image)
                    def map_point(p):
                        p=np.mod(p,box);return sx+p[0]/box*size,58+(1-p[1]/box)*size
                    hx,hy=map_point(target);cx,cy=map_point(camera[i]);draw.ellipse((hx-7,hy-7,hx+7,hy+7),outline='#ffcc7f',width=2);draw.ellipse((cx-4,cy-4,cx+4,cy+4),fill='#f1fcff')
                    ahead=camera[i]+direction[i]*box*.06;fx,fy=map_point(ahead)
                    if abs(fx-cx)<size/2 and abs(fy-cy)<size/2:draw.line((cx,cy,fx,fy),fill='#f1fcff',width=2)
                    y=58+size+20
                    for text,color in [('White: camera / direction','#e6f6ff'),('Gold ring: target halo','#ffcc7f'),('XY overview · full box','#9ebed2'),('','white'),('FOF row %d'%host_row,'#ecf7ff'),('R200c = %.3g Mpc/h'%radius,'#9ebed2'),('Distance = %.3g Mpc/h'%np.linalg.norm(relative),'#9ebed2')]:
                        draw.text((sx,y),text,font=small,fill=color);y+=23
                    draw.rectangle((0,height-43,view_width,height),fill='#08131f');draw.text((20,height-31),'Smoothed mass density + sampled points · camera moves through one frozen epoch',font=small,fill='#a9cadc')
                    image.save(stage/'preview.png') if i==frames//2 else None
                    writer.append_data(np.asarray(image))
            verify_product_provenance(root,source['data_contract'])
            for row in anchors:
                if file_sha256(row['path'])!=row['sha256']:raise ValueError('Halo catalogue source changed during fly-through')
            for row in records:
                if file_sha256(stage/row['raster'])!=row['raster_sha256'] or file_sha256(stage/row['particle_raster'])!=row['particle_raster_sha256']:raise ValueError('Fly-through raster changed')
            atomic_json(stage/'flythrough_contract.json',{'schema':'saved-snapshot-flythrough-v1','science_acceptance':'NOT_ACCEPTED',
                'run':str(root),'source':source,'anchor_sources':anchors,'anchor_host_row':host_row,'target_mpc_h':target.tolist(),
                'waypoints_wrapped_mpc_h':np.mod(np.asarray(points,float),box).tolist(),'path_unwrapped_mpc_h':path.tolist(),
                'frames':records,'fps':fps,'width':width,'height':height,'field_of_view_degrees':fov_degrees,
                'display_log10_limits':[low,high],'sample_sha256':file_sha256(stage/'sampled_particles.npz'),
                'movie_sha256':file_sha256(movie),'algorithm_sha256':file_sha256(__file__),
                'camera_scope':'Arc-length-smoothed periodic waypoint path with eased speed, target-biased gaze, and finite halo standoff. Camera motion is not physical evolution.',
                'target_standoff_mpc_h':stopping_distance,'palette':'navy-cyan-cream; fixed robust log density exposure',
                'exposure_policy':'8th and 99.5th percentiles after rejecting Gaussian tails below max*1e-4 and percentile80*.02',
                'render_scope':'Complete saved DM deposited on a periodic 128-cubed NGP mesh and Gaussian-smoothed by 1.15 cells. Equal-length ray averages of density/mean are display morphology, not emitted light. Sampled particle glints are an additional display overlay; no unsaved motion is implied.',
                'volume_sha256':file_sha256(stage/'density_volume.npz'),'volume_ray_steps':28,'particle_overlay_strength':.22,'particle_display_log10_limits':[plow,phigh]})
            publish_directory(stage,out);return out/'cosmic_flythrough.mp4'
        except BaseException as error:
            atomic_json(stage/'failure.json',{'status':'FAIL','type':type(error).__name__,'message':str(error)});raise
