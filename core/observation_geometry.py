"""Native observation geometry, stellar attenuation and survey measurements.

Readers preserve native identities and explicit per-epoch unit conversions.
Lightcones use no-h comoving distances and proper emission clocks. Lens planes
use proper kpc and Msun; their angular deflections are radians. Dark gravitating
mass does not acquire stellar light, and thin lenses do not model strong-field
black-hole ray tracing. Periodic replicas are not independent realizations.
"""
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicHermiteSpline, RegularGridInterpolator
from scipy.special import erf, expit
from scipy.integrate import cumulative_trapezoid
from .foundation import KPC_CM, MSUN_G, SECONDS_PER_YEAR
from .stream_common import finite, increasing, positive, integer, digest, KMS_TO_KPC_MYR, G, seeded_rng
from .stream.trajectories import real_roots


def read_native_gravitating_particles(root,snapshot,*,species,maximum_particles,memory_gib,namespace):
    import h5py
    from .halos import _snapshot_files
    from .data import verify_native_hdf5_set,native_publication_markers,exact_uint64_ids,dataset_to_cgs_factor,snapshot_total_counts,run_epoch_contract,isolated_epoch_contract
    root=Path(root).resolve();files=_snapshot_files(root,snapshot);verify_native_hdf5_set(files,require_publication=True)
    if not species or any(type(k) is not int or not 0<=k<6 for k in species) or len(set(species))!=len(species):raise ValueError('Explicit distinct native gravitating species required')
    sources={str(p.resolve()):digest(p) for p in [*files,*native_publication_markers(files)]}
    columns={key:[] for key in ('particle_ids','species','position_proper_kpc','velocity_peculiar_kms','mass_msun')};reference=None
    with h5py.File(files[0],'r') as f:count=sum(int(snapshot_total_counts(f['Header'].attrs)[k]) for k in species)
    if count>maximum_particles or count*200>positive(memory_gib,'memory budget')*1024**3:raise MemoryError('Complete native particle selection exceeds workspace')
    for path in files:
        with h5py.File(path,'r') as f:
            p=f['Parameters'].attrs;header=f['Header'].attrs;time=float(header['Time']);comoving=int(p['ComovingIntegrationOn']);h=float(p['HubbleParam'])
            common=(time,comoving,h,float(p['UnitMass_in_g']),float(p['UnitLength_in_cm']),float(p['UnitVelocity_in_cm_per_s']))
            if reference is not None and reference!=common:raise ValueError('Native shards change physical units/epoch')
            from .data import native_time_unit_seconds
            reference=common;time_unit_s=native_time_unit_seconds(p);a=time if comoving else 1.
            if a<=0 or h<=0:raise ValueError('Invalid native scale/hubble parameter')
            for kind in species:
                n=int(header['NumPart_ThisFile'][kind])
                if not n:continue
                group=f['PartType%d'%kind];ids=exact_uint64_ids(group['ParticleIDs'][:],allow_empty=True)
                if len(ids)!=n:raise ValueError('Incomplete native identity census')
                x=group['Coordinates'][:]*dataset_to_cgs_factor(group['Coordinates'].attrs,a,h)/KPC_CM
                v=group['Velocities'][:]*dataset_to_cgs_factor(group['Velocities'].attrs,a,h)/1e5
                if 'Masses' in group:m=group['Masses'][:]*dataset_to_cgs_factor(group['Masses'].attrs,a,h)/MSUN_G
                else:
                    # Native cosmological masses carry 1/h; isolated code masses
                    # use UnitMass_in_g directly, just like their Masses datasets.
                    mass_scale = common[3]/MSUN_G/(h if comoving else 1.)
                    m = np.full(n, float(header['MassTable'][kind])*mass_scale)
                tombstone=(kind==5)&(ids==0)&(m==0);keep=~tombstone
                if not np.isfinite(x).all() or not np.isfinite(v).all() or not np.isfinite(m).all() or np.any(m[keep]<=0):raise ValueError('Invalid native gravitating particle payload')
                for key,value in {'particle_ids':ids,'species':np.full(n,kind,dtype=np.int8),'position_proper_kpc':x,'velocity_peculiar_kms':v,'mass_msun':m}.items():columns[key].append(value[keep])
    arrays={key:np.concatenate(value) if value else np.empty((0,3) if key in {'position_proper_kpc','velocity_peculiar_kms'} else (0,),dtype=np.uint64 if key=='particle_ids' else float) for key,value in columns.items()}
    if len(np.unique(arrays['particle_ids']))!=len(arrays['particle_ids']):raise ValueError('Native selection aliases exact identities')
    for path,expected in sources.items():
        if digest(path)!=expected:raise ValueError('Native particle source changed during reading')
    time,comoving,h,*units=reference
    epoch=run_epoch_contract(root,snapshot,time) if comoving else isolated_epoch_contract(time,time_unit_gyr=time_unit_s/(SECONDS_PER_YEAR*1e9),snapshot=snapshot)
    # Copy unit-dependent metadata out of the HDF handle before closing it.
    if comoving:arrays['position_comoving_mpc']=arrays['position_proper_kpc']/time/1000
    return {'schema':'native-gravitating-particles-v1','namespace':namespace,'epoch':epoch,'source_hashes':sources,
        'species_selection':species,'position_units':'comoving_Mpc_no_h' if comoving else 'proper_kpc',
        'mass_convention':'actual gravitating particle mass; BH composite mass retained; no luminosity assignment'},arrays



def read_gadget_lightcone(path,request,*,maximum_particles):
    import h5py
    from .data import exact_uint64_ids,dataset_to_cgs_factor
    path=Path(path).resolve();before=digest(path);columns={k:[] for k in ('particle_ids','species','position_proper_kpc','velocity_peculiar_kms','mass_msun','scale_factor')}
    with h5py.File(path,'r') as file:
        header=file['Header'].attrs;p=file['Parameters'].attrs
        if int(p['ComovingIntegrationOn'])!=1:raise ValueError('Native cosmological lightcone requires comoving run')
        h=positive(float(p['HubbleParam']),'h');counts=np.asarray(header['NumPart_ThisFile'])
        if sum(map(int,counts))>maximum_particles:raise MemoryError('Native cone exceeds declared particle workspace')
        for kind,count in enumerate(counts):
            n=int(count)
            if not n:continue
            group=file['PartType%d'%kind];ids=exact_uint64_ids(group['ParticleIDs'][:],allow_empty=True)
            if len(ids)!=n or 'Ascale' not in group:raise ValueError('Native cone requires complete exact identities and per-particle emission scale')
            a=finite(group['Ascale'][:],(n,))
            if np.any(a<=0):raise ValueError('Invalid native emission epoch')
            factors=lambda name:np.array([dataset_to_cgs_factor(group[name].attrs,float(q),h) for q in a])
            x=finite(group['Coordinates'][:],(n,3))*factors('Coordinates')[:,None]/KPC_CM
            v=finite(group['Velocities'][:],(n,3))*factors('Velocities')[:,None]/1e5
            if 'Masses' in group:mass=finite(group['Masses'][:],(n,))*factors('Masses')/MSUN_G
            else:
                table=np.asarray(header.get('MassTable',request.get('mass_table_native',[])))
                if table.shape!=(6,) or table[kind]<=0:raise ValueError('Native lightcone lacks per-particle mass and recorded mass table')
                mass=np.full(n,table[kind]*float(p['UnitMass_in_g'])/h/MSUN_G)
            for key,value in {'particle_ids':ids,'species':np.full(n,kind,dtype=np.int8),'position_proper_kpc':x,'velocity_peculiar_kms':v,'mass_msun':mass,'scale_factor':a}.items():columns[key].append(value)
        # Do not evaluate a fallback request key when the native file already
        # records the observer origin.
        origin=np.asarray(header['Origin'] if 'Origin' in header else request['observer_origin_native'],float)
        if origin.shape!=(3,) or not np.isfinite(origin).all():
            raise ValueError('Finite three-component native observer origin required')
    arrays={key:np.concatenate(values,axis=0) if values else np.empty((0,3) if key in {'position_proper_kpc','velocity_peculiar_kms'} else (0,),dtype=np.uint64 if key=='particle_ids' else float) for key,values in columns.items()}
    a=arrays['scale_factor'];arrays['position_comoving_kpc']=arrays['position_proper_kpc']/a[:,None]
    radius=np.linalg.norm(arrays['position_comoving_kpc'],axis=1);aobs=positive(request['observer_scale_factor'],'observer scale')
    arrays.update(distance_comoving_kpc=radius,angular_diameter_distance_kpc=a*radius,luminosity_distance_kpc=aobs*aobs*radius/a,redshift_cosmological=aobs/a-1)
    if digest(path)!=before:raise ValueError('Native emission file changed during reading')
    return {'schema':'gadget-native-lightcone-v1','origin':'native observer-relative signed Cartesian cone coordinates','namespace':request['namespace'],'source_hashes':{str(path):before},'observer_origin_native':origin,
        'upstream_interface':'GADGET-4 lightcone_particle_io: observer-relative signed Coordinates and Ascale; velocities carry a^0.5',
        'identity_convention':'same native ID may recur in distinct periodic replicas; no duplicate-ID rejection across emission observations',
        'units':{'position_comoving_kpc':'comoving kpc no h','distance_comoving_kpc':'comoving kpc no h'},'science_acceptance':'NOT_ACCEPTED'},arrays



C_KMS=299792.458



def background_lightcone(clock,observation_time_myr,request):
    t=increasing(clock['times_myr']);a=increasing(clock['scale_factor'],'scale factor')
    if t.shape!=a.shape or not clock['source_hashes']:raise ValueError('Hashed physical background clock required')
    if not t[0]<=observation_time_myr<=t[-1]:raise ValueError('Observer epoch outside background clock')
    if np.max(np.diff(t))>request['maximum_clock_interval_myr']:raise ValueError('Background clock too sparse for lightcone support')
    error=clock['distance_interpolation_certificate']
    if error['status']!='PASS' or not error['source_hashes']:raise ValueError('Lightcone distances require independent background interpolation support')
    if error['maximum_error_kpc']>request['maximum_distance_error_kpc']:raise ValueError('Background distance error exceeds requested budget')
    # comoving path length, c dt / a, in no-h comoving kpc.
    integral=cumulative_trapezoid(C_KMS*KMS_TO_KPC_MYR/a,t,initial=0.)
    chi=np.interp(observation_time_myr,t,integral)-integral
    use=t<=observation_time_myr
    knots=np.r_[t[use],observation_time_myr] if t[use][-1]!=observation_time_myr else t[use]
    return {'times_myr':knots,'scale_factor':np.interp(knots,t,a),'distance_comoving_kpc':np.interp(knots,t,chi),
        'source_hashes':clock['source_hashes'],'distance_model':'flat FLRW; c integral dt/a; measured clock; piecewise interpolation',
        'maximum_interpolation_error_kpc':float(error['maximum_error_kpc']),
        'spatial_curvature':clock['spatial_curvature']}



def emission_intersections(tracks,background,observer,request):
    if background['spatial_curvature']!=0:raise ValueError('Cartesian replicated lightcone requires flat FLRW')
    bt=increasing(background['times_myr']);ba=increasing(background['scale_factor']);bc=finite(background['distance_comoving_kpc'],bt.shape)
    if np.any(bc<0) or np.any(np.diff(bc)>0):raise ValueError('Past-cone distance must decrease towards observer')
    view=finite(observer['view_direction'],(3,))
    if not np.isclose(np.linalg.norm(view),1.):raise ValueError('Observer view direction must be normalized')
    origin=finite(observer['position_comoving_kpc'],(3,));ov=finite(observer['velocity_peculiar_kms'],(3,))
    box=positive(request['box_comoving_kpc'],'box size');images=np.asarray(request['replica_images'])
    if images.ndim!=2 or images.shape[1]!=3 or images.dtype.kind not in 'iu':raise ValueError('Exact integer replica image list required')
    if len({tuple(q) for q in images})!=len(images):raise ValueError('Duplicate periodic replicas')
    if len(tracks)*len(images)*len(bt)>request['maximum_intersection_cells']:raise MemoryError('Lightcone intersections exceed workspace')
    events=[];unsupported=[]
    for track in tracks:
        if track['units']!='comoving_kpc_peculiar_kms_myr_no_h':raise ValueError('Native comoving lightcone unit convention required')
        t=increasing(track['times_myr']);a=increasing(track['scale_factor']);x=finite(track['position_comoving_kpc'],(len(t),3));v=finite(track['velocity_peculiar_kms'],x.shape)
        recorded=np.asarray(track['image_offsets'])
        if recorded.shape!=x.shape or recorded.dtype.kind not in 'iu':raise ValueError('Lightcone trajectories require actual recorded periodic image offsets')
        x=x+recorded*box
        if np.max(np.abs(a-np.interp(t,bt,ba)))>request['scale_factor_clock_tolerance']:raise ValueError('Track and background clocks disagree')
        spline=CubicHermiteSpline(t,x,v*KMS_TO_KPC_MYR/a[:,None],extrapolate=False)
        recorded_support=np.asarray(track['valid_intervals'])
        if recorded_support.dtype.kind!='b' or recorded_support.shape!=(len(t)-1,):
            raise ValueError('Exact aligned Boolean lightcone interval support required')
        valid=recorded_support&(np.diff(t)<=request['maximum_trajectory_interval_myr'])
        errors=finite(track['interpolation_error_comoving_kpc'],(len(t)-1,))
        valid&=errors<=request['maximum_trajectory_error_kpc']
        for i in range(len(t)-1):
            if not valid[i]:unsupported.append({'identity':track['identity'],'interval_myr':t[i:i+2].tolist()});continue
            left=max(t[i],bt[0]);right=min(t[i+1],bt[-1])
            if left>=right:continue
            knots=np.unique(np.r_[left,bt[(bt>left)&(bt<right)],right])
            for start,stop in zip(knots[:-1],knots[1:]):
                shifted=Polynomial([start-t[i],1.]);xyz=[Polynomial(spline.c[:,i,j][::-1])(shifted) for j in range(3)]
                distance=Polynomial([np.interp(start,bt,bc),(np.interp(stop,bt,bc)-np.interp(start,bt,bc))/(stop-start)])
                for image in images:
                    rel=[xyz[j]+int(image[j])*box-origin[j] for j in range(3)]
                    cone=sum((p*p for p in rel),Polynomial([0.]))-distance*distance
                    roots=real_roots(cone,0.,stop-start)
                    if abs(cone(0.))<=request['root_residual_kpc2']:roots=[0.,*roots]
                    # Half-open intervals avoid duplicate shell-boundary observations.
                    if stop==bt[-1] and abs(cone(stop-start))<=request['root_residual_kpc2']:roots.append(stop-start)
                    for root in roots:
                        emission=start+root;position=np.array([p(root) for p in rel]);radius=np.linalg.norm(position)
                        if radius<=request['minimum_distance_kpc']:continue
                        direction=position/radius
                        if np.dot(direction,view)<np.cos(request['opening_angle_rad']):continue
                        factor=float(np.interp(emission,bt,ba));velocity=np.asarray(spline.derivative()(emission))*factor/KMS_TO_KPC_MYR
                        beta=velocity/C_KMS;obeta=ov/C_KMS
                        if np.dot(beta,beta)>=1 or np.dot(obeta,obeta)>=1:raise ValueError('Superluminal lightcone peculiar velocity')
                        doppler=(1+np.dot(beta,direction))/np.sqrt(1-np.dot(beta,beta))*np.sqrt(1-np.dot(obeta,obeta))/(1+np.dot(obeta,direction))
                        aobs=float(ba[-1]);z=aobs/factor-1;zobs=(1+z)*doppler-1
                        events.append({'identity':track['identity'],'namespace':track['namespace'],'species':track['species'],'replica_image':image.tolist(),
                            'emission_time_myr':float(emission),'scale_factor':factor,'redshift_cosmological':z,'redshift_observed':float(zobs),
                            'distance_comoving_kpc':float(radius),'angular_diameter_distance_kpc':float(factor*radius),
                            'luminosity_distance_kpc':float(aobs*aobs*radius/factor),'position_comoving_kpc':position,'velocity_peculiar_kms':velocity,
                            'mass_msun':float(np.interp(emission,t,finite(track['mass_msun'],t.shape))),
                            'interpolation_error_comoving_kpc':float(errors[i]+background['maximum_interpolation_error_kpc'])})
    keys=set();unique=[]
    for row in sorted(events,key=lambda q:q['emission_time_myr']):
        key=(row['namespace'],str(row['identity']),tuple(row['replica_image']),round(row['emission_time_myr'],10))
        if key not in keys:unique.append(row);keys.add(key)
    arrays={key:np.asarray([row[key] for row in unique]) for key in ('emission_time_myr','scale_factor','redshift_cosmological','redshift_observed','distance_comoving_kpc','angular_diameter_distance_kpc','luminosity_distance_kpc','mass_msun','species')}
    arrays['position_comoving_kpc']=np.array([row['position_comoving_kpc'] for row in unique]).reshape(-1,3)
    arrays['velocity_peculiar_kms']=np.array([row['velocity_peculiar_kms'] for row in unique]).reshape(-1,3)
    return {'schema':'emission-lightcone-v1','origin':'observer-relative Cartesian; distances explicitly comoving or angular/luminosity physical','events':unique,'unsupported_intervals':unsupported,'background':background,'observer':observer,
        'replica_note':'replicas sample the same simulation volume; not independent realizations','redshift_model':'FLRW plus SR peculiar Doppler; gravitational shifts omitted',
        'units':{'position_comoving_kpc':'comoving kpc no h','distance_comoving_kpc':'comoving kpc no h'},'science_acceptance':'NOT_ACCEPTED'},arrays



def species_lens_plane(particles,request):
    """Deposit gravitating mass and solve the thin-plane angular potential.

    Sigma_crit = c^2 D_s/(4*pi*G*D_l*D_ls), with distances in proper kpc
    and G in kpc*(km/s)^2/Msun. Kappa = (Sigma-Sigma_background)/Sigma_crit.
    The FFT solves angular Laplacian(psi)=2*kappa for the periodic contrast;
    its mean is restored as a uniform mass sheet. Deflections are radians.
    """
    x=finite(particles['position_proper_kpc']);mass=finite(particles['mass_msun'],(len(x),));species=np.asarray(particles['species'])
    if x.shape!=(len(mass),3) or species.shape!=mass.shape or np.any(mass<=0):raise ValueError('Actual native lens masses/positions/species required')
    if not particles['source_hashes']:raise ValueError('Lens planes require source provenance')
    n=integer(request['pixels'],'pixels',4);extent=positive(request['half_width_proper_kpc'],'plane extent');distance=positive(request['angular_diameter_distance_kpc'],'lens distance')
    axis=finite(request['line_of_sight'],(3,)).copy()
    if np.linalg.norm(axis)==0:raise ValueError('Lens line of sight cannot vanish')
    axis/=np.linalg.norm(axis)
    horizontal=finite(request['horizontal_direction'],(3,)).copy();horizontal-=np.dot(horizontal,axis)*axis
    if np.linalg.norm(horizontal)<1e-12:raise ValueError('Degenerate lens projection basis')
    horizontal/=np.linalg.norm(horizontal);vertical=np.cross(axis,horizontal)
    rel=x-finite(request['center_proper_kpc'],(3,));depth=rel@axis;xx=rel@horizontal;yy=rel@vertical
    use=(np.abs(depth)<=positive(request['half_depth_proper_kpc'],'plane depth'))&(np.abs(xx)<extent)&(np.abs(yy)<extent)
    spacing=2*extent/n;area=spacing**2;maps={};total=np.zeros((n,n));deposited=0.
    # CIC onto cell centers with explicit clipped edge support, renormalized per particle.
    gx=(xx+extent)/spacing-.5;gy=(yy+extent)/spacing-.5
    for kind in sorted(set(map(int,species[use]))):
        surface=np.zeros((n,n))
        for p in np.flatnonzero(use&(species==kind)):
            ix=int(np.floor(gx[p]));iy=int(np.floor(gy[p]));fx=gx[p]-ix;fy=gy[p]-iy
            cells=[(ix,iy,(1-fx)*(1-fy)),(ix+1,iy,fx*(1-fy)),(ix,iy+1,(1-fx)*fy),(ix+1,iy+1,fx*fy)]
            valid=[(i,j,w) for i,j,w in cells if 0<=i<n and 0<=j<n];norm=sum(w for i,j,w in valid)
            for i,j,w in valid:surface[j,i]+=mass[p]*w/norm/area
        maps['species_%d_surface_density_msun_kpc2'%kind]=surface;total+=surface;deposited+=float(mass[use&(species==kind)].sum())
    ds=positive(request['source_angular_diameter_distance_kpc'],'source distance');dls=positive(request['lens_source_angular_diameter_distance_kpc'],'lens-source distance')
    sigmacrit=C_KMS**2/(4*np.pi*G)*ds/(distance*dls)
    background=float(request['background_surface_density_msun_kpc2'])
    kappa=(total-background)/sigmacrit;mean=float(kappa.mean());contrast=kappa-mean
    pixel_rad=spacing/distance;freq=2*np.pi*np.fft.fftfreq(n,d=pixel_rad);kx,ky=np.meshgrid(freq,freq);k2=kx*kx+ky*ky
    fourier=np.fft.fft2(contrast);potential=np.divide(-2*fourier,k2,out=np.zeros_like(fourier),where=k2>0)
    alphax=np.fft.ifft2(1j*kx*potential).real;alphay=np.fft.ifft2(1j*ky*potential).real
    theta=(np.arange(n)+.5)*pixel_rad-extent/distance;tx,ty=np.meshgrid(theta,theta)
    alphax+=mean*tx;alphay+=mean*ty
    shear1=np.fft.ifft2(-.5*(kx*kx-ky*ky)*potential).real;shear2=np.fft.ifft2(-kx*ky*potential).real
    report={'schema':'species-lens-plane-v1','request':request,'deposited_mass_msun':deposited,'source_hashes':particles['source_hashes'],
        'matched_experiment':request.get('matched_experiment'),
        'critical_surface_density_msun_kpc2':sigmacrit,'boundary':'periodic density contrast plus explicit uniform mass sheet; edge effects require later convergence',
        'matter_encoding':'species-resolved gravitating mass; dark matter has no stellar luminosity','bh_rendering':'BH mass contribution to thin lens; separate strong-field ray tracer required',
        'science_acceptance':'NOT_ACCEPTED','units':{'surface_density_msun_kpc2':'Msun / proper kpc^2','critical_surface_density_msun_kpc2':'Msun / proper kpc^2'}}
    return report,{**maps,'surface_density_msun_kpc2':total,'theta_axis_rad':theta,'kappa':kappa,'shear1':shear1,'shear2':shear2,
        'reduced_deflection_x_rad':alphax,'reduced_deflection_y_rad':alphay,'magnification_inverse':(1-kappa)**2-shear1**2-shear2**2}



def trace_multiple_planes(planes,theta,source,request):
    rays=finite(theta);n=len(rays)
    if rays.shape!=(n,2):raise ValueError('Angular rays require N by 2 radians')
    if not planes:raise ValueError('At least one measured lens plane required')
    chi=finite([r['report']['request']['distance_comoving_kpc'] for r in planes]);cs=positive(source['distance_comoving_kpc'],'source comoving distance')
    if np.any(np.diff(chi)<=0) or np.any(chi<=0) or chi[-1]>=cs:raise ValueError('Lens planes ordered strictly between observer and source')
    if request['spatial_curvature']!=0:raise ValueError('Multiple-plane distance ratios require flat FLRW')
    positions=[];physical_deflections=[];supported=np.ones(n,bool)
    for j,plane in enumerate(planes):
        at=rays.copy()
        for i,alpha in enumerate(physical_deflections):at-=(chi[j]-chi[i])/chi[j]*alpha
        grid=plane['arrays'];axis=finite(grid['theta_axis_rad']);p=plane['report']['request']
        source_ratio=p['lens_source_angular_diameter_distance_kpc']/p['source_angular_diameter_distance_kpc']
        reduced=np.column_stack([RegularGridInterpolator((axis,axis),grid['reduced_deflection_'+coord+'_rad'],bounds_error=False,fill_value=np.nan)(at[:,::-1]) for coord in ('x','y')])
        alpha=reduced/source_ratio;supported&=np.isfinite(alpha).all(axis=1);positions.append(at);physical_deflections.append(alpha)
    beta=rays.copy()
    for i,alpha in enumerate(physical_deflections):beta-=(cs-chi[i])/cs*alpha
    return ({'schema':'multiple-plane-rays-v1','source':source,'plane_count':len(planes),'supported_count':int(supported.sum()),
        'scope':'geometric thin-lens ray mapping; no strong-field BH rendering or time-delay claim','science_acceptance':'NOT_ACCEPTED'},
        {'image_theta_rad':rays,'source_beta_rad':beta,'plane_theta_rad':np.asarray(positions),'ray_supported':supported})


def gas_metal_dust_screen(stellar_positions, observer, gas, wavelength, request):
    """Integrate foreground Gaussian gas kernels into a dust optical depth.

    Positions and kernel widths are proper kpc; gas masses are Msun. Each
    kernel's line integral is its projected Gaussian times the fraction between
    observer and star. Dust mass = gas mass * metal fraction * dust/metals.
    Sigma_dust is converted to g/cm^2, tau = Sigma_dust * kappa(lambda), and
    transmission = exp(-tau). This screen does not compute reprocessed IR.
    """
    stars = finite(stellar_positions, name='stellar positions')
    origin = finite(observer, (3,), 'observer position')
    positions = finite(gas['positions_proper_kpc'], name='gas positions')
    if stars.ndim != 2 or stars.shape[1] != 3 or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError('Physical Cartesian source and gas positions required')
    mass = finite(gas['mass_msun'], (len(positions),), 'gas masses')
    metal = finite(gas['metallicity_mass_fraction'], mass.shape, 'gas metallicity')
    width = finite(gas['smoothing_length_proper_kpc'], mass.shape, 'gas kernel width')
    if not gas['source_hashes'] or gas['metallicity_origin'] != 'native_gas_elements':
        raise ValueError('Dust requires measured native gas metals and provenance')
    if np.any(width <= 0) or np.any(mass <= 0) or np.any((metal < 0) | (metal > 1)):
        raise ValueError('Invalid gas-metal dust material')
    fraction = float(finite(request['dust_to_metals_fraction'], (), 'dust/metals fraction'))
    if not 0 <= fraction <= 1:
        raise ValueError('Dust-to-metal mass fraction outside [0,1]')
    table = request['opacity_table']
    wave = increasing(table['wavelength_angstrom'], 'opacity wavelengths')
    opacity = finite(table['opacity_cm2_per_g_dust'], wave.shape, 'dust opacity')
    target = increasing(wavelength, 'spectrum wavelengths')
    if wave[0] <= 0 or target[0] < wave[0] or target[-1] > wave[-1] or np.any(opacity < 0) or not table['source_sha256']:
        raise ValueError('Named hashed dust opacity table must cover positive spectrum wavelengths')
    if request['model'] != 'native_metal_gaussian_foreground_screen':
        raise ValueError('Only the explicitly named gas-metal Gaussian screen is implemented')
    maximum_pairs = integer(request['maximum_star_gas_pairs'], 'maximum star/gas pairs', 0)
    if len(stars) * len(positions) > maximum_pairs:
        raise MemoryError('Dust sightline workspace exceeded')

    relative = positions - origin
    dust_mass = mass * metal * fraction
    columns = np.zeros(len(stars))
    for i, star in enumerate(stars):
        delta = star - origin
        length = np.linalg.norm(delta)
        if not np.isfinite(length) or length <= 0:
            raise ValueError('Finite nonzero star/observer separation required')
        direction = delta / length
        along = relative @ direction
        impact2 = np.maximum(0, np.einsum('ij,ij->i', relative, relative) - along**2)
        foreground = .5 * (erf((length - along) / (np.sqrt(2) * width)) - erf(-along / (np.sqrt(2) * width)))
        projected = np.exp(-impact2 / (2 * width**2)) / (2 * np.pi * width**2) * foreground
        columns[i] = np.dot(dust_mass, projected) * MSUN_G / KPC_CM**2
    tau = columns[:, None] * np.interp(target, wave, opacity)
    if not np.isfinite(tau).all() or not np.isfinite(columns).all():
        raise ValueError('Dust column/optical depth exceeds the representable numeric range')
    report = {
        'schema': 'gas-metal-dust-v1', 'model': request['model'],
        'geometry': 'Gaussian gas smoothing kernels integrated from observer to each source',
        'opacity_table': table, 'dust_to_metals_fraction': fraction,
        'reprocessed_ir': 'not computed', 'source_hashes': gas['source_hashes'],
        'science_acceptance': 'NOT_ACCEPTED',
        'units': {'dust_surface_density_g_cm2': 'g/cm^2', 'optical_depth': 'dimensionless', 'attenuation': 'dimensionless'},
    }
    return report, {'dust_surface_density_g_cm2': columns, 'optical_depth': tau, 'attenuation': np.exp(-tau)}


def observer_fluxes(sps_report, sps, geometry, filters, *, attenuation=None):
    """Convert native SSP spectra to photon-counting band-averaged f_nu in Jy.

    For observer wavelength lambda_o, lambda_e=lambda_o/(1+z), and
    f_lambda_o=L_lambda_e/[4*pi*D_L^2*(1+z)]. A photon-counting band gives
    <f_nu>=integral(f_lambda*lambda*T dlambda)/(c*integral(T/lambda dlambda)).
    Wavelengths are Angstrom, D_L is converted from kpc to cm, spectra from
    Lsun/Angstrom to erg/s/Angstrom, and 1 Jy = 1e-23 erg/s/cm^2/Hz.
    Unsupported SSPs or uncovered filter wavelengths remain NaN with a mask.
    """
    from .data import exact_uint64_ids
    from .stellar_population import LSUN_ERG_S, C_ANGSTROM_S

    wave = increasing(sps['wavelength_angstrom'], 'SPS wavelengths')
    if wave[0] <= 0:
        raise ValueError('SPS wavelengths must be positive')
    ids = exact_uint64_ids(sps['particle_ids'], allow_empty=True)
    spectra = np.asarray(sps['spectrum_lsun_per_angstrom'], float)
    n = len(ids)
    support = np.asarray(sps['sps_supported'])
    if spectra.shape != (n, len(wave)) or support.shape != (n,) or support.dtype.kind != 'b':
        raise ValueError('SPS spectra and Boolean support must align with stellar identities')
    source_rows = np.flatnonzero(support)
    # Inspect views one source at a time; Boolean-indexing the complete spectral
    # grid would allocate another potentially large N_star by N_wavelength array.
    for i in source_rows:
        if not np.isfinite(spectra[i]).all() or np.any(spectra[i] < 0):
            raise ValueError('Supported SPS spectra require finite nonnegative luminosities')
    redshift = finite(geometry['redshift_cosmological'], (n,), 'source redshift')
    distance = finite(geometry['luminosity_distance_kpc'], (n,), 'luminosity distance') * KPC_CM
    if np.any(redshift <= -1) or np.any(distance <= 0) or not np.isfinite(distance).all():
        raise ValueError('Physical source redshift and luminosity distance required')
    if geometry['namespace'] != sps_report['stellar_namespace']:
        raise ValueError('Flux geometry and SPS stellar namespaces differ')
    geometry_ids = exact_uint64_ids(geometry['particle_ids'], allow_empty=True)
    if not np.array_equal(geometry_ids, ids):
        raise ValueError('Observer fluxes require exact SPS identity/order join')
    if attenuation is not None:
        transmission = finite(attenuation, spectra.shape, 'dust attenuation')
        if np.any((transmission < 0) | (transmission > 1)):
            raise ValueError('Invalid dust attenuation')
        spectra = spectra * transmission

    flux = np.full((n, len(filters)), np.nan)
    supported = np.zeros(flux.shape, bool)
    for j, band in enumerate(filters):
        wavelength = increasing(band['wavelength_angstrom'], 'filter wavelengths')
        throughput = finite(band['transmission'], wavelength.shape, 'filter throughput')
        if wavelength[0] <= 0 or np.any(throughput < 0) or not band['source_sha256']:
            raise ValueError('Actual nonnegative hashed bandpass at positive wavelengths required')
        normalization = np.trapezoid(throughput / wavelength, wavelength)
        if not np.isfinite(normalization) or normalization <= 0:
            raise ValueError('Finite positive filter response required')
        active = throughput > 0
        for i in source_rows:
            rest = wavelength / (1 + redshift[i])
            if rest[active].min() < wave[0] or rest[active].max() > wave[-1]:
                continue
            flambda = np.interp(rest, wave, spectra[i], left=0, right=0) * LSUN_ERG_S / (4 * np.pi * distance[i]**2 * (1 + redshift[i]))
            value = np.trapezoid(flambda * wavelength * throughput, wavelength) / (C_ANGSTROM_S * normalization) / 1e-23
            if not np.isfinite(value):
                raise ValueError('Observer flux exceeds the representable numeric range')
            flux[i, j] = value
            supported[i, j] = True
    return {'truth_flux_jy': flux, 'flux_supported': supported, 'particle_ids': ids}


def noisy_survey(truth, request):
    """Draw correlated measurement noise without replacing intrinsic truth.

    Each source supplies a covariance in Jy^2. Draw only supported bands via
    V*sqrt(eigenvalues)*normal; tiny negative eigenvalues within the authored
    absolute tolerance are clipped to zero. Selection is a separate source mask;
    informative missingness marks observed bands across the parent catalogue.
    """
    from .data import exact_uint64_ids

    ids = exact_uint64_ids(truth['particle_ids'], allow_empty=True)
    flux = np.asarray(truth['truth_flux_jy'], float)
    valid = np.asarray(truth['flux_supported'])
    if flux.ndim != 2 or flux.shape[0] != len(ids) or valid.shape != flux.shape or valid.dtype.kind != 'b':
        raise ValueError('Survey flux matrix and Boolean support must align with source identities')
    if not np.isfinite(flux[valid]).all():
        raise ValueError('Supported truth fluxes must be finite')
    valid = valid.copy()  # Keep the returned catalogue mask independent of input truth.
    n, bands = flux.shape
    covariance = finite(request['measurement_covariance_jy2'], (n, bands, bands), 'survey covariance')
    tolerance = float(finite(request['covariance_tolerance_jy2'], (), 'covariance tolerance'))
    if tolerance < 0:
        raise ValueError('Covariance tolerance must be nonnegative in Jy^2')
    if not np.allclose(covariance, covariance.transpose(0, 2, 1), rtol=1e-10, atol=tolerance):
        raise ValueError('Survey covariance not symmetric')
    covariance = .5 * covariance + .5 * covariance.transpose(0, 2, 1)
    if np.any(np.linalg.eigvalsh(covariance) < -tolerance):
        raise ValueError('Survey covariance must be positive semidefinite')
    selection = request['selection']
    band = integer(selection['band_index'], 'selection band index', 0)
    if band >= bands:
        raise ValueError('Selection band index outside catalogue')
    limit = float(finite(selection['flux_limit_jy'], (), 'selection flux limit'))
    width = positive(selection['width_jy'], 'selection width')
    missing = request['missingness']
    intercept = finite(missing['intercept'], (bands,), 'missingness intercept')
    slope = finite(missing['flux_slope_per_jy'], (bands,), 'missingness slope')

    rng = seeded_rng(request['seed'], 'synthetic-survey')
    noisy = np.full_like(flux, np.nan)
    for i in range(n):
        available = np.flatnonzero(valid[i])
        if len(available):
            eigenvalues, vectors = np.linalg.eigh(covariance[i][np.ix_(available, available)])
            noisy[i, available] = flux[i, available] + vectors @ (np.sqrt(np.maximum(eigenvalues, 0)) * rng.normal(size=len(available)))
    if not np.isfinite(noisy[valid]).all():
        raise ValueError('Noisy supported flux exceeds the representable numeric range')
    probability = np.where(valid[:, band], expit((noisy[:, band] - limit) / width), 0.)
    selected = rng.random(n) < probability
    observed = valid & (rng.random((n, bands)) >= expit(intercept + slope * np.nan_to_num(noisy)))
    measured = np.where(observed, noisy, np.nan)
    report = {
        'schema': 'synthetic-survey-v1', 'seed': request['seed'], 'selection': selection, 'missingness': missing,
        'missingness_convention': 'informative logistic non-observation conditional on latent noisy measured flux; preserved in likelihood contract',
        'truth_mutated': False, 'science_acceptance': 'NOT_ACCEPTED',
        'units': {'truth_flux_jy': 'Jy', 'observed_flux_jy': 'Jy', 'measurement_covariance_jy2': 'Jy^2'},
    }
    return report, {
        'particle_ids': ids, 'truth_flux_jy': flux.copy(), 'truth_supported': valid,
        'observed_flux_jy': measured, 'observed_mask': observed, 'selected_mask': selected,
        'selection_probability': probability, 'measurement_covariance_jy2': covariance,
    }


OBSERVATION_KINDS = {'stellar_sps', 'stellar_dust', 'synthetic_survey', 'emission_lightcone', 'native_lightcone', 'lens_plane', 'multiple_plane_rays'}

def _bound_sources(inputs, record):
    if not record:
        raise ValueError('Observation science requires measured input sources')
    for name, expected in record.items():
        path = inputs.path(name)
        if digest(path) != expected:
            raise ValueError('Observation source changed: ' + name)

def execute_observation_stage(stage, inputs, working):
    """Dispatch an authored stage; preserve the caller's product/namespace contract.

    Native readers own unit conversions. SPS stays lazy so lens/lightcone-only
    workflows do not initialize FSPS. Inputs bind actual sources before use.
    """
    kind = stage['kind']
    parameters = stage['parameters']
    references = stage['inputs']
    if kind == 'stellar_sps':
        from .stellar_population import stellar_sps_truth
        from .stellar_population import synthesize_stellar_light
        if 'native_run_directory' in references:
            truth, sources = stellar_sps_truth(inputs.path(references['native_run_directory']), parameters['snapshot'], memory_bytes=int(inputs.resources['memory_gib'] * 1024 ** 3), namespace=parameters['namespace'])
            inputs.sources.update(sources)
        else:
            truth = inputs.json(references['stellar_truth'])
            _bound_sources(inputs, truth['birth_source_hashes'])
        synthesis = dict(parameters['synthesis'])
        backend = dict(synthesis['backend'])
        if 'data_source_sha256_file' in backend:
            backend['data_source_sha256'] = inputs.json(backend.pop('data_source_sha256_file'))
        backend['sps_home'] = str(inputs.path(backend['sps_home']))
        synthesis['backend'] = backend
        report, arrays = synthesize_stellar_light(truth, {**synthesis, 'memory_gib': inputs.resources['memory_gib']})
        inputs.sources.update(report['model']['source_hashes'])
        arrays['position_kpc'] = np.asarray(truth['position_kpc'])
    elif kind == 'stellar_dust':
        sps_report, sps = inputs.product(references['stellar_sps'])
        gas = inputs.json(references['native_gas'])
        _bound_sources(inputs, gas['source_hashes'])
        _bound_sources(inputs, parameters['dust']['opacity_table']['source_sha256'])
        report, arrays = gas_metal_dust_screen(sps['position_kpc'], parameters['observer_position_kpc'], gas, sps['wavelength_angstrom'], parameters['dust'])
        arrays['particle_ids'] = sps['particle_ids']
        report['sps_namespace'] = sps_report['namespace']
        report['sps_product_sha256'] = digest(inputs.path(references['stellar_sps']))
    elif kind == 'synthetic_survey':
        sps_report, sps = inputs.product(references['stellar_sps'])
        geometry = inputs.json(references['geometry'])
        filters = inputs.json(references['filters'])
        for band in filters:
            _bound_sources(inputs, band['source_sha256'])
        attenuation = None
        if 'dust' in references:
            dust, dust_arrays = inputs.product(references['dust'])
            if dust['sps_product_sha256'] != digest(inputs.path(references['stellar_sps'])) or not np.array_equal(dust_arrays['particle_ids'], sps['particle_ids']):
                raise ValueError('Dust and survey stellar identities differ')
            attenuation = dust_arrays['attenuation']
        truth = observer_fluxes(sps_report, sps, geometry, filters, attenuation=attenuation)
        report, arrays = noisy_survey(truth, parameters['survey'])
        report.update(filters=filters, namespace=sps_report['namespace'], dust='intrinsic_dust_free' if attenuation is None else dust['model'], stellar_truth_sha256=digest(inputs.path(references['stellar_sps'])), geometry=geometry)
    elif kind == 'emission_lightcone':
        clock = inputs.json(references['background_clock'])
        _bound_sources(inputs, clock['source_hashes'])
        _bound_sources(inputs, clock['distance_interpolation_certificate']['source_hashes'])
        background = background_lightcone(clock, parameters['observer']['observation_time_myr'], parameters['background'])
        tracks = inputs.json(references['native_tracks'])
        for row in tracks:
            _bound_sources(inputs, row['source_hashes'])
        report, arrays = emission_intersections(tracks, background, parameters['observer'], parameters['intersections'])
    elif kind == 'native_lightcone':
        report, arrays = read_gadget_lightcone(inputs.path(references['native_file']), parameters, maximum_particles=inputs.resources['maximum_particles'])
        inputs.sources.update(report['source_hashes'])
    elif kind == 'lens_plane':
        if 'native_run_directory' in references:
            native, values = read_native_gravitating_particles(inputs.path(references['native_run_directory']), parameters['snapshot'], species=parameters['species'], maximum_particles=inputs.resources['maximum_particles'], memory_gib=inputs.resources['memory_gib'], namespace=parameters['namespace'])
            particles = {**native, **values}
            inputs.sources.update(native['source_hashes'])
        else:
            particles = inputs.json(references['particles'])
        _bound_sources(inputs, particles['source_hashes'])
        report, arrays = species_lens_plane(particles, parameters['plane'])
    elif kind == 'multiple_plane_rays':
        planes = []
        for reference in references['planes']:
            report, values = inputs.product(reference)
            planes.append({'report': report, 'arrays': values})
        report, arrays = trace_multiple_planes(planes, inputs.json(references['rays_theta_rad']), parameters['source'], parameters['controls'])
    else:
        raise ValueError('Unknown observation stage')
    namespace = 'observation:' + stage['stage_id']
    if kind == 'stellar_sps':
        namespace = truth['namespace']
    elif kind == 'stellar_dust':
        namespace = report['sps_namespace']
    elif kind == 'synthetic_survey':
        namespace = sps_report['namespace']
    elif kind == 'native_lightcone':
        namespace = parameters['namespace']
    return (report, arrays, namespace, None)
