"""Finite MUSIC particle-window measurements and their native input responses.

A window selects particles in the original Lagrangian frame. Density modes
are dimensionless; velocity divergence uses km/s per comoving kpc/h. Responses
follow the actual source hierarchy, retained potentials and padded caches.
Their agreement alone establishes neither a noise covariance nor physical
boundary accuracy; those require the separate declared model and controls.
"""
import json
import math
import re
from pathlib import Path
import numpy as np
from core.music.replay import _budget, convolve_noise_reference

_CORE_ROOT = Path(__file__).resolve().parent.parent


def compensated_source_window_weights(cells,q_comoving_kpc_h,mode,source_shape,*,
        box_comoving_kpc_h,species_factor,mean_mass_over_window_mass,window_weights=None,memory_gib=1.):
    """Complex source weights for an original-q compensated mass window.

    cells and q represent *all* delivered fine particles of one species.
    Native compensation removes their full-population source mean, even when
    window_weights selects only a subset. Mass normalization is explicit;
    this does not substitute particle count for the actual window mass.

    If delta_m = f * mean_mass * (source - full_population_mean), the mode
    is sum(delta_m * selection * exp(-i k.q)) / actual_window_mass.
    Subtracting the mean of the selected phase weights gives its transpose.
    """
    cells=np.asarray(cells)
    q=np.asarray(q_comoving_kpc_h)
    mode=np.asarray(mode)
    shape=tuple(source_shape)
    if (cells.ndim!=2 or cells.shape[1]!=3 or cells.dtype.kind not in 'iu' or not len(cells)
            or q.shape!=cells.shape or q.dtype.kind not in 'fiu' or not np.isfinite(q).all()
            or mode.shape!=(3,) or mode.dtype.kind not in 'iu' or len(shape)!=3
            or any(type(n) is not int or n<=0 for n in shape)
            or not math.isfinite(box_comoving_kpc_h) or box_comoving_kpc_h<=0
            or not math.isfinite(species_factor) or not math.isfinite(mean_mass_over_window_mass) or mean_mass_over_window_mass<=0
            or np.any(cells<0) or np.any(cells>=shape) or np.any(q<0) or np.any(q>=box_comoving_kpc_h)):
        raise ValueError('Actual fine-cell ancestry, periodic original q and explicit mass/window frame required')
    _budget(memory_gib,64*1024**2+64*math.prod(shape)+128*len(cells))
    selected=np.ones(len(cells)) if window_weights is None else np.asarray(window_weights)
    if selected.shape!=(len(cells),) or selected.dtype.kind not in 'fiub' or not np.isfinite(selected).all() or np.any(selected<0) or not np.any(selected>0):
        raise ValueError('Finite nonempty nonnegative original fine-population window required')
    phase=selected*np.exp(-2j*np.pi*(q@mode)/box_comoving_kpc_h)
    # This is the transpose of native compensation over all delivered fine
    # particles, not a residual mean fitted to the observed Fourier mode.
    phase-=np.sum(phase)/len(cells)
    result=np.zeros(shape,dtype=complex)
    np.add.at(result,tuple(cells.T),phase*species_factor*mean_mass_over_window_mass)
    return result


def convolve_noise_adjoint(weights,amplitude,*,patch_length,shift_cells=0.,memory_gib=1.):
    """Real Euclidean adjoint of signed native convolution and staggering."""
    return convolve_noise_reference(weights,amplitude,patch_length=patch_length,
        shift_cells=-shift_cells,memory_gib=memory_gib)


def fourier_splice_adjoint(weights,parent_shape,coarse_start,*,periodic_parent=False,memory_gib=1.):
    """Return parent/fine Euclidean responses of the native padded splice.

    Fourier interpolation multiplies the parent spectrum by8; its adjoint
    multiplies by1 because the fine inverse-transform volume is8 times larger.
    The taper vanishes on every ambiguous coarse Nyquist boundary.
    """
    values=np.asarray(weights)
    shape=tuple(parent_shape)
    start=np.asarray(coarse_start)
    if (values.ndim!=3 or values.dtype.kind not in 'fiu' or any(n<8 or n%4 for n in values.shape)
            or len(shape)!=3 or any(type(n) is not int or n<1 for n in shape)
            or start.shape!=(3,) or start.dtype.kind not in 'iu' or type(periodic_parent) is not bool):
        raise ValueError('Finite real splice weights and actual parent geometry required')
    _budget(memory_gib,64*1024**2+256*(values.size+math.prod(shape)))
    if not np.isfinite(values).all():
        raise ValueError('Finite splice weights required')
    extent=np.asarray(values.shape)//2
    if np.any(extent>shape) or not periodic_parent and (np.any(start<0) or np.any(start+extent>shape)):
        raise ValueError('Splice adjoint footprint escapes actual parent')
    # Use full complex transforms here, independently of the forward rFFT
    # bookkeeping. This includes both signs of every represented real mode.
    axes=[]
    indices=[]
    for axis,n in enumerate(extent):
        k=np.fft.fftfreq(int(n))*n
        broadcast=[1,1,1]
        broadcast[axis]=len(k)
        axes.append(k.reshape(broadcast))
        indices.append(np.rint(k).astype(int)%values.shape[axis])
    blend=np.ones(tuple(extent))
    phase=np.zeros(tuple(extent))
    for k,n in zip(axes,extent):
        ratio=np.abs(k)/(int(n)//4)
        blend*=np.where(ratio<=1/3,1.,np.where(ratio<2/3,np.cos(np.pi/2*(3*ratio-1)),0.))
        phase-=np.pi/2*k/n
    spectrum=np.fft.fftn(values)
    selection=np.ix_(*indices)
    parent_spectrum=spectrum[selection]*blend*np.exp(-1j*phase)
    spectrum[selection]*=1-blend
    fine=np.fft.ifftn(spectrum).real
    footprint=np.fft.ifftn(parent_spectrum).real
    parent=np.zeros(shape)
    parent[np.ix_(*[(int(o)+np.arange(int(n)))%size for o,n,size in zip(start,extent,shape)])]+=footprint
    return parent,fine


def prepared_source_adjoint(weights,mesh,configuration,amplitude,*,gas=False,memory_gib=1.):
    """Push a real weighted sum of all source levels back to cached noise.

    weights is a dict level->real array on the physical Poisson hierarchy.
    Results have the actual padded noise shapes. DC, overwritten parent cells,
    cut geometry, parent-mode dependence and gas staggering remain explicit.
    This provides finite response rows; a covariance still requires the actual
    correlated noise-generation model, not an identity covariance by default.
    """
    cfg=configuration
    low=cfg.getint('setup','levelmin')
    high=cfg.getint('setup','levelmax')
    tf=mesh['hierarchies']['convolution']
    physical=mesh['hierarchies']['poisson']
    margin=mesh['convolution_margin']
    if (type(gas) is not bool or type(margin) is not int or margin<=0 or margin%2
            or cfg.getint('setup','levelmin_TF',fallback=low)!=low
            or not cfg.getboolean('setup','fourier_splicing',fallback=True)
            or cfg.getboolean('setup','fix_mode_amplitude',fallback=False)
            or cfg.getboolean('setup','flip_mode_amplitude',fallback=False)
            or any([row['level'] for row in rows]!=list(range(low,high+1)) for rows in (tf,physical))
            or set(weights)!=set(range(low,high+1))):
        raise ValueError('Actual default Fourier hierarchy and aligned real weights required')
    shapes={row['level']:tuple(n+(0 if row['level']==low else 2*margin) for n in row['shape']) for row in tf}
    _budget(memory_gib,64*1024**2+640*sum(math.prod(s) for s in shapes.values())+64*sum(math.prod(row['shape']) for row in physical))
    work=[]
    for row in physical:
        value=np.asarray(weights[row['level']])
        if value.shape!=tuple(row['shape']) or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError('Finite aligned physical source-window weights required')
        work.append(value.astype(float,copy=True))
    # Transpose the one shared composite-source DC projection first.
    total=sum(float(np.sum(w,dtype=np.longdouble)) for w in work)
    work[0]-=total/work[0].size
    # Forward restriction ran finest->base; its transpose runs base->finest.
    for i in range(1,len(work)):
        child,parent=physical[i],physical[i-1]
        start=(np.asarray(child['offset'])//2-np.asarray(parent['offset']))%2**parent['level']
        extent=np.asarray(child['shape'])//2
        if np.any(np.asarray(child['offset'])%2) or np.any(np.asarray(child['shape'])%2):
            raise ValueError('Complete child restriction cells required')
        if tuple(parent['shape'])!=(2**parent['level'],)*3 and np.any(start+extent>parent['shape']):
            raise ValueError('Physical adjoint child escapes parent')
        selection=np.ix_(*[(int(o)+np.arange(int(n)))%size for o,n,size in zip(start,extent,parent['shape'])])
        spread=work[i-1][selection]/8.
        for axis in range(3):
            spread=np.repeat(spread,2,axis=axis)
        work[i]+=spread
        work[i-1][selection]=0.
    padded=[]
    origins=[]
    for w,row,target in zip(work,tf,physical):
        pad=0 if row['level']==low else margin
        start=np.asarray(target['offset'])-np.asarray(row['offset'])+pad
        shape=shapes[row['level']]
        if np.any(start<pad) or np.any(start+target['shape']>np.asarray(shape)-pad):
            raise ValueError('Adjoint cut escapes original convolution patch')
        values=np.zeros(shape)
        values[tuple(slice(int(o),int(o)+n) for o,n in zip(start,target['shape']))]=w
        padded.append(values)
        origins.append(np.asarray(row['offset'])-pad)
    result={}
    box=cfg.getfloat('setup','boxlength')
    for i in range(len(padded)-1,-1,-1):
        level=tf[i]['level']
        w=padded[i]
        if i:
            if np.any(origins[i]%2):
                raise ValueError('Aligned padded source origins required')
            parent,w=fourier_splice_adjoint(w,padded[i-1].shape,origins[i]//2-origins[i-1],
                periodic_parent=i==1,memory_gib=memory_gib)
            padded[i-1]+=parent
        shift=cfg.getfloat('setup','baryon_staggering',fallback=.5)*2.**(level-high) if gas else 0.
        result[level]=convolve_noise_adjoint(w,amplitude,patch_length=box*w.shape[0]/2**level,
            shift_cells=shift,memory_gib=memory_gib)
    return result,{'schema':'native-prepared-source-adjoint-v1','levels':list(range(low,high+1)),
        'response_coordinates':'Actual retained padded noise cells; real Euclidean weights',
        'global_finest_grid_allocated':False,'noise_covariance':'NOT_ASSUMED; native parent/fine and periodic overlap correlations required',
        'launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}


def validate_compensated_window_request(request):
    from core.zoom import _modes
    if (not isinstance(request,dict) or set(request)!={'schema','mesh','noise_directory','modes','windows'}
            or request['schema']!='native-music-compensated-window-request-v1'
            or not all(isinstance(request[name],str) and request[name] for name in ('mesh','noise_directory'))
            or not isinstance(request['windows'],list) or not request['windows']):
        raise ValueError('Explicit original compensated-window request required')
    modes=_modes(request['modes'])
    if np.any(np.abs(modes)>2**53-1):
        raise ValueError('Exactly representable integer window modes required')
    names=set()
    for window in request['windows']:
        if (not isinstance(window,dict) or not isinstance(window.get('id'),str)
                or not re.fullmatch(r'[a-z][a-z0-9-]{0,63}',window['id']) or window['id'] in names):
            raise ValueError('Distinct explicit window identities required')
        names.add(window['id'])
        if window.get('kind')=='all_fine' and set(window)=={'id','kind'}:
            continue
        if window.get('kind')!='cell_box' or set(window)!={'id','kind','lower_cell','upper_cell'}:
            raise ValueError('Declare all_fine or an explicit original computational cell box')
        lower=np.asarray(window['lower_cell'])
        upper=np.asarray(window['upper_cell'])
        if (lower.shape!=(3,) or upper.shape!=(3,) or lower.dtype.kind not in 'iu' or upper.dtype.kind not in 'iu'
                or np.any(lower<0) or np.any(upper<=lower)):
            raise ValueError('Integer nonempty source-cell window bounds required')


def measure_native_compensated_windows(index,configuration,ic,request,destination,*,memory_gib=1.):
    """Publish mass-normalized modes and explicit mass-conditioning responses.

    Does not assume a covariance model. Noise response rows retain every
    actual cached level; their native correlations must be supplied separately.
    For baryon fraction f_b, compensated gas masses use +(1-f_b) and dark
    matter masses use -f_b, so the combined compensation conserves mass.
    """
    import h5py
    from core.foundation import file_sha256
    from core.data import snapshot_total_counts,export_numeric_audit,epoch_contract
    from core.zoom import _configuration,_modes,_native_cell_levels
    from core.music.replay import transfer_replay_input_bindings,load_native_noise,dms_transfer_amplitude
    validate_compensated_window_request(request)
    configuration,ic,index=map(lambda p:Path(p).resolve(),(configuration,ic,index))
    pins=transfer_replay_input_bindings({'configuration':str(configuration),'source_index':str(index),'product':str(ic),'request':request})
    cfg=_configuration(configuration)
    mesh=json.loads(Path(request['mesh']).read_text())
    archive=json.loads(index.read_text())
    low,high=cfg.getint('setup','levelmin'),cfg.getint('setup','levelmax')
    physical=mesh['hierarchies']['poisson']
    tf=mesh['hierarchies']['convolution']
    if (mesh.get('schema')!='darkmattersim-music-mesh-v1' or mesh.get('scalar_bytes')!=8 or not 2<=low<high<=20
            or any([row['level'] for row in rows]!=list(range(low,high+1)) for rows in (physical,tf))):
        raise ValueError('Actual original nested binary64 mesh required')
    for field in archive['fields']:
        if [(row['level'],row['shape'],row['origin']) for row in field['grids']]!=[(row['level'],row['shape'],row['offset']) for row in physical]:
            raise ValueError('Original compensated-window mesh differs from retained source geometry')
    modes=_modes(request['modes']).astype(np.int64)
    fine=physical[-1]
    shape=tuple(fine['shape'])
    margin=mesh['convolution_margin']
    noises_shape={row['level']:tuple(n+(0 if row['level']==low else 2*margin) for n in row['shape']) for row in tf}
    cells=sum(math.prod(s) for s in noises_shape.values())
    maximum_rows=len(request['windows'])*(4*len(modes)+2)
    with h5py.File(ic) as f:
        h=f['Header'].attrs
        counts=snapshot_total_counts(h)
        box=float(h['BoxSize'])
        a=float(h['Time'])
        shift=np.asarray(h['Music_output_coordinate_shift_box_fraction'],dtype=float)
        means=np.asarray(h['Music_mean_species_masses_native'],dtype=float)
        fb=float(h['OmegaBaryon'])/float(h['Omega0'])
        if (h.get('Music_lpt_order')!=2 or h.get('Music_mixed_fluid_schema')!=1 or h.get('Music_composite_source_normalization_schema')!=1
                or h.get('Music_levelmin')!=low or h.get('Music_levelmax')!=high or h.get('NumFilesPerSnapshot')!=1
                or not np.array_equal(counts,h['NumPart_ThisFile']) or not all(counts[:2]) or shift.shape!=(3,) or not np.isfinite(shift).all()
                or means.shape!=(6,) or not np.isfinite(means).all() or np.any(means[:2]<=0) or not 0<fb<1
                or not math.isclose(a,1/(1+cfg.getfloat('setup','zstart')),rel_tol=1e-12)
                or not math.isclose(box,1000*cfg.getfloat('setup','boxlength'),rel_tol=1e-12)
                or not math.isclose(fb,cfg.getfloat('cosmology','Omega_b')/cfg.getfloat('cosmology','Omega_m'),rel_tol=1e-12)):
            raise ValueError('Actual original mixed-fluid IC frame differs from window configuration')
    _budget(memory_gib,64*1024**2+(768+16*maximum_rows)*cells+256*int(sum(counts[:2])))
    for window in request['windows']:
        if window['kind']=='cell_box' and np.any(np.asarray(window['upper_cell'])>shape):
            raise ValueError('Declared cell window escapes actual finest source patch')
    table=Path(cfg.get('cosmology','transfer_file'))
    if not table.is_absolute():
        table=configuration.parent/table
    amplitude=dms_transfer_amplitude(np.loadtxt(table),cfg,'compensation')
    noises={level:load_native_noise(Path(request['noise_directory'])/f'wnoise_{level:04d}.bin',s,byte_order=archive['byte_order'],memory_gib=memory_gib) for level,s in noises_shape.items()}
    responses={level:[] for level in noises}
    channels=[]
    checks=[]
    windows=[]
    arrays={}
    specs={}
    with h5py.File(ic) as f:
        for species in (0,1):
            group=f[f'PartType{species}']
            n=int(counts[species])
            ids=np.asarray(group['ParticleIDs'])
            centres=np.asarray(group['LagrangianCellCenters'])
            q=np.asarray(group['LagrangianCoordinates'])
            if ids.shape!=(n,) or ids.dtype.kind!='u' or ids.dtype.itemsize!=8:
                raise ValueError('Exact aligned original uint64 particle identities required')
            if any(value.shape!=(n,3) or not np.isfinite(value).all() or np.any(value<0) or np.any(value>=box) for value in (centres,q)):
                raise ValueError('Finite periodic original material centres and q required')
            levels=_native_cell_levels(np.asarray(group['LagrangianCellWidths']),box,low,high)
            if np.any(levels!=high):
                raise ValueError('Original delivered gas/DM fine populations required')
            grid=(centres/box-shift)%1*2**high-.5
            rounded=np.rint(grid)
            if np.any(np.abs(grid-rounded)>1e-9):
                raise ValueError('Original material cells do not align with source mesh')
            cell=(rounded.astype(np.int64)-np.asarray(fine['offset']))%2**high
            if np.any(cell>=shape):
                raise ValueError('Original particle ancestry escapes finest source')
            mass=np.asarray(group['Masses']) if 'Masses' in group else np.full(n,float(f['Header'].attrs['MassTable'][species]))
            if mass.shape!=(n,) or not np.isfinite(mass).all() or np.any(mass<=0):
                raise ValueError('Actual positive aligned native masses required')
            mean=float(means[species])
            factor=1-fb if species==0 else -fb
            idkey=f'species{species}_particle_ids'
            arrays[idkey]=ids
            specs[idkey]={'units':'exact native uint64 particle identity','quantity':idkey,'identity_namespace':f'native-music-ic:{pins[str(ic)]}:PartType{species}'}
            for window in request['windows']:
                selected=np.ones(n,dtype=bool) if window['kind']=='all_fine' else np.all((cell>=window['lower_cell'])&(cell<window['upper_cell']),axis=1)
                number=int(selected.sum())
                if not number:
                    raise ValueError('Requested original particle window is empty')
                actual_mass=float(mass[selected].sum())
                reference_mass=number*mean
                if not math.isfinite(actual_mass) or not math.isfinite(reference_mass):
                    raise ValueError('Original window mass overflow')
                prefix=f'species{species}_{window["id"]}'
                arrays[prefix+'_selection']=selected
                specs[prefix+'_selection']={'units':'dimensionless boolean support','quantity':'Selection in same-row exact original particle ID array'}
                windows.append({'particle_type':species,'window_id':window['id'],'definition':window,'selected_particles':number,
                    'actual_window_mass_native':actual_mass,'reference_window_mass_native':reference_mass,'mass_conditioning':'DETERMINISTIC_FULL_FINE_TOTAL' if number==n else 'EXPLICIT_MASS_RESPONSE_ROW'})
                def append_response(mode,normalization,kind):
                    weight=compensated_source_window_weights(cell,q,mode,shape,box_comoving_kpc_h=box,species_factor=factor,
                        mean_mass_over_window_mass=mean/normalization,window_weights=selected,memory_gib=memory_gib)
                    prediction=0j
                    row_indices=[]
                    for channel,part in (('real',weight.real),('imag',weight.imag)):
                        if kind=='mass_normalization' and channel=='imag':
                            continue
                        physical_weights={row['level']:np.zeros(row['shape']) for row in physical}
                        physical_weights[high]=part
                        response,_=prepared_source_adjoint(physical_weights,mesh,cfg,amplitude,gas=species==0,memory_gib=memory_gib)
                        value=sum(float(np.sum(response[level]*noises[level])) for level in response)
                        prediction+=value*(1 if channel=='real' else 1j)
                        row_indices.append(len(channels))
                        channels.append({'particle_type':species,'window_id':window['id'],'kind':kind,'mode':list(map(int,mode)),
                            'channel':channel,'normalizing_mass_native':normalization})
                        for level in response:
                            responses[level].append(response[level])
                    phase=np.exp(-2j*np.pi*(q@np.asarray(mode))/box)
                    measured=np.sum((mass-mean)*selected*phase)/normalization
                    difference=abs(prediction-measured)
                    norm=abs(measured)
                    error=difference/norm if norm else 0. if difference==0 else math.inf
                    checks.append({'particle_type':species,'window_id':window['id'],'kind':kind,'mode':list(map(int,mode)),
                        'response_rows':row_indices,'measured':[measured.real,measured.imag],'predicted':[prediction.real,prediction.imag],
                        'relative_l2_error':error if math.isfinite(error) else None,'status':'PASS' if error<=1e-7 else 'FAIL'})
                for mode in modes:
                    append_response(mode,actual_mass,'compensated_mass_mode')
                if number!=n:
                    append_response([0,0,0],reference_mass,'mass_normalization')
    for level,values in responses.items():
        key=f'response_noise_level{level}'
        arrays[key]=np.asarray(values)
        specs[key]={'units':'dimensionless real mode or mass contrast per unit native noise scalar','quantity':'Rows follow explicit channel metadata; columns follow actual native padded cache cells'}
    for name in (
        'music/windows.py',
        'music/replay.py',
        'zoom.py',
    ):
        path=(_CORE_ROOT / name)
        pins[str(path.resolve())]=file_sha256(path)
    report={'schema':'native-compensated-window-response-v1','status':'PASS' if checks and all(row['status']=='PASS' for row in checks) else 'FAIL',
        'windows':windows,'channels':channels,'checks':checks,'mesh':mesh,'output_shift_box_fraction':shift.tolist(),
        'maximum_relative_operator_error':1e-7,'scope':'Original compensated-mass component only, with configurable exact source-cell windows/modes and explicit normalization-mass response rows. Full density/velocity and physical boundary acceptance remain separate.',
        'noise_covariance':'NOT_ASSUMED; requires bound native generation model and explicit aperture-mass conditioning',
        'launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
    return export_numeric_audit(destination,report,arrays,specs,algorithm=__file__,context={'epoch':epoch_contract(a)},source_hashes=pins)


def validate_flow_window_request(request):
    from core.zoom import _modes
    if (not isinstance(request,dict) or set(request)-{'response_workspace'}!={'schema','mesh','noise_directory','modes','quantities','windows'}
            or request['schema']!='native-music-flow-window-request-v1'
            or not all(isinstance(request[k],str) and request[k] for k in ('mesh','noise_directory'))
            or not isinstance(request['quantities'],list) or not request['quantities']
            or any(q not in ('density','velocity') for q in request['quantities']) or len(set(request['quantities']))!=len(request['quantities'])
            or not isinstance(request['windows'],list) or not request['windows']):
        raise ValueError('Explicit original flow-window request required')
    if 'response_workspace' in request:
        from core.numeric import validate_response_workspace
        validate_response_workspace(request['response_workspace'])
    modes=_modes(request['modes'])
    if np.any(np.abs(modes)>2**53-1):
        raise ValueError('Exactly representable integer flow modes required')
    names=set()
    for window in request['windows']:
        if (not isinstance(window,dict) or set(window)!={'id','particle_types','refinement_levels'}
                or not isinstance(window['id'],str) or not re.fullmatch(r'[a-z][a-z0-9-]{0,63}',window['id']) or window['id'] in names):
            raise ValueError('Distinct explicit species/refinement window identities required')
        names.add(window['id'])
        types=window['particle_types']
        levels=window['refinement_levels']
        if not isinstance(types,list) or not types or any(type(t) is not int or t not in (0,1,2) for t in types) or len(set(types))!=len(types):
            raise ValueError('Exact distinct original matter species selectors required')
        if levels is not None and (not isinstance(levels,list) or not levels or any(type(l) is not int or not 2<=l<=20 for l in levels) or len(set(levels))!=len(levels)):
            raise ValueError('Distinct dyadic refinement levels or null for all required')


def measure_native_flow_windows(index,configuration,ic,request,destination,*,memory_gib=2.):
    """Replay full measured windows with separate source/potential coefficients.

    Captured potential ghosts remain explicit inputs, not independent noise
    variables or a physical boundary certificate. Density includes the native
    fine-species compensated mass term; velocity uses retained first-order v.

    With phase exp(-i k.q), density is -i k.Psi and velocity divergence is
    +i k.v. Positions and k use comoving kpc/h; the native velocity source
    uses comoving Mpc/h, so its reconstruction length differs by 1000.
    Both observables divide by the actual selected total mass.
    """
    import h5py
    from core.foundation import file_sha256
    from core.data import export_numeric_audit,epoch_contract,snapshot_total_counts
    from core.zoom import _configuration,_modes,_native_cell_levels,linear_density_modes,velocity_divergence_modes
    from core.music.archive import load_music_source_hierarchy,potential_weighted_sum
    from core.music.replay import transfer_replay_input_bindings
    from core.music.adjoint import gradient_hierarchy_adjoint
    from core.numeric import NumericArraySpool, preflight_response_disk
    validate_flow_window_request(request)
    ic,configuration,index=map(lambda p:Path(p).resolve(),(ic,configuration,index))
    pins=transfer_replay_input_bindings({'configuration':str(configuration),'source_index':str(index),'product':str(ic),'request':request})
    cfg=_configuration(configuration)
    mesh=json.loads(Path(request['mesh']).read_text())
    record=json.loads(index.read_text())
    low,high=cfg.getint('setup','levelmin'),cfg.getint('setup','levelmax')
    modes=_modes(request['modes']).astype(np.int64)
    physical=mesh['hierarchies']['poisson']
    if (not 2<=low<high<=20 or mesh.get('scalar_bytes')!=8 or mesh.get('schema')!='darkmattersim-music-mesh-v1'
            or [row['level'] for row in physical]!=list(range(low,high+1))):raise ValueError('Actual original nested source mesh required')
    for field in record['fields']:
        if field.get('scalar_bytes',8)!=(mesh.get('source_archive_potential_scalar_bytes',8) if field['quantity_code'] in (2,4,7) else 8):
            raise ValueError('Captured potential precision differs from native mesh plan')
        if [(r['level'],r['shape'],r['origin']) for r in field['grids']]!=[(r['level'],r['shape'],r['offset']) for r in physical]:
            raise ValueError('Flow window mesh differs from captured source geometry')
    for window in request['windows']:
        if window['refinement_levels'] is not None and any(not low<=level<=high for level in window['refinement_levels']):
            raise ValueError('Window level lies outside actual native hierarchy')
    with h5py.File(ic) as f:
        h=f['Header'].attrs
        counts=snapshot_total_counts(h)
        box=float(h['BoxSize'])
        a=float(h['Time'])
        shift=np.asarray(h['Music_output_coordinate_shift_box_fraction'],dtype=float)
        means=np.asarray(h['Music_mean_species_masses_native'],dtype=float)
        fb=float(h['OmegaBaryon'])/float(h['Omega0'])
        if (h.get('Music_lpt_order')!=2 or h.get('Music_mixed_fluid_schema')!=1 or h.get('Music_composite_source_normalization_schema')!=1
                or h.get('Music_levelmin')!=low or h.get('Music_levelmax')!=high or h.get('NumFilesPerSnapshot')!=1
                or not np.array_equal(counts,h['NumPart_ThisFile']) or np.any(counts[3:]) or not all(counts[:2])
                or shift.shape!=(3,) or not np.isfinite(shift).all() or means.shape!=(6,) or not np.isfinite(means).all() or np.any(means[:2]<=0)
                or not 0<fb<1 or not math.isclose(fb,cfg.getfloat('cosmology','Omega_b')/cfg.getfloat('cosmology','Omega_m'),rel_tol=1e-12)
                or not math.isclose(a,1/(1+cfg.getfloat('setup','zstart')),rel_tol=1e-12)
                or not math.isclose(box,1000*cfg.getfloat('setup','boxlength'),rel_tol=1e-12)):
            raise ValueError('Original mixed-fluid flow-window frame/configuration mismatch')
    raw_cells=sum(math.prod(r['stored_shape']) for field in record['fields'] for r in field['grids'])
    boundary=cfg.getint('poisson','hybrid_extension_boundary',fallback=32) if cfg.get('poisson','hybrid_source_extension',fallback='none')=='bounded_conservative' else 32
    streamed='response_workspace' in request
    from core.music.resources import source_response_resources
    resources=source_response_resources(mesh,[int(n) for n in counts],request['windows'],mode_count=len(modes),
        quantity_count=len(request['quantities']),boundary=boundary)
    if resources['raw_source_cells']!=raw_cells:
        raise ValueError('Captured source layout differs from original14-field allocation model')
    _budget(memory_gib,resources['streamed_memory_guard_bytes'] if streamed else resources['legacy_memory_guard_bytes'])
    publication_preflight=None
    if streamed:
        publication_preflight=preflight_response_disk(destination,request['response_workspace']['disk_gib'],
            payload_bytes=resources['payload_upper_bound_bytes'],array_count=resources['array_count_upper_bound'])
    fields={}
    for preparation in (0,1):
        for label in request['quantities']:
            code,pc=(1,2) if label=='density' else (6,7)
            source,_=load_music_source_hierarchy(index,code,preparation,memory_gib=memory_gib)
            potential,_=load_music_source_hierarchy(index,pc,preparation,include_ghosts=True,memory_gib=memory_gib)
            fields[preparation,label]=(source,potential)
        if 'density' in request['quantities']:
            value,_=load_music_source_hierarchy(index,5,preparation,memory_gib=memory_gib)
            fields[preparation,'compensation']=value[-1]
    arrays=NumericArraySpool(Path(destination).with_name(Path(destination).name+'.rows'),
        disk_gib=request['response_workspace']['disk_gib'],memory_gib=memory_gib) if streamed else {}
    specs={}
    checks=[]
    channels=[]
    windows=[]
    populations={}
    with h5py.File(ic) as f:
        for species in (0,1,2):
            if not counts[species]:
                continue
            group=f[f'PartType{species}']
            n=int(counts[species])
            q=np.asarray(group['LagrangianCoordinates'])
            centres=np.asarray(group['LagrangianCellCenters'])
            ids=np.asarray(group['ParticleIDs'])
            mass=np.asarray(group['Masses']) if 'Masses' in group else np.full(n,float(f['Header'].attrs['MassTable'][species]))
            levels=_native_cell_levels(np.asarray(group['LagrangianCellWidths']),box,low,high)
            if (ids.shape!=(n,) or ids.dtype.kind!='u' or ids.dtype.itemsize!=8 or mass.shape!=(n,) or not np.isfinite(mass).all() or np.any(mass<=0)
                    or any(v.shape!=(n,3) or not np.isfinite(v).all() or np.any(v<0) or np.any(v>=box) for v in (q,centres))):
                raise ValueError('Actual aligned exact identities, original coordinates and positive masses required')
            if species in (0,1) and np.any(levels!=high):
                raise ValueError('Original fine gas/DM population required')
            populations[species]=(q,centres,mass,levels)
            key=f'species{species}_particle_ids'
            arrays[key]=ids
            specs[key]={'units':'identity','quantity':'Exact original native HDF5 row identity',
                'identity_namespace':f'native-music-ic:{pins[str(ic)]}:PartType{species}','ancestry':{'relation':'same actual IC particle; no cross-run identity inference'}}
    for window in request['windows']:
        selections={species:(np.ones(len(populations[species][0]),dtype=bool) if window['refinement_levels'] is None else np.isin(populations[species][3],window['refinement_levels'])) for species in window['particle_types'] if species in populations}
        total=sum(float(populations[species][2][mask].sum()) for species,mask in selections.items())
        if not math.isfinite(total) or total<=0:
            raise ValueError('Requested original species/refinement window is empty')
        windows.append({'definition':window,'actual_total_mass_native':total,'actual_particles':sum(int(m.sum()) for m in selections.values())})
        for species,mask in selections.items():
            key=window['id']+f'_species{species}_selection'
            arrays[key]=mask
            specs[key]={'units':'dimensionless boolean support','quantity':'Selection in same-row exact original species ID array'}
        for label in request['quantities']:
            for mode in modes:
                weights={preparation:{row['level']:np.zeros(row['values'].shape+(3,),dtype=complex) for row in fields[preparation,label][0]} for preparation in (0,1)}
                compensation_weights={}
                # Native gradient outputs are box-normalized displacements
                # or velocity/box-length; restore the appropriate length once.
                k=2*np.pi*mode/box
                length=box if label=='density' else cfg.getfloat('setup','boxlength')
                sign=-1j if label=='density' else 1j
                for species,selected in selections.items():
                    if not selected.any():
                        continue
                    q,centres,mass,levels=populations[species]
                    reference=np.full(int(selected.sum()),means[species]) if species in (0,1) else mass[selected]
                    coefficient=sign*np.exp(-1j*(q[selected]@k))[:,None]*k[None,:]*(reference/total)[:,None]*length
                    active_levels=levels[selected]
                    active_centres=centres[selected]
                    for level in np.unique(active_levels):
                        use=active_levels==level
                        grid=(active_centres[use]/box-shift)%1*2**int(level)-.5
                        rounded=np.rint(grid)
                        if np.any(np.abs(grid-rounded)>1e-9):
                            raise ValueError('Window material centres are misaligned with original source cells')
                        preparations=[(species,1.)] if species in (0,1) else [(0,fb),(1,1-fb)]
                        for preparation,fraction in preparations:
                            row=next(r for r in fields[preparation,label][0] if r['level']==level)
                            cell=(rounded.astype(np.int64)-row['origin'])%2**int(level)
                            if np.any(cell>=row['values'].shape):
                                raise ValueError('Original window ancestry escapes source patch')
                            np.add.at(weights[preparation][int(level)],tuple(cell.T),coefficient[use]*fraction)
                    if label=='density' and species in (0,1):
                        row=fields[species,'compensation']
                        grid=(centres/box-shift)%1*2**high-.5
                        cell=(np.rint(grid).astype(np.int64)-row['origin'])%2**high
                        compensation_weights[species]=compensated_source_window_weights(cell,q,mode,row['values'].shape,box_comoving_kpc_h=box,
                            species_factor=1.,mean_mass_over_window_mass=float(means[species])/total,window_weights=selected,memory_gib=memory_gib)
                prediction=0j
                channel_indices=[]
                for part in ('real','imag'):
                    channel_index=len(channels)
                    channel_indices.append(channel_index)
                    value=0.
                    for preparation in (0,1):
                        if not any(np.any(w) for w in weights[preparation].values()):
                            continue
                        source,potential=fields[preparation,label]
                        sw,pw,_=gradient_hierarchy_adjoint({level:getattr(w,part) for level,w in weights[preparation].items()},source,potential,cfg,gas=preparation==0,memory_gib=memory_gib)
                        value+=sum(float(np.sum(r['values']*sw[r['level']])) for r in source)+sum(potential_weighted_sum(r['values'],pw[r['level']],memory_gib=memory_gib) for r in potential)
                        for kind,rows,response in (('source',source,sw),('potential',potential,pw)):
                            for row in rows:
                                key=f'channel{channel_index}_prepared{preparation}_{kind}_level{row["level"]}'
                                arrays[key]=response[row['level']]
                                specs[key]={'units':'dimensionless density mode per unit source/potential' if label=='density' else 'km/s per comoving kpc/h per (km/s per comoving Mpc/h source/potential)',
                                    'quantity':'Original source or actual ghost-potential response coefficient '+key}
                    for preparation,w in compensation_weights.items():
                        scalar=getattr(w,part)
                        value+=float(np.sum(scalar*fields[preparation,'compensation']['values']))
                        key=f'channel{channel_index}_prepared{preparation}_compensated_source'
                        arrays[key]=scalar
                        specs[key]={'units':'dimensionless density mode per unit prepared compensated source','quantity':key}
                    prediction+=value*(1 if part=='real' else 1j)
                    channels.append({'window_id':window['id'],'quantity':label,'mode':mode.tolist(),'component':part})
                function=linear_density_modes if label=='density' else velocity_divergence_modes
                measured=complex(function(ic,[mode],particle_types=tuple(window['particle_types']),refinement_levels=None if window['refinement_levels'] is None else tuple(window['refinement_levels']),memory_gib=memory_gib/2)[0])
                difference=abs(prediction-measured)
                norm=abs(measured)
                error=difference/norm if norm else 0. if difference==0 else math.inf
                checks.append({'window_id':window['id'],'quantity':label,'mode':mode.tolist(),'response_channels':channel_indices,
                    'measured':[measured.real,measured.imag],'reference':[prediction.real,prediction.imag],'relative_l2_error':error if math.isfinite(error) else None,'status':'PASS' if error<=1e-7 else 'FAIL'})
    from core.music.archive import archive_precision_sources
    pins.update(archive_precision_sources(record))
    for module in (
        'music/windows.py',
        'music/adjoint.py',
        'music/archive.py',
        'music/replay.py',
        'zoom.py',
        'music/resources.py',
        'numeric.py',
        'data.py',
        'foundation.py',
    ):
        path=(_CORE_ROOT / module).resolve()
        pins[str(path)]=file_sha256(path)
    result={'schema':'native-music-full-flow-window-response-v1','status':'PASS' if checks and all(c['status']=='PASS' for c in checks) else 'FAIL',
        'windows':windows,'channels':channels,'checks':checks,'source_archive_index':{'path':str(index),'sha256':pins[str(index)]},
        'maximum_relative_operator_error':1e-7,'scope':'Original full linear density/velocity window assembly replay from immutable sources and captured potential/ghost inputs. Supports configured species/refinement mixtures, original mass/q frames and fine compensated density.',
        'response_resources':resources,'potential_noise_coupling':'NOT_ESTABLISHED','physical_boundary_error_budget':'NOT_ESTABLISHED','launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
    if streamed:
        result['response_publication']={**arrays.metadata(),'preflight':publication_preflight}
    return export_numeric_audit(destination,result,arrays,specs,algorithm=__file__,context={'epoch':epoch_contract(a)},source_hashes=pins)
