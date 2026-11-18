"""Independent replay from native MUSIC noise to sources and IC components.

Signed physical transfers, Fourier refinement and gradients use the declared
native conventions. Captured sources/potentials are comparison inputs only
where explicitly stated. Discrete agreement does not establish continuum
accuracy, physical boundary conditions or a new statistical noise model.
"""
import math
from pathlib import Path
import numpy as np

_CORE_ROOT = Path(__file__).resolve().parent.parent


def _budget(memory_gib,bytes_required):
    if isinstance(memory_gib,(bool,np.bool_)) or not math.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Positive gradient reference workspace required')
    from core.resources import memory_limit_gb
    assigned=memory_limit_gb({'hardware':{'memory_limit_gb':memory_gib}})
    if bytes_required>min(memory_gib,assigned)*1024**3:
        raise MemoryError('Gradient reference exceeds declared or assigned host workspace')


def finite_gradient(potential,*,level,order=6,halo=4,memory_gib=1.):
    """Centred gradient with respect to dimensionless box coordinates.

    At level L the spacing is h=2**(-L), so each derivative contributes 1/h.
    Caller restores the physical box length or velocity-source length once.
    Ghost values are actual inputs; this operator does not solve boundaries.
    """
    from core.music.precision import Binary128Array
    if isinstance(potential,Binary128Array):
        from core.music.precision import wide_derivative
        return wide_derivative(potential,operation='gradient',level=level,order=order,halo=halo,memory_gib=memory_gib)
    values=np.asarray(potential)
    if (type(level) is not int or not 1<=level<=20 or type(order) is not int or order not in (2,4,6)
            or type(halo) is not int or halo<order//2 or values.ndim!=3
            or values.dtype.kind not in 'fiu' or any(n<=2*halo for n in values.shape)):
        raise ValueError('Supported native potential, actual ghosts and gradient order required')
    shape=tuple(n-2*halo for n in values.shape)
    _budget(memory_gib,64*1024**2+values.nbytes+64*math.prod(shape))
    if not np.isfinite(values).all():
        raise ValueError('Finite retained potential required')
    weights={2:[.5],4:[2/3,-1/12],6:[.75,-.15,1/60]}[order]
    result=np.zeros(shape+(3,))
    for axis in range(3):
        for offset,weight in enumerate(weights,1):
            for sign in (-1,1):
                delta=[0,0,0]
                delta[axis]=sign*offset
                result[...,axis]+=sign*weight*values[tuple(slice(halo+d,halo+d+n) for d,n in zip(delta,shape))]
    result*=2**level
    if not np.isfinite(result).all():
        raise ValueError('Gradient reference overflow')
    return result


def hybrid_correction(source,*,gradient_order=6,laplace_order=6,deconvolve_cic=False,memory_gib=1.):
    """Periodic cubic high-frequency correction in native grid-cell units.

    The correction is continuum i*k/k^2 minus the discrete gradient/negative
    Laplacian response. Only the differentiated Nyquist plane is removed.
    Caller applies the native finest-level 2^-level length conversion.
    """
    values=np.asarray(source)
    if (values.ndim!=3 or len(set(values.shape))!=1 or values.shape[0]<4 or values.shape[0]%2
            or values.dtype.kind not in 'fiu' or type(gradient_order) is not int or gradient_order not in (2,4,6)
            or type(laplace_order) is not int or laplace_order not in (2,4,6) or type(deconvolve_cic) is not bool):
        raise ValueError('Even cubic source and explicit supported hybrid operators required')
    _budget(memory_gib,64*1024**2+256*values.size)
    if not np.isfinite(values).all():
        raise ValueError('Finite hybrid source required')
    side=values.shape[0]
    axes=[2*np.pi*np.fft.fftfreq(side)[:,None,None],2*np.pi*np.fft.fftfreq(side)[None,:,None],
        2*np.pi*np.fft.rfftfreq(side)[None,None,:]]
    squared=sum(k*k for k in axes)
    laplace=np.zeros_like(squared)
    weights={2:[1.],4:[4/3,-1/12],6:[1.5,-.15,1/90]}[laplace_order]
    for wave in axes:
        for offset,weight in enumerate(weights,1):
            laplace+=4*weight*np.sin(offset*wave/2)**2
    fourier=np.fft.rfftn(values)
    if deconvolve_cic:
        assignment=np.ones_like(squared)
        for wave in axes:
            assignment*=np.sinc(wave/(2*np.pi))**2
        fourier=fourier/assignment
    gradient_weights={2:[.5],4:[2/3,-1/12],6:[.75,-.15,1/60]}[gradient_order]
    result=np.empty(values.shape+(3,))
    for axis,wave in enumerate(axes):
        discrete=sum(2*weight*np.sin(offset*wave) for offset,weight in enumerate(gradient_weights,1))
        kernel=np.divide(wave,squared,out=np.zeros_like(squared),where=squared>0)
        kernel-=np.divide(discrete,laplace,out=np.zeros_like(squared),where=laplace>0)
        transformed=1j*kernel*fourier
        selection=[slice(None)]*3
        selection[axis]=side//2
        transformed[tuple(selection)]=0
        transformed[0,0,0]=0
        result[...,axis]=np.fft.irfftn(transformed,s=values.shape,axes=(0,1,2))
    if not np.isfinite(result).all():
        raise ValueError('Hybrid correction overflow')
    return result


def compensated_mass_reference(contrast,mean_mass):
    """Native finite-window relative mode; preserve the supplied mean mass.

    Returns the encoded binary64 mass, its encoded relative fluctuation and
    the source-window mean. Nonpositive masses fail rather than being clipped.
    """
    values=np.asarray(contrast)
    if (values.ndim!=1 or not len(values) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()
            or isinstance(mean_mass,(bool,np.bool_)) or not math.isfinite(mean_mass) or mean_mass<=0):
        raise ValueError('Finite aligned compensation and positive native mean mass required')
    removed=float(np.sum(values,dtype=np.longdouble)/len(values))
    mass=mean_mass*(1.+values-removed)
    if not np.isfinite(mass).all() or np.any(mass<=0):
        raise ValueError('Compensated reference produces nonpositive or nonfinite mass')
    return mass,mass/mean_mass-1.,removed


def replay_gradient_hierarchy(source,potential,configuration,*,gas=False,memory_gib=1.):
    """Replay nested correction, native coarsening and base-only centring.

    Explicitly preserves current coarsen_density behavior: restrict the finest
    grid to its parent, then the full transfer base down to the Poisson base.
    It does not silently replace that behavior by all-level restriction.
    """
    from core.zoom import sample_hierarchy_density_reference
    cfg=configuration
    if type(gas) is not bool:
        raise ValueError('Explicit native gas preparation pass required')
    low,high=cfg.getint('setup','levelmin'),cfg.getint('setup','levelmax')
    if low>=high or cfg.getboolean('poisson','kspace',fallback=False):
        raise ValueError('Nested finite-difference gradient configuration required')
    if any([row['level'] for row in rows]!=list(range(low,high+1)) for rows in (source,potential)):
        raise ValueError('Retained gradient hierarchy differs from configured levels')
    order=cfg.getint('poisson','grad_order',fallback=6)
    laplace=cfg.getint('poisson','laplace_order',fallback=6)
    fine=source[-1]
    shape=fine['values'].shape
    mode=cfg.get('poisson','hybrid_source_extension',fallback='none')
    if mode not in ('none','bounded_conservative'):
        raise ValueError('Unsupported native hybrid source extension')
    boundary=cfg.getint('poisson','hybrid_extension_boundary',fallback=32) if mode=='bounded_conservative' else 32
    if boundary<=0:
        raise ValueError('Positive native hybrid boundary required')
    side=max(shape)+2*boundary
    cells=sum(row['values'].size for row in source)
    _budget(memory_gib,64*1024**2+256*side**3+256*cells)
    if mode=='bounded_conservative':
        degree=cfg.getint('poisson','hybrid_extension_degree',fallback=5)
        extended,extension=sample_hierarchy_density_reference(source,np.asarray(fine['origin'])-boundary,
            [side]*3,degree=degree,memory_gib=memory_gib,sampling='cell_average')
    else:
        extended=np.zeros((side,)*3)
        extended[tuple(slice(boundary,boundary+n) for n in shape)]=fine['values']
        extension={'sampling_convention':'native zero padding','shape':[side]*3}
    deconvolve=cfg.getboolean('output','glass_cicdeconvolve',fallback=False) and (
        not gas or cfg.getboolean('setup','do_SPH',fallback=False))
    correction=hybrid_correction(extended,gradient_order=order,laplace_order=laplace,
        deconvolve_cic=deconvolve,memory_gib=memory_gib)
    crop=tuple(slice(boundary,boundary+n) for n in shape)
    result=[]
    for s,p in zip(source,potential):
        if s['level']!=p['level'] or s['origin']!=p['origin']:
            raise ValueError('Native source/potential gradient geometry differs')
        values=finite_gradient(p['values'],level=p['level'],order=order,halo=p['halo'],memory_gib=memory_gib)
        if values.shape[:3]!=s['values'].shape:
            raise ValueError('Native potential gradient shape differs')
        result.append({'level':s['level'],'origin':s['origin'],'values':values})
    result[-1]['values']+=correction[crop]/2**high
    del correction,extended
    # Only finest->parent changes the archived physical hierarchy; subsequent
    # transfer-base coarsening is below levelmin when levelmin_TF==levelmin.
    if cfg.getint('setup','levelmin_TF',fallback=low)!=low:
        raise ValueError('Gradient replay requires actual transfer and Poisson bases to coincide')
    child,parent=result[-1],result[-2]
    nx,ny,nz=child['values'].shape[:3]
    start=(np.asarray(child['origin'])//2-np.asarray(parent['origin']))%2**parent['level']
    extent=np.array([nx,ny,nz])//2
    if parent['values'].shape[:3]!=(2**parent['level'],)*3 and np.any(start+extent>parent['values'].shape[:3]):
        raise ValueError('Native gradient child escapes parent')
    indices=np.ix_(*[(int(a)+np.arange(int(n)))%size for a,n,size in zip(start,extent,parent['values'].shape[:3])])
    parent['values'][indices]=child['values'].reshape(nx//2,2,ny//2,2,nz//2,2,3).mean(axis=(1,3,5))
    removed=np.mean(result[0]['values'],axis=(0,1,2),dtype=np.longdouble).astype(float)
    result[0]['values']-=removed
    return result,{'operator_scope':'Retained-potential finite gradient plus independent FFT hybrid response',
        'gradient_order':order,'laplace_order':laplace,'extension':extension,
        'gas_preparation_pass':gas,'cic_deconvolution_applied':deconvolve,
        'base_only_removed_gradient':removed.tolist(),'restriction':'finest->parent; transfer base below retained physical hierarchy',
        'potential_boundary_accuracy':'NOT_ESTABLISHED','science_acceptance':'NOT_ACCEPTED'}


def audit_music_archive_gradients(index,configuration,ic,destination,*,memory_gib=1.):
    """Compare retained and quadratic displacements and first-order velocity.

    Uses actual original material-cell centres, widths, box restoration and
    background species fractions. Coarse particles combine both prepared
    fluid fields; gas uses its own already staggered native source pass.
    """
    import json
    import h5py
    from core.foundation import file_sha256
    from core.data import export_numeric_audit,epoch_contract,snapshot_total_counts
    from core.music.archive import load_music_source_hierarchy
    from core.zoom import _configuration,_native_cell_levels
    index,configuration,ic=map(lambda p:Path(p).resolve(),(index,configuration,ic))
    pins={str(p):file_sha256(p) for p in (index,configuration,ic)}
    if index.with_name(index.name+'.sha256').read_text().strip()!=pins[str(index)]:
        raise ValueError('Native gradient source index seal mismatch')
    record=json.loads(index.read_text())
    cfg=_configuration(configuration)
    low,high=cfg.getint('setup','levelmin'),cfg.getint('setup','levelmax')
    if cfg.get('output','format')!='arepo' or not cfg.getboolean('setup','use_2LPT') or not cfg.getboolean('cosmology','mixed_2lpt'):
        raise ValueError('Original mixed 2LPT arepo gradient handoff required')
    with h5py.File(ic) as f:
        h=f['Header'].attrs
        a=float(h['Time'])
        box=float(h['BoxSize'])
        component_schema=int(h.get('Music_lpt_component_schema',0))
        component_bulk=np.asarray(h['Music_removed_bulk_velocity_gadget'],dtype=float) if component_schema==1 else None
        counts=snapshot_total_counts(h)
        shift=np.asarray(h['Music_output_coordinate_shift_box_fraction'],dtype=float)
        fb=float(h['OmegaBaryon'])/float(h['Omega0'])
        c2=float(h['Music_2lpt_displacement_coefficient'])
        mean_masses=np.asarray(h['Music_mean_species_masses_native'],dtype=float)
        if (h.get('Music_lpt_order')!=2 or h.get('Music_mixed_fluid_schema')!=1 or h.get('Music_material_cell_schema')!=1
                or h.get('Music_levelmin')!=low or h.get('Music_levelmax')!=high or np.any(counts[3:])
                or not np.array_equal(counts,h['NumPart_ThisFile']) or h.get('NumFilesPerSnapshot')!=1
                or shift.shape!=(3,) or not np.isfinite(shift).all() or not 0<fb<1
                or mean_masses.shape!=(6,) or not np.isfinite(mean_masses).all() or np.any(mean_masses[:2]<=0)
                or not math.isclose(a,1/(1+cfg.getfloat('setup','zstart')),rel_tol=1e-12)
                or not math.isclose(box,1000*cfg.getfloat('setup','boxlength'),rel_tol=1e-12)
                or not math.isclose(c2,cfg.getfloat('cosmology','lpt2_displacement'),rel_tol=1e-12)):
            raise ValueError('Actual native IC species, hierarchy and frame differ from gradient configuration')
        bound=np.asarray(h['Music_max_abs_displacement_native'],dtype=float)
        if bound.shape!=(3,) or not np.isfinite(bound).all() or np.any(bound<0) or np.any(bound>=box/4):
            raise ValueError('Unambiguous native displacement bounds required')
    rows=next(row['grids'] for row in record['fields'] if row['quantity_code']==1 and row['particle_type']==1)
    cells=sum(math.prod(row['shape']) for row in rows)
    boundary=cfg.getint('poisson','hybrid_extension_boundary',fallback=32) if cfg.get('poisson','hybrid_source_extension',fallback='none')=='bounded_conservative' else 32
    side=max(rows[-1]['shape'])+2*boundary
    _budget(memory_gib,64*1024**2+256*side**3+512*(2*cells+int(sum(counts))))
    fields={}
    operators=[]
    compensation={}
    for species in (0,1):
        for source_code,potential_code,label in ((1,2,'density'),(3,4,'quadratic'),(6,7,'velocity')):
            source,_=load_music_source_hierarchy(index,source_code,species,memory_gib=memory_gib)
            potential,_=load_music_source_hierarchy(index,potential_code,species,memory_gib=memory_gib,include_ghosts=True)
            replay,metadata=replay_gradient_hierarchy(source,potential,cfg,gas=species==0,memory_gib=memory_gib)
            fields[species,label]={row['level']:row for row in replay}
            operators.append({'particle_type':species,'quantity':label,**metadata})
        values,_=load_music_source_hierarchy(index,5,species,memory_gib=memory_gib)
        if [row['level'] for row in values]!=list(range(low,high+1)):
            raise ValueError('Native compensation hierarchy differs from configured levels')
        compensation[species]=values[-1]
    arrays={}
    specs={}
    checks=[]
    with h5py.File(ic) as f:
        for species in (0,1,2):
            if not counts[species]:
                continue
            group=f['PartType'+str(species)]
            n=int(counts[species])
            ids=np.asarray(group['ParticleIDs'])
            if ids.shape!=(n,) or ids.dtype.kind!='u' or ids.dtype.itemsize!=8:
                raise ValueError('Exact native uint64 gradient particle identities required')
            centres=np.asarray(group['LagrangianCellCenters'])
            widths=np.asarray(group['LagrangianCellWidths'])
            if centres.shape!=(n,3) or not np.isfinite(centres).all() or np.any(centres<0) or np.any(centres>=box):
                raise ValueError('Actual aligned periodic material centres required')
            levels=_native_cell_levels(widths,box,low,high)
            reference={label:np.empty((n,3)) for label in ('density','quadratic','velocity')}
            for level in np.unique(levels):
                selected=levels==level
                grid_coordinate=np.mod(centres[selected]/box-shift,1.)*2**int(level)-.5
                rounded=np.rint(grid_coordinate)
                if np.any(np.abs(grid_coordinate-rounded)>1e-9):
                    raise ValueError('Native material centres are not aligned with captured gradient cells')
                cell=rounded.astype(np.int64)%2**int(level)
                for label in reference:
                    def sample(t):
                        row=fields[t,label][int(level)]
                        local=(cell-np.asarray(row['origin']))%2**int(level)
                        if np.any(local>=np.array(row['values'].shape[:3])):
                            raise ValueError('Native particle cell escapes captured gradient hierarchy')
                        return row['values'][tuple(local.T)]
                    reference[label][selected]=sample(species) if species in (0,1) else (1-fb)*sample(1)+fb*sample(0)
            actual_first=np.asarray(group['FirstOrderDisplacements'])
            actual_velocity=np.asarray(group['FirstOrderVelocities'])*math.sqrt(a)
            q=np.asarray(group['LagrangianCoordinates'])
            x=np.asarray(group['Coordinates'])
            total=(x-q+box/2)%box-box/2
            if any(v.shape!=(n,3) or not np.isfinite(v).all() for v in (actual_first,actual_velocity,q,x,total)):
                raise ValueError('Finite aligned native displacement and velocity components required')
            if np.any(np.max(np.abs(total),axis=0)>bound+box*1e-12):
                raise ValueError('Delivered displacement exceeds native unwrapping bound')
            comparisons={
                'first_displacement':(actual_first,reference['density']*box,'comoving kpc/h'),
                'first_velocity':(actual_velocity,reference['velocity']*cfg.getfloat('setup','boxlength'),'km/s'),
                'quadratic_displacement':(total-actual_first,reference['quadratic']*(c2*box),'comoving kpc/h'),
                'total_displacement':(total,(reference['density']+c2*reference['quadratic'])*box,'comoving kpc/h')}
            if component_schema==1:
                # Direct components preserve tiny quadratic signals that are
                # lost by subtracting large encoded particle coordinates.
                second=np.asarray(group['SecondOrderDisplacements'])
                second_velocity=np.asarray(group['SecondOrderVelocities'])*math.sqrt(a)
                if any(v.shape!=(n,3) or not np.isfinite(v).all() for v in (second,second_velocity)):
                    raise ValueError('Finite aligned retained second-order LPT components required')
                cv2=cfg.getfloat('cosmology','lpt2_velocity')*cfg.getfloat('cosmology','lpt2_aH_km_s_Mpc')/(cfg.getfloat('cosmology','H0')/100.)
                comparisons['quadratic_displacement']=(second,reference['quadratic']*(c2*box),'comoving kpc/h')
                comparisons['quadratic_velocity']=(second_velocity,reference['quadratic']*(cv2*cfg.getfloat('setup','boxlength')),'km/s')
                coordinate_residual=total-(actual_first+second)
                # Enclose the binary64 additions, unit scaling, barycentric
                # combinations and periodic encoding in the native writer.
                # This is a separate encoding check; the measured component
                # itself still has the unchanged1e-7 operator requirement.
                encoding_bound=64*np.finfo(float).eps*box
                if np.max(np.abs(coordinate_residual))>encoding_bound:
                    raise ValueError('Retained LPT components disagree with delivered particle coordinates beyond binary64 encoding roundoff')
                operators.append({'particle_type':species,'quantity':'component_coordinate_encoding',
                    'maximum_absolute_residual':float(np.max(np.abs(coordinate_residual))),
                    'absolute_arithmetic_bound':encoding_bound,'units':'comoving kpc/h',
                    'basis':'64binary64relative roundoff units times box length; component operator tolerance remains1e-7','status':'PASS'})
                bulk=component_bulk
                if bulk.shape!=(3,) or not np.isfinite(bulk).all():
                    raise ValueError('Explicit finite native momentum zero mode required')
                comparisons['total_velocity']=(np.asarray(group['Velocities'])*math.sqrt(a)+bulk*math.sqrt(a),
                    (reference['velocity']+cv2*reference['quadratic'])*cfg.getfloat('setup','boxlength'),'km/s')
            actual_mass=np.asarray(group['Masses']) if 'Masses' in group else np.full(n,float(f['Header'].attrs['MassTable'][species]))
            if actual_mass.shape!=(n,) or not np.isfinite(actual_mass).all() or np.any(actual_mass<=0):
                raise ValueError('Actual aligned positive native masses required')
            if species in (0,1):
                if np.any(levels!=high):
                    raise ValueError('Fine native species unexpectedly spans coarse hierarchy levels')
                row=compensation[species]
                cell=np.rint(np.mod(centres/box-shift,1.)*2**high-.5).astype(np.int64)%2**high
                local=(cell-np.asarray(row['origin']))%2**high
                if np.any(local>=np.array(row['values'].shape)):
                    raise ValueError('Native species compensation sample escapes captured grid')
                contrast=row['values'][tuple(local.T)]
                expected_mass,expected_fluctuation,removed=compensated_mass_reference(contrast,mean_masses[species])
                comparisons['compensated_density']=(actual_mass/mean_masses[species]-1.,expected_fluctuation,'dimensionless')
                operators.append({'particle_type':species,'quantity':'compensated_density','source_window_mean_removed':removed,
                    'mean_species_mass_native':float(mean_masses[species]),'normalization':'Native finite-window mean only; no fluctuation amplitude renormalization'})
            else:
                expected_mass=(mean_masses[0]+mean_masses[1])*np.exp2(3*(high-levels))
            comparisons['particle_mass']=(actual_mass,expected_mass,'1e10 Msun/h')
            id_key=f'species{species}_particle_ids'
            arrays[id_key]=ids
            specs[id_key]={'units':'identity','quantity':'Exact native IC identity in original HDF5 row order',
                'identity_namespace':'native-music-ic:'+pins[str(ic)]+':PartType'+str(species),
                'ancestry':{'relation':'same actual IC particle; no cross-run identity equivalence'}}
            for label,(measured,expected,units) in comparisons.items():
                residual=measured-expected
                for level in np.unique(levels):
                    selected=levels==level
                    norm=float(np.linalg.norm(expected[selected]))
                    difference=float(np.linalg.norm(residual[selected]))
                    error=difference/norm if norm>0 else 0. if difference==0 else math.inf
                    checks.append({'particle_type':species,'level':int(level),'quantity':label,'particles':int(selected.sum()),
                        'relative_l2_error':error if math.isfinite(error) else None,'status':'PASS' if error<=1e-7 else 'FAIL'})
                for name,value in (('measured',measured),('reference',expected),('residual',residual)):
                    key=f'species{species}_{label}_{name}'
                    arrays[key]=value
                    specs[key]={'units':units,'quantity':key}
    from core.music.archive import archive_precision_sources
    pins.update(archive_precision_sources(record))
    pins[record['archive']['path']]=record['archive']['sha256']
    for path in (Path(__file__).resolve(),_CORE_ROOT/'music/archive.py',_CORE_ROOT/'zoom.py'):
        pins[str(path)]=file_sha256(path)
    report={'schema':'native-music-gradient-replay-v1','status':'PASS' if checks and all(row['status']=='PASS' for row in checks) else 'FAIL',
        'checks':checks,'operators':operators,'maximum_relative_operator_error':1e-7,
        'scope':'Independent retained-potential finite-gradient/FFT hybrid replay and compensated mass encoding at actual species/material cells, including native coarsening, base centring and declared background species masses',
        'potential_boundary_accuracy':'NOT_ESTABLISHED','physical_transfer_window_acceptance':'NOT_ESTABLISHED','science_acceptance':'NOT_ACCEPTED'}
    return export_numeric_audit(destination,report,arrays,specs,algorithm=__file__,context={'epoch':epoch_contract(a)},source_hashes=pins)


def _waves(shape):
    axes=[]
    for axis,n in enumerate(shape):
        k=np.arange(n if axis<2 else n//2+1,dtype=float)
        k[k>n//2]-=n
        broadcast=[1,1,1]
        broadcast[axis]=len(k)
        axes.append(k.reshape(broadcast))
    return axes


def convolve_noise_reference(noise,amplitude,*,patch_length,shift_cells=0.,memory_gib=1.):
    """Point-sampled Fourier convolution, with signed sqrt(P)-like amplitude.

    amplitude(k) must include the physical transfer/primordial normalization,
    but not sqrt(N/V). Native noise has unit cell variance. No cell window is
    silently inserted. A positive shift evaluates the field at shifted nodes.
    For N cells in volume V, sqrt(N/V) converts unit-variance cell noise
    and the supplied sqrt(P) amplitude into the native Fourier convention.
    """
    values=np.asarray(noise)
    if (values.ndim!=3 or len(set(values.shape))!=1 or values.shape[0]<4 or values.shape[0]%2
            or values.dtype.kind not in 'fiu' or not callable(amplitude)
            or not math.isfinite(patch_length) or patch_length<=0 or not math.isfinite(shift_cells)):
        raise ValueError('Even cubic noise, signed transfer and explicit physical patch length required')
    _budget(memory_gib,64*1024**2+192*values.size)
    if not np.isfinite(values).all():
        raise ValueError('Finite original noise required')
    waves=_waves(values.shape)
    k=2*np.pi/patch_length*np.sqrt(sum(w*w for w in waves))
    transfer=np.asarray(amplitude(k))
    if transfer.shape!=k.shape or not np.isfinite(transfer).all():
        raise ValueError('Finite aligned signed transfer required')
    spectrum=np.fft.rfftn(values)*transfer*math.sqrt(values.size/patch_length**3)
    if shift_cells:
        spectrum*=np.exp(2j*np.pi*shift_cells/values.shape[0]*sum(waves))
    spectrum[0,0,0]=0.
    result=np.fft.irfftn(spectrum,s=values.shape,axes=(0,1,2))
    if not np.isfinite(result).all():
        raise ValueError('Transfer reference overflow')
    return result


def fourier_splice_reference(coarse,fine,coarse_start,*,periodic_parent=False,memory_gib=1.):
    """Native Meyer blend with point-centre quarter-coarse-cell phase shift.

    coarse_start locates the footprint of the *padded* fine patch in the
    actual parent array. Only the full base array may wrap periodically.
    """
    parent=np.asarray(coarse)
    child=np.asarray(fine)
    start=np.asarray(coarse_start)
    if (parent.ndim!=3 or child.ndim!=3 or any(n<8 or n%4 for n in child.shape)
            or parent.dtype.kind not in 'fiu' or child.dtype.kind not in 'fiu'
            or start.shape!=(3,) or start.dtype.kind not in 'iu' or type(periodic_parent) is not bool):
        raise ValueError('Aligned even splice arrays and integer parent footprint required')
    _budget(memory_gib,64*1024**2+192*(parent.size+child.size))
    extent=np.asarray(child.shape)//2
    if np.any(extent>parent.shape) or not periodic_parent and (np.any(start<0) or np.any(start+extent>parent.shape)):
        raise ValueError('Fourier splice footprint escapes actual parent')
    if not np.isfinite(parent).all() or not np.isfinite(child).all():
        raise ValueError('Finite splice fields required')
    footprint=parent[np.ix_(*[(int(o)+np.arange(int(n)))%size for o,n,size in zip(start,extent,parent.shape)])]
    c=np.fft.rfftn(footprint)
    f=np.fft.rfftn(child)
    waves=_waves(footprint.shape)
    blend=np.ones(c.shape)
    phase=np.zeros(c.shape)
    for w,n in zip(waves,extent):
        # Native extent is size_t: nc/4 is integer division. A padded fine
        # side60 therefore blends with cutoff7, not7.5.
        ratio=np.abs(w)/(int(n)//4)
        blend*=np.where(ratio<=1/3,1.,np.where(ratio<2/3,np.cos(.5*np.pi*(3*ratio-1)),0.))
        phase-=.5*np.pi*w/n
    indices=[]
    for axis,n in enumerate(extent):
        i=np.arange(n if axis<2 else n//2+1)
        if axis<2:
            i=np.where(i>n//2,i+n,i)
        indices.append(i)
    selection=np.ix_(*indices)
    f[selection]=(1-blend)*f[selection]+blend*c*np.exp(1j*phase)*8.
    return np.fft.irfftn(f,s=child.shape,axes=(0,1,2))


def dms_transfer_amplitude(table,configuration,kind):
    """Signed physical transfer amplitude using native natural splines in log k.

    Returns sqrt(P) for density/relative density, and the corresponding
    velocity-source amplitude in km/s per comoving Mpc/h. Start transfers
    require identical target/start epochs, so no growth fit is introduced.
    """
    from scipy.interpolate import CubicSpline
    rows=np.asarray(table,dtype=float)
    cfg=configuration
    if (rows.ndim!=2 or rows.shape[1]!=5 or len(rows)<6 or not np.isfinite(rows).all()
            or np.any(rows[:,0]<=0) or np.any(np.diff(rows[:,0])<=0)
            or kind not in ('density','compensation','velocity_cdm','velocity_gas')
            or cfg.get('cosmology','transfer')!='DMS_file'
            or cfg.getfloat('setup','zstart')!=cfg.getfloat('cosmology','ztarget')):
        raise ValueError('Supported signed start-epoch DMS transfer rows required')
    h=cfg.getfloat('cosmology','H0')/100
    ns=cfg.getfloat('cosmology','n_s')
    amplitude=cfg.getfloat('cosmology','A_s')
    pivot=cfg.getfloat('cosmology','k_p')/h
    fb=cfg.getfloat('cosmology','Omega_b')/cfg.getfloat('cosmology','Omega_m')
    # Positivity comparisons alone do not reject NaN settings.
    if (not all(math.isfinite(value) for value in (h,ns,amplitude,pivot,fb))
            or not 0<fb<1 or h<=0 or amplitude<=0 or pivot<=0):
        raise ValueError('Physical primordial normalization required')
    functions=[CubicSpline(np.log(rows[:,0]),rows[:,col],bc_type='natural',extrapolate=False) for col in range(1,5)]
    norm=math.sqrt(2*np.pi**2*amplitude*pivot**(1-ns))
    def evaluate(k):
        k=np.asarray(k)
        result=np.zeros(k.shape)
        use=(k>=rows[1,0])&(k<=rows[-2,0])
        x=np.log(k[use])
        if kind=='density':
            value=-((1-fb)*functions[0](x)+fb*functions[1](x))
        elif kind=='compensation':
            value=-(functions[1](x)-functions[0](x))
        else:value=functions[2 if kind=='velocity_cdm' else 3](x)*299792.458/h
        result[use]=norm*k[use]**(ns/2-2)*value
        return result
    return evaluate


def replay_prepared_source(noises,mesh,configuration,amplitude,*,gas=False,memory_gib=1.):
    """Replay actual padded hierarchy, Fourier splicing, cuts and leaf mean.

    Supports the retained DMS default Fourier coarsening route with identical
    transfer/Poisson base levels. The captured source does not enter replay.
    """
    cfg=configuration
    low=cfg.getint('setup','levelmin')
    high=cfg.getint('setup','levelmax')
    if (type(gas) is not bool or cfg.getint('setup','levelmin_TF',fallback=low)!=low
            or not cfg.getboolean('setup','fourier_splicing',fallback=True)
            or cfg.getboolean('setup','fix_mode_amplitude',fallback=False)
            or cfg.getboolean('setup','flip_mode_amplitude',fallback=False)):
        raise ValueError('Declared default Fourier-coarsened source route required')
    margin=mesh['convolution_margin']
    tf=mesh['hierarchies']['convolution']
    physical=mesh['hierarchies']['poisson']
    if type(margin) is not int or margin<=0 or margin%2 or any([row['level'] for row in rows]!=list(range(low,high+1)) for rows in (tf,physical)):
        raise ValueError('Actual contiguous hierarchy and positive even convolution margin required')
    _budget(memory_gib,64*1024**2+256*sum(np.asarray(noises[row['level']]).size for row in tf))
    fields=[]
    previous=None
    previous_origin=None
    box=cfg.getfloat('setup','boxlength')
    for row in tf:
        level=row['level']
        noise=np.asarray(noises[level])
        pad=0 if level==low else margin
        if noise.shape!=tuple(n+2*pad for n in row['shape']):
            raise ValueError('Original retained noise differs from convolution mesh')
        origin=np.asarray(row['offset'])-pad
        shift=cfg.getfloat('setup','baryon_staggering',fallback=.5)*2.**(level-high) if gas else 0.
        values=convolve_noise_reference(noise,amplitude,patch_length=box*noise.shape[0]/2**level,
            shift_cells=shift,memory_gib=memory_gib)
        if previous is not None:
            start=origin//2-previous_origin
            if np.any(origin%2):
                raise ValueError('Padded fine source must align to parent cells')
            values=fourier_splice_reference(previous,values,start,periodic_parent=level==low+1,memory_gib=memory_gib)
        cropped=values[tuple(slice(pad,pad+n) for n in row['shape'])].copy()
        fields.append({'level':level,'origin':list(row['offset']),'values':cropped})
        previous,previous_origin=values,origin
    # The default native cut adjusts the covered *parent* patch's mean. Every
    # such cell is subsequently replaced by complete child averages during
    # composite normalization, so none of that temporary adjustment survives.
    for i,(row,target) in enumerate(zip(fields,physical)):
        start=np.asarray(target['offset'])-np.asarray(row['origin'])
        if np.any(start<0) or np.any(start+target['shape']>row['values'].shape):
            raise ValueError('Physical source cut escapes convolution hierarchy')
        fields[i]={'level':row['level'],'origin':list(target['offset']),
            'values':row['values'][tuple(slice(int(o),int(o)+n) for o,n in zip(start,target['shape']))].copy()}
    for i in range(len(fields)-1,0,-1):
        child,parent=fields[i],fields[i-1]
        nx,ny,nz=child['values'].shape
        start=(np.asarray(child['origin'])//2-np.asarray(parent['origin']))%2**parent['level']
        extent=np.asarray(child['values'].shape)//2
        if parent['values'].shape!=(2**parent['level'],)*3 and np.any(start+extent>parent['values'].shape):
            raise ValueError('Physical child escapes parent')
        selection=np.ix_(*[(int(o)+np.arange(int(n)))%size for o,n,size in zip(start,extent,parent['values'].shape)])
        parent['values'][selection]=child['values'].reshape(nx//2,2,ny//2,2,nz//2,2).mean(axis=(1,3,5))
    removed=float(np.mean(fields[0]['values'],dtype=np.longdouble))
    for row in fields:
        row['values']-=removed
    return fields,{'schema':'native-discrete-prepared-source-replay-v1','common_removed_composite_mean':removed,
        'sampling':'Point-sampled convolution and Fourier splice; child-averaged covered parent cells; common composite DC projection',
        'reference_inputs':'Original noise, signed transfer and actual mesh only; no captured source or potential',
        'gas_preparation_pass':gas,'science_acceptance':'NOT_ACCEPTED'}


def load_native_noise(path,shape,*,byte_order,memory_gib=1.):
    """Read the native binary64 cached noise, validating geometry before load."""
    path=Path(path)
    shape=tuple(shape)
    if byte_order not in ('<','>') or len(shape)!=3 or any(type(n) is not int or n<4 or n%2 for n in shape):
        raise ValueError('Explicit native noise byte order and even mesh shape required')
    _budget(memory_gib,64*1024**2+16*math.prod(shape))
    with path.open('rb') as stream:
        header=np.fromfile(stream,dtype=byte_order+'u4',count=3)
        if tuple(header)!=shape or path.stat().st_size!=12+8*math.prod(shape):
            raise ValueError('Retained noise byte count or geometry differs from actual native mesh')
        result=np.fromfile(stream,dtype=byte_order+'f8',count=math.prod(shape)).reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite retained native noise')
    return result


def transfer_replay_input_bindings(parameters):
    """Freeze all finite replay inputs before workflow cache/reuse decisions."""
    import json
    from core.foundation import file_sha256
    from core.zoom import _configuration
    config=Path(parameters['configuration']).resolve()
    index=Path(parameters['source_index']).resolve()
    request=parameters['request']
    mesh_path=Path(request['mesh']).resolve()
    mesh=json.loads(mesh_path.read_text())
    cfg=_configuration(config)
    record=json.loads(index.read_text())
    transfer=Path(cfg.get('cosmology','transfer_file'))
    if not transfer.is_absolute():
        transfer=config.parent/transfer
    archive=Path(record['archive']['path'])
    if file_sha256(archive)!=record['archive']['sha256']:
        raise ValueError('Retained source archive changed')
    seal=index.with_name(index.name+'.sha256')
    if seal.read_text().strip()!=file_sha256(index):
        raise ValueError('Retained source index seal mismatch')
    paths=[Path(parameters['product']).resolve(),config,index,seal,archive,transfer.resolve(),mesh_path]
    paths.extend(Path(request['noise_directory']).resolve()/f'wnoise_{row["level"]:04d}.bin' for row in mesh['hierarchies']['convolution'])
    return {str(path):file_sha256(path) for path in paths}


def audit_music_archive_transfer(index,configuration,ic,request,destination,*,memory_gib=1.):
    """Bind source replay to retained noise, mesh, transfer, IC and index bytes.

    Uses the existing 1e-7 operator-agreement criterion. A pass establishes
    original finite source assembly, not physical continuum/window accuracy.
    """
    import json,re
    import h5py
    from core.foundation import file_sha256
    from core.data import export_numeric_audit,epoch_contract
    from core.zoom import _configuration
    from core.music.archive import load_music_source_hierarchy
    if (not isinstance(request,dict) or set(request)!={'schema','mesh','noise_directory'}
            or request['schema']!='native-music-transfer-replay-request-v1'):
        raise ValueError('Explicit original native noise/mesh replay request required')
    index,configuration,ic=map(lambda p:Path(p).resolve(),(index,configuration,ic))
    mesh_path=Path(request['mesh']).resolve()
    noise_dir=Path(request['noise_directory']).resolve()
    cfg=_configuration(configuration)
    mesh=json.loads(mesh_path.read_text())
    pins={str(p):file_sha256(p) for p in (index,configuration,ic,mesh_path)}
    if index.with_name(index.name+'.sha256').read_text().strip()!=pins[str(index)]:
        raise ValueError('Native transfer source index seal mismatch')
    record=json.loads(index.read_text())
    low=cfg.getint('setup','levelmin')
    high=cfg.getint('setup','levelmax')
    if (not 2<=low<high<=20 or mesh.get('schema')!='darkmattersim-music-mesh-v1' or mesh.get('scalar_bytes')!=8
            or mesh['convolution_margin']!=cfg.getint('setup','convolution_margin',fallback=4)
            or not cfg.getboolean('cosmology','mixed_2lpt',fallback=False) or not cfg.getboolean('setup','do_SPH',fallback=False)):
        raise ValueError('Actual mixed-fluid nested binary64 source mesh required')
    physical=mesh['hierarchies']['poisson']
    tf=mesh['hierarchies']['convolution']
    if any([row['level'] for row in rows]!=list(range(low,high+1)) for rows in (physical,tf)):
        raise ValueError('Actual transfer mesh differs from configured levels')
    for field in record['fields']:
        if [(row['level'],row['shape'],row['origin']) for row in field['grids']]!=[(row['level'],row['shape'],row['offset']) for row in physical]:
            raise ValueError('Captured source geometry differs from supplied native mesh')
    shapes={row['level']:tuple(n+(0 if row['level']==low else 2*mesh['convolution_margin']) for n in row['shape']) for row in tf}
    table_path=Path(cfg.get('cosmology','transfer_file'))
    if not table_path.is_absolute():
        table_path=configuration.parent/table_path
    table_path=table_path.resolve()
    _budget(memory_gib,64*1024**2+384*sum(math.prod(shape) for shape in shapes.values())+16*table_path.stat().st_size)
    with h5py.File(ic) as f:
        h=f['Header'].attrs
        a=float(h['Time'])
        if (h.get('Music_levelmin')!=low or h.get('Music_levelmax')!=high or h.get('Music_lpt_order')!=2
                or h.get('Music_mixed_fluid_schema')!=1 or h.get('Music_composite_source_normalization_schema')!=1
                or not math.isclose(a,1/(1+cfg.getfloat('setup','zstart')),rel_tol=1e-12)
                or not math.isclose(float(h['BoxSize']),1000*cfg.getfloat('setup','boxlength'),rel_tol=1e-12)):
            raise ValueError('Transfer reference configuration differs from actual normalized native IC')
    pins[str(table_path)]=file_sha256(table_path)
    text=table_path.read_text()
    z=re.search(r'redshift:\s*([^;\s]+)',text)
    if ('DarkMatterSim monofonIC transfer table v1' not in text or 'gauge: N-body' not in text
            or 'Model: '+cfg.get('cosmology','input_model')+';' not in text or not z
            or not math.isclose(float(z.group(1)),cfg.getfloat('setup','zstart'),rel_tol=0,abs_tol=1e-8)):
        raise ValueError('Original signed transfer schema, model, gauge or epoch mismatch')
    table=np.loadtxt(table_path)
    noises={}
    for level,shape in shapes.items():
        path=noise_dir/f'wnoise_{level:04d}.bin'
        pins[str(path)]=file_sha256(path)
        noises[level]=load_native_noise(path,shape,byte_order=record['byte_order'],memory_gib=memory_gib)
    checks=[]
    arrays={}
    specs={}
    preparation=[]
    fb=cfg.getfloat('cosmology','Omega_b')/cfg.getfloat('cosmology','Omega_m')
    for species in (0,1):
        for code,kind in ((1,'density'),(5,'compensation'),(6,'velocity_gas' if species==0 else 'velocity_cdm')):
            predicted,metadata=replay_prepared_source(noises,mesh,cfg,dms_transfer_amplitude(table,cfg,kind),gas=species==0,memory_gib=memory_gib)
            preparation.append({'particle_type':species,'quantity_code':code,**metadata})
            actual,_=load_music_source_hierarchy(index,code,species,memory_gib=memory_gib)
            for original,replayed in zip(actual,predicted):
                if original['level']!=replayed['level'] or original['origin']!=replayed['origin']:
                    raise ValueError('Replayed source frame differs')
                value=replayed['values']*((1-fb if species==0 else -fb) if code==5 else 1.)
                residual=original['values']-value
                norm=float(np.linalg.norm(original['values']))
                difference=float(np.linalg.norm(residual))
                error=difference/norm if norm else 0. if difference==0 else math.inf
                checks.append({'particle_type':species,'quantity_code':code,'level':original['level'],
                    'relative_l2_error':error if math.isfinite(error) else None,'status':'PASS' if error<=1e-7 else 'FAIL'})
                prefix=f'species{species}_quantity{code}_level{original["level"]}_'
                for name,data in (('original_source',original['values']),('reference_source',value),('residual',residual)):
                    arrays[prefix+name]=data
                    specs[prefix+name]={'units':'km/s per comoving Mpc/h' if code==6 else 'dimensionless','quantity':prefix+name}
    pins[record['archive']['path']]=record['archive']['sha256']
    for path in (Path(__file__).resolve(),(_CORE_ROOT/'zoom.py'),(_CORE_ROOT/'music/archive.py')):
        pins[str(path)]=file_sha256(path)
    report={'schema':'native-music-original-transfer-replay-v1','status':'PASS' if checks and all(c['status']=='PASS' for c in checks) else 'FAIL',
        'checks':checks,'preparation':preparation,'maximum_relative_operator_error':1e-7,
        'scope':'Independent original finite source assembly from actual retained padded noise, signed transfer and mesh; no captured source or potential in prediction',
        'physical_window_acceptance':'NOT_ESTABLISHED','boundary_discretization_error_budget':'NOT_ESTABLISHED',
        'launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
    return export_numeric_audit(destination,report,arrays,specs,algorithm=__file__,context={'epoch':epoch_contract(a)},source_hashes=pins)
