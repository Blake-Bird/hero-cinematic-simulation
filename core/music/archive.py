"""Retained native MUSIC fields, hierarchy loading and discrete comparisons.

The archive keeps sources before their solves and solved potentials with
actual ghost cells. Indexing checks bytes and finite values, not whether a
field has correct physical boundaries or transfer normalization. The operator
comparisons below keep those questions separate from discrete agreement.
"""
from pathlib import Path
import math
import struct
import numpy as np
from core.foundation import file_sha256,atomic_json,atomic_text

_CORE_ROOT = Path(__file__).resolve().parent.parent

QUANTITIES={1:'prepared_matter_density',2:'linear_potential',3:'quadratic_source',
    4:'quadratic_potential',5:'compensated_relative_density',6:'prepared_velocity_source',7:'velocity_potential'}


def archive_precision_sources(record):
    """Bind wider arithmetic when this archive actually contains wider fields."""
    if any(row.get('scalar_bytes',8)==16 for row in record['fields']):
        from core.music.precision import reference_binding_sources
        return reference_binding_sources()
    return {}


def potential_weighted_sum(values,weights,*,memory_gib=1.):
    """Accumulate real potential weights without downcasting binary128 input."""
    from core.music.precision import Binary128Array
    if isinstance(values,Binary128Array):
        from core.music.precision import wide_dot
        return wide_dot(values,weights,memory_gib=memory_gib)
    return float(np.sum(values*weights))


def index_music_source_archive(filename,destination,*,memory_gib=1.):
    """Check every field and finite scalar, retaining bounded payload offsets.

    Ghost cells are retained for solved potentials only. Source fields are
    captured before their solves and have no assumed valid ghost population.
    Sampling semantics and the link to the generated IC are separate checks.
    """
    if isinstance(memory_gib,(bool,np.bool_)) or not math.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Positive native source archive indexing workspace required')
    budget=memory_gib*1024**3-64*1024**2
    if budget<1024:
        raise MemoryError('Native source archive index exceeds declared workspace')
    chunk=min(131072,int(budget//64))
    target=Path(destination).resolve()
    if target.exists() or target.with_name(target.name+'.sha256').exists():
        raise FileExistsError('Keep original source archive index evidence')
    path=Path(filename).resolve()
    before=file_sha256(path)
    size=path.stat().st_size
    records=[]
    seen=set()
    geometry=None
    with path.open('rb') as stream:
        magic=stream.read(8)
        if magic not in (b'MUSICS01',b'MUSICS02'):
            raise ValueError('Unknown native source archive magic')
        version=1 if magic==b'MUSICS01' else 2
        marker=stream.read(4)
        if marker==b'\x04\x03\x02\x01':
            endian='<'
        elif marker==b'\x01\x02\x03\x04':
            endian='>'
        else:raise ValueError('Unknown native source archive byte order')
        def unpack(format):
            count=struct.calcsize(endian+format)
            raw=stream.read(count)
            if len(raw)!=count:
                raise ValueError('Truncated native source archive metadata')
            return struct.unpack(endian+format,raw)
        if unpack('I')!=(version,):
            raise ValueError('Unsupported native source archive version')
        while True:
            tag=stream.read(8)
            if tag==b'END00001':
                if stream.read(1):
                    raise ValueError('Trailing native source archive bytes')
                break
            if tag not in ((b'FIELD001',b'FIELD128') if version==2 else (b'FIELD001',)):
                raise ValueError('Incomplete native source archive field or completion seal')
            scalar_bytes=16 if tag==b'FIELD128' else 8
            quantity,species,low,high,halo=unpack('IIIII')
            # Only the six solved potential fields can use wider scalars.
            # Sources and noise retain the native binary64 representation.
            if scalar_bytes==16 and quantity not in (2,4,7):
                raise ValueError('Binary128 payload is only supported for native potentials')
            if (quantity not in QUANTITIES or species not in (0,1) or (quantity,species) in seen
                    or not 1<=low<=high<=20 or halo!=(4 if quantity in (2,4,7) else 0)):
                raise ValueError('Unsupported, repeated or invalid native source archive field')
            seen.add((quantity,species))
            grids=[]
            actual_geometry=[]
            for expected in range(low,high+1):
                level,=unpack('I')
                shape=unpack('QQQ')
                origin=unpack('qqq')
                if (level!=expected or any(n<1 or n>2**level for n in shape)
                        or level==low and (shape!=(2**level,)*3 or origin!=(0,0,0))):
                    raise ValueError('Invalid native source archive hierarchy geometry')
                stored=tuple(n+2*halo for n in shape)
                count=math.prod(stored)
                offset=stream.tell()
                if count*scalar_bytes>size-offset-8:
                    raise ValueError('Truncated native source archive payload')
                remaining=count
                while remaining:
                    take=min(remaining,chunk)
                    if scalar_bytes==8:
                        values=np.fromfile(stream,dtype=endian+'f8',count=take)
                        valid=len(values)==take and np.isfinite(values).all()
                    else:
                        words=np.fromfile(stream,dtype=endian+'u8',count=2*take)
                        valid=len(words)==2*take
                        if valid:
                            # The exponent is in the high 64-bit word; check
                            # it directly so validation never rounds to float64.
                            high_words=words[1::2] if endian=='<' else words[0::2]
                            valid=not np.any(((high_words>>48)&0x7fff)==0x7fff)
                    if not valid:
                        raise ValueError('Invalid native source archive numeric payload')
                    remaining-=take
                grids.append({'level':level,'shape':list(shape),'origin':list(origin),
                    'halo':halo,'stored_shape':list(stored),'byte_offset':offset,'scalar_count':count,'scalar_bytes':scalar_bytes})
                actual_geometry.append((level,shape,origin))
            if geometry is None:
                geometry=actual_geometry
            elif geometry!=actual_geometry:
                raise ValueError('Native captured source/potential hierarchy differs between fields')
            records.append({'quantity_code':quantity,'quantity':QUANTITIES[quantity],'particle_type':species,
                'units':'km/s per comoving Mpc/h' if quantity in (6,7) else 'dimensionless',
                'scalar_bytes':scalar_bytes,'encoding':'IEEE754_binary128' if scalar_bytes==16 else 'IEEE754_binary64','grids':grids})
    if seen!={(q,t) for q in QUANTITIES for t in (0,1)}:
        raise ValueError('Native source archive omits required species/flow components')
    if file_sha256(path)!=before:
        raise ValueError('Native source archive changed during indexing')
    report={'schema':'native-music-source-archive-index-v1','status':'INDEXED_FINITE_PAYLOAD',
        'archive':{'path':str(path),'sha256':before,'bytes':size},'archive_format_version':version,'byte_order':endian,'fields':records,
        'grid_coordinates':'Native computational cell centres (origin + index + 0.5) / 2**level; original-box shift and species staggering must be bound to the actual IC and recipe',
        'velocity_source_units':'Native dimensionless-box Poisson/gradient result multiplied by configured boxlength in Mpc/h yields physical peculiar km/s before output sqrt(a) encoding',
        'sampling_convention':'NOT_INFERRED; actual convolution, restriction, splicing and normalization operators require separate comparison',
        'epoch':'NOT_BOUND; caller must bind the original generated IC and configuration',
        'source_sha256':{str(Path(__file__).resolve()):file_sha256(__file__)},
        'source_accuracy_acceptance':'NOT_ESTABLISHED','science_acceptance':'NOT_ACCEPTED'}
    atomic_json(target,report)
    atomic_text(target.with_name(target.name+'.sha256'),file_sha256(target)+'\n')
    return report


def load_music_source_hierarchy(index,quantity,species,*,memory_gib=1.,include_ghosts=False):
    """Read selected immutable interiors or explicit native ghost grids."""
    import json
    if isinstance(memory_gib,(bool,np.bool_)) or not math.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Positive native source hierarchy workspace required')
    if type(include_ghosts) is not bool:
        raise ValueError('Explicit native source ghost selection required')
    index=Path(index).resolve()
    before=file_sha256(index)
    if index.with_name(index.name+'.sha256').read_text().strip()!=before:
        raise ValueError('Native source index seal mismatch')
    report=json.loads(index.read_text())
    if report.get('schema')!='native-music-source-archive-index-v1' or report.get('status')!='INDEXED_FINITE_PAYLOAD':
        raise ValueError('Unsupported native source archive index')
    if type(quantity) is not int or type(species) is not int:
        raise ValueError('Exact native source quantity/species enum required')
    matches=[row for row in report['fields'] if row['quantity_code']==quantity and row['particle_type']==species]
    if len(matches)!=1:
        raise ValueError('Native source index omits or repeats requested field')
    row=matches[0]
    path=Path(report['archive']['path'])
    if file_sha256(path)!=report['archive']['sha256']:
        raise ValueError('Native source archive changed')
    cells=sum(math.prod(grid['stored_shape'] if include_ghosts else grid['shape']) for grid in row['grids'])
    # Include resident mapped pages and the retained copy; wider finite checks
    # use bounded chunks rather than one hierarchy-sized temporary.
    bytes_per_cell=48 if row.get('scalar_bytes',8)==16 else 24
    if 64*1024**2+bytes_per_cell*cells>memory_gib*1024**3:
        raise MemoryError('Selected native source hierarchy exceeds reference workspace')
    hierarchy=[]
    for grid in row['grids']:
        scalar_bytes=row.get('scalar_bytes',8)
        if scalar_bytes not in (8,16) or grid.get('scalar_bytes',scalar_bytes)!=scalar_bytes:
            raise ValueError('Unsupported native archive scalar encoding')
        shape=tuple(grid['stored_shape'])
        values=np.memmap(path,mode='r',dtype=report['byte_order']+('f8' if scalar_bytes==8 else 'u8'),offset=grid['byte_offset'],shape=shape if scalar_bytes==8 else (*shape,2))
        halo=grid['halo']
        interior=values if include_ghosts else values[tuple(slice(halo,halo+n) for n in grid['shape'])]
        if scalar_bytes==16:
            from core.music.precision import Binary128Array
            payload=Binary128Array(interior,byte_order=report['byte_order'])
        else:payload=np.array(interior,dtype=float,copy=True)
        record={'level':grid['level'],'origin':grid['origin'],'values':payload}
        if include_ghosts:
            record.update(halo=halo,interior_shape=grid['shape'])
        hierarchy.append(record)
    if file_sha256(path)!=report['archive']['sha256'] or file_sha256(index)!=before:
        raise ValueError('Native source evidence changed during loading')
    return hierarchy,report


def quadratic_source_from_potential(potential,*,level,order=6,halo=4,spectral=False,memory_gib=1.):
    """Independent Hessian invariant before native restriction/DC projection.

    In dimensionless box coordinates the source is
    sum over i<j of (Phi_ii * Phi_jj - Phi_ij**2). Each grid derivative brings
    a factor 2**level; do not insert another box-length conversion here.

    Native order 2 composes centred second-order first derivatives. Native
    orders 4 and 6 both compose fourth-order first derivatives for this source.
    The spectral path zeros complete Nyquist planes, matching the declared
    native operator rather than adding a different alias/zero-mode policy.
    """
    from core.music.precision import Binary128Array
    if isinstance(potential,Binary128Array):
        if type(spectral) is not bool or spectral:
            raise ValueError('Binary128 source archive supports finite-difference potentials only')
        from core.music.precision import wide_derivative
        return wide_derivative(potential,operation='quadratic',level=level,order=order,halo=halo,memory_gib=memory_gib)
    values=np.asarray(potential)
    if (type(level) is not int or not 1<=level<=20 or type(order) is not int or order not in (2,4,6)
            or type(halo) is not int or not 0<=halo<=16 or type(spectral) is not bool
            or values.ndim!=3 or values.dtype.kind not in 'fiu' or any(n<=2*halo for n in values.shape)
            or isinstance(memory_gib,(bool,np.bool_)) or not math.isfinite(memory_gib) or memory_gib<=0):
        raise ValueError('Explicit supported native potential geometry/operator/workspace required')
    shape=tuple(n-2*halo for n in values.shape)
    cells=math.prod(shape)
    if 64*1024**2+8*values.size+256*cells>memory_gib*1024**3:
        raise MemoryError('Quadratic source reference exceeds actual hierarchy workspace')
    if not np.isfinite(values).all():
        raise ValueError('Finite native potential samples required')
    if spectral:
        if shape!=(2**level,)*3:
            raise ValueError('Spectral quadratic reference requires the complete actual uniform box')
        interior=values[tuple(slice(halo,halo+n) for n in shape)]
        fourier=np.fft.rfftn(interior)
        fourier[shape[0]//2,:,:]=0
        fourier[:,shape[1]//2,:]=0
        fourier[:,:,shape[2]//2]=0
        axes=[np.fft.fftfreq(shape[0],d=1/shape[0])[:,None,None]*2*np.pi,
            np.fft.fftfreq(shape[1],d=1/shape[1])[None,:,None]*2*np.pi,
            np.fft.rfftfreq(shape[2],d=1/shape[2])[None,None,:]*2*np.pi]
        def hessian(i,j):
            return np.fft.irfftn(-axes[i]*axes[j]*fourier,s=shape,axes=(0,1,2))
    else:
        radius=2 if order==2 else 4
        if halo<radius:
            raise ValueError('Quadratic finite-difference reference needs actual native potential ghost cells')
        stencil={-1:-.5,1:.5} if order==2 else {-2:1/12,-1:-2/3,1:2/3,2:-1/12}
        def hessian(i,j):
            weights={}
            for a,wa in stencil.items():
                for b,wb in stencil.items():
                    offset=[0,0,0]
                    offset[i]+=a
                    offset[j]+=b
                    key=tuple(offset)
                    weights[key]=weights.get(key,0.)+wa*wb
            result=np.zeros(shape)
            for offset,weight in weights.items():
                result+=weight*values[tuple(slice(halo+d,halo+d+n) for d,n in zip(offset,shape))]
            return result*float(2**level)**2
    diagonal=[hessian(i,i) for i in range(3)]
    source=np.zeros(shape)
    for i in range(3):
        for j in range(i+1,3):
            source+=diagonal[i]*diagonal[j]-hessian(i,j)**2
    if not np.isfinite(source).all():
        raise ValueError('Quadratic reference overflow')
    return source


def poisson_source_from_potential(potential,*,level,order=6,halo=4,spectral=False,memory_gib=1.):
    """Evaluate source = -L Phi in native dimensionless box coordinates.

    A grid spacing h = 2**(-level) gives the finite Laplacian its 1/h**2
    factor. A constant potential is a gauge choice and contributes no source.
    Use actual captured ghosts for nested grids; only a full box may use FFTs.
    """
    from core.music.precision import Binary128Array
    if isinstance(potential,Binary128Array):
        if type(spectral) is not bool or spectral:
            raise ValueError('Binary128 source archive supports finite-difference potentials only')
        from core.music.precision import wide_derivative
        return wide_derivative(potential,operation='poisson',level=level,order=order,halo=halo,memory_gib=memory_gib)
    values=np.asarray(potential)
    if (type(level) is not int or not 1<=level<=20 or type(order) is not int or order not in (2,4,6)
            or type(halo) is not int or not 0<=halo<=16 or type(spectral) is not bool
            or values.ndim!=3 or values.dtype.kind not in 'fiu' or any(n<=2*halo for n in values.shape)
            or isinstance(memory_gib,(bool,np.bool_)) or not math.isfinite(memory_gib) or memory_gib<=0):
        raise ValueError('Explicit supported Poisson potential geometry/operator/workspace required')
    shape=tuple(n-2*halo for n in values.shape)
    cells=math.prod(shape)
    if 64*1024**2+8*values.size+128*cells>memory_gib*1024**3:
        raise MemoryError('Poisson source reference exceeds actual hierarchy workspace')
    if not np.isfinite(values).all():
        raise ValueError('Finite native Poisson potential samples required')
    if spectral:
        if shape!=(2**level,)*3:
            raise ValueError('Spectral Poisson reference requires complete actual uniform box')
        interior=values[tuple(slice(halo,halo+n) for n in shape)]
        axes=[np.fft.fftfreq(shape[0],d=1/shape[0])[:,None,None]*2*np.pi,
            np.fft.fftfreq(shape[1],d=1/shape[1])[None,:,None]*2*np.pi,
            np.fft.rfftfreq(shape[2],d=1/shape[2])[None,None,:]*2*np.pi]
        result=np.fft.irfftn(sum(k*k for k in axes)*np.fft.rfftn(interior),s=shape,axes=(0,1,2))
    else:
        radius=order//2
        if halo<radius:
            raise ValueError('Poisson source reference needs actual potential ghost cells')
        centre,weights={2:(-2.,[1.]),4:(-2.5,[4/3,-1/12]),6:(-49/18,[1.5,-3/20,1/90])}[order]
        result=np.zeros(shape)
        origin=values[tuple(slice(halo,halo+n) for n in shape)]
        for axis in range(3):
            result+=centre*origin
            for shift,weight in enumerate(weights,1):
                for sign in (-1,1):
                    offsets=[0,0,0]
                    offsets[axis]=sign*shift
                    result+=weight*values[tuple(slice(halo+d,halo+d+n) for d,n in zip(offsets,shape))]
        result*=-float(2**level)**2
    if not np.isfinite(result).all():
        raise ValueError('Poisson source reference overflow')
    return result


def quadratic_hierarchy_from_potentials(potentials,*,order=6,memory_gib=1.):
    """Replay composed derivatives, child restriction and global base DC.

    This checks implementation against actual captured potentials. The solved
    potential and its boundary values still require an independent reference.
    """
    from core.zoom import _prepare_cell_average_hierarchy
    if not isinstance(potentials,list) or not potentials:
        raise ValueError('Actual native potential hierarchy required')
    cells=sum(math.prod(row['interior_shape']) for row in potentials)
    if 64*1024**2+512*cells>memory_gib*1024**3:
        raise MemoryError('Quadratic hierarchy reference exceeds allocated workspace')
    raw=[]
    for row in potentials:
        source=quadratic_source_from_potential(row['values'],level=row['level'],order=order,halo=row['halo'],memory_gib=memory_gib)
        if list(source.shape)!=row['interior_shape']:
            raise ValueError('Captured potential interior shape differs')
        raw.append({'level':row['level'],'origin':row['origin'],'values':source})
    restricted,_,_=_prepare_cell_average_hierarchy(raw,0,memory_gib,workspace_cells=cells)
    mean=float(np.mean(restricted[0]['values'],dtype=np.longdouble))
    for row in restricted:
        row['values']-=mean
        row['origin']=row['origin'].tolist()
    return restricted,{'removed_base_mean':mean,'order':order,'source_derivative_order':2 if order==2 else 4,
        'restriction':'Equal eight-child averaging from finest to base before global base-mean subtraction',
        'scope':'Actual native potential/source operator replay; potential and boundary accuracy remain separate'}


def audit_music_archive_quadratic_source(index,configuration,ic,destination,*,memory_gib=1.):
    """Publish measured per-species/level agreement with the source operator."""
    import json
    import h5py
    from core.zoom import _configuration
    from core.data import export_numeric_audit,epoch_contract
    index,configuration,ic=map(lambda p:Path(p).resolve(),(index,configuration,ic))
    pins={str(p):file_sha256(p) for p in (index,configuration,ic)}
    if index.with_name(index.name+'.sha256').read_text().strip()!=pins[str(index)]:
        raise ValueError('Native source index seal mismatch')
    record=json.loads(index.read_text())
    cfg=_configuration(configuration)
    low,high=cfg.getint('setup','levelmin'),cfg.getint('setup','levelmax')
    if not cfg.getboolean('setup','use_2LPT') or not cfg.getboolean('cosmology','mixed_2lpt'):
        raise ValueError('Original mixed 2LPT configuration required for source operator comparison')
    with h5py.File(ic) as f:
        h=f['Header'].attrs
        a=float(h['Time'])
        if (h.get('Music_lpt_order')!=2 or h.get('Music_mixed_fluid_schema')!=1
                or h.get('Music_levelmin')!=low or h.get('Music_levelmax')!=high
                or not math.isclose(a,1/(1+cfg.getfloat('setup','zstart')),rel_tol=1e-12)
                or not math.isclose(float(h['BoxSize']),1000*cfg.getfloat('setup','boxlength'),rel_tol=1e-12)):
            raise ValueError('Native source configuration differs from actual IC hierarchy/frame')
    cells=sum(math.prod(g['shape']) for row in record['fields'] if row['quantity_code']==3 for g in row['grids'])
    if isinstance(memory_gib,(bool,np.bool_)) or not math.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Positive source comparison workspace required')
    if 64*1024**2+512*cells>memory_gib*1024**3:
        raise MemoryError('Species source comparison exceeds allocated hierarchy workspace')
    arrays={}
    checks=[]
    projection=[]
    for species in (0,1):
        potential,_=load_music_source_hierarchy(index,2,species,memory_gib=memory_gib,include_ghosts=True)
        actual,_=load_music_source_hierarchy(index,3,species,memory_gib=memory_gib)
        if any([row['level'] for row in rows]!=list(range(low,high+1)) for rows in (potential,actual)):
            raise ValueError('Native captured quadratic hierarchy differs from configured IC levels')
        if low==high:
            expected=[{'level':low,'origin':potential[0]['origin'],'values':quadratic_source_from_potential(
                potential[0]['values'],level=low,halo=potential[0]['halo'],spectral=True,memory_gib=memory_gib)}]
            metadata={'operator':'spectral Hessian; complete native Nyquist planes zeroed'}
        else:
            expected,metadata=quadratic_hierarchy_from_potentials(potential,order=cfg.getint('poisson','grad_order',fallback=6),memory_gib=memory_gib)
        projection.append({'particle_type':species,**metadata})
        if len(expected)!=len(actual):
            raise ValueError('Native source and potential hierarchy lengths differ')
        for reference,measured in zip(expected,actual):
            if reference['level']!=measured['level'] or reference['origin']!=measured['origin'] or reference['values'].shape!=measured['values'].shape:
                raise ValueError('Native source and potential hierarchy windows differ')
            residual=measured['values']-reference['values']
            normalization=float(np.linalg.norm(reference['values']))
            error=float(np.linalg.norm(residual)/normalization) if normalization>0 else math.nan
            status='UNRESOLVED' if not math.isfinite(error) else 'PASS' if error<=1e-7 else 'FAIL'
            prefix=f'species{species}_level{reference["level"]}_'
            for name,value in (('native_source',measured['values']),('reference_source',reference['values']),('residual',residual)):
                arrays[prefix+name]=value
            checks.append({'particle_type':species,'level':reference['level'],'cells':residual.size,
                'relative_l2_error':error if math.isfinite(error) else None,'status':status})
    pins.update(archive_precision_sources(record))
    pins[record['archive']['path']]=record['archive']['sha256']
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    pins[str((_CORE_ROOT/'zoom.py'))]=file_sha256((_CORE_ROOT/'zoom.py'))
    report={'schema':'native-music-archive-quadratic-source-comparison-v1',
        'status':'FAIL' if any(c['status']=='FAIL' for c in checks) else 'PASS' if all(c['status']=='PASS' for c in checks) else 'UNRESOLVED',
        'checks':checks,'projection':projection,'maximum_relative_operator_error':1e-7,
        'scope':'Independent replay of actual captured quadratic source assembly, including native hierarchy restriction/DC or uniform spectral conventions',
        'potential_accuracy_acceptance':'NOT_ESTABLISHED','boundary_acceptance':'NOT_ESTABLISHED',
        'transfer_window_acceptance':'NOT_ESTABLISHED','science_acceptance':'NOT_ACCEPTED'}
    return export_numeric_audit(destination,report,arrays,{name:{'units':'dimensionless','quantity':name} for name in arrays},
        algorithm=__file__,context={'epoch':epoch_contract(a)},source_hashes=pins)


def hierarchy_leaf_support(hierarchy,index):
    """Rectangular Poisson leaves; independent of the particle-output mask."""
    row=hierarchy[index]
    shape=row['values'].shape
    if len(shape)!=3:
        raise ValueError('Scalar three-dimensional native source grid required')
    mask=np.ones(shape,dtype=bool)
    if index+1<len(hierarchy):
        child=hierarchy[index+1]
        origin=np.asarray(child['origin'])
        parent=np.asarray(row['origin'])
        if child['level']!=row['level']+1 or np.any(origin%2) or any(n%2 for n in child['values'].shape):
            raise ValueError('Aligned contiguous native source hierarchy required')
        start=(origin//2-parent)%2**row['level']
        extent=np.array(child['values'].shape)//2
        if shape!=(2**row['level'],)*3 and np.any(start+extent>shape):
            raise ValueError('Native child source escapes finite parent')
        if np.any(extent>shape):
            raise ValueError('Native child source exceeds parent volume')
        indices=[(int(a)+np.arange(int(n)))%size for a,n,size in zip(start,extent,shape)]
        mask[np.ix_(*indices)]=False
    return mask


def audit_music_archive_poisson(index,configuration,ic,destination,*,memory_gib=1.):
    """Check the original prepared source against -L Phi on actual leaves.

    Uses the unchanged configured multigrid relative tolerance. It never fits
    away a residual mean or substitutes the mutated FAS source for the input.
    This checks equations using actual potential ghosts; an independent
    potential/boundary reference is still required for physical acceptance.
    """
    import json
    import h5py
    from core.zoom import _configuration
    from core.data import export_numeric_audit,epoch_contract
    index,configuration,ic=map(lambda p:Path(p).resolve(),(index,configuration,ic))
    pins={str(p):file_sha256(p) for p in (index,configuration,ic)}
    if index.with_name(index.name+'.sha256').read_text().strip()!=pins[str(index)]:
        raise ValueError('Native source index seal mismatch')
    record=json.loads(index.read_text())
    cfg=_configuration(configuration)
    low,high=cfg.getint('setup','levelmin'),cfg.getint('setup','levelmax')
    tolerance=cfg.getfloat('poisson','accuracy',fallback=1e-5)
    order=cfg.getint('poisson','laplace_order',fallback=6)
    if (not 1<=low<high<=20 or cfg.getboolean('poisson','kspace',fallback=False) or not math.isfinite(tolerance) or tolerance<=0
            or not cfg.getboolean('setup','use_2LPT') or not cfg.getboolean('cosmology','mixed_2lpt')):
        raise ValueError('Declared nested finite-difference Poisson configuration/tolerance required')
    with h5py.File(ic) as f:
        h=f['Header'].attrs
        a=float(h['Time'])
        if (h.get('Music_composite_source_normalization_schema')!=1 or h.get('Music_lpt_order')!=2
                or h.get('Music_levelmin')!=low or h.get('Music_levelmax')!=high
                or not math.isclose(a,1/(1+cfg.getfloat('setup','zstart')),rel_tol=1e-12)
                or not math.isclose(float(h['BoxSize']),1000*cfg.getfloat('setup','boxlength'),rel_tol=1e-12)):
            raise ValueError('Current composite-normalized native IC hierarchy/frame required')
    if isinstance(memory_gib,(bool,np.bool_)) or not math.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Positive Poisson comparison workspace required')
    cells=sum(math.prod(g['shape']) for row in record['fields'] if row['quantity_code'] in (1,3,6) for g in row['grids'])
    if 64*1024**2+512*cells>memory_gib*1024**3:
        raise MemoryError('Physical leaf Poisson comparison exceeds allocated hierarchy workspace')
    arrays={}
    specs={}
    checks=[]
    means=[]
    for species in (0,1):
        for quantity,potential_code,label in ((1,2,'matter'),(3,4,'quadratic'),(6,7,'velocity')):
            source,_=load_music_source_hierarchy(index,quantity,species,memory_gib=memory_gib)
            potential,_=load_music_source_hierarchy(index,potential_code,species,memory_gib=memory_gib,include_ghosts=True)
            if any([row['level'] for row in rows]!=list(range(low,high+1)) for rows in (source,potential)):
                raise ValueError('Native captured Poisson hierarchy differs from configured IC levels')
            if len(source)!=len(potential):
                raise ValueError('Native Poisson source/potential hierarchy lengths differ')
            composite_mean=0.
            for position,(s,p) in enumerate(zip(source,potential)):
                if s['level']!=p['level'] or s['origin']!=p['origin']:
                    raise ValueError('Native Poisson source/potential geometry differs')
                inferred=poisson_source_from_potential(p['values'],level=p['level'],order=order,halo=p['halo'],memory_gib=memory_gib)
                if inferred.shape!=s['values'].shape:
                    raise ValueError('Native potential interior and source shapes differ')
                residual=inferred-s['values']
                support=hierarchy_leaf_support(source,position)
                composite_mean+=float(np.sum(s['values'][support],dtype=np.longdouble))*2.**(-3*s['level'])
                if support.any():
                    norm=float(np.linalg.norm(s['values'][support]))
                    difference=float(np.linalg.norm(residual[support]))
                    error=difference/norm if norm>0 else 0. if difference==0 else math.inf
                    checks.append({'particle_type':species,'quantity':label,'level':s['level'],'leaf_cells':int(support.sum()),
                        'relative_l2_residual':error if math.isfinite(error) else None,
                        'original_source_rms':norm/math.sqrt(int(support.sum())),
                        'unmodified_residual_mean':float(np.mean(residual[support])),
                        'status':'PASS' if error<=tolerance else 'FAIL'})
                prefix=f'species{species}_{label}_level{s["level"]}_'
                unit='km/s per comoving Mpc/h' if quantity==6 else 'dimensionless'
                for name,value in (('original_source',s['values']),('inferred_source',inferred),('raw_residual',residual),('leaf_support',support)):
                    arrays[prefix+name]=value
                    specs[prefix+name]={'units':'dimensionless boolean support' if name=='leaf_support' else unit,'quantity':prefix+name}
            means.append({'particle_type':species,'quantity':label,'original_composite_mean':composite_mean})
    pins.update(archive_precision_sources(record))
    pins[record['archive']['path']]=record['archive']['sha256']
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    report={'schema':'native-music-original-poisson-comparison-v1','status':'PASS' if checks and all(c['status']=='PASS' for c in checks) else 'FAIL',
        'checks':checks,'composite_source_means':means,'maximum_relative_residual':tolerance,'laplace_order':order,
        'scope':'Actual rectangular physical-leaf equations against original pre-solve sources; no residual mean fitted or removed',
        'potential_accuracy_acceptance':'NOT_ESTABLISHED','boundary_accuracy_acceptance':'NOT_ESTABLISHED',
        'transfer_window_acceptance':'NOT_ESTABLISHED','science_acceptance':'NOT_ACCEPTED'}
    return export_numeric_audit(destination,report,arrays,specs,algorithm=__file__,context={'epoch':epoch_contract(a)},source_hashes=pins)


def audit_native_quadratic_velocity(filename,configuration,destination,*,memory_gib=1.):
    """Measure v2 = (D2' / D2) aH Psi2 in each species/hierarchy window.

    The 1e-7 assembly tolerance and 1e-6 native component signal floors are
    retained from the existing native quadratic/unit regression controls.
    """
    import h5py
    from core.foundation import KPC_CM
    from core.data import export_numeric_audit, epoch_contract, snapshot_total_counts
    from core.zoom import _configuration, _native_hierarchy_levels, _native_cell_levels
    if isinstance(memory_gib,(bool,np.bool_)) or not math.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Positive quadratic component workspace required')
    budget=memory_gib*1024**3-64*1024**2
    if budget<1024:raise MemoryError('Quadratic component workspace exceeds allocation')
    chunk=min(100000,int(budget//1024))
    paths=[Path(filename).resolve(),Path(configuration).resolve()]
    pins={str(path):file_sha256(path) for path in paths}
    cfg=_configuration(paths[1]);rows={}
    with h5py.File(paths[0]) as f:
        h=f['Header'].attrs
        if h.get('Music_lpt_order')!=2 or h.get('Music_mixed_fluid_schema')!=1:
            raise ValueError('Retained original native mixed 2LPT components required')
        if h.get('Music_momentum_schema')!=1 or not math.isclose(float(h.get('Music_lagrangian_unitlength_cm',math.nan)),KPC_CM,rel_tol=1e-12):
            raise ValueError('Declared original native coordinate and momentum frame required')
        low,high=_native_hierarchy_levels(h)
        a=float(h['Time']);box=float(h['BoxSize']);hubble=float(h['HubbleParam'])
        c2=float(h['Music_2lpt_displacement_coefficient']);v2=float(h['Music_2lpt_velocity_coefficient'])
        aH=cfg.getfloat('cosmology','lpt2_aH_km_s_Mpc')
        if not all(math.isfinite(v) and v>0 for v in (a,box,hubble,c2,v2,aH)):
            raise ValueError('Positive supported native quadratic growth and frame coefficients required')
        comparisons=[(a,1/(1+cfg.getfloat('setup','zstart'))),(box,1000*cfg.getfloat('setup','boxlength')),
            (hubble,cfg.getfloat('cosmology','H0')/100),(c2,cfg.getfloat('cosmology','lpt2_displacement')),
            (v2,cfg.getfloat('cosmology','lpt2_velocity'))]
        if not all(math.isclose(x,y,rel_tol=1e-12,abs_tol=0) for x,y in comparisons):
            raise ValueError('Quadratic component configuration differs from actual native IC')
        factor=v2/c2*aH/(hubble*1000) # physical km/s per comoving kpc/h
        frame=np.asarray(h['Music_removed_bulk_velocity_gadget'],dtype=float)
        bound=np.asarray(h['Music_max_abs_displacement_native'],dtype=float)
        if (frame.shape!=(3,) or bound.shape!=(3,) or not np.isfinite(frame).all()
                or not np.isfinite(bound).all() or np.any(bound<0) or np.any(bound>=box/4)):
            raise ValueError('Native bulk frame or displacement bounds are invalid')
        counts=snapshot_total_counts(h)
        if h.get('NumFilesPerSnapshot')!=1 or np.any(counts[3:]) or not np.array_equal(counts,h['NumPart_ThisFile']):
            raise ValueError('Complete original single-file matter IC required')
        for t in (0,1,2):
            if not counts[t]:continue
            g=f['PartType'+str(t)];n=int(counts[t])
            if any(g[name].shape!=(n,3) for name in ('Coordinates','LagrangianCoordinates','FirstOrderDisplacements','Velocities','FirstOrderVelocities')) or g['LagrangianCellWidths'].shape!=(n,):
                raise ValueError('Native quadratic component arrays are not aligned')
            for start in range(0,n,chunk):
                end=min(n,start+chunk)
                q=g['LagrangianCoordinates'][start:end];x=g['Coordinates'][start:end]
                linear=g['FirstOrderDisplacements'][start:end]
                velocity=g['Velocities'][start:end]+frame-g['FirstOrderVelocities'][start:end]
                if 'ThermalVelocities' in g:velocity-=g['ThermalVelocities'][start:end]
                if any(not np.isfinite(v).all() for v in (q,x,linear,velocity)) or np.any(q<0) or np.any(q>=box) or np.any(x<0) or np.any(x>=box):
                    raise ValueError('Invalid native quadratic flow components')
                displacement=(x-q+box/2)%box-box/2
                if np.any(np.max(np.abs(displacement),axis=0)>bound+box*1e-12):
                    raise ValueError('Native displacement exceeds unwrapping bounds')
                psi2=displacement-linear;physical_v2=velocity*math.sqrt(a)
                residual=physical_v2-factor*psi2
                levels=_native_cell_levels(g['LagrangianCellWidths'][start:end],box,low,high)
                for level in np.unique(levels):
                    keep=levels==level;key=(t,int(level));row=rows.setdefault(key,np.zeros(5))
                    with np.errstate(over='ignore',invalid='ignore'):
                        row+=np.array([keep.sum(),np.sum(psi2[keep]**2),np.sum(velocity[keep]**2),
                            np.sum(physical_v2[keep]**2),np.sum(residual[keep]**2)])
                    if not np.isfinite(row).all():raise ValueError('Quadratic component norm overflow')
    keys=sorted(rows)
    if not keys:raise ValueError('No actual native quadratic windows')
    values=np.asarray([rows[key] for key in keys]);denominator=np.sqrt(values[:,3])
    supported=(np.sqrt(values[:,1])>1e-6)&(np.sqrt(values[:,2])>1e-6)
    errors=np.divide(np.sqrt(values[:,4]),denominator,out=np.full(len(keys),np.nan),where=denominator>0)
    passed=supported&(errors<=1e-7)
    arrays={'particle_types':np.asarray([key[0] for key in keys],dtype=np.int16),
        'refinement_levels':np.asarray([key[1] for key in keys],dtype=np.int16),
        'particle_counts':values[:,0].astype(np.uint64),'quadratic_displacement_l2':np.sqrt(values[:,1]),
        'quadratic_velocity_l2':denominator,'assembly_residual_l2':np.sqrt(values[:,4]),
        'relative_assembly_error':errors,'signal_supported':supported,'assembly_passed':passed}
    units={name:'dimensionless' for name in arrays}
    units.update(quadratic_displacement_l2='comoving kpc/h',quadratic_velocity_l2='km/s',assembly_residual_l2='km/s',
        particle_types='native species enum',refinement_levels='dyadic refinement level',particle_counts='particle count')
    report={'schema':'native-music-quadratic-assembly-v1','status':'PASS' if passed.all() else 'FAIL' if np.any(supported&~passed) else 'UNRESOLVED',
        'maximum_relative_assembly_error':1e-7,'minimum_native_component_l2':1e-6,
        'physical_velocity_per_displacement':factor,'actual_scale_factor':a,
        'scope':'Original output component assembly and units in separate species/hierarchy windows',
        'quadratic_source_acceptance':'NOT_ESTABLISHED','boundary_acceptance':'NOT_ESTABLISHED','science_acceptance':'NOT_ACCEPTED'}
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    pins[str(_CORE_ROOT/'zoom.py')]=file_sha256(_CORE_ROOT/'zoom.py')
    return export_numeric_audit(destination,report,arrays,{key:{'units':units[key],'quantity':key,
        'missing_policy':'NaN means zero quadratic velocity normalization; signal_supported is false'} for key in arrays},
        algorithm=__file__,context={'epoch':epoch_contract(a)},source_hashes=pins)

