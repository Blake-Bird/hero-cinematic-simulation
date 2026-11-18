"""MUSIC local seeded refinement, GSL replay and fixed-prefix moments.

The retained prefix is fixed. Subsequent raw tiles carry the declared global
mt19937 seeds and native Gaussian draws. Rectangular cache footprints preserve
periodic coordinates and inherited DC; they are not independent full cubes.
Moment calculation streams one observable at a time and accumulates a Gram
matrix on disk. Importing this module never loads GSL or draws a field.
"""
import ctypes
import ctypes.util
import math
from pathlib import Path
import numpy as np
from core.music.replay import _budget

_CORE_ROOT = Path(__file__).resolve().parent.parent


def _shape(shape):
    if (len(shape)!=3 or any(type(n) is not int or n<8 or n%2 for n in shape)):
        raise ValueError('Three even native patch dimensions of at least eight required')
    return tuple(shape)


def _symbols(shape):
    coarse=tuple(n//2 for n in shape)
    blend=np.ones(coarse)
    phase=np.zeros(coarse)
    indices=[]
    for axis,n in enumerate(coarse):
        k=np.rint(np.fft.fftfreq(n)*n).astype(int)
        view=[1]*3
        view[axis]=n
        wave=k.reshape(view)
        ratio=np.abs(wave)/(n//2)
        blend*=np.where(ratio<=1/3,1.,np.where(ratio<2/3,np.cos(np.pi/2*(3*ratio-1)),0.))
        phase-=np.pi/2*wave/n
        indices.append(k%shape[axis])
    selection=np.ix_(*indices)
    highpass=np.ones(shape)
    highpass[selection]=np.sqrt(np.maximum(0.,1-blend*blend))
    return blend,phase,highpass,selection


def refine_patch(parent,innovation,*,memory_gib=1.):
    """Blend dimensionless parent noise and independent fine-grid innovations.

    Refinement doubles each axis (eight fine cells per coarse cell). The
    sqrt(8) Fourier factor preserves the white-noise convention: a constant
    parent contributes parent/sqrt(8) to each fine cell, not parent itself.
    The phase aligns the native coarse and fine cell centers.
    """
    parent=np.asarray(parent)
    innovation=np.asarray(innovation)
    shape=_shape(innovation.shape)
    if (parent.shape!=tuple(n//2 for n in shape) or parent.dtype.kind not in 'fiu'
            or innovation.dtype.kind not in 'fiu'):
        raise ValueError('Aligned real parent footprint and raw fine patch required')
    _budget(memory_gib,64*1024**2+256*(parent.size+innovation.size))
    if not np.isfinite(parent).all() or not np.isfinite(innovation).all():
        raise ValueError('Finite native patch values required')
    blend,phase,highpass,selection=_symbols(shape)
    spectrum=np.fft.fftn(innovation)*highpass
    spectrum[selection]+=np.fft.fftn(parent)*blend*np.exp(1j*phase)*math.sqrt(8.)
    return np.fft.ifftn(spectrum).real


def refine_patch_adjoint(weights,*,memory_gib=1.):
    """Exact Euclidean transpose: parent footprint and raw fine draw weights."""
    weights=np.asarray(weights)
    shape=_shape(weights.shape)
    if weights.dtype.kind not in 'fiu':
        raise ValueError('Real patch response required')
    _budget(memory_gib,64*1024**2+256*weights.size)
    if not np.isfinite(weights).all():
        raise ValueError('Finite patch response required')
    blend,phase,highpass,selection=_symbols(shape)
    spectrum=np.fft.fftn(weights)
    parent=np.fft.ifftn(spectrum[selection]*blend*np.exp(-1j*phase)/math.sqrt(8.)).real
    fine=np.fft.ifftn(spectrum*highpass).real
    return parent,fine


def patch_geometry(mesh,output_shift_box_fraction):
    """Native cache coordinates from the bound computational-to-box shift."""
    rows=mesh['hierarchies']['convolution']
    margin=mesh['convolution_margin']
    levels=[r['level'] for r in rows]
    if (not levels or any(type(l) is not int or not 2<=l<=20 for l in levels)
            or levels!=list(range(levels[0],levels[-1]+1)) or type(margin) is not int):
        raise ValueError('Contiguous actual convolution hierarchy required')
    shift=np.asarray(output_shift_box_fraction,dtype=float)
    if shift.shape!=(3,) or not np.isfinite(shift).all():
        raise ValueError('Actual bound output-to-original-box shift required')
    result=[]
    for row in rows:
        level=row['level']
        side=2**level
        cells=shift*side
        rounded=np.rint(cells)
        if np.any(np.abs(cells-rounded)>1e-9):
            raise ValueError('Bound box shift does not align with native cells')
        shape=row['shape']
        offset=row['offset']
        if (len(shape)!=3 or len(offset)!=3 or any(type(n) is not int for n in (*shape,*offset))):
            raise ValueError('Exact native convolution bounds required')
        pads=[0]*3 if level==levels[0] else [margin if margin>0 else n//2 for n in shape]
        padded=_shape(tuple(n+2*p for n,p in zip(shape,pads)))
        origin=tuple(int(o+s-p) for o,s,p in zip(offset,rounded,pads))
        # Refined patches with repeated physical cells have conflicting native
        # in-place writes. Do not assign such a geometry this covariance model.
        if any(n>side for n in padded):
            raise ValueError('Seeded patch exceeds its periodic box')
        if level==levels[0] and padded!=(side,)*3:
            raise ValueError('Conditioned prefix must provide a full periodic base cache')
        result.append({'level':level,'shape':padded,'origin':origin,'side':side})
    return result


def parent_footprint_indices(parent,child):
    """Locate every native coarse read in retained parent coordinates."""
    if child['level']!=parent['level']+1:
        raise ValueError('Adjacent refinement levels required')
    indices=[]
    for origin,n,p_origin,p_size in zip(child['origin'],child['shape'],parent['origin'],parent['shape']):
        # C++ signed integer division truncates toward zero, including x0<0.
        coarse_origin=origin//2 if origin>=0 else -((-origin)//2)
        index=(coarse_origin+np.arange(n//2)-p_origin)%parent['side']
        if np.any(index>=p_size):
            raise ValueError('Native coarse footprint extends outside conditioned/refined parent cache')
        indices.append(index)
    return np.ix_(*indices)


def raw_patch_layout(geometry,seed,*,cube_size=32):
    """Global mt19937 tile seeds and within-tile coordinates, axis factored."""
    side=geometry['side']
    shape=geometry['shape']
    origin=geometry['origin']
    if (type(side) is not int or side<1 or type(seed) is not int or seed<=0
            or type(cube_size) is not int or cube_size<=0 or cube_size&(cube_size-1)
            or side%cube_size or any(n>side or n<=0 for n in shape)):
        raise ValueError('Positive disjoint native seed and nonrepeating periodic tile geometry required')
    cubes=side//cube_size
    if seed+cubes**3-1>0xffffffff:
        raise ValueError('Native mt19937 tile seed exceeds uint32 range')
    coordinates=[(o+np.arange(n,dtype=np.int64))%side for o,n in zip(origin,shape)]
    return [(q//cube_size,q%cube_size) for q in coordinates]



def replay_raw_patch(geometry,seed,*,gsl_library=None,cube_size=32,memory_gib=1.):
    """Replay actual GSL draws tile by tile; no effective finest full cube."""
    axes=raw_patch_layout(geometry,seed,cube_size=cube_size)
    _budget(memory_gib,64*1024**2+16*math.prod(geometry['shape'])+16*cube_size**3)
    library=gsl_library or ctypes.util.find_library('gsl')
    if not library:
        raise RuntimeError('Native GSL library required for seeded-noise replay')
    gsl=ctypes.CDLL(str(library))
    kind=ctypes.c_void_p.in_dll(gsl,'gsl_rng_mt19937')
    gsl.gsl_rng_alloc.argtypes=[ctypes.c_void_p]
    gsl.gsl_rng_alloc.restype=ctypes.c_void_p
    gsl.gsl_rng_set.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    gsl.gsl_ran_ugaussian_ratio_method.argtypes=[ctypes.c_void_p]
    gsl.gsl_ran_ugaussian_ratio_method.restype=ctypes.c_double
    gsl.gsl_rng_free.argtypes=[ctypes.c_void_p]
    rng=gsl.gsl_rng_alloc(kind)
    if not rng:
        raise MemoryError('GSL random generator allocation failed')
    result=np.empty(geometry['shape'])
    cubes=geometry['side']//cube_size
    try:
        for i in np.unique(axes[0][0]):
            for j in np.unique(axes[1][0]):
                for k in np.unique(axes[2][0]):
                    gsl.gsl_rng_set(rng,seed+(int(i)*cubes+int(j))*cubes+int(k))
                    tile=np.fromiter((gsl.gsl_ran_ugaussian_ratio_method(rng) for _ in range(cube_size**3)),dtype=float,count=cube_size**3).reshape((cube_size,)*3)
                    chosen=[np.flatnonzero(axis[0]==q) for axis,q in zip(axes,(i,j,k))]
                    result[np.ix_(*chosen)]=tile[np.ix_(*[axis[1][idx] for axis,idx in zip(axes,chosen)])]
    finally:gsl.gsl_rng_free(rng)
    return result


def hierarchy_response(responses,geometry,*,memory_gib=1.):
    """Transpose one real observable, bounded by local patch sizes.

    Returns fixed-prefix weights plus per-level innovation weights. Call once
    per observable and persist rows to disk for a bounded Gram calculation.
    Response count does not multiply the live FFT workspace.
    """
    levels=[g['level'] for g in geometry]
    if not levels or set(responses)!=set(levels):
        raise ValueError('Every actual noise-cache response required')
    _budget(memory_gib,64*1024**2+288*sum(math.prod(g['shape']) for g in geometry))
    pending={}
    for g in geometry:
        w=np.asarray(responses[g['level']])
        if w.shape!=tuple(g['shape']) or w.dtype.kind not in 'fiu' or not np.isfinite(w).all():
            raise ValueError('Aligned finite scalar noise response required')
        pending[g['level']]=w.astype(float,copy=True)
    result={}
    for position in range(len(geometry)-1,0,-1):
        child=geometry[position]
        parent=geometry[position-1]
        coarse,result[child['level']]=refine_patch_adjoint(pending.pop(child['level']),memory_gib=memory_gib)
        np.add.at(pending[parent['level']],parent_footprint_indices(parent,child),coarse)
    result[levels[0]]=pending[levels[0]]
    return result


def estimate_seeded_moments(geometry,real_outputs,*,chunk_cells=131072):
    """Explicit disk/FFT bounds; neither full finest cube nor all-row FFTs."""
    levels=[g['level'] for g in geometry]
    if (not levels or levels!=list(range(levels[0],levels[-1]+1))
            or type(real_outputs) is not int or real_outputs<=0
            or type(chunk_cells) is not int or chunk_cells<=0):
        raise ValueError('Contiguous local geometry and positive row/chunk counts required')
    for g in geometry:
        if (type(g['side']) is not int or g['side']!=2**g['level']
                or len(g['shape'])!=3 or len(g['origin'])!=3
                or any(type(n) is not int or n<8 or n%2 or n>g['side'] for n in g['shape'])
                or any(type(o) is not int for o in g['origin'])):
            raise ValueError('Exact nonrepeating native local patch geometry required')
    if tuple(geometry[0]['shape'])!=(geometry[0]['side'],)*3:
        raise ValueError('Full retained periodic prefix cache required')
    for p,c in zip(geometry,geometry[1:]):
        parent_footprint_indices(p,c)
    cells=sum(math.prod(g['shape']) for g in geometry)
    stochastic=sum(math.prod(g['shape']) for g in geometry[1:])
    # Conservative reference FFT bound plus supplied/current row, result rows,
    # fixed prefix and finite checks. Producer workspaces are separate.
    row_peak=64*1024**2+416*cells
    gram_peak=(64*1024**2+16*math.prod(geometry[0]['shape'])
        +64*(real_outputs+1)*min(chunk_cells,max(1,stochastic))+64*real_outputs**2)
    return {'schema':'native-local-seeded-moments-resources-v1','real_outputs':real_outputs,
        'levels':levels,'local_cache_cells':cells,'stochastic_patch_cells':stochastic,
        'peak_workspace_bytes':max(row_peak,gram_peak),'response_row_workspace_bytes':row_peak,
        'gram_workspace_bytes':gram_peak,'innovation_disk_bytes':8*real_outputs*stochastic,
        'chunk_cells':chunk_cells,'conditioning':'Entire actual retained prefix fixed',
        'strategy':'One observable transpose at a time; raw innovation rows on disk; cell-chunk Gram accumulation',
        'full_effective_finest_cube_allocated':False,
        'excluded':'Upstream source/Poisson response producer, input products, filesystem cache and output publication',
        'measured_peak_rss':None,'science_acceptance':'NOT_ACCEPTED'}


def _seed_streams(geometry,seeds,cube_size):
    if (type(cube_size) is not int or cube_size<=0 or cube_size&(cube_size-1)
            or set(seeds)!={g['level'] for g in geometry[1:]}):
        raise ValueError('Every seeded refinement stream and power-of-two tile size required')
    intervals=[]
    for g in geometry[1:]:
        seed=seeds[g['level']]
        if g['side']%cube_size or type(seed) is not int or seed<=0:
            raise ValueError('Positive representable native refinement tile seed required')
        end=seed+(g['side']//cube_size)**3-1
        if end>0xffffffff or any(seed<=b and a<=end for a,b in intervals):
            raise ValueError('Native seeded refinement streams overlap or exceed uint32')
        intervals.append((seed,end))
    return intervals


def stream_fixed_prefix_moments(response_rows,geometry,seeds,fixed_prefix,destination,*,
        real_outputs,memory_gib=1.,disk_gib=1.,chunk_cells=131072,cube_size=32):
    """Conditional mean/C of distinct local observables, without fitting.

    The caller must independently bind native configuration, raw seeds, caches
    and model provenance. This mathematical component does not certify them.
    Innovations within a patch are unique periodic cell identities. Disjoint
    native tile-seed intervals make the levels independent. Overlapping cases
    require a separately constructed joint covariance model. This
    single-hierarchy calculation does not establish cross-case independence.
    """
    from core.foundation import file_sha256,atomic_json
    resources=estimate_seeded_moments(geometry,real_outputs,chunk_cells=chunk_cells)
    intervals=_seed_streams(geometry,seeds,cube_size)
    _budget(memory_gib,resources['peak_workspace_bytes'])
    if isinstance(disk_gib,(bool,np.bool_)) or not math.isfinite(disk_gib) or disk_gib<=0:
        raise ValueError('Explicit positive innovation disk budget required')
    if resources['innovation_disk_bytes']>disk_gib*1024**3:
        raise OSError('Local seeded innovation rows exceed declared disk budget')
    prefix=np.asarray(fixed_prefix)
    if prefix.shape!=tuple(geometry[0]['shape']) or prefix.dtype.kind not in 'fiu' or not np.isfinite(prefix).all():
        raise ValueError('Actual finite conditioned prefix cache required')
    target=Path(destination).resolve()
    if target.exists():
        raise FileExistsError('Preserve previous seeded-moments workspaces')
    # Preflight free capacity in addition to the user's explicit allocation.
    import shutil
    ancestor=target.parent
    while not ancestor.exists():
        ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free<resources['innovation_disk_bytes']:
        raise OSError('Insufficient free space for seeded innovation responses')
    target.mkdir(parents=True,exist_ok=False)
    streams={g['level']:(target/f"innovation-level{g['level']}.f64") for g in geometry[1:]}
    mean=np.zeros(real_outputs)
    iterator=iter(response_rows)
    from contextlib import ExitStack
    with ExitStack() as stack:
        files={l:stack.enter_context(p.open('xb')) for l,p in streams.items()}
        for ordinal in range(real_outputs):
            try:row=next(iterator)
            except StopIteration:raise ValueError('Missing declared local response rows') from None
            mapped=hierarchy_response(row,geometry,memory_gib=memory_gib)
            mean[ordinal]=np.sum(mapped[geometry[0]['level']]*prefix,dtype=np.longdouble)
            for level,stream in files.items():
                np.asarray(mapped[level],dtype='<f8').tofile(stream)
            del mapped,row
        try:next(iterator)
        except StopIteration:pass
        else:raise ValueError('Extra undeclared local response rows')
    covariance=np.zeros((real_outputs,real_outputs))
    for g in geometry[1:]:
        cells=math.prod(g['shape'])
        path=streams[g['level']]
        if path.stat().st_size!=real_outputs*cells*8:
            raise ValueError('Incomplete local innovation response file')
        with path.open('rb') as stream:
            for start in range(0,cells,chunk_cells):
                count=min(chunk_cells,cells-start)
                block=np.empty((real_outputs,count))
                for row in range(real_outputs):
                    stream.seek(8*(row*cells+start))
                    data=np.fromfile(stream,dtype='<f8',count=count)
                    if len(data)!=count or not np.isfinite(data).all():
                        raise ValueError('Invalid persisted innovation row')
                    block[row]=data
                covariance+=block@block.T
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError('Conditional local moments overflow')
    metadata={'schema':'native-local-seeded-conditional-moments-v1','resources':resources,
        'seed_intervals':[list(i) for i in intervals],'fixed_prefix_level':geometry[0]['level'],
        'mean_model':'Exact retained prefix response; local DC is inherited from the coarse footprint',
        'innovation_files':[{'level':l,'path':str(p),'sha256':file_sha256(p),'dtype':'<f8',
            'shape':[real_outputs,math.prod(g['shape'])]} for g in geometry[1:] for l,p in [(g['level'],streams[g['level']])]],
        'scope':'Finite conditional moments component; native provenance verified separately; no cross-case independence assumption',
        'regularization':'None','launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(target/'moments-workspace.json',metadata)
    return mean,covariance,metadata


def validate_seeded_covariance(settings):
    if (not isinstance(settings,dict) or set(settings)!={'disk_gib','chunk_cells'}
            or isinstance(settings['disk_gib'],(bool,np.bool_))
            or not isinstance(settings['disk_gib'],(float,int)) or not math.isfinite(settings['disk_gib']) or settings['disk_gib']<=0
            or type(settings['chunk_cells']) is not int or settings['chunk_cells']<=0):
        raise ValueError('Explicit positive local seeded covariance disk budget and integer cell chunk required')


def verified_seeded_window_moments(product,report,model,parent_conditioning,pins,*,memory_gib,
        workspace,settings,response_arrays=None):
    from core.music.covariance import NativeResponseRows
    if response_arrays is None:
        geometry=patch_geometry(report['mesh'],report['output_shift_box_fraction'])
        with NativeResponseRows(product,{g['level']:g['shape'] for g in geometry},len(report['channels']),memory_gib=memory_gib) as rows:
            return verified_seeded_window_moments(product,report,model,parent_conditioning,pins,memory_gib=memory_gib,
                workspace=workspace,settings=settings,response_arrays=rows)
    from core.foundation import file_sha256,canonical_digest
    from core.data import numeric_audit_source_bindings
    from core.music.replay import load_native_noise
    validate_seeded_covariance(settings)
    if parent_conditioning!='fixed_prefix' or model.get('conditioning')!='fixed_prefix' or model.get('status')!='PASS':
        raise ValueError('Verified local model requires explicit fixed-prefix conditioning')
    sources=numeric_audit_source_bindings(report)
    numeric_audit_source_bindings(model)
    geometry=patch_geometry(report['mesh'],report['output_shift_box_fraction'])
    levels=[g['level'] for g in geometry]
    checks=model['checks'][:len(levels)]
    caches=model['native_caches'][:len(levels)]
    if (len(levels)<2 or [r['level'] for r in checks]!=levels or [r['level'] for r in caches]!=levels
            or any(r.get('status')!='PASS' for r in model['checks'])
            or canonical_digest(geometry)!=canonical_digest(model['geometry'][:len(levels)])):
        raise ValueError('Verified local phase model differs from actual cache geometry or hierarchy prefix')
    count=len(report['channels'])
    resources=estimate_seeded_moments(geometry,count,chunk_cells=settings['chunk_cells'])
    streamed=isinstance(response_arrays,NativeResponseRows)
    resident=3*(response_arrays.row_bytes if streamed else sum(v.nbytes for v in response_arrays.values()))
    _budget(memory_gib,resident+resources['peak_workspace_bytes'])
    if not streamed:
        for g in geometry:
            value=response_arrays[f"response_noise_level{g['level']}"]
            if value.shape!=(count,)+tuple(g['shape']) or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
                raise ValueError('Actual finite local cached-noise response rows required')
    cache_binding=[]
    prefix_path=None
    for g,binding in zip(geometry,caches):
        paths=[Path(p) for p in sources if Path(p).name==f"wnoise_{g['level']:04d}.bin"]
        if len(paths)!=1 or file_sha256(paths[0])!=binding['sha256']:
            raise ValueError('Actual response cache differs from the independently verified local phase cache')
        cache_binding.append({'level':g['level'],'native_cache':{'path':str(paths[0]),'sha256':binding['sha256']},
            'verified_cache':binding,'origin_original_cells':list(g['origin']),'status':'HASH_IDENTICAL'})
        if g==geometry[0]:
            prefix_path=paths[0]
    prefix=load_native_noise(prefix_path,geometry[0]['shape'],byte_order=model['byte_order'],memory_gib=memory_gib)
    seeds={row['level']:row['seed'] for row in checks[1:]}
    rows=(response_arrays.row(i) if streamed else {g['level']:response_arrays[f"response_noise_level{g['level']}"][i] for g in geometry} for i in range(count))
    mean,covariance,metadata=stream_fixed_prefix_moments(rows,geometry,seeds,prefix,workspace,
        real_outputs=count,memory_gib=memory_gib-resident/1024**3,disk_gib=settings['disk_gib'],
        chunk_cells=settings['chunk_cells'],cube_size=model['raw_cube_size'])
    for row in metadata['innovation_files']:
        pins[row['path']]=row['sha256']
    path=Path(workspace)/'moments-workspace.json'
    pins[str(path)]=file_sha256(path)
    for name in (
        'music/seeded.py',
        'music/covariance.py',
        'numeric.py',
        'data.py',
        'foundation.py',
    ):
        path=(_CORE_ROOT / name)
        pins[str(path)]=file_sha256(path)
    metadata.update(input_response_residency_bytes=resident,noise_model='Verified fixed retained prefix with seeded local refinement',
        response_input_strategy='Sequential one-observable NPZ member streams' if streamed else 'All supplied response arrays resident and budgeted',
        joint_case_covariance='Not computed: per-hierarchy moments; overlapping cases retain shared innovation identities')
    return mean,covariance,cache_binding,metadata
