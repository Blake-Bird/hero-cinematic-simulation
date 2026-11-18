"""Native MUSIC phase formats and explicit frozen/seeded model verification.

GRAFIC conversion preserves the selected sign, byte order and axes. Cache
encoding changes representation only. The two replay models have different
conditioning and must not substitute for one another; their source/seed/cache
bindings are checked before moment calculations consume the resulting model.
No format check or replay result establishes physical IC accuracy.
"""
import json
import math
import shutil
import struct
from pathlib import Path
import numpy as np
from core.foundation import file_sha256
from core.music.replay import _budget

_CORE_ROOT = Path(__file__).resolve().parent.parent


def read_frozen_grafic(path,side,*,byte_order,positive_sign,memory_gib):
    """Decode the declared GRAFIC marker/scalar width, sign and axis order.

    Float32 samples promote exactly to float64; float64 samples retain their
    values and sign bits unless the caller explicitly requests a sign flip.
    The source file itself is never rewritten by this reader.
    """
    if (type(side) is not int or side<1 or byte_order not in ('<','>')
            or type(positive_sign) is not bool):
        raise ValueError('Explicit native frozen-file representation required')
    _budget(memory_gib,64*1024**2+32*side**3)
    with Path(path).open('rb') as stream:
        first=stream.read(8)
        stream.seek(0)
        order='little' if byte_order=='<' else 'big'
        marker=4 if (len(first)==8 and int.from_bytes(first[:4],order)==16
            and int.from_bytes(first[4:8],order)==side) else 8
        dtype=np.dtype(byte_order+('u4' if marker==4 else 'u8'))
        def record():
            header=stream.read(marker)
            if len(header)!=marker:
                raise ValueError('Truncated GRAFIC record marker')
            length=int(np.frombuffer(header,dtype=dtype)[0])
            if length not in (16,side*side*4,side*side*8):
                raise ValueError('Unsupported GRAFIC record size')
            payload=stream.read(length)
            tail=stream.read(marker)
            if len(payload)!=length or tail!=header:
                raise ValueError('Truncated or inconsistent GRAFIC record')
            return payload
        header=record()
        if len(header)!=16 or tuple(np.frombuffer(header,dtype=byte_order+'u4')[:3])!=(side,)*3:
            raise ValueError('Frozen GRAFIC prefix dimensions differ')
        result=np.empty((side,)*3)
        itemsize=None
        for k in range(side):
            payload=record()
            width=len(payload)//(side*side)
            if width not in (4,8) or itemsize is not None and itemsize!=width:
                raise ValueError('Inconsistent frozen GRAFIC scalar encoding')
            itemsize=width
            plane=np.frombuffer(payload,dtype=byte_order+f'f{width}').reshape(side,side)
            if not np.isfinite(plane).all():
                raise ValueError('Nonfinite frozen GRAFIC prefix')
            result[:,:,k]=plane.T if positive_sign else -plane.T
        if stream.read(1):
            raise ValueError('Unexpected trailing frozen GRAFIC bytes')
    return result


def publish_seeded_noise_model(request,destination,*,memory_gib=1.):
    """Compare retained caches with replay, conditioning on the actual prefix.

    Noise values are dimensionless. Only later levels draw new innovations;
    the base cache must match the supplied frozen file bit for bit. The saved
    moments describe this fixed-prefix model, not an ensemble of new prefixes.
    """
    from core.data import export_numeric_audit
    from core.zoom import _configuration
    from core.music.seeded import (
        patch_geometry, parent_footprint_indices, replay_raw_patch, refine_patch,
        estimate_seeded_moments, _seed_streams,
    )
    from core.music.replay import load_native_noise
    if (not isinstance(request,dict) or set(request)!={'schema','configuration','mesh','noise_directory','output_shift_box_fraction','byte_order','gsl_library'}
            or request['schema']!='native-local-seeded-model-request-v1' or request['byte_order'] not in ('<','>')
            or not isinstance(request['noise_directory'],str) or not request['noise_directory']):
        raise ValueError('Explicit checksum-bound local seeded-noise model request required')
    destination=Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Preserve previous local seeded model evidence')
    pins={}
    def bound(ref):
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:
            raise ValueError('Exact local model input binding required')
        path=Path(ref['path']).resolve()
        if file_sha256(path)!=ref['sha256']:
            raise ValueError('Local model input changed: '+str(path))
        pins[str(path)]=ref['sha256']
        return path
    configuration=bound(request['configuration'])
    mesh_path=bound(request['mesh'])
    gsl=bound(request['gsl_library'])
    cfg=_configuration(configuration)
    mesh=json.loads(mesh_path.read_text())
    geometry=patch_geometry(mesh,request['output_shift_box_fraction'])
    low=geometry[0]['level']
    high=geometry[-1]['level']
    if (low!=cfg.getint('setup','levelmin') or high!=cfg.getint('setup','levelmax') or low==high):
        raise ValueError('Actual generation bounds and at least one seeded refinement required')
    seeds={g['level']:cfg.getint('random',f"seed[{g['level']}]") for g in geometry[1:]}
    cube=cfg.getint('random','cubesize',fallback=32)
    intervals=_seed_streams(geometry,seeds,cube)
    resources=estimate_seeded_moments(geometry,1)
    _budget(memory_gib,resources['peak_workspace_bytes'])
    frozen=Path(cfg.get('random',f'seed[{low}]')).expanduser()
    if not frozen.is_absolute():
        frozen=configuration.parent/frozen
    frozen=frozen.resolve()
    pins[str(frozen)]=file_sha256(frozen)
    directory=Path(request['noise_directory']).resolve()
    caches=[]
    for g in geometry:
        path=directory/f"wnoise_{g['level']:04d}.bin"
        pins[str(path)]=file_sha256(path)
        caches.append({'level':g['level'],'path':str(path),'sha256':pins[str(path)]})
    base=geometry[0]
    full=read_frozen_grafic(frozen,base['side'],byte_order=request['byte_order'],
        positive_sign=cfg.getboolean('random','grafic_sign',fallback=False),memory_gib=memory_gib)
    previous=load_native_noise(caches[0]['path'],base['shape'],byte_order=request['byte_order'],memory_gib=memory_gib)
    indices=np.ix_(*[(o+np.arange(n))%base['side'] for o,n in zip(base['origin'],base['shape'])])
    if not np.array_equal(previous,full[indices]):
        raise ValueError('Actual prefix cache differs from retained frozen GRAFIC bytes')
    del full
    checks=[{'level':low,'status':'PASS','kind':'FIXED_PREFIX','binding':'BITWISE_EQUAL_TO_RETAINED_FILE',
        'relative_l2_error':0.,'maximum_absolute_error':0.,'difference_l2':0.,'field_l2':float(np.linalg.norm(previous.ravel()))}]
    observed_means=[float(previous.mean())]
    predicted_means=[observed_means[0]]
    for ordinal,(parent,child) in enumerate(zip(geometry,geometry[1:]),1):
        raw=replay_raw_patch(child,seeds[child['level']],gsl_library=gsl,cube_size=cube,memory_gib=memory_gib)
        footprint=previous[parent_footprint_indices(parent,child)]
        predicted=refine_patch(footprint,raw,memory_gib=memory_gib)
        actual=load_native_noise(caches[ordinal]['path'],child['shape'],byte_order=request['byte_order'],memory_gib=memory_gib)
        difference=float(np.linalg.norm((actual-predicted).ravel()))
        scale=float(np.linalg.norm(actual.ravel()))
        relative=difference/scale if scale else 0. if difference==0 else math.inf
        maximum=float(np.max(np.abs(actual-predicted)))
        if not all(math.isfinite(v) for v in (difference,scale,maximum)):
            raise ValueError('Local phase replay norm overflow')
        checks.append({'level':child['level'],'kind':'LOCAL_SEEDED_REFINEMENT','seed':seeds[child['level']],
            'seed_interval':list(intervals[ordinal-1]),'relative_l2_error':relative if math.isfinite(relative) else None,
            'difference_l2':difference,'field_l2':scale,'maximum_absolute_error':maximum,'status':'PASS' if relative<=1e-7 else 'FAIL'})
        observed_means.append(float(actual.mean()))
        predicted_means.append(float(footprint.mean()/math.sqrt(8)))
        previous=actual
        del raw,footprint,predicted
    root=_CORE_ROOT
    for relative in (
        'music/phases.py',
        'music/seeded.py',
        'music/replay.py',
        'software/music2/src/plugins/random_music.cc',
        'software/music2/src/plugins/random_music_wnoise_generator.cc',
        'software/music2/src/plugins/random_music_wnoise_generator.hh',
    ):
        path=root/relative
        pins[str(path)]=file_sha256(path)
    arrays={'replay_difference_l2':np.array([c['difference_l2'] for c in checks]),
        'replay_field_l2':np.array([c['field_l2'] for c in checks]),
        'maximum_absolute_replay_errors':np.array([c['maximum_absolute_error'] for c in checks]),
        'observed_patch_means':np.array(observed_means),'predicted_patch_means':np.array(predicted_means)}
    specs={name:{'units':'dimensionless native white noise','quantity':name} for name in arrays}
    report={'schema':'native-local-seeded-noise-model-v1','status':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',
        'checks':checks,'geometry':geometry,'mesh':mesh,'output_shift_box_fraction':request['output_shift_box_fraction'],
        'conditioning':'fixed_prefix','fixed_prefix_file':{'path':str(frozen),'sha256':pins[str(frozen)]},
        'native_caches':caches,'byte_order':request['byte_order'],'raw_cube_size':cube,
        'maximum_relative_operator_error':1e-7,'resources':resources,
        'noise_model':'Actual retained full prefix fixed; local padded Meyer constraints on disjoint global mt19937 tile seeds and GSL Gaussian ratio draws',
        'scope':'Independent full-cache local phase replay and fixed-prefix model binding. Means are diagnostics; no fitted centring/scaling, physical transfer/boundary acceptance, IC writer or evolution.',
        'launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
    return export_numeric_audit(destination,report,arrays,specs,algorithm=__file__,context={'epoch':{'scale_factor':1/(1+cfg.getfloat('setup','zstart'))}},source_hashes=pins)


def encode_parent_noise(cache,destination,*,level,cache_sha256,memory_gib,disk_gib):
    """Encode dimensionless cache samples as little-endian GRAFIC float64.

    Each z plane is transposed to the native file's axis order, without changing
    signs, normalization or random draws. Readback compares float64 bits,
    including signed zero; the caller receives the input/output bindings.
    """
    from core.zoom import _noise
    if type(level) is not int or not 1<=level<=20:
        raise ValueError('Exact supported parent hierarchy level required')
    if any(type(v) not in (int,float) or not np.isfinite(v) or v<=0 for v in (memory_gib,disk_gib)):
        raise ValueError('Positive finite phase-format allocations required')
    cache=Path(cache).resolve()
    out=Path(destination).resolve()
    side=2**level
    plane_bytes=8*side*side
    size=24+side*(8+plane_bytes)
    if out.exists():
        raise FileExistsError('Preserve earlier parent phase-prefix encoding')
    if plane_bytes>2**31-1 or 32*side**3+64*1024**2>memory_gib*1024**3:
        raise MemoryError('Parent phase-format workspace exceeds allocation')
    if file_sha256(cache)!=cache_sha256:
        raise ValueError('Native parent noise cache changed')
    parent=out.parent
    while not parent.exists():
        parent=parent.parent
    if size>disk_gib*1024**3*.8 or size>shutil.disk_usage(parent).free:
        raise MemoryError('Parent phase-prefix output exceeds declared/free disk')
    values=_noise(cache,side,(0,0,0),memory_gib)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as stream:
        stream.write(struct.pack('<6i',16,side,side,side,0,16))
        marker=struct.pack('<i',plane_bytes)
        for k in range(side):
            stream.write(marker)
            stream.write(np.asarray(values[:,:,k].T,dtype='<f8').tobytes(order='C'))
            stream.write(marker)
    decoded=read_frozen_grafic(out,side,byte_order='<',positive_sign=True,memory_gib=memory_gib)
    if not np.array_equal(values.view('=u8'),decoded.view('=u8')) or file_sha256(cache)!=cache_sha256:
        raise ValueError('Native cache/prefix float64 bits changed during encoding')
    report={'schema':'native-parent-phase-format-v1','status':'ENCODED_NATIVE_CACHE_VALUES_EXACTLY',
        'cache':{'path':str(cache),'sha256':cache_sha256},'output':{'path':str(out),'sha256':file_sha256(out)},'level':level,
        'shape':[side]*3,'byte_order':'<','scalar_bytes':8,'grafic_positive_sign':True,'coordinate_shift_cells':[0,0,0],
        'normalization_applied':False,'drawn_new_noise':False,'encoded_bytes':size,'science_acceptance':'NOT_ACCEPTED',
        'scope':'Native cache array indices and float64 values preserved exactly. Parent-generation ownership and cross-generation native phase audits remain caller requirements.',
        'source_hashes':{str(cache):cache_sha256,str(Path(__file__).resolve()):file_sha256(__file__)}}
    return report


def publish_frozen_noise_model(request,destination,*,memory_gib=1.):
    """Replay a complete frozen hierarchy from its original disjoint seeds.

    Unlike the local fixed-prefix model, this verifies the base draw too.
    Canonical output fields retain the measured input samples, while saved
    residuals compare them with independently reconstructed native noise.
    """
    from core.data import export_numeric_audit
    from core.music.seeded import replay_raw_patch
    from core.music.covariance import frozen_noise_refine
    if (not isinstance(request,dict) or set(request)!={'schema','fields','gsl_library','cube_size','byte_order','positive_sign'}
            or request['schema']!='native-full-frozen-model-request-v1' or request['byte_order'] not in ('<','>')
            or type(request['positive_sign']) is not bool or type(request['cube_size']) is not int
            or request['cube_size']<=0 or request['cube_size']&(request['cube_size']-1)
            or not isinstance(request['fields'],list) or not request['fields']):
        raise ValueError('Explicit retained full fields, native seeds, scalar/sign representation and GSL binding required')
    fields=request['fields']
    levels=[r.get('level') for r in fields]
    if (any(type(l) is not int or not 5<=l<=20 for l in levels) or levels!=list(range(levels[0],levels[-1]+1))):
        raise ValueError('Contiguous retained full hierarchy from level5 or higher required')
    streams=[]
    for row in fields:
        if set(row)!={'level','seed','file'} or type(row['seed']) is not int or row['seed']<=0:
            raise ValueError('Each retained field requires an exact positive original seed and file binding')
        side=2**row['level']
        cube=request['cube_size']
        if side%cube:
            raise ValueError('Native full-field side must be divisible by raw cube size')
        end=row['seed']+(side//cube)**3-1
        if end>0xffffffff or any(row['seed']<=b and a<=end for a,b in streams):
            raise ValueError('Full-frozen original tile streams overlap or exceed uint32')
        streams.append((row['seed'],end))
    _budget(memory_gib,64*1024**2+384*sum(2**(3*l) for l in levels))
    target=Path(destination).resolve()
    work=target.with_name(target.name+'.fields')
    if target.exists() or work.exists():
        raise FileExistsError('Preserve previous frozen model publications and fields')
    pins={}
    def bound(ref):
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:
            raise ValueError('Exact frozen-model input binding required')
        path=Path(ref['path']).resolve()
        if file_sha256(path)!=ref['sha256']:
            raise ValueError('Frozen-model input changed')
        pins[str(path)]=ref['sha256']
        return path
    gsl=bound(request['gsl_library'])
    paths=[bound(row['file']) for row in fields]
    parent=work.parent
    while not parent.exists():
        parent=parent.parent
    if shutil.disk_usage(parent).free<sum(4+8*2**(3*l) for l in levels)+64*1024**2:
        raise OSError('Insufficient free space for canonical frozen-model field records')
    work.mkdir(parents=True,exist_ok=False)
    previous=None
    checks=[]
    norms=[]
    differences=[]
    means=[]
    for ordinal,(row,path,interval) in enumerate(zip(fields,paths,streams)):
        level=row['level']
        side=2**level
        actual=read_frozen_grafic(path,side,byte_order=request['byte_order'],positive_sign=request['positive_sign'],memory_gib=memory_gib)
        raw=replay_raw_patch({'side':side,'shape':(side,)*3,'origin':(0,0,0)},row['seed'],gsl_library=gsl,
            cube_size=request['cube_size'],memory_gib=memory_gib)
        if ordinal==0:
            # The native top-grid generator projects the base draw's DC. This
            # is the declared producer operation, not centring observed data.
            predicted=raw-np.mean(raw,dtype=np.float64)
        else:
            predicted=frozen_noise_refine(previous,raw,memory_gib=memory_gib)
        difference=float(np.linalg.norm((predicted-actual).ravel()))
        norm=float(np.linalg.norm(actual.ravel()))
        maximum=float(np.max(np.abs(predicted-actual)))
        if not all(math.isfinite(v) for v in (difference,norm,maximum)):
            raise ValueError('Full-frozen replay norm overflow')
        relative=difference/norm if norm else 0. if difference==0 else math.inf
        checks.append({'level':level,'seed_interval':list(interval),'relative_l2_error':relative if math.isfinite(relative) else None,
            'maximum_absolute_error':maximum,'status':'PASS' if relative<=1e-7 else 'FAIL'})
        output=work/f'level{level}.bin'
        with output.open('xb') as stream:
            np.asarray([side],dtype='<u4').tofile(stream)
            np.asarray(actual,dtype='<f8').tofile(stream)
        pins[str(output)]=file_sha256(output)
        previous=actual
        differences.append(difference)
        norms.append(norm)
        means.append(float(actual.mean()))
        del raw,predicted
    for name in (
        'music/phases.py',
        'music/seeded.py',
        'music/covariance.py',
        'music/replay.py',
        'software/music2/src/plugins/random_music_wnoise_generator.cc',
        'software/music2/src/plugins/random_music_wnoise_generator.hh',
    ):
        path=_CORE_ROOT/name
        pins[str(path.resolve())]=file_sha256(path)
    arrays={'replay_difference_l2':np.asarray(differences),'replay_field_l2':np.asarray(norms),'observed_field_means':np.asarray(means)}
    report={'schema':'native-frozen-noise-generation-model-control-v1','status':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',
        'checks':checks,'maximum_relative_operator_error':1e-7,'raw_cube_size':request['cube_size'],
        'noise_model':'Disjoint global mt19937 tile streams, GSL Gaussian ratio draws, native zero-mean base and full-cube Meyer refinement',
        'scope':'Independent replay of the supplied retained complete frozen hierarchy from original explicit seed streams. Canonical full fields are exact input values, not predictions. Does not validate arbitrary other fields, local seeded patches, physical transfers/boundaries, evolution or science.',
        'launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
    return export_numeric_audit(target,report,arrays,{k:{'units':'dimensionless native white noise','quantity':k} for k in arrays},
        algorithm=__file__,context={'epoch':{'convention':'Primordial phase field; no physical output epoch'}},source_hashes=pins)
