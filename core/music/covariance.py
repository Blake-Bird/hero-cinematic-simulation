"""Verified MUSIC noise moments, window covariance and Gaussian checks.

Real response rows map the same native innovations to measured observables.
Full-frozen, fixed-parent and fixed-prefix local-seeded models stay explicit;
frozen bytes alone do not establish a Gaussian ensemble. Covariance retains
mixed channel units. Correlation and Gaussian checks use dimensionless scaling,
without regularization, eigenvalue clipping or physical-acceptance claims.
"""
import json
import math
import re
import zipfile
from pathlib import Path
import numpy as np
from core.music.replay import _budget

_CORE_ROOT = Path(__file__).resolve().parent.parent


def _refinement_symbols(side):
    axes=[]
    indices=[]
    for axis in range(3):
        k=np.rint(np.fft.fftfreq(side)*side).astype(int)
        shape=[1,1,1]
        shape[axis]=side
        axes.append(k.reshape(shape))
        indices.append(k%(2*side))
    blend=np.ones((side,)*3)
    phase=np.zeros_like(blend)
    for k in axes:
        ratio=np.abs(k)/(side//2)
        blend*=np.where(ratio<=1/3,1.,np.where(ratio<2/3,np.cos(np.pi/2*(3*ratio-1)),0.))
        phase-=np.pi/2*k/side
    innovation=np.ones((2*side,)*3)
    selection=np.ix_(*indices)
    innovation[selection]=np.sqrt(np.maximum(0.,1-blend*blend))
    return blend,phase,innovation,selection


def frozen_noise_refine(parent,innovation,*,memory_gib=1.):
    """Refine the declared full-periodic unit-Gaussian Meyer field.

    The fine cube has eight times as many cells. With NumPy's FFT convention
    the parent spectrum carries sqrt(8); a constant parent therefore becomes
    parent/sqrt(8) in real fine cells. This is white-noise normalization,
    not density interpolation or a fitted variance correction.
    """
    parent=np.asarray(parent)
    innovation=np.asarray(innovation)
    if (parent.ndim!=3 or len(set(parent.shape))!=1 or parent.shape[0]<4 or parent.shape[0]%2
            or innovation.shape!=tuple(2*n for n in parent.shape) or parent.dtype.kind not in 'fiu' or innovation.dtype.kind not in 'fiu'):
        raise ValueError('Aligned real full-cube parent and independent fine draw required')
    _budget(memory_gib,64*1024**2+256*(parent.size+innovation.size))
    if not np.isfinite(parent).all() or not np.isfinite(innovation).all():
        raise ValueError('Finite native Gaussian fields required')
    blend,phase,highpass,selection=_refinement_symbols(parent.shape[0])
    spectrum=np.fft.fftn(innovation)*highpass
    spectrum[selection]+=np.fft.fftn(parent)*blend*np.exp(1j*phase)*math.sqrt(8.)
    return np.fft.ifftn(spectrum).real


def frozen_noise_refine_adjoint(weights,*,memory_gib=1.):
    """Return coarse-field and independent-fine-draw response weights."""
    values=np.asarray(weights)
    if values.ndim!=3 or len(set(values.shape))!=1 or values.shape[0]<8 or values.shape[0]%4 or values.dtype.kind not in 'fiu':
        raise ValueError('Real full-cube refinement response required')
    _budget(memory_gib,64*1024**2+256*values.size)
    if not np.isfinite(values).all():
        raise ValueError('Finite refinement weights required')
    blend,phase,highpass,selection=_refinement_symbols(values.shape[0]//2)
    spectrum=np.fft.fftn(values)
    coarse=np.fft.ifftn(spectrum[selection]*blend*np.exp(-1j*phase)/math.sqrt(8.)).real
    fine=np.fft.ifftn(spectrum*highpass).real
    return coarse,fine


def frozen_hierarchy_innovation_response(responses,mesh,output_shift_box_fraction,*,memory_gib=1.):
    """Map simultaneous real cached-noise responses to independent raw draws.

    responses[level] has shape (number_of_real_outputs, *cached_noise_shape).
    Every periodic cache overlap accumulates onto its original full cube.
    This route allocates the *frozen prefix* full cubes, never an unretained
    effective finest production grid. A local seeded extension needs its own
    declared covariance route and is intentionally rejected by callers.
    """
    tf=mesh['hierarchies']['convolution']
    margin=mesh['convolution_margin']
    shift=np.asarray(output_shift_box_fraction)
    levels=[row['level'] for row in tf]
    if (not levels or levels!=list(range(levels[0],levels[-1]+1)) or set(responses)!=set(levels)
            or any(type(level) is not int or not 2<=level<=20 for level in levels)
            or type(margin) is not int or margin<=0 or shift.shape!=(3,) or not np.isfinite(shift).all()):
        raise ValueError('Actual full frozen hierarchy and output-to-original-box shift required')
    first=np.asarray(responses[levels[0]])
    count=first.shape[0] if first.ndim==4 else 0
    if count<=0:
        raise ValueError('At least one real window response required')
    cells=sum((2**level)**3 for level in levels)
    _budget(memory_gib,64*1024**2+(24*count+512)*cells)
    full={}
    for row in tf:
        level=row['level']
        side=2**level
        pad=0 if level==levels[0] else margin
        expected=tuple(n+2*pad for n in row['shape'])
        value=np.asarray(responses[level])
        if value.shape!=(count,)+expected or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError('Aligned finite cached-noise response rows required')
        offset=shift*side
        rounded=np.rint(offset)
        if np.any(np.abs(offset-rounded)>1e-9):
            raise ValueError('Original frozen noise shift does not align with actual cells')
        origin=np.asarray(row['offset'])-pad+rounded.astype(np.int64)
        indices=[(int(o)+np.arange(n))%side for o,n in zip(origin,expected)]
        full[level]=np.zeros((count,side,side,side))
        for i in range(count):
            # add.at, rather than indexed assignment, preserves repeated
            # periodic images if a cache footprint spans more than one box.
            np.add.at(full[level][i],np.ix_(*indices),value[i])
    result={}
    for level in reversed(levels[1:]):
        result[level]=np.empty_like(full[level])
        for i in range(count):
            coarse,fine=frozen_noise_refine_adjoint(full[level][i],memory_gib=memory_gib)
            full[level-1][i]+=coarse
            result[level][i]=fine
        del full[level]
    result[levels[0]]=full[levels[0]]
    return result,{'schema':'native-full-frozen-innovation-response-v1','levels':levels,'real_outputs':count,
        'noise_model':'Native full-cube Meyer parent constraint with disjoint unit-Gaussian innovation streams',
        'applicability':'Requires bound producer/seed/stream/field provenance; not inferred from frozen bytes alone',
        'periodic_cache_overlap':'Accumulated at original full-cube cell identities',
        'full_cube_levels':levels,'local_seeded_extension_supported':False,'science_acceptance':'NOT_ACCEPTED','launch_authorized':False}


def frozen_response_moments(innovation_response,*,fixed_parent=None,memory_gib=1.):
    """Exact finite real covariance and conditional mean for this noise model.

    For independent unit innovations z_l and real rows R_l, y=sum(R_l*z_l)
    has C=sum(R_l*R_l.T). The ensemble base includes its zero-mean projector.
    With fixed_parent, m=R_base*parent and covariance excludes that base draw.

    With fixed_parent, the parent field is conditioned, not re-randomized.
    Otherwise the base draw is projected to zero mean as in the native top
    generator. No ridge, eigenvalue clipping or empirical variance fit occurs.
    """
    levels=sorted(innovation_response)
    if (not levels or any(type(level) is not int or not 2<=level<=20 for level in levels)
            or levels!=list(range(levels[0],levels[-1]+1))):raise ValueError('Contiguous declared native innovation response required')
    rows=[np.asarray(innovation_response[level]) for level in levels]
    count=rows[0].shape[0] if rows[0].ndim==4 else 0
    if not count or any(row.shape!=(count,)+(2**level,)*3 or row.dtype.kind not in 'fiu' for level,row in zip(levels,rows)):
        raise ValueError('Aligned real full-cube innovation response rows required')
    _budget(memory_gib,64*1024**2+24*sum(row.size for row in rows)+32*count*count)
    if any(not np.isfinite(row).all() for row in rows):
        raise ValueError('Finite covariance responses required')
    mean=np.zeros(count)
    covariance=np.zeros((count,count))
    start=0
    if fixed_parent is not None:
        parent=np.asarray(fixed_parent)
        if parent.shape!=rows[0].shape[1:] or parent.dtype.kind not in 'fiu' or not np.isfinite(parent).all():
            raise ValueError('Actual aligned conditioned parent field required')
        mean=rows[0].reshape(count,-1)@parent.ravel()
        start=1
    for position,row in enumerate(rows[start:],start):
        weights=row.reshape(count,-1)
        if position==0:
            weights=weights-np.mean(weights,axis=1,keepdims=True)
        covariance+=weights@weights.T
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError('Native response moment overflow')
    return mean,covariance


def condition_gaussian_observables(mean,covariance,observed_indices,observed_values,*,
        maximum_condition_number=1e12,memory_gib=1.):
    """Condition explicit Gaussian linear observables, e.g. aperture mass.

    With retained K and observed O, the conditional mean adds
    C_KO solve(C_OO, observed-m_O); covariance subtracts
    C_KO solve(C_OO, C_OK). Use solves, not an explicit matrix inverse.

    The observed indices must be declared independently of the tested mode
    residuals. For a mass-normalized window, include its mass contrast in the
    joint response and condition on that measured mass; then the denominator
    used in the mode response is fixed. No residual mean or variance is fit.
    Singular/ill-conditioned blocks remain unresolved with the same1e12
    default as the existing window Gaussian check; no ridge or clipping.
    """
    mu=np.asarray(mean)
    cov=np.asarray(covariance)
    indices=np.asarray(observed_indices)
    observed=np.asarray(observed_values)
    if (mu.ndim!=1 or len(mu)<2 or mu.dtype.kind not in 'fiu' or cov.shape!=(len(mu),len(mu)) or cov.dtype.kind not in 'fiu'
            or indices.ndim!=1 or not 0<len(indices)<len(mu) or indices.dtype.kind not in 'iu'
            or np.any(indices<0) or np.any(indices>=len(mu)) or len(np.unique(indices))!=len(indices)
            or observed.shape!=indices.shape or observed.dtype.kind not in 'fiu'
            or isinstance(maximum_condition_number,(bool,np.bool_)) or not math.isfinite(maximum_condition_number) or maximum_condition_number<1):
        raise ValueError('Explicit joint real Gaussian moments and distinct observed indices/values required')
    _budget(memory_gib,64*1024**2+128*cov.size)
    if not np.isfinite(mu).all() or not np.isfinite(cov).all() or not np.isfinite(observed).all():
        raise ValueError('Finite conditioning inputs required')
    if not np.allclose(cov,cov.T,rtol=1e-12,atol=32*np.finfo(float).eps*np.max(np.abs(cov))):
        raise ValueError('Symmetric original joint covariance required')
    keep=np.array([i for i in range(len(mu)) if i not in set(indices.tolist())],dtype=int)
    block=cov[np.ix_(indices,indices)]
    eigen=np.linalg.eigvalsh(block)
    report={'schema':'explicit-gaussian-observable-conditioning-v1','observed_indices':indices.tolist(),
        'retained_indices':keep.tolist(),'maximum_condition_number':maximum_condition_number,
        'observed_values':observed.tolist(),'conditioning':'Explicit measured linear observables; not tested-mode residual fitting',
        'launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
    if eigen[-1]<=0 or eigen[0]<=eigen[-1]/maximum_condition_number:
        return None,None,{**report,'status':'UNRESOLVED_COVARIANCE','reason':'Observed covariance block is singular or ill-conditioned'}
    cross=cov[np.ix_(keep,indices)]
    gain=np.linalg.solve(block,cross.T).T
    conditional_mean=mu[keep]+gain@(observed-mu[indices])
    conditional_covariance=cov[np.ix_(keep,keep)]-gain@cross.T
    # Restore mathematical symmetry lost only through matrix-operation
    # rounding. Eigenvalues and diagonal entries are never altered or clipped.
    conditional_covariance=(conditional_covariance+conditional_covariance.T)/2
    if not np.isfinite(conditional_mean).all() or not np.isfinite(conditional_covariance).all():
        raise ValueError('Gaussian conditioning overflow')
    remaining=np.linalg.eigvalsh(conditional_covariance)
    status='CONDITIONED' if remaining[-1]>0 and remaining[0]>remaining[-1]/maximum_condition_number else 'UNRESOLVED_COVARIANCE'
    return conditional_mean,conditional_covariance,{**report,'status':status,
        'observed_block_condition_number':float(eigen[-1]/eigen[0]),
        'reason':'Conditional covariance is singular or ill-conditioned' if status!='CONDITIONED' else 'Explicit observed block conditioned without regularization'}


class NativeResponseRows:
    """Persistent sequential member streams; no all-observable materialization.

    Callers verify the numeric product and retain its input hashes before
    opening this reader. File immutability is checked by the publication's
    existing source bindings. This class handles only bounded array intake.
    """
    def __init__(self,product,shapes,count,*,memory_gib):
        from core.numeric import numpy_member_datasets
        if (type(count) is not int or count<=0 or not shapes or isinstance(memory_gib,(bool,np.bool_))
                or not math.isfinite(memory_gib) or memory_gib<=0):raise ValueError('Explicit response row geometry and memory required')
        self.archive=Path(product).with_name(Path(product).name+'.data')/'arrays.npz'
        self.memory_bytes=int(memory_gib*1024**3)
        self.count=count
        self.shapes=shapes
        self.streams={}
        self.generators=[]
        self.position=0
        self.zip=zipfile.ZipFile(self.archive)
        try:
            names=self.zip.namelist()
            if len(names)!=len(set(names)):
                raise ValueError('Ambiguous response archive member identities')
            class SelectedMember:
                def __init__(self,zipped,info):
                    self.zipped=zipped
                    self.info=info
                def infolist(self):
                    return [self.info]
                def open(self,info):
                    return self.zipped.open(info)
            for level,shape in shapes.items():
                name=f'response_noise_level{level}'
                info=self.zip.getinfo(name+'.npy')
                reader=numpy_member_datasets(SelectedMember(self.zip,info),memory_bytes=self.memory_bytes,require_c_order=True)
                self.generators.append(reader)
                actual,dataset=next(reader)
                if actual!=name or dataset.shape!=(count,)+tuple(shape) or dataset.dtype.kind not in 'fiu':
                    raise ValueError('Sealed response member differs from actual cache row geometry')
                if math.prod(shape)*dataset.dtype.itemsize>self.memory_bytes:
                    raise MemoryError('One native response row exceeds intake workspace')
                self.streams[level]=dataset
        except BaseException:
            self.close()
            raise

    @property
    def row_bytes(self):
        return sum(math.prod(self.shapes[l])*d.dtype.itemsize for l,d in self.streams.items())

    def row(self,index):
        if index!=self.position or index>=self.count:
            raise ValueError('Native response rows must be consumed once in declared order')
        result={level:dataset[(index,)+(slice(None),)*3] for level,dataset in self.streams.items()}
        self.position+=1
        return result

    def small_array(self,name):
        info=self.zip.getinfo(name+'.npy')
        if info.file_size>min(self.memory_bytes,64*1024**2):
            raise MemoryError('Requested response metadata are not a small array')
        with np.load(self.archive,allow_pickle=False) as source:
            return source[name]

    def close(self):
        for reader in self.generators:
            reader.close()
        self.generators=[]
        if self.zip is not None:
            self.zip.close()
            self.zip=None

    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.close()


def validate_window_covariance_request(request):
    if (not isinstance(request,dict) or set(request)-{'seeded_covariance'}!={'schema','noise_model_verification','parent_conditioning'}
            or request['schema']!='native-compensated-window-covariance-request-v1'
            or request['parent_conditioning'] not in ('fixed','ensemble','fixed_prefix')):
        raise ValueError('Explicit verified-noise covariance request and parent conditioning required')
    if request['parent_conditioning']=='fixed_prefix':
        from core.music.seeded import validate_seeded_covariance
        validate_seeded_covariance(request.get('seeded_covariance'))
    elif 'seeded_covariance' in request:
        raise ValueError('Seeded covariance allocation requires fixed-prefix conditioning')
    binding=request['noise_model_verification']
    if (not isinstance(binding,dict) or set(binding)!={'path','sha256'} or not isinstance(binding['path'],str) or not binding['path']
            or not isinstance(binding['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',binding['sha256'])):
        raise ValueError('Checksum-bound native noise-model verification required')


def window_covariance_input_bindings(product,request):
    validate_window_covariance_request(request)
    return _covariance_input_bindings(product,request,'native-compensated-window-response-v1')


def _covariance_input_bindings(product,request,response_schema):
    from core.foundation import file_sha256
    from core.data import verify_numeric_audit,numeric_audit_source_bindings
    product=Path(product).resolve()
    binding=request['noise_model_verification']
    model=Path(binding['path']).resolve()
    if file_sha256(model)!=binding['sha256']:
        raise ValueError('Native covariance model verification checksum changed')
    sources={}
    for path in (product,model):
        report=verify_numeric_audit(path)
        expected=(response_schema,) if path==product else ('native-frozen-noise-generation-model-control-v1','native-local-seeded-noise-model-v1')
        if report.get('schema') not in expected or report.get('status')!='PASS':
            raise ValueError('Successful original response and verified supported native noise model required')
        for source,checksum in numeric_audit_source_bindings(report).items():
            if source in sources and sources[source]!=checksum:
                raise ValueError('Contradictory covariance input provenance')
            sources[source]=checksum
        for asset in (path,path.with_name(path.name+'.sha256'),path.with_name(path.name+'.data')/'arrays.npz'):
            sources[str(asset)]=file_sha256(asset)
    return sources


def _full_noise_field(path,level,memory_gib):
    path=Path(path)
    side=2**level
    _budget(memory_gib,64*1024**2+32*side**3)
    with path.open('rb') as stream:
        raw=stream.read(4)
        if len(raw)!=4 or path.stat().st_size!=4+8*side**3:
            raise ValueError('Verified full noise field size differs from declared level')
        endian='<' if int.from_bytes(raw,'little')==side else '>' if int.from_bytes(raw,'big')==side else None
        if endian is None:
            raise ValueError('Verified full noise field dimensions differ')
        result=np.fromfile(stream,dtype=endian+'f8',count=side**3).reshape((side,)*3)
    if not np.isfinite(result).all():
        raise ValueError('Finite verified full noise field required')
    return result


def publish_compensated_window_covariance(product,request,destination,*,memory_gib=1.):
    """Publish joint/conditioned finite moments; never authorize full evolution.

    This consumes an exact response publication and a checksum-bound verified
    native full-frozen model. Cache values must match that model's actual
    full fields bitwise. Selected-mass rows are conditioned explicitly.
    """
    from core.foundation import file_sha256
    from core.data import verify_numeric_audit,export_numeric_audit
    pins=window_covariance_input_bindings(product,request)
    product=Path(product).resolve()
    report=verify_numeric_audit(product)
    model=verify_numeric_audit(request['noise_model_verification']['path'])
    count=len(report['channels'])
    mean,covariance,cache_binding,metadata=verified_window_noise_moments(product,report,model,request['parent_conditioning'],pins,memory_gib=memory_gib,
        seeded_covariance=request.get('seeded_covariance'),workspace=Path(destination).with_name(Path(destination).name+'.innovations'))
    observed=np.full(count,np.nan)
    mass_indices=[]
    for check in report['checks']:
        measured=np.asarray(check['measured'],dtype=float)
        if measured.shape!=(2,) or not np.isfinite(measured).all():
            raise ValueError('Finite original measured window components required')
        for index in check['response_rows']:
            if type(index) is not int or not 0<=index<count or not np.isnan(observed[index]):
                raise ValueError('Exact distinct original window response row identities required')
            channel=report['channels'][index]
            if any(channel[name]!=check[name] for name in ('particle_type','window_id','kind','mode')):
                raise ValueError('Window response row has incompatible observable ancestry')
            observed[index]=measured[0 if channel['channel']=='real' else 1]
            if channel['kind']=='mass_normalization':
                mass_indices.append(index)
    if not np.isfinite(observed).all():
        raise ValueError('Original window response rows omit measured observables')
    mass_indices=sorted(mass_indices)
    if mass_indices:
        conditional_mean,conditional_covariance,conditioning=condition_gaussian_observables(mean,covariance,mass_indices,observed[mass_indices],memory_gib=memory_gib)
        keep=conditioning['retained_indices']
        status='MOMENTS_MEASURED' if conditioning['status']=='CONDITIONED' else 'UNRESOLVED_COVARIANCE'
    else:
        conditional_mean,conditional_covariance=mean,covariance
        keep=list(range(count))
        eigen=np.linalg.eigvalsh(covariance)
        status='MOMENTS_MEASURED' if eigen[-1]>0 and eigen[0]>eigen[-1]/1e12 else 'UNRESOLVED_COVARIANCE'
        conditioning={'status':'NOT_REQUIRED','reason':'All requested normalizations are deterministic full-fine totals','retained_indices':keep}
    values={'joint_mean':mean,'joint_covariance':covariance,'observed_real_channels':observed,
        'observed_mass_indices':np.asarray(mass_indices,dtype=np.int64),'retained_mode_indices':np.asarray(keep,dtype=np.int64)}
    if conditional_mean is not None:
        values.update(conditioned_mode_mean=conditional_mean,conditioned_mode_covariance=conditional_covariance,
            centered_mode_channels=observed[keep]-conditional_mean,conditioned_covariance_eigenvalues=np.linalg.eigvalsh(conditional_covariance))
    specifications={name:{'units':'dimensionless response row index' if name.endswith('indices') else
        'dimensionless mode squared' if 'covariance' in name else 'dimensionless mode or mass contrast','quantity':name} for name in values}
    for module in (
        'music/covariance.py',
        'music/replay.py',
    ):
        path=(_CORE_ROOT / module).resolve()
        pins[str(path)]=file_sha256(path)
    result={'schema':'native-compensated-window-covariance-v1','status':status,'source_response':{'path':str(product),'sha256':file_sha256(product)},
        'verified_noise_model':request['noise_model_verification'],'parent_conditioning':request['parent_conditioning'],
        'cache_binding':cache_binding,'channels':report['channels'],'normalization_conditioning':conditioning,'noise_response':metadata,
        'maximum_covariance_condition_number':1e12,'scope':'Finite original compensated-mass window covariance from the explicitly verified full-frozen or fixed-prefix local-seeded model and observed aperture-mass conditioning. No Gaussian test, full density/velocity, boundary or evolution acceptance.',
        'launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
    epoch=json.loads(next(iter(report['numeric_data']['array_contract']['datasets'].values()))['epoch_json'])
    return export_numeric_audit(destination,result,values,specifications,algorithm=__file__,context={'epoch':epoch},source_hashes=pins)


def verified_window_noise_moments(product,report,model,parent_conditioning,pins,*,memory_gib,
        seeded_covariance=None,workspace=None,response_arrays=None):
    from core.data import numeric_audit_source_bindings
    from core.stream_common import read_product
    from core.music.replay import load_native_noise
    if model.get('schema')=='native-local-seeded-noise-model-v1':
        from core.music.seeded import verified_seeded_window_moments
        if workspace is None:
            raise ValueError('Owned local innovation workspace required')
        return verified_seeded_window_moments(product,report,model,parent_conditioning,pins,memory_gib=memory_gib,
            workspace=workspace,settings=seeded_covariance,response_arrays=response_arrays)
    if parent_conditioning not in ('fixed','ensemble') or seeded_covariance is not None:
        raise ValueError('Full-frozen model requires fixed-base or ensemble conditioning')
    model_sources=numeric_audit_source_bindings(model)
    response_sources=numeric_audit_source_bindings(report)
    mesh=report['mesh']
    levels=[row['level'] for row in mesh['hierarchies']['convolution']]
    count=len(report['channels'])
    if [row['level'] for row in model['checks']][:len(levels)]!=levels or not count:
        raise ValueError('Verified full-frozen model does not cover the exact actual response prefix')
    streams=[]
    for row in model['checks']:
        interval=row.get('seed_interval',[])
        if (row.get('status')!='PASS' or len(interval)!=2 or any(type(v) is not int for v in interval)
                or not 0<interval[0]<=interval[1]<2**32
                or any(interval[0]<=b and a<=interval[1] for a,b in streams)):
            raise ValueError('Verified disjoint native Gaussian innovation streams required')
        streams.append(interval)
    total=sum((2**level)**3 for level in levels)
    _budget(memory_gib,64*1024**2+(32*count+640)*total)
    if response_arrays is None:
        _,inputs,_=read_product(product,memory_gib=memory_gib)
    else:inputs=response_arrays
    responses={level:inputs[f'response_noise_level{level}'] for level in levels}
    shift=np.asarray(report['output_shift_box_fraction'],dtype=float)
    parent=None
    cache_binding=[]
    for row in mesh['hierarchies']['convolution']:
        level=row['level']
        side=2**level
        fields=[Path(path) for path in model_sources if Path(path).name==f'level{level}.bin']
        caches=[Path(path) for path in response_sources if Path(path).name==f'wnoise_{level:04d}.bin']
        if len(fields)!=1 or len(caches)!=1:
            raise ValueError('Unique verified full-field and cached-noise identity required at every level')
        full=_full_noise_field(fields[0],level,memory_gib)
        pad=0 if level==levels[0] else mesh['convolution_margin']
        shape=tuple(n+2*pad for n in row['shape'])
        offset=shift*side
        rounded=np.rint(offset)
        if shift.shape!=(3,) or not np.isfinite(shift).all() or np.any(np.abs(offset-rounded)>1e-9):
            raise ValueError('Verified original-box cache shift must align with actual cells')
        origin=np.asarray(row['offset'])-pad+rounded.astype(np.int64)
        expected=full[np.ix_(*[(int(o)+np.arange(n))%side for o,n in zip(origin,shape)])]
        # Read the cache using its own validated geometry and native byte order.
        with caches[0].open('rb') as stream:
            raw=stream.read(12)
        endian='<' if tuple(np.frombuffer(raw,dtype='<u4'))==shape else '>'
        cached=load_native_noise(caches[0],shape,byte_order=endian,memory_gib=memory_gib)
        if not np.array_equal(expected,cached):
            raise ValueError('Response cache is not the verified frozen field at its declared original coordinates')
        cache_binding.append({'level':level,'original_full_field':{'path':str(fields[0]),'sha256':pins[str(fields[0])]},
            'native_cache':{'path':str(caches[0]),'sha256':pins[str(caches[0])]},'origin_original_cells':origin.tolist(),'status':'BITWISE_EQUAL'})
        if level==levels[0]:
            parent=full
    del full,expected,cached
    innovation,metadata=frozen_hierarchy_innovation_response(responses,mesh,shift,memory_gib=memory_gib)
    mean,covariance=frozen_response_moments(innovation,fixed_parent=parent if parent_conditioning=='fixed' else None,memory_gib=memory_gib)
    return mean,covariance,cache_binding,metadata


def validate_flow_covariance_request(request):
    if not isinstance(request,dict) or request.get('schema')!='native-flow-window-covariance-request-v1':
        raise ValueError('Explicit full-flow covariance request required')
    validate_window_covariance_request({**request,'schema':'native-compensated-window-covariance-request-v1'})


def flow_covariance_input_bindings(product,request):
    validate_flow_covariance_request(request)
    return _covariance_input_bindings(product,request,'native-music-full-flow-noise-response-v1')


def publish_flow_window_covariance(product,request,destination,*,memory_gib=2.):
    from core.foundation import file_sha256
    from core.data import verify_numeric_audit,export_numeric_audit
    from core.stream_common import read_product
    pins=flow_covariance_input_bindings(product,request)
    product=Path(product).resolve()
    model=verify_numeric_audit(request['noise_model_verification']['path'])
    if request['parent_conditioning']=='fixed_prefix':
        from core.music.seeded import patch_geometry
        report=verify_numeric_audit(product)
        geometry=patch_geometry(report['mesh'],report['output_shift_box_fraction'])
        with NativeResponseRows(product,{g['level']:g['shape'] for g in geometry},len(report['channels']),memory_gib=memory_gib) as rows:
            mean,covariance,cache_binding,metadata=verified_window_noise_moments(product,report,model,request['parent_conditioning'],pins,memory_gib=memory_gib,
                seeded_covariance=request['seeded_covariance'],workspace=Path(destination).with_name(Path(destination).name+'.innovations'),response_arrays=rows)
            observed=rows.small_array('observed_real_channels')
    else:
        report,inputs,_=read_product(product,memory_gib=memory_gib)
        mean,covariance,cache_binding,metadata=verified_window_noise_moments(product,report,model,request['parent_conditioning'],pins,memory_gib=memory_gib,response_arrays=inputs)
        observed=np.asarray(inputs['observed_real_channels'])
    count=len(report['channels'])
    if observed.shape!=(count,) or not np.isfinite(observed).all():
        raise ValueError('Aligned original full-flow observations required')
    units=report['row_units']
    if len(units)!=count:
        raise ValueError('Explicit per-channel full-flow units required')
    diagonal=np.diag(covariance)
    valid=np.isfinite(covariance).all() and np.all(diagonal>0)
    # Zero correlation is a defined physical value, not an unavailable marker.
    # Preserve the actual covariance, but leave correlation/eigenvalues NaN
    # when covariance is nonfinite or a variance is nonpositive.
    correlation=np.full_like(covariance,np.nan,dtype=float)
    eigen=np.full(count,np.nan)
    if valid:
        scale=np.sqrt(diagonal)
        correlation=covariance/scale[:,None]/scale[None,:]
        eigen=np.linalg.eigvalsh(correlation)
    resolved=valid and eigen[-1]>0 and eigen[0]>eigen[-1]/1e12
    values={'joint_mean':mean,'joint_covariance':covariance,'joint_correlation':correlation,
        'correlation_eigenvalues':eigen,'observed_real_channels':observed,'centered_real_channels':observed-mean}
    specifications={name:{'units':'Per-channel units in row_units; covariance element (i,j) has units[i]*units[j]' if name=='joint_covariance' else
        'dimensionless' if name in ('joint_correlation','correlation_eigenvalues') else 'Per-channel units in row_units','quantity':name} for name in values}
    for name in ('joint_correlation','correlation_eigenvalues'):
        specifications[name]['missing_policy']='NaN if covariance has nonfinite entries or any channel variance is nonpositive; inspect unresolved status'
    result={'schema':'native-full-flow-window-covariance-v1','status':'MOMENTS_MEASURED' if resolved else 'UNRESOLVED_COVARIANCE',
        'source_response':{'path':str(product),'sha256':file_sha256(product)},'verified_noise_model':request['noise_model_verification'],
        'parent_conditioning':request['parent_conditioning'],'cache_binding':cache_binding,'channels':report['channels'],'row_units':units,
        'normalization_conditioning':{'status':'NOT_REQUIRED','reason':report['mass_normalization']},'noise_response':metadata,
        'maximum_covariance_condition_number':1e12,'conditioning_operator':'Dimensionless correlation; mixed physical channel units are retained in covariance',
        'scope':'Full original linear density/velocity joint finite moments from coupled native source responses and the explicitly bound full-frozen or fixed-prefix local-seeded noise model. No Gaussian acceptance, continuum boundary or evolution acceptance.',
        'physical_boundary_accuracy':'NOT_ESTABLISHED','launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
    for module in (
        'music/covariance.py',
    ):
        p=(_CORE_ROOT / module).resolve()
        pins[str(p)]=file_sha256(p)
    epoch=json.loads(next(iter(report['numeric_data']['array_contract']['datasets'].values()))['epoch_json'])
    return export_numeric_audit(destination,result,values,specifications,algorithm=__file__,context={'epoch':epoch},source_hashes=pins)


def flow_gaussian_blocks(channels,centered,covariance,*,false_rejection_probability):
    """Reuse the existing Gaussian test on each declared observable block.

    Blocks group one window and one quantity across all requested modes.
    Bonferroni controls the familywise false rejection for correlated blocks.
    A full joint covariance may be singular because observables are redundant;
    every tested block must independently meet the unchanged 1e12 criterion.
    """
    from core.zoom import window_gaussian_check
    centered=np.asarray(centered)
    covariance=np.asarray(covariance)
    count=len(channels)
    if (not count or centered.shape!=(count,) or covariance.shape!=(count,count)
            or not np.isfinite(centered).all() or not np.isfinite(covariance).all()
            or isinstance(false_rejection_probability,(bool,np.bool_))
            or not math.isfinite(false_rejection_probability) or not 0<false_rejection_probability<1):
        raise ValueError('Aligned finite original flow channels and explicit familywise probability required')
    groups={}
    for i,channel in enumerate(channels):
        if (channel['component'] not in ('real','imag') or channel['quantity'] not in ('density','velocity')
                or not isinstance(channel['window_id'],str) or not channel['window_id']):
            raise ValueError('Explicit original flow channel ancestry required')
        groups.setdefault((channel['window_id'],channel['quantity']),{}).setdefault(tuple(channel['mode']),{})
        parts=groups[(channel['window_id'],channel['quantity'])][tuple(channel['mode'])]
        if channel['component'] in parts:
            raise ValueError('Duplicate flow Gaussian channel')
        parts[channel['component']]=i
    checks=[]
    for (window,quantity),modes in groups.items():
        if any(set(parts)!={'real','imag'} for parts in modes.values()):
            raise ValueError('Complete paired real/imaginary Gaussian channels required')
        real=[parts['real'] for parts in modes.values()]
        imag=[parts['imag'] for parts in modes.values()]
        indices=real+imag
        block=covariance[np.ix_(indices,indices)]
        variance=np.diag(block)
        if np.any(variance<=0):
            result={'status':'UNRESOLVED_COVARIANCE','reason':'Nonpositive declared block variance','science_acceptance':'NOT_ACCEPTED'}
        else:
            sigma=np.sqrt(variance)
            correlation=block/sigma[:,None]/sigma[None,:]
            data=centered[indices]/sigma
            half=len(real)
            result=window_gaussian_check(data[:half]+1j*data[half:],correlation,
                false_rejection_probability=false_rejection_probability/len(groups))
        checks.append({'window_id':window,'quantity':quantity,'source_channel_indices':indices,'modes':[list(k) for k in modes],**result})
    status='UNRESOLVED_COVARIANCE' if any(c['status']=='UNRESOLVED_COVARIANCE' for c in checks) else 'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL'
    return {'schema':'native-full-flow-familywise-gaussian-check-v1','status':status,'checks':checks,
        'false_rejection_probability':false_rejection_probability,'allocation':'Bonferroni across declared window/quantity blocks; all requested modes within each block',
        'maximum_covariance_condition_number':1e12,'unit_scaling':'Each real channel divided by its declared model standard deviation before applying the existing test',
        'scope':'Gaussian consistency with the verified native discrete source/noise model; numerical references and physical convergence remain separate',
        'science_acceptance':'NOT_ACCEPTED'}
