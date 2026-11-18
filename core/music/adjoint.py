"""Adjoints from MUSIC windows through potentials, sources and shared noise.

For real arrays the transpose preserves <w, F(x)> = <F.T(w), x>. Potentials
include supported ghost cells; source weights retain their actual hierarchy.
The coupled inverse uses its explicit gauge. Gas, DM and compensation responses
must be added on the SAME noise variables before covariance is evaluated.
No transpose or solver residual establishes a physical field-error bound.
"""
import math
import numpy as np
from core.music.replay import _budget


def finite_gradient_adjoint(weights,*,level,order=6,halo=4,memory_gib=1.):
    """Scatter the real gradient response onto potential and ghost cells.

    Each forward +/- stencil sample receives the corresponding signed weight
    times 2**level, since normalized box spacing is 2**(-level). Ghost weights
    remain explicit; dropping them would change the transpose identity.
    """
    w=np.asarray(weights)
    if (w.ndim!=4 or w.shape[-1]!=3 or any(n<=0 for n in w.shape[:3]) or w.dtype.kind not in 'fiu'
            or type(level) is not int or not 1<=level<=20 or type(order) is not int or order not in (2,4,6)
            or type(halo) is not int or halo<order//2):
        raise ValueError('Aligned real vector weights and explicit finite gradient geometry required')
    shape=tuple(n+2*halo for n in w.shape[:3])
    _budget(memory_gib,64*1024**2+64*math.prod(shape)+w.nbytes)
    if not np.isfinite(w).all():
        raise ValueError('Finite gradient weights required')
    result=np.zeros(shape)
    coeff={2:[.5],4:[2/3,-1/12],6:[.75,-.15,1/60]}[order]
    for axis in range(3):
        for distance,coefficient in enumerate(coeff,1):
            for sign in (-1,1):
                delta=[0,0,0]
                delta[axis]=sign*distance
                selection=tuple(slice(halo+d,halo+d+n) for d,n in zip(delta,w.shape[:3]))
                result[selection]+=sign*coefficient*2**level*w[...,axis]
    if not np.isfinite(result).all():
        raise ValueError('Potential gradient response overflow')
    return result


def hybrid_correction_adjoint(weights,*,gradient_order=6,laplace_order=6,deconvolve_cic=False,memory_gib=1.):
    """Real transpose of periodic vector hybrid response in grid-cell units.

    Full complex transforms provide independent Hermitian bookkeeping from
    the forward half-spectrum implementation. Each differentiated Nyquist
    plane is zero; other components on that plane are retained.
    """
    w=np.asarray(weights)
    if (w.ndim!=4 or w.shape[-1]!=3 or len(set(w.shape[:3]))!=1 or w.shape[0]<4 or w.shape[0]%2
            or w.dtype.kind not in 'fiu' or type(gradient_order) is not int or gradient_order not in (2,4,6)
            or type(laplace_order) is not int or laplace_order not in (2,4,6) or type(deconvolve_cic) is not bool):
        raise ValueError('Real cubic vector weights and supported hybrid operators required')
    _budget(memory_gib,64*1024**2+320*math.prod(w.shape[:3]))
    if not np.isfinite(w).all():
        raise ValueError('Finite hybrid response weights required')
    side=w.shape[0]
    wave=2*np.pi*np.fft.fftfreq(side)
    axes=[wave[:,None,None],wave[None,:,None],wave[None,None,:]]
    squared=sum(k*k for k in axes)
    laplace=np.zeros_like(squared)
    lap_coeff={2:[1.],4:[4/3,-1/12],6:[1.5,-.15,1/90]}[laplace_order]
    for k in axes:
        for distance,coefficient in enumerate(lap_coeff,1):
            laplace+=4*coefficient*np.sin(distance*k/2)**2
    assignment=np.ones_like(squared)
    if deconvolve_cic:
        for k in axes:
            assignment*=np.sinc(k/(2*np.pi))**2
    grad_coeff={2:[.5],4:[2/3,-1/12],6:[.75,-.15,1/60]}[gradient_order]
    spectrum=np.zeros(w.shape[:3],dtype=complex)
    for axis,k in enumerate(axes):
        symbol=sum(2*coefficient*np.sin(distance*k) for distance,coefficient in enumerate(grad_coeff,1))
        response=np.divide(k,squared,out=np.zeros_like(squared),where=squared>0)-np.divide(symbol,laplace,out=np.zeros_like(squared),where=laplace>0)
        selection=[slice(None)]*3
        selection[axis]=side//2
        response[tuple(selection)]=0.
        spectrum-=1j*response*np.fft.fftn(w[...,axis])/assignment
    result=np.fft.ifftn(spectrum).real
    if not np.isfinite(result).all():
        raise ValueError('Hybrid source response overflow')
    return result


def sampled_hierarchy_source_adjoint(hierarchy,origin,weights,*,degree=3,sampling='cell_average',memory_gib=1.):
    """Transpose the bounded native hierarchy source-reconstruction model.

    Returns weights for each supplied scalar source grid. Repeated periodic
    indices accumulate, overwritten fine cells have no parent interpolation
    response, and cell-average preparation restriction is transposed as well.
    No global finest-level field is constructed.
    """
    from core.zoom import _prepare_cell_average_hierarchy
    w=np.asarray(weights)
    start=np.asarray(origin)
    if (w.ndim!=3 or w.dtype.kind not in 'fiu' or any(n<=0 for n in w.shape) or start.shape!=(3,) or start.dtype.kind not in 'iu'
            or type(degree) is not int or degree not in (0,3,5) or sampling not in ('cell_average','cell_centered_point')
            or not isinstance(hierarchy,(list,tuple)) or not hierarchy):
        raise ValueError('Real bounded source weights, integer origin and declared source sampling required')
    sizes=list(w.shape)
    workspace=0
    for _ in hierarchy:
        workspace+=math.prod(sizes)
        sizes=[(n+1)//2+degree+3 for n in sizes]
    _budget(memory_gib,64*1024**2+256*workspace+128*sum(np.asarray(row['values']).size for row in hierarchy))
    if not np.isfinite(w).all():
        raise ValueError('Finite hierarchy reconstruction weights required')
    records,levels,_=_prepare_cell_average_hierarchy(hierarchy,degree,memory_gib,8*workspace,sampling=sampling)
    result=[np.zeros(row['values'].shape) for row in records]
    side=2**levels[-1]
    coordinates=[(int(o)%side+np.arange(n,dtype=np.int64))%side for o,n in zip(start,w.shape)]
    offsets=np.array([0],dtype=int) if degree==0 else np.arange(-(degree+1)//2,(degree+1)//2+1)
    interpolation=np.ones((2,1)) if degree==0 else np.zeros((2,degree+2))
    if degree:
        powers=np.arange(1,degree+2) if sampling=='cell_average' else np.arange(degree+1)
        for branch,stencil in enumerate((offsets[:-1],offsets[1:])):
            matrix=(((stencil[:,None]+.5)**powers-(stencil[:,None]-.5)**powers)/powers if sampling=='cell_average' else stencil[:,None]**powers)
            for child in (0,1):
                left=child/2-.5
                right=(child+1)/2-.5
                moments=((right**powers-left**powers)/powers*2 if sampling=='cell_average' else ((left+right)/2)**powers)
                interpolation[child,branch:branch+degree+1]+=np.linalg.solve(matrix.T,moments)*.5
    def transpose(index,coords,weight):
        row=records[index]
        local=[(c-int(o))%2**row['level'] for c,o in zip(coords,row['origin'])]
        if index==0:
            np.add.at(result[index],np.ix_(*local),weight)
            return
        selected=[np.flatnonzero(c<n) for c,n in zip(local,row['values'].shape)]
        remaining=weight.copy()
        if all(len(s) for s in selected):
            selection=np.ix_(*selected)
            np.add.at(result[index],np.ix_(*[c[s] for c,s in zip(local,selected)]),remaining[selection])
            remaining[selection]=0.
        parent_side=2**records[index-1]['level']
        candidates=[(c[:,None]//2+offsets[None,:])%parent_side for c in coords]
        parents=[np.unique(c) for c in candidates]
        lookup=[np.searchsorted(p,c) for p,c in zip(parents,candidates)]
        for axis in (2,1,0):
            moved=np.moveaxis(remaining,axis,0)
            target=np.zeros((len(parents[axis]),)+moved.shape[1:])
            for column in range(len(offsets)):
                factor=interpolation[coords[axis]%2,column].reshape((len(coords[axis]),)+(1,)*(moved.ndim-1))
                np.add.at(target,lookup[axis][:,column],moved*factor)
            remaining=np.moveaxis(target,0,axis)
        transpose(index-1,parents,remaining)
    transpose(len(records)-1,coordinates,w)
    if sampling=='cell_average':
        for i in range(1,len(records)):
            child,parent=records[i],records[i-1]
            extent=np.asarray(child['values'].shape)//2
            begin=(child['origin']//2-parent['origin'])%2**parent['level']
            selection=np.ix_(*[(int(o)+np.arange(int(n)))%size for o,n,size in zip(begin,extent,parent['values'].shape)])
            spread=result[i-1][selection]/8.
            for axis in range(3):
                spread=np.repeat(spread,2,axis=axis)
            result[i]+=spread
            result[i-1][selection]=0.
    return {row['level']:value for row,value in zip(records,result)}, {'schema':'bounded-source-sampling-adjoint-v1',
        'degree':degree,'sampling':sampling,'origin':start.tolist(),'shape':list(w.shape),
        'global_finest_grid_allocated':False,'scope':'Transpose of the declared bounded source model only; physical source reconstruction error budget remains separate',
        'launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}


def gradient_hierarchy_adjoint(weights,source,potential,configuration,*,gas=False,memory_gib=1.):
    """Transpose captured-potential FD plus bounded hybrid gradient assembly.

    Source and potential are distinct inputs here. Returned potential weights
    explicitly retain boundary/ghost dependence; no independence of those
    potentials and the source/noise is assumed for a covariance calculation.
    """
    cfg=configuration
    low=cfg.getint('setup','levelmin')
    high=cfg.getint('setup','levelmax')
    if (type(gas) is not bool or not low<high or cfg.getboolean('poisson','kspace',fallback=False)
            or cfg.getint('setup','levelmin_TF',fallback=low)!=low
            or any([row['level'] for row in rows]!=list(range(low,high+1)) for rows in (source,potential))
            or set(weights)!=set(range(low,high+1))):
        raise ValueError('Actual nested finite-gradient hierarchy and real response weights required')
    order=cfg.getint('poisson','grad_order',fallback=6)
    laplace=cfg.getint('poisson','laplace_order',fallback=6)
    mode=cfg.get('poisson','hybrid_source_extension',fallback='none')
    if mode not in ('none','bounded_conservative'):
        raise ValueError('Supported actual native source extension required')
    boundary=cfg.getint('poisson','hybrid_extension_boundary',fallback=32) if mode=='bounded_conservative' else 32
    if boundary<=0:
        raise ValueError('Positive native hybrid boundary required')
    cells=sum(row['values'].size for row in source)
    _budget(memory_gib,64*1024**2+256*cells)
    work=[]
    for s,p in zip(source,potential):
        w=np.asarray(weights[s['level']])
        if (s['level']!=p['level'] or s['origin']!=p['origin'] or w.shape!=s['values'].shape+(3,)
                or w.dtype.kind not in 'fiu' or not np.isfinite(w).all()
                or p['values'].shape!=tuple(n+2*p['halo'] for n in s['values'].shape)):
            raise ValueError('Aligned source, potential ghosts and vector response required')
        work.append(w.astype(float,copy=True))
    # Reverse the native base-only centring, then the single finest->parent
    # gradient restriction. It is intentionally not all-level restriction.
    work[0]-=np.mean(work[0],axis=(0,1,2),dtype=np.longdouble).astype(float)
    child,parent=source[-1],source[-2]
    extent=np.asarray(child['values'].shape)//2
    if np.any(np.asarray(child['origin'])%2) or any(n%2 for n in child['values'].shape):
        raise ValueError('Aligned native finest gradient restriction required')
    begin=(np.asarray(child['origin'])//2-np.asarray(parent['origin']))%2**parent['level']
    if parent['values'].shape!=(2**parent['level'],)*3 and np.any(begin+extent>parent['values'].shape):
        raise ValueError('Gradient response child escapes actual parent')
    selection=np.ix_(*[(int(o)+np.arange(int(n)))%size for o,n,size in zip(begin,extent,parent['values'].shape)])
    spread=work[-2][selection]/8.
    for axis in range(3):
        spread=np.repeat(spread,2,axis=axis)
    work[-1]+=spread
    work[-2][selection]=0.
    potential_response={s['level']:finite_gradient_adjoint(w,level=s['level'],order=order,halo=p['halo'],memory_gib=memory_gib)
        for s,p,w in zip(source,potential,work)}
    source_response={row['level']:np.zeros(row['values'].shape) for row in source}
    extension={'status':'ZERO_FINE_RESPONSE'}
    if np.any(work[-1]):
        shape=child['values'].shape
        side=max(shape)+2*boundary
        _budget(memory_gib,64*1024**2+640*side**3+256*cells)
        padded=np.zeros((side,side,side,3))
        crop=tuple(slice(boundary,boundary+n) for n in shape)
        padded[crop]=work[-1]/2**high
        cic=cfg.getboolean('output','glass_cicdeconvolve',fallback=False) and (not gas or cfg.getboolean('setup','do_SPH',fallback=False))
        scalar=hybrid_correction_adjoint(padded,gradient_order=order,laplace_order=laplace,deconvolve_cic=cic,memory_gib=memory_gib)
        del padded
        if mode=='bounded_conservative':
            source_response,extension=sampled_hierarchy_source_adjoint(source,np.asarray(child['origin'])-boundary,scalar,
                degree=cfg.getint('poisson','hybrid_extension_degree',fallback=5),sampling='cell_average',memory_gib=memory_gib)
        else:
            source_response[high]=scalar[crop].copy()
            extension={'status':'NATIVE_ZERO_PADDING'}
    return source_response,potential_response,{'schema':'native-gradient-hierarchy-adjoint-v1',
        'gradient_order':order,'laplace_order':laplace,'source_extension':extension,
        'scope':'Actual gradient assembly transpose with separate immutable-source and captured-potential/ghost inputs',
        'potential_noise_coupling':'NOT_ESTABLISHED_BY_THIS_OPERATOR','boundary_accuracy_acceptance':'NOT_ESTABLISHED',
        'launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}




def potential_source_response(operator,potential_weights,*,tolerance=1e-10,restart=32,max_iterations=128):
    """Transpose the gauge-fixed coupled solve for a gauge-invariant window.

    If A*phi = B*source and the window is w.T*phi, solve A.T*lambda=w;
    then B.T*lambda gives the source coefficients. B includes grid-spacing
    scaling and the explicit covered/gauge rows.

    The returned source coefficients may be used only when status is PASS.
    Solver convergence is not a certificate of continuum boundary accuracy.
    No potential samples or ghost cells are independent random variables.
    """
    interior=operator.potential_adjoint(potential_weights)
    # A common potential constant is a gauge, never a stochastic source.
    scale=float(np.linalg.norm(interior))*math.sqrt(operator.size)
    gauge_response=float(np.sum(interior,dtype=np.longdouble))
    if abs(gauge_response)>1e-10*scale:
        raise ValueError('Physical potential window must be invariant to a common potential constant')
    solution,solve=operator.solve(interior,transpose=True,tolerance=tolerance,restart=restart,max_iterations=max_iterations)
    weights=[]
    for row,covered,level in zip(operator.split(solution),operator.covered,operator.levels):
        weight=row.copy()/float(2**level)**2
        weight[covered]=0.
        weights.append(weight)
    weights[0].flat[0]=0.
    record={'schema':'native-potential-source-response-v1','status':solve['status'],'solve':solve,
        'common_constant_window_response':gauge_response,'gauge_invariance_relative_error':abs(gauge_response)/scale if scale else 0.,
        'removed_physical_row_policy':'Source response is zero at the explicit gauge row; original source compatibility and all physical equations remain mandatory checks',
        'source_response':'Original physical leaf sources; covered source coefficients are zero',
        'physical_boundary_accuracy':'NOT_ESTABLISHED','science_acceptance':'NOT_ACCEPTED','launch_authorized':False}
    return weights,record


def flow_source_noise_response(source_weights,compensation_weights,mesh,configuration,transfer_table,*,quantity,memory_gib=2.):
    """Combine species preparations on their SAME cached-noise variables.

    source_weights: preparation(0/1)->level->real weights, after the coupled
    potential transpose and direct hybrid/source contributions are combined.
    compensation_weights: preparation->finest weights on the native already
    species-weighted compensation field. Their factors are applied once.
    The result contains no assumed covariance or boundary-error acceptance.
    """
    from core.music.windows import prepared_source_adjoint
    from core.music.replay import dms_transfer_amplitude
    if (quantity not in ('density','velocity') or not isinstance(source_weights,dict) or not source_weights
            or not isinstance(compensation_weights,dict)
            or any(type(k) is not int or k not in (0,1) for k in (*source_weights,*compensation_weights))
            or quantity=='velocity' and compensation_weights):
        raise ValueError('Explicit original density/velocity preparation responses required')
    levels=[row['level'] for row in mesh['hierarchies']['poisson']]
    high=levels[-1]
    fb=configuration.getfloat('cosmology','Omega_b')/configuration.getfloat('cosmology','Omega_m')
    if not 0<fb<1:
        raise ValueError('Physical native baryon fraction required')
    result={}
    preparations=[]
    def accumulate(response):
        for level,value in response.items():
            if level not in result:
                result[level]=value.copy()
            else:result[level]+=value
    for preparation,weights in source_weights.items():
        kind='density' if quantity=='density' else 'velocity_gas' if preparation==0 else 'velocity_cdm'
        response,metadata=prepared_source_adjoint(weights,mesh,configuration,
            dms_transfer_amplitude(transfer_table,configuration,kind),gas=preparation==0,memory_gib=memory_gib)
        accumulate(response)
        preparations.append({'preparation':preparation,'quantity':kind,'source_operator':metadata})
    for preparation,weight in compensation_weights.items():
        weights={row['level']:np.zeros(tuple(row['shape'])) for row in mesh['hierarchies']['poisson']}
        weight=np.asarray(weight)
        shape=weights[high].shape
        if weight.shape!=shape or weight.dtype.kind not in 'fiu' or not np.isfinite(weight).all():
            raise ValueError('Finite aligned finest compensated source response required')
        factor=1-fb if preparation==0 else -fb
        weights[high]=weight*factor
        response,metadata=prepared_source_adjoint(weights,mesh,configuration,
            dms_transfer_amplitude(transfer_table,configuration,'compensation'),gas=preparation==0,memory_gib=memory_gib)
        accumulate(response)
        preparations.append({'preparation':preparation,'quantity':'compensation',
            'native_species_factor':factor,'source_operator':metadata})
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Full flow noise response overflow')
    return result,{'schema':'native-flow-source-noise-response-v1','quantity':quantity,'levels':levels,
        'preparations':preparations,'shared_noise_policy':'Gas/DM and compensated contributions summed on the same native cache cells',
        'noise_covariance':'NOT_ASSUMED; verified correlated phase model required',
        'physical_boundary_accuracy':'NOT_ESTABLISHED','launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
