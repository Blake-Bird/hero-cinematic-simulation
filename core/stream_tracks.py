"""Principal stream curves, transported frames and local observables.

No global RMS width substitutes for local transverse covariance. Crossing
tracks require phase-space or ancestry separation; unresolved forks are flagged.
Spline residuals are weighted by centroid uncertainty; the smoothing budget
is dimensionless. Geometry and local widths use the same fitted curve.
"""

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra, connected_components
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter1d, label
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid
from core.stream_common import finite, positive, integer, phase_space, weighted_covariance, normalize, seeded_rng, increasing


@dataclass
class PrincipalCurve:
    splines: tuple
    parameter: np.ndarray
    arclength: np.ndarray
    grid: np.ndarray
    position: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    binormal: np.ndarray
    support: np.ndarray
    fit_variance_kpc2: np.ndarray
    diagnostics: dict

    def evaluate(self, parameter, derivative=0):
        return np.column_stack([s(parameter, nu=derivative) for s in self.splines])


def separate_by_ancestry_and_phase_space(sample, *, progenitor_label=None, unwrapped_phase=None,
                                        phase_origin=0., position_link_kpc, velocity_link_kms, max_edges):
    x,v,m,ids=phase_space(sample); n=len(x)
    position_scale=positive(position_link_kpc,'position_link_kpc')
    velocity_scale=positive(velocity_link_kms,'velocity_link_kms')
    edge_budget=integer(max_edges,'max_edges')
    ancestry=np.full(n,-1,np.int64) if progenitor_label is None else np.asarray(progenitor_label)
    if ancestry.shape!=(n,) or ancestry.dtype.kind not in 'iu' or np.any(ancestry>np.iinfo(np.int64).max):
        raise ValueError('Ancestry labels must be aligned exact signed-range integers')
    ancestry=ancestry.astype(np.int64,copy=False)
    turns=np.zeros(n,np.int64)
    if unwrapped_phase is not None:
        phase=finite(unwrapped_phase,(n,));origin=float(finite(phase_origin,(),'phase origin'))
        turn_values=np.floor((phase-origin)/(2*np.pi))
        if np.any((turn_values < -2.**63)|(turn_values >= 2.**63)):
            raise ValueError('Orbital turn labels exceed the supported integer range')
        turns=turn_values.astype(np.int64)
    labels=np.full(n,-1,np.int64); groups=[]
    for key in sorted(set(zip(ancestry.tolist(),turns.tolist()))):
        rows=np.flatnonzero((ancestry==key[0])&(turns==key[1])); y=np.column_stack((x[rows]/position_scale,v[rows]/velocity_scale))
        tree=cKDTree(y)
        if int(tree.query_ball_point(y,1,return_length=True).sum())>edge_budget:raise MemoryError('Wrap separation graph exceeds edge budget')
        pairs=tree.query_pairs(1,output_type='ndarray')
        graph=coo_matrix((np.ones(len(pairs)*2),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(rows),len(rows))).tocsr()
        count, component=connected_components(graph,directed=False)
        for c in range(count):
            selected=rows[component==c];label=len(groups);labels[selected]=label
            groups.append({'wrap_id':label,'rows':selected,'progenitor_label':key[0],
                'orbital_turn':key[1] if unwrapped_phase is not None else None,
                'wrap_identity_status':'PHASE_AND_ANCESTRY' if unwrapped_phase is not None else 'COHERENT_COMPONENT_ONLY'})
    return groups,labels


def transport_frame(tangent):
    """Minimal-rotation parallel transport; defined through straight segments."""
    tangent=finite(tangent)
    if tangent.ndim!=2 or tangent.shape[1]!=3 or not len(tangent):
        raise ValueError('Frame transport requires a nonempty N by 3 tangent array')
    t,valid=normalize(tangent)
    if not valid.all():raise ValueError('Cannot transport a frame through zero tangent')
    n=np.zeros_like(t);b=np.zeros_like(t)
    axis=np.eye(3)[np.argmin(np.abs(t[0]))]
    n[0],_=normalize(axis-np.dot(axis,t[0])*t[0]);b[0]=np.cross(t[0],n[0])
    reversals=np.zeros(len(t),bool)
    for i in range(1,len(t)):
        cross=np.cross(t[i-1],t[i]); sine=np.linalg.norm(cross);cosine=np.clip(np.dot(t[i-1],t[i]),-1,1)
        previous=n[i-1]
        if cosine < -1+1e-8:
            reversals[i]=True;previous=-previous
        elif sine>1e-12:
            axis=cross/sine
            previous=previous*cosine+np.cross(axis,previous)*sine+axis*np.dot(axis,previous)*(1-cosine)
        n[i],_=normalize(previous-np.dot(previous,t[i])*t[i]);b[i]=np.cross(t[i],n[i])
    return t,n,b,reversals


def _backbone_parameter(x,v,controls):
    n=len(x); k=min(integer(controls['ordering_neighbors'],'ordering_neighbors',2),n-1)
    y=np.column_stack((x/positive(controls['ordering_position_kpc'],'ordering_position_kpc'),
                       v/positive(controls['ordering_velocity_kms'],'ordering_velocity_kms')))
    distance,index=cKDTree(y).query(y,k=k+1)
    row=np.repeat(np.arange(n),k); col=index[:,1:].ravel(); weight=distance[:,1:].ravel()
    max_distance=positive(controls['maximum_ordering_link'],'maximum_ordering_link')
    keep=weight<=max_distance
    graph=coo_matrix((np.maximum(weight[keep],1e-12),(row[keep],col[keep])),shape=(n,n)).tocsr()
    graph=graph.maximum(graph.T)
    count,_=connected_components(graph,directed=False)
    if count!=1:raise ValueError('Track input contains disconnected wraps; separate them before fitting')
    tree=minimum_spanning_tree(graph);tree=tree+tree.T
    a=int(np.argmax(dijkstra(tree,directed=False,indices=0)))
    dist,pred=dijkstra(tree,directed=False,indices=a,return_predecessors=True)
    endpoint=int(np.argmax(dist)); path=[endpoint]
    while path[-1]!=a:path.append(int(pred[path[-1]]))
    path=np.asarray(path[::-1]); spine=x[path]; increments=np.linalg.norm(np.diff(spine,axis=0),axis=1)
    parameter=np.r_[0,np.cumsum(increments)]
    if parameter[-1]<=0:raise ValueError('No resolved track extent')
    nearest=cKDTree(spine).query(x)[1]
    branch_distance=np.linalg.norm(x-spine[nearest],axis=1)
    degree=np.diff(tree.indptr)
    return parameter[nearest],parameter[-1],{'backbone_rows':path.tolist(),'branch_nodes':np.flatnonzero(degree>2).tolist(),
        'branch_distance_kpc':branch_distance,'ordering_method':'six-dimensional nearest-neighbor MST diameter'}


def _fit_given_parameter(x,m,s,length,controls):
    knots=integer(controls['fit_bins'],'fit_bins',5)
    edges=np.linspace(0,length,knots+1); which=np.clip(np.searchsorted(edges,s,side='right')-1,0,knots-1)
    uncertainty_floor=positive(controls['position_uncertainty_floor_kpc'],'position_uncertainty_floor_kpc')
    centers=[];positions=[];weights=[];variance=[]
    for b in range(knots):
        use=which==b
        if use.sum()<2:continue
        mean,cov,neff=weighted_covariance(x[use],m[use])
        centers.append(np.average(s[use],weights=m[use]/m[use].max()));positions.append(mean)
        var=max(np.trace(cov)/max(neff,1),uncertainty_floor**2)
        variance.append(var);weights.append(1/np.sqrt(var))
    if len(centers)<5:raise ValueError('Insufficient occupied track bins for stable cubic derivatives')
    centers=np.asarray(centers);positions=np.asarray(positions);weights=np.asarray(weights)
    unique=np.r_[True,np.diff(centers)>1e-12]
    centers=centers[unique];positions=positions[unique];weights=weights[unique]
    if len(centers)<5:raise ValueError('Insufficient distinct centroids for cubic track derivatives')
    # w=1/sigma [1/kpc], so sum(w*(x-spline))^2 and s are dimensionless.
    smoothing=positive(controls['smoothing_budget'],'smoothing_budget')
    splines=tuple(UnivariateSpline(centers,positions[:,i],w=weights,k=min(3,len(centers)-1),s=smoothing,ext=2) for i in range(3))
    return splines,centers,np.asarray(variance)[unique]


def _project(x,splines,grid):
    track=np.column_stack([s(grid) for s in splines]);nearest=cKDTree(track).query(x)[1];out=np.empty(len(x))
    for i,index in enumerate(nearest):
        lo=grid[max(0,index-2)];hi=grid[min(len(grid)-1,index+2)]
        result=minimize_scalar(lambda z:sum((spl(z)-x[i,j])**2 for j,spl in enumerate(splines)),bounds=(lo,hi),method='bounded')
        if not result.success or not np.isfinite(result.x) or not np.isfinite(result.fun):
            raise ValueError('Closest-track projection did not converge')
        out[i]=result.x
    return out


def fit_uncertainty_aware_principal_curve(sample,controls,*,initial_parameter=None):
    x,v,m,ids=phase_space(sample)
    if len(x)<integer(controls['minimum_track_particles'],'minimum_track_particles',10):raise ValueError('Track underresolved')
    if initial_parameter is None:s,length,diagnostics=_backbone_parameter(x,v,controls)
    else:
        s=finite(initial_parameter,(len(x),));s=s-s.min();length=s.max()
        if length<=0:raise ValueError('No resolved initial track extent')
        diagnostics={'ordering_method':'recorded ancestry/phase parameter'}
    iterations=integer(controls['iterations'],'iterations')
    history=[];boundary_anchors=[]
    for iteration in range(iterations):
        splines,knots,var=_fit_given_parameter(x,m,s,length,controls)
        grid=np.linspace(knots[0],knots[-1],integer(controls['grid_points'],'grid_points',32))
        new=_project(x,splines,grid)
        # The fitted spline interpolates occupied-bin centroids, not the
        # entire original backbone extent. Projecting outer observations onto
        # its clipped endpoints and feeding those labels back into the next
        # fit progressively collapses the parameter domain. Retain the prior
        # ordering labels for those interpolation-boundary anchors. They still
        # remain excluded by the existing derivative/projection support masks.
        boundary=(s<knots[0])|(s>knots[-1])
        new[boundary]=s[boundary];boundary_anchors.append(int(boundary.sum()))
        shift=np.max(abs(new-s));history.append(float(shift));s=new;length=max(length,knots[-1])
        if shift<positive(controls['projection_tolerance_kpc'],'projection_tolerance_kpc'):break
    position=np.column_stack([sp(grid) for sp in splines]);first=np.column_stack([sp(grid,nu=1) for sp in splines])
    tangent,valid=normalize(first)
    if not valid.all():raise ValueError('Spline has unresolved zero-speed segments; split the track')
    tangent,normal,binormal,reversals=transport_frame(tangent)
    # Actual arclength, not the principal-curve fitting parameter.
    arclength=np.r_[0,cumulative_trapezoid(np.linalg.norm(first,axis=1),grid)]
    support=(grid>=knots[1])&(grid<=knots[-2])&~reversals
    support&=history[-1]<controls['projection_tolerance_kpc']
    fit_variance=np.interp(grid,knots,var)
    diagnostics.update({'projection_iterations':history,'converged':history[-1]<controls['projection_tolerance_kpc'],
        'projection_boundary_anchor_counts':boundary_anchors,
        'projection_boundary_policy':'Retain prior ordering labels outside fitted centroid domain; endpoint derivatives remain unsupported',
        'parameter_domain':knots[[0,-1]].tolist(),'smoothing_budget':controls['smoothing_budget'],
        'endpoint_derivatives_supported':False,'reversal_rows':np.flatnonzero(reversals).tolist()})
    return PrincipalCurve(splines,s,arclength,grid,position,tangent,normal,binormal,support,fit_variance,diagnostics)


def project_to_arclength_transverse_frame(sample,curve):
    x,v,m,ids=phase_space(sample);u=_project(x,curve.splines,curve.grid)
    center=curve.evaluate(u); s=np.interp(u,curve.grid,curve.arclength)
    frame=np.stack([np.column_stack([np.interp(u,curve.grid,a[:,j]) for j in range(3)]) for a in (curve.tangent,curve.normal,curve.binormal)],axis=1)
    # Interpolation is followed by Gram-Schmidt to maintain orthonormal axes.
    frame[:,0],_=normalize(frame[:,0]);frame[:,1]-=np.sum(frame[:,0]*frame[:,1],axis=1)[:,None]*frame[:,0]
    frame[:,1],_=normalize(frame[:,1]);frame[:,2]=np.cross(frame[:,0],frame[:,1])
    residual=x-center
    return {'particle_ids':ids,'s_kpc':s,'curve_parameter':u,'transverse_kpc':np.einsum('nij,nj->ni',frame[:,1:],residual),
        'velocity_local_kms':np.einsum('nij,nj->ni',frame,v),'projection_residual_kpc':np.linalg.norm(residual,axis=1),
        'projection_supported':np.interp(u,curve.grid,curve.support.astype(float))>.999}


def choose_bins_from_resolution_and_sampling(s,*,resolution_kpc,minimum_count,maximum_bins):
    s=finite(s)
    if s.ndim!=1 or not len(s):raise ValueError('Binning requires nonempty arclength samples')
    minimum_count=integer(minimum_count,'minimum_count',3);maximum_bins=integer(maximum_bins,'maximum_bins',2)
    length=np.ptp(s); nb=min(maximum_bins,len(s)//minimum_count,int(length/positive(resolution_kpc,'resolution_kpc')))
    if nb<2:raise ValueError('Stream sampling and physical resolution cannot support two bins')
    # Uniform arclength bins are needed for density and power. Empty bins survive.
    return np.linspace(s.min(),s.max(),nb+1)


def measure_local_observables(sample,coordinates,edges,controls):
    x,v,m,ids=phase_space(sample)
    edges=increasing(edges,'arclength bin edges')
    if not np.array_equal(ids,coordinates['particle_ids']):
        raise ValueError('Local coordinates must preserve the exact particle identity order')
    s=finite(coordinates['s_kpc'],(len(ids),),'arclength')
    trans=finite(coordinates['transverse_kpc'],(len(ids),2),'transverse coordinates')
    vl=finite(coordinates['velocity_local_kms'],(len(ids),3),'local velocity')
    projection_support=np.asarray(coordinates['projection_supported'])
    if projection_support.shape!=(len(ids),) or projection_support.dtype.kind!='b':
        raise ValueError('Exact boolean particle projection support required')
    minimum_count=integer(controls['minimum_bin_particles'],'minimum_bin_particles',2)
    minimum_effective=positive(controls['minimum_effective_particles'],'minimum_effective_particles')
    membership=finite(controls.get('membership_probability',np.ones(len(s))),(len(s),));selection=finite(controls.get('selection_probability',np.ones(len(s))),(len(s),))
    if np.any((membership<0)|(membership>1)|(selection<=0)|(selection>1)):raise ValueError('Explicit probabilities must lie in their physical domains')
    weight=finite(m*membership/selection,name='selection-corrected particle weights');bins=np.searchsorted(edges,s,side='right')-1;bins[s==edges[-1]]=len(edges)-2;n=len(edges)-1
    count=np.zeros(n,np.int64);mass=np.zeros(n);mass_var=np.zeros(n);neff=np.zeros(n);width=np.full((n,2,2),np.nan);velocity=np.full((n,3),np.nan);vcov=np.full((n,3,3),np.nan);offset=np.full((n,2),np.nan)
    width_boot=np.full((n,2,2),np.nan);velocity_boot=np.full((n,3,3),np.nan);support=np.zeros(n,bool)
    rng=seeded_rng(controls['seed'],'local-observables'); repetitions=integer(controls['bootstrap_replicates'],'bootstrap_replicates',2)
    for b in range(n):
        use=bins==b;count[b]=use.sum();mass[b]=weight[use].sum()
        # Poisson weighted sampling plus Bernoulli membership uncertainty.
        # Both terms have Msun^2; division by bin width gives density errors.
        mass_var[b]=np.sum(weight[use]**2)+np.sum((m[use]/selection[use])**2*membership[use]*(1-membership[use]))
        if count[b]<minimum_count or weight[use].sum()<=0:continue
        offset[b],width[b],neff[b]=weighted_covariance(trans[use],weight[use]);velocity[b],vcov[b],_=weighted_covariance(vl[use],weight[use])
        support[b]=projection_support[use].all() and neff[b]>=minimum_effective
        boot_w=[];boot_v=[];index=np.flatnonzero(use)
        for _ in range(repetitions):
            rows=rng.choice(index,len(index),replace=True)
            boot_w.append(weighted_covariance(trans[rows],weight[rows])[1]);boot_v.append(weighted_covariance(vl[rows],weight[rows])[1])
        width_boot[b]=np.std(boot_w,axis=0,ddof=1);velocity_boot[b]=np.std(boot_v,axis=0,ddof=1)
    ds=np.diff(edges);density=mass/ds;error=np.sqrt(mass_var)/ds
    baseline=gaussian_filter1d(density,positive(controls['density_baseline_sigma_bins'],'density_baseline_sigma_bins'),mode='nearest')
    residual=density-baseline; significance=np.divide(residual,error,out=np.full(n,np.nan),where=error>0)
    gaps,_=find_peaks(np.where(support,-np.nan_to_num(significance),0),height=controls['feature_significance'])
    caustics,_=find_peaks(np.where(support,np.nan_to_num(significance),0),height=controls['feature_significance'])
    spur=support&(np.linalg.norm(offset,axis=1)>controls['spur_offset_kpc'])
    # A fork is a proposal from transverse two-mode evidence, never a definitive label.
    forks=[]
    for b in np.flatnonzero(support):
        rows=np.flatnonzero(bins==b);y=trans[rows];axis=np.linalg.eigh(width[b])[1][:,-1];values=y@axis
        centers=np.quantile(values,[.25,.75])
        for _ in range(12):
            assign=np.abs(values[:,None]-centers).argmin(axis=1)
            if any(weight[rows][assign==j].sum()<=0 for j in range(2)):break
            centers=np.array([np.average(values[assign==j],weights=weight[rows][assign==j]) for j in range(2)])
        if any(weight[rows][assign==j].sum()<=0 for j in range(2)):continue
        within=sum(np.sum(weight[rows][assign==j]*(values[assign==j]-centers[j])**2) for j in range(2))/weight[rows].sum()
        if abs(np.diff(centers)[0])>controls['fork_separation_sigma']*np.sqrt(max(within,1e-24)) and min(np.bincount(assign,minlength=2))>=controls['minimum_bin_particles']:
            forks.append({'bin':int(b),'centers_kpc':centers.tolist(),'status':'BIFURCATION_CANDIDATE'})
    smooth=np.divide(residual,baseline,out=np.zeros(n),where=baseline>0);window=np.hanning(n)*support
    center=.5*(edges[:-1]+edges[1:]);wavenumber=2*np.pi*np.fft.rfftfreq(n,ds.mean())
    # Bin-width quadrature handles unequal bins; the measured support window is exported.
    norm=np.dot(window*window,ds);kernel=np.exp(-1j*wavenumber[:,None]*center[None,:])
    power=np.full(len(wavenumber),np.nan);shot=np.nan;spectral_window=np.full_like(power,np.nan)
    if norm>0:
        power=np.abs(kernel@(smooth*window*ds))**2/norm
        variance=np.divide(error**2,baseline**2,out=np.zeros(n),where=baseline>0)
        shot=np.sum(window**2*ds**2*variance)/norm
        spectral_window=np.abs(kernel@(window*ds))**2/norm
    progenitor_s=controls.get('progenitor_s_kpc');asymmetry=None
    if progenitor_s is not None:
        lead=weight[s>progenitor_s].sum();trail=weight[s<progenitor_s].sum();asymmetry=(lead-trail)/(lead+trail) if lead+trail>0 else None
    report={'gaps':[{'bin':int(b),'significance':float(-significance[b]),'status':'CANDIDATE_REQUIRES_FORWARD_CONTROL'} for b in gaps],
        'caustics':[int(b) for b in caustics],'forks':forks,'leading_trailing_asymmetry':asymmetry,
        'uncertainty':'particle bootstrap conditional on fitted track; track bootstrap is separate',
        'coordinate_kind':controls.get('coordinate_kind','intrinsic'),'power_window':'Hann; masked bins do not establish gap acceptance'}
    arrays={'bin_edges_kpc':edges,'bin_center_kpc':(edges[:-1]+edges[1:])/2,'count':count,'effective_count':neff,
        'linear_density_msun_kpc':density,'density_sampling_error_msun_kpc':error,'density_baseline_msun_kpc':baseline,
        'transverse_mean_kpc':offset,'width_covariance_kpc2':width,'width_covariance_bootstrap_error_kpc2':width_boot,
        'velocity_mean_kms':velocity,'velocity_covariance_kms2':vcov,'velocity_covariance_bootstrap_error_kms2':velocity_boot,
        'supported':support,'spur_candidate':spur,'density_feature_significance':significance,
        'spectral_support_window':window,'spectral_window_kpc':spectral_window,'power_supported':np.full(len(power),norm>0 and support.sum()>=controls.get('minimum_power_supported_bins',n)),'wavenumber_kpc_inverse':wavenumber,'density_power_kpc':power,'shot_power_kpc':np.full_like(power,shot)}
    from core.stream.measurements import measured_track_support_products
    extra,definitions=measured_track_support_products(report,arrays,controls)
    arrays.update(extra);report.update(definitions)
    return report,arrays


def bootstrap_track(sample,controls,curve,*,replicates,seed,membership_probability=None):
    """Whole-fit uncertainty conditional on retained wrap identity; failed fits survive."""
    x,v,m,ids=phase_space(sample);rng=seeded_rng(seed,'track-bootstrap');tracks=[];failed=[]
    probability=np.ones(len(x)) if membership_probability is None else finite(membership_probability,(len(x),))
    if np.any((probability<0)|(probability>1)):raise ValueError('Track membership probabilities must lie in [0,1]')
    for i in range(integer(replicates,'replicates',2)):
        index=rng.integers(0,len(x),len(x)); weights=np.bincount(index,minlength=len(x))*rng.binomial(1,probability);use=weights>0
        sub={'position_kpc':x[use],'velocity_kms':v[use],'mass_msun':m[use]*weights[use],
             'particle_ids':ids[use],'namespace':sample['namespace']}
        try:
            fitted=fit_uncertainty_aware_principal_curve(sub,controls,initial_parameter=curve.parameter[use])
            if not fitted.diagnostics['converged']:
                raise ValueError('Bootstrap track projection did not converge')
            grid=np.linspace(fitted.grid[0],fitted.grid[-1],len(curve.grid));tracks.append(fitted.evaluate(grid))
        except (ValueError,ArithmeticError) as exc:failed.append({'replicate':i,'reason':str(exc)})
    covariance=np.full((len(curve.grid),3,3),np.nan)
    if len(tracks)>=2:
        d=np.asarray(tracks)-np.mean(tracks,axis=0);covariance=np.einsum('rni,rnj->nij',d,d)/(len(tracks)-1)
    return covariance,{'successful':len(tracks),'failures':failed,'membership_draws':membership_probability is not None,
        'correspondence':'normalized fitted parameter; wrap assignment held fixed'}


# Curvature, viewing geometry and host-relative convexity.

def stable_track_derivatives(parameter,position,*,smoothing_variance_kpc2,weights=None):
    parameter=increasing(parameter,'track parameter',5);position=finite(position)
    if position.ndim!=2 or position.shape[0]!=len(parameter) or position.shape[1] not in (2,3):raise ValueError('Projected/3D track shape mismatch')
    variance=positive(smoothing_variance_kpc2,'smoothing_variance_kpc2')
    w=np.ones(len(parameter)) if weights is None else finite(weights,(len(parameter),))
    if np.any(w<=0):raise ValueError('Derivative fit weights must be positive')
    splines=[UnivariateSpline(parameter,position[:,k],w=w,k=3,s=variance*len(parameter)) for k in range(position.shape[1])]
    return tuple(np.column_stack([s(parameter,nu=d) for s in splines]) for d in range(4))


def compute_curvature(first,second,third=None,*,speed_floor,curvature_floor):
    """Parameter-independent curvature of a planar or Cartesian proper-kpc track.

    With u the fitting parameter and T=x'/|x'|, curvature vector is
    (x''-(x'' dot T)T)/|x'|^2 [1/kpc]. For 3D tracks,
    torsion=((first cross second) dot third)/|first cross second|^2 [1/kpc].
    Straight or zero-speed rows have no resolved curvature normal/torsion.
    """
    first=finite(first)
    if first.ndim!=2 or first.shape[1] not in (2,3):
        raise ValueError('Track derivatives require N by 2 or N by 3 arrays')
    second=finite(second,first.shape)
    speed_floor=positive(speed_floor,'speed_floor')
    curvature_floor=positive(curvature_floor,'curvature_floor')
    speed=np.linalg.norm(first,axis=1)
    tangent=np.divide(first,speed[:,None],out=np.zeros_like(first),where=speed[:,None]>speed_floor)
    transverse=second-np.sum(second*tangent,axis=1)[:,None]*tangent
    vector=np.divide(transverse,speed[:,None]**2,out=np.full_like(transverse,np.nan),where=speed[:,None]>speed_floor)
    curvature=np.linalg.norm(vector,axis=1);valid=(speed>speed_floor)&np.isfinite(curvature)&(curvature>curvature_floor)
    normal=np.divide(vector,curvature[:,None],out=np.full_like(vector,np.nan),where=valid[:,None])
    result={'curvature_kpc_inverse':curvature,'curvature_vector_kpc_inverse':vector,
        'normal':normal,'tangent':tangent,'normal_supported':valid,'parameter_speed':speed}
    if first.shape[1]==2:
        determinant=first[:,0]*second[:,1]-first[:,1]*second[:,0]
        result['signed_curvature_kpc_inverse']=np.divide(determinant,speed**3,out=np.full_like(speed,np.nan),where=speed>speed_floor)
    elif third is not None:
        third=finite(third,first.shape);cross=np.cross(first,second);den=np.sum(cross*cross,axis=1)
        result['torsion_kpc_inverse']=np.divide(np.sum(cross*third,axis=1),den,out=np.full_like(speed,np.nan),where=valid&(den>0))
    return result


def viewing_basis(direction,up=None):
    direction=finite(direction,(3,));los,valid=normalize(direction)
    if not valid:raise ValueError('Viewing direction cannot be zero')
    up=np.eye(3)[np.argmin(abs(los))] if up is None else finite(up,(3,))
    horizontal,hvalid=normalize(np.cross(up,los))
    if not hvalid:raise ValueError('Viewing up vector parallel to line of sight')
    vertical=np.cross(los,horizontal)
    return np.stack([horizontal,vertical,los])


def fibonacci_viewing_ensemble(count):
    count=integer(count,'viewing_count',2);index=np.arange(count);z=1-2*(index+.5)/count
    angle=index*np.pi*(3-np.sqrt(5));r=np.sqrt(1-z*z)
    return np.column_stack((r*np.cos(angle),r*np.sin(angle),z))


def host_relative_diagnostic(projected,geometry,center,*,width_kpc,width_gradient,support,
                             width_margin,maximum_curvature_width):
    """A=-sign(n dot (x-center)); outward normals have A=-1 (convex).

    Exclude tangent passages where d_perp <= margin*R*|dW/ds| and thick
    tracks with kappa*W above the declared limit. W is local RMS width.
    See Starkman et al. 2609.40057, equations 3 and 13-15.
    """
    projected=finite(projected)
    if projected.ndim!=2 or projected.shape[1]!=2:
        raise ValueError('Convexity requires N by 2 projected positions')
    n=len(projected);support=np.asarray(support)
    if support.shape!=(n,) or support.dtype.kind!='b':
        raise ValueError('Exact boolean convexity support required')
    width_kpc=finite(width_kpc,(n,),'projected RMS width')
    width_gradient=np.asarray(width_gradient,float)
    if width_gradient.shape!=(n,) or np.any(width_kpc<0):
        raise ValueError('Aligned nonnegative widths and width gradients required')
    width_margin=positive(width_margin,'width_margin')
    maximum_curvature_width=positive(maximum_curvature_width,'maximum_curvature_width')
    offset=projected-finite(center,(2,));normal=geometry['normal'];dot=np.sum(normal*offset,axis=1)
    radius=np.linalg.norm(offset,axis=1);tangent=geometry['tangent']
    perpendicular=offset-np.sum(offset*tangent,axis=1)[:,None]*tangent
    distance=np.linalg.norm(perpendicular,axis=1)
    near_tangent=distance <= width_margin*radius*np.abs(width_gradient)
    thick=geometry['curvature_kpc_inverse']*width_kpc>=maximum_curvature_width
    valid=support&geometry['normal_supported']&np.isfinite(width_gradient)&np.isfinite(dot)&~near_tangent&~thick
    alignment=np.full(len(dot),np.nan);alignment[valid]=-np.sign(dot[valid])
    return {'alignment':alignment,'convex':valid&(dot>0),'diagnostic_supported':valid,
        'center_normal_distance_kpc':dot,'near_tangent':near_tangent,'finite_width_violation':thick,
        'width_kpc':width_kpc,'width_gradient':width_gradient}


def _segments(mask,s,minimum_length):
    components,count=label(mask);out=[]
    for k in range(1,count+1):
        rows=np.flatnonzero(components==k);length=s[rows[-1]]-s[rows[0]]
        if length>=minimum_length:out.append({'first':int(rows[0]),'last':int(rows[-1]),'length_kpc':float(length)})
    return out


def _covariance_tensor(value, shape, name):
    """Require finite symmetric positive-semidefinite physical covariance."""
    covariance=finite(value,shape,name)
    if not np.allclose(covariance,np.swapaxes(covariance,-1,-2)) or np.any(np.linalg.eigvalsh(covariance)<-1e-12):
        raise ValueError(name+' must be symmetric and positive semidefinite')
    return covariance


def _projected_width_profile(grid, first, geometry, covariance, support, controls):
    """Project local covariance on the current normal; differentiate contiguous support.

    W^2=n^T C n [kpc^2]; s=int |dx/du| du [kpc]. Unsupported gaps and
    foreshortened duplicate s values must not be bridged by dW/ds.
    """
    normal=np.nan_to_num(geometry['normal'])
    width=np.sqrt(np.maximum(np.einsum('ni,nij,nj->n',normal,covariance,normal),0.))
    arc=np.r_[0,cumulative_trapezoid(np.linalg.norm(first,axis=1),grid)]
    floor=positive(controls['arclength_floor_kpc'],'arclength_floor_kpc')
    available=support&geometry['normal_supported']&np.r_[True,np.diff(arc)>floor]
    components,count=label(available)
    gradient=np.full(len(grid),np.nan)
    for component in range(1,count+1):
        rows=np.flatnonzero(components==component)
        if len(rows)>=3:
            # A centered derivative needs a real supported neighbor on each side.
            values=np.gradient(width[rows],arc[rows],edge_order=2)
            gradient[rows[1:-1]]=values[1:-1]
    return arc,width,gradient,available&np.isfinite(gradient)


def measure_viewing_ensemble(curve,*,directions,host_center_kpc,host_center_covariance_kpc2,
                            width_covariance_kpc2,controls,selection=None,track_covariance_kpc2=None):
    directions=finite(directions);n=len(curve.grid);center=finite(host_center_kpc,(3,))
    if directions.ndim!=2 or directions.shape[1]!=3 or not len(directions):
        raise ValueError('Viewing ensemble requires nonempty N by 3 directions')
    center_cov=_covariance_tensor(host_center_covariance_kpc2,(3,3),'center uncertainty [kpc^2]')
    width_cov=_covariance_tensor(width_covariance_kpc2,(n,3,3),'spatial width [kpc^2]')
    track_cov=None if track_covariance_kpc2 is None else _covariance_tensor(
        track_covariance_kpc2,(n,3,3),'track uncertainty [kpc^2]')
    support=np.asarray(curve.support)
    selection=np.ones((len(directions),n),bool) if selection is None else np.asarray(selection)
    if support.shape!=(n,) or support.dtype.kind!='b' or selection.shape!=(len(directions),n) or selection.dtype.kind!='b':
        raise ValueError('Exact boolean track and per-view selection masks required')
    probability_floor=positive(controls['minimum_convex_probability'],'minimum_convex_probability')
    if probability_floor>1:raise ValueError('Convex probability threshold cannot exceed one')
    arrays={};views=[];seed=controls['seed'];replicates=integer(controls['uncertainty_replicates'],'uncertainty_replicates',2)
    d3=tuple(curve.evaluate(curve.grid,d) for d in range(4))
    intrinsic=compute_curvature(*d3[1:],speed_floor=controls['parameter_speed_floor'],curvature_floor=controls['curvature_floor_kpc_inverse'])
    for key,value in intrinsic.items():arrays['intrinsic_'+key]=value
    for k,direction in enumerate(directions):
        basis=viewing_basis(direction);projected=d3[0]@basis[:2].T
        derivatives=tuple(d@basis[:2].T for d in d3[1:])
        geometry=compute_curvature(derivatives[0],derivatives[1],speed_floor=controls['parameter_speed_floor'],curvature_floor=controls['curvature_floor_kpc_inverse'])
        projected_cov=np.einsum('ij,njk,lk->nil',basis[:2],width_cov,basis[:2])
        arc,width,derivative,local_support=_projected_width_profile(curve.grid,derivatives[0],
            geometry,projected_cov,support&selection[k],controls)
        diagnostic=host_relative_diagnostic(projected,geometry,center@basis[:2].T,width_kpc=width,
            width_gradient=derivative,support=local_support,width_margin=controls['width_margin'],maximum_curvature_width=controls['maximum_curvature_width'])
        rng=seeded_rng(seed,'geometry-view',k);convex=[];distances=[];curvatures=[];draw_support=[]
        for _ in range(replicates):
            draw_center=rng.multivariate_normal(center,center_cov)
            draw_geometry=geometry;draw_projected=projected
            draw_width,draw_gradient,draw_local_support=width,derivative,local_support
            if track_cov is not None:
                draw=np.array([rng.multivariate_normal(p,cov) for p,cov in zip(curve.position,track_cov)])
                fitted=stable_track_derivatives(curve.grid,draw@basis[:2].T,smoothing_variance_kpc2=controls['uncertainty_smoothing_variance_kpc2'])
                draw_projected=fitted[0];draw_geometry=compute_curvature(fitted[1],fitted[2],speed_floor=controls['parameter_speed_floor'],curvature_floor=controls['curvature_floor_kpc_inverse'])
                _,draw_width,draw_gradient,draw_local_support=_projected_width_profile(curve.grid,
                    fitted[1],draw_geometry,projected_cov,support&selection[k],controls)
            draw_diag=host_relative_diagnostic(draw_projected,draw_geometry,draw_center@basis[:2].T,
                width_kpc=draw_width,width_gradient=draw_gradient,support=draw_local_support,
                width_margin=controls['width_margin'],maximum_curvature_width=controls['maximum_curvature_width'])
            draw_support.append(draw_diag['diagnostic_supported'])
            convex.append(draw_diag['convex']);distances.append(draw_diag['center_normal_distance_kpc'])
            curvatures.append(draw_geometry['curvature_kpc_inverse'])
        # Unsupported geometry is missing evidence, not a nonconvex realization.
        draw_support=np.asarray(draw_support);available_count=draw_support.sum(axis=0)
        probability=np.divide(np.sum(convex,axis=0),available_count,
            out=np.full(n,np.nan),where=available_count>=2)
        probability_supported=diagnostic['diagnostic_supported']&(available_count>=2)
        confident=probability_supported&(probability>=probability_floor)
        distance_std=np.ma.array(distances,mask=~draw_support).std(axis=0).filled(np.nan)
        curvature_std=np.ma.array(curvatures,mask=~draw_support).std(axis=0).filled(np.nan)
        segments=_segments(confident,arc,positive(controls['minimum_convex_length_kpc'],'minimum_convex_length_kpc'))
        view={'view_id':k,'direction':direction,'basis':basis,'convex_segments':segments,'has_convex_segment':bool(segments),
            'supported_fraction':float(diagnostic['diagnostic_supported'].mean()),
            'shape_interpretation':'one-sided geometric diagnostic; retain baryonic and time-dependent alternatives'}
        views.append(view)
        for key,value in {'projected_kpc':projected,'projected_arclength_kpc':arc,**geometry,**diagnostic,
            'convex_probability':probability,'convex_probability_supported':probability_supported,
            'convex_supported_draw_count':available_count,
            'center_normal_distance_std_kpc':distance_std,'curvature_std_kpc_inverse':curvature_std}.items():arrays['view_%03d_%s'%(k,key)]=value
    return {'schema':'stream-geometry-v1','views':views,'controls':controls,
        'definition':'A=-sign(curvature normal dot host-relative projected position); A=-1 is convex',
        'primary_sources':['https://arxiv.org/html/2609.40057v1','https://arxiv.org/abs/2605.07809'],
        'center_covariance_kpc2':center_cov,'convex_probability_definition':'Convex fraction conditional on supported uncertainty draws; at least two required',
        'track_uncertainty':'refit independent point-covariance draws' if track_covariance_kpc2 is not None else 'center uncertainty only; track uncertainty unresolved',
        'viewing_convex_fraction':np.mean([v['has_convex_segment'] for v in views]),
        'views_are_independent_halos':False},arrays


def match_shape_profile(geometry_report,radial_profile,*,definition):
    """Attach measured density or equipotential profiles without conflating them."""
    if definition not in {'density_shell','isopotential'}:raise ValueError('Declare independent shape estimator')
    radius=increasing(radial_profile['radius_kpc'],'shape radii');b=finite(radial_profile['b_over_a'],radius.shape);c=finite(radial_profile['c_over_a'],radius.shape)
    supported=np.asarray(radial_profile['supported'])
    if supported.shape!=radius.shape or supported.dtype.kind!='b':raise ValueError('Exact boolean radial shape support required')
    if np.any(radius<=0) or np.any((c<=0)|(b<c)|(b>1)):raise ValueError('Invalid physical shape radii or axes')
    return {**geometry_report,'shape_profile_reference':{'definition':definition,'radius_kpc':radius,'b_over_a':b,'c_over_a':c,
        'source_reference':radial_profile['source_reference'],'supported':supported},
        'inference_status':'FORWARD_COMPARISON_INPUT; no unique shape or SIDM cross section inferred'}
