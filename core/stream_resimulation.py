"""Live self-gravitating stellar stripping and calibrated release adapters.

The reference integrator is deterministic direct softened gravity in an
inertial frame, coupled to the actual time-dependent parent field. This supports
collisionless stellar systems; collisional clusters and DM-bearing SIDM systems
need a native physics engine. Particle spray is a distinct approximation.
"""

from pathlib import Path
import importlib.metadata
import json
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial import cKDTree
from core.foundation import atomic_json, atomic_text, file_sha256 as digest
from core.stream_common import G, KMS_TO_KPC_MYR, finite, positive, integer, increasing, phase_space, normalize, weighted_covariance, seeded_rng, json_safe


def initialize_equilibrium_model(specification, *, namespace, seed):
    """Sample the exact isolated isotropic Plummer DF, not a Maxwellian guess.

    f(E) proportional to (-E)^(7/2); p(q) proportional to q^2(1-q^2)^(7/2),
    q=v/v_escape. Numerical force softening is recorded by the integrator and
    requires a softening convergence campaign before physical acceptance.
    """
    if specification['model']!='plummer':raise ValueError('Only the analytical Plummer equilibrium DF is implemented')
    if not isinstance(namespace,str) or not namespace.strip():raise ValueError('A nonblank tracer namespace is required')
    if specification.get('dark_matter_mass_msun',0)>0:raise ValueError('DM-bearing progenitors require native DM/SIDM evolution')
    mass=positive(specification['mass_msun'],'mass_msun');b=positive(specification['scale_kpc'],'scale_kpc')
    n=integer(specification['particles'],'particles',32);rng=seeded_rng(seed,'plummer-equilibrium')
    u=np.clip(rng.random(n),np.finfo(float).eps,1-np.finfo(float).eps)
    radius=b/np.sqrt(u**(-2/3)-1)
    direction,_=normalize(rng.normal(size=(n,3)));x=radius[:,None]*direction
    # q^2*(1-q^2)^(7/2) peaks at q^2=2/9; rejection sampling uses that maximum.
    q=np.empty(n);pending=np.arange(n);max_density=(2/9)*(7/9)**3.5
    while len(pending):
        proposed=rng.random(len(pending));accept=rng.random(len(pending))*max_density < proposed**2*(1-proposed**2)**3.5
        q[pending[accept]]=proposed[accept];pending=pending[~accept]
    escape=np.sqrt(2*G*mass/np.sqrt(radius**2+b*b));direction,_=normalize(rng.normal(size=(n,3)))
    v=direction*(q*escape)[:,None]
    x-=x.mean(axis=0);v-=v.mean(axis=0)
    x+=finite(specification['center_kpc'],(3,));v+=finite(specification['center_velocity_kms'],(3,))
    return {'position_kpc':x,'velocity_kms':v,'mass_msun':np.full(n,mass/n),
        'particle_ids':np.arange(1,n+1,dtype=np.uint64),'namespace':namespace,
        'initialization':{'model':'isolated isotropic Plummer DF','finite_N_center_correction':True,
            'parent_progenitor_reference':specification['parent_progenitor_reference'],
            'equilibrium_scope':'isolated unsoftened; response to actual external tide measured by evolution'}}


def initialize_from_measured_progenitor(sample, specification):
    """Preserve measured native internal state under an explicit phase-space boost."""
    x,v,m,ids=phase_space(sample)
    namespace=specification['tracer_namespace']
    if not len(x) or not isinstance(namespace,str) or not namespace.strip() or namespace==sample['namespace']:
        raise ValueError('Measured progenitor needs particles and a separate nonblank tracer namespace')
    weight=m/m.max()
    center=np.average(x,axis=0,weights=weight);bulk=np.average(v,axis=0,weights=weight)
    return {'position_kpc':x-center+finite(specification['center_kpc'],(3,)),
        'velocity_kms':v-bulk+finite(specification['center_velocity_kms'],(3,)), 'mass_msun':m.copy(),
        'particle_ids':np.arange(1,len(x)+1,dtype=np.uint64),'namespace':specification['tracer_namespace'],
        'parent_particle_ids':ids.copy(),'parent_namespace':sample['namespace'],
        'initialization':{'model':'actual measured progenitor; no equilibrium assumption',
            'parent_progenitor_reference':specification['parent_progenitor_reference']}}


def self_gravity(x,m,softening_kpc,*,chunk_rows):
    """Direct pair forces and per-particle potential with exact self exclusion.

    Phi_i=-G sum(j!=i) m_j/sqrt(|x_i-x_j|^2+eps^2) [(km/s)^2].
    a_i=-G sum(j!=i) m_j*(x_i-x_j)/(|x_i-x_j|^2+eps^2)^(3/2).
    Uniform Plummer softening defines the SAME force and binding potential.
    Work is O(N^2); memory is O(chunk*N). Self terms contribute neither.
    """
    x=finite(x)
    if x.ndim!=2 or x.shape[1]!=3:raise ValueError('Self gravity requires N by 3 Cartesian positions')
    m=finite(m,(len(x),),'gravitating masses')
    if np.any(m<=0):raise ValueError('Gravitating particle masses must be positive')
    eps=positive(softening_kpc,'softening_kpc');chunk=integer(chunk_rows,'chunk_rows')
    phi=np.zeros(len(x));acc=np.zeros_like(x)
    for start in range(0,len(x),chunk):
        stop=min(len(x),start+chunk);delta=x[start:stop,None,:]-x[None,:,:]
        r2=np.einsum('ijk,ijk->ij',delta,delta)+eps**2
        local=np.arange(stop-start);r2[local,np.arange(start,stop)]=np.inf
        phi[start:stop]=-G*np.sum(m[None,:]/np.sqrt(r2),axis=1)
        acc[start:stop]=-G*np.einsum('ij,ijk->ik',m[None,:]/r2**1.5,delta)
    return phi,acc


def measure_bound_remnant(x, v, m, *, softening_kpc, chunk_rows, maximum_iterations):
    """Remove positive-energy particles until the softened self-bound set settles.

    Specific binding energy is |v-v_bulk|^2/2+Phi_self [(km/s)^2]. This is
    a self-binding definition, excluding the external tidal boundary. Starting
    from all stars on each output allows previously stripped stars to recapture.
    """
    x=finite(x);v=finite(v,x.shape);m=finite(m,(len(x),),'gravitating masses')
    if x.ndim!=2 or x.shape[1]!=3 or not len(x) or np.any(m<=0):
        raise ValueError('Binding requires nonempty Cartesian particles with positive masses')
    active=np.ones(len(x),bool)
    for _ in range(integer(maximum_iterations,'maximum_iterations')):
        rows=np.flatnonzero(active)
        # Scale weights before averaging so a common mass-unit factor cancels.
        weight=m[rows]/m[rows].max()
        center=np.average(x[rows],axis=0,weights=weight)
        bulk=np.average(v[rows],axis=0,weights=weight)
        phi,_=self_gravity(x[rows],m[rows],softening_kpc,chunk_rows=chunk_rows)
        energy=finite(.5*np.sum((v[rows]-bulk)**2,axis=1)+phi,name='specific binding energy')
        remaining=energy<0
        if remaining.all():return active,center,bulk
        active[rows]=remaining
        if not active.any():return active,center,bulk
    raise ValueError('Self-bound remnant unresolved within the declared binding iterations')


def _force(x,m,t,host,controls):
    phi,self_acc=self_gravity(x,m,controls['softening_kpc'],chunk_rows=controls['force_chunk_rows'])
    external,ext_acc,_=host.evaluate(x,t)
    external=finite(external,(len(x),),'host potential')
    ext_acc=finite(ext_acc,x.shape,'host acceleration')
    return finite(self_acc+ext_acc,x.shape,'total acceleration'),finite(phi,(len(x),),'self potential'),external


def _state_energy(v, m, self_potential, external_potential):
    """K+U_self+U_host [Msun*(km/s)^2]; pair energy counts each pair once."""
    value=np.sum(.5*m*np.sum(v*v,axis=1))+.5*np.dot(m,self_potential)+np.dot(m,external_potential)
    return float(finite(value,(),'total energy'))


def save_live_checkpoint(path,state,*,binding):
    """Atomic numeric restart; includes all stripping clocks and ancestry."""
    import os
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Keep previous restart boundary')
    temporary=path.with_name('.'+path.name+'.tmp')
    arrays={k:v for k,v in state.items() if isinstance(v,np.ndarray)}
    scalar={k:v for k,v in state.items() if not isinstance(v,np.ndarray)}
    with temporary.open('xb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    atomic_json(str(path)+'.json',{'schema':'live-stream-restart-v1','array_sha256':digest(path),
        'state':json_safe(scalar),'binding':binding})
    atomic_text(str(path)+'.json.sha256',digest(str(path)+'.json')+'\n')


def load_live_checkpoint(path,*,binding,memory_gib):
    import zipfile
    memory=positive(memory_gib,'memory_gib')*1024**3
    path=Path(path).resolve();metadata=Path(str(path)+'.json')
    if metadata.stat().st_size*16>memory/8:raise MemoryError('Restart metadata exceeds memory bound')
    if digest(metadata)!=Path(str(path)+'.json.sha256').read_text().strip():raise ValueError('Restart clock/state metadata seal changed')
    record=json.loads(metadata.read_text())
    if record.get('schema')!='live-stream-restart-v1':raise ValueError('Unknown live restart schema')
    if record['binding']!=binding or digest(path)!=record['array_sha256']:raise ValueError('Restart physics, input or numerical binding changed')
    with zipfile.ZipFile(path) as z:
        members=z.infolist();names=[entry.filename for entry in members]
        if len(names)!=len(set(names)) or any(not name.endswith('.npy') or '/' in name or '\\' in name for name in names):
            raise ValueError('Restart requires flat unique numeric arrays')
        if sum(i.file_size for i in members)*3>memory:raise MemoryError('Restart exceeds memory bound')
    with np.load(path,allow_pickle=False) as f:state={k:f[k] for k in f.files}
    if set(record['state']).intersection(state):raise ValueError('Restart arrays collide with scalar state')
    return {**record['state'],**state}


def evolve_self_gravitating_progenitor(sample,host,output_times_myr,controls,*,checkpoint_directory=None,resume=None,binding=None):
    x,v,m,ids=phase_space(sample);x=x.copy();v=v.copy();times=increasing(output_times_myr)
    if (checkpoint_directory is not None or resume is not None) and (not isinstance(binding,dict) or not binding):
        raise ValueError('Checkpointing requires explicit physics and initial-state binding')
    n=len(x);integer(controls['maximum_particles'],'maximum_particles',32)
    if not n:raise ValueError('Live evolution requires a nonempty progenitor')
    chunk=integer(controls['force_chunk_rows'],'force_chunk_rows')
    softening=positive(controls['softening_kpc'],'softening_kpc')
    minimum_step=positive(controls['minimum_step_myr'],'minimum_step_myr')
    maximum_steps=integer(controls['maximum_steps'],'maximum_steps')
    integer(controls['binding_iterations'],'binding_iterations')
    if n>controls['maximum_particles']:raise MemoryError('Direct live reference exceeds particle bound; choose a native engine')
    estimate=n*len(times)*112+min(chunk,n)*n*64
    if estimate>positive(controls['memory_gib'],'memory_gib')*1024**3:raise MemoryError('Live particle histories/force workspace exceed memory')
    if sample.get('dark_matter_mass_msun',0)>0:raise ValueError('Native SIDM route required for DM-bearing progenitor')
    max_dt=positive(controls['maximum_step_myr'],'maximum_step_myr');eta=positive(controls['dynamical_step_fraction'],'dynamical_step_fraction')
    if minimum_step>max_dt:raise ValueError('Minimum timestep cannot exceed maximum timestep')
    if eta>1:raise ValueError('Dynamical-step fraction must not exceed one')
    release_before=np.full(n,np.nan);release_after=np.full(n,np.nan);release_position=np.full((n,3),np.nan);release_velocity=np.full((n,3),np.nan)
    bound=np.ones(n,bool);time=float(times[0]);step_count=0;start_index=0
    if resume is not None:
        state=load_live_checkpoint(resume,binding=binding,memory_gib=controls['memory_gib'])
        sx,sv,sm,sids=phase_space(state)
        if state['namespace']!=sample['namespace'] or not np.array_equal(sids,ids) or not np.array_equal(sm,m):
            raise ValueError('Restart identity or fixed particle masses differ')
        finite(sx,(n,3),'restart position');finite(sv,(n,3),'restart velocity')
        if 'parent_particle_ids' in sample and not np.array_equal(state.get('parent_particle_ids'),sample['parent_particle_ids']):
            raise ValueError('Restart parent ancestry differs')
        saved_bound=np.asarray(state['bound'])
        if saved_bound.shape!=(n,) or saved_bound.dtype.kind!='b':raise ValueError('Restart needs exact boolean binding mask')
        integer(state['step_count'],'restart step_count',0)
        state['time_myr']=float(finite(state['time_myr'],(),'restart time'))
        before=np.asarray(state['release_before_myr'],float);after=np.asarray(state['release_after_myr'],float)
        if before.shape!=(n,) or after.shape!=(n,) or np.isinf(before).any() or np.isinf(after).any():
            raise ValueError('Restart release clocks must align and use NaN for unavailable times')
        known=np.isfinite(after)
        if np.any(np.isfinite(before)&~known) or np.any(before[known]>after[known]) or np.any(after[known]>state['time_myr']):
            raise ValueError('Restart release brackets are inconsistent with the saved clock')
        for name in ('release_position_kpc','release_velocity_kms'):
            value=np.asarray(state[name],float)
            if value.shape!=(n,3) or not np.isfinite(value[known]).all():
                raise ValueError('Restart needs aligned finite states for recorded releases')
        x=state['position_kpc'];v=state['velocity_kms'];m=state['mass_msun'];time=state['time_myr'];step_count=state['step_count']
        bound=state['bound'];release_before=state['release_before_myr'];release_after=state['release_after_myr']
        release_position=state['release_position_kpc'];release_velocity=state['release_velocity_kms']
        if not np.any(times==time):raise ValueError('Resume must occur at a requested output boundary')
        start_index=int(np.flatnonzero(times==time)[0])
    mass_weights=m/m.max()
    # The KDK endpoint force is exactly the next substep's initial force.
    acceleration,phi,external=_force(x,m,time,host,controls)
    positions=[];velocities=[];bound_history=[];energies=[];centers=[];bulks=[];bound_masses=[];substeps=[];work=[]
    for target in times[start_index:]:
        work_interval=0.;steps=0;previous_output=time
        while time<target:
            omega=np.sqrt(np.max(np.linalg.norm(acceleration,axis=1))/softening)*KMS_TO_KPC_MYR
            bulk=np.average(v,axis=0,weights=mass_weights)
            internal_speed=np.max(np.linalg.norm(v-bulk,axis=1))*KMS_TO_KPC_MYR
            crossing=softening/max(internal_speed,1e-30)
            dt=min(max_dt,eta/max(omega,1e-30),eta*crossing,float(target-time))
            if dt<minimum_step and target-time>minimum_step:raise RuntimeError('Requested force accuracy exceeds minimum timestep budget')
            if step_count>=maximum_steps:raise RuntimeError('Live integration reached declared step limit')
            vhalf=v+.5*dt*KMS_TO_KPC_MYR*acceleration
            newx=x+dt*KMS_TO_KPC_MYR*vhalf
            newtime=min(float(target),time+dt)
            newacc,newphi,newexternal=_force(newx,m,newtime,host,controls)
            newv=vhalf+.5*dt*KMS_TO_KPC_MYR*newacc
            # Explicit-time potential work at fixed midpoint positions; energy
            # is not conserved when a moving/deforming external host does work.
            midpoint=.5*(x+newx)
            p0=finite(host.evaluate(midpoint,time)[0],(n,),'midpoint host potential')
            p1=finite(host.evaluate(midpoint,newtime)[0],(n,),'midpoint host potential')
            work_interval+=float(np.dot(m,p1-p0))
            if newtime<=time:raise ArithmeticError('Live clock cannot advance at requested precision')
            x=finite(newx,(n,3),'evolved position');v=finite(newv,(n,3),'evolved velocity');time=newtime
            acceleration,phi,external=newacc,newphi,newexternal
            step_count+=1;steps+=1
        oldbound=bound.copy()
        bound,center,bulk=measure_bound_remnant(x,v,m,softening_kpc=controls['softening_kpc'],
            chunk_rows=controls['force_chunk_rows'],maximum_iterations=controls['binding_iterations'])
        escaped=oldbound&~bound&~np.isfinite(release_after)
        release_before[escaped]=previous_output if positions else np.nan;release_after[escaped]=time
        release_position[escaped]=x[escaped];release_velocity[escaped]=v[escaped]
        positions.append(x.copy());velocities.append(v.copy());bound_history.append(bound.copy());centers.append(center);bulks.append(bulk)
        energies.append(_state_energy(v,m,phi,external));bound_masses.append(m[bound].sum());substeps.append(steps);work.append(float(finite(work_interval,(),'external work')))
        if checkpoint_directory is not None:
            state={'position_kpc':x,'velocity_kms':v,'mass_msun':m,'particle_ids':ids,'bound':bound,
                'release_before_myr':release_before,'release_after_myr':release_after,'release_position_kpc':release_position,
                'release_velocity_kms':release_velocity,'time_myr':time,'step_count':step_count,'namespace':sample['namespace']}
            if 'parent_particle_ids' in sample:state['parent_particle_ids']=sample['parent_particle_ids']
            path=Path(checkpoint_directory)/('boundary_%06d.npz'%(start_index+len(positions)-1))
            if resume is None or time!=times[start_index]:save_live_checkpoint(path,state,binding=binding)
    arrays={'times_myr':times[start_index:],'position_kpc':np.asarray(positions),'velocity_kms':np.asarray(velocities),
        'mass_msun':m,'particle_ids':ids,'bound':np.asarray(bound_history),'progenitor_center_kpc':np.asarray(centers),
        'progenitor_velocity_kms':np.asarray(bulks),'bound_mass_msun':np.asarray(bound_masses),
        'release_before_myr':release_before,'release_after_myr':release_after,'release_position_kpc':release_position,
        'release_velocity_kms':release_velocity,'total_energy_msun_kms2':np.asarray(energies),
        'external_work_msun_kms2':np.asarray(work),'substeps':np.asarray(substeps,dtype=np.int64)}
    if 'parent_particle_ids' in sample:arrays['parent_particle_ids']=sample['parent_particle_ids']
    report={'schema':'live-stream-resimulation-v1','backend':'direct self-gravitating KDK',
        'host_physics':host.physics,'stellar_sidm_scattering':False,'initialization':sample.get('initialization'),
        'parent_particle_namespace':sample.get('parent_namespace'),
        'binding_definition':'iteratively self-bound softened remnant; time-dependent host has no universal Jacobi energy',
        'release_time_definition':'first measured bound-to-unbound output bracket; recapture retained in bound history',
        'controls':controls,'step_count':step_count,'resume_parent':str(resume) if resume else None,
        'convergence_status':'NOT_EXECUTED','source_scope':'prescribed evolved host; progenitor does not backreact on parent host'}
    return report,arrays


def _integrate_tracers(x,v,start,stop,host,maximum_step_myr,progenitor=None,*,maximum_steps=1000000):
    """KDK tracers in the host plus a prescribed progenitor; no tracer self-force."""
    x=finite(x).copy();v=finite(v,x.shape).copy()
    if x.ndim!=2 or x.shape[1]!=3:raise ValueError('Spray tracers require N by 3 positions')
    time=float(finite(start,(),'release time'));stop=float(finite(stop,(),'final time'))
    step=positive(maximum_step_myr,'maximum_step_myr');limit=integer(maximum_steps,'maximum_steps')
    if stop<time:raise ValueError('Tracer evolution must proceed forward from release')

    def force(position,clock):
        acceleration=finite(host.evaluate(position,clock)[1],position.shape,'host acceleration').copy()
        if progenitor is not None:
            center=finite(progenitor['position'](clock),(3,),'progenitor center')
            mass=float(finite(progenitor['mass'](clock),(),'progenitor mass'))
            if mass<0:raise ValueError('Progenitor mass cannot be negative')
            scale=positive(progenitor['scale_kpc'],'progenitor_scale_kpc')
            delta=position-center
            acceleration-=G*mass*delta/(np.sum(delta*delta,axis=1)+scale**2)[:,None]**1.5
        return finite(acceleration,position.shape,'tracer acceleration')

    acceleration=force(x,time);steps=0
    while time<stop:
        if steps>=limit:raise RuntimeError('Tracer integration reached declared step limit')
        dt=min(step,stop-time);newtime=min(stop,time+dt)
        if newtime<=time:raise ArithmeticError('Tracer clock cannot advance at requested precision')
        half=v+.5*dt*KMS_TO_KPC_MYR*acceleration
        x=finite(x+dt*KMS_TO_KPC_MYR*half,x.shape,'tracer position')
        acceleration=force(x,newtime)
        v=finite(half+.5*dt*KMS_TO_KPC_MYR*acceleration,v.shape,'tracer velocity')
        time=newtime;steps+=1
    return x,v


def _tidal_release_frame(center,velocity,mass,host,time):
    """Instantaneous radial release frame and Jacobi scales, not an exact invariant.

    Omega=(r cross v_rel)/|r|^2 [km/s/kpc], D=|Omega|^2-nR^T H(Phi)nR,
    rJ=(G*M/D)^(1/3) [kpc], vJ=|Omega|*rJ [km/s]. Requires D>0 and a
    nonradial orbit; a time-dependent host still needs live stripping controls.
    """
    host_center,host_velocity,_=host.frame.state(time)
    offset=finite(center,(3,))-finite(host_center,(3,))
    relative=finite(velocity,(3,))-finite(host_velocity,(3,))
    radial,valid=normalize(offset)
    if not valid:raise ValueError('Tidal release is unresolved at the host center')
    omega=np.cross(offset,relative)/np.dot(offset,offset)
    normal,valid=normalize(omega)
    if not valid:raise ValueError('Radial tidal orbit has no resolved release frame')
    _,_,hessian=host.evaluate(np.asarray(center)[None],time)
    hessian=finite(hessian,(1,3,3),'host potential Hessian')[0]
    tide=float(np.dot(omega,omega)-radial@hessian@radial)
    if not np.isfinite(tide) or tide<=0:raise ValueError('No extensive radial tide; live stripping required')
    radius=(G*positive(mass,'progenitor mass')/tide)**(1/3)
    radius=positive(radius,'tidal radius')
    speed=positive(np.linalg.norm(omega)*radius,'tidal speed')
    return np.stack((radial,np.cross(normal,radial),normal)),radius,speed


def calibrated_spray_adapter(request,host,*,calibration):
    """Calibrated tidal release in the actual changing host, including satellites.

    Coefficients are empirical means/covariances of dimensionless release
    offsets, not claimed universal Fardal constants. Calibrations bind to live
    controls and host/progenitor domains. Uncalibrated experiments remain marked.
    """
    times=increasing(request['release_times_myr']);end=float(finite(request['end_time_myr'],(),'spray final time'))
    maximum_step=positive(request['maximum_step_myr'],'maximum_step_myr')
    maximum_steps=integer(request.get('maximum_steps',1000000),'maximum_steps')
    centers=finite(request['progenitor_position_kpc'],(len(times),3));velocity=finite(request['progenitor_velocity_kms'],centers.shape)
    mass=finite(request['progenitor_mass_msun'],(len(times),));scale=positive(request['progenitor_scale_kpc'],'progenitor_scale_kpc')
    if np.any(mass<=0) or end<times[-1]:raise ValueError('Positive masses and forward release times required')
    if not calibration.get('live_control_hashes'):raise ValueError('Spray requires measured live calibration references')
    if type(request.get('allow_uncalibrated_experiment',False)) is not bool:
        raise ValueError('allow_uncalibrated_experiment must be an explicit boolean')
    domain=calibration['domain']; parameters=request['calibration_parameters']
    accepted=calibration.get('status')=='PASS' and all(key in parameters and limits[0]<=parameters[key]<=limits[1] for key,limits in domain.items())
    if not accepted and not request.get('allow_uncalibrated_experiment',False):raise ValueError('Spray outside measured acceptance domain')
    means=finite(calibration['release_mean'],(2,6));cov=finite(calibration['release_covariance'],(2,6,6))
    if not np.allclose(cov,np.swapaxes(cov,-1,-2)) or np.any(np.linalg.eigvalsh(cov)<-1e-12):raise ValueError('Spray release covariance must be positive semidefinite')
    count=integer(request['particles_per_tail_per_release'],'particles_per_tail_per_release');rng=seeded_rng(request['seed'],'calibrated-spray')
    allx=[];allv=[];release=[];tails=[];radii=[];rawx=[];rawv=[];release_weights=[]
    # Require worldline coverage through final integration, separately from the release cadence.
    worldtimes=increasing(request['worldline_times_myr']);worldx=finite(request['worldline_position_kpc'],(len(worldtimes),3));worldv=finite(request['worldline_velocity_kms'],worldx.shape)
    if worldtimes[0]>times[0] or worldtimes[-1]<end:raise ValueError('Progenitor worldline does not cover the spray integration')
    motion=CubicHermiteSpline(worldtimes,worldx,worldv*KMS_TO_KPC_MYR,extrapolate=False)
    worldmass=finite(request['worldline_mass_msun'],(len(worldtimes),))
    if np.any(worldmass<0):raise ValueError('Progenitor worldline masses cannot be negative')
    if not np.allclose(motion(times),centers,rtol=1e-8,atol=1e-10) or not np.allclose(motion(times,1)/KMS_TO_KPC_MYR,velocity,rtol=1e-8,atol=1e-10):
        raise ValueError('Release centers/velocities disagree with the prescribed progenitor worldline')
    if not np.allclose(np.interp(times,worldtimes,worldmass),mass,rtol=1e-8,atol=0.):
        raise ValueError('Release masses disagree with the prescribed progenitor worldline')
    prog={'position':lambda t:motion(t),'mass':lambda t:np.interp(t,worldtimes,worldmass),'scale_kpc':scale}
    loss=finite(request['release_mass_msun'],(len(times),))
    if np.any(loss<0):raise ValueError('Release masses cannot be negative')
    if loss.sum()>mass[0]*(1+1e-10):raise ValueError('Release schedule exceeds the actual progenitor mass budget')
    for k,t in enumerate(times):
        if loss[k]==0:continue
        center=centers[k];vc=velocity[k]
        basis,radius,speed=_tidal_release_frame(center,vc,mass[k],host,t)
        for tail in range(2):
            draw=rng.multivariate_normal(means[tail],cov[tail],size=count)
            x0=center+radius*(draw[:,:3]@basis);v0=vc+speed*(draw[:,3:]@basis)
            xf,vf=_integrate_tracers(x0,v0,float(t),end,host,maximum_step,prog,maximum_steps=maximum_steps)
            allx.append(xf);allv.append(vf);rawx.append(x0);rawv.append(v0);release.extend([t]*count);tails.extend([tail]*count);radii.extend([radius]*count);release_weights.extend([loss[k]/(2*count)]*count)
    n=len(release)
    report={'schema':'stream-spray-v1','adapter':'live-calibrated tidal release','acceptance_domain_status':'CALIBRATED' if accepted else 'EXPERIMENT_ONLY',
        'calibration':calibration,'self_gravity':'prescribed progenitor Plummer force; no mutual tracer gravity', 'host_physics':host.physics,'empty_release_selection':n==0}
    concatenate=lambda values:np.concatenate(values) if values else np.empty((0,3))
    return report,{'position_kpc':concatenate(allx),'velocity_kms':concatenate(allv),
        'particle_ids':np.arange(1,n+1,dtype=np.uint64),'mass_msun':np.asarray(release_weights),
        'release_time_myr':np.asarray(release),'tail':np.asarray(tails,np.int64),'tidal_radius_kpc':np.asarray(radii),
        'release_position_kpc':concatenate(rawx),'release_velocity_kms':concatenate(rawv)}


def compare_live_and_spray(live,spray,*,tolerances,scale_position_kpc,scale_velocity_kms):
    """Forward comparison job: distributions, widths, tracks and release times."""
    scale_position_kpc=positive(scale_position_kpc,'scale_position_kpc')
    scale_velocity_kms=positive(scale_velocity_kms,'scale_velocity_kms')
    live_rows=~live['bound'][-1];lx=live['position_kpc'][-1,live_rows];lv=live['velocity_kms'][-1,live_rows]
    sx=spray['position_kpc'];sv=spray['velocity_kms']
    if len(lx)<8 or len(sx)<8:return {'status':'UNRESOLVED','reason':'insufficient stripped samples'},{}
    lm=live['mass_msun'][live_rows];sm=spray['mass_msun'];lc,lcov,_=weighted_covariance(lx,lm);sc,scov,_=weighted_covariance(sx,sm)
    metrics={'center_kpc':np.linalg.norm(lc-sc),'width_covariance_kpc2':np.linalg.norm(lcov-scov),
        'velocity_center_kms':np.linalg.norm(np.average(lv,axis=0,weights=lm/lm.max())-np.average(sv,axis=0,weights=sm/sm.max()))}
    release=live['release_after_myr'][live_rows];known=np.isfinite(release)
    metrics['release_wasserstein_myr']=wasserstein_distance(release[known],spray['release_time_myr'],u_weights=lm[known],v_weights=sm) if known.any() else np.nan
    ly=np.column_stack((lx/scale_position_kpc,lv/scale_velocity_kms));sy=np.column_stack((sx/scale_position_kpc,sv/scale_velocity_kms))
    distance=cKDTree(sy).query(ly)[0];reverse=cKDTree(ly).query(sy)[0]
    metrics['phase_space_chamfer']=.5*(np.average(distance,weights=lm)+np.average(reverse,weights=sm))
    if set(tolerances)!=set(metrics):raise ValueError('Freeze a dimensional tolerance for every comparison statistic')
    passed=all(np.isfinite(value) and value<=positive(tolerances[key],key) for key,value in metrics.items())
    return {'status':'PASS' if passed else 'FAIL','metrics':metrics,'tolerances':tolerances,
        'meaning':'single-control agreement; independent domain calibration is still required'}, {'live_to_spray_distance':distance,'spray_to_live_distance':reverse}


def gala_spray_adapter(request,potential,*,backend_binding=None):
    """Run pinned Gala as an independent static-host release reference.

    A signed timestep follows upstream's initial/final-state convention. The
    fixed published Fardal coefficients are the default; any Gala modification
    must be explicitly selected and is recorded with the result.
    """
    if backend_binding is None:_,backend_binding=require_backend('gala')
    import astropy.units as u
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.dynamics.mockstream import FardalStreamDF,MockStreamGenerator
    mass=positive(request['mass_msun'],'mass_msun');scale=positive(request['scale_kpc'],'scale_kpc')
    step=float(finite(request['step_myr'],(),'Gala step Myr'))
    if step==0:raise ValueError('Gala timestep cannot be zero')
    steps=integer(request['steps'],'steps');cadence=integer(request['release_every'],'release_every')
    particles=integer(request['particles_per_release'],'particles_per_release')
    modified=request.get('gala_modified',False)
    if type(modified) is not bool:raise ValueError('gala_modified must be an explicit boolean')
    df=FardalStreamDF(gala_modified=modified,random_state=np.random.RandomState(request['seed']))
    progenitor=gd.PhaseSpacePosition(pos=finite(request['position_kpc'],(3,))*u.kpc,
        vel=finite(request['velocity_kms'],(3,))*u.km/u.s)
    progenitor_potential=gp.PlummerPotential(m=mass*u.Msun,b=scale*u.kpc,units=potential.units)
    generator=MockStreamGenerator(df,gp.Hamiltonian(potential),progenitor_potential=progenitor_potential)
    stream,_=generator.run(progenitor,mass*u.Msun,dt=step*u.Myr,n_steps=steps,
        release_every=cadence,n_particles=particles,progress=False)
    position=finite(stream.xyz.to_value(u.kpc).T)
    if position.ndim!=2 or position.shape[1]!=3:raise ValueError('Gala returned invalid Cartesian positions')
    velocity=finite(stream.v_xyz.to_value(u.km/u.s).T,position.shape)
    tail=np.asarray(stream.lead_trail)
    if tail.shape!=(len(position),) or not np.isin(tail,['t','l']).all():
        raise ValueError('Gala returned unknown leading/trailing labels')
    return {'schema':'stream-gala-reference-v1','backend':backend_binding,'upstream':'gala',
        'scope':'independent static-host control; not the evolving-host production route',
        'release_prescription':'Gala-modified Fardal' if modified else 'published Fardal2015',
        'tail_encoding':{'0':'trailing','1':'leading'}}, {
        'position_kpc':position,'velocity_kms':velocity,
        'release_time_myr':finite(stream.release_time.to_value(u.Myr),(len(position),)),
        'tail':(tail=='l').astype(np.int64)}


def calibrate_spray_from_live(controls,host,*,minimum_release_particles,maximum_release_bracket_myr,
                              covariance_floor,parameter_domain,heldout_comparisons):
    """Fit dimensionless spray offsets from actual live escape states.

    Input controls are (report,arrays,source_sha256) tuples. Release-bracket
    limits prevent coarse output timing from masquerading as release physics.
    Held-out comparisons must bind distinct live controls and pass frozen
    track/width/velocity/time tolerances before this produces status PASS.
    """
    maximum_release_bracket_myr=positive(maximum_release_bracket_myr,'maximum_release_bracket_myr')
    minimum_release_particles=integer(minimum_release_particles,'minimum_release_particles',8)
    populations=[[],[]];weights=[[],[]];used=[];failures=[];training_hashes={}
    for report,data,checksum in controls:
        if report.get('schema')!='live-stream-resimulation-v1' or not checksum:
            raise ValueError('Spray calibration requires source-bound live evolution products')
        times=increasing(data['times_myr']);center=data['progenitor_center_kpc'];velocity=data['progenitor_velocity_kms']
        motion=CubicHermiteSpline(times,center,velocity*KMS_TO_KPC_MYR,extrapolate=False)
        mass=np.asarray(data['bound_mass_msun']);rows=np.flatnonzero(np.isfinite(data['release_before_myr'])&np.isfinite(data['release_after_myr']))
        if report['stage_id'] in training_hashes or checksum in training_hashes.values():
            raise ValueError('Release calibration requires distinct source-bound live controls')
        training_hashes[report['stage_id']]=checksum
        for row in rows:
            before=data['release_before_myr'][row];after=data['release_after_myr'][row]
            if after<before or after<times[0] or after>times[-1] or after-before>maximum_release_bracket_myr:
                failures.append({'control':checksum,'particle_id':int(data['particle_ids'][row]),'status':'RELEASE_CADENCE_TOO_COARSE'});continue
            c=motion(after);vc=motion(after,1)/KMS_TO_KPC_MYR
            boundmass=np.interp(after,times,mass)
            try:basis,rj,vj=_tidal_release_frame(c,vc,boundmass,host,after)
            except ValueError as exc:
                failures.append({'control':checksum,'particle_id':int(data['particle_ids'][row]),'reason':str(exc)})
                continue
            displacement=data['release_position_kpc'][row]-c;dv=data['release_velocity_kms'][row]-vc
            if not np.isfinite(displacement).all() or not np.isfinite(dv).all() or vj<=0:continue
            tail=0 if displacement@basis[0]>=0 else 1
            populations[tail].append(np.r_[displacement@basis.T/rj,dv@basis.T/vj]);weights[tail].append(data['mass_msun'][row])
        used.append({'live_source_sha256':checksum,'parameters':report.get('calibration_parameters'),
            'release_position_definition':'upper endpoint of first unbinding bracket; cadence explicitly bounded'})
    means=[];covariance=[]
    for tail in range(2):
        if len(populations[tail])<minimum_release_particles:raise ValueError('Insufficient actual escape states in each tidal tail')
        mean,cov,_=weighted_covariance(np.asarray(populations[tail]),np.asarray(weights[tail]))
        means.append(mean);covariance.append(cov+positive(covariance_floor,'covariance_floor')*np.eye(6))
    heldout_hashes={};accepted=True
    for comparison in heldout_comparisons:
        if comparison.get('schema')!='stream-live-spray-comparison-v1':
            raise ValueError('Heldout acceptance requires actual live/spray comparison reports')
        if comparison['comparison_id'] in heldout_hashes:raise ValueError('Duplicate heldout comparison identity')
        if comparison['status']!='PASS' or not comparison.get('live_source_sha256'):accepted=False
        if comparison.get('live_source_sha256') in training_hashes.values():raise ValueError('Heldout live control overlaps release calibration')
        heldout_hashes[comparison['comparison_id']]=comparison.get('live_source_sha256')
    if not heldout_comparisons:accepted=False
    for key,limits in parameter_domain.items():
        if len(limits)!=2 or not np.isfinite(limits).all() or limits[0]>limits[1]:raise ValueError('Calibration domain requires finite recorded bounds')
    return {'schema':'live-spray-calibration-v1','status':'PASS' if accepted else 'CALIBRATION_WRITTEN_HELDOUT_PENDING',
        'release_mean':np.asarray(means),'release_covariance':np.asarray(covariance),'domain':parameter_domain,
        'live_control_hashes':training_hashes,'heldout_live_hashes':heldout_hashes,'controls':used,'failures':failures,
        'maximum_release_bracket_myr':maximum_release_bracket_myr,'minimum_release_particles':minimum_release_particles,
        'parameter_domain_semantics':'frozen campaign domain; passing one control does not establish the full domain'}


# Optional independent engines share the live generator unit convention.

PIN_FILE=Path(__file__).resolve().parents[1]/'configs/streams.upstream.json'


def require_backend(name):
    """Use the shared native dependency binding for optional stream engines."""
    from core.native import verified_python_backend
    names={'gala':'gala','agama':'agama','StreaMAX':'StreaMAX'}
    if name not in names:raise ValueError('Unknown stream backend: '+str(name))
    records=json.loads(PIN_FILE.read_text())['repositories']
    record=next(row for row in records if row['name']==name)
    module=importlib.import_module(names[name])
    distribution=importlib.metadata.distribution(names[name])
    installation=verified_python_backend(names[name],record,distribution=distribution)
    if str(Path(module.__file__).resolve()) not in installation['installed_assets']:
        raise ValueError('Imported stream backend is outside its verified distribution')
    sources={**installation['source_hashes'],str(PIN_FILE):digest(PIN_FILE),
        str(Path(__file__).resolve()):digest(__file__)}
    return module,{'revision':record['revision'],'source_url':record['url'],
        'installed_source_hashes':sources,'license_files':record['license_files']}


def streamamax_reference(request):
    """Execute the actual pinned upstream JAX generator as a static control.

    StreaMAX currently enforces float32 and its public generator uses analytic
    hosts. Those restrictions are retained rather than claimed equivalent to
    the evolving-host live route. GPU/CPU differences need later controls.
    """
    module,binding=require_backend('StreaMAX')
    parameters=dict(request)
    if parameters.pop('input_units')!='proper_kpc_kms_myr':raise ValueError('Declare proper_kpc_kms_myr reference input units')
    if parameters['type_host'] not in {'PointMass','Isochrone','Plummer','NFW','NFWFlatDensity','MiyamotoNagai','Hernquist','ExpDisk','DiskNFW'}:
        raise ValueError('This adapter supports mass/length-parameterized static StreaMAX hosts')
    parameters['time']=positive(parameters.pop('time_myr'),'time_myr')/1000
    state=finite(parameters['xv_f'],(6,)).copy();state[3:]*=KMS_TO_KPC_MYR*1000;parameters['xv_f']=state
    steps=integer(parameters['n_steps'],'n_steps');particles=integer(parameters['n_particles'],'n_particles')
    # Upstream advances each release to t_release+(time-t_release)*alpha.
    # Only alpha=1 puts all output particles at the same physical final epoch.
    alpha=positive(parameters['alpha'],'alpha')
    if alpha!=1:raise ValueError('Single-epoch StreaMAX products require alpha=1')
    if parameters['type_sat'] not in {'PointMass','Isochrone','Plummer','NFW'}:
        raise ValueError('Unsupported StreaMAX satellite potential')
    for name in ('params_host','params_sat'):
        for key,value in parameters[name].items():finite(value,name=name+'.'+key)
    if parameters.get('type_method','Fardal2015') not in {'Fardal2015','Chen2025'}:
        raise ValueError('Unsupported StreaMAX release prescription')
    if particles%(steps+1)!=0:raise ValueError('Upstream release allocation needs n_particles divisible by n_steps+1')
    times,progenitor,stream,accumulated_radial_offset=module.generate_stream(**parameters)
    progenitor=finite(progenitor,(steps+1,6),'StreaMAX progenitor').copy()
    stream=finite(stream,(particles,6),'StreaMAX stream').copy()
    times=finite(times,(steps+1,),'StreaMAX epochs')
    progenitor[:,3:]/=KMS_TO_KPC_MYR*1000;stream[:,3:]/=KMS_TO_KPC_MYR*1000
    return {'schema':'streamamax-reference-v1','scope':'independent static analytic-host particle spray',
        'precision':'upstream float32','backend':binding,'self_gravity':'upstream prescribed progenitor',
        'native_units':'kpc, kpc/Gyr, Gyr; converted to shared proper kpc, km/s, Myr',
        'mass_parameters':'logM and m_f_sat are base-10 logarithms of masses in Msun',
        'final_time_myr':float(times[-1])*1000,
        'accumulated_offset_semantics':'upstream sum of progenitor-minus-tracer radii over integration steps; not a release state'}, {
        'times_myr':np.asarray(times)*1000,'progenitor_position_kpc':progenitor[:,:3],'progenitor_velocity_kms':progenitor[:,3:],
        'position_kpc':stream[:,:3],'velocity_kms':stream[:,3:],
        'release_time_myr':np.repeat(np.asarray(times)*1000,particles//(steps+1)),
        'accumulated_radial_offset_kpc':np.asarray(accumulated_radial_offset)}
