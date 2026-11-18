"""Actions, angles, frequencies and explicit orbit-validity diagnostics.

Actions: (Jr,Jz,Jphi) in kpc km/s, angles radians, frequencies km/s/kpc.
Spherical quadrature is independent of AGAMA; axisymmetric Staeckel-fudge
uses the project's pinned AGAMA backend. General fields get orbit diagnostics
and unavailable invariant actions instead of invented conserved quantities.
"""

import numpy as np
from scipy.integrate import quad,solve_ivp
from scipy.optimize import brentq,minimize_scalar
from core.stream_common import G, KMS_TO_KPC_MYR, finite, positive, integer, increasing, weighted_covariance


def _phase_coordinates(samples):
    """Return aligned Cartesian positions [proper kpc] and velocities [km/s]."""
    position = finite(samples['position_kpc'], name='position_kpc')
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError('Action/orbit samples require N by 3 Cartesian positions')
    velocity = finite(samples['velocity_kms'], position.shape, 'velocity_kms')
    return position, velocity


class AnalyticSphericalPotential:
    """Independent analytic controls in the shared physical unit convention."""
    def __init__(self,kind,*,mass_msun=None,scale_kpc=None,omega_kms_kpc=None):
        if kind not in {'kepler','isochrone','plummer','harmonic'}:raise ValueError('Unsupported analytic potential')
        self.kind=kind;self.domain='spherical';self.time_dependence=False
        self.mass=positive(mass_msun,'mass_msun') if kind!='harmonic' else None
        self.scale=positive(scale_kpc,'scale_kpc') if kind in {'isochrone','plummer'} else 0.
        self.omega=positive(omega_kms_kpc,'omega_kms_kpc') if kind=='harmonic' else None
        self.escape_energy=np.inf if kind=='harmonic' else 0.

    def radial_value(self,r):
        r=np.asarray(r)
        if self.kind=='kepler':return -G*self.mass/r
        if self.kind=='plummer':return -G*self.mass/np.sqrt(r*r+self.scale**2)
        if self.kind=='isochrone':return -G*self.mass/(self.scale+np.sqrt(self.scale**2+r*r))
        return .5*self.omega**2*r*r

    def radial_derivative(self,r):
        if self.kind=='kepler':return G*self.mass/r**2
        if self.kind=='plummer':return G*self.mass*r/(r*r+self.scale**2)**1.5
        if self.kind=='isochrone':
            q=np.sqrt(self.scale**2+r*r);return G*self.mass*r/(q*(self.scale+q)**2)
        return self.omega**2*r

    def radial_second(self,r):
        if self.kind=='kepler':return -2*G*self.mass/r**3
        if self.kind=='plummer':return G*self.mass*(self.scale**2-2*r*r)/(r*r+self.scale**2)**2.5
        if self.kind=='isochrone':
            q=np.sqrt(self.scale**2+r*r);return G*self.mass*(1/(q*(self.scale+q)**2)-r*r/(q**3*(self.scale+q)**2)-2*r*r/(q*q*(self.scale+q)**3))
        return self.omega**2

    def evaluate(self, x, t=None):
        """Return Phi, acceleration and the Cartesian Hessian of Phi.

        For a radial potential H = (Phi'/r) I + (Phi''-Phi'/r) u u^T.
        Phi is in (km/s)^2; the Hessian adds two inverse proper kpc factors.
        Smooth cores use their analytic r=0 limit; Kepler is singular there.
        """
        x = finite(x)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError('Potential queries require N by 3 Cartesian positions')
        radius = np.linalg.norm(x, axis=1)
        if self.kind == 'kepler' and np.any(radius == 0):
            raise ValueError('Kepler origin singular')
        direction = np.divide(x, radius[:,None], out=np.zeros_like(x), where=radius[:,None] > 0)
        phi = self.radial_value(radius)
        gradient = self.radial_derivative(radius)
        second = np.broadcast_to(self.radial_second(radius), radius.shape)
        # At a smooth isotropic origin, lim(Phi'/r)=Phi''(0).
        ratio = np.divide(gradient, radius, out=second.copy(), where=radius > 0)
        hessian = (ratio[:,None,None]*np.eye(3)
            + (second-ratio)[:,None,None]*direction[:,:,None]*direction[:,None,:])
        return phi, -gradient[:,None]*direction, hessian

    def analytical_actions(self, x, v):
        """Closed bound-orbit actions per unit mass, in proper kpc km/s.

        Spherical actions use Jz=L-|Lz| and Jphi=Lz. Jr is the radial
        cycle integral divided by 2*pi; unbound rows have no finite actions.
        """
        x, v = _phase_coordinates({'position_kpc':x, 'velocity_kms':v})
        energy = .5*np.sum(v*v, axis=1)+self.radial_value(np.linalg.norm(x, axis=1))
        angular_momentum = np.cross(x, v)
        total = np.linalg.norm(angular_momentum, axis=1)
        vertical = angular_momentum[:,2]
        bound = np.isfinite(energy) & (energy < self.escape_energy)
        radial = np.full(len(x), np.nan)
        if self.kind == 'kepler':
            radial[bound] = G*self.mass/np.sqrt(-2*energy[bound])-total[bound]
        elif self.kind == 'isochrone':
            radial[bound] = (G*self.mass/np.sqrt(-2*energy[bound])
                - .5*(total[bound]+np.sqrt(total[bound]**2+4*G*self.mass*self.scale)))
        elif self.kind == 'harmonic':
            radial[bound] = .5*(energy[bound]/self.omega-total[bound])
        else:
            raise ValueError('Plummer has no implemented closed action expression')
        actions = np.column_stack((radial, total-abs(vertical), vertical))
        actions[~bound] = np.nan
        return actions


class SphericalFieldAdapter:
    def __init__(self,field,*,radial_domain_kpc,escape_energy):
        self.field=field;self.radial_domain=increasing(radial_domain_kpc,'radial domain');self.escape_energy=escape_energy
        self.domain='spherical';self.time_dependence=False
    def radial_value(self,r):
        a=np.atleast_1d(r)
        if np.any(a<self.radial_domain[0]) or np.any(a>self.radial_domain[-1]):raise ValueError('Spherical query exceeds calibrated radial field domain')
        result=self.field.evaluate(np.column_stack((a,np.zeros_like(a),np.zeros_like(a))))[0]
        return result if np.ndim(r) else float(result[0])
    def evaluate(self,x,t=None):
        radius=np.linalg.norm(x,axis=1)
        if np.any(radius<self.radial_domain[0]) or np.any(radius>self.radial_domain[-1]):raise ValueError('Orbit exits calibrated spherical radial support')
        return self.field.evaluate(x)


def classify_potential_domain(record):
    symmetry=record['symmetry'];time_dependent=record['time_dependent']
    if type(time_dependent) is not bool or symmetry not in {'spherical','axisymmetric','triaxial','general'}:raise ValueError('Declare potential symmetry/time dependence')
    if symmetry=='spherical':method='spherical_quadrature'
    elif symmetry=='axisymmetric':method='agama_staeckel_fudge'
    else:method='orbit_diagnostics_only'
    if time_dependent and symmetry=='axisymmetric':method='orbit_diagnostics_only'
    return {'symmetry':symmetry,'action_method':method,'time_dependent':time_dependent,
        'interpretation':'frozen epoch approximation' if time_dependent else 'stationary potential',
        'exact_invariants_available':symmetry=='spherical' and not time_dependent,
        'potential_time_variation':record.get('time_variation'), 'symmetry_evidence':record['symmetry_evidence']}


def _radial_integrals(potential,energy,angular_momentum,radius,controls):
    minimum = positive(controls['minimum_radius_kpc'], 'minimum_radius_kpc')
    maximum = positive(controls['maximum_radius_kpc'], 'maximum_radius_kpc')
    root_tolerance = positive(controls['root_tolerance_kpc'], 'root_tolerance_kpc')
    if maximum <= minimum:
        raise ValueError('Turning-point search requires minimum radius < maximum radius')
    grid = np.geomspace(minimum, maximum,
        integer(controls['turning_grid_points'], 'turning_grid_points', 64))

    def speed2(r):
        # v_r^2 = 2(E-Phi) - L^2/r^2, in (km/s)^2.
        return 2*(energy-potential.radial_value(r))-angular_momentum**2/r**2

    values = finite([speed2(r) for r in grid], name='radial speed squared')
    # Include exact grid zeros once, then bracket only strict sign changes.
    roots = list(grid[values == 0])
    for lo, hi, left, right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if left != 0 and right != 0 and np.signbit(left) != np.signbit(right):
            roots.append(brentq(speed2, lo, hi, xtol=root_tolerance))
    roots.sort()
    if len(roots)!=2:
        # Circular or radial singular limits have undefined angles, not fake zeros.
        raise ValueError('Turning points unresolved, circular/radial limit, unbound orbit, or radial domain incomplete')
    rp,ra=roots
    if ra-rp <= root_tolerance:
        raise ValueError('Turning interval is unresolved at the requested radius tolerance')
    if radius<rp-controls['root_tolerance_kpc'] or radius>ra+controls['root_tolerance_kpc']:raise ValueError('Measured state outside radial turning interval')
    # r(q)=midpoint-half*cos(q) removes the square-root endpoint singularity.
    # Jr=(1/pi) integral v_r dr; half_time=integral dr/v_r [kpc/(km/s)].
    midpoint=.5*(rp+ra);half=.5*(ra-rp)
    tolerance=positive(controls['quadrature_relative_tolerance'],'quadrature_relative_tolerance')
    theta=np.arccos(np.clip((midpoint-radius)/half,-1,1))
    def integrand(q,mode):
        r=midpoint-half*np.cos(q);jac=half*np.sin(q);v2=max(float(speed2(r)),0.)
        if mode=='action':return np.sqrt(v2)*jac
        # Quad normally avoids endpoints. Very small v^2 is protected using a
        # local endpoint derivative, not an arbitrary physical speed floor.
        if v2<=1e-24:
            edge=rp if q<np.pi/2 else ra
            h=max(controls['root_tolerance_kpc']*10,edge*1e-6)
            lo,hi=max(minimum,edge-h),min(maximum,edge+h)
            slope=(speed2(hi)-speed2(lo))/(hi-lo)
            if not np.isfinite(slope) or slope == 0:
                raise ValueError('Turning-point slope is unresolved')
            ratio=np.sqrt(2*half/max(abs(slope),1e-300))
        else:ratio=jac/np.sqrt(v2)
        return ratio if mode=='time' else ratio*angular_momentum/r**2
    absolute_floor = float(finite(controls['quadrature_absolute_floor'], (), 'quadrature absolute floor'))
    if absolute_floor < 0:
        raise ValueError('Quadrature absolute floor cannot be negative')
    limit = integer(controls['maximum_quadrature_subintervals'], 'maximum_quadrature_subintervals')
    result, error = {}, {}
    for name, mode, upper in [('jr','action',np.pi), ('half_time','time',np.pi),
            ('half_azimuth','azimuth',np.pi), ('partial_time','time',theta),
            ('partial_azimuth','azimuth',theta)]:
        output = quad(lambda q:integrand(q,mode), 0, upper,
            epsabs=absolute_floor, epsrel=tolerance, limit=limit, full_output=True)
        # QUADPACK appends a diagnostic message when it fails to converge.
        # A finite last estimate alone does not establish a resolved action.
        if len(output) != 3:
            raise ValueError('Radial quadrature unresolved: '+str(output[3]))
        result[name], error[name] = output[:2]
        if not np.isfinite(result[name]) or not np.isfinite(error[name]):
            raise ValueError('Radial quadrature returned a nonfinite integral or error')
    if result['half_time'] <= 0 or result['half_azimuth'] <= 0 or result['jr'] < 0:
        raise ValueError('Radial quadrature requires a positive period and nonnegative action')
    result['jr']/=np.pi;error['jr']/=np.pi
    return rp,ra,result,error


def spherical_actions_angles(samples,potential,controls):
    x,v=_phase_coordinates(samples);n=len(x)
    angular_momentum_floor = positive(controls['angular_momentum_floor_kpc_kms'], 'angular_momentum_floor_kpc_kms')
    actions=np.full((n,3),np.nan);angles=np.full((n,3),np.nan);frequencies=np.full((n,3),np.nan);turning=np.full((n,2),np.nan)
    errors=np.full(n,np.nan);supported=np.zeros(n,bool);angle_supported=np.zeros((n,3),bool);findings=[]
    for i,(p,w) in enumerate(zip(x,v)):
        r=np.linalg.norm(p);lvec=np.cross(p,w);l=np.linalg.norm(lvec);lz=lvec[2]
        if r == 0 or l < angular_momentum_floor:
            findings.append({'row':i,'status':'UNBOUND_OR_RADIAL_DEGENERATE'});continue
        energy=.5*np.dot(w,w)+potential.radial_value(r)
        if not np.isfinite(energy) or energy>=potential.escape_energy:
            findings.append({'row':i,'status':'UNBOUND_OR_RADIAL_DEGENERATE'});continue
        # Exact circular limits have Jr=0 but no defined radial phase. Keeping
        # their real actions avoids falsely rejecting common analytic controls.
        if hasattr(potential,'radial_derivative'):
            radial_speed=np.dot(p,w)/r;gradient=potential.radial_derivative(r)
            circular_residual=abs(l*l/r**3-gradient)
            if abs(radial_speed)<1e-10 and circular_residual<1e-10*max(abs(gradient),1.):
                omega_l=np.sqrt(gradient/r)
                omega_r=np.sqrt(max(0.,potential.radial_second(r)+3*gradient/r))
                actions[i]=[0.,l-abs(lz),lz];frequencies[i]=[omega_r,omega_l,np.sign(lz)*omega_l]
                turning[i]=r;errors[i]=0.;supported[i]=True
                findings.append({'row':i,'status':'CIRCULAR_ACTIONS; RADIAL_PHASE_UNDEFINED'})
                continue
        try:rp,ra,result,error=_radial_integrals(potential,energy,l,r,controls)
        except ValueError as exc:findings.append({'row':i,'status':'UNRESOLVED','reason':str(exc)});continue
        inward=np.dot(p,w)<0
        time=2*result['half_time']-result['partial_time'] if inward else result['partial_time']
        azimuth=2*result['half_azimuth']-result['partial_azimuth'] if inward else result['partial_azimuth']
        # Tr=2*half_time; Omega_r=2*pi/Tr and Omega_L=Delta_psi/Tr.
        # Both frequencies are angular rates in km/s/kpc.
        omega_r=np.pi/result['half_time'];omega_l=result['half_azimuth']/result['half_time']
        actions[i]=[result['jr'],l-abs(lz),lz];frequencies[i]=[omega_r,omega_l,np.sign(lz)*omega_l];turning[i]=[rp,ra];errors[i]=error['jr'];supported[i]=True
        lhat=lvec/l;node=np.cross([0.,0.,1.],lhat);node_norm=np.linalg.norm(node)
        e1=node/node_norm if node_norm>1e-12 else np.array([1.,0.,0.]);e2=np.cross(lhat,e1)
        psi=np.arctan2(np.dot(p,e2),np.dot(p,e1));node_angle=np.arctan2(e1[1],e1[0]);theta_z=psi+omega_l*time-azimuth
        angles[i]=np.mod([omega_r*time,theta_z,node_angle+np.sign(lz)*theta_z],2*np.pi)
        angle_supported[i]=[True,node_norm>1e-12,node_norm>1e-12]
    return {'method':'independent spherical radial quadrature','findings':findings,
        'angle_convention':'radial pericenter zero; latitudinal ascending-node convention; equatorial node angles unavailable'}, {
        'actions_kpc_kms':actions,'angles_rad':angles,'frequencies_kms_kpc':frequencies,'action_supported':supported,
        'angle_supported':angle_supported,'turning_radii_kpc':turning,'jr_quadrature_error_kpc_kms':errors}


def agama_adapter(samples,potential,domain,*,focal_distance_kpc=None):
    from core.halos import pinned_agama_backend
    agama,binding=pinned_agama_backend();x,v=_phase_coordinates(samples)
    if potential is None:raise ValueError('An explicit AGAMA potential is required')
    if domain['symmetry'] not in {'spherical','axisymmetric'}:raise ValueError('AGAMA Staeckel/spherical adapter cannot certify general triaxial actions')
    # The pinned backend returns columns Jr,Jz,Jphi and angular frequencies.
    kwargs={'actions':True,'angles':True,'frequencies':True}
    if focal_distance_kpc is not None:kwargs['fd']=positive(focal_distance_kpc,'focal_distance_kpc')
    action,angle,frequency=agama.actions(potential=potential,point=np.column_stack((x,v)),**kwargs)
    action=np.asarray(action);angle=np.asarray(angle);frequency=np.asarray(frequency)
    if any(value.shape != x.shape for value in (action,angle,frequency)):
        raise ValueError('AGAMA returned action/angle/frequency arrays with unexpected shape')
    supported=np.isfinite(action).all(axis=1)&np.isfinite(frequency).all(axis=1)
    return {'method':'AGAMA spherical' if domain['symmetry']=='spherical' else 'AGAMA Staeckel fudge',
        'backend':binding,'domain':domain,'focal_distance_kpc':focal_distance_kpc}, {
        'actions_kpc_kms':action,'angles_rad':angle,'frequencies_kms_kpc':frequency,
        'action_supported':supported,'angle_supported':np.isfinite(angle)}


def integrate_orbits(samples, potential, times_myr, *, relative_tolerance,
                     position_absolute_tolerance_kpc, velocity_absolute_tolerance_kms,
                     maximum_step_myr, maximum_force_evaluations=1000000):
    """Integrate inertial Cartesian states; output shape is (epochs, particles, 6).

    Initial states belong to times_myr[0]. Phi uses (km/s)^2, so its force
    -grad(Phi) has (km/s)^2/kpc. Multiplying both equations by the shared
    km/s-to-kpc/Myr factor gives dx/dt in kpc/Myr and dv/dt in km/s/Myr.
    """
    times = increasing(times_myr)
    position, velocity = _phase_coordinates(samples)
    tolerance = positive(relative_tolerance, 'relative_tolerance')
    position_floor = positive(position_absolute_tolerance_kpc, 'position_absolute_tolerance_kpc')
    velocity_floor = positive(velocity_absolute_tolerance_kms, 'velocity_absolute_tolerance_kms')
    maximum_step = positive(maximum_step_myr, 'maximum_step_myr')
    budget = integer(maximum_force_evaluations, 'maximum_force_evaluations')
    absolute_tolerance = np.r_[np.full(3, position_floor), np.full(3, velocity_floor)]
    states = np.empty((len(times), len(position), 6))
    evaluations = np.empty(len(position), dtype=np.int64)
    for row, (point, speed) in enumerate(zip(position, velocity)):
        calls = 0

        def derivative(time, state):
            nonlocal calls
            calls += 1
            if calls > budget:
                raise RuntimeError('Independent orbit force-evaluation budget exceeded')
            acceleration = finite(potential.evaluate(state[:3][None], time)[1], (1, 3), 'acceleration')[0]
            return KMS_TO_KPC_MYR*np.r_[state[3:], acceleration]

        result = solve_ivp(derivative, (times[0], times[-1]), np.r_[point, speed],
            method='DOP853', t_eval=times, rtol=tolerance, atol=absolute_tolerance,
            max_step=maximum_step)
        if not result.success:
            raise RuntimeError('Independent orbit integration failed: '+result.message)
        states[:, row] = finite(result.y.T, (len(times), 6), 'integrated orbit')
        evaluations[row] = result.nfev
    return states, evaluations


def recover_frequencies(times_myr,positions):
    """Windowed Fourier peak refined by complex-amplitude maximization.

    Cartesian fundamentals can be harmonics or combinations of action
    frequencies. They are reported as spectral peaks, never auto-identified as
    canonical frequencies. Frequency drift floor includes finite duration.
    """
    times=increasing(times_myr);x=finite(positions)
    if len(times)<32 or not np.allclose(np.diff(times),np.diff(times)[0],rtol=1e-8,atol=0.):raise ValueError('Frequency recovery needs >=32 uniformly sampled epochs')
    if x.shape[0]!=len(times):raise ValueError('Orbit sample count differs from clock')
    shape=x.shape[1:];flat=x.reshape(len(times),-1);window=np.hanning(len(times));dt=times[1]-times[0]
    grid=np.fft.rfftfreq(len(times),dt)*2*np.pi;frequency=np.full(flat.shape[1],np.nan);quality=np.zeros(flat.shape[1])
    for k in range(flat.shape[1]):
        y=(flat[:,k]-flat[:,k].mean())*window;power=np.abs(np.fft.rfft(y))**2
        if power[1:].sum()<=0:continue
        best=int(np.argmax(power[1:]))+1;lo=grid[max(1,best-1)];hi=grid[min(len(grid)-1,best+1)]
        result=minimize_scalar(lambda w:-abs(np.sum(y*np.exp(-1j*w*(times-times[0]))))**2,bounds=(lo,hi),method='bounded')
        if not result.success or not np.isfinite(result.x):
            continue
        frequency[k]=result.x;quality[k]=power[best]/power[1:].sum()
    return frequency.reshape(shape),quality.reshape(shape),2*np.pi/(times[-1]-times[0])


def finite_time_lyapunov(sample, potential, times, controls):
    """Measure finite-time tangent growth along one initial displacement.

    The variational equations are d(delta x)/dt=c*delta v and
    d(delta v)/dt=-c*H(Phi)*delta x, with c converting km/s to kpc/Myr.
    The reported norm uses the declared length and speed scales. A single
    displacement along x is an indicator, not the maximal Lyapunov exponent.
    """
    position = finite(sample['position_kpc'], (3,))
    velocity = finite(sample['velocity_kms'], (3,))
    times = increasing(times)
    length = positive(controls['phase_position_scale_kpc'], 'phase_position_scale_kpc')
    speed = positive(controls['phase_velocity_scale_kms'], 'phase_velocity_scale_kms')
    tolerance = positive(controls['relative_tolerance'], 'relative_tolerance')
    maximum_step = positive(controls['maximum_step_myr'], 'maximum_step_myr')
    absolute_tolerance = finite(controls['variational_absolute_tolerance'])
    if absolute_tolerance.shape not in ((), (12,)) or np.any(absolute_tolerance <= 0):
        raise ValueError('Variational tolerance requires a positive scalar or 12 state tolerances')
    budget = integer(controls.get('maximum_force_evaluations',1000000), 'maximum_force_evaluations')
    initial = np.r_[position, velocity, length, 0., 0., 0., 0., 0.]
    calls = 0

    def derivative(time, state):
        nonlocal calls
        calls += 1
        if calls > budget:
            raise RuntimeError('Variational orbit force-evaluation budget exceeded')
        _, acceleration, hessian = potential.evaluate(state[:3][None], time)
        acceleration = finite(acceleration, (1,3), 'acceleration')[0]
        hessian = finite(hessian, (1,3,3), 'potential Hessian')[0]
        return KMS_TO_KPC_MYR*np.r_[state[3:6], acceleration,
            state[9:12], -hessian@state[6:9]]

    result = solve_ivp(derivative, (times[0],times[-1]), initial,
        t_eval=times, method='DOP853', rtol=tolerance,
        atol=absolute_tolerance, max_step=maximum_step)
    if not result.success:
        raise RuntimeError('Variational orbit integration failed: '+result.message)
    states = finite(result.y.T, (len(times),12), 'variational orbit')
    delta = states[:,6:]/np.r_[np.full(3,length), np.full(3,speed)]
    norm = np.hypot.reduce(delta,axis=1)
    duration = times-times[0]
    exponent = np.divide(np.log(norm/norm[0]), duration,
        out=np.full(len(times),np.nan), where=duration > 0)
    return {'times_myr':times, 'tangent_norm':norm,
        'finite_time_lyapunov_myr_inverse':exponent,
        'meaning':'finite-time indicator; positive finite-time growth alone does not prove asymptotic chaos'}


def _instantaneous_actions(samples, potential, domain, controls, *, time_myr=None,
                           agama_potential=None):
    """Use the declared model at one epoch for every action measurement."""
    method = domain['action_method']
    if method == 'spherical_quadrature':
        if domain['time_dependent']:
            frozen = FrozenHostSphericalView(potential, time_myr)
            samples = frozen.local_sample(samples)
            potential = frozen
        return spherical_actions_angles(samples, potential, controls['quadrature'])
    if method == 'agama_staeckel_fudge':
        return agama_adapter(samples, agama_potential, domain,
            focal_distance_kpc=controls.get('focal_distance_kpc'))
    raise ValueError('No action estimator for the declared potential domain')


def action_history(samples,potential,times,controls,*,domain,agama_potential=None):
    times = increasing(times, minimum=64)
    if not np.allclose(np.diff(times),np.diff(times)[0],rtol=1e-8,atol=0.):
        raise ValueError('Frequency drift comparisons require uniformly sampled epochs')
    positive(controls['refinement_factor'],'refinement_factor')
    if controls['refinement_factor']<=1:raise ValueError('Numerical refinement must be stricter than baseline')
    orbit,counts=integrate_orbits(samples,potential,times,**controls['integrator'])
    actions=[];angles=[];frequencies=[];supported=[];angle_masks=[]
    for clock,state in zip(times,orbit):
        sample={'position_kpc':state[:,:3],'velocity_kms':state[:,3:]}
        if domain['action_method'] == 'orbit_diagnostics_only':
            continue
        _, value = _instantaneous_actions(sample, potential, domain, controls,
            time_myr=clock, agama_potential=agama_potential)
        actions.append(value['actions_kpc_kms']);angles.append(value['angles_rad']);frequencies.append(value['frequencies_kms_kpc']);supported.append(value['action_supported']);angle_masks.append(value['angle_supported'])
    result={'orbit_position_kpc':orbit[:,:,:3],'orbit_velocity_kms':orbit[:,:,3:],'orbit_times_myr':np.asarray(times),'force_evaluations':counts}
    split=len(times)//2
    first,quality1,floor1=recover_frequencies(times[:split],orbit[:split,:,:3]);second,quality2,floor2=recover_frequencies(times[split:],orbit[split:,:,:3])
    result.update({'spectral_frequency_first_myr_inverse':first,'spectral_frequency_second_myr_inverse':second,
        'spectral_frequency_difference_myr_inverse':second-first,'spectral_quality_first':quality1,'spectral_quality_second':quality2,
        'spectral_diffusion_floor_myr_inverse':np.full_like(first,max(floor1,floor2))})
    if actions:
        actions=np.asarray(actions);result.update({'action_history_kpc_kms':actions,'angle_history_rad':np.asarray(angles),
            'angle_history_supported':np.asarray(angle_masks),'frequency_history_kms_kpc':np.asarray(frequencies),'action_history_supported':np.asarray(supported),
            'action_drift_kpc_kms':actions-actions[0]})
        angles=np.asarray(angles);frequencies=np.asarray(frequencies);angle_masks=np.asarray(angle_masks)
        angular_frequency=np.full(angles.shape[1:],np.nan);angle_rate_supported=np.zeros(angles.shape[1:],bool)
        cadence=np.diff(times)
        for particle in range(angles.shape[1]):
            for dimension in range(3):
                a=angles[:,particle,dimension];f=frequencies[:,particle,dimension]
                if not angle_masks[:,particle,dimension].all() or not np.isfinite(a).all() or not np.isfinite(f).all():continue
                if np.any(np.maximum(abs(f[:-1]),abs(f[1:]))*KMS_TO_KPC_MYR*cadence>=np.pi):continue
                angular_frequency[particle,dimension]=np.polyfit(np.asarray(times)-times[0],np.unwrap(a),1)[0]
                angle_rate_supported[particle,dimension]=True
        result['angle_rate_frequency_myr_inverse']=angular_frequency
        result['angle_rate_supported']=angle_rate_supported
        result['angle_rate_minus_canonical_frequency_myr_inverse']=angular_frequency-np.nanmedian(frequencies,axis=0)*KMS_TO_KPC_MYR
        # Numerical drift floor from the same potential and initial conditions,
        # integrated at stricter tolerances; no physical drift is subtracted.
        fine = dict(controls['integrator'])
        for name in ('relative_tolerance', 'position_absolute_tolerance_kpc',
                     'velocity_absolute_tolerance_kms', 'maximum_step_myr'):
            fine[name] /= controls['refinement_factor']
        refined,_=integrate_orbits(samples,potential,times,**fine);fine_actions=[]
        refined_first,_,_=recover_frequencies(times[:split],refined[:split,:,:3])
        refined_second,_,_=recover_frequencies(times[split:],refined[split:,:,:3])
        result['spectral_drift_numerical_difference_myr_inverse']=(refined_second-refined_first)-(second-first)
        for clock,state in zip(times,refined):
            sample={'position_kpc':state[:,:3],'velocity_kms':state[:,3:]}
            _, value = _instantaneous_actions(sample, potential, domain, controls,
                time_myr=clock, agama_potential=agama_potential)
            fine_actions.append(value['actions_kpc_kms'])
        result['numerical_action_difference_kpc_kms']=np.asarray(fine_actions)-actions
    return {'schema':'stream-action-orbit-history-v1','domain':domain,'action_invariance_claim':domain['exact_invariants_available'],
        'spectral_frequencies_are_canonical':False,'diffusion_interpretation':'time variation, nonintegrability, finite-duration spectral resolution and numerical drift retained separately'},result


def measure_action_distribution(actions,*,weights,reference=None):
    a=np.asarray(actions,float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError('Action distributions require N by 3 aligned actions')
    w=finite(weights,(len(a),))
    if np.any(w < 0):raise ValueError('Action weights cannot be negative')
    valid=np.isfinite(a).all(axis=1)
    mean,cov,neff=weighted_covariance(a[valid],w[valid])
    report={'mean_kpc_kms':mean,'covariance_kpc2_kms2':cov,'effective_count':neff,'unavailable_count':int((~valid).sum())}
    if reference is not None:
        r=np.asarray(reference,float)
        if r.shape!=a.shape:raise ValueError('Matched action perturbations require exact aligned identities')
        delta=a-r;good=np.isfinite(delta).all(axis=1)
        report['perturbation_mean_kpc_kms'],report['perturbation_covariance_kpc2_kms2'],_=weighted_covariance(delta[good],w[good])
    return report


def measure_actions(samples,potential,domain_record,controls,*,agama_potential=None):
    domain=classify_potential_domain(domain_record)
    x, v = _phase_coordinates(samples)
    if domain['action_method'] == 'orbit_diagnostics_only':
        n = len(x)
        report = {'method':'orbit diagnostics; no invariant actions claimed'}
        arrays = {'actions_kpc_kms':np.full((n,3),np.nan),
            'angles_rad':np.full((n,3),np.nan), 'frequencies_kms_kpc':np.full((n,3),np.nan),
            'action_supported':np.zeros(n,bool), 'angle_supported':np.zeros((n,3),bool)}
    else:
        report, arrays = _instantaneous_actions(samples, potential, domain, controls,
            time_myr=controls.get('epoch_time_myr'), agama_potential=agama_potential)
    if hasattr(potential,'analytical_actions') and potential.kind!='plummer':
        analytic=potential.analytical_actions(x,v)
        arrays['analytical_actions_kpc_kms']=analytic;arrays['analytical_action_difference_kpc_kms']=arrays['actions_kpc_kms']-analytic
    report.update({'schema':'stream-action-angle-v1','domain':domain,'action_order':['Jr','Jz','Jphi'],
        'action_normalization':'(2pi)^-1 integral p dq','angles_modulo':2*np.pi,
        'physical_acceptance':'requires independent orbit, analytic and timestep campaign measurements'})
    return report,arrays


def independent_action_comparison(samples,potential,controls,*,agama_potential=None):
    """Callable cross-engine job with dimensional floors and retained failures."""
    from core.halos import pinned_agama_backend
    agama,binding=pinned_agama_backend()
    if agama_potential is None:
        if not isinstance(potential,AnalyticSphericalPotential):raise ValueError('Explicit independent AGAMA potential required')
        if potential.kind=='harmonic':
            kwargs={'type':'Harmonic','Omega':potential.omega,'axisRatioY':1.,'axisRatioZ':1.}
        else:
            types={'kepler':'Plummer','isochrone':'Isochrone','plummer':'Plummer'}
            kwargs={'type':types[potential.kind],'mass':potential.mass,'scaleRadius':potential.scale}
        agama_potential=agama.Potential(**kwargs)
    first_report,first=spherical_actions_angles(samples,potential,controls['quadrature'])
    domain={'symmetry':'spherical','action_method':'spherical_quadrature','time_dependent':False}
    second_report,second=agama_adapter(samples,agama_potential,domain)
    valid=first['action_supported']&second['action_supported'];delta=first['actions_kpc_kms']-second['actions_kpc_kms']
    floor=positive(controls['action_floor_kpc_kms'],'action_floor_kpc_kms');relative=positive(controls['relative_tolerance'],'relative_tolerance')
    allowed=relative*np.maximum(abs(second['actions_kpc_kms']),floor)
    passed=valid&np.all(abs(delta)<=allowed,axis=1)
    freq_delta=first['frequencies_kms_kpc']-second['frequencies_kms_kpc']
    return {'schema':'stream-independent-actions-v1','status':'PASS' if valid.size and valid.all() and passed.all() else 'UNRESOLVED_OR_FAIL',
        'engines':[first_report,second_report],'backend':binding,'matched_unit_G_relative_difference':abs(float(agama.G)-G)/G,
        'missing_count':int((~valid).sum()),'failed_count':int((valid&~passed).sum()),
        'angle_origins':'backend conventions retained; compare derivatives/frequencies rather than subtract arbitrary origins'}, {
        'quadrature_actions_kpc_kms':first['actions_kpc_kms'],'agama_actions_kpc_kms':second['actions_kpc_kms'],
        'action_difference_kpc_kms':delta,'allowed_action_difference_kpc_kms':allowed,
        'frequency_difference_kms_kpc':freq_delta,'matched_supported':valid,'comparison_pass':passed}


class FrozenHostSphericalView:
    """Instantaneous radial view of an explicitly spherical evolved host.

    Uses proper host-center offsets and relative inertial velocities. This is
    an approximation for changing fields; the host motion is still included
    by the independent inertial orbit integrator.
    """
    def __init__(self,host,time_myr,escape_energy=0.):
        self.host=host;self.time=float(finite(time_myr, (), 'frozen epoch Myr'));self.escape_energy=escape_energy
        self.center,self.velocity,self.rotation=host.frame.state(time_myr)
    def radial_value(self,r):
        values=np.atleast_1d(r);query=self.center+np.column_stack((values,np.zeros_like(values),np.zeros_like(values)))@self.rotation.T
        phi=self.host.evaluate(query,self.time)[0]
        return phi if np.ndim(r) else float(phi[0])
    def local_sample(self,sample):
        return {'position_kpc':(np.asarray(sample['position_kpc'])-self.center)@self.rotation,
                'velocity_kms':(np.asarray(sample['velocity_kms'])-self.velocity)@self.rotation}
