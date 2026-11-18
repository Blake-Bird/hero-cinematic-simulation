"""Initial Conditions: consolidated scientific implementation."""
from __future__ import annotations

# ========================================================================
# ic_lpt: Independent, dealiased mixed-fluid 2LPT reference and background growth ODE.
from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.signal import resample


@dataclass(frozen=True)
class SecondOrderGrowth:
    scale_factor: float
    displacement_coefficient: float
    velocity_coefficient: float
    linear_growth_rate: float
    seed_scale_factor: float
    seed_kind: str
    ode_relative_tolerance: float


def radiation_growing_history(a, hubble, *, omega_m0, omega_r0, hubble0, rtol=1e-10):
    """Solve the regular pressureless growing mode in the supplied background.

    CLASS's background dust-growth initial condition need not be the regular
    Mészáros mode. It is not used as an implicit higher-order seed here. The
    actual H(a) remains unchanged, including any later smooth EDE contribution.
    This is a pressureless common-flow model, not a Boltzmann species transfer.
    """
    a=np.asarray(a,dtype=float);hubble=np.asarray(hubble,dtype=float)
    if (a.ndim!=1 or len(a)<6 or hubble.shape!=a.shape
            or not np.isfinite(a).all() or not np.isfinite(hubble).all()
            or np.any(a<=0) or np.any(hubble<=0) or np.any(np.diff(a)<=0)):
        raise ValueError('Growing-mode background requires positive ordered a,H samples')
    if not all(math.isfinite(v) and v>0 for v in (omega_m0,omega_r0,hubble0,rtol)) or rtol>1e-4:
        raise ValueError('Regular radiation growth requires positive matter/radiation and tolerance')
    y=omega_m0*a[0]/omega_r0
    if y>1e-4:
        raise ValueError('Background begins too late for the regular radiation seed')
    expected=hubble0*math.sqrt(omega_r0/a[0]**4+omega_m0/a[0]**3)
    if not math.isclose(hubble[0],expected,rel_tol=1e-3):
        raise ValueError('Early background is not radiation/matter dominated; provide an explicit growing seed')
    x=np.log(a);hs=CubicSpline(x,np.log(hubble),extrapolate=False)
    def rhs(at,state):
        hv=float(np.exp(hs(at)))
        om=omega_m0*hubble0**2/(math.exp(3*at)*hv*hv)
        return [state[1],-(2+float(hs(at,1)))*state[1]+1.5*om*state[0]]
    solution=solve_ivp(rhs,(x[0],x[-1]),(1+1.5*y,1.5*y),method='DOP853',
        dense_output=True,rtol=rtol,atol=rtol*1e-5,max_step=.05)
    if not solution.success:raise RuntimeError('Regular pressureless growing-mode integration failed')
    d,dx=solution.sol(x)
    if not np.isfinite(d).all() or not np.isfinite(dx).all() or np.any(d<=0) or np.any(dx<=0):
        raise ValueError('Background has no supported positive regular growing solution')
    return d/d[-1],dx/d


def second_order_growth(a, hubble, linear_growth, linear_growth_rate, *,
                        omega_m0, hubble0, target_a, seed_kind,
                        omega_r0=0., initial_second_order=None, rtol=1e-10):
    """Integrate E_xx+(2+H_x/H)E_x-3/2 Omega_m E=-3/2 Omega_m D².

    x=ln(a); H may use any consistent units. D and f are solver-supplied;
    normalization cancels in E/D². Seeds are explicit: an early radiation
    Mészáros series, a declared EdS limit, or caller-supplied (E,E_x). EDE
    callers must establish the seed regime or supply its growing solution.
    No LCDM H(a) or fixed -3/7 kernel is substituted for the supplied history.
    """
    a, hubble, d, f = [np.asarray(value,dtype=float) for value in (a,hubble,linear_growth,linear_growth_rate)]
    if (a.ndim != 1 or len(a) < 6 or any(v.shape != a.shape for v in (hubble,d,f))
            or any(not np.isfinite(v).all() for v in (a,hubble,d,f))
            or any(np.any(v <= 0) for v in (a,hubble,d,f)) or np.any(np.diff(a) <= 0)):
        raise ValueError('Growth history requires positive ordered a,H,D,f samples')
    if (not all(math.isfinite(v) and v > 0 for v in (omega_m0,hubble0,target_a,rtol))
            or not math.isfinite(omega_r0) or omega_r0 < 0 or not a[0] <= target_a <= a[-1]
            or rtol > 1e-4):
        raise ValueError('Invalid growth cosmology, epoch or ODE tolerance')
    x = np.log(a)
    hs = CubicSpline(x,np.log(hubble),extrapolate=False)
    ds = CubicSpline(x,np.log(d),extrapolate=False)
    fs = CubicSpline(x,f,extrapolate=False)
    # Reduce conditioning without changing any observable coefficient.
    dscale = float(np.exp(ds(math.log(target_a))))
    dseed = d[0]/dscale
    if seed_kind == 'radiation_regular':
        if omega_r0 <= 0:
            raise ValueError('Radiation seed requires positive radiation density')
        y = omega_m0*a[0]/omega_r0
        if y > 1e-4:
            raise ValueError('Background begins too late for the regular radiation seed')
        # Establish that extra components do not dominate at the seed and that
        # D,f agree with the pressureless regular radiation growing solution.
        expected_h = hubble0*math.sqrt(omega_r0/a[0]**4+omega_m0/a[0]**3)
        expected_f = 1.5*y/(1+1.5*y)
        if not math.isclose(hubble[0],expected_h,rel_tol=1e-3):
            raise ValueError('Supplied early background is not radiation/matter dominated; supply its exact seed')
        if not math.isclose(f[0],expected_f,rel_tol=.02,abs_tol=1e-9):
            raise ValueError('Linear seed is not the regular radiation solution; supply its exact seed')
        amplitude = dseed/(1+1.5*y)
        e0 = amplitude**2*(-1.5*y-1.125*y*y+.0625*y**3)
        ex0 = amplitude**2*(-1.5*y-2.25*y*y+.1875*y**3)
    elif seed_kind == 'eds':
        if omega_r0 != 0:
            raise ValueError('EdS seed cannot be used with radiation')
        matter_fraction = omega_m0*hubble0**2/(a[0]**3*hubble[0]**2)
        if not math.isclose(matter_fraction,1.,rel_tol=1e-4) or not math.isclose(f[0],1.,rel_tol=1e-4):
            raise ValueError('EdS seed requires a matter dominated growing-mode initial background')
        e0,ex0 = -3/7*dseed**2,-6/7*dseed**2
    elif seed_kind == 'explicit':
        seed = np.asarray(initial_second_order,dtype=float)
        if seed.shape != (2,) or not np.isfinite(seed).all():
            raise ValueError('Explicit seed requires finite (E,dE/dln a) in the supplied D normalization')
        e0,ex0 = seed/dscale**2
    else:
        raise ValueError('Declare radiation_regular, eds, or explicit higher-order growth seed')
    def rhs(at,state):
        hv = float(np.exp(hs(at)))
        om = omega_m0*hubble0**2/(math.exp(3*at)*hv*hv)
        dv = float(np.exp(ds(at)))/dscale
        return [state[1], -(2+float(hs(at,1)))*state[1]+1.5*om*(state[0]-dv*dv)]
    target_x = math.log(target_a)
    if target_x == x[0]:
        e,ex = e0,ex0
    else:
        solution = solve_ivp(rhs,(x[0],target_x),(e0,ex0),method='DOP853',rtol=rtol,atol=rtol*1e-5,
                             max_step=.05)
        if not solution.success or not np.isfinite(solution.y).all():
            raise RuntimeError('Background-dependent second-order growth ODE failed')
        e,ex = solution.y[:,-1]
    if e >= 0 or ex >= 0:
        raise ValueError('Higher-order growth solution is not in the supported growing branch')
    return SecondOrderGrowth(target_a,float(-e),float(-ex),float(fs(target_x)),
                             float(a[0]),seed_kind,rtol)


def evolve_twofluid_relative_modes(initial, request):
    """Finite-epoch second-order Euler/Poisson and Lagrangian characteristic expansion.

    u = h*v_pec/(a*H) is comoving Mpc/h per ln(a). Both species retain
    independent linear densities/velocities and all quadratic convective
    couplings. Second-order data are explicit finite-epoch seeds, not guessed
    fastest-growing modes. Pressure is either zero or a declared homogeneous
    isothermal baryon sound-speed history. Shell crossing is outside scope.
    """
    a=np.asarray(request['background_scale_factor'],float)
    hubble=np.asarray(request['background_hubble_km_s_mpc'],float)
    if a.ndim!=1 or len(a)<4 or hubble.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(hubble).all() or np.any(a<=0) or np.any(hubble<=0) or np.any(np.diff(a)<=0):
        raise ValueError('Positive ordered complete measured background required')
    start,end=float(request['reference_scale_factor']),float(request['target_scale_factor'])
    if not a[0]<=start<end<=a[-1]:raise ValueError('Finite reference and target epochs must lie in the actual background')
    fraction=float(request['baryon_fraction']);h=float(request['hubble_parameter']);box=float(request['box_mpc_h'])
    om0=float(request['omega_m0']);H0=float(request['hubble0_km_s_mpc'])
    if not np.isfinite([fraction,h,box,om0,H0]).all() or not 0<fraction<1 or min(h,box,om0,H0)<=0:
        raise ValueError('Explicit physical species fraction, box, H0 and background matter source required')
    density=np.asarray(initial['density_first_order'],float)
    if density.ndim!=4 or density.shape[0]!=2 or len(set(density.shape[1:]))!=1 or density.shape[1]<4:
        raise ValueError('Two species of aligned cubic first-order density fields required')
    n=density.shape[1];shape=density.shape[1:];cells=n**3
    if type(request['maximum_grid_cells'])is not int or cells>request['maximum_grid_cells']:
        raise MemoryError('Two-fluid reference grid exceeds its explicit bound')
    memory=float(request['memory_gib'])*1024**3
    if not math.isfinite(memory) or cells*16384>memory:
        raise MemoryError('Dealiased quadratic fields and full ODE workspace exceed allocation')
    state=np.zeros((2,14,*shape))
    for key,slots,expected in [('density_first_order',slice(0,1),(2,*shape)),
            ('velocity_first_order_km_s',slice(1,4),(2,3,*shape)),
            ('density_second_order',slice(4,5),(2,*shape)),
            ('velocity_second_order_km_s',slice(5,8),(2,3,*shape)),
            ('displacement_first_order_mpc_h',slice(8,11),(2,3,*shape)),
            ('displacement_second_order_mpc_h',slice(11,14),(2,3,*shape))]:
        value=np.asarray(initial[key],float)
        if value.shape!=expected or not np.isfinite(value).all():raise ValueError('Finite exact two-fluid seed dimensions required: '+key)
        if value.ndim==4:value=value[:,None]
        state[:,slots]=value
    logH=CubicSpline(np.log(a),np.log(hubble),extrapolate=False)
    for name,index in (('first-order',0),('second-order',4)):
        for species in range(2):
            value=state[species,index]
            if abs(float(value.mean()))>max(1e-13,float(np.max(np.abs(value)))*1e-11):
                raise ValueError('Two-fluid '+name+' density contains an undeclared homogeneous background shift')
    initial_H=float(np.exp(logH(math.log(start))))
    state[:,1:4]*=h/(start*initial_H);state[:,5:8]*=h/(start*initial_H)
    if request['pressure_model'] not in {'pressureless','homogeneous_isothermal_baryons'}:
        raise ValueError('Declared supported pressure model required')
    sound=np.asarray(request['baryon_sound_speed_km_s'],float)
    if sound.shape!=a.shape or not np.isfinite(sound).all() or np.any(sound<0) or request['pressure_model']=='pressureless' and np.any(sound):
        raise ValueError('Explicit sound-speed history consistent with the chosen pressure model required')
    sound2=CubicSpline(np.log(a),sound**2,extrapolate=False)
    kk=np.meshgrid(*([2*np.pi*np.fft.fftfreq(n,d=box/n)]*3),indexing='ij',sparse=True)
    k2=sum(k*k for k in kk);inverse=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
    def gradient(field,axis):return np.fft.ifftn(1j*kk[axis]*np.fft.fftn(field)).real
    def divergence(field):return sum(gradient(field[j],j) for j in range(3))
    def product(left,right):
        # Twice-padded products projected to the original Fourier grid.
        return _resize(_resize(left,2*n)*_resize(right,2*n),n)
    def convection(velocity,field):
        return np.asarray([sum(product(velocity[j],gradient(field[i],j)) for j in range(3)) for i in range(3)])
    # Material mass belongs to q, whereas supplied density is Eulerian. Pull
    # rho(x(q)) det(dx/dq) back consistently through second order. Assigning
    # 1+delta(q) alone would count the displacement density a second time.
    material_weights=[]
    for species in range(2):
        d1,d2=state[species,0],state[species,4];psi1,psi2=state[species,8:11],state[species,11:14]
        trace=divergence(psi1);trace2=divergence(psi2)
        trace_square=sum(product(gradient(psi1[i],j),gradient(psi1[j],i)) for i in range(3) for j in range(3))
        determinant2=trace2+.5*(product(trace,trace)-trace_square)
        pullback=sum(product(psi1[j],gradient(d1,j)) for j in range(3))
        material_weights.append(1+d1+trace+d2+pullback+product(d1,trace)+determinant2)
    material_weights=np.asarray(material_weights)
    if not np.isfinite(material_weights).all() or np.any(material_weights<=0):
        raise ValueError('Initial finite-epoch displacement/density map has no positive perturbative material mass')
    weights=(1-fraction,fraction)
    count=0
    def rhs(clock, flat):
        nonlocal count
        count+=1
        if count>request['maximum_rhs_evaluations']:raise RuntimeError('Two-fluid relative-mode source evaluation budget exhausted')
        y=flat.reshape(state.shape);out=np.zeros_like(y)
        aa=math.exp(clock);HH=float(np.exp(logH(clock)));drag=2+float(logH(clock,1))
        omega=om0*H0**2/(aa**3*HH**2)
        gravity=[]
        for index in (0,4):
            total=weights[0]*y[0,index]+weights[1]*y[1,index]
            ft=np.fft.fftn(total)
            gravity.append(np.asarray([1.5*omega*np.fft.ifftn(1j*k*inverse*ft).real for k in kk]))
        for species in range(2):
            d1,u1,d2,u2=y[species,0],y[species,1:4],y[species,4],y[species,5:8]
            out[species,0]=-divergence(u1)
            out[species,1:4]=-drag*u1+gravity[0]
            out[species,4]=-divergence(u2)-divergence(np.asarray([product(d1,u1[j]) for j in range(3)]))
            out[species,5:8]=-drag*u2+gravity[1]-convection(u1,u1)
            if species==1 and request['pressure_model']=='homogeneous_isothermal_baryons':
                pressure=float(sound2(clock))*h*h/(aa*aa*HH*HH)
                out[species,1:4]-=pressure*np.asarray([gradient(d1,j) for j in range(3)])
                out[species,5:8]-=pressure*np.asarray([gradient(d2,j)-product(d1,gradient(d1,j)) for j in range(3)])
            out[species,8:11]=u1
            out[species,11:14]=u2+convection(y[species,8:11],u1)
        return out.ravel()
    relative=float(request['ode_relative_tolerance']);absolute=float(request['ode_absolute_tolerance']);step=float(request['maximum_step_ln_a'])
    if not np.isfinite([relative,absolute,step]).all() or min(relative,absolute,step)<=0 or relative>=1:
        raise ValueError('Explicit positive finite integration tolerances and step bound required')
    if type(request['maximum_rhs_evaluations'])is not int or request['maximum_rhs_evaluations']<1:
        raise ValueError('Positive exact source-evaluation budget required')
    solution=solve_ivp(rhs,(math.log(start),math.log(end)),state.ravel(),method='DOP853',
        rtol=relative,atol=absolute,max_step=step,t_eval=[math.log(end)])
    if not solution.success or not np.isfinite(solution.y).all():raise RuntimeError('Finite-epoch two-fluid relative evolution failed')
    final=solution.y[:,-1].reshape(state.shape)
    density_final=final[:,0]+final[:,4]
    amplitude=float(request['maximum_density_perturbation'])
    if not 0<amplitude<1 or np.max(np.abs(density_final))>amplitude:
        raise ValueError('Evolved two-fluid density leaves the declared perturbative pre-shell-crossing domain')
    displacement=final[:,8:11]+final[:,11:14]
    displacement_limit=float(request['maximum_displacement_box_fraction'])
    if not math.isfinite(displacement_limit) or not 0<displacement_limit<.5:
        raise ValueError('Explicit perturbative displacement bound in (0,0.5) required')
    if np.max(np.linalg.norm(displacement,axis=1))/box>displacement_limit:
        raise ValueError('Two-fluid Lagrangian displacement exceeds its predeclared box fraction')
    velocity=[]
    final_H=float(np.exp(logH(math.log(end))))
    for species in range(2):
        velocity.append(end*final_H/h*(final[species,1:4]+final[species,5:8]+convection(final[species,8:11],final[species,1:4])))
    arrays={'density_first_order':final[:,0], 'density_second_order':final[:,4],
        'displacement_first_order_mpc_h':final[:,8:11], 'displacement_second_order_mpc_h':final[:,11:14],
        'velocity_peculiar_km_s':np.asarray(velocity),'relative_velocity_peculiar_km_s':velocity[1]-velocity[0],
        'velocity_first_order_peculiar_km_s':end*final_H/h*final[:,1:4],
        'lagrangian_mass_weights':material_weights}
    if np.any(arrays['lagrangian_mass_weights']<=0):raise ValueError('Explicit finite-epoch material density yields nonpositive mass weights')
    report={'schema':'finite-epoch-twofluid-relative-v1','species_order':['cdm','baryons'],
        'reference_scale_factor':start,'target_scale_factor':end,'request':request,'rhs_evaluations':count,
        'method':'second-order Euler/Poisson plus Lagrangian characteristic expansion; twice-padded spectral products',
        'seed_policy':'explicit finite-epoch first/second-order density, velocity and displacement fields',
        'background_policy':'actual supplied H(a), matter Poisson source and declared homogeneous baryonic pressure',
        'mass_policy':'second-order pullback of finite-epoch Eulerian density through the explicit initial Lagrangian map; conserved material weight',
        'domain':'weak perturbations before shell crossing; no third-order or general gas pressure evolution implied',
        'science_acceptance':'NOT_ESTABLISHED'}
    return report,arrays


def _periodic_cell_velocity_and_gradient(field,position,box):
    """Trilinear cell-centred velocity and its exact piecewise gradient."""
    field=np.asarray(field,float);position=np.asarray(position,float);n=field.shape[-1]
    if field.shape!=(3,n,n,n) or position.ndim!=2 or position.shape[1]!=3 or not np.isfinite(position).all():
        raise ValueError('Finite cell-centred vector field and material positions required')
    spacing=box/n;coordinate=np.mod(position,box)/spacing-.5
    low=np.floor(coordinate).astype(np.int64);fraction=coordinate-low
    value=np.zeros((len(position),3));gradient=np.zeros((len(position),3,3))
    for i in (0,1):
        for j in (0,1):
            for k in (0,1):
                bits=np.asarray([i,j,k]);at=(low+bits)%n
                factors=np.where(bits,fraction,1-fraction)
                samples=field[:,at[:,0],at[:,1],at[:,2]].T
                value+=samples*np.prod(factors,axis=1)[:,None]
                for axis in range(3):
                    other=[q for q in range(3) if q!=axis]
                    derivative=(1 if bits[axis] else -1)*np.prod(factors[:,other],axis=1)/spacing
                    gradient[:,:,axis]+=samples*derivative[:,None]
    return value,gradient


def evolve_twofluid_nonlinear_reference(initial,request):
    """Conservative finite-volume Euler/Poisson reference without PT truncation.

    rho_s=1+delta_s; u_s=h*v_s/(a*H); momentum=rho_s*u_s.
    The periodic Rusanov flux and SSPRK3 retain full advective products.
    This is a single-stream fluid reference, not collisionless phase-space
    evolution after shell crossing, a gas energy solver, or a native IC map.
    """
    keys={'schema','domain','background_scale_factor','background_hubble_km_s_mpc','baryon_sound_speed_km_s',
        'reference_scale_factor','target_scale_factor','hubble_parameter','box_mpc_h','baryon_fraction','omega_m0','hubble0_km_s_mpc',
        'pressure_model','maximum_grid_cells','maximum_steps','maximum_rhs_evaluations','maximum_step_rejections',
        'memory_gib','cfl','maximum_step_ln_a','minimum_density_ratio','maximum_density_ratio','mass_relative_tolerance',
        'material_chunk_rows','minimum_material_jacobian','minimum_material_singular_value','maximum_material_deformation_norm',
        'maximum_material_density_relative_error','material_map_maximum_subboxes','material_map_maximum_depth',
        'material_map_minimum_jacobian','require_global_material_injectivity'}
    if isinstance(request,dict) and 'rusanov_dissipation_multiplier' in request:keys.add('rusanov_dissipation_multiplier')
    if not isinstance(request,dict) or set(request)!=keys or request.get('schema')!='twofluid-nonlinear-reference-request-v1':
        raise ValueError('Explicit nonlinear single-stream reference request required')
    dissipation=request.get('rusanov_dissipation_multiplier',1.)
    if isinstance(dissipation,bool) or not isinstance(dissipation,(int,float)) or not math.isfinite(dissipation) or not 1<=dissipation<=16:
        raise ValueError('Rusanov dissipation multiplier must be a finite numerical value between 1 and 16; subcharacteristic fluxes are unsupported')
    scalar_keys={'reference_scale_factor','target_scale_factor','hubble_parameter','box_mpc_h','baryon_fraction','omega_m0','hubble0_km_s_mpc',
        'memory_gib','cfl','maximum_step_ln_a','minimum_density_ratio','maximum_density_ratio','mass_relative_tolerance',
        'minimum_material_jacobian','minimum_material_singular_value','maximum_material_deformation_norm','maximum_material_density_relative_error',
        'material_map_minimum_jacobian'}
    if any(isinstance(request[key],bool) or not isinstance(request[key],(int,float)) for key in scalar_keys):
        raise ValueError('Physical nonlinear scalars must be numerical values, not flags or numeric strings')
    if type(request['require_global_material_injectivity'])is not bool:raise ValueError('Declare the global material-map domain requirement explicitly')
    a=np.asarray(request['background_scale_factor'],float);H=np.asarray(request['background_hubble_km_s_mpc'],float)
    cs=np.asarray(request['baryon_sound_speed_km_s'],float)
    if (a.ndim!=1 or len(a)<4 or H.shape!=a.shape or cs.shape!=a.shape or not all(np.isfinite(v).all() for v in (a,H,cs))
            or np.any(a<=0) or np.any(np.diff(a)<=0) or np.any(H<=0) or np.any(cs<0)):
        raise ValueError('Complete ordered positive background and homogeneous sound-speed history required')
    start=float(request['reference_scale_factor']);end=float(request['target_scale_factor'])
    h=float(request['hubble_parameter']);box=float(request['box_mpc_h']);fraction=float(request['baryon_fraction'])
    om=float(request['omega_m0']);H0=float(request['hubble0_km_s_mpc'])
    if not np.isfinite([start,end,h,box,fraction,om,H0]).all() or not a[0]<=start<end<=a[-1] or min(h,box,om,H0)<=0 or not 0<fraction<1:
        raise ValueError('Finite covered epochs and explicit physical matter/box parameters required')
    if not math.isclose(H0,100*h,rel_tol=1e-12,abs_tol=0.):raise ValueError('Explicit nonlinear H0 and h describe different physical units')
    if request['pressure_model'] not in {'pressureless','homogeneous_isothermal_baryons'} or request['pressure_model']=='pressureless' and np.any(cs):
        raise ValueError('Declare the supported homogeneous pressure model consistently')
    if request['domain']!='single_stream_before_shell_crossing':raise ValueError('Nonlinear fluid reference excludes collisionless multistreaming')
    density=np.asarray(initial['density_contrast'],float);velocity=np.asarray(initial['velocity_peculiar_km_s'],float)
    if density.ndim!=4 or density.shape[0]!=2 or len(set(density.shape[1:]))!=1 or density.shape[1]<4 or velocity.shape!=(2,3,*density.shape[1:]):
        raise ValueError('Aligned full Eulerian cell-average species density and velocity grids required')
    n=density.shape[-1];cells=n**3
    if (type(request['material_map_maximum_subboxes'])is not int or request['material_map_maximum_subboxes']<2*cells
            or type(request['material_map_maximum_depth'])is not int or not 0<=request['material_map_maximum_depth']<=20
            or not 0<request['material_map_minimum_jacobian']<1):
        raise ValueError('Freeze complete-cell material-map work, dyadic depth and Jacobian threshold before integration')
    for key in ('maximum_grid_cells','maximum_steps','maximum_rhs_evaluations','maximum_step_rejections','material_chunk_rows'):
        if type(request[key])is not int or request[key]<1:raise ValueError('Positive exact nonlinear work bound required: '+key)
    memory=float(request['memory_gib'])*1024**3
    if not math.isfinite(memory) or memory<=0 or cells>request['maximum_grid_cells'] or cells*8192>memory:
        raise MemoryError('Nonlinear grid and finite-volume/FFT workspaces exceed allocation')
    jacobian_floor=float(request['minimum_material_jacobian']);singular_floor=float(request['minimum_material_singular_value'])
    deformation_limit=float(request['maximum_material_deformation_norm']);material_limit=float(request['maximum_material_density_relative_error'])
    if not np.isfinite([jacobian_floor,singular_floor,deformation_limit,material_limit]).all() or not 0<jacobian_floor<1 or not 0<singular_floor<1 or deformation_limit<=math.sqrt(3) or not 0<material_limit<1:
        raise ValueError('Explicit positive material map Jacobian, singular-value, deformation and density-consistency bounds required')
    cfl=float(request['cfl']);maxstep=float(request['maximum_step_ln_a']);floor=float(request['minimum_density_ratio'])
    ceiling=float(request['maximum_density_ratio']);mass_tolerance=float(request['mass_relative_tolerance'])
    if not np.isfinite([cfl,maxstep,floor,ceiling,mass_tolerance]).all() or not 0<cfl<=.4 or maxstep<=0 or not 0<floor<1<ceiling or mass_tolerance<=0:
        raise ValueError('Explicit finite stable CFL, time step, density support and mass budget required')
    if not np.isfinite(density).all() or not np.isfinite(velocity).all():raise ValueError('Finite full nonlinear seeds required')
    rho=1+density
    if np.any(rho<=floor) or np.any(rho>=ceiling) or np.any(np.abs(rho.mean(axis=(1,2,3))-1)>mass_tolerance):
        raise ValueError('Initial cell-average species density leaves its declared positive background domain')
    logH=CubicSpline(np.log(a),np.log(H),extrapolate=False);sound=CubicSpline(np.log(a),cs,extrapolate=False)
    clock=math.log(start);finish=math.log(end);dx=box/n
    state=np.concatenate((rho[:,None],rho[:,None]*velocity*h/(start*float(np.exp(logH(clock))))),axis=1)
    material_q=np.stack(np.meshgrid(*([(np.arange(n)+.5)*dx]*3),indexing='ij'),axis=-1).reshape((-1,3))
    material_position=np.broadcast_to(material_q,(2,cells,3)).copy()
    deformation=np.broadcast_to(np.eye(3),(2,cells,3,3)).copy()
    minimum_jacobian=1.;minimum_singular=1.;maximum_material_error=0.
    initial_mass=state[:,0].sum(axis=(1,2,3));initial_momentum=state[:,1:4].sum(axis=(2,3,4))
    wave=np.meshgrid(*([2*np.pi*np.fft.fftfreq(n,d=dx)]*3),indexing='ij',sparse=True)
    k2=sum(k*k for k in wave);inverse=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
    import hashlib
    step_history=hashlib.sha256()
    calls=0;steps=0;rejections=0;minimum=float(rho.min());maximum=float(rho.max());maximum_mass_error=0.
    minimum_realized_step=math.inf;maximum_realized_step=0.;realized_step_sum=0.
    diffusion_update=np.zeros_like(state);diffusion_absolute_update=np.zeros_like(state)
    def primitives(y,t):
        if not np.isfinite(y).all() or np.any(y[:,0]<=floor) or np.any(y[:,0]>=ceiling):
            raise FloatingPointError('Nonlinear intermediate state leaves the declared density support')
        aa=math.exp(t);hh=float(np.exp(logH(t)))
        speed=np.zeros(2);speed[1]=float(sound(t))*h/(aa*hh)
        if speed[1]<0 or not np.isfinite(speed).all():raise ValueError('Interpolated homogeneous pressure history leaves its domain')
        u=y[:,1:4]/y[:,0,None];drag=2+float(logH(t,1));omega=om*H0*H0/(aa**3*hh**2)
        total=(1-fraction)*(y[0,0]-1)+fraction*(y[1,0]-1)
        ft=np.fft.fftn(total)
        g=np.asarray([1.5*omega*np.fft.ifftn(1j*k*inverse*ft).real for k in wave])
        rate=dissipation*sum(float(np.max(np.abs(u[:,axis])+speed[:,None,None,None]))/dx for axis in range(3))+abs(drag)
        return u,speed,drag,g,rate
    def rhs(y,t):
        nonlocal calls
        calls+=1
        if calls>request['maximum_rhs_evaluations']:raise RuntimeError('Nonlinear reference RHS budget exhausted')
        u,speed,drag,g,rate=primitives(y,t);out=np.zeros_like(y);diffusive=np.zeros_like(y)
        for axis in range(3):
            # The same interface flux is removed from one cell and added to
            # its periodic neighbour, conserving each species' extensive mass.
            flux=y*u[:,axis,None];flux[:,0]=y[:,axis+1]
            flux[:,axis+1]+=y[:,0]*speed[:,None,None,None]**2
            shifted=np.roll(y,-1,axis=axis+2);next_flux=np.roll(flux,-1,axis=axis+2)
            characteristic=np.abs(u[:,axis])+speed[:,None,None,None]
            alpha=np.maximum(characteristic,np.roll(characteristic,-1,axis=axis+1))
            diffusive_face=-.5*dissipation*alpha[:,None]*(shifted-y)
            face=.5*(flux+next_flux)+diffusive_face
            out-=(face-np.roll(face,1,axis=axis+2))/dx
            diffusive-=(diffusive_face-np.roll(diffusive_face,1,axis=axis+2))/dx
        out[:,1:4]+=-drag*y[:,1:4]+y[:,0,None]*g[None]
        if not np.isfinite(out).all():raise FloatingPointError('Nonlinear finite-volume source overflows')
        return out,rate,diffusive
    def material_rhs(y,position,F):
        derivative_x=np.empty_like(position);derivative_F=np.empty_like(F)
        field=y[:,1:4]/y[:,0,None]
        for species in range(2):
            for begin in range(0,cells,request['material_chunk_rows']):
                sl=slice(begin,min(cells,begin+request['material_chunk_rows']))
                flow,gradient=_periodic_cell_velocity_and_gradient(field[species],position[species,sl],box)
                derivative_x[species,sl]=flow;derivative_F[species,sl]=np.einsum('nij,njk->nik',gradient,F[species,sl])
        return derivative_x,derivative_F
    def material_support(y,position,F):
        minimum_J=math.inf;minimum_S=math.inf;error=0.
        for species in range(2):
            for begin in range(0,cells,request['material_chunk_rows']):
                sl=slice(begin,min(cells,begin+request['material_chunk_rows']));matrix=F[species,sl]
                if not np.isfinite(matrix).all() or not np.isfinite(position[species,sl]).all():raise FloatingPointError('Material characteristic state overflows')
                J=np.linalg.det(matrix);singular=np.linalg.svd(matrix,compute_uv=False)[:,-1]
                if np.any(J<=jacobian_floor) or np.any(singular<=singular_floor) or np.any(np.linalg.norm(matrix,axis=(1,2))>=deformation_limit):
                    raise FloatingPointError('Material map leaves its declared sampled deformation domain')
                # Density reconstruction is measured against the conserved
                # finite-volume Eulerian field; numerical diffusion remains
                # visible rather than being folded into material mass weights.
                advected=_periodic_trilinear(y[species,0],position[species,sl]-.5*dx,box)
                residual=np.abs(advected*J/rho[species].ravel()[sl]-1)
                if not np.isfinite(residual).all():raise FloatingPointError('Material density consistency overflows')
                error=max(error,float(np.max(residual)));minimum_J=min(minimum_J,float(J.min()));minimum_S=min(minimum_S,float(singular.min()))
        return minimum_J,minimum_S,error
    while clock<finish:
        if steps>=request['maximum_steps']:raise RuntimeError('Nonlinear reference step budget exhausted')
        u,speed,drag,g,rate=primitives(state,clock)
        # Limit gravity-driven velocity changes as well as transport and Hubble
        # drag. Every later RK stage also rechecks its actual transport rate.
        gravity_rate=math.sqrt(float(np.max(np.abs(g)))/dx)
        dt=min(maxstep,cfl/max(rate+gravity_rate,np.finfo(float).tiny),finish-clock)
        while True:
            if dt<=0 or clock+dt==clock:raise RuntimeError('Nonlinear reference clock underflow')
            try:
                r0,q0,d0=rhs(state,clock);dx0,dF0=material_rhs(state,material_position,deformation)
                y1=state+dt*r0;x1=material_position+dt*dx0;F1=deformation+dt*dF0
                r1,q1,d1=rhs(y1,clock+dt);dx1,dF1=material_rhs(y1,x1,F1)
                y2=.75*state+.25*(y1+dt*r1);x2=.75*material_position+.25*(x1+dt*dx1);F2=.75*deformation+.25*(F1+dt*dF1)
                r2,q2,d2=rhs(y2,clock+.5*dt);dx2,dF2=material_rhs(y2,x2,F2)
                candidate=state/3+2*(y2+dt*r2)/3
                candidate_x=material_position/3+2*(x2+dt*dx2)/3;candidate_F=deformation/3+2*(F2+dt*dF2)/3
                primitives(candidate,clock+dt)
                stage_J1,stage_S1,stage_error1=material_support(y1,x1,F1)
                stage_J2,stage_S2,stage_error2=material_support(y2,x2,F2)
                current_J,current_S,current_error=material_support(candidate,candidate_x,candidate_F)
                current_J=min(current_J,stage_J1,stage_J2);current_S=min(current_S,stage_S1,stage_S2)
                current_error=max(current_error,stage_error1,stage_error2)
                if dt*max(q0,q1,q2)>cfl*(1+1e-12):raise FloatingPointError('Intermediate stage violates the declared CFL')
                break
            except FloatingPointError:
                rejections+=1
                if rejections>request['maximum_step_rejections']:raise RuntimeError('Nonlinear positivity/CFL rejection budget exhausted')
                dt*=.5
        if current_error>material_limit:raise RuntimeError('Eulerian/material density consistency exceeds the frozen budget; refine the source grid instead of altering material masses')
        # SSPRK3 accepted-step weights: rejected trial fluxes never enter this ledger.
        diffusion_update+=dt*(d0/6+d1/6+2*d2/3)
        diffusion_absolute_update+=dt*(np.abs(d0)/6+np.abs(d1)/6+2*np.abs(d2)/3)
        if not np.isfinite(diffusion_update).all() or not np.isfinite(diffusion_absolute_update).all():raise ArithmeticError('Reference numerical diffusion ledger overflows')
        state=candidate;material_position=candidate_x;deformation=candidate_F;clock=min(finish,clock+dt);steps+=1
        step_history.update((float(clock).hex()+':'+float(dt).hex()+'\n').encode('ascii'))
        minimum_realized_step=min(minimum_realized_step,dt);maximum_realized_step=max(maximum_realized_step,dt);realized_step_sum+=dt
        minimum_jacobian=min(minimum_jacobian,current_J);minimum_singular=min(minimum_singular,current_S);maximum_material_error=max(maximum_material_error,current_error)
        error=float(np.max(np.abs(state[:,0].sum(axis=(1,2,3))/initial_mass-1)))
        maximum_mass_error=max(maximum_mass_error,error)
        if error>mass_tolerance:raise RuntimeError('Conservative nonlinear species mass exceeds its frozen roundoff budget')
        minimum=min(minimum,float(state[:,0].min()));maximum=max(maximum,float(state[:,0].max()))
    velocity_final=state[:,1:4]/state[:,0,None]*end*float(np.exp(logH(finish)))/h
    arrays={'density_contrast':state[:,0]-1,'velocity_peculiar_km_s':velocity_final,
        'relative_velocity_peculiar_km_s':velocity_final[1]-velocity_final[0],
        'numerical_diffusion_density_update':diffusion_update[:,0],
        'numerical_diffusion_momentum_update':diffusion_update[:,1:4],
        'numerical_diffusion_density_absolute_update':diffusion_absolute_update[:,0],
        'numerical_diffusion_momentum_absolute_update':diffusion_absolute_update[:,1:4],
        'initial_species_grid_mass':initial_mass,'final_species_grid_mass':state[:,0].sum(axis=(1,2,3)),
        'initial_species_grid_momentum':initial_momentum,'final_species_grid_momentum':state[:,1:4].sum(axis=(2,3,4))}
    material_velocity=np.empty((2,cells,3))
    for species in range(2):
        for begin in range(0,cells,request['material_chunk_rows']):
            sl=slice(begin,min(cells,begin+request['material_chunk_rows']))
            material_velocity[species,sl]=_periodic_cell_velocity_and_gradient(velocity_final[species],material_position[species,sl],box)[0]
    arrays.update(material_displacement_mpc_h=np.moveaxis((material_position-material_q).reshape((2,n,n,n,3)),-1,1),
        material_velocity_peculiar_km_s=np.moveaxis(material_velocity.reshape((2,n,n,n,3)),-1,1),
        lagrangian_mass_weights=rho.copy(),material_deformation_gradient=np.moveaxis(deformation.reshape((2,n,n,n,3,3)),(-2,-1),(1,2)),
        material_jacobian=np.linalg.det(deformation).reshape((2,n,n,n)))
    certificate,map_arrays=certify_trilinear_material_map(arrays['material_displacement_mpc_h'],box,
        minimum_jacobian=request['material_map_minimum_jacobian'],maximum_subboxes=request['material_map_maximum_subboxes'],
        maximum_depth=request['material_map_maximum_depth'])
    if request['require_global_material_injectivity'] and certificate['global_injectivity_status']!='CONTRACTION_BOUND_SUPPORTED':
        raise ValueError('Full continuous material map does not meet the explicitly required global injectivity bound')
    arrays.update(map_arrays)
    report={'schema':'finite-epoch-nonlinear-twofluid-reference-v1','request':request,'species_order':['cdm','baryons'],
        'steps':steps,'rhs_evaluations':calls,'rejected_steps':rejections,'minimum_density_ratio':minimum,'maximum_density_ratio':maximum,
        'minimum_realized_step_ln_a':minimum_realized_step,'maximum_realized_step_ln_a':maximum_realized_step,
        'realized_step_sum_ln_a':realized_step_sum,'accepted_step_history_sha256':step_history.hexdigest(),
        'numerical_flux':{'method':'local Lax-Friedrichs / Rusanov','dissipation_multiplier':float(dissipation),
            'diagnostic':'accepted SSPRK3 stage-weighted divergence of the dissipative interface flux only; signed and cumulative absolute cell updates',
            'stage_weights':[1/6,1/6,2/3],'accepted_rhs_contributions':3*steps,
            'signed_density_update_sum':diffusion_update[:,0].sum(axis=(1,2,3)).tolist(),
            'signed_momentum_update_sum':diffusion_update[:,1:4].sum(axis=(2,3,4)).tolist(),
            'scope':'Exact recorded discrete flux contribution, not continuum error, physical viscosity, energy dissipation or independent convergence acceptance'},
        'maximum_species_mass_relative_error':maximum_mass_error,'method':'periodic first-order Rusanov finite volume, SSPRK3, spectral Poisson; full conservative advective fluxes',
        'reference_scale_factor':start,'target_scale_factor':end,'material_grid_origin_mpc_h':[.5*dx]*3,
        'material_method':'same-stage SSPRK3 characteristics and exact piecewise-trilinear velocity-gradient variational flow; initial Eulerian cell centres define finite-epoch material labels',
        'minimum_sampled_material_jacobian':minimum_jacobian,'minimum_sampled_material_singular_value':minimum_singular,
        'maximum_material_density_relative_error':maximum_material_error,
        'interpolated_material_map_domain':certificate,
        'pressure':'declared homogeneous isothermal baryons or pressureless species',
        'limitations':['numerical diffusion requires spatial refinement','single-stream domain needs independent characteristic/shell-crossing evidence',
            'no DM multistream or gas energy evolution','Eulerian grids do not constitute a Lagrangian/native IC handoff'],
        'science_acceptance':'NOT_ESTABLISHED'}
    return report,arrays


def validate_twofluid_eulerian_seed_request(parameters):
    request=parameters['request']
    keys={'schema','snapshot','particle_namespace','grid_size','maximum_particles','memory_gib','chunk_rows',
        'minimum_particles_per_species_cell','maximum_cell_velocity_dispersion_km_s','mass_relative_tolerance'}
    if isinstance(request,dict) and 'measure_pressure_support' in request:keys.add('measure_pressure_support')
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='twofluid-native-eulerian-seed-request-v1':
        raise ValueError('Complete original native Eulerian cell-average measurement request required')
    if 'measure_pressure_support' in request and type(request['measure_pressure_support'])is not bool:raise ValueError('Explicit boolean native pressure measurement selection required')
    for key in ('snapshot','grid_size','maximum_particles','chunk_rows','minimum_particles_per_species_cell'):
        if type(request[key])is not int or request[key]<(0 if key=='snapshot' else 1):raise ValueError('Exact bounded Eulerian measurement integer required: '+key)
    if request['grid_size']<4 or not isinstance(request['particle_namespace'],str) or not request['particle_namespace']:
        raise ValueError('Cubic grid of at least four cells and explicit particle namespace required')
    for key in ('memory_gib','maximum_cell_velocity_dispersion_km_s','mass_relative_tolerance'):
        value=request[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('Explicit positive Eulerian measurement budget required: '+key)
    if request['grid_size']**3*1024+request['maximum_particles']*800>request['memory_gib']*1024**3:
        raise MemoryError('Declared original census and cell moments exceed allocation')
    return request


def twofluid_eulerian_seed_input_bindings(parameters):
    from .sidm_analysis import _stage_native_bindings
    validate_twofluid_eulerian_seed_request(parameters)
    sources=_stage_native_bindings(parameters)
    expected=parameters.get('expected_native_source_hashes')
    if expected is not None and (not isinstance(expected,dict) or not expected or any(sources.get(path)!=checksum for path,checksum in expected.items())):
        raise ValueError('Two-fluid seed native inputs changed after campaign design')
    return sources


def _twofluid_native_pressure_ratio(files,particles,request,sources):
    """Read single-phase P/rho from actual internal energy and caloric EOS."""
    import h5py
    from .data import exact_uint64_ids,dataset_to_cgs_factor
    from .foundation import file_sha256
    gas=np.flatnonzero(particles['species']==0);ids=exact_uint64_ids(particles['particle_ids'][gas]);order=np.argsort(ids);ordered=ids[order]
    ratio=np.zeros(len(particles['particle_ids']));available=np.zeros(len(ratio),bool);seen=np.zeros(len(ids),bool);pressure_model=None
    thermal=np.zeros(len(ratio));thermal_available=np.zeros(len(ratio),bool)
    pressure_work=np.zeros(len(ratio));pressure_work_available=np.zeros(len(ratio),bool)
    for path in files:
        with h5py.File(path,'r') as native:
            header=native['Header'].attrs;p=native['Parameters'].attrs;config=native['Config'].attrs;n=int(header['NumPart_ThisFile'][0])
            if not n:continue
            group=native['PartType0'];a=float(header['Time']);h=float(p['HubbleParam']);isothermal='ISOTHERM_EQS' in config
            gamma=1. if isothermal else float(config.get('GAMMA',5/3))
            if not math.isfinite(gamma) or not (gamma==1 if isothermal else 1<gamma<=2):raise ValueError('Saved gas caloric index is unsupported')
            current={'eos':'native_isothermal' if isothermal else 'native_caloric',
                'adiabatic_index':gamma,'pressure_ratio_mapping':'stored_u' if isothermal else '(gamma_pressure-1)*stored_u',
                'source_flags':sorted(name for name in ('COOLING','STARFORMATION','RESEARCH_GRACKLE','GRACKLE','STELLAR_EVOLUTION','BLACK_HOLES') if name in config),
                'closure_scope':'sampled P/rho only; pressure work, energy/source and SPH-force closure require separate evidence'}
            if pressure_model is not None and current!=pressure_model:raise ValueError('Native pressure model changes between shards')
            pressure_model=current
            complete='InternalEnergy' in group
            if isothermal and current['source_flags']:complete=False
            divergence_available='VelocityDivergence' in group
            if divergence_available:
                field=group['VelocityDivergence']
                if field.shape!=(n,):raise ValueError('Native pressure-work divergence census differs')
                expected={'a_scaling':-2.,'h_scaling':1.,'to_cgs':float(p['UnitVelocity_in_cm_per_s'])/float(p['UnitLength_in_cm'])}
                if any(float(field.attrs.get(key,np.nan))!=value for key,value in expected.items()):raise ValueError('Native pressure-work divergence unit convention differs from pinned snapshot export')
            if 'STARFORMATION' in config:complete &= 'StarFormationRate' in group
            chemistry='RESEARCH_GRACKLE' in config or 'GRACKLE' in config
            if chemistry:complete &= all(name in group for name in ('GrackleStateVersion','GrackleGammaPressure'))
            for name in ('InternalEnergy','StarFormationRate','GrackleStateVersion','GrackleGammaPressure'):
                if name in group and group[name].shape!=(n,):raise ValueError('Native pressure field row census differs: '+name)
            for begin in range(0,n,request['chunk_rows']):
                end=min(n,begin+request['chunk_rows']);current=exact_uint64_ids(group['ParticleIDs'][begin:end],allow_empty=False)
                at=np.searchsorted(ordered,current)
                if np.any(at>=len(ids)) or not np.array_equal(ordered[at],current) or np.any(seen[at]):raise ValueError('Native pressure rows change or duplicate gas identities')
                seen[at]=True;slots=gas[order[at]]
                if not complete:continue
                energy=np.asarray(group['InternalEnergy'][begin:end],float)*dataset_to_cgs_factor(group['InternalEnergy'].attrs,a,h)/1e10
                if not np.isfinite(energy).all() or np.any(energy<0):raise ValueError('Native gas internal energy is nonfinite/negative')
                index=np.full(len(current),gamma);valid=np.ones(len(current),bool)
                if 'STARFORMATION' in config:
                    sfr=np.asarray(group['StarFormationRate'][begin:end],float)
                    if not np.isfinite(sfr).all() or np.any(sfr<0):raise ValueError('Native single-phase/EOS selection has invalid SFR')
                    valid &= sfr==0
                if chemistry:
                    state=exact_uint64_ids(group['GrackleStateVersion'][begin:end],unique=False)
                    index=np.asarray(group['GrackleGammaPressure'][begin:end],float)
                    if np.any(state>4) or not np.isfinite(index).all() or np.any((index[state>0]<=1)|(index[state>0]>gamma+1e-6)):
                        raise ValueError('Native gas pressure caloric state is invalid')
                    attrs=group['GrackleGammaPressure'].attrs
                    if any(float(attrs.get(key,np.nan))!=value for key,value in {'to_cgs':1.,'a_scaling':0.,'h_scaling':0.}.items()):raise ValueError('Native pressure caloric index units differ')
                    valid &= state>0
                value=energy if isothermal else (index-1)*energy
                if not np.isfinite(value[valid]).all() or np.any(value[valid]<0):raise ValueError('Native pressure ratio overflows')
                ratio[slots]=np.where(valid,value,0);available[slots]=valid
                if not isothermal:
                    thermal[slots]=np.where(valid,energy,0);thermal_available[slots]=valid
                if divergence_available:
                    field=group['VelocityDivergence'];divergence=np.asarray(field[begin:end],float)*dataset_to_cgs_factor(field.attrs,a,h)
                    if not np.isfinite(divergence).all():raise ValueError('Native peculiar divergence is nonfinite')
                    work=-value*1e10*divergence
                    if not np.isfinite(work).all():raise ArithmeticError('Native pressure-work specific rate overflows')
                    pressure_work[slots]=np.where(valid,work,0);pressure_work_available[slots]=valid
        if sources.get(str(path.resolve()))!=file_sha256(path):raise ValueError('Native pressure snapshot changed during intake')
    if not seen.all():raise ValueError('Native pressure shard set omits original gas identities')
    return ratio,available,pressure_model,thermal,thermal_available,pressure_work,pressure_work_available


def _twofluid_native_hydro_acceleration(files,particles,request,sources):
    """Read stored native SPH acceleration, retaining unknown force epochs."""
    import h5py
    from pathlib import Path
    from .data import exact_uint64_ids,dataset_to_cgs_factor
    from .foundation import file_sha256
    gas=np.flatnonzero(particles['species']==0);ids=exact_uint64_ids(particles['particle_ids'][gas]);order=np.argsort(ids);ordered=ids[order]
    values=np.zeros((len(particles['particle_ids']),3));available=np.zeros(len(values),bool);seen=np.zeros(len(ids),bool);notes=[]
    epoch_matches=np.zeros(len(values),bool);epoch_known=np.zeros(len(values),bool)
    for path in files:
        with h5py.File(path,'r') as saved:
            header=saved['Header'].attrs;params=saved['Parameters'].attrs;flags=saved['Config'].attrs;n=int(header['NumPart_ThisFile'][0])
            if not n:continue
            group=saved['PartType0'];field=group.get('HydroAcceleration');epoch=float(header['Time']);h=float(params['HubbleParam'])
            gamma=1. if 'ISOTHERM_EQS' in flags else float(flags.get('GAMMA',5/3))
            if not math.isfinite(gamma) or not 1<=gamma<=2:raise ValueError('Native hydro acceleration integration index is invalid')
            usable=field is not None
            if usable:
                if field.shape!=(n,3):raise ValueError('Original native hydro acceleration row/vector census differs')
                expected={'a_scaling':2.-3.*gamma,'h_scaling':1.,'to_cgs':float(params['UnitVelocity_in_cm_per_s'])**2/float(params['UnitLength_in_cm'])}
                usable=all(float(field.attrs.get(key,np.nan))==value for key,value in expected.items())
            if not usable:notes.append({'file':str(Path(path).resolve()),'state':'FIELD_MISSING' if field is None else 'PHYSICAL_UNIT_CONVENTION_UNAVAILABLE'})
            epoch_fields={'HydroForceEvaluationNativeTime','HydroForceEpochVersion'}
            has_epoch=epoch_fields<=set(group)
            if epoch_fields & set(group) and not has_epoch:raise ValueError('Native hydro force epoch fields are incomplete')
            if has_epoch:
                for name in epoch_fields:
                    value=group[name]
                    expected_kind='u' if name=='HydroForceEpochVersion' else 'f'
                    if value.dtype.kind!=expected_kind or value.dtype.itemsize!=8:raise ValueError('Native hydro force epoch requires exact uint64 version and float64 clock storage')
                    if value.shape!=(n,) or any(float(value.attrs.get(key,np.nan))!=expected for key,expected in {'to_cgs':1.,'a_scaling':0.,'h_scaling':0.}.items()):raise ValueError('Native hydro force epoch shape or native-clock units differ')
            for begin in range(0,n,request['chunk_rows']):
                end=min(n,begin+request['chunk_rows']);current=exact_uint64_ids(group['ParticleIDs'][begin:end],allow_empty=False)
                at=np.searchsorted(ordered,current)
                if np.any(at>=len(ids)) or not np.array_equal(ordered[at],current) or np.any(seen[at]):raise ValueError('Native hydro acceleration changes or duplicates gas identities')
                seen[at]=True;slots=gas[order[at]]
                if has_epoch:
                    version=exact_uint64_ids(group['HydroForceEpochVersion'][begin:end],unique=False)
                    clock=np.asarray(group['HydroForceEvaluationNativeTime'][begin:end],float)
                    if np.any(version>1) or not np.isfinite(clock).all() or np.any(clock<0) or np.any(clock[version==1]>epoch):raise ValueError('Native hydro force epoch is unsupported, future or invalid')
                    if np.any((version==0)&(clock!=0)):raise ValueError('Invalidated native hydro force epoch retains a nonzero clock')
                    epoch_known[slots]=version==1;epoch_matches[slots]=(version==1)&(clock==epoch)&usable
                if not usable:continue
                acceleration=np.asarray(field[begin:end],float)*dataset_to_cgs_factor(field.attrs,epoch,h)
                if not np.isfinite(acceleration).all():raise ValueError('Stored native hydro acceleration is nonfinite')
                values[slots]=acceleration;available[slots]=True
        if sources.get(str(Path(path).resolve()))!=file_sha256(path):raise ValueError('Original hydro acceleration snapshot changed during read')
    if not seen.all():raise ValueError('Original native hydro acceleration shard set omits gas identities')
    return values,available,{'frame_status':'EVALUATION_CLOCK_RECORDED_STATE_FRAME_UNVERIFIED' if epoch_known[gas].any() else 'EVALUATION_EPOCH_NOT_RECORDED','unit_failures':notes,
        'evaluation_epoch_known':epoch_known,'evaluation_epoch_matches_snapshot':epoch_matches,
        'scope':'Actual stored native gas SPH acceleration including viscous terms, converted using pinned integration scaling. Availability certifies stored field units/IDs, not a current-epoch force evaluation or pure pressure-force equivalence.'}


def _twofluid_native_expansion_model(parameters,root,sources,*,memory_bytes):
    """Reconstruct the recorded native background route, without H0 defaults."""
    from pathlib import Path
    from .foundation import file_sha256
    keys=('ComovingIntegrationOn','Hubble','HubbleParam','UnitLength_in_cm','UnitVelocity_in_cm_per_s','Omega0','OmegaLambda','OmegaRadiation','CosmologyTableFile')
    missing=[key for key in keys if key not in parameters]
    if missing:return {'state':'MISSING_NATIVE_PARAMETERS','missing':missing}
    if float(parameters['ComovingIntegrationOn'])!=1:raise ValueError('Native expansion reconstruction requires cosmological integration')
    values={key:float(parameters[key]) for key in keys[:-1]}
    if not np.isfinite(list(values.values())).all() or min(values[key] for key in ('Hubble','HubbleParam','UnitLength_in_cm','UnitVelocity_in_cm_per_s'))<=0 or values['Omega0']<0 or values['OmegaRadiation']<0:
        raise ValueError('Native expansion parameters are invalid')
    result={'state':'RECORDED_NATIVE_DEFINITION','hubble0_s_inverse':values['Hubble']*values['UnitVelocity_in_cm_per_s']/values['UnitLength_in_cm']*values['HubbleParam'],
        'omega_m0':values['Omega0'],'omega_lambda':values['OmegaLambda'],'omega_radiation':values['OmegaRadiation'],
        'omega_curvature':1-(values['Omega0']+values['OmegaLambda']+values['OmegaRadiation'])}
    table=parameters['CosmologyTableFile']
    if isinstance(table,bytes):table=table.decode()
    if not isinstance(table,str):raise ValueError('Native expansion table parameter must be an original path string')
    if table in ('','none'):
        result['route']='analytic_native_friedmann'
    else:
        path=Path(table);path=(path if path.is_absolute() else Path(root)/'input'/path).resolve()
        if path.stat().st_size*8>memory_bytes:raise MemoryError('Native expansion table exceeds intake allocation')
        checksum=file_sha256(path)
        if sources.get(str(path))!=checksum:raise ValueError('Native expansion table is not bound to the original completed generation')
        data=np.loadtxt(path,ndmin=2)
        if data.ndim!=2 or data.shape[1]!=4 or len(data)<2 or not np.isfinite(data).all() or np.any(data[:,0]<=0) or np.any(np.diff(data[:,0])<=0) or np.any(data[:,1:]<=0):raise ValueError('Native expansion table has an unsupported a/E/D/f definition')
        if file_sha256(path)!=checksum:raise ValueError('Native expansion table changed during read')
        result.update(route='native_log_linear_E_table',table={'path':str(path),'sha256':checksum},
            table_scale_factors=data[:,0].tolist(),table_E=data[:,1].tolist())
    return result


def _twofluid_native_hubble_s_inverse(model,epoch):
    if model.get('state')!='RECORDED_NATIVE_DEFINITION':return None
    if type(epoch)not in (int,float) or not math.isfinite(epoch) or epoch<=0:raise ValueError('Native expansion epoch must be finite positive')
    if model['route']=='analytic_native_friedmann':
        squared=model['omega_m0']/epoch**3+model['omega_curvature']/epoch**2+model['omega_radiation']/epoch**4+model['omega_lambda']
        if not math.isfinite(squared) or squared<=0:raise ValueError('Native expansion rate has no positive support')
        E=math.sqrt(squared)
    elif model['route']=='native_log_linear_E_table':
        a=np.asarray(model['table_scale_factors'],float);values=np.asarray(model['table_E'],float)
        if not a[0]<=epoch<=a[-1]:raise ValueError('Native expansion epoch is outside the original table')
        E=float(np.exp(np.interp(math.log(epoch),np.log(a),np.log(values))))
    else:raise ValueError('Unsupported recorded native expansion route')
    rate=model['hubble0_s_inverse']*E
    if not math.isfinite(rate) or rate<=0:raise ValueError('Native expansion rate is invalid')
    return rate


def publish_twofluid_eulerian_seed(parameters,destination):
    """Measure full-box mass/momentum cell averages from original gas/DM data.

    NGP is exact membership in disjoint Eulerian cells; no empty cell is filled
    by a synthetic background. Resolved within-cell dispersion stays measured.
    """
    import h5py
    from pathlib import Path
    from .observation_geometry import read_native_gravitating_particles
    from .halos import _snapshot_files
    from .data import snapshot_total_counts,export_numeric_audit
    from .foundation import file_sha256,KPC_CM,MSUN_G,G_CGS,H100_S
    request=validate_twofluid_eulerian_seed_request(parameters);sources=twofluid_eulerian_seed_input_bindings(parameters)
    root=Path(parameters['native_run_directory']).resolve();files=_snapshot_files(root,request['snapshot']);reference=None;expansion_model=None
    if not files:raise FileNotFoundError('Original Eulerian seed snapshot set is missing')
    configuration=root/'input/Config.sh'
    if sources.get(str(configuration))!=file_sha256(configuration):raise ValueError('Eulerian seed requires the original native compile configuration')
    flags={line.split('#',1)[0].strip().split('=',1)[0] for line in configuration.read_text().splitlines()}
    if 'PERIODIC' not in flags:raise ValueError('Full-box periodic Eulerian seed requires a periodic native source')
    for path in files:
        if sources.get(str(path.resolve()))!=file_sha256(path):raise ValueError('Eulerian seed snapshot is outside the original completed native generation')
        with h5py.File(path,'r') as native:
            header=native['Header'].attrs;params=native['Parameters'].attrs;counts=snapshot_total_counts(header)
            current_expansion=_twofluid_native_expansion_model(params,root,sources,memory_bytes=request['memory_gib']*1024**3/16)
            if expansion_model is not None and current_expansion!=expansion_model:raise ValueError('Original native expansion model differs between shards')
            expansion_model=current_expansion
            if any(int(counts[k]) for k in (2,3,4,5)):raise ValueError('Two-fluid reference seed requires a complete pure gas/DM census')
            current=(float(header['Time']),float(params['HubbleParam']),float(params['BoxSize']),float(params['UnitLength_in_cm']),
                float(params['Omega0']),float(params['OmegaBaryon']),int(params['ComovingIntegrationOn']))
            if reference is not None and current!=reference:raise ValueError('Original Eulerian seed shards change cosmology, box or epoch')
            reference=current
    a,h,code_box,L,omega,baryon,comoving=reference
    if not np.isfinite(reference).all() or min(a,h,code_box,L,omega,baryon)<=0 or baryon>=omega or comoving!=1:
        raise ValueError('Original complete cosmological gas/DM box required')
    box=code_box*L/(KPC_CM*1000);fraction=baryon/omega
    population,particles=read_native_gravitating_particles(root,request['snapshot'],species=[1,0],maximum_particles=request['maximum_particles'],
        memory_gib=request['memory_gib']/4,namespace=request['particle_namespace'])
    sources.update(population['source_hashes']);n=request['grid_size'];cells=n**3
    mass=np.zeros((2,cells));momentum=np.zeros((2,3,cells));second=np.zeros((2,cells));counts=np.zeros((2,cells),dtype=np.uint64)
    selected=particles['species'];x=particles['position_comoving_mpc']*h;v=particles['velocity_peculiar_kms'];m=particles['mass_msun']
    for start in range(0,len(m),request['chunk_rows']):
        stop=min(len(m),start+request['chunk_rows']);sl=slice(start,stop)
        index=np.floor(np.mod(x[sl],box)*n/box).astype(np.int64)%n
        cell=(index[:,0]*n+index[:,1])*n+index[:,2]
        for species,kind in enumerate((1,0)):
            keep=selected[sl]==kind;at=cell[keep];weights=m[sl][keep];velocity=v[sl][keep]
            np.add.at(mass[species],at,weights);np.add.at(counts[species],at,np.uint64(1))
            for axis in range(3):np.add.at(momentum[species,axis],at,weights*velocity[:,axis])
            np.add.at(second[species],at,weights*np.sum(velocity*velocity,axis=1))
    if np.any(counts<request['minimum_particles_per_species_cell']) or np.any(mass<=0):
        raise ValueError('Actual native census cannot support every declared Eulerian species cell; refine the source census or coarsen the measurement grid')
    if not all(np.isfinite(value).all() for value in (mass,momentum,second)):
        raise ValueError('Eulerian extensive moment accumulation overflows')
    physical_box_cm=box/h*KPC_CM*1000
    expected_total=3*(H100_S*h)**2/(8*math.pi*G_CGS)*omega*physical_box_cm**3/MSUN_G
    expected=np.asarray([1-fraction,fraction])*expected_total
    measured_mass=mass.sum(axis=1)
    if np.any(np.abs(measured_mass/expected-1)>request['mass_relative_tolerance']):
        raise ValueError('Native gas/DM mass does not represent the complete declared cosmological box')
    mean=momentum/mass[:,None];raw_variance=second/mass-np.sum(mean*mean,axis=1)
    roundoff=128*np.finfo(float).eps*np.maximum(second/mass,np.sum(mean*mean,axis=1))
    if np.any(raw_variance < -roundoff):raise ValueError('Measured Eulerian velocity second moments lose numerical support')
    dispersion=np.sqrt(np.maximum(raw_variance,0.))
    dispersion_upper=np.sqrt(np.maximum(raw_variance,0.)+roundoff)
    if not np.isfinite(dispersion_upper).all() or np.any(dispersion_upper>request['maximum_cell_velocity_dispersion_km_s']):
        raise ValueError('Measured species velocity dispersion exceeds the declared single-stream reference seed budget')
    shape=(n,n,n)
    arrays={'density_contrast':(mass/(expected[:,None]/cells)-1).reshape((2,*shape)),
        'velocity_peculiar_km_s':mean.reshape((2,3,*shape)),'particle_count_per_cell':counts.reshape((2,*shape)),
        'velocity_dispersion_km_s':dispersion.reshape((2,*shape)),
        'velocity_dispersion_roundoff_upper_km_s':dispersion_upper.reshape((2,*shape)),'actual_species_mass_msun':measured_mass}
    if request.get('measure_pressure_support',False):
        pressure,valid,pressure_model,thermal,thermal_valid,pressure_work,work_valid=_twofluid_native_pressure_ratio(files,particles,request,sources)
        hydro_acceleration,hydro_available,hydro_frame=_twofluid_native_hydro_acceleration(files,particles,request,sources)
        force_epoch_matches=hydro_frame.pop('evaluation_epoch_matches_snapshot');force_epoch_known=hydro_frame.pop('evaluation_epoch_known')
        thermal_mass=np.zeros(cells);first=np.zeros(cells)
        energy_mass=np.zeros(cells);energy_sum=np.zeros(cells)
        work_mass=np.zeros(cells);work_sum=np.zeros(cells)
        hydro_mass=np.zeros(cells);hydro_sum=np.zeros((3,cells))
        epoch_mass=np.zeros(cells);known_epoch_mass=np.zeros(cells)
        lower=np.full(cells,np.inf);upper=np.zeros(cells)
        for start in range(0,len(m),request['chunk_rows']):
            sl=slice(start,min(len(m),start+request['chunk_rows']));keep=(selected[sl]==0)&valid[sl]
            index=np.floor(np.mod(x[sl],box)*n/box).astype(np.int64)%n;at=((index[:,0]*n+index[:,1])*n+index[:,2])[keep]
            weights=m[sl][keep];value=pressure[sl][keep]
            np.add.at(thermal_mass,at,weights);np.add.at(first,at,weights*value)
            np.minimum.at(lower,at,value);np.maximum.at(upper,at,value)
            energy_keep=(selected[sl]==0)&thermal_valid[sl]
            energy_at=((index[:,0]*n+index[:,1])*n+index[:,2])[energy_keep]
            np.add.at(energy_mass,energy_at,m[sl][energy_keep])
            np.add.at(energy_sum,energy_at,m[sl][energy_keep]*thermal[sl][energy_keep]*MSUN_G*1e10)
            work_keep=(selected[sl]==0)&work_valid[sl]
            work_at=((index[:,0]*n+index[:,1])*n+index[:,2])[work_keep]
            np.add.at(work_mass,work_at,m[sl][work_keep])
            np.add.at(work_sum,work_at,m[sl][work_keep]*MSUN_G*pressure_work[sl][work_keep])
            hydro_keep=(selected[sl]==0)&hydro_available[sl]
            hydro_at=((index[:,0]*n+index[:,1])*n+index[:,2])[hydro_keep]
            np.add.at(hydro_mass,hydro_at,m[sl][hydro_keep])
            for axis in range(3):np.add.at(hydro_sum[axis],hydro_at,m[sl][hydro_keep]*hydro_acceleration[sl][hydro_keep,axis])
            for mask,total in ((force_epoch_matches,epoch_mass),(force_epoch_known,known_epoch_mass)):
                keep=(selected[sl]==0)&mask[sl];at=((index[:,0]*n+index[:,1])*n+index[:,2])[keep]
                np.add.at(total,at,m[sl][keep])
        supported=thermal_mass==mass[1];mean_pressure=np.divide(first,thermal_mass,out=np.zeros_like(first),where=thermal_mass>0)
        lower[thermal_mass==0]=0
        arrays.update(baryon_pressure_ratio_km2_s2=mean_pressure.reshape(shape),baryon_pressure_ratio_spread_km2_s2=(upper-lower).reshape(shape),
            baryon_pressure_support_available=supported.reshape(shape),baryon_thermal_energy_erg=energy_sum.reshape(shape),
            baryon_thermal_energy_available=(energy_mass==mass[1]).reshape(shape),
            baryon_peculiar_pressure_work_erg_s=work_sum.reshape(shape),baryon_pressure_work_available=(work_mass==mass[1]).reshape(shape))
        hydro_mean=np.divide(hydro_sum,hydro_mass[None],out=np.zeros_like(hydro_sum),where=hydro_mass[None]>0)
        arrays.update(baryon_stored_hydro_acceleration_cm_s2=hydro_mean.reshape((3,*shape)),
            baryon_stored_hydro_acceleration_available=(hydro_mass==mass[1]).reshape(shape),
            baryon_hydro_force_evaluation_epoch_matches_snapshot=(epoch_mass==mass[1]).reshape(shape),
            baryon_hydro_force_evaluation_epoch_known=(known_epoch_mass==mass[1]).reshape(shape),
            baryon_hydro_force_epoch_available=np.zeros(shape,bool))
        if not np.isfinite(hydro_mean).all():raise ArithmeticError('Stored native hydro acceleration accumulation overflows')
        if not np.isfinite(energy_sum).all() or not np.isfinite(work_sum).all():raise ArithmeticError('Native thermal energy/pressure work accumulation overflows')
        if not np.isfinite(mean_pressure).all() or not np.isfinite(upper-lower).all():raise ArithmeticError('Native gas pressure moment accumulation overflows')
    report={'schema':'measured-twofluid-eulerian-seed-v1','execution_status':'COMPLETE','request':request,'species_order':['cdm','baryons'],
        'native_run_directory':str(root),'native_stage_receipt':{'path':str(Path(parameters['native_stage_receipt']).resolve()),
            'sha256':file_sha256(parameters['native_stage_receipt'])},
        'cell_convention':'periodic_eulerian_cell_averages','actual_scale_factor':a,'box_mpc_h':box,'baryon_fraction':fraction,
        'hubble_parameter':h,'omega_m0':omega,'particle_namespace':request['particle_namespace'],'source_hashes':sources,
        'measurement':'disjoint full-box Eulerian cell membership; extensive actual mass and mass-weighted momentum; measured within-cell velocity dispersion',
        'domain_evidence':'Resolved cell dispersion budget only; this is not a proof of no sub-cell multistreaming or shell crossing',
        'native_expansion_model':expansion_model,
        'science_acceptance':'NOT_ESTABLISHED'}
    if request.get('measure_pressure_support',False):report['native_pressure_model']=pressure_model
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    if request.get('measure_pressure_support',False):report['thermal_energy_scope']='Actual caloric gas sum m*u in erg; native isothermal stored coefficient is not thermodynamic internal energy and is unavailable. Instantaneous saved energy only; pressure-work/source closure remains pending.'
    if request.get('measure_pressure_support',False):report['pressure_work_scope']='Actual sampled sum -m*(P/rho)*div(v_peculiar) from native SPH divergence, erg/s. Homogeneous expansion work -3H sum m P/rho, source terms and time integration are not included; availability is mandatory.'
    if request.get('measure_pressure_support',False):report['native_hydro_acceleration_frame']=hydro_frame
    return export_numeric_audit(destination,report,arrays,{name:{'units':'cm / s^2' if name.endswith('cm_s2') else 'erg / s' if name.endswith('erg_s') else 'erg' if name.endswith('energy_erg') else 'Msun' if name.endswith('msun') else '(km / s)^2' if name.endswith('km2_s2') else 'km / s' if 'velocity' in name else 'dimensionless',
        **({'missing_policy':'numeric zero where baryon_thermal_energy_available is false; unavailable is not zero thermodynamic energy'} if name=='baryon_thermal_energy_erg' else {}),
        **({'missing_policy':'numeric zero where baryon_pressure_work_available is false; unavailable is not zero pressure work'} if name=='baryon_peculiar_pressure_work_erg_s' else {}),
        **({'missing_policy':'numeric zero where baryon_stored_hydro_acceleration_available is false; availability does not prove current force epoch'} if name=='baryon_stored_hydro_acceleration_cm_s2' else {}),
        'quantity':'original native Eulerian species '+name} for name in arrays},algorithm=source,source_hashes=sources,
        context={'epoch':population['epoch'],'coordinates':{'frame':'periodic Eulerian cell-average grid; inertial peculiar velocity'}})


def twofluid_eulerian_seed_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Eulerian seed exceeds workflow allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Eulerian seed escapes the owned generation')
    return publish_twofluid_eulerian_seed(parameters,destination)


def twofluid_eulerian_seed_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Measure full-box Eulerian gas/DM cell averages from original native snapshots')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.parameters.resolve();parameters=json.loads(source.read_text())
    publish_twofluid_eulerian_seed(parameters,args.output);return 0


def publish_twofluid_nonlinear_product(parameters,destination):
    from pathlib import Path
    from .sidm_analysis import _load_product
    from .data import export_numeric_audit
    from .foundation import file_sha256
    request=parameters['request']
    report,initial,sources=_load_product(parameters['initial_fields'],parameters['request_base'],request['memory_gib']*1024**3/4)
    if (report.get('schema')!='measured-twofluid-eulerian-seed-v1' or report.get('species_order')!=['cdm','baryons']
            or report.get('cell_convention')!='periodic_eulerian_cell_averages'
            or report.get('actual_scale_factor')!=request['reference_scale_factor']
            or report.get('box_mpc_h')!=request['box_mpc_h'] or report.get('baryon_fraction')!=request['baryon_fraction']
            or report.get('hubble_parameter')!=request['hubble_parameter'] or report.get('omega_m0')!=request['omega_m0']):
        raise ValueError('Nonlinear reference requires exact-epoch full Eulerian cell-average species seeds')
    for name,unit in (('density_contrast','dimensionless'),('velocity_peculiar_km_s','km / s')):
        if report['numeric_data']['array_contract']['datasets'][name]['units']!=unit:
            raise ValueError('Nonlinear seed array physical units differ: '+name)
    result,arrays=evolve_twofluid_nonlinear_reference(initial,request)
    result['initial_fields_reference']=parameters['initial_fields']
    result['request_base']=str(Path(parameters['request_base']).resolve())
    result['native_seed_definition']={key:report[key] for key in ('particle_namespace','source_hashes','actual_scale_factor','box_mpc_h','baryon_fraction','hubble_parameter','omega_m0','cell_convention')}
    arrays['initial_density_contrast']=initial['density_contrast']
    arrays['initial_velocity_peculiar_km_s']=initial['velocity_peculiar_km_s']
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source);result['source_hashes']=sources
    result['actual_scale_factor']=request['target_scale_factor'];result['box_mpc_h']=request['box_mpc_h']
    result['baryon_fraction']=request['baryon_fraction'];result['cell_convention']='periodic_eulerian_cell_averages'
    result['epoch']={'canonical_time':'scale_factor','actual_scale_factor':result['actual_scale_factor'],
        'scale_factor':result['actual_scale_factor'],'redshift':1/result['actual_scale_factor']-1}
    units={name:'comoving Mpc/h' if 'displacement' in name else 'km / s' if 'velocity' in name else 'comoving Mpc/h per ln(a); sum of normalized cell-density momentum' if 'momentum' in name else 'dimensionless' for name in arrays}
    return export_numeric_audit(destination,result,arrays,{name:{'units':units[name],'quantity':'full nonlinear fluid reference '+name} for name in arrays},
        algorithm=source,source_hashes=sources,context={'epoch':result['epoch'],
        'coordinates':{'frame':'periodic Eulerian cell-average grid; component-first peculiar velocity'}})


def _twofluid_occupied_timestep_summary(path, *, start, finish, maximum_bytes, maximum_records):
    """Read GADGET logs.cc occupied bins, not sync spacing or requested caps.

    Time uses native %g formatting; boundary uncertainty is retained as a
    rejected strip rather than assigning rounded rows to an exact epoch.
    Bin widths use the native %.12f log format in ln(a).
    """
    import re
    import hashlib,json
    from pathlib import Path
    path=Path(path)
    if path.stat().st_size>maximum_bytes:raise MemoryError('Native timebin log exceeds its frozen intake bound')
    header=re.compile(r'^Sync-Point (\d+), Time: ([^,]+), Redshift: [^,]+, Systemstep: [^,]+, Dloga: ([^,]+)$')
    binrow=re.compile(r'^\s*([X ])\s+bin=\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+')
    footer=re.compile(r'^(?:Total active:|PM-Step\. Total:)\s+(\d+)\s+(\d+)$')
    records=[];block=None;last=-1;first_time=None;last_time=None;history=hashlib.sha256()
    def commit():
        if block is None:return
        time,bins,active=block
        if active is None:raise ValueError('Truncated occupied native timebin block')
        actual_active=[sum(row[1+kind] for row in bins if row[4]) for kind in (0,1)]
        if actual_active!=active:raise ValueError('Native timebin footer disagrees with synchronized occupied census')
        tolerance=6e-6*abs(time)
        if time-tolerance<=start or time+tolerance>finish:return
        if not bins:raise ValueError('Interior native timebin block lacks occupied particles')
        if any(row[3]<=0 for row in bins):raise ValueError('Occupied interior native timestep is below log precision')
        counts=np.asarray([sum(row[1+kind] for row in bins) for kind in (0,1)],dtype=object)
        if any(value>2**53 for value in counts):raise ValueError('Native timebin census exceeds exact numeric export support')
        counts=counts.astype(np.int64)
        if np.any(counts<=0):raise ValueError('Native timebin census omits gravity or gas particles')
        maximum=[max(row[3] for row in bins if row[1+kind]>0) for kind in (0,1)]
        average=[sum(row[1+kind]*row[3] for row in bins)/counts[kind] for kind in (0,1)]
        records.append([time,*maximum,*average,*map(int,counts)])
        history.update(json.dumps([time,sorted(bins)],separators=(',',':'),allow_nan=False).encode()+b'\n')
        if len(records)>maximum_records:raise MemoryError('Native timebin record budget exceeded')
    with path.open() as stream:
        for line in stream:
            if len(line)>4096:raise ValueError('Native timebin log has an unsupported long row')
            text=line.strip();match=header.fullmatch(text)
            if text.startswith('Sync-Point'):
                if match is None:raise ValueError('Native cosmological timebin header is malformed')
                commit();index=int(match[1]);time=float(match[2]);delta=float(match[3])
                if (last>=0 and index!=last+1) or not np.isfinite([time,delta]).all() or time<=0 or delta<0 or last_time is not None and time<last_time:
                    raise ValueError('Native timebin clock repeats, skips, restarts or is invalid')
                last=index;first_time=time if first_time is None else first_time;last_time=time
                block=[time,[],None]
            elif block is not None and 'bin=' in line:
                if block[2] is not None:raise ValueError('Native timebin rows occur after the completed footer')
                match=binrow.match(line)
                if match is None:raise ValueError('Malformed occupied native timebin row')
                active=match[1]=='X';index,gravity,gas=map(int,match.group(2,3,4));step=float(match[5])
                if not 0<=index<60 or gravity+gas==0:raise ValueError('Native occupied bin has an unsupported index or empty census')
                if any(row[0]==index for row in block[1]) or not np.isfinite(step) or step<0:raise ValueError('Duplicate or invalid occupied native timebin')
                if gravity+gas>0:
                    block[1].append((index,gravity,gas,step,active))
            elif block is not None and (text.startswith('Total active:') or text.startswith('PM-Step. Total:')):
                match=footer.fullmatch(text)
                if match is None or block[2] is not None:raise ValueError('Malformed or repeated native timebin footer')
                block[2]=list(map(int,match.group(1,2)))
    commit()
    if not records or first_time>start*(1+6e-6) or last_time<finish*(1-6e-6):raise ValueError('Native timebin log does not cover the requested evolution interval')
    values=np.asarray(records,float)
    if not np.all(values[:,5:]==values[0,5:]):raise ValueError('Native occupied-bin particle census changes during evolution')
    return values,{'boundary_relative_rounding_strip':6e-6,'bin_width_absolute_rounding':5e-13,
        'occupied_bin_history_sha256':history.hexdigest(),
        'retained_interior_sync_points':len(values),'maximum_occupied_step_ln_a':values[:,1:3].max(axis=0).tolist(),
        'mean_occupied_step_ln_a':values[:,3:5].mean(axis=0).tolist(),
        'scope':'Occupied gravity/hydro bin widths at interior logged synchronization points; not individual particle trajectory steps'}


def _twofluid_uniform_softening_control(identity,flags,*,start,finish):
    """Extract the native uniform softening route; retain other force knobs."""
    from .foundation import KPC_CM
    if flags & {'ADAPTIVE_HYDRO_SOFTENING','INDIVIDUAL_GRAVITY_SOFTENING'}:
        raise ValueError('Uniform softening controls require fixed particle-type softening classes')
    values=[]
    for index in range(6):
        if identity.get('SofteningClassOfPartType'+str(index))!=str(index):
            raise ValueError('Uniform softening controls require the original six particle-type classes')
        pair=[float(identity['SofteningComovingClass'+str(index)]),float(identity['SofteningMaxPhysClass'+str(index)])]
        if not np.isfinite(pair).all() or not 0<pair[1]<=pair[0]:raise ValueError('Native softening limits are invalid')
        values.append(pair)
    if any(pair!=values[0] for pair in values):raise ValueError('Global softening controls require equal original species limits')
    comoving,physical=values[0];h=float(identity['HubbleParam']);length=float(identity['UnitLength_in_cm'])
    if not np.isfinite([h,length]).all() or min(h,length)<=0:raise ValueError('Native softening physical units are invalid')
    for index in range(6):
        for name in ('SofteningComovingClass','SofteningMaxPhysClass'):identity.pop(name+str(index))
    scale=length/(KPC_CM*h)
    return {'force_parameter':'uniform_gravitational_softening',
        'force_parameter_value':min(finish*comoving,physical)*scale,
        'softening_limits_native':[comoving,physical],
        'softening_proper_endpoint_kpc':[min(start*comoving,physical)*scale,min(finish*comoving,physical)*scale],
        'softening_turnover_scale_factor':physical/comoving,
        'scope':'Native Plummer-equivalent global gravitational softening; hydro smoothing and force-support radii are distinct'}


def _twofluid_pm_build_control(root,identity,bound,*,epoch):
    """Bind varying PMGRID to the actual sealed native build signature."""
    import json,re
    from .evolution import verified_build_cache
    from .foundation import canonical_digest,KPC_CM
    signature_path=root/'input/build_signature.json';receipt_path=root/'input/build_receipt.json'
    for path in (signature_path,receipt_path):
        bound(path)
        if path.stat().st_size>1024**2:raise MemoryError('PM build metadata exceeds allocation')
    signature=json.loads(signature_path.read_text())
    if not verified_build_cache(root/'input/Gadget4',signature_path,receipt_path,signature):raise ValueError('PM control native build is not sealed to its actual binary')
    text=(root/'input/Config.sh').read_text()
    if signature.get('config')!=text:raise ValueError('PM build signature disagrees with original compile configuration')
    options={}
    for line in text.splitlines():
        line=line.split('#',1)[0].strip()
        if not line:continue
        name,separator,value=line.partition('=')
        if name in options:raise ValueError('PM compile configuration repeats an option')
        options[name]=value if separator else None
    raw=options.pop('PMGRID',None)
    if raw is None or not re.fullmatch('[0-9]+',raw):raise ValueError('Actual native PMGRID is missing or invalid')
    grid=int(raw)
    if grid<16 or grid&(grid-1):raise ValueError('Supported native PM control requires a power-of-two grid of at least 16')
    if set(options)&{'PLACEHIGHRESREGION','PM_ZOOM_OPTIMIZED','FMM','ALLOW_DIRECT_SUMMATION'}:
        raise ValueError('Native PM control currently requires the ordinary periodic split TreePM route')
    if not {'PERIODIC','SELFGRAVITY'}<=set(options):raise ValueError('Native PM control requires periodic self-gravity')
    normalized={**signature,'config':options}
    if not signature.get('gadget_source_sha256') or not signature.get('compiler_version'):raise ValueError('PM native build signature lacks source/compiler identity')
    identity['Config.sh']=canonical_digest(options)
    identity['Gadget4']={'matched_build_definition_sha256':canonical_digest(normalized)}
    box=float(identity['BoxSize']);length=float(identity['UnitLength_in_cm']);h=float(identity['HubbleParam'])
    if not np.isfinite([box,length,h,epoch]).all() or min(box,length,h,epoch)<=0:raise ValueError('Native PM grid physical units are invalid')
    return {'force_parameter':'PMGRID','force_parameter_value':1./grid,'pm_grid_size':grid,
        'pm_cell_proper_kpc':epoch*box*length/(grid*h*KPC_CM),
        'build_signature_sha256':canonical_digest(signature),
        'time_split_mode':'common_gravity_timestep' if 'TREEPM_NOTIMESPLIT' in options else 'split_gravity_timestep',
        'scope':'Native periodic TreePM mesh cell spacing; fixed ASMTH makes the physical mesh/short-range split scale refine with PMGRID. Common-step PM/Ewald switching remains the native fixed active-population policy.'}


def _twofluid_native_timestep_evidence(native,initial,request):
    import h5py
    import yaml
    from pathlib import Path
    from .sidm_analysis import _stage_native_bindings
    from .evolution import parameter_values,resolve_gadget_ic_files
    from .foundation import file_sha256
    from core.resources import mpi_ranks, omp_threads
    from .halos import _snapshot_files
    root=Path(native['native_run_directory']).resolve()
    sources=_stage_native_bindings({'native_stage_receipt':native['native_stage_receipt']['path'],
        'native_run_directory':str(root),'request':{'memory_gib':request['memory_gib']}})
    if sources.get(native['native_stage_receipt']['path'])!=native['native_stage_receipt']['sha256']:raise ValueError('Native timestep measurement receipt changed')
    def bound(path):
        path=Path(path).resolve()
        if sources.get(str(path))!=file_sha256(path):raise ValueError('Native timestep evidence is outside the completed original run: '+str(path))
        return str(path)
    parameter_path=Path(bound(root/'input/param.txt'))
    if parameter_path.stat().st_size>1024**2:raise MemoryError('Native timestep parameter metadata exceeds intake bound')
    params=parameter_values(parameter_path)
    if params.get('ComovingIntegrationOn')!='1' or float(params['TimeBegin'])!=initial['actual_scale_factor']:
        raise ValueError('Native timestep family requires full original cosmological evolution from its seed epoch')
    cap=float(params['MaxSizeTimestep'])
    if not math.isfinite(cap) or cap<=0:raise ValueError('Native timestep cap is invalid')
    # A copied parameter file alone is not evidence of an operative setting.
    for measurement in (initial,native):
        snapshots=_snapshot_files(root,measurement['request']['snapshot'])
        if not snapshots:raise ValueError('Native timestep evidence lacks its actual saved snapshot set')
        for snapshot in snapshots:
            with h5py.File(bound(snapshot),'r') as saved:
                actual=saved['Parameters'].attrs
                for name,text in params.items():
                    try:expected=float(text)
                    except ValueError:continue
                    if name not in actual or np.ndim(actual[name])!=0 or not np.isfinite(expected) or float(actual[name])!=expected:
                        raise ValueError('Actual saved native parameter differs from launch metadata: '+name)
    identity={key:value for key,value in params.items() if key not in {'MaxSizeTimestep','OutputDir','SnapshotFileBase'}}
    for key in ('InitCondFile','OutputListFilename','TreecoolFile','CosmologyTableFile'):
        if key not in identity:continue
        if key=='CosmologyTableFile' and identity[key]=='none':continue
        value=identity[key];path=Path(value);path=path if path.is_absolute() else root/'input'/path
        if key=='InitCondFile':
            from .data import native_publication_markers
            paths=resolve_gadget_ic_files(path,maximum_shards=request.get('maximum_ic_shards',4096))
            checksums=[sources[bound(shard)] for shard in paths]
            for marker in native_publication_markers(paths):bound(marker)
            identity[key]={'shards_sha256':checksums}
            continue
        identity[key]={'sha256':sources[bound(path)]}
    for name in ('Config.sh','Gadget4'):
        identity[name]=sources[bound(root/'input'/name)]
    configuration=Path(bound(root/'config.yaml'))
    if configuration.stat().st_size>1024**2:raise MemoryError('Native timestep launch configuration exceeds metadata allocation')
    cfg=yaml.safe_load(configuration.read_text())
    identity['execution_layout']={'mpi_ranks':mpi_ranks(cfg),'omp_threads':omp_threads(cfg)}
    output=Path(params['OutputDir']);output=output if output.is_absolute() else root/'input'/output
    log=bound(output/'timebins.txt')
    rows,summary=_twofluid_occupied_timestep_summary(log,start=initial['actual_scale_factor'],finish=native['actual_scale_factor'],
        maximum_bytes=request['maximum_timebin_log_bytes'],maximum_records=request['maximum_timebin_records'])
    if max(summary['maximum_occupied_step_ln_a'])>cap+5e-13:raise ValueError('Occupied native timebin exceeds the configured timestep cap')
    summary['configured_maximum_step_ln_a']=cap
    if request.get('refinement_axis')in ('native_force','native_hydro'):
        flags={line.split('#',1)[0].strip().split('=',1)[0] for line in (root/'input/Config.sh').read_text().splitlines()}
        if 'SELFGRAVITY' not in flags:raise ValueError('Native force controls require active self-gravity')
        if flags & {'FMM','ALLOW_DIRECT_SUMMATION'}:raise ValueError('Relative tree force controls do not support FMM or direct-summation bypass builds')
        identity['MaxSizeTimestep']=cap
        if request.get('refinement_axis')=='native_hydro':
            if 'PRESSURE_ENTROPY_SPH' not in flags or flags & {'TIMEDEP_ART_VISC','RESEARCH_GRACKLE','STARFORMATION','STELLAR_EVOLUTION','BLACK_HOLES'}:
                raise ValueError('Native hydro viscosity control requires fixed-coefficient pressure-entropy SPH without source/EOS operators')
            value=float(identity.pop('ArtBulkViscConst'))
            if not math.isfinite(value) or value<0:raise ValueError('Actual native artificial viscosity is invalid')
            summary.update(hydro_parameter='ArtBulkViscConst',force_parameter_value=value)
        elif request.get('force_parameter')=='PMGRID':
            summary.update(_twofluid_pm_build_control(root,identity,bound,epoch=native['actual_scale_factor']))
        elif request.get('force_parameter','ErrTolForceAcc')=='uniform_gravitational_softening':
            summary.update(_twofluid_uniform_softening_control(identity,flags,start=initial['actual_scale_factor'],finish=native['actual_scale_factor']))
        else:
            if identity.get('TypeOfOpeningCriterion')!='1':raise ValueError('Native relative-force controls require the relative tree opening criterion')
            value=float(identity.pop('ErrTolForceAcc'))
            if not math.isfinite(value) or value<=0:raise ValueError('Actual native relative-force accuracy is invalid')
            summary.update(force_parameter='ErrTolForceAcc',force_parameter_value=value)
    if any(file_sha256(path)!=checksum for path,checksum in sources.items()):raise ValueError('Native timestep evidence changed during intake')
    return identity,rows,summary,sources


def validate_twofluid_native_timestep_campaign_request(parameters):
    import copy
    request=parameters['request']
    keys={'schema','answers_template','initial_conditions','maximum_step_ln_a_levels','maximum_levels',
        'seed_request','target_snapshots','reference_request','comparison_limits','family_limits',
        'resources','followup_limits','native_wall_seconds','native_disk_gib_per_level'}
    if not isinstance(request,dict) or request.get('schema')not in ('twofluid-native-timestep-campaign-request-v1','twofluid-native-force-campaign-request-v1','twofluid-native-softening-campaign-request-v1','twofluid-native-pm-campaign-request-v1','twofluid-native-hydro-campaign-request-v1'):
        raise ValueError('Complete independent native timestep launch campaign required')
    softening=request.get('schema')=='twofluid-native-softening-campaign-request-v1'
    hydro=request.get('schema')=='twofluid-native-hydro-campaign-request-v1'
    pm=request.get('schema')=='twofluid-native-pm-campaign-request-v1'
    force=request.get('schema')=='twofluid-native-force-campaign-request-v1' or softening or pm
    if force and request.get('family_limits',{}).get('force_parameter')!=('PMGRID' if pm else 'uniform_gravitational_softening' if softening else 'ErrTolForceAcc'):
        raise ValueError('Version-1 force launch campaign supports ErrTolForceAcc; uniform softening launch composition remains pending')
    if hydro:keys=(keys-{'maximum_step_ln_a_levels'})|{'artificial_viscosity_levels'}
    if hydro and request.get('family_limits',{}).get('hydro_parameter')!='ArtBulkViscConst':raise ValueError('Unsupported hydro launch control')
    if force:keys=(keys-{'maximum_step_ln_a_levels'})|{'pm_grid_levels' if pm else 'softening_levels' if softening else 'force_accuracy_levels'}
    if set(request)!=keys:raise ValueError('Complete versioned native control campaign fields required')
    levels=request['artificial_viscosity_levels' if hydro else 'pm_grid_levels' if pm else 'softening_levels' if softening else 'force_accuracy_levels' if force else 'maximum_step_ln_a_levels']
    if type(request['maximum_levels'])is not int or request['maximum_levels']<3 or not isinstance(levels,list) or not 3<=len(levels)<=request['maximum_levels']:
        raise ValueError('At least three bounded native timestep levels required')
    if pm:
        if any(type(value)is not int or value<16 or value&(value-1) for value in levels) or any(b<=a for a,b in zip(levels,levels[1:])):
            raise ValueError('Increasing power-of-two native PM grid levels required')
        if max(levels)**3*128>request['resources']['memory_gib']*1024**3:raise MemoryError('Native PM mesh workspace exceeds allocated memory')
    elif softening:
        names={'comoving_kpc_h','maximum_physical_kpc_h'}
        for level in levels:
            if not isinstance(level,dict) or set(level)!=names or any(type(value)not in (int,float) or not math.isfinite(value) or value<=0 for value in level.values()):
                raise ValueError('Explicit positive comoving and physical softening levels required')
            if level['maximum_physical_kpc_h']>level['comoving_kpc_h']:raise ValueError('Physical softening cap exceeds the comoving limit')
        turnover=levels[0]['maximum_physical_kpc_h']/levels[0]['comoving_kpc_h']
        for old,new in zip(levels,levels[1:]):
            if any(new[key]>=old[key] for key in names):raise ValueError('Both softening limits must strictly decrease')
            if not math.isclose(new['maximum_physical_kpc_h']/new['comoving_kpc_h'],turnover,rel_tol=32*np.finfo(float).eps,abs_tol=0):
                raise ValueError('Softening launch campaign changes its turnover epoch')
    elif any(type(value)not in (int,float) or not math.isfinite(value) or value<=0 for value in levels) or any(b>=a for a,b in zip(levels,levels[1:])):
        raise ValueError('Strictly decreasing positive native control values required')
    _validate_twofluid_product_reference(request['answers_template'],'native answer template')
    ic=request['initial_conditions']
    if not isinstance(ic,dict) or set(ic)!={'primary','files','native_generation_report'} or not isinstance(ic['files'],list) or not ic['files']:
        raise ValueError('Complete pinned native IC launch set and generation report required')
    for ref in [ic['primary'],*ic['files'],ic['native_generation_report']]:_validate_twofluid_product_reference(ref,'native launch asset')
    if ic['primary'] not in ic['files'] or len({ref['path'] for ref in ic['files']})!=len(ic['files']):raise ValueError('Native IC primary must occur exactly once in its complete launch set')
    seed=request['seed_request']
    if seed['snapshot']!=0:raise ValueError('Native launch campaign must measure its original initial snapshot zero')
    # Reuse existing physical, target, comparison and allocation validators.
    base={'schema':'twofluid-native-comparison-campaign-request-v1','seed_parameters':{'request':seed,'native_run_directory':'pending','native_stage_receipt':'pending'},
        'target_snapshots':request['target_snapshots'],'grid_sizes':[seed['grid_size']],'maximum_grids':1,
        'maximum_targets':len(request['target_snapshots']),'reference_request':request['reference_request'],
        'comparison_limits':request['comparison_limits'],'resources':request['resources'],'followup_limits':request['followup_limits']}
    validate_twofluid_native_campaign_request({'request':base})
    family={'schema':'twofluid-native-comparison-family-request-v4' if hydro else 'twofluid-native-comparison-family-request-v3' if force else 'twofluid-native-comparison-family-request-v2','refinement_axis':'native_hydro' if hydro else 'native_force' if force else 'native_timestep',**copy.deepcopy(request['family_limits']),
        'comparisons':[{'path':'pending_'+str(i),'sha256':'0'*64} for i in range(len(levels))]}
    validate_twofluid_native_family_request({'request':family})
    if set(request['family_limits'])!=set(family)-{'schema','refinement_axis','comparisons'}:raise ValueError('Campaign owns timestep family references and axis')
    resources=request['resources'];targets=len(request['target_snapshots']);count=len(levels)*(3+3*targets)+targets
    for key in ('native_wall_seconds','native_disk_gib_per_level'):
        value=request[key]
        if type(value)not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive explicit native launch resource budgets required')
    retained=len(levels)*(request['native_disk_gib_per_level']*1024**3+seed['grid_size']**3*(512+targets*8192))
    retained+=targets*len(levels)*family['maximum_timebin_records']*128
    if retained>resources['disk_gib']*1024**3 or count>resources['maximum_stage_count'] or count>request['followup_limits']['maximum_children'] or request['followup_limits']['maximum_depth']<7:
        raise MemoryError('Independent native launch family exceeds retained disk/stage/follow-up bounds')
    if family['memory_gib']>resources['memory_gib'] or seed['grid_size']**3>family['maximum_grid_cells']:raise MemoryError('Native timestep family exceeds comparison allocation')
    return request


def twofluid_native_timestep_campaign_input_bindings(parameters):
    import json
    from pathlib import Path
    from .foundation import file_sha256
    from .evolution import resolve_gadget_ic_files,gadget_ic_prefix
    request=validate_twofluid_native_timestep_campaign_request(parameters);base=Path(parameters['request_base']).resolve();sources={}
    ic=request['initial_conditions']
    for ref in [request['answers_template'],ic['primary'],*ic['files'],ic['native_generation_report']]:
        path=(base/ref['path']).resolve()
        if file_sha256(path)!=ref['sha256']:raise ValueError('Native timestep launch input changed: '+str(path))
        sources[str(path)]=ref['sha256']
    maximum=request['family_limits'].get('maximum_ic_shards',4096)
    paths=resolve_gadget_ic_files(gadget_ic_prefix(base/ic['primary']['path']),maximum_shards=maximum)
    if {str(path.resolve()) for path in paths}!={str((base/ref['path']).resolve()) for ref in ic['files']}:raise ValueError('Declared native timestep IC files differ from actual complete launch set')
    from .data import native_publication_markers
    for marker in native_publication_markers(paths):sources[str(marker.resolve())]=file_sha256(marker)
    for ref in (request['answers_template'],ic['native_generation_report']):
        if (base/ref['path']).stat().st_size>16*1024**2:raise MemoryError('Native launch metadata exceeds its bounded intake')
    suite=Path(__file__).resolve().parents[1]
    for name in ('core/initial_conditions.py','core/evolution.py','core/data.py','core/workflow.py','core/research/generator_controls.py',
                 'core/foundation.py','core/zoom.py','commands.py','core/__init__.py','configs/core_source_layout.json'):
        path=suite/name;sources[str(path)]=file_sha256(path)
    return sources


def compile_twofluid_native_timestep_campaign(request,destination,*,request_base):
    import copy,json
    from pathlib import Path
    from .foundation import atomic_json,file_sha256
    from core.research.capabilities import validate_native_answers
    from .workflow import compile_requested_product_dag,default_registry
    base=Path(request_base).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Retain previous native timestep campaign designs')
    sources=twofluid_native_timestep_campaign_input_bindings({'request':request,'request_base':str(base)})
    template=json.loads((base/request['answers_template']['path']).read_text());prepared=[]
    softening=request['schema']=='twofluid-native-softening-campaign-request-v1'
    hydro=request['schema']=='twofluid-native-hydro-campaign-request-v1'
    pm=request['schema']=='twofluid-native-pm-campaign-request-v1'
    force=request['schema']=='twofluid-native-force-campaign-request-v1' or softening or pm
    level_key='artificial_viscosity_levels' if hydro else 'pm_grid_levels' if pm else 'softening_levels' if softening else 'force_accuracy_levels' if force else 'maximum_step_ln_a_levels'
    answer_key='artificial_viscosity' if hydro else 'pm_grid' if pm else 'err_tol_force_acc' if force else 'max_size_timestep_ln_a'
    for index,cap in enumerate(request[level_key]):
        answers=copy.deepcopy(template)
        if softening:
            answers['softening_kpc_h']=cap['comoving_kpc_h']
            answers['softening_max_physical_kpc_h']=cap['maximum_physical_kpc_h']
        else:answers[answer_key]=cap
        answers['name']='twofluid_native_'+('hydro_' if hydro else 'force_' if force else 'timestep_')+str(index)
        cfg=validate_native_answers(answers,request['resources'])
        if softening:
            from .zoom import softening_plan
            plan=softening_plan(cfg['simulation'])
            if plan['type_classes']!=list(range(6)) or plan['comoving']!=[cap['comoving_kpc_h']]*6 or plan['maximum_physical']!=[cap['maximum_physical_kpc_h']]*6:
                raise ValueError('Public native route does not preserve the requested uniform softening classes')
        elif hydro:
            if cfg['simulation']['physics']!='hydro' or cfg['hydro']['artificial_viscosity']!=cap:raise ValueError('Public route changes requested hydro control')
        elif cfg['simulation'][answer_key]!=cap:raise ValueError('Public native route does not preserve the requested native control value')
        prepared.append(answers)
    stages=[];requested=[];results=[];assets=[{'path':path,'sha256':sha} for path,sha in sources.items()]
    def artifact(stage,path):return {'stage':stage,'artifact':path}
    def product(stage,path):return {'path':artifact(stage,path),'sha256':{**artifact(stage,path),'load':'sha256'}}
    def field(stage,name):return {**artifact(stage,'measurement.json'),'load':'json','field':[name]}
    def add(identity,kind,parameters,path,dependencies=(),resources=None):
        row={'stage_id':identity,'kind':kind,'parameters':parameters,'depends_on':list(dependencies),'outputs':{identity:path},'assets':copy.deepcopy(assets)}
        if resources is not None:row['resources']=resources
        stages.append(row)
    for index,answers in enumerate(prepared):
        tag=str(index);answer_id='answers_'+tag;run_id='evolution_'+tag;initial_id='initial_'+tag
        add(answer_id,'materialize_request',{'request':answers,'filename':'request.json'},'request.json')
        allocation=copy.deepcopy(request['resources']);allocation['disk_gib']=request['native_disk_gib_per_level']
        ic=request['initial_conditions']
        add(run_id,'public_command',{'command':'gadget','wall_seconds':request['native_wall_seconds'],
            'arguments':['--answers',artifact(answer_id,'request.json'),'--existing-ic',{'input':str((base/ic['primary']['path']).resolve())},
                '--native-generation-report',{'input':str((base/ic['native_generation_report']['path']).resolve())},'--run-dir','{generation}/run']},
            'run',[answer_id],allocation)
        seed={'request':copy.deepcopy(request['seed_request']),'native_stage_receipt':artifact(run_id,'stage-receipt.json'),
            'native_run_directory':artifact(run_id,'run'),'destination':'{generation}/measurement.json'}
        add(initial_id,'twofluid_eulerian_seed',seed,'measurement.json',[run_id])
        for snapshot in request['target_snapshots']:
            suffix=tag+'_'+str(snapshot);native_id='native_'+suffix;reference_id='reference_'+suffix;comparison_id='comparison_'+suffix
            native=copy.deepcopy(seed);native['request']['snapshot']=snapshot
            add(native_id,'twofluid_eulerian_seed',native,'measurement.json',[run_id])
            model=copy.deepcopy(request['reference_request']);model['reference_scale_factor']=field(initial_id,'actual_scale_factor');model['target_scale_factor']=field(native_id,'actual_scale_factor')
            add(reference_id,'twofluid_nonlinear_reference',{'request':model,'initial_fields':product(initial_id,'measurement.json'),'request_base':str(base),'destination':'{generation}/reference.json'},'reference.json',[initial_id,native_id])
            comparison={'schema':'twofluid-native-evolution-comparison-request-v1',**copy.deepcopy(request['comparison_limits']),
                'initial_seed':product(initial_id,'measurement.json'),'native_measurement':product(native_id,'measurement.json'),'reference':product(reference_id,'reference.json')}
            add(comparison_id,'twofluid_native_comparison',{'request':comparison,'request_base':str(base),'destination':'{generation}/comparison.json'},'comparison.json',[initial_id,native_id,reference_id])
            results.append({'native_level':index,'control_value':request[level_key][index],'refinement_axis':'native_force' if force else 'native_timestep','target_snapshot':snapshot,'comparison_stage':comparison_id})
    for snapshot in request['target_snapshots']:
        ids=['comparison_'+str(index)+'_'+str(snapshot) for index in range(len(prepared))]
        family={'schema':'twofluid-native-comparison-family-request-v4' if hydro else 'twofluid-native-comparison-family-request-v3' if force else 'twofluid-native-comparison-family-request-v2','refinement_axis':'native_hydro' if hydro else 'native_force' if force else 'native_timestep',**copy.deepcopy(request['family_limits']),
            'comparisons':[product(identity,'comparison.json') for identity in ids]}
        identity='family_'+str(snapshot)
        add(identity,'twofluid_native_family',{'request':family,'request_base':str(base),'destination':'{generation}/family.json'},'family.json',ids);requested.append(identity)
    workflow={'schema':'research-workflow-v1','resources':copy.deepcopy(request['resources']),'followup_limits':copy.deepcopy(request['followup_limits']),'stages':stages,'requested_products':requested}
    compile_requested_product_dag(workflow,default_registry(),request_base=base)
    if any(file_sha256(path)!=sha for path,sha in sources.items()):raise ValueError('Native launch sources changed during compilation')
    destination.mkdir(parents=True);atomic_json(destination/'workflow.json',workflow)
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    report={'schema':'twofluid-native-hydro-campaign-design-v1' if hydro else 'twofluid-native-pm-campaign-design-v1' if pm else 'twofluid-native-softening-campaign-design-v1' if softening else 'twofluid-native-force-campaign-design-v1' if force else 'twofluid-native-timestep-campaign-design-v1','request':request,'workflow':str(destination/'workflow.json'),'workflow_sha256':file_sha256(destination/'workflow.json'),
        'stage_count':len(stages),'comparison_results':results,'source_hashes':sources,'execution_status':'NOT_EXECUTED','science_acceptance':'NOT_ESTABLISHED',
        'scope':'Independent public native launches with one frozen IC/template and one varied native control -> per-run exact seeds and references -> native/reference comparisons -> operative timestep family; actual native execution and physical acceptance remain pending'}
    atomic_json(destination/'design.json',report);return report


def validate_twofluid_pressure_history_request(parameters):
    request=parameters['request']
    keys={'schema','comparisons','maximum_epochs','maximum_grid_cells','memory_gib'}
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='twofluid-native-pressure-history-request-v1':
        raise ValueError('Complete saved native pressure-history request required')
    if type(request['maximum_epochs'])is not int or request['maximum_epochs']<2 or type(request['maximum_grid_cells'])is not int or request['maximum_grid_cells']<8:
        raise ValueError('Explicit native pressure history epoch/grid bounds required')
    refs=request['comparisons']
    if not isinstance(refs,list) or not 2<=len(refs)<=request['maximum_epochs']:raise ValueError('At least two bounded actual evolved pressure comparisons required')
    for ref in refs:_validate_twofluid_product_reference(ref,'pressure history comparison')
    memory=request['memory_gib']
    if type(memory)not in (int,float) or not math.isfinite(memory) or memory<=0:raise ValueError('Positive pressure-history memory allocation required')
    if request['maximum_grid_cells']*(4096+128*request['maximum_epochs'])>memory*1024**3:raise MemoryError('Pressure history retained masks/workspace exceed allocation')
    return request


def publish_twofluid_pressure_history(parameters,destination):
    from pathlib import Path
    from .sidm_analysis import _load_product
    from .data import export_numeric_audit
    from .foundation import file_sha256
    request=validate_twofluid_pressure_history_request(parameters);sources={};output={};epochs=[];labels=[];support=[];definition=None;bindings=[]
    for index,ref in enumerate(request['comparisons']):
        report,arrays,binding=_load_product(ref,parameters['request_base'],request['memory_gib']*1024**3/3);sources.update(binding)
        if report.get('schema')!='measured-twofluid-native-evolution-comparison-v1' or report.get('execution_status')!='COMPLETE':raise ValueError('Saved pressure history needs complete actual comparison products')
        original=report['request'];limits=original.get('pressure_limits',{})
        if limits.get('include_initial_pressure_support')is not True:raise ValueError('Pressure history requires actual initial and evolved pressure support at every epoch')
        initial,_,binding=_load_product(original['initial_seed'],report['request_base'],request['memory_gib']*1024**3/3);sources.update(binding)
        native,_,binding=_load_product(original['native_measurement'],report['request_base'],request['memory_gib']*1024**3/3);sources.update(binding)
        model,_,binding=_load_product(original['reference'],report['request_base'],request['memory_gib']*1024**3/3);sources.update(binding)
        epoch=report['actual_scale_factor'];label=native['request']['snapshot'];n=report['reference_grid_size']
        if epoch!=native['actual_scale_factor'] or epoch!=model['target_scale_factor'] or initial['actual_scale_factor']!=model['reference_scale_factor']:
            raise ValueError('Saved pressure history relabels its original native/reference epochs')
        if model.get('initial_fields_reference',{}).get('sha256')!=original['initial_seed']['sha256']:
            raise ValueError('Saved pressure history reference changes its original initial seed')
        if initial['native_run_directory']!=native['native_run_directory'] or initial['native_stage_receipt']!=native['native_stage_receipt']:
            raise ValueError('Saved pressure history initial/evolved products change native run identity')
        if type(epoch)not in (int,float) or not math.isfinite(epoch) or epoch<=initial['actual_scale_factor'] or epochs and epoch<=epochs[-1]:raise ValueError('Pressure history actual epochs must strictly increase after the seed')
        if type(label)is not int or label<0 or label in labels:raise ValueError('Pressure history repeats or mislabels its native snapshots')
        if type(n)is not int or n<2 or n**3>request['maximum_grid_cells'] or report['native_grid_size']!=n:raise ValueError('Complete matched pressure history grids required')
        current={'run':native['native_run_directory'],'receipt':native['native_stage_receipt'],'namespace':native['particle_namespace'],
            'initial_seed_sha256':original['initial_seed']['sha256'],'grid_size':n,
            'native_pressure_model':native.get('native_pressure_model'),
            'native_expansion_model':native.get('native_expansion_model'),
            'physical_reference':{key:value for key,value in model['request'].items() if key not in {'target_scale_factor','memory_gib','maximum_grid_cells','maximum_steps','maximum_rhs_evaluations','maximum_step_rejections','material_chunk_rows','material_map_maximum_subboxes','material_map_maximum_depth'}},
            'comparison_limits':{key:value for key,value in original.items() if key not in {'schema','initial_seed','reference','native_measurement','memory_gib','maximum_grid_cells'}}}
        if current['native_pressure_model'] is None:raise ValueError('Original native EOS definition is missing from pressure history')
        if definition is not None and current!=definition:raise ValueError('Saved pressure history changes native run, initial seed, grid, model or frozen limits')
        definition=current;epochs.append(epoch);labels.append(label);gates=[]
        for name in ('initial_pressure_within_limits','target_pressure_within_limits','pressure_within_limits'):
            mask=np.asarray(arrays[name])
            if mask.dtype.kind!='b' or mask.shape!=(n,n,n):raise ValueError('Complete original endpoint pressure masks required')
            if report['numeric_data']['array_contract']['datasets'][name]['units']!='dimensionless':raise ValueError('Pressure history mask units differ')
            output['epoch_%03d_'%index+name]=mask.copy();gates.append(bool(mask.all()))
        if not np.array_equal(arrays['pressure_within_limits'],arrays['initial_pressure_within_limits']&arrays['target_pressure_within_limits']):raise ValueError('Pressure history endpoint masks do not close')
        bound=report.get('pressure_model_binding',{})
        expected=_twofluid_pressure_domain_binding(current['native_pressure_model'],model['request']['pressure_model'],gates[-1])
        if bound!=expected:raise ValueError('Pressure history model binding differs from original sampled support')
        # Retain all native/reference failures, even when pressure alone agrees.
        supported=all(gates) and report['status']=='WITHIN_DECLARED_NATIVE_REFERENCE_LIMITS' and bound['state']=='SAMPLED_ISOTHERMAL_PRESSURE_LAW_SUPPORTED'
        support.append(supported);bindings.append({'epoch':epoch,'snapshot':label,'comparison':ref,'pressure_model_binding':bound,'comparison_status':report['status']})
    output.update(actual_scale_factors=np.asarray(epochs),native_snapshot_labels=np.asarray(labels,np.int64),saved_epoch_pressure_supported=np.asarray(support,bool))
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    result={'schema':'measured-twofluid-native-pressure-history-v1','execution_status':'COMPLETE','request':request,'request_base':str(Path(parameters['request_base']).resolve()),
        'fixed_definition':definition,'saved_epoch_bindings':bindings,'status':'SAVED_PRESSURE_EPOCHS_SUPPORTED' if all(support) else 'PRESSURE_HISTORY_UNRESOLVED',
        'scope':'Actual saved initial/evolved pressure coefficient and spread masks from one original native run/seed with fixed reference model and limits. No unsaved/interpolated thermodynamic history, pressure-work/source energy, native SPH-force, shock or continuum acceptance.',
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    return export_numeric_audit(destination,result,output,{name:{'units':'dimensionless','quantity':'saved pressure history '+name} for name in output},algorithm=source,source_hashes=sources,
        context={'epoch':{'canonical_time':'scale_factor','measured_scale_factors':epochs},'coordinates':{'frame':'fixed periodic Eulerian pressure-support cell grid'}})


def validate_twofluid_native_family_request(parameters):
    if parameters['request'].get('schema')=='twofluid-native-pressure-history-request-v1':return validate_twofluid_pressure_history_request(parameters)
    request=parameters['request'];keys={'schema','comparisons','maximum_levels','maximum_grid_cells','memory_gib','maximum_adjacent_error_ratio',
        'density_trend_floor','velocity_trend_floor_km_s'}
    hydro=isinstance(request,dict) and request.get('schema')=='twofluid-native-comparison-family-request-v4'
    force=hydro or isinstance(request,dict) and request.get('schema')=='twofluid-native-comparison-family-request-v3'
    temporal=isinstance(request,dict) and request.get('schema')in ('twofluid-native-comparison-family-request-v2','twofluid-native-comparison-family-request-v3','twofluid-native-comparison-family-request-v4')
    if temporal:keys|={'refinement_axis','maximum_timebin_log_bytes','maximum_timebin_records'}
    if force:keys.add('hydro_parameter' if hydro else 'force_parameter')
    if temporal and 'maximum_ic_shards' in request:keys.add('maximum_ic_shards')
    if not isinstance(request,dict) or set(request)!=keys or request['schema'] not in ('twofluid-native-comparison-family-request-v1','twofluid-native-comparison-family-request-v2','twofluid-native-comparison-family-request-v3','twofluid-native-comparison-family-request-v4'):
        raise ValueError('Complete native/reference grid-family measurement required')
    if temporal:
        if request['refinement_axis']!=('native_hydro' if hydro else 'native_force' if force else 'native_timestep'):raise ValueError('Native control family version and refinement axis disagree')
        if hydro and request['hydro_parameter']!='ArtBulkViscConst':raise ValueError('Unsupported native hydro control parameter')
        if force and not hydro and request['force_parameter']not in ('ErrTolForceAcc','uniform_gravitational_softening','PMGRID'):raise ValueError('Unsupported native force-control parameter')
        for key in ('maximum_timebin_log_bytes','maximum_timebin_records'):
            if type(request[key])is not int or request[key]<1:raise ValueError('Positive bounded native timestep log intake required')
        shards=request.get('maximum_ic_shards',4096)
        if type(shards)is not int or shards<1:raise ValueError('Positive bounded native IC shard inventory required')
    for name,minimum in (('maximum_levels',3),('maximum_grid_cells',64)):
        if type(request[name])is not int or request[name]<minimum:raise ValueError('Exact positive grid-family bounds required')
    if not isinstance(request['comparisons'],list) or not 3<=len(request['comparisons'])<=request['maximum_levels']:
        raise ValueError('At least three actual native comparison levels required')
    for reference in request['comparisons']:_validate_twofluid_product_reference(reference,'native comparison level')
    for name in ('memory_gib','maximum_adjacent_error_ratio','density_trend_floor','velocity_trend_floor_km_s'):
        value=request[name]
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Explicit positive grid-trend budgets required')
    if temporal and request['maximum_timebin_records']*request['maximum_levels']*512>request['memory_gib']*1024**3/8:raise MemoryError('Native timestep records exceed workspace')
    if temporal and request['maximum_timebin_log_bytes']>request['memory_gib']*1024**3/8:raise MemoryError('Native timestep log intake exceeds workspace')
    if temporal and request.get('maximum_ic_shards',4096)*4096>request['memory_gib']*1024**3/8:raise MemoryError('Native IC shard metadata exceeds workspace')
    if request['maximum_adjacent_error_ratio']>1:raise ValueError('Grid error-growth ratio must not exceed one')
    if request['maximum_grid_cells']*4096>request['memory_gib']*1024**3:raise MemoryError('Grid-family comparison workspace exceeds allocation')
    return request


def twofluid_native_family_input_bindings(parameters):
    from .sidm_analysis import _load_product
    request=validate_twofluid_native_family_request(parameters);sources={}
    for reference in request['comparisons']:
        report,_,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/3);sources.update(binding)
        if request['schema']in ('twofluid-native-comparison-family-request-v2','twofluid-native-comparison-family-request-v3','twofluid-native-comparison-family-request-v4'):
            initial,_,binding=_load_product(report['request']['initial_seed'],report['request_base'],request['memory_gib']*1024**3/3);sources.update(binding)
            native,_,binding=_load_product(report['request']['native_measurement'],report['request_base'],request['memory_gib']*1024**3/3);sources.update(binding)
            _,_,_,binding=_twofluid_native_timestep_evidence(native,initial,request);sources.update(binding)
    return sources


def publish_twofluid_native_family(parameters,destination):
    if parameters['request'].get('schema')=='twofluid-native-pressure-history-request-v1':return publish_twofluid_pressure_history(parameters,destination)
    from pathlib import Path
    from .sidm_analysis import _load_product
    from .foundation import file_sha256
    from .data import export_numeric_audit
    request=validate_twofluid_native_family_request(parameters);sources={};means=[];maxima=[];grids=[];support=[];definition=None;seen=set()
    hydro=request['schema']=='twofluid-native-comparison-family-request-v4'
    force=hydro or request['schema']=='twofluid-native-comparison-family-request-v3'
    control_parameter=request['hydro_parameter'] if hydro else request.get('force_parameter')
    temporal=request['schema']in ('twofluid-native-comparison-family-request-v2','twofluid-native-comparison-family-request-v3','twofluid-native-comparison-family-request-v4');steps=[];timebin_rows=[];seed_fields=None
    for reference in request['comparisons']:
        if reference['sha256'] in seen:raise ValueError('Grid family repeats an identical comparison')
        seen.add(reference['sha256']);report,arrays,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/3);sources.update(binding)
        if report.get('schema')!='measured-twofluid-native-evolution-comparison-v1' or report.get('execution_status')!='COMPLETE':
            raise ValueError('Grid family requires actual complete native comparison products')
        base=report['request_base'];initial,initial_arrays,binding=_load_product(report['request']['initial_seed'],base,request['memory_gib']*1024**3/3);sources.update(binding)
        native,_,binding=_load_product(report['request']['native_measurement'],base,request['memory_gib']*1024**3/3);sources.update(binding)
        model,_,binding=_load_product(report['request']['reference'],base,request['memory_gib']*1024**3/3);sources.update(binding)
        current={'native_run':native['native_run_directory'],'receipt':native['native_stage_receipt'],'initial_snapshot':initial['request']['snapshot'],
            'target_snapshot':native['request']['snapshot'],'namespace':native['particle_namespace'],'epoch':report['actual_scale_factor'],
            'physical_request':{key:value for key,value in model['request'].items() if key not in {'maximum_grid_cells','memory_gib','maximum_steps',
                'maximum_rhs_evaluations','maximum_step_rejections','material_chunk_rows','material_map_maximum_subboxes','material_map_maximum_depth'}},
            'comparison_limits':{key:value for key,value in report['request'].items() if key not in {'initial_seed','reference','native_measurement','memory_gib','maximum_grid_cells'}}}
        if temporal:
            identity,rows,summary,binding=_twofluid_native_timestep_evidence(native,initial,request);sources.update(binding)
            current.pop('native_run');current.pop('receipt');current['native_definition']=identity
            current['initial_epoch']=initial['actual_scale_factor']
            # Identical grid fields alone cannot establish identical particle ICs;
            # the actual launch-bound IC bytes above must also agree.
            fields={key:np.asarray(initial_arrays[key]) for key in ('density_contrast','velocity_peculiar_km_s','actual_species_mass_msun')}
            if seed_fields is not None and any(not np.array_equal(fields[key],seed_fields[key]) for key in fields):raise ValueError('Native timestep family changes its measured initial fields')
            seed_fields={key:value.copy() for key,value in fields.items()}
            if force and steps:
                if summary['force_parameter_value']>=steps[-1]['force_parameter_value']:raise ValueError('Native force accuracy must tighten independently at each level')
                if control_parameter=='uniform_gravitational_softening':
                    previous=steps[-1]
                    if any(new>=old for new,old in zip(summary['softening_limits_native'],previous['softening_limits_native'])):
                        raise ValueError('Both native softening limits must decrease independently')
                    if not math.isclose(summary['softening_turnover_scale_factor'],previous['softening_turnover_scale_factor'],rel_tol=32*np.finfo(float).eps,abs_tol=0):
                        raise ValueError('Uniform softening refinement changes its physical/comoving turnover epoch')
                if summary['occupied_bin_history_sha256']!=steps[-1]['occupied_bin_history_sha256']:
                    raise ValueError('Native force controls change the actual logged timestep history')
            if not force and steps and (summary['configured_maximum_step_ln_a']>=steps[-1]['configured_maximum_step_ln_a'] or
                    any(new+1e-12>=old for new,old in zip(summary['maximum_occupied_step_ln_a'],steps[-1]['maximum_occupied_step_ln_a']))):
                raise ValueError('Native timestep family must decrease configured and realized gravity/hydro timestep widths')
            if timebin_rows and not np.array_equal(rows[0,5:],timebin_rows[0][0,5:]):raise ValueError('Native timestep runs change their occupied population census')
            steps.append(summary);timebin_rows.append(rows)
        if definition is not None and current!=definition:raise ValueError('Grid family changes native evolution, physical reference or declared comparison limits')
        definition=current;n=report['reference_grid_size']
        if type(n)is not int or n**3>request['maximum_grid_cells'] or grids and (n!=grids[-1] if temporal else n<=grids[-1]):raise ValueError('Native timestep controls require fixed grids; grid controls require strictly increasing resolutions')
        if report['native_grid_size']!=n:raise ValueError('Grid family measures native and reference on matching grids at each level')
        grids.append(n);mean=[];maximum=[]
        for field in ('density_difference','momentum_difference_km_s','velocity_difference_km_s','relative_velocity_difference_km_s'):
            value=np.asarray(arrays[field],float)
            if not np.isfinite(value).all():raise ValueError('Grid-family error support is nonfinite')
            if field=='density_difference':
                if value.shape!=(2,n,n,n):raise ValueError('Complete density error grid required')
                magnitude=np.abs(value)
            elif field=='relative_velocity_difference_km_s':
                if value.shape!=(3,n,n,n):raise ValueError('Complete relative-velocity error grid required')
                magnitude=np.linalg.norm(value,axis=0)[None]
            else:
                if value.shape!=(2,3,n,n,n):raise ValueError('Complete species vector error grid required')
                magnitude=np.linalg.norm(value,axis=1)
            mean.extend(magnitude.mean(axis=(1,2,3)).tolist());maximum.extend(magnitude.max(axis=(1,2,3)).tolist())
        means.append(mean);maxima.append(maximum)
        gates=[]
        for name in ('density_within_limits','momentum_within_limits','velocity_within_limits','relative_velocity_within_limits','native_dispersion_supported','species_mass_within_limits'):
            mask=np.asarray(arrays[name]);shape=(2,) if name=='species_mass_within_limits' else (n,n,n) if name=='relative_velocity_within_limits' else (2,n,n,n)
            if mask.dtype.kind!='b' or mask.shape!=shape:raise ValueError('Grid family lacks complete original support masks')
            gates.append(bool(mask.all()))
        accepted=all(gates)
        if 'pressure_limits' in report['request']:
            mask=np.asarray(arrays['pressure_within_limits'])
            if mask.dtype.kind!='b' or mask.shape!=(n,n,n):raise ValueError('Grid family lacks complete native pressure support masks')
            accepted &= bool(mask.all())
        if accepted!=(report['status']=='WITHIN_DECLARED_NATIVE_REFERENCE_LIMITS'):raise ValueError('Grid-family support status differs from actual masks')
        support.append(accepted)
    mean=np.asarray(means);maximum=np.asarray(maxima);floor=np.asarray([request['density_trend_floor']]*2+[request['velocity_trend_floor_km_s']]*5)
    mean_limit=request['maximum_adjacent_error_ratio']*mean[:-1]+floor;maximum_limit=request['maximum_adjacent_error_ratio']*maximum[:-1]+floor
    mean_ok=mean[1:]<=mean_limit;maximum_ok=maximum[1:]<=maximum_limit
    output={'grid_sizes':np.asarray(grids,np.int64),'mean_error':mean,'maximum_error':maximum,'mean_next_allowed':mean_limit,
        'maximum_next_allowed':maximum_limit,'mean_trend_supported':mean_ok,'maximum_trend_supported':maximum_ok,'individual_support':np.asarray(support,bool)}
    report={'schema':'measured-twofluid-native-grid-family-v1','execution_status':'COMPLETE','request':request,'request_base':str(Path(parameters['request_base']).resolve()),
        'coordinate_order':['density_cdm','density_baryons','momentum_cdm','momentum_baryons','velocity_cdm','velocity_baryons','relative_velocity'],
        'coordinate_units':['dimensionless']*2+['km / s']*5,'fixed_definition':definition,
        'status':'WITHIN_DECLARED_GRID_TRENDS' if all(support) and mean_ok.all() and maximum_ok.all() else 'GRID_TRENDS_UNRESOLVED',
        'scope':'same-native-run measurement/reference-grid error trends; no native-resolution convergence, continuum order or gas/force model equivalence',
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    if temporal:
        report.update(schema='measured-twofluid-native-timestep-family-v1',native_timestep_evidence=steps,
            status='WITHIN_DECLARED_NATIVE_TIMESTEP_TRENDS' if all(support) and mean_ok.all() and maximum_ok.all() else 'NATIVE_TIMESTEP_TRENDS_UNRESOLVED',
            scope='Independent native evolutions with identical launch IC bytes, binary, force/hydro parameters and fixed reference/grid; occupied-bin gravity/hydro timestep trends and dimensional native/reference errors. Reference discretization and model errors remain independent pending controls.')
        output.update(configured_maximum_step_ln_a=np.asarray([row['configured_maximum_step_ln_a'] for row in steps]),
            realized_maximum_occupied_step_ln_a=np.asarray([row['maximum_occupied_step_ln_a'] for row in steps]),
            realized_mean_occupied_step_ln_a=np.asarray([row['mean_occupied_step_ln_a'] for row in steps]))
        for index,rows in enumerate(timebin_rows):output['level_%03d_occupied_timebins'%index]=rows
        report['timebin_columns']=['scale_factor','maximum_gravity_step_ln_a','maximum_hydro_step_ln_a','mean_gravity_step_ln_a','mean_hydro_step_ln_a','gravity_count','gas_count']
        report['timebin_column_units']=['dimensionless']*7
        if force:
            output['native_force_parameter_values']=np.asarray([row['force_parameter_value'] for row in steps])
            report.update(schema='measured-twofluid-native-force-family-v1',force_parameter=control_parameter,
                status='WITHIN_DECLARED_NATIVE_FORCE_TRENDS' if all(support) and mean_ok.all() and maximum_ok.all() else 'NATIVE_FORCE_TRENDS_UNRESOLVED',
                scope='Independent relative tree-opening accuracy controls with identical IC bytes, binary, hydro/softening/PM/reference settings and complete logged interior occupied-bin histories. Printed-clock/width resolution and endpoint strips remain explicit; this does not establish PM/softening/total-force or continuum acceptance.')
            if hydro:
                output['native_hydro_parameter_values']=output.pop('native_force_parameter_values')
                report.pop('force_parameter')
                report.update(schema='measured-twofluid-native-hydro-family-v1',hydro_parameter=control_parameter,
                    status='WITHIN_DECLARED_HYDRO_VISCOSITY_TRENDS' if all(support) and mean_ok.all() and maximum_ok.all() else 'HYDRO_VISCOSITY_TRENDS_UNRESOLVED',
                    scope='Fixed-coefficient pressure-entropy SPH artificial-viscosity sensitivity with fixed gravity/reference definitions and logged occupied-bin histories. Decreasing viscosity is not spatial convergence or shock-domain acceptance; reference pressure/energy and per-particle timestep closure remain separate.')
            elif control_parameter=='uniform_gravitational_softening':
                output['softening_proper_endpoint_kpc']=np.asarray([row['softening_proper_endpoint_kpc'] for row in steps])
                output['softening_limits_native']=np.asarray([row['softening_limits_native'] for row in steps])
                report['force_parameter_units']='proper kpc at the target epoch'
                report['scope']='Independent uniform gravitational softening with fixed turnover epoch, native tree/PM/hydro/reference definitions and complete logged interior occupied-bin histories; does not establish PM-resolution, SPH smoothing, individual particle timesteps or continuum acceptance.'
            elif control_parameter=='PMGRID':
                output['pm_grid_sizes']=np.asarray([row['pm_grid_size'] for row in steps],np.int64)
                output['pm_cell_proper_kpc']=np.asarray([row['pm_cell_proper_kpc'] for row in steps])
                report['force_parameter_units']='inverse mesh cells per side'
                report['scope']='Independent PMGRID refinement with matched sealed native build definitions except PMGRID, fixed native/reference parameters and logged occupied-bin histories; fixed ASMTH, refined mesh/short-range split together. No isolated long-range-only error, particle-step, continuum or physical acceptance.'
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    specs={name:{'units':'dimensionless' if name.endswith(('supported','step_ln_a','occupied_timebins')) or name in {'grid_sizes','individual_support','native_force_parameter_values','native_hydro_parameter_values'} else 'per-coordinate units in coordinate_units','quantity':'native comparison-family '+name} for name in output}
    if force and control_parameter=='uniform_gravitational_softening':
        specs['native_force_parameter_values']['units']='proper kpc'
        specs['softening_proper_endpoint_kpc']['units']='proper kpc'
        specs['softening_limits_native']['units']='native code length; comoving and maximum-physical limits respectively'
    if force and control_parameter=='PMGRID':
        specs['pm_grid_sizes']['units']='dimensionless'
        specs['pm_cell_proper_kpc']['units']='proper kpc'
    return export_numeric_audit(destination,report,output,specs,algorithm=source,source_hashes=sources,
        context={'epoch':{'canonical_time':'scale_factor','actual_scale_factor':definition['epoch']},'coordinates':{'frame':'periodic Eulerian cell-average errors'}})


def twofluid_native_family_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Grid-family producer exceeds allocated memory')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Grid-family producer escapes its generation')
    return publish_twofluid_native_family(parameters,destination)


def twofluid_native_family_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Measure two-fluid reference-grid or independent native timestep/force-family trends')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text());parameters.setdefault('request_base',str(path.parent))
    publish_twofluid_native_family(parameters,args.output);return 0


def validate_twofluid_native_campaign_request(parameters):
    import copy
    request=parameters['request'];keys={'schema','seed_parameters','target_snapshots','grid_sizes','maximum_grids','maximum_targets',
        'reference_request','comparison_limits','resources','followup_limits'}
    if isinstance(request,dict) and 'family_limits' in request:keys.add('family_limits')
    if isinstance(request,dict) and 'pressure_history_limits' in request:keys.add('pressure_history_limits')
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='twofluid-native-comparison-campaign-request-v1':
        raise ValueError('Complete original-native two-fluid comparison campaign required')
    for name in ('maximum_grids','maximum_targets'):
        if type(request[name])is not int or request[name]<1:raise ValueError('Exact positive campaign family ceilings required')
    grids=request['grid_sizes'];targets=request['target_snapshots'];seed=request['seed_parameters']
    if not isinstance(grids,list) or not 1<=len(grids)<=request['maximum_grids'] or any(type(n)is not int or n<4 for n in grids) or any(b<=a for a,b in zip(grids,grids[1:])):
        raise ValueError('Explicit strictly increasing Eulerian comparison grids required')
    if not isinstance(targets,list) or not 1<=len(targets)<=request['maximum_targets'] or len(set(targets))!=len(targets) or any(type(n)is not int or n<0 or n==seed['request']['snapshot'] for n in targets):
        raise ValueError('Distinct bounded target snapshot labels separate from initial native epoch required')
    validate_twofluid_eulerian_seed_request(seed)
    if not {'native_run_directory','native_stage_receipt'}<=set(seed):raise ValueError('Actual completed native seed origin required')
    comparison={'schema':'twofluid-native-evolution-comparison-request-v1',**request['comparison_limits'],
        **{name:{'path':'pending','sha256':'0'*64} for name in ('initial_seed','reference','native_measurement')}}
    validate_twofluid_native_comparison_request({'request':comparison})
    if 'pressure_limits' in request['comparison_limits'] and not seed['request'].get('measure_pressure_support',False):
        raise ValueError('Native pressure comparison campaign must request original gas pressure measurements')
    if set(request['comparison_limits'])!=set(comparison)-{'schema','initial_seed','reference','native_measurement'}:
        raise ValueError('Campaign owns scientific input references; freeze only dimensional comparison limits')
    reference=request['reference_request']
    if not isinstance(reference,dict) or reference.get('schema')!='twofluid-nonlinear-reference-request-v1':raise ValueError('Full existing nonlinear physical/numerical reference request required')
    resources=request['resources'];stages=len(grids)*(1+3*len(targets));disk=0
    if 'pressure_history_limits' in request:
        history={'schema':'twofluid-native-pressure-history-request-v1',**request['pressure_history_limits'],
            'comparisons':[{'path':'pending_'+str(index),'sha256':'0'*64} for index in range(len(targets))]}
        validate_twofluid_pressure_history_request({'request':history})
        if set(request['pressure_history_limits'])!=set(history)-{'schema','comparisons'}:raise ValueError('Campaign owns pressure history comparison references')
        if not request['comparison_limits'].get('pressure_limits',{}).get('include_initial_pressure_support',False):raise ValueError('Pressure history campaign requires actual initial/target pressure measurements')
        if history['memory_gib']>resources['memory_gib'] or max(grids)**3>history['maximum_grid_cells']:raise MemoryError('Pressure history stage exceeds allocation')
        stages+=len(grids);disk+=sum(n**3*len(targets)*128 for n in grids)
    if 'family_limits' in request:
        check={'schema':'twofluid-native-comparison-family-request-v1',**request['family_limits'],
            'comparisons':[{'path':'pending','sha256':'0'*64} for _ in grids]}
        validate_twofluid_native_family_request({'request':check});stages+=len(targets)
        if set(request['family_limits'])!=set(check)-{'schema','comparisons'}:raise ValueError('Campaign owns family comparison references')
        if check['memory_gib']>resources['memory_gib'] or max(grids)**3>check['maximum_grid_cells']:raise MemoryError('Native grid-family stage exceeds allocation')
    for n in grids:
        current=copy.deepcopy(seed);current['request']['grid_size']=n;validate_twofluid_eulerian_seed_request(current)
        if n**3>min(reference['maximum_grid_cells'],comparison['maximum_grid_cells']):raise MemoryError('Native comparison grid exceeds stage allocation')
        disk+=n**3*(512+len(targets)*8192)
    if disk>resources['disk_gib']*1024**3 or stages>resources['maximum_stage_count'] or stages>request['followup_limits']['maximum_children'] or request['followup_limits']['maximum_depth']<(5 if 'family_limits' in request or 'pressure_history_limits' in request else 4):
        raise MemoryError('Complete native/reference comparison family exceeds disk/stage/follow-up bounds')
    if max(seed['request']['memory_gib'],reference['memory_gib'],comparison['memory_gib'])>resources['memory_gib']:
        raise MemoryError('Native comparison campaign stage exceeds root memory allocation')
    return request


def compile_twofluid_native_comparison_campaign(request,destination,*,request_base):
    """Compose actual native measurements and references at each saved epoch."""
    import copy
    from pathlib import Path
    from .foundation import atomic_json,file_sha256
    from .workflow import compile_requested_product_dag,default_registry
    request=validate_twofluid_native_campaign_request({'request':request});base=Path(request_base).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Retain previous native two-fluid campaign designs')
    seed=copy.deepcopy(request['seed_parameters'])
    for name in ('native_stage_receipt','native_run_directory'):seed[name]=str((base/seed[name]).resolve())
    sources=twofluid_eulerian_seed_input_bindings(seed);stages=[];requested=[];results=[]
    def field(stage,name):return {'stage':stage,'artifact':'measurement.json','load':'json','field':[name]}
    def product(stage,path):return {'path':{'stage':stage,'artifact':path},'sha256':{'stage':stage,'artifact':path,'load':'sha256'}}
    def add(identity,kind,parameters,path,dependencies=()):
        stages.append({'stage_id':identity,'kind':kind,'parameters':parameters,'depends_on':list(dependencies),'outputs':{identity:path}})
    for n in request['grid_sizes']:
        initial_id='initial_'+str(n);initial=copy.deepcopy(seed);initial['request']['grid_size']=n
        initial.update(destination='{generation}/measurement.json',expected_native_source_hashes=sources)
        add(initial_id,'twofluid_eulerian_seed',initial,'measurement.json')
        for snapshot in request['target_snapshots']:
            tag=str(n)+'_'+str(snapshot);native_id='native_'+tag;reference_id='reference_'+tag;comparison_id='comparison_'+tag
            native=copy.deepcopy(initial);native['request']['snapshot']=snapshot
            add(native_id,'twofluid_eulerian_seed',native,'measurement.json')
            model=copy.deepcopy(request['reference_request']);model['reference_scale_factor']=field(initial_id,'actual_scale_factor')
            model['target_scale_factor']=field(native_id,'actual_scale_factor')
            add(reference_id,'twofluid_nonlinear_reference',{'request':model,'initial_fields':product(initial_id,'measurement.json'),
                'request_base':str(base),'destination':'{generation}/reference.json'},'reference.json',[initial_id,native_id])
            comparison={'schema':'twofluid-native-evolution-comparison-request-v1',**copy.deepcopy(request['comparison_limits']),
                'initial_seed':product(initial_id,'measurement.json'),'native_measurement':product(native_id,'measurement.json'),
                'reference':product(reference_id,'reference.json')}
            add(comparison_id,'twofluid_native_comparison',{'request':comparison,'request_base':str(base),'destination':'{generation}/comparison.json'},
                'comparison.json',[initial_id,native_id,reference_id]);requested.append(comparison_id)
            results.append({'grid_size':n,'target_snapshot':snapshot,'comparison_stage':comparison_id})
    if 'pressure_history_limits' in request:
        for n in request['grid_sizes']:
            ids=['comparison_'+str(n)+'_'+str(snapshot) for snapshot in request['target_snapshots']]
            history={'schema':'twofluid-native-pressure-history-request-v1',**copy.deepcopy(request['pressure_history_limits']),
                'comparisons':[product(identity,'comparison.json') for identity in ids]}
            identity='pressure_history_'+str(n)
            add(identity,'twofluid_native_family',{'request':history,'request_base':str(base),'destination':'{generation}/history.json'},'history.json',ids)
            requested.append(identity)
    if 'family_limits' in request:
        for snapshot in request['target_snapshots']:
            ids=['comparison_'+str(n)+'_'+str(snapshot) for n in request['grid_sizes']]
            family={'schema':'twofluid-native-comparison-family-request-v1',**copy.deepcopy(request['family_limits']),
                'comparisons':[product(identity,'comparison.json') for identity in ids]}
            identity='family_'+str(snapshot)
            add(identity,'twofluid_native_family',{'request':family,'request_base':str(base),'destination':'{generation}/family.json'},'family.json',ids)
            requested.append(identity)
    workflow={'schema':'research-workflow-v1','resources':copy.deepcopy(request['resources']),'followup_limits':copy.deepcopy(request['followup_limits']),
        'stages':stages,'requested_products':requested}
    compile_requested_product_dag(workflow,default_registry(),request_base=base)
    destination.mkdir(parents=True);atomic_json(destination/'workflow.json',workflow)
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={'schema':'twofluid-native-comparison-campaign-design-v1','request':request,'workflow':str(destination/'workflow.json'),
        'workflow_sha256':file_sha256(destination/'workflow.json'),'stage_count':len(stages),'comparison_results':results,
        'source_hashes':sources,'execution_status':'NOT_EXECUTED','science_acceptance':'NOT_ESTABLISHED',
        'scope':'completed native full-box evolution -> per-grid exact initial measurements -> target references/native comparisons; no new native run or global convergence inference'}
    atomic_json(destination/'design.json',report);return report


def validate_twofluid_native_comparison_request(parameters):
    request=parameters['request']
    keys={'schema','initial_seed','reference','native_measurement','memory_gib','maximum_grid_cells','maximum_density_difference',
        'maximum_momentum_difference_km_s','maximum_velocity_difference_km_s','maximum_relative_velocity_difference_km_s',
        'maximum_native_velocity_dispersion_km_s','species_mass_relative_tolerance'}
    if isinstance(request,dict) and 'pressure_limits' in request:keys.add('pressure_limits')
    if not isinstance(request,dict) or set(request)!=keys or request['schema']!='twofluid-native-evolution-comparison-request-v1':
        raise ValueError('Complete original native/initial/reference two-fluid comparison required')
    for name in ('initial_seed','reference','native_measurement'):
        if not isinstance(request[name],dict) or set(request[name])!={'path','sha256'}:raise ValueError('Exact original source-bound two-fluid products required')
    if type(request['maximum_grid_cells'])is not int or request['maximum_grid_cells']<8:raise ValueError('Explicit comparison grid workspace ceiling required')
    if 'pressure_limits' in request:
        pressure=request['pressure_limits']
        required={'maximum_pressure_ratio_difference_km2_s2','maximum_pressure_ratio_spread_km2_s2'}
        selectors={'include_initial_pressure_support','include_energy_diagnostics','include_expansion_work_diagnostics'}
        if not isinstance(pressure,dict) or not required<=set(pressure) or set(pressure)-required-{'maximum_pressure_acceleration_difference_cm_s2',*selectors}:raise ValueError('Complete dimensional gas pressure limits required')
        if any(key in pressure and type(pressure[key])is not bool for key in selectors):raise ValueError('Pressure/energy diagnostic selectors require explicit booleans')
        if pressure.get('include_expansion_work_diagnostics',False) and not pressure.get('include_energy_diagnostics',False):raise ValueError('Expansion work requires original native energy/pressure diagnostics')
        for key,value in pressure.items():
            if key in selectors:continue
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Finite nonnegative gas pressure budget required')
    for name in keys-{'schema','initial_seed','reference','native_measurement','maximum_grid_cells','pressure_limits'}:
        value=request[name]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0 or name=='memory_gib' and value==0:
            raise ValueError('Finite explicit dimensional two-fluid budget required: '+name)
    if not 0<request['species_mass_relative_tolerance']<1:raise ValueError('Positive subunit extensive species mass tolerance required')
    if request['maximum_grid_cells']*4096>request['memory_gib']*1024**3:raise MemoryError('Complete native/reference comparison exceeds allocation')
    return request


def twofluid_native_comparison_input_bindings(parameters):
    from .sidm_analysis import _load_product
    request=validate_twofluid_native_comparison_request(parameters);sources={}
    for name in ('initial_seed','reference','native_measurement'):
        _,_,binding=_load_product(request[name],parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
    return sources


def _twofluid_pressure_domain_binding(native_model,reference_model,supported):
    """Validate the sampled EOS definition; never infer energy/history closure."""
    if native_model is not None:
        required={'eos','adiabatic_index','pressure_ratio_mapping','source_flags'}
        if not isinstance(native_model,dict) or not required<=set(native_model):raise ValueError('Complete native pressure-model definition required')
        gamma=native_model['adiabatic_index'];flags=native_model['source_flags']
        if type(gamma)not in (int,float) or not math.isfinite(gamma) or not isinstance(flags,list) or any(not isinstance(flag,str) or not flag for flag in flags) or len(flags)!=len(set(flags)):
            raise ValueError('Native pressure-model caloric index or source flags are invalid')
        if native_model['eos']=='native_isothermal':
            valid=gamma==1 and native_model['pressure_ratio_mapping']=='stored_u'
        elif native_model['eos']=='native_caloric':
            valid=1<gamma<=2 and native_model['pressure_ratio_mapping']=='(gamma_pressure-1)*stored_u'
        else:valid=False
        if not valid:raise ValueError('Native pressure-model mapping and EOS disagree')
    state=('NATIVE_MODEL_NOT_RECORDED' if native_model is None else
        'SAMPLED_ISOTHERMAL_PRESSURE_LAW_SUPPORTED' if native_model['eos']=='native_isothermal' and not native_model['source_flags'] and reference_model=='homogeneous_isothermal_baryons' and supported else
        'PRESSURE_MODEL_CLOSURE_UNRESOLVED')
    return {'native_model':native_model,'reference_model':reference_model,'state':state,
        'scope':'Actual target-epoch pressure-law samples only; initial/history coefficient homogeneity, energy/source closure and native SPH forces remain separate requirements'}


def publish_twofluid_native_comparison(parameters,destination):
    """Measure native Eulerian mass/momentum against the nonlinear reference."""
    from pathlib import Path
    from .sidm_analysis import _load_product
    from .foundation import file_sha256
    from .data import export_numeric_audit
    request=validate_twofluid_native_comparison_request(parameters);sources={};products={}
    for name in ('initial_seed','reference','native_measurement'):
        report,arrays,binding=_load_product(request[name],parameters['request_base'],request['memory_gib']*1024**3/4)
        sources.update(binding);products[name]=(report,arrays)
    initial,iv=products['initial_seed'];reference,rv=products['reference'];native,nv=products['native_measurement']
    if initial.get('schema')!='measured-twofluid-eulerian-seed-v1' or native.get('schema')!='measured-twofluid-eulerian-seed-v1' or reference.get('schema')!='finite-epoch-nonlinear-twofluid-reference-v1':
        raise ValueError('Actual full nonlinear reference and original complete native cell measurements required')
    if reference.get('initial_fields_reference',{}).get('sha256')!=request['initial_seed']['sha256']:
        raise ValueError('Native comparison changes the actual nonlinear initial seed')
    if initial.get('native_run_directory')!=native.get('native_run_directory') or not initial.get('native_run_directory') or initial.get('native_stage_receipt')!=native.get('native_stage_receipt'):
        raise ValueError('Initial and evolved native measurements must belong to the same actual completed native evolution')
    if initial['particle_namespace']!=native['particle_namespace']:raise ValueError('Native two-fluid evolution changes its original identity namespace')
    if initial['actual_scale_factor']!=reference['reference_scale_factor'] or native['actual_scale_factor']!=reference['target_scale_factor']:
        raise ValueError('Native/reference initial and evolved physical epochs differ')
    for name in ('box_mpc_h','baryon_fraction','hubble_parameter','omega_m0','cell_convention','species_order'):
        expected=initial[name]
        measured=reference.get(name,reference.get('request',{}).get(name))
        if measured!=expected or native[name]!=expected:raise ValueError('Native/reference species, units, grid or matter definition differs: '+name)
    for report,arrays in products.values():
        for field,unit in (('density_contrast','dimensionless'),('velocity_peculiar_km_s','km / s')):
            if report['numeric_data']['array_contract']['datasets'][field]['units']!=unit:raise ValueError('Native/reference field units differ')
    def state(arrays):
        density=np.asarray(arrays['density_contrast'],float)+1;velocity=np.asarray(arrays['velocity_peculiar_km_s'],float)
        n=density.shape[-1] if density.ndim==4 else 0
        if n<2 or n**3>request['maximum_grid_cells'] or density.shape!=(2,n,n,n) or velocity.shape!=(2,3,n,n,n) or not np.isfinite(density).all() or not np.isfinite(velocity).all() or np.any(density<=0):
            raise ValueError('Complete positive cell-average species state required')
        return density,velocity,n
    ir,iu,ni=state(iv);rr,ru,n=state(rv);nr,nu,nn=state(nv)
    if ni!=n or not np.array_equal(iv['density_contrast'],rv['initial_density_contrast']) or not np.array_equal(iu,rv['initial_velocity_peculiar_km_s']):
        raise ValueError('Reference does not retain the original exact measured seed grid/state')
    if nn<n:raise ValueError('Native comparison needs the same or a finer Eulerian measurement grid')
    dispersion=np.asarray(nv['velocity_dispersion_roundoff_upper_km_s'],float)
    counts=np.asarray(nv['particle_count_per_cell'])
    if dispersion.shape!=nr.shape or not np.isfinite(dispersion).all() or np.any(dispersion<0) or counts.shape!=nr.shape or counts.dtype.kind not in 'iu' or np.any(counts<1):
        raise ValueError('Native comparison lacks actual supported species cell counts/dispersion')
    def restrict(value):
        return restrict_twofluid_cell_averages(value,n)
    measured_density=restrict(nr);measured_momentum=restrict(nr[:,None]*nu);measured_velocity=measured_momentum/measured_density[:,None]
    difference_density=measured_density-rr;difference_momentum=measured_momentum-rr[:,None]*ru;difference_velocity=measured_velocity-ru
    relative=(measured_velocity[1]-measured_velocity[0])-(ru[1]-ru[0])
    density_ok=np.abs(difference_density)<=request['maximum_density_difference']
    momentum_ok=np.linalg.norm(difference_momentum,axis=1)<=request['maximum_momentum_difference_km_s']
    velocity_ok=np.linalg.norm(difference_velocity,axis=1)<=request['maximum_velocity_difference_km_s']
    relative_ok=np.linalg.norm(relative,axis=0)<=request['maximum_relative_velocity_difference_km_s']
    second=restrict(nr*(dispersion**2+np.sum(nu*nu,axis=1)))/measured_density
    bulk=np.sum(measured_velocity*measured_velocity,axis=1);variance=second-bulk
    roundoff=128*np.finfo(float).eps*np.maximum(second,bulk)
    if not np.isfinite(variance).all() or np.any(variance < -roundoff):raise ValueError('Conservative native dispersion restriction loses second-moment support')
    coarse_dispersion=np.sqrt(np.maximum(variance,0)+roundoff)
    dispersion_ok=coarse_dispersion<=request['maximum_native_velocity_dispersion_km_s']
    mass_initial=np.asarray(iv['actual_species_mass_msun'],float);mass_native=np.asarray(nv['actual_species_mass_msun'],float)
    if mass_initial.shape!=(2,) or mass_native.shape!=(2,) or not np.isfinite(mass_initial).all() or not np.isfinite(mass_native).all() or np.any(mass_initial<=0) or np.any(mass_native<=0):
        raise ValueError('Actual original/native extensive species masses required')
    mass_difference=mass_native/mass_initial-1;mass_ok=np.abs(mass_difference)<=request['species_mass_relative_tolerance']
    output={'density_difference':difference_density,'momentum_difference_km_s':difference_momentum,'velocity_difference_km_s':difference_velocity,
        'relative_velocity_difference_km_s':relative,'density_within_limits':density_ok,'momentum_within_limits':momentum_ok,
        'velocity_within_limits':velocity_ok,'relative_velocity_within_limits':relative_ok,'native_dispersion_supported':dispersion_ok,
        'native_dispersion_upper_km_s':coarse_dispersion,
        'species_mass_relative_difference':mass_difference,'species_mass_within_limits':mass_ok,
        'density_mean_absolute_difference':np.mean(np.abs(difference_density),axis=(1,2,3)),
        'momentum_mean_norm_difference_km_s':np.mean(np.linalg.norm(difference_momentum,axis=1),axis=(1,2,3)),
        'velocity_mean_norm_difference_km_s':np.mean(np.linalg.norm(difference_velocity,axis=1),axis=(1,2,3))}
    pressure_ok=None
    if 'pressure_limits' in request:
        from scipy.interpolate import CubicSpline
        for name in ('baryon_pressure_ratio_km2_s2','baryon_pressure_ratio_spread_km2_s2','baryon_pressure_support_available'):
            if name not in nv or nv[name].shape!=(nn,nn,nn):raise ValueError('Native gas pressure measurement at the comparison epoch required')
        available=np.asarray(nv['baryon_pressure_support_available']);p=np.asarray(nv['baryon_pressure_ratio_km2_s2'],float);spread=np.asarray(nv['baryon_pressure_ratio_spread_km2_s2'],float)
        if available.dtype.kind!='b' or not np.isfinite(p).all() or not np.isfinite(spread).all() or np.any(p<0) or np.any(spread<0):raise ValueError('Native pressure support is invalid')
        if nn!=n:raise ValueError('Pressure spread comparison requires matched native/reference cell grids')
        for name in ('baryon_pressure_ratio_km2_s2','baryon_pressure_ratio_spread_km2_s2'):
            if native['numeric_data']['array_contract']['datasets'][name]['units']!='(km / s)^2':raise ValueError('Native pressure-ratio units differ')
        model=reference['request'];sound=float(CubicSpline(np.log(model['background_scale_factor']),model['baryon_sound_speed_km_s'],extrapolate=False)(math.log(native['actual_scale_factor'])))
        if not math.isfinite(sound) or sound<0:raise ValueError('Reference pressure history is unavailable at the measured epoch')
        pressure_difference=p-sound**2;limits=request['pressure_limits']
        pressure_ok=available&(np.abs(pressure_difference)<=limits['maximum_pressure_ratio_difference_km2_s2'])&(spread<=limits['maximum_pressure_ratio_spread_km2_s2'])
        output.update(pressure_ratio_difference_km2_s2=pressure_difference,native_pressure_ratio_spread_km2_s2=spread,
            native_pressure_support_available=available,pressure_within_limits=pressure_ok)
        if 'maximum_pressure_acceleration_difference_cm_s2' in limits:
            from .foundation import KPC_CM
            proper_dx=reference['box_mpc_h']*1000*KPC_CM*native['actual_scale_factor']/model['hubble_parameter']/n
            if not math.isfinite(proper_dx) or proper_dx<=0:raise ValueError('Actual native/reference proper pressure-gradient cell scale invalid')
            native_pressure=nr[1]*p;reference_pressure=rr[1]*sound**2
            native_acceleration=np.empty((3,n,n,n));reference_acceleration=np.empty_like(native_acceleration);stencil_available=available.copy()
            for axis in range(3):
                stencil_available &= np.roll(available,1,axis=axis)&np.roll(available,-1,axis=axis)
                native_acceleration[axis]=-(np.roll(native_pressure,-1,axis=axis)-np.roll(native_pressure,1,axis=axis))*1e10/(2*proper_dx*nr[1])
                reference_acceleration[axis]=-(np.roll(reference_pressure,-1,axis=axis)-np.roll(reference_pressure,1,axis=axis))*1e10/(2*proper_dx*rr[1])
            if not np.isfinite(native_acceleration).all() or not np.isfinite(reference_acceleration).all():raise ArithmeticError('Cell pressure acceleration overflows')
            force_difference=native_acceleration-reference_acceleration
            force_ok=stencil_available&(np.linalg.norm(force_difference,axis=0)<=limits['maximum_pressure_acceleration_difference_cm_s2'])
            pressure_ok &= force_ok
            output.update(native_pressure_acceleration_cm_s2=np.where(stencil_available[None],native_acceleration,0.),
                reference_pressure_acceleration_cm_s2=reference_acceleration,
                pressure_acceleration_difference_cm_s2=np.where(stencil_available[None],force_difference,0.),
                native_pressure_stencil_available=stencil_available,pressure_acceleration_within_limits=force_ok,
                pressure_within_limits=pressure_ok)
            if 'baryon_stored_hydro_acceleration_cm_s2' in nv:
                stored=np.asarray(nv['baryon_stored_hydro_acceleration_cm_s2'],float)
                stored_available=np.asarray(nv['baryon_stored_hydro_acceleration_available'])
                if stored.shape!=(3,n,n,n) or stored_available.shape!=(n,n,n) or stored_available.dtype.kind!='b' or not np.isfinite(stored).all():raise ValueError('Stored native hydro acceleration grid/availability is invalid')
                if native['numeric_data']['array_contract']['datasets']['baryon_stored_hydro_acceleration_cm_s2']['units']!='cm / s^2':raise ValueError('Stored native hydro acceleration units differ')
                epoch_names=('baryon_hydro_force_evaluation_epoch_known','baryon_hydro_force_evaluation_epoch_matches_snapshot')
                if set(epoch_names)&set(nv):
                    if not set(epoch_names)<=set(nv):raise ValueError('Native hydro evaluation clock masks are incomplete')
                    known,matched=(np.asarray(nv[name]) for name in epoch_names)
                    if any(mask.shape!=(n,n,n) or mask.dtype.kind!='b' for mask in (known,matched)) or np.any(matched&~known) or np.any(matched&~stored_available):raise ValueError('Native hydro evaluation clock masks are inconsistent')
                    output.update(native_hydro_evaluation_epoch_known=known,native_hydro_evaluation_epoch_matches_snapshot=matched)
                diagnostic_available=stored_available&stencil_available
                output.update(stored_hydro_reference_acceleration_difference_cm_s2=np.where(diagnostic_available[None],stored-reference_acceleration,0.),
                    stored_hydro_gradient_diagnostic_available=diagnostic_available,
                    native_hydro_force_comparison_supported=np.zeros((n,n,n),bool))
    if pressure_ok is not None and request['pressure_limits'].get('include_initial_pressure_support',False):
        limits=request['pressure_limits'];shape=(n,n,n)
        initial_model=initial.get('native_pressure_model');native_model=native.get('native_pressure_model')
        if initial_model is None or native_model is None:raise ValueError('Initial pressure closure requires source-recorded EOS definitions at both epochs')
        _twofluid_pressure_domain_binding(initial_model,reference['request']['pressure_model'],False)
        _twofluid_pressure_domain_binding(native_model,reference['request']['pressure_model'],False)
        for name in ('eos','adiabatic_index','pressure_ratio_mapping','source_flags'):
            if initial_model[name]!=native_model[name]:raise ValueError('Initial/evolved pressure model definitions differ: '+name)
        for name in ('baryon_pressure_ratio_km2_s2','baryon_pressure_ratio_spread_km2_s2','baryon_pressure_support_available'):
            if name not in iv or np.asarray(iv[name]).shape!=shape:raise ValueError('Complete initial pressure measurement grid required')
        available=np.asarray(iv['baryon_pressure_support_available']);p=np.asarray(iv['baryon_pressure_ratio_km2_s2'],float)
        spread=np.asarray(iv['baryon_pressure_ratio_spread_km2_s2'],float)
        if available.dtype.kind!='b' or not np.isfinite(p).all() or not np.isfinite(spread).all() or np.any(p<0) or np.any(spread<0):raise ValueError('Initial pressure support is invalid')
        for name in ('baryon_pressure_ratio_km2_s2','baryon_pressure_ratio_spread_km2_s2'):
            if initial['numeric_data']['array_contract']['datasets'][name]['units']!='(km / s)^2':raise ValueError('Initial pressure-ratio units differ')
        model=reference['request'];sound=float(CubicSpline(np.log(model['background_scale_factor']),model['baryon_sound_speed_km_s'],extrapolate=False)(math.log(initial['actual_scale_factor'])))
        if not math.isfinite(sound) or sound<0:raise ValueError('Reference initial pressure history is unavailable')
        difference=p-sound**2
        initial_ok=available&(np.abs(difference)<=limits['maximum_pressure_ratio_difference_km2_s2'])&(spread<=limits['maximum_pressure_ratio_spread_km2_s2'])
        output.update(initial_pressure_ratio_difference_km2_s2=difference,initial_pressure_ratio_spread_km2_s2=spread,
            initial_pressure_support_available=available,initial_pressure_within_limits=initial_ok,
            target_pressure_within_limits=pressure_ok.copy())
        pressure_ok=pressure_ok&initial_ok;output['pressure_within_limits']=pressure_ok
    if request.get('pressure_limits',{}).get('include_energy_diagnostics',False):
        from .foundation import MSUN_G
        factor=MSUN_G*1e10
        native_cell_mass=mass_native[:,None,None,None]*measured_density/measured_density.sum(axis=(1,2,3))[:,None,None,None]
        reference_cell_mass=mass_initial[:,None,None,None]*rr/rr.sum(axis=(1,2,3))[:,None,None,None]
        native_kinetic=.5*native_cell_mass*second*factor
        reference_kinetic=.5*reference_cell_mass*np.sum(ru*ru,axis=1)*factor
        output.update(native_kinetic_energy_erg=native_kinetic,reference_bulk_kinetic_energy_erg=reference_kinetic,
            kinetic_energy_difference_erg=native_kinetic-reference_kinetic)
        expansion_bindings={}
        for label,record,values,size in (('initial',initial,iv,ni),('target',native,nv,nn)):
            if not {'baryon_thermal_energy_erg','baryon_thermal_energy_available'}<=set(values):raise ValueError('Original native thermal-energy measurements are missing')
            energy=np.asarray(values['baryon_thermal_energy_erg'],float);available=np.asarray(values['baryon_thermal_energy_available'])
            if energy.shape!=(size,size,size) or available.shape!=energy.shape or available.dtype.kind!='b' or not np.isfinite(energy).all() or np.any(energy<0):raise ValueError('Complete native caloric thermal-energy grid and availability required')
            if record['numeric_data']['array_contract']['datasets']['baryon_thermal_energy_erg']['units']!='erg':raise ValueError('Native thermal-energy units differ')
            output[label+'_baryon_thermal_energy_erg']=np.where(available,energy,0.);output[label+'_baryon_thermal_energy_available']=available.copy()
            fields={'baryon_peculiar_pressure_work_erg_s','baryon_pressure_work_available'}
            if fields & set(values):
                if not fields<=set(values):raise ValueError('Native pressure-work measurement is incomplete')
                rate=np.asarray(values['baryon_peculiar_pressure_work_erg_s'],float);work_available=np.asarray(values['baryon_pressure_work_available'])
                if rate.shape!=(size,size,size) or work_available.shape!=rate.shape or work_available.dtype.kind!='b' or not np.isfinite(rate).all():raise ValueError('Native pressure-work grid or availability is invalid')
                if record['numeric_data']['array_contract']['datasets']['baryon_peculiar_pressure_work_erg_s']['units']!='erg / s':raise ValueError('Native pressure-work units differ')
                output[label+'_baryon_peculiar_pressure_work_erg_s']=np.where(work_available,rate,0.)
                output[label+'_baryon_pressure_work_available']=work_available.copy()
            if request['pressure_limits'].get('include_expansion_work_diagnostics',False):
                from .foundation import MPC_CM
                fields={'baryon_pressure_ratio_km2_s2','baryon_pressure_support_available'}
                if not fields<=set(values):raise ValueError('Original endpoint pressure measurements required for expansion work')
                pressure=np.asarray(values['baryon_pressure_ratio_km2_s2'],float);pressure_available=np.asarray(values['baryon_pressure_support_available'])
                if pressure.shape!=(size,size,size) or pressure_available.shape!=pressure.shape or pressure_available.dtype.kind!='b' or not np.isfinite(pressure).all() or np.any(pressure<0):raise ValueError('Expansion work pressure field/availability is invalid')
                if record['numeric_data']['array_contract']['datasets']['baryon_pressure_ratio_km2_s2']['units']!='(km / s)^2':raise ValueError('Expansion work pressure units differ')
                model=reference['request'];a=np.asarray(model['background_scale_factor'],float);hubble=np.asarray(model['background_hubble_km_s_mpc'],float)
                if a.ndim!=1 or len(a)<4 or hubble.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(hubble).all() or np.any(a<=0) or np.any(np.diff(a)<=0) or np.any(hubble<=0):raise ValueError('Complete physical reference expansion history required')
                epoch=record['actual_scale_factor']
                if not a[0]<=epoch<=a[-1]:raise ValueError('Expansion work epoch is outside the supplied background')
                H=float(np.exp(CubicSpline(np.log(a),np.log(hubble),extrapolate=False)(math.log(epoch))))*1e5/MPC_CM
                density=np.asarray(values['density_contrast'],float)[1]+1
                mass=float(np.asarray(values['actual_species_mass_msun'])[1])
                cell_mass=mass*density/density.sum()
                expansion=-3*H*cell_mass*factor*pressure
                output[label+'_reference_background_expansion_work_erg_s']=np.where(pressure_available,expansion,0.)
                output[label+'_expansion_work_available']=pressure_available.copy()
                native_model=record.get('native_expansion_model',{'state':'MISSING_NATIVE_DEFINITION'})
                native_H=_twofluid_native_hubble_s_inverse(native_model,float(epoch))
                expansion_bindings[label]={'native_model':native_model,'reference_hubble_s_inverse':H,
                    'state':'NATIVE_DEFINITION_NOT_RECORDED' if native_H is None else 'RECORDED_NATIVE_BACKGROUND_EVALUATED'}
                if native_H is not None:
                    expansion_bindings[label]['native_hubble_s_inverse']=native_H
                    native_expansion=-3*native_H*cell_mass*factor*pressure
                    output[label+'_native_background_expansion_work_erg_s']=np.where(pressure_available,native_expansion,0.)
                    output[label+'_native_reference_hubble_relative_difference']=np.asarray(native_H/H-1)
                    if label+'_baryon_peculiar_pressure_work_erg_s' in output:
                        rate_available=pressure_available&output[label+'_baryon_pressure_work_available']
                        output[label+'_native_pressure_work_rate_erg_s']=np.where(rate_available,native_expansion+output[label+'_baryon_peculiar_pressure_work_erg_s'],0.)
                        output[label+'_native_pressure_work_rate_available']=rate_available
    if any(not np.isfinite(value).all() for value in output.values()):raise ArithmeticError('Native/reference comparison accumulation overflows')
    supported=all(value.all() for value in (density_ok,momentum_ok,velocity_ok,relative_ok,dispersion_ok,mass_ok)) and (pressure_ok is None or pressure_ok.all())
    report={'schema':'measured-twofluid-native-evolution-comparison-v1','execution_status':'COMPLETE','request':request,
        'request_base':str(Path(parameters['request_base']).resolve()),'species_order':['cdm','baryons'],'actual_scale_factor':native['actual_scale_factor'],
        'reference_grid_size':n,'native_grid_size':nn,'restriction':'aligned cell-volume overlap mean density and density-weighted momentum, including nonnested grids; velocity recovered after restriction',
        'momentum_definition':'species normalized density ratio times peculiar velocity, in km/s; not extensive mass momentum',
        'status':'WITHIN_DECLARED_NATIVE_REFERENCE_LIMITS' if supported else 'NATIVE_REFERENCE_UNRESOLVED',
        'physical_domain':'declared single-stream pressureless/homogeneous-isothermal reference; native gas pressure/energy closure and multistream accuracy require separate evidence',
        'scope':'same-run native cell averages and exact-seed reference differences; no inherited physical model equivalence or global accuracy',
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    if pressure_ok is not None:
        report['pressure_support_scope']='actual sampled native EOS P/rho and within-cell spread versus declared reference pressure coefficient c_s^2; not a native acoustic-speed, pressure-gradient or gas-energy closure proof'
        native_model=native.get('native_pressure_model')
        report['pressure_model_binding']=_twofluid_pressure_domain_binding(native_model,reference['request']['pressure_model'],bool(pressure_ok.all()))
        if request['pressure_limits'].get('include_initial_pressure_support',False):
            report['initial_pressure_model_binding']=_twofluid_pressure_domain_binding(initial['native_pressure_model'],reference['request']['pressure_model'],bool(output['initial_pressure_within_limits'].all()))
            report['pressure_epoch_scope']='Initial and target sampled pressure coefficients/spreads with consistent EOS definitions; target-only cell-gradient checks; unsaved/intermediate history and energy/source/SPH-force closure remain pending'
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    if request.get('pressure_limits',{}).get('include_energy_diagnostics',False):
        report['energy_scope']='Actual native species kinetic cell energy includes measured velocity second moments; reference bulk kinetic energy has no unresolved dispersion. Native caloric thermal energy is saved with availability. Isothermal reference lacks an evolved thermodynamic energy/source model; no thermal-reference equality, energy conservation, pressure-work/source or global acceptance is claimed.'
        report['energy_closure_status']='DIAGNOSTICS_ONLY_CLOSURE_PENDING'
        report['energy_grid_sizes']={'initial_thermal':ni,'target_thermal':nn,'kinetic_comparison':n}
        report['pressure_work_status']='SAVED_PECULIAR_RATES_ONLY_EXPANSION_SOURCES_AND_TIME_INTEGRAL_PENDING' if 'target_baryon_peculiar_pressure_work_erg_s' in output else 'ORIGINAL_NATIVE_PRESSURE_WORK_NOT_RECORDED'
        if request['pressure_limits'].get('include_expansion_work_diagnostics',False):
            report['expansion_work_scope']='-3H times actual measured gas-cell mass and P/rho at original endpoints, using the supplied reference log-H spline. Native/reference expansion-background equivalence remains unverified; this is a reference-background diagnostic, not native total pressure-work closure.'
            report['pressure_work_status']='SAVED_RATES_AND_REFERENCE_BACKGROUND_EXPANSION_DIAGNOSTICS_NATIVE_BACKGROUND_SOURCE_AND_TIME_CLOSURE_PENDING'
            report['native_background_work_bindings']=expansion_bindings
            report['native_background_work_scope']='Native H is reconstructed from original saved Hubble, units, h and density parameters or hash-bound native log-linear E table. Relative discrepancies with reference H are retained without a tolerance/acceptance inference. Combined native rates are peculiar plus expansion pressure work only; source heat, cadence integration and thermodynamic closure remain pending.'
    specs={name:{'units':'erg / s' if name.endswith('_erg_s') else 'erg' if name.endswith('_erg') else 'cm / s^2' if name.endswith('cm_s2') else '(km / s)^2' if name.endswith('km2_s2') else 'km / s' if name.endswith('km_s') else 'dimensionless','quantity':'measured native/reference '+name} for name in output}
    for label in ('initial','target'):
        if label+'_baryon_thermal_energy_erg' in specs:specs[label+'_baryon_thermal_energy_erg']['missing_policy']='zero where '+label+'_baryon_thermal_energy_available is false; unavailable is not measured zero thermal energy'
        if label+'_baryon_peculiar_pressure_work_erg_s' in specs:specs[label+'_baryon_peculiar_pressure_work_erg_s']['missing_policy']='zero where '+label+'_baryon_pressure_work_available is false; unavailable is not measured zero work'
        if label+'_reference_background_expansion_work_erg_s' in specs:specs[label+'_reference_background_expansion_work_erg_s']['missing_policy']='zero where '+label+'_expansion_work_available is false; unavailable is not measured zero expansion work'
        if label+'_native_background_expansion_work_erg_s' in specs:specs[label+'_native_background_expansion_work_erg_s']['missing_policy']='zero where '+label+'_expansion_work_available is false; unavailable is not measured zero work'
        if label+'_native_pressure_work_rate_erg_s' in specs:specs[label+'_native_pressure_work_rate_erg_s']['missing_policy']='zero where '+label+'_native_pressure_work_rate_available is false; unavailable is not measured zero work'
    if 'pressure_acceleration_difference_cm_s2' in output:
        report['pressure_gradient_scope']='periodic centred cell gradient of normalized mass-weighted caloric pressure; actual proper cell length; no SPH force or dissipative Rusanov flux equivalence'
        for name in ('native_pressure_acceleration_cm_s2','pressure_acceleration_difference_cm_s2'):
            specs[name]['missing_policy']='numeric zero outside native_pressure_stencil_available; unavailable is not measured zero acceleration'
    if 'stored_hydro_reference_acceleration_difference_cm_s2' in output:
        report['native_sph_force_comparison_status']=('CURRENT_FORCE_STATE_FRAME_NOT_ESTABLISHED' if np.any(output.get('native_hydro_evaluation_epoch_known',False)) else 'CURRENT_FORCE_EPOCH_NOT_ESTABLISHED')
        report['native_sph_force_comparison_scope']='Actual mass-weighted stored native SPH acceleration minus reference cell-pressure acceleration, including native viscosity. Stored-field availability is distinct from current force-epoch support, which remains false until native evaluation-frame evidence is supplied.'
        report['native_hydro_acceleration_frame']=native.get('native_hydro_acceleration_frame',{'frame_status':'NOT_RECORDED'})
        specs['stored_hydro_reference_acceleration_difference_cm_s2']['missing_policy']='zero outside stored_hydro_gradient_diagnostic_available; native_hydro_force_comparison_supported remains false because the force state frame is unproven'
    return export_numeric_audit(destination,report,output,specs,algorithm=source,source_hashes=sources,
        context={'epoch':reference['epoch'],'coordinates':{'frame':'periodic Eulerian cell averages; species/component-first peculiar velocity'}})


def twofluid_native_comparison_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native two-fluid comparison exceeds allocated memory')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Native two-fluid comparison escapes its generation')
    return publish_twofluid_native_comparison(parameters,destination)


def twofluid_native_comparison_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Compare original evolved native two-fluid cells with a nonlinear reference')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text());parameters.setdefault('request_base',str(path.parent))
    publish_twofluid_native_comparison(parameters,args.output);return 0


def validate_twofluid_refinement_request(parameters):
    request=parameters['request']
    keys={'schema','levels','maximum_levels','maximum_grid_cells','memory_gib','density_ratio_floor','momentum_floor_km_s',
        'maximum_initial_density_error','maximum_initial_momentum_error','maximum_density_error','maximum_momentum_error',
        'maximum_relative_velocity_error_km_s','maximum_adjacent_error_ratio'}
    if not isinstance(request,dict):raise ValueError('Nonlinear refinement request must be an object')
    if request.get('schema') in ('twofluid-nonlinear-refinement-request-v2','twofluid-nonlinear-refinement-request-v3','twofluid-nonlinear-refinement-request-v4'):
        keys.add('refinement_axis')
        allowed_axes=('dissipation',) if request['schema']=='twofluid-nonlinear-refinement-request-v4' else ('grid','timestep')
        if request.get('refinement_axis')not in allowed_axes:raise ValueError('Select explicit supported grid, timestep or dissipation comparison')
        if request['schema']in ('twofluid-nonlinear-refinement-request-v3','twofluid-nonlinear-refinement-request-v4'):
            keys.update(('material_chunk_rows','maximum_material_displacement_error_mpc_h','maximum_material_velocity_error_km_s',
                'maximum_material_deformation_error','maximum_material_jacobian_error'))
    elif request.get('schema')!='twofluid-nonlinear-refinement-request-v1':raise ValueError('Unsupported nonlinear refinement request version')
    if set(request)!=keys:raise ValueError('Complete nonlinear refinement request required')
    for name,minimum in (('maximum_levels',3),('maximum_grid_cells',8)):
        if type(request[name])is not int or request[name]<minimum:raise ValueError('Exact bounded nonlinear refinement work required')
    if not isinstance(request['levels'],list) or not 3<=len(request['levels'])<=request['maximum_levels']:raise ValueError('At least three actual nonlinear grid resolutions required')
    if 'material_chunk_rows' in request and (type(request['material_chunk_rows'])is not int or not 1<=request['material_chunk_rows']<=request['maximum_grid_cells']):
        raise ValueError('Bounded positive material sampling chunk required')
    for name in keys-{'schema','levels','maximum_levels','maximum_grid_cells','refinement_axis','material_chunk_rows'}:
        value=request[name]
        if type(value)not in (int,float) or not math.isfinite(value) or value<0 or name in ('memory_gib','density_ratio_floor','momentum_floor_km_s','maximum_adjacent_error_ratio') and value<=0:
            raise ValueError('Finite dimensional/nonnegative refinement budget required: '+name)
    if request['maximum_adjacent_error_ratio']>1:raise ValueError('Refinement error-growth gate must be at most one')
    if request['maximum_grid_cells']*4096>request['memory_gib']*1024**3/2:raise MemoryError('Complete adjacent nonlinear product/restriction workspace exceeds allocation')
    return request


def twofluid_refinement_input_bindings(parameters):
    from core.sidm_analysis import _load_product
    request=validate_twofluid_refinement_request(parameters);sources={}
    for reference in request['levels']:sources.update(_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/4)[2])
    return sources


def restrict_twofluid_cell_averages(value, coarse_grid_size):
    """Volume-overlap restriction of aligned zero-origin periodic cubic cells.

    The input represents piecewise constant cell averages, not point samples.
    Integer endpoints on a common n*m lattice avoid geometric tolerance tests.
    The nested path retains the existing block-mean implementation exactly.
    """
    raw=np.asarray(value)
    if raw.ndim<3 or len(set(raw.shape[-3:]))!=1 or raw.dtype.kind not in 'fiu':
        raise ValueError('Finite real cubic cell-average fields required for conservative restriction')
    fine=raw.shape[-1];n=coarse_grid_size
    if type(n)is not int or not 1<=n<=fine or not np.isfinite(raw).all():
        raise ValueError('Positive exact target size no finer than the finite source grid required')
    data=np.asarray(raw,dtype=float)
    if n==fine:return data
    if fine%n==0:
        ratio=fine//n
        result=data.reshape((*data.shape[:-3],n,ratio,n,ratio,n,ratio)).mean(axis=(-1,-3,-5))
        if not np.isfinite(result).all():raise ArithmeticError('Conservative cell restriction overflows')
        return result
    result=data
    for axis in range(data.ndim-3,data.ndim):
        source=np.moveaxis(result,axis,0)
        reduced=np.empty((n,*source.shape[1:]),dtype=float)
        for coarse in range(n):
            low=coarse*fine;high=(coarse+1)*fine
            begin=low//n;end=(high+n-1)//n
            # Fine cell j has interval [j*n,(j+1)*n]; coarse width is fine.
            weights=np.asarray([min(high,(j+1)*n)-max(low,j*n) for j in range(begin,end)],dtype=float)/fine
            reduced[coarse]=np.tensordot(weights,source[begin:end],axes=(0,0))
        if not np.isfinite(reduced).all():raise ArithmeticError('Conservative overlap restriction overflows')
        result=np.moveaxis(reduced,0,axis)
    return result


def publish_twofluid_refinement(parameters,destination):
    import copy
    from pathlib import Path
    from core.sidm_analysis import _load_product
    from core.foundation import canonical_digest,file_sha256
    from core.data import export_numeric_audit
    request=validate_twofluid_refinement_request(parameters);sources={};previous=None;signature=None;resolutions=[];errors=[];initial_errors=[];signed_moments=[];seen=set()
    axis=request.get('refinement_axis','grid');step_definitions=[];initial_reference=None
    dissipation_levels=[];diffusion_measures=[];step_history=None
    material_enabled=request['schema']in ('twofluid-nonlinear-refinement-request-v3','twofluid-nonlinear-refinement-request-v4');previous_material=None;material_errors=[];material_means=[]
    numerical={'maximum_grid_cells','memory_gib','maximum_steps','maximum_rhs_evaluations','maximum_step_rejections','material_chunk_rows',
        'material_map_maximum_subboxes','material_map_maximum_depth'}
    def state(arrays,initial=False):
        prefix='initial_' if initial else '';rho=np.asarray(arrays[prefix+'density_contrast'],float)+1;velocity=np.asarray(arrays[prefix+'velocity_peculiar_km_s'],float)
        n=rho.shape[-1] if rho.ndim==4 else 0
        if rho.shape!=(2,n,n,n) or velocity.shape!=(2,3,n,n,n) or n**3>request['maximum_grid_cells'] or n<2 or not np.isfinite(rho).all() or not np.isfinite(velocity).all() or np.any(rho<=0):
            raise ValueError('Complete finite positive species cell-average density/velocity required')
        return rho,rho[:,None]*velocity,n
    def restrict(value,n):
        fine=value.shape[-1]
        if axis in ('timestep','dissipation'):
            if fine!=n:raise ValueError('Timestep/dissipation comparison must retain exactly the same Eulerian grid')
            return value
        if fine<=n:raise ValueError('Grid refinement requires strictly increasing grids on the same Eulerian cell origin')
        return restrict_twofluid_cell_averages(value,n)
    def measures(coarse,fine,n):
        rho,momentum,_=coarse;fr,fm,_=fine;rr=restrict(fr,n);rm=restrict(fm,n)
        dr=rho-rr;dm=momentum-rm
        density=np.mean(np.abs(dr),axis=(1,2,3))/np.maximum(np.mean(np.abs(rr),axis=(1,2,3)),request['density_ratio_floor'])
        mom=np.mean(np.linalg.norm(dm,axis=1),axis=(1,2,3))/np.maximum(np.mean(np.linalg.norm(rm,axis=1),axis=(1,2,3)),request['momentum_floor_km_s'])
        vc=momentum/rho[:,None];vf=rm/rr[:,None]
        relative=float(np.mean(np.linalg.norm((vc[1]-vc[0])-(vf[1]-vf[0]),axis=0)))
        return density,mom,relative,dm.mean(axis=(2,3,4))
    def material_state(report,arrays,n):
        displacement,velocity,origin,_=_twofluid_material_fields(report,arrays)
        deformation=np.asarray(arrays['material_deformation_gradient'],float)
        jacobian=np.asarray(arrays['material_jacobian'],float)
        if displacement.shape!=(2,3,n,n,n) or velocity.shape!=displacement.shape or not np.isfinite(displacement).all() or not np.isfinite(velocity).all():
            raise ValueError('Complete actual species material displacement and velocity required')
        for name,unit in (('material_displacement_mpc_h','comoving Mpc/h'),('material_velocity_peculiar_km_s','km / s'),
                ('material_deformation_gradient','dimensionless'),('material_jacobian','dimensionless')):
            if report['numeric_data']['array_contract']['datasets'][name]['units']!=unit:raise ValueError('Material refinement physical units differ: '+name)
        return ([displacement,velocity,deformation.reshape(2,9,n,n,n),jacobian[:,None]],origin,n)
    def material_measures(coarse,fine,box):
        # Compare continuous unwrapped displacement at identical finite-epoch
        # labels. A minimum-image displacement difference can hide whole turns.
        cf,co,n=coarse;ff,fo,_=fine;peak=np.zeros((4,2));total=np.zeros((4,2))
        for begin in range(0,n**3,request['material_chunk_rows']):
            rows=np.arange(begin,min(n**3,begin+request['material_chunk_rows']))
            q=np.column_stack(np.unravel_index(rows,(n,n,n)))*box/n+co
            for coordinate in range(4):
                for species in range(2):
                    left=cf[coordinate][species].reshape(cf[coordinate].shape[1],-1)[:,rows].T
                    right=_periodic_trilinear(ff[coordinate][species],q-fo,box)
                    distance=np.linalg.norm(left-right,axis=1)
                    if not np.isfinite(distance).all():raise ArithmeticError('Material refinement norm overflows')
                    peak[coordinate,species]=max(peak[coordinate,species],float(distance.max()))
                    total[coordinate,species]+=float(distance.sum())
        return peak.ravel(),(total/n**3).ravel()
    for reference in request['levels']:
        if reference['sha256'] in seen:raise ValueError('Refinement repeats an identical nonlinear product')
        seen.add(reference['sha256']);report,arrays,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
        if report.get('schema')!='finite-epoch-nonlinear-twofluid-reference-v1' or report.get('species_order')!=['cdm','baryons'] or report.get('cell_convention')!='periodic_eulerian_cell_averages':raise ValueError('Actual full nonlinear two-fluid Eulerian reference products required')
        if 'native_seed_definition' not in report:raise ValueError('Nonlinear refinement lacks original native seed provenance; regenerate with the current producer')
        contracts=report['numeric_data']['array_contract']['datasets']
        for name,unit in (('density_contrast','dimensionless'),('initial_density_contrast','dimensionless'),('velocity_peculiar_km_s','km / s'),('initial_velocity_peculiar_km_s','km / s')):
            if contracts[name]['units']!=unit:raise ValueError('Nonlinear refinement changes density/velocity physical units')
        physics=copy.deepcopy(report['request'])
        for name in numerical:physics.pop(name,None)
        physics['rusanov_dissipation_multiplier']=physics.get('rusanov_dissipation_multiplier',1.)
        if axis=='timestep':physics.pop('maximum_step_ln_a')
        if axis=='dissipation':physics.pop('rusanov_dissipation_multiplier')
        definition={'physical_request':physics,'native_seed':report['native_seed_definition'],'method':report['method'],'epoch':report['epoch'],'cell_convention':report['cell_convention']}
        current=canonical_digest(definition)
        if signature is not None and signature!=current:raise ValueError('Grid refinement changes original native seed, physics, time integration or domain assumptions')
        signature=current;final=state(arrays);initial=state(arrays,True);n=final[2]
        material=material_state(report,arrays,n) if material_enabled else None
        if axis=='dissipation':
            factor=report['request'].get('rusanov_dissipation_multiplier',1.)
            flux=report.get('numerical_flux',{})
            if type(factor)not in (int,float) or not math.isfinite(factor) or not 1<=factor<=16 or flux.get('dissipation_multiplier')!=factor:raise ValueError('Original operative reference dissipation definition required')
            if dissipation_levels and factor>=dissipation_levels[-1]:raise ValueError('Dissipation sensitivity requires distinct strictly decreasing multipliers')
            current_history=report.get('accepted_step_history_sha256')
            if not isinstance(current_history,str) or len(current_history)!=64 or any(ch not in '0123456789abcdef' for ch in current_history):raise ValueError('Actual accepted reference step-history fingerprint required')
            if step_history is not None and step_history!=current_history:raise ValueError('Dissipation family changes realized timestep history; reduce the common timestep cap')
            step_history=current_history;dissipation_levels.append(factor)
            if initial_reference is not None and report['initial_fields_reference']!=initial_reference:raise ValueError('Dissipation family must use exactly the same original seed product')
            initial_reference=report['initial_fields_reference']
            row=[]
            if flux.get('accepted_rhs_contributions')!=3*report['steps']:raise ValueError('Accepted-stage diffusion accounting differs from actual reference step count')
            for quantity,shape in (('density',(2,n,n,n)),('momentum',(2,3,n,n,n))):
                expected_unit='dimensionless' if quantity=='density' else 'comoving Mpc/h per ln(a); sum of normalized cell-density momentum'
                for suffix in ('_update','_absolute_update'):
                    if contracts['numerical_diffusion_'+quantity+suffix]['units']!=expected_unit:raise ValueError('Reference diffusion ledger units differ')
                signed=np.asarray(arrays['numerical_diffusion_'+quantity+'_update'],float)
                absolute=np.asarray(arrays['numerical_diffusion_'+quantity+'_absolute_update'],float)
                if signed.shape!=shape or absolute.shape!=shape or not np.isfinite(signed).all() or not np.isfinite(absolute).all() or np.any(absolute<0) or np.any(absolute+1e-14*np.maximum(1.,absolute)<np.abs(signed)):raise ValueError('Complete consistent accepted-stage diffusion ledger required')
                axes=tuple(range(1,signed.ndim));row.extend(np.sum(np.abs(signed),axis=axes));row.extend(np.sum(absolute,axis=axes))
            diffusion_measures.append(row)
        del arrays
        step_record=[report['request']['maximum_step_ln_a'],report.get('maximum_realized_step_ln_a'),report['steps']]
        if axis=='timestep':
            if initial_reference is not None and report['initial_fields_reference']!=initial_reference:raise ValueError('Timestep refinement must use the exact same original seed product')
            initial_reference=report['initial_fields_reference']
            if type(step_record[1])not in (int,float) or not math.isfinite(step_record[1]) or step_record[1]<=0 or type(step_record[2])is not int or step_record[2]<1:
                raise ValueError('Actual accepted step diagnostics required; regenerate old nonlinear products')
            if step_definitions and (step_record[0]>=step_definitions[-1][0] or step_record[1]>=step_definitions[-1][1] or step_record[2]<=step_definitions[-1][2]):
                raise ValueError('Timestep limits do not yield strictly finer realized steps and more accepted updates')
        step_definitions.append(step_record)
        if initial[2]!=n:raise ValueError('Nonlinear product changes grid between seed and evolved states')
        if previous is not None:
            coarse_final,coarse_initial=previous
            measured=measures(coarse_final,final,coarse_final[2]);seed_measured=measures(coarse_initial,initial,coarse_initial[2])
            if np.any(seed_measured[0]>request['maximum_initial_density_error']) or np.any(seed_measured[1]>request['maximum_initial_momentum_error']):
                raise ValueError('Restricted original native initial mass/momentum do not match the coarser initial problem')
            errors.append([*measured[0],*measured[1],measured[2]]);initial_errors.append([*seed_measured[0],*seed_measured[1]]);signed_moments.append(measured[3])
            if material_enabled:
                maximum,mean=material_measures(previous_material,material,float(report['request']['box_mpc_h']))
                material_errors.append(maximum);material_means.append(mean)
        resolutions.append(n);previous=(final,initial);previous_material=material
    values=np.asarray(errors);initial_values=np.asarray(initial_errors)
    limits=np.asarray([request['maximum_density_error']]*2+[request['maximum_momentum_error']]*2+[request['maximum_relative_velocity_error_km_s']])
    within=values<=limits;decreasing=values[1:]<=request['maximum_adjacent_error_ratio']*values[:-1]
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    output={'grid_sizes':np.asarray(resolutions,np.int64),'adjacent_errors':values,'initial_restriction_errors':initial_values,
        'adjacent_signed_momentum_difference_km_s':np.asarray(signed_moments),'within_declared_limits':within,'nonincreasing_error_budget':decreasing}
    if axis=='dissipation':
        output['rusanov_dissipation_multipliers']=np.asarray(dissipation_levels)
        output['accepted_numerical_diffusion_update_l1']=np.asarray(diffusion_measures)
        output['operative_numerical_diffusion']=np.any(np.asarray(diffusion_measures)>0,axis=1)
    if axis=='timestep':output['configured_and_realized_step_ln_a']=np.asarray([row[:2] for row in step_definitions]);output['realized_step_counts']=np.asarray([row[2] for row in step_definitions],np.int64)
    if material_enabled:
        material_values=np.asarray(material_errors);material_mean_values=np.asarray(material_means)
        material_limits=np.repeat([request['maximum_material_displacement_error_mpc_h'],request['maximum_material_velocity_error_km_s'],
            request['maximum_material_deformation_error'],request['maximum_material_jacobian_error']],2)
        material_within=material_values<=material_limits
        material_decreasing=material_values[1:]<=request['maximum_adjacent_error_ratio']*material_values[:-1]
        material_mean_decreasing=material_mean_values[1:]<=request['maximum_adjacent_error_ratio']*material_mean_values[:-1]
        output.update(adjacent_material_maximum_errors=material_values,adjacent_material_mean_errors=material_mean_values,
            material_within_declared_limits=material_within,material_maximum_nonincreasing_error_budget=material_decreasing,
            material_mean_nonincreasing_error_budget=material_mean_decreasing)
        within=np.concatenate((within,material_within),axis=1)
        decreasing=np.concatenate((decreasing,material_decreasing,material_mean_decreasing),axis=1)
    if any(not np.isfinite(value).all() for value in output.values()):raise ArithmeticError('Nonlinear restriction comparison overflows')
    report={'schema':'measured-twofluid-nonlinear-refinement-v1','execution_status':'COMPLETE','request':request,'fixed_definition':definition,'fixed_signature':signature,
        'refinement_axis':axis,
        'error_coordinates':['cdm_density_ratio_l1','baryon_density_ratio_l1','cdm_normalized_momentum_l1','baryon_normalized_momentum_l1','relative_velocity_l1_km_s'],
        'restriction':'same-grid physical species density/momentum comparison' if axis in ('timestep','dissipation') else 'aligned cell-volume overlap averages of species density and density-weighted momentum, including nonnested grids; velocity reconstructed after restriction',
        'status':('DECLARED_TIMESTEP_REFINEMENT_BUDGETS_SUPPORTED' if axis=='timestep' else 'DECLARED_GRID_REFINEMENT_BUDGETS_SUPPORTED') if within.all() and decreasing.all() else 'UNRESOLVED_NONLINEAR_REFINEMENT',
        'scope':'same original native seed, explicitly selected adjacent Eulerian grid or operative timestep refinement; no continuum extrapolation, material-map/projection or native evolution acceptance',
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    if material_enabled:
        report['schema']='measured-twofluid-nonlinear-refinement-v2'
        report['material_error_coordinates']=[species+'_'+quantity for quantity in ('displacement_mpc_h','velocity_km_s','deformation_frobenius','jacobian_absolute') for species in ('cdm','baryon')]
        report['material_error_units']=['comoving Mpc/h']*2+['km / s']*2+['dimensionless']*4
        report['material_comparison']='fine piecewise-trilinear material fields at original coarse cell-centred labels; unwrapped displacement, vector velocity, Frobenius deformation and scalar Jacobian norms; every coarse label sampled in bounded chunks'
        report['scope']='same original native seed; adjacent Eulerian and material-characteristic grid/timestep budgets; no continuum extrapolation, finite-volume particle projection or native evolution acceptance'
    if axis=='dissipation':
        report['status']=('DECLARED_DISSIPATION_SENSITIVITY_BUDGETS_SUPPORTED' if within.all() and decreasing.all() else 'UNRESOLVED_DISSIPATION_SENSITIVITY') if output['operative_numerical_diffusion'].all() else 'DISSIPATION_CONTROL_NOT_EXERCISED'
        report['schema']='measured-twofluid-dissipation-sensitivity-v1'
        report['accepted_step_history_sha256']=step_history
        report['diffusion_ledger_coordinates']=[quantity+'_'+measure+'_'+species for quantity in ('density','momentum') for measure in ('signed_update_l1','cumulative_absolute_update_l1') for species in ('cdm','baryon')]
        report['scope']='same original seed, grid, backgrounds, physical model and actual accepted step history; varied supercharacteristic Rusanov multiplier. Eulerian/material sensitivity and flux ledger only; no continuum, physical viscosity, shock, native or scientific acceptance'
    return export_numeric_audit(destination,report,output,{name:{'units':'coordinate-dependent' if name=='accepted_numerical_diffusion_update_l1' else 'km / s' if name.endswith('km_s') else 'coordinate-dependent' if name in ('adjacent_errors','adjacent_material_maximum_errors','adjacent_material_mean_errors') else 'dimensionless','quantity':'nonlinear reference comparison '+name} for name in output},
        algorithm=source,source_hashes=sources,context={'epoch':definition['epoch']})


def twofluid_refinement_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Nonlinear refinement exceeds allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Nonlinear refinement escapes its owned generation')
    return publish_twofluid_refinement(parameters,destination)


def twofluid_refinement_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Measure conservative adjacent nonlinear two-fluid grid refinement')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text());parameters.setdefault('request_base',str(path.parent))
    publish_twofluid_refinement(parameters,args.output);return 0


def validate_twofluid_hierarchy_campaign_request(parameters):
    import copy
    request=parameters['request']
    keys={'schema','maximum_levels','relative_fields','native_loads','projection_request','refinement_request','resources','followup_limits'}
    if not isinstance(request,dict) or set(request)!=keys or request.get('schema')!='twofluid-hierarchy-campaign-request-v1':raise ValueError('Complete original native hierarchy projection campaign required')
    _validate_twofluid_product_reference(request['relative_fields'],'fixed hierarchy material flow')
    if type(request['maximum_levels'])is not int or request['maximum_levels']<3 or not isinstance(request['native_loads'],list) or not 3<=len(request['native_loads'])<=request['maximum_levels']:
        raise ValueError('At least three bounded original nested native loads required')
    resources=request['resources']
    for name in ('mpi_ranks','omp_threads','maximum_stage_count'):
        if type(resources[name])is not int or resources[name]<1:raise ValueError('Exact positive hierarchy campaign allocation required')
    for name in ('memory_gib','disk_gib'):
        if type(resources[name])not in (int,float) or not math.isfinite(resources[name]) or resources[name]<=0:raise ValueError('Finite positive hierarchy campaign allocation required')
    if len(request['native_loads'])+1>resources['maximum_stage_count']:raise MemoryError('Complete hierarchy projection graph exceeds stage ceiling')
    for name in ('maximum_children','maximum_depth'):
        if type(request['followup_limits'][name])is not int or request['followup_limits'][name]<0:raise ValueError('Explicit hierarchy follow-up limits required')
    template=request['projection_request']
    if not isinstance(template,dict) or any(name in template for name in ('relative_fields','native_particle_load','particle_namespace')) or template.get('schema')!='twofluid-native-particle-projection-request-v2':
        raise ValueError('Freeze finite-volume projection operator; campaign owns actual flow/load/namespace bindings')
    seen=set()
    for level in request['native_loads']:
        if not isinstance(level,dict) or set(level)!={'native_particle_load','particle_namespace'}:raise ValueError('Retain exact original load and particle namespace per hierarchy')
        _validate_twofluid_product_reference(level['native_particle_load'],'original hierarchy load')
        identity=level['native_particle_load']['sha256']
        if isinstance(identity,dict):identity=repr(sorted(identity.items()))
        if identity in seen:raise ValueError('Hierarchy campaign repeats an identical original load')
        seen.add(identity)
        candidate=copy.deepcopy(template);candidate.update(level,relative_fields=copy.deepcopy(request['relative_fields']))
        validate_twofluid_particle_request({'request':candidate})
    comparison=copy.deepcopy(request['refinement_request'])
    if not isinstance(comparison,dict) or 'levels' in comparison or comparison.get('refinement_axis')!='hierarchy':raise ValueError('Explicit frozen hierarchy comparison limits required')
    comparison['levels']=[{'path':'generated','sha256':'0'*64} for _ in request['native_loads']]
    validate_twofluid_projection_refinement_request({'request':comparison})
    if max(template['memory_gib'],comparison['memory_gib'])>resources['memory_gib']:raise MemoryError('Hierarchy campaign stages exceed allocated memory')
    return request


def twofluid_hierarchy_campaign_input_bindings(parameters):
    import copy
    request=validate_twofluid_hierarchy_campaign_request(parameters);sources={}
    for level in request['native_loads']:
        candidate=copy.deepcopy(request['projection_request']);candidate.update(level,relative_fields=copy.deepcopy(request['relative_fields']))
        if not isinstance(candidate['native_particle_load']['path'],str):raise ValueError('Hierarchy planner requires an existing original native load')
        sources.update(twofluid_particle_input_bindings({'request':candidate,'request_base':parameters['request_base']}))
    return sources


def compile_twofluid_hierarchy_campaign(request,destination,*,request_base):
    import copy
    from pathlib import Path
    from .foundation import atomic_json,file_sha256
    from .workflow import compile_requested_product_dag,default_registry
    base=Path(request_base).resolve();destination=Path(destination).resolve()
    sources=twofluid_hierarchy_campaign_input_bindings({'request':request,'request_base':str(base)})
    stages=[];levels=[];ids=[]
    flow=copy.deepcopy(request['relative_fields'])
    if not isinstance(flow['path'],str):raise ValueError('Hierarchy planning requires an existing sealed nonlinear material flow')
    flow['path']=str((base/flow['path']).resolve())
    for index,load in enumerate(request['native_loads']):
        identity=f'hierarchy_projection_{index}';projection=copy.deepcopy(request['projection_request']);projection.update(copy.deepcopy(load),relative_fields=copy.deepcopy(flow))
        projection['native_particle_load']['path']=str((base/projection['native_particle_load']['path']).resolve())
        stages.append({'stage_id':identity,'kind':'twofluid_particle_projection',
            'parameters':{'request':projection,'request_base':str(base),'destination':'{generation}/projection.json'},'outputs':{identity:'projection.json'}})
        levels.append({'path':{'stage':identity,'artifact':'projection.json'},'sha256':{'stage':identity,'artifact':'projection.json','load':'sha256'}});ids.append(identity)
    comparison=copy.deepcopy(request['refinement_request']);comparison['levels']=levels
    result_id='native_projection_refinement'
    stages.append({'stage_id':result_id,'kind':'twofluid_projection_refinement','depends_on':ids,
        'parameters':{'request':comparison,'request_base':str(base),'destination':'{generation}/projection-refinement.json'},'outputs':{result_id:'projection-refinement.json'}})
    workflow={'schema':'research-workflow-v1','resources':copy.deepcopy(request['resources']),'followup_limits':copy.deepcopy(request['followup_limits']),
        'stages':stages,'requested_products':[result_id]}
    compile_requested_product_dag(workflow,default_registry(),request_base=base)
    if destination.exists():raise FileExistsError('Retain prior hierarchy campaign designs')
    destination.mkdir(parents=True);atomic_json(destination/'workflow.json',workflow)
    report={'schema':'twofluid-hierarchy-campaign-design-v1','execution_status':'PLANNED_NOT_EXECUTED','request':request,
        'workflow':str(destination/'workflow.json'),'workflow_sha256':file_sha256(destination/'workflow.json'),'stage_count':len(stages),
        'native_source_hashes':sources,'refinement_axis':'hierarchy','planned_result_stages':[result_id],
        'scope':'same exact material flow -> supplied original hierarchy projections -> conservative nested species restriction comparison; geometry support evaluated by actual comparison, no native generation or evolution executed',
        'science_acceptance':'NOT_ESTABLISHED'}
    atomic_json(destination/'design.json',report);return report


def validate_twofluid_refinement_campaign_request(parameters):
    if parameters['request'].get('schema')in ('twofluid-native-timestep-campaign-request-v1','twofluid-native-force-campaign-request-v1','twofluid-native-softening-campaign-request-v1','twofluid-native-pm-campaign-request-v1','twofluid-native-hydro-campaign-request-v1'):
        return validate_twofluid_native_timestep_campaign_request(parameters)
    if parameters['request'].get('schema')=='twofluid-native-comparison-campaign-request-v1':
        return validate_twofluid_native_campaign_request(parameters)
    request=parameters['request']
    if isinstance(request,dict) and request.get('schema')=='twofluid-hierarchy-campaign-request-v1':return validate_twofluid_hierarchy_campaign_request(parameters)
    keys={'schema','grid_sizes','maximum_levels','seed_parameters','nonlinear_request','refinement_request','resources','followup_limits'}
    if not isinstance(request,dict):raise ValueError('Nonlinear campaign request must be an object')
    dissipation=request.get('schema')=='twofluid-dissipation-campaign-request-v1'
    temporal=request.get('schema')in ('twofluid-timestep-campaign-request-v1','twofluid-timestep-campaign-request-v2')
    same_grid=temporal or dissipation
    projected=request.get('schema')in ('twofluid-refinement-campaign-request-v2','twofluid-timestep-campaign-request-v2')
    if projected:keys.update(('projection_request','projection_refinement_request'))
    if same_grid:keys=(keys-{'grid_sizes'})|{'grid_size','rusanov_dissipation_levels' if dissipation else 'maximum_step_ln_a_levels'}
    elif request.get('schema')not in ('twofluid-refinement-campaign-request-v1','twofluid-refinement-campaign-request-v2'):raise ValueError('Unsupported nonlinear campaign version')
    if set(request)!=keys:raise ValueError('Complete nonlinear refinement campaign request required')
    if same_grid:
        sizes=request['rusanov_dissipation_levels' if dissipation else 'maximum_step_ln_a_levels']
        if not isinstance(sizes,list) or any(type(x)not in (int,float) or not math.isfinite(x) or x<=0 for x in sizes) or any(b>=a for a,b in zip(sizes,sizes[1:])):
            raise ValueError('Strictly decreasing finite positive ln(a) timestep levels required')
        if dissipation and any(not 1<=value<=16 for value in sizes):raise ValueError('Dissipation levels must remain between 1 and 16')
        grids=[request['grid_size']]*len(sizes)
    else:grids=request['grid_sizes']
    ceiling=request['maximum_levels']
    if type(ceiling)is not int or ceiling<3 or not isinstance(grids,list) or not 3<=len(grids)<=ceiling or any(type(n)is not int or n<2 for n in grids):raise ValueError('At least three bounded integer nonlinear grid sizes required')
    if not same_grid and any(b<=a for a,b in zip(grids,grids[1:])):raise ValueError('Strictly increasing nonlinear grid resolutions required')
    resources=request['resources']
    for name in ('mpi_ranks','omp_threads','maximum_stage_count'):
        if type(resources[name])is not int or resources[name]<1:raise ValueError('Explicit positive campaign allocation required')
    for name in ('memory_gib','disk_gib'):
        if type(resources[name])not in (int,float) or not math.isfinite(resources[name]) or resources[name]<=0:raise ValueError('Explicit finite campaign allocation required')
    stage_count=(len(grids)+2 if same_grid else 2*len(grids)+1)+(len(grids)+1 if projected else 0)
    if stage_count>resources['maximum_stage_count']:raise MemoryError('Complete nonlinear refinement graph exceeds stage ceiling')
    for name in ('maximum_children','maximum_depth'):
        if type(request['followup_limits'][name])is not int or request['followup_limits'][name]<0:raise ValueError('Explicit follow-up limits required')
    seed=request['seed_parameters'];evolution=request['nonlinear_request'];comparison=request['refinement_request']
    if not isinstance(seed,dict) or set(seed)!={'request','native_stage_receipt','native_run_directory'}:raise ValueError('Original native seed template required')
    if not isinstance(evolution,dict) or evolution.get('schema')!='twofluid-nonlinear-reference-request-v1':raise ValueError('Complete nonlinear evolution template required')
    if not isinstance(comparison,dict) or 'levels' in comparison:raise ValueError('Freeze comparison limits; campaign owns generated level references')
    if temporal and (comparison.get('schema')not in ('twofluid-nonlinear-refinement-request-v2','twofluid-nonlinear-refinement-request-v3') or comparison.get('refinement_axis')!='timestep'):
        raise ValueError('Timestep campaign requires explicit temporal comparison semantics')
    if dissipation and (comparison.get('schema')!='twofluid-nonlinear-refinement-request-v4' or comparison.get('refinement_axis')!='dissipation'):raise ValueError('Dissipation campaign requires explicit flux-sensitivity comparison semantics')
    if not same_grid and comparison.get('refinement_axis','grid')!='grid':raise ValueError('Grid campaign cannot relabel temporal comparisons')
    import copy
    for n in grids:
        candidate=copy.deepcopy(seed);candidate['request']['grid_size']=n;validate_twofluid_eulerian_seed_request(candidate)
        if n**3>evolution['maximum_grid_cells']:raise MemoryError('Nonlinear grid exceeds evolution cell ceiling')
    check=copy.deepcopy(comparison);check['levels']=[{'path':'generated','sha256':'0'*64} for _ in grids];validate_twofluid_refinement_request({'request':check})
    for memory in (seed['request']['memory_gib'],evolution['memory_gib'],comparison['memory_gib']):
        if memory>resources['memory_gib']:raise MemoryError('Nonlinear campaign stage exceeds allocated memory')
    if projected:
        projection=copy.deepcopy(request['projection_request']);measurement=copy.deepcopy(request['projection_refinement_request'])
        if not isinstance(projection,dict) or 'relative_fields' in projection or projection.get('schema')!='twofluid-native-particle-projection-request-v2':raise ValueError('Freeze full finite-volume projection template; campaign owns flow references')
        projection['relative_fields']={'path':'generated','sha256':'0'*64};validate_twofluid_particle_request({'request':projection})
        _validate_twofluid_product_reference(projection['native_particle_load'],'fixed original native leaf load')
        if not isinstance(measurement,dict) or 'levels' in measurement or measurement.get('refinement_axis')!=('timestep' if temporal else 'grid'):raise ValueError('Projection comparison must select the campaign refinement axis')
        measurement['levels']=[{'path':'generated','sha256':'0'*64} for _ in grids];validate_twofluid_projection_refinement_request({'request':measurement})
        if max(projection['memory_gib'],measurement['memory_gib'])>resources['memory_gib']:raise MemoryError('Projection campaign exceeds allocated memory')
    return request


def twofluid_refinement_campaign_input_bindings(parameters):
    if parameters['request'].get('schema')in ('twofluid-native-timestep-campaign-request-v1','twofluid-native-force-campaign-request-v1','twofluid-native-softening-campaign-request-v1','twofluid-native-pm-campaign-request-v1','twofluid-native-hydro-campaign-request-v1'):
        return twofluid_native_timestep_campaign_input_bindings(parameters)
    if parameters['request'].get('schema')=='twofluid-native-comparison-campaign-request-v1':
        request=validate_twofluid_native_campaign_request(parameters)
        import copy
        from pathlib import Path
        seed=copy.deepcopy(request['seed_parameters']);base=Path(parameters['request_base']).resolve()
        for name in ('native_stage_receipt','native_run_directory'):seed[name]=str((base/seed[name]).resolve())
        return twofluid_eulerian_seed_input_bindings(seed)
    import copy
    from pathlib import Path
    request=validate_twofluid_refinement_campaign_request(parameters)
    if request['schema']=='twofluid-hierarchy-campaign-request-v1':return twofluid_hierarchy_campaign_input_bindings(parameters)
    seed=copy.deepcopy(request['seed_parameters']);base=Path(parameters['request_base']).resolve()
    for name in ('native_stage_receipt','native_run_directory'):seed[name]=str((base/seed[name]).resolve())
    sources=twofluid_eulerian_seed_input_bindings(seed)
    if 'projection_request' in request:
        from .foundation import file_sha256
        reference=request['projection_request']['native_particle_load']
        if not isinstance(reference['path'],str):raise ValueError('Projection campaign requires an existing original native load')
        path=(base/reference['path']).resolve()
        if not path.is_file() or path.is_symlink() or file_sha256(path)!=reference['sha256']:raise ValueError('Original projection native load changed')
        sources[str(path)]=reference['sha256']
    return sources


def compile_twofluid_refinement_campaign(request,destination,*,request_base):
    if request.get('schema')in ('twofluid-native-timestep-campaign-request-v1','twofluid-native-force-campaign-request-v1','twofluid-native-softening-campaign-request-v1','twofluid-native-pm-campaign-request-v1','twofluid-native-hydro-campaign-request-v1'):
        return compile_twofluid_native_timestep_campaign(request,destination,request_base=request_base)
    if isinstance(request,dict) and request.get('schema')=='twofluid-native-comparison-campaign-request-v1':
        return compile_twofluid_native_comparison_campaign(request,destination,request_base=request_base)
    import copy
    from pathlib import Path
    from core.foundation import atomic_json,file_sha256
    from core.workflow import compile_requested_product_dag,default_registry
    validate_twofluid_refinement_campaign_request({'request':request})
    if request['schema']=='twofluid-hierarchy-campaign-request-v1':return compile_twofluid_hierarchy_campaign(request,destination,request_base=request_base)
    base=Path(request_base).resolve();destination=Path(destination).resolve()
    seed=copy.deepcopy(request['seed_parameters'])
    for name in ('native_stage_receipt','native_run_directory'):seed[name]=str((base/seed[name]).resolve())
    sources=twofluid_refinement_campaign_input_bindings({'request':request,'request_base':str(base)})
    seed['expected_native_source_hashes']=twofluid_eulerian_seed_input_bindings(seed)
    stages=[];levels=[];evolved=[];projection_levels=[];projection_ids=[]
    def product(stage,filename):return {'path':{'stage':stage,'artifact':filename},'sha256':{'stage':stage,'artifact':filename,'load':'sha256'}}
    dissipation=request['schema']=='twofluid-dissipation-campaign-request-v1'
    temporal=request['schema']in ('twofluid-timestep-campaign-request-v1','twofluid-timestep-campaign-request-v2');same_grid=temporal or dissipation
    grids=[request['grid_size']]*len(request['rusanov_dissipation_levels' if dissipation else 'maximum_step_ln_a_levels']) if same_grid else request['grid_sizes']
    for index,n in enumerate(grids):
        seed_id=f'seed_grid_{n}';evolve_id=f'nonlinear_dissipation_{index}' if dissipation else f'nonlinear_timestep_{index}' if temporal else f'nonlinear_grid_{n}'
        parameters=copy.deepcopy(seed);parameters['request']['grid_size']=n;parameters['destination']='{generation}/seed.json'
        if not same_grid or index==0:stages.append({'stage_id':seed_id,'kind':'twofluid_eulerian_seed','parameters':parameters,'outputs':{seed_id:'seed.json'}})
        evolution=copy.deepcopy(request['nonlinear_request'])
        if temporal:evolution['maximum_step_ln_a']=request['maximum_step_ln_a_levels'][index]
        if dissipation:evolution['rusanov_dissipation_multiplier']=request['rusanov_dissipation_levels'][index]
        stages.append({'stage_id':evolve_id,'kind':'twofluid_nonlinear_reference','depends_on':[seed_id],
            'parameters':{'request':evolution,'initial_fields':product(seed_id,'seed.json'),'request_base':str(base),'destination':'{generation}/nonlinear.json'},
            'outputs':{evolve_id:'nonlinear.json'}})
        levels.append(product(evolve_id,'nonlinear.json'));evolved.append(evolve_id)
        if 'projection_request' in request:
            projection_id=f'projection_level_{index}';projection=copy.deepcopy(request['projection_request'])
            projection['native_particle_load']['path']=str((base/projection['native_particle_load']['path']).resolve())
            projection['relative_fields']=product(evolve_id,'nonlinear.json')
            stages.append({'stage_id':projection_id,'kind':'twofluid_particle_projection','depends_on':[evolve_id],
                'parameters':{'request':projection,'request_base':str(base),'destination':'{generation}/projection.json'},'outputs':{projection_id:'projection.json'}})
            projection_levels.append(product(projection_id,'projection.json'));projection_ids.append(projection_id)
    comparison=copy.deepcopy(request['refinement_request']);comparison['levels']=levels
    stages.append({'stage_id':'nonlinear_grid_refinement','kind':'twofluid_nonlinear_refinement','depends_on':evolved,
        'parameters':{'request':comparison,'request_base':str(base),'destination':'{generation}/refinement.json'},'outputs':{'nonlinear_grid_refinement':'refinement.json'}})
    requested=['nonlinear_grid_refinement']
    if projection_levels:
        measurement=copy.deepcopy(request['projection_refinement_request']);measurement['levels']=projection_levels
        stages.append({'stage_id':'native_projection_refinement','kind':'twofluid_projection_refinement','depends_on':projection_ids,
            'parameters':{'request':measurement,'request_base':str(base),'destination':'{generation}/projection-refinement.json'},'outputs':{'native_projection_refinement':'projection-refinement.json'}})
        requested.append('native_projection_refinement')
    workflow={'schema':'research-workflow-v1','resources':copy.deepcopy(request['resources']),'followup_limits':copy.deepcopy(request['followup_limits']),
        'stages':stages,'requested_products':requested}
    compile_requested_product_dag(workflow,default_registry(),request_base=base)
    if destination.exists():raise FileExistsError('Retain previous nonlinear campaign designs')
    destination.mkdir(parents=True);atomic_json(destination/'workflow.json',workflow)
    report={'schema':'twofluid-refinement-campaign-design-v1','execution_status':'PLANNED_NOT_EXECUTED','request':request,
        'workflow':str(destination/'workflow.json'),'workflow_sha256':file_sha256(destination/'workflow.json'),'stage_count':len(stages),
        'native_source_hashes':sources,'refinement_axis':'dissipation' if dissipation else 'timestep' if temporal else 'grid',
        'planned_result_stages':requested,
        'scope':'same native snapshot seed -> declared nonlinear grid or timestep family -> operative adjacent measurements; optional exact original native leaf projection family; no native evolution acceptance',
        'science_acceptance':'NOT_ESTABLISHED'}
    atomic_json(destination/'design.json',report);return report


def twofluid_refinement_campaign_runner(parameters,context):
    import copy,json
    from pathlib import Path
    from core.foundation import atomic_json,file_sha256
    from core.workflow import safe_id
    request=parameters['request'];allocation=request['resources']
    for name in ('mpi_ranks','omp_threads'):
        if allocation[name]!=context.resources[name]:raise ValueError('Automatic nonlinear campaign changes its execution allocation')
    for name in ('memory_gib','disk_gib','maximum_stage_count'):
        if allocation[name]>context.resources[name]:raise MemoryError('Automatic nonlinear campaign exceeds allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Nonlinear campaign escapes its owned generation')
    report=compile_twofluid_refinement_campaign(request,destination,request_base=parameters['request_base'])
    workflow=json.loads(Path(report['workflow']).read_text());names={row['stage_id']:safe_id(context.stage_id+'__'+row['stage_id']) for row in workflow['stages']}
    def rewrite(value):
        if isinstance(value,list):return [rewrite(item) for item in value]
        if isinstance(value,dict):
            result={key:rewrite(item) for key,item in value.items()}
            if isinstance(result.get('stage'),str) and result['stage'] in names:result['stage']=names[result['stage']]
            return result
        return value
    children=[]
    for original in workflow['stages']:
        row=rewrite(copy.deepcopy(original));row['stage_id']=names[original['stage_id']];row['depends_on']=[names[x] for x in original.get('depends_on',[])]
        row['outputs']={safe_id(context.stage_id+'__'+key):value for key,value in original['outputs'].items()};children.append(row)
    result_id=names[workflow['requested_products'][0]];results={key:names[key] for key in workflow['requested_products']}
    atomic_json(destination/'followups.json',{'followups':children,'refinement_result_stage':result_id,'result_stages':results})
    return {'followups':children,'measurements':{'nonlinear_campaign_design':str(destination/'design.json'),
        'nonlinear_campaign_design_sha256':file_sha256(destination/'design.json'),'planned_refinement_stage':result_id,'planned_result_stages':results}}


def twofluid_refinement_campaign_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Compile nonlinear grid, operative timestep, numerical dissipation or supplied native hierarchy comparison workflows')
    parser.add_argument('request',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.request.resolve();compile_twofluid_refinement_campaign(json.loads(path.read_text()),args.output,request_base=path.parent);return 0


def twofluid_nonlinear_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Nonlinear reference exceeds workflow allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Nonlinear reference escapes its owned generation')
    return publish_twofluid_nonlinear_product(parameters,destination)


def twofluid_nonlinear_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Produce conservative full nonlinear single-stream two-fluid reference grids')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.parameters.resolve();parameters=json.loads(source.read_text())
    parameters.setdefault('request_base',str(source.parent));publish_twofluid_nonlinear_product(parameters,args.output)
    return 0


def twofluid_relative_input_bindings(parameters):
    from .sidm_analysis import _load_product
    _,_,sources=_load_product(parameters['initial_fields'],parameters['request_base'],parameters['request']['memory_gib']*1024**3)
    return sources


def publish_twofluid_relative_product(parameters, destination):
    """Owned public producer of measured-seed finite-epoch two-fluid grids."""
    from pathlib import Path
    from .sidm_analysis import _load_product
    from .foundation import file_sha256
    from .data import export_numeric_audit
    request=parameters['request'];memory=request['memory_gib']*1024**3
    report,initial,sources=_load_product(parameters['initial_fields'],parameters['request_base'],memory)
    epoch=report.get('epoch',{})
    if epoch.get('actual_scale_factor',epoch.get('scale_factor'))!=request['reference_scale_factor']:
        raise ValueError('Two-fluid seed product is not at the exact declared reference epoch')
    required={'density_first_order':'dimensionless','density_second_order':'dimensionless',
        'velocity_first_order_km_s':'km / s','velocity_second_order_km_s':'km / s',
        'displacement_first_order_mpc_h':'comoving Mpc/h','displacement_second_order_mpc_h':'comoving Mpc/h'}
    for name,units in required.items():
        if report['numeric_data']['array_contract']['datasets'][name]['units']!=units:
            raise ValueError('Native two-fluid seed physical units differ: '+name)
    result,arrays=evolve_twofluid_relative_modes(initial,request)
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    result['source_hashes']=sources
    result['epoch']={'canonical_time':'scale_factor','actual_scale_factor':request['target_scale_factor'],
        'scale_factor':request['target_scale_factor'],'redshift':1/request['target_scale_factor']-1}
    units={name:('comoving Mpc/h' if 'displacement' in name else 'km / s' if 'velocity' in name else 'dimensionless') for name in arrays}
    return export_numeric_audit(destination,result,arrays,{name:{'units':units[name],
        'quantity':'finite-epoch perturbative two-fluid '+name.replace('_',' ')} for name in arrays},algorithm=source,
        context={'epoch':result['epoch'],'coordinates':{'frame':'recorded periodic Lagrangian grid; component-first vectors'}},source_hashes=sources)


def _periodic_trilinear(field,coordinates,box):
    """Component-last samples of a scalar/vector periodic q-grid."""
    field=np.asarray(field,float);q=np.asarray(coordinates,float)
    n=field.shape[-1]
    if field.shape[-3:]!=(n,n,n) or field.ndim not in (3,4) or q.ndim!=2 or q.shape[1]!=3:
        raise ValueError('Periodic cubic scalar/component-first vector field and exact q rows required')
    scaled=np.mod(q,box)*n/box;low=np.floor(scaled).astype(np.int64);fraction=scaled-low
    output=np.zeros((len(q),) if field.ndim==3 else (len(q),field.shape[0]))
    for i in (0,1):
        for j in (0,1):
            for k in (0,1):
                index=(low+np.asarray([i,j,k]))%n
                weight=np.prod(np.where(np.asarray([i,j,k]),fraction,1-fraction),axis=1)
                values=field[...,index[:,0],index[:,1],index[:,2]]
                output+=values*weight if field.ndim==3 else values.T*weight[:,None]
    return output


def _spectral_trilinear_bound(field,box):
    """A conservative uniform bound relative to the retained Fourier field.

    Successive linear interpolation errors are bounded by h^2/8 times each
    diagonal second-derivative supremum. The Fourier coefficient L1 norm
    bounds that supremum without assuming a measured point is the maximum.
    """
    field=np.asarray(field,float);n=field.shape[-1];vectors=field[None] if field.ndim==3 else field
    frequencies=2*np.pi*np.fft.fftfreq(n,d=box/n);bounds=[]
    for value in vectors:
        coefficient=np.abs(np.fft.fftn(value))/n**3;bound=0.
        for axis in range(3):
            shape=[1,1,1];shape[axis]=n
            bound+=float(np.sum(coefficient*frequencies.reshape(shape)**2))*(box/n)**2/8
        bounds.append(bound)
    return np.asarray(bounds)


def _native_material_geometry(native,box,baryon_fraction,*,maximum_particles,memory_bytes):
    """Read original dyadic leaf volumes; validate a complete common partition.

    A compensated mass is not a volume estimator. Geometry must be emitted
    directly by the native hierarchy, in exact native particle row order.
    The gas tessellation is the common partition translated by its recorded
    finest-grid staggering; coarse particles carry both translated species.
    """
    from .data import exact_uint64_ids,snapshot_total_counts
    from .foundation import KPC_CM
    header=native['Header'].attrs;counts=snapshot_total_counts(header);total=int(counts.sum())
    if int(header['NumFilesPerSnapshot'])!=1 or any(counts[k] for k in (3,4,5)) or counts[0]!=counts[1] or counts[0]==0:
        raise ValueError('Complete single-file native gas/CDM/coarse material geometry required')
    if total>maximum_particles or total*1024>memory_bytes:raise MemoryError('Native material geometry exceeds its declared particle workspace')
    if header.get('Music_material_cell_schema')!=1 or not math.isclose(float(header.get('Music_material_cell_unitlength_cm',math.nan)),KPC_CM,rel_tol=1e-12):
        raise ValueError('Native material geometry must retain original common cells in comoving kpc/h')
    finest=int(header['Music_levelmax']);coarsest=int(header['Music_levelmin'])
    if not 1<=coarsest<=finest<=20:raise ValueError('Native dyadic hierarchy levels are outside the declared integer domain')
    offset=np.asarray(header['Music_baryon_material_offset_native'],float)/1000
    if offset.shape!=(3,) or not np.isfinite(offset).all() or not np.allclose(offset,np.full(3,box/2**finest/2),rtol=1e-12,atol=0):
        raise ValueError('Actual native gas staggering does not match the common finest-cell convention')
    pieces={name:[] for name in ('particle_ids','species','lagrangian_position_mpc_h','cell_center_mpc_h','cell_width_mpc_h','cell_level')}
    tree={};volume_cells=0;fine_centers={}
    for kind in (0,1,2):
        count=int(counts[kind])
        if not count:continue
        group=native['PartType'+str(kind)]
        ids=exact_uint64_ids(group['ParticleIDs'][:]);q=np.asarray(group['LagrangianCoordinates'][:],float)/1000
        center=np.asarray(group['LagrangianCellCenters'][:],float)/1000;width=np.asarray(group['LagrangianCellWidths'][:],float)/1000
        if len(ids)!=count or q.shape!=(count,3) or center.shape!=q.shape or width.shape!=(count,) or not np.isfinite(q).all() or not np.isfinite(center).all() or not np.isfinite(width).all() or np.any(width<=0):
            raise ValueError('Original native cell geometry has incomplete or nonfinite rows')
        if group['LagrangianCellCenters'].dtype!=np.dtype('float64') or group['LagrangianCoordinates'].dtype!=np.dtype('float64'):
            raise ValueError('Dyadic material projection requires retained double-precision native q and common cell centers')
        if np.any(center<0) or np.any(center>=box) or np.any(q<0) or np.any(q>=box):raise ValueError('Native material q/centers must be periodic and wrapped')
        levels=np.rint(np.log2(box/width)).astype(np.int64)
        if np.any(levels<coarsest) or np.any(levels>finest) or not np.allclose(width,box/np.exp2(levels),rtol=1e-12,atol=0):raise ValueError('Native material widths are not actual dyadic hierarchy volumes')
        if kind in (0,1) and np.any(levels!=finest) or kind==2 and np.any(levels>=finest):raise ValueError('Native material species do not follow fine gas/CDM and combined coarse hierarchy ownership')
        fractional=center/width[:,None]-.5;indices=np.rint(fractional).astype(np.int64)
        if np.any(abs(fractional-indices)>1e-8):raise ValueError('Common native cells do not align to the recorded periodic dyadic partition')
        expected=np.mod(center+(offset if kind==0 else baryon_fraction*offset if kind==2 else 0),box)
        delta=(q-expected+box/2)%box-box/2
        if np.max(abs(delta))>box*2e-13:raise ValueError('Original q is inconsistent with the recorded species staggering or combined coarse COM')
        if kind in (0,1):fine_centers[kind]=center
        if kind!=0:
            for level,index in zip(levels,indices):
                key=(int(level),*(int(value)%2**int(level) for value in index))
                if key in tree:raise ValueError('Original material partition repeats a dyadic cell')
                tree[key]=True;volume_cells+=8**(finest-int(level))
        for name,value in {'particle_ids':ids,'species':np.full(count,kind,np.int8),'lagrangian_position_mpc_h':q,
            'cell_center_mpc_h':center,'cell_width_mpc_h':width,'cell_level':levels}.items():pieces[name].append(value)
    if not np.array_equal(fine_centers[0],fine_centers[1]):raise ValueError('Native gas and CDM finest material cell ordering differs')
    for node in tree:
        level,*index=node
        for parent in range(coarsest,level):
            shift=level-parent;ancestor=(parent,*(value>>shift for value in index))
            if ancestor in tree:raise ValueError('Original material hierarchy overlaps a refined cell and its coarse ancestor')
    if volume_cells!=8**finest:raise ValueError('Original common material leaf cells do not cover the full periodic volume exactly once')
    arrays={name:np.concatenate(values) for name,values in pieces.items()};exact_uint64_ids(arrays['particle_ids'])
    return arrays,{'finest_level':finest,'coarsest_level':coarsest,'baryon_material_offset_mpc_h':offset.tolist(),
        'common_leaf_cells':len(tree),'partition_volume_mpc_h3':box**3,'species_cell_policy':'common CDM partition; baryon partition translated by retained finest-cell staggering'}


def _prepare_material_moment_fields(weight,vector_fields):
    """Validate immutable interpolation grids and cache uniform node bounds."""
    weight=np.asarray(weight,float);n=weight.shape[-1]
    if weight.shape!=(n,n,n) or n<4 or not np.isfinite(weight).all() or np.any(weight<=0):raise ValueError('Positive cubic material grid with at least four nodes per axis required')
    if 'displacement' not in vector_fields or any(np.asarray(field).shape!=(3,n,n,n) or not np.isfinite(np.asarray(field)).all() for field in vector_fields.values()):raise ValueError('Finite aligned material moment vector grids required')
    return {'weight_id':id(weight),'field_ids':{name:id(field) for name,field in vector_fields.items()},
        'maximum_weight':float(np.max(weight)),
        'field_suprema':{name:math.sqrt(sum(float(np.max(np.abs(np.asarray(field)[axis])))**2 for axis in range(3))) for name,field in vector_fields.items()}}


def _periodic_material_cell_moments(weight,vector_fields,center,width,box,*,work_budget,field_error_bounds,weight_error_bound,prepared_fields=None):
    """Exact material moments of periodic piecewise-trilinear retained fields.

    Split the unwrapped cell at every source-grid knot, including periodic
    seams. Two-point Gauss-Legendre in each axis integrates products of two
    trilinear fields and the mass-weighted q moment exactly, apart from
    floating point. This is not exact integration of the Fourier interpolant;
    independent supplied uniform interpolation bounds propagate to averages.
    The local unwrapped frame avoids averaging periodic positions across a
    seam. Coarse species COMs must combine these moments using actual masses.
    """
    import itertools
    center=np.asarray(center,float);weight=np.asarray(weight,float);n=weight.shape[-1]
    width_value=np.asarray(width,float)
    widths=np.full(3,float(width_value)) if width_value.ndim==0 else width_value
    if center.shape!=(3,) or not np.isfinite(center).all() or widths.shape!=(3,) or not np.isfinite(widths).all() or np.any(widths<=0) or np.any(widths>box) or weight.shape!=(n,n,n):raise ValueError('Positive periodic rectangular material cell and scalar mass grid required')
    if set(vector_fields)!=set(field_error_bounds) or any(np.asarray(value).shape!=(3,n,n,n) for value in vector_fields.values()):raise ValueError('Material moments need aligned vector fields and matching interpolation-error bounds')
    if 'displacement' not in vector_fields:raise ValueError('Material centroid requires the actual total species displacement')
    if prepared_fields is None:prepared_fields=_prepare_material_moment_fields(weight,vector_fields)
    if prepared_fields['weight_id']!=id(weight) or prepared_fields['field_ids']!={name:id(field) for name,field in vector_fields.items()}:raise ValueError('Cached material bounds belong to a different retained field allocation')
    if not math.isfinite(weight_error_bound) or weight_error_bound<0 or any(not math.isfinite(value) or value<0 for value in field_error_bounds.values()):raise ValueError('Nonnegative finite uniform interpolation errors required')
    if type(work_budget['maximum_evaluations'])is not int or type(work_budget['evaluations'])is not int or work_budget['maximum_evaluations']<1 or work_budget['evaluations']<0:raise ValueError('Explicit exact positive quadrature work budget required')
    spacing=box/n;intervals=[]
    for axis in range(3):
        low=center[axis]-widths[axis]/2;high=center[axis]+widths[axis]/2
        first=math.floor(low/spacing)+1;last=math.ceil(high/spacing)
        knots=np.arange(first,last,dtype=float)*spacing;knots=knots[(knots>low)&(knots<high)]
        edges=np.r_[low,knots,high];intervals.append(list(zip(edges[:-1],edges[1:])))
    needed=8*math.prod(len(rows) for rows in intervals)
    if needed>work_budget['maximum_evaluations']-work_budget['evaluations']:raise MemoryError('Material quadrature exceeds the frozen source-grid cell evaluation bound')
    work_budget['evaluations']+=needed
    # Trilinear values are convex combinations of grid nodes. Gaussian
    # interior samples alone are not uniform supremum estimates.
    field_suprema=dict(prepared_fields['field_suprema']);maximum_weight=prepared_fields['maximum_weight']
    # Accumulate many subcells with extended precision; publish ordinary
    # double-precision moments in the native contract. Interior interpolation
    # still uses the retained double grids and remains separately bounded.
    total_weight=np.longdouble(0);centroid_numerator=np.zeros(3,dtype=np.longdouble)
    numerators={name:np.zeros(3,dtype=np.longdouble) for name in vector_fields};velocity_second_moment=np.zeros((3,3),dtype=np.longdouble)
    sign=np.asarray(list(itertools.product((-1.,1.),repeat=3)));inverse_sqrt3=1/math.sqrt(3)
    for cube in itertools.product(*intervals):
        low=np.asarray([row[0] for row in cube]);high=np.asarray([row[1] for row in cube]);mid=(low+high)/2;half=(high-low)/2
        points=mid+sign*half*inverse_sqrt3;volume_fraction=float(np.prod((high-low)/widths))/8
        material=_periodic_trilinear(weight,points,box);weighted=material*volume_fraction
        total_weight+=np.sum(weighted,dtype=np.longdouble);maximum_weight=max(maximum_weight,float(material.max()))
        for name,field in vector_fields.items():
            values=_periodic_trilinear(field,points,box);numerators[name]+=np.sum(weighted[:,None]*values,axis=0,dtype=np.longdouble)
            field_suprema[name]=max(field_suprema[name],float(np.linalg.norm(values,axis=1).max()))
            if name=='displacement':centroid_numerator+=np.sum(weighted[:,None]*(points-center+values),axis=0,dtype=np.longdouble)
            if name=='velocity':velocity_second_moment+=np.einsum('n,ni,nj->ij',weighted,values,values,dtype=np.longdouble)
    if not math.isfinite(total_weight) or total_weight<=weight_error_bound:raise ValueError('Material weight integral is not positive beyond its declared interpolation uncertainty')
    averages={name:np.asarray(value/total_weight,dtype=float) for name,value in numerators.items()};centroid=np.asarray(centroid_numerator/total_weight,dtype=float)
    total_weight=float(total_weight)
    errors={}
    for name,average in averages.items():
        field_error=field_error_bounds[name];numerator_error=weight_error_bound*(field_suprema[name]+field_error)+maximum_weight*field_error
        errors[name]=(numerator_error+np.linalg.norm(average)*weight_error_bound)/(total_weight-weight_error_bound)
    position_error=field_error_bounds['displacement'];sup_position=(math.sqrt(3)*float(width_value)/2 if width_value.ndim==0 else float(np.linalg.norm(widths))/2)+field_suprema['displacement']
    errors['centroid_relative_mpc_h']=(weight_error_bound*(sup_position+position_error)+maximum_weight*position_error+np.linalg.norm(centroid)*weight_error_bound)/(total_weight-weight_error_bound)
    return {'mean_material_weight':total_weight,'centroid_relative_mpc_h':centroid,'weighted_vectors':averages,
        'weighted_velocity_second_moment_km_s2':np.asarray(velocity_second_moment/total_weight,dtype=float) if 'velocity' in vector_fields else None,
        'interpolation_error_bounds':errors,'mean_material_weight_error_bound':weight_error_bound,'quadrature_evaluations':needed,
        'integration_scope':'exact tensor Gauss moments of retained piecewise-trilinear fields; floating point and propagated Fourier interpolation errors remain'}


def validate_twofluid_particle_request(parameters):
    request=parameters['request']
    if request.get('schema') not in {'twofluid-native-particle-projection-request-v1','twofluid-native-particle-projection-request-v2'}:raise ValueError('Typed native-q two-fluid particle projection request required')
    if request['schema']=='twofluid-native-particle-projection-request-v2':
        if request.get('projection_mode')!='finite_volume_material_cells':raise ValueError('Version-2 native material projection requires explicit finite-volume cells')
        if type(request['maximum_quadrature_evaluations'])is not int or request['maximum_quadrature_evaluations']<1:raise ValueError('Exact positive native cell quadrature budget required')
    for key in ('maximum_particles','chunk_rows'):
        if type(request[key])is not int or request[key]<1:raise ValueError('Positive exact particle projection bound required: '+key)
    for key in ('memory_gib','displacement_absolute_tolerance_mpc_h','velocity_absolute_tolerance_km_s','material_weight_absolute_tolerance','maximum_mass_rescaling_fraction'):
        if isinstance(request[key],bool) or not isinstance(request[key],(float,int)) or not math.isfinite(request[key]) or request[key]<=0:
            raise ValueError('Explicit positive finite native projection limit required: '+key)
    if request['maximum_mass_rescaling_fraction']>=1:raise ValueError('Native material mass rescaling bound must be below one')
    if not isinstance(request['particle_namespace'],str) or not request['particle_namespace']:raise ValueError('Original native particle identity namespace required')
    return request


def certify_trilinear_material_map(displacement,box,*,minimum_jacobian,maximum_subboxes,maximum_depth):
    """Outward-rounded Jacobian bounds over every retained interpolation cell.

    Each displacement derivative is bilinear in the other two cell coordinates.
    Bounds at their four rectangle corners therefore cover the complete box.
    Adaptive dyadic subdivision tightens determinant interval dependency.
    This certifies the represented spatial interpolant, not physical accuracy.
    """
    import itertools
    displacement=np.asarray(displacement,float)
    if displacement.ndim!=5 or displacement.shape[:2]!=(2,3) or len(set(displacement.shape[2:]))!=1 or not np.isfinite(displacement).all():
        raise ValueError('Finite aligned periodic species displacement grids required')
    n=displacement.shape[-1]
    if n<4 or isinstance(box,bool) or not math.isfinite(box) or box<=0 or not math.isfinite(minimum_jacobian) or minimum_jacobian<=0:
        raise ValueError('Positive box, Jacobian threshold and complete cubic grid required')
    if type(maximum_subboxes)is not int or maximum_subboxes<2*n**3 or type(maximum_depth)is not int or not 0<=maximum_depth<=20:
        raise ValueError('Bounded complete-cell work and dyadic depth in [0,20] required')
    def outward(low,high):
        if not math.isfinite(low) or not math.isfinite(high):raise ArithmeticError('Material-map interval arithmetic overflows')
        return float(np.nextafter(low,-np.inf)),float(np.nextafter(high,np.inf))
    def add(a,b):return outward(a[0]+b[0],a[1]+b[1])
    def multiply(a,b):
        values=(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]);return outward(min(values),max(values))
    scale=outward(n/box,n/box);work=0;deepest=0
    lower=np.empty((2,n,n,n));gradient_bound=np.empty_like(lower)
    positive=((0,1,2),(1,2,0),(2,0,1));negative=((2,1,0),(1,0,2),(0,2,1))
    for species in range(2):
        for cell in itertools.product(range(n),repeat=3):
            nodes={bits:displacement[(species,slice(None),*(int((cell[axis]+bits[axis])%n) for axis in range(3)))]
                for bits in itertools.product((0,1),repeat=3)}
            edges={}
            for axis in range(3):
                others=[q for q in range(3) if q!=axis]
                for component in range(3):
                    values={}
                    for bits in itertools.product((0,1),repeat=2):
                        low_bits=[0,0,0];high_bits=[0,0,0];high_bits[axis]=1
                        for other,bit in zip(others,bits):low_bits[other]=high_bits[other]=bit
                        difference=float(nodes[tuple(high_bits)][component])-float(nodes[tuple(low_bits)][component])
                        values[bits]=multiply(outward(difference,difference),scale)
                    edges[component,axis]=values
            stack=[(np.zeros(3),np.ones(3),0)];cell_lower=math.inf;cell_gradient=0.
            while stack:
                low,high,depth=stack.pop();work+=1;deepest=max(deepest,depth)
                if work>maximum_subboxes:raise RuntimeError('Continuous material-map certificate exhausts its subbox budget')
                matrix={};derivative={}
                for axis in range(3):
                    others=[q for q in range(3) if q!=axis]
                    for component in range(3):
                        evaluations=[]
                        for corner in itertools.product((0,1),repeat=2):
                            t=[high[q] if bit else low[q] for q,bit in zip(others,corner)]
                            value=(0.,0.)
                            for bits,edge in edges[component,axis].items():
                                # Dyadic coordinates/complements are exact in
                                # binary64 at the bounded subdivision depths.
                                weight=(1.,1.)
                                for coordinate,bit in zip(t,bits):
                                    w=float(coordinate if bit else 1-coordinate);weight=multiply(weight,(w,w))
                                value=add(value,multiply(edge,weight))
                            evaluations.append(value)
                        interval=(min(value[0] for value in evaluations),max(value[1] for value in evaluations))
                        derivative[component,axis]=interval
                        matrix[component,axis]=add(interval,(1.,1.)) if component==axis else interval
                determinant=(0.,0.)
                for sign,permutations in ((1,positive),(-1,negative)):
                    for permutation in permutations:
                        term=(1.,1.)
                        for row,column in enumerate(permutation):term=multiply(term,matrix[row,column])
                        if sign<0:term=(-term[1],-term[0])
                        determinant=add(determinant,term)
                if determinant[1]<=minimum_jacobian:
                    raise ValueError('Represented material map cannot meet its minimum Jacobian in a complete interpolation subbox')
                if determinant[0]<=minimum_jacobian:
                    if depth>=maximum_depth:raise RuntimeError('Material-map determinant remains unresolved at the frozen subdivision depth')
                    mid=(low+high)/2
                    for bits in itertools.product((0,1),repeat=3):
                        stack.append((np.where(bits,mid,low),np.where(bits,high,mid),depth+1))
                    continue
                cell_lower=min(cell_lower,determinant[0])
                maximum={key:max(abs(value[0]),abs(value[1])) for key,value in derivative.items()}
                def sum_upper(values):
                    total=(0.,0.)
                    for value in values:total=add(total,(value,value))
                    return total[1]
                row_max=max(sum_upper(maximum[row,column] for column in range(3)) for row in range(3))
                column_max=max(sum_upper(maximum[row,column] for row in range(3)) for column in range(3))
                norm=float(np.nextafter(math.sqrt(multiply((row_max,row_max),(column_max,column_max))[1]),np.inf))
                cell_gradient=max(cell_gradient,norm)
            lower[(species,*cell)]=cell_lower;gradient_bound[(species,*cell)]=cell_gradient
    global_bound=float(gradient_bound.max())
    return ({'schema':'trilinear-material-map-domain-v1','status':'COMPLETE_POSITIVE_CELL_JACOBIAN_BOUNDS',
        'subboxes_evaluated':work,'maximum_depth_reached':deepest,'minimum_jacobian_lower_bound':float(lower.min()),
        'maximum_displacement_gradient_operator_bound':global_bound,
        'global_injectivity_status':'CONTRACTION_BOUND_SUPPORTED' if global_bound<1 else 'NOT_ESTABLISHED_BY_CONTRACTION_BOUND',
        'scope':'complete piecewise-trilinear interpolant cells; global injectivity additionally supported only when the displacement Lipschitz bound is below one; no unresolved physical shell-crossing or solver-accuracy claim'},
        {'interpolant_jacobian_lower_bound':lower,'interpolant_displacement_gradient_operator_bound':gradient_bound})


def _twofluid_material_fields(model,fields):
    """Select genuine perturbative or full material fields without relabeling."""
    kind=model.get('schema')
    if kind=='finite-epoch-twofluid-relative-v1':
        return (fields['displacement_first_order_mpc_h']+fields['displacement_second_order_mpc_h'],
            fields['velocity_peculiar_km_s'],np.zeros(3),{
                'displacement_first_order':fields['displacement_first_order_mpc_h'],
                'velocity_first_order':fields['velocity_first_order_peculiar_km_s']})
    if kind!='finite-epoch-nonlinear-twofluid-reference-v1':raise ValueError('Unsupported measured species material flow product')
    origin=np.asarray(model['material_grid_origin_mpc_h'],float);n=fields['lagrangian_mass_weights'].shape[-1]
    if origin.shape!=(3,) or not np.allclose(origin,.5*model['request']['box_mpc_h']/n,rtol=1e-12,atol=0.):
        raise ValueError('Nonlinear material labels do not match the declared cell-centred grid')
    if model['maximum_material_density_relative_error']>model['request']['maximum_material_density_relative_error']:
        raise ValueError('Nonlinear material map exceeds its measured density-consistency budget')
    J=np.asarray(fields['material_jacobian']);F=np.asarray(fields['material_deformation_gradient'])
    if J.shape!=(2,n,n,n) or F.shape!=(2,3,3,n,n,n) or not np.isfinite(J).all() or not np.isfinite(F).all() or np.any(J<=model['request']['minimum_material_jacobian']):
        raise ValueError('Nonlinear material product lacks its actual supported sampled deformation')
    certificate=model.get('interpolated_material_map_domain',{});bounds=np.asarray(fields.get('interpolant_jacobian_lower_bound'))
    if (certificate.get('schema')!='trilinear-material-map-domain-v1' or certificate.get('status')!='COMPLETE_POSITIVE_CELL_JACOBIAN_BOUNDS'
            or bounds.shape!=(2,n,n,n) or not np.isfinite(bounds).all() or np.any(bounds<=model['request']['material_map_minimum_jacobian'])
            or float(bounds.min())!=certificate['minimum_jacobian_lower_bound']):
        raise ValueError('Nonlinear native projection lacks complete interpolated-cell Jacobian bounds')
    if model['request']['require_global_material_injectivity'] and certificate['global_injectivity_status']!='CONTRACTION_BOUND_SUPPORTED':
        raise ValueError('Native projection changes the requested global material-map injectivity requirement')
    return fields['material_displacement_mpc_h'],fields['material_velocity_peculiar_km_s'],origin,{}


def twofluid_particle_input_bindings(parameters):
    from pathlib import Path
    from .sidm_analysis import _load_product
    from .foundation import file_sha256
    request=validate_twofluid_particle_request(parameters);memory=request['memory_gib']*1024**3
    report,fields,sources=_load_product(request['relative_fields'],parameters['request_base'],memory/2)
    if report.get('schema') not in {'finite-epoch-twofluid-relative-v1','finite-epoch-nonlinear-twofluid-reference-v1'} or report['species_order']!=['cdm','baryons']:
        raise ValueError('Native particle projection requires the sealed evolved two-fluid species fields')
    _twofluid_material_fields(report,fields)
    if report['schema']=='finite-epoch-nonlinear-twofluid-reference-v1' and request['schema']!='twofluid-native-particle-projection-request-v2':
        raise ValueError('Full nonlinear maps require finite-volume material-cell projection; perturbative point diagnostics are unavailable')
    reference=request['native_particle_load'];path=Path(reference['path']).expanduser()
    if not path.is_absolute():path=Path(parameters['request_base'])/path
    path=path.resolve()
    if not path.is_file() or path.is_symlink() or file_sha256(path)!=reference['sha256']:raise ValueError('Original native particle load changed')
    sources[str(path)]=reference['sha256'];return sources


def publish_twofluid_particle_projection(parameters,destination):
    """Project evolved species flow onto exact native uniform-parent q/IDs.

    This is a source-bound materialization product. It does not certify IC
    transfer/epoch convergence and does not substitute point samples for a
    coarse cell's finite-volume combined species moments.
    """
    if parameters['request'].get('schema')=='twofluid-native-particle-projection-request-v2':
        return publish_twofluid_material_projection(parameters,destination)
    from pathlib import Path
    import h5py
    from .sidm_analysis import _load_product
    from .data import exact_uint64_ids,export_numeric_audit,snapshot_total_counts
    from .foundation import file_sha256,KPC_CM,G_MPC_KMS2_MSUN
    request=validate_twofluid_particle_request(parameters);memory=request['memory_gib']*1024**3
    sources=twofluid_particle_input_bindings(parameters)
    model,fields,_=_load_product(request['relative_fields'],parameters['request_base'],memory/2)
    load=Path(request['native_particle_load']['path']).expanduser()
    if not load.is_absolute():load=Path(parameters['request_base'])/load
    load=load.resolve();spec=model['request'];box=float(spec['box_mpc_h']);a=float(model['target_scale_factor']);h=float(spec['hubble_parameter'])
    if not math.isclose(h,float(spec['hubble0_km_s_mpc'])/100,rel_tol=1e-12):raise ValueError('Native particle mass/length units require h=H0/100')
    displacement=fields['displacement_first_order_mpc_h']+fields['displacement_second_order_mpc_h'];velocity=fields['velocity_peculiar_km_s'];weights=fields['lagrangian_mass_weights']
    n=weights.shape[-1]
    if displacement.shape!=(2,3,n,n,n) or velocity.shape!=displacement.shape or weights.shape!=(2,n,n,n):raise ValueError('Evolved species projection field dimensions changed')
    bounds={};limits={'displacement':request['displacement_absolute_tolerance_mpc_h'],'velocity':request['velocity_absolute_tolerance_km_s'],'material_weight':request['material_weight_absolute_tolerance']}
    limits.update(displacement_first_order=limits['displacement'],velocity_first_order=limits['velocity'])
    for name,value in [('displacement',displacement),('velocity',velocity),('material_weight',weights),
        ('displacement_first_order',fields['displacement_first_order_mpc_h']),('velocity_first_order',fields['velocity_first_order_peculiar_km_s'])]:
        bounds[name]=np.asarray([_spectral_trilinear_bound(value[s],box) for s in range(2)])
        if np.max(np.linalg.norm(bounds[name],axis=1))>limits[name]:raise ValueError('Retained grid interpolation exceeds its declared '+name+' scalar/vector norm tolerance; refine the source field grid')
    pieces={name:[] for name in ('particle_ids','species','lagrangian_position_mpc_h','position_comoving_mpc_h','velocity_peculiar_km_s','displacement_first_order_mpc_h','velocity_first_order_peculiar_km_s','material_weight')}
    with h5py.File(load,'r') as native:
        header=native['Header'].attrs;counts=snapshot_total_counts(header);total=int(counts.sum())
        if int(header.get('NumFilesPerSnapshot',1))!=1 or any(counts[k] for k in (2,3,4,5)) or counts[0]!=counts[1] or counts[0]==0:
            raise ValueError('This projection requires a complete two-species uniform native parent; coarse windows and sharded/nested loads have a separate materializer')
        if total>request['maximum_particles'] or total*1024+int(request['chunk_rows'])*512>memory/2:raise MemoryError('Native uniform species projection exceeds its declared particle workspace')
        cells=round(int(counts[0])**(1/3))
        if cells**3!=int(counts[0]) or cells<n:raise ValueError('Native uniform load must resolve every retained source-grid mode')
        if header.get('Music_lagrangian_schema')!=1 or not math.isclose(float(header.get('Music_lagrangian_unitlength_cm',math.nan)),KPC_CM,rel_tol=1e-12):
            raise ValueError('Native projection requires original pre-displacement q in comoving kpc/h')
        if not math.isclose(float(header['BoxSize'])/1000,box,rel_tol=1e-12) or not math.isclose(float(header['Time']),a,rel_tol=1e-12) or not math.isclose(float(header['HubbleParam']),h,rel_tol=1e-12):
            raise ValueError('Native q-load and evolved species fields change the actual box/epoch/h convention')
        for name,value in {'Omega0':float(spec['omega_m0']),'OmegaBaryon':float(spec['omega_m0'])*float(spec['baryon_fraction'])}.items():
            if not math.isclose(float(header[name]),value,rel_tol=1e-10):raise ValueError('Native particle load and evolved flow background density differ: '+name)
        for kind,species in ((0,1),(1,0)):
            group=native['PartType'+str(kind)];ids=exact_uint64_ids(group['ParticleIDs'][:]);q=np.asarray(group['LagrangianCoordinates'][:],float)/1000
            if len(ids)!=int(counts[kind]) or q.shape!=(len(ids),3) or not np.isfinite(q).all() or np.any(q<0) or np.any(q>=box):raise ValueError('Native parent q/ID census is incomplete or unwrapped')
            spacing=box/cells;offset=q[0]/spacing;lattice=q/spacing-offset;rounded=np.rint(lattice)
            index=rounded.astype(np.int64)%cells;linear=(index[:,0]*cells+index[:,1])*cells+index[:,2]
            if np.any(np.abs(lattice-rounded)>1e-9) or len(np.unique(linear))!=len(ids):raise ValueError('Native species q does not cover one complete periodic lattice')
            for start in range(0,len(ids),request['chunk_rows']):
                stop=min(start+request['chunk_rows'],len(ids));rows=q[start:stop]
                for name,value in {'particle_ids':ids[start:stop],'species':np.full(stop-start,kind,np.int8),'lagrangian_position_mpc_h':rows,
                    'position_comoving_mpc_h':np.mod(rows+_periodic_trilinear(displacement[species],rows,box),box),
                    'velocity_peculiar_km_s':_periodic_trilinear(velocity[species],rows,box),'material_weight':_periodic_trilinear(weights[species],rows,box)}.items():pieces[name].append(value)
                pieces['displacement_first_order_mpc_h'].append(_periodic_trilinear(fields['displacement_first_order_mpc_h'][species],rows,box))
                pieces['velocity_first_order_peculiar_km_s'].append(_periodic_trilinear(fields['velocity_first_order_peculiar_km_s'][species],rows,box))
    arrays={name:np.concatenate(parts) for name,parts in pieces.items()};exact_uint64_ids(arrays['particle_ids'])
    if not np.isfinite(arrays['material_weight']).all() or np.any(arrays['material_weight']<=0):raise ValueError('Native material projection has nonpositive particle weights')
    rho=float(spec['omega_m0'])*3*float(spec['hubble0_km_s_mpc'])**2/(8*np.pi*G_MPC_KMS2_MSUN)
    total_mass=rho*(box/h)**3;fraction=float(spec['baryon_fraction']);mass=np.empty(total);rescaling={}
    for kind,share in ((0,fraction),(1,1-fraction)):
        use=arrays['species']==kind;mean=float(arrays['material_weight'][use].mean());factor=1/mean
        if abs(factor-1)>request['maximum_mass_rescaling_fraction']:raise ValueError('Native material quadrature violates the frozen background mass normalization bound')
        mass[use]=total_mass*share/np.sum(use)*arrays['material_weight'][use]*factor;rescaling[str(kind)]=factor
    arrays['mass_msun']=mass
    report={'schema':'twofluid-native-particle-projection-v1','execution_status':'COMPLETE','request':request,'epoch':model['epoch'],
        'relative_fields':{'path':str((Path(parameters['request_base'])/request['relative_fields']['path']).resolve()),'sha256':request['relative_fields']['sha256']},
        'particle_namespace':request['particle_namespace'],'source_hashes':sources,'species_order':[0,1],
        'native_uniform_cells_per_axis':cells,'mass_rescaling':rescaling,'interpolation_error_bounds':{key:value.tolist() for key,value in bounds.items()},
        'measured_total_momentum_msun_km_s':np.sum(mass[:,None]*arrays['velocity_peculiar_km_s'],axis=0).tolist(),
        'bulk_mode_policy':'retain the supplied finite-epoch flow; no unrequested zero-momentum subtraction',
        'mass_definition':'second-order material pullback times original uniform Lagrangian volume; separately normalized actual baryon/CDM background masses',
        'supported_domain':'uniform two-species native parent q-load; no coarse combined-species point sampling','science_acceptance':'NOT_ESTABLISHED'}
    units={'particle_ids':'identity','species':'native species enum','lagrangian_position_mpc_h':'comoving Mpc/h','position_comoving_mpc_h':'comoving Mpc/h',
        'velocity_peculiar_km_s':'km / s','displacement_first_order_mpc_h':'comoving Mpc/h',
        'velocity_first_order_peculiar_km_s':'km / s','material_weight':'dimensionless','mass_msun':'Msun'}
    specifications={name:{'units':units[name],'quantity':'actual native-q two-fluid '+name.replace('_',' ')} for name in arrays}
    specifications['particle_ids']['identity_namespace']=request['particle_namespace']
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    return export_numeric_audit(destination,report,arrays,specifications,algorithm=Path(__file__).resolve(),
        context={'epoch':model['epoch'],'coordinates':{'frame':'original native species periodic Lagrangian q; evolved comoving position and peculiar velocity'}},source_hashes=sources)


def publish_twofluid_material_projection(parameters,destination):
    """Integrate both species on original leaf cells, then combine coarse COMs."""
    from pathlib import Path
    import h5py
    from .sidm_analysis import _load_product
    from .data import export_numeric_audit
    from .foundation import G_MPC_KMS2_MSUN,file_sha256
    request=validate_twofluid_particle_request(parameters)
    if request['schema']!='twofluid-native-particle-projection-request-v2':raise ValueError('Explicit finite-volume material request required')
    memory=request['memory_gib']*1024**3;sources=twofluid_particle_input_bindings(parameters)
    model,fields,_=_load_product(request['relative_fields'],parameters['request_base'],memory/2);spec=model['request']
    box=float(spec['box_mpc_h']);h=float(spec['hubble_parameter']);a=float(spec['target_scale_factor']);fraction=float(spec['baryon_fraction'])
    if not math.isclose(h,float(spec['hubble0_km_s_mpc'])/100,rel_tol=1e-12):raise ValueError('Native material units require h=H0/100')
    load=(Path(parameters['request_base'])/request['native_particle_load']['path']).resolve()
    with h5py.File(load,'r') as native:
        for name,value in {'Time':a,'BoxSize':box*1000,'HubbleParam':h,'Omega0':float(spec['omega_m0']),'OmegaBaryon':float(spec['omega_m0'])*fraction}.items():
            if not math.isclose(float(native['Header'].attrs[name]),value,rel_tol=1e-10):raise ValueError('Native material load differs from the actual evolved '+name)
        arrays,geometry=_native_material_geometry(native,box,fraction,maximum_particles=request['maximum_particles'],memory_bytes=memory/2)
    total=len(arrays['particle_ids']);n=fields['lagrangian_mass_weights'].shape[-1]
    if n>2**geometry['finest_level']:raise ValueError('Actual native finest cell resolution does not cover retained source-grid modes')
    if total*4096+n**3*1024>memory/2:raise MemoryError('Finite-volume species and moment/error arrays exceed their declared workspace')
    displacement,velocity,origin,diagnostics=_twofluid_material_fields(model,fields)
    weight=fields['lagrangian_mass_weights'];vectors={'displacement':displacement,'velocity':velocity,**diagnostics}
    if weight.shape!=(2,n,n,n) or displacement.shape!=(2,3,n,n,n) or velocity.shape!=displacement.shape or not np.isfinite(weight).all() or np.any(weight<=0):
        raise ValueError('Finite positive material weights and aligned species flow vectors required')
    names=list(vectors);species_weights=[weight[index] for index in range(2)];species_vectors=[{name:value[index] for name,value in vectors.items()} for index in range(2)]
    prepared=[_prepare_material_moment_fields(species_weights[index],species_vectors[index]) for index in range(2)]
    bounds=[{name:float(np.linalg.norm(_spectral_trilinear_bound(field,box))) for name,field in species_vectors[index].items()} for index in range(2)]
    weight_errors=np.asarray([float(_spectral_trilinear_bound(value,box)[0]) for value in species_weights])
    if np.max(weight_errors)>request['material_weight_absolute_tolerance']:raise ValueError('Retained material interpolation exceeds the frozen weight tolerance')
    work={'maximum_evaluations':request['maximum_quadrature_evaluations'],'evaluations':0}
    moments=np.zeros((total,2,len(names),3));centroids=np.zeros((total,2,3));moment_errors=np.zeros((total,2,len(names)));centroid_errors=np.zeros((total,2))
    mean_weights=np.zeros((total,2));velocity_second_moments=np.zeros((total,2,3,3));volume=arrays['cell_width_mpc_h']**3;offset=np.asarray(geometry['baryon_material_offset_mpc_h'])
    for row,kind in enumerate(arrays['species']):
        for species in ((1,) if kind==0 else (0,) if kind==1 else (0,1)):
            center=arrays['cell_center_mpc_h'][row]+(offset if species==1 else 0)-origin
            value=_periodic_material_cell_moments(species_weights[species],species_vectors[species],center,arrays['cell_width_mpc_h'][row],box,
                work_budget=work,field_error_bounds=bounds[species],weight_error_bound=weight_errors[species],prepared_fields=prepared[species])
            mean_weights[row,species]=value['mean_material_weight'];centroids[row,species]=value['centroid_relative_mpc_h']+(offset if species==1 else 0)
            velocity_second_moments[row,species]=value['weighted_velocity_second_moment_km_s2']
            centroid_errors[row,species]=value['interpolation_error_bounds']['centroid_relative_mpc_h']
            for index,name in enumerate(names):
                moments[row,species,index]=value['weighted_vectors'][name];moment_errors[row,species,index]=value['interpolation_error_bounds'][name]
    rho=float(spec['omega_m0'])*3*float(spec['hubble0_km_s_mpc'])**2/(8*np.pi*G_MPC_KMS2_MSUN)
    shares=np.asarray([1-fraction,fraction]);raw_mass=rho/h**3*volume[:,None]*mean_weights*shares
    species_mass=raw_mass.copy();mass_errors=np.zeros_like(raw_mass);rescaling={}
    for species in range(2):
        active=mean_weights[:,species]>0;actual=float(raw_mass[:,species].sum());target=rho*(box/h)**3*shares[species];factor=target/actual
        if not math.isfinite(factor) or abs(factor-1)>request['maximum_mass_rescaling_fraction']:raise ValueError('Material integral violates the declared global species normalization bound')
        species_mass[:,species]*=factor;rescaling[str(species)]=factor
        global_relative_error=float(np.sum(volume[active])*rho/h**3*shares[species]*weight_errors[species]/actual)
        if global_relative_error>=1:raise ValueError('Species background normalization is not resolved beyond its interpolation uncertainty')
        relative_error=(weight_errors[species]/mean_weights[active,species]+global_relative_error)/(1-global_relative_error)
        mass_errors[active,species]=species_mass[active,species]*relative_error
    mass=species_mass.sum(axis=1);fractional=species_mass/mass[:,None];uncertainty=mass_errors.sum(axis=1)
    if np.any(mass<=uncertainty):raise ValueError('Combined native material masses are not positive beyond their propagated species error')
    combined=np.sum(fractional[:,:,None,None]*moments,axis=1);centroid=np.sum(fractional[:,:,None]*centroids,axis=1)
    centroid_bound=(np.sum(species_mass*centroid_errors,axis=1)+np.sum(mass_errors*(np.linalg.norm(centroids,axis=2)+np.linalg.norm(centroid,axis=1)[:,None]),axis=1))/(mass-uncertainty)
    vector_bounds=(np.sum(species_mass[:,:,None]*moment_errors,axis=1)+np.sum(mass_errors[:,:,None]*(np.linalg.norm(moments,axis=3)+np.linalg.norm(combined,axis=2)[:,None,:]),axis=1))/(mass-uncertainty)[:,None]
    if np.max(centroid_bound)>request['displacement_absolute_tolerance_mpc_h'] or np.max(vector_bounds[:,names.index('displacement')])>request['displacement_absolute_tolerance_mpc_h']:raise ValueError('Material centroid/displacement uncertainty exceeds the declared bound')
    if 'displacement_first_order' in names and np.max(vector_bounds[:,names.index('displacement_first_order')])>request['displacement_absolute_tolerance_mpc_h']:
        raise ValueError('First-order material displacement uncertainty exceeds the declared bound')
    for name in [name for name in ('velocity','velocity_first_order') if name in names]:
        if np.max(vector_bounds[:,names.index(name)])>request['velocity_absolute_tolerance_km_s']:raise ValueError('Material '+name+' uncertainty exceeds its declared bound')
    integrated_speed_squared=np.sum(fractional*np.trace(velocity_second_moments,axis1=2,axis2=3),axis=1)
    collapsed_speed_squared=np.sum(combined[:,names.index('velocity')]**2,axis=1);unresolved=integrated_speed_squared-collapsed_speed_squared
    kinetic_roundoff=4096*np.finfo(float).eps*np.maximum(1,integrated_speed_squared)
    if np.any(unresolved < -kinetic_roundoff):raise ValueError('Material velocity second moments are inconsistent with their mean beyond the floating-point allowance')
    unresolved=np.maximum(unresolved,0)
    arrays.update(position_comoving_mpc_h=np.mod(arrays['cell_center_mpc_h']+centroid,box),velocity_peculiar_km_s=combined[:,names.index('velocity')],
        mass_msun=mass,material_weight=np.sum(shares[None,:]*mean_weights,axis=1),species_mass_msun=species_mass,
        species_mass_absolute_error_msun=mass_errors,species_centroid_relative_mpc_h=centroids,
        species_centroid_absolute_error_mpc_h=centroid_errors,
        species_velocity_absolute_error_km_s=moment_errors[:,:,names.index('velocity')],
        species_velocity_peculiar_km_s=moments[:,:,names.index('velocity')],centroid_absolute_error_mpc_h=centroid_bound,
        velocity_absolute_error_km_s=vector_bounds[:,names.index('velocity')],species_velocity_second_moment_km_s2=velocity_second_moments,
        unresolved_material_kinetic_energy_msun_km_s2=.5*mass*unresolved)
    if diagnostics:
        arrays.update(displacement_first_order_mpc_h=combined[:,names.index('displacement_first_order')],
            velocity_first_order_peculiar_km_s=combined[:,names.index('velocity_first_order')])
    else:arrays['material_displacement_mpc_h']=combined[:,names.index('displacement')]
    geometry['source_grid_cells_per_axis']=n
    report={'schema':'twofluid-native-particle-projection-v1','execution_status':'COMPLETE','request':request,'epoch':model['epoch'],
        'relative_fields':{'path':str((Path(parameters['request_base'])/request['relative_fields']['path']).resolve()),'sha256':request['relative_fields']['sha256']},
        'projection_mode':'finite_volume_material_cells','material_geometry':geometry,'particle_namespace':request['particle_namespace'],
        'flow_kind':'second_order_perturbative' if diagnostics else 'full_nonlinear_material',
        'material_grid_origin_mpc_h':origin.tolist(),'first_order_diagnostics_available':bool(diagnostics),
        'source_hashes':sources,'species_order':np.unique(arrays['species']).tolist(),'mass_component_order':['cdm','baryons'],'mass_rescaling':rescaling,
        'quadrature_evaluations':work['evaluations'],'measured_total_momentum_msun_km_s':np.sum(mass[:,None]*arrays['velocity_peculiar_km_s'],axis=0).tolist(),
        'mass_definition':'actual integrated species material weights times original leaf volumes; separate bounded global species normalization',
        'coarse_combination':'actual normalized species mass/momentum/position moments in a common unwrapped cell frame',
        'kinetic_energy_policy':'retain unresolved material velocity second moments as diagnostics; a single coarse point retains COM kinetic energy only, with no invented heat injection',
        'supported_domain':'original complete periodic dyadic native leaf partition and retained finite-epoch species Fourier grid; no missing source-grid modes inferred',
        'science_acceptance':'NOT_ESTABLISHED'}
    units={'particle_ids':'identity','species':'native species enum','cell_level':'dyadic hierarchy level'}
    for name in arrays:
        if name not in units:units[name]='Msun' if 'msun' in name else 'km / s' if 'km_s' in name else 'dimensionless' if name=='material_weight' else 'comoving Mpc/h'
    units['species_velocity_second_moment_km_s2']='(km / s)^2';units['unresolved_material_kinetic_energy_msun_km_s2']='Msun * (km / s)^2'
    specifications={name:{'units':units[name],'quantity':'integrated actual native material '+name.replace('_',' ')} for name in arrays};specifications['particle_ids']['identity_namespace']=request['particle_namespace']
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    return export_numeric_audit(destination,report,arrays,specifications,algorithm=Path(__file__).resolve(),context={'epoch':model['epoch'],
        'coordinates':{'frame':'original common periodic dyadic material cells; species-specific staggering and unwrapped moments'}},source_hashes=sources)


def twofluid_particle_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native particle projection exceeds allocated stage memory')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Native particle projection escapes its owned generation')
    return publish_twofluid_particle_projection(parameters,destination)


def twofluid_particle_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Project finite-epoch two-fluid fields onto actual native q/IDs by uniform points or finite-volume material cells')
    parser.add_argument('--parameters',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();parameters=json.loads(args.parameters.read_text());parameters.setdefault('request_base',str(args.parameters.resolve().parent))
    publish_twofluid_particle_projection(parameters,args.output)


def _twofluid_overlap_moment_integrator(destination,projection,model,fields,box,work_budget):
    """Integrate actual retained-flow subleaf moments with original normalization."""
    from .foundation import G_MPC_KMS2_MSUN
    displacement,velocity,origin,diagnostics=_twofluid_material_fields(model,fields)
    if diagnostics:raise ValueError('Nonnested overlap comparison requires the full nonlinear material flow')
    weights=[np.asarray(fields['lagrangian_mass_weights'][species],float) for species in range(2)]
    vectors=[{'displacement':displacement[species],'velocity':velocity[species]} for species in range(2)]
    prepared=[_prepare_material_moment_fields(weights[species],vectors[species]) for species in range(2)]
    bounds=[{name:float(np.linalg.norm(_spectral_trilinear_bound(value,box))) for name,value in vectors[species].items()} for species in range(2)]
    weight_error=[float(_spectral_trilinear_bound(value,box)[0]) for value in weights]
    offset=np.asarray(projection['material_geometry']['baryon_material_offset_mpc_h'],float)
    if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('Original finite physical species offset required')
    spec=model['request'];fraction=float(spec['baryon_fraction']);h=float(spec['hubble_parameter'])
    rho=float(spec['omega_m0'])*3*float(spec['hubble0_km_s_mpc'])**2/(8*np.pi*G_MPC_KMS2_MSUN)
    factors=[];global_error=[]
    for species in range(2):
        mean=float(np.mean(weights[species],dtype=np.longdouble))
        factor=projection['mass_rescaling'][str(species)]
        if type(factor)not in (int,float) or not math.isfinite(factor) or factor<=0 or not math.isclose(factor,1/mean,rel_tol=4096*np.finfo(float).eps,abs_tol=0):raise ValueError('Published projection normalization differs from the actual complete material field')
        relative=weight_error[species]/mean
        if not 0<=relative<1:raise ValueError('Overlap global mass normalization is unresolved')
        factors.append(factor);global_error.append(relative)
    def region(species,center,width):
        center=np.asarray(center,float);width_value=np.asarray(width,float)
        value=_periodic_material_cell_moments(weights[species],vectors[species],center-origin,width,box,
            work_budget=work_budget,field_error_bounds=bounds[species],weight_error_bound=weight_error[species],prepared_fields=prepared[species])
        work_budget['regions']=work_budget.get('regions',0)+1
        if width_value.ndim and np.any(width_value!=width_value[0]):work_budget['rectangular_regions']=work_budget.get('rectangular_regions',0)+1
        mean=value['mean_material_weight'];share=fraction if species==1 else 1-fraction
        volume=float(width_value)**3 if width_value.ndim==0 else float(np.prod(width_value))
        mass=rho/h**3*volume*mean*share*factors[species]
        error=mass*(weight_error[species]/mean+global_error[species])/(1-global_error[species])
        position=center+value['centroid_relative_mpc_h'];flow=value['weighted_vectors']['velocity']
        return mass,error,position,flow,value['interpolation_error_bounds']['centroid_relative_mpc_h'],value['interpolation_error_bounds']['velocity']
    def integrate(species,owner,source_row):
        center=np.asarray(destination['cell_center_mpc_h'][owner],float)+(offset if species==1 else 0)
        return region(species,center,float(destination['cell_width_mpc_h'][owner]))
    integrate.region=region
    return integrate


def _twofluid_periodic_leaf_overlap_geometry(destination,source,box,destination_offset,source_offset,*,maximum_work):
    """Stream exact dyadic-lattice rectangular intersections at changed offsets."""
    import itertools
    if type(maximum_work)is not int or maximum_work<1:raise MemoryError('Periodic leaf overlap matching work exhausted')
    if not math.isfinite(box) or box<=0:raise ValueError('Finite positive material box required')
    for data in (destination,source):
        if not len(data['mass_msun']) or np.asarray(data['cell_level']).dtype.kind not in 'iu' or np.asarray(data['species']).dtype.kind not in 'iu':raise ValueError('Exact nonempty species/level leaf geometry required')
    finest=max(int(np.max(data['cell_level'])) for data in (destination,source))
    if not 1<=finest<=20:raise ValueError('Bounded dyadic leaf levels required')
    lattice=2**(finest+1);budget={'work':0}
    def tick():
        budget['work']+=1
        if budget['work']>maximum_work:raise MemoryError('Periodic leaf overlap matching work exhausted')
    def offset_units(value):
        value=np.asarray(value,float)
        if value.shape!=(3,) or not np.isfinite(value).all() or np.any(value<0) or np.any(value>=box):raise ValueError('Original finite periodic species offset required')
        scaled=value/box*lattice;rounded=np.rint(scaled)
        if np.any(abs(scaled-rounded)>1e-8):raise ValueError('Species staggering does not lie on the recorded half-finest dyadic lattice')
        return tuple(int(x) for x in rounded)
    dest_offset=offset_units(destination_offset);src_offset=offset_units(source_offset)
    def partition(data):
        leaves={};nodes={};records=[];expected=np.zeros((len(data['mass_msun']),2),np.uint64)
        for row in range(len(data['mass_msun'])):
            tick();level=int(data['cell_level'][row]);center=np.asarray(data['cell_center_mpc_h'][row],float);width=float(data['cell_width_mpc_h'][row])
            if not 1<=level<=finest or not np.isfinite(center).all() or center.shape!=(3,) or not math.isclose(width,box/2**level,rel_tol=1e-12):raise ValueError('Original periodic dyadic cell geometry required')
            index=np.rint(center/width-.5).astype(np.int64)
            if np.any(center<0) or np.any(center>=box) or np.any(abs(center/width-.5-index)>1e-8):raise ValueError('Dyadic leaf alignment differs')
            kind=int(data['species'][row])
            if kind not in (0,1,2):raise ValueError('Unsupported native leaf species')
            active=(1,) if kind==0 else (0,) if kind==1 else (0,1)
            if any((not math.isfinite(float(data['species_mass_msun'][row,sp])) or (data['species_mass_msun'][row,sp]<=0 if sp in active else data['species_mass_msun'][row,sp]!=0)) for sp in (0,1)):raise ValueError('Leaf species ownership differs')
            units=lattice//2**level;low=tuple(int(x)*units for x in index)
            for sp in active:
                key=(sp,level,*(int(x) for x in index))
                if key in leaves:raise ValueError('Periodic species partition repeats a leaf')
                leaves[key]=row;expected[row,sp]=np.uint64(units**3);records.append((row,sp,low,units))
        for key,row in leaves.items():
            sp,level,*index=key
            nodes[key]=row
            for ancestor in range(level):
                tick();parent=(sp,ancestor,*(x//2**(level-ancestor) for x in index))
                if parent in leaves:raise ValueError('Periodic species leaves overlap an ancestor')
                nodes.setdefault(parent,None)
        if any(sum(int(value) for value in expected[:,sp])!=lattice**3 for sp in (0,1)):raise ValueError('Species leaves do not exactly cover the periodic material box')
        return nodes,records,expected
    nodes,_,destination_expected=partition(destination)
    _,records,source_expected=partition(source)
    def intersections():
        for source_row,sp,base_low,width in records:
            delta=tuple(src_offset[q]-dest_offset[q] if sp==1 else 0 for q in range(3))
            low=tuple(base_low[q]+delta[q] for q in range(3));high=tuple(low[q]+width for q in range(3))
            images=[range(low[q]//lattice,(high[q]-1)//lattice+1) for q in range(3)]
            for image in itertools.product(*images):
                part_low=tuple(max(low[q],image[q]*lattice)-image[q]*lattice for q in range(3))
                part_high=tuple(min(high[q],(image[q]+1)*lattice)-image[q]*lattice for q in range(3))
                stack=[(0,0,0,0)]
                while stack:
                    level,*index=stack.pop();tick();span=lattice//2**level
                    left=tuple(max(part_low[q],index[q]*span) for q in range(3));right=tuple(min(part_high[q],(index[q]+1)*span) for q in range(3))
                    if any(left[q]>=right[q] for q in range(3)):continue
                    key=(sp,level,*index)
                    if key not in nodes:raise ValueError('Periodic overlap traversal found a missing material volume')
                    owner=nodes[key]
                    if owner is not None:
                        yield owner,source_row,sp,left,right,image
                    else:
                        if level>=finest:raise ValueError('Periodic overlap tree exceeds declared leaf depth')
                        stack.extend((level+1,*(2*index[q]+child[q] for q in range(3))) for child in itertools.product(range(2),repeat=3))
    return intersections(),budget,lattice,destination_expected,source_expected


def _restrict_twofluid_staggered_hierarchy(destination,source,box,destination_offset,source_offset,*,maximum_work,intersection_moments,recovery_diagnostics):
    """Stream retained-field rectangular moments in both original unwrap frames."""
    pieces,budget,lattice,expected,source_expected=_twofluid_periodic_leaf_overlap_geometry(destination,source,box,destination_offset,source_offset,maximum_work=maximum_work)
    count=len(destination['mass_msun']);mass=np.zeros((count,2),np.longdouble);mass_error=np.zeros_like(mass)
    position=np.zeros((count,2,3),np.longdouble);momentum=np.zeros_like(position)
    position_error=np.zeros_like(mass);momentum_error=np.zeros_like(mass);coverage=np.zeros((count,2),np.uint64)
    source_seen=np.zeros(source_expected.shape,bool);current=None;totals=None;group_volume=0
    def recover():
        if current is None:return
        row,sp=current
        if source_seen[row,sp] or group_volume!=int(source_expected[row,sp]):raise ValueError('Rectangular overlaps do not exactly cover the original source species leaf')
        source_seen[row,sp]=True
        m=np.longdouble(source['species_mass_msun'][row,sp]);me=np.longdouble(source['species_mass_absolute_error_msun'][row,sp])
        x=np.asarray(source['cell_center_mpc_h'][row]+source['species_centroid_relative_mpc_h'][row,sp],dtype=np.longdouble)
        v=np.asarray(source['species_velocity_peculiar_km_s'][row,sp],dtype=np.longdouble)
        xe=source['species_centroid_absolute_error_mpc_h'][row,sp];ve=source['species_velocity_absolute_error_km_s'][row,sp]
        if not np.isfinite([m,me,xe,ve]).all() or not np.isfinite(x).all() or not np.isfinite(v).all() or m<=me or min(me,xe,ve)<0:raise ValueError('Published staggered source leaf has unresolved moment bounds')
        eps=4096*np.finfo(float).eps
        if abs(totals[0]-m)>totals[1]+me+eps*max(abs(m),abs(totals[0])):raise ValueError('Staggered overlap masses do not recover the published source leaf')
        for actual,wanted,uncertainty in ((totals[2],m*x,totals[4]+me*np.linalg.norm(x)+(m+me)*xe),(totals[3],m*v,totals[5]+me*np.linalg.norm(v)+(m+me)*ve)):
            if np.linalg.norm(actual-wanted)>uncertainty+eps*max(np.linalg.norm(actual),np.linalg.norm(wanted),np.finfo(float).tiny):raise ValueError('Staggered overlap moments do not recover the published source leaf')
        recovery_diagnostics['source_species_leaves_checked']=recovery_diagnostics.get('source_species_leaves_checked',0)+1
        for name,value in (('maximum_relative_mass_recovery_difference',abs(totals[0]-m)/m),('maximum_position_moment_recovery_difference_per_mass_mpc_h',np.linalg.norm(totals[2]-m*x)/m),('maximum_velocity_moment_recovery_difference_per_mass_km_s',np.linalg.norm(totals[3]-m*v)/m)):
            recovery_diagnostics[name]=max(recovery_diagnostics.get(name,0.),float(value))
    destination_offset=np.asarray(destination_offset,float)
    for owner,row,sp,low,high,image in pieces:
        key=(row,sp)
        if key!=current:
            recover();current=key;totals=[np.longdouble(0),np.longdouble(0),np.zeros(3,np.longdouble),np.zeros(3,np.longdouble),np.longdouble(0),np.longdouble(0)];group_volume=0
        widths=np.asarray([high[q]-low[q] for q in range(3)],float)*box/lattice
        center=(np.asarray(low,float)+np.asarray(high,float))*.5*box/lattice+(destination_offset if sp==1 else 0)
        m,me,x,v,xe,ve=intersection_moments.region(sp,center,widths)
        if not np.isfinite([m,me,xe,ve]).all() or not np.isfinite(x).all() or not np.isfinite(v).all() or m<=me or min(me,xe,ve)<0:raise ValueError('Resolved finite rectangular overlap moments required')
        m=np.longdouble(m);me=np.longdouble(me);x=np.asarray(x,dtype=np.longdouble);v=np.asarray(v,dtype=np.longdouble)
        source_x=x+np.asarray(image,dtype=np.longdouble)*box
        pe=me*np.linalg.norm(x)+(m+me)*xe;source_pe=me*np.linalg.norm(source_x)+(m+me)*xe
        ve_ext=me*np.linalg.norm(v)+(m+me)*ve
        mass[owner,sp]+=m;mass_error[owner,sp]+=me;position[owner,sp]+=m*x;momentum[owner,sp]+=m*v
        position_error[owner,sp]+=pe;momentum_error[owner,sp]+=ve_ext
        volume=math.prod(high[q]-low[q] for q in range(3));group_volume+=volume
        if int(coverage[owner,sp])+volume>int(expected[owner,sp]):raise ValueError('Rectangular overlaps exceed destination volume')
        coverage[owner,sp]+=np.uint64(volume)
        for k,value in enumerate((m,me,m*source_x,m*v,source_pe,ve_ext)):totals[k]+=value
    recover()
    if not np.array_equal(coverage,expected) or not np.array_equal(source_seen,source_expected>0):raise ValueError('Rectangular overlaps omit original source or destination material volume')
    combined=mass.sum(axis=1);error=mass_error.sum(axis=1)
    if np.any(combined<=error):raise ValueError('Staggered restricted mass is unresolved beyond propagated uncertainty')
    x=position.sum(axis=1)/combined[:,None];v=momentum.sum(axis=1)/combined[:,None]
    xe=(position_error.sum(axis=1)+np.linalg.norm(x,axis=1)*error)/(combined-error)
    ve=(momentum_error.sum(axis=1)+np.linalg.norm(v,axis=1)*error)/(combined-error)
    result=dict(destination)
    for name,value in {'mass_msun':combined,'species_mass_msun':mass,'species_mass_absolute_error_msun':mass_error,'unwrapped_position':x,'velocity_peculiar_km_s':v,'centroid_absolute_error_mpc_h':xe,'velocity_absolute_error_km_s':ve}.items():
        result[name]=np.asarray(value,float)
        if not np.isfinite(result[name]).all():raise ArithmeticError('Staggered overlap moment accumulation overflows')
    return result,budget['work']


def _restrict_twofluid_particle_hierarchy(coarse,fine,box,*,maximum_work,intersection_moments=None,recovery_diagnostics=None):
    """Restrict dyadic species volumes and extensive moments exactly.

    An explicit retained-flow intersection integrator enables nonnested
    repartitioning; the legacy route still requires nested refinements.

    Both hierarchies must retain the same physical baryon material offset.
    Species cells are distinct partitions even when a coarse carrier owns
    both; carrier IDs are never used as invented cross-hierarchy identities.
    """
    count=len(coarse['mass_msun']);owners={};descendants={};work=0;refined=False
    if type(maximum_work)is not int or maximum_work<1:raise MemoryError('Hierarchy matching work budget exhausted')
    if not math.isfinite(box) or box<=0:raise ValueError('Finite positive periodic material box required')
    for data in (coarse,fine):
        if not len(data['mass_msun']) or np.asarray(data['cell_level']).dtype.kind not in 'iu' or np.asarray(data['species']).dtype.kind not in 'iu':raise ValueError('Nonempty exact integer hierarchy levels and species required')
    maximum_level=max(int(np.max(coarse['cell_level'])),int(np.max(fine['cell_level'])))
    if not 1<=maximum_level<=20:raise ValueError('Bounded dyadic material hierarchy levels required')
    def key(data,row):
        level=int(data['cell_level'][row]);width=float(data['cell_width_mpc_h'][row]);center=data['cell_center_mpc_h'][row]
        if level<1 or level>maximum_level or not np.isfinite(center).all() or not math.isclose(width,box/2**level,rel_tol=1e-12):raise ValueError('Hierarchy cell has inconsistent dyadic width or center')
        index=np.rint(center/width-.5).astype(np.int64)
        if np.any(center<0) or np.any(center>=box) or np.max(abs(center/width-.5-index))>1e-8:raise ValueError('Hierarchy cells lack their original periodic dyadic alignment')
        return level,tuple(int(x) for x in index)
    def active(data,row):
        kind=int(data['species'][row])
        if kind not in (0,1,2):raise ValueError('Hierarchy carrier has unsupported species ownership')
        expected=(1,) if kind==0 else (0,) if kind==1 else (0,1)
        for species in range(2):
            mass=data['species_mass_msun'][row,species]
            if (species in expected and mass<=0) or (species not in expected and mass!=0):raise ValueError('Hierarchy carrier changes its actual species ownership')
        return expected
    for row in range(count):
        level,index=key(coarse,row)
        for species in active(coarse,row):
            cell=(species,level,*index)
            if cell in owners:raise ValueError('Coarse species hierarchy repeats a leaf')
            owners[cell]=row
    # A leaf cannot coexist with any of its ancestors. This also rejects
    # duplicate coverage independently of the producer's original checks.
    for cell in owners:
        species,level,*index=cell
        for ancestor in range(1,level):
            work+=1
            if work>maximum_work:raise MemoryError('Hierarchy matching work budget exhausted')
            parent=(species,ancestor,*(x//2**(level-ancestor) for x in index))
            if parent in owners:raise ValueError('Coarse material hierarchy overlaps an ancestor leaf')
            if intersection_moments is not None:descendants.setdefault(parent,[]).append(owners[cell])
    for species in range(2):
        if sum(8**(maximum_level-cell[1]) for cell in owners if cell[0]==species)!=8**maximum_level:raise ValueError('Coarse species leaves do not cover the complete periodic material volume')
    mass=np.zeros((count,2),dtype=np.longdouble);mass_error=np.zeros_like(mass)
    position=np.zeros((count,2,3),dtype=np.longdouble);momentum=np.zeros_like(position)
    position_error=np.zeros_like(mass);momentum_error=np.zeros_like(mass)
    coverage=np.zeros((count,2),dtype=np.uint64);seen=set()
    for row in range(len(fine['mass_msun'])):
        level,index=key(fine,row)
        for species in active(fine,row):
            cell=(species,level,*index)
            if cell in seen:raise ValueError('Fine species hierarchy repeats a leaf')
            seen.add(cell);owner=None;owner_level=None
            for ancestor in range(level,0,-1):
                work+=1
                if work>maximum_work:raise MemoryError('Hierarchy matching work budget exhausted')
                owner=owners.get((species,ancestor,*(x//2**(level-ancestor) for x in index)))
                if owner is not None:owner_level=ancestor;break
            if owner is None:
                targets=descendants.get(cell,[]) if intersection_moments is not None else []
                if not targets:raise ValueError('Fine leaf has no coarse species owner; partitions are not nested refinements')
                refined=True
            else:
                targets=[owner];refined|=level>owner_level
            pieces=[]
            for owner in targets:
                work+=1
                if work>maximum_work:raise MemoryError('Hierarchy overlap matching work budget exhausted')
                piece_level=max(level,int(coarse['cell_level'][owner]))
                if level<int(coarse['cell_level'][owner]):
                    # The published fine-side leaf spans several destination
                    # leaves. Only actual retained-field overlap moments may
                    # subdivide it; whole-leaf means cannot supply subleaf data.
                    m,me,x,v,xe,ve=intersection_moments(species,owner,row)
                else:
                    m=fine['species_mass_msun'][row,species];me=fine['species_mass_absolute_error_msun'][row,species]
                    x=fine['cell_center_mpc_h'][row]+fine['species_centroid_relative_mpc_h'][row,species]
                    v=fine['species_velocity_peculiar_km_s'][row,species]
                    xe=fine['species_centroid_absolute_error_mpc_h'][row,species];ve=fine['species_velocity_absolute_error_km_s'][row,species]
                if not np.isfinite([m,me,xe,ve]).all() or not np.isfinite(x).all() or not np.isfinite(v).all() or m<=me or min(me,xe,ve)<0:raise ValueError('Resolved finite overlap mass, centroid and momentum bounds required')
                m=np.longdouble(m);me=np.longdouble(me);x=np.asarray(x,dtype=np.longdouble);v=np.asarray(v,dtype=np.longdouble)
                pe=me*np.linalg.norm(x)+(m+me)*xe;ve_ext=me*np.linalg.norm(v)+(m+me)*ve
                mass[owner,species]+=m;mass_error[owner,species]+=me
                position[owner,species]+=m*x;momentum[owner,species]+=m*v
                position_error[owner,species]+=pe;momentum_error[owner,species]+=ve_ext
                increment=8**(maximum_level-piece_level);expected=8**(maximum_level-int(coarse['cell_level'][owner]))
                if int(coverage[owner,species])+increment>expected:raise ValueError('Fine hierarchy overlaps or exceeds its coarse species volume')
                coverage[owner,species]+=np.uint64(increment)
                pieces.append((m,me,m*x,m*v,pe,ve_ext,increment))
            if owner_level is None:
                # Every split must recover the independently published source
                # leaf, including its original normalization and error bounds.
                if sum(piece[6] for piece in pieces)!=8**(maximum_level-level):raise ValueError('Overlap cells do not exactly cover the source leaf')
                m=np.longdouble(fine['species_mass_msun'][row,species]);me=np.longdouble(fine['species_mass_absolute_error_msun'][row,species])
                x=np.asarray(fine['cell_center_mpc_h'][row]+fine['species_centroid_relative_mpc_h'][row,species],dtype=np.longdouble)
                v=np.asarray(fine['species_velocity_peculiar_km_s'][row,species],dtype=np.longdouble)
                xe=fine['species_centroid_absolute_error_mpc_h'][row,species];ve=fine['species_velocity_absolute_error_km_s'][row,species]
                if not np.isfinite([m,me,xe,ve]).all() or not np.isfinite(x).all() or not np.isfinite(v).all() or m<=me or min(me,xe,ve)<0:raise ValueError('Published source leaf has unresolved or invalid extensive moment bounds')
                totals=[sum((piece[j] for piece in pieces),np.longdouble(0)) for j in range(6)]
                if recovery_diagnostics is not None:
                    recovery_diagnostics['source_species_leaves_checked']=recovery_diagnostics.get('source_species_leaves_checked',0)+1
                    for name,value in (('maximum_relative_mass_recovery_difference',abs(totals[0]-m)/m),('maximum_position_moment_recovery_difference_per_mass_mpc_h',np.linalg.norm(totals[2]-m*x)/m),('maximum_velocity_moment_recovery_difference_per_mass_km_s',np.linalg.norm(totals[3]-m*v)/m)):
                        recovery_diagnostics[name]=max(recovery_diagnostics.get(name,0.),float(value))
                eps=4096*np.finfo(float).eps
                if abs(totals[0]-m)>totals[1]+me+eps*max(abs(m),abs(totals[0])):raise ValueError('Overlap masses do not recover the published source leaf')
                for actual,expected,uncertainty in ((totals[2],m*x,totals[4]+me*np.linalg.norm(x)+(m+me)*xe),(totals[3],m*v,totals[5]+me*np.linalg.norm(v)+(m+me)*ve)):
                    if np.linalg.norm(actual-expected)>uncertainty+eps*max(np.linalg.norm(actual),np.linalg.norm(expected),np.finfo(float).tiny):raise ValueError('Overlap extensive moments do not recover the published source leaf')
    for cell in seen:
        species,level,*index=cell
        for ancestor in range(1,level):
            work+=1
            if work>maximum_work:raise MemoryError('Hierarchy matching work budget exhausted')
            if (species,ancestor,*(x//2**(level-ancestor) for x in index)) in seen:raise ValueError('Fine material hierarchy overlaps an ancestor leaf')
    for species in range(2):
        if sum(8**(maximum_level-cell[1]) for cell in seen if cell[0]==species)!=8**maximum_level:raise ValueError('Source species leaves do not cover the complete periodic material volume')
    for row in range(count):
        for species in active(coarse,row):
            if int(coverage[row,species])!=8**(maximum_level-int(coarse['cell_level'][row])):raise ValueError('Fine species leaves do not exactly cover the original coarse material volume')
    if not refined:raise ValueError('Hierarchy refinement has no actually subdivided material leaf')
    combined_mass=mass.sum(axis=1);combined_error=mass_error.sum(axis=1)
    if np.any(combined_mass<=combined_error):raise ValueError('Restricted native masses are unresolved beyond propagated uncertainty')
    unwrapped=position.sum(axis=1)/combined_mass[:,None];velocity=momentum.sum(axis=1)/combined_mass[:,None]
    xe=(position_error.sum(axis=1)+np.linalg.norm(unwrapped,axis=1)*combined_error)/(combined_mass-combined_error)
    ve=(momentum_error.sum(axis=1)+np.linalg.norm(velocity,axis=1)*combined_error)/(combined_mass-combined_error)
    restricted=dict(coarse)
    for name,value in {'mass_msun':combined_mass,'species_mass_msun':mass,'species_mass_absolute_error_msun':mass_error,
            'unwrapped_position':unwrapped,'velocity_peculiar_km_s':velocity,'centroid_absolute_error_mpc_h':xe,'velocity_absolute_error_km_s':ve}.items():
        restricted[name]=np.asarray(value,float)
        if not np.isfinite(restricted[name]).all():raise ArithmeticError('Restricted material extensive moments overflow')
    return restricted,work


def validate_twofluid_projection_refinement_request(parameters):
    request=parameters['request']
    keys={'schema','levels','refinement_axis','maximum_levels','maximum_particles','chunk_rows','memory_gib',
        'mass_floor_msun','velocity_floor_km_s','maximum_mass_relative_error','maximum_species_mass_relative_error',
        'maximum_position_error_mpc_h','maximum_velocity_error_km_s','maximum_momentum_relative_error','maximum_adjacent_error_ratio'}
    if isinstance(request,dict) and request.get('schema')in ('twofluid-projection-refinement-request-v2','twofluid-projection-refinement-request-v3','twofluid-projection-refinement-request-v4'):keys.add('maximum_cell_matching_work')
    if isinstance(request,dict) and request.get('schema')in ('twofluid-projection-refinement-request-v3','twofluid-projection-refinement-request-v4'):keys.add('maximum_overlap_quadrature_evaluations')
    if not isinstance(request,dict) or set(request)!=keys or request.get('schema')not in ('twofluid-projection-refinement-request-v1','twofluid-projection-refinement-request-v2','twofluid-projection-refinement-request-v3','twofluid-projection-refinement-request-v4'):
        raise ValueError('Complete finite-volume native projection refinement request required')
    allowed=('hierarchy',) if request['schema']in ('twofluid-projection-refinement-request-v3','twofluid-projection-refinement-request-v4') else ('grid','timestep','hierarchy') if request['schema']=='twofluid-projection-refinement-request-v2' else ('grid','timestep')
    if 'maximum_overlap_quadrature_evaluations' in request and (type(request['maximum_overlap_quadrature_evaluations'])is not int or request['maximum_overlap_quadrature_evaluations']<1):raise ValueError('Exact positive overlap quadrature bound required')
    if request['refinement_axis']not in allowed:raise ValueError('Explicit source-grid, timestep or supported hierarchy refinement required')
    if 'maximum_cell_matching_work' in request and (type(request['maximum_cell_matching_work'])is not int or request['maximum_cell_matching_work']<1):raise ValueError('Exact positive hierarchy matching work bound required')
    for key,minimum in (('maximum_levels',3),('maximum_particles',1),('chunk_rows',1)):
        if type(request[key])is not int or request[key]<minimum:raise ValueError('Exact positive projection comparison work bound required: '+key)
    if request['chunk_rows']>request['maximum_particles']:raise ValueError('Projection comparison chunk exceeds census bound')
    if not isinstance(request['levels'],list) or not 3<=len(request['levels'])<=request['maximum_levels']:raise ValueError('At least three actual native projection products required')
    for reference in request['levels']:_validate_twofluid_product_reference(reference,'projection level')
    for key in keys-{'schema','levels','refinement_axis','maximum_levels','maximum_particles','chunk_rows','maximum_cell_matching_work','maximum_overlap_quadrature_evaluations'}:
        value=request[key]
        if type(value)not in (int,float) or not math.isfinite(value) or value<0 or key in ('memory_gib','mass_floor_msun','velocity_floor_km_s','maximum_adjacent_error_ratio') and value<=0:
            raise ValueError('Finite dimensional projection refinement limit required: '+key)
    if request['maximum_adjacent_error_ratio']>1:raise ValueError('Projection refinement growth ratio must be at most one')
    if request['maximum_particles']*4096+request['chunk_rows']*4096>request['memory_gib']*1024**3/2:
        raise MemoryError('Adjacent projection products, sorting and comparison workspace exceed allocation')
    return request


def twofluid_projection_refinement_input_bindings(parameters):
    from .sidm_analysis import _load_product
    request=validate_twofluid_projection_refinement_request(parameters);sources={}
    for reference in request['levels']:
        report,arrays,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/4)
        sources.update(binding);del arrays
        sources.update(_load_product(report['relative_fields'],parameters['request_base'],request['memory_gib']*1024**3/4)[2])
    return sources


def publish_twofluid_projection_refinement(parameters,destination):
    """Compare the same original native leaf cells across refined flow products."""
    import copy
    from pathlib import Path
    from .data import exact_uint64_ids,export_numeric_audit
    from .sidm_analysis import _load_product
    from .foundation import canonical_digest,file_sha256
    request=validate_twofluid_projection_refinement_request(parameters);sources={};seen=set();previous=None;signature=None
    maxima=[];means=[];upper_maxima=[];momentum_changes=[];grids=[];step_records=[];first_seed=None;matching_work=0;hierarchy_flow=None;lineage=[]
    hierarchy=request['refinement_axis']=='hierarchy'
    staggered=request['schema']=='twofluid-projection-refinement-request-v4'
    overlap=request['schema']in ('twofluid-projection-refinement-request-v3','twofluid-projection-refinement-request-v4')
    overlap_work={'maximum_evaluations':request.get('maximum_overlap_quadrature_evaluations',1),'evaluations':0};overlap_recovery={}
    numerical={'maximum_grid_cells','memory_gib','maximum_steps','maximum_rhs_evaluations','maximum_step_rejections','material_chunk_rows',
        'material_map_maximum_subboxes','material_map_maximum_depth'}
    units={'mass_msun':'Msun','species_mass_msun':'Msun','species_mass_absolute_error_msun':'Msun',
        'cell_center_mpc_h':'comoving Mpc/h','cell_width_mpc_h':'comoving Mpc/h','lagrangian_position_mpc_h':'comoving Mpc/h',
        'position_comoving_mpc_h':'comoving Mpc/h','species_centroid_relative_mpc_h':'comoving Mpc/h',
        'velocity_peculiar_km_s':'km / s','velocity_absolute_error_km_s':'km / s','centroid_absolute_error_mpc_h':'comoving Mpc/h'}
    if hierarchy:units.update(species_velocity_peculiar_km_s='km / s',species_velocity_absolute_error_km_s='km / s',species_centroid_absolute_error_mpc_h='comoving Mpc/h')
    for reference in request['levels']:
        if reference['sha256'] in seen:raise ValueError('Projection refinement repeats an identical product')
        seen.add(reference['sha256']);report,arrays,binding=_load_product(reference,parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
        if (report.get('schema')!='twofluid-native-particle-projection-v1' or report.get('flow_kind')!='full_nonlinear_material'
                or report.get('projection_mode')!='finite_volume_material_cells' or report.get('mass_component_order')!=['cdm','baryons']):
            raise ValueError('Actual full nonlinear finite-volume material projection products required')
        contracts=report['numeric_data']['array_contract']['datasets']
        if contracts['particle_ids'].get('identity_namespace')!=report['particle_namespace']:raise ValueError('Projection ID namespace contract differs')
        for name,unit in units.items():
            if contracts[name]['units']!=unit:raise ValueError('Projection comparison physical units differ: '+name)
        ids=exact_uint64_ids(arrays['particle_ids']);count=len(ids)
        lineage.append({'projection':reference,'native_particle_load':report['request']['native_particle_load'],
            'particle_namespace':report['particle_namespace'],'carrier_count':count,'material_geometry':report['material_geometry']})
        if count<1 or count>request['maximum_particles']:raise MemoryError('Projection comparison census exceeds allocation')
        order=np.argsort(ids);ids=ids[order]
        data={name:np.asarray(arrays[name])[order] for name in (*units,'species','cell_level')}
        shapes={'species':(count,),'cell_level':(count,),'mass_msun':(count,),'species_mass_msun':(count,2),
            'species_mass_absolute_error_msun':(count,2),'cell_center_mpc_h':(count,3),'cell_width_mpc_h':(count,),
            'lagrangian_position_mpc_h':(count,3),'position_comoving_mpc_h':(count,3),'species_centroid_relative_mpc_h':(count,2,3),
            'velocity_peculiar_km_s':(count,3),'velocity_absolute_error_km_s':(count,),'centroid_absolute_error_mpc_h':(count,)}
        if hierarchy:shapes.update(species_velocity_peculiar_km_s=(count,2,3),species_velocity_absolute_error_km_s=(count,2),species_centroid_absolute_error_mpc_h=(count,2))
        for name,shape in shapes.items():
            if data[name].shape!=shape or not np.isfinite(data[name]).all():raise ValueError('Complete finite native projection rows required: '+name)
        if np.any(data['mass_msun']<=0) or np.any(data['species_mass_msun']<0) or any(np.any(data[name]<0) for name in ('species_mass_absolute_error_msun','velocity_absolute_error_km_s','centroid_absolute_error_mpc_h')):
            raise ValueError('Positive native masses and nonnegative propagated uncertainties required')
        if np.any(~np.isin(data['species'],(0,1,2))) or np.any(data['cell_width_mpc_h']<=0):raise ValueError('Original gas/CDM/coarse leaf geometry required')
        if data['cell_level'].dtype.kind not in 'iu':raise ValueError('Exact integer native material hierarchy levels required')
        if hierarchy and (np.any(data['species_velocity_absolute_error_km_s']<0) or np.any(data['species_centroid_absolute_error_mpc_h']<0)):
            raise ValueError('Nonnegative species moment uncertainty required; regenerate older projection products')
        tolerance=2048*np.finfo(float).eps*data['mass_msun']
        if np.any(abs(data['species_mass_msun'].sum(axis=1)-data['mass_msun'])>tolerance):raise ValueError('Species material masses do not sum to native carrier mass')
        parent,fields,binding=_load_product(report['relative_fields'],parameters['request_base'],request['memory_gib']*1024**3/4);sources.update(binding)
        if parent.get('schema')!='finite-epoch-nonlinear-twofluid-reference-v1' or parent['epoch']!=report['epoch']:raise ValueError('Projection lacks actual matching nonlinear flow lineage')
        _twofluid_material_fields(parent,fields);grid=fields['lagrangian_mass_weights'].shape[-1];del arrays
        if not overlap:del fields
        if report['material_geometry']['source_grid_cells_per_axis']!=grid:raise ValueError('Projection source grid differs from actual material fields')
        physics=copy.deepcopy(parent['request'])
        for name in numerical:physics.pop(name,None)
        if request['refinement_axis']=='timestep':physics.pop('maximum_step_ln_a')
        projection=copy.deepcopy(report['request']);projection.pop('relative_fields')
        for name in ('memory_gib','maximum_particles','chunk_rows','maximum_quadrature_evaluations'):projection.pop(name,None)
        geometry=copy.deepcopy(report['material_geometry']);geometry.pop('source_grid_cells_per_axis')
        if hierarchy:
            # Different loads and ID namespaces are retained as input lineage,
            # while actual nested species geometry defines the comparison.
            projection.pop('native_particle_load');projection.pop('particle_namespace')
            geometry={'policy':'original per-level species staggering; exact periodic material overlap'} if staggered else {'baryon_material_offset_mpc_h':geometry['baryon_material_offset_mpc_h']}
            flow=report['relative_fields']['sha256']
            if hierarchy_flow is not None and hierarchy_flow!=flow:raise ValueError('Hierarchy refinement must project the exact same sealed material flow')
            hierarchy_flow=flow
        definition={'physics':physics,'seed':parent['native_seed_definition'],'method':parent['method'],'epoch':parent['epoch'],
            'projection':projection,'geometry':geometry,'particle_namespace':'distinct retained namespaces; dyadic species cell matching' if hierarchy else report['particle_namespace']}
        digest=canonical_digest(definition)
        if signature is not None and signature!=digest:raise ValueError('Projection refinement changes native load, seed, operator, physics or epoch')
        signature=digest;box=float(parent['request']['box_mpc_h'])
        data['unwrapped_position']=data['cell_center_mpc_h']+np.sum(data['species_mass_msun'][:,:,None]*data['species_centroid_relative_mpc_h'],axis=1)/data['mass_msun'][:,None]
        wrapped_error=(np.mod(data['unwrapped_position'],box)-data['position_comoving_mpc_h']+box/2)%box-box/2
        if np.max(abs(wrapped_error))>4096*np.finfo(float).eps*max(1,box,float(np.max(abs(data['unwrapped_position'])))):
            raise ValueError('Published wrapped positions differ from actual unwrapped material COMs')
        step=[parent['request']['maximum_step_ln_a'],parent.get('maximum_realized_step_ln_a'),parent['steps']]
        if request['refinement_axis']=='grid':
            if grids and (grid<=grids[-1]):raise ValueError('Projection source grids must strictly increase')
        elif request['refinement_axis']=='timestep':
            if type(step[1])not in (int,float) or not math.isfinite(step[1]) or step[1]<=0 or type(step[2])is not int or step[2]<1:raise ValueError('Actual operative timestep diagnostics required')
            if first_seed is not None and first_seed!=parent['initial_fields_reference']:raise ValueError('Temporal projections change the original exact seed')
            first_seed=parent['initial_fields_reference']
            if grids and (grid!=grids[-1] or step[0]>=step_records[-1][0] or step[1]>=step_records[-1][1] or step[2]<=step_records[-1][2]):raise ValueError('Temporal projections do not refine actual fixed-grid updates')
        actual_data=data
        if previous is not None:
            old_ids,old=previous
            if hierarchy:
                callback=_twofluid_overlap_moment_integrator(old,report,parent,fields,box,overlap_work) if overlap else None
                if staggered:
                    data,used=_restrict_twofluid_staggered_hierarchy(old,actual_data,box,lineage[-2]['material_geometry']['baryon_material_offset_mpc_h'],report['material_geometry']['baryon_material_offset_mpc_h'],maximum_work=request['maximum_cell_matching_work']-matching_work,intersection_moments=callback,recovery_diagnostics=overlap_recovery)
                else:
                    data,used=_restrict_twofluid_particle_hierarchy(old,actual_data,box,maximum_work=request['maximum_cell_matching_work']-matching_work,intersection_moments=callback,recovery_diagnostics=overlap_recovery if overlap else None)
                matching_work+=used;comparison_count=len(old_ids)
            else:
                if not np.array_equal(old_ids,ids):raise ValueError('Projection refinement changes the exact carrier ID census')
                for name in ('species','cell_level','cell_center_mpc_h','cell_width_mpc_h','lagrangian_position_mpc_h'):
                    if not np.array_equal(old[name],data[name]):raise ValueError('Projection refinement changes original material cell geometry: '+name)
                comparison_count=count
            peak=np.zeros(5);upper=np.zeros(5);summed=np.zeros(5);signed=np.zeros(3)
            normalization=float(np.sum(old['mass_msun'],dtype=np.longdouble)) if hierarchy else float(comparison_count)
            if not math.isfinite(normalization) or normalization<=0:raise ArithmeticError('Projection error averaging normalization overflows')
            for begin in range(0,comparison_count,request['chunk_rows']):
                sl=slice(begin,min(comparison_count,begin+request['chunk_rows']));cm=old['mass_msun'][sl];fm=data['mass_msun'][sl]
                cv=old['velocity_peculiar_km_s'][sl];fv=data['velocity_peculiar_km_s'][sl]
                ce=old['species_mass_absolute_error_msun'][sl];fe=data['species_mass_absolute_error_msun'][sl]
                massden=np.maximum(fm,request['mass_floor_msun']);speciesden=np.maximum(data['species_mass_msun'][sl],request['mass_floor_msun'])
                pden=fm*np.maximum(np.linalg.norm(fv,axis=1),request['velocity_floor_km_s'])
                dp=cm[:,None]*cv-fm[:,None]*fv
                error=np.column_stack((abs(cm-fm)/massden,np.max(abs(old['species_mass_msun'][sl]-data['species_mass_msun'][sl])/speciesden,axis=1),
                    np.linalg.norm(old['unwrapped_position'][sl]-data['unwrapped_position'][sl],axis=1),np.linalg.norm(cv-fv,axis=1),np.linalg.norm(dp,axis=1)/pden))
                cve=old['velocity_absolute_error_km_s'][sl];fve=data['velocity_absolute_error_km_s'][sl]
                perror=ce.sum(axis=1)*(np.linalg.norm(cv,axis=1)+cve)+cm*cve+fe.sum(axis=1)*(np.linalg.norm(fv,axis=1)+fve)+fm*fve
                uncertainty=np.column_stack(((ce+fe).sum(axis=1)/massden,np.max((ce+fe)/speciesden,axis=1),
                    old['centroid_absolute_error_mpc_h'][sl]+data['centroid_absolute_error_mpc_h'][sl],cve+fve,perror/pden))
                if not np.isfinite(error).all() or not np.isfinite(uncertainty).all():raise ArithmeticError('Projection comparison error propagation overflows')
                peak=np.maximum(peak,error.max(axis=0));upper=np.maximum(upper,(error+uncertainty).max(axis=0))
                weights=cm/normalization if hierarchy else np.full(len(cm),1/normalization)
                summed+=np.sum(error*weights[:,None],axis=0);signed+=dp.sum(axis=0)
            maxima.append(peak);upper_maxima.append(upper);means.append(summed);momentum_changes.append(signed)
        previous=(ids,actual_data);grids.append(grid);step_records.append(step)
    values=np.asarray(maxima);average=np.asarray(means);upper=np.asarray(upper_maxima)
    limits=np.asarray([request[name] for name in ('maximum_mass_relative_error','maximum_species_mass_relative_error','maximum_position_error_mpc_h','maximum_velocity_error_km_s','maximum_momentum_relative_error')])
    within=upper<=limits;decreasing=values[1:]<=request['maximum_adjacent_error_ratio']*values[:-1];mean_decreasing=average[1:]<=request['maximum_adjacent_error_ratio']*average[:-1]
    output={'adjacent_maximum_errors':values,'adjacent_mean_errors':average,'adjacent_uncertainty_upper_errors':upper,
        'within_declared_limits':within,'maximum_errors_decreasing':decreasing,'mean_errors_decreasing':mean_decreasing,
        'adjacent_total_momentum_difference_msun_km_s':np.asarray(momentum_changes),'source_grid_sizes':np.asarray(grids,np.int64)}
    if request['refinement_axis']=='timestep':output.update(configured_and_realized_step_ln_a=np.asarray([row[:2] for row in step_records]),accepted_step_counts=np.asarray([row[2] for row in step_records],np.int64))
    if any(not np.isfinite(value).all() for value in output.values()):raise ArithmeticError('Projection summary overflows')
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    report={'schema':'measured-twofluid-projection-refinement-v1','execution_status':'COMPLETE','request':request,'fixed_definition':definition,'fixed_signature':signature,
        'error_coordinates':['carrier_mass_relative','species_mass_relative_max','unwrapped_COM_position','carrier_velocity','carrier_momentum_relative'],
        'error_units':['dimensionless','dimensionless','comoving Mpc/h','km / s','dimensionless'],
        'status':'DECLARED_NATIVE_PROJECTION_REFINEMENT_BUDGETS_SUPPORTED' if within.all() and decreasing.all() and mean_decreasing.all() else 'UNRESOLVED_NATIVE_PROJECTION_REFINEMENT',
        'comparison':'all exact-ID original native leaf cells; unwrapped material COM, actual masses and velocities; maximum and unweighted carrier mean errors; propagated interpolation uncertainty adds to maximum differences',
        'scope':'fixed original native material partition, refined nonlinear source grid or operative timestep; no changed native hierarchy, continuum extrapolation, native evolution or global accuracy acceptance',
        'source_hashes':sources,'science_acceptance':'NOT_ESTABLISHED'}
    if hierarchy:
        report.update(schema='measured-twofluid-projection-refinement-v2',cell_matching_work=matching_work,hierarchy_lineage=lineage,
            hierarchy_flow_sha256=hierarchy_flow,
            mean_error_weighting='actual original coarse carrier mass; changing leaf counts do not set the statistical weight',
            comparison='nested dyadic species cells at the same physical staggering; exact-volume fine-to-coarse restriction of mass, momentum and unwrapped position moments; distinct original carrier identities retained as input lineage',
            scope='same exact nonlinear material flow, changed nested original native leaf partitions with unchanged species offset; no simultaneous flow refinement, nonnested translated partitions, continuum extrapolation or native evolution acceptance')
    if overlap:
        report.update(schema='measured-twofluid-projection-overlap-comparison-v1',overlap_quadrature_evaluations=overlap_work['evaluations'],retained_flow_overlap_regions=overlap_work.get('regions',0),source_leaf_recovery=overlap_recovery,
            mean_error_weighting='actual original destination carrier mass; no carrier-count weighting',
            comparison='per-species exact dyadic intersection coverage; original whole-leaf moments where contained, retained-flow reintegration where a source leaf spans destination leaves; source-leaf mass/position/momentum recovery checked',
            scope='same exact sealed material flow and unchanged species staggering; nonnested dyadic repartitioning with mixed local refinement/coarsening. Propagated interpolation uncertainty retained; no changed staggering, nondyadic cells, simultaneous flow refinement or native/scientific acceptance',
            status='DECLARED_NATIVE_OVERLAP_COMPARISON_BUDGETS_SUPPORTED' if within.all() and decreasing.all() and mean_decreasing.all() else 'UNRESOLVED_NATIVE_OVERLAP_COMPARISON')
    if staggered:
        offsets=np.asarray([row['material_geometry']['baryon_material_offset_mpc_h'] for row in lineage],float)
        changed=np.any(offsets[1:]!=offsets[:-1],axis=1)
        output.update(native_baryon_offsets_mpc_h=offsets,staggering_changed_between_levels=changed)
        report.update(schema='measured-twofluid-staggered-overlap-comparison-v1',rectangular_overlap_regions=overlap_work.get('rectangular_regions',0),
            comparison='exact integer-lattice periodic rectangular species intersections with retained-flow reintegration; source recovery in original source unwrap frame and comparison in original destination unwrap frame',
            scope='same sealed material flow with original per-level dyadic native leaf layouts and recorded changed species staggering; no inferred offsets, nondyadic leaves, simultaneous flow refinement or native/scientific acceptance',
            status=('DECLARED_STAGGERED_OVERLAP_BUDGETS_SUPPORTED' if within.all() and decreasing.all() and mean_decreasing.all() else 'UNRESOLVED_STAGGERED_OVERLAP_COMPARISON') if changed.any() else 'STAGGERING_CONTROL_NOT_EXERCISED')
    return export_numeric_audit(destination,report,output,{name:{'units':'comoving Mpc/h' if name=='native_baryon_offsets_mpc_h' else 'Msun * (km / s)' if name.endswith('msun_km_s') else 'coordinate-dependent' if name in ('adjacent_maximum_errors','adjacent_mean_errors','adjacent_uncertainty_upper_errors') else 'dimensionless','quantity':'native material projection refinement '+name} for name in output},algorithm=source,source_hashes=sources,context={'epoch':definition['epoch']})


def twofluid_projection_refinement_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Projection refinement exceeds allocation')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Projection refinement escapes owned generation')
    return publish_twofluid_projection_refinement(parameters,destination)


def twofluid_projection_refinement_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Compare finite-volume native particle projections with exact-ID, nested hierarchy or retained-flow overlap semantics')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text());parameters.setdefault('request_base',str(path.parent))
    publish_twofluid_projection_refinement(parameters,args.output);return 0


def _validate_twofluid_product_reference(reference,name):
    """Accept exact retained assets or same-artifact unresolved workflow seals."""
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('Exact path/SHA256 reference required: '+name)
    path,checksum=reference['path'],reference['sha256']
    if isinstance(path,dict):
        if set(path)!={'stage','artifact'} or not isinstance(checksum,dict) or set(checksum)!={'stage','artifact','load'} or checksum['load']!='sha256' or any(checksum[key]!=path[key] for key in ('stage','artifact')):
            raise ValueError('Unresolved product path/checksum must select the same stage artifact: '+name)
        if any(not isinstance(path[key],str) or not path[key] for key in ('stage','artifact')):raise ValueError('Workflow product reference needs nonempty stage/artifact strings')
    elif not isinstance(path,str) or not path or not isinstance(checksum,str) or len(checksum)!=64 or any(value not in '0123456789abcdef' for value in checksum):
        raise ValueError('Retained product needs a nonempty path and actual lowercase SHA256: '+name)


def validate_twofluid_native_ic_request(parameters):
    request=parameters['request']
    if request.get('schema')!='twofluid-native-ic-request-v1':raise ValueError('Typed finite-epoch native IC request required')
    for name in ('particle_projection','base_native_generation'):
        _validate_twofluid_product_reference(request[name],name)
    if request.get('gas_state_policy')!='preserve_original_native_thermochemical_state':
        raise ValueError('Native flow replacement requires explicit preservation of the supplied gas thermochemical state')
    for name in ('memory_gib','disk_gib'):
        value=request[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError('Positive native writer resource bound required: '+name)
    if type(request['chunk_rows'])is not int or request['chunk_rows']<1:raise ValueError('Exact positive native writer chunk bound required')
    if 'window_comparison' in request:
        import copy
        template=request['window_comparison']
        if not isinstance(template,dict) or 'native_generation' in template:raise ValueError('Native writer window comparison must be an immutable request template without a replaceable generation')
        check=copy.deepcopy(template);check['native_generation']={'path':'bound-at-native-launch','sha256':'0'*64}
        validate_twofluid_window_request({'request':check})
    units=request.get('retained_dataset_units',{})
    if not isinstance(units,dict) or any(not isinstance(key,str) or not isinstance(value,str) or not value.strip() for key,value in units.items()):raise ValueError('Retained native fields need explicit dataset-path unit declarations')
    return request


def twofluid_native_ic_input_bindings(parameters):
    from pathlib import Path
    from .sidm_analysis import _load_product
    from .foundation import file_sha256
    from .zoom import verify_generated_zoom
    request=validate_twofluid_native_ic_request(parameters)
    report,_,sources=_load_product(request['particle_projection'],parameters['request_base'],request['memory_gib']*1024**3/2)
    if report.get('schema')!='twofluid-native-particle-projection-v1':raise ValueError('Actual sealed native-q projection required')
    if report.get('projection_mode')=='finite_volume_material_cells' and 'window_comparison' not in request:raise ValueError('Finite-volume native writer requires its frozen window_comparison launch template')
    # Its retained q-load is the only admissible original, rather than another
    # apparently equivalent IC with different IDs or gas initialization.
    original=report['request']['native_particle_load']
    matches=[path for path,digest in report['source_hashes'].items() if digest==original['sha256'] and path in sources]
    if len(matches)!=1 or file_sha256(matches[0])!=original['sha256']:raise ValueError('Original native IC source binding is ambiguous or changed')
    base=request['base_native_generation'];base_path=(Path(parameters['request_base'])/base['path']).resolve()
    if file_sha256(base_path)!=base['sha256']:raise ValueError('Original native generation changed')
    generation=verify_generated_zoom(base_path)
    if file_sha256(base_path.parent/'ics_gadget.hdf5')!=original['sha256']:raise ValueError('Particle q-load is not the declared original native generation')
    sources[str(base_path)]=base['sha256']
    # Retain every generation artifact: a matching q-load does not make a
    # different noise realization or native transfer recipe interchangeable.
    sources.update({str((base_path.parent/name).resolve()):digest for name,digest in generation['artifacts'].items()})
    return sources


def publish_twofluid_native_ic(parameters,destination):
    """Write an immutable native IC and provenance; no acceptance substitution."""
    import shutil
    from pathlib import Path
    import h5py
    from .sidm_analysis import _load_product
    from .data import exact_uint64_ids,external_hdf5_contract,snapshot_total_counts,coordinate_contract,epoch_contract
    from .foundation import atomic_json,atomic_text,file_sha256
    request=validate_twofluid_native_ic_request(parameters);memory=request['memory_gib']*1024**3
    sources=twofluid_native_ic_input_bindings(parameters)
    projection,arrays,_=_load_product(request['particle_projection'],parameters['request_base'],memory/2)
    original_digest=projection['request']['native_particle_load']['sha256']
    original=Path(next(path for path,digest in projection['source_hashes'].items() if digest==original_digest and path in sources))
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Keep previous native IC generation: '+str(destination))
    total=len(arrays['particle_ids']);chunk=request['chunk_rows']
    if total*512+chunk*512>memory/2:raise MemoryError('Native IC writer exceeds its bounded particle workspace')
    disk_bound=original.stat().st_size+total*256+1024**2
    if disk_bound>request['disk_gib']*1024**3:raise MemoryError('Native IC writer exceeds its declared disk allocation')
    ancestor=destination.parent
    while not ancestor.exists():ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free<disk_bound:raise OSError('Insufficient free space for immutable native IC')
    destination.mkdir(parents=True,exist_ok=False);output=destination/'ics_gadget.hdf5';manifest=destination/'twofluid-generation.json'
    report={'schema':'twofluid-native-ic-generation-v1','execution_status':'IN_PROGRESS','science_acceptance':'NOT_ACCEPTED',
        'request':request,'source_hashes':sources,'particle_namespace':projection['particle_namespace'],'epoch':projection['epoch'],
        'flow_kind':projection.get('flow_kind','second_order_perturbative'),
        'projection_mode':projection.get('projection_mode','uniform_native_q_points'),'material_geometry':projection.get('material_geometry'),
        'particle_projection':{'path':str((Path(parameters['request_base'])/request['particle_projection']['path']).resolve()),'sha256':request['particle_projection']['sha256']},
        'base_native_generation':{'path':str((Path(parameters['request_base'])/request['base_native_generation']['path']).resolve()),'sha256':request['base_native_generation']['sha256']}}
    atomic_json(manifest,report)
    try:
        shutil.copyfile(original,output)
        specifications={}
        with h5py.File(output,'r+') as native:
            header=native['Header'].attrs;counts=snapshot_total_counts(header);a=float(header['Time']);h=float(header['HubbleParam'])
            if total!=int(counts.sum()) or any(counts[k] for k in (3,4,5)) or a<=0 or h<=0:raise ValueError('Native writer requires the actual complete gas/CDM/coarse species census')
            if counts[2] and projection.get('projection_mode')!='finite_volume_material_cells':raise ValueError('Coarse native ICs require the integrated material-cell projection')
            if 'ThermalVelocities' in native['PartType1']:raise ValueError('A cold two-fluid flow cannot replace an independently sampled thermal relic velocity')
            for kind in (0,1,2):
                if not counts[kind]:continue
                group=native['PartType'+str(kind)];use=np.flatnonzero(arrays['species']==kind)
                ids=exact_uint64_ids(group['ParticleIDs'][:])
                if len(use)!=len(ids) or not np.array_equal(ids,arrays['particle_ids'][use]):raise ValueError('Projection changed actual native particle row identities')
                values={'Coordinates':arrays['position_comoving_mpc_h'][use]*1000,
                    'Velocities':arrays['velocity_peculiar_km_s'][use]/math.sqrt(a),
                    'Masses':arrays['mass_msun'][use]*h/1e10}
                if projection.get('flow_kind','second_order_perturbative')=='full_nonlinear_material':
                    for obsolete in ('FirstOrderDisplacements','FirstOrderVelocities'):
                        if obsolete in group:del group[obsolete]
                    values['MaterialDisplacements']=arrays['material_displacement_mpc_h'][use]*1000
                else:
                    values.update(FirstOrderDisplacements=arrays['displacement_first_order_mpc_h'][use]*1000,
                        FirstOrderVelocities=arrays['velocity_first_order_peculiar_km_s'][use]/math.sqrt(a))
                if any(not np.isfinite(value).all() for value in values.values()) or np.any(values['Masses']<=0):raise ValueError('Native projected flow has invalid finite values or masses')
                for first in range(0,len(use),chunk):
                    stop=min(first+chunk,len(use))
                    q=group['LagrangianCoordinates'][first:stop]/1000
                    if not np.array_equal(q,arrays['lagrangian_position_mpc_h'][use[first:stop]]):raise ValueError('Projection changed the original native Lagrangian positions')
                for name,value in values.items():
                    if name in group:del group[name]
                    dataset=group.create_dataset(name,shape=value.shape,dtype='f8',chunks=True,fletcher32=True)
                    for first in range(0,len(use),chunk):dataset[first:first+chunk]=value[first:first+chunk]
            mass_table=np.asarray(header['MassTable'],float).copy();mass_table[:3]=0;header['MassTable']=mass_table
            # Old mixed-fluid growth coefficients and removed-momentum stamps
            # describe the original exporter, not the replacement flow.
            original_music={name:np.asarray(header[name]).tolist() for name in header if name.startswith('Music_')}
            keep={'Music_lagrangian_schema','Music_lagrangian_unitlength_cm','Music_levelmin','Music_levelmax','Music_shift_back','Music_output_coordinate_shift_box_fraction',
                'Music_material_cell_schema','Music_material_cell_unitlength_cm','Music_baryon_material_offset_native'}
            for name in list(header):
                if name.startswith('Music_') and name not in keep:del header[name]
            header['Research_twofluid_flow_schema']=1
            flow_kind=projection.get('flow_kind','second_order_perturbative')
            header['Research_twofluid_flow_kind']=flow_kind
            if flow_kind=='second_order_perturbative':header['Research_twofluid_perturbative_order']=2
            elif 'Research_twofluid_perturbative_order' in header:del header['Research_twofluid_perturbative_order']
            header['Research_twofluid_projection_sha256']=request['particle_projection']['sha256']
            header['Research_twofluid_projection_mode']=projection.get('projection_mode','uniform_native_q_points')
            header['Research_fine_dm_mean_mass_native']=float(arrays['mass_msun'][arrays['species']==1].mean()*h/1e10)
            header['Research_twofluid_bulk_mode_policy']='retain finite-epoch supplied flow'
            header['Flag_DoublePrecision']=1
            units={'Coordinates':'comoving kpc/h','LagrangianCoordinates':'comoving kpc/h',
                'LagrangianCellCenters':'comoving kpc/h','LagrangianCellWidths':'comoving kpc/h','Velocities':'peculiar km/s divided by sqrt(a)',
                'Masses':'1e10 Msun/h','ParticleIDs':'identity','MaterialDisplacements':'comoving kpc/h',
                'FirstOrderDisplacements':'comoving kpc/h','FirstOrderVelocities':'peculiar km/s divided by sqrt(a)'}
            def describe(name,item):
                if isinstance(item,h5py.Dataset):
                    leaf=name.rsplit('/',1)[-1]
                    if leaf not in units and name not in request.get('retained_dataset_units',{}):raise ValueError('Declare the retained native dataset units: '+name)
                    specifications[name]={'units':units.get(leaf,request.get('retained_dataset_units',{}).get(name)),
                        'quantity':('finite-epoch native two-fluid ' if leaf in units else 'unchanged original native state ')+leaf}
                    if leaf=='ParticleIDs':specifications[name]['identity_namespace']=projection['particle_namespace']
            native.visititems(describe)
        sources[str(Path(__file__).resolve())]=file_sha256(__file__)
        contract=external_hdf5_contract(output,specifications,epoch=epoch_contract(a),
            coordinates=coordinate_contract(position_units='comoving kpc/h',velocity_convention='peculiar km/s divided by sqrt(a)',origin='original native periodic box',periodic_wrapping='original periodic box'),
            provenance={'input_hashes':sources,'algorithm_sha256':file_sha256(__file__),'science_acceptance':'NOT_ACCEPTED'})
        for path,digest in sources.items():
            if file_sha256(path)!=digest:raise RuntimeError('Native IC source changed during publication: '+path)
        report.update(execution_status='COMPLETE',native_ic={'path':str(output),'sha256':file_sha256(output)},
            native_dataset_contract=contract,original_music_header=original_music,
            retained_gas_state_policy=request['gas_state_policy'],
            transfer_acceptance='NOT_ESTABLISHED',handoff_acceptance='NOT_ESTABLISHED')
        atomic_json(manifest,report);atomic_text(manifest.with_name(manifest.name+'.sha256'),file_sha256(manifest)+'\n')
        return report
    except Exception as error:
        report.update(execution_status='FAILED',error=str(error));atomic_json(manifest,report);raise


def twofluid_native_ic_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native IC writer exceeds allocated stage memory')
    if parameters['request']['disk_gib']>context.resources['disk_gib']:raise MemoryError('Native IC writer exceeds allocated stage disk')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Native IC publication escapes owned generation')
    return publish_twofluid_native_ic(parameters,destination)


def twofluid_native_ic_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Write finite-epoch projected two-fluid GADGET ICs with original q/ID and gas-state provenance')
    parser.add_argument('--parameters',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();parameters=json.loads(args.parameters.read_text());parameters.setdefault('request_base',str(args.parameters.resolve().parent))
    publish_twofluid_native_ic(parameters,args.output)


def verify_twofluid_native_generation(path,*,memory_gib=1):
    """Read the transformed generation and check exact source/payload lineage."""
    import json
    from pathlib import Path
    import h5py
    from .foundation import file_sha256
    from .data import verify_external_hdf5_contract,exact_uint64_ids
    from .sidm_analysis import _load_product
    path=Path(path).resolve();digest=file_sha256(path);memory=memory_gib*1024**3
    if not math.isfinite(memory_gib) or memory_gib<=0 or path.stat().st_size*8>memory:raise MemoryError('Finite-epoch native generation metadata exceeds allocation')
    report=json.loads(path.read_text())
    if report.get('schema')!='twofluid-native-ic-generation-v1' or report.get('execution_status')!='COMPLETE':raise ValueError('Complete typed finite-epoch native generation required')
    validate_twofluid_native_ic_request({'request':report['request']})
    if path.with_name(path.name+'.sha256').read_text().strip()!=digest:raise ValueError('Finite-epoch native generation report seal changed')
    for source,expected in report['source_hashes'].items():
        if file_sha256(source)!=expected:raise ValueError('Finite-epoch native generation source changed: '+source)
    ic=Path(report['native_ic']['path']).resolve()
    if ic!=path.parent/'ics_gadget.hdf5' or ic.is_symlink() or file_sha256(ic)!=report['native_ic']['sha256']:raise ValueError('Exact immutable transformed native IC required')
    verify_external_hdf5_contract(ic,report['native_dataset_contract'])
    projection,arrays,sources=_load_product(report['particle_projection'],path.parent,memory/2)
    if projection.get('schema')!='twofluid-native-particle-projection-v1' or projection['particle_namespace']!=report['particle_namespace'] or projection['epoch']!=report['epoch']:raise ValueError('Native generation projection identity/epoch changed')
    if any(report['source_hashes'].get(source)!=expected for source,expected in sources.items()):raise ValueError('Generation does not retain the complete actual projection source bindings')
    if len(arrays['particle_ids'])*512>memory/2:raise MemoryError('Native generation comparison exceeds particle workspace')
    with h5py.File(ic,'r') as native:
        header=native['Header'].attrs;a=float(header['Time']);h=float(header['HubbleParam'])
        flow_kind=projection.get('flow_kind','second_order_perturbative')
        if flow_kind not in {'second_order_perturbative','full_nonlinear_material'} or report.get('flow_kind',flow_kind)!=flow_kind:
            raise ValueError('Native generation changes the actual evolved flow definition')
        order=header.get('Research_twofluid_perturbative_order')
        if (header.get('Research_twofluid_flow_schema')!=1 or header.get('Research_twofluid_projection_sha256')!=report['particle_projection']['sha256']
                or flow_kind=='second_order_perturbative' and order!=2
                or flow_kind=='full_nonlinear_material' and (order is not None or header.get('Research_twofluid_flow_kind')!=flow_kind)):
            raise ValueError('Native flow metadata does not bind its actual finite-epoch projection')
        if not math.isclose(a,float(report['epoch']['actual_scale_factor']),rel_tol=1e-12) or not np.array_equal(np.asarray(header['MassTable'])[:3],np.zeros(3)):raise ValueError('Native flow epoch or variable material-mass convention changed')
        if report.get('projection_mode','uniform_native_q_points')!=projection.get('projection_mode','uniform_native_q_points') or report.get('material_geometry')!=projection.get('material_geometry'):raise ValueError('Native material projection mode/geometry differs from its sealed producer')
        chunk=report['request']['chunk_rows']
        for kind in (0,1,2):
            if not np.any(arrays['species']==kind):continue
            group=native['PartType'+str(kind)];rows=np.flatnonzero(arrays['species']==kind)
            if not np.array_equal(exact_uint64_ids(group['ParticleIDs'][:]),arrays['particle_ids'][rows]):raise ValueError('Native flow changed projected particle identities')
            for first in range(0,len(rows),chunk):
                stop=min(first+chunk,len(rows));select=rows[first:stop]
                expected={'LagrangianCoordinates':arrays['lagrangian_position_mpc_h'][select]*1000,
                    'Coordinates':arrays['position_comoving_mpc_h'][select]*1000,
                    'Velocities':arrays['velocity_peculiar_km_s'][select]/math.sqrt(a),
                    'Masses':arrays['mass_msun'][select]*h/1e10}
                if flow_kind=='full_nonlinear_material':
                    if 'FirstOrderDisplacements' in group or 'FirstOrderVelocities' in group:raise ValueError('Full nonlinear native flow retains stale perturbative diagnostic fields')
                    expected['MaterialDisplacements']=arrays['material_displacement_mpc_h'][select]*1000
                else:expected.update(FirstOrderDisplacements=arrays['displacement_first_order_mpc_h'][select]*1000,
                    FirstOrderVelocities=arrays['velocity_first_order_peculiar_km_s'][select]/math.sqrt(a))
                if projection.get('projection_mode')=='finite_volume_material_cells':
                    expected.update(LagrangianCellCenters=arrays['cell_center_mpc_h'][select]*1000,LagrangianCellWidths=arrays['cell_width_mpc_h'][select]*1000)
                for name,value in expected.items():
                    # q was preserved byte for byte; a units round trip can
                    # differ by one float ULP and is bounded separately.
                    actual=group[name][first:stop]
                    same=np.allclose(actual,value,rtol=4*np.finfo(float).eps,atol=0) if name in {'LagrangianCoordinates','LagrangianCellCenters','LagrangianCellWidths'} else np.array_equal(actual,value)
                    if not same:raise ValueError('Native IC differs from sealed projected '+name)
    if file_sha256(path)!=digest:raise RuntimeError('Native transformed generation changed while reading')
    return report,projection,arrays


def verify_twofluid_native_handoff(ic,generation_report,cfg,destination,*,memory_gib=1):
    """Technical delivery of a finite-epoch flow, preserving original proof scope."""
    import json
    from pathlib import Path
    import h5py
    from .foundation import atomic_json,atomic_copy,file_sha256
    from .zoom import verify_generated_zoom,verify_native_handoff,initial_condition_census,_checked_record,validate_coarse_softening_masses,audit_target_coverage
    from .sidm_analysis import _load_product
    from .cosmology import class_background_table
    ic,generation_report,destination=[Path(value).resolve() for value in (ic,generation_report,destination)]
    if destination.exists():raise FileExistsError('Keep previous finite-epoch native handoff')
    generation,projection,particle_arrays=verify_twofluid_native_generation(generation_report,memory_gib=memory_gib)
    nested=bool(np.any(particle_arrays['species']==2));del particle_arrays
    if (ic!=Path(generation['native_ic']['path']) or cfg['simulation']['physics']!='hydro'
            or projection.get('flow_kind','second_order_perturbative')=='second_order_perturbative' and (type(cfg['simulation']['lpt_order_hydro'])is not int or cfg['simulation']['lpt_order_hydro']!=2)):
        raise ValueError('Finite-epoch transformed handoff requires its exact IC and declared second-order hydro flow')
    base_path=_checked_record(generation['base_native_generation']);base=verify_generated_zoom(base_path)
    fields,_,field_sources=_load_product(projection['relative_fields'],generation_report.parent,memory_gib*1024**3/2)
    if any(generation['source_hashes'].get(source)!=digest for source,digest in field_sources.items()):raise ValueError('Native flow does not retain all evolved-field sources')
    spec=fields['request'];h=cfg['cosmology']['H0']/100.;omega=(cfg['cosmology']['omega_b']+cfg['cosmology']['omega_cdm'])/h**2
    for name,expected in {'hubble_parameter':h,'hubble0_km_s_mpc':cfg['cosmology']['H0'],'omega_m0':omega,
        'baryon_fraction':cfg['cosmology']['omega_b']/(cfg['cosmology']['omega_b']+cfg['cosmology']['omega_cdm']),
        'box_mpc_h':cfg['simulation']['box_mpc_h'],'target_scale_factor':1/(1+cfg['simulation']['start_redshift'])}.items():
        if not math.isclose(float(spec[name]),float(expected),rel_tol=1e-12):raise ValueError('Finite-epoch flow cosmology/epoch mismatch: '+name)
    if cfg['cosmology']['model']=='wdm':raise ValueError('Finite-epoch cold two-fluid flow is not a thermal WDM representation')
    if nested:
        if projection.get('projection_mode')!='finite_volume_material_cells' or not cfg['simulation'].get('zoom_softening_mass_ratios'):raise ValueError('Nested transformed flow requires actual material-cell integrals and declared coarse softening classes')
    elif cfg['simulation'].get('zoom_softening_mass_ratios') is not None:raise ValueError('Uniform transformed flow cannot declare nested coarse softening classes')
    with h5py.File(ic,'r') as native:
        if nested:
            if cfg['simulation']['particles_per_dimension']!=2**int(native['Header'].attrs['Music_levelmax']):raise ValueError('Nested transformed finest cell level differs from requested resolution')
        elif len(native['PartType0/ParticleIDs'])!=cfg['simulation']['particles_per_dimension']**3:raise ValueError('Finite-epoch native lattice differs from requested resolution')
    solver_manifest=Path(cfg['spectrum']['file']).resolve();info=json.loads(solver_manifest.read_text())
    background=(solver_manifest.parent/info['source_files']['start']['background']).resolve()
    checksum=info['checksums_sha256'][info['source_files']['start']['background']]
    if file_sha256(background)!=checksum or background.stat().st_size*16>memory_gib*1024**3/2:raise ValueError('Actual background changed or exceeds handoff allocation')
    columns,table=class_background_table(background);scale=1/(1+table[:,columns['z']]);H=table[:,columns['H']]*299792.458
    order=np.argsort(scale);scale,H=scale[order],H[order];knots=np.asarray(spec['background_scale_factor']);rates=np.asarray(spec['background_hubble_km_s_mpc'])
    if not np.isfinite(scale).all() or not np.isfinite(H).all() or np.any(scale<=0) or np.any(H<=0) or np.any(np.diff(scale)<=0):raise ValueError('Actual retained solver expansion history is not positive and strictly ordered')
    if knots.ndim!=1 or len(knots)<4 or rates.shape!=knots.shape or not np.isfinite(knots).all() or not np.isfinite(rates).all() or np.any(knots<=0) or np.any(rates<=0) or np.any(np.diff(knots)<=0):raise ValueError('Finite-epoch flow background knot dimensions/domain changed')
    indices=np.searchsorted(scale,knots);right=np.clip(indices,0,len(scale)-1);left=np.clip(indices-1,0,len(scale)-1)
    nearest=np.where(abs(scale[left]-knots)<abs(scale[right]-knots),left,right)
    if not np.allclose(knots,scale[nearest],rtol=1e-12,atol=0) or not np.allclose(rates,H[nearest],rtol=1e-10,atol=0):
        raise ValueError('Finite-epoch flow must use retained actual solver background knots and H in km/s/Mpc')
    destination.mkdir(parents=True,exist_ok=False)
    report={'status':'IN_PROGRESS','schema':'twofluid-native-handoff-v1','science_acceptance':'NOT_ACCEPTED',
        'generation_report':{'path':str(generation_report),'sha256':file_sha256(generation_report)},
        'technical_scope':'source-bound finite-epoch '+projection.get('flow_kind','second_order_perturbative')+' '+('nested material' if nested else 'uniform')+' species flow; no inherited linear transfer proof for replacement'}
    atomic_json(destination/'handoff.json',report)
    try:
        base_cfg=json.loads(json.dumps(cfg))
        from .zoom import _configuration
        native_options=_configuration(base_path.parent/'music.conf')
        base_cfg['simulation']['lpt_order_hydro']=2 if native_options.getboolean('setup','use_2LPT',fallback=False) else 1
        base_cfg['simulation']['native_generation_report']=str(base_path)
        original_handoff=verify_native_handoff(base_path.parent/'ics_gadget.hdf5',base_path,base_cfg,destination/'original-native',memory_gib=memory_gib)
        if nested:
            report['softening_audit']=validate_coarse_softening_masses([ic],cfg['simulation'])
            region=_checked_record(base['traced_region'])
            report['coverage_audit']=audit_target_coverage(generation_report,region,destination/'coverage.json',memory_gib=memory_gib)
            if report['coverage_audit']['status']!='PASS':raise ValueError('Transformed material cells do not cover the original traced target guards')
        reference=destination/'reference';reference.mkdir()
        atomic_copy(destination/'original-native/reference/transfers.txt',reference/'transfers.txt')
        report.update(status='TECHNICAL_PASS',resource_census=initial_condition_census([ic]),
            structural_audit={'status':'STRUCTURAL_PASS','native_dataset_contract':generation['native_dataset_contract'],'science_acceptance':'NOT_ACCEPTED'},
            velocity_audit={'status':'SOURCE_BOUND_PROJECTION','method':fields['method'],'transfer_acceptance':'NOT_ESTABLISHED'},
            original_native_handoff={'path':str(destination/'original-native/handoff.json'),'sha256':file_sha256(destination/'original-native/handoff.json'),'scope':original_handoff['status']},
            background_source={'path':str(background),'sha256':checksum},
            remaining_science_gates=['independent finite-epoch nonlinear species-flow reference','source-grid and particle-projection convergence',
                'finite-reference-epoch and start-epoch convergence','nonlinear target and observable convergence'])
        if nested:report['remaining_science_gates']+=['window-aware nested IC spectrum/velocity comparison','evolving coarse contamination and target recovery']
        verify_twofluid_native_generation(generation_report,memory_gib=memory_gib)
        if file_sha256(background)!=checksum:raise RuntimeError('Actual solver background changed during finite-epoch handoff')
        atomic_json(destination/'handoff.json',report);return report
    except BaseException as error:
        report.update(status='FAILED',failure={'type':type(error).__name__,'message':str(error)});atomic_json(destination/'handoff.json',report);raise


def _windowed_material_phase_integral(weight,velocity,displacement,center,width,box,modes,order,*,budget,chunk_rows,grid_origin=None):
    """Integrate finite-volume density/momentum phases in an unwrapped cell.

    Unlike polynomial material moments, displaced Fourier phases are not
    integrated exactly by a fixed Gaussian rule. The producer must compare
    independent orders and retain its empirical quadrature uncertainty.
    """
    import itertools
    center=np.asarray(center,float);n=weight.shape[-1];spacing=box/n
    origin=np.zeros(3) if grid_origin is None else np.asarray(grid_origin,float)
    if origin.shape!=(3,) or not np.isfinite(origin).all():raise ValueError('Finite material phase grid origin required')
    intervals=[]
    for axis in range(3):
        low=center[axis]-width/2;high=center[axis]+width/2
        knots=np.arange(math.floor((low-origin[axis])/spacing)+1,math.ceil((high-origin[axis])/spacing),dtype=float)*spacing+origin[axis]
        edges=np.r_[low,knots[(knots>low)&(knots<high)],high];intervals.append(list(zip(edges[:-1],edges[1:])))
    needed=math.prod(len(rows) for rows in intervals)*order**3*len(modes)
    if needed>budget['maximum_phase_evaluations']-budget['phase_evaluations']:raise MemoryError('Windowed material Fourier quadrature exceeds the frozen phase-evaluation budget')
    budget['phase_evaluations']+=needed
    nodes,weights=np.polynomial.legendre.leggauss(order);k=2*np.pi*np.asarray(modes,float)/box
    density=np.zeros(len(k),np.clongdouble);momentum=np.zeros((len(k),3),np.clongdouble)
    for cube in itertools.product(*intervals):
        low=np.asarray([row[0] for row in cube]);high=np.asarray([row[1] for row in cube]);mid=(low+high)/2;half=(high-low)/2
        def samples():
            for i,j,l in itertools.product(range(order),repeat=3):
                yield mid+half*np.asarray([nodes[i],nodes[j],nodes[l]]),float(np.prod(half))*weights[i]*weights[j]*weights[l]
        rows=[];volume=[]
        def accumulate():
            q=np.asarray(rows);dv=np.asarray(volume);material=_periodic_trilinear(weight,q-origin,box)
            position=q+_periodic_trilinear(displacement,q-origin,box);speed=_periodic_trilinear(velocity,q-origin,box)
            phase=np.exp(-1j*(position@k.T));weighted=phase*(material*dv)[:,None]
            density[:]+=np.sum(weighted,axis=0,dtype=np.clongdouble)
            momentum[:]+=np.einsum('nm,ni->mi',weighted,speed,dtype=np.clongdouble)
        for q,dv in samples():
            rows.append(q);volume.append(dv)
            if len(rows)==chunk_rows:accumulate();rows=[];volume=[]
        if rows:accumulate()
    return np.asarray(density,np.complex128),np.asarray(momentum,np.complex128)


def validate_twofluid_window_request(parameters):
    request=parameters['request']
    if request.get('schema')!='twofluid-window-comparison-request-v1':raise ValueError('Typed material-window comparison request required')
    for name in ('maximum_particles','maximum_modes','maximum_windows','chunk_rows','maximum_phase_evaluations'):
        if type(request[name])is not int or request[name]<1:raise ValueError('Exact positive material-window bound required: '+name)
    for name in ('memory_gib','maximum_k_cell_width','density_absolute_tolerance','momentum_absolute_tolerance_km_s','quadrature_density_tolerance','quadrature_momentum_tolerance_km_s'):
        if isinstance(request[name],bool) or not isinstance(request[name],(int,float)) or not math.isfinite(request[name]) or request[name]<=0:raise ValueError('Positive finite material-window tolerance required: '+name)
    if request['maximum_k_cell_width']>np.pi/2:raise ValueError('Material-window comparison must resolve the represented cell below k*width=pi/2')
    orders=request['quadrature_orders']
    if not isinstance(orders,list) or len(orders)<2 or any(type(value)is not int or not 2<=value<=32 for value in orders) or any(right<=left for left,right in zip(orders,orders[1:])):raise ValueError('Explicit increasing Gaussian quadrature orders, at least two, in [2,32] required')
    modes=np.asarray(request['integer_wavevectors'])
    if modes.ndim!=2 or modes.shape[1]!=3 or modes.dtype.kind not in 'iu' or not 1<=len(modes)<=request['maximum_modes'] or np.any(np.all(modes==0,axis=1)) or len(np.unique(modes,axis=0))!=len(modes):raise ValueError('Bounded distinct exact nonzero integer Fourier wavevectors required')
    windows=request['windows'];used_names=set()
    if not isinstance(windows,list) or not 1<=len(windows)<=request['maximum_windows']:raise ValueError('Explicit bounded original material-cell windows required')
    for window in windows:
        name=window['name'];kinds=window['native_types'];low=window['minimum_level'];high=window['maximum_level']
        if not isinstance(name,str) or not name or name in used_names or not isinstance(kinds,list) or not kinds or any(type(kind)is not int or kind not in (0,1,2) for kind in kinds) or len(set(kinds))!=len(kinds) or type(low)is not int or type(high)is not int or not 1<=low<=high<=20:raise ValueError('Window needs a unique name, exact species types and ordered dyadic levels')
        used_names.add(name)
    _validate_twofluid_product_reference(request['native_generation'],'native_generation')
    return request


def twofluid_window_input_bindings(parameters):
    from pathlib import Path
    from .foundation import file_sha256
    from .sidm_analysis import _load_product
    request=validate_twofluid_window_request(parameters);path=(Path(parameters['request_base'])/request['native_generation']['path']).resolve()
    if file_sha256(path)!=request['native_generation']['sha256']:raise ValueError('Material-window native generation changed')
    generation,projection,particles=verify_twofluid_native_generation(path,memory_gib=request['memory_gib']);del particles
    if projection.get('projection_mode')!='finite_volume_material_cells':raise ValueError('Material-window comparison requires actual finite-volume species integrals')
    _,_,sources=_load_product(projection['relative_fields'],path.parent,request['memory_gib']*1024**3/2)
    sources.update(generation['source_hashes']);sources[str(path)]=request['native_generation']['sha256']
    sources[str(path.with_name(path.name+'.sha256'))]=file_sha256(path.with_name(path.name+'.sha256'))
    sources[generation['native_ic']['path']]=generation['native_ic']['sha256'];return sources


def publish_twofluid_window_comparison(parameters,destination):
    """Compare actual native particles with identically windowed species flow.

    This delivery/resolution comparison is conditional on the retained flow.
    It does not establish that its physical seed is an independent correct
    Boltzmann realization or that a target remains uncontaminated later.
    """
    from pathlib import Path
    import h5py
    from .foundation import file_sha256,G_MPC_KMS2_MSUN
    from .sidm_analysis import _load_product
    from .data import export_numeric_audit
    request=validate_twofluid_window_request(parameters);base=Path(parameters['request_base'])
    orders=request['quadrature_orders'];modes=np.asarray(request['integer_wavevectors']);windows=request['windows']
    memory=request['memory_gib']*1024**3;reference=request['native_generation'];path=(base/reference['path']).resolve()
    if file_sha256(path)!=reference['sha256']:raise ValueError('Material-window native generation changed')
    generation,projection,particles=verify_twofluid_native_generation(path,memory_gib=request['memory_gib'])
    if projection.get('projection_mode')!='finite_volume_material_cells':raise ValueError('Material-window comparison requires actual finite-volume species integrals')
    if len(particles['particle_ids'])>request['maximum_particles'] or request['chunk_rows']*len(modes)*512>memory/4:raise MemoryError('Material-window comparison exceeds particle or phase workspace')
    fields,grids,sources=_load_product(projection['relative_fields'],path.parent,memory/2);sources.update(generation['source_hashes'])
    sources[str(path)]=reference['sha256'];sources[str(path.with_name(path.name+'.sha256'))]=file_sha256(path.with_name(path.name+'.sha256'))
    ic=Path(generation['native_ic']['path']);sources[str(ic)]=generation['native_ic']['sha256']
    box=float(fields['request']['box_mpc_h']);h=float(fields['request']['hubble_parameter']);a=float(fields['target_scale_factor']);k=2*np.pi*modes/box
    n=grids['lagrangian_mass_weights'].shape[-1]
    if len(particles['particle_ids'])*2048+n**3*1024+len(windows)*len(modes)*512>memory/2:raise MemoryError('Combined material-window particle, grid and output workspace exceeds its allocation')
    if np.any(abs(modes)>=n/2):raise ValueError('Window modes reach or exceed the actual source-grid Nyquist limit')
    offset=np.asarray(projection['material_geometry']['baryon_material_offset_mpc_h'])
    displacement,material_velocity,grid_origin,_=_twofluid_material_fields(fields,grids)
    fraction=float(fields['request']['baryon_fraction']);rho=float(fields['request']['omega_m0'])*3*float(fields['request']['hubble0_km_s_mpc'])**2/(8*np.pi*G_MPC_KMS2_MSUN)
    species_density=rho/h**3*np.asarray([1-fraction,fraction])*np.asarray([projection['mass_rescaling'][str(index)] for index in (0,1)])
    budget={'maximum_phase_evaluations':request['maximum_phase_evaluations'],'phase_evaluations':0}
    actual_position=np.empty((len(particles['particle_ids']),3));actual_velocity=np.empty_like(actual_position);actual_mass=np.empty(len(actual_position))
    with h5py.File(ic,'r') as native:
        for kind in (0,1,2):
            use=np.flatnonzero(particles['species']==kind)
            if not len(use):continue
            group=native['PartType'+str(kind)];actual_position[use]=group['Coordinates'][:]/1000;actual_velocity[use]=group['Velocities'][:]*math.sqrt(a);actual_mass[use]=group['Masses'][:]*1e10/h
    results={};records=[];used_names=set();all_pass=True
    for window in windows:
        name=window['name'];kinds=window['native_types'];low=window['minimum_level'];high=window['maximum_level']
        if not isinstance(name,str) or not name or name in used_names or not isinstance(kinds,list) or not kinds or any(type(kind)is not int or kind not in (0,1,2) for kind in kinds) or len(set(kinds))!=len(kinds) or type(low)is not int or type(high)is not int or not 1<=low<=high<=20:raise ValueError('Window needs a unique name, exact species types and ordered dyadic levels')
        used_names.add(name);selected=np.flatnonzero(np.isin(particles['species'],kinds)&(particles['cell_level']>=low)&(particles['cell_level']<=high))
        if not len(selected):raise ValueError('Requested original material window is empty: '+name)
        if np.max(np.linalg.norm(k,axis=1))*np.max(particles['cell_width_mpc_h'][selected])>request['maximum_k_cell_width']:raise ValueError('Window Fourier modes are unresolved by its largest represented leaf cell')
        mass_total=float(actual_mass[selected].sum());measured=np.zeros(len(modes),np.complex128);measured_momentum=np.zeros((len(modes),3),np.complex128)
        particle_evaluations=len(selected)*len(modes)
        if particle_evaluations>budget['maximum_phase_evaluations']-budget['phase_evaluations']:raise MemoryError('Native particle Fourier sums exceed the frozen window phase budget')
        budget['phase_evaluations']+=particle_evaluations
        for first in range(0,len(selected),request['chunk_rows']):
            rows=selected[first:first+request['chunk_rows']];phase=np.exp(-1j*(actual_position[rows]@k.T));weighted=actual_mass[rows,None]*phase
            measured+=weighted.sum(axis=0);measured_momentum+=weighted.T@actual_velocity[rows]
        predicted=[]
        for order in orders:
            density=np.zeros(len(modes),np.complex128);momentum=np.zeros((len(modes),3),np.complex128)
            for row in selected:
                for species in (0,1):
                    mass=float(particles['species_mass_msun'][row,species])
                    if mass==0:continue
                    center=particles['cell_center_mpc_h'][row]+(offset if species==1 else 0);width=float(particles['cell_width_mpc_h'][row])
                    constant=species_density[species]
                    d,p=_windowed_material_phase_integral(grids['lagrangian_mass_weights'][species],material_velocity[species],displacement[species],center,width,box,modes,order,budget=budget,chunk_rows=request['chunk_rows'],grid_origin=grid_origin)
                    density+=constant*d;momentum+=constant*p
            predicted.append((density/mass_total,momentum/mass_total))
        density_error=np.max(abs(predicted[-1][0]-predicted[-2][0]));momentum_error=np.max(np.linalg.norm(predicted[-1][1]-predicted[-2][1],axis=1))
        residual_density=abs(measured/mass_total-predicted[-1][0]);residual_momentum=np.linalg.norm(measured_momentum/mass_total-predicted[-1][1],axis=1)
        passed=bool(density_error<=request['quadrature_density_tolerance'] and momentum_error<=request['quadrature_momentum_tolerance_km_s'] and np.max(residual_density)+density_error<=request['density_absolute_tolerance'] and np.max(residual_momentum)+momentum_error<=request['momentum_absolute_tolerance_km_s'])
        all_pass &= passed;prefix='window_'+str(len(records))
        results[prefix+'_particle_density']=measured/mass_total;results[prefix+'_reference_density']=predicted[-1][0]
        results[prefix+'_particle_momentum']=measured_momentum/mass_total;results[prefix+'_reference_momentum']=predicted[-1][1]
        records.append({'name':name,'status':'PASS' if passed else 'FAIL','selected_particle_count':len(selected),'selected_mass_msun':mass_total,
            'quadrature_density_order_difference':float(density_error),'quadrature_momentum_order_difference_km_s':float(momentum_error),
            'maximum_density_residual':float(residual_density.max()),'maximum_momentum_residual_km_s':float(residual_momentum.max()),'array_prefix':prefix})
    results['integer_wavevectors']=modes
    report={'schema':'twofluid-window-comparison-v1','status':'PASS' if all_pass else 'FAIL','request':request,'epoch':fields['epoch'],'windows':records,
        'phase_evaluations':budget['phase_evaluations'],'source_hashes':sources,'particle_ids_unique':True,
        'particle_id_validation':{'method':'exact sealed complete native species identity census','particle_namespace':projection['particle_namespace']},
        'scope':'windowed Fourier mass and physical peculiar momentum of actual material cells against retained species-flow integrals',
        'reference_representation':'periodic piecewise-trilinear material weight, species displacement and velocity; actual normalized background species density',
        'quadrature_scope':'empirical successive-order comparison; not a rigorous error bound or independent physical calibration','science_acceptance':'NOT_ACCEPTED'}
    sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    specs={name:{'units':'km / s' if 'momentum' in name else 'integer mode index' if name=='integer_wavevectors' else 'dimensionless',
        'quantity':'material-window '+name.replace('_',' ')} for name in results}
    return export_numeric_audit(destination,report,results,specs,algorithm=Path(__file__).resolve(),context={'epoch':fields['epoch']},source_hashes=sources)


def twofluid_window_runner(parameters,context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Material-window comparison exceeds allocated stage memory')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Material-window product escapes owned generation')
    return publish_twofluid_window_comparison(parameters,destination)


def twofluid_window_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Compare actual material-window native particle density/momentum to retained species-flow integrals')
    parser.add_argument('--parameters',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();parameters=json.loads(args.parameters.read_text());parameters.setdefault('request_base',str(args.parameters.resolve().parent))
    publish_twofluid_window_comparison(parameters,args.output)


def native_twofluid_window_gate(cfg,run_dir,ic_path):
    """Execute the frozen scoped material comparison before a native launch."""
    import copy,json,datetime
    from pathlib import Path
    from .foundation import file_sha256, atomic_json
    from core.resources import memory_limit_gb
    path=Path(cfg['simulation']['native_generation_report']).resolve();memory=memory_limit_gb(cfg)
    generation,projection,particles=verify_twofluid_native_generation(path,memory_gib=memory)
    if Path(ic_path).resolve()!=Path(generation['native_ic']['path']):raise ValueError('Window evolution gate requires the exact supplied transformed IC')
    template=generation['request'].get('window_comparison')
    if not isinstance(template,dict) or 'native_generation' in template:raise ValueError('Transformed material evolution requires a frozen window_comparison template without a replaceable native_generation')
    request=copy.deepcopy(template);request['native_generation']={'path':str(path),'sha256':file_sha256(path)}
    validate_twofluid_window_request({'request':request})
    if request['memory_gib']>memory:raise MemoryError('Frozen material-window comparison exceeds the allocated native run memory')
    if np.linalg.matrix_rank(np.asarray(request['integer_wavevectors'],float))!=3:raise ValueError('Native material window gate needs modes spanning all three spatial axes')
    # Require each actual native population to be covered in its own window,
    # rather than permitting cancellation in a total-matter-only statistic.
    for kind in np.unique(particles['species']):
        levels=particles['cell_level'][particles['species']==kind]
        if not any(window['native_types']==[int(kind)] and window['minimum_level']<=int(levels.min()) and window['maximum_level']>=int(levels.max()) for window in request['windows']):
            raise ValueError('Native material gate omits a complete separate species/hierarchy window: '+str(kind))
    del particles
    destination=Path(run_dir).resolve()/'input'/('material-window-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    report=publish_twofluid_window_comparison({'request':request,'request_base':str(path.parent)},destination)
    result={**report,'comparison_path':str(destination),'comparison_sha256':file_sha256(destination),
        'physics_validation_scope':report['scope'],'linear_spectrum_acceptance':'NOT_ESTABLISHED'}
    atomic_json(destination.with_name(destination.stem+'-launch.json'),result)
    return result


def twofluid_relative_runner(parameters, context):
    from pathlib import Path
    if parameters['request']['memory_gib']>context.resources['memory_gib']:
        raise MemoryError('Two-fluid fields exceed allocated stage memory')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Two-fluid grids escape the owned stage generation')
    return publish_twofluid_relative_product(parameters,destination)


def twofluid_relative_main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Produce finite-epoch nonlinear two-fluid relative-mode grids from sealed initial fields')
    parser.add_argument('parameters',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();path=args.parameters.resolve();parameters=json.loads(path.read_text())
    parameters.setdefault('request_base',str(path.parent))
    publish_twofluid_relative_product(parameters,args.output)
    return 0


def _field(value,name):
    value = np.asarray(value,dtype=float)
    if (value.ndim != 3 or len(set(value.shape)) != 1 or value.shape[0] < 4
            or not np.isfinite(value).all()):
        raise ValueError(name+' requires a finite cubic grid with at least four cells per axis')
    if abs(float(value.mean())) > max(1e-13,float(np.max(abs(value)))*1e-11):
        raise ValueError(name+' has a nonzero homogeneous density/velocity mode')
    return value


def _resize(values,n):
    for axis in range(3):
        values = resample(values,n,axis=axis)
    return values.real


def mixed_fluid_2lpt(delta_dark,delta_baryon,*,baryon_fraction,box_mpc_h,growth,
                     aH_km_s_Mpc,hubble_parameter,relative_theta=None,relative_velocity_order=0,memory_gib=1.):
    """Common two-fluid flow and compensated masses on a periodic reference mesh.

    Displacements use comoving Mpc/h; velocities are proper peculiar km/s.
    Quadratic products are evaluated on a twice-padded grid, then projected
    back, so high-frequency products cannot alias into retained modes.
    Mass weights are dimensionless multiples of each species mean mass.
    """
    dark,baryon = _field(delta_dark,'dark density'),_field(delta_baryon,'baryon density')
    if dark.shape != baryon.shape:
        raise ValueError('Species grids must coincide in their Lagrangian frame')
    if (not math.isfinite(baryon_fraction) or not 0 < baryon_fraction < 1
            or not all(math.isfinite(v) and v > 0 for v in (box_mpc_h,aH_km_s_Mpc,hubble_parameter,memory_gib))
            or not isinstance(growth,SecondOrderGrowth)):
        raise ValueError('Declare physical species fractions, units and integrated growth')
    if type(relative_velocity_order) is not int or relative_velocity_order not in (0,1):
        raise ValueError('Relative velocity is supported only as an explicit first-order correction')
    relative = None
    if relative_theta is not None:
        relative = _field(relative_theta,'relative velocity divergence')
        if relative.shape != dark.shape or (np.any(relative != 0) and relative_velocity_order != 1):
            raise ValueError('Nonzero relative velocity requires the explicit first-order relative velocity domain')
    n = dark.shape[0]
    if 1200*n**3+64*1024**2 > memory_gib*1024**3:
        raise MemoryError('Dealiased mixed-fluid reference exceeds declared workspace')
    total = (1-baryon_fraction)*dark+baryon_fraction*baryon
    compensated = {'dark':dark-total,'baryon':baryon-total}
    weights = {name:1+value for name,value in compensated.items()}
    if any(np.any(value <= 0) for value in weights.values()):
        raise ValueError('Compensated species densities produce nonpositive particle masses')
    fourier = np.fft.fftn(total)
    # Real even-grid Nyquist derivatives are ambiguous under subcell shifts.
    # Reject rather than silently dropping an energetic supplied mode.
    if n%2 == 0:
        nyquist = max(float(np.max(abs(np.take(fourier,n//2,axis=axis)))) for axis in range(3))
        if nyquist > max(1e-11,float(np.max(abs(fourier)))*1e-10):
            raise ValueError('Reference requires band-limited input without energetic Nyquist modes')
    axes = np.meshgrid(*([2*np.pi*np.fft.fftfreq(n,d=box_mpc_h/n)]*3),indexing='ij',sparse=True)
    k2 = sum(k*k for k in axes)
    inv = np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
    psi1 = np.stack([np.fft.ifftn(1j*k*inv*fourier).real for k in axes],axis=-1)
    padded_hessian = {}
    for i in range(3):
        for j in range(i,3):
            padded_hessian[i,j] = _resize(np.fft.ifftn(axes[i]*axes[j]*inv*fourier).real,2*n)
    source = np.zeros((2*n,)*3)
    for i in range(3):
        for j in range(i+1,3):
            source += padded_hessian[i,i]*padded_hessian[j,j]-padded_hessian[i,j]**2
    source = _resize(source,n)
    zero = float(source.mean())
    source -= zero
    second = np.fft.fftn(source)
    second[0,0,0] = 0.
    basis2 = np.stack([np.fft.ifftn(1j*k*inv*second).real for k in axes],axis=-1)
    psi2 = growth.displacement_coefficient*basis2
    factor = aH_km_s_Mpc/hubble_parameter
    velocity1 = factor*growth.linear_growth_rate*psi1
    velocity2 = factor*growth.velocity_coefficient*basis2
    relative_velocity = np.zeros_like(velocity1)
    if relative is not None:
        # relative_theta is a realized divergence in km/s per comoving Mpc/h.
        # No growing-mode rescaling or higher-order relative kernel is applied.
        relative_fourier = np.fft.fftn(relative)
        relative_velocity = np.stack([np.fft.ifftn(-1j*k*inv*relative_fourier).real for k in axes],axis=-1)
    common_velocity = velocity1+velocity2
    return {'displacement':psi1+psi2,'velocity_km_s':common_velocity,
            'species_velocities_km_s':{'dark':common_velocity-baryon_fraction*relative_velocity,
                                      'baryon':common_velocity+(1-baryon_fraction)*relative_velocity},
            'relative_velocity_order':relative_velocity_order,
            'first_order_displacement':psi1,'second_order_displacement':psi2,
            'first_order_velocity_km_s':velocity1,'second_order_velocity_km_s':velocity2,
            'mass_weights':weights,'compensated_density':compensated,'second_order_source':source,
            'removed_source_zero_mode':zero,'scope':'uniform periodic reference; common growing mode with compensated species masses',
            'reference': 'https://arxiv.org/abs/2008.09124', 'science_acceptance':'NOT_ACCEPTED'}

# ========================================================================
# ic_validation: Physical IC controls, independent of generator success or saved PASS labels.
from pathlib import Path
import configparser
import json
import h5py

from core.foundation import file_sha256
from core.data import epoch_contract, snapshot_total_counts
from core.foundation import KPC_CM
from core.foundation import atomic_json
from core.native import directory_owner


def _positive(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be a finite number')
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(name + ' must be a finite number') from None
    if not math.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        raise ValueError(name + ' is outside its physical range')
    return result


def _unchanged(hashes):
    for path, checksum in hashes.items():
        if file_sha256(path) != checksum:
            raise ValueError('IC validation input changed during reading: ' + path)


def validate_linear_bundle(source, *, model=None, redshift=None,
                           box_mpc_h=None, refinement_level=None, memory_gib=1.):
    """Verify schema-2 solver bytes, gauge, species, epoch and mesh coverage.

    This checks the saved solver input as well as hashes: updating a hash does
    not authorize a different input cosmology. No transfer is growth-rescaled.
    Endpoint samples are excluded, matching the native natural-spline domain.
    """
    from zoom import validate_solver_input
    from core.cosmology import class_background_table
    import re
    source = Path(source).expanduser().resolve()
    before = {str(source): file_sha256(source)}
    info = json.loads(source.read_text())
    if (info.get('schema_version') != 2 or info.get('status') != 'complete'
            or info.get('model') not in {'lcdm', 'ede', 'wdm'}):
        raise ValueError('Linear bundle requires a complete schema-2 solver manifest')
    actual_model = info['model']
    if model is not None and model != actual_model:
        raise ValueError('Linear bundle model mismatch')
    parameters = info['parameters']
    if parameters.get('model') != actual_model:
        raise ValueError('Linear bundle requested model mismatch')
    if info.get('solver') != ('AxiCLASS' if actual_model == 'ede' else 'CLASS'):
        raise ValueError('Linear bundle solver engine mismatch')
    solver = info['class_parameters']['start']
    if (str(solver.get('nbody_gauge_transfer_functions', '')).lower() != 'yes'
            or not {'mTk', 'vTk'} <= {x.strip() for x in str(solver.get('output', '')).split(',')}):
        raise ValueError('Linear bundle requires N-body gauge density and velocity transfers')
    z = _positive(parameters['z_start'], 'start redshift')
    requested = z if redshift is None else _positive(redshift, 'requested redshift')
    if not math.isclose(z, requested, abs_tol=1e-8, rel_tol=0):
        raise ValueError('Linear bundle has no transfers at the exact requested epoch; solver regeneration is required')
    if not math.isclose(float(solver['z_pk']), z, abs_tol=1e-8, rel_tol=0):
        raise ValueError('Linear bundle saved solver epoch mismatch')
    paths = {}
    for kind in ('transfer', 'background', 'ini'):
        relative = info['source_files']['start'][kind]
        if not isinstance(relative, str):
            raise ValueError('Malformed linear bundle source path')
        target = (source.parent / relative).resolve()
        if (not target.is_relative_to(source.parent) or not target.is_file()
                or info['checksums_sha256'].get(relative) != file_sha256(target)):
            raise ValueError('Changed or missing linear bundle source: ' + kind)
        paths[kind] = target
        before[str(target)] = file_sha256(target)
    validate_solver_input(paths['ini'], solver)
    h = _positive(solver['H0'], 'H0') / 100
    for key in ('A_s', 'n_s', 'omega_b', 'omega_cdm'):
        _positive(solver[key], key)
    matching = {'H0': solver['H0'], 'A_s': solver['A_s'], 'n_s': solver['n_s'],
                'Omega_b': float(solver['omega_b']) / h**2}
    # The request calls its dark density Omega_cdm even when CLASS replaces
    # that population with one massive thermal relic.
    if actual_model != 'wdm':
        matching['Omega_cdm'] = float(solver['omega_cdm']) / h**2
        if any(float(solver.get(key, 0)) != 0 for key in ('N_ncdm', 'omega_ncdm', 'Omega_ncdm')):
            raise ValueError('Undeclared massive species in cold linear bundle')
    else:
        if float(solver.get('N_ncdm', 0)) != 1:
            raise ValueError('WDM requires exactly one declared relic species')
        for key in ('m_ncdm', 'T_ncdm', 'deg_ncdm'):
            _positive(solver[key], key)
    for key, value in matching.items():
        if not math.isclose(float(parameters[key]), float(value), rel_tol=1e-12):
            raise ValueError('Linear bundle cosmology mismatch: ' + key)
    budget = _positive(memory_gib, 'memory budget') * 1024**3
    if 12 * paths['transfer'].stat().st_size + 64 * 1024**2 > budget:
        raise MemoryError('Linear transfer validation exceeds declared workspace')
    columns, observed_z = {}, None
    with paths['transfer'].open() as stream:
        for line in stream:
            if not line.lstrip().startswith('#'):
                continue
            match = re.search(r'redshift\s+z\s*=\s*([-+0-9.eE]+)', line, re.I)
            if match:
                observed_z = float(match[1])
            for match in re.finditer(r'(\d+)\s*:\s*([A-Za-z0-9_\[\]]+)', line):
                name, index = match[2].lower(), int(match[1]) - 1
                if name in columns and columns[name] != index:
                    raise ValueError('Conflicting transfer column header')
                columns[name] = index
    # CLASS prints rounded redshifts; exact epoch is separately bound to the
    # unrounded saved z_pk, not inferred from this display header.
    from core.cosmology import class_redshift_matches
    if not class_redshift_matches(observed_z, z):
        raise ValueError('Transfer payload epoch mismatch')
    dark = 'ncdm[0]' if actual_model == 'wdm' else 'cdm'
    required = ('k', 'd_b', 't_b', 'd_' + dark, 't_' + dark, 'h_t_nb_prime', 'k2gamma_nb')
    if not set(required) <= columns.keys():
        raise ValueError('Missing required gauge/species transfer columns: ' + ', '.join(sorted(set(required) - columns.keys())))
    table = np.loadtxt(paths['transfer'], comments='#', ndmin=2)
    if len(table) < 8 or table.shape[1] <= max(columns.values()):
        raise ValueError('Malformed species transfer table')
    fields = table[:, [columns[name] for name in required]]
    k = fields[:, 0]
    if not np.isfinite(fields).all() or np.any(k <= 0) or np.any(np.diff(k) <= 0):
        raise ValueError('Species transfers must be finite with ordered positive k')
    coverage = None
    if (box_mpc_h is None) != (refinement_level is None):
        raise ValueError('Declare both box and finest refinement level for coverage')
    if box_mpc_h is not None:
        box = _positive(box_mpc_h, 'box length')
        if type(refinement_level) is not int or not 1 <= refinement_level <= 20:
            raise ValueError('Refinement level must be an integer in [1,20]')
        needed = [2 * np.pi / box, np.sqrt(3) * np.pi * 2**refinement_level / box]
        if k[1] > needed[0] or k[-2] < needed[1]:
            raise ValueError('Species transfers do not cover the finest mesh diagonal')
        coverage = {'required_k_h_Mpc': list(map(float, needed)), 'status': 'PASS'}
    if 12 * paths['background'].stat().st_size + 64 * 1024**2 > budget:
        raise MemoryError('Background validation exceeds declared workspace')
    bc, data = class_background_table(paths['background'])
    if not {'z', 'H', 'growth_D', 'growth_f'} <= bc.keys():
        raise ValueError('Background requires z, H, growth_D and growth_f')
    a = 1 / (1 + data[:, bc['z']])
    selected = data[:, [bc[name] for name in ('H', 'growth_D', 'growth_f')]]
    if (len(a) < 2 or not np.isfinite(a).all() or np.any(a <= 0)
            or not np.isfinite(selected).all() or np.any(selected <= 0)
            or len(np.unique(a)) != len(a) or not a.min() <= 1 / (1 + z) <= a.max()):
        raise ValueError('Background has invalid samples or lacks the selected epoch')
    _unchanged(before)
    return {'schema': 'ic-linear-bundle-v1', 'status': 'PASS', 'model': actual_model,
            'gauge': 'N-body', 'species': ['baryon', 'thermal_relic' if actual_model == 'wdm' else 'cdm'],
            'epoch': epoch_contract(1 / (1 + z)), 'source_redshift': z,
            'manifest': {'path': str(source), 'sha256': before[str(source)]},
            'sources': {kind: {'path': str(path), 'sha256': before[str(path)]} for kind, path in paths.items()},
            'k_range_h_Mpc': [float(k[1]), float(k[-2])], 'mode_coverage': coverage,
            'background_a_range': [float(a.min()), float(a.max())],
            'science_acceptance': 'NOT_ACCEPTED'}


def resolve_start_epoch(source, redshift, *, regenerate=None, allow_regeneration=False, **validation):
    """Select exact solver epoch or an explicitly authorized, linked child bundle."""
    z = _positive(redshift, 'requested redshift')
    bundle = validate_linear_bundle(source, **validation)
    if math.isclose(bundle['source_redshift'], z, abs_tol=1e-8, rel_tol=0):
        return bundle
    if allow_regeneration is not True or not callable(regenerate):
        raise ValueError('Requested epoch requires explicitly authorized solver regeneration')
    parent_hash = bundle['manifest']['sha256']
    child = Path(regenerate(Path(source).resolve(), z)).resolve()
    if child == Path(source).resolve():
        raise ValueError('Epoch regeneration must create a distinct child bundle')
    info = json.loads(child.read_text())
    lineage = info.get('parent_bundle', {})
    if (lineage.get('sha256') != parent_hash
            or Path(lineage.get('path', '')).resolve() != Path(source).resolve()):
        raise ValueError('Regenerated child lacks exact parent bundle lineage')
    _unchanged({str(Path(source).resolve()): parent_hash})
    validate_linear_bundle(source, **validation)
    return validate_linear_bundle(child, redshift=z, **validation)


def _mode_vectors(modes):
    values = np.asarray(modes)
    if (values.ndim != 2 or values.shape[1] != 3 or not len(values)
            or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
            or not np.isfinite(values).all() or np.any(values != np.rint(values))
            or np.any(np.all(values == 0, axis=1))):
        raise ValueError('Declare finite nonzero integer Fourier mode vectors')
    return values.astype(np.float64)


def measure_ic_modes(ic, modes, *, memory_gib=1., particle_types=(0, 1, 2)):
    """Direct particle density and bulk-velocity modes (no mesh assignment).

    Density uses delivered x; velocity uses original q to isolate the linear
    divergence. ThermalVelocities, when present, are subtracted only from the
    bulk diagnostic. The recorded native momentum shift is restored for transfer
    comparisons. Conservation checks always use the actual delivered velocity.
    """
    modes = _mode_vectors(modes)
    if (not isinstance(particle_types, tuple) or not particle_types
            or len(set(particle_types)) != len(particle_types)
            or any(type(t) is not int or t not in (0, 1, 2) for t in particle_types)):
        raise ValueError('Declare distinct matter particle types')
    budget = _positive(memory_gib, 'memory budget') * 1024**3 - 64 * 1024**2
    cost = 128 + 64 * len(modes)
    if budget < cost:
        raise MemoryError('IC Fourier controls exceed declared workspace')
    chunk = max(1, min(16384, int(budget // cost)))
    density = np.zeros(len(modes), complex)
    theta = np.zeros_like(density)
    total = 0.
    with h5py.File(ic) as source:
        header = source['Header'].attrs
        box = _positive(header['BoxSize'], 'IC box')
        a = _positive(header['Time'], 'IC epoch')
        if (header.get('Music_lagrangian_schema') != 1
                or not math.isclose(float(header.get('Music_lagrangian_unitlength_cm', np.nan)), KPC_CM, rel_tol=1e-12)):
            raise ValueError('Bulk Fourier measurement requires original native q in comoving kpc/h')
        k = modes * (2 * np.pi / box)
        for t in particle_types:
            group = source.get('PartType' + str(t))
            if group is None:
                continue
            n = len(group['ParticleIDs'])
            for first in range(0, n, chunk):
                end = min(n, first + chunk)
                x = np.asarray(group['Coordinates'][first:end], dtype=float)
                q = np.asarray(group['LagrangianCoordinates'][first:end], dtype=float)
                v = np.asarray(group['Velocities'][first:end], dtype=float)
                mass = (np.asarray(group['Masses'][first:end], dtype=float) if 'Masses' in group
                        else np.full(end - first, header['MassTable'][t]))
                if 'ThermalVelocities' in group:
                    thermal = np.asarray(group['ThermalVelocities'][first:end], dtype=float)
                    if thermal.shape != v.shape or not np.isfinite(thermal).all():
                        raise ValueError('Invalid thermal velocity payload')
                    v = v - thermal
                if header.get('Music_momentum_schema') == 1:
                    frame = np.asarray(header.get('Music_removed_bulk_velocity_gadget',[]),dtype=float)
                    if frame.shape != (3,) or not np.isfinite(frame).all():
                        raise ValueError('Invalid native momentum frame shift')
                    v = v+frame
                if (x.shape != (end-first, 3) or q.shape != x.shape or v.shape != x.shape
                        or mass.shape != (end-first,) or any(not np.isfinite(y).all() for y in (x, q, v, mass))
                        or np.any(mass <= 0) or np.any(x < 0) or np.any(x >= box)
                        or np.any(q < 0) or np.any(q >= box)):
                    raise ValueError('Invalid IC mode payload or units/frame')
                density += np.sum(mass[:, None] * np.exp(-1j * x @ k.T), axis=0)
                theta += np.sum(mass[:, None] * (1j * (v @ k.T) * math.sqrt(a)) * np.exp(-1j * q @ k.T), axis=0)
                total += float(mass.sum())
    if total <= 0 or not math.isfinite(total):
        raise ValueError('No finite matter mass for Fourier normalization')
    return {'density': density/total, 'velocity_divergence': theta/total}


def _compare_modes(actual, expected, tolerance):
    expected = np.asarray(expected, dtype=complex)
    actual = np.asarray(actual, dtype=complex)
    if expected.shape != actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('Expected complex Fourier amplitudes must align with measured modes')
    norm = float(np.linalg.norm(expected))
    if norm <= np.finfo(float).tiny:
        residual = float(np.linalg.norm(actual))
        return {'status': 'PASS' if residual <= tolerance else 'FAIL', 'absolute_l2_error': residual,
                'comparison': 'declared zero mode amplitude'}
    residual = float(np.linalg.norm(actual-expected)/norm)
    active = abs(expected) > norm*1e-10
    phase = float(np.max(np.abs(np.angle(actual[active]*expected[active].conj())), initial=0.))
    return {'status': 'PASS' if residual <= tolerance else 'FAIL', 'relative_l2_error': residual,
            'maximum_phase_error_radians': phase}


@directory_owner('ic')
def validate_ic_realization(ic, configuration, destination=None, *, memory_gib=1.,
                            modes=None, expected_density=None, expected_velocity=None,
                            maximum_mode_error=.05, maximum_bulk_km_s=1e-5,
                            maximum_barycenter_shift_cells=.1, parent=None,
                            refinement=None, input_kind=None):
    """Remeasure conservation, identities and optional declared physical targets.

    Missing density/velocity/parent/boundary evidence is PENDING, never PASS.
    A barycenter is measured in unwrapped displacements from native q, avoiding
    the discontinuity of an arithmetic mean across a periodic box boundary.
    """
    from zoom import audit_music_ic
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.optionxform = str
    cfg.read(configuration)
    kind = input_kind or ('parent' if cfg.getint('setup','levelmin') == cfg.getint('setup','levelmax') else 'zoom')
    for value, name in ((maximum_mode_error, 'mode tolerance'), (maximum_bulk_km_s, 'bulk tolerance'),
                        (maximum_barycenter_shift_cells, 'barycenter tolerance')):
        _positive(value, name)
    hashes = {str(Path(p).resolve()): file_sha256(p) for p in (ic, configuration)}
    audit = audit_music_ic(ic, configuration, memory_gib=memory_gib, input_kind=kind)
    displacement, momentum, total, velocity_energy = np.zeros(3), np.zeros(3), 0., 0.
    with h5py.File(ic) as source:
        header = source['Header'].attrs
        box, a = float(header['BoxSize']), float(header['Time'])
        if (header.get('Music_lagrangian_schema') != 1
                or not math.isclose(float(header.get('Music_lagrangian_unitlength_cm', np.nan)), KPC_CM, rel_tol=1e-12)):
            raise ValueError('Barycenter validation requires native original Lagrangian coordinates')
        bound = np.asarray(header.get('Music_max_abs_displacement_native', []), dtype=float)
        if bound.shape != (3,) or not np.isfinite(bound).all() or np.any(bound < 0) or np.any(bound >= box/4):
            raise ValueError('Missing or ambiguous unwrapped displacement bound')
        for t in (0, 1, 2):
            group = source.get('PartType'+str(t))
            if group is None:
                continue
            for first in range(0, len(group['ParticleIDs']), 16384):
                end = min(first+16384, len(group['ParticleIDs']))
                q = np.asarray(group['LagrangianCoordinates'][first:end])
                x = np.asarray(group['Coordinates'][first:end])
                v = np.asarray(group['Velocities'][first:end])*math.sqrt(a)
                mass = (np.asarray(group['Masses'][first:end]) if 'Masses' in group
                        else np.full(end-first, header['MassTable'][t]))
                if q.shape != x.shape or not np.isfinite(q).all() or np.any(q < 0) or np.any(q >= box):
                    raise ValueError('Invalid barycenter reference coordinates')
                psi = (x-q+box/2)%box-box/2
                if np.any(np.abs(psi) > bound + box*1e-12):
                    raise ValueError('Particle displacement exceeds native winding bound')
                displacement += np.sum(mass[:,None]*psi, axis=0)
                momentum += np.sum(mass[:,None]*v, axis=0)
                velocity_energy += float(np.sum(mass[:,None]*v*v))
                total += float(mass.sum())
        spacing = box/2**cfg.getint('setup','levelmin')
    bulk = momentum/total
    center = displacement/total
    checks = {'mass': {'status':'PASS', 'relative_error':audit['mass_closure_relative_error']},
              'exact_ids': {'status':'PASS', 'definition':'unique exact uint64 over all species and refinement levels'},
              'momentum': {'status':'PASS' if np.linalg.norm(bulk) <= maximum_bulk_km_s else 'FAIL',
                           'bulk_velocity_km_s':bulk.tolist(), 'maximum_bulk_km_s':maximum_bulk_km_s,
                           'rms_velocity_km_s':math.sqrt(velocity_energy/total)},
              'barycenter': {'status':'PASS' if np.linalg.norm(center)/spacing <= maximum_barycenter_shift_cells else 'FAIL',
                            'unwrapped_shift_kpc_h':center.tolist(),
                            'shift_parent_cells':float(np.linalg.norm(center)/spacing),
                            'maximum_shift_parent_cells':maximum_barycenter_shift_cells}}
    if modes is None and (expected_density is not None or expected_velocity is not None):
        raise ValueError('Expected Fourier amplitudes require explicit modes')
    measured = measure_ic_modes(ic, modes, memory_gib=memory_gib) if modes is not None else None
    for name, expected, field in (('density_modes',expected_density,'density'), ('velocity_modes',expected_velocity,'velocity_divergence')):
        checks[name] = {'status':'PENDING', 'reason':'No independent phase-matched mode prediction supplied'}
        if expected is not None:
            checks[name] = _compare_modes(measured[field], expected, maximum_mode_error)
    checks['parent_phase_match'] = {'status':'NOT_APPLICABLE' if kind == 'parent' else 'PENDING'}
    if parent is not None:
        from zoom import compare_density_phases
        checks['parent_phase_match'] = compare_density_phases(parent['ic'], ic, parent['noise'],
            parent['child_noise'], parent['destination'], memory_gib=memory_gib)
    checks['refinement_boundaries'] = {'status':'NOT_APPLICABLE' if kind == 'parent' else 'PENDING'}
    if refinement is not None:
        checks['refinement_boundaries'] = validate_refinement_continuity(**refinement)
    statuses = {value['status'] for value in checks.values()}
    status = 'FAIL' if 'FAIL' in statuses else 'INCOMPLETE' if 'PENDING' in statuses else 'PASS'
    _unchanged(hashes)
    report = {'schema':'ic-realization-validation-v1', 'status':status, 'checks':checks,
              'input_kind':kind, 'sources':hashes, 'algorithm_sha256':file_sha256(__file__),
              'epoch':epoch_contract(a), 'structural_audit':audit, 'science_acceptance':'NOT_ACCEPTED'}
    if destination is not None:
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError('Refusing to overwrite IC realization evidence')
        if measured is not None:
            from core.data import export_numeric_audit
            arrays = {'mode_vectors':_mode_vectors(modes).astype(np.int64),
                      'measured_density':measured['density'],
                      'measured_velocity_divergence':measured['velocity_divergence']}
            specs = {'mode_vectors':{'units':'dimensionless','quantity':'integer fundamental-box Fourier wavevectors'},
                     'measured_density':{'units':'dimensionless','quantity':'mass-normalized complex particle density modes'},
                     'measured_velocity_divergence':{'units':'km s^-1 (kpc/h)^-1','quantity':'bulk velocity divergence at original Lagrangian q'}}
            for name,value,units in (('expected_density',expected_density,'dimensionless'),
                                     ('expected_velocity_divergence',expected_velocity,'km s^-1 (kpc/h)^-1')):
                if value is not None:
                    arrays[name] = np.asarray(value,dtype=complex)
                    specs[name] = {'units':units,'quantity':'explicit independent phase-matched Fourier prediction'}
            return export_numeric_audit(destination,report,arrays,specs,algorithm=__file__,
                source_hashes=hashes,context={'epoch':epoch_contract(a),
                    'definition':'IC conservation and declared complex Fourier realization controls'})
        atomic_json(destination, report)
    return report


def validate_refinement_continuity(coarse_positions, coarse_vectors, fine_positions, fine_vectors,
                                   *, box, tolerance, coordinate_tolerance=1e-10):
    """Compare independently sampled coarse/fine vectors at matched boundary q.

    Positions must coincide modulo the declared periodic box. Matching by row
    without verifying q would allow a shifted fine-grid sample to pass.
    """
    box = _positive(box, 'continuity box')
    tolerance = _positive(tolerance, 'continuity tolerance')
    coordinate_tolerance = _positive(coordinate_tolerance, 'coordinate tolerance')
    q, f, x, g = [np.asarray(v, dtype=float) for v in (coarse_positions, coarse_vectors, fine_positions, fine_vectors)]
    if (q.ndim != 2 or q.shape[1] != 3 or not len(q) or any(v.shape != q.shape for v in (f,x,g))
            or any(not np.isfinite(v).all() for v in (q,f,x,g))):
        raise ValueError('Boundary samples require aligned finite (N,3) arrays')
    separation = (x-q+box/2)%box-box/2
    if np.max(np.abs(separation)) > coordinate_tolerance*box:
        raise ValueError('Coarse/fine boundary samples have different Lagrangian coordinates')
    error = float(np.max(np.linalg.norm(g-f,axis=1)))
    return {'status':'PASS' if error <= tolerance else 'FAIL', 'maximum_vector_jump':error,
            'maximum_allowed_vector_jump':tolerance, 'samples':len(q),
            'scope':'declared coincident boundary samples; does not infer unsampled continuity'}


@directory_owner('first')
def compare_ic_generators(first, second, modes, destination=None, *, memory_gib=1.,
                          maximum_mode_error=.02):
    """Compare delivered generator outputs in a shared box, epoch and phase frame.

    Generator IDs need not coincide. Each file must independently have exact,
    unique IDs. Comparisons are density and bulk divergence, not binned power.
    """
    _positive(maximum_mode_error, 'generator comparison tolerance')
    before = {str(Path(p).resolve()): file_sha256(p) for p in (first, second)}
    frames, populations = [], []
    for path in (first, second):
        with h5py.File(path) as source:
            header = source['Header'].attrs
            counts = snapshot_total_counts(header)
            if (header.get('NumFilesPerSnapshot') != 1 or np.any(counts[3:])
                    or not np.array_equal(counts, header['NumPart_ThisFile'])):
                raise ValueError('Generator comparison requires complete matter ICs')
            if 48*sum(map(int,counts))+64*1024**2 > _positive(memory_gib,'memory budget')*1024**3:
                raise MemoryError('Generator comparison identities exceed workspace')
            ids, mass = [], 0.
            species = []
            for t in (0,1,2):
                if not counts[t]:
                    continue
                group = source['PartType'+str(t)]
                values = np.asarray(group['ParticleIDs'])
                if values.dtype != np.uint64 or values.shape != (int(counts[t]),):
                    raise ValueError('Generator comparison requires exact uint64 IDs')
                ids.append(values)
                weights = np.asarray(group['Masses']) if 'Masses' in group else np.full(len(values), header['MassTable'][t])
                if weights.shape != values.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
                    raise ValueError('Invalid generator comparison masses')
                mass += float(weights.sum())
                species.append(t)
            if not ids or len(np.unique(np.concatenate(ids))) != sum(map(len,ids)):
                raise ValueError('Duplicate or empty generator comparison identities')
            populations.append((mass,species))
            frames.append([_positive(header[key],key) for key in ('BoxSize','Time','HubbleParam','Omega0')]
                          +[float(header.get('OmegaBaryon',0.))])
    if not np.allclose(frames[0],frames[1],rtol=1e-12,atol=0):
        raise ValueError('IC generators differ in units, epoch or cosmological frame')
    if populations[0][1] != populations[1][1] or not math.isclose(populations[0][0],populations[1][0],rel_tol=1e-10):
        raise ValueError('IC generators differ in total mass or species')
    measurements = [measure_ic_modes(path,modes,memory_gib=memory_gib) for path in (first,second)]
    checks = {field:_compare_modes(measurements[1][field],measurements[0][field],maximum_mode_error)
              for field in ('density','velocity_divergence')}
    _unchanged(before)
    report = {'schema':'ic-generator-comparison-v1', 'status':'PASS' if all(v['status']=='PASS' for v in checks.values()) else 'FAIL',
              'checks':checks, 'modes':_mode_vectors(modes).astype(int).tolist(), 'sources':before,
              'phase_contract':'complex modes compared directly; same saved noise must be supplied to generators',
              'science_acceptance':'NOT_ACCEPTED', 'algorithm_sha256':file_sha256(__file__)}
    if destination is not None:
        if Path(destination).exists():
            raise FileExistsError('Refusing to overwrite generator comparison evidence')
        from core.data import export_numeric_audit
        arrays = {'mode_vectors':_mode_vectors(modes).astype(np.int64)}
        specs = {'mode_vectors':{'units':'dimensionless','quantity':'integer fundamental-box Fourier wavevectors'}}
        for index,measurement in enumerate(measurements):
            for field,units in (('density','dimensionless'),('velocity_divergence','km s^-1 (kpc/h)^-1')):
                name = ('reference_' if index == 0 else 'candidate_')+field
                arrays[name] = measurement[field]
                specs[name] = {'units':units,'quantity':'direct complex particle Fourier mode'}
        return export_numeric_audit(destination,report,arrays,specs,algorithm=__file__,
            source_hashes=before,context={'definition':'phase-matched generator complex Fourier comparison'})
    return report

# ========================================================================
# ic_wdm: ID-keyed thermal relic prescription for mixed-resolution native MUSIC ICs.
from core.cosmology import wdm_parameters, wdm_thermal_velocity_from_ids, wdm_fd_speed_moments
from core.data import transactional_ic_transform


def _masses(group,header,t,first,end):
    weights = np.asarray(group['Masses'][first:end]) if 'Masses' in group else np.full(end-first,header['MassTable'][t])
    if weights.shape != (end-first,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Relic velocity prescription requires positive finite particle masses')
    return weights


def _thermal_pass(path,cfg,*,write=False,bulk=None):
    total,p = 0.,np.zeros(3)
    levels = {}
    with h5py.File(path,'r+' if write else 'r') as source:
        if not write and (source.attrs.get('ic_transform_operation') == 'native-wdm-thermal-v1'
                          or any('ThermalVelocities' in source.get('PartType'+str(t),{}) for t in (1,2))):
            raise ValueError('Native WDM thermal velocities already assigned')
        header = source['Header'].attrs
        a = float(header['Time'])
        om,ob = float(header['Omega0']),float(header['OmegaBaryon'])
        if not 0 < ob < om or not math.isfinite(a) or a <= 0:
            raise ValueError('Invalid native thermal relic epoch or species fractions')
        h = float(header['HubbleParam'])
        if (not math.isclose(h,float(cfg['cosmology']['H0'])/100,rel_tol=1e-12)
                or not math.isclose(ob,float(cfg['cosmology']['omega_b'])/h**2,rel_tol=1e-12)
                or not math.isclose(om-ob,float(cfg['cosmology']['omega_cdm'])/h**2,rel_tol=1e-12)):
            raise ValueError('Native relic sampling cosmology differs from IC header')
        counts = snapshot_total_counts(header)
        if header.get('NumFilesPerSnapshot') != 1 or not np.array_equal(counts,header['NumPart_ThisFile']) or np.any(counts[3:]):
            raise ValueError('Native thermal prescription requires a complete matter IC')
        for t in (1,2):
            group = source.get('PartType'+str(t))
            if group is None:
                continue
            ids = group['ParticleIDs']
            if ids.dtype != np.uint64 or len(ids) != int(counts[t]):
                raise ValueError('Native thermal prescription requires exact uint64 identities')
            fraction = 1. if t == 1 else (om-ob)/om
            velocity = group['Velocities']
            if velocity.shape != (len(ids),3) or velocity.dtype != np.float64:
                raise ValueError('Native thermal prescription requires double velocities')
            if write:
                thermal_ds = group.create_dataset('ThermalVelocities',shape=velocity.shape,dtype=np.float64,
                    chunks=True,fletcher32=True)
            for first in range(0,len(ids),16384):
                end = min(first+16384,len(ids))
                sample,momentum,beta = wdm_thermal_velocity_from_ids(ids[first:end],cfg,a)
                if np.max(beta,initial=0.) > float(cfg['wdm'].get('max_thermal_speed_fraction_c',.01)):
                    raise ValueError('Relic thermal speed exceeds the configured Newtonian limit')
                sample *= fraction
                weights = _masses(group,header,t,first,end)
                if not np.isfinite(velocity[first:end]).all():
                    raise ValueError('Nonfinite original bulk velocity')
                if write:
                    kick = sample-np.asarray(bulk)*fraction
                    thermal_ds[first:end] = kick
                    velocity[first:end] = velocity[first:end]+kick
                # Remove one common physical relic drift. The resulting
                # coarse velocity is the combined-matter COM of that relic.
                total += float(np.sum(weights)*fraction)
                p += np.sum(weights[:,None]*sample,axis=0)
                for mass in np.unique(weights):
                    key = str(t)+':'+format(float(mass),'.17g')
                    levels[key] = levels.get(key,0)+int(np.count_nonzero(weights == mass))
        if total <= 0 or not math.isfinite(total):
            raise ValueError('No finite represented relic mass')
        if write:
            source['Header'].attrs['WDM_thermal_schema'] = 1
            source['Header'].attrs['WDM_thermal_seed'] = np.uint64(cfg['simulation']['random_seed'])
            source['Header'].attrs['WDM_removed_relic_drift_gadget'] = bulk
    return {'bulk':(p/total).tolist(),'represented_relic_mass_native':total,'mass_level_counts':levels,'scale_factor':a}


def apply_native_wdm_thermal(ic,cfg):
    relic = wdm_parameters(cfg)
    if cfg['wdm']['model'] != 'wdm_thermal_relic' or cfg['wdm'].get('include_thermal_velocities',True) is not True:
        raise ValueError('Native WDM requires the explicitly enabled physical thermal relic prescription')
    seed = cfg['simulation'].get('random_seed')
    if type(seed) is not int or not 0 <= seed <= 2**64-1:
        raise ValueError('Declare an exact unsigned 64-bit thermal random seed')
    def transform(paths):
        if len(paths) != 1:
            raise ValueError('Native WDM currently requires one complete IC file')
        measured = _thermal_pass(paths[0],cfg)
        _thermal_pass(paths[0],cfg,write=True,bulk=measured['bulk'])
        return {'schema':'native-wdm-thermal-v1','status':'PASS','model':'wdm_thermal_relic',
                'random_seed':seed,'class_ncdm_mass_keV':float(cfg['wdm']['thermal_relic_mass_keV']),
                'class_ncdm_temperature_ratio':float(relic['T_ncdm']),
                'scale_factor':measured['scale_factor'],
                'removed_bulk_velocity_gadget_km_s':measured['bulk'],
                'mass_level_counts':measured['mass_level_counts'],
                'expected_speed_moments':wdm_fd_speed_moments(cfg,measured['scale_factor']),
                'coarse_prescription':'combined baryon/relic COM; relic fraction times ID-keyed physical draw',
                'coarse_phase_space':'UNRESOLVED; target aperture must exclude coarse particles',
                'science_acceptance':'NOT_ACCEPTED'}
    report = transactional_ic_transform([ic],transform,operation='native-wdm-thermal-v1',configuration=cfg)
    atomic_json(str(ic)+'.native_wdm.json',{'sampling':report,'ic_sha256':file_sha256(ic),
                'algorithm_sha256':file_sha256(__file__)})
    return verify_native_wdm_thermal(ic,cfg)


def verify_native_wdm_thermal(ic,cfg):
    """Recompute every kick from exact IDs; sidecar PASS is insufficient."""
    path = Path(ic)
    record = json.loads(Path(str(path)+'.native_wdm.json').read_text())
    if record.get('ic_sha256') != file_sha256(path) or record.get('algorithm_sha256') != file_sha256(__file__):
        raise ValueError('Native WDM IC or sampler implementation changed')
    report = record['sampling']
    relic = wdm_parameters(cfg)
    if (report.get('status') != 'PASS' or report.get('random_seed') != cfg['simulation']['random_seed']
            or not math.isclose(report.get('class_ncdm_mass_keV',math.nan),cfg['wdm']['thermal_relic_mass_keV'],rel_tol=1e-12)
            or not math.isclose(report.get('class_ncdm_temperature_ratio',math.nan),relic['T_ncdm'],rel_tol=1e-12)):
        raise ValueError('Native WDM thermal prescription mismatch')
    total,p = 0.,np.zeros(3)
    levels = {}
    with h5py.File(path) as source:
        header = source['Header'].attrs
        a,om,ob = [float(header[key]) for key in ('Time','Omega0','OmegaBaryon')]
        h = float(header['HubbleParam'])
        if (not math.isfinite(h) or h <= 0 or not 0 < ob < om
                or not math.isclose(h,float(cfg['cosmology']['H0'])/100,rel_tol=1e-12)
                or not math.isclose(ob,float(cfg['cosmology']['omega_b'])/h**2,rel_tol=1e-12)
                or not math.isclose(om-ob,float(cfg['cosmology']['omega_cdm'])/h**2,rel_tol=1e-12)):
            raise ValueError('Native thermal verification cosmology differs from IC header')
        if header.get('WDM_thermal_schema') != 1 or not math.isclose(a,report['scale_factor'],rel_tol=1e-12):
            raise ValueError('Native WDM epoch/schema mismatch')
        for t in (1,2):
            group = source.get('PartType'+str(t))
            if group is None:
                continue
            factor = 1. if t == 1 else (om-ob)/om
            thermal = group.get('ThermalVelocities')
            if thermal is None or thermal.shape != group['Velocities'].shape or thermal.dtype != np.float64:
                raise ValueError('Native WDM lacks aligned double thermal velocity vectors')
            for first in range(0,len(thermal),16384):
                end = min(first+16384,len(thermal))
                ids = group['ParticleIDs'][first:end]
                if ids.dtype != np.uint64:
                    raise ValueError('Thermal verification requires exact uint64 IDs')
                sample,_y,_beta = wdm_thermal_velocity_from_ids(ids,cfg,a)
                expected = sample*factor-np.asarray(report['removed_bulk_velocity_gadget_km_s'])*factor
                actual = thermal[first:end]
                if not np.array_equal(actual,expected):
                    raise ValueError('Delivered thermal velocity differs from its exact ID/seed prescription')
                mass = _masses(group,header,t,first,end)
                p += np.sum(actual*mass[:,None],axis=0)
                total += float(np.sum(mass))
                for mass_value in np.unique(mass):
                    key = str(t)+':'+format(float(mass_value),'.17g')
                    levels[key] = levels.get(key,0)+int(np.count_nonzero(mass == mass_value))
    if levels != report['mass_level_counts'] or total <= 0 or np.linalg.norm(p/total) > 1e-10:
        raise ValueError('Native WDM mass-level inventory or thermal momentum closure failed')
    if record['ic_sha256'] != file_sha256(path):
        raise ValueError('Native WDM IC changed during verification')
    return report
