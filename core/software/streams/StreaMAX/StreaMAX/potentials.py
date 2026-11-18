import jax
import jax.numpy as jnp
import numpy as _np

from .utils import get_mat

from .constants import G, EPSILON

# Gauss-Legendre nodes for the density-flat NFW Chandrasekhar 1D integral.
# After the substitution v = cos(theta) the integrand is a smooth rational
# function on [0, 1] (only a single sqrt), so 12 nodes give ~1e-7 accuracy
# on Phi and <1e-3 on the implied density across q in [0.5, 1.5]. The cost
# is computed once at import time. See NFW_density_flat_potential below.
_N_GL_DENSFLAT = 12
_x_gl, _w_gl = _np.polynomial.legendre.leggauss(_N_GL_DENSFLAT)
_V_GL_DENSFLAT = jnp.asarray(0.5 * (_x_gl + 1.0))   # nodes on [0, 1]
_W_GL_DENSFLAT = jnp.asarray(0.5 * _w_gl)            # weights on [0, 1]

# ---------- helpers ----------
@jax.jit
def _shift(x, y, z, p):
    return jnp.array([x - p['x_origin'], y - p['y_origin'], z - p['z_origin']])

@jax.jit
def _rotate(vec, p):
    if 'R_mat' in p:
        R = p['R_mat']  # already includes roll if prepare_params was used
    else:
        R = get_mat(p['dirx'], p['diry'], p['dirz'])
        if 'roll' in p:
            cos_r = jnp.cos(p['roll'])
            sin_r = jnp.sin(p['roll'])
            R_z = jnp.array([[cos_r, -sin_r, 0.0],
                              [sin_r,  cos_r, 0.0],
                              [0.0,    0.0,   1.0]])
            R = R @ R_z
    return R @ vec  # matvec

# ---------- Point Mass ----------
# Phi = - G M / r
@jax.jit
def PointMass_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    r  = _shift(x, y, z, params)
    s  = jnp.sqrt(r @ r + EPSILON)
    M  = params['M'] if 'M' in params else 10.**params['logM']
    return -G * M / s # kpc^2 / Gyr^2

@jax.jit
def PointMass_acceleration(x, y, z, params):
    def potential_vec(pos):
        return PointMass_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def PointMass_hessian(x, y, z, params):
    def potential_vec(pos):
        return PointMass_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- Isochrone ----------
# Phi = - G M / ( b + sqrt(r^2 + b^2) )
@jax.jit
def Isochrone_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'Rs', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    r = _shift(x, y, z, params)
    s  = jnp.sqrt(r @ r + params['Rs']*params['Rs'] + EPSILON)
    M  = params['M'] if 'M' in params else 10.**params['logM']
    return -G * M / (params['Rs'] + s)  # kpc^2 / Gyr^2

@jax.jit
def Isochrone_acceleration(x, y, z, params):
    def potential_vec(pos):
        return Isochrone_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def Isochrone_hessian(x, y, z, params):
    def potential_vec(pos):
        return Isochrone_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- Plummer ----------
# Phi = - G M / sqrt(r^2 + Rs^2)
@jax.jit
def Plummer_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'Rs', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    r = _shift(x, y, z, params)
    s = jnp.sqrt(r @ r + params['Rs']*params['Rs'] + EPSILON)
    M = params['M'] if 'M' in params else 10.**params['logM']
    return -G * M / s  # kpc^2 / Gyr^2

@jax.jit
def Plummer_acceleration(x, y, z, params):
    def potential_vec(pos):
        return Plummer_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def Plummer_hessian(x, y, z, params):
    def potential_vec(pos):
        return Plummer_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- (Triaxial) NFW ----------
# Phi = - G M / r * log(1 + r/Rs),  r = sqrt((rx/a)^2 + (ry/b)^2 + (rz/c)^2)
@jax.jit
def NFW_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'Rs', 'a', 'b', 'c', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin = _shift(x, y, z, params)
    rvec = _rotate(rin, params)  
    rx, ry, rz = rvec
    r = jnp.sqrt((rx/params['a'])**2 + (ry/params['b'])**2 + (rz/params['c'])**2 + EPSILON)
    M = params['M'] if 'M' in params else 10.**params['logM']
    return -G * M * jnp.log(1 + r / params['Rs']) / (r + EPSILON)  # kpc^2 / Gyr^2

@jax.jit
def NFW_acceleration(x, y, z, params):
    def potential_vec(pos):
        return NFW_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def NFW_hessian(x, y, z, params):
    def potential_vec(pos):
        return NFW_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi


# ---------- NFW with flattening applied to the DENSITY (not the potential) ----------
# rho(R, z) = rho_NFW(m),  m^2 = R^2 + z^2 / q^2,   q = params['c'] (a = b = 1).
# Phi is recovered via Chandrasekhar's homoeoidal 1D integral; positive density
# is guaranteed everywhere for any q > 0, unlike potential-flattening which
# breaks positivity outside q ~ [0.7, 1.3]. See StreaMAX_private tests for
# cross-validation against galpy and direct 3D Poisson integration.
#
# After the change of variables v = cos(theta):
#     Phi(R, z) = -(G M q^2 / R_s) * I(R, z, q)
#     I = integral_{v=0..1} dv / [(1 + m_tilde(v)) * D(v)]
#     D(v)       = q^2 + (1 - q^2) v^2
#     m_tilde(v) = v * sqrt(xi^2 / D(v) + zeta^2 / q^2)
#     xi = R / R_s,  zeta = z / R_s

@jax.jit
def NFW_density_flat_potential(x, y, z, params):
    '''
    Density-flattened axisymmetric NFW potential.

    params: dict with keys 'logM', 'Rs', 'c' (= q, the axis ratio),
    'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'.
    The 'a' and 'b' entries are ignored (axisymmetric model assumes a = b = 1).
    '''
    rin  = _shift(x, y, z, params)
    rvec = _rotate(rin, params)
    rx, ry, rz = rvec

    R = jnp.sqrt(rx * rx + ry * ry + EPSILON)
    Z = rz
    q  = params['c']
    Rs = params['Rs']
    M  = params['M'] if 'M' in params else 10.**params['logM']

    xi   = R / Rs
    zeta = Z / Rs

    v  = _V_GL_DENSFLAT
    q2 = q * q
    D  = q2 + (1.0 - q2) * v * v
    m  = v * jnp.sqrt(xi * xi / D + zeta * zeta / q2 + EPSILON)

    integrand = 1.0 / ((1.0 + m) * D)
    integral  = jnp.sum(_W_GL_DENSFLAT * integrand)

    return -(G * M * q2 / Rs) * integral  # kpc^2 / Gyr^2


@jax.jit
def NFW_density_flat_acceleration(x, y, z, params):
    '''
    Acceleration = -grad(NFW_density_flat_potential), computed analytically
    (not via autodiff) since the gradient integrand has the same structure
    as Phi's:
        dI/dxi   = -xi      * sum_v w v^2 / [(1+m)^2 D^2 m]
        dI/dzeta = -zeta/q2 * sum_v w v^2 / [(1+m)^2 D   m]
    Then rotated back to world coords by the same rotation as in Phi.
    About 30% faster than reverse-mode autodiff in vmap'd inner loops.
    '''
    rin  = _shift(x, y, z, params)
    if 'R_mat' in params:
        R_mat = params['R_mat']
    else:
        R_mat = get_mat(params['dirx'], params['diry'], params['dirz'])
        if 'roll' in params:
            cos_r = jnp.cos(params['roll'])
            sin_r = jnp.sin(params['roll'])
            R_z = jnp.array([[cos_r, -sin_r, 0.0],
                              [sin_r,  cos_r, 0.0],
                              [0.0,    0.0,   1.0]])
            R_mat = R_mat @ R_z
    rvec = R_mat @ rin
    rx, ry, rz = rvec

    R2 = rx * rx + ry * ry + EPSILON
    R  = jnp.sqrt(R2)
    Z  = rz
    q  = params['c']
    Rs = params['Rs']
    M  = params['M'] if 'M' in params else 10.**params['logM']

    xi   = R / Rs
    zeta = Z / Rs

    v   = _V_GL_DENSFLAT
    v2  = v * v
    q2  = q * q
    D   = q2 + (1.0 - q2) * v2
    A   = xi * xi / D + zeta * zeta / q2 + EPSILON
    m   = v * jnp.sqrt(A)

    base = v2 / ((1.0 + m) * (1.0 + m) * m + EPSILON)
    dI_dxi   = -xi          * jnp.sum(_W_GL_DENSFLAT * base / (D * D))
    dI_dzeta = -(zeta / q2) * jnp.sum(_W_GL_DENSFLAT * base / D)

    pref = -(G * M * q2 / Rs)
    dPhi_dR = pref * dI_dxi   / Rs
    dPhi_dZ = pref * dI_dzeta / Rs

    F_halo = jnp.array([
        -dPhi_dR * rx / R,
        -dPhi_dR * ry / R,
        -dPhi_dZ,
    ])
    return jnp.dot(R_mat.T, F_halo)  # kpc / Gyr^2


@jax.jit
def NFW_density_flat_hessian(x, y, z, params):
    '''Hessian via autodiff -- matches the style of every other potential in
    this module and is not in the per-particle inner loop, so the autodiff
    overhead is negligible.'''
    def potential_vec(pos):
        return NFW_density_flat_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi


# ---------- Miyamoto-Nagai Disk ----------
# Phi = - G M / sqrt(R^2 + (Rs + sqrt(z^2 + Hs^2))^2)
@jax.jit
def MiyamotoNagai_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'Rs', 'Hs', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin  = _shift(x, y, z, params)
    rvec = _rotate(rin, params)  
    rx, ry, rz = rvec + EPSILON

    R = (rx**2 + ry**2)**0.5

    denom2 = (R**2 + (params['Rs'] + (rz**2 + params['Hs']**2)**0.5)**2)

    M   = params['M'] if 'M' in params else 10.**params['logM']
    Phi = - G * M / (denom2**0.5)

    return Phi

@jax.jit
def MiyamotoNagai_acceleration(x, y, z, params):
    def potential_vec(pos):
        return MiyamotoNagai_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def MiyamotoNagai_hessian(x, y, z, params):
    def potential_vec(pos):
        return MiyamotoNagai_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- Quadratic Bar ----------
# Phi = A * Rs^3 * (R^2 / (Rs + R)^5) * cos(2*phi_bar) * exp(-|z|/Hs)
# where phi_bar = phi - Omega*(t - t0)
# and phi = arctan(y/x)
# and eps = 2*(t-t0)/(t1-t0) - 1
# and val = 0 if t<t0, 1 if t>t1, and 3/16*eps^5 - 5/8*eps^3 + 15/16*eps + 1/2 if t0<=t<=t1
@jax.jit
def Bar_potential(x, y, z, t, params):
    '''
    params: dict with keys 'A', 'Rs', 'Hs', 'Omega', 't0', 't1', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin  = _shift(x, y, z, params)
    rvec = _rotate(rin, params)
    rx, ry, rz = rvec + EPSILON

    R   = jnp.sqrt(rx**2 + ry**2)
    phi = jnp.arctan2(ry, rx)

    phi_bar = phi - params['Omega'] * (t - params['t0'])

    eps = 2.0*(t - params['t0'])/(params['t1'] - params['t0']) - 1.0
    val = jax.lax.cond(
        t < params['t0'],
        lambda _: 0.0,
        lambda _: jax.lax.cond(
            t > params['t1'],
            lambda _: 1.0,
            lambda _: (3.0/16.0)*eps**5 - (5.0/8.0)*eps**3 + (15.0/16.0)*eps + 0.5,
            operand=None
        ),
        operand=None
    )

    amp = params['A'] * params['Rs']**3 * (R**2 / (params['Rs'] + R)**5) * val

    return amp * jnp.cos(2*phi_bar) * (jnp.exp(-(rz/params['Hs'])**2))  # kpc^2 / Gyr^2

@jax.jit
def Bar_acceleration(x, y, z, t, params):
    def potential_vec(pos):
        return Bar_potential(pos[0], pos[1], pos[2], t, params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def Bar_hessian(x, y, z, t, params):
    def potential_vec(pos):
        return Bar_potential(pos[0], pos[1], pos[2], t, params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- Hernquist ----------
# Phi = - G M / (r + Rs)
@jax.jit
def Hernquist_potential(x, y, z, params):
    '''
    params: dict with keys 'logM', 'Rs', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    r = _shift(x, y, z, params)
    s = jnp.sqrt(r @ r + EPSILON)
    M = params['M'] if 'M' in params else 10.**params['logM']
    return -G * M / (s + params['Rs'])  # kpc^2 / Gyr^2

@jax.jit
def Hernquist_acceleration(x, y, z, params):
    def potential_vec(pos):
        return Hernquist_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def Hernquist_hessian(x, y, z, params):
    def potential_vec(pos):
        return Hernquist_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- Logarithmic ----------
# Phi = 0.5 * V0^2 * log(Rc^2 + x^2 + (y^2)/q1^2 + (z^2)/q2^2)
@jax.jit
def Logarithmic_potential(x, y, z, params):
    '''
    params: dict with keys 'V0', 'Rc', 'q1', 'q2', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin  = _shift(x, y, z, params)
    rvec = _rotate(rin, params)  
    rx, ry, rz = rvec + EPSILON

    denom = params['Rc']**2 + rx**2 + (ry**2)/(params['q1']**2) + (rz**2)/(params['q2']**2)

    return 0.5 * params['V0']**2 * jnp.log(denom)  # kpc^2 / Gyr^2

@jax.jit
def Logarithmic_acceleration(x, y, z, params):
    def potential_vec(pos):
        return Logarithmic_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def Logarithmic_hessian(x, y, z, params):
    def potential_vec(pos):
        return Logarithmic_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- ExpDisk ----------
# Phi = - 2 * pi * G * Sigma0 * Rs^2 * [I0(R/(2Rs)) * K1(R/(2Rs))] * exp(-|z|/Hs)
# where I0 and K1 are modified Bessel functions
@jax.jit
def ExpDisk_potential(x, y, z, params):
    '''
    params: dict with keys 'logSigma0', 'Rs', 'Hs', 'x_origin', 'y_origin', 'z_origin', 'dirx', 'diry', 'dirz'
    '''
    rin  = _shift(x, y, z, params)
    rvec = _rotate(rin, params)  
    rx, ry, rz = rvec + EPSILON

    R = jnp.sqrt(rx**2 + ry**2)

    from jax.scipy.special import i0, k1

    bess = i0(R / (2 * params['Rs'])) * k1(R / (2 * params['Rs']))

    Sigma0 = params['Sigma0'] if 'Sigma0' in params else 10.**params['logSigma0']
    amp = -2 * jnp.pi * G * Sigma0 * params['Rs']**2 * bess

    return amp * jnp.exp(-(rz / params['Hs'])**2)  # kpc^2 / Gyr^2

@jax.jit
def ExpDisk_acceleration(x, y, z, params):
    def potential_vec(pos):
        return ExpDisk_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def ExpDisk_hessian(x, y, z, params):
    def potential_vec(pos):
        return ExpDisk_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi

# ---------- NFW + MiyamotoNagai ----------
# Composite potential: Phi = Phi_NFW + Phi_MiyamotoNagai
@jax.jit
def NFW_MiyamotoNagai_potential(x, y, z, params):
    '''
    params: dict with keys 'NFW_params' and 'MN_params', each being a dict of parameters for the respective potentials
    '''
    phi_nfw = NFW_potential(x, y, z, params['NFW_params'])
    phi_mn  = MiyamotoNagai_potential(x, y, z, params['MN_params'])
    return phi_nfw + phi_mn

@jax.jit
def NFW_MiyamotoNagai_acceleration(x, y, z, params):
    def potential_vec(pos):
        return NFW_MiyamotoNagai_potential(pos[0], pos[1], pos[2], params)
    grad_phi = jax.grad(potential_vec)(jnp.array([x, y, z]))
    return -grad_phi

@jax.jit
def NFW_MiyamotoNagai_hessian(x, y, z, params):
    def potential_vec(pos):
        return NFW_MiyamotoNagai_potential(pos[0], pos[1], pos[2], params)
    hess_phi = jax.hessian(potential_vec)(jnp.array([x, y, z]))
    return hess_phi