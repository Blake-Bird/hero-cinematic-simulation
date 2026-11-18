"""Remaining observation estimator sections awaiting ordinary module migration.

Sections retain independent Python namespaces through core.__init__.
The ordinary campaign/intake owner is cosmography.py; public commands import it.
"""

raise ImportError('Import an established core.<section> namespace or use commands.py')























# @research2-module hubble_observers
"""Block 41: expansion truth, trajectory lightcones and blinded local Hubble fits.

Reference: Wojtak et al. (2014), arXiv:1312.0276. Distances are Mpc without h,
peculiar velocities physical km/s. Curvature convention is Omega_k > 0 open.
No assumed LCDM growth or expansion law is substituted for solver backgrounds.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from .observation_data import array, exact_ids, observed_catalogue, read_observations, gls, freeze_result, gaussian_observable, C_KMS, HUBBLE_GYR, fingerprint


@dataclass(frozen=True)
class ExpansionHistory:
    redshift: np.ndarray
    hubble_kms_mpc: np.ndarray
    omega_k: float
    time_at_highest_redshift_gyr: float | None = None
    source_hash: str = ''

    def __post_init__(self):
        z = array(self.redshift, 'background redshift', 1)
        h = array(self.hubble_kms_mpc, 'H(z)', 1, positive=True)
        if len(z) < 4 or h.shape != z.shape or z[0] != 0 or np.any(np.diff(z) <= 0):
            raise ValueError('Background must cover z=0 with >=4 strictly increasing samples')
        if not np.isfinite(self.omega_k):
            raise ValueError('Curvature must be finite')
        z, h = z.copy(), h.copy()
        z.flags.writeable = h.flags.writeable = False
        object.__setattr__(self, 'redshift', z)
        object.__setattr__(self, 'hubble_kms_mpc', h)
        if self.time_at_highest_redshift_gyr is not None and self.time_at_highest_redshift_gyr < 0:
            raise ValueError('High-z time anchor must be nonnegative')

    def H(self, z):
        z = array(z, 'redshift')
        if np.any(z < 0) or np.any(z > self.redshift[-1]):
            raise ValueError('Requested epoch outside solver background coverage')
        return PchipInterpolator(self.redshift, self.hubble_kms_mpc, extrapolate=False)(z)

    def chi(self, z):
        z = array(z, 'redshift')
        self.H(z)
        return np.array([quad(lambda zz: C_KMS / float(self.H(zz)), 0., zz,
                              epsabs=1e-7, epsrel=1e-9)[0] for zz in z.ravel()]).reshape(z.shape)

    def transverse_distance(self, chi):
        chi = array(chi, 'radial comoving distance')
        if abs(self.omega_k) < 1e-12:
            return chi
        radius = C_KMS / self.hubble_kms_mpc[0] / np.sqrt(abs(self.omega_k))
        return radius * (np.sinh(chi / radius) if self.omega_k > 0 else np.sin(chi / radius))

    def angular_distance(self, z1, z2):
        if np.any(np.asarray(z2) <= np.asarray(z1)):
            raise ValueError('Angular-distance endpoints must be ordered')
        return self.transverse_distance(self.chi(z2) - self.chi(z1)) / (1 + np.asarray(z2))

    def luminosity_distance(self, z):
        return (1 + np.asarray(z)) * self.transverse_distance(self.chi(z))

    def lookback_gyr(self, z):
        z = array(z, 'redshift')
        self.H(z)
        return np.array([quad(lambda zz: HUBBLE_GYR / ((1 + zz) * float(self.H(zz))),
                              0., zz, epsabs=1e-10, epsrel=1e-9)[0]
                         for zz in z.ravel()]).reshape(z.shape)

    def cosmic_time_gyr(self, z):
        if self.time_at_highest_redshift_gyr is None:
            raise ValueError('Absolute cosmic time needs a solver high-redshift time anchor')
        return (self.time_at_highest_redshift_gyr + self.lookback_gyr(self.redshift[-1])
                - self.lookback_gyr(z))

    @classmethod
    def from_class(cls, path, *, omega_k):
        try:
            from .cosmology import class_background_table
            from .foundation import file_sha256
        except ImportError:
            from core.cosmology import class_background_table
            from core.foundation import file_sha256
        columns, data = class_background_table(path)
        zkey = next(k for k in columns if k == 'z')
        order = np.argsort(data[:, columns[zkey]])
        z = data[order, columns[zkey]]
        hz = data[order, columns['H']] * C_KMS  # CLASS H is 1/Mpc.
        timekey = next((k for k in columns if k.startswith('proper time')), None)
        anchor = None if timekey is None else float(data[order[-1], columns[timekey]])
        return cls(z, hz, omega_k, anchor, file_sha256(path))


def export_background_expansion(history):
    z = history.redshift
    result = {'schema': 'expansion-truth-v1', 'role': 'privileged_expansion_truth',
              'redshift': z.tolist(), 'H_kms_mpc': history.H(z).tolist(),
              'H0_kms_mpc': float(history.H(0)), 'chi_mpc': history.chi(z).tolist(),
              'lookback_gyr': history.lookback_gyr(z).tolist(),
              'luminosity_distance_mpc': history.luminosity_distance(z).tolist(),
              'omega_k': history.omega_k, 'source_hash': history.source_hash,
              'units': {'distance': 'physical Mpc without h', 'H': 'km/s/Mpc', 'time': 'Gyr'},
              'equations': ['chi=c integral dz/H', 'dt=da/(aH)']}
    if history.time_at_highest_redshift_gyr is not None:
        result['cosmic_time_gyr'] = history.cosmic_time_gyr(z).tolist()
    return result


@dataclass(frozen=True)
class Observer:
    observer_id: str
    position_mpc: np.ndarray
    velocity_kms: np.ndarray
    environment: str
    selection_rule: dict

    def __post_init__(self):
        for name in ('position_mpc', 'velocity_kms'):
            value = array(getattr(self, name), name, 1)
            if value.shape != (3,):
                raise ValueError('Observer position/velocity require three components')
            object.__setattr__(self, name, value.copy())
        if np.linalg.norm(self.velocity_kms) >= C_KMS:
            raise ValueError('Observer peculiar velocity is superluminal')


def choose_recorded_observers(catalogues, selections, *, seed):
    """Select exact halo/galaxy/void IDs or seeded random volume controls."""
    rng = np.random.default_rng(seed)
    observers = []
    for rule in selections:
        kind = rule['kind']
        if kind == 'volume_control':
            box = array(rule['box_mpc'], 'box', 1, positive=True)
            if box.shape != (3,):
                raise ValueError('Control box requires three lengths')
            for i in range(int(rule['count'])):
                observers.append(Observer(f'control:{seed}:{i}', rng.uniform(0, box),
                    np.zeros(3), kind, dict(rule, seed=seed)))
            continue
        table = catalogues[kind]
        ids = exact_ids(table['object_id'])
        mask = np.ones(len(ids), dtype=bool)
        for field, limits in rule.get('cuts', {}).items():
            v = array(table[field], field, 1)
            mask &= (v >= limits[0]) & (v <= limits[1])
        if 'ids' in rule:
            mask &= np.isin(ids.astype(str), np.asarray(rule['ids']).astype(str))
        selected = np.flatnonzero(mask)
        count = int(rule.get('count', len(selected)))
        if count <= 0 or count > len(selected):
            raise ValueError('Observer selection has insufficient candidates')
        for row in np.sort(rng.choice(selected, count, replace=False)):
            observers.append(Observer(f'{kind}:{ids[row]}', np.asarray(table['position_mpc'][row]),
                np.asarray(table['velocity_kms'][row]), kind, dict(rule, seed=seed)))
    if len({o.observer_id for o in observers}) != len(observers):
        raise ValueError('Duplicate observer IDs')
    return observers


def doppler_factor(velocity, direction):
    velocity = array(velocity, 'peculiar velocity')
    beta2 = np.sum(velocity**2, axis=-1) / C_KMS**2
    if np.any(beta2 >= 1):
        raise ValueError('Peculiar velocity must be subluminal')
    return (1 + np.sum(velocity * direction, axis=-1) / C_KMS) / np.sqrt(1 - beta2)


def aberrate_direction(direction, observer_velocity):
    """Lorentz transform source sightline into the moving observer rest frame."""
    n = array(direction, 'sightline')
    beta = np.asarray(observer_velocity) / C_KMS
    b2 = float(beta @ beta)
    if b2 == 0:
        return n.copy()
    gamma = 1 / np.sqrt(1 - b2)
    dot = n @ beta
    # Incoming photon direction is -n; convert transformed momentum back to sightline.
    shifted = (n + (((gamma - 1) * dot / b2 + gamma)[..., None]) * beta)
    shifted /= (gamma * (1 + dot))[..., None]
    return shifted / np.linalg.norm(shifted, axis=-1)[..., None]


def generate_past_lightcone_observations(history, observer, trajectories, *, depth_mpc,
        z_range, seed, log_distance_sigma, redshift_sigma, box_mpc=None,
        sky_mask=None, propagation=None, max_interpolation_gap_redshift=None):
    """Intersect each exact object's unwrapped trajectory with the observer lightcone.

    Input position shape (epochs,objects,3), sorted increasing redshift, comoving
    Mpc without h. Paths must be unwrapped before delivery. A periodic box uses a
    single minimum image and depth <= half the smallest side: no replica supplies
    fictitious independent volume. Propagation may supply measured magnification
    and potential/ISW redshift terms keyed by exact IDs, never an invented correction.
    """
    z = array(trajectories['redshift'], 'trajectory epochs', 1)
    ids = exact_ids(trajectories['object_id'])
    p = array(trajectories['position_mpc'], 'trajectories', 3)
    v = array(trajectories['velocity_kms'], 'peculiar velocities', 3)
    if p.shape != (len(z), len(ids), 3) or v.shape != p.shape or np.any(np.diff(z) <= 0):
        raise ValueError('Trajectory shape or epoch ordering is invalid')
    if log_distance_sigma <= 0 or redshift_sigma < 0 or depth_mpc <= 0:
        raise ValueError('Distance error/depth positive; redshift error nonnegative')
    if z_range[0] < z[0] or z_range[1] > z[-1]:
        raise ValueError('Lightcone request exceeds trajectory coverage')
    history.H(z)
    box = None if box_mpc is None else array(box_mpc, 'periodic box', 1, positive=True)
    if box is not None and (box.shape != (3,) or depth_mpc > min(box) / 2):
        raise ValueError('Independent lightcone depth exceeds periodic minimum-image volume')
    rng = np.random.default_rng(seed)
    rows, missed = [], []
    for j, ident in enumerate(ids):
        delta = p[:, j] - observer.position_mpc
        if box is not None:
            # Keep the same lattice image for the entire unwrapped trajectory.
            delta -= np.rint(delta[0] / box) * box
        pos = PchipInterpolator(z, delta, axis=0, extrapolate=False)
        vel = PchipInterpolator(z, v[:, j], axis=0, extrapolate=False)
        f = lambda zz: float(np.linalg.norm(pos(zz)) - history.chi(zz))
        brackets = [(a, b) for a, b in zip(z[:-1], z[1:])
                    if b >= z_range[0] and a <= z_range[1] and f(a) * f(b) <= 0]
        if len(brackets) != 1:
            missed.append({'object_id': str(ident), 'reason': 'NO_CROSSING' if not brackets else 'MULTIPLE_CROSSINGS'})
            continue
        a, b = brackets[0]
        if max_interpolation_gap_redshift is not None and b - a > max_interpolation_gap_redshift:
            missed.append({'object_id': str(ident), 'reason': 'OUTPUT_CADENCE_UNRESOLVED', 'bracket': [a, b]})
            continue
        zz = brentq(f, a, b, xtol=1e-11)
        dist = np.linalg.norm(pos(zz))
        if not z_range[0] <= zz <= z_range[1] or not 0 < dist <= depth_mpc:
            continue
        n = pos(zz) / dist
        ns = aberrate_direction(n, observer.velocity_kms)
        ra = np.degrees(np.arctan2(ns[1], ns[0])) % 360
        dec = np.degrees(np.arcsin(ns[2]))
        if sky_mask is not None and not sky_mask(ra, dec):
            continue
        ds, do = float(doppler_factor(vel(zz), n)), float(doppler_factor(observer.velocity_kms, n))
        terms = (propagation or {}).get(str(ident), {})
        allowed = {'magnification', 'potential_redshift', 'integrated_redshift'}
        if set(terms) - allowed:
            raise ValueError('Unsupported propagation terms')
        magnification = float(terms.get('magnification', 1.))
        if magnification <= 0:
            raise ValueError('Weak-lensing magnification must be positive')
        g = (1 + terms.get('potential_redshift', 0.)) * (1 + terms.get('integrated_redshift', 0.))
        zobs = (1 + zz) * ds / do * g - 1
        # Peculiar-motion luminosity distance: Ds^2/Do (source bolometric beaming,
        # observer aberration); supported gravitational terms enter reciprocity.
        dl = float(history.luminosity_distance(zz)) * ds**2 / do * g**2 / np.sqrt(magnification)
        rows.append((str(ident), zobs + rng.normal(0, redshift_sigma),
                     dl * np.exp(rng.normal(0, log_distance_sigma)), ra, dec, a, b))
    if not rows:
        raise ValueError('Physical observer selection is empty')
    columns = dict(zip(('object_id', 'redshift', 'distance_mpc', 'ra_deg', 'dec_deg',
                        'bracket_z_low', 'bracket_z_high'), map(list, zip(*rows))))
    columns['log_distance_sigma'] = [log_distance_sigma] * len(rows)
    columns['redshift_sigma'] = [redshift_sigma] * len(rows)
    units = {k: ('Mpc' if k == 'distance_mpc' else 'degree' if k.endswith('_deg') else '1') for k in columns}
    product = observed_catalogue(columns, units=units, namespace=trajectories['namespace'],
        selection={'observer_id': observer.observer_id, 'environment': observer.environment,
            'observer_rule': observer.selection_rule, 'depth_mpc': depth_mpc, 'redshift_range': z_range,
            'seed': seed, 'excluded': missed, 'finite_volume': 'single independent minimum image',
            'propagation_terms': sorted({k for item in (propagation or {}).values() for k in item})})
    return product


def cosmographic_distance_shape(z, q0, j0=1.):
    """c z[1+(1-q0)z/2-(1-q0-3q0²+j0)z²/6], independent of H0."""
    z = array(z, 'redshift', 1, positive=True)
    shape = C_KMS * z * (1 + (1-q0)*z/2 - (1-q0-3*q0*q0+j0)*z*z/6)
    if np.any(shape <= 0):
        raise ValueError('Cosmographic distance shape is outside its declared domain')
    return shape


def fit_distance_redshift_relation(observations, nuisance_model, *, realization_id=None):
    obs = read_observations(observations)
    t = obs['columns']
    z = array(t['redshift'], 'observed redshift', 1)
    d = array(t['distance_mpc'], 'observed distance', 1, positive=True)
    lo, hi = nuisance_model['redshift_range']
    mask = (z >= lo) & (z <= hi)
    if lo <= 0 or hi > nuisance_model['maximum_cosmographic_redshift']:
        raise ValueError('Fit exceeds the explicitly supported cosmographic domain')
    z, d = z[mask], d[mask]
    if len(z) < 2:
        raise ValueError('Local Hubble sample has fewer than two objects')
    # Flow corrections are observations/models supplied independently of simulation truth.
    if 'flow_velocity_kms' in t:
        correction = np.asarray(t['flow_velocity_kms'])[mask] / C_KMS
        z = (1+z)/(1+correction)-1
    shape = cosmographic_distance_shape(z, nuisance_model['q0'], nuisance_model.get('j0', 1.))
    y = np.log(shape / d)
    sd = array(t['log_distance_sigma'], 'log-distance errors', 1, positive=True)[mask]
    sz = array(t['redshift_sigma'], 'redshift errors', 1)[mask]
    # Include independent residual flow variance and observational redshift uncertainty.
    dz = np.minimum(z*.25, np.maximum(1e-7, z*1e-5))
    deriv = (np.log(cosmographic_distance_shape(z+dz, nuisance_model['q0'], nuisance_model.get('j0',1.)))
             - np.log(cosmographic_distance_shape(z-dz, nuisance_model['q0'], nuisance_model.get('j0',1.))))/(2*dz)
    variance = sd**2 + deriv**2 * (sz**2 + (nuisance_model.get('flow_dispersion_kms',0.)/C_KMS)**2)
    c = np.diag(variance)
    if 'log_distance_covariance' in obs['calibration']:
        cc = np.asarray(obs['calibration']['log_distance_covariance'])
        c += cc[np.ix_(mask,mask)]
    fit = gls(np.ones((len(y),1)), y, c, parameter_names=['ln_H0'])
    h0 = float(np.exp(fit['mean'][0]))
    observable = gaussian_observable(h0, h0*h0*fit['covariance'][0,0], 'km/s/Mpc')
    # Exact transformed interval avoids a symmetric approximation for wide posteriors.
    s = np.sqrt(fit['covariance'][0,0])
    observable['interval_68'] = [float(np.exp(fit['mean'][0]-s)),float(np.exp(fit['mean'][0]+s))]
    return freeze_result('local_observer', fit, observables={'H0': observable},
        selection=obs['selection'], object_ids=np.asarray(t['object_id'])[mask],
        namespace=obs['namespace'], nuisance=nuisance_model, realization_id=realization_id,
        environment={'kind': obs['selection'].get('environment')})
# @research2-end-module hubble_observers


























# @research2-module time_delay_cosmography
"""Block 43: simulated mass planes, lenstronomy rendering and frozen cosmography.

Particle lens potentials use isolated softened projected masses, never a
periodic cosmological Poisson field. Multi-plane ray tracing/time delays are
provided by the pinned lenstronomy backend. Fits jointly include image data,
time delays, optional kinematics and declared line-of-sight priors.
Reference: H0LiCOW XIII, arXiv:1907.04869; lenstronomy upstream license retained.
"""

import numpy as np
from scipy.integrate import quad
from .observation_data import array, covariance, read_observations, freeze_result, gaussian_observable, activate_pinned_backend, C_KMS, MPC_KM, ARCSEC_RAD, canonical, fingerprint, exact_ids

try:
    from .foundation import G_MPC_KMS2_MSUN
except ImportError:
    from core.foundation import G_MPC_KMS2_MSUN
DAY_SECONDS=86400.


def time_delay_distance(history,z_lens,z_source):
    dl=history.angular_distance(0.,z_lens)
    ds=history.angular_distance(0.,z_source)
    dls=history.angular_distance(z_lens,z_source)
    return float((1+z_lens)*dl*ds/dls)


def fermat_delay_days(d_dt_mpc, delta_phi_arcsec2):
    return np.asarray(delta_phi_arcsec2)*ARCSEC_RAD**2*d_dt_mpc*MPC_KM/C_KMS/DAY_SECONDS


def _particle_plane(history, plane, z_source, grid):
    z=float(plane['redshift'])
    if not 0<z<z_source:
        raise ValueError('Each mass plane must lie between observer and source')
    xy=array(plane['proper_xy_mpc'],'projected physical coordinates',2)
    mass=array(plane['mass_msun'],'projected masses',1,positive=True)
    exact_ids(plane['particle_id'],'particle_id')
    if xy.shape!=(len(mass),2) or len(plane['particle_id'])!=len(mass):
        raise ValueError('Projected particle shapes differ')
    epsilon=array(plane['softening_mpc'],'proper softening',positive=True)
    epsilon=np.broadcast_to(epsilon,mass.shape)
    dl=float(history.angular_distance(0.,z));ds=float(history.angular_distance(0.,z_source))
    dls=float(history.angular_distance(z,z_source))
    angular=xy/dl/ARCSEC_RAD; soft=epsilon/dl/ARCSEC_RAD
    strength=4*G_MPC_KMS2_MSUN*mass/C_KMS**2*dls/(dl*ds)/ARCSEC_RAD**2
    axis=array(grid,'angular grid',1)
    if len(axis)<4 or np.any(np.diff(axis)<=0):
        raise ValueError('Mass-plane interpolation needs an increasing >=4-point angular grid')
    xx,yy=np.meshgrid(axis,axis,indexing='xy')
    fields={key:np.zeros_like(xx) for key in ('f_','f_x','f_y','f_xx','f_yy','f_xy')}
    # Bounded-memory direct isolated quadrature. Each particle contributes
    # 0.5 theta_E² log(r²+epsilon²), with consistent analytic derivatives.
    for (cx,cy),eps,s in zip(angular,soft,strength):
        dx,dy=xx-cx,yy-cy; r2=dx*dx+dy*dy+eps*eps
        fields['f_']+=.5*s*np.log(r2)
        fields['f_x']+=s*dx/r2;fields['f_y']+=s*dy/r2
        fields['f_xx']+=s*(1/r2-2*dx*dx/r2**2)
        fields['f_yy']+=s*(1/r2-2*dy*dy/r2**2)
        fields['f_xy']-=2*s*dx*dy/r2**2
    return {'redshift':z,'component':plane['component'],'model':'INTERPOL',
        'kwargs':dict(grid_interp_x=axis,grid_interp_y=axis,**fields),
        'particle_count':len(mass),'mass_msun':float(mass.sum()),
        'minimum_softening_arcsec':float(min(soft)),
        'maximum_softening_arcsec':float(max(soft)),
        'grid_step_arcsec':float(max(np.diff(axis))),
        'source_identity':{'namespace':plane['namespace'],'particle_ids_hash':fingerprint(plane['particle_id'])}}


def construct_mass_planes(history, simulated_components, *, z_source, grid_arcsec):
    if not simulated_components:
        raise ValueError('Lens requires at least one physical mass plane')
    planes=[_particle_plane(history,p,z_source,grid_arcsec) for p in simulated_components]
    planes.sort(key=lambda p:p['redshift'])
    return {'role':'lens_mass_truth','z_source':float(z_source), 'planes':planes,
        'distance_convention':'proper Mpc without h; angular potential arcsec^2',
        'domain':'isolated projected softened particles; source-redshift-normalized deflections',
        'resolution_status':'UNRESOLVED_NOT_EXECUTED'}


def _history_astropy(history):
    """Astropy FLRW wrapper delegates E(z) to the actual tabulated solver history."""
    from astropy.cosmology import FLRW
    class TabulatedFLRW(FLRW):
        def efunc(self,z):
            return history.H(z)/history.hubble_kms_mpc[0]
        def inv_efunc(self,z):
            return 1/self.efunc(z)
        def w(self,z):
            raise NotImplementedError('No effective EOS inferred from the background table')
    # FLRW derives Ok0=1-Om0-Ode0-Ogamma0-Onu0. Parameters here encode curvature
    # only; E(z) is entirely delegated to the history above.
    return TabulatedFLRW(H0=history.hubble_kms_mpc[0],Om0=0.,Ode0=1-history.omega_k,Tcmb0=0.)


def render_lensed_observations(history, mass_truth, source, instrument, *, seed,
                               z_lens, kinematics=None, los_constraints=None):
    _,pin=activate_pinned_backend('lenstronomy')
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.PointSource.point_source import PointSource
    rng=np.random.default_rng(seed)
    planes=mass_truth['planes'];zs=mass_truth['z_source']
    lens=LensModel([p['model'] for p in planes],lens_redshift_list=[p['redshift'] for p in planes],
        z_lens=z_lens,z_source=zs,multi_plane=len({p['redshift'] for p in planes})>1,
        cosmo=_history_astropy(history))
    kwargs_lens=[p['kwargs'] for p in planes]
    solver=LensEquationSolver(lens)
    bx,by=source['position_arcsec']
    x,y=solver.image_position_from_source(bx,by,kwargs_lens,
        min_distance=instrument['image_search_step_arcsec'],search_window=instrument['field_arcsec'])
    if len(x)<2:
        raise ValueError('Selected lens/source produces fewer than two time-delay images')
    # Deterministic ordering; all differences are measured relative to image zero.
    order=np.argsort(np.arctan2(y,x));x,y=x[order],y[order]
    delays=lens.arrival_time(x,y,kwargs_lens,x_source=bx,y_source=by)
    dtdiff=delays[1:]-delays[0]
    dtcov=covariance(instrument['time_delay_covariance_days2'],len(dtdiff))
    observed_delays=dtdiff+rng.multivariate_normal(np.zeros(len(dtdiff)),dtcov)
    n=int(instrument['pixels']); pix=float(instrument['pixel_scale_arcsec'])
    if n<4 or pix<=0:
        raise ValueError('Instrument requires positive pixel scale and >=4 pixels')
    sigma=float(instrument['background_sigma']);exposure=float(instrument['exposure_seconds'])
    if sigma<=0 or exposure<=0:
        raise ValueError('Noise and exposure must be positive')
    data_kwargs={'image_data':np.zeros((n,n)), 'background_rms':sigma,'exposure_time':exposure,
                 'ra_at_xy_0':-(n-1)*pix/2,'dec_at_xy_0':-(n-1)*pix/2,
                 'transform_pix2angle':np.diag([pix,pix])}
    data=ImageData(**data_kwargs); psf=PSF(**instrument['psf'])
    light=LightModel(source['light_model_list'])
    point=PointSource(point_source_type_list=['LENSED_POSITION'],lens_model=lens)
    magnification=np.abs(lens.magnification(x,y,kwargs_lens))
    kwargs_ps=[{'ra_image':x,'dec_image':y,'point_amp':magnification*source['point_flux']}]
    image_model=ImageModel(data,psf,lens_model_class=lens,source_model_class=light,
                           point_source_class=point,kwargs_numerics=instrument.get('numerics',{}))
    image=image_model.image(kwargs_lens=kwargs_lens,kwargs_source=source['light_kwargs'],kwargs_ps=kwargs_ps)
    # Native lenstronomy pixel variance: background^2 + positive model/exposure.
    variance=sigma**2+np.maximum(image,0)/exposure
    data_kwargs['image_data']=image+rng.normal(size=image.shape)*np.sqrt(variance)
    data_kwargs['noise_map']=np.sqrt(variance)
    joint={'multi_band_list':[[data_kwargs,instrument['psf'],instrument.get('numerics',{})]],
           'multi_band_type':'single-band','time_delays_measured':observed_delays,
           'time_delays_uncertainties':np.sqrt(np.diag(dtcov))}
    if np.any(np.abs(dtcov-np.diag(np.diag(dtcov)))>1e-12):
        # lenstronomy's built-in scalar-delay likelihood cannot consume this
        # covariance. Preserve it for the explicitly configured custom likelihood.
        joint['time_delay_covariance_days2']=dtcov
    if kinematics is not None:
        joint['kinematic_observations']=kinematics
    return canonical({'schema':'lens-observations-v1','role':'estimator_observations',
        'namespace':source['namespace'],'object_id':source['object_id'],
        'z_lens':z_lens,'z_source':zs,'joint_data':joint,
        'image_positions_arcsec':np.stack([x,y],axis=-1)+rng.normal(0,instrument['astrometric_sigma_arcsec'],size=(len(x),2)),
        'astrometric_sigma_arcsec':instrument['astrometric_sigma_arcsec'],
        'los_constraints':los_constraints or {},
        'selection':{'seed':seed,'instrument':instrument,'image_count':len(x)},
        'backend_revision':pin['commit']})


def _ddt_h0_coefficient(zl,zs,expansion_shape):
    """Compute H0*D_dt using E(z) independent of H0; preserve shape assumptions."""
    from scipy.interpolate import PchipInterpolator
    z=array(expansion_shape['redshift'],'fitting shape redshift',1)
    e=array(expansion_shape['E'],'dimensionless E',1,positive=True)
    if z.shape!=e.shape or z[0]!=0 or np.any(np.diff(z)<=0) or abs(e[0]-1)>1e-10 or zs>z[-1]:
        raise ValueError('Fitting E(z) must be normalized and cover the lens/source')
    f=PchipInterpolator(z,e,extrapolate=False)
    chi=lambda zz:quad(lambda t:1/float(f(t)),0,zz,epsrel=1e-9)[0]
    ok=float(expansion_shape['omega_k'])
    def sk(x):
        if abs(ok)<1e-12:return x
        return (np.sinh(np.sqrt(ok)*x)/np.sqrt(ok) if ok>0
                else np.sin(np.sqrt(-ok)*x)/np.sqrt(-ok))
    return C_KMS*sk(chi(zl))*sk(chi(zs))/sk(chi(zs)-chi(zl))


def fit_lenstronomy_model(observations, declared_model_family, *, realization_id=None):
    obs=read_observations(observations)
    _,pin=activate_pinned_backend('lenstronomy')
    from lenstronomy.Workflow.fitting_sequence import FittingSequence
    import copy
    spec=copy.deepcopy(declared_model_family); joint=copy.deepcopy(obs['joint_data'])
    for band in joint['multi_band_list']:
        for key in ('image_data','noise_map','transform_pix2angle'):
            if key in band[0]:band[0][key]=np.asarray(band[0][key])
        if 'kernel_point_source' in band[1]:band[1]['kernel_point_source']=np.asarray(band[1]['kernel_point_source'])
    constraints=spec['constraints']; likelihood=spec['likelihood'];params=spec['parameters']
    if 'image_positions_arcsec' in obs:
        xy=np.asarray(obs['image_positions_arcsec']);sigma=obs['astrometric_sigma_arcsec']
        if sigma<=0 or xy.ndim!=2 or xy.shape[1]!=2:
            raise ValueError('Lens astrometry requires (N,2) positions and positive uncertainty')
        joint['ra_image_list']=[xy[:,0]];joint['dec_image_list']=[xy[:,1]]
        likelihood.setdefault('image_position_uncertainty',sigma)
        if 'point_source_model' in params:
            params['point_source_model'][0][0]['ra_image']=xy[:,0]
            params['point_source_model'][0][0]['dec_image']=xy[:,1]
            if spec.get('fixed_observed_astrometry',False):
                params['point_source_model'][2][0]['ra_image']=xy[:,0]
                params['point_source_model'][2][0]['dec_image']=xy[:,1]
    if not constraints.get('Ddt_sampling'):
        raise ValueError('Time-delay cosmography requires sampled D_dt')
    if not likelihood.get('time_delay_likelihood'):
        raise ValueError('Joint fit must include time-delay observations')
    model=dict(spec['model'],z_lens=obs['z_lens'],z_source=obs['z_source'])
    if 'kinematic_observations' in joint or 'time_delay_covariance_days2' in joint:
        if 'kinematic_observations' in joint and not constraints.get('kinematic_sampling'):
            raise ValueError('Joint stellar kinematics requires sampled D_d and D_dt')
        likelihood['custom_logL_addition']=_joint_lens_addition(obs,spec,model)
        if 'time_delay_covariance_days2' in joint:
            likelihood['time_delay_likelihood']=False
            joint.pop('time_delay_covariance_days2')
        joint.pop('kinematic_observations',None)
    if obs.get('los_constraints'):
        priors=obs['los_constraints'].get('prior_lens',[])
        if not priors:
            raise ValueError('LOS observations require explicit lens-component priors')
        likelihood['prior_lens']=likelihood.get('prior_lens',[])+priors
    sequence=FittingSequence(joint,model,constraints,likelihood,params,verbose=False)
    state=np.random.get_state()
    np.random.seed(spec['sampling_seed'])
    try:
        chains=sequence.fit_sequence(spec['fitting_sequence'])
    finally:
        np.random.set_state(state)
    samples=[];lens_samples=[]
    for chain in chains:
        if chain[0] in ('MCMC','emcee','zeus'):
            for vector in np.asarray(chain[1]):
                decoded=sequence.param_class.args2kwargs(vector)
                d=float(decoded['kwargs_special']['D_dt'])
                if d>0:
                    samples.append(d);lens_samples.append(decoded['kwargs_lens'])
    if len(samples)<spec['minimum_posterior_samples']:
        raise RuntimeError('Joint lens fit has insufficient posterior samples; point fit is not a posterior')
    profiles={}
    if spec.get('profile_radii_arcsec'):
        from lenstronomy.LensModel.lens_model import LensModel
        profile_lens=LensModel(**{k:v for k,v in model.items() if k in {'lens_model_list',
            'z_lens','z_source','lens_redshift_list','multi_plane'}})
        radii=array(spec['profile_radii_arcsec'],'inferred profile radii',1,positive=True)
        angles=np.linspace(0,2*np.pi,spec['profile_angular_samples'],endpoint=False)
        xx=radii[:,None]*np.cos(angles);yy=radii[:,None]*np.sin(angles)
        measured=[]
        for lens_sample in lens_samples[::spec.get('profile_posterior_stride',1)]:
            measured.append(np.mean(profile_lens.kappa(xx,yy,lens_sample,
                k=spec.get('profile_component_index')),axis=1))
        profiles={'kappa':{'radii_arcsec':radii,'samples':measured}}
    ddt=np.asarray(samples);coef=_ddt_h0_coefficient(obs['z_lens'],obs['z_source'],spec['expansion_shape'])
    h0=coef/ddt
    value=gaussian_observable(float(np.mean(h0)),float(np.var(h0,ddof=1)),'km/s/Mpc')
    value['interval_68']=np.quantile(h0,[.16,.84]).tolist()
    fit={'best_fit':sequence.best_fit(),'D_dt_mpc_samples':ddt,'H0_samples':h0,
         'lens_parameter_samples':lens_samples,'inferred_profiles':profiles,
         'mean':[float(np.mean(ddt))],'covariance':[[float(np.var(ddt,ddof=1))]],
         'parameter_names':['D_dt_mpc'],'backend':pin,
         'posterior_status':'SAMPLED; convergence and independent coverage not executed'}
    return freeze_result('time_delay',fit,observables={'H0':value,
        'D_dt':gaussian_observable(np.mean(ddt),np.var(ddt,ddof=1),'Mpc',obs['z_lens'])},
        selection=obs['selection'],object_ids=[obs['object_id']],namespace=obs['namespace'],
        model=declared_model_family,realization_id=realization_id)


def fit_delay_distance(observations, fermat_model, *, realization_id=None):
    """Covariance-aware reduced analytic lens control, explicitly distinct from imaging fit.

    Fermat differences and their uncertainty must be inferred from a frozen lens
    model, not copied from the generator. External convergence is marginalized
    through supplied posterior draws. Degenerate unconstrained mass sheets remain
    unresolved rather than being fixed invisibly.
    """
    obs=read_observations(observations)
    if fermat_model.get('role')!='frozen_lens_model':
        raise PermissionError('Fermat model must be an independently frozen inference product')
    dt=array(obs['joint_data']['time_delays_measured'],'delays',1)
    phi=array(fermat_model['delta_phi_arcsec2'],'inferred Fermat differences',1)
    if phi.shape!=dt.shape:raise ValueError('Delay and Fermat shapes differ')
    dc=obs['joint_data'].get('time_delay_covariance_days2')
    if dc is None:dc=np.diag(np.asarray(obs['joint_data']['time_delays_uncertainties'])**2)
    dc=covariance(dc,len(dt)); pc=covariance(fermat_model['fermat_covariance'],len(dt),semidefinite=True)
    coeff=ARCSEC_RAD**2*MPC_KM/C_KMS/DAY_SECONDS
    from scipy.optimize import minimize_scalar
    def objective(d):
        cov=dc+(d*coeff)**2*pc;r=dt-d*coeff*phi
        return .5*(r@np.linalg.solve(cov,r)+np.linalg.slogdet(cov)[1])
    bounds=fermat_model['distance_bounds_mpc']
    opt=minimize_scalar(objective,bounds=bounds,method='bounded')
    if not opt.success:raise RuntimeError('Reduced delay fit failed')
    step=max(1e-3,opt.x*1e-4)
    curvature=(objective(opt.x+step)-2*objective(opt.x)+objective(opt.x-step))/step**2
    if curvature<=0:raise ValueError('Delay distance is not identifiable')
    rng=np.random.default_rng(fermat_model['posterior_seed'])
    draws=rng.normal(opt.x,1/np.sqrt(curvature),fermat_model['posterior_draws'])
    kappa=array(fermat_model['external_convergence_samples'],'external convergence',1)
    if np.any(kappa>=1) or not len(kappa):raise ValueError('External convergence is invalid')
    draws=draws/(1-rng.choice(kappa,len(draws)))
    draws=draws[draws>0]
    coef=_ddt_h0_coefficient(obs['z_lens'],obs['z_source'],fermat_model['expansion_shape'])
    h0=coef/draws
    value=gaussian_observable(h0.mean(),h0.var(ddof=1),'km/s/Mpc')
    value['interval_68']=np.quantile(h0,[.16,.84]).tolist()
    return freeze_result('time_delay_reduced',{'D_dt_samples':draws,'H0_samples':h0,
        'negative_log_likelihood':float(opt.fun)},observables={'H0':value},selection=obs['selection'],
        object_ids=[obs['object_id']],namespace=obs['namespace'],model=fermat_model,realization_id=realization_id)


def matched_lens_experiments(cdm_targets,sidm_targets,experiments):
    """Pair by explicit Lagrangian identity and identical observing/source settings."""
    c={r['lagrangian_id']:r for r in cdm_targets};s={r['lagrangian_id']:r for r in sidm_targets}
    if len(c)!=len(cdm_targets) or len(s)!=len(sidm_targets):
        raise ValueError('Duplicate matched-lens identity')
    result=[]
    for e in experiments:
        key=e['lagrangian_id']
        if key not in c or key not in s:raise ValueError('Matched CDM/SIDM lens missing: '+key)
        result.append({'lagrangian_id':key,'cdm':c[key],'sidm':s[key],
            'source':e['source'],'instrument':e['instrument'],'seed':e['seed']})
    return result


def spherical_jeans_aperture_velocity(theta_e_arcsec,gamma,r_eff_arcsec,dd_mpc,ddt_mpc,z_lens,
        anisotropy,aperture,*,radial_samples=384):
    """Deterministic spherical EPL+Hernquist Jeans model with PSF/aperture integration.

    This model is an explicitly declared inference family. Projected simulation
    truth remains independent. Surface-density normalization uses Sigma_crit
    from fitted D_d,D_dt. Numerical radial support/order are recorded by the caller.
    """
    from scipy.special import gamma as gamma_function
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import gaussian_filter
    if not 1<gamma<3 or min(theta_e_arcsec,r_eff_arcsec,dd_mpc,ddt_mpc)<=0:
        raise ValueError('Spherical Jeans domain: positive scales, 1 < mass slope < 3')
    re=theta_e_arcsec*ARCSEC_RAD*dd_mpc;hern=r_eff_arcsec/1.8153*ARCSEC_RAD*dd_mpc
    sigcrit=C_KMS**2/(4*np.pi*G_MPC_KMS2_MSUN)*ddt_mpc/((1+z_lens)*dd_mpc**2)
    rho0=sigcrit*(3-gamma)/(2*re)*gamma_function(gamma/2)/(np.sqrt(np.pi)*gamma_function((gamma-1)/2))
    r=np.geomspace(min(re,hern)*1e-6,max(re,hern)*aperture['radial_extent_factor'],radial_samples)
    nu=hern/(r*(r+hern)**3)
    mass=4*np.pi*rho0*re**gamma*r**(3-gamma)/(3-gamma)
    if anisotropy['kind']=='constant':
        beta=float(anisotropy['beta'])
        if beta>=1:raise ValueError('Stellar anisotropy beta must be <1')
        factor=r**(2*beta);beta_func=lambda rr:beta+np.zeros_like(rr)
    elif anisotropy['kind']=='osipkov_merritt':
        ra=float(anisotropy['r_ani_arcsec'])*ARCSEC_RAD*dd_mpc
        if ra<=0:raise ValueError('Anisotropy radius must be positive')
        factor=r*r+ra*ra;beta_func=lambda rr:rr*rr/(rr*rr+ra*ra)
    else:raise ValueError('Unsupported spherical anisotropy model')
    integrand=nu*G_MPC_KMS2_MSUN*mass/r**2*factor
    integral=-cumulative_trapezoid(integrand[::-1],x=r[::-1],initial=0.)[::-1]
    sr=integral/(nu*factor)
    nuf=PchipInterpolator(np.log(r),np.log(nu),extrapolate=False)
    srf=PchipInterpolator(np.log(r[:-1]),sr[:-1],extrapolate=False)
    # Transform line-of-sight integration to l rather than Abel r: no endpoint singularity.
    projected_r=np.geomspace(r[0]*2,r[-2]/2,radial_samples//2)
    light=np.empty_like(projected_r);second=np.empty_like(projected_r)
    for j,R in enumerate(projected_r):
        l=np.concatenate([[0.],np.geomspace(r[0],np.sqrt(r[-2]**2-R*R),radial_samples)])
        rr=np.minimum(np.sqrt(R*R+l*l),r[-2]);lum=np.exp(nuf(np.log(rr)))
        light[j]=2*np.trapezoid(lum,x=l)
        second[j]=2*np.trapezoid(lum*srf(np.log(rr))*(1-beta_func(rr)*R*R/rr**2),x=l)
    pixel=float(aperture['pixel_arcsec']);extent=float(aperture['half_width_arcsec'])
    count=int(np.ceil(2*extent/pixel))+1
    ax=(np.arange(count)-(count-1)/2)*pixel;xx,yy=np.meshgrid(ax,ax)
    rr=np.maximum(np.hypot(xx,yy)*ARCSEC_RAD*dd_mpc,projected_r[0])
    if rr.max()>projected_r[-1]:raise ValueError('Jeans radial support smaller than kinematic aperture')
    surface=np.interp(np.log(rr),np.log(projected_r),light)
    weighted=np.interp(np.log(rr),np.log(projected_r),second)
    seeing=float(aperture['seeing_sigma_arcsec'])/pixel
    if seeing>0:
        surface=gaussian_filter(surface,seeing,mode='constant')
        weighted=gaussian_filter(weighted,seeing,mode='constant')
    if aperture['kind']=='circular':mask=xx*xx+yy*yy<=aperture['radius_arcsec']**2
    elif aperture['kind']=='slit':
        angle=np.radians(aperture.get('position_angle_deg',0.));xp=xx*np.cos(angle)+yy*np.sin(angle)
        yp=-xx*np.sin(angle)+yy*np.cos(angle)
        mask=(np.abs(xp)<=aperture['width_arcsec']/2)&(np.abs(yp)<=aperture['length_arcsec']/2)
    else:raise ValueError('Unsupported stellar aperture')
    if not mask.any():raise ValueError('No aperture pixels')
    return float(np.sqrt(weighted[mask].sum()/surface[mask].sum()))


def _joint_lens_addition(obs,spec,model):
    """Full delay covariance and deterministic stellar Jeans log likelihood."""
    from lenstronomy.LensModel.lens_model import LensModel
    joint=obs['joint_data'];delay_cov=joint.get('time_delay_covariance_days2')
    kin=joint.get('kinematic_observations')
    delay_precision=None if delay_cov is None else np.linalg.inv(covariance(delay_cov,len(joint['time_delays_measured'])))
    lens=LensModel(**{k:v for k,v in model.items() if k in {'lens_model_list','z_lens','z_source',
        'lens_redshift_list','multi_plane','cosmology_model','cosmology_sampling'}})
    observed_dt=np.asarray(joint['time_delays_measured'])
    if kin is not None:
        observed_v=array(kin['velocity_dispersion_kms'],'stellar dispersions',1,positive=True)
        kin_precision=np.linalg.inv(covariance(kin['covariance_kms2'],len(observed_v)))
        if spec['kinematics']['family']!='spherical_epl_hernquist':
            raise ValueError('Declare the deterministic spherical_epl_hernquist kinematic family')
    def addition(kwargs_lens,kwargs_source=None,kwargs_lens_light=None,kwargs_ps=None,
                 kwargs_special=None,kwargs_extinction=None,**unused):
        total=0.
        if delay_precision is not None:
            positions=kwargs_ps[0]
            phi=lens.fermat_potential(np.asarray(positions['ra_image']),np.asarray(positions['dec_image']),kwargs_lens)
            predicted=fermat_delay_days(kwargs_special['D_dt'],phi[1:]-phi[0])
            r=observed_dt-predicted;total-=.5*r@delay_precision@r
        if kin is not None:
            settings=spec['kinematics'];lm=kwargs_lens[settings['lens_component_index']]
            light=kwargs_lens_light[settings['light_component_index']]
            dd=kwargs_special['D_d'];ddt=kwargs_special['D_dt']
            anis=dict(settings['anisotropy'])
            for key,param in settings.get('sampled_anisotropy',{}).items():anis[key]=kwargs_special[param]
            try:
                prediction=np.asarray([spherical_jeans_aperture_velocity(lm['theta_E'],lm['gamma'],
                    light['R_sersic'],dd,ddt,obs['z_lens'],anis,ap,
                    radial_samples=settings['radial_samples']) for ap in kin['apertures']])
            except ValueError:return -1e100
            r=observed_v-prediction;total-=.5*r@kin_precision@r
        return float(total)
    return addition


def native_host_mass_planes(run_dir,requests,*,memory_gib):
    """Connect the existing native host-component reader to lens projection.

    Sightline bases have two orthonormal physical sky axes. Host-local coordinates
    and physical masses come from the existing Block 11 dataset conversion.
    Each host/snapshot contributes an explicitly named LOS redshift plane.
    """
    try:from .halos import load_native_host_component_samples
    except ImportError:from halos import load_native_host_component_samples
    planes=[];sources={}
    for request in requests:
        basis=array(request['sky_basis'],'lens sky basis',2)
        if basis.shape!=(2,3) or not np.allclose(basis@basis.T,np.eye(2),atol=1e-10):
            raise ValueError('Lens sky basis must contain two orthonormal 3D axes')
        components,frame=load_native_host_component_samples(run_dir,request['snapshot'],
            request['host_row'],request['component_definitions'],request['support_radius_kpc'],
            memory_gib=memory_gib)
        for name,component in components.items():
            if not len(component['particle_ids']):continue
            planes.append({'redshift':request['plane_redshift'],'component':name,
                'proper_xy_mpc':component['offsets_proper_kpc']@basis.T/1000.,
                'mass_msun':component['masses_msun'],'particle_id':component['particle_ids'],
                'softening_mpc':request['component_softening_kpc'][name]/1000.,
                'namespace':request['component_namespaces'][name]})
        sources.update(frame['source_hashes'])
    if not planes:raise ValueError('Native lens selection has no physical mass')
    return planes,sources


def compare_h0_shape_profile_and_external_convergence_recovery(frozen_posterior,truth_path,expected_hash):
    """Privileged lens evaluation available only after a posterior is frozen."""
    from .observation_data import read_frozen, file_sha256
    from pathlib import Path
    import json
    result=read_frozen(frozen_posterior);path=Path(truth_path)
    if file_sha256(path)!=expected_hash:raise ValueError('Lens truth hash mismatch')
    truth=json.loads(path.read_text())
    if truth.get('role')!='privileged_lens_truth':raise PermissionError('Lens evaluation needs a separate privileged truth product')
    checks=[]
    h=result['observables']['H0'];target=float(truth['H0_kms_mpc'])
    checks.append({'quantity':'H0','estimate':h['value'],'target':target,
                  'bias':h['value']-target,'covered_68':h['interval_68'][0]<=target<=h['interval_68'][1]})
    samples=result['fit'].get('lens_parameter_samples',[])
    for parameter in truth.get('parameters',[]):
        component=parameter['component_index'];key=parameter['parameter_name']
        values=np.asarray([p[component][key] for p in samples])
        if not len(values):
            checks.append({'quantity':key,'component_index':component,'status':'NO_POSTERIOR_SAMPLES'});continue
        lo,hi=np.quantile(values,[.16,.84]);target=float(parameter['value'])
        checks.append({'quantity':key,'component_index':component,'estimate':float(values.mean()),
            'sigma':float(values.std(ddof=1)),'target':target,'bias':float(values.mean()-target),
            'interval_68':[float(lo),float(hi)],'covered_68':bool(lo<=target<=hi)})
    profiles=[]
    for item in truth.get('reference_profiles',[]):
        quantity=item['quantity']
        measured=result['fit'].get('inferred_profiles',{}).get(quantity)
        if measured is None:
            profiles.append({'quantity':quantity,'status':'PROFILE_NOT_REQUESTED_IN_FROZEN_FIT'});continue
        if not np.allclose(measured['radii_arcsec'],item['radii_arcsec'],rtol=1e-10,atol=1e-12):
            raise ValueError('Inferred/reference lens radial supports differ')
        draws=array(measured['samples'],'frozen profile samples',2)
        reference=array(item['value'],'reference lens profile',1)
        if draws.shape[1]!=len(reference):raise ValueError('Lens profile shapes differ')
        mean=draws.mean(axis=0);residual=mean-reference
        pcov=np.atleast_2d(np.cov(draws,rowvar=False,ddof=1))
        pcov+=covariance(item['reference_covariance'],len(reference),semidefinite=True)
        pcov=covariance(pcov,len(reference),'lens profile recovery covariance')
        profiles.append({'quantity':quantity,'radii_arcsec':item['radii_arcsec'],
            'estimate':mean.tolist(),'residual':residual.tolist(),
            'covariance':pcov.tolist(),'chi2':float(residual@np.linalg.solve(pcov,residual))})
    return {'schema':'lens-recovery-v1','role':'post_fit_evaluation','parameter_recovery':checks,
        'profile_recovery':profiles,'frozen_posterior_hash':result['frozen_hash'],'truth_hash':expected_hash,
        'scientific_acceptance':'UNRESOLVED_NOT_EXECUTED','validation_receipts':[]}
# @research2-end-module time_delay_cosmography

