"""Host motion, reconstructed fields and extended gravity components.

Hermite motion uses recorded proper velocities. Proper cosmological center
velocities must include Hubble motion before entering this interface. Forces
are evaluated in the inertial frame, avoiding omitted fictitious forces.

Fields return Phi [(km/s)^2], acceleration [(km/s)^2/proper kpc], and
the potential Hessian [(km/s)^2/proper kpc^2]; the tidal tensor is -H(Phi).
"""
from __future__ import annotations


from pathlib import Path
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.integrate import quad
from scipy.spatial.transform import Rotation, RotationSpline
from core.stream_common import finite, increasing, positive, KMS_TO_KPC_MYR, G



def _positions(value):
    """Cartesian query rows in proper kpc; field outputs keep the same row order."""
    value=finite(value)
    if value.ndim!=2 or value.shape[1]!=3:
        raise ValueError('Gravity queries require N by 3 Cartesian positions')
    return value


class HostFrame:
    """Move between measured body coordinates and a common inertial frame.

    Positions are proper kpc, velocities km/s and times Myr. R maps column
    body vectors into inertial vectors. For the row arrays used here,
    x_inertial = x_body R.T + center and
    v_inertial = (v_body + omega_body cross x_body / K) R.T + center_velocity,
    where K converts km/s to kpc/Myr. The rotation term is essential even
    when the body's origin is stationary.
    """
    def __init__(self, times_myr, center_kpc, center_velocity_kms, rotations, *, max_interval_myr,
                 center_covariance_kpc2=None, source_hashes=None):
        self.times = increasing(times_myr)
        n = len(self.times)
        self.center = finite(center_kpc, (n, 3))
        self.velocity = finite(center_velocity_kms, (n, 3))
        matrices = finite(rotations, (n, 3, 3))
        if (not np.allclose(matrices @ matrices.transpose(0, 2, 1), np.eye(3), rtol=0, atol=1e-10)
                or not np.allclose(np.linalg.det(matrices), 1, rtol=0, atol=1e-10)):
            raise ValueError('Recorded local-to-inertial rotations must be proper orthonormal matrices')
        if np.max(np.diff(self.times)) > positive(max_interval_myr, 'max_interval_myr'):
            raise ValueError('Recorded host-motion cadence exceeds the frozen budget')
        self.motion = CubicHermiteSpline(self.times, self.center, self.velocity * KMS_TO_KPC_MYR, extrapolate=False)
        self.orientation = RotationSpline(self.times, Rotation.from_matrix(matrices))
        self.sources = dict(source_hashes or {})
        self.covariance = np.zeros((n, 3, 3)) if center_covariance_kpc2 is None else finite(center_covariance_kpc2, (n, 3, 3))
        if not np.allclose(self.covariance, self.covariance.transpose(0, 2, 1)) or np.any(np.linalg.eigvalsh(self.covariance) < -1e-12):
            raise ValueError('Host center covariance must be positive semidefinite')

    def check_time(self, t):
        if isinstance(t, (bool, np.bool_)) or np.ndim(t) != 0 or not np.isfinite(t) or t < self.times[0] or t > self.times[-1]:
            raise ValueError('Host frame extrapolation is forbidden')

    def state(self, t):
        self.check_time(t)
        return self.motion(t), self.motion(t, 1) / KMS_TO_KPC_MYR, self.orientation(t).as_matrix()

    def to_inertial(self, x_local, v_local, t):
        """Local velocities include rotation-relative derivatives, not peculiar catalog velocities."""
        c, v, r = self.state(t)
        omega_body = self.orientation(t, 1)  # radians/Myr in rotating frame
        x = finite(x_local)
        if not x.ndim or x.shape[-1] != 3:
            raise ValueError('Body positions require Cartesian vectors')
        velocity = finite(v_local, x.shape)
        return x @ r.T + c, (velocity + np.cross(omega_body, x) / KMS_TO_KPC_MYR) @ r.T + v

    def to_local(self, x, v, t):
        c, vc, r = self.state(t)
        position = finite(x)
        if not position.ndim or position.shape[-1] != 3:
            raise ValueError('Inertial positions require Cartesian vectors')
        local = (position - c) @ r
        return local, (finite(v, position.shape) - vc) @ r - np.cross(self.orientation(t, 1), local) / KMS_TO_KPC_MYR

    def covariance_at(self, t):
        self.check_time(t)
        return np.array([[np.interp(t, self.times, self.covariance[:, i, j]) for j in range(3)] for i in range(3)])


class SatelliteWorldline:
    """Recorded extended Plummer perturber; no inferred point-mass substitute."""
    def __init__(self, record):
        self.times = increasing(record['times_myr'])
        self.mass = finite(record['mass_msun'], (len(self.times),))
        self.scale = finite(record['scale_kpc'], (len(self.times),))
        if np.any(self.mass < 0) or np.any(self.scale <= 0): raise ValueError('Invalid extended satellite worldline')
        if record['profile'] != 'plummer': raise ValueError('Only declared Plummer worldlines are supported here')
        self.motion = CubicHermiteSpline(self.times, finite(record['position_kpc'], (len(self.times), 3)),
            finite(record['velocity_kms'], (len(self.times), 3)) * KMS_TO_KPC_MYR, extrapolate=False)
        self.identity = record['identity']

    def evaluate(self, x, t):
        """Return Plummer phi, acceleration and potential Hessian in physical units.

        With q = |x-center|^2 + b^2: phi = -GM/sqrt(q), a = -GM d/q^(3/2),
        and H = GM[I/q^(3/2) - 3 dd.T/q^(5/2)]. The tidal tensor is -H.
        """
        if isinstance(t, (bool, np.bool_)) or np.ndim(t) != 0 or not np.isfinite(t) or t < self.times[0] or t > self.times[-1]:
            raise ValueError('Satellite worldline extrapolation forbidden')
        x = finite(x)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError('Satellite queries require N by 3 proper-kpc positions')
        d = x - self.motion(t)
        m = np.interp(t, self.times, self.mass); b = np.interp(t, self.times, self.scale)
        r2 = np.sum(d*d, axis=1) + b*b
        phi = -G*m / np.sqrt(r2)
        acc = -G*m * d / r2[:, None]**1.5
        hess = G*m * (np.eye(3)[None] / r2[:, None, None]**1.5 - 3*d[:, :, None]*d[:, None, :] / r2[:, None, None]**2.5)
        return phi, acc, hess


class TimeDependentHost:
    """Interpolate measured body-frame fields, then return inertial forces.

    Forces and Hessians transform as a_global = R a_body and
    H_global = R H_body R.T. Integrating in the inertial frame avoids adding
    Coriolis/centrifugal terms. Temporal field interpolation is a declared
    approximation inside measured support, not a convergence certificate.
    """
    def __init__(self, frame, fields, field_times_myr, *, support_radius_kpc, physics,
                 satellites=(), satellite_accounting='already_in_field'):
        self.frame = frame; self.times = increasing(field_times_myr)
        self.fields = tuple(fields); self.support = positive(support_radius_kpc, 'support_radius_kpc')
        if len(self.fields) != len(self.times): raise ValueError('Every field epoch needs actual reconstructed coefficients')
        if self.times[0] < frame.times[0] or self.times[-1] > frame.times[-1]: raise ValueError('Frame coverage incomplete')
        if physics not in {'collisionless', 'elastic_sidm'}: raise ValueError('Declare evolved host DM physics')
        self.physics = physics; self.satellites = tuple(satellites)
        if satellite_accounting not in {'already_in_field', 'excluded_from_field'}: raise ValueError('Declare satellite accounting')
        if self.satellites and satellite_accounting != 'excluded_from_field':
            raise ValueError('Satellite gravity would be counted twice')
        self.satellite_accounting = satellite_accounting

    def evaluate(self, x, t):
        self.frame.check_time(t)
        if t < self.times[0] or t > self.times[-1]: raise ValueError('Host field extrapolation forbidden')
        x = finite(x)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError('Host queries require N by 3 proper-kpc positions')
        center, _, r = self.frame.state(t)
        local = (x - center) @ r
        if np.any(np.linalg.norm(local, axis=1) > self.support): raise ValueError('Orbit exits calibrated host field support')
        k = min(len(self.times)-2, max(0, np.searchsorted(self.times, t, side='right')-1))
        f = (t-self.times[k])/(self.times[k+1]-self.times[k])
        # At a saved epoch only that measured field contributes. Requiring the
        # unused neighbor can reject a query outside the neighbor's support.
        if f==0:phi,a,h=self.fields[k].evaluate(local)
        elif f==1:phi,a,h=self.fields[k+1].evaluate(local)
        else:
            left=self.fields[k].evaluate(local);right=self.fields[k+1].evaluate(local)
            phi,a,h=tuple((1-f)*l+f*q for l,q in zip(left,right))
        phi=finite(phi,(len(x),),'host potential').copy()
        a=finite(a,x.shape,'host acceleration')
        h=finite(h,(len(x),3,3),'host potential Hessian')
        a = a @ r.T
        h = np.einsum('ij,njk,lk->nil', r, h, r)
        for satellite in self.satellites:
            p, sa, sh = satellite.evaluate(x, t); phi += p; a += sa; h += sh
        return phi, a, h

    @classmethod
    def from_products(cls, frame_record, field_records, *, memory_gib):
        from core.stream_common import read_product
        from core.halos import ReconstructedHostPotential
        frame = HostFrame(**frame_record)
        models = []; times = []; support = []; physics = set(); sources = dict(frame.sources)
        for record in field_records:
            path = Path(record['product']).resolve()
            report, _, bound = read_product(path, memory_gib=memory_gib)
            if report.get('field_validation_status') != 'PASS': raise ValueError('Unresolved parent field cannot drive live streams')
            reconstructed=ReconstructedHostPotential.from_coefficients(path.parent, report['potential_coefficients'])
            # Block11 coefficients are in parent Cartesian offset axes. Rotate
            # each measured epoch into the moving body's axes before temporal
            # interpolation, then rotate the interpolated force back at t.
            rotation=frame.state(record['proper_time_myr'])[2]
            class BodyAlignedField:
                def __init__(self,model,rotation):self.model=model;self.rotation=rotation
                def evaluate(self,query):
                    r=self.rotation
                    phi,acc,hess=self.model.evaluate(query@r.T)
                    return phi,acc@r,np.einsum('ji,njk,kl->nil',r,hess,r)
            models.append(BodyAlignedField(reconstructed,rotation))
            times.append(record['proper_time_myr']); support.append(report['request']['support_radius_proper_kpc'])
            physics.add(record['physics'])
            coefficient_sources={str((path.parent/row['file']).resolve()):row['sha256']
                for row in report['potential_coefficients']}
            for group in (bound,report['source_hashes'],coefficient_sources):
                for source,checksum in group.items():
                    source=str(Path(source).resolve())
                    if source in sources and sources[source]!=checksum:
                        raise ValueError('Host epochs bind conflicting generations of the same source')
                    sources[source]=checksum
        if len(physics) != 1: raise ValueError('Mixed nonlinear host physics in one field history')
        host = cls(frame, models, times, support_radius_kpc=min(support), physics=physics.pop())
        host.sources = sources
        return host


# Extended spherical and nonaxisymmetric components in the same physical units.

def _nfw_enclosed_factor(radius_over_scale):
    """log(1+z)-z/(1+z), with its small-z series to avoid cancellation."""
    z=finite(radius_over_scale,name='NFW scaled radius')
    if np.any(z<0):raise ValueError('NFW scaled radius cannot be negative')
    result=np.array(np.log1p(z)-z/(1+z))
    small=z<1e-3;local=z[small]
    # The z term cancels: f(z)=sum(k>=2) (-1)^k*(k-1)*z^k/k.
    result[small]=sum((-1)**k*(k-1)*local**k/k for k in range(2,9))
    return result


class RadialProfile:
    """Measured enclosed mass with finite-mass exterior and declared inner core.

    A monotone cubic mass interpolant gives rho=M'/(4 pi r^2). The potential
    is computed from the same M, not an independently fitted NFW potential.
    The outer truncation is the last measured radius. Central extrapolation
    is permitted only via the explicit constant-density-core assumption.
    """
    def __init__(self, record):
        self.record = record
        self.kind = record['kind']
        if self.kind not in {'point_mass', 'plummer', 'hernquist', 'nfw_truncated', 'enclosed_mass'}:
            raise ValueError('Unsupported physical perturber profile')
        self.mass = float(finite(record['mass_msun'],(),'mass_msun'))
        if self.mass<0:raise ValueError('Profile mass cannot be negative')
        self.minimum_radius = positive(record['minimum_supported_radius_kpc'], 'minimum_supported_radius_kpc')
        if self.kind in {'plummer', 'hernquist', 'nfw_truncated'}:
            self.scale = positive(record['scale_kpc'], 'scale_kpc')
        if self.kind == 'nfw_truncated':
            self.truncation = positive(record['truncation_radius_kpc'], 'truncation_radius_kpc')
            self.normalization = positive(_nfw_enclosed_factor(self.truncation/self.scale),'NFW mass normalization')
        if self.kind == 'enclosed_mass':
            self.radius = increasing(record['radius_kpc'], 'profile radius')
            self.enclosed = finite(record['enclosed_mass_msun'], self.radius.shape)
            if self.radius[0] <= 0 or np.any(self.enclosed < 0) or np.any(np.diff(self.enclosed) < 0):
                raise ValueError('Positive radial support and monotone enclosed mass required')
            if not np.isclose(self.enclosed[-1], self.mass, rtol=1e-8,atol=0.):
                raise ValueError('Bound mass must equal measured truncated enclosed mass')
            if record['inner_policy'] != 'constant_density_core':
                raise ValueError('Declare the supported inner mass extension')
            self.interpolator = PchipInterpolator(self.radius, self.enclosed, extrapolate=False)
            self.truncation = self.radius[-1]
            self.tail_integral = np.array([quad(lambda r: float(self.interpolator(r)) / r**2,
                self.radius[k], self.radius[k + 1], epsabs=0., epsrel=1e-9)[0] for k in range(len(self.radius) - 1)])
            self.cumulative_tail = np.r_[np.cumsum(self.tail_integral[::-1])[::-1], 0.]

    def mass_and_derivative(self, radius):
        r = finite(radius,name='profile radius')
        if np.any(r<0):raise ValueError('Profile radii cannot be negative')
        if self.mass==0:return np.zeros_like(r),np.zeros_like(r)
        if self.kind == 'point_mass': return np.full_like(r, self.mass), np.zeros_like(r)
        if self.kind == 'plummer':
            q = r*r + self.scale**2
            return self.mass*r**3/q**1.5, 3*self.mass*r*r*self.scale**2/q**2.5
        if self.kind == 'hernquist':
            return self.mass*r*r/(r+self.scale)**2, 2*self.mass*r*self.scale/(r+self.scale)**3
        if self.kind == 'nfw_truncated':
            z = np.minimum(r, self.truncation) / self.scale
            m = self.mass * _nfw_enclosed_factor(z) / self.normalization
            d = self.mass*r/(r+self.scale)**2 / self.normalization
            return m, np.where(r < self.truncation, d, 0.)
        clipped = np.clip(r, self.radius[0], self.truncation)
        m = self.interpolator(clipped); d = self.interpolator.derivative()(clipped)
        inner = r < self.radius[0]; outer = r >= self.truncation
        m = np.where(inner, self.enclosed[0]*(r/self.radius[0])**3, m)
        d = np.where(inner, 3*self.enclosed[0]*r*r/self.radius[0]**3, d)
        return np.where(outer, self.mass, m), np.where(outer, 0., d)

    def potential(self, radius):
        r = finite(radius,name='profile radius')
        if np.any(r<0):raise ValueError('Profile radii cannot be negative')
        if self.mass==0:return np.zeros_like(r)
        if self.kind == 'point_mass': return -G*self.mass/r
        if self.kind == 'plummer': return -G*self.mass/np.sqrt(r*r+self.scale**2)
        if self.kind == 'hernquist': return -G*self.mass/(r+self.scale)
        if self.kind == 'nfw_truncated':
            safe = np.maximum(r, np.finfo(float).tiny)
            central = np.divide(np.log1p(safe/self.scale), safe,
                out=np.full_like(safe, 1/self.scale), where=safe > 1e-150)
            # Exterior matching fixes the otherwise arbitrary potential constant.
            shift = G*self.mass/self.normalization*np.log1p(self.truncation/self.scale)/self.truncation - G*self.mass/self.truncation
            return np.where(r < self.truncation, -G*self.mass/self.normalization*central+shift, -G*self.mass/safe)
        # Phi(r)=-G[M(Rt)/Rt + integral(r,Rt) M(q)/q^2 dq].
        # Its derivative is +G*M(r)/r^2, matching the force below.
        result = np.empty_like(r)
        for index, value in np.ndenumerate(r):
            if value >= self.truncation:
                result[index] = -G*self.mass/value
            elif value < self.radius[0]:
                tail = self.cumulative_tail[0] + self.enclosed[0]*(self.radius[0]**2-value**2)/(2*self.radius[0]**3)
                result[index] = -G*(self.mass/self.truncation+tail)
            else:
                k = min(len(self.radius)-2, np.searchsorted(self.radius, value, side='right')-1)
                tail = self.cumulative_tail[k+1] + quad(lambda q: float(self.interpolator(q))/q**2,
                    value, self.radius[k+1], epsabs=0., epsrel=1e-9)[0]
                result[index] = -G*(self.mass/self.truncation+tail)
        return result

    def evaluate(self, x, t=None):
        x = _positions(x)
        if self.mass==0:return np.zeros(len(x)),np.zeros_like(x),np.zeros((len(x),3,3))
        r = np.linalg.norm(x, axis=1)
        if np.any(r < self.minimum_radius):
            raise ValueError('Orbit enters unresolved/singular perturber radius; refine or use supported profile')
        m, derivative = self.mass_and_derivative(r)
        radial = G*m/r**2; transverse = G*m/r**3
        # Phi'=G*M/r^2; Phi''=G*M'/r^2-2*G*M/r^3.
        # H=(Phi'/r)I+(Phi''-Phi'/r)uuT, in (km/s)^2/kpc^2.
        second = G*derivative/r**2-2*G*m/r**3
        u = x/r[:, None]
        return self.potential(r), -radial[:, None]*u, transverse[:, None, None]*np.eye(3) + (second-transverse)[:, None, None]*u[:, :, None]*u[:, None, :]


class EvolvingPerturber:
    """Interpolate actual epoch forces/potentials, preserving disruption flags.

    Each profile carries its own measured epoch mass, matched to the worldline.
    Interpolating potential, force and Hessian with the SAME weight retains
    their derivative relationships; zero-mass epochs contribute zero gravity.
    """
    def __init__(self, record):
        from core.stream.trajectories import Worldline
        self.record = record; self.line = Worldline(record['worldline'])
        self.identity = self.line.key
        self.profiles = tuple(RadialProfile(q) for q in record['profiles'])
        if len(self.profiles) != len(self.line.times): raise ValueError('One measured profile per trajectory epoch required')
        self.support = record['support_label']
        self.category = record['category']
        if self.category not in {'black_hole', 'cdm_subhalo', 'wdm_subhalo', 'sidm_subhalo', 'gmc', 'satellite', 'bulge'}:
            raise ValueError('Declare physical perturber category')
        self.provenance = record['provenance']
        if self.support == 'simulation_derived' and not self.provenance.get('source_hashes'):
            raise ValueError('Simulation-derived components require measurement provenance')
        for profile, mass in zip(self.profiles, self.line.mass):
            if not np.isclose(profile.mass, mass, rtol=1e-6,atol=0.):
                raise ValueError('Worldline bound mass and evolved profile mass disagree')

    def evaluate(self, x, t):
        k = self.line.interval(t); center, _ = self.line.state(t)
        fraction = (t-self.line.times[k])/np.diff(self.line.times)[k]
        query=_positions(x)-center
        if fraction==0:return self.profiles[k].evaluate(query)
        if fraction==1:return self.profiles[k+1].evaluate(query)
        left = self.profiles[k].evaluate(query)
        right = self.profiles[k+1].evaluate(query)
        return tuple((1-fraction)*a+fraction*b for a, b in zip(left, right))


class DifferentiatedPotential:
    """Consistent scalar-field finite differences for analytic nonradial models.

    The derivative spacing is a recorded numerical parameter; it must receive
    independent refinement controls. It is never a softening length.
    The deformed Plummer bar is an imposed potential family; it is not a
    Ferrers density model or a guarantee of positive density at every radius.
    """
    def __init__(self, record, frame=None):
        self.record = record; self.frame = frame
        self.identity = record['identity']; self.support = record['support_label']; self.category = record['kind']
        if self.category not in {'miyamoto_nagai_disk','rotating_bar','logarithmic_spiral'}:
            raise ValueError('Unsupported differentiated potential')
        self.spacing = positive(record['derivative_spacing_kpc'], 'derivative_spacing_kpc')
        self.minimum_radius = positive(record['minimum_supported_radius_kpc'], 'minimum_supported_radius_kpc')
        self.maximum_radius = positive(record['maximum_supported_radius_kpc'], 'maximum_supported_radius_kpc')
        if self.maximum_radius-self.minimum_radius<=4*self.spacing:
            raise ValueError('Component support must contain the complete derivative stencil')
        if self.spacing*2 >= self.minimum_radius: raise ValueError('Derivative spacing must resolve inner support')
        if self.support == 'simulation_derived' and not record.get('provenance', {}).get('source_hashes'):
            raise ValueError('A measured component needs actual source hashes')
        self.times = increasing(record['times_myr'])
        self.phase = finite(record.get('phase_rad', np.zeros(len(self.times))), self.times.shape)

    def pattern_speed_at(self,t):
        t=float(finite(t,(),'pattern epoch'))
        if t<self.times[0] or t>self.times[-1]: raise ValueError('Pattern speed extrapolation forbidden')
        k=min(len(self.times)-2,max(0,np.searchsorted(self.times,t,side='right')-1))
        return float(np.diff(self.phase)[k]/np.diff(self.times)[k]/KMS_TO_KPC_MYR)

    def _parameter(self, name, t):
        value = finite(self.record[name])
        if value.ndim == 0: return float(value)
        if value.shape != self.times.shape: raise ValueError('Component histories must match times')
        return float(np.interp(t, self.times, value))

    def scalar(self, x, t):
        x=_positions(x);t=float(finite(t,(),'component epoch'))
        if t < self.times[0] or t > self.times[-1]: raise ValueError('Component history extrapolation forbidden')
        if self.frame is not None:
            center, _, rotation = self.frame.state(t); q = (x-center)@rotation
        else: q = x
        R = np.hypot(q[:, 0], q[:, 1]); z = q[:, 2]
        kind = self.record['kind']
        if kind == 'miyamoto_nagai_disk':
            a = positive(self._parameter('radial_scale_kpc', t), 'disk scale')
            b = positive(self._parameter('vertical_scale_kpc', t), 'disk thickness')
            return -G*positive(self._parameter('mass_msun', t), 'disk mass')/np.sqrt(R*R+(a+np.sqrt(z*z+b*b))**2)
        phase = np.interp(t, self.times, self.phase)
        phi = np.arctan2(q[:, 1], q[:, 0])-phase
        if kind == 'rotating_bar':
            # Smooth rotating triaxial Plummer potential; its density need not
            # match a Ferrers bar. Model family remains explicit in provenance.
            axis = finite(self.record['axis_scales_kpc'], (3,))
            if np.any(axis <= 0): raise ValueError('Positive bar axes required')
            xx = q[:, 0]*np.cos(phase)+q[:, 1]*np.sin(phase)
            yy = -q[:, 0]*np.sin(phase)+q[:, 1]*np.cos(phase)
            soft = positive(self.record['softening_kpc'], 'bar core')
            return -G*positive(self._parameter('mass_msun', t),'bar mass')/np.sqrt(soft**2+xx**2+yy**2*(axis[0]/axis[1])**2+z*z*(axis[0]/axis[2])**2)
        if kind == 'logarithmic_spiral':
            arms = self.record['arms']
            if type(arms) is not int or arms < 1: raise ValueError('Positive integer spiral arm number required')
            pitch = self.record['pitch_angle_rad']
            if not 0 < abs(pitch) < np.pi/2: raise ValueError('Supported nonzero spiral pitch required')
            radius = positive(self.record['reference_radius_kpc'], 'spiral reference radius')
            radial_scale = positive(self.record['radial_scale_kpc'], 'spiral radial scale')
            vertical_scale = positive(self.record['vertical_scale_kpc'], 'spiral vertical scale')
            if np.any(R<=0):raise ValueError('Logarithmic spiral is undefined on its cylindrical axis')
            angle = arms*(phi-np.log(R/radius)/np.tan(pitch))
            return self._parameter('amplitude_kms2', t)*np.exp(-R/radial_scale)*np.exp(-.5*(z/vertical_scale)**2)*np.cos(angle)
        raise ValueError('Unsupported differentiated potential')

    def evaluate(self, x, t):
        x = _positions(x); h = self.spacing
        center,_,rotation=self.frame.state(t) if self.frame else (np.zeros(3),None,np.eye(3))
        if self.category=='logarithmic_spiral' and np.any(np.linalg.norm(((x-center)@rotation)[:,:2],axis=1)<=2*h):
            raise ValueError('Spiral derivative stencil touches its unresolved cylindrical axis')
        radius = np.linalg.norm(x-center, axis=1)
        if np.any(radius < self.minimum_radius+2*h) or np.any(radius > self.maximum_radius-2*h):
            raise ValueError('Derivative stencil exits declared component support')
        base = self.scalar(x, t); gradient = np.empty_like(x); hessian = np.empty((len(x), 3, 3))
        for i in range(3):
            e = np.eye(3)[i]*h
            fm2 = self.scalar(x-2*e, t); fm = self.scalar(x-e, t)
            fp = self.scalar(x+e, t); fp2 = self.scalar(x+2*e, t)
            # Fourth-order axial derivatives; mixed derivatives below are second-order.
            gradient[:, i] = (fm2-8*fm+8*fp-fp2)/(12*h)
            hessian[:, i, i] = (-fp2+16*fp-30*base+16*fm-fm2)/(12*h*h)
            for j in range(i):
                f = np.eye(3)[j]*h
                mixed = (self.scalar(x+e+f, t)-self.scalar(x+e-f, t)-self.scalar(x-e+f, t)+self.scalar(x-e-f, t))/(4*h*h)
                hessian[:, i, j] = mixed; hessian[:, j, i] = mixed
        return finite(base,(len(x),),'component potential'),-finite(gradient,x.shape,'component gradient'),finite(hessian,(len(x),3,3),'component Hessian')


class ComponentField:
    def __init__(self, components, *, accounting):
        self.components = dict(components)
        self.accounting = accounting
        if not self.components or set(self.components) != set(accounting):
            raise ValueError('Each actual component requires a matching accounting record')
        for name, entry in accounting.items():
            if type(entry['gravity_count']) is not int or entry['gravity_count'] != 1: raise ValueError('Each gravity source must be included exactly once: '+name)
            if entry['role'] not in {'smooth_host', 'disk', 'satellite', 'dark_candidate', 'bar', 'spiral', 'bulge', 'gas', 'wake'}:
                raise ValueError('Unknown component role')
        source_ids=set()
        for name,entry in accounting.items():
            declared=set(entry.get('native_source_ids',[]))
            if source_ids.intersection(declared): raise ValueError('Native mass source appears in multiple gravity components')
            source_ids.update(declared)
        self.physics = 'imposed_components'

    def evaluate(self, x, t):
        x=_positions(x)
        n = len(x); result = [np.zeros(n), np.zeros((n, 3)), np.zeros((n, 3, 3))]
        for component in self.components.values():
            values=component.evaluate(x,t)
            if len(values)!=3:raise ValueError('Components must return potential, acceleration and Hessian')
            for i,value in enumerate(values):result[i]+=finite(value,result[i].shape,'component field')
        return tuple(result)


class FrozenField:
    def __init__(self, field, epoch): self.field = field; self.epoch = float(finite(epoch,(),'frozen epoch'))
    def evaluate(self, x, t): return self.field.evaluate(x, self.epoch)


class MassScaledField:
    """Explicit fixed-shape imposed-field mass scaling; live response separate."""
    def __init__(self,field,multiplier):self.field=field;self.multiplier=positive(multiplier,'component mass multiplier')
    def evaluate(self,x,t):return tuple(self.multiplier*value for value in self.field.evaluate(x,t))
    def __getattr__(self,name):
        if name in {'phase','times','pattern_speed_at'}:return getattr(self.field,name)
        raise AttributeError(name)


class FixedCenterHost:
    """Freeze only recorded host translation; keep deformation/orientation.

    Evaluating the translating host at shifted query coordinates gives a
    controlled no-reflex field without introducing a fictitious force.
    """
    def __init__(self, host, epoch):
        self.host = host; self.reference_center = host.frame.state(epoch)[0]
    def evaluate(self, x, t):
        center = self.host.frame.state(t)[0]
        return self.host.evaluate(x+center-self.reference_center, t)


def construct_component_control(full, intervention):
    operation = intervention['operation']; names = intervention.get('components', [])
    if len(names)!=len(set(names)):raise ValueError('Intervention components must be distinct')
    if any(name not in full.components for name in names): raise ValueError('Intervention refers to absent component')
    if operation == 'full': return full
    if operation in {'subtract', 'only'}:
        selected = {name: value for name, value in full.components.items() if (name in names) == (operation == 'only')}
    elif operation == 'freeze':
        selected = {name: FrozenField(value, intervention['epoch_myr']) if name in names else value for name, value in full.components.items()}
    elif operation == 'remove_reflex':
        selected = dict(full.components)
        for name in names:
            if not hasattr(selected[name], 'frame'): raise ValueError('Reflex control requires actual translating host frame')
            selected[name] = FixedCenterHost(selected[name], intervention['epoch_myr'])
    elif operation == 'scale_mass':
        multiplier=positive(intervention['mass_multiplier'],'component mass multiplier')
        if not names or multiplier==1:raise ValueError('Mass scaling requires selected components and an actual mass change')
        selected={name:MassScaledField(value,multiplier) if name in names else value for name,value in full.components.items()}
    else: raise ValueError('Unsupported declared intervention')
    if not selected: raise ValueError('Empty controlled field; supply a declared zero field if physically intended')
    return ComponentField(selected, accounting={name: full.accounting[name] for name in selected})


def fourier_torque_diagnostics(position, acceleration, mass, radial_edges):
    x = _positions(position); a = finite(acceleration, x.shape); m = finite(mass, (len(x),))
    if np.any(m<0):raise ValueError('Fourier/torque masses cannot be negative')
    edges = increasing(radial_edges); r = np.hypot(x[:, 0], x[:, 1]); phi = np.arctan2(x[:, 1], x[:, 0])
    harmonic = np.full((len(edges)-1, 4, 2), np.nan); torque = np.full(len(edges)-1, np.nan); support = np.zeros(len(edges)-1, bool)
    for k in range(len(edges)-1):
        use = (r >= edges[k]) & ((r <= edges[k+1]) if k==len(edges)-2 else (r < edges[k+1])) & (m > 0)
        if not use.any(): continue
        support[k] = True; weight=m[use]/m[use].max();weight/=weight.sum()
        for mode in range(1, 5):
            harmonic[k, mode-1] = [np.dot(weight, np.cos(mode*phi[use])), np.dot(weight, np.sin(mode*phi[use]))]
        torque[k] = np.dot(m[use], np.cross(x[use], a[use])[:, 2])
    return {'fourier_m1_m4': harmonic, 'torque_z_msun_kms2': torque, 'fourier_supported': support,
        'interpretation': 'mass Fourier moments and actual z torque; resonances require orbital frequencies'}



class FlattenedLogPotential:
    """Potential-axis model with an analytically implied Poisson density.

    q_phi is explicitly a potential flattening. The implied density generally
    has different radius-dependent axis ratios and can be negative for overly
    flat configurations. Such models are rejected in their declared domain.
    """
    def __init__(self, *, velocity_scale_kms, core_radius_kpc, potential_b_over_a, potential_c_over_a,
                 rotation, minimum_radius_kpc, maximum_radius_kpc):
        self.speed = positive(velocity_scale_kms, 'velocity_scale_kms')
        self.core = positive(core_radius_kpc, 'core_radius_kpc')
        self.b = positive(potential_b_over_a, 'potential_b_over_a')
        self.c = positive(potential_c_over_a, 'potential_c_over_a')
        if not self.c <= self.b <= 1:
            raise ValueError('Ordered potential axes required')
        self.rotation = finite(rotation, (3, 3))
        if not np.allclose(self.rotation.T@self.rotation, np.eye(3)) or not np.isclose(np.linalg.det(self.rotation), 1):
            raise ValueError('Proper orthogonal halo orientation required')
        self.metric = np.diag([1., self.b**-2, self.c**-2])
        self.minimum = positive(minimum_radius_kpc, 'minimum_radius_kpc')
        self.maximum = positive(maximum_radius_kpc, 'maximum_radius_kpc')
        if self.minimum >= self.maximum:
            raise ValueError('Nonempty potential support required')
        # Positivity for all directions in this finite radial domain follows
        # from the smallest numerator coefficient at maximum radius.
        trace = np.trace(self.metric)
        worst = min(trace*m-2*m*m for m in np.diag(self.metric))
        if trace*self.core**2 + min(0., worst)*self.maximum**2 <= 0:
            raise ValueError('Flattened logarithmic potential has nonpositive implied density in its declared domain')

    def evaluate(self, x, t=None):
        x = _positions(x)
        local = x@self.rotation
        radius = np.linalg.norm(local, axis=1)
        if np.any((radius < self.minimum)|(radius > self.maximum)):
            raise ValueError('Halo shape model exceeds support')
        y = local@self.metric
        q = self.core**2+np.sum(local*y, axis=1)
        potential = .5*self.speed**2*np.log(q/self.core**2)
        gradient = self.speed**2*y/q[:, None]
        hessian = self.speed**2*(self.metric[None]/q[:, None, None]-2*y[:, :, None]*y[:, None, :]/q[:, None, None]**2)
        acc = -gradient@self.rotation.T
        hessian = np.einsum('ij,njk,lk->nil', self.rotation, hessian, self.rotation)
        return potential, acc, hessian

    def density(self, x):
        from core.stream_common import G
        return np.trace(self.evaluate(x)[2], axis1=1, axis2=2)/(4*np.pi*G)

    def density_axis_ratios(self, major_radius_kpc):
        from scipy.optimize import brentq
        r = positive(major_radius_kpc, 'major_radius_kpc')
        axis = self.rotation.T
        target = self.density((axis[0]*r)[None])[0]
        radii = [r]
        for direction in axis[1:]:
            def residual(s):
                return self.density((direction*s)[None])[0]-target
            lower = self.minimum*(1+1e-10)
            upper = self.maximum*(1-1e-10)
            if residual(lower)*residual(upper) > 0:
                raise ValueError('Density isosurface not bracketed inside supported model')
            radii.append(brentq(residual, lower, upper))
        ordered = np.sort(radii)[::-1]
        return {'definition': 'density isosurface axes along model principal directions',
            'density_b_over_a': ordered[1]/ordered[0], 'density_c_over_a': ordered[2]/ordered[0],
            'potential_b_over_a': self.b, 'potential_c_over_a': self.c, 'radius_kpc': r}


