"""Shared stream contracts; all lengths proper kpc, velocities km/s, times Myr.

Native conversions remain in data.py. Resimulation particles are tracers with
separate namespaces, never new native stellar populations.
"""

from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
from core.foundation import G_MPC_KMS2_MSUN, KPC_CM, SECONDS_PER_YEAR, file_sha256 as digest
from core.data import exact_uint64_ids, export_numeric_audit, coordinate_contract, verify_numeric_audit

# G changes from Mpc*(km/s)^2/Msun to proper kpc*(km/s)^2/Msun.
G = G_MPC_KMS2_MSUN * 1000
# One km/s sustained for one Myr, expressed in proper kpc.
KMS_TO_KPC_MYR = 1e5 * SECONDS_PER_YEAR * 1e6 / KPC_CM


def finite(value, shape=None, name='array'):
    a = np.asarray(value, dtype=float)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError(name + ' requires finite arrays of the declared shape')
    return a


def positive(value, name):
    scalar = np.asarray(value)
    if scalar.shape != () or scalar.dtype.kind not in 'fiu' or not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(name + ' must be a finite positive numeric scalar')
    return float(scalar)


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an explicit integer >= ' + str(minimum))
    return value


def increasing(value, name='times', minimum=2):
    a = finite(value, name=name)
    if a.ndim != 1 or len(a) < minimum or np.any(np.diff(a) <= 0):
        raise ValueError(name + ' must be strictly increasing')
    return a


def phase_space(sample):
    x = finite(sample['position_kpc'], name='position_kpc')
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError('Positions require N by 3 Cartesian proper kpc')
    v = finite(sample['velocity_kms'], x.shape, 'velocity_kms')
    m = finite(sample['mass_msun'], (len(x),), 'mass_msun')
    ids = exact_uint64_ids(sample['particle_ids'], allow_empty=True)
    if len(ids) != len(x) or np.any(m <= 0):
        raise ValueError('Unique exact identities and positive measured masses required')
    namespace = sample.get('namespace')
    if not isinstance(namespace,str) or not namespace.strip():
        raise ValueError('Explicit particle namespace required')
    return x, v, m, ids


def weighted_covariance(x, weight):
    """Return the weighted population mean, covariance and effective count.

    For p_i=w_i/sum(w), C=sum p_i (x_i-mean)(x_i-mean)^T and
    N_eff=1/sum(p_i^2). C has squared input units. This describes the
    measured population; it is neither an unbiased sample estimator nor
    the covariance of its mean. Zero total support returns NaNs and N_eff=0.
    """
    x = finite(x,name='covariance values')
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError('Covariance values require an N by D matrix')
    w = finite(weight,(len(x),),'covariance weights')
    if np.any(w < 0):
        raise ValueError('Population covariance weights cannot be negative')
    if not len(x) or not np.any(w > 0):
        dimension = x.shape[1]
        return np.full(dimension,np.nan), np.full((dimension,dimension),np.nan), 0.
    # A common positive weight scale cancels. Divide first so neither sum(w)
    # nor sum(w^2) overflows/underflows solely because of mass-unit scale.
    scaled = w/w.max()
    probability = scaled/scaled.sum()
    with np.errstate(over='ignore',invalid='ignore'):
        mean = np.sum(x*probability[:,None],axis=0)
        residual = x-mean
        covariance = np.einsum('n,ni,nj->ij',probability,residual,residual)
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError('Population covariance exceeds the representable numeric range')
    return mean, covariance, float(1/np.dot(probability,probability))


def normalize(x, floor=1e-14):
    """Return unit vectors and explicit support; unresolved vectors stay zero."""
    x = np.asarray(x,float)
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError('Normalization requires nonempty Cartesian vector axes')
    floor = positive(floor,'normalization floor')
    # hypot avoids overflowing a sum of squares when the norm itself fits.
    norm = np.hypot.reduce(x,axis=-1,keepdims=True)
    supported = np.isfinite(norm) & (norm > floor)
    unit = np.divide(x,norm,out=np.zeros_like(x),where=supported)
    return unit, supported[...,0]


def seeded_rng(seed, *labels):
    integer(seed, 'seed', 0)
    payload = json.dumps([seed, *labels], sort_keys=True, separators=(',', ':')).encode()
    return np.random.Generator(np.random.Philox(int.from_bytes(hashlib.sha256(payload).digest()[:16], 'big')))


def json_safe(value):
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, np.generic): return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value


def publish_stream_product(destination, report, arrays, units, *, sources, namespace, epoch=None):
    """Use the existing immutable numeric publisher with per-array semantics."""
    sources = dict(sources)
    if report.get('matched_experiment') is None and isinstance(report.get('request'), dict):
        report = {**report, 'matched_experiment': report['request'].get('matched_experiment')}
    sources[str(Path(__file__).resolve())] = digest(__file__)
    support_source=(Path(__file__).parent/'stream'/'measurements.py').resolve()
    sources[str(support_source)]=digest(support_source)
    coordinates = coordinate_contract(position_units='proper kpc',
        velocity_convention=report.get('velocity_convention','km/s; declared host or inertial frame'),
        origin=report.get('origin', 'host-centered proper Cartesian'),
        periodic_wrapping=report.get('periodic_wrapping','already unwrapped; no periodic resimulation'))
    specifications = {}
    for key in arrays:
        if key not in units: raise ValueError('Missing units for ' + key)
        row = {'units': units[key], 'quantity': key,
               'missing_policy': 'NaN and explicit support masks for unresolved measurements'}
        if 'comoving' in key:
            row['coordinates']=coordinate_contract(position_units=units[key],velocity_convention='peculiar km/s when supplied; no Hubble flow added',origin=report.get('origin','native simulation Cartesian origin'),periodic_wrapping='recorded images or native observer-relative cone coordinates')
        if key.endswith('particle_ids'):
            row['identity_namespace'] = report.get('parent_particle_namespace',namespace) if key=='parent_particle_ids' else namespace
        specifications[key] = row
    return export_numeric_audit(destination, json_safe({**report, 'namespace': namespace}), arrays, specifications,
        algorithm=__file__, context={'epoch': epoch, 'coordinates': coordinates}, source_hashes=sources)


def read_product(path, *, memory_gib):
    """Read verified arrays within a declared expanded-data workspace.

    The factor of three reserves space for decoding and caller reductions.
    It is an array-workspace estimate, not a bound on total process RSS.
    """
    budget = positive(memory_gib,'memory_gib')*1024**3
    path = Path(path).resolve()
    archive = path.with_name(path.name+'.data')/'arrays.npz'
    if path.stat().st_size*8 > budget:
        raise MemoryError('Stream product metadata exceeds declared memory')
    bindings = {str(path):digest(path),str(archive):digest(archive)}
    with zipfile.ZipFile(archive) as packed:
        members = packed.infolist()
        names = [member.filename for member in members]
        if len(set(names)) != len(names) or any(not name.endswith('.npy') or '/' in name or '\\' in name for name in names):
            raise ValueError('Stream product requires distinct flat numeric NPY members')
        if sum(member.file_size for member in members)*3 > budget:
            raise MemoryError('Stream input arrays exceed declared memory')
    report = verify_numeric_audit(path)
    with np.load(archive,allow_pickle=False) as packed:
        arrays = {name:packed[name] for name in packed.files}
    for source,checksum in bindings.items():
        if digest(source) != checksum:
            raise ValueError('Stream product changed while its arrays were read')
    return report, arrays, bindings


def stream_array_units(arrays):
    """Explicit suffix semantics for the fixed numerical producer vocabulary."""
    units={}
    for key in arrays:
        if key=='training_theta':
            unit='mixed; recorded parameter_priors column units'
        elif key=='observable_features':
            unit='dimensionless normalized observables/covariance/masks; encoder contract in report'
        elif key.endswith('number_density_mpc3_dex'):
            unit='count / comoving Mpc^3 / dex no h'
        elif key.endswith('number_density_mpc3'):
            unit='count / comoving Mpc^3 no h'
        elif key.endswith('kelvin'):
            unit='K'
        elif key.endswith('g_cm2'):
            unit='g / cm^2'
        elif key.endswith('year_inverse'):
            unit='1 / year'
        elif key.endswith('sfr_msun_year'):
            unit='Msun / year'
        elif key.endswith('surface_density_msun_kpc2'):
            unit='Msun / proper kpc^2'
        elif key.endswith('spectrum_lsun_per_angstrom'):
            unit='FSPS Lsun / Angstrom'
        elif key.endswith('luminosity_lsun'):
            unit='FSPS Lsun'
        elif key.endswith('angstrom'):
            unit='Angstrom'
        elif key.endswith('jy2'):
            unit='Jy^2'
        elif key.endswith('jy'):
            unit='Jy'
        elif key.endswith('masyr'):
            unit='mas/year'
        elif key in {'ra_deg','dec_deg'}:
            unit='degree'
        elif 'posterior_samples' in key or key=='chain':
            unit='mixed; per-column parameter_priors units'
        elif 'observed_phase_space' in key or 'log_mass_sensitivity' in key:
            unit='mixed; declared observer phase-space column units'
        elif 'paired_mean_covariance' in key:
            unit='mixed; Cartesian kpc and km/s covariance, coordinate order in report'
        elif 'torque_kms2' in key:
            unit='(km/s)^2'
        elif key.endswith('particle_ids'):
            unit='identity'
        elif key.endswith('msun_kms2'):
            unit='Msun (km/s)^2'
        elif key.endswith('msun_kpc'):
            unit='Msun / proper kpc'
        elif key.endswith('kpc2_kms2'):
            unit='proper kpc^2 (km/s)^2'
        elif key.endswith('kpc_kms'):
            unit='proper kpc km/s'
        elif key.endswith('kms_kpc'):
            unit='km/s / proper kpc'
        elif key.endswith('kms2'):
            unit='(km/s)^2'
        elif key.endswith('kpc3'):
            unit='proper kpc^3'
        elif key.endswith('kpc2'):
            unit='proper kpc^2'
        elif key.endswith('kpc_inverse'):
            unit='1 / proper kpc'
        elif key.endswith('myr_inverse'):
            unit='1 / Myr'
        elif key.endswith('kpc') or key.endswith(('curve_grid','curve_parameter')):
            unit='proper kpc'
        elif key.endswith('kms'):
            unit='km/s'
        elif key.endswith('msun'):
            unit='Msun'
        elif key.endswith('myr'):
            unit='Myr'
        elif key.endswith('rad'):
            unit='radian'
        else:unit='dimensionless'
        units[key]=unit
    return units
