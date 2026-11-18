"""Observed-data contracts, covariance fits and immutable estimator products.

All estimator distances are physical Mpc, velocities km/s, times Gyr. No
h-scaled coordinate may cross the observation boundary. Truth is published in
an independent product and is never accepted by an estimator reader.
"""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
import json
import tempfile
import numpy as np
from scipy.linalg import solve_triangular
from .foundation import atomic_json, atomic_text, publish_directory, file_sha256, canonical_digest, C_SI, MPC_CM, SECONDS_PER_YEAR

C_KMS = C_SI / 1000.
MPC_KM = MPC_CM / 1e5
GYR_S = SECONDS_PER_YEAR * 1e9
HUBBLE_GYR = MPC_KM / GYR_S
ARCSEC_RAD = np.pi / (180.0 * 3600.0)
FORBIDDEN = frozenset({'truth', 'H0_truth', 'hubble_param', 'input_H0', 'true_distance',
                       'true_age', 'cosmology_bundle', 'simulation_parameters',
                       'positions_mpc_h', 'coordinates_kpc_h', 'truth_ref'})

def array(value, name, ndim=None, positive=False):
    raw = np.asarray(value)
    if raw.dtype.kind in 'bc':
        raise ValueError(name + ': physical arrays cannot contain Boolean or complex values')
    x = np.asarray(raw, dtype=float)
    if (not np.isfinite(x).all() or (ndim is not None and x.ndim != ndim)
            or (positive and np.any(x <= 0))):
        raise ValueError(f'{name}: invalid dimensions, nonfinite or nonpositive values')
    return x


def exact_ids(values, name='object_id'):
    """Preserve one exact integer or text identity type; never coerce via float.

    Python integer lists spanning the signed-int64 boundary need an explicit
    uint64 dtype. Letting NumPy infer their dtype can round large identities.
    Numeric/text mixtures and Boolean IDs are rejected rather than retagged.
    """
    if isinstance(values, (list, tuple)):
        if not values:
            raw = np.empty(0, dtype=np.uint64)
        elif all(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) for value in values):
            low, high = int(min(values)), int(max(values))
            if -(2**63) <= low and high < 2**63:
                raw = np.asarray(values, dtype=np.int64)
            elif 0 <= low and high < 2**64:
                raw = np.asarray(values, dtype=np.uint64)
            else:
                raise ValueError(name + ' integer identities must fit exact int64 or uint64')
        elif all(isinstance(value, str) for value in values):
            raw = np.asarray(values, dtype=str)
        elif all(isinstance(value, (bytes, np.bytes_)) for value in values):
            raw = np.char.decode(np.asarray(values), 'utf-8')
        else:
            raise ValueError(name + ' requires one exact integer or text identity type')
    else:
        raw = np.asarray(values)
        if raw.dtype.kind == 'S':
            raw = np.char.decode(raw, 'utf-8')
    if raw.ndim != 1 or raw.dtype.kind not in 'iuU':
        raise ValueError(name + ' requires exact integer or string identifiers, never floats')
    if len(np.unique(raw)) != len(raw):
        raise ValueError(name + ' contains duplicate identifiers')
    return raw


def covariance(value, n, name='covariance', semidefinite=False):
    """Validate a symmetric covariance in the declared observable's squared units.

    Semidefinite roundoff tolerance scales with this matrix's eigenvalues, not
    an arbitrary unit-sized floor. A small covariance cannot therefore hide a
    large relative negative variance just because its numerical units are small.
    """
    if type(n) is not int or n < 1 or type(semidefinite) is not bool:
        raise ValueError(name + ' requires a positive dimension and explicit PSD policy')
    matrix = array(value, name, 2)
    if matrix.shape != (n, n):
        raise ValueError(name + ' must have matching square shape')
    scale = np.max(np.abs(matrix))
    normalized = matrix / scale if scale else matrix
    if not np.allclose(normalized, normalized.T, rtol=1e-10, atol=1e-14):
        raise ValueError(name + ' must be symmetric relative to its own variance scale')
    # Check the symmetric part in dimensionless units, but return the original
    # supplied entries. Validation must not rewrite a measured covariance;
    # averaging raw subnormal entries can also erase a valid tiny variance.
    symmetric = .5 * normalized + .5 * normalized.T
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if semidefinite:
        invalid = eigenvalues[0] < -1e-12 * np.max(np.abs(eigenvalues))
    else:
        invalid = eigenvalues[0] <= 0
    if invalid:
        raise ValueError(name + ' must be positive ' + ('semidefinite' if semidefinite else 'definite'))
    return matrix


def canonical(value):
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, np.generic):
            raise TypeError('Extended-precision JSON scalars require an explicit lossless representation')
        return canonical(scalar)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def fingerprint(value):
    """Use the shared deterministic JSON digest after scientific-type conversion."""
    return canonical_digest(canonical(value))


def reject_truth(value, trail='observation'):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN or str(key).startswith(('truth_', 'native_', 'true_')):
                raise PermissionError(f'Estimation boundary rejects {trail}.{key}')
            reject_truth(item, trail + '.' + str(key))
    elif isinstance(value, (tuple, list)):
        for item in value:
            reject_truth(item, trail)


def observed_catalogue(columns, *, units, selection, namespace, covariance_matrix=None,
                       calibration=None, provenance=None):
    reject_truth(columns)
    reject_truth(selection)
    reject_truth(calibration or {})
    if not namespace:
        raise ValueError('An exact identity namespace is required')
    ids = exact_ids(columns['object_id'])
    for name, values in columns.items():
        if len(values) != len(ids):
            raise ValueError('Column length differs: ' + name)
        if name not in units:
            raise ValueError('Missing explicit units for ' + name)
    if any('/h' in str(u) or 'h^-1' in str(u) for u in units.values()):
        raise ValueError('Estimator units must not contain native h scaling')
    result = {'schema': 'cosmography-observations-v1', 'role': 'estimator_observations',
              'namespace': namespace, 'columns': canonical(columns), 'units': dict(units),
              'selection': canonical(selection), 'calibration': canonical(calibration or {}),
              'provenance': canonical(provenance or {})}
    if covariance_matrix is not None:
        result['covariance'] = covariance(covariance_matrix, len(ids)).tolist()
    reject_truth(result)
    return result


def read_observations(source):
    product = json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else source
    reject_truth(product)
    def hidden_input(value):
        if isinstance(value,Mapping):
            for key,item in value.items():
                if key in {'H0','H0_kms_mpc','H_kms_mpc','h','input_h','z_cosmological','cosmic_time_gyr'}:
                    raise PermissionError('Observed estimator input contains hidden background field: '+key)
                hidden_input(item)
        elif isinstance(value,(tuple,list)):
            for item in value:hidden_input(item)
    hidden_input(product)
    if product.get('role') != 'estimator_observations':
        raise PermissionError('Only estimator_observations may enter a fit')
    if 'columns' in product:
        exact_ids(product['columns']['object_id'])
    return product


def gls(design, data, cov, *, parameter_names=None):
    """Fit y=A*p+noise with the declared observation covariance C.

    With C=L*L.T, whiten B=L^-1*A and b=L^-1*y, then solve using B's SVD.
    This avoids forming normal equations, which square the design's condition
    number. Parameter covariance is V*diag(1/s^2)*V.T; influence maps original
    observations to fitted parameters. C is known, so covariance is not scaled
    by the fitted chi-square. Units follow the supplied data/design columns.
    """
    matrix, values = array(design, 'design', 2), array(data, 'data', 1)
    count, parameters = matrix.shape
    if count != len(values) or not parameters or parameters > count:
        raise ValueError('GLS design is empty, underdetermined or has incompatible shape')
    names = [str(i) for i in range(parameters)] if parameter_names is None else list(parameter_names)
    if len(names) != parameters or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError('GLS parameter names must be distinct and match design columns')
    cholesky = np.linalg.cholesky(covariance(cov, count))
    whitened = solve_triangular(cholesky, matrix, lower=True)
    whitened_values = solve_triangular(cholesky, values, lower=True)
    left, singular, right = np.linalg.svd(whitened, full_matrices=False)
    tolerance = np.finfo(float).eps * max(whitened.shape) * singular[0]
    if np.any(singular <= tolerance):
        raise ValueError('Calibration/design parameters are not identifiable')
    scaled_right = right.T / singular
    mean = array(scaled_right @ (left.T @ whitened_values), 'fitted parameters', 1)
    parameter_covariance = array(scaled_right @ scaled_right.T, 'parameter covariance', 2)
    residual = array(values - matrix @ mean, 'fit residual', 1)
    influence = array(solve_triangular(cholesky, (scaled_right @ left.T).T, lower=True, trans='T').T, 'parameter influence', 2)
    whitened_residual = solve_triangular(cholesky, residual, lower=True)
    chi2 = float(whitened_residual @ whitened_residual)
    if not np.isfinite(chi2):
        raise ValueError('GLS chi-square exceeds the representable numeric range')
    return {'mean': mean, 'covariance': parameter_covariance, 'residual': residual,
            'chi2': chi2, 'dof': count - parameters, 'influence': influence, 'parameter_names': names}




def freeze_result(method, fit, *, observables, selection, object_ids=(), namespace='',
                  nuisance=None, model=None, realization_id=None, environment=None,
                  calibration_ids=(), acceptance=None):
    if not isinstance(method, str) or not method.strip():
        raise ValueError('Frozen estimator method must be named explicitly')
    identifiers = exact_ids(object_ids)
    if len(identifiers) and (not isinstance(namespace, str) or not namespace.strip()):
        raise ValueError('Frozen object identities require an explicit namespace')
    payload = {'schema': 'cosmography-frozen-result-v1', 'role': 'frozen_estimator_result',
               'method': method, 'fit': canonical(fit), 'observables': canonical(observables),
               'selection': canonical(selection), 'object_ids': list(map(str, identifiers)),
               'namespace': namespace, 'nuisance': canonical(nuisance or {}),
               'model': canonical(model or {}), 'realization_id': realization_id,
               'environment': canonical(environment or {}), 'calibration_ids': list(calibration_ids),
               'execution_status': 'COMPLETED',
               'scientific_acceptance': acceptance or 'UNRESOLVED_NOT_EXECUTED',
               'validation_receipts': []}
    reject_truth(payload)
    payload['frozen_hash'] = fingerprint(payload)
    return payload


def read_frozen(source):
    p = json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else dict(source)
    if p.get('schema') != 'cosmography-frozen-result-v1' or p.get('role') != 'frozen_estimator_result' or p.get('execution_status') != 'COMPLETED':
        raise PermissionError('Evaluation requires a completed frozen estimator result')
    reject_truth(p)
    h = p.pop('frozen_hash', None)
    if not h or fingerprint(p) != h:
        raise ValueError('Frozen estimator product changed after fitting')
    p['frozen_hash'] = h
    return p


@dataclass(frozen=True)
class EvaluationReader:
    """Explicit privileged capability; instantiate only in post-fit evaluation."""
    path: Path
    expected_hash: str

    def load_truth(self, frozen_results):
        if not frozen_results:
            raise PermissionError('Truth access requires at least one frozen estimator result')
        for result in frozen_results:
            read_frozen(result)
        if file_sha256(self.path) != self.expected_hash:
            raise ValueError('Truth product hash differs')
        truth = json.loads(self.path.read_text())
        if truth.get('role') != 'privileged_expansion_truth':
            raise PermissionError('Not an expansion truth product')
        return truth


def publish_product(destination, payload, *, inputs=(), request=None, files=None):
    """Immutable generation using existing durable publication helpers.

    A fingerprint includes input bytes, request and all first-party cosmography
    module hashes. Integrity checks here protect publication, not science acceptance.
    """
    for name in files or {}:
        artifact = Path(name)
        if artifact.is_absolute() or '..' in artifact.parts or artifact.as_posix() in {'product.json', 'numeric.hdf5', 'manifest.json', '.'}:
            raise ValueError('Artifact path escapes or replaces a reserved generation product')
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_hashes = {str(Path(p).resolve()): file_sha256(p) for p in inputs}
    # Bind the ordinary comparison/figure owner as well as remaining estimator
    # sections. Moving an implementation must not drop it from frozen products.
    sources = sorted(set(Path(__file__).parent.glob('*cosmograph*.py')) |
                     {Path(__file__).parent / n for n in ('observations.py', 'observation_data.py', 'stellar_observations.py', 'expansion_products.py', 'foundation.py', 'data.py',
                         'halos.py', 'native.py', 'cosmology.py', 'bao.py', 'initial_conditions.py', 'stream/inference.py')})
    sources.extend([Path(__file__).parent/'__init__.py', Path(__file__).parent.parent/'configs/core_source_layout.json'])
    code = {str(p.resolve().relative_to(Path(__file__).parent.parent)): file_sha256(p) for p in sources}
    pin_path = Path(__file__).parent.parent / 'configs/cosmography.upstream.json'
    if pin_path.is_file():
        input_hashes[str(pin_path.resolve())] = file_sha256(pin_path)
    sig = fingerprint({'inputs': input_hashes, 'request': request, 'code': code})
    if destination.exists():
        manifest = json.loads((destination / 'manifest.json').read_text())
        if manifest['fingerprint'] != sig:
            raise FileExistsError('Existing immutable generation belongs to a different request')
        for name, digest in manifest['files_sha256'].items():
            if file_sha256(destination / name) != digest:
                raise ValueError('Published product integrity failure: ' + name)
        if fingerprint(json.loads((destination/'product.json').read_text())) != fingerprint(payload):
            raise FileExistsError('Existing immutable generation contains a different scientific product')
        return manifest
    # Failed unpublished generations remain available; no cleanup hides them.
    stage = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
    atomic_json(stage / 'product.json', canonical(payload))
    write_numeric_companion(stage / 'numeric.hdf5',payload)
    for name, content in (files or {}).items():
        atomic_text(stage / name, content)
    manifest = {'schema': 'cosmography-generation-v1', 'fingerprint': sig,
                'matched_experiment': (request or {}).get('matched_experiment'),
                'input_hashes': input_hashes, 'source_hashes': code,
                'request': canonical(request), 'execution_status': 'COMPLETED',
                'scientific_acceptance': payload.get('scientific_acceptance', 'UNRESOLVED_NOT_EXECUTED'),
                'validation_receipts': [], 'files_sha256':
                {str(p.relative_to(stage)): file_sha256(p) for p in stage.rglob('*') if p.is_file()}}
    atomic_json(stage / 'manifest.json', manifest)
    publish_directory(stage, destination)
    return manifest


def gaussian_observable(value, variance, units, redshift=None):
    if not np.isfinite(value) or not np.isfinite(variance) or variance <= 0:
        raise ValueError('Observable mean/variance must be finite with positive variance')
    sigma = float(np.sqrt(variance))
    return {'value': float(value), 'sigma': sigma, 'units': units,
            'interval_68': [float(value - sigma), float(value + sigma)],
            'redshift': redshift}


def activate_pinned_backend(name):
    """Load the source-bound installed wheel, including its compiled extensions."""
    import importlib
    from core.native import verified_python_backend
    project = Path(__file__).parent.parent
    pins = json.loads((project / 'configs/cosmography.upstream.json').read_text())
    record = pins['repositories'][name]
    declarations = json.loads((project / 'configs/runtime/dependencies.json').read_text())
    matches = [item for group in declarations['profiles'].values() for item in group['packages']
               if item.get('name') == name]
    if len(matches) != 1 or matches[0]['revision'] != record['commit']:
        raise RuntimeError('Backend declaration differs from scientific pin: ' + name)
    declaration = matches[0]
    binding = verified_python_backend(name, {'revision': record['commit'],
        'directory': declaration['source_path'], 'source_sha256': declaration['source_sha256']})
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError('Pinned '+name+' requires its documented runtime dependencies') from exc
    origin = getattr(module, '__file__', None)
    if origin is None or str(Path(origin).resolve()) not in binding['installed_assets']:
        raise RuntimeError('Imported '+name+' is outside its verified wheel; use a fresh process')
    return module, record


def write_numeric_companion(path,product):
    """Attach the existing GADGET dataset contract to scientific numeric arrays.

    Heterogeneous parameter axes are named in product.json and repeated in
    attributes. Their physical units depend on the estimator/model; this writer
    does not invent a single physical unit for a mixed-parameter covariance.
    """
    import h5py
    from core.data import write_dataset_contract
    fit=product.get('fit',{})
    product_hash=fingerprint(product)
    with h5py.File(path,'w') as output:
        output.attrs['schema']='cosmography-numeric-companion-v1'
        output.attrs['role']=product.get('role','scientific_product')
        output.attrs['product_hash']=product_hash
        output.attrs['parameter_names_json']=json.dumps(fit.get('parameter_names',[]))
        output.attrs['parameter_units_json']=json.dumps(fit.get('parameter_units',[]))
        output.attrs['scientific_acceptance']='UNRESOLVED_NOT_EXECUTED'
        arrays={}
        for key,unit in [('redshift','1'),('H_kms_mpc','km/s/Mpc'),('chi_mpc','Mpc'),
                         ('lookback_gyr','Gyr'),('cosmic_time_gyr','Gyr'),
                         ('luminosity_distance_mpc','Mpc')]:
            if key in product:arrays[key]=(product[key],unit)
        for key,value in fit.items():
            if key in ('mean','covariance','residual','influence') and value is not None:
                arrays['fit/'+key]=(value,'method-specific parameter-axis units; names in product.json' if key!='residual'
                    else 'observed data units declared in product.json')
            elif key in ('D_dt_mpc_samples','D_dt_samples','H0_samples','H_z_covariance','ratio_covariance'):
                arrays['fit/'+key]=(value,'Mpc' if key.startswith('D_dt') else 'km/s/Mpc' if key=='H0_samples'
                                       else '(km/s/Mpc)^2' if key=='H_z_covariance' else '1')
        for key,observable in product.get('observables',{}).items():
            arrays['observables/'+key+'/mean']=([observable['value']],observable['units'])
            arrays['observables/'+key+'/sigma']=([observable['sigma']],observable['units'])
            arrays['observables/'+key+'/interval_68']=(observable['interval_68'],observable['units'])
        for key,(value,unit) in arrays.items():
            data=array(value,key)
            kwargs={'chunks':True,'fletcher32':True,'compression':'gzip'} if data.ndim and data.size else {}
            ds=output.create_dataset(key,data=data,**kwargs)
            write_dataset_contract(ds,units=unit,quantity=key,
                coordinates={'distance_scaling':'no h factors','frame':'declared observer or inference parameter frame'},
                epoch={'redshift':product.get('observables',{}).get(key.split('/')[1] if '/' in key else '',{}).get('redshift')},
                provenance={'product_hash':product_hash,'method':product.get('method'),
                            'parameter_names':fit.get('parameter_names',[]),'parameter_units':fit.get('parameter_units',[])},quality='NOT_ACCEPTED')
