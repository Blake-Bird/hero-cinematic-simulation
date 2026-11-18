"""Static-fixture contracts for the native-to-static host reference bridge.

These are numerical and provenance controls, not a GADGET or stream simulation.
All paths and source bytes are created under pytest's disposable tmp_path.
"""
from copy import deepcopy
from decimal import Decimal, localcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.foundation import file_sha256
from core.stream.host_reference import native_nfw_reference
from core.stream.workflow import (
    _bind_reference_sources,
    _check_backend_host_force,
    _require_native_reference_units,
)
from core.stream_common import G


def _unit_reports():
    native_units = {'particle_ids': 'identity', 'position_kpc': 'proper kpc',
                    'velocity_kms': 'km / s', 'mass_msun': 'Msun',
                    'times_myr': 'Myr', 'periodic_box_proper_kpc': 'proper kpc'}
    frame_units = {**native_units, 'origin_proper_kpc': 'proper kpc',
                   'origin_velocity_peculiar_kms': 'km / s',
                   'hubble_kms_mpc': 'km / s / Mpc'}
    def datasets(units):
        result = {name: {'units': unit} for name, unit in units.items()}
        result['particle_ids']['identity_namespace'] = 'native:stars'
        return {'array_contract': {'datasets': result}}
    return ({'namespace': 'native:stars', 'numeric_data': datasets(native_units)},
            {'numeric_data': datasets(frame_units)})


def _case(tmp_path, radii=(1., 2., 4.), *, rs=10.):
    epoch = {'snapshot_storage_index': 7, 'scale_factor': .5}
    fit = {'status': 'MEASURED', 'model': 'NFW log-density least squares with mass-shot-noise weights',
           'definition': {'fit_min_kpc': min(radii) / 2, 'fit_max_kpc': max(radii) * 2},
           'components': {'total': {'status': 'MEASURED', 'rs_proper_kpc': rs,
                                    'rho_s_msun_kpc3': 1e7}}}
    run = str(tmp_path / 'native-run')
    structure = {'schema': 'native-halo-structure-v1', 'source_origin': 'native_snapshot',
                 'field_validation_status': 'PASS', 'request': {'run_directory': run},
                 'source_context': {'epoch': epoch, 'center_comoving_kpc_h': [14., 28., 42.],
                                    'scale_factor': .5, 'h': .7}, 'concentration': fit}
    frame = {'schema': 'native-encounter-frame-history-v1', 'source_origin': 'native_snapshot_and_catalogue',
             'source_history_sha256': 'native-hash', 'epochs': [epoch], 'namespace': 'native:stars',
             'frame': 'host-fixed', 'frame_definition_sha256': 'frame-definition'}
    native = {'schema': 'measured-native-stream-worldlines-v1', 'execution_status': 'COMPLETE',
              'request': {'species': 4}, 'epochs': [epoch], 'namespace': 'native:stars',
              'native_run_directory': run}
    history = {'origin_proper_kpc': np.array([[10., 20., 30.]])}
    x = np.column_stack((np.asarray(radii), np.zeros((len(radii), 2))))
    mass = 4 * np.pi * 1e7 * rs**3
    # High-precision reference avoids reproducing the implementation's small-u
    # subtraction or its series approximation.
    with localcontext() as ctx:
        ctx.prec = 60
        force = []
        for radius in radii:
            u = Decimal(str(radius)) / Decimal(str(rs))
            shape = (Decimal(1) + u).ln() - u / (Decimal(1) + u)
            force.append(-G * mass * float(shape) / radius**2)
    data = {'validation_query_positions_proper_kpc': x,
            'validation_acceleration_direct_km2_s2_kpc': np.column_stack((force, np.zeros((len(radii), 2))))}
    controls = {'index': 0, 'structure_sha256': 'structure-hash', 'frame_sha256': 'frame-hash',
                'native_sha256': 'native-hash', 'force_relative_tolerance': 1e-8,
                'force_floor_km2_s2_kpc': 1e-30}
    return structure, data, frame, history, native, controls


def test_nfw_force_preserves_tiny_radius_and_native_bindings(tmp_path):
    case = _case(tmp_path, (1e-8, 2e-8, 3e-8))
    report, arrays = native_nfw_reference(*case[:5], **case[5])
    assert report['status'] == 'PASS'
    assert report['frame_history_sha256'] == 'frame-hash'
    assert report['native_history_sha256'] == 'native-hash'
    np.testing.assert_allclose(arrays['model_acceleration_kms2_per_kpc'],
                               arrays['direct_acceleration_kms2_per_kpc'], rtol=1e-8, atol=0.)


@pytest.mark.parametrize('mutation, message', [
    ('origin', 'origin'), ('epoch', 'epoch'), ('run', 'same run'),
    ('native_hash', 'exact frame conversion'), ('species', 'native stars'),
    ('fit_status', 'resolved total-density'), ('host_status', 'passed reconstructed-field'),
])
def test_nfw_rejects_mismatched_scientific_context(tmp_path, mutation, message):
    structure, data, frame, history, native, controls = deepcopy(_case(tmp_path))
    if mutation == 'origin': history['origin_proper_kpc'][0, 0] += 1.
    elif mutation == 'epoch': structure['source_context']['epoch']['snapshot_storage_index'] = 8
    elif mutation == 'run': native['native_run_directory'] = str(tmp_path / 'different-run')
    elif mutation == 'native_hash': frame['source_history_sha256'] = 'different'
    elif mutation == 'species': native['request']['species'] = 1
    elif mutation == 'fit_status': structure['concentration']['components']['total']['status'] = 'FIT_UNRESOLVED'
    else: structure['field_validation_status'] = 'UNRESOLVED'
    with pytest.raises(ValueError, match=message):
        native_nfw_reference(structure, data, frame, history, native, **controls)


def test_native_frame_requires_physical_units_and_identity_namespace():
    native, frame = _unit_reports()
    _require_native_reference_units(native, frame)
    frame['numeric_data']['array_contract']['datasets']['velocity_kms']['units'] = 'native code velocity'
    with pytest.raises(ValueError, match='frame boundary unit'):
        _require_native_reference_units(native, frame)
    native, frame = _unit_reports()
    frame['numeric_data']['array_contract']['datasets']['particle_ids']['identity_namespace'] = 'other:stars'
    with pytest.raises(ValueError, match='identity namespace'):
        _require_native_reference_units(native, frame)


def test_reference_sources_recheck_payload_and_detect_conflicts(tmp_path):
    path = tmp_path / 'source.dat'
    path.write_bytes(b'original source')
    checksum = file_sha256(path)
    provenance = json.dumps({'sources': {str(path): checksum}})
    report = {'numeric_data': {'array_contract': {'datasets': {'position': {'provenance_json': provenance}}}}}
    inputs = SimpleNamespace(sources={})
    _bind_reference_sources(inputs, report)
    assert inputs.sources[str(path)] == checksum
    with pytest.raises(ValueError, match='Conflicting source generations'):
        _bind_reference_sources(SimpleNamespace(sources={str(path): 'different'}), report)
    path.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='source changed'):
        _bind_reference_sources(SimpleNamespace(sources={}), report)


def test_backend_force_budget_pass_and_retained_failure(tmp_path):
    host = {'force_relative_tolerance': .1, 'force_floor_km2_s2_kpc': 1.}
    queries = np.eye(3)
    direct = 10. * np.eye(3)
    saved = {'query_positions_kpc': queries, 'direct_acceleration_kms2_per_kpc': direct,
             'force_allowed_kms2_per_kpc': np.ones(3)}
    summary, arrays = _check_backend_host_force(host, saved, direct + .2 * np.eye(3), tmp_path, 'control')
    assert summary['status'] == 'PASS'
    np.testing.assert_allclose(arrays['backend_force_error_kms2_per_kpc'], .2)
    with pytest.raises(ValueError, match='exceeded the frozen withheld budget'):
        _check_backend_host_force(host, saved, direct + 2. * np.eye(3), tmp_path, 'control')
    assert json.loads((tmp_path / 'backend-force-failure.json').read_text())['status'] == 'FAIL'
    with np.load(tmp_path / 'backend-force-failure.npz', allow_pickle=False) as archive:
        np.testing.assert_allclose(archive['backend_acceleration_kms2_per_kpc'], direct + 2. * np.eye(3))
        np.testing.assert_allclose(archive['allowed_kms2_per_kpc'], 1.)
