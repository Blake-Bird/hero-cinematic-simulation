"""Source-bound spherical NFW references from a measured native host.

This is a restricted static approximation, not an evolving-host reconstruction.
The density fit and independent direct-force samples are produced by the native
halo-structure workflow; no optimization uses those force-validation samples.
"""
import numpy as np

from core.stream_common import G, finite, positive


def native_nfw_reference(structure, data, frame, history, native, *, index,
                         structure_sha256, frame_sha256, native_sha256,
                         force_relative_tolerance, force_floor_km2_s2_kpc):
    if (structure.get('schema') != 'native-halo-structure-v1'
            or structure.get('source_origin') != 'native_snapshot'
            or structure.get('field_validation_status') != 'PASS'):
        raise ValueError('Require a native host structure with passed reconstructed-field controls')
    if (frame.get('schema') != 'native-encounter-frame-history-v1'
            or frame.get('source_origin') != 'native_snapshot_and_catalogue'
            or frame.get('source_history_sha256') != native_sha256
            or native.get('schema') != 'measured-native-stream-worldlines-v1'
            or native.get('execution_status') != 'COMPLETE'):
        raise ValueError('Require the actual native stellar history and its exact frame conversion')
    if native.get('request', {}).get('species') != 4:
        raise ValueError('The reference cohort must be native stars')
    if type(index) is not int or index < 0 or index >= len(frame['epochs']):
        raise ValueError('Choose an explicit saved frame epoch')
    if frame['epochs'] != native['epochs'] or frame['namespace'] != native['namespace']:
        raise ValueError('Frame and native identity/clock do not match')
    from pathlib import Path
    if Path(native['native_run_directory']).resolve() != Path(structure['request']['run_directory']).resolve():
        raise ValueError('Native host and stellar history must come from the same run')
    epoch = frame['epochs'][index]
    context = structure['source_context']
    if epoch != context['epoch']:
        raise ValueError('Host fit and stellar reference must use the identical saved epoch')
    center = finite(context['center_comoving_kpc_h'], (3,), 'host centre') * context['scale_factor'] / context['h']
    origin = finite(history['origin_proper_kpc'], (len(frame['epochs']), 3), 'frame origins')[index]
    # A different periodic image is not assumed equivalent: it must first be
    # explicitly aligned by the source frame producer.
    if not np.allclose(origin, center, rtol=0., atol=1e-7):
        raise ValueError('Frozen frame origin does not equal the native host centre in proper kpc')
    fit = structure.get('concentration', {})
    row = fit.get('components', {}).get('total', {})
    if fit.get('model') != 'NFW log-density least squares with mass-shot-noise weights' or fit.get('status') != 'MEASURED' or row.get('status') != 'MEASURED':
        raise ValueError('Require a resolved total-density NFW concentration fit')
    rs = positive(row['rs_proper_kpc'], 'NFW scale radius')
    rho = positive(row['rho_s_msun_kpc3'], 'NFW scale density')
    mass = 4 * np.pi * rho * rs**3
    if not np.isfinite(mass):
        raise ValueError('NFW scale mass is not finite')
    domain = [positive(fit['definition']['fit_min_kpc'], 'fit minimum'),
              positive(fit['definition']['fit_max_kpc'], 'fit maximum')]
    if domain[0] >= domain[1]:
        raise ValueError('NFW fit domain must increase')
    x = finite(data['validation_query_positions_proper_kpc'], name='withheld query positions')
    direct = finite(data['validation_acceleration_direct_km2_s2_kpc'], x.shape, 'withheld direct force')
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError('Require at least three independent withheld force locations')
    radius = np.linalg.norm(x, axis=1)
    if np.any((radius < domain[0]) | (radius > domain[1])):
        raise ValueError('Every withheld force location must be within the fitted radial domain')
    u = radius / rs
    # The direct difference loses significant digits for r << r_s. Evaluate
    # the convergent series there, without ever evaluating the singular r=0.
    enclosed_fraction = np.empty_like(u)
    small = u < 1e-3
    us = u[small]
    enclosed_fraction[small] = us**2 * (0.5 + us * (-2/3 + us * (3/4 + us * (-4/5 + us * (5/6)))))
    enclosed_fraction[~small] = np.log1p(u[~small]) - u[~small] / (1 + u[~small])
    model = -G * mass * (enclosed_fraction / radius**2)[:, None] * (x / radius[:, None])
    if not np.isfinite(model).all():
        raise ValueError('NFW force is outside representable numeric range')
    residual = np.linalg.norm(model - direct, axis=1)
    if not np.isfinite(residual).all():
        raise ValueError('NFW force residual is outside representable numeric range')
    floor = positive(force_floor_km2_s2_kpc, 'force floor')
    tolerance = positive(force_relative_tolerance, 'force relative tolerance')
    allowed = tolerance * np.maximum(np.linalg.norm(direct, axis=1), floor)
    return {
        'schema': 'native-nfw-host-reference-v1', 'status': 'PASS' if np.all(residual <= allowed) else 'FAIL',
        'frame': frame['frame'], 'frame_definition_sha256': frame['frame_definition_sha256'], 'epoch': epoch,
        'frame_history_sha256': frame_sha256, 'native_history_sha256': native_sha256,
        'host_structure_sha256': structure_sha256, 'fit_domain_proper_kpc': domain,
        'scale_radius_proper_kpc': rs, 'scale_density_msun_kpc3': rho, 'scale_mass_msun': float(mass),
        'force_relative_tolerance': tolerance, 'force_floor_km2_s2_kpc': floor,
        'maximum_force_budget_fraction': float(np.max(residual / allowed)),
        'force_reference': 'withheld isolated unsoftened direct Newtonian force of declared native aperture components',
        'limitations': 'Spherical total-density NFW fitted at one epoch; finite-source validation is not the periodic GADGET force, orbit-wide accuracy, host evolution, resolution convergence, or science acceptance. Frame-origin acceleration is not modeled.',
        'science_acceptance': 'NOT_ACCEPTED',
    }, {'query_positions_kpc': x, 'direct_acceleration_kms2_per_kpc': direct,
        'model_acceleration_kms2_per_kpc': model, 'force_error_kms2_per_kpc': residual,
        'force_allowed_kms2_per_kpc': allowed}
