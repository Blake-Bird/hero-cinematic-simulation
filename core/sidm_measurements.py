"""Measured SIDM shell, core, component and satellite diagnostics.

Inputs are exact native identities, Msun, proper kpc and peculiar km/s.
Density and velocity moments retain their measured apertures and support;
resolved phase-space heat flux is distinct from unresolved collision transport.
Saved collapse indicators and scattering-associated unbinding are conditional
measurements, not proof of physical attribution or numerical convergence.
"""
from __future__ import annotations
import math
import numpy as np
from .foundation import KPC_CM, MSUN_G
from .data import exact_uint64_ids
from .sidm_physics import _positive


SIDM_FIELDS = {
    'interaction_count': 'SIDMInteractionCount',
    'exposure': 'SIDMExposure',
    'support_length_proper_kpc': 'SIDMScatteringSupportLength',
    'coarse_support_fraction': 'SIDMKernelCoarseSupportFraction',
    'omitted_scattering_upper_bound': 'SIDMOmittedScatteringUpperBound',
}


HISTORY_FIELDS = {
    'density_msun_kpc3': 'Msun / proper kpc^3',
    'density_shot_noise_msun_kpc3': 'Msun / proper kpc^3',
    'enclosed_mass_msun': 'Msun',
    'velocity_dispersion_squared_km2_s2': '(km / s)^2',
    'radial_velocity_variance_km2_s2': '(km / s)^2',
    'anisotropy_beta': 'dimensionless',
    'axis_ratio_ba': 'dimensionless', 'axis_ratio_ca': 'dimensionless',
    'heat_flux_radial_erg_cm2_s': 'erg / cm^2 / s',
    'heat_flux_sampling_error_erg_cm2_s': 'erg / cm^2 / s',
    'heat_luminosity_erg_s': 'erg / s',
    'mean_interaction_count': 'dimensionless',
    'mean_exposure': 'dimensionless',
    'maximum_coarse_support_fraction': 'dimensionless',
    'maximum_omitted_scattering_upper_bound': 'dimensionless',
}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError('Explicit integer ' + name + ' >= ' + str(minimum) + ' required')
    return value


def _finite(value, shape=None, name='array'):
    result = np.asarray(value, float)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError(name + ' must have its declared shape and finite values')
    return result


def _supported(value, shape, name):
    result = np.asarray(value)
    if result.dtype.kind != 'b' or result.shape != shape:
        raise ValueError(name + ' requires an exact boolean support array')
    return result


def _physical_samples(sample):
    ids = exact_uint64_ids(sample['particle_ids'],allow_empty=True)
    x = _finite(sample['offsets_proper_kpc'], (len(ids), 3), 'proper offsets')
    v = _finite(sample['velocities_peculiar_km_s'], x.shape, 'peculiar velocities')
    mass = _finite(sample['masses_msun'], (len(ids),), 'physical masses')
    if np.any(mass <= 0) or not np.isfinite(mass.sum()):
        raise ValueError('Native representatives require positive finite complete masses')
    return ids, x, v, mass


def _mass_summary(mass):
    """Total Msun, effective count and the Poisson mass norm sqrt(sum m_i^2).

    Input masses are already finite and positive from _physical_samples.
    Scaling by their maximum before squaring preserves units in the norm and
    avoids overflow/underflow in N_eff=(sum m_i)^2/sum(m_i^2). Empty selections
    have zero measured mass, count and mass norm.
    """
    if not len(mass):
        return 0., 0., 0.
    maximum = float(mass.max())
    scaled = mass / maximum
    squared_sum = float(np.dot(scaled,scaled))
    effective = float(scaled.sum())**2 / squared_sum
    return float(mass.sum()), effective, maximum*math.sqrt(squared_sum)


def measure_sidm_shells(sample, request):
    """Density, kinematics, tensor shapes and phase-space third moments.

    The third moment is 1/(2 V) sum m |c|^2 c_r with c relative to the
    mass-weighted local shell mean. It is the resolved coarse-grained kinetic
    heat flux after removing the shell Cartesian mean; coherent rotation is
    not a thermal-conduction model. Unresolved collisional transport is not
    invented. Its conditional particle sampling error is not convergence.
    """
    from .halos import spherical_velocity_variances

    ids, x, velocity, mass = _physical_samples(sample)
    edges = _finite(request['radial_edges_proper_kpc'], name='radial edges')
    if edges.ndim != 1 or len(edges) < 4 or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError('At least three increasing physical radial shells required')
    minimum = _integer(request['minimum_shell_particles'], 'minimum shell particles', 3)
    minimum_effective = _positive(request['minimum_effective_particles'], 'effective population floor')
    min_radius = _positive(request['minimum_resolved_radius_proper_kpc'], 'resolved radius', zero=True)
    gap = _positive(request['minimum_shape_eigenvalue_gap_fraction'], 'shape orientation separation')
    max_coarse = _positive(request['maximum_coarse_support_fraction'], 'coarse support ceiling', zero=True)
    max_omitted = _positive(request['maximum_omitted_scattering_upper_bound'], 'omitted hazard ceiling', zero=True)
    if max_coarse > 1 or gap >= 1:
        raise ValueError('Fractional support and eigengap limits have finite declared domains')
    bulk = _finite(request['bulk_velocity_km_s'], (3,), 'halo bulk velocity')
    velocity = velocity - bulk
    n = len(edges) - 1
    radius = np.linalg.norm(x, axis=1)
    shell = np.searchsorted(edges, radius, side='right') - 1
    shell[radius == edges[-1]] = n - 1
    volume = 4 * np.pi / 3 * np.diff(edges ** 3)
    midpoint = ((edges[:-1] ** 3 + edges[1:] ** 3) / 2) ** (1 / 3)
    arrays = {name: np.full(n, np.nan) for name in HISTORY_FIELDS}
    arrays.update({'shell_count': np.zeros(n, np.uint64), 'effective_particle_count': np.zeros(n),
                   'profile_supported': np.zeros(n, bool), 'transport_supported': np.zeros(n, bool),
                   'scattering_supported': np.zeros(n, bool), 'shape_supported': np.zeros(n, bool),
                   'enclosed_mass_supported': np.zeros(n, bool), 'enclosed_effective_particle_count': np.zeros(n),
                   'axis_orientation_supported': np.zeros((n, 3), bool),
                   'axes_columns': np.full((n, 3, 3), np.nan),
                   'radius_proper_kpc': midpoint, 'radial_edges_proper_kpc': edges,
                   'mean_velocity_km_s': np.full((n, 3), np.nan)})
    diagnostic = sample.get('sidm', {})
    has_diagnostics = all(key in diagnostic for key in SIDM_FIELDS)
    if has_diagnostics:
        count = exact_uint64_ids(diagnostic['interaction_count'], unique=False)
        if count.shape != (len(ids),):
            raise ValueError('Exact macro interaction counters require aligned native IDs')
        for name in SIDM_FIELDS.keys() - {'interaction_count'}:
            value = _finite(diagnostic[name], (len(ids),), name)
            if np.any(value < 0) or name == 'coarse_support_fraction' and np.any(value > 1):
                raise ValueError('Native scattering diagnostics leave their physical domain')
    for k in range(n):
        select = shell == k
        m, xx, vv = mass[select], x[select], velocity[select]
        total, neff, mass_norm = _mass_summary(m)
        arrays['shell_count'][k] = len(m)
        enclosed = radius <= edges[k + 1]
        enclosed_mass = mass[enclosed]
        enclosed_total, enclosed_effective, _ = _mass_summary(enclosed_mass)
        arrays['enclosed_mass_msun'][k] = enclosed_total
        if len(enclosed_mass):
            arrays['enclosed_effective_particle_count'][k] = enclosed_effective
            arrays['enclosed_mass_supported'][k] = (len(enclosed_mass) >= minimum
                and enclosed_effective >= minimum_effective and edges[k + 1] >= min_radius)
            if has_diagnostics:
                arrays['enclosed_mass_supported'][k] &= (
                    np.max(np.asarray(diagnostic['coarse_support_fraction'])[enclosed]) <= max_coarse
                    and np.max(np.asarray(diagnostic['omitted_scattering_upper_bound'])[enclosed]) <= max_omitted)
        arrays['density_msun_kpc3'][k] = total / volume[k]
        arrays['density_shot_noise_msun_kpc3'][k] = mass_norm / volume[k]
        if not len(m):
            continue
        arrays['effective_particle_count'][k] = neff
        support = len(m) >= minimum and neff >= minimum_effective and edges[k] >= min_radius
        arrays['profile_supported'][k] = support
        mean = np.sum(m[:, None] * vv, axis=0) / total
        c = vv - mean
        covariance = np.einsum('n,ni,nj->ij', m, c, c) / total
        arrays['mean_velocity_km_s'][k] = mean
        arrays['velocity_dispersion_squared_km2_s2'][k] = np.trace(covariance) / 3
        rr = np.linalg.norm(xx, axis=1)
        direction = np.divide(xx, rr[:, None], out=np.zeros_like(xx), where=rr[:, None] > 0)
        variance = spherical_velocity_variances(xx, vv, m)
        radial_variance = variance[0]
        arrays['radial_velocity_variance_km2_s2'][k] = radial_variance
        if radial_variance > 0:
            arrays['anisotropy_beta'][k] = 1 - (variance[1]+variance[2]) / (2*radial_variance)
        tensor = np.einsum('n,ni,nj->ij', m, xx, xx) / total
        eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, -1] *= -1
        shape_support = support and eigenvalues[-1] > 0
        if shape_support:
            arrays['axis_ratio_ba'][k] = np.sqrt(eigenvalues[1] / eigenvalues[0])
            arrays['axis_ratio_ca'][k] = np.sqrt(eigenvalues[2] / eigenvalues[0])
            arrays['axes_columns'][k] = eigenvectors
            delta = np.abs(eigenvalues[:, None] - eigenvalues[None, :]) / eigenvalues[0]
            arrays['axis_orientation_supported'][k] = np.all((delta >= gap) | np.eye(3, dtype=bool), axis=1)
        arrays['shape_supported'][k] = shape_support
        contribution = 0.5 * np.sum(c * c, axis=1) * np.sum(c * direction, axis=1)
        average = np.dot(m, contribution) / total
        # Conversion: Msun/kpc^3 * (km/s)^3 = erg/(cm^2 s).
        to_flux = MSUN_G / KPC_CM ** 3 * 1e15
        flux = total / volume[k] * average * to_flux
        error = np.sqrt(np.sum((m * (contribution - average)) ** 2)) / volume[k] * to_flux
        if neff > 1:
            error *= np.sqrt(neff / (neff - 1))
        arrays['heat_flux_radial_erg_cm2_s'][k] = flux
        arrays['heat_flux_sampling_error_erg_cm2_s'][k] = error
        arrays['heat_luminosity_erg_s'][k] = 4 * np.pi * (midpoint[k] * KPC_CM) ** 2 * flux
        arrays['transport_supported'][k] = support and np.all(rr > 0)
        if has_diagnostics:
            arrays['mean_interaction_count'][k] = np.dot(m, count[select].astype(float)) / total
            arrays['mean_exposure'][k] = np.dot(m, np.asarray(diagnostic['exposure'])[select]) / total
            coarse = np.asarray(diagnostic['coarse_support_fraction'])[select]
            omitted = np.asarray(diagnostic['omitted_scattering_upper_bound'])[select]
            arrays['maximum_coarse_support_fraction'][k] = coarse.max()
            arrays['maximum_omitted_scattering_upper_bound'][k] = omitted.max()
            arrays['scattering_supported'][k] = support and coarse.max() <= max_coarse and omitted.max() <= max_omitted
            if not arrays['scattering_supported'][k]:
                arrays['profile_supported'][k] = False
                arrays['shape_supported'][k] = False
                arrays['axis_orientation_supported'][k] = False
                arrays['transport_supported'][k] = False
    report = {'shape_definition': 'mass-weighted unreduced shell tensor about the measured halo center',
        'transport_definition': 'resolved third central peculiar-velocity moment; radial outward positive',
        'transport_scope': 'coarse-grained kinetic heat flux; no unresolved collision-term reconstruction',
        'anisotropy_definition': '1-(var(v_theta)+var(v_phi))/(2*var(v_r)); subtract each local spherical-component mean',
        'anisotropy_frame': 'declared halo rest frame, simulation-z polar axis; origin or zero radial variance is unresolved',
        'uncertainty_scope': 'conditional particle sampling only; correlated dynamics and numerical/model biases separate',
        'interaction_scope': 'macro algorithm events and cumulative angular-domain optical depth; no resolved microscopic collision catalog',
        'support_limits': {key: request[key] for key in ('minimum_shell_particles', 'minimum_effective_particles',
            'minimum_resolved_radius_proper_kpc', 'maximum_coarse_support_fraction', 'maximum_omitted_scattering_upper_bound')},
        'scattering_diagnostics_available': has_diagnostics, 'science_acceptance': 'NOT_ACCEPTED'}
    for name, value in arrays.items():
        if name in HISTORY_FIELDS and np.any(np.isinf(value)):
            raise FloatingPointError('SIDM shell estimator overflow: ' + name)
    return report, arrays


def measure_core_definitions(sample, profile, request):
    """Retain central aperture density, saved-bin crossing brackets and a fit.

    The density-half and logarithmic-slope radii have explicit saved-bin
    brackets. The cored-isothermal fit covariance is conditional on the shot
    noise weighting and is not a physical resolution uncertainty.
    """
    _, x, v, mass = _physical_samples(sample)
    aperture = _positive(request['central_aperture_proper_kpc'], 'central aperture')
    minimum = _integer(request['minimum_central_particles'], 'central population', 3)
    r = np.linalg.norm(x, axis=1)
    select = r <= aperture
    central_mass, central_effective, central_mass_norm = _mass_summary(mass[select])
    supported = (int(select.sum()) >= minimum and aperture >= request['minimum_resolved_radius_proper_kpc']
        and central_effective >= request['minimum_effective_particles'])
    diagnostics = sample.get('sidm', {})
    if select.any() and diagnostics:
        supported = bool(supported and np.max(np.asarray(diagnostics['coarse_support_fraction'])[select]) <= request['maximum_coarse_support_fraction']
            and np.max(np.asarray(diagnostics['omitted_scattering_upper_bound'])[select]) <= request['maximum_omitted_scattering_upper_bound'])
    central_volume = 4 * np.pi * aperture ** 3 / 3
    rho = float(central_mass / central_volume)
    center = {'radius_proper_kpc': aperture, 'count': int(select.sum()), 'mass_msun': float(central_mass),
        'effective_particle_count': central_effective,
        'density_msun_kpc3': rho, 'density_shot_noise_msun_kpc3': central_mass_norm / central_volume,
        'supported': bool(supported), 'dispersion_squared_km2_s2': None}
    if supported:
        mean = np.average(v[select], axis=0, weights=mass[select])
        center['dispersion_squared_km2_s2'] = float(np.sum(mass[select, None] * (v[select] - mean) ** 2) / central_mass / 3)
    radius, density = profile['radius_proper_kpc'], profile['density_msun_kpc3']
    mask = profile['profile_supported'] & (density > 0)
    definitions = {'density_half': {'status': 'UNRESOLVED', 'radius_interval_proper_kpc': None},
                   'slope_minus_one': {'status': 'UNRESOLVED', 'radius_interval_proper_kpc': None},
                   'cored_isothermal_fit': {'status': 'UNRESOLVED'}}
    if supported and rho > 0:
        target = rho / 2
        crossings = [i for i in range(len(radius) - 1) if mask[i] and mask[i + 1]
                     and density[i] >= target and density[i + 1] < target]
        if len(crossings) == 1:
            i = crossings[0]
            definitions['density_half'] = {'status': 'BRACKETED', 'radius_interval_proper_kpc': radius[i:i + 2].tolist(),
                'density_threshold_msun_kpc3': target, 'definition': 'half the measured fixed-aperture central density'}
        elif len(crossings) > 1:
            definitions['density_half']['status'] = 'MULTIPLE_CROSSINGS'
    slopes = np.full(len(radius) - 1, np.nan)
    pairs = mask[:-1] & mask[1:] & (radius[:-1] > 0)
    slopes[pairs] = np.log(density[1:][pairs] / density[:-1][pairs]) / np.log(radius[1:][pairs] / radius[:-1][pairs])
    slope_radius = np.sqrt(radius[:-1] * radius[1:])
    crossings = [i for i in range(len(slopes) - 1) if np.isfinite(slopes[i:i + 2]).all()
                 and slopes[i] > -1 and slopes[i + 1] <= -1]
    if len(crossings) == 1:
        i = crossings[0]
        definitions['slope_minus_one'] = {'status': 'BRACKETED', 'radius_interval_proper_kpc': slope_radius[i:i + 2].tolist(),
            'definition': 'first supported transition of finite-bin d ln rho / d ln r through minus one'}
    elif len(crossings) > 1:
        definitions['slope_minus_one']['status'] = 'MULTIPLE_CROSSINGS'
    fit = request['core_fit']
    low, high = _finite(fit['radius_interval_proper_kpc'], (2,), 'fit interval')
    rmin, rmax = _finite(fit['core_radius_bounds_proper_kpc'], (2,), 'core radius bounds')
    if not 0 < low < high or not 0 < rmin < rmax:
        raise ValueError('Explicit increasing fit domain and core radius bounds required')
    fitmask = mask & (radius >= low) & (radius <= high) & (profile['density_shot_noise_msun_kpc3'] > 0)
    if int(fitmask.sum()) >= _integer(fit['minimum_bins'], 'core fit bins', 4):
        from scipy.optimize import least_squares

        rr, dd, err = radius[fitmask], density[fitmask], profile['density_shot_noise_msun_kpc3'][fitmask]
        density_floor = _positive(fit['density_floor_msun_kpc3'], 'core density floor')
        peak = max(float(dd.max()), density_floor)
        ceiling = peak * _positive(fit['maximum_density_multiple'], 'density fit ceiling')
        if not math.isfinite(ceiling) or ceiling <= density_floor:
            raise ValueError('Finite density fit ceiling above its floor required')
        initial = [np.clip(np.log(peak), np.log(density_floor), np.log(ceiling)),
                   (np.log(rmin) + np.log(rmax)) / 2]
        def residual(parameters):
            prediction = np.exp(parameters[0]) / (1 + (rr / np.exp(parameters[1])) ** 2)
            return (prediction - dd) / err
        result = least_squares(residual, initial, bounds=([np.log(density_floor), np.log(rmin)],
            [np.log(ceiling), np.log(rmax)]),
            max_nfev=_integer(fit['maximum_function_evaluations'], 'fit evaluation bound'))
        definitions['cored_isothermal_fit']['reason'] = str(result.message)
        if result.success and np.isfinite(result.x).all() and np.isfinite(result.jac).all():
            # Inverting J.T@J squares its condition number. SVD gives the same
            # conditional log-parameter covariance with a direct rank check.
            _, singular, modes = np.linalg.svd(result.jac, full_matrices=False)
            tolerance = np.finfo(float).eps * max(result.jac.shape) * singular[0]
            definitions['cored_isothermal_fit']['reason'] = 'Unresolved fit Jacobian rank or covariance'
            if len(singular) == 2 and singular[-1] > tolerance:
                dof = int(fitmask.sum()) - 2
                scaled_modes = modes.T / singular
                cov = (scaled_modes @ scaled_modes.T) * (2*result.cost/dof)
                if np.isfinite(cov).all():
                    boundary = bool(np.any(result.active_mask))
                    definitions['cored_isothermal_fit'] = {'status': 'BOUNDARY_LIMITED' if boundary else 'FIT_CONDITIONAL',
                        'central_density_msun_kpc3': float(np.exp(result.x[0])), 'core_radius_proper_kpc': float(np.exp(result.x[1])),
                        'log_parameter_covariance': cov.tolist(), 'degrees_of_freedom': dof, 'weighted_residual_sum': float(2*result.cost),
                        'model': 'rho0 / (1 + (r/rc)^2)', 'uncertainty_scope': 'conditional fit covariance; finite-bin/resolution/model errors separate'}
    return {'central_aperture': center, 'definitions': definitions}, {'log_density_slope': slopes, 'slope_radius_proper_kpc': slope_radius}


def component_centroids_and_misalignment(components, request):
    """Actual component mass centroids and resolved tensor-axis alignments."""
    aperture = _positive(request['aperture_proper_kpc'], 'component aperture')
    minimum = _integer(request['minimum_particles'], 'component population', 3)
    gap = _positive(request['minimum_eigenvalue_gap_fraction'], 'component orientation eigengap')
    records = {}
    for name, sample in components.items():
        _, x, _, m = _physical_samples(sample)
        keep = np.linalg.norm(x, axis=1) <= aperture
        if int(keep.sum()) < minimum:
            records[name] = {'status': 'UNRESOLVED', 'count': int(keep.sum())}
            continue
        x, m = x[keep], m[keep]
        center = np.average(x, axis=0, weights=m)
        xx = x - center
        values, axes = np.linalg.eigh(np.einsum('n,ni,nj->ij', m, xx, xx) / m.sum())
        order = np.argsort(values)[::-1]
        values, axes = values[order], axes[:, order]
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1
        resolved = np.zeros(3, bool)
        if values[0] > 0:
            delta = np.abs(values[:, None] - values[None, :]) / values[0]
            resolved = np.all((delta >= gap) | np.eye(3, dtype=bool), axis=1)
        records[name] = {'status': 'MEASURED', 'count': len(m), 'centroid_offset_proper_kpc': center.tolist(),
            'axes_columns': axes.tolist(), 'axis_orientation_supported': resolved.tolist(), 'mass_msun': float(m.sum())}
    offsets = []
    if records.get('dm', {}).get('status') == 'MEASURED':
        dm = records['dm']
        for name, row in records.items():
            if name == 'dm' or row['status'] != 'MEASURED':
                continue
            displacement = np.asarray(row['centroid_offset_proper_kpc']) - dm['centroid_offset_proper_kpc']
            supported = np.asarray(row['axis_orientation_supported']) & np.asarray(dm['axis_orientation_supported'])
            angles = np.degrees(np.arccos(np.clip(np.abs(np.sum(np.asarray(row['axes_columns']) * dm['axes_columns'], axis=0)), 0, 1)))
            offsets.append({'component': name, 'dm_centroid_displacement_proper_kpc': displacement.tolist(),
                'offset_proper_kpc': float(np.linalg.norm(displacement)),
                'unsigned_axis_misalignment_deg': [float(a) if ok else None for a, ok in zip(angles, supported)]})
    return {'components': records, 'dm_baryon_comparisons': offsets,
        'definition': 'mass centroid and centered unreduced shape tensor within the same measured proper aperture',
        'uncertainty_scope': 'finite population and centering systematics require separate matched controls'}


def classify_sidm_history(epochs, request):
    """Persistent saved-density crossings with outward resolved heat evidence."""
    ratio = _positive(request['collapse_density_ratio'], 'collapse ratio')
    consecutive = _integer(request['minimum_consecutive_crossings'], 'collapse persistence', 2)
    if ratio <= 1:
        raise ValueError('Collapse threshold must exceed initial measured central density')
    if not epochs:
        raise ValueError('Measured history is empty')
    time = _finite([row['elapsed_time_gyr'] for row in epochs], name='physical history time')
    if len(time) > 1 and np.any(np.diff(time) <= 0):
        raise ValueError('Native history times must strictly increase')
    baseline = epochs[0]['core']['central_aperture']
    rows = []
    densities = [row['core']['central_aperture']['density_msun_kpc3'] for row in epochs]
    support = _supported([row['core']['central_aperture']['supported'] for row in epochs],
                         (len(epochs),), 'central aperture history support')
    for density, supported in zip(densities,support):
        if supported:
            _positive(density,'supported central density')
    threshold = baseline['density_msun_kpc3'] * ratio
    if support[0] and not math.isfinite(threshold):
        raise FloatingPointError('Supported SIDM density threshold exceeds numeric range')
    bracket = None
    for k, row in enumerate(epochs):
        phase = 'UNRESOLVED'
        if support[k] and baseline['supported'] and baseline['density_msun_kpc3'] > 0:
            phase = 'CENTRAL_DENSITY_MEASURED'
            if k and support[k - 1]:
                phase = 'DENSITY_DECLINING' if densities[k] < densities[k - 1] else 'DENSITY_RISING' if densities[k] > densities[k - 1] else 'DENSITY_STATIONARY'
        heat = row.get('core_boundary_heat_luminosity_erg_s')
        heating_evidence = bool(heat is not None and math.isfinite(heat) and heat > 0)
        sigma = row['core']['central_aperture']['dispersion_squared_km2_s2']
        previous_sigma = epochs[k - 1]['core']['central_aperture']['dispersion_squared_km2_s2'] if k else None
        rising_dispersion = bool(sigma is not None and previous_sigma is not None
            and math.isfinite(sigma) and math.isfinite(previous_sigma) and sigma > previous_sigma)
        if (k >= 1 and k + consecutive <= len(epochs) and baseline['supported'] and all(support[k:k + consecutive])
                and all(d >= threshold for d in densities[k:k + consecutive]) and support[k - 1] and densities[k - 1] < threshold):
            if bracket is None:
                bracket = [float(time[k - 1]), float(time[k])]
            phase = 'PERSISTENT_DENSITY_THRESHOLD'
            if heating_evidence and rising_dispersion:
                phase = 'GRAVOTHERMAL_COLLAPSE_INDICATORS'
        rows.append({'snapshot': row['snapshot'], 'elapsed_time_gyr': float(time[k]), 'phase': phase,
            'outward_resolved_heat_flux': heating_evidence, 'rising_central_dispersion': rising_dispersion})
    return {'epochs': rows, 'density_threshold_msun_kpc3': threshold,
        'persistent_crossing_time_interval_gyr': bracket, 'status': 'BRACKETED' if bracket else 'NOT_BRACKETED',
        'definition': 'saved fixed-aperture density persistence; collapse indicators additionally require rising dispersion and outward resolved heat',
        'caveat': 'density crossings alone do not distinguish physical collapse from numerical heating or baryonic compression'}


def satellite_survival_and_transport(initial, current, request):
    """Separate exact ancestry retention, bound loss, and measured unbinding.

    Evaporation attribution requires measured energy, a declared potential
    gauge and a bound-to-unbound crossing with native scattering history. It
    remains a candidate diagnostic rather than proof of collision causality.
    """
    first_ids = exact_uint64_ids(initial['particle_ids'])
    current_ids = exact_uint64_ids(current['particle_ids'])
    m0 = _finite(initial['mass_msun'], (len(first_ids),), 'initial member mass')
    m = _finite(current['mass_msun'], (len(current_ids),), 'current member mass')
    b0 = _supported(initial['bound'], (len(first_ids),), 'initial bound flags')
    bound = _supported(current['bound'], (len(current_ids),), 'current bound flags')
    if np.any(m0 <= 0) or np.any(m <= 0):
        raise ValueError('Satellite native masses must remain positive')
    floor = _positive(request['mass_absolute_floor_msun'], 'satellite mass floor')
    order = np.argsort(current_ids)
    positions = np.searchsorted(current_ids[order], first_ids)
    present = positions < len(current_ids)
    safe = np.minimum(positions, max(len(current_ids) - 1, 0))
    if len(current_ids):
        present &= current_ids[order][safe] == first_ids
    else:
        present[:] = False
    old = np.flatnonzero(present)
    new = order[positions[present]]
    original_bound_mass = float(m0[b0].sum())
    retained_bound = b0[old] & bound[new]
    retained_mass = float(m[new[retained_bound]].sum())
    lost_binding = b0[old] & ~bound[new]
    minimum = _integer(request['minimum_bound_particles'], 'satellite survival floor')
    output = {'initial_bound_mass_msun': original_bound_mass, 'current_bound_mass_msun': float(m[bound].sum()),
        'retained_original_bound_mass_msun': retained_mass,
        'retained_original_bound_mass_fraction': retained_mass / max(original_bound_mass, floor),
        'observed_bound_to_unbound_count': int(lost_binding.sum()),
        'missing_original_member_count': int((~present).sum()), 'current_bound_count': int(bound.sum()),
        'status': 'RESOLVED_SURVIVOR' if bound.sum() >= minimum else 'UNRESOLVED_OR_DISRUPTED',
        'unobserved_ids_policy': 'missing native identities remain unknown; absence is not evaporation',
        'evaporation_candidates': {'status': 'MISSING_PHASE_SPACE_ENERGY', 'count': None}}
    keys = {'specific_energy_km2_s2', 'interaction_count'}
    if keys <= set(initial) and keys <= set(current):
        if request['potential_gauge'] not in {'instantaneous_satellite_escape_zero', 'declared_tidal_saddle_zero'}:
            raise ValueError('A satellite binding energy requires its explicit escape/tidal gauge')
        e0 = _finite(initial['specific_energy_km2_s2'], (len(first_ids),), 'initial binding energy')
        energy = _finite(current['specific_energy_km2_s2'], (len(current_ids),), 'current binding energy')
        c0 = exact_uint64_ids(initial['interaction_count'], unique=False)
        count = exact_uint64_ids(current['interaction_count'], unique=False)
        if c0.shape != e0.shape or count.shape != energy.shape or np.any(count[new] < c0[old]):
            raise ValueError('Satellite counters regressed or fail the exact identity join')
        candidates = lost_binding & (e0[old] < 0) & (energy[new] >= 0) & (count[new] > c0[old])
        output['evaporation_candidates'] = {'status': 'MEASURED_CANDIDATES', 'count': int(candidates.sum()),
            'mass_msun': float(m[new[candidates]].sum()), 'potential_gauge': request['potential_gauge'],
            'attribution': 'bound-to-unbound positive-energy crossing with intervening scattering; matched force/tide controls still required'}
    if 'host_offset_proper_kpc' in current and 'host_relative_velocity_km_s' in current:
        x = _finite(current['host_offset_proper_kpc'], (3,), 'satellite orbit offset')
        v = _finite(current['host_relative_velocity_km_s'], (3,), 'satellite orbit velocity')
        output['orbit'] = {'radius_proper_kpc': float(np.linalg.norm(x)), 'speed_km_s': float(np.linalg.norm(v)),
            'radial_speed_km_s': float(np.dot(x, v) / np.linalg.norm(x)) if np.linalg.norm(x) > 0 else None,
            'specific_angular_momentum_kpc_km_s': np.cross(x, v).tolist()}
    return output
