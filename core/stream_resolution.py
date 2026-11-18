"""Stellar-stream sampling, force support and actual output-cadence gates.

Preflight estimates are separate from measured aperture counts and convergence.
Recycled SSP sampling uses native saved-aperture counts rather than an assumed
initial gas-mass bound.
"""
from pathlib import Path
import json
from .foundation import KPC_CM, SECONDS_PER_YEAR, _number, file_sha256, validate_spec


def validate_stream_resolution(request):
    required = {'minimum_width_proper_kpc', 'minimum_dispersion_km_s', 'minimum_progenitor_mass_msun',
                'minimum_perturber_mass_msun', 'minimum_star_members', 'minimum_dm_members',
                'maximum_softening_width_ratio', 'maximum_output_crossing_fraction', 'maximum_relative_speed_km_s'}
    if not isinstance(request, dict) or set(request) != required:
        raise ValueError('Stream resolution requires exactly ' + ', '.join(sorted(required)))
    for key, value in request.items():
        _number(value, key, positive=True)
    for key in ('minimum_star_members', 'minimum_dm_members'):
        if type(request[key]) is not int or request[key] < 3:
            raise ValueError(f'{key} must be an integer >=3')
    for key in ('maximum_softening_width_ratio', 'maximum_output_crossing_fraction'):
        if request[key] > 1:
            raise ValueError(f'{key} must be <=1')
    return request


def _crossing_time_myr(request):
    kpc_per_myr_per_kms = 1.e5 * 1.e6 * SECONDS_PER_YEAR / KPC_CM
    speed = max(request['maximum_relative_speed_km_s'], request['minimum_dispersion_km_s'])
    return request['minimum_width_proper_kpc'] / (speed * kpc_per_myr_per_kms)


def recycled_stellar_sampling_observables(spec):
    """Require existing measured SSP-aperture/convergence routes for recycling.

    Delayed returns grow gas recipients before subsequent stellar births. The
    initial gas mass therefore cannot supply an upper bound for SSP sampling.
    Actual native stellar counts are evaluated by the existing resolution
    budget producer, using its declared apertures and observable convergence.
    """
    from .zoom import validate_resolution_request
    request = spec.get('resolution_budget')
    if request is None:
        raise ValueError(
            'Recycled stellar-stream claims require resolution_budget with '
            'species-4 cold_stream observables and measured native aperture counts'
        )
    request = validate_resolution_request(request)
    streams = [row for row in request['observables'] if row['kind'] == 'cold_stream']
    minimum = spec['stream_resolution']['minimum_star_members']
    if not streams or any(row['species'] != 4 or row['minimum_members'] < minimum for row in streams):
        raise ValueError(
            'Recycled stellar sampling requires species-4 cold_stream observables '
            'with minimum_members at least stream_resolution.minimum_star_members'
        )
    return {row['name'] for row in streams}


def stream_feasibility(request, *, dm_mass_msun, tracer_mass_upper_msun,
                       softening_proper_kpc, output_intervals_myr):
    """Check declared resolution scales; unresolved tracer masses stay explicit.

    GADGET spline forces are Newtonian beyond 2.8 softenings. Output cadence is
    compared with the fastest declared encounter/dispersion crossing time.
    A missing pre-evolution SSP mass bound requires measured stellar counts;
    it never supplies a passing analytic sampling estimate.
    """
    validate_stream_resolution(request)
    for name, value in [('dm_mass_msun', dm_mass_msun), ('softening_proper_kpc', softening_proper_kpc)]:
        _number(value, name, positive=True)
    if tracer_mass_upper_msun is not None:
        _number(tracer_mass_upper_msun, 'tracer_mass_upper_msun', positive=True)
    if not isinstance(output_intervals_myr, (list, tuple)) or not output_intervals_myr:
        raise ValueError('At least two science epochs with measured time intervals are required')
    for value in output_intervals_myr:
        _number(value, 'output interval', positive=True)
    values = {
        'perturber_dm_sampling': (request['minimum_perturber_mass_msun'] / dm_mass_msun, request['minimum_dm_members'], 'minimum'),
        'force_support_width_ratio': (2.8 * softening_proper_kpc / request['minimum_width_proper_kpc'], request['maximum_softening_width_ratio'], 'maximum'),
        'output_crossing_fraction': (max(output_intervals_myr) / _crossing_time_myr(request), request['maximum_output_crossing_fraction'], 'maximum'),
    }
    if tracer_mass_upper_msun is not None:
        values['progenitor_star_sampling'] = (
            request['minimum_progenitor_mass_msun'] / tracer_mass_upper_msun,
            request['minimum_star_members'], 'minimum',
        )
    checks = {
        name: {'measured': value, 'limit': limit, 'direction': direction,
               'passed': bool(value >= limit if direction == 'minimum' else value <= limit)}
        for name, (value, limit, direction) in values.items()
    }
    if tracer_mass_upper_msun is None:
        checks['progenitor_star_sampling'] = {
            'measured': None, 'limit': request['minimum_star_members'],
            'direction': 'minimum', 'passed': None,
            'status': 'REQUIRES_SAVED_APERTURE_COUNTS',
        }
    failed = any(check['passed'] is False for check in checks.values())
    status = 'FAIL' if failed else 'REQUIRES_SAVED_SAMPLING' if tracer_mass_upper_msun is None else 'PASS'
    return {'status': status, 'checks': checks,
            'scope': 'Necessary resolution planning; actual members and observable convergence remain required'}


def saved_stream_cadence(spec, run_dir):
    """Measure actual saved gaps using the shared CLASS scale-factor clock."""
    import numpy as np
    from .cosmology import run_proper_time_clock, proper_time_intervals
    from .evolution import load_config, collect_gadget_snapshot_schedule
    validate_spec(spec)
    request = spec['stream_resolution']
    if request is None:
        return {'status': 'NOT_REQUESTED'}
    root = Path(run_dir).resolve()
    native_path = root / 'manifest.json'
    original_hash = file_sha256(native_path)
    native = json.loads(native_path.read_text())
    background = native['input_artifacts']['boltzmann_background']
    background_path = Path(background['path']).expanduser()
    if not background_path.is_absolute():
        background_path = root / background_path
    background_path = background_path.resolve()
    if not background_path.is_relative_to(root) or file_sha256(background_path) != background['sha256']:
        raise ValueError('Saved stream cadence requires unchanged run-local CLASS background')
    cfg = load_config(root / 'config.yaml')
    recycling = cfg.get('stellar_evolution', {}).get('enabled', False)
    if type(recycling) is not bool:
        raise ValueError('stellar_evolution.enabled must be an explicit boolean')
    schedule = collect_gadget_snapshot_schedule(root, cfg)
    actual = np.asarray([row['actual_redshift'] for row in schedule['rows']], dtype=float)
    if (actual.ndim != 1 or len(actual) < 2 or not np.isfinite(actual).all()
            or np.any(actual < 0) or np.any(np.diff(actual) >= 0)):
        raise ValueError('Saved epochs must follow increasing proper time')
    timing = proper_time_intervals(run_proper_time_clock(root), 1 / (1 + actual))
    intervals = np.asarray(timing['intervals_gyr']) * 1000
    measured = float(np.max(intervals) / _crossing_time_myr(request))
    upper = timing['interval_upper_bounds_gyr']
    fraction_upper = None if upper is None else max(upper) * 1000 / _crossing_time_myr(request)
    status = (
        'UNRESOLVED_CLOCK' if fraction_upper is None else
        'PASS' if fraction_upper <= request['maximum_output_crossing_fraction'] else 'FAIL'
    )
    if file_sha256(native_path) != original_hash or file_sha256(background_path) != background['sha256']:
        raise ValueError('Native run/background changed while measuring saved cadence')
    return {
        'status': status, 'measured_output_crossing_fraction': measured,
        'output_crossing_fraction_upper_bound': fraction_upper,
        'maximum_output_crossing_fraction': request['maximum_output_crossing_fraction'],
        'actual_redshifts': actual.tolist(), 'actual_proper_times_gyr': timing['proper_times_gyr'],
        'actual_intervals_myr': intervals.tolist(), 'clock_intervals': timing,
        'stellar_recycling_enabled': recycling,
        'native_manifest_sha256': original_hash, 'background': background,
        'requested_epoch_status': schedule['status'],
        'scope': 'Actual saved cadence with tabulated-clock numerical bounds; continuum cosmology, sampling and convergence remain separate',
    }


def configured_stream_preflight(spec, cfg, background_path):
    """Plan uniform-run scales and require actual recycled-SSP sampling later.

    Multimass zooms continue through their explicit measured species/aperture
    route; a uniform-grid approximation is not applied to those inputs.
    """
    import numpy as np
    from .cosmology import background_proper_time_clock, proper_time_intervals
    from .evolution import particle_mass_msun_h
    validate_spec(spec)
    request = spec['stream_resolution']
    if request is None:
        return {'status': 'NOT_REQUESTED', 'scope': 'no resolved stream claim'}
    sim = cfg['simulation']
    if sim.get('zoom_softening_mass_ratios'):
        raise ValueError('Zoom stream feasibility requires actual species masses; uniform mass estimate is unsupported')
    if sim['physics'] != 'hydro' or not cfg.get('hydro', {}).get('star_formation'):
        raise ValueError('Resolved stellar streams require enabled star formation')
    recycling = cfg.get('stellar_evolution', {}).get('enabled', False)
    if type(recycling) is not bool:
        raise ValueError('stellar_evolution.enabled must be an explicit boolean')
    required_observables = recycled_stellar_sampling_observables(spec) if recycling else set()
    outputs = np.asarray(sim['snapshot_redshift_list'])
    if (outputs.ndim != 1 or len(outputs) < 2 or outputs.dtype.kind not in 'fiu'
            or not np.isfinite(outputs).all() or np.any(outputs < 0)
            or np.any(np.diff(outputs.astype(float)) >= 0)):
        raise ValueError('Science epochs must be finite, distinct, and in decreasing redshift order')
    timing = proper_time_intervals(background_proper_time_clock(background_path), 1 / (1 + outputs))
    nominal = [gap * 1000 for gap in timing['intervals_gyr']]
    upper = timing['interval_upper_bounds_gyr']
    interval_bounds = None if upper is None else [gap * 1000 for gap in upper]
    h = cfg['cosmology']['H0'] / 100.
    dm_mass = particle_mass_msun_h(cfg, dm_only=False) / h
    initial_gas_mass = dm_mass * cfg['cosmology']['omega_b'] / cfg['cosmology']['omega_cdm']
    # Returns add mass to gas before later births. Only non-recycling runs
    # retain the initial gas mass as an SSP upper bound.
    tracer_bound = None if recycling else initial_gas_mass
    softening = max(min(sim['softening_kpc_h'] / (1 + z), sim['softening_max_physical_kpc_h']) / h for z in outputs)
    report = stream_feasibility(
        request, dm_mass_msun=dm_mass, tracer_mass_upper_msun=tracer_bound,
        softening_proper_kpc=softening, output_intervals_myr=interval_bounds or nominal,
    )
    if upper is None:
        report['checks']['output_crossing_fraction'].update(passed=None, status='UNBOUNDED_CLOCK')
        report['status'] = (
            'FAIL' if any(check['passed'] is False for check in report['checks'].values())
            else 'UNRESOLVED_CLOCK'
        )
    if file_sha256(background_path) != timing['clock_source']['sha256']:
        raise ValueError('CLASS background changed during stream preflight')
    report['inputs'] = {
        'background_sha256': timing['clock_source']['sha256'], 'dm_mass_msun': dm_mass,
        'initial_gas_particle_mass_msun': initial_gas_mass, 'tracer_mass_upper_msun': tracer_bound,
        'maximum_softening_proper_kpc': softening, 'output_intervals_myr': nominal,
        'output_interval_upper_bounds_myr': interval_bounds, 'clock_intervals': timing,
    }
    report['stellar_sampling'] = {
        'policy': 'measured native SSP aperture counts and convergence' if recycling else 'initial uniform gas mass upper bound plus measured follow-up',
        'required_resolution_observables': sorted(required_observables),
        'pre_evolution_mass_bound_available': not recycling,
    }
    return report
