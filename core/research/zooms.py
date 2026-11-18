"""Prepare, recover and measure bounded native zoom campaigns.

Parent and zoom identities belong to different namespaces. Original MUSIC q
coordinates map parent-cell material to fine zoom representatives; evolved
IDs then locate descendants. Saved native epochs and measured contamination,
force errors and observables determine support for each requested history.
"""
from __future__ import annotations

import json
import math
import numpy as np
import sys
import uuid
import zipfile
from contextlib import contextmanager
from copy import deepcopy
from core.foundation import atomic_json, canonical_digest, file_sha256
from core.native import native_run_lock, run_native_command
from core.foundation import merge_source_bindings as _merge
from core.workflow import path_at
from pathlib import Path

CORE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CORE_DIRECTORY.parent

# Zoom campaign


def _finite(value, label, *, minimum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or integer and type(value) is not int:
        raise ValueError('Explicit finite zoom parameter required: '+label)
    if minimum is not None and value < minimum:
        raise ValueError('Zoom parameter is below the declared supported bound: '+label)
    return value


def _recipe_inputs(recipe, base, *, allowed_tokens):
    """Resolve immutable command assets while allowing only owned future paths."""
    import re
    from commands import COMMANDS
    argv = recipe.get('argv')
    if not isinstance(argv,list) or not argv or any(not isinstance(arg,str) or not arg for arg in argv):
        raise ValueError('Zoom recipes require explicit nonempty argv strings')
    _finite(recipe['wall_seconds'], 'recipe.wall_seconds', minimum=1e-12)
    sources = {}
    for arg in argv:
        if set(re.findall(r'\{([^{}]+)\}',arg))-allowed_tokens:
            raise ValueError('Zoom recipe contains an unknown future-path token: '+arg)
    if '{suite}/commands.py' in argv:
        index = argv.index('{suite}/commands.py')+1
        if index >= len(argv) or argv[index] not in COMMANDS:
            raise ValueError('Zoom recipe has no registered public command')
        if argv[index] in {'zoom-campaign','workflow'}:
            raise ValueError('A zoom attempt recipe cannot recursively dispatch a campaign')
    executable = argv[0]
    if executable != '{python}':
        binary = path_at(base,executable)
        if not binary.is_file() or not recipe.get('executable_sha256') or file_sha256(binary) != recipe['executable_sha256']:
            raise ValueError('External zoom recipe executable requires an exact pinned binary')
        sources[str(binary)] = recipe['executable_sha256']
    for asset in recipe.get('input_files', []):
        path = path_at(base,asset['path'])
        if file_sha256(path) != asset['sha256']:
            raise ValueError('Zoom recipe asset changed before any evolution')
        sources[str(path)] = asset['sha256']
    for flag in ('--request','--campaign'):
        if flag in argv:
            if argv.count(flag) != 1 or argv.index(flag)+1 >= len(argv):
                raise ValueError('Zoom recipe has an ambiguous request argument')
            argument = argv[argv.index(flag)+1]
            if any('{'+token+'}' in argument for token in allowed_tokens-{'suite','python','request_base'}):
                if not recipe.get('produced_request_contract'):
                    raise ValueError('A future diagnostic request needs a frozen produced_request_contract')
                continue
            argument = argument.replace('{suite}',str(PROJECT_ROOT)).replace('{request_base}',str(base))
            path = path_at(base,argument)
            if str(path) not in sources:
                raise ValueError('Every zoom recipe request must be listed with its exact input_files hash')
    return sources


def preflight_zoom_campaign(request, *, request_base=None, allocated_resources=None):
    """Freeze every target/refinement recipe before tracing or native generation."""
    from core.research.capabilities import CapabilityRegistry, require_supported_combination
    if request.get('schema') != 'automatic-zoom-campaign-v1':
        raise ValueError('Expected automatic-zoom-campaign-v1')
    if type(request.get('run_all_variants',False)) is not bool:
        raise ValueError('Zoom campaign run_all_variants must be an explicit boolean')
    base = Path(request_base or '.').resolve()
    resources = request['resources']
    for name in ('mpi_ranks','omp_threads','maximum_stage_count','maximum_tracers'):
        _finite(resources[name], 'resources.'+name, minimum=1, integer=True)
    for name in ('memory_gib','disk_gib','wall_hours'):
        _finite(resources[name], 'resources.'+name, minimum=1e-12)
    if allocated_resources is not None:
        for name in ('mpi_ranks','omp_threads','memory_gib','disk_gib'):
            if resources[name] > allocated_resources[name]:
                raise ValueError('Zoom campaign exceeds its allocated stage '+name+'; configuration unchanged')
    parent, epochs = request['parent'], request['epochs']
    _finite(parent['snapshot'], 'parent.snapshot', minimum=0, integer=True)
    paths = {key:path_at(base,parent[key]) for key in ('run_dir','initial_ic','source_manifest','noise','spectrum_manifest')}
    if not paths['run_dir'].is_dir():
        raise FileNotFoundError('Actual parent catalogue/run directory required')
    sources = {}
    for key in ('initial_ic','source_manifest','noise','spectrum_manifest'):
        digest = parent['sha256'][key]
        if file_sha256(paths[key]) != digest:
            raise ValueError('Frozen parent zoom input changed: '+key)
        sources[str(paths[key])] = digest
    preview_binding = request.get('selection_preview')
    if preview_binding is not None:
        if (not isinstance(preview_binding, dict) or set(preview_binding) != {'path','sha256'}
                or not isinstance(preview_binding['path'], str) or not preview_binding['path']
                or not isinstance(preview_binding['sha256'], str)
                or len(preview_binding['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in preview_binding['sha256'])):
            raise ValueError('Selection preview requires an exact path and lowercase SHA-256')
        preview_path = path_at(base, preview_binding['path'])
        if not preview_path.is_file() or preview_path.stat().st_size > 16*1024**2:
            raise ValueError('Selection preview must be an existing bounded JSON file')
        if file_sha256(preview_path) != preview_binding['sha256']:
            raise ValueError('Reviewed zoom target selection preview changed')
        sources[str(preview_path)] = preview_binding['sha256']
    for key, hash_key in (('binary','binary_sha256'),('native_validation','validation_sha256')):
        path = path_at(base,request['generator'][key])
        if file_sha256(path) != request['generator'][hash_key]:
            raise ValueError('Frozen native zoom generator/control changed')
        sources[str(path)] = request['generator'][hash_key]
    import os
    if not os.access(path_at(base,request['generator']['binary']),os.X_OK):
        raise ValueError('Selected native zoom generator is not executable')
    start, end = epochs['z_start'], epochs['z_end']
    _finite(start, 'epochs.z_start', minimum=0)
    _finite(end, 'epochs.z_end', minimum=-1+1e-12)
    if start <= end:
        raise ValueError('Zoom evolution needs an earlier explicit start redshift')
    _finite(epochs['maximum_redshift_error'], 'epochs.maximum_redshift_error', minimum=0)
    history = epochs['backbone_redshifts']
    if not isinstance(history,list) or not history or any(isinstance(z,bool) or not isinstance(z,(int,float)) or not math.isfinite(z) or not end <= z <= start for z in history) or any(b >= a for a,b in zip(history,history[1:])):
        raise ValueError('Strictly decreasing user-selected zoom output history required')
    snapshots = request['recovery']['snapshots']
    if not isinstance(snapshots,list) or len(snapshots) < len(history) or len(set(snapshots)) != len(snapshots) or any(type(s) is not int or s < 0 for s in snapshots):
        raise ValueError('Distinct actual recovery snapshots must cover the full requested history')
    selection = request['selection_rules']
    for name in ('minimum_dm_particles','maximum_targets'):
        _finite(selection[name], 'selection.'+name, minimum=1, integer=True)
    if selection['ranking'] not in {'dm_count_descending','catalogue_order'}:
        raise ValueError('Recorded target ranking is unsupported')
    explicit = request.get('explicit_host_rows', [])
    if len(set(explicit)) != len(explicit) or len(explicit) > selection['maximum_targets'] or any(type(row) is not int or row < 0 for row in explicit):
        raise ValueError('Zoom explicit target selections are duplicated, invalid or exceed budget')
    rules = request['recovery']['rules']
    _finite(rules['minimum_shared_particles'], 'minimum_shared_particles', minimum=1, integer=True)
    for name in ('minimum_material_fraction','maximum_runner_up_fraction'):
        _finite(rules[name], name, minimum=0)
        if rules[name] > 1:
            raise ValueError('Descendant material support fraction exceeds one')
    if not 0 < rules['minimum_material_fraction'] <= 1 or not 0 <= rules['maximum_runner_up_fraction'] < rules['minimum_material_fraction']:
        raise ValueError('Material match needs a positive supported fraction and a smaller runner-up threshold')
    _finite(request['trace']['aperture_r200c_factor'], 'trace aperture', minimum=1e-12)
    staging = request['staging']
    staging_keys={'box_mpc_h','parent_level','padding_cells','poisson_accuracy','wdm_request','lpt_order',
        'lpt_seed_kind','lpt_seed','phase_noise_by_level','epoch_regeneration_lineage','bounded_source_settings','source_archive_max_mib'}
    if set(staging)-staging_keys:
        raise ValueError('Zoom staging contains unsupported settings before any native tracing')
    from core.zoom import validate_bounded_source_settings
    validate_bounded_source_settings(staging.get('bounded_source_settings'),lpt_order=staging.get('lpt_order',1))
    validate_campaign_source_retention(staging)
    _finite(staging['box_mpc_h'], 'staging box', minimum=1e-12)
    _finite(staging['parent_level'], 'parent level', minimum=5, integer=True)
    _finite(staging['padding_cells'], 'padding cells', minimum=2, integer=True)
    _finite(staging['poisson_accuracy'], 'Poisson accuracy', minimum=1e-300)
    if staging['parent_level'] > 20 or staging['poisson_accuracy'] > 1e-8 or staging.get('lpt_order',1) not in (1,2):
        raise ValueError('Zoom staging lies outside the supported level/Poisson/LPT domain')
    if type(staging.get('lpt_order',1)) is not int:
        raise ValueError('Zoom perturbation order must be an exact supported integer')
    if staging.get('lpt_order',1) == 1 and (staging.get('lpt_seed_kind','radiation_regular') != 'radiation_regular' or staging.get('lpt_seed') is not None):
        raise ValueError('Higher-order seed settings require an actual 2LPT zoom prescription')
    for level,asset in (staging.get('phase_noise_by_level') or {}).items():
        if not isinstance(level,str) or not level.isdigit() or not isinstance(asset,dict) or not asset.get('path') or not asset.get('sha256'):
            raise ValueError('Frozen refinement phase noise requires exact per-level input pins')
        path=path_at(base,asset['path'])
        if file_sha256(path)!=asset['sha256']:
            raise ValueError('Frozen refinement phase-noise input changed before tracing')
        sources[str(path)]=asset['sha256']
    variants = request['refinement_variants']
    maximum = request['followup_limits']['maximum_attempts_per_target']
    _finite(maximum, 'maximum attempts', minimum=1, integer=True)
    if not isinstance(variants,list) or not 1 <= len(variants) <= maximum:
        raise ValueError('Explicit bounded zoom refinement variants required')
    if selection['maximum_targets']*len(variants) > resources['maximum_stage_count']:
        raise ValueError('Zoom target/refinement matrix exceeds the declared execution budget')
    frozen = dict(parent['seeds_by_level'])
    if str(staging['parent_level']) not in frozen:
        raise ValueError('Zoom phase matching needs the exact established parent-level seed')
    previous_level, previous_padding = staging['parent_level'], 0
    for variant in variants:
        level = _finite(variant['refinement_level'], 'refinement level', minimum=staging['parent_level'], integer=True)
        padding = _finite(variant['padding_comoving_kpc_h'], 'refinement padding', minimum=0)
        seeds = variant['seeds_by_level']
        if level > 20 or level < previous_level or padding < previous_padding:
            raise ValueError('Zoom refinement must stay supported and cannot shrink its region/resolution')
        if set(seeds) != {str(value) for value in range(staging['parent_level'],level+1)} or any(type(seed) is not int or not 0 <= seed <= 2147483647 for seed in seeds.values()):
            raise ValueError('One valid exact frozen MUSIC seed is required for every refinement level')
        if any(seeds.get(key) != value for key,value in frozen.items()):
            raise ValueError('A zoom refinement changes previously established parent/fine phase seeds')
        frozen.update(seeds)
        previous_level, previous_padding = level, padding
    spectrum = json.loads(paths['spectrum_manifest'].read_text())
    for filename,digest in spectrum.get('checksums_sha256',{}).items():
        path = (paths['spectrum_manifest'].parent/filename).resolve()
        if not path.is_relative_to(paths['spectrum_manifest'].parent) or file_sha256(path) != digest:
            raise ValueError('Zoom staging solver transfer/background input changed before tracing')
        sources[str(path)] = digest
    if not math.isclose(spectrum['parameters']['z_start'],start,rel_tol=0,abs_tol=1e-8):
        raise ValueError('Zoom start epoch differs from its exact regenerated spectrum bundle')
    physics = request['physics']
    if physics.get('interaction') not in {'collisionless','elastic_sidm'} or physics.get('linear_model') not in {'cdm','wdm'}:
        raise ValueError('Zoom physics is outside the registered native model domain')
    if (spectrum.get('model') == 'wdm') != (physics['linear_model'] == 'wdm'):
        raise ValueError('Zoom linear-model declaration differs from the exact solver bundle')
    if physics['linear_model'] == 'wdm':
        wdm = staging.get('wdm_request')
        if not isinstance(wdm,dict) or set(wdm) != {'wdm','thermal_seed','lcdm_manifest'} or wdm['wdm'].get('include_thermal_velocities',True) is not True:
            raise ValueError('WDM refinements require the full physical thermal prescription and cold reference')
        _finite(wdm['thermal_seed'], 'thermal seed', minimum=0, integer=True)
        if wdm['thermal_seed'] >= 2**64:
            raise ValueError('WDM thermal seed exceeds its exact uint64 domain')
        cold = path_at(base,wdm['lcdm_manifest'])
        sources[str(cold)] = file_sha256(cold)
    elif staging.get('wdm_request') is not None:
        raise ValueError('Cold zoom physics cannot carry an unrelated WDM thermal prescription')
    sidm = physics['interaction'] == 'elastic_sidm'
    if sidm:
        guard = request['sidm_guard']
        _finite(guard['maximum_comoving_support_kpc_h'], 'SIDM comoving support', minimum=1e-12)
        _finite(guard['guard_multiplier'], 'SIDM guard multiplier', minimum=1)
    fields = {'coarse_mass_fraction','force_error','observable_error'} | ({'kernel_coarse_support_fraction','omitted_scattering_upper_bound'} if sidm else set())
    for field in fields:
        requirement = request['region_requirements'][field]
        if requirement.get('units') != 'dimensionless':
            raise ValueError('Native zoom adequacy bounds use dimensionless errors/fractions')
        _finite(requirement['maximum'], 'region requirement '+field, minimum=0)
        if field in {'coarse_mass_fraction','kernel_coarse_support_fraction'} and requirement['maximum'] > 1:
            raise ValueError('Zoom mass/kernel fraction threshold exceeds one')
    require_supported_combination({'dark_matter':{'linear_model':physics['linear_model'],'interactions':{'kind':physics['interaction']}},
        'cosmology_model':spectrum.get('model','lcdm'), 'combination_specification':request.get('combination_specification')}, request_base=base)
    tokens = {'python','suite','request_base','generation','ic','run','attempt','region','target_id','answers','recovery'}
    for name in ('evolution','diagnostics'):
        _merge(sources,_recipe_inputs(request[name],base,allowed_tokens=tokens))
    output = Path(request['diagnostics']['output'])
    if output.is_absolute() or '..' in output.parts or str(output) in {'','.','attempt.json'}:
        raise ValueError('Zoom diagnostic publication must stay within its retained attempt')
    diagnostic_contract = request['diagnostics'].get('produced_request_contract')
    if diagnostic_contract is None:
        argv = request['diagnostics']['argv']
        if '--request' not in argv:
            raise ValueError('Zoom diagnostics need an exact pinned measurement request')
        argument = argv[argv.index('--request')+1].replace('{suite}',str(PROJECT_ROOT)).replace('{request_base}',str(base))
        diagnostic_contract = json.loads(path_at(base,argument).read_text())
    argv = request['diagnostics']['argv']
    if argv[:3] == ['{python}','{suite}/commands.py','measure-zoom-campaign']:
        for flag,value in {'--run':'{run}','--recovery':'{recovery}','--output':'{attempt}/'+output.as_posix()}.items():
            if argv.count(flag) != 1 or argv.index(flag)+1 >= len(argv) or argv[argv.index(flag)+1] != value:
                raise ValueError('Zoom diagnostics must bind its actual '+flag+' product')
    validate_zoom_diagnostic_request(diagnostic_contract, snapshots=snapshots, sidm=sidm, memory_gib=resources['memory_gib'])
    prepared = prepare_zoom_evolution_inputs(request,request_base=base)
    for variant in prepared:
        _merge(sources,variant.get('source_hashes',{}))
    capabilities = CapabilityRegistry.from_spec(request['capabilities'],base)
    capabilities.require(request['required_capabilities'],request.get('domain_parameters',{}))
    return {'schema':'zoom-campaign-preflight-v1', 'request_hash':canonical_digest(request),
        'source_hashes':sources, 'prepared_evolution_inputs':prepared,
        'maximum_target_attempts':selection['maximum_targets']*len(variants),
        'configuration_changes':[]}


def select_catalogue_targets(parent_run, snapshot, rules, explicit_overrides=(), *, memory_gib):
    """Select epoch-local FOF rows and retain the measurements behind the choice.

    Explicit rows have priority and can be outside the automatic count interval;
    the selection records that fact. They are not a resolution or contamination
    acceptance. Tracing still requires a positive radius and actual membership.
    """
    from core.zoom import read_host_catalogue
    from halos import _read_catalogue_series
    cat = read_host_catalogue(parent_run, snapshot, memory_gib=memory_gib)
    counts = cat['lengths']
    low = rules['minimum_dm_particles']
    high = rules.get('maximum_dm_particles')
    if type(low) is not int or low < 1 or high is not None and (type(high) is not int or high < low):
        raise ValueError('Explicit valid target DM count interval required')
    if rules['ranking'] not in {'dm_count_descending', 'catalogue_order'}:
        raise ValueError('Unknown recorded target ranking')
    mask = counts[:,1] >= low
    if high is not None:
        mask &= counts[:,1] <= high
    mask &= np.asarray(cat['radii']) > 0
    rows = list(np.flatnonzero(mask).astype(int))
    if rules['ranking'] == 'dm_count_descending':
        rows.sort(key=lambda row: (-int(counts[row,1]), row))
    overrides = list(explicit_overrides)
    if len(set(overrides)) != len(overrides) or any(type(q) is not int or not 0 <= q < cat['group_count'] for q in overrides):
        raise ValueError('Explicit host overrides must be distinct valid native rows')
    maximum = rules['maximum_targets']
    if type(maximum) is not int or maximum < 1 or len(overrides) > maximum:
        raise ValueError('Target budget cannot silently discard explicit overrides')
    chosen = (overrides+[q for q in rows if q not in overrides])[:maximum]
    if not chosen:
        raise ValueError('Recorded target selection is physically empty')
    # This field is optional in FOF-only catalogues. Never substitute total FOF
    # mass for the distinct spherical-overdensity M200c definition.
    mass200 = _read_catalogue_series(cat['catalogues'], 'Group', 'Group_M_Crit200', required=False)
    if mass200 is not None:
        mass200 = np.asarray(mass200)
        if mass200.shape != (cat['group_count'],) or mass200.dtype.kind not in 'fiu':
            raise ValueError('Measured M200c is not aligned with native FOF rows')
    a, box, hubble = cat['frame']
    catalogue_hashes = [q['sha256'] for q in cat['sources']]
    targets = []
    for row in chosen:
        radius = float(cat['radii'][row])
        issues = []
        if int(counts[row,1]) < low:
            issues.append('below_minimum_dm_particles')
        if high is not None and int(counts[row,1]) > high:
            issues.append('above_maximum_dm_particles')
        if radius <= 0:
            issues.append('nonpositive_r200c_cannot_trace')
        measured_mass = None if mass200 is None else float(mass200[row])
        if measured_mass is not None:
            measured_mass *= 1e10
            if not math.isfinite(measured_mass) or measured_mass <= 0:
                measured_mass = None
        targets.append({'target_id': f'snapshot{snapshot}_host{row}', 'host_row': int(row),
            'epoch_local_identity_namespace': canonical_digest({'catalogue_sha256': catalogue_hashes,
                'snapshot': snapshot, 'host_row': int(row)}),
            'dm_count': int(counts[row,1]), 'stellar_count': int(counts[row,4]),
            'selection_origin': 'explicit_override' if row in overrides else 'recorded_catalogue_rule',
            'automatic_rule_qualified': not issues, 'selection_issues': issues,
            'center_comoving_kpc_h': (np.asarray(cat['positions'][row],float) % box).tolist(),
            'r200c_comoving_kpc_h': radius, 'r200c_proper_kpc': radius*a/hubble,
            'm200c_msun_h': measured_mass,
            'm200c_status': 'unavailable_or_nonpositive' if measured_mass is None else 'measured_catalogue_field'})
    # Optional fields were read after the core catalogue reader's own final hash
    # check, so bind those reads to the same immutable source bytes as well.
    for source in cat['sources'] + cat['snapshot_sources']:
        if file_sha256(source['path']) != source['sha256']:
            raise ValueError('Host catalogue/snapshot changed during target selection')
    return {'schema': 'zoom-target-selection-v1', 'parent_run': str(Path(parent_run).resolve()),
        'snapshot': snapshot, 'rules': rules, 'explicit_overrides': overrides,
        'epoch': {'scale_factor': float(a), 'redshift': float(1/a-1), 'hubble_param': float(hubble)},
        'coordinates': {'position_units':'comoving kpc/h', 'origin':'periodic box origin',
            'box_comoving_kpc_h':float(box), 'periodic_wrapping':'[0, box)',
            'radius_definition':'native Group_R_Crit200; proper radius = comoving radius * a / h',
            'mass_definition':'native Group_M_Crit200 in Msun/h when available'},
        'automatic_candidate_count': len(rows),
        'source_hashes': {q['path']: q['sha256'] for q in cat['sources']+cat['snapshot_sources']},
        'targets': targets, 'remaining_selected_targets': len(set(rows)-set(chosen)),
        'science_acceptance':'NOT_ACCEPTED',
        'scope':'Epoch-local candidate selection only; trace exact IDs and measure phase fidelity, descendant recovery, resolution and contamination before science use.'}


def select_zoom_targets_main():
    """Print a read-only, source-bound candidate selection from a saved FOF frame."""
    import argparse
    parser = argparse.ArgumentParser(description=select_zoom_targets_main.__doc__)
    parser.add_argument('parent_run', type=Path)
    parser.add_argument('--snapshot', type=int, required=True)
    parser.add_argument('--memory-gib', type=float, required=True)
    parser.add_argument('--minimum-dm-particles', type=int, default=1000,
        help='Candidate support floor, not a convergence criterion (default: 1000)')
    parser.add_argument('--maximum-dm-particles', type=int)
    parser.add_argument('--maximum-targets', type=int, default=1)
    parser.add_argument('--ranking', choices=['dm_count_descending','catalogue_order'], default='dm_count_descending')
    parser.add_argument('--host-row', type=int, action='append', default=[],
        help='Prioritize an exact epoch-local FOF row; repeat for multiple rows')
    args = parser.parse_args()
    rules = {'minimum_dm_particles':args.minimum_dm_particles, 'maximum_targets':args.maximum_targets,
             'ranking':args.ranking}
    if args.maximum_dm_particles is not None:
        rules['maximum_dm_particles'] = args.maximum_dm_particles
    result = select_catalogue_targets(args.parent_run, args.snapshot, rules, args.host_row, memory_gib=args.memory_gib)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


def _cell_keys(q, box, parent_level):
    """Flatten wrapped grid cells: key = (i*N+j)*N+k, N=2**level."""
    if type(parent_level) is not int or not 1 <= parent_level <= 20:
        raise ValueError('Parent cell level must be an exact integer from 1 through 20')
    _positive_measurement(box,'Comoving parent box')
    q = np.asarray(q, float)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all() or np.any(q < 0) or np.any(q >= box):
        raise ValueError('Wrapped native Lagrangian positions required')
    side = 2**parent_level
    indices = np.floor(q/box*side).astype(np.uint64)
    return (indices[:,0]*np.uint64(side)+indices[:,1])*np.uint64(side)+indices[:,2]


def map_lagrangian_target_to_zoom_ids(selected_particles, zoom_ic, *, parent_level, box_comoving_kpc_h,
                                     memory_gib, maximum_tracers):
    """Cell-volume material assignment; coordinates are original MUSIC q.

    Coarse particles are excluded. A parent cell contributes each fine DM
    representative whose original q lies in that cell. This records material
    overlap and does not imply persistent galaxy or bound-core identity.
    """
    import h5py
    from core.data import exact_uint64_ids
    if type(parent_level) is not int or not 1 <= parent_level <= 20:
        raise ValueError('Supported exact parent-cell grid level required')
    if type(maximum_tracers) is not int or maximum_tracers < 1:
        raise ValueError('Positive maximum material tracer census required')
    budget = int(_positive_measurement(memory_gib,'Material mapping memory')*1024**3)
    sources = {str(Path(p).resolve()): file_sha256(p) for p in (selected_particles, zoom_ic)}
    with h5py.File(selected_particles) as source:
        q = source['PartType1/LagrangianCoordinates']
        if len(q)*64 > budget/2:
            raise MemoryError('Parent material cell set exceeds declared memory')
        positions = q[:]
        membership = source.get('PartType1/SelectedHostMember')
        if membership is not None:
            mask = _exact_support_mask(membership[:],(len(positions),))
            positions = positions[mask]
        parent_cells = np.unique(_cell_keys(positions, box_comoving_kpc_h, parent_level))
        if not len(parent_cells):
            raise ValueError('Selected parent host has no original material cells')
    selected = []
    total = 0
    with h5py.File(zoom_ic) as source:
        if int(source['Header'].attrs.get('Music_lagrangian_schema', 0)) != 1:
            raise ValueError('Zoom needs original native Lagrangian coordinates')
        group = source['PartType1']
        for first in range(0, len(group['ParticleIDs']), max(1, min(100000, budget//256))):
            stop = min(first+min(100000, budget//256), len(group['ParticleIDs']))
            if stop <= first:
                raise MemoryError('Material recovery memory budget is too small')
            q = group['LagrangianCoordinates'][first:stop]
            keys = _cell_keys(q, box_comoving_kpc_h, parent_level)
            mask = np.isin(keys, parent_cells)
            ids = exact_uint64_ids(group['ParticleIDs'][first:stop])[mask]
            total += len(ids)
            if total > maximum_tracers or total*32+parent_cells.nbytes*3 > budget:
                raise MemoryError('Exact descendant tracer census exceeds recorded bounds')
            selected.append(ids)
    ids = np.sort(np.concatenate(selected)) if selected else np.empty(0, np.uint64)
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Fine zoom material selection empty or duplicate')
    for p, expected in sources.items():
        if file_sha256(p) != expected:
            raise ValueError('Material mapping source changed')
    return ids, {'schema': 'parent-zoom-material-map-v1', 'mapping': 'original q parent-cell volume overlap',
        'parent_level': parent_level, 'box_comoving_kpc_h': box_comoving_kpc_h,
        'parent_cell_count': len(parent_cells), 'fine_zoom_tracer_count': len(ids),
        'source_hashes': sources, 'identity_scope': 'zoom exact particle namespace; material continuity, not galaxy ancestry'}


def recover_lagrangian_descendant(zoom_run, tracer_ids, snapshots, rules, *, memory_gib):
    """Preserve every candidate, ambiguous split, unbound material and disruption."""
    from core.zoom import read_host_catalogue
    from core.halos import snapshot_id_shards
    from core.data import exact_uint64_ids
    import h5py
    ids = exact_uint64_ids(tracer_ids)
    if len(np.unique(ids)) != len(ids) or not len(ids):
        raise ValueError('Unique nonempty exact material tracer IDs required')
    if len(set(snapshots)) != len(snapshots) or not snapshots:
        raise ValueError('Distinct requested recovery epochs required')
    if not 0 < rules['minimum_material_fraction'] <= 1 or not 0 <= rules['maximum_runner_up_fraction'] < rules['minimum_material_fraction']:
        raise ValueError('Explicit material match/ambiguity thresholds required')
    if type(rules['minimum_shared_particles']) is not int or rules['minimum_shared_particles'] < 1:
        raise ValueError('Explicit material match count threshold required')
    ids = np.sort(ids)
    records = []
    sources = {}
    for snapshot in snapshots:
        cat = read_host_catalogue(zoom_run, snapshot, memory_gib=memory_gib)
        n = cat['group_count']
        if len(ids)*32+n*64 > memory_gib*1024**3/2:
            raise MemoryError('Descendant vote workspace exceeds declared memory')
        votes = np.zeros(n, dtype=np.int64)
        seen = np.zeros(len(ids), bool)
        starts, lengths = cat['offsets'][:,1], cat['lengths'][:,1]
        active = np.flatnonzero(lengths > 0)
        order = np.argsort(starts[active], kind='stable')
        active = active[order]
        starts = starts[active]
        ends = starts+lengths[active]
        shards, _ = snapshot_id_shards(cat['snapshots'], 1)
        unbound = 0
        for shard in shards:
            with h5py.File(shard['path']) as handle:
                group = handle.get('PartType1')
                if group is None:
                    continue
                for first in range(0, len(group['ParticleIDs']), 100000):
                    values = exact_uint64_ids(group['ParticleIDs'][first:first+100000])
                    indices = np.searchsorted(ids, values)
                    valid = indices < len(ids)
                    valid[valid] = ids[indices[valid]] == values[valid]
                    matched = indices[valid]
                    if len(np.unique(matched)) != len(matched) or seen[matched].any():
                        raise ValueError('Duplicate exact tracer in recovery snapshot')
                    seen[matched] = True
                    locations = shard['start']+first+np.flatnonzero(valid)
                    candidates = np.searchsorted(starts, locations, side='right')-1
                    grouped = candidates >= 0
                    grouped[grouped] &= locations[grouped] < ends[candidates[grouped]]
                    np.add.at(votes, active[candidates[grouped]], 1)
                    unbound += int((~grouped).sum())
        ranked = np.argsort(-votes, kind='stable')
        choices = [{'host_row': int(row), 'shared_particles': int(votes[row]),
                    'material_fraction': float(votes[row]/len(ids))} for row in ranked if votes[row] > 0]
        winner = choices[0] if choices else None
        runner = choices[1]['material_fraction'] if len(choices) > 1 else 0
        accepted = winner is not None and winner['shared_particles'] >= rules['minimum_shared_particles'] and winner['material_fraction'] >= rules['minimum_material_fraction'] and runner <= rules['maximum_runner_up_fraction']
        status = 'MATCHED' if accepted else 'DISRUPTED_OR_UNBOUND' if not choices else 'AMBIGUOUS_OR_WEAK_MATCH'
        source = {q['path']: q['sha256'] for q in cat['sources']+cat['snapshot_sources']}
        _merge(sources,source)
        records.append({'snapshot': snapshot, 'scale_factor': cat['frame'][0], 'status': status,
            'descendant': winner if accepted else None, 'candidates': choices,
            'present_tracers': int(seen.sum()), 'missing_tracers': int((~seen).sum()), 'unbound_tracers': unbound,
            'source_hashes': source})
    for path, checksum in sources.items():
        if file_sha256(path) != checksum:
            raise ValueError('Descendant recovery source changed')
    return {'schema': 'lagrangian-descendant-recovery-v1', 'rules': rules, 'tracer_count': len(ids),
        'records': records, 'status': 'PASS' if all(q['status'] == 'MATCHED' for q in records) else 'UNRESOLVED',
        'source_hashes': sources, 'science_acceptance': 'UNRESOLVED'}


def measure_region_adequacy(recovery, diagnostics, requirements, *, sidm=False):
    """Every requested history epoch requires measured contamination/force/science.

    Diagnostic measurements have retained native-source hashes and explicit
    units. Omitted SIDM scattering is bounded separately from coarse mass.
    """
    rows = []
    sources = {}
    by_epoch = {q['snapshot']: q for q in diagnostics}
    if len(by_epoch) != len(diagnostics):
        raise ValueError('Duplicate region diagnostic epochs')
    required_fields = {'coarse_mass_fraction', 'force_error', 'observable_error'}
    if sidm:
        required_fields |= {'kernel_coarse_support_fraction', 'omitted_scattering_upper_bound'}
    for field in required_fields:
        limit = requirements[field]
        if not isinstance(limit.get('units'), str) or not limit['units'] or isinstance(limit.get('maximum'), bool) or not math.isfinite(limit['maximum']) or limit['maximum'] < 0:
            raise ValueError('Finite nonnegative predeclared region limit and units required: '+field)
    for match in recovery['records']:
        row = by_epoch.get(match['snapshot'])
        failures = []
        if match['status'] != 'MATCHED':
            failures.append('unresolved material descendant')
        if row is None:
            failures.append('missing measured history diagnostics')
        else:
            if not math.isfinite(row.get('scale_factor', float('nan'))) or not math.isclose(row['scale_factor'], match['scale_factor'], rel_tol=0, abs_tol=1e-12):
                raise ValueError('Region measurements belong to a different actual native epoch')
            if row.get('measurement_origin') != 'native_evolved_zoom' or not row.get('source_hashes'):
                raise ValueError('Native evolved zoom measurement provenance required')
            if not set(match['source_hashes']).issubset(row['source_hashes']):
                raise ValueError('Adequacy diagnostics omit the current recovery native sources')
            for path, checksum in row['source_hashes'].items():
                if file_sha256(path) != checksum or path in match['source_hashes'] and checksum != match['source_hashes'][path]:
                    raise ValueError('Region diagnostics bind stale/different native data')
                if path in sources and sources[path] != checksum:
                    raise ValueError('Zoom region history contains conflicting source bindings')
                sources[path] = checksum
            for field in required_fields:
                measured = row[field]
                limit = requirements[field]
                value = measured['value']
                if isinstance(value, bool) or measured['units'] != limit['units'] or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid measured region diagnostic: ' + field)
                if field in {'coarse_mass_fraction', 'kernel_coarse_support_fraction'} and value > 1:
                    raise ValueError('Measured support/mass fraction is outside [0,1]')
                if measured.get('supported') is not True or value > limit['maximum']:
                    failures.append(field)
        rows.append({'snapshot': match['snapshot'], 'accepted': not failures, 'failures': failures})
    return {'schema': 'zoom-history-region-adequacy-v1', 'status': 'PASS' if rows and all(q['accepted'] for q in rows) else 'UNRESOLVED',
        'records': rows, 'requirements': requirements, 'sidm': sidm,
        'execution_status':'COMPLETE','source_hashes':sources,
        'science_acceptance': 'DECLARED_REGION_ONLY' if rows and all(q['accepted'] for q in rows) else 'UNRESOLVED'}


def _run_recipe(recipe, replacements, cwd, receipt, inputs):
    import re
    argv = []
    for value in recipe['argv']:
        if not isinstance(value, str):
            raise ValueError('Command recipe arguments must be strings')
        for key, replacement in replacements.items():
            value = value.replace('{'+key+'}', str(replacement))
        if re.search(r'\{[^{}]+\}',value):
            raise ValueError('Zoom recipe retains an unresolved path token')
        argv.append(value)
    extra = []
    for asset in recipe.get('input_files', []):
        path = path_at(replacements['request_base'],asset['path'])
        if file_sha256(path) != asset['sha256']:
            raise ValueError('Frozen zoom recipe input changed before invocation')
        extra.append(path)
    with receipt.with_suffix('.log').open('wb') as log:
        run_native_command(argv, cwd=cwd, receipt_path=receipt, inputs=tuple(inputs)+tuple(extra),
                           timeout=recipe['wall_seconds'], stdout=log, stderr=log)


def _publish_attempt_pointer(target_directory, index, attempt, record):
    """Sealed per-variant generation locator; interrupted payloads stay intact."""
    from core.foundation import atomic_text
    directory = Path(target_directory).resolve()
    generation = Path(attempt).resolve()
    if generation.parent != directory or not generation.name.startswith(f'attempt_{index:03d}'):
        raise ValueError('Zoom attempt pointer must own one target generation')
    receipt = generation/'attempt.json'
    if record['execution_status'] == 'COMPLETE':
        atomic_text(receipt.with_name('attempt.json.sha256'),file_sha256(receipt)+'\n')
    pointer = directory/f'attempt_{index:03d}.current.json'
    atomic_json(pointer,{'schema':'zoom-attempt-pointer-v1','variant_index':index,
        'generation':generation.name,'signature':record['signature'],
        'execution_status':record['execution_status']})
    atomic_text(pointer.with_name(pointer.name+'.sha256'),file_sha256(pointer)+'\n')


def _retained_attempt_path(target_directory, index, signature):
    directory = Path(target_directory).resolve()
    default = directory/f'attempt_{index:03d}'
    pointer = directory/f'attempt_{index:03d}.current.json'
    if not pointer.exists():
        return default
    if file_sha256(pointer) != pointer.with_name(pointer.name+'.sha256').read_text().strip():
        raise ValueError('Zoom attempt generation pointer changed')
    value = json.loads(pointer.read_text())
    name = value.get('generation')
    if value.get('schema') != 'zoom-attempt-pointer-v1' or value.get('variant_index') != index or value.get('signature') != signature:
        raise ValueError('Zoom attempt pointer belongs to another variant/request')
    if not isinstance(name,str) or Path(name).name != name or not name.startswith(f'attempt_{index:03d}'):
        raise ValueError('Zoom attempt pointer is not a target-owned generation')
    generation = (directory/name).resolve()
    if generation.parent != directory or not (generation/'attempt.json').is_file():
        raise ValueError('Zoom attempt generation is missing or escaped its target')
    return generation


def _verify_selection_preview(binding, base, selection):
    """Bind an operator-reviewed selection to the actual just-read catalogue."""
    path = path_at(base,binding['path'])
    if file_sha256(path) != binding['sha256']:
        raise ValueError('Reviewed selection changed during campaign preflight')
    preview = json.loads(path.read_text())
    if not isinstance(preview,dict) or preview.get('schema') != 'zoom-target-selection-v1' or preview != selection:
        raise ValueError('Reviewed selection differs from the current source-bound target selection; inspect and prepare a new campaign')
    if file_sha256(path) != binding['sha256']:
        raise ValueError('Reviewed selection changed during comparison')


def run_zoom_campaign(request, destination, request_base=None, resume=False):
    from core.zoom import trace_catalogue_host, stage_zoom_inputs, generate_zoom, verify_generated_zoom
    from core.workflow import tree_hashes
    base = Path(request_base or '.').resolve()
    preflight = preflight_zoom_campaign(request,request_base=base)
    destination = Path(destination).resolve()
    parent, resources = request['parent'],request['resources']
    paths = {key:path_at(base,parent[key]) for key in
        ('run_dir','initial_ic','source_manifest','noise','spectrum_manifest')}
    generator = path_at(base,request['generator']['binary'])
    proof = path_at(base,request['generator']['native_validation'])
    epochs = request['epochs']
    sidm = request['physics']['interaction'] == 'elastic_sidm'
    variants = request['refinement_variants']
    evolution_inputs = preflight['prepared_evolution_inputs']
    selection = select_catalogue_targets(paths['run_dir'], parent['snapshot'], request['selection_rules'],
        request.get('explicit_host_rows', []), memory_gib=resources['memory_gib'])
    if request.get('selection_preview') is not None:
        _verify_selection_preview(request['selection_preview'],base,selection)
    with native_run_lock(destination, operation='automatic native zoom campaign'):
        destination.mkdir(parents=True, exist_ok=True)
        signature = canonical_digest({'request': request, 'selection': selection,
            'source_hash': file_sha256(__file__),
            'implementation_sources':{name:file_sha256(CORE_DIRECTORY/name) for name in
                ('zoom.py','sidm.py','evolution.py','data.py','halos.py','foundation.py',
                 'workflow.py','research/capabilities.py')}})
        campaign_path = destination/'zoom-campaign.json'
        if campaign_path.exists():
            if not resume:
                raise FileExistsError('Existing zoom campaigns require explicit resume')
            previous = json.loads(campaign_path.read_text())
            if previous.get('signature') != signature:
                raise ValueError('Zoom campaign inputs or implementation changed; choose a new destination')
        results = []
        def publish_campaign(status, *, active_target=None, active_attempt=None, failure=None):
            from core.foundation import atomic_text
            result = {'schema':'automatic-zoom-campaign-result-v1','signature':signature,
                'request_hash':canonical_digest(request),'selection':selection,'targets':results,
                'execution_status':status,'science_acceptance':'UNRESOLVED',
                'input_hashes':preflight['source_hashes'],
                'source_hashes':{str(Path(__file__).resolve()):file_sha256(__file__)},
                'active_target':active_target,'active_attempt':active_attempt,
                'failure':failure, 'bounded_target_attempts':preflight['maximum_target_attempts']}
            if status == 'COMPLETE':
                result['status'] = 'PASS' if results and all(row['region_status']=='ACCEPTED_DECLARED_HISTORY' for row in results) else 'UNRESOLVED'
            atomic_json(campaign_path,result)
            if status == 'COMPLETE':
                atomic_text(campaign_path.with_suffix('.sha256'),file_sha256(campaign_path)+'\n')
            return result
        publish_campaign('RUNNING')
        for target in selection['targets']:
            target_results = []
            for index, variant in enumerate(variants):
                target_directory = destination/target['target_id']
                attempt = _retained_attempt_path(target_directory,index,signature) if resume else target_directory/f'attempt_{index:03d}'
                receipt_path = attempt/'attempt.json'
                if resume and receipt_path.is_file():
                    saved = json.loads(receipt_path.read_text())
                    if saved['signature'] != signature:
                        raise ValueError('Zoom resume inputs/configuration changed; select a new destination')
                    if saved.get('execution_status') == 'COMPLETE':
                        seal = receipt_path.with_name('attempt.json.sha256')
                        if not seal.is_file() or file_sha256(receipt_path) != seal.read_text().strip():
                            raise ValueError('Completed zoom attempt is unsealed or changed')
                        for p, checksum in saved['files_sha256'].items():
                            if file_sha256(attempt/p) != checksum:
                                raise ValueError('Retained zoom attempt payload changed')
                        target_results.append(saved)
                        if saved['adequacy']['status'] == 'PASS' and not request.get('run_all_variants',False):
                            break
                        continue
                if attempt.exists():
                    # Preserve failed native evolution instead of overwriting it.
                    attempt = attempt.with_name(attempt.name+'_'+uuid.uuid4().hex[:12])
                    receipt_path = attempt/'attempt.json'
                attempt.mkdir(parents=True)
                record = {'schema': 'zoom-attempt-v1', 'signature': signature, 'target': target,
                          'variant': variant, 'execution_status': 'RUNNING', 'science_acceptance': 'UNRESOLVED'}
                atomic_json(receipt_path, record)
                _publish_attempt_pointer(target_directory,index,attempt,record)
                try:
                    publish_campaign('RUNNING',active_target=target['target_id'],active_attempt=str(attempt))
                    for path,digest in preflight['source_hashes'].items():
                        if file_sha256(path) != digest:
                            raise ValueError('Frozen campaign input changed before target/refinement evolution')
                    guard = variant['padding_comoving_kpc_h']
                    if sidm:
                        guard += request['sidm_guard']['maximum_comoving_support_kpc_h']*request['sidm_guard']['guard_multiplier']
                    region_dir = attempt/'traced'
                    trace_catalogue_host(paths['run_dir'], parent['snapshot'], target['host_row'],
                        paths['initial_ic'], paths['source_manifest'], region_dir,
                        aperture_r200c_factor=request['trace']['aperture_r200c_factor'],
                        padding_comoving_kpc_h=guard, memory_gib=resources['memory_gib'])
                    region_path = region_dir/'region.json'
                    staged = attempt/'staging'
                    staging_params = deepcopy(request['staging'])
                    if staging_params.get('wdm_request'):
                        staging_params['wdm_request']['lcdm_manifest'] = str(path_at(base,staging_params['wdm_request']['lcdm_manifest']))
                    for asset in (staging_params.get('phase_noise_by_level') or {}).values():
                        asset['path']=str(path_at(base,asset['path']))
                    if staging_params.get('phase_noise_by_level'):
                        staging_params['phase_noise_by_level']={level:asset for level,asset in staging_params['phase_noise_by_level'].items()
                            if int(level)<=variant['refinement_level']}
                    staging_params.update(refinement_level=variant['refinement_level'], seeds_by_level=variant['seeds_by_level'])
                    epoch_lineage = staging_params.pop('epoch_regeneration_lineage', None)
                    stage_zoom_inputs(paths['spectrum_manifest'], staged, **staging_params)
                    generated = attempt/'generation'
                    generate_zoom(staged/'zoom_template.conf', generator, generated,
                        native_validation=proof, traced_region=region_path,
                        threads=resources['omp_threads'], memory_gib=resources['memory_gib'],
                        native_memory_gib=resources['memory_gib'], native_disk_gib=resources['disk_gib'],
                        native_wall_hours=resources['wall_hours'], parent_ic=paths['initial_ic'], parent_noise=paths['noise'])
                    generated_report = verify_generated_zoom(generated/'generation.json')
                    if request['physics']['linear_model'] == 'wdm':
                        if not isinstance(generated_report,dict):
                            generated_report = json.loads((generated/'generation.json').read_text())
                        if not generated_report.get('wdm_thermal') or not generated_report.get('wdm_staging'):
                            raise ValueError('WDM zoom generation omitted physical thermal provenance')
                    zoom_run = attempt/'run'
                    replacements = {'python': sys.executable, 'suite': PROJECT_ROOT,
                        'request_base':base,
                        'generation': generated, 'ic': generated/'ics_gadget.hdf5', 'run': zoom_run,
                        'attempt': attempt, 'region': region_path, 'target_id': target['target_id']}
                    answers_path=publish_zoom_variant_answers(evolution_inputs[index],attempt/'native-answers.json')
                    runtime_flow=evolution_inputs[index].get('runtime_flow_request')
                    if runtime_flow is not None:
                        from core.music.staging import prepare_generated_flow_answers
                        runtime_flow=deepcopy(runtime_flow)
                        runtime_flow['answers']={'path':str(answers_path),'sha256':file_sha256(answers_path)}
                        flow=prepare_generated_flow_answers(generated/'generation.json',runtime_flow,attempt/'flow-prepared')
                        answers_path=Path(flow['answers']['path'])
                        record['flow_preparation']={'path':str(attempt/'flow-prepared/generated-preparation.json'),
                            'sha256':file_sha256(attempt/'flow-prepared/generated-preparation.json')}
                    replacements['answers']=answers_path
                    _run_recipe(request['evolution'],replacements,attempt,attempt/'evolution-invocation.json',
                        (generated/'generation.json',generated/'ics_gadget.hdf5',answers_path,
                         Path(evolution_inputs[index]['template_path'])))
                    ids, mapping = map_lagrangian_target_to_zoom_ids(region_dir/'selected_particles.hdf5',
                        generated/'ics_gadget.hdf5', parent_level=staging_params['parent_level'],
                        box_comoving_kpc_h=staging_params['box_mpc_h']*1000,
                        memory_gib=resources['memory_gib'], maximum_tracers=resources['maximum_tracers'])
                    recovery = recover_lagrangian_descendant(zoom_run, ids, request['recovery']['snapshots'],
                        request['recovery']['rules'], memory_gib=resources['memory_gib'])
                    epoch_coverage = require_history_epoch_coverage(recovery, epochs['backbone_redshifts'],
                        maximum_redshift_error=epochs['maximum_redshift_error'], memory_gib=resources['memory_gib'])
                    atomic_json(attempt/'material-map.json', mapping)
                    atomic_json(attempt/'recovery.json', recovery)
                    replacements['recovery'] = attempt/'recovery.json'
                    _run_recipe(request['diagnostics'], replacements, attempt, attempt/'diagnostics-invocation.json',
                        (attempt/'recovery.json', generated/'generation.json', generated/'ics_gadget.hdf5'))
                    measured = json.loads((attempt/request['diagnostics']['output']).read_text())
                    if measured.get('schema') != 'native-zoom-history-diagnostics-v1' or measured.get('recovery_sha256') != file_sha256(attempt/'recovery.json'):
                        raise ValueError('Region diagnostics do not bind this exact descendant recovery')
                    adequacy = measure_region_adequacy(recovery, measured['epochs'], request['region_requirements'], sidm=sidm)
                    atomic_json(attempt/'region-adequacy.json',adequacy)
                    record.update(execution_status='COMPLETE', adequacy=adequacy,
                        material_mapping=mapping, recovery=recovery, epoch_coverage=epoch_coverage, fine_guard_comoving_kpc_h=guard,
                        phase_matching='existing generator parent-noise/native phase controls',
                        epoch_lineage=epoch_lineage,
                        source_hashes=selection['source_hashes'])
                    record['files_sha256'] = {k: v for k, v in tree_hashes(attempt).items() if k not in {'attempt.json','attempt.json.sha256'}}
                    atomic_json(receipt_path, record)
                    _publish_attempt_pointer(target_directory,index,attempt,record)
                    target_results.append(record)
                    if adequacy['status'] == 'PASS' and not request.get('run_all_variants',False):
                        break
                except BaseException as exc:
                    record.update(execution_status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'FAILED', reason=str(exc))
                    atomic_json(receipt_path, record)
                    _publish_attempt_pointer(target_directory,index,attempt,record)
                    publish_campaign(record['execution_status'],active_target=target['target_id'],active_attempt=str(attempt),failure=str(exc))
                    raise
            accepted = bool(target_results and target_results[-1]['adequacy']['status'] == 'PASS')
            if request.get('run_all_variants',False):
                accepted = len(target_results)==len(variants) and all(row['adequacy']['status']=='PASS' for row in target_results)
            results.append({'target': target, 'attempts': target_results, 'region_status': 'ACCEPTED_DECLARED_HISTORY' if accepted else 'UNRESOLVED',
                            'remaining_requirement': None if accepted else 'Explicit region/resolution refinement bound exhausted'})
            publish_campaign('RUNNING')
        return publish_campaign('COMPLETE')


def zoom_campaign_main():
    import argparse
    parser = argparse.ArgumentParser(description='Trace, stage, evolve, recover and refine bounded native zooms')
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--request-base', type=Path, help='Base for relative campaign inputs; default is the campaign file directory')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = run_zoom_campaign(json.loads(args.campaign.read_text()), args.output,
                              request_base=args.request_base or args.campaign.parent, resume=args.resume)
    print(json.dumps(result, indent=2))


def require_history_epoch_coverage(recovery, requested_redshifts, *, maximum_redshift_error, memory_gib):
    """Bind each requested epoch to a distinct actual saved recovery epoch.

    Native clock quantization is allowed only within the explicitly declared
    redshift error. Different epoch support never inherits acceptance by row
    number, interpolation or assumed linear growth.
    """
    if isinstance(maximum_redshift_error, bool) or not math.isfinite(maximum_redshift_error) or maximum_redshift_error < 0:
        raise ValueError('Explicit finite nonnegative epoch matching error required')
    if not requested_redshifts or len(set(requested_redshifts)) != len(requested_redshifts):
        raise ValueError('Distinct requested history epochs required')
    if isinstance(memory_gib, bool) or not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError('Explicit positive epoch-assignment memory budget required')
    if len(requested_redshifts)*len(recovery['records'])*48 > memory_gib*1024**3:
        raise MemoryError('Actual/requested epoch assignment exceeds declared workspace')
    if len({q['snapshot'] for q in recovery['records']}) != len(recovery['records']):
        raise ValueError('Actual epoch coverage contains duplicate saved snapshot identities')
    actual = []
    for row in recovery['records']:
        a = row['scale_factor']
        if isinstance(a, bool) or not math.isfinite(a) or a <= 0:
            raise ValueError('Finite positive actual native scale factor required')
        actual.append((row['snapshot'], 1./a-1.))
    if len({row['scale_factor'] for row in recovery['records']}) != len(recovery['records']):
        raise ValueError('Actual epoch coverage contains duplicate saved physical epochs')
    # Solve the whole rectangular assignment so one early nearest choice
    # cannot consume the only supported output for a later requested epoch.
    from scipy.optimize import linear_sum_assignment
    requested = np.asarray(requested_redshifts, dtype=float)
    if not np.isfinite(requested).all() or np.any(requested <= -1):
        raise ValueError('Finite supported requested redshifts required')
    if any(isinstance(z, bool) for z in requested_redshifts):
        raise ValueError('Boolean values are not requested epochs')
    if len(actual) < len(requested):
        raise ValueError('Too few actual outputs for the requested history')
    cost = np.abs(requested[:,None]-np.asarray([z for _,z in actual])[None,:])
    allowed = cost <= maximum_redshift_error
    cost = np.where(allowed, cost, np.inf)
    try:
        rows, columns = linear_sum_assignment(cost)
    except ValueError:
        raise ValueError('Actual native outputs do not cover every requested epoch within its error bound') from None
    if len(rows) != len(requested) or not np.isfinite(cost[rows, columns]).all():
        raise ValueError('Actual native outputs do not cover every requested epoch')
    optimum = float(cost[rows, columns].sum())
    for row, column in zip(rows, columns):
        alternate = cost.copy()
        alternate[row, column] = np.inf
        try:
            other_rows, other_columns = linear_sum_assignment(alternate)
        except ValueError:
            continue
        if len(other_rows) == len(requested) and float(alternate[other_rows, other_columns].sum()) == optimum:
            raise ValueError('Requested zoom history has ambiguous equally supported saved-output assignments')
    assignments = {int(row): int(column) for row,column in zip(rows, columns)}
    matches = []
    for row, redshift in enumerate(requested_redshifts):
        column = assignments[row]
        snapshot, z = actual[column]
        matches.append({'requested_redshift': redshift, 'actual_redshift': z, 'snapshot': snapshot,
                        'absolute_redshift_error': float(cost[row, column])})
    return {'schema': 'zoom-requested-actual-epoch-coverage-v1', 'status': 'PASS',
            'maximum_redshift_error': maximum_redshift_error, 'matches': matches}


# Zoom diagnostics


def validate_zoom_diagnostic_request(request, *, snapshots=None, sidm=None, memory_gib=None):
    """Require all configured measurement producers before zoom evolution."""
    if not isinstance(request,dict):
        raise ValueError('Frozen native zoom diagnostic request required')
    for name in ('aperture_r200c_factor','fine_dm_mass_msun','memory_gib'):
        _positive_measurement(request[name],name)
    from core.zoom import _fine_mass_upper
    _fine_mass_upper(request['fine_dm_mass_msun'], request.get('fine_dm_mass_range_msun'))
    if memory_gib is not None and request['memory_gib'] > memory_gib:
        raise MemoryError('Zoom diagnostic request exceeds its campaign allocation')
    limit = request['maximum_coarse_mass_fraction']
    if isinstance(limit,bool) or not isinstance(limit,(int,float)) or not math.isfinite(limit) or not 0 <= limit <= 1:
        raise ValueError('Explicit native coarse mass fraction limit required')
    if type(request.get('sidm',False)) is not bool:
        raise ValueError('Zoom diagnostic SIDM selection must be a boolean')
    if sidm is not None and request.get('sidm',False) != sidm:
        raise ValueError('Zoom diagnostic interaction declaration differs from its campaign')
    controls = request['epoch_controls']
    if not isinstance(controls,dict) or not controls:
        raise ValueError('Native zoom history requires explicit per-epoch control producers')
    if snapshots is not None and any(str(snapshot) not in controls for snapshot in snapshots):
        raise ValueError('Zoom diagnostics omit a requested recovery epoch')
    for snapshot,row in controls.items():
        if not isinstance(snapshot,str) or not snapshot.isdigit() or not isinstance(row,dict):
            raise ValueError('Diagnostic controls need exact native snapshot keys')
        for kind in ('force','observable'):
            reference = row[kind]
            if not isinstance(reference,dict) or not ((reference.get('path') and reference.get('sha256')) or (reference.get('receipt') and reference.get('product'))):
                raise ValueError('Native zoom '+kind+' control needs a frozen path/hash or actual sealed stage product')
            _positive_measurement(reference['dimensional_floor_km2_s2_kpc' if kind == 'force' else 'dimensional_floor'],kind+' floor')
        for name in ('residual_array','reference_array','support_array'):
            if not isinstance(row['observable'].get(name),str) or not row['observable'][name]:
                raise ValueError('Observable zoom control lacks an actual array contract: '+name)
        if request.get('sidm') and 'native_sidm_support' not in request:
            reference = row['sidm_support']
            if not ((reference.get('path') and reference.get('sha256')) or (reference.get('receipt') and reference.get('product'))):
                raise ValueError('SIDM zoom support requires a native producer or frozen measured support report')
    if request.get('native_sidm_support') is not None:
        if not request.get('sidm'):
            raise ValueError('Native SIDM support cannot accompany collisionless diagnostics')
        native = request['native_sidm_support']
        for key in ('chunk_rows','maximum_native_particles','maximum_selected_particles'):
            if type(native[key]) is not int or native[key] < 1:
                raise ValueError('Bounded native SIDM aperture intake required: '+key)
        if native['chunk_rows']*256 > request['memory_gib']*1024**3:
            raise MemoryError('Native SIDM diagnostic stream exceeds declared workspace')
        if native.get('omitted_bound_scope') != 'cumulative_since_native_initialization':
            raise ValueError('Native omitted scattering bounds retain the complete initialized source history')
    return request


def _native_support_aperture(cat, center, radius, request):
    """Reduce emitted coarse-support fields for every fine DM particle in the aperture.

    Optical depth is a cumulative omitted-scattering upper bound, and kernel
    fraction is the actual most recently measured partner support. Neither
    field is reconstructed from coarse mass fractions or invented events.
    """
    import h5py
    from core.data import verify_native_hdf5_set, native_publication_markers
    from core.zoom import _fine_mass_upper
    options = request['native_sidm_support']
    paths = cat['snapshots']
    verify_native_hdf5_set(paths, require_publication=True)
    a, box, h = cat['frame']
    if radius >= box*a/h/2:
        raise ValueError('SIDM diagnostic aperture must fit within half the proper periodic box')
    fine_upper = _fine_mass_upper(request['fine_dm_mass_msun'],request.get('fine_dm_mass_range_msun'))
    fields = {'kernel_coarse_support_fraction':'SIDMKernelCoarseSupportFraction',
        'omitted_scattering_upper_bound':'SIDMOmittedScatteringUpperBound'}
    maxima = {name:0. for name in fields}
    selected = examined = 0
    sources = {str(Path(path).resolve()):file_sha256(path) for path in [*paths,*native_publication_markers(paths)]}
    for path in paths:
        with h5py.File(path,'r') as handle:
            group = handle.get('PartType1')
            if group is None:
                continue
            count = len(group['Coordinates'])
            examined += count
            if examined > options['maximum_native_particles']:
                raise MemoryError('Native SIDM aperture census exceeds its frozen work bound')
            names = ['SIDMCoarseGuardVersion',*fields.values()]
            if any(name not in group or group[name].shape != (count,) for name in names):
                raise ValueError('Native zoom snapshot lacks emitted coarse-support/hazard measurements')
            table_mass = float(handle['Header'].attrs['MassTable'][1])
            if 'Masses' not in group and (not math.isfinite(table_mass) or table_mass <= 0):
                raise ValueError('Fine SIDM particle selection requires actual positive masses')
            for begin in range(0,count,options['chunk_rows']):
                end = min(count,begin+options['chunk_rows'])
                position = np.asarray(group['Coordinates'][begin:end],float)
                mass = (np.asarray(group['Masses'][begin:end],float) if 'Masses' in group else np.full(end-begin,table_mass))*1e10/h
                if position.shape != (end-begin,3) or not np.isfinite(position).all() or not np.isfinite(mass).all() or np.any(mass <= 0):
                    raise ValueError('Native SIDM aperture positions/masses are invalid')
                delta = (position-np.asarray(center)+box/2)%box-box/2
                mask = (np.linalg.norm(delta,axis=1)*a/h <= radius)&(mass <= fine_upper*(1+1e-10))
                if not mask.any():
                    continue
                selected += int(mask.sum())
                if selected > options['maximum_selected_particles']:
                    raise MemoryError('Fine SIDM aperture selection exceeds the recorded bound')
                version = np.asarray(group['SIDMCoarseGuardVersion'][begin:end])[mask]
                if version.dtype.kind not in 'iu' or np.any(version != 1):
                    raise ValueError('Selected fine DM particles have no actual measured coarse-support state')
                for name,dataset in fields.items():
                    values = np.asarray(group[dataset][begin:end],float)[mask]
                    if not np.isfinite(values).all() or np.any(values < 0) or name == 'kernel_coarse_support_fraction' and np.any(values > 1):
                        raise ValueError('Native SIDM coarse-support field is outside its physical domain')
                    maxima[name] = max(maxima[name],float(values.max()))
    if not selected:
        raise ValueError('Native SIDM diagnostic aperture has no measured fine DM particles')
    for path,digest in sources.items():
        if file_sha256(path) != digest:
            raise ValueError('Native SIDM support source changed while streaming')
    return {name:{'units':'dimensionless','value':value,'supported':True,
        'measurement_count':selected,'supported_count':selected,
        'reduction':'maximum actual native field over every fine DM particle inside the aperture',
        'scope':options['omitted_bound_scope'] if name == 'omitted_scattering_upper_bound' else 'actual saved-epoch kernel support'}
        for name,value in maxima.items()}, sources


def _positive_measurement(value, label):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(label+' requires a finite positive numeric value')
    return float(value)


def _exact_support_mask(value, shape):
    mask = np.asarray(value)
    if mask.dtype.kind != 'b' or mask.shape != shape:
        raise ValueError('Measured support must be an exact boolean array matching every bin')
    return mask


def _control_archive_budget(archive,memory_gib):
    """Bound expanded NPZ bytes plus reduction workspace before any reader."""
    budget = _positive_measurement(memory_gib,'Diagnostic memory budget')*1024**3
    with zipfile.ZipFile(archive) as packed:
        members = packed.infolist()
        names = [q.filename for q in members]
        if len(names) != len(set(names)) or any(not q.endswith('.npy') for q in names):
            raise ValueError('Control archive needs distinct numeric NPY members')
        expanded = sum(q.file_size for q in members)
        if expanded*6 > budget:
            raise MemoryError('Expanded native control arrays and reduction workspace exceed declared memory')


@contextmanager
def _bounded_control_arrays(archive,memory_gib):
    _control_archive_budget(archive,memory_gib)
    with np.load(archive,allow_pickle=False) as arrays:
        yield arrays


def _error_summary(error, support):
    values = np.asarray(error,float)
    mask = _exact_support_mask(support,values.shape)
    if not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Finite nonnegative measured relative errors required')
    flat = values.reshape(-1)
    worst = int(np.argmax(flat))
    return {'units':'dimensionless','value':float(flat[worst]),'supported':bool(mask.all()),
        'measurement_count':int(values.size),'supported_count':int(mask.sum()),
        'worst_flat_index':worst,'array_shape':list(values.shape),
        'reduction':'maximum over every retained measured bin; unsupported bins prevent acceptance'}


def _control(reference, base, native_sources, memory_gib):
    from core.data import verify_numeric_audit, numeric_audit_arrays_path
    if reference.get('receipt') is not None:
        from core.workflow import verify_completed_receipt
        receipt_path=path_at(base,reference['receipt'])
        receipt=verify_completed_receipt(receipt_path)
        product=receipt['products'][reference['product']]
        if product['kind']!='file':
            raise ValueError('Zoom control must be an actual measured file product')
        path=(Path(receipt['generation'])/product['path']).resolve()
        if receipt['files_sha256'].get(product['path'])!=file_sha256(path):
            raise ValueError('Zoom control differs from its actual completed producer')
    else:
        path = path_at(base, reference['path'])
        if file_sha256(path) != reference['sha256']:
            raise ValueError('Zoom diagnostic control receipt changed')
    # Existing audit intake also expands arrays: bound its immutable sibling
    # archive before invoking that reader, not only before our reduction.
    _control_archive_budget(path.with_name(path.name+'.data')/'arrays.npz',memory_gib)
    report = verify_numeric_audit(path)
    from core.data import numeric_audit_source_bindings
    sources = numeric_audit_source_bindings(report)
    if not all(sources.get(p) == expected for p, expected in native_sources.items()):
        raise ValueError('Zoom diagnostic control omits current native snapshot/catalogue sources')
    for path_name, expected in sources.items():
        if file_sha256(path_name) != expected:
            raise ValueError('Zoom diagnostic source changed')
    archive = numeric_audit_arrays_path(path, report)
    return report, archive, {**sources, str(path): file_sha256(path), str(archive): file_sha256(archive)}


def measure_native_zoom_history(run_dir, recovery_path, request, *, request_base):
    from core.zoom import read_host_catalogue, contamination_epoch
    validate_zoom_diagnostic_request(request)
    base = Path(request_base).resolve()
    recovery = json.loads(Path(recovery_path).read_text())
    if recovery.get('schema') != 'lagrangian-descendant-recovery-v1':
        raise ValueError('Native diagnostics require exact Lagrangian recovery records')
    aperture = _positive_measurement(request['aperture_r200c_factor'],'Diagnostic aperture factor')
    _positive_measurement(request['fine_dm_mass_msun'],'Fine DM mass')
    _positive_measurement(request['memory_gib'],'Diagnostic memory budget')
    recovery_digest = file_sha256(recovery_path)
    rows = []
    for match in recovery['records']:
        if match['status'] != 'MATCHED':
            continue  # Recovery retains the missing epoch; adequacy cannot accept it.
        snapshot = match['snapshot']
        cat = read_host_catalogue(run_dir, snapshot, memory_gib=request['memory_gib'])
        row = match['descendant']['host_row']
        if type(row) is not int or not 0 <= row < cat['group_count']:
            raise ValueError('Recovered host row is outside the current native catalogue')
        a, box, h = cat['frame']
        if not math.isclose(a,match['scale_factor'],rel_tol=0,abs_tol=1e-12):
            raise ValueError('Recovery and diagnostics use different native epochs')
        radius = float(cat['radii'][row])*a/h*aperture
        _positive_measurement(radius,'Recovered diagnostic aperture')
        contamination = contamination_epoch(cat['snapshots'], snapshot,
            center_comoving_kpc_h=cat['positions'][row], radius_proper_kpc=radius,
            fine_dm_mass_msun=request['fine_dm_mass_msun'],
            fine_dm_mass_range_msun=request.get('fine_dm_mass_range_msun'),
            maximum_coarse_mass_fraction=request['maximum_coarse_mass_fraction'])
        sources = {q['path']: q['sha256'] for q in cat['sources']+cat['snapshot_sources']}
        if any(sources.get(p) != expected for p,expected in match['source_hashes'].items()):
            raise ValueError('Recovery sources differ from the current native catalogue/snapshot')
        diagnostic = {'snapshot': snapshot, 'host_row': int(row), 'scale_factor': a,
            'measurement_origin': 'native_evolved_zoom', 'radius_proper_kpc': radius,
            'coarse_mass_fraction': {'units': 'dimensionless',
                'value': contamination['coarse_mass_fraction'] if contamination['coarse_mass_fraction'] is not None else 1.,
                'supported': contamination['status'] != 'EMPTY_REGION'},
            'source_hashes': dict(sources), 'native_contamination': contamination}
        controls = request['epoch_controls'][str(snapshot)]
        force, archive, binding = _control(controls['force'], base, sources,request['memory_gib'])
        force_fields = force['numeric_data']['array_contract']['datasets']
        for name in ('validation_force_absolute_error','validation_acceleration_direct_km2_s2_kpc'):
            if force_fields[name]['units'] != '(km/s)^2/proper kpc':
                raise ValueError('Zoom force controls require acceleration in (km/s)^2/proper kpc')
        with _bounded_control_arrays(archive,request['memory_gib']) as arrays:
            residual = np.asarray(arrays['validation_force_absolute_error'], float)
            acceleration = np.asarray(arrays['validation_acceleration_direct_km2_s2_kpc'], float)
            floor = _positive_measurement(controls['force']['dimensional_floor_km2_s2_kpc'],'Force dimensional floor')
            if residual.ndim != 1 or not len(residual) or acceleration.shape != (len(residual),3) or not np.isfinite(residual).all() or np.any(residual < 0) or not np.isfinite(acceleration).all():
                raise ValueError('Actual withheld force measurements and dimensional floor required')
            # |delta g|/max(|g_reference|, g_floor) is dimensionless;
            # both accelerations and the floor use km^2 s^-2 kpc^-1.
            norm = np.hypot.reduce(np.abs(acceleration),axis=1)
            if not np.isfinite(norm).all():
                raise ValueError('Withheld acceleration norm is outside the supported numeric range')
            error = residual/np.maximum(norm,floor)
            if not np.isfinite(error).all():
                raise ValueError('Nonfinite withheld force measurements')
            supported = force.get('field_validation_status') == 'PASS' or force.get('status') == 'PASS'
            diagnostic['force_error'] = {**_error_summary(error,np.full(error.shape,supported,dtype=bool)),
                'reference':'withheld reconstructed field vs isolated Newtonian source; periodic force is a separate control'}
        _merge(diagnostic['source_hashes'],binding)
        observable, archive, binding = _control(controls['observable'], base, sources,request['memory_gib'])
        fields = observable['numeric_data']['array_contract']['datasets']
        reference = controls['observable']
        units = fields[reference['residual_array']]['units']
        if not isinstance(units,str) or not units or units.startswith('UNDECLARED') or units != fields[reference['reference_array']]['units']:
            raise ValueError('Zoom observable residual and reference must have the same physical units')
        with _bounded_control_arrays(archive,request['memory_gib']) as arrays:
            residual = np.asarray(arrays[controls['observable']['residual_array']], float)
            scale = np.asarray(arrays[controls['observable']['reference_array']], float)
            floor = _positive_measurement(controls['observable']['dimensional_floor'],'Observable dimensional floor')
            if not math.isfinite(floor) or floor <= 0 or residual.shape != scale.shape or residual.size == 0 or not np.isfinite(residual).all() or not np.isfinite(scale).all():
                raise ValueError('Measured observable residual/reference with dimensional floors required')
            # The residual, reference and floor carry the same observable
            # units. Retain every bin and its exact measured support flag.
            error = np.abs(residual)/np.maximum(np.abs(scale), floor)
            supported = _exact_support_mask(arrays[controls['observable']['support_array']],residual.shape)
            diagnostic['observable_error'] = _error_summary(error,supported)
        _merge(diagnostic['source_hashes'],binding)
        if request.get('sidm') and request.get('native_sidm_support') is not None:
            support, binding = _native_support_aperture(cat,cat['positions'][row],radius,request)
            diagnostic.update(support)
            _merge(diagnostic['source_hashes'],binding)
        elif request.get('sidm'):
            support, archive, binding = _control(controls['sidm_support'], base, sources,request['memory_gib'])
            with _bounded_control_arrays(archive,request['memory_gib']) as arrays:
                for key in ('kernel_coarse_support_fraction', 'omitted_scattering_upper_bound'):
                    measured = np.asarray(arrays[key], float)
                    if not measured.size or not np.isfinite(measured).all() or np.any(measured < 0):
                        raise ValueError('Measured SIDM support/hazard upper bounds required')
                    if key == 'kernel_coarse_support_fraction' and np.any(measured > 1):
                        raise ValueError('Measured SIDM kernel support fractions exceed unity')
                    diagnostic[key] = _error_summary(measured,np.full(measured.shape,support.get('status') == 'PASS',dtype=bool))
            _merge(diagnostic['source_hashes'],binding)
        rows.append(diagnostic)
    if file_sha256(recovery_path) != recovery_digest:
        raise ValueError('Native recovery changed during diagnostic measurement')
    for diagnostic in rows:
        for path,expected in diagnostic['source_hashes'].items():
            if file_sha256(path) != expected:
                raise ValueError('Native diagnostic source changed during measurement')
    return {'schema': 'native-zoom-history-diagnostics-v1', 'epochs': rows,
        'recovery_sha256': recovery_digest, 'source_hashes':{name:digest for row in rows for name,digest in row['source_hashes'].items()},
        'input_hashes':{str(Path(recovery_path).resolve()):recovery_digest},
        'execution_status':'COMPLETE','scientific_acceptance': 'UNRESOLVED'}


def zoom_diagnostic_input_bindings(parameters):
    from core.workflow import verify_completed_receipt
    request=validate_zoom_diagnostic_request(parameters['request'])
    base=Path(parameters['request_base']).resolve()
    recovery=Path(parameters['recovery_path']).resolve()
    if recovery.stat().st_size*8>request['memory_gib']*1024**3:
        raise MemoryError('Zoom recovery metadata exceeds its diagnostic allocation')
    report=json.loads(recovery.read_text())
    if report.get('schema')!='lagrangian-descendant-recovery-v1':
        raise ValueError('Zoom diagnostic input requires actual retained material recovery')
    sources={str(recovery):file_sha256(recovery)}
    for row in report['records']:
        for name,digest in row['source_hashes'].items():
            if file_sha256(name)!=digest:
                raise ValueError('Actual zoom recovery input changed')
            sources[name]=digest
    native_epochs={str(row['snapshot']):row['source_hashes'] for row in report['records']}
    for snapshot,row in request['epoch_controls'].items():
        for kind in ('force','observable','sidm_support'):
            if kind not in row:
                continue
            reference=row[kind]
            if reference.get('receipt') is not None:
                path=path_at(base,reference['receipt']);record=verify_completed_receipt(path)
                sources[str(path)]=file_sha256(path)
                _merge(sources,record['source_hashes'])
                _merge(sources,record['input_hashes'])
                _merge(sources,{str(Path(record['generation'])/name):digest for name,digest in record['files_sha256'].items()})
            else:
                path=path_at(base,reference['path'])
                if file_sha256(path)!=reference['sha256']:
                    raise ValueError('Retained zoom measurement input changed')
                sources[str(path)]=reference['sha256']
                if snapshot in native_epochs:
                    _,_,bindings=_control(reference,base,native_epochs[snapshot],request['memory_gib'])
                    _merge(sources,bindings)
    return sources


def native_zoom_diagnostic_runner(parameters,context):
    if parameters['request']['memory_gib']>context.resources['memory_gib']:
        raise MemoryError('Native zoom history diagnostic exceeds the allocated stage workspace')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation) or destination.exists():
        raise ValueError('Native zoom history diagnostics need a new owned publication path')
    result=measure_native_zoom_history(parameters['run_dir'],parameters['recovery_path'],parameters['request'],request_base=parameters['request_base'])
    atomic_json(destination,result)
    return {}


def zoom_diagnostics_main():
    import argparse
    p = argparse.ArgumentParser(description='Measure native zoom contamination and retained science convergence')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--recovery', type=Path, required=True)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Native diagnostics cannot overwrite history')
    result = measure_native_zoom_history(a.run, a.recovery, json.loads(a.request.read_text()), request_base=a.request.parent)
    atomic_json(a.output, result)


# Zoom recipe inputs


def validate_campaign_source_retention(staging):
    budget=staging.get('source_archive_max_mib')
    if budget is not None and (type(budget) is not int or not 0<budget<=(2**64-1)//1024**2 or staging.get('lpt_order',1)!=2):
        raise ValueError('Campaign source retention requires exact positive representable MiB budget and mixed2LPT staging')
    return budget


def prepare_zoom_evolution_inputs(request,*,request_base):
    from core.workflow import path_at
    from core.research.capabilities import validate_native_answers
    from commands import redshift_schedule
    evolution=request['evolution'];argv=evolution['argv']
    if not isinstance(argv,list) or any(not isinstance(arg,str) or not arg for arg in argv):
        raise ValueError('Zoom evolution requires a concrete public argv recipe')
    required={'--answers':'{answers}','--existing-ic':'{ic}',
        '--native-generation-report':'{generation}/generation.json','--run-dir':'{run}'}
    if argv[:3]!=['{python}','{suite}/commands.py','gadget']:
        raise ValueError('Zoom native evolution must use the exact actual public GADGET route')
    for flag,value in required.items():
        if argv.count(flag)!=1 or argv.index(flag)+1>=len(argv) or argv[argv.index(flag)+1]!=value:
            raise ValueError('Zoom evolution must bind its actual generated '+flag+' input')
    reference=evolution['answers_template'];path=path_at(request_base,reference['path'])
    if path.stat().st_size>16*1024**2 or file_sha256(path)!=reference['sha256']:
        raise ValueError('Frozen zoom physical-answer template changed or exceeds intake bound')
    template=json.loads(path.read_text());resources=request['resources'];epochs=request['epochs']
    runtime_flow=None;runtime_pins={}
    if evolution.get('flow_preparation') is not None:
        from core.music.staging import generated_flow_bindings
        ref=evolution['flow_preparation']
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('Exact generated-flow recipe binding required')
        recipe=path_at(request_base,ref['path'])
        if recipe.stat().st_size>16*1024**2 or file_sha256(recipe)!=ref['sha256']:raise ValueError('Zoom runtime flow recipe changed or exceeds intake bound')
        runtime_flow=json.loads(recipe.read_text());runtime_pins=generated_flow_bindings(runtime_flow)
        runtime_pins[str(recipe)]=ref['sha256']
        if validate_campaign_source_retention(request['staging']) is None:
            raise ValueError('Runtime original-flow preparation requires source retention in campaign staging')
    prepared=[]
    allowed={'particles_per_dimension','softening_kpc_h','softening_max_physical_kpc_h','pm_grid',
        'max_size_timestep_ln_a','min_size_timestep_ln_a','err_tol_int_accuracy','err_tol_force_acc','err_tol_theta'}
    interaction=request['physics']['interaction']
    if interaction not in {'collisionless','elastic_sidm'}:raise ValueError('Zoom interaction model is unsupported')
    for index,variant in enumerate(request['refinement_variants']):
        overrides=variant.get('evolution_overrides',{})
        if not isinstance(overrides,dict) or not set(overrides)<=allowed:
            raise ValueError('Zoom refinements may change only recorded resolution/integration settings')
        answers=deepcopy(template);answers.update(deepcopy(overrides))
        answers['name']='zoom_refinement_'+str(index)
        sidm=answers.get('sidm',{})
        if bool(sidm.get('enabled',False))!=(interaction=='elastic_sidm'):
            raise ValueError('Zoom evolution answers disagree with the declared interaction physics')
        schedule=redshift_schedule(str(answers['schedule']),epochs['z_start'])
        if schedule!=epochs['backbone_redshifts'] or not math.isclose(schedule[-1],epochs['z_end'],rel_tol=0,abs_tol=1e-10):
            raise ValueError('Zoom evolution changes the user-selected output history or endpoint')
        source=path_at(request_base,answers['cosmology'])
        expected=path_at(request_base,request['parent']['spectrum_manifest'])
        if source!=expected:raise ValueError('Zoom evolution does not use the exact pinned regenerated cosmology bundle')
        answers['cosmology']=str(source)
        actual_configuration=validate_native_answers(answers,resources)
        dependency_pins=dict(runtime_pins)
        if runtime_flow is not None:
            _merge(dependency_pins,generated_flow_bindings(runtime_flow,levels=range(request['staging']['parent_level'],variant['refinement_level']+1)))
        if answers.get('native_flow_window_comparison') is not None:
            from core.music.flow import original_flow_model_bindings
            _merge(dependency_pins,original_flow_model_bindings(answers['native_flow_window_comparison'],
                levels=range(request['staging']['parent_level'],variant['refinement_level']+1)))
        if interaction=='elastic_sidm':
            h=float(actual_configuration['cosmology']['H0'])/100.
            # Comoving kpc/h = proper kpc * h/a. A fixed proper support
            # has its largest comoving radius at the earliest requested a.
            support=float(sidm['kernel_support_proper_kpc'])*(1+epochs['z_start'])*float(h)
            if support>request['sidm_guard']['maximum_comoving_support_kpc_h']:
                raise ValueError('Actual native SIDM kernel exceeds the declared high-resolution comoving guard')
        prepared.append({'answers':answers,'answers_hash':canonical_digest(answers),
            'template_path':str(path),'template_sha256':reference['sha256'],'source_hashes':dependency_pins,'runtime_flow_request':deepcopy(runtime_flow)})
    return prepared


def publish_zoom_variant_answers(prepared,destination):
    path=Path(destination).resolve()
    if file_sha256(prepared['template_path'])!=prepared['template_sha256']:
        raise ValueError('Frozen native zoom physical template changed before evolution')
    if canonical_digest(prepared['answers'])!=prepared['answers_hash']:
        raise ValueError('Prepared zoom physical inputs changed before publication')
    for source,digest in prepared.get('source_hashes',{}).items():
        if file_sha256(source)!=digest:raise ValueError('Prepared zoom flow/model dependency changed before evolution')
    atomic_json(path,prepared['answers'])
    return path
