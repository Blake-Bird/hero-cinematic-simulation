"""Matched native SIDM/CDM lensing and stellar-stream forward responses.

Lenses use species-resolved thin-plane mass maps at a common recorded CLASS
background, not an H0 posterior. Stellar contrasts use supported, birth-matched
common IDs on a frozen CDM track, not a causal gap detection or a full-population
response. Missing native targets and unsupported samples remain explicit.
"""
from pathlib import Path
import copy
import json
import math
import numpy as np
from .foundation import atomic_json, canonical_digest, file_sha256, merge_source_bindings as _merge
from .data import exact_uint64_ids


# Species-resolved thin-lens response

def lensing_controls(path, digest, memory_gib):
    from .sidm_physics import _positive
    _positive(memory_gib, "native lens workspace")
    path = Path(path).resolve()
    if file_sha256(path) != digest:raise ValueError('Original native lensing controls changed')
    if path.stat().st_size * 24 > memory_gib * 1024**3 / 4:raise MemoryError('Native lensing controls exceed metadata allocation')
    settings = json.loads(path.read_text())
    required = {'schema', 'source_redshift', 'pixels', 'half_width_proper_kpc', 'half_depth_proper_kpc',
        'line_of_sight', 'horizontal_direction', 'background_surface_density_msun_kpc2',
        'minimum_particles_per_pixel', 'maximum_coarse_mass_fraction', 'maximum_particles', 'observables'}
    if set(settings) != required or settings['schema'] != 'native-recovered-sidm-lensing-controls-v1':
        raise ValueError('Complete explicit native lens geometry and numerical support controls required')
    for key in ('source_redshift', 'half_width_proper_kpc', 'half_depth_proper_kpc'):
        if type(settings[key]) not in (int, float) or not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError('Positive finite native lens geometry required')
    for key, minimum in [('pixels', 4), ('minimum_particles_per_pixel', 1), ('maximum_particles', 1)]:
        if type(settings[key]) is not int or settings[key] < minimum:raise ValueError('Exact positive native lens census/grid controls required')
    if settings['pixels']**2 * 1024 > memory_gib * 1024**3 / 4:raise MemoryError('Native lens grid exceeds allocation')
    if (type(settings['maximum_coarse_mass_fraction']) not in (int, float)
            or not math.isfinite(settings['maximum_coarse_mass_fraction']) or not 0 <= settings['maximum_coarse_mass_fraction'] <= 1):
        raise ValueError('Explicit finite native lens coarse contamination ceiling required')
    background = settings['background_surface_density_msun_kpc2']
    if type(background) not in (int, float) or not math.isfinite(background):raise ValueError('Explicit finite native lens mass sheet required')
    axis = np.asarray(settings['line_of_sight'], float); horizontal = np.asarray(settings['horizontal_direction'], float)
    if axis.shape != (3,) or horizontal.shape != (3,) or not np.isfinite(axis).all() or not np.isfinite(horizontal).all() or np.linalg.norm(axis) == 0:
        raise ValueError('Finite nondegenerate native lens projection directions required')
    axis_norm = np.linalg.norm(axis)
    horizontal_norm = np.linalg.norm(horizontal)
    if not math.isfinite(axis_norm) or not math.isfinite(horizontal_norm):
        raise ValueError('Native lens projection norm exceeds numeric range')
    axis = axis / axis_norm
    transverse_norm = np.linalg.norm(horizontal - np.dot(horizontal, axis)*axis)
    if not math.isfinite(transverse_norm) or transverse_norm < 1e-12:
        raise ValueError('Degenerate native lens projection basis')
    units = {'surface_density_msun_kpc2': 'Msun / proper kpc^2', 'kappa': 'dimensionless', 'shear1': 'dimensionless',
        'shear2': 'dimensionless', 'reduced_deflection_x_rad': 'rad', 'reduced_deflection_y_rad': 'rad', 'magnification_inverse': 'dimensionless'}
    rows = settings['observables']
    if not isinstance(rows, list) or not rows or len({r['array'] for r in rows}) != len(rows):raise ValueError('Distinct original native lens observable selectors required')
    for row in rows:
        if (set(row) != {'array', 'units', 'absolute_floor'} or units.get(row['array']) != row['units']
                or type(row['absolute_floor']) not in (int, float) or not math.isfinite(row['absolute_floor']) or row['absolute_floor'] <= 0):
            raise ValueError('Original lens observable unit and positive dimensional floor required')
    return settings


def lensing_inputs(parameters):
    from .sidm_products import sidm_matched_inputs, sidm_history_inputs
    memory = parameters['memory_gib']
    if type(memory) not in (int, float) or not math.isfinite(memory) or memory <= 0:raise ValueError('Positive finite native lens allocation required')
    template = Path(parameters['template_path']).resolve(); settings = lensing_controls(template, parameters['template_sha256'], memory)
    pins = {str(template): file_sha256(template)}; targets = {}; histories = {}
    pairpath = Path(parameters['pair_result']).resolve()
    if pairpath.stat().st_size * 24 > memory * 1024**3 / 4:raise MemoryError('Native lens pair metadata exceeds allocation')
    pair = json.loads(pairpath.read_text())
    if pair.get('schema') != 'native-recovered-sidm-pair-result-v1' or pair.get('execution_status') != 'COMPLETE':
        raise ValueError('Original completed native matched target result required')
    _merge(pins, {str(pairpath): file_sha256(pairpath)}); _merge(pins, pair['source_hashes'])
    for label in ('cdm', 'sidm'):
        path = Path(parameters[label+'_result']).resolve()
        if path.stat().st_size * 24 > memory * 1024**3 / 4:raise MemoryError('Native lens target metadata exceeds allocation')
        target = json.loads(path.read_text()); targets[label] = target
        if target.get('schema') != 'native-recovered-sidm-result-v1' or target.get('execution_status') != 'COMPLETE':
            raise ValueError('Completed original recovered native lens target required')
        _merge(pins, {str(path): file_sha256(path)}); _merge(pins, target['source_hashes'])
        if target['status'] == 'MEASURED_NATIVE_TARGET_HISTORY':
            reference = target['history']; product = Path(reference['path']).resolve()
            if not product.is_relative_to(path.parent) or file_sha256(product) != reference['sha256']:
                raise ValueError('Native lens history changed or escapes its generation')
            history, _, sources = sidm_history_inputs(product, run_directory=parameters[label+'_run_directory'], memory_gib=memory)
            if history['dark_matter_model'] != ('collisionless' if label == 'cdm' else 'elastic_sidm'):
                raise ValueError('Native lens comparison relabels actual evolved dark matter physics')
            if history['matching'] != target['matching'] or [r['snapshot'] for r in history['epochs']] != [r['snapshot'] for r in target['memberships']]:
                raise ValueError('Native lens target changes its original matching/saved epoch history')
            original_members = {r['snapshot']: r['selection'] for r in history['request']['epochs']}
            for row in target['memberships']:
                ref = row['product']; member = Path(ref['path']).resolve()
                if original_members.get(row['snapshot']) != ref:raise ValueError('Native lens population is not the original matched history membership')
                if not member.is_relative_to(path.parent) or file_sha256(member) != ref['sha256']:raise ValueError('Original native lens member product changed')
            histories[label] = history; _merge(pins, sources)
        elif target['status'] != 'UNRESOLVED_TARGET_HISTORY':raise ValueError('Unknown native lens target recovery status')
    if pair['status'] == 'MEASURED_NATIVE_MATCH':
        if len(histories) != 2:raise ValueError('Measured native lens pair lacks an original branch history')
        ref = pair['matched_history']; path = Path(ref['path']).resolve()
        if not path.is_relative_to(pairpath.parent) or file_sha256(path) != ref['sha256']:raise ValueError('Original matched lens history changed')
        request = {'schema': 'native-sidm-matched-figure-request-v1', 'matched_product': ref, 'history_base_directory': str(pairpath.parent),
            'native_runs': {label: parameters[label+'_run_directory'] for label in ('cdm', 'sidm')}}
        report, _, sources, _, _ = sidm_matched_inputs(request, memory_gib=memory)
        for row in report['request']['histories']:
            if row['product'] != targets[row['label']]['history']:raise ValueError('Lens controls do not bind these exact original paired target histories')
        _merge(pins, sources)
    elif pair['status'] != 'UNRESOLVED_NATIVE_MATCH':raise ValueError('Unknown native matched lens status')
    for p, h in pins.items():
        if file_sha256(p) != h:raise ValueError('Native lens input source changed')
    return settings, targets, pair, histories, pins


def plane_census(member, arrays, settings):
    """Conservative nearest-cell count and full-cylinder contamination census.

    The mass map remains the original CIC estimator. Count support is not an
    independent covariance or a claim of force/resolution convergence.
    """
    from .data import exact_uint64_ids
    ids = exact_uint64_ids(arrays['particle_ids']); n = len(ids)
    if n > settings['maximum_particles']:raise MemoryError('Native lens population exceeds original census ceiling')
    center = np.asarray(arrays['center_proper_kpc'], float); x = np.asarray(arrays['position_proper_kpc'], float)
    mass = np.asarray(arrays['mass_msun'], float); species = np.asarray(arrays['species'])
    if (center.shape != (3,) or x.shape != (n, 3) or mass.shape != (n,) or species.shape != (n,)
            or species.dtype.kind not in 'iu' or np.any(species < 0) or np.any(species > 5) or np.any(mass <= 0)
            or not np.isfinite(x).all() or not np.isfinite(mass).all() or not np.isfinite(center).all()):
        raise ValueError('Complete finite original native lens population required')
    radius = math.sqrt(2*settings['half_width_proper_kpc']**2 + settings['half_depth_proper_kpc']**2)
    if radius > member['request']['support_radius_proper_kpc']:raise ValueError('Original native aperture does not contain the complete lens cylinder')
    if np.any(np.linalg.norm(x-center, axis=1) > member['request']['support_radius_proper_kpc']*(1+1e-12)):
        raise ValueError('Native lens particles are not in the original unwrapped proper aperture chart')
    axis = np.asarray(settings['line_of_sight'], float).copy(); axis /= np.linalg.norm(axis)
    horizontal = np.asarray(settings['horizontal_direction'], float).copy(); horizontal -= np.dot(horizontal, axis)*axis; horizontal /= np.linalg.norm(horizontal)
    rel = x-center; xx = rel@horizontal; yy = rel@np.cross(axis, horizontal); depth = rel@axis; width = settings['half_width_proper_kpc']; pixels = settings['pixels']
    selected = (abs(xx) < width) & (abs(yy) < width) & (abs(depth) <= settings['half_depth_proper_kpc'])
    ix = np.floor((xx[selected]+width)/(2*width)*pixels).astype(int); iy = np.floor((yy[selected]+width)/(2*width)*pixels).astype(int)
    counts = np.zeros((pixels, pixels), np.int64)
    resolved = species[selected] != 2
    np.add.at(counts, (iy[resolved], ix[resolved]), 1)
    cell_mass = np.zeros((pixels, pixels)); cell_coarse = np.zeros((pixels, pixels))
    np.add.at(cell_mass, (iy, ix), mass[selected])
    np.add.at(cell_coarse, (iy[~resolved], ix[~resolved]), mass[selected][~resolved])
    total = float(mass[selected].sum()); coarse = float(mass[selected & (species == 2)].sum())
    if not math.isfinite(total) or not np.isfinite(cell_mass).all() or not np.isfinite(cell_coarse).all():
        raise FloatingPointError('Native lens census mass exceeds numeric range')
    fraction = coarse/total if total else None
    supported = counts >= settings['minimum_particles_per_pixel']
    pixel_fraction = np.divide(cell_coarse, cell_mass, out=np.zeros_like(cell_mass), where=cell_mass > 0)
    supported &= pixel_fraction <= settings['maximum_coarse_mass_fraction']
    if fraction is None or fraction > settings['maximum_coarse_mass_fraction']:supported[:] = False
    return counts, supported, {'selected_particle_count': int(selected.sum()), 'selected_mass_msun': total,
        'coarse_mass_msun': coarse, 'coarse_mass_fraction': fraction,
        'maximum_nearest_pixel_coarse_mass_fraction': float(pixel_fraction.max()),
        'count_definition': 'nearest projected noncoarse pixel census within complete cylinder; mass maps retain original CIC deposition; both cylinder and nearestpixel coarse mass fractions screened',
        'support_scope': 'Declared native sampling/contamination screen only; boundary, force, mass-resolution and lens-inference convergence unaccepted'}


def native_recovered_lensing_runner(parameters, context):
    from .sidm_analysis import _load_product, publish_sidm_derived, publish_sidm_observations
    from .analysis import load_cfg
    from .hubble_observers import ExpansionHistory
    from .data import export_numeric_audit
    from .sidm_products import export_native_sidm_extended
    memory = parameters['memory_gib']
    if memory > context.resources['memory_gib']:raise MemoryError('Native lens stage exceeds allocation')
    settings, targets, pair, histories, pins = lensing_inputs(parameters); out = context.generation
    observations = []; comparisons = []; native_runs = []; epoch_records = []
    result = {'schema': 'native-recovered-sidm-lensing-result-v1', 'execution_status': 'COMPLETE', 'science_acceptance': 'NOT_ACCEPTED',
        'source_hashes': pins, 'h0_status': 'NOT_ESTIMATED',
        'scope': 'Original thin-lens species mass response at fixed common CLASS background; not an H0 posterior. Observational time-delay inference is a separate branch.'}
    if pair['status'] == 'MEASURED_NATIVE_MATCH':
        backgrounds = {}; configs = {}
        for label in ('cdm', 'sidm'):
            run = Path(parameters[label+'_run_directory']).resolve(); configs[label] = load_cfg(run)
            manifest = run/'manifest.json'; record = json.loads(manifest.read_text()); ref = record['input_artifacts']['boltzmann_background']
            background = Path(ref['path']).resolve()
            if background.stat().st_size*32 > memory*1024**3/8:raise MemoryError('CLASS lens-distance background exceeds allocated intake')
            if file_sha256(background) != ref['sha256'] or ref['sha256'] != histories[label]['matching']['background_sha256']:
                raise ValueError('Lens distances do not bind the exact actual evolved CLASS background')
            _merge(pins, {str(background): ref['sha256'], str(manifest): file_sha256(manifest), str(run/'config.yaml'): file_sha256(run/'config.yaml')})
            if configs[label]['cosmology'].get('Omega_k', 0.) != 0:raise ValueError('This native lens recipe requires its explicitly flat common background')
            backgrounds[label] = background
            receipt = histories[label]['native_stage_receipt']
            native_runs.append({'run_directory': str(run), 'stage_receipt': {'path': receipt, 'sha256': file_sha256(receipt)}})
        if file_sha256(backgrounds['cdm']) != file_sha256(backgrounds['sidm']):raise ValueError('Paired native lenses change background expansion')
        expansion = ExpansionHistory.from_class(backgrounds['cdm'], omega_k=0.)
        by_epoch = {label: {r['snapshot']: r for r in targets[label]['memberships']} for label in ('cdm', 'sidm')}
        if set(by_epoch['cdm']) != set(by_epoch['sidm']):raise ValueError('Native paired lenses require the same exact saved epoch census')
        for snapshot in sorted(by_epoch['cdm']):
            loaded = {}
            for label in ('cdm', 'sidm'):
                row = by_epoch[label][snapshot]; member, arrays, sources = _load_product(row['product'], out, memory*1024**3/3)
                if (member.get('schema') != 'native-halo-membership-measurement-v1' or member['request']['host_row'] != row['host_row']
                        or member['request']['snapshot'] != snapshot or member['request']['species'] != [0, 1, 2, 4, 5]
                        or member['matching'] != histories[label]['matching']):
                    raise ValueError('Lens population changes the actual recovered host/epoch/species definition')
                _merge(pins, sources); loaded[label] = (row, member, arrays)
            epochs = [loaded[k][1]['epoch'] for k in ('cdm', 'sidm')]
            if any(e['canonical_time'] != 'scale_factor' for e in epochs) or epochs[0]['actual_scale_factor'] != epochs[1]['actual_scale_factor']:
                raise ValueError('Native lens branches change actual saved physical epochs')
            a = epochs[0]['actual_scale_factor']; z = 1/a-1
            if not 0 < z < settings['source_redshift']:
                observations.append({'snapshot': snapshot, 'status': 'OUTSIDE_OBSERVER_LENS_SOURCE_ORDER', 'actual_lens_redshift': z}); continue
            distances = {'angular_diameter_distance_kpc': float(expansion.angular_distance(0., z))*1000,
                'source_angular_diameter_distance_kpc': float(expansion.angular_distance(0., settings['source_redshift']))*1000,
                'lens_source_angular_diameter_distance_kpc': float(expansion.angular_distance(z, settings['source_redshift']))*1000}
            matching = {'native_target_matching': histories['cdm']['matching'], 'lens_controls_sha256': parameters['template_sha256'],
                'actual_scale_factor': a, 'source_redshift': settings['source_redshift'], 'lens_distances_proper_kpc': distances,
                'center_policy': 'each original recovered FOF proper center; no observed image selection or lens fitting',
                'physical_selection': 'same proper cylinder, native species, projection basis and source redshift; different evolved mass is measured'}
            refs = {}
            for label in ('cdm', 'sidm'):
                row, member, arrays = loaded[label]; counts, supported, census = plane_census(member, arrays, settings)
                physical = {k: copy.deepcopy(settings[k]) for k in ('pixels', 'half_width_proper_kpc', 'half_depth_proper_kpc',
                    'line_of_sight', 'horizontal_direction', 'background_surface_density_msun_kpc2')}
                physical.update(**distances, center_proper_kpc=arrays['center_proper_kpc'].tolist(), matched_experiment=matching,
                    sidm_baryonic_control=histories[label]['matching']['baryon_control_sha256'])
                request = {'schema': 'sidm-derived-science-request-v1', 'product': 'lens_plane', 'memory_gib': memory,
                    'inputs': [row['product']], 'settings': physical, 'matched_experiment': matching}
                path = out/f'lens-{label}-{snapshot:03d}.json'; publish_sidm_derived({'request': request, 'request_base': str(out)}, path)
                report, values, sources = _load_product({'path': str(path), 'sha256': file_sha256(path)}, out, memory*1024**3/3)
                surface = np.asarray(values['surface_density_msun_kpc2'])
                coarse_key = 'species_2_surface_density_msun_kpc2'
                if coarse_key not in values and census['coarse_mass_msun'] > 0:
                    raise ValueError('Native lens producer omitted a measured coarse-species mass map')
                # No deposited type-2 population has exactly zero coarse mass.
                coarse = np.asarray(values[coarse_key]) if coarse_key in values else np.zeros_like(surface)
                if (surface.shape != counts.shape or coarse.shape != surface.shape
                        or not np.isfinite(surface).all() or not np.isfinite(coarse).all()
                        or np.any(surface < 0) or np.any(coarse < 0)):
                    raise ValueError('Original species lens mass maps must be finite, nonnegative and aligned')
                # A coarse particle contributes to several CIC cells. Nearest
                # pixel counts alone cannot certify contamination of that map.
                cic_fraction = np.divide(coarse, surface, out=np.zeros_like(surface), where=surface > 0)
                supported &= (surface > 0) & (cic_fraction <= settings['maximum_coarse_mass_fraction'])
                census = {**census, 'maximum_cic_coarse_mass_fraction': float(cic_fraction.max()),
                    'support_scope': census['support_scope']+'; also screened against actual CIC-deposited species-2/total surface density'}
                values = {**values, 'native_sampling_supported': supported, 'nearest_pixel_particle_count': counts,
                          'cic_coarse_mass_fraction': cic_fraction}
                specs = {k: {'units': report['numeric_data']['array_contract']['datasets'][k]['units'], 'quantity': k} for k in values if k in report['numeric_data']['array_contract']['datasets']}
                specs.update(native_sampling_supported={'units': 'dimensionless', 'quantity': census['support_scope']},
                    nearest_pixel_particle_count={'units': 'dimensionless', 'quantity': census['count_definition']},
                    cic_coarse_mass_fraction={'units': 'dimensionless', 'quantity': 'Original CIC-deposited coarse species-2 mass / total native mass per pixel'})
                adapted = out/f'lens-supported-{label}-{snapshot:03d}.json'
                _merge(sources,pins)
                export_numeric_audit(adapted, {'schema': 'native-sidm-supported-lens-v1', 'epoch': member['epoch'], 'matched_experiment': matching,
                    'original_plane': {'path': str(path), 'sha256': file_sha256(path)}, 'census': census, 'science_acceptance': 'NOT_ACCEPTED'}, values, specs,
                    algorithm=__file__, context={'epoch': member['epoch']}, source_hashes=sources)
                refs[label] = {'path': str(adapted), 'sha256': file_sha256(adapted)}
                observations.append({'snapshot': snapshot, 'branch': label, 'status': 'MEASURED_NATIVE_LENS_PLANE', 'census': census, 'product': refs[label]})
            comparisons.append({'comparison_id': 'native-lens-snapshot-'+str(snapshot), 'family': 'lensing', 'collisionless': refs['cdm'], 'sidm': refs['sidm'],
                'matching': matching, 'observables': [{**row, 'support_array': 'native_sampling_supported'} for row in settings['observables']]})
            epoch_records.append({'snapshot': snapshot, 'actual_scale_factor': a, 'lens_redshift': z, 'distances_proper_kpc': distances})
    if comparisons:
        path = out/'lensing-response.json'; request = {'schema': 'sidm-observational-comparison-request-v1', 'memory_gib': memory,
            'maximum_comparisons': len(comparisons), 'comparisons': comparisons}
        publish_sidm_observations({'request': request, 'request_base': str(out)}, path)
        export = {'schema': 'native-sidm-extended-figures-v1', 'operation': 'observations', 'product': {'path': str(path), 'sha256': file_sha256(path)},
            'producer_base_directory': str(out), 'native_runs': native_runs, 'phase_baseline_run_directory': parameters['cdm_run_directory']}
        figures = export_native_sidm_extended(export, out/'lensing-figures', memory_gib=memory)
        result.update(status='MEASURED_NATIVE_LENSING_CONTRAST', variants=sorted({f['variant'] for f in figures['figures']}), product=export['product'])
    else:
        result.update(status='UNRESOLVED_NATIVE_LENSING_PAIR_OR_GEOMETRY', variants=[])
    result.update(observations=observations, lens_epochs=epoch_records)
    for p, h in pins.items():
        if file_sha256(p) != h:raise ValueError('Native lens input source changed during execution')
    atomic_json(out/'sidm-lensing-result.json', result)
    return {'execution_status': 'COMPLETE', 'measurements': {'sidm_lensing_h0_response': result}}


# Fixed-track, common-birth-state stellar response

def stream_controls(path,digest,memory):
    from .sidm_physics import _positive
    _positive(memory, "native stellar response workspace")
    path=Path(path).resolve()
    if file_sha256(path)!=digest:raise ValueError('Original native SIDM stream controls changed')
    if path.stat().st_size*24>memory*1024**3/8:raise MemoryError('Native SIDM stream controls exceed allocation')
    settings=json.loads(path.read_text())
    keys={'schema','maximum_candidates','maximum_candidate_pairs','maximum_particles','matching_controls','bin_fractions','measurement_controls','observables'}
    if set(settings)!=keys or settings['schema']!='native-recovered-sidm-stream-controls-v1':raise ValueError('Complete native SIDM fixed-track controls required')
    for key in ('maximum_candidates','maximum_candidate_pairs','maximum_particles'):
        if type(settings[key]) is not int or settings[key]<1:raise ValueError('Exact positive native stream census required')
    rules=settings['matching_controls']
    if (set(rules)!={'minimum_shared_particles','minimum_material_fraction','maximum_runner_up_fraction'}
            or type(rules['minimum_shared_particles']) is not int or rules['minimum_shared_particles']<1):raise ValueError('Original exact shared-particle stream matching rules required')
    for key in ('minimum_material_fraction','maximum_runner_up_fraction'):
        if type(rules[key]) not in (int,float) or not math.isfinite(rules[key]):raise ValueError('Finite original native stream matching fractions required')
    if not 0<=rules['maximum_runner_up_fraction']<rules['minimum_material_fraction']<=1:raise ValueError('Native stream retention/runner-up fractions inconsistent')
    f=np.asarray(settings['bin_fractions'],float)
    if f.ndim!=1 or len(f)<3 or not np.isfinite(f).all() or not 0<f[0]<f[-1]<1 or np.any(np.diff(f)<=0):raise ValueError('Ordered fixed-bin fractions strictly inside reference-track support required')
    measurement=settings['measurement_controls']
    if set(measurement)!={'projection_chunk_rows','minimum_bin_particles','label_resolution_kpc'}:raise ValueError('Original fixed-track estimator controls required')
    for name,minimum in [('projection_chunk_rows',1),('minimum_bin_particles',2)]:
        if type(measurement[name]) is not int or measurement[name]<minimum:raise ValueError('Exact positive fixed-track measurement counts required')
    if type(measurement['label_resolution_kpc']) not in (int,float) or not math.isfinite(measurement['label_resolution_kpc']) or measurement['label_resolution_kpc']<=0:raise ValueError('Positive finite original Lagrangian label resolution required')
    units={'linear_density_msun_kpc':'Msun / proper kpc','centroid_kpc':'proper kpc','width_covariance_kpc2':'proper kpc^2','velocity_mean_kms':'km / s'}
    rows=settings['observables']
    if not isinstance(rows,list) or not rows or len({r['array'] for r in rows})!=len(rows):raise ValueError('Distinct fixed-track observable selectors required')
    for row in rows:
        if set(row)!={'array','units','absolute_floor'} or units.get(row['array'])!=row['units'] or type(row['absolute_floor']) not in (int,float) or not math.isfinite(row['absolute_floor']) or row['absolute_floor']<=0:
            raise ValueError('Original fixed-track physical units and positive dimensional floors required')
    return settings


def stream_inputs(parameters):
    from .workflow import verify_completed_receipt
    from .sidm_products import sidm_matched_inputs
    memory=parameters['memory_gib']
    if type(memory) not in (int,float) or not math.isfinite(memory) or memory<=0:raise ValueError('Positive native SIDM stream allocation required')
    template=Path(parameters['template_path']).resolve();settings=stream_controls(template,parameters['template_sha256'],memory)
    pins={str(template):file_sha256(template)};pairpath=Path(parameters['pair_result']).resolve()
    if pairpath.stat().st_size*24>memory*1024**3/8:raise MemoryError('Native SIDM pair metadata exceeds allocation')
    pair=json.loads(pairpath.read_text());_merge(pins,{str(pairpath):file_sha256(pairpath)});_merge(pins,pair['source_hashes'])
    if pair.get('schema')!='native-recovered-sidm-pair-result-v1' or pair.get('execution_status')!='COMPLETE':raise ValueError('Original completed native SIDM matched target required')
    histories={};measurements={};owners={}
    if pair['status']=='MEASURED_NATIVE_MATCH':
        ref=pair['matched_history'];p=Path(ref['path']).resolve()
        if not p.is_relative_to(pairpath.parent) or file_sha256(p)!=ref['sha256']:raise ValueError('Original native matched stream target changed')
        _,_,sources,_,histories=sidm_matched_inputs({'schema':'native-sidm-matched-figure-request-v1','matched_product':ref,'history_base_directory':str(pairpath.parent),
            'native_runs':{label:parameters[label+'_run_directory'] for label in ('cdm','sidm')}},memory_gib=memory);_merge(pins,sources)
    elif pair['status']!='UNRESOLVED_NATIVE_MATCH':raise ValueError('Unknown native SIDM matched target status')
    for label in ('cdm','sidm'):
        receipt=Path(parameters[label+'_stream_receipt']).resolve()
        if receipt.stat().st_size*24>memory*1024**3/8:raise MemoryError('Native stellar producer receipt exceeds allocation')
        owned=verify_completed_receipt(receipt)
        if owned['kind']!='native_recovered_stream_measurements':raise ValueError('Original recovered native stellar measurement stage required')
        root=Path(owned['generation']).resolve();owners[label]=root
        p=Path(parameters[label+'_stream_history']).resolve()
        if not p.is_relative_to(root) or owned['files_sha256'].get(str(p.relative_to(root)))!=file_sha256(p):raise ValueError('Native stream history is outside its original measurement generation')
        if p.stat().st_size*24>memory*1024**3/8:raise MemoryError('Native SIDM stellar history metadata exceeds allocation')
        record=json.loads(p.read_text())
        if record.get('schema')!='native-recovered-stream-measurement-history-v1':raise ValueError('Actual recovered native stellar measurement history required')
        snapshots=[row['snapshot'] for row in record['records']]
        if any(type(s) is not int or s<0 for s in snapshots) or len(set(snapshots))!=len(snapshots):raise ValueError('Distinct exact native stellar saved epochs required')
        _merge(pins,owned['source_hashes']);_merge(pins,owned['input_hashes']);_merge(pins,record['source_hashes'])
        _merge(pins,{str(receipt):file_sha256(receipt),str(p):file_sha256(p)})
        measurements[label]=record
    bridge=None
    if histories:
        from .foundation import product_provenance
        namespaces={};initials={}
        for label in ('cdm','sidm'):
            run=Path(parameters[label+'_run_directory']).resolve()
            provenance=product_provenance(run,algorithm=__file__,selection={'purpose':'verify original stellar namespace against actual common initial conditions'},
                source_files=[run/'config.yaml'],coordinates={'frame':'identity only; no coordinate conversion'})
            namespaces[label]=provenance['identity_namespaces']['native_particle'];initials[label]=[]
            _merge(pins,{str(run/'manifest.json'):file_sha256(run/'manifest.json'),str(run/'config.yaml'):file_sha256(run/'config.yaml')})
            for row in provenance['initial_conditions']:
                if row.get('status')!='VERIFIED':raise ValueError('Native stellar namespace bridge requires actual verified IC files')
                if file_sha256(row['path'])!=row['sha256']:raise ValueError('Native stellar shared IC bridge changed')
                _merge(pins,{row['path']:row['sha256']});initials[label].append(row['sha256'])
        if not initials['cdm'] or sorted(initials['cdm'])!=sorted(initials['sidm']):raise ValueError('Run-specific stellar namespaces do not share exact native IC bytes')
        bridge={'original_native_namespaces':namespaces,'verified_shared_ic_sha256':sorted(initials['cdm']),
            'comparison_namespace':'native-shared-ic-particles:'+canonical_digest(sorted(initials['cdm'])),
            'scope':'Explicit namespace bridge through identical actual IC byte sets; native stellar ID equality additionally requires equal recorded birth states'}
    for p,h in pins.items():
        if file_sha256(p)!=h:raise ValueError('Native SIDM stream input source changed')
    return settings,pair,histories,measurements,owners,bridge,pins


def mutual_cohort_matches(left,right,rules,maximum_pairs):
    """Select mutual best exact-ID overlaps with explicit runner-up margins.

    Overlap fractions use each full candidate's membership, not the chosen
    intersection. Candidate count/identity matching alone does not establish
    common stellar formation; matched_birth_weights checks that separately.
    """
    if len(left)*len(right)>maximum_pairs:raise MemoryError('Native stellar cohort candidate-pair ceiling exceeded')
    votes=np.zeros((len(left),len(right)),np.int64)
    for i,a in enumerate(left):
        for j,b in enumerate(right):
            if a.get('matching_namespace',a['namespace'])==b.get('matching_namespace',b['namespace']):votes[i,j]=len(np.intersect1d(a['ids'],b['ids'],assume_unique=True))
    result=[]
    for i,a in enumerate(left):
        if not len(right):break
        order=np.argsort(-votes[i],kind='stable');j=int(order[0]);shared=int(votes[i,j]);other=np.argsort(-votes[:,j],kind='stable')
        runner_left=int(votes[i,order[1]]) if len(order)>1 else 0;runner_right=int(votes[other[1],j]) if len(other)>1 else 0
        if (int(other[0])==i and shared>=rules['minimum_shared_particles'] and shared>max(runner_left,runner_right)
                and shared/len(a['ids'])>=rules['minimum_material_fraction'] and shared/len(right[j]['ids'])>=rules['minimum_material_fraction']
                and runner_left/len(a['ids'])<=rules['maximum_runner_up_fraction'] and runner_right/len(right[j]['ids'])<=rules['maximum_runner_up_fraction']):
            result.append((i,j,{'shared_particles':shared,'cdm_particles':len(a['ids']),'sidm_particles':len(right[j]['ids']),
                'cdm_runner_up_shared_particles':runner_left,'sidm_runner_up_shared_particles':runner_right}))
    return result


def _selected_wraps(record,snapshot,root,settings,pins,namespace,expected_host,native):
    from .stream.native_encounters import _cohort
    from .sidm_analysis import _load_product
    output=[]
    def load(ref):
        path=Path(ref['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Native stellar product escapes its original generation')
        report,arrays,sources=_load_product(ref,root,settings['memory_gib']*1024**3/8);_merge(pins,sources)
        return report,arrays
    if record.get('product') is None:return output
    path=Path(record['product']['path']).resolve()
    if not path.is_relative_to(root) or file_sha256(path)!=record['product']['sha256']:raise ValueError('Native per-epoch stream result changed or escapes generation')
    if path.stat().st_size*24>settings['memory_gib']*1024**3/8:raise MemoryError('Native stellar candidate metadata exceeds allocation')
    epoch=json.loads(path.read_text());_merge(pins,{str(path):file_sha256(path)});_merge(pins,epoch['source_hashes'])
    if epoch.get('schema')!='native-catalogue-stream-measurement-result-v1':raise ValueError('Original native stellar candidate measurements required')
    if len(epoch['candidates'])>settings['maximum_candidates']:raise MemoryError('Native stellar candidate ceiling exceeded')
    for row in epoch['candidates']:
        cohort=_cohort(row,settings['memory_gib'],pins,snapshot)
        if cohort is None:continue
        coordinates,arrays=load(row['products']['coordinates']);sidecar,selection=load(row['products']['selection_context'])
        if coordinates['namespace']!=namespace:raise ValueError('Native stellar response changes the actual shared particle namespace')
        context=coordinates.get('source_context')
        if context is None:raise ValueError('Native stellar response lacks original measured host chart')
        if context.get('host_fof_row')!=expected_host or context.get('snapshot')!=snapshot:raise ValueError('Native stellar selection changes the independently recovered host or saved epoch')
        for binding in context.get('source_sha256',[]):
            if native.get(str(Path(binding['path']).resolve()))!=binding['sha256']:raise ValueError('Native stellar chart belongs to a different evolution generation')
        if not context.get('source_sha256'):raise ValueError('Native stellar chart lacks original snapshot provenance')
        if sidecar.get('native_birth_context') is None:continue
        birth,birth_arrays=load(sidecar['native_birth_context'])
        if birth.get('schema')!='native-stream-birth-context-v1' or birth['namespace']!=namespace or birth['snapshot']!=snapshot:raise ValueError('Native stellar birth state changes namespace or saved epoch')
        from .stream.measurements import FIELDS
        contracts=birth['numeric_data']['array_contract']['datasets']
        for name in ('SSPParentGasID','SSPBirthStateVersion','SSPBirthMode','StellarRandomCounter','SSPInitialMass',
                'SSPInitialLockedMass','SSPPromptReturnedMass','SSPBirthScaleFactor','SSPBirthMetallicity'):
            if contracts[name]['units']!=FIELDS[name][0]:raise ValueError('Original native stellar birth units differ')
        if contracts['SSPParentGasID'].get('identity_namespace')!=namespace:raise ValueError('Original stellar parent-gas identity namespace differs')
        declarations=coordinates['numeric_data']['array_contract']['datasets']
        for wrap in coordinates['wraps']:
            if wrap['status']!='MEASURED':continue
            prefix='wrap_%04d_'%wrap['wrap_id'];ids=exact_uint64_ids(arrays[prefix+'particle_ids'],allow_empty=False)
            for name,allowed in [('position_kpc',{'proper kpc'}),('velocity_kms',{'km/s','km / s'}),('mass_msun',{'Msun'}),('curve_position_kpc',{'proper kpc'})]:
                if declarations[prefix+name]['units'] not in allowed:raise ValueError('Original native stellar coordinate units differ')
            if len(ids)>settings['maximum_particles']:raise MemoryError('Native matched stellar wrap exceeds allocation')
            if not np.isin(ids,cohort['ids']).all():raise ValueError('Native measured wrap changes original supported candidate membership')
            output.append({'namespace':namespace,'ids':np.sort(ids),'candidate_id':row['candidate_id'],'wrap_id':wrap['wrap_id'],
                'coordinates':coordinates,'arrays':arrays,'prefix':prefix,'birth':birth_arrays,'epoch':coordinates['epoch'],
                'source_coordinates':row['products']['coordinates'],'source_birth_context':sidecar['native_birth_context']})
    if len(output)>settings['maximum_candidates'] or sum(len(r['ids'])*1024 for r in output)>settings['memory_gib']*1024**3/3:
        raise MemoryError('Retained native stellar wraps exceed allocated census/workspace')
    return output


def matched_birth_weights(left,right,ids):
    """Return common SSP birth locked masses in Msun, or unresolved birth state.

    Exact ID equality is insufficient: parent gas IDs, formation counters,
    birth mode/epoch/metallicity and gross/locked/prompt-return masses must
    match. These frozen birth weights describe the common selected cohort;
    they do not measure current stellar mass loss or omitted populations.
    """
    ids=exact_uint64_ids(ids,allow_empty=False)
    selected=[]
    for row in (left,right):
        birth=row['birth'];available=exact_uint64_ids(birth['particle_ids']);order=np.argsort(available);at=np.searchsorted(available[order],ids)
        if np.any(at>=len(order)) or not np.array_equal(available[order][at],ids):raise ValueError('Native birth census misses selected exact stellar identities')
        rows=order[at];mask=np.asarray(birth['birth_context_available'])
        if mask.dtype!=np.dtype(bool) or mask.shape!=available.shape:raise ValueError('Exact native birth availability mask required')
        if not mask[rows].all():return None
        integer_fields=('SSPParentGasID','SSPBirthStateVersion','SSPBirthMode','StellarRandomCounter')
        values={}
        for key in integer_fields:
            array=exact_uint64_ids(birth[key],unique=False)
            if array.shape!=available.shape:raise ValueError('Native stellar birth identity/counter shape differs')
            values[key]=array[rows]
        if not np.all(values['SSPBirthStateVersion']==1):return None
        for key in ('SSPInitialMass','SSPInitialLockedMass','SSPPromptReturnedMass','SSPBirthScaleFactor','SSPBirthMetallicity'):
            array=np.asarray(birth[key])
            if array.shape!=available.shape or not np.isfinite(array[rows]).all() or np.any(array[rows]<0):raise ValueError('Invalid original native stellar birth quantity')
            values[key]=array[rows]
        selected.append(values)
    if any(not np.array_equal(selected[0][key],selected[1][key]) for key in selected[0]):return None
    weights=selected[0]['SSPInitialLockedMass']
    if not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('Positive original common stellar birth locked masses required')
    return weights


def native_recovered_stream_response_runner(parameters,context):
    from .sidm_analysis import publish_sidm_observations
    from .stream_forward_model import TrackCoordinates,measure_track_density_velocity
    from .data import export_numeric_audit
    from .sidm_products import export_native_sidm_extended
    from .sidm_analysis import _stage_native_bindings,_load_product
    memory=parameters['memory_gib']
    if memory>context.resources['memory_gib']:raise MemoryError('Native SIDM stellar response exceeds stage allocation')
    settings,pair,histories,measurements,owners,bridge,pins=stream_inputs(parameters);settings={**settings,'memory_gib':memory}
    records=[];comparisons=[];out=context.generation;native_runs=[]
    if pair['status']=='MEASURED_NATIVE_MATCH':
        epochs={label:{r['snapshot']:r for r in measurements[label]['records']} for label in ('cdm','sidm')}
        if set(epochs['cdm'])!=set(epochs['sidm']):raise ValueError('Paired native stellar responses change actual saved epoch census')
        native_sources={}
        for label in ('cdm','sidm'):
            receipt=histories[label]['native_stage_receipt']
            native_sources[label]=_stage_native_bindings({'request':{'memory_gib':memory},
                'native_run_directory':parameters[label+'_run_directory'],'native_stage_receipt':receipt})
            _merge(pins,native_sources[label])
            native_runs.append({'run_directory':parameters[label+'_run_directory'],
                'stage_receipt':{'path':receipt,'sha256':file_sha256(receipt)}})
        namespace=bridge['comparison_namespace']
        for snapshot in sorted(epochs['cdm']):
            populations={}
            for label in ('cdm','sidm'):
                history=histories[label];selected=[r for r in history['request']['epochs'] if r['snapshot']==snapshot]
                if not selected:
                    populations[label]=[];continue
                if len(selected)!=1:raise ValueError('Duplicate original recovered host epoch for native stellar response')
                member,_,sources=_load_product(selected[0]['selection'],out,memory*1024**3/8);_merge(pins,sources)
                native=native_sources[label]
                populations[label]=_selected_wraps(epochs[label][snapshot],snapshot,owners[label],settings,pins,bridge['original_native_namespaces'][label],member['request']['host_row'],native)
                for population in populations[label]:population['matching_namespace']=namespace
            matched=mutual_cohort_matches(populations['cdm'],populations['sidm'],settings['matching_controls'],settings['maximum_candidate_pairs'])
            if not matched:records.append({'snapshot':snapshot,'status':'UNRESOLVED_NATIVE_SUPPORTED_COHORT_MATCH'});continue
            for i,j,overlap in matched:
                left,right=populations['cdm'][i],populations['sidm'][j];ids=np.intersect1d(left['ids'],right['ids'],assume_unique=True)
                weights=matched_birth_weights(left,right,ids)
                if weights is None:records.append({'snapshot':snapshot,'status':'UNRESOLVED_NATIVE_COMMON_BIRTH_STATE','overlap':overlap});continue
                if left['epoch']['actual_scale_factor']!=right['epoch']['actual_scale_factor']:raise ValueError('Native stellar response changes actual saved physical epoch')
                prefix=left['prefix'];arrays=left['arrays'];support=np.asarray(arrays[prefix+'curve_support']);vertices=np.asarray(arrays[prefix+'curve_position_kpc'])
                if support.dtype!=np.dtype(bool) or support.shape!=(len(vertices),):raise ValueError('Original native CDM track support shape differs')
                chosen=np.flatnonzero(support)
                if len(chosen)<2 or np.any(np.diff(chosen)!=1) or np.any(np.linalg.norm(np.diff(vertices[chosen],axis=0),axis=1)<=0):
                    records.append({'snapshot':snapshot,'status':'UNRESOLVED_NATIVE_REFERENCE_TRACK_SUPPORT','overlap':overlap});continue
                track=TrackCoordinates({'position_kpc':vertices[chosen],'origin':'original native host-centred proper Cartesian chart'})
                controls={**settings['measurement_controls'],'bin_edges_kpc':(np.asarray(settings['bin_fractions'])*track.arc[-1]).tolist()}
                if settings['measurement_controls']['projection_chunk_rows']*len(chosen)*128+len(ids)*4096>memory*1024**3/2:raise MemoryError('Native fixed-track stellar response workspace exceeds allocation')
                matching={'native_target_matching':histories['cdm']['matching'],'stream_controls_sha256':parameters['template_sha256'],
                    'stellar_namespace_bridge':bridge,
                    'actual_scale_factor':left['epoch']['actual_scale_factor'],'common_exact_stellar_ids_sha256':canonical_digest([int(k) for k in ids]),
                    'fixed_reference_track':left['source_coordinates'],'fixed_track_wrap_id':left['wrap_id'],'physical_bin_edges_kpc':controls['bin_edges_kpc'],
                    'selection':'Both independently supported native stream selections; mutual exactID-overlap rules and equal recorded native birth states; common intersection only',
                    'weight_definition':'Frozen common recorded SSPInitialLockedMass; current bound loss and population changes outside the intersection unmeasured',
                    'velocity_definition':'Source native host-relative peculiar velocity in fixed proper Cartesian basis; no trajectory derivative/action inference'}
                refs={};ordinal=len(comparisons)
                for label,row in (('cdm',left),('sidm',right)):
                    a=row['arrays'];p=row['prefix'];actual=exact_uint64_ids(a[p+'particle_ids']);order=np.argsort(actual);at=np.searchsorted(actual[order],ids);take=order[at]
                    values={'particle_ids':ids,'position_kpc':np.asarray(a[p+'position_kpc'])[take][None],
                        'velocity_kms':np.asarray(a[p+'velocity_kms'])[take][None],'mass_msun':weights[None]}
                    measured=measure_track_density_velocity(values,track,controls,particle_weights=weights)
                    output={'common_particle_ids':ids,'common_birth_locked_mass_msun':weights,'bin_edges_kpc':measured['bin_edges_kpc'],'bin_center_kpc':measured['bin_center_kpc']}
                    specs={'common_particle_ids':{'units':'identity','identity_namespace':namespace,'quantity':'Exact common birth-matched native stellar cohort'},
                        'common_birth_locked_mass_msun':{'units':'Msun','quantity':'Frozen original common SSP birth locked masses'},
                        'bin_edges_kpc':{'units':'proper kpc','quantity':'Common fixed reference track physical bin edges'},'bin_center_kpc':{'units':'proper kpc','quantity':'Common fixed reference track physical bin centres'}}
                    for observable in settings['observables']:
                        key=observable['array'];array=measured[key][0];mask=measured['measurement_supported'][0]
                        shape=mask.shape+(1,)*(array.ndim-1);mask=np.broadcast_to(mask.reshape(shape),array.shape).copy()
                        output[key]=np.where(mask,array,np.nan);output[key+'_supported']=mask
                        specs[key]={'units':observable['units'],'quantity':'Original native fixed-track '+key};specs[key+'_supported']={'units':'dimensionless','quantity':'Original fixed-track particle-count support'}
                    path=out/f'stream-{ordinal:03d}-{label}.json'
                    export_numeric_audit(path,{'schema':'native-sidm-fixed-track-stream-v1','epoch':row['epoch'],'matched_experiment':matching,
                        'overlap':overlap,'source_coordinates':row['source_coordinates'],'source_birth_context':row['source_birth_context'],
                        'science_acceptance':'NOT_ACCEPTED','scope':'Common postselected stellar cohort at one actual saved epoch, frozen CDM track/birth masses; no causal gap or full-population SIDM attribution'},
                        output,specs,algorithm=__file__,context={'epoch':row['epoch']},source_hashes=pins)
                    refs[label]={'path':str(path),'sha256':file_sha256(path)}
                comparisons.append({'comparison_id':'native-stellar-response-'+str(ordinal),'family':'streams','collisionless':refs['cdm'],'sidm':refs['sidm'],
                    'matching':matching,'observables':[{**r,'support_array':r['array']+'_supported'} for r in settings['observables']]})
                records.append({'snapshot':snapshot,'status':'MEASURED_NATIVE_COMMON_STELLAR_PROFILE','overlap':overlap,'products':refs})
    result={'schema':'native-recovered-sidm-stream-response-result-v1','execution_status':'COMPLETE','science_acceptance':'NOT_ACCEPTED','source_hashes':pins,
        'records':records,'scope':'Supported native, exact birth-matched common stellar cohort; fixed CDM track and initial locked masses. Not a full-population or uniquely causal SIDM response.'}
    if comparisons:
        path=out/'stream-response.json';request={'schema':'sidm-observational-comparison-request-v1','memory_gib':memory,'maximum_comparisons':len(comparisons),'comparisons':comparisons}
        publish_sidm_observations({'request':request,'request_base':str(out)},path)
        export={'schema':'native-sidm-extended-figures-v1','operation':'observations','product':{'path':str(path),'sha256':file_sha256(path)},
            'producer_base_directory':str(out),'native_runs':native_runs,'phase_baseline_run_directory':parameters['cdm_run_directory']}
        figures=export_native_sidm_extended(export,out/'stream-figures',memory_gib=memory)
        result.update(status='MEASURED_NATIVE_COMMON_STELLAR_CONTRAST',variants=sorted({r['variant'] for r in figures['figures']}),product=export['product'])
    else:result.update(status='UNRESOLVED_NATIVE_STELLAR_PROFILE_PAIR',variants=[])
    for p,h in pins.items():
        if file_sha256(p)!=h:raise ValueError('Native stellar response source changed during execution')
    atomic_json(out/'sidm-stream-result.json',result)
    return {'execution_status':'COMPLETE','measurements':{'sidm_stream_profile_response':result}}
