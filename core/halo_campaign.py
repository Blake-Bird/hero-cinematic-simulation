"""Prepare halo product campaigns and measure exact recovered descendants.

Recipes connect existing abundance, structure and assembly producers. Recovery
uses the recorded target identity and actual saved epoch; unresolved descendants
remain unavailable instead of selecting a fallback host. Scientific estimators
and figure adaptation stay in halos.py, assembly.py and halo_products.py.
"""
from pathlib import Path
import copy
import json
import math
import zipfile
import numpy as np
from .data import numeric_audit_source_bindings, exact_uint64_ids, verify_dataset_semantics
from .stream_common import read_product, positive
from .foundation import atomic_json, file_sha256
from .foundation import merge_source_bindings as _merge


GROUPS = {
    'analysis_bundles': ('analysis-figures', ['differential_hmf', 'cumulative_hmf']),
    'stellar_abundance_requests': ('abundance', ['stellar_mass_function', 'vmax_function']),
    'host_abundance_requests': ('host-abundance', ['reliability_limits', 'rare_high_z_abundance']),
    'matched_abundance_requests': ('matched-host-abundance', ['hmf_ratio']),
    'structure_products': ('structure-figures', ['density_profile', 'circular_velocity', 'velocity_dispersion',
        'anisotropy', 'concentration_mass', 'radial_axis_ratios', 'triaxiality', 'axis_orientation_twist', 'density_potential_shape']),
    'environment_products': ('splashback-figures', ['splashback_slope_radius_depth_width']),
}



ASSEMBLY = ['mass_scale_factor', 'log_mass_accretion', 'fast_slow_transition', 'spin_history',
    'concentration_history', 'shape_history', 'pericenter_infall_merger_events', 'stellar_assembly', 'quenching_morphology']



def prepare_native_halo_campaign(request, destination):
    from .workflow import compile_requested_product_dag, default_registry
    from .flagship_products import flagship_product_input_bindings
    required = {'schema', 'memory_gib', 'disk_gib', 'assembly_campaign', *GROUPS}
    if set(request) != required or request['schema'] != 'native-halo-campaign-v1':
        raise ValueError('Complete native26-variant halo campaign request required')
    for key in ('memory_gib', 'disk_gib'):
        if type(request[key]) not in (int, float) or not math.isfinite(request[key]) or request[key] <= 0:
            raise ValueError('Positive finite native halo '+key+' allocation required')
    out = Path(destination).resolve()
    if out.exists():
        raise FileExistsError('Keep previous native halo campaign')
    memory = request['memory_gib']; pins = {}; stages = []; routes = {}
    def bind(binding, *, directory=False):
        if set(binding) != {'path', 'sha256'}:
            raise ValueError('Exact native halo source path/hash required')
        path = Path(binding['path']).resolve()
        source = path/'analysis_provenance.json' if directory else path
        if source.stat().st_size*24 > memory*1024**3/8:
            raise MemoryError('Native halo recipe metadata exceeds allocation')
        if file_sha256(source) != binding['sha256']:
            raise ValueError('Native halo campaign source changed')
        _merge(pins, {str(source): binding['sha256']})
        return path
    assembly = bind(request['assembly_campaign']); manifest = json.loads(assembly.read_text())
    if (manifest.get('schema') != 'native-assembly-campaign-prepared-v1'
            or manifest.get('status') != 'RECIPES_PREPARED' or manifest.get('orbital_recipe_configured') is not True):
        raise ValueError('Full assembly/orbital producer preparation is required for all26variants')
    for name, checksum in manifest['source_hashes'].items():
        if file_sha256(name) != checksum:
            raise ValueError('Prepared native assembly source changed')
    _merge(pins, manifest['source_hashes'])
    workflow = bind(manifest['workflow']); inherited = json.loads(workflow.read_text())
    if inherited['resources']['memory_gib'] > memory or inherited['resources']['disk_gib'] > request['disk_gib']:
        raise MemoryError('Native halo campaign cannot reduce prepared assembly allocation')
    stages.extend(copy.deepcopy(inherited['stages']))
    requested = list(inherited['requested_products'])
    routes.update({variant: ['assembly-event-figures'] if variant == 'pericenter_infall_merger_events' else ['assembly-history'] for variant in ASSEMBLY})
    for group, (operation, variants) in GROUPS.items():
        bindings = request[group]
        if not isinstance(bindings, list) or not bindings:
            raise ValueError('Native halo campaign needs explicit '+group)
        for i, binding in enumerate(bindings):
            product = bind(binding, directory=operation == 'analysis-figures')
            name = group.replace('_', '-')+'-'+str(i)
            parameters = {'operation': operation, 'product': str(product),
                          'destination': '{generation}/products', 'memory_gib': memory}
            _merge(pins, flagship_product_input_bindings(parameters))
            stages.append({'stage_id': name, 'kind': 'flagship_product', 'parameters': parameters,
                           'outputs': {name: 'products'}})
            requested.append(name)
            for variant in variants:
                routes.setdefault(variant, []).append(name)
    if len(routes) != 26:
        raise ValueError('Native halo recipe must cover exactly26 unique variant routes')
    # Retained input files and native producer-stage references both participate
    # in workflow fingerprints. Inputs are verified again before cache reuse.
    for stage in stages:
        # Inherited stages can bind inputs not listed in the campaign manifest
        # (for example, their authored answer files). Keep those declarations;
        # adding export inputs must not erase the producer's dependencies.
        existing=stage.get('input_files',[])
        combined={}
        for binding in existing:
            _merge(combined,{binding['path']:binding['sha256']})
        _merge(combined,pins)
        existing_paths={binding['path'] for binding in existing}
        stage['input_files']=[*existing,*[
            {'path':name,'sha256':checksum} for name,checksum in sorted(combined.items())
            if name not in existing_paths]]
    spec = copy.deepcopy(inherited)
    spec.update(stages=stages, requested_products=requested)
    spec['resources'].update(memory_gib=memory, disk_gib=request['disk_gib'], maximum_stage_count=len(stages))
    registry = default_registry(); plan = compile_requested_product_dag(spec, registry, request_base=out)
    executable = {stage['stage_id'] for stage in plan['stages']}
    if any(stage not in executable for names in routes.values() for stage in names):
        raise ValueError('Every native halo variant route needs a compiled producer stage')
    code = {str(Path(__file__).resolve()): file_sha256(__file__)}
    for stage in plan['stages']:
        _merge(code, registry.source_hashes(stage['kind']))
    for name, checksum in {**pins, **code}.items():
        if file_sha256(name) != checksum:
            raise ValueError('Native halo preparation source changed during compilation')
    out.mkdir(parents=True)
    atomic_json(out/'request.json', request); atomic_json(out/'workflow.json', spec)
    result = {'schema': 'native-halo-campaign-prepared-v1', 'status': 'RECIPES_PREPARED',
        'workflow': {'path': str(out/'workflow.json'), 'sha256': file_sha256(out/'workflow.json')},
        'variant_routes': routes, 'stage_count': len(plan['stages']), 'source_hashes': {**pins, **code},
        'science_acceptance': 'NOT_ACCEPTED',
        'scope': '26implemented producer-to-export routes. Actual native support/quality, populations, fitted-field validation and convergence remain explicit at execution.'}
    atomic_json(out/'manifest.json', result)
    return result



TEMPLATE_FIELDS={'schema','components','support_radius_proper_kpc','radial_edges_proper_kpc',
    'bulk_velocity_peculiar_km_s','shape_controls','expansion_sequence',
    'calibration_positions_proper_kpc','validation_positions_proper_kpc','field_error_budget',
    'equipotential_controls','concentration','memory_gib'}



def validate_structure_template(template,*,memory_gib):
    if not isinstance(template,dict) or set(template)!=TEMPLATE_FIELDS or template['schema']!='native-halo-structure-v1':
        raise ValueError('Complete native structure template without run/snapshot/host selectors required')
    value=template['memory_gib']
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=memory_gib:
        raise MemoryError('Recovered structure template exceeds its workflow memory allocation')
    # Physical estimator controls retain their existing native validation at
    # execution. This adapter changes only the three runtime selectors.
    return template



def validate_recovered_structure(parameters):
    if type(parameters['snapshot']) is not int or parameters['snapshot']<0:
        raise ValueError('Exact saved recovered-structure snapshot identity required')
    if type(parameters['export_figures']) is not bool:
        raise ValueError('Explicit recovered structure figure choice required')
    value=parameters['memory_gib']
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError('Explicit positive recovered structure memory required')
    digest=parameters['template_sha256']
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Exact structure template checksum required')



def recovered_structure_runner(parameters,context):
    from .zoom import read_host_catalogue
    from .halos import publish_halo_structure
    validate_recovered_structure(parameters)
    memory=parameters['memory_gib']
    if memory>context.resources['memory_gib']:raise MemoryError('Recovered structure exceeds workflow memory')
    template_path=Path(parameters['template_path']).resolve();recovery_path=Path(parameters['recovery_path']).resolve()
    limit=min(16*1024**2,int(memory*1024**3/16))
    if max(template_path.stat().st_size,recovery_path.stat().st_size)>limit:
        raise MemoryError('Recovered structure metadata exceeds bounded intake')
    if file_sha256(template_path)!=parameters['template_sha256']:
        raise ValueError('Recovered structure template changed')
    template=validate_structure_template(json.loads(template_path.read_text()),memory_gib=memory)
    recovery=json.loads(recovery_path.read_text())
    if recovery.get('schema')!='lagrangian-descendant-recovery-v1':raise ValueError('Exact material descendant recovery required')
    pins={str(template_path):parameters['template_sha256'],str(recovery_path):file_sha256(recovery_path)}
    for source,digest in recovery['source_hashes'].items():
        if source in pins and pins[source]!=digest:raise ValueError('Conflicting recovered structure source bindings')
        pins[source]=digest
    for source,digest in pins.items():
        if file_sha256(source)!=digest:raise ValueError('Recovered structure source changed: '+source)
    matches=[row for row in recovery['records'] if row['snapshot']==parameters['snapshot']]
    if len(matches)!=1:raise ValueError('One exact recovery record required for selected structure epoch')
    match=matches[0];out=context.generation/'measurement'
    if match.get('status') not in {'MATCHED','DISRUPTED_OR_UNBOUND','AMBIGUOUS_OR_WEAK_MATCH'}:
        raise ValueError('Unknown native material recovery status')
    scale=match.get('scale_factor')
    if type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0:
        raise ValueError('Exact positive recovered scale factor required')
    if any(recovery['source_hashes'].get(source)!=digest for source,digest in match.get('source_hashes',{}).items()):
        raise ValueError('Recovery epoch sources lack their complete history binding')
    if out.exists():raise FileExistsError('Preserve earlier recovered structure measurements')
    report={'schema':'recovered-native-halo-structure-v1','snapshot':parameters['snapshot'],
        'recovery_status':match['status'],'host_row':None,'source_hashes':pins,
        'science_acceptance':'NOT_ACCEPTED','launch_authorized':False}
    if match['status']!='MATCHED':
        if match.get('descendant') is not None:raise ValueError('Unresolved recovery cannot supply a fallback host')
        out.mkdir();report.update(status='UNRESOLVED_TARGET',measurement=None,figures=None)
        atomic_json(out/'status.json',report);return report
    row=match['descendant']['host_row']
    if type(row) is not int or row<0:raise ValueError('Exact recovered descendant host row required')
    run=Path(parameters['run_dir']).resolve()
    cat=read_host_catalogue(run,parameters['snapshot'],memory_gib=memory)
    if row>=cat['group_count']:raise ValueError('Recovered structure host is outside the native catalogue')
    sources={entry['path']:entry['sha256'] for entry in cat['sources']+cat['snapshot_sources']}
    if not match['source_hashes'] or any(sources.get(source)!=digest for source,digest in match['source_hashes'].items()):
        raise ValueError('Structure run/catalogue differs from exact recovery sources')
    if not math.isclose(cat['frame'][0],match['scale_factor'],rel_tol=0,abs_tol=1e-12):
        raise ValueError('Structure epoch differs from recovered epoch')
    for source,digest in sources.items():
        if source in pins and pins[source]!=digest:raise ValueError('Conflicting recovered native source')
        pins[source]=digest
    request=copy.deepcopy(template)
    request.update(run_directory=str(run),snapshot=parameters['snapshot'],host_row=row)
    out.mkdir();atomic_json(out/'native-request.json',request)
    pins[str(out/'native-request.json')]=file_sha256(out/'native-request.json')
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    report.update(status='RUNNING',host_row=row,measurement=None,figures=None)
    atomic_json(out/'status.json',report)
    try:
        measured=publish_halo_structure(request,out/'structure.json',request_source_hashes=pins)
        gallery=None
        if parameters['export_figures']:
            from .halo_products import export_native_structure_figures
            gallery=export_native_structure_figures(out/'structure.json',out/'figures',memory_gib=memory)
        for source,digest in pins.items():
            if file_sha256(source)!=digest:raise ValueError('Recovered structure input changed during measurement')
        report.update(status='MEASURED',measurement={'path':str(out/'structure.json'),
            'sha256':file_sha256(out/'structure.json')},field_validation_status=measured['field_validation_status'],figures=gallery)
        atomic_json(out/'status.json',report);return report
    except BaseException as error:
        report.update(status='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED',
            failure={'type':type(error).__name__,'message':str(error)})
        atomic_json(out/'status.json',report);raise


# Attach measured native profiles and galaxy properties to exact ancestry.

def _binding(binding):
    path = Path(binding['path']).resolve()
    if file_sha256(path) != binding['sha256']:
        raise ValueError('Native assembly measurement binding changed')
    return path



def _key(key):
    if (not isinstance(key, list) or len(key) != 3 or not isinstance(key[0], str)
            or type(key[1]) is not int or not 0 <= key[1] < 2**64
            or type(key[2]) is not int or key[2] < 0):
        raise ValueError('Exact namespace/uint64/saved-epoch assembly key required')
    return tuple(key)



def validate_assembly_measurements(parameters):
    memory = parameters['memory_gib']
    if type(memory) not in (int, float) or not np.isfinite(memory) or memory <= 0:
        raise ValueError('Positive finite native assembly memory required')
    for name in ('maximum_nodes', 'maximum_edges', 'maximum_measurements'):
        if type(parameters[name]) is not int or parameters[name] < 1:
            raise ValueError('Positive exact native assembly '+name+' required')



def _galaxy_attributes(binding, run, key, epoch, memory_gib):
    """Resolved native central galaxy, with bound mass and matching bound SFR."""
    import h5py
    from .halos import verify_galaxy_catalogue
    path = _binding(binding); verified = verify_galaxy_catalogue(path, run)
    if verified['snapshot'] != key[2]:
        raise ValueError('Native assembly galaxy has a different saved epoch')
    pins = {str(path): file_sha256(path)}; result = {}
    with h5py.File(path) as f:
        contract = json.loads(f.attrs['data_contract_json'])
        for record in (contract['input_hashes']+contract['initial_conditions']+
                list(contract['boltzmann_inputs'].values())+contract['algorithm_sources']+
                [contract['executable'], contract['parameter_file'], contract['compile_options']]):
            if record.get('status') != 'UNAVAILABLE':
                _merge(pins, {record['path']: record['sha256']})
        names = ('HostFOFID', 'GalaxyID', 'CentralFlag', 'Resolved', 'StellarMass', 'SFR')
        if sum(f[n].size*f[n].dtype.itemsize for n in names)*16 > memory_gib*1024**3:
            raise MemoryError('Native central-galaxy columns exceed assembly allocation')
        expected = ('identity', 'identity', 'boolean selection mask', 'boolean selection mask', 'Msun/h', 'Msun/year')
        for name, unit in zip(names, expected):
            verify_dataset_semantics(f[name])
            if f[name].attrs['units'] != unit or json.loads(f[name].attrs['epoch_json']) != epoch:
                raise ValueError('Native assembly galaxy units or clock mismatch')
        if f['HostFOFID'].attrs['identity_namespace'] != key[0]:
            raise ValueError('Native galaxy and assembly host namespaces differ')
        ids = exact_uint64_ids(f['HostFOFID'][:], unique=False)
        central = f['CentralFlag'][:]; resolved = f['Resolved'][:]
        if central.dtype.kind != 'b' or resolved.dtype.kind != 'b' or central.shape != ids.shape or resolved.shape != ids.shape:
            raise ValueError('Native central/quality masks must be exact aligned booleans')
        selected = np.flatnonzero((ids == key[1]) & central)
        if len(selected) > 1:
            raise ValueError('Ambiguous native central galaxy for one FOF host')
        if len(selected) and resolved[selected[0]]:
            i = int(selected[0])
            from .analysis import load_cfg
            h = float(load_cfg(run)['cosmology']['H0'])/100.
            if not np.isfinite(h) or h <= 0:
                raise ValueError('Native galaxy physical mass requires positive recorded h')
            mass = float(f['StellarMass'][i])/h; sfr = float(f['SFR'][i])
            if np.isfinite(mass) and mass >= 0:
                result['stellar_mass_msun'] = mass
            if np.isfinite(sfr) and sfr >= 0:
                result['sfr_msun_year'] = sfr
            result['native_central_galaxy_key'] = [f['GalaxyID'].attrs['identity_namespace'], int(f['GalaxyID'][i]), key[2]]
    return result, pins



def structure_assembly_attributes(report, arrays, controls):
    """Adapt original profiles and support, without refitting or interpolation."""
    component = controls['component']; index = controls['profile_bin']
    if type(index) is not int or index < 0:
        raise ValueError('Exact native assembly profile bin required')
    prefix = 'profile_'+component+'_'
    radius = arrays[prefix+'radial_outer_edge_proper_kpc']
    if index >= len(radius):
        raise ValueError('Assembly bin outside original native structure grid')
    specs = report['numeric_data']['array_contract']['datasets']
    units = {'radial_outer_edge_proper_kpc': 'proper kpc', 'enclosed_mass_msun': 'Msun',
             'enclosed_angular_momentum_msun_kpc_km_s': 'Msun proper kpc km/s',
             'bullock_spin_enclosed': 'dimensionless'}
    for name, unit in units.items():
        if specs[prefix+name]['units'] != unit:
            raise ValueError('Original native structure units differ: '+name)
    mass = float(arrays[prefix+'enclosed_mass_msun'][index])
    angular = np.asarray(arrays[prefix+'enclosed_angular_momentum_msun_kpc_km_s'][index], float)
    spin = float(arrays[prefix+'bullock_spin_enclosed'][index])
    result = {}; definitions = {'spin': {
        'definition': 'native enclosed Bullock |J|/(sqrt(2) M Vc r); finite proper aperture, peculiar velocities',
        'component': component, 'radius_proper_kpc': float(radius[index]),
        'specific_angular_momentum': 'native enclosed J/M; proper kpc km/s'}}
    if np.isfinite(mass) and mass > 0 and angular.shape == (3,) and np.isfinite(angular).all() and np.isfinite(spin):
        result.update(spin_dimensionless=spin, angular_momentum_kpc_kms=(angular/mass).tolist())
    density = report.get('density_shapes', {}).get(component, {})
    shell_index = controls['density_shell']; potential_index = controls['potential_level']
    for value in (shell_index, potential_index):
        if type(value) is not int or value < 0:
            raise ValueError('Exact nonnegative original shape indices required')
    if density:
        if shell_index >= len(density['shells']):
            raise ValueError('Assembly density shell outside native shape product')
        row = density['shells'][shell_index]
        definitions['density_shape'] = {k: density[k] for k in ('window', 'estimator')}
        definitions['density_shape'].update(lower_radius_kpc=row['lower_radius_kpc'], upper_radius_kpc=row['upper_radius_kpc'])
        if row['status'] == 'MEASURED':
            result['density_axis_ratios'] = [row['b_over_a'], row['c_over_a']]
    potential = report.get('equipotential_shapes', {})
    if potential:
        if potential_index >= len(potential['levels']):
            raise ValueError('Assembly potential level outside native shape product')
        row = potential['levels'][potential_index]
        definitions['potential_shape'] = {'reference_radius_proper_kpc': row['reference_radius_proper_kpc'],
                                        'definition': potential['definition']}
        if row['status'] == 'MEASURED' and report['field_validation_status'] == 'PASS':
            result['potential_axis_ratios'] = [row['b_over_a'], row['c_over_a']]
    fit = report.get('concentration', {})
    if fit.get('status') == 'MEASURED':
        definitions['concentration'] = fit['definition']
        row = fit['components'].get(component, {})
        if row.get('status') == 'MEASURED':
            result['concentration'] = row['concentration']
    stellar = controls['stellar_component']
    if stellar is not None:
        from .analysis import _shape_tensor_components
        if report['request']['components'][stellar]['species'] != [4]:
            raise ValueError('Stellar morphology requires a native PartType4-only component')
        for name, unit in (('offsets_proper_kpc', 'proper kpc'), ('masses_msun', 'Msun')):
            if specs['parent_'+stellar+'_'+name]['units'] != unit:
                raise ValueError('Native stellar morphology units differ')
        x = arrays['parent_'+stellar+'_offsets_proper_kpc']; masses = arrays['parent_'+stellar+'_masses_msun']
        keep = np.linalg.norm(x, axis=1) <= radius[index]
        minimum = report['request']['shape_controls']['minimum_particles']
        measured = _shape_tensor_components(x[keep], weights=masses[keep]) if keep.sum() >= minimum else None
        if measured is not None:
            eigen, axes, _ = measured
            result['stellar_morphology_tensor_kpc2'] = ((axes*eigen)@axes.T).ravel().tolist()
        definitions['stellar_morphology'] = {'component': stellar, 'origin': 'original native FOF centre',
            'radius_proper_kpc': float(radius[index]), 'membership': report['request']['components'][stellar].get('membership', 'aperture_all'),
            'estimator': 'existing mass quadrupole about the recorded host origin; no recentering'}
    # No new object-quality pass is inferred from potential calibration or
    # plotting support. Existing catalogue quality remains authoritative.
    result['assembly_measurement_definitions'] = definitions
    return result



def assembly_measurement_inputs(parameters):
    from .halos import verify_identity_graph, verify_halo_structure
    validate_assembly_measurements(parameters); memory = parameters['memory_gib']
    root = Path(parameters['identity_graph']).resolve(); template = Path(parameters['template_path']).resolve()
    if max((root/'report.json').stat().st_size, template.stat().st_size)*24 > memory*1024**3/4:
        raise MemoryError('Native assembly metadata exceeds allocation')
    if file_sha256(template) != parameters['template_sha256']:
        raise ValueError('Native assembly template changed')
    with zipfile.ZipFile(root/'nodes.npz') as z:
        if sum(row.file_size for row in z.infolist())*8 > memory*1024**3/4:
            raise MemoryError('Native assembly graph arrays exceed allocation')
    graph = verify_identity_graph(root)
    if len(graph['nodes']) > parameters['maximum_nodes'] or len(graph['edges']) > parameters['maximum_edges']:
        raise MemoryError('Native assembly graph exceeds exact declared limits')
    design = json.loads(template.read_text())
    if (set(design) != {'schema', 'observations', 'controls'}
            or design.get('schema') != 'native-assembly-measurements-v1'
            or not isinstance(design['observations'], list) or not design['observations']
            or len(design['observations']) > parameters['maximum_measurements']
            or set(design['controls']) != {'component', 'profile_bin', 'density_shell', 'potential_level', 'stellar_component'}):
        raise ValueError('Bounded native assembly measurement template required')
    pins = {**graph['sources'], str(template): file_sha256(template),
            **{str(root/name): file_sha256(root/name) for name in ('report.json', 'report.sha256', 'nodes.npz', 'index.sqlite')}}
    nodes = {tuple(row['key']): row for row in graph['nodes']}; updates = {}; definitions = []
    for row in design['observations']:
        key = _key(row['key'])
        if key not in nodes or key in updates:
            raise ValueError('Missing or duplicate exact native assembly observation')
        path = _binding(row['structure'])
        report, arrays, sources = read_product(path, memory_gib=memory/4)
        verify_halo_structure(path); _merge(pins, sources); _merge(pins, numeric_audit_source_bindings(report))
        request = report['request']; run = Path(request['run_directory']).resolve()
        if (key != (str(run)+':fof:catalogue-row', request['host_row'], request['snapshot'])
                or nodes[key]['kind'] != 'halo'):
            raise ValueError('Structure must join its exact native FOF observation; no cross-finder row matching')
        epoch = report['source_context']['epoch']; actual = graph['epochs'][str(key[2])]
        clock = 'actual_scale_factor' if epoch['canonical_time'] == 'scale_factor' else 'native_time_code'
        if (epoch['canonical_time'] != actual['canonical_time'] or epoch.get(clock) is None
                or epoch.get(clock) != actual.get(clock)):
            raise ValueError('Structure and assembly graph actual clocks differ')
        attributes = structure_assembly_attributes(report, arrays, design['controls'])
        if row.get('galaxy_catalogue') is not None:
            stellar = design['controls']['stellar_component']
            if stellar is not None and request['components'][stellar].get('membership') != 'subfind_central_bound':
                raise ValueError('Central-galaxy quenching/morphology requires the same native central membership')
            measured, sources = _galaxy_attributes(row['galaxy_catalogue'], run, key, epoch, memory/4)
            attributes.update(measured); _merge(pins, sources)
        for name, value in attributes.items():
            if name in nodes[key]['attributes'] and nodes[key]['attributes'][name] != value:
                raise ValueError('Native assembly measurement conflicts with existing attribute: '+name)
        attributes['native_assembly_measurement_source'] = {'path': str(path), 'sha256': file_sha256(path)}
        updates[key] = attributes; definitions.append(attributes['assembly_measurement_definitions'])
    # Fixed physical extraction geometry is required across epochs. Native
    # concentration reference density may evolve and remains recorded in full.
    for definition in definitions[1:]:
        for name in ('spin', 'density_shape', 'potential_shape', 'stellar_morphology'):
            if definition.get(name) != definitions[0].get(name):
                raise ValueError('Assembly measurement aperture/shape definition changes across epochs')
    for path, checksum in pins.items():
        if file_sha256(path) != checksum:
            raise ValueError('Native assembly source changed during intake')
    return graph, updates, pins



def native_assembly_measurements_runner(parameters, context):
    from .halos import IdentityGraph
    if parameters['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('Native assembly exceeds stage allocation')
    original, updates, pins = assembly_measurement_inputs(parameters)
    graph = IdentityGraph(maximum_nodes=parameters['maximum_nodes'], maximum_edges=parameters['maximum_edges'])
    for epoch, clock in original['epochs'].items():
        graph.add_epoch(int(epoch), clock)
    for node in original['nodes']:
        key = tuple(node['key']); attributes = copy.deepcopy(node['attributes']); attributes.update(updates.get(key, {}))
        graph.add_node(*key, kind=node['kind'], status=node['status'], attributes=attributes)
    for edge in original['edges']:
        graph.add_edge(edge['source'], edge['target'], edge['kind'], evidence=edge['evidence'])
    pins[str(Path(__file__).resolve())] = file_sha256(__file__)
    result = graph.publish(context.generation/'identity-graph', source_hashes=pins)
    summary = {'schema': 'native-assembly-measurements-result-v1', 'execution_status': 'COMPLETE',
        'original_graph_fingerprint': original['fingerprint'], 'graph_fingerprint': result['fingerprint'],
        'observations_attached': len(updates), 'original_quality_and_ancestry_preserved': True,
        'science_acceptance': 'NOT_ACCEPTED'}
    atomic_json(context.generation/'measurements.json', summary)
    return summary



def native_assembly_history_inputs(parameters):
    from .halos import verify_identity_graph
    memory = parameters['memory_gib']
    if type(memory) not in (int, float) or not np.isfinite(memory) or memory <= 0:
        raise ValueError('Positive finite native assembly history memory required')
    root = Path(parameters['identity_graph']).resolve(); path = Path(parameters['template_path']).resolve()
    if max((root/'report.json').stat().st_size, path.stat().st_size)*24 > memory*1024**3/4:
        raise MemoryError('Native assembly history metadata exceeds allocation')
    if file_sha256(path) != parameters['template_sha256']:
        raise ValueError('Native assembly history template changed')
    with zipfile.ZipFile(root/'nodes.npz') as z:
        if sum(row.file_size for row in z.infolist())*8 > memory*1024**3/4:
            raise MemoryError('Native assembly history graph arrays exceed allocation')
    graph = verify_identity_graph(root); template = json.loads(path.read_text())
    if (set(template) != {'schema', 'target', 'branch', 'events', 'accretion_fit'}
            or template['schema'] != 'native-assembly-history-template-v1'):
        raise ValueError('Complete native assembly history template required')
    _key(template['target'])
    if {'epoch_time_myr', 'epoch_scale_factor'} & set(template['branch']):
        raise ValueError('Assembly clocks must derive from original native graph epochs')
    pins = {**graph['sources'], str(path): file_sha256(path),
            **{str(root/name): file_sha256(root/name) for name in ('report.json', 'report.sha256', 'nodes.npz', 'index.sqlite')}}
    for clock in graph['epochs'].values():
        source = clock.get('cosmic_time_source', {})
        if source.get('path'):
            if file_sha256(source['path']) != source['sha256']:
                raise ValueError('Native assembly proper-time source changed')
            pins[source['path']] = source['sha256']
    return graph, template, pins



def native_assembly_history_runner(parameters, context):
    """Run original branch/rate/event estimator on the measured native graph."""
    from core.stream.workflow import Inputs, execute_stream_stage
    from core.stream_common import stream_array_units
    from core.stream_common import publish_stream_product
    from .assembly import export_native_assembly_figures
    if parameters['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('Native assembly history exceeds stage allocation')
    graph, template, pins = native_assembly_history_inputs(parameters)
    clocks = graph['epochs']; times = {}; scales = {}
    for epoch, clock in clocks.items():
        if clock['canonical_time'] != 'scale_factor' or clock.get('cosmic_time_gyr') is None:
            raise ValueError('Assembly requires original cosmological proper-time graph clock')
        times[epoch] = clock['cosmic_time_gyr']*1000.; scales[epoch] = clock['actual_scale_factor']
    settings = {key: copy.deepcopy(template[key]) for key in ('target', 'branch', 'events', 'accretion_fit')}
    settings['branch'].update(epoch_time_myr=times, epoch_scale_factor=scales)
    stage = {'stage_id': 'native-assembly', 'kind': 'assembly_history',
             'inputs': {'native_graph_directory': str(Path(parameters['identity_graph']).resolve())},
             'parameters': settings}
    inputs = Inputs(context.generation, {}, {'memory_gib': parameters['memory_gib'], 'maximum_particles': 1})
    inputs.sources.update(pins)
    from . import source_path
    for source in (Path(__file__).resolve(), source_path('assembly'), source_path('stream.workflow')):
        inputs.sources[str(source)] = file_sha256(source)
    report, arrays, namespace, epoch = execute_stream_stage(stage, inputs, context.generation)
    mass_definitions = {row['attributes'].get('mass_definition') for row in report['native_branch']}
    if len(mass_definitions) != 1 or not next(iter(mass_definitions)):
        raise ValueError('Native assembly branch changes or omits its mass definition')
    report['mass_definition'] = next(iter(mass_definitions))
    report.update(request=stage, source_hashes=inputs.sources)
    path = context.generation/'assembly.json'
    publish_stream_product(path, report, arrays, stream_array_units(arrays), sources=inputs.sources, namespace=namespace, epoch=epoch)
    figures = export_native_assembly_figures(path, parameters['identity_graph'], context.generation/'figures', memory_gib=parameters['memory_gib'])
    result = {'schema': 'native-assembly-history-result-v1', 'execution_status': 'COMPLETE',
        'product': str(path), 'figure_count': len(figures['figures']),
        'native_quality_and_ancestry_preserved': True, 'science_acceptance': 'NOT_ACCEPTED'}
    atomic_json(context.generation/'history-result.json', result)
    return result



def prepare_native_assembly_campaign(request, destination):
    """Connect native measurement joins, original assembly and optional orbits."""
    from .workflow import default_registry, compile_requested_product_dag
    if request.get('schema') != 'native-assembly-campaign-v1':
        raise ValueError('Explicit native assembly campaign request required')
    out = Path(destination).resolve()
    if out.exists():
        raise FileExistsError('Keep previous native assembly preparation')
    p = copy.deepcopy(request['measurement_parameters'])
    graph, _, pins = assembly_measurement_inputs(p)
    memory = p['memory_gib']; disk = request['disk_gib']
    if type(disk) not in (int, float) or not np.isfinite(disk) or disk <= 0:
        raise ValueError('Positive finite native assembly disk allocation required')
    path = _binding(request['history_template'])
    hp = {'identity_graph': p['identity_graph'], 'template_path': str(path),
          'template_sha256': file_sha256(path), 'memory_gib': memory}
    _, template, sources = native_assembly_history_inputs(hp); _merge(pins,sources)
    if not any(row['key'] == template['target'] for row in graph['nodes']):
        raise ValueError('Native assembly recipe target is absent from original graph')
    hp['identity_graph'] = {'stage': 'assembly-measurements', 'artifact': 'identity-graph'}
    stages = [
        {'stage_id': 'assembly-measurements', 'kind': 'native_assembly_measurements', 'parameters': p,
         'outputs': {'assembly-graph': 'identity-graph', 'assembly-measurements': 'measurements.json'}},
        {'stage_id': 'assembly-history', 'kind': 'native_assembly_history', 'parameters': hp,
         'outputs': {'assembly-history': 'assembly.json', 'assembly-figures': 'figures'}}]
    requested = ['assembly-history', 'assembly-figures']
    orbit = request.get('orbits')
    if orbit is not None:
        from .flagship_orbits import native_orbit_inputs
        _merge(pins,native_orbit_inputs(p['identity_graph'], orbit, memory_gib=memory)[2])
        # Recorded catalogue images bind the original graph fingerprint.
        # Keep that exact graph as the orbital producer's source rather than
        # rewriting image provenance after attaching measurement attributes.
        stages.extend([
            {'stage_id': 'assembly-orbits', 'kind': 'flagship_product', 'parameters': {
                'operation': 'assembly-orbits', 'product': p['identity_graph'], 'request': orbit,
                'destination': '{generation}/orbits.json', 'memory_gib': memory},
             'outputs': {'assembly-orbits': 'orbits.json'}},
            {'stage_id': 'assembly-event-figures', 'kind': 'flagship_product', 'parameters': {
                'operation': 'assembly-event-figures', 'product': {'stage': 'assembly-orbits', 'artifact': 'orbits.json'},
                'destination': '{generation}/figures', 'memory_gib': memory},
             'outputs': {'assembly-event-figures': 'figures'}}])
        requested.append('assembly-event-figures')
    for row in stages:
        row['input_files'] = [{'path': name, 'sha256': digest} for name, digest in sorted(pins.items())]
    spec = {'schema': 'research-workflow-v1', 'resources': {'mpi_ranks': 1, 'omp_threads': 1,
        'memory_gib': memory, 'disk_gib': disk, 'maximum_stage_count': len(stages)},
        'physics_spec': {}, 'capabilities': [], 'followup_limits': {'maximum_children': 0, 'maximum_depth': 0},
        'requested_products': requested, 'stages': stages}
    registry = default_registry(); plan = compile_requested_product_dag(spec, registry, request_base=out)
    code = {}
    for row in plan['stages']:
        code.update(registry.source_hashes(row['kind']))
    for name, digest in {**pins, **code}.items():
        if file_sha256(name) != digest:
            raise ValueError('Native assembly preparation source changed')
    out.mkdir(parents=True)
    atomic_json(out/'request.json', request); atomic_json(out/'workflow.json', spec)
    result = {'schema': 'native-assembly-campaign-prepared-v1', 'status': 'RECIPES_PREPARED',
        'workflow': {'path': str(out/'workflow.json'), 'sha256': file_sha256(out/'workflow.json')},
        'source_hashes': {**pins, **code}, 'stage_count': len(plan['stages']),
        'measurement_observations': len(json.loads(Path(p['template_path']).read_text())['observations']),
        'orbital_recipe_configured': orbit is not None, 'science_acceptance': 'NOT_ACCEPTED',
        'scope': 'Exact FOF structure/central-galaxy joins and original native ancestry/quality; orbital path requires separately recorded original images'}
    atomic_json(out/'manifest.json', result)
    return result



# Native cosmic-web field, halo alignments and persistent-host histories

WEB_VARIANTS=['filament_map','spin_alignment','shape_alignment','web_evolution']


def _web_field_design(parameters):
    """Read explicit mesh/filter controls and bound their memory allocation.

    Smoothing lengths are comoving kpc; halo shell measurements remain proper
    kpc. An odd mesh avoids Nyquist-axis ambiguity in the original tidal-field
    estimator. The design selects its thresholds; this adapter invents none.
    """
    memory=parameters['memory_gib'];positive(memory,'native web allocation');path=Path(parameters['design_path']).resolve()
    if path.stat().st_size*16>memory*1024**3:raise MemoryError('Web metadata exceeds allocation')
    if file_sha256(path)!=parameters['design_sha256']:raise ValueError('Native web field design changed')
    design=json.loads(path.read_text())
    if set(design)!={'schema','environment'} or design['schema']!='native-web-field-design-v1':
        raise ValueError('Explicit native web field/estimator design required')
    fields=design['environment'];required={'mesh_size','smoothing_comoving_kpc','web_threshold','eigenvalue_gap_floor',
        'splashback_component','slope_controls','splashback_controls','alignment_controls','orbital_history'}
    if set(fields)-{'slope_window_study'}!=required:raise ValueError('Complete original native environment controls required')
    n=fields['mesh_size'];smooth=fields['smoothing_comoving_kpc']
    if type(n) is not int or n<5 or n%2!=1 or not isinstance(smooth,list) or not smooth or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in smooth) or len(set(smooth))!=len(smooth):
        raise ValueError('Original odd web mesh and distinct finite nonnegative smoothing scales required')
    if n**3*(512+208*len(smooth))+32*1024**2>memory*1024**3:raise MemoryError('Original native web mesh workspace exceeds allocation')
    return fields,{str(path):file_sha256(path)}


def native_web_epoch_inputs(parameters):
    """Bind one web field to the same native run/epoch as its halo structure.

    Optional DisPerSE skeletons retain their original published topology and
    source snapshots. They do not replace the filtered tidal-grid estimator.
    """
    from .halos import _snapshot_files,verify_halo_structure
    from .data import native_publication_markers,verify_native_hdf5_set,numeric_audit_source_bindings
    fields,pins=_web_field_design(parameters);run=Path(parameters['run_directory']).resolve();snapshot=parameters['snapshot']
    if type(snapshot) is not int or snapshot<0:raise ValueError('Exact native web saved epoch required')
    files=_snapshot_files(run,snapshot);verify_native_hdf5_set(files,require_publication=True)
    for path in files+native_publication_markers(files):_merge(pins,{str(Path(path).resolve()):file_sha256(path)})
    for name in ('config.yaml','manifest.json'):_merge(pins,{str(run/name):file_sha256(run/name)})
    path=Path(parameters['halo_structure_product']).resolve();structure=verify_halo_structure(path)
    if Path(structure['request']['run_directory']).resolve()!=run or structure['request']['snapshot']!=snapshot:
        raise ValueError('Web and halo structure must bind the same actual native run/epoch')
    _merge(pins,numeric_audit_source_bindings(structure))
    for asset in (path,path.with_name(path.name+'.sha256'),path.with_name(path.name+'.data')/'arrays.npz',path.with_name(path.name+'.data')/'contract.json'):
        _merge(pins,{str(asset):file_sha256(asset)})
    binding=parameters['filament_skeleton']
    if binding is not None:
        from .analysis import read_bound_filament_graph
        snapshot_sources={str(Path(p).resolve()):pins[str(Path(p).resolve())] for p in files}
        _,sources,_=read_bound_filament_graph(binding,run,snapshot,snapshot_sources,memory_bytes=int(parameters['memory_gib']*1024**3/4))
        _merge(pins,sources)
    request={'schema':'native-environment-v1','run_directory':str(run),'snapshot':snapshot,
        'halo_structure_products':[str(path)],'memory_gib':parameters['memory_gib'],**copy.deepcopy(fields)}
    if binding is not None:request['filament_skeleton']=copy.deepcopy(binding)
    return request,structure,pins


def native_web_epoch_runner(parameters,context):
    from .analysis import publish_native_environment,verify_native_environment
    from .halo_products import export_native_web_figures
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native web field exceeds stage allocation')
    request,_,pins=native_web_epoch_inputs(parameters);path=context.generation/'web-environment.json'
    publish_native_environment(request,path,request_source_hashes=pins);report=verify_native_environment(path)
    gallery=export_native_web_figures(path,context.generation/'figures',memory_gib=parameters['memory_gib'])
    return {'execution_status':'COMPLETE','science_acceptance':'NOT_ACCEPTED','variants':WEB_VARIANTS[:3],
        'figure_count':len(gallery['figures']),'massive_particle_rows':report['massive_particle_rows'],
        'support':'Original filtered-density, eigengap, native spin and density-shape orientation support; no forced filament detections'}


def native_web_history_inputs(parameters):
    """Bind saved web fields along a recorded halo ancestry with proper clocks.

    The host is persistent through graph links; voxel labels or nearest visual
    filaments are not cross-epoch identities. Maximum intervals are in Myr.
    """
    from .halos import IdentityQueries,verify_halo_structure
    from .analysis import verify_native_environment,_bind_structure_graph_observation
    from .data import numeric_audit_source_bindings
    memory=parameters['memory_gib'];positive(memory,'native web allocation')
    root=Path(parameters['identity_graph_directory']).resolve()
    if sum((root/name).stat().st_size for name in ('report.json','index.sqlite','nodes.npz'))*24>memory*1024**3:raise MemoryError('Web history graph exceeds allocation')
    graph=IdentityQueries(root);nodes={tuple(n['key']):n for n in graph.report['nodes']};pins=dict(graph.report['sources'])
    for name in ('report.json','report.sha256','index.sqlite','nodes.npz'):_merge(pins,{str(root/name):file_sha256(root/name)})
    observations=parameters['observations']
    if not isinstance(observations,list) or len(observations)<2:raise ValueError('At least two actual native web history observations required')
    previous=None
    for observation in observations:
        if set(observation)!={'environment_product','halo_structure_product','graph_node'}:raise ValueError('Exact native environment/structure/graph observation required')
        environment=verify_native_environment(observation['environment_product']);structure=verify_halo_structure(observation['halo_structure_product'])
        key=_key(observation['graph_node'])
        if key[2]!=structure['request']['snapshot']:raise ValueError('Exact web graph storage epoch differs from native structure')
        _bind_structure_graph_observation(structure,key,graph,nodes,previous);previous=key
        if (Path(environment['request']['run_directory']).resolve()!=Path(structure['request']['run_directory']).resolve()
                or environment['request']['snapshot']!=structure['request']['snapshot']):raise ValueError('Native web history environment/structure epochs differ')
        for name,product in [('environment_product',environment),('halo_structure_product',structure)]:
            _merge(pins,numeric_audit_source_bindings(product));path=Path(observation[name]).resolve()
            for asset in (path,path.with_name(path.name+'.sha256'),path.with_name(path.name+'.data')/'arrays.npz',path.with_name(path.name+'.data')/'contract.json'):_merge(pins,{str(asset):file_sha256(asset)})
    positive(parameters['maximum_interval_myr'],'web history interval Myr')
    return pins


def native_web_history_runner(parameters,context):
    from .analysis import publish_environment_history,verify_environment_history
    from .halo_products import export_native_web_history
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Web history exceeds stage allocation')
    pins=native_web_history_inputs(parameters)
    request={'schema':'native-environment-history-v1',**{k:copy.deepcopy(parameters[k]) for k in ('identity_graph_directory','observations','maximum_interval_myr','memory_gib')}}
    path=context.generation/'web-history.json';publish_environment_history(request,path,request_source_hashes=pins);verify_environment_history(path)
    gallery=export_native_web_history(path,context.generation/'figures',memory_gib=parameters['memory_gib'])
    return {'execution_status':'COMPLETE','science_acceptance':'NOT_ACCEPTED','variants':WEB_VARIANTS[3:],
        'figure_count':len(gallery['figures']),'support':'Original graph ancestry and clock-bounded saved intervals; no cross-epoch voxel identity inferred'}


def prepare_native_web_campaign(request,destination):
    """Connect authored field/halo observations to epoch and history outputs.

    This is analysis of retained native data. It prepares no preset physical
    run and does not execute the web/figure stages while compiling their graph.
    """
    from .workflow import default_registry,compile_requested_product_dag
    from .halos import IdentityQueries
    from .analysis import _bind_structure_graph_observation
    required={'schema','memory_gib','disk_gib','field_design','identity_graph_directory','observations','maximum_interval_myr'}
    if set(request)!=required or request['schema']!='native-web-campaign-v1':raise ValueError('Complete native four-variant web campaign required')
    for key in ('memory_gib','disk_gib','maximum_interval_myr'):positive(request[key],key)
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native web preparation')
    rows=request['observations']
    if not isinstance(rows,list) or len(rows)<2:raise ValueError('At least two native web/halo observations required')
    memory=request['memory_gib'];root=Path(request['identity_graph_directory']).resolve()
    if sum((root/name).stat().st_size for name in ('report.json','index.sqlite','nodes.npz'))*24>memory*1024**3:raise MemoryError('Web preparation graph exceeds allocation')
    graph=IdentityQueries(root);nodes={tuple(n['key']):n for n in graph.report['nodes']};pins=dict(graph.report['sources'])
    for name in ('report.json','report.sha256','index.sqlite','nodes.npz'):_merge(pins,{str(root/name):file_sha256(root/name)})
    stages=[];observations=[];requested=[];previous=None;native_run=None;snapshots=set()
    for i,row in enumerate(rows):
        if set(row)!={'halo_structure_product','graph_node','filament_skeleton'}:raise ValueError('Complete exact native web observation required')
        binding=row['halo_structure_product'];path=Path(binding['path']).resolve()
        if file_sha256(path)!=binding['sha256']:raise ValueError('Native web structure binding changed')
        from .halos import verify_halo_structure
        structure=verify_halo_structure(path);run=Path(structure['request']['run_directory']).resolve();snapshot=structure['request']['snapshot']
        if (native_run is not None and run!=native_run) or snapshot in snapshots:raise ValueError('Native web campaign requires one run and distinct saved observations')
        native_run=run;snapshots.add(snapshot);key=_key(row['graph_node'])
        if key[2]!=snapshot:raise ValueError('Exact web graph storage epoch differs from native structure')
        _bind_structure_graph_observation(structure,key,graph,nodes,previous);previous=key
        parameters={'run_directory':str(run),'snapshot':snapshot,'halo_structure_product':str(path),'filament_skeleton':copy.deepcopy(row['filament_skeleton']),
            'design_path':str(Path(request['field_design']['path']).resolve()),'design_sha256':request['field_design']['sha256'],'memory_gib':memory}
        _,_,sources=native_web_epoch_inputs(parameters);_merge(pins,sources);name=f'web-epoch-{i:03d}'
        stages.append({'stage_id':name,'kind':'native_web_epoch','parameters':parameters,'outputs':{name:'web-environment.json',name+'-figures':'figures'}});requested.append(name+'-figures')
        observations.append({'environment_product':{'stage':name,'artifact':'web-environment.json'},'halo_structure_product':str(path),'graph_node':list(key)})
    stages.append({'stage_id':'web-history','kind':'native_web_history','parameters':{'identity_graph_directory':str(root),
        'observations':observations,'memory_gib':memory,'maximum_interval_myr':request['maximum_interval_myr']},'outputs':{'web-history':'web-history.json','web-history-figures':'figures'}})
    for stage in stages:stage['input_files']=[{'path':p,'sha256':h} for p,h in sorted(pins.items())]
    spec={'schema':'research-workflow-v1','resources':{'mpi_ranks':1,'omp_threads':1,'memory_gib':memory,'disk_gib':request['disk_gib'],'maximum_stage_count':len(stages)},
        'physics_spec':{},'capabilities':[],'followup_limits':{'maximum_children':0,'maximum_depth':0},'requested_products':requested+['web-history-figures'],'stages':stages}
    registry=default_registry();plan=compile_requested_product_dag(spec,registry,request_base=out)
    for kind in dict.fromkeys(stage['kind'] for stage in plan['stages']):
        _merge(pins,registry.source_hashes(kind))
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Native web source changed during preparation')
    out.mkdir(parents=True);atomic_json(out/'request.json',request);atomic_json(out/'workflow.json',spec)
    result={'schema':'native-web-campaign-prepared-v1','status':'RECIPES_PREPARED','stage_count':len(stages),'variants':WEB_VARIANTS,
        'source_hashes':pins,'workflow':{'path':str(out/'workflow.json'),'sha256':file_sha256(out/'workflow.json')},
        'science_acceptance':'NOT_ACCEPTED','scope':'Complete native T-web epoch field/halo alignment producers and original persistent-host proper-time history; optional original published arc topology'}
    atomic_json(out/'manifest.json',result);return result
