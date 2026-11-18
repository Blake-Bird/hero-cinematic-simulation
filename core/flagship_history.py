"""Connect native campaign artifacts to the existing material history estimators."""
from pathlib import Path
import copy
import json
import math
import sys
import numpy as np
from .foundation import atomic_json, file_sha256


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Positive finite history bound required: ' + label)


def validate_material_map(parameters):
    for name in ('memory_gib', 'box_comoving_kpc_h'):
        _positive(parameters[name], name)
    if type(parameters['parent_level']) is not int or not 1 <= parameters['parent_level'] <= 20:
        raise ValueError('Exact original parent level required')
    if type(parameters['maximum_tracers']) is not int or parameters['maximum_tracers'] < 1:
        raise ValueError('Exact positive material tracer limit required')


def _analysis_stage(options,*,identifier,evolution,recovery,contamination):
    """Use the existing public analysis route on an owned copy of the run."""
    from commands import parse_products
    if not isinstance(options,dict) or set(options)!={'products','wall_hours','find_halos','copy_disk_gib'}:
        raise ValueError('Explicit analysis products, wall_hours, find_halos and copy_disk_gib required')
    products=options['products'];_positive(options['wall_hours'],'analysis wall_hours')
    _positive(options['copy_disk_gib'],'analysis copy disk')
    if (not isinstance(products,list) or not products or any(not isinstance(p,str) for p in products)
            or len(set(products))!=len(products) or type(options['find_halos']) is not bool):
        raise ValueError('Distinct named analysis products and exact finder choice required')
    # These public products require additional explicit measurement requests,
    # or a separate scene/inference producer. Do not hide missing inputs by
    # allowing incomplete analysis or selecting an unconfigured "all" route.
    extra={'all','stellar_components','stellar_histories','zoom_contamination',
        'inference_diagnostics','radial_shapes','blender'}
    if set(products)&extra:raise ValueError('Selected analysis product needs its separate explicit request/producer')
    if parse_products(products)!=products:raise ValueError('Use exact public analysis product names')
    name='analyze_'+identifier
    args=[{'stage':evolution,'artifact':'run'},'--graphs',','.join(products)]
    if options['find_halos']:args.append('--find-halos')
    return {'stage_id':name,'kind':'public_command','depends_on':[recovery]+([contamination] if contamination else []),
        'parameters':{'command':'analysis','wall_seconds':options['wall_hours']*3600,
            'copy_run_input':0,'copy_run_max_bytes':math.ceil(options['copy_disk_gib']*1024**3),
            'arguments':args},'outputs':{name:'private_run'}}


def validate_material_recovery(parameters):
    _positive(parameters['memory_gib'], 'memory_gib')
    snapshots = parameters['snapshots']
    if (not isinstance(snapshots, list) or not snapshots or len(set(snapshots)) != len(snapshots)
            or any(type(s) is not int or s < 0 for s in snapshots)):
        raise ValueError('Distinct exact saved snapshot identities required')
    epochs = parameters['requested_redshifts']
    if (not isinstance(epochs, list) or len(epochs) != len(snapshots)
            or any(type(z) not in (int, float) or not math.isfinite(z) or z < 0 for z in epochs)
            or len(set(epochs)) != len(epochs)):
        raise ValueError('Distinct requested redshifts must cover the saved snapshot list')
    error = parameters['maximum_redshift_error']
    if type(error) not in (int, float) or not math.isfinite(error) or error < 0:
        raise ValueError('Explicit nonnegative epoch error required')
    rules = parameters['rules']
    for name in ('minimum_material_fraction', 'maximum_runner_up_fraction'):
        if type(rules[name]) not in (int, float) or not math.isfinite(rules[name]):
            raise ValueError('Finite explicit material recovery rules required')
    if not 0 < rules['minimum_material_fraction'] <= 1 or not 0 <= rules['maximum_runner_up_fraction'] < rules['minimum_material_fraction']:
        raise ValueError('Explicit material match/ambiguity thresholds required')
    if type(rules['minimum_shared_particles']) is not int or rules['minimum_shared_particles'] < 1:
        raise ValueError('Exact positive shared-particle threshold required')


def _history_answer_template(case, branch, evolution, stages, bind):
    """Read the frozen schedule without replacing runtime flow answers."""
    argv=evolution['parameters']['arguments']
    if argv.count('--answers')!=1 or argv.index('--answers')+1>=len(argv):
        raise ValueError('History evolution requires one explicit answers reference')
    reference=argv[argv.index('--answers')+1]
    key=(branch+'_' if branch else '')+'flow_preparation_stage'
    producer=case.get(key)
    def stage_input(stage,path):
        path=Path(path).resolve()
        matches=[ref for ref in stage.get('input_files',[]) if Path(ref['path']).resolve()==path]
        if len(matches)!=1:raise ValueError('History answers need one exact stage input binding')
        return bind(matches[0])
    if isinstance(reference,dict) and set(reference)=={'input'}:
        if producer is not None:
            raise ValueError('History evolution bypasses its declared flow preparation')
        return stage_input(evolution,reference['input'])
    if (not producer or reference!={'stage':producer,'artifact':'prepared/answers.json'}
            or producer not in stages):
        raise ValueError('History runtime answers must come from the declared flow preparation')
    stage=stages[producer];parameters=stage['parameters'];args=parameters['arguments']
    if (parameters.get('command')!='prepare-generated-flow' or args.count('--request')!=1
            or args.index('--request')+1>=len(args)):
        raise ValueError('History runtime answers require the public generated-flow producer')
    generation=case.get('generation_stage',case.get('shared_generation_stage'))
    if (args.count('--generation-report')!=1 or args.index('--generation-report')+1>=len(args)
            or args[args.index('--generation-report')+1]!={'stage':generation,'artifact':'generated/generation.json'}):
        raise ValueError('History flow preparation uses another case generation')
    request_ref=args[args.index('--request')+1]
    if not isinstance(request_ref,dict) or set(request_ref)!={'input'}:
        raise ValueError('History flow preparation needs a pinned request')
    request=json.loads(stage_input(stage,request_ref['input']).read_text())
    if request.get('schema')!='native-generated-flow-preparation-request-v1':
        raise ValueError('History requires the generated-flow answer template contract')
    template=request['answers'];path=bind(template)
    # Both preparation and evolution freeze this same physical template.
    # The stage-produced answers add the actual mesh-dependent gate at runtime.
    for owner in (stage,evolution):
        if stage_input(owner,path)!=path:raise ValueError('History physical template differs across producers')
    return path


def material_map_runner(parameters, context):
    """Map original parent cells to exact uint64 IDs in the generated zoom.

    The parent level identifies the original Lagrangian cell grid, which can
    differ from the zoom's coarsest retained level. Box length is comoving
    kpc/h; the mapping and its fine-DM mass inventory stay bound to those ICs.
    """
    import h5py
    from .zoom import verify_generated_zoom, fine_dm_ic_inventory
    from core.research.zooms import map_lagrangian_target_to_zoom_ids
    validate_material_map(parameters)
    if parameters['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('Material mapping exceeds workflow memory allocation')
    report = Path(parameters['generation_report']).resolve()
    generation = verify_generated_zoom(report)
    ic = Path(parameters['zoom_ic']).resolve()
    if ic != report.parent / 'ics_gadget.hdf5':
        raise ValueError('Material mapping IC differs from the verified generation')
    traced = generation['traced_region']; region_path = Path(traced['path']).resolve()
    if file_sha256(region_path) != traced['sha256']:
        raise ValueError('Generated target tracing source changed')
    region = json.loads(region_path.read_text())
    selected = Path(parameters['selected_particles']).resolve()
    if selected != region_path.parent / 'selected_particles.hdf5' or file_sha256(selected) != region['selected_particles_sha256']:
        raise ValueError('Material selection differs from the generated traced target')
    with h5py.File(ic) as source:
        actual_box = float(source['Header'].attrs['BoxSize'])
    if not math.isclose(actual_box, parameters['box_comoving_kpc_h'], rel_tol=1e-12):
        raise ValueError('Material mapping box differs from the actual native IC')
    ids, mapping = map_lagrangian_target_to_zoom_ids(parameters['selected_particles'], ic,
        parent_level=parameters['parent_level'], box_comoving_kpc_h=parameters['box_comoving_kpc_h'],
        memory_gib=parameters['memory_gib'], maximum_tracers=parameters['maximum_tracers'])
    mapping['fine_dm_initial_inventory'] = fine_dm_ic_inventory([{'path':str(ic),'sha256':mapping['source_hashes'][str(ic)]}])
    path = context.generation / 'tracer-ids.npy'
    np.save(path, ids, allow_pickle=False)
    mapping['tracer_ids'] = {'path': str(path), 'sha256': file_sha256(path), 'dtype': 'uint64'}
    mapping['source_hashes'].update({str(report): file_sha256(report), str(region_path): traced['sha256']})
    atomic_json(context.generation / 'material-map.json', mapping)
    return {'tracer_count': len(ids), 'science_acceptance': 'NOT_ACCEPTED'}


def material_recovery_runner(parameters, context):
    from core.research.zooms import recover_lagrangian_descendant, require_history_epoch_coverage
    validate_material_recovery(parameters)
    if parameters['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('Material recovery exceeds workflow memory allocation')
    mapping_path = Path(parameters['material_map']).resolve()
    mapping = json.loads(mapping_path.read_text())
    if mapping.get('schema') != 'parent-zoom-material-map-v1':
        raise ValueError('Exact parent-to-zoom material map required')
    binding = mapping['tracer_ids']; path = Path(binding['path']).resolve()
    if path != mapping_path.parent / 'tracer-ids.npy' or file_sha256(path) != binding['sha256']:
        raise ValueError('Material tracer artifact changed or escaped its producer')
    pins = {str(mapping_path): file_sha256(mapping_path), str(path): binding['sha256'], **mapping['source_hashes']}
    for source, digest in pins.items():
        if file_sha256(source) != digest:
            raise ValueError('Material mapping source changed: ' + source)
    # Map before reading; a malicious NPY shape must not cause a large allocation.
    ids = np.load(path, mmap_mode='r', allow_pickle=False)
    if ids.dtype != np.dtype('uint64') or ids.ndim != 1 or len(ids) != mapping['fine_zoom_tracer_count']:
        raise ValueError('Exact uint64 material tracer inventory required')
    if ids.nbytes * 4 > parameters['memory_gib'] * 1024**3 / 2:
        raise MemoryError('Material tracer workspace exceeds declared memory')
    recovery = recover_lagrangian_descendant(parameters['run_dir'], ids, parameters['snapshots'],
        parameters['rules'], memory_gib=parameters['memory_gib'])
    recovery['source_hashes'].update(pins)
    # Preserve recovery evidence even when the separate epoch-coverage gate fails.
    atomic_json(context.generation / 'recovery.json', recovery)
    coverage = require_history_epoch_coverage(recovery, parameters['requested_redshifts'],
        maximum_redshift_error=parameters['maximum_redshift_error'], memory_gib=parameters['memory_gib'])
    atomic_json(context.generation / 'epoch-coverage.json', coverage)
    for source, digest in recovery['source_hashes'].items():
        if file_sha256(source) != digest:
            raise ValueError('Recovery input changed during measurement: ' + source)
    return {'status': recovery['status'], 'epoch_coverage': coverage['status'], 'science_acceptance': 'UNRESOLVED'}


def validate_contamination_history(parameters):
    for name in ('memory_gib', 'aperture_r200c_factor'):
        _positive(parameters[name], name)
    # Existing catalogue reader reserves 64 MiB; contamination uses fixed
    # 100,000-row streaming chunks. Reserve a further 32 MiB for those arrays.
    if parameters['memory_gib'] < 96 / 1024:
        raise MemoryError('Native contamination history requires at least 96 MiB workspace')
    limit = parameters['maximum_coarse_mass_fraction']
    if type(limit) not in (int, float) or not math.isfinite(limit) or not 0 <= limit <= 1:
        raise ValueError('Explicit native coarse mass fraction limit in [0,1] required')


def contamination_history_runner(parameters, context):
    """Measure recovered halo apertures; unresolved targets retain empty support."""
    from .zoom import _run_lineage, read_host_catalogue, contamination_epoch
    validate_contamination_history(parameters)
    if parameters['memory_gib'] > context.resources['memory_gib']:
        raise MemoryError('Contamination history exceeds workflow memory allocation')
    recovery_path = Path(parameters['recovery_path']).resolve()
    map_path = Path(parameters['material_map']).resolve()
    recovery = json.loads(recovery_path.read_text()); mapping = json.loads(map_path.read_text())
    if recovery.get('schema') != 'lagrangian-descendant-recovery-v1' or mapping.get('schema') != 'parent-zoom-material-map-v1':
        raise ValueError('Native contamination needs exact recovery and original material map')
    pins = {str(recovery_path): file_sha256(recovery_path), str(map_path): file_sha256(map_path)}
    for record in (mapping, recovery):
        for source, digest in record['source_hashes'].items():
            if source in pins and pins[source] != digest:
                raise ValueError('Conflicting contamination input source bindings')
            pins[source] = digest
    for source, digest in pins.items():
        if file_sha256(source) != digest:
            raise ValueError('Contamination history input changed: ' + source)
    lineage = _run_lineage(parameters['run_dir'])
    initial = mapping['fine_dm_initial_inventory']
    if sorted(q['sha256'] for q in initial['initial_conditions']) != sorted(q['sha256'] for q in lineage['initial_conditions']):
        raise ValueError('Evolved run uses a different IC material namespace')
    for key in ('fine_dm_mass_msun', 'fine_dm_mass_range_msun', 'h'):
        if initial[key] != lineage[key]:
            raise ValueError('Native IC fine mass inventory differs from material mapping')
    pins[lineage['path']] = lineage['sha256']
    pins.update({q['path']:q['sha256'] for q in lineage['initial_conditions']})
    records = []; snapshots = set()
    for match in recovery['records']:
        snapshot = match['snapshot']
        if type(snapshot) is not int or snapshot < 0 or snapshot in snapshots:
            raise ValueError('Distinct exact contamination snapshot identities required')
        snapshots.add(snapshot)
        if match['status'] != 'MATCHED':
            records.append({'snapshot':snapshot,'scale_factor':match['scale_factor'],
                'status':'UNRESOLVED_TARGET','recovery_status':match['status'], 'native_contamination':None})
            continue
        cat = read_host_catalogue(parameters['run_dir'], snapshot, memory_gib=parameters['memory_gib'])
        row = match['descendant']['host_row']; a, box, h = cat['frame']
        if type(row) is not int or not 0 <= row < cat['group_count']:
            raise ValueError('Recovered contamination host is outside the native catalogue')
        if not math.isclose(a,match['scale_factor'],rel_tol=0,abs_tol=1e-12) or not math.isclose(h,lineage['h'],rel_tol=1e-12):
            raise ValueError('Native contamination and recovery/IC frames differ')
        sources = {q['path']:q['sha256'] for q in cat['sources']+cat['snapshot_sources']}
        if any(sources.get(source) != digest for source,digest in match['source_hashes'].items()):
            raise ValueError('Recovery uses different catalogue/snapshot sources')
        radius = float(cat['radii'][row])*a/h*parameters['aperture_r200c_factor']
        measured = contamination_epoch(cat['snapshots'], snapshot, center_comoving_kpc_h=cat['positions'][row],
            radius_proper_kpc=radius, fine_dm_mass_msun=lineage['fine_dm_mass_msun'],
            fine_dm_mass_range_msun=lineage['fine_dm_mass_range_msun'],
            maximum_coarse_mass_fraction=parameters['maximum_coarse_mass_fraction'])
        if not math.isclose(measured['scale_factor'],a,rel_tol=0,abs_tol=1e-12):
            raise ValueError('Contamination measurement differs from the recovered epoch')
        records.append({'snapshot':snapshot,'scale_factor':a,'host_row':row,'status':measured['status'],
            'aperture_definition':'native R200c times declared factor; proper kpc', 'native_contamination':measured})
        for source, digest in sources.items():
            if source in pins and pins[source] != digest:
                raise ValueError('Native contamination catalogue sources changed')
            pins[source] = digest
    for source,digest in pins.items():
        if file_sha256(source) != digest:
            raise ValueError('Contamination input changed during measurement: ' + source)
    status = 'PASS' if records and all(q['status']=='PASS' for q in records) else 'FAIL' if any(q['status']=='FAIL' for q in records) else 'UNRESOLVED'
    result = {'schema':'native-zoom-contamination-history-v1','status':status,'records':records,
        'source_hashes':pins,'fine_dm_initial_inventory':initial,
        'aperture_r200c_factor':parameters['aperture_r200c_factor'],
        'maximum_coarse_mass_fraction':parameters['maximum_coarse_mass_fraction'],
        'scope':'Recovered halo apertures only; progenitor/stream/encounter apertures and force/observable adequacy remain separate',
        'execution_status':'COMPLETE','science_acceptance':'UNRESOLVED'}
    atomic_json(context.generation/'contamination.json',result)
    if status=='FAIL':
        raise ValueError('Recovered halo exceeds the frozen coarse-mass criterion; contamination.json retained')
    return result


def prepare_history_campaign(request, destination):
    """Add selected measurements to the caller's native generation/run recipes.

    Shared-IC branches reuse one material map and recover each descendant
    separately. Recovery matches actual saved epochs to the requested redshift
    schedule; unresolved targets remain unsupported. Apertures, source audits,
    streams, encounters and analysis are added only when explicitly requested.
    """
    from .workflow import compile_requested_product_dag, safe_id
    from .zoom import _configuration
    from commands import redshift_schedule
    if request.get('schema') != 'flagship-history-campaign-request-v1':
        raise ValueError('Explicit native history campaign request required')
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError('Preserve previous history preparation')
    pins = {}
    def bind(reference):
        if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
            raise ValueError('Exact history input binding required')
        path = Path(reference['path']).resolve()
        if not path.is_file() or file_sha256(path) != reference['sha256']:
            raise ValueError('History campaign input changed: ' + str(path))
        pins[str(path)] = reference['sha256']
        return path
    manifest = json.loads(bind(request['campaign']).read_text())
    allowed = {'flagship-native-campaign-v1', 'flagship-production-campaign-v1', 'flagship-matched-native-campaign-v1'}
    if manifest.get('schema') not in allowed or manifest.get('status') != 'RECIPES_PREPARED':
        raise ValueError('Prepared native generation/evolution campaign required')
    for source, digest in manifest['source_hashes'].items():
        bind({'path': source, 'sha256': digest})
    spec = json.loads(bind(manifest['workflow']).read_text())
    stages = spec['stages']; by_id = {s['stage_id']: s for s in stages}
    cases = manifest['cases']
    if not isinstance(cases, list) or not cases or any(not isinstance(c, dict) for c in cases):
        raise ValueError('History campaign requires a nonempty list of prepared targets')
    case_ids = [safe_id(case['id']) for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError('Distinct prepared history target identities required')
    if set(request['cases']) != set(case_ids):
        raise ValueError('Explicit history controls must cover exactly the prepared targets')
    records = []
    for case in cases:
        identifier = safe_id(case['id']); controls = request['cases'][identifier]
        gen = case.get('generation_stage', case.get('shared_generation_stage'))
        stage = by_id[gen]; args = stage['parameters']['arguments']
        def argument(flag):
            if args.count(flag) != 1 or args.index(flag) + 1 >= len(args):
                raise ValueError('Unambiguous native generator argument required: ' + flag)
            ref = args[args.index(flag) + 1]
            if not isinstance(ref, dict) or set(ref) != {'input'}:
                raise ValueError('Pinned native generator file required')
            path = Path(ref['input']).resolve()
            if str(path) not in pins:
                raise ValueError('Native generator file lacks source binding')
            return path
        region_path = argument('--traced-region'); region = json.loads(region_path.read_text())
        if region['catalogue_binding']['host_fof_row'] != case['host_row']:
            raise ValueError('History host differs from traced target')
        particles = region_path.parent / 'selected_particles.hdf5'
        bind({'path': str(particles), 'sha256': region['selected_particles_sha256']})
        configuration = Path(args[0]['input']).resolve()
        if str(configuration) not in pins:
            raise ValueError('History generator configuration lacks source binding')
        cfg = _configuration(configuration)
        parent_level = case.get('original_parent_level', cfg.getint('setup', 'levelmin'))
        if controls['parent_level'] != parent_level:
            raise ValueError('History mapping must use the original parent cell namespace')
        common = {k: copy.deepcopy(controls[k]) for k in ('parent_level', 'maximum_tracers', 'memory_gib')}
        common['box_comoving_kpc_h'] = cfg.getfloat('setup', 'boxlength') * 1000
        validate_material_map(common)
        if common['memory_gib'] > spec['resources']['memory_gib']:
            raise MemoryError('History allocation exceeds campaign memory')
        mapping = 'map_' + identifier
        stages.append({'stage_id': mapping, 'kind': 'zoom_material_map', 'parameters': {**common,
            'selected_particles': {'input': str(particles)}, 'zoom_ic': {'stage': gen, 'artifact': 'generated/ics_gadget.hdf5'},
            'generation_report': {'stage': gen, 'artifact': 'generated/generation.json'}},
            'input_files': [{'path': str(particles), 'sha256': pins[str(particles)]}],
            'outputs': {mapping: 'material-map.json', mapping + '_ids': 'tracer-ids.npy'}})
        branches = [(None, case['evolution_stage'])] if 'evolution_stage' in case else [(b, case[b + '_evolution_stage']) for b in ('cdm', 'sidm')]
        for branch, evolution in branches:
            branch_id = (branch + '_' if branch else '') + identifier
            evo = by_id[evolution]; argv = evo['parameters']['arguments']
            answers_path = _history_answer_template(case,branch,evo,by_id,bind)
            answers = json.loads(answers_path.read_text())
            epochs = redshift_schedule(str(answers['schedule']), cfg.getfloat('setup', 'zstart'))
            params = {k: copy.deepcopy(controls[k]) for k in ('memory_gib', 'snapshots', 'rules', 'maximum_redshift_error')}
            params['requested_redshifts'] = epochs
            validate_material_recovery(params)
            recovery = 'recover_' + branch_id
            stages.append({'stage_id': recovery, 'kind': 'zoom_material_recovery', 'parameters': {**params,
                'run_dir': {'stage': evolution, 'artifact': 'run'}, 'material_map': {'stage': mapping, 'artifact': 'material-map.json'}},
                'outputs': {recovery: 'recovery.json', recovery + '_epochs': 'epoch-coverage.json'}})
            spec['requested_products'].extend([recovery, recovery + '_epochs'])
            record = {'id': identifier, 'branch': branch, 'mapping_stage': mapping, 'recovery_stage': recovery}
            if controls.get('restart_exercise') is not None:
                settings=controls['restart_exercise']
                common_restart={'snapshot','pause_step','memory_gib','wall_seconds'}
                if not isinstance(settings,dict):raise ValueError('Complete frozen restart exercise settings required')
                if set(settings)==common_restart|{'template'}:
                    restart_binding=settings['template']
                elif set(settings)==common_restart|{'templates_by_branch'}:
                    selected_templates=settings['templates_by_branch']
                    if (not isinstance(selected_templates,dict) or set(selected_templates)!={b for b,_ in branches}
                            or branch not in selected_templates):
                        raise ValueError('Restart exercise needs exactly one frozen template for each actual native branch')
                    restart_binding=selected_templates[branch]
                else:raise ValueError('Complete frozen restart exercise settings required')
                template=bind(restart_binding)
                def evolution_argument(flag):
                    if argv.count(flag)!=1 or argv.index(flag)+1>=len(argv):raise ValueError('Restart exercise requires the original evolution '+flag)
                    return copy.deepcopy(argv[argv.index(flag)+1])
                params={k:settings[k] for k in ('snapshot','pause_step','memory_gib','wall_seconds')}
                params.update(answers_path=evolution_argument('--answers'),existing_ic=evolution_argument('--existing-ic'),
                    generation_report=evolution_argument('--native-generation-report'),template_path={'input':str(template)},template_sha256=restart_binding['sha256'])
                from core.research.restart_campaign import validate_restart_exercise
                validate_restart_exercise(params)
                if settings['memory_gib']>spec['resources']['memory_gib']:raise MemoryError('Restart exercise exceeds campaign memory')
                from core.research.restart_campaign import prepare_exercise_comparison
                from core.research.capabilities import validate_native_answers
                if template.stat().st_size>min(16*1024**2,settings['memory_gib']*1024**3/16):
                    raise MemoryError('Restart template exceeds bounded campaign preparation intake')
                diagnostic=copy.deepcopy(answers)
                diagnostic.update(restart_source_capsule=True,restart_exercise_pause_step=0)
                selected=validate_native_answers(diagnostic,spec['resources'])
                prepare_exercise_comparison(json.loads(template.read_text()),selected,
                    workspace=target/'restart-contracts'/branch_id,snapshot=settings['snapshot'],memory_gib=settings['memory_gib'])
                name='restart_exercise_'+branch_id
                stages.append({'stage_id':name,'kind':'campaign_restart_exercise','parameters':params,'depends_on':[evolution],
                    'input_files':[restart_binding],'outputs':{name:'restart-comparison.json'}})
                spec['requested_products'].append(name);record['restart_exercise_stage']=name
            if controls.get('restart_comparison') is not None:
                settings=controls['restart_comparison']
                if not isinstance(settings,dict) or set(settings)!={'uninterrupted_stage','template','snapshot','memory_gib'}:
                    raise ValueError('Complete frozen native restart comparison controls required')
                baseline=settings['uninterrupted_stage']
                if baseline==evolution or baseline not in by_id or by_id[baseline]['kind']!='public_command' or by_id[baseline]['parameters'].get('command')!='gadget':
                    raise ValueError('Restart comparison requires a distinct declared uninterrupted native evolution stage')
                template=bind(settings['template'])
                from core.research.restart_campaign import validate_campaign_restart_comparison
                params={'uninterrupted_run':{'stage':baseline,'artifact':'run'},'resumed_run':{'stage':evolution,'artifact':'run'},
                    'template_path':{'input':str(template)},'template_sha256':settings['template']['sha256'],
                    'snapshot':settings['snapshot'],'memory_gib':settings['memory_gib']}
                validate_campaign_restart_comparison(params)
                if settings['memory_gib']>spec['resources']['memory_gib']:raise MemoryError('Restart comparison exceeds campaign memory')
                name='restart_compare_'+branch_id
                stages.append({'stage_id':name,'kind':'campaign_restart_comparison','parameters':params,
                    'input_files':[settings['template']],'outputs':{name:'restart-comparison.json'}})
                spec['requested_products'].append(name);record['restart_comparison_stage']=name
            contamination=None
            if controls.get('contamination') is not None:
                contamination = 'contamination_' + branch_id
                options = {**copy.deepcopy(controls['contamination']), 'memory_gib':controls['memory_gib']}
                validate_contamination_history(options)
                stages.append({'stage_id':contamination,'kind':'zoom_contamination_history','parameters':{**options,
                    'run_dir':{'stage':evolution,'artifact':'run'}, 'recovery_path':{'stage':recovery,'artifact':'recovery.json'},
                    'material_map':{'stage':mapping,'artifact':'material-map.json'}},
                    'outputs':{contamination:'contamination.json'}})
                spec['requested_products'].append(contamination)
                record['contamination_stage'] = contamination
            if controls.get('native_stellar_apertures') is not None:
                from .apertures import validate_recovered_apertures,validate_recovered_catalogue_controls
                settings=controls['native_stellar_apertures']
                required={'controls','snapshots','memory_gib','maximum_particles','maximum_apertures','maximum_coarse_mass_fraction'}
                if not isinstance(settings,dict) or set(settings)-{'identity_graph','calibration','encounter_products'}!=required:
                    raise ValueError('Complete frozen native recovered stellar aperture settings required')
                control_path=bind(settings['controls']);inputs=[settings['controls']]
                options={key:copy.deepcopy(settings[key]) for key in required-{'controls'}}
                validate_recovered_apertures(options)
                if options['memory_gib']>spec['resources']['memory_gib']:raise MemoryError('Native stellar apertures exceed campaign memory')
                if not set(options['snapshots'])<=set(controls['snapshots']):raise ValueError('Native stellar apertures require actual recovered campaign epochs')
                if control_path.stat().st_size>min(1024**2,options['memory_gib']*1024**3/16):raise MemoryError('Native stellar controls exceed bounded intake')
                validate_recovered_catalogue_controls(json.loads(control_path.read_text()),options['memory_gib'])
                def native_input(reference,graph=False):
                    if isinstance(reference,dict) and set(reference)=={'path','sha256'}:
                        path=bind(reference);inputs.append(reference)
                        if graph:
                            if path.name!='report.json':raise ValueError('Native identity graph file binding must name its report.json')
                            index=path.parent/'index.sqlite';reference={'path':str(index),'sha256':file_sha256(index)}
                            bind(reference);inputs.append(reference)
                            return str(path.parent)
                        return {'input':str(path)}
                    if isinstance(reference,dict) and set(reference)=={'stage','artifact'} and reference['stage'] in by_id:
                        return copy.deepcopy(reference)
                    raise ValueError('Native aperture selection must bind a file or existing producer artifact')
                for name in ('identity_graph','calibration'):
                    if settings.get(name) is not None:options[name]=native_input(settings[name],graph=name=='identity_graph')
                if settings.get('encounter_products') is not None:
                    selections=settings['encounter_products']
                    if not isinstance(selections,list) or not selections:raise ValueError('Explicit native encounter product bindings required')
                    options['encounter_products']=[];seen_epochs=set()
                    for selected in selections:
                        if not isinstance(selected,dict) or set(selected)!={'snapshot','encounter_product','stream_history'}:
                            raise ValueError('Complete native encounter aperture producer binding required')
                        epoch=selected['snapshot']
                        if type(epoch) is not int or epoch in seen_epochs or epoch not in options['snapshots']:
                            raise ValueError('Distinct recovered native encounter aperture epochs required')
                        seen_epochs.add(epoch);options['encounter_products'].append({'snapshot':epoch,
                            **{name:native_input(selected[name]) for name in ('encounter_product','stream_history')}})
                name='stellar_apertures_'+branch_id
                stages.append({'stage_id':name,'kind':'native_recovered_apertures','parameters':{**options,
                    'run_dir':{'stage':evolution,'artifact':'run'},'recovery_path':{'stage':recovery,'artifact':'recovery.json'},
                    'controls_path':{'input':str(control_path)},'controls_sha256':settings['controls']['sha256']},
                    'input_files':inputs,'outputs':{name:'contamination-history.json'}})
                spec['requested_products'].append(name);record.setdefault('aperture_contamination_stages',[]).append(name)
                record['native_stellar_aperture_stage']=name
            if controls.get('aperture_products') is not None:
                from .apertures import validate_aperture_contamination
                selections=controls['aperture_products']
                if not isinstance(selections,list) or not selections:raise ValueError('Explicit measured science aperture products required')
                record.setdefault('aperture_contamination_stages',[]);seen_apertures=set()
                for selection in selections:
                    if not isinstance(selection,dict) or set(selection)-{'producer','stream_history','snapshot'}!={'id','product','maximum_apertures','maximum_coarse_mass_fraction'}:
                        raise ValueError('Complete bounded science aperture contamination selection required')
                    selection_id=safe_id(selection['id'])
                    if selection_id in seen_apertures:raise ValueError('Distinct aperture product selections required')
                    seen_apertures.add(selection_id);product=selection['product'];inputs=[]
                    if isinstance(product,dict) and set(product)=={'path','sha256'}:
                        aperture_input={'input':str(bind(product))};inputs=[product]
                    elif isinstance(product,dict) and set(product)=={'stage','artifact'} and product['stage'] in by_id:
                        aperture_input=copy.deepcopy(product)
                    else:raise ValueError('Measured aperture input must be a bound file or declared existing producer artifact')
                    name='aperture_'+branch_id+'_'+selection_id
                    params={'memory_gib':controls['memory_gib'],'maximum_apertures':selection['maximum_apertures'],
                        'maximum_coarse_mass_fraction':selection['maximum_coarse_mass_fraction']}
                    validate_aperture_contamination(params)
                    if 'producer' in selection:
                        producer=name+'_selection'
                        producer_params={'run_dir':{'stage':evolution,'artifact':'run'},
                            'memory_gib':controls['memory_gib'],'maximum_apertures':selection['maximum_apertures']}
                        if selection['producer']=='native_stellar_catalogue':
                            if {'snapshot','stream_history'}&set(selection):raise ValueError('Stellar apertures derive the actual saved epoch from the catalogue')
                            producer_kind='native_catalogue_apertures';producer_params['catalogue_product']=aperture_input
                        elif selection['producer']=='native_encounter_catalogue':
                            if not {'snapshot','stream_history'}<=set(selection):raise ValueError('Native encounter apertures require converted stellar history and exact saved epoch')
                            history=selection['stream_history']
                            if isinstance(history,dict) and set(history)=={'path','sha256'}:
                                history_input={'input':str(bind(history))};inputs.append(history)
                            elif isinstance(history,dict) and set(history)=={'stage','artifact'} and history['stage'] in by_id:
                                history_input=copy.deepcopy(history)
                            else:raise ValueError('Native encounter stellar history must bind an actual file or existing producer artifact')
                            producer_kind='native_encounter_apertures';producer_params.update(encounter_product=aperture_input,
                                stream_history=history_input,snapshot=selection['snapshot'])
                            from .apertures import validate_encounter_apertures
                            validate_encounter_apertures(producer_params)
                            if selection['snapshot'] not in controls['snapshots']:raise ValueError('Encounter aperture saved epoch is outside the recovered campaign')
                        else:raise ValueError('Supported native aperture measurement producer required')
                        stages.append({'stage_id':producer,'kind':producer_kind,'parameters':producer_params,
                            'input_files':inputs,'outputs':{producer:'apertures.json'}})
                        aperture_input={'stage':producer,'artifact':'apertures.json'};inputs=[]
                    elif {'snapshot','stream_history'}&set(selection):raise ValueError('Extra history/snapshot settings require the native encounter aperture producer')
                    stages.append({'stage_id':name,'kind':'science_aperture_contamination','parameters':{**params,
                        'run_dir':{'stage':evolution,'artifact':'run'},'aperture_product':aperture_input},
                        'input_files':inputs,'outputs':{name:'contamination.json'}})
                    spec['requested_products'].append(name);record['aperture_contamination_stages'].append(name)
            if controls.get('hydro_history') is not None:
                from .stellar_returns import validate_stellar_history_stage
                settings=controls['hydro_history']
                if not isinstance(settings,dict) or set(settings)!={'snapshots','memory_gib','tolerances'}:
                    raise ValueError('Complete frozen native hydro/source history settings required')
                validate_stellar_history_stage(settings)
                if not set(settings['snapshots'])<=set(controls['snapshots']):raise ValueError('Native source audit requires actual recovered saved epochs')
                if settings['memory_gib']>spec['resources']['memory_gib']:raise MemoryError('Native source history exceeds campaign allocation')
                name='hydro_history_'+branch_id
                stages.append({'stage_id':name,'kind':'native_stellar_history','parameters':{
                    **copy.deepcopy(settings),'run_dir':{'stage':evolution,'artifact':'run'}},
                    'depends_on':[recovery]+([contamination] if contamination else [])+record.get('aperture_contamination_stages',[]),
                    'outputs':{name:'hydro-history.json'}})
                spec['requested_products'].append(name);record['hydro_history_stage']=name
            if controls.get('stream_measurements') is not None:
                from .stream.measurements import validate_catalogue_stream_measurements,validate_stream_measurement_template
                settings=controls['stream_measurements']
                if not isinstance(settings,dict) or set(settings)!={'template','snapshots','memory_gib','maximum_particles','maximum_candidates'}:
                    raise ValueError('Complete frozen native stream measurement settings required')
                if not record.get('native_stellar_aperture_stage'):raise ValueError('Native stream measurements require the recovered stellar catalogue producer')
                validate_catalogue_stream_measurements(settings)
                selected=settings['snapshots']
                if not isinstance(selected,list) or not selected or any(type(x) is not int for x in selected) or len(set(selected))!=len(selected) or not set(selected)<=set(controls['native_stellar_apertures']['snapshots']):
                    raise ValueError('Stream measurements require distinct actual recovered stellar catalogue epochs')
                if settings['memory_gib']>spec['resources']['memory_gib']:raise MemoryError('Native stream measurements exceed campaign memory')
                template=bind(settings['template'])
                if template.stat().st_size>min(16*1024**2,settings['memory_gib']*1024**3/32):raise MemoryError('Native stream template exceeds bounded intake')
                validate_stream_measurement_template(json.loads(template.read_text()))
                name='stream_measurements_'+branch_id
                stages.append({'stage_id':name,'kind':'native_recovered_stream_measurements','parameters':{
                    **{key:copy.deepcopy(settings[key]) for key in ('snapshots','memory_gib','maximum_particles','maximum_candidates')},
                    'run_dir':{'stage':evolution,'artifact':'run'},
                    'aperture_history':{'stage':record['native_stellar_aperture_stage'],'artifact':'contamination-history.json'},
                    **({'hydro_history':{'stage':record['hydro_history_stage'],'artifact':'hydro-history.json'}} if record.get('hydro_history_stage') else {}),
                    'template_path':{'input':str(template)},'template_sha256':settings['template']['sha256']},
                    'depends_on':([record['hydro_history_stage']] if record.get('hydro_history_stage') else [recovery]),
                    'input_files':[settings['template']],'outputs':{name:'stream-history.json'}})
                spec['requested_products'].append(name);record['stream_measurement_stage']=name
            if controls.get('native_encounters') is not None:
                from .stream.native_encounters import validate_recovered_encounter_template
                settings=controls['native_encounters']
                if not isinstance(settings,dict) or set(settings)!={'template','memory_gib'}:raise ValueError('Complete recovered native encounter settings required')
                _positive(settings['memory_gib'],'native encounter workspace')
                if settings['memory_gib']>spec['resources']['memory_gib']:raise MemoryError('Recovered encounters exceed campaign memory')
                if not record.get('stream_measurement_stage'):raise ValueError('Recovered encounters require the actual recovered stream measurement producer')
                template=bind(settings['template'])
                if template.stat().st_size*16>settings['memory_gib']*1024**3/16:raise MemoryError('Recovered encounter template exceeds intake allocation')
                control=validate_recovered_encounter_template(json.loads(template.read_text()))
                if not set(control['snapshots'])<=set(controls['stream_measurements']['snapshots']):raise ValueError('Encounter epochs must come from actual recovered stream measurement requests')
                name='encounters_'+branch_id
                stages.append({'stage_id':name,'kind':'recovered_native_encounters','parameters':{
                    'run_dir':{'stage':evolution,'artifact':'run'},'native_stage_receipt':{'stage':evolution,'artifact':'stage-receipt.json'},
                    'stream_measurement_history':{'stage':record['stream_measurement_stage'],'artifact':'stream-history.json'},
                    'template_path':{'input':str(template)},'template_sha256':settings['template']['sha256'],'memory_gib':settings['memory_gib']},
                    'input_files':[settings['template']],'outputs':{name:'encounter-history.json'}})
                spec['requested_products'].append(name);record['native_encounter_stage']=name
            if controls.get('analysis') is not None:
                analysis=_analysis_stage(controls['analysis'],identifier=branch_id,
                    evolution=evolution,recovery=recovery,contamination=contamination)
                analysis['depends_on']+=record.get('aperture_contamination_stages',[])
                if record.get('hydro_history_stage'):analysis['depends_on'].append(record['hydro_history_stage'])
                if record.get('stream_measurement_stage'):analysis['depends_on'].append(record['stream_measurement_stage'])
                if record.get('native_encounter_stage'):analysis['depends_on'].append(record['native_encounter_stage'])
                stages.append(analysis);spec['requested_products'].append(analysis['stage_id'])
                record['analysis_stage']=analysis['stage_id']
            if controls.get('structure') is not None:
                from .halo_campaign import validate_structure_template
                settings=controls['structure']
                if not isinstance(settings,dict) or set(settings)!={'template','snapshots','export_figures'}:
                    raise ValueError('Recovered structure needs explicit template, snapshots and figure choice')
                selected=settings['snapshots']
                if (not isinstance(selected,list) or not selected or any(type(s) is not int for s in selected)
                        or len(set(selected))!=len(selected) or not set(selected)<=set(controls['snapshots'])
                        or type(settings['export_figures']) is not bool):
                    raise ValueError('Recovered structure epochs must be distinct exact recovery snapshots')
                template_path=bind(settings['template'])
                if template_path.stat().st_size>min(16*1024**2,controls['memory_gib']*1024**3/16):
                    raise MemoryError('Recovered structure template exceeds bounded preparation intake')
                template=validate_structure_template(json.loads(template_path.read_text()),memory_gib=controls['memory_gib'])
                record['structure_stages']=[]
                for snapshot in selected:
                    name='structure_'+branch_id+'_snapshot'+str(snapshot)
                    stages.append({'stage_id':name,'kind':'zoom_recovered_structure',
                        'depends_on':[recovery]+([contamination] if contamination else [])+record.get('aperture_contamination_stages',[]),
                        'parameters':{'run_dir':{'stage':evolution,'artifact':'run'},
                            'recovery_path':{'stage':recovery,'artifact':'recovery.json'},
                            'template_path':{'input':str(template_path)},'template_sha256':pins[str(template_path)],
                            'snapshot':snapshot,'memory_gib':template['memory_gib'],'export_figures':settings['export_figures']},
                        'input_files':[settings['template']],'outputs':{name:'measurement'}})
                    spec['requested_products'].append(name);record['structure_stages'].append(name)
            records.append(record)
    spec['resources']['maximum_stage_count'] = len(stages)
    copy_bytes=sum(s['parameters'].get('copy_run_max_bytes',0) for s in stages if s['kind']=='public_command')
    retained_bytes=None
    if copy_bytes:
        import shutil
        if manifest.get('resource_preflight') is not None:
            estimate=json.loads(bind(manifest['resource_preflight']).read_text())
            retained_bytes=estimate['campaign_storage_estimate_bytes']
        elif manifest.get('storage',{}).get('conservative_combined_estimate_bytes') is not None:
            retained_bytes=manifest['storage']['conservative_combined_estimate_bytes']
        else:
            retained_bytes=request.get('retained_campaign_disk_bytes')
        if type(retained_bytes) is not int or retained_bytes<0:
            raise ValueError('Analysis needs the original retained campaign disk estimate before adding private copies')
        probe=target.parent
        while not probe.exists():probe=probe.parent
        required_bytes=retained_bytes+copy_bytes
        deferred=request.get('defer_storage_to_production_admission',False)
        if type(deferred) is not bool:raise ValueError('Explicit production storage-preparation selection required')
        if required_bytes>spec['resources']['disk_gib']*1024**3:
            raise MemoryError('Retained private analysis copies exceed declared/available campaign disk')
        if deferred:
            admission=[s for s in stages if s['kind']=='flagship_production_admission']
            if manifest.get('schema')!='flagship-production-campaign-v1' or len(admission)!=1:
                raise ValueError('Deferred storage preparation requires the original production runtime allocation gate')
            # The same available-disk requirement moves to the first runtime
            # admission stage, before any native generation or private copy.
            required_gib=math.ceil(required_bytes/1024**3)
            contract=admission[0]['parameters']['request']
            contract['storage_gib']=max(contract['storage_gib'],required_gib)
            contract['requested_resources']['disk_gib']=max(contract['requested_resources']['disk_gib'],required_gib)
        elif required_bytes>shutil.disk_usage(probe).free:
            raise MemoryError('Retained private analysis copies exceed declared/available campaign disk')
    compile_requested_product_dag(spec, request_base=target)
    for source in (Path(__file__), Path(__file__).with_name('workflow.py'), *sorted((Path(__file__).parent/'research').glob('*.py')),Path(__file__).with_name('analysis.py'),Path(__file__).with_name('halo_campaign.py'),Path(__file__).with_name('halo_products.py'),
                   Path(__file__).with_name('apertures.py'),
                   (Path(__file__).parent/'stream'/'measurements.py'),(Path(__file__).parent/'stream'/'native_encounters.py'),(Path(__file__).parent/'stream'/'perturbers.py'),Path(__file__).with_name('streams.py'),Path(__file__).resolve().parents[1]/'commands.py'):
        pins[str(source.resolve())] = file_sha256(source)
    target.mkdir(parents=True); path = target / 'workflow.json'; atomic_json(path, spec)
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'commands.py'), 'workflow', '--campaign', str(path), '--output', str(target / 'execution')]
    result = {'schema': 'flagship-history-campaign-v1', 'status': 'RECIPES_PREPARED', 'cases': records,
        'source_hashes': pins, 'workflow': {'path': str(path), 'sha256': file_sha256(path)}, 'stage_count': len(stages),
        'commands': {'check': command + ['--dry-run'], 'run': command, 'resume': command + ['--resume']},
        'storage_preparation':{'retained_bytes':retained_bytes,'private_copy_max_bytes':copy_bytes,'available_storage_gate':'ORIGINAL_PRODUCTION_RUNTIME_ADMISSION' if request.get('defer_storage_to_production_admission') else 'PREPARATION_DECLARED_AND_AVAILABLE'},
        'launch_ready': False, 'science_acceptance': 'NOT_ACCEPTED',
        'remaining_requirements': manifest['remaining_requirements'] + ['Measured force/observable controls and downstream display remain required; native aperture measurement support depends on actual ancestry, classifier calibration and encounter products'],
        'halo_contamination_configured':all('contamination_stage' in row for row in records),
        'saved_run_analysis_configured':all('analysis_stage' in row for row in records),
        'recovered_structure_configured':all(row.get('structure_stages') for row in records),
        'native_science_aperture_history_configured':all('native_stellar_aperture_stage' in row for row in records),
        'native_stream_measurements_configured':all('stream_measurement_stage' in row for row in records),
        'native_encounters_configured':all('native_encounter_stage' in row for row in records),
        'private_analysis_copy_disk_bound_bytes':copy_bytes,
        'retained_campaign_disk_estimate_bytes':retained_bytes,
        'analysis_policy':'Public analysis on a private owned run copy after recovery and configured contamination; unavailable selected products fail explicitly. Runtime copy checks declared/free disk. Source run remains immutable; full 102-variant/target-specific measurement routes remain separate requirements.'}
    for source, digest in pins.items():
        if file_sha256(source) != digest:
            raise ValueError('History preparation input changed: ' + source)
    atomic_json(target / 'manifest.json', result)
    return result
