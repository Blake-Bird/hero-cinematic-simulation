"""Bind selected flagship profiles to the existing native workflow runners."""
from pathlib import Path
import configparser
import copy
import json
import math
import sys
from .foundation import atomic_json, file_sha256
from .flagship_profiles import profile_input_bindings, case_parent_bindings


def _matched_generation_parameters(stage, pins):
    """Compare independently staged inputs by content, retaining both pins."""
    parameters=copy.deepcopy(stage['parameters'])
    arguments=parameters['arguments']
    reference=arguments[0]
    if not isinstance(reference,dict) or set(reference)!={'input'}:
        raise ValueError('Matched generation requires an explicit configuration input')
    path=Path(reference['input']).resolve()
    bindings={str(Path(item['path']).resolve()):item['sha256'] for item in stage['input_files']}
    if str(path) not in bindings or file_sha256(path)!=bindings[str(path)]:
        raise ValueError('Matched generator configuration changed')
    pins[str(path)]=bindings[str(path)]
    cfg=configparser.ConfigParser(interpolation=None);cfg.optionxform=str
    with path.open() as stream:cfg.read_file(stream)
    settings={section:dict(cfg.items(section)) for section in cfg.sections()}
    # Restaging writes a private copy of the same physical transfer table.
    # Bind its bytes; every other native option, including phase paths, must
    # remain identical. No general path stripping or physics normalization.
    transfer=settings.get('cosmology',{}).get('transfer_file')
    if transfer is not None:
        table=Path(transfer)
        if not table.is_absolute():table=path.parent/table
        table=table.resolve();digest=file_sha256(table)
        if str(table) not in bindings or bindings[str(table)]!=digest:
            raise ValueError('Matched generator transfer table changed or is unbound')
        pins[str(table)]=digest
        settings['cosmology']['transfer_file']={'sha256':digest}
    arguments[0]={'configuration':settings}
    return parameters


def prepare_native_campaign(request, destination):
    """Prepare generation/evolution recipes; admission gates run before launch."""
    from .workflow import safe_id, compile_requested_product_dag
    from core.research.capabilities import validate_native_answers
    from commands import redshift_schedule
    if request.get('schema') != 'flagship-native-campaign-request-v1':
        raise ValueError('Explicit flagship native campaign request required')
    target=Path(destination).resolve()
    if target.exists():raise FileExistsError('Preserve existing native campaign preparation')
    pins={}
    def bind(reference):
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:
            raise ValueError('Exact native campaign input binding required')
        path=Path(reference['path']).resolve()
        if not path.is_file() or file_sha256(path)!=reference['sha256']:
            raise ValueError('Native campaign input changed: '+str(path))
        pins[str(path)]=reference['sha256'];return path
    profile=json.loads(bind(request['profile']).read_text())
    values,profile_pins=profile_input_bindings(profile);pins.update(profile_pins)
    branch=profile.get('branch',profile['name'])
    if branch not in {'miniature','local-preparation','sidm'}:
        raise ValueError('This adapter requires retained generated geometries; production bridge handoff remains separate')
    resource=profile['resources'];controls=request['execution']
    for name in ('generation_wall_hours','evolution_wall_hours','audit_memory_gib'):
        value=controls[name]
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
            raise ValueError('Explicit positive native campaign execution bound required')
    if controls['audit_memory_gib']>resource['memory_gib']:raise MemoryError('Native audit exceeds profile allocation')
    binary=bind(request['generator']['binary']);validation=bind(request['generator']['native_validation'])
    from .zoom import validate_native_music_regression
    validate_native_music_regression(binary,validation)
    flow_template=None
    if request.get('flow_preparation') is not None:
        flow_template=json.loads(bind(request['flow_preparation']).read_text())
        from core.music.staging import generated_flow_bindings
        pins.update(generated_flow_bindings(flow_template))
        if not isinstance(request.get('case_staging'),dict) or set(request['case_staging'])!={'source_archive_max_mib','bounded_source_settings'}:
            raise ValueError('Runtime flow preparation requires explicit source-retaining bounded case staging')
    epoch_bundle=request.get('start_epoch_bundle')
    epoch_info=values['cosmology'] if epoch_bundle is None else json.loads(bind(epoch_bundle).read_text())
    if epoch_bundle is not None and flow_template is None:raise ValueError('Epoch restaging requires the original runtime mixed-fluid flow preparation')
    zoom_schedule=request.get('zoom_schedule',request['schedule'])
    schedule=redshift_schedule(str(zoom_schedule),float(epoch_info['parameters']['z_start']))
    if request['schedule']!=values['parent_answers']['schedule']:
        raise ValueError('Native campaign must retain the parent output schedule')
    if len(schedule)!=resource['saved_epochs']:raise ValueError('Saved epoch budget differs from the actual requested schedule')
    choices=request['numerical_choices']
    if choices.get('softening_policy') not in {'fixed_fraction_of_fine_cell','fixed_reference'} or type(choices.get('softening_parent_level')) is not int:
        raise ValueError('Record explicit reference softening and refinement policy')
    if type(choices.get('pm_grid')) is not int or choices['pm_grid']<2:raise ValueError('Explicit native PM grid required')
    target.mkdir(parents=True);stages=[];products=[];cases=[]
    try:
        from .flagship_products import preflight_flagship_profile
        budget=preflight_flagship_profile(profile,target/'profile-resources.json',output_directory=target.parent)
        pins.update(budget['source_hashes'])
        for case in profile['zoom_cases']:
            identifier=safe_id(case['id']);generation=json.loads(bind(case['generation']).read_text())
            parent,parent_pins=case_parent_bindings(case,generation);pins.update(parent_pins)
            configuration=bind(generation['source_configuration']);region=bind(generation['traced_region'])
            native=configparser.ConfigParser(interpolation=None);native.optionxform=str;native.read(configuration)
            if native.getint('setup','levelmax')!=case['level']:
                raise ValueError('Selected refinement differs from its actual generator configuration')
            parent_level=native.getint('setup','levelmin')
            if parent_level!=profile['phases']['parent_level']:
                raise ValueError('Retained campaign geometry requires the actual original parent grid')
            if native.getboolean('setup','baryons',fallback=False) and native.getboolean('setup','use_2LPT',fallback=False) and flow_template is None:
                raise ValueError('Original mixed-fluid campaign requires a bound runtime flow-preparation recipe')
            restaging=None
            if flow_template is not None:
                from core.music.staging import restage_campaign_case
                restaging=restage_campaign_case(configuration,target/(identifier+'.staged'),**request['case_staging'],**({} if epoch_bundle is None else {'linear_bundle':epoch_bundle}))
                pins.update(restaging['source_hashes'])
                for key in ('staging','configuration'):
                    pins[restaging[key]['path']]=restaging[key]['sha256']
                configuration=Path(restaging['configuration']['path'])
                native=configparser.ConfigParser(interpolation=None);native.optionxform=str;native.read(configuration)
            region_data=json.loads(region.read_text());parent_ic=Path(region_data['initial_ic']['path']).resolve()
            noise=[Path(p).resolve() for p in parent if Path(p).resolve()!=parent_ic]
            if len(noise)!=1:raise ValueError('Exactly one original parent noise input required')
            # Pin every external random-field and transfer-table input actually
            # referenced by this configuration; no old generated IC is reused.
            from .zoom import _music_noise_files
            for path in _music_noise_files(native,configuration).values():pins[str(path)]=file_sha256(path)
            transfer=Path(native['cosmology']['transfer_file'])
            if not transfer.is_absolute():transfer=configuration.parent/transfer
            transfer=transfer.resolve();pins[str(transfer)]=file_sha256(transfer)
            answers=copy.deepcopy(values['combined_physics_answers'])
            factor=1. if choices['softening_policy']=='fixed_reference' else 2.**(choices['softening_parent_level']-case['level'])
            answers.update(name=identifier+'-'+branch,cosmology=str(Path((profile['cosmology'] if epoch_bundle is None else epoch_bundle)['path']).resolve()),
                schedule=zoom_schedule,particles_per_dimension=2**case['level'],pm_grid=choices['pm_grid'],
                softening_kpc_h=answers['softening_kpc_h']*factor,
                softening_max_physical_kpc_h=answers['softening_max_physical_kpc_h']*factor,
                mpi_ranks=resource['mpi_ranks'],omp_threads=resource['omp_threads'],build_jobs=resource['build_jobs'],
                memory_limit_gb=resource['memory_gib'])
            config=validate_native_answers(answers,resource)
            native_engine_inputs=[]
            if config.get('halos',{}).get('native_mergertree',False):
                from .native_gadget_ancestry import recorded_link_sources
                pins.update(recorded_link_sources())
                native_engine_inputs.extend({'path':p,'sha256':h} for p,h in recorded_link_sources().items())
            if config.get('sidm',{}).get('enabled',False):
                native_sidm_source=Path(__file__).with_name('software')/'gadget4/src/sidm/research_distributed_step.cc'
                pins[str(native_sidm_source.resolve())]=file_sha256(native_sidm_source)
                native_engine_inputs.append({'path':str(native_sidm_source.resolve()),'sha256':file_sha256(native_sidm_source)})
            if config['simulation']['snapshot_redshift_list']!=schedule:raise ValueError('Native campaign schedule changed during validation')
            answer_path=target/(identifier+'.answers.json');atomic_json(answer_path,answers)
            answer_ref={'path':str(answer_path),'sha256':file_sha256(answer_path)}
            gen='generate_'+identifier;evolve='evolve_'+identifier
            prepare=None;flow_ref=None
            if flow_template is not None:
                from core.music.staging import generated_flow_bindings
                flow=copy.deepcopy(flow_template);flow['answers']=answer_ref
                pins.update(generated_flow_bindings(flow,levels=range(parent_level,case['level']+1)))
                flow_path=target/(identifier+'.flow-request.json');atomic_json(flow_path,flow)
                flow_ref={'path':str(flow_path),'sha256':file_sha256(flow_path)};prepare='prepare_flow_'+identifier
            common_inputs=[{'path':p,'sha256':sha} for p,sha in pins.items()]
            ref=lambda path:{'input':str(path)}
            stages.append({'stage_id':gen,'kind':'public_command','parameters':{'command':'zoom',
                'wall_seconds':controls['generation_wall_hours']*3600,
                'arguments':[ref(configuration),'{generation}/generated','--binary',ref(binary),
                    '--native-validation',ref(validation),'--traced-region',ref(region),
                    '--parent-ic',ref(parent_ic),'--parent-noise',ref(noise[0]),
                    '--threads',str(resource['omp_threads']),'--audit-memory-gib',str(controls['audit_memory_gib']),
                    '--native-memory-gib',str(resource['memory_gib']),'--native-disk-gib',str(resource['disk_gib']),
                    '--native-wall-hours',str(controls['generation_wall_hours'])]},
                'input_files':common_inputs,'outputs':{gen:'generated'}})
            if prepare is not None:
                stages.append({'stage_id':prepare,'kind':'public_command','depends_on':[gen],
                    'parameters':{'command':'prepare-generated-flow','wall_seconds':controls['generation_wall_hours']*3600,
                        'arguments':['--request',ref(flow_ref['path']),'--generation-report',{'stage':gen,'artifact':'generated/generation.json'},
                            '--output','{generation}/prepared']},'input_files':[flow_ref,answer_ref], 'outputs':{prepare:'prepared'}})
            stages.append({'stage_id':evolve,'kind':'public_command','depends_on':[gen,prepare] if prepare is not None else [gen],
                'parameters':{'command':'gadget','wall_seconds':controls['evolution_wall_hours']*3600,
                    'arguments':['--answers',{'stage':prepare,'artifact':'prepared/answers.json'} if prepare is not None else ref(answer_path),'--existing-ic',{'stage':gen,'artifact':'generated/ics_gadget.hdf5'},
                        '--native-generation-report',{'stage':gen,'artifact':'generated/generation.json'},
                        '--run-dir','{generation}/run']},'input_files':[answer_ref,*native_engine_inputs], 'outputs':{evolve:'run'}})
            products.append(evolve);cases.append({'id':identifier,'host_row':case['host_row'],'level':case['level'],
                'answers':answer_ref,'generation_stage':gen,'evolution_stage':evolve,'flow_preparation_stage':prepare,
                'runtime_flow_request':flow_ref,'case_restaging':restaging})
        resources={key:resource[key] for key in ('mpi_ranks','omp_threads','memory_gib','disk_gib')}
        resources['maximum_stage_count']=len(stages)
        from core.research.checkpoints import evolution_checkpoint_policy
        for stage in stages:
            if stage['parameters'].get('command')=='gadget':stage['checkpoint_resume']=evolution_checkpoint_policy()
        helper=Path(__file__).resolve().parent/'research/checkpoints.py';pins[str(helper.resolve())]=file_sha256(helper)
        spec={'schema':'research-workflow-v1','resources':resources,
            'followup_limits':{'maximum_children':0,'maximum_depth':0},'physics_spec':{'physics':values['combined_physics_answers']['physics']},
            'capabilities':[],'requested_products':products,'stages':stages}
        plan=compile_requested_product_dag(spec,request_base=target)
        path=target/'workflow.json';atomic_json(path,spec)
        root=Path(__file__).resolve().parents[1]
        for source in (
            Path(__file__),
            root/'core/flagship_profiles.py',
            root/'core/music/staging.py',
            root/'core/zoom.py',
            root/'core/workflow.py',
            root/'core/research/generator_controls.py',
            root/'commands.py',
        ):
            pins[str(source)]=file_sha256(source)
        for source,digest in pins.items():
            if file_sha256(source)!=digest:raise ValueError('Campaign input changed during preparation')
        manifest={'schema':'flagship-native-campaign-v1','status':'RECIPES_PREPARED','branch':branch,'cases':cases,
            'source_hashes':pins,'workflow':{'path':str(path),'sha256':file_sha256(path)},
            'commands':{'check':[sys.executable,str(root/'commands.py'),'workflow','--campaign',str(path),'--output',str(target/'execution'),'--dry-run'],
                'run':[sys.executable,str(root/'commands.py'),'workflow','--campaign',str(path),'--output',str(target/'execution')],
                'resume':[sys.executable,str(root/'commands.py'),'workflow','--campaign',str(path),'--output',str(target/'execution'),'--resume']},
            'resource_preflight':{'path':str(target/'profile-resources.json'),'sha256':file_sha256(target/'profile-resources.json')},
            'stage_count':len(plan['stages']),'launch_ready':False,'science_acceptance':'NOT_ACCEPTED',
            'scope':'Selected native generation/evolution only; current admission gates must pass; analysis/display are separate campaign connections',
            'remaining_requirements':['Current native generator regression and sealed staging','Native IC/handoff acceptance','Recovery, diagnostics, analysis and display campaign connections']}
        atomic_json(target/'manifest.json',manifest);return manifest
    except Exception as error:
        atomic_json(target/'failure.json',{'status':'FAIL','error':str(error),'completed_cases':cases});raise


def prepare_matched_campaign(request, destination):
    """Share one generated IC artifact between matched CDM/SIDM evolutions."""
    from .workflow import compile_requested_product_dag
    if request.get('schema')!='flagship-matched-native-campaign-request-v1':
        raise ValueError('Explicit matched native campaign request required')
    target=Path(destination).resolve()
    if target.exists():raise FileExistsError('Preserve existing matched native campaign')
    pins={};loaded={};specs={}
    for branch in ('cdm','sidm'):
        ref=request[branch];path=Path(ref['path']).resolve()
        if file_sha256(path)!=ref['sha256']:raise ValueError('Matched native branch manifest changed')
        manifest=json.loads(path.read_text())
        if manifest.get('schema')!='flagship-native-campaign-v1' or manifest.get('status')!='RECIPES_PREPARED':
            raise ValueError('Prepared actual native branch required')
        for source,digest in manifest['source_hashes'].items():
            if file_sha256(source)!=digest:raise ValueError('Matched native branch source changed: '+source)
            if source in pins and pins[source]!=digest:raise ValueError('Matched branches disagree about source identity')
            pins[source]=digest
        pins[str(path)]=ref['sha256'];workflow=manifest['workflow']
        if file_sha256(workflow['path'])!=workflow['sha256']:raise ValueError('Matched branch workflow changed')
        pins[workflow['path']]=workflow['sha256'];loaded[branch]=manifest;specs[branch]=json.loads(Path(workflow['path']).read_text())
    if loaded['sidm']['branch']!='sidm' or loaded['cdm']['branch']=='sidm':
        raise ValueError('Matched campaign requires actual distinct CDM and SIDM branches')
    if specs['cdm']['resources']!=specs['sidm']['resources']:raise ValueError('Matched native allocation must be identical')
    import shutil
    storage=0;margin=0
    for manifest in loaded.values():
        reference=manifest['resource_preflight'];path=Path(reference['path'])
        if file_sha256(path)!=reference['sha256']:raise ValueError('Matched profile resource report changed')
        pins[str(path)]=reference['sha256'];budget=json.loads(path.read_text())
        storage+=budget['campaign_storage_estimate_bytes']
        margin=max(margin,budget['resources']['disk_margin_gib']*1024**3)
    probe=target.parent
    while not probe.exists():probe=probe.parent
    available=shutil.disk_usage(probe).free
    usable=min(specs['cdm']['resources']['disk_gib']*1024**3,available-margin)
    if storage>usable:raise MemoryError('Combined matched branches exceed fresh available/declared storage')

    cases={branch:{c['id']:c for c in manifest['cases']} for branch,manifest in loaded.items()}
    if set(cases['cdm'])!=set(cases['sidm']):raise ValueError('Matched native target/resolution cases differ')
    stages=[];products=[];records=[]
    branch_stages={b:{s['stage_id']:s for s in spec['stages']} for b,spec in specs.items()}
    for identifier,left in cases['cdm'].items():
        right=cases['sidm'][identifier]
        if (left['host_row'],left['level'])!=(right['host_row'],right['level']):raise ValueError('Matched host or refinement changed')
        answers={}
        for b,c in (('cdm',left),('sidm',right)):
            ref=c['answers']
            if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Matched native answers changed')
            pins[ref['path']]=ref['sha256'];answers[b]=json.loads(Path(ref['path']).read_text())
        def common(a):return {k:v for k,v in a.items() if k not in ('sidm','name')}
        if common(answers['cdm'])!=common(answers['sidm']) or answers['cdm'].get('sidm',{}).get('enabled',False) or answers['sidm'].get('sidm',{}).get('enabled') is not True:
            raise ValueError('Matched native answers change more than the explicit SIDM branch')
        gen=copy.deepcopy(branch_stages['cdm'][left['generation_stage']]);other=branch_stages['sidm'][right['generation_stage']]
        if _matched_generation_parameters(gen,pins)!=_matched_generation_parameters(other,pins):
            raise ValueError('Matched native branches must generate exactly the same IC configuration')
        inputs={item['path']:item['sha256'] for item in gen['input_files']}
        for item in other['input_files']:
            if item['path'] in inputs and inputs[item['path']]!=item['sha256']:raise ValueError('Matched generator inputs disagree')
            inputs[item['path']]=item['sha256']
        gen['input_files']=[{'path':p,'sha256':h} for p,h in inputs.items()];stages.append(gen)
        row={'id':identifier,'host_row':left['host_row'],'level':left['level'],'shared_generation_stage':gen['stage_id']}
        for b,c in (('cdm',left),('sidm',right)):
            stage=copy.deepcopy(branch_stages[b][c['evolution_stage']]);old=stage['stage_id'];new=b+'_'+old
            stage['stage_id']=new;stage['depends_on']=[gen['stage_id']]
            preparation=c.get('flow_preparation_stage')
            if preparation is not None:
                prepared=copy.deepcopy(branch_stages[b][preparation]);prepared_name=b+'_'+preparation
                prepared['stage_id']=prepared_name;prepared['depends_on']=[gen['stage_id']];prepared['outputs']={prepared_name:'prepared'}
                for argument in prepared['parameters']['arguments']:
                    if isinstance(argument,dict) and argument.get('stage')==c['generation_stage']:argument['stage']=gen['stage_id']
                stages.append(prepared);stage['depends_on'].append(prepared_name)
                for argument in stage['parameters']['arguments']:
                    if isinstance(argument,dict) and argument.get('stage')==preparation:argument['stage']=prepared_name
                row[b+'_flow_preparation_stage']=prepared_name
            # Both branch argument vectors point to this very same immutable
            # generation, including the actual generation receipt.
            for argument in stage['parameters']['arguments']:
                if isinstance(argument,dict) and argument.get('stage')==c['generation_stage']:argument['stage']=gen['stage_id']
            stage['outputs']={new:'run'};stages.append(stage);products.append(new);row[b+'_evolution_stage']=new
        records.append(row)
    spec=copy.deepcopy(specs['cdm']);spec.update(stages=stages,requested_products=products)
    spec['resources']['maximum_stage_count']=len(stages)
    compile_requested_product_dag(spec,request_base=target)
    target.mkdir(parents=True);path=target/'workflow.json';atomic_json(path,spec)
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,str(root/'commands.py'),'workflow','--campaign',str(path),'--output',str(target/'execution')]
    manifest={'schema':'flagship-matched-native-campaign-v1','status':'RECIPES_PREPARED','cases':records,
        'workflow':{'path':str(path),'sha256':file_sha256(path)},'source_hashes':pins,'stage_count':len(stages),
        'commands':{'check':command+['--dry-run'],'run':command,'resume':command+['--resume']},
        'matching':'One shared generated IC and generation receipt per actual target/refinement; common baryons, phases, epochs and numerical controls; only selected SIDM settings differ',
        'storage':{'conservative_combined_estimate_bytes':storage,'fresh_available_bytes':available,'usable_budget_bytes':usable,'policy':'Sequential stages share generation; sum of separate branch estimates conservatively retains duplicate generation storage'},
        'launch_ready':False,'science_acceptance':'NOT_ACCEPTED','remaining_requirements':loaded['sidm']['remaining_requirements']}
    for source,digest in pins.items():
        if file_sha256(source)!=digest:raise ValueError('Matched input changed during preparation')
    atomic_json(target/'manifest.json',manifest);return manifest
