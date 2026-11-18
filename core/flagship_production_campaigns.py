"""Production bridge recipes delegated to the existing native workflow engine."""
from pathlib import Path
import configparser
import json
import math
import shutil
import sys
from .foundation import atomic_json, file_sha256


def _binding(reference):
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:
        raise ValueError('Exact production campaign asset binding required')
    path=Path(reference['path']).resolve()
    if not path.is_file() or file_sha256(path)!=reference['sha256']:
        raise ValueError('Production campaign input changed: '+str(path))
    return path


def validate_production_admission_contract(parameters):
    request=parameters['request']
    if request.get('schema')!='flagship-production-admission-v1':raise ValueError('Explicit production admission contract required')
    for key in ('generation_memory_gib','storage_gib'):
        value=request[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive production admission limit required')
    if type(request['generation_threads']) is not int or request['generation_threads']<1:raise ValueError('Exact generation thread allocation required')
    expected=request['requested_resources']
    for key in ('memory_gib','disk_gib','mpi_ranks','omp_threads','cpu_slots'):
        value=expected[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive declared production resource required')
        if key in ('mpi_ranks','omp_threads','cpu_slots') and type(value) is not int:raise ValueError('Exact production rank/thread/CPU control required')
    if expected['mpi_ranks']*expected['omp_threads']>expected['cpu_slots']:raise ValueError('Declared production parallelism exceeds CPU allocation')
    if not isinstance(request['staging_records'],list) or not request['staging_records']:raise ValueError('Production admission must bind actual staging')
    for ref in [*request['staging_records'],*([request['allocation']] if request.get('allocation') is not None else [])]:
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('Exact production admission input binding required')


def validate_production_admission(parameters, *, output_parent=None):
    """Allocation/intake gate before any native generation; never science acceptance."""
    from core.resources import memory_limit_gb, cpu_count
    from .zoom import verify_staged_zoom_inputs
    validate_production_admission_contract(parameters)
    request=parameters['request']
    if request.get('allocation') is None:
        raise MemoryError('Production hardware remains unallocated; prepared requirements do not authorize generation')
    path=_binding(request['allocation']);allocation=json.loads(path.read_text())
    if allocation.get('schema')!='flagship-execution-allocation-v1' or allocation.get('status')!='ALLOCATED':
        raise ValueError('An explicit allocated execution resource record is required')
    source=_binding(allocation['source'])
    expected=request['requested_resources'];actual=allocation['resources']
    for key in ('memory_gib','disk_gib','mpi_ranks','omp_threads','cpu_slots'):
        value=actual[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0 or value<expected[key] or key in ('mpi_ranks','omp_threads','cpu_slots') and type(value) is not int:
            raise MemoryError('Allocated production resources are insufficient: '+key)
    # MUSIC is a threaded process on this host; the existing assigned-RAM
    # detector includes Linux scheduler/cgroup limits. Its auto reserve remains.
    host_budget=memory_limit_gb({'hardware':{'memory_limit_gb':'auto'}})
    if host_budget<request['generation_memory_gib']:
        raise MemoryError('MUSIC generation working set exceeds the current host assigned-memory budget')
    if cpu_count()<request['generation_threads']:raise MemoryError('Current host has too few assigned generation CPUs')
    checked_output=Path(output_parent or request['output_parent']).resolve()
    probe=checked_output
    while not probe.exists():probe=probe.parent
    if shutil.disk_usage(probe).free<request['storage_gib']*1024**3:
        raise MemoryError('Production storage requirement exceeds current free disk')
    for ref in request['staging_records']:
        staging=_binding(ref);verify_staged_zoom_inputs(staging.parent)
    return {'schema':'flagship-production-admission-result-v1','status':'NECESSARY_ALLOCATION_AND_INTAKE_CHECKS_PASS',
        'allocation':request['allocation'],'allocation_source':{'path':str(source),'sha256':file_sha256(source)},
        'host_assigned_memory_budget_gib':host_budget,'output_parent_checked':str(checked_output),'science_acceptance':'NOT_ACCEPTED',
        'scope':'Declared allocation capacity, current generation-host budget, disk and sealed staging only; distributed per-rank/node feasibility and numerical acceptance remain separate'}


def production_admission_runner(parameters,context):
    result=validate_production_admission(parameters,output_parent=context.destination)
    atomic_json(context.generation/'production-admission.json',result)
    return result


def prepare_production_campaign(request,destination):
    from .workflow import compile_requested_product_dag,safe_id
    from core.research.capabilities import validate_native_answers
    from .flagship_profiles import production_bridge_bindings
    from .zoom import apply_traced_region
    if request.get('schema')!='flagship-production-campaign-request-v1':raise ValueError('Explicit production campaign request required')
    target=Path(destination).resolve()
    if target.exists():raise FileExistsError('Preserve previous production campaign preparation')
    pins={}
    def bind(ref):
        path=_binding(ref);pins[str(path)]=ref['sha256'];return path
    staging=json.loads(bind(request['staging']).read_text());settings=json.loads(bind(request['evolution_settings']).read_text())
    hardware=json.loads(bind(request['hardware_requirements']).read_text())
    if staging.get('schema')!='flagship-production-input-staging-v1' or staging.get('status')!='INPUTS_STAGED':raise ValueError('Retained production staging required')
    if settings.get('schema')!='flagship-production-evolution-settings-v1' or settings.get('status')!='SETTINGS_PREPARED':raise ValueError('Complete prepared production evolution settings required')
    if hardware.get('schema')!='flagship-production-hardware-requirements-v1':raise ValueError('Prepared production hardware requirements required')
    # Staging may be historical immutable input material; admission rechecks
    # sealed staging under current source before any large native operation.
    for record in (settings,hardware):
        for source,digest in record['source_hashes'].items():
            if file_sha256(source)!=digest:raise ValueError('Production setting/sizing source changed: '+source)
            pins[source]=digest
    parent=request['original_parent'];parent_ic=bind(parent['ic']);parent_noise=bind(parent['noise'])
    binary=bind(request['generator']['binary']);validation=bind(request['generator']['native_validation'])
    if type(parent['level']) is not int:raise ValueError('Exact original parent level required')
    execution=request['execution']
    flow_templates=request.get('flow_preparation')
    if flow_templates is not None and (not isinstance(flow_templates,dict) or set(flow_templates)!={case['id'] for case in staging['cases']}):
        raise ValueError('Every production target needs its own runtime flow preparation contract')
    for key in ('generation_wall_hours','evolution_wall_hours','audit_memory_gib'):
        if type(execution[key]) not in (int,float) or not math.isfinite(execution[key]) or execution[key]<=0:raise ValueError('Explicit positive production execution bounds required')
    cases={row['id']:row for row in settings['cases']}
    if len(cases)!=len(settings['cases']) or set(cases)!={row['id'] for row in staging['cases']} or set(request['regions'])!=set(cases):
        raise ValueError('Production staging, settings and traced targets must match exactly')
    resources={**settings['requested_resources'],'disk_gib':hardware['requested_campaign_storage_gib'],
        'cpu_slots':request['cpu_slots']}
    if type(resources['cpu_slots']) is not int or resources['cpu_slots']<resources['mpi_ranks']*resources['omp_threads']:
        raise ValueError('Production MPI/thread controls exceed proposed CPU allocation')
    admission={'schema':'flagship-production-admission-v1','allocation':request.get('allocation'),
        'requested_resources':resources,'generation_memory_gib':hardware['requested_generation_memory_gib'],
        'generation_threads':resources['omp_threads'],'storage_gib':hardware['requested_campaign_storage_gib'],
        'output_parent':str(target/'execution'),'staging_records':[row['staging'] for row in staging['cases']]}
    if request.get('allocation') is not None:bind(request['allocation'])
    stages=[{'stage_id':'production_admission','kind':'flagship_production_admission','parameters':{'request':admission},
        'outputs':{'production_admission':'production-admission.json'}}];products=[];records=[]
    if flow_templates is not None:target.mkdir(parents=True)
    for case in staging['cases']:
        identifier=safe_id(case['id']);answer=cases[identifier]
        if case['parent_level']!=parent['level'] or answer['effective_level']!=case['level'] or answer['host_row']!=case['host_row']:
            raise ValueError('Production original parent, target or resolution changed')
        configuration=bind(case['configuration']);stage_path=bind(case['staging']);region=bind(request['regions'][identifier]);traced=json.loads(region.read_text())
        if traced['catalogue_binding']['host_fof_row']!=case['host_row'] or traced['initial_ic']!=parent['ic']:
            raise ValueError('Production traced target has a different original parent or host identity')
        cfg=configparser.ConfigParser(interpolation=None);cfg.optionxform=str;cfg.read(configuration)
        if cfg.getint('setup','levelmin')!=case['generation_base_level'] or cfg.getint('setup','levelmax')!=case['level']:
            raise ValueError('Production bridge/refinement differs from actual configuration')
        staged_record=json.loads(stage_path.read_text())
        template=stage_path.parent/'zoom_template.conf'
        template_ref={'path':str(template),'sha256':staged_record['artifacts']['zoom_template.conf']}
        generation_configuration=bind(template_ref)
        restaging=None
        if flow_templates is not None:
            from core.music.staging import restage_campaign_case
            controls=request['case_staging'][identifier]
            restaging=restage_campaign_case(generation_configuration,target/(identifier+'.staged'),
                source_archive_max_mib=controls['source_archive_max_mib'],bounded_source_settings=controls['bounded_source_settings'])
            pins.update(restaging['source_hashes'])
            generation_configuration=bind(restaging['configuration']);template_ref=restaging['configuration']
            admission['staging_records']=[ref for ref in admission['staging_records'] if ref!=case['staging']]+[restaging['staging']]
            fresh=json.loads(bind(restaging['staging']).read_text())
            for relative,digest in fresh.get('sealed_assets',fresh['artifacts']).items():
                bind({'path':str(generation_configuration.parent/relative),'sha256':digest})
        generated_cfg=configparser.ConfigParser(interpolation=None);generated_cfg.optionxform=str;generated_cfg.read(generation_configuration)
        apply_traced_region(generated_cfg,region)
        for key in ('levelmin','levelmax','boxlength','ref_center','ref_extent'):
            if generated_cfg.get('setup',key)!=cfg.get('setup',key):
                raise ValueError('Production generation template and traced planning geometry differ: '+key)
        for relative,digest in staged_record.get('sealed_assets',staged_record['artifacts']).items():
            asset=(stage_path.parent/relative).resolve()
            if not asset.is_relative_to(stage_path.parent):raise ValueError('Sealed production asset escapes staging')
            bind({'path':str(asset),'sha256':digest})

        for ref in case['original_frozen_prefix'].values():bind(ref)
        level,bridge,bridge_pins=production_bridge_bindings(case['periodic_bridge'],{'phase_noise_by_level':case['original_frozen_prefix']},case['seeds_by_level'])
        if level!=case['generation_base_level']:raise ValueError('Executed bridge does not supply the selected generation base')
        pins.update(bridge_pins)
        answer_path=bind(answer['answers']);answers=json.loads(answer_path.read_text());validate_native_answers(answers,resources)
        if answers.get('native_flow_window_comparison') is not None:
            from core.music.flow import original_flow_model_bindings
            pins.update(original_flow_model_bindings(answers['native_flow_window_comparison'],levels=range(case['generation_base_level'],case['level']+1)))
        if answers['particles_per_dimension']!=2**case['level']:raise ValueError('Production evolution changes effective resolution')
        gen='generate_'+identifier;evolve='evolve_'+identifier;ref=lambda p:{'input':str(p)}
        stages.append({'stage_id':gen,'kind':'public_command','depends_on':['production_admission'],
            'parameters':{'command':'zoom','wall_seconds':execution['generation_wall_hours']*3600,
                'arguments':[ref(generation_configuration),'{generation}/generated','--binary',ref(binary),'--native-validation',ref(validation),
                    '--traced-region',ref(region),'--parent-ic',ref(parent_ic),'--parent-noise',ref(parent_noise),
                    '--threads',str(resources['omp_threads']),'--audit-memory-gib',str(execution['audit_memory_gib']),
                    '--native-memory-gib',str(hardware['requested_generation_memory_gib']),
                    '--native-disk-gib',str(hardware['requested_campaign_storage_gib']),
                    '--native-wall-hours',str(execution['generation_wall_hours'])]},
            'input_files':[], 'outputs':{gen:'generated'}})
        answers_reference=ref(answer_path);dependencies=[gen];flow_ref=None;flow_stage=None
        if flow_templates is not None:
            from core.music.staging import generated_flow_bindings
            flow=json.loads(bind(flow_templates[identifier]).read_text());flow['answers']=answer['answers']
            pins.update(generated_flow_bindings(flow,levels=range(case['generation_base_level'],case['level']+1)))
            flow_path=target/(identifier+'.flow-request.json');atomic_json(flow_path,flow);flow_ref={'path':str(flow_path),'sha256':file_sha256(flow_path)};bind(flow_ref)
            flow_stage='prepare_flow_'+identifier
            stages.append({'stage_id':flow_stage,'kind':'public_command','depends_on':[gen],
                'parameters':{'command':'prepare-generated-flow','wall_seconds':execution['generation_wall_hours']*3600,
                    'arguments':['--request',ref(flow_path),'--generation-report',{'stage':gen,'artifact':'generated/generation.json'},'--output','{generation}/prepared']},
                'input_files':[flow_ref,answer['answers']],'outputs':{flow_stage:'prepared'}})
            answers_reference={'stage':flow_stage,'artifact':'prepared/answers.json'};dependencies.append(flow_stage)
        elif generated_cfg.getboolean('setup','use_2LPT',fallback=False) and generated_cfg.getboolean('setup','baryons',fallback=False) and answers.get('native_flow_window_comparison') is None:
            raise ValueError('Production mixed-fluid evolution requires a runtime flow preparation recipe or fully bound original-window launch request')
        stages.append({'stage_id':evolve,'kind':'public_command','depends_on':dependencies,
            'parameters':{'command':'gadget','wall_seconds':execution['evolution_wall_hours']*3600,
                'arguments':['--answers',answers_reference,'--existing-ic',{'stage':gen,'artifact':'generated/ics_gadget.hdf5'},
                    '--native-generation-report',{'stage':gen,'artifact':'generated/generation.json'},'--run-dir','{generation}/run']},
            'input_files':[answer['answers']],'outputs':{evolve:'run'}})
        products.append(evolve);records.append({'id':identifier,'host_row':case['host_row'],'original_parent_level':parent['level'],
            'generation_base_level':level,'effective_level':case['level'],'planning_configuration':case['configuration'],'generation_configuration':template_ref,'generation_stage':gen,'evolution_stage':evolve,
            'flow_preparation_stage':flow_stage,'runtime_flow_request':flow_ref,'case_restaging':restaging})
    from core.research.checkpoints import evolution_checkpoint_policy
    for stage in stages:
        if stage['parameters'].get('command')=='gadget':stage['checkpoint_resume']=evolution_checkpoint_policy()
    helper=Path(__file__).resolve().parent/'research/checkpoints.py';pins[str(helper.resolve())]=file_sha256(helper)
    for source in (Path(__file__),Path(__file__).with_name('workflow.py')):pins[str(source.resolve())]=file_sha256(source)
    all_inputs=[{'path':path,'sha256':digest} for path,digest in pins.items()]
    for stage in stages:
        if stage['stage_id']=='production_admission' or stage['parameters'].get('command')=='zoom':stage['input_files']=all_inputs
    spec={'schema':'research-workflow-v1','resources':{k:resources[k] for k in ('mpi_ranks','omp_threads','memory_gib','disk_gib')},
        'followup_limits':{'maximum_children':0,'maximum_depth':0},'physics_spec':{'physics':'hydro'},'capabilities':[],
        'stages':stages,'requested_products':products}
    spec['resources']['maximum_stage_count']=len(stages);compile_requested_product_dag(spec,request_base=target)
    target.mkdir(parents=True,exist_ok=flow_templates is not None);path=target/'workflow.json';atomic_json(path,spec)
    root=Path(__file__).resolve().parents[1];command=[sys.executable,str(root/'commands.py'),'workflow','--campaign',str(path),'--output',str(target/'execution')]
    result={'schema':'flagship-production-campaign-v1','status':'RECIPES_PREPARED','cases':records,'source_hashes':pins,
        'workflow':{'path':str(path),'sha256':file_sha256(path)},'stage_count':len(stages),
        'commands':{'check':command+['--dry-run'],'run':command,'resume':command+['--resume']},
        'allocation':request.get('allocation'),'launch_ready':False,'science_acceptance':'NOT_ACCEPTED',
        'scope':'Production bridge generation/evolution recipes with a prelaunch allocation/intake gate; no generation, evolution, hardware allocation or numerical acceptance performed',
        'remaining_requirements':['Actual allocated resources and generation host capacity','Current sealed staging and native regression evidence','Native original-parent/bridge/species/2LPT IC handoff','Distributed rank/node feasibility and complete physics accounting','Recovery, analysis and display campaign stages']}
    for source,digest in pins.items():
        if file_sha256(source)!=digest:raise ValueError('Production campaign source changed during preparation')
    atomic_json(target/'manifest.json',result);return result
