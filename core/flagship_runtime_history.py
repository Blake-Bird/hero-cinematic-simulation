"""Extend completed runtime zoom generations with original native history DAGs.

Only an exactly matched planned generator may reuse an already completed
generation. Native generation receipts and science products remain immutable.
"""
from pathlib import Path
import copy
import json
from .foundation import atomic_json,file_sha256
from .foundation import merge_source_bindings as _merge


def runtime_history_inputs(parameters):
    from core.research.convergence import _owned
    from .zoom import verify_generated_zoom,verify_staged_zoom_inputs
    required={'schema','kind','zoom_preparation','zoom_preparation_receipt','cases','parent_answers','combined_answers',
        'numerical_choices','generator','execution','flow_preparation','case_staging','history_template','continuation'}
    if set(parameters)!=required or parameters['schema']!='flagship-runtime-history-request-v1' or parameters['kind'] not in {'miniature','local-preparation','sidm'}:raise ValueError('Complete source-bound runtime history preparation required')
    pins={};memory=parameters['execution']['audit_memory_gib']
    def bind(reference):
        if set(reference)!={'path','sha256'}:raise ValueError('Exact runtime history input binding required')
        path=Path(reference['path']).resolve()
        if file_sha256(path)!=reference['sha256']:raise ValueError('Runtime history source changed')
        _merge(pins,{str(path):reference['sha256']});return path
    source=bind(parameters['zoom_preparation']);_owned(source,parameters['zoom_preparation_receipt'],'flagship_zoom_preparation',pins)
    planned=json.loads(source.read_text())
    if planned.get('schema')!='flagship-zoom-prepared-v1' or planned.get('execution_status')!='RECIPES_PREPARED':raise ValueError('Original runtime traced-zoom preparation required')
    expected={row['id']:row for row in planned['cases']};cases=parameters['cases']
    if not isinstance(cases,list) or len(cases)!=len(expected) or {row['id'] for row in cases}!=set(expected):raise ValueError('Actual generated cases must cover the complete configured zoom matrix')
    _merge(pins,planned['source_hashes']);staged={}
    for case in cases:
        before=expected[case['id']]
        for key in ('host_row','level','generation_stage','traced_region','configuration'):
            if case[key]!=before[key]:raise ValueError('Completed runtime generation changes the declared traced case')
        report_path=bind(case['generation']);owner=_owned(report_path,case['generation_receipt'],'public_command',pins)
        if owner['stage_id']!=case['generation_stage']:raise ValueError('Completed generation receipt belongs to a different configured stage')
        generation=verify_generated_zoom(report_path)
        if generation.get('publication_contract')!='native-ic-generation-v1' or generation.get('linear_audit_status')!='PASS' or generation.get('phase_audit',{}).get('status')!='PASS':raise ValueError('Runtime zoom requires its actual original accepted phase/linear IC audits')
        if generation['traced_region']!=case['traced_region'] or generation['source_configuration']!=case['configuration']:raise ValueError('Native generation consumed different traced geometry or configuration')
        directory=Path(generation['source_configuration']['path']).parent
        staged[case['id']]=verify_staged_zoom_inputs(directory)
        _merge(pins,{str(directory/'staging.json'):file_sha256(directory/'staging.json'),**{str(report_path.parent/name):sha for name,sha in generation['artifacts'].items()}})
    bind(parameters['parent_answers']);bind(parameters['flow_preparation'])
    for reference in parameters['generator'].values():bind(reference)
    def templates(value):
        if isinstance(value,list):
            for item in value:templates(item)
        elif isinstance(value,dict):
            if set(value)=={'path','sha256'}:bind(value)
            else:
                for item in value.values():templates(item)
    templates(parameters['history_template']);templates(parameters['combined_answers'])
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    for name,sha in pins.items():
        if file_sha256(name)!=sha:raise ValueError('Runtime history lineage source changed')
    return planned,staged,pins


def _rewrite(value,mapping):
    if isinstance(value,list):return [_rewrite(item,mapping) for item in value]
    if isinstance(value,dict):
        result={key:_rewrite(item,mapping) for key,item in value.items()}
        if 'stage' in result and 'artifact' in result:result['stage']=mapping.get(result['stage'],result['stage'])
        return result
    return value


def runtime_history_runner(parameters,context):
    from .flagship_campaigns import prepare_native_campaign,_matched_generation_parameters
    from .flagship_history import prepare_history_campaign
    from core.research.capabilities import validate_native_answers
    from .workflow import compile_requested_product_dag
    planned,staged,pins=runtime_history_inputs(parameters);out=context.generation
    ref=lambda path:{'path':str(Path(path).resolve()),'sha256':file_sha256(path)}
    descriptors=[value['solver_manifest'] for value in staged.values()]
    if not descriptors or any(value!=descriptors[0] for value in descriptors):raise ValueError('Runtime zooms must share their exact actual CLASS bundle')
    answers=copy.deepcopy(parameters['combined_answers']);answers['cosmology']=descriptors[0]['path']
    cfg=validate_native_answers(answers,context.resources)
    atomic_json(out/'combined-answers.json',answers)
    profile={'schema':'flagship-run-profile-v1','name':'runtime-'+parameters['kind'],'branch':parameters['kind'],
        'resources':copy.deepcopy(context.resources),'parent_answers':parameters['parent_answers'],'combined_physics_answers':ref(out/'combined-answers.json'),
        'cosmology':descriptors[0],'phases':{'parent_seed':answers['seed'],'parent_level':planned['parent_level'],'hierarchy':'Actual native parent prefix and original disjoint local streams'},
        'physics':{'gas':True,'cooling':bool(cfg['hydro']['cooling']),'star_formation':bool(cfg['hydro']['star_formation']),
            'black_holes':bool(cfg.get('black_holes',{}).get('enabled',False)),'delayed_returns':bool(cfg.get('stellar_evolution',{}).get('enabled',False))},
        'science_resolution':{'accepted':False,'status':'NATIVE_CONVERGENCE_PENDING'},
        'zoom_cases':[{key:copy.deepcopy(case[key]) for key in ('id','host_row','level','generation')} for case in parameters['cases']]}
    atomic_json(out/'profile.json',profile)
    parent=json.loads(Path(parameters['parent_answers']['path']).read_text())
    request={'schema':'flagship-native-campaign-request-v1','profile':ref(out/'profile.json'),'generator':parameters['generator'],
        'execution':parameters['execution'],'schedule':parent['schedule'],'numerical_choices':parameters['numerical_choices'],
        'flow_preparation':parameters['flow_preparation'],'case_staging':parameters['case_staging']}
    prepared=prepare_native_campaign(request,out/'native-recipes');spec=json.loads(Path(prepared['workflow']['path']).read_text())
    actual_rows={row['stage_id']:row for row in planned['generation_stage_definitions']};by_id={row['stage_id']:row for row in spec['stages']};mapping={};replacement={}
    for case in prepared['cases']:
        actual=next(row for row in parameters['cases'] if row['id']==case['id']);old=case['generation_stage'];sid=actual['generation_stage']
        if _matched_generation_parameters(by_id[old],pins)!=_matched_generation_parameters(actual_rows[sid],pins):raise ValueError('Cannot reuse a native generation after changing its actual generator settings')
        mapping[old]=sid;replacement[old]=copy.deepcopy(actual_rows[sid]);case['generation_stage']=sid
    rows=[]
    for row in spec['stages']:
        if row['stage_id'] in replacement:rows.append(replacement[row['stage_id']]);continue
        row=_rewrite(copy.deepcopy(row),mapping);row['depends_on']=[mapping.get(sid,sid) for sid in row.get('depends_on',[])]
        if row['kind']=='public_command' and row['parameters']['command']=='gadget':row['parameters']['arguments'].append('--evolution-only')
        rows.append(row)
    spec['stages']=rows;spec['resources']=copy.deepcopy(context.resources)
    compile_requested_product_dag(spec,request_base=out)
    atomic_json(out/'derived-native-workflow.json',spec)
    prepared.update(workflow=ref(out/'derived-native-workflow.json'),source_hashes={**prepared['source_hashes'],**pins},
        generation_reuse='Verified completed original generations; exact original generator parameter comparison passed')
    atomic_json(out/'derived-native-campaign.json',prepared)
    controls={}
    for case in prepared['cases']:
        control=copy.deepcopy(parameters['history_template']);control['parent_level']=planned['parent_level']
        restart=control.get('restart_exercise')
        if restart is not None and 'template' not in restart:
            from core.research.restart_campaign import freeze_restart_template
            native_answers=json.loads(Path(case['answers']['path']).read_text());native_answers.update(restart_source_capsule=True,restart_exercise_pause_step=0)
            native_cfg=validate_native_answers(native_answers,context.resources)
            template=freeze_restart_template(native_cfg,workspace=out/'restart-diagnostics'/case['id'],snapshot=restart['snapshot'],
                identity_namespace='original-IC-particle-identities:'+case['id'],maximum_particles=control['maximum_tracers'],
                maximum_events=100000,maximum_commits=100000,memory_gib=restart['memory_gib'])
            path=out/(case['id']+'.restart-template.json');atomic_json(path,template);restart['template']=ref(path)
        controls[case['id']]=control
    history=prepare_history_campaign({'schema':'flagship-history-campaign-request-v1','campaign':ref(out/'derived-native-campaign.json'),'cases':controls},out/'history')
    workflow=json.loads(Path(history['workflow']['path']).read_text());reused=set(mapping.values());children=[row for row in workflow['stages'] if row['stage_id'] not in reused]
    def completed_cases():
        def artifact(sid,name):return {'path':{'stage':sid,'artifact':name},'sha256':{'stage':sid,'artifact':name,'load':'sha256'}}
        cases=[]
        for record in history['cases']:
            case=next(row for row in prepared['cases'] if row['id']==record['id'])
            if record.get('branch') is not None:raise ValueError('Runtime display needs an explicit paired-branch campaign adapter')
            sid=case['evolution_stage']
            result={'id':record['id'],'run':{'stage':sid,'artifact':'run'},'native_receipt':{'stage':sid,'artifact':'stage-receipt.json'},'products':[]}
            for key,name,kind in [('recovery_stage','recovery.json','zoom_material_recovery'),
                    ('contamination_stage','contamination.json','zoom_contamination_history'),
                    ('hydro_history_stage','hydro-history.json','native_stellar_history'),
                    ('stream_measurement_stage','stream-history.json','native_recovered_stream_measurements'),
                    ('native_encounter_stage','encounter-history.json','recovered_native_encounters'),
                    ('analysis_stage','private_run/results/analysis_provenance.json','public_command')]:
                if record.get(key):
                    stage=record[key];result['products'].append({'role':key,'kind':kind,'reference':artifact(stage,name),
                        'receipt':{'stage':stage,'artifact':'stage-receipt.json'}})
            for stage in record.get('structure_stages',[]):
                result['products'].append({'role':'structure','kind':'zoom_recovered_structure','reference':artifact(stage,'measurement/status.json'),
                    'receipt':{'stage':stage,'artifact':'stage-receipt.json'}})
            cases.append(result)
        return cases
    if parameters['continuation'] is not None:
        def bind(value):
            if isinstance(value,list):return [bind(item) for item in value]
            if isinstance(value,dict):
                if value=={'runtime_history_manifest':True}:return ref(out/'history/manifest.json')
                if value=={'runtime_native_campaign':True}:return ref(out/'derived-native-campaign.json')
                if value=={'runtime_history_receipt':True}:return {'stage':context.stage_id,'artifact':'stage-receipt.json'}
                if value=={'runtime_history_cases':True}:return completed_cases()
                return {key:bind(item) for key,item in value.items()}
            return value
        final=bind(copy.deepcopy(parameters['continuation']));final['depends_on']=list(dict.fromkeys([*final.get('depends_on',[]),*[row['stage_id'] for row in children]]));children.append(final)
    for name,sha in pins.items():
        if file_sha256(name)!=sha:raise ValueError('Runtime history source changed during preparation')
    atomic_json(out/'runtime-history-preparation.json',{'schema':'flagship-runtime-history-prepared-v1','execution_status':'RECIPES_PREPARED',
        'history_manifest':ref(out/'history/manifest.json'),'native_campaign':ref(out/'derived-native-campaign.json'),
        'reused_generation_stages':sorted(reused),'new_stage_count':len(children),
        'source_hashes':pins,'native_evolution':'NOT_EXECUTED','science_acceptance':'NOT_ACCEPTED'})
    return {'followups':children,'measurements':{'reused_completed_generations':len(reused),'new_native_history_stages':len(children),'science_acceptance':'NOT_ACCEPTED'}}
