"""Compose original production histories with existing measurement/display followups."""
from pathlib import Path
import copy,json,math
from .foundation import atomic_json,file_sha256
from .foundation import merge_source_bindings as _merge


def connect_production_products(campaign,settings,destination):
    from .flagship_history import prepare_history_campaign
    from .workflow import compile_requested_product_dag,default_registry
    from .flagship_portable import source_assets
    required={'schema','history_cases','measurements','display','maximum_stage_count','followup_limits'}
    if set(settings)-{'postprocessing_resources'}!=required or settings['schema']!='flagship-production-products-v1':raise ValueError('Complete production history/measurement/display settings required')
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Preserve original production downstream preparation')
    out.mkdir(parents=True);pins={};root=Path(__file__).resolve().parents[1]
    settings=source_assets(copy.deepcopy(settings),root,json_directory=out/'resolved-assets',pins=pins)
    ref=lambda p:{'path':str(p),'sha256':file_sha256(p)}
    manifest=json.loads(Path(campaign['path']).read_text())
    if file_sha256(campaign['path'])!=campaign['sha256']:raise ValueError('Production compiler manifest changed')
    _merge(pins,manifest['source_hashes']);_merge(pins,{campaign['path']:campaign['sha256']})
    original_workflow=Path(manifest['workflow']['path']).resolve()
    if file_sha256(original_workflow)!=manifest['workflow']['sha256']:raise ValueError('Original production workflow changed')
    resource_spec=json.loads(original_workflow.read_text());resources=resource_spec['resources']
    native_storage=[row['parameters']['request']['storage_gib'] for row in resource_spec['stages'] if row['kind']=='flagship_production_admission']
    if len(native_storage)!=1:raise ValueError('One original production allocation/storage gate required')
    native_resources=copy.deepcopy(resources)
    for row in resource_spec['stages']:
        if row['kind']=='public_command':row.setdefault('resources',copy.deepcopy(native_resources))
    if settings.get('postprocessing_resources') is not None:
        budget=settings['postprocessing_resources']
        if set(budget)!={'memory_gib','disk_gib'}:raise ValueError('Explicit production postprocessing memory/storage allocation required')
        for key,value in budget.items():
            if type(value) not in (int,float) or not math.isfinite(value) or value<resources[key]:raise ValueError('Postprocessing allocation cannot reduce original native resource requirements')
            resources[key]=value
        for row in resource_spec['stages']:
            if row['kind']=='flagship_production_admission':
                request=row['parameters']['request']
                for key,value in budget.items():request['requested_resources'][key]=value
                request['storage_gib']=max(request['storage_gib'],budget['disk_gib'])
        resource_spec['resources']=resources
        atomic_json(out/'resource-workflow.json',resource_spec)
        adjusted=copy.deepcopy(manifest);adjusted['workflow']=ref(out/'resource-workflow.json')
        adjusted['source_hashes']={**manifest['source_hashes'],**pins,str(original_workflow):file_sha256(original_workflow),
            str(Path(__file__).resolve()):file_sha256(__file__),str(out/'resource-workflow.json'):file_sha256(out/'resource-workflow.json')}
        adjusted['resource_extension']='Explicit larger downstream memory/storage; original generation geometry/physics and admission retained'
        atomic_json(out/'resource-campaign.json',adjusted);campaign=ref(out/'resource-campaign.json')
    history=prepare_history_campaign({'schema':'flagship-history-campaign-request-v1','campaign':campaign,'cases':settings['history_cases'],
        'retained_campaign_disk_bytes':math.ceil(native_storage[0]*1024**3),'defer_storage_to_production_admission':True},out/'history')
    spec=json.loads(Path(history['workflow']['path']).read_text());resources=spec['resources'];resources['maximum_stage_count']=settings['maximum_stage_count']
    spec['followup_limits']=copy.deepcopy(settings['followup_limits'])
    maximum=settings['maximum_stage_count']
    if type(maximum) is not int or maximum<len(spec['stages'])+2:raise ValueError('Production downstream stages exceed selected workflow limit')
    measurements=settings['measurements'];display=settings['display']
    if set(measurements)!={'memory_gib','maximum_metadata_files','options'}:raise ValueError('Complete original production measurement settings required')
    if set(display)!={'memory_gib','disk_gib','maximum_metadata_files','maximum_figures','scene'}:raise ValueError('Complete original production display settings required')
    def bounds(value):
        if isinstance(value,list):
            for item in value:bounds(item)
        elif isinstance(value,dict):
            for key,item in value.items():
                if key in {'memory_gib','disk_gib','copy_disk_gib'} and (type(item) not in (int,float) or item<=0 or item>resources['memory_gib' if key=='memory_gib' else 'disk_gib']):raise MemoryError('Production downstream '+key+' exceeds original campaign allocation')
                bounds(item)
    bounds(settings['history_cases']);bounds(measurements);bounds(display)
    definitions={row['stage_id']:row for row in spec['stages']};cases=[];linear=None
    original_cases={row['id']:row for row in manifest['cases']}
    def artifact(sid,name):return {'path':{'stage':sid,'artifact':name},'sha256':{'stage':sid,'artifact':name,'load':'sha256'}}
    for row in history['cases']:
        if row.get('branch') is not None:raise ValueError('Production downstream requires unpaired original production cases')
        case=original_cases[row['id']];sid=case['evolution_stage'];evolution=definitions[sid]
        # Production runtime flow can own answers dynamically; its frozen template
        # is explicitly recorded on the original compiler case/settings chain.
        from .flagship_history import _history_answer_template
        def bind(reference):
            path=Path(reference['path']).resolve()
            if file_sha256(path)!=reference['sha256']:raise ValueError('Production downstream answer source changed')
            _merge(pins,{str(path):reference['sha256']});return path
        answer_path=_history_answer_template(case,None,evolution,definitions,bind)
        values=json.loads(answer_path.read_text());bundle=Path(values['cosmology']).resolve()
        if linear is not None and linear!=bundle:raise ValueError('Production cases must retain the exact common CLASS bundle')
        linear=bundle
        # Pin the retained solver bundle as an actual native evolution input.
        # The measurement intake subsequently verifies this original receipt.
        binding=ref(bundle)
        inputs=evolution.setdefault('input_files',[])
        if not any(Path(item['path']).resolve()==bundle for item in inputs):inputs.append(binding)
        elif any(item['sha256']!=binding['sha256'] for item in inputs if Path(item['path']).resolve()==bundle):raise ValueError('Production evolution linear bundle binding changed')
        result={'id':row['id'],'run':{'stage':sid,'artifact':'run'},'native_receipt':{'stage':sid,'artifact':'stage-receipt.json'},'products':[]}
        for name,path,kind in [('recovery_stage','recovery.json','zoom_material_recovery'),('contamination_stage','contamination.json','zoom_contamination_history'),
            ('hydro_history_stage','hydro-history.json','native_stellar_history'),('stream_measurement_stage','stream-history.json','native_recovered_stream_measurements'),
            ('native_encounter_stage','encounter-history.json','recovered_native_encounters'),('analysis_stage','private_run/results/analysis_provenance.json','public_command')]:
            if row.get(name):result['products'].append({'role':name,'kind':kind,'reference':artifact(row[name],path),'receipt':{'stage':row[name],'artifact':'stage-receipt.json'}})
        if not row.get('hydro_history_stage') or not row.get('analysis_stage'):raise ValueError('Connected production needs original hydro history and public analysis producers')
        for sid in row.get('structure_stages',[]):result['products'].append({'role':'structure','kind':'zoom_recovered_structure','reference':artifact(sid,'measurement/status.json'),'receipt':{'stage':sid,'artifact':'stage-receipt.json'}})
        cases.append(result)
    if linear is None:raise ValueError('At least one original production case required')
    display_stage={'stage_id':'production_display','kind':'flagship_runtime_display','parameters':{'schema':'flagship-runtime-display-request-v1',
        'prefix':'production_display','cases':{'runtime_measurement_cases':True},**display},'outputs':{'production_display':'display-preparation.json'}}
    stage={'stage_id':'production_measurements','kind':'flagship_runtime_measurements','parameters':{'schema':'flagship-runtime-measurements-request-v1',
        'prefix':'production_measure','cases':cases,'linear_bundle':ref(linear),'linear_receipt':None,**measurements,'continuation':display_stage},
        'depends_on':[row['stage_id'] for row in spec['stages']],'outputs':{'production_measurements':'measurement-preparation.json'}}
    spec['stages'].append(stage);spec['requested_products'].append('production_measurements')
    registry=default_registry();compile_requested_product_dag(spec,registry,request_base=out)
    _merge(pins,history['source_hashes']);_merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__),str(linear):file_sha256(linear)})
    for row in spec['stages']:_merge(pins,registry.source_hashes(row['kind']))
    for path,sha in pins.items():
        if file_sha256(path)!=sha:raise ValueError('Production downstream source changed during preparation')
    atomic_json(out/'workflow.json',spec)
    result={'schema':'flagship-production-products-prepared-v1','status':'RECIPES_PREPARED','workflow':ref(out/'workflow.json'),
        'source_hashes':pins,'stage_count':len(spec['stages']),'case_count':len(cases),'native_execution':'NOT_EXECUTED','science_acceptance':'NOT_ACCEPTED'}
    atomic_json(out/'manifest.json',result);return result
