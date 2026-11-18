"""Freeze original production or matched SIDM compiler output for the launcher."""
from pathlib import Path
import copy
import json
from .foundation import atomic_json,file_sha256
from .foundation import merge_source_bindings as _merge


BUILDERS={'production':('flagship-production-campaign-request-v1','flagship_production_campaigns','prepare_production_campaign'),
    'sidm':('native-recovered-sidm-campaign-v1','sidm_recovery','prepare_recovered_sidm_campaign')}


def materialize_campaign_profile(profile,request_path,workspace,*,overrides=None,downstream_request=None):
    """Input campaign must retain its own native assets and original admission gates."""
    import importlib
    from .workflow import safe_id,compile_requested_product_dag,default_registry,positive_number
    from .sidm_campaign import _no_placeholders
    if set(profile)!={'schema','name','kind','build_engines'} or profile['schema']!='flagship-campaign-profile-v1' or profile['kind'] not in BUILDERS:
        raise ValueError('Explicit original production or matched SIDM campaign profile required')
    safe_id(profile['name']);source=Path(request_path).expanduser().resolve()
    if source.stat().st_size>16*1024**2:raise MemoryError('Campaign request exceeds bounded metadata intake')
    request=json.loads(source.read_text());_no_placeholders(request)
    schema,module,function=BUILDERS[profile['kind']]
    if request.get('schema')!=schema:raise ValueError('Selected profile requires original '+schema)
    if overrides:raise ValueError('Campaign resources belong to original campaign inputs; regenerate those inputs to change allocation')
    out=Path(workspace).resolve()
    if out.exists():raise FileExistsError('Preserve earlier original campaign preparation')
    out.mkdir(parents=True)
    try:
        builder=getattr(importlib.import_module('.'+module,'core'),function)
        # The original compiler verifies every scientific input, source binding,
        # geometry/resource setting and producer contract. Never relabel or reseal it.
        prepared=builder(request,out/'campaign')
        downstream_path=None
        if downstream_request is not None:
            if profile['kind']!='production':raise ValueError('Production downstream settings require the original production campaign')
            from .flagship_production_products import connect_production_products
            downstream_path=Path(downstream_request).resolve()
            if downstream_path.stat().st_size>16*1024**2:raise MemoryError('Production downstream request exceeds metadata bound')
            controls=json.loads(downstream_path.read_text());_no_placeholders(controls)
            prepared=connect_production_products({'path':str(out/'campaign/manifest.json'),'sha256':file_sha256(out/'campaign/manifest.json')},controls,out/'products')
        workflow_ref=prepared['workflow'];path=Path(workflow_ref['path']).resolve()
        if file_sha256(path)!=workflow_ref['sha256']:raise ValueError('Original compiler workflow changed')
        spec=json.loads(path.read_text());resources=copy.deepcopy(spec['resources'])
        resources.setdefault('build_jobs',1)
        for key in ('memory_gib','disk_gib','mpi_ranks','omp_threads','build_jobs','maximum_stage_count'):
            positive_number(resources[key],key,key not in {'memory_gib','disk_gib'})
        spec['resources']=resources
        registry=default_registry();compile_requested_product_dag(spec,registry,request_base=out/'campaign')
        pins={str(source):file_sha256(source),str(path):workflow_ref['sha256'],str(Path(__file__).resolve()):file_sha256(__file__)}
        _merge(pins,prepared['source_hashes'])
        if downstream_path is not None:_merge(pins,{str(downstream_path):file_sha256(downstream_path)})
        for row in spec['stages']:_merge(pins,registry.source_hashes(row['kind']))
        for name,sha in pins.items():
            if file_sha256(name)!=sha:raise ValueError('Original campaign input/source changed during launcher handoff')
        result={'schema':'flagship-launch-profile-v1','name':profile['name'],'kind':profile['kind'],'resources':resources,
            'build_engines':profile['build_engines'],'workflows':{'run':spec},'input_files':[{'path':name,'sha256':sha} for name,sha in pins.items()]}
        if downstream_path is not None:
            for operation in ('postprocess','render'):
                sid='retained_'+operation
                result['workflows'][operation]={'schema':'research-workflow-v1','physics_spec':{},'capabilities':[],
                    'resources':copy.deepcopy(resources),'followup_limits':copy.deepcopy(spec['followup_limits']),
                    'stages':[{'stage_id':sid,'kind':'flagship_retained_operation','parameters':{'operation':operation,
                        'source_workflow':{'launch_source_workflow':True},'memory_gib':resources['memory_gib'],'maximum_stages':resources['maximum_stage_count']},
                        'outputs':{sid:'retained-operation.json'}}],'requested_products':[sid]}
        atomic_json(out/'resolved-profile.json',result)
        atomic_json(out/'campaign-handoff.json',{'schema':'flagship-campaign-handoff-v1','execution_status':'RECIPES_PREPARED',
            'kind':profile['kind'],'source_hashes':pins,'original_compiler_manifest':{'path':str(out/'campaign/manifest.json'),'sha256':file_sha256(out/'campaign/manifest.json')},
            'native_execution':'NOT_EXECUTED','science_acceptance':'NOT_ACCEPTED'})
        return result,out/'resolved-profile.json'
    except BaseException as error:
        atomic_json(out/'failure.json',{'execution_status':'FAILED','type':type(error).__name__,'message':str(error),'science_acceptance':'NOT_ACCEPTED'});raise
