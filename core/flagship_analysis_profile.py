"""Portable published-BAO branches over the original CLASS and analysis engines."""
from pathlib import Path
import copy
import json
from .foundation import atomic_json,file_sha256
from .foundation import merge_source_bindings as _merge


def portable_analysis_profile(request,workspace,*,overrides=None):
    from .flagship_portable import source_assets
    from .flagship_launch import _build_engines
    from .workflow import positive_number,safe_id,compile_requested_product_dag
    from .bao import validate_bao_inference_settings
    from core.linear import validate
    required={'schema','name','kind','resources','class_configuration','observations','analysis','wall_seconds'}
    if set(request)!=required or request['schema']!='flagship-portable-bao-profile-v1' or request['kind'] not in {'observational','inference'}:
        raise ValueError('Complete portable measured BAO branch profile required')
    root=Path(__file__).resolve().parents[1];pins={};value=source_assets(copy.deepcopy(request),root,pins=pins)
    safe_id(value['name']);resources={**value['resources'],**(overrides or {})}
    for key in ('memory_gib','disk_gib','mpi_ranks','omp_threads','build_jobs','maximum_stage_count'):
        positive_number(resources[key],key,key not in {'memory_gib','disk_gib'})
    if set(value['wall_seconds'])!={'class','drag','analysis'}:raise ValueError('Explicit CLASS/drag/analysis wall bounds required')
    for key,seconds in value['wall_seconds'].items():positive_number(seconds,key+' wall seconds')
    if (value['kind']=='inference')!=(value['analysis'] is not None):raise ValueError('Posterior settings belong to explicit inference branch')
    if value['analysis'] is not None:validate_bao_inference_settings(value['analysis'],resources['memory_gib'])
    config=Path(value['class_configuration']['path']);parameters=json.loads(config.read_text());validate(parameters)
    out=Path(workspace).resolve()
    if out.exists():raise FileExistsError('Preserve earlier portable analysis preparation')
    out.mkdir(parents=True)
    try:
        engines=_build_engines(['class'],resources,pins);binary=engines[0]['binary']
        cosmology={'stage_id':'cosmology','kind':'public_command','parameters':{'command':'class','wall_seconds':value['wall_seconds']['class'],
            'arguments':['--config',{'input':str(config)},'--binary',{'input':binary['path']},'--output','{generation}/class']},
            'outputs':{'cosmology':'class/'+parameters['name']}}
        linear='class/'+parameters['name']+'/spectrum.json'
        row={'stage_id':'prepare_bao','kind':'flagship_runtime_bao','parameters':{'schema':'flagship-runtime-bao-request-v1','kind':value['kind'],
            'linear_bundle':{'path':{'stage':'cosmology','artifact':linear},'sha256':{'stage':'cosmology','artifact':linear,'load':'sha256'}},
            'linear_receipt':{'stage':'cosmology','artifact':'stage-receipt.json'},'observations':value['observations'],
            'analysis':value['analysis'],'wall_seconds':value['wall_seconds'],'resources':resources},'outputs':{'bao_preparation':'bao-preparation.json'}}
        for stage in (cosmology,row):stage['input_files']=[{'path':name,'sha256':sha} for name,sha in pins.items()]
        workflow={'schema':'research-workflow-v1','physics_spec':{},'capabilities':[],
            'resources':{key:{'launch_resource':key} for key in resources},'followup_limits':{'maximum_children':8,'maximum_depth':3},
            'stages':[cosmology,row],'requested_products':['bao_preparation']}
        compile_requested_product_dag({**workflow,'resources':resources},request_base=root)
        _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
        result={'schema':'flagship-launch-profile-v1','name':value['name'],'kind':value['kind'],'resources':resources,
            'build_engines':['class'],'workflows':{'run':workflow},'input_files':[{'path':name,'sha256':sha} for name,sha in pins.items()]}
        atomic_json(out/'resolved-profile.json',result)
        return result,out/'resolved-profile.json'
    except BaseException as error:
        atomic_json(out/'failure.json',{'status':'FAILED','type':type(error).__name__,'message':str(error)});raise


def runtime_bao_inputs(parameters):
    from core.research.convergence import _owned
    from .initial_conditions import validate_linear_bundle
    from .workflow import positive_number
    required={'schema','kind','linear_bundle','linear_receipt','observations','analysis','wall_seconds','resources'}
    if set(parameters)!=required or parameters['schema']!='flagship-runtime-bao-request-v1' or parameters['kind'] not in {'observational','inference'}:raise ValueError('Complete owned BAO runtime request required')
    pins={};ref=parameters['linear_bundle'];path=Path(ref['path']).resolve()
    if set(ref)!={'path','sha256'} or file_sha256(path)!=ref['sha256']:raise ValueError('BAO original CLASS bundle changed')
    _owned(path,parameters['linear_receipt'],'public_command',pins)
    memory=parameters['resources']['memory_gib'];positive_number(memory,'BAO memory')
    validate_linear_bundle(path,memory_gib=memory)
    observations=parameters['observations']
    if set(observations)!={'schema','source_origin','maximum_rows','mean','covariance','provenance'}:raise ValueError('Complete published measured BAO intake required')
    for key in ('mean','covariance','provenance'):
        ref=observations[key];source=Path(ref['path']).resolve()
        if set(ref)!={'path','sha256'} or file_sha256(source)!=ref['sha256']:raise ValueError('Published BAO source changed')
        _merge(pins,{str(source):ref['sha256']})
    return path,pins


def runtime_bao_runner(parameters,context):
    from core.native import run_native_command
    from .bao import ingest_compressed_bao
    from .analysis_figure_campaign import prepare_analysis_figure_campaign
    from .bao import parse_drag_sound_horizon
    for key in ('memory_gib','disk_gib'):
        if parameters['resources'][key]>context.resources[key]:raise MemoryError('BAO branch exceeds selected '+key)
    bundle,pins=runtime_bao_inputs(parameters);info=json.loads(bundle.read_text());out=context.generation
    measured=out/'observations.json';ingest_compressed_bao(parameters['observations'],measured)
    original=(bundle.parent/info['source_files']['z0']['ini']).resolve()
    if not original.is_relative_to(bundle.parent) or file_sha256(original)!=info['checksums_sha256'][info['source_files']['z0']['ini']]:raise ValueError('BAO frozen CLASS settings changed')
    binary=Path(info['solver_binary']['path']).resolve()
    if file_sha256(binary)!=info['solver_binary']['sha256']:raise ValueError('BAO frozen CLASS binary changed')
    drag=out/'drag';drag.mkdir();derived=drag/'input.ini';changes={'root':str(drag/'class_'),'thermodynamics_verbose':'1'};lines=[]
    for line in original.read_text().splitlines():
        key=line.split('=',1)[0].strip() if '=' in line and not line.lstrip().startswith('#') else None
        if key not in changes:lines.append(line)
    derived.write_text('\n'.join([*lines,*[key+' = '+value for key,value in changes.items()]])+'\n')
    log=drag/'solver.log';invocation=drag/'invocation.json'
    with log.open('wb') as stream:
        run_native_command([str(binary),str(derived)],cwd=drag,receipt_path=invocation,inputs=[binary,derived],
            timeout=parameters['wall_seconds']['drag'],stdout=stream,stderr=stream)
    parse_drag_sound_horizon(log.read_text())
    ref=lambda path:{'path':str(path),'sha256':file_sha256(path)}
    sources={str(path):file_sha256(path) for path in (bundle,binary,original,derived)}
    atomic_json(drag/'execution.json',{'schema':'class-drag-calibration-execution-v1','status':'COMPLETE','source_hashes':sources,
        'changed_settings':changes,'physical_settings_unchanged':True,'science_acceptance':'NOT_ACCEPTED'})
    kind='bao_compressed_posterior' if parameters['kind']=='inference' else 'bao_compressed'
    settings=copy.deepcopy(parameters['analysis']) if parameters['analysis'] is not None else {}
    if kind=='bao_compressed_posterior':settings['checkpoint_directory']=str(context.destination/'checkpoints'/out.parent.name/out.name/'bao-posterior')
    inputs={'observations':str(measured),'linear_bundle':str(bundle),'drag_execution':str(drag/'execution.json'),
        'drag_log':str(log),'drag_invocation':str(invocation)}
    campaign={'schema':'cosmography-campaign-v1','experiment_id':'published-bao-'+parameters['kind'],
        'resources':{'memory_gib':parameters['resources']['memory_gib'],'maximum_stage_count':1,'maximum_objects':64,'maximum_spectral_pixels':64},
        'stages':[{'stage_id':'bao','kind':kind,'depends_on':[],'inputs':inputs,'parameters':settings}]}
    path=out/'campaign.json';atomic_json(path,campaign)
    figures=[{'variant':'bao_cmb_background','producer_stage':'bao'}]
    if kind=='bao_compressed_posterior':figures.append({'variant':'actual_posterior','producer_stage':'bao'})
    prepared=prepare_analysis_figure_campaign({'schema':'analysis-figure-campaign-request-v1','campaign':ref(path),
        'resources':parameters['resources'],'wall_seconds':parameters['wall_seconds']['analysis'],'figures':figures},out/'analysis-recipes')
    workflow=json.loads(Path(prepared['workflow']['path']).read_text());rows=workflow['stages']
    galleries=[]
    for row in rows:
        if row['kind'] not in {'hubble_figures','inference_figures'}:continue
        sid=row['stage_id'];artifact='figures/gallery/gallery.json'
        galleries.append({'path':{'stage':sid,'artifact':artifact},'sha256':{'stage':sid,'artifact':artifact,'load':'sha256'}})
    rows.append({'stage_id':'bao_gallery','kind':'combined_graph_gallery','parameters':{'request':{
        'schema':'combined-scientific-gallery-request-v1','galleries':galleries,'scenes':[],
        'memory_gib':parameters['resources']['memory_gib'],'disk_gib':parameters['resources']['disk_gib'],'maximum_figures':16}},
        'outputs':{'bao_gallery':'gallery'}})
    _merge(pins,sources)
    for source,sha in pins.items():
        if file_sha256(source)!=sha:raise ValueError('Original BAO branch input changed during preparation')
    atomic_json(out/'bao-preparation.json',{'schema':'flagship-runtime-bao-prepared-v1','execution_status':'RECIPES_PREPARED',
        'source_hashes':pins,'observations':ref(measured),'analysis_preparation':ref(out/'analysis-recipes/manifest.json'),
        'variants':[row['variant'] for row in figures],'posterior_execution':'NOT_EXECUTED','science_acceptance':'NOT_ACCEPTED'})
    return {'followups':rows,'measurements':{'variants':[row['variant'] for row in figures],'science_acceptance':'NOT_ACCEPTED'}}
