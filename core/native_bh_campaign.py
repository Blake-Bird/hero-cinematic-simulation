"""Native BH history/journal/image producers and executable graph recipes."""
from pathlib import Path
import copy
import json
import numpy as np
from .foundation import atomic_json,file_sha256


def _positive(memory):
    if type(memory) not in (int,float) or not np.isfinite(memory) or memory<=0:
        raise ValueError('Positive finite native BH workspace required')


def configured_bh_output_contract(run):
    """Check the existing model/build generators configure native records."""
    from .analysis import load_cfg
    from .evolution import gadget_config_text
    from .black_holes import native_parameters
    cfg=load_cfg(run);parameters=native_parameters(cfg.get('black_holes',{'enabled':False}))
    if not parameters:raise ValueError('BH producer recipe requires an enabled native model')
    flags={line.split('=',1)[0].strip() for line in gadget_config_text(cfg).splitlines() if line.strip() and not line.lstrip().startswith('#')}
    required={'BLACK_HOLES','IDS_64BIT','OUTPUT_EXACT_SYNCHRONIZED','FORCE_EQUAL_TIMESTEPS'}
    if not required<=flags:raise ValueError('Native BH graph output requires synchronized exact-ID source records')
    return {'configured_flags':sorted(required),'dynamics_enabled':bool(parameters.get('BHDynamicsEnabled',0)),
        'event_policy':'Native seed/accretion journals always configured with BLACK_HOLES; merger/recoil/binary events require the explicit selected native dynamics model. No event occurrence guaranteed.',
        'periodic_image_policy':'PERIODIC native per-drift versioned image counters; absent/unsupported records stay unresolved',
        'snapshot_state':'Versioned BH ancestry, source integrals/corrections, native saved epochs and mass/energy/rate fields'}


def bh_history_inputs(parameters):
    from .analysis import snapshot_ids
    from .halos import _snapshot_files
    _positive(parameters['memory_gib']);run=Path(parameters['run_directory']).resolve()
    sources={str(run/name):file_sha256(run/name) for name in ('config.yaml','manifest.json')}
    snapshots=snapshot_ids(run)
    if not snapshots:raise ValueError('Native BH history needs actual saved snapshots')
    if type(parameters['maximum_saved_epochs']) is not int or len(snapshots)>parameters['maximum_saved_epochs']:
        raise MemoryError('Native BH saved epoch census exceeds declared bound')
    for snap in snapshots:
        for path in _snapshot_files(run,snap):sources[str(path.resolve())]=file_sha256(path)
    return sources


def native_bh_history_runner(parameters,context):
    from .black_holes import analyze_black_holes
    from .flagship_bh import export_native_bh_figures
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('BH history exceeds stage allocation')
    pins=bh_history_inputs(parameters)
    measured=analyze_black_holes(parameters['run_directory'],output_directory=context.generation/'history',memory_gib=parameters['memory_gib'])
    gallery=export_native_bh_figures(context.generation/'history',context.generation/'figures',
        run_directory=parameters['run_directory'],memory_gib=parameters['memory_gib'])
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Native BH history source changed')
    result={'schema':'native-bh-history-stage-v1','execution_status':'COMPLETE','measurement':measured,
        'figure_count':len(gallery['figures']),'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(context.generation/'history-result.json',result);return result


def publish_recoil_images(directory,run,identifier,destination,*,memory_gib):
    import h5py
    from .flagship_bh import bh_figure_input_bindings
    from .periodic_images import read_native_image_state
    from .halos import _snapshot_files
    from .data import exact_uint64_ids,export_numeric_audit
    record,history,run,pins=bh_figure_input_bindings(directory,run_directory=run,memory_gib=memory_gib)
    exact_uint64_ids([identifier]);epochs=record['epochs'];n=len(epochs)
    if n*512>memory_gib*1024**3/4:raise MemoryError('Native recoil image epochs exceed allocation')
    offsets=np.zeros((n,3),np.int64);support=np.zeros(n,bool);anchors=np.full(n,np.nan)
    for i,epoch in enumerate(epochs):
        seen=False
        for path in _snapshot_files(run,epoch['snapshot']):
            if pins.get(str(path.resolve()))!=file_sha256(path):raise ValueError('Recoil images require original history snapshot generation')
            with h5py.File(path) as f:
                if 'PartType5' not in f:continue
                group=f['PartType5'];step=max(1,min(100000,int(memory_gib*1024**3/1024)))
                for start in range(0,len(group['ParticleIDs']),step):
                    stop=min(start+step,len(group['ParticleIDs']));ids=exact_uint64_ids(group['ParticleIDs'][start:stop])
                    at=np.flatnonzero(ids==identifier)
                    if not len(at):continue
                    if seen or len(at)!=1:raise ValueError('Duplicate exact recoil survivor in native snapshot')
                    seen=True;state=read_native_image_state(group,start=start,stop=stop,memory_bytes=int(memory_gib*1024**3/2))
                    if state is not None:
                        j=int(at[0]);offsets[i]=state['image_offsets'][j];support[i]=state['supported'][j];anchors[i]=state['anchor_native_time'][j]
    valid=anchors[support]
    if len(valid)>1 and np.any(valid!=valid[0]):support[:]=False
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Native recoil image source changed during read')
    return export_numeric_audit(destination,{'schema':'native-bh-recoil-images-v1',
        'history_sha256':file_sha256(history),'particle_id':identifier,
        'namespace':record['data_contract']['identity_namespaces']['black_hole'],
        'snapshots':[e['snapshot'] for e in epochs],
        'recording':'Original per-drift native image counters; legacy missing state explicitly unsupported; no winding inference',
        'science_acceptance':'NOT_ACCEPTED'},
        {'image_offsets':offsets,'image_supported':support,'anchor_native_time':anchors},
        {key:{'units':'dimensionless','quantity':key,'missing_policy':'Per-epoch image_supported=false; NaN unavailable anchor'} for key in ('image_offsets','image_supported','anchor_native_time')},
        algorithm=__file__,context={'epoch':{'row_epoch_contracts':[e['epoch_contract'] for e in epochs]}},source_hashes=pins)


def native_bh_event_figures_runner(parameters,context):
    from .flagship_bh import (bh_figure_input_bindings,bh_event_input_bindings,
        export_native_bh_pair_separation,export_native_bh_merger_timeline,export_native_bh_recoil_trajectory)
    _positive(parameters['memory_gib'])
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('BH event graphs exceed stage allocation')
    record,history,run,pins=bh_figure_input_bindings(parameters['history_directory'],run_directory=parameters['run_directory'],memory_gib=parameters['memory_gib'])
    journal,sources=bh_event_input_bindings(parameters['hydro_history'],memory_gib=parameters['memory_gib']);pins.update(sources)
    if Path(journal['request']['run_directory']).resolve()!=run:raise ValueError('Native BH graph histories refer to different runs')
    namespace=record['data_contract']['identity_namespaces']['black_hole']
    ids=sorted({row['particle_id'] for row in record['history']})
    pairs=parameters['pair_ids']
    if pairs is None:pairs=ids[:2] if len(ids)>=2 else []
    elif not isinstance(pairs,list) or len(pairs)!=2 or len(set(pairs))!=2 or not set(pairs)<=set(ids):
        raise ValueError('Selected native BH pair requires two distinct observed exact IDs')
    galleries=[];unresolved=[]
    if pairs:
        gallery=export_native_bh_pair_separation(parameters['history_directory'],{
            'schema':'native-bh-pair-separation-v1','namespace':namespace,'particle_ids':pairs},
            context.generation/'pair',run_directory=run,memory_gib=parameters['memory_gib']);galleries.append(gallery)
    else:unresolved.append({'variant':'pair_separation','reason':'Fewer than two observed native BH identities'})
    galleries.append(export_native_bh_merger_timeline(parameters['hydro_history'],context.generation/'mergers',memory_gib=parameters['memory_gib']))
    mergers=[row for interval in journal['intervals'] for row in interval['black_hole_rows'] if row['kind']=='merger']
    selected=parameters['recoil_selection']
    if selected is None:
        selected={'survivor_id':mergers[0]['particle_id'],'event_counter':mergers[0]['counter']} if mergers else {'survivor_id':ids[0],'event_counter':0} if ids else None
    if selected is not None:
        if set(selected)!={'survivor_id','event_counter'}:raise ValueError('Exact native recoil event selection required')
        image=context.generation/'recoil-images.json'
        publish_recoil_images(parameters['history_directory'],run,selected['survivor_id'],image,memory_gib=parameters['memory_gib'])
        galleries.append(export_native_bh_recoil_trajectory(parameters['history_directory'],{
            'schema':'native-bh-recoil-trajectory-v1','namespace':namespace,**selected,
            'hydro_history':{'path':str(Path(parameters['hydro_history']).resolve()),'sha256':file_sha256(parameters['hydro_history'])},
            'recorded_images':{'path':str(image),'sha256':file_sha256(image)}},context.generation/'recoil',run_directory=run,memory_gib=parameters['memory_gib']))
    else:unresolved.append({'variant':'recoil_trajectory','reason':'No observed native BH survivor; no recoil detection invented'})
    result={'schema':'native-bh-event-figure-stage-v1','execution_status':'COMPLETE','selected_pair_ids':pairs,
        'selection_policy':'Explicit selected IDs or first two sorted observed IDs for a distance-only display; no bound binary claim',
        'selected_recoil':selected,'native_merger_count':len(mergers),'unresolved':unresolved,
        'figure_count':sum(len(g['figures']) for g in galleries),'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(context.generation/'event-figures.json',result);return result


def bh_event_stage_inputs(parameters):
    from .flagship_bh import bh_figure_input_bindings,bh_event_input_bindings
    pins=bh_figure_input_bindings(parameters['history_directory'],run_directory=parameters['run_directory'],memory_gib=parameters['memory_gib'])[-1]
    pins.update(bh_event_input_bindings(parameters['hydro_history'],memory_gib=parameters['memory_gib'])[-1])
    return pins


def bh_matched_stage_inputs(parameters):
    from .flagship_bh_response import response_inputs
    path=Path(parameters['pair_request_path']).resolve()
    if file_sha256(path)!=parameters['pair_request_sha256']:raise ValueError('Native BH pair request changed')
    covariance=Path(parameters['joint_covariance']).resolve();request=json.loads(path.read_text())
    request.update(schema='native-bh-matched-dm-response-v1',joint_covariance={'path':str(covariance),'sha256':file_sha256(covariance)})
    return {str(path):file_sha256(path),**response_inputs(request,memory_gib=parameters['memory_gib'])[-1]}


def native_bh_response_runner(parameters,context):
    from .flagship_bh_response import export_native_bh_dm_response
    _positive(parameters['memory_gib'])
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('BH response exceeds stage allocation')
    path=Path(parameters['pair_request_path']).resolve()
    if file_sha256(path)!=parameters['pair_request_sha256']:raise ValueError('Native BH paired request changed')
    request=json.loads(path.read_text());covariance=Path(parameters['joint_covariance']).resolve()
    request.update(schema='native-bh-matched-dm-response-v1',joint_covariance={'path':str(covariance),'sha256':file_sha256(covariance)})
    atomic_json(context.generation/'response-request.json',request)
    gallery=export_native_bh_dm_response(request,context.generation/'response',memory_gib=parameters['memory_gib'])
    return {'execution_status':'COMPLETE','figure_count':len(gallery['figures']),'science_acceptance':'NOT_ACCEPTED'}


def prepare_native_bh_campaign(request,destination):
    from .workflow import default_registry,compile_requested_product_dag
    from .flagship_bh_response import covariance_input_bindings
    from .stars import validate_native_hydro_tolerances
    if set(request)!={'schema','run_directory','memory_gib','disk_gib','snapshots','tolerances','pair_ids','recoil_selection','covariance_request'} or request['schema']!='native-bh-campaign-v1':
        raise ValueError('Complete native8-variant BH campaign request required')
    _positive(request['memory_gib']);_positive(request['disk_gib']);out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native BH campaign preparation')
    snapshots=request['snapshots']
    if not isinstance(snapshots,list) or len(snapshots)<2 or any(type(s) is not int or s<0 for s in snapshots) or len(set(snapshots))!=len(snapshots):
        raise ValueError('At least two distinct exact native BH audit snapshots required')
    memory=request['memory_gib'];run=Path(request['run_directory']).resolve();validate_native_hydro_tolerances(request['tolerances'])
    output_contract=configured_bh_output_contract(run)
    p={'run_directory':str(run),'maximum_saved_epochs':len(snapshots),'memory_gib':memory};pins=bh_history_inputs(p)
    from .analysis import snapshot_ids
    if snapshot_ids(run)!=snapshots:raise ValueError('BH graph/audit recipes must use exactly the actual saved snapshot set')
    binding=request['covariance_request'];cp=Path(binding['path']).resolve()
    params={'request_path':str(cp),'request_sha256':binding['sha256'],'memory_gib':memory}
    covariance,loaded,sources=covariance_input_bindings(params);pins.update(sources)
    if run not in {Path(report['request']['run_directory']).resolve() for report in (loaded[0],loaded[2])}:
        raise ValueError('BH graph run must be one of the actual matched structure branches')
    pair_path=out/'pair-request.json'
    ref=lambda stage,artifact:{'stage':stage,'artifact':artifact}
    stages=[{'stage_id':'bh-history','kind':'native_bh_history','parameters':p,'outputs':{'bh-history':'history','bh-history-figures':'figures'}},
        {'stage_id':'bh-journal','kind':'native_stellar_history','parameters':{
            'run_dir':str(run),'snapshots':snapshots,'memory_gib':memory,'tolerances':request['tolerances']},'outputs':{'bh-journal':'hydro-history.json'}},
        {'stage_id':'bh-events','kind':'native_bh_event_figures','parameters':{
            'history_directory':ref('bh-history','history'),'hydro_history':ref('bh-journal','hydro-history.json'),
            'run_directory':str(run),'memory_gib':memory,'pair_ids':request['pair_ids'],'recoil_selection':request['recoil_selection']},
            'outputs':{'bh-events':'event-figures.json'}},
        {'stage_id':'bh-covariance','kind':'native_bh_response_covariance','parameters':params,'outputs':{'bh-covariance':'joint-covariance.json'}},
        {'stage_id':'bh-response','kind':'native_bh_matched_response','parameters':{
            'pair_request_path':str(pair_path),'pair_request_sha256':None,
            'joint_covariance':ref('bh-covariance','joint-covariance.json'),'memory_gib':memory},'outputs':{'bh-response':'response'}}]
    # The exact emitted request hash is determined from the same serializer as
    # publication; no future runtime path or content is guessed.
    # atomic_json uses pretty JSON; compute those exact bytes before writing.
    import hashlib
    pair_bytes=(json.dumps(covariance['pair'],indent=2,allow_nan=False)+'\n').encode()
    stages[-1]['parameters']['pair_request_sha256']=hashlib.sha256(pair_bytes).hexdigest()
    for stage in stages:stage['input_files']=[{'path':path,'sha256':digest} for path,digest in sorted(pins.items())]
    spec={'schema':'research-workflow-v1','resources':{'mpi_ranks':1,'omp_threads':1,'memory_gib':memory,'disk_gib':request['disk_gib'],'maximum_stage_count':len(stages)},
        'physics_spec':{},'capabilities':[],'followup_limits':{'maximum_children':0,'maximum_depth':0},
        'requested_products':['bh-history-figures','bh-events','bh-response'],'stages':stages}
    registry=default_registry();plan=compile_requested_product_dag(spec,registry,request_base=out);code={}
    for stage in plan['stages']:code.update(registry.source_hashes(stage['kind']))
    for path,digest in {**pins,**code}.items():
        if file_sha256(path)!=digest:raise ValueError('Native BH campaign source changed during preparation')
    out.mkdir(parents=True);pair_path.write_bytes(pair_bytes)
    atomic_json(out/'request.json',request);atomic_json(out/'workflow.json',spec)
    result={'schema':'native-bh-campaign-prepared-v1','status':'RECIPES_PREPARED','stage_count':len(plan['stages']),
        'workflow':{'path':str(out/'workflow.json'),'sha256':file_sha256(out/'workflow.json')},'source_hashes':{**pins,**code,str(pair_path):file_sha256(pair_path)},
        'variants':['bh_growth','eddington_ratio','feedback_energy','pair_separation','merger_timeline','recoil_trajectory','dm_density_response','dm_shape_response'],
        'native_output_contract':output_contract,
        'science_acceptance':'NOT_ACCEPTED','scope':'Native retained-run source/history/image producers plus exact-source matched structure covariance/export; runtime absence of BH pairs/mergers/shape support remains unresolved.'}
    atomic_json(out/'manifest.json',result);return result
