"""Prepare, display and refine encounters from actual native histories.

Saved snapshot positions and reconstructed event positions remain separate.
Spatial coordinates use proper kpc, physical clocks use Myr, and native restart
clocks retain their own units (scale factor or elapsed L/V). Refinement plans
need verified earlier checkpoints; denser output does not certify dynamics.
"""
import copy
import json
import math
from pathlib import Path
import zipfile
import numpy as np
from ..foundation import atomic_json, file_sha256


def _merge_source_bindings(pins, additional):
    """Retain one source generation across every input to this workflow."""
    for source, checksum in additional.items():
        if source in pins and pins[source] != checksum:
            raise ValueError('Conflicting native encounter source generations: ' + source)
        pins[source] = checksum


def validate_encounter_display(parameters):
    for key in ('maximum_objects','maximum_vertices','frame_stride'):
        if type(parameters[key]) is not int or parameters[key]<1:raise ValueError('Positive exact encounter display '+key+' required')
    m=parameters['memory_gib']
    if type(m) not in (int,float) or not np.isfinite(m) or m<.1:raise ValueError('Bounded encounter display memory required')



def encounter_display_inputs(parameters):
    from core.stream_common import read_product
    from ..data import numeric_audit_source_bindings
    validate_encounter_display(parameters);pins={};histories=[]
    for name in ('stream_history','perturber_history','encounters'):
        path=Path(parameters[name]).resolve()
        if path.stat().st_size*16>parameters['memory_gib']*1024**3/8:raise MemoryError('Encounter display metadata exceeds workspace')
        report,arrays,sources=read_product(path,memory_gib=parameters['memory_gib']/8)
        _merge_source_bindings(pins,sources)
        _merge_source_bindings(pins,numeric_audit_source_bindings(report))
        histories.append((report,arrays))
    (stream,_),(perturbers,_),(events,_)=histories
    for report in (stream,perturbers):
        if report.get('schema')!='native-encounter-frame-history-v1':raise ValueError('Converted native worldline display inputs required')
    if stream['epochs']!=perturbers['epochs'] or stream['frame_definition_sha256']!=perturbers['frame_definition_sha256']:
        raise ValueError('Encounter display changes saved epochs or common native chart')
    if (events.get('schema')!='stream-encounters-v1' or events['stream_history_sha256']!=file_sha256(parameters['stream_history'])
            or events['perturber_history_sha256']!=file_sha256(parameters['perturber_history'])):
        raise ValueError('Encounter display catalogue must bind the exact source histories')
    return histories,pins



def encounter_display_input_bindings(parameters):return encounter_display_inputs(parameters)[1]



def native_encounter_display_runner(parameters,context):
    from ..data import export_numeric_audit,coordinate_contract
    from core.research.display import assemble_measured_display
    from core.stream.trajectories import Worldline
    validate_encounter_display(parameters)
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Encounter display exceeds stage allocation')
    histories,pins=encounter_display_inputs(parameters);(stream,_),(perturbers,_),(events,event_arrays)=histories
    epochs=stream['epochs'];lines=[('stream',line) for line in stream['worldlines']]+[('subhalo',line) for line in perturbers['worldlines']]
    if len(lines)*len(epochs)>parameters['maximum_objects'] or len(lines)*len(epochs)*2>parameters['maximum_vertices']:
        raise MemoryError('Encounter display object/vertex census exceeds frozen limits')
    frame=coordinate_contract(position_units='proper kpc',velocity_convention='proper position derivative km/s in source-bound fixed Cartesian frame',
        origin=stream['frame'],periodic_wrapping='recorded native periodic images already applied, no inferred winding')
    keys=[line['namespace']+':'+line['identity'] for _,line in lines]
    if len(set(keys))!=len(keys):raise ValueError('Distinct exact encounter worldline identities required for display')
    effective_support=[Worldline(line).valid for _,line in lines]
    pins[str(Path(__file__).resolve())]=file_sha256(__file__);timeline=[];maps=[]
    for ordinal,epoch in enumerate(epochs):
        products=[]
        for index,(kind,line) in enumerate(lines):
            supported=np.asarray(line.get('observation_supported',[True]*len(epochs)))
            if supported.dtype.kind!='b' or supported.shape!=(len(epochs),):raise ValueError('Exact native observation support required')
            if not supported[ordinal]:continue
            keys=line.get('native_observation_keys')
            if keys:
                anchor=next(key for key in keys if key is not None);identifier=int(anchor[1]);namespace=anchor[0]+':track-anchor:'+str(anchor[2])
            else:
                if not isinstance(line['identity'],str) or not line['identity'].isdecimal():raise ValueError('Native stellar display needs its original exact decimal particle identity')
                identifier=int(line['identity']);namespace=line['namespace']
            if not 0<=identifier<2**64:raise ValueError('Exact uint64 native encounter display identity required')
            x=np.asarray(line['position_kpc'],float);flags=effective_support[index]
            if x.shape!=(len(epochs),3) or not np.isfinite(x).all() or flags.dtype.kind!='b' or flags.shape!=(len(epochs)-1,):
                raise ValueError('Aligned saved native positions and interval flags required')
            previous=max(0,ordinal-1);valid=ordinal>0 and bool(flags[previous]) and bool(supported[previous])
            path=context.generation/('saved-%04d-line-%04d.json'%(ordinal,index))
            arrays={'object_ids':np.array([identifier],np.uint64),'position':x[ordinal:ordinal+1],
                'points':x[[previous,ordinal]][None,:,:],'supported':np.array([valid],bool)}
            specifications={'object_ids':{'units':'identity','quantity':'original exact particle or native track-anchor object ID','identity_namespace':namespace},
                'position':{'units':'proper kpc','quantity':'actual saved native position','coordinates':frame},
                'points':{'units':'proper kpc','quantity':'two actual saved native positions, no trajectory resampling','coordinates':frame},
                'supported':{'units':'dimensionless','quantity':'recorded supported interval ending at this actual saved epoch'}}
            export_numeric_audit(path,{'schema':'native-encounter-saved-display-v1','native_identity':line['identity'],
                'native_namespace':line['namespace'],'native_observation_keys':keys,'snapshot':epoch['snapshot_storage_index'],
                'scope':'Actual saved native endpoints; display holds last measured state between epochs, no dynamical interpolation',
                'science_acceptance':'NOT_ACCEPTED'},arrays,specifications,algorithm=__file__,source_hashes=pins,context={'epoch':epoch,'coordinates':frame})
            # Stream curves can display either native star or perturber paths;
            # physical kind stays in the attached source identity metadata.
            products.append({'path':str(path),'sha256':file_sha256(path),'kind':'stream','fields':{
                'object_id':'object_ids','position':'position','points':'points','geometry_supported':'supported'}})
            maps.append({'epoch_ordinal':ordinal,'kind':kind,'namespace':namespace,'object_id_decimal':str(identifier),
                'native_identity':line['identity'],'native_namespace':line['namespace'],'product':str(path)})
        timeline.append({'frame':1+ordinal*parameters['frame_stride'],'snapshot':epoch['snapshot_storage_index'],'epoch_contract':epoch,'products':products})
    request={'schema':'measured-display-build-v1','memory_gib':parameters['memory_gib'],'maximum_objects':parameters['maximum_objects'],
        'maximum_geometry_vertices':parameters['maximum_vertices'],'maximum_geometry_objects':parameters['maximum_objects'],
        'frame_contract':frame,'timeline':timeline}
    atomic_json(context.generation/'display-request.json',request);atomic_json(context.generation/'native-identity-map.json',{'schema':'native-encounter-display-identities-v1','records':maps})
    receipt=assemble_measured_display(request,context.generation/'display',request_base=context.generation)
    # Event positions are reconstructed at their own event clocks, not saved
    # snapshot observations. Retain explicit event times/support and source
    # identities for the inspector/event-overlay consumer.
    event_path=context.generation/'reconstructed-impacts.json'
    export_numeric_audit(event_path,{'schema':'native-encounter-reconstructed-impact-display-v1','events':events['events'],
        'position_meaning':'Hermite closest-approach reconstruction at encounter_time_myr; not actual saved snapshot centers',
        'science_acceptance':'NOT_ACCEPTED'},
        {'time_myr':event_arrays['encounter_time_myr'],'stream_position_kpc':np.array([e['stream_position_kpc'] for e in events['events']]).reshape(-1,3),
         'perturber_position_kpc':np.array([e['perturber_position_kpc'] for e in events['events']]).reshape(-1,3),'geometry_supported':event_arrays['geometry_supported']},
        {'time_myr':{'units':'Myr','quantity':'reconstructed proper event clock'},
         'stream_position_kpc':{'units':'proper kpc','quantity':'reconstructed stream encounter position','coordinates':frame},
         'perturber_position_kpc':{'units':'proper kpc','quantity':'reconstructed perturber encounter position','coordinates':frame},
         'geometry_supported':{'units':'dimensionless','quantity':'original interpolation certificate and geometry budget support'}},
        algorithm=__file__,source_hashes=pins,context={'epoch':{'representation':'reconstructed event times, not saved snapshot epochs'},'coordinates':frame})
    result={'schema':'native-encounter-display-result-v1','saved_display':receipt,'event_impacts':{'path':str(event_path),'sha256':file_sha256(event_path)},
        'science_acceptance':'NOT_ACCEPTED','source_hashes':pins}
    atomic_json(context.generation/'encounter-display.json',result);return result



def validate_encounter_refinement(parameters):
    for key,minimum in (('ranks',1),('omp_threads',1),('maximum_branches',0),('maximum_outputs',2)):
        if type(parameters[key]) is not int or parameters[key]<minimum:raise ValueError('Exact bounded encounter refinement '+key+' required')
    for key in ('memory_gib','disk_gib','wall_hours','maximum_timestep'):
        if type(parameters[key]) not in (int,float) or not math.isfinite(parameters[key]) or parameters[key]<=0:
            raise ValueError('Positive finite encounter refinement '+key+' required')
    if parameters['maximum_outputs']>1100:raise ValueError('Native refined output budget exceeds existing scheduler limit')
    directories=parameters['checkpoint_directories']
    if not isinstance(directories,list) or any(not isinstance(path,str) or not path for path in directories) or len(set(directories))!=len(directories):
        raise ValueError('Explicit distinct retained checkpoint directories required')



def refinement_input_bindings(parameters):
    from core.stream_common import read_product
    from ..data import numeric_audit_source_bindings,verify_native_checkpoints
    validate_encounter_refinement(parameters);pins={}
    for name in ('encounters','stream_history'):
        path=Path(parameters[name]).resolve()
        if path.stat().st_size*16>parameters['memory_gib']*1024**3/8:raise MemoryError('Encounter refinement metadata exceeds intake budget')
        report,_,sources=read_product(path,memory_gib=parameters['memory_gib']/8)
        _merge_source_bindings(pins,sources)
        _merge_source_bindings(pins,numeric_audit_source_bindings(report))
    run=Path(parameters['run_dir']).resolve()
    for name in ('param.txt','Config.sh','Gadget4'):
        path=run/'input'/name
        _merge_source_bindings(pins,{str(path):file_sha256(path)})
    for directory in parameters['checkpoint_directories']:
        verified=verify_native_checkpoints(directory,parameters['ranks'],require_receipts=True)
        for row in verified['files']:
            path=Path(row['path'])
            _merge_source_bindings(pins,{str(path):file_sha256(path)})
            seal=Path(str(path)+'.checksum.json')
            _merge_source_bindings(pins,{str(seal):file_sha256(seal)})
            if row.get('epoch_metadata'):
                value=row['epoch_metadata']
                _merge_source_bindings(pins,{value['path']:value['sha256']})
    return pins



def native_encounter_refinement_runner(parameters,context):
    from core.stream_common import read_product
    from ..evolution import parameter_values,choose_restart,merge_and_quantize_native_output_times
    from ..cosmology import run_proper_time_clock,interpolate_proper_time
    from ..data import proper_time_error_at_epoch
    from scipy.optimize import brentq
    validate_encounter_refinement(parameters)
    if parameters['memory_gib']>context.resources['memory_gib'] or parameters['ranks']>context.resources['mpi_ranks'] or parameters['omp_threads']>context.resources['omp_threads']:
        raise MemoryError('Native encounter branch exceeds workflow allocation')
    pins=refinement_input_bindings(parameters);memory=parameters['memory_gib'];parent=Path(parameters['run_dir']).resolve()
    encounters,_,_=read_product(parameters['encounters'],memory_gib=memory/8)
    frame,values,_=read_product(parameters['stream_history'],memory_gib=memory/8)
    if encounters.get('schema')!='stream-encounters-v1' or encounters['stream_history_sha256']!=file_sha256(parameters['stream_history']):
        raise ValueError('Refinement requires exact source-bound native encounter histories')
    if frame.get('schema')!='native-encounter-frame-history-v1':raise ValueError('Converted native saved stream clock required')
    raw,_,sources=read_product(frame['source_history'],memory_gib=memory/8)
    if file_sha256(frame['source_history'])!=frame['source_history_sha256'] or Path(raw['native_run_directory']).resolve()!=parent:
        raise ValueError('Encounter refinement belongs to another native source run')
    from ..data import numeric_audit_source_bindings
    _merge_source_bindings(pins,sources)
    _merge_source_bindings(pins,numeric_audit_source_bindings(raw))
    params=parameter_values(parent/'input/param.txt');comoving=int(params['ComovingIntegrationOn'])==1
    begin,end=float(params['TimeBegin']),float(params['TimeMax']);maximum=parameters['maximum_timestep']
    if maximum>float(params['MaxSizeTimestep']):raise ValueError('Refinement cannot increase the original native timestep')
    config=(parent/'input/Config.sh').read_text().splitlines()
    if 'OUTPUT_NON_SYNCHRONIZED_ALLOWED' in config:raise ValueError('Encounter branches require the existing synchronized output clock')
    clock={'time_begin':begin,'time_max':end,'maximum_timestep':maximum,'timebins':60 if 'ENLARGE_DYNAMIC_RANGE_IN_TIME' in config else 29,
        'comoving':comoving,'output_mode':'exact_synchronized' if 'OUTPUT_EXACT_SYNCHRONIZED' in config else 'timestep_grid'}
    source=None
    if comoving:
        scale,ages,source=run_proper_time_clock(parent)
        if scale is None or source.get('status')!='VERIFIED':raise ValueError('Encounter refinement needs the original verified physical background clock')
        pins[source['path']]=source['sha256'];pins[str(parent/'manifest.json')]=file_sha256(parent/'manifest.json')
        def physical(native):return float(interpolate_proper_time(scale,ages,native)*1000)
        def native(time):
            if not physical(begin)<=time<=physical(end):raise ValueError('Encounter refinement window leaves evolved physical clock')
            return float(brentq(lambda a:physical(a)-time,begin,end,xtol=1e-14))
        def error(native):return proper_time_error_at_epoch(source,native).get('absolute_bound_gyr')
    else:
        units={float(epoch['time_unit_gyr']) for epoch in frame['epochs']}
        if len(units)!=1:raise ValueError('Isolated encounter history changes native time units')
        factor=units.pop()*1000
        if not math.isfinite(factor) or factor<=0:raise ValueError('Positive actual native isolated time conversion required')
        def physical(native):return float(native*factor)
        def native(time):
            result=float(time/factor)
            if not begin<=result<=end:raise ValueError('Isolated encounter window leaves evolved native clock')
            return result
        def error(native):return 0.
    actual=np.asarray(values['times_myr']);native_values=[epoch['actual_scale_factor'] if comoving else epoch['native_time_code'] for epoch in frame['epochs']]
    if not np.allclose([physical(q) for q in native_values],actual,rtol=1e-12,atol=1e-9):raise ValueError('Encounter refinement physical and native clocks disagree')
    events={event['event_id']:event for event in encounters['events']};decisions=[];queued=[]
    if len(events)!=len(encounters['events']):raise ValueError('Encounter refinement event identities are duplicated')
    for row in encounters['event_branch_requests']:
        event=events[row['event_id']]
        if row['interval_myr']!=event['bracket_myr'] or row['targets']!=[event['stream_identity'],event['perturber_identity']]:
            raise ValueError('Refinement request changes the reconstructed event identity or bracket')
        lo,hi=map(float,row['interval_myr']);spacing=float(row['suggested_maximum_interval_myr'])
        if not np.isfinite([lo,hi,spacing]).all() or hi<=lo or spacing<=0:raise ValueError('Finite bounded physical refinement window required')
        a,b=native(lo),native(hi);bounds=[error(a),error(b)]
        decision={'event_id':event['event_id'],'interval_myr':[lo,hi],'geometry_error_target_kpc':row['target_geometry_error_kpc']}
        if any(q is None for q in bounds):decision['status']='UNBOUNDED_BACKGROUND_CLOCK';decisions.append(decision);continue
        clock_error=max(bounds)*1000;spacing-=2*clock_error
        if spacing<=0:decision['status']='REQUIRES_REFINED_BACKGROUND_CLOCK';decisions.append(decision);continue
        count=max(1,math.ceil((hi-lo)/spacing))
        if count+2>parameters['maximum_outputs']:decision['status']='OUTPUT_LIMIT_REACHED';decisions.append(decision);continue
        if len(queued)>=parameters['maximum_branches']:decision['status']='FOLLOWUP_LIMIT_REACHED';decisions.append(decision);continue
        available=choose_restart(parameters['checkpoint_directories'],parameters['ranks'],a,clock_representation='scale_factor' if comoving else 'native_time_L_over_V')
        decision['restart_availability']=available
        if available['status']!='AVAILABLE':decision['status']='REQUIRES_EARLIER_CHECKPOINT_OR_IC_RERUN';decisions.append(decision);continue
        requested=sorted(set([native(q) for q in np.linspace(lo,hi,count+1)]+[end]))
        schedule=merge_and_quantize_native_output_times(requested,[],clock,maximum_outputs=parameters['maximum_outputs'])
        measured=np.array([physical(q) for q in schedule['unique_planned_output_times']])
        # Quantization must retain both window boundaries and its requested
        # physical cadence. It never certifies trajectory geometry by itself.
        covered=measured[(measured>=lo-clock_error)&(measured<=hi+clock_error)]
        if len(covered)<2 or covered[0]>lo+clock_error or covered[-1]<hi-clock_error or np.max(np.diff(covered))+2*clock_error>float(row['suggested_maximum_interval_myr'])*(1+1e-12):
            decision['status']='REQUIRES_SMALLER_NATIVE_TIMESTEP';decisions.append(decision);continue
        branch={'run_dir':str(parent),'checkpoint_directory':available['directory'],'schedule':schedule,'event_ids':[event['event_id']],
            **{key:parameters[key] for key in ('ranks','omp_threads','wall_hours','memory_gib','disk_gib')}}
        path=context.generation/('branch-%03d.json'%len(queued));atomic_json(path,branch);queued.append(str(path))
        decision.update(status='PREPARED_VERIFIED_CHECKPOINT_BRANCH',request={'path':str(path),'sha256':file_sha256(path)},
            remaining_acceptance='Execute original native branch, remeasure actual outputs and obtain interpolation/geometry evidence; finer cadence alone does not certify dynamics')
        decisions.append(decision)
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Encounter refinement input changed')
    result={'schema':'native-encounter-refinement-plan-v1','status':'RECIPES_PREPARED','decisions':decisions,'branch_requests':queued,
        'source_hashes':pins,'science_acceptance':'NOT_ACCEPTED','execution_command':'commands.py event-cadence REQUEST FRESH_DESTINATION --execute-branch'}
    atomic_json(context.generation/'refinement-plan.json',result);return result



def prepare_native_encounter_campaign(request,destination):
    from ..workflow import compile_requested_product_dag,default_registry
    from core.stream_common import read_product
    from ..data import numeric_audit_source_bindings
    from ..halos import verify_identity_graph
    from .perturbers import validate_perturber_history
    required={'schema','run_directory','stream_history','identity_graph','perturber_template','frame','encounter_controls',
        'memory_gib','disk_gib','maximum_particles','maximum_nodes','maximum_tracks'}
    if not isinstance(request,dict) or set(request)-{'refinement'}!=required or request['schema']!='native-encounter-campaign-request-v1':
        raise ValueError('Complete native encounter campaign request required')
    memory=request['memory_gib'];validate_perturber_history(request)
    if type(request['disk_gib']) not in (int,float) or not math.isfinite(request['disk_gib']) or request['disk_gib']<=0:raise ValueError('Positive native encounter disk allocation required')
    pins={};inputs={}
    for name in ('stream_history','perturber_template'):
        binding=request[name]
        if not isinstance(binding,dict) or set(binding)!={'path','sha256'}:raise ValueError('Exact retained encounter input path/hash required')
        path=Path(binding['path']).resolve()
        if path.stat().st_size*16>memory*1024**3/8:raise MemoryError('Native encounter recipe metadata exceeds allocation')
        if file_sha256(path)!=binding['sha256']:raise ValueError('Retained native encounter input changed')
        pins[str(path)]=binding['sha256'];inputs[name]=path
    history,_,sources=read_product(inputs['stream_history'],memory_gib=memory/4)
    if history.get('schema')!='measured-native-stream-worldlines-v1' or history.get('execution_status')!='COMPLETE':
        raise ValueError('Original measured native stream history required')
    if Path(history['native_run_directory']).resolve()!=Path(request['run_directory']).resolve():raise ValueError('Native encounter history belongs to another run')
    if history.get('native_self_gravity_enabled') is not True:raise ValueError('Original self-gravitating native stream evolution required')
    if history.get('request',{}).get('species')!=4:raise ValueError('Native stream encounters require actual selected stellar histories')
    _merge_source_bindings(pins,sources)
    _merge_source_bindings(pins,numeric_audit_source_bindings(history))
    graph_path=Path(request['identity_graph']).resolve()
    if (graph_path/'report.json').stat().st_size*16>memory*1024**3/8:raise MemoryError('Native encounter graph metadata exceeds allocation')
    with zipfile.ZipFile(graph_path/'nodes.npz') as archive:
        if sum(row.file_size for row in archive.infolist())*4>memory*1024**3/8:raise MemoryError('Encounter graph archive exceeds intake allocation')
    graph=verify_identity_graph(graph_path)
    if len(graph['nodes'])>request['maximum_nodes']:raise MemoryError('Native encounter identity graph exceeds selected census')
    _merge_source_bindings(pins,graph['sources'])
    _merge_source_bindings(pins,{str(graph_path/name):file_sha256(graph_path/name) for name in ('report.json','index.sqlite','nodes.npz')})
    frame=copy.deepcopy(request['frame'])
    if frame.get('schema')!='native-encounter-frame-v1':raise ValueError('Explicit original saved-time encounter chart required')
    selections=frame.get('worldline_selections')
    if not isinstance(selections,list) or not selections:raise ValueError('Explicit exact selected stellar worldlines required by encounter reconstruction')
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep prior encounter campaign preparation')
    def ref(stage,artifact,load=None):
        value={'stage':stage,'artifact':artifact}
        if load:value['load']=load
        return value
    def product(stage,operation,source,artifact,options=None):
        parameters={'operation':operation,'product':source,'destination':'{generation}/'+artifact,'memory_gib':memory}
        parameters.update(options or {})
        return {'stage_id':stage,'kind':'flagship_product','parameters':parameters,'outputs':{stage:artifact}}
    stages=[]
    if history['periodic']:
        stages.append(product('stream-images','encounter-images',str(inputs['stream_history']),'images.json'))
        frame['recorded_images']={'path':ref('stream-images','images.json'),'sha256':ref('stream-images','images.json','sha256')}
    stages.append(product('stream-frame','encounter-frame',str(inputs['stream_history']),'frame.json',{'request':frame}))
    params={key:request[key] for key in ('memory_gib','maximum_particles','maximum_nodes','maximum_tracks')}
    params.update(run_dir=str(Path(request['run_directory']).resolve()),identity_graph=str(graph_path),
        stream_frame_history=ref('stream-frame','frame.json'),template_path=str(inputs['perturber_template']),
        template_sha256=request['perturber_template']['sha256'])
    stages.append({'stage_id':'perturber-history','kind':'native_perturber_history','parameters':params,'outputs':{'perturbers':'perturber-history.json'}})
    catalogue={'schema':'native-encounter-catalogue-v1','controls':copy.deepcopy(request['encounter_controls']),
        'perturber_history':{'path':ref('perturber-history','perturber-history.json'),'sha256':ref('perturber-history','perturber-history.json','sha256')}}
    stages.append(product('encounter-catalogue','encounter-catalogue',ref('stream-frame','frame.json'),'encounters.json',{'request':catalogue}))
    for stage,source in (('encounter-figures',ref('encounter-catalogue','encounters.json')),('perturber-figures',ref('perturber-history','perturber-history.json')),
            ('stream-worldline-figures',ref('stream-frame','frame.json'))):
        stages.append(product(stage,'encounter-figures',source,'figures'))
    objects=(len(selections)+request['maximum_tracks'])*len(history['epochs'])
    stages.append({'stage_id':'encounter-display','kind':'native_encounter_display','parameters':{
        'stream_history':ref('stream-frame','frame.json'),'perturber_history':ref('perturber-history','perturber-history.json'),
        'encounters':ref('encounter-catalogue','encounters.json'),'memory_gib':memory,'maximum_objects':objects,
        'maximum_vertices':2*objects,'frame_stride':1},'outputs':{'encounter-display':'encounter-display.json',
        'saved-display':'display','impact-display':'reconstructed-impacts.json','identity-map':'native-identity-map.json'}})
    requested=['encounter-figures','perturber-figures','stream-worldline-figures','encounter-display']
    refinement=request.get('refinement')
    if refinement is not None:
        keys={'checkpoint_directories','ranks','omp_threads','maximum_branches','maximum_outputs','maximum_timestep','wall_hours'}
        if not isinstance(refinement,dict) or set(refinement)!=keys:raise ValueError('Complete native encounter checkpoint refinement settings required')
        options={**refinement,'memory_gib':memory,'disk_gib':request['disk_gib'],'run_dir':str(Path(request['run_directory']).resolve()),
            'stream_history':ref('stream-frame','frame.json'),'encounters':ref('encounter-catalogue','encounters.json')}
        validate_encounter_refinement(options)
        stages.append({'stage_id':'encounter-refinement','kind':'native_encounter_refinement','parameters':options,
            'outputs':{'encounter-refinement':'refinement-plan.json'}})
        requested.append('encounter-refinement')
    # Dependency closure is derived by the existing compiler from producer
    # references. Bind all retained sources before starting any future stage.
    for stage in stages:stage['input_files']=[{'path':path,'sha256':digest} for path,digest in sorted(pins.items())]
    spec={'schema':'research-workflow-v1','resources':{'mpi_ranks':refinement['ranks'] if refinement else 1,'omp_threads':refinement['omp_threads'] if refinement else 1,'memory_gib':memory,'disk_gib':request['disk_gib'],'maximum_stage_count':len(stages)},
        'physics_spec':{},'capabilities':[],'followup_limits':{'maximum_children':0,'maximum_depth':0},
        'requested_products':requested,'stages':stages}
    registry=default_registry();plan=compile_requested_product_dag(spec,registry,request_base=out)
    code={}
    for stage in plan['stages']:code.update(registry.source_hashes(stage['kind']))
    code[str(Path(__file__).resolve())]=file_sha256(__file__)
    for path,digest in {**pins,**code}.items():
        if file_sha256(path)!=digest:raise ValueError('Native encounter campaign inputs changed')
    out.mkdir(parents=True);atomic_json(out/'request.json',request);atomic_json(out/'workflow.json',spec)
    result={'schema':'native-encounter-campaign-v1','status':'RECIPES_PREPARED','workflow':{'path':str(out/'workflow.json'),'sha256':file_sha256(out/'workflow.json')},
        'source_hashes':{**pins,**code},'stage_count':len(plan['stages']),'science_acceptance':'NOT_ACCEPTED',
        'remaining_requirements':['Native source and interval support are checked at execution; unavailable detections remain unresolved.',
            'Event refinement requests require a verified earlier native checkpoint and explicit branch resources.',
            'Matched native intervention responses are a separate registered producer, not manufactured by encounter geometry.']}
    atomic_json(out/'manifest.json',result);return result


# Recover fixed stellar cohorts across actual saved epochs.

def validate_recovered_encounters(parameters):
    m=parameters['memory_gib']
    if type(m) not in (int,float) or not np.isfinite(m) or m<.1:raise ValueError('Bounded recovered encounter workspace required')



def validate_recovered_encounter_template(template):
    required={'schema','snapshots','maximum_cohorts','maximum_selected_particles','maximum_census_particles','maximum_nodes','maximum_tracks',
        'frame','identity_graph','perturber_template','encounter_controls','refinement'}
    if not isinstance(template,dict) or set(template)!=required or template['schema']!='recovered-native-encounters-v1':
        raise ValueError('Complete recovered native encounter template required')
    snapshots=template['snapshots']
    if not isinstance(snapshots,list) or len(snapshots)<2 or any(type(s) is not int or s<0 for s in snapshots) or len(set(snapshots))!=len(snapshots):
        raise ValueError('At least two distinct actual saved encounter epochs required')
    for name in ('maximum_cohorts','maximum_selected_particles','maximum_census_particles','maximum_nodes','maximum_tracks'):
        if type(template[name]) is not int or template[name]<1:raise ValueError('Exact positive recovered encounter '+name+' required')
    for name in ('identity_graph','perturber_template'):
        binding=template[name]
        if binding is not None and (not isinstance(binding,dict) or set(binding)!={'path','sha256'}):raise ValueError('Exact source-bound '+name+' or explicit unavailable null required')
    frame=template['frame']
    if not isinstance(frame,dict) or set(frame)!={'frame','origin_proper_kpc','origin_velocity_peculiar_kms','maximum_interval_myr'}:
        raise ValueError('Complete fixed native chart and saved-time origin trajectory required')
    for name in ('origin_proper_kpc','origin_velocity_peculiar_kms'):
        array=np.asarray(frame[name])
        if array.shape!=(len(snapshots),3) or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():raise ValueError('Actual saved encounter frame origin/velocity required')
    if not isinstance(frame['frame'],str) or not frame['frame'] or type(frame['maximum_interval_myr']) not in (int,float) or not np.isfinite(frame['maximum_interval_myr']) or frame['maximum_interval_myr']<=0:
        raise ValueError('Explicit fixed encounter frame name and positive physical cadence limit required')
    return template



def _load(parameters):
    validate_recovered_encounters(parameters);memory=parameters['memory_gib'];pins={}
    metadata_bytes=0
    def read(path,expected=None):
        nonlocal metadata_bytes
        path=Path(path).resolve()
        if path.stat().st_size*16>memory*1024**3/16:raise MemoryError('Recovered encounter metadata exceeds intake allocation')
        metadata_bytes+=path.stat().st_size*16
        if metadata_bytes>memory*1024**3/4:raise MemoryError('Retained recovered encounter metadata exceeds total workspace')
        actual=file_sha256(path)
        if expected is not None and actual!=expected:raise ValueError('Recovered encounter input changed')
        value=json.loads(path.read_text());pins[str(path)]=actual
        for source,digest in value.get('source_hashes',{}).items():
            if file_sha256(source)!=digest:raise ValueError('Recovered encounter source changed')
            if source in pins and pins[source]!=digest:raise ValueError('Recovered encounter sources conflict')
            pins[source]=digest
        return value
    template=validate_recovered_encounter_template(read(parameters['template_path'],parameters['template_sha256']))
    history=read(parameters['stream_measurement_history'])
    if history.get('schema')!='native-recovered-stream-measurement-history-v1':raise ValueError('Actual recovered stream measurement history required')
    epochs={}
    for row in history['records']:
        snapshot=row['snapshot']
        if type(snapshot) is not int or snapshot<0 or snapshot in epochs:raise ValueError('Distinct exact recovered stream saved labels required')
        epochs[snapshot]=None if row.get('product') is None else read(row['product']['path'],row['product']['sha256'])
        if epochs[snapshot] is not None and epochs[snapshot].get('schema')!='native-catalogue-stream-measurement-result-v1':raise ValueError('Original per-epoch native stream measurement result required')
    if not set(template['snapshots'])<=set(epochs):raise ValueError('Recovered encounter request loses an actual stream measurement epoch')
    for name in ('identity_graph','perturber_template'):
        binding=template[name]
        if binding is not None:
            path=Path(binding['path']).resolve()
            if name=='identity_graph':path=path/'report.json'
            value=read(path,binding['sha256'])
            if name=='identity_graph':
                from ..halos import verify_identity_graph
                import zipfile
                with zipfile.ZipFile(path.parent/'nodes.npz') as archive:
                    if sum(row.file_size for row in archive.infolist())*4>memory*1024**3/8:raise MemoryError('Recovered encounter graph arrays exceed workspace')
                verified=verify_identity_graph(path.parent)
                for asset in ('report.sha256','index.sqlite','nodes.npz'):pins[str(path.parent/asset)]=file_sha256(path.parent/asset)
                _merge_source_bindings(pins,verified['sources'])
            else:
                for track in value.get('tracks',[]):
                    certificate=track.get('interpolation_certificate')
                    if isinstance(certificate,dict):
                        for source,digest in certificate.get('source_hashes',{}).items():
                            if file_sha256(source)!=digest:raise ValueError('Recovered perturber interpolation source changed')
                            pins[source]=digest
    cohorts={};count=0
    for snapshot in template['snapshots']:
        selected=[];epoch=epochs[snapshot]
        for row in [] if epoch is None else epoch['candidates']:
            value=_cohort(row,memory,pins,snapshot)
            if value is not None:
                count+=len(value['ids'])
                if count*512>memory*1024**3/4:raise MemoryError('Recovered exact cohort identity joins exceed total workspace')
                selected.append(value)
        cohorts[snapshot]=selected
    return template,cohorts,pins



def recovered_encounter_input_bindings(parameters):return _load(parameters)[2]



def _cohort(row,memory,pins,snapshot):
    from ..sidm_analysis import _load_product
    selection=row['selection']
    if selection.get('status')!='SUPPORTED_SELECTION' or not row['products'].get('coordinates'):return None
    product=row['products']['coordinates'];context=row['products']['selection_context']
    def merge(sources):
        for path,digest in sources.items():
            if path in pins and pins[path]!=digest:raise ValueError('Recovered native cohort source generations conflict')
            pins[path]=digest
    report,arrays,sources=_load_product(product,'.',memory*1024**3/16);merge(sources)
    sidecar,values,sources=_load_product(context,'.',memory*1024**3/16);merge(sources)
    if report.get('schema')!='stream-coordinates-v1' or sidecar.get('schema')!='native-stellar-selection-context-v1':
        raise ValueError('Original selected stream coordinates and exact native selection sidecar required')
    if report.get('epoch',{}).get('snapshot_storage_index')!=snapshot or sidecar.get('epoch',{}).get('snapshot_storage_index')!=snapshot:
        raise ValueError('Recovered native cohort belongs to another actual saved epoch')
    ids=arrays['particle_ids'];other=values['particle_ids']
    # Both sides must stay integer-valued before comparison: NumPy can compare
    # uint64 with float64 after rounding away low ID bits above 2**53.
    if (ids.dtype!=np.dtype('uint64') or other.dtype!=np.dtype('uint64')
            or ids.ndim!=1 or other.shape!=ids.shape or not len(ids)
            or len(np.unique(ids))!=len(ids) or len(np.unique(other))!=len(other)
            or not np.array_equal(np.sort(ids),np.sort(other))):
        raise ValueError('Native encounter cohort loses exact distinct selected particle identities')
    if report['namespace']!=sidecar['namespace']:raise ValueError('Native encounter selection changes particle namespace')
    labels=np.asarray(values['progenitor_label'])
    if labels.shape!=ids.shape or labels.dtype.kind not in 'iu' or np.any(labels<0):raise ValueError('Exact known native progenitor membership required')
    lookup={}
    for entry in sidecar['progenitor_identity_catalogue']:
        label=entry['label'];namespace=entry['namespace'];identity=entry['identity_decimal']
        if (type(label) is not int or label<0 or label in lookup
                or not isinstance(namespace,str) or not namespace
                or not isinstance(identity,str) or not identity.isdecimal()
                or not 0<=int(identity)<2**64):
            raise ValueError('Distinct labelled namespace-qualified uint64 progenitor identities required')
        lookup[label]=(namespace,str(int(identity)))
    if any(int(label) not in lookup for label in labels):
        raise ValueError('Selected native progenitor label has no exact identity')
    for key in ('ancestry_known','binding_known'):
        flags=np.asarray(values[key])
        if flags.dtype.kind!='b' or flags.shape!=other.shape or not flags.all():return None
    brackets=np.asarray(values['stripping_bracket_available'])
    if brackets.dtype.kind!='b' or brackets.shape!=other.shape or not brackets.any():return None
    membership=tuple(sorted((int(pid),*lookup[int(label)]) for pid,label in zip(other,labels)))
    identities=tuple(sorted({lookup[int(label)] for label in labels}))
    return {'namespace':report['namespace'],'ids':np.sort(ids),'progenitors':identities,'membership':membership,'candidate_id':row['candidate_id']}



def recovered_native_encounters_runner(parameters,context):
    from ..stream_forward_model import publish_native_stream_worldlines
    from ..workflow import execute_workflow
    validate_recovered_encounters(parameters);memory=parameters['memory_gib']
    if memory>context.resources['memory_gib']:raise MemoryError('Recovered native encounter intake exceeds workflow allocation')
    template,candidates,pins=_load(parameters);snapshots=template['snapshots'];records=[]
    seeds=candidates[snapshots[0]]
    if len(seeds)>template['maximum_cohorts']:raise MemoryError('Recovered native encounter cohort count exceeds frozen allocation')
    for index,seed in enumerate(seeds):
        record={'reference_candidate_id':seed['candidate_id'],'namespace':seed['namespace'],'particle_count':len(seed['ids']),
            'progenitor_keys':[list(key) for key in seed['progenitors']],'snapshots':snapshots,'status':'UNRESOLVED'}
        if len(seed['ids'])>template['maximum_selected_particles']:raise MemoryError('Native encounter cohort exceeds exact selected census bound')
        continuity=[]
        fixed_membership=set(seed['membership'])
        for snapshot in snapshots:
            # A growing stream may include newly stripped stars. Keep the
            # original complete cohort unchanged, rather than requiring the
            # entire later catalogue population to stay artificially fixed.
            found=[value for value in candidates[snapshot] if value['namespace']==seed['namespace'] and fixed_membership<=set(value['membership'])]
            continuity.append({'snapshot':snapshot,'matched_candidate_ids':[value['candidate_id'] for value in found],
                'additional_catalogue_particles':[len(value['ids'])-len(seed['ids']) for value in found]})
        record['cohort_continuity']=continuity
        if any(len(row['matched_candidate_ids'])!=1 for row in continuity):
            record['reason']='Exact calibrated cohort/progenitor selection is missing or ambiguous at one or more saved epochs; no row matching, identity intersection or membership reassignment'
            records.append(record);continue
        if template['identity_graph'] is None or template['perturber_template'] is None:
            record['reason']='Native perturber identity graph or recorded trajectory/profile template unavailable; no perturber invented'
            records.append(record);continue
        folder=context.generation/('cohort-%04d'%index);folder.mkdir()
        worldlines=folder/'native-stream-worldlines.json'
        request={'schema':'native-stream-worldline-request-v1','snapshots':snapshots,'species':4,'particle_ids':[int(pid) for pid in seed['ids']],
            'particle_namespace':seed['namespace'],'maximum_snapshots':len(snapshots),'maximum_selected_particles':template['maximum_selected_particles'],
            'maximum_census_particles':template['maximum_census_particles'],'memory_gib':memory}
        raw=publish_native_stream_worldlines({'request':request,'native_stage_receipt':parameters['native_stage_receipt'],
            'native_run_directory':parameters['run_dir']},worldlines)
        for source,digest in raw['source_hashes'].items():
            if source in pins and pins[source]!=digest:raise ValueError('Native worldline source conflicts with recovered stream selection')
            pins[source]=digest
        if raw['periodic']:
            from ..periodic_images import publish_native_worldline_images
            images=folder/'native-image-support.json'
            support=publish_native_worldline_images(worldlines,images,memory_gib=memory)
            record['native_image_support']={'path':str(images),'sha256':file_sha256(images)}
            if support['status']!='PASS':
                record['reason']='Recorded native periodic image support is unavailable or discontinuous; no temporal winding inferred'
                records.append(record);continue
        frame={'schema':'native-encounter-frame-v1',**copy.deepcopy(template['frame']),
            'worldline_selections':[{'particle_id':int(pid),'namespace':seed['namespace'],
                'valid_intervals':[True]*(len(snapshots)-1)} for pid in seed['ids']]}
        # Exact native star presence establishes saved endpoints only. No
        # interpolation certificate is fabricated; the reconstruction retains
        # cadence masks and emits refinement requests for uncertified geometry.
        campaign={'schema':'native-encounter-campaign-request-v1','run_directory':parameters['run_dir'],
            'stream_history':{'path':str(worldlines),'sha256':file_sha256(worldlines)},
            'identity_graph':template['identity_graph']['path'],'perturber_template':template['perturber_template'],
            'frame':frame,'encounter_controls':template['encounter_controls'],'refinement':template['refinement'],
            'memory_gib':memory,'disk_gib':context.resources['disk_gib'],
            **{name:template[name] for name in ('maximum_nodes','maximum_tracks')},'maximum_particles':template['maximum_census_particles']}
        prepared=prepare_native_encounter_campaign(campaign,folder/'prepared')
        spec=json.loads(Path(prepared['workflow']['path']).read_text())
        result=execute_workflow(spec,folder/'measurements',request_base=folder/'prepared')
        record.update(status=result['execution_status'],native_stream_history={'path':str(worldlines),'sha256':file_sha256(worldlines)},
            prepared_campaign={'path':str(folder/'prepared/manifest.json'),'sha256':file_sha256(folder/'prepared/manifest.json')},
            workflow_receipt={'path':str(folder/'measurements/workflow.json'),'sha256':file_sha256(folder/'measurements/workflow.json')})
        record['products']={}
        for stage in spec['stages']:
            generation=Path(result['stages'][stage['stage_id']]['generation'])
            for name,relative in stage['outputs'].items():
                artifact=generation/relative
                record['products'][name]={'path':str(artifact),'sha256':file_sha256(artifact)} if artifact.is_file() else {'path':str(artifact),'kind':'directory'}
        records.append(record)
    if not seeds:records.append({'status':'UNRESOLVED','reason':'No supported calibrated native stellar cohort at the first requested saved epoch; no encounter population fabricated','snapshots':snapshots})
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Recovered native encounter input changed')
    result={'schema':'native-recovered-encounter-history-v1','records':records,'source_hashes':pins,'science_acceptance':'NOT_ACCEPTED',
        'status':'MEASURED_WITH_SUPPORT_MASKS' if any(row['status']=='COMPLETE' for row in records) else 'UNRESOLVED',
        'identity_policy':'Complete reference-epoch stellar cohort retained with exact per-ID progenitor association at every saved epoch. Later catalogue additions remain outside this unchanged cohort. Candidate row numbers never establish continuity.'}
    atomic_json(context.generation/'encounter-history.json',result);return result

