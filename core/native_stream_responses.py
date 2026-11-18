"""Native paired histories → existing fixed-track response estimators.

The initial selected cohort, clock and chart stay fixed across branches. These
are conditional paired measurements; density deficits, transverse offsets and
action changes do not establish a detected gap or a unique causal perturber.
"""
import copy
import json
import tempfile
from pathlib import Path
import numpy as np
from .foundation import atomic_json,file_sha256

VARIANTS=('before_after_controls','gap_growth','action_diffusion')


def _request(parameters):
    memory=parameters['memory_gib']
    if type(memory) not in (int,float) or not np.isfinite(memory) or memory<.1:raise ValueError('Bounded response memory required')
    path=Path(parameters['request_path']).resolve()
    if path.stat().st_size*16>memory*1024**3/16:raise MemoryError('Response request exceeds bounded intake')
    if file_sha256(path)!=parameters['request_sha256']:raise ValueError('Frozen response request changed')
    request=json.loads(path.read_text())
    keys={'schema','comparison','reference_frame','intervention_frame','track_epoch_index','curve_controls','bin_fractions',
        'measurement_controls','model_error_covariance','bootstrap_replicates','seed','feature_controls','actions','export_figures','figure_dpi'}
    if set(request)!=keys or request['schema']!='native-stream-response-request-v1':raise ValueError('Complete frozen native paired response request required')
    if type(request['track_epoch_index']) is not int or request['track_epoch_index']<0:raise ValueError('Exact saved track epoch ordinal required')
    fractions=np.asarray(request['bin_fractions'],float)
    if fractions.ndim!=1 or len(fractions)<3 or not np.isfinite(fractions).all() or fractions[0]<=0 or fractions[-1]>=1 or np.any(np.diff(fractions)<=0):
        raise ValueError('Resolved bin fractions strictly inside the supported reference-track domain required')
    if set(request['measurement_controls'])!={'projection_chunk_rows','minimum_bin_particles','label_resolution_kpc'}:
        raise ValueError('Complete existing fixed-track measurement controls required')
    if set(request['feature_controls'])!={'minimum_prominence','minimum_separation_kpc'}:raise ValueError('Explicit existing feature-selection controls required')
    if type(request['export_figures']) is not bool or type(request['figure_dpi']) is not int or not 32<=request['figure_dpi']<=2400:
        raise ValueError('Bounded registered response export settings required')
    actions=request['actions']
    if actions is not None and (not isinstance(actions,dict) or set(actions)!={'potential','potential_domain','controls','frame_definition_sha256'} or actions['potential'].get('kind') not in {'coefficients','time_dependent_host'}):
        raise ValueError('Source-bound measured host action inputs or explicit unavailable null required')
    return request,path


def validate_native_response(parameters):
    if type(parameters['memory_gib']) not in (int,float) or not np.isfinite(parameters['memory_gib']) or parameters['memory_gib']<.1:
        raise ValueError('Bounded positive native response memory required')


def _load(parameters,*,replay):
    from .sidm_analysis import _load_product
    from .data import proper_position_derivative_kms,apply_recorded_periodic_images
    from .stream_forward_model import publish_native_stream_comparison
    from .flagship_products import require_native_worldline_units
    request,path=_request(parameters);memory=parameters['memory_gib'];pins={str(path):parameters['request_sha256']}
    def load(binding,base=path.parent):
        report,arrays,sources=_load_product(binding,base,memory*1024**3/8)
        for name,digest in sources.items():
            if name in pins and pins[name]!=digest:raise ValueError('Conflicting paired native source generations')
            pins[name]=digest
        return report,arrays
    paired,contrast=load(request['comparison'])
    if paired.get('schema')!='measured-native-stream-paired-comparison-v1' or paired.get('execution_status')!='COMPLETE':
        raise ValueError('Original completed native paired comparison required')
    original=paired['request'];original_base=Path(paired['request_base']).resolve()
    if original['memory_gib']>memory:raise MemoryError('Original native paired comparison exceeds response allocation')
    frames=[];raws=[]
    for name in ('reference','intervention'):
        raw,values=load(original[name],original_base);frame,converted=load(request[name+'_frame'])
        if (raw.get('schema')!='measured-native-stream-worldlines-v1' or raw.get('execution_status')!='COMPLETE'
                or raw.get('native_self_gravity_enabled') is not True):raise ValueError('Actual self-gravitating native history required')
        if raw.get('request',{}).get('species')!=4:raise ValueError('Native stellar responses require the original exact selected stellar species')
        require_native_worldline_units(raw)
        if frame.get('schema')!='native-encounter-frame-history-v1' or frame.get('source_history_sha256')!=original[name]['sha256']:
            raise ValueError('Response frame does not bind the original paired native history')
        if Path(frame['source_history']).resolve()!=(original_base/original[name]['path']).resolve():raise ValueError('Response frame changes its original native history path')
        if frame['epochs']!=raw['epochs'] or frame['namespace']!=raw['namespace']:raise ValueError('Response frame changes native epochs or namespace')
        for key in ('particle_ids','times_myr','mass_msun','periodic_box_proper_kpc'):
            if not np.array_equal(values[key],converted[key]):raise ValueError('Response frame changes native cohort/clock/mass: '+key)
        x=values['position_kpc']
        if raw['periodic']:
            if frame.get('periodic_images_applied') is not True:raise ValueError('Native response lacks recorded periodic images')
            x=apply_recorded_periodic_images(x,converted['recorded_image_offsets'],box_proper_kpc_by_epoch=values['periodic_box_proper_kpc'])
        velocity=np.array([proper_position_derivative_kms(x[k],values['velocity_kms'][k],hubble_kms_mpc=converted['hubble_kms_mpc'][k],
            origin_proper_kpc=converted['origin_proper_kpc'][k],origin_velocity_peculiar_kms=converted['origin_velocity_peculiar_kms'][k]) for k in range(len(x))])
        if not np.array_equal(x-converted['origin_proper_kpc'][:,None,:],converted['position_kpc']) or not np.array_equal(velocity,converted['velocity_kms']):
            raise ValueError('Response frame is not the recorded native position derivative conversion')
        units=frame['numeric_data']['array_contract']['datasets']
        for key,unit in (('position_kpc','proper kpc'),('velocity_kms','km / s'),('mass_msun','Msun'),('times_myr','Myr'),('particle_ids','identity')):
            if units[key]['units']!=unit:raise ValueError('Response frame unit mismatch: '+key)
        frames.append((frame,converted));raws.append((raw,values))
    left,right=frames
    if left[0]['frame_definition_sha256']!=right[0]['frame_definition_sha256'] or left[0]['frame']!=right[0]['frame']:
        raise ValueError('Paired native responses require the same clock-bound origin and fixed axes')
    if request['actions'] is not None and request['actions']['frame_definition_sha256']!=left[0]['frame_definition_sha256']:
        raise ValueError('Measured action field must declare the same physical phase-space frame as the paired responses')
    for key in ('particle_ids','times_myr'):
        if not np.array_equal(left[1][key],right[1][key]):raise ValueError('Native response loses exact paired identity/epoch order')
    for key in ('origin_proper_kpc','origin_velocity_peculiar_kms','hubble_kms_mpc'):
        if not np.array_equal(left[1][key],right[1][key]):raise ValueError('Native response changes the frozen physical chart: '+key)
    if sum(a['position_kpc'].size for _,a in frames)*512>memory*1024**3:raise MemoryError('Paired response workspace exceeds allocation')
    if replay:
        # Reuse the existing immutable-population, physical-epoch, box,
        # native-execution and initial-state gates instead of accepting labels.
        with tempfile.TemporaryDirectory(prefix='native-paired-response-') as temp:
            dest=Path(temp)/'paired.json'
            publish_native_stream_comparison({'request':original,'request_base':str(original_base)},dest)
            from core.stream_common import read_product
            _,replayed,_=read_product(dest,memory_gib=memory/8)
            if set(replayed)!=set(contrast) or any(not np.array_equal(replayed[k],contrast[k],equal_nan=True) for k in contrast):
                raise ValueError('Original native paired comparison does not replay')
    return request,path,paired,frames,pins


def native_response_input_bindings(parameters):
    request,path,_,frames,pins=_load(parameters,replay=False)
    if request['actions'] is not None:
        from core.stream.workflow import Inputs,_potential
        inputs=Inputs(path.parent,{}, {'memory_gib':parameters['memory_gib']/4,'maximum_particles':len(frames[0][1]['particle_ids'])})
        _potential(inputs,request['actions']['potential']);pins.update(inputs.sources)
    return pins


def paired_action_evolution(reference,intervention,reference_support,intervention_support):
    """Conditional squared paired changes; never a stationary diffusion fit."""
    left=np.asarray(reference);right=np.asarray(intervention);lm=np.asarray(reference_support);rm=np.asarray(intervention_support)
    if left.ndim!=3 or left.shape[-1]!=3 or not left.shape[0] or right.shape!=left.shape or lm.shape!=left.shape[:2] or rm.shape!=lm.shape or lm.dtype.kind!='b' or rm.dtype.kind!='b':
        raise ValueError('Aligned action components and exact boolean host-action support required')
    delta=right-left;support=lm&rm
    support&=support[0][None,:]&np.isfinite(delta).all(axis=2)&np.isfinite(delta[0]).all(axis=1)[None,:]
    centered=delta-delta[0][None,:,:];mean_square=np.full((len(left),3),np.nan)
    for ordinal,mask in enumerate(support):
        if mask.any():mean_square[ordinal]=np.mean(centered[ordinal,mask]**2,axis=0)
    return delta,support,mean_square


def native_response_runner(parameters,context):
    from core.stream_tracks import fit_uncertainty_aware_principal_curve
    from .stream_forward_model import TrackCoordinates,measure_track_density_velocity,compare_control_effects
    from core.stream.encounters import measure_multiple_features
    from core.stream_common import publish_stream_product
    from core.stream.workflow import Inputs, _potential
    from core.stream_common import stream_array_units
    from core.action_angle import measure_actions
    validate_native_response(parameters);memory=parameters['memory_gib']
    if memory>context.resources['memory_gib']:raise MemoryError('Native responses exceed workflow allocation')
    request,path,paired,frames,pins=_load(parameters,replay=True)
    left,right=[a for _,a in frames];times=left['times_myr'];ids=left['particle_ids'];weights=left['mass_msun'][0]
    k=request['track_epoch_index']
    if k>=len(times):raise ValueError('Reference-track epoch is not actually saved')
    sample={'position_kpc':left['position_kpc'][k],'velocity_kms':left['velocity_kms'][k],
        'mass_msun':weights,'particle_ids':ids,'namespace':frames[0][0]['namespace']}
    grid=request['curve_controls']['grid_points'];neighbors=request['curve_controls']['ordering_neighbors'];chunk=request['measurement_controls']['projection_chunk_rows']
    if any(type(x) is not int or x<1 for x in (grid,neighbors,chunk)):raise ValueError('Exact positive existing track workspace controls required')
    workspace=len(ids)*min(neighbors,len(ids))*128+grid*chunk*128+len(times)*(len(ids)+len(request['bin_fractions']))*2048
    if workspace>memory*1024**3:raise MemoryError('Fixed track, moment and paired response workspace exceeds allocation')
    report={'schema':'native-stream-responses-v1','request':request,'paired_comparison':request['comparison'],
        'namespace':sample['namespace'],'initial_state_sha256':paired['initial_state_sha256'],'frame':frames[0][0]['frame'],
        'frame_definition_sha256':frames[0][0]['frame_definition_sha256'],'epochs':frames[0][0]['epochs'],
        'source_origin':'synthetic_control' if any(f.get('fixture') or f.get('source_origin')=='synthetic_control' for f,_ in frames) else 'native_snapshot_and_catalogue',
        'science_acceptance':'NOT_ACCEPTED','variants':{v:{'status':'UNRESOLVED'} for v in VARIANTS},
        'population_scope':'Original exact native selected stellar cohort; no additional detected-stream claim',
        'weight_definition':'Frozen original gravitating masses for fixed-track moments; existing paired bootstrap uses unweighted particle means',
        'velocity_definition':'Proper position derivative in the shared fixed Cartesian chart, including the recorded H*r term; action field must bind this same frame',
        'causal_intervention_status':paired['causal_intervention_status'],'density_definition':'intervention/reference frozen-cohort line density; not an automatically detected causal gap'}
    arrays={'particle_ids':ids,'times_myr':times}
    try:
        curve=fit_uncertainty_aware_principal_curve(sample,request['curve_controls'])
        chosen=np.flatnonzero(curve.support)
        report['reference_curve_fit']=curve.diagnostics
        if len(chosen)<2 or np.any(np.diff(chosen)!=1):raise ValueError('Reference principal track lacks contiguous converged derivative support')
        track=TrackCoordinates({'position_kpc':curve.position[chosen],'origin':frames[0][0]['frame']})
        if np.any(track.reversals):raise ValueError('Reference track contains unresolved orientation reversals')
    except (ValueError,ArithmeticError) as error:
        report.update(status='UNRESOLVED',reason=str(error))
    else:
        controls={**request['measurement_controls'],'bin_edges_kpc':(np.asarray(request['bin_fractions'])*track.arc[-1]).tolist()}
        histories=[]
        for name,values in zip(('reference','intervention'),(left,right)):
            measured=measure_track_density_velocity(values,track,controls,particle_weights=weights)
            histories.append({**values,**measured})
            for key in ('linear_density_msun_kpc','centroid_kpc','width_covariance_kpc2','velocity_mean_kms','measurement_supported','bin_count'):
                arrays[name+'_'+key]=measured[key]
        comparison,delta=compare_control_effects(*histories,model_error_covariance=request['model_error_covariance'],
            bootstrap_replicates=request['bootstrap_replicates'],seed=request['seed'])
        arrays.update(delta);arrays['bin_center_kpc']=histories[0]['bin_center_kpc'];arrays['bin_edges_kpc']=histories[0]['bin_edges_kpc']
        arrays['reference_track_position_kpc']=track.vertices
        report.update(status='MEASURED_WITH_SUPPORT_MASKS',paired_response=comparison,track_epoch_index=k,
            track_policy='One reference-branch supported track frozen for every epoch and branch; bins strictly interior avoid endpoint-clipped particles')
        # Retain the original paired bin mask before morphology selection.
        supported=delta['gap_supported'];ratio=np.where(supported,delta['density_ratio'],np.nan)
        offset=np.where(supported[:,:,None],delta['delta_centroid_kpc'],np.nan)
        report['features']=[{'saved_epoch_ordinal':i,**measure_multiple_features(ratio[i],offset[i],arrays['bin_edges_kpc'],**request['feature_controls'])} for i in range(len(times))]
        arrays['density_ratio']=ratio;arrays['density_deficit']=1-ratio
        report['variants']['before_after_controls']={'status':'MEASURED' if supported.any() else 'UNRESOLVED','supported_entries':int(supported.sum())}
        report['variants']['gap_growth']={**report['variants']['before_after_controls'],'definition':'Saved-time signed density deficits on the same fixed bins; no feature-identity or monotonic growth assumption'}
        action_delta=np.full((len(times),len(ids),3),np.nan);action_support=np.zeros((len(times),len(ids)),bool)
        mean_square=np.full((len(times),3),np.nan)
        if request['actions'] is not None:
            actions=request['actions'];inputs=Inputs(path.parent,{}, {'memory_gib':memory/4,'maximum_particles':len(ids)})
            potential,agama_potential=_potential(inputs,actions['potential']);pins.update(inputs.sources)
            products=[];report['action_measurements']=[]
            for values in (left,right):
                rows=[];masks=[];details=[]
                for ordinal,time in enumerate(times):
                    controls=copy.deepcopy(actions['controls']);controls['epoch_time_myr']=float(time)
                    state={**sample,'position_kpc':values['position_kpc'][ordinal],'velocity_kms':values['velocity_kms'][ordinal]}
                    detail,values_at_time=measure_actions(state,potential,actions['potential_domain'],controls,agama_potential=agama_potential);details.append(detail)
                    rows.append(values_at_time['actions_kpc_kms']);masks.append(values_at_time['action_supported'])
                products.append((np.asarray(rows),np.asarray(masks)))
                report['action_measurements'].append(details)
            action_delta,action_support,mean_square=paired_action_evolution(products[0][0],products[1][0],products[0][1],products[1][1])
        arrays.update(paired_action_change_kpc_kms=action_delta,action_supported=action_support,paired_action_mean_square_kpc2_kms2=mean_square,
            action_epoch_supported=action_support.any(axis=1))
        report['variants']['action_diffusion']={'status':'MEASURED' if action_support.any() else 'UNRESOLVED',
            'definition':'Mean squared change of paired action differences relative to the first saved epoch; conditional on common measured host action support, not a stationary diffusion coefficient',
            'reason':None if request['actions'] is not None else 'Measured host action inputs unavailable'}
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    dest=context.generation/'stream-responses.json'
    result=publish_stream_product(dest,report,arrays,stream_array_units(arrays),sources=pins,namespace=sample['namespace'],epoch={'actual_saved_epochs':report['epochs']})
    if request['export_figures']:
        export_native_response_figures(dest,context.generation/'response-figures',memory_gib=memory/2,dpi=request['figure_dpi'])
    return result


def export_native_response_figures(product,destination,*,memory_gib=1.,dpi=160):
    from core.stream_common import read_product
    from .data import export_numeric_audit,numeric_audit_source_bindings
    from core.research.display import export_scientific_suite
    report,arrays,pins=read_product(product,memory_gib=memory_gib/3);pins.update(numeric_audit_source_bindings(report))
    if report.get('schema')!='native-stream-responses-v1':raise ValueError('Original native response product required')
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep prior response figure exports')
    out.mkdir(parents=True);sources={};figures=[]
    for variant in VARIANTS:
        fields={'time':arrays['times_myr']};specs={'time':{'units':'Myr','quantity':'actual saved proper clock'}};series=[]
        panels=[]
        if variant=='before_after_controls' and 'bin_center_kpc' in arrays:
            fields['s']=arrays['bin_center_kpc'];specs['s']={'units':'proper kpc','quantity':'frozen reference track bin center'}
            density=[];spur=[];velocity=[]
            for ordinal in sorted({0,len(arrays['times_myr'])-1}):
                mask='support_'+str(ordinal);fields[mask]=arrays['gap_supported'][ordinal]
                specs[mask]={'units':'dimensionless','quantity':'common paired native bin support'}
                label='t='+format(float(arrays['times_myr'][ordinal]),'.6g')+' Myr'
                for branch in ('reference','intervention'):
                    key=branch+'_'+str(ordinal);fields[key]=arrays[branch+'_linear_density_msun_kpc'][ordinal]
                    specs[key]={'units':'Msun / proper kpc','quantity':'frozen initial mass weighted line density'}
                    density.append({'source':variant,'kind':'line','label':branch+' '+label,'x':'s','y':key,'support':mask})
                for j in range(3):
                    key='velocity_'+str(ordinal)+'_'+str(j);fields[key]=arrays['delta_velocity_mean_kms'][ordinal,:,j]
                    specs[key]={'units':'km/s','quantity':'paired fixed-track local velocity change'}
                    velocity.append({'source':variant,'kind':'line','label':'component '+str(j)+' '+label,'x':'s','y':key,'support':mask})
                for j in range(2):
                    key='spur_'+str(ordinal)+'_'+str(j);fields[key]=arrays['delta_centroid_kpc'][ordinal,:,j]
                    specs[key]={'units':'proper kpc','quantity':'paired transverse centroid offset; no automatic spur detection'}
                    spur.append({'source':variant,'kind':'line','label':'component '+str(j)+' '+label,'x':'s','y':key,'support':mask})
            for name,title,unit,group in (('density','Matched line densities','Msun / proper kpc',density),
                    ('spur','Matched transverse offsets','proper kpc',spur),('velocity','Matched local velocity changes','km/s',velocity)):
                panels.append({'panel_id':name,'title':title,'quality_status':'PENDING_CONVERGENCE','selection':report['population_scope'],
                    'xlabel':'Frozen reference arclength [proper kpc]','ylabel':title+' ['+unit+']','x_units':'proper kpc','y_units':unit,'series':group})
            series=density
        elif variant=='gap_growth' and 'bin_center_kpc' in arrays:
            for j in range(len(arrays['bin_center_kpc'])):
                key='bin_'+str(j);mask=key+'_supported';fields[key]=arrays['density_deficit'][:,j];fields[mask]=arrays['gap_supported'][:,j]
                specs[key]={'units':'dimensionless','quantity':'signed intervention/reference density deficit'};specs[mask]={'units':'dimensionless','quantity':'original paired bin support'}
                series.append({'source':variant,'kind':'line','label':'bin '+str(j),'x':'time','y':key,'support':mask})
        elif variant=='action_diffusion' and 'paired_action_mean_square_kpc2_kms2' in arrays:
            for j in range(3):
                key='action_'+str(j);fields[key]=arrays['paired_action_mean_square_kpc2_kms2'][:,j]
                specs[key]={'units':'proper kpc^2 (km/s)^2','quantity':'mean squared paired action change from first saved epoch'}
                series.append({'source':variant,'kind':'line','label':'action '+str(j),'x':'time','y':key,'support':'supported'})
            fields['supported']=arrays['action_epoch_supported'];specs['supported']={'units':'dimensionless','quantity':'common native host action support'}
        if not series:
            fields.update(unavailable=np.full(len(fields['time']),np.nan),supported=np.zeros(len(fields['time']),bool))
            specs.update(unavailable={'units':'dimensionless','quantity':'unavailable measurement'},supported={'units':'dimensionless','quantity':'no measurement support'})
            series=[{'source':variant,'kind':'line','label':'unresolved','x':'time','y':'unavailable','support':'supported'}]
        path=out/(variant+'.json');sources[variant]=str(path)
        export_numeric_audit(path,{'schema':'native-response-figure-v1','variant':variant,'definition':report['variants'][variant],
            'science_acceptance':'NOT_ACCEPTED'},fields,specs,algorithm=__file__,source_hashes=pins,context={'epoch':{'actual_saved_epochs':report['epochs']}})
        unit=specs[series[0]['y']]['units']
        figures.append({'figure_id':'native-'+variant,'family':'encounters','variant':variant,
            'title':variant.replace('_',' ').capitalize()+' — '+report['variants'][variant]['status'],
            'science_status':'SYNTHETIC_CONTROL_NOT_ACCEPTED' if report['source_origin']=='synthetic_control' else 'NOT_ACCEPTED',
            'panels':panels or [{'panel_id':'saved','title':'Fixed reference chart; original paired support',
                'quality_status':'PENDING_CONVERGENCE','selection':report['population_scope'],
                'xlabel':'Saved time [Myr]','ylabel':variant+' ['+unit+']','x_units':'Myr','y_units':unit,'series':series}]})
    return export_scientific_suite({'sources':sources,'figures':figures,'dpi':dpi,'maximum_panels_per_figure':3,'organized_outputs':True},
        out/'gallery',base=out,memory_gib=memory_gib)
