"""Original encounter histories to global saved overlays and local event views."""
from pathlib import Path
import json
import numpy as np
from .foundation import atomic_json,file_sha256,canonical_digest


def prepare_native_scene_overlay(request,combined,root,*,memory_gib):
    from .stream.native_encounters import encounter_display_inputs
    from core.research.display import _reference
    from core.stream_common import read_product
    from .data import coordinate_contract,export_numeric_audit,numeric_audit_source_bindings
    from core.stream.trajectories import Worldline
    from core.research.display import assemble_measured_display
    from .foundation import merge_source_bindings as _merge
    from .blender import verify_timeline_assets,saved_stream_frame_schedule
    required={'stream_history','perturber_history','encounters','maximum_objects','maximum_vertices','marker_radius_mpc_h','line_width_mpc_h','event_marker_radius_kpc'}
    if set(request)!=required:raise ValueError('Complete source-bound native scene overlay request required')
    for name in ['marker_radius_mpc_h','line_width_mpc_h','event_marker_radius_kpc']:
        if type(request[name]) not in (int,float) or not np.isfinite(request[name]) or request[name]<=0:raise ValueError('Positive explicit symbolic overlay size required')
    paths={key:str(_reference(request[key])) for key in ['stream_history','perturber_history','encounters']}
    params={**paths,'memory_gib':memory_gib,'maximum_objects':request['maximum_objects'],'maximum_vertices':request['maximum_vertices'],'frame_stride':1}
    histories,pins=encounter_display_inputs(params);(stream,values),(perturbers,_),(events,event_arrays)=histories
    original=Path(stream['source_history']).resolve()
    if file_sha256(original)!=stream['source_history_sha256']:raise ValueError('Encounter chart original native history changed')
    native,_,native_pins=read_product(original,memory_gib=memory_gib/8)
    _merge(pins,native_pins);_merge(pins,numeric_audit_source_bindings(native))
    if native.get('schema')!='measured-native-stream-worldlines-v1' or Path(native['native_run_directory']).resolve()!=Path(combined['run_dir']).resolve() or native['epochs']!=stream['epochs']:raise ValueError('Encounter overlay is not from this original native scene run/clock')
    for name,unit in [('origin_proper_kpc','proper kpc'),('times_myr','Myr')]:
        if stream['numeric_data']['array_contract']['datasets'][name]['units']!=unit:raise ValueError('Original native chart units differ')
    epochs=stream['epochs'];times=values['times_myr'];origin=values['origin_proper_kpc'];n=len(epochs)
    if origin.shape!=(n,3) or times.shape!=(n,) or not np.isfinite(origin).all():raise ValueError('Finite original chart origin at each saved epoch required')
    timeline=verify_timeline_assets(Path(root)/combined['species_manifests']['dm']);saved=timeline['snapshots'];stride=combined['frames_per_snapshot']
    lookup={row['snapshot']:(ordinal,row) for ordinal,row in enumerate(saved)};frames=[];scales=[]
    for epoch,time in zip(epochs,times,strict=True):
        if epoch.get('canonical_time')!='scale_factor':raise ValueError('Cosmological native overlay requires cosmological saved clock')
        ordinal,row=lookup[epoch['snapshot_storage_index']]
        if epoch['scale_factor']!=row['scale_factor'] or time!=1000*row['cosmic_time_gyr']:raise ValueError('Encounter overlay changes actual saved scene clock')
        frames.append(1 if ordinal==0 else ordinal*stride);scales.append(combined['hubble_parameter']/(1000*row['scale_factor']))
    saved_stream_frame_schedule(times.tolist(),scene_frames=frames,scene_frame_end=(len(saved)-1)*stride)
    scale=np.asarray(scales);lines=[('selected_stellar_worldline',line) for line in stream['worldlines']]+[('native_perturber_worldline',line) for line in perturbers['worldlines']]
    objects=len(lines)*n+2*len(events['events'])
    if objects>request['maximum_objects'] or objects*6>request['maximum_vertices']:raise MemoryError('Native overlay worldline/impact object and geometry census exceeds aggregate allocation')
    folder=Path(root)/'native_encounter_overlay';folder.mkdir(exist_ok=False);display_timeline=[];identity_map=[]
    frame=coordinate_contract(position_units='comoving Mpc/h',velocity_convention='not a velocity product; original proper derivatives retained in source',
        origin='global simulation Cartesian box origin',periodic_wrapping='Original recorded periodic images preserved; no additional wrapping or winding inference')
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    for ordinal,epoch in enumerate(epochs):
        products=[]
        for index,(role,line) in enumerate(lines):
            support=np.asarray(line.get('observation_supported',[True]*n))
            if support.dtype.kind!='b' or support.shape!=(n,):raise ValueError('Original exact observation support required')
            if not support[ordinal]:continue
            curve=Worldline(line);x=np.asarray(line['position_kpc'],float)
            if x.shape!=(n,3) or not np.isfinite(x).all():raise ValueError('Original finite native worldline positions required')
            # Each endpoint uses its own saved origin AND scale factor. Using
            # the current epoch's conversion for both would distort the path.
            global_x=(x+origin)*scale[:,None]
            keys=line.get('native_observation_keys')
            if keys:
                anchor=next(key for key in keys if key is not None);identifier=int(anchor[1]);namespace=anchor[0]+':track-anchor:'+str(anchor[2])
            else:
                if not isinstance(line['identity'],str) or not line['identity'].isdecimal():raise ValueError('Exact original native stellar identity required')
                identifier=int(line['identity']);namespace=line['namespace']
            if not 0<=identifier<2**64:raise ValueError('Native overlay identity exceeds uint64')
            previous=max(0,ordinal-1);valid=ordinal>0 and bool(curve.valid[previous]) and bool(support[previous])
            path=folder/('epoch-%04d-line-%04d.json'%(ordinal,index))
            arrays={'object_ids':np.array([identifier],np.uint64),'position':global_x[ordinal:ordinal+1],
                'points':global_x[[previous,ordinal]][None,:,:],'supported':np.array([valid],bool)}
            specs={'object_ids':{'units':'identity','quantity':'original exact particle or native track anchor identity','identity_namespace':namespace},
                'position':{'units':'comoving Mpc/h','quantity':'actual saved native center transformed once to global scene coordinates','coordinates':frame},
                'points':{'units':'comoving Mpc/h','quantity':'two actual saved native endpoints, each transformed at its own epoch','coordinates':frame},
                'supported':{'units':'dimensionless','quantity':'original interval and observation support'}}
            export_numeric_audit(path,{'schema':'native-scene-worldline-overlay-v1','native_identity':line['identity'],'native_namespace':line['namespace'],
                'native_observation_keys':keys,'role':role,'snapshot':epoch['snapshot_storage_index'],'science_acceptance':'NOT_ACCEPTED',
                'scope':'Actual saved worldline endpoints; symbolic display width; hold saved state without new dynamics'},arrays,specs,
                algorithm=__file__,source_hashes=pins,context={'epoch':epoch,'coordinates':frame})
            products.append({'path':str(path),'sha256':file_sha256(path),'kind':'stream','fields':{'object_id':'object_ids','position':'position','points':'points','geometry_supported':'supported'},
                'display_radius':request['marker_radius_mpc_h'],'display_line_width':request['line_width_mpc_h']})
            identity_map.append({'epoch':epoch,'namespace':namespace,'object_id_decimal':str(identifier),'role':role,'original_native_identity':line['identity'],'native_observation_keys':keys})
        display_timeline.append({'frame':frames[ordinal],'snapshot':epoch['snapshot_storage_index'],'epoch_contract':epoch,'products':products})
    build={'schema':'measured-display-build-v1','memory_gib':memory_gib,'maximum_objects':request['maximum_objects'],
        'maximum_geometry_vertices':request['maximum_vertices'],'maximum_geometry_objects':request['maximum_objects'],'frame_contract':frame,'timeline':display_timeline}
    atomic_json(folder/'display-request.json',build);assemble_measured_display(build,Path(root)/'measured_display',request_base=folder)
    combined['measured_display']={'path':'measured_display/display.json','sha256':file_sha256(Path(root)/'measured_display/display.json')}
    atomic_json(folder/'native-identities.json',{'schema':'native-scene-overlay-identities-v1','records':identity_map})
    # A source chart origin between snapshots is not a measured global origin.
    # Keep reconstructed impacts in clearly labelled LOCAL proper-kpc views.
    event_views=[]
    support=event_arrays['geometry_supported']
    if support.dtype.kind!='b' or support.shape!=(len(events['events']),):raise ValueError('Original exact per-event geometry support required')
    for index,event in enumerate(events['events']):
        points=np.asarray([event['stream_position_kpc'],event['perturber_position_kpc']],float)
        if points.shape!=(2,3) or not np.isfinite(points).all():raise ValueError('Finite original reconstructed impact positions required')
        if type(event['time_myr']) not in (float,int) or not np.isfinite(event['time_myr']) or not times[0]<=event['time_myr']<=times[-1]:raise ValueError('Original reconstructed impact must lie within recorded source clock coverage')
        event_views.append({'event_id':event['event_id'],'time_myr':float(event['time_myr']),'positions_proper_kpc':points.tolist(),
            'stream_identity':event['stream_identity'],'perturber_identity':event['perturber_identity'],'geometry_supported':bool(event_arrays['geometry_supported'][index]),
            'marker_radius_kpc':request['event_marker_radius_kpc'],'frame':stream['frame'],'position_meaning':'Reconstructed event positions in original local chart; not actual saved snapshot centers; no inferred global origin at event time'})
    if len(event_views)*2>request['maximum_objects']:raise MemoryError('Reconstructed impact marker census exceeds scene allocation')
    event_path=folder/'local-impacts.json';atomic_json(event_path,{'schema':'native-scene-local-impacts-v1','views':event_views,'source_hashes':pins,'science_acceptance':'NOT_ACCEPTED'})
    combined['local_impact_views']={'path':str(event_path.relative_to(root)),'sha256':file_sha256(event_path)}
    if any(file_sha256(path)!=sha for path,sha in pins.items()):raise ValueError('Original encounter scene sources changed during preparation')
    return {'worldline_records':len(identity_map),'local_event_views':len(event_views),'source_hashes':pins}
