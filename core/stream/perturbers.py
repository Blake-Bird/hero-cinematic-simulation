"""Exact native ancestry and bound memberships → encounter worldlines.

This adapter never joins equal catalogue rows across epochs or unwraps a
trajectory by nearest images. The caller supplies recorded images and interval
support; native descendant edges establish continuity. Sparse observations
remain sparse, and unverified interpolation still requests refinement.
"""
from pathlib import Path
from collections import Counter
import json
import math
import zipfile
import numpy as np
from ..foundation import file_sha256
from .native_encounters import _merge_source_bindings


def validate_perturber_history(parameters):
    for name in ('maximum_particles','maximum_nodes','maximum_tracks'):
        if type(parameters[name]) is not int or parameters[name]<1:
            raise ValueError('Exact positive perturber '+name+' required')
    value=parameters['memory_gib']
    if type(value) not in (int,float) or not math.isfinite(value) or value<.1:
        raise ValueError('Bounded positive perturber history memory required')


def _key(value):
    if (not isinstance(value,(list,tuple)) or len(value)!=3 or not isinstance(value[0],str) or not value[0]
            or type(value[1]) is not int or not 0<=value[1]<2**64 or type(value[2]) is not int or value[2]<0):
        raise ValueError('Exact namespace/object-ID/saved-epoch key required')
    return tuple(value)


def perturber_history_input_bindings(parameters):
    """Bind external graph, frame payload, controls and certificates on resume."""
    from ..halos import verify_identity_graph
    from core.stream_common import read_product
    from ..data import numeric_audit_source_bindings
    validate_perturber_history(parameters);memory=parameters['memory_gib'];result={}
    graph_path=Path(parameters['identity_graph']).resolve()
    template_path=Path(parameters['template_path']).resolve();frame_path=Path(parameters['stream_frame_history']).resolve()
    for path in (graph_path/'report.json',frame_path,template_path):
        if path.stat().st_size*16>memory*1024**3/8:raise MemoryError('Perturber metadata exceeds bounded intake')
    with zipfile.ZipFile(graph_path/'nodes.npz') as archive:
        if sum(x.file_size for x in archive.infolist())*4>memory*1024**3/8:raise MemoryError('Identity graph arrays exceed bounded intake')
    graph=verify_identity_graph(graph_path)
    if len(graph['nodes'])>parameters['maximum_nodes']:raise MemoryError('Native identity graph exceeds selected intake budget')
    if file_sha256(template_path)!=parameters['template_sha256']:raise ValueError('Frozen perturber template changed')
    template=json.loads(template_path.read_text())
    _merge_source_bindings(result,graph['sources'])
    _merge_source_bindings(result,{str(graph_path/name):file_sha256(graph_path/name) for name in ('report.json','index.sqlite','nodes.npz')})
    record,_,sources=read_product(frame_path,memory_gib=memory/8)
    _merge_source_bindings(result,sources)
    _merge_source_bindings(result,numeric_audit_source_bindings(record))
    _merge_source_bindings(result,{str(template_path):parameters['template_sha256']})
    for track in template['tracks']:
        certificate=track.get('interpolation_certificate')
        if isinstance(certificate,dict):_merge_source_bindings(result,certificate.get('source_hashes',{}))
    for path,digest in result.items():
        if file_sha256(path)!=digest:raise ValueError('Native perturber source/certificate changed')
    return result


def native_perturber_history_runner(parameters,context):
    from ..halos import verify_identity_graph,IdentityQueries,_snapshot_files
    from core.stream_common import read_product, publish_stream_product
    from ..observation_geometry import read_native_gravitating_particles
    from ..data import numeric_audit_source_bindings,proper_position_derivative_kms,apply_recorded_periodic_images
    from core.stream.trajectories import Worldline
    validate_perturber_history(parameters);memory=parameters['memory_gib']
    if memory>context.resources['memory_gib']:raise MemoryError('Perturber history exceeds stage memory')
    graph_path=Path(parameters['identity_graph']).resolve();frame_path=Path(parameters['stream_frame_history']).resolve()
    template_path=Path(parameters['template_path']).resolve();pins={}
    for path in (graph_path/'report.json',frame_path,template_path):
        if path.stat().st_size*16>memory*1024**3/8:raise MemoryError('Perturber metadata exceeds bounded intake')
    with zipfile.ZipFile(graph_path/'nodes.npz') as archive:
        if sum(x.file_size for x in archive.infolist())*4>memory*1024**3/8:
            raise MemoryError('Identity graph arrays exceed bounded intake')
    if file_sha256(template_path)!=parameters['template_sha256']:raise ValueError('Frozen perturber template changed')
    template=json.loads(template_path.read_text())
    if set(template)!={'schema','tracks','maximum_interval_myr','radial_edges_kpc','species','particle_namespace'} or template['schema']!='native-perturber-history-template-v1':
        raise ValueError('Complete frozen native perturber history template required')
    tracks=template['tracks'];edges=np.asarray(template['radial_edges_kpc'],float)
    species=template['species']
    if (not isinstance(species,list) or not species or any(type(k) is not int or not 0<=k<6 for k in species)
            or len(set(species))!=len(species) or not isinstance(template['particle_namespace'],str) or not template['particle_namespace']):
        raise ValueError('Explicit native species union and exact member namespace required')
    if (edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all() or edges[0]!=0 or np.any(np.diff(edges)<=0)):
        raise ValueError('Increasing proper radial edges beginning at zero required')
    if not isinstance(tracks,list) or not tracks or len(tracks)>parameters['maximum_tracks']:
        raise ValueError('Explicit bounded selected perturber tracks required')
    limit=template['maximum_interval_myr']
    if type(limit) not in (int,float) or not math.isfinite(limit) or limit<=0:raise ValueError('Positive physical cadence limit required')
    graph=verify_identity_graph(graph_path)
    if len(graph['nodes'])>parameters['maximum_nodes']:raise MemoryError('Native identity graph exceeds selected intake budget')
    _merge_source_bindings(pins,graph['sources']);_merge_source_bindings(pins,{str(graph_path/name):file_sha256(graph_path/name) for name in ('report.json','index.sqlite','nodes.npz')})
    _merge_source_bindings(pins,{str(template_path):parameters['template_sha256']})
    frame,values,sources=read_product(frame_path,memory_gib=memory/8)
    _merge_source_bindings(pins,sources);_merge_source_bindings(pins,numeric_audit_source_bindings(frame))
    if frame.get('schema')!='native-encounter-frame-history-v1':raise ValueError('Converted native stream clock/origin frame required')
    periodic=frame.get('periodic_images_applied')
    if type(periodic) is not bool:raise ValueError('Explicit native periodic/isolated frame required')
    n=len(frame['epochs']);times=np.asarray(values['times_myr'])
    if n<2 or times.shape!=(n,) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Increasing actual saved physical epochs required')
    units=frame['numeric_data']['array_contract']['datasets']
    for name,unit in (('times_myr','Myr'),('origin_proper_kpc','proper kpc'),('origin_velocity_peculiar_kms','km / s'),('hubble_kms_mpc','km / s / Mpc')):
        if units[name]['units']!=unit:raise ValueError('Converted stream frame units differ: '+name)
    labels=[epoch['snapshot_storage_index'] for epoch in frame['epochs']]
    if any(type(x) is not int or x<0 for x in labels) or len(set(labels))!=n:raise ValueError('Distinct actual native saved labels required')
    nodes={_key(row['key']):row for row in graph['nodes']}
    descendants={(_key(row['source']),_key(row['target'])) for row in graph['edges']
        if row['kind']=='descendant' and row['evidence'].get('origin')=='native_ancestry'}
    outgoing=Counter(left for left,right in descendants)
    queries=IdentityQueries(graph_path);worldlines=[];unresolved=[];identities=set()
    for track in tracks:
        required={'observations','image_offsets','image_supported','valid_intervals','position_covariance_kpc2','interpolation_error_kpc','interpolation_certificate','physics'}
        if not isinstance(track,dict) or set(track)!=required:raise ValueError('Complete explicit perturber observations/images/support required')
        observations=track['observations']
        if not isinstance(observations,list) or len(observations)!=n:raise ValueError('One exact observation or unavailable null per saved epoch required')
        keys=[None if key is None else _key(key) for key in observations]
        for label,key in zip(labels,keys):
            if key is not None and (key[2]!=label or key not in nodes):raise ValueError('Perturber observation differs from its actual native saved key')
        for ordinal,key in enumerate(keys):
            if key is not None and graph['epochs'][str(key[2])]!=frame['epochs'][ordinal]:
                raise ValueError('Native identity graph and stream frame saved clocks differ')
        identity=next((key for key in keys if key is not None),None)
        if identity is None:raise ValueError('Selected perturber has no recorded native observation')
        if identity in identities:raise ValueError('Duplicate selected perturber track')
        identities.add(identity)
        flags=np.asarray(track['valid_intervals']);images=np.asarray(track['image_offsets']);image_support=np.asarray(track['image_supported'])
        if flags.shape!=(n-1,) or flags.dtype.kind!='b' or images.shape!=(n,3) or images.dtype.kind not in 'iu' or image_support.shape!=(n,) or image_support.dtype.kind!='b':
            raise ValueError('Exact recorded periodic images and boolean saved interval support required')
        if not periodic and np.any(images):raise ValueError('Isolated native positions cannot carry periodic image shifts')
        positions=[];velocities=[];masses=[];profiles=[];available=np.zeros(n,bool);reasons=[]
        # Missing endpoints use a finite placeholder only behind false support.
        # They are never used for interpolation or plotted as observations.
        for ordinal,key in enumerate(keys):
            row=None if key is None else nodes[key];attrs={} if row is None else row['attributes']
            membership_key=key
            finder_available=True
            if row is not None and row['kind']=='tree' and attrs.get('original_finder_key') is not None:
                membership_key=_key(attrs['original_finder_key'])
                if membership_key[2]!=key[2]:raise ValueError('Native tree/finder reference crosses saved epochs')
                finder=nodes.get(membership_key)
                finder_available=finder is not None and finder['status']=='DETECTED'
                # Tree IDs establish ancestry. The actual native finder
                # observation supplies its measured phase space/membership.
                # Predicted tree rows never regain detection through a join.
                if finder is not None and finder['status']=='DETECTED':attrs=finder['attributes']
            reason=None
            if row is None or row['status']!='DETECTED':reason='MISSING_OR_AMBIGUOUS_NATIVE_OBSERVATION'
            elif not finder_available:reason='NATIVE_ORIGINAL_FINDER_OBSERVATION_UNAVAILABLE'
            elif row['kind'] not in {'subhalo','tree','bh'}:reason='NOT_A_BOUND_PERTURBER_KIND'
            elif any(name not in attrs for name in ('position_proper_kpc','velocity_peculiar_kms')):reason='NATIVE_PHASE_COORDINATES_UNAVAILABLE'
            elif not image_support[ordinal]:reason='RECORDED_IMAGE_UNAVAILABLE'
            members=np.array([],dtype=np.uint64) if reason else queries.members_at_epoch(membership_key,particle_namespace=template['particle_namespace'],bound_only=True)
            if reason is None and not len(members):reason='NATIVE_BOUND_MEMBERSHIP_UNAVAILABLE'
            if reason:
                positions.append([0.,0.,0.]);velocities.append([0.,0.,0.]);masses.append(0.);profiles.append(None);reasons.append({'snapshot':labels[ordinal],'reason':reason});continue
            native,census=read_native_gravitating_particles(parameters['run_dir'],labels[ordinal],species=template['species'],
                maximum_particles=parameters['maximum_particles'],memory_gib=memory/4,namespace=template['particle_namespace'])
            _merge_source_bindings(pins,native['source_hashes'])
            if native['epoch']!=frame['epochs'][ordinal]:raise ValueError('Native perturber particle clock differs from stream frame')
            # Every native snapshot read must be part of the graph's recorded
            # inputs, so another run cannot be attached by changing its label.
            snapshots=_snapshot_files(parameters['run_dir'],labels[ordinal])
            if not snapshots or any(graph['sources'].get(str(Path(path).resolve()))!=file_sha256(path) for path in snapshots):
                raise ValueError('Perturber graph does not bind the actual native snapshot generation')
            ids=census['particle_ids'];order=np.argsort(ids);at=np.searchsorted(ids[order],members)
            if np.any(at>=len(order)) or not np.array_equal(ids[order[at]],members):raise ValueError('Native bound perturber loses an exact member identity')
            chosen=order[at];center=np.asarray(attrs['position_proper_kpc'],float);velocity=np.asarray(attrs['velocity_peculiar_kms'],float)
            if center.shape!=(3,) or velocity.shape!=(3,) or not np.isfinite([center,velocity]).all():raise ValueError('Finite native perturber phase coordinates required')
            box=float(values['periodic_box_proper_kpc'][ordinal]);delta=census['position_proper_kpc'][chosen]-center
            if periodic:
                if box<=0 or edges[-1]>=box/2:raise ValueError('Unique local proper profile domain required')
                delta-=box*np.rint(delta/box)
            radius=np.linalg.norm(delta,axis=1);mass=census['mass_msun'][chosen]
            # Direct sums over native bound membership; no analytic fit or
            # renormalization to a catalogue mass with a different definition.
            # M(<=R) = sum(m_i for r_i<=R), with R in proper kpc and m_i
            # in Msun. Minimum images above affect only this local spatial
            # profile; recorded images alone determine temporal worldlines.
            sort=np.argsort(radius);cumulative=np.cumsum(mass[sort]);where=np.searchsorted(radius[sort],edges[1:],side='right')
            enclosed=np.r_[0.,cumulative][where];total=float(mass.sum())
            profiles.append({'kind':'enclosed_mass','radius_kpc':edges[1:].tolist(),'enclosed_mass_msun':enclosed.tolist(),
                'mass_msun':total,'mass_definition':'sum of actual selected native bound gravitating member masses',
                'member_count':len(members),'profile_support':'MEASURED_WITHIN_DECLARED_RADII',
                'object_key':[membership_key[0],str(membership_key[1]),membership_key[2]],
                'persistent_track_observation_key':[key[0],str(key[1]),key[2]]})
            absolute=apply_recorded_periodic_images(center[None],images[ordinal:ordinal+1],box_proper_kpc_by_epoch=[box])[0] if periodic else center
            positions.append((absolute-values['origin_proper_kpc'][ordinal]).tolist())
            velocities.append(proper_position_derivative_kms(absolute[None],velocity[None],hubble_kms_mpc=values['hubble_kms_mpc'][ordinal],
                origin_proper_kpc=values['origin_proper_kpc'][ordinal],origin_velocity_peculiar_kms=values['origin_velocity_peculiar_kms'][ordinal])[0].tolist())
            masses.append(total);available[ordinal]=True
        continuous=np.array([(left,right) in descendants and outgoing.get(left)==1 for left,right in zip(keys,keys[1:])])
        valid=flags&available[:-1]&available[1:]&continuous&(np.diff(times)<=limit)
        record={'identity':identity[0]+':'+str(identity[1])+'@'+str(identity[2]),'namespace':'native-perturber-track',
            'frame':frame['frame'],'units':'proper_kpc_kms_myr','times_myr':times.tolist(),'position_kpc':positions,
            'velocity_kms':velocities,'mass_msun':masses,'profile_history':profiles,'valid_intervals':valid.tolist(),
            'observation_supported':available.tolist(),'native_observation_keys':[[k[0],str(k[1]),k[2]] if k else None for k in keys],
            'maximum_interval_myr':limit,'position_covariance_kpc2':track['position_covariance_kpc2'],
            'interpolation_error_kpc':track['interpolation_error_kpc'],'interpolation_certificate':track['interpolation_certificate'],
            'physics':track['physics'],'profile_support':'NATIVE_BOUND_MEMBERS_ONLY','source_history':str(graph_path/'report.json')}
        Worldline(record);worldlines.append(record)
        if isinstance(track['interpolation_certificate'],dict):_merge_source_bindings(pins,track['interpolation_certificate'].get('source_hashes',{}))
        unresolved.append({'identity':record['identity'],'observations':reasons,'unsupported_interval_indices':np.flatnonzero(~valid).tolist(),
            'missing_native_descendant_interval_indices':np.flatnonzero(~continuous).tolist()})
    _merge_source_bindings(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Native perturber history source changed')
    result={'schema':'native-encounter-frame-history-v1','namespace':'native-perturber-track','frame':frame['frame'],
        'frame_definition_sha256':frame['frame_definition_sha256'],'epochs':frame['epochs'],'worldlines':worldlines,
        'periodic_images_applied':periodic,
        'unresolved':unresolved,'science_acceptance':'NOT_ACCEPTED','source_graph':str(graph_path),'source_stream_frame':str(frame_path),
        'source_origin':'synthetic_control' if frame.get('fixture') or frame.get('source_origin')=='synthetic_control' else 'native_snapshot_and_identity_graph',
        'velocity_convention':'proper position derivative km/s; same saved clock/origin/H as converted stream',
        'periodic_wrapping':'explicit recorded images; local profile minimum images only, no inferred temporal winding' if periodic else 'isolated native positions; no periodic image shifts'}
    return publish_stream_product(context.generation/'perturber-history.json',result,{'times_myr':times},{'times_myr':'Myr'},
        sources=pins,namespace=result['namespace'],epoch={'actual_saved_epochs':frame['epochs']})
