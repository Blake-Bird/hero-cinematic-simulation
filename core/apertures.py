"""Measured native selection apertures and their contamination histories.

Catalogue cohorts define mass-centered bounding spheres. Encounter spheres use
actual saved stellar positions and the declared search radius, preserving the
separate reconstructed event time. Centers are original-box comoving kpc/h;
radii are proper kpc, with x_comoving = x_proper*h/a before periodic wrapping.
Each aperture is measured separately: overlaps are not added into a union.
Missing ancestry, uncertain classification and unresolved events stay explicit.
"""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .foundation import atomic_json, file_sha256, product_provenance, canonical_digest, merge_source_bindings as _merge


def validate_aperture_contamination(parameters):
    if type(parameters['memory_gib']) not in (int,float) or not math.isfinite(parameters['memory_gib']) or parameters['memory_gib']<96/1024:
        raise ValueError('At least96MiB native aperture contamination workspace required')
    if type(parameters['maximum_apertures']) is not int or parameters['maximum_apertures']<1:
        raise ValueError('Exact positive aperture intake bound required')
    value=parameters['maximum_coarse_mass_fraction']
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('Explicit native coarse mass fraction criterion required')



def aperture_contamination_runner(parameters,context):
    from .zoom import _run_lineage,_snapshot_files,contamination_epoch
    validate_aperture_contamination(parameters)
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Aperture contamination exceeds stage allocation')
    path=Path(parameters['aperture_product']).resolve()
    if path.stat().st_size>min(16*1024**2,parameters['memory_gib']*1024**3/16):raise MemoryError('Aperture product metadata exceeds bounded intake')
    record=json.loads(path.read_text())
    if record.get('schema')!='native-science-apertures-v1':raise ValueError('Measured native science aperture product required')
    snapshot=record['snapshot']
    if type(snapshot) is not int or snapshot<0:raise ValueError('Exact actual aperture snapshot required')
    apertures=record['apertures']
    if not isinstance(apertures,list) or len(apertures)>parameters['maximum_apertures']:raise ValueError('Aperture count exceeds explicit intake bound')
    lineage=_run_lineage(parameters['run_dir']);files=_snapshot_files(parameters['run_dir'],snapshot)
    sources={str(Path(p).resolve()):file_sha256(p) for p in files}
    if record.get('native_snapshot_sources')!=sources:raise ValueError('Apertures refer to different actual native snapshot sources')
    if record.get('coordinate_frame')!='original_box_comoving_kpc_h':raise ValueError('Explicit original periodic comoving kpc/h aperture centers required')
    pins={str(path):file_sha256(path),**sources,lineage['path']:lineage['sha256']}
    pins.update({row['path']:row['sha256'] for row in lineage['initial_conditions']})
    provenance=record.get('source_hashes')
    if not isinstance(provenance,dict) or not provenance:raise ValueError('Measured aperture producer provenance required')
    for source,digest in provenance.items():
        if source in pins and pins[source]!=digest:raise ValueError('Conflicting aperture producer source bindings')
        pins[source]=digest
    for source,digest in pins.items():
        if file_sha256(source)!=digest:raise ValueError('Aperture producer/native input changed')
    # Read only the first verified snapshot header for the observed epoch;
    # the existing estimator checks every shard and its unit/frame contract.
    import h5py
    with h5py.File(files[0]) as f:
        actual_scale=float(f['Header'].attrs['Time']);actual_h=float(f['Parameters'].attrs['HubbleParam'])
    if not math.isclose(actual_h,lineage['h'],rel_tol=1e-12):raise ValueError('Aperture native frame differs from initial mass inventory')
    if type(record['scale_factor']) not in (int,float) or not math.isfinite(record['scale_factor']) or not math.isclose(actual_scale,record['scale_factor'],rel_tol=0,abs_tol=1e-12):
        raise ValueError('Measured aperture epoch differs from actual saved native time')
    results=[];seen=set()
    for row in apertures:
        identifier=row['id']
        if not isinstance(identifier,str) or not identifier or identifier in seen:raise ValueError('Distinct persistent science aperture identifiers required')
        seen.add(identifier)
        if row['family'] not in ('halo','progenitor','stream','encounter'):raise ValueError('Supported native aperture family required')
        if row['status'] not in ('MEASURED','UNRESOLVED','UNSUPPORTED'):raise ValueError('Explicit measured or unresolved aperture support required')
        support={key:row[key] for key in ('aperture_definition','stellar_member_count','particle_ids_sha256',
            'event_time_myr','saved_time_myr','stream_identity','perturber_identity') if key in row}
        if row['status']!='MEASURED':
            if not isinstance(row.get('reason'),str) or not row['reason']:raise ValueError('Unresolved aperture needs its actual support reason')
            results.append({'id':identifier,'family':row['family'],**support,'status':'UNRESOLVED_TARGET','producer_status':row['status'],'reason':row['reason'],'native_contamination':None})
            continue
        if not isinstance(row.get('selection_identity'),str) or not row['selection_identity']:raise ValueError('Measured selection identity required for contamination support')
        radius=row['radius_proper_kpc'];center=row['center_comoving_kpc_h']
        if (type(radius) not in (int,float) or not math.isfinite(radius) or radius<=0
                or not isinstance(center,list) or len(center)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in center)):
            raise ValueError('Finite measured center and positive proper aperture radius required')
        measured=contamination_epoch(files,snapshot,center_comoving_kpc_h=row['center_comoving_kpc_h'],
            radius_proper_kpc=row['radius_proper_kpc'],fine_dm_mass_msun=lineage['fine_dm_mass_msun'],
            fine_dm_mass_range_msun=lineage['fine_dm_mass_range_msun'],maximum_coarse_mass_fraction=parameters['maximum_coarse_mass_fraction'])
        results.append({'id':identifier,'family':row['family'],**support,'selection_identity':row['selection_identity'],'status':measured['status'],
            'center_comoving_kpc_h':row['center_comoving_kpc_h'],'radius_proper_kpc':row['radius_proper_kpc'],
            'native_contamination':measured})
    for source,digest in pins.items():
        if file_sha256(source)!=digest:raise ValueError('Aperture contamination input changed during measurement')
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    pins[str(Path(__file__).with_name('zoom.py').resolve())]=file_sha256(Path(__file__).with_name('zoom.py'))
    result={'schema':'native-science-aperture-contamination-v1','status':'FAIL' if any(r['status']=='FAIL' for r in results) else 'PASS' if results and all(r['status']=='PASS' for r in results) else 'UNRESOLVED',
        'snapshot':snapshot,'scale_factor':actual_scale,'records':results,'source_hashes':pins,
        'maximum_coarse_mass_fraction':parameters['maximum_coarse_mass_fraction'],
        'scope':'Actual masses in individually measured spherical support apertures; overlapping spheres are separate, not a summed stream tube or encounter world volume',
        'science_acceptance':'UNRESOLVED','execution_status':'COMPLETE'}
    atomic_json(context.generation/'contamination.json',result)
    if result['status']=='FAIL':
        raise ValueError('Measured science aperture exceeds the frozen coarse-mass criterion; contamination.json retained')
    return result



def catalogue_apertures(report, arrays, *, snapshot_sources, maximum_apertures):
    """Bound measured stellar selections by spheres in their saved spatial chart.

    Center = sum(m_i*x_i)/sum(m_i); radius = max(|x_i-center|), in proper
    kpc. Progenitor cores require known bound membership, while thin-stream
    selections require resolved, calibrated classification. Unsupported
    selections retain a reason rather than acquiring invented geometry.
    """
    if report.get('schema') != 'tidal-stream-catalogue-v1':
        raise ValueError('Native tidal stellar catalogue required for selection apertures')
    context = report.get('source_context', {})
    if type(context.get('snapshot')) is not int or context['snapshot'] < 0:
        raise ValueError('Actual native catalogue snapshot identity required')
    supplied = {row['path']: row['sha256'] for row in context['source_sha256']}
    if any(supplied.get(path) != digest for path, digest in snapshot_sources.items()):
        raise ValueError('Stellar catalogue refers to another native snapshot')
    a, h, box = (context[name] for name in ('scale_factor', 'h', 'box_comoving_kpc_h'))
    if any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0 for value in (a, h, box)):
        raise ValueError('Finite positive native catalogue frame required')
    origin = np.asarray(context['host_center_comoving_kpc_h'], dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all(): raise ValueError('Measured native host origin required')
    namespace = report.get('namespace')
    if not isinstance(namespace, str) or not namespace: raise ValueError('Exact native particle identity namespace required')
    ids = np.asarray(arrays['particle_ids']); n = len(ids)
    if ids.ndim != 1 or ids.dtype.kind != 'u' or ids.dtype.itemsize != 8 or len(np.unique(ids)) != n:
        raise ValueError('Distinct exact native uint64 stellar identities required')
    x = np.asarray(arrays['position_kpc']); m = np.asarray(arrays['mass_msun'])
    if x.shape != (n, 3) or m.shape != (n,) or not np.isfinite(x).all() or not np.isfinite(m).all() or np.any(m <= 0):
        raise ValueError('Finite measured host-relative stellar positions and positive masses required')
    labels = {}
    for name in ('progenitor_label', 'candidate_id'):
        labels[name] = np.asarray(arrays[name])
        if labels[name].shape != (n,) or labels[name].dtype.kind not in 'iu': raise ValueError('Exact stellar selection labels required')
    bound = np.asarray(arrays['bound']); known = np.asarray(arrays['binding_known'])
    if any(value.shape != (n,) or value.dtype.kind != 'b' for value in (bound, known)):
        raise ValueError('Explicit native bound-membership support masks required')
    rows = []
    def unresolved(identifier, family, reason):
        rows.append({'id': identifier, 'family': family, 'status': 'UNRESOLVED', 'reason': reason})
    def sphere(identifier, family, mask, selection):
        count = int(mask.sum())
        if not count:
            unresolved(identifier, family, 'No current supported stellar members for this measured selection'); return
        positions = x[mask]; mass = m[mask]
        # A common mass scale cancels from the normalized weights. Scaling
        # first avoids overflowing the sum of individually finite masses.
        weights = mass / mass.max()
        weights /= weights.sum()
        center = np.sum(positions * weights[:, None], axis=0)
        radius = float(np.linalg.norm(positions - center, axis=1).max())
        if not math.isfinite(radius) or radius <= 0 or radius >= box * a / h / 2:
            unresolved(identifier, family, 'Selected cohort has no positive unique periodic bounding-sphere support'); return
        cohort = np.sort(ids[mask]).astype('<u8', copy=False)
        rows.append({'id': identifier, 'family': family, 'status': 'MEASURED',
            'selection_identity': json.dumps({'namespace': namespace, 'selection': selection}, sort_keys=True),
            'center_comoving_kpc_h': ((origin + center * h / a) % box).tolist(),
            'radius_proper_kpc': radius, 'stellar_member_count': count,
            'particle_ids_sha256': hashlib.sha256(cohort.tobytes()).hexdigest(),
            'aperture_definition': 'Mass-weighted center and maximum observed member radius; bounding sphere of this selected stellar cohort'})
    keys = report['progenitor_keys']
    for label in sorted(set(labels['progenitor_label'].tolist()) - {-1}):
        if not 0 <= label < len(keys): raise ValueError('Stellar ancestry label has no persistent progenitor identity')
        identity = keys[label]
        if not isinstance(identity, (list, tuple)) or len(identity) != 2 or not isinstance(identity[0], str) or type(identity[1]) is not int or not 0 <= identity[1] < 2**64:
            raise ValueError('Namespace-qualified exact persistent progenitor identity required')
        sphere('progenitor-' + str(label), 'progenitor', (labels['progenitor_label'] == label) & bound & known, identity)
    if not any(row['family'] == 'progenitor' for row in rows):
        unresolved('progenitor-unresolved', 'progenitor', 'No measured persistent progenitor ownership in the native catalogue')
    seen = set()
    for candidate in report['candidates']:
        cid = candidate['candidate_id']
        if type(cid) is not int or cid < 0 or cid in seen: raise ValueError('Distinct exact native candidate identities required')
        seen.add(cid); mask = labels['candidate_id'] == cid
        if type(candidate['particle_count']) is not int or int(mask.sum()) != candidate['particle_count']: raise ValueError('Native candidate membership count differs from its catalogue')
        decision = candidate['classification']; identifier = 'stream-' + str(cid)
        if candidate['resolution_supported'] is not True or decision.get('status') != 'CALIBRATED_DOMAIN' or decision.get('classification') != 'thin_stream':
            unresolved(identifier, 'stream', 'Candidate is not a resolved calibrated thin stream: ' + str(decision.get('status')) + '/' + str(decision.get('classification')))
        else: sphere(identifier, 'stream', mask, {'candidate_id': cid, 'classification': decision})
    if not seen: unresolved('stream-unresolved', 'stream', 'No coherent native stellar candidate selected')
    if set(labels['candidate_id'].tolist()) - {-1} - seen: raise ValueError('Native stellar candidate labels have no catalogue records')
    if len(rows) > maximum_apertures: raise MemoryError('Measured stellar aperture count exceeds frozen allocation')
    return {'schema': 'native-science-apertures-v1', 'snapshot': context['snapshot'], 'scale_factor': a,
        'coordinate_frame': 'original_box_comoving_kpc_h', 'native_snapshot_sources': snapshot_sources,
        'apertures': rows, 'source_hashes': supplied,
        'scope': 'Actual native stellar progenitor cores and calibrated thin-stream cohort bounding spheres; no encounter or worldtube geometry inferred'}



def validate_catalogue_apertures(parameters):
    if type(parameters['maximum_apertures']) is not int or parameters['maximum_apertures'] < 1:
        raise ValueError('Exact positive stellar aperture allocation required')
    if type(parameters['memory_gib']) not in (int, float) or not math.isfinite(parameters['memory_gib']) or parameters['memory_gib'] <= 0:
        raise ValueError('Positive bounded stellar aperture memory required')



def native_catalogue_apertures_runner(parameters, context):
    from core.stream_common import read_product
    from .zoom import _snapshot_files
    from .data import numeric_audit_source_bindings
    validate_catalogue_apertures(parameters)
    if parameters['memory_gib'] > context.resources['memory_gib']: raise MemoryError('Stellar aperture producer exceeds stage allocation')
    path = Path(parameters['catalogue_product']).resolve()
    if path.stat().st_size * 8 > parameters['memory_gib'] * 1024**3:
        raise MemoryError('Native stellar catalogue metadata exceeds bounded intake')
    report, arrays, pins = read_product(path, memory_gib=parameters['memory_gib']/4)
    def merge(additional):
        for source, digest in additional.items():
            if source in pins and pins[source] != digest: raise ValueError('Conflicting stellar aperture source bindings')
            pins[source] = digest
    merge(numeric_audit_source_bindings(report))
    history = values = None
    if 'source_context' not in report and 'native_worldlines' in report:
        # The flagship catalogue producer uses a selected native worldline
        # cohort and an explicit spatial chart, rather than an entire FOF host.
        source = Path(report['native_worldlines']).resolve()
        if file_sha256(source) != report['native_worldlines_sha256']:
            raise ValueError('Original native stellar worldline selection changed')
        history, values, more = read_product(source, memory_gib=parameters['memory_gib']/4)
        merge(more); merge(numeric_audit_source_bindings(history))
        if history.get('schema') != 'measured-native-stream-worldlines-v1' or history['request']['species'] != 4:
            raise ValueError('Original native type-4 stellar history required')
        if Path(history['native_run_directory']).resolve() != Path(parameters['run_dir']).resolve():
            raise ValueError('Stellar history belongs to another native evolution')
        if history['namespace'] != report['namespace']: raise ValueError('Native stellar aperture identity namespace changed')
        ordinal = report['request']['epoch_index']
        if type(ordinal) is not int or not 0 <= ordinal < len(history['epochs']): raise ValueError('Exact selected native history epoch required')
        snapshot = history['epochs'][ordinal]['snapshot_storage_index']
    else: snapshot = report['source_context']['snapshot']
    if type(snapshot) is not int or snapshot < 0: raise ValueError('Exact selected native snapshot label required')
    files = _snapshot_files(parameters['run_dir'], snapshot)
    if not files: raise ValueError('Native stellar catalogue has no actual saved snapshot')
    snapshots = {str(Path(p).resolve()): file_sha256(p) for p in files}
    import h5py
    with h5py.File(files[0]) as native:
        header, model = native['Header'].attrs, native['Parameters'].attrs
        actual = (float(header['Time']), float(model['HubbleParam']), float(header['BoxSize']))
        if int(model['ComovingIntegrationOn']) != 1: raise ValueError('Cosmological native stellar aperture frame required')
    if history is not None:
        a, h, box = actual
        center = np.asarray(report['request']['chart_center_proper_kpc'], dtype=float)
        report = {**report, 'source_context': {'snapshot': snapshot, 'scale_factor': a, 'h': h,
            'box_comoving_kpc_h': box, 'host_center_comoving_kpc_h': (center*h/a).tolist(),
            'source_sha256': [{'path': p, 'sha256': digest} for p, digest in snapshots.items()]}}
        positions = values['position_kpc'][ordinal] - center
        if history['periodic']:
            proper_box = box*a/h; positions -= proper_box*np.floor(positions/proper_box+.5)
        for name, observed in (('position_kpc', positions), ('mass_msun', values['mass_msun'][ordinal]), ('particle_ids', values['particle_ids'])):
            if not np.array_equal(arrays[name], observed): raise ValueError('Catalogue differs from its original native worldline cohort')
        from .observation_geometry import read_native_gravitating_particles
        native_report, census = read_native_gravitating_particles(parameters['run_dir'], snapshot, species=[4],
            maximum_particles=history['request']['maximum_census_particles'], memory_gib=parameters['memory_gib']/4, namespace=history['namespace'])
        merge(native_report['source_hashes'])
        order = np.argsort(census['particle_ids']); at = np.searchsorted(census['particle_ids'][order], values['particle_ids'])
        if np.any(at >= len(order)) or not np.array_equal(census['particle_ids'][order[at]], values['particle_ids']):
            raise ValueError('Native stellar aperture cohort loses exact original identities')
        selected = order[at]
        if not np.array_equal(census['position_proper_kpc'][selected], values['position_kpc'][ordinal]) or not np.array_equal(census['mass_msun'][selected], values['mass_msun'][ordinal]):
            raise ValueError('Stellar worldline cohort differs from actual native saved positions/masses')
        del census, values, positions
    declared = tuple(report['source_context'][name] for name in ('scale_factor', 'h', 'box_comoving_kpc_h'))
    if any(type(value) not in (int, float) or not math.isclose(value, observed, rel_tol=1e-12, abs_tol=0)
           for value, observed in zip(declared, actual)):
        raise ValueError('Stellar aperture frame differs from actual native snapshot')
    if history is None:
        from .halos import _stellar_host_particles
        host = report['source_context']['host_fof_row']
        if type(host) is not int or host < 0: raise ValueError('Exact original native stellar host row required')
        positions, _, masses, identities, observed_context = _stellar_host_particles(
            Path(parameters['run_dir']), report['source_context']['snapshot'], host, parameters['memory_gib']/4)
        for name, observed in (('position_kpc', positions), ('mass_msun', masses), ('particle_ids', identities)):
            if not np.array_equal(arrays[name], observed):
                raise ValueError('Stellar aperture catalogue changes actual native membership or measured '+name)
        for name in ('host_center_comoving_kpc_h', 'source_sha256'):
            if report['source_context'][name] != observed_context[name]:
                raise ValueError('Stellar aperture catalogue changes its native host/source frame')
        del positions, masses, identities
    result = catalogue_apertures(report, arrays, snapshot_sources=snapshots, maximum_apertures=parameters['maximum_apertures'])
    for source, digest in {**report.get('source_hashes', {}), **result['source_hashes']}.items():
        if source in pins and pins[source] != digest: raise ValueError('Conflicting stellar aperture provenance')
        pins[source] = digest
    for source, digest in pins.items():
        if file_sha256(source) != digest: raise ValueError('Native stellar selection source changed before aperture publication')
    pins[str(Path(__file__).resolve())] = file_sha256(__file__)
    result['source_hashes'] = pins
    atomic_json(context.generation/'apertures.json', result)
    return result



def validate_encounter_apertures(parameters):
    for name in ('maximum_apertures','snapshot'):
        if type(parameters[name]) is not int or parameters[name] < (1 if name=='maximum_apertures' else 0):
            raise ValueError('Exact bounded native encounter '+name+' required')
    memory=parameters['memory_gib']
    if type(memory) not in (int,float) or not math.isfinite(memory) or memory<=0:
        raise ValueError('Positive native encounter aperture memory required')



def native_encounter_apertures_runner(parameters,context):
    from core.stream_common import read_product
    from .data import numeric_audit_source_bindings,exact_uint64_ids
    from .observation_geometry import read_native_gravitating_particles
    from .zoom import _snapshot_files
    import h5py
    validate_encounter_apertures(parameters)
    memory=parameters['memory_gib'];snapshot=parameters['snapshot']
    if memory>context.resources['memory_gib']:raise MemoryError('Encounter aperture producer exceeds stage allocation')
    pins={}
    def merge(additional):
        for path,digest in additional.items():
            if path in pins and pins[path]!=digest:raise ValueError('Conflicting native encounter aperture provenance')
            pins[path]=digest
    def load(path):
        path=Path(path).resolve()
        if path.stat().st_size*8>memory*1024**3/4:raise MemoryError('Native encounter metadata exceeds frozen intake allocation')
        report,arrays,sources=read_product(path,memory_gib=memory/8)
        merge(sources);merge(numeric_audit_source_bindings(report))
        return report,arrays,path
    events,_,event_path=load(parameters['encounter_product'])
    frame,converted,frame_path=load(parameters['stream_history'])
    if events.get('schema')!='stream-encounters-v1' or events.get('request',{}).get('schema')!='native-encounter-catalogue-v1':
        raise ValueError('Measured native encounter catalogue required for saved-time apertures')
    if frame.get('schema')!='native-encounter-frame-history-v1' or events['stream_history_sha256']!=file_sha256(frame_path):
        raise ValueError('Encounter catalogue does not bind this converted native stellar history')
    raw,values,raw_path=load(frame['source_history'])
    if (frame['source_history_sha256']!=file_sha256(raw_path) or raw.get('schema')!='measured-native-stream-worldlines-v1'
            or raw['request']['species']!=4 or raw['periodic'] is not True):
        raise ValueError('Original periodic native stellar worldlines required')
    if Path(raw['native_run_directory']).resolve()!=Path(parameters['run_dir']).resolve():
        raise ValueError('Encounter stellar history belongs to another native run')
    if frame['namespace']!=raw['namespace']:raise ValueError('Encounter frame changes native stellar namespace')
    labels=[row['snapshot_storage_index'] for row in raw['epochs']]
    if any(type(label) is not int for label in labels) or len(set(labels))!=len(labels) or snapshot not in labels:
        raise ValueError('Distinct exact actual saved native epochs required')
    ordinal=labels.index(snapshot);ids=exact_uint64_ids(values['particle_ids'])
    if len(np.unique(ids))!=len(ids):raise ValueError('Native encounter stellar identities are duplicated')
    for name in ('particle_ids','times_myr','mass_msun','periodic_box_proper_kpc'):
        if not np.array_equal(converted[name],values[name]):raise ValueError('Encounter frame changes the original native saved cohort/clock')
    origin=np.asarray(converted['origin_proper_kpc']);images=np.asarray(converted['recorded_image_offsets'])
    if origin.shape!=(len(labels),3) or images.shape!=values['position_kpc'].shape or images.dtype.kind not in 'iu':
        raise ValueError('Explicit original native frame origins and exact recorded spatial images required')
    expected=values['position_kpc']+images*np.asarray(values['periodic_box_proper_kpc'])[:,None,None]-origin[:,None,:]
    if not np.array_equal(converted['position_kpc'],expected):raise ValueError('Encounter frame differs from the actual recorded native positions/images')
    del expected,converted
    files=_snapshot_files(parameters['run_dir'],snapshot)
    if not files:raise ValueError('Actual saved native encounter snapshot required')
    snapshots={str(Path(path).resolve()):file_sha256(path) for path in files}
    if any(pins.get(path)!=digest for path,digest in snapshots.items()):raise ValueError('Encounter history does not bind the actual native snapshot')
    with h5py.File(files[0]) as native:
        header,model=native['Header'].attrs,native['Parameters'].attrs
        a,h,box=float(header['Time']),float(model['HubbleParam']),float(header['BoxSize'])
        if int(model['ComovingIntegrationOn'])!=1:raise ValueError('Cosmological native encounter aperture frame required')
    if not np.isfinite([a,h,box]).all() or min(a,h,box)<=0 or not math.isclose(float(values['periodic_box_proper_kpc'][ordinal]),box*a/h,rel_tol=1e-12):
        raise ValueError('Native encounter history and actual saved spatial frame differ')
    census_report,census=read_native_gravitating_particles(parameters['run_dir'],snapshot,species=[4],
        maximum_particles=raw['request']['maximum_census_particles'],memory_gib=memory/4,namespace=raw['namespace'])
    merge(census_report['source_hashes'])
    order=np.argsort(census['particle_ids']);at=np.searchsorted(census['particle_ids'][order],ids)
    if np.any(at>=len(order)) or not np.array_equal(census['particle_ids'][order[at]],ids):
        raise ValueError('Saved encounter aperture loses an exact native stellar identity')
    selected=order[at]
    if not np.array_equal(census['position_proper_kpc'][selected],values['position_kpc'][ordinal]) or not np.array_equal(census['mass_msun'][selected],values['mass_msun'][ordinal]):
        raise ValueError('Saved encounter aperture changes actual native positions/masses')
    del census
    radius=events['controls']['search_radius_kpc']
    if events['request']['controls']!=events['controls']:raise ValueError('Conflicting frozen encounter search controls')
    if type(radius) not in (int,float) or not math.isfinite(radius) or not 0<radius<box*a/h/2:
        raise ValueError('Positive unique periodic saved-time encounter search aperture required')
    rows=events['events']
    if not isinstance(rows,list) or len(rows)>parameters['maximum_apertures']:raise MemoryError('Encounter aperture count exceeds frozen allocation')
    lookup={raw['namespace']+':'+str(int(pid)):index for index,pid in enumerate(ids)}
    time=float(values['times_myr'][ordinal]);apertures=[];seen=set()
    for event in rows:
        identifier=event['event_id']
        if not isinstance(identifier,str) or not identifier or identifier in seen:raise ValueError('Distinct measured encounter identities required')
        seen.add(identifier)
        row={'id':identifier,'family':'encounter','event_time_myr':event['time_myr'],'saved_time_myr':time,
             'stream_identity':event['stream_identity'],'perturber_identity':event['perturber_identity']}
        if event['stream_identity'] not in lookup:raise ValueError('Encounter refers to another native stellar cohort')
        bracket=np.asarray(event['bracket_myr'],dtype=float)
        if bracket.shape!=(2,) or not np.isfinite(bracket).all() or bracket[1]<=bracket[0]:raise ValueError('Actual encounter saved-time bracket required')
        if event['sampling_status']!='WITHIN_RECORDED_GEOMETRY_BUDGET' or event['interpolation_certified'] is not True:
            row.update(status='UNRESOLVED',reason='Encounter geometry requires refinement; no accepted encounter aperture inferred')
        elif not bracket[0]<=time<=bracket[1]:
            row.update(status='UNRESOLVED',reason='This actual saved epoch lies outside the reconstructed encounter bracket')
        else:
            position=values['position_kpc'][ordinal,lookup[event['stream_identity']]]
            row.update(status='MEASURED',selection_identity=json.dumps({'event':identifier,'stream_identity':event['stream_identity'],
                'perturber_identity':event['perturber_identity'],'native_snapshot':snapshot},sort_keys=True),
                center_comoving_kpc_h=((position*h/a)%box).tolist(),radius_proper_kpc=radius,
                aperture_definition='Frozen encounter search-radius sphere at the actual saved native stellar position; not reconstructed event-time mass density')
        apertures.append(row)
    if not apertures:
        apertures=[{'id':'encounter-unresolved','family':'encounter','status':'UNRESOLVED',
                    'reason':'No reconstructed native encounter selected; no encounter aperture exists for measurement'}]
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Native encounter aperture input changed before publication')
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    result={'schema':'native-science-apertures-v1','snapshot':snapshot,'scale_factor':a,
        'coordinate_frame':'original_box_comoving_kpc_h','native_snapshot_sources':snapshots,'source_hashes':pins,
        'apertures':apertures,'scope':'Actual saved-time native encounter search support; geometry uncertainty, temporal gaps and reconstructed event time remain separate'}
    atomic_json(context.generation/'apertures.json',result)
    return result



def validate_recovered_apertures(parameters):
    for name in ('maximum_particles','maximum_apertures'):
        if type(parameters[name]) is not int or parameters[name]<1:raise ValueError('Positive exact recovered aperture '+name+' required')
    value=parameters['memory_gib']
    if type(value) not in (int,float) or not math.isfinite(value) or value<3*96/1024:raise ValueError('Bounded native recovered aperture memory required')
    snapshots=parameters['snapshots']
    if not isinstance(snapshots,list) or not snapshots or any(type(x) is not int or x<0 for x in snapshots) or len(set(snapshots))!=len(snapshots):
        raise ValueError('Distinct exact native recovered aperture snapshots required')
    fraction=parameters['maximum_coarse_mass_fraction']
    if type(fraction) not in (int,float) or not math.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('Frozen native contamination criterion required')



def validate_recovered_catalogue_controls(controls,memory_gib):
    required={'minimum_particles','maximum_neighbor_edges','minimum_neighbors','link_position_kpc','link_velocity_kms',
              'split_known_progenitors','minimum_posterior','minimum_heldout_precision'}
    if not isinstance(controls,dict) or set(controls)!=required:raise ValueError('Complete frozen native stellar catalogue controls required')
    for name,minimum in (('minimum_particles',3),('minimum_neighbors',2),('maximum_neighbor_edges',1)):
        if type(controls[name]) is not int or controls[name]<minimum:raise ValueError('Exact bounded native catalogue '+name+' required')
    for name in ('link_position_kpc','link_velocity_kms'):
        if type(controls[name]) not in (int,float) or not math.isfinite(controls[name]) or controls[name]<=0:raise ValueError('Positive native catalogue '+name+' required')
    for name in ('minimum_posterior','minimum_heldout_precision'):
        if type(controls[name]) not in (int,float) or not math.isfinite(controls[name]) or not 0<=controls[name]<=1:raise ValueError('Frozen classifier support criterion required')
    if type(controls['split_known_progenitors']) is not bool:raise ValueError('Explicit ancestry split policy required')
    if controls['maximum_neighbor_edges']*32>memory_gib*1024**3/9:raise MemoryError('Stellar catalogue neighbor budget exceeds allocated workspace')



def recovered_apertures_runner(parameters,context):
    from .zoom import read_host_catalogue,_snapshot_files
    from .halos import _read_catalogue_series
    from .data import convert_native_catalogue_kinematics
    from core.stream.workflow import Inputs, execute_stream_stage
    from core.stream_common import stream_array_units
    from core.stream_common import publish_stream_product
    import h5py
    validate_recovered_apertures(parameters);memory=parameters['memory_gib']
    if memory>context.resources['memory_gib']:raise MemoryError('Recovered apertures exceed campaign allocation')
    recovery_path=Path(parameters['recovery_path']).resolve();controls_path=Path(parameters['controls_path']).resolve()
    for path in (recovery_path,controls_path):
        if path.stat().st_size*8>memory*1024**3/8:raise MemoryError('Recovered aperture metadata exceeds bounded intake')
    if file_sha256(controls_path)!=parameters['controls_sha256']:raise ValueError('Frozen native stellar catalogue controls changed')
    recovery=json.loads(recovery_path.read_text());controls=json.loads(controls_path.read_text())
    validate_recovered_catalogue_controls(controls,memory)
    if recovery.get('schema')!='lagrangian-descendant-recovery-v1':raise ValueError('Actual persistent-membership recovery required')
    pins={str(recovery_path):file_sha256(recovery_path),str(controls_path):file_sha256(controls_path)}
    manifest=Path(parameters['run_dir']).resolve()/'manifest.json'
    pins[str(manifest)]=file_sha256(manifest)
    def merge(additional):
        for path,digest in additional.items():
            if path in pins and pins[path]!=digest:raise ValueError('Conflicting recovered aperture provenance')
            pins[path]=digest
    merge(recovery['source_hashes'])
    matches={}
    for row in recovery['records']:
        if type(row['snapshot']) is not int or row['snapshot'] in matches:raise ValueError('Distinct actual native recovery snapshots required')
        matches[row['snapshot']]=row
    if not set(parameters['snapshots'])<=set(matches):raise ValueError('Recovered aperture request loses an actual recovery epoch')
    encounters={}
    for selection in parameters.get('encounter_products',[]):
        if set(selection)!={'snapshot','encounter_product','stream_history'} or type(selection['snapshot']) is not int or selection['snapshot'] in encounters:
            raise ValueError('Distinct native encounter producer bindings required per saved aperture epoch')
        if selection['snapshot'] not in parameters['snapshots']:raise ValueError('Encounter product is outside requested saved aperture epochs')
        encounters[selection['snapshot']]=selection
    results=[]
    for snapshot in parameters['snapshots']:
        folder=context.generation/('snapshot-'+str(snapshot));folder.mkdir()
        files=_snapshot_files(parameters['run_dir'],snapshot)
        if not files:raise ValueError('Recovered aperture epoch has no actual native snapshot')
        sources={str(Path(path).resolve()):file_sha256(path) for path in files};merge(sources)
        with h5py.File(files[0]) as native:
            header,model=native['Header'].attrs,native['Parameters'].attrs
            a=float(header['Time']);h=float(model['HubbleParam'])
            unit_length=float(model['UnitLength_in_cm']);unit_velocity=float(model['UnitVelocity_in_cm_per_s'])
        match=matches[snapshot]
        if not math.isclose(a,match['scale_factor'],rel_tol=0,abs_tol=1e-12):raise ValueError('Recovered stellar aperture epoch differs from actual saved time')
        rows=[];catalogue_path=None
        def unresolved(family,reason):rows.append({'id':family+'-unresolved','family':family,'status':'UNRESOLVED','reason':reason})
        if match['status']!='MATCHED':
            for family in ('progenitor','stream'):unresolved(family,'Persistent host recovery is '+str(match['status']))
        else:
            host=match['descendant']['host_row']
            if type(host) is not int or host<0:raise ValueError('Exact recovered native host required for stellar selections')
            cat=read_host_catalogue(parameters['run_dir'],snapshot,memory_gib=memory/3,host_row=host)
            merge({row['path']:row['sha256'] for row in cat['sources']+cat['snapshot_sources']})
            if any(pins.get(source)!=digest for source,digest in match['source_hashes'].items()):
                raise ValueError('Recovered stellar aperture uses a different catalogue or snapshot')
            count=int(cat['lengths'][host,4])
            if count>parameters['maximum_particles']:raise MemoryError('Recovered native stellar population exceeds explicit census bound')
            if count==0:
                for family in ('progenitor','stream'):unresolved(family,'Recovered native FOF host has zero stellar members at this saved epoch')
            else:
                velocities=_read_catalogue_series(cat['catalogues'],'Group','GroupVel')
                bulk=convert_native_catalogue_kinematics(velocities[host:host+1],'GroupVel',scale_factor=a,hubble_param=h,
                    unit_length_cm=unit_length,unit_velocity_cm_s=unit_velocity)[0]
                provenance=product_provenance(parameters['run_dir'],algorithm=__file__,selection={'host_fof_row':host},
                    source_files=files,coordinates={'frame':'native periodic host-centered proper Cartesian'},snapshot=snapshot)
                namespace=provenance['identity_namespaces']['native_particle']
                inputs=Inputs(Path(parameters['run_dir']),{}, {'memory_gib':memory/3,'maximum_particles':parameters['maximum_particles']})
                refs={'run_directory':str(Path(parameters['run_dir']).resolve())}
                for name in ('identity_graph','calibration'):
                    if parameters.get(name):refs[name]=parameters[name]
                stage={'stage_id':'recovered-native-stellar-catalogue','kind':'catalogue','inputs':refs,
                    'parameters':{'snapshot':snapshot,'host_row':host,'particle_namespace':namespace,
                        'host_bulk_velocity_kms':bulk.tolist(),'controls':controls}}
                report,arrays,native_namespace,epoch=execute_stream_stage(stage,inputs,folder)
                merge(inputs.sources)
                for name in ('streams.py','halos.py','stars.py','data.py','foundation.py'):
                    path=Path(__file__).with_name(name).resolve();merge({str(path):file_sha256(path)})
                report.update(request=stage,source_hashes=dict(pins),execution_status='COMPLETE')
                catalogue_path=folder/'stellar-catalogue.json'
                publish_stream_product(catalogue_path,report,arrays,stream_array_units(arrays),sources=pins,namespace=native_namespace,epoch=epoch)
                del arrays,inputs,report
                selection_folder=folder/'stellar-selection';selection_folder.mkdir()
                selected=native_catalogue_apertures_runner({'run_dir':parameters['run_dir'],'catalogue_product':str(catalogue_path),
                    'memory_gib':memory,'maximum_apertures':parameters['maximum_apertures']},
                    SimpleNamespace(generation=selection_folder,resources=context.resources))
                rows.extend(selected['apertures']);merge(selected['source_hashes'])
        if snapshot in encounters:
            event_folder=folder/'encounter-selection';event_folder.mkdir()
            selected=native_encounter_apertures_runner({**encounters[snapshot],'run_dir':parameters['run_dir'],
                'memory_gib':memory,'maximum_apertures':parameters['maximum_apertures']},
                SimpleNamespace(generation=event_folder,resources=context.resources))
            rows.extend(selected['apertures']);merge(selected['source_hashes'])
        else:
            unresolved('encounter','No source-bound native encounter measurement supplied for this saved epoch; encounter aperture support remains unavailable')
        if len(rows)>parameters['maximum_apertures']:raise MemoryError('Combined recovered aperture count exceeds explicit allocation')
        for path,digest in pins.items():
            if file_sha256(path)!=digest:raise ValueError('Recovered aperture input changed during production')
        aperture_path=folder/'apertures.json'
        atomic_json(aperture_path,{'schema':'native-science-apertures-v1','snapshot':snapshot,'scale_factor':a,
            'coordinate_frame':'original_box_comoving_kpc_h','native_snapshot_sources':sources,'source_hashes':dict(pins),
            'apertures':rows,'scope':'Measured native selections only; unclassified populations and unavailable encounter products remain unresolved'})
        measured_folder=folder/'mass-contamination';measured_folder.mkdir()
        measured=aperture_contamination_runner({'run_dir':parameters['run_dir'],'aperture_product':str(aperture_path),
            'memory_gib':memory,'maximum_apertures':parameters['maximum_apertures'],
            'maximum_coarse_mass_fraction':parameters['maximum_coarse_mass_fraction']},
            SimpleNamespace(generation=measured_folder,resources=context.resources))
        result_path=measured_folder/'contamination.json'
        results.append({'snapshot':snapshot,'scale_factor':a,'status':measured['status'],
            'catalogue':({'path':str(catalogue_path),'sha256':file_sha256(catalogue_path)} if catalogue_path else None),
            'apertures':{'path':str(aperture_path),'sha256':file_sha256(aperture_path)},
            'contamination':{'path':str(result_path),'sha256':file_sha256(result_path)}})
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Recovered aperture input changed before history publication')
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    result={'schema':'native-recovered-science-aperture-history-v1','execution_status':'COMPLETE',
        'status':'PASS' if results and all(row['status']=='PASS' for row in results) else 'UNRESOLVED',
        'records':results,'source_hashes':pins,'science_acceptance':'NOT_ACCEPTED',
        'limitations':'Classification/ancestry/event data and numerical resolution determine support; successful execution never supplies missing detections or acceptance'}
    atomic_json(context.generation/'contamination-history.json',result)
    return result



# Measured fine-DM kernel support and cumulative omitted scattering in zooms

def validate_zoom_support(parameters):
    request=parameters['request']
    if request.get('schema')!='native-sidm-zoom-support-request-v1':
        raise ValueError('Explicit native SIDM zoom-support request required')
    for key in ('maximum_particles','maximum_native_particles','chunk_rows'):
        if type(request[key]) is not int or request[key]<=0:
            raise ValueError('Positive explicit native support allocation required: '+key)
    if isinstance(request['memory_gib'],bool) or not np.isfinite(request['memory_gib']) or request['memory_gib']<=0:
        raise ValueError('Positive finite native support workspace required')
    if type(request['snapshot']) is not int or request['snapshot']<0:
        raise ValueError('Exact actual native support snapshot required')
    ids=request['selected_particle_ids']
    if not isinstance(ids,list) or not ids or len(ids)>request['maximum_particles'] or any(type(i) is not int or not 0<=i<2**64 for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Distinct bounded exact selected fine-DM identities required')
    if request['maximum_particles']*256+request['chunk_rows']*128>request['memory_gib']*1024**3:
        raise MemoryError('Native support arrays and streaming workspace exceed allocation')
    for field in ('kernel_coarse_support_fraction','omitted_scattering_upper_bound'):
        value=request['maximum'][field]
        if isinstance(value,bool) or not np.isfinite(value) or value<0 or field=='kernel_coarse_support_fraction' and value>1:
            raise ValueError('Predeclared physical native support limit required: '+field)
    if request.get('omitted_bound_scope')!='cumulative_since_native_initialization':
        raise ValueError('Native omitted-scattering bound retains its cumulative source history')
    if not isinstance(request.get('identity_namespace'),str) or not request['identity_namespace']:
        raise ValueError('Native fine-particle support identities require their actual namespace')


def zoom_support_input_bindings(parameters):
    """Bind original native support fields to their completed evolution payload."""
    from .workflow import verify_completed_receipt
    from .native import original_native_snapshot_files
    from .data import verify_native_hdf5_set, native_publication_markers

    validate_zoom_support(parameters)
    path = Path(parameters['native_stage_receipt']).resolve()
    if path.stat().st_size * 8 > parameters['request']['memory_gib'] * 1024**3:
        raise MemoryError('Native support receipt exceeds retained input workspace')
    receipt = verify_completed_receipt(path)
    generation = Path(receipt['generation']).resolve()
    files = original_native_snapshot_files(parameters['native_run_directory'], parameters['request']['snapshot'])
    verify_native_hdf5_set(files, require_publication=True)
    sources = {str(path): file_sha256(path)}
    _merge(sources, receipt['source_hashes'])
    _merge(sources, receipt['input_hashes'])
    for file in [*files, *native_publication_markers(files)]:
        file = Path(file).resolve()
        if not file.is_relative_to(generation):
            raise ValueError('Zoom support source is outside the actual completed native evolution generation')
        checksum = file_sha256(file)
        if receipt['files_sha256'].get(file.relative_to(generation).as_posix()) != checksum:
            raise ValueError('Zoom support source differs from the completed native evolution payload')
        _merge(sources, {str(file): checksum})
    return sources


def publish_native_zoom_support(parameters,destination):
    """Read selected fine-DM kernel fractions and cumulative omitted depth.

    The kernel diagnostic is coarse partner mass density / all partner mass
    density, with self-pairs excluded. Omitted scattering is a dimensionless
    cumulative optical-depth upper bound, not an observed collision count.
    Each selected exact ID keeps its caller-specified output order.
    """
    import h5py
    from .data import export_numeric_audit,exact_uint64_ids,snapshot_total_counts,run_epoch_contract,isolated_epoch_contract
    from .native import original_native_snapshot_files
    request=parameters['request']
    sources=zoom_support_input_bindings(parameters)
    selected=np.asarray(request['selected_particle_ids'],dtype=np.uint64)
    order=np.argsort(selected)
    sorted_ids=selected[order]
    n=len(selected)
    values={key:np.full(n,np.nan) for key in ('kernel_coarse_support_fraction','omitted_scattering_upper_bound')}
    found=np.zeros(n,dtype=bool)
    fields={'kernel_coarse_support_fraction':'SIDMKernelCoarseSupportFraction',
        'omitted_scattering_upper_bound':'SIDMOmittedScatteringUpperBound'}
    reference=None
    examined=0
    for file in original_native_snapshot_files(parameters['native_run_directory'],request['snapshot']):
        with h5py.File(file,'r') as handle:
            header=handle['Header'].attrs
            native_count=int(snapshot_total_counts(header)[1])
            if native_count>request['maximum_native_particles']:
                raise MemoryError('Native fine-particle census exceeds support intake work bound')
            params=handle['Parameters'].attrs
            time=float(header['Time']);comoving=bool(params['ComovingIntegrationOn'])
            current=(time,comoving,float(params['HubbleParam']),float(params['UnitLength_in_cm'])/float(params['UnitVelocity_in_cm_per_s']))
            if reference is not None and current!=reference:
                raise ValueError('Native zoom-support shards change physical epoch or unit conventions')
            reference=current
            if 'PartType1' not in handle:continue
            group=handle['PartType1'];count=len(group['ParticleIDs']);examined+=count
            if examined>request['maximum_native_particles']:
                raise MemoryError('Actual native support streaming work exceeds the fine-particle bound')
            required=['ParticleIDs','SIDMCoarseGuardVersion',*fields.values()]
            if any(key not in group or len(group[key])!=count for key in required):
                raise ValueError('Actual native snapshot lacks measured coarse-support fields or their version census')
            for start in range(0,count,request['chunk_rows']):
                stop=min(count,start+request['chunk_rows'])
                ids=exact_uint64_ids(group['ParticleIDs'][start:stop],unique=True)
                # Reuse one exact-ID index for both membership and output order.
                location=np.searchsorted(sorted_ids,ids)
                keep=location<n
                keep[keep]=sorted_ids[location[keep]]==ids[keep]
                if not keep.any():continue
                version=np.asarray(group['SIDMCoarseGuardVersion'][start:stop])[keep]
                if version.dtype.kind not in 'iu' or np.any(version!=1):
                    raise ValueError('Selected fine particles have unmeasured native coarse-support state')
                targets=order[location[keep]]
                if found[targets].any():raise ValueError('Selected native fine identity repeats across snapshot shards')
                found[targets]=True
                for field,dataset in fields.items():
                    measured=np.asarray(group[dataset][start:stop],float)[keep]
                    if not np.isfinite(measured).all() or np.any(measured<0) or field=='kernel_coarse_support_fraction' and np.any(measured>1):
                        raise ValueError('Selected native coarse-support field has invalid physical values')
                    values[field][targets]=measured
    if reference is None or not found.all():
        raise ValueError('Requested fine-particle support selection is absent from the actual native snapshot')
    time,comoving,h,time_unit=reference
    epoch=run_epoch_contract(parameters['native_run_directory'],request['snapshot'],time) if comoving else isolated_epoch_contract(
        time,time_unit_gyr=time_unit/h/31557600/1e9,snapshot=request['snapshot'])
    maxima={key:float(value.max()) for key,value in values.items()}
    gates={key:maximum<=request['maximum'][key] for key,maximum in maxima.items()}
    report={'schema':'native-sidm-zoom-support-v1','request_hash':canonical_digest(request),
        'epoch':epoch,'snapshot':request['snapshot'],'selected_fine_particle_count':n,
        'maximum_measured':maxima,'predeclared_maximum':request['maximum'],'gates':gates,
        'status':'PASS' if all(gates.values()) else 'FAIL','execution_status':'COMPLETE','source_hashes':sources,
        'kernel_fraction_definition':'coarse partner kernel mass density / all partner kernel mass density; excludes self-pairs',
        'omitted_bound_scope':request['omitted_bound_scope'],
        'omitted_bound_definition':'upper bound on cumulative omitted fine/coarse scattering optical depth; not a count of observed collisions',
        'science_acceptance':'CONTROL_ONLY','native_stage_receipt':str(Path(parameters['native_stage_receipt']).resolve())}
    source=Path(__file__).resolve();sources[str(source)]=file_sha256(source)
    for path,digest in sources.items():
        if file_sha256(path)!=digest:raise ValueError('Native zoom-support sources changed before publication')
    arrays={'particle_ids':selected,**values}
    specifications={key:{'units':'dimensionless','annotations':{'meaning':key.replace('_',' ')}} for key in values}
    specifications['particle_ids']={'units':'identity','identity_namespace':request['identity_namespace']}
    return export_numeric_audit(destination,report,arrays,specifications,algorithm=source,
        context={'epoch':epoch,'coordinates':{'selection':'exact native fine-DM particle identities'}},source_hashes=sources)


def native_zoom_support_runner(parameters,context):
    if parameters['request']['memory_gib']>context.resources['memory_gib']:
        raise MemoryError('Native SIDM zoom-support workspace exceeds allocated stage memory')
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Native support output escapes its owned generation')
    return publish_native_zoom_support(parameters,destination)


def native_zoom_support_main():
    import argparse
    parser=argparse.ArgumentParser(description='Publish measured native coarse-kernel and omitted-scattering zoom bounds')
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.request.stat().st_size*8>16*1024**2:raise MemoryError('Native support request exceeds bounded metadata intake')
    parameters=json.loads(args.request.read_text())
    print(json.dumps(publish_native_zoom_support(parameters,args.output),indent=2))
