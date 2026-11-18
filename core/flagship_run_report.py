"""Report original workflow evidence without promoting it to science acceptance."""
from pathlib import Path
import json
from .foundation import atomic_json,atomic_text,file_sha256


CONNECTED_KINDS={'flagship_parent_preparation','flagship_zoom_preparation','flagship_runtime_history',
    'flagship_runtime_measurements','flagship_runtime_display','combined_graph_gallery','flagship_runtime_scene'}


def _native_inventory(run,receipt):
    """Read owned saved headers and identity metadata; never infer missing IDs."""
    import h5py
    from .analysis import snapshot_ids,load_cfg
    from .halos import _snapshot_files
    from .foundation import run_namespace_binding
    cfg=load_cfg(run);root=Path(receipt['generation']).resolve();rows=[]
    for snapshot in snapshot_ids(run):
        shards=[]
        for source in _snapshot_files(run,snapshot):
            path=Path(source).resolve();relative=path.relative_to(root).as_posix()
            if receipt['files_sha256'].get(relative)!=file_sha256(path):raise ValueError('Saved census source is outside original native ownership')
            with h5py.File(path) as handle:
                header=handle['Header'].attrs;species=[]
                for kind in range(6):
                    group=handle.get('PartType%d'%kind)
                    if group is None:continue
                    ids=group.get('ParticleIDs');namespace=None if ids is None else ids.attrs.get('identity_namespace')
                    if isinstance(namespace,bytes):namespace=namespace.decode()
                    species.append({'particle_type':kind,'particle_count':None if ids is None else int(ids.size),
                        'identity_dtype':None if ids is None else str(ids.dtype),'identity_namespace':namespace,
                        'identity_status':'EXACT_UINT64_STORAGE' if ids is not None and str(ids.dtype)=='uint64' else 'UNAVAILABLE_OR_INVALID'})
                shards.append({'source':{'path':str(path),'sha256':receipt['files_sha256'][relative]},
                    'native_time':float(header['Time']),'redshift':float(header['Redshift']),
                    'particles_this_file':[int(x) for x in header['NumPart_ThisFile']],
                    'mass_table_native':[float(x) for x in header['MassTable']],'species':species})
        rows.append({'snapshot_storage_index':snapshot,'shards':shards})
    return {'native_run':str(run),'requested_redshifts':cfg['simulation']['snapshot_redshift_list'],
        'run_namespace':run_namespace_binding(run),'saved_epochs':rows,
        'identity_scope':'Owned native uint64 storage metadata and full-file hashes; global uniqueness, ancestry and exact selected memberships remain in original handoff/recovery products.'}


def publish_run_report(destination,*,resources,operation,connected=False):
    from .workflow import verify_completed_receipt
    out=Path(destination).resolve();state_path=out/'workflow.json';state=json.loads(state_path.read_text())
    maximum=resources.get('maximum_stage_count',10000);limit=resources['memory_gib']*1024**3/4
    if len(state.get('stages',{}))>maximum:raise MemoryError('Run report stage inventory exceeds allocation')
    seen=set();artifacts=[];stages=[];native=[];galleries=[];scenes=[];renders=[];display=[];samples=[];unresolved=[];contracts=[];count=0;used=0
    schemas={'lagrangian-descendant-recovery-v1','native-zoom-contamination-history-v1',
        'native-recovered-science-aperture-history-v1','native-recovered-stream-measurement-history-v1',
        'native-recovered-encounter-history-v1','recovered-native-halo-structure-v1',
        'native-catalogue-stream-measurement-result-v1',
        'flagship-native-branches-result-v1','native-cooling-birth-result-v1',
        'flagship-runtime-measurements-prepared-v1','flagship-runtime-display-prepared-v1'}
    for sid,entry in state.get('stages',{}).items():
        root=Path(entry['generation']).resolve();path=root/'stage-receipt.json'
        if not root.is_relative_to(out/'generations') or file_sha256(path)!=entry['receipt_sha256']:raise ValueError('Run report stage differs from original workflow ownership')
        receipt=verify_completed_receipt(path)
        if receipt['stage_id']!=sid or receipt['fingerprint']!=entry['fingerprint']:raise ValueError('Run report stage identity changed')
        seen.add(receipt['kind']);stages.append({'stage_id':sid,'kind':receipt['kind'],'receipt':{'path':str(path),'sha256':entry['receipt_sha256']},
            'measurements':receipt.get('measurements',{}),'validation_receipts':receipt.get('validation_receipts',[]),
            'stored_bytes':sum((root/name).stat().st_size for name in receipt['files_sha256'])})
        if receipt['kind']=='public_command' and 'run/manifest.json' in receipt['files_sha256'] and 'invocation.json' in receipt['files_sha256']:
            invocation=json.loads((root/'invocation.json').read_text())
            if 'gadget' in invocation.get('argv',invocation.get('command',[])):
                native.append(_native_inventory(root/'run',receipt))
        for name,sha in receipt['files_sha256'].items():
            source=root/name
            if source.suffix!='.json':continue
            count+=1;size=source.stat().st_size
            if count>50000:raise MemoryError('Run report metadata file census exceeds bound')
            if size>16*1024**2:continue
            used+=size*16
            if used>limit:raise MemoryError('Run report retained metadata exceeds allocation')
            value=json.loads(source.read_text())
            if not isinstance(value,dict):continue
            ref={'path':str(source),'sha256':sha};schema=value.get('schema');kind=receipt['kind']
            numeric=value.get('numeric_data');contract=numeric.get('array_contract') if isinstance(numeric,dict) else None
            if isinstance(contract,dict):
                declarations=contract.get('datasets',{})
                contracts.append({'stage_id':sid,'reference':ref,'schema':schema,'epoch':contract.get('epoch'),
                    'coordinates':contract.get('coordinates'),
                    'datasets':{key:{name:spec.get(name) for name in ('units','coordinates','identity_namespace')} for key,spec in declarations.items()}})
            if schema in schemas or name in {'epoch-coverage.json','restart-comparison.json'} or 'handoff' in source.name:
                counts={}
                for row in value.get('records',[]) if isinstance(value.get('records',[]),list) else []:
                    status=row.get('status','UNREPORTED') if isinstance(row,dict) else 'UNREPORTED';counts[status]=counts.get(status,0)+1
                    if isinstance(status,str) and status.startswith(('UNRESOLVED','AMBIGUOUS','DISRUPTED','UNSUPPORTED','NO_')):
                        unresolved.append({'stage_id':sid,'evidence':ref,'record':{key:row[key] for key in ('snapshot','candidate_id','status','reason') if key in row}})
                artifacts.append({'stage_id':sid,'kind':kind,'schema':schema,'reference':ref,
                    'status':value.get('status',value.get('execution_status')),'record_status_counts':counts})
                pending=value.get('unresolved',[])
                if isinstance(pending,list):
                    for row in pending:unresolved.append({'stage_id':sid,'evidence':ref,'record':row})
            if name=='local-resource-samples.json':samples.append({'stage_id':sid,'reference':ref,
                'sampled_local_peak_rss_bytes':value.get('sampled_local_peak_rss_bytes'),'coverage':value.get('coverage_status')})
            if schema=='combined-scientific-gallery-v1' and kind=='combined_graph_gallery':
                entries=value['entries'];keys={(row['family'],row['variant']) for row in entries}
                registry=json.loads((Path(__file__).resolve().parents[1]/'configs/research/graph_producers.json').read_text())
                expected={(row['family'],row['variant']) for row in registry['entries']}
                if value['variant_count']!=102 or len(entries)!=102 or len(keys)!=102 or keys!=expected:raise ValueError('Run report requires exactly102 distinct original graph entries')
                galleries.append({'reference':ref,'variant_count':102,'exported_views':len(value['views']),
                    'variants':[{'family':row['family'],'variant':row['variant'],'measurement_status':row['measurement_status'],'view_ids':row['view_ids']} for row in entries]})
            if schema=='flagship-runtime-display-prepared-v1' and kind=='flagship_runtime_display':display.append(value)
            if schema=='flagship-runtime-scene-result-v1' and kind=='flagship_runtime_scene':scenes.append({'stage_id':sid,'reference':ref,
                'native_run':value['native_run'],'scene':value['scene'],'render_ids':value['render_ids'],'scientific_gallery_views':value['scientific_gallery_views']})
            if schema=='saved-scene-render-v1' and kind=='flagship_runtime_scene':renders.append({'stage_id':sid,'reference':ref,'render_id':source.parent.parent.name,
                'projection':value['request']['projection'],'frame_count':len(value['artifacts']),'camera':value['camera'],'execution_status':value['execution_status']})
    missing=[]
    if connected not in {False,True,'production'}:raise ValueError('Explicit supported connected report mode required')
    if connected:
        required=CONNECTED_KINDS if operation in {'test','run'} else {'flagship_retained_operation','flagship_runtime_display','combined_graph_gallery','flagship_runtime_scene'}
        if connected=='production' and operation=='run':required={'flagship_production_admission','zoom_material_recovery','native_stellar_history','flagship_runtime_measurements','flagship_runtime_display','combined_graph_gallery','flagship_runtime_scene'}
        missing.extend({'kind':kind,'reason':'Required connected producer has no completed original receipt'} for kind in sorted(required-seen))
        completed=set(state.get('stages',{}))
        for owner in display:
            missing.extend({'stage_id':sid,'reason':'Declared scene followup has no completed receipt'} for sid in owner['scene_stage_ids'] if sid not in completed)
        for scene in scenes:
            present={row['render_id'] for row in renders if row['stage_id']==scene['stage_id'] and row['execution_status']=='COMPLETE'}
            missing.extend({'stage_id':scene['stage_id'],'render_id':name,'reason':'Declared render lacks its owned completed render receipt'} for name in scene['render_ids'] if name not in present)
    status=state.get('execution_status','UNKNOWN')
    if status=='COMPLETE' and missing:status='INCOMPLETE'
    result={'schema':'flagship-run-report-v1','execution_status':status,'scheduler_execution_status':state.get('execution_status'),
        'operation':operation,'workflow':{'path':str(state_path),'sha256':file_sha256(state_path)},'stages':stages,
        'native_saved_inventory':native,'audit_products':artifacts,'graph_galleries':galleries,'scenes':scenes,'renders':renders,
        'unit_frame_identity_contracts':contracts,
        'resource_samples':samples,'retained_stage_storage_bytes':sum(row['stored_bytes'] for row in stages),
        'unresolved':unresolved,'missing_connections':missing,'remaining_requirements':state.get('remaining_requirements',[]),
        'scientific_exclusions':state.get('scientific_exclusions',[]),'science_acceptance':'NOT_ACCEPTED',
        'report_implementation':{'path':str(Path(__file__).resolve()),'sha256':file_sha256(__file__)},
        'scope':'Verified original completed stage receipts and owned metadata; missing science support stays unresolved. Storage includes retained stage payloads; RSS sampling coverage is reported exactly and is not a machine-wide peak certificate.'}
    path=out/'run-report.json';atomic_json(path,result);atomic_text(path.with_name(path.name+'.sha256'),file_sha256(path)+'\n')
    return result
