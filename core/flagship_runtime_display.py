"""Completed native history inventories to original gallery and scene producers."""
from pathlib import Path
import copy
import json
from .foundation import atomic_json,file_sha256
from .foundation import merge_source_bindings as _merge


ROLES={'recovery_stage':'zoom_material_recovery','contamination_stage':'zoom_contamination_history',
    'hydro_history_stage':'native_stellar_history','stream_measurement_stage':'native_recovered_stream_measurements',
    'native_encounter_stage':'recovered_native_encounters','structure':'zoom_recovered_structure',
    'analysis_stage':'public_command','figure_export':'flagship_product','bh_history':'native_bh_history',
    'bh_events':'native_bh_event_figures','baryon_figures':'native_baryon_figures','web_epoch':'native_web_epoch',
    'native_branches':'native_branch_products'}


def validate_scene_options(options):
    from .workflow import positive_number,safe_id
    required={'maximum_points','quality','frames_per_snapshot','maximum_saved_vertices','maximum_selection_points',
        'maximum_apertures','marker_radius_kpc','encounter_overlay','renders'}
    if set(options)!=required:raise ValueError('Complete native scene and render controls required')
    for key in required-{'quality','encounter_overlay','renders'}:positive_number(options[key],key,key!='marker_radius_kpc')
    if options['quality'] not in {'preview','high'} or not 100<=options['maximum_points']<=5000000 or not 2<=options['frames_per_snapshot']<=240:raise ValueError('Supported original scene sampling controls required')
    overlay=options['encounter_overlay']
    if set(overlay)!={'maximum_objects','maximum_vertices','marker_radius_mpc_h','line_width_mpc_h','event_marker_radius_kpc'}:raise ValueError('Complete native encounter overlay bounds required')
    for key in overlay:positive_number(overlay[key],key,key.startswith('maximum_'))
    renders=options['renders']
    if not isinstance(renders,list) or not renders or len({row['id'] for row in renders})!=len(renders):raise ValueError('Distinct explicit native render requests required')
    for row in renders:
        required={'id','projection','width','height','samples','frame_start','frame_end','fps','device','wall_seconds'}
        if set(row)-{'scene_name'}!=required:raise ValueError('Complete native render settings required')
        if 'scene_name' in row and (not isinstance(row['scene_name'],str) or not row['scene_name']):raise ValueError('Exact saved native scene name required')
        safe_id(row['id'])
        for key in required-{'id','projection','device'}:positive_number(row[key],key,True)
        if row['projection'] not in {'active','perspective','dome'} or row['device'] not in {'AUTO','CPU','CUDA','OPTIX','HIP','ONEAPI','METAL'}:raise ValueError('Supported native render projection/device required')
        if row['frame_end']<row['frame_start'] or row['projection']=='dome' and row['width']!=row['height']:raise ValueError('Ordered render coverage and square dome required')


def _render_storage_bound(options,memory,disk):
    total=0
    for row in options['renders']:
        pixels=row['width']*row['height']
        if pixels*128>memory*1024**3*.5:raise MemoryError('Native scene render raster exceeds allocation')
        total+=pixels*8*(row['frame_end']-row['frame_start']+1)
    if total>disk*1024**3*.8:raise MemoryError('Aggregate scene render sequence estimate exceeds allocation')
    return total


def display_preparation_inputs(parameters):
    from core.research.convergence import _owned
    from .sidm_analysis import _stage_native_bindings
    required={'schema','prefix','cases','memory_gib','disk_gib','maximum_metadata_files','maximum_figures','scene'}
    if set(parameters)!=required or parameters['schema']!='flagship-runtime-display-request-v1':raise ValueError('Complete completed-history display request required')
    from .workflow import positive_number,safe_id
    safe_id(parameters['prefix'])
    for key in ('memory_gib','disk_gib','maximum_metadata_files','maximum_figures'):
        positive_number(parameters[key],key,key.startswith('maximum_'))
    validate_scene_options(parameters['scene'])
    _render_storage_bound(parameters['scene'],parameters['memory_gib'],parameters['disk_gib'])
    records,pins=history_case_inputs(parameters['cases'],parameters['memory_gib'],parameters['maximum_metadata_files'])
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    return records,pins


def history_case_inputs(cases,memory_gib,maximum_metadata_files):
    """Use the same original native/history ownership gates for each consumer."""
    from core.research.convergence import _owned
    from .sidm_analysis import _stage_native_bindings
    from .workflow import safe_id
    if not isinstance(cases,list) or not cases or len({case['id'] for case in cases})!=len(cases):raise ValueError('Distinct explicit native display cases required')
    pins={};records=[];count=0;size=0
    for case in cases:
        if set(case)!={'id','run','native_receipt','products'}:raise ValueError('Complete native display case required')
        safe_id(case['id']);run=Path(case['run']).resolve()
        _merge(pins,_stage_native_bindings({'native_run_directory':str(run),'native_stage_receipt':case['native_receipt'],'request':{'memory_gib':memory_gib}}))
        manifest=run/'manifest.json'
        if pins.get(str(manifest))!=file_sha256(manifest) or json.loads(manifest.read_text()).get('status')!='SIMULATION_COMPLETE':raise ValueError('Original completed native display run required')
        _owned(manifest,case['native_receipt'],'public_command',pins)
        owned=[]
        for product in case['products']:
            if set(product)!={'role','kind','reference','receipt'} or ROLES.get(product['role'])!=product['kind']:raise ValueError('Original native history producer role required')
            ref=product['reference'];path=Path(ref['path']).resolve()
            if set(ref)!={'path','sha256'} or file_sha256(path)!=ref['sha256']:raise ValueError('Native display history changed')
            receipt=_owned(path,product['receipt'],product['kind'],pins);root=Path(receipt['generation']).resolve()
            files={str(root/name):sha for name,sha in receipt['files_sha256'].items()};_merge(pins,files)
            metadata=[]
            for name,sha in files.items():
                source=Path(name)
                if source.suffix!='.json' or source.name=='stage-receipt.json':continue
                count+=1;size+=source.stat().st_size*32
                if count>maximum_metadata_files or size>memory_gib*1024**3/4:raise MemoryError('Native display metadata inventory exceeds allocation')
                value=json.loads(source.read_text())
                if isinstance(value,dict):metadata.append((source,value))
            owned.append((product,metadata))
        records.append((case,owned))
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    for name,sha in pins.items():
        if file_sha256(name)!=sha:raise ValueError('Native display lineage changed')
    return records,pins


def display_preparation_runner(parameters,context):
    from .workflow import safe_id
    for key in ('memory_gib','disk_gib'):
        if parameters[key]>context.resources[key]:raise MemoryError('Native display exceeds launch '+key)
    records,pins=display_preparation_inputs(parameters);rows=[];galleries={};scenes=[];unresolved=[]
    def ref(path):return {'path':str(path),'sha256':file_sha256(path)}
    for case,owned in records:
        progenitors=[];contamination=[copy.deepcopy(product['reference']) for product,_ in owned if product['role']=='contamination_stage'];streams=[];overlays=[];seen=set()
        for product,metadata in owned:
            for path,value in metadata:
                if path in seen:continue
                seen.add(path)
                schema=value.get('schema')
                if schema=='scientific-figure-suite-v1':galleries[str(path)]=ref(path)
                if schema=='native-stream-progenitor-view-v1':progenitors.append(ref(path))
                if schema=='native-recovered-encounter-history-v1':
                    for ordinal,cohort in enumerate(value['records']):
                        if cohort['status']!='COMPLETE':
                            unresolved.append({'case':case['id'],'role':'encounters','status':cohort['status'],'reason':cohort.get('reason')});continue
                        raw=cohort['native_stream_history'];source=Path(raw['path']).resolve()
                        if pins.get(str(source))!=raw['sha256']:raise ValueError('Native worldlines are outside the owned history inventory')
                        sid=safe_id(parameters['prefix']+'_'+case['id']+'_stream_'+str(ordinal))
                        rows.append({'stage_id':sid,'kind':'flagship_product','parameters':{'operation':'stream-timeline','product':str(source),
                            'destination':'{generation}/inspector','memory_gib':parameters['memory_gib'],
                            'maximum_saved_vertices':parameters['scene']['maximum_saved_vertices']},
                            'input_files':[{'path':name,'sha256':sha} for name,sha in pins.items()], 'outputs':{sid:'inspector'}})
                        streams.append({'inspector':{'path':{'stage':sid,'artifact':'inspector/inspector.json'},
                            'sha256':{'stage':sid,'artifact':'inspector/inspector.json','load':'sha256'}},
                            'maximum_saved_vertices':parameters['scene']['maximum_saved_vertices'],'marker_radius_kpc':parameters['scene']['marker_radius_kpc']})
                        products=cohort['products']
                        overlay={key:copy.deepcopy(products[name]) for key,name in [('stream_history','stream-frame'),('perturber_history','perturbers'),('encounters','encounter-catalogue')]}
                        for binding in overlay.values():
                            if pins.get(str(Path(binding['path']).resolve()))!=binding['sha256']:raise ValueError('Encounter overlay is outside owned native history')
                        overlays.append(overlay)
                if schema in {'native-recovered-stream-measurement-history-v1','recovered-native-halo-structure-v1'} and value.get('status') in {'UNRESOLVED','UNRESOLVED_TARGET'}:
                    unresolved.append({'case':case['id'],'role':product['role'],'status':value['status']})
        # A scene per measured cohort keeps each encounter chart explicit.
        # The base scene still carries every stellar timeline and selection.
        for ordinal,overlay in enumerate([None,*overlays]):
            sid=safe_id(parameters['prefix']+'_'+case['id']+'_scene_'+str(ordinal))
            scenes.append({'stage_id':sid,'kind':'flagship_runtime_scene','parameters':{'schema':'flagship-runtime-scene-request-v1',
                'run':case['run'],'native_receipt':case['native_receipt'],'gallery':None,'streams':streams,
                'progenitors':progenitors,'contamination':contamination,'encounter_overlay':overlay,
                'analysis_binding':({'case':{**copy.deepcopy(case),'products':[copy.deepcopy(product) for product,_ in owned if product['role']=='analysis_stage']},
                    'maximum_metadata_files':parameters['maximum_metadata_files']} if any(product['role']=='analysis_stage' for product,_ in owned) else None),
                'options':copy.deepcopy(parameters['scene']),'memory_gib':parameters['memory_gib'],'disk_gib':parameters['disk_gib']},
                'input_files':[{'path':name,'sha256':sha} for name,sha in pins.items()],
                'outputs':{sid:'scene',sid+'_report':'scene-result.json'}})
    gallery_id=safe_id(parameters['prefix']+'_gallery')
    rows.append({'stage_id':gallery_id,'kind':'combined_graph_gallery','parameters':{'request':{'schema':'combined-scientific-gallery-request-v1',
        'galleries':list(galleries.values()),'scenes':[],'memory_gib':parameters['memory_gib'],'disk_gib':parameters['disk_gib'],
        'maximum_figures':parameters['maximum_figures']}},'input_files':[{'path':name,'sha256':sha} for name,sha in pins.items()],
        'outputs':{gallery_id:'gallery'}})
    for scene in scenes:
        scene['parameters']['gallery']={'path':{'stage':gallery_id,'artifact':'gallery/gallery.json'},'sha256':{'stage':gallery_id,'artifact':'gallery/gallery.json','load':'sha256'}}
    rows.extend(scenes)
    atomic_json(context.generation/'display-preparation.json',{'schema':'flagship-runtime-display-prepared-v1','execution_status':'RECIPES_PREPARED',
        'source_hashes':pins,'original_galleries':list(galleries.values()),'scene_stage_ids':[row['stage_id'] for row in scenes],
        'unresolved':unresolved,'science_acceptance':'NOT_ACCEPTED','native_scene_execution':'NOT_EXECUTED'})
    return {'followups':rows,'measurements':{'original_galleries':len(galleries),'scene_count':len(scenes),'unresolved':unresolved,'science_acceptance':'NOT_ACCEPTED'}}


def runtime_scene_runner(parameters,context):
    import sys
    from core.research.display import verify_combined_gallery
    from .sidm_analysis import _stage_native_bindings
    from core.research.convergence import _owned
    from core.native import run_native_command
    import commands
    required={'schema','run','native_receipt','gallery','streams','progenitors','contamination','encounter_overlay','options','memory_gib','disk_gib'}
    if set(parameters)-{'analysis_binding'}!=required or parameters['schema']!='flagship-runtime-scene-request-v1':raise ValueError('Complete native scene request required')
    for key in ('memory_gib','disk_gib'):
        if parameters[key]>context.resources[key]:raise MemoryError('Native scene exceeds launch allocation')
    pins=_stage_native_bindings({'native_run_directory':parameters['run'],'native_stage_receipt':parameters['native_receipt'],'request':{'memory_gib':parameters['memory_gib']}})
    native_manifest=Path(parameters['run']).resolve()/'manifest.json'
    _owned(native_manifest,parameters['native_receipt'],'public_command',pins)
    if json.loads(native_manifest.read_text()).get('status')!='SIMULATION_COMPLETE':raise ValueError('Completed original native scene evolution required')
    gallery=parameters['gallery'];path=Path(gallery['path']).resolve()
    if file_sha256(path)!=gallery['sha256']:raise ValueError('Native scene gallery changed')
    record=verify_combined_gallery(path,memory_gib=parameters['memory_gib']);run=Path(parameters['run']).resolve()
    private_sources={};copy_evidence=None
    if parameters.get('analysis_binding') is not None:
        from .flagship_runtime_measurements import private_analysis_scene_sources
        private_sources,analysis_pins,copy_evidence=private_analysis_scene_sources(parameters['analysis_binding'],run,parameters['memory_gib'])
        _merge(pins,analysis_pins)
    views=[row['view_id'] for row in record['views'] if any(Path(name).resolve().is_relative_to(run) or private_sources.get(str(Path(name).resolve()))==sha
        for name,sha in row['context']['source_products'].items())]
    options=parameters['options'];validate_scene_options(options)
    render_bytes=_render_storage_bound(options,parameters['memory_gib'],parameters['disk_gib']);overlay=parameters['encounter_overlay']
    if overlay is not None:overlay={**overlay,**options['encounter_overlay']}
    request={'schema':'combined-scene-attachments-request-v1','memory_gib':parameters['memory_gib'],'disk_gib':parameters['disk_gib'],
        'streams':parameters['streams'],'scientific_gallery':{'reference':gallery,'view_ids':views,
            **({'analysis_binding':parameters['analysis_binding']} if parameters.get('analysis_binding') is not None else {})},
        'native_selections':{'progenitor_views':parameters['progenitors'],'contamination':parameters['contamination'],
            'maximum_points':options['maximum_selection_points'],'maximum_apertures':options['maximum_apertures'],'marker_radius_kpc':options['marker_radius_kpc']},
        **({'encounter_overlay':overlay} if overlay is not None else {})}
    manifest=commands.export(run,options['maximum_points'],options['quality'],options['frames_per_snapshot'],
        scene_attachments=request,output_directory=context.generation/'scene')
    commands.build_scene(manifest,preview=False);scene=manifest.parent/'blender_universe.blend'
    used=sum(path.stat().st_size for path in context.generation.rglob('*') if path.is_file())
    if used+render_bytes>parameters['disk_gib']*1024**3:raise MemoryError('Scene assets and aggregate render estimate exceed allocation')
    rendered=[];unresolved_renders=[]
    for render in options['renders']:
        if render.get('scene_name','').startswith('Native | '):
            selected=json.loads((manifest.parent/'native_selections/selections.json').read_text())
            if not any(type(row['selection']) is int for row in selected['contamination_apertures']):
                unresolved_renders.append({'id':render['id'],'status':'UNRESOLVED_NATIVE_FOCUS',
                    'reason':'No measured recovered host support exists; no focus target or camera center invented'})
                continue
        arguments=[sys.executable,str(Path(__file__).resolve().parents[1]/'commands.py'),'render-scene',str(scene),'--output',str(context.generation/'scene'/render['id'])]
        for key,value in render.items():
            if key!='id':arguments.extend(['--'+key.replace('_','-'),str(value)])
        arguments.extend(['--memory-gib',str(parameters['memory_gib']),'--disk-gib',str(parameters['disk_gib'])])
        log=context.generation/('render-'+render['id']+'.log')
        with log.open('wb') as stream:
            run_native_command(arguments,cwd=context.generation,receipt_path=context.generation/('render-'+render['id']+'.invocation.json'),
                inputs=[scene],timeout=render['wall_seconds'],stdout=stream,stderr=stream)
        rendered.append(render['id'])
        if sum(path.stat().st_size for path in context.generation.rglob('*') if path.is_file())>parameters['disk_gib']*1024**3:raise MemoryError('Actual scene assets and renders exceed allocation')
    for name,sha in pins.items():
        if file_sha256(name)!=sha:raise ValueError('Native scene export mutated its original evolution inputs')
    atomic_json(context.generation/'scene-result.json',{'schema':'flagship-runtime-scene-result-v1','execution_status':'COMPLETE',
        'native_run':str(run),'source_hashes':pins,'manifest':{'path':str(manifest),'sha256':file_sha256(manifest)},
        'scene':{'path':str(scene),'sha256':file_sha256(scene)},'render_ids':rendered,'unresolved_renders':unresolved_renders,
        'scientific_gallery_views':views,'private_analysis_lineage':copy_evidence,'science_acceptance':'NOT_ACCEPTED'})
    return {'measurements':{'scientific_gallery_views':len(views),'render_count':len(rendered),'unresolved_render_count':len(unresolved_renders),'science_acceptance':'NOT_ACCEPTED'}}


def artwork_parent_render(row):
    """Select the profile's full-parent views; other views belong to zooms."""
    return row['id'].removesuffix('-still') in {'overview', 'dome', 'gas'}



def export_artwork_parent(run):
    """Keep a full-box movie of the actual uniform parent, outside its seal."""
    from core.workflow import verify_completed_receipt, execute_workflow
    import commands
    import subprocess
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    profile = json.loads((run/'artwork-profile.json').read_text())['template']
    state_path = run/'simulation/workflow.json'
    state = json.loads(state_path.read_text())
    entry = state['stages'][profile['parent']['prefix']+'_evolve']
    generation = Path(entry['generation']).resolve()
    if not generation.is_relative_to((run/'simulation/generations').resolve()): raise ValueError('Foreign parent generation')
    receipt_path = generation/'stage-receipt.json'
    if file_sha256(receipt_path) != entry['receipt_sha256']: raise ValueError('Parent owner changed')
    receipt = verify_completed_receipt(receipt_path)
    if receipt['kind'] != 'public_command': raise ValueError('Original completed native parent required')
    native = generation/'run'
    for name in ('config.yaml', 'manifest.json'):
        if receipt['files_sha256'].get('run/'+name) != file_sha256(native/name): raise ValueError('Parent output is not owned')
    if json.loads((native/'manifest.json').read_text())['status'] != 'SIMULATION_COMPLETE': raise ValueError('Completed parent required')
    # Run the existing finders and map/graph producers on their ordinary
    # immutable private copy. Never add catalogues to the sealed parent run.
    analysis_id = 'parent_artwork_analysis'
    analysis_stage = {'stage_id': analysis_id, 'kind': 'public_command',
        'parameters': {'command': 'analysis', 'wall_seconds': 2592000,
            'copy_run_input': 0, 'copy_run_max_bytes': int(profile['resources']['disk_gib']*1024**3*.25),
            'arguments': [{'input': str(native)}, '--graphs', 'hmf,pk,profiles,kinematics,spin,subhalos,hydro,maps', '--find-halos']},
        'input_files': [{'path': str(generation/name), 'sha256': digest} for name, digest in receipt['files_sha256'].items()]
            + [{'path': str(receipt_path), 'sha256': file_sha256(receipt_path)}],
        'outputs': {analysis_id: 'private_run'}}
    spec = {'schema': 'research-workflow-v1', 'physics_spec': {}, 'capabilities': [],
        'resources': {**profile['resources'], 'maximum_stage_count': 1},
        'followup_limits': {'maximum_children': 1, 'maximum_depth': 1},
        'stages': [analysis_stage], 'requested_products': [analysis_id]}
    atomic_json(run/'parent-analysis-request.json', spec)
    execute_workflow(spec, run/'parent-analysis', request_base=run)
    analysis_state = json.loads((run/'parent-analysis/workflow.json').read_text())
    analysis_generation = Path(analysis_state['stages'][analysis_id]['generation'])
    verify_completed_receipt(analysis_generation/'stage-receipt.json')
    display_source = analysis_generation/'private_run'
    scene_settings = profile['display']['scene']
    manifest = commands.export(display_source, scene_settings['maximum_points'], 'high', scene_settings['frames_per_snapshot'],
                               output_directory=run/'parent-display')
    commands.build_scene(manifest, preview=False)
    scene = manifest.parent/'blender_universe.blend'
    for render in scene_settings['renders']:
        if not artwork_parent_render(render): continue
        arguments = [sys.executable, str(ROOT/'commands.py'), 'render-scene', str(scene), '--output', str(manifest.parent/render['id'])]
        for key, value in render.items():
            if key != 'id': arguments.extend(['--'+key.replace('_', '-'), str(value)])
        for key in ('memory_gib', 'disk_gib'):
            arguments.extend(['--'+key.replace('_', '-'), str(profile['display'][key])])
        subprocess.run(arguments, cwd=ROOT, check=True)
    verify_completed_receipt(receipt_path)
    atomic_json(manifest.parent/'native-parent-origin.json', {'schema': 'native-parent-artwork-origin-v1',
        'native_run': str(native), 'original_receipt': str(receipt_path), 'receipt_sha256': file_sha256(receipt_path),
        'owned_private_analysis': str(analysis_generation), 'display_source': str(display_source),
        'original_workflow': str(state_path), 'workflow_sha256': file_sha256(state_path),
        'source_sha256': file_sha256(__file__), 'native_inputs_preserved': True, 'synthetic_particles_created': 0})



def encode_artwork_movies(run):
    """Encode owned native frame sequences; interpolation adds no saved epochs."""
    import shutil
    import subprocess
    videos = run/'videos'
    videos.mkdir(exist_ok=True)
    rows = []
    stills = []
    renders = sorted((run/'simulation').rglob('frames/render.json'))
    renders += sorted((run/'parent-display').rglob('frames/render.json'))
    for index, path in enumerate(renders):
        receipt = json.loads(path.read_text())
        if receipt['execution_status'] != 'COMPLETE': raise ValueError('Unfinished native render')
        if file_sha256(path) != path.with_suffix('.json.sha256').read_text().strip(): raise ValueError('Render seal changed')
        request = receipt['request']
        start = request['frame_start']
        end = request['frame_end']
        for item in receipt['artifacts']:
            if file_sha256(path.parent/item['path']) != item['sha256']: raise ValueError('Native render frame changed')
        if end == start:
            source = path.parent/receipt['artifacts'][0]['path']
            target = videos/('still-%03d-%s.png' % (index, request['projection']))
            if target.exists(): raise FileExistsError('Preserve existing artwork still')
            shutil.copy2(source, target)
            stills.append({'image': str(target), 'sha256': file_sha256(target), 'original_render': str(path),
                           'original_render_sha256': file_sha256(path)})
            continue
        target = videos/('sequence-%03d-%s.mp4' % (index, request['projection']))
        subprocess.run(['ffmpeg', '-nostdin', '-n', '-framerate', str(request['fps']), '-start_number', str(start),
            '-i', str(path.parent/'frame_%08d.png'), '-frames:v', str(end-start+1), '-c:v', 'libx264',
            '-crf', '16', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], check=True)
        rows.append({'video': str(target), 'sha256': file_sha256(target), 'original_render': str(path),
                     'original_render_sha256': file_sha256(path)})
    if not rows: raise ValueError('No completed native movies; inspect the original run report')
    atomic_json(videos/'inventory.json', {'videos': rows, 'stills': stills, 'scope': 'Presentation copies of original native display frames'})
