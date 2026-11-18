"""Native baryon histories and cooling/birth/cloud recipes through existing estimators."""
from pathlib import Path
import json
import math
import numpy as np
from .foundation import atomic_json, file_sha256, KPC_CM
from .foundation import merge_source_bindings as _merge

VARIANTS=['gas_phase_diagram','star_formation_history','mass_return','metals_history',
          'luminosity_history','color_history','first_cooling_birth','primordial_cloud_morphology']


def _positive(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError('Positive finite native baryon allocation required')


def _bound(binding,memory):
    if set(binding)!={'path','sha256'}:raise ValueError('Exact baryon path/hash binding required')
    path=Path(binding['path']).resolve()
    if path.stat().st_size*16>memory*1024**3:raise MemoryError('Baryon request metadata exceeds allocation')
    if file_sha256(path)!=binding['sha256']:raise ValueError('Native baryon source changed')
    return path


def baryon_figure_inputs(parameters):
    """Validate the original hydro history and frozen six-figure design.
    
    parameters names hydro_history, design_path, design_sha256 and memory_gib.
    Verify gas phase-bin definitions, synthesis backend hashes and two distinct
    color filters against the pinned FSPS model. Return the design settings,
    hydro history report and combined source hashes. This performs bounded
    intake and backend setup without exporting figures."""
    from .baryon_products import hydro_baryon_inputs,gas_phase_inputs
    from .stellar_population import pinned_fsps
    memory=parameters['memory_gib'];_positive(memory)
    report,_,pins=hydro_baryon_inputs(parameters['hydro_history'],memory_gib=memory)
    design=_bound({'path':parameters['design_path'],'sha256':parameters['design_sha256']},memory)
    settings=json.loads(design.read_text())
    if set(settings)!={'schema','gas_phase','synthesis','color_filters'} or settings['schema']!='native-baryon-figure-design-v1':
        raise ValueError('Explicit gas edges and pinned SSP model/color design required')
    binding={'path':str(Path(parameters['hydro_history']).resolve()),'sha256':file_sha256(parameters['hydro_history'])}
    gas_phase_inputs({'schema':'native-gas-phase-request-v1','hydro_history':binding,**settings['gas_phase']},memory_gib=memory)
    _,model=pinned_fsps(settings['synthesis']['backend']);_merge(pins,model['source_hashes'])
    colors=settings['color_filters']
    if not isinstance(colors,list) or len(colors)!=2 or colors[0]==colors[1] or any(c not in model['filters'] for c in colors):
        raise ValueError('Two distinct model-pinned color filters required')
    _merge(pins,{str(design):file_sha256(design)})
    return settings,report,pins


def native_baryon_figures_runner(parameters,context):
    """Export six baryon-history recipes inside an owned workflow generation.
    
    Read the sealed hydro history and frozen gas/synthesis design through
    baryon_figure_inputs. Export gas phase, SFR, mass return, metal, luminosity
    and color histories under context.generation. Population synthesis calls the
    pinned FSPS backend selected by the design. Original support masks and native
    accounting status remain in the output contracts.
    
    Write baryon-figures.json with figure count, variants and source hashes and
    return the same result mapping. Missing synthesis support, invalid designs
    and insufficient stage memory raise before successful completion."""
    from .baryon_products import (
        export_native_gas_phase, export_native_sfr_history, export_native_mass_return,
        export_native_metal_history, export_native_population_history,
    )
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Baryon figures exceed stage allocation')
    settings,report,pins=baryon_figure_inputs(parameters);memory=parameters['memory_gib'];out=context.generation
    binding={'path':str(Path(parameters['hydro_history']).resolve()),'sha256':file_sha256(parameters['hydro_history'])}
    galleries=[export_native_gas_phase({'schema':'native-gas-phase-request-v1','hydro_history':binding,**settings['gas_phase']},out/'gas',memory_gib=memory)]
    for name,export in [('sfr',export_native_sfr_history),('returns',export_native_mass_return),('metals',export_native_metal_history)]:
        galleries.append(export(parameters['hydro_history'],out/name,memory_gib=memory))
    galleries.append(export_native_population_history({'schema':'native-population-history-request-v1','hydro_history':binding,
        **{k:settings[k] for k in ('synthesis','color_filters')}},out/'population',memory_gib=memory))
    result={'schema':'native-baryon-figures-v1','execution_status':'COMPLETE','variants':VARIANTS[:6],
        'figure_count':sum(len(g['figures']) for g in galleries),'native_accounting_status':report['status'],
        'science_acceptance':'NOT_ACCEPTED','source_hashes':pins}
    atomic_json(out/'baryon-figures.json',result);return result


def native_resolution_fields(run,snapshot,metadata,graph,namespace,*,memory_gib):
    """Original native Hsml and exact saved ownership; never infer smoothing."""
    import h5py
    from .halos import _snapshot_files
    from .data import exact_uint64_ids,native_physical_unit_scales
    p=metadata['parameters'];scales=native_physical_unit_scales(**{k:p[k] for k in
        ('unit_mass_g','unit_length_cm','unit_velocity_cm_s','comoving','hubble_param')})
    factor=scales['comoving_length_cm' if p['comoving'] else 'length_cm']/KPC_CM*(metadata['native_time'] if p['comoving'] else 1.)
    ids=[];lengths=[];pins={};count=0
    for file in _snapshot_files(run,snapshot):
        path=Path(file).resolve();pins[str(path)]=file_sha256(path)
        with h5py.File(path) as f:
            if 'PartType0' not in f:continue
            group=f['PartType0'];n=len(group['ParticleIDs']);count+=n
            if count*256>memory_gib*1024**3:raise MemoryError('Native gas smoothing census exceeds allocation')
            if 'SmoothingLength' not in group:raise ValueError('Original native SmoothingLength output required; no separation estimate')
            identity=exact_uint64_ids(group['ParticleIDs'][:],allow_empty=True);h=np.asarray(group['SmoothingLength'][:],float)
            if h.shape!=(n,) or not np.isfinite(h).all() or np.any(h<=0):raise ValueError('Positive aligned native gas smoothing lengths required')
            ids.append(identity);lengths.append(h*factor)
    ids=np.concatenate(ids) if ids else np.empty(0,np.uint64)
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate exact gas IDs across native smoothing shards')
    owners={};valid=set(map(int,ids))
    for edge in graph['edges']:
        if edge['kind']!='bound_owner':continue
        src=edge['source'];dst=edge['target']
        if src[0]!=namespace or src[2]!=snapshot or src[1] not in valid:continue
        key=str(src[1])
        if key in owners and owners[key]!=dst:raise ValueError('Ambiguous native gas bound ownership')
        owners[key]=dst
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Native smoothing source changed while reading')
    return {'particle_ids':ids,'smoothing_length_proper_kpc':np.concatenate(lengths) if lengths else np.empty(0),
            'host_key_by_gas_id':owners},pins


def cooling_birth_inputs(parameters):
    """Bind a native gas history to its identity graph and host assembly product.
    
    parameters supplies hydro_history, identity_graph, assembly_product and the
    hash-bound cooling design. Verify the graph files, exact run and epoch
    agreements, host branch keys, saved snapshots and numerical product sources.
    Return hydro report, graph, assembly report, assembly arrays, design settings
    and source hashes. These checks prevent mixing gas measurements from a
    separate native run or assigning them to unrelated host observations."""
    from .baryon_products import hydro_baryon_inputs
    from .halos import verify_identity_graph
    from core.stream_common import read_product
    from .data import numeric_audit_source_bindings
    memory=parameters['memory_gib'];_positive(memory)
    report,_,pins=hydro_baryon_inputs(parameters['hydro_history'],memory_gib=memory)
    design=_bound({'path':parameters['design_path'],'sha256':parameters['design_sha256']},memory)
    settings=json.loads(design.read_text());required={'schema','cooling','cloud','streaming_ics','bound_systems'}
    if set(settings)!=required or settings['schema']!='native-cooling-birth-design-v1':raise ValueError('Complete original cooling/cloud design required')
    graphdir=Path(parameters['identity_graph']).resolve()
    if sum((graphdir/n).stat().st_size for n in ('report.json','index.sqlite','nodes.npz'))*24>memory*1024**3:
        raise MemoryError('Native first-star graph exceeds allocation')
    graph=verify_identity_graph(graphdir)
    _merge(pins,{str(graphdir/n):file_sha256(graphdir/n) for n in ('report.json','index.sqlite','nodes.npz','report.sha256')})
    _merge(pins,graph['sources'])
    assembly,arrays,apins=read_product(parameters['assembly_product'],memory_gib=memory/3)
    _merge(pins,apins);_merge(pins,numeric_audit_source_bindings(assembly));_merge(pins,{str(design):file_sha256(design)})
    from . import source_path
    owner=source_path('assembly')
    if assembly.get('schema')!='halo-main-branch-v1' or pins.get(str(owner))!=file_sha256(owner):
        raise ValueError('Original source-bound native assembly producer required')
    if assembly['source_graph_fingerprint']!=graph['fingerprint']:raise ValueError('First-star ancestry and assembly graph generations differ')
    branch=assembly['branch_identity_history'];snapshots=[e['snapshot'] for e in report['epochs']]
    if len(branch)!=len(assembly['native_branch']) or any(k[2] not in snapshots for k in branch):
        raise ValueError('Assembly branch does not use actual native saved baryon epochs')
    from .halos import _snapshot_files
    for meta in report['epochs']:
        epoch=graph['epochs'].get(str(meta['snapshot']),graph['epochs'].get(meta['snapshot']))
        if epoch!=meta['epoch']:raise ValueError('Cooling ancestry graph and native census actual epoch contracts differ')
        for path in _snapshot_files(report['request']['run_directory'],meta['snapshot']):
            if graph['sources'].get(str(Path(path).resolve()))!=file_sha256(path):
                raise ValueError('Cooling ancestry graph must bind this actual native snapshot generation')
    return report,graph,assembly,arrays,settings,pins


def native_cooling_births_runner(parameters,context):
    """Measure cooling, native birth ancestry and cloud geometry at saved epochs.
    
    Load the inputs with cooling_birth_inputs, then read each original gas census
    and measured smoothing lengths. Match native birth-journal IDs to graph
    observations and join cooling to births through exact gas parents and host
    branches. Derive primordial capability support from the evolved native
    network. cloud_morphology measures candidates separately at each epoch.
    
    Write first-stars.json, cloud-NNN.json and their sealed numeric bundles,
    registered figure galleries and cooling-births.json in context.generation.
    Return the summary with joined birth count, cloud statuses, figure count and
    source hashes. Resolution/physics decisions remain in the numerical products."""
    from .stellar_population import gas_epoch,native_births
    from .first_stars import identify_cooling_events,join_births_and_assembly,primordial_cloud_support,link_cooling_births_and_bound_systems,cloud_morphology
    from core.stream_common import publish_stream_product
    from .first_stars import export_native_first_star_figures,export_native_cloud_figures
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native cooling producer exceeds stage allocation')
    hydro,graph,assembly,history,design,pins=cooling_birth_inputs(parameters);memory=parameters['memory_gib']
    run=hydro['request']['run_directory'];namespace=hydro['provenance']['identity_namespaces']['native_particle']
    branch=assembly['branch_identity_history'];host_by_epoch={k[2]:k for k in branch};epochs=[]
    for meta in hydro['epochs']:
        resolution,sources=native_resolution_fields(run,meta['snapshot'],meta,graph,namespace,memory_gib=memory/4);_merge(pins,sources)
        epoch,sources=gas_epoch(run,meta['snapshot'],memory_bytes=int(memory*1024**3/4),namespace=namespace,
            host_identity=host_by_epoch.get(meta['snapshot']),resolution_fields=resolution)
        _merge(pins,sources);epochs.append(epoch)
    times=[e['time_myr'] for e in epochs]
    if len(times)<2 or np.any(np.diff(times)<=0):raise ValueError('Increasing actual native proper-time cooling epochs required')
    # A native birth is observed at the first graph node of that exact SSP ID.
    # The physical application time still comes from the original birth journal.
    birth_epochs={}
    for node in graph['nodes']:
        key=node['key']
        if node['kind'] in {'ssp','star'} and key[0]==namespace and key[2] in host_by_epoch:
            birth_epochs[str(key[1])]=min(key[2],birth_epochs.get(str(key[1]),key[2]))
    births=[]
    for before,after in zip(epochs[:-1],epochs[1:]):
        rows,sources=native_births(run,before['native_metadata'],after['native_metadata'],journal_directory=Path(run)/'data',
            namespace=namespace,epoch_by_id=birth_epochs,memory_bytes=int(memory*1024**3/4),observation_epoch=after['native_metadata']['snapshot'])
        births.extend(rows);_merge(pins,sources)
    cooling=identify_cooling_events(epochs,design['cooling'])
    joined=join_births_and_assembly(graph,births,history,{'searched_interval_myr':[times[0],times[-1]],'branch_identity_history':branch})
    # Capabilities are measured from the evolved native model, never asserted by request.
    enabled=[name for name,value in epochs[-1]['actual_native_physics'].items() if value]
    if any(e['actual_native_physics']!=epochs[-1]['actual_native_physics'] for e in epochs):raise ValueError('Cooling native physics changes across history')
    capabilities={'enabled':enabled,'source_hashes':pins}
    support=primordial_cloud_support(capabilities,design['streaming_ics'],design['cloud']['resolution'])
    bound={**design['bound_systems'],'epoch_time_myr':{str(e['snapshot']):t for e,t in zip(hydro['epochs'],times)}}
    linked=link_cooling_births_and_bound_systems(cooling,joined,graph,assembly,bound)
    report={'schema':'first-star-environments-v1',**cooling,**joined,**linked,'primordial_support':support,'host_mass_definition':assembly.get('mass_definition'),
        'host_identity_history':branch,'science_acceptance':'NOT_ACCEPTED'}
    arrays={'cooling_event_time_myr':np.asarray([r['event_time_myr'] for r in cooling['cooling_events']]),
            'birth_time_myr':np.asarray([r['applied_time_myr'] for r in joined['native_births']])}
    from . import source_path
    for source in (Path(__file__).resolve(),source_path('first_stars'),source_path('stellar_population')):_merge(pins,{str(source):file_sha256(source)})
    first=context.generation/'first-stars.json'
    publish_stream_product(first,report,arrays,{k:'Myr' for k in arrays},sources=pins,namespace=namespace)
    galleries=[export_native_first_star_figures(first,context.generation/'first-star-figures',memory_gib=memory)]
    statuses=[]
    for i,epoch in enumerate(epochs):
        # The existing native evolution build explicitly enables PERIODIC.
        from .halos import _snapshot_files
        import h5py
        with h5py.File(_snapshot_files(run,epoch['native_metadata']['snapshot'])[0]) as f:
            box=float(f['Header'].attrs['BoxSize'])
        p=epoch['native_metadata']['parameters']
        epoch['gas']['periodic_box_proper_kpc']=box*p['unit_length_cm']/(p['hubble_param'] if p['comoving'] else 1.)*(epoch['native_metadata']['native_time'] if p['comoving'] else 1.)/KPC_CM
        measured,arrays=cloud_morphology(epoch['gas'],design['cloud'],support=support)
        measured.update(source_hashes=pins,periodic_wrapping='Original wrapped native points; compact saved-time minimum-image cloud moments; ambiguous half-box groups explicitly rejected',
            origin='native simulation Cartesian origin')
        units={k:'proper kpc' if k in {'cloud_position_kpc','cloud_center_kpc','cloud_axes_kpc'} or k.endswith('_kpc') else
            'identity' if k.endswith('particle_ids') else 'Msun' if k.endswith('_msun') else 'km/s' if k.endswith('_kms') else 'dimensionless' for k in arrays}
        path=context.generation/f'cloud-{i:03d}.json';publish_stream_product(path,measured,arrays,units,sources=pins,namespace=namespace,epoch=epoch['gas']['epoch'])
        galleries.append(export_native_cloud_figures(path,context.generation/f'cloud-figures-{i:03d}',memory_gib=memory));statuses.append(measured['status'])
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Native cooling sources changed during publication')
    result={'schema':'native-cooling-birth-result-v1','execution_status':'COMPLETE','variants':VARIANTS[6:],
        'cloud_statuses':statuses,'native_joined_birth_count':len(joined['native_births']),
        'figure_count':sum(len(g['figures']) for g in galleries),'science_acceptance':'NOT_ACCEPTED','source_hashes':pins}
    atomic_json(context.generation/'cooling-births.json',result);return result


def prepare_native_baryon_campaign(request,destination):
    """Prepare a three-stage workflow for an existing native baryon run.
    
    request supplies the exact run and saved labels, source-accounting tolerances,
    memory/disk budgets, hash-bound figure/cooling designs, identity graph and
    assembly product. Verify the actual snapshot inventory and collect all input
    and producer source hashes. Build the dependency graph for a hydro journal
    audit, six baryon figures and the cooling/birth/cloud products.
    
    Write request.json, workflow.json and manifest.json to a new destination.
    Return the prepared campaign manifest with workflow path/checksum, requested
    variants and source bindings. Execution is a later workflow operation."""
    from .workflow import compile_requested_product_dag,default_registry
    from .stellar_returns import validate_stellar_history_stage
    from .analysis import snapshot_ids
    required={'schema','run_directory','snapshots','tolerances','memory_gib','disk_gib','figure_design','cooling_design','identity_graph','assembly_product'}
    if set(request)!=required or request['schema']!='native-baryon-campaign-v1':raise ValueError('Complete native eight-variant baryon request required')
    memory=request['memory_gib'];_positive(memory);_positive(request['disk_gib']);out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native baryon preparation')
    run=Path(request['run_directory']).resolve();audit={k:request[k] for k in ('snapshots','tolerances','memory_gib')};audit['run_dir']=str(run)
    validate_stellar_history_stage(audit)
    if snapshot_ids(run)!=request['snapshots']:raise ValueError('Baryon recipe must use exact actual saved snapshot set')
    pins={};designs={}
    for name in ('figure_design','cooling_design','assembly_product'):
        path=_bound(request[name],memory);designs[name]=str(path);pins[str(path)]=file_sha256(path)
        if name!='assembly_product':
            design=json.loads(path.read_text());schema='native-baryon-figure-design-v1' if name=='figure_design' else 'native-cooling-birth-design-v1'
            keys={'schema','gas_phase','synthesis','color_filters'} if name=='figure_design' else {'schema','cooling','cloud','streaming_ics','bound_systems'}
            if set(design)!=keys or design['schema']!=schema:raise ValueError('Complete native '+name+' contract required')
    graph=Path(request['identity_graph']).resolve()
    for name in ('report.json','index.sqlite','nodes.npz','report.sha256'):pins[str(graph/name)]=file_sha256(graph/name)
    from .halos import _snapshot_files
    for name in ('config.yaml','manifest.json'):pins[str(run/name)]=file_sha256(run/name)
    for snapshot in request['snapshots']:
        for path in _snapshot_files(run,snapshot):pins[str(Path(path).resolve())]=file_sha256(path)
    ref={'stage':'baryon-journal','artifact':'hydro-history.json'}
    stages=[{'stage_id':'baryon-journal','kind':'native_stellar_history','parameters':audit,'outputs':{'baryon-journal':'hydro-history.json'}},
        {'stage_id':'baryon-figures','kind':'native_baryon_figures','parameters':{'hydro_history':ref,'memory_gib':memory,
            'design_path':designs['figure_design'],'design_sha256':request['figure_design']['sha256']},'outputs':{'baryon-figures':'baryon-figures.json'}},
        {'stage_id':'baryon-cooling','kind':'native_cooling_births','parameters':{'hydro_history':ref,'memory_gib':memory,
            'design_path':designs['cooling_design'],'design_sha256':request['cooling_design']['sha256'],
            'identity_graph':str(graph),'assembly_product':designs['assembly_product']},'outputs':{'baryon-cooling':'cooling-births.json'}}]
    for stage in stages:stage['input_files']=[{'path':p,'sha256':h} for p,h in sorted(pins.items())]
    spec={'schema':'research-workflow-v1','resources':{'mpi_ranks':1,'omp_threads':1,'memory_gib':memory,'disk_gib':request['disk_gib'],'maximum_stage_count':len(stages)},
        'physics_spec':{},'capabilities':[],'followup_limits':{'maximum_children':0,'maximum_depth':0},
        'requested_products':['baryon-figures','baryon-cooling'],'stages':stages}
    registry=default_registry();plan=compile_requested_product_dag(spec,registry,request_base=out);code={}
    for stage in plan['stages']:_merge(code,registry.source_hashes(stage['kind']))
    for path,digest in {**pins,**code}.items():
        if file_sha256(path)!=digest:raise ValueError('Native baryon source changed during preparation')
    out.mkdir(parents=True);atomic_json(out/'request.json',request);atomic_json(out/'workflow.json',spec)
    result={'schema':'native-baryon-campaign-prepared-v1','status':'RECIPES_PREPARED','stage_count':len(stages),'variants':VARIANTS,
        'workflow':{'path':str(out/'workflow.json'),'sha256':file_sha256(out/'workflow.json')},'source_hashes':{**pins,**code},
        'science_acceptance':'NOT_ACCEPTED','scope':'Original source audit, six baryon history exports and native gas smoothing/cooling/birth ancestry/cloud producers. Domain and resolution failures remain visible.'}
    atomic_json(out/'manifest.json',result);return result
