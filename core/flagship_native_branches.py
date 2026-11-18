"""Original recorded native ancestry to assembly, cooling and web histories."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
from .foundation import atomic_json,file_sha256
from .foundation import merge_source_bindings as _merge


def native_branch_inputs(parameters):
    from .flagship_runtime_display import history_case_inputs
    from .workflow import positive_number
    if set(parameters)!={'case','design','memory_gib','maximum_metadata_files'}:raise ValueError('Complete owned native branch request required')
    positive_number(parameters['memory_gib'],'branch memory');positive_number(parameters['maximum_metadata_files'],'branch metadata limit',True)
    records,pins=history_case_inputs([parameters['case']],parameters['memory_gib'],parameters['maximum_metadata_files'])
    def bind(ref):
        if set(ref)!={'path','sha256'}:raise ValueError('Exact native branch design input required')
        path=Path(ref['path']).resolve()
        if file_sha256(path)!=ref['sha256']:raise ValueError('Native branch design changed')
        _merge(pins,{str(path):ref['sha256']});return path
    path=bind(parameters['design'])
    if path.stat().st_size*32>parameters['memory_gib']*1024**3/8:raise MemoryError('Native branch controls exceed metadata bound')
    design=json.loads(path.read_text())
    required={'schema','limits','measurement_controls','history','cooling_design','web_design','maximum_interval_myr'}
    if set(design)!=required or design['schema']!='flagship-native-branches-design-v1':raise ValueError('Complete original ancestry/assembly/cooling/web controls required')
    limits=design['limits']
    if set(limits)!={'maximum_nodes','maximum_edges','maximum_subhalos','maximum_particles','minimum_host_particles','maximum_measurements'}:raise ValueError('Complete native graph census limits required')
    for key,value in limits.items():positive_number(value,key,True)
    positive_number(design['maximum_interval_myr'],'web history cadence')
    for key in ('cooling_design','web_design'):
        if design[key] is not None:bind(design[key])
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    return records[0],design,pins


def native_branch_runner(parameters,context):
    from .analysis import snapshot_ids
    from .native_gadget_ancestry import build_native_fof_ancestry
    from .halo_campaign import native_assembly_measurements_runner, native_assembly_history_runner
    from .native_baryon_campaign import native_cooling_births_runner
    from .halo_campaign import native_web_epoch_runner,native_web_history_runner
    from core.stream_common import read_product
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native branch exceeds selected allocation')
    (case,owned),design,pins=native_branch_inputs(parameters);out=context.generation/'products';out.mkdir()
    memory=parameters['memory_gib'];run=Path(case['run']).resolve();epochs=snapshot_ids(run)
    def sub(name):
        path=out/name;path.mkdir();return SimpleNamespace(generation=path,resources=context.resources)
    def bound(path):return {'path':str(path),'sha256':file_sha256(path)}
    def owned_record(role):
        matches=[(p,m) for p,m in owned if p['role']==role]
        if len(matches)!=1:raise ValueError('One original owned '+role+' required for native history branch')
        p,metadata=matches[0];path=Path(p['reference']['path']).resolve()
        return path,next(value for source,value in metadata if source==path)
    recovery_path,recovery=owned_record('recovery_stage')
    if not epochs or [row['snapshot'] for row in recovery['records']]!=epochs:raise ValueError('Native history branch must retain the exact saved recovery epoch set')
    last=recovery['records'][-1];unresolved=[]
    if last['status']!='MATCHED':
        result={'schema':'flagship-native-branches-result-v1','execution_status':'COMPLETE','status':'UNRESOLVED_TARGET',
            'native_run':str(run),'source_hashes':pins,'unresolved':[{'reason':'Final target has no unambiguous material recovery; no substitute host selected','recovery_status':last['status']}],
            'science_acceptance':'NOT_ACCEPTED'}
        atomic_json(out/'branch-result.json',result);return result
    limits=design['limits'];graph,missing=build_native_fof_ancestry(run,epochs,out/'native-graph',native_sources=pins,memory_gib=memory,
        **{key:limits[key] for key in limits if key!='maximum_measurements'})
    unresolved.extend({'role':'recorded_ancestry','record':row} for row in missing)
    namespace=str(run)+':fof:catalogue-row';target=[namespace,last['descendant']['host_row'],epochs[-1]]
    if not any(row['key']==target for row in graph['nodes']):raise ValueError('Recovered final target absent from original recorded native graph')
    observations=[];structures={}
    for product,metadata in owned:
        if product['role']!='structure':continue
        path=Path(product['reference']['path']).resolve();status=next(value for source,value in metadata if source==path)
        if status['status']!='MEASURED':
            unresolved.append({'role':'structure','snapshot':status['snapshot'],'status':status['status']});continue
        reference=status['measurement'];path=Path(reference['path']).resolve()
        if pins.get(str(path))!=reference['sha256']:raise ValueError('Native branch structure is outside its original completed producer')
        key=[namespace,status['host_row'],status['snapshot']]
        observations.append({'key':key,'structure':reference});structures[tuple(key)]=str(path)
    graph_path=out/'native-graph'
    if observations:
        template=out/'measurement-template.json';atomic_json(template,{'schema':'native-assembly-measurements-v1','controls':design['measurement_controls'],'observations':observations})
        measured=sub('assembly-measurements')
        native_assembly_measurements_runner({'identity_graph':str(graph_path),'template_path':str(template),'template_sha256':file_sha256(template),
            'memory_gib':memory,**{key:limits[key] for key in ('maximum_nodes','maximum_edges','maximum_measurements')}},measured)
        graph_path=measured.generation/'identity-graph'
    else:unresolved.append({'role':'assembly','reason':'No supported recovered structure observations; native FOF mass ancestry retained, other observables remain unavailable'})
    template=out/'history-template.json';atomic_json(template,{'schema':'native-assembly-history-template-v1','target':target,**copy.deepcopy(design['history'])})
    assembly_context=sub('assembly-history')
    native_assembly_history_runner({'identity_graph':str(graph_path),'template_path':str(template),'template_sha256':file_sha256(template),'memory_gib':memory},assembly_context)
    assembly=assembly_context.generation/'assembly.json';history,_,_=read_product(assembly,memory_gib=memory/4)
    branch={tuple(key) for key in history['branch_identity_history']}
    if design['cooling_design'] is not None:
        from .flagship_native_births import build_native_birth_graph
        hydro,hydro_record=owned_record('hydro_history_stage');settings=design['cooling_design']
        birth_graph,birth_target,birth_missing=build_native_birth_graph(run,epochs,out/'birth-ownership',
            final_host_row=last['descendant']['host_row'],namespace=hydro_record['provenance']['identity_namespaces']['native_particle'],
            native_sources=pins,limits=limits,memory_gib=memory,epoch_contracts={row['snapshot']:row['epoch'] for row in hydro_record['epochs']})
        unresolved.extend({'role':'cooling_ancestry','record':row} for row in birth_missing)
        if birth_graph is not None:
            template=out/'birth-history-template.json';atomic_json(template,{'schema':'native-assembly-history-template-v1','target':birth_target,**copy.deepcopy(design['history'])})
            birth_history=sub('birth-assembly-history')
            native_assembly_history_runner({'identity_graph':str(birth_graph),'template_path':str(template),'template_sha256':file_sha256(template),'memory_gib':memory},birth_history)
            native_cooling_births_runner({'hydro_history':str(hydro),'memory_gib':memory,'design_path':settings['path'],'design_sha256':settings['sha256'],
                'identity_graph':str(birth_graph),'assembly_product':str(birth_history.generation/'assembly.json')},sub('cooling'))
        else:unresolved.append({'role':'cooling','reason':'No unique native bound central in recovered final FOF host; no FOF membership substitute used'})
    if design['web_design'] is not None:
        web=[];settings=design['web_design']
        for key in sorted(branch,key=lambda row:row[2]):
            if key not in structures:continue
            ctx=sub('web-snapshot-'+str(key[2]));source=structures[key]
            native_web_epoch_runner({'run_directory':str(run),'snapshot':key[2],'halo_structure_product':source,'filament_skeleton':None,
                'design_path':settings['path'],'design_sha256':settings['sha256'],'memory_gib':memory},ctx)
            web.append({'environment_product':str(ctx.generation/'web-environment.json'),'halo_structure_product':source,'graph_node':list(key)})
        if len(web)>=2:
            native_web_history_runner({'identity_graph_directory':str(graph_path),'observations':web,'memory_gib':memory,
                'maximum_interval_myr':design['maximum_interval_myr']},sub('web-history'))
        else:unresolved.append({'role':'web_history','reason':'Fewer than two original measured structure observations on the same recorded native assembly branch'})
    for path,sha in pins.items():
        if file_sha256(path)!=sha:raise ValueError('Native branch source changed during measurement')
    result={'schema':'flagship-native-branches-result-v1','execution_status':'COMPLETE','status':'MEASURED_WITH_SUPPORT_MASKS',
        'native_run':str(run),'source_hashes':pins,'original_graph':bound(out/'native-graph/report.json'),
        'assembly_product':bound(assembly),'unresolved':unresolved,'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(out/'branch-result.json',result);return result
