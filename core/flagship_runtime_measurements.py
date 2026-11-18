"""Connect completed native histories to the existing measurement/export stages."""
from pathlib import Path
import copy
import json
from .foundation import atomic_json,file_sha256
from .foundation import merge_source_bindings as _merge


def measurement_preparation_inputs(parameters):
    from .flagship_runtime_display import history_case_inputs
    from core.research.convergence import _owned
    from .power import power_input_bindings
    from .workflow import positive_number,safe_id
    required={'schema','prefix','cases','linear_bundle','linear_receipt','memory_gib','maximum_metadata_files','options','continuation'}
    if set(parameters)!=required or parameters['schema']!='flagship-runtime-measurements-request-v1':raise ValueError('Complete native measurement preparation required')
    safe_id(parameters['prefix']);positive_number(parameters['memory_gib'],'measurement memory')
    positive_number(parameters['maximum_metadata_files'],'metadata limit',True)
    options=parameters['options']
    if set(options)-{'branch_design'}!={'analysis_exports','black_holes','baryon_design','web_design'} or type(options['analysis_exports']) is not bool:raise ValueError('Explicit native measurement selections required')
    bh=options['black_holes']
    if set(bh)!={'enabled','pair_ids','recoil_selection'} or type(bh['enabled']) is not bool:raise ValueError('Explicit native BH measurement selections required')
    records,pins=history_case_inputs(parameters['cases'],parameters['memory_gib'],parameters['maximum_metadata_files'])
    def bind(ref):
        if set(ref)!={'path','sha256'}:raise ValueError('Exact native measurement input reference required')
        path=Path(ref['path']).resolve()
        if file_sha256(path)!=ref['sha256']:raise ValueError('Native measurement input changed')
        _merge(pins,{str(path):ref['sha256']});return path
    linear=bind(parameters['linear_bundle'])
    if parameters['linear_receipt'] is None:
        # Production may retain its original solver bundle as an evolution input.
        # Require actual native input ownership and exact saved configuration.
        from .sidm_analysis import _stage_native_bindings
        from .analysis import load_cfg
        for case,_ in records:
            native=_stage_native_bindings({'native_run_directory':case['run'],'native_stage_receipt':case['native_receipt'],'request':{'memory_gib':parameters['memory_gib']}})
            selected=Path(load_cfg(case['run'])['spectrum']['file']).resolve()
            if selected!=linear or native.get(str(linear))!=parameters['linear_bundle']['sha256']:
                raise ValueError('Retained linear bundle must be the exact owned native evolution input in every case')
            _merge(pins,native)
    else:_owned(linear,parameters['linear_receipt'],'public_command',pins)
    _merge(pins,power_input_bindings({'operation':'power-linear','product':str(linear),'memory_gib':parameters['memory_gib']}))
    for key in ('baryon_design','web_design','branch_design'):
        if options.get(key) is not None:bind(options[key])
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    for name,sha in pins.items():
        if file_sha256(name)!=sha:raise ValueError('Native measurement lineage changed')
    return records,pins


def verified_private_analysis(case,product,metadata,pins):
    """Keep the private analysis provenance, and verify original copied inputs."""
    from .analysis import verify_analysis_bundle
    provenance=Path(product['reference']['path']).resolve();bundle=provenance.parent
    record=verify_analysis_bundle(bundle,verify_sources=True);private=Path(record['source_run']).resolve();native=Path(case['run']).resolve()
    copies=[value for path,value in metadata if path.name=='run-copy.json' and Path(value.get('source','')).resolve()==native and Path(value.get('destination','')).resolve()==private]
    if len(copies)!=1 or bundle!=private/'results':raise ValueError('Analysis needs its original owned private-copy record')
    # Compare every retained native data input, not just counts, seeds or names.
    compared=[]
    for name,sha in pins.items():
        source=Path(name)
        if not source.is_relative_to(native):continue
        relative=source.relative_to(native)
        if relative.as_posix()!='config.yaml' and (not relative.parts or relative.parts[0]!='data'):continue
        copied=private/relative
        if not copied.is_file() or file_sha256(copied)!=sha:raise ValueError('Private analysis changed an original native data/configuration input')
        compared.append({'original':str(source),'copy':str(copied),'sha256':sha})
    if not compared or not any(Path(row['original']).suffix=='.hdf5' for row in compared):raise ValueError('Private analysis must retain actual original native HDF5 inputs')
    return bundle,compared


def private_analysis_scene_sources(binding,run,memory_gib):
    """Verify owned analysis and exact copied native inputs at each display boundary."""
    from .flagship_runtime_display import history_case_inputs
    if set(binding)!={'case','maximum_metadata_files'}:raise ValueError('Complete owned analysis scene binding required')
    case=binding['case']
    if Path(case['run']).resolve()!=Path(run).resolve():raise ValueError('Analysis scene binding belongs to another native run')
    if len(case['products'])!=1 or case['products'][0]['role']!='analysis_stage':raise ValueError('One original public analysis producer required for scene binding')
    records,pins=history_case_inputs([case],memory_gib,binding['maximum_metadata_files'])
    _,owned=records[0];bundle,compared=verified_private_analysis(case,*owned[0],pins)
    sources={name:sha for name,sha in pins.items() if Path(name).is_relative_to(bundle)}
    sources.update({row['copy']:row['sha256'] for row in compared})
    _merge(pins,sources)
    return sources,pins,{'original_native_run':str(Path(run).resolve()),'analysis_bundle':str(bundle),
        'byte_equivalent_original_inputs':compared,'meaning':'Original analysis source identity preserved; association proven by owned byte-equivalent native inputs.'}


def measurement_preparation_runner(parameters,context):
    from .workflow import safe_id
    from .analysis import load_cfg,snapshot_ids
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native measurements exceed launch allocation')
    records,pins=measurement_preparation_inputs(parameters);rows=[];cases=[];unresolved=[];copy_evidence=[]
    memory=parameters['memory_gib'];prefix=parameters['prefix'];options=parameters['options']
    def artifact(sid,name):return {'path':{'stage':sid,'artifact':name},'sha256':{'stage':sid,'artifact':name,'load':'sha256'}}
    def add(case,label,kind,params,outputs,role,report):
        sid=safe_id(prefix+'_'+case['id']+'_'+label)
        rows.append({'stage_id':sid,'kind':kind,'parameters':params,'input_files':[{'path':name,'sha256':sha} for name,sha in pins.items()],
            'outputs':{safe_id(sid+'_'+str(i)):name for i,name in enumerate(outputs)}})
        case['products'].append({'role':role,'kind':kind,'reference':artifact(sid,report),'receipt':{'stage':sid,'artifact':'stage-receipt.json'}})
        return sid
    def figures(case,label,operation,product):
        return add(case,label,'flagship_product',{'operation':operation,'product':str(product),
            'destination':'{generation}/products','memory_gib':memory},['products'],'figure_export','products/gallery/gallery.json')
    for original,owned in records:
        case=copy.deepcopy(original);cfg=load_cfg(case['run']);epochs=snapshot_ids(case['run'])
        if not epochs:raise ValueError('Actual native saved epochs required for measurements')
        hydro=[(product,metadata) for product,metadata in owned if product['role']=='hydro_history_stage']
        if len(hydro)!=1:raise ValueError('One original native hydro/source history required for selected combined physics')
        history=hydro[0][0]['reference']['path']
        journal=json.loads(Path(history).read_text())
        if Path(journal['request']['run_directory']).resolve()!=Path(case['run']).resolve():raise ValueError('Native measurement hydro history belongs to another run')
        for label,operation in [('star_formation','sfr-history-figures'),('mass_return','mass-return-figures'),('metals','metal-history-figures')]:figures(case,label,operation,history)
        if not cases:figures(case,'linear_power','power-linear',parameters['linear_bundle']['path'])
        if options['analysis_exports']:
            analyses=[(product,metadata) for product,metadata in owned if product['role']=='analysis_stage']
            if len(analyses)!=1:raise ValueError('Selected native analysis exports require one original public analysis stage')
            bundle,compared=verified_private_analysis(original,*analyses[0],pins)
            copy_evidence.append({'case':case['id'],'analysis_bundle':str(bundle),'original_native_run':case['run'],'byte_equivalent_original_inputs':compared,
                'meaning':'Analysis keeps its original private source_run and output provenance; no run identity is relabelled.'})
            figures(case,'analysis_figures','analysis-figures',bundle);figures(case,'evolved_power','power-evolved',bundle)
        if options['black_holes']['enabled']:
            if not cfg.get('black_holes',{}).get('enabled'):raise ValueError('Selected BH measurement must use an actually enabled native BH model')
            sid=add(case,'bh_history','native_bh_history',{'run_directory':case['run'],'maximum_saved_epochs':len(epochs),'memory_gib':memory},
                ['history','figures','history-result.json'],'bh_history','history-result.json')
            add(case,'bh_events','native_bh_event_figures',{'history_directory':{'stage':sid,'artifact':'history'},'hydro_history':history,
                'run_directory':case['run'],'memory_gib':memory,'pair_ids':copy.deepcopy(options['black_holes']['pair_ids']),
                'recoil_selection':copy.deepcopy(options['black_holes']['recoil_selection'])},['event-figures.json'],'bh_events','event-figures.json')
        if options['baryon_design'] is not None:
            design=options['baryon_design']
            add(case,'baryon_figures','native_baryon_figures',{'hydro_history':history,'memory_gib':memory,
                'design_path':design['path'],'design_sha256':design['sha256']},['baryon-figures.json'],'baryon_figures','baryon-figures.json')
        else:unresolved.append({'case':case['id'],'variants':['gas_phase_diagram','luminosity_history','color_history'],
            'status':'REQUIRED_INPUT_NOT_SELECTED','reason':'Complete gas-phase edges and pinned stellar-population model/filter design required; no model or photometry fabricated'})
        if options['web_design'] is not None:
            design=options['web_design']
            for product,metadata in owned:
                if product['role']!='structure':continue
                path=Path(product['reference']['path']).resolve();status=next(value for source,value in metadata if source==path)
                if status['status']!='MEASURED':
                    unresolved.append({'case':case['id'],'family':'web','status':'UNRESOLVED_TARGET','snapshot':status['snapshot']});continue
                structure=status['measurement'];source=Path(structure['path']).resolve()
                if pins.get(str(source))!=structure['sha256']:raise ValueError('Web measurement must use its original owned halo structure')
                add(case,'web_epoch_'+str(status['snapshot']),'native_web_epoch',{'run_directory':case['run'],'snapshot':status['snapshot'],
                    'halo_structure_product':str(source),'filament_skeleton':None,'design_path':design['path'],'design_sha256':design['sha256'],
                    'memory_gib':memory},['web-environment.json','figures'],'web_epoch','web-environment.json')
        if options.get('branch_design') is not None:
            add(case,'native_branches','native_branch_products',{'case':copy.deepcopy(original),'design':options['branch_design'],
                'memory_gib':memory,'maximum_metadata_files':parameters['maximum_metadata_files']},['products'],'native_branches','products/branch-result.json')
        cases.append(case)
    if parameters['continuation'] is not None:
        def bind(value):
            if isinstance(value,list):return [bind(item) for item in value]
            if isinstance(value,dict):
                if value=={'runtime_measurement_cases':True}:return copy.deepcopy(cases)
                return {key:bind(item) for key,item in value.items()}
            return value
        final=bind(copy.deepcopy(parameters['continuation']));final['depends_on']=list(dict.fromkeys([*final.get('depends_on',[]),*[row['stage_id'] for row in rows]]));rows.append(final)
    for name,sha in pins.items():
        if file_sha256(name)!=sha:raise ValueError('Native measurement preparation changed original sources')
    atomic_json(context.generation/'measurement-preparation.json',{'schema':'flagship-runtime-measurements-prepared-v1','execution_status':'RECIPES_PREPARED',
        'source_hashes':pins,'case_count':len(cases),'new_stage_count':len(rows),'private_analysis_lineage':copy_evidence,
        'cases':cases,'measurement_request':{**copy.deepcopy(parameters),'continuation':None},
        'display_continuation':copy.deepcopy(parameters['continuation']),
        'unresolved':unresolved,'science_acceptance':'NOT_ACCEPTED','measurement_execution':'NOT_EXECUTED'})
    return {'followups':rows,'measurements':{'case_count':len(cases),'new_stage_count':len(rows),'unresolved':unresolved,'science_acceptance':'NOT_ACCEPTED'}}
