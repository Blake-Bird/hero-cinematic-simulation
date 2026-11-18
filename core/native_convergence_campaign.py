"""Connect prepared native cases to the original measured convergence engine.

Preparation never evolves a case or promotes a resource estimate to science
acceptance. Actual materialized native settings, source phases and measured
support remain separate gates.
"""
from pathlib import Path
import copy
import json
import shutil
from .foundation import atomic_json,file_sha256,canonical_digest
from .foundation import merge_source_bindings as _merge

VARIANTS={'mass_resolution':'mass_resolution','softening':'softening','timestep':'timestep',
    'output_cadence':'output_cadence','start_epoch':'start_redshift','finder':'finder_tree_variation'}


def _read(reference,pins,memory_gib):
    if set(reference)!={'path','sha256'}:raise ValueError('Exact native convergence input path/hash required')
    path=Path(reference['path']).resolve()
    if path.stat().st_size*32>memory_gib*1024**3/8:raise MemoryError('Native convergence metadata exceeds allocation')
    if file_sha256(path)!=reference['sha256']:raise ValueError('Native convergence input changed')
    _merge(pins,{str(path):reference['sha256']});return json.loads(path.read_text())


def _prefix(value,mapping):
    if isinstance(value,list):return [_prefix(item,mapping) for item in value]
    if isinstance(value,dict):
        result={key:_prefix(item,mapping) for key,item in value.items()}
        if 'stage' in result and 'artifact' in result:result['stage']=mapping[result['stage']]
        return result
    return value


def _at(value,path):
    for key in path:value=value[key]
    return value


def _without(value,paths):
    value=copy.deepcopy(value)
    for path in paths:
        node=value
        for key in path[:-1]:node=node[key]
        if path[-1] not in node:raise ValueError('Convergence cannot erase an absent native parameter')
        del node[path[-1]]
    return value


def prepare_native_convergence(request,destination):
    """Compile actual prepared cases; reject merely relabelled control values."""
    from .workflow import default_registry,compile_requested_product_dag,safe_id
    from core.research.measurements import attach_native_measurement_stage
    from core.research.capabilities import validate_native_answers
    required={'schema','control','runs','resources','answer_bindings','measurement','comparison','bin_coordinate'}
    if set(request)!=required or request['schema']!='native-convergence-family-request-v1' or request['control'] not in VARIANTS:raise ValueError('Complete native convergence family request required')
    control=request['control'];resources=request['resources'];memory=resources['memory_gib'];out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Preserve previous native convergence preparation')
    runs=request['runs']
    if not isinstance(runs,list) or len(runs)<2:raise ValueError('At least two actual prepared native cases required')
    if len({canonical_digest(row['control_value']) for row in runs})!=len(runs):raise ValueError('Convergence requires distinct actual numerical control values')
    bindings=request['answer_bindings']
    if not isinstance(bindings,list) or not bindings:raise ValueError('Actual changed native answer fields required')
    if control=='finder' and request['measurement']['request']['mode']!='numeric_product':raise ValueError('Finder/tree convergence requires a measured catalogue/tree observable, not a finder-independent particle profile')
    paths=[row['path'] for row in bindings]
    if len({tuple(path) for path in paths})!=len(paths) or any(not path or any(not isinstance(key,str) for key in path) for path in paths):raise ValueError('Distinct actual native answer paths required')
    if any(key in {'mpi_ranks','omp_threads','memory_limit_gb','build_jobs','seed'} for path in paths for key in path):raise ValueError('Numerical convergence cannot relabel phases or alter resource allocations')
    # One family changes exactly its known native numerical quantity. Additional
    # fields must be explicitly derived components of that same control.
    allowed={'mass_resolution':{('particles_per_dimension',)},'softening':{('softening_kpc_h',),('softening_max_physical_kpc_h',)},
        'timestep':{('max_size_timestep_ln_a',)},'output_cadence':{('schedule',)},'start_epoch':{('cosmology',)},'finder':{('halo_finder',),('native_mergertree',),('fof_linking_length',)}}
    if not {tuple(path) for path in paths}<=allowed[control]:raise ValueError('Native answer bindings change unrelated physics or unsupported numerical controls')
    pins={};records=[];rows=[];seen=set();baseline=None;baseline_generation=None;baseline_measurement=None;budget_total=0
    baseline_generator_settings=None;baseline_linear=None
    out.mkdir(parents=True)
    for ordinal,entry in enumerate(runs):
        if set(entry)!={'run_id','campaign','case_id','control_value'}:raise ValueError('Complete actual case/control selection required')
        rid=safe_id(entry['run_id'])
        if rid in seen:raise ValueError('Repeated native convergence run identity')
        seen.add(rid);campaign=_read(entry['campaign'],pins,memory)
        if campaign.get('schema')!='flagship-native-campaign-v1' or campaign.get('status')!='RECIPES_PREPARED':raise ValueError('Original prepared native generation/evolution campaign required')
        for name,sha in campaign['source_hashes'].items():
            if file_sha256(name)!=sha:raise ValueError('Prepared native campaign source changed; prepare it again')
        _merge(pins,campaign['source_hashes'])
        source=_read(campaign['workflow'],pins,memory);case=next((item for item in campaign['cases'] if item['id']==entry['case_id']),None)
        if case is None:raise ValueError('Selected native convergence case absent from prepared campaign')
        answers=_read(case['answers'],pins,memory);cfg=validate_native_answers(answers,resources)
        expected=[entry['control_value']*row.get('scale',1.) if row.get('scale',1.)!=1 else entry['control_value'] for row in bindings]
        if control=='output_cadence':expected=[','.join(format(float(z),'.17g') for z in entry['control_value'])]
        actual=[_at(answers,path) for path in paths]
        if control=='output_cadence':
            if cfg['simulation']['snapshot_redshift_list']!=entry['control_value']:raise ValueError('Actual native schedule differs from declared cadence control')
        elif control=='start_epoch':
            descriptor=Path(answers['cosmology']).resolve();bundle=json.loads(descriptor.read_text());_merge(pins,{str(descriptor):file_sha256(descriptor)})
            if bundle['parameters']['z_start']!=entry['control_value']:raise ValueError('Start-redshift control lacks its actual regenerated linear bundle')
            linear=copy.deepcopy(bundle['parameters']);del linear['z_start']
            if baseline_linear is None:baseline_linear=linear
            elif linear!=baseline_linear:raise ValueError('Start-redshift convergence changes unrelated linear physics')
        elif actual!=expected:raise ValueError('Prepared native settings do not apply declared convergence control')
        normalized=_without(answers,[['name'],*paths])
        if baseline is None:baseline=normalized
        elif normalized!=baseline:raise ValueError('Native convergence changes unrelated settings, phases or physical model')
        graph={row['stage_id']:row for row in source['stages']};closure=set()
        def visit(sid):
            if sid in closure:return
            closure.add(sid);row=graph[sid]
            from .workflow import _stage_references
            for dependency in set(row.get('depends_on',[]))|set(_stage_references(row['parameters'])):visit(dependency)
        visit(case['evolution_stage'])
        branch={**copy.deepcopy(source),'stages':[copy.deepcopy(row) for row in source['stages'] if row['stage_id'] in closure],
            'requested_products':[next(iter(graph[case['evolution_stage']]['outputs']))]}
        settings=copy.deepcopy(request['measurement']);settings.update(native_stage_id=case['evolution_stage'],measurement_stage_id='measurement',product_name='convergence_measurement')
        if settings['request']['comparison_epoch']['coordinate']!='actual_scale_factor':raise ValueError('Cosmological native preparation requires an explicit comparison scale factor')
        settings['request']['snapshot']=next((index for index,z in enumerate(cfg['simulation']['snapshot_redshift_list']) if abs(1/(1+z)-settings['request']['comparison_epoch']['value'])<=settings['request']['comparison_epoch']['absolute_tolerance']),None)
        if settings['request']['snapshot'] is None:raise ValueError('Convergence observable epoch absent from actual native saved schedule')
        if control=='finder' and settings.get('catalogue_profile') is not None:
            producer='catalogue_convergence_profile'
            branch['stages'].append({'stage_id':producer,'kind':'native_convergence_catalogue',
                'parameters':{'request':copy.deepcopy(settings['request']),'selection':settings.pop('catalogue_profile'),
                    'native_run_directory':{'stage':case['evolution_stage'],'artifact':settings['native_run_artifact']},
                    'native_stage_receipt':{'stage':case['evolution_stage'],'artifact':'stage-receipt.json'},
                    'paired_reference':'{reference_measurement}'},'outputs':{'catalogue_convergence_profile':'catalogue-profile.json'}})
            settings['numeric_product']={'stage':producer,'artifact':'catalogue-profile.json'}
        metadata={'schema':'native-convergence-variant-metadata-v1','run_id':rid,'specification':{},'changed_controls':[] if ordinal==0 else [control],
            'control_value':entry['control_value'],'planned_control_dimensions':[control],'baseline_run_id':runs[0]['run_id']}
        # The measured-native adapter consumes original run-spec control paths.
        from core.research.convergence import CONTROLS
        specification={'seed':answers['seed'],'dark_matter':{'interactions':{'kind':'elastic_sidm' if cfg.get('sidm',{}).get('enabled') else 'collisionless'}}}
        if cfg.get('sidm',{}).get('enabled'):specification['dark_matter']['interactions']['seed']=cfg['sidm']['seed']
        node=specification
        for key in CONTROLS[control][:-1]:node=node.setdefault(key,{})
        node[CONTROLS[control][-1]]=entry['control_value']
        spec_path=out/(rid+'.spec.json');atomic_json(spec_path,specification)
        metadata.update(specification=specification,specification_path=str(spec_path),specification_sha256=file_sha256(spec_path),
            common_signature=canonical_digest({'native_baseline':baseline,'control':control}),common_model_family_hash=canonical_digest(baseline))
        meta_path=out/(rid+'.metadata.json');atomic_json(meta_path,metadata)
        branch=attach_native_measurement_stage(branch,settings)
        mapping={row['stage_id']:safe_id(rid+'__'+row['stage_id']) for row in branch['stages']}
        from .flagship_campaigns import _matched_generation_parameters
        generator_settings=_matched_generation_parameters(graph[case['generation_stage']],pins)
        reuse=ordinal>0 and control not in {'mass_resolution','start_epoch'}
        if ordinal==0:baseline_generator_settings=generator_settings
        elif reuse and generator_settings!=baseline_generator_settings:raise ValueError('Non-IC convergence changes actual generation settings')
        if reuse:mapping[case['generation_stage']]=baseline_generation
        generation=mapping[case['generation_stage']]
        gate=safe_id(rid+'__phase_pair')
        for row in branch['stages']:
            if reuse and row['stage_id']==case['generation_stage']:continue
            row=_prefix(row,mapping);row['stage_id']=mapping[row['stage_id']];row['depends_on']=[mapping[sid] for sid in row.get('depends_on',[])]
            row['outputs']={safe_id(rid+'__'+name):artifact for name,artifact in row['outputs'].items()}
            if row['stage_id']==mapping[case['evolution_stage']]:row['depends_on'].append(gate)
            def replace(value):
                if isinstance(value,dict):return {key:replace(item) for key,item in value.items()}
                if isinstance(value,list):return [replace(item) for item in value]
                if value=='{variant_metadata}':return str(meta_path)
                if value=='{reference_measurement}':return '' if ordinal==0 else {'stage':baseline_measurement,'artifact':'convergence-measurement.json'}
                return value
            row=replace(row);rows.append(row)
        measurement=mapping['measurement']
        if ordinal==0:baseline_measurement=measurement;baseline_generation=generation
        rows.append({'stage_id':gate,'kind':'native_convergence_phase_pair','parameters':{
            'control':control,'memory_gib':memory,
            'baseline_generation':{'stage':baseline_generation,'artifact':'generated/generation.json'},
            'current_generation':{'stage':generation,'artifact':'generated/generation.json'},
            'baseline_receipt':{'stage':baseline_generation,'artifact':'stage-receipt.json'},
            'current_receipt':{'stage':generation,'artifact':'stage-receipt.json'}},'outputs':{safe_id(rid+'__phase_pair'):'phase-pair.json'}})
        records.append({'run_id':rid,'case_id':case['id'],'control_value':entry['control_value'],'measurement_stage':measurement,'generation_stage':generation,
            'metadata':{'path':str(meta_path),'sha256':file_sha256(meta_path)},'actual_answers':case['answers'],'actual_saved_redshifts':cfg['simulation']['snapshot_redshift_list']})
        budget=_read(campaign['resource_preflight'],pins,memory);case_budget=next(item for item in budget['cases'] if item['id']==case['id'])
        _merge(pins,budget.get('source_hashes',{}))
        if budget['status']!='ESTIMATE_PASS':raise MemoryError('Native convergence case resource plan did not pass')
        if case_budget['peak_memory_estimate_bytes']>memory*1024**3:raise MemoryError('Native convergence case exceeds current memory allocation')
        for key in ('mpi_ranks','omp_threads'):
            if budget['resources'][key]!=resources[key]:raise ValueError('Native convergence allocation differs from original resource planning')
        budget_total+=case_budget['storage_estimate_bytes'] if 'storage_estimate_bytes' in case_budget else max(budget['campaign_storage_estimate_bytes'],0)
    if budget_total>resources['disk_gib']*1024**3*.8:raise MemoryError('Aggregate retained convergence runs exceed disk allocation')
    if budget_total>shutil.disk_usage(out).free-resources.get('disk_margin_gib',0)*1024**3:raise MemoryError('Aggregate retained convergence runs exceed actual free disk')
    if len(rows)+1>resources['maximum_stage_count']:raise ValueError('Convergence family exceeds explicit stage allocation')
    comparison=copy.deepcopy(request['comparison']);comparison.update(control=control,reference_run=records[0]['run_id'])
    references=[{'path':{'stage':record['measurement_stage'],'artifact':'convergence-measurement.json'},'sha256':{'stage':record['measurement_stage'],'artifact':'convergence-measurement.json','load':'sha256'}} for record in records]
    receipts=[{'stage':record['measurement_stage'],'artifact':'stage-receipt.json'} for record in records]
    rows.append({'stage_id':'figures','kind':'native_convergence_figures','parameters':{'request':{'comparison':comparison,'measurements':references,'bin_coordinate':request['bin_coordinate']},'measurement_receipts':receipts,'memory_gib':memory},'outputs':{'convergence_figures':'figures'}})
    spec={'schema':'research-workflow-v1','resources':resources,'physics_spec':{},'capabilities':[],
        'followup_limits':{'maximum_children':0,'maximum_depth':0},'requested_products':['convergence_figures'],'stages':rows}
    registry=default_registry();plan=compile_requested_product_dag(spec,registry,request_base=out)
    for row in plan['stages']:_merge(pins,registry.source_hashes(row['kind']))
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    for path,sha in pins.items():
        if file_sha256(path)!=sha:raise ValueError('Native convergence sources changed during compilation')
    atomic_json(out/'workflow.json',spec)
    result={'schema':'native-convergence-family-prepared-v1','execution_status':'RECIPES_PREPARED','control':control,'variant':VARIANTS[control],'runs':records,
        'workflow':{'path':str(out/'workflow.json'),'sha256':file_sha256(out/'workflow.json')},'source_hashes':pins,'stage_count':len(rows),'aggregate_storage_estimate_bytes':budget_total,
        'native_execution':'NOT_EXECUTED','science_acceptance':'NOT_ACCEPTED',
        'remaining_requirements':['Actual cross-run native phase/IC and target matching must pass before comparisons; source preparation is not acceptance.']}
    atomic_json(out/'manifest.json',result);return result
