"""Trace actual completed parent targets and prepare bounded native zoom stages.

Target memberships and parent phase bytes come from their completed native
owners. Generation and evolution remain ordinary original workflow stages.
"""
from pathlib import Path
import copy
import io
import json
import shutil
from .foundation import atomic_json,atomic_text,file_sha256
from .foundation import merge_source_bindings as _merge


def validate_zoom_preparation(parameters):
    from .workflow import positive_number,safe_id
    required={'schema','prefix','native_run_directory','native_stage_receipt','generation_report','generation_stage_receipt',
        'parent_noise','snapshot','selection_rules','explicit_host_rows','staging','generator','execution','continuation'}
    if set(parameters)!=required or parameters['schema']!='flagship-zoom-preparation-v1':raise ValueError('Complete actual-parent zoom preparation request required')
    safe_id(parameters['prefix'])
    if type(parameters['snapshot']) is not int or parameters['snapshot']<0:raise ValueError('Actual parent snapshot index required')
    staging=parameters['staging']
    if set(staging)-{'frozen_phase_prefix'}!={'refinement_levels','seeds_by_level','aperture_r200c_factor','padding_comoving_kpc_h','padding_cells','poisson_accuracy','bounded_source_settings','source_archive_max_mib'}:raise ValueError('Complete bounded zoom hierarchy/trace settings required')
    levels=staging['refinement_levels']
    if not isinstance(levels,list) or not levels or any(type(level) is not int or not 5<=level<=20 for level in levels) or levels!=sorted(set(levels)):raise ValueError('Distinct increasing configured refinement levels required')
    positive_number(staging['aperture_r200c_factor'],'trace aperture')
    if type(staging['padding_comoving_kpc_h']) not in (int,float) or staging['padding_comoving_kpc_h']<0:raise ValueError('Explicit nonnegative physical trace padding required')
    if set(parameters['execution'])!={'generation_wall_hours','audit_memory_gib'}:raise ValueError('Explicit zoom generation wall and audit bounds required')
    for key,value in parameters['execution'].items():positive_number(value,key)
    if set(parameters['generator'])!={'binary','native_validation'}:raise ValueError('Exact current zoom generator and admission bindings required')
    if not isinstance(parameters['explicit_host_rows'],list) or any(type(row) is not int or row<0 for row in parameters['explicit_host_rows']) or len(set(parameters['explicit_host_rows']))!=len(parameters['explicit_host_rows']):raise ValueError('Distinct exact explicit native host rows required')
    if parameters['continuation'] is not None and not isinstance(parameters['continuation'],dict):raise ValueError('Explicit continuation stage or null required')


def zoom_preparation_inputs(parameters):
    from .sidm_analysis import _stage_native_bindings
    from .workflow import verify_completed_receipt
    from .analysis import load_cfg,_initial_condition_paths
    from .zoom import verify_generated_zoom,verify_staged_zoom_inputs,_configuration,validate_native_music_regression,require_mixed_lpt_validation,_music_seed_stream_ranges
    validate_zoom_preparation(parameters);memory=parameters['execution']['audit_memory_gib']
    run=Path(parameters['native_run_directory']).resolve();receipt_path=Path(parameters['native_stage_receipt']).resolve()
    native=verify_completed_receipt(receipt_path)
    if native['kind']!='public_command':raise ValueError('Actual completed public native parent evolution required')
    pins=_stage_native_bindings({'native_run_directory':str(run),'native_stage_receipt':str(receipt_path),'request':{'memory_gib':memory}})
    manifest_path=run/'manifest.json'
    if pins.get(str(manifest_path))!=file_sha256(manifest_path) or json.loads(manifest_path.read_text()).get('status')!='SIMULATION_COMPLETE':raise ValueError('Actual complete owned parent evolution manifest required')
    gen_path=Path(parameters['generation_report']).resolve();owner_path=Path(parameters['generation_stage_receipt']).resolve();owner=verify_completed_receipt(owner_path);root=Path(owner['generation']).resolve()
    if owner['kind']!='public_command' or not gen_path.is_relative_to(root) or owner['files_sha256'].get(gen_path.relative_to(root).as_posix())!=file_sha256(gen_path):raise ValueError('Actual parent IC generation must be owned by its original completed producer')
    _merge(pins,{str(root/name):sha for name,sha in owner['files_sha256'].items()});_merge(pins,owner['source_hashes']);_merge(pins,owner['input_hashes']);_merge(pins,{str(owner_path):file_sha256(owner_path)})
    generation=verify_generated_zoom(gen_path)
    if generation.get('publication_contract')!='native-ic-generation-v1':raise ValueError('Original contracted native parent generation required')
    config=load_cfg(run);cfg=_configuration(gen_path.parent/'music.conf');low=cfg.getint('setup','levelmin')
    if low!=cfg.getint('setup','levelmax') or config['simulation']['particles_per_dimension']!=2**low:raise ValueError('Uniform actual parent hierarchy required before zoom tracing')
    if Path(config['simulation']['native_generation_report']).resolve()!=gen_path:raise ValueError('Parent evolution used a different original IC generation')
    initial=[Path(p).resolve() for p in _initial_condition_paths(run)]
    if len(initial)!=1 or file_sha256(initial[0])!=generation['artifacts']['ics_gadget.hdf5']:raise ValueError('Parent evolution initial conditions differ from the actual generated parent')
    noise=Path(parameters['parent_noise']).resolve()
    if noise!=gen_path.parent/('wnoise_%04d.bin'%low) or generation['artifacts'].get(noise.name)!=file_sha256(noise):raise ValueError('Frozen parent noise must be an original owned generation cache')
    seeds=parameters['staging']['seeds_by_level'];high=max(parameters['staging']['refinement_levels'])
    original_staging=verify_staged_zoom_inputs(Path(generation['source_configuration']['path']).parent)
    if set(seeds)!={str(level) for level in range(low,high+1)} or seeds[str(low)]!=config['simulation']['random_seed'] or original_staging['seeds_by_level'][str(low)]!=seeds[str(low)]:raise ValueError('Zoom hierarchy changes actual parent seed or omits intermediate levels')
    if min(parameters['staging']['refinement_levels'])<=low:raise ValueError('Configured zoom levels must refine the actual parent')
    _music_seed_stream_ranges(seeds)
    frozen=parameters['staging'].get('frozen_phase_prefix',{})
    if frozen:
        levels=sorted(map(int,frozen))
        if levels!=list(range(low,max(levels)+1)) or max(levels)>high:raise ValueError('Declared frozen phase prefix must be contiguous and within configured refinement')
        for reference in frozen.values():
            if set(reference)!={'path','sha256'} or file_sha256(reference['path'])!=reference['sha256']:raise ValueError('Declared retained refinement phase input changed')
            _merge(pins,{str(Path(reference['path']).resolve()):reference['sha256']})
    descriptor=Path(config['spectrum']['file']).resolve()
    if pins.get(str(descriptor))!=file_sha256(descriptor):raise ValueError('Parent linear bundle is outside its actual native lineage')
    for reference in parameters['generator'].values():
        if set(reference)!={'path','sha256'} or file_sha256(reference['path'])!=reference['sha256']:raise ValueError('Zoom generator/admission binding changed')
        _merge(pins,{str(Path(reference['path']).resolve()):reference['sha256']})
    proof=validate_native_music_regression(parameters['generator']['binary']['path'],parameters['generator']['native_validation']['path']);require_mixed_lpt_validation(proof,parameters['generator']['binary']['path'])
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    for name,sha in pins.items():
        if file_sha256(name)!=sha:raise ValueError('Parent lineage changed during zoom intake')
    return pins,config,low,initial[0]


def zoom_preparation_runner(parameters,context):
    from core.research.zooms import select_catalogue_targets
    from .zoom import trace_catalogue_host,stage_zoom_inputs,verify_staged_zoom_inputs,_configuration,apply_traced_region,plan_native_music
    from core.music.phases import encode_parent_noise
    from .workflow import safe_id
    from core.resources import periodic_image_resource_allowance
    if parameters['execution']['audit_memory_gib']>context.resources['memory_gib']:raise MemoryError('Zoom trace/phase audit exceeds workflow allocation')
    pins,config,low,initial=zoom_preparation_inputs(parameters);out=context.generation;settings=parameters['staging'];memory=parameters['execution']['audit_memory_gib']
    selected=select_catalogue_targets(parameters['native_run_directory'],parameters['snapshot'],parameters['selection_rules'],parameters['explicit_host_rows'],memory_gib=memory)
    _merge(pins,selected['source_hashes'])
    count=len(selected['targets'])*len(settings['refinement_levels'])
    if count+(parameters['continuation'] is not None)>context.resources['maximum_stage_count']:raise ValueError('Configured target/refinement matrix exceeds workflow stage allocation')
    noise=Path(parameters['parent_noise']).resolve()
    prefix=encode_parent_noise(noise,out/'parent-prefix.grafic',level=low,cache_sha256=file_sha256(noise),memory_gib=memory,disk_gib=context.resources['disk_gib'])
    frozen=copy.deepcopy(settings.get('frozen_phase_prefix',{}))
    if frozen:
        from core.music.phases import read_frozen_grafic
        import numpy as np
        expected=read_frozen_grafic(frozen[str(low)]['path'],2**low,byte_order='<',positive_sign=True,memory_gib=memory)
        actual=read_frozen_grafic(prefix['output']['path'],2**low,byte_order='<',positive_sign=True,memory_gib=memory)
        if not np.array_equal(expected.view('=u8'),actual.view('=u8')):raise ValueError('Actual parent cache differs from the declared established frozen phase prefix')
    frozen[str(low)]=prefix['output'];rows=[];cases=[];total_disk=0
    run=Path(parameters['native_run_directory']).resolve();descriptor=Path(config['spectrum']['file']).resolve()
    for target in selected['targets']:
        host=target['host_row'];region_dir=out/('host%d-region'%host)
        trace_catalogue_host(run,parameters['snapshot'],host,initial,run/'manifest.json',region_dir,
            aperture_r200c_factor=settings['aperture_r200c_factor'],padding_comoving_kpc_h=settings['padding_comoving_kpc_h'],memory_gib=memory)
        region=region_dir/'region.json'
        for level in settings['refinement_levels']:
            identifier=safe_id(parameters['prefix']+'_host%d_level%d'%(host,level));stage_id=safe_id(identifier+'_generate');directory=out/identifier
            staged=stage_zoom_inputs(descriptor,directory,box_mpc_h=config['simulation']['box_mpc_h'],parent_level=low,refinement_level=level,
                seeds_by_level={str(i):settings['seeds_by_level'][str(i)] for i in range(low,level+1)},
                padding_cells=settings['padding_cells'],poisson_accuracy=settings['poisson_accuracy'],lpt_order=2,
                phase_noise_by_level={key:value for key,value in frozen.items() if int(key)<=level},bounded_source_settings=settings['bounded_source_settings'],source_archive_max_mib=settings['source_archive_max_mib'])
            verify_staged_zoom_inputs(directory);configuration=directory/'zoom_template.conf'
            planner=_configuration(configuration);apply_traced_region(planner,region);stream=io.StringIO();planner.write(stream)
            planner_input=out/(identifier+'.planner.conf');atomic_text(planner_input,stream.getvalue())
            report=plan_native_music(planner_input,parameters['generator']['binary']['path'],out/(identifier+'.planning'),
                memory_gib=context.resources['memory_gib'],disk_gib=context.resources['disk_gib'],threads=context.resources['omp_threads'])
            if report['status']!='PASS':raise MemoryError('Actual traced zoom resource plan failed: '+identifier)
            bound=report['particle_count_upper_bound'];images=periodic_image_resource_allowance(bound)
            peak=bound*context.resources.get('evolution_bytes_per_particle',2048)+images+512*bound
            saved=context.resources.get('saved_epochs',len(config['simulation']['snapshot_redshift_list']));copies=context.resources.get('checkpoint_copies',2)
            storage=report['estimated_disk_bytes']+(bound*context.resources.get('snapshot_bytes_per_particle',256)+images)*(saved+copies)+64*bound*saved
            if peak>context.resources['memory_gib']*1024**3:raise MemoryError('Traced zoom evolution geometry bound exceeds memory allocation')
            total_disk+=storage
            assets={**pins,**{str(directory/name):sha for name,sha in staged['sealed_assets'].items()},
                str(directory/'staging.json'):file_sha256(directory/'staging.json'),str(directory/'staging.sha256'):file_sha256(directory/'staging.sha256'),
                str(region):file_sha256(region),str(region_dir/'selected_particles.hdf5'):file_sha256(region_dir/'selected_particles.hdf5'),
                prefix['output']['path']:prefix['output']['sha256']}
            generator=parameters['generator'];wall=parameters['execution']['generation_wall_hours']
            row={'stage_id':stage_id,'kind':'public_command','parameters':{'command':'zoom','wall_seconds':wall*3600,
                'arguments':[{'input':str(configuration)},'{generation}/generated','--binary',{'input':generator['binary']['path']},
                    '--native-validation',{'input':generator['native_validation']['path']},'--traced-region',{'input':str(region)},
                    '--parent-ic',{'input':str(initial)},'--parent-noise',{'input':str(noise)},'--threads',str(context.resources['omp_threads']),
                    '--audit-memory-gib',str(memory),'--native-memory-gib',str(context.resources['memory_gib']),
                    '--native-disk-gib',str(context.resources['disk_gib']),'--native-wall-hours',str(wall)]},
                'input_files':[{'path':name,'sha256':sha} for name,sha in assets.items()],'outputs':{safe_id(identifier+'_ic'):'generated'}}
            rows.append(row);cases.append({'id':identifier,'host_row':host,'level':level,'generation_stage':stage_id,
                'generation':{'path':{'stage':stage_id,'artifact':'generated/generation.json'},'sha256':{'stage':stage_id,'artifact':'generated/generation.json','load':'sha256'}},
                'generation_receipt':{'stage':stage_id,'artifact':'stage-receipt.json'},
                'traced_region':{'path':str(region),'sha256':file_sha256(region)},'configuration':{'path':str(configuration),'sha256':file_sha256(configuration)},
                'resource_estimate':{'particle_count_upper_bound':bound,'evolution_memory_bytes':peak,'retained_storage_bytes':storage,'scope':'Geometry estimate; actual generated/evolved census and RSS remain runtime requirements'}})
    prepared_bytes=sum(path.stat().st_size for path in out.rglob('*') if path.is_file() and '.native-run-locks' not in path.relative_to(out).parts)
    free=shutil.disk_usage(out).free-context.resources.get('disk_margin_gib',0)*1024**3
    if total_disk+prepared_bytes>context.resources['disk_gib']*1024**3 or total_disk>free:raise MemoryError('Aggregate configured zoom matrix exceeds retained output disk budget')
    def resolve(value):
        if isinstance(value,list):return [resolve(item) for item in value]
        if isinstance(value,dict):
            if value=={'zoom_cases':True}:return copy.deepcopy(cases)
            if value=={'zoom_preparation_reference':True}:
                return {'path':{'stage':context.stage_id,'artifact':'zoom-preparation.json'},
                    'sha256':{'stage':context.stage_id,'artifact':'zoom-preparation.json','load':'sha256'}}
            if value=={'zoom_preparation_source':True}:return str(out/'zoom-preparation.json')
            if value=={'zoom_preparation_receipt':True}:return {'stage':context.stage_id,'artifact':'stage-receipt.json'}
            return {key:resolve(item) for key,item in value.items()}
        return value
    if parameters['continuation'] is not None:
        following=resolve(parameters['continuation']);following['depends_on']=list(dict.fromkeys([*following.get('depends_on',[]),*[row['stage_id'] for row in rows]]));rows.append(following)
    for name,sha in pins.items():
        if file_sha256(name)!=sha:raise ValueError('Parent source changed during zoom preparation')
    atomic_json(out/'zoom-preparation.json',{'schema':'flagship-zoom-prepared-v1','execution_status':'RECIPES_PREPARED','selection':selected,
        'parent_run':str(run),'parent_level':low,'frozen_parent_prefix':prefix,'cases':cases,'source_hashes':pins,
        'phase_policy':{'declared_frozen_levels':sorted(map(int,frozen)),'subsequent_levels':'Original seeded local refinement after the declared frozen prefix'},
        'generation_stage_definitions':rows[:count],'aggregate_storage_estimate_bytes':total_disk+prepared_bytes,'preparation_bytes':prepared_bytes,
        'native_zoom_generation':'NOT_EXECUTED','science_acceptance':'NOT_ACCEPTED'})
    return {'followups':rows,'measurements':{'configured_native_targets':len(selected['targets']),'configured_zoom_cases':count,'science_acceptance':'NOT_ACCEPTED'}}
