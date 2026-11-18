"""Prepare native parent followups from an actual completed CLASS stage.

The existing workflow engine executes the returned stages and owns their
checkpoints. This stage writes physical requests; it does not generate ICs or
evolve particles, and never supplies guessed future file hashes.
"""
from pathlib import Path
import copy
import json
from .foundation import atomic_json,file_sha256
from .foundation import merge_source_bindings as _merge


def validate_parent_preparation(parameters):
    from .workflow import positive_number,safe_id
    required={'schema','prefix','linear_bundle','linear_stage_receipt','answers','staging','generator','execution','continuation'}
    if set(parameters)!=required or parameters['schema']!='flagship-parent-preparation-v1':raise ValueError('Complete native parent preparation request required')
    safe_id(parameters['prefix'])
    if not isinstance(parameters['answers'],dict) or parameters['answers'].get('physics')!='hydro':raise ValueError('Native MUSIC parent requires explicit hydro answers')
    staging=parameters['staging']
    if set(staging)-{'frozen_parent_noise'}!={'box_mpc_h','parent_level','seed','padding_cells','poisson_accuracy','lpt_order'}:raise ValueError('Complete fixed parent hierarchy settings required')
    if type(staging['parent_level']) is not int or not 5<=staging['parent_level']<=20 or type(staging['seed']) is not int or staging['seed']<1:raise ValueError('Exact supported parent level and seed required')
    if staging['lpt_order']!=2 or type(staging['lpt_order']) is not int:raise ValueError('Retain the selected native mixed-fluid second-order parent')
    if parameters['answers'].get('seed')!=staging['seed'] or parameters['answers'].get('particles_per_dimension')!=2**staging['parent_level'] or parameters['answers'].get('box_mpc_h')!=staging['box_mpc_h']:raise ValueError('Parent answer box, resolution and phase must match actual staging')
    if set(parameters['execution'])!={'generation_wall_hours','evolution_wall_hours','audit_memory_gib'}:raise ValueError('Explicit bounded parent execution settings required')
    for key,value in parameters['execution'].items():positive_number(value,key)
    if set(parameters['generator'])!={'binary','native_validation'}:raise ValueError('Exact native generator and admission inputs required')
    if parameters['continuation'] is not None and not isinstance(parameters['continuation'],dict):raise ValueError('Explicit continuation stage or null required')


def parent_preparation_inputs(parameters):
    from .workflow import verify_completed_receipt
    from .initial_conditions import validate_linear_bundle
    from .zoom import validate_native_music_regression,require_mixed_lpt_validation
    validate_parent_preparation(parameters);pins={};memory=parameters['execution']['audit_memory_gib']
    reference=parameters['linear_bundle'];path=Path(reference['path']).resolve()
    if set(reference)!={'path','sha256'} or file_sha256(path)!=reference['sha256']:raise ValueError('Completed actual linear bundle changed')
    receipt_path=Path(parameters['linear_stage_receipt']).resolve();receipt=verify_completed_receipt(receipt_path);root=Path(receipt['generation']).resolve()
    if receipt['kind']!='public_command' or not path.is_relative_to(root) or receipt['files_sha256'].get(path.relative_to(root).as_posix())!=reference['sha256']:raise ValueError('Parent must consume the actual linear bundle owned by its completed CLASS producer')
    _merge(pins,{str(root/name):sha for name,sha in receipt['files_sha256'].items()});_merge(pins,receipt['source_hashes']);_merge(pins,receipt['input_hashes'])
    _merge(pins,{str(receipt_path):file_sha256(receipt_path),str(receipt_path.with_suffix('.sha256')):file_sha256(receipt_path.with_suffix('.sha256'))})
    validate_linear_bundle(path,memory_gib=memory,box_mpc_h=parameters['staging']['box_mpc_h'],refinement_level=parameters['staging']['parent_level'])
    generator=parameters['generator']
    for reference in generator.values():
        if set(reference)!={'path','sha256'} or file_sha256(reference['path'])!=reference['sha256']:raise ValueError('Native parent generator/admission input changed')
        _merge(pins,{str(Path(reference['path']).resolve()):reference['sha256']})
    frozen=parameters['staging'].get('frozen_parent_noise')
    if frozen is not None:
        if set(frozen)!={'path','sha256'} or file_sha256(frozen['path'])!=frozen['sha256']:raise ValueError('Declared frozen parent phase input changed')
        _merge(pins,{str(Path(frozen['path']).resolve()):frozen['sha256']})
    proof=validate_native_music_regression(generator['binary']['path'],generator['native_validation']['path'])
    require_mixed_lpt_validation(proof,generator['binary']['path'])
    mixed=proof['mixed_fluid_2lpt']['receipt'];mixed_path=Path(mixed['path']).resolve()
    if file_sha256(mixed_path)!=mixed['sha256']:raise ValueError('Native parent mixed-fluid admission receipt changed')
    _merge(pins,{str(mixed_path):mixed['sha256'],**json.loads(mixed_path.read_text())['source_hashes']})
    for name in ('flagship_parent_preparation.py','research/checkpoints.py'):
        path=(Path(__file__).resolve().parent/name).resolve();_merge(pins,{str(path):file_sha256(path)})
    for name,sha in pins.items():
        if file_sha256(name)!=sha:raise ValueError('Native parent preparation source changed')
    return pins


def parent_preparation_runner(parameters,context):
    from .zoom import stage_zoom_inputs,verify_staged_zoom_inputs
    from core.research.capabilities import validate_native_answers
    from .workflow import safe_id
    from core.research.checkpoints import evolution_checkpoint_policy
    if parameters['execution']['audit_memory_gib']>context.resources['memory_gib']:raise MemoryError('Native parent audit exceeds launch allocation')
    pins=parent_preparation_inputs(parameters);out=context.generation;settings=parameters['staging'];prefix=parameters['prefix']
    answers=copy.deepcopy(parameters['answers']);answers['cosmology']=str(Path(parameters['linear_bundle']['path']).resolve())
    cfg=validate_native_answers(answers,context.resources)
    if cfg['simulation']['random_seed']!=settings['seed']:raise ValueError('Actual parent builder changed the selected phase seed')
    staged=stage_zoom_inputs(parameters['linear_bundle']['path'],out/'staged',box_mpc_h=settings['box_mpc_h'],
        parent_level=settings['parent_level'],refinement_level=settings['parent_level'],seeds_by_level={str(settings['parent_level']):settings['seed']},
        padding_cells=settings['padding_cells'],poisson_accuracy=settings['poisson_accuracy'],lpt_order=settings['lpt_order'],
        phase_noise_by_level=None if settings.get('frozen_parent_noise') is None else {str(settings['parent_level']):settings['frozen_parent_noise']})
    verify_staged_zoom_inputs(out/'staged');atomic_json(out/'answers.json',answers)
    assets={**pins,**{str(out/'staged'/name):sha for name,sha in staged['sealed_assets'].items()},
        str(out/'staged/staging.json'):file_sha256(out/'staged/staging.json'),str(out/'staged/staging.sha256'):file_sha256(out/'staged/staging.sha256'),
        str(out/'answers.json'):file_sha256(out/'answers.json')}
    generation=safe_id(prefix+'_generate');evolution=safe_id(prefix+'_evolve')
    generator=parameters['generator'];control=parameters['execution']
    rows=[{'stage_id':generation,'kind':'public_command','parameters':{'command':'zoom','wall_seconds':control['generation_wall_hours']*3600,
        'arguments':[{'input':str(out/'staged/parent.conf')},'{generation}/generated','--binary',{'input':generator['binary']['path']},
            '--native-validation',{'input':generator['native_validation']['path']},'--threads',str(context.resources['omp_threads']),
            '--audit-memory-gib',str(control['audit_memory_gib']),'--native-memory-gib',str(context.resources['memory_gib']),
            '--native-disk-gib',str(context.resources['disk_gib']),'--native-wall-hours',str(control['generation_wall_hours'])]},
        'input_files':[{'path':name,'sha256':sha} for name,sha in assets.items()],
        'outputs':{safe_id(prefix+'_generated_ic'):'generated'}},
        {'stage_id':evolution,'kind':'public_command','parameters':{'command':'gadget','wall_seconds':control['evolution_wall_hours']*3600,
            'arguments':['--answers',{'input':str(out/'answers.json')},'--existing-ic',{'stage':generation,'artifact':'generated/ics_gadget.hdf5'},
                '--native-generation-report',{'stage':generation,'artifact':'generated/generation.json'},'--run-dir','{generation}/run','--evolution-only']},
            'checkpoint_resume':evolution_checkpoint_policy(),'input_files':[{'path':name,'sha256':sha} for name,sha in assets.items()],
            'outputs':{safe_id(prefix+'_native_run'):'run'}}]
    bindings={'run':{'stage':evolution,'artifact':'run'},'run_receipt':{'stage':evolution,'artifact':'stage-receipt.json'},
        'ic':{'stage':generation,'artifact':'generated/ics_gadget.hdf5'},'generation_report':{'stage':generation,'artifact':'generated/generation.json'},
        'generation_receipt':{'stage':generation,'artifact':'stage-receipt.json'},
        'noise':{'stage':generation,'artifact':'generated/wnoise_%04d.bin'%settings['parent_level']}}
    def resolve(value):
        if isinstance(value,list):return [resolve(item) for item in value]
        if isinstance(value,dict):
            if set(value)=={'parent_output'}:
                if value['parent_output']=='answers_reference':
                    return {'path':{'stage':context.stage_id,'artifact':'answers.json'},
                        'sha256':{'stage':context.stage_id,'artifact':'answers.json','load':'sha256'}}
                if value['parent_output'] not in bindings:raise ValueError('Unknown actual parent output reference')
                return copy.deepcopy(bindings[value['parent_output']])
            return {key:resolve(item) for key,item in value.items()}
        return value
    if parameters['continuation'] is not None:
        continuation=resolve(parameters['continuation']);continuation['depends_on']=list(dict.fromkeys([*continuation.get('depends_on',[]),evolution]))
        rows.append(continuation)
    for name,sha in pins.items():
        if file_sha256(name)!=sha:raise ValueError('Parent preparation source changed during publication')
    atomic_json(out/'parent-preparation.json',{'schema':'flagship-parent-prepared-v1','execution_status':'RECIPES_PREPARED',
        'source_hashes':pins,'answers_sha256':file_sha256(out/'answers.json'),'linear_bundle':parameters['linear_bundle'],
        'parent_level':settings['parent_level'],'phase_seed':settings['seed'],'saved_redshifts':cfg['simulation']['snapshot_redshift_list'],
        'followup_stage_ids':[row['stage_id'] for row in rows],'science_acceptance':'NOT_ACCEPTED','native_parent_execution':'NOT_EXECUTED'})
    return {'followups':rows,'measurements':{'prepared_parent_stages':len(rows),'science_acceptance':'NOT_ACCEPTED'}}
