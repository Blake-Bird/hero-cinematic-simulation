"""Prepare native response campaigns and bind their actual generated inputs.

Public IC receipts retain the selected monofonIC transfer/phase conventions.
Training and heldout recipes use disjoint caller-supplied phase/operator seeds;
this design alone does not demonstrate empirical independence or calibration.
Paired stellar responses preserve actual clocks, periodic images, cohort
identities and intervention lineage. Preparation writes recipes, not simulations.
"""
import copy
import json
import math
import numpy as np
from pathlib import Path
from .foundation import atomic_json, canonical_digest, file_sha256, merge_source_bindings as _merge


# Public initial conditions and joint response preparation

def prepare_native_response_branch(request,destination):
    """Prepare IC generation, target measurements and heldout response recipes.

    The generated command sequence waits for real IC receipts before preparing
    evolution. A recipe or distinct seed is not accepted calibration evidence.
    """
    from .sidm_campaign import _no_placeholders
    from core.research.capabilities import validate_native_answers
    required={'schema','base_answers','resources','realizations','wall_seconds_per_generation','wall_seconds_per_native_run',
        'target_controls','training','calibration','parameter_bindings','heldout_parameter_offsets','degeneracy_settings'}
    if set(request)!=required or request['schema']!='native-joint-response-branch-v1':raise ValueError('Complete independent native response branch request required')
    _no_placeholders(request);out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Preserve prior native response branch preparation')
    memory=request['resources']['memory_gib'];reference=request['target_controls'];settings=response_target_controls(reference['path'],reference['sha256'],memory)
    base=request['base_answers'];answers=Path(base['path']).resolve()
    if answers.stat().st_size*24>memory*1024**3/8:raise MemoryError('Native response base answers exceed allocation')
    if file_sha256(answers)!=base['sha256']:raise ValueError('Native response base answer settings changed')
    cfg=validate_native_answers(json.loads(answers.read_text()),request['resources'])
    if cfg.get('sidm',{}).get('enabled') is not True or cfg['simulation']['physics']!='hydro':raise ValueError('Joint SIDM/baryon response requires actual enabled SIDM and hydro physics')
    redshifts=cfg['simulation']['snapshot_redshift_list']
    if settings['history']['snapshots']!=list(range(len(redshifts))):raise ValueError('Response target storage selectors must cover the complete actual configured saved schedule')
    pins={str(answers):base['sha256'],str(Path(reference['path']).resolve()):reference['sha256']}
    out.mkdir(parents=True)
    generation=prepare_joint_ic_generation({'schema':'native-joint-ic-generation-request-v1',**{key:copy.deepcopy(request[key]) for key in
        ('base_answers','resources','realizations','wall_seconds_per_generation')}},out/'ic-generation')
    _merge(pins,generation['source_hashes'])
    value=lambda key:{'campaign_value':key}
    design={'schema':'native-joint-response-design-v1',**{key:copy.deepcopy(request[key]) for key in
        ('base_answers','resources','wall_seconds_per_native_run','training','calibration','parameter_bindings','heldout_parameter_offsets','degeneracy_settings')},
        'snapshots':settings['history']['snapshots'],'followup_limits':{'maximum_children':0,'maximum_depth':0},
        'measurement_stages':[{'stage_id':'target','kind':'native_response_target','parameters':{
            **{key:value(key) for key in ('native_run_directory','native_stage_receipt','baseline_run_directory','baseline_stage_receipt')},
            'template_path':str(Path(reference['path']).resolve()),'template_sha256':reference['sha256'],'memory_gib':memory},
            'input_files':[copy.deepcopy(reference)],'outputs':{'target':'target-measurement.json'}}],
        'forward_vector':{'schema':'native-forward-observable-vector-request-v1','maximum_observables':len(settings['observables']),
            'observables':[{'name':row['name'],'product':{'path':{'stage':'target','artifact':'target-measurement.json'},
                'sha256':{'stage':'target','artifact':'target-measurement.json','load':'sha256'}},
                'array':f'observable_{i:03d}','support_array':f'supported_{i:03d}','indices':[0],'units':row['units']} for i,row in enumerate(settings['observables'])]}}
    atomic_json(out/'response-design.json',design)
    # Output paths are derived now; IC file hashes are derived only after the
    # original generation workflow completes. The second command enforces that.
    root=Path(__file__).resolve().parents[1];execution=out/'ic-execution';response=out/'response'
    commands=[['python',str(root/'commands.py'),'workflow','--campaign',generation['workflow']['path'],'--output',str(execution)],
        ['python',str(root/'commands.py'),'native-prepare','joint-response',str(out/'response-design.json'),'--ic-manifest',str(out/'ic-generation/manifest.json'),
            '--ic-execution',str(execution),'--output',str(response)],
        ['python',str(root/'commands.py'),'workflow','--campaign',str(response/'workflow.json'),'--output',str(out/'response-execution')]]
    _merge(pins,{str(root/'core/response_campaign.py'):file_sha256(__file__),str(out/'response-design.json'):file_sha256(out/'response-design.json')})
    for name in ('commands.py','core/campaigns.py'):
        _merge(pins,{str(root/name):file_sha256(root/name)})
    for p,h in pins.items():
        if file_sha256(p)!=h:raise ValueError('Native response branch changed during preparation')
    result={'schema':'native-joint-response-branch-prepared-v1','execution_status':'RECIPES_PREPARED','variant':'sidm_baryonic_degeneracy',
        'ic_generation':generation['workflow'],'response_design':{'path':str(out/'response-design.json'),'sha256':file_sha256(out/'response-design.json')},
        'commands':commands,'source_hashes':pins,'native_execution':'NOT_EXECUTED','science_acceptance':'NOT_ACCEPTED',
        'scope':'Independent native training/heldout ICs -> paired physical parameter evolution -> baseline cohort recovery -> original native history -> original finite differences and heldout calibration -> original degeneracy and PNG/SVG/PDF/numeric export. Unsupported native targets or calibration stop with original evidence; no synthetic calibration substituted.'}
    atomic_json(out/'manifest.json',result);return result


def public_ic_inputs(parameters):
    from core.workflow import verify_completed_receipt
    from core.analysis import load_cfg
    from core.evolution import initial_condition_files
    memory=parameters['memory_gib']
    if type(memory) not in (int,float) or not math.isfinite(memory) or memory<=0:raise ValueError('Positive native public IC intake allocation required')
    receipt=Path(parameters['source_receipt']).resolve();run=Path(parameters['source_run']).resolve()
    if receipt.stat().st_size*24>memory*1024**3/8:raise MemoryError('Native IC-only stage receipt exceeds allocation')
    owned=verify_completed_receipt(receipt);root=Path(owned['generation']).resolve()
    if owned['kind']!='public_command' or not run.is_relative_to(root):raise ValueError('Original completed public IC-only generation required')
    pins={str(root/p):h for p,h in owned['files_sha256'].items()};_merge(pins,owned['source_hashes']);_merge(pins,owned['input_hashes']);_merge(pins,{str(receipt):file_sha256(receipt)})
    manifest=run/'manifest.json'
    if manifest.stat().st_size*24>memory*1024**3/8:raise MemoryError('Native IC-only manifest exceeds allocation')
    if pins.get(str(manifest))!=file_sha256(manifest):raise ValueError('Native IC-only manifest is outside its original generation')
    record=json.loads(manifest.read_text());cfg=load_cfg(run)
    if (record.get('status')!='IC_STAGE_COMPLETE' or record.get('initial_condition_stage',{}).get('status')!='COMPLETE'
            or record['initial_condition_stage'].get('validated') is not True or record['initial_condition_stage'].get('generated_by_run') is not True):
        raise ValueError('Actual generated and validated public IC-only result required; evolution or dry-run is not IC generation evidence')
    if record.get('initial_ic_power_spectrum',{}).get('status')!='PASS':raise ValueError('Original IC power-spectrum gate did not pass')
    if type(parameters['phase_seed']) is not int or cfg['simulation']['random_seed']!=parameters['phase_seed']:raise ValueError('Native response phase seed differs from actual IC generation')
    if cfg['simulation']['physics']!='hydro' or cfg['simulation'].get('ic_engine_hydro')!='monofonic':
        raise ValueError('This public-IC adapter requires original validated uniform hydro monofonIC generation; MUSIC retains its separate numerical proof')
    primary=Path(record['initial_condition_stage']['path']).resolve()
    files=initial_condition_files(primary)
    files = [Path(path).resolve() for path in files]
    if not files or primary not in files:raise ValueError('Native public IC primary shard is absent from its complete set')
    bindings=[]
    for path in files:
        checksum = file_sha256(path)
        if not path.is_relative_to(root) or pins.get(str(path)) != checksum:
            raise ValueError('Public IC shard is outside its original generation')
        bindings.append({'path':str(path),'sha256':checksum})
    transfer=run/'input/monofonic_input_transfer.txt'
    if pins.get(str(transfer))!=file_sha256(transfer):raise ValueError('Original generated monofonIC transfer/velocity validation source required')
    descriptor=Path(cfg['spectrum']['file']).resolve()
    _merge(pins,{str(descriptor):file_sha256(descriptor)})
    contract={'schema':'validated-public-native-ic-v1','source_run':str(run),'source_receipt':{'path':str(receipt),'sha256':pins[str(receipt)]},
        'phase_seed':parameters['phase_seed'],'primary':next(b for b in bindings if b['path']==str(primary)),'files':bindings,
        'phase_asset':{'path':str(run/'config.yaml'),'sha256':pins[str(run/'config.yaml')]},
        'linear_bundle':{'path':str(descriptor),'sha256':pins[str(descriptor)]},
        'transfer_validation':{'path':str(transfer),'sha256':pins[str(transfer)]},
        'initial_condition_validation':record['initial_conditions'],'power_validation':record['initial_ic_power_spectrum'],
        'source_config':{'physics':'hydro','ic_engine_hydro':'monofonic','box_mpc_h':cfg['simulation']['box_mpc_h'],
            'particles_per_dimension':cfg['simulation']['particles_per_dimension']},
        'scope':'Actual original validated IC-only generation. Source transfer/velocity and power gates must run again when reused. Independent seed design is not empirical independence/science acceptance.',
        'science_acceptance':'NOT_ACCEPTED'}
    for path,h in pins.items():
        if file_sha256(path)!=h:raise ValueError('Native public IC input changed')
    return contract,pins


def public_ic_runner(parameters,context):
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native public IC intake exceeds stage allocation')
    contract,pins=public_ic_inputs(parameters)
    atomic_json(context.generation/'validated-ic.json',{**contract,'source_hashes':pins})
    return {'execution_status':'COMPLETE','measurements':{'native_validated_ic':{'phase_seed':contract['phase_seed'],
        'particle_shards':len(contract['files']),'science_acceptance':'NOT_ACCEPTED'}}}


def load_public_ic(path,*,memory_gib):
    if type(memory_gib) not in (int,float) or not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError('Public IC intake needs a finite positive memory allocation')
    path=Path(path).resolve()
    if path.stat().st_size*24>memory_gib*1024**3/8:raise MemoryError('Validated public IC metadata exceeds allocation')
    original=file_sha256(path)
    report=json.loads(path.read_text())
    if report.get('schema')!='validated-public-native-ic-v1':raise ValueError('Original validated public-native-IC contract required')
    receipt=report['source_receipt']
    if file_sha256(receipt['path'])!=receipt['sha256']:raise ValueError('Original IC-only stage receipt changed')
    actual,pins=public_ic_inputs({'source_run':report['source_run'],'source_receipt':receipt['path'],'phase_seed':report['phase_seed'],'memory_gib':memory_gib})
    if any(report.get(key)!=value for key,value in actual.items()):raise ValueError('Validated public IC contract does not replay its actual generator receipt')
    for p,h in report['source_hashes'].items():
        if file_sha256(p)!=h:raise ValueError('Validated public IC source changed')
    if file_sha256(path)!=original:raise ValueError('Validated public IC capsule changed during intake')
    _merge(pins,report['source_hashes']);_merge(pins,{str(path):original})
    return report,pins


def apply_public_ic(cfg,path,existing_ic):
    """Bind the original transfer source; do not skip any native IC checks."""
    from core.resources import memory_limit_gb
    memory=memory_limit_gb(cfg)
    report,pins=load_public_ic(path,memory_gib=memory)
    if Path(existing_ic).resolve()!=Path(report['primary']['path']).resolve():raise ValueError('Requested existing IC differs from its actual public generation')
    if cfg['simulation']['random_seed']!=report['phase_seed']:raise ValueError('Native evolution relabels the original IC phase seed')
    source=report['source_config']
    if any(cfg['simulation'].get(k)!=source[k] for k in source):raise ValueError('Native response changes the original IC species, generator, box or particle load')
    if file_sha256(cfg['spectrum']['file'])!=report['linear_bundle']['sha256']:raise ValueError('Native response changes the original IC linear cosmology')
    cfg['simulation']['monofonic_transfer_validation_source']=report['transfer_validation']['path']
    return report,pins


def public_ic_command_bindings(parameters):
    """Read original IC provenance before evolution or reuse of a saved stage."""
    if parameters.get('command') != 'gadget':
        return {}
    arguments = parameters.get('arguments',[])
    if '--validated-public-ic' not in arguments:
        return {}

    def option(flag):
        if arguments.count(flag) != 1:
            raise ValueError('Exactly one '+flag+' input is required')
        index = arguments.index(flag)+1
        if index >= len(arguments) or not isinstance(arguments[index],str) or not arguments[index]:
            raise ValueError('A resolved file path must follow '+flag)
        return arguments[index]

    capsule = option('--validated-public-ic')
    answer = Path(option('--answers')).resolve()
    existing_ic = option('--existing-ic')
    if answer.stat().st_size > 16*1024**2:
        raise MemoryError('Native answer metadata exceeds input allocation')
    value = json.loads(answer.read_text())
    from core.research.capabilities import validate_native_answers
    resources = {'memory_gib':value['memory_limit_gb'],
        'mpi_ranks':value['mpi_ranks'],'omp_threads':value['omp_threads']}
    cfg = validate_native_answers(value,resources)
    _, pins = apply_public_ic(cfg,capsule,existing_ic)
    _merge(pins,{str(answer):file_sha256(answer)})
    return pins


def prepare_response_from_generated_ics(design,ic_manifest,ic_execution,destination,*,request_base):
    """Attach completed generator products to the original response compiler.

    Preparation cannot invent hashes of future ICs. This second preparation
    step consumes their actual completed workflow receipts after generation.
    It never executes response simulations or grants calibration acceptance.
    """
    from core.workflow import verify_completed_receipt
    from core.sidm_analysis import compile_joint_response_campaign
    if design.get('schema')!='native-joint-response-design-v1' or 'realizations' in design:
        raise ValueError('Response design must omit future IC bindings; completed generators supply them')
    path=Path(ic_manifest).resolve();manifest=json.loads(path.read_text());execution=Path(ic_execution).resolve()
    if manifest.get('schema')!='native-joint-ic-generation-prepared-v1':raise ValueError('Original native IC generation preparation required')
    workflow=Path(manifest['workflow']['path']).resolve()
    if file_sha256(workflow)!=manifest['workflow']['sha256']:raise ValueError('Original IC generation workflow changed')
    pins={str(path):file_sha256(path),str(workflow):file_sha256(workflow)};_merge(pins,manifest['source_hashes'])
    for p,h in pins.items():
        if file_sha256(p)!=h:raise ValueError('Original IC preparation source changed; prepare a fresh recipe')
    statepath=execution/'workflow.json';state=json.loads(statepath.read_text())
    if (state.get('schema')!='research-workflow-receipt-v1' or state.get('execution_status')!='COMPLETE'
            or state['specification_hash']!=canonical_digest(json.loads(workflow.read_text()))):
        raise ValueError('Actual completed original IC generation workflow required')
    _merge(pins,{str(statepath):file_sha256(statepath)})
    request=copy.deepcopy(design);request['schema']='native-joint-response-campaign-request-v1';request['realizations']={'training':[],'heldout':[]}
    for row in manifest['realizations']:
        receipts={}
        for key,kind in [('generation_stage','public_command'),('validation_stage','native_public_ic')]:
            entry=state['stages'][row[key]];generation=Path(entry['generation']).resolve();receipt=generation/'stage-receipt.json'
            if not generation.is_relative_to(execution/'generations') or file_sha256(receipt)!=entry['receipt_sha256']:raise ValueError('Actual owned IC generation receipt changed')
            owned=verify_completed_receipt(receipt)
            if owned['kind']!=kind or Path(owned['generation']).resolve()!=generation:raise ValueError('Unexpected IC generation/validation producer')
            receipts[key]=(generation,receipt,owned)
            _merge(pins,{str(receipt):file_sha256(receipt)})
        directory,receipt,owned=receipts['validation_stage'];capsule=directory/'validated-ic.json'
        if owned['files_sha256'].get('validated-ic.json')!=file_sha256(capsule):raise ValueError('Verified IC capsule is outside its original generation')
        report,binding=load_public_ic(capsule,memory_gib=design['resources']['memory_gib']);_merge(pins,binding)
        original,original_receipt,_=receipts['generation_stage']
        if (report['phase_seed']!=row['phase_seed'] or Path(report['source_run']).resolve()!=original/'run'
                or Path(report['source_receipt']['path']).resolve()!=original_receipt):raise ValueError('IC realization differs from the actual original generation')
        request['realizations'][row['role']].append({key:row[key] for key in ('realization_id','phase_seed','sidm_operator_seed')})
        request['realizations'][row['role']][-1]['initial_conditions']={'validated_public_ic':{'path':str(capsule),'sha256':file_sha256(capsule)}}
    result=compile_joint_response_campaign(request,destination,request_base=request_base)
    for p,h in pins.items():
        if file_sha256(p)!=h:raise ValueError('Generated IC response binding changed during preparation')
    atomic_json(Path(destination)/'generated-ic-bindings.json',{'schema':'native-generated-joint-response-bindings-v1',
        'source_hashes':pins,'realizations':request['realizations'],'native_execution':'IC_GENERATION_ONLY;RESPONSE_NOT_EXECUTED',
        'science_acceptance':'NOT_ACCEPTED'})
    return result


def prepare_joint_ic_generation(request,destination):
    """Prepare disjoint actual public generator jobs, without generating ICs."""
    from core.research.capabilities import validate_native_answers
    from core.workflow import default_registry,compile_requested_product_dag,safe_id
    required={'schema','base_answers','realizations','resources','wall_seconds_per_generation'}
    if set(request)!=required or request['schema']!='native-joint-ic-generation-request-v1':raise ValueError('Complete native calibration IC generation request required')
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Preserve prior native calibration IC preparation')
    resources=copy.deepcopy(request['resources']);memory=resources['memory_gib'];wall=request['wall_seconds_per_generation']
    if type(wall) not in (int,float) or not math.isfinite(wall) or wall<=0:raise ValueError('Positive finite IC generator wall allocation required')
    reference=request['base_answers'];path=Path(reference['path']).resolve()
    if file_sha256(path)!=reference['sha256']:raise ValueError('Original native calibration base answers changed')
    if path.stat().st_size*24>memory*1024**3/8:raise MemoryError('Native IC base answers exceed metadata allocation')
    answers=json.loads(path.read_text());cfg=validate_native_answers(answers,resources)
    if cfg['simulation']['physics']!='hydro' or cfg['simulation'].get('ic_engine_hydro')!='monofonic':raise ValueError('Original uniform hydro monofonIC answers required for this IC generation branch')
    realizations=request['realizations']
    if set(realizations)!={'training','heldout'}:raise ValueError('Separate native training and heldout generation roles required')
    seen=set();seeds=set();operators=set();rows=[]
    for role,values in realizations.items():
        if not isinstance(values,list) or len(values)<3:raise ValueError('Retain at least three independent native training and heldout IC/operator realizations')
        for row in values:
            if set(row)!={'realization_id','phase_seed','sidm_operator_seed'}:raise ValueError('Explicit native realization identity/IC/operator seeds required')
            name=safe_id(row['realization_id']);phase=row['phase_seed'];operator=row['sidm_operator_seed']
            if name in seen or type(phase) is not int or not 0<=phase<2**64 or phase in seeds:raise ValueError('Training and heldout IC seeds/identities must be distinct')
            if type(operator) is not int or not 0<=operator<2**64 or operator in operators:raise ValueError('Training and heldout SIDM operator seeds must be distinct')
            seen.add(name);seeds.add(phase);operators.add(operator);rows.append({'role':role,**row})
    if len(rows)*2>resources['maximum_stage_count']:raise MemoryError('Full native IC generation/verification graph exceeds stage ceiling')
    out.mkdir(parents=True);stages=[];pins={str(path):reference['sha256']};records=[]
    for row in rows:
        name=row['realization_id'];current=copy.deepcopy(answers);current['name']='calibration-ic-'+name;current['seed']=row['phase_seed']
        if current.get('sidm',{}).get('enabled',False):current['sidm']['operator_seed']=row['sidm_operator_seed']
        validate_native_answers(current,resources)
        answer=out/(name+'.answers.json');atomic_json(answer,current);binding={'path':str(answer),'sha256':file_sha256(answer)};_merge(pins,{binding['path']:binding['sha256']})
        generate='generate_'+name;verify='validate_'+name
        stages.append({'stage_id':generate,'kind':'public_command','parameters':{'command':'gadget','wall_seconds':wall,
            'arguments':['--answers',{'input':str(answer)},'--ic-only','--run-dir','{generation}/run']},'input_files':[binding],
            'outputs':{generate:'run'}})
        stages.append({'stage_id':verify,'kind':'native_public_ic','parameters':{'source_run':{'stage':generate,'artifact':'run'},
            'source_receipt':{'stage':generate,'artifact':'stage-receipt.json'},'phase_seed':row['phase_seed'],'memory_gib':memory},
            'outputs':{verify:'validated-ic.json'}})
        records.append({**row,'generation_stage':generate,'validation_stage':verify})
    workflow={'schema':'research-workflow-v1','resources':resources,'physics_spec':{},'capabilities':[],
        'followup_limits':{'maximum_children':0,'maximum_depth':0},'requested_products':[r['validation_stage'] for r in records],'stages':stages}
    registry=default_registry();compile_requested_product_dag(workflow,registry,request_base=out)
    for kind in dict.fromkeys(stage['kind'] for stage in stages):
        _merge(pins,registry.source_hashes(kind))
    for p,h in pins.items():
        if file_sha256(p)!=h:raise ValueError('Native IC generator source changed during preparation')
    atomic_json(out/'workflow.json',workflow)
    result={'schema':'native-joint-ic-generation-prepared-v1','status':'RECIPES_PREPARED','realizations':records,
        'workflow':{'path':str(out/'workflow.json'),'sha256':file_sha256(out/'workflow.json')},'source_hashes':pins,
        'native_execution':'NOT_EXECUTED','science_acceptance':'NOT_ACCEPTED',
        'scope':'Distinct actual public monofonIC generation seeds and original IC-only validation stages; no realization reused between training/heldout. Native response/calibration stages still attach separately.'}
    atomic_json(out/'manifest.json',result);return result


# Paired native stream response intake and campaign extension

def validate_paired_response(parameters):
    m=parameters['memory_gib']
    if type(m) not in (int,float) or not math.isfinite(m) or m<.1:raise ValueError('Bounded native paired-response memory required')


def _paired_response_template(parameters):
    validate_paired_response(parameters);path=Path(parameters['template_path']).resolve()
    if path.stat().st_size*16>parameters['memory_gib']*1024**3/16:raise MemoryError('Native response template exceeds workspace')
    if file_sha256(path)!=parameters['template_sha256']:raise ValueError('Frozen native response template changed')
    template=json.loads(path.read_text())
    if set(template)!={'schema','frame_name','origin_policy','response_controls'} or template['schema']!='native-paired-response-template-v1':
        raise ValueError('Complete frozen native paired-response template required')
    if template['origin_policy']!='native_box_origin' or not isinstance(template['frame_name'],str) or not template['frame_name']:
        raise ValueError('Explicit unchanged native box-origin Cartesian response chart required')
    required={'track_epoch_index','curve_controls','bin_fractions','measurement_controls','model_error_covariance',
        'bootstrap_replicates','seed','feature_controls','actions','export_figures','figure_dpi'}
    if set(template['response_controls'])!=required:raise ValueError('Complete existing native response estimator controls required')
    return template,path


def _paired_response_inputs(parameters):
    from .sidm_analysis import _load_product
    template,path=_paired_response_template(parameters);memory=parameters['memory_gib'];comparison=Path(parameters['comparison']).resolve()
    paired,_,pins=_load_product({'path':str(comparison),'sha256':file_sha256(comparison)},comparison.parent,memory*1024**3/8)
    _merge(pins,{str(path):parameters['template_sha256']})
    if paired.get('schema')!='measured-native-stream-paired-comparison-v1' or paired.get('execution_status')!='COMPLETE':
        raise ValueError('Original completed paired native comparison required')
    histories=[];base=Path(paired['request_base']).resolve()
    for name in ('reference','intervention'):
        binding=paired['request'][name];raw,_,sources=_load_product(binding,base,memory*1024**3/8)
        if raw.get('schema')!='measured-native-stream-worldlines-v1' or raw.get('request',{}).get('species')!=4:
            raise ValueError('Native paired response orchestration requires original selected stellar histories')
        _merge(pins,sources)
        histories.append((raw,(base/binding['path']).resolve()))
    if histories[0][0]['periodic']!=histories[1][0]['periodic']:raise ValueError('Native response branches change boundary topology')
    left,right=[raw['epochs'] for raw,_ in histories]
    if len(left)!=len(right):raise ValueError('Native response branches require the same physical epoch count')
    for a,b in zip(left,right):
        kind=a['canonical_time']
        if kind!=b['canonical_time']:raise ValueError('Native response branches change clock representation')
        if kind=='scale_factor':
            keys=('actual_scale_factor',)
        elif kind=='isolated_native_elapsed_time':
            keys=('native_time_code','time_unit_gyr')
        else:
            raise ValueError('Native paired response requires a supported physical clock representation')
        for key in keys:
            time_a,time_b=a.get(key),b.get(key)
            if (isinstance(time_a,bool) or not isinstance(time_a,(int,float)) or not math.isfinite(time_a)
                    or isinstance(time_b,bool) or not isinstance(time_b,(int,float)) or not math.isfinite(time_b)
                    or time_a!=time_b):
                raise ValueError('Native response branches change or omit actual physical epochs')
        if ((kind=='scale_factor' and a['actual_scale_factor']<=0)
                or (kind=='isolated_native_elapsed_time' and a['time_unit_gyr']<=0)):
            raise ValueError('Native paired response requires a positive scale factor or isolated time unit')
    lineage=None
    if parameters.get('native_result') is not None:
        result_path=Path(parameters['native_result']).resolve()
        lineage,_,sources=_load_product({'path':str(result_path),'sha256':file_sha256(result_path)},result_path.parent,memory*1024**3/8)
        if lineage.get('schema')=='measured-native-live-counterfactual-result-v1':references=[lineage['paired_comparison']]
        elif lineage.get('schema')=='measured-native-late-threeway-comparison-v1':
            references=[lineage['request'][name] for name in ('restart_comparison','intervention_comparison','total_comparison')]
        else:raise ValueError('Original native intervention lineage/threeway result required')
        if file_sha256(comparison) not in {ref['sha256'] for ref in references}:raise ValueError('Native intervention result does not bind this paired comparison')
        _merge(pins,sources)
    return template,paired,histories,lineage,pins


def paired_response_input_bindings(parameters):
    template,_,histories,_,pins=_paired_response_inputs(parameters)
    actions=template['response_controls']['actions']
    if actions is not None:
        from core.stream.workflow import Inputs,_potential
        inputs=Inputs(Path(parameters['template_path']).resolve().parent,{}, {'memory_gib':parameters['memory_gib']/4,
            'maximum_particles':histories[0][0]['request'].get('maximum_selected_particles',1)})
        _potential(inputs,actions['potential']);_merge(pins,inputs.sources)
    return pins


def native_paired_response_runner(parameters,context):
    from .periodic_images import publish_native_worldline_images
    from .flagship_products import native_encounter_frame
    from .native_stream_responses import native_response_runner
    validate_paired_response(parameters);memory=parameters['memory_gib']
    if memory>context.resources['memory_gib']:raise MemoryError('Native paired response exceeds workflow allocation')
    template,paired,histories,lineage,pins=_paired_response_inputs(parameters);frames={};unsupported=[]
    for name,(raw,path) in zip(('reference','intervention'),histories):
        n=len(raw['epochs']);request={'schema':'native-encounter-frame-v1','frame':template['frame_name'],
            'origin_proper_kpc':[[0.,0.,0.]]*n,'origin_velocity_peculiar_kms':[[0.,0.,0.]]*n}
        # The frame conversion uses original native H(t), saved times and
        # per-drift image counters. It does not interpolate a trajectory or
        # select a cadence; this consumer measures actual saved snapshots.
        if raw['periodic']:
            images=context.generation/(name+'-images.json')
            support=publish_native_worldline_images(path,images,memory_gib=memory)
            if support['status']!='PASS':unsupported.append({'branch':name,'reason':'Recorded native image support unavailable/discontinuous','product':str(images)});continue
            request['recorded_images']={'path':str(images),'sha256':file_sha256(images)}
        frame=context.generation/(name+'-frame.json');native_encounter_frame(path,request,frame,memory_gib=memory)
        frames[name]={'path':str(frame),'sha256':file_sha256(frame)}
    result={'schema':'native-paired-response-result-v1','comparison':{'path':str(Path(parameters['comparison']).resolve()),'sha256':file_sha256(parameters['comparison'])},
        'science_acceptance':'NOT_ACCEPTED','source_hashes':pins,'native_intervention_evidence':None if lineage is None else {
            'path':str(Path(parameters['native_result']).resolve()),'sha256':file_sha256(parameters['native_result']),
            'schema':lineage['schema'],'original_status':lineage.get('causal_intervention_status',lineage.get('status'))},
        'scope':'Original paired native stellar cohort and conditional response estimators; native intervention domain/lineage limits remain in the retained source result'}
    if unsupported:result.update(status='UNRESOLVED',unsupported=unsupported)
    else:
        request={'schema':'native-stream-response-request-v1',**copy.deepcopy(template['response_controls']),
            'comparison':result['comparison'],'reference_frame':frames['reference'],'intervention_frame':frames['intervention']}
        request_path=context.generation/'response-request.json';atomic_json(request_path,request)
        numerical=native_response_runner({'request_path':str(request_path),'request_sha256':file_sha256(request_path),'memory_gib':memory},context)
        product=context.generation/'stream-responses.json'
        result.update(status=numerical['status'],response={'path':str(product),'sha256':file_sha256(product)},variants=numerical['variants'])
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Native paired response input changed')
    atomic_json(context.generation/'paired-response.json',result);return result


def prepare_native_response_campaign(design_path,template_binding,destination):
    from .workflow import compile_requested_product_dag,default_registry,safe_id
    design_path=Path(design_path).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Keep previous native response campaign preparations')
    design=json.loads(design_path.read_text())
    if design.get('schema') not in {'native-live-counterfactual-campaign-design-v1','native-late-counterfactual-campaign-design-v1'}:
        raise ValueError('Original native intervention campaign design required')
    workflow=Path(design['workflow']).resolve()
    if file_sha256(workflow)!=design['workflow_sha256']:raise ValueError('Original native intervention workflow changed')
    sources={str(design_path):file_sha256(design_path),str(workflow):design['workflow_sha256']}
    _merge(sources,design['source_hashes'])
    template_path=Path(template_binding['path']).resolve()
    if file_sha256(template_path)!=template_binding['sha256']:raise ValueError('Frozen native response campaign template changed')
    spec=json.loads(workflow.read_text());memory=spec['resources']['memory_gib']
    _paired_response_template({'template_path':str(template_path),'template_sha256':template_binding['sha256'],'memory_gib':memory})
    _merge(sources,{str(template_path):template_binding['sha256']})
    original=list(spec['stages']);ids={row['stage_id'] for row in original};created=[]
    mappings=design.get('mapped_native_branches',[])
    for row in original:
        if row['kind']!='native_stream_comparison':continue
        identity=safe_id(row['stage_id']+'_responses')
        if identity in ids:raise ValueError('Native response stage identity collides with original campaign')
        # Native compiler outputs are explicit. Resolve each comparison from
        # its declared producer artifact rather than guessing a future path.
        outputs=list(row['outputs'].values())
        if len(outputs)!=1:raise ValueError('Native paired producer must declare one exact scientific comparison output')
        parameters={'comparison':{'stage':row['stage_id'],'artifact':outputs[0]},'template_path':{'input':str(template_path)},
            'template_sha256':template_binding['sha256'],'memory_gib':memory}
        matches=[mapping['native_result_product'] for mapping in mappings if mapping['comparison_product']==row['stage_id']]
        if design['schema']=='native-late-counterfactual-campaign-design-v1':matches=['late_threeway']
        dependencies=[row['stage_id']]
        if matches:
            if len(set(matches))!=1:raise ValueError('Paired comparison has ambiguous native intervention result ownership')
            native_id=matches[0];native=next(stage for stage in original if stage['stage_id']==native_id);artifacts=list(native['outputs'].values())
            if len(artifacts)!=1:raise ValueError('Native intervention result must declare one exact source product')
            parameters['native_result']={'stage':native_id,'artifact':artifacts[0]};dependencies.append(native_id)
        spec['stages'].append({'stage_id':identity,'kind':'native_paired_stream_responses','parameters':parameters,'depends_on':dependencies,
            'input_files':[{'path':str(template_path),'sha256':template_binding['sha256']}],
            'outputs':{identity:'paired-response.json'}})
        spec['requested_products'].append(identity);ids.add(identity);created.append(identity)
    if not created:raise ValueError('Native intervention design has no paired comparison producers')
    spec['resources']['maximum_stage_count']+=len(created)
    registry=default_registry();compile_requested_product_dag(spec,registry,request_base=design_path.parent)
    for kind in dict.fromkeys(row['kind'] for row in spec['stages']):
        _merge(sources,registry.source_hashes(kind))
    _merge(sources,{str(Path(__file__).resolve()):file_sha256(__file__)})
    for path,digest in sources.items():
        if file_sha256(path)!=digest:raise ValueError('Native intervention/response source changed')
    destination.mkdir(parents=True);atomic_json(destination/'workflow.json',spec)
    result={'schema':'native-intervention-response-campaign-v1','status':'RECIPES_PREPARED','source_design':str(design_path),
        'workflow':{'path':str(destination/'workflow.json'),'sha256':file_sha256(destination/'workflow.json')},
        'response_stages':created,'source_hashes':sources,'science_acceptance':'NOT_ACCEPTED',
        'scope':'Existing native intervention graph plus source-bound frame/image/pair-response consumers. Preparation executes no native evolution; original domain/lineage gates unchanged.'}
    atomic_json(destination/'manifest.json',result);return result


# Baseline-selected native material targets and final-epoch observables

def response_target_controls(path,sha,memory):
    """Read explicit saved-epoch selectors and dimensional scalar measurements.

    The final selector is the latest stored epoch. Radial bins index declared
    proper-kpc shell edges; central density is one aperture scalar per epoch.
    """
    path=Path(path).resolve()
    if type(memory) not in (int,float) or not math.isfinite(memory) or memory<=0:raise ValueError('Positive response target allocation required')
    if path.stat().st_size*24>memory*1024**3/8:raise MemoryError('Response target controls exceed allocation')
    if file_sha256(path)!=sha:raise ValueError('Original response target controls changed')
    value=json.loads(path.read_text())
    if set(value)!={'schema','selection_policy','matching_controls','history','observables'} or value['schema']!='native-response-target-controls-v1':raise ValueError('Complete native response target controls required')
    if value['selection_policy']!='largest_clean_fine_dm_fof_at_last_saved_epoch':raise ValueError('Explicit fixed baseline target population policy required')
    rules=value['matching_controls']
    if (set(rules)!={'minimum_shared_particles','minimum_material_fraction','maximum_runner_up_fraction'}
            or type(rules['minimum_shared_particles']) is not int or rules['minimum_shared_particles']<100
            or type(rules['minimum_material_fraction']) not in (int,float) or not .5<=rules['minimum_material_fraction']<=1
            or type(rules['maximum_runner_up_fraction']) not in (int,float) or not 0<=rules['maximum_runner_up_fraction']<=.2):
        raise ValueError('Retain original count/material/runner-up target gates or stricter settings')
    history=value['history'];snapshots=history['snapshots']
    if (history.get('schema')!='native-recovered-sidm-controls-v1' or not isinstance(snapshots,list) or len(snapshots)<2
            or any(type(i) is not int or i<0 for i in snapshots) or snapshots!=sorted(set(snapshots))):
        raise ValueError('Complete distinct native history epochs required')
    if type(history['maximum_particles']) is not int or history['maximum_particles']<100:raise ValueError('Exact native particle ceiling required')
    if history['species_components'].get('1')!='dm':raise ValueError('Original fine type1 science DM required')
    controls=history['history_controls'];allowed={'central_density_msun_kpc3':('central_supported','Msun / proper kpc^3'),
        'density_msun_kpc3':('profile_supported','Msun / proper kpc^3'),
        'velocity_dispersion_squared_km2_s2':('profile_supported','(km / s)^2')}
    names=set()
    if not isinstance(value['observables'],list) or not value['observables']:raise ValueError('Explicit original scalar response observable selectors required')
    for row in value['observables']:
        if set(row)!={'name','array','support_array','radial_bin','units'} or row['array'] not in allowed or (row['support_array'],row['units'])!=allowed[row['array']]:raise ValueError('Original response observable/support/unit contract required')
        if not isinstance(row['name'],str) or not row['name'] or row['name'] in names:raise ValueError('Distinct named response observables required')
        names.add(row['name']);index=row['radial_bin']
        if row['array']=='central_density_msun_kpc3':
            if index is not None:raise ValueError('Original central aperture is scalar, not a radial bin')
        elif type(index) is not int or not 0<=index<len(controls['radial_edges_proper_kpc'])-1:raise ValueError('Exact original radial bin required')
    return value


def response_target_inputs(parameters):
    from .analysis import load_cfg,_initial_condition_paths
    from .sidm_analysis import _stage_native_bindings
    memory=parameters['memory_gib'];settings=response_target_controls(parameters['template_path'],parameters['template_sha256'],memory)
    runs={};configs={};pins={};initial={}
    for label,runkey,receiptkey in [('current','native_run_directory','native_stage_receipt'),('baseline','baseline_run_directory','baseline_stage_receipt')]:
        run=Path(parameters[runkey]).resolve();runs[label]=run;configs[label]=load_cfg(run)
        native=_stage_native_bindings({'request':{'memory_gib':memory},'native_run_directory':str(run),'native_stage_receipt':parameters[receiptkey]})
        _merge(pins,native)
        paths=[Path(p).resolve() for p in _initial_condition_paths(run)]
        if not paths:raise ValueError('Actual completed response evolution must bind its original IC shards')
        hashes=[]
        for path in paths:
            sha=file_sha256(path)
            if native.get(str(path))!=sha:raise ValueError('Response IC shard is outside original native evolution bindings')
            hashes.append(sha)
        initial[label]=sorted(hashes)
    if initial['current']!=initial['baseline']:raise ValueError('Response target matching requires identical complete actual IC byte sets within each realization')
    if configs['current']['simulation']['random_seed']!=configs['baseline']['simulation']['random_seed']:raise ValueError('Response target changes actual realization phase seed')
    if configs['current'].get('sidm',{}).get('operator_seed')!=configs['baseline'].get('sidm',{}).get('operator_seed'):raise ValueError('Response target changes paired SIDM operator realization')
    _merge(pins,{str(Path(parameters['template_path']).resolve()):parameters['template_sha256']})
    return settings,runs,configs,pins,'native_fine_dm_ic:'+canonical_digest(initial['current'])


def baseline_target(catalogue,rules,maximum,memory):
    """Select the largest clean baseline fine-DM FOF population by exact count.

    R200c must be measured and positive; coarse-DM contamination is excluded.
    Original catalogue row breaks exact count ties within this baseline only.
    Other parameter points recover material by exact IDs, never row equality.
    """
    from .data import snapshot_id_shards,read_selected_group_ids,exact_uint64_ids
    count=catalogue['lengths'][:,1]
    valid=np.flatnonzero((count>=rules['minimum_shared_particles'])&(catalogue['lengths'][:,2]==0)&(catalogue['radii']>0))
    if not len(valid):return None,np.empty(0,np.uint64)
    # A row only breaks ties inside this single baseline catalogue. Every
    # subsequent host is independently recovered from exact particle IDs.
    # Compare exact Python integers: uint64 counts above int64's maximum
    # must not wrap and change which physical population is selected.
    row=min(map(int,valid),key=lambda index:(-int(count[index]),index))
    size=int(count[row])
    if size>maximum or size*128>memory*1024**3/3:raise MemoryError('Complete selected baseline fine-DM cohort exceeds allocation')
    shards,_=snapshot_id_shards(catalogue['snapshots'],1)
    ids,_=read_selected_group_ids(catalogue['offsets'],catalogue['lengths'],np.array([row],np.int64),shards,particle_type=1,max_ids=maximum)
    ids=exact_uint64_ids(ids)
    if len(ids)!=size:raise ValueError('Original baseline FOF fine-DM span is incomplete')
    return row,ids


def native_response_target_runner(parameters,context):
    """Recover one baseline-selected cohort and measure its original history.

    Density is Msun/proper-kpc^3 and velocity variance is (km/s)^2. Each
    exported scalar retains the producer's Boolean support. Missing recovery
    remains NaN/unsupported, never a fabricated zero response.
    """
    from .zoom import read_host_catalogue
    from core.research.zooms import recover_lagrangian_descendant
    from .sidm_recovery import recovered_sidm_inputs
    from .sidm_campaign import native_branch_measurements
    from .sidm_analysis import publish_sidm_membership,publish_sidm_history,_load_product
    from .data import export_numeric_audit, run_epoch_contract
    import h5py
    memory=parameters['memory_gib']
    if memory>context.resources['memory_gib']:raise MemoryError('Native response target exceeds stage allocation')
    settings,runs,configs,pins,namespace=response_target_inputs(parameters);out=context.generation
    history=settings['history'];snapshots=history['snapshots'];final=snapshots[-1]
    catalogue=read_host_catalogue(runs['baseline'],final,memory_gib=memory)
    for binding in catalogue['sources']+catalogue['snapshot_sources']:
        if pins.get(binding['path'])!=binding['sha256']:raise ValueError('Baseline target catalogue differs from its original completed evolution')
    row,ids=baseline_target(catalogue,settings['matching_controls'],history['maximum_particles'],memory)
    status='UNRESOLVED_BASELINE_TARGET';recovery=None;native_history=None
    values={};supports={};epoch=None
    # Retain an actual source epoch even if there is no resolved target.
    current=catalogue if runs['current']==runs['baseline'] else read_host_catalogue(runs['current'],final,memory_gib=memory)
    for binding in current['sources']+current['snapshot_sources']:
        if pins.get(binding['path'])!=binding['sha256']:raise ValueError('Current target epoch differs from original completed evolution')
    for selected_catalogue in (catalogue,current):
        with h5py.File(selected_catalogue['snapshots'][0],'r') as snapshot_file:
            if int(snapshot_file['Parameters'].attrs['ComovingIntegrationOn'])!=1:
                raise ValueError('Native response FOF/R200c target requires a cosmological clock')
    epoch=run_epoch_contract(runs['current'],final,float(current['frame'][0]))
    if row is not None:
        np.save(out/'baseline-fine-dm-ids.npy',ids,allow_pickle=False)
        target={'namespace':namespace+':baseline_fof_cohort:'+canonical_digest(ids.tolist()),'object_id':row}
        recovery=recover_lagrangian_descendant(runs['current'],ids,snapshots,settings['matching_controls'],memory_gib=memory)
        _merge(pins,recovery['source_hashes']);atomic_json(out/'recovery.json',recovery)
        template=out/'history-controls.json';atomic_json(template,history)
        arguments={'native_run_directory':str(runs['current']),'native_stage_receipt':parameters['native_stage_receipt'],
            'recovery_path':str(out/'recovery.json'),'template_path':str(template),'template_sha256':file_sha256(template),
            'particle_namespace':namespace,'matching_identity':target,'memory_gib':memory}
        _,_,accepted,_,matching,enabled,bindings=recovered_sidm_inputs(arguments);_merge(pins,bindings)
        if len(accepted)==len(snapshots):
            branch={'schema':'native-sidm-branch-measurements-v1','observations':accepted,'maximum_particles':history['maximum_particles'],
                'support_radius_proper_kpc':history['support_radius_proper_kpc'],'species_components':history['species_components'],
                'history_controls':history['history_controls'],'satellite':None}
            templates=native_branch_measurements(branch,memory_gib=memory);epochs=[]
            base={'native_run_directory':str(runs['current']),'native_stage_receipt':parameters['native_stage_receipt'],'request_base':str(out)}
            for i,entry in enumerate(accepted):
                request=copy.deepcopy(templates[i]['parameters']['request']);request.update(particle_namespace=namespace,matching=matching)
                path=out/f'members-{i:03d}.json';measured=publish_sidm_membership({**base,'request':request},path)
                epochs.append({'snapshot':entry['snapshot'],'elapsed_time_gyr':measured['elapsed_time_gyr'],'selection':{'path':str(path),'sha256':file_sha256(path)}})
            request=copy.deepcopy(templates[-1]['parameters']['request']);request.update(particle_namespace=namespace,matching=matching,
                dark_matter_model='elastic_sidm' if enabled else 'collisionless',require_sidm_diagnostics=enabled,epochs=epochs)
            path=out/'native-history.json';publish_sidm_history({**base,'request':request},path)
            native_history={'path':str(path),'sha256':file_sha256(path)}
            report,arrays,bindings=_load_product(native_history,out,memory*1024**3);_merge(pins,bindings)
            if [e['snapshot'] for e in report['epochs']]!=snapshots:raise ValueError('Original response target history loses requested epochs')
            measured_epoch=report['epochs'][-1]['epoch']
            if measured_epoch['actual_scale_factor']!=epoch['actual_scale_factor']:
                raise ValueError('Original response measurement changes actual saved target epoch')
            epoch=measured_epoch
            status='MEASURED_NATIVE_TARGET'
            for i,selector in enumerate(settings['observables']):
                contracts=report['numeric_data']['array_contract']['datasets']
                if contracts[selector['array']]['units']!=selector['units'] or contracts[selector['support_array']]['units']!='dimensionless':raise ValueError('Original native response observable units differ')
                index=(-1,) if selector['radial_bin'] is None else (-1,selector['radial_bin'])
                field=np.asarray(arrays[selector['array']]);mask=np.asarray(arrays[selector['support_array']])
                if mask.dtype!=np.dtype(bool) or mask.shape!=field.shape:raise ValueError('Original response target scalar support shape differs')
                supported=bool(mask[index]);value=float(field[index])
                if supported and not math.isfinite(value):raise ValueError('Original supported response measurement is nonfinite')
                values[f'observable_{i:03d}']=np.array([value if supported else np.nan]);supports[f'supported_{i:03d}']=np.array([supported])
        else:status='UNRESOLVED_MATERIAL_HISTORY'
    for i,selector in enumerate(settings['observables']):
        values.setdefault(f'observable_{i:03d}',np.array([np.nan]));supports.setdefault(f'supported_{i:03d}',np.array([False]))
    definitions={'selection_policy':settings['selection_policy'],'cohort':'baseline final-epoch exact fine-DM FOF membership; original material recovery at each parameter point',
        'matching_controls':settings['matching_controls'],'measurement_controls':history,'observable_selectors':settings['observables'],
        'scope':'conditional baseline-selected halo population per independent realization; local physical response, no global posterior or production-resolution acceptance'}
    report={'schema':'native-response-target-measurement-v1','execution_status':'COMPLETE','status':status,'epoch':epoch,
        'physical_definitions':definitions,'native_stage_receipt':str(Path(parameters['native_stage_receipt']).resolve()),
        'baseline_stage_receipt':str(Path(parameters['baseline_stage_receipt']).resolve()),'baseline_host_row':row,'native_history':native_history,
        'recovery_status':None if recovery is None else recovery['status'],'source_hashes':pins,'science_acceptance':'NOT_ACCEPTED'}
    specifications={}
    for i,selector in enumerate(settings['observables']):
        specifications[f'observable_{i:03d}']={'units':selector['units'],'annotations':{'original_selector':selector},'quantity':selector['name']}
        specifications[f'supported_{i:03d}']={'units':'dimensionless','quantity':'original native target and scalar support'}
    for p,h in pins.items():
        if file_sha256(p)!=h:raise ValueError('Native response target source changed during measurement')
    export_numeric_audit(out/'target-measurement.json',report,{**values,**supports},specifications,algorithm=__file__,context={'epoch':epoch},source_hashes=pins)
    return {'execution_status':'COMPLETE','measurements':{'native_response_target':{'status':status,'observables':len(values),'science_acceptance':'NOT_ACCEPTED'}}}
