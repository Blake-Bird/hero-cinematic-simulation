"""A frozen launch interface over the existing scientific workflow engine.

Preparing and inspecting a launch never executes its miniature. Numerical
producers, native admission and checkpoint recovery remain in core.workflow.
"""
from pathlib import Path
from . import source_path
import copy
import json
from .foundation import atomic_json,atomic_text,file_sha256,canonical_digest

OPERATIONS=('test','run','postprocess','render')


def _read(path):
    path=Path(path).expanduser().resolve()
    if path.stat().st_size>64*1024**2:raise MemoryError('Launch metadata exceeds intake bound')
    return json.loads(path.read_text())


def _pin(pins,path,sha=None):
    path=Path(path).expanduser().resolve();actual=file_sha256(path)
    if sha is not None and actual!=sha:raise ValueError('Frozen launch input changed: '+str(path))
    if str(path) in pins and pins[str(path)]!=actual:raise ValueError('Conflicting launch source binding')
    pins[str(path)]=actual


def _resources(value,resources):
    if isinstance(value,list):return [_resources(item,resources) for item in value]
    if isinstance(value,dict):
        if set(value)=={'launch_resource'}:
            if value['launch_resource'] not in resources:raise ValueError('Unknown explicit launch resource')
            return resources[value['launch_resource']]
        return {key:_resources(item,resources) for key,item in value.items()}
    return value


def _build_engines(names,resources,pins):
    from core.native import build_engine, external_native_source
    from core.resources import runtime_paths
    allowed={'class','axiclass','monofonic','music2','rockstar','consistent-trees','velociraptor','disperse'}
    if not isinstance(names,list) or len(set(names))!=len(names) or set(names)-allowed:raise ValueError('Distinct supported native engines required')
    results=[]
    for name in names:
        _,sources=external_native_source(name)
        path=runtime_paths().build_root/'native'/('installed-'+name+'.json')
        installed=_read(path) if path.exists() else {}
        binary=Path(installed.get('executable','/nonexistent-research2-engine'))
        valid=(installed.get('schema')=='research2-native-install-v1' and installed.get('status')=='COMPLETE'
            and installed.get('source_hashes')==sources and binary.is_file()
            and file_sha256(binary)==installed.get('executable_sha256'))
        if not valid:build_engine(name,resources['build_jobs'])
        installed=_read(path);binary=Path(installed['executable'])
        if installed['source_hashes']!=sources or file_sha256(binary)!=installed['executable_sha256']:raise ValueError('Native build/source binding changed')
        _pin(pins,path);_pin(pins,binary,installed['executable_sha256'])
        for source,sha in sources.items():_pin(pins,Path(__file__).resolve().parent/'software'/name/source,sha)
        results.append({'engine':name,'receipt':{'path':str(path),'sha256':file_sha256(path)},'binary':{'path':str(binary),'sha256':file_sha256(binary)}})
    return results


def prepare_launch(profile_path,destination,*,overrides=None,campaign_request=None,downstream_request=None):
    from .workflow import default_registry,execute_workflow,compile_requested_product_dag,_input_references,path_at,positive_number
    profile_path=Path(profile_path).expanduser().resolve();profile=_read(profile_path);out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Preserve previous frozen launch preparation')
    campaign_profile=profile.get('schema')=='flagship-campaign-profile-v1';campaign_profile_path=None
    if campaign_profile:
        if campaign_request is None:raise ValueError('Production/SIDM profile requires --campaign-request from its original source-bound campaign preparer')
        from .flagship_campaign_profile import materialize_campaign_profile
        profile,campaign_profile_path=materialize_campaign_profile(profile,campaign_request,out.with_name(out.name+'.inputs'),overrides=overrides,downstream_request=downstream_request);overrides=None
    elif campaign_request is not None or downstream_request is not None:raise ValueError('--campaign-request applies only to original production/SIDM campaign profiles')
    analysis_profile=profile.get('schema')=='flagship-portable-bao-profile-v1';analysis_request=None
    if analysis_profile:
        from .flagship_analysis_profile import portable_analysis_profile
        profile,analysis_request=portable_analysis_profile(profile,out.with_name(out.name+'.inputs'),overrides=overrides);overrides=None
    portable=profile.get('schema') in {'flagship-portable-connected-request-v1','flagship-portable-local-seeded-request-v1'};portable_request=None
    if portable:
        from .flagship_portable import materialize_portable_profile
        profile,portable_request=materialize_portable_profile(profile,out.with_name(out.name+'.inputs'),overrides=overrides);overrides=None
    connected=profile.get('schema')=='flagship-connected-profile-request-v1'
    if connected:
        from core.flagship_profiles import connected_launch_profile
        profile=connected_launch_profile(profile,request_base=profile_path.parent,overrides=overrides);overrides=None
    required={'schema','name','kind','resources','build_engines','workflows','input_files'}
    if set(profile)!=required or profile['schema']!='flagship-launch-profile-v1' or profile['kind'] not in {'miniature','local-preparation','production','sidm','observational','inference'}:raise ValueError('Complete flagship launch profile required')
    workflows=profile['workflows']
    if not isinstance(workflows,dict) or set(workflows)-set(OPERATIONS) or 'run' not in workflows or (profile['kind'] in {'miniature','local-preparation'} and set(workflows)!=set(OPERATIONS)):
        raise ValueError('Explicit supported operation workflows required; miniature/local profiles require all four')
    resources={**profile['resources'],**(overrides or {})}
    for key in ('memory_gib','disk_gib','mpi_ranks','omp_threads','build_jobs','maximum_stage_count'):
        positive_number(resources[key],key,key in {'mpi_ranks','omp_threads','build_jobs','maximum_stage_count'})
    pins={};_pin(pins,profile_path);_pin(pins,__file__)
    if campaign_profile:
        _pin(pins,Path(__file__).with_name('flagship_campaign_profile.py'));_pin(pins,campaign_profile_path)
    if analysis_profile:
        _pin(pins,Path(__file__).with_name('flagship_analysis_profile.py'));_pin(pins,analysis_request)
    if portable:
        _pin(pins,Path(__file__).with_name('flagship_portable.py'));_pin(pins,portable_request)
        for name,sha in _read(portable_request.parent/'portable-preparation.json')['source_hashes'].items():_pin(pins,name,sha)
    if connected:_pin(pins,source_path('flagship_profiles'))
    for helper in ('foundation.py','native.py','workflow.py','flagship_run_report.py'):_pin(pins,Path(__file__).with_name(helper))
    for helper in sorted((Path(__file__).parent/'research').glob('*.py')):
        _pin(pins,helper)
    _pin(pins,Path(__file__).resolve().parents[1]/'commands.py')
    for reference in profile['input_files']:
        if set(reference)!={'path','sha256'}:raise ValueError('Exact launch asset path/hash required')
        _pin(pins,path_at(profile_path.parent,reference['path']),reference['sha256'])
    out.mkdir(parents=True)
    try:
        engines=_build_engines(profile['build_engines'],resources,pins)
        registry=default_registry();workflows={}
        for operation,template in profile['workflows'].items():
            spec=_resources(copy.deepcopy(template),resources)
            if spec['resources']!=resources:raise ValueError('Workflow must use the complete frozen launch allocation')
            if campaign_profile and profile['kind']=='production':
                # Preparation may describe hardware that has not been allocated.
                # Execution and public dry-run still enforce the original native admission.
                plan=compile_requested_product_dag(spec,registry,request_base=profile_path.parent)
            else:
                plan=execute_workflow(spec,out/('preflight-'+operation),request_base=profile_path.parent,dry_run=True,registry=registry)
            for row in plan['stages']:
                for name,sha in registry.source_hashes(row['kind']).items():_pin(pins,name,sha)
                for value in _input_references(row['parameters']):_pin(pins,path_at(profile_path.parent,value))
                for reference in row.get('input_files',[]):_pin(pins,path_at(profile_path.parent,reference['path']),reference['sha256'])
            path=out/(operation+'.workflow.json');atomic_json(path,spec)
            workflows[operation]={'path':str(path),'sha256':file_sha256(path),'stage_count':len(plan['stages'])}
        for path,sha in pins.items():_pin({},path,sha)
        record={'schema':'prepared-flagship-launch-v1','execution_status':'PREPARED','name':profile['name'],'kind':profile['kind'],
            'resources':resources,'request_base':str(profile_path.parent),'source_hashes':pins,'workflows':workflows,'native_engines':engines,
            'profile_sha256':file_sha256(profile_path),'science_acceptance':'NOT_ACCEPTED','miniature_execution':'NOT_EXECUTED','connected_pipeline':('production' if campaign_profile and downstream_request is not None else connected),
            'preflight_scope':'STATIC_DAG_ONLY_RUNTIME_ALLOCATION_ADMISSION_REQUIRED' if campaign_profile and profile['kind']=='production' else 'ORIGINAL_WORKFLOW_DRY_RUN'}
        atomic_json(out/'launch.json',record);atomic_text(out/'launch.sha256',file_sha256(out/'launch.json')+'\n')
        return record
    except BaseException as error:
        atomic_json(out/'preparation-failure.json',{'execution_status':'FAILED','type':type(error).__name__,'message':str(error),'science_acceptance':'NOT_ACCEPTED'})
        raise


def load_launch(path):
    path=Path(path).expanduser().resolve();path=path/'launch.json' if path.is_dir() else path
    record=_read(path)
    if record.get('schema')!='prepared-flagship-launch-v1' or record.get('execution_status')!='PREPARED':raise ValueError('Completed frozen launch preparation required')
    if path.with_suffix('.sha256').read_text().strip()!=file_sha256(path):raise ValueError('Frozen launch manifest seal changed')
    for name,sha in record['source_hashes'].items():_pin({},name,sha)
    for reference in record['workflows'].values():_pin({},reference['path'],reference['sha256'])
    return path,record


def execute_launch(prepared,destination,operation,*,resume=False,dry_run=False,source_output=None):
    from .workflow import execute_workflow
    if operation not in OPERATIONS:raise ValueError('Supported flagship operation required')
    path,launch=load_launch(prepared);out=Path(destination).expanduser().resolve()
    if operation=='test' and launch['kind']!='miniature':raise ValueError('test requires the explicitly bounded miniature profile')
    if operation not in launch['workflows']:raise ValueError('Selected branch supports '+', '.join(launch['workflows'])+' plus resume/status; its graph exports are included in run')
    reference=launch['workflows'][operation];spec=_read(reference['path'])
    binding_path=out/'launch-binding.json'
    previous=_read(binding_path) if binding_path.exists() else None
    source=None;used_source=False
    if source_output is not None:
        source_path=Path(source_output).expanduser().resolve()/'workflow.json'
        source={'path':str(source_path),'sha256':file_sha256(source_path)}
    elif resume and previous is not None:source=previous.get('source_workflow')
    def bind_source(value):
        nonlocal used_source
        if isinstance(value,list):return [bind_source(item) for item in value]
        if isinstance(value,dict):
            if value=={'launch_source_workflow':True}:
                if operation not in {'postprocess','render'} or source is None:raise ValueError('Retained postprocess/render requires --source-output from the original completed run')
                _pin({},source['path'],source['sha256']);used_source=True;return copy.deepcopy(source)
            return {key:bind_source(item) for key,item in value.items()}
        return value
    spec=bind_source(spec)
    if source_output is not None and not used_source:raise ValueError('This frozen operation does not consume a retained source output')
    binding={'schema':'flagship-operation-binding-v1','launch_path':str(path),'launch_sha256':file_sha256(path),
        'operation':operation,'workflow_sha256':reference['sha256'],'resources':launch['resources']}
    if used_source:binding.update(source_workflow=source,resolved_workflow_sha256=canonical_digest(spec))
    if binding_path.exists():
        if _read(binding_path)!=binding:raise ValueError('Resume must preserve the exact frozen launch, operation and allocation')
        if not resume and not dry_run:raise FileExistsError('Existing launch requires resume')
    elif resume and (out/'workflow.json').exists():raise ValueError('Cannot adopt an unrelated workflow as this launch')
    if not dry_run:
        out.mkdir(parents=True,exist_ok=True);atomic_json(binding_path,binding)
    try:
        result=execute_workflow(spec,out,request_base=launch['request_base'],resume=resume,dry_run=dry_run)
    except BaseException:
        if not dry_run and (out/'workflow.json').exists():
            try:
                from .flagship_run_report import publish_run_report
                publish_run_report(out,resources=launch['resources'],operation=operation,connected=launch.get('connected_pipeline',False))
            except Exception as error:
                atomic_json(out/'run-report-error.json',{'execution_status':'FAILED','type':type(error).__name__,'message':str(error),
                    'scope':'Report failed; original workflow exception and receipts remain authoritative'})
        raise
    if not dry_run:
        from .flagship_run_report import publish_run_report
        report=publish_run_report(out,resources=launch['resources'],operation=operation,connected=launch.get('connected_pipeline',False))
        result={**result,'scheduler_execution_status':result['execution_status'],'execution_status':report['execution_status'],
            'run_report':{'path':str(out/'run-report.json'),'sha256':file_sha256(out/'run-report.json')}}
    return result


def launch_status(destination):
    """Report the original engine's state; never manufacture a success receipt."""
    out=Path(destination).expanduser().resolve();binding_path=out/'launch-binding.json';state_path=out/'workflow.json'
    if not binding_path.is_file():raise FileNotFoundError('No flagship launch binding at requested output')
    binding=_read(binding_path);state=_read(state_path) if state_path.is_file() else {}
    stages=[]
    for sid,reference in state.get('stages',{}).items():
        stages.append({'stage_id':sid,'generation':reference['generation'],'receipt':str(Path(reference['generation'])/'stage-receipt.json')})
    attempts=[]
    for receipt in (out/'generations').glob('*/*/stage-receipt.json'):
        record=_read(receipt)
        if record.get('execution_status') in {'RUNNING','FAILED','INTERRUPTED'}:
            invocation=receipt.parent/'invocation.json';native=_read(invocation) if invocation.is_file() else {}
            attempts.append({'stage_id':record.get('stage_id'),'recorded_status':record.get('execution_status'),
                'generation':str(receipt.parent),'started_unix':record.get('started_unix'),
                'native_invocation_status':native.get('status'),'native_pid':native.get('pid'),
                'native_returncode':native.get('returncode'),
                'liveness':'RECORDED_STATE_ONLY; process liveness has not been established'})
    report_path=out/'run-report.json';report=None
    if report_path.is_file() and report_path.with_name(report_path.name+'.sha256').is_file():
        if file_sha256(report_path)!=report_path.with_name(report_path.name+'.sha256').read_text().strip():raise ValueError('Run report seal changed')
        candidate=_read(report_path)
        if state_path.exists() and candidate['workflow']['sha256']==file_sha256(state_path):report=candidate
    return {'schema':'flagship-launch-status-v1','operation':binding['operation'],'execution_status':report['execution_status'] if report is not None else state.get('execution_status','BOUND_NOT_STARTED'),
        'completed_stage_records':len(stages),'stages':stages,'failed_stage':state.get('failed_stage'),
        'incomplete_attempt_records':sorted(attempts,key=lambda row:row.get('started_unix') or 0),
        'failure':state.get('failure'),'remaining_requirements':state.get('remaining_requirements',[]),
        'scientific_exclusions':state.get('scientific_exclusions',[]),'workflow_receipt':str(state_path),
        'run_report':None if report is None else {'path':str(report_path),'sha256':file_sha256(report_path),'missing_connections':report['missing_connections']},
        'science_acceptance':state.get('scientific_acceptance','UNRESOLVED')}


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Prepare and operate a frozen flagship workflow with the original scientific engine')
    sub=parser.add_subparsers(dest='operation',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--request',type=Path,required=True,help='Your explicit campaign settings; no bundled run preset is selected');prep.add_argument('--output',type=Path,required=True);prep.add_argument('--campaign-request',type=Path);prep.add_argument('--downstream-request',type=Path)
    for key,typ in [('memory_gib',float),('disk_gib',float),('mpi_ranks',int),('omp_threads',int),('build_jobs',int),('maximum_stage_count',int)]:prep.add_argument('--'+key.replace('_','-'),type=typ)
    for operation in (*OPERATIONS,'resume'):
        command=sub.add_parser(operation);command.add_argument('--prepared',type=Path,required=True);command.add_argument('--output',type=Path,required=True);command.add_argument('--dry-run',action='store_true');command.add_argument('--source-output',type=Path)
    status=sub.add_parser('status');status.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.operation=='prepare':result=prepare_launch(args.request,args.output,campaign_request=args.campaign_request,downstream_request=args.downstream_request,overrides={key:getattr(args,key) for key in ('memory_gib','disk_gib','mpi_ranks','omp_threads','build_jobs','maximum_stage_count') if getattr(args,key) is not None})
    elif args.operation=='status':result=launch_status(args.output)
    else:
        operation=_read(args.output/'launch-binding.json')['operation'] if args.operation=='resume' else args.operation
        result=execute_launch(args.prepared,args.output,operation,resume=args.operation=='resume',dry_run=args.dry_run,source_output=args.source_output)
    print(json.dumps(result,indent=2))
    return 0 if args.operation in {'prepare','status'} or args.dry_run or result.get('execution_status')=='COMPLETE' else 1
