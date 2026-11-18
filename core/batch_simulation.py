"""Collect multiple interactive run configurations, then execute sequentially."""
from __future__ import annotations
import datetime
import json
import sys
import uuid
from pathlib import Path


def configure_and_run_queue(count,paths,*,linear_output=None,binary=None):
    from core import linear
    from core.native import native_engine_directory
    from core.foundation import atomic_json
    from core.simulation_queue_index import write_queue_index
    from commands import build_config,run_gadget_main
    if type(count) is not int or count<2:raise ValueError('At least two runs are required for a simulation queue')
    queue=paths.state_root/'simulation_queues'/('queue_'+uuid.uuid4().hex);queue.mkdir(parents=True)
    report={'schema':'simulation-queue-v1','status':'CONFIGURING','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'runs':[],'science_acceptance':'NOT_ACCEPTED','scope':'Sequential execution; completed runs are never replaced.'}
    def save():
        report['results_index']=str(queue/'index.html')
        atomic_json(queue/'queue_report.json',report)
        write_queue_index(queue,report)
    save();seen=set()
    try:
        for i in range(count):
            print('\n=== Configure simulation %d of %d ==='%(i+1,count),flush=True)
            cfg=linear.prompt_config();linear.validate(cfg)
            solver=Path(binary).expanduser().resolve() if binary else native_engine_directory('axiclass' if cfg['model']=='ede' else 'class')/'class'
            if not solver.is_file():raise FileNotFoundError('Compiled solver missing: '+str(solver))
            output_root=Path(linear_output).expanduser().resolve() if linear_output else paths.runs_root/'linear'
            # The spectrum is generated during configuration because the normal
            # particle questionnaire derives its starting redshift from it.
            print('Preparing this run’s linear spectrum before particle settings.',flush=True)
            spectrum=linear.run(cfg,output_root,solver)
            from core.setup_walkthrough import collect_simulation
            answers=collect_simulation(spectrum/'spectrum.json', queue/'setup.json', outputs=True)
            _,run_dir=build_config(answers);run_dir=Path(run_dir).resolve()
            if run_dir in seen or run_dir.exists():raise FileExistsError('Each queued simulation needs a distinct unused run directory: '+str(run_dir))
            seen.add(run_dir);answers_path=queue/('answers_%03d.json'%(i+1));atomic_json(answers_path,answers)
            report['runs'].append({'run':str(run_dir),'linear_bundle':str(spectrum),'answers':str(answers_path),'status':'PENDING'});save()
        print('\nQueued simulations:')
        for i,row in enumerate(report['runs'],1):print('  %d. %s'%(i,row['run']))
        report['status']='RUNNING';save()
        previous=sys.argv
        try:
            for row in report['runs']:
                row['status']='RUNNING';save();print('\nStarting '+row['run'],flush=True)
                try:
                    sys.argv=['commands.py gadget','--answers',row['answers'],'--run-dir',row['run']]
                    result=run_gadget_main()
                    if result not in (None,0):raise RuntimeError('Simulation command returned '+str(result))
                    row['status']='COMPLETE'
                    row['simulation_status']='COMPLETE'
                    records=sorted((Path(row['run'])/'results').glob('workflow_*Z.json'))
                    if not records:
                        row.update(status='INCOMPLETE',analysis_reason='No analysis workflow record was produced.')
                    else:
                        row['workflow_record']=str(records[-1])
                        workflow=json.loads(records[-1].read_text())
                        row['analysis_status']=workflow.get('status','UNKNOWN')
                        row['artifacts']=workflow.get('artifacts',{})
                        row['unavailable']=workflow.get('unavailable',{})
                        if workflow.get('status')!='COMPLETE':
                            row.update(status='INCOMPLETE',analysis_reason='Analysis status: '+str(workflow.get('status','UNKNOWN')))
                except Exception as error:
                    row.update(status='FAILED',error={'type':type(error).__name__,'message':str(error)})
                    records=sorted((Path(row['run'])/'results').glob('workflow_*Z.json'))
                    if records:
                        # Post-processing creates this record only after the
                        # simulation stage returns successfully.
                        row['simulation_status']='COMPLETE'
                        row['workflow_record']=str(records[-1])
                        try:
                            workflow=json.loads(records[-1].read_text())
                            row['analysis_status']=workflow.get('status','UNKNOWN')
                            row['artifacts']=workflow.get('artifacts',{})
                            row['unavailable']=workflow.get('unavailable',{})
                        except (OSError,ValueError) as record_error:
                            row['analysis_status']='UNKNOWN'
                            row['analysis_reason']='Could not read analysis record: '+str(record_error)
                save()
        finally:sys.argv=previous
        report['status']='COMPLETE' if all(row['status']=='COMPLETE' for row in report['runs']) else 'INCOMPLETE';save()
        print('Simulation queue report: '+str(queue/'queue_report.json'),flush=True)
        print('Open queue results: '+report['results_index'],flush=True)
        return 0 if report['status']=='COMPLETE' else 1
    except BaseException as error:
        report['status']='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED'
        for row in report['runs']:
            if row['status']=='RUNNING':row['status']=report['status']
        report['failure']={'type':type(error).__name__,'message':str(error)};save();raise
