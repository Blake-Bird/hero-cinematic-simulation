"""Sequential saved-run analysis and explicit pairwise comparison publication."""
from __future__ import annotations
import datetime
import json
from pathlib import Path


def run_batch_analysis(runs,products,destination,*,reference=0,find_halos=False,allow_incomplete=False,compare=True):
    from commands import process_run,parse_products
    from core.foundation import atomic_json
    from core.batch_results import write_batch_index
    from core.visualization import make_comparison_maps
    out=Path(destination).expanduser().resolve()
    if out.exists():raise FileExistsError('Choose a fresh batch analysis directory')
    if not isinstance(runs,(list,tuple)) or not runs:raise ValueError('Select at least one saved run')
    roots=[Path(x).expanduser().resolve() for x in runs]
    if len(set(roots))!=len(roots):raise ValueError('Each selected run must be unique')
    if type(reference) is not int or not 0<=reference<len(roots):raise ValueError('Reference must identify a selected run')
    if any(not (r/'config.yaml').is_file() for r in roots):raise FileNotFoundError('Every selected run needs config.yaml')
    selected=parse_products(products)
    if any(x in selected for x in ('stellar_components','stellar_histories','zoom_contamination','inference_diagnostics','radial_shapes')):
        raise ValueError('Specialized per-run requests are not supported in batch mode; run them individually')
    out.mkdir(parents=True);report={'schema':'batch-analysis-v1','status':'RUNNING','science_acceptance':'NOT_ACCEPTED',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':[{'run':str(r),'status':'PENDING'} for r in roots],
        'requested_products':selected,'reference':reference,'comparisons':[],'suite_comparisons':{},'comparison_policy':'Reference-to-each-other plus all-run scientific overlays for suites of three or more; saved HMF, power spectrum, correlation curves and projected maps when supported. Phase matching is not inferred.'}
    def save():
        report['results_index']=str(out/'index.html')
        atomic_json(out/'batch_report.json',report)
        try:
            write_batch_index(out,report)
            report.pop('index_error',None)
        except Exception as error:
            report['index_error']={'type':type(error).__name__,'message':str(error)}
        atomic_json(out/'batch_report.json',report)
    save()
    try:
        for i,root in enumerate(roots):
            row=report['runs'][i];row['status']='RUNNING';save()
            existing_records=set((root/'results').glob('workflow_*Z.json'))
            try:
                import yaml
                config=yaml.safe_load((root/'config.yaml').read_text())
                simulation=config.get('simulation',{});cosmology=config.get('cosmology',{})
                row['settings']={
                    'model':cosmology.get('model'),'H0':cosmology.get('H0'),
                    'box_mpc_h':simulation.get('box_mpc_h'),
                    'particles_per_dimension':simulation.get('particles_per_dimension'),
                    'physics':simulation.get('physics'),
                    'requested_redshifts':simulation.get('snapshot_redshift_list',[])}
                result=process_run(root,selected,find_halos=find_halos,allow_incomplete=allow_incomplete)
                row.update({'status':result.get('status','UNKNOWN'),'product_status':result.get('product_status','UNKNOWN'),
                    'products':result.get('products',[]),'artifacts':result.get('artifacts',{}),'unavailable':result.get('unavailable',{})})
                if result.get('workflow_record'):row['workflow_record']=result['workflow_record']
            except Exception as error:
                row.update({'status':'FAILED','error':{'type':type(error).__name__,'message':str(error)}})
                records=set((root/'results').glob('workflow_*Z.json'))-existing_records
                if len(records)==1:
                    record=records.pop();row['workflow_record']=str(record)
                    try:
                        partial=json.loads(record.read_text())
                        if Path(partial['run']).resolve()!=root:raise ValueError('Workflow record belongs to another run')
                        for key in ('products','product_status','artifacts','unavailable'):
                            if key in partial:row[key]=partial[key]
                    except (OSError,ValueError,KeyError) as record_error:
                        row['record_error']=str(record_error)
            save()
        if compare and len(roots)>1:
            for i,root in enumerate(roots):
                if i==reference:continue
                row={'reference_run':str(roots[reference]),'other_run':str(root),'status':'RUNNING'};report['comparisons'].append(row);save()
                if any(report['runs'][j]['status'] not in ('COMPLETE','INCOMPLETE') for j in (reference,i)):
                    row.update(status='SKIPPED',reason='At least one individual analysis failed or was interrupted');save();continue
                row['products']={}
                left=report['runs'][reference].get('artifacts',{}).get('analysis');right=report['runs'][i].get('artifacts',{}).get('analysis')
                common=set(report['runs'][reference].get('products',[])) & set(report['runs'][i].get('products',[]))
                for product in ('hmf','pk','correlation'):
                    if product not in common:continue
                    try:
                        from core.batch_comparison import compare_selected_hmf,compare_selected_curves
                        if not left or not right:raise ValueError('No selected scientific analysis bundle for one or both runs')
                        dest=out/(product+'_comparisons')/('pair_%03d'%i)
                        figure=(compare_selected_hmf(roots[reference],left,root,right,dest) if product=='hmf' else
                                compare_selected_curves(roots[reference],left,root,right,dest,product=product))
                        row['products'][product]={'status':'COMPLETE','figure':str(figure)}
                    except Exception as error:row['products'][product]={'status':'UNAVAILABLE','reason':str(error)}
                try:
                    result=make_comparison_maps(roots[reference],root,outdir=out/'comparison_maps',fields=['dm'],projections=['xy'])
                    row['products']['maps']={'status':'COMPLETE','map_bundle':str(result)}
                except Exception as error:row['products']['maps']={'status':'UNAVAILABLE','reason':str(error),'error_type':type(error).__name__}
                row['status']='COMPLETE' if all(p['status']=='COMPLETE' for p in row['products'].values()) else 'INCOMPLETE'
                row['scope']='Descriptive comparisons; no phase-matching, convergence or causal claim.'
                save()
        if compare and len(roots)>2:
            from core.suite_comparison import compare_suite
            present=set().union(*(set(row.get('products',[])) for row in report['runs']))
            for product in ('hmf','pk','correlation'):
                if product not in present:continue
                row={'status':'RUNNING'};report['suite_comparisons'][product]=row;save()
                try:
                    bundles=[item.get('artifacts',{}).get('analysis') for item in report['runs']]
                    if any(not bundle for bundle in bundles):
                        raise ValueError('At least one selected run has no scientific analysis bundle; all-run overlays require every run')
                    figure=compare_suite(roots,bundles,out/'suite_comparisons'/product,product=product)
                    row.update(status='COMPLETE',figure=str(figure),pdf=str(figure.with_suffix('.pdf')),
                               data_csv=str(figure.parent/'curves.csv'),record=str(figure.parent/'comparison.json'))
                except Exception as error:
                    row.update(status='UNAVAILABLE',reason=str(error))
                save()
        failed=any(x['status']!='COMPLETE' for x in report['runs'])
        incomplete=any(x['status']!='COMPLETE' for x in report['comparisons'])
        incomplete=incomplete or any(x['status']!='COMPLETE' for x in report['suite_comparisons'].values())
        report['status']='INCOMPLETE' if failed or incomplete else 'COMPLETE'
        save()
        if report.get('index_error'):
            report['status']='INCOMPLETE';atomic_json(out/'batch_report.json',report)
        return report
    except BaseException as error:
        for row in report['runs']+report['comparisons']+list(report['suite_comparisons'].values()):
            if row['status']=='RUNNING':row['status']='INTERRUPTED'
        report['status']='INTERRUPTED';report['interruption']={'type':type(error).__name__,'message':str(error)};save();raise
