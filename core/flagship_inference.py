"""Six registered inference figures from original completed inference products.

Display adapters never fit a posterior, accept a forward model, calibrate a
model or turn failed support into acceptance. Original statistical arrays and
lineage decisions remain attached to every exported numerical bridge.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from .foundation import file_sha256
from .foundation import merge_source_bindings as _merge

VARIANTS={'actual_posterior','posterior_predictive','sbc_ranks','coverage','ood_support','graph_baseline_gain'}


def validate_inference_figure_stage(parameters):
    request=parameters['request'];memory=parameters['memory_gib']
    if type(memory) not in (int,float) or not math.isfinite(memory) or memory<=0:raise ValueError('Positive inference graph stage allocation required')
    if set(request)!={'schema','variant','product'} or request.get('schema')!='native-inference-figure-request-v1' or request['variant'] not in VARIANTS:raise ValueError('Complete inference graph request required')


def inference_figure_runner(parameters,context):
    from .foundation import atomic_json
    validate_inference_figure_stage(parameters)
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Inference graph stage exceeds allocation')
    result=export_inference_figures(parameters['request'],context.generation/'figures',memory_gib=parameters['memory_gib'])
    summary={'variants':sorted({row['variant'] for row in result['figures']}),'figure_count':len(result['figures']),'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(context.generation/'inference-figures.json',summary)
    return {'execution_status':'COMPLETE','measurements':{'inference_figures':summary}}


def inference_product_inputs(reference,*,memory_gib):
    from core.stream_common import read_product, json_safe
    from .data import numeric_audit_source_bindings
    from . import source_path
    if type(memory_gib) not in (int,float) or not math.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Positive finite inference figure allocation required')
    if set(reference)!={'path','sha256'}:raise ValueError('Exact original inference product binding required')
    path=Path(reference['path']).resolve();receipt=path.parent/'stage.json'
    if path.name!='product.json' or file_sha256(path)!=reference['sha256']:raise ValueError('Original inference product changed')
    if not receipt.is_file() and (path.parent/'manifest.json').is_file():
        from .expansion_products import cosmography_generation_inputs
        report,manifest,pins=cosmography_generation_inputs(reference,memory_gib=memory_gib)
        if manifest['request']['kind']!='bao_compressed_posterior' or report.get('schema')!='stream-posterior-v1':raise ValueError('Original sampled cosmography posterior required')
        values=path.parent/'numeric_arrays.json'
        if values.stat().st_size*24>memory_gib*1024**3/2 or pins.get(str(values))!=file_sha256(values):raise ValueError('Original cosmography posterior arrays exceed allocation or escaped publication')
        numeric=json.loads(values.read_text());arrays={key:np.asarray(value) for key,value in numeric['arrays'].items()}
        if any(row['units']!='dimensionless' for row in report['parameter_priors']):raise ValueError('BAO dilation posterior requires original dimensionless parameter units')
        return report,arrays,{'request':manifest['request']},pins
    if max(path.stat().st_size,receipt.stat().st_size)*24>memory_gib*1024**3/3:raise MemoryError('Inference figure metadata exceeds allocation')
    stage=json.loads(receipt.read_text())
    if stage.get('schema')!='stream-stage-receipt-v1' or stage.get('execution_status')!='COMPLETE' or stage['product_sha256']!=reference['sha256']:raise ValueError('Original completed stream inference producer required')
    fingerprint=hashlib.sha256(json.dumps(json_safe({'request':stage['request'],'resources':stage['resources'],
        'code':stage['code_hashes'],'sources':stage['source_hashes']}),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if stage['fingerprint']!=fingerprint:raise ValueError('Original inference stage fingerprint changed')
    report,arrays,pins=read_product(path,memory_gib=memory_gib/2);_merge(pins,numeric_audit_source_bindings(report))
    if report.get('execution_status')!='COMPLETE' or report.get('request')!=stage['request']:raise ValueError('Original inference report differs from its producer stage')
    for sources in [stage['code_hashes'],stage['source_hashes']]:
        for p,h in sources.items():
            if file_sha256(p)!=h:raise ValueError('Original inference producer/input changed')
        _merge(pins,sources)
    owner=source_path('stream.workflow').resolve()
    if stage['code_hashes'].get(str(owner))!=file_sha256(owner):raise ValueError('Original inference workflow source binding required')
    _merge(pins,{str(receipt):file_sha256(receipt)})
    return report,arrays,stage,pins


def _finite(value,shape=None):
    value=np.asarray(value,float)
    if (shape is not None and value.shape!=shape) or not np.isfinite(value).all():raise ValueError('Original inference array shape or finite-value contract differs')
    return value


def inference_figure_arrays(request,*,memory_gib):
    if set(request)!={'schema','variant','product'} or request.get('schema')!='native-inference-figure-request-v1' or request['variant'] not in VARIANTS:raise ValueError('Complete registered inference figure request required')
    report,original,stage,pins=inference_product_inputs(request['product'],memory_gib=memory_gib)
    variant=request['variant'];schema=report.get('schema');kind=stage['request']['kind'];arrays={};units={};panels=[]
    metadata={'original_producer':report,'original_stage':stage['request'],'science_acceptance':'NOT_ACCEPTED',
        'scope':'Original completed inference output; sampling diagnostics, calibration decisions, independent lineage and OOD support retained; export grants no scientific acceptance.'}
    def curve(label,x,y,xunit,yunit,title):
        key='x_'+label;value='y_'+label
        arrays[key]=np.asarray(x);arrays[value]=np.asarray(y);units[key]=xunit;units[value]=yunit
        panels.append({'label':label,'x':key,'y':value,'title':title})
    if variant=='actual_posterior':
        if schema not in {'stream-posterior-v1','halo-shape-population-posterior-v1','stream-sbi-posterior-v1'} or kind not in {'stream_inference','halo_shape_inference','sbi_posterior','bao_compressed_posterior'}:raise ValueError('Actual completed sampler or calibrated SBI posterior required')
        if schema=='stream-sbi-posterior-v1' and report.get('status')!='INFERRED':raise ValueError('No actual posterior exists for unsupported OOD input')
        if schema!='stream-sbi-posterior-v1' and report.get('chain_written') is not True:raise ValueError('Actual sampler chain required; point estimates cannot substitute')
        priors=report['parameter_priors'];names=[row['name'] for row in priors]
        samples=_finite(original['posterior_samples']);weights=_finite(original['posterior_weight'])
        if samples.ndim!=2 or samples.shape[1]!=len(names) or len(samples)<2 or weights.shape!=(len(samples),) or np.any(weights<0) or not np.any(weights>0):raise ValueError('Aligned actual posterior samples and weights required')
        if len(names)>16:raise MemoryError('Explicit inference figure parameter ceiling exceeded')
        # A common positive scale cancels from the weighted CDF. Normalize
        # before summing so finite, very large weights cannot overflow.
        weights=weights/weights.max()
        weights=weights/weights.sum()
        for j,row in enumerate(priors):
            if not row.get('units'):raise ValueError('Original posterior parameter units required')
            order=np.argsort(samples[:,j],kind='stable');curve(str(j),samples[order,j],np.cumsum(weights[order]),row['units'],'dimensionless',row['name']+' posterior weighted CDF')
        metadata['parameter_order']=names;metadata['posterior_scope']=report.get('diagnostics',report.get('calibration_status',report.get('status')))
    elif variant=='posterior_predictive':
        if schema!='stream-posterior-predictive-v1' or kind!='posterior_predictive':raise ValueError('Original posterior predictive replications required')
        discrepancy=_finite(original['predictive_discrepancy'])
        if discrepancy.ndim!=2 or discrepancy.shape[1]!=2 or not len(discrepancy) or np.any(discrepancy<0):raise ValueError('Original observed/replicated discrepancy pairs required')
        predicted=np.asarray(original['predictive_mean']);replicated=np.asarray(original['predictive_replication'])
        if predicted.shape!=replicated.shape or predicted.ndim<2 or predicted.shape[0]!=len(discrepancy):raise ValueError('Original predictive draws and discrepancy rows differ')
        tail=float(np.mean(discrepancy[:,1]>=discrepancy[:,0]))
        if tail!=report['posterior_predictive_tail_probability']:raise ValueError('Original predictive tail probability does not replay')
        curve('replication',discrepancy[:,0],discrepancy[:,1],'dimensionless','dimensionless','Observed and replicated covariance-weighted discrepancies')
        metadata['scope']+=' Plot displays discrepancies; original physical predictive means, replications and selected posterior sample rows remain in the source product.'
    elif variant in {'sbc_ranks','coverage'}:
        if schema!='stream-sbi-calibration-v1' or kind!='sbi_calibration' or report.get('evaluation_status')!='EXECUTED':raise ValueError('Actual frozen-model heldout SBC/coverage evaluation required')
        rows=np.asarray(original['independent_test_local_rows']);covered=np.asarray(original['credible_covered']);ranks=np.asarray(original['sbc_rank']);pit=_finite(original['sbc_pit'])
        if (rows.ndim!=1 or rows.dtype.kind not in 'iu' or len(rows)!=report['independent_groups'] or len(np.unique(rows))!=len(rows) or not len(rows)
                or covered.dtype!=np.dtype(bool) or covered.ndim!=2 or ranks.shape!=covered.shape or ranks.dtype.kind not in 'iu'
                or np.any(rows<0) or np.any(rows>=len(covered)) or pit.shape!=(len(rows),covered.shape[1]) or np.any((pit<0)|(pit>1))):raise ValueError('Original independent heldout row/rank/coverage contracts differ')
        draws=report['calibration_controls']['posterior_draws']
        if np.any(ranks<0) or np.any(ranks>draws):raise ValueError('SBC rank exceeds original finite posterior draw count')
        if variant=='sbc_ranks':
            for j in range(covered.shape[1]):curve(str(j),np.sort(pit[:,j]),np.arange(1,len(rows)+1)/len(rows),'dimensionless','dimensionless','Parameter '+str(j)+' independent randomized rank CDF')
            metadata['rank_scope']='Original randomized finite-draw PIT and original independent component selection; all original integer ranks retained in source.'
        else:
            mean=covered[rows].mean(axis=0);interval=_finite(report['coverage_interval'],(covered.shape[1],2))
            if np.any(interval<0) or np.any(interval>1) or np.any(interval[:,0]>mean) or np.any(interval[:,1]<mean):raise ValueError('Original independent coverage intervals differ from covered trials')
            curve('coverage',np.arange(len(mean)),mean,'dimensionless','dimensionless','Independent heldout coverage; original acceptance '+str(report['calibration_accepted']))
            curve('lower',np.arange(len(mean)),interval[:,0],'dimensionless','dimensionless','Original binomial interval lower bound')
            curve('upper',np.arange(len(mean)),interval[:,1],'dimensionless','dimensionless','Original binomial interval upper bound')
        metadata['calibration_accepted']=report['calibration_accepted'];metadata['scope']+=' Failed calibration remains visible and cannot become deployment acceptance.'
    elif variant=='ood_support':
        if schema!='stream-sbi-posterior-v1' or kind!='sbi_posterior':raise ValueError('Actual SBI inference OOD decision required')
        ood=report['ood'];distance=float(ood['mahalanobis2']);outside=float(ood['outside_fraction'])
        if not np.isfinite([distance,outside]).all() or distance<0 or not 0<=outside<=1 or type(ood['supported']) is not bool:raise ValueError('Original measured OOD metrics required')
        controls=stage['request']['parameters']['inference']['ood']
        actual=distance<=controls['maximum_mahalanobis2'] and outside<=controls['maximum_outside_fraction']
        if actual!=ood['supported']:raise ValueError('Original OOD decision differs from recorded thresholds')
        curve('distance',[0.],[distance],'dimensionless','dimensionless','OOD squared Mahalanobis distance; supported '+str(ood['supported']))
        curve('outside',[0.],[outside],'dimensionless','dimensionless','OOD fraction outside training ranges')
        metadata['ood_thresholds']=controls;metadata['ood_supported']=ood['supported']
    else:
        if schema!='stream-graph-baseline-gain-v1' or kind!='sbi_graph_gain' or report.get('evaluation_status')!='EXECUTED':raise ValueError('Actual paired calibrated graph/baseline evaluation required')
        gain=_finite(original['paired_log_score_gain']);bootstrap=_finite(original['bootstrap_gain'])
        if gain.ndim!=1 or bootstrap.ndim!=1 or not len(gain) or not len(bootstrap) or len(gain)!=report['paired_independent_groups']:raise ValueError('Original paired independent graph/baseline scores required')
        probability=float(np.mean(bootstrap>report['controls']['minimum_mean_gain']))
        if not np.isclose(gain.mean(),report['mean_log_score_gain'],rtol=1e-13,atol=0) or probability!=report['gain_probability'] or (probability>=report['controls']['minimum_gain_probability'])!=report['accepted']:raise ValueError('Original graph/baseline gain decision does not replay')
        curve('paired',np.arange(len(gain)),gain,'dimensionless','dimensionless','Independent paired log-score gain; acceptance '+str(report['accepted']))
        curve('bootstrap',np.sort(bootstrap),np.arange(1,len(bootstrap)+1)/len(bootstrap),'dimensionless','dimensionless','Original paired bootstrap mean-gain CDF')
        metadata['gain_accepted']=report['accepted']
    return arrays,units,metadata,panels,pins


def export_inference_figures(request,destination,*,memory_gib=1.):
    from .data import export_numeric_audit
    from core.research.display import export_scientific_suite
    arrays,units,metadata,descriptions,pins=inference_figure_arrays(request,memory_gib=memory_gib)
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Preserve previous inference figure export')
    out.mkdir(parents=True);_merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    epoch={'meaning':'Original source observation/training/heldout epochs and lineage'};product=out/'inference-measurements.json'
    export_numeric_audit(product,{'schema':'native-inference-figure-product-v1','variant':request['variant'],'epoch':epoch,**metadata},arrays,
        {key:{'units':units[key],'quantity':key,'missing_policy':'Original inference source and rejection/acceptance decisions retained'} for key in arrays},algorithm=__file__,context={'epoch':epoch},source_hashes=pins)
    panels=[]
    for row in descriptions:
        panels.append({'panel_id':row['label'],'title':row['title'],'quality_status':'ORIGINAL_INFERENCE_SCOPE','selection':metadata['scope'],
            'xlabel':row['x']+' ['+units[row['x']]+']','ylabel':row['y']+' ['+units[row['y']]+']','x_units':units[row['x']],'y_units':units[row['y']],
            'series':[{'source':'measurement','kind':'scatter','x':row['x'],'y':row['y'],'label':row['title'],'marker_size':8}]})
    if any(file_sha256(p)!=h for p,h in pins.items()):raise ValueError('Inference figure source changed during export')
    return export_scientific_suite({'sources':{'measurement':str(product)},'dpi':160,'maximum_panels_per_figure':4,'organized_outputs':True,
        'figures':[{'figure_id':'native-'+request['variant'],'family':'inference','variant':request['variant'],'title':request['variant'].replace('_',' '),
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':panels}]},out/'gallery',base=out,memory_gib=memory_gib)
