"""Connect original cosmography/stream analysis campaigns to registered figures."""
from pathlib import Path
import copy
import json
import math
from .foundation import atomic_json,file_sha256
from .foundation import merge_source_bindings as _merge

ROUTES={'distance_ladder':({'distance_ladder'},'hubble_figures'),
    'time_delay_posteriors':({'time_delay'},'hubble_figures'),
    'hubble_method_comparison':({'hubble_comparison'},'hubble_figures'),
    'bao_cmb_background':({'bao_compressed','bao_compressed_posterior'},'hubble_figures'),
    'actual_posterior':({'stream_inference','halo_shape_inference','sbi_posterior','bao_compressed_posterior'},'inference_figures'),
    'posterior_predictive':({'posterior_predictive'},'inference_figures'),
    'sbc_ranks':({'sbi_calibration'},'inference_figures'),'coverage':({'sbi_calibration'},'inference_figures'),
    'ood_support':({'sbi_posterior'},'inference_figures'),'graph_baseline_gain':({'sbi_graph_gain'},'inference_figures')}


def prepare_analysis_figure_campaign(request,destination):
    from .flagship_profiles import analysis_campaign_bindings
    from .workflow import default_registry,compile_requested_product_dag,safe_id
    required={'schema','campaign','resources','wall_seconds','figures'}
    if set(request)!=required or request['schema']!='analysis-figure-campaign-request-v1':raise ValueError('Complete original analysis/figure campaign request required')
    resources=request['resources'];memory=resources['memory_gib'];reference=request['campaign'];source=Path(reference['path']).resolve()
    if source.stat().st_size*24>memory*1024**3/8 or file_sha256(source)!=reference['sha256']:raise ValueError('Original analysis campaign changed or exceeds allocation')
    campaign=json.loads(source.read_text());is_cosmography=campaign.get('schema')=='cosmography-campaign-v1'
    branch='inference' if not is_cosmography or any(s['kind']=='bao_compressed_posterior' for s in campaign['stages']) else 'observational'
    campaign,pins=analysis_campaign_bindings(branch,reference,resources)
    rows=request['figures'];wall=request['wall_seconds']
    if type(wall) not in (int,float) or not math.isfinite(wall) or wall<=0:raise ValueError('Positive bounded original analysis wall allocation required')
    if not isinstance(rows,list) or not rows or len(rows)+1>resources['maximum_stage_count']:raise ValueError('Bounded nonempty graph route selections required')
    producers={s['stage_id']:s for s in campaign['stages']};seen=set();out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Preserve previous analysis figure preparation')
    stages=[{'stage_id':'analysis','kind':'public_command','parameters':{'command':'cosmography' if is_cosmography else 'streams',
        'arguments':[{'input':str(source)},'--output','{generation}/analysis'],'wall_seconds':wall},
        'input_files':[{'path':p,'sha256':h} for p,h in pins.items()],'outputs':{'analysis':'analysis'}}]
    products=[]
    for selection in rows:
        if set(selection)!={'variant','producer_stage'} or selection['variant'] not in ROUTES:raise ValueError('Explicit registered graph/producer selector required')
        variant=selection['variant'];producer=producers[selection['producer_stage']];allowed,kind=ROUTES[variant]
        if producer['kind'] not in allowed or variant in seen:raise ValueError('Graph variant needs its distinct original scientific producer')
        seen.add(variant);sid=safe_id('figure_'+variant);artifact='analysis/'+producer['stage_id']+'/product.json'
        figure={'schema':'native-hubble-figure-request-v1' if kind=='hubble_figures' else 'native-inference-figure-request-v1','variant':variant}
        if variant=='bao_cmb_background':
            bound={}
            for key in ('observations','linear_bundle','drag_execution','drag_log','drag_invocation'):
                value=producer['inputs'][key]
                if isinstance(value,str):
                    path=Path(value);path=(source.parent/path).resolve() if not path.is_absolute() else path.resolve()
                    bound[key]={'path':str(path),'sha256':file_sha256(path)}
                elif isinstance(value,dict) and set(value)<={'stage','artifact'}:
                    name='analysis/'+value['stage']+'/'+value.get('artifact','product.json')
                    bound[key]={'path':{'stage':'analysis','artifact':name},'sha256':{'stage':'analysis','artifact':name,'load':'sha256'}}
                else:raise ValueError('BAO figure must retain original actual likelihood source paths')
            figure['bao_request']={'schema':'conditional-compressed-bao-request-v1',**bound}
        else:figure['product']={'path':{'stage':'analysis','artifact':artifact},'sha256':{'stage':'analysis','artifact':artifact,'load':'sha256'}}
        stages.append({'stage_id':sid,'kind':kind,'parameters':{'request':figure,'memory_gib':memory},'depends_on':['analysis'],'outputs':{variant:'figures'}})
        products.append(variant)
    workflow={'schema':'research-workflow-v1','resources':copy.deepcopy(resources),'physics_spec':{},'capabilities':[],
        'followup_limits':{'maximum_children':0,'maximum_depth':0},'requested_products':products,'stages':stages}
    registry=default_registry();compile_requested_product_dag(workflow,registry,request_base=source.parent)
    for row in stages:_merge(pins,registry.source_hashes(row['kind']))
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    for p,h in pins.items():
        if file_sha256(p)!=h:raise ValueError('Analysis figure source changed during preparation')
    out.mkdir(parents=True);atomic_json(out/'workflow.json',workflow)
    result={'schema':'analysis-figure-campaign-prepared-v1','status':'RECIPES_PREPARED','variants':products,'stage_count':len(stages),
        'workflow':{'path':str(out/'workflow.json'),'sha256':file_sha256(out/'workflow.json')},'source_hashes':pins,
        'analysis_execution':'NOT_EXECUTED','science_acceptance':'NOT_ACCEPTED',
        'scope':'Original selected cosmography or stream inference campaign, then original registered graph adapters. Missing observational inputs/accepted forward domains/calibration remain original failures; preparation never invents them.'}
    atomic_json(out/'manifest.json',result);return result
