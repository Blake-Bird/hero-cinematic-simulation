"""Domain-gated SBI generation, frozen models, calibration and stage dispatch.

Physical forward acceptance comes first. Parent/lineage components stay in
one train, validation or held-out split; held-out truth is evaluation-only.
Training success does not authorize deployment. Torch and upstream SBI load
inside selected operations, while stage names remain available without them.
"""
from __future__ import annotations

from pathlib import Path
import copy
import hashlib
import importlib
import importlib.metadata
import json
import numpy as np
from scipy.stats import binomtest, kstest
from core.stream_common import finite, positive, integer, seeded_rng, json_safe, digest
from core.stream.inference import ParameterSpace, estimator_only, observation_detection_probability
from core.foundation import atomic_json
from core.native import native_run_lock

_CORE_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _CORE_ROOT.parent


PIN_FILE=_PROJECT_ROOT/'configs/sbi.upstream.json'


def pinned_sbi():
    """Bind the real SBI package from a pinned VCS or external bootstrap wheel."""
    from core.native import verified_python_backend
    records=json.loads(PIN_FILE.read_text())['repositories']
    matches=[record for record in records if record['name']=='sbi']
    if len(matches)!=1:
        raise ValueError('Exactly one pinned SBI backend record required')
    record=matches[0]
    root=_PROJECT_ROOT/record['directory']
    sources={str(PIN_FILE):digest(PIN_FILE)}
    for relative,expected in record['source_sha256'].items():
        path=root/relative
        if digest(path)!=expected:
            raise ValueError('Bundled upstream SBI source changed: '+relative)
        sources[str(path)]=expected
    try:
        distribution=importlib.metadata.distribution('sbi')
    except importlib.metadata.PackageNotFoundError as error:
        raise ImportError('Run bootstrap.py --install --profile sbi before requesting SBI') from error
    binding=verified_python_backend('sbi',record,distribution=distribution)
    module=importlib.import_module('sbi')
    if str(Path(module.__file__).resolve()) not in binding['installed_assets']:
        raise ValueError('Loaded SBI is outside its verified distribution; start a clean process')
    sources.update(binding['source_hashes'])
    # Serialized network classes and prior helpers must match the frozen model,
    # as well as upstream package bytes. This does not execute those modules.
    for path in (Path(__file__).resolve(),_CORE_ROOT/'stream/encoders.py',_CORE_ROOT/'stream/inference.py'):
        sources[str(path)]=digest(path)
    return {'revision':record['revision'],'source_url':record['url'],'source_hashes':sources,
        'license_files':record['license_files'],'installed_origin':binding['origin']}


def forward_signature(definitions,bindings,code_hashes):
    payload=json.dumps(json_safe({'definitions':definitions,'bindings':bindings,'code_hashes':code_hashes}),sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(payload).hexdigest()


def require_forward_model_accepted_for(receipt,domain,*,expected_signature,priors):
    if receipt['schema']!='accepted-forward-domain-v1' or receipt['science_acceptance']!='ACCEPTED':
        raise ValueError('SBI requires an executed, accepted forward-model domain receipt')
    if receipt['forward_signature']!=expected_signature:
        raise ValueError('Forward receipt belongs to different code/model/input assets')
    if domain['domain_id']!=receipt['domain_id']:
        raise ValueError('Forward-domain identity mismatch')
    required={'force','time_integration','progenitor','observation','selection','model_discrepancy'}
    controls=receipt['capability_receipts']
    if not required.issubset(controls) or any(controls[name]['status']!='PASS' for name in required):
        raise ValueError('Forward acceptance lacks required physical/observational controls')
    for control in controls.values():
        for path,expected in control.get('source_hashes',{}).items():
            if digest(path)!=expected:
                raise ValueError('Forward-domain control source changed')
    for prior in priors:
        support=receipt['parameter_bounds'][prior['name']]
        lo,hi=prior['bounds']
        if lo<support[0] or hi>support[1]:
            raise ValueError('Prior extends beyond accepted forward domain: '+prior['name'])
    if receipt['observable_contract']!=domain['observable_contract']:
        raise ValueError('Observable support differs from accepted domain')
    return receipt


def lineage_disjoint_splits(records,*,seed,fractions,minimum_groups_per_split):
    """Connected components of shared parent *or* shared lineage stay together."""
    fraction=finite(fractions,(3,))
    if np.any(fraction<=0) or not np.isclose(fraction.sum(),1):
        raise ValueError('Positive train/validation/test fractions summing to one required')
    n=len(records)
    parent=list(range(n))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    lookup={}
    for i,record in enumerate(records):
        if not record['parent_realization'] or not record['lineage_ids']:
            raise ValueError('Recorded parent realization and full lineage group required')
        labels=[('parent',record['parent_realization']),*[('lineage',q) for q in record['lineage_ids']]]
        for label in labels:
            if label in lookup:
                parent[root(i)]=root(lookup[label])
            else: lookup[label]=i
    components={}
    for i in range(n):
        components.setdefault(root(i),[]).append(i)
    groups=list(components.values())
    rng=seeded_rng(seed,'lineage-split')
    rng.shuffle(groups)
    minimum=integer(minimum_groups_per_split,'minimum_groups_per_split')
    if len(groups)<3*minimum:
        raise ValueError('Insufficient independent parent/lineage components for disjoint splits')
    counts=np.full(3,minimum,int)
    remaining=len(groups)-counts.sum()
    if remaining:
        extra=rng.multinomial(remaining,fraction)
        counts+=extra
    splits={}
    offset=0
    component_ids={}
    for name,count in zip(('train','validation','test'),counts):
        selected=groups[offset:offset+count]
        offset+=count
        splits[name]=np.array([i for group in selected for i in group],int)
        component_ids[name]=[min(group) for group in selected]
    return splits,{'method':'union-find connected parent-or-lineage grouping before seeded group split',
        'fractions':fraction,'seed':seed,'component_groups':component_ids,'independent_component_count':len(groups)}


def generate_training_experiments(designs,simulators,encoder,request,*,space,privileged_truth_destination):
    """Sample recorded priors, evolve real fields, then apply observation noise.

    Selection is conditioned per theta using bounded rejection. This targets
    p(theta|selected observations) under the declared design prior, rather
    than silently changing the prior by dropping undetected theta draws.
    """
    n=integer(request['experiments'],'experiments',3)
    rng=seeded_rng(request['seed'],'sbi-design')
    if n*encoder.output_size*8>positive(request['memory_gib'],'memory_gib')*1024**3:
        raise MemoryError('SBI training observations exceed memory budget')
    theta=[]
    features=[]
    records=[]
    for i in range(n):
        design=designs[i%len(designs)]
        simulator=simulators[design['design_id']]
        parameters=space.transform(rng.uniform(1e-10,1-1e-10,space.ndim))
        prediction=simulator(space.mapping(parameters))
        value=finite(prediction['values'])
        count,dimensions=value.shape
        covariance=finite(design['observation_covariance'],(value.size,value.size))+finite(prediction['model_error_covariance'],(value.size,value.size))
        if not np.allclose(covariance,covariance.T):
            raise ValueError('SBI observation covariance not symmetric')
        eigen,vectors=np.linalg.eigh(covariance)
        if eigen.min()<-1e-12:
            raise ValueError('SBI observation covariance not positive semidefinite')
        root=vectors*np.sqrt(np.maximum(eigen,0))[None,:]
        accepted=False
        for attempt in range(integer(request['maximum_selection_attempts'],'maximum_selection_attempts')):
            noisy=(value.ravel()+root@rng.normal(size=value.size)).reshape(value.shape)
            detection=observation_detection_probability(noisy,design['detection_rule'])
            if np.all(rng.random(count)<detection):
                accepted=True
                break
        if not accepted:
            raise RuntimeError('Conditional observation selection exhausted attempt budget; do not drop prior designs')
        missing_probability=finite(design['missing_probability'],value.shape)
        if np.any((missing_probability<0)|(missing_probability>1)):
            raise ValueError('Invalid declared missingness probabilities')
        missing=rng.random(value.shape)<missing_probability
        noisy[missing]=np.nan
        observation={'columns':encoder.columns,'values':noisy,'missing':missing,'covariance':covariance}
        features.append(encoder.encode(observation))
        theta.append(parameters)
        records.append({'experiment_id':'experiment-%08d'%i,'design_id':design['design_id'],
            'parent_realization':design['parent_realization'],'lineage_ids':design['lineage_ids'],
            'observation_draw':i,'selection_attempts':attempt+1})
    splits,split_record=lineage_disjoint_splits(records,seed=request['split_seed'],fractions=request['split_fractions'],minimum_groups_per_split=request['minimum_groups_per_split'])
    report={'schema':'stream-sbi-dataset-v1','records':records,'split_record':split_record,
        'parameter_priors':space.records,'parameter_names':space.names,'encoder':encoder.record,'seed':request['seed'],
        'selection_scope':'bounded conditional observable selection at each prior theta',
        'generation_status':'SIMULATED','science_acceptance':'NOT_ACCEPTED'}
    all_theta=np.asarray(theta)
    atomic_json(privileged_truth_destination,json_safe({'schema':'stream-sbi-heldout-truth-v1',
        'test_rows':splits['test'],'true_parameters':all_theta[splits['test']],
        'experiment_ids':[records[i]['experiment_id'] for i in splits['test']], 'split_record':split_record}))
    public_theta=all_theta.copy()
    public_theta[splits['test']]=np.nan
    report['privileged_truth_artifact']=Path(privileged_truth_destination).name
    return report,{'training_theta':public_theta,'observable_features':np.asarray(features),
        **{name+'_rows':rows for name,rows in splits.items()}}


# Each coordinate uses its recorded physical units and finite prior bounds.
# A log-uniform density is 1 / (q * log(high/low)); subtracting logarithms
# avoids forming a potentially overflowing high/low ratio.
def build_recorded_torch_prior(records):
    import torch
    from torch.distributions import Distribution,constraints,Normal
    class RecordedPrior(Distribution):
        arg_constraints={}
        has_rsample=False
        def __init__(self):
            self.records=records
            self.low=torch.tensor([p['bounds'][0] for p in records],dtype=torch.float32)
            self.high=torch.tensor([p['bounds'][1] for p in records],dtype=torch.float32)
            super().__init__(batch_shape=torch.Size(),event_shape=torch.Size([len(records)]),validate_args=False)
        @property
        def support(self):
            return constraints.independent(constraints.interval(self.low,self.high),1)
        def sample(self,sample_shape=torch.Size()):
            u=torch.rand(tuple(sample_shape)+(len(records),)).clamp(1e-7,1-1e-7)
            result=torch.empty_like(u)
            for j,row in enumerate(records):
                low,high=row['bounds']
                if row['kind']=='uniform':
                    result[...,j]=low+(high-low)*u[...,j]
                elif row['kind']=='loguniform':
                    result[...,j]=torch.exp(np.log(low)+(np.log(high)-np.log(low))*u[...,j])
                else:
                    normal=Normal(row['mean'],row['sigma'])
                    a=normal.cdf(torch.tensor(low))
                    b=normal.cdf(torch.tensor(high))
                    result[...,j]=normal.icdf(a+(b-a)*u[...,j])
            return result
        def log_prob(self,value):
            total=torch.zeros(value.shape[:-1],device=value.device)
            for j,row in enumerate(records):
                low,high=row['bounds']
                q=value[...,j]
                if row['kind']=='uniform':
                    term=torch.full_like(q,-np.log(high-low))
                elif row['kind']=='loguniform':
                    term=-torch.log(q.clamp_min(torch.finfo(q.dtype).tiny))-np.log((np.log(high)-np.log(low)))
                else:
                    normal=Normal(torch.tensor(row['mean'],device=q.device),torch.tensor(row['sigma'],device=q.device))
                    norm=normal.cdf(torch.tensor(high,device=q.device))-normal.cdf(torch.tensor(low,device=q.device))
                    term=normal.log_prob(q)-torch.log(norm)
                total+=term
            support=((value>=self.low.to(value.device))&(value<=self.high.to(value.device))).all(dim=-1)
            return total.masked_fill(~support,-float('inf'))
    return RecordedPrior()


def train_posterior(dataset,arrays,request,directory,*,forward_receipt):
    """Use actual sbi flow builders; optimize on explicitly disjoint loaders.

    This bypasses the upstream trainer's random row split, not its density
    estimator. The heldout test truth is never passed to network construction,
    normalization, early stopping or optimizer updates.
    """
    import torch
    from core.stream.encoders import SummaryEncoder,StreamGraphEncoder
    backend=pinned_sbi()
    from sbi.neural_nets import posterior_nn
    from sbi.inference.posteriors import DirectPosterior
    from core.stream.inference import pinned_sampler_backend
    serializer,serializer_binding=pinned_sampler_backend('cloudpickle')
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    train=np.asarray(arrays['train_rows'],int)
    validation=np.asarray(arrays['validation_rows'],int)
    test=np.asarray(arrays['test_rows'],int)
    if any(set(a)&set(b) for a,b in ((train,validation),(train,test),(validation,test))):
        raise ValueError('SBI split rows overlap')
    if not len(train) or not len(validation):
        raise ValueError('Nonempty lineage-disjoint optimization sets required')
    theta=np.asarray(arrays['training_theta'])
    feature=np.asarray(arrays['observable_features'])
    if np.isfinite(theta[test]).any():
        raise ValueError('Test truth must be withheld from public training tensors')
    # Only train/validation tensors enter the trainer. Do not serialize test
    # truth into the posterior, the normalization or the model checkpoint.
    train_theta=torch.tensor(theta[train],dtype=torch.float32)
    train_x=torch.tensor(feature[train],dtype=torch.float32)
    val_theta=torch.tensor(theta[validation],dtype=torch.float32)
    val_x=torch.tensor(feature[validation],dtype=torch.float32)
    seed=integer(request['seed'],'seed',0)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    architecture=request['architecture']
    definition=dataset['encoder']
    dimensions=len(definition['columns'])
    if architecture=='summary':
        embedding=SummaryEncoder(definition['maximum_nodes'],dimensions)
    elif architecture=='graph':
        if not request.get('experimental_graph_training'):
            raise ValueError('Graph candidate training must explicitly remain experimental until baseline-gain calibration')
        embedding=StreamGraphEncoder(definition,**request['graph'])
        estimate=request['batch_size']*definition['maximum_nodes']**2*(request['graph']['hidden_features']+dimensions+8)*8
        if estimate>request['memory_gib']*1024**3:
            raise MemoryError('Graph pair workspace exceeds budget')
    else:raise ValueError('Architecture must be summary or genuine message-passing graph')
    factory=posterior_nn(model=request['density_estimator'],embedding_net=embedding,z_score_x='none',
        hidden_features=request['flow_hidden_features'],num_transforms=request['flow_transforms'])
    network=factory(train_theta,train_x)
    optimizer=torch.optim.Adam(network.parameters(),lr=positive(request['learning_rate'],'learning_rate'))
    batch=integer(request['batch_size'],'batch_size')
    epochs=integer(request['maximum_epochs'],'maximum_epochs')
    patience=integer(request['early_stopping_patience'],'early_stopping_patience')
    maximum_steps=integer(request['maximum_optimizer_steps'],'maximum_optimizer_steps')
    best=float('inf')
    stale=0
    step=0
    loss_history=[]
    best_state=None
    with native_run_lock(directory,operation='lineage-disjoint SBI optimizer'):
        for epoch in range(epochs):
            network.train()
            order=torch.randperm(len(train_theta))
            training_loss=0.
            # The optimizer-step cap may stop partway through an epoch.
            # Average over rows actually visited, including a short last batch.
            trained_rows=0
            for begin in range(0,len(order),batch):
                rows=order[begin:begin+batch]
                optimizer.zero_grad()
                loss=network.loss(train_theta[rows],condition=train_x[rows]).mean()
                if not torch.isfinite(loss):
                    raise ArithmeticError('Nonfinite posterior training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(),request['gradient_clip_norm'])
                optimizer.step()
                training_loss+=float(loss.detach())*len(rows)
                trained_rows+=len(rows)
                step+=1
                if step>=maximum_steps:
                    break
            network.eval()
            with torch.no_grad():
                losses=[network.loss(val_theta[i:i+batch],condition=val_x[i:i+batch]).sum() for i in range(0,len(val_theta),batch)]
                validation_loss=float(torch.stack(losses).sum()/len(val_theta))
            loss_history.append([training_loss/trained_rows,validation_loss])
            if validation_loss<best-request['minimum_validation_improvement']:
                best=validation_loss
                best_state=copy.deepcopy(network.state_dict())
                stale=0
            else:stale+=1
            if stale>=patience or step>=maximum_steps:
                break
        if best_state is None:
            raise RuntimeError('No finite validation-selected model state')
        network.load_state_dict(best_state)
        network.eval()
        prior=build_recorded_torch_prior(dataset['parameter_priors'])
        posterior=DirectPosterior(posterior_estimator=network,prior=prior,device='cpu')
        payload={'posterior':posterior,'encoder':definition,'priors':dataset['parameter_priors'],'architecture':architecture}
        temporary=directory/'posterior.pkl.tmp'
        with temporary.open('wb') as stream:
            serializer.dump(payload,stream)
        temporary.replace(directory/'posterior.pkl')
        model_hash=digest(directory/'posterior.pkl')
        summary={'schema':'stream-sbi-frozen-model-v1','model_sha256':model_hash,'model_file':str((directory/'posterior.pkl').resolve()),
            'architecture':architecture,'backend':backend,'serializer':serializer_binding,'encoder':definition,
            'parameter_priors':dataset['parameter_priors'],'training_request':request,'training_split':dataset['split_record'],
            'optimizer_steps':step,'epochs':len(loss_history),'best_validation_loss':best,
            'forward_receipt':forward_receipt,'freeze_status':'FROZEN_BEFORE_TEST_TRUTH','calibration_status':'NOT_EXECUTED',
            'deployment_status':'DISABLED_PENDING_INDEPENDENT_CALIBRATION','science_acceptance':'NOT_ACCEPTED'}
        atomic_json(directory/'model.json',json_safe(summary))
    return summary,{'optimization_loss':np.asarray(loss_history),'model_digest_bytes':np.frombuffer(bytes.fromhex(model_hash),dtype=np.uint8)}


def load_frozen_model(record):
    """Trusted local pickle with exact immutable model/source binding."""
    import pickle
    import torch
    current_backend=pinned_sbi()
    from core.stream.inference import pinned_sampler_backend
    _,current_serializer=pinned_sampler_backend('cloudpickle')
    for current,original in ((current_backend,record['backend']),(current_serializer,record['serializer'])):
        for source,expected in current['source_hashes'].items():
            if original['source_hashes'].get(source)!=expected:
                raise ValueError('Frozen SBI implementation, backend or serializer binding changed')
    path=Path(record['model_file'])
    if digest(path)!=record['model_sha256']:
        raise ValueError('Frozen SBI model bytes changed')
    if record['freeze_status']!='FROZEN_BEFORE_TEST_TRUTH':
        raise ValueError('No frozen pre-test model')
    for source,expected in record['backend']['source_hashes'].items():
        if digest(source)!=expected:
            raise ValueError('Frozen SBI backend asset changed')
    with path.open('rb') as stream:
        model=pickle.load(stream)
    model['posterior'].posterior_estimator.eval()
    return model


def calibrate_frozen_model(model_record,dataset,arrays,request,*,privileged_truth):
    """Read heldout truth only after frozen model and group exclusions exist."""
    import torch
    model=load_frozen_model(model_record)
    posterior=model['posterior']
    rows=np.asarray(arrays['test_rows'],int)
    if not len(rows):
        raise ValueError('No independent test realization rows')
    if dataset['split_record']!=model_record['training_split']:
        raise ValueError('Calibration dataset lineage split differs from frozen training model')
    if privileged_truth['schema']!='stream-sbi-heldout-truth-v1' or not np.array_equal(rows,privileged_truth['test_rows']):
        raise ValueError('Test truth rows differ from frozen heldout design')
    if privileged_truth['experiment_ids']!=[dataset['records'][i]['experiment_id'] for i in rows]:
        raise ValueError('Heldout experiment ID mismatch')
    truth_lookup={int(row):theta for row,theta in zip(rows,privileged_truth['true_parameters'])}
    calibrated_rows=[]
    ranks=[]
    covered=[]
    scores=[]
    draw_count=integer(request['posterior_draws'],'posterior_draws',32)
    rng=seeded_rng(request['seed'],'sbc-tie')
    torch.manual_seed(request['seed'])
    for index in rows:
        x=torch.tensor(arrays['observable_features'][index],dtype=torch.float32)[None]
        truth=finite(truth_lookup[int(index)],(len(dataset['parameter_priors']),))
        samples=posterior.sample((draw_count,),x=x,show_progress_bars=False).detach().cpu().numpy()
        # Randomized tie ranks remain discrete uniforms for finite draws.
        rank=np.sum(samples<truth,axis=0)+np.array([rng.integers(0,int(ties)+1) for ties in np.sum(samples==truth,axis=0)])
        ranks.append(rank)
        lo,hi=np.quantile(samples,[(1-request['credible_mass'])/2,(1+request['credible_mass'])/2],axis=0)
        covered.append((truth>=lo)&(truth<=hi))
        with torch.no_grad():
            scores.append(float(posterior.log_prob(torch.tensor(truth,dtype=torch.float32)[None],x=x)[0]))
        calibrated_rows.append({'experiment_id':dataset['records'][index]['experiment_id'],'parent_realization':dataset['records'][index]['parent_realization'],
            'lineage_ids':dataset['records'][index]['lineage_ids']})
    ranks=np.asarray(ranks)
    coverage=np.asarray(covered)
    # Correlated multiple experiments within a realization are not independent
    # binomial trials. Aggregate one selected case per connected component.
    # Existing split_record stores test component roots, which are exact row
    # indices from union-find grouping and remain globally stable.
    independent=list(dataset['split_record']['component_groups']['test'])
    local={int(row):i for i,row in enumerate(rows)}
    independent_local=[local[i] for i in independent if i in local]
    if len(independent_local)<request['minimum_independent_test_groups']:
        raise ValueError('Insufficient independent test groups for calibration')
    selected=coverage[independent_local]
    intervals=[]
    pvalues=[]
    for j in range(selected.shape[1]):
        result=binomtest(int(selected[:,j].sum()),len(selected),p=request['credible_mass'])
        interval=result.proportion_ci(confidence_level=request['coverage_confidence'])
        intervals.append([interval.low,interval.high])
        pvalues.append(result.pvalue)
    # Randomized PIT values have continuous uniform null under finite-rank SBC.
    pit=(ranks[independent_local]+rng.random((len(independent_local),ranks.shape[1])))/(draw_count+1)
    sbc_pvalue=np.array([kstest(pit[:,j],'uniform').pvalue for j in range(pit.shape[1])])
    tolerance=positive(request['maximum_coverage_absolute_error'],'coverage tolerance')
    intervals=np.asarray(intervals)
    coverage_equivalent=(intervals[:,0]>=request['credible_mass']-tolerance)&(intervals[:,1]<=request['credible_mass']+tolerance)
    # Bonferroni controls the recorded family of marginal coverage/SBC decisions.
    family_size=2*selected.shape[1]
    coverage_adjusted=np.minimum(1,np.asarray(pvalues)*family_size)
    sbc_adjusted=np.minimum(1,sbc_pvalue*family_size)
    accepted=bool(coverage_equivalent.all() and np.all(coverage_adjusted>=request['minimum_coverage_pvalue']) and np.all(sbc_adjusted>=request['minimum_sbc_pvalue']))
    report={'schema':'stream-sbi-calibration-v1','model_sha256':model_record['model_sha256'],
        'heldout_cases':calibrated_rows,'independent_groups':len(independent_local),'credible_mass':request['credible_mass'],
        'coverage_equivalence_supported':coverage_equivalent,'multiple_comparison_method':'Bonferroni, coverage and SBC parameter families','coverage_interval':intervals,'coverage_pvalue':pvalues,'sbc_pvalue':sbc_pvalue,'calibration_controls':request,
        'calibration_accepted':accepted,'science_acceptance':'NOT_ACCEPTED','evaluation_status':'EXECUTED'}
    return report,{'sbc_rank':ranks,'credible_covered':coverage,'heldout_log_score':np.asarray(scores),
        'independent_test_local_rows':np.asarray(independent_local,int),'sbc_pit':pit}


def graph_baseline_gain(graph_report,graph_arrays,baseline_report,baseline_arrays,*,minimum_mean_gain,minimum_gain_probability,seed,bootstrap_draws):
    if graph_report['heldout_cases']!=baseline_report['heldout_cases']:
        raise ValueError('Graph/baseline comparison must use identical frozen heldout realizations')
    if not graph_report['calibration_accepted'] or not baseline_report['calibration_accepted']:
        raise ValueError('Both posterior families require independent calibration')
    rows=np.asarray(graph_arrays['independent_test_local_rows'],int)
    if not np.array_equal(rows,baseline_arrays['independent_test_local_rows']):
        raise ValueError('Paired calibration group selections differ')
    difference=(graph_arrays['heldout_log_score']-baseline_arrays['heldout_log_score'])[rows]
    rng=seeded_rng(seed,'graph-baseline-gain')
    mean=np.array([difference[rng.integers(0,len(difference),len(difference))].mean() for _ in range(bootstrap_draws)])
    probability=float(np.mean(mean>minimum_mean_gain))
    return {'schema':'stream-graph-baseline-gain-v1','graph_model_sha256':graph_report['model_sha256'],
        'baseline_model_sha256':baseline_report['model_sha256'],'paired_independent_groups':len(rows),
        'mean_log_score_gain':float(difference.mean()),'gain_probability':probability,
        'accepted':probability>=minimum_gain_probability,'controls':{'minimum_mean_gain':minimum_mean_gain,'minimum_gain_probability':minimum_gain_probability,
            'seed':seed,'bootstrap_draws':bootstrap_draws},'evaluation_status':'EXECUTED','science_acceptance':'NOT_ACCEPTED'}, {'paired_log_score_gain':difference,'bootstrap_gain':mean}


def fit_ood_reference(features,*,covariance_floor,range_padding):
    x=finite(features)
    mean=x.mean(axis=0)
    cov=np.atleast_2d(np.cov(x,rowvar=False))+np.eye(x.shape[1])*positive(covariance_floor,'covariance_floor')
    return {'mean':mean,'covariance':cov,'lower':x.min(axis=0)-range_padding,'upper':x.max(axis=0)+range_padding,
        'fit_scope':'training observations only; no heldout truth'}


def evaluate_ood(feature,reference,*,maximum_mahalanobis2,maximum_outside_fraction):
    value=finite(feature)
    mean=finite(reference['mean'],value.shape)
    root=np.linalg.cholesky(finite(reference['covariance'],(len(value),len(value))))
    delta=np.linalg.solve(root,value-mean)
    distance=float(delta@delta)
    outside=float(np.mean((value<reference['lower'])|(value>reference['upper'])))
    return {'supported':distance<=maximum_mahalanobis2 and outside<=maximum_outside_fraction,'mahalanobis2':distance,'outside_fraction':outside}


def infer_calibrated_posterior(model_record,calibration,observation,request,*,ood_reference,graph_gain=None):
    import torch
    from core.stream.encoders import ObservableEncoder,SummaryEncoder
    if not calibration['calibration_accepted'] or calibration['model_sha256']!=model_record['model_sha256']:
        raise ValueError('Independent calibration for the actual frozen model is required')
    if model_record['architecture']=='graph' and (not graph_gain or not graph_gain['accepted'] or graph_gain['graph_model_sha256']!=model_record['model_sha256']):
        raise ValueError('Graph inference requires demonstrated calibrated-baseline gain')
    estimator_only(observation)
    encoder=ObservableEncoder(model_record['encoder'])
    feature=encoder.encode(observation)
    summary=SummaryEncoder(encoder.maximum_nodes,len(encoder.columns))(torch.tensor(feature)[None]).detach().numpy()[0]
    ood=evaluate_ood(summary,ood_reference,**request['ood'])
    if not ood['supported']:
        return {'schema':'stream-sbi-posterior-v1','status':'OUT_OF_DOMAIN','ood':ood,'science_acceptance':'NOT_ACCEPTED'}, {'posterior_samples':np.empty((0,len(model_record['parameter_priors']))),'posterior_weight':np.empty(0)}
    model=load_frozen_model(model_record)
    torch.manual_seed(request['seed'])
    samples=model['posterior'].sample((integer(request['draws'],'draws'),),x=torch.tensor(feature)[None],show_progress_bars=False).detach().numpy()
    report={'schema':'stream-sbi-posterior-v1','status':'INFERRED','model_sha256':model_record['model_sha256'],
        'architecture':model_record['architecture'],'parameter_priors':model_record['parameter_priors'],'ood':ood,
        'calibration':calibration,'graph_gain':graph_gain,'science_acceptance':'NOT_ACCEPTED'}
    return report,{'posterior_samples':samples,'posterior_weight':np.full(len(samples),1/len(samples))}


SBI_KINDS={'sbi_forward_signature','sbi_dataset','sbi_train','sbi_calibration','sbi_graph_gain','sbi_posterior'}


def execute_sbi_stage(stage,inputs,working):
    kind=stage['kind']
    p=stage['parameters']
    refs=stage['inputs']
    namespace='sbi:'+stage['stage_id']
    if kind in {'sbi_dataset','sbi_forward_signature'}:
        from core.stream.encoders import ObservableEncoder
        from core.stream.experiments import make_physical_forward,build_component_field
        from core.stream.inference import ParameterSpace
        designs=inputs.json(refs['designs'])
        priors=p['priors']
        bindings=p['bindings']
        space=ParameterSpace(priors)
        definitions={}
        simulators={}
        from core.stream.workflow import Inputs
        domain_inputs=Inputs(inputs.base,inputs.published,inputs.resources)
        code_paths = [(_CORE_ROOT/name) for name in
            ('streams.py','stream_common.py','action_angle.py','stream_tracks.py','stream_resimulation.py','gravity_fields.py','stream_catalogue.py','halos.py','data.py','foundation.py','stars.py','cosmology.py')]
        code_paths.extend(sorted((_CORE_ROOT/'stream').glob('*.py')))
        code_paths.extend(sorted((_CORE_ROOT/'research').glob('*.py')))
        for source in code_paths:
            domain_inputs.sources[str(source.resolve())] = digest(source)
        for row in designs:
            if row['design_id'] in definitions:
                raise ValueError('Duplicate SBI design ID')
            definition=domain_inputs.json(row['forward_definition'])
            definition=copy.deepcopy(definition)
            definition['integration'].update(memory_gib=inputs.resources['memory_gib'],maximum_particles=inputs.resources['maximum_particles'])
            build_component_field(definition['field'],domain_inputs)
            definitions[row['design_id']]=definition
            simulators[row['design_id']]=make_physical_forward(definition,bindings,domain_inputs)
        inputs.sources.update(domain_inputs.sources)
        sources=dict(domain_inputs.sources)
        signature=forward_signature(definitions,bindings,sources)
        if kind=='sbi_forward_signature':
            return {'schema':'sbi-forward-signature-v1','forward_signature':signature,'definitions':definitions,'bindings':bindings,'domain':p['domain'],'source_hashes':sources,'science_acceptance':'NOT_ACCEPTED'}, {'signature_bytes':np.frombuffer(bytes.fromhex(signature),dtype=np.uint8)},namespace,None
        receipt=inputs.json(refs['accepted_forward_domain'])
        require_forward_model_accepted_for(receipt,p['domain'],expected_signature=signature,priors=priors)
        request={**p['generation'],'memory_gib':inputs.resources['memory_gib']}
        report,arrays=generate_training_experiments(designs,simulators,ObservableEncoder(p['encoder']),request,
            space=space,privileged_truth_destination=working/'heldout_truth.json')
        report.update(forward_receipt=receipt,forward_signature=signature)
    elif kind=='sbi_train':
        dataset,values=inputs.product(refs['dataset'])
        receipt=inputs.json(refs['accepted_forward_domain'])
        require_forward_model_accepted_for(receipt,p['domain'],expected_signature=dataset['forward_signature'],priors=dataset['parameter_priors'])
        path=Path(p['model_directory'])
        path=path if path.is_absolute() else inputs.base/path
        request={**p['training'],'memory_gib':inputs.resources['memory_gib'],'dataset_sha256':digest(inputs.path(refs['dataset']))}
        report,arrays=train_posterior(dataset,values,request,path,forward_receipt=receipt)
        inputs.sources.update(report['backend']['source_hashes'])
        inputs.sources.update(report['serializer']['source_hashes'])
        from core.stream.encoders import SummaryEncoder
        import torch
        train_rows=values['train_rows']
        encoded=torch.tensor(values['observable_features'][train_rows],dtype=torch.float32)
        summary=SummaryEncoder(dataset['encoder']['maximum_nodes'],len(dataset['encoder']['columns']))(encoded).detach().numpy()
        report['ood_reference']=fit_ood_reference(summary,**p['ood_reference'])
        from core.foundation import atomic_json
        atomic_json(path/'model.json',json_safe(report))
    elif kind=='sbi_calibration':
        model=inputs.json(refs['frozen_model'])
        load_frozen_model(model)
        dataset,values=inputs.product(refs['dataset'])
        if model['training_request']['dataset_sha256']!=digest(inputs.path(refs['dataset'])):
            raise ValueError('Calibration dataset differs from actual training dataset')
        # The trusted evaluation-only truth read happens after freeze binding.
        truth=inputs.json(refs['privileged_truth'])
        report,arrays=calibrate_frozen_model(model,dataset,values,p['calibration'],privileged_truth=truth)
    elif kind=='sbi_graph_gain':
        graph,g=inputs.product(refs['graph_calibration'])
        baseline,b=inputs.product(refs['baseline_calibration'])
        report,arrays=graph_baseline_gain(graph,g,baseline,b,**p['controls'])
    elif kind=='sbi_posterior':
        model=inputs.json(refs['frozen_model'])
        calibration=inputs.json(refs['calibration'])
        gain=inputs.json(refs['graph_gain']) if refs.get('graph_gain') else None
        observation=inputs.json(refs['observation'])
        report,arrays=infer_calibrated_posterior(model,calibration,observation,p['inference'],ood_reference=model['ood_reference'],graph_gain=gain)
    else:raise ValueError('Unknown SBI stage')
    return report,arrays,namespace,None


def prepare_native_sbi_branch(request,destination):
    from core.stream.workflow import Inputs,validate_stream_campaign_request
    from core.stream.encoders import ObservableEncoder
    from core.analysis_figure_campaign import prepare_analysis_figure_campaign
    import math
    import tempfile
    from core.foundation import merge_source_bindings as _merge
    from core.foundation import file_sha256
    required={'schema','designs','accepted_forward_domain','observations','selection','predictive_design_id','domain','priors','bindings','encoder',
        'generation','summary_training','graph_training','ood_reference','calibration','gain','inference','predictive','predictive_sampler','resources','wall_seconds'}
    if set(request)!=required or request['schema']!='native-sbi-branch-request-v1':
        raise ValueError('Complete native SBI training/calibration/deployment branch request required')
    out=Path(destination).resolve()
    if out.exists():
        raise FileExistsError('Preserve prior native SBI branch preparation')
    resources=request['resources']
    memory=resources['memory_gib']
    pins={}
    paths={}
    if type(memory) not in (int,float) or not math.isfinite(memory) or memory<=0:
        raise ValueError('Positive native SBI allocation required')
    for key in ['designs','accepted_forward_domain','observations','selection']:
        ref=request[key]
        if set(ref)!={'path','sha256'}:
            raise ValueError('Exact original SBI design/acceptance/observation/selection binding required')
        path=Path(ref['path']).resolve()
        if path.stat().st_size*32>memory*1024**3/8:
            raise MemoryError('Original SBI metadata exceeds allocation')
        if file_sha256(path)!=ref['sha256']:
            raise ValueError('Original SBI source changed')
        paths[key]=path
        pins[str(path)]=ref['sha256']
    designs=json.loads(paths['designs'].read_text())
    observation=json.loads(paths['observations'].read_text())
    estimator_only(observation)
    priors=ParameterSpace(request['priors'])
    encoder=ObservableEncoder(request['encoder'])
    encoder.encode(observation)
    if {b['name'] for b in request['bindings']}!=set(priors.names):
        raise ValueError('Every actual SBI physical/nuisance coordinate needs its original forward binding')
    if not isinstance(designs,list) or not designs or len({d['design_id'] for d in designs})!=len(designs):
        raise ValueError('Distinct actual native forward-design identities required')
    # Resolve actual design assets against the original design list's location.
    # Do not synthesize extra parent/lineage IDs to meet ensemble floors.
    normalized=copy.deepcopy(designs)
    chosen=[]
    for design in normalized:
        source=Path(design['forward_definition'])
        if not source.is_absolute():
            raise ValueError('Frozen native SBI forward definitions must use actual absolute paths; do not relocate relative scientific assets')
        source=source.resolve()
        if source.stat().st_size*32>memory*1024**3/8:
            raise MemoryError('Native SBI forward definition exceeds allocation')
        design['forward_definition']=str(source)
        pins[str(source)]=file_sha256(source)
        if design['design_id']==request['predictive_design_id']:
            chosen.append(design)
    if len(chosen)!=1:
        raise ValueError('Predictive forward must be one original accepted native design')
    generation=request['generation']
    count=generation['experiments']
    if type(count) is not int or count<3 or count*encoder.output_size*8>memory*1024**3/2:
        raise MemoryError('Native SBI retained training features exceed allocation')
    records=[{'experiment_id':'experiment-%08d'%i,'parent_realization':normalized[i%len(normalized)]['parent_realization'],
        'lineage_ids':normalized[i%len(normalized)]['lineage_ids']} for i in range(count)]
    splits,lineage=lineage_disjoint_splits(records,seed=generation['split_seed'],fractions=generation['split_fractions'],minimum_groups_per_split=generation['minimum_groups_per_split'])
    if len(lineage['component_groups']['test'])<request['calibration']['minimum_independent_test_groups']:
        raise ValueError('Actual disjoint design cannot meet the unchanged independent heldout calibration floor')
    summary=request['summary_training']
    graph=request['graph_training']
    if summary['architecture']!='summary' or graph['architecture']!='graph' or graph.get('experimental_graph_training') is not True:
        raise ValueError('Original summary baseline and explicit experimental graph training required')
    if set(request['inference'])!={'seed','draws','ood'}:
        raise ValueError('Original bounded deployment controls required')
    stream_resources={'memory_gib':memory,'maximum_particles':resources['maximum_particles'],'maximum_stages':16}
    signature_stage={'stage_id':'signature','kind':'sbi_forward_signature','inputs':{'designs':str(paths['designs'])},
        'parameters':{key:copy.deepcopy(request[key]) for key in ['domain','priors','bindings','encoder']}}
    # Original signature execution only resolves source assets and constructs
    # field objects; it never calls a forward simulator or creates training data.
    # Existing definitions may use relative paths, so use the original base.
    inputs=Inputs(paths['designs'].parent,{},stream_resources)
    with tempfile.TemporaryDirectory(prefix='sbi-signature-preparation-') as tmp:
        signature,_,_,_=execute_sbi_stage(signature_stage,inputs,Path(tmp))
    accepted=json.loads(paths['accepted_forward_domain'].read_text())
    require_forward_model_accepted_for(accepted,request['domain'],expected_signature=signature['forward_signature'],priors=request['priors'])
    relocated=Inputs(out,{},stream_resources)
    with tempfile.TemporaryDirectory(prefix='sbi-signature-relocation-') as tmp:
        replay,_,_,_=execute_sbi_stage(signature_stage,relocated,Path(tmp))
    if replay['forward_signature']!=signature['forward_signature']:
        raise ValueError('Prepared campaign relocation changes accepted native forward inputs; freeze absolute scientific asset paths')
    _merge(pins,inputs.sources)
    _merge(pins,signature['source_hashes'])
    for control in accepted['capability_receipts'].values():
        _merge(pins,control.get('source_hashes',{}))
    ref=lambda sid,artifact='product.json':{'stage':sid,'artifact':artifact}
    stages=[signature_stage,{'stage_id':'dataset','kind':'sbi_dataset','depends_on':['signature'],
        'inputs':{'designs':str(paths['designs']),'accepted_forward_domain':str(paths['accepted_forward_domain'])},
        'parameters':{**{key:copy.deepcopy(request[key]) for key in ['domain','priors','bindings','encoder']},'generation':copy.deepcopy(generation)}}]
    for label,training in [('summary',summary),('graph',graph)]:
        stages.append({'stage_id':label,'kind':'sbi_train','depends_on':['dataset'],'inputs':{'dataset':ref('dataset'),
            'accepted_forward_domain':str(paths['accepted_forward_domain'])},'parameters':{'domain':copy.deepcopy(request['domain']),
            'model_directory':str(out/(label+'-frozen-model')),'training':copy.deepcopy(training),'ood_reference':copy.deepcopy(request['ood_reference'])}})
        stages.append({'stage_id':label+'_calibration','kind':'sbi_calibration','depends_on':['dataset',label],
            'inputs':{'dataset':ref('dataset'),'frozen_model':ref(label),'privileged_truth':ref('dataset','heldout_truth.json')},
            'parameters':{'calibration':copy.deepcopy(request['calibration'])}})
    stages.append({'stage_id':'graph_gain','kind':'sbi_graph_gain','depends_on':['summary_calibration','graph_calibration'],
        'inputs':{'graph_calibration':ref('graph_calibration'),'baseline_calibration':ref('summary_calibration')},'parameters':{'controls':copy.deepcopy(request['gain'])}})
    stages.append({'stage_id':'posterior','kind':'sbi_posterior','depends_on':['graph','graph_calibration','graph_gain'],
        'inputs':{'frozen_model':ref('graph'),'calibration':ref('graph_calibration'),'graph_gain':ref('graph_gain'),'observation':str(paths['observations'])},
        'parameters':{'inference':copy.deepcopy(request['inference'])}})
    # SBI training marginalizes over native design/parent identities. A
    # designated single-design predictive must use its own physically fitted
    # conditional posterior, not relabel draws from that design mixture.
    physical_inputs={'forward_definition':chosen[0]['forward_definition'],'observations':str(paths['observations']),'selection':str(paths['selection'])}
    stages.append({'stage_id':'physical_posterior','kind':'stream_inference','inputs':copy.deepcopy(physical_inputs),
        'parameters':{'priors':copy.deepcopy(request['priors']),'bindings':copy.deepcopy(request['bindings']),
            'sampler':copy.deepcopy(request['predictive_sampler']),'checkpoint_directory':str(out/'conditional-physical-posterior-checkpoints')}})
    stages.append({'stage_id':'predictive','kind':'posterior_predictive','depends_on':['physical_posterior'],
        'inputs':{'posterior':ref('physical_posterior'),**physical_inputs},
        'parameters':{'priors':copy.deepcopy(request['priors']),'bindings':copy.deepcopy(request['bindings']),'controls':copy.deepcopy(request['predictive'])}})
    campaign={'schema':'stream-campaign-v1','experiment_id':'native-independent-summary-graph-inference','resources':stream_resources,'stages':stages}
    validate_stream_campaign_request(campaign)
    for p,h in pins.items():
        if file_sha256(p)!=h:
            raise ValueError('Native SBI preparation input changed')
    out.mkdir(parents=True)
    path=out/'campaign.json'
    atomic_json(path,campaign)
    selection=[('actual_posterior','posterior'),('posterior_predictive','predictive'),('sbc_ranks','graph_calibration'),
        ('coverage','graph_calibration'),('ood_support','posterior'),('graph_baseline_gain','graph_gain')]
    figure_request={'schema':'analysis-figure-campaign-request-v1','campaign':{'path':str(path),'sha256':file_sha256(path)},
        'resources':{key:resources[key] for key in ['memory_gib','disk_gib','mpi_ranks','omp_threads','maximum_stage_count']},'wall_seconds':request['wall_seconds'],
        'figures':[{'variant':variant,'producer_stage':sid} for variant,sid in selection]}
    atomic_json(out/'figure-request.json',figure_request)
    figures=prepare_analysis_figure_campaign(figure_request,out/'figures')
    _merge(pins,figures['source_hashes'])
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    result={'schema':'native-sbi-branch-prepared-v1','status':'RECIPES_PREPARED','campaign':{'path':str(path),'sha256':file_sha256(path)},
        'workflow':figures['workflow'],'source_hashes':pins,'forward_signature':signature['forward_signature'],'actual_lineage_split':json.loads(json.dumps(lineage,default=lambda a:a.tolist())),
        'variants':figures['variants'],'analysis_execution':'SIGNATURE_CONSTRUCTION_ONLY;NO_FORWARD_SIMULATION_TRAINING_OR_CALIBRATION',
        'science_acceptance':'NOT_ACCEPTED','scope':'Original accepted physical forward domain remains mandatory. Both SBI architectures use one actual lineage-disjoint dataset; original frozen-model calibration, identical heldout baseline comparison, OOD and deployment gates unchanged. The designated-design physical posterior predictive uses its own original conditional physical likelihood/sampler; it is separate from the SBI posterior marginalized over training design identities.'}
    atomic_json(out/'manifest.json',result)
    return result
