"""Native stellar debris catalogue with measured ancestry and calibrated ambiguity.

Connected components in explicitly scaled six-dimensional space are proposals,
not stream detections. Classification uses independently labeled controls.
Unknown ancestry, sparse candidates and failed controls remain visible.
"""

from collections import defaultdict, deque
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp
from scipy.linalg import solve_triangular
from scipy.stats import beta
from core.data import exact_uint64_ids
from core.stream_common import finite, phase_space, weighted_covariance, positive, integer, normalize

CLASSES = ('bound_cluster', 'bound_dwarf', 'thin_stream', 'tidal_tail', 'shell_plume', 'disk', 'bridge', 'background')
FEATURES = ('log_count', 'log_elongation', 'log_flattening', 'log_width_kpc', 'log_dispersion_kms',
            'orbital_alignment', 'radial_velocity_fraction', 'radial_shell_thickness', 'bound_fraction', 'ancestry_purity',
            'log_linear_density_msun_kpc')


def trace_progenitors(ids, namespace, snapshot, graph_directory, *, memory_gib=1.):
    """Trace exact owner observations and explicit native owner continuity.

    A SUBFIND row is epoch-specific. Equal row numbers in different snapshots
    do not establish the same progenitor; only unbranched native descendant
    edges join owner observations. Mergers/fissions remain distinct branches.
    """
    from core.halos import verify_identity_graph
    ids=exact_uint64_ids(ids,allow_empty=True)
    snapshot=integer(snapshot,'snapshot',0)
    if not isinstance(namespace,str) or not namespace.strip():raise ValueError('Particle namespace is required')
    if (Path(graph_directory)/'report.json').stat().st_size*6>positive(memory_gib,'memory_gib')*1024**3:
        raise MemoryError('Identity graph ancestry intake exceeds declared memory')
    report=verify_identity_graph(graph_directory)
    observations={tuple(row['key']):row for row in report['nodes']}
    owner_kinds={'subhalo','galaxy','progenitor'}
    wanted=set(map(int,ids));history=defaultdict(list)
    previous=defaultdict(set);following=defaultdict(set)
    for edge in report['edges']:
        source,target=tuple(edge['source']),tuple(edge['target'])
        if source[2]>snapshot or target[2]>snapshot:continue
        if edge['kind']=='bound_owner' and source[0]==namespace and source[1] in wanted and observations[target]['kind'] in owner_kinds:
            history[source[1]].append((source[2],target,edge['evidence']))
        if (edge['kind']=='descendant' and edge['evidence'].get('origin')=='native_ancestry'
                and observations[source]['kind'] in owner_kinds and observations[target]['kind'] in owner_kinds
                and observations[source]['status']=='DETECTED' and observations[target]['status']=='DETECTED'):
            previous[target].add(source);following[source].add(target)
    anchors={}

    def owner_anchor(owner):
        trail=[];seen=set();current=owner
        while current not in anchors:
            if current in seen:raise ValueError('Cyclic native progenitor continuity')
            seen.add(current);trail.append(current)
            parents=previous[current]
            if len(parents)!=1:break
            parent=next(iter(parents))
            if len(following[parent])!=1:break
            current=parent
        anchor=anchors.get(current,current)
        for key in trail:anchors[key]=anchor
        return anchor

    labels=np.full(len(ids),-1,np.int64);keys=[];key_index={};records=[]
    current_bound=np.zeros(len(ids),bool);current_known=np.zeros(len(ids),bool)
    for row,pid in enumerate(ids):
        entries=sorted(history[int(pid)],key=lambda item:item[0])
        epochs=defaultdict(set)
        for epoch,owner,_ in entries:epochs[epoch].add(owner)
        ambiguous=any(len(owners)>1 for owners in epochs.values())
        past=[entry for entry in entries if entry[0]<snapshot]
        present=[entry for entry in entries if entry[0]==snapshot]
        owner=None if ambiguous else past[-1][1] if past else present[-1][1] if present else None
        if owner is not None:
            anchor=owner_anchor(owner)
            if anchor not in key_index:key_index[anchor]=len(keys);keys.append(anchor)
            labels[row]=key_index[anchor]
        observation=observations.get((namespace,int(pid),snapshot))
        measured=bool(observation and observation['status'] in {'DETECTED','UNBOUND'}
            and observation['attributes'].get('bound_membership_status')=='MEASURED')
        current_known[row]=bool(present) or measured
        current_bound[row]=bool(present)
        # Absence from the graph is missing evidence, never an unbinding event.
        stripped=bool(past and not present and measured and observation['status']=='UNBOUND')
        records.append({'particle_id':int(pid),'progenitor':owner,
            'status':'AMBIGUOUS' if ambiguous else 'KNOWN' if owner else 'UNAVAILABLE',
            'bound_to_unbound_bracket':[past[-1][0],snapshot] if stripped else None,
            'bound_history':[{'epoch':epoch,'owner':owner,'evidence':evidence} for epoch,owner,evidence in entries]})
    return {'labels':labels,'keys':keys,'records':records,'current_bound':current_bound,
        'binding_known':current_known,'graph_fingerprint':report['fingerprint'],
        'identity_scope':'epoch-qualified owner observations joined only by unbranched native descendant edges'}


def discover_coherent_debris(sample, controls, *, ancestry_labels=None):
    """Density-connected proposals in explicitly scaled Cartesian phase space.

    Neighbor counts include the particle itself. When splitting progenitors,
    unknown ancestry is its own population: it cannot bridge two known owners.
    Count and expansion use the same population, radius and core definition.
    """
    x,v,_,_=phase_space(sample)
    minimum=integer(controls['minimum_particles'],'minimum_particles',3)
    max_edges=integer(controls['maximum_neighbor_edges'],'maximum_neighbor_edges')
    neighbors=integer(controls['minimum_neighbors'],'minimum_neighbors',2)
    split=controls['split_known_progenitors']
    if type(split) is not bool:raise ValueError('split_known_progenitors must be an explicit boolean')
    ancestry=np.full(len(x),-1,np.int64) if ancestry_labels is None else np.asarray(ancestry_labels)
    if ancestry.shape!=(len(x),) or ancestry.dtype.kind not in 'iu':raise ValueError('Aligned exact ancestry labels required')
    y=np.column_stack((x/positive(controls['link_position_kpc'],'link_position_kpc'),
        v/positive(controls['link_velocity_kms'],'link_velocity_kms')))
    groups=[np.flatnonzero(ancestry==key) for key in np.unique(ancestry)] if split else [np.arange(len(x))]
    labels=np.full(len(x),-1,np.int64);candidates=[];edge_count=0
    for group in groups:
        if not len(group):continue
        tree=cKDTree(y[group]);counts=tree.query_ball_point(y[group],1.,return_length=True)
        edge_count+=int(counts.sum())
        if edge_count>max_edges:raise MemoryError('Debris neighbor graph exceeds declared edge limit')
        adjacent=tree.query_ball_point(y[group],1.);core=counts>=neighbors
        visited=np.zeros(len(group),bool)
        for root in np.flatnonzero(core):
            if visited[root]:continue
            queue=deque([int(root)]);visited[root]=True;local=[]
            while queue:
                index=queue.popleft();local.append(index)
                if not core[index]:continue
                for neighbor in adjacent[index]:
                    if not visited[neighbor]:visited[neighbor]=True;queue.append(neighbor)
            local=np.asarray(local,dtype=np.int64);rows=np.sort(group[local]);cid=len(candidates)
            labels[rows]=cid
            candidates.append({'candidate_id':cid,'rows':rows,'resolution_supported':len(rows)>=minimum,
                'core_count':int(core[local].sum())})
    return candidates,labels


def fit_stream_shell_tail_alternatives(x, v, m, *, bound, binding_known, progenitor_labels):
    """Mass-weighted shape descriptors, not an independent stream detection.

    Spatial eigenvalues lambda1>=lambda2>=lambda3 have kpc^2 units.
    Transverse RMS width is sqrt(lambda2+lambda3); dispersion is the square
    root of the velocity-covariance trace [km/s]. The density feature uses
    M/(2*sqrt(lambda1)), a length proxy rather than a measured track density.
    Binding fraction uses only particles whose native membership is measured.
    """
    mean, covariance, neff = weighted_covariance(x, m)
    evals, axes = np.linalg.eigh(covariance); order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0); axes = axes[:, order]
    velocity, velocity_covariance, _ = weighted_covariance(v, m)
    probability=m/m.max();probability/=probability.sum()
    radius = np.linalg.norm(x, axis=1); radial, valid = normalize(x)
    vr = np.sum(v * radial, axis=1); speed2 = np.sum(v*v, axis=1)
    floor = np.finfo(float).eps
    elongation = np.sqrt((evals[0]+floor)/(evals[1]+floor))
    flattening = np.sqrt((evals[1]+floor)/(evals[2]+floor))
    tangent = axes[:, 0]
    aligned = np.dot(probability,np.abs(v@tangent)/np.maximum(np.linalg.norm(v,axis=1),1e-12))
    known = progenitor_labels >= 0
    # Compact labels prevent a sparse/high owner number allocating a huge bincount.
    _,inverse=np.unique(progenitor_labels[known],return_inverse=True)
    populations = np.bincount(inverse,weights=probability[known]) if known.any() else np.array([])
    purity = float(populations.max()) if len(populations) else 0.
    bound_fraction = np.sum(probability[bound & binding_known])/np.sum(probability[binding_known]) if binding_known.any() else np.nan
    width = np.sqrt(evals[1]+evals[2]); dispersion = np.sqrt(np.trace(velocity_covariance))
    rmean = np.dot(probability,radius)
    thickness = np.sqrt(np.dot(probability,(radius-rmean)**2))/max(rmean, 1e-12)
    radial_fraction = np.dot(probability,np.divide(vr*vr,speed2,out=np.zeros_like(vr),where=speed2>0))
    if not valid.all():radial_fraction=np.nan  # Host-centered radial direction unresolved.
    feature = np.array([np.log(len(x)), np.log(max(elongation,1e-12)), np.log(max(flattening,1e-12)),
        np.log(max(width,1e-12)), np.log(max(dispersion,1e-12)), aligned, radial_fraction, thickness, bound_fraction, purity,
        np.log(m.sum()/max(2*np.sqrt(evals[0]),1e-12))])
    return feature, {'center_kpc': mean, 'velocity_kms': velocity, 'spatial_covariance_kpc2': covariance,
        'velocity_covariance_kms2': velocity_covariance, 'principal_axes': axes, 'axis_variance_kpc2': evals,
        'effective_particle_count': neff, 'ancestry_purity': purity, 'binding_available': bool(binding_known.all()),
        'bound_fraction': bound_fraction, 'radial_shell_thickness': thickness}


class CalibratedClassifier:
    """Gaussian likelihoods and class priors frozen from independent controls.

    Held-out confusion rows are truth, columns are predictions. Posterior
    probabilities remain model-dependent; held-out precision is separately saved.
    """
    def __init__(self, record):
        if tuple(record['features']) != FEATURES or tuple(record['classes']) != CLASSES:
            raise ValueError('Classifier feature/class definition mismatch')
        n, d = len(CLASSES), len(FEATURES)
        self.mean = finite(record['mean'], (n, d)); self.cov = finite(record['covariance'], (n, d, d))
        if not np.allclose(self.cov,np.swapaxes(self.cov,-1,-2)) or np.any(np.linalg.eigvalsh(self.cov) <= 0): raise ValueError('Classifier covariance must be positive definite')
        self.prior = finite(record['prior'], (n,)); self.confusion = finite(record['heldout_confusion_counts'], (n,n))
        if np.any(self.prior <= 0) or not np.isclose(self.prior.sum(), 1) or np.any(self.confusion < 0): raise ValueError('Invalid calibrated class populations')
        if not record.get('independent_control_hashes'): raise ValueError('Classifier must retain independent control provenance')
        self.cholesky=np.linalg.cholesky(self.cov)
        self.log_determinant=2*np.log(np.diagonal(self.cholesky,axis1=1,axis2=2)).sum(axis=1)
        self.record = record
        self.lower = finite(record['feature_lower'], (d,)); self.upper = finite(record['feature_upper'], (d,))
        if np.any(self.lower>self.upper):raise ValueError('Classifier feature bounds must be ordered')
        if np.any(self.confusion!=np.floor(self.confusion)):raise ValueError('Heldout confusion values must be actual integer counts')
        self.level=float(finite(record.get('calibration_confidence_level',.95),(),'calibration confidence level'))
        if not 0<self.level<1:raise ValueError('Calibration confidence level must lie between zero and one')
        self.precision=[];self.precision_lower=[];self.recall=[]
        for k in range(n):
            total=self.confusion[:,k].sum();correct=self.confusion[k,k];truth=self.confusion[k].sum()
            self.precision.append(correct/total if total else 0.)
            self.precision_lower.append(float(beta.ppf((1-self.level)/2,correct+.5,total-correct+.5)) if total else 0.)
            self.recall.append(correct/truth if truth else None)

    def classify(self, feature, controls):
        feature=np.asarray(feature,float)
        if feature.shape!=(len(FEATURES),):raise ValueError('One complete aligned classifier feature vector required')
        thresholds=[]
        for name in ('minimum_posterior','minimum_heldout_precision'):
            value=float(finite(controls[name],(),name))
            if not 0<=value<=1:raise ValueError(name+' must be a probability')
            thresholds.append(value)
        if not np.isfinite(feature).all() or np.any(feature < self.lower) or np.any(feature > self.upper):
            return {'classification': 'ambiguous', 'status': 'OUTSIDE_CALIBRATION', 'posterior': None}
        scores = []
        for k in range(len(CLASSES)):
            delta = feature-self.mean[k]
            # Mahalanobis distance uses the cached covariance factor: |L^-1 delta|^2.
            whitened=solve_triangular(self.cholesky[k],delta,lower=True)
            scores.append(np.log(self.prior[k])-.5*(np.dot(whitened,whitened)+self.log_determinant[k]))
        scores=finite(scores,(len(CLASSES),),'classifier log scores')
        posterior = np.exp(scores-logsumexp(scores)); best = int(np.argmax(posterior))
        count=self.confusion[:,best].sum()
        precision=self.precision[best];precision_lower=self.precision_lower[best]
        recall=self.recall[best]
        # Zero heldout predictions are unavailable evidence, even at threshold0.
        accepted=count>0 and posterior[best]>=thresholds[0] and precision_lower>=thresholds[1]
        return {'classification': CLASSES[best] if accepted else 'ambiguous', 'best_alternative': CLASSES[best],
                'status': 'CALIBRATED_DOMAIN' if accepted else 'LOW_CONFIDENCE', 'posterior': posterior,
                'heldout_precision': precision, 'heldout_precision_lower_bound':precision_lower,
                'calibration_confidence_level':self.level,'heldout_recall_completeness':recall,'heldout_count':count}


def fit_classifier_controls(features, labels, *, heldout_features, heldout_labels, ridge, control_hashes):
    """Authorable calibration job; callers supply disjoint evolved controls."""
    x = finite(features); y = np.asarray(labels); hx = finite(heldout_features); hy = np.asarray(heldout_labels)
    if x.ndim!=2 or hx.ndim!=2 or x.shape[1] != len(FEATURES) or hx.shape[1] != len(FEATURES) or y.shape!=(len(x),) or hy.shape!=(len(hx),): raise ValueError('Calibration shape mismatch')
    if set(y) != set(CLASSES) or not set(hy) <= set(CLASSES): raise ValueError('All physical alternatives require calibration examples')
    ridge=positive(ridge,'ridge')
    means=[]; cov=[]; prior=[]
    for label in CLASSES:
        rows=x[y==label]
        if len(rows)<len(FEATURES)+2: raise ValueError('Insufficient independent controls for '+label)
        means.append(rows.mean(axis=0)); cov.append(np.cov(rows.T)+ridge*np.eye(len(FEATURES))); prior.append(len(rows)/len(x))
    record={'features':FEATURES,'classes':CLASSES,'mean':means,'covariance':cov,'prior':prior,
        'heldout_confusion_counts':np.zeros((len(CLASSES),len(CLASSES)),dtype=np.int64), 'feature_lower':x.min(axis=0),
        'feature_upper':x.max(axis=0),'independent_control_hashes':control_hashes}
    model=CalibratedClassifier(record); confusion=np.zeros((len(CLASSES),len(CLASSES)),int); outside=0
    for feature, truth in zip(hx,hy):
        prediction=model.classify(feature,{'minimum_posterior':0.,'minimum_heldout_precision':0.})
        if prediction['posterior'] is None: outside+=1; continue
        confusion[CLASSES.index(truth),int(np.argmax(prediction['posterior']))]+=1
    record['heldout_confusion_counts']=confusion;record['heldout_outside_count']=outside
    return record


def build_stream_catalogue(sample, request, *, ancestry=None, calibration=None):
    x,v,m,ids=phase_space(sample); n=len(x)
    if ancestry is None:
        ancestry={'labels':np.full(n,-1,np.int64),'records':[], 'keys':[],
                  'current_bound':np.zeros(n,bool),'binding_known':np.zeros(n,bool)}
    for name in ('current_bound','binding_known'):
        value=np.asarray(ancestry[name])
        if value.shape!=(n,) or value.dtype.kind!='b':raise ValueError('Exact aligned '+name+' mask required')
    progenitor_labels=np.asarray(ancestry['labels'])
    if progenitor_labels.shape!=(n,) or progenitor_labels.dtype.kind not in 'iu' or np.any(progenitor_labels < -1) or np.any(progenitor_labels>=len(ancestry['keys'])):
        raise ValueError('Progenitor labels must index actual recorded owner keys or be unavailable (-1)')
    ancestry=dict(ancestry)
    ancestry['labels']=progenitor_labels
    ancestry['current_bound']=np.asarray(ancestry['current_bound'])
    ancestry['binding_known']=np.asarray(ancestry['binding_known'])
    candidates, labels=discover_coherent_debris(sample,request,ancestry_labels=ancestry['labels'])
    model=CalibratedClassifier(calibration) if calibration is not None else None
    rows=[]; features=[]
    for candidate in candidates:
        index=candidate['rows']
        feature,morphology=fit_stream_shell_tail_alternatives(x[index],v[index],m[index],
            bound=ancestry['current_bound'][index],binding_known=ancestry['binding_known'][index],progenitor_labels=ancestry['labels'][index])
        decision=model.classify(feature,request) if model else {'classification':'ambiguous','status':'CALIBRATION_UNAVAILABLE'}
        if not candidate['resolution_supported']: decision={'classification':'ambiguous','status':'INSUFFICIENT_RESOLUTION'}
        rows.append({k:v for k,v in candidate.items() if k!='rows'} | {'particle_count':len(index),'mass_msun':m[index].sum(),
            'classification':decision,'morphology':morphology,'wrap_status':'REQUIRES_TRACK_AND_PHASE_HISTORY'})
        features.append(feature)
    return {'schema':'tidal-stream-catalogue-v1','candidates':rows,'features':FEATURES,
            'ancestry':ancestry['records'],'progenitor_keys':ancestry['keys'], 'classification_calibration':calibration,
            'progenitor_identity_scope':ancestry.get('identity_scope','unavailable'),
            'feature_units':['dimensionless','dimensionless','dimensionless','ln(width / proper kpc)','ln(dispersion / km/s)',
                'dimensionless','dimensionless','dimensionless','dimensionless','dimensionless','ln(linear density / (Msun/proper kpc))'],
            'empty_selection':not bool(rows)}, {'position_kpc':x,'velocity_kms':v,'mass_msun':m,'particle_ids':ids,
            'candidate_id':labels,'progenitor_label':ancestry['labels'], 'bound':ancestry['current_bound'],
            'binding_known':ancestry['binding_known'], 'candidate_features':np.asarray(features).reshape(-1,len(FEATURES))}
