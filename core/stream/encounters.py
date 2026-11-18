"""Matched perturber templates and resolved compound stream encounters.

Worldlines, profiles and gravity accounting remain explicit. A bound system
needs resolved energy and trajectory support; several gaps alone cannot prove
one. Templates describe forward models. Classification and evidence remain
conditional on held-out realizations and the declared physical model family.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.signal import find_peaks
from scipy.special import logsumexp
from core.stream_common import finite, positive, integer
from core.gravity_fields import EvolvingPerturber, ComponentField
from core.stream_forward_model import (
    run_counterfactuals, integrate_stream, TrackCoordinates,
    measure_track_density_velocity, freeze_common_initial_stream_state,
)


def find_bound_nested_substructure(identity_graph, catalogues, *, minimum_bound_particles, maximum_binding_error_kms2):
    """Use recorded direct parent edges and resolved binding, never proximity.

    The portable identity graph maps persistent namespace-qualified keys to
    exact catalogue rows. Evolving trees may change parents between epochs.
    Missing history is retained as rejected evidence rather than joined across.
    """
    minimum = integer(minimum_bound_particles, 'minimum_bound_particles', 2)
    error_budget = positive(maximum_binding_error_kms2, 'maximum_binding_error_kms2')
    if identity_graph['schema'] != 'nested-identity-history-v1':
        raise ValueError('Expected exact nested identity history')
    epochs = {str(row['epoch_id']): row for row in catalogues}
    resolved = []
    rejected = []
    for edge in identity_graph['edges']:
        key = str(edge['epoch_id'])
        catalogue = epochs.get(key)
        reason = None
        if catalogue is None:
            reason = 'missing_epoch'
        else:
            members = {row['identity']: row for row in catalogue['objects']}
            parent = members.get(edge['parent'])
            child = members.get(edge['child'])
            if parent is None or child is None:
                reason = 'missing_exact_identity'
            elif child['direct_parent'] != parent['identity']:
                reason = 'inconsistent_parent_edge'
            elif child['identity'] == parent['identity']:
                reason = 'self_parent'
            elif parent.get('direct_parent') == child['identity']:
                reason = 'cyclic_parent'
            elif child['bound_particle_count'] < minimum:
                reason = 'unresolved_child'
            elif not child['binding_resolved']:
                reason = 'binding_unresolved'
            elif not np.isfinite(np.asarray([
                    child['binding_error_kms2'], child['relative_specific_energy_kms2'],
                    child['tidal_radius_kpc'], child['resolved_radius_kpc'],
                    ], dtype=float)).all():
                reason = 'nonfinite_binding_or_radius'
            elif child['binding_error_kms2'] < 0:
                reason = 'invalid_binding_error'
            elif child['tidal_radius_kpc'] <= 0 or child['resolved_radius_kpc'] < 0:
                reason = 'invalid_radius_support'
            elif child['binding_error_kms2'] > error_budget:
                reason = 'binding_error_budget'
            elif child['relative_specific_energy_kms2']+child['binding_error_kms2'] >= 0:
                reason = 'not_confidently_bound'
            elif not child.get('worldline') or not parent.get('worldline'):
                reason = 'missing_worldline'
            elif child.get('disrupted') or child.get('evaporated'):
                reason = 'disrupted_or_evaporated'
            elif child['tidal_radius_kpc'] <= child['resolved_radius_kpc']:
                reason = 'unresolved_tidal_support'
        if reason:
            rejected.append({'edge': edge, 'reason': reason})
            continue
        resolved.append({'edge': edge, 'parent': parent, 'child': child, 'epoch_id': key,
            'binding_definition': child['binding_definition'], 'source_hashes': catalogue['source_hashes']})
    return {'schema': 'resolved-nested-substructure-v1', 'resolved': resolved, 'rejected': rejected,
        'selection': 'direct tree parent + resolved confidently negative binding energy + trajectory/profile support',
        'science_acceptance': 'NOT_ACCEPTED'}


def require_resolved_binding_and_worldlines(record, times):
    parent = record['parent_component']
    child = record['child_component']
    if record['binding_status'] != 'RESOLVED_BOUND':
        raise ValueError('A genuine bound nested-system record is required')
    if not record['binding_source_hashes']:
        raise ValueError('Nested binding source provenance required')
    if not record.get('resolution_passed'):
        raise ValueError('Nested resolution gates have not passed')
    models = (EvolvingPerturber(parent), EvolvingPerturber(child))
    if models[0].line.frame != models[1].line.frame:
        raise ValueError('Nested worldlines require a common inertial frame')
    if record['profile_accounting'] != 'parent_excludes_child_mass':
        raise ValueError('Parent profile must exclude child mass; inclusive profiles double-count the compound system')
    for model in models:
        for t in times:
            model.line.interval(t)
        for k in range(len(model.line.times)-1):
            if model.line.times[k] < times[-1] and model.line.times[k+1] > times[0] and not model.line.valid[k]:
                raise ValueError('Nested history crosses missing/disruption intervals')
    if record.get('physics') == 'elastic_sidm':
        for model in models:
            if not model.record.get('evaporation_history') or not model.record.get('disruption_history'):
                raise ValueError('SIDM nested systems require actual evaporation/disruption histories')
    return models


def evolve_compound_encounter(sample, host, nested, request, *, moving_frame=None):
    """Evolve parent+child and paired controls from the same initial stream.

    The parent profile excludes child mass. Each body contributes gravity
    once; subtracting parent-only and child-only responses then adding the
    no-candidate response isolates the nonadditive dynamical interaction.
    Positions, velocities and clocks use proper kpc, km/s and Myr.
    """
    parent, child = require_resolved_binding_and_worldlines(nested, request['times_myr'])
    if any(entry['role'] == 'dark_candidate' for entry in host.accounting.values()):
        raise ValueError('Host field must exclude the nested candidates')
    if {'nested_parent', 'nested_child'} & host.components.keys():
        raise ValueError('Nested component names collide with the host field')
    components = {**host.components, 'nested_parent': parent, 'nested_child': child}
    accounting = {**host.accounting, 'nested_parent': {'gravity_count': 1, 'role': 'dark_candidate'},
        'nested_child': {'gravity_count': 1, 'role': 'dark_candidate'}}
    full = ComponentField(components, accounting=accounting)
    controls = copy.deepcopy(request['forward'])
    controls['reference_intervention'] = 'neither'
    controls['interventions'] = [
        {'name': 'compound', 'operation': 'full'},
        {'name': 'parent_only', 'operation': 'subtract', 'components': ['nested_child']},
        {'name': 'child_only', 'operation': 'subtract', 'components': ['nested_parent']},
        {'name': 'neither', 'operation': 'subtract', 'components': ['nested_parent', 'nested_child']}]
    # Four paired interventions are required; do not enlarge a declared cap.
    integer(controls['maximum_interventions'], 'maximum_interventions', 4)
    report, arrays = run_counterfactuals(sample, full, request['times_myr'], controls, moving_frame=moving_frame)
    isolated = []
    for index, matched in enumerate(nested.get('isolated_matches', [])):
        component = EvolvingPerturber(matched)
        name = 'isolated_%03d' % index
        field = ComponentField({**host.components, name: component}, accounting={**host.accounting, name: {'gravity_count': 1, 'role': 'dark_candidate'}})
        one = copy.deepcopy(controls)
        one['interventions'] = [{'name': name, 'operation': 'full'}, {'name': 'neither', 'operation': 'subtract', 'components': [name]}]
        subreport, subarrays = run_counterfactuals(sample, field, request['times_myr'], one, moving_frame=moving_frame)
        arrays.update({name+'__'+key: value for key, value in subarrays.items()})
        isolated.append(subreport)
    # Compound minus the linear sum is an actual dynamical interaction term.
    for key in ('position_kpc', 'velocity_kms', 'linear_density_msun_kpc'):
        arrays['nonadditive_'+key] = arrays['compound__'+key]-arrays['parent_only__'+key]-arrays['child_only__'+key]+arrays['neither__'+key]
    report.update({'schema': 'compound-stream-encounter-v1', 'nested_system': nested, 'isolated_matches': isolated,
        'physical_scope': 'actual separately evolved nested profiles on recorded trajectories; imposed host',
        'multiple_feature_inference': 'multiple gaps alone do not establish a bound compound encounter'})
    return report, arrays


def measure_multiple_features(density_ratio, transverse_offset, edges_kpc, *, minimum_prominence, minimum_separation_kpc):
    ratio = np.asarray(density_ratio, float)
    offset = np.asarray(transverse_offset, float)
    edges = finite(edges_kpc)
    centers = .5*(edges[:-1]+edges[1:])
    if ratio.shape != centers.shape or offset.shape != (len(centers), 2):
        raise ValueError('Feature arrays must share resolved bins')
    valid = np.isfinite(ratio)&np.isfinite(offset).all(axis=1)
    # Split contiguous support. A missing bin is not a low-density detection.
    groups = np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid)) > 1)+1)
    features = []
    for group in groups:
        if len(group) < 3:
            continue
        peaks, detail = find_peaks(1-ratio[group], prominence=minimum_prominence)
        ordered = sorted(zip(peaks, detail['prominences']), key=lambda q: -q[1])
        accepted = []
        for peak, prominence in ordered:
            b = group[peak]
            if any(abs(centers[b]-centers[other]) < minimum_separation_kpc for other in accepted):
                continue
            accepted.append(b)
            features.append({'bin': int(b), 's_kpc': float(centers[b]), 'gap_depth': float(1-ratio[b]),
                'prominence': float(prominence), 'spur_kpc': offset[b].tolist()})
    return {'features': sorted(features, key=lambda q: q['s_kpc']), 'supported_bins': valid,
        'alternative_explanations': ['repeated passage', 'independent encounters', 'epicyclic overdensity', 'selection window', 'baryonic perturber']}


def identifiability_matrix(feature_predictions, covariance, parameter_steps):
    """Whiten actual symmetric finite-difference forward predictions.

    J[:, i] = (features_plus - features_minus)/(2*parameter_step[i]).
    For C=L L.T, whiten with solve(L,J), then F=J.T C^-1 J. The SVD
    directions describe local sensitivity, not unique perturber attribution.
    """
    predictions = finite(feature_predictions)
    if predictions.ndim != 3 or predictions.shape[1] != 2:
        raise ValueError('Expected parameter by minus/plus by feature predictions')
    steps = finite(parameter_steps, (len(predictions),))
    if np.any(steps <= 0):
        raise ValueError('Positive finite-difference steps required')
    covariance = finite(covariance, (predictions.shape[2], predictions.shape[2]))
    if not np.allclose(covariance, covariance.T):
        raise ValueError('Feature covariance must be symmetric before whitening')
    root = np.linalg.cholesky(covariance)
    derivatives = (predictions[:, 1]-predictions[:, 0])/(2*steps[:, None])
    whitened = np.linalg.solve(root, derivatives.T)
    _, singular, vectors = np.linalg.svd(whitened, full_matrices=True)
    return {'fisher_information': whitened.T@whitened, 'sensitivity_singular_values': singular,
        'parameter_directions': vectors, 'meaning': 'local design identifiability, not posterior recovery or unique attribution'}


def nested_edges_from_native_graph(report):
    """Adapt block09's exact graph host edges without new identity conventions."""
    nodes = {tuple(row['key']): row for row in report['nodes']}
    edges = []
    skipped = []
    for row in report['edges']:
        if row['kind'] != 'host':
            continue
        child = tuple(row['source'])
        parent = tuple(row['target'])
        left = nodes.get(child)
        right = nodes.get(parent)
        if left is None or right is None:
            raise ValueError('Native graph host edge has missing endpoint')
        if left['kind'] != 'subhalo' or right['kind'] != 'subhalo':
            continue
        if child[2] != parent[2]:
            raise ValueError('Nested host link crosses saved epochs')
        if left['status'] != 'DETECTED' or right['status'] != 'DETECTED':
            skipped.append({'edge': row, 'reason': 'missing_or_ambiguous_native_detection'})
            continue
        edges.append({'epoch_id': str(child[2]), 'child': child[0]+':'+str(child[1]),
            'parent': parent[0]+':'+str(parent[1]), 'native_keys': {'child': list(child), 'parent': list(parent)},
            'native_evidence': row['evidence']})
    return {'schema': 'nested-identity-history-v1', 'edges': edges, 'native_graph_fingerprint': report['fingerprint'], 'skipped': skipped}


def matched_perturber_record(worldline, template):
    """Match a profile history to declared masses and the same flyby geometry.

    Mass rescaling requires an explicit homologous diagnostic policy. Such a
    profile does not become a measured evolved halo merely by being matched.
    """
    record = copy.deepcopy(template)
    record['worldline'] = copy.deepcopy(worldline)
    masses = finite(worldline['mass_msun'], (len(worldline['times_myr']),))
    profiles = record['profiles']
    if len(profiles) != len(masses):
        raise ValueError('Actual epoch profile histories required')
    for p, mass in zip(profiles, masses):
        original = positive(p['mass_msun'], 'profile mass')
        if not np.isclose(original, mass, rtol=1e-6):
            if record.get('mass_matching_policy') != 'explicit_homologous_mass_rescaling':
                raise ValueError('Mass mismatch: measured templates may not be silently rescaled')
            if p['kind'] == 'enclosed_mass':
                p['enclosed_mass_msun'] = (np.asarray(p['enclosed_mass_msun'])*mass/original).tolist()
            p['mass_msun'] = float(mass)
            record.setdefault('provenance', {})['rescaling_scope'] = 'homologous diagnostic, not actual evolved measurement'
    if record['category'] == 'black_hole':
        if any(p['kind'] not in {'point_mass', 'plummer'} for p in profiles):
            raise ValueError('Compact BH templates require supported point/softened core model')
        if not record.get('wandering_history'):
            raise ValueError('Record recoil/wandering history or explicit synthetic-control label')
    if record['category'] in {'wdm_subhalo', 'sidm_subhalo'}:
        if any(p['kind'] != 'enclosed_mass' for p in profiles) and not record['provenance'].get('justified_analytic_profile'):
            raise ValueError('Evolved WDM/SIDM profiles or explicitly justified analytic alternatives required')
    return record


def observed_phase_space(arrays, observer_record):
    from core.stars import ObserverFrame, to_observer
    observer = ObserverFrame(**observer_record)
    values = to_observer(arrays['position_kpc'][-1], arrays['velocity_kms'][-1], observer)
    return finite(values, (len(arrays['particle_ids']), 6))


def generate_template_ensemble(sample, host, designs, request, *, moving_frame=None):
    frozen = freeze_common_initial_stream_state(sample)
    count = sum(len(d['templates']) for d in designs)
    if count > integer(request['maximum_templates'], 'maximum_templates'):
        raise ValueError('Template count exceeds budget')
    memory = request['integration']['memory_gib']*1024**3
    n = len(sample['particle_ids'])
    nt = len(request['times_myr'])
    if count*n*nt*320 > memory:
        raise MemoryError('Retained template histories exceed campaign memory')
    track = TrackCoordinates(request['track'])
    records = []
    packed = {}
    if any(entry['role'] == 'dark_candidate' for entry in host.accounting.values()):
        raise ValueError('Template host must exclude all matched candidates to avoid double counting')
    for design_index, design in enumerate(designs):
        geometry = design['worldline']
        for template_index, template in enumerate(design['templates']):
            name = 'design_%04d_template_%04d' % (design_index, template_index)
            component_record = matched_perturber_record(geometry, template)
            perturber = EvolvingPerturber(component_record)
            if name in host.components:
                raise ValueError('Candidate name collides with host component')
            components = {**host.components, name: perturber}
            accounting = {**host.accounting, name: {'gravity_count': 1, 'role': 'dark_candidate' if 'subhalo' in perturber.category or perturber.category == 'black_hole' else 'gas'}}
            field = ComponentField(components, accounting=accounting)
            report, arrays = integrate_stream(frozen, field, request['times_myr'], request['integration'], moving_frame=moving_frame)
            arrays.update(measure_track_density_velocity(arrays, track, request['measurement']))
            observed = observed_phase_space(arrays, request['observer'])
            arrays['observed_phase_space'] = observed
            sensitivity = None
            if request.get('sensitivity_log_mass_step'):
                epsilon = positive(request['sensitivity_log_mass_step'], 'sensitivity_log_mass_step')
                predictions = []
                for sign in (-1, 1):
                    modified = copy.deepcopy(component_record)
                    factor = np.exp(sign*epsilon)
                    modified['worldline']['mass_msun'] = (np.asarray(modified['worldline']['mass_msun'])*factor).tolist()
                    for p in modified['profiles']:
                        p['mass_msun'] *= factor
                        if p['kind'] == 'enclosed_mass':
                            p['enclosed_mass_msun'] = (np.asarray(p['enclosed_mass_msun'])*factor).tolist()
                    trial = ComponentField({**host.components, name: EvolvingPerturber(modified)}, accounting=accounting)
                    _, a = integrate_stream(frozen, trial, request['times_myr'], request['integration'], moving_frame=moving_frame)
                    predictions.append(observed_phase_space(a, request['observer']))
                sensitivity = (predictions[1]-predictions[0])/(2*epsilon)
                # Wrap longitude differences across the observer coordinate seam.
                sensitivity[:, 0] = np.arctan2(np.sin(predictions[1][:, 0]-predictions[0][:, 0]), np.cos(predictions[1][:, 0]-predictions[0][:, 0]))/(2*epsilon)
                arrays['log_mass_sensitivity'] = sensitivity
            packed.update({name+'__'+key: value for key, value in arrays.items()})
            records.append({'template_id': name, 'design_id': design['design_id'], 'realization_group': design['realization_group'],
                'category': perturber.category, 'component': component_record, 'forward': report,
                'initial_state_sha256': frozen['state_sha256'], 'observer': request['observer'],
                'matched_geometry': geometry, 'sensitivity_computed': sensitivity is not None})
    return {'schema': 'stream-perturber-templates-v1', 'templates': records,
        'classification_status': 'UNCALIBRATED', 'science_acceptance': 'NOT_ACCEPTED',
        'confusers': request['confuser_classes'], 'phase_space_columns': ['longitude_rad', 'latitude_rad', 'distance_kpc', 'radial_velocity_kms', 'mu_longitude_star_masyr', 'mu_latitude_masyr']}, packed


class TemplateClassifier:
    """Gaussian feature classifier with independent group holdout and shrinkage."""
    def __init__(self, means, covariances, labels, prior):
        self.means = finite(means)
        self.covariances = finite(covariances)
        self.labels = tuple(labels)
        self.prior = finite(prior, (len(labels),))
        if self.means.ndim != 2 or self.covariances.shape != (len(labels), self.means.shape[1], self.means.shape[1]):
            raise ValueError('Classifier dimensions disagree')
        if np.any(self.prior <= 0) or not np.isclose(self.prior.sum(), 1):
            raise ValueError('Normalized positive class probabilities required')
        self.cholesky = []
        for covariance in self.covariances:
            if not np.allclose(covariance, covariance.T):
                raise ValueError('Class covariance must be symmetric')
            self.cholesky.append(np.linalg.cholesky(covariance))

    def probabilities(self, features):
        x = finite(features)
        scores = []
        for mean, cholesky, prior in zip(self.means, self.cholesky, self.prior):
            residual = np.linalg.solve(cholesky, (x-mean).T).T
            scores.append(-.5*np.sum(residual**2, axis=1)-np.log(np.diag(cholesky)).sum()+np.log(prior))
        scores = np.array(scores).T
        return np.exp(scores-logsumexp(scores, axis=1)[:, None])


def calibrate_template_classifier(features, class_labels, realization_groups, *, train_groups, heldout_groups, covariance_floor, shrinkage, confidence_threshold):
    x = finite(features)
    labels = np.asarray(class_labels)
    groups = np.asarray(realization_groups)
    if labels.shape != (len(x),) or groups.shape != labels.shape:
        raise ValueError('One class and independent realization group per row')
    if set(train_groups) & set(heldout_groups):
        raise ValueError('Training/heldout realizations overlap')
    floor = finite(covariance_floor, (x.shape[1], x.shape[1]))
    np.linalg.cholesky(floor)
    if not 0 <= shrinkage <= 1 or not .5 < confidence_threshold <= 1:
        raise ValueError('Invalid calibration controls')
    train = np.isin(groups, train_groups)
    heldout = np.isin(groups, heldout_groups)
    if not heldout.any():
        raise ValueError('Independent heldout cases required')
    classes = sorted(set(labels[train].tolist()))
    means = []
    covariances = []
    counts = []
    for label in classes:
        values = x[train & (labels == label)]
        if len(values) < 3 or len(set(groups[train & (labels==label)].tolist()))<3:
            raise ValueError('At least three independent training realization groups per class required')
        means.append(values.mean(axis=0))
        covariance = np.atleast_2d(np.cov(values, rowvar=False))
        covariances.append((1-shrinkage)*covariance+shrinkage*np.diag(np.diag(covariance))+floor)
        counts.append(len(values))
    if any(label not in classes for label in labels[heldout]):
        raise ValueError('Heldout class absent from training')
    classifier = TemplateClassifier(means, covariances, classes, np.asarray(counts)/sum(counts))
    probabilities = classifier.probabilities(x[heldout])
    predicted = np.argmax(probabilities, axis=1)
    confusion = np.zeros((len(classes), len(classes)+1), int)
    for truth, pred, prob in zip(labels[heldout], predicted, probabilities.max(axis=1)):
        confusion[classes.index(truth), pred if prob >= confidence_threshold else -1] += 1
    return {'schema': 'stream-template-classifier-v1', 'classes': classes, 'training_groups': train_groups, 'heldout_groups': heldout_groups,
        'confidence_threshold': confidence_threshold, 'confusion_columns': [*classes, 'ambiguous'], 'confusion_matrix': confusion,
        'classification_scope': 'heldout feature domain only; observational calibration and physical model error retained',
        'science_acceptance': 'NOT_ACCEPTED'}, {'class_mean': np.array(means), 'class_covariance': np.array(covariances),
        'class_prior': np.asarray(counts)/sum(counts), 'heldout_probability': probabilities, 'confusion_matrix': confusion}


def posterior_model_comparison(log_evidence, log_evidence_error, model_prior):
    """Combine log evidences and model priors using stable log normalization.

    Pairwise log-Bayes-factor errors add the supplied variances, assuming
    independent evidence estimates. They omit shared estimator covariance
    and physical-model uncertainty; probabilities remain prior-dependent.
    """
    evidence = finite(log_evidence)
    error = finite(log_evidence_error, evidence.shape)
    prior = finite(model_prior, evidence.shape)
    if np.any(error < 0) or np.any(prior <= 0) or not np.isclose(prior.sum(), 1):
        raise ValueError('Valid evidence errors and normalized model priors required')
    logp = evidence+np.log(prior)
    return {'model_probability': np.exp(logp-logsumexp(logp)), 'pairwise_log_bayes_factor': evidence[:, None]-evidence,
            'pairwise_log_bayes_factor_error': np.sqrt(error[:, None]**2+error[None]**2),
            'scope': 'conditional on the supplied physical model families and prior volumes'}
