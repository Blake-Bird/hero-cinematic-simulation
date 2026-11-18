"""Matched cosmology/object comparisons, population statistics and workflows.

Compare common Lagrangian identities with explicit non-h-scaled physical units.
Within-realization object scatter is distinct from independent-realization
covariance. WDM discreteness screens describe numerical reliability, not a
physical cutoff. EDE contrasts refer to background/linear-input differences;
this module does not introduce a nonlinear EDE field into Gadget.
"""
from copy import deepcopy
from pathlib import Path
import json
import math
import numpy as np
from .data import exact_uint64_ids
from .foundation import atomic_json, canonical_digest, file_sha256
from .stream_common import finite, increasing, positive, integer, digest


OBSERVABLE_FAMILIES={'power_spectrum','halo_mass_function','stellar_mass_function','vmax_function',
    'rare_high_redshift_abundance','assembly','mergers','concentration','spin','density_shape','potential_shape',
    'splashback','cosmic_web','satellites','stellar_history','black_hole_history','stream_width','stream_heating',
    'stream_gaps','stream_actions'}



def require_matched_run_family(runs,request):
    names=[row['run_id'] for row in runs]
    if len(names)!=len(set(names)):raise ValueError('Unique matched run identities required')
    reference=next(row for row in runs if row['run_id']==request['reference_run'])
    for row in runs:
        for key in ('lagrangian_namespace','primordial_phases_sha256','box_comoving_mpc','resolution_signature'):
            if row[key]!=reference[key]:raise ValueError('Matched comparison lacks common '+key)
        if row['mass_units']!='Msun_no_h' or row['length_units']!='comoving_Mpc_no_h':raise ValueError('Common explicit non-h-scaled comparison units required')
        changed=[axis for axis in ('background_signature','linear_transfer_signature','interaction_signature') if row[axis]!=reference[axis]]
        declared=row['contrast_axes']
        if set(changed)!=set(declared):raise ValueError('Background/cutoff/interaction changes differ from declared contrast')
        if declared==['interaction_signature'] and row['linear_transfer_signature']!=reference['linear_transfer_signature']:
            raise ValueError('Interaction-only CDM/SIDM control must share linear inputs')
        if not row['source_hashes']:raise ValueError('Actual run provenance required')
    return reference



def _patch(record):
    ids=exact_uint64_ids(record['lagrangian_ids'],allow_empty=False)
    mass=finite(record['lagrangian_mass_msun'],(len(ids),))
    if len(np.unique(ids))!=len(ids) or np.any(mass<=0):raise ValueError('Unique parent patch IDs with positive common initial masses required')
    return {int(identifier):float(weight) for identifier,weight in zip(ids,mass)}



def match_lagrangian_patches(left,right,request):
    """One-to-one global matching with retained ambiguity/split/merge evidence."""
    from scipy.optimize import linear_sum_assignment
    if len(left)*len(right)>integer(request['maximum_match_cells'],'maximum_match_cells'):raise MemoryError('Lagrangian assignment matrix exceeds request limit')
    lpatch=[_patch(q) for q in left];rpatch=[_patch(q) for q in right]
    # Patch masses are fixed across candidate pairs. Compute each total once
    # for J = M_shared / (M_left + M_right - M_shared).
    left_mass=[sum(patch.values()) for patch in lpatch]
    right_mass=[sum(patch.values()) for patch in rpatch]
    inverted={}
    for j,patch in enumerate(rpatch):
        for identifier in patch:inverted.setdefault(identifier,[]).append(j)
    score=np.zeros((len(left),len(right)));left_fraction=np.zeros_like(score);right_fraction=np.zeros_like(score)
    for i,patch in enumerate(lpatch):
        common={}
        for identifier,mass in patch.items():
            for j in inverted.get(identifier,[]):
                if not np.isclose(mass,rpatch[j][identifier],rtol=request['initial_mass_relative_tolerance'],atol=0):
                    raise ValueError('Common Lagrangian representative masses disagree; supply exact parent-IC equivalence mapping')
                common[j]=common.get(j,0.)+mass
        for j,overlap in common.items():
            a=left_mass[i];b=right_mass[j]
            score[i,j]=overlap/(a+b-overlap);left_fraction[i,j]=overlap/a;right_fraction[i,j]=overlap/b
    matches=[];unresolved=[];assigned_left=set();assigned_right=set()
    if score.size:
        rows,columns=linear_sum_assignment(-score)
        for i,j in zip(rows,columns):
            alternatives=np.r_[np.delete(score[i],j),np.delete(score[:,j],i)]
            margin=score[i,j]-(alternatives.max() if len(alternatives) else 0.)
            accepted=score[i,j]>=request['minimum_jaccard'] and margin>=request['minimum_assignment_margin']
            row={'left_identity':left[i]['identity'],'right_identity':right[j]['identity'],'left_row':int(i),'right_row':int(j),
                'jaccard_mass':float(score[i,j]),'left_overlap_fraction':float(left_fraction[i,j]),'right_overlap_fraction':float(right_fraction[i,j]),
                'assignment_margin':float(margin),'status':'MATCHED' if accepted else 'AMBIGUOUS_OR_UNMATCHED'}
            (matches if accepted else unresolved).append(row)
            if accepted:assigned_left.add(int(i));assigned_right.add(int(j))
    return {'schema':'lagrangian-object-matches-v1','matches':matches,'unresolved':unresolved,
        'unmatched_left':[q['identity'] for i,q in enumerate(left) if i not in assigned_left],
        'unmatched_right':[q['identity'] for j,q in enumerate(right) if j not in assigned_right],
        'method':'common parent-IC mass-weighted Jaccard, global assignment and ambiguity margin'}, {
        'matching_jaccard':score,'left_overlap_fraction':left_fraction,'right_overlap_fraction':right_fraction}



def wdm_reliability(record,objects,controls):
    """Wang–White discreteness scale plus recorded Lagrangian-shape screening."""
    if record['linear_model']!='wdm':return np.ones(len(objects),bool),{'scope':'not a WDM sample'}
    density=positive(record['mean_density_msun_mpc3'],'mean_density_msun_mpc3')
    spacing=positive(record['initial_spacing_comoving_mpc'],'initial_spacing_comoving_mpc')
    peak=positive(record['dimensionless_power_peak_mpc_inverse'],'dimensionless_power_peak_mpc_inverse')
    limiting=10.1*density*spacing/peak**2
    floor=positive(controls['discreteness_mass_multiplier'],'discreteness_mass_multiplier')*limiting
    accepted=[];reasons=[]
    for obj in objects:
        reason=[]
        if obj['peak_mass_msun']<floor:reason.append('below_discreteness_mass_floor')
        if obj['lagrangian_c_over_a']<controls['minimum_lagrangian_c_over_a']:reason.append('spurious_lagrangian_shape_candidate')
        if obj['bound_particle_count']<controls['minimum_bound_particles']:reason.append('particle_resolution')
        if not obj['history_quality_passed']:reason.append('unresolved_peak_mass_history')
        accepted.append(not reason);reasons.append({'identity':obj['identity'],'reasons':reason})
    return np.asarray(accepted,bool),{'wang_white_mass_msun':limiting,'adopted_mass_floor_msun':floor,'controls':controls,'objects':reasons,
        'interpretation':'numerical reliability screen, not a universal astrophysical cutoff'}



def compare_common_observables(left,right,matches,definitions,*,left_reliable,right_reliable,matching_covariance):
    results={};arrays={};units={}
    for name,contract in definitions.items():
        if contract['family'] not in OBSERVABLE_FAMILIES:raise ValueError('Unknown declared comparison family')
        values=[];errors=[];identities=[];rejected=[]
        for pair in matches['matches']:
            i=pair['left_row'];j=pair['right_row'];a=left[i]['observables'].get(name);b=right[j]['observables'].get(name)
            reason=None
            if not left_reliable[i] or not right_reliable[j]:reason='numerical_reliability'
            elif a is None or b is None:reason='missing_measurement'
            elif a['definition_hash']!=contract['definition_hash'] or b['definition_hash']!=contract['definition_hash']:reason='different_estimator_definition'
            elif a['units']!=contract['units'] or b['units']!=contract['units']:reason='different_units'
            elif not np.all(a['supported']) or not np.all(b['supported']):reason='failed_or_unresolved_bin'
            if reason:rejected.append({'pair':pair,'reason':reason});continue
            x=finite(a['value']);y=finite(b['value'],x.shape)
            covariance=finite(a['covariance'],(x.size,x.size))+finite(b['covariance'],(x.size,x.size))
            if 'paired_measurement_covariance' in pair:
                cross=finite(pair['paired_measurement_covariance'][name],covariance.shape);covariance-=cross+cross.T
            if not np.allclose(covariance,covariance.T) or np.linalg.eigvalsh(covariance).min()<-1e-10:raise ValueError('Paired measurement covariance is not PSD')
            values.append((y-x).ravel());errors.append(covariance);identities.append([pair['left_identity'],pair['right_identity']])
        if not values:
            results[name]={'status':'NO_SUPPORTED_MATCHES','rejected':rejected,'definition':contract};continue
        delta=np.asarray(values);n,p=delta.shape;systematic=finite(matching_covariance[name],(p,p))
        if not np.allclose(systematic,systematic.T) or np.linalg.eigvalsh(systematic).min()<-1e-10:raise ValueError('Matching covariance must be positive semidefinite')
        mean=delta.mean(axis=0)
        sample=np.atleast_2d(np.cov(delta,rowvar=False))/n if n>1 else np.full((p,p),np.nan)
        measurement=np.sum(errors,axis=0)/n**2
        for suffix in ('matched_delta','mean_delta'):units[name+'__'+suffix]=contract['units']
        for suffix in ('object_sampling_covariance','measurement_covariance','matching_covariance'):units[name+'__'+suffix]='('+contract['units']+')^2'
        arrays[name+'__matched_delta']=delta;arrays[name+'__mean_delta']=mean
        arrays[name+'__object_sampling_covariance']=sample;arrays[name+'__measurement_covariance']=measurement;arrays[name+'__matching_covariance']=systematic
        results[name]={'status':'MEASURED','pair_identities':identities,'definition':contract,'rejected':rejected,
            'independent_realizations':False,'covariance_scope':'within-one-realization matched objects; ensemble covariance is separate'}
    return {'schema':'matched-cosmology-observables-v1','observables':results,'units':units,'science_acceptance':'NOT_ACCEPTED'},arrays



def ensemble_matched_covariance(realization_products,*,minimum_realizations):
    if len(realization_products)<max(2,minimum_realizations):raise ValueError('Independent realization covariance requires the declared ensemble size')
    if len({row['realization_id'] for row in realization_products})!=len(realization_products):raise ValueError('Repeated realization would inflate ensemble covariance')
    values=finite([row['mean_difference'] for row in realization_products]);cov=np.atleast_2d(np.cov(values,rowvar=False))
    return {'ensemble_mean':values.mean(axis=0),'realization_covariance':cov,'ensemble_mean_covariance':cov/len(values),
        'independent_realization_count':len(values)}



def stream_thermometer(physical,*,numerical_controls,measurement_covariance,maximum_numerical_fraction):
    """Subtract declared matched numerical heating only with controlled support."""
    measured=finite(physical);control=finite(numerical_controls)
    if control.ndim!=2 or len(control)<2 or control.shape[1:]!=measured.shape:raise ValueError('Numerical heating controls require independent matched vectors')
    numerical=control.mean(axis=0);noise=np.atleast_2d(np.cov(control,rowvar=False))/len(control)
    covariance=finite(measurement_covariance,(measured.size,measured.size))+noise
    fraction=np.divide(abs(numerical),abs(measured),out=np.full_like(measured,np.inf),where=abs(measured)>0)
    supported=fraction<=maximum_numerical_fraction
    return {'physical_heating_estimate':measured-numerical,'numerical_heating_estimate':numerical,
        'heating_covariance':covariance,'thermometer_supported':supported,'numerical_heating_fraction':fraction,
        'interpretation':'matched numerical-control correction; no unique DM-temperature inference from width alone'}



def population_function(values,edges,*,volume_comoving_mpc3,reliable,selection_probability=None):
    """Inverse-inclusion weighted abundance per comoving Mpc^3 per dex.

    Each selected object contributes 1/p; Poisson shot variance sums 1/p^2.
    These conditional count errors exclude clustering and selection-model
    uncertainty. Cumulative thresholds include objects above the display range.
    """
    value=np.asarray(values,float);edge=increasing(edges,'population edges');mask=np.asarray(reliable)
    if mask.dtype.kind!='b':raise ValueError('Explicit boolean population reliability mask required')
    if value.ndim!=1 or mask.shape!=value.shape or np.any(~np.isfinite(value[mask])) or np.any(value[mask]<=0) or np.any(edge<=0):raise ValueError('Positive population quantity and exact reliability mask required')
    probability=np.ones(len(value)) if selection_probability is None else finite(selection_probability,value.shape)
    if np.any((probability<=0)|(probability>1)):raise ValueError('Population inclusion probabilities must be in (0,1]')
    weight=1/probability;volume=positive(volume_comoving_mpc3,'comoving census volume');width=np.diff(np.log10(edge))
    count=np.histogram(value[mask],edge,weights=weight[mask])[0];variance=np.histogram(value[mask],edge,weights=weight[mask]**2)[0]
    # Cumulative threshold includes measured objects above the top plotting bin.
    cumulative=np.array([weight[mask&(value>=q)].sum() for q in edge[:-1]])
    cumulative_var=np.array([(weight[mask&(value>=q)]**2).sum() for q in edge[:-1]])
    return {'quantity_bin_center':np.sqrt(edge[:-1]*edge[1:]),'quantity_bin_edges':edge,'weighted_count':count,
        'differential_number_density_mpc3_dex':count/(volume*width),'differential_shot_variance':variance/(volume*width)**2,
        'cumulative_number_density_mpc3':cumulative/volume,'cumulative_shot_variance':cumulative_var/volume**2,
        'population_bin_supported':count>0,'raw_selected_count':np.histogram(value[mask],edge)[0]}



def cic_power_spectrum(position_comoving_mpc,mass_msun,request):
    x=finite(position_comoving_mpc);mass=finite(mass_msun,(len(x),));box=positive(request['box_comoving_mpc'],'box');ngrid=integer(request['mesh_size'],'mesh',4)
    if x.shape!=(len(mass),3) or np.any(mass<=0) or not len(mass):raise ValueError('Nonempty actual positive native gravitating mass census required')
    if ngrid**3*80>request['memory_gib']*1024**3:raise MemoryError('Density/FFT mesh exceeds workspace')
    cell=box/ngrid;u=np.mod(x,box)/cell;base=np.floor(u).astype(np.int64);fraction=u-base;grid=np.zeros((ngrid,ngrid,ngrid))
    for i in (0,1):
        for j in (0,1):
            for k in (0,1):
                index=(base+np.array([i,j,k]))%ngrid
                weight=mass*np.prod(np.where(np.array([i,j,k]),fraction,1-fraction),axis=1)
                np.add.at(grid,tuple(index.T),weight)
    delta=grid/grid.mean()-1;transform=np.fft.rfftn(delta)/ngrid**3
    component=2*np.pi*np.fft.fftfreq(ngrid,d=cell);half=2*np.pi*np.fft.rfftfreq(ngrid,d=cell)
    kx,ky,kz=np.meshgrid(component,component,half,indexing='ij');kr=np.sqrt(kx*kx+ky*ky+kz*kz)
    window=np.sinc(kx*cell/(2*np.pi))**2*np.sinc(ky*cell/(2*np.pi))**2*np.sinc(kz*cell/(2*np.pi))**2
    measured=abs(transform/window)**2*box**3;shot=box**3*np.dot(mass,mass)/mass.sum()**2
    edges=increasing(request['k_edges_mpc_inverse'],'k bins');multiplicity=np.ones(kr.shape);multiplicity[:,:,1:]=2
    if ngrid%2==0:multiplicity[:,:,-1]=1
    use=(kr>0)&(kr<=request['maximum_nyquist_fraction']*np.pi/cell)
    modes=np.histogram(kr[use],edges,weights=multiplicity[use])[0];weighted=np.histogram(kr[use],edges,weights=measured[use]*multiplicity[use])[0]
    spectrum=np.divide(weighted,modes,out=np.full(len(modes),np.nan),where=modes>0)
    return ({'schema':'native-density-power-v1','request':request,'shot_noise_mpc3':shot,'shot_noise_model':'mass-weighted Poisson subtraction; clustered discreteness/aliasing require independent convergence',
        'units':{'k_mpc_inverse':'1 / comoving Mpc no h','power_mpc3':'comoving Mpc^3 no h','raw_power_mpc3':'comoving Mpc^3 no h'},'science_acceptance':'NOT_ACCEPTED'},
        {'k_mpc_inverse':np.sqrt(edges[:-1]*edges[1:]),'raw_power_mpc3':spectrum,'power_mpc3':spectrum-shot if request['subtract_poisson_shot_noise'] else spectrum,
        'mode_count':modes,'power_supported':modes>=request['minimum_modes'],'k_edges_mpc_inverse':edges})



def contrast_population_functions(left,right,request):
    if not np.array_equal(left['quantity_bin_edges'],right['quantity_bin_edges']):raise ValueError('Population contrasts require identical bin definitions')
    x=np.asarray(left['differential_number_density_mpc3_dex']);y=np.asarray(right['differential_number_density_mpc3_dex'])
    support=left['population_bin_supported']&right['population_bin_supported']&(x>0)
    ratio=np.divide(y,x,out=np.full_like(y,np.nan),where=support)
    covariance=finite(request['paired_count_covariance'],(len(x),len(x)))
    difference=y-x;total=np.diag(left['differential_shot_variance']+right['differential_shot_variance'])-covariance-covariance.T
    if np.any(np.linalg.eigvalsh(total)<-request['covariance_tolerance']):raise ValueError('Paired abundance covariance inconsistent with measured variances')
    return {'population_ratio':ratio,'population_difference_mpc3_dex':difference,'population_difference_covariance':total,'comparison_supported':support}



def false_discovery_control(pvalues, *, rate, arbitrary_dependence=True):
    """Return the FDR selection mask and adjusted probabilities in input order.

    For sorted p_(i), q_(i) = min_{j >= i}(n*c_n*p_(j)/j), capped at one.
    Benjamini--Yekutieli uses c_n = sum(1/i) for arbitrary dependence;
    c_n = 1 is Benjamini--Hochberg and requires independent or suitably
    positively dependent null tests. Both p and the requested rate are unitless.
    """
    pvalues = finite(pvalues, name='null probabilities')
    rate = positive(rate, 'false discovery rate')
    if pvalues.ndim != 1 or np.any((pvalues < 0) | (pvalues > 1)) or rate >= 1:
        raise ValueError('One-dimensional probabilities in [0, 1] and an FDR rate in (0, 1) required')
    if type(arbitrary_dependence) is not bool:
        raise ValueError('Declare arbitrary_dependence as a Boolean')
    count = len(pvalues)
    if not count:
        return np.empty(0, bool), np.empty(0)
    ranks = np.arange(1, count + 1)
    order = np.argsort(pvalues)
    correction = np.sum(1 / ranks) if arbitrary_dependence else 1.
    ranked_qvalues = pvalues[order] * count * correction / ranks
    # The reverse cumulative minimum makes the adjusted values monotone.
    adjusted = np.minimum.accumulate(ranked_qvalues[::-1])[::-1].clip(0, 1)
    qvalues = np.empty(count)
    qvalues[order] = adjusted
    return qvalues <= rate, qvalues


def discover_outliers(records, request):
    """Compare reliable candidates with independent reference distributions.

    Values and reference means share one physical unit; each variance has its
    square. The additive error model assumes the three error contributions
    have no cross covariance: z = (value - mean) / sqrt(sum(variances)).
    Empirical controls must be comparable independent standardized residuals.
    Selection weights describe inclusion/abundance, not a correction to null
    probabilities. FDR selection identifies candidates, not confirmed physics.
    """
    pvalues = []
    rows = []
    for record in records:
        if not record['source_hashes']:
            raise ValueError('Discovery record requires actual product provenance')
        selection = positive(record['selection_probability'], 'selection_probability')
        if selection > 1:
            raise ValueError('Selection probability exceeds one')
        for key in ('reliability_passed', 'reference_independent'):
            if type(record[key]) is not bool:
                raise ValueError(key + ' must be a Boolean')
        if not record['reliability_passed'] or not record['reference_independent']:
            rows.append({**record, 'discovery_status': 'UNRESOLVED_REFERENCE_OR_RELIABILITY', 'pvalue': None})
            continue
        value, mean = finite([record['value'], record['reference_mean']], (2,), 'candidate and reference mean')
        variances = finite([record['measurement_variance'], record['reference_variance'],
                           record['model_error_variance']], (3,), 'discovery variances')
        if np.any(variances < 0):
            raise ValueError('Individual discovery variances cannot be negative')
        variance = positive(variances.sum(), 'total discovery variance')
        standardized = float((value - mean) / np.sqrt(variance))
        if not math.isfinite(standardized):
            raise ValueError('Standardized discovery residual must be finite')
        if record['reference_distribution'] == 'gaussian':
            # Load the optional statistical backend only for this null model.
            from scipy.stats import norm
            pvalue = float(2 * norm.sf(abs(standardized)))
        elif record['reference_distribution'] == 'independent_empirical':
            controls = finite(record['reference_standardized_residuals'], name='empirical null residuals')
            if controls.ndim != 1 or not len(controls):
                raise ValueError('Empirical null requires a nonempty vector of independent controls')
            # The plus-one rank estimate avoids a zero probability merely
            # because a finite control sample contains no more extreme value.
            pvalue = float((1 + np.sum(abs(controls) >= abs(standardized))) / (len(controls) + 1))
        else:
            raise ValueError('Declare a Gaussian or independent empirical null')
        pvalues.append(pvalue)
        rows.append({**record, 'standardized_residual': standardized, 'pvalue': pvalue,
                     'inverse_selection_weight': 1 / selection, 'discovery_status': 'PENDING_MULTIPLE_TESTING'})
    rejected, qvalues = false_discovery_control(
        pvalues, rate=request['false_discovery_rate'], arbitrary_dependence=request['arbitrary_dependence'])
    tested_rows = (row for row in rows if row['pvalue'] is not None)
    for index, row in enumerate(tested_rows):
        row['adjusted_qvalue'] = float(qvalues[index])
        row['discovery_status'] = 'FDR_SELECTED_CANDIDATE' if rejected[index] else 'NOT_SELECTED'
    return {'schema': 'cosmology-discovery-catalogue-v1', 'objects': rows, 'multiple_testing': request,
            'selection_scope': 'recorded survey/halo inclusion probability; weighted abundance separate from null significance',
            'science_acceptance': 'NOT_ACCEPTED'}, {
                'tested_pvalue': np.asarray(pvalues), 'adjusted_qvalue': qvalues, 'selected': rejected}


MATCHED_KINDS={'matched_cosmologies','matched_ensemble','discovery_catalogue','stream_thermometer','density_power','population_function','population_contrast','observable_convergence'}



def execute_matched_stage(stage,inputs,working):
    kind=stage['kind'];p=stage['parameters'];refs=stage['inputs']
    if kind=='observable_convergence':
        from core.research.convergence import compare_observable_convergence
        records=inputs.json(refs['measurements'])
        for row in records:
            for name,expected in row['source_hashes'].items():
                if digest(inputs.path(name))!=expected:raise ValueError('Convergence measured source changed')
        report,arrays=compare_observable_convergence(records,p['controls'])
    elif kind=='density_power':
        if 'native_run_directory' in refs:
            from .observation_geometry import read_native_gravitating_particles
            report_native,values=read_native_gravitating_particles(inputs.path(refs['native_run_directory']),p['snapshot'],species=p['species'],maximum_particles=inputs.resources['maximum_particles'],memory_gib=inputs.resources['memory_gib'],namespace=p['namespace'])
            particles={**report_native,**values,'cosmology_signature':inputs.json(refs['cosmology_signature'])};inputs.sources.update(report_native['source_hashes'])
        else:particles=inputs.json(refs['particles'])
        if particles['position_units']!='comoving_Mpc_no_h':raise ValueError('Native density mesh needs explicit common comoving no-h units')
        if not particles['source_hashes']:raise ValueError('Native density measurements need source provenance')
        for name,expected in particles['source_hashes'].items():
            if digest(inputs.path(name))!=expected:raise ValueError('Native density source changed')
        report,arrays=cic_power_spectrum(particles['position_comoving_mpc'],particles['mass_msun'],{**p['spectrum'],'memory_gib':inputs.resources['memory_gib']})
        report['cosmology_signature']=particles['cosmology_signature']
    elif kind=='population_function':
        catalogue=inputs.json(refs['catalogue']);rows=catalogue['objects'];quantity=p['quantity']
        if not catalogue['source_hashes']:raise ValueError('Native population measurements need source provenance')
        for name,expected in catalogue['source_hashes'].items():
            if digest(inputs.path(name))!=expected:raise ValueError('Native population source changed')
        reliable,screen=wdm_reliability(catalogue['run'],rows,p['wdm_reliability'])
        values=[q[quantity] for q in rows];available=np.asarray([q['measurement_supported'][quantity] for q in rows],bool)
        arrays=population_function(values,p['edges'],volume_comoving_mpc3=catalogue['run']['box_comoving_mpc']**3,reliable=reliable&available)
        report={'schema':'native-population-function-v1','quantity':quantity,'units_of_quantity':p['quantity_units'],'definition_hash':p['definition_hash'],'units':{'quantity_bin_edges':p['quantity_units'],'quantity_bin_center':p['quantity_units'],'differential_shot_variance':'(count / comoving Mpc^3 / dex)^2','cumulative_shot_variance':'(count / comoving Mpc^3)^2'},'reliability':screen,'cosmology_signature':catalogue['run'],'science_acceptance':'NOT_ACCEPTED'}
    elif kind=='population_contrast':
        l,left=inputs.product(refs['left']);r,right=inputs.product(refs['right'])
        if l['definition_hash']!=r['definition_hash']:raise ValueError('Abundance definitions differ')
        arrays=contrast_population_functions(left,right,p['controls']);report={'schema':'matched-population-contrast-v1','left':l,'right':r,'science_acceptance':'NOT_ACCEPTED'}
    elif kind=='matched_cosmologies':
        runs=inputs.json(refs['run_family']);reference=require_matched_run_family(runs,p['family'])
        left=inputs.json(reference['object_catalogue']);results=[];arrays={};units={}
        left_quality,left_screen=wdm_reliability(reference,left,p['wdm_reliability'])
        for run in runs:
            for path,expected in run['source_hashes'].items():
                actual=inputs.path(path)
                if digest(actual)!=expected:raise ValueError('Matched native run source changed')
            if run['run_id']==reference['run_id']:continue
            right=inputs.json(run['object_catalogue']);right_quality,right_screen=wdm_reliability(run,right,p['wdm_reliability'])
            matches,match_arrays=match_lagrangian_patches(left,right,p['matching'])
            comparison,values=compare_common_observables(left,right,matches,p['observable_definitions'],
                left_reliable=left_quality,right_reliable=right_quality,matching_covariance=p['matching_covariance'])
            results.append({'run_id':run['run_id'],'contrast_axes':run['contrast_axes'],'matches':matches,
                'comparison':comparison,'left_reliability':left_screen,'right_reliability':right_screen})
            units.update({run['run_id']+'__'+key:unit for key,unit in comparison.get('units',{}).items()})
            arrays.update({run['run_id']+'__'+key:value for key,value in {**match_arrays,**values}.items()})
        report={'schema':'matched-cosmology-family-v1','reference_run':reference['run_id'],'results':results,
            'run_family':runs,'units':units,'requested_definitions':p['observable_definitions'],'science_acceptance':'NOT_ACCEPTED'}
    elif kind=='matched_ensemble':
        rows=inputs.json(refs['realizations']);result=ensemble_matched_covariance(rows,**p['controls'])
        report={'schema':'matched-independent-ensemble-v1','realization_count':result.pop('independent_realization_count'),'science_acceptance':'NOT_ACCEPTED'};arrays=result
    elif kind=='stream_thermometer':
        values=inputs.json(refs['measurements']);result=stream_thermometer(**values,**p['controls'])
        report={'schema':'matched-stream-thermometer-v1','interpretation':result.pop('interpretation'),'science_acceptance':'NOT_ACCEPTED'};arrays=result
    elif kind=='discovery_catalogue':
        rows=inputs.json(refs['candidates'])
        for row in rows:
            for source,expected in row['source_hashes'].items():
                path=inputs.path(source)
                if digest(path)!=expected:raise ValueError('Discovery measurement source changed')
        report,arrays=discover_outliers(rows,p['controls'])
    else:raise ValueError('Unknown matched-comparison stage')
    return report,arrays,'matched:'+stage['stage_id'],None



CONTROL_SECTIONS={'black_holes','stellar_evolution','chemistry','hydro_model'}



def compile_matched_hydro_pilot(request,destination,*,request_base):
    from .workflow import path_at,safe_id
    if request.get('schema')!='matched-native-hydro-pilot-request-v1':raise ValueError('Matched native hydro pilot request required')
    sources={}
    if 'base_answers_reference' in request:
        if 'base_answers' in request:
            raise ValueError('Select either inline or one pinned complete native answer file')
        reference=request['base_answers_reference'];path=path_at(request_base,reference['path'])
        if path.stat().st_size>16*1024**2 or file_sha256(path)!=reference['sha256']:
            raise ValueError('Matched base native answers changed or exceed metadata intake')
        answers=json.loads(path.read_text());sources[str(path)]=reference['sha256']
    else:
        answers=request['base_answers']
    if answers.get('physics')!='hydro':raise ValueError('Matched stellar/BH pilot requires actual hydro evolution')
    variants=request['variants']
    if type(request['maximum_variants']) is not int or not isinstance(variants,list) or not 2<=len(variants)<=request['maximum_variants']:
        raise ValueError('At least two explicitly bounded matched physical variants required')
    source=path_at(request_base,request['existing_ic']['path'])
    for asset in request['existing_ic']['files']:
        path=path_at(request_base,asset['path'])
        if file_sha256(path)!=asset['sha256']:raise ValueError('Shared matched native IC shard changed')
        sources[str(path)]=asset['sha256']
    if not sources:raise ValueError('Matched pilot requires actual delivered native IC files')
    if not source.is_file() or str(source) not in sources:
        raise ValueError('Matched --existing-ic must identify an actual pinned delivered IC shard')
    phase=path_at(request_base,request['phase_asset']['path'])
    if file_sha256(phase)!=request['phase_asset']['sha256']:raise ValueError('Matched phase/noise asset changed')
    sources[str(phase)]=request['phase_asset']['sha256']
    if type(answers.get('seed')) is not int or answers['seed']<0 or any(type(answers.get(key)) is not int or answers[key]<1 for key in ('particles_per_dimension','mpi_ranks','omp_threads')):
        raise ValueError('Matched native pilot needs an exact nonnegative seed and positive resolution/CPU resources')
    for answer,key in (('mpi_ranks','mpi_ranks'),('omp_threads','omp_threads'),('memory_limit_gb','memory_gib')):
        if answers.get(answer)!=request['resources'][key]:
            raise ValueError('Matched native allocation differs from its frozen workflow resources')
    if isinstance(request['wall_seconds'],bool) or not isinstance(request['wall_seconds'],(int,float)) or not math.isfinite(request['wall_seconds']) or request['wall_seconds']<=0:
        raise ValueError('Finite positive matched native wall-time allocation required')
    snapshots=request['snapshots']
    if not isinstance(snapshots,list) or len(snapshots)<2 or len(set(snapshots))!=len(snapshots) or any(type(i) is not int or i<0 for i in snapshots):
        raise ValueError('Matched physical controls require bounded distinct saved native history indices')
    if request['comparison_snapshot'] not in snapshots:
        raise ValueError('Matched physical comparison must use a retained actual saved history selector')
    if request['comparison']['memory_gib']>request['resources']['memory_gib']:
        raise MemoryError('Matched hydro measurements exceed their frozen allocation')
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Matched hydro workflow publications are immutable')
    destination.mkdir(parents=True)
    stages=[];runs=[];names=set();variant_signatures=set();physical_sections={};galaxy_products={}
    frozen_common={key:value for key,value in answers.items() if key not in CONTROL_SECTIONS and key!='name'}
    for variant in variants:
        label=safe_id(variant['label'])
        if label in names:raise ValueError('Repeated matched physical variant label')
        names.add(label)
        patch=variant['changes']
        if not isinstance(patch,dict) or not set(patch)<=CONTROL_SECTIONS:
            raise ValueError('Matched hydro variants may change only explicitly declared physical source sections')
        current=deepcopy(answers)
        def merge(target,changes):
            for key,value in changes.items():
                if isinstance(value,dict) and isinstance(target.get(key),dict):
                    merge(target[key],value)
                else:
                    target[key]=deepcopy(value)
        merge(current,patch);current['name']=label
        sections={key:value for key,value in current.items() if key in CONTROL_SECTIONS}
        signature=canonical_digest(sections)
        if signature in variant_signatures:
            raise ValueError('Matched physical variants must actually differ in their delivered model/source settings')
        variant_signatures.add(signature);physical_sections[label]=sections
        if current.get('hydro_model','star_formation') not in {'adiabatic','cooling','star_formation'}:
            raise ValueError('Matched hydro control has an unknown native gas model')
        for key in ('black_holes','stellar_evolution','chemistry'):
            if key in current and (not isinstance(current[key],dict) or type(current[key].get('enabled')) is not bool):
                raise ValueError('Matched physical source switches require exact enabled booleans: '+key)
        if {key:value for key,value in current.items() if key not in CONTROL_SECTIONS and key!='name'}!=frozen_common:
            raise ValueError('Matched hydro variant changes phases, resolution, resources or unrelated numerical configuration')
        answer_path=destination/(label+'.answers.json');atomic_json(answer_path,current)
        sid=safe_id('native_'+label);product=safe_id('evolution_'+label)
        arguments=['--answers',{'input':str(answer_path)},'--existing-ic',{'input':str(source)},'--run-dir','{generation}/run']
        if request.get('native_generation_report'):
            asset=request['native_generation_report'];path=path_at(request_base,asset['path'])
            if file_sha256(path)!=asset['sha256']:raise ValueError('Matched native generation/phase handoff changed')
            sources[str(path)]=asset['sha256'];arguments+=['--native-generation-report',{'input':str(path)}]
        stages.append({'stage_id':sid,'kind':'public_command','parameters':{'command':'gadget','arguments':arguments,
            'wall_seconds':request['wall_seconds']},'outputs':{product:'run'},
            'input_files':[{'path':path,'sha256':digest} for path,digest in sources.items()],
            'product_contracts':{product:{'reader':'native_snapshot_series','snapshots':request['snapshots']}}})
        if request.get('galaxy_measurement') and current.get('hydro_model','star_formation')=='star_formation':
            from core.research.pilots import validate_native_galaxy_measurement
            galaxy_request={**deepcopy(request['galaxy_measurement']),
                'schema':'native-pilot-galaxy-measurement-request-v1','snapshot':request['comparison_snapshot']}
            parameters={'request':galaxy_request,'native_run_directory':{'stage':sid,'artifact':'run'},
                'native_stage_receipt':{'stage':sid,'artifact':'stage-receipt.json'},
                'destination':'{generation}/measured-galaxies'}
            validate_native_galaxy_measurement(parameters)
            if galaxy_request['memory_gib']>request['resources']['memory_gib']:
                raise MemoryError('Matched native galaxy support measurement exceeds frozen memory')
            galaxy_product=safe_id('galaxy_measurement_'+label)
            stages.append({'stage_id':safe_id('galaxies_'+label),'kind':'native_pilot_galaxy_measurement',
                'parameters':parameters,'outputs':{galaxy_product:'measured-galaxies/galaxy-measurement.json'},
                'depends_on':[sid]})
            galaxy_products[label]=galaxy_product
        runs.append({'label':label,'variant':'hydro_bh' if current.get('black_holes',{}).get('enabled',False) else 'hydro',
            'seed':current['seed'],'resolution':current['particles_per_dimension'],
            'run_directory':{'stage':sid,'artifact':'run'},'snapshot':request['comparison_snapshot'],
            'host_center_proper_kpc':request['host_center_proper_kpc'],
            'phase_asset':str(phase),'phase_sha256':request['phase_asset']['sha256']})
    if request['reference_label'] not in names:raise ValueError('Matched physical comparison reference is absent')
    comparison={**deepcopy(request['comparison']),'runs':runs,'reference_label':request['reference_label']}
    stages.append({'stage_id':'matched_hydro_measurement','kind':'matched_hydro_measurement',
        'parameters':{'request':comparison,'destination':'{generation}/matched-hydro.json'},
        'outputs':{'matched_hydro_products':'matched-hydro.json'},'depends_on':[row['stage_id'] for row in stages]})
    workflow={'schema':'research-workflow-v1','physics_spec':{'physics':'hydro',
        'star_formation':any(row.get('hydro_model','star_formation')=='star_formation' for row in physical_sections.values()),
        'black_holes':any(row.get('black_holes',{}).get('enabled',False) for row in physical_sections.values())},
        'resources':deepcopy(request['resources']),'followup_limits':deepcopy(request['followup_limits']),
        'stages':stages,'requested_products':['matched_hydro_products',*galaxy_products.values()]}
    if len(stages)>request['resources']['maximum_stage_count']:raise ValueError('Matched physical workflow exceeds declared stage allocation')
    atomic_json(destination/'workflow.json',workflow)
    report={'schema':'matched-native-hydro-pilot-design-v1','workflow':str(destination/'workflow.json'),
        'workflow_sha256':file_sha256(destination/'workflow.json'),'shared_ic_sources':sources,
        'common_answers_hash':canonical_digest(frozen_common),'physical_variants':variants,
        'delivered_physical_sections':physical_sections,'comparison_snapshot':request['comparison_snapshot'],
        'native_galaxy_measurement_products':galaxy_products,
        'execution_status':'NOT_EXECUTED','science_acceptance':'NOT_ESTABLISHED'}
    atomic_json(destination/'matched-pilot-design.json',report)
    return report



def matched_hydro_measurement_runner(parameters,context):
    from .stars import measure_hydro_comparison
    destination=Path(parameters['destination']).resolve()
    if not destination.is_relative_to(context.generation):raise ValueError('Matched hydro measurement escapes owned generation')
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Matched hydro measurement exceeds allocated workspace')
    return measure_hydro_comparison(parameters['request'],destination)



def matched_hydro_pilot_main():
    import argparse
    parser=argparse.ArgumentParser(description='Compile a matched native hydro/stellar/BH control workflow')
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.request.stat().st_size>16*1024**2:raise MemoryError('Matched hydro request exceeds bounded metadata intake')
    print(json.dumps(compile_matched_hydro_pilot(json.loads(args.request.read_text()),args.output,
        request_base=args.request.resolve().parent),indent=2))

