"""Read, compare and export linear and measured native power spectra.

Native errors retain their estimator meaning. Matched spectra require actual
phase evidence and full joint covariance. Ensembles use distinct parent
realizations: their independence is a declared generation-design assumption,
separate from measured within-pair phase agreement and numerical convergence.
No spectra are simulated by this module.
"""
from pathlib import Path
import json
import re
import numpy as np
from .foundation import file_sha256, atomic_json
from .data import export_numeric_audit, numeric_audit_source_bindings
from core.stream_common import read_product
from core.research.display import export_scientific_suite


def _allocation(memory_gib):
    if isinstance(memory_gib,(bool,np.bool_)) or not np.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Explicit finite positive power workspace allocation required')
    return float(memory_gib)


def power_input_bindings(parameters):
    """Bind every current producer input before workflow cache reuse."""
    _allocation(parameters['memory_gib'])
    operation=parameters['operation'];path=Path(parameters['product']).resolve()
    if operation=='power-linear':
        from .initial_conditions import validate_linear_bundle
        report=validate_linear_bundle(path,memory_gib=parameters['memory_gib'])
        info=json.loads(path.read_text());relative=info['source_files']['start']['pk'];pk=(path.parent/relative).resolve()
        if not pk.is_relative_to(path.parent) or file_sha256(pk)!=info['checksums_sha256'][relative]:raise ValueError('Linear power input changed')
        return {str(path):file_sha256(path),str(pk):file_sha256(pk),
                **{row['path']:row['sha256'] for row in report['sources'].values()}}
    if operation=='power-evolved':
        from .analysis import verify_analysis_bundle
        report=verify_analysis_bundle(path,verify_sources=True)
        return {str(path/'analysis_provenance.json'):file_sha256(path/'analysis_provenance.json'),
                str(path/'analysis_products.h5'):file_sha256(path/'analysis_products.h5'),
                str(Path(report['source_run'])/'config.yaml'):file_sha256(Path(report['source_run'])/'config.yaml')}
    if operation=='power-ensemble':
        *_,sources=_read_ensemble(json.loads(path.read_text()),memory_gib=parameters['memory_gib'])
        return {str(path):file_sha256(path),**sources}
    if operation!='power-matched':raise ValueError('Known power producer operation required')
    request=json.loads(path.read_text());sources={str(path):file_sha256(path)}
    from .analysis import _initial_condition_paths
    for name in ('left','right','joint_covariance'):
        binding=request[name];product=Path(binding['path']).resolve()
        if file_sha256(product)!=binding['sha256']:raise ValueError('Matched power input changed')
        report,_,pins=bound_power_product(binding,memory_gib=parameters['memory_gib']/4);sources.update(pins)
        if name!='joint_covariance':
            run=Path(report['native_run'])
            for source in [run/'config.yaml',run/'manifest.json',*_initial_condition_paths(run)]:sources[str(source.resolve())]=file_sha256(source)
    return sources


def _publish(path,report,arrays,units,sources):
    return export_numeric_audit(path,report,arrays,
        {key:{'units':unit,'quantity':'declared power measurement '+key,
              'missing_policy':'explicit support mask; signed/nonfinite original bins retained'} for key,unit in units.items()},
        algorithm=__file__,context={'epoch':report['epoch']},
        source_hashes={**sources,str(Path(__file__).resolve()):file_sha256(__file__)})


def _figure(name,variant,source,series,xunit,yunit,epoch):
    return {'figure_id':name,'family':'cosmology','variant':variant,'title':variant.replace('_',' '),
        'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'power',
        'title':'Source measurements; scientific convergence pending','quality_status':'PENDING_CONVERGENCE',
        'selection':'original signed arrays with explicit estimator support; no positivity clipping',
        'xlabel':'k ['+xunit+']','ylabel':variant+' ['+yunit+']','x_units':xunit,'y_units':yunit,'xscale':'log',
        'series':[{'source':source,'kind':'errorbar' if row.get('error') else 'line',
                   'x':'k','support':'supported',**row} for row in series]}]}


def export_linear_power(manifest,destination,*,memory_gib=1.):
    memory_gib=_allocation(memory_gib)
    from .initial_conditions import validate_linear_bundle
    from .cosmology import class_redshift_matches, write_species_transfer_validation_table
    path=Path(manifest).resolve();out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous linear power exports')
    if path.stat().st_size*12>memory_gib*1024**3:raise MemoryError('Linear manifest exceeds allocation')
    validation=validate_linear_bundle(path,memory_gib=memory_gib)
    info=json.loads(path.read_text());h=float(info['h']);epoch=validation['epoch']
    relative=info['source_files']['start']['pk'];pk=(path.parent/relative).resolve()
    if not pk.is_relative_to(path.parent) or file_sha256(pk)!=info['checksums_sha256'][relative]:
        raise ValueError('Linear power source changed or escaped bundle')
    if pk.stat().st_size*12>memory_gib*1024**3:raise MemoryError('Linear power table exceeds allocation')
    header=pk.read_text()
    if 'k (h/Mpc)' not in header or 'P (Mpc/h)^3' not in header:raise ValueError('Explicit CLASS power units required')
    match=re.search(r'redshift\s+z\s*=\s*([-+0-9.eE]+)',header)
    if match is None or not class_redshift_matches(float(match[1]), info['parameters']['z_start']):
        raise ValueError('Linear power payload has a different solver epoch')
    table=np.loadtxt(pk,comments='#',ndmin=2)
    if table.shape[1]!=2 or not np.isfinite(table).all() or np.any(table[:,0]<=0) or np.any(np.diff(table[:,0])<=0):
        raise ValueError('Ordered finite CLASS k/P columns required')
    out.mkdir(parents=True);sources={str(path):file_sha256(path),str(pk):file_sha256(pk)}
    sources.update({row['path']:row['sha256'] for row in validation['sources'].values()})
    product=out/'input-power.json';arrays={'k':table[:,0]*h,'power':table[:,1]/h**3,'supported':np.ones(len(table),bool)}
    _publish(product,{'schema':'flagship-input-power-v1','epoch':epoch,'solver_bundle':validation,
        'normalization':info['parameters'],'matter_definition':info['matter_definition'],
        'conversion':'k_physical=h*k_native; P_physical=P_native/h^3',
        'uncertainty':'deterministic linear solver output; solver convergence not established','science_acceptance':'NOT_ACCEPTED'},
        arrays,{'k':'1 / Mpc','power':'Mpc^3','supported':'dimensionless'},sources)
    transfer=out/'species-reference.txt';solver=info['class_parameters']['start']
    config={'cosmology':{'H0':solver['H0'],'omega_b':solver['omega_b'],'omega_cdm':solver['omega_cdm'],
        'model':info['model']},'wdm':info.get('wdm',{})}
    write_species_transfer_validation_table(validation['sources']['transfer']['path'],transfer,config,info['model'],info['parameters']['z_start'])
    values=np.loadtxt(transfer,comments='#',ndmin=2);dark=4 if info['model']=='wdm' else 1
    arrays={'k':values[:,0]*h,'matter':values[:,3],'baryon':values[:,2],'dark':values[:,dark],
            'theta_baryon':values[:,6],'theta_dark':values[:,7 if info['model']=='wdm' else 5],
            'supported':np.isfinite(values[:,[0,2,3,dark]]).all(axis=1),
            'velocity_supported':np.isfinite(values[:,[6,7 if info['model']=='wdm' else 5]]).all(axis=1)}
    transfer_product=out/'signed-transfers.json'
    _publish(transfer_product,{'schema':'flagship-signed-transfers-v1','epoch':epoch,'gauge':'N-body',
        'normalization':'original signed CLASS transfer amplitudes; no square or absolute value',
        'velocity_reference':str(transfer),'solver_bundle':validation,'science_acceptance':'NOT_ACCEPTED'},arrays,
        {'k':'1 / Mpc','matter':'dimensionless','baryon':'dimensionless','dark':'dimensionless','supported':'dimensionless',
         'theta_baryon':'Mpc^-1','theta_dark':'Mpc^-1','velocity_supported':'dimensionless'},sources)
    figures=[_figure('linear-input-power','input_power','input',[{'y':'power','label':'input total matter'}],'1 / Mpc','Mpc^3',epoch),
        _figure('signed-species-transfers','transfer_comparison','transfers',
            [{'y':key,'label':key} for key in ('matter','baryon','dark')],'1 / Mpc','dimensionless',epoch)]
    velocity=_figure('signed-species-velocities','transfer_comparison','transfers',
        [{'y':key,'label':key,'support':'velocity_supported'} for key in ('theta_baryon','theta_dark')],
        '1 / Mpc','Mpc^-1',epoch)['panels'][0]
    velocity['panel_id']='signed_velocity_transfer';figures[1]['panels'].append(velocity)
    for source,checksum in sources.items():
        if file_sha256(source)!=checksum:raise ValueError('Linear input changed during export')
    return export_scientific_suite({'sources':{'input':str(product),'transfers':str(transfer_product)},
        'figures':figures,'dpi':160,'maximum_panels_per_figure':2,'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)


def paired_power_statistics(left, right, joint_covariance, *, denominator_floor):
    """Propagate the measured covariance of two spectra in matching bins.

    A and B are powers in Mpc^3; covariance rows/columns are ordered [A, B]
    and have units Mpc^6. The residual is B-A (Mpc^3), the ratio B/A is
    dimensionless. Errors use C_out = J C_in J.T, retaining correlations
    between spectra and between bins. Ratio errors use a first-order expansion;
    the caller supplies a positive denominator floor in Mpc^3 to exclude bins
    near zero. Negative shot-noise-subtracted powers remain valid measurements.
    """
    a = np.asarray(left, float)
    b = np.asarray(right, float)
    cov = np.asarray(joint_covariance, float)
    if a.ndim != 1 or not len(a) or b.shape != a.shape or cov.shape != (2*len(a), 2*len(a)):
        raise ValueError('Aligned power bins and full [A,B] joint covariance required')
    if (isinstance(denominator_floor, (bool, np.bool_))
            or not np.isfinite(denominator_floor) or denominator_floor <= 0):
        raise ValueError('Explicit positive physical denominator support floor required')
    if not np.isfinite(cov).all() or not np.allclose(cov, cov.T, rtol=1e-12, atol=0):
        raise ValueError('Finite symmetric measured joint covariance required')
    scale = max(float(np.max(np.abs(cov), initial=0)), np.finfo(float).tiny)
    if np.linalg.eigvalsh(cov).min(initial=0) < -128*np.finfo(float).eps*scale*max(1, len(cov)):
        raise ValueError('Measured power covariance is not positive semidefinite')

    # d(B-A)/d[A,B] = [-I,I]. Shared noise can therefore cancel; treating
    # the spectra as independent would lose the supplied cross-covariance.
    n = len(a)
    residual = b-a
    difference = np.c_[-np.eye(n), np.eye(n)]
    residual_cov = difference @ cov @ difference.T

    support = np.isfinite(a) & np.isfinite(b) & (np.abs(a) >= denominator_floor)
    ratio = np.divide(b, a, out=np.full_like(a, np.nan), where=support)
    jac = np.zeros((n, 2*n))
    at = np.flatnonzero(support)
    # d(B/A)/dA = -(B/A)/A and d(B/A)/dB = 1/A. Dividing in this order
    # avoids overflowing A**2 when the ratio and derivative are representable.
    jac[at, at] = -ratio[at]/a[at]
    jac[at, n+at] = 1/a[at]
    ratio_cov = jac @ cov @ jac.T
    ratio_cov[~support, :] = np.nan
    ratio_cov[:, ~support] = np.nan
    return {
        'residual': residual,
        'residual_covariance': residual_cov,
        'residual_error': np.sqrt(np.maximum(np.diag(residual_cov), 0)),
        'ratio': ratio,
        'ratio_covariance': ratio_cov,
        'ratio_error': np.sqrt(np.maximum(np.diag(ratio_cov), 0)),
        'supported': support,
        'residual_supported': np.isfinite(a) & np.isfinite(b),
    }


def export_evolved_power(bundle,destination,*,memory_gib=1.):
    memory_gib=_allocation(memory_gib)
    import h5py
    from .analysis import verify_analysis_bundle,load_cfg
    from .data import verify_dataset_semantics
    bundle=Path(bundle).resolve();out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous evolved power exports')
    provenance=verify_analysis_bundle(bundle,verify_sources=True)
    run=Path(provenance['source_run']).resolve();cfg=load_cfg(run);h=float(cfg['cosmology']['H0'])/100.
    if not np.isfinite(h) or h<=0:raise ValueError('Native run H0 required for physical power units')
    source=bundle/'analysis_products.h5';sources={str(source):file_sha256(source),
        str(bundle/'analysis_provenance.json'):file_sha256(bundle/'analysis_provenance.json'),
        str(run/'config.yaml'):file_sha256(run/'config.yaml')}
    products={};figures=[];out.mkdir(parents=True)
    with h5py.File(source) as native:
        for label in sorted(native):
            path=label+'/pk/data'
            if not label.startswith('snap_') or path not in native:continue
            group=native[path];names={'k':'k_h_Mpc','power':'P_Mpc_h3','error':'P_gaussian_1sigma_Mpc_h3',
                'mode_count':'mode_count','shot_noise':'P_shot_noise_Mpc_h3','raw_power':'P_raw_shot_included_Mpc_h3',
                'fold_segment':'fold_segment'}
            if any(name not in group for name in names.values()):raise ValueError('Incomplete native matter-power estimator record')
            arrays={};epoch=None
            for key,name in names.items():
                data=group[name];verify_dataset_semantics(data)
                if data.ndim!=1 or data.size*512>memory_gib*1024**3:raise MemoryError('Native power arrays exceed workspace')
                current=json.loads(data.attrs['epoch_json'])
                if epoch is not None and current!=epoch:raise ValueError('Native spectrum datasets have different saved epochs')
                epoch=current;arrays[key]=data[:]
                expected='h/Mpc' if key=='k' else '(Mpc/h)^3' if key in ('power','error','shot_noise','raw_power') else None
                if expected and data.attrs['units']!=expected:raise ValueError('Native power units mismatch: '+name)
            if any(value.shape!=arrays['k'].shape for value in arrays.values()):raise ValueError('Native estimator arrays have different bins')
            if not np.isfinite(arrays['k']).all() or np.any(arrays['k']<=0) or np.any(np.diff(arrays['k'])<=0):
                raise ValueError('Native power requires ordered positive finite k bins')
            if arrays['mode_count'].dtype.kind not in 'iu' or np.any(arrays['mode_count']<0):
                raise ValueError('Native Fourier mode counts require exact nonnegative integers')
            arrays['k']*=h
            for key in ('power','error','shot_noise','raw_power'):arrays[key]/=h**3
            arrays['supported']=np.isfinite(arrays['power'])&np.isfinite(arrays['error'])&(arrays['error']>=0)&(arrays['mode_count']>0)
            def text(name):
                value=group[name][()]
                return value.decode() if isinstance(value,bytes) else str(value)
            policy={'assignment':'GADGET folded CIC-window-deconvolved native spectrum',
                'shot_noise':'native measured variable-mass shot noise subtracted; retained in variance',
                'particle_selection':text('particle_selection'),'uncertainty_method':text('uncertainty_method'),
                'native_column_normalization':text('native_column_normalization'),
                'mode_count_precision':text('mode_count_precision'),
                'fold_factor':int(group['fold_factor'][()]),'pm_grid':int(group['pm_grid'][()])}
            product=out/(label+'-power.json');products[label]=str(product)
            _publish(product,{'schema':'flagship-evolved-power-v1','epoch':epoch,'native_run':str(run),
                'native_analysis':str(bundle),'source_dataset':path,'estimator_policy':policy,
                'science_acceptance':'NOT_ACCEPTED','support_scope':'occupied native bins with finite errors; numerical convergence not established'},
                arrays,{key:'1 / Mpc' if key=='k' else 'Mpc^3' if key in ('power','error','shot_noise','raw_power') else 'dimensionless' for key in arrays},sources)
            figures.append(_figure(label+'-evolved-power','evolved_power',label,[{'y':'power','error':'error','label':'native signed total matter'}],'1 / Mpc','Mpc^3',epoch))
    if file_sha256(source)!=sources[str(source)]:raise ValueError('Native analysis changed during export')
    atomic_json(out/'products.json',{'schema':'flagship-evolved-power-index-v1','products':products,'science_acceptance':'NOT_ACCEPTED'})
    return export_scientific_suite({'sources':products,'figures':figures,'dpi':160,'maximum_panels_per_figure':1,
        'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)


def export_matched_power(request,destination,*,memory_gib=1.):
    memory_gib=_allocation(memory_gib)
    from .analysis import compare_initial_ic_phases,load_cfg,_initial_condition_paths
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous matched power evidence')
    if request.get('schema')!='flagship-matched-power-v1':raise ValueError('Explicit matched power request required')
    records=[];sources={}
    for key in ('left','right','joint_covariance'):
        binding=request[key];path=Path(binding['path']).resolve()
        if file_sha256(path)!=binding['sha256']:raise ValueError('Matched power input changed: '+key)
        report,arrays,pins=bound_power_product(binding,memory_gib=memory_gib/4);sources.update(pins);records.append((report,arrays))
    (ra,a),(rb,b),(rc,c)=records
    for report in (ra,rb):
        if report.get('schema')!='flagship-evolved-power-v1':raise ValueError('Source-bound native evolved power product required')
        specs=report['numeric_data']['array_contract']['datasets']
        if specs['k']['units']!='1 / Mpc' or specs['power']['units']!='Mpc^3':raise ValueError('Matched power physical units differ')
    if ra['epoch']['actual_scale_factor']!=rb['epoch']['actual_scale_factor'] or not np.array_equal(a['k'],b['k']):
        raise ValueError('Matched power requires identical actual epochs and physical k bins; no extrapolation/interpolation')
    if ra['estimator_policy']!=rb['estimator_policy']:raise ValueError('Assignment/shot-noise/species estimator policies differ')
    if rc.get('left_sha256')!=request['left']['sha256'] or rc.get('right_sha256')!=request['right']['sha256']:
        raise ValueError('Joint covariance does not bind the exact ordered native products')
    if rc['numeric_data']['array_contract']['datasets']['joint_covariance']['units']!='Mpc^6':
        raise ValueError('Joint covariance must use physical power-squared units')
    configs=[load_cfg(report['native_run']) for report in (ra,rb)]
    mesh=min(*(int(cfg['simulation']['particles_per_dimension']) for cfg in configs),
             *(int(cfg.get('validation',{}).get('ic_phase',{}).get('max_mesh_size',128)) for cfg in configs))
    if mesh**3*512>memory_gib*1024**3/2:raise MemoryError('Existing native phase estimator exceeds paired allocation')
    for report in (ra,rb):
        run=Path(report['native_run'])
        for path in [run/'config.yaml',run/'manifest.json',*_initial_condition_paths(run)]:sources[str(path.resolve())]=file_sha256(path)
    phase=compare_initial_ic_phases(ra['native_run'],rb['native_run'],*configs)
    if phase['status']!='PASS':raise ValueError('Actual native IC phase comparison did not pass: '+json.dumps(phase))
    n=len(a['k'])
    if n*n*256>memory_gib*1024**3/2:raise MemoryError('Paired covariance propagation exceeds allocation')
    values=paired_power_statistics(a['power'],b['power'],c['joint_covariance'],denominator_floor=request['denominator_floor_mpc3'])
    physical_phase_limit=phase['k_max_h_Mpc']*min(float(cfg['cosmology']['H0'])/100. for cfg in configs)
    phase_supported=a['k']<=physical_phase_limit
    values['supported']&=a['supported']&b['supported']&phase_supported
    values['residual_supported']&=a['supported']&b['supported']&phase_supported
    values['phase_supported']=phase_supported
    values['k']=a['k'];units={key:'Mpc^6' if key=='residual_covariance' else 'Mpc^3' if key in ('residual','residual_error') else '1 / Mpc' if key=='k' else 'dimensionless' for key in values}
    out.mkdir(parents=True);product=out/'matched-power.json'
    _publish(product,{'schema':'flagship-matched-power-product-v1','epoch':ra['epoch'],'phase_measurement':phase,
        'request':request,'covariance_policy':'supplied measured full [A,B] covariance; delta-method ratio errors',
        'science_acceptance':'NOT_ACCEPTED'},values,units,sources)
    figures=[_figure('matched-power-ratio','power_ratios','paired',[{'y':'ratio','error':'ratio_error','label':'B/A'}],'1 / Mpc','dimensionless',ra['epoch']),
        _figure('matched-power-residual','matched_residuals','paired',[{'y':'residual','error':'residual_error','label':'B-A','support':'residual_supported'}],'1 / Mpc','Mpc^3',ra['epoch'])]
    for path,checksum in sources.items():
        if file_sha256(path)!=checksum:raise ValueError('Matched source changed during measurement')
    return export_scientific_suite({'sources':{'paired':str(product)},'figures':figures,'dpi':160,
        'maximum_panels_per_figure':1,'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)

# Paired realizations and their measured ensemble covariance.

def bound_power_product(binding, *, memory_gib):
    """Verify numeric payload and original producer sources before using a cache."""
    path = Path(binding['path']).resolve()
    if file_sha256(path) != binding['sha256']:
        raise ValueError('Power product binding changed')
    report, arrays, pins = read_product(path, memory_gib=memory_gib)
    # Older externally measured covariance records can be source-free; native
    # spectra and this producer always carry complete original source bindings.
    if report.get('schema') == 'flagship-evolved-power-v1':
        pins.update(numeric_audit_source_bindings(report))
        from .analysis import verify_analysis_bundle
        original = verify_analysis_bundle(Path(report['native_analysis']), verify_sources=True)
        if Path(original['source_run']).resolve() != Path(report['native_run']).resolve():
            raise ValueError('Power product and original native analysis belong to different runs')
    else:
        for specification in report['numeric_data']['array_contract']['datasets'].values():
            for source, checksum in json.loads(specification['provenance_json']).get('sources', {}).items():
                if file_sha256(source) != checksum:
                    raise ValueError('Original power producer source changed: ' + source)
                pins[str(Path(source).resolve())] = checksum
    return report, arrays, pins



def ensemble_power_statistics(left, right, supported, *, minimum_realizations,
                              denominator_floor):
    """Full [mean(A),mean(B)] covariance; retain unsupported rows as NaN."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    mask = np.asarray(supported)
    if (a.ndim != 2 or b.shape != a.shape or not a.shape[1]
            or mask.dtype.kind != 'b' or mask.shape != a.shape):
        raise ValueError('Paired realization-by-bin powers and boolean support required')
    if type(minimum_realizations) is not int or minimum_realizations < 2 or len(a) < minimum_realizations:
        raise ValueError('Declared ensemble size must be at least two realizations')
    # Complete-case common support gives one PSD covariance, unlike pairwise
    # deletion, which can produce a non-PSD covariance with different samples.
    common = np.all(mask & np.isfinite(a) & np.isfinite(b), axis=0)
    n = a.shape[1]
    values = {'left_power': np.full(n, np.nan), 'right_power': np.full(n, np.nan),
              'joint_covariance': np.full((2*n, 2*n), np.nan),
              'realization_covariance': np.full((2*n, 2*n), np.nan),
              'covariance_supported': common}
    if not common.any():
        raise ValueError('No bins supported in every declared realization')
    from .matched import ensemble_matched_covariance
    vectors = np.c_[a[:, common], b[:, common]]
    reduced = ensemble_matched_covariance([
        {'realization_id': str(i), 'mean_difference': row} for i, row in enumerate(vectors)
    ], minimum_realizations=minimum_realizations)
    # The existing reducer field is named mean_difference. Here its actual
    # input is explicitly the concatenated [P_A,P_B] vector, not a difference.
    indices = np.r_[np.flatnonzero(common), n + np.flatnonzero(common)]
    values['joint_covariance'][np.ix_(indices, indices)] = reduced['ensemble_mean_covariance']
    values['realization_covariance'][np.ix_(indices, indices)] = reduced['realization_covariance']
    m = int(common.sum())
    values['left_power'][common] = reduced['ensemble_mean'][:m]
    values['right_power'][common] = reduced['ensemble_mean'][m:]
    paired = paired_power_statistics(values['left_power'][common], values['right_power'][common],
        reduced['ensemble_mean_covariance'], denominator_floor=denominator_floor)
    for key, value in paired.items():
        if value.ndim == 2:
            target = np.full((n, n), np.nan)
            target[np.ix_(common, common)] = value
        else:
            target = np.zeros(n, bool) if value.dtype.kind == 'b' else np.full(n, np.nan)
            target[common] = value
        values[key] = target
    return values



def _read_ensemble(request, *, memory_gib):
    from .analysis import load_cfg, _initial_condition_paths, compare_initial_ic_phases
    if request.get('schema') != 'native-paired-power-ensemble-v1':
        raise ValueError('Explicit native paired-power ensemble request required')
    design = request['independence_design']
    if (design.get('assertion') != 'independent_parent_random_streams'
            or not isinstance(design.get('description'), str) or not design['description'].strip()):
        raise ValueError('Explicit parent-generation independence design required')
    evidence = Path(design['evidence']['path']).resolve()
    if file_sha256(evidence) != design['evidence']['sha256']:
        raise ValueError('Independence design evidence changed')
    rows = request['realizations']
    if len(json.dumps(request))*32 > memory_gib*1024**3/8:
        raise MemoryError('Ensemble recipe metadata exceeds allocation')
    minimum = request['minimum_realizations']
    if type(minimum) is not int or minimum < 2 or len(rows) < minimum:
        raise ValueError('Declared independent ensemble size is insufficient')
    if len({row['realization_id'] for row in rows}) != len(rows):
        raise ValueError('Repeated realization identity')
    if len({row['parent_phase_namespace'] for row in rows}) != len(rows):
        raise ValueError('Targets/refinements sharing a parent cannot be independent realizations')
    sources = {str(evidence): file_sha256(evidence)}
    reference = None; left = []; right = []; support = []; phases = []
    seeds = [set(), set()]; ic_digests = [set(), set()]
    for row in rows:
        pair = []; runs = []; configs = []
        for side, name in enumerate(('left', 'right')):
            report, arrays, pins = bound_power_product(row[name], memory_gib=memory_gib/8)
            sources.update(pins)
            if report.get('schema') != 'flagship-evolved-power-v1':
                raise ValueError('Original native evolved-power product required')
            specs = report['numeric_data']['array_contract']['datasets']
            if specs['k']['units'] != '1 / Mpc' or specs['power']['units'] != 'Mpc^3':
                raise ValueError('Physical power units required')
            signature = (report['epoch']['actual_scale_factor'], report['estimator_policy'])
            if reference is None:
                reference = (signature, arrays['k'].copy(), report['epoch'])
            if signature != reference[0] or not np.array_equal(arrays['k'], reference[1]):
                raise ValueError('Exact common actual epoch, physical k grid and estimator policy required')
            run = Path(report['native_run']).resolve(); cfg = load_cfg(run)
            manifest = json.loads((run/'manifest.json').read_text())
            seed = cfg['simulation']['random_seed']
            if type(seed) is not int or manifest.get('random_seed', manifest.get('simulation_parameters', {}).get('random_seed')) != seed:
                raise ValueError('Native manifest/configuration generation seed mismatch')
            paths = [run/'config.yaml', run/'manifest.json', *_initial_condition_paths(run)]
            ic_keys = tuple(sorted(file_sha256(p) for p in paths[2:]))
            if not ic_keys or seed in seeds[side] or ic_keys in ic_digests[side]:
                raise ValueError('Repeated native parent seed or IC payload in independent ensemble')
            seeds[side].add(seed); ic_digests[side].add(ic_keys)
            sources.update({str(p.resolve()): file_sha256(p) for p in paths})
            pair.append(arrays); runs.append(run); configs.append(cfg)
            n = len(reference[1])
            if n*n*1024 + len(rows)*n*128 > memory_gib*1024**3/2:
                raise MemoryError('Measured ensemble covariance exceeds allocation')
        if configs[0]['simulation']['random_seed'] != configs[1]['simulation']['random_seed']:
            raise ValueError('Paired generation seeds differ from declared parent-stream design')
        mesh = min(*(int(c['simulation']['particles_per_dimension']) for c in configs),
                   *(int(c.get('validation', {}).get('ic_phase', {}).get('max_mesh_size', 128)) for c in configs))
        if mesh**3*512 > memory_gib*1024**3/2:
            raise MemoryError('Native paired-phase measurement exceeds allocation')
        phase = compare_initial_ic_phases(*runs, *configs)
        if phase['status'] != 'PASS':
            raise ValueError('Actual within-pair native IC phase comparison did not pass')
        phases.append({'realization_id': row['realization_id'], 'measurement': phase})
        physical_limit = phase['k_max_h_Mpc'] * min(c['cosmology']['H0']/100. for c in configs)
        left.append(pair[0]['power']); right.append(pair[1]['power'])
        support.append(pair[0]['supported'] & pair[1]['supported'] & (reference[1] <= physical_limit))
    return reference, left, right, support, phases, sources



def export_power_ensemble(request, destination, *, memory_gib):
    memory_gib = _allocation(memory_gib)
    out = Path(destination).resolve()
    if out.exists():
        raise FileExistsError('Keep previous measured power ensemble')
    reference, a, b, support, phases, sources = _read_ensemble(request, memory_gib=memory_gib)
    values = ensemble_power_statistics(a, b, support,
        minimum_realizations=request['minimum_realizations'],
        denominator_floor=request['denominator_floor_mpc3'])
    values['k'] = reference[1]
    units = {key: 'Mpc^6' if key in ('joint_covariance', 'realization_covariance', 'residual_covariance')
             else 'Mpc^3' if key in ('left_power', 'right_power', 'residual', 'residual_error')
             else '1 / Mpc' if key == 'k' else 'dimensionless' for key in values}
    sources[str(Path(__file__).resolve())] = file_sha256(__file__)
    product = out/'ensemble-power.json'
    report = {'schema': 'native-paired-power-ensemble-product-v1', 'epoch': reference[2],
        'estimator_policy': reference[0][1], 'request': request, 'phase_measurements': phases,
        'independent_realization_count': len(a),
        'covariance_definition': 'Unbiased sample covariance of ordered [P_A,P_B] divided by N for the ensemble mean; full cross-bin and A/B blocks retained',
        'comparison_definition': 'Ratio of ensemble means and difference of ensemble means; delta-method ratio uncertainty',
        'support_definition': 'Intersection across every realization and measured within-pair phase range; unsupported covariance rows remain NaN',
        'independence_status': 'CONDITIONAL_ON_SOURCE_GENERATION_DESIGN; distinct seeds and IC hashes checked, statistical independence not established by phase agreement',
        'science_acceptance': 'NOT_ACCEPTED'}
    _publish(product, report, values, units, sources)
    figures = [
        _figure('ensemble-power-ratio', 'power_ratios', 'ensemble',
                [{'y': 'ratio', 'error': 'ratio_error', 'label': 'mean(B)/mean(A)'}], '1 / Mpc', 'dimensionless', reference[2]),
        _figure('ensemble-power-residual', 'matched_residuals', 'ensemble',
                [{'y': 'residual', 'error': 'residual_error', 'support': 'residual_supported', 'label': 'mean(B)-mean(A)'}], '1 / Mpc', 'Mpc^3', reference[2])]
    for path, checksum in sources.items():
        if file_sha256(path) != checksum:
            raise ValueError('Ensemble power source changed during measurement')
    result = export_scientific_suite({'sources': {'ensemble': str(product)}, 'figures': figures,
        'dpi': 160, 'maximum_panels_per_figure': 1, 'organized_outputs': True}, out/'gallery', base=out, memory_gib=memory_gib)
    atomic_json(out/'products.json', {'schema': 'native-paired-power-ensemble-index-v1',
        'product': str(product), 'science_acceptance': 'NOT_ACCEPTED'})
    return result



def prepare_power_campaign(request, destination):
    """Freeze all five graph routes from existing native and linear products.

    This prepares postprocessing only. It never launches native evolution.
    The ensemble request names the exact retained paired products and original
    independent generation design; it cannot reinterpret four sibling zooms
    as four independent realizations.
    """
    from .workflow import compile_requested_product_dag, default_registry
    if request.get('schema') != 'native-power-campaign-v1':
        raise ValueError('Explicit five-variant power campaign request required')
    memory = _allocation(request['memory_gib'])
    disk = request['disk_gib']
    if type(disk) not in (int, float) or not np.isfinite(disk) or disk <= 0:
        raise ValueError('Positive finite power campaign disk allocation required')
    out = Path(destination).resolve()
    if out.exists():
        raise FileExistsError('Keep previous power campaign')
    # Ensemble metadata is frozen as a new original input, with all nested
    # scientific inputs verified by the registered workflow intake.
    ensemble = request['ensemble_request']
    *_, sources = _read_ensemble(ensemble, memory_gib=memory)
    stages = []
    def stage(name, operation, product):
        return {'stage_id': name, 'kind': 'flagship_product', 'parameters': {
            'operation': operation, 'product': str(product),
            'destination': '{generation}/products', 'memory_gib': memory},
            'outputs': {name: 'products'}}
    linear = request['linear_spectrum']
    if file_sha256(linear['path']) != linear['sha256']:
        raise ValueError('Linear spectrum binding changed')
    stages.append(stage('linear-power', 'power-linear', Path(linear['path']).resolve()))
    bundles = request['native_analysis_bundles']
    if not bundles:
        raise ValueError('At least one sealed native analysis bundle required')
    for i, binding in enumerate(bundles):
        path = Path(binding['path']).resolve()
        if file_sha256(path/'analysis_provenance.json') != binding['sha256']:
            raise ValueError('Native analysis provenance changed')
        stages.append(stage('evolved-power-'+str(i), 'power-evolved', path))
    for row in stages:
        sources.update(power_input_bindings(row['parameters']))
    ensemble_path = out/'ensemble-request.json'
    stages.append(stage('matched-power', 'power-ensemble', ensemble_path))
    # No unresolved tokens: the compiler validates this actual future request
    # path; preparation writes it atomically before publishing the manifest.
    frozen_sources = [{'path': p, 'sha256': digest} for p, digest in sorted(sources.items())]
    for row in stages:
        row['input_files'] = frozen_sources
    spec = {'schema': 'research-workflow-v1', 'resources': {
        'mpi_ranks': 1, 'omp_threads': 1, 'memory_gib': memory, 'disk_gib': disk,
        'maximum_stage_count': len(stages)}, 'physics_spec': {}, 'capabilities': [],
        'followup_limits': {'maximum_children': 0, 'maximum_depth': 0},
        'requested_products': [row['stage_id'] for row in stages], 'stages': stages}
    registry = default_registry()
    plan = compile_requested_product_dag(spec, registry, request_base=out)
    code = {}
    for row in plan['stages']:
        code.update(registry.source_hashes(row['kind']))
    for path, checksum in {**sources, **code}.items():
        if file_sha256(path) != checksum:
            raise ValueError('Power campaign input changed during preparation')
    out.mkdir(parents=True)
    atomic_json(ensemble_path, ensemble)
    atomic_json(out/'request.json', request)
    atomic_json(out/'workflow.json', spec)
    manifest = {'schema': 'native-power-campaign-prepared-v1', 'status': 'RECIPES_PREPARED',
        'workflow': {'path': str(out/'workflow.json'), 'sha256': file_sha256(out/'workflow.json')},
        'source_hashes': {**sources, **code, str(ensemble_path): file_sha256(ensemble_path)},
        'variants': ['input_power', 'transfer_comparison', 'evolved_power', 'power_ratios', 'matched_residuals'],
        'stage_count': len(stages), 'science_acceptance': 'NOT_ACCEPTED',
        'scope': 'Postprocess retained source products; ensemble independence is conditional on declared generation design; native convergence pending'}
    atomic_json(out/'manifest.json', manifest)
    return manifest

