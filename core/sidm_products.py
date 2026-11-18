"""SIDM history, matched contrasts, derived figures and campaign export routes.

Use measured native histories, their physical units and original support masks.
A branch contrast is delta Q = Q_branch - Q_baseline. The dimensionless plotted
contrast is delta Q / max(abs(Q_baseline), the declared dimensional floor),
not a branch/baseline ratio. Paired differences alone do not give independent
ensemble covariance, and a density rise alone does not prove core collapse.
Native evolution lives outside this module; these reducers use saved native
measurements and export their scientific figures.
"""
from pathlib import Path
import copy
import tempfile
import numpy as np
from .foundation import atomic_json, file_sha256, merge_source_bindings as _merge
from .data import exact_uint64_ids, export_numeric_audit, numeric_audit_source_bindings
from .halo_campaign import _key


# History and shell measurements

def sidm_history_inputs(product, *, run_directory, memory_gib):
    """Load a completed native history and its original evolution inputs.

    The plotted model and matching rules must be those used by the producer.
    This reads saved results; it never launches another native simulation.
    """
    from .sidm_analysis import _load_product, _stage_native_bindings

    path = Path(product).resolve()
    report, arrays, sources = _load_product(
        {'path': str(path), 'sha256': file_sha256(path)}, path.parent, memory_gib * 1024**3 / 2)
    if report.get('schema') != 'sidm-halo-history-v1' or report.get('execution_status') != 'COMPLETE':
        raise ValueError('Completed original native SIDM halo history required')
    request = report['request']
    if report['matching'] != request['matching'] or report['dark_matter_model'] != request['dark_matter_model']:
        raise ValueError('SIDM report relabels original matching or evolved model')
    if request['memory_gib'] > memory_gib:
        raise MemoryError('SIDM native receipt intake exceeds allocation')
    native = _stage_native_bindings({
        'request': request, 'native_run_directory': str(Path(run_directory).resolve()),
        'native_stage_receipt': report['native_stage_receipt']})
    if any(sources.get(path) != checksum for path, checksum in native.items()):
        raise ValueError('SIDM history omits original native execution bindings')
    return report, arrays, sources


def sidm_history_arrays(report, arrays):
    """Expose saved central densities, support and categorical phases.

    Density is M(<r)/(4*pi*r^3/3) in Msun per proper kpc cubed, measured in
    the producer's fixed aperture. Time is the saved elapsed time in Gyr.
    Phase numbers only index the returned vocabulary; they are not a
    continuous collapse clock or a separate physical measurement.
    """
    from .sidm_measurements import classify_sidm_history

    contracts = report['numeric_data']['array_contract']['datasets']
    rows = report['epochs']
    count = len(rows)
    if not count:
        raise ValueError('Original SIDM epoch census is empty')
    units = {'elapsed_time_gyr': 'Gyr', 'central_density_msun_kpc3': 'Msun / proper kpc^3',
             'central_supported': 'dimensionless', 'halo_ids': 'identity'}
    for key, unit in units.items():
        if contracts[key]['units'] != unit:
            raise ValueError('Original SIDM history unit differs: ' + key)
    time = np.asarray(arrays['elapsed_time_gyr'])
    density = np.asarray(arrays['central_density_msun_kpc3'])
    support = np.asarray(arrays['central_supported'])
    ids = exact_uint64_ids(arrays['halo_ids'], unique=False)
    if any(value.shape != (count,) for value in (time, density, support, ids)):
        raise ValueError('Original SIDM saved epoch dimensions differ')
    if support.dtype != np.dtype(bool) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError('Original SIDM saved time/support census differs')
    matching = report['matching']
    if (contracts['halo_ids']['identity_namespace'] != matching['halo_match_namespace']
            or np.any(ids != matching['halo_match_id'])):
        raise ValueError('Original SIDM matched halo identity differs')
    expected_density = [row['core']['central_aperture']['density_msun_kpc3'] for row in rows]
    expected_support = [row['core']['central_aperture']['supported'] for row in rows]
    if (not np.array_equal(density, expected_density)
            or not np.array_equal(time, [row['elapsed_time_gyr'] for row in rows])
            or not np.array_equal(support, expected_support)):
        raise ValueError('SIDM saved central arrays differ from original records')
    if classify_sidm_history(rows, report['request']['collapse']) != report['phase_classification']:
        raise ValueError('Original SIDM collapse classification does not replay')
    vocabulary = ['UNRESOLVED', 'CENTRAL_DENSITY_MEASURED', 'DENSITY_DECLINING',
                  'DENSITY_RISING', 'DENSITY_STATIONARY', 'PERSISTENT_DENSITY_THRESHOLD',
                  'GRAVOTHERMAL_COLLAPSE_INDICATORS']
    phase = np.asarray([vocabulary.index(row['phase'])
                        for row in report['phase_classification']['epochs']], np.int64)
    return {'elapsed_time_gyr': time, 'central_density': density, 'central_supported': support,
            'halo_ids': ids, 'phase_category': phase}, vocabulary


def sidm_shell_figure_arrays(report,arrays,ordinal):
    """Select one saved shell row, retaining separate measurement masks.

    Radii are proper kpc; density is Msun/kpc^3; one-dimensional velocity
    variance is (km/s)^2. Radial kinetic heat flux is erg/(cm^2 s), outward
    positive. Finite failed kernel diagnostics remain visible as diagnostics.
    """
    from .sidm_measurements import HISTORY_FIELDS
    declarations=report['numeric_data']['array_contract']['datasets']
    fields={'radius_proper_kpc':'proper kpc',**{k:HISTORY_FIELDS[k] for k in [
        'density_msun_kpc3','density_shot_noise_msun_kpc3','velocity_dispersion_squared_km2_s2','anisotropy_beta',
        'heat_flux_radial_erg_cm2_s','heat_flux_sampling_error_erg_cm2_s','maximum_coarse_support_fraction','maximum_omitted_scattering_upper_bound']},
        'profile_supported':'dimensionless','transport_supported':'dimensionless','scattering_supported':'dimensionless'}
    values={}
    for key,unit in fields.items():
        if declarations[key]['units']!=unit:raise ValueError('Original SIDM shell unit differs: '+key)
        value=np.asarray(arrays[key])
        if value.ndim!=2 or value.shape[0]!=len(report['epochs']):raise ValueError('Original SIDM shell epoch dimensions differ')
        values[key]=value[ordinal]
    shape=values['radius_proper_kpc'].shape
    if any(v.shape!=shape for v in values.values()) or any(values[k].dtype!=np.dtype(bool) for k in ['profile_supported','transport_supported','scattering_supported']):
        raise ValueError('Aligned original SIDM shell support required')
    if not np.isfinite(values['radius_proper_kpc']).all() or np.any(values['radius_proper_kpc']<=0):raise ValueError('Positive measured SIDM shell radii required')
    # Diagnostic values that FAIL the science support ceiling must remain visible.
    values['kernel_diagnostic_available']=np.isfinite(values['maximum_coarse_support_fraction'])&np.isfinite(values['maximum_omitted_scattering_upper_bound'])
    fields['kernel_diagnostic_available']='dimensionless'
    return values,fields


def export_native_sidm_history(product,destination,*,run_directory,memory_gib=2.):
    from core.research.display import export_scientific_suite
    report,arrays,sources=sidm_history_inputs(product,run_directory=run_directory,memory_gib=memory_gib)
    history,vocabulary=sidm_history_arrays(report,arrays);out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous SIDM figures')
    out.mkdir(parents=True);products={};figures=[];sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    epoch={'canonical_time':'actual original elapsed time','saved_native_epochs':[row['epoch'] for row in report['epochs']]}
    def publish(name,values,units,metadata,selected_epoch):
        target=out/(name+'.json');products[name]=str(target)
        specs={k:{'units':u,'quantity':k,'missing_policy':'original support masks and nonfinite values retained'} for k,u in units.items()}
        if 'halo_ids' in specs:specs['halo_ids']['identity_namespace']=report['matching']['halo_match_namespace']
        export_numeric_audit(target,{'schema':'native-sidm-figure-v1','epoch':selected_epoch,'science_acceptance':'NOT_ACCEPTED',
            'original_history':str(Path(product).resolve()),'dark_matter_model':report['dark_matter_model'],**metadata},values,specs,
            algorithm=__file__,context={'epoch':selected_epoch},source_hashes=sources)
    def panel(name,title,series,xunit,yunit,xlabel,ylabel):
        return {'panel_id':name,'title':title,'quality_status':'PENDING_CONVERGENCE','selection':'original complete native membership and original science support; no tolerance changes',
            'xlabel':xlabel,'ylabel':ylabel,'x_units':xunit,'y_units':yunit,'series':series}
    def figure(name,variant,title,panels,selected_epoch):
        title=('Synthetic control: ' if report.get('source_origin')=='synthetic_control' else '')+title+' ('+report['dark_matter_model']+')'
        return {'figure_id':name,'family':'sidm','variant':variant,'title':title,'science_status':'NOT_ACCEPTED','epoch':selected_epoch,'panels':panels}
    publish('history',history,{'elapsed_time_gyr':'Gyr','central_density':'Msun / proper kpc^3','central_supported':'dimensionless',
        'halo_ids':'identity','phase_category':'dimensionless'}, {'phase_vocabulary':vocabulary,'original_phase_classification':report['phase_classification'],
            'definition':'original fixed-aperture central density and original saved density/heat/dispersion classification; no continuous collapse time inferred'},epoch)
    figures.append(figure('sidm-core-evolution','sidm_core_evolution','Original native SIDM central density',[
        panel('core','Fixed original proper central aperture',[{'source':'history','kind':kind,'x':'elapsed_time_gyr','y':'central_density',
            'support':'central_supported','label':label,'marker_size':10} for kind,label in [('line','native history; gaps retained'),('scatter','saved measurements')]],
            'Gyr','Msun / proper kpc^3','Original elapsed time [Gyr]','Central density [Msun / proper kpc^3]')],epoch))
    figures.append(figure('sidm-collapse-phase','collapse_phase','Original SIDM saved phase diagnostics',[
        panel('phase','Categorical phase indices; definitions retained in numeric product',[{'source':'history','kind':'scatter',
            'x':'elapsed_time_gyr','y':'phase_category','label':'original source classifier','marker_size':10}],
            'Gyr','dimensionless','Original elapsed time [Gyr]','Original phase category [dimensionless]')],epoch))
    components=sorted({comparison['component'] for row in report['epochs'] for comparison in row.get('components',{}).get('dm_baryon_comparisons',[])})
    for index,component in enumerate(components):
        values={'elapsed_time_gyr':history['elapsed_time_gyr']}
        for axis in range(3):
            angles=[]
            for row in report['epochs']:
                matches=[c for c in row.get('components',{}).get('dm_baryon_comparisons',[]) if c['component']==component]
                if len(matches)>1:raise ValueError('Original SIDM component alignment duplicated')
                angle=matches[0]['unsigned_axis_misalignment_deg'][axis] if matches else None
                if angle is not None and (not np.isfinite(angle) or not 0<=angle<=90):raise ValueError('Original unsigned component misalignment invalid')
                angles.append(np.nan if angle is None else angle)
            values[f'axis_{axis}_degrees']=np.asarray(angles);values[f'axis_{axis}_supported']=np.isfinite(angles)
        name=f'sidm-component-alignment-{index:02d}'
        publish(name,values,{k:'Gyr' if k=='elapsed_time_gyr' else 'degrees' if k.endswith('_degrees') else 'dimensionless' for k in values},
            {'component':component,'definition':'original centered unreduced component-axis angles in fixed proper aperture; original unresolved eigenaxes retained',
                'uncertainty':'original finite population and centering systematics; no inferred angle covariance'},epoch)
        figures.append(figure(name,'sidm_shape_baryon_misalignment','Original DM/'+component+' shape-axis misalignment',[
            panel('angles','Unoriented corresponding principal axes; degenerate axes masked',[
                {'source':name,'kind':'scatter','x':'elapsed_time_gyr','y':f'axis_{axis}_degrees','support':f'axis_{axis}_supported',
                    'label':label,'marker_size':8} for axis,label in enumerate(['major axis','intermediate axis','minor axis'])],
                'Gyr','degrees','Original elapsed time [Gyr]','Unsigned component-axis angle [degrees]')],epoch))
    for ordinal,row in enumerate(report['epochs']):
        values,units=sidm_shell_figure_arrays(report,arrays,ordinal);name=f'shells-{ordinal:03d}'
        publish(name,values,units,{'original_measurement':row['measurement'],'physical_definitions':report['physical_definitions'],
            'kernel_display':'finite recorded diagnostic maxima retained even when science support fails; no passing reclassification'},row['epoch'])
        def series(y,support,label,error=None):
            return {'source':name,'kind':'errorbar' if error else 'scatter','x':'radius_proper_kpc','y':y,'support':support,'label':label,
                **({'error':error} if error else {'marker_size':8})}
        profiles=[]
        for key,label,error in [('density_msun_kpc3','DM density','density_shot_noise_msun_kpc3'),
                ('velocity_dispersion_squared_km2_s2','One-dimensional variance',None),('anisotropy_beta','Velocity anisotropy beta',None)]:
            profiles.append(panel(key,label,[series(key,'profile_supported',label,error)],'proper kpc',units[key],
                'Measured shell radius [proper kpc]',label+' ['+units[key]+']'))
        figures.append(figure('sidm-profiles-'+str(ordinal),'sidm_density_dispersion_anisotropy','Original native DM profiles',profiles,row['epoch']))
        figures.append(figure('sidm-heat-'+str(ordinal),'sidm_heat_flux','Original resolved kinetic heat flux',[
            panel('heat','Third central peculiar-velocity moment; outward positive',[
                series('heat_flux_radial_erg_cm2_s','transport_supported','conditional particle-sampling error','heat_flux_sampling_error_erg_cm2_s')],
                'proper kpc',units['heat_flux_radial_erg_cm2_s'],'Measured shell radius [proper kpc]','Radial kinetic heat flux [erg / cm^2 / s]')],row['epoch']))
        kernels=[]
        for key,label in [('maximum_coarse_support_fraction','Maximum coarse kernel fraction'),('maximum_omitted_scattering_upper_bound','Maximum omitted hazard bound')]:
            kernels.append(panel(key,label,[series(key,'kernel_diagnostic_available','original recorded diagnostic; failed values retained')],
                'proper kpc','dimensionless','Measured shell radius [proper kpc]',label+' [dimensionless]'))
        figures.append(figure('sidm-kernel-'+str(ordinal),'sidm_kernel_support','Native kernel contamination/omission diagnostics',kernels,row['epoch']))
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('SIDM native inputs changed during export')
    return export_scientific_suite({'sources':products,'figures':figures,'dpi':160,'maximum_panels_per_figure':3,
        'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)


# Matched collisionless/SIDM contrasts

def sidm_matched_inputs(request,*,memory_gib):
    from .sidm_analysis import _load_product,publish_sidm_matched
    from .analysis import compare_initial_ic_phases,_initial_condition_paths,load_cfg
    if request.get('schema')!='native-sidm-matched-figure-request-v1':raise ValueError('Explicit matched SIDM figure request required')
    report,arrays,sources=_load_product(request['matched_product'],'.',memory_gib*1024**3/2)
    if report.get('schema')!='sidm-matched-science-v1' or report.get('execution_status')!='COMPLETE':raise ValueError('Original completed matched SIDM product required')
    original=report['request']
    if original['memory_gib']>memory_gib:raise MemoryError('Original matched workspace exceeds allocation')
    labels={row['label'] for row in original['histories']}
    if set(request['native_runs'])!=labels:raise ValueError('Every original history requires its actual native run binding')
    base=Path(request['history_base_directory']).resolve();runs={};histories={}
    for row in original['histories']:
        path=Path(row['product']['path']);path=path if path.is_absolute() else base/path
        if sources.get(str(path.resolve()))!=row['product']['sha256']:raise ValueError('Matched history path is not an original bound source')
        run=Path(request['native_runs'][row['label']]).resolve()
        history,_,pins=sidm_history_inputs(path,run_directory=run,memory_gib=memory_gib)
        _merge(sources,pins);runs[row['label']]=run;histories[row['label']]=history
    baseline=original['baseline_label'];phases={}
    if histories[baseline]['dark_matter_model']!='collisionless' or any(
            history['dark_matter_model']!='elastic_sidm' for label,history in histories.items() if label!=baseline):
        raise ValueError('SIDM/CDM figure route requires actual collisionless baseline and elastic SIDM branches')
    configs_by_label={label:load_cfg(run) for label,run in runs.items()}
    for label,run in runs.items():
        if label==baseline:continue
        pair=[runs[baseline],run];configs=[configs_by_label[baseline],configs_by_label[label]]
        mesh=min(*(int(c['simulation']['particles_per_dimension']) for c in configs),
            *(int(c.get('validation',{}).get('ic_phase',{}).get('max_mesh_size',128)) for c in configs))
        if mesh**3*512>memory_gib*1024**3/3:raise MemoryError('Actual SIDM IC phase comparison exceeds allocation')
        for p in pair:
            for asset in [p/'config.yaml',p/'manifest.json',*_initial_condition_paths(p)]:
                _merge(sources,{str(asset.resolve()):file_sha256(asset)})
        phase=compare_initial_ic_phases(*pair,*configs)
        if phase['status']!='PASS':raise ValueError('Actual SIDM/CDM IC phases failed comparison')
        phases[label]=phase
    # Reuse the original exact membership/aperture/floor reducer; do not replace
    # it with a looser renderer-side interpretation of labels or support.
    with tempfile.TemporaryDirectory(prefix='sidm-matched-replay-') as temporary:
        path=Path(temporary)/'replay.json'
        replay=publish_sidm_matched({'request':original,'request_base':str(base)},path)
        _,replayed,_=_load_product({'path':str(path),'sha256':file_sha256(path)},path.parent,memory_gib*1024**3/2)
        if replay['comparisons']!=report['comparisons'] or replay['matching']!=report['matching']:
            raise ValueError('Original matched SIDM metadata does not replay')
        if set(replayed)!=set(arrays) or any(replayed[k].dtype!=arrays[k].dtype or replayed[k].shape!=arrays[k].shape or
            not np.array_equal(replayed[k],arrays[k],equal_nan=True) for k in arrays):
            raise ValueError('Original matched SIDM arrays do not replay')
    return report,arrays,sources,phases,histories


def matched_contrast_arrays(report,arrays,comparison,name,ordinal=None):
    """Expose an original signed contrast and its dimensionless floor scaling.

    History abscissae are Gyr; shell abscissae are proper kpc. Keep the
    producer's paired support mask and physical units on the signed delta.
    """
    value=comparison['observables'][name];contracts=report['numeric_data']['array_contract']['datasets']
    for key,unit in [(value['difference_array'],value['units']),(value['relative_difference_array'],'dimensionless'),(value['support_array'],'dimensionless')]:
        if contracts[key]['units']!=unit:raise ValueError('Original matched SIDM contrast unit differs')
    difference=np.asarray(arrays[value['difference_array']]);relative=np.asarray(arrays[value['relative_difference_array']]);support=np.asarray(arrays[value['support_array']])
    if difference.shape!=relative.shape or support.shape!=difference.shape or support.dtype!=np.dtype(bool):raise ValueError('Original matched SIDM contrast support differs')
    if ordinal is None:
        x=np.asarray(arrays['elapsed_time_gyr']);unit='Gyr'
        if contracts['elapsed_time_gyr']['units']!=unit or difference.shape!=x.shape:raise ValueError('Original central SIDM contrast time dimensions differ')
    else:
        radius=np.asarray(arrays['radius_proper_kpc'])
        if contracts['radius_proper_kpc']['units']!='proper kpc' or difference.ndim!=2 or radius.shape!=difference.shape:
            raise ValueError('Original shell SIDM contrast dimensions differ')
        x=radius[ordinal];difference=difference[ordinal];relative=relative[ordinal];support=support[ordinal];unit='proper kpc'
    if np.any(~np.isfinite(difference[support])) or np.any(~np.isfinite(relative[support])):raise ValueError('Original supported SIDM contrast nonfinite')
    return {'x':x,'signed_difference':difference,'relative_difference':relative,'supported':support},unit,value['units']


def export_native_sidm_matched(request,destination,*,memory_gib=2.):
    from core.research.display import export_scientific_suite
    report,arrays,sources,phases,histories=sidm_matched_inputs(request,memory_gib=memory_gib)
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous matched SIDM figures')
    out.mkdir(parents=True);products={};figures=[];sources[str(Path(__file__).resolve())]=file_sha256(__file__)
    baseline=histories[report['request']['baseline_label']]
    for index,comparison in enumerate(report['comparisons']):
        for observable in comparison['observables']:
            central=observable=='central_density_msun_kpc3'
            for ordinal in ([None] if central else range(len(baseline['epochs']))):
                values,xunit,yunit=matched_contrast_arrays(report,arrays,comparison,observable,ordinal)
                name=f'sidm-contrast-{index:02d}-{observable}-'+('history' if central else str(ordinal));target=out/(name+'.json');products[name]=str(target)
                epoch={'canonical_time':'original matched saved epochs','saved_native_epochs':[row['epoch'] for row in baseline['epochs']]} if central else baseline['epochs'][ordinal]['epoch']
                export_numeric_audit(target,{'schema':'native-sidm-matched-figure-v1','epoch':epoch,'original_comparison':comparison,
                    'observable':observable,'actual_phase_comparison':phases[comparison['label']],
                    'relative_definition':'original signed difference divided by max(abs(baseline), original frozen dimensional floor); not a ratio',
                    'uncertainty':'original paired measurement only; no inferred independent ensemble covariance/error bars',
                    'science_acceptance':'NOT_ACCEPTED'},values,{k:{'units':xunit if k=='x' else yunit if k=='signed_difference' else 'dimensionless',
                        'quantity':k,'missing_policy':'original paired support masks retained'} for k in values},algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
                panels=[]
                for key,unit,title in [('signed_difference',yunit,'Signed branch-minus-baseline difference'),('relative_difference','dimensionless','Original floor-scaled signed difference')]:
                    panels.append({'panel_id':key,'title':title,'quality_status':'PENDING_CONVERGENCE',
                        'selection':comparison['interpretation'],'xlabel':('Original elapsed time' if central else 'Original shell radius')+' ['+xunit+']',
                        'ylabel':key+' ['+unit+']','x_units':xunit,'y_units':unit,'series':[{'source':name,'kind':'scatter',
                            'x':'x','y':key,'support':'supported','label':comparison['label']+' minus '+comparison['baseline_label'],'marker_size':8}]})
                figures.append({'figure_id':name,'family':'sidm','variant':'sidm_cdm_controls',
                    'title':('Synthetic control: ' if report.get('source_origin')=='synthetic_control' else '')+comparison['interpretation']+': '+observable,
                    'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':panels})
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('Native matched SIDM inputs changed during export')
    return export_scientific_suite({'sources':products,'figures':figures,'dpi':160,'maximum_panels_per_figure':2,
        'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)


# Satellite, calibrated-mode and observational figures

def extended_inputs(request,*,memory_gib):
    from .sidm_analysis import _load_product,_stage_native_bindings,publish_sidm_derived,publish_sidm_observations
    required={'schema','operation','product','producer_base_directory','native_runs','phase_baseline_run_directory'}
    if set(request)!=required or request['schema']!='native-sidm-extended-figures-v1' or request['operation'] not in {'satellite','degeneracy','observations'}:
        raise ValueError('Complete original SIDM derived/observational figure request required')
    base=Path(request['producer_base_directory']).resolve();report,arrays,pins=_load_product(request['product'],base,memory_gib*1024**3/2)
    operation=request['operation'];expected='sidm-observational-comparison-v1' if operation=='observations' else 'sidm-derived-science-v1'
    if report.get('schema')!=expected or report.get('execution_status')!='COMPLETE' or operation!='observations' and report.get('product')!=operation:
        raise ValueError('Original completed SIDM producer differs from requested figure operation')
    owner=Path(__file__).with_name('sidm.py').resolve()
    if pins.get(str(owner))!=file_sha256(owner):raise ValueError('Original SIDM numerical producer source binding required')
    bindings=request['native_runs']
    if not isinstance(bindings,list) or not bindings:raise ValueError('Actual completed native generation bindings required')
    runs=[]
    for binding in bindings:
        if set(binding)!={'run_directory','stage_receipt'}:raise ValueError('Exact native run/stage receipt binding required')
        receipt=Path(binding['stage_receipt']['path']).resolve()
        if file_sha256(receipt)!=binding['stage_receipt']['sha256']:raise ValueError('Original completed native stage receipt changed')
        run=Path(binding['run_directory']).resolve()
        if run in runs:raise ValueError('Duplicate native SIDM derived run binding')
        native=_stage_native_bindings({'request':{'memory_gib':memory_gib},'native_run_directory':str(run),'native_stage_receipt':str(receipt)})
        if any(pins.get(p)!=h for p,h in native.items()):raise ValueError('SIDM derived product omits original native evolution bindings')
        _merge(pins,native);runs.append(run)
    phases={}
    if operation=='observations':
        from .analysis import load_cfg
        from .analysis import compare_initial_ic_phases,_initial_condition_paths
        baseline=Path(request['phase_baseline_run_directory']).resolve() if request['phase_baseline_run_directory'] is not None else None
        if baseline not in runs:raise ValueError('Actual collisionless phase baseline required for observational SIDM response')
        configs={p:load_cfg(p) for p in runs}
        if configs[baseline].get('sidm',{}).get('enabled',False) is not False:raise ValueError('Observational baseline is not collisionless')
        sidm=[p for p in runs if p!=baseline and configs[p].get('sidm',{}).get('enabled',False) is True]
        if not sidm or len(sidm)!=len(runs)-1:raise ValueError('Actual enabled SIDM comparison branches required')
        for run in sidm:
            cfg=[configs[baseline],configs[run]];mesh=min(*(int(c['simulation']['particles_per_dimension']) for c in cfg),*(int(c.get('validation',{}).get('ic_phase',{}).get('max_mesh_size',128)) for c in cfg))
            if mesh**3*512>memory_gib*1024**3/3:raise MemoryError('Actual SIDM observational IC phase comparison exceeds allocation')
            for directory in (baseline,run):
                for path in [directory/'config.yaml',directory/'manifest.json',*_initial_condition_paths(directory)]:_merge(pins,{str(path.resolve()):file_sha256(path)})
            result=compare_initial_ic_phases(baseline,run,*cfg)
            if result['status']!='PASS':raise ValueError('Actual SIDM/CDM observational IC phases failed comparison')
            phases[str(run)]=result
    elif request['phase_baseline_run_directory'] is not None:raise ValueError('Single-product derived figures do not accept an unused phase baseline')
    # Recompute with the original calibrated domain, source masks and estimator.
    # A sealed but numerically altered downstream result cannot become a figure.
    function=publish_sidm_observations if operation=='observations' else publish_sidm_derived
    with tempfile.TemporaryDirectory(prefix='sidm-extended-replay-') as temporary:
        target=Path(temporary)/'replay.json';replay=function({'request':report['request'],'request_base':str(base)},target)
        sealed,values,_=_load_product({'path':str(target),'sha256':file_sha256(target)},target.parent,memory_gib*1024**3/2)
        if set(values)!=set(arrays) or any(values[k].dtype!=arrays[k].dtype or values[k].shape!=arrays[k].shape or not np.array_equal(values[k],arrays[k],equal_nan=True) for k in arrays):
            raise ValueError('Original SIDM derived/observational arrays do not replay')
        for name in arrays:
            original=report['numeric_data']['array_contract']['datasets'][name];expected=sealed['numeric_data']['array_contract']['datasets'][name]
            if original['units']!=expected['units'] or original.get('identity_namespace')!=expected.get('identity_namespace'):
                raise ValueError('Original SIDM derived/observational array units do not replay')
        key='comparisons' if operation=='observations' else 'result'
        if replay[key]!=report[key]:raise ValueError('Original SIDM derived/observational metadata does not replay')
    return report,arrays,pins,phases


def _publish(path,report,arrays,units,pins):
    return export_numeric_audit(path,report,arrays,{k:{'units':u,'quantity':k,'missing_policy':'Original support/NaNs, calibration and matched selection retained'} for k,u in units.items()},
        algorithm=__file__,context={'epoch':report['epoch']},source_hashes=pins)


def export_native_sidm_extended(request,destination,*,memory_gib=2.):
    from core.research.display import export_scientific_suite
    from .sidm_analysis import _load_product
    report,arrays,pins,phases=extended_inputs(request,memory_gib=memory_gib);out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous extended SIDM figures')
    out.mkdir(parents=True);_merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    products={};figures=[];epoch={'meaning':'actual original producer epoch/selection','original':report.get('epoch'),'source_request':report['request']}
    def emit(name,variant,title,values,units,metadata,panels):
        path=out/(name+'.json');products[name]=str(path)
        _publish(path,{'schema':'native-sidm-extended-figure-v1','epoch':epoch,'science_acceptance':'NOT_ACCEPTED','original_producer':report,
            'actual_phase_comparisons':phases,**metadata},values,units,pins)
        figures.append({'figure_id':name,'family':'sidm','variant':variant,'title':title,'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':panels})
    def panel(name,title,x,y,xunit,yunit,support,label):
        series={'source':name,'kind':'scatter','x':x,'y':y,'label':label,'marker_size':10}
        if support is not None:series['support']=support
        return {'panel_id':y,'title':title,'quality_status':'CONDITIONAL_ORIGINAL_SUPPORT','selection':'Original native producer selection and unchanged masks',
            'xlabel':x+' ['+xunit+']','ylabel':y+' ['+yunit+']','x_units':xunit,'y_units':yunit,'series':[series]}
    if request['operation']=='satellite':
        ref=report['request']['inputs'][-1];current,_,sources=_load_product(ref,request['producer_base_directory'],memory_gib*1024**3/4);_merge(pins,sources)
        time=float(current['elapsed_time_gyr'])
        if not np.isfinite(time):raise ValueError('Original native satellite proper time required')
        values={'actual_time_gyr':np.asarray([time]),**{k:np.asarray(arrays[k]) for k in ('retained_original_bound_mass_fraction','current_bound_mass_msun','resolved_survival_supported')}}
        if any(v.shape!=(1,) for v in values.values()) or values['resolved_survival_supported'].dtype!=np.dtype(bool):raise ValueError('Original satellite scalar/support shape differs')
        units={'actual_time_gyr':'Gyr','retained_original_bound_mass_fraction':'dimensionless','current_bound_mass_msun':'Msun','resolved_survival_supported':'dimensionless'}
        name='native-sidm-satellite-survival';panels=[panel(name,'Exact initial cohort retention; missing members remain unknown','actual_time_gyr','retained_original_bound_mass_fraction','Gyr','dimensionless',None,'original bound-ID retention'),
            panel(name,'Current selected bound population; original minimum count gate','actual_time_gyr','current_bound_mass_msun','Gyr','Msun','resolved_survival_supported','current bound population')]
        emit(name,'sidm_satellite_survival','Native satellite bound-mass retention',values,units,
            {'definition':report['result'],'attribution':'Observed binding/cohort retention; evaporation candidates are not proof of collision causality'},panels)
    elif request['operation']=='degeneracy':
        result=report['result'];n=len(result['parameter_names']);names=result['parameter_names']
        singular=np.asarray(arrays['whitened_singular_values']);modes=np.asarray(arrays['parameter_modes_rows']);resolved=np.asarray(arrays['resolved_modes'])
        if singular.shape!=(n,) or modes.shape!=(n,n) or resolved.shape!=(n,) or resolved.dtype!=np.dtype(bool):raise ValueError('Original calibrated SVD mode shapes differ')
        if not np.isfinite(singular).all() or np.any(singular<0) or not np.isfinite(modes).all():raise ValueError('Invalid original calibrated SVD modes')
        values={'mode_index':np.arange(n,dtype=np.int64),'parameter_index':np.arange(n,dtype=np.int64),'whitened_singular_value':singular,'resolved':resolved}
        for i,row in enumerate(modes):values[f'mode_{i:03d}']=row
        unit='dimensionless in declared scaled parameter coordinates';units={k:'dimensionless' if k.endswith('index') or k=='resolved' else unit for k in values};name='native-sidm-baryonic-degeneracy'
        panels=[panel(name,'All original singular values; unresolved modes retained','mode_index','whitened_singular_value','dimensionless',unit,None,'calibrated local SVD spectrum')]
        for i in range(n):panels.append(panel(name,'Original scaled-parameter mode '+str(i),'parameter_index',f'mode_{i:03d}','dimensionless',unit,None,'mode loading; parameter order in numerical product'))
        emit(name,'sidm_baryonic_degeneracy','Calibrated SIDM/baryon local response modes',values,units,{'parameter_order':names,
            'parameter_scales':result['parameter_scale_definition'],'calibration':result['calibration_receipts'],'scope':result['scope'],'posterior_acceptance':'NOT_EVALUATED'},panels)
    else:
        for i,comparison in enumerate(report['comparisons']):
            family=comparison['family'];variant='sidm_stream_profile_response' if family=='streams' else 'sidm_lensing_h0_response'
            if family not in {'streams','lensing','h0_estimator'}:raise ValueError('Original matched SIDM forward family differs')
            for j,observable in enumerate(comparison['observables']):
                prefix=observable['array_prefix'];difference=np.asarray(arrays[prefix+'_difference']);relative=np.asarray(arrays[prefix+'_relative_difference']);support=np.asarray(arrays[prefix+'_supported'])
                if difference.shape!=relative.shape or support.shape!=difference.shape or support.dtype!=np.dtype(bool):raise ValueError('Original observational contrast/support shapes differ')
                if not np.isfinite(difference[support]).all() or not np.isfinite(relative[support]).all():raise ValueError('Supported SIDM observational contrast is nonfinite')
                values={'original_bin_index':np.arange(difference.size,dtype=np.int64),'signed_difference':difference.reshape(-1),'floor_scaled_difference':relative.reshape(-1),'supported':support.reshape(-1)}
                units={'original_bin_index':'dimensionless','signed_difference':observable['units'],'floor_scaled_difference':'dimensionless','supported':'dimensionless'}
                name=f'native-sidm-forward-{i:03d}-{j:03d}';panels=[panel(name,'Original producer bin ordering; physical bin definitions retained in source','original_bin_index','signed_difference','dimensionless',observable['units'],'supported',comparison['comparison_id']),
                    panel(name,'Original dimensional floor scaling; signed difference, not a ratio','original_bin_index','floor_scaled_difference','dimensionless','dimensionless','supported',comparison['comparison_id'])]
                emit(name,variant,'Matched native SIDM '+family+' response',values,units,{'matching':comparison['matching'],'original_array_shape':list(difference.shape),
                    'source_observable':observable,'uncertainty':'Original paired forward measurements; no independent-run covariance or new uncertainty inferred',
                    'h0_scope':report['h0_scope']},panels)
    for path,checksum in pins.items():
        if file_sha256(path)!=checksum:raise ValueError('SIDM extended source changed during export')
    return export_scientific_suite({'sources':products,'figures':figures,'dpi':160,'maximum_panels_per_figure':max(len(f['panels']) for f in figures),'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)


# Campaign inputs and export dispatch

def runtime_request(parameters):
    """Bind actual resolved producer generations; no guessed future hashes."""
    request=copy.deepcopy(parameters['request']);operation=parameters['operation']
    if operation=='extended':
        request['product']={'path':str(Path(parameters['product']).resolve()),'sha256':file_sha256(parameters['product'])}
        if 'native_runs' in parameters:
            request['native_runs']=[{'run_directory':str(Path(row['run_directory']).resolve()),
                'stage_receipt':{'path':str(Path(row['stage_receipt']).resolve()),'sha256':file_sha256(row['stage_receipt'])}}
                for row in parameters['native_runs']]
    elif operation=='centroids':
        request['history']={'path':str(Path(parameters['product']).resolve()),'sha256':file_sha256(parameters['product'])}
        request['run_directory']=parameters['run_directory']
        if 'assembly_product' in request:
            path=request.pop('assembly_product');request['assembly']={'path':str(Path(path).resolve()),'sha256':file_sha256(path)}
        for row in request['observations']:
            if 'structure_product' in row:
                path=row.pop('structure_product');row['structure']={'path':str(Path(path).resolve()),'sha256':file_sha256(path)}
    elif operation=='matched':
        request['matched_product']={'path':str(Path(parameters['product']).resolve()),'sha256':file_sha256(parameters['product'])}
    return request


def sidm_export_inputs(parameters):
    """Resolve the native inputs used by the selected export operation."""
    memory=parameters['memory_gib'];operation=parameters['operation']
    if operation in {'history','scattering'}:
        return sidm_history_inputs(parameters['product'],run_directory=parameters['run_directory'],memory_gib=memory)[2]
    if operation=='extended':
        return extended_inputs(runtime_request(parameters),memory_gib=memory)[2]
    if operation=='centroids':
        return centroid_inputs(runtime_request(parameters),memory_gib=memory)[3]
    if operation=='matched':
        return sidm_matched_inputs(runtime_request(parameters),memory_gib=memory)[2]
    raise ValueError('Unknown native SIDM producer/export operation')


def native_sidm_export_runner(parameters,context):
    """Export available measurements and report requested variants with no data."""
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('SIDM figure producer exceeds stage allocation')
    operation=parameters['operation'];memory=parameters['memory_gib'];destination=context.generation/'figures'
    if operation=='history':
        result=export_native_sidm_history(parameters['product'],destination,run_directory=parameters['run_directory'],memory_gib=memory)
    elif operation=='scattering':
        result=export_native_scattering_rates(parameters['product'],destination,run_directory=parameters['run_directory'],memory_gib=memory)
    elif operation=='centroids':
        result=export_native_sidm_centroids(runtime_request(parameters),destination,memory_gib=memory)
    elif operation=='extended':
        result=export_native_sidm_extended(runtime_request(parameters),destination,memory_gib=memory)
    elif operation=='matched':
        result=export_native_sidm_matched(runtime_request(parameters),destination,memory_gib=memory)
    else:raise ValueError('Unknown native SIDM producer/export operation')
    actual=sorted({row['variant'] for row in result['figures']});expected=parameters.get('expected_variants',[])
    summary={'execution_status':'COMPLETE','figure_count':len(result['figures']),
        'variants':actual,'unresolved_variants':[{'variant':v,'reason':'Original native producer has no measurement for this requested variant; no detection or zero-valued observable invented'} for v in expected if v not in actual],
        'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(context.generation/'sidm-export.json',summary)
    return {**summary,'measurements':{'sidm_graphs':summary}}


# Fixed native cohort: exact counter increments, recorded exposure and figures

def _scattering_state(record):
    ids=exact_uint64_ids(record['particle_ids'],allow_empty=True)
    n=len(ids);available=np.asarray(record['available'])
    counts=np.asarray(record['interaction_count']);exposure=np.asarray(record['exposure']);mass=np.asarray(record['mass_msun'])
    if available.shape!=(n,) or available.dtype!=np.dtype(bool):raise ValueError('Exact aligned native availability mask required')
    if counts.shape!=(n,) or counts.dtype!=np.dtype('uint64'):raise ValueError('Original uint64 native interaction counters required')
    if exposure.shape!=(n,) or mass.shape!=(n,) or exposure.dtype.kind!='f' or mass.dtype.kind!='f':raise ValueError('Aligned physical native mass and exposure arrays required')
    if not np.isfinite(exposure[available]).all() or np.any(exposure[available]<0) or not np.isfinite(mass[available]).all() or np.any(mass[available]<=0):
        raise ValueError('Finite nonnegative exposure and positive native masses required')
    return ids,available,counts,exposure,mass


def fixed_cohort_interval(before,after,*,elapsed_interval_gyr,physics_interval_supported):
    """Subtract integers before float conversion; incomplete totals stay NaN.

    Per particle, r_i = (C_i,after - C_i,before) / dt and h_i = delta tau_i / dt,
    in Gyr^-1. Aggregate rates use initial-mass weights m_i/sum(m_i), only
    for a complete cohort. C counts native rare-event participations; tau
    includes rare and frequent optical depth, so h is not generally a rare
    collision-count prediction. Exact uint64 increments are also saved.

    Both states must be aligned to the same original exact-ID cohort, with
    missing states masked. The caller must separately certify interval physics
    against its original kernel/coarse-support criteria; endpoint diagnostics
    alone cannot certify every intermediate scattering step.
    """
    if type(elapsed_interval_gyr) not in (int,float) or not np.isfinite(elapsed_interval_gyr) or elapsed_interval_gyr<=0:
        raise ValueError('Positive actual proper elapsed interval in Gyr required')
    if type(physics_interval_supported) is not bool:raise ValueError('Explicit independently certified interval physics support required')
    ids,p0,c0,e0,m0=_scattering_state(before);other,p1,c1,e1,m1=_scattering_state(after)
    if not np.array_equal(ids,other):raise ValueError('Native scattering intervals require identical fixed exact-ID order')
    available=p0&p1
    if np.any(c1[available]<c0[available]) or np.any(e1[available]<e0[available]):raise ValueError('Native cumulative scattering counter/exposure regressed')
    if not np.array_equal(m0[available],m1[available]):raise ValueError('Fixed elastic fine-DM cohort mass changed')
    delta=np.zeros(len(ids),np.uint64);delta[available]=c1[available]-c0[available]
    depth=np.full(len(ids),np.nan);depth[available]=e1[available]-e0[available]
    rate=np.full(len(ids),np.nan);rate[available]=delta[available].astype(float)/elapsed_interval_gyr
    hazard=depth/elapsed_interval_gyr
    if not np.isfinite(rate[available]).all() or not np.isfinite(hazard[available]).all():raise ValueError('Native scattering rate overflow')
    complete=bool(len(ids)>0 and available.all());supported=complete and physics_interval_supported
    if complete:
        # Scale the weights before reducing to avoid overflow; the actual
        # individual masses and uint64 increments are retained unchanged.
        weight=m0/np.max(m0);weight=weight/np.sum(weight)
        measured=float(np.dot(weight,rate));expected=float(np.dot(weight,hazard))
    else:measured=expected=np.nan
    summary={'original_cohort_count':len(ids),'observed_interval_count':int(available.sum()),
        'complete_cohort':complete,'physics_interval_supported':physics_interval_supported,
        'aggregate_supported':supported,'diagnostic_complete_cohort_rate_per_gyr':measured,
        'diagnostic_complete_cohort_expected_rate_per_gyr':expected,
        'supported_rate_per_gyr':measured if supported else np.nan,
        'supported_expected_rate_per_gyr':expected if supported else np.nan,
        'definition':'initial-mass-weighted fixed exact-ID cohort native participation increments / actual proper elapsed Gyr; no unique pair-count claim',
        'uncertainty':'realized counter rate and native integrated expected depth are separate diagnostics; neither a Poisson error bar nor independent-run uncertainty is inferred'}
    return summary,{'particle_ids':ids,'interval_observed':available,'interaction_count_increment':delta,
        'exposure_increment':depth,'participation_rate_per_gyr':rate,'expected_rate_per_gyr':hazard,
        'physics_supported':available&physics_interval_supported}


def _guard_state(run,snapshot,ids,*,memory_gib):
    """Read the native per-step admission contract; never infer it from endpoints."""
    import h5py
    from .halos import _snapshot_files
    ids=exact_uint64_ids(ids,allow_empty=True);order=np.argsort(ids);sorted_ids=ids[order]
    version=np.full(len(ids),-1,np.int64);seen=np.zeros(len(ids),bool);limits=None;pins={}
    names=('SIDMReferenceMaximumKernelCoarseSupportFraction','SIDMReferenceMaximumOmittedScatteringUpperBound')
    step=max(1,min(131072,int(memory_gib*1024**3/512)))
    for file in _snapshot_files(Path(run),snapshot):
        path=Path(file).resolve();_merge(pins,{str(path):file_sha256(path)})
        with h5py.File(path) as f:
            current=[float(f['Parameters'].attrs[n]) if n in f['Parameters'].attrs else None for n in names]
            if limits is not None and current!=limits:raise ValueError('Native SIDM guard limits differ across snapshot shards')
            limits=current;group=f.get('PartType1')
            if group is None:continue
            for start in range(0,len(group['ParticleIDs']),step):
                native=exact_uint64_ids(group['ParticleIDs'][start:start+step]);loc=np.searchsorted(sorted_ids,native)
                keep=loc<len(ids)
                if len(ids):keep&=sorted_ids[np.minimum(loc,len(ids)-1)]==native
                else:keep[:]=False
                positions=order[loc[keep]]
                if seen[positions].any():raise ValueError('Duplicate native fixed-cohort guard identity')
                seen[positions]=True
                if 'SIDMCoarseGuardVersion' in group:
                    value=np.asarray(group['SIDMCoarseGuardVersion'][start:start+step])
                    if value.dtype.kind not in 'iu' or value.shape!=native.shape or np.any(value>1):raise ValueError('Unsupported native coarse-guard version')
                    version[positions]=value[keep]
    return version,limits,pins


def native_scattering_history(product,run_directory,*,memory_gib):
    from .observation_geometry import read_native_gravitating_particles
    from .sidm_analysis import _native_diagnostics
    history,stored,pins=sidm_history_inputs(product,run_directory=run_directory,memory_gib=memory_gib)
    pins=dict(pins)
    if history['dark_matter_model']!='elastic_sidm':raise ValueError('Recorded scattering rates require an actual enabled native SIDM history')
    ids=exact_uint64_ids(stored['first_saved_dm_member_ids'],allow_empty=True);epochs=history['epochs'];request=history['request']
    declaration=history['numeric_data']['array_contract']['datasets']['first_saved_dm_member_ids']
    if declaration['units']!='identity' or declaration.get('identity_namespace')!=request['particle_namespace']:
        raise ValueError('Original fixed SIDM cohort must retain exact native identity units/namespace')
    if len(epochs)<2:raise ValueError('Scattering rates require two actual saved native epochs')
    if len(ids)*len(epochs)*384>memory_gib*1024**3/2:raise MemoryError('Fixed-cohort native rate histories exceed allocation')
    states=[];guards=[];limits=[];coarse=[];omitted=[]
    for epoch in epochs:
        native,arrays=read_native_gravitating_particles(run_directory,epoch['snapshot'],species=[1],maximum_particles=request['maximum_particles'],
            memory_gib=memory_gib/4,namespace=request['particle_namespace']);_merge(pins,native['source_hashes'])
        original=exact_uint64_ids(arrays['particle_ids'],allow_empty=True);order=np.argsort(original);positions=np.searchsorted(original[order],ids)
        present=positions<len(original)
        if len(original):present&=original[order][np.minimum(positions,len(original)-1)]==ids
        else:present[:]=False
        mass=np.full(len(ids),np.nan);mass[present]=np.asarray(arrays['mass_msun'])[order[positions[present]]]
        diagnostic=_native_diagnostics(run_directory,epoch['snapshot'],ids[present],{**request,'memory_gib':memory_gib/4,'require_sidm_diagnostics':True}) if present.any() else {}
        counts=np.zeros(len(ids),np.uint64);exposure=np.full(len(ids),np.nan);c=np.full(len(ids),np.nan);o=c.copy()
        if present.any():
            raw=np.asarray(diagnostic['interaction_count'])
            if raw.dtype!=np.dtype('uint64'):raise ValueError('Native scattering counters must remain exact uint64')
            counts[present]=raw;exposure[present]=diagnostic['exposure'];c[present]=diagnostic['coarse_support_fraction'];o[present]=diagnostic['omitted_scattering_upper_bound']
        states.append({'particle_ids':ids,'available':present,'mass_msun':mass,'interaction_count':counts,'exposure':exposure})
        version,cap,sources=_guard_state(run_directory,epoch['snapshot'],ids,memory_gib=memory_gib/4);_merge(pins,sources)
        guards.append(version);limits.append(cap);coarse.append(c);omitted.append(o)
    # Version 1 has a collective rejection at every native scattering step.
    # Require original completed evolution to bind this implementation source;
    # merely possessing a snapshot field with the right name is insufficient.
    owner=Path(__file__).resolve().parent/'software/gadget4/src/sidm/research_distributed_step.cc'
    operator_bound=pins.get(str(owner))==file_sha256(owner)
    cap=[request['maximum_coarse_support_fraction'],request['maximum_omitted_scattering_upper_bound']]
    from .analysis import _proper_clock_error_bounds_myr
    clock_errors=_proper_clock_error_bounds_myr([e['epoch'] for e in epochs])/1000.
    for i,e in enumerate(epochs):
        if e['epoch'].get('canonical_time')=='isolated_native_elapsed_time':clock_errors[i]=0.
    rows=[];packed={};time=np.asarray([e['elapsed_time_gyr'] for e in epochs],float)
    for i,(before,after) in enumerate(zip(states[:-1],states[1:])):
        caps=limits[i+1]
        admitted=bool(operator_bound and caps is not None and all(v is not None and np.isfinite(v) and 0<=v<=ceiling for v,ceiling in zip(caps,cap))
            and caps[0]<=1 and np.all(guards[i+1]==1)
            and np.all((guards[i]==1)|((guards[i]==0)&(before['interaction_count']==0)&(before['exposure']==0)))
            and np.all(np.isfinite(coarse[i+1])) and np.all(coarse[i+1]<=cap[0])
            and np.all(np.isfinite(omitted[i+1])) and np.all(omitted[i+1]<=cap[1]))
        operator_admitted=admitted;dt=float(time[i+1]-time[i]);error=float(clock_errors[i]+clock_errors[i+1])
        clock_admitted=bool(np.isfinite(error) and error<dt)
        row,values=fixed_cohort_interval(before,after,elapsed_interval_gyr=dt,physics_interval_supported=admitted and clock_admitted)
        row.update(operator_interval_guard_supported=operator_admitted,clock_interval_supported=clock_admitted,
            duration_numerical_error_bound_gyr=error if np.isfinite(error) else None,
            conditional_duration_bracket_gyr=[dt-error,dt+error] if np.isfinite(error) else None)
        row.update(snapshot_bracket=[epochs[i]['snapshot'],epochs[i+1]['snapshot']],operator_guard_source_bound=operator_bound,
            native_per_step_guard_limits=caps,original_support_ceilings=cap)
        rows.append(row)
        for key,value in values.items():
            if key!='particle_ids':packed.setdefault(key,[]).append(value)
    packed={k:np.stack(v) for k,v in packed.items()};packed['particle_ids']=ids
    for key in ['available','mass_msun','interaction_count','exposure']:packed['saved_'+key]=np.stack([s[key] for s in states])
    packed['saved_guard_version']=np.stack(guards);packed['saved_coarse_support_fraction']=np.stack(coarse);packed['saved_omitted_scattering_upper_bound']=np.stack(omitted)
    packed['saved_time_gyr']=time;packed['interval_end_gyr']=time[1:];packed['actual_time_brackets_gyr']=np.column_stack([time[:-1],time[1:]])
    for name in ('complete_cohort','physics_interval_supported','aggregate_supported','diagnostic_complete_cohort_rate_per_gyr',
                 'diagnostic_complete_cohort_expected_rate_per_gyr','supported_rate_per_gyr','supported_expected_rate_per_gyr'):
        packed[name]=np.asarray([r[name] for r in rows])
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    for path,checksum in pins.items():
        if file_sha256(path)!=checksum:raise ValueError('Native scattering source changed during measurement')
    return {'schema':'native-sidm-fixed-cohort-rates-v1','execution_status':'COMPLETE','source_origin':history.get('source_origin','native'),
        'epoch':{'saved_native_epochs':[e['epoch'] for e in epochs]},'source_history':str(Path(product).resolve()),
        'particle_namespace':request['particle_namespace'],'intervals':rows,'science_acceptance':'NOT_ACCEPTED',
        'definition':'fixed first-saved fine-DM membership; every original ID tracked across full native snapshots, including particles leaving the original aperture',
        'native_operator_semantics':'SIDMInteractionCount records native rare-event participations; SIDMExposure integrates rare plus frequent angular-domain optical depth. Optical-depth rate is not an expected rare-event count in hybrid/frequent schemes; no Poisson equality or unresolved microscopic collision count is inferred.',
        'missing_policy':'uint64 zero placeholders for absent count differences are explicitly masked; unavailable rates/exposures and incomplete totals NaN; no partial-census normalization',
        'physics_admission':'original completed-source-bound native version1 per-step coarse guard and caps at least as strict as original history, plus a finite source clock envelope with positive lower duration bound; absent or insufficient evidence masks support'},packed,pins


def export_native_scattering_rates(product,destination,*,run_directory,memory_gib=2.):
    from core.research.display import export_scientific_suite
    from core.stream_common import json_safe
    report,arrays,pins=native_scattering_history(product,run_directory,memory_gib=memory_gib)
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native scattering rates')
    out.mkdir(parents=True);path=out/'scattering-rates.json';specs={}
    for key in arrays:
        unit='identity' if key=='particle_ids' else 'Msun' if key=='saved_mass_msun' else 'Gyr^-1' if key.endswith('_per_gyr') else 'Gyr' if key.endswith('_gyr') else 'dimensionless'
        specs[key]={'units':unit,'quantity':key,'missing_policy':report['missing_policy']}
        if key=='particle_ids':specs[key]['identity_namespace']=report['particle_namespace']
    # Unsupported interval rates remain NaN in the numerical sidecar. JSON
    # metadata records their absence as null; the support gate is unchanged.
    export_numeric_audit(path,json_safe(report),arrays,specs,algorithm=__file__,context={'epoch':report['epoch']},source_hashes=pins)
    panels=[]
    for flag,prefix,title in [('complete_cohort','diagnostic_complete_cohort_','Recorded fixed-cohort diagnostics; physical admission separate'),('aggregate_supported','supported_','Original native interval physics gate retained')]:
        panels.append({'panel_id':flag,'title':title,'quality_status':'PENDING_CONVERGENCE','selection':report['definition'],
            'xlabel':'Actual interval end [Gyr]','ylabel':'Native rare-event or optical-depth rate [Gyr^-1]','x_units':'Gyr','y_units':'Gyr^-1',
            'series':[{'source':'rates','kind':'scatter','x':'interval_end_gyr','y':prefix+suffix,'support':flag,'label':label,'marker_size':9}
                for suffix,label in [('rate_per_gyr','native rare-event participations'),('expected_rate_per_gyr','rare + frequent optical depth')]]})
    return export_scientific_suite({'sources':{'rates':str(path)},'figures':[{'figure_id':'native-sidm-scattering-rates','family':'sidm','variant':'scattering_rates',
        'title':('Synthetic control: ' if report['source_origin']=='synthetic_control' else '')+'Native fixed-cohort scattering rates','science_status':'NOT_ACCEPTED','epoch':report['epoch'],'panels':panels}],
        'dpi':160,'maximum_panels_per_figure':2,'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)


# Component centroid figures joined to recorded native ancestry

def centroid_inputs(request,*,memory_gib):
    from .sidm_analysis import _load_product
    from .halos import IdentityQueries,verify_halo_structure
    from .assembly import reconstruct_quality_masked_main_branch
    from .analysis import _bind_structure_graph_observation
    from . import source_path
    required={'schema','history','run_directory','producer_base_directory','assembly','identity_graph_directory','observations'}
    if set(request)!=required or request['schema']!='native-sidm-centroid-events-v1':raise ValueError('Complete original SIDM centroid/native-merger join request required')
    path=Path(request['history']['path']).resolve()
    if file_sha256(path)!=request['history']['sha256']:raise ValueError('Native SIDM centroid history changed')
    history,_,pins=sidm_history_inputs(path,run_directory=request['run_directory'],memory_gib=memory_gib)
    pins=dict(pins)
    root=Path(request['identity_graph_directory']).resolve()
    if sum((root/n).stat().st_size for n in ('report.json','index.sqlite','nodes.npz'))*24>memory_gib*1024**3/3:raise MemoryError('Native SIDM centroid graph exceeds allocation')
    graph=IdentityQueries(root);_merge(pins,graph.report['sources'])
    for name in ('report.json','report.sha256','index.sqlite','nodes.npz'):_merge(pins,{str(root/name):file_sha256(root/name)})
    assembly,_,sources=_load_product(request['assembly'],request['producer_base_directory'],memory_gib*1024**3/3);_merge(pins,sources)
    owner=source_path('assembly')
    if assembly.get('schema')!='halo-main-branch-v1' or pins.get(str(owner))!=file_sha256(owner) or assembly['source_graph_fingerprint']!=graph.report['fingerprint']:
        raise ValueError('Original source-bound matching native assembly graph required')
    parameters=assembly['request']['parameters'];replay,_=reconstruct_quality_masked_main_branch(graph.report,parameters['target'],parameters['branch'])
    if replay['native_mergers']!=assembly['native_mergers'] or replay['branch_identity_history']!=assembly['branch_identity_history']:
        raise ValueError('Original native assembly merger selection does not replay')
    rows=request['observations']
    if not isinstance(rows,list) or len(rows)!=len(history['epochs']):raise ValueError('Every SIDM centroid epoch needs exact native graph/structure identity')
    branch={tuple(k) for k in replay['branch_identity_history']};nodes={tuple(n['key']):n for n in graph.report['nodes']};previous=None;keys=[]
    for row,record in zip(rows,history['epochs']):
        if set(row)!={'structure','graph_node'}:raise ValueError('Exact centroid structure and graph observation required')
        key=_key(row['graph_node']);product=Path(row['structure']['path']).resolve()
        if file_sha256(product)!=row['structure']['sha256']:raise ValueError('Native centroid host structure changed')
        structure=verify_halo_structure(product);_merge(pins,numeric_audit_source_bindings(structure));_merge(pins,{str(product):file_sha256(product)})
        if key not in branch or key[2]!=record['snapshot'] or structure['request']['snapshot']!=record['snapshot'] or Path(structure['request']['run_directory']).resolve()!=Path(request['run_directory']).resolve():
            raise ValueError('Native centroid host/merger history changes run, branch or exact saved epoch')
        _bind_structure_graph_observation(structure,key,graph,nodes,previous);previous=key;keys.append(key)
        original=next(e for e in history['request']['epochs'] if e['snapshot']==record['snapshot'])
        selection,_,sources=_load_product(original['selection'],request['producer_base_directory'],memory_gib*1024**3/4);_merge(pins,sources)
        if selection.get('schema')!='native-halo-membership-measurement-v1' or selection['request']['host_row']!=structure['request']['host_row']:
            raise ValueError('SIDM centroid population and merger selection use different native FOF hosts')
        center=np.asarray(structure['source_context']['center_comoving_kpc_h'])*structure['source_context']['scale_factor']/structure['source_context']['h']
        if not np.allclose(center,record['center_proper_kpc'],rtol=1e-12,atol=1e-12):raise ValueError('Original native centroid/host center differs')
        if record['epoch'].get('cosmic_time_gyr')!=structure['source_context']['epoch'].get('cosmic_time_gyr') or record['epoch']['actual_scale_factor']!=structure['source_context']['epoch']['actual_scale_factor']:
            raise ValueError('Native centroid/merger clock differs')
    return history,replay,keys,pins


def centroid_event_arrays(history,assembly,keys):
    """Align measured component-centroid offsets with recorded native joins.

    Each displacement is x_component - x_DM in proper kpc; its plotted scalar
    is the Euclidean norm. Time is actual elapsed Gyr. A join at a saved epoch
    is an ancestry annotation, not a continuous merger time or a causal claim.
    """
    names=sorted({r['component'] for e in history['epochs'] for r in e['components']['dm_baryon_comparisons']})
    n=len(keys);time=np.asarray([e['elapsed_time_gyr'] for e in history['epochs']],float)
    if time.shape!=(n,) or not np.isfinite(time).all() or np.any(np.diff(time)<=0):raise ValueError('Increasing actual native centroid times required')
    joined={tuple(m['descendant']) for m in assembly['native_mergers']};event=np.asarray([tuple(k) in joined for k in keys],bool)
    arrays={'actual_time_gyr':time,'native_join_at_saved_epoch':event}
    for j,name in enumerate(names):
        value=np.full(n,np.nan);vector=np.full((n,3),np.nan);support=np.zeros(n,bool)
        for i,epoch in enumerate(history['epochs']):
            rows=[r for r in epoch['components']['dm_baryon_comparisons'] if r['component']==name]
            if len(rows)>1:raise ValueError('Ambiguous original native component centroid')
            if not rows:continue
            row=rows[0];x=np.asarray(row['dm_centroid_displacement_proper_kpc'],float);offset=row['offset_proper_kpc']
            if x.shape!=(3,) or not np.isfinite(x).all() or not np.isfinite(offset) or offset<0 or not np.isclose(np.linalg.norm(x),offset,rtol=1e-12,atol=1e-12):raise ValueError('Original measured centroid displacement/norm differs')
            value[i]=offset;vector[i]=x;support[i]=True
        prefix=f'component_{j:03d}_';arrays[prefix+'offset_proper_kpc']=value;arrays[prefix+'displacement_proper_kpc']=vector
        arrays[prefix+'measured']=support;arrays[prefix+'native_join_measured']=support&event
    if not names:
        arrays['unavailable_offset_proper_kpc']=np.full(n,np.nan);arrays['centroid_available']=np.zeros(n,bool)
    return names,arrays


def export_native_sidm_centroids(request,destination,*,memory_gib=2.):
    from core.research.display import export_scientific_suite
    history,assembly,keys,pins=centroid_inputs(request,memory_gib=memory_gib);names,arrays=centroid_event_arrays(history,assembly,keys)
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native SIDM merger-centroid figures')
    out.mkdir(parents=True);path=out/'centroid-events.json';epoch={'saved_native_epochs':[e['epoch'] for e in history['epochs']]}
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    export_numeric_audit(path,{'schema':'native-sidm-centroid-events-v1','epoch':epoch,'original_history':str(Path(request['history']['path']).resolve()),
        'component_order':names,'native_mergers':assembly['native_mergers'],'native_graph_keys':[list(k) for k in keys],
        'selection':'Original DM/baryon mass centroids in the same declared proper aperture; original minimum-population criteria',
        'event_definition':assembly.get('native_event_definition','Native non-main progenitor joins selected descendant at actual saved epoch; continuous coalescence time is unresolved'),
        'uncertainty':'Original finite-population/centering systematics remain unaccepted; no merger causality inferred from simultaneous centroid offset','science_acceptance':'NOT_ACCEPTED'},arrays,
        {k:{'units':'proper kpc' if k.endswith('_kpc') else 'Gyr' if k.endswith('_gyr') else 'dimensionless','quantity':k,'missing_policy':'Absent components and unavailable joins remain masked; raw saved offsets retained'} for k in arrays},
        algorithm=__file__,context={'epoch':epoch},source_hashes=pins)
    figures=[]
    for j,name in enumerate(names):
        prefix=f'component_{j:03d}_';panels=[]
        for support,title in [('measured','Original saved-time centroid offsets'),('native_join_measured','Offsets at recorded native progenitor joins only')]:
            panels.append({'panel_id':support,'title':title,'quality_status':'CONDITIONAL_NATIVE_MEASUREMENT','selection':name+' minus DM centroid in original aperture',
                'xlabel':'Actual saved time [Gyr]','ylabel':'Centroid offset [proper kpc]','x_units':'Gyr','y_units':'proper kpc',
                'series':[{'source':'centroids','kind':'scatter','x':'actual_time_gyr','y':prefix+'offset_proper_kpc','support':prefix+support,'label':name+' / DM','marker_size':10}]})
        figures.append({'figure_id':f'native-sidm-centroid-{j:03d}','family':'sidm','variant':'sidm_merger_centroid_offsets','title':'Native '+name+' / DM centroids and recorded ancestry joins',
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':panels})
    if not figures:
        # A DM-only run still has a complete executable route; the missing
        # baryon measurement is explicit rather than a fabricated zero offset.
        figures=[{'figure_id':'native-sidm-centroid-unresolved','family':'sidm','variant':'sidm_merger_centroid_offsets',
            'title':'No measured native baryonic component centroid','science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{
                'panel_id':'missing','title':'No supported DM/baryon centroid comparison','quality_status':'NO_NATIVE_BARYON_CENTROID','selection':'Original component census',
                'xlabel':'Actual saved time [Gyr]','ylabel':'Centroid offset [proper kpc]','x_units':'Gyr','y_units':'proper kpc',
                'series':[{'source':'centroids','kind':'scatter','x':'actual_time_gyr','y':'unavailable_offset_proper_kpc','label':'No measured baryonic centroid','support':'centroid_available'}]}]}]
    for source,digest in pins.items():
        if file_sha256(source)!=digest:raise ValueError('Native centroid source changed during export')
    return export_scientific_suite({'sources':{'centroids':str(path)},'figures':figures,'dpi':160,'maximum_panels_per_figure':2,'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)
