"""Recovered native target -> original SIDM membership/history producers.

No caller supplies a replacement host row or guessed elapsed time. Missing or
ambiguous recovery remains unresolved; genuine ingestion/accounting errors
still raise and stop the workflow with their original evidence.
"""
from pathlib import Path
import copy
import json
import math
import numpy as np
from .data import exact_uint64_ids
from .foundation import atomic_json,file_sha256,canonical_digest
from .foundation import merge_source_bindings as _merge


def recovered_sidm_inputs(parameters):
    """Join actual recovered hosts, native model settings and source identities.

    Each accepted host row belongs to its saved native snapshot. Matching
    retains the common IC particle namespace and physical background clock;
    unresolved recovery stays separate from measured histories.
    """
    from .analysis import load_cfg
    from .sidm_analysis import _stage_native_bindings
    memory=parameters['memory_gib']
    if type(memory) not in (int,float) or not math.isfinite(memory) or memory<=0:raise ValueError('Positive finite recovered SIDM allocation required')
    run=Path(parameters['native_run_directory']).resolve();recovery=Path(parameters['recovery_path']).resolve();template=Path(parameters['template_path']).resolve()
    if max(recovery.stat().st_size,template.stat().st_size)*24>memory*1024**3/4:raise MemoryError('Recovered SIDM metadata exceeds allocation')
    if file_sha256(template)!=parameters['template_sha256']:raise ValueError('Original recovered SIDM physical controls changed')
    settings=json.loads(template.read_text());required={'schema','snapshots','maximum_particles','support_radius_proper_kpc','species_components','history_controls'}
    if set(settings)!=required or settings['schema']!='native-recovered-sidm-controls-v1':raise ValueError('Complete original recovered SIDM physical control template required')
    snapshots=settings['snapshots']
    if not isinstance(snapshots,list) or len(snapshots)<2 or len(set(snapshots))!=len(snapshots) or any(type(x) is not int or x<0 for x in snapshots):raise ValueError('Exact saved recovered SIDM snapshot selectors required')
    record=json.loads(recovery.read_text())
    if record.get('schema')!='lagrangian-descendant-recovery-v1':raise ValueError('Original material recovery producer required')
    cfg=load_cfg(run)
    manifest=run/'manifest.json';raw=json.loads(manifest.read_text())
    background=raw['input_artifacts']['boltzmann_background'];bg=Path(background['path']).resolve()
    if file_sha256(bg)!=background['sha256']:raise ValueError('Native SIDM proper-clock background changed')
    linear=Path(cfg['spectrum']['file']).resolve();enabled=cfg.get('sidm',{}).get('enabled',False)
    if type(enabled) is not bool:raise ValueError('Actual native SIDM model enablement must remain boolean')
    namespace=parameters.get('particle_namespace');ic=None
    if parameters.get('initial_conditions') is not None:
        ic=Path(parameters['initial_conditions']).resolve();actual='native_music_ic:'+file_sha256(ic)
        if namespace is not None and namespace!=actual:raise ValueError('Declared SIDM namespace differs from the actual shared IC bytes')
        namespace=actual
    if not isinstance(namespace,str) or not namespace:raise ValueError('Source-bound native particle namespace required')
    target=parameters['matching_identity']
    if set(target)!={'namespace','object_id'} or not isinstance(target['namespace'],str) or not target['namespace'] or type(target['object_id']) is not int or not 0<=target['object_id']<2**64:
        raise ValueError('Exact original matched target identity required')
    matching={'phase_seed':cfg['simulation']['random_seed'],'linear_bundle_sha256':file_sha256(linear),
        'background_sha256':background['sha256'],'halo_match_namespace':target['namespace'],'halo_match_id':target['object_id'],
        'baryon_control':'original actual native gas/stars/BH/source configuration',
        'baryon_control_sha256':canonical_digest({'physics':cfg['simulation']['physics'],**{name:cfg.get(name,{}) for name in ('hydro','black_holes','stellar_evolution','chemistry','external_potential','baryonic_force','stellar_sources')}})}
    native=_stage_native_bindings({'request':{'memory_gib':memory,'matching':matching,'dark_matter_model':'elastic_sidm' if enabled else 'collisionless'},
        'native_run_directory':str(run),'native_stage_receipt':parameters['native_stage_receipt']})
    pins=dict(native)
    if ic is not None:
        if native.get(str(ic))!=file_sha256(ic):raise ValueError('SIDM identity namespace requires this exact owned native initial-condition generation')
        _merge(pins,{str(ic):file_sha256(ic)})
    _merge(pins,{str(p):file_sha256(p) for p in [recovery,template,run/'config.yaml',manifest,linear,bg]})
    if not record.get('source_hashes'):raise ValueError('Original recovered target has no native source provenance')
    for path,digest in record['source_hashes'].items():
        if file_sha256(path)!=digest:raise ValueError('Recovered target source changed')
    _merge(pins,record['source_hashes'])
    rows={}
    for row in record['records']:
        snapshot=row['snapshot']
        if snapshot in rows:raise ValueError('Duplicate original recovered target saved epoch')
        rows[snapshot]=row
        if snapshot not in snapshots:continue
        # The recovery may also bind the original parent and traced material.
        # Its measured selected snapshot/catalogue files must belong to this
        # actual completed evolution, never a different branch's generation.
        import re
        for path,digest in row.get('source_hashes',{}).items():
            filename=Path(path).name
            if filename.endswith('.hdf5') and re.match(r'(?:grouped_snapshot|snapshot|fof_subhalo_tab)_',filename):
                if native.get(str(Path(path).resolve()))!=digest:raise ValueError('Recovery does not bind this actual native SIDM/CDM branch')
    if set(rows)!=set(snapshots):raise ValueError('Recovered SIDM requires exactly the original requested saved epoch records')
    accepted=[];unresolved=[]
    for snapshot in snapshots:
        row=rows[snapshot]
        if row['status']!='MATCHED':
            unresolved.append({'snapshot':snapshot,'status':row['status'],'reason':'Original material/runner-up retention criteria did not produce an unambiguous target'});continue
        host=row.get('descendant',{}).get('host_row')
        if type(host) is not int or host<0:raise ValueError('Exact measured recovered FOF row required')
        accepted.append({'snapshot':snapshot,'host_row':host})
    settings={**settings,'native_particle_namespace':namespace}
    return settings,record,accepted,unresolved,matching,enabled,pins


def native_recovered_sidm_runner(parameters,context):
    from .sidm_analysis import publish_sidm_membership,publish_sidm_history,_load_product
    from .sidm_products import export_native_sidm_history
    from .sidm_products import export_native_scattering_rates
    from .sidm_campaign import native_branch_measurements
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Recovered SIDM history exceeds stage allocation')
    settings,recovery,accepted,unresolved,matching,enabled,pins=recovered_sidm_inputs(parameters)
    out=context.generation;memory=parameters['memory_gib'];run=parameters['native_run_directory'];receipt=parameters['native_stage_receipt']
    base={'native_run_directory':run,'native_stage_receipt':receipt,'request_base':str(out)}
    if len(accepted)<2:
        result={'schema':'native-recovered-sidm-result-v1','status':'UNRESOLVED_TARGET_HISTORY','execution_status':'COMPLETE',
            'available_native_target_observations':accepted,'unresolved_native_target_observations':unresolved,
            'reason':'Fewer than two actual target observations passed the unchanged native material recovery rules',
            'matching':matching,'source_hashes':pins,'science_acceptance':'NOT_ACCEPTED'}
        atomic_json(out/'sidm-history-result.json',result)
        return {'execution_status':'COMPLETE','measurements':{'sidm_target_history':result}}
    # Generate complete original request shapes, then bind each actual product.
    # This also prevents a controls file from replacing ID/model/clock bindings.
    branch={'schema':'native-sidm-branch-measurements-v1','observations':accepted,'maximum_particles':settings['maximum_particles'],
        'support_radius_proper_kpc':settings['support_radius_proper_kpc'],'species_components':settings['species_components'],
        'history_controls':settings['history_controls'],'satellite':None}
    templates=native_branch_measurements(branch,memory_gib=memory);epochs=[];members=[]
    for i,row in enumerate(accepted):
        request=copy.deepcopy(templates[i]['parameters']['request'])
        request.update(particle_namespace=settings['native_particle_namespace'],matching=matching)
        path=out/f'membership-{i:03d}.json';report=publish_sidm_membership({**base,'request':request},path)
        epochs.append({'snapshot':row['snapshot'],'elapsed_time_gyr':report['elapsed_time_gyr'],
            'selection':{'path':str(path),'sha256':file_sha256(path)}})
        members.append({'snapshot':row['snapshot'],'host_row':row['host_row'],'product':epochs[-1]['selection']})
    request=copy.deepcopy(templates[-1]['parameters']['request'])
    request.update(particle_namespace=settings['native_particle_namespace'],matching=matching,dark_matter_model='elastic_sidm' if enabled else 'collisionless',
        require_sidm_diagnostics=enabled,epochs=epochs)
    history=out/'sidm-history.json';report=publish_sidm_history({**base,'request':request},history)
    gallery=export_native_sidm_history(history,out/'history-figures',run_directory=run,memory_gib=memory)
    rates=export_native_scattering_rates(history,out/'scattering-figures',run_directory=run,memory_gib=memory) if enabled else None
    variants=sorted({f['variant'] for f in gallery['figures']}|({f['variant'] for f in rates['figures']} if rates else set()))
    result={'schema':'native-recovered-sidm-result-v1','status':'MEASURED_NATIVE_TARGET_HISTORY','execution_status':'COMPLETE',
        'matching':matching,'dark_matter_model':request['dark_matter_model'],'source_hashes':pins,'memberships':members,
        'history':{'path':str(history),'sha256':file_sha256(history)},'variants':variants,
        'unresolved_native_target_observations':unresolved,'science_acceptance':'NOT_ACCEPTED',
        'scope':'Original recovered target, complete native aperture census, actual proper times and original SIDM estimators; missing saved targets remain explicit; sparse observations do not establish unresolved collapse timing.'}
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Recovered SIDM source changed during producer execution')
    atomic_json(out/'sidm-history-result.json',result)
    return {'execution_status':'COMPLETE','measurements':{'sidm_target_history':result}}


def prepare_recovered_sidm_campaign(request,destination):
    """Append actual recovered-target SIDM producers to an owned paired recipe."""
    from .workflow import compile_requested_product_dag,default_registry
    required={'schema','history_campaign','controls','memory_gib'}
    if set(request)-{'matched_controls','satellite_controls','centroid_controls','lensing_controls','stream_controls'}!=required or request['schema']!='native-recovered-sidm-campaign-v1':raise ValueError('Complete recovered native SIDM campaign request required')
    memory=request['memory_gib']
    if type(memory) not in (int,float) or not math.isfinite(memory) or memory<=0:raise ValueError('Positive finite recovered SIDM campaign workspace required')
    out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous recovered native SIDM preparation')
    pins={}
    def bind(reference):
        if set(reference)!={'path','sha256'}:raise ValueError('Exact recovered SIDM asset path/hash required')
        path=Path(reference['path']).resolve()
        if file_sha256(path)!=reference['sha256']:raise ValueError('Recovered SIDM campaign source changed')
        _merge(pins,{str(path):reference['sha256']});return path
    manifest=json.loads(bind(request['history_campaign']).read_text())
    if manifest.get('schema')!='flagship-history-campaign-v1' or manifest.get('status')!='RECIPES_PREPARED':raise ValueError('Prepared original native target recovery/history campaign required')
    for path,checksum in manifest['source_hashes'].items():bind({'path':path,'sha256':checksum})
    workflow=json.loads(bind(manifest['workflow']).read_text());rows=manifest['cases'];by_id={s['stage_id']:s for s in workflow['stages']}
    if set(request['controls'])!={r['id'] for r in rows}:raise ValueError('SIDM measurement controls must cover exactly the actual configured native cases')
    if memory>workflow['resources']['memory_gib']:raise MemoryError('Recovered SIDM postprocessing exceeds paired native allocation')
    results=[];seen=set();controls_by_case={}
    for row in rows:
        identifier=row['id'];branch=row['branch']
        if branch not in {'cdm','sidm'} or (identifier,branch) in seen:raise ValueError('Distinct actual paired CDM/SIDM recovery recipes required')
        seen.add((identifier,branch));recovery=by_id[row['recovery_stage']];mapping=by_id[row['mapping_stage']]
        run=copy.deepcopy(recovery['parameters']['run_dir']);evolution=run['stage']
        if set(run)!={'stage','artifact'} or run['artifact']!='run' or evolution not in by_id:raise ValueError('Recovery must reference an owned actual native evolution')
        if identifier not in controls_by_case:
            settings_path=bind(request['controls'][identifier])
            controls_by_case[identifier]=(settings_path,json.loads(settings_path.read_text()))
        settings_path,settings=controls_by_case[identifier]
        if settings['snapshots']!=recovery['parameters']['snapshots']:raise ValueError('SIDM history must use exactly its original recovered native saved epoch selectors')
        # The original trace identifies the common parent target, independently
        # of each branch's recovered FOF row at every subsequent saved epoch.
        selected=Path(mapping['parameters']['selected_particles']['input']).resolve();region=selected.parent/'region.json'
        if not region.is_file():raise ValueError('Original traced material selection requires its source region contract')
        raw=json.loads(region.read_text());_merge(pins,{str(region):file_sha256(region)})
        if file_sha256(selected)!=raw['selected_particles_sha256']:raise ValueError('SIDM common parent material selection changed')
        _merge(pins,{str(selected):file_sha256(selected)})
        target={'namespace':'original-parent-catalogue:'+canonical_digest(raw['catalogue_binding']),
                'object_id':raw['catalogue_binding']['host_fof_row']}
        initial=copy.deepcopy(mapping['parameters']['zoom_ic']);name='sidm_history_'+branch+'_'+identifier
        params={'native_run_directory':run,'native_stage_receipt':{'stage':evolution,'artifact':'stage-receipt.json'},
            'initial_conditions':initial,'recovery_path':{'stage':row['recovery_stage'],'artifact':'recovery.json'},
            'template_path':str(settings_path),'template_sha256':file_sha256(settings_path),'matching_identity':target,'memory_gib':memory}
        stage={'stage_id':name,'kind':'native_recovered_sidm','parameters':params,'input_files':[request['controls'][identifier]],
            'outputs':{name:'sidm-history-result.json'}}
        workflow['stages'].append(stage);workflow['requested_products'].extend(stage['outputs'])
        results.append({'id':identifier,'branch':branch,'stage_id':name,'native_evolution':evolution,'matching_identity':target})
    if any((identifier,branch) not in seen for identifier in request['controls'] for branch in ('cdm','sidm')):raise ValueError('Every configured SIDM target needs both original native branch recoveries')
    satellite_records=[]
    if request.get('satellite_controls') is not None:
        controls=bind(request['satellite_controls']);satellite_controls(controls,file_sha256(controls),memory)
        for row in results:
            name='sidm_satellite_'+row['branch']+'_'+row['id']
            params={'native_run_directory':{'stage':row['native_evolution'],'artifact':'run'},
                'native_stage_receipt':{'stage':row['native_evolution'],'artifact':'stage-receipt.json'},
                'target_result':{'stage':row['stage_id'],'artifact':'sidm-history-result.json'},
                'template_path':str(controls),'template_sha256':file_sha256(controls),'memory_gib':memory}
            stage={'stage_id':name,'kind':'native_recovered_sidm_satellite','parameters':params,
                'input_files':[request['satellite_controls']],'outputs':{name:'sidm-satellite-result.json'}}
            workflow['stages'].append(stage);workflow['requested_products'].append(name)
            satellite_records.append({'id':row['id'],'branch':row['branch'],'stage_id':name})
    centroid_records=[]
    if request.get('centroid_controls') is not None:
        if set(request['centroid_controls'])!=set(request['controls']):raise ValueError('Native centroid controls must cover every actual configured case')
        for row in results:
            reference=request['centroid_controls'][row['id']];controls=bind(reference)
            settings=centroid_controls(controls,file_sha256(controls),memory)
            if settings['snapshots']!=controls_by_case[row['id']][1]['snapshots']:
                raise ValueError('Native centroid output selectors must match original recovered history')
            bind(settings['structure_template'])
            name='sidm_centroids_'+row['branch']+'_'+row['id']
            params={'native_run_directory':{'stage':row['native_evolution'],'artifact':'run'},
                'native_stage_receipt':{'stage':row['native_evolution'],'artifact':'stage-receipt.json'},
                'target_result':{'stage':row['stage_id'],'artifact':'sidm-history-result.json'},
                'template_path':str(controls),'template_sha256':file_sha256(controls),'memory_gib':memory}
            workflow['stages'].append({'stage_id':name,'kind':'native_recovered_sidm_centroids','parameters':params,
                'input_files':[reference,settings['structure_template']],'outputs':{name:'sidm-centroid-result.json'}})
            workflow['requested_products'].append(name);centroid_records.append({'id':row['id'],'branch':row['branch'],'stage_id':name})
    pair_records=[]
    if request.get('matched_controls') is not None:
        controls=bind(request['matched_controls'])
        for identifier in request['controls']:
            branches={row['branch']:row for row in results if row['id']==identifier}
            name='sidm_pair_'+identifier
            parameters={'template_path':str(controls),'template_sha256':file_sha256(controls),'memory_gib':memory}
            for branch in ('cdm','sidm'):
                source=branches[branch]
                parameters[branch+'_result']={'stage':source['stage_id'],'artifact':'sidm-history-result.json'}
                parameters[branch+'_run_directory']={'stage':source['native_evolution'],'artifact':'run'}
            workflow['stages'].append({'stage_id':name,'kind':'native_recovered_sidm_pair','parameters':parameters,
                'input_files':[request['matched_controls']],'outputs':{name:'sidm-pair-result.json'}})
            workflow['requested_products'].append(name);pair_records.append({'id':identifier,'stage_id':name})
    lensing_records=[]
    if request.get('lensing_controls') is not None:
        from .sidm_forward import lensing_controls
        if not pair_records:raise ValueError('Native lensing requires original matched history/IC phase producer stages')
        controls=bind(request['lensing_controls']);settings=lensing_controls(controls,file_sha256(controls),memory)
        for identifier in request['controls']:
            history_controls=controls_by_case[identifier][1]
            if (2*settings['half_width_proper_kpc']**2+settings['half_depth_proper_kpc']**2)**.5>history_controls['support_radius_proper_kpc']:
                raise ValueError('Native lens cylinder exceeds original complete membership aperture')
            branches={r['branch']:r for r in results if r['id']==identifier}
            pair=next(r for r in pair_records if r['id']==identifier);name='sidm_lensing_'+identifier
            params={'pair_result':{'stage':pair['stage_id'],'artifact':'sidm-pair-result.json'},
                'template_path':str(controls),'template_sha256':file_sha256(controls),'memory_gib':memory}
            for label,row in branches.items():
                params[label+'_result']={'stage':row['stage_id'],'artifact':'sidm-history-result.json'}
                params[label+'_run_directory']={'stage':row['native_evolution'],'artifact':'run'}
            workflow['stages'].append({'stage_id':name,'kind':'native_recovered_sidm_lensing','parameters':params,
                'input_files':[request['lensing_controls']],'outputs':{name:'sidm-lensing-result.json'}})
            workflow['requested_products'].append(name);lensing_records.append({'id':identifier,'stage_id':name})
    stream_records=[]
    if request.get('stream_controls') is not None:
        from .sidm_forward import stream_controls
        if not pair_records:raise ValueError('Native stellar response requires original matched target/phase producer stages')
        controls=bind(request['stream_controls']);stream_controls(controls,file_sha256(controls),memory)
        for identifier in request['controls']:
            branches={r['branch']:r for r in results if r['id']==identifier};pair=next(r for r in pair_records if r['id']==identifier)
            name='sidm_stream_response_'+identifier
            params={'pair_result':{'stage':pair['stage_id'],'artifact':'sidm-pair-result.json'},'template_path':str(controls),'template_sha256':file_sha256(controls),'memory_gib':memory}
            for label,row in branches.items():
                original=next(r for r in rows if r['id']==identifier and r['branch']==label)
                stage=original.get('stream_measurement_stage')
                if stage not in by_id or by_id[stage]['kind']!='native_recovered_stream_measurements':raise ValueError('Original source-bound recovered stellar measurement stage required for each native branch')
                params[label+'_stream_history']={'stage':stage,'artifact':'stream-history.json'}
                params[label+'_stream_receipt']={'stage':stage,'artifact':'stage-receipt.json'}
                params[label+'_run_directory']={'stage':row['native_evolution'],'artifact':'run'}
            workflow['stages'].append({'stage_id':name,'kind':'native_recovered_sidm_stream_response','parameters':params,
                'input_files':[request['stream_controls']],'outputs':{name:'sidm-stream-result.json'}})
            workflow['requested_products'].append(name);stream_records.append({'id':identifier,'stage_id':name})
    # The upstream history builder records its generated graph size here.
    # Extend that derived count along with the newly authored recovery stages.
    workflow['resources']['maximum_stage_count']=len(workflow['stages'])
    registry=default_registry()
    compile_requested_product_dag(workflow,registry,request_base=out)
    for row in workflow['stages']:
        _merge(pins,registry.source_hashes(row['kind']))
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    for path,checksum in pins.items():
        if file_sha256(path)!=checksum:raise ValueError('Recovered SIDM producer source changed during preparation')
    out.mkdir(parents=True);atomic_json(out/'request.json',request);atomic_json(out/'workflow.json',workflow)
    result={'schema':'native-recovered-sidm-campaign-prepared-v1','status':'RECIPES_PREPARED','cases':results,'paired_history_comparisons':pair_records,'satellite_histories':satellite_records,'centroid_histories':centroid_records,'paired_lensing_responses':lensing_records,'paired_stellar_responses':stream_records,
        'workflow':{'path':str(out/'workflow.json'),'sha256':file_sha256(out/'workflow.json')},'source_hashes':pins,
        'stage_count':len(workflow['stages']),'native_execution':'NOT_EXECUTED','science_acceptance':'NOT_ACCEPTED',
        'scope':'Actual recovered native target rows and clocks feed original membership/SIDM estimators, optional matched controls and exact bound satellite continuity; unavailable targets remain explicit. Native ancestry/forward/calibration connections remain separately prepared.'}
    atomic_json(out/'manifest.json',result);return result


def recovered_sidm_pair_inputs(parameters):
    from .sidm_products import sidm_history_inputs
    memory=parameters['memory_gib']
    if type(memory) not in (int,float) or not math.isfinite(memory) or memory<=0:raise ValueError('Positive native SIDM pair allocation required')
    template=Path(parameters['template_path']).resolve()
    if file_sha256(template)!=parameters['template_sha256']:raise ValueError('Original SIDM matched controls changed')
    controls=json.loads(template.read_text())
    required={'schema','maximum_histories','observables','absolute_floors','matching_controls'}
    if set(controls)!=required or controls['schema']!='native-recovered-sidm-pair-controls-v1':raise ValueError('Complete original native SIDM paired estimator controls required')
    pins={str(template):file_sha256(template)};results={}
    for label in ('cdm','sidm'):
        path=Path(parameters[label+'_result']).resolve()
        if path.stat().st_size*24>memory*1024**3/4:raise MemoryError('Native paired SIDM result metadata exceeds allocation')
        record=json.loads(path.read_text())
        if record.get('schema')!='native-recovered-sidm-result-v1' or record.get('execution_status')!='COMPLETE':raise ValueError('Completed original recovered SIDM result required')
        _merge(pins,{str(path):file_sha256(path)});_merge(pins,record['source_hashes']);results[label]=record
        if record['status']=='MEASURED_NATIVE_TARGET_HISTORY':
            binding=record['history'];product=Path(binding['path']).resolve()
            if not product.is_relative_to(path.parent) or file_sha256(product)!=binding['sha256']:raise ValueError('Recovered native SIDM history changed or escapes its original generation')
            report,_,sources=sidm_history_inputs(product,run_directory=parameters[label+'_run_directory'],memory_gib=memory);_merge(pins,sources)
            expected='collisionless' if label=='cdm' else 'elastic_sidm'
            if report['dark_matter_model']!=expected:raise ValueError('Recovered matched SIDM pair relabels the actual evolved model')
        elif record['status']!='UNRESOLVED_TARGET_HISTORY':raise ValueError('Unknown original recovered SIDM status')
    for path,digest in pins.items():
        if file_sha256(path)!=digest:raise ValueError('Recovered native SIDM pair source changed')
    return controls,results,pins


def native_recovered_sidm_pair_runner(parameters,context):
    from .sidm_analysis import publish_sidm_matched
    from .sidm_products import export_native_sidm_matched
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Recovered native pair exceeds stage allocation')
    controls,results,pins=recovered_sidm_pair_inputs(parameters);out=context.generation
    missing=[label for label,r in results.items() if r['status']!='MEASURED_NATIVE_TARGET_HISTORY']
    if missing:
        result={'schema':'native-recovered-sidm-pair-result-v1','status':'UNRESOLVED_NATIVE_MATCH','execution_status':'COMPLETE',
            'unresolved_branches':missing,'reason':'Original material recovery lacks enough supported native target observations; no paired measurement inferred',
            'source_hashes':pins,'science_acceptance':'NOT_ACCEPTED'}
        atomic_json(out/'sidm-pair-result.json',result)
        return {'execution_status':'COMPLETE','measurements':{'sidm_matched_history':result}}
    request={'schema':'sidm-matched-science-request-v1','memory_gib':parameters['memory_gib'],'baseline_label':'cdm',
        'histories':[{'label':label,'product':results[label]['history']} for label in ('cdm','sidm')],
        **{k:copy.deepcopy(v) for k,v in controls.items() if k!='schema'}}
    path=out/'matched-history.json';publish_sidm_matched({'request':request,'request_base':str(out)},path)
    # Actual IC phase measurement and exact-source native replay remain
    # mandatory in the original exporter; a failed phase comparison raises.
    figures=export_native_sidm_matched({'schema':'native-sidm-matched-figure-request-v1',
        'matched_product':{'path':str(path),'sha256':file_sha256(path)},'history_base_directory':str(out),
        'native_runs':{label:parameters[label+'_run_directory'] for label in ('cdm','sidm')}},out/'figures',memory_gib=parameters['memory_gib'])
    result={'schema':'native-recovered-sidm-pair-result-v1','status':'MEASURED_NATIVE_MATCH','execution_status':'COMPLETE',
        'matched_history':{'path':str(path),'sha256':file_sha256(path)},'figure_count':len(figures['figures']),
        'variants':sorted({f['variant'] for f in figures['figures']}),'source_hashes':pins,'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(out/'sidm-pair-result.json',result)
    return {'execution_status':'COMPLETE','measurements':{'sidm_matched_history':result}}


# Recovered native centroid ancestry and fixed bound satellite cohorts

def centroid_controls(path, digest, memory_gib):
    from .halo_campaign import validate_structure_template
    path = Path(path).resolve()
    if file_sha256(path) != digest:
        raise ValueError('Original centroid/ancestry controls changed')
    if path.stat().st_size * 24 > memory_gib * 1024**3 / 4:
        raise MemoryError('Native centroid controls exceed allocation')
    settings = json.loads(path.read_text())
    keys = {'schema', 'snapshots', 'maximum_nodes', 'maximum_edges', 'maximum_subhalos', 'maximum_particles',
            'structure_template', 'branch_controls'}
    if set(settings) != keys or settings['schema'] != 'native-recovered-sidm-centroid-controls-v1':
        raise ValueError('Complete original native centroid/ancestry controls required')
    for name in ('maximum_nodes', 'maximum_edges', 'maximum_subhalos', 'maximum_particles'):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError('Exact positive native centroid census limits required')
    snapshots = settings['snapshots']
    if (not isinstance(snapshots, list) or len(snapshots) < 2 or any(type(s) is not int or s < 0 for s in snapshots)
            or snapshots != sorted(set(snapshots)) or any(b != a + 1 for a, b in zip(snapshots, snapshots[1:]))):
        raise ValueError('Native centroid ancestry requires every consecutive saved output selector')
    branch = settings['branch_controls']
    if set(branch) != {'maximum_branch_epochs', 'maximum_interval_myr', 'main_branch_transfer_margin_msun'}:
        raise ValueError('Original branch support/margin/length criteria required; clocks derive from native outputs')
    if type(branch['maximum_branch_epochs']) is not int or branch['maximum_branch_epochs'] < 2:
        raise ValueError('Exact native branch epoch ceiling required')
    for name in ('maximum_interval_myr', 'main_branch_transfer_margin_msun'):
        if type(branch[name]) not in (int, float) or not math.isfinite(branch[name]) or branch[name] < 0:
            raise ValueError('Finite nonnegative original branch physical criteria required')
    if branch['maximum_interval_myr'] <= 0:
        raise ValueError('Positive original native branch cadence limit required')
    reference = settings['structure_template']
    if set(reference) != {'path', 'sha256'}:
        raise ValueError('Exact original halo structure template binding required')
    structure = Path(reference['path']).resolve()
    if file_sha256(structure) != reference['sha256']:
        raise ValueError('Original centroid halo structure controls changed')
    if structure.stat().st_size * 24 > memory_gib * 1024**3 / 4:
        raise MemoryError('Original centroid structure controls exceed allocation')
    validate_structure_template(json.loads(structure.read_text()), memory_gib=memory_gib)
    return settings


def recovered_centroid_inputs(parameters):
    from .sidm_products import sidm_history_inputs
    memory = parameters['memory_gib']
    if type(memory) not in (int, float) or not math.isfinite(memory) or memory <= 0:
        raise ValueError('Positive native centroid memory allocation required')
    path = Path(parameters['target_result']).resolve(); template = Path(parameters['template_path']).resolve()
    settings = centroid_controls(template, parameters['template_sha256'], memory)
    if path.stat().st_size * 24 > memory * 1024**3 / 4:
        raise MemoryError('Native centroid target metadata exceeds allocation')
    target = json.loads(path.read_text())
    if target.get('schema') != 'native-recovered-sidm-result-v1' or target.get('execution_status') != 'COMPLETE':
        raise ValueError('Original completed recovered SIDM target required')
    pins = {str(path): file_sha256(path), str(template): file_sha256(template)}; _merge(pins, target['source_hashes'])
    structure = settings['structure_template']; _merge(pins, {structure['path']: structure['sha256']})
    history = None
    if target['status'] == 'MEASURED_NATIVE_TARGET_HISTORY':
        reference = target['history']; product = Path(reference['path']).resolve()
        if not product.is_relative_to(path.parent) or file_sha256(product) != reference['sha256']:
            raise ValueError('Recovered centroid history changed or escapes its producer generation')
        history, _, sources = sidm_history_inputs(product, run_directory=parameters['native_run_directory'], memory_gib=memory)
        if (Path(history['native_stage_receipt']).resolve() != Path(parameters['native_stage_receipt']).resolve()
                or history['matching'] != target['matching'] or history['dark_matter_model'] != target['dark_matter_model']):
            raise ValueError('Recovered centroid target changes its original evolution/model/matching')
        if [r['snapshot'] for r in history['epochs']] != [r['snapshot'] for r in target['memberships']]:
            raise ValueError('Recovered centroid memberships change the original saved history selections')
        if any(r['snapshot'] not in settings['snapshots'] for r in history['epochs']):
            raise ValueError('Native centroid controls omit an original saved target epoch')
        _merge(pins, sources)
    elif target['status'] != 'UNRESOLVED_TARGET_HISTORY':
        raise ValueError('Unknown original recovered target status')
    for p, h in pins.items():
        if file_sha256(p) != h:
            raise ValueError('Native recovered centroid source changed')
    return settings, target, history, pins


def _native_main_keys(graph, target, maximum):
    """Collect at most maximum native main-link epochs, rejecting cycles.

    Exactly maximum epochs are permitted. Recorded main-progenitor links own
    this ancestry; material-overlap matches do not substitute for a missing link.
    """
    incoming = {}
    for edge in graph['edges']:
        if edge['kind'] == 'descendant':
            incoming.setdefault(tuple(edge['target']), []).append(edge)
    keys = set(); key = tuple(target)
    while True:
        if key in keys:raise ValueError('Cycle in recorded native central/FOF ancestry')
        keys.add(key)
        if len(keys) > maximum:raise MemoryError('Native main branch exceeds original epoch ceiling')
        parents = [e for e in incoming.get(key, []) if e['evidence'].get('main_progenitor') is True]
        if len(parents) > 1:raise ValueError('Multiple recorded native main progenitors')
        if not parents:break
        key = tuple(parents[0]['source'])
    return keys


def native_recovered_centroid_runner(parameters, context):
    from .analysis import load_cfg
    from .sidm_analysis import _stage_native_bindings
    from .native_gadget_ancestry import build_native_fof_ancestry
    from .assembly import reconstruct_quality_masked_main_branch
    from .halos import publish_halo_structure
    from .sidm_products import export_native_sidm_centroids
    from core.stream_common import publish_stream_product
    from core.stream_common import stream_array_units
    from . import source_path
    memory = parameters['memory_gib']
    if memory > context.resources['memory_gib']:raise MemoryError('Native centroid stage exceeds allocation')
    settings, target, history, pins = recovered_centroid_inputs(parameters)
    out = context.generation; observations = []; graph = None; figures = None; missing = []
    result = {'schema': 'native-recovered-sidm-centroid-result-v1', 'execution_status': 'COMPLETE',
        'science_acceptance': 'NOT_ACCEPTED', 'source_hashes': pins,
        'event_definition': 'FOF host membership joins through GADGET recorded previous-central descendant links; saved join epochs, not continuous subhalo/galaxy coalescence times',
        'unresolved_policy': 'Missing native main links or a different main branch remain explicit; material recovery is not substituted for native ancestry'}
    if history is None:
        result.update(status='UNRESOLVED_NATIVE_TARGET_HISTORY', variants=[])
    else:
        run = parameters['native_run_directory']; receipt = parameters['native_stage_receipt']; cfg = load_cfg(run)
        if cfg.get('halos', {}).get('native_mergertree') is not True:
            raise ValueError('Native centroid branch must actually enable GADGET MERGERTREE output')
        native = _stage_native_bindings({'request': {'memory_gib': memory}, 'native_run_directory': run, 'native_stage_receipt': receipt})
        graph, missing = build_native_fof_ancestry(run, settings['snapshots'], out / 'native-ancestry', native_sources=native,
            **{k: settings[k] for k in ('maximum_nodes', 'maximum_edges', 'maximum_subhalos', 'maximum_particles')},
            minimum_host_particles=cfg['halos']['min_particles'], memory_gib=memory)
        namespace = str(Path(run).resolve()) + ':fof:catalogue-row'
        keys = [[namespace, row['host_row'], row['snapshot']] for row in target['memberships']]
        branch = copy.deepcopy(settings['branch_controls']); coverage = _native_main_keys(graph, keys[-1], branch['maximum_branch_epochs'])
        if len(coverage) < 2 or any(tuple(key) not in coverage for key in keys):
            result.update(status='UNRESOLVED_NATIVE_ANCESTRY_MATCH', variants=[],
                original_recovered_keys=keys, recorded_native_main_keys=[list(k) for k in sorted(coverage)])
        else:
            branch.update(epoch_time_myr={str(k): v['cosmic_time_gyr'] * 1000 for k, v in graph['epochs'].items()},
                epoch_scale_factor={str(k): v['actual_scale_factor'] for k, v in graph['epochs'].items()})
            assembly, arrays = reconstruct_quality_masked_main_branch(graph, keys[-1], branch)
            assembly['native_event_definition']=result['event_definition']
            assembly['request'] = {'stage_id': 'native-centroid-main-branch', 'kind': 'assembly_history',
                'parameters': {'target': keys[-1], 'branch': branch}}
            assembly['source_hashes'] = dict(graph['sources'])
            _merge(assembly['source_hashes'], {str(source_path('assembly')): file_sha256(source_path('assembly'))})
            product = out / 'assembly.json'
            publish_stream_product(product, assembly, arrays, stream_array_units(arrays), sources=assembly['source_hashes'], namespace=namespace,
                epoch={'saved_native_epochs': settings['snapshots']})
            template = json.loads(Path(settings['structure_template']['path']).read_text())
            for row, key in zip(target['memberships'], keys):
                # Reuse the full original structure producer and its source,
                # clock, field-calibration, particle and support gates.
                path = out / f"structure-{row['snapshot']:03d}.json"
                publish_halo_structure({**copy.deepcopy(template), 'run_directory': run,
                    'snapshot': row['snapshot'], 'host_row': row['host_row']}, path)
                observations.append({'structure': {'path': str(path), 'sha256': file_sha256(path)}, 'graph_node': key})
            request = {'schema': 'native-sidm-centroid-events-v1', 'history': target['history'], 'run_directory': run,
                'producer_base_directory': str(out), 'assembly': {'path': str(product), 'sha256': file_sha256(product)},
                'identity_graph_directory': str(out / 'native-ancestry'), 'observations': observations}
            figures = export_native_sidm_centroids(request, out / 'centroid-figures', memory_gib=memory)
            result.update(status='MEASURED_NATIVE_CENTROIDS_AND_FOF_JOINS', variants=sorted({f['variant'] for f in figures['figures']}),
                assembly=request['assembly'], observations=observations)
    result.update(native_unresolved_links=missing, native_graph_fingerprint=None if graph is None else graph['fingerprint'])
    for p, h in pins.items():
        if file_sha256(p) != h:raise ValueError('Native recovered centroid inputs changed during execution')
    atomic_json(out / 'sidm-centroid-result.json', result)
    return {'execution_status': 'COMPLETE', 'measurements': {'sidm_merger_centroid_offsets': result}}


def satellite_controls(path, digest, memory_gib):
    path = Path(path).resolve()
    if path.stat().st_size * 24 > memory_gib * 1024**3 / 4:
        raise MemoryError('Native satellite controls exceed metadata allocation')
    if file_sha256(path) != digest:
        raise ValueError('Original satellite physical controls changed')
    controls = json.loads(path.read_text())
    required = {'schema', 'maximum_subhalos', 'maximum_particles', 'binding_controls', 'survival_controls', 'matching_controls'}
    if set(controls) != required or controls['schema'] != 'native-recovered-sidm-satellite-controls-v1':
        raise ValueError('Complete native satellite controls required')
    for key in ('maximum_subhalos', 'maximum_particles'):
        if type(controls[key]) is not int or controls[key] < 1:
            raise ValueError('Positive exact satellite census ceiling required')
    rules = controls['matching_controls']
    if set(rules) != {'minimum_material_fraction', 'maximum_runner_up_fraction', 'minimum_shared_particles'}:
        raise ValueError('Explicit original material retention/count/runner-up rules required')
    for key in ('minimum_material_fraction', 'maximum_runner_up_fraction'):
        if type(rules[key]) not in (int, float) or not math.isfinite(rules[key]):
            raise ValueError('Finite satellite material continuity criteria required')
    if not 0 <= rules['maximum_runner_up_fraction'] < rules['minimum_material_fraction'] <= 1:
        raise ValueError('Invalid satellite retention/runner-up criteria')
    if type(rules['minimum_shared_particles']) is not int or rules['minimum_shared_particles'] < 1:
        raise ValueError('Exact satellite shared particle count required')
    binding = controls['binding_controls']
    required = {'maximum_bound_particles', 'maximum_pair_evaluations', 'potential_chunk_rows',
                'potential_softening_proper_kpc', 'potential_gauge', 'species'}
    if (set(binding) != required or binding['species'] != [1]
            or not isinstance(binding['species'], list) or type(binding['species'][0]) is not int):
        raise ValueError('Complete original fine-DM satellite binding controls required')
    from .sidm_analysis import validate_sidm_satellite_request
    validate_sidm_satellite_request({**binding, 'schema': 'sidm-native-satellite-request-v1', 'snapshot': 0,
        'subhalo_row': 0, 'maximum_particles': controls['maximum_particles'], 'memory_gib': memory_gib,
        'particle_namespace': 'validation:only', 'satellite_identity': {'validation': True}, 'require_sidm_diagnostics': False})
    survival = controls['survival_controls']
    if set(survival) != {'minimum_bound_particles', 'mass_absolute_floor_msun', 'potential_gauge'}:
        raise ValueError('Complete original satellite survival criteria required')
    if (type(survival['minimum_bound_particles']) is not int or survival['minimum_bound_particles'] < 1
            or type(survival['mass_absolute_floor_msun']) not in (int, float)
            or not math.isfinite(survival['mass_absolute_floor_msun']) or survival['mass_absolute_floor_msun'] <= 0
            or survival['potential_gauge'] != binding['potential_gauge']):
        raise ValueError('Invalid native satellite survival support/gauge')
    return controls


def satellite_inputs(parameters):
    from .sidm_analysis import _load_product, _stage_native_bindings
    memory = parameters['memory_gib']
    if type(memory) not in (int, float) or not math.isfinite(memory) or memory <= 0:
        raise ValueError('Positive finite native satellite allocation required')
    controls = satellite_controls(parameters['template_path'], parameters['template_sha256'], memory)
    path = Path(parameters['target_result']).resolve()
    if path.stat().st_size * 24 > memory * 1024**3 / 4:
        raise MemoryError('Native satellite target metadata exceeds allocation')
    record = json.loads(path.read_text())
    if record.get('schema') != 'native-recovered-sidm-result-v1' or record.get('execution_status') != 'COMPLETE':
        raise ValueError('Original completed recovered native target required')
    if record['status'] not in {'MEASURED_NATIVE_TARGET_HISTORY', 'UNRESOLVED_TARGET_HISTORY'}:
        raise ValueError('Unknown original recovered native target status')
    pins = {str(path): file_sha256(path), str(Path(parameters['template_path']).resolve()): parameters['template_sha256']}
    _merge(pins, record['source_hashes'])
    if record['status'] == 'MEASURED_NATIVE_TARGET_HISTORY':
        native = _stage_native_bindings({'native_run_directory': parameters['native_run_directory'],
            'native_stage_receipt': parameters['native_stage_receipt'], 'request': {'memory_gib': memory,
                'matching': record['matching'], 'dark_matter_model': record['dark_matter_model']}})
        if any(pins.get(p) != h for p, h in native.items()):
            raise ValueError('Satellite target omits this actual native evolution generation')
        for row in record['memberships']:
            ref = row['product']; member = Path(ref['path']).resolve()
            if not member.is_relative_to(path.parent) or file_sha256(member) != ref['sha256']:
                raise ValueError('Native satellite host membership changed or escapes its generation')
            report, _, sources = _load_product(ref, path.parent, memory * 1024**3 / 2)
            if (report.get('schema') != 'native-halo-membership-measurement-v1'
                    or report['request']['snapshot'] != row['snapshot'] or report['request']['host_row'] != row['host_row']):
                raise ValueError('Satellite host selection differs from original recovered native membership')
            _merge(pins, sources)
    for p, h in pins.items():
        if file_sha256(p) != h:
            raise ValueError('Native satellite input source changed')
    return controls, record, pins


def match_fixed_satellite_cohort(anchor_ids, fields, snapshots, *, maximum_particles, memory_gib):
    """Vote exact original bound IDs over all spans, including centrals.

    Central/other-host votes contribute to ambiguity. They cannot be silently
    discarded in order to promote a weaker satellite match.
    """
    import h5py
    from .data import snapshot_id_shards
    ids = np.sort(exact_uint64_ids(anchor_ids))
    if not len(ids) or len(ids) * 128 + len(fields['SubhaloGroupNr']) * 512 > memory_gib * 1024**3 / 2:
        raise MemoryError('Native satellite exact cohort join exceeds workspace')
    shards, total = snapshot_id_shards(snapshots, 1)
    if total > maximum_particles:
        raise MemoryError('Native satellite continuity exceeds particle ceiling')
    starts = fields['SubhaloOffsetType'][:, 1]; lengths = fields['SubhaloLenType'][:, 1]
    active = np.flatnonzero(lengths > 0); active = active[np.argsort(starts[active])]
    votes = np.zeros(len(starts), dtype=np.int64); seen = np.zeros(len(ids), dtype=bool)
    chunk = max(1, min(100000, int(memory_gib * 1024**3 // 4096)))
    for shard in shards:
        if not shard['count']:
            continue
        with h5py.File(shard['path']) as f:
            dataset = f['PartType1/ParticleIDs']
            for first in range(0, shard['count'], chunk):
                values = exact_uint64_ids(dataset[first:first + chunk])
                indices = np.searchsorted(ids, values); valid = indices < len(ids)
                valid[valid] = ids[indices[valid]] == values[valid]
                selected = indices[valid]
                if np.any(seen[selected]) or len(np.unique(selected)) != len(selected):
                    raise ValueError('Original satellite tracer appears more than once in the native snapshot')
                seen[selected] = True
                positions = shard['start'] + first + np.flatnonzero(valid)
                location = np.searchsorted(starts[active], positions, side='right') - 1
                grouped = location >= 0
                if len(active):
                    grouped[grouped] = positions[grouped] < (starts[active[location[grouped]]] + lengths[active[location[grouped]]])
                    np.add.at(votes, active[location[grouped]], 1)
    ranked = np.argsort(-votes, kind='stable')
    winner = int(ranked[0]) if len(ranked) and votes[ranked[0]] > 0 else None
    return {'subhalo_row': winner, 'shared_particles': int(votes[winner]) if winner is not None else 0,
            'runner_up_shared_particles': int(votes[ranked[1]]) if len(ranked) > 1 else 0,
            'anchor_count': len(ids), 'anchor_particles_present': int(seen.sum())}


def accepted_satellite_match(match, fields, host_row, rules):
    row = match['subhalo_row']; count = match['anchor_count']
    if row is None:
        return False
    return (match['shared_particles'] >= rules['minimum_shared_particles']
        and match['shared_particles'] / count >= rules['minimum_material_fraction']
        and match['runner_up_shared_particles'] / count <= rules['maximum_runner_up_fraction']
        and match['shared_particles'] > match['runner_up_shared_particles']
        and int(fields['SubhaloRankInGr'][row]) > 0
        and int(fields['SubhaloGroupNr'][row]) == host_row
        and int(fields['SubhaloLenType'][row, 2]) == 0)


def native_recovered_satellite_runner(parameters, context):
    from .sidm_analysis import publish_sidm_satellite, publish_sidm_derived, _stage_native_bindings
    from .halos import _bound_particle_slice, read_native_subhalo_spans
    from .sidm_products import export_native_sidm_extended
    memory = parameters['memory_gib']
    if memory > context.resources['memory_gib']:
        raise MemoryError('Native satellite producer exceeds stage allocation')
    controls, target, pins = satellite_inputs(parameters)
    out = context.generation; observations = []; products = []; figures = []; identity = None; anchor = None; initial = None
    result = {'schema': 'native-recovered-sidm-satellite-result-v1', 'execution_status': 'COMPLETE',
        'science_acceptance': 'NOT_ACCEPTED', 'source_hashes': pins, 'controls': controls,
        'identity_scope': 'Same-run fixed initial bound fine-DM cohort continuity; not a native merger-tree identity or collision-causality claim',
        'unobserved_policy': 'Missing/ambiguous/non-satellite/other-host matches remain unknown, never zero retention or evaporation'}
    if target['status'] == 'MEASURED_NATIVE_TARGET_HISTORY':
        run = parameters['native_run_directory']; receipt = parameters['native_stage_receipt']; enabled = target['dark_matter_model'] == 'elastic_sidm'
        native = _stage_native_bindings({'native_run_directory': run, 'native_stage_receipt': receipt, 'request': {'memory_gib': memory}})
        rules = controls['matching_controls']; minimum = max(rules['minimum_shared_particles'], controls['survival_controls']['minimum_bound_particles'])
        for i, epoch in enumerate(target['memberships']):
            snapshot = epoch['snapshot']; fields, snapshots, _ = read_native_subhalo_spans(run, snapshot, native_sources=native,
                maximum_subhalos=controls['maximum_subhalos'], maximum_particles=controls['maximum_particles'], memory_gib=memory)
            if anchor is None:
                candidates = np.flatnonzero((fields['SubhaloGroupNr'] == epoch['host_row']) & (fields['SubhaloRankInGr'] > 0)
                    & (fields['SubhaloLenType'][:, 1] >= minimum) & (fields['SubhaloLenType'][:, 2] == 0))
                if not len(candidates):
                    observations.append({'snapshot': snapshot, 'status': 'UNRESOLVED_NO_SUPPORTED_NONCENTRAL_ANCHOR'}); continue
                # Declared deterministic choice: largest clean bound fine-DM
                # count, then original catalogue row for an exact count tie.
                row = min(map(int, candidates), key=lambda k: (-int(fields['SubhaloLenType'][k, 1]), k))
                count = int(fields['SubhaloLenType'][row, 1])
                if count > controls['binding_controls']['maximum_bound_particles'] or count * 128 > memory * 1024**3 / 4:
                    raise MemoryError('Selected original satellite exceeds bound-cohort allocation')
                anchor = exact_uint64_ids(_bound_particle_slice(snapshots, 1, int(fields['SubhaloOffsetType'][row, 1]), count, ('ParticleIDs',))['ParticleIDs'])
                if len(anchor) != count:
                    raise ValueError('Selected native satellite bound cohort is incomplete')
                identity = {'namespace': 'fixed-native-bound-fine-dm:' + canonical_digest({'native_run': str(Path(run).resolve()),
                    'snapshot': snapshot, 'subhalo_row': row, 'sorted_exact_ids': [int(k) for k in np.sort(anchor)]}), 'object_id': row}
                match = {'subhalo_row': row, 'shared_particles': count, 'runner_up_shared_particles': 0, 'anchor_count': count, 'anchor_particles_present': count}
            else:
                match = match_fixed_satellite_cohort(anchor, fields, snapshots, maximum_particles=controls['maximum_particles'], memory_gib=memory)
                if not accepted_satellite_match(match, fields, epoch['host_row'], rules):
                    observations.append({'snapshot': snapshot, 'status': 'UNRESOLVED_NATIVE_SATELLITE_IDENTITY', 'matching': match}); continue
                row = match['subhalo_row']
            request = {**copy.deepcopy(controls['binding_controls']), 'schema': 'sidm-native-satellite-request-v1', 'snapshot': snapshot,
                'subhalo_row': row, 'maximum_particles': controls['maximum_particles'], 'memory_gib': memory,
                'particle_namespace': None,
                'satellite_identity': identity, 'require_sidm_diagnostics': enabled, 'cohort_policy': 'initial_bound_only',
                'original_membership': epoch['product'] if initial is None else initial}
            # Read identity from the original physical population, never from a
            # case label or native row that can change between saved epochs.
            from .sidm_analysis import _load_product
            host, _, _ = _load_product(epoch['product'], Path(parameters['target_result']).parent, memory * 1024**3 / 2)
            request['particle_namespace'] = host['request']['particle_namespace']
            path = out / f'satellite-{i:03d}.json'
            report = publish_sidm_satellite({'request': request, 'request_base': str(out), 'native_run_directory': run, 'native_stage_receipt': receipt}, path)
            ref = {'path': str(path), 'sha256': file_sha256(path)}
            observations.append({'snapshot': snapshot, 'status': 'MEASURED_NATIVE_BOUND_POPULATION', 'matching': match,
                'elapsed_time_gyr': report['elapsed_time_gyr'], 'product': ref})
            if initial is None:
                initial = ref; continue
            derived = out / f'survival-{i:03d}.json'
            publish_sidm_derived({'request': {'schema': 'sidm-derived-science-request-v1', 'product': 'satellite', 'memory_gib': memory,
                'inputs': [initial, ref], 'settings': controls['survival_controls'], 'matched_experiment': target['matching']}, 'request_base': str(out)}, derived)
            export = {'schema': 'native-sidm-extended-figures-v1', 'operation': 'satellite',
                'product': {'path': str(derived), 'sha256': file_sha256(derived)}, 'producer_base_directory': str(out),
                'native_runs': [{'run_directory': run, 'stage_receipt': {'path': receipt, 'sha256': file_sha256(receipt)}}], 'phase_baseline_run_directory': None}
            gallery = export_native_sidm_extended(export, out / f'figures-{i:03d}', memory_gib=memory)
            figures.extend(gallery['figures']); products.append(export['product'])
    result.update(status='MEASURED_NATIVE_SATELLITE_SURVIVAL' if products else 'UNRESOLVED_NATIVE_SATELLITE_HISTORY',
        observations=observations, derived_products=products, variants=sorted({f['variant'] for f in figures}), satellite_identity=identity,
        unresolved_native_target_observations=target.get('unresolved_native_target_observations', []))
    for p, h in pins.items():
        if file_sha256(p) != h:
            raise ValueError('Native satellite source changed during execution')
    atomic_json(out / 'sidm-satellite-result.json', result)
    return {'execution_status': 'COMPLETE', 'measurements': {'sidm_satellite_survival': result}}
