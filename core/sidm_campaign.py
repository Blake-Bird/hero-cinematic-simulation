"""SIDM campaign preparation, native measurement requests and resource intake.

Campaigns use caller-authored settings, common CDM/SIDM initial conditions and
explicit physical controls. Measurement templates reference actual native
outputs; saved labels never substitute for physical elapsed time. Resource
summaries describe the supplied completed intervals, without extrapolating a
whole-run cost. Preparing a campaign does not execute it.
"""
from pathlib import Path
import copy
import json
import math
from .foundation import atomic_json,file_sha256,canonical_digest
from .foundation import merge_source_bindings as _merge

VARIANTS=['sidm_core_evolution','collapse_phase','scattering_rates','sidm_cdm_controls',
 'sidm_density_dispersion_anisotropy','sidm_shape_baryon_misalignment','sidm_heat_flux','sidm_kernel_support',
 'sidm_satellite_survival','sidm_merger_centroid_offsets','sidm_stream_profile_response','sidm_lensing_h0_response','sidm_baryonic_degeneracy']
CAPABILITIES={'history':set(VARIANTS[:2]+VARIANTS[4:8]),'scattering':{'scattering_rates'},'matched':{'sidm_cdm_controls'},
 'centroids':{'sidm_merger_centroid_offsets'},'satellite':{'sidm_satellite_survival'},'degeneracy':{'sidm_baryonic_degeneracy'},
 'observations':{'sidm_stream_profile_response','sidm_lensing_h0_response'}}


def _positive(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive finite native SIDM allocation required')


def _no_placeholders(value):
    if isinstance(value,dict):
        for key,item in value.items():_no_placeholders(key);_no_placeholders(item)
    elif isinstance(value,list):
        for item in value:_no_placeholders(item)
    elif isinstance(value,str) and any(token in value for token in ('FILL_','REPLACE_')):
        raise ValueError('Executable SIDM recipes cannot contain unresolved placeholder values')


def prepare_native_sidm_campaign(request,destination,*,request_base):
    """Prepare caller-defined CDM/SIDM branches and their measurement graph.

    Both branches retain the same IC shards, linear model and non-SIDM
    controls. Actual products and saved physical times resolve from completed
    producer stages. The returned manifest describes preparation, not evolution.
    """
    from .workflow import path_at,safe_id,default_registry,compile_requested_product_dag
    from core.research.capabilities import validate_native_answers
    from .sidm_physics import sidm_manifest_record
    from .evolution import gadget_config_text
    required={'schema','base_answers','initial_conditions','phase_asset','sidm','background','matching','resources',
        'wall_seconds_per_native_run','snapshots','measurement_design','auxiliary_workflows'}
    if set(request)!=required or request['schema']!='native-sidm-campaign-v1':raise ValueError('Complete native matched SIDM/all13 campaign request required')
    _no_placeholders(request);out=Path(destination).resolve();base=Path(request_base).resolve()
    if out.exists():raise FileExistsError('Keep previous native SIDM preparation')
    resources=copy.deepcopy(request['resources'])
    for key in ('memory_gib','disk_gib'):_positive(resources[key])
    for key in ('mpi_ranks','omp_threads','maximum_stage_count'):
        if type(resources[key]) is not int or resources[key]<1:raise ValueError('Exact positive native SIDM resource counts required')
    _positive(request['wall_seconds_per_native_run']);pins={}
    def asset(binding):
        if set(binding)!={'path','sha256'}:raise ValueError('Exact retained SIDM source asset binding required')
        path=path_at(base,binding['path'])
        if not path.is_file() or path.is_symlink() or file_sha256(path)!=binding['sha256']:raise ValueError('Native SIDM retained asset changed: '+str(path))
        _merge(pins,{str(path):binding['sha256']});return path
    answer_path=asset(request['base_answers']);design_path=asset(request['measurement_design'])
    if max(answer_path.stat().st_size,design_path.stat().st_size)*16>resources['memory_gib']*1024**3:raise MemoryError('SIDM design metadata exceeds allocation')
    answers=json.loads(answer_path.read_text());design=json.loads(design_path.read_text());_no_placeholders(answers);_no_placeholders(design)
    if set(design)!={'schema','per_branch','postprocess','variant_routes'} or design['schema']!='native-sidm-measurement-design-v1' or set(design['per_branch'])!={'cdm','sidm'}:
        raise ValueError('Complete native SIDM per-branch measurement and all13 route design required')
    snapshots=request['snapshots']
    if not isinstance(snapshots,list) or len(snapshots)<2 or any(type(x) is not int or x<0 for x in snapshots) or len(set(snapshots))!=len(snapshots):raise ValueError('At least two exact saved native SIDM/CDM selectors required')
    ic=request['initial_conditions']
    if set(ic)!={'primary','files','native_generation_report'}:raise ValueError('Complete shared native IC generation binding required')
    primary=asset(ic['primary']);files=[asset(b) for b in ic['files']]
    if primary not in files or len(set(files))!=len(files):raise ValueError('Shared native IC primary must appear exactly once in the complete shard list')
    generation=asset(ic['native_generation_report'])
    asset(request['phase_asset'])  # Retain the declared phase source with the common IC inputs.
    background=asset(request['background'])
    if request['sidm'].get('enabled') is not True:raise ValueError('Actual enabled SIDM physical model required')
    cases={};configs={};contexts={};models={};serialized={}
    shared=copy.deepcopy(answers);shared['force_equal_timesteps']=True
    for label,enabled in [('cdm',False),('sidm',True)]:
        current=copy.deepcopy(shared);current['name']='matched-'+label;current['sidm']=copy.deepcopy(request['sidm']) if enabled else {'enabled':False}
        if enabled:current['sidm']['physical_model']['path']=str(path_at(base,current['sidm']['physical_model']['path']))
        cfg=validate_native_answers(current,resources);flags=set(gadget_config_text(cfg).splitlines())
        if 'FORCE_EQUAL_TIMESTEPS' not in flags or 'HIERARCHICAL_GRAVITY' in flags:raise ValueError('Matched SIDM/CDM must both use global synchronized gravity steps')
        configs[label]=cfg;cases[label]=current;models[label]=sidm_manifest_record(cfg.get('sidm',{'enabled':False}))
        for binding in models[label].get('assets',[]):
            path=Path(binding['path']).resolve()
            if file_sha256(path)!=binding['sha256']:raise ValueError('Native SIDM physical table changed')
            _merge(pins,{str(path):binding['sha256']})
        serialized[label]=(json.dumps(current,indent=2,allow_nan=False)+'\n').encode()
    # Only the nonlinear interaction model differs. This preserves all other
    # base answer controls, including gas/stars/BH choices and output schedule.
    if {k:v for k,v in cases['cdm'].items() if k not in {'name','sidm'}}!={k:v for k,v in cases['sidm'].items() if k not in {'name','sidm'}}:
        raise ValueError('SIDM/CDM physical contrast changes unrelated numerical or physical controls')
    linear=Path(configs['cdm']['spectrum']['file']).resolve();_merge(pins,{str(linear):file_sha256(linear)})
    if configs['sidm']['spectrum']!=configs['cdm']['spectrum'] or configs['sidm']['cosmology']!=configs['cdm']['cosmology']:
        raise ValueError('SIDM nonlinear contrast cannot replace the common linear cosmology/transfer bundle')
    descriptor=json.loads(linear.read_text());declared=descriptor.get('checksums_sha256',{})
    if not any((linear.parent/name).resolve()==background and checksum==pins[str(background)] for name,checksum in declared.items()):
        raise ValueError('SIDM matched proper-clock background is not an exact source of the selected linear bundle')
    matching=copy.deepcopy(request['matching'])
    if set(matching)!={'halo_match_namespace','halo_match_id','baryon_control'} or not isinstance(matching['halo_match_namespace'],str) or not matching['halo_match_namespace'] or not isinstance(matching['baryon_control'],str) or not matching['baryon_control'] or type(matching['halo_match_id']) is not int or not 0<=matching['halo_match_id']<2**64:
        raise ValueError('Exact common native target identity and named baryon control required')
    matching.update(phase_seed=cases['cdm']['seed'],linear_bundle_sha256=pins[str(linear)],background_sha256=pins[str(background)])
    baryons=[]
    for cfg in configs.values():baryons.append(canonical_digest({'physics':cfg['simulation']['physics'],**{name:cfg.get(name,{}) for name in ('hydro','black_holes','stellar_evolution','chemistry','external_potential','baryonic_force','stellar_sources')}}))
    if len(set(baryons))!=1:raise ValueError('Matched SIDM/CDM native baryonic controls differ')
    matching['baryon_control_sha256']=baryons[0]
    # Reuse hashes checked by asset(); the final pass still detects input changes.
    namespace='matched-native-ic:'+canonical_digest({str(p):pins[str(p)] for p in files})
    stages=[];requested=[]
    registry=default_registry()
    for binding in request['auxiliary_workflows']:
        path=asset(binding);spec=json.loads(path.read_text());_no_placeholders(spec)
        for key in ('memory_gib','disk_gib','mpi_ranks','omp_threads'):
            if spec['resources'][key]>resources[key]:raise MemoryError('Auxiliary native calibration/forward workflow exceeds common SIDM allocation')
        compile_requested_product_dag(spec,registry,request_base=path.parent)
        stages.extend(copy.deepcopy(spec['stages']));requested.extend(spec['requested_products'])
    for label in ('cdm','sidm'):
        name=label+'_native';answer=out/(label+'.answers.json')
        import hashlib
        stage_pins={**pins,str(answer):hashlib.sha256(serialized[label]).hexdigest()}
        stages.append({'stage_id':name,'kind':'public_command','parameters':{'command':'gadget','arguments':['--answers',{'input':str(answer)},'--existing-ic',{'input':str(primary)},
            '--native-generation-report',{'input':str(generation)},'--run-dir','{generation}/run'],'wall_seconds':request['wall_seconds_per_native_run']},
            'input_files':[{'path':p,'sha256':h} for p,h in sorted(stage_pins.items())],
            'outputs':{name:'run'},'product_contracts':{name:{'reader':'native_snapshot_series','snapshots':snapshots}}})
        contexts[label]={'native_run_directory':{'stage':name,'artifact':'run'},'native_stage_receipt':{'stage':name,'artifact':'stage-receipt.json'},
            'native_matching':matching,'particle_namespace':namespace,'dark_matter_model':'collisionless' if label=='cdm' else 'elastic_sidm','require_sidm_diagnostics':label=='sidm','producer_base_directory':str(base)}
        templates=design['per_branch'][label]
        if isinstance(templates,dict):
            if [r['snapshot'] for r in templates.get('observations',[])]!=snapshots:
                raise ValueError('Generated native SIDM measurement selections must cover the exact configured snapshot set')
            templates=native_branch_measurements(templates,memory_gib=resources['memory_gib'])
        if not isinstance(templates,list) or not templates:raise ValueError('Both SIDM/CDM branches need native measurement producers')
        local={safe_id(row['stage_id']):label+'_'+row['stage_id'] for row in templates}
        if len(local)!=len(templates):raise ValueError('Repeated native SIDM measurement template stage ID')
        def expand(value,context,scope):
            if isinstance(value,dict):
                if set(value)=={'campaign_value'}:
                    if value['campaign_value'] not in context:raise ValueError('Unknown native SIDM campaign substitution')
                    return copy.deepcopy(context[value['campaign_value']])
                if 'stage' in value and set(value)<={'stage','artifact','load','field'}:
                    if value['stage'] not in scope:raise ValueError('Native SIDM per-branch template references an undeclared local producer')
                    return {**value,'stage':scope[value['stage']]}
                return {k:expand(v,context,scope) for k,v in value.items()}
            if isinstance(value,list):return [expand(v,context,scope) for v in value]
            return value
        accessible={row['stage_id']:row['stage_id'] for row in stages}|local
        for template in templates:
            row=expand(copy.deepcopy(template),contexts[label],accessible);row['stage_id']=local[template['stage_id']]
            if row['kind']=='public_command':raise ValueError('Scientific measurement templates cannot inject additional native evolution commands')
            registry.get(row['kind']);validate_sidm_template(row);row['depends_on']=[name,*[accessible[x] for x in template.get('depends_on',[])]]
            row['outputs']={label+'_'+k:v for k,v in template['outputs'].items()}
            if row.get('product_contracts'):row['product_contracts']={label+'_'+k:v for k,v in row['product_contracts'].items()}
            stages.append(row)
    global_context={'native_runs':[{'run_directory':contexts[k]['native_run_directory'],'stage_receipt':contexts[k]['native_stage_receipt']} for k in ('cdm','sidm')],
        'native_run_paths':{k:contexts[k]['native_run_directory'] for k in ('cdm','sidm')},'native_matching':matching,'particle_namespace':namespace,
        'producer_base_directory':str(base),'phase_baseline_run_directory':contexts['cdm']['native_run_directory']}
    global_scope={row['stage_id']:row['stage_id'] for row in stages}|{row['stage_id']:row['stage_id'] for row in design['postprocess']}
    for template in design['postprocess']:
        row=expand(copy.deepcopy(template),global_context,global_scope);registry.get(row['kind']);validate_sidm_template(row)
        if row['kind']=='public_command':raise ValueError('Postprocessing cannot inject unconfigured native evolution commands')
        stages.append(row)
    ids=[r['stage_id'] for r in stages]
    if len(set(ids))!=len(ids):raise ValueError('Native SIDM and auxiliary stage IDs collide')
    routes=design['variant_routes']
    if set(routes)!=set(VARIANTS):raise ValueError('Native SIDM campaign must connect exactly all13 variant routes')
    by_id={r['stage_id']:r for r in stages}
    for variant,targets in routes.items():
        if not isinstance(targets,list) or not targets:raise ValueError('Every SIDM variant requires a reachable exporter')
        for target in targets:
            row=by_id.get(target)
            if row is None or row['kind']!='native_sidm_export':raise ValueError('SIDM variant route must reference an implemented native SIDM exporter')
            op=row['parameters']['operation'];cap=op if op!='extended' else row['parameters']['request']['operation']
            if variant not in CAPABILITIES.get(cap,set()):raise ValueError('SIDM graph route is connected to the wrong physical producer')
            product=row['parameters']['product']
            if isinstance(product,dict) and 'stage' in product:
                producer=by_id.get(product['stage']);kind=None if producer is None else producer['kind']
                expected_kind={'history':'sidm_history','scattering':'sidm_history','centroids':'sidm_history',
                    'matched':'sidm_matched','satellite':'sidm_derived','degeneracy':'sidm_derived','observations':'sidm_observations'}[cap]
                if kind!=expected_kind:raise ValueError('SIDM graph route has the wrong upstream native measurement schema')
                if cap in {'satellite','degeneracy'} and producer['parameters']['request']['product']!=cap:
                    raise ValueError('SIDM derived graph route has the wrong calibrated/native producer')
                if cap=='observations':
                    comparisons=producer['parameters']['request']['comparisons']
                    if isinstance(comparisons,list):
                        families={c['family'] for c in comparisons}
                        if variant=='sidm_stream_profile_response' and 'streams' not in families or variant=='sidm_lensing_h0_response' and not families&{'lensing','h0_estimator'}:
                            raise ValueError('SIDM observational route omits its actual forward measurement family')
            else:
                from .workflow import _stage_references
                if list(_stage_references(row['parameters'])):raise ValueError('Native SIDM graph product must be an owned producer artifact or fully frozen retained input')
                from .sidm_products import sidm_export_inputs
                _merge(pins,sidm_export_inputs(row['parameters']))
            expected=row['parameters'].setdefault('expected_variants',[])
            if variant not in expected:expected.append(variant)
            requested.extend(row['outputs'])
    if len(stages)>resources['maximum_stage_count']:raise MemoryError('Complete all13 SIDM producer DAG exceeds frozen stage ceiling')
    workflow={'schema':'research-workflow-v1','resources':resources,'physics_spec':{},'capabilities':[],
        'followup_limits':{'maximum_children':0,'maximum_depth':0},'requested_products':list(dict.fromkeys(requested)),'stages':stages}
    # Stage-referenced paths/hashes are resolved only from owned completed
    # native/measurement generations by the existing workflow runner.
    # Answer files are new immutable preparation artifacts, written after all
    # physical source checks; the ordinary compiler checks the whole graph.
    out.mkdir(parents=True)
    for label in ('cdm','sidm'):(out/(label+'.answers.json')).write_bytes(serialized[label])
    try:plan=compile_requested_product_dag(workflow,registry,request_base=base)
    except BaseException as exc:
        atomic_json(out/'preparation-failure.json',{'status':'FAIL','reason':str(exc),'native_execution':'NOT_EXECUTED'});raise
    for row in plan['stages']:
        _merge(pins,registry.source_hashes(row['kind']))
    _merge(pins,{str(Path(__file__).resolve()):file_sha256(__file__)})
    for path,checksum in pins.items():
        if file_sha256(path)!=checksum:raise ValueError('Native SIDM source changed during preparation')
    atomic_json(out/'request.json',request);atomic_json(out/'workflow.json',workflow)
    result={'schema':'native-sidm-campaign-prepared-v1','status':'RECIPES_PREPARED','variants':VARIANTS,'variant_routes':routes,'stage_count':len(stages),
        'native_cases':{label:{'answers':{'path':str(out/(label+'.answers.json')),'sha256':file_sha256(out/(label+'.answers.json'))},
            'sidm_model':models[label],'force_equal_timesteps':True} for label in ('cdm','sidm')},
        'matching':matching,'source_hashes':pins,'workflow':{'path':str(out/'workflow.json'),'sha256':file_sha256(out/'workflow.json')},
        'scope':'Matched actual common IC source, identical non-SIDM physics/numerical settings, native membership/history/forward/calibration templates and all13 graph routes. Numerical convergence is a separate campaign; no science acceptance.',
        'native_execution':'NOT_EXECUTED','science_acceptance':'NOT_ACCEPTED'}
    atomic_json(out/'manifest.json',result);return result


# Required measurement fields and authored native branch templates

SIDM_TEMPLATE_FIELDS={
 'sidm_membership':('sidm-native-membership-request-v1',{'snapshot','host_row','maximum_particles','memory_gib','support_radius_proper_kpc','species','particle_namespace','matching'}),
 'sidm_history':('sidm-halo-history-request-v1',{'maximum_particles','maximum_epochs','maximum_halos','memory_gib','particle_namespace','species','species_components','dark_matter_model','require_sidm_diagnostics','matching','epochs','radial_edges_proper_kpc','minimum_shell_particles','minimum_effective_particles','minimum_resolved_radius_proper_kpc','minimum_shape_eigenvalue_gap_fraction','maximum_coarse_support_fraction','maximum_omitted_scattering_upper_bound','central_aperture_proper_kpc','minimum_central_particles','core_fit','component_centroids','collapse'}),
 'sidm_satellite':('sidm-native-satellite-request-v1',{'snapshot','subhalo_row','maximum_particles','maximum_bound_particles','maximum_pair_evaluations','potential_chunk_rows','memory_gib','potential_softening_proper_kpc','potential_gauge','species','particle_namespace','satellite_identity','require_sidm_diagnostics','original_membership'}),
 'sidm_matched':('sidm-matched-science-request-v1',{'memory_gib','maximum_histories','baseline_label','histories','observables','absolute_floors','matching_controls'}),
 'sidm_observations':('sidm-observational-comparison-request-v1',{'memory_gib','maximum_comparisons','comparisons'}),
 'sidm_derived':('sidm-derived-science-request-v1',{'memory_gib','product','inputs','settings'}),
}


def validate_sidm_template(row):
    """Check required fields before dependent artifacts resolve at execution.

    The native producers still own physical units, identities and calibration
    checks; this early structural check does not accept a scientific result.
    """
    kind=row['kind']
    if kind not in SIDM_TEMPLATE_FIELDS:return
    request=row['parameters']['request']
    if isinstance(request,dict) and 'stage' in request and set(request)<={'stage','artifact','load','field'}:
        if request.get('load')!='json':raise ValueError('Dynamically produced SIDM requests must load an actual JSON artifact')
        return
    schema,fields=SIDM_TEMPLATE_FIELDS[kind]
    if not isinstance(request,dict) or request.get('schema')!=schema or not fields<=set(request):
        raise ValueError('Incomplete original '+kind+' request fields: '+','.join(sorted(fields-set(request) if isinstance(request,dict) else fields)))
    if kind=='sidm_history':
        for row in request['epochs']:
            if set(row)!={'snapshot','elapsed_time_gyr','selection'}:raise ValueError('History epochs require actual native clock and measured membership binding')
    if kind=='sidm_observations':
        for comparison in request['comparisons']:
            if not {'comparison_id','family','collisionless','sidm','matching','observables'}<=set(comparison):raise ValueError('Complete native SIDM matched forward contrast required')
    if kind=='sidm_derived' and request['product']=='satellite' and len(request['inputs'])!=2:
        raise ValueError('Satellite survival requires initial and current native bound measurements')
    if kind=='sidm_derived' and request['product']=='degeneracy' and len(request['inputs'])!=1:
        raise ValueError('SIDM degeneracy requires one independently calibrated joint response')


def native_branch_measurements(settings,*,memory_gib):
    """Build native membership/history requests from explicit host selections.

    Support radii and apertures are proper kpc. Only native fine type-1 DM
    participates in SIDM science; other species retain their stated component.
    Physical elapsed Gyr comes from each membership producer, rather than
    interpreting a snapshot number as a clock.
    """
    _positive(memory_gib)
    required={'schema','observations','maximum_particles','support_radius_proper_kpc','species_components','history_controls','satellite'}
    if set(settings)!=required or settings['schema']!='native-sidm-branch-measurements-v1':raise ValueError('Complete source-defined SIDM branch measurements required')
    _no_placeholders(settings);observations=settings['observations']
    if not isinstance(observations,list) or len(observations)<2:raise ValueError('At least two explicit native host selections required')
    snapshots=[r['snapshot'] for r in observations]
    if any(type(x) is not int or x<0 for x in snapshots) or len(set(snapshots))!=len(snapshots):raise ValueError('Distinct exact native saved selections required')
    species=settings['species_components']
    if not isinstance(species,dict) or species.get('1')!='dm' or any(k not in map(str,range(6)) for k in species):raise ValueError('Explicit fine-DM and other native species/component mapping required')
    if any(name=='dm' and key!='1' for key,name in species.items()):raise ValueError('Only fine type1 can be SIDM science DM')
    maximum=settings['maximum_particles']
    if type(maximum) is not int or maximum<1:raise ValueError('Positive exact native census and workspace required')
    value=lambda k:{'campaign_value':k}
    ref=lambda name,**extra:{'stage':name,'artifact':'product.json',**extra}
    product=lambda name:{'path':ref(name),'sha256':ref(name,load='sha256')}
    def stage(name,kind,request):
        result={'stage_id':name,'kind':kind,'parameters':{'request':request,'request_base':value('producer_base_directory'),
            'destination':'{generation}/product.json','native_run_directory':value('native_run_directory'),
            'native_stage_receipt':value('native_stage_receipt')},'outputs':{name:'product.json'}}
        validate_sidm_template(result);return result
    stages=[];epochs=[]
    for i,row in enumerate(observations):
        if set(row)!={'snapshot','host_row'}:raise ValueError('Exact native saved snapshot/FOF-row observation required')
        if type(row['host_row']) is not int and not isinstance(row['host_row'],dict):raise ValueError('Source-selected exact FOF row or actual producer reference required')
        if type(row['host_row']) is int and row['host_row']<0:raise ValueError('Native FOF row cannot be negative')
        name=f'members_{i:03d}'
        request={'schema':'sidm-native-membership-request-v1',**copy.deepcopy(row),'maximum_particles':maximum,'memory_gib':memory_gib,
            'support_radius_proper_kpc':settings['support_radius_proper_kpc'],'species':[int(k) for k in species],
            'particle_namespace':value('particle_namespace'),'matching':value('native_matching')}
        stages.append(stage(name,'sidm_membership',request))
        epochs.append({'snapshot':row['snapshot'],'elapsed_time_gyr':ref(name,load='json',field=['elapsed_time_gyr']),'selection':product(name)})
    fixed={'schema','memory_gib','maximum_particles','maximum_epochs','maximum_halos','species','species_components',
        'particle_namespace','matching','dark_matter_model','require_sidm_diagnostics','epochs'}
    if set(settings['history_controls'])&fixed:raise ValueError('Physical history controls cannot replace generated native identity/epoch/model bindings')
    request={'schema':'sidm-halo-history-request-v1','memory_gib':memory_gib,'maximum_particles':maximum,'maximum_epochs':len(epochs),
        'maximum_halos':1,'species':[int(k) for k in species],'species_components':copy.deepcopy(species),
        'particle_namespace':value('particle_namespace'),'matching':value('native_matching'),'dark_matter_model':value('dark_matter_model'),
        'require_sidm_diagnostics':value('require_sidm_diagnostics'),'epochs':epochs,**copy.deepcopy(settings['history_controls'])}
    if max(request['radial_edges_proper_kpc'])>settings['support_radius_proper_kpc'] or request['component_centroids']['aperture_proper_kpc']>settings['support_radius_proper_kpc']:
        raise ValueError('SIDM profile/centroid apertures exceed the complete native membership support')
    stages.append(stage('history','sidm_history',request))
    satellite=settings['satellite']
    if satellite is not None:
        if set(satellite)!={'identity','subhalo_rows','controls'} or len(satellite['subhalo_rows'])!=len(epochs):raise ValueError('Exact saved native satellite lineage selections and physical controls required')
        for i,(epoch,row) in enumerate(zip(epochs,satellite['subhalo_rows'])):
            if type(row) is not int and not isinstance(row,dict):raise ValueError('Exact native satellite row or source-selected row reference required')
            request={**copy.deepcopy(satellite['controls']),'schema':'sidm-native-satellite-request-v1','snapshot':epoch['snapshot'],'subhalo_row':row,
                'memory_gib':memory_gib,'maximum_particles':maximum,'particle_namespace':value('particle_namespace'),
                'satellite_identity':copy.deepcopy(satellite['identity']),'require_sidm_diagnostics':value('require_sidm_diagnostics'),
                'original_membership':product('members_000' if i==0 else 'satellite_000')}
            stages.append(stage(f'satellite_{i:03d}','sidm_satellite',request))
    return stages


# Resource measurements from completed native logs

RESOURCE_COUNTERS=('local_candidate_pairs','received_ghost_records','matching_rounds','remote_endpoint_updates',
          'exchange_sent_bytes','exchange_received_bytes','completed_pair_operations')


def validate_sidm_resources(parameters):
    request=parameters['request']
    if request.get('schema')!='native-sidm-resource-request-v1':raise ValueError('Native SIDM resource request required')
    for name in ('maximum_logs','expected_mpi_ranks'):
        if type(request[name]) is not int or request[name]<1:raise ValueError('Positive exact resource bound required: '+name)
    if request['expected_mpi_ranks']>65536:raise ValueError('Native event identity rank capacity exceeded')
    files=request['event_files']
    if not isinstance(files,list) or not 1<=len(files)<=request['maximum_logs'] or len(set(files))!=len(files):
        raise ValueError('Distinct bounded actual native resource log files required')
    _positive(request['memory_gib'])
    if len(files)*16384>request['memory_gib']*1024**3:raise MemoryError('Native resource log metadata exceeds workspace')
    intervals=request['expected_intervals']
    if not isinstance(intervals,list) or not intervals or len({canonical_digest(row) for row in intervals})!=len(intervals):
        raise ValueError('Distinct actual native resource intervals required')
    for row in intervals:
        if set(row)!={'native_begin','native_end'} or any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) for value in row.values()) or row['native_end']<=row['native_begin']:
            raise ValueError('Native resource interval must retain finite increasing native coordinates')


def sidm_resource_input_bindings(parameters):
    """Resolve logs from their completed native generation without relabeling it."""
    from .workflow import verify_completed_receipt

    validate_sidm_resources(parameters)
    request = parameters['request']
    path = Path(parameters['native_stage_receipt']).resolve()
    if path.stat().st_size * 8 > request['memory_gib'] * 1024**3:
        raise MemoryError('Native resource receipt exceeds input workspace')
    receipt = verify_completed_receipt(path)
    root = Path(receipt['generation']).resolve()
    sources = {str(path): file_sha256(path)}
    _merge(sources, receipt['source_hashes'])
    _merge(sources, receipt['input_hashes'])
    for name in request['event_files']:
        file = Path(name).resolve()
        if not file.is_relative_to(root):
            raise ValueError('SIDM resource log is outside the actual completed native evolution payload')
        checksum = file_sha256(file)
        if receipt['files_sha256'].get(file.relative_to(root).as_posix()) != checksum:
            raise ValueError('SIDM resource log differs from the completed native evolution payload')
        _merge(sources, {str(file): checksum})
    return sources


def _resource_counter(attributes,name):
    value=attributes[name]
    if isinstance(value,bytes):value=value.decode('ascii')
    if not isinstance(value,str) or not value or len(value)>20 or not value.isascii() or not value.isdecimal():
        raise ValueError('Native scheduling counter must retain its exact unsigned decimal encoding: '+name)
    result=int(value)
    if result>=2**64:raise ValueError('Native scheduling counter exceeds uint64')
    return result


def _resource_measurement(attributes):
    if attributes.get('resource_measurement_schema')!='native-sidm-source-resources-v1':
        return None
    if attributes.get('resource_measurement_origin')!='original_completed_log_producer':
        raise ValueError('Unknown native resource producer scope')
    cpu_status=attributes['cpu_measurement_status'];rss_status=attributes['rss_measurement_status']
    if cpu_status not in {'MEASURED','UNAVAILABLE'} or rss_status not in {'MEASURED','UNAVAILABLE'}:
        raise ValueError('Invalid native resource availability status')
    result={'cpu_available':cpu_status=='MEASURED','rss_available':rss_status=='MEASURED',
        'origin':attributes['resource_measurement_origin'],
        'cpu_scope':attributes['cpu_scope'],'rss_scope':attributes['rss_scope'],
        'exchange_memory_scope':attributes['exchange_memory_scope'],'table_memory_scope':attributes['table_memory_scope']}
    for key in ('native_pair_cpu_seconds','native_rare_cpu_seconds','native_frequent_cpu_seconds'):
        result[key]=float(attributes[key])
        if not math.isfinite(result[key]) or result[key]<0:raise ValueError('Invalid measured native CPU time')
    for key in ('native_rare_operator_calls','native_frequent_operator_calls','process_peak_rss_bytes',
                'exchange_buffer_payload_peak_bytes','kernel_lookup_table_memory_bytes','angular_lookup_table_memory_bytes'):
        result[key]=_resource_counter(attributes,key)
    return result


def measure_native_sidm_resources(parameters):
    """Reduce complete rank/interval telemetry without double-counting channels.

    Sum work, sent bytes and calling-thread CPU seconds. Synchronized matching
    rounds count once per interval. Hybrid rare/frequent logs share scheduling
    telemetry, which is retained once per rank and interval. RSS is a process
    lifetime peak: sum(max_t RSS_rank) is only an upper bound on simultaneous
    memory, not max_t(sum(RSS_rank)). No missing metric becomes a measured zero.
    """
    import h5py
    request=parameters['request'];sources=sidm_resource_input_bindings(parameters)
    expected={(row['native_begin'],row['native_end']) for row in request['expected_intervals']}
    records={};model=None;namespace=None;coordinate=None;storage=0
    for name in request['event_files']:
        file=Path(name).resolve();storage+=file.stat().st_size
        with h5py.File(file,'r') as handle:
            if handle.attrs.get('publication_state')!='COMPLETE':raise ValueError('Native resource log is incomplete')
            group=handle['SIDMScatteringEvents'];a=group.attrs
            if a.get('schema')!='native-sidm-event-log-v1' or a.get('pair_schedule') not in {
                    'mutual_canonical_endpoint_choices','single_rank_canonical_all_pairs',
                    'mutual_random_priority_endpoint_choices','single_rank_random_priority_all_pairs'}:
                raise ValueError('Actual native SIDM scheduling measurements required')
            current_model=a['angular_model_hash'];current_namespace=a['event_identity_namespace'];current_coordinate=a['native_clock_coordinate']
            if (model is not None and model!=current_model) or (namespace is not None and namespace!=current_namespace) or (coordinate is not None and coordinate!=current_coordinate):
                raise ValueError('Native resource logs change physical model, event namespace or clock')
            model,namespace,coordinate=current_model,current_namespace,current_coordinate
            interval=(float(a['native_interval_begin']),float(a['native_interval_end']))
            rank,ranks=_resource_counter(a,'mpi_rank'),_resource_counter(a,'mpi_ranks')
            if interval not in expected or ranks!=request['expected_mpi_ranks'] or rank>=ranks:
                raise ValueError('Native resource log differs from declared actual interval/rank coverage')
            counters={key:_resource_counter(a,key) for key in RESOURCE_COUNTERS}
            schedule=a['pair_schedule']
            if schedule in {'single_rank_canonical_all_pairs','single_rank_random_priority_all_pairs'} and (ranks!=1 or rank or any(counters[key] for key in RESOURCE_COUNTERS if key not in {'local_candidate_pairs','completed_pair_operations'})):
                raise ValueError('Reference native schedule cannot claim MPI exchange work')
            resources=_resource_measurement(a)
            if resources and resources['cpu_available']:
                hybrid=a.get('source_composition')=='small_half_large_full_small_half'
                operations=counters['completed_pair_operations']
                expected_rare=operations if hybrid or a['operator_kind']=='rare_elastic_pair' else 0
                expected_frequent=2*operations if hybrid else operations if a['operator_kind']=='frequent_drag_diffusion_pair' else 0
                if resources['native_rare_operator_calls']!=expected_rare or resources['native_frequent_operator_calls']!=expected_frequent:
                    raise ValueError('Native measured primitive-call census differs from executed pair schedule')
            row={'interval':list(interval),'rank':rank,'counters':counters,'resources':resources,'pair_schedule':schedule,
                'physical_elapsed_seconds':float(a['physical_elapsed_seconds']),
                'source_substep_count':_resource_counter(a,'source_substep_count'),'channels':{},
                'required_channels':['rare_elastic_pair','frequent_drag_diffusion_pair'] if
                    a.get('source_composition')=='small_half_large_full_small_half' else [a['operator_kind']]}
            if not math.isfinite(row['physical_elapsed_seconds']) or row['physical_elapsed_seconds']<=0 or row['source_substep_count']<1:
                raise ValueError('Native source resource interval lacks measured positive physical duration/substeps')
            key=(interval,rank);kind=a['operator_kind']
            if kind not in {'rare_elastic_pair','frequent_drag_diffusion_pair'}:raise ValueError('Unknown native measured source channel')
            if key in records:
                old=records[key]
                if old['resources']!=resources or old['pair_schedule']!=schedule or old['counters']!=counters or old['physical_elapsed_seconds']!=row['physical_elapsed_seconds'] or old['source_substep_count']!=row['source_substep_count'] or old['required_channels']!=row['required_channels'] or kind in old['channels']:
                    raise ValueError('Repeated or inconsistent native source telemetry channels would double-count resources')
                row=old
            row['channels'][kind]={'retained_update_records':len(group['EventID']),'path':str(file)}
            records[key]=row
    summaries=[]
    for interval in sorted(expected):
        members=[records.get((interval,rank)) for rank in range(request['expected_mpi_ranks'])]
        if any(row is None for row in members):raise ValueError('Native SIDM resource interval omits an MPI rank, including empty-work ranks')
        if any(set(row['channels'])!=set(row['required_channels']) for row in members):
            raise ValueError('Native hybrid resource coverage omits a rare or effective diffusion channel')
        if len({row['counters']['matching_rounds'] for row in members})!=1 or len({row['source_substep_count'] for row in members})!=1 or len({row['physical_elapsed_seconds'] for row in members})!=1:
            raise ValueError('Synchronized native ranks disagree about actual source interval scheduling')
        totals={key:sum(row['counters'][key] for row in members) for key in RESOURCE_COUNTERS}
        if totals['exchange_sent_bytes']!=totals['exchange_received_bytes']:raise ValueError('Complete native communication coverage has unmatched sent/received byte census')
        summaries.append({'interval':list(interval),'totals':totals,
            'synchronization_rounds':members[0]['counters']['matching_rounds'],
            'source_substeps':members[0]['source_substep_count'],
            'rare_event_records':sum(row['channels'].get('rare_elastic_pair',{}).get('retained_update_records',0) for row in members),
            'frequent_update_records':sum(row['channels'].get('frequent_drag_diffusion_pair',{}).get('retained_update_records',0) for row in members)})
    metrics={'sidm_pair_operations':sum(row['totals']['completed_pair_operations'] for row in summaries),
        'sidm_exchange_bytes':sum(row['totals']['exchange_sent_bytes'] for row in summaries),
        'sidm_synchronization_rounds':sum(row['synchronization_rounds'] for row in summaries),
        'sidm_source_substeps':sum(row['source_substeps'] for row in summaries),
        'sidm_event_log_bytes':storage,'sidm_frequent_updates':sum(row['frequent_update_records'] for row in summaries)}
    rank_records=list(records.values())
    resources=[row['resources'] for row in rank_records]
    unmeasured=[]
    if all(value and value['cpu_available'] for value in resources):
        for native,metric in (('native_pair_cpu_seconds','sidm_pair_cpu_seconds'),
                              ('native_rare_cpu_seconds','sidm_rare_cpu_seconds'),
                              ('native_frequent_cpu_seconds','sidm_frequent_cpu_seconds')):
            metrics[metric]=sum(value[native] for value in resources)
        metrics['sidm_rare_operator_calls']=sum(value['native_rare_operator_calls'] for value in resources)
        metrics['sidm_frequent_operator_calls']=sum(value['native_frequent_operator_calls'] for value in resources)
    else:
        unmeasured.extend(['sidm_pair_cpu_seconds','sidm_rare_cpu_seconds','sidm_frequent_cpu_seconds'])
    if all(value and value['rss_available'] for value in resources):
        rank_peaks={rank:max(row['resources']['process_peak_rss_bytes'] for row in rank_records if row['rank']==rank)
                    for rank in range(request['expected_mpi_ranks'])}
        metrics['sidm_peak_rank_rss_bytes']=max(rank_peaks.values())
        metrics['sidm_sum_rank_rss_peaks_upper_bound_bytes']=sum(rank_peaks.values())
    else:
        rank_peaks={}
        unmeasured.append('sidm_peak_rank_rss_bytes')
    if all(resources):
        metrics['sidm_exchange_buffer_payload_peak_rank_bytes']=max(value['exchange_buffer_payload_peak_bytes'] for value in resources)
        metrics['sidm_kernel_lookup_table_memory_bytes']=sum(max(row['resources']['kernel_lookup_table_memory_bytes']
            for row in rank_records if row['rank']==rank) for rank in range(request['expected_mpi_ranks']))
        metrics['sidm_angular_lookup_table_memory_bytes']=sum(max(row['resources']['angular_lookup_table_memory_bytes']
            for row in rank_records if row['rank']==rank) for rank in range(request['expected_mpi_ranks']))
    else:
        unmeasured.extend(['sidm_exchange_buffer_payload_peak_rank_bytes','sidm_kernel_lookup_table_memory_bytes',
                           'sidm_angular_lookup_table_memory_bytes'])
    costs={}
    for channel in ('rare','frequent'):
        seconds=metrics.get('sidm_'+channel+'_cpu_seconds')
        calls=metrics.get('sidm_'+channel+'_operator_calls')
        costs[channel]={'cpu_seconds_per_operator_call':seconds/calls if seconds is not None and calls else None,
            'operator_calls':calls,'scope':'actual retained calling-thread primitive scopes; not an extrapolated full-run cost'}
    for path,digest in sources.items():
        if file_sha256(path)!=digest:raise ValueError('Native scheduling source changed during resource publication')
    return {'schema':'native-sidm-resource-measurements-v1','request_hash':canonical_digest(request),'measurements':metrics,
        'intervals':summaries,'rank_records':list(records.values()),'source_hashes':sources,
        'angular_model_hash':model,'event_identity_namespace':namespace,'native_clock_coordinate':coordinate,
        'execution_status':'COMPLETE','science_acceptance':'NOT_ESTABLISHED',
        'coverage':'exact supplied native intervals and complete MPI rank set; no full-run extrapolation',
         'unmeasured':unmeasured,
        'rank_process_peak_rss_bytes':rank_peaks,'measured_operator_costs':costs,
        'measurement_scope':{'cpu':'sum actual calling-thread operator scopes over supplied intervals/ranks; includes logged update writes',
            'rss':'individual process lifetime high-water marks through supplied source intervals; sum of peaks is an upper bound, not a simultaneous measurement',
            'exchange_memory':'largest measured single-exchange vector capacity payload per rank; excludes allocator overhead and MPI-internal buffers',
            'lookup_tables':'measured retained differential-table vector capacity payload; analytic angular models and kernels allocate no dynamic lookup tables',
            'restart_replay':'completed logs retain measurements of their original producer; replay timings are not substituted'},
        'counting_policy':'sum work and sent bytes; synchronized rounds counted once per interval; hybrid channel telemetry deduplicated'}


def native_sidm_resources_runner(parameters,context):
    if parameters['request']['memory_gib']>context.resources['memory_gib']:raise MemoryError('Native resource decoder exceeds allocated memory')
    result=measure_native_sidm_resources(parameters)
    atomic_json(context.generation/'native-sidm-resources.json',result)
    return {'measurements':result['measurements'],'scientific_exclusions':[{'reason':reason} for reason in result['unmeasured']]}


# Caller-authored physical control plans; generation does not execute them

CONTROL_DEFINITIONS = {
    "zero_cross_section": {"observable": "particle-ID matched positions and physical peculiar velocities", "comparison": "collisionless gravity/source run", "dimensions": ["gravity_timestep", "mpi_ranks"]},
    "pair_conservation": {"observable": "pair momentum, center-of-mass velocity and kinetic-energy residuals", "comparison": "analytic equal-mass elastic update", "dimensions": ["native_velocity_precision", "relative_speed", "angle"]},
    "uniform_beam": {"observable": "per-particle microscopic exposure, rare event rate and velocity-distribution CDF", "comparison": "uniform density rate and independent OpenGadget3 run", "dimensions": ["macro_mass", "kernel_support", "source_timestep"]},
    "angular_distribution": {"observable": "conditional scattering-cosine CDF at fixed relative speeds", "comparison": "retained normalized differential interpolant", "dimensions": ["speed", "cdf_tolerance", "particle_convention"]},
    "thermal_equilibrium": {"observable": "velocity-distribution moments and equilibrium-relaxation trajectory", "comparison": "matched OpenGadget3 thermal population", "dimensions": ["temperature", "particle_number", "source_timestep"]},
    "physical_comoving": {"observable": "matched physical rate, elapsed proper time and pair conservation", "comparison": "same physical source population represented in comoving and physical units", "dimensions": ["scale_factor", "clock_tolerance", "gravity_timestep"]},
    "mpi_boundaries": {"observable": "unique unordered-pair census, owner commits, keyed draw identities and exposure", "comparison": "same particle-ID population across MPI counts", "dimensions": ["mpi_ranks", "boundary_location", "ghost_workspace"]},
    "restart_continuation": {"observable": "all native particle/source/global states at matched synchronized epochs", "comparison": "uninterrupted and restarted runs with identical build/model/table", "dimensions": ["checkpoint_epoch", "operator_seed", "scheme"]},
    "constant_table_agreement": {"observable": "same differential kernel moments, pair draws and ensemble rates", "comparison": "analytic constant-isotropic and constant differential table", "dimensions": ["table_grid", "source_timestep", "particle_number"]},
    "rare_frequent_hybrid": {"observable": "transport moments and matched ensemble relaxation", "comparison": "same forward-peaked kernel in each permitted small-angle domain", "dimensions": ["source_timestep", "split_angle", "neglected_moment_ratio"]},
    "isolated_core_formation": {"observable": "density/core radius, anisotropy, velocity dispersion and mass conservation", "comparison": "matched OpenGadget3 halo trajectory", "dimensions": ["macro_mass", "softening", "kernel_support", "source_timestep"]},
    "gravothermal_collapse": {"observable": "central density, temperature, collapse time and outward energy transport", "comparison": "matched independent gravothermal halo campaign", "dimensions": ["macro_mass", "softening", "kernel_support", "gravity_timestep", "source_timestep"]},
    "mass_softening_kernel_convergence": {"observable": "science-aperture profiles and trajectory envelopes", "comparison": "resolved sequence at fixed physical model", "dimensions": ["macro_mass", "softening", "kernel_support", "kernel_kind"]},
    "timestep_split_convergence": {"observable": "angular transport and science trajectory changes", "comparison": "successive gravity/source timestep and disjoint split-angle refinements", "dimensions": ["gravity_timestep", "source_timestep", "split_angle"]},
    "hydro_sidm": {"observable": "DM pair conservation, baryonic response and gravity/source splitting convergence", "comparison": "matched hydro+SIDM and hydro+collisionless controls", "dimensions": ["gas_resolution", "gravity_timestep", "source_timestep", "baryonic_model"]},
    "sidm_zoom": {"observable": "fine/coarse support fraction, cumulative omitted-scattering bound and aperture profiles", "comparison": "expanded fine-region regenerations and high-resolution parent", "dimensions": ["fine_region_padding", "coarse_mass_ratio", "kernel_support", "guard_budget"]},
}


def control_recipe_matrix(base_configuration, variations, *, independent_engine="OpenGadget3"):
    """Expand every required control into explicit, unexecuted retained jobs.

    Every control dimension needs at least two declared values. Authored cases
    must cover every level without duplicate joint coordinates; only those
    authored cases are emitted. No convergence threshold or acceptance result
    is supplied by this planner.
    The actual native materializer consumes each case's `configuration` and
    control-specific `physical_input` supplied by the user/campaign generator.
    """
    if independent_engine != "OpenGadget3":
        raise ValueError("OpenGadget3 is the required independent rare/frequent reference engine")
    if not isinstance(variations, dict) or set(variations) != set(CONTROL_DEFINITIONS):
        raise ValueError("Every required native SIDM control needs an explicit variation recipe")
    jobs = []
    for name, definition in CONTROL_DEFINITIONS.items():
        request = variations[name]
        if not isinstance(request, dict) or not request.get("physical_input") or not request.get("acceptance_bounds"):
            raise ValueError(f"{name}: retained physical input and predeclared acceptance bounds required")
        dimensions = request.get("dimensions", {})
        if set(dimensions) != set(definition["dimensions"]):
            raise ValueError(f"{name}: explicit dimension values must cover the complete physical control")
        cases = request.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"{name}: actual native configuration cases required")
        declared = {}
        for dimension, values in dimensions.items():
            if not isinstance(values, list) or len(values) < 2:
                raise ValueError(f"{name}: {dimension} needs distinct refinement values")
            hashes = {canonical_digest({"value": value}) for value in values}
            if len(hashes) != len(values):
                raise ValueError(f"{name}: {dimension} needs distinct refinement values")
            declared[dimension] = hashes
        covered = {dimension: set() for dimension in dimensions}
        seen_coordinates = set()
        for index, case in enumerate(cases):
            if not isinstance(case, dict) or set(case) != {"configuration_overrides", "coordinates"}:
                raise ValueError(f"{name}: each case must bind actual configuration overrides and dimension coordinates")
            coordinates = case["coordinates"]
            if not isinstance(coordinates, dict) or set(coordinates) != set(dimensions):
                raise ValueError(f"{name}: case coordinates must belong to all declared control axes")
            hashes = {key: canonical_digest({"value": value}) for key, value in coordinates.items()}
            if any(value not in declared[key] for key, value in hashes.items()):
                raise ValueError(f"{name}: case coordinates must belong to all declared control axes")
            signature = canonical_digest(coordinates)
            if signature in seen_coordinates:
                raise ValueError(f"{name}: repeated control coordinates do not provide a refinement")
            seen_coordinates.add(signature)
            for key, value in hashes.items():
                covered[key].add(value)
            config = copy.deepcopy(base_configuration)
            _merge_configuration(config, case["configuration_overrides"])
            jobs.append({"control": name, "case": index, "configuration": config,
                         "coordinates": case["coordinates"], "physical_input": request["physical_input"],
                         "acceptance_bounds": request["acceptance_bounds"], "observable": definition["observable"],
                         "comparison": definition["comparison"], "independent_engine": independent_engine,
                         "execution_status": "NOT_RUN", "science_acceptance": "NOT_ESTABLISHED"})
        # Cover every declared level; the caller chooses which combinations.
        for dimension, values in declared.items():
            if covered[dimension] != values:
                raise ValueError(f"{name}: cases omit declared {dimension} refinement levels")
    return {"schema": "native-sidm-complete-control-matrix-v2", "base_configuration_hash": canonical_digest(base_configuration),
            "jobs": jobs, "required_controls": list(CONTROL_DEFINITIONS), "execution_status": "NOT_RUN"}


def _merge_configuration(target, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict):
            if not isinstance(target.get(key), dict):
                raise ValueError("Control override path must exist in the physical native configuration")
            _merge_configuration(target[key], value)
        else:
            if key not in target:
                raise ValueError("Control override cannot silently introduce an unused native parameter")
            target[key] = value


def sidm_control_plan_main():
    """Author the complete required native control matrix; execute no cases."""
    import argparse
    parser = argparse.ArgumentParser(description='Prepare all required native SIDM control configurations and predeclared comparisons')
    parser.add_argument('request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep previous SIDM control matrix')
    path = args.request.expanduser().resolve()
    digest = file_sha256(path)
    request = json.loads(path.read_text())
    matrix = control_recipe_matrix(request['base_configuration'], request['variations'],
        independent_engine=request['independent_engine'])
    matrix['source_hashes'] = {str(path): digest, str(Path(__file__).resolve()): file_sha256(__file__)}
    if file_sha256(path) != digest:
        raise ValueError('SIDM control plan source changed during materialization')
    atomic_json(args.output, matrix)
    return 0
