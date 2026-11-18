"""Stage native MUSIC cases and bind evolution to their actual generated mesh.

Restaging preserves physical settings and phase inputs, with an explicit
optional start-epoch replacement. Generated-flow preparation uses the exact
new IC, source archive and mesh; it never substitutes an old generated field.
These functions prepare inputs. The separate native handoff and flow checks
still have to pass before evolution.
"""
import json
import math
from pathlib import Path
from core.foundation import file_sha256, atomic_json

_CORE_ROOT = Path(__file__).resolve().parent.parent


def restage_campaign_case(configuration,destination,*,source_archive_max_mib,bounded_source_settings,linear_bundle=None):
    """Recreate a case with explicit source retention and bounded corrections.

    An optional replacement linear bundle may change the starting epoch and
    its derived 2LPT coefficients. Every other physical parameter and phase
    input must agree; restaging is not permission to change the cosmology.
    """
    from core.zoom import _configuration,verify_staged_zoom_inputs,stage_zoom_inputs,validate_bounded_source_settings
    source=Path(configuration).resolve()
    target=Path(destination).resolve()
    cfg=_configuration(source)
    output=target.with_name(target.name+'.restaging.json')
    # Respect a report or legacy sidecar left by an earlier preparation.
    if output.exists() or output.with_name(output.name+'.sha256').exists():
        raise FileExistsError('Preserve earlier case-restaging provenance')
    old=verify_staged_zoom_inputs(source.parent)
    order=old['perturbation_order']
    settings=validate_bounded_source_settings(bounded_source_settings,lpt_order=order)
    if settings is None:
        raise ValueError('Explicit bounded correction settings required for original-flow case staging')
    if (type(source_archive_max_mib) is not int or not 0<source_archive_max_mib<=(2**64-1)//1024**2 or order!=2):
        raise ValueError('Case source retention requires mixed2LPT and exact positive representable disk allocation')
    spectrum=Path(old['solver_manifest']['path']).resolve()
    if file_sha256(spectrum)!=old['solver_manifest']['sha256']:
        raise ValueError('Case source solver manifest changed')
    pins={str(source):file_sha256(source),str(source.parent/'staging.json'):file_sha256(source.parent/'staging.json'),
        str(spectrum):file_sha256(spectrum)}
    if linear_bundle is not None:
        import copy,json
        if set(linear_bundle)!={'path','sha256'}:
            raise ValueError('Exact regenerated epoch bundle binding required')
        replacement=Path(linear_bundle['path']).resolve()
        if file_sha256(replacement)!=linear_bundle['sha256']:
            raise ValueError('Regenerated epoch bundle changed')
        previous=json.loads(spectrum.read_text())
        current=json.loads(replacement.read_text())
        def physical(record):
            values=copy.deepcopy(record['parameters'])
            values.pop('name',None)
            values.pop('z_start',None)
            return values
        if physical(previous)!=physical(current):
            raise ValueError('Start-epoch restaging changes unrelated linear physics')
        if previous['parameters']['z_start']==current['parameters']['z_start']:
            raise ValueError('Start-epoch restaging requires a different actual epoch')
        pins[str(replacement)]=linear_bundle['sha256']
        spectrum=replacement
    for ref in old.get('phase_noise_by_level',{}).values():
        path=Path(ref['path']).resolve()
        if file_sha256(path)!=ref['sha256']:
            raise ValueError('Case original frozen phase file changed')
        pins[str(path)]=ref['sha256']
    stage_zoom_inputs(spectrum,target,box_mpc_h=old['box_mpc_h'],parent_level=old['parent_level'],
        refinement_level=old['refinement_level'],seeds_by_level=old['seeds_by_level'],padding_cells=cfg.getint('setup','padding'),
        poisson_accuracy=cfg.getfloat('poisson','accuracy'),wdm_request=None if old.get('wdm') is None else old['wdm']['request'],
        lpt_order=order,lpt_seed_kind=old.get('lpt_seed_kind','radiation_regular'),lpt_seed=old.get('lpt_seed'),
        phase_noise_by_level=old.get('phase_noise_by_level'),bounded_source_settings=settings,source_archive_max_mib=source_archive_max_mib)
    actual=verify_staged_zoom_inputs(target)
    for key in ('box_mpc_h','parent_level','refinement_level','seeds_by_level','phase_noise_by_level','perturbation_order','lpt_seed_kind','lpt_seed'):
        if actual.get(key)!=old.get(key):
            raise ValueError('Restaged case changes original physical/phase setting: '+key)
    new_cfg=_configuration(target/'zoom_template.conf')
    for section,keys in {'setup':('boxlength','zstart','levelmin','levelmax','padding','baryons','use_2LPT'),
        'cosmology':('Omega_m','Omega_b','Omega_L','H0','A_s','n_s','transfer','ztarget','mixed_2lpt','lpt2_displacement','lpt2_velocity','lpt2_aH_km_s_Mpc')}.items():
        for key in keys:
            if linear_bundle is not None and (section,key) in {('setup','zstart'),('cosmology','ztarget'),('cosmology','lpt2_displacement'),('cosmology','lpt2_velocity'),('cosmology','lpt2_aH_km_s_Mpc')}:
                continue
            if cfg.get(section,key)!=new_cfg.get(section,key):
                raise ValueError('Restaged case changes native physical option: '+section+'/'+key)
    for name in ('music/staging.py','zoom.py'):
        path=(_CORE_ROOT/name)
        pins[str(path)]=file_sha256(path)
    for path,digest in pins.items():
        if file_sha256(path)!=digest:
            raise ValueError('Original case input changed during restaging')
    # Keep provenance outside the sealed staging directory: extra files inside
    # that directory would invalidate its exact asset inventory.
    report={'schema':'native-campaign-case-restaging-v1','status':'STAGED','source_hashes':pins,
        'staging':{'path':str(target/'staging.json'),'sha256':file_sha256(target/'staging.json')},
        'configuration':{'path':str(target/'zoom_template.conf'),'sha256':file_sha256(target/'zoom_template.conf')},
        'phase_policy':'Original explicit seed hierarchy and exact frozen files unchanged',
        'bounded_source_settings':settings,'source_archive_max_mib':source_archive_max_mib,
        'regenerated_epoch_bundle':linear_bundle,
        'scope':'Current public physical-input restaging with source retention and explicit bounded correction. No native fields, IC generation, numerical handoff or evolution.',
        'launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
    atomic_json(output,report)
    return report


def preflight_generated_flow_evolution(row,plan,context):
    """Inspect physical answers behind the registered runtime flow producer.

    Returns False for other answer-materializer routes, which retain their
    own preflight. It never creates future answers or substitutes static ones.
    """
    parameters=row['parameters']
    if parameters.get('command')!='gadget':
        return False
    from core.workflow import path_at
    from core.research.capabilities import validate_native_answers
    from core.zoom import _configuration
    arguments=parameters.get('arguments',[])
    def option(argv,flag):
        if argv.count(flag)!=1 or argv.index(flag)+1>=len(argv):
            raise ValueError('Generated flow needs exactly one '+flag)
        return argv[argv.index(flag)+1]
    answers=option(arguments,'--answers')
    if not isinstance(answers,dict) or 'stage' not in answers:
        return False
    stages={stage['stage_id']:stage for stage in plan['stages']}
    producer=stages.get(answers['stage'])
    if not producer or producer['kind']!='public_command' or producer['parameters'].get('command')!='prepare-generated-flow':
        return False
    if answers!={'stage':producer['stage_id'],'artifact':'prepared/answers.json'}:
        raise ValueError('Generated evolution needs the actual prepared flow answers artifact')
    argv=producer['parameters']['arguments']
    if option(argv,'--output')!='{generation}/prepared':
        raise ValueError('Generated flow output differs from consumed answers')
    generation=option(argv,'--generation-report')
    if (not isinstance(generation,dict) or set(generation)!={'stage','artifact'} or generation['artifact']!='generated/generation.json'
            or option(arguments,'--native-generation-report')!=generation
            or option(arguments,'--existing-ic')!={'stage':generation['stage'],'artifact':'generated/ics_gadget.hdf5'}):
        raise ValueError('Flow preparation and evolution must consume the same exact generated IC/report')
    if generation['stage'] not in producer.get('depends_on',[]) or producer['stage_id'] not in row.get('depends_on',[]):
        raise ValueError('Generated flow dependencies must precede evolution')
    generator=stages.get(generation['stage'])
    if not generator or generator['kind']!='public_command' or generator['parameters'].get('command')!='zoom':
        raise ValueError('Generated flow requires the declared native zoom producer')
    def frozen_input(value):
        if not isinstance(value,dict) or set(value)!={'input'}:
            raise ValueError('Inspectable frozen generated-flow input required')
        path=path_at(context.base,value['input'])
        if not path.is_file() or path.stat().st_size>16*1024**2:
            raise ValueError('Generated-flow metadata missing or oversized')
        return path
    configuration=frozen_input(generator['parameters']['arguments'][0])
    cfg=_configuration(configuration)
    if not cfg.getboolean('output','retain_source_hierarchy',fallback=False):
        raise ValueError('Generated flow requires retained source hierarchy')
    request=json.loads(frozen_input(option(argv,'--request')).read_text())
    generated_flow_bindings(request,levels=range(cfg.getint('setup','levelmin'),cfg.getint('setup','levelmax')+1))
    template=json.loads(Path(request['answers']['path']).read_text())
    template['particles_per_dimension']=2**cfg.getint('setup','levelmax')
    resources=row.get('resources',context.resources)
    physical=validate_native_answers(template,resources,row.get('canonical_run_constraints'))
    # isclose preserves the absolute tolerance and rejects NaN comparisons.
    if (template.get('physics')!='hydro' or not math.isclose(float(template['box_mpc_h']),cfg.getfloat('setup','boxlength'),rel_tol=0.,abs_tol=1e-12)
            or not math.isclose(float(physical['simulation']['start_redshift']),cfg.getfloat('setup','zstart'),rel_tol=0.,abs_tol=1e-12)):
        raise ValueError('Generated zoom and evolution physical settings disagree')
    return True


def generated_flow_bindings(request,*,levels=None):
    from core.music.flow import validate_original_flow_launch_request, original_flow_model_bindings
    required={'schema','answers','boundary_adapter_build','modes','parent_conditioning','solve','response_workspace'}
    if (not isinstance(request,dict) or set(request)-{'seeded_covariance','noise_model_verification','noise_model_generation'}!=required
            or ('noise_model_verification' in request)==('noise_model_generation' in request)
            or request['schema']!='native-generated-flow-preparation-request-v1'):
        raise ValueError('Complete generated-flow preparation request required')
    answers=request['answers']
    if not isinstance(answers,dict) or set(answers)!={'path','sha256'}:
        raise ValueError('Exact physical answer-template binding required')
    path=Path(answers['path']).resolve()
    if file_sha256(path)!=answers['sha256']:
        raise ValueError('Generated-flow answer template changed')
    if 'noise_model_generation' in request:
        from core.music.boundary import verify_boundary_adapter_build
        controls=request['noise_model_generation']
        if (not isinstance(controls,dict) or set(controls)!={'schema','gsl_library','byte_order','memory_gib'}
                or controls['schema']!='native-generated-local-phase-model-v1' or controls['byte_order'] not in ('<','>')
                or type(controls['memory_gib']) not in (int,float) or not math.isfinite(controls['memory_gib']) or controls['memory_gib']<=0
                or request['parent_conditioning']!='fixed_prefix' or 'seeded_covariance' not in request):
            raise ValueError('Explicit generated local phase verifier, fixed-prefix covariance and workspace required')
        library=controls['gsl_library']
        if not isinstance(library,dict) or set(library)!={'path','sha256'} or file_sha256(library['path'])!=library['sha256']:
            raise ValueError('Generated phase verification GSL binding changed')
        build=Path(request['boundary_adapter_build']['path']).resolve()
        if file_sha256(build)!=request['boundary_adapter_build']['sha256']:
            raise ValueError('Generated phase boundary adapter binding changed')
        binary,record=verify_boundary_adapter_build(build)
        # Reuse the full launch contract validator with an explicit future
        # proof identity. It is never used as runtime evidence or published.
        launch={k:v for k,v in request.items() if k not in ('schema','answers','noise_model_generation')}
        launch.update(schema='native-original-flow-launch-request-v1',noise_model_verification={'path':'runtime-phase-model.json','sha256':'0'*64},
            windows=[{'id':'intake-only','particle_types':[0],'refinement_levels':None}])
        validate_original_flow_launch_request(launch)
        pins={str(path):answers['sha256'],str(Path(library['path']).resolve()):library['sha256'],str(build):file_sha256(build),str(binary):file_sha256(binary),**record['source_sha256']}
        for name in (
            'music/staging.py',
            'music/phases.py',
            'music/seeded.py',
            'music/replay.py',
            'music/flow.py',
        ):
            source=(_CORE_ROOT / name).resolve()
            pins[str(source)]=file_sha256(source)
        return pins
    # Window coverage is derived from the actual mesh at runtime. The model,
    # disk allocations and modes can still be validated before generation.
    launch={k:v for k,v in request.items() if k not in ('schema','answers')}
    launch.update(schema='native-original-flow-launch-request-v1',windows=[{'id':'intake-only','particle_types':[0],'refinement_levels':None}])
    validate_original_flow_launch_request(launch)
    return {str(path):answers['sha256'],**original_flow_model_bindings(launch,levels=levels)}


def prepare_generated_flow_answers(generation_report,request,destination):
    """Build answers from this generation's retained IC and source archive.

    If requested, verify the local phase model against the actual generated
    cache coordinates first. A failed verification preserves its diagnostics
    and prevents physical answer preparation. The public answer path stays
    stable whether verification is supplied or performed here.
    """
    from core.zoom import verify_generated_zoom,_configuration
    from core.music.flow import prepare_original_flow_answers
    pins=generated_flow_bindings(request)
    target=Path(destination).resolve()
    if target.exists():
        raise FileExistsError('Preserve previous generated-flow preparation')
    generation=Path(generation_report).resolve()
    verify_generated_zoom(generation)
    base=generation.parent
    configuration=base/'music.conf'
    mesh=base/'planning/native-mesh.json'
    cfg=_configuration(configuration)
    if not cfg.getboolean('output','retain_source_hierarchy',fallback=False):
        raise ValueError('Generated zoom must retain its original source hierarchy for the mandatory flow gate')
    if not (base/'ics_gadget.hdf5').is_file() or not (base/'source-hierarchy.bin').is_file():
        raise ValueError('Actual generated IC and original source hierarchy required before preparing evolution')
    actual={k:v for k,v in request.items() if k!='schema'}
    if 'noise_model_generation' in actual:
        import h5py
        from core.music.phases import publish_seeded_noise_model
        controls=actual.pop('noise_model_generation')
        target.mkdir(parents=True,exist_ok=False)
        with h5py.File(base/'ics_gadget.hdf5') as f:
            shift=f['Header'].attrs['Music_output_coordinate_shift_box_fraction'].tolist()
        model=target/'generated-local-phase-model.json'
        phase_request={'schema':'native-local-seeded-model-request-v1',
            'configuration':{'path':str(configuration),'sha256':file_sha256(configuration)},
            'mesh':{'path':str(mesh),'sha256':file_sha256(mesh)},'noise_directory':str(base),
            'output_shift_box_fraction':shift,'byte_order':controls['byte_order'],'gsl_library':controls['gsl_library']}
        atomic_json(target/'generated-local-phase-request.json',phase_request)
        measured=publish_seeded_noise_model(phase_request,model,memory_gib=controls['memory_gib'])
        if measured['status']!='PASS':
            raise ValueError('Actual generated local phase model verification failed; preserved measured evidence')
        actual['noise_model_verification']={'path':str(model),'sha256':file_sha256(model)}
        preparation_target=target/'flow'
    else:preparation_target=target
    actual.update(schema='native-flow-answer-preparation-request-v1',configuration={'path':str(configuration),'sha256':file_sha256(configuration)},
        mesh={'path':str(mesh),'sha256':file_sha256(mesh)})
    prepared=prepare_original_flow_answers(actual,preparation_target)
    if preparation_target!=target:
        # The campaign's declared artifact remains prepared/answers.json;
        # the nested preparation receipt retains its original answer binding.
        atomic_json(target/'answers.json',json.loads(Path(prepared['answers']['path']).read_text()))
        prepared['answers']={'path':str(target/'answers.json'),'sha256':file_sha256(target/'answers.json')}
    pins.update({str(generation):file_sha256(generation),str(configuration):file_sha256(configuration),str(mesh):file_sha256(mesh)})
    pins.update(prepared['source_hashes'])
    pins[prepared['answers']['path']]=prepared['answers']['sha256']
    for name in ('music/staging.py','music/flow.py','zoom.py'):
        path=(_CORE_ROOT / name)
        pins[str(path)]=file_sha256(path)
    proof={'schema':'native-generated-flow-preparation-v1','status':'PREPARED','generation_report':{'path':str(generation),'sha256':pins[str(generation)]},
        'answers':prepared['answers'],'launch_request':prepared['launch_request'],'populations_by_level':prepared['populations_by_level'],
        'source_hashes':pins,'scope':'Runtime answer preparation from the exact verified generated zoom and actual mesh; original flow/technical gates still run before evolution',
        'science_acceptance':'NOT_ACCEPTED','launch_authorized':False}
    for path,digest in pins.items():
        if file_sha256(path)!=digest:
            raise ValueError('Generated-flow dependency changed during preparation')
    output=target/'generated-preparation.json'
    atomic_json(output,proof)
    return proof
