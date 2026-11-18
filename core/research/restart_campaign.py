"""Optional native restart campaigns: freeze, pause/resume, compare.

Templates are fixed before either branch runs. Comparisons bind the actual
native executable, ICs, checkpoint and complete source/integrator/hydro/global
state. These strict numerical reproducibility diagnostics are separate from
physical conservation/convergence tolerances and from scientific acceptance.
"""
import copy
import json
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from core.foundation import atomic_json, file_sha256

_SOURCE_ROOT = Path(__file__).resolve().parents[2]


def lifecycle_tolerance_catalog(features):
    from core.stars import SSP_BIRTH_EVENT_COLUMNS
    from core.black_holes import _BH_EVENT_FIELDS
    from core.research.restarts import INTEGER_FIELDS
    catalog = set()
    commits = {'native_time', 'native_step_begin', 'unit_mass_g', 'unit_time_s',
               'unit_velocity_cm_s', 'unit_length_cm', 'hubble_param', 'factor_sn', 'seed'}
    families = {}
    if 'STARFORMATION' in features:
        families['stellar_birth'] = {'stellar_birth': set(SSP_BIRTH_EVENT_COLUMNS)}
    if 'STELLAR_EVOLUTION' in features:
        families['stellar_return'] = {'stellar_return': {
            'native_time', 'age_before_years', 'age_after_years', 'mass_before_msun',
            'mass_after_msun', 'velocity_before_km_s', 'velocity_after_km_s',
            'internal_energy_before_cm2_s2', 'internal_energy_after_cm2_s2',
            'elements_before_msun', 'elements_after_msun', 'injected_energy_erg'}}
    if 'BLACK_HOLES' in features:
        events = {}
        for kind, native in _BH_EVENT_FIELDS.items():
            fields = set(native) | {'native_time'}
            # Mirror the public ingestion unit-column naming contract. Both
            # native and converted columns are compared; string extras unused.
            for key in native:
                prefix = key[:-7]
                if key.endswith('_native'):
                    suffix = ('_msun_km_s' if 'momentum' in key else
                              '_km_s' if 'velocity' in key else
                              '_cm2_s2' if key in {'internal_energy_before_native', 'internal_energy_after_native'} else
                              '_erg' if 'energy' in key else '_msun' if 'mass' in key else None)
                    if suffix:
                        fields.add(prefix + suffix)
                elif key.endswith('_g') and 'mass' in key:
                    fields.add(key[:-2] + '_msun')
                elif key.endswith('_cm_s') and 'velocity' in key:
                    fields.add(key[:-5] + '_km_s')
                elif key == 'radiated_momentum_g_cm_s':
                    fields.add('radiated_momentum_msun_km_s')
            events[kind] = fields
        families['black_holes'] = events
    for family, events in families.items():
        catalog.update(family + '.commits.' + field for field in commits - INTEGER_FIELDS)
        for kind, fields in events.items():
            catalog.update(family + '.events.' + kind + '.' + field for field in fields - INTEGER_FIELDS)
    return sorted(families), {key: {'absolute': 0., 'relative': 0.} for key in sorted(catalog)}


def freeze_restart_template(cfg, *, workspace, snapshot, identity_namespace,
                            maximum_particles, maximum_events, maximum_commits, memory_gib):
    """Author a complete comparison before either run, using native defaults.

    Explicit census/event bounds must be chosen by the campaign preparer.
    Unsupported native array expressions fail rather than guessing dimensions.
    """
    from core.evolution import gadget_config_text
    from core.research.restarts import _compiled_condition, _layout
    from core.research.release import prepare_native_release_comparison
    for value in (maximum_particles, maximum_events, maximum_commits):
        if type(value) is not int or value < 1:
            raise ValueError('Positive exact restart allocation bounds required')
    root = _SOURCE_ROOT
    workspace = Path(workspace).resolve()
    options = dict((line.split('=', 1) + ['1'])[:2] for line in gadget_config_text(cfg).splitlines() if line)
    features = sorted(options)
    if 'RESEARCH_CHECKPOINT_EXERCISE' not in features:
        raise ValueError('Restart exercise configuration must explicitly enable its native diagnostic')
    zero = lambda names: {name: {'absolute': 0., 'relative': 0.} for name in names}
    request = {'schema': 'native-restart-comparison-request-v1', 'maximum_particles': maximum_particles,
               'maximum_source_fields': 1, 'memory_gib': memory_gib, 'species': list(range(6)),
               'identity_namespace': identity_namespace, 'source_fields': [],
               'tolerances': zero(('position_proper_kpc', 'velocity_peculiar_kms', 'mass_msun'))}
    schemas = {'source_capsule': 'native-restart-source-capsule-layout-v1',
               'integrator_state': 'native-restart-integrator-capsule-layout-v1',
               'hydro_particle_state': 'native-restart-hydro-particle-layout-v1',
               'global_state': 'native-restart-global-state-layout-v1'}
    nt = int(options.get('NTYPES', 6))
    ns = int(options.get('NSOFTCLASSES', nt))
    nh = int(options.get('NSOFTCLASSES_HYDRO', 64)) if 'ADAPTIVE_HYDRO_SOFTENING' in features else 0
    outputs = len(cfg['simulation']['snapshot_redshift_list'])
    lengths = {'TIMEBINS': 60 if 'ENLARGE_DYNAMIC_RANGE_IN_TIME' in features else 29,
               'NTYPES': nt, 'NSOFTCLASSES': ns, 'NSOFTCLASSES+NSOFTCLASSES_HYDRO': ns + nh,
               'NSOFTCLASSES+NSOFTCLASSES_HYDRO+2': ns + nh + 2, 'OutputListLength': outputs}
    path = root / 'configs/runtime/restart_layouts.json'
    reference = {'path': str(path), 'sha256': file_sha256(path)}
    for scope, schema in schemas.items():
        layout, _ = _layout(reference, schema)
        settings = {'layout': dict(reference)}
        if scope == 'source_capsule':
            active = layout['float_columns']
        else:
            settings['compiled_features'] = features
            if scope == 'integrator_state':
                active = [row['name'] for row in layout['float_columns'] if _compiled_condition(row['condition'], set(features))]
                settings['coordinate_bits'] = 128 if 'POSITIONS_IN_128BIT' in features else 64 if 'POSITIONS_IN_64BIT' in features else 32
            elif scope == 'hydro_particle_state':
                active = [row['dataset'] for row in layout['fields'] if
                          (row['feature'] is None or row['feature'] in features) and row['memory_type'] != 'MEM_UINT64']
            else:
                active = [row['name'] for row in layout['fields'] if row['type'] == 'float' and _compiled_condition(row['condition'], set(features))]
                active += [row['name'] for row in layout['arrays'] if row['type'] == 'float']
                settings.update(maximum_attribute_bytes=16*1024**2,
                                array_lengths={row['name']: lengths[row['length']] for row in layout['arrays']})
            if scope in {'integrator_state', 'hydro_particle_state'}:
                settings['read_chunk_particles'] = min(16384, maximum_particles)
        settings['float_tolerances'] = zero(active)
        request[scope] = settings
    for side in ('uninterrupted', 'resumed'):
        request[side] = {'run_directory': str(workspace / side / 'run'), 'snapshot': snapshot}
    families, tolerances = lifecycle_tolerance_catalog(features)
    if families:
        request['lifecycle_journals'] = {'kinds': families, 'directories': {
            side: {kind: str(workspace / side / 'run/data') for kind in families} for side in ('uninterrupted', 'resumed')},
            'maximum_events_per_family': maximum_events, 'maximum_commits_per_family': maximum_commits,
            'maximum_tolerance_fields': max(1, len(tolerances)), 'float_tolerances': tolerances}
    return prepare_native_release_comparison(request, {}, release_root=root, workspace=workspace)['request']


def validate_campaign_restart_comparison(parameters):
    if type(parameters['snapshot']) is not int or parameters['snapshot']<0:
        raise ValueError('Exact saved restart-comparison snapshot required')
    if type(parameters['memory_gib']) not in (int,float) or not math.isfinite(parameters['memory_gib']) or parameters['memory_gib']<=0:
        raise ValueError('Explicit restart-comparison memory allocation required')
    if parameters['uninterrupted_run']==parameters['resumed_run']:
        raise ValueError('Distinct uninterrupted and resumed runs required')


def campaign_restart_comparison_runner(parameters,context):
    from core.research.restarts import compare_native_restart
    from core.research.release import prepare_native_release_comparison, publish_native_release_comparison
    validate_campaign_restart_comparison(parameters)
    memory=parameters['memory_gib']
    if memory>context.resources['memory_gib']:
        raise MemoryError('Restart comparison exceeds workflow memory')
    template_path=Path(parameters['template_path']).resolve()
    if file_sha256(template_path)!=parameters['template_sha256'] or template_path.stat().st_size>min(16*1024**2,memory*1024**3/16):
        raise ValueError('Frozen bounded complete restart comparison template changed')
    template=json.loads(template_path.read_text())
    bindings={str(template_path):parameters['template_sha256'],str(Path(__file__).resolve()):file_sha256(__file__)}
    if template.get('memory_gib')!=memory:
        raise ValueError('Template and workflow comparison allocations differ')
    runs=[Path(parameters[name]).resolve() for name in ('uninterrupted_run','resumed_run')]
    for run in runs:
        bindings[str(run/'manifest.json')]=file_sha256(run/'manifest.json')
    manifests=[json.loads((run/'manifest.json').read_text()) for run in runs]
    for manifest in manifests:
        if manifest.get('status')!='SIMULATION_COMPLETE':
            raise ValueError('Restart comparison requires completed actual native runs')
    if manifests[0].get('resume_attempts'):
        raise ValueError('Uninterrupted reference contains native continuation attempts')
    attempts=manifests[1].get('resume_attempts',[])
    if not attempts or attempts[-1].get('status')!='COMPLETE':
        raise ValueError('Candidate lacks an actual completed native resume')
    handoffs=[manifest['gadget_handoff'] for manifest in manifests]
    if (handoffs[0]['executable']['sha256']!=handoffs[1]['executable']['sha256']
            or sorted(r['sha256'] for r in handoffs[0]['initial_conditions'])!=sorted(r['sha256'] for r in handoffs[1]['initial_conditions'])):
        raise ValueError('Restart reference/candidate change native binary or original IC identities')
    for run,handoff in zip(runs,handoffs):
        if file_sha256(handoff['executable']['path'])!=handoff['executable']['sha256']:
            raise ValueError('Restart executable changed')
        for name in ('parameter_file','compiled_options_file','snapshot_schedule_file'):
            ref=handoff[name]
            if file_sha256(ref['path'])!=ref['sha256']:
                raise ValueError('Restart immutable native handoff input changed')
    if handoffs[0]['compiled_options_file']['sha256']!=handoffs[1]['compiled_options_file']['sha256']:
        raise ValueError('Restart physical/diagnostic native compiled options differ')
    if handoffs[0]['snapshot_schedule_file']['sha256']!=handoffs[1]['snapshot_schedule_file']['sha256']:
        raise ValueError('Restart output schedules differ')
    from core.evolution import parameter_values
    original,candidate=[parameter_values(h['parameter_file']['path']) for h in handoffs]
    if 'RESEARCH_CHECKPOINT_EXERCISE' in Path(handoffs[0]['compiled_options_file']['path']).read_text().splitlines():
        if original.get('NativeControlPauseStep')!='0' or int(candidate.get('NativeControlPauseStep','0'))<=0:
            raise ValueError('Restart exercise must compare unpaused reference against declared positive candidate pause step')
        original.pop('NativeControlPauseStep')
        candidate.pop('NativeControlPauseStep')
    for name in ('OutputDir','OutputListFilename','TimeLimitCPU','MaxMemSize','InitCondFile'):
        original.pop(name,None)
        candidate.pop(name,None)
    if original!=candidate:
        raise ValueError('Restart reference/candidate change numerical or physical parameters')
    text=Path(handoffs[0]['compiled_options_file']['path']).read_text()
    features={line.split('#',1)[0].strip().split('=',1)[0] for line in text.splitlines() if line.split('#',1)[0].strip()}
    for scope in ('integrator_state','hydro_particle_state','global_state'):
        if set(template[scope]['compiled_features'])!=features:
            raise ValueError('Restart comparison template differs from actual compiled state features')
    bits=128 if 'POSITIONS_IN_128BIT' in features else 64 if 'POSITIONS_IN_64BIT' in features else 32
    if template['integrator_state']['coordinate_bits']!=bits:
        raise ValueError('Restart integer coordinate width differs from actual native configuration')
    marker=runs[1]/'checkpoint-published.json'
    resume=runs[1].parent/'invocation.json'
    baseline=runs[0].parent/'invocation.json'
    for name in (marker,resume,baseline):
        if not name.is_file():
            raise ValueError('Actual checkpoint and workflow invocation artifacts required: '+str(name))
    if json.loads(resume.read_text()).get('input_hashes',{}).get(str(marker))!=file_sha256(marker):
        raise ValueError('Actual native resume invocation does not bind the verified checkpoint marker')
    archive=runs[1].parent/'checkpoint-archives'/file_sha256(marker)/'archive.json'
    retained=json.loads(archive.read_text())
    checkpoint=json.loads(marker.read_text())
    if (retained.get('schema')!='campaign-interrupted-checkpoint-archive-v1' or retained.get('checkpoint_marker_sha256')!=file_sha256(marker)
            or not checkpoint.get('files_sha256') or retained.get('original_files_sha256')!=checkpoint['files_sha256']):
        raise ValueError('Actual interrupted checkpoint payload archive is incomplete')
    bindings[str(archive)]=file_sha256(archive)
    for source,copy in retained['retained_copies'].items():
        if copy['sha256']!=checkpoint['files_sha256'].get(source) or file_sha256(copy['path'])!=copy['sha256']:
            raise ValueError('Actual interrupted checkpoint retained payload changed')
        bindings[copy['path']]=copy['sha256']
    if set(retained['retained_copies'])!=set(checkpoint['files_sha256']):
        raise ValueError('Interrupted checkpoint archive omits native shards/sidecars')
    for side,run in zip(('uninterrupted','resumed'),runs):
        template[side]={'run_directory':str(run),'snapshot':parameters['snapshot']}
        for kind in (template.get('lifecycle_journals') or {}).get('kinds',[]):
            template['lifecycle_journals']['directories'][side][kind]=str(run/'data')
    # This existing preparer requires every active typed layout member and
    # compiled lifecycle family. It does not infer or loosen tolerances.
    source_root=_SOURCE_ROOT
    workspace=Path(os.path.commonpath([str(context.generation.resolve()),*(str(run) for run in runs)]))
    prepared=prepare_native_release_comparison(template,{},release_root=source_root,workspace=workspace)
    request_path=context.generation/'restart-request.json'
    publish_native_release_comparison(prepared,request_path,marker=marker,resume_invocation=resume,uninterrupted_invocation=baseline)
    result=compare_native_restart(json.loads(request_path.read_text()),context.generation/'comparison')
    for path,digest in bindings.items():
        if file_sha256(path)!=digest:
            raise ValueError('Campaign restart input changed during comparison')
        result['source_hashes'][path]=digest
    atomic_json(context.generation/'restart-comparison.json',result)
    if result['status']!='PASS':
        raise ValueError('Actual native restart state comparison failed; retained measurements identify the failing fields')
    return result


def validate_restart_exercise(parameters):
    if type(parameters['pause_step']) is not int or not 1<=parameters['pause_step']<=1000000:
        raise ValueError('Explicit positive bounded restart exercise pause step required')
    if type(parameters['snapshot']) is not int or parameters['snapshot']<0:
        raise ValueError('Exact restart comparison snapshot required')
    for name in ('memory_gib','wall_seconds'):
        if type(parameters[name]) not in (int,float) or not math.isfinite(parameters[name]) or parameters[name]<=0:
            raise ValueError('Positive restart exercise '+name+' required')


def prepare_exercise_comparison(template,cfg,*,workspace,snapshot,memory_gib):
    """Validate all selected state scopes before any native branch starts."""
    from core.evolution import gadget_config_text
    from core.research.release import prepare_native_release_comparison
    template=copy.deepcopy(template)
    workspace=Path(workspace).resolve()
    features={line.split('=',1)[0] for line in gadget_config_text(cfg).splitlines() if line}
    for scope in ('integrator_state','hydro_particle_state','global_state'):
        if set(template[scope]['compiled_features'])!=features:
            raise ValueError('Frozen comparison template does not match actual selected native physics')
    if template['memory_gib']!=memory_gib:
        raise ValueError('Restart exercise comparison memory differs from frozen template')
    for side in ('uninterrupted','resumed'):
        template[side]={'run_directory':str(workspace/side/'run'),'snapshot':snapshot}
        for kind in (template.get('lifecycle_journals') or {}).get('kinds',[]):
            template['lifecycle_journals']['directories'][side][kind]=str(workspace/side/'run/data')
    prepare_native_release_comparison(template,{},release_root=_SOURCE_ROOT,workspace=workspace)
    return template


def restart_exercise_runner(parameters,context):
    from core.research.capabilities import validate_native_answers
    from core.evolution import gadget_config_text,parameter_values
    from core.research.checkpoints import evolution_checkpoint_policy
    from core.workflow import public_command_runner,command_runner
    validate_restart_exercise(parameters)
    if parameters['memory_gib']>context.resources['memory_gib']:
        raise MemoryError('Restart exercise exceeds stage memory')
    root=_SOURCE_ROOT
    generation=context.generation.resolve()
    answers_path=Path(parameters['answers_path']).resolve()
    template_path=Path(parameters['template_path']).resolve()
    for path in (answers_path,template_path):
        if path.stat().st_size>min(16*1024**2,parameters['memory_gib']*1024**3/16):
            raise MemoryError('Restart exercise metadata exceeds bounded intake')
    if file_sha256(template_path)!=parameters['template_sha256']:
        raise ValueError('Frozen native restart exercise template changed')
    original=json.loads(answers_path.read_text())
    template=json.loads(template_path.read_text())
    if 'restart_exercise_pause_step' in original:
        raise ValueError('Base campaign answers must retain ordinary evolution semantics')
    configurations=[]
    paths=[]
    for name,step in (('uninterrupted',0),('resumed',parameters['pause_step'])):
        answers=copy.deepcopy(original)
        answers.update(restart_source_capsule=True,restart_exercise_pause_step=step)
        cfg=validate_native_answers(answers,context.resources)
        configurations.append(cfg)
        path=generation/(name+'.answers.json')
        atomic_json(path,answers)
        paths.append(path)
    if gadget_config_text(configurations[0])!=gadget_config_text(configurations[1]):
        raise ValueError('Restart exercise changes compiled physics between branches')
    template=prepare_exercise_comparison(template,configurations[0],workspace=generation,
        snapshot=parameters['snapshot'],memory_gib=parameters['memory_gib'])
    frozen=generation/'frozen-comparison-template.json'
    atomic_json(frozen,template)
    intake={str(path):file_sha256(path) for path in (answers_path,template_path,Path(parameters['existing_ic']),Path(parameters['generation_report']))}
    atomic_json(generation/'exercise-intake.json',{'source_hashes':intake,'diagnostic_changes':{'restart_source_capsule':True,
        'reference_pause_step':0,'candidate_pause_step':parameters['pause_step']},'physical_settings_policy':'All original answers retained except these explicit restart diagnostics; no timestep/force/physics/tolerance change'})
    contexts=[]
    for name in ('uninterrupted','resumed'):
        folder=generation/name
        folder.mkdir()
        contexts.append(SimpleNamespace(generation=folder,base=getattr(context,'base',root),resources=context.resources,resume_checkpoint=None))
    def arguments(path,run):
        return ['--answers',str(path),'--existing-ic',str(parameters['existing_ic']),'--native-generation-report',str(parameters['generation_report']),
            '--run-dir',str(run),'--evolution-only']
    public_command_runner({'command':'gadget','wall_seconds':parameters['wall_seconds'],'arguments':arguments(paths[0],contexts[0].generation/'run')},contexts[0])
    try:
        public_command_runner({'command':'gadget','wall_seconds':parameters['wall_seconds'],'arguments':arguments(paths[1],contexts[1].generation/'run')},contexts[1])
    except subprocess.CalledProcessError:
        # Incomplete output collection is expected only after the actual
        # native pause has published a complete interior checkpoint.
        pass
    else:raise ValueError('Restart candidate completed without the declared interior pause')
    candidate=contexts[1].generation
    marker=candidate/'run/checkpoint-published.json'
    if not marker.is_file():
        raise ValueError('Restart exercise did not produce a verified native checkpoint; original failure logs retained')
    checkpoint=json.loads(marker.read_text())
    params=parameter_values(candidate/'run/input/param.txt')
    clock=checkpoint['checkpoint_integrity']['clock_value']
    if not float(params['TimeBegin'])<clock<float(params['TimeMax']):
        raise ValueError('Restart exercise checkpoint must be strictly inside the native timeline')
    contexts[1].resume_checkpoint={'marker_path':str(marker),'policy':evolution_checkpoint_policy()}
    command_runner({'argv':evolution_checkpoint_policy()['argv'],'wall_seconds':parameters['wall_seconds']},contexts[1])
    for path,digest in intake.items():
        if file_sha256(path)!=digest:
            raise ValueError('Original restart exercise inputs changed during execution')
    result=campaign_restart_comparison_runner({'uninterrupted_run':str(contexts[0].generation/'run'),'resumed_run':str(candidate/'run'),
        'snapshot':parameters['snapshot'],'memory_gib':parameters['memory_gib'],'template_path':str(frozen),'template_sha256':file_sha256(frozen)},context)
    return result
