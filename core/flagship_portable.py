"""Resolve portable source assets through existing builders and admission gates."""
from pathlib import Path
import copy
import ctypes.util
import json
import os
import subprocess
import sys
from .foundation import atomic_json, file_sha256
from core.native import run_native_command


def source_assets(value,root,*,json_directory=None,pins=None):
    """Resolve shipped relative inputs without changing their recorded bytes."""
    root=Path(root).resolve()
    if isinstance(value,list):return [source_assets(item,root,json_directory=json_directory,pins=pins) for item in value]
    if isinstance(value,dict):
        if 'project_directory' in value:
            if set(value)!={'project_directory'}:raise ValueError('Exact project directory reference required')
            relative=Path(value['project_directory']);path=(root/relative).resolve()
            if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root) or not path.is_dir():raise ValueError('Portable directory escapes or is absent from the extracted project')
            return str(path)
        if 'project_json' in value:
            if set(value)!={'project_json','sha256'}:raise ValueError('Exact portable JSON template/hash required')
            ref=source_assets({'project_asset':value['project_json'],'sha256':value['sha256']},root,pins=pins)
            if json_directory is None:return copy.deepcopy(value)
            source=Path(ref['path'])
            if source.stat().st_size>16*1024**2:raise MemoryError('Portable JSON template exceeds metadata bound')
            contents=source_assets(json.loads(source.read_text()),root,json_directory=json_directory,pins=pins)
            target=Path(json_directory)/ (value['sha256']+'.json');target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():atomic_json(target,contents)
            elif json.loads(target.read_text())!=contents:raise ValueError('Conflicting portable JSON template resolution')
            return {'path':str(target.resolve()),'sha256':file_sha256(target)}
        if 'project_asset' in value:
            if set(value)!={'project_asset','sha256'}:raise ValueError('Exact portable source asset/hash required')
            relative=Path(value['project_asset']);path=(root/relative).resolve()
            if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root) or not path.is_file():raise ValueError('Portable asset escapes the extracted source project')
            if file_sha256(path)!=value['sha256']:raise ValueError('Portable source asset changed')
            if pins is not None:pins[str(path)]=value['sha256']
            return {'path':str(path),'sha256':value['sha256']}
        return {key:source_assets(item,root,json_directory=json_directory,pins=pins) for key,item in value.items()}
    return value


def gsl_binary():
    """Resolve the installed GSL shared library and bind its actual bytes."""
    candidate=ctypes.util.find_library('gsl');paths=[]
    if candidate:
        path=Path(candidate)
        if path.is_absolute():paths.append(path)
        else:
            for key in ('LD_LIBRARY_PATH','DYLD_LIBRARY_PATH'):
                paths.extend(Path(folder)/candidate for folder in os.environ.get(key,'').split(os.pathsep) if folder)
    probe=subprocess.run(['pkg-config','--variable=libdir','gsl'],capture_output=True,text=True,check=True)
    directory=Path(probe.stdout.strip()).resolve()
    if candidate:paths.append(directory/Path(candidate).name)
    paths.extend([directory/'libgsl.dylib',directory/'libgsl.so'])
    for path in paths:
        path=path.resolve()
        if path.is_file():return {'path':str(path),'sha256':file_sha256(path)}
    raise FileNotFoundError('Installed GSL shared-library file required for original frozen-phase replay')


def native_admission(binary,monofonic,destination,*,reuse,wall_seconds):
    from .zoom import validate_native_music_regression,require_mixed_lpt_validation
    target=Path(destination).resolve();root=Path(__file__).resolve().parents[1]
    if reuse is not None:
        if set(reuse)!={'path','sha256'} or file_sha256(reuse['path'])!=reuse['sha256']:raise ValueError('Requested native admission input changed')
        path=Path(reuse['path']).resolve()
    else:
        target.mkdir(parents=True,exist_ok=False)
        commands=[('transfer',[sys.executable,str(root/'commands.py'),'native-music-transfer',str(target/'transfer'),
            '--prepare','--nested','--binary',str(binary)]),
            ('mixed',[sys.executable,str(root/'commands.py'),'native-music-mixed2lpt','--output',str(target/'mixed'),
                '--binary',str(binary),'--monofonic',str(monofonic),'--legacy-proof',str(target/'transfer/native-transfer-suite.json')])]
        for name,command in commands:
            with (target/(name+'.log')).open('wb') as stream:
                run_native_command(command,cwd=root,receipt_path=target/(name+'.invocation.json'),inputs=[Path(binary),Path(monofonic),root/'core/research/generator_controls.py',root/'commands.py'],
                    timeout=wall_seconds,stdout=stream,stderr=stream)
        path=target/'mixed/native-transfer-suite.json'
    proof=validate_native_music_regression(binary,path);require_mixed_lpt_validation(proof,binary)
    return {'path':str(path),'sha256':file_sha256(path)}


def materialize_portable_profile(request,workspace,*,overrides=None):
    from .flagship_launch import _build_engines
    from core.music.boundary import prepare_boundary_adapter
    from core.music.phases import publish_frozen_noise_model
    from .data import verify_numeric_audit,numeric_audit_source_bindings
    from .workflow import positive_number
    required={'schema','template','preparation'}
    local_seeded=request.get('schema')=='flagship-portable-local-seeded-request-v1'
    if set(request)!=required or request['schema'] not in {'flagship-portable-connected-request-v1','flagship-portable-local-seeded-request-v1'}:raise ValueError('Complete portable connected or local-seeded request required')
    root=Path(__file__).resolve().parents[1];pins={};value=source_assets(copy.deepcopy(request),root,pins=pins)
    settings=value['preparation'];required={'engines','native_admission','boundary_adapter','phase_model','flow'}
    if set(settings)!=required:raise ValueError('Complete portable engine/phase/flow preparation required')
    template=value['template'];resources={**template['resources'],**(overrides or {})}
    if template.get('schema')!='flagship-connected-profile-request-v1':raise ValueError('Original connected profile template required')
    for key in ('memory_gib','disk_gib','mpi_ranks','omp_threads','build_jobs','maximum_stage_count'):
        positive_number(resources[key],key,key in {'mpi_ranks','omp_threads','build_jobs','maximum_stage_count'})
    admission=settings['native_admission']
    if set(admission)!={'reuse','wall_seconds'}:raise ValueError('Explicit native admission reuse selection and wall bound required')
    positive_number(admission['wall_seconds'],'native admission wall seconds')
    phase=settings['phase_model']
    if set(phase)!={'fields','cube_size','byte_order','positive_sign','memory_gib'}:raise ValueError('Complete retained full frozen phase-model settings required')
    positive_number(phase['memory_gib'],'phase replay memory')
    if phase['memory_gib']>resources['memory_gib']:raise MemoryError('Phase replay exceeds selected allocation')
    fields={str(row['level']):row for row in phase['fields']}
    frozen=template['zoom']['staging'].get('frozen_phase_prefix',{})
    if local_seeded:
        parent_level=template['parent']['staging']['parent_level'];key=str(parent_level)
        if (set(frozen)!={key} or key not in fields or fields[key]['file']!=frozen[key]
                or template['parent']['staging']['frozen_parent_noise']!=frozen[key]
                or max(map(int,fields))!=parent_level
                or fields[key]['seed']!=template['parent']['staging']['seed']
                or fields[key]['seed']!=template['parent']['answers']['seed']
                or fields[key]['seed']!=template['zoom']['staging']['seeds_by_level'].get(key)):
            raise ValueError('Local refinement requires the last independently verified complete frozen field as its exact parent prefix')
        flow_controls=settings['flow']
        if (flow_controls.get('parent_conditioning')!='fixed_prefix'
                or 'seeded_covariance' not in flow_controls
                or set(flow_controls)!={'modes','parent_conditioning','solve','response_workspace','seeded_covariance'}):
            raise ValueError('Local-seeded profile requires original fixed-prefix covariance and explicit response disk allocations')
    else:
        if set(frozen)!=set(fields) or any(fields[level]['file']!=ref for level,ref in frozen.items()):raise ValueError('Portable model must verify the exact complete declared frozen hierarchy')
        if max(template['zoom']['staging']['refinement_levels'])>max(map(int,fields)):raise ValueError('This full-frozen profile requires declared fields through its requested refinement; use the separate original local-seeded strategy for higher levels')
    target=Path(workspace).expanduser().resolve()
    if target.exists():raise FileExistsError('Preserve previous portable preparation; choose a fresh output')
    target.mkdir(parents=True)
    try:
        value=source_assets(value,root,json_directory=target/'resolved-assets',pins=pins)
        settings=value['preparation'];template=value['template'];phase=settings['phase_model']
        engines=_build_engines(settings['engines'],resources,pins);binaries={item['engine']:item['binary'] for item in engines}
        if not {'class','music2','monofonic'}<=set(binaries):raise ValueError('Portable preparation needs actual CLASS, MUSIC and monofonIC builds')
        proof=native_admission(binaries['music2']['path'],binaries['monofonic']['path'],target/'native-admission',
            reuse=admission['reuse'],wall_seconds=admission['wall_seconds'])
        boundary=prepare_boundary_adapter(target/'boundary-adapter',**settings['boundary_adapter'])
        gsl=gsl_binary();model=target/'frozen-phase-model.json'
        result=publish_frozen_noise_model({'schema':'native-full-frozen-model-request-v1',**{key:phase[key] for key in phase if key!='memory_gib'},'gsl_library':gsl},model,memory_gib=phase['memory_gib'])
        if result['status']!='PASS':raise ValueError('Original frozen phase-model replay rejected the selected files; evidence retained')
        pins.update(numeric_audit_source_bindings(verify_numeric_audit(model)))
        pins[str(model)]=file_sha256(model)
        pins[proof['path']]=proof['sha256']
        pins.update(boundary['source_sha256'])
        pins[str(target/'boundary-adapter/receipt.json')]=file_sha256(target/'boundary-adapter/receipt.json')
        pins[boundary['binary']['path']]=boundary['binary']['sha256']
        answers=target/'combined-answer-template.json';atomic_json(answers,template['history']['combined_answers'])
        flow={'schema':'native-generated-flow-preparation-request-v1','answers':{'path':str(answers),'sha256':file_sha256(answers)},
            'boundary_adapter_build':{'path':str(target/'boundary-adapter/receipt.json'),'sha256':file_sha256(target/'boundary-adapter/receipt.json')},**settings['flow']}
        if local_seeded:
            # The parent hierarchy was independently replayed above. Each fine
            # patch must pass the existing generated local-phase verifier after
            # its actual caches/mesh exist, followed by the original flow gate.
            flow['noise_model_generation']={'schema':'native-generated-local-phase-model-v1',
                'gsl_library':gsl,'byte_order':phase['byte_order'],'memory_gib':phase['memory_gib']}
        else:flow['noise_model_verification']={'path':str(model),'sha256':file_sha256(model)}
        from core.music.staging import generated_flow_bindings
        generated_flow_bindings(flow)
        flow_path=target/'flow-preparation.json';atomic_json(flow_path,flow)
        def resolve(node):
            if isinstance(node,list):return [resolve(item) for item in node]
            if isinstance(node,dict):
                if set(node)=={'native_engine_binary'}:
                    if node['native_engine_binary'] not in binaries:raise ValueError('Portable template requests an engine that was not built')
                    return copy.deepcopy(binaries[node['native_engine_binary']])
                if node=={'native_generator_admission':True}:return copy.deepcopy(proof)
                if node=={'prepared_flow_template':True}:return {'path':str(flow_path),'sha256':file_sha256(flow_path)}
                return {key:resolve(item) for key,item in node.items()}
            return node
        connected=resolve(template);connected['resources']=resources
        # Existing build receipts are checked again by the original launch
        # preparer; no historical receipt or absolute source path is rewritten.
        connected['build_engines']=settings['engines']
        path=target/'connected-request.json';atomic_json(path,connected)
        atomic_json(target/'portable-preparation.json',{'schema':'flagship-portable-preparation-v1','execution_status':'PREPARED',
            'source_hashes':pins,'engines':engines,'native_admission':proof,'boundary_adapter_binary':boundary['binary'],
            'phase_model':{'path':str(model),'sha256':file_sha256(model)},'local_seeded_refinement':local_seeded,'connected_request':{'path':str(path),'sha256':file_sha256(path)},
            'full_miniature':'NOT_EXECUTED','science_acceptance':'NOT_ACCEPTED'})
        return connected,path
    except BaseException as error:
        atomic_json(target/'failure.json',{'status':'FAILED','type':type(error).__name__,'message':str(error),'science_acceptance':'NOT_ACCEPTED'})
        raise
