"""Prepare and inspect an explicitly requested portable-release control campaign.

These helpers check copied source, native builds, interrupted/resumed state and
fresh Blender loading. They run only through their explicit public commands.
Passing these technical controls does not establish physical convergence.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path, PurePosixPath
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
import zipfile
import uuid
from core.foundation import atomic_json, atomic_text, file_sha256, canonical_digest
from core.native import run_native_command, native_run_lock


# Clean release
PROFILES = {'collisionless', 'hydro_stellar_bh', 'sidm'}
PROFILE_FEATURES = {
    'collisionless': {'RESEARCH_RESTART_STATE_CAPSULE'},
    'hydro_stellar_bh': {'COOLING','STARFORMATION','STELLAR_EVOLUTION','BLACK_HOLES','RESEARCH_RESTART_STATE_CAPSULE'},
    'sidm': {'RESEARCH_SIDM_REFERENCE','RESEARCH_RESTART_STATE_CAPSULE'},
}
CONTROL_KINDS = {'public_workflow', 'interrupt_resume', 'blender_fresh_load', 'patch_reproduction'}


def extract_to_clean_unrelated_directory(archive_path, destination):
    from core.release import verify_portable_zip
    inventory = verify_portable_zip(Path(archive_path))
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Clean release destination must be new and unrelated to the source checkout')
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode) or not relative.parts or relative.parts[0] != 'research2':
                raise ValueError('Only regular source release members allowed')
            path = destination/Path(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, path.open('xb') as output:
                shutil.copyfileobj(source, output, 1024*1024)
                output.flush()
                os.fsync(output.fileno())
            path.chmod(0o755 if mode & 0o111 else 0o644)
    root = destination/'research2'
    for relative, expected in inventory['files_sha256'].items():
        if file_sha256(root/relative) != expected:
            raise ValueError('Extracted release payload mismatch')
    return root, inventory


def _argv(recipe, replacements):
    import re
    known = {'python','suite','output','source_inventory_hash','profile','profile_output','native_binary'}
    result = []
    for arg in recipe['argv']:
        if not isinstance(arg, str) or not arg:
            raise ValueError('Explicit argv strings required')
        if set(re.findall(r'\{([^{}]+)\}',arg))-known:
            raise ValueError('Release recipe contains an unknown path token')
        for key, value in replacements.items():
            arg = arg.replace('{'+key+'}', str(value))
        if replacements and re.search(r'\{[^{}]+\}',arg):
            raise ValueError('Release recipe retains an unresolved execution path token')
        result.append(arg)
    return result


def _seconds(value,label):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(label+' requires finite positive seconds')
    return value


def _retained_path(value,replacements,cwd,workspace):
    path = Path(_argv({'argv':[value]},replacements)[0])
    if not path.is_absolute():
        path = Path(cwd)/path
    path = path.resolve()
    if not path.is_relative_to(Path(workspace).resolve()):
        raise ValueError('Release control artifact escaped the unrelated workspace')
    return path


def _run(recipe, replacements, cwd, output, *, inputs=()):
    with output.with_suffix('.log').open('wb') as log:
        run_native_command(_argv(recipe, replacements), cwd=cwd, receipt_path=output,
                           inputs=inputs, timeout=_seconds(recipe['wall_seconds'],'Native control wall limit'), stdout=log, stderr=log)


def reproduce_native_patches(recipe, release_root, destination):
    """Apply shipped patches to exact recorded upstream tree; compare bytes."""
    destination = Path(destination)
    input_hashes={}
    if 'bundle_manifest' in recipe:
        manifest_path=_retained_path(recipe['bundle_manifest'],{},release_root,release_root)
        input_hashes[str(manifest_path)]=file_sha256(manifest_path)
        recipe=json.loads(manifest_path.read_text())
        if recipe.get('schema')!='native-source-patch-bundle-v1':
            raise ValueError('Complete native source patch bundle required')
    required=('pristine_source','pristine_hashes','patches','compare_files')
    if any(not recipe.get(key) for key in required):
        raise ValueError('Native patch reproduction cannot use empty baseline, patch or source inventories')
    from core.release import tree_hashes
    pristine = _retained_path(recipe['pristine_source'],{},release_root,release_root)
    if destination.exists():
        raise FileExistsError('Patch reproduction target must be new')
    if recipe.get('schema')=='native-source-patch-bundle-v1' and tree_hashes(pristine)!=recipe['pristine_hashes']:
        raise ValueError('Pristine native source tree differs from the complete pinned baseline')
    shutil.copytree(pristine, destination, ignore=shutil.ignore_patterns('.git', 'build', 'build_cache'))
    for relative, expected in recipe['pristine_hashes'].items():
        if file_sha256(_retained_path(relative,{},destination,destination)) != expected:
            raise ValueError('Pristine native baseline changed')
        input_hashes[str(pristine/relative)]=expected
    patches = []
    for item in recipe['patches']:
        patch = _retained_path(item['path'],{},release_root,release_root)
        if type(item.get('strip',1)) is not int or item.get('strip',1) < 0:
            raise ValueError('Native patch strip count must be a nonnegative integer')
        if file_sha256(patch) != item['sha256']:
            raise ValueError('Shipped patch changed')
        input_hashes[str(patch)]=item['sha256']
        run_native_command(['git', 'apply', '-p'+str(item.get('strip', 1)), str(patch)],
            cwd=destination, receipt_path=destination.parent/(destination.name+'-patch-'+str(len(patches))+'.json'), inputs=(patch,), timeout=300)
        patches.append(item)
    rows = []
    for relative, shipped in recipe['compare_files'].items():
        patched = _retained_path(relative,{},destination,destination)
        target = _retained_path(shipped,{},release_root,release_root)
        identical = file_sha256(patched) == file_sha256(target)
        rows.append({'patched': relative, 'shipped': shipped, 'identical': identical,
                     'sha256': file_sha256(target)})
    if recipe.get('schema')=='native-source-patch-bundle-v1':
        if tree_hashes(destination)!=recipe['shipped_hashes']:
            raise ValueError('Patched native source tree has missing, extra or changed portable files')
        shipped_root=_retained_path(recipe['shipped_source'],{},release_root,release_root)
        if tree_hashes(shipped_root)!=recipe['shipped_hashes']:
            raise ValueError('Shipped native tree changed after complete patch authoring')
    if not rows or not all(q['identical'] for q in rows):
        raise ValueError('Native patches do not reproduce shipped implementation bytes')
    measurement=destination.parent/(destination.name+'-source-comparison.json')
    atomic_json(measurement,{'schema':'native-source-reproduction-measurements-v1','files':rows,
        'input_hashes':input_hashes,'patched_tree':str(destination.resolve())})
    return {'status': 'PASS','execution_status':'COMPLETE','patches': patches, 'files': rows,
        'source_hashes':{str(Path(__file__).resolve()):file_sha256(__file__),
            **{str((Path(release_root)/recipe['compare_files'][row['patched']]).resolve()):row['sha256'] for row in rows}},
        'input_hashes':input_hashes,
        'files_sha256':{str(measurement.resolve()):file_sha256(measurement),
            **{str((destination/row['patched']).resolve()):row['sha256'] for row in rows}}}


def interrupt_resume_and_verify(recipe, replacements, cwd, output, *, workspace=None, native_binary=None):
    """Interrupt a live owned run only after actual coherent native publication."""
    from core.native import invocation_record, stop_owned_process_groups, verify_invocation_inputs, native_owned_fds
    workspace = Path(workspace or cwd).resolve()
    command = _argv(recipe['initial'],replacements)
    marker = _retained_path(recipe['checkpoint_marker'],replacements,cwd,workspace)
    proof_path = _retained_path(recipe['comparison_receipt'],replacements,cwd,workspace)
    if marker.exists() or proof_path.exists():
        raise FileExistsError('Fresh checkpoint/comparison artifacts already exist')
    timeout = _seconds(recipe['checkpoint_wait_seconds'],'Checkpoint publication timeout')
    grace = _seconds(recipe.get('stop_grace_seconds',2.),'Native stop grace')
    inputs = (native_binary,) if native_binary is not None else ()
    record = invocation_record(command,cwd,inputs=inputs)
    record.update(status='WAITING_FOR_CHECKPOINT',execution_status='RUNNING')
    atomic_json(output,record)
    process = None
    started = time.monotonic()
    last_publication_error = None
    try:
        prepared_comparison=None
        if recipe.get('native_comparison'):
            
            prepared_comparison=prepare_native_release_comparison(recipe['native_comparison'],replacements,
                release_root=cwd,workspace=workspace)
            if proof_path.name!='restart-comparison.json':
                raise ValueError('Built-in native comparison publishes restart-comparison.json in its new generation')
            _run(recipe['uninterrupted'],replacements,cwd,output.with_suffix('.uninterrupted.json'),inputs=inputs)
            record['comparison_template_hash']=prepared_comparison['template_hash']
            record['prepared_comparison_hash']=prepared_comparison['prepared_request_hash']
            record['uninterrupted_invocation']={'path':str(output.with_suffix('.uninterrupted.json')),
                'sha256':file_sha256(output.with_suffix('.uninterrupted.json'))}
            atomic_json(output,record)
        started = time.monotonic()
        with output.with_suffix('.initial.log').open('wb') as log:
            process = subprocess.Popen([record['resolved_executable'], *command[1:]],cwd=cwd,start_new_session=True,
                pass_fds=native_owned_fds(),stdout=log,stderr=log)
            record['pid'] = process.pid
            atomic_json(output,record)
            while not marker.is_file():
                if process.poll() is not None:
                    raise RuntimeError('Native run exited before complete checkpoint publication')
                if time.monotonic()-started > timeout:
                    raise TimeoutError('Native checkpoint publication timeout: '+str(last_publication_error))
                if recipe.get('checkpoint_run_dir') is not None:
                    from core.research.checkpoints import publish_native_checkpoint_marker
                    run = _retained_path(recipe['checkpoint_run_dir'],replacements,cwd,workspace)
                    if native_binary is None:
                        raise ValueError('Native checkpoint observer requires this profile executable')
                    try:
                        publish_native_checkpoint_marker(run,marker,mpi_ranks=recipe['checkpoint_mpi_ranks'],
                            executable_sha256=file_sha256(native_binary),workspace=workspace,
                            minimum_generation=recipe.get('minimum_checkpoint_generation',1))
                    except (OSError,ValueError,KeyError) as error:
                        last_publication_error = str(error)
                if not marker.is_file():
                    time.sleep(.1)
            marker_hash = file_sha256(marker)
            checkpoint = json.loads(marker.read_text())
            seal = marker.with_name(marker.name+'.sha256')
            if not seal.is_file() or seal.read_text().strip() != marker_hash:
                raise ValueError('Native checkpoint publication marker is unsealed or changed')
            if checkpoint.get('schema') != 'native-checkpoint-publication-v1' or checkpoint.get('status') != 'COMPLETE' or not checkpoint.get('files_sha256'):
                raise ValueError('Checkpoint marker does not identify complete native payloads')
            if native_binary is not None and checkpoint.get('executable_sha256') != file_sha256(native_binary):
                raise ValueError('Checkpoint marker belongs to another native profile executable')
            checkpoint_sources = dict(checkpoint.get('source_hashes',{}))
            if not checkpoint_sources:
                raise ValueError('Checkpoint marker lacks exact original input bindings')
            from core.data import verify_native_checkpoints
            run = _retained_path(checkpoint['run_dir'],{},cwd,workspace)
            actual = verify_native_checkpoints(run/'data/restartfiles',checkpoint['mpi_ranks'],require_receipts=True)
            if actual != checkpoint.get('checkpoint_integrity'):
                raise ValueError('Published marker does not match the actual coherent native checkpoint')
            checkpoint_files = {}
            for name,expected in checkpoint['files_sha256'].items():
                path = _retained_path(name,{},marker.parent,workspace)
                if file_sha256(path) != expected:
                    raise ValueError('Checkpoint shard changed before interruption')
                checkpoint_files[str(path)] = expected
            if file_sha256(marker) != marker_hash:
                raise ValueError('Checkpoint marker changed during intake')
            if process.poll() is not None:
                raise RuntimeError('Checkpoint producer completed naturally; no live interruption was performed')
            record.update(status='INTERRUPTING_NATIVE_RUN',checkpoint_marker_sha256=marker_hash,
                checkpoint_files_sha256=checkpoint_files)
            atomic_json(output,record)
            stop_owned_process_groups([process],grace_seconds=grace)
            record['interrupted_returncode'] = process.returncode
            for bindings in (checkpoint_files, checkpoint_sources):
                for name,expected in bindings.items():
                    if file_sha256(name) != expected:
                        raise ValueError('Interrupted checkpoint/input differs from published generation')
            # Later native checkpoints may replace these live filenames. Retain the
            # actual interrupted bytes independently before resuming the process.
            from core.foundation import atomic_copy
            archive=output.with_suffix('.checkpoint-archive')
            archive.mkdir()
            copies={}
            for index,(name,expected) in enumerate(sorted(checkpoint_files.items())):
                retained=archive/('shard_'+format(index,'06d'))
                atomic_copy(name,retained)
                if file_sha256(retained)!=expected:
                    raise ValueError('Interrupted checkpoint changed during retained archive publication')
                copies[name]={'path':str(retained.resolve()),'sha256':expected}
            archive_manifest=archive/'checkpoint-archive.json'
            atomic_json(archive_manifest,{'schema':'interrupted-native-checkpoint-archive-v1',
                'checkpoint_marker_sha256':marker_hash,'original_files_sha256':checkpoint_files,
                'retained_copies':copies,'checkpoint_integrity':checkpoint['checkpoint_integrity']})
            record['checkpoint_archive']={'path':str(archive_manifest.resolve()),'sha256':file_sha256(archive_manifest)}
            verify_invocation_inputs(record)
        _run(recipe['resume'],replacements,cwd,output.with_suffix('.resume.json'),inputs=(marker,*inputs))
        if prepared_comparison is not None:
            
            request_path=output.with_suffix('.comparison-request.json')
            publish_native_release_comparison(prepared_comparison,request_path,marker=marker,
                resume_invocation=output.with_suffix('.resume.json'),
                uninterrupted_invocation=output.with_suffix('.uninterrupted.json'))
            comparator={'argv':[str(replacements['python']),str(Path(cwd)/'commands.py'),
                'restart-comparison','--request',str(request_path),'--output',str(proof_path.parent)],
                'wall_seconds':recipe['comparison_wall_seconds']}
            _run(comparator,{},cwd,output.with_suffix('.comparison.json'),
                inputs=(marker,request_path,output.with_suffix('.resume.json'),output.with_suffix('.uninterrupted.json'),*inputs,
                    *prepared_comparison['layout_sources']))
        else:
            _run(recipe['comparison'],replacements,cwd,output.with_suffix('.comparison.json'),inputs=(marker,*inputs))
        proof = json.loads(proof_path.read_text())
        if proof.get('status') != 'PASS' or proof.get('checkpoint_marker_sha256') != marker_hash or proof.get('execution_status') != 'COMPLETE':
            raise ValueError('Restart comparison does not bind the executed checkpoint')
        if not proof.get('files_sha256'):
            raise ValueError('Restart comparison lacks actual retained measured outputs')
        for name,expected in proof['files_sha256'].items():
            path = _retained_path(name,{},proof_path.parent,workspace)
            if file_sha256(path) != expected:
                raise ValueError('Restart comparison measurement changed')
        record.update(status='PASS',execution_status='COMPLETE',
            checkpoint_marker_sha256=marker_hash,comparison_receipt={'path':str(proof_path),'sha256':file_sha256(proof_path)},
            source_hashes=proof.get('source_hashes',{}),
            files_sha256={**proof['files_sha256'],str(proof_path):file_sha256(proof_path),
                str(archive_manifest.resolve()):file_sha256(archive_manifest),
                **{entry['path']:entry['sha256'] for entry in copies.values()},
                str(output.with_suffix('.resume.json')):file_sha256(output.with_suffix('.resume.json'))},
            publication_observer_error=last_publication_error)
        atomic_json(output,record)
        return record
    except BaseException as error:
        record.update(status='FAILED',execution_status='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED',
            failure={'type':type(error).__name__,'message':str(error)},publication_observer_error=last_publication_error)
        atomic_json(output,record)
        raise
    finally:
        stop_owned_process_groups([process])


def verify_control_evidence(path, *, kind, profile, source_inventory, release_root=None):
    value = json.loads(Path(path).read_text())
    if value.get('schema') == 'blender-fresh-load-v1':
        for label in ('source_hashes','input_hashes','files_sha256'):
            if not value.get(label):
                raise ValueError('Fresh Blender control lacks retained actual '+label)
            for name,digest in value[label].items():
                artifact = Path(name)
                if not artifact.is_absolute():
                    artifact = Path(path).parent/artifact
                if file_sha256(artifact) != digest:
                    raise ValueError('Fresh Blender control source/scene/invocation changed')
                if label == 'source_hashes' and release_root is not None:
                    root = Path(release_root).resolve()
                    if not artifact.resolve().is_relative_to(root) or source_inventory['files_sha256'].get(artifact.resolve().relative_to(root).as_posix()) != digest:
                        raise ValueError('Fresh Blender control executed source outside the exact extracted release')
        scene = Path(value['scene_artifact']).resolve()
        if value['input_hashes'].get(str(scene)) != value['blend_sha256'] or file_sha256(scene) != value['blend_sha256']:
            raise ValueError('Fresh Blender control does not bind its exact retained scene')
    if kind == 'public_workflow' and value.get('schema') == 'research-workflow-receipt-v1':
        from core.workflow import verify_completed_receipt
        if value.get('execution_status') != 'COMPLETE' or not value.get('stages'):
            raise ValueError('Clean public workflow is incomplete')
        products = {}
        sources={};inputs={str(Path(path).resolve()):file_sha256(path)}
        for stage in value['stages'].values():
            receipt_path = Path(stage['generation'])/'stage-receipt.json'
            if file_sha256(receipt_path) != stage['receipt_sha256']:
                raise ValueError('Clean workflow stage receipt changed')
            receipt = verify_completed_receipt(receipt_path)
            sources.update(receipt['source_hashes']);inputs.update(receipt['input_hashes'])
            inputs[str(receipt_path.resolve())]=file_sha256(receipt_path)
            if release_root is not None:
                root = Path(release_root).resolve()
                for source, checksum in receipt['source_hashes'].items():
                    source_path = Path(source).resolve()
                    if not source_path.is_relative_to(root):
                        raise ValueError('Clean public control executed first-party code outside extracted release')
                    relative = source_path.relative_to(root).as_posix()
                    if source_inventory['files_sha256'].get(relative) != checksum:
                        raise ValueError('Clean public control source differs from shipped release inventory')
            products.update({str(Path(stage['generation'])/p): h for p,h in receipt['files_sha256'].items()})
        return {'status': 'PASS', 'execution_status': 'COMPLETE', 'control_kind': kind, 'profile': profile,
                'source_inventory_hash': canonical_digest(source_inventory['files_sha256']),
                'source_hashes':sources,'input_hashes':inputs,
                'files_sha256': products, 'scope': 'clean public execution/product integrity; scientific acceptance separate'}
    if kind == 'blender_fresh_load' and value.get('schema') == 'blender-fresh-load-v1':
        if value.get('status') != 'PASS' or value.get('execution_status') != 'COMPLETE' or not value.get('epochs'):
            raise ValueError('Fresh scene lacks passing actual epoch controls')
        return {**value, 'control_kind': kind, 'profile': profile,
                'source_inventory_hash': canonical_digest(source_inventory['files_sha256']),
                'evidence_sha256': file_sha256(path)}
    if value.get('status') != 'PASS' or value.get('execution_status') != 'COMPLETE':
        raise ValueError('Clean release control needs executed passing measurements')
    if value.get('control_kind') != kind or value.get('profile') != profile:
        raise ValueError('Release control identity/profile mismatch')
    if value.get('source_inventory_hash') != canonical_digest(source_inventory['files_sha256']):
        raise ValueError('Release control source inventory mismatch')
    if not value.get('files_sha256'):
        raise ValueError('Control has no retained numerical/visual outputs')
    for filename, checksum in value['files_sha256'].items():
        p = Path(filename)
        if not p.is_absolute():
            p = Path(path).parent/p
        if file_sha256(p) != checksum:
            raise ValueError('Release control measurement changed')
    return value


def _execute_clean_release(archive, destination, request):
    if platform.system() != 'Linux':
        raise RuntimeError('Clean native release acceptance must run on Linux')
    if request.get('schema') != 'clean-linux-release-campaign-v1':
        raise ValueError('Expected clean-linux-release-campaign-v1')
    profiles = request['profiles']
    if set(profiles) != PROFILES:
        raise ValueError('Collisionless, hydro/stellar/BH and SIDM Linux build profiles are mandatory')
    # Check the whole configured matrix before any extraction or native build.
    for name,profile in profiles.items():
        if not PROFILE_FEATURES[name]<=set(profile['required_features']):
            raise ValueError('Clean release feature declarations omit mandatory native physics/state diagnostics: '+name)
        controls = profile['controls']
        kinds = [control['kind'] for control in controls]
        if len(kinds) != len(set(kinds)):
            raise ValueError('Native release profiles must not repeat control-kind decisions')
        if not CONTROL_KINDS <= {control['kind'] for control in controls}:
            raise ValueError('Every native profile needs all four release control kinds')
        _argv(profile['build'],{})
        _seconds(profile['build']['wall_seconds'],'Native build wall limit')
        for control in controls:
            kind = control['kind']
            if kind not in CONTROL_KINDS:
                raise ValueError('Unknown clean release control kind')
            recipe = control['recipe']
            if kind == 'interrupt_resume':
                branches=('initial','resume','uninterrupted') if recipe.get('native_comparison') else ('initial','resume','comparison')
                for branch in branches:
                    _argv(recipe[branch],{})
                    _seconds(recipe[branch]['wall_seconds'],'Restart recipe wall limit')
                if recipe.get('native_comparison'):
                    if 'comparison' in recipe:
                        raise ValueError('Built-in native restart comparison cannot also specify an unrelated external comparator')
                    _seconds(recipe['comparison_wall_seconds'],'Native restart comparison wall limit')
                _seconds(recipe['checkpoint_wait_seconds'],'Checkpoint timeout')
                if recipe.get('checkpoint_run_dir') is not None and (type(recipe.get('checkpoint_mpi_ranks')) is not int or recipe['checkpoint_mpi_ranks'] < 1):
                    raise ValueError('Checkpoint observer needs the exact original MPI rank count')
            elif kind != 'patch_reproduction':
                _argv(recipe,{})
                _seconds(recipe['wall_seconds'],'Release control wall limit')
    destination = Path(destination).resolve()
    with native_run_lock(destination, operation='clean Linux release campaign'):
        root,inventory = extract_to_clean_unrelated_directory(archive,destination)
        runner_sources = {}
        observer_root = Path(__file__).resolve().parents[2]
        for relative in (
            'core/research/release.py', 'core/research/checkpoints.py',
            'core/research/capabilities.py', 'core/research/generator_controls.py', 'core/release.py',
            'core/data.py', 'core/foundation.py', 'core/cosmology.py', 'core/native.py',
            'core/resources.py', 'core/__init__.py', 'configs/core_source_layout.json',
        ):
            digest = file_sha256(observer_root/relative)
            if inventory['files_sha256'].get(relative) != digest:
                raise ValueError('Release observer/reader implementation differs from shipped archive: '+relative)
            runner_sources[relative] = digest
        output = destination/'release-results'
        output.mkdir()
        records = []
        progress = {'schema':'clean-release-progress-v1','execution_status':'RUNNING',
                    'request_hash':canonical_digest(request),'profiles':{},'active_profile':None,'active_control':None}
        atomic_json(output/'progress.json',progress)
        replacements = {'python': sys.executable, 'suite': root, 'output': output,
                        'source_inventory_hash': canonical_digest(inventory['files_sha256'])}
        # Freeze the entire configured comparison matrix before any profile build/evolution.
        
        for name,profile in profiles.items():
            bindings={**replacements,'profile':name,'profile_output':output/name}
            for control in profile['controls']:
                if control['kind']=='interrupt_resume' and control['recipe'].get('native_comparison'):
                    prepare_native_release_comparison(control['recipe']['native_comparison'],bindings,
                        release_root=root,workspace=destination)
        for name, profile in profiles.items():
            kinds = {q['kind'] for q in profile['controls']}
            if not CONTROL_KINDS <= kinds:
                raise ValueError('Every native profile requires workflows, restart, fresh Blender and patch controls')
            folder = output/name
            folder.mkdir()
            progress['active_profile'] = name
            progress['active_control'] = 'build'
            progress['profiles'][name] = {'execution_status':'RUNNING','completed_controls':[]}
            atomic_json(output/'progress.json',progress)
            replacements.update(profile=name, profile_output=folder)
            _run(profile['build'], replacements, root, folder/'build-invocation.json')
            binary = Path(_argv({'argv':[profile['binary']]},replacements)[0])
            if not binary.is_absolute():
                binary = root/binary
            binary = binary.resolve()
            if not binary.is_relative_to(destination):
                raise ValueError('Clean native build binary lies outside the unrelated release workspace')
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise ValueError('Native feature profile did not produce an executable')
            signature_path = Path(_argv({'argv':[profile['build_signature']]},replacements)[0])
            if not signature_path.is_absolute():
                signature_path = root/signature_path
            signature_path = signature_path.resolve()
            if not signature_path.is_relative_to(destination):
                raise ValueError('Clean native build signature lies outside the release workspace')
            signature = json.loads(signature_path.read_text())
            config=signature.get('compiled_features',str(signature.get('config','')).splitlines())
            features={str(line).split('#',1)[0].strip().split('=',1)[0]
                for line in config if str(line).split('#',1)[0].strip()}
            if not set(profile['required_features']) <= features:
                raise ValueError('Native release binary profile lacks required compiled features')
            build_receipt_path=_retained_path(profile.get('build_receipt',str(signature_path.with_name('build_receipt.json'))),
                replacements,root,destination)
            build_receipt=json.loads(build_receipt_path.read_text())
            if build_receipt.get('schema')!='native-build-cache-v1' or build_receipt.get('status')!='COMPLETE' or \
                    build_receipt.get('signature_sha256')!=canonical_digest(signature) or \
                    build_receipt.get('binary_sha256')!=file_sha256(binary):
                raise ValueError('Native release binary lacks its current completed public build receipt')
            replacements['native_binary'] = binary
            controls = []
            for index, control in enumerate(profile['controls']):
                receipt = folder/f'control_{index:03d}.json'
                progress['active_control'] = {'index':index,'kind':control['kind']}
                atomic_json(output/'progress.json',progress)
                if control['kind'] == 'patch_reproduction':
                    result = reproduce_native_patches(control['recipe'], root, folder/f'patched_{index}')
                elif control['kind'] == 'interrupt_resume':
                    result = interrupt_resume_and_verify(control['recipe'],replacements,root,receipt,workspace=destination,native_binary=binary)
                else:
                    _run(control['recipe'], replacements, root, receipt, inputs=(binary,))
                    proof_path = Path(_argv({'argv': [control['evidence']]}, replacements)[0])
                    if not proof_path.is_absolute():
                        proof_path = root/proof_path
                    if not proof_path.resolve().is_relative_to(destination):
                        raise ValueError('Clean release control evidence must be retained in this unrelated workspace')
                    result = verify_control_evidence(proof_path, kind=control['kind'], profile=name, source_inventory=inventory, release_root=root)
                retained = folder/f'control_{index:03d}-result.json'
                if result.get('status')!='PASS' or result.get('execution_status')!='COMPLETE':
                    raise ValueError('Completed release control cannot publish incomplete passing evidence')
                atomic_json(retained,{**result,'control_kind':control['kind'],'profile':name,
                    'source_inventory_hash':canonical_digest(inventory['files_sha256'])})
                controls.append({'kind':control['kind'],'path':str(retained),'sha256':file_sha256(retained)})
                progress['profiles'][name]['completed_controls'].append({'index':index,'kind':control['kind'],
                    'path':str(retained),'sha256':file_sha256(retained)})
                atomic_json(output/'progress.json',progress)
            records.append({'profile': name, 'binary_sha256': file_sha256(binary),
                            'build_signature_sha256':file_sha256(signature_path),
                            'build_receipt_sha256':file_sha256(build_receipt_path),
                            'binary':{'path':str(binary),'sha256':file_sha256(binary)},
                            'build_signature':{'path':str(signature_path),'sha256':file_sha256(signature_path)},
                            'build_receipt':{'path':str(build_receipt_path),'sha256':file_sha256(build_receipt_path)},
                            'required_features':profile['required_features'],'controls': controls})
            progress['profiles'][name]['execution_status'] = 'COMPLETE'
            atomic_json(output/'progress.json',progress)
        report = {'schema': 'clean-linux-release-v1', 'platform': platform.system(),
            'status': 'PASS', 'execution_status': 'COMPLETE', 'archive_sha256': file_sha256(archive),
            'source_inventory_hash':canonical_digest(inventory['files_sha256']),'profiles':records,
            'release_inventory':{'path':str(root/'RELEASE.json'),'sha256':file_sha256(root/'RELEASE.json')},
            'release_runner_source_sha256':{str(root/name):digest for name,digest in runner_sources.items()},
            'source_hashes':{str(root/name):digest for name,digest in runner_sources.items()},
            'input_hashes':{str(Path(archive).resolve()):file_sha256(archive),
                str(root/'RELEASE.json'):file_sha256(root/'RELEASE.json')},
            'files_sha256':{control['path']:control['sha256'] for profile in records for control in profile['controls']},
            'science_acceptance': 'RELEASE_CONTROLS_ONLY'}
        atomic_json(output/'clean-release.json', report)
        from core.foundation import atomic_text
        atomic_text((output/'clean-release.json').with_suffix('.sha256'),file_sha256(output/'clean-release.json')+'\n')
        progress.update(execution_status='COMPLETE',active_profile=None,active_control=None,
                        result_sha256=file_sha256(output/'clean-release.json'))
        atomic_json(output/'progress.json',progress)
        return report



def verify_clean_release(archive, destination, request):
    """Retain precise failure/interruption lineage without overwriting evidence."""
    destination = Path(destination).resolve()
    started = time.time()
    try:
        request_hash = canonical_digest(request)
    except (ValueError,TypeError):
        request_hash = None
    try:
        return _execute_clean_release(archive,destination,request)
    except BaseException as exc:
        failure = {'schema':'clean-release-failure-v1',
            'execution_status':'INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'FAILED',
            'destination':str(destination),'request_hash':request_hash,
            'archive':str(Path(archive).resolve()),'started_unix':started,'finished_unix':time.time(),
            'failure':{'type':type(exc).__name__,'message':str(exc)},
            'source_hashes':{str(Path(__file__).resolve()):file_sha256(__file__)},
            'science_acceptance':'NOT_ESTABLISHED'}
        progress_path = destination/'release-results/progress.json'
        if progress_path.is_file():
            failure['progress'] = json.loads(progress_path.read_text())
            failure['progress_sha256'] = file_sha256(progress_path)
        # A sibling history survives extraction failure and does not mutate a
        # pre-existing extraction directory that this invocation never owned.
        history = destination.parent/'.release-failure-history'
        atomic_json(history/(destination.name+'-'+uuid.uuid4().hex+'.json'),failure)
        raise


def clean_release_main():
    import argparse
    parser = argparse.ArgumentParser(description='Execute clean Linux builds and retained release controls')
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_clean_release(args.archive, args.output, json.loads(args.campaign.read_text())), indent=2))


# Release restart requests
def expand_release_values(value, replacements):
    if isinstance(value, str):
        for name, replacement in replacements.items():
            value = value.replace('{'+name+'}', str(replacement))
        return value
    if isinstance(value, list):
        return [expand_release_values(item, replacements) for item in value]
    if isinstance(value, dict):
        return {name: expand_release_values(item, replacements) for name, item in value.items()}
    return value


def _tolerances(values, required):
    if not isinstance(values, dict) or set(values) != set(required):
        raise ValueError('Release comparison must freeze every active member tolerance before evolution')
    for limits in values.values():
        if not isinstance(limits, dict) or set(limits) != {'absolute', 'relative'} or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 for value in limits.values()):
            raise ValueError('Frozen finite nonnegative absolute/relative release tolerances required')


def prepare_native_release_comparison(template, replacements, *, release_root, workspace):
    """Resolve immutable profile settings before launching either native branch."""
    from core.research.restarts import _layout
    from core.research.restarts import _compiled_condition
    if not isinstance(template, dict) or template.get('schema') != 'native-restart-comparison-request-v1':
        raise ValueError('Built-in release comparison needs a complete frozen native request template')
    if any(key in template for key in ('checkpoint_marker','resume_invocation','uninterrupted_invocation')):
        raise ValueError('Actual release marker/resume bindings must be generated by the runner')
    request = expand_release_values(deepcopy(template), replacements)
    for key in ('maximum_particles', 'maximum_source_fields'):
        if type(request.get(key)) is not int or request[key] < 1:
            raise ValueError('Explicit positive native release comparison bounds required')
    memory = request.get('memory_gib')
    if isinstance(memory, bool) or not isinstance(memory, (int, float)) or not math.isfinite(memory) or memory <= 0:
        raise ValueError('Explicit positive native release comparison memory required')
    if request.get('species') != list(range(6)):
        raise ValueError('Clean release native restart comparisons must include all six native species')
    if not isinstance(request.get('identity_namespace'), str) or not request['identity_namespace']:
        raise ValueError('Explicit native release identity namespace required')
    if not isinstance(request.get('source_fields'), list) or len(request['source_fields']) > request['maximum_source_fields']:
        raise ValueError('Bounded native source-field list required')
    _tolerances(request['tolerances'], ('position_proper_kpc', 'velocity_peculiar_kms', 'mass_msun'))
    pins = {}
    layouts = {}
    schemas = {'source_capsule': 'native-restart-source-capsule-layout-v1',
               'integrator_state': 'native-restart-integrator-capsule-layout-v1',
               'hydro_particle_state': 'native-restart-hydro-particle-layout-v1',
               'global_state': 'native-restart-global-state-layout-v1'}
    root = Path(release_root).resolve()
    workspace = Path(workspace).resolve()
    for scope, schema in schemas.items():
        if not request.get(scope):
            raise ValueError('Clean native release comparison cannot omit '+scope)
        settings = request[scope]
        reference = settings['layout']
        path = Path(reference['path'])
        if not path.is_absolute():
            path = root/path
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError('Native comparison layout must come from the extracted release')
        reference['path'] = str(path)
        layout, bindings = _layout(reference, schema)
        layouts[scope] = layout
        pins.update(bindings)
        if scope == 'source_capsule':
            active = layout['float_columns']
        else:
            features = settings['compiled_features']
            if not isinstance(features, list) or any(not isinstance(name, str) for name in features) or len(set(features)) != len(features):
                raise ValueError('Distinct frozen feature names required for native release comparison')
            if scope == 'integrator_state':
                active = [row['name'] for row in layout['float_columns']
                          if _compiled_condition(row['condition'], set(features))]
                if type(settings['coordinate_bits']) is not int or settings['coordinate_bits'] not in {32, 64, 128}:
                    raise ValueError('Frozen supported native coordinate width required')
            elif scope == 'hydro_particle_state':
                active = [row['dataset'] for row in layout['fields']
                          if (row['feature'] is None or row['feature'] in features) and row['memory_type'] != 'MEM_UINT64']
            else:
                active = [row['name'] for row in layout['fields'] if row['type'] == 'float' and
                          _compiled_condition(row['condition'], set(features))]
                active += [row['name'] for row in layout['arrays'] if row['type'] == 'float']
                if type(settings.get('maximum_attribute_bytes')) is not int or settings['maximum_attribute_bytes'] < 1:
                    raise ValueError('Frozen native global attribute bound required')
                lengths = settings['array_lengths']
                if set(lengths) != {row['name'] for row in layout['arrays']} or any(type(value) is not int or value < 0 for value in lengths.values()):
                    raise ValueError('Freeze every native global table length before evolution')
        _tolerances(settings['float_tolerances'], active)
        if scope in {'integrator_state', 'hydro_particle_state'} and (
                type(settings.get('read_chunk_particles')) is not int or
                not 0 < settings['read_chunk_particles'] <= request['maximum_particles']):
            raise ValueError('Frozen bounded native diagnostic read chunk required')
    profiles = [request[name]['compiled_features'] for name in ('integrator_state', 'hydro_particle_state', 'global_state')]
    if any(set(features) != set(profiles[0]) for features in profiles[1:]):
        raise ValueError('Release state scopes disagree about their native build profile')
    features = set(profiles[0])
    if 'RESEARCH_RESTART_STATE_CAPSULE' not in features:
        raise ValueError('Native clean restart profile must compile its state diagnostics')
    for side in ('uninterrupted', 'resumed'):
        endpoint = request[side]
        path = Path(endpoint['run_directory'])
        if not path.is_absolute():
            path = root/path
        path = path.resolve()
        if not path.is_relative_to(workspace):
            raise ValueError('Native release endpoint must be retained in the clean workspace')
        endpoint['run_directory'] = str(path)
        if type(endpoint['snapshot']) is not int or endpoint['snapshot'] < 0:
            raise ValueError('Frozen actual native output index required')
    if request['uninterrupted']['run_directory'] == request['resumed']['run_directory']:
        raise ValueError('Uninterrupted and resumed controls require distinct native run directories')
    journals = {name for name, flag in (('stellar_birth', 'STARFORMATION'),
                ('stellar_return', 'STELLAR_EVOLUTION'), ('black_holes', 'BLACK_HOLES')) if flag in features}
    if journals:
        history = request.get('lifecycle_journals')
        if not history or set(history['kinds']) != journals:
            raise ValueError('Clean release profile must compare every compiled lifecycle journal family')
        for side in ('uninterrupted', 'resumed'):
            for kind in journals:
                path = Path(history['directories'][side][kind])
                if not path.is_absolute():
                    path = root/path
                path = path.resolve()
                if not path.is_relative_to(workspace):
                    raise ValueError('Clean release journal prefix must be retained in the workspace')
                history['directories'][side][kind] = str(path)
        for key in ('maximum_events_per_family', 'maximum_commits_per_family', 'maximum_tolerance_fields'):
            if type(history.get(key)) is not int or history[key] < 1:
                raise ValueError('Freeze bounded native lifecycle comparison allocations')
        _tolerances(history['float_tolerances'], history['float_tolerances'].keys())
        if len(history['float_tolerances']) > history['maximum_tolerance_fields']:
            raise ValueError('Lifecycle tolerance catalog exceeds frozen allocation')
        request['require_lifecycle_history'] = True
    elif request.get('lifecycle_journals'):
        raise ValueError('Collisionless profile cannot claim nonexistent lifecycle journals')
    request.update(require_complete_source_state=True, require_particle_integration_state=True,
                   require_global_state=True)
    return {'request': request, 'template_hash': canonical_digest(template),
            'prepared_request_hash': canonical_digest(request), 'layout_sources': pins,
            'execution_status': 'REQUEST_PREPARED_ONLY'}


def publish_native_release_comparison(prepared, destination, *, marker, resume_invocation,
                                      uninterrupted_invocation):
    for path, digest in prepared['layout_sources'].items():
        if file_sha256(path) != digest:
            raise ValueError('Native release comparison layout changed after preflight')
    if canonical_digest(prepared['request']) != prepared['prepared_request_hash']:
        raise ValueError('Frozen native release comparison changed during branch evolution')
    request = deepcopy(prepared['request'])
    for name, value in (('checkpoint_marker', marker), ('resume_invocation', resume_invocation),
                        ('uninterrupted_invocation',uninterrupted_invocation)):
        path = Path(value).resolve()
        request[name] = {'path': str(path), 'sha256': file_sha256(path)}
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Native release comparison requests are immutable')
    atomic_json(destination, request)
    return request


# Release campaign recipes
def compile_clean_release_campaign(request, destination, *, source_root):
    
    
    from core.research.capabilities import validate_native_answers
    from core.evolution import gadget_config_text
    root = Path(source_root).resolve()
    destination = Path(destination).resolve()
    if request.get('schema') != 'clean-linux-release-recipes-request-v1' or set(request['profiles']) != PROFILES:
        raise ValueError('Explicit collisionless, hydro_stellar_bh and sidm release recipes required')
    if not destination.is_relative_to(root) or destination.exists():
        raise ValueError('Compiled release settings need a new directory inside the packaged source tree')
    maximum = request['maximum_metadata_bytes']
    if type(maximum) is not int or maximum < 1:
        raise ValueError('Explicit bounded release recipe metadata required')
    sources = {}
    payloads = {}
    profiles = {}

    def asset(reference):
        if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
            raise ValueError('Release input requires an exact packaged path/hash pin')
        path = Path(reference['path'])
        if not path.is_absolute():
            path = root/path
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink() or file_sha256(path) != reference['sha256']:
            raise ValueError('Native release input must be current and retained inside the source package')
        sources[str(path)] = reference['sha256']
        return path

    def json_input(reference):
        path = asset(reference)
        if path.stat().st_size > maximum:
            raise MemoryError('Release input metadata exceeds explicit intake bound')
        return path, json.loads(path.read_text())

    def portable(path):
        return '{suite}/'+path.relative_to(root).as_posix()

    for name, settings in request['profiles'].items():
        answer_path, answers = json_input(settings['answers'])
        _, workflow = json_input(settings['public_workflow'])
        ic = asset(settings['existing_ic'])
        generation = asset(settings['native_generation_report'])
        if answers.get('restart_source_capsule') is not True:
            raise ValueError('Release answers must explicitly enable the native restart state diagnostics')
        configuration = validate_native_answers(answers, settings['resources'])
        config_text = gadget_config_text(configuration)
        features = {line.split('#', 1)[0].strip().split('=', 1)[0] for line in config_text.splitlines()
                    if line.split('#', 1)[0].strip()}
        required = {'RESEARCH_RESTART_STATE_CAPSULE'}
        if name == 'hydro_stellar_bh':
            required |= {'COOLING', 'STARFORMATION', 'STELLAR_EVOLUTION', 'BLACK_HOLES'}
            if answers.get('physics') != 'hydro' or answers.get('sidm', {}).get('enabled', False):
                raise ValueError('Hydro release profile needs its declared gas/stellar/BH physics without silently adding SIDM')
        elif name == 'sidm':
            required |= {'RESEARCH_SIDM_REFERENCE'}
            if not answers.get('sidm', {}).get('enabled', False):
                raise ValueError('SIDM release answers must enable the actual native operator')
        elif answers.get('physics') != 'dm' or answers.get('sidm', {}).get('enabled', False):
            raise ValueError('Collisionless release profile must retain DM-only collisionless physics')
        if not required <= features:
            raise ValueError('Actual public native configuration omits mandatory release features: '+name)
        if workflow.get('schema') != 'research-workflow-v1' or not workflow.get('stages'):
            raise ValueError('Each profile needs a concrete registered small public workflow')
        for field in ('mpi_ranks', 'omp_threads', 'memory_gib'):
            if workflow['resources'][field] != settings['resources'][field]:
                raise ValueError('Release public workflow cannot silently change its native allocation')
        requested = settings['scene_product']
        producers = [stage for stage in workflow['stages'] if requested in stage['outputs']]
        if len(producers) != 1 or requested not in workflow['requested_products']:
            raise ValueError('Fresh Blender scene must be an actual required workflow product with one producer')
        if not isinstance(settings['scene_artifact'], str) or Path(settings['scene_artifact']).suffix != '.blend':
            raise ValueError('Explicit actual .blend artifact path inside the scene product required')
        relative_scene = Path(settings['scene_artifact'])
        if relative_scene.is_absolute() or '..' in relative_scene.parts:
            raise ValueError('Scene artifact must remain inside its completed workflow product')
        for key in ('build_wall_seconds', 'evolution_wall_seconds', 'workflow_wall_seconds',
                    'blender_wall_seconds', 'comparison_wall_seconds', 'checkpoint_wait_seconds'):
            _seconds(settings[key], 'Release '+key)
        if type(settings['minimum_checkpoint_generation']) is not int or settings['minimum_checkpoint_generation']<1:
            raise ValueError('Release checkpoint generation must be an explicit positive integer')
        if not isinstance(settings['blender_executable'],str) or not settings['blender_executable']:
            raise ValueError('Explicit Blender executable required for the fresh-load control')
        comparison = deepcopy(settings['native_comparison'])
        for scope in ('integrator_state', 'hydro_particle_state', 'global_state'):
            if set(comparison[scope]['compiled_features']) != features:
                raise ValueError('Frozen restart state profile differs from the actual public native configuration')
        for side, run in (('uninterrupted', 'reference'), ('resumed', 'interrupted')):
            comparison[side]['run_directory'] = '{profile_output}/restart/'+run
            for kind in (comparison.get('lifecycle_journals') or {}).get('kinds', []):
                comparison['lifecycle_journals']['directories'][side][kind] = '{profile_output}/restart/'+run+'/data'
        # Use a planned workspace solely to resolve request paths; nothing is executed or measured.
        planned_workspace = root.parent/(root.name+'-planned-clean-release')
        prepare_native_release_comparison(comparison, {'suite': root, 'profile_output': planned_workspace/'release-results'/name},
                                         release_root=root, workspace=planned_workspace)

        def native(run, prepare=False):
            argv = ['{python}', '{suite}/commands.py', 'gadget', '--answers', portable(answer_path),
                    '--run-dir', run, '--existing-ic', portable(ic), '--native-generation-report', portable(generation)]
            if prepare:
                argv.append('--prepare-only')
            return {'argv': argv, 'wall_seconds': settings['build_wall_seconds'] if prepare else settings['evolution_wall_seconds']}

        controls = [
            {'kind': 'public_workflow', 'recipe': {'argv': ['{python}', '{suite}/commands.py', 'workflow',
                '--campaign', portable(asset(settings['public_workflow'])), '--output', '{profile_output}/workflow'],
                'wall_seconds': settings['workflow_wall_seconds']}, 'evidence': '{profile_output}/workflow/workflow.json'},
            {'kind': 'blender_fresh_load', 'recipe': {'argv': ['{python}', '{suite}/commands.py',
                'release-blender-control', '--workflow', '{profile_output}/workflow/workflow.json',
                '--scene-product', requested, '--scene-artifact', relative_scene.as_posix(),
                '--blender', settings['blender_executable'], '--wall-seconds',str(settings['blender_wall_seconds']),
                '--output', '{profile_output}/fresh-load'],
                'wall_seconds': settings['blender_wall_seconds']}, 'evidence': '{profile_output}/fresh-load/fresh-load.json'},
            {'kind': 'interrupt_resume', 'recipe': {
                'uninterrupted': native('{profile_output}/restart/reference'),
                'initial': native('{profile_output}/restart/interrupted'),
                'resume': {'argv': ['{python}', '{suite}/commands.py', 'gadget', '--run-dir',
                    '{profile_output}/restart/interrupted', '--resume'], 'wall_seconds': settings['evolution_wall_seconds']},
                'checkpoint_marker': '{profile_output}/restart/checkpoint-published.json',
                'checkpoint_wait_seconds': settings['checkpoint_wait_seconds'],
                'checkpoint_run_dir': '{profile_output}/restart/interrupted',
                'checkpoint_mpi_ranks': settings['resources']['mpi_ranks'],
                'minimum_checkpoint_generation': settings['minimum_checkpoint_generation'],
                'native_comparison': comparison, 'comparison_wall_seconds': settings['comparison_wall_seconds'],
                'comparison_receipt': '{profile_output}/restart-comparison/restart-comparison.json'}},
            {'kind': 'patch_reproduction', 'recipe': {'bundle_manifest': 'core/patches/gadget4-full-source/manifest.json'}},
        ]
        profiles[name] = {'binary': '{profile_output}/prepared/input/Gadget4',
            'build_signature': '{profile_output}/prepared/input/build_signature.json',
            'build_receipt': '{profile_output}/prepared/input/build_receipt.json',
            'build': native('{profile_output}/prepared', prepare=True),
            'required_features': sorted(required), 'controls': controls}
        payloads[name+'.Config.sh'] = config_text
    campaign = {'schema': 'clean-linux-release-campaign-v1', 'profiles': profiles}
    for path,digest in sources.items():
        if file_sha256(path)!=digest:
            raise ValueError('Frozen physical release input changed during recipe compilation')
    destination.mkdir(parents=True)
    atomic_json(destination/'campaign.json', campaign)
    for name, content in payloads.items():
        atomic_text(destination/name, content)
    sources[str(Path(__file__).resolve())] = file_sha256(__file__)
    report = {'schema': 'compiled-clean-linux-release-recipes-v1', 'request_hash': canonical_digest(request),
        'campaign': str(destination/'campaign.json'), 'campaign_sha256': file_sha256(destination/'campaign.json'),
        'source_hashes': sources, 'execution_status': 'NOT_EXECUTED', 'science_acceptance': 'NOT_ESTABLISHED'}
    atomic_json(destination/'recipe.json', report)
    return report


def release_campaign_recipes_main():
    import argparse
    parser = argparse.ArgumentParser(description='Compile explicit clean Linux native feature/control recipes without executing them')
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.request.stat().st_size > 16*1024**2:
        raise MemoryError('Release recipe request exceeds metadata intake limit')
    print(json.dumps(compile_clean_release_campaign(json.loads(args.request.read_text()), args.output,
                                                   source_root=Path(__file__).resolve().parents[2]), indent=2))


# Release blender control
def run_release_blender_control(workflow_path, product, artifact, blender, destination, *, wall_seconds):
    from core.workflow import verify_completed_receipt
    
    _seconds(wall_seconds,'Fresh Blender control wall limit')
    resolved_blender = Path(shutil.which(str(blender)) or str(blender)).expanduser().resolve()
    if not resolved_blender.is_file():
        raise FileNotFoundError('Fresh-load control requires the actual selected Blender executable')
    workflow_path = Path(workflow_path).resolve()
    if workflow_path.stat().st_size > 16*1024**2:
        raise MemoryError('Fresh scene workflow receipt exceeds metadata bound')
    workflow = json.loads(workflow_path.read_text())
    if workflow.get('schema') != 'research-workflow-receipt-v1' or workflow.get('execution_status') != 'COMPLETE':
        raise ValueError('Release scene requires an actual completed public workflow')
    matches = []
    for stage in workflow['stages'].values():
        path = Path(stage['generation'])/'stage-receipt.json'
        if file_sha256(path) != stage['receipt_sha256']:
            raise ValueError('Completed release scene stage receipt changed')
        receipt = verify_completed_receipt(path)
        if product in receipt['products']:
            matches.append((path, receipt))
    if len(matches) != 1:
        raise ValueError('Fresh release scene needs exactly one actual owned workflow product')
    receipt_path, receipt = matches[0]
    reference = receipt['products'][product]
    base = Path(receipt['generation']).resolve()
    output_product = (base/reference['path']).resolve()
    relative = Path(artifact)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Fresh scene artifact must remain inside its owned workflow product')
    if reference['kind'] == 'directory':
        scene = (output_product/relative).resolve()
        if not scene.is_relative_to(output_product):
            raise ValueError('Fresh scene escaped its published product')
    else:
        if relative.as_posix() != output_product.name:
            raise ValueError('File scene product must name its exact published .blend artifact')
        scene = output_product
    if scene.suffix != '.blend' or not scene.is_relative_to(base) or not scene.is_file():
        raise ValueError('Completed workflow scene product is not a retained .blend')
    relative_scene = scene.relative_to(base).as_posix()
    if receipt['files_sha256'].get(relative_scene) != file_sha256(scene):
        raise ValueError('Fresh scene differs from the actual sealed producer artifact')
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Fresh scene controls retain immutable publication directories')
    destination.mkdir(parents=True)
    script = Path(__file__).resolve().parents[1]/'display_fresh_load.py'
    proof_path = destination/'fresh-load.json'
    inputs = [workflow_path, receipt_path, scene, script, resolved_blender]
    for name in receipt['files_sha256']:
        path = (base/name).resolve()
        if path.is_relative_to(output_product if reference['kind'] == 'directory' else scene.parent):
            inputs.append(path)
    with (destination/'blender.log').open('wb') as log:
        run_native_command([str(resolved_blender), '--background', str(scene), '--python', str(script), '--', str(proof_path)],
            cwd=base, receipt_path=destination/'invocation.json', inputs=tuple(dict.fromkeys(inputs)),
            timeout=wall_seconds, stdout=log, stderr=log)
    result = json.loads(proof_path.read_text())
    if result.get('schema') != 'blender-fresh-load-v1' or result.get('blend_sha256') != file_sha256(scene) or \
            result.get('execution_status') != 'COMPLETE' or result.get('status') != 'PASS' or not result.get('epochs'):
        raise ValueError('Fresh Blender process did not establish the exact scene timeline control')
    result.update(source_hashes={str(script): file_sha256(script),str(Path(__file__).resolve()):file_sha256(__file__)},
        input_hashes={str(path):file_sha256(path) for path in dict.fromkeys(inputs)},
        files_sha256={str(destination/'invocation.json'):file_sha256(destination/'invocation.json')},
        blender_executable={'path':str(resolved_blender),'sha256':file_sha256(resolved_blender)},
        workflow_scene_product=product, scene_artifact=str(scene), science_acceptance='SCENE_FRESH_LOAD_AND_MEASURED_TIMELINE_ONLY')
    atomic_json(proof_path, result)
    from core.foundation import atomic_text
    atomic_text(proof_path.with_suffix('.sha256'),file_sha256(proof_path)+'\n')
    return result


def release_blender_control_main():
    import argparse
    parser = argparse.ArgumentParser(description='Fresh-load a sealed scene product from an actual completed workflow')
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--scene-product', required=True)
    parser.add_argument('--scene-artifact', required=True)
    parser.add_argument('--blender', required=True)
    parser.add_argument('--wall-seconds', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_release_blender_control(args.workflow,args.scene_product,args.scene_artifact,
        args.blender,args.output,wall_seconds=args.wall_seconds),indent=2))
