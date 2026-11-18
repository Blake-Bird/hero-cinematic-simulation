"""Publish restart markers from coherent native rank checkpoint generations.

Uses the existing GADGET CRC/layout/clock readers. Merely finding restart files
or waiting a fixed duration never establishes that a checkpoint is complete.
"""
from __future__ import annotations


from pathlib import Path
import json
import sys
from core.foundation import atomic_json, atomic_text, file_sha256
from core.native import native_run_lock


def publish_native_checkpoint_marker(run_dir, marker, *, mpi_ranks, executable_sha256,
                                     minimum_generation=1, workspace=None):
    from core.data import verify_native_checkpoints,verify_checkpoint_epochs
    run = Path(run_dir).resolve()
    marker = Path(marker).resolve()
    root = Path(workspace).resolve() if workspace is not None else run
    if not run.is_relative_to(root) or not marker.is_relative_to(root):
        raise ValueError('Checkpoint run/marker escaped the owned workspace')
    if type(mpi_ranks) is not int or mpi_ranks < 1 or type(minimum_generation) is not int or minimum_generation < 1:
        raise ValueError('Exact original MPI count and positive minimum checkpoint generation required')
    with native_run_lock(marker,operation='publish observed native checkpoint marker'):
        if marker.exists():
            raise FileExistsError('Checkpoint markers are immutable')
        manifest = json.loads((run/'manifest.json').read_text())
        handoff = manifest['gadget_handoff']
        if handoff['mpi_ranks'] != mpi_ranks:
            raise ValueError('Checkpoint rank count differs from original native handoff')
        binary = handoff['executable']
        if binary['sha256'] != executable_sha256:
            raise ValueError('Checkpoint belongs to another native executable/profile')
        restart = run/'data/restartfiles'
        checkpoint = verify_native_checkpoints(restart,mpi_ranks,require_receipts=True)
        if any(row['generation'] < minimum_generation for row in checkpoint['files']):
            raise ValueError('No requested native checkpoint generation is published yet')
        epochs = verify_checkpoint_epochs(restart,checkpoint,run)
        files = {}
        for row in checkpoint['files']:
            payload = Path(row['path']).resolve()
            if not payload.is_relative_to(run):
                raise ValueError('Native checkpoint shard escaped its run')
            files[str(payload)] = row['sha256']
            files[str(payload)+'.checksum.json'] = row['receipt_sha256']
            epoch = row.get('epoch_metadata')
            if epoch is None:
                raise ValueError('Complete checkpoint requires native rank epoch sidecars')
            files[epoch['path']] = epoch['sha256']
        sources = {}

        def bind(reference):
            """Manifest-relative assets belong to the run, not the shell's cwd."""
            path = Path(reference['path']).expanduser()
            path = (path if path.is_absolute() else run/path).resolve()
            expected = reference['sha256']
            if not expected or str(path) in sources and sources[str(path)] != expected:
                raise ValueError('Missing or contradictory checkpoint asset binding: '+str(path))
            sources[str(path)] = expected

        bind(binary)
        for name in ('parameter_file','compiled_options_file','snapshot_schedule_file'):
            reference = handoff[name]
            if not reference.get('sha256'):
                raise ValueError('Checkpoint immutable native handoff input is unsealed: '+name)
            bind(reference)
        for reference in handoff['initial_conditions']:
            bind(reference)
        for reference in [*handoff.get('physical_source_assets',[]),*manifest.get('input_artifacts',{}).values()]:
            if reference.get('path') and reference.get('sha256'):
                bind(reference)
        config = run/'config.yaml'
        if file_sha256(config) != manifest['config_sha256']:
            raise ValueError('Checkpoint original run configuration changed')
        sources[str(config)] = manifest['config_sha256']
        for bindings in (files, sources):
            for path,expected in bindings.items():
                if workspace is not None and not Path(path).resolve().is_relative_to(root):
                    raise ValueError('Clean checkpoint input escaped the unrelated workspace')
                if file_sha256(path) != expected:
                    raise ValueError('Checkpoint generation/input changed during marker publication')
        # All current ranks must still describe the same generation after clock
        # and SHA256 work. A concurrent replacement is retried by the observer.
        if verify_native_checkpoints(restart,mpi_ranks,require_receipts=True) != checkpoint:
            raise ValueError('Native checkpoint changed during marker publication')
        record = {'schema':'native-checkpoint-publication-v1','status':'COMPLETE',
            'run_dir':str(run),'mpi_ranks':mpi_ranks,'generation':checkpoint['files'][0]['generation'],
            'executable_sha256':executable_sha256,'checkpoint_integrity':checkpoint,
            'native_epoch_records':epochs,'files_sha256':files,'source_hashes':sources,
            'publication_basis':'coherent actual native rank payload/checksum/layout/epoch verification',
            'scientific_acceptance':'NOT_ESTABLISHED'}
        atomic_json(marker,record)
        atomic_text(marker.with_name(marker.name+'.sha256'),file_sha256(marker)+'\n')
        return record


def checkpoint_marker_main():
    import argparse
    parser = argparse.ArgumentParser(description='Publish an immutable marker from actual coherent native checkpoints')
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--marker',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--mpi-ranks',type=int,required=True)
    parser.add_argument('--executable-sha256',required=True)
    parser.add_argument('--minimum-generation',type=int,default=1)
    args = parser.parse_args()
    result = publish_native_checkpoint_marker(args.run,args.marker,mpi_ranks=args.mpi_ranks,
        executable_sha256=args.executable_sha256,minimum_generation=args.minimum_generation,workspace=args.workspace)
    print(json.dumps(result,indent=2))


def archive_resume_checkpoint(context):
    """Keep interrupted shards before resume replaces them; publish only a full copy."""
    import shutil
    import tempfile
    from core.foundation import atomic_copy, publish_directory
    marker = Path(context.resume_checkpoint['marker_path']).resolve()
    digest = file_sha256(marker)
    record = json.loads(marker.read_text())
    if file_sha256(marker) != digest:
        raise ValueError('Checkpoint marker changed while reading the archive inputs')
    files = record['files_sha256']
    destination = context.generation/'checkpoint-archives'/digest
    if destination.exists():
        saved = json.loads((destination/'archive.json').read_text())
        if saved.get('checkpoint_marker_sha256') != digest or saved.get('original_files_sha256') != files:
            raise ValueError('Retained resume checkpoint archive differs from the current verified marker')
        for copy in saved['retained_copies'].values():
            if file_sha256(copy['path']) != copy['sha256']:
                raise ValueError('Retained interrupted checkpoint archive changed')
        return
    size = sum(Path(name).stat().st_size for name in files)
    current = sum(path.stat().st_size for path in context.generation.rglob('*') if path.is_file())
    if size + current > context.resources['disk_gib'] * 1024**3 or size > shutil.disk_usage(context.generation).free:
        raise MemoryError('Retained interrupted checkpoint archive exceeds stage/free disk budget')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.checkpoint-archive-', dir=destination.parent))
    copies = {}
    # A failed copy retains a visibly incomplete stage. The final destination
    # appears only when every payload and the index are durable together.
    for index, (name, expected) in enumerate(files.items()):
        if file_sha256(name) != expected:
            raise ValueError('Interrupted checkpoint changed before archive publication')
        target = stage/str(index)
        atomic_copy(name, target)
        if file_sha256(target) != expected:
            raise ValueError('Retained checkpoint archive copy changed')
        copies[name] = {'path': str(destination/str(index)), 'sha256': expected}
    if file_sha256(marker) != digest:
        raise ValueError('Checkpoint marker changed during archive publication')
    atomic_json(stage/'archive.json', {
        'schema': 'campaign-interrupted-checkpoint-archive-v1',
        'checkpoint_marker_sha256': digest, 'original_files_sha256': files,
        'retained_copies': copies,
    })
    publish_directory(stage, destination)


def evolution_checkpoint_policy():
    """The supported continuation command retains the prepared run's configuration."""
    root = Path(__file__).resolve().parents[2]
    return {'marker':'run/checkpoint-published.json',
        'argv':[sys.executable,str(root/'commands.py'),'gadget','--run-dir','{generation}/run','--resume']}


def validate_evolution_checkpoint_policy(row):
    policy=row.get('checkpoint_resume')
    if policy is None:return
    if (row['kind']!='public_command' or row['parameters'].get('command')!='gadget'
            or policy!=evolution_checkpoint_policy()):
        raise ValueError('Public evolution checkpoint continuation must retain the exact registered resume command and marker')
    argv=row['parameters'].get('arguments',[])
    if (argv.count('--run-dir') != 1 or argv.index('--run-dir') + 1 >= len(argv)
            or argv[argv.index('--run-dir') + 1] != '{generation}/run'):
        raise ValueError('Checkpoint continuation requires the declared owned evolution run directory')


def retain_failed_evolution_checkpoint(generation,arguments):
    """Publish only actual verified shards after the owned process has stopped.

    The caller preserves its original exception. Failure to verify a checkpoint
    is a diagnostic, never a fabricated resumable state or successful run.
    """
    generation=Path(generation).resolve()
    output=generation/'checkpoint-intake.json'
    try:
        run=Path(arguments[arguments.index('--run-dir')+1]).resolve()
        if run!=generation/'run':raise ValueError('Native checkpoint publication escapes its owned stage')
        manifest=json.loads((run/'manifest.json').read_text());handoff=manifest['gadget_handoff']
        marker=run/'checkpoint-published.json'
        if marker.exists():
            # A failed resumed attempt can publish a newer generation. Keep
            # the earlier marker as immutable attempt evidence.
            from uuid import uuid4
            prior=marker.with_name('checkpoint-published.previous-'+uuid4().hex+'.json')
            marker.rename(prior)
            seal=marker.with_name(marker.name+'.sha256')
            if seal.exists():seal.rename(prior.with_name(prior.name+'.sha256'))
        result=publish_native_checkpoint_marker(run,marker,mpi_ranks=handoff['mpi_ranks'],
            executable_sha256=handoff['executable']['sha256'])
        atomic_json(output,{'status':'VERIFIED_RESUMABLE_CHECKPOINT','marker':{'path':str(marker),'sha256':file_sha256(marker)},
            'generation':result['generation'],'scope':'Native checkpoint integrity only; interrupted evolution is not complete'})
    except Exception as error:
        atomic_json(output,{'status':'NO_VERIFIED_RESUMABLE_CHECKPOINT','error_type':type(error).__name__,'error':str(error),
            'scope':'Original evolution failure remains authoritative; no checkpoint admission was granted'})
