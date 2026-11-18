"""Shared units, durable file publication and scientific product contracts.

Native process ownership and software identity live in core.native. This module
uses only the standard library until a scientific validator is requested.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from types import MappingProxyType

# Adopted Python conversion constants. Native engines retain their own values.

G_CGS = 6.67430e-8
C_SI = 299_792_458.0  # Exact SI speed of light in vacuum, metres per second.
MSUN_G = 1.98847e33
MPC_CM = 3.0856775814913673e24
KPC_CM = MPC_CM / 1000.
SECONDS_PER_YEAR = 31557600.

# rho_crit,0 = 3 H100^2/(8 pi G). Converting g/cm^3 to
# (Msun/h)/(Mpc/h)^3 absorbs the h^2 factor into the displayed units.
H100_S = 1.e7 / MPC_CM
RHO_CRIT_MSUN_H_MPC_H3 = 3.*H100_S**2/(8.*math.pi*G_CGS)*MPC_CM**3/MSUN_G
G_MPC_KMS2_MSUN = G_CGS*MSUN_G/MPC_CM/1.e10


def physical_constants_contract():
    """Declare adopted conversion constants; native engines retain their own values."""
    return {'schema':'adopted-physical-constants-v1','scope':'Python producer conversions only; no native engine equivalence inferred',
        'constants':{
            'G_CGS':{'value':G_CGS,'units':'cm^3 g^-1 s^-2','status':'adopted measured constant'},
            'C_SI':{'value':C_SI,'units':'m/s','status':'exact SI definition'},
            'MSUN_G':{'value':MSUN_G,'units':'g','status':'adopted solar mass conversion'},
            'MPC_CM':{'value':MPC_CM,'units':'cm','status':'adopted parsec conversion'},
            'KPC_CM':{'value':KPC_CM,'units':'cm','definition':'MPC_CM / 1000'},
            'SECONDS_PER_YEAR':{'value':SECONDS_PER_YEAR,'units':'s','definition':'Julian year: 365.25 days of 86400 seconds'},
            'RHO_CRIT_MSUN_H_MPC_H3':{'value':RHO_CRIT_MSUN_H_MPC_H3,'units':'(Msun/h)/(Mpc/h)^3',
                'definition':'3 H100^2/(8 pi G) converted using adopted gram/length constants; h^2 absorbed into displayed units'}}}


ROOT = Path(__file__).resolve().parent


def file_sha256(path):
    """Hash any source/input asset with SHA-256 using bounded 1 MiB blocks.

    A byte digest identifies the exact file; it is not a scientific accuracy
    or simulation-convergence claim. Streaming also works inside Blender's
    bundled Python without scientific-package imports.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_blender_runtime(destination):
    """Copy the standard-library-only Blender runtime beside portable assets.

    These modules are needed when an export leaves the source tree. Avoiding
    imports of NumPy/HDF5 here keeps the runtime usable in Blender's Python.
    """
    destination = Path(destination).expanduser().resolve()
    if destination == ROOT.parent or destination.is_relative_to(ROOT):
        raise ValueError('Portable Blender exports must not overwrite the source runtime')
    destination.mkdir(parents=True, exist_ok=True)
    source, target = ROOT / 'blender.py', destination / 'blender.py'
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    runtime_package = destination/'core'
    runtime_package.mkdir(exist_ok=True)
    # Portable scenes use ordinary modules. They need neither the repository's
    # historical source-section loader nor its unrelated campaign algorithms.
    (runtime_package/'__init__.py').write_text(
        '"""Portable measured-scene runtime; no simulation backends."""\n'
    )
    for name in ('foundation.py', 'resources.py', 'native.py', 'blender_scene.py', 'display_runtime.py', 'research_inspector.py', 'display_fresh_load.py'):
        source, target = ROOT/name, runtime_package/name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    return destination / 'blender.py'


def choose_run_directory(requested=None, *, root=None):
    """Resolve a chosen run folder; relative names use the external runs root."""
    from core.resources import runtime_paths
    base = Path(root).expanduser().resolve() if root is not None else runtime_paths().runs_root
    if requested:
        path = Path(requested).expanduser()
        return path.resolve() if path.is_absolute() else (base / path).resolve()
    answer = ""
    while not answer:
        answer = input("Run output folder (required): ").strip()
        if not answer:
            print("Enter a folder for this run or stage.")
    path = Path(answer).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def verify_numpy_container(path,contract,*,memory_bytes=1024**3):
    """Shared stdlib integrity check for the central NPZ contract in portable scenes."""
    import zipfile
    if isinstance(memory_bytes,bool) or not isinstance(memory_bytes,int) or memory_bytes<=0:
        raise ValueError('NPZ workspace must be a positive integer byte count')
    if contract.get('schema')!='darkmattersim-data-contract-v1' or contract.get('container')!='pickle-free NPZ':
        raise ValueError('Unsupported central NPZ container contract')
    if file_sha256(path)!=contract.get('sha256'):raise ValueError('NPZ archive contract checksum mismatch')
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('NPZ duplicate member names are ambiguous')
        expected={name+'.npy' for name in contract['datasets']}
        if set(names)!=expected:raise ValueError('NPZ contract member inventory mismatch')
        for member in archive.infolist():
            with archive.open(member) as stream:
                while stream.read(min(memory_bytes,1024*1024)):pass
    if file_sha256(path)!=contract['sha256']:raise ValueError('NPZ archive changed during verification')
    return contract


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    """Never leave a half-written status file if interrupted while writing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            name = stream.name
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        fsync_directory(path.parent)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def atomic_text(path, value, *, overwrite=True):
    """Publish complete UTF-8 native input text, optionally refusing replacement.

    overwrite=False links a fully flushed temporary file into place only if
    the destination is absent. Concurrent writers cannot replace one another.
    """
    if not isinstance(value, str):
        raise TypeError('Atomic text publication requires a string')
    payload = value.encode('utf-8')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(name, path)
        else:
            os.link(name, path)
        fsync_directory(path.parent)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def atomic_copy(source, destination):
    """Stage a verified file copy, preserving mode/times and the old destination."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Atomic copy source and destination must differ')
    before = file_sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with source.open('rb') as original:
            with tempfile.NamedTemporaryFile(mode='wb', dir=destination.parent, delete=False) as copied:
                name = copied.name
                shutil.copyfileobj(original, copied, length=1024 * 1024)
                copied.flush()
                metadata = source.stat()
                os.fchmod(copied.fileno(), stat.S_IMODE(metadata.st_mode))
                os.utime(name, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
                os.fsync(copied.fileno())
        if file_sha256(source) != before or file_sha256(name) != before:
            raise ValueError('Atomic copy source changed or staged bytes differ: '+str(source))
        os.replace(name, destination)
        fsync_directory(destination.parent)
        return before
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def fsync_directory(directory):
    """Persist directory entries on POSIX; unsupported directory fsync is explicit."""
    import errno
    if os.name != 'posix':
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
    finally:
        os.close(descriptor)


def publish_directory(stage, destination):
    """Flush all closed staged products before atomic publication; preserve failures."""
    stage, destination = Path(stage), Path(destination)
    if destination.exists():
        raise FileExistsError('Immutable product destination already exists: '+str(destination))
    for path in sorted(stage.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    for directory in sorted([p for p in stage.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
        fsync_directory(directory)
    fsync_directory(stage)
    stage.rename(destination)
    fsync_directory(destination.parent)


def publish_file(stage, destination):
    """Persist a closed immutable file and retain staged bytes on failure.

    The caller holds the shared destination ownership lock. Concurrent writes
    outside supported producers are outside the publication contract.
    """
    stage,destination=Path(stage),Path(destination)
    if destination.exists():raise FileExistsError('Immutable product destination already exists: '+str(destination))
    with stage.open('rb') as source:os.fsync(source.fileno())
    stage.rename(destination)
    fsync_directory(destination.parent)


def verify_artifact_index(results_root, *, record=None):
    """Validate relocatable artifact links and bytes; never certify the science."""
    root = Path(results_root).expanduser().resolve()
    if root.is_relative_to(ROOT.parent.resolve()):
        raise ValueError('Run artifacts belong outside the portable source tree')
    if record is None:
        record = json.loads((root/'artifact-index.json').read_text())
    if (not isinstance(record, dict) or set(record) != {'schema','run_namespace','artifacts'}
            or record['schema'] != 'artifact-index-v1' or
            not isinstance(record['run_namespace'], str) or not record['run_namespace'].strip()
            or not isinstance(record['artifacts'], list)):
        raise ValueError('Complete run-owned artifact index required')
    required = {'artifact_id','run_namespace','catalogue_id','product_generation','object_selection',
        'epoch_selection','numeric_sources','recipe_hash','support_status','display_transform','relative_path','sha256'}
    identities, paths = set(), set()

    def file_binding(binding):
        relative, checksum = binding.get('relative_path'), binding.get('sha256')
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or '\\' in relative or any(p in {'.','..'} for p in relative.split('/'))
                or not isinstance(checksum, str) or len(checksum)!=64
                or any(c not in '0123456789abcdef' for c in checksum)):
            raise ValueError('Artifact binding requires a safe relative path and SHA256')
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_sha256(path)!=checksum:
            raise ValueError('Artifact link is missing, escaped or changed: '+relative)

    for row in record['artifacts']:
        if not isinstance(row,dict) or set(row)!=required:
            raise ValueError('Complete artifact metadata required')
        if row['run_namespace'] != record['run_namespace']:
            raise ValueError('Artifact belongs to another run namespace')
        for name in ('artifact_id','product_generation','object_selection','epoch_selection','display_transform'):
            if not isinstance(row[name],str) or not row[name].strip():
                raise ValueError('Explicit artifact '+name+' required')
        cid = row['catalogue_id']
        if cid is not None and (type(cid) is not int or not 1<=cid<=900):
            raise ValueError('Catalogue ID must retain its exact 1–900 meaning or be explicitly uncatalogued')
        recipe = row['recipe_hash']
        if not isinstance(recipe,str) or len(recipe)!=64 or any(c not in '0123456789abcdef' for c in recipe):
            raise ValueError('Artifact recipe requires SHA256')
        if row['support_status'] not in {'SUPPORTED','UNRESOLVED','INCOMPLETE','INAPPLICABLE'}:
            raise ValueError('Explicit artifact support status required')
        file_binding(row)
        if row['artifact_id'] in identities or row['relative_path'] in paths:
            raise ValueError('Duplicate artifact identity or output path')
        identities.add(row['artifact_id']);paths.add(row['relative_path'])
        if not isinstance(row['numeric_sources'],list) or not row['numeric_sources']:
            raise ValueError('Artifact must bind actual numeric sources')
        for source in row['numeric_sources']:
            if not isinstance(source,dict) or set(source)!={'relative_path','sha256','run_namespace'}:
                raise ValueError('Exact numeric source binding required')
            if source['run_namespace'] != record['run_namespace']:
                raise ValueError('Artifact numeric source belongs to another run namespace')
            file_binding(source)
    return record


def register_artifacts(results_root, artifacts):
    """Register producer-verified publications under one locked atomic index.

    The producer must verify its scientific/publication contract first. This
    index checks ownership, relative links and payload hashes; registration is
    not a physics acceptance decision or an executed catalogue-coverage claim.
    Existing identities are immutable; identical registration is idempotent.
    """
    from core.native import native_run_lock
    root = Path(results_root).expanduser().resolve()
    if root.is_relative_to(ROOT.parent.resolve()):
        raise ValueError('Run artifacts belong outside the portable source tree')
    if not isinstance(artifacts,list) or not artifacts:
        raise ValueError('Nonempty verified artifact records required')
    namespace = artifacts[0].get('run_namespace') if isinstance(artifacts[0],dict) else None
    verify_artifact_index(root, record={'schema':'artifact-index-v1',
        'run_namespace':namespace,'artifacts':artifacts})
    with native_run_lock(root, operation='publish run artifact index'):
        path = root/'artifact-index.json'
        current = verify_artifact_index(root) if path.exists() else {
            'schema':'artifact-index-v1','run_namespace':namespace,'artifacts':[]}
        by_id = {row['artifact_id']:row for row in current['artifacts']}
        for row in artifacts:
            if not isinstance(row,dict) or 'artifact_id' not in row:
                raise ValueError('Artifact identity required')
            previous = by_id.get(row['artifact_id'])
            if previous is not None and previous != row:
                raise ValueError('Cannot replace a registered artifact generation')
            by_id[row['artifact_id']] = row
        candidate = {**current,'artifacts':sorted(by_id.values(),key=lambda row:row['artifact_id'])}
        verify_artifact_index(root, record=candidate)
        if candidate != current or not path.exists():
            atomic_json(path, candidate)
        return candidate


def run_namespace_binding(run_dir, *, initialize=False):
    """Read persistent run ownership, allocating it only at an authorized launch.

    Replicas of one run retain its namespace after relocation. Independently
    launched runs get distinct UUIDs even with identical IC bytes and settings.
    Missing legacy ownership is unresolved; path or content hashes cannot prove
    that two identical simulations are the same run.
    """
    import uuid
    from core.native import native_run_lock
    if type(initialize) is not bool:
        raise ValueError('Explicit boolean namespace initialization required')
    root = Path(run_dir).expanduser().resolve()
    path = root/'manifest.json'

    def read():
        before = file_sha256(path) if path.is_file() else None
        record = json.loads(path.read_text()) if path.is_file() else {}
        if (file_sha256(path) if path.is_file() else None) != before:
            raise ValueError('Run manifest changed while reading ownership')
        if not isinstance(record, dict):
            raise ValueError('Run manifest must be an object')
        value = record.get('run_namespace')
        if value is not None and (not isinstance(value, str) or not value or
                len(value) > 256 or any(ch.isspace() or ord(ch) < 32 for ch in value)):
            raise ValueError('Invalid persistent run namespace')
        return record, value, before

    if initialize:
        with native_run_lock(root, operation='initialize native run namespace'):
            record, value, manifest_hash = read()
            if value is None:
                if not path.is_file():
                    raise FileNotFoundError('Prepare run manifest before native launch')
                value = 'research2-run:'+uuid.uuid4().hex
                record['run_namespace'] = value
                record.setdefault('run_id', value.removeprefix('research2-run:'))
                atomic_json(path, record)
                manifest_hash = file_sha256(path)
    else:
        _, value, manifest_hash = read()
    return {'status':'DECLARED' if value is not None else 'UNDECLARED',
            'run_namespace':value,
            'manifest_sha256':manifest_hash,
            'scope':'Persistent run ownership only; not particle ancestry or physics acceptance'}


def product_provenance(run_dir, *, algorithm, selection, source_files, coordinates, snapshot=None, helper_sources=()):
    """Canonical product lineage from verified native inputs, with honest absence flags."""
    from core.data import CANONICAL_SCHEMA, IDENTITY_KINDS
    root = Path(run_dir).expanduser().resolve()
    path = root/'manifest.json'
    manifest_hash = file_sha256(path) if path.is_file() else None
    native = json.loads(path.read_text()) if path.is_file() else {}
    sources = []
    for entry in source_files:
        source = Path(entry['path'] if isinstance(entry,dict) else entry).resolve()
        checksum = file_sha256(source)
        if isinstance(entry,dict) and checksum != entry.get('sha256'):
            raise ValueError('Product native input changed: '+str(source))
        sources.append({'path':str(source),'sha256':checksum})
    handoff = native.get('gadget_handoff',{})
    def artifact(entry):
        if not entry:
            return {'status':'UNAVAILABLE'}
        p = Path(entry['path']).expanduser()
        if not p.is_absolute():
            p = root/p
        if not p.is_file() or file_sha256(p) != entry.get('sha256'):
            raise ValueError('Product native provenance artifact must remain unchanged; missing or changed: '+str(p))
        return {'status':'VERIFIED','path':str(p.resolve()),'sha256':entry['sha256']}
    inputs = {name:artifact(entry) for name,entry in native.get('input_artifacts',{}).items()}
    initial = [artifact(entry) for entry in handoff.get('initial_conditions',[])]
    lineage = {'initial_conditions':[i['sha256'] for i in initial],
               'cosmology':native.get('cosmology_parameters'),
               'simulation':native.get('simulation_parameters'),'hydro':native.get('hydro_parameters'),
               'model':native.get('model_parameters'),'seed':native.get('random_seed')}
    ownership = run_namespace_binding(root)
    namespace = (ownership['run_namespace'] if ownership['status']=='DECLARED' else
        'native-run:'+canonical_digest(lineage) if initial else 'unverified-saved-input:'+canonical_digest(sources))
    namespaces = {kind:namespace+':'+kind+(f':snapshot:{snapshot}' if snapshot is not None and kind in ('halo','subhalo','galaxy') else '')
                  for kind in IDENTITY_KINDS}
    namespaces['native_particle'] = namespace+':native_particle_union'
    config_path = root/'config.yaml'
    config_hash = file_sha256(config_path) if config_path.is_file() else None
    if native.get('config_sha256') is not None and native['config_sha256'] != config_hash:
        raise ValueError('Product run configuration changed after native preparation')
    algorithm = Path(algorithm).resolve()
    helpers = [algorithm,Path(__file__),Path(__file__).with_name('data.py'),
               Path(__file__).with_name('foundation.py'),Path(__file__).with_name('resources.py'),Path(__file__).with_name('native.py'),Path(__file__).with_name('linear.py'),Path(__file__).with_name('stream_resolution.py'),Path(__file__).with_name('cosmology.py'),
               Path(__file__).with_name('__init__.py')]
    source_layout = Path(__file__).resolve().parent.parent/'configs/core_source_layout.json'
    if source_layout.is_file():
        helpers.append(source_layout)
    helpers=list(dict.fromkeys([p.resolve() for p in helpers]+[Path(p).resolve() for p in helper_sources]))
    if (file_sha256(path) if path.is_file() else None) != manifest_hash or ownership['manifest_sha256'] != manifest_hash:
        raise ValueError('Run manifest changed during provenance construction')
    return {'schema':CANONICAL_SCHEMA,'run_namespace':ownership['run_namespace'],
            'run_ownership':ownership,'adopted_physical_constants':physical_constants_contract(),'input_hashes':sources,'run_manifest_sha256':manifest_hash,
            'config_sha256':config_hash,'executable':artifact(handoff.get('executable')),
            'parameter_file':artifact(handoff.get('parameter_file')),
            'compile_options':artifact(handoff.get('compiled_options_file')),
            'git_commit':{'status':'UNAVAILABLE','reason':'First-party workspace has no pinned Git commit; helper source hashes identify the implementation'},
            'native_source_revisions':native.get('software_revisions',{}),
            'native_revision_scope':'declared pinned revisions; build/source measurements remain in the native manifest',
            'cosmology_sha256':canonical_digest(native['cosmology_parameters']) if 'cosmology_parameters' in native else None,
            'boltzmann_inputs':inputs,'initial_conditions':initial,'random_seed':native.get('random_seed'),
            'algorithm_version':CANONICAL_SCHEMA,'algorithm_source_sha256':{p.name:file_sha256(p) for p in helpers},
            'algorithm_sources':[{'path':str(p.resolve()),'sha256':file_sha256(p)} for p in helpers],
            'coordinates':coordinates,'selection':selection,'identity_namespaces':namespaces,
            'identity_scope':'verified native IC lineage' if initial else 'unverified run lineage; cross-epoch ownership is not accepted',
            'warnings':native.get('warnings',[]),'limits':['This contract is provenance, not scientific or convergence acceptance']}


def verify_product_provenance(run_dir, contract):
    """Reject source/native/config changes between measurement and publication/reuse."""
    root = Path(run_dir).resolve()
    records = (contract['input_hashes']+contract['initial_conditions']
               +list(contract['boltzmann_inputs'].values())
               +[contract['executable'],contract['parameter_file'],contract['compile_options']])
    for entry in records:
        if entry.get('status') == 'UNAVAILABLE':
            continue
        if not Path(entry['path']).is_file() or file_sha256(entry['path']) != entry['sha256']:
            raise ValueError('Product provenance source changed: '+entry['path'])
    for key,name in (('config_sha256','config.yaml'),('run_manifest_sha256','manifest.json')):
        expected = contract[key]
        if expected is not None and (not (root/name).is_file() or file_sha256(root/name) != expected):
            raise ValueError('Product provenance run metadata changed: '+name)
    for entry in contract['algorithm_sources']:
        path = Path(entry['path'])
        if not path.is_file() or file_sha256(path) != entry['sha256']:
            raise ValueError('Product provenance implementation changed: '+str(path))
    return contract


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive')
    return float(value)


def validation_campaign_record(entry):
    """Resolve a shipped deferred campaign by portable path and exact content hash.

    Loading the campaign never launches a job. This additive spec field leaves
    earlier schema-1 specifications and scientific evidence rules unchanged.
    """
    if not isinstance(entry,dict) or set(entry)!={'path','sha256'}:
        raise ValueError('validation_campaign requires portable path and sha256')
    if not isinstance(entry['path'],str) or not isinstance(entry['sha256'],str):
        raise ValueError('validation_campaign path/hash must be strings')
    root=Path(__file__).resolve().parents[1]
    relative=Path(entry['path'])
    path=(root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Deferred campaign must reference a shipped project-relative file')
    if file_sha256(path)!=entry['sha256']:
        raise ValueError('Deferred validation campaign changed; version and rebind the specification')
    value=json.loads(path.read_text())
    if value.get('schema')!='research-validation-campaign-v1' or value.get('execution_state')!='VALIDATION_PENDING':
        raise ValueError('Coding-first campaign must explicitly record pending execution')
    if value.get('science_status')!='NOT_ACCEPTED':
        raise ValueError('An unexecuted campaign cannot declare science acceptance')
    for name in value.get('unit_controls',[]):
        control=(root/name).resolve()
        if not control.is_relative_to(root) or not control.is_file():
            raise ValueError('Deferred campaign references a missing control: '+str(name))
    return {'path':relative.as_posix(),'sha256':entry['sha256'],'configuration':value,
            'execution_state':'VALIDATION_PENDING','science_status':'NOT_ACCEPTED'}


def validate_spec(spec, products=None):
    """Require physical definitions, declared domains, criteria and selections."""
    if isinstance(spec, dict) and spec.get('schema_version') == 2:
        return validate_run_spec_v2(spec)
    required = {'schema_version', 'experiment_id', 'cosmology_role', 'target_population',
                'species', 'resource_budget', 'observables', 'excluded_claims', 'stream_resolution'}
    if not isinstance(spec, dict) or not required <= set(spec) or set(spec)-required-{'validation_campaign','start_epoch_campaign','resolution_budget','cosmography_campaign'} or type(spec['schema_version']) is not int or spec['schema_version'] != 1:
        raise ValueError('Research specification requires schema_version=1, optional validation_campaign/start_epoch_campaign and required ' + ', '.join(sorted(required)))
    if 'cosmography_campaign' in spec:
        from core.cosmography import validate_campaign_request
        validate_campaign_request(spec['cosmography_campaign'])
    if 'validation_campaign' in spec:
        validation_campaign_record(spec['validation_campaign'])
    if 'start_epoch_campaign' in spec:
        from core.evolution import validate_start_epoch_request
        validate_start_epoch_request(spec['start_epoch_campaign'],spec['observables'])
    if 'resolution_budget' in spec:
        from core.zoom import validate_resolution_request
        validate_resolution_request(spec['resolution_budget'])
    for key in ('experiment_id', 'target_population'):
        if not isinstance(spec[key], str) or not spec[key].strip():
            raise ValueError(f'{key} must describe the declared experiment')
    if spec['cosmology_role'] != 'conditioned_on_input_H0':
        raise ValueError('Legacy simulation/stream observables require cosmology_role=conditioned_on_input_H0; declare independent blinded expansion estimators in cosmography_campaign')
    for key in ('species', 'excluded_claims'):
        if not isinstance(spec[key], list) or not spec[key] or any(not isinstance(x, str) or not x.strip() for x in spec[key]):
            raise ValueError(f'{key} must be a nonempty list of names')
    if set(spec['species']) - {'dark_matter', 'gas', 'stars', 'black_holes'}:
        raise ValueError('Unsupported physical species requested')
    budget = spec['resource_budget']
    if not isinstance(budget, dict) or set(budget) != {'memory_gib', 'disk_gib', 'wall_hours', 'cpu_slots'}:
        raise ValueError('Declare memory_gib, disk_gib, wall_hours and cpu_slots resource limits')
    for key, value in budget.items():
        _number(value, key, positive=True)
    if type(budget['cpu_slots']) is not int:
        raise ValueError('cpu_slots must be an integer')
    observables = spec['observables']
    if not isinstance(observables, dict) or not observables:
        raise ValueError('Declare per-product observables')
    if products is not None and set(products) - set(observables):
        raise ValueError('Products absent from research specification: ' + ', '.join(sorted(set(products) - set(observables))))
    for name, item in observables.items():
        if not isinstance(item, dict) or set(item) != {'definition', 'units', 'selection', 'required_data', 'domain', 'tests'}:
            raise ValueError(f'{name}: declare definition, units, selection, required_data, domain and tests')
        for key in ('definition', 'units', 'selection', 'domain'):
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError(f'{name}.{key} must be explicit')
        if not isinstance(item['required_data'], list) or not item['required_data'] or any(not isinstance(x, str) or not x for x in item['required_data']):
            raise ValueError(f'{name}: required_data must name actual inputs')
        if not isinstance(item['tests'], dict) or not item['tests']:
            raise ValueError(f'{name}: at least one acceptance test is required')
        for test, rule in item['tests'].items():
            if not isinstance(rule, dict) or set(rule) != {'metric', 'units', 'minimum', 'maximum', 'evidence_kind'}:
                raise ValueError(f'{name}.{test}: malformed acceptance rule')
            if not isinstance(rule['metric'], str) or not rule['metric'] or not isinstance(rule['units'], str) or not rule['units']:
                raise ValueError(f'{name}.{test}: name metric and units')
            if rule['evidence_kind'] not in {'analytic_control', 'native_regression', 'pilot', 'convergence', 'population', 'recovery', 'render'}:
                raise ValueError(f'{name}.{test}: unsupported evidence kind')
            low, high = rule['minimum'], rule['maximum']
            if low is None and high is None:
                raise ValueError(f'{name}.{test}: acceptance needs a numerical bound')
            if low is not None:
                _number(low, test + '.minimum')
            if high is not None:
                _number(high, test + '.maximum')
            if low is not None and high is not None and low > high:
                raise ValueError(f'{name}.{test}: reversed limits')
        kinds = {rule['evidence_kind'] for rule in item['tests'].values()}
        if name in {'maps', 'blender'}:
            if 'render' not in kinds:
                raise ValueError(f'{name}: visual acceptance requires render evidence')
        elif not {'pilot', 'convergence'} <= kinds:
            raise ValueError(f'{name}: scientific acceptance requires pilot and convergence evidence')
    if {'stellar_components', 'stellar_histories'} & set(observables) and spec['stream_resolution'] is None:
        raise ValueError('Stellar-stream observables require declared resolution scales')
    if spec['stream_resolution'] is not None:
        from .stream_resolution import validate_stream_resolution
        validate_stream_resolution(spec['stream_resolution'])
    return spec


def validate_run_spec_v2(spec):
    """Canonical executable specification; clocks and units stay in existing contracts."""
    import math
    import copy
    required = {'schema_version', 'cosmology_bundle', 'epochs', 'dark_matter', 'resources', 'science'}
    optional = {'legacy_analysis', 'migration', 'capabilities', 'native_config', 'combination_specification'}
    if set(spec) - required - optional or not required <= set(spec) or type(spec['schema_version']) is not int or spec['schema_version'] != 2:
        raise ValueError('Complete version-2 run specification required')
    for name in ('epochs', 'dark_matter', 'resources', 'science'):
        if not isinstance(spec[name], dict):
            raise ValueError('Run specification section must be a mapping: ' + name)
    epoch = spec['epochs']
    if not {'z_start', 'z_end', 'backbone_redshifts', 'start_epoch_comparison'} <= set(epoch):
        raise ValueError('Every starting, ending, output and comparison epoch must be explicit')
    for name in ('z_start', 'z_end'):
        value = epoch[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= -1:
            raise ValueError('Finite supported redshift greater than minus one required: ' + name)
    if epoch['z_start'] <= epoch['z_end']:
        raise ValueError('Starting redshift must precede the ending epoch')
    schedule = epoch['backbone_redshifts']
    if (not isinstance(schedule, list) or not schedule or any(isinstance(z, bool) or not isinstance(z, (float, int))
            or not math.isfinite(z) or not epoch['z_end'] <= z <= epoch['z_start'] for z in schedule)
            or any(a <= b for a, b in zip(schedule, schedule[1:]))):
        raise ValueError('Explicit strictly descending output epochs inside the requested history required')
    comparison = epoch['start_epoch_comparison']
    if (not isinstance(comparison, list) or len(set(comparison)) != len(comparison)
            or any(isinstance(z, bool) or not isinstance(z, (float, int)) or not math.isfinite(z) or z <= epoch['z_end'] for z in comparison)):
        raise ValueError('Distinct finite explicit comparison starts required')
    dm = spec['dark_matter']
    if dm.get('linear_model') not in {'cdm', 'wdm'} or dm.get('interactions', {}).get('kind') not in {'collisionless', 'elastic_sidm'}:
        raise ValueError('Separate supported linear and nonlinear interaction models required')
    resources = spec['resources']
    for name in ('mpi_ranks', 'omp_threads', 'maximum_stage_count'):
        if type(resources.get(name)) is not int or resources[name] <= 0:
            raise ValueError('Explicit positive resource integer required: ' + name)
    for name in ('memory_gib', 'disk_gib'):
        _number(resources.get(name), name, positive=True)
    limits = resources.get('followup_limits', {})
    for name in ('maximum_children', 'maximum_depth'):
        if type(limits.get(name)) is not int or limits[name] < 0:
            raise ValueError('Explicit nonnegative follow-up limit required: ' + name)
    science = spec['science']
    selected, recipes, options = science.get('products'), science.get('product_recipes',{}), science.get('product_options',{})
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or any(not isinstance(p, str) or not p for p in selected):
        raise ValueError('Distinct explicitly selected science products required')
    if (not isinstance(recipes, dict) or not isinstance(options,dict) or not (recipes or options)
            or any(not isinstance(name,str) or not isinstance(row,dict) for name,row in [*recipes.items(),*options.items()])):
        raise ValueError('Explicit custom recipes or standard scientific product options required')
    if set(recipes)&set(options):raise ValueError('A product must select one custom recipe or standard producer option')
    if set(science)-{'products','product_recipes','product_options'}:raise ValueError('Unknown version-2 science intake field')
    bundle = spec['cosmology_bundle']
    if not isinstance(bundle, dict) or set(bundle) != {'path', 'sha256', 'background_model'}:
        raise ValueError('Original exact solver bundle and independent background model required')
    if bundle['background_model'] not in {'lcdm', 'ede'} or not isinstance(bundle['path'], str) or not bundle['path']:
        raise ValueError('Supported background and portable linear bundle path required')
    if not isinstance(bundle['sha256'], str) or len(bundle['sha256']) != 64 or any(c not in '0123456789abcdef' for c in bundle['sha256']):
        raise ValueError('Exact solver bundle SHA256 required')
    if 'legacy_analysis' in spec:
        validate_spec(spec['legacy_analysis'])
    return copy.deepcopy(spec)


def migrate_legacy_run_spec(legacy, explicit):
    """Retain original bytes/meaning; never guess missing epochs or physics."""
    import copy
    validate_spec(legacy)
    if legacy['schema_version'] != 1 or not isinstance(explicit, dict):
        raise ValueError('Version-1 legacy specification and explicit missing run inputs required')
    result = copy.deepcopy(explicit)
    result['schema_version'] = 2
    result['legacy_analysis'] = copy.deepcopy(legacy)
    result['migration'] = {'schema': 'run-spec-migration-v1', 'original_spec_sha256': canonical_digest(legacy),
        'original_schema_version': 1, 'missing_run_fields_policy': 'explicit caller inputs only',
        'historical_manifest_reinterpretation': False}
    return validate_run_spec_v2(result)


def legacy_science_view(spec):
    """V1-only analysis routes receive their original frozen description."""
    validated = validate_spec(spec)
    if validated['schema_version'] == 1:
        return validated
    if 'legacy_analysis' not in validated:
        raise ValueError('This historical analysis route requires an explicit legacy_analysis view; use the v2 product workflow otherwise')
    return validated['legacy_analysis']


def acceptance_report(manifest, evidence, root):
    """Recompute checks; require this exact manifest and all evidence file hashes.

    Evidence JSON is keyed product/test and each entry holds manifest_sha256,
    evidence_kind, metric, units, value and files [{path, sha256}]. The value
    is retained as a measurement claim whose backing files can be audited.
    """
    if (not isinstance(manifest, dict) or not isinstance(manifest.get('products'), list)
            or not manifest['products'] or any(not isinstance(x, str) or not x for x in manifest['products'])
            or len(set(manifest['products'])) != len(manifest['products'])):
        raise ValueError('Manifest requires a nonempty unique selected-product list')
    if not isinstance(evidence, dict):
        raise ValueError('Evidence must be an object keyed by product/test')
    spec = legacy_science_view(manifest['research_spec'])
    validate_spec(spec, manifest['products'])
    manifest_hash = canonical_digest(manifest)
    root = Path(root).resolve()
    context_checks = {}
    for name, paths, base in (
        ('configuration', {'config.yaml': manifest['config_sha256']} if 'config_sha256' in manifest else {}, root),
        ('input_data', manifest.get('data_sha256', {}), root),
        ('source_code', manifest.get('source_sha256', {}), Path(__file__).resolve().parents[1]),
    ):
        if not isinstance(paths, dict):
            raise ValueError('Manifest checksum inventory must be an object: ' + name)
        changed = []
        for relative, expected in paths.items():
            path = (base / relative).resolve()
            if not path.is_relative_to(base) or not path.is_file() or file_sha256(path) != expected:
                changed.append(relative)
        context_checks[name] = {'status': 'FAIL' if changed else 'PASS', 'changed_or_missing': changed}
    if 'data_sha256' in manifest:
        current_data = {str(p.relative_to(root)) for p in (root / 'data').rglob('*') if p.is_file()}
        added = sorted(current_data - set(manifest['data_sha256']))
        if added:
            context_checks['input_data'].update(status='FAIL', added=added)
    if spec['stream_resolution'] is not None:
        try:
            from .stream_resolution import saved_stream_cadence
            context_checks['saved_stream_cadence'] = saved_stream_cadence(spec, root)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            context_checks['saved_stream_cadence'] = {'status': 'FAIL', 'reason': str(exc)}
    recycled_observables = set()
    if context_checks.get('saved_stream_cadence', {}).get('stellar_recycling_enabled') is True:
        context_checks['recycled_stellar_sampling'] = {
            'status': 'FAIL',
            'reason': 'Measured native stellar aperture counts and convergence have not been verified',
        }
        try:
            from .stream_resolution import recycled_stellar_sampling_observables
            recycled_observables = recycled_stellar_sampling_observables(spec)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            context_checks['recycled_stellar_sampling']['reason'] = str(exc)
    if 'start_epoch_campaign' in spec:
        try:
            from core.evolution import verify_start_epoch_comparison
            context_checks['start_epoch_campaign']=verify_start_epoch_comparison(
                evidence.get('start_epoch_campaign'),spec['start_epoch_campaign'])
        except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
            context_checks['start_epoch_campaign']={'status':'FAIL','reason':str(exc)}
    checks = {}
    if 'resolution_budget' in spec:
        try:
            path = (root/evidence['resolution_budget']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Resolution receipt must be retained in this run')
            if file_sha256(path) != path.with_name('provenance.sha256').read_text().strip():
                raise ValueError('Resolution receipt seal mismatch')
            report = json.loads(path.read_text())
            verify_product_provenance(root,report['data_contract'])
            from core.zoom import evaluate_resolution
            budgets = []
            stellar_sampling = []
            for row in report['records']:
                from core.data import verify_numeric_audit
                numeric=row['resolution_numeric_audit']
                numeric_path=path.parent/numeric['path']
                if file_sha256(numeric_path)!=numeric['sha256']:raise ValueError('Resolution numeric audit changed')
                verify_numeric_audit(numeric_path)
                saved = row['resolution_budget']
                if saved['request'] != spec['resolution_budget'] or saved['census']['measurement_origin'] != 'native_saved_snapshot':
                    raise ValueError('Resolution request or measured census origin mismatch')
                for source,checksum in saved['source_hashes'].items():
                    if file_sha256(source) != checksum:raise ValueError('Resolution measurement source changed')
                measured = saved.get('convergence')
                if measured is None:raise ValueError('Measured resolution convergence is missing')
                result = evaluate_resolution(saved['target'],saved['census'],saved['request'],convergence=measured['receipt_path'])
                budgets.append(result['science_acceptance'])
                if recycled_observables:
                    observed = {item['name']: item for item in result['observables']
                                if item['name'] in recycled_observables}
                    if set(observed) != recycled_observables:
                        raise ValueError('Measured resolution budget loses a required stellar-stream observable')
                    stellar_sampling.extend(item['science_acceptance'] for item in observed.values())
            context_checks['resolution_budget']={'status':'PASS' if budgets and all(x=='ACCEPTED' for x in budgets) else 'FAIL'}
            if recycled_observables:
                context_checks['recycled_stellar_sampling'] = {
                    'status': 'PASS' if stellar_sampling and all(value == 'ACCEPTED' for value in stellar_sampling) else 'FAIL',
                    'required_observables': sorted(recycled_observables),
                    'evaluated_aperture_observables': len(stellar_sampling),
                    'scope': 'Actual native species-4 counts and the original observable convergence criteria',
                }
        except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
            context_checks['resolution_budget']={'status':'FAIL','reason':str(exc)}
    for product in manifest['products']:
        checks[product] = {}
        for test, rule in spec['observables'][product]['tests'].items():
            record = evidence.get(product, {}).get(test) if isinstance(evidence.get(product, {}), dict) else None
            reason = None
            try:
                if not isinstance(record, dict) or record.get('manifest_sha256') != manifest_hash:
                    raise ValueError('missing evidence or mismatched manifest')
                for key in ('evidence_kind', 'metric', 'units'):
                    if record.get(key) != rule[key]:
                        raise ValueError('evidence contract mismatch: ' + key)
                value = _number(record.get('value'), 'measurement')
                files = record.get('files')
                if not isinstance(files, list) or not files:
                    raise ValueError('measurement has no supporting files')
                for item in files:
                    path = (root / item['path']).resolve()
                    if not path.is_relative_to(root) or not path.is_file() or file_sha256(path) != item['sha256']:
                        raise ValueError('missing, changed or external evidence file')
                if rule['minimum'] is not None and value < rule['minimum']:
                    raise ValueError('measurement below minimum')
                if rule['maximum'] is not None and value > rule['maximum']:
                    raise ValueError('measurement above maximum')
            except (ValueError, KeyError, TypeError, OSError) as exc:
                reason = str(exc)
            checks[product][test] = {'status': 'PASS' if reason is None else 'FAIL', 'reason': reason}
    return {'schema_version': 1, 'manifest_sha256': manifest_hash,
            'status': 'ACCEPTED' if (all(x['status'] == 'PASS' for tests in checks.values() for x in tests.values())
                                     and all(x['status'] == 'PASS' for x in context_checks.values())) else 'NOT_ACCEPTED',
            'checks': checks, 'context_checks': context_checks, 'scope': 'declared observables and domains only'}


# Immutable requests and results shared by scientific workflows.

def _freeze(value):
    """Freeze JSON-like workflow settings so queued work cannot change them."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ProductRef:
    """An explicit product identity and the unchanged manifest describing it."""
    product_id: str
    manifest_path: Path
    content_hash: str
    schema_version: int

    def __post_init__(self):
        if not isinstance(self.product_id, str) or not self.product_id.strip():
            raise ValueError('Explicit product ID required')
        if not isinstance(self.content_hash, str) or not re.fullmatch('[0-9a-f]{64}', self.content_hash):
            raise ValueError('Exact SHA256 required')
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError('Positive schema version required')
        object.__setattr__(self, 'manifest_path', Path(self.manifest_path).expanduser().resolve())

    def verify(self):
        if file_sha256(self.manifest_path) != self.content_hash:
            raise ValueError('Input product manifest changed')
        return self


@dataclass(frozen=True)
class StageRequest:
    """Frozen scientific settings and explicit input/output product identities."""
    stage_id: str
    parameters: Mapping
    inputs: tuple[ProductRef, ...]
    requested_products: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise ValueError('Explicit stage ID required')
        if not isinstance(self.parameters, Mapping):
            raise ValueError('Stage parameters must be a mapping')
        json.dumps(dict(self.parameters), allow_nan=False)
        inputs = tuple(self.inputs)
        products = tuple(self.requested_products)
        if not all(isinstance(product, ProductRef) for product in inputs):
            raise ValueError('Stage inputs require ProductRef')
        if (not products or any(not isinstance(product, str) or not product for product in products)
                or len(set(products)) != len(products)):
            raise ValueError('Distinct explicit requested products required')
        object.__setattr__(self, 'parameters', _freeze(self.parameters))
        object.__setattr__(self, 'inputs', inputs)
        object.__setattr__(self, 'requested_products', products)

    def verify_inputs(self):
        for product in self.inputs:
            product.verify()


@dataclass(frozen=True)
class StageResult:
    """Stage execution outcome; completeness does not imply physical acceptance."""
    execution_status: str
    products: tuple[ProductRef, ...]
    findings: tuple
    validation_receipts: tuple

    def __post_init__(self):
        if self.execution_status not in {'COMPLETE', 'FAILED', 'SKIPPED', 'INCOMPLETE'}:
            raise ValueError('Explicit execution status required')
        products = tuple(self.products)
        if any(not isinstance(product, ProductRef) for product in products):
            raise ValueError('Result products require ProductRef')
        object.__setattr__(self, 'products', products)
        for name in ('findings', 'validation_receipts'):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


@dataclass(frozen=True)
class PotentialState:
    """A measured potential's time, frame, components and declared error budget."""
    epoch: Mapping
    frame: Mapping
    component_refs: tuple[ProductRef, ...]
    error_budget_ref: ProductRef

    def __post_init__(self):
        if (not isinstance(self.epoch, Mapping)
                or self.epoch.get('canonical_time') not in {'scale_factor', 'isolated_native_elapsed_time'}):
            raise ValueError('Use the shared explicit epoch contract')
        if not isinstance(self.frame, Mapping) or not self.frame:
            raise ValueError('Use the shared explicit coordinate contract')
        components = tuple(self.component_refs)
        if (not isinstance(self.error_budget_ref, ProductRef)
                or any(not isinstance(product, ProductRef) for product in components)):
            raise ValueError('Potential components and error budget require ProductRef')
        object.__setattr__(self, 'epoch', _freeze(self.epoch))
        object.__setattr__(self, 'frame', _freeze(self.frame))
        object.__setattr__(self, 'component_refs', components)


def project_python_sources(project, *, include_verification=True):
    """Return actual first-party source files, excluding engines and build copies."""
    project=Path(project).resolve()
    files = set(project.glob('*.py')) | {
        path for path in (project/'core').rglob('*.py')
        if not {'software', 'patches', '__pycache__'}.intersection(path.relative_to(project/'core').parts)
    }
    if not include_verification:files.discard(project/'verification.py')
    return sorted(path for path in files if path.is_file())


def merge_source_bindings(pins, sources):
    """Add canonical source paths without replacing a different generation."""
    for path, checksum in sources.items():
        path = str(Path(path).resolve())
        if path in pins and pins[path] != checksum:
            raise ValueError('Conflicting source generations: ' + path)
        pins[path] = checksum

