"""Portable source selection and the corresponding native GADGET patch payload.

Scientific tables and attribution stay in releases. Upstream documentation and
generated products are excluded; project packaging retains the operator README.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import argparse
import zipfile
import copy
from pathlib import Path



def attribution_asset(path):
    """Attribution remains distributed even when upstream manuals are omitted."""
    name = Path(path).name.lower()
    return name.startswith((
        'license', 'licence', 'copying', 'copyright', 'notice', 'authors',
        'citation', 'attribution',
    )) or name in {'contributors', 'contributors.txt'}


def nonruntime_document(path):
    """Identify documentation without confusing licenses with disposable notes."""
    path = Path(path)
    name = path.name.lower()
    if attribution_asset(path):
        return False
    return (
        any(part.lower() in {
            'doc', 'docs', 'documentation', 'manual', 'apidocs', 'paper',
        } for part in path.parts)
        or name.startswith(('readme', 'changelog', 'change_log', 'changes', 'release_notes'))
        or name in {'install', 'news', 'todo', 'doxyfile', 'whatsnew'}
        or path.suffix.lower() in {
            '.md', '.rst', '.pdf', '.ipynb', '.tex', '.rtf', '.doc', '.docx',
            '.ppt', '.pptx', '.dox',
        }
    )


def nonruntime_asset(path):
    """Exclude generated upstream outputs/fixtures while retaining model tables."""
    path = Path(path)
    if attribution_asset(path):
        return False
    if nonruntime_document(path):
        return True
    if path.name.lower() in {
        'remaining_gap_coding_progress.json', 'grackle_3.4.1.source.tar.gz',
        'mathgl-1.11.3-2.tar.gz',
    } or '_tmp.' in path.name.lower():
        return True
    suffix = path.suffix.lower()
    parts = {part.lower() for part in path.parts}
    if suffix in {'.log', '.blend', '.blend1', '.mp4', '.webm', '.mov', '.avi'}:
        return True
    if suffix in {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.eps', '.ps'}:
        return not parts.intersection({'assets', 'icons', 'templates', 'shaders'})
    fixtures = {
        'test', 'tests', 'testing', 'test_data', 'test-data', 'expected-results',
        'paper', 'examples',
    }
    fixture_suffixes = {
        '.npz', '.npy', '.pkl', '.pickle', '.h5', '.hdf5', '.fits', '.dat',
        '.out', '.bin', '.txt', '.html', '.htm', '.xml', '.csv', '.gz',
    }
    return (
        bool(parts.intersection(fixtures))
        or path.name.lower().startswith('example_')
    ) and suffix in fixture_suffixes


def native_release_payload(pristine, shipped, selected, *, source_root, bundle_root):
    """Author the native delta against exactly the files selected for release.

    Missing excluded documentation is allowed when repackaging an extracted
    release. Missing or changed runtime sources and physical tables are not.
    """
    from .foundation import file_sha256
    source_root = Path(source_root).resolve()
    pristine = Path(pristine).resolve()
    shipped = Path(shipped).resolve()
    provenance_path = source_root / 'core/software/upstream/gadget4.provenance.json'
    provenance = json.loads(provenance_path.read_text())
    # Verify every retained retrieval byte; an extracted release intentionally
    # lacks the excluded nonruntime paths recorded by the original inventory.
    absent = []
    for name, expected in provenance['files_sha256'].items():
        if not (pristine / name).exists() and nonruntime_asset(name):
            absent.append(name)
            continue
        if file_sha256(pristine / name) != expected:
            raise ValueError('Original retrieved native source changed: ' + name)
    # Both trees must see the same selection even when given a generator.
    # macOS exposes temporary directories through /var -> /private/var.
    # Compare canonical paths on both sides, or a complete native tree can
    # incorrectly project to an empty set when the caller uses the alias.
    selected = tuple(Path(path).resolve() for path in selected)
    trees = {
        label: {
            path.relative_to(root).as_posix(): path
            for path in selected if path.is_relative_to(root)
        }
        for label, root in [('pristine', pristine), ('shipped', shipped)]
    }
    projection = copy.deepcopy(provenance)
    projection['files_sha256'] = {
        name: file_sha256(path) for name, path in trees['pristine'].items()
    }
    projection['file_scope'] = (
        'Explicit portable release subset; original retrieval inventory '
        'verified before projection; documentation excluded'
    )
    projection['original_retrieval_provenance_sha256'] = file_sha256(provenance_path)
    with tempfile.TemporaryDirectory(prefix='native-release-projection-') as tmp:
        base = Path(tmp)
        for label, files in trees.items():
            for name, path in files.items():
                target = base / label / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        payload = build_native_patch_payload(
            base / 'pristine', base / 'shipped',
            repository_url=provenance['repository_url'], commit=provenance['commit'],
            upstream_release_path='core/software/upstream/gadget4',
            shipped_release_path='core/software/gadget4',
            bundle_release_path=bundle_root, upstream_provenance=projection,
        )
    manifest = json.loads(payload['manifest.json'])
    manifest['release_projection'] = {
        'original_retrieval_provenance_sha256': file_sha256(provenance_path),
        'excluded_paths_absent_from_retrieval_tree': sorted(absent),
        'excluded_pristine_paths': sorted(
            set(provenance['files_sha256']) - set(projection['files_sha256'])
        ),
        'policy': (
            'Original source bytes verified; original patch author/compiler '
            'algorithms retained; only documentation/generated fixtures/build '
            'artifacts excluded'
        ),
    }
    payload['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    return payload


SOURCE_ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox", "build_cache", "build", "dist", "notebooks", "runs", "results", "outputs", "output", "comparison_outputs", ".native-run-locks", "validation_history", ".source_history"}
EXCLUDE_SUFFIXES = {".pdf", ".pyc", ".pyo", ".o", ".opp", ".a", ".so", ".dylib", ".dll", ".exe", ".ipynb", ".blend", ".blend1", ".zip"}
GENERATED_ROOT_FILES = {'AUDIT_FILE_INVENTORY.json', 'coding_checklist.json',
                        'implementation_status.json', '.Rhistory',
                        'BLOCK01_10_REQUIREMENT_AUDIT.json'}
# ELF, PE and all Mach-O byte orders/universal headers. Some scientific tables
# are binary, so only executable/object signatures are excluded, not all data.
NATIVE_MAGIC = {b'\x7fELF', b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca', b'\xca\xfe\xba\xbf'}


def native_file(path: Path) -> bool:
    """Identify machine-specific native code by content, including extensionless binaries."""
    with path.open("rb") as stream:
        first = stream.read(4)
    return first in NATIVE_MAGIC or first[:2] == b'MZ'


def source_files():
    """Yield portable assets in sorted order; preserve engine tables and licensing."""
    paths = []
    for directory, dirs, files in os.walk(SOURCE_ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith((".dSYM", ".egg-info")))
        for name in files:
            path = Path(directory) / name
            # Ship the operator guide; exclude upstream documentation as usual.
            operator_guide = (path == SOURCE_ROOT / 'README.md' or
                              (path.is_relative_to(SOURCE_ROOT/'docs') and path.suffix.lower() == '.md'))
            if not operator_guide and nonruntime_asset(path.relative_to(SOURCE_ROOT)):continue
            if path.parent == SOURCE_ROOT and name.startswith('ewald_table_') and path.suffix == '.dat':
                continue
            if path.parent == SOURCE_ROOT and (name in GENERATED_ROOT_FILES or (path.suffix.lower() == '.md' and not operator_guide)):
                continue
            required_overlay = path in {SOURCE_ROOT/'core/patches/gadget4-source/native-source.zip',
                                        SOURCE_ROOT/'core/patches/gadget4-source/native-source-current.zip'}
            if name.startswith((".DS_Store", "._")) or name in {".git", "validation.json", "RELEASE.json"} or (path.suffix in EXCLUDE_SUFFIXES and not required_overlay):
                continue
            if path.is_symlink():
                if not path.exists() and 'workflow-references' in path.parts and 'docs' in path.parts:
                    # Optional upstream documentation fixtures are not runtime
                    # assets; never dereference their absent test-data targets.
                    continue
                if not path.resolve().is_relative_to(SOURCE_ROOT) or not path.is_file():
                    raise ValueError("Archive requires an internal regular-file symlink target: " + str(path))
            if not native_file(path):
                paths.append(path)
    yield from sorted(paths, key=lambda p: p.relative_to(SOURCE_ROOT).as_posix())


def release_inventory(paths) -> dict:
    """Hash exactly the distributed source assets; no scientific-success claims."""
    return {"schema_version": 1, "distribution": "portable scientific source", "engines": (SOURCE_ROOT / "core/software-revisions").read_text().splitlines(), "files_sha256": {p.relative_to(SOURCE_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def build_portable_zip(destination: Path) -> Path:
    """Atomically write a reproducible ZIP with normalized dates and retained file modes."""
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError('Keep previous portable release; select a new archive destination')
    if destination.is_relative_to(SOURCE_ROOT):
        raise ValueError("Write the release outside the source project")
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = list(source_files())
    bundle_root='core/patches/gadget4-full-source'
    pristine_root=SOURCE_ROOT/'core/software/upstream/gadget4'
    if not pristine_root.is_dir():
        raise FileNotFoundError('Portable native release needs its retained pinned GADGET upstream baseline')
    native_payload=native_release_payload(pristine_root,SOURCE_ROOT/'core/software/gadget4',paths,
        source_root=SOURCE_ROOT,bundle_root=bundle_root)
    # Generate against the exact sources being packaged; stale historic patches
    # do not stand in for the complete current implementation delta.
    paths=[path for path in paths if not path.relative_to(SOURCE_ROOT).as_posix().startswith(bundle_root+'/')]
    release=release_inventory(paths)
    for name,content in native_payload.items():
        release['files_sha256'][bundle_root+'/'+name]=hashlib.sha256(content).hexdigest()
    native_manifest=json.loads(native_payload['manifest.json'])
    for tree_name,root_name in (('pristine_hashes','core/software/upstream/gadget4'),('shipped_hashes','core/software/gadget4')):
        retained={name[len(root_name)+1:]:digest for name,digest in release['files_sha256'].items() if name.startswith(root_name+'/')}
        if retained!=native_manifest[tree_name]:
            raise ValueError('Native patch source snapshot differs from the source archive inventory: '+root_name)
    release['native_patch_bundles']=[bundle_root+'/manifest.json']
    inventory = json.dumps(release, indent=2, sort_keys=True) + "\n"
    # Separate publishers must not share a fixed temporary path.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + '.', suffix='.tmp', dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in paths:
                info = zipfile.ZipInfo("research2/" + path.relative_to(SOURCE_ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (path.stat().st_mode & 0xffff) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                content = path.read_bytes()
                relative = path.relative_to(SOURCE_ROOT).as_posix()
                if hashlib.sha256(content).hexdigest() != release['files_sha256'][relative]:
                    raise RuntimeError('Source changed during portable archive writing: ' + relative)
                archive.writestr(info, content)
            for name,content in sorted(native_payload.items()):
                info=zipfile.ZipInfo('research2/'+bundle_root+'/'+name,date_time=(2026,1,1,0,0,0))
                info.create_system=3;info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(info,content)
            info=zipfile.ZipInfo('research2/RELEASE.json',date_time=(2026,1,1,0,0,0))
            info.create_system=3;info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,inventory)
        verify_portable_zip(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Archive: {destination}\nSource files: {len(paths)}")
    return destination


def verify_portable_zip(path: Path) -> dict:
    """Check every archived source before extraction; reject unsafe ZIP paths."""
    from pathlib import PurePosixPath
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate release member')
        for name in names:
            parts=PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or not name.startswith('research2/') or '\\' in name:raise ValueError('Unsafe release member')
        inventory=json.loads(archive.read('research2/RELEASE.json'))
        expected={'research2/'+name for name in inventory['files_sha256']}|{'research2/RELEASE.json'}
        if set(names)!=expected:raise ValueError('Release manifest/member mismatch')
        for name,digest in inventory['files_sha256'].items():
            if hashlib.sha256(archive.read('research2/'+name)).hexdigest()!=digest:raise ValueError('Release source checksum mismatch: '+name)
    return inventory


def package_suite_main() -> None:
    """Package the current sources to the requested external ZIP destination."""
    parser = argparse.ArgumentParser(description='Create a portable source archive with its operator guide and required native sources.\n\nPinned engine sources, required physical tables, patches, licenses and\nfirst-party code ship together. Local environments, generated science\nproducts, caches and native executables never enter the ZIP.\nRebuild native engines after extraction with bootstrap.py on the target host.')
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument('--verify',type=Path,help='verify an existing release before extraction')
    args=parser.parse_args()
    if args.verify:
        report=verify_portable_zip(args.verify);print(json.dumps({'status':'PASS','files':len(report['files_sha256'])},indent=2))
    else:build_portable_zip(args.output)


# Author a complete native source delta for portable release packaging.
def portable_native_tree(root):
    root=Path(root).resolve();files={}
    for directory,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=sorted(name for name in dirs if name not in EXCLUDE_DIRS and not name.endswith(('.dSYM','.egg-info')))
        for name in sorted(names):
            path=Path(directory)/name
            if path.suffix in EXCLUDE_SUFFIXES or name.startswith(('.DS_Store','._')) or name in {'validation.json','RELEASE.json'}:
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError('Native source baseline must contain regular files: '+str(path))
            relative=path.relative_to(root).as_posix()
            if any(ord(character)<32 for character in relative):raise ValueError('Unsafe native source filename')
            if not native_file(path):files[relative]=path
    if not files or 'LICENSE' not in files or 'Makefile' not in files:
        raise ValueError('Complete licensed native source tree required')
    return files


def tree_hashes(root):
    from core.foundation import file_sha256
    return {name:file_sha256(path) for name,path in portable_native_tree(root).items()}


def build_native_patch_payload(upstream_root,shipped_root,*,repository_url,commit,
                               upstream_release_path,shipped_release_path,bundle_release_path,upstream_provenance):
    if len(commit)!=40 or any(character not in '0123456789abcdef' for character in commit):
        raise ValueError('Full pinned upstream Git commit required')
    upstream=portable_native_tree(upstream_root);shipped=portable_native_tree(shipped_root)
    if upstream_provenance.get('schema')!='retained-native-upstream-v1' or upstream_provenance.get('repository_url')!=repository_url or upstream_provenance.get('commit')!=commit:
        raise ValueError('Upstream source provenance differs from the pinned repository/commit')
    if {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in upstream.items()}!=upstream_provenance['files_sha256']:
        raise ValueError('Retained pristine source files changed after upstream retrieval')
    with tempfile.TemporaryDirectory(prefix='native-source-delta-') as temporary:
        base=Path(temporary)
        inventories={}
        for name,files in (('pristine',upstream),('shipped',shipped)):
            inventories[name]={}
            for relative,path in files.items():
                target=base/name/relative;target.parent.mkdir(parents=True,exist_ok=True)
                content=path.read_bytes();target.write_bytes(content)
                target.chmod(0o755 if path.stat().st_mode&0o111 else 0o644)
                inventories[name][relative]=hashlib.sha256(content).hexdigest()
        completed=subprocess.run(['git','diff','--no-index','--binary','--no-renames','--src-prefix=a/',
            '--dst-prefix=b/','--no-ext-diff','--no-textconv','--no-color','--','pristine','shipped'],cwd=base,capture_output=True,timeout=120)
        if completed.returncode not in (0,1):
            raise RuntimeError('Native source delta generation failed: '+completed.stderr.decode(errors='replace'))
        patch=completed.stdout
    if not patch:raise ValueError('Expected an actual source delta from the pinned upstream')
    # The Git paths contain a/pristine/ and b/shipped/: strip two components.
    patch_name='full-source.patch'
    manifest={'schema':'native-source-patch-bundle-v1','engine':'gadget4',
        'upstream':{'repository_url':repository_url,'commit':commit},
        'upstream_provenance_sha256':hashlib.sha256(json.dumps(upstream_provenance,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'pristine_source':upstream_release_path,'shipped_source':shipped_release_path,
        'pristine_hashes':inventories['pristine'],'shipped_hashes':inventories['shipped'],
        'patches':[{'path':bundle_release_path+'/'+patch_name,'sha256':hashlib.sha256(patch).hexdigest(),'strip':2}],
        'compare_files':{name:shipped_release_path+'/'+name for name in inventories['shipped']},
        'deleted_files':sorted(set(inventories['pristine'])-set(inventories['shipped'])),
        'tree_scope':'all portable regular native source/asset files under the two recorded roots',
        'execution_status':'SOURCE_DELTA_AUTHORED','patch_reproduction':'NOT_EXECUTED'}
    return {patch_name:patch,'manifest.json':(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()}
