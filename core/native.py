"""Native software identity, staging, builds and owned process execution.

Builds stay outside the checkout. Presence and source identity do not establish
scientific capability; run-specific physics and acceptance remain with callers.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import core.resources as resources
from core.foundation import atomic_json, canonical_digest, file_sha256

ROOT = Path(__file__).resolve().parent


def native_engine_directory(name):
    """Locate an externally built engine; retain legacy discovery during migration."""
    if not isinstance(name, str) or not name.replace('-', '_').isidentifier():
        raise ValueError('Safe native engine name required')
    record_path = resources.runtime_paths().build_root/'native'/('installed-'+name+'.json')
    if not record_path.is_file():
        return ROOT/'software'/name
    record = json.loads(record_path.read_text())
    if record.get('schema') != 'research2-native-install-v1' or record.get('status') != 'COMPLETE':
        raise ValueError('Incomplete external native engine receipt')
    directory = Path(record['directory']).resolve()
    if not directory.is_relative_to(resources.runtime_paths().build_root):
        raise ValueError('Native installation escapes external build root')
    binary = Path(record['executable']).resolve()
    if not binary.is_relative_to(directory):
        raise ValueError('Native executable escapes its installation directory')
    if file_sha256(binary) != record['executable_sha256']:
        raise ValueError('External native executable changed: '+name)
    source = ROOT/'software'/name
    for relative, expected in record['source_hashes'].items():
        path = (source/relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(source.resolve()):
            raise ValueError('Native source receipt escapes its engine')
        if file_sha256(path) != expected:
            raise ValueError('Native source changed since external build: '+name)
    return directory


def external_native_source(name):
    """Stage a source-bound copy for engines whose makefiles write in-tree."""
    if not isinstance(name, str) or not name.replace('-', '_').isidentifier():
        raise ValueError('Safe native engine name required')
    source = (ROOT/'software'/name).resolve()
    if not source.is_relative_to((ROOT/'software').resolve()):
        raise ValueError('Native source escapes the bundled software directory')
    if not source.is_dir():
        raise FileNotFoundError('Missing bundled native source: '+str(source))
    excluded = {'.git', 'build', '__pycache__', 'dist', '.venv'}
    binary_suffixes = {'.o', '.opp', '.a', '.so', '.dylib', '.pyc'}
    def ignored(directory, names):
        result = []
        for item in names:
            path = Path(directory)/item
            if item in excluded or path.suffix in binary_suffixes or item == '.DS_Store':
                result.append(item)
                continue
            if path.is_symlink():
                target = path.resolve(strict=True)
                if not target.is_relative_to(source):
                    raise ValueError('Native source link escapes its engine: '+str(path))
                # The pinned build engines have no directory aliases. Reject
                # them explicitly rather than copy bytes absent from os.walk's
                # non-following inventory, including recursive directory links.
                if target.is_dir():
                    raise ValueError('Native source directory aliases are unsupported: '+str(path))
            if path.is_file():
                with path.open('rb') as stream:
                    magic = stream.read(4)
                if magic in {b'\x7fELF', b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xca\xfe\xba\xbe'} or magic[:2] == b'MZ':
                    result.append(item)
        return result
    files = {}
    for directory, dirs, names in os.walk(source):
        skip = set(ignored(directory, dirs+names))
        dirs[:] = [item for item in dirs if item not in skip]
        for item in names:
            path = Path(directory)/item
            if item not in skip and path.is_file():
                files[path.relative_to(source).as_posix()] = file_sha256(path)
    import platform
    toolchain = {'platform': platform.platform(), 'machine': platform.machine(),
                 'environment': {key: os.environ.get(key) for key in
                     ('CC', 'CXX', 'FC', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS')}}
    for command in ('cc', 'c++', 'mpicc', 'mpicxx', 'cmake', 'make'):
        binary = shutil.which(command)
        if binary:
            toolchain[command] = {'path': str(Path(binary).resolve()), 'sha256': file_sha256(binary)}
    identity = canonical_digest({'sources': files, 'toolchain': toolchain})
    destination = resources.runtime_paths().build_root/'native'/name/identity
    with native_run_lock(destination.parent, operation='native source staging '+name):
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix='.source-stage-', dir=destination.parent))
            shutil.copytree(source, stage, ignore=ignored, dirs_exist_ok=True)
            staged_files = {path.relative_to(stage).as_posix(): file_sha256(path)
                            for path in stage.rglob('*') if path.is_file()}
            if staged_files != files:
                raise ValueError('Native source changed while staging; retained copy: '+str(stage))
            os.rename(stage, destination)
        for relative, expected in files.items():
            if file_sha256(destination/relative) != expected:
                raise ValueError('External build source is changed: '+relative)
    return destination, files


def external_tool_inventory():
    """Record discoverable solver/render tools and their reported versions.

    A failed or unsupported ``--version`` probe remains unknown; executable
    presence alone is not treated as proof of capability. Pinned source
    revisions and build feature flags are recorded separately by run/build
    manifests.
    """
    native_locations, discovery_errors = {}, {}
    for name in ('class', 'axiclass', 'music2', 'monofonic', 'velociraptor',
                 'rockstar', 'consistent-trees', 'disperse'):
        try:
            native_locations[name] = native_engine_directory(name)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            # Inventory describes optional failures. Actual selected consumers
            # still reject an invalid installation through the same resolver.
            native_locations[name] = None
            discovery_errors[name] = str(exc)
    probes = {
        "blender": ("blender", ["--version"]),
        "ffmpeg": ("ffmpeg", ["-version"]),
        "paraview_pvpython": ("pvpython", ["--version"]),
        "disperse_mse": ("mse", ["--version"]),
        "disperse_skelconv": ("skelconv", ["--version"]),
        "rockstar": ("rockstar", ["--version"]),
    }
    tools = {}
    for label, (command, arguments) in probes.items():
        executable = shutil.which(command)
        engine = 'disperse' if label.startswith('disperse_') else label
        directory = native_locations.get(engine)
        if not executable and directory is not None:
            candidate = directory/('build/src/mse/'+command if engine == 'disperse' else command)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                executable = str(candidate)
        if label == "blender" and not executable:
            app_binary = Path("/Applications/Blender.app/Contents/MacOS/Blender")
            if app_binary.is_file():
                executable = str(app_binary)
        if not executable:
            tools[label] = {"status": "MISSING", "command": command, "path": None, "version": None}
            continue
        try:
            result = subprocess.run(
                [executable, *arguments], capture_output=True, text=True,
                timeout=6, check=False,
            )
            lines = [line.strip() for line in (result.stdout + "\n" + result.stderr).splitlines() if line.strip()]
            tools[label] = {
                "status": "VERSION_REPORTED" if lines and result.returncode == 0 else "VERSION_UNKNOWN",
                "command": command,
                "path": str(Path(executable).resolve()),
                "version": " | ".join(lines[:3])[:600] if lines else None,
                "version_probe_exit_code": int(result.returncode),
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            tools[label] = {
                "status": "VERSION_UNKNOWN",
                "command": command,
                "path": str(Path(executable).resolve()),
                "version": None,
                "version_probe_error": type(exc).__name__,
            }

    source_candidates = {
        "gadget4": [ROOT / "software/gadget4"],
        "class": [ROOT / "software/class", ROOT / "software/class_public"],
        "axiclass": [ROOT / "software/axiclass", ROOT / "software/AxiCLASS"],
        "monofonic": [ROOT / "software/monofonic"],
        "music2": [ROOT / "software/music2"],
        "velociraptor": [ROOT / "software/velociraptor"],
        "disperse": [ROOT / "software/disperse"],
        "rockstar": [ROOT / "software/rockstar"],
        "consistent_trees": [ROOT / "software/consistent-trees", ROOT / "software/consistent_trees"],
    }
    declared_revisions = {}
    revisions_path = ROOT / "software-revisions"
    if revisions_path.is_file():
        for line in revisions_path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 3 and not fields[0].startswith("#"):
                revision = {"repository": fields[1], "pinned_commit": fields[2]}
                declared_revisions[fields[0]] = revision
                declared_revisions[fields[0].replace("-", "_")] = revision

    def source_record(name, candidates):
        path = next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])
        available = path.is_dir()
        record = {
            "available": available,
            "path": str(path.resolve()) if available else None,
            **declared_revisions.get(name, {}),
        }
        if not available:
            record["revision_status"] = "SOURCE_MISSING"
            return record
        try:
            commit = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=4, check=False,
            )
            if commit.returncode == 0 and commit.stdout.strip():
                record["checked_out_commit"] = commit.stdout.strip()
                pinned = record.get("pinned_commit")
                record["revision_status"] = (
                    "PIN_MATCH" if pinned and commit.stdout.strip() == pinned
                    else "PIN_MISMATCH" if pinned
                    else "COMMIT_RECORDED"
                )
                dirty = subprocess.run(
                    ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                    capture_output=True, text=True, timeout=4, check=False,
                )
                record["tracked_worktree_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
            else:
                record["revision_status"] = "PINNED_REVISION_DECLARED_SOURCE_NOT_GIT"
        except (OSError, subprocess.TimeoutExpired):
            record["revision_status"] = "REVISION_CHECK_UNAVAILABLE"
        return record

    sources = {name: source_record(name, paths) for name, paths in source_candidates.items()}

    relative_binaries = {
        'class': ['class'], 'axiclass': ['class'], 'music2': ['build/MUSIC'],
        'monofonic': ['build/monofonIC', 'build_gnu_sdk/monofonIC'],
        'velociraptor': ['build/stf'], 'rockstar': ['rockstar', 'rockstar-galaxies'],
        'consistent-trees': ['gravitational_consistency'],
        'disperse': ['build/src/mse/mse', 'build/src/mse/skelconv'],
    }
    binary_candidates = {name.replace('-', '_'):
        [native_locations[name]/relative for relative in relatives] if native_locations[name] is not None else []
        for name, relatives in relative_binaries.items()}

    def binary_record(candidates):
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            return {
                "available": False, "executable": False, "path": None,
                "sha256": None, "version": None, "version_status": "BINARY_MISSING",
            }
        try:
            digest = file_sha256(path)
            return {
                "available": True,
                "executable": os.access(path, os.X_OK),
                "path": str(path.resolve()),
                "sha256": digest,
                "version": None,
                "version_status": "UNKNOWN_BINARY_HAS_NO_STABLE_VERSION_PROBE",
            }
        except OSError as exc:
            return {
                "available": True,
                "executable": os.access(path, os.X_OK),
                "path": str(path.resolve()),
                "sha256": None,
                "version": None,
                "version_status": "BINARY_READ_FAILED: " + type(exc).__name__,
            }

    binaries = {name: binary_record(paths) for name, paths in binary_candidates.items()}
    for name, error in discovery_errors.items():
        binaries[name.replace('-', '_')].update(version_status='INVALID_INSTALLATION', installation_error=error)
    tree_script_paths = {
        "rockstar_config_generator": ('rockstar', 'scripts/gen_merger_cfg.pl'),
        "tree_builder": ('consistent-trees', 'do_merger_tree.pl'),
        "catalogue_converter": ('consistent-trees', 'halo_trees_to_catalog.pl'),
    }
    tree_scripts = {name: binary_record([native_locations[engine]/relative]
                    if native_locations[engine] is not None else [])
                    for name, (engine, relative) in tree_script_paths.items()}
    tree_scripts["perl_interpreter"] = {
        "available": shutil.which("perl") is not None,
        "path": shutil.which("perl"),
    }
    gadget_cache = resources.runtime_paths().build_root / 'gadget4'
    cached_gadget_builds = sorted(
        path for path in gadget_cache.glob("*/Gadget4") if path.is_file()
    ) if gadget_cache.is_dir() else []
    gadget_build_records = []
    ngenic_build_records = []
    for executable_path in cached_gadget_builds:
        signature_path = executable_path.parent / "build_signature.json"
        record = {
            "cache_key": executable_path.parent.name,
            "executable_path": str(executable_path.resolve()),
            "executable_size_bytes": executable_path.stat().st_size,
            "build_signature_path": str(signature_path.resolve()) if signature_path.is_file() else None,
            "signature_status": "MISSING",
            "gadget_revision": None,
            "physics_patch": None,
            "compiled_features": [],
        }
        if signature_path.is_file():
            try:
                signature = json.loads(signature_path.read_text(encoding="utf-8"))
                config_lines = str(signature.get("config", "")).splitlines()
                record.update({
                    "signature_status": "READABLE",
                    "gadget_revision": signature.get("gadget_revision"),
                    "physics_patch": signature.get("darkmattersim_physics_patch"),
                    "compiled_features": config_lines,
                })
                if (any(line.startswith("NGENIC=") for line in config_lines)
                        and "CREATE_GRID" in config_lines):
                    ngenic_build_records.append(record)
            except (OSError, ValueError, TypeError):
                record["signature_status"] = "INVALID"
        gadget_build_records.append(record)
    binaries["gadget4_cached_builds"] = {
        "available": bool(cached_gadget_builds),
        "count": len(cached_gadget_builds),
        "paths": [str(path.resolve()) for path in cached_gadget_builds[:20]],
        "additional_build_count": max(0, len(cached_gadget_builds) - 20),
        "build_records": gadget_build_records[:20],
        "version": None,
        "version_status": "BUILD_SIGNATURES_ARE_RUN_SPECIFIC",
    }
    binaries["ngenic"] = {
        "available": bool(ngenic_build_records),
        "lpt2_available": any("NGENIC_2LPT" in record["compiled_features"] for record in ngenic_build_records),
        "detected_build_count": len(ngenic_build_records),
        "build_records": ngenic_build_records[:20],
        "version": None,
        "version_status": (
            "DETECTED_FROM_GADGET_BUILD_SIGNATURE"
            if ngenic_build_records else "NO_CACHED_NGENIC_BUILD_DETECTED"
        ),
    }
    features = {
        "disperse_filament_pipeline": (
            "EXECUTABLES_PRESENT" if tools["disperse_mse"]["path"] and tools["disperse_skelconv"]["path"]
            else "EXECUTABLES_MISSING"
        ),
        "blender_export_or_render": "EXECUTABLE_PRESENT" if tools["blender"]["path"] else "EXECUTABLE_MISSING",
        "ffmpeg_movie_encoding": "EXECUTABLE_PRESENT" if tools["ffmpeg"]["path"] else "EXECUTABLE_MISSING",
        "paraview_xdmf_rendering": "EXECUTABLE_PRESENT" if tools["paraview_pvpython"]["path"] else "EXECUTABLE_MISSING",
        "gadget4_source_for_new_builds": "SOURCE_PRESENT" if sources["gadget4"]["available"] else "SOURCE_MISSING",
        "class_source_for_new_builds": "SOURCE_PRESENT" if sources["class"]["available"] else "SOURCE_MISSING",
        "axiclass_source_for_new_builds": "SOURCE_PRESENT" if sources["axiclass"]["available"] else "SOURCE_MISSING",
        "rockstar_source": "SOURCE_PRESENT" if sources["rockstar"]["available"] else "SOURCE_MISSING",
        "consistent_trees_source": "SOURCE_PRESENT" if sources["consistent_trees"]["available"] else "SOURCE_MISSING",
        "monofonic_source": "SOURCE_PRESENT" if sources["monofonic"]["available"] else "SOURCE_MISSING",
        "solver_build_features": "CHECKED_PER_RUN_FROM_BUILD_SIGNATURE",
        "n_genic": (
            "NGENIC_2LPT_BUILD_AVAILABLE" if binaries["ngenic"]["lpt2_available"]
            else "NO_CACHED_NGENIC_2LPT_BUILD_DETECTED"
        ),
        "class_binary": "EXECUTABLE_PRESENT" if binaries["class"]["available"] and binaries["class"]["executable"] else "EXECUTABLE_MISSING",
        "axiclass_binary": "EXECUTABLE_PRESENT" if binaries["axiclass"]["available"] and binaries["axiclass"]["executable"] else "EXECUTABLE_MISSING",
        "monofonic_binary": "EXECUTABLE_PRESENT" if binaries["monofonic"]["available"] and binaries["monofonic"]["executable"] else "EXECUTABLE_MISSING",
        "rockstar_binary": "EXECUTABLE_PRESENT" if binaries["rockstar"]["available"] and binaries["rockstar"]["executable"] else "EXECUTABLE_MISSING",
        "consistent_trees_pipeline": (
            "SCRIPTS_AND_PERL_PRESENT"
            if all(tree_scripts[name]["available"] for name in tree_script_paths)
            and tree_scripts["perl_interpreter"]["available"]
            else "SCRIPTS_OR_PERL_MISSING"
        ),
        "gadget4_cached_build": "BUILD_PRESENT" if binaries["gadget4_cached_builds"]["available"] else "BUILD_MISSING",
    }
    return {
        "schema": "darkmattersim-external-tools-v2",
        "tools": tools,
        "sources": sources,
        "binaries": binaries,
        "scripts": {"consistent_trees": tree_scripts},
        "features": features,
    }


def build_engine(name: str, jobs: int) -> None:
    """Build one native executable, without downloading redundant engines.

    monofonIC receives tabulated species transfers from our CLASS run, so its
    embedded CLASS and optional HACC GenericIO download paths are disabled.
    CLASS's executable target avoids installing competing classy Python modules.
    """
    from core.resources import runtime_paths
    if name not in {'class', 'axiclass', 'monofonic', 'music2', 'velociraptor',
                    'disperse', 'rockstar', 'consistent-trees'}:
        raise ValueError('Unknown native build engine: '+str(name))
    if type(jobs) is not int or not 1 <= jobs <= resources.cpu_thread_budget():
        raise ValueError('Build jobs must be a positive integer within the allocated CPU budget')
    # Source staging uses this same reentrant owner. A second build must not
    # clean or reconfigure files while the first compiler is using them.
    with native_run_lock(runtime_paths().build_root/'native'/name, operation='native engine build '+name):
        source, source_hashes = external_native_source(name)
        if not source.is_dir():
            raise FileNotFoundError("Missing bundled source: " + str(source))
        with native_run_lock(source, operation='native source/executable build '+name):
            def build_step(command, label):
                # Compilers inherit the source owner, including when the Python
                # wrapper is killed. The shared runner cleans owned descendants.
                return run_native_command(
                    command, cwd=source,
                    receipt_path=source/'build-invocations'/(label+'.json'),
                    check=True, infer_argument_inputs=False,
                )

            if name in {"class", "axiclass"}:
                build_step(["make", "clean"], 'clean')
                build_step(["make", f"-j{jobs}", "class"], 'compile')
                binary = source / "class"
            elif name in {"monofonic", "music2", "velociraptor", "disperse"}:
                build = source / "build"
                # Ubuntu 22.04's CMake predates --fresh. Reset only generated CMake
                # configuration before configuring the dedicated engine build folder.
                (build / "CMakeCache.txt").unlink(missing_ok=True)
                if (build / "CMakeFiles").is_dir():
                    shutil.rmtree(build / "CMakeFiles")
                command = ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"]
                if name == 'disperse':
                    command += ['-DCMAKE_POLICY_VERSION_MINIMUM=3.5', '-DNO_CGAL=true', '-DNO_SDL=true',
                        '-DNO_MATHGL=true', '-DNO_QT=true', '-DUSE_TR1=false', '-DCMAKE_CXX_FLAGS=-std=c++11']
                    try:
                        gsl_prefix = subprocess.check_output(['pkg-config', '--variable=prefix', 'gsl'], text=True).strip()
                    except Exception:
                        gsl_prefix = '/usr'
                    command.append('-DGSL_DIR=' + gsl_prefix)
                    cfitsio_prefix = None
                    try:
                        cfitsio_prefix = subprocess.check_output(['pkg-config', '--variable=prefix', 'cfitsio'], text=True).strip()
                    except Exception:
                        for cand in ['/usr', '/usr/local', os.path.expanduser('~/.local')]:
                            if (Path(cand)/'include/fitsio.h').is_file() or (Path(cand)/'include/cfitsio/fitsio.h').is_file():
                                cfitsio_prefix = cand
                                break
                    if cfitsio_prefix:
                        command.append('-DCFITSIO_DIR=' + cfitsio_prefix)
                    else:
                        command.append('-DNO_CFITSIO=true')
                if name == "monofonic":
                    command += ["-DENABLE_CLASS=OFF", "-DENABLE_GENERICIO=OFF", "-DENABLE_MPI=ON", "-DENABLE_PANPHASIA=OFF"]
                elif name == "music2":
                    command += ["-DENABLE_CLASS=OFF", "-DENABLE_PANPHASIA=OFF", "-DMUSIC_ENABLE_WIDE_MIXED_POTENTIAL=ON"]
                elif name == "velociraptor":
                    # Single measured host; exact-neighbor density over every star.
                    # Optional STARON population fields are excluded, not fabricated.
                    command += ["-DCMAKE_POLICY_VERSION_MINIMUM=3.5", "-DBUILD_TESTING=OFF",
                                "-DVR_MPI=OFF", "-DVR_OPENMP=ON", "-DVR_HDF5=ON",
                                "-DVR_ALLOWPARALLELHDF5=OFF", "-DVR_USE_STAR=OFF",
                                "-DVR_GADGET_LONGID=ON", "-DVR_LONG_INT=ON",
                                "-DVR_STRUCTURE_DEN=OFF", "-DVR_HALO_DEN=OFF",
                                "-DNBODY_SINGLE_PRECISION=OFF", "-DNBODY_SINGLE_PARTICLE_PRECISION=OFF"]
                    serial_hdf5 = Path("/usr/bin/h5cc.serial")
                    if not serial_hdf5.is_file():
                        serial_hdf5 = Path("/usr/bin/h5cc")
                    if sys.platform.startswith("linux") and serial_hdf5.is_file():
                        command += ["-DHDF5_C_COMPILER_EXECUTABLE=" + str(serial_hdf5)]
                # Apple Clang lacks a bundled OpenMP runtime. Respect explicit CC/CXX;
                # otherwise use Homebrew GCC, whose runtime is already integrated.
                if sys.platform == "darwin" and not os.environ.get("CC") and not os.environ.get("CXX"):
                    candidates = sorted(Path("/opt/homebrew/bin").glob("gcc-[0-9]*"), key=lambda p: int(p.name.split("-")[-1]))
                    if candidates:
                        cc = candidates[-1]
                        cxx = cc.with_name("g++-" + cc.name.split("-")[-1])
                        command += ["-DCMAKE_C_COMPILER=" + str(cc), "-DCMAKE_CXX_COMPILER=" + str(cxx)]
                if name == "music2":
                    command.append("-DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs")
                build_step(command, 'configure')
                targets = ['--target', 'mse', 'skelconv'] if name == 'disperse' else []
                build_step(["cmake", "--build", str(build), *targets, "--parallel", str(jobs)], 'compile')
                binary = build / {"monofonic": "monofonIC", "music2": "MUSIC", "velociraptor": "stf", 'disperse': 'src/mse/mse'}[name]
                if name == 'music2':
                    from core.music.precision import prepare_binary128_reference
                    prepare_binary128_reference()
            elif name in {"rockstar", "consistent-trees"}:
                command = ["make", f"-j{jobs}"]
                if name == "rockstar":
                    package = "hdf5"
                    if subprocess.run(["pkg-config", "--exists", package], check=False).returncode:
                        package = "hdf5-openmpi"
                    flags = subprocess.check_output(["pkg-config", "--cflags", "--libs", package], text=True).strip()
                    # Modern Linux places Sun RPC headers/library in libtirpc.
                    if sys.platform.startswith("linux"):
                        flags += " " + subprocess.check_output(
                            ["pkg-config", "--cflags", "--libs", "libtirpc"], text=True,
                        ).strip()
                    command += ["with_hdf5", "HDF5_FLAGS=-DH5_USE_16_API -DENABLE_HDF5 " + flags]
                if name == "consistent-trees" and sys.platform == "darwin":
                    compilers = [p for p in Path("/opt/homebrew/bin").glob("gcc-*") if p.name.removeprefix("gcc-").isdigit()]
                    if not compilers:
                        raise RuntimeError("Consistent Trees requires GNU GCC with OpenMP on macOS")
                    command.append("CC=" + str(max(compilers, key=lambda p: int(p.name.split("-")[-1]))))
                build_step(command, 'compile')
                if name == "rockstar":
                    build_step(["make", f"-j{jobs}", "parents"], 'parents')
                binary = source / ("rockstar" if name == "rockstar" else "gravitational_consistency")
            else:
                raise ValueError("Unknown engine: " + name)
            if not binary.is_file():
                raise RuntimeError("Build did not create " + str(binary))
            atomic_json(runtime_paths().build_root/'native'/('installed-'+name+'.json'), {
                'schema': 'research2-native-install-v1', 'status': 'COMPLETE',
                'directory': str(source), 'executable': str(binary), 'executable_sha256': file_sha256(binary),
                'source_hashes': source_hashes, 'science_acceptance': 'NOT_ESTABLISHED'})
            print("Built " + str(binary))


# Kernel ownership uses persistent lock inodes; diagnostic receipts are separate.

_held={}
_guard=threading.RLock()


def _native_run_lock_paths(run_dir):
    root=Path(run_dir).expanduser().resolve()
    key=hashlib.sha256(str(root).encode()).hexdigest()
    directory=root.parent/'.native-run-locks'
    return root,directory/(key+'.lock'),directory/(key+'.json')


@contextmanager
def native_run_lock(run_dir, *, operation='native workflow'):
    """Fail promptly if another process/thread owns this run, including symlink aliases.

    Lock files are outside the run so initial preparation still requires an empty
    run directory. Kernel ownership, rather than the diagnostic PID, is authoritative.
    """
    import fcntl
    root,path,receipt=_native_run_lock_paths(run_dir)
    owner=(os.getpid(),threading.get_ident())
    key=str(root)
    with _guard:
        current=_held.get(key)
        if current and current['owner']==owner:
            current['depth']+=1
        else:
            if current and current['owner'][0]==owner[0]:
                raise RuntimeError('Native run is already owned by another thread: '+key)
            path.parent.mkdir(parents=True,exist_ok=True)
            fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
            try:
                try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    try:details=json.loads(receipt.read_text())
                    except (OSError,ValueError):details={'status':'OWNER_DETAILS_UNAVAILABLE'}
                    raise RuntimeError('Native run is already owned: '+str(root)+'; owner='+json.dumps(details,sort_keys=True)) from exc
                metadata={'schema':'native-run-owner-v1','status':'WRAPPER_ACTIVE','run_dir':key,
                          'pid':owner[0],'thread_id':owner[1],'hostname':socket.gethostname(),'operation':operation,
                          'started_utc':datetime.now(timezone.utc).isoformat(),
                          'ownership_authority':'Kernel flock; diagnostic PID is not proof of liveness'}
                atomic_json(receipt,metadata)
                _held[key]={'fd':fd,'owner':owner,'depth':1,'receipt':receipt,'metadata':metadata}
            except BaseException:
                os.close(fd);raise
    try:
        yield
    finally:
        with _guard:
            current=_held[key]
            current['depth']-=1
            if current['depth']==0:
                _held.pop(key)
                try:
                    metadata=dict(current['metadata']);metadata.update(status='WRAPPER_RELEASED',finished_utc=datetime.now(timezone.utc).isoformat())
                    atomic_json(current['receipt'],metadata)
                finally:
                    # Close rather than LOCK_UN: inherited native child descriptors
                    # must retain ownership if the wrapper is interrupted or killed.
                    os.close(current['fd'])


def native_run_lock_fds(run_dir):
    root,_,_=_native_run_lock_paths(run_dir)
    with _guard:
        current=_held.get(str(root))
        return (current['fd'],) if current and current['owner']==(os.getpid(),threading.get_ident()) else ()


def native_owned_fds():
    """Pass all nested current-thread owners to children of an isolated wrapper."""
    owner=(os.getpid(),threading.get_ident())
    with _guard:return tuple(sorted({record['fd'] for record in _held.values() if record['owner']==owner}))


def native_run_owner(function):
    """Keep existing public signatures and nested native calls under one owner."""
    signature=inspect.signature(function)
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        root=signature.bind(*args,**kwargs).arguments['run_dir']
        with native_run_lock(root,operation=function.__name__):
            return function(*args,**kwargs)
    return wrapped


def directory_owner(parameter, *, default_relative=None):
    """Reuse the same kernel ownership for non-run artifact producer/consumer roots."""
    def decorate(function):
        signature=inspect.signature(function)
        @functools.wraps(function)
        def wrapped(*args,**kwargs):
            bound=signature.bind(*args,**kwargs);bound.apply_defaults()
            root=bound.arguments[parameter]
            if root is None and default_relative is not None:
                root=Path(bound.arguments['run_dir'])/default_relative
            if root is None:raise ValueError('Artifact ownership requires a destination directory')
            with native_run_lock(root,operation=function.__name__):return function(*args,**kwargs)
        return wrapped
    return decorate

def stop_owned_process_groups(processes,*,grace_seconds=2.):
    """Terminate only new-session groups created by this wrapper, then reap parents."""
    processes=[process for process in processes if process is not None]
    for process in processes:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+grace_seconds
    remaining=list(processes)
    while remaining and time.monotonic()<deadline:
        live=[]
        for process in remaining:
            process.poll()
            try:os.killpg(process.pid,0)
            except ProcessLookupError:continue
            live.append(process)
        remaining=live
        if remaining:time.sleep(.05)
    for process in remaining:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    for process in processes:
        try:process.wait(timeout=2.)
        except subprocess.TimeoutExpired:raise RuntimeError('Owned native parent did not exit after SIGKILL')


def invocation_record(command,cwd,*,inputs=(),environment=None,infer_argument_inputs=True):
    """Bind the actual executable and explicit input files before spawning it."""
    if isinstance(command, (str, bytes)):
        raise ValueError('Native command must be an argument list')
    command = [str(value) for value in command]
    if not command or not command[0]:
        raise ValueError('Native command requires an executable')
    cwd = Path(cwd).resolve()
    supplied = os.environ if environment is None else environment
    if '/' in command[0]:
        executable = Path(command[0])
        if not executable.is_absolute():
            executable = cwd/executable
    else:
        # Popen resolves PATH after changing to the child's working directory.
        # Resolve relative PATH entries there too, using the child's environment.
        search = os.pathsep.join(str((cwd/entry).resolve()) for entry in os.get_exec_path(supplied))
        selected = shutil.which(command[0], path=search)
        if selected is None:
            raise FileNotFoundError('Native executable missing from child PATH: '+command[0])
        executable = Path(selected)
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError('Native executable missing: '+str(executable))
    files = [executable]
    # Normal solver arguments often name input files. Build target names can
    # name existing outputs, so builds explicitly disable this inference.
    for value in command[1:] if infer_argument_inputs else ():
        path = Path(value)
        if not path.is_absolute():
            path = cwd/path
        if path.is_file():
            files.append(path.resolve())
    for value in inputs:
        path = Path(value)
        if not path.is_absolute():
            path = cwd/path
        if not path.is_file():
            raise FileNotFoundError('Declared native input missing: '+str(path))
        files.append(path.resolve())
    # Record only declared scientific inputs, never the full process environment.
    scientific_environment={}
    for name in ('RESEARCH_SIDM_REFERENCE_TABLE','RESEARCH_BH_WAKE_DIAGNOSTIC_SCHEDULE'):
        if name not in supplied:continue
        value=supplied[name]
        if not isinstance(value,str) or not value:raise ValueError('Scientific environment input needs a nonempty path: '+name)
        path=Path(value)
        if not path.is_absolute():path=cwd/path
        path=path.resolve()
        if not path.is_file():raise FileNotFoundError('Scientific environment input missing: '+str(path))
        files.append(path);scientific_environment[name]=str(path)
    return {'schema':'native-command-invocation-v1','status':'INVOCATION_RECORDED',
        'recorded_utc':datetime.now(timezone.utc).isoformat(),'command':command,'cwd':str(cwd),
        'resolved_executable':str(executable),
        'input_hashes':{str(path):file_sha256(path) for path in dict.fromkeys(files)},
        'scientific_environment':scientific_environment,
        'build_provenance_status':'UNDECLARED; executable bytes measured, compiler/source equivalence not inferred',
        'wrapper_algorithm_sha256':file_sha256(__file__),'science_acceptance':'NOT_ACCEPTED'}


def verify_invocation_inputs(record):
    for path,checksum in record['input_hashes'].items():
        if file_sha256(path)!=checksum:raise ValueError('Native invocation input changed: '+path)


def run_native_command(command,*,cwd,receipt_path,inputs=(),check=True,timeout=None,process_observer=None,infer_argument_inputs=True,**options):
    """Retain spawn/return/failure evidence and clean descendants on every exit."""
    if options.get('shell') or options.get('executable') is not None:
        raise ValueError('Tracked native commands require their executable as the first argument')
    if options.get('start_new_session') is False:raise ValueError('Tracked native commands require owned new sessions')
    options['start_new_session']=True
    options['pass_fds']=tuple(sorted(set(options.get('pass_fds',()))|set(native_owned_fds())))
    record=invocation_record(command,cwd,inputs=inputs,environment=options.get('env'),
                             infer_argument_inputs=infer_argument_inputs)
    atomic_json(receipt_path,record)
    command = record['command']
    execution_command = [record['resolved_executable'], *command[1:]]
    process=None
    try:
        process=subprocess.Popen(execution_command,cwd=record['cwd'],**options)
        record.update(status='SPAWNED',pid=process.pid);atomic_json(receipt_path,record)
        if process_observer is not None:process_observer.start(process.pid)
        code=process.wait(timeout=timeout)
        verify_invocation_inputs(record)
        record.update(status='RETURNED',returncode=code,finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(receipt_path,record)
        if check and code:raise subprocess.CalledProcessError(code,command)
        return subprocess.CompletedProcess(command,code)
    except BaseException as exc:
        record.update(status='FAILED',failure={'type':type(exc).__name__,'message':str(exc)})
        atomic_json(receipt_path,record)
        raise
    finally:
        try:stop_owned_process_groups([process])
        finally:
            if process_observer is not None:process_observer.stop()



def verified_python_backend(name, record, *, distribution=None):
    """Verify a pinned VCS wheel or bootstrap's source-bound external installation.

    Local wheels need the external install receipt plus exact declared source
    and installed-file inventories. A local direct_url alone is insufficient.
    Every returned source is tied to bytes on the current host.
    """
    import base64
    import importlib.metadata
    import sys
    dist = distribution or importlib.metadata.distribution(name)
    origin = json.loads(dist.read_text('direct_url.json') or '{}')
    prefix = Path(sys.prefix).resolve()
    installed = {}
    for entry in dist.files or []:
        path = Path(dist.locate_file(entry)).resolve()
        if not path.is_file() or path.suffix == '.pyc':
            continue
        expected = getattr(entry, 'hash', None)
        if expected is not None:
            check = hashlib.new(expected.mode)
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    check.update(block)
            value = base64.urlsafe_b64encode(check.digest()).decode().rstrip('=')
            if value != expected.value:
                raise ValueError('Installed package RECORD mismatch: '+str(path))
        installed[str(path)] = file_sha256(path)
    if not installed:
        raise ValueError('Pinned backend has no installed-file inventory: '+name)
    revision = record['revision']
    installed_assets = dict(installed)
    if origin.get('vcs_info', {}).get('commit_id') == revision:
        return {'revision':revision, 'origin':'pinned_vcs', 'source_hashes':installed,
                'installed_assets':installed_assets}
    if origin.get('dir_info', {}).get('editable'):
        raise ValueError('Portable backend receipts require installed wheel bytes, not editable sources')
    if any(not Path(path).is_relative_to(prefix) for path in installed_assets):
        raise ValueError('Installed backend asset escapes its external environment')
    receipt_path = prefix/'installation.json'
    if prefix.is_relative_to(ROOT.parent.resolve()) or not receipt_path.is_file():
        raise ValueError('Backend requires pinned VCS provenance or an external bootstrap receipt: '+name)
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('schema') != 'research2-installed-environment-v1'
            or Path(receipt.get('environment_root', '')).resolve() != prefix):
        raise ValueError('Backend installation receipt belongs to another environment')
    packages = [item for item in receipt.get('bundled_packages', [])
                if item['name'].lower().replace('_', '-') == name.lower().replace('_', '-')]
    if len(packages) != 1:
        raise ValueError('Pinned backend needs exactly one installed source record: '+name)
    binding = packages[0]
    declarations_path = (ROOT.parent/'configs/runtime/dependencies.json').resolve()
    declarations = json.loads(declarations_path.read_text())
    declared = [item for group in declarations['profiles'].values() for item in group['packages']
                if item.get('name', '').lower().replace('_', '-') == name.lower().replace('_', '-')]
    if len(declared) != 1 or any(binding.get(key) != declared[0].get(key)
            for key in ('revision', 'source_path', 'source_sha256')):
        raise ValueError('Backend install no longer matches the portable dependency declaration')
    if binding['revision'] != revision or binding['source_path'] != record['directory']:
        raise ValueError('Backend source revision or package root differs from the scientific pin')
    if any(binding['source_sha256'].get(key) != value for key,value in record['source_sha256'].items()):
        raise ValueError('Backend source receipt omits pinned scientific source assets')
    source = (ROOT.parent/binding['source_path']).resolve()
    for relative, expected in binding['source_sha256'].items():
        path = (source/relative).resolve()
        if not path.is_relative_to(source) or file_sha256(path) != expected:
            raise ValueError('Pinned backend source changed: '+relative)
        installed[str(path)] = expected
    asset_binding = binding.get('installed_asset_sha256', {})
    actual_assets = {}
    for path, value in list(installed.items()):
        p = Path(path)
        if p.is_relative_to(prefix):
            actual_assets[p.relative_to(prefix).as_posix()] = value
    if not asset_binding or actual_assets != asset_binding:
        raise ValueError('Backend installed assets differ from the bootstrap receipt')
    installed[str(receipt_path)] = file_sha256(receipt_path)
    installed[str(declarations_path)] = file_sha256(declarations_path)
    return {'revision':revision, 'origin':'verified_external_bootstrap', 'source_hashes':installed,
            'installed_assets':installed_assets}


def native_build_input_paths(engine_root):
    """Enumerate GADGET build inputs identically for builds and current receipts.

    Typed restart .inc fragments and Random123 headers affect executable state
    just as translation units do. Generated bytecode/build files are excluded.
    Returned paths retain source-relative names for reproducible build hashing.
    """
    root = Path(engine_root).resolve()
    suffixes = {'.c','.cc','.cpp','.h','.hpp','.inc'}
    inputs = [p for p in (root/'src').rglob('*') if p.is_file() and p.suffix in suffixes]
    inputs += [p for p in (root/'buildsystem').rglob('*') if p.is_file()
               and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo','.o','.so'}]
    inputs += [p for p in (root/'third_party/random123').rglob('*') if p.is_file()
               and (p.suffix in {'.h','.hpp'} or p.name.upper().startswith(('LICENSE','COPYING')))]
    template = root/'src/gitversion/version'
    if template.is_file():inputs.append(template)
    inputs += [root/name for name in ('Makefile','Makefile.systype','Template-Config.sh')
               if (root/name).is_file()]
    if any(not p.resolve().is_relative_to(root) for p in inputs):
        raise ValueError('Native build input escapes the engine source tree')
    return sorted(set(inputs))


# Necessary native particle-array capacity before any solver launch.

def native_particle_index_requirement(count):
    """Return the necessary rank floor imposed by native local array indices.

    If f is ALLOC_TOLERANCE, usable balanced capacity per rank is
    floor((2^31-1)*(1-2*f)); ranks must be at least ceil(N/capacity).
    Counts are exact integers, and Fraction preserves the allocation reserve
    without floating rounding. This is an index-capacity limit, not a RAM,
    domain-balance or scheduler-feasibility estimate.
    """
    import re
    from fractions import Fraction
    from .foundation import file_sha256
    if type(count) is not int or count<=0:raise ValueError('Exact positive native particle count required')
    root=Path(__file__).resolve().parent/'software/gadget4/src'
    constants=root/'data/constants.h';declarations=root/'data/simparticles.h';allocator=root/'io/snap_io.cc'
    match=re.search(r'^#define ALLOC_TOLERANCE ([0-9.]+)\s*$',constants.read_text(),re.MULTILINE)
    if not match or not re.search(r'\bint\s+NumPart\s*;',declarations.read_text()) or 'max_load / (1.0 - 2 * ALLOC_TOLERANCE)' not in allocator.read_text():
        raise ValueError('Native particle storage/allocation contract changed; recalculate rank requirements')
    tolerance=Fraction(match.group(1));retained=1-2*tolerance
    if not 0<retained<1:raise ValueError('Native allocation reserve is outside supported bounds')
    integer_max=2**31-1;capacity=integer_max*retained.numerator//retained.denominator
    return {'count':count,'signed_local_index_bits':32,'signed_local_index_max':integer_max,
        'native_allocation_tolerance':float(tolerance),'maximum_balanced_live_particles_per_rank':capacity,
        'minimum_mpi_ranks':(count+capacity-1)//capacity,
        'source_hashes':{str(path):file_sha256(path) for path in (Path(__file__).resolve(),constants,declarations,allocator)},
        'scope':'Necessary signed-int particle-array capacity floor with native allocation reserve; assumes 32-bit int toolchain. Domain imbalance, tree indices, gas growth, MPI buffers and per-node memory can require more ranks.'}



class NativeParticleCapacityError(MemoryError):
    def __init__(self,report):
        self.report=report
        super().__init__('Native particle-index capacity rejected: '+str(report['total_particle_count'])+
            ' particles on '+str(report['mpi_ranks'])+' MPI ranks require at least '+
            str(report['requirement']['minimum_mpi_ranks'])+' ranks for particle-array capacity; configuration unchanged')



def validate_native_particle_capacity(census,ranks):
    if type(ranks) is not int or ranks<1:raise ValueError('Exact positive MPI rank allocation required')
    counts=census.get('particles_by_type')
    if census.get('status') not in {'CENSUS_COMPLETE','PLANNED_UNIFORM_IC'} or not isinstance(counts,list) or len(counts)!=6 or any(type(n) is not int or n<0 for n in counts):
        raise ValueError('Complete six-species IC census or explicit uniform IC recipe count required')
    requirement=native_particle_index_requirement(sum(counts))
    passed=ranks>=requirement['minimum_mpi_ranks']
    report={'schema':'native-particle-capacity-preflight-v1','status':'NECESSARY_CAPACITY_PASS' if passed else 'FAIL',
        'particles_by_type':counts,'total_particle_count':sum(counts),'mpi_ranks':ranks,
        'count_origin':census['status'],'native_execution_performed':False,
        'requirement':requirement,'ic_files':census.get('files',[]),
        'scope':'Necessary particle-array signed-int capacity only. Does not certify domain imbalance, tree indexing, source buffers, memory, scheduler allocation or physics.',
        'science_acceptance':'NOT_ACCEPTED'}
    if not passed:raise NativeParticleCapacityError(report)
    return report



def preflight_native_ic_capacity(paths,ranks):
    from core.resources import initial_condition_census
    return validate_native_particle_capacity(initial_condition_census(paths),ranks)



def original_native_snapshot_files(root,snapshot):
    """Find original data/snapshot_NNN HDF5 shards in native file order.

    Restart and source-conservation intake use these original particle files;
    group-ordered catalogue exports have a separate selection path.
    """
    from .analysis import find_hdf5_files
    files = find_hdf5_files(Path(root)/'data','snapshot_%03d'%snapshot)
    if not files:
        raise FileNotFoundError('Original saved native particle snapshot required')
    return files
