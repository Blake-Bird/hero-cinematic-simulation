"""Stdlib-only setup: python3 bootstrap.py --install --profile base.
Writable environments and builds belong to external RESEARCH2_STATE_ROOT.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

sys.dont_write_bytecode = True
SOURCE_ROOT = Path(__file__).resolve().parent


def paths(state_root=None):
    value = state_root or os.environ.get('RESEARCH2_STATE_ROOT')
    state = Path(value).expanduser().resolve() if value else SOURCE_ROOT.parent/'research2-runtime'
    if state.is_relative_to(SOURCE_ROOT):
        raise ValueError('Runtime state must be outside the source tree')
    return state, state/'environments/default'


def dependency_plan(profile):
    record = json.loads((SOURCE_ROOT/'configs/runtime/dependencies.json').read_text())
    if record.get('schema') != 'research2-dependencies-v1':
        raise ValueError('Unknown dependency schema')
    groups = record['profiles']
    selected = list(groups) if profile == 'all' else [profile]
    active, done, packages = set(), set(), []
    def visit(name):
        if name in active or name not in groups:
            raise ValueError('Unknown or cyclic dependency profile: '+name)
        if name in done:
            return
        active.add(name)
        for dependency in groups[name]['dependencies']:
            visit(dependency)
        packages.extend(groups[name]['packages'])
        active.remove(name)
        done.add(name)
    for name in selected:
        visit(name)
    return packages, record


def check_python_profile(profile):
    """Verify selected installed distributions and exact bundled provenance.

    Called only inside the selected external environment. This checks package
    versions and installed/source bytes; native scientific acceptance is separate.
    The launcher remains stdlib-only until it reaches this explicit check.
    """
    import importlib.metadata
    from packaging.requirements import Requirement
    from packaging.version import Version
    from core.native import verified_python_backend
    packages, declaration = dependency_plan(profile)
    checks = []
    for item in packages:
        row = {'requirement':item.get('requirement', item.get('name')), 'status':'PASS'}
        try:
            if 'requirement' in item:
                requirement = Requirement(item['requirement'])
                if requirement.marker and not requirement.marker.evaluate():
                    row['status'] = 'NOT_APPLICABLE'
                else:
                    version = importlib.metadata.version(requirement.name)
                    row['installed_version'] = version
                    if version not in requirement.specifier:
                        raise ValueError('Installed version violates selected requirement')
            else:
                row['installed_version'] = importlib.metadata.version(item['name'])
                expected_version = item.get('build_environment',{}).get(
                    'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_'+item['name'].upper().replace('-','_'))
                if expected_version is not None and Version(row['installed_version']) != Version(expected_version):
                    raise ValueError('Installed version differs from declared pinned-source SCM version')
                binding = verified_python_backend(item['name'], {
                    'revision':item['revision'], 'directory':item['source_path'],
                    'source_sha256':item['source_sha256']})
                row.update(revision=binding['revision'], provenance_origin=binding['origin'],
                    verified_source_count=len(binding['source_hashes']))
        except (ValueError, OSError, KeyError, importlib.metadata.PackageNotFoundError) as error:
            row.update(status='FAIL', reason=str(error))
        checks.append(row)
    return {'schema':'research2-python-profile-check-v1', 'profile':profile,
        'status':'FAIL' if any(row['status']=='FAIL' for row in checks) else 'PASS',
        'environment_root':str(Path(sys.prefix).resolve()), 'checks':checks,
        'declarations_sha256':hashlib.sha256(json.dumps(declaration,sort_keys=True).encode()).hexdigest(),
        'scope':'Installed package versions and source-bound wheel bytes only; optional runtime model assets and native science controls are separate',
        'science_acceptance':'NOT_ESTABLISHED'}


def install_packages(python, packages, state, env):
    bundled = []
    for item in packages:
        if 'requirement' in item:
            target = item['requirement']
        else:
            source = (SOURCE_ROOT/item['source_path']).resolve()
            if not source.is_relative_to(SOURCE_ROOT) or not source.is_dir():
                raise ValueError('Missing bundled source: '+str(source))
            if not any((source/name).is_file() for name in ('pyproject.toml', 'setup.py')):
                raise ValueError('Bundled Python dependency must name its package root: '+str(source))
            for relative, expected in item['source_sha256'].items():
                path = (source/relative).resolve()
                if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError('Bundled dependency source changed: '+str(path))
            identity = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
            build_root = state/'builds/python'/identity
            build_root.mkdir(parents=True, exist_ok=True)
            copy = Path(tempfile.mkdtemp(dir=build_root, prefix='install-'))/source.name
            shutil.copytree(source, copy, ignore=shutil.ignore_patterns('.git', '__pycache__', 'build', 'dist', '*.pyc', '*.o'))
            # Upstream build metadata may require a README after documentation
            # was intentionally excluded from the portable source release.
            # Supply neutral build-only metadata in this temporary copy.
            for readme_name in ('README','README.md','README.rst'):
                readme=copy/readme_name
                if not readme.exists():readme.write_text('Source distribution of '+item['name']+' at pinned revision '+item['revision']+'. Runtime source and licenses are retained.\n')
            target = str(copy)
            installation = {'name': item['name'], 'revision': item['revision'],
                'source_path': item['source_path'], 'installed_from': target,
                'source_sha256': item['source_sha256'], 'upstream_manifest': item['upstream_manifest'],
                'build_environment':item.get('build_environment', {})}
        options = item.get('pip_options', [])
        if not isinstance(options, list) or any(not isinstance(value, str) for value in options):
            raise ValueError('Dependency pip options must be an argument list')
        build_environment = item.get('build_environment', {})
        if (not isinstance(build_environment, dict) or any(not key.startswith('SETUPTOOLS_SCM_PRETEND_VERSION_FOR_')
                or not isinstance(value, str) for key,value in build_environment.items())):
            raise ValueError('Only declared package SCM version overrides are supported')
        package_env = dict(env, **build_environment)
        if sys.platform == 'darwin' and item.get('name') in {'pycorr', 'pyrecon', 'Corrfunc'}:
            # Apple Clang needs the separately installed OpenMP headers and
            # runtime. Preserve caller flags and use pkg-config's real prefix.
            probe = subprocess.run(['pkg-config', '--variable=prefix', 'libomp'],
                                   capture_output=True, text=True, check=False)
            prefix = Path(probe.stdout.strip()) if probe.returncode == 0 and probe.stdout.strip() else Path('/opt/homebrew/opt/libomp')
            if (prefix/'include/omp.h').is_file():
                for key, flag in [('CPPFLAGS', '-I'+str(prefix/'include')),
                                  ('LDFLAGS', '-L'+str(prefix/'lib'))]:
                    package_env[key] = ' '.join(filter(None, [package_env.get(key), flag]))
        subprocess.run([str(python), '-m', 'pip', 'install', *options, target], check=True, env=package_env)
        if 'source_path' in item:
            # Record installed bytes using the target interpreter's distribution inventory.
            script = '''import hashlib, importlib.metadata, json, pathlib, sys
root = pathlib.Path(sys.prefix).resolve()
distribution = importlib.metadata.distribution(sys.argv[1])
assets = {}
for entry in distribution.files or []:
    path = pathlib.Path(distribution.locate_file(entry)).resolve()
    if not path.is_file() or path.suffix == '.pyc':
        continue
    if not path.is_relative_to(root):
        raise ValueError('Installed dependency asset escapes its environment')
    assets[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
if not assets:
    raise ValueError('Installed dependency has no recorded files')
print(json.dumps(assets))
'''
            assets = subprocess.check_output([str(python), '-B', '-c', script, item['name']],
                                             text=True, env=env)
            installation['installed_asset_sha256'] = json.loads(assets)
            bundled.append(installation)
    return bundled


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--profile', default='base', help='Dependency profile from configs/runtime/dependencies.json, or all')
    parser.add_argument('--state-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check', action='store_true', help='Verify the selected Python profile and host build prerequisites')
    parser.add_argument('--check-python-profile', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--build-all', action='store_true')
    args, forwarded = parser.parse_known_args(argv)
    if os.name == 'nt':
        parser.error('Use Ubuntu on WSL2; native Windows is unsupported')
    if sys.version_info < (3, 10):
        parser.error('Python 3.10 or newer required')
    state, environment = paths(args.state_root)
    packages, declaration = dependency_plan(args.profile)
    if args.check_python_profile:
        if args.install or args.dry_run or args.check or args.build_all or forwarded:
            parser.error('Internal profile check cannot be combined with execution actions')
        if Path(sys.prefix).resolve() != environment.resolve():
            parser.error('Profile verification must run inside the selected external environment')
        report = check_python_profile(args.profile)
        print(json.dumps(report,indent=2))
        return 0 if report['status']=='PASS' else 1
    if args.dry_run:
        print(json.dumps({'state_root': str(state), 'environment': str(environment), 'packages': packages}, indent=2))
        return 0
    python = environment/'bin/python'
    env = dict(os.environ, RESEARCH2_STATE_ROOT=str(state), PYTHONDONTWRITEBYTECODE='1', PIP_CACHE_DIR=str(state/'cache/pip'))
    if args.install:
        if not python.is_file():
            venv.EnvBuilder(with_pip=True).create(environment)
        else:
            details = json.loads(subprocess.check_output([str(python), '-B', '-c',
                'import sys,json; print(json.dumps({"prefix":sys.prefix,"version":list(sys.version_info[:3])}))'], text=True))
            if Path(details['prefix']).resolve() != environment.resolve() or details['version'] < [3, 10]:
                raise ValueError('Existing runtime environment has an incompatible interpreter')
        temporary = state/'cache/tmp'
        temporary.mkdir(parents=True, exist_ok=True)
        env['TMPDIR'] = str(temporary)
        destination = environment/'installation.json'
        previous = json.loads(destination.read_text()) if destination.is_file() else {}
        if previous:
            if (previous.get('schema') != 'research2-installed-environment-v1'
                    or previous.get('environment_root') is not None
                    and Path(previous['environment_root']).resolve() != environment.resolve()):
                raise ValueError('Existing installation receipt has a different schema or environment')
            history = state/'evidence/dependencies'
            history.mkdir(parents=True, exist_ok=True)
            checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
            saved = history/(checksum+'.json')
            if not saved.exists():
                shutil.copy2(destination, saved)
        for package_index, package in enumerate(packages):
            bundled = install_packages(python, [package], state, env)
            merged = {item['name'].lower().replace('_', '-'):item for item in previous.get('bundled_packages', [])}
            for item in bundled:
                merged[item['name'].lower().replace('_', '-')] = item
            bundled = list(merged.values())
            frozen = subprocess.check_output([str(python), '-m', 'pip', 'freeze', '--all'], text=True, env=env)
            actual_python = json.loads(subprocess.check_output([str(python), '-B', '-c',
                'import sys,json,platform; print(json.dumps({"version":list(sys.version_info[:3]),"implementation":sys.implementation.name,"machine":platform.machine()}))'], text=True, env=env))
            receipt = {'schema': 'research2-installed-environment-v1', 'profile': args.profile,
                       'platform': sys.platform, 'python': str(python), 'packages': frozen.splitlines(),
                       'environment_root':str(environment.resolve()),
                       'actual_interpreter':actual_python,
                       'installed_profiles':list(dict.fromkeys(previous.get('installed_profiles', [previous['profile']] if previous else [])+([args.profile] if package_index == len(packages)-1 else []))),
                       'installation_status':'COMPLETE' if package_index == len(packages)-1 else 'PARTIAL',
                       'bundled_packages': bundled,
                       'declarations_sha256': hashlib.sha256(json.dumps(declaration, sort_keys=True).encode()).hexdigest(),
                       'science_acceptance': 'NOT_ESTABLISHED'}
            destination = environment/'installation.json'
            fd, tmp = tempfile.mkstemp(dir=environment, prefix='.installation-')
            with os.fdopen(fd, 'w') as stream:
                json.dump(receipt, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, destination)
            previous = receipt
    if args.check or args.build_all or forwarded:
        if not python.is_file():
            parser.error('Install external environment first with --install')
        profile_result = subprocess.call([str(python), '-B', str(SOURCE_ROOT/'bootstrap.py'),
            '--check-python-profile', '--profile', args.profile, '--state-root', str(state)], env=env)
        if profile_result:
            return profile_result
        delegated = (['--check'] if args.check else [])+(['--build-all'] if args.build_all else [])+forwarded
        return subprocess.call([str(python), '-B', str(SOURCE_ROOT/'commands.py'), 'bootstrap', *delegated], env=env)
    if not args.install:
        parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
