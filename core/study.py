"""Conservative orchestration of existing native and saved-output commands.

This is intentionally a narrow adapter: it does not manufacture missing scientific
handoffs or promote successful process execution to scientific acceptance.
"""
from __future__ import annotations

import argparse
import datetime
from copy import deepcopy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.study-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _time():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _path(base, value):
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _directory_hashes(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('Expected output directory missing: ' + str(directory))
    result = {}
    for file in sorted(directory.rglob('*')):
        if file.is_symlink():
            raise ValueError('Cannot seal output containing symlinks: ' + str(file))
        if file.is_file():
            result[str(file.relative_to(directory))] = _hash(file)
    if not result:
        raise ValueError('Expected output directory is empty: ' + str(directory))
    return result


def _artifact_hashes(value):
    """Seal absolute paths explicitly exposed in a workflow's artifacts."""
    paths = set()
    def visit(item):
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str) and item.startswith('/'):
            paths.add(item)
    visit(value)
    result = {}
    for name in sorted(paths):
        path = Path(name)
        if path.is_symlink():
            raise ValueError('Cannot seal artifact symlink: ' + name)
        if path.is_file():
            result[name] = {'file': _hash(path)}
        elif path.is_dir():
            result[name] = {'directory': _directory_hashes(path)}
        else:
            raise ValueError('Workflow references a missing artifact: ' + name)
    return result



def _verify_target_selection(selection, *, run, configuration):
    """Bind a selector result to this study and its immutable native sources."""
    if (selection.get('schema') != 'zoom-target-selection-v1'
            or Path(selection.get('parent_run', '')).resolve() != run
            or selection.get('snapshot') != configuration['snapshot']
            or selection.get('rules') != configuration['rules']
            or selection.get('explicit_overrides') != configuration['host_rows']
            or selection.get('science_acceptance') != 'NOT_ACCEPTED'
            or not isinstance(selection.get('targets'), list) or not selection['targets']):
        raise ValueError('Target selection does not match this study request')
    bindings = selection.get('source_hashes')
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('Target selection lacks its catalogue and snapshot source hashes')
    for name, checksum in bindings.items():
        path = Path(name)
        if (not path.is_absolute() or not path.resolve().is_relative_to(run)
                or not path.is_file() or _hash(path) != checksum):
            raise ValueError('Target-selection source is missing, outside the native run, or changed: ' + name)
    return bindings



def _zoom_outputs(directory):
    """Seal campaign products, excluding only native-run advisory locks."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('Zoom campaign output directory is missing')
    result = {}
    for file in sorted(directory.rglob('*')):
        if '.native-run-locks' in file.relative_to(directory).parts:
            continue
        if file.is_symlink():
            raise ValueError('Cannot seal zoom output containing symlinks: ' + str(file))
        if file.is_file():
            result[str(file.relative_to(directory))] = _hash(file)
    if not result:
        raise ValueError('Zoom campaign output directory is empty')
    return result


def _build_zoom_request(template, *, run, spectrum, preview, expected_ic_sha256):
    """Fill only source-derived parent bindings; never infer physics parameters."""
    request = deepcopy(template)
    manifest_path = run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'SIMULATION_COMPLETE':
        raise ValueError('Zoom parent native evolution is not complete')
    initial = manifest.get('gadget_handoff', {}).get('initial_conditions', [])
    if len(initial) != 1 or not isinstance(initial[0].get('path'), str):
        raise ValueError('Physical zoom requires an unambiguous single-file native parent IC')
    ic = Path(initial[0]['path']).resolve()
    if (not ic.is_file() or _hash(ic) != initial[0].get('sha256')
            or initial[0].get('sha256') != expected_ic_sha256):
        raise ValueError('Native parent IC changed from the recorded GADGET handoff')
    if not spectrum.is_file() or not preview.is_file():
        raise ValueError('Study spectrum and target selection must exist before zoom preparation')
    parent = request['parent']
    parent.update(run_dir=str(run), initial_ic=str(ic), source_manifest=str(manifest_path),
                  spectrum_manifest=str(spectrum))
    parent['sha256'].update(initial_ic=_hash(ic), source_manifest=_hash(manifest_path),
                            spectrum_manifest=_hash(spectrum))
    request['selection_preview'] = {'path': str(preview), 'sha256': _hash(preview)}
    return request


def _tree(run):
    # Results are published by downstream stages; the solver's frozen input and
    # saved particle data are separately hashed so postprocessing cannot change
    # the evidence used to resume evolution.
    result = {}
    for part in ('config.yaml', 'manifest.json', 'input', 'data'):
        path = run / part
        files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else []
        for file in files:
            if file.is_symlink():
                raise ValueError('Cannot seal a native tree containing symlinks: ' + str(file))
            result[str(file.relative_to(run))] = _hash(file)
    if 'config.yaml' not in result or 'manifest.json' not in result or not any(k.startswith('data/') for k in result):
        raise ValueError('Native tree lacks config, manifest, or data; cannot seal completion')
    return result


def _versions():
    versions = {'python': sys.version}
    for name in ('numpy', 'scipy', 'h5py', 'matplotlib', 'PyYAML'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def plan(spec_path):
    spec_path = Path(spec_path).expanduser().resolve()
    spec = json.loads(spec_path.read_text())
    if not isinstance(spec, dict) or spec.get('schema') != 'research2-study-v1':
        raise ValueError('Expected a research2-study-v1 JSON object')
    allowed = {'schema', 'answers', 'run_dir', 'state_dir', 'analysis', 'visualization', 'requested_integrations', 'notes', 'linear', 'target_selection', 'zoom', 'parent_ic', 'blacklight', 'streams'}
    extra = set(spec) - allowed
    if extra:
        raise ValueError('Unknown study keys: ' + ', '.join(sorted(extra)))
    base = spec_path.parent
    answers_path = _path(base, spec['answers'])
    run = _path(base, spec['run_dir'])
    state = _path(base, spec['state_dir'])
    if run == ROOT or ROOT in run.parents:
        raise ValueError('run_dir must be outside the source tree')
    if state == ROOT or ROOT in state.parents or state == run or run in state.parents or state in run.parents:
        raise ValueError('state_dir must be outside the source tree and separate from run_dir')
    answers = json.loads(answers_path.read_text())
    if not isinstance(answers, dict):
        raise ValueError('answers must be a JSON object')
    missing = []
    stages = []
    linear = spec.get('linear')
    linear_config_path = None
    linear_configuration = None
    derived_answers = None
    if linear is not None:
        if not isinstance(linear, dict) or set(linear) - {'config', 'binary'} or 'config' not in linear:
            raise ValueError('linear requires config and optionally binary')
        linear_config_path = _path(base, linear['config'])
        cfg = json.loads(linear_config_path.read_text())
        linear_configuration = cfg
        if not isinstance(cfg, dict) or cfg.get('model') not in ('lcdm', 'ede'):
            raise ValueError('linear config must declare lcdm or ede')
        name = cfg.get('name')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
            raise ValueError('linear config requires a safe solver run name')
        spectrum_dir = state / 'linear' / name
        spectrum = spectrum_dir / 'spectrum.json'
        # Never silently replace a conflicting authored cosmology. An omitted
        # value is allowed only when this stage generates the exact bundle.
        if 'cosmology' in answers:
            if not isinstance(answers['cosmology'], str) or _path(ROOT, answers['cosmology']) not in (spectrum, spectrum_dir):
                raise ValueError('answers cosmology must be omitted or name the generated linear bundle')
        derived_answers = dict(answers, cosmology=str(spectrum))
        generated_answers_path = state / 'gadget-answers.json'
        command = [sys.executable, str(ROOT / 'commands.py'), 'class', '--config', str(linear_config_path), '--output', str(state / 'linear')]
        linear_inputs = [str(linear_config_path)]
        if 'binary' in linear:
            binary = _path(base, linear['binary'])
            command.extend(['--binary', str(binary)])
            linear_inputs.append(str(binary))
            if not binary.is_file():
                missing.append('Linear solver binary missing: ' + str(binary))
        stages.append({'id': 'linear', 'depends_on': [], 'command': command,
                       'inputs': linear_inputs, 'expected_products': [str(spectrum)],
                       'software': ['AxiCLASS' if cfg['model'] == 'ede' else 'CLASS'],
                       'resources': 'See linear configuration and native solver; no measured cost is implied.'})
        gadget_answers = generated_answers_path
        evolution_dependencies = ['linear']
    else:
        if not isinstance(answers.get('cosmology'), str):
            raise ValueError('answers must specify a completed cosmology spectrum when linear is omitted')
        # The existing gadget CLI resolves answer paths against its working directory.
        spectrum = _path(ROOT, answers['cosmology'])
        if spectrum.is_dir():
            spectrum = spectrum / 'spectrum.json'
        if not spectrum.is_file():
            missing.append('Cosmology spectrum manifest missing: ' + str(spectrum))
        gadget_answers = answers_path
        evolution_dependencies = []
    command = [sys.executable, str(ROOT / 'commands.py'), 'gadget', '--answers', str(gadget_answers), '--run-dir', str(run), '--evolution-only']
    native_ic = spec.get('parent_ic')
    native_ic_files = []
    if native_ic is not None:
        if not isinstance(native_ic, dict) or set(native_ic) != {'path', 'sha256', 'generation_report', 'generation_report_sha256'}:
            raise ValueError('parent_ic needs the native MUSIC IC and generation report, each with SHA-256')
        for path_key, pin_key, flag in (('path', 'sha256', '--existing-ic'),
                                        ('generation_report', 'generation_report_sha256', '--native-generation-report')):
            if not isinstance(native_ic[path_key], str) or not re.fullmatch(r'[0-9a-f]{64}', str(native_ic[pin_key])):
                raise ValueError('parent_ic assets need a path and lowercase SHA-256')
            asset = _path(base, native_ic[path_key])
            command.extend([flag, str(asset)])
            if not asset.is_file() or _hash(asset) != native_ic[pin_key]:
                missing.append('Native parent IC asset missing or changed: ' + str(asset))
            else:
                native_ic_files.append(asset)
    stages.append({'id': 'evolution', 'depends_on': evolution_dependencies, 'command': command,
                   'inputs': [str(gadget_answers), str(spectrum)] + [str(p) for p in native_ic_files], 'expected_products': [str(run / 'manifest.json'), str(run / 'data')],
                   'software': ['configured IC generator', 'GADGET-4', 'MPI'],
                   'resources': 'See answers and native configuration; no measured cost is implied.'})
    analysis = spec.get('analysis')
    if analysis is not None:
        if not isinstance(analysis, dict) or set(analysis) - {'graphs', 'requests', 'find_halos'}:
            raise ValueError('analysis supports graphs, requests, and find_halos')
        graphs = analysis.get('graphs')
        if not isinstance(graphs, list) or not graphs or any(not isinstance(x, str) or not x or ',' in x for x in graphs) or any(x in ('all', 'available') for x in graphs):
            raise ValueError('analysis.graphs must explicitly name one or more graph products')
        command = [sys.executable, str(ROOT / 'commands.py'), 'analysis', str(run), '--graphs', ','.join(graphs), '--state-root', str(state / 'analysis-state')]
        if type(analysis.get('find_halos', False)) is not bool:
            raise ValueError('analysis.find_halos must be a boolean')
        if analysis.get('find_halos', False):
            command.append('--find-halos')
        requests = analysis.get('requests', {})
        flags = {'stellar_search': '--stellar-search', 'stellar_history': '--stellar-history', 'zoom_regions': '--zoom-regions', 'inference_diagnostics': '--inference-diagnostics', 'radial_shapes': '--radial-shapes', 'research_spec': '--research-spec', 'evidence': '--evidence'}
        if not isinstance(requests, dict) or set(requests) - set(flags):
            raise ValueError('Unknown analysis request')
        inputs = [str(run / 'config.yaml'), str(run / 'data')]
        for key, value in sorted(requests.items()):
            path = _path(base, value)
            inputs.append(str(path))
            command.extend([flags[key], str(path)])
            if not path.is_file():
                missing.append('Analysis request missing: ' + str(path))
        stages.append({'id': 'analysis', 'depends_on': ['evolution'], 'command': command, 'inputs': inputs,
                       'expected_products': [str(run / 'results' / 'workflow_*.json')], 'software': ['Research2 saved-run analysis'],
                       'resources': 'Depends on selected products; inspect saved-run analysis dry run after evolution.'})
    selection = spec.get('target_selection')
    if selection is not None:
        permitted = {'snapshot', 'memory_gib', 'minimum_dm_particles', 'maximum_dm_particles',
                     'maximum_targets', 'ranking', 'host_rows'}
        if not isinstance(selection, dict) or set(selection) - permitted:
            raise ValueError('target_selection contains unsupported settings')
        snapshot = selection.get('snapshot')
        memory = selection.get('memory_gib')
        minimum = selection.get('minimum_dm_particles', 1000)
        maximum = selection.get('maximum_dm_particles')
        max_targets = selection.get('maximum_targets', 1)
        ranking = selection.get('ranking', 'dm_count_descending')
        rows = selection.get('host_rows', [])
        if (type(snapshot) is not int or snapshot < 0 or type(memory) not in (int, float)
                or not math.isfinite(memory) or memory <= 0
                or type(minimum) is not int or minimum < 1
                or (maximum is not None and (type(maximum) is not int or maximum < minimum))
                or type(max_targets) is not int or max_targets < 1
                or ranking not in ('dm_count_descending', 'catalogue_order')
                or not isinstance(rows, list) or any(type(row) is not int or row < 0 for row in rows)
                or len(set(rows)) != len(rows) or len(rows) > max_targets):
            raise ValueError('Invalid target selection epoch, limits, ranking, memory, or explicit rows')
        rules = {'minimum_dm_particles': minimum, 'maximum_targets': max_targets, 'ranking': ranking}
        command = [sys.executable, str(ROOT / 'commands.py'), 'select-zoom-targets', str(run),
                   '--snapshot', str(snapshot), '--memory-gib', str(memory),
                   '--minimum-dm-particles', str(minimum), '--maximum-targets', str(max_targets),
                   '--ranking', ranking]
        if maximum is not None:
            rules['maximum_dm_particles'] = maximum
            command += ['--maximum-dm-particles', str(maximum)]
        for row in rows:
            command += ['--host-row', str(row)]
        dependencies = ['analysis'] if analysis is not None and analysis.get('find_halos', False) else ['evolution']
        stages.append({'id': 'target_selection', 'depends_on': dependencies, 'command': command,
                       'inputs': [str(run / 'data'), str(run / 'config.yaml')],
                       'expected_products': [str(state / 'target-selection.json')],
                       'software': ['Research2 native FOF catalogue selector'],
                       'resources': {'memory_gib': memory, 'note': 'selection bound; not an RSS or evolution estimate'},
                       'configuration': {'snapshot': snapshot, 'rules': rules, 'host_rows': rows}})
    zoom = spec.get('zoom')
    zoom_template_path = None
    zoom_template = None
    zoom_fixed_files = []
    if zoom is not None:
        if not isinstance(zoom, dict) or set(zoom) != {'template'} or selection is None or native_ic is None:
            raise ValueError('zoom requires a template, target_selection, and native MUSIC parent_ic')
        zoom_template_path = _path(base, zoom['template'])
        zoom_template = json.loads(zoom_template_path.read_text())
        if not isinstance(zoom_template, dict) or zoom_template.get('schema') != 'automatic-zoom-campaign-v1':
            raise ValueError('zoom template must be an automatic-zoom-campaign-v1 request')
        parent = zoom_template.get('parent', {})
        pins = parent.get('sha256', {}) if isinstance(parent, dict) else {}
        marker = '${study}'
        if (not isinstance(parent, dict) or not isinstance(pins, dict)
                or any(parent.get(key) != marker for key in ('run_dir', 'initial_ic', 'source_manifest', 'spectrum_manifest'))
                or any(pins.get(key) != marker for key in ('initial_ic', 'source_manifest', 'spectrum_manifest'))
                or zoom_template.get('selection_preview', marker) != marker):
            raise ValueError('Zoom template needs ${study} markers for derived parent assets and preview')
        if (type(parent.get('snapshot')) is not int or parent['snapshot'] != snapshot or zoom_template.get('selection_rules') != rules
                or zoom_template.get('explicit_host_rows', []) != rows):
            raise ValueError('Zoom parent epoch, selection rules and overrides must exactly match target_selection')
        if not isinstance(parent.get('noise'), str) or not parent['noise'] or not re.fullmatch(r'[0-9a-f]{64}', str(pins.get('noise', ''))):
            raise ValueError('Zoom template must pin the actual parent coarse-noise path and SHA-256')
        noise = _path(zoom_template_path.parent, parent['noise'])
        if not noise.is_file() or _hash(noise) != pins['noise']:
            missing.append('Pinned parent MUSIC noise missing or changed: ' + str(noise))
        else:
            zoom_fixed_files.append(noise)
        for container, path_key, hash_key in ((zoom_template.get('generator', {}), 'binary', 'binary_sha256'),
                                               (zoom_template.get('generator', {}), 'native_validation', 'validation_sha256'),
                                               (zoom_template.get('evolution', {}).get('answers_template', {}), 'path', 'sha256')):
            if not isinstance(container, dict) or not isinstance(container.get(path_key), str) or not re.fullmatch(r'[0-9a-f]{64}', str(container.get(hash_key, ''))):
                raise ValueError('Zoom template needs explicit pinned generator, validation and answers assets')
            asset = _path(zoom_template_path.parent, container[path_key])
            if not asset.is_file() or _hash(asset) != container[hash_key]:
                missing.append('Pinned zoom asset missing or changed: ' + str(asset))
            else:
                zoom_fixed_files.append(asset)
        for collection in (zoom_template.get('evolution', {}).get('input_files', []),
                           zoom_template.get('diagnostics', {}).get('input_files', []),
                           list((zoom_template.get('staging', {}).get('phase_noise_by_level') or {}).values())):
            if not isinstance(collection, list):
                raise ValueError('Zoom pinned input lists must be arrays')
            for item in collection:
                if (not isinstance(item, dict) or not isinstance(item.get('path'), str)
                        or not re.fullmatch(r'[0-9a-f]{64}', str(item.get('sha256', '')))):
                    raise ValueError('Zoom recipe and phase inputs need exact paths and SHA-256')
                asset = _path(zoom_template_path.parent, item['path'])
                if not asset.is_file() or _hash(asset) != item['sha256']:
                    missing.append('Pinned zoom input missing or changed: ' + str(asset))
                else:
                    zoom_fixed_files.append(asset)
        zoom_request = state / 'zoom-campaign.request.json'
        zoom_output = state / 'zoom'
        command = [sys.executable, str(ROOT / 'commands.py'), 'zoom-campaign', '--campaign', str(zoom_request),
                   '--request-base', str(zoom_template_path.parent), '--output', str(zoom_output)]
        stages.append({'id': 'zoom', 'depends_on': ['target_selection'], 'command': command,
                       'inputs': [str(zoom_template_path), str(noise), str(run / 'manifest.json'), str(spectrum),
                                  str(state / 'target-selection.json')],
                       'expected_products': [str(zoom_output / 'zoom-campaign.json')],
                       'software': ['MUSIC', 'GADGET-4', 'configured diagnostic software'],
                       'resources': zoom_template.get('resources', {}),
                       'request_base': str(zoom_template_path.parent)})
    blacklight = spec.get('blacklight')
    blacklight_files = []
    if blacklight is not None:
        required = {'source', 'binary', 'binary_sha256', 'input', 'provenance', 'memory_gib',
                    'wall_seconds', 'display_max', 'display_stretch'}
        allowed = required | {'palette'}
        if not isinstance(blacklight, dict) or not required <= set(blacklight) or set(blacklight) - allowed:
            raise ValueError('blacklight needs an explicit source, binary, input, provenance, resource and display settings')
        if not re.fullmatch(r'[0-9a-f]{64}', str(blacklight['binary_sha256'])):
            raise ValueError('Blacklight executable needs a lowercase SHA-256')
        for key in ('memory_gib', 'display_max', 'display_stretch'):
            value = blacklight[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Blacklight '+key+' must be finite and positive')
        if type(blacklight['wall_seconds']) is not int or blacklight['wall_seconds'] < 1:
            raise ValueError('Blacklight wall_seconds must be a positive integer')
        palette = blacklight.get('palette', 'ember')
        if palette not in ('ember', 'ice'):
            raise ValueError('Blacklight palette must be ember or ice')
        source = _path(base, blacklight['source'])
        binary = _path(base, blacklight['binary'])
        image_input = _path(base, blacklight['input'])
        provenance = _path(base, blacklight['provenance'])
        if not source.is_dir():
            missing.append('Blacklight pinned source checkout missing: ' + str(source))
        for path in (binary, image_input, provenance):
            if not path.is_file():
                missing.append('Blacklight input missing: ' + str(path))
            else:
                blacklight_files.append(path)
        if binary.is_file() and _hash(binary) != blacklight['binary_sha256']:
            missing.append('Blacklight binary hash mismatch: ' + str(binary))
        output = state / 'blacklight' / 'attempt_001'
        command = [sys.executable, '-m', 'core.blacklight_bridge', '--source', str(source),
                   '--binary', str(binary), '--binary-sha256', blacklight['binary_sha256'],
                   '--input', str(image_input), '--provenance', str(provenance), '--output', str(output),
                   '--memory-gib', str(blacklight['memory_gib']), '--wall-seconds', str(blacklight['wall_seconds']),
                   '--display-max', str(blacklight['display_max']), '--display-stretch', str(blacklight['display_stretch']),
                   '--palette', palette]
        stages.append({'id': 'blacklight', 'depends_on': [], 'command': command,
                       'inputs': [str(source), str(binary), str(image_input), str(provenance)],
                       'expected_products': [str(output / 'handoff.json')],
                       'software': ['Blacklight external pinned checkout and executable'],
                       'resources': {'memory_gib': blacklight['memory_gib'], 'wall_seconds': blacklight['wall_seconds']},
                       'scope': 'Independent analytic illustration or explicitly supplied GRMHD; no GADGET plasma is inferred.'})
    streams = spec.get('streams')
    streams_campaign_path = None
    streams_files = []
    if streams is not None:
        if not isinstance(streams, dict) or 'campaign' not in streams:
            raise ValueError('streams requires campaign (file path or inline dictionary)')
        campaign_val = streams['campaign']
        if isinstance(campaign_val, str):
            streams_campaign_path = _path(base, campaign_val)
            if not streams_campaign_path.is_file():
                missing.append('Streams campaign file missing: ' + str(streams_campaign_path))
            else:
                streams_files.append(streams_campaign_path)
        elif isinstance(campaign_val, dict):
            streams_campaign_path = state / 'stream-campaign.json'
        else:
            raise ValueError('streams.campaign must be a file path string or campaign dictionary')
        streams_output = state / 'streams'
        streams_command = [sys.executable, str(ROOT / 'commands.py'), 'streams', str(streams_campaign_path),
                           '--output', str(streams_output)]
        stream_deps = ['analysis'] if analysis is not None else ['evolution']
        stages.append({'id': 'streams', 'depends_on': stream_deps, 'command': streams_command,
                       'inputs': [str(streams_campaign_path), str(run / 'data')],
                       'expected_products': [str(streams_output)],
                       'software': ['Research2 streams framework (Gala / StreaMAX / native)'],
                       'resources': streams.get('resources', {'memory_gib': 4.0}),
                       'campaign_inline': campaign_val if isinstance(campaign_val, dict) else None})
    visual = spec.get('visualization')
    if visual is not None:
        if not isinstance(visual, dict) or set(visual) - {'max_points', 'quality', 'frames_per_snapshot'}:
            raise ValueError('visualization supports max_points, quality, frames_per_snapshot')
        points, quality, frames = visual.get('max_points', 100000), visual.get('quality', 'preview'), visual.get('frames_per_snapshot', 8)
        if type(points) is not int or not 100 <= points <= 5000000 or quality not in ('preview', 'high') or type(frames) is not int or not 2 <= frames <= 240:
            raise ValueError('Invalid visualization point budget, quality, or frame count')
        scene = state / 'scene'
        command = [sys.executable, str(ROOT / 'commands.py'), 'blender', str(run), '--max-points', str(points), '--quality', quality, '--frames-per-snapshot', str(frames), '--output', str(scene)]
        stages.append({'id': 'scene', 'depends_on': ['evolution'], 'command': command, 'inputs': [str(run / 'data')],
                       'expected_products': [str(scene)], 'software': ['Research2 Blender exporter'],
                       'resources': 'Export only; Blender scene build and rendering are separate.'})
    requested = spec.get('requested_integrations', [])
    if not isinstance(requested, list) or any(not isinstance(item, str) or not item for item in requested):
        raise ValueError('requested_integrations must be a list of names')
    for item in requested:
        if ((item == 'zoom' and zoom is not None) or (item == 'target_selection' and selection is not None)
                or (item == 'blacklight' and blacklight is not None) or (item == 'streams' and streams is not None)):
            continue
        stages.append({'id': 'unsupported:' + item, 'depends_on': ['evolution'], 'status': 'UNSUPPORTED',
                       'reason': 'No automatic study adapter is implemented; use its standalone workflow after reviewing inputs.'})
    source_files = [ROOT / 'commands.py'] + sorted((ROOT / 'core').rglob('*.py'))
    source_hashes = {str(p.relative_to(ROOT)): _hash(p) for p in source_files if p.is_file()}
    input_files = [spec_path, answers_path, *native_ic_files, *blacklight_files]
    if streams_files:
        input_files += streams_files
    if linear_config_path is not None:
        input_files.append(linear_config_path)
        if 'binary' in linear and _path(base, linear['binary']).is_file():
            input_files.append(_path(base, linear['binary']))
    elif spectrum.is_file():
        input_files.append(spectrum)
    if analysis is not None:
        input_files += [_path(base, p) for p in analysis.get('requests', {}).values() if _path(base, p).is_file()]
    if zoom_template_path is not None:
        input_files += [zoom_template_path, *zoom_fixed_files]
    return {'schema': 'research2-study-plan-v1', 'created_utc': _time(), 'spec': spec,
            'inputs_sha256': {str(p): _hash(p) for p in input_files}, 'source_sha256': source_hashes,
            'software_versions': _versions(), 'available_tools': {name: shutil.which(name) for name in ('mpirun', 'mpicc', 'make', 'blender')},
            'linear_configuration': linear_configuration, 'zoom_template': zoom_template, 'spectrum_path': str(spectrum), 'native_answers': answers, 'derived_native_answers': derived_answers, 'run_dir': str(run), 'state_dir': str(state), 'stages': stages,
            'missing_inputs': missing, 'science_status': 'NOT_ACCEPTED',
            'limitations': ['Planning does not validate native configuration, linked spectrum files, compiled features, quota, or scientific acceptance.',
                            'Source hash inventory covers Python only, not native code, binaries, tables, linked spectra or system libraries.',
                            'Zoom requires an explicit fully configured MUSIC campaign and validated parent noise; stellar progenitor selection, stream adapters, and render execution are not automatically connected.']}


def execute(plan_data, retry_postprocessing=False, resume_zoom=False, retry_blacklight=False):
    if plan_data['missing_inputs'] or any(s.get('status') == 'UNSUPPORTED' for s in plan_data['stages']):
        raise ValueError('Resolve missing inputs and unsupported requested integrations before execution')
    state = Path(plan_data['state_dir'])
    run = Path(plan_data['run_dir'])
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'study.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record_path = state / 'study.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            previous = record['plan']
            for key in ('spec', 'inputs_sha256', 'source_sha256', 'software_versions'):
                if previous[key] != plan_data[key]:
                    raise ValueError('Study inputs, source, or software changed; use a new state and run directory: ' + key)
        else:
            if run.exists() and any(run.iterdir()):
                raise ValueError('Existing run is not owned by this study; choose a fresh run directory')
            if any(s['id'] == 'linear' for s in plan_data['stages']) and (state / 'linear').exists():
                raise ValueError('Existing linear output is not owned by this study; choose a fresh state directory')
            record = {'schema': 'research2-study-record-v1', 'plan': plan_data,
                      'stages': {s['id']: {'status': 'PLANNED', 'attempts': []} for s in plan_data['stages']},
                      'science_status': 'NOT_ACCEPTED'}
            _atomic(record_path, record)
        for stage in plan_data['stages']:
            name = stage['id']
            old = record['stages'].get(name)
            if old and old['status'] == 'COMPLETE':
                if name == 'linear' and _directory_hashes(Path(stage['expected_products'][0]).parent) != old['outputs_sha256']:
                    raise ValueError('Linear bundle changed since completion; refusing automatic reuse')
                if name == 'evolution':
                    # Halo finding may add native catalogues and ordered snapshots.
                    # Permit them only when a completed analysis sealed the full
                    # resulting input/data tree, while every evolution byte remains.
                    current_native = _tree(run)
                    original = old['native_sha256']
                    if any(current_native.get(path) != checksum for path, checksum in original.items()):
                        raise ValueError('Original native evidence changed since completion; refusing automatic reuse')
                    analysis_record = record['stages'].get('analysis', {})
                    allowed_native = (analysis_record.get('post_analysis_native_sha256')
                                      if analysis_record.get('status') == 'COMPLETE' else original)
                    if allowed_native is None or current_native != allowed_native:
                        raise ValueError('Native tree contains unsealed changes; inspect interrupted analysis and preserve its outputs before explicit recovery')
                    if plan_data.get('derived_native_answers') is not None:
                        derived_path = state / 'gadget-answers.json'
                        if not derived_path.is_file() or json.loads(derived_path.read_text()) != plan_data['derived_native_answers']:
                            raise ValueError('Generated GADGET answers changed; refusing automatic reuse')
                if name == 'analysis':
                    if _directory_hashes(run / 'results') != old['outputs_sha256']:
                        raise ValueError('Analysis outputs changed since completion; refusing automatic reuse')
                    workflow = json.loads(Path(old['workflow']['path']).read_text())
                    if _artifact_hashes(workflow.get('artifacts', {})) != old['artifact_sha256']:
                        raise ValueError('Referenced analysis artifacts changed; refusing automatic reuse')
                if name == 'target_selection':
                    selected_path = Path(stage['expected_products'][0])
                    if not selected_path.is_file() or _hash(selected_path) != old['selection_sha256']:
                        raise ValueError('Target selection changed since completion; refusing automatic reuse')
                    _verify_target_selection(json.loads(selected_path.read_text()), run=run,
                                             configuration=stage['configuration'])
                if name == 'zoom':
                    if _zoom_outputs(state / 'zoom') != old['outputs_sha256']:
                        raise ValueError('Zoom campaign outputs changed since completion; refusing automatic reuse')
                    request_path = state / 'zoom-campaign.request.json'
                    if not request_path.is_file() or _hash(request_path) != old['request_sha256']:
                        raise ValueError('Generated zoom campaign request changed')
                    for path, checksum in old['campaign_inputs_sha256'].items():
                        if not Path(path).is_file() or _hash(path) != checksum:
                            raise ValueError('Zoom campaign source changed: ' + path)
                if name == 'blacklight':
                    output = Path(old['output'])
                    if not output.is_relative_to(state / 'blacklight') or _directory_hashes(output) != old['outputs_sha256']:
                        raise ValueError('Blacklight output changed since completion')
                    handoff = json.loads((output / 'handoff.json').read_text())
                    from core.blacklight_bridge import _source
                    if _source(handoff['source_path'])[1] != handoff['source_sha256']:
                        raise ValueError('Blacklight source changed since completion')
                    for path, checksum in handoff['input_hashes'].items():
                        if not Path(path).is_file() or _hash(path) != checksum:
                            raise ValueError('Blacklight input changed: ' + path)
                if name == 'scene' and _directory_hashes(state / 'scene') != old['outputs_sha256']:
                    raise ValueError('Scene outputs changed since completion; refusing automatic reuse')
                if name == 'streams' and _directory_hashes(state / 'streams') != old['outputs_sha256']:
                    raise ValueError('Streams outputs changed since completion; refusing automatic reuse')
                continue
            if old and old['status'] != 'PLANNED' and name == 'linear':
                raise ValueError('Linear calculation did not complete; inspect and preserve its output and use a fresh study state')
            if old and old['status'] != 'PLANNED' and name == 'evolution':
                raise ValueError('Evolution did not complete in this study. Inspect native manifest and use gadget --resume only when its checkpoint contract permits; automatic adoption is not implemented.')
            if old and old['status'] != 'PLANNED' and name == 'zoom' and not resume_zoom:
                raise ValueError('Zoom campaign was interrupted or failed; inspect its evidence and explicitly use --resume-zoom')
            if old and old['status'] != 'PLANNED' and name == 'blacklight' and not retry_blacklight:
                raise ValueError('Blacklight calculation failed or was interrupted; inspect evidence and explicitly use --retry-blacklight')
            if old and old['status'] != 'PLANNED' and name not in ('zoom', 'blacklight') and not retry_postprocessing:
                raise ValueError('Postprocessing was interrupted or failed; inspect its log and outputs, then use --retry-postprocessing')
            if old and old['status'] != 'PLANNED' and name == 'scene' and (state / 'scene').exists():
                raise ValueError('Scene output already exists; inspect and move it to a safe location before retry')
            attempt = len(old.get('attempts', [])) + 1 if old else 1
            entry = old or {'attempts': []}
            log = state / (name + '-' + str(attempt) + '.log')
            attempt_record = {'started_utc': _time(), 'command': stage['command'], 'log': str(log)}
            entry['attempts'].append(attempt_record)
            entry['status'] = 'RUNNING'
            record['stages'][name] = entry
            _atomic(record_path, record)
            try:
                for filename, expected in plan_data['inputs_sha256'].items():
                    if not Path(filename).is_file() or _hash(filename) != expected:
                        raise ValueError('Study input changed since planning: ' + filename)
                if name == 'evolution' and plan_data.get('derived_native_answers') is not None:
                    derived_path = state / 'gadget-answers.json'
                    if derived_path.exists():
                        if json.loads(derived_path.read_text()) != plan_data['derived_native_answers']:
                            raise ValueError('Generated GADGET answers changed; refusing automatic reuse')
                    else:
                        _atomic(derived_path, plan_data['derived_native_answers'])
                if name == 'streams' and stage.get('campaign_inline') is not None:
                    campaign_path = state / 'stream-campaign.json'
                    if not campaign_path.is_file():
                        _atomic(campaign_path, stage['campaign_inline'])
                command = list(stage['command'])
                if name == 'zoom':
                    request_path = state / 'zoom-campaign.request.json'
                    zoom_request = _build_zoom_request(plan_data['zoom_template'], run=run,
                        spectrum=Path(plan_data['spectrum_path']), preview=state / 'target-selection.json',
                        expected_ic_sha256=plan_data['spec']['parent_ic']['sha256'])
                    if request_path.exists():
                        if json.loads(request_path.read_text()) != zoom_request:
                            raise ValueError('Generated zoom request changed; refusing to replace prior evidence')
                    else:
                        _atomic(request_path, zoom_request)
                    zoom_dir = state / 'zoom'
                    if zoom_dir.exists():
                        if not resume_zoom or not (zoom_dir / 'zoom-campaign.json').is_file():
                            raise ValueError('Existing zoom output lacks an explicitly resumed campaign receipt; inspect it manually')
                        command.append('--resume')
                    attempt_record['command'] = command
                    _atomic(record_path, record)
                if name == 'blacklight':
                    blacklight_output = state / 'blacklight' / ('attempt_%03d' % attempt)
                    if blacklight_output.exists():
                        raise FileExistsError('Preserve existing Blacklight attempt: ' + str(blacklight_output))
                    command[command.index('--output') + 1] = str(blacklight_output)
                    attempt_record['command'] = command
                    _atomic(record_path, record)
                before = set((run / 'results').glob('workflow_*.json')) if name == 'analysis' else set()
                with log.open('x') as output:
                    if name == 'target_selection':
                        # Keep stdout separate because it is the selector's machine-readable receipt.
                        selection_stdout = state / (name + '-' + str(attempt) + '.stdout.json')
                        with selection_stdout.open('x') as selected:
                            proc = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                  stdout=selected, stderr=output, check=False)
                    else:
                        proc = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                              stdout=output, stderr=subprocess.STDOUT, check=False)
                if proc.returncode:
                    raise RuntimeError('Command exited ' + str(proc.returncode))
                for filename, expected in plan_data['inputs_sha256'].items():
                    if not Path(filename).is_file() or _hash(filename) != expected:
                        raise ValueError('Study input changed during stage: ' + filename)
                if name == 'linear':
                    spectrum_dir = Path(stage['expected_products'][0]).parent
                    # The native consumer independently checks the sealed spectrum.
                    status = json.loads((spectrum_dir / 'execution_status.json').read_text())
                    manifest = json.loads((spectrum_dir / 'spectrum.json').read_text())
                    if status.get('status') != 'COMPLETE' or manifest.get('status') != 'complete' or status.get('manifest_sha256') != _hash(spectrum_dir / 'spectrum.json'):
                        raise RuntimeError('Linear publication is not complete and sealed')
                    entry['outputs_sha256'] = _directory_hashes(spectrum_dir)
                elif name == 'evolution':
                    manifest = json.loads((run / 'manifest.json').read_text())
                    if manifest.get('status') != 'SIMULATION_COMPLETE':
                        raise RuntimeError('Native manifest does not report SIMULATION_COMPLETE')
                    entry['native_sha256'] = _tree(run)
                elif name == 'analysis':
                    created = set((run / 'results').glob('workflow_*.json')) - before
                    created = {p for p in created if re.fullmatch(r'workflow_\d{8}T\d{12}Z\.json', p.name)}
                    if len(created) != 1:
                        raise RuntimeError('Expected exactly one new analysis workflow record')
                    workflow = created.pop()
                    result = json.loads(workflow.read_text())
                    if result.get('status') != 'COMPLETE':
                        raise RuntimeError('Analysis record is not COMPLETE: ' + str(workflow))
                    entry['workflow'] = {'path': str(workflow), 'sha256': _hash(workflow)}
                    entry['outputs_sha256'] = _directory_hashes(run / 'results')
                    entry['artifact_sha256'] = _artifact_hashes(result.get('artifacts', {}))
                    current_native = _tree(run)
                    original = record['stages']['evolution']['native_sha256']
                    if any(current_native.get(path) != checksum for path, checksum in original.items()):
                        raise ValueError('Analysis altered original native evidence; refusing to seal the result')
                    entry['post_analysis_native_sha256'] = current_native
                elif name == 'target_selection':
                    selection_result = json.loads(selection_stdout.read_text())
                    _verify_target_selection(selection_result, run=run, configuration=stage['configuration'])
                    selected_path = Path(stage['expected_products'][0])
                    if selected_path.exists():
                        if json.loads(selected_path.read_text()) != selection_result:
                            raise ValueError('Existing target selection differs; preserve it and inspect the retry')
                    else:
                        _atomic(selected_path, selection_result)
                    entry['selection_sha256'] = _hash(selected_path)
                    entry['selection_stdout'] = str(selection_stdout)
                elif name == 'zoom':
                    campaign = state / 'zoom' / 'zoom-campaign.json'
                    seal = campaign.with_suffix('.sha256')
                    if not campaign.is_file() or not seal.is_file() or _hash(campaign) != seal.read_text().strip():
                        raise ValueError('Zoom campaign completion receipt is missing or unsealed')
                    result = json.loads(campaign.read_text())
                    if result.get('schema') != 'automatic-zoom-campaign-result-v1' or result.get('execution_status') != 'COMPLETE':
                        raise ValueError('Zoom campaign did not complete')
                    entry['request_sha256'] = _hash(state / 'zoom-campaign.request.json')
                    entry['campaign_inputs_sha256'] = result['input_hashes']
                    entry['outputs_sha256'] = _zoom_outputs(state / 'zoom')
                    entry['region_status'] = result.get('status', 'UNRESOLVED')
                    entry['science_acceptance'] = result.get('science_acceptance', 'UNRESOLVED')
                elif name == 'blacklight':
                    handoff_path = blacklight_output / 'handoff.json'
                    handoff = json.loads(handoff_path.read_text())
                    if (handoff.get('schema') != 'blacklight-handoff-v1' or handoff.get('status') != 'COMPLETE'
                            or handoff.get('science_acceptance') != 'NOT_ACCEPTED'):
                        raise ValueError('Blacklight handoff is not complete or has an unexpected science status')
                    for field in ('image', 'display'):
                        artifact = handoff[field]
                        if not Path(artifact['path']).is_file() or _hash(artifact['path']) != artifact['sha256']:
                            raise ValueError('Blacklight '+field+' output is missing or changed')
                    entry['output'] = str(blacklight_output)
                    entry['outputs_sha256'] = _directory_hashes(blacklight_output)
                    entry['model_type'] = handoff['model_type']
                    entry['physical_origin'] = handoff['input_provenance']['origin']
                    entry['science_acceptance'] = handoff['science_acceptance']
                elif name == 'streams':
                    if not (state / 'streams').is_dir():
                        raise RuntimeError('Streams stage did not produce an output directory')
                    entry['outputs_sha256'] = _directory_hashes(state / 'streams')
                    entry['science_acceptance'] = 'NOT_ACCEPTED'
                else:
                    scene_manifest = state / 'scene' / 'blender_universe.json'
                    if not scene_manifest.is_file():
                        raise RuntimeError('Scene exporter did not produce its combined manifest')
                    entry['outputs_sha256'] = _directory_hashes(state / 'scene')
                entry['status'] = 'COMPLETE'
                attempt_record.update(finished_utc=_time(), status='COMPLETE')
            except BaseException as exc:
                entry['status'] = 'FAILED'
                attempt_record.update(finished_utc=_time(), status='FAILED', error=str(exc))
                _atomic(record_path, record)
                raise
            _atomic(record_path, record)
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plan or run a bounded native Research2 study.')
    parser.add_argument('study', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Print a read-only plan; do not create state or run commands')
    parser.add_argument('--retry-postprocessing', action='store_true', help='Retry failed postprocessing after inspecting its outputs')
    parser.add_argument('--resume-zoom', action='store_true', help='Explicitly resume an interrupted source-bound physical zoom campaign')
    parser.add_argument('--retry-blacklight', action='store_true', help='Explicitly retry a failed independent Blacklight calculation in a fresh attempt')
    args = parser.parse_args(argv)
    result = plan(args.study)
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return 0
    record = execute(result, retry_postprocessing=args.retry_postprocessing, resume_zoom=args.resume_zoom, retry_blacklight=args.retry_blacklight)
    print(json.dumps({'record': str(Path(result['state_dir']) / 'study.json'),
        'stages': {k: {'status': v['status'], **({'region_status': v['region_status'],
                    'science_acceptance': v['science_acceptance']} if 'region_status' in v else {}),
                    **({'physical_origin': v['physical_origin'], 'science_acceptance': v['science_acceptance']}
                       if 'physical_origin' in v else {})}
                   for k, v in record['stages'].items()},
        'science_status': record['science_status']}, indent=2))
    return 0
