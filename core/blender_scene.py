"""Portable measured display packages and exact-identity epoch scene intake.

Display coordinates must carry existing data.py contracts. This consumer never
converts native units or calculates unmeasured physics. Macro SIDM interactions
are diagnostic markers, never physical dark-matter particle collision imagery.
"""

from pathlib import Path
import json
import hashlib
import math
import shutil
from .foundation import atomic_json, file_sha256, canonical_digest
from core.native import native_run_lock

KINDS = {'halo', 'subhalo', 'galaxy', 'black_hole', 'stream', 'shape', 'tide', 'event', 'sidm'}


def measured_epoch_clock(epoch):
    """Return the producer's actual clock without converting or inventing time."""
    if not isinstance(epoch,dict):
        raise ValueError('Display epoch requires its actual shared native clock contract')
    kind = epoch.get('canonical_time')
    key = {'scale_factor':'scale_factor','isolated_native_elapsed_time':'native_time_code'}.get(kind)
    if key is None:
        raise ValueError('Display requires an explicit cosmological or isolated native clock')
    value = epoch.get(key)
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Display native clocks must be finite measured numbers')
    if (kind == 'scale_factor' and value <= 0) or (kind != 'scale_factor' and value < 0):
        raise ValueError('Display native clock lies outside its physical domain')
    return kind,value


def _identity(record):
    namespace = record['namespace']
    value = record['object_id']
    if not isinstance(namespace,str) or not namespace.strip() or isinstance(value, bool):
        raise ValueError('Explicit namespace and uint64 identity required')
    if isinstance(value, str):
        if not value.isdecimal() or str(int(value)) != value:
            raise ValueError('Canonical decimal exact object identity required')
        value = int(value)
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError('Object identity must be an exact uint64, never a float')
    return namespace, str(value)


def load_verified_display_products(manifest_path):
    path = Path(manifest_path).resolve()
    encoded = path.read_bytes()
    value = json.loads(encoded)
    if value.get('schema') != 'measured-display-v1':
        raise ValueError('Expected measured-display-v1')
    coordinate_keys = {'schema', 'position_units', 'position_representation', 'position_h_scaled', 'origin', 'periodic_wrapping', 'velocity_convention'}
    if not coordinate_keys <= set(value.get('frame_contract', {})) or value['coordinate_units'] != value['frame_contract']['position_units']:
        raise ValueError('Existing coordinate contract and explicit display units required')
    if hashlib.sha256(encoded).hexdigest() != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Display manifest seal changed')
    seen_epochs = set()
    prior = None
    clock_kind,prior_clock = None,None
    timeline = value['timeline']
    if len(timeline) < 2:
        raise ValueError('At least two saved display epochs required')
    for epoch in timeline:
        if type(epoch['frame']) is not int or epoch['frame'] < 1 or epoch['snapshot'] in seen_epochs:
            raise ValueError('Distinct saved snapshots and positive integer frames required')
        if prior is not None and epoch['frame'] <= prior:
            raise ValueError('Timeline frames must be strictly ordered')
        prior = epoch['frame']
        seen_epochs.add(epoch['snapshot'])
        current_kind,current_clock = measured_epoch_clock(epoch.get('epoch_contract'))
        if clock_kind is not None and (current_kind != clock_kind or current_clock <= prior_clock):
            raise ValueError('Saved display epochs must follow one strictly increasing actual native clock')
        clock_kind,prior_clock = current_kind,current_clock
        identities = set()
        for event in epoch.get('lifecycle_events',[]):
            for ref in event.get('source_links',[]):
                p = (path.parent/ref['path']).resolve()
                if not p.is_relative_to(path.parent) or file_sha256(p) != ref['sha256']:
                    raise ValueError('Lifecycle source changed or escaped package')
        for record in epoch['objects']:
            key = _identity(record)
            if key in identities or record['kind'] not in KINDS:
                raise ValueError('Duplicate/unknown measured object at one epoch')
            identities.add(key)
            if record.get('measurement_origin') != 'retained_scientific_product':
                raise ValueError('Measured scientific product origin required')
            if not record.get('source_links'):
                raise ValueError('Every inspector object needs source product links')
            for ref in record['source_links']:
                p = (path.parent/ref['path']).resolve()
                if not p.is_relative_to(path.parent) or file_sha256(p) != ref['sha256']:
                    raise ValueError('Inspector source changed or escaped package')
            layers = record.get('measured_geometry_layers',[])
            if not isinstance(layers,list) or len({layer.get('kind') for layer in layers}) != len(layers):
                raise ValueError('Geometry layers require a list of distinct measured kinds')
            for layer in layers:
                if _identity(layer) != _identity(record) or layer.get('position') != record.get('position'):
                    raise ValueError('Geometry layer changes its exact object identity or measured position')
                if layer.get('kind') not in KINDS-{'sidm',record['kind']} or layer.get('measurement_origin') != 'retained_scientific_product' or layer.get('measured_geometry_layers'):
                    raise ValueError('Geometry layers require distinct flat retained measured representations')
                if type(layer.get('geometry_supported')) is not bool or not layer.get('source_links'):
                    raise ValueError('Geometry layers require producer support and source links')
                for ref in layer['source_links']:
                    p = (path.parent/ref['path']).resolve()
                    if not p.is_relative_to(path.parent) or file_sha256(p) != ref['sha256'] or ref not in record['source_links']:
                        raise ValueError('Geometry source changed, escaped or lost its inspector binding')
            for diagnostic in record.get('measured_diagnostics',[]):
                if _identity(diagnostic) != _identity(record) or diagnostic.get('position') != record.get('position'):
                    raise ValueError('Diagnostic overlay changes its exact object identity or measured frame position')
                if diagnostic.get('kind') != 'sidm' or diagnostic.get('representation') != 'measured_macro_diagnostics' or diagnostic.get('measurement_origin') != 'retained_scientific_product':
                    raise ValueError('Diagnostic overlays require retained SIDM macro measurements')
                if type(diagnostic.get('geometry_supported')) is not bool or not diagnostic.get('source_links'):
                    raise ValueError('Diagnostic overlays require explicit producer support and source links')
                for ref in diagnostic['source_links']:
                    p = (path.parent/ref['path']).resolve()
                    if not p.is_relative_to(path.parent) or file_sha256(p) != ref['sha256'] or ref not in record['source_links']:
                        raise ValueError('Diagnostic source changed, escaped or is absent from its exact-object inspector links')
            for key in ('geometry_asset', 'profile_asset', 'event_asset'):
                if key in record:
                    ref = record[key]
                    p = (path.parent/ref['path']).resolve()
                    if not p.is_relative_to(path.parent) or file_sha256(p) != ref['sha256']:
                        raise ValueError('Measured object asset changed or escaped package')
            for ancestor in record.get('parents', []):
                _identity(ancestor)
            if record['kind'] == 'sidm' and record.get('representation') != 'measured_macro_diagnostics':
                raise ValueError('SIDM display must represent measured macro diagnostics')
    measured_lifecycle_index(value)
    for asset in value.get('assets', []):
        p = (path.parent/asset['path']).resolve()
        if not p.is_relative_to(path.parent) or file_sha256(p) != asset['sha256']:
            raise ValueError('Display asset missing or changed')
    return value


def join_exact_object_states(manifest):
    tracks = {}
    for epoch in manifest['timeline']:
        for record in epoch['objects']:
            key = _identity(record)
            tracks.setdefault(key, []).append({'frame': epoch['frame'], 'snapshot': epoch['snapshot'],
                'epoch_contract': epoch['epoch_contract'], 'record': record})
    return tracks


def visibility_schedule(manifest):
    all_keys = sorted(join_exact_object_states(manifest))
    return {key: [{'frame': ep['frame'], 'present': key in {_identity(q) for q in ep['objects']}}
                 for ep in manifest['timeline']] for key in all_keys}


def measured_lifecycle_index(manifest):
    """Exact catalogue presence plus explicit sourced events, without inferred deaths.

    A first/last detection is an observation boundary. It is never promoted to a
    birth/destruction event. Absent epochs between detections remain gaps even
    when another object occupies the old catalogue row.
    """
    timeline = manifest['timeline']
    tracks = join_exact_object_states(manifest)
    index = {}
    for identity, states in tracks.items():
        detected_frames = {q['frame'] for q in states}
        presence = [ep['frame'] in detected_frames for ep in timeline]
        first = presence.index(True)
        last = len(presence)-1-presence[::-1].index(True)
        intervals = []
        start = None
        for ordinal, present in enumerate(presence+[False]):
            if present and start is None:
                start = ordinal
            elif not present and start is not None:
                intervals.append({'first_epoch_index':start,'last_epoch_index':ordinal-1,
                    'first_frame':timeline[start]['frame'],'last_frame':timeline[ordinal-1]['frame']})
                start = None
        key = identity[0]+':'+identity[1]
        index[key] = {'namespace':identity[0],'object_id':identity[1],
            'first_detection_epoch_index':first,'last_detection_epoch_index':last,
            'detected_intervals':intervals,
            'missing_detection_epoch_indices':[q for q in range(first,last+1) if not presence[q]],
            'presence':presence,'measured_events':[],
            'boundary_semantics':'first/last detection do not establish physical birth/death'}
    event_ids = set()
    allowed = {'birth','death','merger','missing_detection','reappearance','stripping','mass_return'}
    for ordinal, epoch in enumerate(timeline):
        for event in epoch.get('lifecycle_events',[]):
            event_identity = _identity(event)
            if event_identity in event_ids:
                raise ValueError('Lifecycle event identity must be globally unique')
            event_ids.add(event_identity)
            if event.get('event_type') not in allowed or event.get('measurement_origin') != 'retained_scientific_product':
                raise ValueError('Lifecycle events require explicit measured types and origins')
            if not event.get('source_links') or not event.get('subjects'):
                raise ValueError('Lifecycle events require measured source links and exact subjects')
            subject_keys = [_identity(q) for q in event['subjects']]
            if len(subject_keys) != len(set(subject_keys)):
                raise ValueError('Duplicate lifecycle subject')
            for subject in subject_keys:
                key = subject[0]+':'+subject[1]
                if key not in index:
                    raise ValueError('Lifecycle subject is outside the selected measured tracks')
                present = index[key]['presence'][ordinal]
                if event['event_type'] == 'missing_detection' and present:
                    raise ValueError('Missing-detection event contradicts a saved detection')
                if event['event_type'] in {'birth','reappearance'} and not present:
                    raise ValueError('Birth/reappearance event requires its saved detection')
                index[key]['measured_events'].append({'epoch_index':ordinal,
                    'frame':epoch['frame'],'snapshot':epoch['snapshot'],'event':event})
            # Parent identities are metadata, not nearest-neighbour matches. An
            # ancestor may lie outside this display selection; retain its ID.
            for parent in event.get('parents',[]):
                _identity(parent)
    return index


def package_display_products(manifest_path, destination):
    source = Path(manifest_path).resolve()
    source_digest = source.with_suffix('.sha256').read_text().strip()
    manifest = load_verified_display_products(source)
    destination = Path(destination).resolve()
    with native_run_lock(destination, operation='package measured display assets'):
        if destination.exists():
            raise FileExistsError('Display packages are immutable')
        destination.mkdir(parents=True)
        assets = {}
        def copy(ref):
            original = (source.parent/ref['path']).resolve()
            digest = file_sha256(original)
            if digest != ref['sha256']:
                raise ValueError('Measured source changed during asset packaging')
            group = ref.get('bundle')
            if group is not None:
                if not isinstance(group,dict) or set(group) != {'id','relative_path'}:
                    raise ValueError('Numeric asset bundle metadata is malformed')
                identity = group['id']
                relative = Path(group['relative_path'])
                if not isinstance(identity,str) or len(identity) != 64 or any(q not in '0123456789abcdef' for q in identity) or relative.is_absolute() or '..' in relative.parts or not relative.parts:
                    raise ValueError('Numeric asset bundle path/identity is unsafe')
                filename = 'assets/product-'+identity+'/'+relative.as_posix()
            else:
                filename = 'assets/'+digest+original.suffix
            target = destination/filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(original, target)
            if file_sha256(target) != digest:
                raise ValueError('Packaged measured asset changed during copy')
            assets[filename] = {'path':filename,'sha256':digest,**({'bundle':group} if group is not None else {})}
            return {**ref, 'path': filename, 'sha256': digest}
        for epoch in manifest['timeline']:
            for event in epoch.get('lifecycle_events',[]):
                event['source_links'] = [copy(ref) for ref in event['source_links']]
            for record in epoch['objects']:
                record['source_links'] = [copy(ref) for ref in record['source_links']]
                for layer in record.get('measured_geometry_layers',[]):
                    layer['source_links'] = [copy(ref) for ref in layer['source_links']]
                    for key in ('geometry_asset','profile_asset','event_asset'):
                        if key in layer:
                            layer[key] = copy(layer[key])
                for diagnostic in record.get('measured_diagnostics',[]):
                    diagnostic['source_links'] = [copy(ref) for ref in diagnostic['source_links']]
                for key in ('geometry_asset', 'profile_asset', 'event_asset'):
                    if key in record:
                        record[key] = copy(record[key])
        for asset in manifest.get('assets', []):
            copy(asset)
        manifest['assets'] = list(assets.values())
        if file_sha256(source) != source_digest or source.with_suffix('.sha256').read_text().strip() != source_digest:
            raise ValueError('Display manifest changed during asset packaging')
        manifest['source_manifest_sha256'] = source_digest
        manifest['science_acceptance'] = 'UNRESOLVED'
        atomic_json(destination/'display.json', manifest)
        from .foundation import atomic_text
        atomic_text(destination/'display.sha256', file_sha256(destination/'display.json')+'\n')
        for name in ('display_runtime.py', 'research_inspector.py'):
            shutil.copy2(Path(__file__).with_name(name), destination/name)
        return {'schema': 'display-package-receipt-v1', 'manifest': str(destination/'display.json'),
            'sha256': file_sha256(destination/'display.json'), 'assets': manifest['assets'],
            'runtime_sources': {n: file_sha256(destination/n) for n in ('display_runtime.py', 'research_inspector.py')},
            'execution_status': 'COMPLETE', 'fresh_load_status': 'NOT_EXECUTED'}


def attach_measured_display(bpy, manifest_path):
    """Attach to an existing unified universe or a new display-only scene."""
    manifest = load_verified_display_products(manifest_path)
    from .display_runtime import install_runtime
    from .research_inspector import install_inspector
    install_runtime(bpy, manifest, Path(manifest_path).parent, lifecycle=measured_lifecycle_index(manifest))
    install_inspector(bpy)
    return manifest


def verify_fresh_load_and_timeline_endpoints(blender_binary, blend_path, output, *, wall_seconds):
    """Authored release control. Only executes when explicitly called."""
    import sys
    from core.native import run_native_command
    blend = Path(blend_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Fresh-load receipts cannot overwrite history')
    script = Path(__file__).with_name('display_fresh_load.py')
    log = output.with_suffix('.log')
    with log.open('wb') as stream:
        run_native_command([str(blender_binary), '--background', str(blend), '--python', str(script), '--', str(output)],
            cwd=blend.parent, receipt_path=output.with_suffix('.invocation.json'),
            inputs=(blend, script), timeout=wall_seconds, stdout=stream, stderr=stream)
    receipt = json.loads(output.read_text())
    if receipt.get('status') != 'PASS' or receipt.get('blend_sha256') != file_sha256(blend):
        raise ValueError('Fresh-loaded scene endpoint control failed')
    return receipt


def scene_main():
    import argparse
    parser = argparse.ArgumentParser(description='Package or build measured scientific inspector objects')
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    if args.build:
        import bpy
        output = args.output.resolve()
        if output.suffix.lower() != '.blend':
            raise ValueError('Measured scene output requires a .blend filename')
        with native_run_lock(output,operation='save immutable measured scene'):
            if output.exists():
                raise FileExistsError('Measured scene files cannot overwrite retained generations')
            output.parent.mkdir(parents=True,exist_ok=True)
            package = output.with_name(output.stem+'.assets')
            package_display_products(args.manifest,package)
            attach_measured_display(bpy,package/'display.json')
            bpy.context.scene['research_display_base'] = '//'+package.name
            bpy.ops.wm.save_as_mainfile(filepath=str(output))
    else:
        print(json.dumps(package_display_products(args.manifest, args.output), indent=2))
