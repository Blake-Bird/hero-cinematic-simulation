"""Pack independently ray-traced Blacklight screens into a discrete Blender sequence.

A screen can be presented in a flat frame or inside a fisheye dome. Neither
presentation retraces rays, creates missing sky, interpolates plasma, or supplies
slow-light physics. See docs/blacklight.md for the required input contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or positive and value <= 0:
        raise ValueError(name + ' must be an explicit finite ' + ('positive ' if positive else '') + 'number')
    return float(value)


def inspect_sequence(path):
    """Inspect all source-bound frames and return an immutable presentation plan."""
    path = Path(path).expanduser().resolve()
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Sequence request exceeds 1 MiB')
    request = json.loads(path.read_text())
    if set(request) != {'schema', 'fps', 'gravitational_times_per_second', 'end_observer_time_tg', 'memory_gib', 'frames'} or request['schema'] != 'blacklight-sequence-request-v1':
        raise ValueError('Declare the complete blacklight-sequence-request-v1 contract')
    fps = request['fps']
    if type(fps) is not int or not 1 <= fps <= 120:
        raise ValueError('Choose an integer presentation frame rate between 1 and 120')
    rate = _finite(request['gravitational_times_per_second'], 'gravitational_times_per_second', positive=True)
    memory = _finite(request['memory_gib'], 'memory_gib', positive=True)
    rows = request['frames']
    if not isinstance(rows, list) or not 2 <= len(rows) <= 128:
        raise ValueError('Supply 2 to 128 independently calculated frames')
    resolved, expected_settings, expected_provenance, expected_display, expected_revision = [], None, None, None, None
    previous_time = None
    previous_snapshot_time = None
    snapshot_times = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'handoff', 'observer_time_tg'} or not isinstance(row['handoff'], str):
            raise ValueError('Each frame requires exactly a handoff path and declared observer time')
        time = _finite(row['observer_time_tg'], 'observer_time_tg')
        if previous_time is not None and time <= previous_time:
            raise ValueError('Observer times must be strictly increasing')
        previous_time = time
        handoff = Path(row['handoff']).expanduser()
        handoff = (path.parent / handoff).resolve() if not handoff.is_absolute() else handoff.resolve()
        if handoff.stat().st_size > 8 * 1024 * 1024:
            raise ValueError('A frame handoff exceeds 8 MiB')
        handoff_hash = _digest(handoff)
        record = json.loads(handoff.read_text())
        if record.get('schema') != 'blacklight-handoff-v1' or record.get('status') != 'COMPLETE':
            raise ValueError('Every source frame must be a completed Blacklight handoff')
        image, numeric = handoff.parent / 'display.png', handoff.parent / 'image.npz'
        if _digest(image) != record['display']['sha256'] or _digest(numeric) != record['image']['sha256']:
            raise ValueError('Frame display or quantitative intensity hash changed')
        settings = record['settings']
        if settings.get('slow_light_on', 'false') != 'false' or settings.get('simulation_multiple', 'false') != 'false':
            raise ValueError('Only separately calculated fast-light images are supported')
        # Physical inputs and numerical settings must be identical. Camera
        # settings may vary and are recorded for each independently traced view.
        fixed_settings = {k: v for k, v in settings.items() if k not in {'output_file', 'simulation_file'} and (not k.startswith('camera_') or k in {'camera_type', 'camera_resolution'})}
        provenance = record['input_provenance']
        mode = record['model_type']
        common = {k: provenance.get(k) for k in ('origin', 'metric', 'spin', 'mass_msun', 'mass_gravitational_cm', 'density_unit_g_cm3', 'format', 'coordinates', 'emission_assumptions')}
        if mode == 'simulation':
            if provenance.get('origin') != 'independent_grmhd':
                raise ValueError('Simulation frames require declared independent GRMHD inputs')
            snapshot = provenance.get('snapshot_sha256')
            snapshot_time = _finite(provenance.get('snapshot_time_tg'), 'snapshot_time_tg')
            if not isinstance(snapshot, str) or len(snapshot) != 64:
                raise ValueError('GRMHD frame needs a declared source snapshot hash')
            if previous_snapshot_time is not None and snapshot_time < previous_snapshot_time:
                raise ValueError('Declared GRMHD snapshot times must be nondecreasing')
            if snapshot in snapshot_times and snapshot_times[snapshot] != snapshot_time:
                raise ValueError('The same snapshot hash cannot have inconsistent declared times')
            snapshot_times[snapshot] = snapshot_time
            previous_snapshot_time = snapshot_time
        elif mode != 'formula' or provenance.get('origin') != 'analytic_illustration':
            raise ValueError('Formula frames require explicit analytic illustration provenance')
        display = {key:record['display'][key] for key in ('palette', 'intensity_min', 'intensity_max', 'asinh_stretch')}
        version = (record['upstream']['revision'], record['binary_sha256'], mode)
        if expected_settings is None:
            expected_settings, expected_provenance, expected_display, expected_revision = fixed_settings, common, display, version
        elif (fixed_settings, common, display, version) != (expected_settings, expected_provenance, expected_display, expected_revision):
            raise ValueError('Frame physics, settings, display mapping, source or executable differ')
        if _digest(handoff) != handoff_hash:
            raise ValueError('A frame handoff changed while being inspected')
        resolved.append({'observer_time_tg':time, 'handoff_path':str(handoff), 'handoff_sha256':handoff_hash,
                         'display_path':str(image), 'display_sha256':record['display']['sha256'],
                         'numeric_path':str(numeric), 'numeric_sha256':record['image']['sha256'], 'record':record})
    end = _finite(request['end_observer_time_tg'], 'end_observer_time_tg')
    if end <= resolved[-1]['observer_time_tg']:
        raise ValueError('End observer time must follow the final supplied frame')
    first = resolved[0]['observer_time_tg']
    frames = [1 + round((row['observer_time_tg'] - first) * fps / rate) for row in resolved]
    end_frame = 1 + round((end - first) * fps / rate)
    if any(b <= a for a, b in zip(frames, frames[1:])) or end_frame <= frames[-1] or end_frame > 100001:
        raise ValueError('Time mapping must give distinct frame positions and at most 100000 presentation frames')
    resolution = int(expected_settings['camera_resolution'])
    if resolution**2 * len(rows) * 4 * 3 > memory * 1024**3:
        raise MemoryError('Conservative packed RGBA workspace estimate exceeds declared memory')
    for row, frame in zip(resolved, frames):
        row['scene_frame'] = frame
    return {'schema':'blacklight-sequence-plan-v1', 'request_path':str(path), 'request_sha256':_digest(path),
            'fps':fps, 'gravitational_times_per_second':rate, 'end_observer_time_tg':end,
            'frame_start':1, 'frame_end':end_frame-1, 'frames':resolved,
            'presentation':'Hold each separately traced image until the next; no temporal or ray interpolation',
            'timing_scope':'Observer and GRMHD snapshot times are independently supplied declarations in GM/c^3; no causal time mapping or slow-light transfer is computed',
            'source_scope':'Finite Blacklight screens, not all-sky ray maps', 'science_acceptance':'NOT_ACCEPTED'}


def build_scene(plan, destination, *, projection, width, height, exposure, screen_distance):
    import bpy
    destination = Path(destination).expanduser().resolve()
    if destination.suffix.lower() != '.blend' or destination.exists() or destination == ROOT or ROOT in destination.parents:
        raise ValueError('Choose a fresh .blend path outside the source tree')
    if projection not in {'flat', 'dome-screen'} or not 128 <= width <= 16384 or not 128 <= height <= 16384:
        raise ValueError('Supported projection and canvas dimensions required')
    if projection == 'dome-screen' and width != height:
        raise ValueError('Equidistant dome-screen presentation needs a square canvas')
    _finite(exposure, 'exposure'); _finite(screen_distance, 'screen_distance', positive=True)
    if not -10 <= exposure <= 10 or not 1 <= screen_distance <= 100:
        raise ValueError('Display exposure or screen distance outside supported range')
    metadata = json.dumps(plan, sort_keys=True)
    if len(metadata.encode()) > 16 * 1024 * 1024:
        raise ValueError('Embedded sequence metadata exceeds 16 MiB')
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'; scene.render.resolution_x = width; scene.render.resolution_y = height
    scene.render.resolution_percentage = 100; scene.render.fps = plan['fps']; scene.render.fps_base = 1
    scene.frame_start = plan['frame_start']; scene.frame_end = plan['frame_end']
    scene.view_settings.view_transform = 'Standard'; scene.view_settings.exposure = exposure
    if scene.world is None:
        scene.world = bpy.data.worlds.new('Blacklight background')
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get('Background')
    if background is not None:
        background.inputs['Color'].default_value = (0., 0., 0., 1.)
    for i, row in enumerate(plan['frames']):
        if _digest(row['handoff_path']) != row['handoff_sha256'] or _digest(row['display_path']) != row['display_sha256'] or _digest(row['numeric_path']) != row['numeric_sha256']:
            raise ValueError('Source frame changed before packing')
        bpy.ops.mesh.primitive_plane_add(size=2., location=(0, 0, 0))
        obj = bpy.context.object; obj.name = 'Ray-traced frame %04d' % i
        image = bpy.data.images.load(row['display_path'], check_existing=False)
        resolution = int(row['record']['settings']['camera_resolution'])
        if tuple(image.size) != (resolution, resolution):
            raise ValueError('Packed screen dimensions do not match the recorded camera')
        image.pack()
        mat = bpy.data.materials.new('Frame %04d fixed false-color mapping' % i); mat.use_nodes = True
        nodes = mat.node_tree.nodes; nodes.clear()
        tex = nodes.new('ShaderNodeTexImage'); tex.image = image
        emit = nodes.new('ShaderNodeEmission'); transparent = nodes.new('ShaderNodeBsdfTransparent')
        mix = nodes.new('ShaderNodeMixShader'); out = nodes.new('ShaderNodeOutputMaterial')
        links = mat.node_tree.links
        links.new(tex.outputs['Color'], emit.inputs['Color']); links.new(tex.outputs['Alpha'], mix.inputs[0])
        links.new(transparent.outputs['BSDF'], mix.inputs[1]); links.new(emit.outputs['Emission'], mix.inputs[2]); links.new(mix.outputs['Shader'], out.inputs['Surface'])
        obj.data.materials.append(mat)
        start = row['scene_frame']; stop = plan['frames'][i+1]['scene_frame'] if i+1 < len(plan['frames']) else plan['frame_end']+1
        for flag in ('hide_render','hide_viewport'):
            for frame, hidden in ((start-1, True), (start, False), (stop, True)):
                setattr(obj, flag, hidden); obj.keyframe_insert(data_path=flag, frame=frame)
        if obj.animation_data and obj.animation_data.action:
            for curve in obj.animation_data.action.fcurves:
                for keyframe in curve.keyframe_points:
                    keyframe.interpolation = 'CONSTANT'
        obj['handoff_sha256'] = row['handoff_sha256']; obj['numeric_sha256'] = row['numeric_sha256']
        obj['declared_observer_time_tg'] = row['observer_time_tg']
    # Unsupported or negative intensity is transparent in every PNG and appears
    # magenta, separately from a valid finite black pixel.
    bpy.ops.mesh.primitive_plane_add(size=2., location=(0,0,-.01))
    plate = bpy.context.object; plate.name = 'Unsupported rays magenta'
    mat = bpy.data.materials.new('Unsupported pixel indicator'); mat.use_nodes = True
    nodes=mat.node_tree.nodes; nodes.clear(); emit=nodes.new('ShaderNodeEmission')
    emit.inputs['Color'].default_value=(.8,0.,.5,1.); out=nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(emit.outputs['Emission'],out.inputs['Surface']); plate.data.materials.append(mat)
    camera_data = bpy.data.cameras.new('Presentation camera')
    if projection == 'flat':
        camera_data.type='ORTHO'; camera_data.ortho_scale=max(2.4, 2.7*width/height)
        name='Flat_Presentation_Camera'
    else:
        camera_data.type='PANO'
        settings = camera_data if hasattr(camera_data,'panorama_type') else camera_data.cycles
        settings.panorama_type='FISHEYE_EQUIDISTANT'; settings.fisheye_fov=math.pi
        name='Fulldome_Camera'
    camera=bpy.data.objects.new(name,camera_data); scene.collection.objects.link(camera)
    camera.location=(0,0,screen_distance); scene.camera=camera
    scene['blacklight_sequence_json']=metadata; scene['builder_sha256']=_digest(Path(__file__).resolve())
    scene['presentation_projection']=projection
    scene['projection_scope']='Fisheye view of a finite screen, not a 180-degree Kerr all-sky ray trace' if projection=='dome-screen' else 'Orthographic view of a finite ray-traced screen'
    scene['science_acceptance']='NOT_ACCEPTED'
    scene['presentation_settings_json']=json.dumps({'width':width,'height':height,'exposure':exposure,'screen_distance':screen_distance,'projection':projection})
    scene.frame_set(1)
    destination.parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request',type=Path); parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--output',type=Path); parser.add_argument('--projection',choices=['flat','dome-screen'],default='flat')
    parser.add_argument('--width',type=int,default=1920); parser.add_argument('--height',type=int,default=1080)
    parser.add_argument('--exposure',type=float,default=0.); parser.add_argument('--screen-distance',type=float,default=1.7)
    args=parser.parse_args(argv)
    plan=inspect_sequence(args.request)
    if args.audit_only:
        print(json.dumps(plan,indent=2)); return
    if args.output is None:
        parser.error('--output is required unless --audit-only is selected')
    build_scene(plan,args.output,projection=args.projection,width=args.width,height=args.height,
                exposure=args.exposure,screen_distance=args.screen_distance)


if __name__=='__main__':
    main()
