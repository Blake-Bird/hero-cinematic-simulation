"""Run inside Blender to audit exact scene timeline, packed code and asset paths."""
import json
from pathlib import Path
import sys

import bpy

scene = bpy.context.scene
metadata = {key: str(scene[key])[:160] for key in scene.keys()}
epochs = json.loads(scene.get('dmsim_hud_epochs_json', '[]'))
for epoch in epochs:
    epoch['redshift'] = 1 / epoch['scale_factor'] - 1
markers = [{'name': m.name, 'frame': m.frame} for m in scene.timeline_markers]
images = []
for image in bpy.data.images:
    if image.source == 'FILE' and image.filepath and not image.packed_file:
        path = Path(bpy.path.abspath(image.filepath))
        images.append({'image': image.name, 'path': str(path), 'exists': path.is_file()})
result = {'scene': bpy.data.filepath, 'objects': len(scene.objects), 'markers': markers,
          'range': [scene.frame_start, scene.frame_end], 'initial_saved_frame': scene.frame_current,
          'camera_type': scene.camera.data.type if scene.camera else None, 'texts': sorted(bpy.data.texts.keys()),
          'custom_properties': metadata, 'external_images': images,
          'halo_inspector_registered': hasattr(bpy.types, 'DMSIM_PT_halo_inspector'), 'epochs': epochs}
if '_with_redshift_timeline' in Path(bpy.data.filepath).stem:
    if len(markers) != 86 or scene.frame_current != 84:
        raise ValueError('The entry scene must open at saved z=14.4 and retain 86 epoch markers')
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    bpy.context.view_layer.update()
    box=float(scene['dmsim_box_mpc_h'])
    corners=[world_to_camera_view(scene,scene.camera,Vector((x,y,z)))
             for x in (0,box) for y in (0,box) for z in (0,box)]
    if any(point.z <= 0 or not 0 <= point.x <= 1 or not 0 <= point.y <= 1 for point in corners):
        raise ValueError('The saved entry camera crops the physical box')
    result['box_corners_inside_camera']=True
if epochs and len(epochs) != 86:
    raise ValueError(f'Unexpected retained epoch count {len(epochs)}')
if not result['halo_inspector_registered'] or any(not image['exists'] for image in images):
    raise ValueError('Missing halo inspector or external image')
test_frames = [e['frame'] for e in epochs if e['snapshot'] in (0, 21, 54, 85)]
if not test_frames:
    test_frames = [scene.frame_start, scene.frame_end]
for frame in test_frames:
    scene.frame_set(frame)
    result.setdefault('observed_frames', []).append(
        {'frame': scene.frame_current, 'camera': scene.camera.name if scene.camera else None,
         'visible_meshes': sum(o.type == 'MESH' and not o.hide_render for o in scene.objects)})
output = Path(sys.argv[sys.argv.index('--') + 1])
result['status'] = 'PASS'
output.write_text(json.dumps(result, indent=2) + '\n')
