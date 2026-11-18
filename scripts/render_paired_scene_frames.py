"""Run inside Blender. Render a shared camera recipe at each exact saved frame."""
import json
import math
from pathlib import Path
import sys

import bpy

out = Path(sys.argv[sys.argv.index('--')+1])
out.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
epochs = json.loads(scene['dmsim_hud_epochs_json'])
if len(epochs) != 86:
    raise ValueError('The combined scene should contain 86 saved epochs')
box = float(scene['dmsim_box_mpc_h'])
camera = scene.camera
target = camera.parent
if target is None:
    raise ValueError('Missing camera orbit target')
target.animation_data_clear()
target.location = (box/2,)*3
camera.data.type = 'ORTHO'
camera.data.ortho_scale = box * 1.70
camera.location = (box*1.8, -box*2., box*1.25)
hud = bpy.data.objects.get('Scientific_HUD_Overlay')
if hud is not None:
    hud.hide_render = True
scene.render.resolution_x = scene.render.resolution_y = 800
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.cycles.samples = 20
scene.render.film_transparent = False
scene.world.use_nodes = True
background = scene.world.node_tree.nodes.get('Background')
background.inputs['Color'].default_value = (.003, .009, .016, 1.)
background.inputs['Strength'].default_value = .25
for material in bpy.data.materials:
    if material.name.startswith('Particle dm') and material.node_tree:
        for node in material.node_tree.nodes:
            if node.type == 'EMISSION':
                node.inputs['Color'].default_value = (.10, .65, .89, 1.)
                node.inputs['Strength'].default_value = 2.8
for group in bpy.data.node_groups:
    if group.name.startswith('DarkMatterSim cinematic geometry'):
        for node in group.nodes:
            if node.bl_idname == 'GeometryNodeMeshIcoSphere':
                node.inputs['Radius'].default_value = box * .00076
for index, epoch in enumerate(epochs):
    target.rotation_euler[2] = math.radians(-12 + 24*index/(len(epochs)-1))
    scene.frame_set(epoch['frame'])
    file = out / f'snapshot_{epoch["snapshot"]:03d}.png'
    if file.exists() and file.stat().st_size > 0:
        continue
    scene.render.filepath = str(file)
    bpy.ops.render.render(write_still=True)
(out / 'saved_frames.json').write_text(json.dumps(epochs, indent=2)+'\n')
