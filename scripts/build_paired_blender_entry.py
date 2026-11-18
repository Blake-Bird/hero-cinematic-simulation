"""Run in Blender to publish the readable entry view from the verified rich scene."""
import json
from pathlib import Path
import sys

import bpy

repo = Path(sys.argv[sys.argv.index('--')+1]).resolve()
model = sys.argv[sys.argv.index('--')+2]
sys.path.insert(0,str(repo/'core'))
import blender

base = Path(bpy.data.filepath).resolve().parent
original = base/'blender_universe.blend'
manifest = base/'blender_universe.json'
destination = base/f'{model}_Explore_z_014p400_with_redshift_timeline.blend'


def builder(source, output, render):
    blender.configure_saved_universe_view(bpy, bpy.context.scene)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))


blender.publish_scene_product(manifest,destination,builder,input_schema='combined-scene-v1',
    algorithm=repo/'core/blender.py',manifest_verifier=blender.verify_combined_scene_manifest)
print('ENTRY_SCENE',destination,flush=True)
