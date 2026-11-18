"""Run inside Blender to write a losslessly compressed copy without scene edits."""
from pathlib import Path
import json,sys
import bpy
out=Path(sys.argv[sys.argv.index('--')+1]);receipt=Path(sys.argv[sys.argv.index('--')+2])
scene=bpy.context.scene
before={'objects':len(scene.objects),'meshes':len(bpy.data.meshes),'frame':scene.frame_current,'range':[scene.frame_start,scene.frame_end],'markers':[(m.name,m.frame) for m in scene.timeline_markers],'packed_images':sum(bool(i.packed_file) for i in bpy.data.images),'texts':sorted(bpy.data.texts.keys())}
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(out),compress=True,copy=True,relative_remap=False,check_existing=False)
receipt.write_text(json.dumps({'operation':'Blender save-as compressed copy; no scene edits','before':before,'compressed_file':str(out)},indent=2)+'\n')
