"""Export an offline Blender timeline of sampled saved DM; no trajectory interpolation."""
import sys,json,os,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,h5py
from core.analysis import snapshot_ids,read_snapshot_header
from core.foundation import file_sha256
root=Path(sys.argv[1]).resolve();dest=root/'analysis/05_Movies_and_3D/Refreshed_Blender_timeline';dest.mkdir(parents=True,exist_ok=True)
rows=[]
for s in snapshot_ids(root):
 h,files=read_snapshot_header(root,s);xyz=[];ids=[]
 for p in files:
  with h5py.File(p) as f:
   if 'PartType1' not in f:continue
   xyz.append(f['PartType1/Coordinates'][::256]/1000);ids.append(f['PartType1/ParticleIDs'][::256])
 name=f'sampled_epoch_{s:03d}.npz';np.savez_compressed(dest/name,positions_comoving_mpc_h=np.vstack(xyz),particle_ids=np.concatenate(ids))
 rows.append({'snapshot':s,'redshift':1/float(h['Time'])-1,'scale_factor':float(h['Time']),'asset':name,'sha256':file_sha256(dest/name),'source_files':[{'path':str(p.relative_to(root)),'sha256':file_sha256(p)} for p in files]})
 print('Blender sample',s,flush=True)
contract={'schema':'saved-dmo-blender-atlas-v1','epochs':rows,'box_comoving_mpc_h':float(h['BoxSize'])/1000,'sampling':'Every 256th type-1 row in each native shard; exact IDs retained. Systematic row sample, not a halo completeness measurement.','time':'8 frames per saved epoch at 24 fps; no unsaved particle motion or equal-cosmic-time spacing.','original_manifest_sha256':file_sha256(root/'manifest.json'),'algorithm_sha256':file_sha256(__file__)}
(dest/'scene_source.json').write_text(json.dumps(contract,indent=2))
blender_source=r'''
import bpy,json,math,numpy as np
from pathlib import Path
from mathutils import Vector
base=Path(__file__).resolve().parent;data=json.loads((base/'scene_source.json').read_text());box=data['box_comoving_mpc_h']
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=16;scene.render.resolution_x=1000;scene.render.resolution_y=1000;scene.render.resolution_percentage=100;scene.render.fps=24;scene.frame_start=1;scene.frame_end=len(data['epochs'])*8
scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.002,.006,.013,1);scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.2
mat=bpy.data.materials.new('Sampled dark matter · display emission');mat.diffuse_color=(.15,.8,.95,1);mat.use_nodes=True;nodes=mat.node_tree.nodes;nodes.clear();out=nodes.new('ShaderNodeOutputMaterial');em=nodes.new('ShaderNodeEmission');em.inputs['Color'].default_value=(.1,.7,.95,1);em.inputs['Strength'].default_value=1.7;mat.node_tree.links.new(em.outputs[0],out.inputs[0])
ng=bpy.data.node_groups.new('Saved particle glyphs','GeometryNodeTree');ng.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry');ng.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
n=ng.nodes;l=ng.links;inp=n.new('NodeGroupInput');output=n.new('NodeGroupOutput');points=n.new('GeometryNodeMeshToPoints');points.mode='VERTICES';ico=n.new('GeometryNodeMeshIcoSphere');ico.inputs['Radius'].default_value=box*.0018;ico.inputs['Subdivisions'].default_value=1;sm=n.new('GeometryNodeSetMaterial');sm.inputs['Material'].default_value=mat;ins=n.new('GeometryNodeInstanceOnPoints');l.new(inp.outputs['Geometry'],points.inputs['Mesh']);l.new(ico.outputs['Mesh'],sm.inputs['Geometry']);l.new(sm.outputs['Geometry'],ins.inputs['Instance']);l.new(points.outputs['Points'],ins.inputs['Points']);l.new(ins.outputs['Instances'],output.inputs['Geometry'])
for i,row in enumerate(data['epochs']):
 a=np.load(base/row['asset']);p=a['positions_comoving_mpc_h'];mesh=bpy.data.meshes.new('Saved epoch %03d'%row['snapshot']);mesh.from_pydata(p.tolist(),[],[]);mesh.update();o=bpy.data.objects.new('z %.5f · snapshot %03d'%(row['redshift'],row['snapshot']),mesh);scene.collection.objects.link(o);mod=o.modifiers.new('Particle glyphs','NODES');mod.node_group=ng;o['snapshot']=row['snapshot'];o['redshift']=row['redshift'];o['exact_id_asset']=row['asset'];start=i*8+1;end=start+8
 for f,hide in [(1,True),(start,False),(end,True)]:
  o.hide_viewport=hide;o.hide_render=hide;o.keyframe_insert('hide_viewport',frame=f);o.keyframe_insert('hide_render',frame=f)
 scene.timeline_markers.new('z=%.4f · saved %03d'%(row['redshift'],row['snapshot']),frame=start)
# Transparent box edges in comoving units.
line=bpy.data.materials.new('Box boundary');line.diffuse_color=(.2,.35,.45,1)
verts=[(x,y,z) for x in (0,box) for y in (0,box) for z in (0,box)]
for i,p in enumerate(verts):
 for j,q in enumerate(verts):
  if j>i and sum(p[k]!=q[k] for k in range(3))==1:
   cv=bpy.data.curves.new('Box edge','CURVE');cv.dimensions='3D';cv.bevel_depth=box*.0007;sp=cv.splines.new('POLY');sp.points.add(1);sp.points[0].co=(*p,1);sp.points[1].co=(*q,1);ob=bpy.data.objects.new('Comoving box boundary',cv);scene.collection.objects.link(ob);ob.data.materials.append(line)
bpy.ops.object.camera_add(location=(box*1.8,-box*1.5,box*1.35));cam=bpy.context.object;cam.rotation_euler=((Vector((box/2,)*3)-cam.location).to_track_quat('-Z','Y').to_euler());cam.data.type='ORTHO';cam.data.ortho_scale=box*1.85;scene.camera=cam
scene['Scientific scope']='Sampled native DM glyphs; display emission is artificial. No galaxy luminosity, interpolated dynamics, or LCDM comparison.';scene['Units']='comoving Mpc/h; exact native IDs in adjacent NPZ assets';scene['Source']='scene_source.json'
scene.frame_set(scene.frame_end)
for screen in bpy.data.screens:
 for area in screen.areas:
  if area.type=='VIEW_3D':area.spaces.active.region_3d.view_distance=box*2;area.spaces.active.region_3d.view_location=Vector((box/2,)*3);area.spaces.active.shading.type='MATERIAL'
text=bpy.data.texts.new('READ_ME');text.write(json.dumps({k:v for k,v in data.items() if k!='epochs'},indent=2)+'\nAll 86 saved epochs retained. Eight frames per epoch, 24 fps. Use the timeline markers to jump to z=14.4 and z=0. No interpolation.\n')
bpy.ops.wm.save_as_mainfile(filepath=str(base/'Dark_matter_full_saved_timeline.blend'));scene.render.filepath=str(base/'Preview_z0.png');bpy.ops.render.render(write_still=True)
'''
(dest/'build_scene.py').write_text(blender_source)
subprocess.run(['/Applications/Blender.app/Contents/MacOS/Blender','--background','--threads','4','--python',str(dest/'build_scene.py')],check=True,stdout=(dest/'build.log').open('w'),stderr=subprocess.STDOUT)
contract['scene_sha256']=file_sha256(dest/'Dark_matter_full_saved_timeline.blend');contract['builder_sha256']=file_sha256(dest/'build_scene.py');contract['preview_sha256']=file_sha256(dest/'Preview_z0.png');(dest/'scene_source.json').write_text(json.dumps(contract,indent=2));(dest/'README.md').write_text('# Refreshed dark matter timeline\n\nOpen Dark_matter_full_saved_timeline.blend in Blender. All 86 saved epochs are embedded as meshes. Timeline markers show the redshift. Each epoch is held for eight frames at 24 fps; there is no interpolation of particle trajectories. The final saved frame shows z=0.\n\nCoordinates are comoving Mpc/h. One in every 256 native rows per shard is shown with a display glyph. Exact sampled particle IDs and positions are retained in the NPZ assets. Artificial emission colors are for visibility, not luminosity. No external geometry files or Python execution are needed to view the saved scene.\n\nscene_source.json records source hashes and samples; build_scene.py can rebuild the scene.\n')
print('Blender timeline ready',dest,flush=True)
