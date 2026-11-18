"""Create a portable annotated Blender scene from a completed Blacklight handoff.

Run with Blender's Python; source intensity is packed as a display image, not
recomputed or reinterpreted as a resolved GADGET horizon-scale calculation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def main():
    import bpy
    arguments=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('handoff',type=Path)
    parser.add_argument('--output',type=Path,required=True,help='New .blend file')
    parser.add_argument('--width',type=int,default=1920)
    parser.add_argument('--height',type=int,default=1080)
    parser.add_argument('--exposure',type=float,default=0.)
    parser.add_argument('--image-scale',type=float,default=1.)
    args=parser.parse_args(arguments)
    if (not 128<=args.width<=16384 or not 128<=args.height<=16384
            or not math.isfinite(args.exposure) or not -10<=args.exposure<=10
            or not math.isfinite(args.image_scale) or not .1<=args.image_scale<=10):
        raise ValueError('Unsupported canvas, exposure or image scale')
    source=args.handoff.resolve();record=json.loads(source.read_text());output=args.output.resolve()
    root=Path(__file__).resolve().parents[1]
    if output.suffix.lower()!='.blend' or output.exists() or output==root or root in output.parents:
        raise ValueError('Choose a fresh .blend path outside the source tree')
    if record.get('schema')!='blacklight-handoff-v1' or record.get('status')!='COMPLETE':
        raise ValueError('A completed Blacklight handoff is required')
    image=source.parent/'display.png';numerical=source.parent/'image.npz'
    if digest(image)!=record['display']['sha256'] or digest(numerical)!=record['image']['sha256']:
        raise ValueError('Display or quantitative intensity differs from handoff')
    metadata=json.dumps(record,sort_keys=True)
    if len(metadata.encode('utf-8'))>8*1024*1024:
        raise ValueError('Handoff metadata exceeds the portable scene bound')
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene
    scene.render.engine='CYCLES';scene.render.resolution_x=args.width;scene.render.resolution_y=args.height
    scene.render.resolution_percentage=100;scene.view_settings.view_transform='Standard';scene.view_settings.exposure=args.exposure
    if scene.world is None:scene.world=bpy.data.worlds.new('Blacklight background')
    scene.world.color=(0.,0.,0.);scene.world.use_nodes=True
    background=scene.world.node_tree.nodes.get('Background')
    if background is not None:background.inputs['Color'].default_value=(0.,0.,0.,1.)
    # This is a packed screen image. The scene camera does not recompute rays.
    bpy.ops.mesh.primitive_plane_add(size=2*args.image_scale,location=(0,0,0))
    plate=bpy.context.object;plate.name='Blacklight fixed camera image'
    image_data=bpy.data.images.load(str(image),check_existing=False);image_data.pack()
    material=bpy.data.materials.new('Recorded intensity display mapping');material.use_nodes=True
    nodes=material.node_tree.nodes;nodes.clear()
    tex=nodes.new('ShaderNodeTexImage');tex.image=image_data
    tex.interpolation='Linear'
    emission=nodes.new('ShaderNodeEmission');emission.inputs['Strength'].default_value=1.
    surface=nodes.new('ShaderNodeOutputMaterial')
    transparent=nodes.new('ShaderNodeBsdfTransparent')
    mix=nodes.new('ShaderNodeMixShader')
    material.node_tree.links.new(tex.outputs['Color'],emission.inputs['Color'])
    material.node_tree.links.new(tex.outputs['Alpha'],mix.inputs[0])
    material.node_tree.links.new(transparent.outputs['BSDF'],mix.inputs[1])
    material.node_tree.links.new(emission.outputs['Emission'],mix.inputs[2])
    material.node_tree.links.new(mix.outputs['Shader'],surface.inputs['Surface'])
    plate.data.materials.append(material)
    # Magenta backplate makes missing/negative pixels visually distinguishable
    # from physically black, finite pixels in the source image.
    bpy.ops.mesh.primitive_plane_add(size=2*args.image_scale,location=(0,0,-.01))
    mask_plate=bpy.context.object;mask_plate.name='Unsupported rays magenta'
    mask_material=bpy.data.materials.new('Unsupported pixel indicator');mask_material.use_nodes=True
    mn=mask_material.node_tree.nodes;mn.clear()
    me=mn.new('ShaderNodeEmission');me.inputs['Color'].default_value=(.8,0.,.5,1.);me.inputs['Strength'].default_value=1.
    mo=mn.new('ShaderNodeOutputMaterial');mask_material.node_tree.links.new(me.outputs['Emission'],mo.inputs['Surface'])
    mask_plate.data.materials.append(mask_material)
    camera_data=bpy.data.cameras.new('Display camera');camera_data.type='ORTHO'
    camera=bpy.data.objects.new('Display camera',camera_data);scene.collection.objects.link(camera)
    camera.location=(0,0,10);scene.camera=camera
    aspect=args.width/args.height
    camera_data.ortho_scale=max(2.4*args.image_scale,2.7*args.image_scale*aspect)
    label='KERR / '+('INDEPENDENT GRMHD' if record['model_type']=='simulation' else 'ANALYTIC ILLUSTRATION')
    text_data=bpy.data.curves.new('Model and origin','FONT');text_data.body=label
    text_data.align_x='CENTER';text_data.size=.08*args.image_scale
    text_obj=bpy.data.objects.new('Model and origin',text_data);scene.collection.objects.link(text_obj)
    text_obj.location=(0,-1.12*args.image_scale,.02)
    text_material=bpy.data.materials.new('Model label emission');text_material.use_nodes=True
    tn=text_material.node_tree.nodes;tn.clear()
    te=tn.new('ShaderNodeEmission');te.inputs['Color'].default_value=(.8,.9,1.,1.)
    to=tn.new('ShaderNodeOutputMaterial');text_material.node_tree.links.new(te.outputs['Emission'],to.inputs['Surface'])
    text_data.materials.append(text_material)
    mask_text=text_data.copy();mask_text.body='MAGENTA = UNSUPPORTED PIXEL';mask_text.size=.045*args.image_scale
    mask_obj=bpy.data.objects.new('Unsupported pixel legend',mask_text);scene.collection.objects.link(mask_obj)
    mask_obj.location=(0,-1.2*args.image_scale,.02)
    scene['blacklight_handoff_json']=metadata
    scene['handoff_sha256']=digest(source);scene['display_sha256']=digest(image)
    scene['scene_builder_sha256']=digest(Path(__file__).resolve())
    scene['quantitative_npz_sha256']=digest(numerical)
    scene['display_interpretation']='Packed false-color screen; transparent unsupported pixels reveal magenta; not a new ray trace or proof of physical accuracy'
    scene['science_acceptance']='NOT_ACCEPTED'
    scene['display_settings_json']=json.dumps({'width':args.width,'height':args.height,
        'exposure':args.exposure,'image_scale':args.image_scale,'camera':'orthographic display camera'})
    output.parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))


if __name__=='__main__':
    main()
