"""Blender-only 32-point production-helper scene and independent reload check.

native_checks.calculate launches this file with Blender --python. Its input
is the worker's points.json in explicit display units. Build mode writes one
editable .blend and one PNG; verify mode reopens the .blend in a fresh process
and compares every point, then writes verified-scene.json. Production material,
geometry and camera helpers are used; the marker radius is display styling.
"""
from pathlib import Path
import json
import math
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    """Build or independently reopen the scene selected by the fixed worker."""
    import bpy
    from core.blender import (_make_material, _make_geometry_nodes,
        _add_lighting_and_camera, disable_cycles_denoising, set_output_format)
    args=sys.argv[sys.argv.index('--')+1:];mode,path=args
    request=Path(path).resolve();output=request.parent
    points=json.loads(request.read_text())['points']
    assert len(points)==32 and all(len(p)==3 and all(math.isfinite(x) for x in p) for p in points)
    if mode=='build':
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mesh=bpy.data.meshes.new('Actual input points');mesh.from_pydata(points, [], []);mesh.update()
        obj=bpy.data.objects.new('Optional32Particles',mesh);bpy.context.scene.collection.objects.link(obj)
        obj.data.materials.append(_make_material('dm'))
        _make_geometry_nodes(obj,'dm',1.,crop_periodic=False)
        for modifier in obj.modifiers:
            if modifier.type=='NODES':
                for node in modifier.node_group.nodes:
                    if node.bl_idname=='GeometryNodeMeshIcoSphere':node.inputs['Radius'].default_value=.025
        scene=bpy.context.scene
        _add_lighting_and_camera(scene,(.5,.5,.5),1.,2,orbit_camera=False)
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=4
        disable_cycles_denoising(scene);set_output_format(scene,'PNG')
        scene.render.resolution_x=256;scene.render.resolution_y=192;scene.render.resolution_percentage=100
        scene.render.threads_mode='FIXED';scene.render.threads=2
        scene.render.filepath=str(output/'preview.png')
        scene['origin']='synthetic_visual_control';scene['science_acceptance']='NOT_ACCEPTED'
        scene['display_glyph_radius']=.025;scene['coordinate_units']='display units'
        bpy.context.preferences.filepaths.save_version=0
        bpy.ops.wm.save_as_mainfile(filepath=str(output/'particles.blend'))
        bpy.ops.render.render(write_still=True)
    elif mode=='verify':
        bpy.ops.wm.open_mainfile(filepath=str(output/'particles.blend'))
        obj=bpy.data.objects['Optional32Particles'];actual=[list(v.co) for v in obj.data.vertices]
        residual=max(abs(a-b) for p,q in zip(actual,points) for a,b in zip(p,q))
        matches=len(actual)==32 and residual<=1e-6
        assert matches and bpy.context.scene.camera is not None
        assert bpy.context.scene['origin']=='synthetic_visual_control'
        result=dict(particles=len(actual),positions_match=matches,maximum_coordinate_roundtrip_error=residual,
            tolerance_display_units=1e-6,cameras=sum(o.type=='CAMERA' for o in bpy.context.scene.objects),
            lights=sum(o.type=='LIGHT' for o in bpy.context.scene.objects),
            scope='Fresh Blender process reopened the saved scene and compared every original point in order; float32 mesh roundtrip tolerance is explicit.')
        (output/'verified-scene.json').write_text(json.dumps(result,indent=2))
    else:
        raise ValueError('Unknown scene operation')


if __name__=='__main__':
    main()
