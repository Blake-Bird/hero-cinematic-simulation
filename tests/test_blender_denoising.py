import unittest
from types import SimpleNamespace
from core.blender import disable_cycles_denoising


class DenoisingTests(unittest.TestCase):
    def test_disables_scene_and_every_view_layer(self):
        scene=SimpleNamespace(cycles=SimpleNamespace(use_denoising=True),view_layers=[SimpleNamespace(cycles=SimpleNamespace(use_denoising=True)),SimpleNamespace(cycles=SimpleNamespace(use_denoising=True))])
        disable_cycles_denoising(scene)
        self.assertFalse(scene.cycles.use_denoising)
        self.assertTrue(all(not layer.cycles.use_denoising for layer in scene.view_layers))

    def test_tolerates_older_missing_layer_setting(self):
        scene=SimpleNamespace(cycles=SimpleNamespace(),view_layers=[SimpleNamespace(),SimpleNamespace(cycles=SimpleNamespace())])
        disable_cycles_denoising(scene)

    def test_mutes_nested_compositor_denoise_nodes(self):
        denoise=SimpleNamespace(bl_idname='CompositorNodeDenoise',mute=False)
        unrelated=SimpleNamespace(bl_idname='CompositorNodeBlur',mute=False)
        nested=SimpleNamespace(nodes=[denoise,unrelated])
        scene=SimpleNamespace(cycles=SimpleNamespace(use_denoising=True),view_layers=[],
                              node_tree=SimpleNamespace(nodes=[SimpleNamespace(node_tree=nested)]))
        disable_cycles_denoising(scene)
        self.assertTrue(denoise.mute)
        self.assertFalse(unrelated.mute)
