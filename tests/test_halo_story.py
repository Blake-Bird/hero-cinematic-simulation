import unittest
import numpy as np
from core.merger_tree_visual import branch_identity
from core.halo_story import label_positions
from core.cosmic_flythrough import camera_samples, display_limits, perspective_density_raster


class HaloStoryTests(unittest.TestCase):
    def test_alias_follows_massive_progenitor_not_reused_catalogue_id(self):
        tree={'nodes':[{'id':'9007199254740993','mass_msun_h':100},
                       {'id':'8','mass_msun_h':70},{'id':'4','mass_msun_h':40},
                       {'id':'9','mass_msun_h':20}], 'edges':[[1,0],[2,1],[3,0]]}
        for i,node in enumerate(tree['nodes']):node['snapshot']=10-i
        result=branch_identity(tree)
        self.assertEqual([r['alias'] for r in result],['H0','H0','H0','H1'])
        self.assertEqual(result[1]['color'],result[2]['color'])

    def test_camera_standoff_and_fixed_exposure_ignore_numerical_tails(self):
        p,d=camera_samples(np.array([[0.,0,0],[10.,0,0]]),60,stop_distance=2.)
        self.assertAlmostEqual(p[-1,0],8.)
        self.assertTrue(np.all(np.diff(p[:,0])>=0))
        low,high=display_limits([np.r_[np.full(10000,1e-30),np.linspace(1,100,1000)]])
        self.assertGreater(low,0)
        self.assertGreater(high,low)

    def test_collocated_halo_labels_stay_separate_and_in_frame(self):
        rects=label_positions([(300,300)]*4,720,720)
        for a in rects:
            self.assertTrue(0<=a[0]<a[2]<=720)
            self.assertTrue(0<=a[1]<a[3]<=720)
        for i,a in enumerate(rects):
            for b in rects[i+1:]:
                area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                self.assertEqual(area,0)

    def test_uniform_density_volume_has_no_artificial_perspective_gradient(self):
        image=perspective_density_raster(np.full((8,8,8),2.,dtype=float),[.1,.3,.5],[1.,.2,.1],1.,160,96,60.)
        np.testing.assert_allclose(image,2.,rtol=1e-6)
