import unittest
from core.merger_tree_map import map_location


class MergerTreeMapTests(unittest.TestCase):
    def setUp(self):
        self.meta={'box_size_mpc_h':10.,'projection':'xy','slice':{'line_of_sight_axis':2,'center_mpc_h':.1,'thickness_mpc_h':.6}}

    def test_periodic_slab_and_screen_orientation(self):
        value=map_location({'position_mpc_h':[2.,3.,9.9]},self.meta)
        self.assertEqual(value['status'],'IN_SLAB')
        self.assertEqual(value['x'],20.)
        self.assertEqual(value['y'],70.)
        self.assertEqual(map_location({'position_mpc_h':[2.,3.,5.]},self.meta)['status'],'OUTSIDE_SLAB')
        self.assertEqual(map_location({},self.meta)['status'],'POSITION_UNAVAILABLE')

    def test_rejects_wrong_line_of_sight(self):
        self.meta['slice']['line_of_sight_axis']=1
        with self.assertRaises(ValueError):map_location({'position_mpc_h':[1.,2.,3.]},self.meta)
