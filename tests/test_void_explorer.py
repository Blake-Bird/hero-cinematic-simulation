import unittest
import numpy as np
from core.void_explorer import density_minima,nearest_minimum,enclosed_density


class VoidExplorerTests(unittest.TestCase):
    def test_periodic_nearest_minimum_across_boundary(self):
        field=np.ones((16,)*3);field[15,4,4]=.2;field[8,4,4]=.1
        minima=density_minima(field,.8)
        np.testing.assert_array_equal(nearest_minimum(field,minima,[0,4,4],2),[15,4,4])
        self.assertIsNone(nearest_minimum(field,minima,[4,10,10],1))

    def test_spherical_aperture_and_box_limit(self):
        field=np.ones((16,)*3);center=np.array([0,0,0]);offset=(np.arange(16)+8)%16-8
        d2=offset[:,None,None]**2+offset[None,:,None]**2+offset[None,None,:]**2
        field[d2<=4]=.1
        profile=enclosed_density(field,center,16,.5)
        self.assertEqual(profile['radius_mpc_h'],2.)
        self.assertEqual(profile['status'],'MEASURED_GRID_APERTURE')
        profile=enclosed_density(np.full((16,)*3,.1),center,16,.8)
        self.assertEqual(profile['radius_mpc_h'],7.)
        self.assertEqual(profile['status'],'LOWER_BOUND')

    def test_uniform_field_has_no_identifiable_minimum(self):
        self.assertEqual(len(density_minima(np.ones((16,)*3)*.5,.8)),0)
