import unittest
import numpy as np
from core.allsky_map import angular_mass_grid


class AllSkyTests(unittest.TestCase):
    def test_equal_area_mass_conservation_and_observer_coincidence(self):
        points=np.array([[1,0,0],[-1,0,0],[0,0,1],[0,0,-1],[0,0,0]],float)
        weights=np.array([1,2,3,4,5],float)
        result=angular_mass_grid(points,weights,8,4)
        self.assertAlmostEqual(result['solid_angle_sr_per_pixel']*32,4*np.pi)
        self.assertAlmostEqual(result['mass_per_sr'].sum()*result['solid_angle_sr_per_pixel'],10.)
        self.assertEqual(result['excluded_observer_coincident_count'],1)

    def test_invalid_weight_fails(self):
        with self.assertRaises(ValueError):angular_mass_grid([[1,0,0]],[-1],8,4)


if __name__=='__main__':unittest.main()
