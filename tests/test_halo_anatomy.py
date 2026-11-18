import unittest
import numpy as np
from core.halo_anatomy import radial_statistics


class HaloAnatomyTests(unittest.TestCase):
    def test_mass_weighted_velocity_dispersion_sufficient_statistics(self):
        radius=np.array([.1,.2,.3,1.5])
        velocity=np.array([[1.,0,0],[3.,0,0],[5.,0,0],[0.,2,0]])
        mass=np.array([1.,2.,1.,3.])
        bins=radial_statistics(radius,velocity,mass,np.array([0.,1.,2.]))
        self.assertEqual(bins[0,0],4.)
        self.assertEqual(bins[1,0],3.)
        mean=bins[2:5,0]/bins[0,0]
        dispersion2=bins[5,0]/bins[0,0]-(mean**2).sum()
        self.assertAlmostEqual(dispersion2,2.)
        self.assertEqual(bins[0,1],3.)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            radial_statistics([1],[[0,0,0]],[-1],[0,2])
        with self.assertRaises(ValueError):
            radial_statistics([1],[[float('nan'),0,0]],[1],[0,2])
