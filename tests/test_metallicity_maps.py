import unittest
import numpy as np
from core.metallicity_maps import metallicity_projection


class MetallicityTests(unittest.TestCase):
    def test_mass_weighted_fraction_and_true_zero(self):
        xyz=np.array([[.5,.5,.5],[.5,.5,.5],[1.5,.5,.5]])
        value,mass=metallicity_projection([(xyz,np.array([1.,3.,2.]),np.array([.1,.3,0.]))],2,2.,(0,1),2,None,None)
        self.assertAlmostEqual(value[0,0],.25)
        self.assertEqual(value[0,1],0.)
        self.assertEqual(mass.sum(),6.)
        self.assertTrue(np.isnan(value[1,1]))

    def test_rejects_invalid_fraction_and_alignment(self):
        with self.assertRaises(ValueError):metallicity_projection([(np.zeros((1,3)),[1.],[1.1])],8,1.,(0,1),2,None,None)
        with self.assertRaises(ValueError):metallicity_projection([(np.zeros((2,3)),[1.],[.1])],8,1.,(0,1),2,None,None)
