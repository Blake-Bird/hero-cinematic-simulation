import unittest
import numpy as np
from core.halo_fingerprint import common_aperture_histogram
from core.analysis import _mutual_best_halo_matches


class HaloFingerprintTests(unittest.TestCase):
    def test_common_aperture_uses_periodic_geometry(self):
        p=np.array([[9.9,5.,5.],[.2,5.,5.],[5.,5.,5.]])
        image,mass,count=common_aperture_histogram(p,[2.,3.,7.],[0.,5.,5.],10.,1.,16,[0.,.15,.5,1.])
        self.assertEqual(image.sum(),5.)
        np.testing.assert_array_equal(mass,[2.,3.,0.])
        np.testing.assert_array_equal(count,[1,1,0])

    def test_reciprocal_best_excludes_nonmutual(self):
        a={'dm_ids':np.array([1,2,3,4,5,6]),'dm_group_indices':np.array([0,0,0,1,1,1]),'group_indices':np.array([0,1]),'n_dm':np.array([3,3])}
        b={'dm_ids':np.array([1,2,3,4,5,6]),'dm_group_indices':np.array([0,0,0,0,1,1]),'group_indices':np.array([0,1]),'n_dm':np.array([4,2])}
        result=_mutual_best_halo_matches(a,b,1,.1)
        np.testing.assert_array_equal(result['a'],[0,1])
        np.testing.assert_array_equal(result['b'],[0,1])
        np.testing.assert_array_equal(result['shared'],[3,2])
