import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from core.void_particle_history import sphere_ids,overlap_metrics


class VoidParticleHistoryTests(unittest.TestCase):
    def test_periodic_sphere_keeps_exact_large_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'s.hdf5'
            with h5py.File(path,'w') as f:
                g=f.create_group('PartType1');g['Coordinates']=np.array([[9900,5000,5000],[100,5000,5000],[4000,5000,5000]],float);g['ParticleIDs']=np.array([2**60+3,2**60+1,2**60+2],dtype='u8')
            ids=sphere_ids([path],[0,5,5],10,.2,max_ids=10)
            np.testing.assert_array_equal(ids,np.array([2**60+1,2**60+3],dtype='u8'))
            with self.assertRaises(MemoryError):sphere_ids([path],[0,5,5],10,.2,max_ids=1)

    def test_overlap_includes_missing_anchor_in_denominator(self):
        a=np.array([1,2,3,4],dtype='u8');b=np.array([2,3,5],dtype='u8')
        metrics,shared=overlap_metrics(a,b,3)
        self.assertEqual(metrics['shared_count'],2);self.assertEqual(metrics['fraction_anchor'],.5)
        self.assertAlmostEqual(metrics['fraction_current'],2/3);self.assertAlmostEqual(metrics['jaccard'],.4)
        np.testing.assert_array_equal(shared,[2,3])
        with self.assertRaises(ValueError):overlap_metrics(np.array([1,1]),b,2)
