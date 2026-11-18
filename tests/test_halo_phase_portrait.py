import tempfile
import unittest
from pathlib import Path

import numpy as np
from core.halo_phase_portrait import phase_space_histogram, export_phase_portrait


class PhasePortraitTests(unittest.TestCase):
    def test_radial_velocities_and_weights_preserved(self):
        d = np.column_stack([np.linspace(.1,1.9,40),np.zeros((40,2))])
        v = np.column_stack([np.tile([-10.,10.],20)+50,np.zeros((40,2))])
        weights = np.full(40,2.)
        result = phase_space_histogram(d,v,weights,1.,bins=12)
        self.assertAlmostEqual(result['bulk_velocity_km_s'][0],50.)
        self.assertEqual(result['sampled_particles'],40)
        self.assertAlmostEqual(result['weighted_count'].sum(),80.)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'phase.png'
            export_phase_portrait(result,path,snapshot=3,halo_row=2,redshift=1.)
            self.assertTrue(path.exists())
            with np.load(path.with_suffix('.phase_data.npz')) as data:
                np.testing.assert_allclose(data['weighted_count'],result['weighted_count'])

    def test_insufficient_inner_particles_is_unavailable(self):
        self.assertIsNone(phase_space_histogram(np.ones((3,3)),np.ones((3,3)),np.ones(3),1.))


if __name__ == '__main__':
    unittest.main()
