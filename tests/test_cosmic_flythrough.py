import unittest
import numpy as np
from core.cosmic_flythrough import unwrap_camera_path,camera_samples,perspective_mass_raster


class FlythroughTests(unittest.TestCase):
    def test_periodic_short_path_and_camera_endpoint(self):
        path=unwrap_camera_path([[9,0,0],[1,0,0]],10.)
        np.testing.assert_allclose(path,[[9,0,0],[11,0,0]])
        camera,direction=camera_samples(path,3)
        self.assertGreater(camera[-1,0],10.)
        self.assertLess(camera[-1,0],11.)
        np.testing.assert_allclose(direction,[[1,0,0]]*3)

    def test_projection_rejects_behind_camera_and_retains_mass(self):
        points=np.array([[1.,0,0],[-1.,0,0]])
        raster,count=perspective_mass_raster(points,[2.,5.],[0,0,0],[1,0,0],10.,64,64,60.)
        self.assertEqual(count,1)
        self.assertAlmostEqual(float(raster.sum()),2.,places=5)
        with self.assertRaisesRegex(ValueError,'direction'):
            perspective_mass_raster(points,[2.,5.],[0,0,0],[0,0,0],10.,64,64,60.)


if __name__=='__main__':unittest.main()
