import unittest
import numpy as np
from core.lens_map import thin_lens_from_surface


class ThinLensTests(unittest.TestCase):
    def test_uniform_sheet_has_zero_shear(self):
        result=thin_lens_from_surface(np.full((16,16),4e8),2.,1e6,2e6,1e6,0.)
        self.assertGreater(result['critical_surface_density_msun_kpc2'],0)
        np.testing.assert_allclose(result['shear1'],0,atol=1e-14)
        np.testing.assert_allclose(result['shear2'],0,atol=1e-14)
        np.testing.assert_allclose(result['magnification_inverse'],(1-result['kappa'])**2)

    def test_x_plane_wave_has_expected_shear(self):
        wave=np.cos(2*np.pi*np.arange(16)/16)
        surface=np.tile(1e8*(2+wave),(16,1))
        result=thin_lens_from_surface(surface,2.,1e6,2e6,1e6,0.)
        np.testing.assert_allclose(result['shear1'],result['kappa']-result['kappa'].mean(),atol=1e-14)
        np.testing.assert_allclose(result['shear2'],0,atol=1e-14)
        with self.assertRaises(ValueError):thin_lens_from_surface(surface,2.,0,2e6,1e6,0.)


if __name__=='__main__':unittest.main()
