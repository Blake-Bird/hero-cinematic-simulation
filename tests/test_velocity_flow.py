import unittest
import numpy as np
from core.velocity_flow import flow_bins


class VelocityFlowTests(unittest.TestCase):
    def test_weighted_mean_keeps_stationary_particles(self):
        pos=np.array([[.2,.2,.1],[.2,.2,.9],[.8,.8,.5]])
        vel=np.array([[10.,0,0],[0.,0,0],[100.,0,0]])
        mass=np.array([1.,3.,1.])
        total,count,mx,my=flow_bins(pos,vel,mass,1.,8,(0,1))
        self.assertEqual(count[1,1],2)
        self.assertAlmostEqual(mx[1,1]/total[1,1],2.5)
        self.assertEqual(my[1,1],0.)
        slab=flow_bins(pos,vel,mass,1.,8,(0,1),slice_axis=2,slice_center=0.,slice_thickness=.3)
        self.assertEqual(slab[1].sum(),2)
        self.assertAlmostEqual(slab[2][1,1]/slab[0][1,1],2.5)

    def test_nonfinite_velocity_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'invalid'):
            flow_bins([[.1,.1,.1]],[[float('nan'),0,0]],[1.],1.,8,(0,1))


if __name__=='__main__':unittest.main()
