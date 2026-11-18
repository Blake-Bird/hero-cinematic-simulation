import unittest
import numpy as np
from core.filament_flythrough import hessian_ridge_mask,ridge_route


class FilamentFlythroughTests(unittest.TestCase):
    def test_periodic_route_is_connected_and_stays_on_ridge(self):
        mask=np.zeros((16,16,16),bool)
        for x in [12,13,14,15,0,1,2]:mask[x,4,4]=True
        path,info=ridge_route(mask,[.1,4.,4.],16.,max_host_distance_mpc_h=1.,path_length_mpc_h=5.)
        self.assertTrue(mask[tuple(path.T)].all())
        delta=(np.diff(path,axis=0)+8)%16-8
        self.assertTrue(np.all(np.max(abs(delta),axis=1)<=1))
        self.assertGreaterEqual(info['ridge_path_length_mpc_h'],3.)
        self.assertLess(info['host_to_ridge_distance_mpc_h'],1.)

    def test_rejects_absent_or_distant_ridge(self):
        mask=np.zeros((16,)*3,bool);mask[8,8,8]=True
        with self.assertRaisesRegex(ValueError,'No ridge cell'):ridge_route(mask,[0,0,0],16,max_host_distance_mpc_h=1,path_length_mpc_h=4)
        with self.assertRaises(ValueError):hessian_ridge_mask(np.full((16,)*3,np.nan),sigma=1,threshold_factor=1,density_percentile=50)
