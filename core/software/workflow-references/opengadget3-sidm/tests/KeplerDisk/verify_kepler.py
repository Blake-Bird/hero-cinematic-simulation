import sys
import pynbody
import unittest
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
import matplotlib.pyplot as plt

class KeplerDiskTestCase(unittest.TestCase):

    def setUp(self):
        self.density_solution = 1.0
        self.center           = [4.0, 4.0, 0.0]

    def runTest(self):
        tolerance = 0.3
        f         = pynbody.load("snap_048.hdf5")
        rho       = f['rho'].view(type=np.ndarray)
        pos       = f['pos'].view(type=np.ndarray)
        r         = np.sqrt((pos[:,0]-self.center[0])**2 + (pos[:,1]-self.center[1])**2 + (pos[:,2]-self.center[2])**2)

        solution  = self.density_solution
        L1        = np.linalg.norm((rho - solution), ord=1)/r.size
        print ('Error norm : ',L1,'    (tolerance : ',tolerance,')')

        self.plot_result(r, rho, L1)

        self.assertTrue(L1 < tolerance, f"{type_of_hydro_solver} Kepler disk density L1 error exceeds tolerance")

    def plot_result(self, r, rho, L1):
        plt.figure(figsize=(8,6))
        rc('font',**{'family':'serif','serif':['Times']})
        rc('text', usetex=True)
        plt.rcParams.update({'font.size': 12})
        plt.title(f"Kepler disk test using {type_of_hydro_solver}\n Error norm : {L1}")
        plt.xlabel("Radius, r")
        plt.ylabel(r"Density, $\rho$")
        plt.scatter(r, rho)
        plt.plot([min(r),max(r)], [self.density_solution,self.density_solution],color="black")
        plt.savefig(f"KeplerDisk-{type_of_hydro_solver}.png")
        plt.close()

if __name__ == '__main__':
    type_of_hydro_solver = sys.argv[1] if len(sys.argv) > 1 else "NotSpecified"
    unittest.main(argv=[sys.argv[0]])
