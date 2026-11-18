import sys
import pynbody
import unittest
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
import matplotlib.pyplot as plt

class SodShockTestCase(unittest.TestCase):

    def setUp(self):
        self.data         = np.loadtxt("tests/SodShock/adsod-solution.dat")
        self.x_solution   = self.data[:,0]
        self.rho_solution = self.data[:,1]


    def density_solution(self, x):
        return np.interp(x, self.x_solution, self.rho_solution)


    def runTest(self):
        tolerance = 1.5e-2 # increased tolerance as MFM is just above 0.005 even if working properly
        f         = pynbody.load("snap_005")
        rho       = f['rho'].view(type=np.ndarray)
        pos       = f['pos'].view(type=np.ndarray)
        x         = pos[:,0].view(type=np.ndarray)

        # Use mask to hide periodic edge effects (only compare central shock solution)
        mask       = np.where((x > 0.25) & (x < 0.75))
        #mask       = np.where((x > 0.0) & (x < 1.0))
        x_masked   = x[mask]
        rho_masked = rho[mask]

        solution  = self.density_solution(x_masked[:])
        L1        = np.linalg.norm((rho_masked - solution), ord=1)/x_masked.size
        print ('Error norm : ',L1,'    (tolerance : ',tolerance,')')

        self.plot_result(x_masked, rho_masked, solution, L1)
        
        self.assertTrue(L1 < tolerance, "SPH Sod shock density L1 error exceeds tolerance")

    def plot_result(self, x_masked, rho_masked, solution, L1):
        plt.figure(figsize=(8,6))
        rc('font',**{'family':'serif','serif':['Times']})
        rc('text', usetex=True)
        plt.rcParams.update({'font.size': 12})
        plt.title(f"Sod Shock test using {type_of_hydro_solver}\n Error norm : {L1}")
        plt.xlabel("x")
        plt.ylabel(r"Density, $\rho$")
        plt.scatter(x_masked, rho_masked)
        plt.plot(self.x_solution, self.rho_solution,color="black")
        plt.savefig(f"SodShock-{type_of_hydro_solver}.png")
        plt.close()

if __name__ == '__main__':
    type_of_hydro_solver = sys.argv[1] if len(sys.argv) > 1 else "Not Specified"
    unittest.main(argv=[sys.argv[0]])
