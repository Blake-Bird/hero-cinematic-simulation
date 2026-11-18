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
        self.data         = np.loadtxt("tests/SedovBlastwave/sedov-solution.dat")
        self.R_solution   = self.data[:,0]
        self.rho_solution = self.data[:,1]

    def density_solution(self, R):
        return np.interp(R, self.R_solution, self.rho_solution)

    def runTest(self):
        tolerance = 0.7
        f         = pynbody.load("snap_007")
        # abvc      = f['abvc'].view(type=np.ndarray)
        rho       = f['rho'].view(type=np.ndarray)
        pos       = f['pos'].view(type=np.ndarray)
        x         = pos[:,0].view(type=np.ndarray) - 0.5
        y         = pos[:,1].view(type=np.ndarray) - 0.5
        R         = np.sqrt(x*x + y*y)

        solution  = self.density_solution(R)
        L1        = np.linalg.norm((rho - solution), ord=1)/x.size
        print ('Error norm : ',L1,'    (tolerance : ',tolerance,')')

        self.plot_result(type_of_hydro_solver, R, rho, L1)
        
        self.assertTrue(L1 < tolerance, "Sod shock density L1 error exceeds tolerance")

    def plot_result(self, type_of_sim, R, rho, L1):
        plt.figure(figsize=(8,6))
        rc('font',**{'family':'serif','serif':['Times']})
        rc('text', usetex=True)
        plt.rcParams.update({'font.size': 12})
        plt.title(f"Sedov Blastwave test using {type_of_sim}\n Error norm : {L1}")
        plt.xlabel("Radius, $r$")
        plt.ylabel(r"Density, $\rho$")
        plt.scatter(R, rho)
        # plt.scatter(R, abvc)
        plt.plot(self.R_solution, self.rho_solution,color="black")
        plt.savefig(f"Sedov_{type_of_sim}.png")
        plt.close()

if __name__ == '__main__':
    type_of_hydro_solver = sys.argv[1] if len(sys.argv) > 1 else "Not Specified"
    unittest.main(argv=[sys.argv[0]])
