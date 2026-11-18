import sys
import math
import pynbody
import unittest
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
import matplotlib.pyplot as plt


class TemperatureStep1aTestCase(unittest.TestCase):

    def setUp(self):
        return

    def energy_solution(self, time, x, rho):
        h = 0.7
        kappa_cv = 1.0*h
        alpha = kappa_cv / np.mean(rho)
        t = time / h
        u0 = 1500.0
        du = 1000.0
        xm = 50.0
        uprime = np.array([u0 + du/2.0 * math.erf((xi - xm)/ (2.0 * math.sqrt(alpha*t))) for xi in x])
        return uprime


    def runTest(self):
        tolerance = 0.65
        f         = pynbody.load("snap_010")
        rho       = f['rho'].view(type=np.ndarray)
        pos       = f['pos'].view(type=np.ndarray)
        u         = f['u'].view(type=np.ndarray)
        x         = pos[:,0].view(type=np.ndarray)

        time = 0.005 #from header, corresponds to snap_010

        # Use mask to hide periodic edge effects (only compare central shock solution)
        mask       = np.where((x > 25) & (x < 75))
        x_masked   = x[mask]
        rho_masked = rho[mask]
        u_masked   = u[mask]

        solution  = self.energy_solution(time, x_masked[:], rho_masked[:])
        L1        = np.linalg.norm((u_masked - solution), ord=1)/x.size
        print ('Error norm : ',L1,'    (tolerance : ',tolerance,')')

        self.plot_result(x_masked, u_masked, solution, L1)
        
        self.assertTrue(L1 < tolerance, "Temperature step L1 error exceeds tolerance")

    def plot_result(self, x_masked, u_masked, solution, L1):
        # Sort the arrays
        sorted_indices = np.argsort(x_masked)
        x_masked= x_masked[sorted_indices]
        solution = solution[sorted_indices]
        u_masked = u_masked[sorted_indices]
        
        plt.figure(figsize=(8,6))
        rc('font',**{'family':'serif','serif':['Times']})
        rc('text', usetex=True)
        plt.rcParams.update({'font.size': 12})
        plt.title(f"Temperature step 1a {type_of_config}\n Error norm : {L1}")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.scatter(x_masked, u_masked)
        plt.plot(x_masked, solution,color="black")
        plt.savefig(f"TemperatureStep1a{type_of_config}.png")
        plt.close()

if __name__ == '__main__':
    type_of_config = sys.argv[1] if len(sys.argv) > 1 else ""
    unittest.main(argv=[sys.argv[0]])
