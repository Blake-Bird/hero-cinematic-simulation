import sys
import pynbody
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
import matplotlib.pyplot as plt

class SoundwaveTestCase(unittest.TestCase):

    def setUp(self):
        self.twopi      = 2.0*np.pi
        self.rho0       = 1.0
        self.press0     = 1.0
        self.gamma      = 1.6666666666666666666
        self.wavelength = 1.0
        self.amp        = 0.0002
        self.csound     = np.sqrt(self.gamma*self.press0/self.rho0)
        self.k          = 2.0*np.pi/self.wavelength
        self.omega      = 2.0*np.pi*self.csound/self.wavelength

    def density_solution(self, x, t):
        return self.rho0*(1.0 + self.amp*np.sin(self.k*x - self.omega*t))

    def runTest(self):
        tolerance = 5.0e-4
        f         = pynbody.load("snap_003")
        rho       = f['rho'].view(type=np.ndarray)
        pos       = f['pos'].view(type=np.ndarray)
        x         = pos[:,0].view(type=np.ndarray)
        simTime   = f.header.time
        solution  = self.density_solution(x[:], simTime)
        L1        = np.linalg.norm((rho - solution - np.mean(rho)+1), ord=1)/x.size # there might be some slight offset in the density due to different kernel for MFM
        print ('Error norm : ',L1,'    (tolerance : ',tolerance,')')
        
        self.plot_result(x, rho, solution, L1)
        
        self.assertTrue(L1 < tolerance, "SPH soundwave density L1 error exceeds tolerance")

    def plot_result(self, x, rho, solution, L1):
        # Sort the arrays
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        rho = rho[sorted_indices]
        solution = solution[sorted_indices]

        plt.figure(figsize=(8,6))
        rc('font',**{'family':'serif','serif':['Times']})
        rc('text', usetex=True)
        plt.rcParams.update({'font.size': 12})
        plt.title(f"Soundwave test using {type_of_hydro_solver}\n Error norm : {L1}")
        plt.xlabel("x")
        plt.ylabel(r"Density, $\rho$")
        plt.scatter(x, rho)
        plt.plot(x, solution,color="black")
        plt.savefig(f"Soundwave_{type_of_hydro_solver}.png")
        plt.close()
        
if __name__ == '__main__':
    type_of_hydro_solver = sys.argv[1] if len(sys.argv) > 1 else "Not Specified"
    unittest.main(argv=[sys.argv[0]])
