import os
import sys
import math
import g3read
import fnmatch
import pynbody
import unittest
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
import matplotlib.pyplot as plt

class FreefallCollapseTestCase(unittest.TestCase):

    def setUp(self):
        self.mtot          = 1.0
        self.radius        = 1.0
        self.rho0          = 3.0*self.mtot/(4.0*np.pi*math.pow(self.radius, 3.0))
        self.tff           = np.sqrt(3.0*3.1415/32.0/self.rho0)
        self.rad_solution  = np.arange(0.0, self.radius, 0.01*self.radius)
        self.arad_solution = -self.rad_solution

    def loadSnapData(self, filename):
        f = g3read.GadgetFile(filename)
        
        t = f.header.time
        pos  = f.read_new("POS ", -1) 
        mass =  f.read_new("MASS", -1) 
        
        x = pos[:,0] - 1 
        y = pos[:,1] - 1 
        z = pos[:,2] - 1
        
        r = np.sqrt(x*x + y*y + z*z)
        acc = g3read.read_new(filename, "ACCE", -1)
        ar = (acc[:, 0] * x + acc[:, 1] * y + acc[:, 2] * z) / r
        
        return t, r, ar, mass

    def FreefallSolution(self, mfrac):
        r0  = math.pow(mfrac, 0.33333333333)
        r   = np.arange(0.0, 1.0001*r0, 0.01*r0)
        t   = np.arccos(np.sqrt(r/r0)) + np.sqrt(r/r0)*np.sqrt(1.0 - r/r0)
        r  *= self.radius
        t  *= self.tff*2.0/3.14157
        return t,r

    def lagrangian_radii(self, r, m, mfrac):
        rorder      = np.argsort(r)
        m_ordered   = m[rorder]
        mcumulative = np.cumsum(m_ordered)
        mtotal      = mcumulative[-1]
        mlag        = mfrac*mtotal
        index       = np.searchsorted(mcumulative,mlag)
        return 0.5*(r[rorder[index-1]] + r[rorder[index]])


    def interpolate_solution(self, x, xsol, ysol):
        return np.interp(x, xsol, ysol)

    def test_acceleration(self):
        tolerance = 1.0e-2
        t, r, ar, mass = self.loadSnapData("snap_000")
        
        solution  = self.interpolate_solution(r[:], self.rad_solution[:], self.arad_solution[:])
        L1        = np.linalg.norm((ar - solution), ord=1)/r.size
        print ('Error norm : ',L1,'    (tolerance : ',tolerance,')')

        self.plot_acceleration(type_of_hydro_solver, r, ar, L1)
        
        self.assertTrue(L1 < tolerance, "Tree-gravity acceleration L1 error exceeds tolerance")

    def test_lagrangian(self):
        mfrac       = 0.5
        tolerance   = 2.0e-3
        filenames   = []
        tsnaps      = []
        rlagrangian = []

        for file in os.listdir('.'):
            if fnmatch.fnmatch(file, 'snap_*'):
                filenames.append(file)

        for file in sorted(filenames):
            t, r, ar, mass = self.loadSnapData(file)
            rlag = self.lagrangian_radii(r, mass, mfrac)
            tsnaps.append(t)
            rlagrangian.append(rlag)

        t_solution, r_solution = self.FreefallSolution(mfrac)
        solution = self.interpolate_solution(tsnaps, t_solution, r_solution)
        L1       = np.linalg.norm((rlagrangian - solution), ord=1)/r.size
        print ('Error norm : ',L1,'    (tolerance : ',tolerance,')')

        self.plot_lagrangian(type_of_hydro_solver, tsnaps, rlagrangian, t_solution, r_solution, L1)
            
        self.assertTrue(L1 < tolerance, "Freefall collapse Lagrangian radius L1 error exceeds tolerance")

    def plot_lagrangian(self, type_of_sim, tsnaps, rlagrangian, t_solution, r_solution, L1):
        plt.figure(figsize=(8,6))
        rc('font',**{'family':'serif','serif':['Times']})
        rc('text', usetex=True)
        plt.rcParams.update({'font.size': 12})
        plt.title(f"Freefall lagrangian test using {type_of_sim} \n Error norm : {L1}")
        plt.xlabel("Time, $t$")
        plt.ylabel("Radius, $r$")
        plt.scatter(tsnaps, rlagrangian)
        plt.plot(t_solution, r_solution,color="black")
        plt.savefig(f"Freefall_lagran_{type_of_sim}.png")
        plt.close()
    
    def plot_acceleration(self, type_of_sim, r, ar, L1):
        plt.figure(figsize=(8,6))
        rc('font',**{'family':'serif','serif':['Times']})
        rc('text', usetex=True)
        plt.rcParams.update({'font.size': 12})
        plt.title(f"Freefall lagrangian test using {type_of_sim}\n Error norm : {L1}")
        plt.xlabel("Radius, $r$")
        plt.ylabel("Computed acceleration,$a$")
        plt.scatter(r, ar)
        plt.plot(self.rad_solution, self.arad_solution,color="black")
        plt.savefig(f"Freefall_acc_{type_of_sim}.png")
        plt.close()
        
if __name__ == '__main__':
    type_of_hydro_solver = sys.argv[1] if len(sys.argv) > 1 else "Not Specified"
    unittest.main(argv=[sys.argv[0]])
