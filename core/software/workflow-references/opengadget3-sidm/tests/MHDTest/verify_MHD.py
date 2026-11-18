import g3read
import unittest
import numpy as np
from matplotlib import rc
from astropy.io import ascii
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MHDTest(unittest.TestCase):

    def setUp(self):
        self.tolerance = 0.05
        self.fname = 'snap_021'
        ref_fname = 'tests/MHDTest/AthenaData.txt'
        self.reference_data = ascii.read(ref_fname)

    def plot_comparison(self,POS,RHO,VEL,e,U,BFLD,RMS_array,DIVB,HSML,bb,time):
        print("Starting plotting")
        # 3x2 diagram
        xdim = 3
        ydim = 2 
        
        xlim = [-15,30]

        plt.rcParams.update({'font.size': 14})
        rc('font',**{'family':'serif','serif':['Times']})
        rc('text', usetex=True)
        plt.figure(figsize=(14,8))
        plt.suptitle("Comparison of reference data (red) and simulation data (blue)\n at t = "+time)

        # first plot x vs rho
        plt.subplot(ydim, xdim, 1)
        plt.subplots_adjust(hspace=0.4, wspace=0.35)
        
        plt.xlim(xlim)
        plt.xlabel("x")
        plt.ylabel("rho")
        plt.plot(self.reference_data['x'],self.reference_data['d'],c='r')

        plt.title("RMS RHO="+str(RMS_array["RHO"])[:5])
        plt.scatter(POS[:,0],RHO,color="blue",s=1)

        # second plot x vs Vx,Vz
        plt.subplot(ydim, xdim, 2)
        plt.xlim(xlim)
        plt.xlabel("x")
        plt.ylabel("Vx,Vz")
        plt.plot(self.reference_data['x'],self.reference_data['vx'],label="Vx",c='r',linestyle='--')
        plt.plot(self.reference_data['x'],self.reference_data['vz'],label="Vz",c='r')
        plt.legend(loc='upper right')

        plt.title("RMS Vx="+str(RMS_array["VELx"])[:5]+", Vz="+str(RMS_array["VELz"])[:5])
        plt.scatter(POS[:,0],VEL[:,0],color="blue",s=1)
        plt.scatter(POS[:,0],VEL[:,2],color="blue",s=1)

        # third plot x vs u,E
        plt.subplot(ydim, xdim, 3)
        plt.xlim(xlim)          
        plt.xlabel("x")
        plt.ylabel("u,E")
        plt.plot(self.reference_data['x'],self.reference_data['E'],label="E",c='r',linestyle='--')
        plt.plot(self.reference_data['x'],self.reference_data['P'],label="P",c='r')
        plt.legend(loc='upper right')

        plt.title("RMS e="+str(RMS_array["e"])[:5]+", U="+str(RMS_array["U"])[:5])
        plt.scatter(POS[:,0],e,color="blue",s=1)
        plt.scatter(POS[:,0],U,color="blue",s=1)

        # fourth plot x vs Vy
        plt.subplot(ydim, xdim, 4)
        plt.xlim(xlim)
        plt.xlabel("x")
        plt.ylabel("Vy")
        plt.plot(self.reference_data['x'],self.reference_data['vy'],c='r')

        plt.title("RMS Vy="+str(RMS_array["VELy"])[:5])
        plt.scatter(POS[:,0],VEL[:,1],color="blue",s=1)

        # fith plot Bx,By,Bz
        plt.subplot(ydim, xdim, 5)
        plt.xlim(xlim)
        plt.xlabel("x")
        plt.ylabel("Bx,By,Bz")
        plt.plot(self.reference_data['x'],self.reference_data['bx'],label="Bx",c='r',linestyle='--')
        plt.plot(self.reference_data['x'],self.reference_data['by'],label="By",c='r',linestyle=':')
        plt.plot(self.reference_data['x'],self.reference_data['bz'],label="Bz",c='r')
        plt.legend(loc='upper right')
        
        plt.title("RMS Bx="+str(RMS_array["Bx"])[:5]+", By="+str(RMS_array["By"])[:5]+", Bz="+str(RMS_array["Bz"])[:5])
        plt.scatter(POS[:,0],BFLD[:,0],color="blue",s=1)
        plt.scatter(POS[:,0],BFLD[:,1],color="blue",s=1)
        plt.scatter(POS[:,0],BFLD[:,2],color="blue",s=1)

        # sixth plot x vs div(b)*h/b [%]
        plt.subplot(ydim, xdim, 6)
        plt.xlim(xlim)
        plt.xlabel("x")
        plt.ylim([0,0.2])
        plt.ylabel("div(b)*hsml/b [%]")

        plt.title("RMS divb*hsml/dd="+str(RMS_array["divb*hsml/dd"])[:5])
        plt.scatter(POS[:,0],DIVB*HSML/bb,color="blue",s=1)
        
        print("Saving plot.")
        plt.savefig("MHD_test.png")
        plt.close()
        
    def runTest(self):
        # some constants
        gam1 = (5./3.-1)
        xH=0.76
        xNe=1.0
        yhelium = (1. - xH) / (4. * xH)
        mu = (1. + 4. * yhelium) / (1. + 3. * yhelium + xNe)

        # read snapshot
        f = g3read.GadgetFile(self.fname)
        MASS = f.read_new("MASS", 0)
        POS = f.read_new("POS ", 0)
        VEL = f.read_new("VEL ", 0)
        U = f.read_new("U   ", 0)
        HSML = f.read_new("HSML", 0)
        RHO = f.read_new("RHO ", 0)
        BFLD = f.read_new("BFLD", 0)
        DIVB = np.abs(f.read_new("DIVB", 0)) # abs
        PHI = f.read_new("PHI ", 0) * 50
        time = str(f.header.time)[:5]
            
        # shift positions
        nshift=np.max(POS[:,0])/4. * 2
        POS[:,0]=POS[:,0]-nshift

        # calculate stuff
        bb=np.sqrt(BFLD[:,0]**2+BFLD[:,1]**2+BFLD[:,2]**2)
        vv=np.sqrt(VEL[:,0]**2+VEL[:,1]**2+VEL[:,2]**2)

        e=U*RHO+0.5*bb**2+0.5*vv**2*RHO
        U=gam1*U*RHO
        
        # calculate the differences between data and reference
        x_step = np.mean(np.diff(self.reference_data['x']))
        rho_ar = np.zeros(len(self.reference_data['x']))
        velx_ar = np.zeros(len(self.reference_data['x']))
        vely_ar = np.zeros(len(self.reference_data['x']))
        velz_ar = np.zeros(len(self.reference_data['x']))
        e_ar = np.zeros(len(self.reference_data['x']))
        u_ar = np.zeros(len(self.reference_data['x']))
        bx_ar = np.zeros(len(self.reference_data['x']))
        by_ar = np.zeros(len(self.reference_data['x']))
        bz_ar = np.zeros(len(self.reference_data['x']))

        # construct new arrays and bin datapoints
        for i in range(len(self.reference_data['x'][:-1])):
            index = np.where( abs(POS[:,0]  - self.reference_data['x'][i]) < x_step/2 )[0]
            
            rho_ar[i]=np.mean(RHO[index])
            
            velx_ar[i]=np.mean(VEL[index,0])
            vely_ar[i]=np.mean(VEL[index,1])
            velz_ar[i]=np.mean(VEL[index,2])
            
            e_ar[i]=np.mean(e[index])
            u_ar[i]=np.mean(U[index])
            
            bx_ar[i]=np.mean(BFLD[index,0])
            by_ar[i]=np.mean(BFLD[index,1])
            bz_ar[i]=np.mean(BFLD[index,2])
        
        # calcualate rms values
        rho_rms = np.sqrt(np.mean((self.reference_data['d']-rho_ar)**2))

        velx_rms = np.sqrt(np.mean((self.reference_data['vx']-velx_ar)**2))
        vely_rms = np.sqrt(np.mean((self.reference_data['vy']-vely_ar)**2))
        velz_rms = np.sqrt(np.mean((self.reference_data['vz']-velz_ar)**2))

        e_rms = np.sqrt(np.mean((self.reference_data['E']-e_ar)**2))
        u_rms = np.sqrt(np.mean((self.reference_data['P']-u_ar)**2))

        bx_rms = np.sqrt(np.mean((self.reference_data['bx']-bx_ar)**2))
        by_rms = np.sqrt(np.mean((self.reference_data['by']-by_ar)**2))
        bz_rms = np.sqrt(np.mean((self.reference_data['bz']-bz_ar)**2))

        divb_rms = np.sqrt(np.mean((DIVB*HSML/bb)**2))

        # collect results of rms calculation
        RMS_array = {"RHO":rho_rms,"VELx":velx_rms,"VELy":vely_rms,"VELz":velz_rms,"e":e_rms,"U":u_rms,"Bx":bx_rms,"By":by_rms,"Bz":bz_rms,"divb*hsml/dd":divb_rms}
        
        # plotting
        self.plot_comparison(POS,RHO,VEL,e,U,BFLD,RMS_array,DIVB,HSML,bb, time)

        # self.assertTrue(rho_rms < self.tolerance, "RHO rms exceeds tolerance")
        # self.assertTrue(velx_rms < self.tolerance, "VELX rms exceeds tolerance")
        # self.assertTrue(vely_rms < self.tolerance, "VELY rms exceeds tolerance")
        # self.assertTrue(velz_rms < self.tolerance, "VELZ rms exceeds tolerance")
        # self.assertTrue(e_rms < self.tolerance, "E rms exceeds tolerance")
        # self.assertTrue(u_rms < self.tolerance, "U rms exceeds tolerance")
        # self.assertTrue(bx_rms < self.tolerance, "Bx rms exceeds tolerance")
        # self.assertTrue(by_rms < self.tolerance, "By rms exceeds tolerance")
        # self.assertTrue(bz_rms < self.tolerance, "Bz rms exceeds tolerance")
        self.assertTrue(divb_rms < self.tolerance, "divb*hsml/dd rms exceeds tolerance")
    
if __name__ == '__main__':
    unittest.main()
