import g3read
import numpy as np
import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as cl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

class asinTest(unittest.TestCase):

    def setUp(self):
        self.data   = np.loadtxt("tests/asin1413816_1/AsinStellarMassFunction.txt")
        self.mass   = self.data[1]
        self.counts = self.data[0]
        self.snapfile = "snap_009"
        self.groupfile = "groups_009/sub_009"
        
    def runTest(self):
        tolerance = 42 # arbitrary value
        f = g3read.GadgetFile(self.snapfile)
        m_star = f.read_new("MASS", 4)

        bins = np.append(self.mass-np.diff(self.mass)[0]/2, self.mass[-1]+np.diff(self.mass)[0]/2)
        hist = np.histogram(m_star, bins=bins)

        difference = np.nansum( (hist[0] - self.counts)**2 ) / np.nansum(self.counts) / self.mass.size

        print("Diff = ", difference)

        # plt.title("Stellar Mass Function - comparison")
        # plt.ylabel("Number of Stars")
        # plt.xlabel("Stellar Mass [10^10 M_sun]")
        # plt.yscale('log')
        # plt.errorbar(self.mass, self.counts,fmt="o",label="Reference with poisson errors", xerr=np.diff(self.mass)[0]/2, yerr=np.sqrt(self.counts))
        # plt.hist(m_star, bins=bins,label="Current Run")
        # plt.legend()
        # plt.savefig("test.png")

        self.plot_PhaseDiagram(f)
        self.plot_Magorrian(f)
        
        self.assertTrue(difference < tolerance, "Asin stellar mass function exceeds tolerance")

    def plot_PhaseDiagram(self, f, NBIN= 500, ColorMap = 'nipy_spectral' ):
        ## read temperature and density:
        rho = f.read_new('RHO ', 0)
        mass = f.read_new('MASS', 0)
        temp = f.read_new('TEMP', 0)
        
        # now plot phase diagram
        fig = plt.figure(figsize=(14,6))
        gs = GridSpec(1, 2, width_ratios=[1, 1.2])

        plt.rc('font',**{'family':'serif','serif':['Times']})
        plt.rc('text', usetex=True)
        plt.rcParams.update({'font.size': 20})      
        plt.suptitle(r'Temperatur vs density at z = %.2f'%(f.header.redshift))
        # first plot as scatter
        ax1 = fig.add_subplot(gs[0,0])
        ax1.tick_params(direction='in', which='both')
        
        #plt.subplots_adjust(wspace=0)
        ax1.scatter(np.log10(rho), np.log10(temp), c = mass, cmap =  ColorMap, s = 1, alpha = 0.01 )
        ax1.tick_params(direction='in', which='both')
        
        ax1.set_title("Scatter plot")
        ax1.set_xlabel('log(Density)')
        ax1.set_ylabel('log(Temperature)')
        ax1.grid()

        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        
        # second plot as grid
        ax2 = fig.add_subplot(gs[0,1])
        ax2.tick_params(direction='in', which='both')
        hist, xedges, yedges, im = ax2.hist2d(np.log10(rho), np.log10(temp), bins = NBIN, cmap = ColorMap, norm = cl.LogNorm())
        ax2.set_title("2d histogram")
        ax2.set_xlabel('log(Density)')
        ax2.grid()
        
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        
        ax2.tick_params(axis='y',which='both',labelleft=False)
  
        plt.colorbar(im, label=r"$N_{gas}$ per bin")
        plt.subplots_adjust(wspace=0.05)
        plt.savefig('Asin_PhaseDiagram.png', bbox_inches = 'tight')
        plt.close() 
    
    def plot_Magorrian(self, f, iplot = 1, Association = 0, CLim = [], xLim = [], yLim = [], ColorMap = 'viridis'):
        subfind = g3read.read_new(self.groupfile, ['SPOS','RHMS','SMST'], 1, is_snap=False)
        
        bh_num = f.header.npart[5]
        bh_pos = f.read_new('POS ', 5)
        bhma = f.read_new('BHMA', 5)

        sbh_num = subfind['SPOS'].shape[0] 
        sbh_pos = subfind['SPOS']
        sbh_smst = subfind['SMST']

        rhms = subfind['RHMS']
    
        dist = np.zeros(max(sbh_num, bh_num), dtype = float)
        star_mass = np.zeros(max(sbh_num, bh_num), dtype = float)
        bh_mass = np.zeros(max(sbh_num, bh_num), dtype = float)

        if Association == 0: ##associate bh to sbhs 
            for j in range(sbh_num):
                #stdout.write('\r Completed: %d%%' % int(j*100/sbh_num)+'% SBH number:'+str(j)+'\x1b[K')
                #stdout.flush()
                d            = np.linalg.norm(sbh_pos[j]-bh_pos, axis = 1)
                argmin_dist  = np.argmin(d)
                mindist      = min(d)
                dist[j] = np.where(mindist < 2*rhms[j], mindist,  0)
                if dist[j]:
                    bh_mass[j]   = bhma[argmin_dist]*1e10
                    star_mass[j] = sbh_smst[j][4]*1e10
        elif Association == 1: ##associate sbhs to bhs 
            for j in range(bh_num):
                #stdout.write('\r Completed: %d%%' % int(j*100/bh_num)+'% SBH number:'+str(j)+'\x1b[K ')
                #stdout.flush()
                d = np.linalg.norm(bh_pos[j]-sbh_pos, axis = 1)
                argmin_dist = np.argmin(d)
                mindist = min(d)
                dist[j] = np.where(mindist < 2*rhms[argmin_dist], mindist,  0)
                if dist[j]:
                    bh_mass[j] = bhma[j]*1e10
                    star_mass[j] = sbh_smst[argmin_dist][4]*1e10
        
        bh_mass = bh_mass[bh_mass != 0]
        star_mass = star_mass[star_mass != 0]
        dist = dist[dist != 0]
        print('Length:', len(bh_mass), len(star_mass))

        c = dist
        ###### cmap normalization and limits :
        if CLim != []:
            vmin = CLim[0]
            vmax = CLim[1]
        else :
            vmin = min(c)
            vmax = max(c)
        #####  limits :
        # if xLim != []:
        #     xlim = xLim 
        # else:
        #     xlim = [sorted(star_mass)[2], max(star_mass)]
        # if yLim != []:
        #     ylim = yLim
        # else:
        #     ylim = [sorted(bh_mass)[2], max(bh_mass)]

        fig = plt.figure(figsize=(8,6))
        plt.rc('font',**{'family':'serif','serif':['Times']})
        plt.rc('text', usetex=True)
        plt.rcParams.update({'font.size': 20})
        
        ax = fig.add_subplot()
        sc = ax.scatter(star_mass, bh_mass, cmap = ColorMap, c=c, norm = cl.Normalize(vmin, vmax) )
        # ax.set_ylim(ylim)
        # ax.set_xlim(xlim)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid()
        plt.colorbar(sc,label="Distance between BH and sBHs")
        #### title and labels
        ax.set_ylabel('BH mass [$M_\odot/h$]', fontsize = 14)
        ax.set_xlabel('Star mass [$M_\odot/h$]', fontsize = 14)
        ax.set_title(r'$M_{BH} \ vs \ M_* \  at \ z = %.2f$'%(f.header.redshift))

        #plotting options 
        plt.savefig('Asin_Magorrian.png',bbox_inches = 'tight')
        plt.close()
        
if __name__ == '__main__':
    unittest.main()
