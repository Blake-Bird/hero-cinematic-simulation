import matplotlib
import numpy as np
import g3read as g3
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
matplotlib.use("Agg") 
 
def plot_merger():
    files = [f for f in os.listdir(".") if f.startswith("snap_")]
    files_sorted = sorted(files,key=lambda x: int(x.split("_")[1]))

    mass_array = np.zeros(len(files_sorted))
    time_array = np.zeros(len(files_sorted))
    sfr = np.zeros(len(files_sorted) - 1)

    f = g3.GadgetFile(files_sorted[-1])
    t_max = f.header.time

    for i, file in enumerate(files_sorted[:]):
        f = g3.GadgetFile(file)
        
        mass_array[i] = np.sum(g3.read_new(file, "MASS", 4))
        time_array[i] = f.header.time
        
        gas = g3.read_new(file, "POS ", 0)   # (N,3)
        stars_1 = g3.read_new(file, "POS ", 2)
        stars_2 = g3.read_new(file, "POS ", 3)
        stars = np.append(stars_1,stars_2,axis=0)
        new_stars = g3.read_new(file, "POS ", 4)
        print(f.header.npart)

        plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 2)
        # plot x-y plane
        plt.subplot(gs[0, 0])
        plt.scatter(gas[:,0],gas[:,1],s=1,zorder=0)
        plt.scatter(stars[:,0],stars[:,1], color="orange",s=1,zorder=1)
        plt.scatter(new_stars[:,0],new_stars[:,1], color="red",s=1,zorder=2)
        plt.xlabel("x [kpc]")
        plt.ylabel("y [kpc]")
        plt.xlim(-150,150)
        plt.ylim(-150,150)
        
        # plot x-z plane
        plt.subplot(gs[0, 1])
        plt.scatter(gas[:,0],gas[:,2],s=1, label='gas',zorder=1)
        plt.scatter(stars[:,0],stars[:,2],s=1, color="orange", label="stars",zorder=1)
        plt.scatter(new_stars[:,0],new_stars[:,2],s=1,color="red",label="new stars",zorder=2)
        plt.xlabel("x [kpc]")
        plt.ylabel("z [kpc]")
        plt.xlim(-150,150)
        plt.ylim(-150,150)
        plt.legend(loc="upper right")
        
        # calculate simple proxy for star formation rate
        if i > 0:
            dm = mass_array[i] - mass_array[i-1]
            dt = (time_array[i] - time_array[i-1])
            sfr[i-1] = dm / dt
        else:
            dt = 0
            
        # plot starformation rate
        plt.subplot(gs[1,:])
        plt.bar(time_array[:-1], sfr, width = dt, align='edge')
        plt.ylabel("SFR proxy [M_sol/Gyr]")
        plt.xlabel("t [Gyr]")
        plt.xlim(0,t_max)
        plt.ylim(0,0.04)
        
        plt.savefig(file.split("_")[1]+".png")
        plt.close()
        
if __name__ == "__main__":
    plot_merger()
