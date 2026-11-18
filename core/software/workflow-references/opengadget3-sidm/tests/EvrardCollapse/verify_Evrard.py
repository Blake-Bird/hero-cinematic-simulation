import matplotlib
matplotlib.use("Agg")  
import unittest 
import numpy as np
import g3read as g3
import subprocess
import matplotlib.pyplot as plt

def show_profiles():
    # ----- Units -----
    M = 1.5815
    R = 1.0
    G = 1.0
    eps_star = G * M / R
    rho_star = 3.0 * M / (4.0 * np.pi * R**3)
    p_star   = eps_star * rho_star
    t_star   = np.sqrt(R**3 / (G * M))
    gam1     = (5.0/3.0 - 1.0)

    print(f" =>  M={M}, rho_*={rho_star}, p_*={p_star}, t_*={t_star}, eps_*={eps_star}")

    for k in range(1, 24):  # 1..23 inclusive
        snr = f"{k:02d}"
        snap = f"snap_0{snr}"
        ppm_file = f"tests/EvrardCollapse/PPM/PPM_{snr}"

        print (f"name= {snap}")
        # ----- Read snapshot fields -----
        f = g3.GadgetFile(snap)
        try:
            h = f.read_new("HEAD", 0)
        except Exception:
            h = getattr(f, "header", {})
        x    = f.read_new("POS ", 0)   # (N,3)
        rho  = f.read_new("RHO ", 0)   # (N,)
        u    = f.read_new("U   ", 0)   # (N,)
        m    = f.read_new("MASS", 0)   # (N,)
        hsml = f.read_new("HSML", 0)   # (N,)

        # time/title
        time_val = None
        if isinstance(h, dict) and "time" in h:
            time_val = float(h["time"])
        elif hasattr(h, "time"):
            time_val = float(getattr(h, "time"))
        title = f"t/t*={time_val/t_star:.2f}" if time_val is not None else ""

        # pressure
        p = gam1 * u * rho

        # radius
        rr = np.sqrt(np.sum(x[:, :3]**2, axis=1))

        # ----- Read PPM profile (r, dens, presh, dummy) -----
        try:
            ppm = np.loadtxt(ppm_file, comments="#")
            r_ppm   = ppm[:, 0]
            dens_ppm= ppm[:, 1]
            pres_ppm= ppm[:, 2]
        except Exception as e:
            r_ppm = dens_ppm = pres_ppm = None
            print(f"[warn] Could not read {ppm_file}: {e}")

        # ----- Plot: log–log, limits like IDL -----
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.0002, 2.0)
        ax.set_ylim(0.04, 4000.0)
        ax.set_xlabel('r / R', fontsize=14)
        ax.set_ylabel('rho/rho* , p/p*/10', fontsize=14)
        if title:
            ax.set_title(title, fontsize=14)

        # Particle scatters (black)
        ax.plot(rr / R, rho / rho_star, 'o', ms=2, color='k', label='ρ/ρ* (particles)')
        ax.plot(rr / R, (p / p_star) / 10.0, 'o', ms=2, color='k', alpha=0.5, label='p/p*/10 (particles)')

        # Vertical line at min(hsml)
        hs_min = np.min(hsml) if np.ndim(hsml) > 0 else float(hsml)
        ax.plot([hs_min / R, hs_min / R], [1e-3, 1e6], color='0.6', linewidth=1)

        # PPM overlays (colored line)
        if r_ppm is not None:
            ax.plot(r_ppm / R, dens_ppm / rho_star, linewidth=2, color='C1', label='ρ/ρ* (PPM)')
            if r_ppm is not None:
                ax.plot(r_ppm / R, dens_ppm / rho_star,
                        linewidth=2, color='C1', label='ρ/ρ* (PPM)')
                ax.plot(r_ppm / R, pres_ppm / p_star / 10.0,
                        linewidth=2, color='C1', alpha=0.7, label='p/p*/10 (PPM)')

        # Keep legend small to avoid clutter
        ax.legend(fontsize=9, loc='best', frameon=False)

        fig.tight_layout()
        fig.savefig(f"profile_0{snr}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved profile_0{snr}.png")

def show_energies(filename='energy.txt'):
    """
    Equivalent of the IDL 'show_energies' routine.
    Reads 'energy.txt' (columns: t, Eint, Epot, Ekin),
    computes normalized units, and plots the energies.
    """
    # --- Read the table ---
    # Ignore comment lines starting with '#'
    data = np.loadtxt(filename, comments='#')
    t, Eint, Epot, Ekin = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # --- Physical constants and normalization units ---
    M = 1.5815
    R = 1.0
    G = 1.0

    eps_star = G * M / R
    rho_star = 3.0 * M / (4.0 * np.pi * R**3)
    p_star   = eps_star * rho_star
    t_star   = np.sqrt(R**3 / (G * M))

    print(f" => M={M}, rho_*={rho_star:.6f}, p_*={p_star:.6f}, "
          f"t_*={t_star:.6f}, eps_*={eps_star:.6f}")

    # --- Normalize quantities ---
    t_norm  = t / t_star
    Eint_n  = Eint / (eps_star * M)
    Epot_n  = Epot / (eps_star * M)
    Ekin_n  = Ekin / (eps_star * M)
    Etot_n  = (Eint + Epot + Ekin) / (eps_star * M)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 3)
    ax.set_ylim(-3, 2)
    ax.set_xlabel("t / t*", fontsize=14)
    ax.set_ylabel("E / e*", fontsize=14)

    ax.plot(t_norm, Eint_n, label="E_int")
    ax.plot(t_norm, Epot_n, label="E_pot")
    ax.plot(t_norm, Ekin_n, label="E_kin")
    ax.plot(t_norm, Etot_n, label="E_tot", linewidth=2)

    ax.legend()
    fig.tight_layout()
    fig.savefig("energies.png", dpi=150)
    plt.close(fig)
    print("Saved 'energies.png'.")

def show_collapse(nsnaps=27):
    """
    Equivalent of the IDL show_collapse routine:
    Reads 'snap_000'...'snap_026', plots x-y projection of positions,
    and saves PNG frames 'frame_000.png', etc.
    """
    for i in range(nsnaps):
        snr = f"{i:03d}"
        snap = f"snap_{snr}"
        fout = f"frame_{snr}.png"

        try:
            f = g3.GadgetFile(snap)
            pos = f.read_new("POS ", 0)
        except Exception as e:
            print(f"Skipping {snap}: {e}")
            continue

        # Plot equivalent of IDL: x[0,*], x[1,*]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(pos[:, 0], pos[:, 1], 'o', ms=2, color='k')  # psym=3 → small filled circle

        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')

        fig.savefig(fout, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved {fout}")

class EvrardTest(unittest.TestCase):

    def pressure(self, gamma, U, rho):
        return (gamma - 1.0) * U * rho

    def radius(self, x, y, z):
        return np.sqrt(x*x + y*y + z*z)

    def v_rad(self, x, y, z, vx, vy, vz):
        r = self.radius(x, y, z)
        return (vx*x + vy*y + vz*z) / r
    
    def solution(self, value, solution_x_ar, solution_y_ar):
        return np.interp(value, solution_x_ar, solution_y_ar)
    
    def test_properties(self):
        tolerance_rho = 8
        tolerance_p = 1
        tolerance_v = 1
        failures = []
        # -----------------------
        # Constants and scalings
        # -----------------------
        G       = 1.0
        Rstar   = 1.0
        Mstar   = 1.5815

        rhostar = 3.0 * Mstar / (4.0 * np.pi * Rstar**3)
        epsstar = G * Mstar / Rstar
        pstar   = rhostar * epsstar
        vstar   = np.sqrt(G * Mstar / Rstar)

        gamma = 5.0 / 3.0

        # -----------------------
        # Plot loop
        # -----------------------
        for i in range(1, 25):
            snapfile = f"snap_{i:03d}"
            ppmfile  = f"tests/EvrardCollapse/PPM/PPM_{i:02d}"
            
            # Load data
            pos, vel, mass, rho, ein = self.readsnap(snapfile)
            ppm  = np.loadtxt(ppmfile, comments="#")

            # Snapshot quantities
            rho    = rho / rhostar
            p_snap = self.pressure(gamma, ein, rho) / pstar
            r_snap = self.radius(pos[:,0], pos[:,1], pos[:,2])
            v_snap = self.v_rad(
                    pos[:,0], pos[:,1], pos[:,2],
                    vel[:,0], vel[:,1], vel[:,2]
                ) / vstar

            self.plot_evolution(i, r_snap, rho, ppm, p_snap, v_snap)
            
            rho_sol = self.solution(rho,ppm[:,0],ppm[:,1])
            L1 = np.linalg.norm((rho - rho_sol), ord=1)/ np.linalg.norm(rho, ord=1)
            print ('Error norm : ',L1,'    (tolerance : ',tolerance_rho,')')
            if not (L1 < tolerance_rho):
                failures.append(f"Snap: {i} Density error too large: {L1}, tolerance {tolerance_rho}")

            p_sol = self.solution(p_snap,ppm[:,0],ppm[:,2])
            L1 = np.linalg.norm((p_snap - p_sol), ord=1)/ np.linalg.norm(p_snap, ord=1)
            print ('Error norm : ',L1,'    (tolerance : ',tolerance_p,')')
            if not (L1 < tolerance_p):
                failures.append(f"Snap: {i} Pressure error too large: {L1}, tolerance {tolerance_p}")
                
            v_sol = self.solution(v_snap,ppm[:,0],ppm[:,3])
            L1 = np.linalg.norm(v_snap - v_sol, ord=1) / np.linalg.norm(v_snap, ord=1)
            print ('Error norm : ',L1,'    (tolerance : ',tolerance_v,')')
            if not (L1 < tolerance_v):
                failures.append(f"Snap: {i} Velocity error too large: {L1}, tolerance {tolerance_v}")
        #create movie
        self.create_movie()
        # Final check
        self.assertTrue(not failures,"Failures:\n" + "\n".join(failures))
    
    def create_movie(self):
    
        cmd = [
            "ffmpeg", "-y",
            "-framerate", "5",
            "-i", "evolution_%03d.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "evolution.mp4",
        ]

        subprocess.run(cmd, check=True)
      
    def plot_evolution(self, i, r_snap, rho, ppm, p_snap, v_snap):
        # -----------------------
        # Figure layout (3 panels)
        # -----------------------
        fig, axes = plt.subplots(
            3, 1, figsize=(6, 8), sharex=True,
            gridspec_kw=dict(hspace=0.05)
        )

        # ---- Density ----
        ax = axes[0]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.005, 3.0)
        ax.set_ylim(3e-3, 3e3)
        ax.scatter(r_snap, rho, s=8, color="#ff7f0e")
        ax.plot(ppm[:,0], ppm[:,1])
        ax.set_ylabel("Density")
        ax.set_title(f"snapshot {i:03d}", y=0.85)

        # ---- Pressure ----
        ax = axes[1]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(3e-5, 3e3)
        ax.scatter(r_snap, p_snap, s=8, color="#ff7f0e")
        ax.plot(ppm[:,0], ppm[:,2])
        ax.set_ylabel("Pressure")

        # ---- Velocity ----
        ax = axes[2]
        ax.set_xscale("log")
        ax.set_ylim(-3, 7)
        ax.scatter(r_snap, v_snap, s=8, color="#ff7f0e")
        ax.plot(ppm[:,0], ppm[:,3])
        ax.set_ylabel("Velocity")
        ax.set_xlabel("Radius")

        plt.suptitle("Evrard collapse test")
        plt.savefig(f"evolution_{i:03d}.png")
        plt.close(fig)
        print(f"Saved properties evolution_{i:03d}.png")

    def readsnap(self, filename):
        print(f"Reading snapshot file {filename}...")

        f = g3.GadgetFile(filename)
        nparttab = f.header.npart
        masstab  = f.header.mass
        boxsize = f.header.BoxSize

        print(f"  box size: {boxsize:g}")
        print(f"  total: {nparttab.sum():8d} particles")
        for i in range(6):
            print(f"  type {i+1}: {nparttab[i]:8d} particles, mass {masstab[i]:12.4e}")

        # Gas particles only (type 0 in Python)
        npart = nparttab[0]

        # ----------------------
        # Read particle blocks
        # ----------------------
        pos = f.read_new("POS ", 0)        # (N,3)
        vel = f.read_new("VEL ", 0)        # (N,3)
        pid = f.read_new("ID  ", 0)        # (N,)
        rho = f.read_new("RHO ", 0)        # (N,)
        ein = f.read_new("U   ", 0)        # (N,)

        # MASS may or may not exist
        try:
            mass = f.read_new("MASS", 0)
            havemass = True
        except Exception:
            print("no individual masses present; using mass table")
            mass = np.full(npart, masstab[0])
            havemass = False
            
        return pos, vel, mass, rho, ein
        
if __name__ == "__main__":
    # show_collapse()
    # show_profiles()
    # show_energies()
    unittest.main()
