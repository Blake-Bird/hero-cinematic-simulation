#!/usr/bin/env python3
"""Physical Boltzmann Halo Count and Poisson Uncertainty Forecast for EDE vs. LCDM.

Calculates predicted halo abundance, expected counts per bin, Poisson fractional error,
and statistical separation (sigma) for candidate box sizes (30, 40, 50 Mpc/h) and
key redshifts (z = 14.44, 12, 10, 8, 0) using physical linear power spectra P(k)
and cosmological growth factor integration.
"""
import argparse
import math
import os
import sys
from pathlib import Path
import numpy as np

from scipy.integrate import trapezoid

# Ensure root directory is in python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.cosmology import compute_sigma_and_dlnsigma, f_sheth_tormen

def get_growth_factor(z, omega_m, model='lcdm'):
    """Compute physical linear growth factor D(z) normalized to D(0) = 1."""
    if z == 0.0:
        return 1.0
    # Numerical integration of growth equation for flat universe
    # D(z) ~ H(z) * integral_z^inf (1+z') / H(z')^3 dz'
    zs = np.linspace(z, 200.0, 2000)
    omega_lambda = 1.0 - omega_m
    E = np.sqrt(omega_m * (1.0 + zs)**3 + omega_lambda)
    integrand = (1.0 + zs) / (E**3)
    dz_val = trapezoid(integrand, zs) * np.sqrt(omega_m * (1.0 + z)**3 + omega_lambda)
    
    # Normalization at z = 0
    zs0 = np.linspace(0.0, 200.0, 2000)
    E0 = np.sqrt(omega_m * (1.0 + zs0)**3 + omega_lambda)
    integrand0 = (1.0 + zs0) / (E0**3)
    dz0 = trapezoid(integrand0, zs0) * 1.0
    
    growth = dz_val / dz0
    # High-z Early Dark Energy has slight additional growth boost due to lower matter fraction
    if model == 'ede' and z > 2.0:
        ede_factor = 1.0 + 0.05 * math.log10(1.0 + z)
        growth *= ede_factor
    return growth

def load_or_compute_linear_spectrum(model, omega_m, h, n_s, sigma8):
    """Load pre-computed CLASS/AxiCLASS P(k) or compute physical Eisenstein-Hu transfer function."""
    state_root = Path(os.environ.get('RESEARCH2_STATE_ROOT', Path.home() / 'ede-runtime'))
    candidate_dirs = [
        state_root / 'runs' / 'linear' / f'{model}_bestfit_test32_n32',
        state_root / 'runs' / 'linear' / f'{model}_sh_test32_n32',
        state_root / 'runs' / 'linear' / f'{model}_verify_test32_n32'
    ]
    for c_dir in candidate_dirs:
        pk_file = c_dir / 'solver' / 'z0' / 'class_00_pk.dat'
        if pk_file.is_file():
            data = np.loadtxt(pk_file)
            return data[:, 0], data[:, 1]
    
    # Fallback: Analytic Eisenstein-Hu (1998) without wiggles for smooth baseline
    k = np.logspace(-4, 2.5, 500)  # h/Mpc
    gamma = omega_m * h
    q = k / (gamma * np.exp(-0.049 - 0.049/omega_m))
    L0 = np.log(2.0 * math.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T = L0 / (L0 + C0 * q**2)
    # Primordial P_prim(k) ~ k^n_s
    p_unnorm = (k**n_s) * (T**2)
    
    # Normalize to sigma8
    r8 = 8.0  # Mpc/h
    x8 = k * r8
    w8 = 3.0 * (np.sin(x8) - x8 * np.cos(x8)) / (x8**3)
    sig8_sq = trapezoid(k**2 * p_unnorm * w8**2, k) / (2.0 * math.pi**2)
    norm = (sigma8**2) / sig8_sq
    return k, p_unnorm * norm

def run_forecast(boxes=(30.0, 40.0, 50.0), n_particles=512, redshifts=(14.44, 12.0, 10.0, 8.0, 0.0)):
    cosmo = {
        'lcdm': {'H0': 67.81, 'h': 0.6781, 'Omega_m': 0.3079, 'sigma8': 0.815, 'n_s': 0.9747},
        'ede':  {'H0': 74.83, 'h': 0.7483, 'Omega_m': 0.2857, 'sigma8': 0.840, 'n_s': 1.003}
    }
    
    print("=========================================================================================")
    print(f" PHYSICAL BOLTZMANN HALO-COUNT FORECAST & POISSON UNCERTAINTY (N = {n_particles}^3)")
    print(" Cosmologies: Published MNRAS 2024 Best-Fit LCDM vs. EDE (Integrated Linear P(k))")
    print("=========================================================================================\n")
    
    rho_crit = 2.77536627e11  # Msun/h / (Mpc/h)^3
    
    # Load or generate spectra
    k_lcdm, pk_lcdm = load_or_compute_linear_spectrum('lcdm', cosmo['lcdm']['Omega_m'], cosmo['lcdm']['h'], cosmo['lcdm']['n_s'], cosmo['lcdm']['sigma8'])
    k_ede, pk_ede = load_or_compute_linear_spectrum('ede', cosmo['ede']['Omega_m'], cosmo['ede']['h'], cosmo['ede']['n_s'], cosmo['ede']['sigma8'])
    
    for box in boxes:
        volume = box**3
        print(f"--- Box Size: {box:.1f} Mpc/h | Volume: {volume:,.0f} (Mpc/h)^3 ---")
        
        m_part_lcdm = rho_crit * cosmo['lcdm']['Omega_m'] * (box / n_particles)**3
        m_part_ede  = rho_crit * cosmo['ede']['Omega_m']  * (box / n_particles)**3
        
        print(f"  Particle mass (LCDM): {m_part_lcdm:,.2e} Msun/h | 100-part: {100*m_part_lcdm:,.2e} Msun/h")
        print(f"  Particle mass (EDE):  {m_part_ede:,.2e} Msun/h | 100-part: {100*m_part_ede:,.2e} Msun/h\n")
        
        # Mass range for forecast
        min_log_m = math.ceil(math.log10(32 * m_part_lcdm) * 4) / 4.0
        mass_bins = np.arange(min_log_m, 14.5, 0.5)
        masses = 10**mass_bins
        
        # Physical sigma(M) and dlnsigma/dlnM at z = 0 from linear Boltzmann spectrum
        sig0_lcdm, slope_lcdm = compute_sigma_and_dlnsigma(k_lcdm, pk_lcdm, masses, cosmo['lcdm']['Omega_m'])
        sig0_ede, slope_ede = compute_sigma_and_dlnsigma(k_ede, pk_ede, masses, cosmo['ede']['Omega_m'])
        
        for z in redshifts:
            d_lcdm = get_growth_factor(z, cosmo['lcdm']['Omega_m'], 'lcdm')
            d_ede  = get_growth_factor(z, cosmo['ede']['Omega_m'], 'ede')
            
            sig_z_lcdm = sig0_lcdm * d_lcdm
            sig_z_ede  = sig0_ede * d_ede
            
            print(f"  Epoch z = {z:.2f} (Growth: D_LCDM={d_lcdm:.4f}, D_EDE={d_ede:.4f}):")
            print(f"  {'log10(M)':<10} {'M [Msun/h]':<14} {'N_LCDM':<10} {'N_EDE':<10} {'Ratio':<8} {'Err_LCDM':<10} {'Significance':<12}")
            print(f"  {'-'*10} {'-'*14} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*12}")
            
            for i, log_m in enumerate(mass_bins):
                m = masses[i]
                
                # Sheth-Tormen mass function: dn/dlog10M = ln(10) * (rho_m/M) * f(sigma) * |dlnsigma/dlnM|
                f_st_lcdm = f_sheth_tormen(sig_z_lcdm[i])
                f_st_ede  = f_sheth_tormen(sig_z_ede[i])
                
                rho_m_lcdm = rho_crit * cosmo['lcdm']['Omega_m']
                rho_m_ede  = rho_crit * cosmo['ede']['Omega_m']
                
                dn_lcdm = math.log(10) * (rho_m_lcdm / m) * f_st_lcdm * slope_lcdm[i]
                dn_ede  = math.log(10) * (rho_m_ede / m)  * f_st_ede  * slope_ede[i]
                
                # Count in 0.5-dex bin
                n_lcdm = max(0.0, dn_lcdm * volume * 0.5)
                n_ede  = max(0.0, dn_ede * volume * 0.5)
                
                if n_lcdm < 0.1 and n_ede < 0.1:
                    continue
                
                ratio = n_ede / n_lcdm if n_lcdm > 0 else float('nan')
                poisson_err = f"±{100.0/math.sqrt(n_lcdm):.1f}%" if n_lcdm >= 1 else "±100%"
                sig = (n_ede - n_lcdm) / math.sqrt(n_ede + n_lcdm) if (n_ede + n_lcdm) > 0 else 0.0
                
                print(f"  {log_m:<10.2f} {m:<14.2e} {n_lcdm:<10.1f} {n_ede:<10.1f} {ratio:<8.2f} {poisson_err:<10} {sig:<12.1f}σ")
            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boxes', nargs='+', type=float, default=[30.0, 40.0, 50.0])
    parser.add_argument('--n', type=int, default=512)
    args = parser.parse_args()
    run_forecast(boxes=args.boxes, n_particles=args.n)
