/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Helper function for MFM simulations.
 *
 * This file contains several helper function for MFM simulations.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef _OPENGADGET3_MFM_HPP_
#define _OPENGADGET3_MFM_HPP_

#include "../CodeBase/allvars.hpp"

/**
 * @brief Synchronize Entropy and Internal Energy.
 *
 * Synchronize Entropy and EntropyPred with InternalEnergy and InternalEnergyPred, which are the main variables for the
 * MFM hydro-calculation.
 *
 * @param i Index of the particle.
 */
void mfm_synchronize_entropy(int i)
{
#ifndef GT_DUST
    SphP[i].Entropy = eos->GetEntropyFromU(SphP[i].Density, SphP[i].InternalEnergy, units);
    SphP[i].EntropyPred = eos->GetEntropyFromU(SphP[i].Density, SphP[i].InternalEnergyPred, units);
#else // GT_DUST
    SphP[i].Entropy =
        eos->GetEntropyFromU(SphP[i].Density * (1. - SphP[i].TotalDustFrac), SphP[i].InternalEnergy, units);
    SphP[i].EntropyPred =
        eos->GetEntropyFromU(SphP[i].Density * (1. - SphP[i].TotalDustFracPred), SphP[i].InternalEnergyPred, units);
#endif // GT_DUST
}

/**
 * @brief Scale dQdt for mass changes.
 *
 * Scale dQdt according to change in mass using the new mass, e.g. after star formation.
 *
 * @param i Index of the particle.
 * @param od_mass Initial mass.
 * @param new_mass Final mass.
 */
void mfm_update_mass(int i, MyFloat old_mass, MyFloat new_mass)
{
    if ((P[i].Type == type_gas) && (old_mass > 0) && (new_mass > 0))
    {
        MyAtLeastDouble factor = new_mass / old_mass;
        // components with information about HydroAccel
        for (int j = 0; j < mfm_numdims; j++)
        {
            SphP[i].dQdt.GetMomentum()[j] *= factor;
        }
        // component with information about InternalEnergy / Entropy
        SphP[i].dQdt.GetEnergy() *= factor;
        SphP[i].Density *= factor; // according to MFM density definition based on NumDens (which stays cosntant.
    }
}

/**
 * @brief Scale dQdt for mass changes.
 *
 * Scale dQdt according to change in mass using the mass change, e.g. after star formation.
 *
 * @param i Index of the particle.
 * @param mass_change Additional mass to be added.
 */
void mfm_add_mass(int i, MyFloat mass_change)
{
    if ((P[i].Type == type_gas) && (P[i].Mass > 0) && (P[i].Mass + mass_change > 0))
    {
        MyAtLeastDouble factor = (P[i].Mass + mass_change) / P[i].Mass;
        // components with information about HydroAccel
        for (int j = 0; j < mfm_numdims; j++)
        {
            SphP[i].dQdt.GetMomentum()[j] *= factor;
        }
        // component with information about InternalEnergy / Entropy
        SphP[i].dQdt.GetEnergy() *= factor;
        SphP[i].Density *= factor; // according to MFM density definition based on NumDens (which stays cosntant.
    }
}

#ifdef DIVBCLEANING_DEDNER
/**
 * @brief Get inverse damping time for Phi field.
 *
 * Return the inverse of the damping time for the Phi field used for Dedner cleaning.
 *
 * @param i Index of the particle.
 * @return Damping time for Dedner cleaning.
 */
MyFloat get_t_damp_phi_inv(int i)
{
    // first estimate based on the fastest signal velocity.
    MyAtLeastDouble damping_speed = SphP[i].MaxSignalVel;
    // second estimate based on the cleaning field magnetic energy
    MyAtLeastDouble sqrt_mu0_inv = sqrt(mu0_inv);
    MyAtLeastDouble fast_magnetosonic_speed =
        eos->FastMagnetosonicSpeed(SphP[i].Density, SphP[i].InternalEnergy, sqrt_mu0_inv * SphP[i].B[0],
                                   sqrt_mu0_inv * SphP[i].B[1], sqrt_mu0_inv * SphP[i].B[2]);
    MyAtLeastDouble cleaning_energy_speed = 2.0 * sqrt_mu0_inv * SphP[i].Phi / SphP[i].MaxSignalVel;
    damping_speed =
        std::max(damping_speed, std::sqrt(fast_magnetosonic_speed * fast_magnetosonic_speed + cleaning_energy_speed));
    // also compare to the global value
    damping_speed = std::max(damping_speed, (MyAtLeastDouble)0.1 * FastestWaveSpeed);
    // todo: add more advanced value, copare Hopkins & Raives (D6)
    if (!std::isfinite(damping_speed))
    {
        damping_speed = 0.0;
    }
    return All.DivBcleaningParabolicSigma * damping_speed / get_mfm_cell_size(i);
}

/**
 * @brief Get fastest wavespeed across simulation domain.
 *
 * Return the fastest wavespeed (MaxSignalVel or FastMagnetosonicSpeed) among all particles within the simulation.
 */
MyFloat get_fastest_wave_speed()
{
    // first find the fastest wavespeed on local particles.
    MyFloat fastest_wave_speed = 0.0;

    MyAtLeastDouble sqrt_mu0_inv = sqrt(mu0_inv);
    for (int i = 0; i < N_gas; ++i)
    {
        MyFloat this_fastest_wave_speed =
            std::max(SphP[i].MaxSignalVel,
                     eos->FastMagnetosonicSpeed(SphP[i].Density, SphP[i].InternalEnergy, sqrt_mu0_inv * SphP[i].B[0],
                                                sqrt_mu0_inv * SphP[i].B[1], sqrt_mu0_inv * SphP[i].B[2]));
        fastest_wave_speed = std::max(fastest_wave_speed, this_fastest_wave_speed);
    }

    // then check across other MPI ranks
    MyFloat global_fastest_wave_speed;
    MPI_Allreduce(&fastest_wave_speed, &global_fastest_wave_speed, 1, MPI_MyFloat, MPI_MAX, MYMPI_COMM_WORLD);
    return global_fastest_wave_speed;
}
#endif // DIVBCLEANING_DEDNER


#ifdef LMB_MFM_ENERGY_CLOSURE_LIMITER
/**
 * @brief Face-closure limiter on the internal-energy change of a cell.
 *
 * Energy analogue of get_dtb_facearea_limiter, gating on the same closure residual. A cell whose
 * effective faces fail to close (Sum_j Aij != 0, i.e. FaceAreaSum != 0) carries a spurious momentum
 * flux: for a uniform state it is dp = -P Sum_j Aij. Subtracting the kinetic-energy change from the
 * lab-frame energy flux cancels the v.dp part exactly and is left with -|dp|^2 dt / (2 m), a
 * strictly negative internal-energy source quadratic in the residual. That source does not scale
 * with u, so its fractional bite grows as a cell cools: the cell reaches the temperature floor,
 * loses pressure support and is then compressed to the resolution limit, where its hydrodynamic
 * acceleration and with it its timestep criterion collapse. This bounds |du| over the interval in
 * cells with a large residual. Returns a factor in (0,1]; 1 = no limiting, and always 1 in cells
 * whose faces close.
 *
 * As for DtB the tolerance is proportional to the cell's Courant number dt*c_sig/L, so it bounds
 * the change per unit time rather than per step; a per-step bound would grant a cell that falls N
 * timebins 2^N more drain per unit time, which is the same feedback loop it is meant to break.
 *
 * Only the energy channel is limited. MFM's momentum flux is antisymmetric across each face and so
 * conserves momentum to round-off; clamping dp would break that exactly. Bounding du instead gives
 * up exact conservation of (internal + kinetic) energy in the affected cells, which is the price of
 * not letting the residual drain them.
 *
 * @param i Index of the particle.
 * @param du Magnitude of the specific internal-energy change over the interval (physical).
 * @param u Specific internal energy the change is measured against (physical).
 * @param dt_entr_phys Length of the interval in physical time.
 */
MyFloat get_energy_facearea_factor(int i, MyAtLeastDouble du, MyAtLeastDouble u, double dt_entr_phys)
{
    if (dt_entr_phys <= 0.0 || !(du > 0.0) || !(u > 0.0))
        return 1.0;

    // closure residual (L1 norm of the effective-face-area vector sum) vs the expected cell surface.
    const MyAtLeastDouble area_sum =
        std::abs(SphP[i].FaceAreaSum[0]) + std::abs(SphP[i].FaceAreaSum[1]) + std::abs(SphP[i].FaceAreaSum[2]);
#if NUMDIMS == 1
    const MyAtLeastDouble area_norm = 2.0;
#elif NUMDIMS == 2
    const MyAtLeastDouble area_norm = 2.0 * M_PI * P[i].Hsml;
#else // NUMDIMS == 3
    const MyAtLeastDouble area_norm = 4.0 * M_PI * P[i].Hsml * P[i].Hsml;
#endif // NUMDIMS

    if (area_norm > 0.0 && area_sum / area_norm > All.MfmEnergyClosureMin)
    {
        // Courant number of this interval, c_sig and L as in get_timestep(). Clipped at the value
        // the Courant criterion itself allows, which a MaxSignalVel rise mid-step can exceed.
        const MyAtLeastDouble csig = units->afac3 * SphP[i].MaxSignalVel;
        const MyAtLeastDouble lcell = units->length_to_physical * get_mfm_cell_size(i);
        MyAtLeastDouble courant = 2.0 * All.CourantFac;
        if (csig > 0.0 && lcell > 0.0)
        {
            courant = std::min(courant, dt_entr_phys * csig / lcell);
        }

        MyAtLeastDouble tol = courant * std::max((MyAtLeastDouble)All.MfmEnergyTolMin,
                                                 area_norm / (All.MfmEnergyClosureWeight * area_sum));
        tol *= u; // give the tolerance the dimensions of du
        if (du > tol)
            return (MyFloat)(tol / du);
    }
    return 1.0;
}

/**
 * @brief Face-closure limiter on the internal-energy flux dQdt.GetEnergy().
 *
 * Applies get_energy_facearea_factor to the flux itself, which is possible only when
 * dQdt.GetEnergy() has already been reduced to an internal-energy rate by
 * convert_total_to_internal_energy_flux. Under MFM_CONSERVATIVE_TOTAL_ENERGY it still carries the
 * kinetic part, and the limit is applied to the internal-energy increment in the drift and kick
 * instead (limit_energy_change_facearea).
 *
 * @param i Index of the particle.
 * @param dt_entr_phys Length of the interval in physical time; du = dQdt.GetEnergy()/Mass * dt.
 */
MyFloat get_energy_facearea_limiter(int i, double dt_entr_phys)
{
    const MyAtLeastDouble du = std::abs(SphP[i].dQdt.GetEnergy()) / P[i].Mass * dt_entr_phys;
    return get_energy_facearea_factor(i, du, SphP[i].InternalEnergyPred, dt_entr_phys);
}
#endif // LMB_MFM_ENERGY_CLOSURE_LIMITER

/**
 * @brief Convert MFM Godunov fluxes to hydrodynamical changes.
 *
 * Calculate the hydrodynamical acceleration, entropy change etc from the MFM Godunoc fluxes.
 * This function should be called for every drift/kick, as the conversion factors are time-dependent.
 *
 * @param i Index of the particle.
 * @param tstart_step Begin of timestep.
 * @param tend_step End of timestep.
 */
void mfm_fluxes_to_code(int i, integertime tstart_step, integertime tend_step)
{
    double dt_hydrokick, dt_entr_phys, dt_entr, dt_drift;
    double dt_hydrokick_correction, dt_entr_correction, dt_drift_correction, dt_drift_div_dt_entr;

    if (All.ComovingIntegrationOn)
    {
        dt_hydrokick = get_hydrokick_factor(tstart_step, tend_step);
        dt_entr = (tend_step - tstart_step) * All.Timebase_interval;
        dt_entr_phys = units->timestep_to_physical * (tend_step - tstart_step) * All.Timebase_interval;
        dt_drift = get_drift_factor(tstart_step, tend_step);

        if (dt_entr_phys == 0)
        {
            dt_hydrokick_correction = dt_entr_correction = dt_drift_correction = dt_drift_div_dt_entr
                = 0.0; // effectively set all changes to zero. This is important in the zeroth Sync-Point.
        }
        else
        {
            dt_hydrokick_correction = dt_entr_phys / dt_hydrokick;
            dt_entr_correction = dt_entr_phys / dt_entr;
            dt_drift_correction = dt_entr_phys / dt_drift;
            dt_drift_div_dt_entr = dt_drift / dt_entr;
        }
    }
    else
    {
            dt_hydrokick = dt_entr = dt_entr_phys = dt_drift = (tend_step - tstart_step) * All.Timebase_interval;

        dt_hydrokick_correction = dt_entr_correction = dt_drift_correction = dt_drift_div_dt_entr
            = 1.0;
    }
    if (dt_entr_phys == 0)
    {
        dt_hydrokick_correction = dt_entr_correction = dt_drift_correction = dt_drift_div_dt_entr
            = 1.0;
    }

    // Update HydroAccel used in code
    for (int k = 0; k < mfm_numdims; k++)
    {
        SphP[i].HydroAccel[k] =
            units->velocity_to_comoving * SphP[i].dQdt.GetMomentum()[k] / P[i].Mass * dt_hydrokick_correction;
    }

#ifdef GT_DUST
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        SphP[i].DtDustFrac[idust] = SphP[i].dQdt_dust[idust].GetMass() / P[i].Mass;
        for (int k = 0; k < mfm_numdims; k++)
        {
            SphP[i].HydroAccel[k] += SphP[i].dQdt_dust[idust].GetMomentum()[k] / P[i].Mass;
            if (SphP[i].Wprim_dust[idust].GetDensity() > 1e-20)
            {
                SphP[i].DtDeltaVel[idust][k] =
                    (SphP[i].dQdt_dust[idust].GetMomentum()[k] -
                     SphP[i].Wprim_dust[idust].GetVelocity()[k] * SphP[i].dQdt_dust[idust].GetMass()) /
                    (SphP[i].Wprim_dust[idust].GetDensity() / SphP[i].NumDens);
            }
            else
            {
                SphP[i].DtDeltaVel[idust][k] = 0.0;
            }

            SphP[i].DtDeltaVel[idust][k] -=
                (SphP[i].dQdt.GetMomentum()[k] - SphP[i].Wprim.GetVelocity()[k] * SphP[i].dQdt.GetMass()) /
                (SphP[i].Wprim.GetDensity() / SphP[i].NumDens);
        }
    }
#endif // GT_DUST

    MyAtLeastDouble div_vel = limited_divvel(i);
    // DivVel is stored in code units (= a^2 * physical divergence). Terms that carry a *_correction
    // time base (dt_mag_correction / dt_entr_correction, which fold in the physical dt) need the
    // physical divergence; equals div_vel for non-comoving runs.
    const MyAtLeastDouble divv_phys = div_vel * units->velocity_to_physical / units->length_to_physical;

    // Update DtEntropy / DtInternalEnergy used in code. Normally dQdt.GetEnergy() is the internal-energy
    // flux (physical). Under MFM_CONSERVATIVE_TOTAL_ENERGY it still carries the kinetic part (only the
    // magnetic change was removed in convert_...); the drift/kick then subtract the actual kinetic change.
    SphP[i].DtInternalEnergy = SphP[i].dQdt.GetEnergy() / P[i].Mass * dt_entr_correction;

#ifdef GT_DUST
    SphP[i].DtInternalEnergy /= (1. - SphP[i].TotalDustFracPred);
#endif // GT_DUST

    if (All.ComovingIntegrationOn)
    {
        SphP[i].DtInternalEnergy -= 3 * GAMMA_MINUS1 * SphP[i].InternalEnergyPred * units->hubble_a *
                                    dt_entr_correction; // adiabatic hubble flux correction
    }

#ifndef GT_DUST
    SphP[i].DtEntropy =
        eos->GetEntropyFromU(SphP[i].Density, SphP[i].dQdt.GetEnergy() / P[i].Mass, units) * dt_entr_correction;
    SphP[i].DtEntropy += GAMMA_MINUS1 * eos->GetEntropyFromU(SphP[i].Density, SphP[i].InternalEnergyPred, units) *
                         div_vel *
                         dt_drift_div_dt_entr; // reduce by adiabatic energy change (only rough approximation!)
#else // GT_DUST
    SphP[i].DtEntropy =
        eos->GetEntropyFromU(SphP[i].Density / (1. - SphP[i].TotalDustFrac),
                             SphP[i].dQdt.GetEnergy() / P[i].Mass / (1. - SphP[i].TotalDustFrac), units) *
        dt_entr_correction;
    SphP[i].DtEntropy +=
        GAMMA_MINUS1 *
        eos->GetEntropyFromU(SphP[i].Density / (1. - SphP[i].TotalDustFrac), SphP[i].InternalEnergyPred, units) *
        div_vel * dt_drift_div_dt_entr; // reduce by adiabatic energy change (only rough approximation!)
    SphP[i].DtEntropy += GAMMA_MINUS1 / SphP[i].Wprim.GetDensity() *
                         eos->GetEntropyFromU(SphP[i].Wprim.GetDensity(), SphP[i].InternalEnergyPred, units) *
                         SphP[i].dQdt.GetMass() * SphP[i].NumDens;
#endif // GT_DUST

    // Energy-Entropy switch from gizmo. Disabled under MFM_CONSERVATIVE_TOTAL_ENERGY: it would
    // overwrite the (internal+kinetic) energy rate with an adiabatic internal-only rate and break the
    // conservative u-recompute done in the drift/kick.
    MyAtLeastDouble e_thermal, e_kinetic, e_potential;
    MyAtLeastDouble e_magnetic{};
    e_potential = 0;
    for (int k = 0; k < 3; k++)
    {
        MyAtLeastDouble grav_acc = units->accel_to_physical * P[i].GravAccel[k];
#ifdef PMGRID
        grav_acc += units->accel_to_physical * P[i].GravPM[k];
#endif // PMGRID
        e_potential += grav_acc * grav_acc;
    }
    e_potential =
        P[i].Mass * sqrt(e_potential) *
        ((MyFloat)0.5 /*is KERNEL_CORE_SIZE */ * units->length_to_physical * P[i].Hsml); // = M*|a_grav|*h (physical)
    e_kinetic = 0.5 * P[i].Mass * units->velocity_to_physical * units->velocity_to_physical * SphP[i].MaxVelSquareNgb;
    MyAtLeastDouble dEnt_Gravity = 0; // only for MFV
    MyAtLeastDouble InternalEnergy = SphP[i].InternalEnergyPred;
    MyAtLeastDouble dEnt = InternalEnergy + SphP[i].DtInternalEnergy * dt_entr + dEnt_Gravity;
    e_thermal = max(0.5 * InternalEnergy, dEnt) * P[i].Mass;

#ifdef GT_DUST
    e_potential /= (1. - SphP[i].TotalDustFracPred);
    e_kinetic /= (1. - SphP[i].TotalDustFracPred);
    e_thermal /= (1. - SphP[i].TotalDustFracPred);
#endif // GT_DUST

    // this is the actual switch. in many cases, only one is sufficient (KE or potential).
    if (All.EpotSwitchFraction * e_potential + All.EkinSwitchFraction * e_kinetic >
        e_thermal + e_magnetic) // recommended values on the order of 1e-2, use 0 for both to ignore the switch.
    {
        SphP[i].DtInternalEnergy = -GAMMA_MINUS1 * SphP[i].InternalEnergyPred * divv_phys * dt_entr_correction;
#ifdef GT_DUST
        SphP[i].DtInternalEnergy /= (1. - SphP[i].TotalDustFracPred);
#endif // GT_DUST
        if (All.ComovingIntegrationOn)
        {
            SphP[i].DtInternalEnergy -=
                3 * GAMMA_MINUS1 * SphP[i].InternalEnergyPred * units->hubble_a * dt_entr_correction;
        }

        SphP[i].DtEntropy = (MyFloat)0.0; // assume purely adiabatic changes
    }

    // add Extra variables
    SphP[i].DtEntropy += SphP[i].DtEntropyExtra;
#ifndef GT_DUST
    SphP[i].DtInternalEnergy += eos->GetUFromEntropy(SphP[i].Density, SphP[i].DtEntropyExtra, units);
#else // GT_DUST
    SphP[i].DtInternalEnergy +=
        eos->GetUFromEntropy(SphP[i].Density / (1. - SphP[i].TotalDustFracPred), SphP[i].DtEntropyExtra, units);
#endif // GT_DUST
    for (int k = 0; k < 3; k++)
    {
        SphP[i].HydroAccel[k] += SphP[i].HydroAccelExtra[k];
    }

#ifdef GT_DUST_DIFFUSION
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        for (int k = 0; k < NUMDIMS; k++)
        {
            SphP[i].DtDeltaVel[idust][k] += SphP[i].DtDeltaVelExtra[idust][k];
        }
    }
#endif // GT_DUST_DIFFUSION
}

/**
 * @brief Return the size of MFM cells.
 *
 * The MFM particle size is not the Hsml, but should be reduced by N_ngb^(1/NUMDIMS). We can express this using the
 * inverse number density, which corresponds to the effective volume.
 *
 * @param i Index of the particle.
 */
MyFloat get_mfm_cell_size(int i)
{
#if NUMDIMS == 1
    return 1.0 / (NORM_COEFF * SphP[i].NumDens);
#elif NUMDIMS == 2
    return 1.0 / std::sqrt(NORM_COEFF * SphP[i].NumDens);
#else // NUMDIMS == 3
    return 1.0 / std::cbrt(NORM_COEFF * SphP[i].NumDens);
#endif // NUMDIMS
}

#endif // _OPENGADGET3_MFM_HPP_
