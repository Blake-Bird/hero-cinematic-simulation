/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file for extra viscosity and artificial conductivity routines.
 *
 * @author OpenGadget3-development team, in particular Ludwig Böss, Klaus Dolag, Frederick Groth, Tirso Marín Gilabert
 * and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "../CodeBase/allvars.hpp"
#if !(defined ACC_INCLUDE_HYDRA) && defined(ACC_HYDRA) // we reinclude this same file if ACC is active
#include "../OpenACC/gpuallvars_branch.hpp"
#endif // !(defined ACC_INCLUDE_HYDRA) && defined(ACC_HYDRA)

#ifdef MAGNETIC
#include "mhd.hpp"
#endif // MAGNETIC

#if defined(AA_ATMOSPHERE_FORCE) && !defined(NOACCEL)
#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Add the atmospheric forces to the hydrodynamic acceleration of a particle.
 *
 * This function subtracts the atmosphere force from the y-component of the hydrodynamic acceleration
 * of a particle represented by index 'i' in the SPH particle array 'SphP'. If the hydrodynamic
 * scheme is of type HYDRO_MFM, the force is also subtracted from the y-component of the extra hydrodynamic
 * acceleration.
 *
 * @param i Index of the particle in the SPH particle array.
 */
static inline void add_aa_atmosphere_force(int i)
{
    SphP[i].HydroAccel[1] -= AA_ATMOSPHERE_FORCE;
#if GADGET_HYDRO == HYDRO_MFM
    SphP[i].HydroAccelExtra[1] -= AA_ATMOSPHERE_FORCE;
#endif // GADGET_HYDRO == HYDRO_MFM
}
#endif // ACC_INCLUDE_HYDRA
#endif // if defined(AA_ATMOSPHERE_FORCE) && !defined(NOACCEL)

#ifdef NAVIERSTOKES
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Evaluates the contribution of Navier-Stokes viscosity to the acceleration of particles with OpenACC support.
 *
 * This function computes the contribution of Navier-Stokes viscosity to the acceleration of a particle
 * represented by the local and neighboring particles, using the provided hydrodynamic data and kernel information.
 * If enabled, it also considers the bulk viscosity contribution to the acceleration.
 *
 * @param local Pointer to the local hydrodynamic data structure.
 * @param kernel Pointer to the kernel structure containing smoothing lengths and derivatives.
 * @param j Index of the neighboring particle.
 * @param a3inv Inverse of the cube of the scale factor.
 * @param P Pointer to the gravity walk data structure.
 * @param SphP Pointer to the hydrodynamic data structure of the neighboring particles.
 * @param out Pointer to the output hydrodynamic data structure containing the computed accelerations.
 */
void navierstokes_viscosity_evaluate_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                             MyAtLeastDouble a3inv, struct gravity_walkdata *P,
                                             struct hydrodata_in *SphP, struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Evaluates the contribution of Navier-Stokes viscosity to the acceleration of particles.
 *
 * This function computes the contribution of Navier-Stokes viscosity to the acceleration of a particle
 * represented by the local and neighboring particles, using the provided hydrodynamic data and kernel information.
 * If enabled, it also considers the bulk viscosity contribution to the acceleration.
 *
 * @param local Pointer to the local hydrodynamic data structure.
 * @param kernel Pointer to the kernel structure containing smoothing lengths and derivatives.
 * @param j Index of the neighboring particle.
 * @param a3inv Inverse of the cube of the scale factor.
 * @param out Pointer to the output hydrodynamic data structure containing the computed accelerations.
 */
static inline void navierstokes_viscosity_evaluate(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                                   MyAtLeastDouble a3inv, struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
    double faci = local->Mass / (local->Density * local->Density) * kernel->dwk_i / kernel->r;

#ifdef NAVIERSTOKES_ANISO
    MyAtLeastDouble B_square, norm_B;
    MyAtLeastDouble aniso_i[3][3];
    MyAtLeastDouble uni_B[3];

    B_square =
        local->BPred[0] * local->BPred[0] + local->BPred[1] * local->BPred[1] + local->BPred[2] * local->BPred[2];

    norm_B = sqrt(B_square);

    uni_B[0] = local->BPred[0] / norm_B;
    uni_B[1] = local->BPred[1] / norm_B;
    uni_B[2] = local->BPred[2] / norm_B;

    for (int k = 0; k < 3; k++)
    {
        aniso_i[k][0] = uni_B[k] * uni_B[0];
        aniso_i[k][1] = uni_B[k] * uni_B[1];
        aniso_i[k][2] = uni_B[k] * uni_B[2];
    }

    aniso_i[0][0] -= 1. / 3.;
    aniso_i[1][1] -= 1. / 3.;
    aniso_i[2][2] -= 1. / 3.;

    faci *= local->P_aniso;

#else
    faci *= local->shear_visc_coeff;

#endif // NAVIERSTOKES_ANISO

    double facj = P[j].Mass / (SphP[j].Density * SphP[j].Density) * kernel->dwk_j / kernel->r;

#ifdef NAVIERSTOKES_ANISO
    MyAtLeastDouble aniso_j[3][3];

    B_square =
        SphP[j].BPred[0] * SphP[j].BPred[0] + SphP[j].BPred[1] * SphP[j].BPred[1] + SphP[j].BPred[2] * SphP[j].BPred[2];

    norm_B = sqrt(B_square);

    uni_B[0] = SphP[j].BPred[0] / norm_B;
    uni_B[1] = SphP[j].BPred[1] / norm_B;
    uni_B[2] = SphP[j].BPred[2] / norm_B;

    for (int k = 0; k < 3; k++)
    {
        aniso_j[k][0] = uni_B[k] * uni_B[0];
        aniso_j[k][1] = uni_B[k] * uni_B[1];
        aniso_j[k][2] = uni_B[k] * uni_B[2];
    }

    aniso_j[0][0] -= 1. / 3.;
    aniso_j[1][1] -= 1. / 3.;
    aniso_j[2][2] -= 1. / 3.;

    facj *= SphP[j].P_aniso;

#else
    facj *= SphP[j].shear_visc_coeff;

#endif // NAVIERSTOKES_ANISO


#ifdef WINDS
    if (P[j].Type == type_gas)
    {
        if (SphP[j].DelayTime > 0) /* No visc for wind particles */
        {
            faci = facj = 0;
        }
    }
#endif // WINDS

    /* Acceleration due to the shear viscosity */
#ifndef NAVIERSTOKES_ANISO
    out->Acc[0] += faci * (local->StressDiag[0] * kernel->dx + local->StressOffDiag[0] * kernel->dy +
                           local->StressOffDiag[1] * kernel->dz) +
                   facj * (SphP[j].StressDiag[0] * kernel->dx + SphP[j].StressOffDiag[0] * kernel->dy +
                           SphP[j].StressOffDiag[1] * kernel->dz);

    out->Acc[1] += faci * (local->StressOffDiag[0] * kernel->dx + local->StressDiag[1] * kernel->dy +
                           local->StressOffDiag[2] * kernel->dz) +
                   facj * (SphP[j].StressOffDiag[0] * kernel->dx + SphP[j].StressDiag[1] * kernel->dy +
                           SphP[j].StressOffDiag[2] * kernel->dz);

    out->Acc[2] += faci * (local->StressOffDiag[1] * kernel->dx + local->StressOffDiag[2] * kernel->dy +
                           local->StressDiag[2] * kernel->dz) +
                   facj * (SphP[j].StressOffDiag[1] * kernel->dx + SphP[j].StressOffDiag[2] * kernel->dy +
                           SphP[j].StressDiag[2] * kernel->dz);

#else
    out->Acc[0] += faci * (aniso_i[0][0] * kernel->dx + aniso_i[0][1] * kernel->dy + aniso_i[0][2] * kernel->dz) +
                   facj * (aniso_j[0][0] * kernel->dx + aniso_j[0][1] * kernel->dy + aniso_j[0][2] * kernel->dz);

    out->Acc[1] += faci * (aniso_i[1][0] * kernel->dx + aniso_i[1][1] * kernel->dy + aniso_i[1][2] * kernel->dz) +
                   facj * (aniso_j[1][0] * kernel->dx + aniso_j[1][1] * kernel->dy + aniso_j[1][2] * kernel->dz);

    out->Acc[2] += faci * (aniso_i[2][0] * kernel->dx + aniso_i[2][1] * kernel->dy + aniso_i[2][2] * kernel->dz) +
                   facj * (aniso_j[2][0] * kernel->dx + aniso_j[2][1] * kernel->dy + aniso_j[2][2] * kernel->dz);
#endif // NAVIERSTOKES_ANISO

    /*Acceleration due to the bulk viscosity */
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Computes the final contribution of Navier-Stokes viscosity to the entropy evolution of a particle.
 *
 * This function calculates the final contribution of Navier-Stokes viscosity to the entropy evolution of a particle
 * after considering the shear viscosity and bulk viscosity effects. It updates the entropy change rate of the particle.
 *
 * @param i Index of the particle.
 * @param hubble_a2 The square of the Hubble parameter scaled by the expansion factor.
 * @param a3inv Inverse of the cube of the scale factor.
 */
static inline void navierstokes_viscosity_final(int i, MyAtLeastDouble hubble_a2, MyAtLeastDouble a3inv)
{
    MyAtLeastDouble fac = 0;

#ifdef NAVIERSTOKES_ANISO
    /* Pi_ab * grad(v) */
    fac = SphP[i].P_aniso * SphP[i].P_aniso / (3. * SphP[i].shear_visc_coeff);

#else
    MyAtLeastDouble sigma2 = 0.0;
    /* sigma_ab * sigma_ab */
    for (int k = 0; k < 3; k++)
    {
        sigma2 +=
            SphP[i].StressDiag[k] * SphP[i].StressDiag[k] + 2 * SphP[i].StressOffDiag[k] * SphP[i].StressOffDiag[k];
    }

    fac = 0.5 * sigma2 * SphP[i].shear_visc_coeff;

#endif // NAVIERSTOKES_ANISO

#ifdef NAVIERSTOKES_TIMESTEP
    SphP[i].ViscEntropyChange = GAMMA_MINUS1 / (hubble_a2 * pow(SphP[i].Density, GAMMA_MINUS1)) * fac / SphP[i].Density;

#if GADGET_HYDRO == HYDRO_MFM
    SphP[i].DtEntropyExtra +=
#else // GADGET_HYDRO == HYDRO_MFM
    SphP[i].DtEntropy +=
#endif // GADGET_HYDRO == HYDRO_MFM
        SphP[i].ViscEntropyChange;

#else
#if GADGET_HYDRO == HYDRO_MFM
    SphP[i].DtEntropyExtra +=
#else // GADGET_HYDRO == HYDRO_MFM
    SphP[i].DtEntropy +=
#endif // GADGET_HYDRO == HYDRO_MFM
        GAMMA_MINUS1 / (hubble_a2 * pow(SphP[i].Density, GAMMA_MINUS1)) * fac / SphP[i].Density;
#endif // NAVIERSTOKES_TIMESTEP

#ifdef OUTPUT_ENERGY_VISC
    MyAtLeastDouble dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;
    MyAtLeastDouble change_E = GAMMA_MINUS1 / (hubble_a2 * pow(SphP[i].Density, GAMMA_MINUS1)) * fac / SphP[i].Density;
    SphP[i].visc_E += change_E * dt;
#endif // OUTPUT_ENERGY_VISC

    /* these entropy increases directly follow from the general heat transfer equation */
}
#endif // ACC_INCLUDE_HYDRA
#endif // NAVIERSTOKES

#ifdef WINDS
#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Handles wind particle properties and decoupling during wind initialization.
 *
 * This function is responsible for handling wind particle properties and decoupling
 * during wind initialization. If a particle's delay time is greater than zero, indicating
 * that it is part of a wind event, it resets various properties of the particle such as
 * hydrodynamic acceleration, entropy change rate, and magnetic field properties. It also
 * calculates the wind speed and adjusts the maximum signal velocity of the particle accordingly.
 *
 * @param i Index of the particle for which wind properties are handled.
 * @param fac_mu Factor controlling the wind speed.
 * @param a3inv Inverse of the scale factor cubed.
 */
static inline void do_wind_final(int i, MyAtLeastDouble fac_mu, MyAtLeastDouble a3inv)
{
    /* if we have winds, we decouple particles briefly if delaytime>0 */
    if (SphP[i].DelayTime > 0)
    {
#if GADGET_HYDRO == HYDRO_MFM
        SphP[i].dQdt = (MyAtLeastDouble)0.0;
        SphP[i].dQdt.GetEnergy() =
            eos->GetDtQFromDtEntropy((MyAtLeastDouble)0.0, SphP[i].EntropyPred, SphP[i].InternalEnergyPred, P[i].Mass,
                                     SphP[i].DivVel, All.cf_hubble_a);
        SphP[i].DtInternalEnergy =
            eos->GetDtUFromDtEntropy((MyFloat)0.0, SphP[i].EntropyPred, SphP[i].InternalEnergyPred, SphP[i].DivVel,
                                     All.cf_hubble_a, All.ComovingIntegrationOn);
        SphP[i].DtEntropyExtra = 0;
#endif // GADGET_HYDRO == HYDRO_MFM

        for (int k = 0; k < 3; k++)
        {
            SphP[i].HydroAccel[k] = 0;
        }

        SphP[i].DtEntropy = 0;

#ifdef MAGNETIC
        for (int k = 0; k < 3; k++)
        {
            SphP[i].magacc[k] = 0.0;
            SphP[i].magcorr[k] = 0.0;
            SphP[i].DtB[k] = 0.0;
            SphP[i].BPred[k] = SphP[i].WindB[k] / SphP[i].Density;
#if GADGET_HYDRO == HYDRO_MFM
            SphP[i].BVPred[k] = SphP[i].WindB[k] / P[i].Mass;
            for (int k = 0; k < 3; k++)
            {
                SphP[i].DtBV[k] = -SphP[i].DivVel * SphP[i].BPred[k] * dt_mag_correction;
            }
            if (All.ComovingIntegrationOn)
            {
                for (int k = 0; k < 3; k++)
                {
                    SphP[i].DtBV[k] += All.cf_hubble_a * SphP[i].BPred[k] * dt_mag_correction;
                }
            }
#endif // GADGET_HYDRO == HYDRO_MFM
        }
#endif // MAGNETIC

        MyAtLeastDouble windspeed, hsml_c;

#if ((WINDS + 0 == 0) && !defined(LT_STELLAREVOLUTION)) /* SH03 adaptive wind velocity case*/
        windspeed =
            sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) / All.WindEfficiency) *
            All.Time;
#else
#if WINDS + 0 > 0
        windspeed = WINDS * All.Time;
#else
        int IMFi, SFi;
        get_SF_index(i, &SFi, &IMFi);
        windspeed = sqrt(2 * SFs[SFi].WindEnergyFraction * SFs[SFi].totFactorSN * SFs[SFi].EgySpecSN /
                         (1 - SFs[SFi].totFactorSN) / SFs[SFi].WindEfficiency) *
                    All.Time;
#endif
#endif
        windspeed *= fac_mu;

#ifdef QUICK_LYALPHA
        hsml_c = pow(All.WindFreeTravelDensFac * All.OverDensThresh / SphP[i].Density, (1. / 3.));
#else // QUICK_LYALPHA
        MyAtLeastDouble myPhysDensThresh;
#ifdef LT_STELLAREVOLUTION
        myPhysDensThresh = get_PhysDensThresh(i);
#else // LT_STELLAREVOLUTION
        myPhysDensThresh = All.PhysDensThresh;
#endif // LT_STELLAREVOLUTION
        hsml_c = pow(All.WindFreeTravelDensFac * myPhysDensThresh / (SphP[i].Density * a3inv), (1. / 3.));
#endif // QUICK_LYALPHA
        SphP[i].MaxSignalVel = hsml_c * max((2 * windspeed), (MyAtLeastDouble)SphP[i].MaxSignalVel);
    }
}
#endif // ACC_INCLUDE_HYDRA
#endif // WINDS

#ifdef ARTIFICIAL_CONDUCTIVITY
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Evaluates artificial conductivity between neighboring particles with OpenACC support.
 *
 * This function evaluates the artificial conductivity between neighboring particles.
 * It calculates the conductivity factor based on the mass of the neighboring particle,
 * inverse of the density product, and the kernel gradient. Depending on the simulation
 * settings, it adjusts the conductivity calculation for multiphase particles, wind particles,
 * and whether time-dependent conductivity is enabled. Finally, it updates the rate of entropy
 * change based on the calculated conductivity factor, the conductivity coefficient, the
 * difference in internal energy between the particles, and the velocity difference between them.
 *
 * @param local Pointer to the hydrodynamic data of the local particle.
 * @param kernel Pointer to the kernel properties used for smoothing.
 * @param j Index of the neighboring particle.
 * @param P Pointer to the gravity walk data structure.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 * @param out Pointer to the hydrodynamic output data structure for storing results.
 */
void artificial_conductivity_evaluate_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                              struct gravity_walkdata *P, struct hydrodata_in *SphP,
                                              struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Evaluates artificial conductivity between neighboring particles.
 *
 * This function evaluates the artificial conductivity between neighboring particles.
 * It calculates the conductivity factor based on the mass of the neighboring particle,
 * inverse of the density product, and the kernel gradient. Depending on the simulation
 * settings, it adjusts the conductivity calculation for multiphase particles, wind particles,
 * and whether time-dependent conductivity is enabled. Finally, it updates the rate of entropy
 * change based on the calculated conductivity factor, the conductivity coefficient, the
 * difference in internal energy between the particles, and the velocity difference between them.
 * Folowing the description of Beck et al. 2016
 *
 * @param local Pointer to the hydrodynamic data of the local particle.
 * @param kernel Pointer to the kernel properties used for smoothing.
 * @param j Index of the neighboring particle.
 * @param out Pointer to the hydrodynamic output data structure for storing results.
 */
static inline void artificial_conductivity_evaluate(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                                    struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
    // first and last part in EQ (26) of Beck+ 2016. Note that <F_ij> is equivalent to dwk_ij
    MyAtLeastDouble c_factor = P[j].Mass * kernel->rho_ij_inv * kernel->dwk_ij;
    // vsigc is the signal velocity based on the soundspeed only, EQ (27) of Beck+ 2016
    MyAtLeastDouble vsigc = sqrt(abs((MyAtLeastDouble)local->Pressure - SphP[j].Pressure) * kernel->rho_ij_inv);
    // simple way to recover u, the missing (gamma-1) is taken care at the end once
    MyAtLeastDouble u_i = local->Pressure * kernel->inv_density_i;
    MyAtLeastDouble u_j = SphP[j].Pressure / SphP[j].Density;

#ifdef TIME_DEP_ART_COND
    // averaged alpha, see text after EQ (27) in Beck+ 2016
    MyAtLeastDouble alpha_c = 0.5 * (local->Calpha + SphP[j].Calpha);

// we exclude wind particles from the artificial conduction
#ifdef WINDS
    if (P[j].Type == type_gas)
    {
        if (SphP[j].DelayTime > 0 || local->DelayTime > 0.0)
        {
            alpha_c = 0.;
        }
    }
#endif // WINDS

// MUPPI also excludes multi phase particles and others

#else // TIME_DEP_ART_COND
    // fixed alpha to be used in EQ (26) of Beck+ 2016 when no timedependence is used
    MyAtLeastDouble alpha_c = All.ArtCondConstant;
#endif // TIME_DEP_ART_COND
       // final equation (26) from Beck+ 2016. Note that this is still du and will be converted to dEntr in
       // the final operation loop. Note also it is u_i - u_j to have the correct sign, as dW is negative,
       // (the u_j - u_i is a typo in the original paper!)
    out->DtEntropy += c_factor * alpha_c * vsigc * (u_i - u_j) / GAMMA_MINUS1;

#ifdef GT_DUST
    u_i /= (1. - local->TotalDustFracPred);
    u_j /= (1. - SphP[j].TotalDustFracPred);
    c_factor /= 0.5 * ((1. - local->TotalDustFracPred) + (1. - SphP[j].TotalDustFracPred));
    out->DtInternalEnergySPH += c_factor * alpha_c * vsigc * (u_i - u_j) / GAMMA_MINUS1;
#endif // GT_DUST
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA
#endif // ARTIFICIAL_CONDUCTIVITY



#ifdef TIME_DEP_ART_COND
#if !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0)
#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Evaluates the artificial conductivity limit for a particle.
 *
 * This function calculates the artificial conductivity limit for a particle based on its
 * hydrodynamic acceleration and gravitational acceleration. It computes the projection and
 * correction vectors for the hydrodynamic acceleration, ensuring that the correction vector is
 * perpendicular to the gravitational acceleration. If the hydrodynamic acceleration is positive,
 * it calculates the conductivity limit using the corrected acceleration. Otherwise, it sets the
 * conductivity limit to zero.
 * Folowing the description of Beck et al. 2016
 *
 * @param i Index of the particle for which to evaluate the artificial conductivity limit.
 */
static inline void eval_art_cond(int i)
{
    MyAtLeastDouble c_proj[3], c_corr[3];
    // absolute value of hydro force (squared), as in lower part of EQ (28) in Becki+ 2016
    MyAtLeastDouble c_hydro = SphP[i].HydroAccel[0] * SphP[i].HydroAccel[0] +
                              SphP[i].HydroAccel[1] * SphP[i].HydroAccel[1] +
                              SphP[i].HydroAccel[2] * SphP[i].HydroAccel[2];
    // vector product of hydro force and gravity force, as in upper part of EQ (28) in Beck+ 2016
    MyAtLeastDouble c_scalar = P[i].GravAccel[0] * SphP[i].HydroAccel[0] + P[i].GravAccel[1] * SphP[i].HydroAccel[1] +
                               P[i].GravAccel[2] * SphP[i].HydroAccel[2];

    if (c_hydro > 0)
    {
        // Projected part of gravity force F^p on hydro forece, see equation (28) of Beck+ 2016
        for (int k = 0; k < 3; k++)
        {
            c_proj[k] = SphP[i].HydroAccel[k] * c_scalar / c_hydro;
        }
        // Gravitational adjusted hydro force F^c, see equation (29) of Beck+ 2016
        for (int k = 0; k < 3; k++)
        {
            c_corr[k] = SphP[i].HydroAccel[k] + c_proj[k];
        }
        // delta_c limiter, see equation (30) in Beck+ 2016. Note that we set q=1
        SphP[i].Climit = (SphP[i].HydroAccel[0] * c_corr[0] + SphP[i].HydroAccel[1] * c_corr[1] +
                          SphP[i].HydroAccel[2] * c_corr[2]) /
                         c_hydro;
    }
    else
    {
        SphP[i].Climit = 0.0;
    }
}
#endif // ACC_INCLUDE_HYDRA
#endif // !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0)
#endif // TIME_DEP_ART_COND

#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Calculates the signal velocity between two particles with OpenACC support.
 *
 * This function calculates the signal velocity between two particles based on the given
 * hydrodynamic and magnetic properties, as well as simulation parameters. If magnetic signal
 * velocity computation is enabled, it considers the Alfven speed and sound speed. Otherwise, it
 * uses the sum of the sound speeds.
 *
 * @param local Pointer to the hydrodynamic data of the local particle.
 * @param kernel Pointer to the kernel properties.
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation.
 * @param P Pointer to the particle data structure.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 * @param out Pointer to the output hydrodynamic data.
 */
void calculate_vsig_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                            struct hydrodata_additional *additional_variables, struct gravity_walkdata *P,
                            struct hydrodata_in *SphP, struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Calculates the signal velocity between two particles.
 *
 * This function calculates the signal velocity between two particles based on the given
 * hydrodynamic and magnetic properties, as well as simulation parameters. If magnetic signal
 * velocity computation is enabled, it considers the Alfven speed and sound speed. Otherwise, it
 * uses the sum of the sound speeds.
 *
 * @param local Pointer to the hydrodynamic data of the local particle.
 * @param kernel Pointer to the kernel properties.
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation.
 * @param out Pointer to the output hydrodynamic data.
 */
static inline void calculate_vsig(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                  struct hydrodata_additional *additional_variables, struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
#ifdef MAGNETIC
    // Alfven speed squared, in the comoving convention (v_comoving = a * v_physical) that
    // kernel->sound_j and hence vcsa2_j below use. B is stored comoving (B_code = a^2 B_phys,
    // see bfld_to_physical) over a comoving density, so b2*mu0_inv/Density comes out as
    // a*v_alfven_phys^2 and needs one more factor of a.
    kernel->alfven2_j = kernel->b2_j * mu0_inv / SphP[j].Density * units->atime;

    // sound speed squared + alfven speed squared for particle i, see Dolag&Stasyszyn 2009 (eq. 12, upper part)
    additional_variables->vcsa2_j = kernel->sound_j * kernel->sound_j + kernel->alfven2_j;

    // magnetic field vector of particle j projected along kernel gradient
    MyAtLeastDouble Bpro2_j =
        (additional_variables->bpred_j[0] * kernel->dx + additional_variables->bpred_j[1] * kernel->dy +
         additional_variables->bpred_j[2] * kernel->dz) /
        kernel->r;
    Bpro2_j *= Bpro2_j; // squared

    // Alfven speed squared along the kernel gradient, carrying the same factor of atime as
    // alfven2_j above so that the whole dispersion relation is in the comoving convention.
    MyAtLeastDouble alfven2_par_j = Bpro2_j * mu0_inv / SphP[j].Density * units->atime;

    // speed of fast magnetosonic waves for particle j, see Dolag&Stasyszyn 2009 (eq. 12)
    additional_variables->magneticspeed_j =
        sqrt(0.5 * ((MyAtLeastDouble)additional_variables->vcsa2_j +
                    sqrt((MyAtLeastDouble)max((additional_variables->vcsa2_j * additional_variables->vcsa2_j -
                                               (MyAtLeastDouble)4 * kernel->sound_j * kernel->sound_j * alfven2_par_j),
                                              (MyAtLeastDouble)0))));

    // magnetic field vector of particle i projected along kernel gradient
    MyAtLeastDouble Bpro2_i =
        (local->BPred[0] * kernel->dx + local->BPred[1] * kernel->dy + local->BPred[2] * kernel->dz) / kernel->r;
    Bpro2_i *= Bpro2_i; // squared

    // Alfven speed squared along the kernel gradient, comoving convention as for particle j.
    MyAtLeastDouble alfven2_par_i = Bpro2_i * mu0_inv / local->Density * units->atime;

    // speed of fast magnetosonic waves for particle i, see Dolag&Stasyszyn 2009 (eq. 12)
    additional_variables->magneticspeed_i =
        sqrt(0.5 * (MyAtLeastDouble)(additional_variables->vcsa2_i +
                                     sqrt((MyAtLeastDouble)max(
                                         (additional_variables->vcsa2_i * additional_variables->vcsa2_i -
                                          (MyAtLeastDouble)4 * kernel->sound_i * kernel->sound_i * alfven2_par_i),
                                         (MyAtLeastDouble)0))));

    // maximum signal velocity of fast magneto-sonic waves
    kernel->vsig = additional_variables->magneticspeed_i + additional_variables->magneticspeed_j;
#else // MAGNETIC
    // maximum signal velocity of sound waves
    kernel->vsig = kernel->sound_i + kernel->sound_j;
#endif // MAGNETIC

#if defined(WINDS) && defined(AR_MAX_DELAYTIME_AFFECTS_SIGNALVEL)
    /* reasons: (1)  we avoid very small timebins on zoomins,
       (2) if winds are decoupled, why should they contribute anyway */
    if (SphP[j].DelayTime <= AR_MAX_DELAYTIME_AFFECTS_SIGNALVEL)
#endif // if defined(WINDS)  && defined(AR_MAX_DELAYTIME_AFFECTS_SIGNALVEL)
    {
#ifdef AR_FIX_TIME_DEP_ART_COND
        if (kernel->vsig > out->MaxSignalVelOut)
        {
            out->MaxSignalVelOut = kernel->vsig;
        }
#else // AR_FIX_TIME_DEP_ART_COND
        if (kernel->vsig > out->MaxSignalVel)
        {
            out->MaxSignalVel = kernel->vsig;
        }
#endif // AR_FIX_TIME_DEP_ART_COND
    }
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Computes the effective value for two particles, on the chosen averaging method.
 *
 * This function the effective value for any interaction constant based on the according values of the
 * two particles given a choice of the averaging techniques. The available methods are:
 * - 0: Maximum of the two input values.
 * - 1: Minimum of the two input values.
 * - 2: Arithmetic mean of the two input values.
 * - 3: Geometric mean of the two input values.
 * - 4: Harmonic mean of the two input values.
 *
 * @param mode defines the averaging mode used
 * @param val1 local value of the first particle.
 * @param val2 local value of the second interacting particle.
 * @return The computed bulk viscosity value.
 */
inline double compute_effective_value(const int mode, const double &val1, const double &val2)
#else // ACC_INCLUDE_HYDRA
inline double compute_effective_value_openacc(const int mode, const double &val1, const double &val2)
#endif // ACC_INCLUDE_HYDRA
{
    switch (mode)
    {
    case 0:
        return max(val1, val2);
    case 1:
        return min(val1, val2);
    case 2:
        return 0.5 * (val1 + val2);
    case 3:
        return sqrt(val1 * val2);
    case 4:
        return 2 * val1 * val2 / (val1 + val2);
    default:
#ifndef ACC_INCLUDE_HYDRA
        ENDRUN("Wrong value for mode in compute_effective_value");
#endif // ACC_INCLUDE_HYDRA
        break;
    }
#ifndef ACC_INCLUDE_HYDRA
    ENDRUN("impossible to reach ?! in compute_effective_value");
#endif // ACC_INCLUDE_HYDRA
    return -1;
}

#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Calculates artificial viscosity and artificial hydrodynamical diffusion with OpenACC support.
 *
 * This function computes the artificial viscosity and artificial hydrodynamical diffusion between
 * two particles based on their hydrodynamic properties, simulation parameters, and gravitational
 * interactions. It evaluates the necessity of artificial viscosity based on the dot product of the
 * velocity gradient and the relative velocity of particles. If necessary, it computes the
 * artificial viscosity and updates the signal velocity accordingly. Additionally, it limits the
 * magnitude of viscous acceleration to avoid excessive changes in particle velocities. Finally, it
 * calculates the contribution of artificial viscosity and diffusion to the particle's acceleration
 * and entropy evolution.
 *
 * @param local Pointer to the hydrodynamic data of the local particle.
 * @param kernel Pointer to the kernel properties.
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation.
 * @param fac_mu Factor for artificial viscosity calculation.
 * @param fac_vsic_fix Factor for limiting the magnitude of viscous acceleration.
 * @param P Pointer to the particle data.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 * @param out Pointer to the output hydrodynamic data.
 */
void calculate_art_visc_and_ahydro_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                           struct hydrodata_additional *additional_variables, MyAtLeastDouble fac_mu,
                                           MyAtLeastDouble fac_vsic_fix, struct gravity_walkdata *P,
                                           struct hydrodata_in *SphP, struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Calculates artificial viscosity and artificial hydrodynamical diffusion.
 *
 * This function computes the artificial viscosity and artificial hydrodynamical diffusion between
 * two particles based on their hydrodynamic properties, simulation parameters, and gravitational
 * interactions. It evaluates the necessity of artificial viscosity based on the dot product of the
 * velocity gradient and the relative velocity of particles. If necessary, it computes the
 * artificial viscosity and updates the signal velocity accordingly. Additionally, it limits the
 * magnitude of viscous acceleration to avoid excessive changes in particle velocities. Finally, it
 * calculates the contribution of artificial viscosity and diffusion to the particle's acceleration
 * and entropy evolution.
 *
 * @param local Pointer to the hydrodynamic data of the local particle.
 * @param kernel Pointer to the kernel properties.
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation.
 * @param fac_mu Factor for artificial viscosity calculation.
 * @param fac_vsic_fix Factor for limiting the magnitude of viscous acceleration.
 * @param out Pointer to the output hydrodynamic data.
 */
static inline void calculate_art_visc_and_ahydro(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                                 struct hydrodata_additional *additional_variables,
                                                 MyAtLeastDouble fac_mu, MyAtLeastDouble fac_vsic_fix,
                                                 struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
    /* kernel->r is divided into repeatedly below; compute the reciprocal once and multiply instead */
    MyAtLeastDouble inv_kernel_r = 1. / kernel->r;
    MyAtLeastDouble visc;
#if defined(TIME_DEP_ART_VISC) && (TIME_DEP_ART_VISC == 3)
    MyAtLeastDouble visc_i, visc_j;
#endif // defined(TIME_DEP_ART_VISC) && (TIME_DEP_ART_VISC == 3)
    if (kernel->vdotr < 0) /* ... artificial viscosity */
    {
#ifndef CONVENTIONAL_VISCOSITY
        MyAtLeastDouble mu_ij = fac_mu * kernel->vdotr * inv_kernel_r; /* note: this is negative! */
#else // CONVENTIONAL_VISCOSITY
        MyAtLeastDouble h_ij = 0.5 * (kernel->h_i + kernel->h_j);
        MyAtLeastDouble mu_ij = fac_mu * h_ij * kernel->vdotr / (kernel->r2 + 0.0001 * h_ij * h_ij);
#endif // CONVENTIONAL_VISCOSITY

#if defined(ROTB_VISCOSITY_LIMITER) && defined(MAGNETIC)
        MyAtLeastDouble absRotB =
            sqrt((MyAtLeastDouble)(SphP[j].RotB[1] * SphP[j].RotB[1] + SphP[j].RotB[2] * SphP[j].RotB[2] +
                                   SphP[j].RotB[3] * SphP[j].RotB[3]));

        // Dolag+2005, Eq. 5 equivalent for MHD using the Alfvén velocity instead of sound speed (tbd)
        MyAtLeastDouble fj_mhd =
            abs(SphP[j].divB) / (abs(SphP[j].divB) + absRotB +
                                 BALSARA_FAC_FOR_CSND_OVER_H * sqrt((MyAtLeastDouble)kernel->alfven2_j) / kernel->h_j);

        MyAtLeastDouble f_balsara_mhd = 0.5 * (local->F1MHD + fj_mhd);
#endif // defined(ROTB_VISCOSITY_LIMITER) && defined(MAGNETIC)

#if !defined(TIME_DEP_ART_VISC) || (TIME_DEP_ART_VISC < 3)
        MyAtLeastDouble f_balsara_mhd = 1.0;
        MyAtLeastDouble f_balsara = 1.0;
#endif

#ifndef NO_BALSARA_LIMITER
#if !defined(TIME_DEP_ART_VISC) || (TIME_DEP_ART_VISC < 3)
        f_balsara = 0.5 * (local->F_Balsara + SphP[j].F_Balsara);
#endif // defined(TIME_DEP_ART_VISC) && (TIME_DEP_ART_VISC < 3)
#endif // NO_BALSARA_LIMITER

#if defined(CONVENTIONAL_VISCOSITY) || !defined(TIME_DEP_ART_VISC) || (TIME_DEP_ART_VISC != 3)
#ifndef ACC_INCLUDE_HYDRA
        MyAtLeastDouble BulkVisc_ij = compute_effective_value(Average_Art_Visc, local->alpha, SphP[j].alpha);
#else // ACC_INCLUDE_HYDRA
        MyAtLeastDouble BulkVisc_ij = compute_effective_value_openacc(Average_Art_Visc, local->alpha, SphP[j].alpha);
#endif // ACC_INCLUDE_HYDRA
#endif // CONVENTIONAL_VISCOSITY || !TIME_DEP_ART_VISC && TIME_DEP_ART_VISC != 3

        MyAtLeastDouble betaAV_i = All.ArtBulkViscBeta;
        MyAtLeastDouble betaAV_j = All.ArtBulkViscBeta;

#if defined(TIME_DEP_ART_VISC) && (TIME_DEP_ART_VISC == 3)
        betaAV_i *= local->alpha;
        betaAV_j *= SphP[j].alpha;
        if (betaAV_i < Art_Visc_Beta_Min)
        {
            betaAV_i = Art_Visc_Beta_Min;
        }
        if (betaAV_j < Art_Visc_Beta_Min)
        {
            betaAV_j = Art_Visc_Beta_Min;
        }
#endif // TIME_DEP_ART_VISC == 3
        MyAtLeastDouble betaAV = 0.5 * (betaAV_i + betaAV_j);

#ifndef CONVENTIONAL_VISCOSITY
#if defined(TIME_DEP_ART_VISC) && (TIME_DEP_ART_VISC == 3)
        // Following Price 2018, seperating the quadratic term, not multiplying it by alpha
        visc_i = (0.5 * local->alpha * kernel->vsig * (-mu_ij) + betaAV_i * mu_ij * mu_ij) * kernel->inv_density_i;
        visc_j = (0.5 * SphP[j].alpha * kernel->vsig * (-mu_ij) + betaAV_j * mu_ij * mu_ij) / SphP[j].Density;
        visc = visc_i;
#else // (TIME_DEP_ART_VISC == 3)
      // Folowing Gadget2 and Gadget3 original implementation
        visc = 0.5 * BulkVisc_ij * (kernel->vsig - betaAV * mu_ij) * (-mu_ij);
#endif // (TIME_DEP_ART_VISC == 3)
#else // CONVENTIONAL_VISCOSITY
        MyAtLeastDouble c_ij = 0.5 * (kernel->sound_i + kernel->sound_j);
        visc = (-BulkVisc_ij * mu_ij * c_ij + 2 * BulkVisc_ij * mu_ij * mu_ij);
#endif // CONVENTIONAL_VISCOSITY

#if !defined(TIME_DEP_ART_VISC) || (TIME_DEP_ART_VISC < 3)
        visc *= kernel->rho_ij_inv * f_balsara * f_balsara_mhd;
#endif // !defined(TIME_DEP_ART_VISC) || (TIME_DEP_ART_VISC < 3)

        kernel->vsig -= betaAV * mu_ij;

        /* .... end artificial viscosity evaluation */
        /* now make sure that viscous acceleration is not too large */
#ifndef NOVISCOSITYLIMITER
        MyAtLeastDouble dt =
            2 * max((local->TimeBin ? (1 << local->TimeBin) : 0), (P[j].TimeBin ? (1 << P[j].TimeBin) : 0)) *
            All.Timebase_interval;
        if (dt > 0 && kernel->dwk_ij < 0)
        {
#ifdef BLACK_HOLES
            if ((local->Mass + P[j].Mass) > 0)
#endif // BLACK_HOLES
            {
                visc = min(visc, (MyFloat)0.5 * fac_vsic_fix * kernel->vdotr /
                                     ((local->Mass + P[j].Mass) * kernel->dwk_ij * kernel->r * dt));
#if defined(TIME_DEP_ART_VISC) && (TIME_DEP_ART_VISC == 3)
                if (kernel->dwk_i < 0)
                    visc_i = min(visc_i, (MyFloat)0.5 * fac_vsic_fix * kernel->vdotr /
                                             ((local->Mass + P[j].Mass) * kernel->dwk_i * kernel->r * dt));
                if (kernel->dwk_j < 0)
                    visc_j = min(visc_j, (MyFloat)0.5 * fac_vsic_fix * kernel->vdotr /
                                             ((local->Mass + P[j].Mass) * kernel->dwk_j * kernel->r * dt));
#endif // defined(TIME_DEP_ART_VISC) && (TIME_DEP_ART_VISC == 3)
            }
        }
#endif // NOVISCOSITYLIMITER

#ifdef AR_FIX_TIME_DEP_ART_COND
        if (kernel->vsig > out->MaxSignalVelOut)
        {
            out->MaxSignalVelOut = kernel->vsig;
        }
#else // AR_FIX_TIME_DEP_ART_COND
        if (kernel->vsig > out->MaxSignalVel)
        {
            out->MaxSignalVel = kernel->vsig;
        }
#endif // AR_FIX_TIME_DEP_ART_COND
    }
    else
    {
        visc = 0;
#if defined(TIME_DEP_ART_VISC) && (TIME_DEP_ART_VISC == 3)
        visc_i = 0;
        visc_j = 0;
#endif // defined(TIME_DEP_ART_VISC) && (TIME_DEP_ART_VISC == 3)
    }

#if GADGET_HYDRO == HYDRO_SPH
    MyAtLeastDouble hfc_visc, hfc_visc_dE;
#if !defined(TIME_DEP_ART_VISC) || (TIME_DEP_ART_VISC < 3)
    hfc_visc = P[j].Mass * visc * kernel->dwk_ij * inv_kernel_r;
    hfc_visc_dE = hfc_visc;
#else // !defined(TIME_DEP_ART_VISC) || (TIME_DEP_ART_VISC < 3)
    hfc_visc = 0.5 * P[j].Mass * inv_kernel_r * (visc_i * kernel->dwk_i + visc_j * kernel->dwk_j);
    hfc_visc_dE = P[j].Mass * inv_kernel_r * visc_i * kernel->dwk_i;
#endif // !defined(TIME_DEP_ART_VISC) || (TIME_DEP_ART_VISC < 3)

#ifdef GT_DUST
    MyAtLeastDouble hfc_visc_u =
        0.5 * P[j].Mass * visc / (1. - local->TotalDustFracPred) * kernel->vdotr * kernel->dwk_ij * inv_kernel_r;
    hfc_visc *= 1. - 0.5 * (local->TotalDustFracPred + SphP[j].TotalDustFracPred) *
                         additional_variables->DeltaVelDissipationSwitch;
    hfc_visc_dE *= 1. - 0.5 * (local->TotalDustFracPred + SphP[j].TotalDustFracPred) *
                            additional_variables->DeltaVelDissipationSwitch;
    hfc_visc_u *= 1. - 0.5 * (local->TotalDustFracPred + SphP[j].TotalDustFracPred) *
                           additional_variables->DeltaVelDissipationSwitch;
#endif // GT_DUST

    /* Formulation derived from the Lagrangian */
    MyAtLeastDouble hfc = hfc_visc + P[j].Mass *
                                         (additional_variables->p_over_rho2_i * kernel->dwk_i +
                                          additional_variables->p_over_rho2_j * kernel->dwk_j) *
                                         inv_kernel_r;
#elif GADGET_HYDRO == HYDRO_PESPH
    MyAtLeastDouble hfc_visc = P[j].Mass * visc * kernel->dwk_ij * inv_kernel_r;
    MyAtLeastDouble hfc_visc_dE = hfc_visc;
    MyAtLeastDouble hfc = hfc_visc + P[j].Mass * pow(local->Entropy * SphP[j].Entropy, GAMMA_INV) *
                                         (pow(local->Pressure, 1.0 - 2.0 * GAMMA_INV) * kernel->dwk_i +
                                          pow(SphP[j].Pressure, 1.0 - 2.0 * GAMMA_INV) * kernel->dwk_j) *
                                         inv_kernel_r;
#elif GADGET_HYDRO == HYDRO_MFM
    visc = 0; // KD: As we do not multiply alpha with beta in the visc term, this is currently the only way
              // to have mu_ij in the signal vel for the time stepping and get zero art viscosity for MFM
    MyAtLeastDouble hfc_visc = P[j].Mass * visc * kernel->dwk_ij * inv_kernel_r;
    MyAtLeastDouble hfc_visc_dE = hfc_visc;
    MyAtLeastDouble hfc = hfc_visc;
#endif // GADGET_HYDRO

#ifdef WINDS
    if (P[j].Type == type_gas)
    {
        if (SphP[j].DelayTime > 0) /* No force by wind particles */
        {
            hfc = hfc_visc = hfc_visc_dE = 0;
        }
    }
#endif // WINDS

#ifndef NOACCEL
    out->Acc[0] += -hfc * kernel->dx;
    out->Acc[1] += -hfc * kernel->dy;
    out->Acc[2] += -hfc * kernel->dz;
#endif // NOACCEL

    out->DtEntropy += 0.5 * hfc_visc_dE * kernel->vdotr;

#ifdef GT_DUST
#if GADGET_HYDRO == HYDRO_SPH
    out->DtInternalEnergySPH += hfc_visc_u;

    hfc += (additional_variables->DeltaVelDissipationSwitch - 1.) * hfc_visc;
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        out->DtDeltaVel[idust][0] += hfc / (1. - local->TotalDustFracPred) * kernel->dx;
        out->DtDeltaVel[idust][1] += hfc / (1. - local->TotalDustFracPred) * kernel->dy;
        out->DtDeltaVel[idust][2] += hfc / (1. - local->TotalDustFracPred) * kernel->dz;
    }
#endif // GADGET_HYDRO
#endif // GT_DUST
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Scales the change in energy rate to the change in entropy rate.
 *
 * This function translates the rate of change of energy (`DtEntropy`) into the rate of change of
 * entropy (`DtEntropy`) for a given particle. It utilizes the polytropic equation of state to
 * scale the energy change rate, considering the particle's density and the cosmological expansion
 * rate (`hubble_a2`). If the simulation hydrodynamics scheme is MFM, it also
 * scales the additional entropy change rate (`DtEntropyExtra`) accordingly.
 *
 * @param i Index of the particle.
 * @param hubble_a2 The square of the Hubble parameter, representing the cosmological expansion rate.
 */
static inline void scale_dt_entropy(int i, MyAtLeastDouble hubble_a2)
{
    /* Translate energy change rate into entropy change rate */
    SphP[i].DtEntropy *= GAMMA_MINUS1 / (hubble_a2 * pow(SphP[i].Density, GAMMA_MINUS1));
#if GADGET_HYDRO == HYDRO_MFM
    SphP[i].DtEntropyExtra *= GAMMA_MINUS1 / (hubble_a2 * pow(SphP[i].Density, GAMMA_MINUS1));
#endif // GADGET_HYDRO == HYDRO_MFM
}
#endif // ACC_INCLUDE_HYDRA

#if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Scales the gradient of energy by the appropriate factor.
 *
 * This function scales the gradient of energy (`GradEnergy`) for a given particle by dividing
 * each component by the product of the adiabatic index (`GAMMA_MINUS1`) and the particle's
 * density. This adjustment ensures that the gradient of energy is appropriately scaled for
 * further computations or analysis.
 *
 * @param i Index of the particle.
 */
static inline void scale_energy_gradient(int i)
{
    for (int k = 0; k < 3; k++)
    {
        SphP[i].GradEnergy[k] /= GAMMA_MINUS1 * SphP[i].Density;
    }
}
#endif // ACC_INCLUDE_HYDRA
#endif // if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)

#ifdef AA_OUTPUT_GRADS
#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Scales the gradient of pressure by the particle's density.
 *
 * This function scales the gradient of pressure (`GradPressure`) for a given particle by
 * dividing each component by the particle's density. This adjustment ensures that the gradient
 * of pressure is appropriately scaled for further computations or analysis.
 *
 * @param i Index of the particle.
 */
static inline void scale_pressure_gradient(int i)
{
    for (int k = 0; k < 3; k++)
    {
        SphP[i].GradPressure[k] /= SphP[i].Density;
    }
}
#endif // ACC_INCLUDE_HYDRA
#endif // AA_OUTPUT_GRADS

#if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Calculates the gradients of energy and pressure between neighboring particles with OpenACC support.
 *
 * This function calculates the gradients of energy and pressure between the local particle
 * and its neighboring particle. It updates the gradient of energy (`GradEnergy`) and optionally
 * the gradient of pressure (`GradPressure`) in the output data structure based on the given properties
 * and kernel values.
 *
 * @param local Pointer to the local properties of the particle.
 * @param kernel Pointer to the kernel properties used for interpolation.
 * @param j Index of the neighboring particle.
 * @param P Pointer to the particle data structure.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 * @param a3inv Inverse of the scale factor cubed.
 * @param out Pointer to the output data structure where the results are stored.
 */
void calculate_energy_and_pressure_gradients_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                                     struct gravity_walkdata *P, struct hydrodata_in *SphP,
                                                     MyAtLeastDouble a3inv, struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Calculates the gradients of energy and pressure between neighboring particles.
 *
 * This function calculates the gradients of energy and pressure between the local particle
 * and its neighboring particle. It updates the gradient of energy (`GradEnergy`) and optionally
 * the gradient of pressure (`GradPressure`) in the output data structure based on the given properties
 * and kernel values.
 *
 * @param local Pointer to the local properties of the particle.
 * @param kernel Pointer to the kernel properties used for interpolation.
 * @param j Index of the neighboring particle.
 * @param a3inv Inverse of the scale factor cubed.
 * @param out Pointer to the output data structure where the results are stored.
 */
static inline void calculate_energy_and_pressure_gradients(struct hydrodata_in *local, struct kernel_hydra *kernel,
                                                           int j, MyAtLeastDouble a3inv, struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
#if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
    MyAtLeastDouble gradEnergyVal = kernel->mj_r * kernel->dwk_i *
                                    (SphP[j].Entropy * pow(SphP[j].Density * a3inv, GAMMA_MINUS1) -
                                     local->Entropy * pow(local->Density * a3inv, GAMMA_MINUS1));
    out->GradEnergy[0] += gradEnergyVal * kernel->dx;
    out->GradEnergy[1] += gradEnergyVal * kernel->dy;
    out->GradEnergy[2] += gradEnergyVal * kernel->dz;

#endif // if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
#ifdef AA_OUTPUT_GRADS
    MyAtLeastDouble gradPressureVal =
        kernel->mj_r * kernel->dwk_i *
        (SphP[j].Entropy * pow(SphP[j].Density * a3inv, GAMMA) - local->Entropy * pow(local->Density * a3inv, GAMMA));
    out->GradPressure[0] += gradPressureVal * kernel->dx;
    out->GradPressure[1] += gradPressureVal * kernel->dy;
    out->GradPressure[2] += gradPressureVal * kernel->dz;
#endif // AA_OUTPUT_GRADS
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)

#ifdef GT_DUST
#if GADGET_HYDRO == HYDRO_SPH
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Calculates the acceleration, rate of change of dust fraction, delta velocity, and internal energy for dust
 * particles.
 *
 * This function calculates the acceleration, rate of change of dust fraction, delta velocity, and internal energy
 * for dust particles based on the properties of the local particle and its neighboring particle. The calculations
 * depend on various factors including pressure, density, entropy, and velocity gradients.
 *
 * @param local Pointer to the hydrodynamic data of the local particle.
 * @param kernel Pointer to the kernel properties.
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation.
 * @param P Pointer to the particle data.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 * @param out Pointer to the output hydrodynamic data.
 */
void calculate_ahydro_dust_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                   struct hydrodata_additional *additional_variables, struct gravity_walkdata *P,
                                   struct hydrodata_in *SphP, struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Calculates the acceleration, rate of change of dust fraction, delta velocity, and internal energy for dust
 * particles.
 *
 * This function calculates the acceleration, rate of change of dust fraction, delta velocity, and internal energy
 * for dust particles based on the properties of the local particle and its neighboring particle. The calculations
 * depend on various factors including pressure, density, entropy, and velocity gradients.
 *
 * @param local Pointer to the local properties of the dust particle.
 * @param kernel Pointer to the kernel properties used for interpolation.
 * @param j Index of the neighboring particle.
 * @param additional_variables Pointer to additional variables used in the calculation.
 * @param out Pointer to the output data structure where the results are stored.
 */
static inline void calculate_ahydro_dust(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                         struct hydrodata_additional *additional_variables, struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
    if (kernel->r > 0)
    {

        MyFloat term1, term2, term3, term4, term5, term6, dvdotr_i, dvdotr_j;
        MyFloat inv_density_i = 1. / local->Density;
        MyFloat inv_density_j = 1. / SphP[j].Density;
        MyFloat inv_kernel_r = 1. / kernel->r;

        MyFloat dv2_i =
            pow(local->DeltaVelPred[0][0], 2) + pow(local->DeltaVelPred[0][1], 2) + pow(local->DeltaVelPred[0][2], 2);
        MyFloat dv2_j = pow(SphP[j].DeltaVelPred[0][0], 2) + pow(SphP[j].DeltaVelPred[0][1], 2) +
                        pow(SphP[j].DeltaVelPred[0][2], 2);

        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            dvdotr_i =
                (local->DeltaVelPred[idust][0] - local->TotalDustFracPred * local->TotalDeltaVelPred[0]) * kernel->dx +
                (local->DeltaVelPred[idust][1] - local->TotalDustFracPred * local->TotalDeltaVelPred[1]) * kernel->dy +
                (local->DeltaVelPred[idust][2] - local->TotalDustFracPred * local->TotalDeltaVelPred[2]) * kernel->dz;
            dvdotr_j = (SphP[j].DeltaVelPred[idust][0] - SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[0]) *
                           kernel->dx +
                       (SphP[j].DeltaVelPred[idust][1] - SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[1]) *
                           kernel->dy +
                       (SphP[j].DeltaVelPred[idust][2] - SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[2]) *
                           kernel->dz;

            term1 = -P[j].Mass * kernel->dwk_i * dvdotr_i * inv_kernel_r * dustfrac(local->DustFracPred[idust]) *
                    inv_density_i;
            term2 = -P[j].Mass * kernel->dwk_j * dvdotr_j * inv_kernel_r * dustfrac(SphP[j].DustFracPred[idust]) *
                    inv_density_j;

            /************************************** DUST FRACTION ***************************************/
#if ((GT_DUST_PARAM + 0) == 0)
            out->DtDustFrac[idust] += term1 + term2;
#elif ((GT_DUST_PARAM + 0) == 1)
            out->DtDustFrac[idust] +=
                -0.5 * (1. + pow(local->DustFracPred[idust], 2)) * P[j].Mass * SphP[j].DustFracPred[idust] *
                (dvdotr_i * kernel->dwk_i * inv_kernel_r * inv_density_i +
                 (1 + pow(local->DustFracPred[idust], 2)) / (1. + pow(SphP[j].DustFracPred[idust], 2)) * dvdotr_j *
                     kernel->dwk_j * inv_kernel_r * inv_density_j);
#endif // GT_DUST_PARAM

            /***************************************** VELOCITY ******************************************/
            for (int k = 0; k < NUMDIMS; k++)
                out->Acc[k] += term1 * local->DeltaVelPred[idust][k] + term2 * SphP[j].DeltaVelPred[idust][k];

            /*************************************** DELTA VELOCITY **************************************/
            term3 = 0.0;
            for (int k = 0; k < NUMDIMS; k++)
            {
                term3 +=
                    0.5 *
                    (local->DeltaVelPred[idust][k] *
                         (local->DeltaVelPred[idust][k] - 2. * local->TotalDustFracPred * local->TotalDeltaVelPred[k]) -
                     SphP[j].DeltaVelPred[idust][k] * (SphP[j].DeltaVelPred[idust][k] -
                                                       2. * SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[k]));

                term3 += local->TotalDustFracPred * local->TotalDeltaVelPred[k] *
                         (local->DeltaVelPred[idust][k] - SphP[j].DeltaVelPred[idust][k]);

                term3 += local->DeltaVelPred[idust][k] *
                         ((local->DeltaVelPred[idust][k] - local->TotalDustFracPred * local->TotalDeltaVelPred[k]) -
                          (SphP[j].DeltaVelPred[idust][k] - SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[k]));
            }
            term4 = local->DeltaVelPred[idust][0] * kernel->dx + local->DeltaVelPred[idust][1] * kernel->dy +
                    local->DeltaVelPred[idust][2] * kernel->dz;
            term5 = local->TotalDustFracPred * local->TotalDeltaVelPred[0] * kernel->dx +
                    local->TotalDustFracPred * local->TotalDeltaVelPred[1] * kernel->dy +
                    local->TotalDustFracPred * local->TotalDeltaVelPred[2] * kernel->dz;

            out->DtDeltaVel[idust][0] += P[j].Mass * inv_density_i * term3 * kernel->dwk_i * kernel->dx;
            out->DtDeltaVel[idust][1] += P[j].Mass * inv_density_i * term3 * kernel->dwk_i * kernel->dy;
            out->DtDeltaVel[idust][2] += P[j].Mass * inv_density_i * term3 * kernel->dwk_i * kernel->dz;

            for (int k = 0; k < NUMDIMS; k++)
            {
                out->DtDeltaVel[idust][k] += P[j].Mass * inv_density_i * term4 * kernel->dwk_i * inv_kernel_r *
                                             ((local->Vel[k] + local->DeltaVelPred[idust][k] -
                                               local->TotalDustFracPred * local->TotalDeltaVelPred[k]) -
                                              (SphP[j].VelPred[k] + SphP[j].DeltaVelPred[idust][k] -
                                               SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[k]));
                out->DtDeltaVel[idust][k] -= P[j].Mass * inv_density_i * term5 * kernel->dwk_i * inv_kernel_r *
                                             (local->DeltaVelPred[idust][k] - SphP[j].DeltaVelPred[idust][k]);
            }
        }

        /************************************** INTERNAL ENERGY **************************************/
        term5 = P[j].Mass * additional_variables->p_over_rho2_i / (1. - local->TotalDustFracPred) * kernel->vdotr *
                kernel->dwk_i * inv_kernel_r;

        dvdotr_i = local->TotalDeltaVelPred[0] * kernel->dx + local->TotalDeltaVelPred[1] * kernel->dy +
                   local->TotalDeltaVelPred[2] * kernel->dz;
        term6 = -P[j].Mass * inv_density_i * (local->InternalEnergySPHPred - SphP[j].InternalEnergySPHPred) *
                local->TotalDustFracPred * dvdotr_i * kernel->dwk_i * inv_kernel_r;

        out->DtInternalEnergySPH += term5 + term6;
        /***********************************************************************************************/
    }
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Calculates the artificial dissipation of DeltaVel by averaging between the conservative and the
 * non-conservative formulations.
 *
 * This function computes the artificial dissipation of DeltaVel (and the corresponding dissipation in InternalEnergy,
 * if needed) by using both formulations presented in Laibe and Price, 2014b, meaning the conservative and the
 * non-conservative ones. Both are needed to obtain optimal results in all benchmarks. A switch is used which uses the
 * conservative approach for mainly shearing flows, while the non-conservative one is used for strong shocks.
 *
 * @param local Pointer to the hydrodynamic data of the local particle.
 * @param kernel Pointer to the kernel properties.
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation.
 * @param P Pointer to the particle data.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 * @param out Pointer to the output hydrodynamic data.
 */
void calculate_dust_dissipation_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                        struct hydrodata_additional *additional_variables, struct gravity_walkdata *P,
                                        struct hydrodata_in *SphP, struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Calculates the artificial dissipation of DeltaVel by averaging between the conservative and the
 * non-conservative formulations.
 *
 * This function computes the artificial dissipation of DeltaVel (and the corresponding dissipation in InternalEnergy,
 * if needed) by using both formulations presented in Laibe and Price, 2014b, meaning the conservative and the
 * non-conservative ones. Both are needed to obtain optimal results in all benchmarks. A switch is used which uses the
 * conservative approach for mainly shearing flows, while the non-conservative one is used for strong shocks.
 *
 * @param local Pointer to the local properties of the dust particle.
 * @param kernel Pointer to the kernel properties used for interpolation.
 * @param j Index of the neighboring particle.
 * @param additional_variables Pointer to additional variables used in the calculation.
 * @param out Pointer to the output data structure where the results are stored.
 */
static inline void calculate_dust_dissipation(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                              struct hydrodata_additional *additional_variables,
                                              struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
    if (kernel->r > 0)
    {
        /*********** ARTIFICIAL DISSIPATION  **********/
        MyFloat etaInternalEnergy = 0.0;
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            MyFloat etaDeltaVel = 0.0;
            MyFloat etaDeltaVel0 = 0.0;
            MyFloat etaDeltaVel1 = 0.0;

            if (kernel->vdotr < 0.)
            {
                etaDeltaVel0 += kernel->vsig * kernel->rho_ij_inv / onem_dustfrac(local->DustFracPred[idust]) *
                                kernel->vsig_DeltaVel[idust];
            }

            if (kernel->vsig_DeltaVel[idust] < 0.)
            {
                etaDeltaVel1 += fabs(kernel->vsig_DeltaVel[idust]) * kernel->rho_ij_inv * kernel->DeltaVeldotr2[idust];
                etaInternalEnergy += fabs(kernel->vsig_DeltaVel[idust]) * pow(kernel->DeltaVeldotr2[idust], 2);
            }

            etaDeltaVel =
                (etaDeltaVel1 - etaDeltaVel0) * additional_variables->DeltaVelDissipationSwitch + etaDeltaVel0;

            MyFloat DtDeltaVelDiss = P[j].Mass * etaDeltaVel * kernel->dwk_ij * All.DeltaVelDissipationConstant;
            out->DtDeltaVel[idust][0] += DtDeltaVelDiss * kernel->dx / kernel->r;
            out->DtDeltaVel[idust][1] += DtDeltaVelDiss * kernel->dy / kernel->r;
            out->DtDeltaVel[idust][2] += DtDeltaVelDiss * kernel->dz / kernel->r;
        }

        out->DtInternalEnergySPH += -0.5 / (1. - local->TotalDustFracPred) * P[j].Mass * kernel->rho_ij_inv *
                                    etaInternalEnergy * kernel->dwk_ij *
                                    additional_variables->DeltaVelDissipationSwitch;

        /***********************************************************************************************/
    }
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA
#endif // GADGET_HYDRO

#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Calculates the diffusion of dust through a pressure-like term in the dust momentum equation.
 *
 * This function calculates the diffusion of dust by using a pressure-like term in the dust momentum equation.
 * This results in new terms in the acceleration, DtDeltaVel and DtInternalEnergy equations.
 *
 * @param local Pointer to the hydrodynamic data of the local particle.
 * @param kernel Pointer to the kernel properties.
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation.
 * @param P Pointer to the particle data.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 * @param out Pointer to the output hydrodynamic data.
 */
void calculate_dust_diffusion_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                      struct hydrodata_additional *additional_variables, struct gravity_walkdata *P,
                                      struct hydrodata_in *SphP, struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Calculates the diffusion of dust through a pressure-like term in the dust momentum equation.
 *
 * This function calculates the diffusion of dust by using a pressure-like term in the dust momentum equation.
 * This results in new terms in the acceleration, DtDeltaVel and DtInternalEnergy equations.
 *
 * @param local Pointer to the local properties of the dust particle.
 * @param kernel Pointer to the kernel properties used for interpolation.
 * @param j Index of the neighboring particle.
 * @param additional_variables Pointer to additional variables used in the calculation.
 * @param out Pointer to the output data structure where the results are stored.
 */
static inline void calculate_dust_diffusion(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                            struct hydrodata_additional *additional_variables,
                                            struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
    if (kernel->r > 0)
    {
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            MyFloat chi_a = dustfrac(local->DustFracPred[idust]) / local->dust_fraction_smooth[idust];
            MyFloat chi_b = dustfrac(SphP[j].DustFracPred[idust]) / SphP[j].dust_fraction_smooth[idust];
            MyFloat D = 0.;
            if ((chi_a > 0.) && (chi_b > 0.))
            {
                D += 2. * ((pow(chi_a - 1., 2) / chi_a) + (pow(chi_b - 1., 2) / chi_b));
                SphP[j].DiffCoeff[idust] = pow(chi_b - 1., 2) / chi_b;
            }

            D = std::min(D, 20.);

            MyFloat diff = 0.;

#if ((GT_DUST_PARAM + 0) == 0)
            diff += -D * SphP[j].DragCoefficient[idust] / 3. / local->Density / local->DustFracPred[idust] /
                    pow(1. + local->DustFracPred[idust], 2) * P[j].Mass / SphP[j].Density *
                    (SphP[j].DustFracPred[idust] - local->DustFracPred[idust]) * kernel->dwk_ij;
#elif ((GT_DUST_PARAM + 0) == 1)
            MyFloat mean_s = 0.5 * (local->DustFracPred[idust] + SphP[j].DustFracPred[idust]);
            diff += -2. / 3. * D * SphP[j].DragCoefficient[idust] * mean_s / local->Density * P[j].Mass /
                    SphP[j].Density * (SphP[j].DustFracPred[idust] - local->DustFracPred[idust]) * kernel->dwk_ij;
#endif // GT_DUST_PARAM

            out->Acc[0] += diff * kernel->dx / kernel->r;
            out->Acc[1] += diff * kernel->dy / kernel->r;
            out->Acc[2] += diff * kernel->dz / kernel->r;

            out->DtDeltaVel[idust][0] += diff / additional_variables->mean_dust_fraction * kernel->dx / kernel->r;
            out->DtDeltaVel[idust][1] += diff / additional_variables->mean_dust_fraction * kernel->dy / kernel->r;
            out->DtDeltaVel[idust][2] += diff / additional_variables->mean_dust_fraction * kernel->dz / kernel->r;

            out->DtInternalEnergySPH += -diff / onem_dustfrac(local->DustFracPred[idust]) * kernel->vsig_dust;
        }
    }
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA
#endif // GT_DUST

#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Sets additional variables and kernel properties for the local particle with OpenACC support.
 *
 * This function sets additional variables and kernel properties for the local particle,
 * which are used in various calculations, such as sound speed, magnetic properties,
 * and kernel parameters. These properties are essential for hydrodynamic and
 * magnetohydrodynamic simulations.
 *
 * @param local Pointer to the local properties of the particle.
 * @param out Pointer to the output data structure where the results are stored.
 * @param atime Scale factor representing the expansion of the universe.
 * @param P Pointer to the gravity walk data structure.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 * @param additional_variables Pointer to additional variables used in the calculation.
 * @param kernel Pointer to the kernel properties used for interpolation.
 */
void set_additional_variables_and_kernel_i_openacc(struct hydrodata_in *local, struct hydrodata_out *out,
                                                   struct gravity_walkdata *P, struct hydrodata_in *SphP,
                                                   MyAtLeastDouble atime,
                                                   struct hydrodata_additional *additional_variables,
                                                   struct kernel_hydra *kernel)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Sets additional variables and kernel properties for the local particle.
 *
 * This function sets additional variables and kernel properties for the local particle,
 * which are used in various calculations, such as sound speed, magnetic properties,
 * and kernel parameters. These properties are essential for hydrodynamic and
 * magnetohydrodynamic simulations.
 *
 * @param local Pointer to the local properties of the particle.
 * @param out Pointer to the output data structure where the results are stored.
 * @param atime Scale factor representing the expansion of the universe.
 * @param additional_variables Pointer to additional variables used in the calculation.
 * @param kernel Pointer to the kernel properties used for interpolation.
 */
static inline void set_additional_variables_and_kernel_i(struct hydrodata_in *local, struct hydrodata_out *out,
                                                         MyAtLeastDouble atime,
                                                         struct hydrodata_additional *additional_variables,
                                                         struct kernel_hydra *kernel)
#endif // ACC_INCLUDE_HYDRA
{
    kernel->inv_density_i = 1. / local->Density;
    additional_variables->p_over_rho2_i = local->Pressure / (local->Density * local->Density);
#if GADGET_HYDRO == HYDRO_MFM
    // kernel->sound_i feeds kernel->vsig and expects a comoving code velocity
    kernel->sound_i = local->SoundSpeed * units->velocity_to_comoving;
#else // GADGET_HYDRO == HYDRO_MFM

    kernel->sound_i = sqrt((MyAtLeastDouble)GAMMA * additional_variables->p_over_rho2_i * local->Density);

#endif // GADGET_HYDRO == HYDRO_MFM
#ifdef GT_DUST
    kernel->sound_i /= sqrt((MyAtLeastDouble)1. - local->TotalDustFracPred);
#endif // GT_DUST
    kernel->h_i = local->Hsml;
    /* precompute once per target particle instead of once per neighbour pair (see kernel_hydra) */
    kernel_hinv(kernel->h_i, &kernel->hinv_i, &kernel->hinv3_i, &kernel->hinv4_i);
#ifdef AR_FIX_TIME_DEP_ART_COND
    out->MaxSignalVelOut = kernel->sound_i;
#else /* / AR_FIX_TIME_DEP_ART_COND */
    out->MaxSignalVel = kernel->sound_i;
#endif // AR_FIX_TIME_DEP_ART_COND

#ifdef MAGNETIC
    additional_variables->mf_windfac = 1.0;
    additional_variables->mf_Ind = 1.0 / local->Density;
    additional_variables->mf_i = mu0_inv * pow(atime, 3 * GAMMA - 2) / (local->Density * local->Density);
    additional_variables->mf_j = mu0_inv * pow(atime, 3 * GAMMA - 2);
#if defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION)
    additional_variables->mf_dissInd = local->Density * atime * atime;
#endif // if defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION)
#if defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION_HEAT)
    additional_variables->mf_dissEnt = 0.5 * mu0_inv * atime * atime;
#endif // if  defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION_HEAT)
    kernel->b2_i =
        local->BPred[0] * local->BPred[0] + local->BPred[1] * local->BPred[1] + local->BPred[2] * local->BPred[2];
    for (int k = 0; k < 3; k++)
    {
        for (int l = 0; l < 3; l++)
        {
            additional_variables->MagneticStressTensor_i[k][l] = local->BPred[k] * local->BPred[l];
        }
    }
    for (int k = 0; k < 3; k++)
    {
        additional_variables->MagneticStressTensor_i[k][k] -= 0.5 * kernel->b2_i;
    }
    // Comoving convention, to match kernel->sound_i (see the note on alfven2_j).
    kernel->alfven2_i = kernel->b2_i * mu0_inv / local->Density * units->atime;
    additional_variables->vcsa2_i = kernel->sound_i * kernel->sound_i + kernel->alfven2_i;

#ifdef MAGNETIC_DIFFUSION_LIMIT
    // half the velocity of the fast magnetosonic wave
    MyAtLeastDouble v_diff = sqrt((MyAtLeastDouble)0.5 * additional_variables->vcsa2_i);

    if (All.MagneticDiffSpeed > v_diff)
    {
        v_diff = 0.5 * (All.MagneticDiffSpeed + v_diff);
    }

    additional_variables->mf_dissLimit = 0.0;
    MyAtLeastDouble dissDist = v_diff * (local->TimeBin ? (1 << local->TimeBin) : 0) * All.Timebase_interval;
    if (All.ComovingIntegrationOn)
    {
        dissDist /= hubble_function(All.Time);
    }

    if (get_random_number(local->ID + 7) <= (dissDist / local->Hsml))
    {
        additional_variables->mf_dissLimit = 1.0;
    }
#endif // MAGNETIC_DIFFUSION_LIMIT

#endif // MAGNETIC
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Sets additional variables and kernel properties for the interaction between particles i and j with OpenACC
 * support.
 *
 * This function sets additional variables and kernel properties for the interaction between particles i and j,
 * which are used in various calculations, such as sound speed, magnetic properties, and kernel parameters.
 * These properties are essential for hydrodynamic and magnetohydrodynamic simulations.
 *
 * @param local Pointer to the local properties of particle i.
 * @param j Index of particle j in the simulation.
 * @param hubble_a2 The square of the Hubble expansion rate.
 * @param P Pointer to the gravity walk data structure.
 * @param SpP Pointer to the hydrodynamic particle data.
 * @param additional_variables Pointer to additional variables used in the calculation.
 * @param kernel Pointer to the kernel properties used for interpolation.
 */
void set_additional_variables_and_kernel_ij_openacc(struct hydrodata_in *local, int j, struct gravity_walkdata *P,
                                                    struct hydrodata_in *SphP, MyAtLeastDouble hubble_a2,
                                                    struct hydrodata_additional *additional_variables,
                                                    struct kernel_hydra *kernel)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Sets additional variables and kernel properties for the interaction between particles i and j.
 *
 * This function sets additional variables and kernel properties for the interaction between particles i and j,
 * which are used in various calculations, such as sound speed, magnetic properties, and kernel parameters.
 * These properties are essential for hydrodynamic and magnetohydrodynamic simulations.
 *
 * @param local Pointer to the local properties of particle i.
 * @param j Index of particle j in the simulation.
 * @param hubble_a2 The square of the Hubble expansion rate.
 * @param additional_variables Pointer to additional variables used in the calculation.
 * @param kernel Pointer to the kernel properties used for interpolation.
 */
static inline void set_additional_variables_and_kernel_ij(struct hydrodata_in *local, int j, MyAtLeastDouble hubble_a2,
                                                          struct hydrodata_additional *additional_variables,
                                                          struct kernel_hydra *kernel)
#endif // ACC_INCLUDE_HYDRA
{
    additional_variables->p_over_rho2_j = SphP[j].Pressure / (SphP[j].Density * SphP[j].Density);

#if GADGET_HYDRO == HYDRO_MFM
    // Comoving code velocity, to match kernel->sound_i
    kernel->sound_j = eos->SoundSpeed(SphP[j].InternalEnergyPred) * units->velocity_to_comoving;
#else // GADGET_HYDRO == HYDRO_MFM

    kernel->sound_j = sqrt((MyAtLeastDouble)GAMMA * additional_variables->p_over_rho2_j * SphP[j].Density);
#ifdef GT_DUST
    kernel->sound_j /= sqrt((MyAtLeastDouble)1. - SphP[j].TotalDustFracPred);
#endif // GT_DUST

#endif // GADGET_HYDRO == HYDRO_MFM
    kernel->dvx = local->Vel[0] - SphP[j].VelPred[0];
    kernel->dvy = local->Vel[1] - SphP[j].VelPred[1];
    kernel->dvz = local->Vel[2] - SphP[j].VelPred[2];
#ifdef DISK_SLICE
    kernel->dvz -= kernel->dv_phi_disk;
#endif // DISK_SLICE

#ifdef GT_DUST
    kernel->dvx -= local->TotalDustFracPred * local->TotalDeltaVelPred[0] -
                   SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[0];
    kernel->dvy -= local->TotalDustFracPred * local->TotalDeltaVelPred[1] -
                   SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[1];
    kernel->dvz -= local->TotalDustFracPred * local->TotalDeltaVelPred[2] -
                   SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[2];
#endif // GT_DUST

    kernel->vdotr = kernel->dx * kernel->dvx + kernel->dy * kernel->dvy + kernel->dz * kernel->dvz;
    kernel->rho_ij_inv = 2.0 / (local->Density + SphP[j].Density);

    if (All.ComovingIntegrationOn)
    {
        // add the cosmological contribution (hubble expansion): v_hubble * r,
        // where: v_hubble_physical = Hubble*r_physical,
        // r_physical = r*a, v_hubble_code = v_hubble_physical*a,
        // leading to the given form:
        kernel->vdotr += hubble_a2 * kernel->r2;
    }

    if (kernel->r < kernel->h_i)
    {
        MyAtLeastDouble u = kernel->r * kernel->hinv_i;
#if (defined(WINDS) && defined(MAGNETIC)) || (GADGET_HYDRO == HYDRO_MFM) || defined(LMB_SPECTRAL_CRs_SN_SEEDING) ||    \
    defined(DK_SPECTRAL_SN_SEEDING)
        kernel_main(u, kernel->hinv3_i, kernel->hinv4_i, &kernel->wk_i, &kernel->dwk_i, 0);
#else // if defined(WINDS) && defined(MAGNETIC)
        kernel_main(u, kernel->hinv3_i, kernel->hinv4_i, &kernel->wk_i, &kernel->dwk_i, 1);
#endif // if defined(WINDS) && defined(MAGNETIC)
    }
    else
    {
        kernel->dwk_i = 0;
        kernel->wk_i = 0;
    }

    if (kernel->r < kernel->h_j)
    {
        MyAtLeastDouble hinv, hinv3, hinv4;
        kernel_hinv(kernel->h_j, &hinv, &hinv3, &hinv4);
        MyAtLeastDouble u = kernel->r * hinv;
#if (defined(WINDS) && defined(MAGNETIC)) || (GADGET_HYDRO == HYDRO_MFM)
        kernel_main(u, hinv3, hinv4, &kernel->wk_j, &kernel->dwk_j, 0);
#else // if defined(WINDS) && defined(MAGNETIC)
        kernel_main(u, hinv3, hinv4, &kernel->wk_j, &kernel->dwk_j, 1);
#endif // if defined(WINDS) && defined(MAGNETIC)
    }
    else
    {
        kernel->dwk_j = 0;
        kernel->wk_j = 0;
    }
#ifdef ACC_INCLUDE_HYDRA // I dont know why PGI doesnt like the other way to assign
    kernel->dwk_ij = 0.5 * kernel->dwk_i;
    kernel->dwk_ij += 0.5 * kernel->dwk_j;
#else // ACC_INCLUDE_HYDRA
    kernel->dwk_ij = 0.5 * (kernel->dwk_i + kernel->dwk_j);
#endif // ACC_INCLUDE_HYDRA

#if GADGET_HYDRO == HYDRO_SPH
    kernel->dwk_i *= local->DhsmlDensityFactor;
    kernel->dwk_j *= SphP[j].DhsmlDensityFactor;
#elif GADGET_HYDRO == HYDRO_PESPH
    kernel->dwk_i *=
        (1.0 - local->DhsmlPressureFactor * local->DhsmlNumDensFactor / (pow(local->Entropy, GAMMA_INV) * local->Mass));
    kernel->dwk_j *= (1.0 - SphP[j].DhsmlPressureFactor * SphP[j].DhsmlNumDensFactor /
                                (pow(SphP[j].Entropy, GAMMA_INV) * P[j].Mass));
#elif GADGET_HYDRO == HYDRO_MFM
    kernel->dwk_i *= local->DhsmlNumDensFactor;
    kernel->dwk_j *= SphP[j].DhsmlNumDensFactor;
#endif // GADGET_HYDRO

#ifdef MAGNETIC
    kernel->mj_r = P[j].Mass / kernel->r;
    additional_variables->mf_windfac = 1;
#ifdef WINDS
    if (P[j].Type == type_gas)
    {
        if (SphP[j].DelayTime > 0)
        { /* ignore the wind particles */
            additional_variables->mf_windfac = 0;
        }
    }
#endif // WINDS
    kernel->mf_Ind = additional_variables->mf_windfac * additional_variables->mf_Ind * kernel->mj_r * kernel->dwk_i;
    kernel->mf_i = additional_variables->mf_windfac * additional_variables->mf_i * kernel->mj_r * kernel->dwk_i;
    kernel->mf_j = additional_variables->mf_windfac * additional_variables->mf_j * kernel->mj_r * kernel->dwk_j /
                   (SphP[j].Density * SphP[j].Density);
#if defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION)
    MyAtLeastDouble dissfac = kernel->mj_r * kernel->dwk_ij * kernel->rho_ij_inv * kernel->rho_ij_inv;
#ifdef MAGNETIC_DIFFUSION_LIMIT
    dissfac *= additional_variables->mf_dissLimit;
#endif // MAGNETIC_DIFFUSION_LIMIT
    kernel->mf_dissInd = additional_variables->mf_windfac * additional_variables->mf_dissInd * dissfac;
#endif // if  defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION)
#if defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION_HEAT)
    kernel->mf_dissEnt = additional_variables->mf_windfac * additional_variables->mf_dissEnt * dissfac;
#endif // if  defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION_HEAT)
    additional_variables->bpred_j[0] = SphP[j].BPred[0];
    additional_variables->bpred_j[1] = SphP[j].BPred[1];
    additional_variables->bpred_j[2] = SphP[j].BPred[2];
#ifdef SFR
    MyAtLeastDouble XColdCloudCorrection = pow(cbrt(1.0 - SphP[j].XColdCloud), 2);
    additional_variables->bpred_j[0] *= XColdCloudCorrection;
    additional_variables->bpred_j[1] *= XColdCloudCorrection;
    additional_variables->bpred_j[2] *= XColdCloudCorrection;
#endif // SFR
    kernel->dBx = local->BPred[0] - additional_variables->bpred_j[0];
    kernel->dBy = local->BPred[1] - additional_variables->bpred_j[1];
    kernel->dBz = local->BPred[2] - additional_variables->bpred_j[2];
    kernel->b2_j = additional_variables->bpred_j[0] * additional_variables->bpred_j[0] +
                   additional_variables->bpred_j[1] * additional_variables->bpred_j[1] +
                   additional_variables->bpred_j[2] * additional_variables->bpred_j[2];
#endif // MAGNETIC
#ifdef GT_DUST
#ifdef TIME_DEP_ART_VISC
    MyFloat R_ab = 0.5 * (fabs(local->visc_R) + fabs(SphP[j].visc_R));
    R_ab = min(R_ab, 1.);
    R_ab = max(R_ab, 0.);
    additional_variables->DeltaVelDissipationSwitch = 1.0 / (1. + exp(20 * (R_ab - 0.25)));
#endif // TIME_DEP_ART_VISC
    additional_variables->mean_dust_fraction =
        0.5 * (dustfrac(local->DustFracPred[0]) + dustfrac(SphP[j].DustFracPred[0]));

    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        kernel->vsig_DeltaVel[idust] = ((local->DeltaVelPred[idust][0] - SphP[j].DeltaVelPred[idust][0]) * kernel->dx +
                                        (local->DeltaVelPred[idust][1] - SphP[j].DeltaVelPred[idust][1]) * kernel->dy +
                                        (local->DeltaVelPred[idust][2] - SphP[j].DeltaVelPred[idust][2]) * kernel->dz) /
                                       kernel->r;

        kernel->DeltaVeldotr2[idust] =
            (dustfrac(local->DustFracPred[idust]) *
                 (local->DeltaVelPred[idust][0] - local->TotalDustFracPred * local->TotalDeltaVelPred[0]) -
             dustfrac(SphP[j].DustFracPred[idust]) *
                 (SphP[j].DeltaVelPred[idust][0] - SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[0])) *
                kernel->dx +
            (dustfrac(local->DustFracPred[idust]) *
                 (local->DeltaVelPred[idust][1] - local->TotalDustFracPred * local->TotalDeltaVelPred[1]) -
             dustfrac(SphP[j].DustFracPred[idust]) *
                 (SphP[j].DeltaVelPred[idust][1] - SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[1])) *
                kernel->dy +
            (dustfrac(local->DustFracPred[idust]) *
                 (local->DeltaVelPred[idust][2] - local->TotalDustFracPred * local->TotalDeltaVelPred[2]) -
             dustfrac(SphP[j].DustFracPred[idust]) *
                 (SphP[j].DeltaVelPred[idust][2] - SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[2])) *
                kernel->dz;
        kernel->DeltaVeldotr2[idust] /= kernel->r;
    }

    /*kernel->DeltaVeldotr2[0] = (kernel->dx * (local->DeltaVelPred[0][0] - SphP[j].DeltaVelPred[0][0]) +
                             kernel->dy * (local->DeltaVelPred[0][1] - SphP[j].DeltaVelPred[0][1]) +
                             kernel->dz * (local->DeltaVelPred[0][2] - SphP[j].DeltaVelPred[0][2])) /
                            kernel->r;*/

    kernel->vsig_dust = (local->Vel[0] + onem_dustfrac(local->DustFracPred[0]) * local->DeltaVelPred[0][0] -
                         SphP[j].VelPred[0] - onem_dustfrac(SphP[j].DustFracPred[0]) * SphP[j].DeltaVelPred[0][0]) *
                        kernel->dx / kernel->r;
    kernel->vsig_dust += (local->Vel[1] + onem_dustfrac(local->DustFracPred[0]) * local->DeltaVelPred[0][1] -
                          SphP[j].VelPred[1] - onem_dustfrac(SphP[j].DustFracPred[0]) * SphP[j].DeltaVelPred[0][1]) *
                         kernel->dy / kernel->r;
    kernel->vsig_dust += (local->Vel[2] + onem_dustfrac(local->DustFracPred[0]) * local->DeltaVelPred[0][2] -
                          SphP[j].VelPred[2] - onem_dustfrac(SphP[j].DustFracPred[0]) * SphP[j].DeltaVelPred[0][2]) *
                         kernel->dz / kernel->r;
#endif // GT_DUST
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Sets the distances and smoothing lengths between particle i and particle j with OpenACC support.
 *
 * This function computes and sets the distances and smoothing lengths between particle i and particle j.
 * It calculates the distance components (dx, dy, dz) and the squared distance (r2) between the positions of
 * the two particles. Additionally, it sets the smoothing length (h_j) of particle j.
 *
 * @param local Pointer to the local properties of particle i.
 * @param j Index of particle j in the simulation.
 * @param P Pointer to the gravity walk data structure.
 * @param kernel Pointer to the kernel properties used for interpolation.
 */
void set_kernel_distances_ij_openacc(struct hydrodata_in *local, int j, struct gravity_walkdata *P,
                                     struct kernel_hydra *kernel)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Sets the distances and smoothing lengths between particle i and particle j.
 *
 * This function computes and sets the distances and smoothing lengths between particle i and particle j.
 * It calculates the distance components (dx, dy, dz) and the squared distance (r2) between the positions of
 * the two particles. Additionally, it sets the smoothing length (h_j) of particle j.
 *
 * @param local Pointer to the local properties of particle i.
 * @param j Index of particle j in the simulation.
 * @param kernel Pointer to the kernel properties used for interpolation.
 */
static inline void set_kernel_distances_ij(struct hydrodata_in *local, int j, struct kernel_hydra *kernel)
#endif // ACC_INCLUDE_HYDRA
{
    kernel->dx = local->Pos[0] - P[j].Pos[0];
    kernel->dy = local->Pos[1] - P[j].Pos[1];
    kernel->dz = local->Pos[2] - P[j].Pos[2];

#ifdef PERIODIC /*  now find the closest image in the given box size  */
    kernel->dx = NEAREST_X(kernel->dx);
    kernel->dy = NEAREST_Y(kernel->dy);
    kernel->dz = NEAREST_Z(kernel->dz);
#endif // PERIODIC

#if GADGET_HYDRO == HYDRO_MFM
    kernel->dr[0] = -kernel->dx;
    if constexpr (NUMDIMS > 1)
    {
        kernel->dr[1] = -kernel->dy;
    }
    if constexpr (NUMDIMS == 3)
    {
        kernel->dr[2] = -kernel->dz;
    }
#endif // GADGET_HYDRO == HYDRO_MFM

#ifdef DISK_SLICE
    kernel->dv_phi_disk = 0;
    wrap_velocity_disk_slice(P[j].Pos, local->Pos, kernel->dv_phi_disk);
#endif // DISK_SLICE

    kernel->r2 = kernel->dx * kernel->dx + kernel->dy * kernel->dy + kernel->dz * kernel->dz;
#ifdef ACC_INCLUDE_HYDRA
    kernel->h_j = P[j].Hsml;
#else // ACC_INCLUDE_HYDRA
    kernel->h_j = P[j].Hsml;
#endif // ACC_INCLUDE_HYDRA
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifdef RT_RAD_PRESSURE
#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Calculates the radiation pressure acting on a particle.
 *
 * This function calculates the radiation pressure acting on a particle.
 * It computes the acceleration due to radiation pressure along each coordinate axis (x, y, z) for the particle.
 * The calculation involves summing contributions from different energy bins (n_gamma[j] * nu[j]) and scaling
 * by various constants to convert units appropriately. The resulting acceleration is stored in the RadAccel
 * array of the particle's hydro properties.
 *
 * @param i Index of the particle for which radiation pressure is calculated.
 */
static inline void calculate_rad_pressure(int i)
{
    MyAtLeastDouble dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;

    if (All.Time != All.TimeBegin)
    {
        for (int k = 0; k < 3; k++)
        {
            SphP[i].RadAccel[k] = 0.0;

            for (int j = 0; j < N_BINS; j++)
            {
                SphP[i].RadAccel[k] += SphP[i].n_gamma[j] * nu[j];
            }

            SphP[i].RadAccel[k] *= SphP[i].n[k] / P[i].Mass * ELECTRONVOLT_IN_ERGS / All.UnitEnergy_in_cgs *
                                   All.HubbleParam / (C_LIGHT / All.UnitVelocity_in_cm_per_s) / dt / SphP[i].Density;
        }
    }
}
#endif // ACC_INCLUDE_HYDRA
#endif // RT_RAD_PRESSURE

#ifdef SPH_BND_PARTICLES
#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Sets boundary particles to initial values.
 *
 * This function sets boundary particles to their initial values. Boundary particles are identified based on their ID,
 * and if the particle ID is 0, the function resets certain hydrodynamic and thermodynamic properties to zero.
 * Specifically, it sets the hydrodynamic acceleration, entropy change rate, viscous entropy change rate (if enabled),
 * and various other quantities to zero. If the hydrodynamic formulation is MFM, additional quantities related to
 * energy evolution and magnetic fields are also reset to zero.
 *
 * @param i Index of the particle to be set.
 */
void set_bnd_particles(int i)
{
    if (P[i].ID == 0)
    {
        for (int k = 0; k < 3; k++)
        {
            SphP[i].HydroAccel[k] = 0;
        }
        SphP[i].DtEntropy = 0;
#ifdef NAVIERSTOKES_TIMESTEP
        SphP[i].ViscEntropyChange = 0;
#endif // NAVIERSTOKES_TIMESTEP
#ifdef OUTPUT_ENERGY_VISC
        SphP[i].visc_E = 0.0;
#endif // OUTPUT_ENERGY_VISC
#if GADGET_HYDRO == HYDRO_MFM
        SphP[i].dQdt = (MyFloat)0.0;
        SphP[i].dQdt[mfm_numdims + 1] = eos->GetDtQFromDtEntropy(0.0, SphP[i].EntropyPred, SphP[i].InternalEnergyPred,
                                                                 P[i].Mass, 0.0 /*SphP[i].DivVel*/, All.cf_hubble_a);
        SphP[i].DtInternalEnergy =
            eos->GetDtUFromDtEntropy(0.0, SphP[i].EntropyPred, SphP[i].InternalEnergyPred, 0.0 /*SphP[i].DivVel*/,
                                     All.cf_hubble_a, All.ComovingIntegrationOn);
        SphP[i].DtEntropyExtra = 0.0;
        for (int k = 0; k < 3; k++)
        {
            SphP[i].HydroAccelExtra[k] = 0.0;
#ifdef MAGNETIC
            SphP[i].DtBExtra[k] = 0.0;
#endif // MAGNETIC
        }
        // also set Extra quanitities to zero
#endif // if GADGET_HYDRO == HYDRO_MFM
#ifdef GT_DUST
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            SphP[i].DtDustFrac[idust] = 0.0;
            for (int k = 0; k < 3; k++)
            {
                SphP[i].DtDeltaVel[idust][k] = 0.0;
            }
        }
#endif // GT_DUST
    }
}
#endif // ACC_INCLUDE_HYDRA
#endif // SPH_BND_PARTICLES
