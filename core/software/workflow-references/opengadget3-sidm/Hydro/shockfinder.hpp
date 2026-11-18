/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file for the shock finder.
 *
 * @author OpenGadget3-development team, in particular Ludwig Böss and Klaus Dolag.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef SHOCKFINDER_HPP
// #define SHOCKFINDER_HPP

#include "../CodeBase/allvars.hpp"

/* Beck et al. 2016 SPH-style shock finder: kernel-projected pre/post states
 * accumulated during the hydro flux walk. Gated by SHOCKFINDER_SPH, derived from
 * SHOCKFINDER under HYDRO_SPH / HYDRO_PESPH (see CodeBase/switches.hpp). The MFM
 * counterpart is Mfm/mfm_shockfinder.cpp, gated by SHOCKFINDER_MFM. */
#ifdef SHOCKFINDER_SPH
#ifdef ACC_INCLUDE_HYDRA
/**
 * @brief Calculates upstream and downstream quantities related to shock finding using OpenACC.
 *
 * This function computes the upstream and downstream quantities relevant for shock finding based on the provided local
 * information, kernel properties, and particle data. It projects the kernel extent onto the shock normal and evaluates
 * various quantities such as mass-weighted velocities, densities, pressures, and sound speeds for particles located
 * upstream and downstream of the shock. Additionally, it calculates Alfven velocities and shock obliquities if enabled
 * by compilation flags.
 *
 * @param local Pointer to the local hydrodynamic data of the target particle.
 * @param out Pointer to the output structure where the computed quantities will be stored.
 * @param kernel Pointer to the kernel properties used for smoothing.
 * @param j Index of the particle for which upstream and downstream quantities are computed.
 * @param P Array of gravity walk data for particles.
 * @param SphP Array of hydrodynamic data for particles.
 */
void shockfinder_calculate_up_and_downstream_quantities_openacc(struct hydrodata_in *local, struct hydrodata_out *out,
                                                                struct kernel_hydra *kernel, int j,
                                                                struct gravity_walkdata *P, struct hydrodata_in *SphP)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Calculates upstream and downstream quantities related to shock finding.
 *
 * This function computes the upstream and downstream quantities relevant for shock finding based on the provided local
 * information, kernel properties, and particle data. It projects the kernel extent onto the shock normal and evaluates
 * various quantities such as mass-weighted velocities, densities, pressures, and sound speeds for particles located
 * upstream and downstream of the shock. Additionally, it calculates Alfven velocities and shock obliquities if enabled
 * by compilation flags.
 *
 * @param local Pointer to the local hydrodynamic data of the target particle.
 * @param out Pointer to the output structure where the computed quantities will be stored.
 * @param kernel Pointer to the kernel properties used for smoothing.
 * @param j Index of the particle for which upstream and downstream quantities are computed.
 */
static inline void shockfinder_calculate_up_and_downstream_quantities(struct hydrodata_in *local,
                                                                      struct hydrodata_out *out,
                                                                      struct kernel_hydra *kernel, int j)
#endif // ACC_INCLUDE_HYDRA
{
    // project kernel extent onto shock normal
    MyAtLeastDouble Shock_X1_r =
        kernel->dx * local->Shock_N1[0] + kernel->dy * local->Shock_N1[1] + kernel->dz * local->Shock_N1[2];

    MyAtLeastDouble hinv, hinv3, hinv4; // inverse hsml^1, ^3, ^4
    kernel_hinv(local->Hsml, &hinv, &hinv3, &hinv4);

    MyAtLeastDouble u = abs(1.0 - abs(Shock_X1_r * hinv));
    if (u < 1.0)
    {
        MyAtLeastDouble wk_i, dwk_i;
        kernel_main(u, hinv3, hinv4, &wk_i, &dwk_i, -1);
        // mass of particle j times inverse kernel weight
        MyAtLeastDouble mj_wki = P[j].Mass * wk_i;

        // inverted kernel weight
        u = sqrt(kernel->r2 - Shock_X1_r * Shock_X1_r) * hinv;
        kernel_main(u, hinv3, hinv4, &wk_i, &dwk_i, -1);
        mj_wki *= P[j].Mass * wk_i;

#ifdef MAGNETIC
        // Alfven velocity squared, in the comoving convention of the kernel-averaged signal
        // speed and of Shock_Speed: B is stored comoving (B_code = a^2 B_phys) over a comoving
        // density, so BPred^2*mu0_inv/Density comes out as a*v_alfven_phys^2 and needs one
        // more factor of a to enter the Alfvenic Mach number.
        MyAtLeastDouble v_alfven2 = (SphP[j].BPred[0] * SphP[j].BPred[0] + SphP[j].BPred[1] * SphP[j].BPred[1] +
                                     SphP[j].BPred[2] * SphP[j].BPred[2]) *
                                    mu0_inv / SphP[j].Density * All.cf_atime;
#endif // MAGNETIC

        // upstream quantities
        if (Shock_X1_r < 0)
        {
            out->Shock_Up_V1[0] += mj_wki * SphP[j].VelPred[0];
            out->Shock_Up_V1[1] += mj_wki * SphP[j].VelPred[1];
            out->Shock_Up_V1[2] += mj_wki * SphP[j].VelPred[2];
            out->Shock_Up_Rho += mj_wki * SphP[j].Density;
            out->Shock_Up_Weight += mj_wki;
            out->Shock_Up_Pressure += mj_wki * SphP[j].Pressure;

            // sound speed squared
            MyAtLeastDouble shock_up_signal2 = GAMMA * SphP[j].Pressure / SphP[j].Density;

            // assign contribution of particle j to local sound speed
            out->Shock_Up_Signal += mj_wki * sqrt(shock_up_signal2);

#ifdef MAGNETIC
            // Upstream Alfven speed needed to get Alfven Mach number
            out->Shock_Up_Alfven += mj_wki * sqrt(v_alfven2);

            // compute contribution of particle j to local shock obliquity via the cosine similarity
            out->Shock_Obliquity +=
                mj_wki * acos(abs((SphP[j].BPred[0] * local->Shock_N1[0] + SphP[j].BPred[1] * local->Shock_N1[1] +
                                   SphP[j].BPred[2] * local->Shock_N1[2]) /
                                  sqrt(SphP[j].BPred[0] * SphP[j].BPred[0] + SphP[j].BPred[1] * SphP[j].BPred[1] +
                                       SphP[j].BPred[2] * SphP[j].BPred[2])));
#endif // MAGNETIC
        }
        // downstream quantities
        else
        {
            out->Shock_Down_V1[0] += mj_wki * SphP[j].VelPred[0];
            out->Shock_Down_V1[1] += mj_wki * SphP[j].VelPred[1];
            out->Shock_Down_V1[2] += mj_wki * SphP[j].VelPred[2];
            out->Shock_Down_Rho += mj_wki * SphP[j].Density;
            out->Shock_Down_Weight += mj_wki;
            out->Shock_Down_Pressure += mj_wki * SphP[j].Pressure;


        }
    }

#ifndef SHOCKFINDER_SIMPLE
    // Perpendicular to shock normal
    MyAtLeastDouble Shock_X2_r =
        kernel->dx * local->Shock_N2[0] + kernel->dy * local->Shock_N2[1] + kernel->dz * local->Shock_N2[2];

    u = abs(1.0 - abs(Shock_X2_r * hinv));
    if (u < 1.0)
    {
        MyAtLeastDouble wk_i, dwk_i;
        kernel_main(u, hinv3, hinv4, &wk_i, &dwk_i, -1);
        MyAtLeastDouble mj_wki = P[j].Mass * wk_i;

        u = sqrt(kernel->r2 - Shock_X2_r * Shock_X2_r) * hinv;
        kernel_main(u, hinv3, hinv4, &wk_i, &dwk_i, -1);
        mj_wki *= P[j].Mass * wk_i;

        if (Shock_X2_r < 0)
        {
            out->Shock_Up_V2[0] += mj_wki * SphP[j].VelPred[0];
            out->Shock_Up_V2[1] += mj_wki * SphP[j].VelPred[1];
            out->Shock_Up_V2[2] += mj_wki * SphP[j].VelPred[2];
            out->Shock_Up_Weight23[0] += mj_wki;
        }
        else
        {
            out->Shock_Down_V2[0] += mj_wki * SphP[j].VelPred[0];
            out->Shock_Down_V2[1] += mj_wki * SphP[j].VelPred[1];
            out->Shock_Down_V2[2] += mj_wki * SphP[j].VelPred[2];
            out->Shock_Down_Weight23[0] += mj_wki;
        }
    }

    /* Perpendicular to shock normal */
    MyAtLeastDouble Shock_X3_r =
        kernel->dx * local->Shock_N3[0] + kernel->dy * local->Shock_N3[1] + kernel->dz * local->Shock_N3[2];

    u = abs(1.0 - abs(Shock_X3_r * hinv));
    if (u < 1.0)
    {
        MyAtLeastDouble wk_i, dwk_i;
        kernel_main(u, hinv3, hinv4, &wk_i, &dwk_i, -1);
        MyAtLeastDouble mj_wki = P[j].Mass * wk_i;

        u = sqrt(kernel->r2 - Shock_X3_r * Shock_X3_r) * hinv;
        kernel_main(u, hinv3, hinv4, &wk_i, &dwk_i, -1);
        mj_wki *= P[j].Mass * wk_i;

        if (Shock_X3_r < 0)
        {
            out->Shock_Up_V3[0] += mj_wki * SphP[j].VelPred[0];
            out->Shock_Up_V3[1] += mj_wki * SphP[j].VelPred[1];
            out->Shock_Up_V3[2] += mj_wki * SphP[j].VelPred[2];
            out->Shock_Up_Weight23[1] += mj_wki;
        }
        else
        {
            out->Shock_Down_V3[0] += mj_wki * SphP[j].VelPred[0];
            out->Shock_Down_V3[1] += mj_wki * SphP[j].VelPred[1];
            out->Shock_Down_V3[2] += mj_wki * SphP[j].VelPred[2];
            out->Shock_Down_Weight23[1] += mj_wki;
        }
    }
#endif // SHOCKFINDER_SIMPLE
}

#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief This function initializes the shock properties of the particle.
 * This function applies the weighting of the contributing particles in the kernel,
 * applies filtering to avoid finding "shocks" in shear flows and computes the shock properties
 * from the Rankine-Hugoniot jump conditions.
 */
static inline void shockfinder_calculate_shock_properties(int i)
{
    // apply weights
    SphP[i].Shock_Up_Rho /= SphP[i].Shock_Up_Weight;
    SphP[i].Shock_Down_Rho /= SphP[i].Shock_Down_Weight;
    SphP[i].Shock_Up_Pressure /= SphP[i].Shock_Up_Weight;
    SphP[i].Shock_Down_Pressure /= SphP[i].Shock_Down_Weight;
    SphP[i].Shock_Up_Signal /= SphP[i].Shock_Up_Weight;


#ifdef MAGNETIC
    SphP[i].Shock_Obliquity /= SphP[i].Shock_Up_Weight;
    SphP[i].Shock_Up_Alfven /= SphP[i].Shock_Up_Weight;
#endif // MAGNETIC

    // upstream velocity projected onto the shock normal
    MyAtLeastDouble Shock_Up_V1 =
        (SphP[i].Shock_Up_V1[0] * SphP[i].Shock_N1[0] + SphP[i].Shock_Up_V1[1] * SphP[i].Shock_N1[1] +
         SphP[i].Shock_Up_V1[2] * SphP[i].Shock_N1[2]) /
        SphP[i].Shock_Up_Weight;

    // downstream velocity projected onto the shock normal
    MyAtLeastDouble Shock_Down_V1 =
        (SphP[i].Shock_Down_V1[0] * SphP[i].Shock_N1[0] + SphP[i].Shock_Down_V1[1] * SphP[i].Shock_N1[1] +
         SphP[i].Shock_Down_V1[2] * SphP[i].Shock_N1[2]) /
        SphP[i].Shock_Down_Weight;

    // up-/downstream velocity jump within the kernel
    MyAtLeastDouble Shock_Veldiv1 = Shock_Down_V1 - Shock_Up_V1;

    // initialize properties as 0
    SphP[i].Shock_Mach = 0.0;
    SphP[i].Shock_Speed = 0.0;
    SphP[i].Shock_Compress = 1.0;

    // check if velocity jump is positive along the shock normal
    if (Shock_Veldiv1 > 0)
    {
#ifndef SHOCKFINDER_SIMPLE
        MyAtLeastDouble Shock_Up_V2 =
            (SphP[i].Shock_Up_V2[0] * SphP[i].Shock_N2[0] + SphP[i].Shock_Up_V2[1] * SphP[i].Shock_N2[1] +
             SphP[i].Shock_Up_V2[2] * SphP[i].Shock_N2[2]) /
            SphP[i].Shock_Up_Weight23[0];

        MyAtLeastDouble Shock_Down_V2 =
            (SphP[i].Shock_Down_V2[0] * SphP[i].Shock_N2[0] + SphP[i].Shock_Down_V2[1] * SphP[i].Shock_N2[1] +
             SphP[i].Shock_Down_V2[2] * SphP[i].Shock_N2[2]) /
            SphP[i].Shock_Down_Weight23[0];

        MyAtLeastDouble Shock_Up_V3 =
            (SphP[i].Shock_Up_V3[0] * SphP[i].Shock_N3[0] + SphP[i].Shock_Up_V3[1] * SphP[i].Shock_N3[1] +
             SphP[i].Shock_Up_V3[2] * SphP[i].Shock_N3[2]) /
            SphP[i].Shock_Up_Weight23[1];

        MyAtLeastDouble Shock_Down_V3 =
            (SphP[i].Shock_Down_V3[0] * SphP[i].Shock_N3[0] + SphP[i].Shock_Down_V3[1] * SphP[i].Shock_N3[1] +
             SphP[i].Shock_Down_V3[2] * SphP[i].Shock_N3[2]) /
            SphP[i].Shock_Down_Weight23[1];

        // shock speed perpendicular to shock normal
        MyAtLeastDouble Shock_Veldiv2 = abs(Shock_Down_V2 - Shock_Up_V2);
        MyAtLeastDouble Shock_Veldiv3 = abs(Shock_Down_V3 - Shock_Up_V3);

        if (Shock_Veldiv1 > 2.0 * Shock_Veldiv2)
        {
            if (Shock_Veldiv1 > 2.0 * Shock_Veldiv3)
            {
#endif // SHOCKFINDER_SIMPLE
                if (SphP[i].Shock_Down_Pressure > 1.05 * SphP[i].Shock_Up_Pressure)
                {
                    if (Shock_Veldiv1 > 0.025 * (Shock_Down_V1 + Shock_Up_V1))
                    {
                        SphP[i].Shock_Compress = SphP[i].Shock_Down_Rho / SphP[i].Shock_Up_Rho;

                        if (SphP[i].Shock_Compress < 1.0)
                        {
                            SphP[i].Shock_Compress = 1.0;
                        }

                        if (SphP[i].Shock_Compress > 1.05)
                        {
                            MyAtLeastDouble Shock_Compress = 1.0 - 1.0 / SphP[i].Shock_Compress;
                            if (SphP[i].DivVel < 0)
                            {
                                MyAtLeastDouble shock_f =
                                    abs(SphP[i].DivVel) / (abs(SphP[i].DivVel) + abs(SphP[i].CurlVel));
                                if (shock_f > 0.9)
                                {
                                    SphP[i].Shock_Speed = Shock_Veldiv1 / Shock_Compress;
                                    SphP[i].Shock_Mach = SphP[i].Shock_Speed / SphP[i].Shock_Up_Signal;
#ifdef MAGNETIC
                                    // Computes the Alfvén Mach number as: M_A = v_s / v_A
                                    SphP[i].Shock_Mach_Alfven = SphP[i].Shock_Speed / SphP[i].Shock_Up_Alfven;

#endif // MAGNETIC
                                } // if (shock_f > 0.9)
                            } // if (SphP[i].DivVel < 0)
                        } // if (SphP[i].Shock_Compress > 1.05)
                    } // if (Shock_Veldiv1 > 0.025 * (Shock_Down_V1 + Shock_Up_V1))
                } // if (SphP[i].Shock_Down_Pressure > 1.05 * SphP[i].Shock_Up_Pressure)
#ifndef SHOCKFINDER_SIMPLE
            } // if (Shock_Veldiv1 > 2.0 * Shock_Veldiv3)
        } // if (Shock_Veldiv1 > 2.0 * Shock_Veldiv2)
#endif // SHOCKFINDER_SIMPLE
    } // if (Shock_Veldiv1 > 0)
}
#endif // ACC_INCLUDE_HYDRA

#endif // SHOCKFINDER_SPH

#endif // SHOCKFINDER_HPP
