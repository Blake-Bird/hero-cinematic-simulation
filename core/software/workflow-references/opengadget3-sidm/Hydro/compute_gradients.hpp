/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file is the header file for the computation of gradients.
 *
 * @author OpenGadget3-development team, in particular Ludwig Böss, Klaus Dolag, Frederick Groth and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "../CodeBase/allvars.hpp"
#if !(defined ACC_INCLUDE_GRADIENTS) && defined(ACC_GRADIENTS) // we reinclude this same file if ACC is active
#include "../OpenACC/gpuallvars_branch.hpp"
#endif // !(defined ACC_INCLUDE_GRADIENTS) && defined(ACC_GRADIENTS)

#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
#ifdef ACC_INCLUDE_GRADIENTS
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Evaluates the contribution of turbulent velocity to the velocity dispersion of a particle with OpenACC
 * support.
 *
 * This function computes the contribution of turbulent velocity to the velocity dispersion of a particle
 * by comparing the velocity of the particle with the bulk velocity of its neighboring particles. It updates
 * the root-mean-square (RMS) velocity dispersion of the particle.
 *
 * @param local Pointer to the local hydrodynamic data of the particle.
 * @param kernel Pointer to the hydrodynamic kernel data used for smoothing.
 * @param j Index of the neighboring particle.
 * @param P Pointer to the gravity walk data structure.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particles.
 * @param out Pointer to the output hydrodynamic data structure for storing the computed results.
 */
void turbulent_velocity_evaluate_openacc(struct graddata_in *local, struct kernel_density *kernel, int j,
                                         struct graddata_in *SphP, struct graddata_out *out)
#else // ACC_INCLUDE_GRADIENTS

/**
 * @brief Evaluates the contribution of turbulent velocity to the velocity dispersion of a particle.
 *
 * This function computes the contribution of turbulent velocity to the velocity dispersion of a particle
 * by comparing the velocity of the particle with the bulk velocity of its neighboring particles. It updates
 * the root-mean-square (RMS) velocity dispersion of the particle.
 *
 * @param local Pointer to the local hydrodynamic data of the particle.
 * @param kernel Pointer to the hydrodynamic kernel data used for smoothing.
 * @param j Index of the neighboring particle.
 * @param out Pointer to the output hydrodynamic data structure for storing the computed results.
 */
static inline void turbulent_velocity_evaluate(struct graddata_in *local, struct kernel_density *kernel, int j,
                                               struct graddata_out *out)
#endif // ACC_INCLUDE_GRADIENTS
{
    out->Vrms += (SphP[j].VelPred[0] - local->Vbulk[0]) * (SphP[j].VelPred[0] - local->Vbulk[0]) +
                 (SphP[j].VelPred[1] - local->Vbulk[1]) * (SphP[j].VelPred[1] - local->Vbulk[1]) +
                 (SphP[j].VelPred[2] - local->Vbulk[2]) * (SphP[j].VelPred[2] - local->Vbulk[2]);

#if defined(JD_DECOMPOSE_VTURB)
    MyFloat vturb[3], e_rad[3], l_rad, vrad[3], vtan[3];

    /* remove laminar velocity of j's velocity */
    vturb[0] = SphP[j].VelPred[0] - local->Vbulk[0];
    vturb[1] = SphP[j].VelPred[1] - local->Vbulk[1];
    vturb[2] = SphP[j].VelPred[2] - local->Vbulk[2];

    /* unit vector from i to j */
    e_rad[0] = kernel->dx / kernel->r;
    e_rad[1] = kernel->dy / kernel->r;
    e_rad[2] = kernel->dz / kernel->r;

    /* project vturb onto radial unit vector */
    l_rad = vturb[0] * e_rad[0] + vturb[1] * e_rad[1] + vturb[2] * e_rad[2];

    /* radial component of vturb */
    vrad[0] = e_rad[0] * l_rad;
    vrad[1] = e_rad[1] * l_rad;
    vrad[2] = e_rad[2] * l_rad;

    /* tangential component of vturb */
    vtan[0] = vturb[0] - vrad[0];
    vtan[1] = vturb[1] - vrad[1];
    vtan[2] = vturb[2] - vrad[2];

    /* We need dispersion in the end */
    out->Vrad = l_rad;
    out->Vtan = vtan[0] * vtan[0] + vtan[1] * vtan[1] + vtan[2] * vtan[2];
#endif // defined(JD_DECOMPOSE_VTURB)
}
#ifdef ACC_INCLUDE_GRADIENTS
ACC_DECLARE_ROUTINE_AFTER
#endif

#ifndef ACC_INCLUDE_GRADIENTS
/**
 * @brief Calculates the turbulent velocity dispersion of a particle.
 *
 * This function computes the turbulent velocity dispersion of a particle based on its predicted velocity
 * and bulk velocity. It updates the root-mean-square (RMS) velocity dispersion of the particle. Optionally,
 * if the JD_DECOMPOSE_VTURB flag is defined, it decomposes the turbulent velocity into radial and tangential
 * components and computes their respective RMS values.
 *
 * @param i Index of the particle for which turbulent velocity is calculated.
 */
static inline void calculate_turbulent_velocity(int i)
{
    if (SphP[i].Density > 0)
    {
        /* do the root mean of the added squares */
        SphP[i].Vrms = sqrt((MyAtLeastDouble)SphP[i].Vrms / P[i].TrueNGB);
#if defined(JD_DECOMPOSE_VTURB)
        SphP[i].Vtan = sqrt((MyAtLeastDouble)SphP[i].Vtan / P[i].TrueNGB);
        SphP[i].Vrad = sqrt((MyAtLeastDouble)SphP[i].Vrad / P[i].TrueNGB);
#endif // defined(JD_DECOMPOSE_VTURB)
    }
    else
    {
        SphP[i].Vrms = 0;
#if defined(JD_DECOMPOSE_VTURB)
        SphP[i].Vtan = sqrt((MyAtLeastDouble)SphP[i].Vtan / P[i].TrueNGB);
        SphP[i].Vrad = sqrt((MyAtLeastDouble)SphP[i].Vrad / P[i].TrueNGB);
#endif // defined(JD_DECOMPOSE_VTURB)
    }
}
#endif // ACC_INCLUDE_GRADIENTS
#endif // defined(JD_VTURB)  || defined(KD_LIMIT_DIVV)

#ifndef ACC_INCLUDE_GRADIENTS
/**
 * @brief Calculates the rotation given a Matrix of derivatives.
 *
 * This function computes the rotation of a field given the Matrix of derivatives, which is expressed as Array of
 * Vectors.
 *
 * @param dMatrix Array of Vectors that contains all spatial derivatives, passed as reference.
 * @param Rot Array containing the resulting rotation.
 */
template <typename T> void calculate_rotation(const Vector<NUMDIMS, MyAtLeastDouble> (&dMatrix)[3], T (&Rot)[3])
{
#if NUMDIMS == 1
    Rot[0] = 0.0;
#elif NUMDIMS == 2
    Rot[0] = -dMatrix[2][1];
#elif NUMDIMS == 3
    Rot[0] = dMatrix[1][2] - dMatrix[2][1];
#endif // NUMDIMS
#if NUMDIMS < 3
    Rot[1] = dMatrix[2][0];
#else // NUMDIMS < 3
    Rot[1] = dMatrix[2][0] - dMatrix[0][2];
#endif // NUMDIMS < 3
#if NUMDIMS == 1
    Rot[2] = -dMatrix[1][0];
#else // NUMDIMS == 1;
    Rot[2] = dMatrix[0][1] - dMatrix[1][0];
#endif // NUMDIMS == 1
}
#endif // ACC_INCLUDE_GRADIENTS
