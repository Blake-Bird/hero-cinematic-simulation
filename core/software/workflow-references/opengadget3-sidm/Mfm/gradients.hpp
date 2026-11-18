/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares central functions to calculate the MFM gradients.
 *
 * This file declares functions to compute the MFM gradients inside the main
 * gradient loop.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
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

#ifndef ACC_INCLUDE_GRADIENTS
/**
 * @brief Computes the second order accurate MFM gradients from the previously evaluated Matrix.
 *
 * This function performs the final step of the MFM gradient calculation.
 *
 * @param i Index of the particle.
 * @param GradStoreOut pointer to (array of) structures that saves some local output variables used only in the gradient
 * loop.
 */
static inline void mfm_gradients_final(int i, grad_store_out *GradStoreOut)
{
#ifdef WINDS
    if (SphP[i].DelayTime <= 0)
#endif // WINDS
    {
        MyAtLeastDouble sqdConditionNumber;
        MyAtLeastDouble det_tmp = SphP[i].Bgrad.GetDeterminant();
        if (std::isnormal(det_tmp) and abs(det_tmp) > small_float)
        {
            // Matrix has to be inverted to calculate the gradients.
            // Groth+2023 Eq. (29)
            sqdConditionNumber = SphP[i].Bgrad.GetConditionNumberAndInvertMatrix(det_tmp);
            // Check the accuracy of the integral gradients (using the square of the condition number)
        }
        else
        {
            // Matrix is not invertable as it is.
            // ensure the matrix is invertable
            MyAtLeastDouble correction = SphP[i].Bgrad.GetTrace();
            correction = ((correction >= 0) - (correction < 0)) /*as sign, but returning 1 for 0*/ *
                         std::max(1.05 * fabs(correction) / (NUMDIMS * 1.0e4), MyAtLeastDouble(1e-15));

            while (true)
            {
                for (int k = 0; k < NUMDIMS; k++)
                {
                    SphP[i].Bgrad(k, k) += correction;
                }
                det_tmp = SphP[i].Bgrad.GetDeterminant();
                if (std::isnormal(det_tmp) and abs(det_tmp) >= 1.0e-30)
                {
                    break;
                }
                correction *= 1.2;
            }
            // now invert the corrected matrix
            SphP[i].Bgrad.InvertMatrix(det_tmp);

            sqdConditionNumber = 1e30; // some high number, as we in principle don't want to use it.
        }
        SphP[i].ConditionNumber = sqrt(sqdConditionNumber);

        if (sqdConditionNumber < 1.0e4)
        {
            // If integral gradients are ok, then continue to use to compute gradients
            // compare Groth+2023 Eq. (35)
            for (int var = 0; var < numvar; var++)
            {
                SphP[i].gradW[var] = SphP[i].Bgrad * SphP[i].gradW[var];
#ifdef GT_DUST
                for (int idust = 0; idust < GT_NUM_DUST; idust++)
                {
                    SphP[i].gradW_dust[idust][var] = SphP[i].Bgrad * SphP[i].gradW_dust[idust][var];
                }
#endif // GT_DUST
            }
#ifdef OUTPUT_GRADIENT_ORDER
            SphP[i].GradientOrder = 2.0;
#endif // OUTPUT_GRADIENT_ORDER
        }
        else
        {
            // Otherwise, use less accurate but more stable SPH-like gradients
            for (int var = 0; var < numvar; var++)
            {
                SphP[i].gradW[var] = GradStoreOut[i].gradWSph[var];
#ifdef GT_DUST
                for (int idust = 0; idust < GT_NUM_DUST; idust++)
                {
                    SphP[i].gradW_dust[idust][var] = GradStoreOut[i].gradWSph_dust[idust][var];
                }
#endif // GT_DUST
            }
#ifdef OUTPUT_GRADIENT_ORDER
            SphP[i].GradientOrder = 1.0;
#endif // OUTPUT_GRADIENT_ORDER
        }
        for (int k = 0; k < numvar; k++)
        { // avoid nan values
            for (int kk = 0; kk < NUMDIMS; kk++)
            {
                if (!std::isnormal(SphP[i].gradW[k][kk]))
                {
                    SphP[i].gradW[k][kk] = (MyFloat)0.0;
                }
#ifdef GT_DUST
                for (int idust = 0; idust < GT_NUM_DUST; idust++)
                {
                    if (!std::isnormal(SphP[i].gradW_dust[idust][k][kk]))
                    {
                        SphP[i].gradW_dust[idust][k][kk] = (MyFloat)0.0;
                    }
                }
#endif // GT_DUST
            }
        }

        // normalize the smoothed velocity by mass
        SphP[i].Vel_smooth /= GradStoreOut[i].neighbour_mass;
    }
}
#endif // ACC_INCLUDE_GRADIENTS

#ifdef ACC_INCLUDE_GRADIENTS
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Evaluates the MFM gradient matrix with OpenACC support.
 *
 * @param local Reference to the local hydrodynamic data structure.
 * @param kernel Reference to the kernel structure containing smoothing lengths and derivatives.
 * @param j Index of the neighboring particle.
 * @param P Pointer to the gravity walk data structure.
 * @param SphP Pointer to the hydrodynamic data structure of the neighboring particles.
 * @param out Reference to the output hydrodynamic data structure containing the computed gradients.
 */
void mfm_gradients_evaluate_openacc(struct graddata_in &local, struct kernel_density &kernel, int j,
                                    struct gravity_walkdata *P, struct graddata_in *SphP, struct graddata_out &out)
#else // ACC_INCLUDE_GRADIENTS
/**
 * @brief Evaluates the MFM gradient matrix.
 *
 * @param local Reference to the local hydrodynamic data structure.
 * @param kernel Reference to the kernel structure containing smoothing lengths and derivatives.
 * @param j Index of the neighboring particle.
 * @param out Pointer to the output hydrodynamic data structure containing the computed gradients.
 */
static inline void mfm_gradients_evaluate(struct graddata_in &local, struct kernel_density &kernel, int j,
                                          struct graddata_out &out)
#endif // ACC_INCLUDE_GRADIENTS
{
    MyAtLeastDouble c_mfm = 1.0;
#ifdef WINDS
    if ((SphP[j].DelayTime > 0) || (local.DelayTime > 0))
    { /* decouple wind particles */
        c_mfm = 0.0;
    }
#endif // WINDS

    Vector<mfm_numdims> dr;
    dr[0] = -kernel.dx;
    if constexpr (mfm_numdims > 1)
    {
        dr[1] = -kernel.dy;
    }
    if constexpr (mfm_numdims == 3)
    {
        dr[2] = -kernel.dz;
    }
    MyAtLeastDouble wk = kernel.wk * local.Volume;
    MyAtLeastDouble dwk = kernel.dwk * local.Volume; // used for SPH-like gradients
    // calculate the gradient matrix for second order gradients.
    // compare Groth+2023 Eq. (30)
    for (int k = 0; k < NUMDIMS; k++)
    {
        // we use a SymmetricMatrix, such that only half of the matrix has to be filled with values.
        for (int kk = 0; kk <= k; kk++)
        {
            out.Ematrix(k, kk) += c_mfm * dr[k] * dr[kk] * wk;
        }
    }

    // prepare for the second order gradient calculation
    // compare Groth+2023 Eq. (35)
    for (int var = 0; var < numvar; var++)
    {
        const MyAtLeastDouble dW_wk = c_mfm * (kernel.Wj[var] - local.Wprim[var]) * wk;
#ifdef GT_DUST
        MyAtLeastDouble dW_dust_wk[GT_NUM_DUST];
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            dW_dust_wk[idust] = c_mfm * (SphP[j].Wprim_dust[idust][var] - local.Wprim_dust[idust][var]) * wk;
        }
#endif // GT_DUST
        for (int k = 0; k < NUMDIMS; k++)
        {
            out.gradW[var][k] += dr[k] * dW_wk;
#ifdef GT_DUST
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                out.gradW_dust[idust][var][k] += dr[k] * dW_dust_wk[idust];
            }
#endif // GT_DUST
        }
    }

    // Special slope limiter terms
#if MFM_SLOPE_LIMITER == LIMITER_AREPO || MFM_SLOPE_LIMITER == LIMITER_SCALAR || MFM_SLOPE_LIMITER == LIMITER_GIZMO
#ifdef WINDS
    if ((SphP[j].DelayTime <= 0) && (local.DelayTime <= 0))
#endif // WINDS
        for (int var = 0; var < numvar; var++)
        {
            out.Wmin[var] = std::min(out.Wmin[var], kernel.Wj[var]);
            out.Wmax[var] = std::max(out.Wmax[var], kernel.Wj[var]);
#ifdef GT_DUST
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                out.Wmin_dust[idust][var] = std::min(out.Wmin_dust[idust][var], SphP[j].Wprim_dust[idust][var]);
                out.Wmax_dust[idust][var] = std::max(out.Wmax_dust[idust][var], SphP[j].Wprim_dust[idust][var]);
            }
#endif // GT_DUST
        }
#endif // MFM_SLOPE_LIMITER == LIMITER_AREPO || MFM_SLOPE_LIMITER == LIMITER_SCALAR || MFM_SLOPE_LIMITER ==
       // LIMITER_GIZMO
#if MFM_SLOPE_LIMITER == LIMITER_SCALAR || MFM_SLOPE_LIMITER == LIMITER_GIZMO
#ifdef WINDS
    if ((SphP[j].DelayTime <= 0) && (local.DelayTime <= 0))
#endif // WINDS
    {
        out.DistNgbSqdMax = std::max(out.DistNgbSqdMax, MyFloat(kernel.r2));
    }
#endif // MFM_SLOPE_LIMITER == LIMITER_SCALAR || MFM_SLOPE_LIMITER == LIMITER_GIZMO
    // Velocity smoothed over the neighbors used in the MUSCL scheme
    for (int k = 0; k < mfm_numdims; k++)
    {
        out.Vel_smooth[k] += c_mfm * P[j].Mass * kernel.Wj.GetVelocity()[k];
    }
    // the velocity is weighted by mass
    out.neighbour_mass += c_mfm * P[j].Mass;

    if (kernel.r > 0)
    {
        const MyAtLeastDouble invr = 1.0 / kernel.r;
        // first order SPH like gradient
        for (int var = 0; var < numvar; var++)
        {
            const MyAtLeastDouble dW_dwkr = c_mfm * (kernel.Wj[var] - local.Wprim[var]) * dwk * invr;
#ifdef GT_DUST
            MyAtLeastDouble dW_dust_dwkr[GT_NUM_DUST];
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                dW_dust_dwkr[idust] =
                    c_mfm * (SphP[j].Wprim_dust[idust][var] - local.Wprim_dust[idust][var]) * dwk * invr;
            }
#endif // GT_DUST
            for (int k = 0; k < NUMDIMS; k++)
            {
                out.gradWSph[var][k] -= dr[k] * dW_dwkr;
#ifdef GT_DUST
                for (int idust = 0; idust < GT_NUM_DUST; idust++)
                {
                    out.gradWSph_dust[idust][var][k] -= dr[k] * dW_dust_dwkr[idust];
                }
#endif // GT_DUST
            }
        }

        Vector<mfm_numdims> dv;
        for (int k = 0; k < mfm_numdims; k++)
        {
            dv[k] = c_mfm * (kernel.Wj.GetVelocity()[k] - local.Wprim.GetVelocity()[k]);
        }
#ifdef WINDS
        if ((SphP[j].DelayTime <= 0) && (local.DelayTime <= 0))
#endif // WINDS
        {
            out.MaxVelSquareNgb = std::max(out.MaxVelSquareNgb, dv.GetMagnitudeSquared());
        }

    } // if(kernel.r > 0)
}
#ifdef ACC_INCLUDE_GRADIENTS
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_GRADIENTS
