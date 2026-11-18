/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief SPH density computation and smoothing length determination.
 *
 * @author OpenGadget3-development team, in particular Ludwig Böss, Klaus Dolag, Frederick Groth and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#include "../System/communication.hpp"
#include "kernel.hpp"

#ifndef ACC_INCLUDE_HSML // we reinclude this same file if ACC is active
// int DoSecondaryOnGPU;

#define LOCK_NEXPORT
#define UNLOCK_NEXPORT

// #define HSML_TOLERANCE 0.001
// #define HSML_TOLERANCE 0.0001
// #define HSML_TOLERANCE 0.00001
// #define HSML_TOLERANCE 0.000001

constexpr double HSML_TOLERANCE = 0.00001;

#ifndef HSML_DEBUG
#define HSML_DEBUG -1
#endif // HSML_DEBUG
#define ITER_DEBUG 50

#ifdef KD_RESTRICT_NEIGHBOURS
constexpr double REDUCTION_LOW_BOUND = 0.5;
constexpr double INCREASING_HIGH_BOUND = 10.0;
#endif // KD_RESTRICT_NEIGHBOURS

/*
   0 = Midpoint
   1 = linear in x,y
   2 = monotonic (from Antonio)
   3 = cubic in x
   4 = cubic in y (original)
   5 = powerlaw
*/

#ifndef NGB_INTERPOLATOR
#define NGB_INTERPOLATOR 5
#endif // NGB_INTERPOLATOR

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#pragma message("NGB_INTERPOLATOR set to " STR(NGB_INTERPOLATOR))

#undef STR
#undef STR_HELPER
/**
 * @brief Perform monotonic cubic interpolation between left and right bounds.
 *
 * This function guarantees monotonicity in the interpolation between left and right bounds
 * by interpolating the cubic function between the provided left and right values.
 * It computes the interpolated value at a given desired point and returns the result.
 *
 * @param leftn Value of the left bound.
 * @param lefthsml Cubed value of the left bound.
 * @param rightn Value of the right bound.
 * @param righthsml Cubed value of the right bound.
 * @param desngb Desired point for interpolation.
 * @param iter Iteration count (used for debugging purposes).
 * @return Returns the interpolated value at the desired point.
 */
#if NGB_INTERPOLATOR == 2
inline static double monotonic_cubic_interpolator(MyFloat leftn, MyFloat lefthsml, MyFloat rightn, MyFloat righthsml,
                                                  MyFloat desngb, int iter)
{
    double x1 = leftn;
    double y1 = pow((MyAtLeastDouble)lefthsml, 3);
    double x2 = rightn;
    double y2 = pow((MyAtLeaastDouble)righthsml, 3);

    double x = desngb;

    double q1 = 0., m1 = y1 / x1;
    double m2 = (y2 - y1) / (x2 - x1);
    double q2 = y1 - m2 * x1;
    double q3 = 0., m3 = y2 / x2;

    double ret = x < x1 ? (m1 * x + q1) : (x < x2 ? (m2 * x + q2) : (m3 * x + q3));

    VERBOSE_IF((ret < 0) || (iter > ITER_DEBUG), 0,
               "monotonic_cubic_interpolator: x1 = %g; y1 = %g; x2 =  %g; y2 = %g; x = %g; ret = %g; lefthsml = %g; "
               "right hsml = %g",
               double(x1), double(y1), double(x2), double(y2), double(x), double(ret), double(lefthsml),
               double(righthsml));

    return pow((MyAtLeastDouble)ret, 1. / 3.);
}
#endif // NGB_INTERPOLATOR

/**
 * @brief Perform interpolation for estimating the new smoothing length.
 *
 * This function performs interpolation to estimate the new smoothing length (`hsml_new`)
 * based on the provided left and right bounds (`lefthsml`, `righthsml`, `leftn`, `rightn`)
 * and the desired neighbor count (`desngb`).
 *
 * @param lefthsml Cubed value of the left bound smoothing length.
 * @param righthsml Cubed value of the right bound smoothing length.
 * @param leftn Value of the left bound.
 * @param rightn Value of the right bound.
 * @param desngb Desired neighbor count for interpolation.
 * @param iter Iteration count (used for experimental purposes and debugging).
 * @return Returns the estimated new smoothing length.
 */
inline static MyFloat ngb_interpolator_dual(MyFloat lefthsml, MyFloat righthsml, MyFloat leftn, MyFloat rightn,
                                            MyFloat desngb, int iter)
{
    MyFloat hsml_new;
    if ((iter + 1) % 8 == 0 || leftn == 0) // Experimental, avoiding some situation of slow convergence
    {
        hsml_new = 0.5 * (lefthsml + righthsml);
    }
    else
    {
#if NGB_INTERPOLATOR == 0
        hsml_new = 0.5 * (lefthsml + righthsml);
#elif NGB_INTERPOLATOR == 1
        lefthsml + (desngb - leftn) / (rightn - leftn) * (righthsml - lefthsml);
#elif NGB_INTERPOLATOR == 2
        hsml_new = monotonic_cubic_interpolator(leftn, lefthsml, rightn, righthsml, desngb, iter);
#elif NGB_INTERPOLATOR == 3
        hsml_new =
            lefthsml + pow((MyAtLeastDouble)(desngb - leftn) / (rightn - leftn), NUMDIMS) * (righthsml - lefthsml);
#elif NGB_INTERPOLATOR == 4
        MyFloat frac = (rightn - desngb) / (rightn - leftn);
        MyFloat arg = pow((MyAtLeastDouble)righthsml, NUMDIMS) -
                      frac * (pow((MyAtLeastDouble)righthsml, NUMDIMS) - pow((MyAtLeastDouble)lefthsml, NUMDIMS));
        if (arg < 0.0)
        {
            hsml_new = pow((MyAtLeastDouble)desngb / rightn * pow(righthsml, NUMDIMS), 1. / NUMDIMS);
        }
        else
        {
            hsml_new = pow((MyAtLeastDouble)arg, 1.0 / NUMDIMS);
        }
#elif NGB_INTERPOLATOR == 5
        hsml_new = lefthsml * exp2(log2((MyAtLeastDouble)desngb / leftn) * log2((MyAtLeastDouble)righthsml / lefthsml) /
                                   log2((MyAtLeastDouble)rightn / leftn));
#else // NGB_INTERPOLATOR
        ENDRUN_MAIN("NGB_INTERPOLATOR not set in range 0..5 !!!!");
#endif // NGB_INTERPOLATOR
    }

    if (iter >= ITER_DEBUG)
    {
        VERBOSE_ALL(2, "HSML_DEBUG int_dual (%d): task=%d Hsml=%g/%g N=%g/%g/%g  -> %g", iter, ThisTask, lefthsml,
                    righthsml, leftn, desngb, rightn, hsml_new);
    }

    if (!(hsml_new > 0 && hsml_new < 1e10))
    {
        VERBOSE_ALL(2, "interpolate_dual (%d): hsml=%g Left=%g/%g Right=%g/%g des=%g", ThisTask, double(hsml_new),
                    double(lefthsml), double(leftn), double(righthsml), double(rightn), double(desngb));
    }
    return hsml_new;
}

/**
 * @brief Perform interpolation for estimating the new smoothing length with a single bound.
 *
 * This function performs interpolation to estimate the new smoothing length (`hsml_new`)
 * based on the provided right bound (`righthsml`, `rightn`) and the desired neighbor count (`desngb`).
 * It is used when only a single bound is available for interpolation.
 *
 * @param righthsml Cubed value of the right bound smoothing length.
 * @param rightn Value of the right bound.
 * @param desngb Desired neighbor count for interpolation.
 * @param iter Iteration count (used for experimental purposes and debugging).
 * @return Returns the estimated new smoothing length.
 */
inline static MyFloat ngb_interpolator_single(MyFloat righthsml, MyFloat rightn, MyFloat desngb, int iter)
{
    MyFloat hsml_new;
    if (rightn > 0)
    {
        if (iter > 3) // Experimental, avoiding some situation of slow convergence
        {
            if (rightn > desngb)
            {
                hsml_new = righthsml * 0.5;
            }
            else
            {
                hsml_new = righthsml * 2;
            }
        }
        else
        {
#if NGB_INTERPOLATOR == 0
            hsml_new = desngb / rightn * righthsml;
#elif NGB_INTERPOLATOR == 1
            hsml_new = desngb / rightn * righthsml;
#elif NGB_INTERPOLATOR == 2
            hsml_new = monotonic_cubic_interpolator(0, 0, rightn, righthsml, desngb, iter);
#elif NGB_INTERPOLATOR == 3
            hsml_new = pow((MyAtLeastDouble)desngb / rightn, NUMDIMS) * righthsml;
#elif NGB_INTERPOLATOR == 4
            hsml_new = pow((MyAtLeastDouble)desngb / rightn * pow((MyAtLeastDouble)righthsml, NUMDIMS), 1. / NUMDIMS);
#elif NGB_INTERPOLATOR == 5
            hsml_new = pow((MyAtLeastDouble)desngb / rightn * pow((MyAtLeastDouble)righthsml, NUMDIMS), 1. / NUMDIMS);
#else // NGB_INTERPOLATOR
            ENDRUN_MAIN("NGB_INTERPOLATOR not set in range 0..5 !!!!");
#endif // NGB_INTERPOLATOR
        }
    }
    else
    {
        hsml_new = righthsml * 2;
    }

    if (iter >= ITER_DEBUG)
    {
        VERBOSE_ALL(2, "HSML_DEBUG int_singl (%d): task=%d Hsml=%g N=%g Ndes=%g  -> %g", iter, ThisTask, righthsml,
                    rightn, desngb, hsml_new);
    }

    if (!(hsml_new > 0 && hsml_new < 1e10))
    {
        VERBOSE_ALL(2, "interpolate_single (Task=%d,iter=%d): hsml=%g Right=%g/%g des=%g", ThisTask, iter,
                    double(hsml_new), double(righthsml), double(rightn), double(desngb));
    }

    return hsml_new;
}

/**
 * @brief Check if widening the search range is allowed based on the true number of neighbors.
 *
 * This function checks if widening the search range is allowed given the limits on the true
 * number of neighbors (`minn`, `leftn`, `rightn`, `maxn`) compared to the current smoothing lengths
 * (`hsml`, `lefthsml`, `righthsml`). It adjusts the smoothing lengths accordingly if the search range
 * needs to be widened.
 *
 * @param hsml Pointer to the current smoothing length.
 * @param lefthsml Pointer to the left bound smoothing length.
 * @param righthsml Pointer to the right bound smoothing length.
 * @param minn Minimum true neighbor count.
 * @param leftn Left bound true neighbor count.
 * @param rightn Right bound true neighbor count.
 * @param maxn Maximum true neighbor count.
 * @return Returns 1 if the search range is widened, 0 otherwise.
 */
inline static int ngb_check_widened_range(MyFloat *hsml, MyFloat *lefthsml, MyFloat *righthsml, MyFloat minn,
                                          MyFloat leftn, MyFloat rightn, MyFloat maxn, int iter)
{
    int ret = 0;

    if (iter >= ITER_DEBUG)
    {
        VERBOSE_ALL(2, "HSML_DEBUG check in (%d): task=%d Hsml=%g/%g/%g N=%g/%g/%g/%g", iter, ThisTask, *lefthsml,
                    *hsml, *righthsml, minn, leftn, rightn, maxn);
    }

    if (*hsml > *righthsml && *righthsml > 0) /* This can happen becaus of Ngb vs TrueNGB */
    {
        if (rightn < maxn)
        {
            *righthsml = 0;
        }
        else
        {
            *hsml = (*righthsml - 0.5 * (*righthsml - *lefthsml));
        }
        ret = 1;
    }
    else if (*hsml < *lefthsml) /* This can happen becaus of Ngb vs TrueNGB */
    {
        if (leftn > minn)
        {
            *lefthsml = 0;
        }
        else
        {
            *hsml = (*lefthsml + 0.5 * (*righthsml - *lefthsml));
        }
        ret = 1;
    }
    if (!(*hsml > 0 && *hsml < 1e10))
    {
        VERBOSE_ALL(2, "check bounds (%d): hsml=%g Left=%g Right=%g n=%g/%g/%g/%g", ThisTask, double(*hsml),
                    double(*lefthsml), double(*righthsml), double(minn), double(leftn), double(rightn), double(maxn));
    }

    if (iter >= ITER_DEBUG)
    {
        VERBOSE_ALL(2, "HSML_DEBUG check -> (%d): task=%d Hsml=%g/%g/%g N=%g/%g/%g/%g", iter, ThisTask, *lefthsml,
                    *hsml, *righthsml, minn, leftn, rightn, maxn);
    }

    return ret;
}

/**
 * @brief Structure for communication during the density computation. Holds data that is sent to other processors.
 */
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
static struct hsmldata_in
{
    MyLongDouble Pos[3];
    MyFloat VelPred[3];
    MyFloat Hsml;
#ifdef WINDS
    MyFloat DelayTime;
#endif // WINDS
#if defined(LT_STELLAREVOLUTION) || defined(AXION_DM) || defined(BLACK_HOLES)
    int Type;
#endif // defined(LT_STELLAREVOLUTION)  || defined(AXION_DM) || defined(BLACK_HOLES)
#ifdef GT_DUST
    MyFloat DeltaVelPred[GT_NUM_DUST][3];
    MyFloat DustFracPred[GT_NUM_DUST];
    MyFloat TotalDeltaVelPred[3];
    MyFloat TotalDustFracPred;
#endif // GT_DUST
#ifdef CS_DECOUPLING
    MyFloat CS_DensityOld;
    MyFloat Entropy;
#ifndef LT_STELLAREVOLUTION
    int Type;
#endif // LT_STELLAREVOLUTION
#endif // CS_DECOUPLING
} *HsmlDataIn, *HsmlDataGet;

/**
 * @brief Structure for communication during the density computation.
 *
 * It holds data that is received from other processors.
 */
static struct hsmldata_out
{
    MyAtLeastDouble NumNgb;
    int TrueNGB;
    MyAtLeastDouble Rho;
#if GADGET_HYDRO == HYDRO_SPH
    MyAtLeastDouble DhsmlDensity;
#elif GADGET_HYDRO == HYDRO_PESPH
    MyAtLeastDouble NumDens;
    MyAtLeastDouble Pressure;
    MyAtLeastDouble DhsmlNumDens;
    MyAtLeastDouble DhsmlPressure;
#elif GADGET_HYDRO == HYDRO_MFM
    MyAtLeastDouble NumDens;
    MyAtLeastDouble DhsmlNumDens;
#endif // GADGET_HYDRO
    MyAtLeastDouble DivVel; // This is the first order DivVel calculation. Only the sign is needed in compute_gradients.
                            // A second order value will be re-calculated there for later usage.

#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
    MyAtLeastDouble Vbulk[3]; /*!< Bulk velocity inside kernel */
#endif // JD_VTURB || KD_LIMIT_DIVV
#ifdef GT_DUST
    MyAtLeastDouble DivVelMixture;
#endif // GT_DUST
} *HsmlDataResult, *HsmlDataOut;

#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP
#include "../CodeBase/greentree.hpp"

#ifdef ACC_HSML
void particle2in_hsml(struct base_hsml_in *in, int i);
#else // ACC_HSML
void particle2in_hsml(struct hsmldata_in *in, int i);
void out2particle_hsml(struct hsmldata_out *out, int i, int mode);
#endif // ACC_HSML

/**
 * @brief Convert particle data to an input structure for HSML evaluation.
 *
 * This function converts particle data, such as position, velocity, smoothing length (HSML), and particle type, into an
 * input structure suitable for the evaluation of the local smoothing length (HSML). The input structure is then used to
 * perform calculations related to HSML determination.
 *
 * @param in Pointer to the input structure for storing converted particle data.
 * @param i Index of the particle for which to convert data.
 */
void particle2in_hsml(struct hsmldata_in *in, int i)
{
    for (int k = 0; k < 3; k++)
    {
        in->Pos[k] = P[i].Pos[k];
        if (P[i].Type == type_gas)
        {
            in->VelPred[k] = SphP[i].VelPred[k];
        }
        else
        {
            in->VelPred[k] = P[i].Vel[k];
        }
    }

    in->Hsml = P[i].Hsml;

#if defined(LT_STELLAREVOLUTION) || defined(AXION_DM) || defined(BLACK_HOLES)
    in->Type = P[i].Type;
#endif //  defined(LT_STELLAREVOLUTION) || defined(AXION_DM) || defined(BLACK_HOLES)

#ifdef CS_DECOUPLING
    in->CS_DensityOld = SphP[i].CS_DensityOld;
    in->Entropy = SphP[i].Entropy;
#ifndef LT_STELLAREVOLUTION
    in->Type = P[i].Type;
#endif // LT_STELLAREVOLUTION
#endif // CS_DECOUPLING

#ifdef WINDS
    if (P[i].Type == type_gas)
    {
        in->DelayTime = SphP[i].DelayTime;
    }
    else
    {
        in->DelayTime = 0;
    }
#endif // WINDS
#ifdef GT_DUST
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        for (int k = 0; k < 3; k++)
        {
            in->DeltaVelPred[idust][k] = SphP[i].DeltaVelPred[idust][k];
        }
        in->DustFracPred[idust] = SphP[i].DustFracPred[idust];
    }

    for (int k = 0; k < 3; k++)
    {
        in->TotalDeltaVelPred[k] = SphP[i].TotalDeltaVelPred[k];
    }
    in->TotalDustFracPred = SphP[i].TotalDustFracPred;
#endif // GT_DUST
}

/**
 * @brief Update particle properties with the output of HSML evaluation.
 *
 * This function updates the properties of a particle based on the output of HSML evaluation. The updated properties
 * include neighbor counts, density-related factors, and other relevant quantities depending on the particle type.
 *
 * @param out Pointer to the output structure containing the results of HSML evaluation.
 * @param i Index of the particle to update.
 * @param mode Mode of updating: 0 for addition and 1 for replacement.
 */
void out2particle_hsml(struct hsmldata_out *out, int i, int mode)
{
    ASSIGN_ADD(P[i].NumNgb, out->NumNgb, mode);
    ASSIGN_ADD(P[i].TrueNGB, out->TrueNGB, mode);

    if (P[i].Type == type_gas)
    {
#if GADGET_HYDRO == HYDRO_SPH
        ASSIGN_ADD(SphP[i].Density, out->Rho, mode);
        ASSIGN_ADD(SphP[i].DhsmlDensityFactor, out->DhsmlDensity, mode);
#elif GADGET_HYDRO == HYDRO_PESPH
        ASSIGN_ADD(SphP[i].NumDens, out->NumDens, mode);
        ASSIGN_ADD(SphP[i].Pressure, out->Pressure, mode);
        ASSIGN_ADD(SphP[i].DhsmlNumDensFactor, out->DhsmlNumDens, mode);
        ASSIGN_ADD(SphP[i].DhsmlPressureFactor, out->DhsmlPressure, mode);
#elif GADGET_HYDRO == HYDRO_MFM
        ASSIGN_ADD(SphP[i].NumDens, out->NumDens, mode);
        ASSIGN_ADD(SphP[i].DhsmlNumDensFactor, out->DhsmlNumDens, mode);
#endif // GADGET_HYDRO

        ASSIGN_ADD(SphP[i].DivVel, out->DivVel, mode);
#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
        for (int k = 0; k < 3; k++)
        {
            ASSIGN_ADD(SphP[i].Vbulk[k], out->Vbulk[k], mode);
        }
#endif // JD_VTURB || KD_LIMIT_DIVV
    }

#ifdef BLACK_HOLES
    if (P[i].Type == type_bh && All.BlackHoleHotColdAccretionOn >= 0)
    {
        assert(BHP[P[i].pt.BHID].PID == i);
        ASSIGN_ADD(BPP(i).BH_Density, out->Rho, mode);
    }
#endif // BLACK_HOLES
#ifdef GT_DUST
    ASSIGN_ADD(SphP[i].DivVelMixture, out->DivVelMixture, mode);
#endif // GT_DUST
}

/**
 * @brief Zero out the smoothing length and density-related quantities for a particle.
 *
 * If density for non SPH particles (like stars/BHs) are computed, it (at least theoretically) can be
 * that all (non wind) neighbours are on different MPI ranks, so we need to zero sumation values. As it
 * is not clear weather using and if statement and zeroing only non SPH particles is faster, we just zero all
 * values where we are not sure.
 */
void zero_particle_hsml(int i)
{
    P[i].NumNgb = 0;
    P[i].TrueNGB = 0;

    if (P[i].Type == type_gas)
    {
#if GADGET_HYDRO == HYDRO_SPH
        SphP[i].Density = 0;
        SphP[i].DhsmlDensityFactor = 0;
#elif GADGET_HYDRO == HYDRO_PESPH
        SphP[i].NumDens = 0;
        SphP[i].Pressure = 0;
        SphP[i].DhsmlNumDensFactor = 0;
        SphP[i].DhsmlPressureFactor = 0;
#elif GADGET_HYDRO == HYDRO_MFM
        SphP[i].NumDens = 0;
        SphP[i].DhsmlNumDensFactor = 0;
#endif // GADGET_HYDRO
        SphP[i].DivVel = 0;
#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
        SphP[i].Vbulk[0] = SphP[i].Vbulk[1] = SphP[i].Vbulk[2] = 0;
#endif // JD_VTURB || KD_LIMIT_DIVV
    }
#ifdef BLACK_HOLES
    if (P[i].Type == type_bh)
    {
        assert(BHP[P[i].pt.BHID].PID == i);
        BPP(i).BH_Density = 0;
    }
#endif // BLACK_HOLES
#ifdef GT_DUST
    SphP[i].DivVelMixture = 0;
#endif // GT_DUST
}

#ifdef DIVBCLEANING_TRICCO
/**
 * @brief Compute the Tricco cleaning velocity.
 *
 * This function cleans the velocity c_h from Tricco+2016 (eq. 27)
 */
MyAtLeastDouble get_tricco_cleaning_vel(int i)
{

    // skip cleaning if we are above the density threshold for star formation
#ifdef SFR
#ifdef LT_STELLAREVOLUTION
    MyFloat myPhysDensThresh = get_PhysDensThresh(i);
#else // EB_SFR_MAGNETIC
    MyFloat myPhysDensThresh = All.PhysDensThresh;
#endif // LT_STELLAREVOLUTION
    if (SphP[i].Density * All.cf_a3inv >= myPhysDensThresh)
        return 0.0;
#endif // SFR

    // sound speed squared
    MyAtLeastDouble csnd2 = GAMMA * SphP[i].Pressure / SphP[i].Density;

    // Alfven velocity squared, in the same comoving convention as csnd2 above: B is stored
    // comoving (B_code = a^2 B_phys) over a comoving density, so BPred^2*mu0_inv/Density comes
    // out as a*v_alfven_phys^2 and needs one more factor of a. CleanVel has to be comoving
    // because its consumers divide it by a (velocity_to_physical) or feed it to MaxSignalVel.
    MyAtLeastDouble v_alfven2 = (SphP[i].BPred[0] * SphP[i].BPred[0] + SphP[i].BPred[1] * SphP[i].BPred[1] +
                                 SphP[i].BPred[2] * SphP[i].BPred[2]) *
                                mu0_inv / SphP[i].Density * All.cf_atime;

    // clean velocity
    return All.DivBOverCleanFac * sqrt(csnd2 + v_alfven2);
}
#endif // DIVBCLEANING_TRICCO

#if !(defined ACC_INCLUDE_HSML) && defined(ACC_HSML) // we reinclude this same file if ACC is active
#include "../OpenACC/gpuallvars_branch.hpp"
#include "../OpenACC/find_hsml.hpp"
#endif // !(defined ACC_INCLUDE_HSML) && defined(ACC_HSML)

/**
 * @brief Compute the local density for each active SPH particle.
 *
 * This function computes the local density for each active SPH particle, the
 * number of neighbours in the current smoothing radius, and the divergence
 * and rotation of the velocity field. The pressure is updated as well.  If a
 * particle with its smoothing region is fully inside the local domain, it is
 * not exported to the other processors. The function also detects particles
 * that have a number of neighbours outside the allowed tolerance range. For
 * these particles, the smoothing length is adjusted accordingly, and the
 * density() computation is called again.  Note that the smoothing length is
 * not allowed to fall below the lower bound set by MinGasHsml (this may mean
 * that one has to deal with substantially more than normal number of
 * neighbours.)
 */
void find_hsml(void)
{
    TIMER_PUSH(1, "FIND_HSML");
    MyFloat *LeftHsml, *RightHsml;
    MyFloat *LeftNweighted, *RightNweighted;
    int *LeftNtrue, *RightNtrue;
    int ndone, ndone_flag, npleft, iter = 0, nitergas = 0, niterstars = 0, niterbhs = 0;
    int nexit_hfit = 0, nexit_dh = 0, nexit_nlim = 0, nexit_hwin = 0, nexit_slim = 0, nincr_bound = 0;
    int place;
    long long ntot;
    // MyAtLeastDouble fac; // seems not to be used
    int save_NextParticle;
    // long long n_exported = 0;
    TIMER_PUSH(2, "Setup_Left/Right");
    TIMER_PUSH(1, "HSML_SETUP");
    /* Here we set the limits for the NGB search:
       Defaults are always set to allow any possible number
       Therefore all combinations of NGB and weigthed NGB
       can be implemented */
    MyFloat DesN[6], MinN[6], MaxN[6], MinHsml[6], MaxHsml[6];
    int MinNtrue[6], MaxNtrue[6], MinNtrueDes[6], MaxNtrueDes[6];

    for (int i = 0; i < 6; i++)
    {
        MinN[i] = MinNtrue[i] = MinHsml[i] = DesN[i] = 0;
        MaxN[i] = MaxNtrue[i] = All.MaxPart;
        MinNtrueDes[i] = MaxNtrueDes[i] = All.DesNumNgb;
#ifdef PERIODIC
        MaxHsml[i] = (min(min(boxSize_X, boxSize_Y), boxSize_Z)); // / 2;
#else // PERIODIC
        MaxHsml[i] = 1e30; // All.BoxSize / 2;
#endif // PERIODIC
    }

    DesN[0] = All.DesNumNgb;
#if (GADGET_HYDRO == HYDRO_SPH) || (GADGET_HYDRO == HYDRO_MFM)
    MinN[0] = All.DesNumNgb * (1.0 - All.MaxNumNgbDeviationRel);
    MaxN[0] = All.DesNumNgb * (1.0 + All.MaxNumNgbDeviationRel);
#ifdef KD_RESTRICT_NEIGHBOURS
    MinNtrue[0] = All.DesNumNgb * REDUCTION_LOW_BOUND;
    MinNtrueDes[0] = MinNtrue[0] + 1;
    MaxNtrue[0] = All.DesNumNgb * INCREASING_HIGH_BOUND;
    MaxNtrueDes[0] = MaxNtrue[0] - 1;
#endif // KD_RESTRICT_NEIGHBOURS
#else // GADGET_HYDRO
    MinNtrue[0] = All.DesNumNgb * (1.0 - All.MaxNumNgbDeviationRel);
    MaxNtrue[0] = All.DesNumNgb * (1.0 + All.MaxNumNgbDeviationRel);
#endif // GADGET_HYDRO
    MinHsml[0] = All.MinGasHsml;

#ifdef MAXHSML
    MaxHsml[0] = All.MaxHsml;
#endif // MAXHSML

#ifdef BLACK_HOLES
    DesN[5] = All.DesNumNgb * All.BlackHoleNgbFactor;
    MinN[5] = DesN[5] * (1.0 - All.MaxNumNgbDeviationRel);
    MaxN[5] = DesN[5] * (1.0 + All.MaxNumNgbDeviationRel);
#ifdef KD_RESTRICT_NEIGHBOURS
    MinNtrue[5] = (All.DesNumNgb * All.BlackHoleNgbFactor) * REDUCTION_LOW_BOUND;
    MinNtrueDes[5] = MinNtrue[5] + 1;
    MaxNtrue[5] = (All.DesNumNgb * All.BlackHoleNgbFactor) * INCREASING_HIGH_BOUND;
    MaxNtrueDes[5] = MaxNtrue[5] - 1;
#endif // KD_RESTRICT_NEIGHBOURS
    MaxHsml[5] = All.BlackHoleMaxAccretionRadius;
#endif // BLACK_HOLES

#ifdef LT_STELLAREVOLUTION
    DesN[4] = All.DesNumNgbSN;

    MinHsml[4] = All.MinSpreadLength;
    MaxHsml[4] = All.MaxChemSpreadL;

#ifdef LT_USE_TRUENGB
    MinNtrue[4] = All.DesNumNgbSN * (1.0 - All.MaxNumNgbDevRelSN);
    MaxNtrue[4] = All.DesNumNgbSN * (1.0 + All.MaxNumNgbDevRelSN);
    MinNtrueDes[4] = All.DesNumNgbSN;
    MaxNtrueDes[4] = All.DesNumNgbSN;
#else // LT_USE_TRUENGB
    MinN[4] = All.DesNumNgbSN * (1.0 - All.MaxNumNgbDevRelSN);
    MaxN[4] = All.DesNumNgbSN * (1.0 + All.MaxNumNgbDevRelSN);
#if defined(KD_RESTRICT_NEIGHBOURS) && !defined(LT_USE_TRUENGB)
    MinNtrue[4] = All.DesNumNgbSN * REDUCTION_LOW_BOUND;
    MinNtrueDes[4] = MinNtrue[4] + 1;
    MaxNtrue[4] = All.DesNumNgbSN * INCREASING_HIGH_BOUND;
    MaxNtrueDes[4] = MaxNtrue[4] - 1;
#endif // KD_RESTRICT_NEIGHBOURS && !LT_USE_TRUENGB
#endif // LT_USE_TRUENGB

#endif // LT_STELLAREVOLUTION

    for (int iii = 0; iii < 6; iii++)
    {
        VERBOSE(2, "Bounds [%d]: hsml: %g/%g TNgb: %d/%g/%d  Ngb: %g/%g/%g  DesTNgb: %d/%d", iii, double(MinHsml[iii]),
                double(MaxHsml[iii]), MinNtrue[iii], double(DesN[iii]), MaxNtrue[iii], double(MinN[iii]),
                double(DesN[iii]), double(MaxN[iii]), MinNtrueDes[iii], MaxNtrueDes[iii]);
    }

    int NTaskTimesNumPart;
    NTaskTimesNumPart = ((int)(maxThreads)) * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    RecyclingBuffer = (int *)mymalloc("RecyclingBuffer", NTaskTimesNumPart * sizeof(int));

    LeftHsml = (MyFloat *)mymalloc("LeftHsml", NumPart * sizeof(MyFloat));
    RightHsml = (MyFloat *)mymalloc("RightHsml", NumPart * sizeof(MyFloat));
    LeftNweighted = (MyFloat *)mymalloc("LeftNweighted", NumPart * sizeof(MyFloat));
    RightNweighted = (MyFloat *)mymalloc("RightNweighted", NumPart * sizeof(MyFloat));
    LeftNtrue = (int *)mymalloc("LeftNtrue", NumPart * sizeof(int));
    RightNtrue = (int *)mymalloc("RightNtrue", NumPart * sizeof(int));

    unsigned int hitMaxChemSpreadL = 0;

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */

#ifdef _OPENMP
#pragma omp parallel for if (NActivePart > 100)
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP

        LeftHsml[i] = RightHsml[i] = LeftNweighted[i] = RightNweighted[i] = 0;
        LeftNtrue[i] = RightNtrue[i] = 0;

        if (P[i].ID == MyIDType(HSML_DEBUG))
        {
            VERBOSE_ALL(2, "HSML_DEBUG Entry: i=%d task=%d ID=%llu Hsml=%g", i, ThisTask, (unsigned long long)P[i].ID,
                        (float)P[i].Hsml);
            fflush(stdout);
        }
        double max09 = 0.9 * MaxHsml[P[i].Type];
        P[i].Hsml = (P[i].Hsml > max09 ? max09 : P[i].Hsml);

        if (hsml_isactive(i))
        {
            zero_particle_hsml(i);
        }

        if (P[i].ID == MyIDType(HSML_DEBUG))
        {
            VERBOSE_ALL(
                2, "HSML_DEBUG Setup: i=%d task=%d ID=%llu Hsml=%g/%g/%d Left=%g/%g/%d Right=%g/%g/%d Right-Left=%g", i,
                ThisTask, (unsigned long long)P[i].ID, (float)P[i].Hsml, (float)P[i].NumNgb, P[i].TrueNGB,
                (float)LeftHsml[i], (float)LeftNweighted[i], LeftNtrue[i], (float)RightHsml[i],
                (float)RightNweighted[i], RightNtrue[i], (float)(RightHsml[i] - LeftHsml[i]));
            fflush(stdout);
        }
    } // for ActiveParticleList

    /* allocate buffers to arrange communication */

    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize =
        (int)((MyBufferSize * 1024 * 1024) /
              (sizeof(struct data_index) + 3 * sizeof(struct data_nodelist) + sizeof(struct hsmldata_in) +
               sizeof(struct hsmldata_out) + sizemax(sizeof(struct hsmldata_in), sizeof(struct hsmldata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    VERBOSE(2, "Hsml: using %g MB for buffering, %g MB remaining for other buffers", MyBufferSize / 1.0,
            FreeBytes / (1024.0 * 1024.0));

#ifdef ACC_HSML
    int DoPrimaryFirstIterationOnGPU = 0, DoPrimaryOnGPU = 0, should_drift;
    struct hsmldata_out *LiteDOut = NULL;
    int *HsmlActiveParticleList = NULL;
    struct hsmldata_in *LiteDIn = openacc_hsml_init(should_drift, DoPrimaryFirstIterationOnGPU);
#else // ACC_HSML
    int DoPrimaryOnGPU = 0;
    int should_drift = 1;
#endif // ACC_HSML

    TIMER_POP(1, "HSML_SETUP");
    TIMER_POP(2, "Setup_Left/Right");

    do
    {

#ifdef ACC_HSML
        int NHsmlActivePart = openacc_hsml_iteration_init(should_drift, DoPrimaryFirstIterationOnGPU, DoPrimaryOnGPU,
                                                          HsmlActiveParticleList, LiteDOut, iter);
        if (DoPrimaryOnGPU)
        {
#ifdef ACC_OPENACC
#pragma acc enter data create(HsmlActiveParticleList [0:NHsmlActivePart], LiteDIn [0:NumPart],                         \
                              LiteDOut [0:NHsmlActivePart])
#pragma acc update device(HsmlActiveParticleList [0:NHsmlActivePart], LiteDIn [0:NumPart],                             \
                          LiteDOut [0:NHsmlActivePart], WalkLiteP [0:NumPart]) ACC_ASYNC_COPY1
#elif defined(ACC_OPENMP)

#pragma omp target enter data map(alloc                                                                                \
                                  : HsmlActiveParticleList [0:NHsmlActivePart], LiteDIn [0:NumPart],                   \
                                    LiteDOut [0:NHsmlActivePart])

#pragma omp target update to(HsmlActiveParticleList [0:NHsmlActivePart], LiteDIn [0:NumPart],                          \
                             LiteDOut [0:NHsmlActivePart], WalkLiteP [0:NumPart])

#endif // ACC_OPENMP
        }

        TIMER_PUSH(1, "ACC_I");
        openacc_hsml_first_phase(DoPrimaryOnGPU, NHsmlActivePart, HsmlActiveParticleList, LiteDOut, LiteDIn);
        TIMER_POP(1, "ACC_I");

        { // this parenthesys is contains the density data region. It is closed at the end of the iterations.
#endif // ACC_HSML

            NextParticle = FirstActiveParticle; /* beginn with this index */
            int group_particles = 1;
            do
            {
                TIMER_PUSH(2, "Primary");
                BufferFullFlag = 0;
                Nexport = 0;
                save_NextParticle = NextParticle;

                TIMER_PUSH(1, "HSML_COMPUTE");

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
                {
#ifdef _OPENMP
                    int thread_id = omp_get_thread_num();
#else // _OPENMP
                int thread_id = 0;
#endif // _OPENMP
                    ar_green_evaluate_primary(
                        hsml_isactive, hsml_evaluate, Module_HSML,
#ifdef CS_DECOUPLING
                        TreeWalkType_Gadget3TreeWalk,
#else // CS_DECOUPLING
                    TreeWalkType_GreenTreeWalk,
#endif // CS_DECOUPLING
                        NextParticleLoop_ActiveParticleList, NeighbourSearchCriteria_Density,
#ifdef CS_DECOUPLING
                        cs_group_criteria<particle_data *>, // actually not called as TreeWalkType_Gadget3TreeWalk is
                                                            // used
                        cs_treewalk_filter_variable_decoupling_criteria_IPhase<particle_data *>,
#endif // CS_DECOUPLING
                        &thread_id, DoPrimaryOnGPU, should_drift, group_particles);
                }

#ifdef ADD_EXTRA_BARRIER_FOR_EXACT_INBALANCE_MEASURE
                TIMER_POPPUSH(1, "HSML_COMPUTE", "HSML_WAIT");
                MPI_Barrier(MYMPI_COMM_WORLD); // Wait for all tasks to synchronize
                TIMER_POP(1, "HSML_WAIT");
#else // ADD_EXTRA_BARRIER_FOR_EXACT_INBALANCE_MEASURE
            TIMER_POP(1, "HSML_COMPUTE");
#endif // ADD_EXTRA_BARRIER_FOR_EXACT_INBALANCE_MEASURE

                TIMER_POPPUSH(2, "Primary", "Exchange");
                TIMER_PUSH(1, "HSML_COMM_PREP");
                /* if we are not grouping particles and SendRecv fails, then we panic as there is no memory to process
                 * even a single particles */
                /*and  if green tree fails, we retry without it */

                int panic_on_error = !group_particles;
                int is_send_receive_ok = prepare_sendrec_offset_send(save_NextParticle, panic_on_error);

                TIMER_POP(1, "HSML_COMM_PREP");

                if (!is_send_receive_ok)
                {
                    TIMER_POP(2, "Exchange");
                    group_particles = 0;
                    continue;
                }

                TIMER_PUSH(1, "HSML_COPY");

                HsmlDataGet = (struct hsmldata_in *)mymalloc("HsmlDataGet", Nimport * sizeof(struct hsmldata_in));
                DataNodeListGet =
                    (struct data_nodelist *)mymalloc("DataNodeListGet", Nimport * sizeof(struct data_nodelist));
                HsmlDataIn = (struct hsmldata_in *)mymalloc("HsmlDataIn", Nexport * sizeof(struct hsmldata_in));
                DataNodeListIn =
                    (struct data_nodelist *)mymalloc("DataNodeListIn", Nexport * sizeof(struct data_nodelist));

                /* prepare particle data for export */
                for (int j = 0; j < Nexport; j++)
                {
                    place = DataIndexTable[j].Index;
                    particle2in_hsml(&HsmlDataIn[j], place);
                }

                copy_data_nodelist();

                /* exchange particle data */
                TIMER_POPPUSH(1, "HSML_COPY", "HSML_COMM_EXC");
                templated_exchange_data_in_get(HsmlDataIn, HsmlDataGet, 1, TAG_DENS_A);
                myfree(DataNodeListIn);
                myfree(HsmlDataIn);
                HsmlDataResult =
                    (struct hsmldata_out *)mymalloc("HsmlDataResult", Nimport * sizeof(struct hsmldata_out));
                HsmlDataOut = (struct hsmldata_out *)mymalloc("HsmlDataOut", Nexport * sizeof(struct hsmldata_out));

                report_memory_usage(&HighMark_sphhsml, "SPH_HSML");

                TIMER_POP(1, "HSML_COMM_EXC");
                TIMER_POPPUSH(2, "Exchange", "Secondary");
                /* now do the particles that were sent to us */
                TIMER_PUSH(1, "HSML_COMPUTE");
                NextJ = 0;

#ifdef ACC_HSML
                openacc_hsml_secondary_phase(HsmlDataGet, HsmlDataResult, LiteDIn, DoPrimaryOnGPU, should_drift);
#else // ACC_HSML
#ifdef _OPENMP // ACC
#pragma omp parallel
#endif // _OPENMP
            {
#ifdef _OPENMP
                int thread_id = omp_get_thread_num();
#else // _OPENMP
                int thread_id = 0;
#endif // _OPENMP
                ar_green_evaluate_secondary(
                    HsmlDataGet, hsml_evaluate, Module_HSML,
#ifdef CS_DECOUPLING
                    TreeWalkType_Gadget3TreeWalk,
#else // CS_DECOUPLING
                    TreeWalkType_GreenTreeWalk,
#endif // CS_DECOUPLING
                    NeighbourSearchCriteria_Density,
#ifdef CS_DECOUPLING
                    cs_group_criteria<hsmldata_in *>, // actually not called as TreeWalkType_Gadget3TreeWalk is used
                    cs_treewalk_filter_variable_decoupling_criteria_IIPhase<hsmldata_in *>,
#endif // CS_DECOUPLING
                    &thread_id, should_drift);
            }
#endif // ACC_HSML

                TIMER_POP(1, "HSML_COMPUTE");

                if (NextParticle < 0)
                {
                    ndone_flag = 1;
                }
                else
                {
                    ndone_flag = 0;
                }

                TIMER_PUSH(1, "HSML_WAIT");
                MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
                TIMER_POP(1, "HSML_WAIT");

                TIMER_POPPUSH(2, "Secondary", "Send_Results");

                TIMER_PUSH(1, "HSML_COMM_EXC");
                templated_exchange_data_Sendrecv(HsmlDataResult, HsmlDataOut, TAG_DENS_B, Recv_offset, Recv_count,
                                                 Send_offset, Send_count);
                TIMER_POPPUSH(1, "HSML_COMM_EXC", "HSML_COPY");

                /* add the result to the local particles */
                for (int j = 0; j < Nexport; j++)
                {
                    place = DataIndexTable[j].Index;
                    out2particle_hsml(&HsmlDataOut[j], place, 1);
                }
                TIMER_POP(1, "HSML_COPY");
                TIMER_POP(2, "Send_Results");
                myfree(HsmlDataOut);
                myfree(HsmlDataResult);
                myfree(DataNodeListGet);
                myfree(HsmlDataGet);
            } while (ndone < NTask);
#ifdef ACC_HSML
            TIMER_PUSH(1, "ACC_WAIT");
            if (DoPrimaryOnGPU)
            {
#ifdef ACC_OPENACC
#pragma acc wait(1)
#pragma acc update host(LiteDOut [0:NHsmlActivePart])
#pragma acc exit data delete (HsmlActiveParticleList [0:NHsmlActivePart], LiteDIn [0:NumPart],                         \
                              LiteDOut [0:NHsmlActivePart])
#elif defined(ACC_OPENMP)
#pragma omp taskwait
#pragma omp target update from(LiteDOut [0:NHsmlActivePart])
#pragma omp target exit data map(delete                                                                                \
                                 : HsmlActiveParticleList [0:NHsmlActivePart], LiteDIn [0:NumPart],                    \
                                   LiteDOut [0:NHsmlActivePart])
#endif // ACC_OPENACC | ACC_OPENMP
            }
            TIMER_POP(1, "ACC_WAIT");
        } // and data

        TIMER_PUSH(1, "ACC_MERGE");
        if (DoPrimaryOnGPU)
        {
#pragma omp parallel for
            for (int counter = 0; counter < NHsmlActivePart; counter++)
            {
                int i = HsmlActiveParticleList[counter];
                out2particle_hsml(&LiteDOut[counter], i, 1);
            }
        }
        TIMER_POP(1, "ACC_MERGE");
#endif // ACC_HSML

        /* do final operations on results */
        TIMER_PUSH(2, "Final");
        TIMER_PUSH(1, "HSML_FINAL");
        npleft = 0;

        // precompute since constant over loop
#ifdef WENDLAND_C2_KERNEL
        MyAtLeastDouble pow_DesNumNgb = pow(All.DesNumNgb * 0.01, -0.977);
#endif // WENDLAND_C2_KERNEL
#ifdef WENDLAND_C4_KERNEL
        MyAtLeastDouble pow_DesNumNgb = pow(All.DesNumNgb * 0.01, -1.579);
#endif // WENDLAND_C4_KERNEL
#ifdef WENDLAND_C6_KERNEL
        MyAtLeastDouble pow_DesNumNgb = pow(All.DesNumNgb * 0.01, -2.236);
#endif // WENDLAND_C6_KERNEL

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : npleft, nitergas, niterstars, niterbhs, nexit_hfit, nexit_dh, nexit_nlim,       \
                                       nexit_slim, nexit_hwin, nincr_bound,                                            \
                                       hitMaxChemSpreadL) // private(fac)	// if(NActivePart>80)
        for (int il = 0; il < NActivePart; il++)

#else // _OPENMP
        for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])

#endif // _OPENMP
        {
#ifdef _OPENMP
            int i = ActiveParticleList[il];
#endif // _OPENMP
            if (hsml_isactive(i))
            {
                // explicit power avoids a pow call for a compile-time-known small integer exponent
#ifdef ONEDIM
                MyAtLeastDouble hsml_pow_numdims = P[i].Hsml;
#elif defined(TWODIMS)
                MyAtLeastDouble hsml_pow_numdims = P[i].Hsml * P[i].Hsml;
#else // ONEDIM || TWODIMS
            MyAtLeastDouble hsml_pow_numdims = P[i].Hsml * P[i].Hsml * P[i].Hsml;
#endif // ONEDIM
                P[i].NumNgb *= NORM_COEFF * hsml_pow_numdims; /* 4.0/3 * PI = 4.188790204786 */

                if (IS_SPH_ALIVE(P[i]))
                {

#if GADGET_HYDRO == HYDRO_PESPH
                    // SphP[i].p.Pressure = powf(SphP[i].p.Pressure, 1.0); //GAMMA);
                    SphP[i].Density = SphP[i].NumDens * P[i].Mass;
#elif GADGET_HYDRO == HYDRO_MFM
                    SphP[i].Density = SphP[i].NumDens * P[i].Mass;
#endif // GADGET_HYDRO

#if GADGET_HYDRO == HYDRO_SPH
                    SphP[i].DhsmlDensityFactor *= P[i].Hsml / (NUMDIMS * SphP[i].Density);
                    if (SphP[i].DhsmlDensityFactor >
                        -0.9) /* note: this would be -1 if only a single particle at zero lag is found */
                    {
                        SphP[i].DhsmlDensityFactor = 1 / (1 + SphP[i].DhsmlDensityFactor);
                    }
                    else
                    {
                        SphP[i].DhsmlDensityFactor = 1;
                    }
#elif GADGET_HYDRO == HYDRO_PESPH
                    SphP[i].DhsmlPressureFactor *= P[i].Hsml / (NUMDIMS * SphP[i].NumDens);
                    SphP[i].DhsmlNumDensFactor *= P[i].Hsml / (NUMDIMS * SphP[i].NumDens);
                    if (SphP[i].DhsmlNumDensFactor >
                        -0.9) /* note: this would be -1 if only a single particle at zero lag is found */
                    {
                        SphP[i].DhsmlNumDensFactor = 1 / (1 + SphP[i].DhsmlNumDensFactor);
                    }
                    else
                    {
                        SphP[i].DhsmlNumDensFactor = 1;
                    }
#elif GADGET_HYDRO == HYDRO_MFM
                SphP[i].DhsmlNumDensFactor *= P[i].Hsml / (NUMDIMS * SphP[i].NumDens);
                if (SphP[i].DhsmlNumDensFactor >
                    -0.9) /* note: this would be -1 if only a single particle at zero lag is found */
                {
                    SphP[i].DhsmlNumDensFactor = 1 / (1 + SphP[i].DhsmlNumDensFactor);
                }
                else
                {
                    SphP[i].DhsmlNumDensFactor = 1;
                }
#endif // GADGET_HYDRO

#if !defined(NO_BIAS_CORRECTION) && (NUMDIMS == 3) /* Dehnen & Aly 2012, eq (18), (19) */
#if defined(WENDLAND_C2_KERNEL) || defined(WENDLAND_C4_KERNEL) || defined(WENDLAND_C6_KERNEL)
                    MyAtLeastDouble kernel_wk = NORM / (P[i].Hsml * P[i].Hsml * P[i].Hsml);
#endif // WENDLAND_C2_KERNEL || WENDLAND_C4_KERNEL || WENDLAND_C6_KERNEL
                    MyAtLeastDouble wc_correction = 0;

#ifdef WENDLAND_C2_KERNEL
                    wc_correction = 0.0294 * pow_DesNumNgb * P[i].Mass * kernel_wk;
#endif // WENDLAND_C2_KERNEL
#ifdef WENDLAND_C4_KERNEL
                    wc_correction = 0.01342 * pow_DesNumNgb * P[i].Mass * kernel_wk;
#endif // WENDLAND_C4_KERNEL
#ifdef WENDLAND_C6_KERNEL
                    wc_correction = 0.0116 * pow_DesNumNgb * P[i].Mass * kernel_wk;
#endif // WENDLAND_C6_KERNEL

                    if (wc_correction < 0.2 * SphP[i].Density)
                    {
                        SphP[i].Density -= wc_correction;
                    }
#endif // NO_BIAS_CORRECTION

                    SphP[i].Pressure = get_pressure(i);

                    if (SphP[i].Density > 0)
                    {
                        SphP[i].DivVel /= SphP[i].Density;
#ifdef GT_DUST
                        SphP[i].DivVelMixture /= SphP[i].Density;
#endif // GT_DUST
#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
                        SphP[i].Vbulk[0] /= SphP[i].Density;
                        SphP[i].Vbulk[1] /= SphP[i].Density;
                        SphP[i].Vbulk[2] /= SphP[i].Density;
#endif // JD_VTURB || KD_LIMIT_DIVV
                    }
                    else
                    {
                        SphP[i].DivVel = 0;
#ifdef GT_DUST
                        SphP[i].DivVelMixture = 0;
#endif // GT_DUST
#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
                        SphP[i].Vbulk[0] = SphP[i].Vbulk[1] = SphP[i].Vbulk[2] = 0;
#endif // JD_VTURB || KD_LIMIT_DIVV
                    }
#ifdef DEBUG
                    if (!(SphP[i].DivVel > -2000 && SphP[i].DivVel < 2000))
                    {
                        VERBOSE_ALL(
                            0, "WARNING: task=%d: Gas Particle id=%llu,m=%g,rho=%g strange divV value in findhsml: %g",
                            ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, SphP[i].Density, SphP[i].DivVel);
                    }
#endif // DEBUG

#ifdef DIVBCLEANING_TRICCO
                    // compute cleaning velocity (Tricco+2016, eq. 27)
                    SphP[i].CleanVel = get_tricco_cleaning_vel(i);
#endif // DIVBCLEANING_TRICCO

#if GADGET_HYDRO == HYDRO_MFM
                    // We choose the units internally that mu0=1 for simplicity. This means no mu0 is required
                    // internally for the force calculation within the Riemann solver. When assigning magnetic field
                    // values back, we have to ensure correct unit conversion back to original units.
                    SphP[i].Wprim = WFluidVector<mfm_numdims>(SphP[i].VelPred,
                                                              SphP[i].Density, SphP[i].Pressure);
                    SphP[i].Wmin = SphP[i].Wprim;
                    SphP[i].Wmax = SphP[i].Wprim;
                    SphP[i].dQdt = (MyFloat)0.0; // zero out before MFM calculation
                    for (int j = 0; j < numvar; j++)
                    {
                        SphP[i].AlphaSlope[j] = (MyFloat)1.0;
                    }
                    SphP[i].DtEntropyExtra = (MyFloat)0.0;
                    for (int k = 0; k < 3; k++)
                    {
                        SphP[i].HydroAccelExtra[k] = (MyAtLeastDouble)0.0;
                    }
#ifdef GT_DUST
#if NUMDIMS == 1
                    Vector<NUMDIMS> Wvel(SphP[i].VelPred[0] - SphP[i].TotalDustFracPred * SphP[i].TotalDeltaVelPred[0]);
#elif NUMDIMS == 2
                    Vector<NUMDIMS> Wvel(SphP[i].VelPred[0] - SphP[i].TotalDustFracPred * SphP[i].TotalDeltaVelPred[0],
                                         SphP[i].VelPred[1] - SphP[i].TotalDustFracPred * SphP[i].TotalDeltaVelPred[1]);
#elif NUMDIMS == 3
                    Vector<NUMDIMS> Wvel(SphP[i].VelPred[0] - SphP[i].TotalDustFracPred * SphP[i].TotalDeltaVelPred[0],
                                         SphP[i].VelPred[1] - SphP[i].TotalDustFracPred * SphP[i].TotalDeltaVelPred[1],
                                         SphP[i].VelPred[2] - SphP[i].TotalDustFracPred * SphP[i].TotalDeltaVelPred[2]);
#endif // NUMDIMS

                    SphP[i].Wprim = WFluidVector<NUMDIMS>(Wvel, SphP[i].Density * (1. - SphP[i].TotalDustFracPred),
                                                          SphP[i].Pressure);

                    SphP[i].Wmin = SphP[i].Wprim;
                    SphP[i].Wmax = SphP[i].Wprim;

                    for (int idust = 0; idust < GT_NUM_DUST; idust++)
                    {
#if NUMDIMS == 1
                        Vector<NUMDIMS> Wvel_dust(SphP[i].VelPred[0] -
                                                  SphP[i].TotalDustFracPred * SphP[i].TotalDeltaVelPred[0] +
                                                  SphP[i].DeltaVelPred[idust][0]);
#elif NUMDIMS == 2
                        Vector<NUMDIMS> Wvel_dust(
                            SphP[i].VelPred[0] - SphP[i].TotalDustFracPred * SphP[i].TotalDeltaVelPred[0] +
                                SphP[i].DeltaVelPred[idust][0],
                            SphP[i].VelPred[1] - SphP[i].TotalDustFracPred * SphP[i].TotalDeltaVelPred[1] +
                                SphP[i].DeltaVelPred[idust][1]);
#elif NUMDIMS == 3
                        Vector<NUMDIMS> Wvel_dust(
                            SphP[i].VelPred[0] - SphP[i].TotalDustFracPred * SphP[i].TotalDeltaVelPred[0] +
                                SphP[i].DeltaVelPred[idust][0],
                            SphP[i].VelPred[1] - SphP[i].TotalDustFracPred * SphP[i].TotalDeltaVelPred[1] +
                                SphP[i].DeltaVelPred[idust][1],
                            SphP[i].VelPred[2] - SphP[i].TotalDustFracPred * SphP[i].TotalDeltaVelPred[2] +
                                SphP[i].DeltaVelPred[idust][2]);
#endif // NUMDIMS

                        SphP[i].Wprim_dust[idust] = WFluidVector<NUMDIMS>(
                            Wvel_dust, SphP[i].Density * dustfrac(SphP[i].DustFracPred[idust]), MyFloat(0.0));

                        SphP[i].Wmin_dust[idust] = SphP[i].Wprim_dust[idust];
                        SphP[i].Wmax_dust[idust] = SphP[i].Wprim_dust[idust];
                        SphP[i].dQdt_dust[idust] = (MyFloat)0.0; // zero out before MFM calculation
                        for (int j = 0; j < numvar; j++)
                        {
                            SphP[i].AlphaSlope_dust[idust][j] = (MyFloat)1.0;
                        }

#ifdef GT_DUST_DIFFUSION
                        for (int k = 0; k < 3; k++)
                        {
                            SphP[i].DtDeltaVelExtra[idust][k] = (MyFloat)0.0;
                        }
#endif // GT_DUST_DIFFUSION
                    }
#endif // GT_DUST
#ifdef MFM_FACE_CLOSURE_RESIDUAL
                    // FaceAreaSum (closure residual) is accumulated (+=) during the MFM flux walk
                    for (int k = 0; k < 3; k++)
                    {
                        SphP[i].FaceAreaSum[k] = (MyFloat)0.0;
                    }
#endif // MFM_FACE_CLOSURE_RESIDUAL
#endif // GADGET_HYDRO == HYDRO_MFM
                    if (P[i].ID == MyIDType(HSML_DEBUG))
                    {
                        VERBOSE_ALL(2, "HSML_DEBUG Final: i=%d task=%d ID=%llu Hsml=%g NGB=%g tNGB=%d rho=%g", i,
                                    ThisTask, (unsigned long long)P[i].ID, (float)P[i].Hsml, (float)P[i].NumNgb,
                                    P[i].TrueNGB, (float)SphP[i].Density);
                        fflush(stdout);
                    }
                } // IS_SPH_ALIVE

                /* now check whether we had enough neighbours */
                int redo_particle = 0;
                MyFloat hsml_old = P[i].Hsml;

                if (HSML_DEBUG >= 0)
                {
                    if (P[i].Type == type_gas)
                        if (SphP[i].Density > 1e5)
                        {
                            VERBOSE_ALL(
                                2, "HSML_ALARM: strange density i=%d task=%d ID=%llu Hsml=%g NGB=%g tNGB=%d rho=%g", i,
                                ThisTask, (unsigned long long)P[i].ID, (float)P[i].Hsml, (float)P[i].NumNgb,
                                P[i].TrueNGB, (float)SphP[i].Density);
                            fflush(stdout);
                        }
                }

                if ((iter >= ITER_DEBUG) || P[i].ID == MyIDType(HSML_DEBUG))
                {
                    VERBOSE_ALL(2,
                                "HSML_DEBUG Before (%d): i=%d task=%d ID=%llu Type=%d Hsml=%g/%g/%d Left=%g/%g/%d "
                                "Right=%g/%g/%d Right-Left=%g/%g",
                                iter, i, ThisTask, (unsigned long long)P[i].ID, P[i].Type, (float)P[i].Hsml,
                                (float)P[i].NumNgb, P[i].TrueNGB, (float)LeftHsml[i], (float)LeftNweighted[i],
                                LeftNtrue[i], (float)RightHsml[i], (float)RightNweighted[i], RightNtrue[i],
                                (float)(RightHsml[i] - LeftHsml[i]), (float)abs(hsml_old - P[i].Hsml));
                }

                /* First Check if the change in bounds are already very small */
                if ((RightHsml[i] - LeftHsml[i]) < P[i].Hsml * (HSML_TOLERANCE * iter) && LeftHsml[i] > 0 &&
                    RightHsml[i] > 0)
                {
                    nexit_hwin++;
                    redo_particle = 0; /* therefore we stop here */
                }
                else
                {
                    /* Case all requirements are met, outer loop is True neighbours, inner loop is Weighted neighbours
                     */
                    if (P[i].TrueNGB >= MinNtrue[P[i].Type] &&
                        P[i].TrueNGB <= MaxNtrue[P[i].Type]) /* Total Neighbours are fine */
                    {
                        if ((iter >= ITER_DEBUG) || P[i].ID == MyIDType(HSML_DEBUG))
                        {
                            VERBOSE_ALL(
                                2, "HSML_DEBUG Iterating on weighted numbers for i=%d task=%d ID=%llu Ngb=%g/%g/%g", i,
                                ThisTask, (unsigned long long)P[i].ID, MinN[P[i].Type], P[i].NumNgb, MaxN[P[i].Type]);
                        }
                        if (P[i].NumNgb >= MinN[P[i].Type] &&
                            P[i].NumNgb <= MaxN[P[i].Type]) /* Weighted Neighbours are fine */
                        {
                            nexit_hfit++;
                            redo_particle = 0; /* Nothing to do */
                        }
                        else if (P[i].NumNgb < MinN[P[i].Type]) /* Need larger hsml      */
                        {
                            redo_particle = 1;
                            LeftHsml[i] = P[i].Hsml;
                            LeftNweighted[i] = P[i].NumNgb;
                            LeftNtrue[i] = P[i].TrueNGB;
                            if (RightHsml[i] > 0)
                            {
                                P[i].Hsml = ngb_interpolator_dual(LeftHsml[i], RightHsml[i], LeftNweighted[i],
                                                                  RightNweighted[i], DesN[P[i].Type], iter);
                            }
                            else
                            {
#ifdef KD_FAST_SINGLE_GUESS
                                P[i].Hsml = ngb_interpolator_single(P[i].Hsml, P[i].NumNgb, DesN[P[i].Type], iter);
#else // KD_FAST_SINGLE_GUESS
                                double fac = 1.26;
#if GADGET_HYDRO == HYDRO_SPH
                                if (P[i].Type == type_gas)
                                {
                                    fac = 1 - (P[i].NumNgb - DesN[P[i].Type]) / (NUMDIMS * P[i].NumNgb) *
                                                  SphP[i].DhsmlDensityFactor;
                                }
#elif GADGET_HYDRO == HYDRO_MFM
                                if (P[i].Type == type_gas)
                                {
                                    fac = 1 - (P[i].NumNgb - DesN[P[i].Type]) / (NUMDIMS * P[i].NumNgb) *
                                                  SphP[i].DhsmlNumDensFactor;
                                }
#endif // GADGET_HYDRO
                                if (fac < 1.26 && fac > 1.05)
                                {
                                    P[i].Hsml *= fac;
                                }
                                else
                                {
                                    P[i].Hsml *= 1.26;
                                }
#endif // KD_FAST_SINGLE_GUESS
                            }
                            nincr_bound +=
                                ngb_check_widened_range(&P[i].Hsml, &LeftHsml[i], &RightHsml[i], MinNtrue[P[i].Type],
                                                        LeftNtrue[i], RightNtrue[i], MaxNtrue[P[i].Type], iter);
                        }
                        else if (P[i].NumNgb > MaxN[P[i].Type]) /* Need smaler hsml      */
                        {
                            redo_particle = 1;
                            RightHsml[i] = P[i].Hsml;
                            RightNweighted[i] = P[i].NumNgb;
                            RightNtrue[i] = P[i].TrueNGB;
                            if (LeftHsml[i] > 0)
                            {
                                P[i].Hsml = ngb_interpolator_dual(LeftHsml[i], RightHsml[i], LeftNweighted[i],
                                                                  RightNweighted[i], DesN[P[i].Type], iter);
                            }
                            else
                            {
#ifdef KD_FAST_SINGLE_GUESS
                                P[i].Hsml = ngb_interpolator_single(P[i].Hsml, P[i].NumNgb, DesN[P[i].Type], iter);
#else // KD_FAST_SINGLE_GUESS
                                double fac = 1.26;
#if GADGET_HYDRO == HYDRO_SPH
                                if (P[i].Type == type_gas)
                                {
                                    fac = 1 - (P[i].NumNgb - DesN[P[i].Type]) / (NUMDIMS * P[i].NumNgb) *
                                                  SphP[i].DhsmlDensityFactor;
                                }
#elif GADGET_HYDRO == HYDRO_MFM
                                if (P[i].Type == type_gas)
                                {
                                    fac = 1 - (P[i].NumNgb - DesN[P[i].Type]) / (NUMDIMS * P[i].NumNgb) *
                                                  SphP[i].DhsmlNumDensFactor;
                                }
#endif // GADGET_HYDRO
                                if (fac > 1 / 1.26 && fac < 1 / 1.05)
                                {
                                    P[i].Hsml *= fac;
                                }
                                else
                                {
                                    P[i].Hsml /= 1.26;
                                }
#endif // KD_FAST_SINGLE_GUESS
                            }
                            nincr_bound +=
                                ngb_check_widened_range(&P[i].Hsml, &LeftHsml[i], &RightHsml[i], MinNtrue[P[i].Type],
                                                        LeftNtrue[i], RightNtrue[i], MaxNtrue[P[i].Type], iter);
                        }
                        else
                        {
                            ENDRUN("NumNgb seems to be no number, this should not happen (ID=%llu)!!!!",
                                   (unsigned long long)P[i].ID); /* Can't happen !! */
                        }
                    }
                    else if (P[i].TrueNGB < MinNtrue[P[i].Type]) /* Would like to grow hsml */
                    {
                        if ((iter >= ITER_DEBUG) || P[i].ID == MyIDType(HSML_DEBUG))
                        {
                            VERBOSE_ALL(2, "HSML_DEBUG True NGB numbers too small for i=%d task=%d ID=%llu", i,
                                        ThisTask, (unsigned long long)P[i].ID);
                        }
                        redo_particle = 1;
                        if (P[i].NumNgb > MaxN[P[i].Type]) /* Weighted Ngb shrink hsml (!!) */
                        {
                            nexit_nlim++;
                            redo_particle = 0; /* therefore we stop here */
                        }
                        else
                        {
                            LeftHsml[i] = P[i].Hsml;
                            LeftNweighted[i] = P[i].NumNgb;
                            LeftNtrue[i] = P[i].TrueNGB;
                            if (RightNtrue[i] > MinNtrueDes[P[i].Type]) /* We are close to the limit, so simple
                                                                           interpolation is better ... */
                            {
                                P[i].Hsml = ngb_interpolator_dual(LeftHsml[i], RightHsml[i], LeftNtrue[i],
                                                                  RightNtrue[i], MinNtrueDes[P[i].Type], iter);
                            }
                            else if (RightHsml[i] > 0)
                            {
                                P[i].Hsml = 0.5 * (LeftHsml[i] + RightHsml[i]);
                            }
                            else
                            {
                                P[i].Hsml =
                                    ngb_interpolator_single(LeftHsml[i], LeftNtrue[i], MinNtrueDes[P[i].Type], iter);
                            }
                            nincr_bound +=
                                ngb_check_widened_range(&P[i].Hsml, &LeftHsml[i], &RightHsml[i], MinN[P[i].Type],
                                                        LeftNweighted[i], RightNweighted[i], MaxN[P[i].Type], iter);
                        }
                    }
                    else if (P[i].TrueNGB > MaxNtrue[P[i].Type]) /* Would like to shrink hsml */
                    {
                        if ((iter >= ITER_DEBUG) || P[i].ID == MyIDType(HSML_DEBUG))
                        {
                            VERBOSE_ALL(2, "HSML_DEBUG True NGB numbers too large for i=%d task=%d ID=%llu", i,
                                        ThisTask, (unsigned long long)P[i].ID);
                        }
                        redo_particle = 1;
                        if (P[i].NumNgb < MinN[P[i].Type]) /* Weighted Ngb grow hsml (!!) */
                        {
                            nexit_nlim++;
                            redo_particle = 0; /* therefore we stop here */
                        }
                        else
                        {
                            RightHsml[i] = P[i].Hsml;
                            RightNweighted[i] = P[i].NumNgb;
                            RightNtrue[i] = P[i].TrueNGB;
                            if (LeftHsml[i] > 0 &&
                                LeftNtrue[i] < MaxNtrueDes[P[i].Type]) /* We are close to the limit, so simple
                                                                          interpolation is better ... */
                            {
                                P[i].Hsml = ngb_interpolator_dual(LeftHsml[i], RightHsml[i], LeftNtrue[i],
                                                                  RightNtrue[i], MaxNtrueDes[P[i].Type], iter);
                            }
                            else if (LeftHsml[i] > 0)
                            {
                                P[i].Hsml = 0.5 * (LeftHsml[i] + RightHsml[i]);
                            }
                            else
                            {
                                P[i].Hsml =
                                    ngb_interpolator_single(RightHsml[i], RightNtrue[i], MaxNtrueDes[P[i].Type], iter);
                            }
                            nincr_bound +=
                                ngb_check_widened_range(&P[i].Hsml, &LeftHsml[i], &RightHsml[i], MinN[P[i].Type],
                                                        LeftNweighted[i], RightNweighted[i], MaxN[P[i].Type], iter);
                        }
                    }
                    else
                    {
                        ENDRUN("This can not happen !!!!");
                    }
                }

                if (redo_particle == 1) /* Check special Cases */
                {
                    /* Running against lower bound */
                    if (P[i].Hsml < MinHsml[P[i].Type] && P[i].Hsml < hsml_old)
                    {
                        P[i].Hsml = MinHsml[P[i].Type];
                    }
                    /* Running against upper bound */
                    else if ((P[i].Hsml > MaxHsml[P[i].Type]) && P[i].Hsml > hsml_old)
                    {

                        // implement soft limit for stars
                        if (P[i].Type == type_star)
                        {
                            if (P[i].TrueNGB >= 1)
                            {
                                P[i].Hsml = hsml_old;
                                redo_particle = 0;
                                nexit_nlim++;
                                nexit_slim++;
                                hitMaxChemSpreadL++;
                            }
                        }
                        else
                        {
                            P[i].Hsml = MaxHsml[P[i].Type];
                        }
                    }

                    if (abs(hsml_old - P[i].Hsml) < P[i].Hsml * (HSML_TOLERANCE * iter)) /* Too small change in hsml */
                    {
                        nexit_dh++;
                        redo_particle = 0;
                        P[i].Hsml = hsml_old;
                    }
                }

                if (redo_particle == 1) /* need to redo this particle */
                {
                    if ((iter >= ITER_DEBUG) || P[i].ID == MyIDType(HSML_DEBUG))
                    {
                        VERBOSE_ALL(2,
                                    "HSML_DEBUG After (%d): i=%d task=%d ID=%llu Hsml=%g/%g/%d Left=%g/%g/%d "
                                    "Right=%g/%g/%d Right-Left=%g/%g",
                                    iter, i, ThisTask, (unsigned long long)P[i].ID, (float)P[i].Hsml,
                                    (float)P[i].NumNgb, P[i].TrueNGB, (float)LeftHsml[i], (float)LeftNweighted[i],
                                    LeftNtrue[i], (float)RightHsml[i], (float)RightNweighted[i], RightNtrue[i],
                                    (float)(RightHsml[i] - LeftHsml[i]), (float)abs(hsml_old - P[i].Hsml));
                    }
                    npleft++;
                    if (P[i].Type == type_gas)
                    {
                        nitergas++;
                    }
                    else if (P[i].Type == type_star)
                    {
                        niterstars++;
                    }
                    else if (P[i].Type == type_bh)
                    {
                        niterbhs++;
                    }
                }
                else
                {
                    P[i].TimeBin = -P[i].TimeBin - 1; /* Mark as inactive */
                }
            } // if
        } // for
        sumup_large_ints(1, &npleft, &ntot);

        if (ntot > 0)
        {
            iter++;

            if (iter > 0 && ThisTask == 0)
            {
                VERBOSE_ALL(2, "ngb iteration %d: need to repeat for %d%09d particles.", iter, (int)(ntot / 1000000000),
                            (int)(ntot % 1000000000));
            }

            ENDRUN_IF(iter > MAXITER, "failed to converge in neighbour iteration in density()");
        }
#ifdef ACC_HSML
        TIMER_POPPUSH(1, "HSML_FINAL", "ACC_UPDATE_WALKLITE");
        if (DoPrimaryOnGPU)
        {
#pragma omp parallel for
            for (int il = 0; il < NHsmlActivePart; il++)
            {
                int i = HsmlActiveParticleList[il];
                WalkLiteP[i].Hsml = P[i].Hsml;
                // assert(i<N_gas);
                // particle2in_hsml(&LiteDIn[i], i);
            }
#pragma omp parallel for
            for (int il = 0; il < NHsmlActivePart; il++)
            {
                int i = HsmlActiveParticleList[il];
                // WalkLiteP[i].Hsml = P[i].Hsml;
                // assert(i<N_gas);
                particle2in_hsml(&LiteDIn[i], i);
            }
        }
        if (DoPrimaryOnGPU)
        {
            myfree(LiteDOut);
        }
        myfree(HsmlActiveParticleList);

        TIMER_POPPUSH(1, "ACC_UPDATE_WALKLITE", "HSML_FINAL");
#endif // ACC_HSML

        TIMER_POP(1, "HSML_FINAL");
        TIMER_POP(2, "Final");

    } while (ntot > 0);

    TIMER_PUSH(2, "Extra");
#ifdef LT_STELLAREVOLUTION
    TIMER_PUSH(1, "HSML_STATS_MAXL");

    unsigned int tot_hitMaxChemSpreadL = 0;

    MPI_Reduce(&hitMaxChemSpreadL, &tot_hitMaxChemSpreadL, 1, MPI_UNSIGNED, MPI_SUM, 0, MYMPI_COMM_WORLD);
    if (ThisTask == 0 && tot_hitMaxChemSpreadL > 0)
    {
        VERBOSE(2, "%u particles have hit on the maximum allowed spreading hsml (%g)", tot_hitMaxChemSpreadL,
                All.MaxChemSpreadL);
    }
    TIMER_POP(1, "HSML_STATS_MAXL");
#endif // LT_STELLAREVOLUTION

    TIMER_PUSH(1, "HSML_UNMARK");
    /* mark as active again */
#ifdef _OPENMP
#pragma omp parallel for
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP
        P[i].TimeBin = ((P[i].TimeBin < 0) ? -P[i].TimeBin - 1 : P[i].TimeBin);

        if (P[i].ID == MyIDType(HSML_DEBUG))
        {
            if (P[i].Type == type_gas)
            {
                VERBOSE_ALL(2, "HSML_DEBUG On exit: i=%d task=%d ID=%llu Hsml=%g NGB=%g tNGB=%d rho=%g", i, ThisTask,
                            (unsigned long long)P[i].ID, (float)P[i].Hsml, (float)P[i].NumNgb, P[i].TrueNGB,
                            (float)SphP[i].Density);
            }
            else
            {
                VERBOSE_ALL(2, "HSML_DEBUG On exit: i=%d task=%d ID=%llu Hsml=%g NGB=%g tNGB=%d", i, ThisTask,
                            (unsigned long long)P[i].ID, (float)P[i].Hsml, (float)P[i].NumNgb, P[i].TrueNGB);
            }
        }

        if (HSML_DEBUG >= 0)
        {
            if (P[i].Type == type_gas)
                if (SphP[i].Density > 1e5)
                {
                    VERBOSE_ALL(
                        2, "HSML_ALARM: strange density on exit i=%d task=%d ID=%llu Hsml=%g NGB=%g tNGB=%d rho=%g", i,
                        ThisTask, (unsigned long long)P[i].ID, (float)P[i].Hsml, (float)P[i].NumNgb, P[i].TrueNGB,
                        (float)SphP[i].Density);
                }
        }
    }

#ifdef ACC_HSML
    myfree(LiteDIn);
#endif // ACC_HSML

    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(RightNtrue);
    myfree(LeftNtrue);
    myfree(RightNweighted);
    myfree(LeftNweighted);
    myfree(RightHsml);
    myfree(LeftHsml);
    myfree(RecyclingBuffer);
    myfree(Ngblist);

    TIMER_POPPUSH(1, "HSML_UNMARK", "HSML_STATS_EXIT");

    long long nitergastot, niterstarstot, niterbhstot, NActivePartTot = 0;
    sumup_large_ints(1, &nitergas, &nitergastot);
    sumup_large_ints(1, &niterstars, &niterstarstot);
    sumup_large_ints(1, &niterbhs, &niterbhstot);
#ifdef _OPENMP
    sumup_large_ints(1, &NActivePart, &NActivePartTot);
#endif // _OPENMP

    int nexit_hfit_tot, nexit_dh_tot, nexit_nlim_tot, nexit_slim_tot, nexit_hwin_tot, nincr_bound_tot;
    MPI_Allreduce(&nexit_hfit, &nexit_hfit_tot, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&nexit_dh, &nexit_dh_tot, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&nexit_nlim, &nexit_nlim_tot, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&nexit_slim, &nexit_slim_tot, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&nexit_hwin, &nexit_hwin_tot, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&nincr_bound, &nincr_bound_tot, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    VERBOSE(2,
            "HSML SUMMARY: t=%g iter=%d act=%llu n(gas/star/bh)=(%llu/%llu/%llu) "
            "exit(hfit/dh/nlim/slim/hwin)=(%d/%d/%d/%d/%d) enlarge=%d",
            All.Time, iter, NActivePartTot, nitergastot, niterstarstot, niterbhstot, nexit_hfit_tot, nexit_dh_tot,
            nexit_nlim_tot, nexit_slim_tot, nexit_hwin_tot, nincr_bound_tot);
    TIMER_POP(1, "HSML_STATS_EXIT");
    TIMER_POP(2, "Extra");
    TIMER_POP(1, "FIND_HSML");
}

#endif // ifndef ACC_INCLUDE_HSML

#ifdef ACC_INCLUDE_HSML
#warning "we are entering a gpu only compiled code"
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Evaluate the local smoothing length (HSML) for a particle.
 *
 * This function computes the local smoothing length (HSML) for a particle specified by the target index. The local HSML
 * is determined based on the particle's neighbors within the specified search radius. The target particle may either be
 * local, or reside in the communication buffer. The computation can be performed in different modes:
 *
 * - Mode 0: Regular evaluation of local HSML.
 * - Mode 1: Special mode for handling active particles.
 * - Mode 2: Special mode for handling target particles.
 * - Mode 3: Count the number of neighbors within the search radius without evaluating local HSML.
 *
 * The function iterates over the neighbors of the target particle and calculates properties relevant to the
 * determination of the local HSML.
 *
 * @param active_target The index of the active target particle for which to evaluate local HSML.
 * @param p_target The index of the primary target particle for which to evaluate local HSML.
 * @param mode The evaluation mode (0, 1, 2, or 3).
 * @param ngblist An array containing the indices of neighbors of the target particle.
 * @param numngb The number of neighbors of the target particle.
 * @param HsmlDataGet An array of input data structures containing information about the neighbors.
 * @param HsmlDataResult An array of output data structures to store the results of local HSML evaluation.
 * @param GP An array of gravity walk data structures.
 * @param P An array of input data structures containing particle information.
 * @param i_node The node index for the target particle.
 * @param node The node index.
 * @return An integer indicating the success of the local HSML evaluation process.
 */
inline int hsml_evaluate_local(int active_target, int p_target, int mode, int *ngblist, int numngb,
                               struct hsmldata_in *HsmlDataGet, struct hsmldata_out *HsmlDataResult,
                               struct gravity_walkdata *GP, struct hsmldata_in *P, int i_node, int node)
#else // ACC_INCLUDE_HSML
/**
 * @brief Evaluate the smoothing length (HSML) for a particle.
 *
 * This function calculates the smoothing length (HSML) for a particle specified by the target index. The HSML is
 * evaluated based on the particle's neighbors within the specified search radius. The target particle may either be
 * local, or reside in the communication buffer. The computation can be performed in different modes:
 * - Mode 0: Regular evaluation of HSML.
 * - Mode 1: Special mode for handling active particles.
 * - Mode 2: Special mode for handling target particles.
 * - Mode 3: Count the number of neighbors within the search radius without evaluating HSML.
 *
 * The function iterates over the neighbors of the target particle and computes properties such as density, pressure,
 * and gradient of velocity divergence.
 *
 * @param target The index of the target particle for which to evaluate HSML.
 * @param mode The evaluation mode (0, 1, 2, or 3).
 * @param ngblist An array containing the indices of neighbors of the target particle.
 * @param extended_numngb_inbox The number of extended neighbors in the inbox.
 * @param i_startnode The start node index for the neighbor search.
 * @param node The node index.
 * @param i_node The node index for the target particle.
 * @param additional_data Additional data pointer.
 * @return An integer indicating the success of the evaluation process.
 */
inline int hsml_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node,
                         int i_node, void *additional_data)
#endif // ACC_INCLUDE_HSML
{
#ifdef ACC_INCLUDE_HSML
    int target;
    int numngb_inbox;
    int extended_numngb_inbox = numngb;
    int startnode;
#else // ACC_INCLUDE_HSML
    int startnode, numngb_inbox;
#endif // ACC_INCLUDE_HSML

    MyAtLeastDouble h2, u;
    struct kernel_density kernel;
    struct hsmldata_in local;
    hsmldata_out out{};

#if ACC_INCLUDE_HSML // BECOUSE OPENACC MUST BE THE WEIRDO WHO HAS NOT STANDARD ROUTINES
    memset_gpu((unsigned char *)&out, 0, sizeof(struct hsmldata_out));
#endif // ACC_INCLUDE_HSML

#if ACC_INCLUDE_HSML
    struct hsmldata_in *SphP = P;
    if (mode == 2)
    {
        target = p_target;
    }
    if (mode == 1)
    {
        target = active_target;
    }
#endif // ACC_INCLUDE_HSML

    if (mode == 0 || mode == 2 || mode == 3)
    {
#ifdef ACC_INCLUDE_HSML
        memcpy_gpu(&local, &P[target], sizeof(struct hsmldata_in));
        // local = P[target];
#else // ACC_INCLUDE_HSML
        particle2in_hsml(&local, target);
#endif // ACC_INCLUDE_HSML
    }
    else
    {
#ifdef ACC_INCLUDE_HSML
        memcpy_gpu(&local, &HsmlDataGet[target], sizeof(struct hsmldata_in));
#else // ACC_INCLUDE_HSML
        local = HsmlDataGet[target];
#endif // ACC_INCLUDE_HSML
    }

    h2 = local.Hsml * local.Hsml;
    kernel_hinv(local.Hsml, &kernel.hinv, &kernel.hinv3, &kernel.hinv4);

    startnode = node;
    // int initial_startnode = startnode;

#ifdef ACC_INCLUDE_HSML
    short check = i_node > 0;
    if (check)
    {
        memcpy_gpu(&out, &HsmlDataResult[active_target], sizeof(struct hsmldata_out));
    }
#else // ACC_INCLUDE_HSML

    if (i_node > 0)
    {
        out = HsmlDataResult[target];
    } // end if(i_node > 0)
#endif // ACC_INCLUDE_HSML

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            // better to keep those preprocessing stuff indented..

            numngb_inbox = extended_numngb_inbox;
#ifndef ACC_INCLUDE_HSML
            startnode = *i_startnode;
#else // ACC_INCLUDE_HSML
            startnode = -1;
#endif // ACC_INCLUDE_HSML

#if defined(DENSITY_LESS_NGB_FACTOR) && !defined(ACC_INCLUDE_HSML)
            ENDRUN_IF(
                numngb_inbox > NumPart / DENSITY_LESS_NGB_FACTOR,
                "hsml: Number of neighbours found (%d) exceeds allowed number (%d) derived from "
                "NumPart/DENSITY_LESS_NGB_FACTOR, to continue, decreas DENSITY_LESS_NGB_FACTOR (currently %f, min "
                "possible val would be 1) !!",
                numngb_inbox, (int)(NumPart / DENSITY_LESS_NGB_FACTOR), (double)DENSITY_LESS_NGB_FACTOR);
#endif // defined(DENSITY_LESS_NGB_FACTOR) && !defined(ACC_INCLUDE_HSML)

#ifdef ACC_HSML
            if (mode == 3)
            {
                return numngb_inbox;
            }
#endif // ACC_HSML
            if (numngb_inbox < 0)
            {
                return -1;
            }

            for (int n = 0; n < numngb_inbox; n++)
            {
                int j = ngblist[n];

                // distance test first:
                // the tree-walk box search admits candidates outside the true kernel sphere, so reject those before
                // touching mass/wind/id fields of j below
                kernel.dx = local.Pos[0] - P[j].Pos[0];
                kernel.dy = local.Pos[1] - P[j].Pos[1];
                kernel.dz = local.Pos[2] - P[j].Pos[2];

#ifdef PERIODIC /*  now find the closest image in the given box size  */
                kernel.dx = NEAREST_X(kernel.dx);
                kernel.dy = NEAREST_Y(kernel.dy);
                kernel.dz = NEAREST_Z(kernel.dz);
#endif // PERIODIC
                MyAtLeastDouble r2 = kernel.dx * kernel.dx + kernel.dy * kernel.dy + kernel.dz * kernel.dz;
                if (r2 >= h2)
                {
                    continue;
                }

                MyAtLeastDouble mass_j;
                //		assert(j<N_gas);
#ifdef ACC_INCLUDE_HSML
                mass_j = GP[j].Mass;
#else // ACC_INCLUDE_HSML

                mass_j = P[j].Mass;

#endif // ACC_INCLUDE_HSML

#ifdef WINDS
#if defined(LT_STELLAREVOLUTION) || defined(AXION_DM) || defined(BLACK_HOLES)
                if (local.Type == type_gas && SphP[j].DelayTime > 0 &&
                    !(local.DelayTime > 0)) /* Wind in density only for wind particles */
                {
                    continue;
                }
                if (local.Type == type_bh && SphP[j].DelayTime > 0) /* Wind does not contribute to density for BHs */
                {
                    continue;
                }
#ifndef DONOT_EXCLUDE_WINDS_FROM_FEEDBACK /* Exclude wind from density for Stars */
                if (local.Type == type_star && SphP[j].DelayTime > 0)
                {
                    continue;
                }
#endif // DONOT_EXCLUDE_WINDS_FROM_FEEDBACK
#else // defined(LT_STELLAREVOLUTION) || defined(AXION_DM) || defined(BLACK_HOLES)
                if (SphP[j].DelayTime > 0 && !(local.DelayTime > 0)) /* Wind in density only for wind particles */
                {
                    continue;
                }
#endif // defined(LT_STELLAREVOLUTION) || defined(AXION_DM) || defined(BLACK_HOLES)
#endif // WINDS

#ifdef BLACK_HOLES
                if (mass_j == 0)
                {
                    continue;
                }
#endif // BLACK_HOLES
#ifdef FG_REMOVE_BND
                if (P[j].ID == 0)
                {
                    continue;
                }
#endif // FG_REMOVE_BND

                out.TrueNGB++;
                kernel.r = sqrt(r2);
                u = kernel.r * kernel.hinv;

                kernel_main(u, kernel.hinv3, kernel.hinv4, &kernel.wk, &kernel.dwk, 0);
                kernel.mj_wk = mass_j * kernel.wk;

                kernel.mj_dwk_r = mass_j * kernel.dwk / kernel.r;

                kernel.dvx = local.VelPred[0] - SphP[j].VelPred[0];
                kernel.dvy = local.VelPred[1] - SphP[j].VelPred[1];
                kernel.dvz = local.VelPred[2] - SphP[j].VelPred[2];
#ifdef DISK_SLICE
                kernel.dvz -= kernel.dv_phi_disk;
#endif // DISK_SLICE
#ifdef GT_DUST
                kernel.dvx_mixture = kernel.dvx;
                kernel.dvy_mixture = kernel.dvy;
                kernel.dvz_mixture = kernel.dvz;
                kernel.dvx -= local.TotalDustFracPred * local.TotalDeltaVelPred[0] -
                              SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[0];
                kernel.dvy -= local.TotalDustFracPred * local.TotalDeltaVelPred[1] -
                              SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[1];
                kernel.dvz -= local.TotalDustFracPred * local.TotalDeltaVelPred[2] -
                              SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[2];
#endif // GT_DUST
       // first order calculation to be used in compute_gradients
                if (kernel.r > 0)
                {
                    out.DivVel +=
                        -kernel.mj_dwk_r * (kernel.dvx * kernel.dx + kernel.dvy * kernel.dy + kernel.dvz * kernel.dz);
#ifdef GT_DUST
                    out.DivVelMixture +=
                        -kernel.mj_dwk_r * (kernel.dvx_mixture * kernel.dx + kernel.dvy_mixture * kernel.dy +
                                            kernel.dvz_mixture * kernel.dz);
#endif // GT_DUST
                }

#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
                out.Vbulk[0] += kernel.mj_wk * SphP[j].VelPred[0];
                out.Vbulk[1] += kernel.mj_wk * SphP[j].VelPred[1];
                out.Vbulk[2] += kernel.mj_wk * SphP[j].VelPred[2];
#ifdef DISK_SLICE
                out.Vbulk[2] += kernel.mj_wk * kernel.dv_phi_disk;
#endif // DISK_SLICE
#endif // JD_VTURB || KD_LIMIT_DIVV

#if GADGET_HYDRO == HYDRO_SPH
                out.Rho += kernel.mj_wk;
#elif GADGET_HYDRO == HYDRO_PESPH
                out.NumDens += kernel.wk;
                out.Pressure += pow((MyAtLeastDouble)SphP[j].Entropy, GAMMA_INV) * kernel.mj_wk;
#elif GADGET_HYDRO == HYDRO_MFM
                out.Rho += kernel.mj_wk;
                out.NumDens += kernel.wk;
#endif // GADGET_HYDRO

                out.NumNgb += kernel.wk; // scaling is done in final operations

#if GADGET_HYDRO == HYDRO_SPH
                out.DhsmlDensity += -mass_j * (NUMDIMS * kernel.hinv * kernel.wk + u * kernel.dwk);
#elif GADGET_HYDRO == HYDRO_PESPH
                out.DhsmlNumDens += -1.0 * (NUMDIMS * kernel.hinv * kernel.wk + u * kernel.dwk);
                out.DhsmlPressure += -mass_j * pow((MyAtLeastDouble)SphP[j].Entropy, GAMMA_INV) *
                                     (NUMDIMS * kernel.hinv * kernel.wk + u * kernel.dwk);
#elif GADGET_HYDRO == HYDRO_MFM
                out.DhsmlNumDens += -1.0 * (NUMDIMS * kernel.hinv * kernel.wk + u * kernel.dwk);
#endif // GADGET_HYDRO
            }
        }
    }
#ifdef ACC_INCLUDE_HSML
    target = active_target;
#endif // ACC_INCLUDE_HSML

    if (mode == 3)
    {
        return 0;
    }

    if (mode == 0)
    {
#ifndef ACC_INCLUDE_HSML
        out2particle_hsml(&out, target, 0);
#endif // ACC_INCLUDE_HSML
    }
    else
    {
#ifdef ACC_INCLUDE_HSML

        memcpy_gpu(&HsmlDataResult[target], &out, sizeof(struct hsmldata_out));
#else // ACC_INCLUDE_HSML
        HsmlDataResult[target] = out;
#endif // ACC_INCLUDE_HSML
    }
    return 0;
}
#ifdef ACC_INCLUDE_HSML
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HSML

#ifndef ACC_INCLUDE_HSML
/**
 * @brief Check if a particle with the given index is active based on its time bin.
 *
 * This function checks if a particle with the given index is active based on its time bin. A particle is considered
 * active if its time bin is non-negative and it meets one of the following conditions:
 *
 * - It is alive as determined by `IS_SPH_ALIVE`.
 * - It is active as a star particle as determined by `IS_STAR_ACTIVE`.
 * - It is a black hole particle as determined by `IS_BLACKHOLE`.
 *
 * @param n The index of the particle to check.
 * @return An integer representing whether the particle is active (1) or not (0).
 */
inline int hsml_isactive(int n)
{
    if (P[n].TimeBin < 0)
    {
        return 0;
    }

    int active = IS_SPH_ALIVE(P[n]);
    active += IS_STAR_ACTIVE(P, n);
    active += IS_BLACKHOLE(P[n]);

    return active;
}

#endif // ACC_INCLUDE_HSML

#ifdef ACC_HSML
#if !defined(ACC_INCLUDE_HSML) && defined(ACC)
#warning "re including everything"
#define ACC_INCLUDE_HSML 1
#include "find_hsml.cpp"
#undef ACC_INCLUDE_HSML
#endif // !defined(ACC_INCLUDE_HSML)  && defined(ACC)
#endif // ACC_HSML
