/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief SPH density computation and smoothing length determination
 *
 * This file contains the "first SPH loop", where the SPH densities and some
 * auxiliary quantities are computed.  There is also functionality that
 * corrects the smoothing length if needed.
 *
 * @author OpenGadget3-development team, in particular Ludwig Böss, Klaus Dolag, Frederick Groth and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */


#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/greentree.hpp"
#include "../CodeBase/proto.hpp"

#include "../Mfm/Matrix.hpp"
#include "../System/communication.hpp"
#include "kernel.hpp"

#ifdef TIME_DEP_ART_VISC
#include "viscosity.hpp"
#endif // TIME_DEP_ART_VISC

#ifndef ACC_INCLUDE_GRADIENTS // we reinclude this same file if ACC is active
// int DoSecondaryOnGPU;

#ifdef LT_STELLAREVOLUTION
int LastActive, SaveFirstActiveParticle;
#endif // LT_STELLAREVOLUTION

#if defined(ACC_GRADIENTS) && defined(LT_STELLAREVOLUTION)
int WalCool_n[6];
int *SpeciesIdx;
int *SpeciesPos;
#endif

/**
 * @brief Structure for properties only needed within this function.
 */
static struct grad_store_out
{
#if (GADGET_HYDRO == HYDRO_MFM)
    Vector<NUMDIMS> gradWSph[numvar]; /*!< SPH computation of primitive gradients */
    MyFloat neighbour_mass;
#ifdef GT_DUST
    std::array<std::array<Vector<NUMDIMS>, numvar>, GT_NUM_DUST> gradWSph_dust;
#endif // GT_DUST
#endif // (GADGET_HYDRO == HYDRO_MFM)
    SymmetricMatrix<NUMDIMS, MyAtLeastDouble> DistanceMatrix;
    Vector<NUMDIMS, MyAtLeastDouble> dVelMatrix[3];
#ifdef MAGNETIC
    Vector<NUMDIMS, MyAtLeastDouble> dBfldMatrix[3];
#endif // MAGNETIC
#ifdef GT_DUST
    Vector<NUMDIMS, MyAtLeastDouble> dVelMixtureMatrix[NUMDIMS];
#endif // GT_DUST
} *GradStoreOut;

/**
 * @brief Structure for communication during the density computation. Holds data that is sent to other processors.
 */
static struct graddata_in
{
    MyLongDouble Pos[3];
#ifdef ACC_GRADIENTS
    MyFloat Pressure;
#if (defined(SFR) && defined(MAGNETIC)) || defined(LT_STELLAREVOLUTION)
    MyFloat XColdCloud;
#endif // (defined(SFR) && defined(MAGNETIC)) || defined(LT_STELLAREVOLUTION)
#ifdef LT_STELLAREVOLUTION
    MyFloat Temperature;
    MyFloat Metals[LT_NMetP]; /*!< H is not stored here */
#endif // LT_STELLAREVOLUTION
    MyFloat VelPred[3];
#endif // ACC_GRADIENTS
    MyFloat Vel[3];
#ifdef ACC_GRADIENTS
#ifdef BLACK_HOLES
    short int TimeBin;
    MyFloat Mass;
#endif // BLACK_HOLES
    MyFloat Entropy;
    MyFloat DivVel; /*!< local velocity divergence, at this point only first order */
#endif // ACC_GRADIENTS

    MyFloat Hsml;
#ifdef WINDS
    MyFloat DelayTime;
#endif // WINDS
#ifdef MAGNETIC
    MyFloat BPred[3];
#endif // MAGNETIC
#if defined(SHOCKFINDER) && !defined(ACC_GRADIENTS)
    MyFloat Pressure;
#endif // SHOCKFINDER

#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
    MyFloat Vbulk[3];
#endif // JD_VTURB || KD_LIMIT_DIVV

#ifdef TIME_DEP_ART_COND
    MyFloat Density; /*!< local density */
    MyFloat EntropyPred;
#if !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0) && defined(TIME_DEP_ART_COND_GSMOOTH)
    MyAtLeastDouble GravAccel[3]; /*!< particle acceleration due to gravity */

#endif // !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0) && defined(TIME_DEP_ART_COND_GSMOOTH)
#elif defined(ACC_GRADIENTS) // TIME_DEP_ART_COND
    MyFloat Density; /*!< local density */
#endif // TIME_DEP_ART_COND
#if defined(LT_STELLAREVOLUTION) || defined(AXION_DM) || defined(BLACK_HOLES)
    int Type;
#endif // defined(LT_STELLAREVOLUTION) || defined(AXION_DM) || defined(BLACK_HOLES)
#if (GADGET_HYDRO == HYDRO_MFM)
    MyFloat Volume;
    WFluidVector<mfm_numdims> Wprim;
#ifdef GT_DUST
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> Wprim_dust;
#endif // GT_DUST
#endif // (GADGET_HYDRO == HYDRO_MFM)
#ifdef GT_DUST
    MyFloat DustFracPred[GT_NUM_DUST];
    MyFloat DeltaVelPred[GT_NUM_DUST][NUMDIMS];
    MyFloat TotalDustFracPred;
    MyFloat TotalDeltaVelPred[NUMDIMS];
#endif // GT_DUST
} *GradDataIn, *GradDataGet;

/**
 * @brief Structure for storing various computed gradients and related quantities.
 */
static struct graddata_out
{
#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
    MyAtLeastDouble Vrms; /*!< RMS velocity inside kernel around Vbulk */
#if defined(JD_VTURB) && defined(JD_DECOMPOSE_VTURB)
    MyAtLeastDouble Vrad;
    MyAtLeastDouble Vtan;
#endif // JD_VTURB && JD_DECOMPOSE_VTURB
#endif // JD_VTURB || KD_LIMIT_DIVV

#ifdef SHOCKFINDER
    MyAtLeastDouble Shock_N1[3];
#endif // SHOCKFINDER
#ifdef TIME_DEP_ART_COND
    MyAtLeastDouble GradA[3];
#if !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0) && defined(TIME_DEP_ART_COND_GSMOOTH)
    MyAtLeastDouble Cgrav[3];
#endif // !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0) && defined(TIME_DEP_ART_COND_GSMOOTH)
#endif // TIME_DEP_ART_COND
#ifdef TIME_DEP_ART_VISC
#if (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
    MyAtLeastDouble visc_R;
#endif // (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
#endif // TIME_DEP_ART_VISC

    // Matrices for second order gradient estimate
    SymmetricMatrix<NUMDIMS, MyAtLeastDouble> DistanceMatrix;
    Vector<NUMDIMS, MyAtLeastDouble> dVelMatrix[3];

#ifdef MAGNETIC
    Vector<NUMDIMS, MyAtLeastDouble> dBfldMatrix[3];

    MyAtLeastDouble divB;
#if (defined(TIME_DEP_MAGN_DISP) && (TIME_DEP_MAGN_DISP == 1)) || defined(OUTPUT_ROTB)
    MyAtLeastDouble RotB[3];
#endif // (defined(TIME_DEP_MAGN_DISP) && (TIME_DEP_MAGN_DISP == 1)) || defined(OUTPUT_ROTB)

#ifdef OUTPUT_MAGNETIC_CURVATURE
    MyAtLeastDouble MagCurv[3];
#endif // OUTPUT_MAGNETIC_CURVATURE
#endif // MAGNETIC

#ifdef RADTRANSFER_FLUXLIMITER
    MyAtLeastDouble Grad_ngamma[3][N_BINS];
#endif // RADTRANSFER_FLUXLIMITER

#ifdef CONDUCTION_SATURATION
    MyAtLeastDouble GradEntr[3];
#endif // CONDUCTION_SATURATION


#ifdef LT_STELLAREVOLUTION
    MyAtLeastDouble weight;
#endif // LT_STELLAREVOLUTION

#ifdef BLACK_HOLES
    MyAtLeastDouble ColdRho;
    MyAtLeastDouble HotRho;
    MyAtLeastDouble Metallicity_mjwk;
    MyAtLeastDouble Temp_mjwk;
    MyAtLeastDouble P_ext_mjwk;
    MyAtLeastDouble M_fb;
    MyAtLeastDouble rho_SFR;
    MyAtLeastDouble SmoothedEntr;
    MyAtLeastDouble SmoothedColdEntr;
    MyAtLeastDouble SmoothedHotEntr;
    int BH_TimeBinGasNeighbor;
    MyAtLeastDouble GasVel[3];
    MyAtLeastDouble ColdGasVel[3];
    MyAtLeastDouble HotGasVel[3];
#ifdef KD_BH_USE_TOPHAT_FOR_RADIO_MODE_AD_MASSWEIGHTS
    MyAtLeastDouble TotalSurrMass;
#endif
#endif // BLACK_HOLES


#if (defined(GM_MV_ANGMOM) && defined(MV_GM_AGNMUPPI)) || defined(LS_BLACK_HOLES_SPIN_EVOLUTION)
    MyAtLeastDouble Vphi[3], VphiMod;
#endif // defined(GM_MV_ANGMOM) && defined(MV_GM_AGNMUPPI) || defined(LS_BLACK_HOLES_SPIN_EVOLUTION)

#if (GADGET_HYDRO == HYDRO_MFM)
    SymmetricMatrix<NUMDIMS> Ematrix;
    Vector<NUMDIMS> gradW[numvar];
    Vector<NUMDIMS> gradWSph[numvar];
#if MFM_SLOPE_LIMITER == LIMITER_AREPO || MFM_SLOPE_LIMITER == LIMITER_SCALAR || MFM_SLOPE_LIMITER == LIMITER_GIZMO
    WFluidVector<mfm_numdims> Wmin;
    WFluidVector<mfm_numdims> Wmax;
#endif // MFM_SLOPE_LIMITER == LIMITER_AREPO || MFM_SLOPE_LIMITER == LIMITER_SCALAR || MFM_SLOPE_LIMITER ==
       // LIMITER_GIZMO
#if MFM_SLOPE_LIMITER == LIMITER_SCALAR || MFM_SLOPE_LIMITER == LIMITER_GIZMO
    MyFloat DistNgbSqdMax;
#endif // MFM_SLOPE_LIMITER == LIMITER_SCALAR || MFM_SLOPE_LIMITER == LIMITER_GIZMO
    Vector<mfm_numdims> Vel_smooth;
    MyAtLeastDouble neighbour_mass;
    MyFloat MaxVelSquareNgb;
#ifdef GT_DUST
    std::array<std::array<Vector<NUMDIMS>, numvar>, GT_NUM_DUST> gradW_dust;
    std::array<std::array<Vector<NUMDIMS>, numvar>, GT_NUM_DUST> gradWSph_dust;
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> Wmin_dust;
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> Wmax_dust;
#endif // GT_DUST
#endif // (GADGET_HYDRO == HYDRO_MFM)
#ifdef GT_DUST
    Vector<NUMDIMS, MyAtLeastDouble> dVelMixtureMatrix[NUMDIMS];
#ifdef GT_DUST_DIFFUSION
    MyAtLeastDouble dust_fraction_smooth[GT_NUM_DUST];
    MyAtLeastDouble neigh_mass;
#endif // GT_DUST_DIFFUSION
#endif // GT_DUST
} *GradDataResult, *GradDataOut;

#ifdef ACC_GRADIENTS
void particle2in_gradients(struct base_graddata_in *in, int i);
void gradients_evaluate_extra_physics_gas(struct base_graddata_in *local, struct graddata_out *out,
                                          struct kernel_density *kernel, int j, int gastype);
ACC_DECLARE_ROUTINE_BEFORE
void gradients_evaluate_extra_physics_gas_gpu(struct graddata_in *local, struct graddata_out *out,
                                              struct kernel_density *kernel, int j, MyAtLeastDouble mass_j,
                                              struct graddata_in *SphP);
ACC_DECLARE_ROUTINE_AFTER
#else // ACC_GRADIENTS
void particle2in_gradients(struct graddata_in *in, int i);
void out2particle_gradients(struct graddata_out *out, int i, int mode);
#endif // ACC_GRADIENTS

/**
 * @brief Copies particle properties to the input structure for gradient computation.
 *
 * This function copies relevant properties from particle data to the input structure
 * used for computing gradients.
 *
 * @param in Pointer to the input structure for gradient computation.
 * @param i Index of the particle whose properties are being copied.
 */
void particle2in_gradients(struct graddata_in *in, int i)
{
    for (int k = 0; k < 3; k++)
    {
        in->Pos[k] = P[i].Pos[k];
    }

    in->Hsml = P[i].Hsml;

#if defined(LT_STELLAREVOLUTION) || defined(AXION_DM) || defined(BLACK_HOLES)
    in->Type = P[i].Type;
#endif //  defined(LT_STELLAREVOLUTION) || defined(AXION_DM) || defined(BLACK_HOLES)

    if (P[i].Type != 0)
    { // this automatically means we are dealing with stars or black holes
        for (int k = 0; k < 3; k++)
        {
            in->Vel[k] = P[i].Vel[k];
        }
    }
    else
    {
        for (int k = 0; k < 3; k++)
        {
            in->Vel[k] = SphP[i].VelPred[k];
#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
            in->Vbulk[k] = SphP[i].Vbulk[k];
#endif // JD_VTURB || KD_LIMIT_DIVV
        }
    }

#ifdef ACC_GRADIENTS
    if (P[i].Type == type_gas)
    {
        in->Pressure = SphP[i].Pressure;
#if (defined(SFR) && defined(MAGNETIC)) || defined(LT_STELLAREVOLUTION)
        in->XColdCloud = SphP[i].XColdCloud;
#endif // (defined(SFR) && defined(MAGNETIC)) || defined(LT_STELLAREVOLUTION)
#ifdef LT_STELLAREVOLUTION
        for (int k = 0; k < LT_NMetP; k++)
        {
            in->Metals[k] = SphP[i].Metals[k];
        }
        in->Temperature = SphP[i].Temperature;

#endif // LT_STELLAREVOLUTION
        for (int k = 0; k < 3; k++)
        {
            in->VelPred[k] = SphP[i].VelPred[k];
        }
        in->Entropy = SphP[i].Entropy;
        in->DivVel = SphP[i].DivVel;
        in->Density = SphP[i].Density;
    }
#ifdef BLACK_HOLES
    in->TimeBin = P[i].TimeBin;
    in->Mass = P[i].Mass;
#endif // BLACK_HOLES
#endif // ACC_GRADIENTS

    if (P[i].Type == type_gas)
    {
#ifdef TIME_DEP_ART_COND
        in->Density = SphP[i].Density;
        in->EntropyPred = SphP[i].EntropyPred;
#if !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0) && defined(TIME_DEP_ART_COND_GSMOOTH)
        for (int k = 0; k < 3; k++)
        {
            in->GravAccel[k] = P[i].GravAccel[k];
        }
#endif // !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0) && defined(TIME_DEP_ART_COND_GSMOOTH)
#endif // TIME_DEP_ART_COND
#if defined(SHOCKFINDER) && !defined(ACC_GRADIENTS)
        in->Pressure = SphP[i].Pressure;
#endif // SHOCKFINDER

#ifdef WINDS
        in->DelayTime = SphP[i].DelayTime;
#endif // WINDS

#ifdef MAGNETIC
        for (int k = 0; k < 3; k++)
        {
#ifdef SFR
            in->BPred[k] = SphP[i].BPred[k] * pow(cbrt(1.0 - SphP[i].XColdCloud), 2);
#else // SFR
            in->BPred[k] = SphP[i].BPred[k];
#endif // SFR
        }
#endif // MAGNETIC

#if (GADGET_HYDRO == HYDRO_MFM)
        in->Volume = MyFloat(1.0) / SphP[i].NumDens;
        for (int var = 0; var < numvar; var++)
        {
            in->Wprim[var] = SphP[i].Wprim[var];
        }
#ifdef GT_DUST
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            for (int var = 0; var < numvar; var++)
            {
                in->Wprim_dust[idust][var] = SphP[i].Wprim_dust[idust][var];
            }
        }
#endif // GT_DUST
#endif // (GADGET_HYDRO == HYDRO_MFM)
#ifdef GT_DUST
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            in->DustFracPred[idust] = SphP[i].DustFracPred[idust];
            for (int k = 0; k < NUMDIMS; k++)
            {
                in->DeltaVelPred[idust][k] = SphP[i].DeltaVelPred[idust][k];
            }
        }
        in->TotalDustFracPred = SphP[i].TotalDustFracPred;
        for (int k = 0; k < NUMDIMS; k++)
        {
            in->TotalDeltaVelPred[k] = SphP[i].TotalDeltaVelPred[k];
        }
#endif // GT_DUST
    }
    else
    {
#ifdef WINDS
        in->DelayTime = 0;
#endif // WINDS
#if (GADGET_HYDRO == HYDRO_MFM)
        // Set to reasonable numbers, should not be used for non Gas particles
        in->Volume = 1;
        for (int var = 0; var < numvar; var++)
        {
            in->Wprim[var] = 0;
#ifdef GT_DUST
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                in->Wprim_dust[idust][var] = 0;
            }
#endif // GT_DUST
        }
#endif // (GADGET_HYDRO == HYDRO_MFM)
    }
}

/**
 * @brief Copies computed gradients to particle properties.
 *
 * This function copies computed gradients from the output structure to particle properties.
 *
 * @param out Pointer to the output structure containing computed gradients.
 * @param i Index of the particle.
 * @param mode Specifies whether to assign or add the gradients to particle properties.
 */
void out2particle_gradients(struct graddata_out *out, int i, int mode)
{
    if (P[i].Type == type_gas)
    {

#ifdef CONDUCTION_SATURATION
        for (int k = 0; k < 3; k++)
        {
            ASSIGN_ADD(SphP[i].GradEntr[k], out->GradEntr[k], mode);
        }
#endif // CONDUCTION_SATURATION



#ifdef RADTRANSFER_FLUXLIMITER
        for (int k = 0; k < N_BINS; k++)
        {
            for (int j = 0; j < 3; j++)
            {
                ASSIGN_ADD(SphP[i].Grad_ngamma[j][k], out->Grad_ngamma[j][k], mode);
            }
        }
#endif // RADTRANSFER_FLUXLIMITER

#ifdef TIME_DEP_ART_COND
        for (int k = 0; k < 3; k++)
        {
            ASSIGN_ADD(SphP[i].GradA[k], out->GradA[k], mode);
        }
#if !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0) && defined(TIME_DEP_ART_COND_GSMOOTH)
        for (int k = 0; k < 3; k++)
        {
            ASSIGN_ADD(SphP[i].Cgrav[k], out->Cgrav[k], mode);
        }
#endif // !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0) && defined(TIME_DEP_ART_COND_GSMOOTH)
#endif // TIME_DEP_ART_COND

#ifdef SHOCKFINDER
        for (int k = 0; k < 3; k++)
        {
            ASSIGN_ADD(SphP[i].Shock_N1[k], out->Shock_N1[k], mode);
        }
#endif // SHOCKFINDER

#ifdef TIME_DEP_ART_VISC
#if (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
        ASSIGN_ADD(SphP[i].visc_R, out->visc_R, mode);
#endif // (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2)  || (TIME_DEP_ART_VISC == 3)
#endif // TIME_DEP_ART_VISC

#if (defined(TIME_DEP_MAGN_DISP) && (TIME_DEP_MAGN_DISP == 1)) || defined(OUTPUT_ROTB)
        for (int k = 0; k < 3; k++)
        {
            ASSIGN_ADD(SphP[i].RotB[k], out->RotB[k], mode);
        }
#endif // (defined(TIME_DEP_MAGN_DISP) && (TIME_DEP_MAGN_DISP == 1)) || defined(OUTPUT_ROTB)

        ASSIGN_ADD(GradStoreOut[i].DistanceMatrix, out->DistanceMatrix, mode);

#ifdef OUTPUT_MAGNETIC_CURVATURE
        for (int k = 0; k < 3; k++)
        {
            ASSIGN_ADD(SphP[i].MagCurv[k], out->MagCurv[k], mode);
        }
#endif // OUTPUT_MAGNETIC_CURVATURE

        for (int j = 0; j < 3; ++j)
        {
            ASSIGN_ADD(GradStoreOut[i].dVelMatrix[j], out->dVelMatrix[j], mode);
#ifdef MAGNETIC
            ASSIGN_ADD(GradStoreOut[i].dBfldMatrix[j], out->dBfldMatrix[j], mode);
#endif // MAGNETIC
        }

#ifdef MAGNETIC
        ASSIGN_ADD(SphP[i].divB, out->divB, mode);
#endif // MAGNETIC

#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
        ASSIGN_ADD(SphP[i].Vrms, out->Vrms, mode);
#endif // JD_VTURB || KD_LIMIT_DIVV

#if defined(JD_VTURB) && defined(JD_DECOMPOSE_VTURB)
        ASSIGN_ADD(SphP[i].Vtan, out->Vtan, mode);
        ASSIGN_ADD(SphP[i].Vrad, out->Vrad, mode);
#endif // JD_VTURB && JD_DECOMPOSE_VTURB

#if (GADGET_HYDRO == HYDRO_MFM)
        ASSIGN_ADD(SphP[i].Bgrad, out->Ematrix, mode);
        for (int j = 0; j < numvar; ++j)
        {
            for (int k = 0; k < NUMDIMS; ++k)
            {
                ASSIGN_ADD(SphP[i].gradW[j][k], out->gradW[j][k], mode);
                ASSIGN_ADD(GradStoreOut[i].gradWSph[j][k], out->gradWSph[j][k], mode);
            }
        }
#if MFM_SLOPE_LIMITER == LIMITER_AREPO || MFM_SLOPE_LIMITER == LIMITER_SCALAR || MFM_SLOPE_LIMITER == LIMITER_GIZMO
        for (int k = 0; k < numvar; k++)
        {
            ASSIGN_MIN(SphP[i].Wmin[k], out->Wmin[k], mode);
            ASSIGN_MAX(SphP[i].Wmax[k], out->Wmax[k], mode);
        }
#endif // MFM_SLOPE_LIMITER == LIMITER_AREPO || MFM_SLOPE_LIMITER == LIMITER_SCALAR || MFM_SLOPE_LIMITER ==
       // LIMITER_GIZMO
#if MFM_SLOPE_LIMITER == LIMITER_SCALAR || MFM_SLOPE_LIMITER == LIMITER_GIZMO
        ASSIGN_MAX(SphP[i].DistNgbSqdMax, out->DistNgbSqdMax, mode);
#endif // MFM_SLOPE_LIMITER == LIMITER_SCALAR || MFM_SLOPE_LIMITER == LIMITER_GIZMO
        ASSIGN_MAX(SphP[i].MaxVelSquareNgb, out->MaxVelSquareNgb, mode);
        for (int k = 0; k < mfm_numdims; k++)
        {
            ASSIGN_ADD(SphP[i].Vel_smooth[k], out->Vel_smooth[k], mode);
        }
        ASSIGN_ADD(GradStoreOut[i].neighbour_mass, out->neighbour_mass, mode);
#ifdef GT_DUST
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            for (int j = 0; j < numvar; ++j)
            {
                for (int k = 0; k < NUMDIMS; ++k)
                {
                    ASSIGN_ADD(SphP[i].gradW_dust[idust][j][k], out->gradW_dust[idust][j][k], mode);
                    ASSIGN_ADD(GradStoreOut[i].gradWSph_dust[idust][j][k], out->gradWSph_dust[idust][j][k], mode);
                }
            }
            for (int k = 0; k < NUMDIMS; ++k)
            {
                ASSIGN_MIN(SphP[i].Wmin_dust[idust][k], out->Wmin_dust[idust][k], mode);
                ASSIGN_MAX(SphP[i].Wmax_dust[idust][k], out->Wmax_dust[idust][k], mode);
            }
        }
#endif // GT_DUST
#endif // (GADGET_HYDRO == HYDRO_MFM)
#ifdef GT_DUST
        for (int j = 0; j < NUMDIMS; j++)
        {
            for (int k = 0; k < NUMDIMS; k++)
            {
                ASSIGN_ADD(GradStoreOut[i].dVelMixtureMatrix[j][k], out->dVelMixtureMatrix[j][k], mode);
            }
        }
#ifdef GT_DUST_DIFFUSION
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            ASSIGN_ADD(SphP[i].dust_fraction_smooth[idust], out->dust_fraction_smooth[idust], mode);
        }
        ASSIGN_ADD(SphP[i].neigh_mass, out->neigh_mass, mode);
#endif // GT_DUST_DIFFUSION
#endif // GT_DUST
    } // P[i].Type == type_gas

#ifdef LT_STELLAREVOLUTION
    if (P[i].Type == type_star)
    {
        ASSIGN_ADD(MetP[P[i].pt.MetID].weight, out->weight, mode);
    }
#endif // LT_STELLAREVOLUTION

#ifdef BLACK_HOLES
    if (P[i].Type == type_bh && All.BlackHoleHotColdAccretionOn >= 0)
    {
        ASSIGN_ADD(BPP(i).BH_Entropy, out->SmoothedEntr, mode);
        for (int k = 0; k < 3; k++)
        {
            ASSIGN_ADD(BPP(i).BH_SurroundingGasVel[k], out->GasVel[k], mode);
        }
#ifdef KD_BH_USE_TOPHAT_FOR_RADIO_MODE_AD_MASSWEIGHTS
        ASSIGN_ADD(BPP(i).BH_TotalSurroundingMass, out->TotalSurrMass, mode);
#endif

        if (All.BlackHoleHotColdAccretionOn == 1)
        {
            ASSIGN_ADD(BPP(i).BH_ColdDensity, out->ColdRho, mode);
            ASSIGN_ADD(BPP(i).BH_ColdEntropy, out->SmoothedColdEntr, mode);
            ASSIGN_ADD(BPP(i).BH_HotDensity, out->HotRho, mode);
            ASSIGN_ADD(BPP(i).BH_HotEntropy, out->SmoothedHotEntr, mode);
            for (int k = 0; k < 3; k++)
            {
                ASSIGN_ADD(BPP(i).BH_SurroundingColdGasVel[k], out->ColdGasVel[k], mode);
                ASSIGN_ADD(BPP(i).BH_SurroundingHotGasVel[k], out->HotGasVel[k], mode);
            }
        }

#if (defined(GM_MV_ANGMOM) && defined(MV_GM_AGNMUPPI)) || defined(LS_BLACK_HOLES_SPIN_EVOLUTION)
        for (int j = 0; j < 3; j++)
        {
            ASSIGN_ADD(BPP(i).Vphi[j], out->Vphi[j], mode);
        }
#endif // defined(GM_MV_ANGMOM) && defined(MV_GM_AGNMUPPI) || defined(LS_BLACK_HOLES_SPIN_EVOLUTION)
    }

    if (P[i].Type == type_bh)
    {
        if (mode == 0)
        {
            BPP(i).BH_TimeBinGasNeighbor = out->BH_TimeBinGasNeighbor;
        }
        else
        {
            if (BPP(i).BH_TimeBinGasNeighbor > out->BH_TimeBinGasNeighbor)
                BPP(i).BH_TimeBinGasNeighbor = out->BH_TimeBinGasNeighbor;
        }
    }
#endif // BLACK_HOLES
}

#if !(defined ACC_INCLUDE_GRADIENTS) && defined(ACC_GRADIENTS) // we reinclude this same file if ACC is active
#include "../OpenACC/gpuallvars_branch.hpp"
#include "../OpenACC/gradients.hpp"
#endif // !(defined ACC_INCLUDE_GRADIENTS) && defined(ACC_GRADIENTS)

#endif
#if GADGET_HYDRO == HYDRO_MFM
#include "../Mfm/gradients.hpp"
#endif // GADGET_HYDRO == HYDRO_MFM
#include "compute_gradients.hpp"
#ifndef ACC_INCLUDE_GRADIENTS

/**
 * @brief Computes the SPH density and other quantities.
 *
 * This function computes the local density for each active SPH particle, the
 * number of neighbours in the current smoothing radius, and the divergence
 * and rotation of the velocity field.  The pressure is updated as well.  If a
 * particle with its smoothing region is fully inside the local domain, it is
 * not exported to the other processors. The function also detects particles
 * that have a number of neighbours outside the allowed tolerance range. For
 * these particles, the smoothing length is adjusted accordingly, and the
 * density() computation is called again.  Note that the smoothing length is
 * not allowed to fall below the lower bound set by MinGasHsml (this may mean
 * that one has to deal with substantially more than normal number of
 * neighbours.)
 */
void compute_unified_gradients(void)
{
    TIMER_PUSH(1, "COMPUTE_UNIFIED_GRADIENTS");
    int ndone, group_particles = 1;
    TIMER_PUSH(2, "Initializing");
#ifdef LT_STELLAREVOLUTION
    // some statistics about stars hsml and ngb
    double avg_star[2] = {0.0};
    double minmax_star[4] = {1e10, 1e10, 0.0, 0.0};
#endif // LT_STELLAREVOLUTION

#ifdef MAGNETIC
    MyAtLeastDouble efak;

    if (All.ComovingIntegrationOn)
    {
        efak = 1. / All.Time / All.HubbleParam;
    }
    else
    {
        efak = 1;
    }
#endif // MAGNETIC

#ifdef TIME_DEP_ART_VISC
    MyAtLeastDouble hubble_a, hubble_a2;
    if (All.ComovingIntegrationOn)
    {
        hubble_a = hubble_function(All.Time);
        hubble_a2 = 1. / (All.Time * All.Time * hubble_a);
    }
    else
    {
        hubble_a = hubble_a2 = 1;
    }
#endif // TIME_DEP_ART_VISC

    // needs to be done befoe grad_store_out is created, as this adds particles to the active list
#ifdef LT_STELLAREVOLUTION
    TIMER_PUSH(1, "GRAD_APPEND_CHEM_ACT");
    unsigned int appended = 0, tot_appended, alreadyactive = 0, tot_alreadyactive;
#ifndef LT_LOCAL_IRA
    if (N_stars > 0)
#endif // LT_LOCAL_IRA
        appended = append_chemicallyactive_particles(&alreadyactive);
    MPI_Allreduce(&appended, &tot_appended, 1, MPI_UNSIGNED, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&alreadyactive, &tot_alreadyactive, 1, MPI_UNSIGNED, MPI_SUM, MYMPI_COMM_WORLD);
    if (ThisTask == 0 && (tot_appended > 0 || tot_alreadyactive > 0))
    {
        VERBOSE(2, "%u chemically active particles queued for density calculation (%u already active)..", tot_appended,
                tot_alreadyactive);
    }
    TIMER_POP(1, "GRAD_APPEND_CHEM_ACT");
#endif // LT_STELLAREVOLUTION

    TIMER_PUSH(1, "GRAD_ZEROING");
    // allocate storage for second order gradients
    GradStoreOut = (grad_store_out *)mymalloc("GradStoreOut", NumPart * sizeof(grad_store_out));
    grad_store_out GradStoreOut_zero{};
#ifdef _OPENMP
#pragma omp parallel for if (NActivePart > 5000)
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP

        if (gradients_isactive(i))
        {
            memcpy(&GradStoreOut[i], &GradStoreOut_zero, sizeof(grad_store_out));
        }
    }

    int NTaskTimesNumPart = ((int)(maxThreads)) * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    RecyclingBuffer = (int *)mymalloc("RecyclingBuffer", NTaskTimesNumPart * sizeof(int));

#if defined(BLACK_HOLES) || defined(LT_STELLAREVOLUTION)
#ifdef _OPENMP
#pragma omp parallel for if (NActivePart > 1000)
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP
        if (gradients_isactive(i))
        {
#ifdef BLACK_HOLES
            if (P[i].Type == type_bh)
            {
                BPP(i).SwallowID = 0;
                BPP(i).SwallowPot = -1e30;
            }
#endif // BLACK_HOLES
#ifdef LT_STELLAREVOLUTION
            if (P[i].Type == type_star)
            {
                MetP[P[i].pt.MetID].weight = 0;
            }
#endif // LT_STELLAREVOLUTION
        } // if
    } // for
#endif // BLACK_HOLES || LT_STELLAREVOLUTION

    /* allocate buffers to arrange communication */
    TIMER_POP(1, "GRAD_ZEROING");
    TIMER_POP(2, "Initializing");

    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize =
        (int)((MyBufferSize * 1024 * 1024) /
              (sizeof(struct data_index) + 3 * sizeof(struct data_nodelist) + sizeof(struct graddata_in) +
               sizeof(struct graddata_out) + sizemax(sizeof(struct graddata_in), sizeof(struct graddata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    VERBOSE(2, "Gradients: using %g MB for buffering, %g MB remaining for other buffers", MyBufferSize / 1.0,
            FreeBytes / (1024.0 * 1024.0));

    NextParticle = FirstActiveParticle; /* beginn with this index */

#ifdef ACC_GRADIENTS
    struct graddata_in *LiteGIn;
    struct graddata_out *LiteGOut;

    int *GradientsActiveParticleList;
    int NGradientsActivePart;
    int should_drift = 0;
    int DoPrimaryOnGPU = 0;
    TIMER_PUSH(1, "ACC_INIT");
    NGradientsActivePart =
        openacc_gradients_init(should_drift, DoPrimaryOnGPU, GradientsActiveParticleList, LiteGOut, LiteGIn);

    TIMER_POPPUSH(1, "ACC_INIT", "ACC_UPLOAD_AND_I");
    if (DoPrimaryOnGPU)
    {

#ifdef ACC_OPENACC

#pragma acc enter data create(LiteGIn [0:NumPart], GradientsActiveParticleList [0:NGradientsActivePart],               \
                              LiteGOut [0:NGradientsActivePart])
#pragma acc update device(LiteGIn [0:NumPart], GradientsActiveParticleList [0:NGradientsActivePart],                   \
                          WalkLiteP [0:NumPart]) ACC_ASYNC_COPY1

#elif defined(ACC_OPENMP)

#pragma omp target enter data map(alloc                                                                                \
                                  : LiteGIn [0:NumPart], GradientsActiveParticleList [0:NGradientsActivePart],         \
                                    LiteGOut [0:NGradientsActivePart])
#pragma omp target update to(LiteGIn [0:NumPart], GradientsActiveParticleList [0:NGradientsActivePart],                \
                             WalkLiteP [0:NumPart])
#endif // ACC_OPENMP
    }
    {
        openacc_gradients_primary_phase(DoPrimaryOnGPU, NGradientsActivePart, GradientsActiveParticleList, LiteGOut,
                                        LiteGIn);
        TIMER_POP(1, "ACC_UPLOAD_AND_I");
#else // ACC_GRADIENTS
    int DoPrimaryOnGPU = 0;
    int should_drift = 1;
#endif // ACC_GRADIENTS
        do
        {
            TIMER_PUSH(2, "Primary");
            BufferFullFlag = 0;
            Nexport = 0;
            int save_NextParticle = NextParticle;

            for (int j = 0; j < NTask; j++)
            {
                Exportflag[j] = -1;
            }

            /* do local particles and prepare export list */
            TIMER_PUSH(1, "GRAD_COMPUTE");
#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
            {
#ifdef _OPENMP
                int thread_id = omp_get_thread_num();
#else // _OPENMP
            int thread_id = 0;
#endif // _OPENMP
                ar_green_evaluate_primary(gradients_isactive, gradients_evaluate, Module_Gradients,
                                          TreeWalkType_GreenTreeWalk, NextParticleLoop_ActiveParticleList,
                                          NeighbourSearchCriteria_Density, &thread_id, DoPrimaryOnGPU, should_drift,
                                          group_particles);
            }

#ifdef ADD_EXTRA_BARRIER_FOR_EXACT_INBALANCE_MEASURE
            TIMER_POPPUSH(1, "GRAD_COMPUTE", "GRAD_WAIT");
            MPI_Barrier(MYMPI_COMM_WORLD); // Wait for all tasks to synchronize
            TIMER_POP(1, "GRAD_WAIT");
#else // ADD_EXTRA_BARRIER_FOR_EXACT_INBALANCE_MEASURE
        TIMER_POP(1, "GRAD_COMPUTE");
#endif // ADD_EXTRA_BARRIER_FOR_EXACT_INBALANCE_MEASURE
            TIMER_POPPUSH(2, "Primary", "Exchange");
            TIMER_PUSH(1, "GRAD_COMM_PREP");
            /* if we are not grouping particles and SendRecv fails, then we panic as there is no memory to process even
             * a single particles */
            /*and  if green tree fails, we retry without it */

            int panic_on_error = !group_particles;
            int is_send_receive_ok = prepare_sendrec_offset_send(save_NextParticle, panic_on_error);

            TIMER_POP(1, "GRAD_COMM_PREP");

            if (!is_send_receive_ok)
            {
                TIMER_POP(2, "Exchange");
                group_particles = 0;
                continue;
            }

            TIMER_PUSH(1, "GRAD_COPY");

            GradDataGet = (struct graddata_in *)mymalloc("GradDataGet", Nimport * sizeof(struct graddata_in));
            DataNodeListGet =
                (struct data_nodelist *)mymalloc("DataNodeListGet", Nimport * sizeof(struct data_nodelist));
            GradDataIn = (struct graddata_in *)mymalloc("GradDataIn", Nexport * sizeof(struct graddata_in));
            DataNodeListIn = (struct data_nodelist *)mymalloc("DataNodeListIn", Nexport * sizeof(struct data_nodelist));

            /* prepare particle data for export */
            for (int j = 0; j < Nexport; j++)
            {
                int place = DataIndexTable[j].Index;
                particle2in_gradients(&GradDataIn[j], place);
            }

            copy_data_nodelist();

            /* exchange particle data */
            TIMER_POPPUSH(1, "GRAD_COPY", "GRAD_COMM_EXC");
            templated_exchange_data_in_get(GradDataIn, GradDataGet, 1, TAG_DENS_A);
            TIMER_POP(1, "GRAD_COMM_EXC");

            TIMER_POPPUSH(2, "Exchange", "Secondary");

            TIMER_PUSH(1, "GRAD_ZERO_AND_COPY");
            myfree(DataNodeListIn);
            myfree(GradDataIn);
            GradDataResult = (struct graddata_out *)mymalloc("GradDataResult", Nimport * sizeof(struct graddata_out));
            GradDataOut = (struct graddata_out *)mymalloc("GradDataOut", Nexport * sizeof(struct graddata_out));
            graddata_out GradDataOut_zero{};
            for (int j = 0; j < Nimport; ++j)
            {
                memcpy(&GradDataResult[j], &GradDataOut_zero, sizeof(graddata_out));
            }

            report_memory_usage(&HighMark_sphgrad, "SPH_GRADIENTS");
            TIMER_POPPUSH(1, "GRAD_ZERO_AND_COPY", "GRAD_COMPUTE");
            /* now do the particles that were sent to us */
            NextJ = 0;

#ifdef ACC_GRADIENTS
            openacc_gradients_secondary_phase(GradDataGet, GradDataResult, LiteGIn, DoPrimaryOnGPU, should_drift);
#else // ACC_GRADIENTS
#ifdef _OPENMP // ACC
#pragma omp parallel
#endif // _OPENMP
        {
#ifdef _OPENMP
            int mainthreadid = omp_get_thread_num();
#else // _OPENMP
            int mainthreadid = 0;
#endif // _OPENMP
            ar_green_evaluate_secondary(GradDataGet, gradients_evaluate, Module_Gradients, TreeWalkType_GreenTreeWalk,
                                        NeighbourSearchCriteria_Density, &mainthreadid, should_drift);
        }
#endif // ACC_GRADIENTS

            TIMER_POPPUSH(1, "GRAD_COMPUTE", "GRAD_WAIT");

            int ndone_flag;
            if (NextParticle < 0)
            {
                ndone_flag = 1;
            }
            else
            {
                ndone_flag = 0;
            }
            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
            TIMER_POP(1, "GRAD_WAIT");

            TIMER_POPPUSH(2, "Secondary", "Send_Results");

            TIMER_PUSH(1, "GRAD_COMM_EXC");
            templated_exchange_data_Sendrecv(GradDataResult, GradDataOut, TAG_DENS_B, Recv_offset, Recv_count,
                                             Send_offset, Send_count);
            TIMER_POPPUSH(1, "GRAD_COMM_EXC", "GRAD_COPY");

            /* add the result to the local particles */
            for (int j = 0; j < Nexport; j++)
            {
                int place = DataIndexTable[j].Index;
                out2particle_gradients(&GradDataOut[j], place, 1);
            }
            TIMER_POP(1, "GRAD_COPY");
            TIMER_POP(2, "Send_Results");

            myfree(GradDataOut);
            myfree(GradDataResult);
            myfree(DataNodeListGet);
            myfree(GradDataGet);
        } while (ndone < NTask);
#ifdef ACC_GRADIENTS
        TIMER_PUSH(8, "ACC_WAIT");
        if (DoPrimaryOnGPU)
        {
#ifdef ACC_OPENACC
#pragma acc wait(1)
#elif defined(ACC_OPENMP)
#pragma omp taskwait
#endif
        }
        TIMER_POPPUSH(8, "ACC_WAIT", "ACC_DOWNLOAD");
    } // end data
    if (DoPrimaryOnGPU)
    {
#ifdef ACC_OPENACC
#pragma acc update host(LiteGOut [0:NGradientsActivePart])
#pragma acc exit data delete (GradientsActiveParticleList [0:NGradientsActivePart], LiteGIn [0:NumPart],               \
                              LiteGOut [0:NGradientsActivePart])
#elif defined(ACC_OPENMP)
#pragma omp target update from(LiteGOut [0:NGradientsActivePart])
#pragma omp target exit data map(delete                                                                                \
                                 : GradientsActiveParticleList [0:NGradientsActivePart]) map(delete                    \
                                                                                             : LiteGIn [0:NumPart])    \
    map(delete                                                                                                         \
        : LiteGOut [0:NGradientsActivePart])
#endif
    }
    TIMER_POP(8, "ACC_DOWNLOAD");
#endif // ACC_GRADIENTS

    TIMER_PUSH(2, "Final_Operations");
    TIMER_PUSH(1, "GRAD_COPY");
#ifdef ACC_GRADIENTS
    if (DoPrimaryOnGPU)
    {
#pragma omp parallel for
        for (int counter = 0; counter < NGradientsActivePart; counter++)
        {
            int i = GradientsActiveParticleList[counter];
            out2particle_gradients(&LiteGOut[counter], i, 1);
        }
    }
#endif // ACC_GRADIENTS

    TIMER_POPPUSH(1, "GRAD_COPY", "GRAD_COMPUTE_FINAL");

    /* do final operations on results */
#ifdef _OPENMP
#pragma omp parallel for // private(fac)	// if(NActivePart>80)
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP

        if (gradients_isactive(i))
        {
            if (IS_SPH_ALIVE(P[i]))
            {
// master switch to enforce first order gradients
#ifndef FIRST_ORDER_GRADIENTS
                /* This code calculates higher order velocity gradients using the matrix inversion method.
                 It uses the MFM Matrix class for matrix operations where possible.
                 The method is described in detail in:  Cullen & Dehnen (2010) and Price (2012)
                 The original implementation in Gadget3 is described in Beck+16a, MNRAS, 455, 2, 2110-2130
                 We follow the implementation by Hu C.-Y., Naab T., Walch S., Moster B. P., Oser L., 2014, MNRAS, 443,
                 1173 (Hu+14)*/

                // Calculate the determinant of the distance matrix
                MyAtLeastDouble detChi = GradStoreOut[i].DistanceMatrix.GetDeterminant();

                /*  First, check if determinant is a real number and not too small, to avoid divergence in the
                   inversion. Second, check the condition number and invert the distance matrix if it matches the
                   constraints. If both tests pass we use the second order gradients. Otherwise we fall back to first
                   order.
                */
                if (isnormal(detChi) && abs(detChi) > small_float &&
                    GradStoreOut[i].DistanceMatrix.CheckConditionNumberAndInvertMatrix(detChi))
                {
                    // integral gradients are okay.
                    // continue to use these to compute gradients

                    // Multiply the inverted matrix with the velocity data and store the result back in the original
                    // velocity array Corresponds to X matrix in Hu+14, above Eq. A4
                    for (int k = 0; k < 3; ++k)
                    {
                        GradStoreOut[i].dVelMatrix[k] = GradStoreOut[i].DistanceMatrix * GradStoreOut[i].dVelMatrix[k];
                    }

                    // Calculate the divergence of the velocity (DivVel) from the trace of the X matrix
                    // Hu+14, Eq. A6
                    SphP[i].DivVel = 0.0;
                    for (int k = 0; k < NUMDIMS; ++k)
                    {
                        SphP[i].DivVel += GradStoreOut[i].dVelMatrix[k][k];
                    }

                    // Hu+14, Eq. A8
                    MyAtLeastDouble Rot[3];
                    calculate_rotation(GradStoreOut[i].dVelMatrix, Rot);
#ifdef OUTPUT_VORTICITY
                    for (int k = 0; k < 3; ++k)
                    {
                        SphP[i].Rot[k] = Rot[k];
                    }
#endif // OUTPUT_VORTICITY

                    // The CurlVel is the magnitude of the rotational velocity vector
                    SphP[i].CurlVel = sqrt(Rot[0] * Rot[0] + Rot[1] * Rot[1] + Rot[2] * Rot[2]);

#ifdef DEBUG
                    if (!(SphP[i].DivVel > -2000 && SphP[i].DivVel < 2000))
                    {
                        MyAtLeastDouble trace_dvel = 0.0;
                        for (int k = 0; k < NUMDIMS; ++k)
                        {
                            trace_dvel += GradStoreOut[i].dVelMatrix[k][k];
                        }
                        VERBOSE_ALL(0,
                                    "WARNING: task=%d: Gas Particle id=%llu,m=%e,rho=%e,divV=%e strange 2nd order divV "
                                    "value in findgradients: %e",
                                    ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, SphP[i].Density,
                                    SphP[i].DivVel, trace_dvel);
                    }
#endif // DEBUG

#ifdef MAGNETIC
                    // Multiply the inverted matrix with the magnetic field data and store the result back in the
                    // original magnetic field array Corresponds to X matrix in Hu+14, above Eq. A4
                    for (int k = 0; k < 3; ++k)
                    {
                        GradStoreOut[i].dBfldMatrix[k] =
                            GradStoreOut[i].DistanceMatrix * GradStoreOut[i].dBfldMatrix[k];
                    }

                    // Calculate the divergence of the magnetic field (divB) from the trace of the X matrix
                    // Hu+14, Eq. A6
                    SphP[i].divB = 0.0;
                    for (int k = 0; k < NUMDIMS; ++k)
                    {
                        SphP[i].divB += GradStoreOut[i].dBfldMatrix[k][k];
                    }

#if (defined(TIME_DEP_MAGN_DISP) && (TIME_DEP_MAGN_DISP == 1)) || defined(OUTPUT_ROTB) ||                              \
    defined(ROTB_VISCOSITY_LIMITER)
                    // Hu+14, Eq. A8 for magnetic field
                    calculate_rotation(GradStoreOut[i].dBfldMatrix, SphP[i].RotB);
#endif // (defined(TIME_DEP_MAGN_DISP) && (TIME_DEP_MAGN_DISP == 1)) || defined(OUTPUT_ROTB)

#ifdef OUTPUT_MAGNETIC_CURVATURE
                    MyAtLeastDouble Bnorm_inv2 =
                        1 / (SphP[i].BPred[0] * SphP[i].BPred[0] + SphP[i].BPred[1] * SphP[i].BPred[1] +
                             SphP[i].BPred[2] * SphP[i].BPred[2]);
                    for (int k = 0; k < 3; ++k)
                    {
                        SphP[i].MagCurv[k] = 0.0;
                        for (int j = 0; j < NUMDIMS; ++j)
                        {
                            SphP[i].MagCurv[k] += SphP[i].BPred[j] * GradStoreOut[i].dBfldMatrix[k][j]
                        }
                        SphP[i].MagCurv[k] *= Bnorm_inv2;
                    }
#endif // OUTPUT_MAGNETIC_CURVATURE

#ifdef HIGHER_ORDER_INDUCTION
                    for (int k = 0; k < 3; ++k)
                    {
                        for (int j = 0; j < NUMDIMS; ++j)
                        {
                            SphP[i].dVelMatrix[k][j] = GradStoreOut[i].dVelMatrix[k][j];
                        }
                    }
#endif // HIGHER_ORDER_INDUCTION
#endif // MAGNETIC

#ifdef OUTPUT_GRADIENT_ORDER
                    SphP[i].GradientOrder = 2.0;
#endif // OUTPUT_GRADIENT_ORDER

#ifdef OUTPUT_GRADIENT_MATRICES
                    for (int k = 0; k < NUMDIMS; ++k)
                    {
                        for (int j = 0; j < NUMDIMS; ++j)
                        {
                            SphP[i].DistanceMatrix[k][j] = GradStoreOut[i].DistanceMatrix(k, j);
                        }
                    }
                    for (int k = 0; k < 3; ++k)
                    {
                        for (int j = 0; j < NUMDIMS; ++j)
                        {
                            SphP[i].dVelMatrix[k][j] = GradStoreOut[i].dVelMatrix[k][j];
                        }
                    }
#endif // OUTPUT_GRADIENT_MATRICES
#ifdef GT_DUST
                    for (int k = 0; k < NUMDIMS; ++k)
                    {
                        GradStoreOut[i].dVelMixtureMatrix[k] =
                            GradStoreOut[i].DistanceMatrix * GradStoreOut[i].dVelMixtureMatrix[k];
                    }
                    for (int k = 0; k < NUMDIMS; ++k)
                    {
                        SphP[i].DivVelMixture += GradStoreOut[i].dVelMixtureMatrix[k][k];
                    }

#endif // GT_DUST
                } // end of check if determinant is positive and lager 0
                else
#endif // ifndef FIRST_ORDER_GRADIENTS
                { // else case for second order gradients
                    MyAtLeastDouble inv_density = 1.0 / SphP[i].Density;
                    for (int k = 0; k < 3; ++k)
                    {
                        for (int kk = 0; kk < NUMDIMS; ++kk)
                        {
                            GradStoreOut[i].dVelMatrix[k][kk] *= -inv_density;
                        }
                    }

                    // Fall back to 1st order estimate (DivVel is already computed in find_hsml.c)
                    MyAtLeastDouble Rot[3];
                    calculate_rotation(GradStoreOut[i].dVelMatrix, Rot);
#ifdef OUTPUT_VORTICITY
                    for (int k = 0; k < 3; ++k)
                    {
                        SphP[i].Rot[k] = Rot[k];
                    }
#endif // OUTPUT_VORTICITY

                    // first order CurlVel
                    SphP[i].CurlVel = sqrt(Rot[0] * Rot[0] + Rot[1] * Rot[1] + Rot[2] * Rot[2]);

#ifdef MAGNETIC
                    // first order divB and RotB estimate computed before -> needs normalisation
                    SphP[i].divB /= SphP[i].Density;

#if (defined(TIME_DEP_MAGN_DISP) && (TIME_DEP_MAGN_DISP == 1)) || defined(OUTPUT_ROTB) ||                              \
    defined(ROTB_VISCOSITY_LIMITER)
                    SphP[i].RotB[0] /= SphP[i].Density;
                    SphP[i].RotB[1] /= SphP[i].Density;
                    SphP[i].RotB[2] /= SphP[i].Density;
#endif // (defined(TIME_DEP_MAGN_DISP) && (TIME_DEP_MAGN_DISP == 1)) || defined(OUTPUT_ROTB)

#ifdef OUTPUT_MAGNETIC_CURVATURE
                    SphP[i].MagCurv[0] /= SphP[i].Density;
                    SphP[i].MagCurv[1] /= SphP[i].Density;
                    SphP[i].MagCurv[2] /= SphP[i].Density;
#endif // OUTPUT_MAGNETIC_CURVATURE

#ifdef HIGHER_ORDER_INDUCTION
                    // if only a first order estimate for the magnetic field is valid
                    // it makes no sense to solve the induction equation to higher order
                    for (int k = 0; k < 3; ++k)
                    {
                        for (int j = 0; j < NUMDIMS; ++j)
                        {
                            SphP[i].dVelMatrix[k][j] = 0.0;
                        }
                    }
#endif // HIGHER_ORDER_INDUCTION
#endif // MAGNETIC

#ifdef OUTPUT_GRADIENT_ORDER
                    SphP[i].GradientOrder = 1.0;
#endif // OUTPUT_GRADIENT_ORDER
                } // first order case

#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
                calculate_turbulent_velocity(i);
#endif // JD_VTURB || KD_LIMIT_DIVV

#ifdef KD_LIMIT_DIVV
                {
                    // KD: KD_LIMIT_DIVV = 5 seems to be a good save margin
                    MyAtLeastDouble max_divv = KD_LIMIT_DIVV * SphP[i].Vrms / P[i].Hsml;
                    if (abs(SphP[i].DivVel) > max_divv)
                    {
                        // KD: Not sure if we should limit it or set it directly to zero
                        VERBOSE(2, "WARNING: task=%d: Gas Particle id=%llu,m=%e,rho=%e,divV=%e limit divV to %e",
                                ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, SphP[i].Density,
                                SphP[i].DivVel, max_divv);
                        SphP[i].CurlVel = SphP[i].CurlVel * max_divv / abs(SphP[i].DivVel);
                        SphP[i].DivVel = SphP[i].DivVel / abs(SphP[i].DivVel) * max_divv;
                    }
                }
#endif // KD_LIMIT_DIVV

#ifdef NAVIERSTOKES
                // fill up remaining indices with zero to be consistent with old implementation
                double dvel[3][3];
                for (int kk = 0; kk < 3; kk++)
                {
                    for (int k = 0; k < NUMDIMS; ++k)
                    {
                        dvel[k][kk] = GradStoreOut[i].dVelMatrix[kk][k];
                    }
                    for (int k = NUMDIMS; k < 3; ++k)
                    {
                        dvel[k][kk] = 0.0;
                    }
                }

                for (int k = 0; k < 3; ++k)
                {
                    SphP[i].StressDiag[k] = 2 * dvel[k][k] - 2.0 / 3 * SphP[i].DivVel;
                }

                SphP[i].StressOffDiag[0] = dvel[0][1] + dvel[1][0]; /* xy */
                SphP[i].StressOffDiag[1] = dvel[0][2] + dvel[2][0]; /* xz */
                SphP[i].StressOffDiag[2] = dvel[1][2] + dvel[2][1]; /* yz */

                SphP[i].shear_visc_coeff = All.NavierStokes_ShearViscosity;

#ifndef NAVIERSTOKES_CONSTANT
                const MyAtLeastDouble fr = 0.76;
                const MyAtLeastDouble yhelium = (1.0 - fr) / (4.0 * fr);
                const MyAtLeastDouble mean_mol_weight = (1.0 + 4.0 * yhelium) / (1.0 + 3.0 * yhelium + 1.0);
                MyAtLeastDouble a3inv;
                if (All.ComovingIntegrationOn)
                {
                    a3inv = 1 / (All.Time * All.Time * All.Time);
                }
                else
                {
                    a3inv = 1.0;
                }
                MyAtLeastDouble Temperature = SphP[i].EntropyPred * pow(SphP[i].Density * a3inv, GAMMA_MINUS1) *
                                              PROTONMASS * mean_mol_weight / BOLTZMANN * All.UnitEnergy_in_cgs /
                                              All.UnitMass_in_g;

                SphP[i].shear_visc_coeff *= pow(Temperature, 2.5); /*multiplied by T^5/2 */

#ifdef NAVIERSTOKES_VISCOSITY_SATURATION
                MyAtLeastDouble sigma2 = 0.0;
                MyAtLeastDouble sound_speed = sqrt(GAMMA * SphP[i].Pressure / SphP[i].Density);

                for (int k = 0; k < 3; k++)
                {
                    sigma2 += SphP[i].StressDiag[k] * SphP[i].StressDiag[k] +
                              2. * SphP[i].StressOffDiag[k] * SphP[i].StressOffDiag[k];
                }

                MyAtLeastDouble IonMeanFreePath =
                    All.IonMeanFreePath * Temperature * Temperature / SphP[i].Density; /* T^2/Density */
                MyAtLeastDouble VelLengthScale = 2. * sound_speed / sqrt(sigma2);

                SphP[i].shear_visc_coeff /= (1. + 4.2 * (IonMeanFreePath / VelLengthScale));

#endif // NAVIERSTOKES_VISCOSITY_SATURATION
#endif // !NAVIERSTOKES_CONSTANT

#ifdef NAVIERSTOKES_ANISO
                MyAtLeastDouble B_square, norm_B;
                MyAtLeastDouble uni_B[3];

                B_square = SphP[i].BPred[0] * SphP[i].BPred[0] + SphP[i].BPred[1] * SphP[i].BPred[1] +
                           SphP[i].BPred[2] * SphP[i].BPred[2];

                norm_B = sqrt(B_square);

                uni_B[0] = SphP[i].BPred[0] / norm_B;
                uni_B[1] = SphP[i].BPred[1] / norm_B;
                uni_B[2] = SphP[i].BPred[2] / norm_B;

                SphP[i].P_aniso = 0.0;

                for (int k = 0; k < 3; k++)
                {
                    SphP[i].P_aniso += uni_B[k] * uni_B[0] * dvel[k][0] + uni_B[k] * uni_B[1] * dvel[k][1] +
                                       uni_B[k] * uni_B[2] * dvel[k][2];
                }

                SphP[i].P_aniso *= 3.;
                SphP[i].P_aniso -= SphP[i].DivVel;

                SphP[i].P_aniso *= SphP[i].shear_visc_coeff;

#ifdef TMG_PLASMA_INST
                MyAtLeastDouble plasma_inst;

                if (All.ComovingIntegrationOn)
                {
                    plasma_inst = B_square * All.Time / (4 * M_PI);
                }
                else
                {
                    plasma_inst = B_square / (4 * M_PI);
                }

                if (SphP[i].P_aniso < (-plasma_inst))
                {
                    SphP[i].P_aniso = (-plasma_inst);
                }
                if (SphP[i].P_aniso > (plasma_inst / 2.))
                {
                    SphP[i].P_aniso = (plasma_inst / 2.);
                }

#endif // TMG_PLASMA_INST
#endif // NAVIERSTOKES_ANISO

#endif // NAVIERSTOKES

#ifdef CONDUCTION_SATURATION
                SphP[i].GradEntr[0] /= SphP[i].Density;
                SphP[i].GradEntr[1] /= SphP[i].Density;
                SphP[i].GradEntr[2] /= SphP[i].Density;
#endif // CONDUCTION_SATURATION


#ifdef RADTRANSFER_FLUXLIMITER
                for (int k = 0; k < N_BINS; k++)
                {
                    SphP[i].Grad_ngamma[0][k] /= SphP[i].Density;
                    SphP[i].Grad_ngamma[1][k] /= SphP[i].Density;
                    SphP[i].Grad_ngamma[2][k] /= SphP[i].Density;
                }
#endif // RADTRANSFER_FLUXLIMITER

#ifdef SHOCKFINDER
                /* Normalize shock normal and change sign -> gradient points UP pressure jump
                    Division with density drops out when normalizing.
                    Shared between SPH and MFM  shockfinder. */
                MyAtLeastDouble Shock_Abs1 =
                    -1.0 / sqrt(SphP[i].Shock_N1[0] * SphP[i].Shock_N1[0] + SphP[i].Shock_N1[1] * SphP[i].Shock_N1[1] +
                                SphP[i].Shock_N1[2] * SphP[i].Shock_N1[2]);
                SphP[i].Shock_N1[0] *= Shock_Abs1;
                SphP[i].Shock_N1[1] *= Shock_Abs1;
                SphP[i].Shock_N1[2] *= Shock_Abs1;
#if defined(SHOCKFINDER_SPH) && !defined(SHOCKFINDER_SIMPLE)
                // Beck+16 only: perpendicular shock-frame axes Shock_N2, Shock_N3 for
                // the kernel-projected accumulators. The MFM finder needs only Shock_N1.
                MyAtLeastDouble Shock_W2[3];
                Shock_W2[0] = 0;
                Shock_W2[1] = -SphP[i].Shock_N1[2];
                Shock_W2[2] = SphP[i].Shock_N1[1];

                // second shock normal from projection of the pressure gradient
                MyAtLeastDouble Shock_Scalar1 = SphP[i].Shock_N1[0] * Shock_W2[0] + SphP[i].Shock_N1[1] * Shock_W2[1] +
                                                SphP[i].Shock_N1[2] * Shock_W2[2];
                SphP[i].Shock_N2[0] = Shock_W2[0] - Shock_Scalar1 * SphP[i].Shock_N1[0];
                SphP[i].Shock_N2[1] = Shock_W2[1] - Shock_Scalar1 * SphP[i].Shock_N1[1];
                SphP[i].Shock_N2[2] = Shock_W2[2] - Shock_Scalar1 * SphP[i].Shock_N1[2];

                // normalize
                MyAtLeastDouble Shock_Abs2 =
                    1.0 / sqrt(SphP[i].Shock_N2[0] * SphP[i].Shock_N2[0] + SphP[i].Shock_N2[1] * SphP[i].Shock_N2[1] +
                               SphP[i].Shock_N2[2] * SphP[i].Shock_N2[2]);
                SphP[i].Shock_N2[0] *= Shock_Abs2;
                SphP[i].Shock_N2[1] *= Shock_Abs2;
                SphP[i].Shock_N2[2] *= Shock_Abs2;

                // third shock normal from cross product of the previous two
                SphP[i].Shock_N3[0] =
                    SphP[i].Shock_N1[1] * SphP[i].Shock_N2[2] - SphP[i].Shock_N1[2] * SphP[i].Shock_N2[1];
                SphP[i].Shock_N3[1] =
                    SphP[i].Shock_N1[2] * SphP[i].Shock_N2[0] - SphP[i].Shock_N1[0] * SphP[i].Shock_N2[2];
                SphP[i].Shock_N3[2] =
                    SphP[i].Shock_N1[0] * SphP[i].Shock_N2[1] - SphP[i].Shock_N1[1] * SphP[i].Shock_N2[0];

                // normalize
                MyAtLeastDouble Shock_Abs3 =
                    1.0 / sqrt(SphP[i].Shock_N3[0] * SphP[i].Shock_N3[0] + SphP[i].Shock_N3[1] * SphP[i].Shock_N3[1] +
                               SphP[i].Shock_N3[2] * SphP[i].Shock_N3[2]);
                SphP[i].Shock_N3[0] *= Shock_Abs3;
                SphP[i].Shock_N3[1] *= Shock_Abs3;
                SphP[i].Shock_N3[2] *= Shock_Abs3;
#endif // SHOCKFINDER_SPH && !SHOCKFINDER_SIMPLE
#endif // SHOCKFINDER

#ifdef TIME_DEP_ART_COND
                // abs value of gradient of u, upper part of equation (25) in Beck+ 2016
                MyAtLeastDouble grad_A =
                    sqrt(SphP[i].GradA[0] * SphP[i].GradA[0] + SphP[i].GradA[1] * SphP[i].GradA[1] +
                         SphP[i].GradA[2] * SphP[i].GradA[2]) /
                    SphP[i].Density;

                // ahpha for time dependent artificial conduction, using 1/2 instead of 1/3, but note
                // the difference in the formulation of hsml in the original Tricco & Price 2013 paper
                // As we exclude artificial conduction in compression, we can increase this factor to begin with
                SphP[i].Calpha = P[i].Hsml * grad_A / (SphP[i].EntropyPred * pow(SphP[i].Density, GAMMA_MINUS1));

                if (SphP[i].Calpha > All.ArtCondConstant)
                {
                    SphP[i].Calpha = All.ArtCondConstant;
                }
                if (SphP[i].Calpha < All.ArtCondMin)
                {
                    SphP[i].Calpha = All.ArtCondMin;
                }
#ifdef NEW_COND_LIMIT
                MyAtLeastDouble min_div_vel =
                    sqrt(SphP[i].VelPred[0] * SphP[i].VelPred[0] + SphP[i].VelPred[1] * SphP[i].VelPred[1] +
                         SphP[i].VelPred[2] * SphP[i].VelPred[2]) /
                    P[i].Hsml * DIVV_PRECISION_COND;
#else // NEW_COND_LIMIT
                MyAtLeastDouble min_div_vel = 0.0;
#endif // NEW_COND_LIMIT

                if (SphP[i].DivVel < -min_div_vel)
                {
                    SphP[i].Calpha = All.ArtCondMin;
                }

#ifndef NOGRAVITY
#if (TIME_DEP_ART_COND > 0)
                if (SphP[i].Climit > 1.0)
                {
                    SphP[i].Climit = 1.0;
                }
                if (SphP[i].Climit < 0.0)
                {
                    SphP[i].Climit = 0.0;
                }
                SphP[i].Calpha *= pow(SphP[i].Climit, TIME_DEP_ART_COND);
#ifdef TIME_DEP_ART_COND_GSMOOTH
                SphP[i].Cgrav[0] /= SphP[i].Density;
                SphP[i].Cgrav[1] /= SphP[i].Density;
                SphP[i].Cgrav[2] /= SphP[i].Density;
#endif // TIME_DEP_ART_COND_GSMOOTH
#endif // (TIME_DEP_ART_COND > 0)
#endif // NOGRAVITY

#ifdef WINDS
                if (SphP[i].DelayTime > 0)
                {
                    SphP[i].Calpha = 0.0;
                }
#endif // WINDS
#endif // WINDS

#if defined(TIME_DEP_MAGN_DISP) && ((TIME_DEP_MAGN_DISP == 2) || (TIME_DEP_MAGN_DISP == 3))
                MyAtLeastDouble fieldB = 0.0, normB = 0.0;

                for (int k = 0; k < 3; k++)
                {
                    for (int ii = 0; ii < NUMDIMS; ii++)
                    {
                        fieldB += GradStoreOut[i].dBfldMatrix[k][ii] * GradStoreOut[i].dBfldMatrix[k][ii];
                    }
                    normB += SphP[i].BPred[k] * SphP[i].BPred[k];
                }

#ifdef SFR
                normB *= pow(cbrt(1.0 - SphP[i].XColdCloud), 4);
#endif // SFR

                if (normB > 0.0)
                {
                    SphP[i].Balpha = P[i].Hsml * sqrt(fieldB / normB) / SphP[i].Density;
                }
                else
                {
                    SphP[i].Balpha = All.ArtificialMagneticDissipationMin;
                }

                if (SphP[i].Balpha > All.ArtificialMagneticDissipationConstant)
                {
                    SphP[i].Balpha = All.ArtificialMagneticDissipationConstant;
                }
                if (SphP[i].Balpha < All.ArtificialMagneticDissipationMin)
                {
                    SphP[i].Balpha = All.ArtificialMagneticDissipationMin;
                }

#if (TIME_DEP_MAGN_DISP == 3)
                SphP[i].DtBalpha = 0.0;
#endif // (TIME_DEP_MAGN_DISP == 2)
#endif // defined(TIME_DEP_MAGN_DISP) && ((TIME_DEP_MAGN_DISP == 2) || (TIME_DEP_MAGN_DISP == 3))

                MyAtLeastDouble sound = sqrt(GAMMA * SphP[i].Pressure / SphP[i].Density);

                // Balsare 1995, Eq. 4 / Dolag+2005, Eq. 5 (4.12.2024: removed the / fac_mu as all should be in code
                // units)
                SphP[i].F_Balsara = abs(SphP[i].DivVel) / (abs(SphP[i].DivVel) + SphP[i].CurlVel +
                                                           BALSARA_FAC_FOR_CSND_OVER_H * sound / (2 * P[i].Hsml));

#ifndef TIME_DEP_ART_VISC
                SphP[i].alpha = All.ArtBulkViscConst;
#else // !TIME_DEP_ART_VISC
                eval_art_visc(i, hubble_a, hubble_a2);
#endif // !TIME_DEP_ART_VISC

#if (GADGET_HYDRO == HYDRO_MFM)
                mfm_gradients_final(i, GradStoreOut);
#endif // (GADGET_HYDRO == HYDRO_MFM)
            } // IS_SPH_ALIVE

#ifdef BLACK_HOLES
            if (P[i].Type == type_bh && All.BlackHoleHotColdAccretionOn >= 0)
            {
                assert(BHP[P[i].pt.BHID].PID == i);
                if (BPP(i).BH_Density > 0)
                {
                    BPP(i).BH_Entropy /= BPP(i).BH_Density;
                    BPP(i).BH_SurroundingGasVel[0] /= BPP(i).BH_Density;
                    BPP(i).BH_SurroundingGasVel[1] /= BPP(i).BH_Density;
                    BPP(i).BH_SurroundingGasVel[2] /= BPP(i).BH_Density;
                }
                if (All.BlackHoleHotColdAccretionOn == 1)
                {
                    if (BPP(i).BH_ColdDensity > 0)
                    {
                        BPP(i).BH_ColdEntropy /= BPP(i).BH_ColdDensity;
                        BPP(i).BH_SurroundingColdGasVel[0] /= BPP(i).BH_ColdDensity;
                        BPP(i).BH_SurroundingColdGasVel[1] /= BPP(i).BH_ColdDensity;
                        BPP(i).BH_SurroundingColdGasVel[2] /= BPP(i).BH_ColdDensity;
                    }
                    if (BPP(i).BH_HotDensity > 0)
                    {
                        BPP(i).BH_HotEntropy /= BPP(i).BH_HotDensity;
                        BPP(i).BH_SurroundingHotGasVel[0] /= BPP(i).BH_HotDensity;
                        BPP(i).BH_SurroundingHotGasVel[1] /= BPP(i).BH_HotDensity;
                        BPP(i).BH_SurroundingHotGasVel[2] /= BPP(i).BH_HotDensity;
                    }
                }

#if defined(GM_MV_ANGMOM)
                BPP(i).VphiMod = 0.0;
                if (BPP(i).BH_ColdDensity > 0)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        BPP(i).VphiMod += BPP(i).Vphi[j] * BPP(i).Vphi[j];
                    }

                    BPP(i).VphiMod = sqrt(BPP(i).VphiMod) / BPP(i).BH_ColdDensity / P[i].Hsml;
                }
                else
                {
                    BPP(i).VphiMod = 1.e33; /* GM: I want the limiter NOT to work in this case */
                }
#endif // GM_MV_ANGMOM
            }
#endif // BLACK_HOLES

#ifdef LT_STELLAREVOLUTION
            if (P[i].Type == type_star)
            {
                avg_star[0] += P[i].Hsml;
                avg_star[1] += P[i].TrueNGB;
                minmax_star[0] = (minmax_star[0] > P[i].Hsml ? P[i].Hsml : minmax_star[0]); // min hsml
                minmax_star[1] = (minmax_star[1] > P[i].TrueNGB ? P[i].TrueNGB : minmax_star[1]); // min tngb
                minmax_star[2] = (minmax_star[2] < P[i].Hsml ? P[i].Hsml : minmax_star[2]); // max hsml
                minmax_star[3] = (minmax_star[3] < P[i].TrueNGB ? P[i].TrueNGB : minmax_star[3]); // max tngb
            }
#endif // LT_STELLAREVOLUTION

#ifdef GT_DUST
#ifdef GT_DUST_DIFFUSION
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                SphP[i].dust_fraction_smooth[idust] /= SphP[i].neigh_mass;
            }
#endif // GT_DUST_DIFFUSION
#endif // GT_DUST
        } // if
    } // for
    TIMER_POP(1, "GRAD_COMPUTE_FINAL");
    TIMER_POP(2, "Final_Operations");

#ifdef ACC_GRADIENTS
    /*
    if (DoPrimaryOnGPU)
    {
#pragma omp parallel for
        for (int il = 0; il < NGradientsActivePart; il++)
        {
            int i = GradientsActiveParticleList[il];
            WalkLiteP[i].Hsml = P[i].Hsml;
            particle2in_gradients(&LiteGIn[i], i);
        }
    }
    */
    myfree(LiteGOut);
    myfree(GradientsActiveParticleList);
#endif // ACC_GRADIENTS

#ifdef LT_STELLAREVOLUTION

    if (tot_appended + tot_alreadyactive > 0)
    {
        double avg_star_tot[2], minmax_star_tot[4];
        MPI_Reduce(&avg_star[0], &avg_star_tot[0], 2, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
        MPI_Reduce(&minmax_star[0], &minmax_star_tot[0], 2, MPI_DOUBLE, MPI_MIN, 0, MYMPI_COMM_WORLD);
        MPI_Reduce(&minmax_star[2], &minmax_star_tot[2], 2, MPI_DOUBLE, MPI_MAX, 0, MYMPI_COMM_WORLD);

        if (ThisTask == 0)
        {
            double inv_tot = 1.0 / ((double)(tot_appended + tot_alreadyactive));
            VERBOSE(2, "STARSRHO avg_hsml %g avg_tngb %g min_hsml %g min_tngb %g max_hsml %g max_tngb %g",
                    avg_star_tot[0] * inv_tot, avg_star_tot[1] * inv_tot, minmax_star_tot[0], minmax_star_tot[1],
                    minmax_star_tot[2], minmax_star_tot[3]);
        }
    }
#endif // LT_STELLAREVOLUTION

#ifdef ACC_GRADIENTS
    myfree(LiteGIn);
#endif // ACC_GRADIENTS
    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(RecyclingBuffer);
    myfree(Ngblist);
    myfree(GradStoreOut);

#ifdef LT_STELLAREVOLUTION
    TIMER_PUSH(1, "GRAD_REMOVE_CHEM_ACT");
#ifndef LT_LOCAL_IRA
    if (N_stars > 0)
#endif // LT_LOCAL_IRA
        drop_chemicallyactive_particles();
    TIMER_POP(1, "GRAD_REMOVE_CHEM_ACT");
#endif // LT_STELLAREVOLUTION
    TIMER_POP(1, "COMPUTE_UNIFIED_GRADIENTS");
}

#endif // ifndef ACC_INCLUDE_GRADIENTS

#ifdef ACC_INCLUDE_GRADIENTS
/* I, (AR)had to bring this function before density evaluare, pgi-cc may have problems with definitions _after_ their
 * actual call */
ACC_DECLARE_ROUTINE_BEFORE
#warning "gpu routine"
/**
 * @brief Evaluate extra physics for gas particles on the GPU
 *
 * This function evaluates extra physics for gas particles, including contributions
 * to various quantities such as bulk velocities, entropies, velocities, densities,
 * metallicities, temperatures, and feedback.
 *
 * @param local Pointer to the local data structure
 * @param out Pointer to the output data structure
 * @param kernel Pointer to the kernel density structure
 * @param j Index of the gas particle
 * @param mass_j Mass of the gas particle
 * @param SphP Pointer to the SPH data structure
 */
void gradients_evaluate_extra_physics_gas_gpu(struct graddata_in *local, struct graddata_out *out,
                                              struct kernel_density *kernel, int j, MyAtLeastDouble mass_j,
                                              struct graddata_in *SphP)
#else // ACC_INCLUDE_GRADIENTS
#ifdef ACC_GRADIENTS
/**
 * @brief Evaluate extra physics for gas particles
 *
 * This function evaluates extra physics for gas particles, including contributions
 * to various quantities such as bulk velocities, entropies, velocities, densities,
 * metallicities, temperatures, and feedback.
 *
 * @param local Pointer to the local data structure
 * @param out Pointer to the output data structure
 * @param kernel Pointer to the kernel density structure
 * @param j Index of the gas particle
 * @param mass_j Mass of the gas particle
 * @param SphP Pointer to the SPH data structure
 */
void gradients_evaluate_extra_physics_gas(struct graddata_in *local, struct graddata_out *out,
                                          struct kernel_density *kernel, int j, MyAtLeastDouble mass_j)
#else // ACC_GRADIENTS
/**
 * @brief Evaluate extra physics for gas particles
 *
 * This function evaluates extra physics for gas particles, including contributions
 * to various quantities such as bulk velocities, entropies, velocities, densities,
 * metallicities, temperatures, and feedback.
 *
 * @param local Pointer to the local data structure
 * @param out Pointer to the output data structure
 * @param kernel Pointer to the kernel density structure
 * @param j Index of the gas particle
 * @param mass_j Mass of the gas particle
 * @param SphP Pointer to the SPH data structure
 */
void gradients_evaluate_extra_physics_gas(struct graddata_in *local, struct graddata_out *out,
                                          struct kernel_density *kernel, int j, MyAtLeastDouble mass_j)
#endif // ACC_GRADIENTS
#endif // ACC_INCLUDE_GRADIENTS
{
#ifdef ACC_INCLUDE_GRADIENTS
    struct graddata_in *P = SphP;
#endif // ACC_INCLUDE_GRADIENTS

#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
#ifdef WINDS
    if (SphP[j].DelayTime <= 0)
#endif // WINDS
    {
#ifdef ACC_INCLUDE_GRADIENTS
        turbulent_velocity_evaluate_openacc(local, kernel, j, SphP, out);
#else // ACC_INCLUDE_GRADIENTS
        turbulent_velocity_evaluate(local, kernel, j, out);
#endif // ACC_INCLUDE_GRADIENTS
    }
#endif // JD_VTURB || KD_LIMIT_DIVV

#ifdef BLACK_HOLES
    if (local->Type == type_bh && P[j].Type == type_gas && All.BlackHoleHotColdAccretionOn >= 0)
    {
#ifdef WINDS
        /* Here we exclude wind particles for the relative velocity calculations for BHs */
        if (!(SphP[j].DelayTime > 0))
#endif // ifdef WINDS
        {
            out->SmoothedEntr += kernel->mj_wk * SphP[j].Entropy;
            out->GasVel[0] += kernel->mj_wk * SphP[j].VelPred[0];
            out->GasVel[1] += kernel->mj_wk * SphP[j].VelPred[1];
            out->GasVel[2] += kernel->mj_wk * SphP[j].VelPred[2];
#ifdef KD_BH_USE_TOPHAT_FOR_RADIO_MODE_AD_MASSWEIGHTS
            out->TotalSurrMass += P[j].Mass;
#endif
            if (All.BlackHoleHotColdAccretionOn == 1 || (All.BlackHoleLimitFeedbackOn & 2) == 2)
            {
                float Temperature, xclouds;
#ifdef LT_STELLAREVOLUTION
                Temperature = SphP[j].Temperature;
                xclouds = SphP[j].XColdCloud;
#else // LT_STELLAREVOLUTION
      // FIX IT: Here we need to compute the temperature !!!!
      // get_starformation_rate(j, &Temperature, &xclouds);
#endif // LT_STELLAREVOLUTION
                if (All.BlackHoleHotColdAccretionOn == 1)
                {
                    if (Temperature < All.BlackHoleColdGasTemperatureTresh || xclouds > 0.01)
                    {
                        out->ColdRho += kernel->mj_wk;
                        out->SmoothedColdEntr += kernel->mj_wk * SphP[j].Entropy;
                        out->ColdGasVel[0] += kernel->mj_wk * SphP[j].VelPred[0];
                        out->ColdGasVel[1] += kernel->mj_wk * SphP[j].VelPred[1];
                        out->ColdGasVel[2] += kernel->mj_wk * SphP[j].VelPred[2];
#if defined(GM_MV_ANGMOM) || defined(LS_BLACK_HOLES_SPIN_EVOLUTION)
                        out->Vphi[0] += kernel->mj_wk * (kernel->dy * (local->Vel[2] - SphP[j].VelPred[2]) -
                                                         kernel->dz * (local->Vel[1] - SphP[j].VelPred[1]));
                        out->Vphi[1] += kernel->mj_wk * (kernel->dz * (local->Vel[0] - SphP[j].VelPred[0]) -
                                                         kernel->dx * (local->Vel[2] - SphP[j].VelPred[2]));
                        out->Vphi[2] += kernel->mj_wk * (kernel->dx * (local->Vel[1] - SphP[j].VelPred[1]) -
                                                         kernel->dy * (local->Vel[0] - SphP[j].VelPred[0]));
#endif // GM_MV_ANGMOM || defined(LS_BLACK_HOLES_SPIN_EVOLUTION)
                    }
                    else
                    {
                        out->HotRho += kernel->mj_wk;
                        out->SmoothedHotEntr += kernel->mj_wk * SphP[j].Entropy;
                        out->HotGasVel[0] += kernel->mj_wk * SphP[j].VelPred[0];
                        out->HotGasVel[1] += kernel->mj_wk * SphP[j].VelPred[1];
                        out->HotGasVel[2] += kernel->mj_wk * SphP[j].VelPred[2];
#if (defined(LS_BLACK_HOLES_SPIN_EVOLUTION) && defined(HOT_GAS_CHANNEL_SPIN_EVOL))
                        out->Vphi[0] += kernel->mj_wk * (kernel->dy * kernel->dvz - kernel->dz * kernel->dvy);
                        out->Vphi[1] += kernel->mj_wk * (kernel->dz * kernel->dvx - kernel->dx * kernel->dvz);
                        out->Vphi[2] += kernel->mj_wk * (kernel->dx * kernel->dvy - kernel->dy * kernel->dvx);
#endif
                    }
                }
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
                else
                {
                    out->Vphi[0] += kernel->mj_wk * (kernel->dy * kernel->dvz - kernel->dz * kernel->dvy);
                    out->Vphi[1] += kernel->mj_wk * (kernel->dz * kernel->dvx - kernel->dx * kernel->dvz);
                    out->Vphi[2] += kernel->mj_wk * (kernel->dx * kernel->dvy - kernel->dy * kernel->dvx);
                }
#endif
                if ((All.BlackHoleLimitFeedbackOn & 2) == 2)
                {
                    //              MyAtLeastDouble a3inv, Metallicities[LT_NMet], myPhysDensThresh;// WARNING:
                    //              set_metallicities accepts double*, check types of that array.
                    MyAtLeastDouble a3inv, myPhysDensThresh;
                    if (All.ComovingIntegrationOn)
                    {
                        a3inv = 1 / (All.Time * All.Time * All.Time);
                    }
                    else
                    {
                        a3inv = 1;
                    }

#ifdef LT_METAL_COOLING_WAL
                    double Metallicities[LT_NMet];
#ifndef ACC_INCLUDE_GRADIENTS // we can call endrun only outside a GPU region
                    ENDRUN_IF(LT_NMet < get_cool_n_el(), "Error in LT_NMet");
#endif // ACC_INCLUDE_GRADIENTS
#ifndef ACC_INCLUDE_GRADIENTS
                    set_metallicities(j, &Metallicities[0], a3inv);
#else // ACC_INCLUDE_GRADIENTS
      // set_metallicities_gpu(j, &Metallicities[0], a3inv, SphP, P);
                    set_metallicities_gpu(j, Metallicities, a3inv);
#endif // ACC_INCLUDE_GRADIENTS

                    // Needs to be fixed and filled in case of std cooling on metals
                    out->Metallicity_mjwk += Metallicities[0] * kernel->mj_wk;
#endif // LT_METAL_COOLING_WAL

                    out->Temp_mjwk += Temperature * kernel->mj_wk;
                    out->M_fb += mass_j;
                    out->P_ext_mjwk += SphP[j].Pressure * kernel->mj_wk;

#ifdef SFR
#ifdef LT_STELLAREVOLUTION
#ifdef ACC_INCLUDE_GRADIENTS
                    myPhysDensThresh = get_PhysDensThresh_gpu(j);
#else // ACC_INCLUDE_GRADIENTS
                    myPhysDensThresh = get_PhysDensThresh(j);
#endif // ACC_INCLUDE_GRADIENTS
#else // EB_SFR_MAGNETIC
                    myPhysDensThresh = All.PhysDensThresh;
#endif // LT_STELLAREVOLUTION

                    out->rho_SFR += myPhysDensThresh * kernel->mj_wk;
#endif // SFR
                } // if((All.BlackHoleLimitFeedbackOn & 2) == 2)
            } // if(All.BlackHoleHotColdAccretionOn == 1 || (All.BlackHoleLimitFeedbackOn & 2) == 2)
        } // else of if(SphP[j].DelayTime > 0)
    } // if(local->Type == type_bh && P[j].Type == type_gas)
    if (local->Type == type_bh && P[j].Type == type_gas)
    {
        if (out->BH_TimeBinGasNeighbor > P[j].TimeBin)
            out->BH_TimeBinGasNeighbor = P[j].TimeBin;
    }
#endif // BLACK_HOLES

    if (kernel->r > 0)
    {
#ifdef WINDS
        /* Here we exclude wind particles for calculations  */
        if (!(SphP[j].DelayTime > 0))
#endif // ifdef WINDS
        {
#ifdef CONDUCTION_SATURATION
            out->GradEntr[0] += kernel->mj_dwk_r * kernel->dx * SphP[j].Entropy;
            out->GradEntr[1] += kernel->mj_dwk_r * kernel->dy * SphP[j].Entropy;
            out->GradEntr[2] += kernel->mj_dwk_r * kernel->dz * SphP[j].Entropy;
#endif // CONDUCTION_SATURATION



#ifdef RADTRANSFER_FLUXLIMITER
            for (int k = 0; k < N_BINS; k++)
            {
                out->Grad_ngamma[0][k] += kernel->mj_dwk_r * kernel->dx * SphP[j].n_gamma[k];
                out->Grad_ngamma[1][k] += kernel->mj_dwk_r * kernel->dy * SphP[j].n_gamma[k];
                out->Grad_ngamma[2][k] += kernel->mj_dwk_r * kernel->dz * SphP[j].n_gamma[k];
            }
#endif // RADTRANSFER_FLUXLIMITER

#ifdef TIME_DEP_ART_COND
            // Gradient of u, needed for equation (25) in Beck+ 2016
#ifndef GT_DUST
            MyAtLeastDouble dA =
                (SphP[j].EntropyPred * pow(SphP[j].Density, GAMMA_MINUS1) - kernel->local_entropy_density_term);
#else // GT_DUST
            MyAtLeastDouble dA =
                (SphP[j].EntropyPred * pow(SphP[j].Density * onem_dustfrac(SphP[j].DustFracPred[0]), GAMMA_MINUS1) -
                 kernel->local_entropy_density_term);
#endif // GT_DUST
            out->GradA[0] += kernel->mj_dwk_r * kernel->dx * dA;
            out->GradA[1] += kernel->mj_dwk_r * kernel->dy * dA;
            out->GradA[2] += kernel->mj_dwk_r * kernel->dz * dA;
#if !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0) && defined(TIME_DEP_ART_COND_GSMOOTH)
            out->Cgrav[0] += kernel->mj_wk * P[j].GravAccel[0];
            out->Cgrav[1] += kernel->mj_wk * P[j].GravAccel[1];
            out->Cgrav[2] += kernel->mj_wk * P[j].GravAccel[2];
#endif // !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0) && defined(TIME_DEP_ART_COND_GSMOOTH)
#endif // TIME_DEP_ART_COND

#ifdef SHOCKFINDER
            // shock normal vector, Beck+16b, Eq. 1
            out->Shock_N1[0] += kernel->mj_dwk_r * kernel->dx * (SphP[j].Pressure - local->Pressure);
            out->Shock_N1[1] += kernel->mj_dwk_r * kernel->dy * (SphP[j].Pressure - local->Pressure);
            out->Shock_N1[2] += kernel->mj_dwk_r * kernel->dz * (SphP[j].Pressure - local->Pressure);
#endif // SHOCKFINDER

#ifdef TIME_DEP_ART_VISC
#if (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
            // This will compensate for the hubble expansion in cosmological simulations
            MyAtLeastDouble hubble_correction = 0.0;
            if (All.ComovingIntegrationOn)
            {
                hubble_correction = All.cf_hubble_a;
            }
            // Beck+16a, Eq. 12, without rho -> later!
            if (SphP[j].DivVel + hubble_correction > 0.0)
            {
                out->visc_R += kernel->mj_wk;
            }
            else if (SphP[j].DivVel + hubble_correction < 0.0)
            {
                out->visc_R -= kernel->mj_wk;
            }
#endif // (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
#endif // TIME_DEP_ART_VISC

#ifdef MAGNETIC
            MyAtLeastDouble bpred_j[3];
#ifdef SFR
            MyAtLeastDouble XColdCloudCbrt = cbrt(1.0 - SphP[j].XColdCloud);
            MyAtLeastDouble XColdCloudCorrection = XColdCloudCbrt * XColdCloudCbrt;
#else // SFR
            MyAtLeastDouble XColdCloudCorrection = 1;
#endif // SFR
       //  XColdCloudCorrection is already applied to local->BPred
            bpred_j[0] = SphP[j].BPred[0] * XColdCloudCorrection;
            bpred_j[1] = SphP[j].BPred[1] * XColdCloudCorrection;
            bpred_j[2] = SphP[j].BPred[2] * XColdCloudCorrection;
            MyAtLeastDouble dbx = local->BPred[0] - bpred_j[0];
            MyAtLeastDouble dby = local->BPred[1] - bpred_j[1];
            MyAtLeastDouble dbz = local->BPred[2] - bpred_j[2];
#endif // MAGNETIC

            /* Fill M matrix (Hu+14, Eq. A4)  */
            /* We use a SymmetricMatrix here, such that only half of it needs to be filled*/

#ifdef ACC_INCLUDE_GRADIENTS
            MyAtLeastDouble *distance_matrix = out->DistanceMatrix.get_raw();
            distance_matrix[SYMMETRIC_INDEX(0, 0)] += kernel->mj_dwk_r * kernel->dx * kernel->dx;
#if NUMDIMS > 1
            distance_matrix[SYMMETRIC_INDEX(0, 1)] += kernel->mj_dwk_r * kernel->dy * kernel->dx;
            distance_matrix[SYMMETRIC_INDEX(1, 1)] += kernel->mj_dwk_r * kernel->dy * kernel->dy;
#endif // NUMDIMS > 1
#if NUMDIMS == 3
            distance_matrix[SYMMETRIC_INDEX(0, 2)] += kernel->mj_dwk_r * kernel->dz * kernel->dx;
            distance_matrix[SYMMETRIC_INDEX(1, 2)] += kernel->mj_dwk_r * kernel->dz * kernel->dy;
            distance_matrix[SYMMETRIC_INDEX(2, 2)] += kernel->mj_dwk_r * kernel->dz * kernel->dz;
#endif // NUMDIMS == 3
#else // ACC_INCLUDE_GRADIENTS
            out->DistanceMatrix(0, 0) += kernel->mj_dwk_r * kernel->dx * kernel->dx;
#if NUMDIMS > 1
            out->DistanceMatrix(0, 1) += kernel->mj_dwk_r * kernel->dy * kernel->dx;
            out->DistanceMatrix(1, 1) += kernel->mj_dwk_r * kernel->dy * kernel->dy;
#endif // NUMDIMS > 1
#if NUMDIMS == 3
            out->DistanceMatrix(0, 2) += kernel->mj_dwk_r * kernel->dz * kernel->dx;
            out->DistanceMatrix(1, 2) += kernel->mj_dwk_r * kernel->dz * kernel->dy;
            out->DistanceMatrix(2, 2) += kernel->mj_dwk_r * kernel->dz * kernel->dz;
#endif // NUMDIMS == 3
#endif // ACC_INCLUDE_GRADIENTS

#ifdef ACC_INCLUDE_GRADIENTS
            out->dVelMatrix[0].get_raw()[0] += kernel->mj_dwk_r * kernel->dvx * kernel->dx;
            out->dVelMatrix[1].get_raw()[0] += kernel->mj_dwk_r * kernel->dvy * kernel->dx;
            out->dVelMatrix[2].get_raw()[0] += kernel->mj_dwk_r * kernel->dvz * kernel->dx;
#if NUMDIMS > 1
            out->dVelMatrix[0].get_raw()[1] += kernel->mj_dwk_r * kernel->dvx * kernel->dy;
            out->dVelMatrix[1].get_raw()[1] += kernel->mj_dwk_r * kernel->dvy * kernel->dy;
            out->dVelMatrix[2].get_raw()[1] += kernel->mj_dwk_r * kernel->dvz * kernel->dy;
#endif // NUMDIMS > 1
#if NUMDIMS == 3
            out->dVelMatrix[0].get_raw()[2] += kernel->mj_dwk_r * kernel->dvx * kernel->dz;
            out->dVelMatrix[1].get_raw()[2] += kernel->mj_dwk_r * kernel->dvy * kernel->dz;
            out->dVelMatrix[2].get_raw()[2] += kernel->mj_dwk_r * kernel->dvz * kernel->dz;
#endif // NUMDIMS == 3
#else // ACC_INCLUDE_GRADIENTS
            /* Fill Y matrix (Hu+14, Eq. A4) */
            out->dVelMatrix[0][0] += kernel->mj_dwk_r * kernel->dvx * kernel->dx;
            out->dVelMatrix[1][0] += kernel->mj_dwk_r * kernel->dvy * kernel->dx;
            out->dVelMatrix[2][0] += kernel->mj_dwk_r * kernel->dvz * kernel->dx;
#if NUMDIMS > 1
            out->dVelMatrix[0][1] += kernel->mj_dwk_r * kernel->dvx * kernel->dy;
            out->dVelMatrix[1][1] += kernel->mj_dwk_r * kernel->dvy * kernel->dy;
            out->dVelMatrix[2][1] += kernel->mj_dwk_r * kernel->dvz * kernel->dy;
#endif // NUMDIMS > 1
#if NUMDIMS == 3
            out->dVelMatrix[0][2] += kernel->mj_dwk_r * kernel->dvx * kernel->dz;
            out->dVelMatrix[1][2] += kernel->mj_dwk_r * kernel->dvy * kernel->dz;
            out->dVelMatrix[2][2] += kernel->mj_dwk_r * kernel->dvz * kernel->dz;
#endif // NUMDIMS == 3
#endif // ACC_INCLUDE_GRADIENTS

#ifdef MAGNETIC
#ifdef ACC_INCLUDE_GRADIENTS
            /* Fill Y matrix (Hu+14, Eq. A4) */
            out->dBfldMatrix[0].get_raw()[0] += kernel->mj_dwk_r * dbx * kernel->dx;
            out->dBfldMatrix[1].get_raw()[0] += kernel->mj_dwk_r * dby * kernel->dx;
            out->dBfldMatrix[2].get_raw()[0] += kernel->mj_dwk_r * dbz * kernel->dx;
#if NUMDIMS > 1
            out->dBfldMatrix[0].get_raw()[1] += kernel->mj_dwk_r * dbx * kernel->dy;
            out->dBfldMatrix[1].get_raw()[1] += kernel->mj_dwk_r * dby * kernel->dy;
            out->dBfldMatrix[2].get_raw()[1] += kernel->mj_dwk_r * dbz * kernel->dy;
#endif // NUMDIMS > 1
#if NUMDIMS == 3
            out->dBfldMatrix[0].get_raw()[2] += kernel->mj_dwk_r * dbx * kernel->dz;
            out->dBfldMatrix[1].get_raw()[2] += kernel->mj_dwk_r * dby * kernel->dz;
            out->dBfldMatrix[2].get_raw()[2] += kernel->mj_dwk_r * dbz * kernel->dz;
#endif // NUMDIMS == 3
#else // ACC_INCLUDE_GRADIENTS
            /* Fill Y matrix (Hu+14, Eq. A4) */
            out->dBfldMatrix[0][0] += kernel->mj_dwk_r * dbx * kernel->dx;
            out->dBfldMatrix[1][0] += kernel->mj_dwk_r * dby * kernel->dx;
            out->dBfldMatrix[2][0] += kernel->mj_dwk_r * dbz * kernel->dx;
#if NUMDIMS > 1
            out->dBfldMatrix[0][1] += kernel->mj_dwk_r * dbx * kernel->dy;
            out->dBfldMatrix[1][1] += kernel->mj_dwk_r * dby * kernel->dy;
            out->dBfldMatrix[2][1] += kernel->mj_dwk_r * dbz * kernel->dy;
#endif // NUMDIMS > 1
#if NUMDIMS == 3
            out->dBfldMatrix[0][2] += kernel->mj_dwk_r * dbx * kernel->dz;
            out->dBfldMatrix[1][2] += kernel->mj_dwk_r * dby * kernel->dz;
            out->dBfldMatrix[2][2] += kernel->mj_dwk_r * dbz * kernel->dz;
#endif // NUMDIMS == 3
#endif // ACC_INCLUDE_GRADIENTS

            // first order value of divB
            out->divB += -kernel->mj_dwk_r * (dbx * kernel->dx + dby * kernel->dy + dbz * kernel->dz);

#if (defined(TIME_DEP_MAGN_DISP) && (TIME_DEP_MAGN_DISP == 1)) || defined(OUTPUT_ROTB)
            // first order estimates for RotB
            out->RotB[0] += kernel->mj_dwk_r * (kernel->dz * dby - kernel->dy * dbz);
            out->RotB[1] += kernel->mj_dwk_r * (kernel->dx * dbz - kernel->dz * dbx);
            out->RotB[2] += kernel->mj_dwk_r * (kernel->dy * dbx - kernel->dx * dby);
#endif // (defined(TIME_DEP_MAGN_DISP) && (TIME_DEP_MAGN_DISP == 1)) || defined(OUTPUT_ROTB)

#ifdef OUTPUT_MAGNETIC_CURVATURE

            /*
                Compute curvature of magnetic field - initially implemented by Ulli Steinwandel

                The curvature of the magnetic field is defined as the following (vector signs omitted):
                κ = ( b ⋅ ∇ ) b
                where b = B / |B|

                In component form, this is:
                κ_1 = b_x ∂_x b_x + b_y ∂_y b_x + b_z ∂_z b_x
                κ_2 = b_x ∂_x b_y + b_y ∂_y b_y + b_z ∂_z b_y
                κ_3 = b_x ∂_x b_z + b_y ∂_y b_z + b_z ∂_z b_z

                where b_i is the i-th component of the magnetic field unit vector and ∂_i is the partial derivate
                in the i-direction.

                Please note that this is only the first order accurate estimate of the magnetic field curvature.
                Second order accurate estimate would require the nomalisation of B to be done after the second order
               gradients are computed.
            */

            // components of magnetic field curvature vector
            out->MagCurv[0] += kernel->mj_dwk_r *
                               (local->BPred[0] * kernel->dx * dbx + local->BPred[1] * kernel->dy * dbx +
                                local->BPred[2] * kernel->dz * dbx) *
                               kernel->Bnorm_local_inv2;

            out->MagCurv[1] += kernel->mj_dwk_r *
                               (local->BPred[0] * kernel->dx * dby + local->BPred[1] * kernel->dy * dby +
                                local->BPred[2] * kernel->dz * dby) *
                               kernel->Bnorm_local_inv2;

            out->MagCurv[2] += kernel->mj_dwk_r *
                               (local->BPred[0] * kernel->dx * dbz + local->BPred[1] * kernel->dy * dbz +
                                local->BPred[2] * kernel->dz * dbz) *
                               kernel->Bnorm_local_inv2;

#endif // OUTPUT_MAGNETIC_CURVATURE

#endif // MAGNETIC
#ifdef GT_DUST
            out->dVelMixtureMatrix[0][0] += kernel->mj_dwk_r * kernel->dvx_mixture * kernel->dx;
            out->dVelMixtureMatrix[1][0] += kernel->mj_dwk_r * kernel->dvy_mixture * kernel->dx;
            out->dVelMixtureMatrix[2][0] += kernel->mj_dwk_r * kernel->dvz_mixture * kernel->dx;
#if NUMDIMS > 1
            out->dVelMixtureMatrix[0][1] += kernel->mj_dwk_r * kernel->dvx_mixture * kernel->dy;
            out->dVelMixtureMatrix[1][1] += kernel->mj_dwk_r * kernel->dvy_mixture * kernel->dy;
            out->dVelMixtureMatrix[2][1] += kernel->mj_dwk_r * kernel->dvz_mixture * kernel->dy;
#endif // NUMDIMS > 1
#if NUMDIMS == 3
            out->dVelMixtureMatrix[0][2] += kernel->mj_dwk_r * kernel->dvx_mixture * kernel->dz;
            out->dVelMixtureMatrix[1][2] += kernel->mj_dwk_r * kernel->dvy_mixture * kernel->dz;
            out->dVelMixtureMatrix[2][2] += kernel->mj_dwk_r * kernel->dvz_mixture * kernel->dz;
#endif // NUMDIMS == 3
#ifdef GT_DUST_DIFFUSION
            out->neigh_mass += mass_j;
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                out->dust_fraction_smooth[idust] += mass_j * dustfrac(SphP[j].DustFracPred[idust]);
            }
#endif // GT_DUST_DIFFUSION
#endif // GT_DUST
        } // !(SphP[j].DelayTime > 0) (e.g. not a wind particles
    } // r > 0
}
#ifdef ACC_INCLUDE_GRADIENTS
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_GRADIENTS

#ifdef ACC_INCLUDE_GRADIENTS
#warning "we are entering a gpu only compiled code"
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Evaluate gradients for particles.
 *
 * This function computes gradients for SPH particles. It
 * iterates over each particle and its neighbors to calculate various quantities
 * related to the particle's state and interactions with its surroundings.
 * This function represents the core of the SPH density computation. The
 * target particle may either be local, or reside in the communication
 * buffer.
 *
 * @param active_target The index of the active target particle for which gradients are being evaluated.
 * @param p_target The index of the passive target particle (not used in this function).
 * @param mode The mode of operation, indicating the context in which the function is called.
 * @param ngblist An array of indices representing the neighbors of the active target particle.
 * @param numngb The number of neighbors of the active target particle.
 * @param GradDataGet An array containing input gradient data for particles.
 * @param GradDataResult An array to store the resulting gradient data.
 * @param GP An array of data structures containing additional gravitational information.
 * @param P An array of input particle data.
 * @param i_node The index of the current particle in the neighbor search.
 * @param node The index of the current node in the neighbor search tree.
 *
 * @return Returns 0 on success, -1 if the number of neighbors is invalid, or the number
 * of neighbors when the mode is 3 (indicating a neighbor count request).
 */
inline int gradients_evaluate_local(int active_target, int p_target, int mode, int *ngblist, int numngb,
                                    struct graddata_in *GradDataGet, struct graddata_out *GradDataResult,
                                    struct gravity_walkdata *GP, struct graddata_in *P, int i_node, int node)
#else // ACC_INCLUDE_GRADIENTS
/**
 * @brief Evaluate gradients for particles.
 *
 * This function computes gradients for all SPH particles. It
 * iterates over each particle and its neighbors to calculate various quantities
 * related to the particle's state and interactions with its surroundings.
 * This function represents the core of the SPH density computation. The
 * target particle may either be local, or reside in the communication
 * buffer.
 *
 * @param target The index of the target particle for which gradients are being evaluated.
 * @param mode The mode of operation, indicating the context in which the function is called.
 * @param ngblist An array of indices representing the neighbors of the target particle.
 * @param extended_numngb_inbox The number of extended neighbors in the inbox.
 * @param i_startnode The index of the starting node for neighbor search.
 * @param node The index of the current node in the neighbor search tree.
 * @param i_node The index of the current particle in the neighbor search.
 * @param additional_data Additional data structure pointer for special cases.
 *
 * @return Returns 0 on success, -1 if the number of neighbors is invalid, or the number
 * of neighbors when the mode is 3 (indicating a neighbor count request).
 */
inline int gradients_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node,
                              int i_node, void *additional_data)
#endif // ACC_INCLUDE_GRADIENTS
{
#ifdef ACC_INCLUDE_GRADIENTS
    int target;
    int numngb_inbox;
    int extended_numngb_inbox = numngb;
#else // ACC_INCLUDE_GRADIENTS
    int numngb_inbox; //, active_target = target;
#endif // ACC_INCLUDE_GRADIENTS

    MyAtLeastDouble h2, mass_j;
    kernel_density kernel;
    graddata_in local;
    graddata_out out{};

#if ACC_INCLUDE_GRADIENTS // BECOUSE OPENACC MUST BE THE WEIRDO WHO HAS NOT STANDARD ROUTINES
    // TODO: I guess this is not necessary anymore, but double check!
    memset_gpu((unsigned char *)&out, 0, sizeof(struct graddata_out));
#endif // ACC_INCLUDE_GRADIENTS

#ifdef BLACK_HOLES
    out.BH_TimeBinGasNeighbor = TIMEBINS;
#endif // BLACK_HOLES

#if ACC_INCLUDE_GRADIENTS
    struct graddata_in *SphP = P;
    if (mode == 2)
    {
        target = p_target;
    }
    if (mode == 1)
    {
        target = active_target;
    }
#endif // ACC_INCLUDE_GRADIENTS

    if (mode == 0 || mode == 2 || mode == 3)
    {
#ifdef ACC_INCLUDE_GRADIENTS
        memcpy_gpu(&local, &P[target], sizeof(struct graddata_in));
        // local = P[target];
#else // ACC_INCLUDE_GRADIENTS
        particle2in_gradients(&local, target);
#endif // ACC_INCLUDE_GRADIENTS
    }
    else
    {
#ifdef ACC_INCLUDE_GRADIENTS
        memcpy_gpu(&local, &GradDataGet[target], sizeof(struct graddata_in));
#else // ACC_INCLUDE_GRADIENTS
        local = GradDataGet[target];
#endif // ACC_INCLUDE_GRADIENTS
    }

    h2 = local.Hsml * local.Hsml;
    kernel_hinv(local.Hsml, &kernel.hinv, &kernel.hinv3, &kernel.hinv4);

    /* The following quantities only depend on the target particle (local), which is fixed for the whole
     * neighbour loop below. Precompute them once here instead of recomputing per neighbour. */
#ifdef TIME_DEP_ART_COND
#ifndef GT_DUST
    kernel.local_entropy_density_term = local.EntropyPred * pow(local.Density, GAMMA_MINUS1);
#else // GT_DUST
    kernel.local_entropy_density_term =
        local.EntropyPred * pow(local.Density * onem_dustfrac(local.DustFracPred[0]), GAMMA_MINUS1);
#endif // GT_DUST
#endif // TIME_DEP_ART_COND

#ifdef MAGNETIC
#ifdef OUTPUT_MAGNETIC_CURVATURE
    kernel.Bnorm_local_inv2 =
        1 / (local.BPred[0] * local.BPred[0] + local.BPred[1] * local.BPred[1] + local.BPred[2] * local.BPred[2]);
#endif // OUTPUT_MAGNETIC_CURVATURE
#endif // MAGNETIC

    int startnode = node;

#ifdef ACC_INCLUDE_GRADIENTS
    if (i_node > 0)
    {
        memcpy_gpu(&out, &GradDataResult[active_target], sizeof(struct graddata_out));
    }
#else // ACC_INCLUDE_GRADIENTS

    if (i_node > 0)
    {
        out = GradDataResult[target];
    } // end if(i_node > 0)
#endif // ACC_INCLUDE_GRADIENTS

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            // better to keep those preprocessing stuff indented..

            numngb_inbox = extended_numngb_inbox;
#ifndef ACC_INCLUDE_GRADIENTS
            startnode = *i_startnode;
#else // ACC_INCLUDE_GRADIENTS
            startnode = -1;
#endif // ACC_INCLUDE_GRADIENTS

#if defined(DENSITY_LESS_NGB_FACTOR) && !defined(ACC_INCLUDE_GRADIENTS)
            ENDRUN_IF(numngb_inbox > NumPart / DENSITY_LESS_NGB_FACTOR,
                      "gradients: Number of neighbours found (%d) exceeds allowed number (%d) derived from "
                      "NumPart/DENSITY_LESS_NGB_FACTOR, to continue, decreas DENSITY_LESS_NGB_FACTOR "
                      "(currently %f, min possible val would be 1) !!",
                      numngb_inbox, (int)(NumPart / DENSITY_LESS_NGB_FACTOR), (double)DENSITY_LESS_NGB_FACTOR);
#endif // defined(DENSITY_LESS_NGB_FACTOR) && !defined(ACC_INCLUDE_GRADIENTS)

#ifdef ACC_GRADIENTS
            if (mode == 3)
            {
                return numngb_inbox;
            }
#endif // ACC_GRADIENTS
            if (numngb_inbox < 0)
            {
                return -1;
            }

            for (int n = 0; n < numngb_inbox; n++)
            {
                int j = ngblist[n];

#ifdef ACC_INCLUDE_GRADIENTS
                mass_j = GP[j].Mass;
#else // ACC_INCLUDE_GRADIENTS

                mass_j = P[j].Mass;

#endif // ACC_INCLUDE_GRADIENTS

#ifdef WINDS
#if defined(LT_STELLAREVOLUTION) || defined(AXION_DM) || defined(BLACK_HOLES)
                if (local.Type == type_gas && SphP[j].DelayTime > 0 &&
                    !(local.DelayTime > 0)) /* Wind in density only for wind particles */
                    continue;
                if (local.Type == type_bh && SphP[j].DelayTime > 0) /* Wind does not contribute to density for BHs */
                    continue;
#if !defined(DONOT_EXCLUDE_WINDS_FROM_FEEDBACK) /* Exclude wind from density for Stars */
                if (local.Type == type_star && SphP[j].DelayTime > 0)
                    continue;
#endif
#else // defined(LT_STELLAREVOLUTION) || defined(AXION_DM) || defined(BLACK_HOLES)
                if (SphP[j].DelayTime > 0 && !(local.DelayTime > 0)) /* Wind in density only for wind particles */
                    continue;
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

                kernel.dx = local.Pos[0] - P[j].Pos[0];
                kernel.dy = local.Pos[1] - P[j].Pos[1];
                kernel.dz = local.Pos[2] - P[j].Pos[2];

#ifdef PERIODIC /*  now find the closest image in the given box size  */
                kernel.dx = NEAREST_X(kernel.dx);
                kernel.dy = NEAREST_Y(kernel.dy);
                kernel.dz = NEAREST_Z(kernel.dz);
#endif // PERIODIC
                kernel.r2 = kernel.dx * kernel.dx + kernel.dy * kernel.dy + kernel.dz * kernel.dz;

                if (kernel.r2 < h2)
                {
                    kernel.r = sqrt(kernel.r2);
                    MyAtLeastDouble u = kernel.r * kernel.hinv;

                    kernel_main(u, kernel.hinv3, kernel.hinv4, &kernel.wk, &kernel.dwk, 0);
                    kernel.mj_wk = mass_j * kernel.wk;

#ifdef LT_STELLAREVOLUTION
                    if (local.Type == type_star)
                    {
                        // Note that mass_j in case of MUPPI is already the reduce, true gas mass
                        out.weight += lt_get_weight(local.Type, mass_j, kernel.wk, SphP[j].Density);
                    }
#endif // LT_STELLAREVOLUTION

#if GADGET_HYDRO == HYDRO_MFM
                    kernel.Wj = SphP[j].Wprim;
#endif // GADGET_HYDRO == HYDRO_MFM
#ifdef DISK_SLICE
                    kernel.dv_phi_disk = 0;
                    wrap_velocity_disk_slice(P[j].Pos, local.Pos, kernel.dv_phi_disk);
#if GADGET_HYDRO == HYDRO_MFM
                    kernel.Wj.GetVelocity()[2] += kernel.dv_phi_disk;
#endif // GADGET_HYDRO == HYDRO_MFM
#endif // DISK_SLICE

#if (GADGET_HYDRO == HYDRO_MFM)
#ifdef ACC_INCLUDE_GRADIENTS
                    mfm_gradients_evaluate_openacc(local, kernel, j, GP, SphP, out);
#else // ACC_INCLUDE_GRADIENTS
                    mfm_gradients_evaluate(local, kernel, j, out);
#endif // ACC_INCLUDE_GRADIENTS
#endif // (GADGET_HYDRO == HYDRO_MFM)

                    if (kernel.r > 0)
                    {
                        kernel.mj_dwk_r = mass_j * kernel.dwk / kernel.r;

                        kernel.dvx = local.Vel[0] - SphP[j].VelPred[0];
                        kernel.dvy = local.Vel[1] - SphP[j].VelPred[1];
                        kernel.dvz = local.Vel[2] - SphP[j].VelPred[2];
#ifdef DISK_SLICE
                        kernel.dvz += kernel.dv_phi_disk;
#endif // DISK_SLICE
#ifdef GT_DUST
                        kernel.dvx_mixture = kernel.dvx;
                        kernel.dvy_mixture = kernel.dvy;
                        kernel.dvz_mixture = kernel.dvz;
                        kernel.dvx -= local.TotalDustFracPred * local.TotalDeltaVelPred[0] -
                                      local.TotalDustFracPred * local.TotalDeltaVelPred[0];
                        kernel.dvy -= local.TotalDustFracPred * local.TotalDeltaVelPred[1] -
                                      SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[1];
                        kernel.dvz -= local.TotalDustFracPred * local.TotalDeltaVelPred[2] -
                                      SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[2];
#endif // GT_DUST

                    } // kernel.r > 0
#ifdef ACC_INCLUDE_GRADIENTS
                    gradients_evaluate_extra_physics_gas_gpu(&local, &out, &kernel, j, mass_j, SphP);
#else // ACC_INCLUDE_GRADIENTS
                    gradients_evaluate_extra_physics_gas(&local, &out, &kernel, j, mass_j);
#endif // ACC_INCLUDE_GRADIENTS
                } // kernel.r2 < h2
            } // for numngb_inbox
        } // while startnode >= 0
    } // while startnode >= 0
#ifdef ACC_INCLUDE_GRADIENTS
    target = active_target;
#endif // ACC_INCLUDE_GRADIENTS

    if (mode == 3)
    {
        return 0;
    }

    if (mode == 0)
    {
#ifndef ACC_INCLUDE_GRADIENTS
        out2particle_gradients(&out, target, 0);
#endif // ACC_INCLUDE_GRADIENTS
    }
    else
    {
#ifdef ACC_INCLUDE_GRADIENTS
        memcpy_gpu(&GradDataResult[target], &out, sizeof(struct graddata_out));
#else // ACC_INCLUDE_GRADIENTS
        GradDataResult[target] = out;
#endif // ACC_INCLUDE_GRADIENTS
    }
    return 0;
}
#ifdef ACC_INCLUDE_GRADIENTS
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_GRADIENTS

#ifndef ACC_INCLUDE_GRADIENTS
/**
 * @brief Check if a particle is active.
 *
 * This function determines if a particle is active within the simulation,
 * based on its type and other criteria. It checks if the particle is alive (for SPH particle),
 * if it is a star particle, or if it is a black hole particle. Additionally, it
 * may exclude boundary particles if boundary removal is enabled.
 *
 * @param n The index of the particle to check.
 *
 * @return Returns 1 if the particle is active, 0 otherwise.
 */
inline int gradients_isactive(int n)
{
    int active = IS_SPH_ALIVE(P[n]);
    active += IS_STAR_ACTIVE(P, n);
    active += IS_BLACKHOLE(P[n]);
#ifdef FG_REMOVE_BND
    if (P[n].ID == 0)
    {
        return 0;
    }
#endif // FG_REMOVE_BND

    return active;
}
#ifdef ACC_INCLUDE_GRADIENTS
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_GRADIENTS

#ifdef LT_STELLAREVOLUTION
/**
 * @brief Append chemically active particles to the active particle list.
 *
 * This function appends chemically active particles to the active particle list,
 * updating the list of active particles accordingly. It iterates over gas and
 * star particles, checking their time bins and chemical activity status, and
 * adds them to the active particle list if they are chemically active and their
 * time bins are active. The function also counts the number of particles that were
 * already active.
 *
 * @param already_active Pointer to an unsigned integer where the count of particles
 *                       already active will be stored.
 *
 * @return Returns the number of chemically active particles that were appended
 *         to the active particle list.
 */
int append_chemicallyactive_particles(unsigned int *already_active)
{
    int prev, appended_particles = 0, alreadyactive_particles = 0;

    LastActive = prev = FirstActiveParticle;
    while (LastActive >= 0)
    {
        prev = LastActive;
        LastActive = NextActiveParticle[LastActive];
    }

    LastActive = prev;

#ifdef LT_LOCAL_IRA
#pragma omp parallel for shared(prev, appended_particles) reduction(+ : alreadyactive_particles)
    for (int i = 0; i < N_gas; i++)
    {
#ifdef BLACK_HOLES
        if (P[i].Mass == 0)
            continue;
#endif

        int n = P[i].TimeBin;
        if (TimeBinActive[n])
        {
            alreadyactive_particles += (P[i].Type == type_star) && is_chemically_active(i);
            continue;
        }

        n = SphP[i].ChemTimeBin;

        if (TimeBinActive[n])
        {
#pragma omp critical(_add_chem_dens_)
            {
#ifdef _OPENMP
                ActiveParticleList[NActivePart + appended_particles] = i;
#endif // _OPENMP
                appended_particles++;

                if (prev >= 0)
                {
                    NextActiveParticle[prev] = i;
                }
                else if (prev == -1)
                {
                    FirstActiveParticle = i;
                }

                prev = i;
            } // omp critical
        } // TimebinActive
    } // for i=0,N_gas-1
#endif // LT_LOCAL_IRA

#pragma omp parallel for shared(prev, appended_particles) reduction(+ : alreadyactive_particles)
    for (int istar = 0; istar < N_stars; istar++)
    {
        int i = MetP[istar].PID;
        int n = P[i].TimeBin;
        if (TimeBinActive[n])
        {
            alreadyactive_particles += (P[i].Type == type_star) && is_chemically_active(i);
            continue;
        }

        n = MetP[istar].ChemTimeBin;
        if (TimeBinActive[n])
        {
#pragma omp critical(_add_chem_dens_)
            {
#ifdef _OPENMP
                ActiveParticleList[NActivePart + appended_particles] = i;
#endif // _OPENMP
                appended_particles++;

                if (prev >= 0)
                {
                    NextActiveParticle[prev] = i;
                }
                else if (prev == -1)
                {
                    FirstActiveParticle = i;
                }

                prev = i;
            } // omp critical
        } // if TimeBinActive
    } // for istar=0,N_stars-1

    if (prev >= 0)
    {
        NextActiveParticle[prev] = -1;
    }

    /* KD: There would be a more optimized way to do it !! */
#ifdef _OPENMP
    NActivePartMerk = NActivePart;
    NActivePart += appended_particles;
#endif // _OPENMP

    *already_active = alreadyactive_particles;

    return appended_particles;
}

/**
 * @brief Drops chemically active particles.
 *
 * This function drops chemically active particles by updating the active particle list.
 * If there are active particles, it updates the next active particle pointer for the last active particle.
 * If there are no active particles, it resets the first active particle pointer.
 * Additionally, it updates the number of active particles if OpenMP is enabled.
 */
void drop_chemicallyactive_particles(void)
{
    if (LastActive >= 0)
    {
        NextActiveParticle[LastActive] = -1;
    }
    else
    {
        FirstActiveParticle = -1;
    }

#ifdef _OPENMP
    NActivePart = NActivePartMerk;
#endif // _OPENMP
}
#endif // LT_STELLAREVOLUTION

#endif // ACC_INCLUDE_GRADIENTS

#ifdef ACC_GRADIENTS
#if !defined(ACC_INCLUDE_GRADIENTS) && defined(ACC)
#warning "re including everything"
#define ACC_INCLUDE_GRADIENTS 1
#include "compute_gradients.cpp"
#undef ACC_INCLUDE_GRADIENTS
#endif // !defined(ACC_INCLUDE_GRADIENTS)  && defined(ACC)
#endif // ACC_GRADIENTS
