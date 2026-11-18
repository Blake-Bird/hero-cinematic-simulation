/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This contains functions necessary for the hydro_force function.
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

/**
 * @brief Prepares input data for the Hydra hydrodynamics computation based on particle properties.
 *
 * This function prepares input data for the Hydra hydrodynamics computation based on the properties of a given
 * particle.
 *
 * @param in Pointer to the hydrodata_in structure to store the input data for the Hydra computation.
 * @param i Index of the particle for which the input data is prepared.
 *
 * If GADGET_HYDRO is set to HYDRO_MFM, additional parameters related to MFM hydrodynamics are prepared.
 *
 * The function prepares the following input data for the Hydra computation:
 * - Particle position and velocity
 * - Sound speed
 * - Time step
 * - Volume
 * - Various quantities related to hydrodynamics, such as pressure, density, entropy, and acceleration
 * - Magnetic field properties (if MAGNETIC is defined)
 * - Shock-related properties (if SHOCKFINDER is defined)
 * - Cosmic ray-related properties (if LMB_SPECTRAL_CRs is defined)
 * - Shear viscosity (if NAVIERSTOKES is defined)
 * - Dust-related properties (if GT_DUST is defined)
 */
static inline void particle2in_hydra(struct hydrodata_in *in, int i)
{
#ifdef ACC_HYDRA
    in->wakeup = SPHP(i, wakeup);
#endif
#if GADGET_HYDRO == HYDRO_MFM
#ifndef NOMUSCL
    for (int k = 0; k < 3; k++)
    {
        in->GravAccel[k] = P[i].GravAccel[k];
#ifdef PMGRID
        in->GravAccel[k] += P[i].GravPM[k];
#endif // PMGRID
    }
#endif // NOMUSCL
    in->SoundSpeed = eos->SoundSpeed(SphP[i].InternalEnergyPred);
    in->Ti_begstep = P[i].Ti_begstep;
    in->Volume = MyFloat(1.0) / SphP[i].NumDens;
    for (int var = 0; var < numvar; var++)
    {
        in->Wprim[var] = SphP[i].Wprim[var];
        in->AlphaSlope[var] = SphP[i].AlphaSlope[var];
        in->gradW[var] = SphP[i].gradW[var];
    }
    in->Bgrad = SphP[i].Bgrad;
    in->ConditionNumber = SphP[i].ConditionNumber;
#ifdef GT_DUST

    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        for (int var = 0; var < numvar; var++)
        {
            in->Wprim_dust[idust][var] = SphP[i].Wprim_dust[idust][var];
            in->AlphaSlope_dust[idust][var] = SphP[i].AlphaSlope_dust[idust][var];
            in->gradW_dust[idust][var] = SphP[i].gradW_dust[idust][var];
        }
    }
#endif // GT_DUST
#ifdef ACC_HYDRA
    in->NumDens = SphP[i].NumDens;
    in->InternalEnergyPred = SphP[i].InternalEnergyPred;
#endif // ACC_HYDRA
#endif // GADGET_HYDRO == HYDRO_MFM

    for (int k = 0; k < 3; k++)
    {
        in->Pos[k] = P[i].Pos[k];

#ifdef ACC_HYDRA
#ifdef MAGNETIC
        in->b2.BPred[k] = SphP[i].BPred[k];
#endif // MAGNETIC
        in->VelPred[k] = SphP[i].VelPred[k];
#endif // ACC_HYDRA

        in->Vel[k] = SphP[i].VelPred[k];
    }

#if defined(WINDS) || defined(GM_MUPPI)
    in->DelayTime = SphP[i].DelayTime; /*!< remaining maximum decoupling time of wind particle */
#endif // WINDS || GM_MUPPI

#ifdef ACC_HYDRA
    in->MaxSignalVel = SphP[i].MaxSignalVel;
    in->DivVel = SphP[i].DivVel;
    in->CurlVel = SphP[i].CurlVel;
    in->Density = SphP[i].Density;
#endif // ACC_HYDRA

    in->Hsml = P[i].Hsml;
    in->Mass = P[i].Mass;

    in->Density = SphP[i].Density;
#if GADGET_HYDRO == HYDRO_SPH
    in->DhsmlDensityFactor = SphP[i].DhsmlDensityFactor;
    in->Pressure = SphP[i].Pressure;
#elif GADGET_HYDRO == HYDRO_PESPH
    in->DhsmlNumDensFactor = SphP[i].DhsmlNumDensFactor;
    in->DhsmlPressureFactor = SphP[i].DhsmlPressureFactor;
    in->Pressure = SphP[i].Pressure;
#endif // GADGET_HYDRO
    in->TimeBin = P[i].TimeBin;

#if GADGET_HYDRO == HYDRO_PESPH
    in->Entropy = SphP[i].Entropy;
#endif // GADGET_HYDRO
#ifdef CS_DECOUPLING
    in->Entropy = SphP[i].Entropy;
    //    in->Density = SphP[i].Density;
    in->Type = P[i].Type;
#endif //  CS_DECOUPLING


#ifdef TIME_DEP_ART_COND
    in->Calpha = SphP[i].Calpha;
#ifndef NOGRAVITY
#if (TIME_DEP_ART_COND > 0)
    in->Climit = SphP[i].Climit;
#endif // (TIME_DEP_ART_COND > 0)
#endif // NOGRAVITY
#endif // TIME_DEP_ART_COND

#if defined(ROTB_VISCOSITY_LIMITER) && defined(MAGNETIC)
    MyAtLeastDouble vAlfven = sqrt((SphP[i].BPred[1] * SphP[i].BPred[1] + SphP[i].BPred[2] * SphP[i].BPred[2] +
                                    SphP[i].BPred[3] * SphP[i].BPred[3]) *
                                   mu0_inv / SphP[i].Density);

    MyAtLeastDouble absRotB = sqrt(
        (SphP[i].RotB[1] * SphP[i].RotB[1] + SphP[i].RotB[2] * SphP[i].RotB[2] + SphP[i].RotB[3] * SphP[i].RotB[3]));
    // equivalent to Dolag+2005, Eq. 5 for MHD using the Alfvén velocity instead of sound speed (tbd)
    in->F1MHD = abs(SphP[i].divB) / (abs(SphP[i].divB) + absRotB + BALSARA_FAC_FOR_CSND_OVER_H * vAlfven / P[i].Hsml);
#endif // defined(ROTB_VISCOSITY_LIMITER) && defined(MAGNETIC)

#if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(NAVIERSTOKES)
    in->Entropy = SphP[i].Entropy;
#endif // CONDUCTION_INCLUDEMAGNETIC || NAVIERSTOKES

#ifdef JD_VTURB
    in->Vbulk[0] = SphP[i].Vbulk[0];
    in->Vbulk[1] = SphP[i].Vbulk[1];
    in->Vbulk[2] = SphP[i].Vbulk[2];
#endif // JD_VTURB

#ifdef SHOCKFINDER_SPH
    for (int k = 0; k < 3; k++)
    {
        in->Shock_N1[k] = SphP[i].Shock_N1[k];
#ifndef SHOCKFINDER_SIMPLE
        in->Shock_N2[k] = SphP[i].Shock_N2[k];
        in->Shock_N3[k] = SphP[i].Shock_N3[k];
#endif // SHOCKFINDER_SIMPLE
    }
#endif // SHOCKFINDER_SPH

#ifdef MAGNETIC
    for (int k = 0; k < 3; k++)
    {
        in->BPred[k] = SphP[i].BPred[k];
#ifdef SFR
        in->BPred[k] *= pow(cbrt(1.0 - SphP[i].XColdCloud), 2);
#endif // SFR
    }
#ifdef TIME_DEP_MAGN_DISP
    in->Balpha = SphP[i].Balpha;
#endif // TIME_DEP_MAGN_DISP
#ifdef DIVBCLEANING_TRICCO
    in->PsiPred = SphP[i].PsiPred;
    in->CleanVel = SphP[i].CleanVel;
#endif // DIVBCLEANING_TRICCO
#ifdef DIVBCLEANING_DEDNER
    in->PhiPred = SphP[i].PhiPred;
#endif // DIVBCLEANING_DEDNER
#ifdef HIGHER_ORDER_INDUCTION
    for (int k = 0; k < 3; k++)
    {
        for (int j = 0; j < 3; j++)
        {
            in->dVelMatrix[k][j] = SphP[i].dVelMatrix[k][j];
        }
    }
#endif
#endif // MAGNETIC

    in->alpha = SphP[i].alpha;
    in->F_Balsara = SphP[i].F_Balsara;

#if defined(PARTICLE_DEBUG) || defined(MAGNETIC_DIFFUSION_LIMIT) || (GADGET_HYDRO == HYDRO_MFM)
    in->ID = P[i].ID;
#endif // if defined(PARTICLE_DEBUG) || defined(MAGNETIC_DIFFUSION_LIMIT) || (GADGET_HYDRO == HYDRO_MFM)

#ifdef NAVIERSTOKES
    for (int k = 0; k < 3; k++)
    {
        in->StressDiag[k] = SphP[i].StressDiag[k];
        in->StressOffDiag[k] = SphP[i].StressOffDiag[k];
    }
    in->shear_visc_coeff = SphP[i].shear_visc_coeff;
#ifdef OUTPUT_ENERGY_VISC
    in->visc_E = SphP[i].visc_E;
#endif // OUTPUT_ENERGY_VISC
#ifdef NAVIERSTOKES_ANISO
    in->P_aniso = SphP[i].P_aniso;
#endif // NAVIERSTOKES_ANISO
#endif // NAVIERSTOKES

#ifdef GT_DUST
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        in->DustFracPred[idust] = SphP[i].DustFracPred[idust];
        for (int k = 0; k < 3; k++)
        {
            in->DeltaVelPred[idust][k] = SphP[i].DeltaVelPred[idust][k];
        }
        in->dust_fraction_smooth[idust] = SphP[i].dust_fraction_smooth[idust];
    }
    in->TotalDustFracPred = SphP[i].TotalDustFracPred;
    for (int k = 0; k < 3; k++)
        in->TotalDeltaVelPred[k] = SphP[i].TotalDeltaVelPred[k];
    in->InternalEnergySPHPred = SphP[i].InternalEnergySPHPred;

#ifdef TIME_DEP_ART_VISC
    in->visc_R = SphP[i].visc_R;
#endif // TIME_DEP_ART_VISC
#endif // GT_DUST
}

/**
 * @brief Updates particle properties using output data from the Hydra hydrodynamics computation.
 *
 * This function updates various properties of particles based on the output data from the Hydra hydrodynamics
 * computation. It is used to assign hydrodynamics-related properties to particles, such as acceleration, entropy change
 * rate, and maximum signal velocity.
 *
 * @param out Pointer to the hydrodata_out structure containing the output data from the Hydra computation.
 * @param i Index of the particle to be updated.
 * @param mode Specifies the mode of updating. A value of 0 indicates a non-assigned particle, while a value of 1
 * indicates an assigned particle.
 *
 * If GADGET_HYDRO is set to HYDRO_MFM, additional parameters related to MFM hydrodynamics are updated.
 *
 * @param mfm_mode Specifies the MFM mode if GADGET_HYDRO is set to HYDRO_MFM.
 *
 * The function updates the following properties of the particle:
 * - Hydrodynamic acceleration
 * - Entropy change rate
 * - Maximum signal velocity
 * - Magnetic field properties (if MAGNETIC is defined)
 * - Shock-related properties (if SHOCKFINDER is defined)
 * - Cosmic ray-related properties (if LMB_SPECTRAL_CRs_ARTIFICIAL_CONDUCTIVITY or DK_SPECTRAL_SN_SEEDING is defined)
 * - Dust-related properties (if GT_DUST is defined)
 */
static inline void out2particle_hydra(struct hydrodata_out *out, int i, int mode
#if GADGET_HYDRO == HYDRO_MFM
                                      ,
                                      int mfm_mode
#endif // GADGET_HYDRO == HYDRO_MFM
)
{
#ifdef AR_FIX_ASSIGN
#ifdef AR_FIX_ASSIGN_VERBOSE
    if (mode == 1 && !P[i].assigned)
    {
        VERBOSE_OUTPUT("mode==1 && !P[i].assigned ID=%llu Type=%d .MaxSignalVel %g <- %g", (unsigned long long)P[i].ID,
                       P[i].Type,
#ifdef AR_FIX_TIME_DEP_ART_COND
                       HydroStoreOut[i].MaxSignalVelOut, out->MaxSignalVelOut
#else // AR_FIX_TIME_DEP_ART_COND
                       SphP[i].MaxSignalVel, out->MaxSignalVel
#endif // AR_FIX_TIME_DEP_ART_COND
        );
    }

#endif // AR_FIX_ASSIGN_VERBOSE
    if (P[i].assigned)
    {
        mode = 1;
    }
    else
    {
        mode = 0;
    }
    P[i].assigned = 1;
#endif // AR_FIX_ASSIGN

#if GADGET_HYDRO == HYDRO_MFM
    for (int k = 0; k < numvar; k++)
    {
        ASSIGN_ADD(SphP[i].dQdt[k], out->dQdt[k], mfm_mode);
    }
#ifdef MFM_FACE_CLOSURE_RESIDUAL
    for (int k = 0; k < 3; k++)
    {
        ASSIGN_ADD(SphP[i].FaceAreaSum[k], out->FaceAreaSum[k], mfm_mode);
    }
#endif // MFM_FACE_CLOSURE_RESIDUAL
#ifdef MAGNETIC
    ASSIGN_ADD(SphP[i].divBMfm, out->divBMfm, mfm_mode);

#ifdef DIVBCLEANING_TRICCO
    for (int k = 0; k < 3; k++)
    {
        ASSIGN_ADD(SphP[i].GradPsi[k], out->GradPsi[k], mfm_mode);
    }
#endif // DIVBCLEANING_TRICCO
#endif // MAGNETIC
#ifdef GT_DUST
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        for (int k = 0; k < numvar; k++)
        {
            ASSIGN_ADD(SphP[i].dQdt_dust[idust][k], out->dQdt_dust[idust][k], mfm_mode);
        }
    }
#endif // GT_DUST
#endif // GADGET_HYDRO == HYDRO_MFM

    for (int k = 0; k < 3; k++)
    {
#if GADGET_HYDRO == HYDRO_MFM
        ASSIGN_ADD(SphP[i].HydroAccelExtra[k], out->Acc[k], mode);
#else // GADGET_HYDRO == HYDRO_MFM
        ASSIGN_ADD(SphP[i].HydroAccel[k], out->Acc[k], mode);
#ifdef DIVBCLEANING_TRICCO
        ASSIGN_ADD(SphP[i].GradPsi[k], out->GradPsi[k], mode);
#endif // DIVBCLEANING_TRICCO
#endif
    }
#if GADGET_HYDRO == HYDRO_MFM
    ASSIGN_ADD(SphP[i].DtEntropyExtra, out->DtEntropy, mode);
#else // GADGET_HYDRO == HYDRO_MFM
    ASSIGN_ADD(SphP[i].DtEntropy, out->DtEntropy, mode);
#endif

#if defined(MAGNETIC) && defined(DIVBCLEANING_DEDNER) && (GADGET_HYDRO == HYDRO_MFM)
    for (int k = 0; k < 3; k++)
    {
        ASSIGN_ADD(HydroStoreOutMfm[i].gradPhi[k], out->gradPhi[k], mfm_mode);
        ASSIGN_ADD(HydroStoreOutMfm[i].gradPhi_dB[k], out->gradPhi_dB[k], mfm_mode);
    }
#endif // defined(MAGNETIC) && defined(DIVBCLEANING_DEDNER) && GADGET_HYDRO == HYDRO_MFM

#ifdef AR_FIX_TIME_DEP_ART_COND
    if (mode == 0)
    {
        HydroStoreOut[i].MaxSignalVelOut = out->MaxSignalVelOut;
    }
    else
    {
        if (HydroStoreOut[i].MaxSignalVelOut < out->MaxSignalVelOut)
        {
            HydroStoreOut[i].MaxSignalVelOut = out->MaxSignalVelOut;
        }
    }
#else // AR_FIX_TIME_DEP_ART_COND
    if (mode == 0)
    {
        SphP[i].MaxSignalVel = out->MaxSignalVel;
    }
    else
    {
        if (SphP[i].MaxSignalVel < out->MaxSignalVel)
        {
            SphP[i].MaxSignalVel = out->MaxSignalVel;
        }
    }
#endif // AR_FIX_TIME_DEP_ART_COND




#ifdef SHOCKFINDER_SPH
    for (int k = 0; k < 3; k++)
    {
        ASSIGN_ADD(SphP[i].Shock_Up_V1[k], out->Shock_Up_V1[k], mode);
        ASSIGN_ADD(SphP[i].Shock_Down_V1[k], out->Shock_Down_V1[k], mode);
    }
    ASSIGN_ADD(SphP[i].Shock_Up_Signal, out->Shock_Up_Signal, mode);
    ASSIGN_ADD(SphP[i].Shock_Up_Weight, out->Shock_Up_Weight, mode);
    ASSIGN_ADD(SphP[i].Shock_Down_Weight, out->Shock_Down_Weight, mode);
    ASSIGN_ADD(SphP[i].Shock_Up_Rho, out->Shock_Up_Rho, mode);
    ASSIGN_ADD(SphP[i].Shock_Down_Rho, out->Shock_Down_Rho, mode);
    ASSIGN_ADD(SphP[i].Shock_Up_Pressure, out->Shock_Up_Pressure, mode);
    ASSIGN_ADD(SphP[i].Shock_Down_Pressure, out->Shock_Down_Pressure, mode);
#ifndef SHOCKFINDER_SIMPLE
    for (int k = 0; k < 3; k++)
    {
        ASSIGN_ADD(SphP[i].Shock_Up_V2[k], out->Shock_Up_V2[k], mode);
        ASSIGN_ADD(SphP[i].Shock_Down_V2[k], out->Shock_Down_V2[k], mode);
        ASSIGN_ADD(SphP[i].Shock_Up_V3[k], out->Shock_Up_V3[k], mode);
        ASSIGN_ADD(SphP[i].Shock_Down_V3[k], out->Shock_Down_V3[k], mode);
    }
    ASSIGN_ADD(SphP[i].Shock_Up_Weight23[0], out->Shock_Up_Weight23[0], mode);
    ASSIGN_ADD(SphP[i].Shock_Down_Weight23[0], out->Shock_Down_Weight23[0], mode);
    ASSIGN_ADD(SphP[i].Shock_Up_Weight23[1], out->Shock_Up_Weight23[1], mode);
    ASSIGN_ADD(SphP[i].Shock_Down_Weight23[1], out->Shock_Down_Weight23[1], mode);
#endif // SHOCKFINDER_SIMPLE
#ifdef MAGNETIC
    ASSIGN_ADD(SphP[i].Shock_Mach_Alfven, out->Shock_Mach_Alfven, mode);
    ASSIGN_ADD(SphP[i].Shock_Up_Alfven, out->Shock_Up_Alfven, mode);
    ASSIGN_ADD(SphP[i].Shock_Obliquity, out->Shock_Obliquity, mode);
#endif // MAGNETIC
#endif // SHOCKFINDER_SPH

#ifdef MAGNETIC
    for (int k = 0; k < 3; k++)
    {
#ifdef WINDS
        ASSIGN_ADD(SphP[i].WindB[k], out->WindB[k], mode);
#endif // WINDS
#if GADGET_HYDRO == HYDRO_MFM
        ASSIGN_ADD(SphP[i].DtBExtra[k], out->DtB[k], mode);
#else // GADGET_HYDRO == HYDRO_MFM
#if defined(FOLLOW_ID)
        if (P[i].ID == FOLLOW_ID && k == 0)
        {
            printf("HOI DEBUGSUM: mode=%d DtB[0]=%g out.DtB[0]=%g\n", mode, SphP[i].DtB[k], out->DtB[k]);
        }
#endif
        ASSIGN_ADD(SphP[i].DtB[k], out->DtB[k], mode);
#endif // GADGET_HYDRO == HYDRO_MFM
        ASSIGN_ADD(SphP[i].magacc[k], out->magacc[k], mode);
        ASSIGN_ADD(SphP[i].magcorr[k], out->magcorr[k], mode);
#ifdef DIVBCLEANING_DEDNER
        ASSIGN_ADD(SphP[i].GradPhi[k], out->GradPhi[k], mode);
#endif // DIVBCLEANING_DEDNER
    }
#endif // MAGNETIC

#if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
    for (int k = 0; k < 3; k++)
    {
        ASSIGN_ADD(SphP[i].GradEnergy[k], out->GradEnergy[k], mode);
    }
#endif // if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
#ifdef AA_OUTPUT_GRADS
    for (int k = 0; k < 3; k++)
    {
        ASSIGN_ADD(SphP[i].GradPressure[k], out->GradPressure[k], mode);
    }
#endif // AA_OUTPUT_GRADS

#ifdef GT_DUST
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        ASSIGN_ADD(SphP[i].DtDustFrac[idust], out->DtDustFrac[idust], mode);
        for (int k = 0; k < 3; k++)
        {
            ASSIGN_ADD(SphP[i].DtDeltaVel[idust][k], out->DtDeltaVel[idust][k], mode);
#if GADGET_HYDRO == HYDRO_MFM
            ASSIGN_ADD(SphP[i].DtDeltaVelExtra[idust][k], out->DtDeltaVel[idust][k], mode);
#endif // GADGET_HYDRO == HYDRO_MFM
        }
    }
    ASSIGN_ADD(SphP[i].DtInternalEnergySPH, out->DtInternalEnergySPH, mode);

#endif // GT_DUST
}
