/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief  Structures for hydrodynamics calculations.
 *
 * @author OpenGadget3-development team, in particular Ludwig Böss, Klaus Dolag, Frederick Groth and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef HYDRA_STRUCTS_HPP
#define HYDRA_STRUCTS_HPP

#ifdef AR_FIX_TIME_DEP_ART_COND
/**
 * @brief Structure to store the maximum signal velocity for the time-dependent artificial conductivity
 */
struct hydro_store_out
{
    MyAtLeastDouble MaxSignalVelOut;
};
#endif // AR_FIX_TIME_DEP_ART_COND

#if defined(MAGNETIC) && defined(DIVBCLEANING_DEDNER) && GADGET_HYDRO == HYDRO_MFM
/**
 * @brief Structure to store the gradient of the potential
 */
struct hydro_store_out_mfm
{
    // these are also build by summation (if DONT_BIN_MFM_FLUXES is not active). Thus, also choose a high precision
    // here.
    MyAtLeastDouble gradPhi[3];
    MyAtLeastDouble gradPhi_dB[3];
};
#endif // defined(MAGNETIC) && defined(DIVBCLEANING_DEDNER) && GADGET_HYDRO==HYDRO_MFM

/**
 * @brief Structure representing input data for hydro computations.
 *
 * This struct holds information about the state and properties of a particle
 * relevant for the hydro calculation.
 */
struct hydrodata_in
{
#if GADGET_HYDRO == HYDRO_MFM
#ifndef NOMUSCL
    MyAtLeastDouble GravAccel[3];
#endif // NOMUSCL
    MyFloat SoundSpeed;
    integertime Ti_begstep;
    MyFloat Volume;
    WFluidVector<mfm_numdims> Wprim;
    MyFloat AlphaSlope[numvar];
    Vector<NUMDIMS> gradW[numvar];
    SymmetricMatrix<NUMDIMS> Bgrad;
    MyFloat ConditionNumber;
#ifdef GT_DUST
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> Wprim_dust;
    MyFloat AlphaSlope_dust[GT_NUM_DUST][numvar];
    std::array<std::array<Vector<NUMDIMS>, numvar>, GT_NUM_DUST> gradW_dust;
#endif // GT_DUST
#ifdef ACC_HYDRA
    MyFloat NumDens;
    MyFloat InternalEnergyPred;
#endif // ACC_HYDRA
#endif // GADGET_HYDRO == HYDRO_MFM
    MyLongDouble Pos[3];
    MyFloat Vel[3];
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat Density;
    MyFloat Pressure;
    MyFloat F1MHD;
    short int wakeup;
    short int TimeBin;
#if defined(WINDS) || defined(GM_MUPPI)
    MyFloat DelayTime;
#endif
#ifdef ACC_HYDRA
    MyFloat VelPred[3];
    MyFloat DivVel;
    MyFloat CurlVel;
    MyFloat MaxSignalVel;
#ifdef MAGNETIC
    union {
        MyFloat BPred[3];
    } b2;
#endif // MAGNETIC

#endif // ACC_HYDRA
#if GADGET_HYDRO == HYDRO_SPH
    MyFloat DhsmlDensityFactor;
#elif GADGET_HYDRO == HYDRO_PESPH
    MyFloat DhsmlNumDensFactor;
    MyFloat DhsmlPressureFactor;
#elif GADGET_HYDRO == HYDRO_MFM
    MyFloat DhsmlNumDensFactor;
#endif // GADGET_HYDRO
#if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(NAVIERSTOKES) || (GADGET_HYDRO == HYDRO_PESPH)
    MyFloat Entropy;
#endif // CONDUCTION_INCLUDEMAGNETIC || NAVIERSTOKES
#ifdef CS_DECOUPLING
    MyFloat Entropy;
    MyIDType Type;
#endif // CS_DECOUPLING
#if defined(PARTICLE_DEBUG) || defined(MAGNETIC_DIFFUSION_LIMIT) || (GADGET_HYDRO == HYDRO_MFM)
    MyIDType ID; /*!< particle identifier */
#endif // if defined(PARTICLE_DEBUG) || defined(MAGNETIC_DIFFUSION_LIMIT) || (GADGET_HYDRO == HYDRO_MFM)

#ifdef JD_VTURB
    MyFloat Vbulk[3];
#endif // JD_VTURB

#ifdef SHOCKFINDER_SPH
    MyFloat Shock_N1[3];
#ifndef SHOCKFINDER_SIMPLE
    MyFloat Shock_N2[3];
    MyFloat Shock_N3[3];
#endif // SHOCKFINDER_SIMPLE
#endif // SHOCKFINDER_SPH

#ifdef TIME_DEP_ART_COND
    MyFloat Calpha;
#ifndef NOGRAVITY
#if (TIME_DEP_ART_COND > 0)
    MyFloat Climit;
#endif // (TIME_DEP_ART_COND > 0)
#endif // NOGRAVITY
#endif // TIME_DEP_ART_COND

#ifdef NAVIERSTOKES
    MyFloat StressOffDiag[3];
    MyFloat StressDiag[3];
    MyFloat shear_visc_coeff;
#ifdef OUTPUT_ENERGY_VISC
    MyFloat visc_E;
#endif // OUTPUT_ENERGY_VISC
#ifdef NAVIERSTOKES_ANISO
    MyFloat P_aniso;
#endif // NAVIERSTOKES_ANISO
#endif // NAVIERSTOKES


#ifdef MAGNETIC
    MyFloat BPred[3];
#ifdef TIME_DEP_MAGN_DISP
    MyFloat Balpha;
#endif // TIME_DEP_MAGN_DISP
#ifdef DIVBCLEANING_TRICCO
    MyFloat PsiPred;
    MyFloat CleanVel;
#endif
#ifdef DIVBCLEANING_DEDNER
    MyFloat PhiPred;
#endif // DIVBCLEANING_DEDNER
#ifdef HIGHER_ORDER_INDUCTION
    MyFloat dVelMatrix[3][3];
#endif
#endif // MAGNETIC

    MyFloat alpha;
    MyFloat F_Balsara;


#ifdef GT_DUST
    MyAtLeastDouble DustFracPred[GT_NUM_DUST];
    MyAtLeastDouble TotalDustFracPred;
    MyAtLeastDouble DeltaVelPred[GT_NUM_DUST][3];
    MyAtLeastDouble TotalDeltaVelPred[3];
    MyAtLeastDouble InternalEnergySPHPred;
    MyAtLeastDouble dust_fraction_smooth[GT_NUM_DUST];
    MyAtLeastDouble visc_R;
    MyAtLeastDouble GradDustFrac;
#endif // GT_DUST
};

/**
 * @brief Structure representing output data for hydro computations.
 *
 * This struct holds information about the state and properties of a particle
 * relevant for the hydro calculation.
 */
struct hydrodata_out
{
#if GADGET_HYDRO == HYDRO_MFM
    QFluidVector<mfm_numdims, MyAtLeastDouble> dQdt;
#ifdef MFM_FACE_CLOSURE_RESIDUAL
    MyAtLeastDouble FaceAreaSum[3];
#endif // MFM_FACE_CLOSURE_RESIDUAL
#ifdef MAGNETIC
    MyAtLeastDouble divBMfm;
#ifdef DIVBCLEANING_DEDNER
    MyAtLeastDouble gradPhi[3];
    MyAtLeastDouble gradPhi_dB[3];
#endif // DIVBCLEANING_DEDNER
#endif // MAGNETIC
#ifdef GT_DUST
    std::array<QFluidVector<mfm_numdims>, GT_NUM_DUST> dQdt_dust;
#endif // GT_DUST
#endif // GADGET_HYDRO == HYDRO_MFM
    MyAtLeastDouble Acc[3];
    MyAtLeastDouble DtEntropy;
#ifdef AR_FIX_TIME_DEP_ART_COND
    MyFloat MaxSignalVelOut;
#else // AR_FIX_TIME_DEP_ART_COND
    MyFloat MaxSignalVel;
#endif // AR_FIX_TIME_DEP_ART_COND




#ifdef SHOCKFINDER
    MyAtLeastDouble Shock_Up_V1[3];
    MyAtLeastDouble Shock_Down_V1[3];
    MyAtLeastDouble Shock_Up_Signal;
    MyAtLeastDouble Shock_Up_Weight;
    MyAtLeastDouble Shock_Down_Weight;
    MyAtLeastDouble Shock_Up_Rho;
    MyAtLeastDouble Shock_Down_Rho;
    MyAtLeastDouble Shock_Up_Pressure;
    MyAtLeastDouble Shock_Down_Pressure;

#ifndef SHOCKFINDER_SIMPLE
    MyAtLeastDouble Shock_Up_V2[3];
    MyAtLeastDouble Shock_Down_V2[3];
    MyAtLeastDouble Shock_Up_V3[3];
    MyAtLeastDouble Shock_Down_V3[3];
    MyAtLeastDouble Shock_Up_Weight23[2];
    MyAtLeastDouble Shock_Down_Weight23[2];
#endif // SHOCKFINDER_SIMPLE

#ifdef MAGNETIC
    MyAtLeastDouble Shock_Mach_Alfven;
    MyAtLeastDouble Shock_Up_Alfven;
    MyAtLeastDouble Shock_Obliquity;
#endif // MAGNETIC

#endif // SHOCKFINDER

#ifdef MAGNETIC
#ifdef WINDS
    MyAtLeastDouble WindB[3];
#endif // WINDS
    MyAtLeastDouble DtB[3];
    MyAtLeastDouble magacc[3];
    MyAtLeastDouble magcorr[3];
#ifdef DIVBCLEANING_TRICCO
    MyAtLeastDouble GradPsi[3];
#endif // DIVBCLEANING_TRICCO
#ifdef DIVBCLEANING_DEDNER
    MyAtLeastDouble GradPhi[3];
#endif // DIVBCLEANING_DEDNER
#endif // MAGNETIC

#if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
    MyAtLeastDouble GradEnergy[3];
#endif // if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
#ifdef AA_OUTPUT_GRADS
    MyAtLeastDouble GradPressure[3];
#endif // AA_OUTPUT_GRADS
#ifdef GT_DUST
    MyAtLeastDouble DtDustFrac[GT_NUM_DUST];
    MyAtLeastDouble DtDeltaVel[GT_NUM_DUST][3];
    MyAtLeastDouble DtInternalEnergySPH;
#endif // GT_DUST
};

/**
 * @brief Structure representing additonal data for the hydro computations.
 *
 * This struct holds additional information about the state and properties of a particle
 * relevant for the hydro calculation.
 */
struct hydrodata_additional
{
    MyAtLeastDouble p_over_rho2_i;
    MyAtLeastDouble p_over_rho2_j;
#ifdef MAGNETIC
    MyAtLeastDouble magneticspeed_i;
    MyAtLeastDouble magneticspeed_j;
    MyAtLeastDouble vcsa2_i, vcsa2_j;
    MyAtLeastDouble mf_windfac;
    MyAtLeastDouble mf_Ind;
    double mf_i;
    double mf_j;
#if defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION)
    MyAtLeastDouble mf_dissInd;
#endif // if defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION)
#if defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION_HEAT)
    MyAtLeastDouble mf_dissEnt;
#endif // if  defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION_HEAT)
    MyAtLeastDouble MagneticStressTensor_i[3][3], MagneticStressTensor_j[3][3];
#ifdef MAGNETIC_DIFFUSION_LIMIT
    MyAtLeastDouble mf_dissLimit;
#endif // MAGNETIC_DIFFUSION_LIMIT
    MyAtLeastDouble bpred_j[3];
#endif // MAGNETIC
#ifdef GT_DUST
    MyAtLeastDouble DeltaVelDissipationSwitch;
    MyAtLeastDouble mean_dust_fraction;
#endif // GT_DUST
#if GADGET_HYDRO == HYDRO_MFM
    MyAtLeastDouble vol_j;
    MyAtLeastDouble Amag;
    Vector<NUMDIMS, MyAtLeastDouble> Aunit;
    Vector<mfm_numdims, MyAtLeastDouble> vface;
    Vector<NUMDIMS, MyAtLeastDouble> drFacei;
    Vector<NUMDIMS, MyAtLeastDouble> drFacej;
    WFluidVector<mfm_numdims, MyAtLeastDouble> Wi;
    WFluidVector<mfm_numdims, MyAtLeastDouble> Wj;
    Vector<NUMDIMS, MyAtLeastDouble> gradWi[numvar];
    Vector<NUMDIMS, MyAtLeastDouble> gradWj[numvar];
#ifdef GT_DUST
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> Wi_dust;
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> Wj_dust;
    std::array<std::array<Vector<NUMDIMS>, numvar>, GT_NUM_DUST> gradWi_dust;
    std::array<std::array<Vector<NUMDIMS>, numvar>, GT_NUM_DUST> gradWj_dust;
#endif // GT_DUST
#endif // GADGET_HYDRO == HYDRO_MFM
};

// define useful aliases for MFM
#if GADGET_HYDRO == HYDRO_MFM
#define fluxdata_in hydrodata_in
#define fluxdata_out hydrodata_out
#endif // GADGET_HYDRO == HYDRO_MFM

#endif // HYDRA_STRUCTS_HPP
