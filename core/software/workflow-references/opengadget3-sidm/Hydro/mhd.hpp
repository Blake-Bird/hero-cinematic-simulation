/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief MHD header file.
 *
 * @author OpenGadget3-development team, in particular Ludwig Böss, Klaus Dolag and Ulrich Steinwandel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MHD_HPP
// #define MHD_HPP

#ifdef MAGNETIC

#include "../CodeBase/allvars.hpp"
#if !(defined ACC_INCLUDE_HYDRA) && defined(ACC_HYDRA) // we reinclude this same file if ACC is active
#include "../OpenACC/gpuallvars_branch.hpp"
#endif // !(defined ACC_INCLUDE_HYDRA) && defined(ACC_HYDRA)

#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Computes the divergence of magnetic forces for a particle with OpenACC support.
 *
 * This function calculates the divergence of magnetic forces acting on a particle.
 * It first computes the magnetic pressure tensor components based on the predicted
 * magnetic field values of the particle and its neighbor. Then, it subtracts half
 * of the magnetic pressure from the diagonal elements of the pressure tensor.
 * Next, it calculates the contribution of magnetic forces to the acceleration of
 * the particle along each spatial direction. Finally, it computes the correction
 * term for the magnetic forces and applies it to the acceleration, ensuring
 * consistency with the divergence cleaning scheme.
 *
 * @param local Pointer to the local hydrodynamic data of the particle.
 * @param kernel Pointer to the kernel properties for the interaction.
 * @param j Index of the neighbor particle.
 * @param additional_variables Pointer to additional variables used in the computation.
 * @param P Pointer to the gravity walk data structure.
 * @param SphP Pointer to the hydrodynamic data of the neighbor particle.
 * @param out Pointer to the output hydrodynamic data of the particle.
 */
void lorentz_force_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                           struct hydrodata_additional *additional_variables, struct gravity_walkdata *P,
                           struct hydrodata_in *SphP, struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Computes the divergence of magnetic forces for a particle.
 *
 * This function calculates the divergence of magnetic forces acting on a particle.
 * It first computes the magnetic pressure tensor components based on the predicted
 * magnetic field values of the particle and its neighbor. Then, it subtracts half
 * of the magnetic pressure from the diagonal elements of the pressure tensor.
 * Next, it calculates the contribution of magnetic forces to the acceleration of
 * the particle along each spatial direction. Finally, it computes the correction
 * term for the magnetic forces and applies it to the acceleration, ensuring
 * consistency with the divergence cleaning scheme.
 *
 * @param local Pointer to the local hydrodynamic data of the particle.
 * @param kernel Pointer to the kernel properties for the interaction.
 * @param j Index of the neighbor particle.
 * @param additional_variables Pointer to additional variables used in the computation.
 * @param out Pointer to the output hydrodynamic data of the particle.
 */
static inline void lorentz_force(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                 struct hydrodata_additional *additional_variables, struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
    // Fill in the magnetic stress tensor components, see Dolag & Stasyszyn (2009), Eq. 16
    for (int k = 0; k < 3; k++)
    {
        for (int l = 0; l < 3; l++)
        {
            additional_variables->MagneticStressTensor_j[k][l] =
                additional_variables->bpred_j[k] * additional_variables->bpred_j[l];
        }
    }
    // Subtract half of the magnetic pressure from the diagonal elements
    for (int k = 0; k < 3; k++)
    {
        additional_variables->MagneticStressTensor_j[k][k] -= 0.5 * kernel->b2_j;
    }

    // compute magnetic acceleration from the magnetic stress tensor
    // see Dolag & Stasyszyn (2009), Eq. 17
    for (int k = 0; k < 3; k++)
    {
        out->magacc[k] += (additional_variables->MagneticStressTensor_i[k][0] * kernel->mf_i +
                           additional_variables->MagneticStressTensor_j[k][0] * kernel->mf_j) *
                              kernel->dx +
                          (additional_variables->MagneticStressTensor_i[k][1] * kernel->mf_i +
                           additional_variables->MagneticStressTensor_j[k][1] * kernel->mf_j) *
                              kernel->dy +
                          (additional_variables->MagneticStressTensor_i[k][2] * kernel->mf_i +
                           additional_variables->MagneticStressTensor_j[k][2] * kernel->mf_j) *
                              kernel->dz;
    }

    // compute the correction term for the contribution of divB
    // see Dolag & Stasyszyn (2009), Eq. 23
    for (int k = 0; k < 3; k++)
    {
        out->magcorr[k] +=
            local->BPred[k] *
            ((local->BPred[0] * kernel->mf_i + additional_variables->bpred_j[0] * kernel->mf_j) * kernel->dx +
             (local->BPred[1] * kernel->mf_i + additional_variables->bpred_j[1] * kernel->mf_j) * kernel->dy +
             (local->BPred[2] * kernel->mf_i + additional_variables->bpred_j[2] * kernel->mf_j) * kernel->dz);
    }
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Finalizes the calculation of magnetic divergence cleaning forces for a particle.
 *
 * This function completes the computation of magnetic divergence cleaning forces for a particle.
 * It first calculates the magnitude of the correction term `phiphi` and the magnitude of the
 * acceleration term `tmpb`. Then, it compares the ratio of `phiphi` to `tmpb` with the predefined
 * threshold `DIVBFORCECORRECTION_FACTOR`. If this ratio exceeds `DIVBFORCECORRECTION_FACTOR`,
 * the correction term is rescaled to ensure consistency with the chosen divergence cleaning scheme.
 * Finally, the corrected magnetic forces are added to the hydrodynamic acceleration of the particle.
 *
 * @param i Index of the particle for which the magnetic forces are being computed.
 */
static inline void lorentz_force_final(int i)
{
    // Absolute value of divB subtraction factor
    MyAtLeastDouble NormMagCorrection =
        sqrt((MyAtLeastDouble)SphP[i].magcorr[0] * SphP[i].magcorr[0] + SphP[i].magcorr[1] * SphP[i].magcorr[1] +
             SphP[i].magcorr[2] * SphP[i].magcorr[2]);

    // Absolute value of Lorentz force
    MyAtLeastDouble NormMagAcceleration =
        sqrt((MyAtLeastDouble)SphP[i].magacc[0] * SphP[i].magacc[0] + SphP[i].magacc[1] * SphP[i].magacc[1] +
             SphP[i].magacc[2] * SphP[i].magacc[2]);

    // limiter for divB force correction factor
    if (NormMagCorrection > DIVBFORCECORRECTION_FACTOR * NormMagAcceleration)
    {
        for (int k = 0; k < 3; k++)
        {
#ifndef DIVBFORCECORRECTION_EXPONENT
            SphP[i].magcorr[k] *= DIVBFORCECORRECTION_FACTOR * NormMagAcceleration / NormMagCorrection;
#else // DIVBFORCECORRECTION_EXPONENT
            SphP[i].magcorr[k] *=
                pow(DIVBFORCECORRECTION_FACTOR * NormMagAcceleration / NormMagCorrection, DIVBFORCECORRECTION_EXPONENT);
#endif // DIVBFORCECORRECTION_EXPONENT
        }
    }

#ifdef NO_DIVBFORCE_SUBTRACTION
    for (int k = 0; k < 3; k++)
    {
        SphP[i].magcorr[k] = 0.0;
    }
#endif // NO_DIVBFORCE_SUBTRACTION

    for (int k = 0; k < 3; k++)
    {
        SphP[i].HydroAccel[k] += (SphP[i].magacc[k] - SphP[i].magcorr[k]);
    }
}
#endif // ACC_INCLUDE_HYDRA

#ifdef WINDS
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Computes the contribution to the magnetic field from a neighboring particle's predicted magnetic field with
 * OpenACC support.
 *
 * This function calculates the contribution to the magnetic field from a neighboring particle's predicted
 * magnetic field, weighted by the smoothing kernel function. The result is added to the magnetic field
 * of the local particle.
 *
 * @param local Pointer to the local particle's hydrodynamic data.
 * @param kernel Pointer to the kernel parameters used for smoothing.
 * @param j Index of the neighboring particle from which the magnetic field contribution is obtained.
 * @param P Pointer to the gravity walk data structure.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 * @param out Pointer to the output hydrodynamic data structure where the computed magnetic field contribution will be
 * stored.
 */
void do_magnetic_wind_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                              struct gravity_walkdata *P, struct hydrodata_in *SphP, struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Computes the contribution to the magnetic field from a neighboring particle's predicted magnetic field.
 *
 * This function calculates the contribution to the magnetic field from a neighboring particle's predicted
 * magnetic field, weighted by the smoothing kernel function. The result is added to the magnetic field
 * of the local particle.
 *
 * @param local Pointer to the local particle's hydrodynamic data.
 * @param kernel Pointer to the kernel parameters used for smoothing.
 * @param j Index of the neighboring particle from which the magnetic field contribution is obtained.
 * @param out Pointer to the output hydrodynamic data structure where the computed magnetic field contribution will be
 * stored.
 */
static inline void do_magnetic_wind(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                    struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
    out->WindB[0] += P[j].Mass * SphP[j].BPred[0] * kernel->wk_i;
    out->WindB[1] += P[j].Mass * SphP[j].BPred[1] * kernel->wk_i;
    out->WindB[2] += P[j].Mass * SphP[j].BPred[2] * kernel->wk_i;
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA
#endif // WINDS

#ifdef MAGNETIC_SN_SEEDING
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Computes the contribution to the magnetic field from supernova seeding with OpenACC support.
 *
 * This function calculates the contribution to the magnetic field from supernova seeding,
 * weighted by the smoothing kernel function. The result is added to the magnetic field
 * of the local particle.
 *
 * @param local Pointer to the local particle's hydrodynamic data.
 * @param kernel Pointer to the kernel parameters used for smoothing.
 * @param j Index of the neighboring particle from which the magnetic field contribution is obtained.
 * @param additional_variables Pointer to additional hydrodynamic data required for computation.
 * @param P Pointer to the gravity walk data structure.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 * @param out Pointer to the output hydrodynamic data structure where the computed magnetic field contribution will be
 * stored.
 */
void do_magnetic_sn_seeding_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                    struct hydrodata_additional *additional_variables, struct gravity_walkdata *P,
                                    struct hydrodata_in *SphP, struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Computes the contribution to the magnetic field from supernova seeding.
 *
 * This function calculates the contribution to the magnetic field from supernova seeding,
 * weighted by the smoothing kernel function. The result is added to the magnetic field
 * of the local particle.
 *
 * @param local Pointer to the local particle's hydrodynamic data.
 * @param kernel Pointer to the kernel parameters used for smoothing.
 * @param j Index of the neighboring particle from which the magnetic field contribution is obtained.
 * @param additional_variables Pointer to additional hydrodynamic data required for computation.
 * @param out Pointer to the output hydrodynamic data structure where the computed magnetic field contribution will be
 * stored.
 */
static inline void do_magnetic_sn_seeding(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                          struct hydrodata_additional *additional_variables, struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
    if (kernel->h_j >= kernel->r)
    {
        // scalar product between magnetic seeding and kernel
        // -> e_r in Beck+13, Eq. 10
        MyAtLeastDouble ms_r = kernel->dx * SphP[j].MagSeedRate[0] * All.cf_hubble_a +
                               kernel->dy * SphP[j].MagSeedRate[1] * All.cf_hubble_a +
                               kernel->dz * SphP[j].MagSeedRate[2] * All.cf_hubble_a;

        // magnetic softening, "f" in Beck+13
        MyAtLeastDouble soft_r = All.SnSeedSoftening * kernel->h_j;

        // normalisation of direction + wind boost factor
        MyAtLeastDouble r_norm = additional_variables->mf_windfac / (pow(kernel->r, 3) + pow(soft_r, 3));

        // Beck+13, Eq. 10
        out->DtB[0] +=
            r_norm * (3.0 * ms_r * kernel->dx / pow(kernel->r, 2) - SphP[j].MagSeedRate[0] * All.cf_hubble_a);
        out->DtB[1] +=
            r_norm * (3.0 * ms_r * kernel->dy / pow(kernel->r, 2) - SphP[j].MagSeedRate[1] * All.cf_hubble_a);
        out->DtB[2] +=
            r_norm * (3.0 * ms_r * kernel->dz / pow(kernel->r, 2) - SphP[j].MagSeedRate[2] * All.cf_hubble_a);
    }
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA
#endif // MAGNETIC_SN_SEEDING

#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Computes the magnetic induction due to hydrodynamic motion with OpenACC support.
 *
 * This function calculates the magnetic induction due to hydrodynamic motion of the local particle,
 * influenced by its neighbors, using the curl of the magnetic field and the velocity gradient.
 *
 * @param local Pointer to the local particle's hydrodynamic data.
 * @param kernel Pointer to the kernel parameters used for smoothing.
 * @param j Index of the neighboring particle.
 * @param P Pointer to the gravity walk data structure.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 * @param out Pointer to the output hydrodynamic data structure where the computed magnetic induction will be stored.
 */
void do_magnetic_induction_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                   struct gravity_walkdata *P, struct hydrodata_in *SphP, struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Computes the magnetic induction due to hydrodynamic motion.
 *
 * This function calculates the magnetic induction due to hydrodynamic motion of the local particle,
 * influenced by its neighbors, using the curl of the magnetic field and the velocity gradient.
 *
 * @param local Pointer to the local particle's hydrodynamic data.
 * @param kernel Pointer to the kernel parameters used for smoothing.
 * @param j Index of the neighboring particle.
 * @param out Pointer to the output hydrodynamic data structure where the computed magnetic induction will be stored.
 */
static inline void do_magnetic_induction(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                         struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{

    // #define FOLLOW_ID 558131

    // first order
    MyAtLeastDouble ddvx = kernel->dvx;
    MyAtLeastDouble ddvy = kernel->dvy;
    MyAtLeastDouble ddvz = kernel->dvz;

#if defined(FOLLOW_ID) && defined(PARTICLE_DEBUG)
    if (local->ID == FOLLOW_ID)
    {
        MyAtLeastDouble dtbx = kernel->mf_Ind * ((local->BPred[0] * ddvy - local->BPred[1] * ddvx) * kernel->dy +
                                                 (local->BPred[0] * ddvz - local->BPred[2] * ddvx) * kernel->dz);
        printf("HOI DEBUG: j=%d DtB[0]=%g 1st=%g", j, out->DtB[0], abs(dtbx));
#ifndef HIGHER_ORDER_INDUCTION
        printf(" 2nd=0");
#endif // HIGHER_ORDER_INDUCTION
    }
#endif // FOLLOW_ID && PARTICLE_DEBUG

#ifdef HIGHER_ORDER_INDUCTION
    // second order
    ddvx = kernel->dx * local->dVelMatrix[0][0] + kernel->dy * local->dVelMatrix[0][1] +
           kernel->dz * local->dVelMatrix[0][2];
    ddvy = kernel->dx * local->dVelMatrix[1][0] + kernel->dy * local->dVelMatrix[1][1] +
           kernel->dz * local->dVelMatrix[1][2];
    ddvz = kernel->dx * local->dVelMatrix[2][0] + kernel->dy * local->dVelMatrix[2][1] +
           kernel->dz * local->dVelMatrix[2][2];

#if defined(FOLLOW_ID) && defined(PARTICLE_DEBUG)
    if (local->ID == FOLLOW_ID)
    {
        MyAtLeastDouble dtbx = kernel->mf_Ind * ((local->BPred[0] * ddvy - local->BPred[1] * ddvx) * kernel->dy +
                                                 (local->BPred[0] * ddvz - local->BPred[2] * ddvx) * kernel->dz);
        printf(" 2nd=%g", abs(dtbx));
    }
#endif // FOLLOW_ID && PARTICLE_DEBUG
#endif // HIGHER_ORDER_INDUCTION

    out->DtB[0] += kernel->mf_Ind * ((local->BPred[0] * ddvy - local->BPred[1] * ddvx) * kernel->dy +
                                     (local->BPred[0] * ddvz - local->BPred[2] * ddvx) * kernel->dz);

#if defined(FOLLOW_ID) && defined(PARTICLE_DEBUG)
    if (local->ID == FOLLOW_ID)
    {
        printf(" (new) DtB[0]=%g\n", out->DtB[0]);
    }
#endif // FOLLOW_ID && PARTICLE_DEBUG

    out->DtB[1] += kernel->mf_Ind * ((local->BPred[1] * ddvz - local->BPred[2] * ddvy) * kernel->dz +
                                     (local->BPred[1] * ddvx - local->BPred[0] * ddvy) * kernel->dx);
    out->DtB[2] += kernel->mf_Ind * ((local->BPred[2] * ddvx - local->BPred[0] * ddvz) * kernel->dx +
                                     (local->BPred[2] * ddvy - local->BPred[1] * ddvz) * kernel->dy);
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

// Tricco (2016)
#ifdef DIVBCLEANING_TRICCO
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Calculates the gradient of the scalar field psi for the Tricco et al. (2016) cleaning with OpenACC support.
 *
 * This function computes the gradient of the scalar field psi, as used in the Tricco et al. (2016) method,
 * which is a smoothed approximation of the velocity divergence divided by density squared.
 *
 * @param local Pointer to the local particle's hydrodynamic data.
 * @param kernel Pointer to the kernel parameters used for smoothing.
 * @param j Index of the neighboring particle.
 * @param atime Scale factor used for time integration.
 * @param P Pointer to the gravity walk data structure.
 * @param SpH Pointer to the hydrodynamic data of the neighboring particles.
 * @param out Pointer to the output hydrodynamic data structure where the computed gradient will be stored.
 */
void tricco_calculate_gradpsi_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                      MyAtLeastDouble atime, struct gravity_walkdata *P, struct hydrodata_in *SphP,
                                      struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Calculates the gradient of the scalar field psi for the Tricco et al. (2016) cleaning.
 *
 * This function computes the gradient of the scalar field psi, as used in the Tricco et al. (2016) method,
 * which is a smoothed approximation of the velocity divergence divided by density squared.
 *
 * @param local Pointer to the local particle's hydrodynamic data.
 * @param kernel Pointer to the kernel parameters used for smoothing.
 * @param j Index of the neighboring particle.
 * @param atime Scale factor used for time integration.
 * @param out Pointer to the output hydrodynamic data structure where the computed gradient will be stored.
 */
static inline void tricco_calculate_gradpsi(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                            struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
    MyAtLeastDouble psi_over_rho2_i = local->CleanVel * local->PsiPred / (local->Density * local->Density);
    MyAtLeastDouble psi_over_rho2_j = SphP[j].CleanVel * SphP[j].PsiPred / (SphP[j].Density * SphP[j].Density);

    MyAtLeastDouble phifac = kernel->mj_r;

    /* eq. 25 of Tricco et al. (2016) */
    phifac *= local->Density * (psi_over_rho2_i * kernel->dwk_i + psi_over_rho2_j * kernel->dwk_j);

    out->GradPsi[0] += phifac * kernel->dx;
    out->GradPsi[1] += phifac * kernel->dy;
    out->GradPsi[2] += phifac * kernel->dz;
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Performs the Tricco et al. (2016) cleaning procedure to evolve magnetic fields.
 *
 * This function implements the Tricco et al. (2016) cleaning procedure to evolve magnetic fields
 * in SPH simulations. It calculates the time derivative of the scalar field phi and updates the magnetic field
 * accordingly. Additionally, it dissipates the parabolic term as heat.
 *
 * @param i Index of the particle for which the cleaning procedure is performed.
 * @param atime Scale factor used for time integration.
 * @param hubble_a2 Square of the Hubble parameter.
 */
static inline void perform_tricco_cleaning(int i)
{

    // skip cleaning if we are above the density threshold for star formation
#ifdef SFR
#ifdef LT_STELLAREVOLUTION
    MyFloat myPhysDensThresh = get_PhysDensThresh(i);
#else // EB_SFR_MAGNETIC
    MyFloat myPhysDensThresh = All.PhysDensThresh;
#endif // LT_STELLAREVOLUTION
    if (SphP[i].Density * All.cf_a3inv >= myPhysDensThresh)
    {
        SphP[i].DtPsi = 0.0;
        return;
    }
#endif // SFR

    MyAtLeastDouble tmpb, phiphi;
    /* Compute the parabolic term here, as this is the only one of the three below that we will convert into heat */
    MyAtLeastDouble damping_term = SphP[i].PsiPred * SphP[i].CleanVel / P[i].Hsml * All.DivBcleaningParabolicSigma;

    /* Do dPhi / dt in three part */

    /* First, compute divB part (hyperbolic term)*/
    phiphi = -SphP[i].CleanVel * SphP[i].divB;
    /* Second, compute damping term (parabolic term) */
    phiphi -= damping_term;
    /* Third, add velocity divergence term for completness (energy conservation, contribution should be minor) */
    phiphi -= 0.5 * SphP[i].PsiPred * SphP[i].DivVel;

    /* Update DtPhi. All the terms should have negative sign and we need to compensate for the 1/Ha^2 in dt_mag */
    SphP[i].DtPsi = phiphi;

    /* Dissipate the parabolic term as heat, see Tricco et al. (2016) eq. 20 */
#if GADGET_HYDRO == HYDRO_MFM
    SphP[i].DtEntropyExtra +=
#else // GADGET_HYDRO == HYDRO_MFM
    SphP[i].DtEntropy +=
#endif // GADGET_HYDRO == HYDRO_MFM
        SphP[i].PsiPred * damping_term * GAMMA_MINUS1 /
        (units->hubble_a * units->atime * units->atime * pow(SphP[i].Density * units->a3inv, GAMMA_MINUS1)) * mu0_inv /
        SphP[i].Density;

    /* dB / dt */
    SphP[i].DtB[0] -= SphP[i].GradPsi[0];
    SphP[i].DtB[1] -= SphP[i].GradPsi[1];
    SphP[i].DtB[2] -= SphP[i].GradPsi[2];
#if GADGET_HYDRO == HYDRO_MFM
    SphP[i].DtBExtra[0] += SphP[i].GradPsi[0];
    SphP[i].DtBExtra[1] += SphP[i].GradPsi[1];
    SphP[i].DtBExtra[2] += SphP[i].GradPsi[2];
#endif // GADGET_HYDRO == HYDRO_MFM
}

/**
 * @brief Updates the maximum signal velocity using the Tricco et al. (2016) cleaning.
 *
 * This function updates the maximum signal velocity of a particle based on the Tricco et al. (2016)
 * criterion. It computes the maximum of the product of the cleaning velocity and a user-defined factor
 * (`All.DivBOverCleanFac`) and the current maximum signal velocity of the particle.
 *
 * @param i Index of the particle for which the maximum signal velocity is updated.
 */
void tricco_update_signal_velocity(int i)
{
    SphP[i].MaxSignalVel = max((MyFloat)All.DivBOverCleanFac * SphP[i].CleanVel, SphP[i].MaxSignalVel);
}
#endif // DIVBCLEANING_TRICCO
#endif // ACC_INCLUDE_HYDRA

#if defined(DIVBCLEANING_DEDNER) && GADGET_HYDRO != HYDRO_MFM
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Calculates the gradient of the dedner cleaning function.
 *
 * This function calculates the gradient of the dedner cleaning function between two particles
 * using the Dedner approach. It computes the dedner cleaning function gradient based on the
 * difference in the predicted dedner cleaning function of the local and neighbor particles.
 *
 * @param local Pointer to the local particle data structure.
 * @param kernel Pointer to the kernel structure containing kernel information.
 * @param j Index of the neighbor particle.
 * @param atime The current scale factor of the simulation.
 * @param P Pointer to the gravity walk data structure.
 * @param SphP Pointer to the hydrodynamic data of the neighbor particle.
 * @param out Pointer to the output structure where the calculated gradient is stored.
 */
void dedner_calculate_gradphi_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                      MyAtLeastDouble atime, struct gravity_walkdata *P, struct hydrodata_in *SphP,
                                      struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Calculates the gradient of the dedner cleaning function.
 *
 * This function calculates the gradient of the dedner cleaning function between two particles
 * using the Dedner approach. It computes the dedner cleaning function gradient based on the
 * difference in the predicted dedner cleaning function of the local and neighbor particles.
 *
 * @param local Pointer to the local particle data structure.
 * @param kernel Pointer to the kernel structure containing kernel information.
 * @param j Index of the neighbor particle.
 * @param atime The current scale factor of the simulation.
 * @param out Pointer to the output structure where the calculated gradient is stored.
 */
static inline void dedner_calculate_gradphi(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                            MyAtLeastDouble atime, struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
    MyAtLeastDouble phifac = kernel->mf_Ind * atime * atime;
    phifac *= (local->PhiPred - SphP[j].PhiPred);
    out->GradPhi[0] += phifac * kernel->dx;
    out->GradPhi[1] += phifac * kernel->dy;
    out->GradPhi[2] += phifac * kernel->dz;
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Performs Dedner cleaning of magnetic fields.
 *
 * This function performs Dedner cleaning of magnetic fields for a given particle. It computes the
 * correction to the magnetic field and entropy based on the gradient of the cleaning function
 * and the divergence of the magnetic field.
 *
 * @param i Index of the particle for which Dedner cleaning is performed.
 * @param atime The current scale factor of the simulation.
 * @param hubble_a2 The square of the Hubble parameter.
 */
static inline void perform_dedner_cleaning(int i, MyAtLeastDouble atime, MyAtLeastDouble hubble_a2)
{
    MyAtLeastDouble phiphi;
    MyAtLeastDouble tmpb = 0.5 * SphP[i].MaxSignalVel;
    phiphi = tmpb * tmpb * All.DivBcleaningHyperbolicSigma * SphP[i].divB;
    phiphi += SphP[i].PhiPred * 0.5 * SphP[i].MaxSignalVel / P[i].Hsml * All.DivBcleaningParabolicSigma;

    SphP[i].DtPhi = -phiphi * atime * atime; /* Compensate for the 1/Ha^2 in dt_mag */

    phiphi = sqrt((MyAtLeastDouble)SphP[i].GradPhi[0] * SphP[i].GradPhi[0] + SphP[i].GradPhi[1] * SphP[i].GradPhi[1] +
                  SphP[i].GradPhi[2] * SphP[i].GradPhi[2]);

    tmpb = sqrt((MyAtLeastDouble)SphP[i].DtB[0] * SphP[i].DtB[0] + SphP[i].DtB[1] * SphP[i].DtB[1] +
                SphP[i].DtB[2] * SphP[i].DtB[2]);

    if (phiphi > All.DivBcleaningQ * tmpb && tmpb != 0)
    {
        for (int k = 0; k < 3; k++)
        {
            SphP[i].GradPhi[k] *= tmpb * All.DivBcleaningQ / phiphi;
        }
    }
#if GADGET_HYDRO == HYDRO_MFM
    SphP[i].DtEntropyExtra -=
#else // GADGET_HYDRO == HYDRO_MFM
    SphP[i].DtEntropy -=
#endif // GADGET_HYDRO == HYDRO_MFM
        (SphP[i].BPred[0] * SphP[i].GradPhi[0] + SphP[i].BPred[1] * SphP[i].GradPhi[1] +
         SphP[i].BPred[2] * SphP[i].GradPhi[2]) *
        GAMMA_MINUS1 / (hubble_a2 * pow(SphP[i].Density, GAMMA_MINUS1)) * mu0_inv / SphP[i].Density;

    SphP[i].DtB[0] += SphP[i].GradPhi[0];
    SphP[i].DtB[1] += SphP[i].GradPhi[1];
    SphP[i].DtB[2] += SphP[i].GradPhi[2];
#if GADGET_HYDRO == HYDRO_MFM
    SphP[i].DtBExtra[0] += SphP[i].GradPhi[0];
    SphP[i].DtBExtra[1] += SphP[i].GradPhi[1];
    SphP[i].DtBExtra[2] += SphP[i].GradPhi[2];
#endif // GADGET_HYDRO == HYDRO_MFM
}
#endif // ACC_INCLUDE_HYDRA
#endif // defined(DIVBCLEANING_DEDNER) && GADGET_HYDRO != HYDRO_MFM

#if defined(TIME_DEP_MAGN_DISP) && ((TIME_DEP_MAGN_DISP == 0) || (TIME_DEP_MAGN_DISP == 1))
#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Calculates the artificial magnetic dissipation term.
 *
 * This function calculates the artificial magnetic dissipation term for a given particle based on
 * its properties and simulation parameters. The dissipation term is computed either based on the
 * divergence of the magnetic field alone or considering the rotation of the magnetic field if
 * enabled.
 *
 * @param i Index of the particle for which the magnetic dissipation term is calculated.
 * @param fac_mu The magnetic dissipation factor.
 */
static inline void calculate_magn_disp(int i, MyAtLeastDouble fac_mu)
{
    SphP[i].DtBalpha = (All.ArtificialMagneticDissipationMin - SphP[i].Balpha) *
                       All.ArtificialMagneticDissipationDecaytime * 0.5 * SphP[i].MaxSignalVel / (P[i].Hsml * fac_mu);
#if (TIME_DEP_MAGN_DISP == 0)
    SphP[i].DtBalpha += All.ArtificialMagneticDissipationSource * abs(SphP[i].divB) / sqrt(mu0 * SphP[i].Density);
#endif // (TIME_DEP_MAGN_DISP==0)
#if (TIME_DEP_MAGN_DISP == 1)
    SphP[i].DtBalpha +=
        All.ArtificialMagneticDissipationSource / sqrt(mu0 * SphP[i].Density) *
        max(abs(SphP[i].divB), abs(sqrt((MyAtLeastDouble)SphP[i].RotB[0] * SphP[i].RotB[0] +
                                        SphP[i].RotB[1] * SphP[i].RotB[1] + SphP[i].RotB[2] * SphP[i].RotB[2])));
#endif // (TIME_DEP_MAGN_DISP==1)
}
#endif // ACC_INCLUDE_HYDRA
#endif // defined(TIME_DEP_MAGN_DISP) && ((TIME_DEP_MAGN_DISP==0) || (TIME_DEP_MAGN_DISP==1))

#ifdef MAGNETIC_DIFFUSION
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Calculates the contribution of magnetic diffusion to the evolution of magnetic fields with OpenACC support.
 *
 * This function calculates the contribution of magnetic diffusion to the evolution of magnetic fields
 * for a given particle. It updates the time derivative of the magnetic field (`DtB`) based on the
 * magnetic resistivity (`MagneticEta`) and the spatial derivatives of the magnetic field components.
 * Additionally, if magnetic diffusion heat is enabled, it adjusts the time derivative of entropy (`DtEntropy`)
 * accordingly.
 *
 * @param local Pointer to the local properties of the particle.
 * @param kernel Pointer to the kernel properties used for interpolation.
 * @param j Index of the neighboring particle.
 * @param P Pointer to the particle data structure.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 * @param out Pointer to the output data structure where the results are stored.
 */
void calculate_magnetic_diffusion_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                          struct gravity_walkdata *P, struct hydrodata_in *SphP,
                                          struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Calculates the contribution of magnetic diffusion to the evolution of magnetic fields.
 *
 * This function calculates the contribution of magnetic diffusion to the evolution of magnetic fields
 * for a given particle. It updates the time derivative of the magnetic field (`DtB`) based on the
 * magnetic resistivity (`MagneticEta`) and the spatial derivatives of the magnetic field components.
 * Additionally, if magnetic diffusion heat is enabled, it adjusts the time derivative of entropy (`DtEntropy`)
 * accordingly.
 *
 * @param local Pointer to the local properties of the particle.
 * @param kernel Pointer to the kernel properties used for interpolation.
 * @param j Index of the neighboring particle.
 * @param out Pointer to the output data structure where the results are stored.
 */
static inline void calculate_magnetic_diffusion(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                                struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
    out->DtB[0] += 2.0 * All.MagneticEta * kernel->mf_dissInd * kernel->dBx;
    out->DtB[1] += 2.0 * All.MagneticEta * kernel->mf_dissInd * kernel->dBy;
    out->DtB[2] += 2.0 * All.MagneticEta * kernel->mf_dissInd * kernel->dBz;
#ifdef MAGNETIC_DIFFUSION_HEAT
    out->DtEntropy -= 2.0 * All.MagneticEta * kernel->mf_dissEnt *
                      (kernel->dBx * kernel->dBx + kernel->dBy * kernel->dBy + kernel->dBz * kernel->dBz);
#endif // MAGNETIC_DIFFUSION_HEAT
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA
#endif // MAGNETIC_DIFFUSION

#ifdef ARTIFICIAL_MAGNETIC_DISSIPATION
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Calculates the contribution of magnetic dissipation to the evolution of magnetic fields.
 *
 * This function calculates the contribution of magnetic dissipation to the evolution of magnetic fields
 * for a given particle. It updates the time derivative of the magnetic field (`DtB`) and the time derivative
 * of entropy (`DtEntropy`) based on the local properties, kernel properties, and additional variables.
 *
 * @param local Pointer to the local properties of the particle.
 * @param kernel Pointer to the kernel properties used for interpolation.
 * @param j Index of the neighboring particle.
 * @param additional_variables Pointer to additional variables needed for the calculation.
 * @param P Pointer to the particle data structure.
 * @param SpP Pointer to the hydrodynamic data of the neighboring particle.
 * @param out Pointer to the output data structure where the results are stored.
 */
void calculate_magnetic_dissipation_openacc(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                            struct hydrodata_additional *additional_variables,
                                            struct gravity_walkdata *P, struct hydrodata_in *SphP,
                                            struct hydrodata_out *out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Calculates the contribution of magnetic dissipation to the evolution of magnetic fields.
 *
 * This function calculates the contribution of magnetic dissipation to the evolution of magnetic fields
 * for a given particle. It updates the time derivative of the magnetic field (`DtB`) and the time derivative
 * of entropy (`DtEntropy`) based on the local properties, kernel properties, and additional variables.
 *
 * @param local Pointer to the local properties of the particle.
 * @param kernel Pointer to the kernel properties used for interpolation.
 * @param j Index of the neighboring particle.
 * @param additional_variables Pointer to additional variables needed for the calculation.
 * @param out Pointer to the output data structure where the results are stored.
 */
static inline void calculate_magnetic_dissipation(struct hydrodata_in *local, struct kernel_hydra *kernel, int j,
                                                  struct hydrodata_additional *additional_variables,
                                                  struct hydrodata_out *out)
#endif // ACC_INCLUDE_HYDRA
{
    MyAtLeastDouble vsigb = 0.5 * (additional_variables->magneticspeed_i + additional_variables->magneticspeed_j);
#ifdef TIME_DEP_MAGN_DISP
    MyAtLeastDouble eta = 0.5 * (local->Balpha + SphP[j].Balpha) * vsigb * kernel->r;
#else // TIME_DEP_MAGN_DISP
    MyAtLeastDouble eta = All.ArtificialMagneticDissipationConstant * vsigb * kernel->r;
#endif // TIME_DEP_MAGN_DISP

    // Boundary "sponge" resistivity for vacuum boundaries in zoom-in simulations
#ifdef LMB_MHD_ILLCOND_RESISTIVITY
    if (local->ConditionNumber > (MyFloat)1.0e6 || SphP[j].ConditionNumber > (MyFloat)1.0e6)
    {
        const MyAtLeastDouble eta_boundary = vsigb * kernel->r;
        eta = DMAX(eta, eta_boundary);
    }
#endif // LMB_MHD_ILLCOND_RESISTIVITY

    out->DtEntropy -=
        eta * kernel->mf_dissEnt * (kernel->dBx * kernel->dBx + kernel->dBy * kernel->dBy + kernel->dBz * kernel->dBz);
    out->DtB[0] += eta * kernel->mf_dissInd * kernel->dBx;
    out->DtB[1] += eta * kernel->mf_dissInd * kernel->dBy;
    out->DtB[2] += eta * kernel->mf_dissInd * kernel->dBz;
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA
#endif // ARTIFICIAL_MAGNETIC_DISSIPATION

#endif // MAGNETIC
#endif // MHD_HPP
