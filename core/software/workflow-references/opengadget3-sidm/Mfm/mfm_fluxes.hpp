/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file contaning functions necessary for MFM flux calculation.
 *
 * This file contains the functions necessary to performs the MFM flux calculation.
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
#if !(defined ACC_INCLUDE_HYDRA) && defined(ACC_HYDRA) // we reinclude this same file if ACC is active
#include "../OpenACC/gpuallvars_branch.hpp"
#endif // !(defined ACC_INCLUDE_HYDRA) && defined(ACC_HYDRA)

#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Check if a particle is active for MFM flux calculation.
 *
 * A particle is considered active for MFM fluxes if it is type 0 and, optionally,
 * if it is not a wind particle, has a positive mass (with BLACK_HOLES), and
 * a non-zero ID (FG_REMOVE_BND).
 *
 * @param n The index of the particle to check.
 * @return True if the particle is active for MFM flux calculation, false otherwise.
 */
static inline int flux_isactive(int n)
{
    return (P[n].Type == type_gas
#ifdef WINDS
            && SphP[n].DelayTime <= 0
#endif // WINDS
#ifdef BLACK_HOLES
            && P[n].Mass > 0
#endif // BLACK_HOLES
#ifdef FG_REMOVE_BND
            && P[n].ID != 0
#endif // FG_REMOVE_BND
    );
}
#endif // ACC_INCLUDE_HYDRA




#ifndef ACC_INCLUDE_HYDRA
/**
 * @brief Convert the total energy change to the internal energy change.
 *
 * This function substracts the kinetic energy change from the total energy change, leaving only the internal energy
 * change. If magnetic fields are active, also the magnetic energy change is substracted. The substraction includes
 * second-order terms of the time derivative.
 *
 * @param i Index of the particle in the SPH particle array.
 * @param dt_entr_phys Physical timestep.
 */
static inline void convert_total_to_internal_energy_flux(int i, double dt_entr_phys, Units *units)
{
    // Subtract the kinetic-energy change so dQ.GetEnergy() becomes an INTERNAL-energy flux. Under
    // MFM_CONSERVATIVE_TOTAL_ENERGY this loop is skipped: dQ.GetEnergy() keeps the kinetic part and u
    // is recomputed from the actual post-kick velocity in predict.cpp/kicks.cpp (total-energy conserving).
    for (int k = 0; k < mfm_numdims; k++)
    {
#ifndef GT_DUST
        SphP[i].dQdt.GetEnergy() -= (units->velocity_to_physical * SphP[i].Wprim.GetVelocity()[k] +
                                     (MyFloat)0.5 * SphP[i].dQdt.GetMomentum()[k] * dt_entr_phys / P[i].Mass) *
                                    SphP[i].dQdt.GetMomentum()[k]; // from total energy flux to internal energy flux
                                                                   // (physical units) // without the
        // dQ term, it produces too high internal energies for the shock.
#else // GT_DUST
        SphP[i].dQdt.GetEnergy() -=
            (units->velocity_to_physical * SphP[i].Wprim.GetVelocity()[k] +
             (MyFloat)0.5 * SphP[i].dQdt.GetMomentum()[k] * dt_entr_phys / P[i].Mass /
                 (1. - SphP[i].TotalDustFracPred) -
             (MyFloat)0.5 * units->velocity_to_physical * SphP[i].Wprim.GetVelocity()[k] * SphP[i].dQdt.GetMass() *
                 dt_entr_phys / P[i].Mass / (1. - SphP[i].TotalDustFracPred)) *
            (SphP[i].dQdt.GetMomentum()[k] -
             units->velocity_to_physical * SphP[i].Wprim.GetVelocity()[k] * SphP[i].dQdt.GetMass());
#endif // GT_DUST
    }

#ifdef GT_DUST
    // Additional terms due to non-zero gas mass flux
    SphP[i].dQdt.GetEnergy() -= SphP[i].Wprim.GetSpecificTotalEnergy(GAMMA) * SphP[i].dQdt.GetMass();
#endif // GT_DUST
}
#endif // ACC_INCLUDE_HYDRA

#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Evaluates the face between neighboring particles.
 *
 * This function computes the face vector and area based on the distance between two particles. The face position is
 * computed second-order based on the smoothing length of the two particles.
 *
 * @param local local hydrodynamic data structure (passed as reference).
 * @param kernel structure containing smoothing lengths and derivatives (passed as reference).
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation (passed as reference).
 * @param P Pointer to the particle data.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 */
void calculate_face_openacc(struct hydrodata_in &local, struct kernel_hydra &kernel, int j,
                            struct hydrodata_additional &additional_variables, struct gravity_walkdata *P,
                            struct hydrodata_in *SphP)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Evaluates the face between neighboring particles.
 *
 * This function computes the face vector and area based on the distance between two particles. The face position is
 * computed second-order based on the smoothing length of the two particles.
 *
 * @param local local hydrodynamic data structure (passed as reference).
 * @param kernel structure containing smoothing lengths and derivatives (passed as reference).
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation (passed as reference).
 */
static inline void calculate_face(struct hydrodata_in &local, struct kernel_hydra &kernel, int j,
                                  struct hydrodata_additional &additional_variables)
#endif // ACC_INCLUDE_HYDRA
{
    // Compute psi-tilde values (for integral gradient terms)
    additional_variables.vol_j = MyFloat(1.0) / SphP[j].NumDens;
    Vector<NUMDIMS, MyAtLeastDouble> psitildai =
        SphP[j].Bgrad * kernel.dr * (-kernel.wk_j * additional_variables.vol_j);
    Vector<NUMDIMS, MyAtLeastDouble> psitildaj = local.Bgrad * kernel.dr * (kernel.wk_i * local.Volume);

    // Compute the face pseudo-area, Aij, and corresponding unit vector, Aunit

    Vector<NUMDIMS, MyAtLeastDouble> Aij = psitildaj * local.Volume - psitildai * additional_variables.vol_j;
    additional_variables.Amag = Aij.GetMagnitude();
    additional_variables.Aunit = Aij.GetUnitVector();

    // Compute position and velocity of the working face (i.e. where the Riemann problem is solved)
    // s_star_ij = 0; /*first order*/
    MyAtLeastDouble s_star_ij = 0.5 * kernel.r * (P[j].Hsml - local.Hsml) / (local.Hsml + P[j].Hsml); /*second order */
    MyAtLeastDouble s_i = 0.5 * kernel.r - s_star_ij;
    MyAtLeastDouble s_j = 0.5 * kernel.r + s_star_ij;
    const MyAtLeastDouble r_inv = 1.0 / kernel.r;
    for (int k = 0; k < NUMDIMS; k++)
    {
        // rface[k] = MyFloat(0.5)*(2.0*local.Pos[k] + kernel.dr[k]);
        // additional_variables.drFacej[k] = rface[k] - local.Pos[k] - kernel.dr[k];
        // additional_variables.drFacei[k] = rface[k] - local.Pos[k];
        additional_variables.drFacej[k] = -kernel.dr[k] * r_inv * s_j;
        additional_variables.drFacei[k] = kernel.dr[k] * r_inv * s_i;
        MyAtLeastDouble v_i, v_j;

        v_i = local.Wprim.GetVelocity()[k];
        v_j = SphP[j].Wprim.GetVelocity()[k];

#ifdef GT_DUST
        v_i += local.TotalDustFracPred * local.TotalDeltaVelPred[k];
        v_j += SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[k];
#endif // GT_DUST

        // additional_variables.vface[k] = MyFloat(0.5)*(v_i + v_j);
        additional_variables.vface[k] =
            (s_j * v_i + s_i * v_j) * r_inv; // allows for face to be off-center (second-order)
    }

    if ((local.ConditionNumber > 1e12) || (SphP[j].ConditionNumber > 1e12 /*gizmo value */) ||
        (Vector<NUMDIMS>::DotProduct(additional_variables.Aunit, additional_variables.drFacei) < 0.0))
    {
        MyAtLeastDouble wt_i, wt_j;
        additional_variables.Aunit = additional_variables.drFacei.GetUnitVector();
#ifdef COOLING
        if (abs(local.Volume - additional_variables.vol_j) /
                std::min((MyAtLeastDouble)local.Volume, additional_variables.vol_j) >
            1.25)
        {
            wt_i = wt_j =
                (local.Volume * P[j].Hsml + additional_variables.vol_j * local.Hsml) / (local.Hsml + P[j].Hsml);
        }
#else // COOLING
        if (abs(local.Volume - additional_variables.vol_j) /
                std::min((MyAtLeastDouble)local.Volume, additional_variables.vol_j) >
            1.5)
        {
            wt_i = wt_j =
                (MyFloat)2.0 * local.Volume * additional_variables.vol_j / (local.Volume + additional_variables.vol_j);
        }
#endif // COOLING
        else
        {
            wt_i = local.Volume;
            wt_j = additional_variables.vol_j;
        }
        additional_variables.Amag =
            -(wt_i * local.Volume * kernel.dwk_i + wt_j * additional_variables.vol_j * kernel.dwk_j) * r_inv;
    }
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Set additional_variables for MFM.
 *
 * This function prepares for the flux calculation by setting local variables for the interaction.
 *
 * @param local local hydrodynamic data structure (passed as reference).
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation (passed as reference).
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 */
void set_additional_variables_ij_mfm_openacc(struct hydrodata_in &local, int j,
                                             struct hydrodata_additional &additional_variables,
                                             struct hydrodata_in *SphP)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Set additional_variables for MFM.
 *
 * This function prepares for the flux calculation by setting local variables for the interaction.
 *
 * @param local local hydrodynamic data structure (passed as reference).
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation (passed as reference).
 */
static inline void set_additional_variables_ij_mfm(struct hydrodata_in &local, int j,
                                                   struct hydrodata_additional &additional_variables)
#endif // ACC_INCLUDE_HYDRA
{
    additional_variables.Wi = local.Wprim;
    additional_variables.Wj = SphP[j].Wprim;
#ifdef DISK_SLICE
    wrap_velocity_disk_slice(P[j].Pos, local.Pos, additional_variables.Wj.GetVelocity()[2]);
#endif // DISK_SLICE

#ifdef GT_DUST
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        additional_variables.Wi_dust[idust] = local.Wprim_dust[idust];
        additional_variables.Wj_dust[idust] = SphP[j].Wprim_dust[idust];
    }
#endif // GT_DUST
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifndef NOMUSCL
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Perform a half-step interpolation according to the MUSCL scheme for MFM.
 *
 * This function interpolates the hydrodynamical properties by half a timestep using the estimated time derivative.
 * Based on the updated quantities, it updates the face as well.
 *
 * @param local local hydrodynamic data structure (passed as reference).
 * @param kernel structure containing smoothing lengths and derivatives (passed as reference).
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation (passed as reference).
 * @param P Pointer to the particle data.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 */
void do_muscl_halfstep_interpolation_openacc(struct hydrodata_in &local, struct kernel_hydra &kernel, int j,
                                             struct hydrodata_additional &additional_variables,
                                             struct gravity_walkdata *P, struct hydrodata_in *SphP)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Perform a half-step interpolation according to the MUSCL scheme for MFM.
 *
 * This function interpolates the hydrodynamical properties by half a timestep using the estimated time derivative.
 * Based on the updated quantities, it updates the face as well.
 *
 * @param local local hydrodynamic data structure (passed as reference).
 * @param kernel structure containing smoothing lengths and derivatives (passed as reference).
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation (passed as reference).
 */
static inline void do_muscl_halfstep_interpolation(struct hydrodata_in &local, struct kernel_hydra &kernel, int j,
                                                   struct hydrodata_additional &additional_variables)
#endif // ACC_INCLUDE_HYDRA
{
    // compare to the Muscl implementation in Gandalf (Hubber+2018, section 2.2.5).
    // Calculate the predicted left-hand state at the half-timestep
    WFluidVector<mfm_numdims> Wdot;
    Vector<mfm_numdims> gradWi_full[numvar];
    // fill the missing dimensions
    for (int k = 0; k < numvar; k++)
    {
        for (int kk = 0; kk < NUMDIMS; kk++)
        {
            gradWi_full[k][kk] = additional_variables.gradWi[k][kk];
        }
        for (int kk = NUMDIMS; kk < mfm_numdims; kk++)
        {
            gradWi_full[k][kk] = 0.0;
        }
    }
    // changes are calculated from the gradients.
    Wdot = WFluidVector<mfm_numdims>::ComputePrimitiveTimeDerivative(
        local.SoundSpeed * local.SoundSpeed, additional_variables.Wi, gradWi_full, additional_variables.vface);
    // update the left-hand state
    additional_variables.Wi += MyFloat(0.5) * dt_entr * Wdot;

    // Calculate the predicted right-hand state at the half-timestep
    const MyAtLeastDouble SoundSpeed_j = eos->SoundSpeed(SphP[j].InternalEnergyPred);
    // fill the missing dimensions
    Vector<mfm_numdims> gradWj_full[numvar];
    for (int k = 0; k < numvar; k++)
    {
        for (int kk = 0; kk < NUMDIMS; kk++)
        {
            gradWj_full[k][kk] = additional_variables.gradWj[k][kk];
        }
        for (int kk = NUMDIMS; kk < mfm_numdims; kk++)
        {
            gradWj_full[k][kk] = 0.0;
        }
    }
    // changes are calculated from the gradients.
    Wdot = WFluidVector<mfm_numdims>::ComputePrimitiveTimeDerivative(
        SoundSpeed_j * SoundSpeed_j, additional_variables.Wj, gradWj_full, additional_variables.vface);
    // now, also the right-hand state can be updated.
    additional_variables.Wj += MyFloat(0.5) * dt_entr * Wdot;

    // add the gravitational contribution
    for (int k = 0; k < mfm_numdims; k++)
    {
        additional_variables.Wi.GetVelocity()[k] += MyFloat(0.5) * dt_gravkick * local.GravAccel[k];
        additional_variables.Wj.GetVelocity()[k] += MyFloat(0.5) * dt_gravkick * P[j].GravAccel[k];
#ifdef PMGRID
        // local.GravAccel already contains the GravPM contribution.
        additional_variables.Wj.GetVelocity()[k] += MyFloat(0.5) * dt_gravkick * P[j].GravPM[k];
#endif // PMGRID
    }
    // now that the two states are updated, we have to update the face velocity:
    s_star_ij = 0.5 * kernel.r * (P[j].Hsml - local.Hsml) / (local.Hsml + P[j].Hsml); // second order
    s_i = 0.5 * kernel.r - s_star_ij;
    s_j = 0.5 * kernel.r + s_star_ij;
    const MyAtLeastDouble r_inv = 1.0 / kernel.r;
    for (int k = 0; k < NUMDIMS; k++)
    {
        s_i -= 0.5 * dt_entr * additional_variables.Wi.GetVelocity()[k] * kernel.dr[k] * r_inv;
        s_j += 0.5 * dt_entr * additional_variables.Wj.GetVelocity()[k] * kernel.dr[k] * r_inv;
    }

    for (int k = 0; k < mfm_numdims; k++)
    {
        MyAtLeastDouble v_i, v_j;
        v_i = additional_variables.Wi.GetVelocity()[k];
        v_j = additional_variables.Wj.GetVelocity()[k];
#ifdef GT_DUST
        v_i += local.TotalDustFracPred * local.TotalDeltaVelPred[k];
        v_j += SphP[j].TotalDustFracPred * SphP[j].TotalDeltaVelPred[k];
#endif // GT_DUST
       // additional_variables.vface[k] = MyFloat(0.5)*(v_i + v_j); // first order
        additional_variables.vface[k] =
            (s_j * v_i + s_i * v_j) * r_inv; // allows for face to be off-center (second-order)
    }
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA
#endif // NOMUSCL

#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Convert all units from comoving to physical.
 *
 * Convert hydrodynamical quantities of both interacting particles as well as the face area from comoving to physical
 * units, which are required by the Riemann solver.
 *
 * @param additional_variables Additional variables needed for the computation (passed as reference).
 */
void units_to_physical_openacc(struct hydrodata_additional &additional_variables, Units *units)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Convert all units from comoving to physical.
 *
 * Convert hydrodynamical quantities of both interacting particles as well as the face area from comoving to physical
 * units, which are required by the Riemann solver.
 *
 * @param additional_variables Additional variables needed for the computation (passed as reference).
 */
static inline void units_to_physical(struct hydrodata_additional &additional_variables, Units *units)
#endif // ACC_INCLUDE_HYDRA
{
    if (All.ComovingIntegrationOn)
    {
        // Convert all quantities that are used within the Riemann solver to physical units
        // velocity
        for (int k = 0; k < mfm_numdims; k++)
        {
            additional_variables.vface[k] = units->velocity_to_physical * additional_variables.vface[k];
            additional_variables.Wi.GetVelocity()[k] =
                units->velocity_to_physical * additional_variables.Wi.GetVelocity()[k];
            additional_variables.Wj.GetVelocity()[k] =
                units->velocity_to_physical * additional_variables.Wj.GetVelocity()[k];
        }
        // density
        additional_variables.Wi.GetDensity() = units->density_to_physical * additional_variables.Wi.GetDensity();
        additional_variables.Wj.GetDensity() = units->density_to_physical * additional_variables.Wj.GetDensity();
        // pressure
        additional_variables.Wi.GetPressure() = units->pressure_to_physical * additional_variables.Wi.GetPressure();
        additional_variables.Wj.GetPressure() = units->pressure_to_physical * additional_variables.Wj.GetPressure();

        // area, so length*length
        additional_variables.Amag = units->length_to_physical * units->length_to_physical * additional_variables.Amag;
    }
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Evaluate godunov fluxes for MFM.
 *
 * This function uses a given Riemann solver to calculate the Godunov fluxes for MFM. Fluxes are added to the dQ(dt)
 * variable both for the local particle and the neighbor. In the MFM_MAGNETIC case, also the necessary advective term is
 * applied.
 *
 * @param local local hydrodynamic data structure (passed as reference).
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation (passed as reference).
 * @param dt_entr_phys Physical timestep.
 * @param SphP Pointer to the hydrodynamic data of the neighboring particle.
 * @param out output hydrodynamic data structure containing the computed fluxes (passed as reference).
 */
void evaluate_godunov_fluxes_openacc(struct hydrodata_in &local, int j,
                                     struct hydrodata_additional &additional_variables, double dt_entr_phys,
                                     struct hydrodata_in *SphP, struct hydrodata_out &out)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Evaluate godunov fluxes for MFM.
 *
 * This function uses a given Riemann solver to calculate the Godunov fluxes for MFM. Fluxes are added to the dQ(dt)
 * variable both for the local particle and the neighbor. In the MFM_MAGNETIC case, also the necessary advective term is
 * applied.
 *
 * @param local local hydrodynamic data structure (passed as reference).
 * @param j Index of the neighboring particle.
 * @param additional_variables Additional variables needed for the computation (passed as reference).
 * @param dt_entr_phys Physical timestep.
 * @param out output hydrodynamic data structure containing the computed fluxes (passed as reference).
 */
static inline void evaluate_godunov_fluxes(struct hydrodata_in &local, int j,
                                           struct hydrodata_additional &additional_variables, double dt_entr_phys,
                                           struct hydrodata_out &out)
#endif // ACC_INCLUDE_HYDRA
{
    // make sure the face vector contains all relevant dimensions.
    // in particular for 1/2d MHD, the original vector is only 1/2d, but the results of the Riemann solver are 3d.
    Vector<mfm_numdims, MyAtLeastDouble> Aunit_full;
    for (int k = 0; k < NUMDIMS; k++)
    {
        Aunit_full[k] = additional_variables.Aunit[k];
    }
    for (int k = NUMDIMS; k < mfm_numdims; k++)
    {
        Aunit_full[k] = 0.0;
    }
    // use the Riemann solver to compute fluxes.
#ifndef GT_DUST
    FluxVector<mfm_numdims, MyAtLeastDouble> flux = fluxSolver->ComputeFluxes(
        true, Aunit_full, additional_variables.vface, additional_variables.Wi, additional_variables.Wj);
#else // GT_DUST
    FluxVector<mfm_numdims, MyAtLeastDouble> flux;
    std::array<FluxVector<mfm_numdims>, GT_NUM_DUST> flux_dust;
    std::array<QFluidVector<mfm_numdims>, GT_NUM_DUST> Qflux_dust;

    fluxSolver->ComputeFluxesMultifluid(false, Aunit_full, additional_variables.vface, additional_variables.Wi,
                                        additional_variables.Wj, additional_variables.Wi_dust,
                                        additional_variables.Wj_dust, flux, flux_dust);

    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        Qflux_dust[idust] = flux_dust[idust] * additional_variables.Amag;
        out.dQdt_dust[idust] -= Qflux_dust[idust];
    }
#endif // GT_DUST
    // now we can assign the fluxes to the fluid vectors
    QFluidVector<mfm_numdims, MyAtLeastDouble> Qflux = flux * additional_variables.Amag;

    out.dQdt -= Qflux;

#ifdef MFM_FACE_CLOSURE_RESIDUAL
    // vector sum of the effective face areas = the cell-closure residual (ideally 0). i gets the
    // outward normal (i->j); used by the face-closure DtB and internal-energy limiters to detect
    // ill-conditioned cells.
    for (int k = 0; k < 3; k++)
    {
        out.FaceAreaSum[k] += additional_variables.Amag * Aunit_full[k];
    }
#endif // MFM_FACE_CLOSURE_RESIDUAL


#ifndef DONT_BIN_MFM_FLUXES
    // If neighbour is in longer bin, then record opposite flux here
    // only in combination with timebin criterion in the beginning!

    // If neighbour is also at the beginning of its step, then record dQdt
    if (flux_isactive(j))
    {
#ifdef _OPENMP
        omp_set_lock(&MfmNgbLockList[j]);
#endif // _OPENMP
        SphP[j].dQdt += Qflux;
#ifdef MFM_FACE_CLOSURE_RESIDUAL
        // closure residual: j gets the opposite outward normal (j->i), so -A_vec_ij.
        for (int k = 0; k < 3; k++)
        {
            SphP[j].FaceAreaSum[k] -= additional_variables.Amag * Aunit_full[k];
        }
#endif // MFM_FACE_CLOSURE_RESIDUAL
#ifdef GT_DUST
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            SphP[j].dQdt_dust[idust] += Qflux_dust[idust];
        }
#endif // GT_DUST
#ifdef _OPENMP
        omp_unset_lock(&MfmNgbLockList[j]);
#endif // _OPENMP
    }
#endif // DONT_BIN_MFM_FLUXES
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA
